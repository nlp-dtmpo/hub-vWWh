import os
import random
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch

# ---------------- Config ----------------
PRETRAINED_MODEL_LOCAL = "models/bert-base-chinese"  # 你本地模型路径（若存在）
OUTPUT_DIR = "./tnews_finetuned"
NUM_LABELS = None  # leave None and infer from dataset
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 64
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True if (torch.cuda.is_available()) else False

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- Load dataset ----------------
print("Loading TNEWS dataset from Hugging Face datasets (CLUE.tnews)...")
raw = load_dataset("clue", "tnews")# train/validation/test 都会自动下载

#print(raw)
#以下是输出的raw
"""
DatasetDict({
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 10000
    })
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 53360
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 10000
    })
})
"""

# 1. 【关键步骤】删除官方的 test 集，因为它的标签是无效的，会导致 CUDA 报错
# CLUE 的官方 test 集是拿来比赛打榜的，里面的标签（label）全是 -1（无效值）。 
# 当你的代码运行到 trainer.evaluate(tokenized["test"]) 时，模型试图计算预测值和标签 -1 之间的误差（Loss）。
# 由于 -1 不在合法的 0-14 范围内，GPU 直接抛出了 Assertion failed 异常，导致程序崩溃。
if "test" in raw:
    print("Detected official test set with invalid labels. Dropping it...")
    del raw["test"]

# 2. 从官方 validation 集中切分出一半作为我们的 test 集
#    官方 validation 有 10000 条，切分后：5000 条验证，5000 条测试
print("Splitting official validation set to create a valid test set...")
split_dataset = raw["validation"].train_test_split(test_size=0.5, seed=SEED)
raw["validation"] = split_dataset["train"]
raw["test"] = split_dataset["test"]

print("New dataset structure:")
print(raw)

# Map labels: 重新计算标签数量（此时 train 中的标签是完整的 0-14）
label_list = sorted(list(set(raw["train"]["label"])))
NUM_LABELS = len(label_list)
print(f"Number of labels: {NUM_LABELS}")

# ---------------- Tokenizer & model ----------------
model_name_or_path = PRETRAINED_MODEL_LOCAL
print("Using model:", model_name_or_path)

tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path, use_fast=True)

def preprocess_batch(batch):
    # text field in tnews is 'sentence' (HuggingFace CLUE tnews), label field is 'label'
    texts = batch["sentence"] if "sentence" in batch else batch["text"]
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = batch["label"]
    return enc

print("Tokenizing dataset (this may take a minute)...")
tokenized = raw.map(preprocess_batch, batched=True, remove_columns=raw["train"].column_names)

# set format for PyTorch
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------- Metrics ----------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}

# ---------------- Model ----------------
model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=NUM_LABELS)
model.to(DEVICE)

# ---------------- TrainingArguments & Trainer ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    #如果已经有较好的训练数据 , 那么将下方的arg设置为False
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=FP16,
    report_to="none"  # disable wandb or others; set as needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# ---------------- Train ----------------
trainer.train()

# ---------------- Evaluate on test ----------------
print("Evaluating on test set...")
metrics = trainer.evaluate(eval_dataset=tokenized["test"])
print("Test metrics:", metrics)

# ---------------- Save best model and tokenizer ----------------
print("Saving best model to", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---------------- Single-sample inference function ----------------
from transformers import pipeline
nlp = pipeline("text-classification", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR, device=0 if DEVICE=="cuda" else -1)

def predict_single(text):
    pred = nlp(text, truncation=True, max_length=MAX_LENGTH)
    # CLUE tnews maps labels to ints; pipeline returns label like 'LABEL_0' -> we map index
    label_token = pred[0]["label"]  # e.g., "LABEL_3"
    idx = int(label_token.split("_")[-1])
    score = pred[0]["score"]
    return idx, score

# Demo
samples = [
    "昨天国家队在比赛中表现出色夺得冠军",
    "如何优化深度学习模型的训练速度？"
]
for s in samples:
    idx, score = predict_single(s)
    print(f"Text: {s}\nPred label id: {idx}, confidence: {score}\n")

