import os
# 配置 Hugging Face 镜像站（核心配置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from datasets import load_dataset, get_dataset_infos
from transformers import BertTokenizer, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from collections import Counter
import numpy as np

MODEL_NAME = "bert-base-chinese"
# 数据集名称：hugging face官方TNEWS数据集（中文新闻15分类）
DATASET_NAME = "uer/tnews"
# 分类类别数：TNEWS固定15个新闻类别（科技、娱乐、体育等）
NUM_LABELS = 15
# 文本最大长度：BERT输入序列最大512，根据TNEWS文本长度设为128（兼顾效率与效果）
MAX_SEQ_LENGTH = 64
# 定义小样本数量（训练5000、验证500、测试500）
TRAIN_SIZE = 500
VAL_SIZE = 50
TEST_SIZE = 50
RANDOM_SEED = 42  # 固定随机种子，保证抽样结果可复现

print("开始下载数据")
dataset = load_dataset("clue", "tnews")
print("开始加载tokenizer")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, model_max_length=MAX_SEQ_LENGTH)
print("开始加载model")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=15,
    ignore_mismatched_sizes=True
)

train_dataset = dataset["train"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_SIZE))
validation_dataset = dataset["validation"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_SIZE))
test_dataset = dataset["test"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_SIZE))
print(f"训练集：{len(train_dataset)}条 | 验证集：{len(dataset['validation'])}条 | 测试集：{len(test_dataset)}条")

train_labels = train_dataset['label']
label_counts = Counter(train_labels)
print(f"训练集类别分布: {label_counts}")


def preprocess_function(examples):
    """数据预处理函数"""
    texts = examples['sentence']

    # 对文本进行编码
    encoding = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )

    # 如果有标签，则添加标签
    if 'label' in examples:
        encoding['labels'] = torch.tensor(examples['label'])

    return encoding


train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=train_dataset.column_names
)

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=test_dataset.column_names
)
for dataset in [train_dataset, test_dataset]:
    dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}


# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 训练输出目录，用于保存模型和状态
    num_train_epochs=2,  # 训练的总轮数
    per_device_train_batch_size=16,  # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,  # 评估时每个设备的批次大小
    warmup_steps=500,  # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,  # 权重衰减，用于防止过拟合
    logging_dir='./logs',  # 日志存储目录
    logging_steps=100,  # 每隔100步记录一次日志
    eval_strategy="epoch",  # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",  # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,  # 训练结束后加载效果最好的模型
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 评估数据集
    compute_metrics=compute_metrics,  # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

print("开始训练")
# 开始训练模型
trainer.train()
print("评估模型")
# 在测试集上进行最终评估
trainer.evaluate()
