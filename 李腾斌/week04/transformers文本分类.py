import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 1. 加载CSV数据集
dataset = load_dataset("csv", data_files="news_dataset.csv")

# 划分训练/测试
dataset = dataset["train"].train_test_split(test_size=0.2)

# 2. 加载BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

# 3. 预处理
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

train_loader = DataLoader(encoded_dataset["train"], batch_size=4, shuffle=True)
test_loader = DataLoader(encoded_dataset["test"], batch_size=4)

# 4. 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 5

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# 5. 验证准确率
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

acc = accuracy_score(true_labels, preds)
print("Test Accuracy:", acc)

# 6. 新样本预测
labels_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()

    return labels_map[pred]

print("\n===== New Sample Prediction =====")
print(predict("Microsoft invests heavily in AI research"))
print(predict("The football match ended with a last-minute goal"))
