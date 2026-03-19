import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# 1. 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 读取数据
data_df = pd.read_csv('Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv', sep=',')
texts = data_df['text'].astype(str).tolist()

lbl = LabelEncoder()
labels = lbl.fit_transform(data_df['label'])

print("类别标签:", lbl.classes_)

# 3. 数据划分
x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,      # 20%作为测试集
    stratify=labels,    # 按标签比例分层抽样
    random_state=42     # 随机种子，保证可重复性
)

# 4. 定义dataset
class ClassifyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=40):
        # 初始化：保存文本、标签、分词器和最大长度
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # 返回数据集大小
        return len(self.texts)

    def __getitem__(self, idx):
        # 获取单个样本：将文本转换为BERT输入格式
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),              # 文本的token IDs
            'attention_mask': encoding['attention_mask'].squeeze(0),    # 注意力掩码
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # 标签
        }


# 5. 定义model、分词器
tokenizer = BertTokenizer.from_pretrained('models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'models/google-bert/bert-base-chinese',
    num_labels=len(lbl.classes_)
).to(device)


# 6. 定义数据加载器
train_loader = DataLoader(
    ClassifyDataset(x_train, y_train, tokenizer),
    batch_size=128, # 每批128个样本
    shuffle=True    # 训练时打乱数据
)

test_loader = DataLoader(
    ClassifyDataset(x_test, y_test, tokenizer),
    batch_size=128,
    shuffle=False   # 测试时不需要打乱
)

# 7. 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)


# 8. 模型训练、验证函数
def train_epoch(model, dataloader):
    model.train() # 设置为训练模式
    total_loss, correct, total = 0, 0, 0

    for batch in dataloader:
        optimizer.zero_grad()   # 清空梯度

        # 前向传播
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )

        loss = outputs.loss # 计算损失
        loss.backward()     # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()   # 更新参数

        # 计算准确率
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)    # 获取预测类别索引
        correct += (preds == batch['labels'].to(device)).sum().item()
        total += batch['labels'].size(0)

    return total_loss / len(dataloader), correct / total # 平均损失和准确率


def evaluate(model, dataloader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)

            correct += (preds == batch['labels'].to(device)).sum().item()
            total += batch['labels'].size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    # 类别准确率
    class_acc = {}
    for i, cls in enumerate(lbl.classes_):
        mask = np.array(all_labels) == i
        if mask.any():
            class_acc[cls] = (np.array(all_preds)[mask] == i).mean()

    return total_loss / len(dataloader), correct / total, class_acc

# 9. 预测函数
def predict(text):
    model.eval()
    # 对输入文本进行编码
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=40,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )

        probs = F.softmax(outputs.logits, dim=1)
        idx = probs.argmax(dim=1).item()

    return lbl.inverse_transform([idx])[0], probs[0][idx].item()

# 10. 模型训练
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc, class_acc = evaluate(model, test_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    print("类别准确率:")
    for k, v in class_acc.items():
        print(f"  {k}: {v:.4f}")

# 保存权重等文件
SAVE_DIR = "results"
model.save_pretrained(SAVE_DIR)         # 保存模型权重
tokenizer.save_pretrained(SAVE_DIR)     # 保存分词器

# 同时保存label encoder
import pickle
with open(f"{SAVE_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(lbl, f) # 保存标签编码器

print(f"\n label encoder：`{SAVE_DIR}`")


# 示例
strr = [
    "今天天气真好，心情特别愉快",
    "这个消息让我非常震惊",
    "我对这个结果感到失望",
    "你想不想去吃午饭？",
    "哦！我被选中了！",
    "我身体不舒服，肚子好痛"
]
for text in strr:
    label, conf = predict(text)
    print(f"\n{text}\n预测: {label} | 置信度: {conf:.2%}")