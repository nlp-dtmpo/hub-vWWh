import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据
dataset_df = pd.read_csv("movie_dataset.csv", sep="\t", header=None)


# ==================== 修正1：提取第一个标签 ====================
# 提取每个文本的第一个标签（假设标签用逗号分隔）
def extract_first_label(label_str):
    # 分割标签并取第一个
    labels = label_str.split(',')
    return labels[0].strip() if labels else label_str.strip()


# 应用函数提取第一个标签
first_labels = [extract_first_label(label) for label in dataset_df[1].values[:5000]]
texts = list(dataset_df[0].values[:5000])

# ==================== 修正2：使用英文BERT模型 ====================
# 初始化 LabelEncoder
lbl = LabelEncoder()
# 拟合第一个标签数据
labels = lbl.fit_transform(first_labels)

print(f"标签类别数量: {len(lbl.classes_)}")
print(f"标签类别: {lbl.classes_}")

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,  # 文本数据
    labels,  # 对应的数字标签
    test_size=0.2,  # 测试集比例为20%
    stratify=labels  # 确保训练集和测试集的标签分布一致
)

# 使用英文BERT模型（处理英文文本效果更好）
# 需要先下载模型，可以运行：from transformers import BertTokenizer, BertForSequenceClassification
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(r'D:\python_code\extension\models\google-bert\bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(r'D:\python_code\extension\models\google-bert\bert-base-uncased', num_labels=len(lbl.classes_))

# 使用分词器对训练集和测试集的文本进行编码
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}


# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练模型
print("开始训练模型...")
trainer.train()

# 在测试集上进行最终评估
print("\n模型评估结果:")
eval_results = trainer.evaluate()
print(eval_results)


# ==================== 对新文本进行分类 ====================

def predict_text(text):
    """
    对单条文本进行分类预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用分词器处理文本
    encoding = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
    encoding = {key: value.to(device) for key, value in encoding.items()}
    # 设置模型为评估模式
    model.eval()

    # 进行预测
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(predictions, dim=-1).item()

    # 将预测的数字标签转换回原始文本标签
    predicted_label = lbl.inverse_transform([predicted_class_id])[0]
    confidence = predictions[0][predicted_class_id].item()

    # 获取所有类别的置信度
    all_confidences = {lbl.inverse_transform([i])[0]: predictions[0][i].item() for i in range(len(lbl.classes_))}

    return predicted_label, confidence, all_confidences


# 新文本
new_texts = [
    "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope.",
    "A mysterious warrior teams up with the daughter and son of a deposed Chinese Emperor to defeat their cruel brother, who seeks their deaths."
]

print("\n" + "=" * 80)
print("新文本分类结果:")
print("=" * 80)

for i, text in enumerate(new_texts, 1):
    predicted_label, confidence, all_confidences = predict_text(text)
    print(f"\n文本 {i}:")
    print(f"预测类别: {predicted_label}")
    print(f"置信度: {confidence:.4f}")
    print(f"文本长度: {len(text)} 字符")

    # 显示置信度最高的前3个类别
    sorted_confidences = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)
    print("置信度最高的前3个类别:")
    for j, (label, conf) in enumerate(sorted_confidences[:3], 1):
        print(f"  {j}. {label}: {conf:.4f}")

    print(f"文本摘要: {text[:100]}...")

# # 保存模型和标签编码器
# print("\n" + "=" * 80)
# print("保存模型...")
# model.save_pretrained('./saved_bert_model')
# tokenizer.save_pretrained('./saved_bert_model')

# import pickle
#
# with open('./saved_bert_model/label_encoder.pkl', 'wb') as f:
#     pickle.dump(lbl, f)
#
# print("模型和标签编码器已保存到 './saved_bert_model' 目录")
# print(f"可用的类别标签: {list(lbl.classes_)}")