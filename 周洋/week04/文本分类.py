import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')


# ==================== 1. 明确标签映射 ====================
class SymptomClassifierConfig:
    """症状分类器配置类"""
    # 定义明确的标签映射
    LABEL_MAP = {
        0: "头痛",
        1: "胸痛",
        2: "腹痛"
    }

    # 反向映射（类别名 -> 数字标签）
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    # 类别数量
    NUM_CLASSES = len(LABEL_MAP)

    @classmethod
    def get_label_name(cls, label_id):
        """获取标签ID对应的类别名"""
        return cls.LABEL_MAP.get(label_id, f"未知类别_{label_id}")

    @classmethod
    def get_label_id(cls, label_name):
        """获取类别名对应的标签ID"""
        return cls.REVERSE_LABEL_MAP.get(label_name, -1)


# ==================== 2. 创建数据集（使用明确标签） ====================
def create_symptom_dataset():
    """创建症状分类数据集"""
    # 创建数据时使用明确的标签名
    data = {
        'text': [
            # 头痛相关症状
            "我头痛得很厉害，像要裂开一样",
            "早上起来就感觉头晕头痛",
            "太阳穴两边都在跳着痛",
            "后脑勺持续钝痛，有时会恶心",
            "一紧张就会偏头痛",
            "头痛伴有畏光，想吐的感觉",
            "头部有搏动性疼痛，持续几小时",
            "头痛时脖子也很僵硬",
            "感冒后头痛加重",
            "压力大时太阳穴痛",

            # 胸痛相关症状
            "胸口闷痛，呼吸不畅",
            "左胸刺痛，持续几秒钟",
            "胸骨后有烧灼感",
            "深呼吸时胸痛加重",
            "胸部压榨性疼痛",
            "胸痛放射到左臂",
            "活动后胸痛胸闷",
            "躺下时胸痛更明显",
            "胸部锐痛，位置不固定",
            "胸痛伴有心悸",

            # 腹痛相关症状
            "腹部绞痛，位置在右下腹",
            "胃部隐痛，饭后加重",
            "腹部胀痛，咕咕叫",
            "小腹坠痛，想上厕所",
            "右上腹钝痛",
            "肚脐周围阵发性疼痛",
            "吃了生冷食物后腹痛",
            "腹部隐痛伴有腹泻",
            "月经期下腹痛",
            "腹部疼痛位置游走"
        ],
        'label_name': ['头痛'] * 10 + ['胸痛'] * 10 + ['腹痛'] * 10,  # 使用类别名
        'label_id': [0] * 10 + [1] * 10 + [2] * 10  # 对应的数字标签
    }

    df = pd.DataFrame(data)
    return df


# 创建数据集
df = create_symptom_dataset()
print("=== 数据集信息 ===")
print(f"数据集大小: {len(df)}")
print(f"类别映射: {SymptomClassifierConfig.LABEL_MAP}")
print("\n数据集样例:")
print(df.head())
print("\n类别分布:")
print(df['label_name'].value_counts())


# ==================== 3. 数据集类（包含标签映射） ====================
class SymptomDataset(Dataset):
    """症状分类数据集类"""

    def __init__(self, texts, label_ids, tokenizer, max_len=128):
        self.texts = texts
        self.label_ids = label_ids  # 使用数字标签
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 存储标签映射信息
        self.label_map = SymptomClassifierConfig.LABEL_MAP
        self.num_classes = SymptomClassifierConfig.NUM_CLASSES

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_id = self.label_ids[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label_ids': torch.tensor(label_id, dtype=torch.long)
        }

    def get_label_name(self, label_id):
        """获取标签ID对应的类别名"""
        return self.label_map.get(label_id, "未知")


# 划分数据集
train_texts, val_texts, train_label_ids, val_label_ids = train_test_split(
    df['text'].values,
    df['label_id'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label_id'].values
)

# ==================== 4. 加载BERT模型 ====================
MODEL_NAME = 'bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=SymptomClassifierConfig.NUM_CLASSES,  # 使用配置的类别数
    output_attentions=False,
    output_hidden_states=False
)


# ==================== 5. 创建数据加载器 ====================
def create_data_loader(texts, label_ids, tokenizer, max_len, batch_size, shuffle=True):
    dataset = SymptomDataset(
        texts=texts,
        label_ids=label_ids,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


BATCH_SIZE = 8
MAX_LEN = 64

train_data_loader = create_data_loader(
    train_texts, train_label_ids, tokenizer, MAX_LEN, BATCH_SIZE
)

val_data_loader = create_data_loader(
    val_texts, val_label_ids, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False
)


# ==================== 6. 训练和评估函数 ====================
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label_ids'].to(device)  # 注意这里改为label_ids

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_ids'].to(device)  # 注意这里改为label_ids

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# ==================== 7. 模型微调 ====================
EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 训练模型
print("\n=== 开始训练 ===")
history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler,
        len(train_texts)
    )

    print(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        device,
        len(val_texts)
    )

    print(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
    print()


# ==================== 8. 模型评估（使用标签映射） ====================
def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_ids'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return predictions, prediction_probs, real_values


# 获取预测结果
y_pred, y_pred_probs, y_test = get_predictions(model, val_data_loader, device)

# 转换为标签名
y_test_names = [SymptomClassifierConfig.get_label_name(label_id) for label_id in y_test.numpy()]
y_pred_names = [SymptomClassifierConfig.get_label_name(label_id) for label_id in y_pred.numpy()]

print("\n=== 模型评估 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n详细分类报告:")
print(classification_report(y_test_names, y_pred_names, target_names=list(SymptomClassifierConfig.LABEL_MAP.values())))


# ==================== 9. 测试新样本（完整演示） ====================
class SymptomClassifier:
    """完整的症状分类器"""

    def __init__(self, model, tokenizer, config_class=SymptomClassifierConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config_class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def predict(self, text, return_all_info=False):
        """预测症状类别"""
        self.model.eval()

        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

        # 获取预测结果
        prob_values = probs[0].cpu().numpy()
        predicted_id = torch.argmax(probs, dim=1).item()
        predicted_name = self.config.get_label_name(predicted_id)
        confidence = prob_values[predicted_id]

        if return_all_info:
            # 返回所有类别的详细信息
            all_probs = {
                self.config.get_label_name(i): float(prob_values[i])
                for i in range(self.config.NUM_CLASSES)
            }

            return {
                'text': text,
                'predicted_label_id': predicted_id,
                'predicted_label_name': predicted_name,
                'confidence': float(confidence),
                'all_probabilities': all_probs,
                'label_mapping': self.config.LABEL_MAP
            }
        else:
            return predicted_name, float(confidence)


# 创建分类器
classifier = SymptomClassifier(model, tokenizer)

# 测试样本
test_samples = [
    "我头很痛，感觉要裂开了",
    "胸口一阵阵刺痛，呼吸有点困难",
    "肚子疼得厉害，想上厕所",
    "胃部不舒服，有点恶心",
    "左侧胸部有压迫感",
    "后脑勺发紧，眼睛也胀痛"
]

print("\n=== 新样本测试 ===")
print("=" * 80)

for i, text in enumerate(test_samples, 1):
    print(f"\n测试样本 {i}:")
    print(f"输入文本: 「{text}」")

    # 获取完整预测信息
    result = classifier.predict(text, return_all_info=True)

    print(f"预测结果: {result['predicted_label_name']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"标签ID: {result['predicted_label_id']}")
    print("所有类别概率:")
    for label_name, prob in result['all_probabilities'].items():
        print(f"  {label_name}: {prob:.2%}")

    print("-" * 80)

# ==================== 10. 保存模型和标签映射 ====================
import json
import os


def save_classifier(classifier, save_dir):
    """保存完整的分类器（模型 + 配置）"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 保存模型
    classifier.model.save_pretrained(save_dir)

    # 2. 保存分词器
    classifier.tokenizer.save_pretrained(save_dir)

    # 3. 保存配置信息（包括标签映射）
    config_info = {
        'label_map': classifier.config.LABEL_MAP,
        'num_classes': classifier.config.NUM_CLASSES,
        'model_name': 'bert-base-chinese',
        'description': '医疗症状分类器'
    }

    config_path = os.path.join(save_dir, 'classifier_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    print(f"分类器已保存到: {save_dir}")
    print(f"标签映射: {config_info['label_map']}")


# 保存分类器
save_classifier(classifier, './medical_symptom_classifier')


# ==================== 11. 加载保存的分类器 ====================
def load_classifier(save_dir):
    """加载保存的分类器"""
    # 1. 加载配置
    config_path = os.path.join(save_dir, 'classifier_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_info = json.load(f)

    # 2. 创建配置类
    class LoadedClassifierConfig:
        LABEL_MAP = config_info['label_map']
        NUM_CLASSES = config_info['num_classes']

        @classmethod
        def get_label_name(cls, label_id):
            return cls.LABEL_MAP.get(str(label_id), f"未知类别_{label_id}")

    # 3. 加载模型和分词器
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)

    # 4. 创建分类器实例
    classifier = SymptomClassifier(model, tokenizer, LoadedClassifierConfig)

    print(f"分类器已从 {save_dir} 加载")
    print(f"标签映射: {LoadedClassifierConfig.LABEL_MAP}")

    return classifier


# 测试加载功能
print("\n=== 测试模型加载功能 ===")
loaded_classifier = load_classifier('./medical_symptom_classifier')

# 用加载的分类器进行预测
test_text = "我的头真的很痛"
prediction, confidence = loaded_classifier.predict(test_text)
print(f"输入: {test_text}")
print(f"预测: {prediction} (置信度: {confidence:.2%})")