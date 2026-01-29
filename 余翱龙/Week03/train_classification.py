import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv("../dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40


# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts  # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long)  # 文本对应的标签
        self.char_to_index = char_to_index  # 字符到索引的映射关系
        self.max_len = max_len  # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


class UnifiedClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type='lstm'):
        super(UnifiedClassifier, self).__init__()

        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'rnn':
            self.rnn_layer = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'lstm':
            self.rnn_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'gru':
            self.rnn_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)

        if self.model_type == 'lstm':
            output, (hidden, _) = self.rnn_layer(embedded)
            # 对于LSTM，取最后一个时间步的隐藏状态
            final_hidden = hidden[-1]
        else:  # RNN or GRU
            output, hidden = self.rnn_layer(embedded)
            # 对于RNN和GRU，取最后一个时间步的隐藏状态
            final_hidden = hidden[-1]

        out = self.fc(final_hidden)
        return out


def save_complete_model(model, model_path, char_to_index, label_to_index, model_config):
    """保存完整的模型信息"""
    complete_model_data = {
        'model_state_dict': model.state_dict(),
        'char_to_index': char_to_index,
        'label_to_index': label_to_index,
        'model_config': model_config,
        'max_len': max_len
    }
    torch.save(complete_model_data, model_path)
    print(f"完整模型已保存至 {model_path}")


def load_complete_model(model_path):
    """加载完整模型信息"""
    complete_model_data = torch.load(model_path)

    # 提取配置
    model_config = complete_model_data['model_config']
    char_to_index = complete_model_data['char_to_index']
    label_to_index = complete_model_data['label_to_index']
    max_len_loaded = complete_model_data['max_len']

    # 创建模型
    model = UnifiedClassifier(**model_config)
    model.load_state_dict(complete_model_data['model_state_dict'])
    model.eval()

    return model, char_to_index, label_to_index, max_len_loaded


# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 128  # 64 128
hidden_dim = 256  # 128 256
output_dim = len(label_to_index)

# rnn_model = UnifiedClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'rnn')
lstm_model = UnifiedClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'lstm')
# gru_model = UnifiedClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 'gru')
model = lstm_model  # [rnn_model, lstm_model, gru_model]
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_path = 'res/lstm_model.pt'

# 定义模型配置
model_config = {
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'hidden_dim': hidden_dim,
    'output_dim': output_dim,
    'model_type': 'lstm_model' # [rnn_model, lstm_model, gru_model]
}

# 在训练开始前初始化损失记录列表
train_losses = []
num_epochs = 4
if os.path.exists(model_path):
    print("载入已有的模型权重及配置...")
    model, char_to_index, label_to_index, loaded_max_len = load_complete_model(model_path)
    index_to_label = {i: label for label, i in label_to_index.items()}
else:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    save_complete_model(model, model_path, char_to_index, label_to_index, model_config)
    print("训练损失历史:", train_losses)


def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


def evaluate_model_on_test_set(test_file_path, model, char_to_index, max_len, label_to_index):
    """
    读取测试集并评估模型性能

    Args:
        test_file_path: 测试集文件路径
        model: 训练好的模型
        char_to_index: 字符到索引的映射
        max_len: 最大文本长度
        label_to_index: 标签到索引的映射
    """
    try:
        # 读取测试集
        test_dataset = pd.read_csv(test_file_path, sep="\t", header=None)
        test_texts = test_dataset[0].tolist()
        test_string_labels = test_dataset[1].tolist()

        # 将字符串标签转换为数值标签
        test_numerical_labels = [label_to_index[label] for label in test_string_labels if label in label_to_index]

        # 创建测试数据集
        test_lstm_dataset = CharLSTMDataset(test_texts, test_numerical_labels, char_to_index, max_len)
        test_dataloader = DataLoader(test_lstm_dataset, batch_size=32, shuffle=False)

        # 开始评估
        model.eval()
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for inputs, true_labels in test_dataloader:
                outputs = model(inputs)
                _, predicted_indices = torch.max(outputs, 1)

                all_predictions.extend(predicted_indices.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())

        # 计算评估指标
        accuracy = accuracy_score(all_true_labels, all_predictions)
        classification_rep = classification_report(
            all_true_labels,
            all_predictions,
            target_names=list(label_to_index.keys()),
            digits=4
        )
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)

        print(f"测试集准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_rep)
        print("\n混淆矩阵:")
        print(conf_matrix)

        # 返回评估结果
        return {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'true_labels': all_true_labels
        }

    except FileNotFoundError:
        print(f"错误: 找不到测试集文件 {test_file_path}")
        return None
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        return None


test_file_path = "../dataset.csv"

# 如果存在测试集文件，则进行评估
if os.path.exists(test_file_path):
    print("\n开始对测试集进行评估...")
    evaluation_results = evaluate_model_on_test_set(
        test_file_path=test_file_path,
        model=model,
        char_to_index=char_to_index,
        max_len=max_len,
        label_to_index=label_to_index
    )

    if evaluation_results:
        print("测试集评估完成！")
else:
    print(f"\n未找到测试集文件 {test_file_path}，跳过评估步骤")
index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
