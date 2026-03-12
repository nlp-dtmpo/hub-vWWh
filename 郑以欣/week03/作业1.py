import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.model_selection import train_test_split

# 读取数据
dataset = pd.read_csv("../Week03/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 划分训练集和测试集（7:3）
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, string_labels, test_size=0.3, random_state=42, stratify=string_labels
)

# 创建标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
train_numerical_labels = [label_to_index[label] for label in train_labels]
test_numerical_labels = [label_to_index[label] for label in test_labels]

# 创建字符到索引的映射
char_to_index = {'<pad>': 0}
for text in texts:  # 使用所有文本构建词表
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
index_to_label = {i: label for label, i in label_to_index.items()}

max_len = 40


# 自定义数据集
class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 创建数据集
train_dataset = CharRNNDataset(train_texts, train_numerical_labels, char_to_index, max_len)
test_dataset = CharRNNDataset(test_texts, test_numerical_labels, char_to_index, max_len)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义不同的模型类
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='rnn'):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:  # 默认使用简单RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, x):
        embedded = self.embedding(x)

        if self.rnn_type.lower() == 'lstm':
            rnn_out, (hidden, _) = self.rnn(embedded)
        else:  # RNN 或 GRU
            rnn_out, hidden = self.rnn(embedded)

        # 取最后一个时间步的隐状态
        out = self.fc(hidden.squeeze(0))
        return out


# 训练和评估函数
def train_and_evaluate(model_type, model_name, num_epochs=10, lr=0.001):
    print(f"\n{'=' * 60}")
    print(f"训练 {model_name} 模型")
    print(f"{'=' * 60}")

    # 初始化模型
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)

    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type=model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    start_time = time.time()
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1:2d}/{num_epochs}], 训练损失: {epoch_loss:.4f}")

    training_time = time.time() - start_time

    # 评估模型
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"训练时间: {training_time:.2f}秒")
    print(f"测试准确率: {accuracy:.2f}%")

    return {
        'model_type': model_type,
        'model_name': model_name,
        'train_losses': train_losses,
        'accuracy': accuracy,
        'training_time': training_time,
        'model': model
    }


# 分别训练三种模型
results = []

# 1. 训练RNN模型
rnn_result = train_and_evaluate('rnn', 'Simple RNN', num_epochs=10)
results.append(rnn_result)

# 2. 训练LSTM模型
lstm_result = train_and_evaluate('lstm', 'LSTM', num_epochs=10)
results.append(lstm_result)

# 3. 训练GRU模型
gru_result = train_and_evaluate('gru', 'GRU', num_epochs=10)
results.append(gru_result)

# 打印对比结果
print(f"\n{'=' * 80}")
print(f"{'模型对比结果':^80}")
print(f"{'=' * 80}")
print(f"{'模型名称':<15} {'测试准确率':<15} {'训练时间(秒)':<15} {'最终训练损失':<15}")
print(f"{'-' * 80}")

for result in results:
    print(f"{result['model_name']:<15} {result['accuracy']:<15.2f}% "
          f"{result['training_time']:<15.2f} {result['train_losses'][-1]:<15.4f}")

print(f"{'=' * 80}")

# 找出最佳模型
best_result = max(results, key=lambda x: x['accuracy'])
print(f"\n🎉 最佳模型: {best_result['model_name']}")
print(f"   测试准确率: {best_result['accuracy']:.2f}%")


# 使用最佳模型进行预测示例
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    """使用训练好的模型进行预测"""
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


# 测试示例文本
print(f"\n{'=' * 80}")
print(f"{'预测示例':^80}")
print(f"{'=' * 80}")

test_cases = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的音乐",
    "今天上海的温度怎么样",
    "带我去最近的加油站"
]

best_model = best_result['model']

for test_text in test_cases:
    try:
        predicted = classify_text_gru(test_text, best_model, char_to_index, max_len, index_to_label)
        print(f"输入: '{test_text}'")
        print(f"预测: '{predicted}'\n")
    except:
        print(f"输入: '{test_text}'")
        print(f"预测: '未知类别' (可能不在训练标签中)\n")

# 可视化训练损失对比
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['train_losses'], label=result['model_name'], linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison: RNN vs LSTM vs GRU')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = './提交的作业/week03/rnn_lstm_gru_comparison.png'
plt.savefig('rnn_lstm_gru_comparison.png', dpi=150, bbox_inches='tight')

print(f"\n📊 训练损失对比图已保存为 'rnn_lstm_gru_comparison.png'")
