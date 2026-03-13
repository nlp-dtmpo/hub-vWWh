import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# 设置随机种子以保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# 设置cuDNN以获得可重复的结果
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 加载数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符编码
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


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


# ==================== 模型定义 ====================

# 1. RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=1, bidirectional=False, dropout=0.3):
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN层
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )

        self.dropout = nn.Dropout(dropout)

        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc1 = nn.Linear(rnn_output_dim, rnn_output_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(rnn_output_dim // 2, output_dim)

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)

        rnn_out, hidden_state = self.rnn(embedded)

        if self.bidirectional:
            forward_hidden = hidden_state[-2, :, :]
            backward_hidden = hidden_state[-1, :, :]
            combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            combined_hidden = hidden_state[-1, :, :]

        combined_hidden = self.dropout(combined_hidden)

        out = self.fc1(combined_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# 2. GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=2, bidirectional=True, dropout=0.3):
        super(GRUClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc1 = nn.Linear(gru_output_dim, gru_output_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(gru_output_dim // 2, output_dim)

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)

        gru_out, hidden_state = self.gru(embedded)

        if self.bidirectional:
            forward_hidden = hidden_state[-2, :, :]
            backward_hidden = hidden_state[-1, :, :]
            combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            combined_hidden = hidden_state[-1, :, :]

        combined_hidden = self.dropout(combined_hidden)

        out = self.fc1(combined_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# 3. LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=2, bidirectional=True, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(lstm_output_dim // 2, output_dim)

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)

        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        if self.bidirectional:
            forward_hidden = hidden_state[-2, :, :]
            backward_hidden = hidden_state[-1, :, :]
            combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            combined_hidden = hidden_state[-1, :, :]

        combined_hidden = self.dropout(combined_hidden)

        out = self.fc1(combined_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# ==================== GPU训练和评估函数 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, model_name="Model", device=device):
    """训练模型并返回训练历史"""
    # 将模型移动到GPU
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            # 将数据移动到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # 验证阶段
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    training_time = time.time() - start_time
    print(f"{model_name} 训练完成，总时间: {training_time:.2f}秒")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'training_time': training_time
    }


def evaluate_model(model, dataloader, criterion=None, device=device):
    """评估模型性能"""
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            # 将数据移动到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())  # 移回CPU进行存储
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions) * 100

    if criterion:
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    else:
        return accuracy


def predict_single_text(text, model, char_to_index, max_len, index_to_label, device=device):
    """预测单个文本"""
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    model.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()

    _, predicted_index = torch.max(output, 1)
    predicted_label = index_to_label[predicted_index.cpu().item()]  # 移回CPU

    return predicted_label, confidence


def compare_models(models_dict, test_loader, device=device):
    """比较不同模型的性能"""
    results = {}

    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)

    for name, model in models_dict.items():
        accuracy = evaluate_model(model, test_loader, device=device)
        results[name] = accuracy

        print(f"{name}: 测试准确率 = {accuracy:.2f}%")

    print("=" * 60)

    # 找出最佳模型
    best_model_name = max(results, key=results.get)
    print(f"\n最佳模型: {best_model_name} (准确率: {results[best_model_name]:.2f}%)")

    return results


def plot_training_history(histories, metric='accuracy'):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 5))

    if metric == 'accuracy':
        plt.subplot(1, 2, 1)
        for name, history in histories.items():
            plt.plot(history['train_accuracies'], label=f'{name} (训练)')
        plt.title('训练准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        for name, history in histories.items():
            plt.plot(history['val_accuracies'], label=f'{name} (验证)')
        plt.title('验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True)

    elif metric == 'loss':
        plt.subplot(1, 2, 1)
        for name, history in histories.items():
            plt.plot(history['train_losses'], label=f'{name} (训练)')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        for name, history in histories.items():
            plt.plot(history['val_losses'], label=f'{name} (验证)')
        plt.title('验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict, training_times):
    """绘制模型对比图表"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 准确率对比
    models = list(results_dict.keys())
    accuracies = list(results_dict.values())

    bars = axes[0].bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0].set_title('模型测试准确率对比')
    axes[0].set_ylabel('准确率 (%)')
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{acc:.2f}%', ha='center', va='bottom')

    # 训练时间对比
    times = [training_times[model] for model in models]
    bars2 = axes[1].bar(models, times, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1].set_title('模型训练时间对比')
    axes[1].set_ylabel('时间 (秒)')
    axes[1].grid(axis='y', alpha=0.3)

    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{time_val:.2f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# ==================== GPU内存优化函数 ====================

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU内存")


def monitor_gpu_memory():
    """监控GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        cached = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU内存使用: {allocated:.2f} GB (已分配) / {cached:.2f} GB (缓存)")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print(f"\n开始运行，使用设备: {device}")
    monitor_gpu_memory()

    # 创建数据集
    dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)

    # 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\n数据集划分:")
    print(f"- 训练集: {train_size} 个样本")
    print(f"- 验证集: {val_size} 个样本")
    print(f"- 测试集: {test_size} 个样本")
    print(f"- 类别数: {len(label_to_index)}")
    print(f"- 词表大小: {vocab_size}")

    # 创建数据加载器
    batch_size = 64  # GPU可以处理更大的batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n批量大小: {batch_size}")

    # 模型参数
    embedding_dim = 128
    hidden_dim = 256
    output_dim = len(label_to_index)

    # ==================== 训练三个模型 ====================

    # 1. 训练RNN模型
    print("\n" + "=" * 60)
    print("训练RNN模型...")
    print("=" * 60)

    clear_gpu_memory()

    rnn_model = RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        bidirectional=False,
        dropout=0.3
    ).to(device)  # 创建时直接移到GPU

    rnn_criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

    rnn_history = train_model(
        rnn_model, train_loader, val_loader, rnn_criterion, rnn_optimizer,
        num_epochs=10, model_name="RNN", device=device
    )

    monitor_gpu_memory()

    # 2. 训练GRU模型
    print("\n" + "=" * 60)
    print("训练GRU模型...")
    print("=" * 60)

    clear_gpu_memory()

    gru_model = GRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    ).to(device)

    gru_criterion = nn.CrossEntropyLoss()
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

    gru_history = train_model(
        gru_model, train_loader, val_loader, gru_criterion, gru_optimizer,
        num_epochs=10, model_name="GRU", device=device
    )

    monitor_gpu_memory()

    # 3. 训练LSTM模型
    print("\n" + "=" * 60)
    print("训练LSTM模型...")
    print("=" * 60)

    clear_gpu_memory()

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    ).to(device)

    lstm_criterion = nn.CrossEntropyLoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    lstm_history = train_model(
        lstm_model, train_loader, val_loader, lstm_criterion, lstm_optimizer,
        num_epochs=10, model_name="LSTM", device=device
    )

    monitor_gpu_memory()

    # ==================== 模型对比 ====================

    # 收集所有模型
    models_dict = {
        'RNN': rnn_model,
        'GRU': gru_model,
        'LSTM': lstm_model
    }

    # 在测试集上评估所有模型
    test_results = compare_models(models_dict, test_loader, device=device)

    # 收集训练历史
    histories = {
        'RNN': rnn_history,
        'GRU': gru_history,
        'LSTM': lstm_history
    }

    training_times = {
        'RNN': rnn_history['training_time'],
        'GRU': gru_history['training_time'],
        'LSTM': lstm_history['training_time']
    }

    # ==================== 可视化结果 ====================

    print("\n生成可视化图表...")

    # 绘制准确率对比图
    plot_training_history(histories, metric='accuracy')

    # 绘制损失对比图
    plot_training_history(histories, metric='loss')

    # 绘制模型性能对比图
    plot_model_comparison(test_results, training_times)

    # ==================== 模型参数统计 ====================

    print("\n" + "=" * 60)
    print("模型参数统计")
    print("=" * 60)

    for name, model in models_dict.items():
        model.cpu()  # 移回CPU以节省GPU内存
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}模型:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 训练时间: {training_times[name]:.2f}秒")
        model.to(device)  # 移回GPU

    # ==================== 测试单个文本 ====================

    print("\n" + "=" * 60)
    print("测试单个文本分类")
    print("=" * 60)

    # 创建反向标签映射
    index_to_label = {i: label for label, i in label_to_index.items()}

    test_texts = [
        "帮我导航到北京",
        "查询明天北京的天气",
        "播放周杰伦的音乐",
        "打开微信",
        "今天股市行情怎么样",
        "设置明天早上7点的闹钟",
        "翻译英文到中文"
    ]

    # 使用最佳模型进行预测
    best_model_name = max(test_results, key=test_results.get)
    best_model = models_dict[best_model_name]

    print(f"\n使用最佳模型 ({best_model_name}) 进行预测:")
    print("-" * 40)

    for text in test_texts:
        predictions = {}
        confidences = {}

        # 使用所有模型进行预测
        for name, model in models_dict.items():
            pred_label, confidence = predict_single_text(
                text, model, char_to_index, max_len, index_to_label, device
            )
            predictions[name] = pred_label
            confidences[name] = confidence

        print(f"\n输入文本: '{text}'")
        for name in ['RNN', 'GRU', 'LSTM']:
            print(f"  {name}: {predictions[name]} (置信度: {confidences[name]:.2%})")

    # 清理GPU内存
    clear_gpu_memory()
    print("\n程序执行完毕！")