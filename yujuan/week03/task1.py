import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

max_len = 40

class CharLSTMDataset(Dataset):
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

# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# ==================== GRU 分类模型 ====================
class GRUClassifier(nn.Module):
    """
    GRU 文本分类器
    结构：Embedding -> GRU -> 取最后时刻隐状态 -> 全连接 -> 输出
    与 LSTM 区别：GRU 只有 hidden_state，没有 cell_state
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        # 词表大小 -> 嵌入维度（可训练）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU：输入维度 embedding_dim，隐状态维度 hidden_dim，batch_first=True 表示输入形状为 (batch, seq, feature)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # 隐状态 -> 类别数
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len) -> embedded: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # GRU 输出：gru_out (batch, seq_len, hidden_dim), hidden (1, batch, hidden_dim)
        # GRU 没有 cell_state，只有 hidden；hidden 为最后时刻的隐状态
        gru_out, hidden = self.gru(embedded)
        # hidden.squeeze(0) -> (batch, hidden_dim)，再经过全连接得到 logits
        out = self.fc(hidden.squeeze(0))
        return out


# ==================== RNN 分类模型 ====================
class RNNClassifier(nn.Module):
    """
    基础 RNN 文本分类器
    结构：Embedding -> RNN(vanilla) -> 取最后时刻隐状态 -> 全连接 -> 输出
    与 LSTM/GRU 区别：基础 RNN 只有隐状态，无门控，长序列易梯度消失
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # nn.RNN：输入维度 embedding_dim，隐状态维度 hidden_dim，batch_first=True
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len) -> embedded: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # RNN 输出：rnn_out (batch, seq_len, hidden_dim), hidden (1, batch, hidden_dim)
        rnn_out, hidden = self.rnn(embedded)
        # 取最后时刻隐状态 (batch, hidden_dim)，再全连接得到 logits
        out = self.fc(hidden.squeeze(0))
        return out


lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器：当 loss 连续 3 个 epoch 不降时，lr 乘 0.5（可选）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

num_epochs = 20  # 增加训练轮数，4 轮通常不够
max_grad_norm = 1.0  # 梯度裁剪，防止梯度爆炸导致 loss 很大

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪：限制梯度的 L2 范数，避免爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    avg_loss = running_loss / len(dataloader)
    scheduler.step(avg_loss)  # 根据本 epoch 平均 loss 调整学习率
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

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

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
