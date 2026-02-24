import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (修改为 sin 函数)
X_numpy = np.sort(np.random.rand(200, 1) * 10, axis=0) # 排序方便绘图
y_numpy = np.sin(X_numpy) + np.random.randn(200, 1) * 0.1 # sin(x) + 噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成：拟合目标为 sin(x)")
print("---" * 10)

# 2. 构建多层网络 (替换原有的 a, b 参数)
# 使用 Sequential 构建：输入(1) -> 隐藏层(64) -> 激活层 -> 输出(1)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),       # 拟合曲线建议用 Tanh 或 ReLU
    torch.nn.Linear(64, 1)
)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
# 注意：这里传入的是 model.parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# 4. 训练模型
num_epochs = 2000 # 增加训练次数以保证拟合效果
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 绘图可视化
with torch.no_grad():
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data (sin)', color='blue', alpha=0.5)
plt.plot(X_numpy, y_predicted.numpy(), label='Neural Network Fit', color='red', linewidth=3)
plt.title("Fitting Sine Wave with Multi-Layer Network")
plt.legend()
plt.show()