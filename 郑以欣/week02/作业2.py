import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成sin函数数据
x_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)  # 生成-2π到2π的数据
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(200, 1)  # sin函数加上噪声

# 转换为PyTorch张量
X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("生成sin函数数据完成。")
print(f"数据点数量: {len(X)}")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        # 三层网络结构
        self.fc1 = nn.Linear(1, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 128)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(128, 64)  # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(64, 1)  # 隐藏层3到输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 3. 创建模型、损失函数和优化器
model = SinNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("模型结构:")
print(model)
print("---" * 10)

# 4. 训练模型
num_epochs = 5000
print("开始训练...")

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印一次
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 5. 可视化结果
# 生成更密集的点用于绘制平滑曲线
x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
X_plot = torch.from_numpy(x_plot).float()

# 获取模型预测结果
model.eval()  # 设置为评估模式
with torch.no_grad():
    y_plot_pred = model(X_plot)
    y_plot_np = y_plot_pred.numpy()

# 创建一个大图，包含三个子图
plt.figure(figsize=(15, 4))

# 子图1: 原始数据和拟合结果
plt.subplot(1, 3, 1)
plt.scatter(x_numpy, y_numpy, label='训练数据 (带噪声)', color='blue', alpha=0.5, s=10)
plt.plot(x_plot, y_plot_np, label='神经网络拟合', color='red', linewidth=2)
plt.plot(x_plot, np.sin(x_plot), label='真实 sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('神经网络拟合 sin(x) 函数')
plt.legend()
plt.grid(True)

# 子图2: 拟合误差
plt.subplot(1, 3, 2)
y_true_sin = np.sin(x_numpy)
with torch.no_grad():
    y_train_pred = model(X)
    y_train_np = y_train_pred.numpy()

error = y_train_np - y_true_sin
plt.scatter(x_numpy, error, color='purple', alpha=0.5, s=10)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('误差 (预测 - 真实)')
plt.title('拟合误差')
plt.grid(True)

# 子图3: 测试模型在训练范围外的表现
plt.subplot(1, 3, 3)
x_extended = np.linspace(-4 * np.pi, 4 * np.pi, 600).reshape(-1, 1)
X_extended = torch.from_numpy(x_extended).float()

with torch.no_grad():
    y_extended_pred = model(X_extended)
    y_extended_np = y_extended_pred.numpy()

plt.plot(x_extended, y_extended_np, label='神经网络预测', color='red', linewidth=2)
plt.plot(x_extended, np.sin(x_extended), label='真实 sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('在更大范围内的预测 (-4π 到 4π)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. 打印一些统计信息
print("拟合统计信息:")
print(f"最大绝对误差: {np.max(np.abs(error)):.4f}")
print(f"平均绝对误差: {np.mean(np.abs(error)):.4f}")
print(f"误差标准差: {np.std(error):.4f}")

# 测试几个具体的点
test_points = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]).reshape(-1, 1)
X_test = torch.from_numpy(test_points).float()

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

print("\n测试点预测:")
for i in range(len(test_points)):
    x_val = test_points[i, 0]
    y_true = np.sin(x_val)
    y_pred = y_test_pred[i].item()
    error_val = y_pred - y_true
    print(f"sin({x_val:.2f}) = {y_true:.4f}, 预测: {y_pred:.4f}, 误差: {error_val:.4f}")