import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = dataset = pd.read_csv("../dataset.csv", sep="\t", header=None, names=['text', 'label'])

# 按照标签进行分层抽样划分
train_data, test_data = train_test_split(
    df,
    test_size=0.3,  # 测试集占30%，训练集占70%
    stratify=df['label'],  # 按照label列进行分层抽样
    random_state=42  # 设置随机种子保证结果可重现
)

# 分别保存训练集和测试集
train_data.to_csv('train_dataset.csv', index=False, header=False)
test_data.to_csv('test_dataset.csv', index=False, header=False)

print(f"原始数据集大小: {len(df)}")
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

# 查看各类别的分布情况
print("\n训练集中各类别数量:")
print(train_data['label'].value_counts())
print("\n测试集中各类别数量:")
print(test_data['label'].value_counts())
