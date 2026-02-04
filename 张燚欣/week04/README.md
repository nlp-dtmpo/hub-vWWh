# BERT在AG News数据集上的微调项目

## 📋 项目描述
使用BERT-base模型在AG News公开数据集上进行微调，实现新闻文本分类。

## 🎯 任务要求
1. 使用公开数据集（AG News）
2. 复现BERT-base模型的微调过程
3. 输入新样本进行测试验证
4. 至少3个类别（实际为4个类别）

## 📊 数据集
- **数据集名称**: AG News
- **类别数量**: 4个
- **类别标签**: 
  - 0: World（世界新闻）
  - 1: Sports（体育新闻）
  - 2: Business（商业新闻）
  - 3: Sci/Tech（科技新闻）
- **数据量**: 约12000条训练数据，7600条测试数据

## 🛠 技术栈
- **深度学习框架**: PyTorch
- **预训练模型**: BERT-base-uncased
- **分词器**: BERT Tokenizer
- **优化器**: AdamW
- **学习率调度**: Linear Schedule with Warmup

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt