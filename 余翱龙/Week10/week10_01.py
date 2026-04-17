# 核心依赖：数值计算、图像处理、CLIP模型、PyTorch、可视化
import numpy as np
from PIL import Image
import requests
from sklearn.preprocessing import normalize
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import matplotlib.pyplot as plt

# 加载预训练的中文CLIP模型与处理器（指定本地模型路径）
model = ChineseCLIPModel.from_pretrained('models/chinese-clip-vit-base-patch16/')
processor = ChineseCLIPProcessor.from_pretrained('models/chinese-clip-vit-base-patch16/')

# 定义图片路径并加载
image_path = "images.jpg"
img = [Image.open(image_path)]

# 处理器将图片转为模型输入张量（'pt'表示PyTorch张量格式）
input = processor(images=img, return_tensors='pt')

# 推理模式下提取图片特征（禁用梯度计算提升效率）
img_image_feat = []
with torch.no_grad():
    image_feature = model.get_image_features(**input)  # 提取图片向量
    image_feature = image_feature.data.numpy()          # 转为NumPy数组
    img_image_feat.append(image_feature)

# 拼接并归一化图片特征（归一化用于后续余弦相似度计算）
img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)

# 定义候选文本列表（预设需匹配的物体描述）
texts = ['这是一只 小狗', '这是一只 小猫', '这是一只 小鸟', '这是一只 鱼', '这是 树']

# 处理器将文本转为模型输入张量
inputs = processor(text=texts, return_tensors='pt', padding=True)

# 推理模式下提取文本特征
img_texts_feat = []
with torch.no_grad():
    text_features = model.get_text_features(**inputs)  # 提取文本向量
    text_features = text_features.data.numpy()
    img_texts_feat.append(text_features)

# 拼接并归一化文本特征
img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)

# 打印特征形状（验证向量维度是否匹配）
print(img_texts_feat.shape)

# 计算图片特征与所有文本特征的点积（即余弦相似度，因已归一化）
sim_result = np.dot(img_image_feat[0], img_texts_feat.T)

# 按相似度降序排序并取Top1匹配（排除自身后取最相似的）
sim_idx = sim_result.argsort()[::-1][0]

# 输出结果：打印最匹配的文本
print(f"最匹配的文本是: {texts[sim_idx]}")