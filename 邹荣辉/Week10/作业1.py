from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import torch.nn.functional as F

# 1. 加载模型
model_path = "C:/Users/Administrator/Desktop/研究生/ai2026/AI大模型学习/models/chinese-clip-vit-base-patch16"
model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path)

# 2. 候选类别标签
candidate_labels = ["狗", "猫", "鸟", "汽车", "房子", "人", "树", "花"]

# 3. 文本编码（一次性对所有标签进行编码，得到文本特征矩阵）
text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    # L2 归一化
    text_features = F.normalize(text_features, dim=1)

# 4. 图像编码
image_path = "C:/Users/Administrator/Desktop/研究生/ai2026/AI大模型学习/Week10/dog.jpg"
image = Image.open(image_path)
image_inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    image_features = F.normalize(image_features, dim=1)

# 5. 计算相似度并分类
# 相似度 = 图像特征与文本特征的点积（形状: 1 x num_labels）
similarity = image_features @ text_features.T
probs = similarity.softmax(dim=1)

# 6. 输出结果
pred_idx = probs.argmax().item()
print(f"预测类别: {candidate_labels[pred_idx]}, 置信度: {probs[0][pred_idx]:.4f}")

# 打印所有类别的概率
print("\n各类别概率：")
for label, prob in zip(candidate_labels, probs[0]):
    print(f"{label}: {prob:.4f}")