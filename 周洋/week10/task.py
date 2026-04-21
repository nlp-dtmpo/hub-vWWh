import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 加载模型
model = ChineseCLIPModel.from_pretrained("./model/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./model/chinese-clip-vit-base-patch16")
model.eval()

# 加载图片
image = Image.open("./dog.jpg")  # 替换为你的图片路径

# 定义类别
classes = ["狗", "猫", "鸟", "鱼", "兔子", "仓鼠", "乌龟"]

# 分类
inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

# 打印结果
for cls, prob in zip(classes, probs[0]):
    print(f"{cls}: {prob.item():.4f}")