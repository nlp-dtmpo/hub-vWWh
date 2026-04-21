import torch
import clip
from PIL import Image

# 1. 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 加载本地图片
image_path = "dog.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 4. 定义候选类别（Zero-Shot核心）
text_descriptions = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person"
]

# 5. 文本编码
text = clip.tokenize(text_descriptions).to(device)

# 6. 推理（不计算梯度）
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 7. 归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 8. 计算相似度（softmax 得到概率）
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 9. 输出结果
values, indices = similarity[0].topk(len(text_descriptions))

print("分类结果：")
for value, index in zip(values, indices):
    print(f"{text_descriptions[index]}: {value.item():.4f}")