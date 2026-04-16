import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# ------------------------------
# 1. 加载模型和处理器
# ------------------------------
model_name = "./model/chinese-clip-vit-base-patch16"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChineseCLIPModel.from_pretrained(model_name).to(device)
processor = ChineseCLIPProcessor.from_pretrained(model_name)

# ------------------------------
# 2. 定义中文类别标签
# ------------------------------
class_names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
# 可选：添加提示模板（中文CLIP可直接使用类别词，也可加修饰）
texts = [f"{c}的照片" for c in class_names]  # 例如 "猫的照片"

# ------------------------------
# 3. 预处理文本（只需一次）
# ------------------------------
text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    if not isinstance(text_features, torch.Tensor):
        text_features = text_features.pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


# ------------------------------
# 4. 图像分类函数
# ------------------------------
def classify_image(image_path, top_k=3):
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 预处理图像（尺寸、归一化等）
    image_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        print(type(image_features))
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算相似度（余弦相似度 -> logits）
        logits_per_image = (image_features @ text_features.T) * model.logit_scale.exp()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

    # 返回 top-k
    top_indices = probs.argsort()[::-1][:top_k]
    results = [(class_names[i], probs[i]) for i in top_indices]
    return results


# ------------------------------
# 5. 使用示例
# ------------------------------
if __name__ == "__main__":
    img_path = "Bengal_103_jpg.rf.bb684501a53abf381a6bf07f9cd2243c.jpg"  # 替换为实际图片路径
    predictions = classify_image(img_path)
    print("预测结果：")
    for label, prob in predictions:
        print(f"  {label}: {prob:.4f}")
