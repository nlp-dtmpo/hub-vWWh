import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel


def normalize(features):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norm + 1e-10)


IMAGE_PATH = "./dog.jpeg"
CLASS_LABELS = ["猫", "风景", "汽车", "狗", "美食", "人物", "植物"]

model = ChineseCLIPModel.from_pretrained("./models/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./models/chinese-clip-vit-base-patch16")
model.eval()
torch.set_grad_enabled(False)

image = Image.open(IMAGE_PATH)
image_inputs = processor(images=image, return_tensors="pt")

text_inputs = processor(text=CLASS_LABELS, padding=True, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)
image_features = image_features.data.numpy()
image_features = normalize(image_features)

text_features = model.get_text_features(**text_inputs)
text_features = text_features.data.numpy()
text_features = normalize(text_features)

sim_result = np.dot(image_features, text_features.T).squeeze()
sim_idx = sim_result.argsort()[::-1][:3]
print('文本识别结果[1-3]: ', [CLASS_LABELS[x] for x in sim_idx])

sorted_idx = sim_result.argsort()[::-1]
for idx in sorted_idx:
    print(f"类别：{CLASS_LABELS[idx]} | 相似度：{sim_result[idx]:.4f}")