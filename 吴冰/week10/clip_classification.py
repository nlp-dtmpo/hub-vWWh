import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch


img_path = './data/v2-bd4aef6ccbef9a4842bb8a5db268613c_r.jpg'
image = Image.open(img_path)


import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100  # 提高分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 8)

model = ChineseCLIPModel.from_pretrained("./model/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("./model/chinese-clip-vit-base-patch16") # 预处理

candidate_classes = [
    "一只狗", "一只猫", "一辆汽车", "一朵花", "一个人",
    "一座山", "一片海", "一棵树", "一只鸟", "一栋建筑"
]


image_inputs = processor(images=image, return_tensors="pt")
text_inputs = processor(text=candidate_classes, padding=True, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    text_inputs= model.get_text_features(**text_inputs)

    image_features = image_features.pooler_output.numpy()
    text_inputs= text_inputs.pooler_output.numpy()


img_image_feat = normalize(image_features)
te_text_feat = normalize(text_inputs)
sim_result = np.dot(img_image_feat, te_text_feat.T)
print("sim_result:",sim_result)

sim_idx = sim_result.argmax()

print('文本识别结果: ', sim_idx)
print('对应的类别: ', candidate_classes[sim_idx])


print('\n所有类别的相似度分数:')
for i, cls in enumerate(candidate_classes):
    print(f'{cls}: {sim_result[0][i]:.4f}')
