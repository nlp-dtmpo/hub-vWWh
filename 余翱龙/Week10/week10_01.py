import numpy as np
from PIL import Image
import requests
from sklearn.preprocessing import normalize
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import matplotlib.pyplot as plt
model = ChineseCLIPModel.from_pretrained('models/chinese-clip-vit-base-patch16/')
processor = ChineseCLIPProcessor.frompretrained('models/chineseclipvitbasepatch16/')
image_path = "images.jpg"
img = [Image.open(image_path)]
input = processor(images=img, return_tensors='pt')
imgimagefeat = []
with torch.nograd():
 image_feature = model.get_image_features(**input)
 imagefeature = image_feature.data.numpy()
 imgimagefeat.append(imagefeature)
img_image_feat = np.vstack(img_image_feat)
imgimagefeat = normalize(imgimagefeat)
img_texts_feat = []
texts = ['这是⼀只：⼩狗', '这是⼀只：⼩猫', '这是⼀只：⼩⻦', '这是⼀只：⻥', '这是⼀只：树']
inputs = processor(text=texts, return_tensors='pt', padding=True)
with torch.no_grad():
 textfeatures = model.gettextfeatures(**inputs)
 textfeatures = textfeatures.data.numpy()
 img_texts_feat.append(text_features)
imgtextsfeat = np.vstack(imgtextsfeat)
img_texts_feat = normalize(img_texts_feat)
print(imgtextsfeat.shape)
sim_result = np.dot(img_image_feat[0], img_texts_feat.T)
simidx = simresult.argsort()[::-1][0]