import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification

label_list = ['健康与医疗', '出行与交通', '工作与业务沟通', '日常问候与寒暄', '购物与消费', '饮食与餐饮']

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encoding.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)



classify_text = ['今天吃饭了吗？', '我想吃北京烤鸭', '今天工作有点多', '我要买最新的苹果手机']
device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('./model/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./model/bert-base-chinese', num_labels=6)

model.load_state_dict(torch.load('./weights/bert.pt'))
model.to(device)

test_encoding = tokenizer(list(classify_text), truncation=True, padding=True, max_length=100)
test_dataset = NewsDataset(test_encoding, [0] * len(classify_text))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model.eval()
pred = []
for batch in test_loader:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs[1]
    logits = logits.detach().cpu().numpy()
    pred += list(np.argmax(logits, axis=1).flatten())

for idx, text in enumerate(classify_text):
    print(text, '   预测结果 ---->', label_list[pred[idx]])