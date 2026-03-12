import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

dataset_df = pd.read_csv('./dataset.csv', sep=' ', header=None)

lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df[0])
num_to_label = dict(zip(range(len(lbl.classes_)), lbl.classes_))
print(num_to_label)
texts = list(dataset_df[1].values)

x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

tokenizer = BertTokenizer.from_pretrained('model/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('model/bert-base-chinese', num_labels=6)

train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=100)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=100)

train_dataset = Dataset.from_dict({
    'input_ids': train_encoding['input_ids'],
    'attention_mask': train_encoding['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encoding['input_ids'],
    'attention_mask': test_encoding['attention_mask'],
    'labels': test_labels
})

train_args = TrainingArguments(
    output_dir='./weights',
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./log',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accurary': (predictions == labels).mean()}



trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    torch.save(best_model.state_dict(), 'weights/bert.pt')
    
    
    

