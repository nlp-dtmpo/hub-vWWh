import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

class FAQMatcher(nn.Module):
    def __init__(self):
        super(FAQMatcher, self).__init__()
        # 1. 加载 bert-base-chinese
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Mean Pooling (获取句向量)
        # attention_mask.unsqueeze(-1) 是为了扩充维度以便广播乘法
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # 3. L2 归一化 (方便后续直接算点积)
        return torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

# --- 训练配置 (应用论文策略) ---

model = FAQMatcher()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 策略 A: 分层学习率 (Layer-wise Learning Rate Decay)
# 论文建议基准 LR = 2e-5, Decay Factor = 0.95
base_lr = 2e-5
decay_factor = 0.95
optimizer_grouped_parameters = []

# 对 BERT 的 12 层 Encoder 应用分层衰减
for layer_i in range(12):
    layer_lr = base_lr * (decay_factor ** (11 - layer_i)) # 顶层lr大，底层lr小
    optimizer_grouped_parameters.append({
        'params': model.bert.encoder.layer[layer_i].parameters(),
        'lr': layer_lr
    })

# 对 Embedding 层和 Pooler 层应用最低的学习率
optimizer_grouped_parameters.append({
    'params': model.bert.embeddings.parameters(),
    'lr': base_lr * (decay_factor ** 12)
})

# 初始化优化器
optimizer = AdamW(optimizer_grouped_parameters)

# 策略 B: Slanted Triangular Learning Rates (Warmup)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(total_steps * 0.1), # 10% steps for warmup
    num_training_steps=total_steps
)

# Loss Function: Contrastive Loss 或 Triplet Loss
criterion = nn.CosineEmbeddingLoss(margin=0.2)

# --- 训练循环 (简化版) ---
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        # batch 包含: anchor_input, positive_input, negative_input
        # 1. 编码
        anchor_vec = model(batch['anchor_ids'], batch['anchor_mask'])
        pos_vec = model(batch['pos_ids'], batch['pos_mask'])
        
        # 2. 计算 Loss (让正例相似度接近 1)
        # target=1 表示这两个向量应该相似
        loss = criterion(anchor_vec, pos_vec, torch.ones(anchor_vec.size(0)).to(device))
        
        # 3. 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step() # 更新学习率
        optimizer.zero_grad()
