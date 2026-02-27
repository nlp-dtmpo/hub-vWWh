流程图：

![流程图](流程图.png)
核心代码解析

2.1 加载模型和分词器

```python
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

```


# 加载中文 BERT 分词器和模型
```
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
```
2.2 文本编码过程
```
text = "酒店环境很好，服务贴心"

# 使用分词器编码
encoding = tokenizer(
    text,
    truncation=True,     # 超长截断
    padding=True,        # 不足填充
    max_length=64,       # 最大长度
    return_tensors='pt'  # 返回 PyTorch 张量
)

# encoding 包含：
# - input_ids: 文本的数字表示
# - attention_mask: 注意力掩码
```
2.3 获取文本向量
```
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

with torch.no_grad():
    outputs = model(**encoding)
    # 取 [CLS] 标记的向量作为整句话的表示
    sentence_embedding = outputs.last_hidden_state[:, 0, :]
```
---
四、相似度计算方法
获得文本向量后，可以使用余弦相似度计算两段文本的相似程度：
```
from torch.nn.functional import cosine_similarity

# 假设有两个文本的向量
vec1 = get_embedding("酒店服务很好")
vec2 = get_embedding("宾馆服务不错")

# 计算余弦相似度（范围：-1到1，越接近1越相似）
similarity = cosine_similarity(vec1, vec2)
print(f"相似度: {similarity.item():.4f}")
```
相似度计算流程：
```
数据库问题 - BERT编码 - 向量A ----
                              两者对比余弦相似度 = 语义相似度 -> 得到相似度分数
用户问题 - BERT编码 - 向量B ----
```
