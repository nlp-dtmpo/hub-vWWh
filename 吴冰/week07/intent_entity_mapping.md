作业1: 阅读 02-joint-bert-training-only 代码，并回答以下问题：

◦ bert 文本分类和 实体识别有什么关系，分别使用什么loss？

◦ 多任务训练 loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？

作业2: 基础 02-joint-bert-training-only 中的数据集，希望你自己写一个提示词能完成任务（信息解析的智能对话系统）


回答：

## 作业1：

### 1. BERT文本分类和实体识别的关系及损失函数

在联合模型中，BERT文本分类（意图识别）和实体识别（槽位填充）共享同一个BERT编码层，但使用不同的输出层进行预测：

**关系**：

两者共享BERT的上下文编码表示（如`pooler_output`用于意图分类，`token_output`用于实体识别），这允许模型同时学习句子级和token级语义特征。

意图分类基于整个句子的表示（CLS token的输出），而实体识别基于每个token的表示，形成多任务学习框架。

**损失函数**：

**意图分类损失（seq_loss）**：使用交叉熵损失（`nn.CrossEntropyLoss`），计算意图标签的预测误差。输出是序列级标签（如"QUERY"、"BOOK"），损失基于`seq_output`和`seq_label_ids`。

**实体识别损失（token_loss）**：同样使用交叉熵损失，但仅对非padding的token计算（通过`active_loss`掩码过滤）。输出是token级BIO标签（如"B-city"、"I-city"），损失基于`active_logits`和`active_labels`。

**输出数据类型形成原因**：

意图分类输出为标量标签（如整数索引），因为意图是句子级单一类别，通过`argmax`从`seq_output`获取。

实体识别输出为序列标签（列表），因为每个token需独立预测，且BIO格式需处理实体边界（如连续"B-"和"I-"标签合并为实体）。

### 2. 多任务训练 `loss = seq_loss + token_loss`的坏处及不平衡处理

##### **多任务训练** `**loss = seq_loss + token_loss**`**的坏处：**

1. **尺度不一致问题**：

意图分类损失（seq_loss）和实体识别损失（token_loss）可能处于不同数量级

实体识别通常损失更大（因token数量多），会主导梯度更新方向

导致意图分类任务训练不足

1. **收敛速度差异**：

意图分类收敛更快（类别少，任务相对简单）

实体识别收敛较慢（标签稀疏，大量"O"标签）

模型过早偏向意图分类任务

1. **梯度冲突**：

两个任务的优化方向可能冲突

简单相加无法协调梯度方向

##### 常见的不平衡情况及处理方案：

1. **损失尺度不平衡**：

- **现象**：`token_loss`>> `seq_loss`
- **解决方案**：

引入可学习权重：`loss = α * seq_loss + β * token_loss`

动态调整α和β（如GradNorm算法）

根据验证集性能自动调整权重

1. **收敛速度不平衡**：

- **现象**：意图分类准确率已达90%，实体识别F1仅70%
- **解决方案**：

课程学习：先训练简单任务（意图分类），再逐步加入困难任务

冻结共享层：先固定BERT参数训练任务头，再联合微调

1. **数据分布不平衡**：

- **现象**：某些意图/实体类别样本极少
- **解决方案**：

重采样：过采样稀有类别

损失加权：为稀有类别分配更高权重

1. **梯度方向冲突**：

- **现象**：两个任务的梯度余弦相似度为负值
- **解决方案**：

PCGrad：投影冲突梯度

GradVac：梯度方差衰减控制

##### 检索内容中的处理局限：

当前实现仅简单相加损失（`loss = seq_loss + token_loss`），未引入任何不平衡处理机制。这可能导致：

实体识别任务主导训练过程

意图分类性能未充分优化

稀有实体类别识别效果差


### 作业2：

#### 信息解析提示词设计

你是一个专业信息抽取专家，请对下面的文本抽取其领域类别、意图类别和实体标签：

- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story（从intents.txt获取）
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION（从intents.txt获取）
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time（从slots.txt获取）

最终输出格式填充为JSON：

json

{

"domain": "领域标签",

"intent": "意图标签",

"slots": {

"实体类型": "实体原始值"

}
