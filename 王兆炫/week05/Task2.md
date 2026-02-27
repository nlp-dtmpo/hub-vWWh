## Q2 : 何使用BERT编码 , 以及如何计算相似度

### 基座模型

> 推荐系统和信息检索中 , 一个常见的架构范式是采用双塔模型(Twin-tower)

所以此处选用的微调方式是 : Sentence-BERT（SBERT）--Reimers & Gurevych 提出的、用于产出高质量句向量的具体方法/实现。

基座模型选用bert-base-chinese , 架构模式为SBERT

---

### 训练与微调策略
以下微调策略基于week03提到的论文

论文指出，在特定领域数据上继续进行 Masked Language Model (MLM) 训练，可以显著提升效果。

#### 数据来源：
1. 导出 faq_knowledge 表中的 standard_questions 和 ans_txt。
2. 导出 faq_similar_questions 中的 question_txt。
3. (如果有) 历史人工客服的聊天记录。

#### 操作步骤：
1. 构建无监督语料库（纯文本）。
2. 加载 bert-base-chinese 权重。
3. 开启 MLM (Masked Language Model) 任务进行训练。
> 目的：让 BERT 理解公司的业务术语（例如理解“开票”和“发票”在业务里的上下文关联），而不是只懂通用中文。

#### 数据构造 (Triplet Data)：
我们需要构造三元组 (Anchor, Positive, Negative)：
+ Anchor (锚点)：用户真实提问 或 相似问法。
+ Positive (正例)：对应的标准问法 (关联 faq_knowledge.id)。
+ Negative (负例)：其他类目的问题 (Hard Negative 效果更好)。

#### 应用论文中的三大优化策略：
1. 长文本处理 (Dealing with long texts)：
 + 策略：论文提出 head+tail 截断法（保留开头 128 token + 结尾 382 token）优于单纯截断。
 + 应用：虽然 FAQ 问题通常较短，但如果我们在训练中将“答案文本 (ans_txt)”作为辅助特征输入 BERT，必须使用此策略截断长答案，保留首尾关键信息。
2. 分层学习率衰减 (Layer-wise Decreasing Layer Rate)：
论文核心发现：BERT 底层捕捉语法，顶层捕捉语义。底层参数不需要剧烈变化。超参数保留与论文中推荐相同
+ 衰减因子 $\varepsilon=0.95$
+ 学习率 $2e-5$

3. 学习率调度 (Slanted Triangular Learning Rates)：
+ 策略：Warmup（预热）+ Linear Decay（线性衰减）。
+ 设置：Warmup 步数占总步数的 10%，快速将学习率升至峰值，然后缓慢下降。这能有效避免灾难性遗忘 (Catastrophic Forgetting)。


---

### 文本编码与相似度计算

#### 文本编码
1. Tokenizer：使用 bert-base-chinese 的 tokenizer 将文本转为 Input IDs。
2. Model Inference：输入模型，获取最后一层隐藏状态 (Last Hidden State)。
3. Pooling ：
 + 虽然论文在分类任务中使用了 [CLS] token，但在语义相似度（Embedding）任务中，业界公认 Mean Pooling (对所有 Token 向量求平均) 效果更稳健。
 + 操作：取最后一层除 padding 外所有 token 向量的平均值。

#### 相似度计算
1. 归一化 (Normalization)：
  + 归一化后 , 两向量的余弦相似度就等于两向量点积
  + 对生成的向量进行 L2 归一化。

2. 检索 :
   + 设用户向量为 $V_u$ , 候选向量为 $V_q$
   + Score= $V_u\cdot V_q$
  

---

### 代码实现demo

见[faq_demo.py](faq_demo.py)

---

### 结合数据库的运行逻辑

1. 数据变更同步：
  + 当 MySQL 中 environment='DRAFT' 的数据变更。
  + 后端提取文本 -> 调用微调后的 BERT (执行 forward 函数) -> 得到 768维向量。
  + 存入向量数据库：{ "vector": [...], "environment": "DRAFT", "relation_id": 101 }。

2. 用户提问检索:
   + 用户输入 -> 调用BERT -> 得到向量 $V_{user}$
   + 向量库检索
   ```python
     # 必须带上 Task 1 强调的环境隔离
    milvus.search(
    data=[V_user], 
    expr="environment == 'ONLINE'", 
    limit=1
    )
   ```
   + 计算得到score

3. 结果判定(Task1流程)
   + 根据score的值得出置信度 , 做出相应的回答/调用LLM


