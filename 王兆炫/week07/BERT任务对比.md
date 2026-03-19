本文档对比BERT分别用于文本分类及实体识别,得到二者的区别以及联系 

> 命名实体识别（Named Entity Recognition，简称 NER）是自然语言处理（NLP）中的一项基础任务，它的目标是识别文本中具有特定意义的实体，并将其分类到预定义的类别中。

### 共同点

+ 都以 BERT/Transformer 提供的上下文化 token 表示为基础。

+ 都进行微调（fine-tuning）：在预训练模型上接一个任务特定的头（head）并联合训练。

+ 输入处理相同：分词（WordPiece/Subword）、加上特殊 token（[CLS],[SEP]）、attention mask。

+ 优点与限制相同：对上下文建模强、对长序列受限（max length）、对低资源任务需要数据增强或迁移学习手段。

### 核心区别
1) 输出结构

文本分类：单个向量→一个/多个标签（softmax 或 sigmoid）。

实体识别：每个 token（或每个 word）→ 一个标签序列（常用 BIO/BIES 等编码）或 span 列表。

2) Head 的形式

分类 head：通常在 [CLS] 上接一个全连接层 + softmax/cross-entropy。

序列标注 head：对每个 token 的隐藏向量接线性层 → token-level softmax；常与 CRF 层结合以建模标签间依赖；也有基于 span 的方法（直接预测起始/结束位置与类别）。

3) 损失函数

分类：句级交叉熵或多标签的 binary-cross-entropy。

NER：对每个 token 的交叉熵（或 CRF 的对数似然），实体级别常用严格的 entity-F1 来评估。

4) 标签与数据

文本分类：需要句/文档级标签（相对容易标注）。

NER：需要序列级、带边界的标注（标注成本高，且质量敏感）。

5) Tokenization 与标签对齐

分类：子词对齐问题少（只用 [CLS]）。

NER：子词（WordPiece）拆分会带来标签映射问题：常见策略有只给第一个 subword 标注实体，或把所有 subword 都赋同标签，或在 decode 时合并。

6) 解码与后处理

分类：直接 argmax/threshold。

NER：需要将 token 标签序列转换为实体边界（合并子词、处理 BIO 转换、合并重叠等），span-based 模型还需去重与约束。

7) 评估指标

分类：accuracy、macro/micro F1、precision/recall 等。

NER：常用严格的 entity-level F1（边界和类别都需正确）；token-level F1 也常被报告，但实体级更重要。

8) 计算与内存

NER 对长序列和每个 token 都要计算输出，训练/推理内存占用更高，batch size 需要更小；而短文本分类通常更省资源。
