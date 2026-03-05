### 读取数据集

一个是json格式读取,另一个是pdfplumber文件识别pdf结果读取

输出结果
```bash
questions.json
{'question': '“前排座椅通风”的相关内容在第几页？', 'answer': '', 'reference': ''}


pages:  354
装载货物
前排储物空间 副仪表台
■ 遮阳板上的卡片/票据夹。 01无线充电板/置物板
■ 手套箱。
02杯托
■ 车门存储空间。
■ 眼镜盒。 03储物箱
■ 前副仪表台。
■ 前排中央扶手储物箱。 注意！
■ 切勿强行将尺寸不合适的容器卡入杯托。
24
```

---

### TFIDF检索

> task : 查找对应关键词出现在pdf的第几页/与pdf第几页最相关
利用TF-IDF算法, 生成了 submit_tfidf_retrieval_top1.json（只给一个最准的答案）和 top10.json（给出前 10 个可能的页面）。

---

### BM25检索

直接理解为TF-IDF的优化版算法即可,同样得到两个json文件 : submit_bm25_retrieval_top1.json 和 top10.json。

---

### BERT检索

比较了两个目前中文领域最顶尖的 Embedding 模型：

+ BGE (智源)：由北京智源人工智能研究院开发。它的特点是“稳且准”，非常适合处理你这种技术手册。

+ Jina (Jina AI)：特点是支持更长的文本窗口（v2 版本甚至是 8k 长度）。

进阶点：不再使用 jieba 分词！因为深度学习模型（Transformer 架构）能自动理解上下文，不需要手动切词。它直接计算“语义距离”。

结果输出：生成了submit_bge_retrieval_top1.json , submit_jina_retrieval_top1.json。

---

### BERT-Segment检索

核心进阶：引入了 split_text_fixed_size(text, 40)。

把每一页强行切成每 40 个字一段的 Chunk , 使得检索更精细 , 通常结果更准确

---

### 多路召回

> 单纯用向量检索（BGE）有时会丢失精确的关键词（如某个零件型号），而单纯用 BM25 无法理解意思。

核心算法：RRF (Reciprocal Rank Fusion)：

  + 代码里的 1 / (idx + k) 就是 RRF 算法。

  + 它不看具体的原始得分，只看排名。如果一个页面在 BGE 和 BM25 里都排在前几名，它的最终得分会极高。

结果：取长补短，生成的 submit_fusion_bge+bm25_retrieval.json 效果通常优于任何单一算法。

---

### 重排序 (Re-ranking)

操作：

+ 先用多路召回筛选出前几个最可能的页面。

+ 调用 bge-reranker-base 模型。这是一个 Cross-Encoder（交叉编码器） 模型。

+ 深度对比：它会把“问题”和“文档内容”交叉编码输入给模型。这比向量检索更耗时，但精度极高。适用于更小规模的精细化排序

---

### RAG 问答

形成完整的RAG链路

+ Retrieve（检索）：用多路召回找出最相关的 PDF 页面。

+ Augmented（增强）：把找到的 PDF 原文放进一个精心设计的 Prompt（提示词） 里。

+ Generate（生成）：调用外部大模型（如智谱 GLM 或 GPT）。






