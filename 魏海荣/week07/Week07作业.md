# 作业1: 阅读 02-joint-bert-training-only 代码，并回答以下问题：
- bert 文本分类和 实体识别有什么关系，分别使用什么loss？
    
    - BERT 文本分类主要是做的意图识别，从宏观的去理解一句话的目的，实体识别是提取关键的信息片段，侧重于微观细节，bert 文本分类和 实体识别其实都是对语义的理解，实体信息有助于推断意图，意图也有助于识别实体，俩个任务起到相互促进的效果：实体识别帮助模型关注关键词语，意图识别帮助模型理解上下文。

    - BERT 文本分类 在代码中使用的是交叉熵损失函数。
    - 实体识别也同样使用了交叉熵损失函数，但加了mask过滤padding

- 多任务训练  loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？

    - loss尺度不一样，如果意图loss很小了，但是实体loss还是很大，会导致模型会优先优化实体识别，忽略意图识别。
    - 收敛速度不一致：意图分类（全局任务）通常收敛更快；实体识别（序列任务）需要更多训练，这样会导致一个任务过拟合，另一个欠拟合。
    - 梯度冲突：如果两个任务的梯度方向相反，会相互抵消，如果梯度大小悬殊，小梯度任务会被淹没。


作业2: 基础 02-joint-bert-training-only  中的数据集，希望你自己写一个提示词能完成任务（信息解析的智能对话系统）

```
你是一个智能语音助手的自然语言理解（NLU）模块。你的任务是将用户的输入文本解析成结构化的指令数据。你需要识别出用户的领域(Domain)、意图(Intent)，并抽取所有相关的关键信息槽位(Slots)。


用户提问：{#Query#}

输出格式：
你的输出必须是一个严格的JSON对象，包含以下四个字段：

text:  (字符串) 原始的用户输入文本
domain: (字符串) 识别出的领域，例如app、bus、cinemas、contacts、cookbook、epg、flight、health、map、message、music、news、novel、poetry、telephone、train、translation、tvchannel、video等。
intent: (字符串) 识别出的意图，例如DIAL、DOWNLOA、LAUNCH、PLAY、POSITION 、QUERY 、ROUTE 、SEND、SENDCONTACTS、TRANSLATION 等。
slots: (对象) 一个键值对对象，用于存放抽取出的关键信息。如果没有任何槽位，则为一个空对象 {}

例1：
Input： 教我做馒头。

Output:
{
    "text": "教我做馒头。",
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {
      "dishName": "馒头"
    }
}

例2：
Input： 张绍刚的综艺节目

Output:
{
    "text": "张绍刚的综艺节目",
    "domain": "video",
    "intent": "QUERY",
    "slots": {
      "artist": "张绍刚",
      "category": "节目",
      "tag": "综艺"
    }
}
```

