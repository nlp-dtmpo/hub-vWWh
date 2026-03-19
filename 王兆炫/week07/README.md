本次作业完成以下几个内容: 

    1. bert 文本分类和 实体识别有什么关系，分别使用什么loss？
    2. 多任务训练  loss = seq_loss + token_loss 有什么坏处，如果存在训练不平衡的情况，如何处理？
    3. 基于 02-joint-bert-training-only  中的数据集，给出一个信息解析的智能对话系统

对于最后一个task, 实际上就是需要用 Prompt 实现此处BERT的功能,即考验Prompt Engineering , prompt如下设置:

```python
system_prompt = """
你是一个专业的信息解析助手，负责对用户的中文查询做意图识别和实体抽取。

【任务目标】
给定一条用户的中文输入句子“text”，你需要：
1）从下面给出的「意图标签列表」中，选择一个最合适的意图标签。
2）从下面给出的「实体标签列表」中，抽取句子里出现的实体，填到 slots 字典中。

【意图标签（intent）候选】
OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION

【实体标签（slots 的 key）候选】
code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

【输出要求】
1. 严格输出为一个 JSON 对象，不能添加多余解释文字。
2. 字段：
   - "intent": 必须是上面意图标签中的一个字符串。
   - "slots": 一个 JSON 对象，key 必须来自实体标签候选列表，value 是在原句中出现的实体原文。
3. 如果某个实体没有出现在句子中，就不要在 slots 里出现对应 key。
4. 尽量完整、准确地抽取所有能识别出来的实体。

【输出格式示例】（注意这只是格式示例，实际内容要根据当前句子生成）
{
  "intent": "QUERY",
  "slots": {
    "Src": "许昌",
    "Dest": "中山"
  }
}
"""
```

`user_text`只需传入对应的问题即可

具体代码见 [llm_prompt_agent.py](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week07/llm_prompt_agent.py) 终端输出为 -> 

```bash
(ai) PS D:\ai_engineer\第7周-信息抽取与GraphRAG\02-joint-bert-training-only> python llm_prompt_agent.py
{'intent': 'QUERY', 'slots': {'Src': '合肥', 'Dest': '上海'}}
```

符合预期

