import openai 
import json

client = openai.OpenAI(
    api_key="sk-bc3f2e9c549f422d93f28d6eb7005da3",
    base_url="https://api.deepseek.com",
)

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

def parse_text(text: str):
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    content = completion.choices[0].message.content
    # content 应该是 JSON 字符串，按需要做 json.loads
    result = json.loads(content)
    return result

if __name__ == "__main__":
    text = "从合肥到上海可以到哪坐车？"
    result = parse_text(text)

    print(result)
