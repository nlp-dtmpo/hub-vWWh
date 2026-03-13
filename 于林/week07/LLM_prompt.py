# 基于 openai 客户端的使用
import openai

client = openai.OpenAI(
    api_key = "sk-pL9wStGbuCTF5uV9971eE079089f4c3e8fD0A52eAdCe08E0",
    base_url = "https://openkey.cloud/v1"
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system",
         "content": """你是一个信息抽取的专家，请对文本进行意图识别、实体识别
        领域类别有：music app radio lottery stock novel weather match map website news message contacts translation tvchannel cinemas cookbook joke riddle telephone video train poetry flight epg health email bus story
        意图标签有：OPEN SEARCH REPLAY_ALL NUMBER_QUERY DIAL CLOSEPRICE_QUERY SEND LAUNCH PLAY REPLY RISERATE_QUERY DOWNLOAD QUERY LOOK_BACK CREATE FORWARD DATE_QUERY SENDCONTACTS DEFAULT TRANSLATION VIEW NaN ROUTE POSITION
        实体标签有：code Src startDate_dateOrig film endLoc_city artistRole location_country location_area author startLoc_city season dishNamet media datetime_date episode teleOperator questionWord receiver ingredient name startDate_time startDate_date location_province endLoc_poi artist dynasty area location_poi relIssue Dest content keyword target startLoc_area tvchannel type song queryField awayName headNum homeName decade payment popularity tag startLoc_poi date startLoc_province endLoc_province location_city absIssue utensil scoreDescr dishName endLoc_area resolution yesterday timeDescr category subfocus theatre datetime_time

        输出为json格式，模板为：
        {
        "text": "",
        "domain": "",
        "intent": "",
        "slots": {
          "实体识别结果": "",
          "实体识别标签": ""
            }
        } 其中 domain为领域类别 intent为意图标签 slot为实体识别结果和标签"""},
         {"role": "user", "content":"帮忙打开一下酷狗音乐播放音乐行不"}

    ]
)
# 打印结果
print(completion.choices)


