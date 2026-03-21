from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-9bf45d961ac64f75a3b6a64c7fd08817", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现文本翻译
指定写的tool的格式
"""
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

class translate_text(BaseModel):
    """将一段文本从源语言翻译为目标语言，并提取语言信息和翻译结果。"""
    source_language: str = Field(description="原始语种，例如：'英文'")
    target_language: str = Field(description="目标语种，例如：'中文'")
    sentence: str = Field(description="翻译后的文本。必须根据原始文本和语言信息生成正确的翻译，不能直接复制用户输入。")
result = ExtractionAgent(model_name = "qwen-plus").call('请将 "good morning" 翻译成中文，并告诉我源语言和目标语言是什么。', translate_text)
print(result)

# class Text(BaseModel):
#     """抽取句子的摘要"""
#     abstract: str = Field(description="摘要结果")
# result = ExtractionAgent(model_name = "qwen-plus").call("20年来，中国探月工程从无到有、从小到大、从弱到强。党的十八大后，一个个探月工程任务连续成功，不断刷新世界月球探测史的中国纪录嫦娥三号实现我国探测器首次地外天体软着陆和巡视探测，总书记肯定“在人类攀登科技高峰征程中刷新了中国高度”；", Text)
# print(result)
#
#
# class Text(BaseModel):
#     """文本问答内容解析"""
#     search: bool = Field(description="是否需要搜索")
#     keywords: List[str] = Field(description="待选关键词")
#     intent: Literal["查询客服问题", "查询产品问题", "查询系统问题", "其他"] = Field(description="意图")
# result = ExtractionAgent(model_name = "qwen-plus").call('汽车发动和轮胎出故障了，如何处理？', Text)
# print(result)
#
#
# class Text(BaseModel):
#     """文本问答内容解析"""
#     time: List[str] = Field(description="时间")
#     particate: List[str] = Field(description="选手")
#     competition: List[str] = Field(description="赛事名称")
# result = ExtractionAgent(model_name = "qwen-plus").call('2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办。', Text)
# print(result)
