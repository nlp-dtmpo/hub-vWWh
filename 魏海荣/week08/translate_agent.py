from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class ExtractionAgent:
    """一个简单的文本解析工具，使用指定的模型进行信息提取"""
    def __init__(self, model_name:str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        # 构建消息列表，包含用户输入的提示
        messages = [{
            "role": "user",
            "content": user_prompt
        }]

        # 定义工具列表，包含一个函数工具，描述了如何提取信息
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
            # print('RAW ARGUMENTS', arguments)
            
            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None
        
class Translation(BaseModel):
    """翻译智能体实体：自动提取 原始语种、目标语种、待翻译文本"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text_to_translate: str = Field(description="待翻译文本")

class TranslationAgent:
    """一个简单的翻译智能体，使用指定的模型进行翻译"""
    def __init__(self, model_name:str):
        self.model_name = model_name

    def translate(self, source_language, target_language, text_to_translate):

        prompt = f"""你是专业翻译助手，请严格按照要求翻译：
                        请把 {source_language} 翻译为 {target_language}。 \n
                        待翻译文本为：{text_to_translate} \n 
                        只输出翻译结果，不要多余解释。""".strip()

        # 构建消息列表，包含用户输入的提示
        messages = [{
            "role": "user",
            "content": prompt
        }]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        try:
            # 提取翻译结果
            translation_result = response.choices[0].message.content
            return translation_result
        except:
            print('ERROR', response.choices[0].message)
            return None

if __name__ == "__main__":

    query = "请帮我把I have around 1 million tokens worth of stories from a single author all set in the same universe but across different settings.翻译成中文。"
    # query = '帮我将good！翻译为中文'

    extraction = ExtractionAgent(model_name = "qwen-plus").call(query, Translation)
    print("提取结果: " + str(extraction))


    translation_result = TranslationAgent(model_name="qwen-plus").translate(
        source_language=extraction.source_language,
        target_language=extraction.target_language, 
        text_to_translate=extraction.text_to_translate
    )
    print(translation_result)