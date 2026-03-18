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

class TranslationTask(BaseModel):
    """识别并提取翻译任务信息"""
    source_language: str = Field(description="原始语种（自动识别）")
    target_language: str = Field(description="目标语种（要翻译成的语言）")
    text_to_translate: str = Field(description="待翻译的文本内容")
    auto_detect: bool = Field(description="是否需要自动识别原始语种", default=True)

# 测试1: 识别翻译任务 - "good！" 翻译成中文
result = ExtractionAgent(model_name="qwen-plus").call(
    '帮我将good！翻译为中文',
    TranslationTask
)
print("=== 翻译任务识别 ===")
print(f"原始语种: {result.source_language}")
print(f"目标语种: {result.target_language}")
print(f"待翻译文本: {result.text_to_translate}")
print(f"自动识别: {result.auto_detect}")
print()

# 测试2: 更复杂的翻译请求
result = ExtractionAgent(model_name="qwen-plus").call(
    '请把"Hello, how are you?"这句话翻译成法语',
    TranslationTask
)
print("=== 英译法任务 ===")
print(f"原始语种: {result.source_language}")
print(f"目标语种: {result.target_language}")
print(f"待翻译文本: {result.text_to_translate}")
print()