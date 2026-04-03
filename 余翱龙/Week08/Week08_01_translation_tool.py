from pydantic import BaseModel, Field
from typing_extensions import Literal
import json

import openai

client = openai.OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationRequest(BaseModel):
    """语言翻译请求"""
    source_language: Literal["英文", "中文", "日文", "法文", "德文", "西班牙文", "韩文", "其他"] = Field(description="原始语种")
    target_language: Literal["英文", "中文", "日文", "法文", "德文", "西班牙文", "韩文", "其他"] = Field(description="目标语种")
    text_to_translate: str = Field(description="待翻译的文本")

class TranslationResult(BaseModel):
    """语言翻译结果"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    original_text: str = Field(description="待翻译的文本")
    translated_text: str = Field(description="翻译结果")

class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt: str) -> TranslationResult:
        # 定义翻译工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "translate_text",
                    "description": "根据用户需求将文本从一种语言翻译为另一种语言",
                    "parameters": {
                        "type": "object",
                        "properties": TranslationRequest.model_json_schema()['properties'],
                        "required": TranslationRequest.model_json_schema()['required'],
                    },
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的翻译助手。请准确识别用户想要翻译的语言和内容，提取出原始语种、目标语种和待翻译的文本。"
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            # 提取翻译请求参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            request = TranslationRequest.model_validate_json(arguments)

            # 执行翻译
            translated_text = self._perform_translation(request)

            # 返回完整的翻译结果
            return TranslationResult(
                source_language=request.source_language,
                target_language=request.target_language,
                original_text=request.text_to_translate,
                translated_text=translated_text
            )
        except Exception as e:
            print(f'翻译出错: {e}')
            print('原始响应:', response.choices[0].message)
            return None

    def _perform_translation(self, request: TranslationRequest) -> str:
        # 使用大模型执行实际翻译
        messages = [
            {
                "role": "system",
                "content": f"你是一个专业的翻译专家。请将以下文本从{request.source_language}翻译为{request.target_language}。只返回翻译结果，不要添加任何解释。"
            },
            {
                "role": "user",
                "content": request.text_to_translate
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )

        return response.choices[0].message.content.strip()

# 使用示例
if __name__ == "__main__":
    agent = TranslationAgent()

    # 测试用例
    test_cases = [
        "帮我将good！翻译为中文",
        "请把'Hello, world!'翻译成英文",
        "我想了解'こんにちは'的中文意思",
        "将'Bonjour'翻译成英文"
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case}")
        result = agent.call(test_case)
        if result:
            print(f"原始语种: {result.source_language}")
            print(f"目标语种: {result.target_language}")
            print(f"待翻译文本: {result.original_text}")
            print(f"翻译结果: {result.translated_text}")
        else:
            print("翻译失败")

        # 根据示例格式输出
        if i == 1:  # 第一个测试用例按照示例格式输出
            print(f"\n示例格式输出:")
            print(f"({result.source_language}、{result.target_language}、{result.original_text}) {result.translated_text}")