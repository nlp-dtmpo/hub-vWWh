from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel, Field
import openai


class TranslationTask(BaseModel):
    """自动识别翻译任务参数"""

    source_language: str = Field(
        description="原始语种，如 中文/英文/日文；若用户未指定可填 auto"
    )
    target_language: str = Field(
        description="目标语种，如 中文/英文/日文；若用户未指定，默认填 中文"
    )
    text: str = Field(description="待翻译的原文")


class TranslationOutput(BaseModel):
    """翻译结果"""

    source_language: str
    target_language: str
    text: str
    translated_text: str


class ExtractionAgent:
    """
    基于 tools(function call) 的结构化信息抽取器。
    参考 04_Pydantic与Tools.py 的实现风格。
    """

    def __init__(self, client: openai.OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def call(self, user_prompt: str, response_model: type[BaseModel]) -> Optional[BaseModel]:
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],
                    "description": schema.get("description", "结构化抽取"),
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", []),
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个信息抽取助手。"
                    "请把用户输入解析为工具参数。"
                    "如果用户未给出目标语种，默认 target_language=中文；"
                    "如果未给出原始语种，source_language=auto。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception:
            return None


class TranslationAgent:
    """文本翻译智能体：自动抽取翻译任务 + 执行翻译"""

    def __init__(self, client: openai.OpenAI, model_name: str = "qwen-plus"):
        self.client = client
        self.model_name = model_name
        self.extractor = ExtractionAgent(client=client, model_name=model_name)

    def _fallback_extract(self, user_prompt: str) -> TranslationTask:
        """
        当 tool 抽取失败时的兜底：
        - source_language 默认为 auto
        - target_language 默认为 中文
        - text 直接使用用户输入
        """
        return TranslationTask(
            source_language="auto",
            target_language="中文",
            text=user_prompt.strip(),
        )

    def _translate(self, task: TranslationTask) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业翻译助手。"
                    "只输出翻译后的文本，不要解释。"
                    "source_language=auto 时请先自行识别原文语种再翻译。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(task.model_dump(), ensure_ascii=False),
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def run(self, user_prompt: str) -> TranslationOutput:
        task = self.extractor.call(user_prompt=user_prompt, response_model=TranslationTask)
        if task is None:
            task = self._fallback_extract(user_prompt)

        translated_text = self._translate(task)
        return TranslationOutput(
            source_language=task.source_language,
            target_language=task.target_language,
            text=task.text,
            translated_text=translated_text,
        )


def build_client() -> openai.OpenAI:
    """
    从环境变量创建 OpenAI 兼容客户端。
    必填：
    - OPENAI_API_KEY
    可选：
    - OPENAI_BASE_URL (默认 DashScope 兼容地址)
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("请先设置环境变量 OPENAI_API_KEY")

    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()

    return openai.OpenAI(api_key=api_key, base_url=base_url)


if __name__ == "__main__":
    # 示例：
    # python translation_agent.py
    # 然后输入：
    # 1) 把“今天天气不错”翻译成英文
    # 2) Translate this to Chinese: Deep learning changes everything.
    # 3) こんにちは、元気ですか？（默认翻译到中文）
    client = build_client()
    agent = TranslationAgent(client=client, model_name="qwen-plus")

    prompt = input("请输入翻译需求：").strip()
    result = agent.run(prompt)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
