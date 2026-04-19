import os
from typing import List

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class TextOutput(BaseModel):
    """判断用户的请求是否是对文本进行情感分类 或者 进行实体识别"""
    is_text: bool

class EntityItem(BaseModel):
    name: str
    type: str

class EntityOutput(BaseModel):
    entities: List[EntityItem]

guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-flash",
    instructions="判断用户的请求是否是对文本进行情感分类 或者 进行实体识别。如果是，'is_text'应该设置为 True, json 返回",
    output_type=TextOutput,
)

# 情感分类代理
emotion_agent = Agent(
    name="Emotion Agent",
    model="qwen-flash",
    handoff_description="负责对文本进行情感分类。",
    instructions="你是一名专业的情感分类专家，请你为用户输入的文本进行情感分类，判断用户文本其中蕴含的情感。如果有多种情感，只输出最符合的一个即可。",
)

# 实体识别代理
ocr_agent = Agent(
    name="OCR Agent",
    model="qwen-flash",
    handoff_description="负责对文本进行实体识别。",
    instructions="你是一名专业的实体识别专家，负责从文本中提取所有有意义的实体（包括地点、自然物、抽象概念等）。并以json格式返回。",
    output_type=EntityOutput,
)

async def text_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为情感分类 或者 实体识别
    如果不是，则触发阻断（tripwire）
    """
    print(f"\n[Guardrail Check] 正在检查输入：{input_data}...")

    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    final_output = result.final_output_as(TextOutput)

    tripwire_triggered = not final_output.is_text

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )

triage_agent = Agent(
    name="Triage Agent",
    model="qwen-flash",
    instructions="你的任务是根据用户输入的请求，判断应该将请求分发给 'emotion agent' 还是 'ocr agent'",
    handoffs=[emotion_agent, ocr_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=text_guardrail),
    ]
)

async def main():
    try:
        query = "帮我分析下'我谢谢你啊'这句话蕴含的情感"
        print(f"用户提问：{query}")
        result = await Runner.run(triage_agent, query)
        print("\n流程通过，最终输出：")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n 错误！ 守卫阻断触发：", e)

    print("\n" + "="*50)
    print("="*50)

    try:
        query = "帮我对'总有一条蜿蜒在童话镇里梦幻的河，分割了理想，分割现实，又到前方的山路汇合'进行实体识别"
        print(f"用户提问：{query}")
        result = await Runner.run(triage_agent, query)
        print("\n流程通过，最终输出：")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n 错误！ 守卫阻断触发：", e)

    print("\n" + "=" * 50)
    print("=" * 50)

    try:
        query = "《童话镇》这首歌的歌词是什么？"
        print(f"用户提问：{query}")
        result = await Runner.run(triage_agent, query)
        print("\n流程通过，最终输出：")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n 错误！ 守卫阻断触发：", e)

if __name__ == "__main__":
    asyncio.run(main())
