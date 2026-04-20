import os
import asyncio
from agents.exceptions import InputGuardrailTripwireTriggered
from agents.extensions.visualization import draw_graph
from unittest import expectedFailure
from openai.types.responses.response_function_shell_tool_call_output import Output
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents import set_default_openai_api, set_tracing_disabled
from typing import Any, Optional

os.environ["OPENAI_API_KEY"] = "***"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class ClassificationOutput(BaseModel):
    """用于判断用户请求是否属于情感分类和实体识别的结构"""
    is_classification: bool


guardrail_agent = Agent[Any](

    name="Classification Agent",
    model="qwen-max",
    instructions="""
    您的任务是根据用户的请求，判断是否属于情感分类或实体识别。
    情感分类：判断文本的情感倾向（如积极、消极、中性）
    实体识别：识别文本中的实体（如人物、地点、组织、产品等）
    如果用户的请求属于情感分类或实体识别，请返回 {"is_classification": true}
    否则返回 {"is_classification": false}
    请严格按照JSON格式返回结果，只包含is_classification字段。
    示例：
    输入："今天天气真好，我心情非常愉快！"
    输出：{"is_classification": true}
    输入："苹果公司在2023年发布了iPhone 15系列手机。"
    输出：{"is_classification": true}
    """,
    output_type=ClassificationOutput
)

# 情感分类代理
sentiment_classification_agent = Agent(
    name="sentiment classification agent",
    model="qwen-max",
    instructions="""
    您是一个情感分类专家，负责判断用户输入的情感倾向。
    请将情感分为三类：positive（积极）、negative（消极）或neutral（中性）。
    请提供详细的分析和判断理由。
    """
)

# 实体识别代理
entity_recognition_agent = Agent(
    name="entity recognition agent",
    model="qwen-max",
    handoff_description="负责处理所有实体识别问题的专家代理。",
    instructions="""
    您是一个实体识别专家，负责从用户输入的文本中识别出各种实体。
    请识别以下类型的实体：
    - person: 人物
    - location: 地点
    - organization: 组织
    - time: 时间
    - date: 日期
    - event: 事件
    - product: 产品
    - quantity: 数量
    请列出识别出的所有实体及其类型。
    """
)


async def classification_guardrail(ctx, agent, input_data):
    print(f"[guardrail check] 正在输入：{input_data}...")
    try:
        result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
        final_output = result.final_output_as(ClassificationOutput)
        tripwire_triggered = not final_output.is_classification

        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=tripwire_triggered
            )
        
    except Exception as e:
        print(f"[guardrail check] 错误：{e}")
        # 出错时默认不阻断，避免因格式问题导致正常请求被拒绝
        return GuardrailFunctionOutput(
            output_info=None,
            tripwire_triggered=False
        )
    


triage_agent = Agent[Any](
    name="Triage agent",
    model="qwen-max",
    instructions="您的任务是根据用户的请求，判断是否属于情感分类或实体识别。",
    handoffs=[sentiment_classification_agent, entity_recognition_agent],
    input_guardrails=[InputGuardrail(guardrail_function=classification_guardrail)]
)


async def main():
    print("--- 启动中文代理系统示例 ---")
    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "今天天气真好，我心情非常愉快！"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "2022年世界杯在卡塔尔举行，阿根廷队获得了冠军。"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "一个直角三角形的两条直角边分别为3和4，求斜边的长度。"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)


if __name__ == "__main__":
    asyncio.run(main())
    try:
        draw_graph(triage_agent, filename="classification.png")
    except:
        print("绘制流程图失败,默认跳过")
