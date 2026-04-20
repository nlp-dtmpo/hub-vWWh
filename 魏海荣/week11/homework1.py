import os
os.environ["OPENAI_API_KEY"] = "sk-XXX"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from typing import Optional
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class HomeworkOutput(BaseModel):
    """用于判断用户请求是否属于情感分类或实体识别的结构"""
    is_ner_or_sa: bool


guardrail_agent_prompt = """你是一个严格的分类器，只判断用户输入是否属于以下两种任务：
    1. 情感分析（Sentiment Analysis）
    2. 实体识别（Named Entity Recognition）
    【允许的输入（属于任务）】:
        - 这家餐厅的服务态度很差
        - 这部电影我非常喜欢
        - 张三在2025年加入了北京字节跳动
        - 请提取这句话里的实体
        - 分析这句话的情感
    【不允许的输入（不属于任务）】
        - 今天天气怎么样
        - 帮我写一首诗
        - 1+1等于几
        - 推荐电影
        - 你是谁
    如果属于任务，返回 is_ner_or_sa=True
    否则返回 is_ner_or_sa=False
    必须严格输出JSON！
    """

# 守卫检查代理 - 》 本质也是通过大模型调用完成的
guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions= guardrail_agent_prompt.strip(), # 对话中的 开头 system message
    output_type=HomeworkOutput, 
)

sa_agent = Agent(
    name="Sentiment Analysis Agent",
    model="qwen-max",
    handoff_description="""负责处理所有情感分类问题的专家代理。你是情感分析专家。""",
    instructions="""您是专业的情感分析专家。请分析用户输入的文本，并提供情感分类结果（如积极、消极、中性）以及简要的分析理由。
        输出格式：
        情感：积极/消极/中性\n
        理由：xxx""".strip(),
)

ner_agent = Agent(
    name="NER Agent",
    model="qwen-max",
    handoff_description="负责处理所有实体识别问题的专家代理。",
    instructions="您是专业的实体识别专家。请从用户输入的文本中识别出所有的实体，并分类（如人名、地名、组织等）。"
)

async def homework_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为情感分类或实体识别相关问题。
    如果不是 ('is_ner_or_sa' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")
    
    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    
    # 解析输出
    final_output = result.final_output_as(HomeworkOutput)
    
    tripwire_triggered = not final_output.is_ner_or_sa
        
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered # 是否触发了 tripwire。如果触发，代理的执行将被停止
    )


# 定义一个主代理，使用守卫来检查输入，并根据检查结果将请求转发给不同的专家代理
triage_agent = Agent(
    name="Triage Agent",
    model="qwen-max",
    instructions="""根据用户输入的内容，将请求路由到适当的专家代理（情感分析或实体识别）。
                    如果用户需要情感分析 → 交给 Sentiment Analysis Agent
                    如果用户需要提取实体 → 交给 NER Agent""".strip(),
    handoffs=[sa_agent, ner_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    print("--- 启动中文代理 ---")
    
    print("\n" + "="*50)
    print("="*50)
    try:
        query = "今天心情特别好， everything goes well！"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query) # 异步运行
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)
        
    
    print("\n" + "="*50)
    print("="*50)
    try:
        query = "苹果公司将于 2026 年 9 月发布新一代 iPhone。"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)
        
    
    print("\n" + "="*50)
    print("="*50)
    try:
        query = "你觉得明天深圳的天气怎么样？"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output) # 这行应该不会被执行
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)


if __name__ == "__main__":
    asyncio.run(main())