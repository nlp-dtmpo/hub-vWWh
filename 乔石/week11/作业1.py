import asyncio
import os

from agents import Agent, Runner

os.environ["OPENAI_API_KEY"] = "sk-***"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

sentiment_classify_agent = Agent(
    name="Sentiment Classify Agent",
    model="qwen3.5-flash",
    handoff_description="负责处理情感分类问题的专家代理",
    instructions="您是情感分类专家。请对该文本进行一个准确的情感分类"
)

NERAgent = Agent(
    name="NERAgent",
    model="qwen3.5-flash",
    handoff_description="负责文本实体识别与提取，处理所有命名实体标注任务代理",
    instructions="""
    你是一个专业的实体识别助手，任务是对输入的文本进行实体识别与标注。
    请识别以下**标准实体类型**：
    - PER：人名（如张三、李白、马斯克）
    - LOC：地名（如北京、上海、纽约、长江）
    - ORG：组织名（公司、政府机构、学校，如阿里巴巴、清华大学、联合国）
    - DATE：日期（如2025年10月1日、昨天、明年）
    - TIME：时间（如上午9点、晚上8:30）
    - MONEY：金额（如100元、50万美元）
    - NUM：数字（如3个、100台）

    输出要求：
    1. 只返回结构化JSON，无多余文字
    2. 格式：{"实体类型": ["实体1", "实体2"], ...}
    3. 无对应实体则返回空列表
    """
)

triage_agent = Agent(
    name="Triage Agent",
    model="qwen3.5-flash",
    instructions="您的任务是根据用户的输入内容，判断应该将请求分派给 'Sentiment Classify Agent' 还是 'NERAgent'。",
    handoffs=[sentiment_classify_agent, NERAgent]
)


async def main():
    query = "帮我买一张从深圳到北京的车票"
    print(f"用户的提问是：{query}")
    result = await Runner.run(triage_agent, query)
    print("\n**✅ 流程通过，最终输出:**")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())