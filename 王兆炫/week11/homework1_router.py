import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from openai import AsyncOpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxx"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL")
MODEL = os.environ.get("OPENAI_MODEL", "qwen-flash")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else AsyncOpenAI(api_key=API_KEY)

sentiment_agent = Agent(
    name="SentimentAgent",
    instructions=(
        "你是情感分类专家。"
        "只做文本情感分类，标签仅可为：正向/负向/中性。"
        "输出格式：情感标签、置信度(0-1)、简短理由。"
    ),
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    model_settings=ModelSettings(parallel_tool_calls=False),
)

ner_agent = Agent(
    name="NerAgent",
    instructions=(
        "你是实体识别专家。"
        "识别文本中的实体，类别包括：人名、组织、地点、时间、金额、产品、事件。"
        "输出严格 JSON 数组，每项包含 entity、type、evidence。"
    ),
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    model_settings=ModelSettings(parallel_tool_calls=False),
)

triage_agent = Agent(
    name="TriageAgent",
    instructions=(
        "你是路由主agent。"
        "如果用户要求情绪/情感判断，handoff 给 SentimentAgent；"
        "如果用户要求实体抽取/命名实体识别，handoff 给 NerAgent；"
        "若不明确，先反问用户希望做'情感分类'还是'实体识别'。"
    ),
    handoffs=[sentiment_agent, ner_agent],
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    model_settings=ModelSettings(parallel_tool_calls=False),
)

async def main():
    print("输入 'exit' 退出")
    while True:
        text = input("\n用户输入: ").strip()
        if text.lower() == "exit":
            break
        result = await Runner.run(triage_agent, input=text)
        print("\nAgent输出:\n", result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
