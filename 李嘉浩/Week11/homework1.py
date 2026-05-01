import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-bfc69abc05ae4a91bd6e172fddaa9a35"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 意图识别 -》 路由
# 用户提问 -》 类型1  类型2  类型3

emotion_agent = Agent(
    name="intention_agent",
    model="qwen-max",
    instructions="你是一个文本情感分类专家。根据用户输入的文字，判别输入的文本是正向还是负向",
)

entity_agent = Agent(
    name="intention_agent",
    model="qwen-max",
    instructions="你是一个文本实体识别专家。根据用户输入的文字，判别出文字中的实体，例如人名、地名",
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[emotion_agent, entity_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    try:
        draw_graph(triage_agent, filename="路由Handoffs")
    except:
        print("绘制agent失败，默认跳过。。。")
    
    msg = input("你好，我可以帮你进行情感分类与实体识别，你还有什么问题？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent#如果我保证后面的问题是同一领域的问题，agent = result.current_agent 可以让LLM消耗的token减少，因为减少了一次路由,直接变为当前的agent来回答


if __name__ == "__main__":
    asyncio.run(main())