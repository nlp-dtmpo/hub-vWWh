

import asyncio
import uuid
import os
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

os.environ["OPENAI_API_KEY"] = "sk-cbf8dce472b74ddfbf3210f7d52dd758"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

Emo_agent = Agent(
    name="Emo_agent",
    model="qwen-max",
    instructions="你是一个情感分类专家，你只能回答情感分类问题。不能说其他内容,不能引导用户，只能回答情感分类问题",
)

Entity_agent = Agent(
    name="Entity_agent",
    model="qwen-max",
    instructions="你是一个实体抽取专家，当用户需要进行实体识别、提取文本中的命名实体（如地点、人名、时间、航班号等）时，使用此工具。不能说其他的内容,只能回答用户的实体识别问题",
)

triage_agent= Agent(
    name="Triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[Emo_agent, Entity_agent],
)


async def main():
    conversation_id = str(uuid.uuid4().hex[:16])
    # try:
    #     draw_graph(triage_agent, filename="路由Handoffs")
    # except:
    #     print("绘制agent失败，默认跳过。。。")

    print("欢迎来到路由Handoffs")
    msg = "你好，请告我你的需要"
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    while True:
        with trace("Routing example",group_id=conversation_id):
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
                    print(f"\n")

        inputs = result.to_input_list()
        print("\n")
        print(result.current_agent.name)
        print("-" * 50)
        user_msg = input("你好，请输入你的问题(exit退出)：")
        if user_msg.lower() == 'exit':
            print('程序退出')
            break
        inputs.append({"content": user_msg, "role": "user"})
        agent = triage_agent

if __name__ == "__main__":
    asyncio.run(main())
