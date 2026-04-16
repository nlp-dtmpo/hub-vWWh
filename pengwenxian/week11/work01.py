from typing import Any


import os
import uuid
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-89beb5cc538544fc9ab0ad56bcf6f044"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

entity_recognize_agent = Agent(
    name='entity_recognize_agent',
    instructions='请识别用户输入的文本中的实体',
    model='qwen3.5-27b'
)

sentiment_analysis_agent = Agent(
    name='sentiment_analysis_agent',
    instructions='请对用户输入的文本进行情感分类',
    model='qwen3.5-27b'
)

triage_agent = Agent(
    name='triage_agent',
    instructions='请对用户输入的文本进行分析，判断其需求是实体识别还是情感分类，分发到不同模型来解析',
    model='qwen3.5-27b',
    handoffs=[entity_recognize_agent, sentiment_analysis_agent]
)


async def main():
    msg = input('你好，我是一个实体识别/情感分类的智能体，可以帮你解答这两个问题')
    inputs: list[TResponseInputItem] = [{'role': 'user', 'content': msg}]
    agent = triage_agent
    while True:
        with trace('router_work', group_id = uuid.uuid4()):
            result = Runner.run_streamed(
                agent,
                input=inputs
            )

            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end='', flush=True)
                if isinstance(data, ResponseContentPartDoneEvent):
                    print('\n')

        inputs = result.to_input_list()

        user_input = input('请输入你要询问的问题：')
        inputs.append({
            'role': 'user',
            'content': user_input
        })
        agent=result.current_agent

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
            

