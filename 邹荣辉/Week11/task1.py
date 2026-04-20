import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-bc1ff8425426455688f964cba80e8b06"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


classification_agent = Agent(
    name="classification_agent",
    model="qwen-max",
    instructions="你是小王，擅长文本情感分类。输出格式：首先说'我是小王，情感分类结果：XX'，其中XX是正面、负面或中性，然后再给出简短分析。",
)

language_agent = Agent(
    name="language_agent",
    model="qwen-max",
    instructions="你是小李，擅长文本翻译。请提取用户输入中的待翻译文本，直接给出翻译结果。回答时先说明你是谁，然后给出翻译。",
)

recognition_agent = Agent(
    name="recognition_agent",
    model="qwen-max",
    instructions="你是小张，擅长文本实体识别，请提取出用户提出需要实体识别的文本，回答问题的时候先告诉我你是谁。",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    model="qwen-max",
    instructions=(
        "You are classification / language and recognition agent. You use the tools given to you to response."
        "If asked for multiple task, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[
        classification_agent.as_tool(
            tool_name="sentiment_classification",
            tool_description="解决文本情感分类",
        ),
        language_agent.as_tool(
            tool_name="translate_language",
            tool_description="进行文本翻译",
        ),
        recognition_agent.as_tool(
            tool_name="entity_recognition",
            tool_description="进行文本实体识别",
        ),
    ],
)

async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    try:
        draw_graph(orchestrator_agent, filename="Orchestrator路由")
    except:
        print("绘制agent失败，默认跳过。。。")

    msg = input("你好，我可以帮你进行文本情感分类、翻译和文本实体识别。输入 'exit' 或 'quit' 退出程序。")
    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("再见！")
            break

        with trace("Orchestrator"):
            orchestrator_result = await Runner.run(orchestrator_agent, user_input)
            final_output = None
            for item in reversed(orchestrator_result.new_items):
                try:
                    if hasattr(item, 'output') and item.output:
                        final_output = item.output
                        break
                    elif hasattr(item, 'text') and item.text:
                        final_output = item.text
                        break
                except:
                    pass
            if final_output:
                print(final_output)

if __name__ == "__main__":
    asyncio.run(main())