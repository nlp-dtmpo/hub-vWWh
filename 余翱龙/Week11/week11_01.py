import os
from agents import Agent, Runner, function_tool, set_default_openai_key
from agents import set_default_openai_api, set_tracing_disabled

# 设置OpenAI API密钥（核心配置）
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
# ---------------------- 子Agent 1：情感分类 ----------------------
sentiment_agent = Agent(
    name="情感分类助手",
    model="qwen3-max",  # 可替换为gpt-4o
    instructions="你是专业的情感分类专家，只返回情感结果：积极、消极、中性，不要多余文字"
)

# ---------------------- 子Agent 2：实体识别 ----------------------
entity_agent = Agent(
    name="实体识别助手",
    model="qwen3-max",
    instructions="你是实体识别专家，提取文本中的实体并分类，格式清晰，无多余内容"
)



# 主Agent：接收用户请求，自动选择调用哪个子Agent
main_agent = Agent(
    name="任务调度主助手",
    model="qwen3-max",
    instructions="""
    你的职责：
    1. 分析用户输入的需求
    2. 如果用户需要【情感分析/情感判断/情绪分类】，调用 run_sentiment_analysis
    3. 如果用户需要【提取实体/实体识别/找人名地名】，调用 run_entity_recognition
    4. 如果同时需要，两个工具都调用
    5. 最终返回整理后的结果
    """,
    tools=[sentiment_agent.as_tool(tool_name="sentiment_analysis",tool_description="sentiment_analysis"),
           entity_agent.as_tool(tool_name="entity_recognition",tool_description="entity_recognition")]  # 绑定两个子任务工具
)

# ---------------------- 程序入口 ----------------------
if __name__ == "__main__":

    while True:
        user_input = input("\n请输入你的请求（输入 exit 退出）：")
        if user_input.lower() == "exit":
            print("程序已退出！")
            break

        # 主Agent运行，自动调度子任务
        print("\n正在处理请求...")
        result = Runner.run_sync(main_agent, user_input)

        # 输出最终结果
        print("\n" + "=" * 30 + "最终结果" + "=" * 30)
        print(result.final_output)