import os
import asyncio
from agents import Agent, Runner

# ==========================================
# 1. 核心修复：通过设置无效地址来强制关闭追踪
# ==========================================
# 设置一个无效的 URL，让追踪功能瞬间失败，从而不再卡顿
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:0"

# 你的阿里云配置
os.environ["OPENAI_API_KEY"] = "sk-c9b1982f0e674957ba9da72ce95922d6"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ==========================================
# 2. 定义 Agent (名字必须是英文)
# ==========================================

# 子 Agent 1: 情感分析
sentiment_agent = Agent(
    name="SentimentBot",
    instructions="你是一个情感分析专家。请分析用户输入的情绪（积极/消极），并给出简短评价。",
)

# 子 Agent 2: 实体识别
entity_agent = Agent(
    name="EntityBot",
    instructions="你是一个实体识别专家。请从文本中提取人名、地名、组织机构名，并以列表形式展示。",
)

# 主 Agent: 负责分发
manager_agent = Agent(
    name="Manager",
    instructions="你是一个任务分发器。如果用户输入关于心情的内容，转给 SentimentBot；如果关于提取人名地名，转给 EntityBot。",
    handoffs=[sentiment_agent, entity_agent]
)

# ==========================================
# 3. 主运行逻辑
# ==========================================
async def main():
    print("--- AI 多智能体作业演示开始 ---")
    print("你可以输入关于 '情感' 或 '实体提取' 的内容来测试。输入 'quit' 退出。")

    while True:
        # 1. 获取输入
        user_input = input("\n请输入你的请求: ")
        if user_input.lower() == 'quit':
            break

        # 2. 运行 Agent
        # 注意：这里不需要额外的配置，直接用 Runner.run
        result = await Runner.run(manager_agent, user_input)

        # 3. 【关键修复】打印结果
        # result.final_output 包含了子 Agent 处理完后的最终文字
        print(f"最终回复: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())