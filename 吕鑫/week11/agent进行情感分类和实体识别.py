import os
import asyncio
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

# 请提前在环境变量中配置 API Key；这里保留兼容接口地址默认值
os.environ.setdefault("OPENAI_API_KEY", "sk-35609c8a5c4e42c6bc8b38888615c54b")
os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class SentimentOutput(BaseModel):
    sentiment: Literal["积极", "消极", "中性"] = Field(
        description="文本情感分类结果",
        validation_alias=AliasChoices("sentiment", "情感"),
    )
    confidence: float = Field(description="0 到 1 之间的置信度")
    reason: str = Field(
        default="模型未提供判断依据",
        description="给出简短判断依据",
        validation_alias=AliasChoices("reason", "依据", "理由"),
    )


class EntityItem(BaseModel):
    name: str = Field(
        description="实体名称",
        validation_alias=AliasChoices("name", "value", "实体", "名称"),
    )
    entity_type: str = Field(description="实体类型，如人名、地名、组织、时间等")


class EntityRecognitionOutput(BaseModel):
    entities: list[EntityItem] = Field(description="识别出的实体列表")
    summary: str = Field(description="简短总结识别结果")


sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen-max",
    handoff_description="负责文本情感分类。",
    instructions=(
        "你是情感分类专家。"
        "当用户要求判断文本情绪、情感倾向、正负面时，你负责处理。"
        "你必须返回合法 json。"
        "只输出结构化结果。"
        "请优先使用字段 sentiment、confidence、reason。"
        "情感标签只能是：积极、消极、中性。"
        "confidence 必须是 0 到 1 之间的小数。"
    ),
    output_type=SentimentOutput,
)


entity_agent = Agent(
    name="entity_agent",
    model="qwen-max",
    handoff_description="负责文本实体识别。",
    instructions=(
        "你是实体识别专家。"
        "当用户要求识别人名、地名、组织名、时间、地点或其他命名实体时，你负责处理。"
        "你必须返回合法 json。"
        "只输出结构化结果。"
        "请优先使用字段 entities，且每个实体项使用字段 name 和 entity_type。"
        "如果没有识别到实体，entities 返回空列表，并在 summary 中说明。"
    ),
    output_type=EntityRecognitionOutput,
)


triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions=(
        "你是主路由 agent。"
        "你的唯一职责是根据用户请求，把任务交给最合适的子 agent。"
        "如果用户要求做情感分类、情绪判断、正负面分析，转交给 sentiment_agent。"
        "如果用户要求做实体识别、命名实体识别、提取人名地名组织名时间地点，转交给 entity_agent。"
        "不要自己直接完成任务，必须使用 handoff。"
    ),
    handoffs=[sentiment_agent, entity_agent],
)


def format_output(result) -> str:
    agent_name = result.last_agent.name if result.last_agent else "unknown_agent"

    if agent_name == "sentiment_agent":
        output = result.final_output_as(SentimentOutput)
        return (
            f"已路由到: 情感分类 Agent\n"
            f"情感: {output.sentiment}\n"
            f"置信度: {output.confidence}\n"
            f"依据: {output.reason}"
        )

    if agent_name == "entity_agent":
        output = result.final_output_as(EntityRecognitionOutput)
        entity_lines = [f"- {item.name}（{item.entity_type}）" for item in output.entities]
        entities_text = "\n".join(entity_lines) if entity_lines else "- 未识别到实体"
        return (
            f"已路由到: 实体识别 Agent\n"
            f"识别结果:\n{entities_text}\n"
            f"总结: {output.summary}"
        )

    return str(result.final_output)


async def run_once(user_input: str):
    result = await Runner.run(triage_agent, user_input)
    print(format_output(result))


async def main():
    print("多 Agent 文本处理系统已启动")
    print("输入示例：")
    print("1. 请判断这段文本的情感：这家餐厅服务很好，我非常满意。")
    print("2. 请识别这段文本中的实体：2026年张三在北京大学参加了腾讯举办的论坛。")
    print("输入 quit 退出。")

    while True:
        user_input = input("\n请输入请求: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("程序已退出")
            break
        if not user_input:
            print("请输入有效内容")
            continue

        try:
            await run_once(user_input)
        except Exception as exc:
            print(f"执行失败: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
