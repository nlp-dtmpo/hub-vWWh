import os

# 配置API密钥
os.environ["OPENAI_API_KEY"] = "sk-51e0dfbaca11238ab8abcdbd3c201857"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import json
from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# ========== 子Agent 1: 情感分类（不使用 output_type）==========
sentiment_agent = Agent(
    name="Sentiment Analysis Agent",
    model="qwen-max",
    handoff_description="专门负责分析文本情感倾向的代理，可以判断正面、负面、中性情感。",
    instructions="""你是专业的情感分析专家。请分析用户输入的文本情感倾向。

【重要】请严格按照以下JSON格式返回结果，不要包含任何其他文字：
{
    "sentiment": "positive或negative或neutral",
    "confidence": 0.95,
    "explanation": "判断依据说明"
}

注意：
- sentiment只能是 positive、negative 或 neutral
- confidence是0-1之间的数字
- 要客观分析，避免主观臆断""",
)

# ========== 子Agent 2: 实体识别 ==========
entity_agent = Agent(
    name="Entity Recognition Agent",
    model="qwen-max",
    handoff_description="专门负责识别文本中的实体信息，如人名、地名、组织名、时间、数字等。",
    instructions="""你是专业的实体识别专家。请分析用户输入的文本，识别出所有重要的实体信息。

请以JSON格式返回识别结果，格式如下：
{
    "entities": ["实体1", "实体2", "实体3"],
    "entity_details": [
        {"text": "马云", "type": "PERSON"},
        {"text": "1999年", "type": "DATE"},
        {"text": "杭州", "type": "LOCATION"},
        {"text": "阿里巴巴集团", "type": "ORGANIZATION"}
    ]
}

实体类型包括：PERSON（人名）、LOCATION（地名）、ORGANIZATION（组织名）、DATE（日期时间）、NUMBER（数字）、MISC（其他）

只返回JSON，不要有其他解释文字。""",
)

# ========== 主Agent - 负责分发任务 ==========
main_agent = Agent(
    name="Main Router Agent",
    model="qwen-max",
    instructions="""你是任务分发代理。你需要根据用户的需求，将任务分发给合适的专业代理：

    1. 如果用户想分析文本的情感倾向（如判断心情、评价好坏、情绪分析等），请转交给 'Sentiment Analysis Agent'
    2. 如果用户想提取文本中的实体信息（如识别人物、地点、时间、组织等），请转交给 'Entity Recognition Agent'

    请准确理解用户意图，选择正确的代理来完成任务。

    注意：直接转交给对应的专业代理，不要自己回答。""",
    handoffs=[sentiment_agent, entity_agent],
)


# ========== 辅助函数：解析JSON输出 ==========
def parse_json_response(response_text: str):
    """从Agent返回的文本中提取JSON"""
    try:
        # 尝试直接解析
        return json.loads(response_text)
    except:
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return None


# ========== 主函数 ==========
async def main():
    print("=" * 60)
    print("多Agent系统演示 - 情感分析 & 实体识别")
    print("=" * 60)

    # 测试用例
    test_cases = [
        {
            "query": "我今天心情特别好，终于完成了所有工作！",
            "description": "情感分析测试 - 正面情感"
        },
        {
            "query": "这个产品质量太差了，我非常失望和愤怒。",
            "description": "情感分析测试 - 负面情感"
        },
        {
            "query": "请识别这段文本中的实体：马云于1999年在杭州创立了阿里巴巴集团。",
            "description": "实体识别测试"
        },
        {
            "query": "分析一下这句话的情感：今天天气不错。",
            "description": "情感分析测试 - 中性情感"
        },
        {
            "query": "提取实体：苹果公司CEO蒂姆·库克在2024年访问了中国北京。",
            "description": "实体识别测试"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}: {test['description']}")
        print(f"{'=' * 60}")
        print(f"用户输入: {test['query']}")

        try:
            # 运行Agent
            result = await Runner.run(main_agent, test['query'])

            print(f"\n✅ 执行成功")
            print(f"原始输出:\n{result.final_output}")

            # 尝试解析JSON（如果是情感分析结果）
            parsed = parse_json_response(result.final_output)
            if parsed and 'sentiment' in parsed:
                print(f"\n解析后的情感分析结果:")
                print(f"  情感: {parsed['sentiment']}")
                print(f"  置信度: {parsed.get('confidence', 'N/A')}")
                print(f"  说明: {parsed.get('explanation', 'N/A')}")

        except Exception as e:
            print(f"\n❌ 执行失败: {e}")

        print("\n" + "-" * 40)
        await asyncio.sleep(1)  # 避免请求过快

    # 额外演示：直接使用特定Agent
    print("\n" + "=" * 60)
    print("直接使用特定Agent示例")
    print("=" * 60)

    # 直接使用情感分析Agent
    query = "这个电影真是太好看了，我给它五星好评！"
    print(f"\n直接使用情感分析Agent")
    print(f"输入: {query}")
    result = await Runner.run(sentiment_agent, query)
    print(f"结果: {result.final_output}")

    # 直接使用实体识别Agent
    query = "李白是唐代伟大的浪漫主义诗人，出生于碎叶城（今吉尔吉斯斯坦托克马克）。"
    print(f"\n直接使用实体识别Agent")
    print(f"输入: {query}")
    result = await Runner.run(entity_agent, query)
    print(f"结果: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())