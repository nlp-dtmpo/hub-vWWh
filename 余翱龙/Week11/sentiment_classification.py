import os
from typing import Annotated
from agents import Agent, Runner, trace
from pydantic import BaseModel
from fastmcp import FastMCP
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server used to sentiment_analysis.""",
)
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
"""
[Tool(name='get_today_daily_news', title=None, description="Retrieves a list of today's daily news bulletin items from the external API.", inputSchema={'properties': {}, 'type': 'object'}, outputSchema=None, icons=None, annotations=None, meta={'_fastmcp': {'tags': []}})
"""

math_tutor_agent = Agent(
        name="Math Tutor",
        model="gemini-3.1-flash-lite-preview",
        instructions="用于情感分析"
    )
@mcp.tool
async def sentiment_analysis(sentence: Annotated[str, "需要进行情感分析的文本句子"]):
    """对输入文本进行情感倾向分析"""
    print("调用情感分析。。。")
    return await Runner.run(math_tutor_agent, "请分析这句话，并给出相应的情感分析结果：" + sentence)
