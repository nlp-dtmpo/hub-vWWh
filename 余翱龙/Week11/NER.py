import os
from typing import Annotated
from agents import Agent, Runner, trace
from pydantic import BaseModel
from fastmcp import FastMCP
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server used to NER.""",
)
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
"""
[Tool(name='get_today_daily_news', title=None, description="Retrieves a list of today's daily news bulletin items from the external API.", inputSchema={'properties': {}, 'type': 'object'}, outputSchema=None, icons=None, annotations=None, meta={'_fastmcp': {'tags': []}})
"""

math_tutor_agent = Agent(
        name="Math Tutor",
        model="gemini-3.1-flash-lite-preview",
        instructions="用于实体识别"
    )
@mcp.tool
async def NER_analysis(sentence: Annotated[str, "需要进行实体识别的文本句子"]):
    """对输入文本进行命名实体识别(NER)分析"""
    print("调用NER。。。")
    return await Runner.run(math_tutor_agent, "对输入的内容进行实体识别(NER)：" + sentence)
