from datetime import datetime

from fastmcp import FastMCP
mcp = FastMCP(
    name="My-Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


# 乘法计算
@mcp.tool
def get_multiply(num1, num2):
    """
    计算两个数字的乘积
    """
    return num1 * num2


# 文本长度统计Tool
@mcp.tool
def get_text_length(input_text: str) -> int:
    """
    统计文本的总字符数（包含中文、英文、空格、标点）
    """
    return len(input_text)


# 获取当前时间Tool
@mcp.tool
def get_current_time() -> str:
    """
    获取当前日期时间
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")