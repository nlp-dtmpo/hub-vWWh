import requests
from datetime import datetime
import random

from fastmcp import FastMCP
mcp = FastMCP(
    name="Custom-MCP-Server",
    instructions="""This server contains some custom tools(generate_password(), get_current_time(), sum_two_num()).""",
)

@mcp.tool()
def generate_password(length: int = 8) -> str:
    """生成指定长度的随机密码"""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%"
    return "".join(random.choice(chars) for _ in range(length))

@mcp.tool()
def get_current_time() -> str:
    """获取当前北京时间"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间：{now}"

@mcp.tool()
def sum_two_num(a: float, b: float) -> float:
    """两个数字相加"""
    return a + b