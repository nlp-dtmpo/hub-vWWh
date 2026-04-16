tool.py
--------------
import re
from datetime import datetime
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

EMPLOYEE_PROFILES = {
    "E1001": {
        "employee_id": "E1001",
        "name": "张敏",
        "department": "人力资源部",
        "position": "招聘经理",
        "direct_manager": "刘洋",
        "employment_status": "在职",
        "level": "P6",
    },
    "E1002": {
        "employee_id": "E1002",
        "name": "王磊",
        "department": "财务部",
        "position": "费用会计",
        "direct_manager": "陈洁",
        "employment_status": "在职",
        "level": "P5",
    },
    "E1003": {
        "employee_id": "E1003",
        "name": "李娜",
        "department": "销售部",
        "position": "客户经理",
        "direct_manager": "周强",
        "employment_status": "试用期",
        "level": "P4",
    },
}

LEAVE_BALANCES = {
    "E1001": {"annual_leave_total": 10, "annual_leave_used": 3},
    "E1002": {"annual_leave_total": 8, "annual_leave_used": 2},
    "E1003": {"annual_leave_total": 5, "annual_leave_used": 1},
}

EXPENSE_POLICIES = {
    "P4": {
        "国内差旅住宿": {
            "standard": "300元/晚",
            "rule": "按实际入住晚数报销，单晚不超过300元，需提供发票。",
        },
        "餐补": {
            "standard": "80元/天",
            "rule": "按出差自然日发放，不与客户招待重复报销。",
        },
        "市内交通": {
            "standard": "100元/天",
            "rule": "优先公共交通，打车需填写事由。",
        },
    },
    "P5": {
        "国内差旅住宿": {
            "standard": "400元/晚",
            "rule": "按实际入住晚数报销，单晚不超过400元，需提供发票。",
        },
        "餐补": {
            "standard": "100元/天",
            "rule": "按出差自然日发放，不与客户招待重复报销。",
        },
        "市内交通": {
            "standard": "120元/天",
            "rule": "优先公共交通，特殊时段打车需备注。",
        },
    },
    "P6": {
        "国内差旅住宿": {
            "standard": "500元/晚",
            "rule": "一线城市可上浮至550元/晚，需附酒店发票与行程说明。",
        },
        "餐补": {
            "standard": "120元/天",
            "rule": "按出差自然日发放，跨城当天可按1天计算。",
        },
        "市内交通": {
            "standard": "150元/天",
            "rule": "可据实报销出租车费用，超标需主管审批。",
        },
    },
    "P8": {
        "国内差旅住宿": {
            "standard": "800元/晚",
            "rule": "可据实报销，建议优先选择公司协议酒店。",
        },
        "餐补": {
            "standard": "180元/天",
            "rule": "按出差自然日发放，客户宴请需单独说明。",
        },
        "市内交通": {
            "standard": "300元/天",
            "rule": "可据实报销专车费用，建议保留行程单。",
        },
    },
}

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"


@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


@mcp.tool
def query_employee_profile(employee_id: Annotated[str, "员工编号，例如 E1001"]):
    """Query employee basic profile including department, position, manager and status."""
    employee_id = employee_id.strip().upper()
    profile = EMPLOYEE_PROFILES.get(employee_id)
    if not profile:
        return {
            "employee_id": employee_id,
            "found": False,
            "message": f"未找到员工 {employee_id} 的档案信息。",
        }
    return {
        "found": True,
        **profile,
    }


@mcp.tool
def create_leave_summary(
    employee_id: Annotated[str, "员工编号，例如 E1001"],
    start_date: Annotated[str, "请假开始日期，格式 YYYY-MM-DD"],
    end_date: Annotated[str, "请假结束日期，格式 YYYY-MM-DD"],
):
    """Create a leave summary with leave days, remaining quota and reminder."""
    employee_id = employee_id.strip().upper()
    profile = EMPLOYEE_PROFILES.get(employee_id)
    if not profile:
        return {
            "employee_id": employee_id,
            "success": False,
            "message": f"未找到员工 {employee_id}，无法生成请假摘要。",
        }

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return {
            "employee_id": employee_id,
            "success": False,
            "message": "日期格式错误，请使用 YYYY-MM-DD。",
        }

    if end < start:
        return {
            "employee_id": employee_id,
            "success": False,
            "message": "结束日期不能早于开始日期。",
        }

    leave_days = (end - start).days + 1
    balance = LEAVE_BALANCES.get(employee_id, {"annual_leave_total": 5, "annual_leave_used": 0})
    remaining_before = balance["annual_leave_total"] - balance["annual_leave_used"]
    remaining_after = remaining_before - leave_days

    reminder = "额度充足，可按流程提交审批。"
    if remaining_after < 0:
        reminder = "请假天数已超过年假剩余额度，需改走事假或补充审批说明。"
    elif remaining_after <= 2:
        reminder = "请假后年假余额较少，建议确认后续排班安排。"

    return {
        "success": True,
        "employee_id": employee_id,
        "employee_name": profile["name"],
        "department": profile["department"],
        "start_date": start_date,
        "end_date": end_date,
        "leave_days": leave_days,
        "annual_leave_total": balance["annual_leave_total"],
        "annual_leave_used": balance["annual_leave_used"],
        "annual_leave_remaining_before": remaining_before,
        "annual_leave_remaining_after": remaining_after,
        "reminder": reminder,
    }


@mcp.tool
def query_expense_policy(
    level: Annotated[str, "员工职级，例如 P4、P5、P6"],
    expense_type: Annotated[str, "费用类型，例如 国内差旅住宿、餐补、市内交通"],
):
    """Query expense reimbursement policy by employee level and expense type."""
    normalized_level = level.strip().upper()
    normalized_expense_type = expense_type.strip()
    level_policy = EXPENSE_POLICIES.get(normalized_level)

    if not level_policy:
        return {
            "found": False,
            "level": normalized_level,
            "expense_type": normalized_expense_type,
            "message": f"未配置 {normalized_level} 的报销标准。",
        }

    policy = level_policy.get(normalized_expense_type)
    if not policy:
        return {
            "found": False,
            "level": normalized_level,
            "expense_type": normalized_expense_type,
            "message": f"未找到 {normalized_level} 在 {normalized_expense_type} 场景下的报销标准。",
        }

    return {
        "found": True,
        "level": normalized_level,
        "expense_type": normalized_expense_type,
        "standard": policy["standard"],
        "rule": policy["rule"],
    }

----------
import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-35609c8a5c4e42c6bc8b38888615c54b"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

async def run(mcp_server: MCPServer):
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    agent = Agent(
        name="Assistant",
        instructions="你是qwen，擅长回答各类问题。",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model="qwen-flash",
            openai_client=external_client,
        )
    )

    # 调用 get_today_daily_news 工具 -》 得到结果 -》 大模型message格式 汇总为输入-》 qwen-flash -》 结果
    message = "最近有什么新闻？"
    print(f"Running: {message}")

    """
    [
        {
            "role": "system",
            "content": "你是qwen，擅长回答各类问题。"
        },
        {
            "role": "user",
            "content": "最近有什么新闻？ get_today_daily_news 调用结果"
        },
    ]
    """

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


    # 调用 get_city_weather 工具 -》 得到结果 -》 大模型message格式 汇总为输入-》 qwen-flash -》 结果
    message = "武汉最近的天气怎么样？"
    print(f"Running: {message}")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print("\n")

    # 调用 query_employee_profile 工具
    message = "帮我查一下员工 E1001 的基本信息"
    print(f"Running: {message}")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print("\n")

    # 调用 create_leave_summary 工具
    message = "员工 E1001 从 2026-04-20 到 2026-04-22 请假，帮我生成请假摘要"
    print(f"Running: {message}")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print("\n")

    # 调用 query_expense_policy 工具
    message = "P6 级别的国内差旅住宿报销标准是多少？"
    print(f"Running: {message}")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


async def main():
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
    )as server:
        await run(server)

if __name__ == "__main__":
    asyncio.run(main())
