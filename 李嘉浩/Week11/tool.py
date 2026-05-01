import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

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
def query_btc_info(future_year: Annotated[str, "未来的某个年份"]):
    """Query btc price in 2030,2040,2050."""
    # TODO 基于用户
    if int(future_year) >= 2030 and  int(future_year) < 2040:
        return "BTCUSDT:$1000000"
    elif int(future_year) >= 2040 and  int(future_year) < 2050:
        return "BTCUSDT:$2000000"
    else:
        return "BTCUSDT:$100000000"
    
    
import requests

@mcp.tool
def get_london_gold_price():
    """
    获取伦敦黄金现货价格
    
    """
    # 接口地址和参数
    url = "https://api.jijinhao.com/sQuoteCenter/realTime.htm"
    params = {
        "code": "JO_92233",   # 现货黄金代码
        # "_": timestamp       # 可选的时间戳防缓存参数，requests会自动处理
    }

    # 模拟浏览器请求头，避免被反爬
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://quote.cngold.org/",
    }

    try:
        # 1. 发送请求
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.encoding = "utf-8"
        response.raise_for_status()

        # 2. 返回的是一个JavaScript片段：var hq_str = "数据1,数据2,...";
        js_text = response.text

        # 提取双引号内的核心数据字符串
        import re
        match = re.search(r'var\s+hq_str\s*=\s*"([^"]+)"', js_text)
        if not match:
            print("未找到行情数据，接口可能已变更")
            return None

        raw_data = match.group(1)
        fields = raw_data.split(",")

        # 3. 字段解析（根据实际观察顺序，后续可能变动）
        # 示例数据："现货黄金,0,4621.52,4566.36,4635.57,4559.59,0,0,59861.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-05-01,18:14:58,00,2,-55.1602,-1.1936,4566.36,4567.21,4623.3,100,2026-05-01,18:14:57,"
        # 索引:0 → 名称
        #      3 → 最新价（本例中索引3值为4566.36）
        #      1,2,4,5,... 其他字段表示开盘、收盘、最高、最低等
        name = fields[0] if len(fields) > 0 else ""

        # 关键行情字段说明（基于经验映射，建议打印全部字段自行核对）
        price_index = None
        # 动态查找最新价：通常在字段3或4位置，具体需调试，这里预置猜测索引3
        if len(fields) > 3:
            price_index = 3

        latest_price = float(fields[price_index]) if price_index is not None and fields[price_index].replace(".", "", 1).isdigit() else None
        open_price = float(fields[1]) if len(fields) > 1 and fields[1].replace(".", "", 1).isdigit() else None   # 开盘价
        pre_close = float(fields[2]) if len(fields) > 2 and fields[2].replace(".", "", 1).isdigit() else None    # 昨收价
        high = float(fields[4]) if len(fields) > 4 and fields[4].replace(".", "", 1).isdigit() else None         # 最高价
        low = float(fields[5]) if len(fields) > 5 and fields[5].replace(".", "", 1).isdigit() else None          # 最低价
        volume = fields[8] if len(fields) > 8 else None                                                         # 成交量（可能为字符串）

        # 涨跌额与涨跌幅
        change = float(fields[34]) if len(fields) > 34 and fields[34].replace("-", "", 1).replace(".", "", 1).isdigit() else None
        change_percent = float(fields[35]) if len(fields) > 35 and fields[35].replace("-", "", 1).replace(".", "", 1).isdigit() else None

        # 时间字段
        date = fields[30] if len(fields) > 30 else None   # 日期
        time = fields[31] if len(fields) > 31 else None   # 时间

        # 构建结果字典
        gold_data = {
            "name": name,
            "latest_price": latest_price,
            "open": open_price,
            "pre_close": pre_close,
            "high": high,
            "low": low,
            "change": change,
            "change_percent": change_percent,
            "volume": volume,
            "datetime": f"{date} {time}" if date and time else None,
            "raw_data": fields   # 保留原始字段便于调试
        }

        print(f"获取成功！最新价：{latest_price} 美元/盎司")
        return gold_data

    except requests.exceptions.RequestException as e:
        print(f"网络请求出错: {e}")
        return None
    except Exception as e:
        print(f"解析失败: {e}")
        return None
    
    
@mcp.tool
def count_zeros(num_str: str) -> int:
    """
    统计数字字符串中字符 '0' 出现的次数。
    
    Args:
        num_str: 数字字符串，例如 "102030" 或 "-0.001"
    
    Returns:
        字符 '0' 的个数
    """
    return num_str.count('0')

