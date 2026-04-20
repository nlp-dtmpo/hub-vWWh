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
def text_statistics(
        text: Annotated[str, "需要统计的文本内容"]
):
    """统计文本的各项基本信息，包括字符数、单词数、句子数等。"""
    try:
        # 字符数（不含空格）
        char_count = len(text.replace(" ", ""))

        # 总字符数（含空格）
        total_chars = len(text)

        # 单词数（以空格分隔）
        word_count = len(text.split())

        # 句子数（以句号、问号、感叹号分隔）
        import re
        sentences = re.split(r'[。！？!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])

        # 中文字符数
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

        # 数字个数
        digit_count = len(re.findall(r'\d', text))

        # 字母个数
        letter_count = len(re.findall(r'[a-zA-Z]', text))

        return {
            "总字符数（含空格）": total_chars,
            "字符数（不含空格）": char_count,
            "单词数": word_count,
            "句子数": sentence_count,
            "中文字符数": chinese_chars,
            "数字个数": digit_count,
            "字母个数": letter_count
        }
    except Exception as e:
        return f"统计失败：{str(e)}"


@mcp.tool
def round_number(
        number: Annotated[float, "需要取整的数字"],
        decimals: Annotated[int, "保留的小数位数（0-5）"] = 2
):
    """将数字四舍五入到指定的小数位数。"""
    try:
        if decimals < 0:
            decimals = 0
        if decimals > 5:
            decimals = 5

        rounded = round(number, decimals)
        return f"{number} 四舍五入保留 {decimals} 位小数后是：{rounded}"
    except:
        return "取整失败，请输入有效的数字"

@mcp.tool
def reverse_string(
        text: Annotated[str, "需要反转的字符串"]
):
    """将输入的字符串反转顺序。"""
    try:
        reversed_text = text[::-1]
        return f"原字符串：{text}\n反转后：{reversed_text}"
    except:
        return "反转失败，请输入有效的字符串"