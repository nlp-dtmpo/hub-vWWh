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
def translate_text(
    text: Annotated[str, "The text to be translated"],
    source_lang: Annotated[str, "Source language code, e.g., 'en', 'zh', 'ja'. Default 'auto'"] = "auto",
    target_lang: Annotated[str, "Target language code, e.g., 'zh', 'en', 'fr'. Default 'zh'"] = "zh"
):
    """
    Translates text from source language to target language using MyMemory API.
    Supports many languages (en, zh, ja, ko, fr, de, es, etc.).
    """
    url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"{source_lang}|{target_lang}",
        "de": "a@b.c"  # 可选邮箱，用于提升限额
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        # MyMemory 返回翻译结果在 responseData 里
        translated = data.get("responseData", {}).get("translatedText", "")
        return translated
    except Exception as e:
        return f"Translation failed: {str(e)}"


import re
from typing import Annotated, Optional

# 模拟薪资数据库：岗位关键词 -> (月薪下限, 月薪上限, 单位, 数据来源)
# 实际使用时可以从 JSON 文件或数据库加载
SALARY_DB = {
    "python工程师": (15000, 30000, "元/月", "某招聘平台2025年数据"),
    "java工程师": (14000, 28000, "元/月", "某招聘平台2025年数据"),
    "产品经理": (18000, 35000, "元/月", "某招聘平台2025年数据"),
    "前端开发": (12000, 25000, "元/月", "某招聘平台2025年数据"),
    "ui设计师": (10000, 20000, "元/月", "某招聘平台2025年数据"),
    "数据分析师": (12000, 26000, "元/月", "某招聘平台2025年数据"),
    "运维工程师": (10000, 22000, "元/月", "某招聘平台2025年数据"),
}


@mcp.tool
def get_job_salary(
        job_title: Annotated[str, "Job title or keyword, e.g., 'Python工程师', '产品经理'"],
        city: Annotated[Optional[str], "City name, e.g., '北京', '上海' (optional)"] = None
):
    """
    Query the estimated salary range for a given job title, optionally filtered by city.
    Returns a string with salary info.
    """
    # 标准化输入：转小写，去除空格
    normalized = job_title.strip().lower()
    # 模糊匹配：查找包含关键词的岗位
    matched = []
    for key, (low, high, unit, source) in SALARY_DB.items():
        if normalized in key or key in normalized:
            matched.append((key, low, high, unit, source))

    if not matched:
        return f"未找到岗位 '{job_title}' 的薪资数据。可尝试关键词：{', '.join(SALARY_DB.keys())}"

    # 如果有多个匹配，取第一个（或可合并）
    key, low, high, unit, source = matched[0]
    city_str = f" 在{city}" if city else ""
    return f"岗位：{key}{city_str}，月薪范围：{low:,} ~ {high:,} {unit}（{source}）"


import math
from typing import Annotated

@mcp.tool
def calculate_body_fat(
    height_cm: Annotated[float, "Height in centimeters (e.g., 175.5)"],
    weight_kg: Annotated[float, "Weight in kilograms (e.g., 70.5)"],
    age: Annotated[int, "Age in years (should be between 18 and 60 for accurate results)"],
    gender: Annotated[str, "Gender: 'male' or 'female'"]
):
    """
    Estimate body fat percentage using height, weight, age, and gender.
    Based on the Deurenberg formula (valid for adults aged 18–60).
    Returns body fat percentage and a health interpretation.
    """
    # 输入校验
    if height_cm <= 0 or weight_kg <= 0 or age <= 0:
        return "错误：身高、体重、年龄必须为正数。"
    if age < 18 or age > 60:
        return "警告：公式主要适用于18-60岁成年人，结果可能不准确。"
    if gender not in ["male", "female"]:
        return "错误：性别必须是 'male' 或 'female'。"

    try:
        # 计算 BMI
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)

        # 性别编码：男性=1，女性=0
        gender_code = 1 if gender == "male" else 0

        # Deurenberg 公式（1991年版本）
        # 体脂率(%) = 1.20 * BMI + 0.23 * Age - 10.8 * gender_code - 5.4
        body_fat = 1.20 * bmi + 0.23 * age - 10.8 * gender_code - 5.4

        # 限制结果在合理范围（5%～45%，超出则裁剪）
        body_fat = max(5.0, min(45.0, body_fat))
        body_fat_rounded = round(body_fat, 1)

        # 健康评估（基于性别和体脂率范围，参考ACSM标准）
        if gender == "male":
            if body_fat < 6:
                health = "体脂率过低（<6%），存在健康风险"
            elif body_fat < 14:
                health = "优秀（运动员级别）"
            elif body_fat < 21:
                health = "良好（健康范围）"
            elif body_fat < 25:
                health = "偏高（超重风险）"
            else:
                health = "过高（肥胖相关风险增加）"
        else:  # female
            if body_fat < 12:
                health = "体脂率过低（<12%），存在健康风险"
            elif body_fat < 20:
                health = "优秀（运动员级别）"
            elif body_fat < 28:
                health = "良好（健康范围）"
            elif body_fat < 32:
                health = "偏高（超重风险）"
            else:
                health = "过高（肥胖相关风险增加）"

        # 构建返回消息
        result = (
            f"📊 体脂率估算结果（基于身高/体重/年龄/性别）：\n"
            f"• BMI：{bmi:.1f} kg/m²\n"
            f"• 体脂率：{body_fat_rounded}%\n"
            f"• 评估：{health}\n"
            f"⚠️ 注意：此公式为估算值，仅供参考。精确测量需使用皮脂钳或双能X射线吸收法。"
        )
        return result

    except Exception as e:

        return f"计算失败：{str(e)}"
