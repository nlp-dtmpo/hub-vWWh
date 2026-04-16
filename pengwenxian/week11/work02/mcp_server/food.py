from datetime import datetime

import requests
from fastmcp import FastMCP

mcp = FastMCP(
    name="Life-Assistant-MCP-Server",
    instructions="""This server provides meal, local cuisine and clothing recommendation tools.""",
)

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

WEEKDAY_MEAL_PLANS = {
    0: {
        "breakfast": ["牛奶燕麦", "水煮蛋", "全麦面包"],
        "lunch": ["黑椒鸡胸肉", "西兰花", "糙米饭"],
        "dinner": ["番茄豆腐汤", "清炒时蔬", "玉米"],
        "tips": "周一适合清淡高蛋白饮食，帮助恢复工作状态。",
    },
    1: {
        "breakfast": ["小米南瓜粥", "蒸红薯", "坚果"],
        "lunch": ["青椒牛肉", "炒菌菇", "米饭"],
        "dinner": ["虾仁蒸蛋", "凉拌黄瓜", "杂粮饭"],
        "tips": "周二适合补充优质蛋白和膳食纤维。",
    },
    2: {
        "breakfast": ["豆浆", "鸡蛋饼", "苹果"],
        "lunch": ["宫保鸡丁", "蒜蓉生菜", "米饭"],
        "dinner": ["冬瓜排骨汤", "清炒娃娃菜", "紫薯"],
        "tips": "周中容易疲劳，饮食建议兼顾能量和清爽口感。",
    },
    3: {
        "breakfast": ["酸奶水果杯", "全麦三明治"],
        "lunch": ["土豆炖牛腩", "凉拌木耳", "米饭"],
        "dinner": ["菌菇鸡汤", "清蒸南瓜", "青菜"],
        "tips": "周四可适当增加铁和维生素摄入。",
    },
    4: {
        "breakfast": ["豆腐脑", "小包子", "橙子"],
        "lunch": ["照烧鸡腿", "手撕包菜", "米饭"],
        "dinner": ["番茄意面", "蔬菜沙拉"],
        "tips": "周五可以稍微丰富一点，但避免过度油腻。",
    },
    5: {
        "breakfast": ["皮蛋瘦肉粥", "鸡蛋", "时令水果"],
        "lunch": ["香煎三文鱼", "芦笋", "土豆泥"],
        "dinner": ["牛肉蔬菜锅", "荞麦面"],
        "tips": "周末适合均衡搭配，也可以给自己做点喜欢的健康餐。",
    },
    6: {
        "breakfast": ["牛油果吐司", "拿铁", "香蕉"],
        "lunch": ["番茄肥牛", "清炒西葫芦", "米饭"],
        "dinner": ["山药排骨汤", "凉拌海带丝", "杂粮粥"],
        "tips": "周日饮食以温和、易消化为主，为新一周做准备。",
    },
}

REGIONAL_FOODS = {
    "北京": ["北京烤鸭", "炸酱面", "卤煮火烧", "豆汁焦圈"],
    "上海": ["生煎包", "葱油拌面", "排骨年糕", "蟹粉小笼"],
    "天津": ["狗不理包子", "煎饼果子", "锅巴菜", "耳朵眼炸糕"],
    "重庆": ["重庆火锅", "小面", "酸辣粉", "辣子鸡"],
    "四川": ["麻婆豆腐", "回锅肉", "串串香", "钟水饺"],
    "成都": ["夫妻肺片", "担担面", "龙抄手", "甜水面"],
    "湖南": ["剁椒鱼头", "小炒黄牛肉", "口味虾", "糖油粑粑"],
    "长沙": ["臭豆腐", "糖油粑粑", "口味虾", "米粉"],
    "广东": ["白切鸡", "烧鹅", "肠粉", "煲仔饭"],
    "广州": ["虾饺", "艇仔粥", "云吞面", "双皮奶"],
    "深圳": ["椰子鸡", "潮汕牛肉火锅", "肠粉", "烧腊饭"],
    "福建": ["佛跳墙", "沙县小吃", "海蛎煎", "土笋冻"],
    "厦门": ["沙茶面", "土笋冻", "海蛎煎", "花生汤"],
    "浙江": ["西湖醋鱼", "东坡肉", "片儿川", "定胜糕"],
    "杭州": ["龙井虾仁", "片儿川", "东坡肉", "葱包桧"],
    "江苏": ["盐水鸭", "蟹黄汤包", "锅盖面", "扬州炒饭"],
    "南京": ["盐水鸭", "鸭血粉丝汤", "小笼包", "梅花糕"],
    "陕西": ["肉夹馍", "凉皮", "羊肉泡馍", "biangbiang面"],
    "西安": ["肉夹馍", "甑糕", "羊肉泡馍", "冰峰"],
    "云南": ["过桥米线", "汽锅鸡", "鲜花饼", "菌子火锅"],
    "昆明": ["过桥米线", "鲜花饼", "汽锅鸡", "小锅米线"],
    "湖北": ["热干面", "排骨藕汤", "豆皮", "三鲜豆皮"],
    "武汉": ["热干面", "周黑鸭", "豆皮", "排骨藕汤"],
    "东北": ["锅包肉", "地三鲜", "铁锅炖", "杀猪菜"],
    "哈尔滨": ["红肠", "锅包肉", "大列巴", "杀猪菜"],
}


def _match_region(address: str) -> str | None:
    for region in REGIONAL_FOODS:
        if region in address:
            return region
    return None


def _get_clothing_level(temp_c: int) -> dict:
    if temp_c >= 30:
        return {
            "level": "炎热",
            "suggestion": "建议穿短袖、短裤、裙装等清凉透气衣物，并注意防晒补水。",
        }
    if temp_c >= 24:
        return {
            "level": "偏热",
            "suggestion": "建议穿短袖或轻薄衬衫，早晚可带一件薄外套。",
        }
    if temp_c >= 18:
        return {
            "level": "舒适",
            "suggestion": "建议穿长袖、薄卫衣、衬衫或轻便外套，整体以舒适为主。",
        }
    if temp_c >= 10:
        return {
            "level": "微凉",
            "suggestion": "建议穿针织衫、卫衣、风衣等，并注意早晚温差。",
        }
    if temp_c >= 0:
        return {
            "level": "寒冷",
            "suggestion": "建议穿毛衣、厚外套、大衣，必要时加围巾保暖。",
        }
    return {
        "level": "严寒",
        "suggestion": "建议穿羽绒服、保暖内搭、帽子和围巾，减少长时间户外停留。",
    }


def _fetch_weather(address: str) -> dict:
    response = requests.get(
        f"https://wttr.in/{address}",
        params={"format": "j1"},
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    weather_data = response.json()
    current = weather_data["current_condition"][0]
    nearest_area = weather_data.get("nearest_area", [{}])[0]

    return {
        "location": nearest_area.get("areaName", [{"value": address}])[0]["value"],
        "temperature_c": int(current["temp_C"]),
        "feels_like_c": int(current["FeelsLikeC"]),
        "humidity": current["humidity"],
        "weather_desc": current["weatherDesc"][0]["value"],
        "wind_speed_kmph": current["windspeedKmph"],
    }


@mcp.tool
def recommend_daily_meals():
    """根据今天是周几，推荐一日三餐食谱。"""
    weekday = datetime.now().weekday()
    meal_plan = WEEKDAY_MEAL_PLANS[weekday]
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "weekday": WEEKDAY_NAMES[weekday],
        "breakfast": meal_plan["breakfast"],
        "lunch": meal_plan["lunch"],
        "dinner": meal_plan["dinner"],
        "tips": meal_plan["tips"],
    }


@mcp.tool
def recommend_local_foods(address: str):
    """根据用户地址推荐当地特色美食。"""
    address = address.strip()
    if not address:
        return {
            "success": False,
            "message": "请输入地址信息，例如：北京市朝阳区、四川成都、广东深圳南山区。",
        }

    matched_region = _match_region(address)
    if not matched_region:
        return {
            "success": True,
            "address": address,
            "matched_region": "未匹配到特定城市",
            "foods": ["当地米粉", "特色烧烤", "家常炖菜", "时令小吃"],
            "message": "暂未识别到明确城市，先给你一些通用地方美食灵感。",
        }

    return {
        "success": True,
        "address": address,
        "matched_region": matched_region,
        "foods": REGIONAL_FOODS[matched_region],
        "message": f"已根据地址匹配到 {matched_region}，以下是推荐的当地美食。",
    }


@mcp.tool
def recommend_clothing_by_weather(address: str):
    """根据用户地址查询当前天气，并推荐穿衣建议。"""
    address = address.strip()
    if not address:
        return {
            "success": False,
            "message": "请输入地址信息，例如：上海浦东新区、北京市海淀区、广州市天河区。",
        }

    try:
        weather = _fetch_weather(address)
        clothing = _get_clothing_level(weather["feels_like_c"])
        return {
            "success": True,
            "address": address,
            "location": weather["location"],
            "weather": weather["weather_desc"],
            "temperature_c": weather["temperature_c"],
            "feels_like_c": weather["feels_like_c"],
            "humidity": weather["humidity"],
            "wind_speed_kmph": weather["wind_speed_kmph"],
            "clothing_level": clothing["level"],
            "suggestion": clothing["suggestion"],
        }
    except Exception as exc:
        return {
            "success": False,
            "address": address,
            "message": f"天气查询失败：{exc}",
        }
