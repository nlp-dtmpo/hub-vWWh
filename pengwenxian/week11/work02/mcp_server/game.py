from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP(
    name="Game-Recommendation-MCP-Server",
    instructions="""This server provides game recommendations by weekday, platform and player preference.""",
)

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

WEEKDAY_GAME_PLANS = {
    0: {
        "theme": "轻松恢复状态",
        "genres": ["休闲解压", "剧情探索"],
        "games": ["星露谷物语", "纪念碑谷", "胡闹厨房"],
        "tips": "周一适合玩节奏较慢、负担较低的游戏，放松但不过度熬夜。",
    },
    1: {
        "theme": "短时高反馈",
        "genres": ["动作闯关", "卡牌策略"],
        "games": ["杀戮尖塔", "哈迪斯", "土豆兄弟"],
        "tips": "周二适合一局一局可中断的游戏，方便学习或工作间隙游玩。",
    },
    2: {
        "theme": "中周提神",
        "genres": ["竞技对抗", "团队合作"],
        "games": ["无畏契约", "英雄联盟", "Apex Legends"],
        "tips": "周三可以来一点竞技游戏，但建议控制时长，避免情绪上头。",
    },
    3: {
        "theme": "策略思考",
        "genres": ["经营模拟", "战棋策略"],
        "games": ["文明6", "火焰纹章", "双点校园"],
        "tips": "周四适合玩更需要思考和规划的游戏，体验慢慢推进的成就感。",
    },
    4: {
        "theme": "周末预热",
        "genres": ["开放世界", "多人联机"],
        "games": ["原神", "我的世界", "双人成行"],
        "tips": "周五适合和朋友一起玩，或者开一个能持续体验的开放世界。",
    },
    5: {
        "theme": "沉浸长线体验",
        "genres": ["角色扮演", "大型冒险"],
        "games": ["黑神话：悟空", "赛博朋克2077", "艾尔登法环"],
        "tips": "周六时间更完整，适合推进主线、刷副本或体验大作。",
    },
    6: {
        "theme": "轻松收尾",
        "genres": ["治愈休闲", "合作派对"],
        "games": ["动物森友会", "糖豆人", "马里奥赛车8"],
        "tips": "周日适合轻松愉快的游戏，为新一周留出精力。",
    },
}

PLATFORM_GAME_MAP = {
    "pc": ["英雄联盟", "无畏契约", "CS2", "赛博朋克2077", "文明6"],
    "电脑": ["英雄联盟", "无畏契约", "CS2", "赛博朋克2077", "文明6"],
    "steam": ["哈迪斯", "星露谷物语", "双人成行", "土豆兄弟", "杀戮尖塔"],
    "ps5": ["战神", "漫威蜘蛛侠", "最终幻想7 重生", "地平线 西之绝境"],
    "playstation": ["战神", "漫威蜘蛛侠", "最终幻想7 重生", "地平线 西之绝境"],
    "switch": ["塞尔达传说 王国之泪", "动物森友会", "马里奥赛车8", "喷射战士3"],
    "xbox": ["极限竞速：地平线5", "光环：无限", "星空", "盗贼之海"],
    "手机": ["王者荣耀", "和平精英", "崩坏：星穹铁道", "金铲铲之战"],
    "mobile": ["王者荣耀", "和平精英", "崩坏：星穹铁道", "金铲铲之战"],
}

PREFERENCE_GAME_MAP = {
    "休闲": {
        "genres": ["治愈", "轻松", "解压"],
        "games": ["星露谷物语", "动物森友会", "纪念碑谷"],
    },
    "竞技": {
        "genres": ["MOBA", "射击", "对抗"],
        "games": ["英雄联盟", "无畏契约", "CS2"],
    },
    "剧情": {
        "genres": ["叙事", "冒险", "角色扮演"],
        "games": ["底特律：变人", "巫师3", "最后生还者"],
    },
    "联机": {
        "genres": ["合作", "派对", "社交"],
        "games": ["双人成行", "胡闹厨房", "我的世界"],
    },
    "开放世界": {
        "genres": ["探索", "收集", "长线成长"],
        "games": ["原神", "塞尔达传说 王国之泪", "艾尔登法环"],
    },
    "策略": {
        "genres": ["经营", "卡牌", "战棋"],
        "games": ["文明6", "杀戮尖塔", "炉石传说"],
    },
    "恐怖": {
        "genres": ["惊悚", "氛围", "求生"],
        "games": ["港诡实录", "逃生", "生化危机4 重制版"],
    },
}


def _match_keyword(text: str, mapping: dict) -> str | None:
    text = text.lower()
    for keyword in mapping:
        if keyword.lower() in text:
            return keyword
    return None


@mcp.tool
def recommend_games_by_weekday():
    """根据当前周几推荐适合游玩的游戏。"""
    weekday = datetime.now().weekday()
    plan = WEEKDAY_GAME_PLANS[weekday]
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "weekday": WEEKDAY_NAMES[weekday],
        "theme": plan["theme"],
        "genres": plan["genres"],
        "games": plan["games"],
        "tips": plan["tips"],
    }


@mcp.tool
def recommend_games_by_platform(platform: str):
    """根据用户设备平台推荐对应游戏。"""
    platform = platform.strip()
    if not platform:
        return {
            "success": False,
            "message": "请输入游戏平台，例如：PC、Steam、PS5、Switch、手机。",
        }

    matched = _match_keyword(platform, PLATFORM_GAME_MAP)
    if not matched:
        return {
            "success": True,
            "platform": platform,
            "matched_platform": "未匹配到特定平台",
            "games": ["星露谷物语", "我的世界", "原神"],
            "message": "暂未识别到明确平台，先给你一些跨平台或常见推荐。",
        }

    return {
        "success": True,
        "platform": platform,
        "matched_platform": matched,
        "games": PLATFORM_GAME_MAP[matched],
        "message": f"已根据 {matched} 平台给出适合游玩的游戏推荐。",
    }


@mcp.tool
def recommend_games_by_preference(preference: str):
    """根据用户偏好推荐对应类型和游戏。"""
    preference = preference.strip()
    if not preference:
        return {
            "success": False,
            "message": "请输入偏好，例如：休闲、竞技、剧情、联机、开放世界、策略。",
        }

    matched = _match_keyword(preference, PREFERENCE_GAME_MAP)
    if not matched:
        return {
            "success": True,
            "preference": preference,
            "matched_preference": "未匹配到明确偏好",
            "genres": ["休闲", "剧情", "联机"],
            "games": ["星露谷物语", "双人成行", "巫师3"],
            "message": "暂未识别到你的偏好，先给你一组比较通用的高口碑游戏。",
        }

    result = PREFERENCE_GAME_MAP[matched]
    return {
        "success": True,
        "preference": preference,
        "matched_preference": matched,
        "genres": result["genres"],
        "games": result["games"],
        "message": f"已根据 {matched} 偏好给你推荐对应游戏。",
    }
