from fastmcp import FastMCP

mcp = FastMCP(
    name="Medicine-Recommendation-MCP-Server",
    instructions="""This server provides medicine recommendations by disease, region and symptoms.""",
)

MEDICAL_NOTICE = (
    "以下内容仅供常见家庭用药参考，不能替代医生诊断。孕妇、儿童、老人、慢病患者或症状严重时，请及时线下就医。"
)

COMMON_DISEASE_MEDICINES = {
    "感冒": {
        "medicines": ["感冒灵颗粒", "氨咖黄敏胶囊", "板蓝根颗粒"],
        "advice": "注意休息、多喝温水；若持续高烧或症状加重，建议就医。",
    },
    "发烧": {
        "medicines": ["布洛芬", "对乙酰氨基酚"],
        "advice": "体温超过 38.5 摄氏度可考虑退烧药；若高热不退，请及时就医。",
    },
    "咳嗽": {
        "medicines": ["川贝枇杷膏", "右美沙芬", "氨溴索口服液"],
        "advice": "若咳嗽超过一周或伴随胸闷气短，建议尽快检查。",
    },
    "咽喉痛": {
        "medicines": ["西瓜霜含片", "金嗓子喉片", "蒲地蓝消炎口服液"],
        "advice": "避免辛辣刺激食物，注意补水；若吞咽困难或反复发作，建议就医。",
    },
    "头痛": {
        "medicines": ["布洛芬", "对乙酰氨基酚", "正天丸"],
        "advice": "先排查熬夜、疲劳等因素；若频繁头痛或剧烈头痛，请及时就医。",
    },
    "腹泻": {
        "medicines": ["蒙脱石散", "口服补液盐", "益生菌"],
        "advice": "重点是补水补盐；若腹泻严重、便血或持续时间长，建议就医。",
    },
    "胃痛": {
        "medicines": ["铝碳酸镁", "奥美拉唑", "胃苏颗粒"],
        "advice": "避免空腹和刺激性饮食；若反复胃痛或黑便，需尽快就医。",
    },
    "过敏": {
        "medicines": ["氯雷他定", "西替利嗪", "炉甘石洗剂"],
        "advice": "先远离过敏源；若出现呼吸困难、喉头紧缩等情况，应立即就医。",
    },
    "晕车": {
        "medicines": ["茶苯海明", "晕车贴"],
        "advice": "出行前服用更合适，避免空腹或过饱乘车。",
    },
}

REGIONAL_COMMON_MEDICINES = {
    "北京": ["感冒灵颗粒", "布洛芬", "氯雷他定", "蒙脱石散"],
    "上海": ["对乙酰氨基酚", "奥美拉唑", "西瓜霜含片", "炉甘石洗剂"],
    "广东": ["藿香正气水", "蒙脱石散", "板蓝根颗粒", "氯雷他定"],
    "广州": ["藿香正气水", "清热解毒口服液", "蒙脱石散", "布洛芬"],
    "深圳": ["藿香正气水", "布洛芬", "西替利嗪", "肠胃宁片"],
    "四川": ["健胃消食片", "蒙脱石散", "金嗓子喉片", "布洛芬"],
    "成都": ["健胃消食片", "蒲地蓝消炎口服液", "蒙脱石散", "川贝枇杷膏"],
    "重庆": ["健胃消食片", "蒙脱石散", "西瓜霜含片", "布洛芬"],
    "湖南": ["清热解毒颗粒", "肠炎宁", "蒙脱石散", "西瓜霜含片"],
    "湖北": ["感冒灵颗粒", "藿香正气水", "川贝枇杷膏", "布洛芬"],
    "浙江": ["氯雷他定", "西替利嗪", "布洛芬", "奥美拉唑"],
    "江苏": ["感冒灵颗粒", "蒲地蓝消炎口服液", "蒙脱石散", "对乙酰氨基酚"],
    "东北": ["感冒灵颗粒", "板蓝根颗粒", "布洛芬", "川贝枇杷膏"],
    "云南": ["藿香正气水", "蒙脱石散", "氯雷他定", "布洛芬"],
}

SYMPTOM_MEDICINES = {
    "流鼻涕": {
        "possible_issue": "普通感冒或过敏性鼻炎",
        "medicines": ["氯雷他定", "感冒灵颗粒", "西替利嗪"],
    },
    "鼻塞": {
        "possible_issue": "感冒、鼻炎",
        "medicines": ["感冒灵颗粒", "氯雷他定", "生理盐水喷雾"],
    },
    "咳嗽": {
        "possible_issue": "上呼吸道感染或咽喉刺激",
        "medicines": ["川贝枇杷膏", "右美沙芬", "氨溴索口服液"],
    },
    "喉咙痛": {
        "possible_issue": "咽炎、上火或感冒",
        "medicines": ["西瓜霜含片", "金嗓子喉片", "蒲地蓝消炎口服液"],
    },
    "发热": {
        "possible_issue": "感染性疾病常见表现",
        "medicines": ["布洛芬", "对乙酰氨基酚"],
    },
    "头痛": {
        "possible_issue": "疲劳、感冒、紧张性头痛",
        "medicines": ["布洛芬", "对乙酰氨基酚", "正天丸"],
    },
    "拉肚子": {
        "possible_issue": "饮食不洁、肠胃炎",
        "medicines": ["蒙脱石散", "口服补液盐", "益生菌"],
    },
    "恶心": {
        "possible_issue": "胃肠不适、晕车或消化问题",
        "medicines": ["藿香正气水", "健胃消食片", "茶苯海明"],
    },
    "胃胀": {
        "possible_issue": "消化不良",
        "medicines": ["健胃消食片", "铝碳酸镁", "奥美拉唑"],
    },
    "皮肤瘙痒": {
        "possible_issue": "过敏或皮肤刺激",
        "medicines": ["氯雷他定", "西替利嗪", "炉甘石洗剂"],
    },
}


def _match_region(address: str) -> str | None:
    for region in REGIONAL_COMMON_MEDICINES:
        if region in address:
            return region
    return None


def _match_keyword(text: str, mapping: dict) -> str | None:
    for keyword in mapping:
        if keyword in text:
            return keyword
    return None


@mcp.tool
def recommend_medicine_by_common_disease(disease: str):
    """根据常见病推荐常用药品。"""
    disease = disease.strip()
    if not disease:
        return {
            "success": False,
            "message": "请输入常见病名称，例如：感冒、发烧、咳嗽、腹泻。",
        }

    matched = _match_keyword(disease, COMMON_DISEASE_MEDICINES)
    if not matched:
        return {
            "success": True,
            "disease": disease,
            "medicines": ["布洛芬", "蒙脱石散", "氯雷他定"],
            "advice": "未匹配到明确病名，建议结合具体症状或咨询医生后再用药。",
            "notice": MEDICAL_NOTICE,
        }

    result = COMMON_DISEASE_MEDICINES[matched]
    return {
        "success": True,
        "disease": disease,
        "matched_disease": matched,
        "medicines": result["medicines"],
        "advice": result["advice"],
        "notice": MEDICAL_NOTICE,
    }


@mcp.tool
def recommend_regional_common_medicines(address: str):
    """根据不同地区推荐常备药品。"""
    address = address.strip()
    if not address:
        return {
            "success": False,
            "message": "请输入地区信息，例如：广东深圳、北京朝阳区、四川成都。",
        }

    matched_region = _match_region(address)
    if not matched_region:
        return {
            "success": True,
            "address": address,
            "matched_region": "未匹配到特定地区",
            "medicines": ["感冒灵颗粒", "布洛芬", "蒙脱石散", "氯雷他定"],
            "advice": "未识别到具体地区，先提供较通用的家庭常备药建议。",
            "notice": MEDICAL_NOTICE,
        }

    return {
        "success": True,
        "address": address,
        "matched_region": matched_region,
        "medicines": REGIONAL_COMMON_MEDICINES[matched_region],
        "advice": f"已根据 {matched_region} 的气候和常见用药习惯给出常备药建议。",
        "notice": MEDICAL_NOTICE,
    }


@mcp.tool
def recommend_medicine_by_symptom(symptom: str):
    """根据用户症状推荐对应药品。"""
    symptom = symptom.strip()
    if not symptom:
        return {
            "success": False,
            "message": "请输入症状描述，例如：流鼻涕、咳嗽、喉咙痛、拉肚子。",
        }

    matched = _match_keyword(symptom, SYMPTOM_MEDICINES)
    if not matched:
        return {
            "success": True,
            "symptom": symptom,
            "possible_issue": "症状未明确匹配",
            "medicines": ["请先测量体温", "请补充完整症状"],
            "advice": "建议补充持续时间、是否发热、是否疼痛等信息，必要时及时就医。",
            "notice": MEDICAL_NOTICE,
        }

    result = SYMPTOM_MEDICINES[matched]
    return {
        "success": True,
        "symptom": symptom,
        "matched_symptom": matched,
        "possible_issue": result["possible_issue"],
        "medicines": result["medicines"],
        "advice": "建议严格按照说明书用药；若出现持续高热、呼吸困难、剧烈疼痛等情况，请立即就医。",
        "notice": MEDICAL_NOTICE,
    }
