###########################################################
# 以下是新增工具，用于补充实现企业职能助手

@mcp.tool
def query_leave_balance(user_name: Annotated[str, "用户名"]):
    """Query annual leave balance based on the username."""
    if len(user_name) == 2:
        return {"user_name": user_name, "annual_leave_days_left": 5}
    elif len(user_name) == 3:
        return {"user_name": user_name, "annual_leave_days_left": 8}
    else:
        return {"user_name": user_name, "annual_leave_days_left": 10}


@mcp.tool
def query_payday():
    """Query monthly payday policy."""
    return {"payday": "每月10号", "remark": "遇节假日提前发放"}


@mcp.tool
def create_meeting_summary(
    topic: Annotated[str, "会议主题"],
    notes: Annotated[str, "会议原始记录"],
):
    """Create a simple meeting summary based on topic and notes."""
    summary = notes[:120] + ("..." if len(notes) > 120 else "")
    return {
        "topic": topic,
        "summary": summary,
        "action_items": [
            "确认需求优先级",
            "分配负责人并设定截止时间",
            "下次例会同步进展",
        ],
    }
