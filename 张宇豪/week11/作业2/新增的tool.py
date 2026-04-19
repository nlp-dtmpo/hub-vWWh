@mcp.tool
def get_holiday_info(date_text: Annotated[str, "The date to query (format: 'yyyy-MM-dd', e.g., '2026-05-01')"]):
    """Checks if a specific date is a holiday, workday, or weekend in China."""
    try:
        # 查询指定日期的节假日安排
        return requests.get(f"https://whyta.cn/api/tx/jiejiari?key={TOKEN}&date={date_text}").json()["result"]
    except:
        return []

  @mcp.tool
def get_music_comment_saying():
    """Retrieves a popular or emotional music comment (often called 'Netease Cloud Music Comment') from the API."""
    try:
        # 获取随机一条网易云热评
        return requests.get(f"https://whyta.cn/api/tx/hotreview?key={TOKEN}").json()["result"]["content"]
    except:
        return []

  @mcp.tool
def get_weibo_hot_news():
    """Retrieves a list of trending topics or hot search items from Weibo using the API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/weibohot?key={TOKEN}").json()["result"]["list"]
    except:
        return []
