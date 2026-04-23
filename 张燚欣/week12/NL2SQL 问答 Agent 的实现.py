import sqlite3
import re
import os


class SQLAgent:
    """Chinook 数据库纯规则 NL2SQL Agent（适配单数表名）"""

    def __init__(self, db_path="chinook.db"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                "找不到 chinook.db！\n"
                "请将下载的 Chinook_Sqlite.sqlite 改名为 chinook.db 并放在与本脚本相同的文件夹里。"
            )
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # 获取所有业务表名（过滤掉 sqlite_ 内部表）
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        self.tables = [r[0] for r in self.cursor.fetchall()]

        # 检查关键表是否存在（根据实际表名 Employee / Customer）
        required = {"Employee", "Customer"}
        missing = required - set(self.tables)
        if missing:
            self.conn.close()
            raise RuntimeError(f"缺少关键表: {missing}，请使用完整的 Chinook 数据库。")

        # 中文表名 -> 实际英文表名（适配你的 chinook.db 的单数形式）
        self.cn_map = {
            "员工": "Employee",
            "客户": "Customer",
            "发票": "Invoice",
            "发票明细": "InvoiceLine",
            "曲目": "Trace",          # 你的数据库里是 Trace，不是 Tracks
            "专辑": "Album",
            "艺术家": "Artist",
            "流派": "Genre",
            "媒体类型": "MediaType",
            "播放列表": "Playlist",
            "播放列表曲目": "PlaylistTrack"
        }

    def _exec(self, sql):
        """执行 SQL 并返回所有结果"""
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"[SQL错误] {e}")
            return None

    def _generate_sql(self, question: str) -> str | None:
        q = question.lower()

        # 问题3（优先）：同时问了客户和员工个数
        if "客户" in q and "员工" in q:
            return ("SELECT "
                    "(SELECT COUNT(*) FROM Customer) AS customers, "
                    "(SELECT COUNT(*) FROM Employee) AS employees;")

        # 问题2：某一张表有多少条记录（如“员工表中有多少条记录”）
        for cn, en in self.cn_map.items():
            if cn in q and re.search(r"(多少|几条|数量|数)", q):
                return f"SELECT COUNT(*) FROM {en};"

        # 问题1：总共有多少张表（只统计业务表）
        if re.search(r"(多少|几张|数量).*表", q) or ("表" in q and "多少" in q):
            return "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"

        return None

    def ask(self, question: str) -> str:
        sql = self._generate_sql(question)
        if not sql:
            return "抱歉，我无法理解您的问题，请换一种问法。"
        print(f"[执行 SQL] {sql}")
        result = self._exec(sql)
        if result is None:
            return "查询执行失败。"
        if not result:
            return "查询结果为空。"
        # 单行单列
        if len(result) == 1 and len(result[0]) == 1:
            return f"结果是：{result[0][0]}。"
        # 单行多列
        if len(result) == 1 and len(result[0]) > 1:
            cols = [d[0] for d in self.cursor.description]
            parts = [f"{c}：{v}" for c, v in zip(cols, result[0])]
            return f"查询结果：{'，'.join(parts)}。"
        return str(result)

    def close(self):
        self.conn.close()


# ── 测试 ──
if __name__ == "__main__":
    try:
        agent = SQLAgent("chinook.db")
    except Exception as e:
        print(f"初始化失败：{e}")
        exit(1)

    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]

    for q in questions:
        print(f"\n👤 用户提问：{q}")
        print(f"🤖 Agent 回答：{agent.ask(q)}")

    agent.close()