import os
import sqlite3
from openai import OpenAI

# =========================
# 配置千问
# =========================
os.environ["OPENAI_API_KEY"] = "sk-51e0dfbaca164esab8abcdbd3c201857"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


class ChinookAgent:

    def __init__(self, db_path="chinook.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    # -------------------------
    # 获取真实 schema
    # -------------------------
    def get_schema(self):

        self.cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)

        tables = [x[0] for x in self.cursor.fetchall()]

        schema = []

        for table in tables:
            self.cursor.execute(f"PRAGMA table_info('{table}')")
            cols = self.cursor.fetchall()

            col_names = [c[1] for c in cols]

            schema.append(
                f"表名: {table}\n字段: {', '.join(col_names)}"
            )

        return "\n\n".join(schema)

    # -------------------------
    # 生成 SQL
    # -------------------------
    def nl2sql(self, question):

        schema = self.get_schema()

        prompt = f"""
你是 SQLite 专家。

下面是数据库真实表结构（必须严格使用这些表名）：

{schema}

要求：
1. 只能使用上面出现过的表名
2. 表名区分单复数，禁止猜测
3. 只输出 SQL
4. 不解释
5. SQLite语法

用户问题：
{question}
"""

        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        sql = resp.choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()

        return sql

    # -------------------------
    # 执行 SQL
    # -------------------------
    def run_sql(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    # -------------------------
    # 对外问答
    # -------------------------
    def ask(self, question):
        try:
            sql = self.nl2sql(question)
            result = self.run_sql(sql)

            return {
                "question": question,
                "sql": sql,
                "result": result
            }

        except Exception as e:
            return {
                "question": question,
                "error": str(e)
            }


# =========================
# 测试
# =========================
agent = ChinookAgent("chinook.db")

questions = [
    "数据库中总共有多少张表",
    "员工表中有多少条记录",
    "在数据库中所有客户个数和员工个数分别是多少"
]

for q in questions:
    res = agent.ask(q)

    print("问题：", q)

    if "error" in res:
        print("错误：", res["error"])
    else:
        print("SQL：", res["sql"])
        print("结果：", res["result"])

    print("-" * 60)