import sqlite3 # py 自带的模块
import time
import jwt
import requests
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text # ORM 框架
from itertools import combinations
from tqdm import tqdm
from typing import Union
import numpy as np
import traceback
import pandas as pd
import re
# sqlite 单机关系型数据库

# 连接到Chinook数据库
conn = sqlite3.connect('chinook.db') # 数据库文件，包含多张表

"""
sqlite vs mysql
- sqlite 单机数据库，嵌入式数据库，和编程语言集成完善，不需要额外安装软件；
- sqlite 默认是一个文件单数据库，mysql 支持 多数据库；
- sqlite 默认是本地使用，并且是单线程使用，不支持远程连接；
"""

# 创建一个游标对象
cursor = conn.cursor()

# 获取数据库中所有表的名称
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

'''数据库解析'''

class DBParser:
    '''DBParser 数据库的解析'''
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        mysql: mysql://root:123456@localhost:3306/mydb?charset=utf8mb4
        sqlite: sqlite://chinook.db
        '''

        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 链接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 查看表名
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names() # 获取table信息

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            # 累计外键（获取并存储外键信息）
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息（反射表对象和字段元数据）
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

            # 对当前字段进行统计
            for column_meta in table_columns:
                # 获取当前字段
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计unique
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 统计most frequency value
                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计missing个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计max
                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 统计min
                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                # 任意取值
                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None]
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            # 获取表样例（第一行）
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]

    def check_sql(self, sql) -> Union[bool, str]:
        '''检查sql是否合理
        返回: (是否可以运行, 报错信息)
        '''
        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql))   # 关键：使用 text() 包装
            return True, 'ok'
        except Exception:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> list:
        '''运行SQL并返回结果（列表形式）'''
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))   # 关键：使用 text() 包装
            if result.returns_rows:
                return [list(row) for row in result.fetchall()]
            else:
                conn.commit()
                return []

parser = DBParser('sqlite:///./chinook.db')

# 生成数据库结构描述（给大模型用）
def get_db_schema() -> str:
    """生成数据库结构的文本描述（用于给 LLM 理解）"""
    schema = "SQLite 数据库 chinook.db 包含以下表：\n"
    for table in parser.table_names:
        schema += f"\n表名：{table}\n"
        fields = parser.get_table_fields(table)
        for idx, row in fields.iterrows():
            # 字段基本信息
            field_desc = f"  - {row['name']} ({row['type']})"
            if row['primary_key']:
                field_desc += " 主键"
            if row['nullable']:
                field_desc += " 可空"
            # 添加示例值（如果有）
            sample_vals = row.get('random', [])
            if sample_vals and sample_vals[0] is not None:
                field_desc += f" 示例值：{sample_vals[0]}"
            schema += field_desc + "\n"
        # 添加外键关系
        fks = parser.get_data_relations()
        table_fks = fks[fks['constrained_table'] == table]
        for _, fk in table_fks.iterrows():
            schema += f"  外键：{fk['constrained_columns'][0]} 引用 {fk['referred_table']}.{fk['referred_columns'][0]}\n"
    return schema

# 缓存 schema 描述（避免重复生成）
DB_SCHEMA_CACHE = get_db_schema()


# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

def ask_glm(question, nretry=5):
    if nretry == 0:
        return None

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("04503c3c1ea845d9934516c87c684d12.Gu2MCAbs2Ze2WAXG", 1000)
    }
    data = {
        "model": "glm-3-turbo",
        "p": 0.5,
        "messages": [{"role": "user", "content": question}]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()
    except:
        return ask_glm(question, nretry-1)

def extract_sql_from_response(response_text: str) -> str:
    """从大模型返回结果中提取SQL语句"""
    # 匹配 ```sql ... ``` 格式
    pattern = r"```sql(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 匹配普通SQL
    sql_pattern = r"(SELECT.*?;)"
    match = re.search(sql_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return response_text.strip()

def nl2sql_agent(question: str, max_retries: int = 2) -> str:
    """
    通用 NL2SQL Agent：接收自然语言问题，返回自然语言答案
    内部调用三个子 Agent 协作完成
    """
    print(f"用户问题：{question}")
    print("正在生成SQL...")

    prompt = f"""
你是专业的SQL生成专家，只能基于下面的数据库结构生成可直接运行的SQLite语句。
要求：
1. 只返回SQL代码，不要解释，不要多余文字
2. 表名、字段名必须严格匹配下面结构
3. 统计数量用COUNT(*)
4. 结果必须是标准SQLite语法

数据库结构：
{DB_SCHEMA_CACHE}

用户问题：{question}
请生成SQL：
"""
    # 调用大模型
    resp = ask_glm(prompt)
    if "error" in resp or "choices" not in resp:
        return f"大模型调用失败：{resp}"
    
    answer = resp["choices"][0]["message"]["content"]
    sql = extract_sql_from_response(answer)

    print(f"生成SQL：{sql}")
    print("正在执行查询...")

    # 执行SQL
    result = parser.execute_sql(sql)
    print(f"查询结果：{result}")

     # 生成自然语言回答
    final_prompt = f"""
用户问题：{question}
SQL执行结果：{result}
请用简洁自然的中文回答问题，只返回答案，不要多余内容
"""
    final_resp = ask_glm(final_prompt)
    if "choices" in final_resp:
        return f"\n答案：{final_resp['choices'][0]['message']['content']}"
    else:
        return f"\n答案：{result}"


if __name__ == "__main__" or True:  # 在 notebook 中会执行
    # 问题1
    print("\n" + "-"*50)
    q1 = "数据库中总共有多少张表"
    print(nl2sql_agent(q1))
    
    # 问题2
    print("\n" + "-"*50)
    q2 = "员工表中有多少条记录"
    print(nl2sql_agent(q2))
    
    # 问题3
    print("\n" + "-"*50)
    q3 = "在数据库中所有客户个数和员工个数分别是多少"
    print(nl2sql_agent(q3))