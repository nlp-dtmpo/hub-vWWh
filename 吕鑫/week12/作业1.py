import sqlite3 # py 自带的模块
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
tables

----------
 ('sqlite_sequence',),
 ('artists',),
 ('customers',),
 ('employees',),
 ('genres',),
 ('invoices',),
 ('invoice_items',),
 ('media_types',),
 ('playlists',),
 ('playlist_track',),
 ('tracks',),
 ('sqlite_stat1',)]
-----------------
'''数据库解析'''
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text # ORM 框架
import pandas as pd

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

        # 查看表明
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names() # 获取table信息

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            # 累计外键
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
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

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            self.conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        result = self.conn.execute(text(sql))
        return list(result)

parser = DBParser('sqlite:///./chinook.db')
------------------
Table -> albums
Table -> artists
Table -> customers
Table -> employees
Table -> genres
Table -> invoice_items
Table -> invoices
Table -> media_types
Table -> playlist_track
Table -> playlists
Table -> tracks
-------------------
import time
import jwt
import requests
from itertools import combinations
import numpy as np
from tqdm import tqdm

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
      'Authorization': generate_token("899f3c20ae8f4d05bac395d7efe852c1.vefTY6c6AlMFP6sX", 1000)
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

  # 数据库拓展
# 用户提问的生成
question_prompt = '''你是一个专业的数据库专家，现在需要从用户的角度提问模拟生成一个提问。提问是自然语言，且计数和统计类型的问题，请直接输出具体提问，不需要有其他输出：

表名称：{table_name}

需要提问和统计的字段：{field}

表{table_name}样例如下：
{data_sample_mk}

表{table_name} schema如下：
{data_schema}
'''

answer_prompt = '''你是一个专业的数据库专家，现在需要你结合表{table_name}的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：

表名称：{table_name}

数据表样例如下：
{data_sample_mk}

数据表schema如下：
{data_schema}

提问：{question}
'''

question_rewrite_prompt = '''你是一个专业的数据库专家，现在需要从用户的角度提问模拟生成一个提问。现在需要你将的下面的提问，转换为用户提问的风格。请直接输出提问，不需要有其他输出，不要直接提到表明：

原始问题：{question}

查询的表：{table_name}
'''

answer_rewrite_prompt = '''你是一个专业的数据库专家，将下面的问题回答组织为自然语言。：

原始问题：{question}

执行SQL：{sql}

原始结果：{answer}
'''

company_name_rewrite_prompt = '''将下面的公司的中文缩写名称，如剔除公司名称中的地域信息，或剔除公司名中的有限责任公司等信息。不要输出其他内容，不是英文缩写名称。

原始公司名：{company_name}
'''

gt_qes_answer = []

# 对于每张表
for table_name in parser.table_names[:50]:
    # 表样例
    data_sample = parser.get_table_sample(table_name)
    data_sample_mk = data_sample.to_string(index=False)

    # 表格式
    data_schema = parser.get_table_fields(table_name).to_string()
    data_fields = list(data_sample.columns)

    # 待选字段
    candidate_fields = list(data_fields) + list(combinations(data_fields, 2)) + list(combinations(data_fields, 3))
    candidate_fields = [' 和 '.join(x) if isinstance(x, tuple) else x for x in candidate_fields]
    pool1, pool2 = candidate_fields[:20], candidate_fields[20:]
    sampled = (
        list(np.random.choice(pool1, min(8, len(pool1)), replace=False)) if pool1 else []
    ) + (
        list(np.random.choice(pool2, min(6, len(pool2)), replace=False)) if pool2 else []
    )
    candidate_fields = sampled

    # 对每个待选字段生成查询逻辑
    for field in tqdm(candidate_fields[:]):
        # 重试次数
        for _ in range(5):
            # 生成提问
            try:
                input_str = question_prompt.format(table_name=table_name, data_sample_mk=data_sample_mk, data_schema=data_schema, field=field)
                question = ask_glm(input_str)['choices'][0]['message']['content']

                # 生成答案SQL
                input_str = answer_prompt.format(table_name=table_name, data_sample_mk=data_sample_mk, data_schema=data_schema, question=question)
                answer = ask_glm(input_str)['choices'][0]['message']['content']
                answer = answer.strip('`').strip('\n').replace('sql\n', '')

                # 判断SQL是否符合逻辑
                flag, _ = parser.check_sql(answer)
                if not flag:
                    continue

                # 获取SQL答案
                sql_answer = parser.execute_sql(answer)
                if len(sql_answer) > 1:
                    continue
                sql_answer = sql_answer[0]
                sql_answer = ' '.join([str(x) for x in sql_answer])

                # 将提问改写，更加符合用户风格
                input_str = question_rewrite_prompt.format(question=question, table_name=table_name)
                question = ask_glm(input_str)['choices'][0]['message']['content']

                # 将SQL和结果改为为自然语言
                input_str = answer_rewrite_prompt.format(question=question, sql=answer, answer=sql_answer)
                nl_answer = ask_glm(input_str)['choices'][0]['message']['content']

                gt_qes_answer.append([
                    question, table_name, answer, sql_answer, nl_answer
                ])
                break

            except:
                continue



  ## NL2SQL Agent —— 基于 chinook.db 的自然语言问答
# ── NL2SQL Agent（纯 LLM 驱动）────────────────────────
from openai import OpenAI

client = OpenAI(
    api_key="sk-35609c8a5c4e42c6bc8b38888615c54b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen-plus"

nl2sql_prompt = '''你是一个专业的 SQLite 数据库专家。请根据数据库结构和用户问题生成正确的 SQL。
要求：
1. 只输出 SQL 语句本身
2. 不要输出解释、注释、markdown 代码块
3. 优先使用最简单、最直接的 SQL

数据库结构：
{schema}

用户问题：{question}
'''

def build_schema():
    '''把所有表的字段信息拼成文本，作为 LLM 的数据库上下文。'''
    lines = []
    for table_name in parser.table_names:
        fields = parser.get_table_fields(table_name).reset_index()[['name', 'type']]
        field_text = '\n'.join(
            [f"- {row['name']}: {row['type']}" for _, row in fields.iterrows()]
        )
        lines.append(f"表 {table_name}\n{field_text}")
    return "\n\n".join(lines)

DB_SCHEMA = build_schema()

def call_llm(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def clean_sql(sql):
    sql = sql.strip()
    if sql.startswith('```'):
        sql = sql.strip('`')
        if sql.lower().startswith('sql'):
            sql = sql[3:].strip()
    sql = sql.replace('```', '').strip()
    if sql.lower().startswith('sql'):
        sql = sql[3:].strip()
    return sql

def nl2sql_agent(user_question):
    # Step 1: LLM 生成 SQL
    prompt = nl2sql_prompt.format(schema=DB_SCHEMA, question=user_question)
    try:
        sql = clean_sql(call_llm(prompt))
    except Exception as e:
        print(f"生成 SQL 失败: {type(e).__name__}: {e}")
        return

    # Step 2: 执行 SQL
    try:
        sql_answer = parser.execute_sql(sql)
    except Exception as e:
        print(f"SQL 执行失败: {type(e).__name__}: {e}")
        print(f"生成的 SQL: {sql}")
        return

    # Step 3: LLM 组织自然语言答案
    answer_prompt = answer_rewrite_prompt.format(
        question=user_question,
        sql=sql,
        answer=str(sql_answer)
    )
    try:
        nl_answer = call_llm(answer_prompt)
    except Exception as e:
        print(f"生成回答失败: {type(e).__name__}: {e}")
        print(f"SQL: {sql}")
        print(f"结果: {sql_answer}")
        return

    print(f"问题: {user_question}")
    print(f"SQL : {sql}")
    print(f"结果: {sql_answer}")
    print(f"回答: {nl_answer}")
    print('-' * 60)

print('nl2sql_agent 定义完成')
------------------
请输入问题，直接回车或输入 exit 结束。
请输入你的问题： 数据库有几张表
问题: 数据库有几张表
SQL : SELECT COUNT(*) FROM sqlite_master WHERE type = 'table';
结果: [(13,)]
回答: 该数据库中共有 13 张表。
------------------------------------------------------------
请输入你的问题： 员工表有几条记录
问题: 员工表有几条记录
SQL : SELECT COUNT(*) FROM employees;
结果: [(8,)]
回答: 员工表中共有 8 条记录。
------------------------------------------------------------
请输入你的问题： 在数据库中所有客户个数和员工个数分别是多少
问题: 在数据库中所有客户个数和员工个数分别是多少
SQL : SELECT (SELECT COUNT(*) FROM customers) AS customer_count, (SELECT COUNT(*) FROM employees) AS employee_count;
结果: [(59, 8)]
回答: 在该数据库中，客户总数为59人，员工总数为8人。
------------------------------------------------------------
quit
------------------------------------------------------------
