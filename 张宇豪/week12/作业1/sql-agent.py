'''数据库解析'''
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text  # ORM 框架
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
            with self.engine.connect() as conn:
                conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                return list(result)
        except Exception as e:
            print(f"执行报错：{e}")
            return []


from openai import OpenAI
import re
import os

class SQLAgent:
    def __init__(self, db_url, api_key):
        self.parser = DBParser(db_url)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def run(self, question):
        schema_info = ""
        for table in self.parser.table_names:
            schema_info += f"表名: {table}\n{self.parser.get_table_fields(table).to_markdown()}\n\n"

        gen_sql_prompt = f"""
        你是一个 SQL 专家。根据以下数据库结构：
        {schema_info}
        回答用户问题：{question}
        请直接输出 SQL，不要解释。
        """

        sql = self.ask_qwen(gen_sql_prompt)
        sql = self.clean_sql(sql)
        print(sql)

        if self.parser.check_sql(sql):
            try:
                db_result = self.parser.execute_sql(sql)

                final_prompt = f"""
                用户问题：{question}
                执行 SQL：{sql}
                查询结果：{db_result}
                请根据查询结果用自然语言回答用户。
                """
                return self.ask_qwen(final_prompt)
            except Exception as e:
                return f"抱歉，我生成的 SQL 出错了：{e}"
        else:
            print("抱歉，我生成的 SQL 有错误")

    def ask_qwen(self, question, nretry=5):
        if nretry == 0:
            return None

        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据库专家。"},
                    {"role": "user", "content": question}
                ],
                temperature=0.5
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"请求失败，正在重试... 错误原因：{e}")
            return self.ask_qwen(question, nretry - 1)

    def clean_sql(self, text):
        sql_match = re.search(r"```sql\n?(.*?)\n?```", text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return text.strip().strip('`').replace('sql\n', '')

sql_agent = SQLAgent('sqlite:///./chinook.db', os.getenv('OPENAI_API_KEY'))
while True:
    user_input = input('输入你想问的问题：')
    if user_input.lower() == 'quit' or user_input.lower() == 'exit':
        break
    else:
        print(sql_agent.run(user_input))
