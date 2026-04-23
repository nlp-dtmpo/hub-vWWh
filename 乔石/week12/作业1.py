import sqlite3
import traceback
from typing import Union, Tuple, List, Any

import pandas as pd
from sqlalchemy import create_engine, inspect, Table, MetaData, select, func, Row, text
from openai import OpenAI


class DBParser:
    """数据库解析"""

    def __init__(self, db_url):
        """
        初始化：传入数据库地址
        """
        self.db_url = db_url

        # 建立数据库链接
        self.engine = create_engine(self.db_url, echo=False)
        self.connect = self.engine.connect()

        # 查看表信息
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()
        print(f"所有表：{self.table_names}")

        self._table_fields = {}
        self.foreign_keys = []
        self._table_sample = {}

        for table_name in self.table_names:
            print(f"Table: {table_name}")
            self._table_fields[table_name] = {}

            # 累计外键
            self.foreign_keys += [
                {
                    "constrained_table": table_name,
                    "constrained_columns": x['constrained_columns'],
                    "referred_table": x["referred_table"],
                    "referred_columns": x["referred_columns"]
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x["name"]: x for x in table_columns}

            # 对当前字段进行统计
            for column_meta in table_columns:
                column_instance = getattr(table_instance.columns, column_meta["name"])

                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.connect.execute(query).fetchone()[0]
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
                    top1_value = self.connect.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计missing个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.connect.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计max
                query = select(func.max(column_instance))
                max_value = self.connect.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 统计min
                query = select(func.min(column_instance))
                min_value = self.connect.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                # 任意取值
                query = select(column_instance).limit(10)
                random_value = self.connect.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None]
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            # 获取表样例（第一行）
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.connect.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        """获取数据表样例"""
        return self._table_sample[table_name]

    def check_sql(self, sql) -> tuple[bool, str]:
        """检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        """
        try:
            sql_text = text(sql)
            self.connect.execute(sql_text)
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> Any | None:
        """运行SQL"""
        try:
            sql_text = text(sql)
            with self.engine.connect() as conn:
                result = conn.execute(sql_text).fetchone()[0]
                return result
        except Exception as e:
            print(f"execute sql error: {str(e)}")
            return None


class SQLAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-***",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.parser = DBParser("sqlite:///./chinook.db")
        self.sql_generation_prompt = '''你是一个专业的数据库专家，现在需要你结合数据库的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：

        数据库包含以下表：{table_names}

        各表的主要字段如下：
        {table_schemas}

        提问：{question}

        请生成对应的SQL查询语句：
        '''
        self.answer_rewrite_prompt = '''你是一个专业的数据库专家，将下面的问题回答组织为自然语言。：

        原始问题：{question}

        执行SQL：{sql}

        原始结果：{answer}
        '''

    def get_database_info(self):
        table_names = ", ".join(self.parser.table_names)
        table_schemas = ""
        for table_name in self.parser.table_names:
            fields_df = self.parser.get_table_fields(table_name)
            fields_info = []
            for field_name in fields_df.index:
                field_type = fields_df.loc[field_name, 'type']
                fields_info.append(f"{field_name} ({field_type})")
            table_schemas += f"{table_name}: {', '.join(fields_info)}\n"

        return table_names, table_schemas

    def generate_sql(self, question):
        table_names, table_schemas = self.get_database_info()
        prompt = self.sql_generation_prompt.format(
            table_names=table_names,
            table_schemas=table_schemas,
            question=question
        )
        sql = self.ask_qwen(prompt)
        if sql:
            # 清理SQL格式
            sql = sql.strip('`').strip('\n').replace('sql\n', '')
            return sql
        return None

    def execute_query(self, sql):
        try:
            result = self.parser.execute_sql(sql)
            return result
        except Exception as e:
            return f"error execute sql: {str(e)}"

    def generate_answer(self, question, sql, result):
        if isinstance(result, list):
            result_str = str(result)
        else:
            result_str = str(result)
        prompt = self.answer_rewrite_prompt.format(
            question=question,
            sql=sql,
            answer=result_str
        )
        response = self.ask_qwen(prompt)
        if response:
            return response
        return "无法生成回答"

    def ask_qwen(self, prompt, retry=1):
        if retry == 0:
            return None
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"ask qwen error: {str(e)}")
            return self.ask_qwen(prompt, retry - 1)

    def answer_question(self, question):
        sql = self.generate_sql(question)
        if not sql:
            return "无法生成sql"

        result = self.execute_query(sql)

        answer = self.generate_answer(question, sql, result)

        return answer


if __name__ == "__main__":
    sqlAgent = SQLAgent()
    question1 = "数据库总共有多少张表？"
    answer1 = sqlAgent.answer_question(question1)
    print(f"问题1: {question1}")
    print(f"回答: {answer1}\n")

    question2 = "员工表中有多少条记录？"
    answer2 = sqlAgent.answer_question(question2)
    print(f"问题2: {question2}")
    print(f"回答: {answer2}\n")

    question3 = "在数据库中所有客户个数和员工个数分别是多少？"
    answer3 = sqlAgent.answer_question(question3)
    print(f"问题3: {question3}")
    print(f"回答: {answer3}\n")
