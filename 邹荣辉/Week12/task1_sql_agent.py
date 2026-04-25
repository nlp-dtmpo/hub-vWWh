from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import pandas as pd
import requests

class DBParser:
    def __init__(self, db_url:str) -> None:
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        self._table_fields = {}
        self.foreign_keys = []
        self._table_sample = {}

        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

            for column_meta in table_columns:
                column_instance = getattr(table_instance.columns, column_meta['name'])

                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    try:
                        top1_value = self.conn.execute(query).fetchone()[0]
                        self._table_fields[table_name][column_meta['name']]['mode'] = top1_value
                    except:
                        pass

                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None]
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            query = select(table_instance).limit(5)
            try:
                rows = self.conn.execute(query).fetchall()
                self._table_sample[table_name] = pd.DataFrame(rows)
                self._table_sample[table_name].columns = [x['name'] for x in table_columns]
            except:
                pass

    def get_table_fields(self, table_name) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        return self._table_sample[table_name]

    def check_sql(self, sql) -> Union[bool, str]:
        try:
            self.conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql):
        result = self.conn.execute(text(sql))
        return list(result)

def ask_qwen(question, api_key, nretry=5):
    if nretry == 0:
        return None

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": "qwen-max",
        "temperature": 0.5,
        "messages": [{"role": "user", "content": question}]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        return response.json()
    except:
        return ask_qwen(question, api_key, nretry-1)

def main():
    api_key = "sk-08e082b30dbc43d1a83d9603d641e6d9"
    
    parser = DBParser('sqlite:///./chinook.db')
    
    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]
    
    planer_prompt = '''你是一个数据库专家，现在需要规划如何回答用户的问题。请分析问题需要查询哪些表和字段，不需要输出代码，只需要说明分析思路。问题：{question}'''
    
    answer_prompt = '''你是一个专业的数据库专家，现在需要你结合数据库信息，为以下问题生成对应的SQL语句。请直接输出SQL，不需要有其他输出：

数据库包含以下表：{table_names}

问题：{question}
'''
    
    question_rewrite_prompt = '''你是一个专业的数据库专家，现在需要你将下面的提问转换为更自然的用户提问风格。请直接输出改写后的提问，不需要有其他输出：

原始问题：{question}
'''
    
    answer_rewrite_prompt = '''你是一个专业的数据库专家，将下面的问题回答组织为自然语言：

原始问题：{question}

执行SQL：{sql}

原始结果：{answer}
'''

    for idx, question in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"提问{idx}: {question}")
        print('='*50)
        
        if "多少张表" in question:
            table_count = len(parser.table_names)
            print("\n【步骤1：直接统计】")
            print(f"数据库中共有 {table_count} 张表")
            
            print("\n【步骤2：改写提问】")
            rewrite_prompt = question_rewrite_prompt.format(question=question)
            rewrite_response = ask_qwen(rewrite_prompt, api_key)
            if rewrite_response and 'choices' in rewrite_response:
                rewritten_question = rewrite_response['choices'][0]['message']['content']
                print(f"改写后提问：{rewritten_question}")
            
            print("\n【步骤3：生成自然语言答案】")
            sql = "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            result_str = str([(table_count,)])
            nl_prompt = answer_rewrite_prompt.format(question=question, sql=sql, answer=result_str)
            nl_response = ask_qwen(nl_prompt, api_key)
            if nl_response and 'choices' in nl_response:
                final_answer = nl_response['choices'][0]['message']['content']
                print(f"最终答案：{final_answer}")
            
            print("\n" + '='*50 + "\n")
            continue
        
        print("\n【步骤1：规划分析】")
        plan_prompt = planer_prompt.format(question=question)
        plan_response = ask_qwen(plan_prompt, api_key)
        if plan_response and 'choices' in plan_response:
            plan_result = plan_response['choices'][0]['message']['content']
            print(f"分析思路：{plan_result}")
        
        print("\n【步骤2：生成SQL】")
        table_names_str = ", ".join(parser.table_names)
        sql_prompt = answer_prompt.format(table_names=table_names_str, question=question)
        sql_response = ask_qwen(sql_prompt, api_key)
        if sql_response and 'choices' in sql_response:
            sql = sql_response['choices'][0]['message']['content']
            sql = sql.strip('`').strip('\n').replace('sql\n', '')
            print(f"生成SQL：{sql}")
        
        print("\n【步骤3：验证SQL】")
        flag, msg = parser.check_sql(sql)
        if flag:
            print("SQL验证通过")
        else:
            print(f"SQL验证失败: {msg}")
            continue
        
        print("\n【步骤4：执行SQL】")
        sql_result = parser.execute_sql(sql)
        print(f"执行结果：{sql_result}")
        
        print("\n【步骤5：改写提问】")
        rewrite_prompt = question_rewrite_prompt.format(question=question)
        rewrite_response = ask_qwen(rewrite_prompt, api_key)
        if rewrite_response and 'choices' in rewrite_response:
            rewritten_question = rewrite_response['choices'][0]['message']['content']
            print(f"改写后提问：{rewritten_question}")
        
        print("\n【步骤6：生成自然语言答案】")
        result_str = str(sql_result) if len(sql_result) <= 3 else str(sql_result[:3]) + "..."
        nl_prompt = answer_rewrite_prompt.format(question=question, sql=sql, answer=result_str)
        nl_response = ask_qwen(nl_prompt, api_key)
        if nl_response and 'choices' in nl_response:
            final_answer = nl_response['choices'][0]['message']['content']
            print(f"最终答案：{final_answer}")
        
        print("\n" + '='*50 + "\n")

if __name__ == "__main__":
    main()
