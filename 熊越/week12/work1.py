import sqlite3
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData  # ORM 框架


class DB:
    def __init__(self):
        self.conn = sqlite3.connect('chinook.db')

    def get_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"chinook.db总共{len(tables)}张表")

    def get_table_columns(self, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"{table_name}表有{row_count}行数据")

    def get_table_data(self, cus_table_name,employees_table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT  COUNT(*) FROM {cus_table_name};")
        data = cursor.fetchall()[0]
        print(f"客户表{cus_table_name}表共计{data}行数据：")

        cursor.execute(f"SELECT  COUNT(*) FROM {employees_table_name};")
        data = cursor.fetchall()[0]
        print(f"员工表{cus_table_name}表共计{data}行数据：")


if __name__ == '__main__':
    db = DB()
    db.get_tables()
    db.get_table_columns(table_name="employees")
    db.get_table_data(employees_table_name="employees",cus_table_name="customers")
