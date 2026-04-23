# Week12 作业：真实 NL2SQL Agent（纯 Python）

## 目标
基于 `chinook.db` 完成一个最小可运行问答 Agent（NL2SQL），并能回答：

1. 数据库中总共有多少张表；
2. 员工表中有多少条记录；
3. 在数据库中所有客户个数和员工个数分别是多少。

## 文件说明
- `sql_nl2sql_agent.py`：主程序。包含真实 LLM 调用、schema 读取、SQL 生成/重试、SQL 执行、答案组织和交互入口。

## 运行方式
在 `Week12/hw/` 目录执行：

```bash
python sql_nl2sql_agent.py
```

## 先改 API Key
在 `sql_nl2sql_agent.py` 中把下面占位符替换为真实 key：

- `PLACEHOLDER_API_KEY = "YOUR_API_KEY_HERE"`

也可以在初始化 `ChinookNL2SQLAgent` 时传入你自己的 `api_key`、`base_url`、`model`。

## 代码结构（按函数理解）
- `ChinookNL2SQLAgent`：核心 Agent 类。
  - `get_schema_text() -> str`：自动读取数据库 schema，给 LLM 作为上下文。
  - `nl2sql(...) -> str`：调用 LLM 生成 SQL。
  - `ensure_readonly_sql(sql: str)`：只允许 `SELECT/WITH`，防止误写库。
  - `execute_sql(sql: str) -> List[Tuple[Any, ...]]`：执行 SQL 并返回结果。
  - `summarize_answer(...) -> str`：调用 LLM 把 SQL 结果转成自然语言回答。
  - `ask(question: str) -> AgentResponse`：总入口，串起完整流程。
- `get_default_db_path() -> Path`：自动定位 `chinook.db` 的默认路径。
- `run_interactive_demo() -> None`：交互式提问，可输入任意数据库相关问题。

## 为什么这样设计
- 这是“真正的 NL2SQL”：不是固定三问，而是按问题实时生成 SQL。
- 带有报错重试机制：第一次 SQL 失败时，会把错误反馈给模型再次生成。
- 读写安全可控：内置只读 SQL 检查，避免破坏数据库。
- 用类型标注和数据类 `AgentResponse`，方便你阅读和二次修改。

## 可扩展方向
- 加入“字段释义/术语库”作为额外 prompt 上下文，提升业务问题准确率。
- 扩展多轮对话记忆（把上轮问题和 SQL 加入消息历史）。
- 增加结果可视化（例如自动转成 DataFrame 并展示前 N 行）。
