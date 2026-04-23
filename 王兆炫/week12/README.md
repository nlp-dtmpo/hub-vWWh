本次作业完成以下的内容：


1: 参考sql agent，实现一下基于 chinook.db 数据集进行问答agent（nl2sql），需要能回答如下提问：
+ 提问1: 数据库中总共有多少张表；
+ 提问2: 员工表中有多少条记录
+ 提问3: 在数据库中所有客户个数和员工个数分别是多少

> nl2sql, 即natural language to sql, 核心作用是：让不会写 SQL 的人，也能直接用中文或英文去查询数据库。

此处设计为了真实的agent框架,可以根据用户的提问, 实时生成SQL, 执行SQL, 解析SQL结果, 其中生成SQL和解析SQL是两次LLM参与的过程, 是较为良好的工程实现, 具体实现情况见:
[nl2sql](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week12/nl2sql_agent.md)




2: 阅读 06-stock-bi-agent 代码，回答如下问题：
> 以下直接给出回答
+ 什么是前后端分离？
  - 前端只负责界面展示、用户交互、调用接口；
  - 后端只负责业务逻辑、数据存储、模型调用、对外提供 API；
  - 前端 Streamlit 负责交互，后端 FastAPI 负责模型与数据，二者通过 HTTP/SSE 通信。
+ 历史对话如何存储，以及如何将历史对话作为大模型的下一次输入；
  - 分为业务层历史和Agent运行态历史, 分为`chat_session`：会话维度信息（session_id、title、user_id）和 `chat_message`：每条消息（role、content、create_time、feedback）
  - 业务层写入逻辑在`services/chat.py`中, 用户提问通过`append_message2db(..., "user", content)`写入, 模型回复通过` append_message2db(..., "assistant", assistant_message)
`写入, 存在`server.db`中
  - Agent运行态历史通过使用 `AdvancedSQLiteSession` 存在 `assert/conversations.db`。
```python
session = AdvancedSQLiteSession(
    session_id=session_id,
    db_path="./assert/conversations.db",
    create_tables=True
)
# 将上下文上下文绑定到同一个 session_id，下一轮继续带着上下文推理。
result = Runner.run_streamed(agent, input=content, session=session)
```
  - (额外, 用于页面展示与切换)前端本地历史, 前端用 st.session_state.messages 维护当前页面消息列表；切换会话时会调用后端 /v1/chat/get 拉历史记录重新渲染。
