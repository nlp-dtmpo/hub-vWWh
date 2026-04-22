什么是前后端分离？
答：前端后端分别独立开发，通过api进行前后端交互


06-stock-bi-agent代码中历史对话如何存储，以及如何将历史对话作为大模型的下一次输入
答：06-stock-bi-agent代码中，是将用户历史对话记录存储到关系型数据库中，    
session = AdvancedSQLiteSession(
        session_id=session_id, # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True)

通过session_id将历史对话取出来通过content字段传入模型中
 result = Runner.run_streamed(agent, input=content, session=session)
