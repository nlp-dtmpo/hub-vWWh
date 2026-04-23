1. 什么是前后端分离？                                                                                                                                                     
  前端只负责页面展示、用户交互、发请求，比如 Streamlit 页面。                                                                                                             
  后端只负责业务逻辑、数据库、调用大模型、调用工具，比如 FastAPI 服务。                                                                                                   
  前后端通过 HTTP API 或 SSE 流式接口通信，而不是后端直接生成完整页面。                                                                                                   
                                                                                                                                                                            
  在当前项目中：                                                                                                                                                            
                                                                                                                                                                            
  后端入口是 main_server.py:18，创建 FastAPI()。                                                                                                                          
  后端注册接口在 main_server.py:26 到 main_server.py:31，包括 /v1/chat、/stock 等。
  前端聊天页面在 demo/chat/chat.py:94 调用后端 http://127.0.0.1:8000/v1/chat/。                                                                                           
  前端用 requests.post(..., stream=True) 接收后端 SSE 流式输出，见 demo/chat/chat.py:112。                                                                                
  页面收到流式内容后用 Streamlit 渲染，见 demo/chat/chat.py:253 到 demo/chat/chat.py:263。                                                                                
                                                                                                                                                                            
  Streamlit 前端 → HTTP 请求 → FastAPI 后端 → 大模型 / MCP 工具 / 数据库                                                                                                
  2. 历史对话如何存储？
  项目里历史对话有两类存储。

  第一类：业务侧聊天记录，存到 SQLite 数据库 assert/sever.db。

  表结构在 models/orm.py：
  chat_session 表：存一次会话的元信息，比如用户、session_id、标题、开始时间，见 models/orm.py:35。
  chat_message 表：存每一条消息，比如 role、content、创建时间，见 models/orm.py:47。
  数据库地址是 sqlite:///./assert/sever.db，见 models/orm.py:61。

  写入消息的位置在 services/chat.py:292：

  append_message2db(session_id, role, content) 会把消息写入 chat_message。
  大模型回复完成后，会调用 append_message2db(session_id, "assistant", assistant_message)，见 services/chat.py:173。
  用户消息也会在服务逻辑里写入业务数据库，用于页面历史展示和会话列表。                                                                                                    
                                                                                                                                                                            
  第二类：大模型 Agent 的上下文状态，存到 SQLite 数据库 assert/conversations.db。                                                                                           
                                                                                                                                                                            
  关键代码在 services/chat.py:142：                                                                                                                                         
                  
  session = AdvancedSQLiteSession(                                                                                                                                          
      session_id=session_id,                                                                                                                                                
      db_path="./assert/conversations.db",                                                                                                                                  
      create_tables=True                                                                                                                                                    
  )                                                                                                                                                                         
                                                                                                                                                                            
  这里的作用是：把某个 session_id 对应的大模型对话上下文保存起来，供下一轮调用继续使用。                                                                                    
                  
  3. 历史对话如何作为大模型下一次输入？                                                                                                                                     
                  
  项目不是手动把所有历史消息拼成一个大列表传给模型，而是使用 AdvancedSQLiteSession 自动管理历史上下文。                                                                     
                  
  前端每次发消息时带上 session_id，见 demo/chat/chat.py:101 到 demo/chat/chat.py:106。                                                                                    
  后端收到请求后，用这个 session_id 创建同一个 AdvancedSQLiteSession，见 services/chat.py:143。
  调用大模型时传入 session=session，见 services/chat.py:163 和 services/chat.py:198。                                                                                     
                                                                                                                                                                          
  result = Runner.run_streamed(agent, input=content, session=session)                                                                                                     
  这里的 input=content 是当前用户的新问题，session=session 负责把之前同一 session_id 的历史对话带进去。这样模型下一次回答时就能“记得”前面的上下文。                                                                                                                   
