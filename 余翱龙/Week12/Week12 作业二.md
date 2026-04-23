## 什么是前后端分离？

**前端（页面 / 交互）和后端（数据 / 逻辑）分开独立开发、独立部署，通过接口（API）进行数据通信**，不再混在一起写。

## 历史对话处理与应用

基于代码分析，会话保存通过**三层架构**实现：

### 1. **数据库表设计（SQLite）**
在 `models/orm.py` 中定义两张表：
- **`chat_session`**：存储会话元数据（session_id、标题、创建时间、用户ID）
- **`chat_message`**：存储每条消息（角色、内容、关联的会话ID）

### 2. **后端保存流程**（`services/chat.py`）

**初始化会话**（第75-105行 `init_chat_session`）：
```python
# 创建会话记录
chat_session_record = ChatSessionTable(
    user_id=user_id[0],
    session_id=session_id,
    title=user_question,  # 用首条消息作为标题
)
session.add(chat_session_record)

# 添加system消息
message_recod = ChatMessageTable(
    chat_id=chat_session_record.id,
    role="system",
    content=get_init_message(task)
)
session.add(message_recod)
```


**保存每条消息**（第292-302行 `append_message2db`）：
```python
def append_message2db(session_id: str, role: str, content: str):
    # 查找会话ID
    message_recod = session.query(ChatSessionTable.id)\
        .filter(ChatSessionTable.session_id == session_id).first()
    # 插入消息记录
    message_recod = ChatMessageTable(
        chat_id=message_recod[0],
        role=role,
        content=content
    )
    session.add(message_recod)
    session.commit()
```


**流式对话时实时保存**（第118、174、222行）：
- 用户消息：调用 `append_message2db(session_id, "user", content)` 
- AI回复：流式输出完成后调用 `append_message2db(session_id, "assistant", assistant_message)`

### 3. **前端读取流程**（`chat_list.py` 第7行）
```python
# 获取用户所有会话列表
data = requests.post("http://127.0.0.1:8000/v1/chat/list?user_name=" + user_name)

# 进入具体会话时加载历史消息（chat.py 第56行）
data = requests.post("http://127.0.0.1:8000/v1/chat/get?session_id=" + session_id)
```


### 4. **删除功能**（第246-256行 `delete_chat_session`）
```python
# 先删除该会话的所有消息
session.query(ChatMessageTable).where(ChatMessageTable.chat_id == session_id[0]).delete()
# 再删除会话记录
session.query(ChatSessionTable).where(ChatSessionTable.id == session_id[0]).delete()
session.commit()
```

**总结**：采用关系型数据库持久化，后端统一管理数据一致性，前端通过REST API进行CRUD操作。