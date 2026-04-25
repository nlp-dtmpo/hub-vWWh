# 股票BI智能代理系统 - 技术问答

## 1. 什么是前后端分离？

### 1.1 概念定义

**前后端分离**是一种现代软件架构模式，将应用程序的**前端层**（用户界面）和**后端层**（业务逻辑、数据处理）解耦为独立的模块，各自独立开发、测试和部署。

### 1.2 本项目中的架构分层

| 层级 | 技术实现 | 主要职责 | 运行端口 |
| :--- | :--- | :--- | :--- |
| **前端展示层** | Streamlit | 用户界面渲染、交互逻辑处理、数据可视化 | Streamlit随机分配 |
| **后端业务层** | FastAPI | RESTful API提供、业务逻辑处理、用户认证 | 8000 |
| **工具服务层** | FastMCP | MCP协议工具调用、外部API集成 | 8900 |
| **数据持久层** | SQLite + SQLAlchemy | 数据存储、ORM映射、事务管理 | - |

### 1.3 前后端交互流程

```
用户操作 (Streamlit前端)
        ↓ HTTP请求
FastAPI后端 (业务逻辑处理)
        ↓ SQL查询
SQLite数据库 (数据读写)
        ↓ HTTP响应
Streamlit前端 (结果展示)
```

### 1.4 分离优势

| 优势 | 说明 |
| :--- | :--- |
| **解耦性** | 前端和后端可独立演进，互不影响 |
| **复用性** | 后端API可服务于多个前端（Web、APP、小程序） |
| **技术灵活** | 前端可选择React/Vue/Streamlit，后端可选择Python/Java/Go |
| **团队协作** | 前后端工程师可并行开发，提高效率 |
| **部署独立** | 可根据负载独立扩展前端或后端 |

---

## 2. 历史对话存储与大模型输入机制

### 2.1 双重存储策略

本项目采用**双重数据库存储**方案，分别服务于业务管理和大模型上下文管理。

#### ① 业务数据库存储

**存储位置**：`./assert/sever.db`

**数据表结构**：

**chat_session 表（会话元数据）**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| id | Integer | 主键 |
| user_id | Integer | 关联用户ID |
| session_id | String | 会话唯一标识 |
| title | String | 会话标题（首条消息） |
| start_time | DateTime | 会话开始时间 |
| feedback | Boolean | 用户反馈状态 |
| feedback_time | DateTime | 反馈时间 |

**chat_message 表（消息记录）**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| id | Integer | 主键 |
| chat_id | Integer | 关联会话ID |
| role | String | 角色（system/user/assistant） |
| content | Text | 消息内容 |
| generated_sql | Text | 生成的SQL（预留） |
| generated_code | Text | 生成的代码（预留） |
| create_time | DateTime | 创建时间 |

**核心代码**（`services/chat.py`）：

```python
def init_chat_session(user_name, user_question, session_id, task):
    """初始化会话，创建会话记录和系统消息"""
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(
            UserTable.user_name == user_name
        ).first()
        
        # 创建会话记录
        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question,
        )
        session.add(chat_session_record)
        session.commit()
        
        # 创建系统提示词消息
        message_record = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_init_message(task)
        )
        session.add(message_record)
        session.commit()

def append_message2db(session_id, role, content):
    """追加消息到数据库"""
    with SessionLocal() as session:
        chat_id = session.query(ChatSessionTable.id).filter(
            ChatSessionTable.session_id == session_id
        ).first()
        
        if chat_id:
            message_record = ChatMessageTable(
                chat_id=chat_id[0],
                role=role,
                content=content
            )
            session.add(message_record)
            session.commit()
```

#### ② 大模型会话存储

**存储位置**：`./assert/conversations.db`

**技术方案**：使用 OpenAI Agent SDK 提供的 `AdvancedSQLiteSession`

```python
session = AdvancedSQLiteSession(
    session_id=session_id,           # 与业务会话关联
    db_path="./assert/conversations.db",
    create_tables=True
)
```

### 2.2 历史对话作为大模型输入的流程

**核心流程图**：

```
用户提问 → 检查会话状态 → 存储用户消息 → 加载历史对话 → 调用大模型 → 存储响应 → 返回结果
```

**详细步骤**（`services/chat.py:chat()` 函数）：

```python
async def chat(user_name, session_id, task, content, tools=[]):
    # 步骤1: 检查并初始化会话
    if session_id:
        record = session.query(ChatSessionTable).filter(
            ChatSessionTable.session_id == session_id
        ).first()
        if not record:
            init_chat_session(user_name, content, session_id, task)
    
    # 步骤2: 存储用户消息到业务数据库
    append_message2db(session_id, "user", content)
    
    # 步骤3: 获取系统提示词
    instructions = get_init_message(task)
    
    # 步骤4: 初始化大模型会话（自动加载历史对话）
    llm_session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )
    
    # 步骤5: 创建Agent并执行
    agent = Agent(
        name="Assistant",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
        mcp_servers=[mcp_server] if tools else None
    )
    
    # 步骤6: 流式调用（历史对话自动包含在session中）
    result = Runner.run_streamed(agent, input=content, session=llm_session)
    
    # 步骤7: 存储大模型响应
    append_message2db(session_id, "assistant", assistant_message)
```

### 2.3 大模型输入消息结构

SDK 自动组装的消息格式：

```python
messages = [
    {
        "role": "system", 
        "content": "你是一位名叫小呆助手的专业人工智能...（根据任务类型动态生成）"
    },
    {"role": "user", "content": "什么是市盈率？"},
    {"role": "assistant", "content": "市盈率（PE）是股价与每股收益的比率..."},
    {"role": "user", "content": "帮我分析贵州茅台的PE"},  # 当前问题
]
```

### 2.4 任务类型与系统提示词

根据不同任务类型，系统提示词会动态调整：

| 任务类型 | 特点 | 系统提示词重点 |
| :--- | :--- | :--- |
| **股票分析** | 专业金融分析 | 使用P/E、EPS、Beta等术语，提供结构化分析 |
| **数据BI** | 数据分析处理 | 输出可执行代码（SQL/Python），逻辑严谨 |
| **通用对话** | 日常聊天 | 友好、轻松、富有同理心 |

### 2.5 技术亮点

| 技术点 | 说明 |
| :--- | :--- |
| **Session关联** | 通过 `session_id` 实现业务数据库与大模型会话的关联 |
| **自动上下文管理** | `AdvancedSQLiteSession` 自动加载历史对话 |
| **流式响应** | 使用 SSE（Server-Sent Events）实现实时对话体验 |
| **工具调用集成** | 支持通过 MCP 协议调用外部工具（股票数据、新闻等） |

---

## 总结

1. **前后端分离**：通过 Streamlit（前端）、FastAPI（后端）、FastMCP（工具服务）的分层架构，实现了解耦和独立演进能力。

2. **对话存储与输入**：采用双重数据库策略，业务数据库存储完整对话记录供管理查询，大模型会话数据库专门服务于上下文管理；通过 `session_id` 关联，`AdvancedSQLiteSession` 自动将历史对话注入大模型输入，实现多轮对话记忆功能。