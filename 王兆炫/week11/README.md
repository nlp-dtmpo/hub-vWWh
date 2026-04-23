本次作业完成下列任务:

Task1: 安装openai-agents框架，实现如下的一个程序：
+ 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
+ 子agent 1: 对文本进行情感分类
+ 子agent 2: 对文本进行实体识别

此处实现见 [作业一](https://github.com/Birchove/ai_learning/edit/main/%E7%8E%8B%E5%85%86%E7%82%AB/week11/homework1_router.py)

---

Task2: 为4-项目案例-企业职能助手，增加3个自定义的tool 工具，实现自定义的功能，并在对话框完成调用

(自然语言 -> 工具选择 ->  工具执行结果)

此处实现见 [作业二](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week11/homework2_tools_addition.py)

此处选择的三个工具是-> 
+ query_leave_balance(user_name)：查询年假余额
+ query_payday()：查询发薪日规则
+ create_meeting_summary(topic, notes)：生成会议纪要摘要

选择原则是因为, 这些tools容易通过自然语言触发

启动企业工具问答助手时, 需要注意以下两个细节:
1. 启动 MCP Server

在mcp_server目录运行: `python mcp_server_main.py` , 并保持该端口常驻

2. 启动对话前端（Streamlit）

在项目目录运行: `streamlit run steamlit_demo.py` 即可在网页端(本地ip)查看项目demo, 勾选tools后可以使用对应tools

可以看到, 通过streamlit, mcp服务已上线

<img width="2559" height="1340" alt="image" src="https://github.com/user-attachments/assets/b85d7ff0-d318-4280-bb71-d3944065de59" />
