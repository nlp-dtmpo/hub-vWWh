本次作业完成下列任务:

Task1: 安装openai-agents框架，实现如下的一个程序：
+ 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
+ 子agent 1: 对文本进行情感分类
+ 子agent 2: 对文本进行实体识别

此处实现见 [作业一](https://github.com/Birchove/ai_learning/edit/main/%E7%8E%8B%E5%85%86%E7%82%AB/week11/homework1_router.py)

Task2: 为4-项目案例-企业职能助手，增加3个自定义的tool 工具，实现自定义的功能，并在对话框完成调用

(自然语言 -> 工具选择 ->  工具执行结果)

此处选择的三个工具是-> 
+ query_leave_balance(user_name)：查询年假余额
+ query_payday()：查询发薪日规则
+ create_meeting_summary(topic, notes)：生成会议纪要摘要

选择原则是因为, 这些tools容易通过自然语言触发
