本次作业完成以下的内容：


1: 参考sql agent，实现一下基于 chinook.db 数据集进行问答agent（nl2sql），需要能回答如下提问：
+ 提问1: 数据库中总共有多少张表；
+ 提问2: 员工表中有多少条记录
+ 提问3: 在数据库中所有客户个数和员工个数分别是多少

> nl2sql, 即natural language to sql, 核心作用是：让不会写 SQL 的人，也能直接用中文或英文去查询数据库。

此处设计为了真实的agent框架,可以根据用户的提问, 实时生成SQL, 执行SQL, 解析SQL结果, 其中生成SQL和解析SQL是两次LLM参与的过程, 是较为良好的工程实现, 具体实现情况见:
[nl2sql](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week12/nl2sql_agent.md)




2: 阅读 06-stock-bi-agent 代码，回答如下问题：
+ 什么是前后端分离？
+ 历史对话如何存储，以及如何将历史对话作为大模型的下一次输入；
