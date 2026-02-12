## Q1 : 数据库的设计
参照阿里云的实现 , 我们知道要有两种类型的数据库 : MySQL存储结构化数据 , 向量数据库存储非结构化数据

### MySQL数据库

1. 类目表`faq_catgories`
   + 字段 id , parent_id(类目表可以无限极分类 , 所以此处实现使用树形结构) , name , level , sort_order
   + 对应功能：文档中的 [FAQ类目管理]，用于前端展示目录树，以及算法端按类目过滤检索范围。
2. 主知识表`faq_knowledge`
   + 字段 id , category_id(关联类目) , standard_questions(标准问法) , ans_txt(文本答案或HTML) , start_time , end_time , status(启用/停用)
   + 最终实现查询和返回给用户的内容
3. `faq_similar_questions` (相似问法表)
   + id , knowledge_id(关联主知识表) , question_txt(提问内容)
   + 算法命中率关键 , 标准问和相似问需要一起被量化

---
### 向量数据库
存储经过 BERT 编码后的数学向量，用于计算相似度。实现非结构化匹配：解决“字面不同但语义相同”的问题。

1. Vector_Field : embedding
2. Scalar Field :
   + relation_id：指向 MySQL 中 faq_knowledge 的主键 ID。
   + category_id：用于支持“只在某个类目下搜索”。
---
### 双环境下适配
阿里云文档中提到的“正式环境/测试环境（或草稿态/发布态）”的双环境管理机制，是企业级知识库系统的标配。双环境的核心是隔离性 , 为了实现这个诉求 , 我们需要对上述两个数据库的设计进行调整:

#### MySQL数据库改动
1. 类目表`faq_catgories` 无需改动 , 类目应是全局的 , 否则维护关系混乱
2. 主知识表`faq_knowledge`
  + 新增字段 :
    - `group_id`：逻辑 ID。代表“同一个问题”。Draft 记录和 Online 记录的 group_id 必须相同，这样程序才知道它们是同一个问题的不同状态。
    - `environment`：枚举值 ('DRAFT', 'ONLINE')。用于区分环境。
  + 当用户问问题时，系统查询 WHERE environment = 'ONLINE'；而运营人员编辑时，操作的是 WHERE environment = 'DRAFT' 的数据。
3. `faq_similar_questions` (相似问法表)
  + 无需改动
  + 因为knowledge_id 关联主知识表 , 而主知识表中 Draft 和 Online 是两条不同的记录，所以相似问法也分别对应这两条记录。

#### 向量数据库改动
1. 新增字段 :
   environment ：必须增加此标量字段（Scalar Field），值设为 'DRAFT' 或 'ONLINE'。
2. 实现细节 :
   + 在进行 在线检索（用户提问） 时，查询语句必须带上过滤条件 `filter: "environment == 'ONLINE'"`。
   + 在进行 运营测试（预览效果） 时，查询语句可以设为 `filter: "environment == 'DRAFT'"`。
   + 发布(Publish)时 , 执行以下事务:
     - 删除 `environment='ONLINE'` 中对应的 group_id 的旧记录（以及对应的相似问）。
     - 将 `environment='DRAFT'` 的记录复制一份，environment 修改为 'ONLINE'。
     - 对应修改向量数据库内容
---
## model的选择 , 以及BERT和LLM如何协作

技术路线--基于向量数据库,以BERT语义检索为主,用LLM辅助(润色/兜底)

对于用户提问使用BERT算法获取最相似的top1并返回对应score: { "relation_id": 55, "score": 0.92 }
   + 若score>0.85 , 算法认为命中了预设knowledge , 直接返回匹配结果
   + 若score $\in$ (0.6,0.85) , 算法认为结果是不确定的 , 返回澄清式话术 : “您是想问【退款时效】相关的问题吗？”（利用 MySQL 中的标准问标题进行反问）, 让用户做选项式判断
     - 若用户选择"是" , 则转到上一条情况
     - 若用户选择"否" , 则转到下一条情况
   + 若score<0.65 , 算法认为结果不匹配 , 此时调用LLM , 传入用户原话术 , Prompt：“你是一个客服助手。用户问：{User_Query}。知识库中未找到答案，请礼貌地安抚用户，并建议其联系人工客服。” 或者尝试用通用知识回答（取决于业务风险偏好）。













