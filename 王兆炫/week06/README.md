本次hw完成以下任务 , 主要服务于RAG的工程化实现
+ 本地配置es环境,学习es的基础操作
+ 掌握示例RAG项目代码

---

### ES

这部分详细的过程见[ES基础练习](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week06/ES%E5%9F%BA%E7%A1%80%E7%BB%83%E4%B9%A0.md)

#### 使用ES的原因
实现RAG一个很重要的基础是对于DataBase的存取

我们知道DataBase分为关系型数据库和非关系型数据库(向量数据库),如果特别针对于处理向量数据库,那么Milvus是一个更适宜的选择 ; 我们在这里选择使用elasticsearch(ES)是因为:
+ ES同时支持两种数据库的存取
+ ES支持目前业内 BM25 首筛 + 向量重排 形成的混合检索,候选集更优质
> BM25 可以理解为改进的TF-IDF算法


更详细的原因 :

+ 成熟的全文检索能力（BM25 / 高级分析器）：对长文本、模糊匹配、多语言分词、短语/前缀搜索等场景表现稳定可靠。

+ 可扩展的分布式架构：水平扩展、分片/副本控制、快照恢复和集群监控，为生产级 RAG 提供可靠性与可用性。

+ 聚合/过滤与元数据检索：支持复杂过滤、聚合和布尔逻辑，便于在检索阶段加入业务约束（时间、来源、类型等）。

+ 混合检索（Hybrid: sparse + dense）可实现：可以把传统倒排索引检索（BM25）与向量相似度（dense vectors）结合，得到高质量候选集并再排序。

+ 丰富的生态/运维工具链：Kibana、Beats、Elastic APM、安全插件、审计等便于管控与观测。

+ 灵活的查询 DSL 与脚本评分：便于实现二次排序、打分融合与自定义召回策略。

#### 使用ES的细节

即使我们已经在对应环境中安装了elasticsearch库,但是我们仍然需要在本地安装的elasticsearch文件夹中,
找到`elasticsearch\bin\elasticsearch.bat`,每次都要先在后台运行这个程序,才可以进行其余操作


必须在后台运行 elasticsearch.bat 是因为 Elasticsearch 是一个独立的服务（JVM 进程 + HTTP API）, Python 的 elasticsearch 客户端只是一个 HTTP 客户端——它不会自己“启动”数据库进程。 只有服务在运行并监听（默认 http://localhost:9200），Python 客户端才能连接并执行查询/写入。

---

### RAG练习

这部分内容见[RAG练习](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week06/RAG%E7%BB%83%E4%B9%A0.md)

从R,A,G,即检索,增强,生成,逐步给出了对应的py文件以及作用,最终实现了完整的RAG问答流程

