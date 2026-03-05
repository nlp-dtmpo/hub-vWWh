配置好es,在后台打开bat文件后,就可以只在vs code中修改调试代码了

---
### ES测试.py
以下是运行es_测试.py得到的结果:
```bash
==================================================

--- 正在测试常见的 Elasticsearch 内置分词器 ---

使用分词器：standard
原始文本: 'Hello, world! This is a test.'
分词结果: ['hello', 'world', 'this', 'is', 'a', 'test']

使用分词器：simple
原始文本: 'Hello, world! This is a test.'
分词结果: ['hello', 'world', 'this', 'is', 'a', 'test']

使用分词器：whitespace
原始文本: 'Hello, world! This is a test.'
分词结果: ['Hello,', 'world!', 'This', 'is', 'a', 'test.']

使用分词器：english
原始文本: 'Hello, world! This is a test.'
分词结果: ['hello', 'world', 'test']

==================================================

--- 正在测试 IK 分词器 ---

使用 IK 分词器：ik_smart
原始文本: '我在使用Elasticsearch，这是我的测试。'
分词结果: ['我', '在', '使用', 'elasticsearch', '这是', '我', '的', '测试']

使用 IK 分词器：ik_max_word
原始文本: '我在使用Elasticsearch，这是我的测试。'
分词结果: ['我', '在', '使用', 'elasticsearch', '这是', '我', '的', '测试']
```
---
### ES_基础.py

这份代码分为四个步骤: 连接 -> 建立schema(Mapping) -> 数据入库/刷新 -> 查询

#### 测试连接
`es_client.ping()` 可以返回客户端是否正常连接

#### 定义与映射
下面给出mapping的json代码以及对应line的解释
```python
mapping = {
  # 1. Settings (索引设置): 定义索引的物理存储特性
  "settings": {
    "number_of_shards": 1,   # 分片数：数据被分成多少块。单机环境下设为 1 即可。
    "number_of_replicas": 0  # 副本数：数据的备份数量。单机开发模式建议设为 0 以节省资源。
  },
  
  # 2. Mappings (映射定义): 定义字段的数据类型和分词规则（类似于 SQL 的表结构）
  "mappings": {
    "properties": {
      
      # 标题字段
      "title": {
        "type": "text",             # 数据类型为 text，支持全文检索（会被分词）
        "analyzer": "ik_max_word",  # 索引时使用：最细粒度分词（尽可能拆出更多词，增加被搜到的概率）
        "search_analyzer": "ik_smart" # 搜索时使用：智能分词（更符合人类语言习惯，避免搜出太多无关项）
      },
      
      # 内容字段
      "content": {
        "type": "text",             # 数据类型为 text
        "analyzer": "ik_max_word",  # 存储时拆得最细
        "search_analyzer": "ik_smart" # 搜索时按逻辑拆分
      },
      
      # 标签字段
      "tags": { 
        "type": "keyword"           # 数据类型为 keyword，不分词。必须完全匹配才能搜到，常用于分类或标签筛选。
      },
      
      # 作者字段
      "author": { 
        "type": "keyword"           # 数据类型为 keyword，通常用于精确过滤（如：查找“张三”的所有文章）
      },
      
      # 创建时间字段
      "created_at": { 
        "type": "date"              # 数据类型为 date，支持时间范围查询（如：查询最近一周的博文）
      }
    }
  }
}
```
而后就是类似于request的操作方式 : response , client , query 等操作,较固定

#### 数据入库

数据实时传入对应网页,可以通过 http://localhost:9200/blog_posts_py/_search?pretty 查询

也可以通过此网页显示的结构,得到对应的元素/值的索引方式

<img width="1109" height="1182" alt="image" src="https://github.com/user-attachments/assets/6bb1adc9-0b76-469d-a637-53310f612f87" />


如此处response['hits']['hits']可以获取hits列表,从而得到hit['_score']等次级信息

---
### ES进阶

实现框架与ES基础基本相同,但在一些细节上进行了优化

#### 区别1 - 增加精准id控制

`es.index(index=index_name, id="A001", document=doc_1)`

在ES基础中,index自动生成,这意味着每次运行脚本,都会append两条数据 ; 而此处指定了id,每次运行只会覆写,更加精确

#### 区别2 - 多字段检索 (multi_match)

```python
"multi_match": {
    "query": "智能",
    "fields": ["name", "description"]
}
```

增加了更多的match搜索 , 这在后台会为两个字段分别计算得分，然后取最大值（默认行为）作为文档的最终 _score。

#### 区别3 - 复合filter

```python
      "filter": [
                    {"term": {"category": "电子产品"}},
                    {"term": {"on_sale": True}},
                    {"range": {"price": {"lt": 1000}}}
                ]
```

增加了多term字段匹配,增加了range范围搜索, lt 代表 Less Than（小于），此外还有 gt (大于)、lte (小于等于) 等。

#### 区别4 - 聚合查询

```python
res_3 = es.search(
    index=index_name,
    body={
        "aggs": {
            "products_by_category": {   # 1. 这是你自己起的名字（相当于变量名）
                "terms": {              # 2. 聚合类型：terms 代表“按词条分类”
                    "field": "category",    # 3. 按哪个字段分类？这里按“类别”
                    "size": 10          # 4. 最多返回前 10 个分类
                }
            }
        },
        "size": 0  # 不返回文档结果，只返回聚合结果
    }
)
```

ES中的aggs实际上相当于 SQL 里的 `GROUP BY` , 聚合的对象就是filed , 注意filed的对象必须是`keyword`

---
### ES向量检索

| 维度   | ES 基础/进阶 (传统)                                 | ES 向量检索 (现代)                                   |
|--------|-----------------------------------------------------|------------------------------------------------------|
| 匹配原理 | 文本匹配：找相同的字符/单词（依赖分词器）。          | 语义匹配：找数学空间中距离最近的点。                  |
| 理解能力 | 搜“电脑”，可能搜不到“笔记本”。                      | 搜“电脑”，能自动通过向量关联到“笔记本”。              |
| 数据形态 | 存储的是字符串。                                     | 存储的是高维稠密向量 (dense_vector)。                |
| 计算方式 | 倒排索引过滤（词频/逆文档频率）。                    | k-最近邻算法 (kNN)：计算余弦相似度。                  |


#### 进阶点

1. 引入TransFormer

此处增加了SentenceTransformer的判断,不再是单纯的文本匹配

2. 映射类型变为DenseVector

```python
"text_vector": {
    "type": "dense_vector",
    "dims": 512,
    "index": True,
    "similarity": "cosine"
}
```

使用余弦相似度计算相似性

3. 使用knn查询

```python
"knn": {
    "field": "text_vector",
    "query_vector": query_vector,
    "k": 3
}
```
> KNN（K-Nearest Neighbors，K-近邻算法）是一种基于实例的监督学习方法，用于分类和回归。即选取向量空间内的临近点

4. 更进一步的混合搜索

```python
    body={
        "knn": {
            "field": "text_vector",
            "query_vector": query_vector,
            "k": 3,
            "num_candidates": 10
        },
        "query": {
            "match": {
                "text": "技术"
            }
        },
        "fields": ["text"],
        "_source": False
    }
```

既要满足关键词匹配(query),又要语义匹配(knn),同时,依旧使用filed聚类


