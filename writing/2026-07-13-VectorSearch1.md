---
layout: mypost
title: 向量检索工程化实践————极端准确率
categories: 向量检索
address: 北京🎑
extMath: true
mermaid: true
special_tag: 更新中
show_footer_image: true
tags:
- 向量检索
description: 
---
在RAG或者在做向量检索中很多情况需要检索的指标高，比如在某些极端情况中可能需要ACC≥0.99（比如说语音识别邻域可能就需要对于说话人其准确率达到如此高的指标），下面简单总结具体实践过程中用到的优化策略去逼近ACC上限。
> **必须提一嘴**：去提升ACC上限同时极大概率会去损失部分值（理论上很难做到既要又要情况发送）

对于下面内容主要从如下几个方面出发进行介绍：1、模型层出发提升ACC；2、优化策略提升ACC。
## 向量检索前置知识
**何为向量检索**，简单来说向量检索就是直接将我的输入通过embedding模型进行编码而后存到向量数据库中，在输入新的内容是在进行embedding编码而后去数据库中检索即可。**如何存储向量数据**，一般而言存储向量数据工具很多如Milvus、Chroma、Qdrantd等这里仅介绍[Milvus数据库](https://milvus.io/zh)（主要简单介绍基本使用），开始之前在Milvus中两种概念简单了解：**距离度量以及索引策略**，对于这两种前者负责判断两个向量之间是不是相似的，后者判断如何快速的从数据库中进行检索。对于[向量编码模型排行榜](https://huggingface.co/spaces/mteb/leaderboard)
### 距离度量以及索引策略
对于距离度量以及索引策略在创建表之前就已经规定好了不可改变，比如说创建过程：
```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 创建列名，规定其名称、数据类型等
schema = client.create_schema()
schema.add_field(
    field_name="id",
    datatype="INT64",
    is_primary=True
)
schema.add_field(
    field_name="embedding",
    datatype="FLOAT_VECTOR",
    dim=768
)
# 对于我的向量内容去规定索引方式以及距离度量方式
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    index_name="vector_index"
)
client.create_collection(
    collection_name="docs",
    schema=schema,
    index_params=index_params
)
```
直接搬运[^1]中对于距离度量策略如下;

|             距离度量策略              |                                                                       计算方式                                                                       |                                          优缺点                                           |
|:-------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|
|        **COSINE（余弦相似度）**        | $\displaystyle \text{Cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}=\frac{\sum_{i=1}^{n}A_iB_i}{\sqrt{\sum_{i=1}^{n}A_i^2}\sqrt{\sum_{i=1}^{n}B_i^2}}$ |    **优点**不受向量长度影响，更关注语义方向，是文本 Embedding、RAG 中最常用的距离度量。**缺点**无法体现向量长度差异，通常需要进行归一化处理。    |
| **L2（Euclidean Distance，欧氏距离）** |                                             $\displaystyle L2(A,B)=\sqrt{\sum_{i=1}^{n}(A_i-B_i)^2}$                                             |           **优点**计算简单、直观，适用于图像特征、科学计算等场景。**缺点**受向量长度影响较大，文本语义检索效果通常不如 Cosine。           |
|    **IP（Inner Product，内积）**     |                                              $\displaystyle IP(A,B)=A\cdot B=\sum_{i=1}^{n}A_iB_i$                                               | **优点**计算效率高，若向量已归一化，则效果与 Cosine 基本一致，因此推荐系统和部分 Embedding 模型广泛采用。**缺点**未归一化时容易受到向量长度影响。 |
|        **HAMMING（汉明距离）**        |                                 $\displaystyle H(A,B)=\sum_{i=1}^{n}[A_i\neq B_i]$，其中 $[A_i\neq B_i]$ 表示对应位是否不同。                                 |        **优点**计算速度快，适用于二值向量、哈希、指纹等数据。**缺点**仅支持 Binary Vector，不适用于浮点型 Embedding。         |
|       **JACCARD（杰卡德距离）**        |                                     $\displaystyle J(A,B)=1-\frac{\vert A\cap B \vert}{\vert A\cup B \vert}$                                     |          **优点**适用于集合、标签和二值特征的相似性计算。**缺点**仅支持 Binary Vector，不适用于文本 Embedding。           |

其中索引策略用于**加速向量检索**。如果没有索引，每次查询都需要遍历所有向量进行计算（全量搜索），当数据规模达到百万甚至千万级时，检索效率会大幅下降。因此，Milvus 通过不同的索引策略，在**查询速度、召回率（Recall）和存储开销**之间进行平衡。

| 索引策略 | 原理 | 优点 | 缺点 | 适用场景 |
|:---------:|:-----|:-----|:-----|:---------|
| **FLAT** | 暴力遍历所有向量，计算与查询向量的距离。 | 检索结果 100% 精确，实现简单。 | 查询速度最慢，不适合大规模数据。 | 数据量较小（<10 万）、算法测试、效果验证。 |
| **IVF_FLAT** | 先将向量聚类，查询时仅搜索最相近的几个聚类。 | 查询速度快，内存占用适中。 | 召回率略低于 FLAT，需要调节 `nlist`、`nprobe` 参数。 | 百万级数据、通用向量检索。 |
| **HNSW** | 构建多层小世界图（Hierarchical Navigable Small World），通过图搜索快速找到近邻。 | 查询速度快、召回率高，是目前最常用的索引。 | 建索引时间较长，占用内存较大。 | RAG、知识库、AI 搜索、推荐系统。 |
| **IVF_PQ** | 在 IVF 基础上，对向量进行乘积量化（Product Quantization）压缩。 | 大幅降低内存占用，适合海量数据。 | 精度略有损失，建索引复杂。 | 亿级以上向量、内存受限场景。 |
| **DISKANN** | 将部分索引存储在磁盘，通过磁盘和内存协同完成检索。 | 支持超大规模数据，降低内存需求。 | 建索引耗时较长，对 SSD 性能要求较高。 | 十亿级向量检索、大规模生产环境。 |

### Milvus中增/删/查/改
对于数据库就离不开：增加/删除/查找/改动这几种方式，简单描述如下：

| 操作 | API | 说明 | 特点 |
|:----:|:----|:-----|:-----|
| **增加（Insert）** | `insert()` | 向 Collection 中插入新的向量及其元数据。 | 最常用操作，支持批量插入。 |
| **查询（Query / Search）** | `query()`、`search()` | `query()` 根据主键或过滤条件查询；`search()` 根据向量相似度检索。 | Milvus 最核心的功能是 `search()`。 |
| **修改（Upsert）** | `upsert()` | 根据主键更新数据；若主键不存在则插入新数据。 | Milvus 不支持传统 SQL 的 `UPDATE`，通常使用 Upsert 实现更新。 |
| **删除（Delete）** | `delete()` | 根据主键或过滤条件删除数据。 | 删除后数据不会立即从磁盘移除，而是标记删除，后续由后台进行垃圾回收（Compaction）。 |

具体代码如下：
```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# =========================
# 1. 新增（Insert）
# =========================
client.insert(
    collection_name="docs",
    data=[
        {
            "id": 1,
            "title": "FastAPI 教程",
            "embedding": vector
        }
    ]
)

# =========================
# 2. 查询（Query：根据条件）
# =========================
result = client.query(
    collection_name="docs",
    filter="id == 1",
    output_fields=["title"]
)

# =========================
# 3. 向量检索（Search）
# =========================
results = client.search(
    collection_name="docs",
    data=[query_vector],
    anns_field="embedding",
    limit=5
)

# =========================
# 4. 更新（Upsert）
# 主键存在则更新，不存在则插入
# =========================
client.upsert(
    collection_name="docs",
    data=[
        {
            "id": 1,
            "title": "新版 FastAPI 教程",
            "embedding": new_vector
        }
    ]
)

# =========================
# 5. 删除（Delete）
# =========================
client.delete(
    collection_name="docs",
    filter="id == 1"
)
```

## 向量检索工程化实践
姑且将数据库检索优化分为两类：1、单条数据检索，简单理解为输入单条数据A就必须立马从数据库中得到检索结果；2、批次数据检索，简单理解为输入一批数据需要得到这批数据中的检索结果
### 批次数据检索优化

## 参考
[^1]: [https://milvus.io/docs/zh/metric.md?tab=floating](https://milvus.io/docs/zh/metric.md?tab=floating)
