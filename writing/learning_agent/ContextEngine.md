---
layout: mypost
title: A Survey of Context Engineering for Large Language Models — 论文精读
categories: [Agent, LLM]
tags: [Context Engineering, Survey, LLM]
extMath: true
show: false
---

# A Survey of Context Engineering for Large Language Models — 论文精读

> **论文**: [A Survey of Context Engineering for Large Language Models](https://arxiv.org/abs/2507.13334)
> **作者**: Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, Shenghua Liu (15人，主要来自中科院计算技术研究所)
> **时间**: 2025年7月 (v2)
> **规模**: 166页, 综述1400+篇论文
> **资源**: [GitHub](https://github.com/Meirtz/Awesome-Context-Engineering) | [HuggingFace](https://huggingface.co/papers/2507.13334)

---

## 1. 核心定义

论文将 **Context Engineering（上下文工程）** 正式确立为一门学科，超越传统的 Prompt Engineering。

### 形式化定义

LLM的自回归概率模型为:

$$P_{\theta}(Y|C) = \prod_{t=1}^{T} P_{\theta}(y_t | y_{<t}, C)$$

**传统Prompt Engineering**: `C = prompt` (单一静态字符串)

**Context Engineering** 将上下文重新定义为多个信息组件的动态结构化组装:

$$C = \mathcal{A}(c_{instr}, c_{know}, c_{tools}, c_{mem}, c_{state}, c_{query})$$

其中 $\mathcal{A}$ 是高层编排/组装函数:

| 组件 | 含义 | 对应章节 |
|------|------|----------|
| $c_{instr}$ | 系统指令与行为规则 | 基础组件 |
| $c_{know}$ | 外部知识 (RAG检索、知识图谱) | RAG系统 |
| $c_{tools}$ | 工具定义与函数签名 (Function Calling) | 工具集成 |
| $c_{mem}$ | 历史交互的持久化信息 | 记忆系统 |
| $c_{state}$ | 用户/世界/多智能体的动态状态 | 多智能体 |
| $c_{query}$ | 用户的即时查询请求 | — |

### 优化目标

上下文工程被形式化为寻找最优上下文生成函数集 $\mathcal{F} = \{\mathcal{A}, \text{Retrieve}, \text{Select}, ...\}$ 的优化问题:

$$\mathcal{F}^* = \arg\max_{\mathcal{F}} \mathbb{E}_{\tau \sim \mathcal{T}} \left[ \text{Reward}(P_\theta(Y | C_{\mathcal{F}}(\tau)), Y^*_\tau) \right]$$

**约束条件**: 上下文长度限制 $|C| \leq L_{max}$、计算资源限制

### 信息论优化视角

知识检索可被框架化为信息论最优问题，目标是选择与目标答案 $Y^*$ 互信息最大的知识:

$$\text{Retrieve}^* = \arg\max_{\text{Retrieve}} I(Y^*; c_{know} | c_{query})$$

这确保了检索的上下文不仅是语义相似的，更是对解决任务**信息量最大化**的。

### 贝叶斯上下文推理

整个上下文工程过程也可通过贝叶斯推理的视角看待——不是确定性地构建上下文，而是推断最优上下文后验概率:

$$P(C | c_{query}, \text{History}, \text{World}) \propto P(c_{query} | C) \cdot P(C | \text{History}, \text{World})$$

决策论目标: 找到最大化期望奖励的上下文 $C^*$:

$$C^* = \arg\max_{C} \int P(Y | C, c_{query}) \cdot \text{Reward}(Y, Y^*) \, dY \cdot P(C | c_{query}, ...)$$

贝叶斯形式化提供了处理不确定性、自适应检索更新先验、以及在多步推理中维护信念状态的原则性方法。

### 上下文标度 (Context Scaling)

上下文标度包含两个基本维度:
- **长度标度**: 从千级到百万级token的扩展，涉及注意力机制、内存管理的架构创新
- **多模态与结构标度**: 扩展上下文到时间上下文、空间上下文、参与者状态、意图上下文、文化上下文等多维信息结构

---

## 2. Prompt Engineering vs Context Engineering

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|---------------------|
| **模型** | $C = prompt$ (静态字符串) | $C = \mathcal{A}(c_1, ..., c_n)$ (动态结构化组装) |
| **目标** | $\arg\max_{prompt} P_\theta(Y|prompt)$ | 对 $\mathcal{F}$ 进行系统级优化 |
| **复杂度** | 字符串空间搜索 | 函数集合优化 |
| **状态** | 无状态 | 有状态 ($c_{mem}$, $c_{state}$) |
| **扩展性** | 随长度增长脆弱 | 模块化组合管理复杂性 |
| **信息** | 固定内容 | 约束下最大化任务相关信息 |

---

## 3. 分类框架总览

论文提出**两层、七个子领域**的统一分类体系:

```
Context Engineering
├── 基础组件层 (Foundational Components)
│   ├── 3.1 上下文检索与生成 (Context Retrieval & Generation)
│   ├── 3.2 上下文处理 (Context Processing)
│   └── 3.3 上下文管理 (Context Management)
│
└── 系统实现层 (System Implementations)
    ├── 3.4 检索增强生成 (RAG)
    ├── 3.5 记忆系统 (Memory Systems)
    ├── 3.6 工具集成推理 (Tool-Integrated Reasoning)
    └── 3.7 多智能体系统 (Multi-Agent Systems)
```

---

## 4. 基础组件层详解

### 4.1 上下文检索与生成

**提示驱动生成:**
- **Zero-shot / Few-shot**: 基础提示方法
- **Chain-of-Thought (CoT)**: 模拟人类分步推理, MultiArith准确率 17.7% → 78.7%
- **Tree-of-Thoughts (ToT)**: 层次化推理结构, 支持探索/前瞻/回溯, Game of 24 成功率 4% → 74%
- **Graph-of-Thoughts (GoT)**: 图结构建模推理节点依赖, 质量提升62%, 计算成本降低31%
- **Cognitive Prompting**: 基于Guilford智力结构模型, GPT-4在AIME2024上 26.7% → 43.3%
- **Automatic Prompt Engineer (APE)**: 搜索算法自动发现最优提示结构

**外部知识检索:**
- 密集检索 (Dense Retrieval): 向量数据库 + 语义相似度
- 稀疏检索 (Sparse Retrieval): BM25等关键词匹配
- 混合检索 (Hybrid): 密集+稀疏融合
- 知识图谱检索: 结构化知识集成

**动态上下文组装:**
- 多组件编排与优先级排序
- 任务优化组装策略
- 跨模态信息融合

### 4.2 上下文处理

**长序列处理 (O(n²) → O(n)):**
- **Mamba / Mamba-2**: 状态空间模型 (SSM), 固定大小隐状态实现线性复杂度
- **FlashAttention 系列** (1/2/3/4): IO感知精确注意力, FlashAttention-2 速度约2倍提升
- **Ring Attention**: 分布式长序列处理，跨设备块计算+通信重叠
- **LongNet**: 膨胀注意力，指数增长注意力场，处理超十亿token
- **LongRoPE**: 两阶段 (微调256K → 位置插值) 实现2048K token上下文窗口
- **YaRN**: NTK插值 + 线性插值 + 注意力分布校正
- **Infini-attention**: 压缩记忆融入单Transformer块，有限内存处理无限输入
- **StreamingLLM**: 保留Attention Sink + 最近KV Cache, 400万token序列达22.2倍加速
- **Self-Extend**: 双层注意力 (分组注意力 + 邻居注意力)，无需微调处理长上下文
- **Mistral-7B**: 输入从4K→128K tokens 需122倍计算量增加
- **Llama 3.1 8B**: 每128K-token请求需16GB显存

**自我优化与适应:**
- **Self-Refine**: 同一模型充当生成器+反馈提供者+精炼器，GPT-4通过此方法获得约20%的绝对性能提升
- **Reflexion**: 情景记忆 + 语言化反馈，维护反思文本用于未来决策
- **Multi-Aspect Feedback**: 冻结LM与外部工具集成，多维度错误检测
- **N-CRITICS**: 集成多评价者 (生成LM + 其他模型)，编译反馈指导迭代精炼
- **SELF**: 教授LLM元技能 (自我反馈、自我精炼)，持续自我进化
- **Agent-R**: Monte Carlo Tree Search (MCTS) 构建训练数据纠正错误路径
- **Self-Developing**: LLM自主发现、实现、精炼自身的改进算法 (作为可执行代码)
- **A2R**: 多维度显式评估 (正确性、引用质量等)，迭代精炼输出
- **I-SHEEP**: 从零开始持续自我对齐，生成/评估/过滤/训练高质量合成数据
- **Creator**: LLM创建并使用自己的工具，四模块流程 (创建→决策→执行→识别)

**Long Chain-of-Thought (长链思维):**
- OpenAI-o1, DeepSeek-R1, QwQ, Gemini 2.0 Flash Thinking 等模型实现
- 增加推理步长 (即使不添加新信息) 通过测试时扩展 (test-time scaling) 显著增强推理能力
- 优化策略: best-of-N采样、Zero-Thinking/Less-Thinking模式、显式紧凑CoT

**多模态与结构化集成:**
- 文本 + 图像 + 音频 + 视频融合
- 知识图谱嵌入 (GraphToken, StructGPT)
- 图神经网络-LLM混合架构

### 4.3 上下文管理

**上下文窗口约束:**
- "Lost in the Middle" 现象: 中间位置信息容易被忽视
- 上下文窗口限制 $L_{max}$ 硬约束

**记忆层次架构:**
- **工作记忆**: 当前对话窗口内上下文
- **短期记忆**: 会话级持久化
- **长期记忆**: 跨会话外部存储
- **MemGPT**: OS启发的虚拟上下文管理 (Main Context = RAM, External Context = Disk, 通过Function Call实现页面调度/自主换页)
- **MemoryBank**: 基于艾宾浩斯遗忘曲线动态调整记忆强度，考虑时间和重要性
- **ReadAgent**: 分集分页 → 记忆要点提取 → 交互式查找
- **PagedAttention (vLLM)**: 操作系统虚拟内存和分页技术启发的KV Cache管理
- **CAMELoT**: 认知工作空间模型
- 记忆存储组织: 中心化系统 (高效但难以扩展)、去中心化系统 (减少上下文溢出但增加响应时间)、混合方案
- 记忆饱和问题: 过多的历史交互存储导致检索效率低下

**上下文压缩技术:**
- **ICAE** (In-context Autoencoder): 自动编码器压缩，实现**4×上下文压缩**，将长上下文压缩为紧凑记忆槽
- **RCC** (Recurrent Context Compression): 循环压缩 + 指令重构
- **ACRE** (Activation Refilling): **双层KV Cache** (L1全局信息 + L2详细信息)，动态回填
- **LLMLingua / LongLLMLingua**: 提示压缩框架
- **PagedAttention (vLLM)**: 分页KV Cache管理
- **StreamingLLM**: 保留Attention Sink tokens + 最新KV Cache，**22.2×加速**，处理400万token
- **H₂O (Heavy Hitter Oracle)**: 基于注意力分数的KV Cache淘汰，吞吐量提升**29×**，延迟降低**1.9×**
- **Rolling Buffer Cache**: 固定注意力跨度，32K token序列内存降低约**8×**
- **KV Cache量化**: KIVI, KVQuant
- **KCache**: K Cache存高带宽内存，V Cache存CPU内存
- **分布式处理**: DistAttention跨GPU集群分布式注意力计算

KV Cache大小公式: `2 × L × n_kv × d_head × s × b`
- MHA > MQA > GQA 对缓存的压缩效果

---

## 5. 系统实现层详解

### 5.1 检索增强生成 (RAG) — 五阶段演进

| 范式 | 时间 | 核心特征 |
|------|------|----------|
| **Naïve RAG** | 2020-2021 | 简单 Index → Retrieve → Generate 线性管道 (Lewis et al., 2020) |
| **Advanced RAG** | 2022-2023 | 检索前/后优化: 滑动窗口分块、重排序、查询重写 |
| **Modular RAG** | 2023-2024 | 乐高式可重组模块架构, 支持线性/条件/分支/循环流程; FlashRAG工具包 |
| **Graph RAG** | 2024 | 知识图谱增强: Microsoft GraphRAG (实体抽取→社区检测Leiden算法→层次摘要), LightRAG (双层检索), KAG |
| **Agentic RAG** | 2024-2025 | 自主检索决策: Self-RAG, ReAct驱动编排, AirRAG (MCTS推理), 多智能体协调 |

**Modular RAG 三层架构**: L1 Module → L2 Sub-module → L3 Operator
- 算子包括: Query Expansion, Hybrid Retrieval, Re-ranking, Compression, Verification

### 5.2 记忆系统

**三种记忆类型:**
- **情景记忆 (Episodic)**: 过去对话摘要
- **语义记忆 (Semantic)**: 向量数据库中的历史交互
- **偏好记忆 (Preference)**: 稳定的用户偏好事实

**关键设计:**
- 记忆层次: 短期 → 工作 → 长期
- 重构记忆: 片段化存储 + 吸引子动力学, AI驱动间隙填充
- 选择性遗忘: 艾宾浩斯遗忘曲线建模
- 冲突消解: 新旧信息一致性维护
- **Mem0**: 多层级记忆 (User/Session/Agent) + 嵌入检索
- **MemOS**: 记忆操作系统抽象

### 5.3 工具集成推理

- **ReAct范式**: Thought → Action → Observation 循环
- **Function Calling**: 标准化API接口调用
- **Toolformer**: 自监督工具学习
- **Gorilla**: LLM驱动的工具API调用
- **ToolLLM**: 大规模工具集成框架
- **工具发现**: 语义搜索工具描述 (准确率提升3倍)
- **执行环境**: 沙箱化 (Pyodide WASM), 权限系统

### 5.4 多智能体系统

**通信协议标准化:**
- **MCP** (Model Context Protocol): Agent-工具通信
- **A2A** (Agent-to-Agent): Agent间通信
- **ACP** (Agent Communication Protocol): 通用通信协议

**架构模式:**
- 主从架构 (Supervisor Pattern)
- 对等网络 (Peer-to-Peer)
- 层次化编排 (Hierarchical)

**协调机制:**
- 任务分解 → 分配 → 协调 → 聚合
- 共识机制: 投票、法定人数
- 信誉机制、博弈协商

**主要框架**: AutoGen, MetaGPT, CrewAI, LangGraph Swarm, OpenAI Swarm

---

## 6. 核心发现: 理解-生成不对称性

这是论文最重要的洞察——当前LLM存在**根本性的不对称**:

| 能力 | 状态 |
|------|------|
| **理解复杂上下文** (长文档、多模态、多轮对话、Agent编排) | ✅ 卓越 |
| **生成同等复杂的长篇输出** | ❌ 显著局限 |

**关键证据:**
- 推理链从10步扩展到100步: 性能下降 **73%** (KV Cache中间信息丢失)
- 输出从1K → 10K tokens: 事实一致性从 **92% → 45%**
- 长文本人类接受率: **88% → 22%**
- GAIA基准测试: 人类准确率92%, 最先进AI仅15%

弥合这一"理解-生成鸿沟"被确定为**未来研究的首要任务**。

---

## 7. 未来研究方向

1. **统一理论框架** — 基于信息论、贝叶斯优化的数学基础
2. **长篇生成技术** — 逻辑连贯性、事实一致性、规划深度的系统提升
3. **记忆增强架构** — 更高效的多层次记忆系统
4. **跨模态无缝集成** — 文本/视觉/音频/图/表格的统一上下文
5. **注意力效率** — 突破二次复杂度, KV Cache优化
6. **协议标准化** — Agent通信、工具调用、多智能体编排的统一标准
7. **安全与伦理** — 鲁棒性、透明度、公平性、生产可靠性
8. **评估体系** — 组件级诊断 + 系统级集成评估 + 应用特定基准

---

## 8. 关键挑战

- 理论基础落后于经验进展
- 长上下文处理计算开销巨大
- 幻觉和上下文不忠实性依然存在
- 研究碎片化: RAG/记忆/工具/Agent被孤立研究
- **评估鸿沟**: 缺乏面向上下文工程系统的标准化基准
- 当前指标 (BLEU, ROUGE等) 不足以捕获组合式/多步/协作行为

---

## 9. 技术演进脉络 (2020-2025)

```
2020-2021: 基础架构阶段
  ├── 简单RAG (Elasticsearch + LLM拼接)
  ├── 神经图灵机 (NTM)
  └── 基础工具调用

2022-2023: 优化迭代阶段
  ├── 向量数据库检索 (Milvus, FAISS)
  ├── 记忆网络 (MemNN) + 注意力机制
  ├── CoT/ToT/GoT 推理提示
  └── 多工具协同

2024-2025: 智能融合阶段
  ├── Agentic RAG (自主检索决策)
  ├── 情境记忆系统 (三维记忆模型)
  ├── GraphRAG (微软/LightRAG/KAG)
  ├── 多智能体社会智能 (信誉机制、博弈协商)
  └── MCP/A2A标准化协议
```

**演进主线**: 单点能力 → 记忆增强 → 工具扩展 → 集体协作

---

## 10. 实践技术栈

| 层级 | 技术选型 |
|------|----------|
| 检索层 | ChromaDB + FAISS + 交叉编码器重排序 |
| 内存层 | Redis (短期) + 对象存储 (长期) |
| 编排层 | LangChain/LangGraph + Kubernetes |
| 监控层 | Prometheus + Grafana |
| 通信协议 | MCP (Model Context Protocol) |

---

## 11. Context Engineering 四类操作策略

| 策略 | 目的 | 实现 |
|------|------|------|
| **Write** | 创建清晰有用的上下文 | StateGraph + 类型化状态; InMemoryStore长期记忆 |
| **Select** | 仅选取最相关信息 | RAG检索; 语义工具选择; 嵌入索引记忆 |
| **Compress** | 压缩上下文节省token | 摘要节点; 工具输出实时压缩 (token减少~48%) |
| **Isolate** | 隔离不同上下文类型 | Sub-agent架构; 沙箱环境; Pydantic schema状态隔离 |

---

## 12. 论文评价与定位

这篇综述是目前**Context Engineering领域最权威的系统性分类框架**, 核心贡献在于:

1. **统一了碎片化的研究**: 将RAG、记忆系统、工具使用、多智能体等孤立领域纳入统一框架
2. **提出了形式化定义**: 超越Prompt Engineering, 给出了上下文工程的数学形式化
3. **揭示了关键瓶颈**: 理解-生成不对称性为整个领域指明了方向
4. **建立了技术路线图**: 从基础组件到系统实现的完整层次结构, 对研究者和工程师均有指导价值

对于从事RAG架构、LLM Agent、记忆增强系统、上下文优化的工程师和研究者, 这篇论文是必读的基础参考文献。
