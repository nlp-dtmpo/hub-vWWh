### 前言

> 以下只对DS-V3论文中的Intro部分做了阅读笔记

在读开源模型的论文时, 总能被其作者的宏大胸襟所感染, 颇有"安得广厦千万间"之姿. 以下是DeepSeek - V3论文的Intro部分, 
我能确信这帮人是真的为了自己所做的事业而自豪的, 这些人是清楚自己正在为人类开疆扩土的, 粘在下方:

<img width="1460" height="950" alt="image" src="https://github.com/user-attachments/assets/788cdaeb-4976-4f1a-94fd-b1d608a76db9" />

---

### 模型的architecture

在上面的图片中, 可以看到对应的高光部分, 介绍了DS-V3的基础架构:
1. 延续DS-V2的架构, 采用`Multi-head Latent Attention (MLA) `保证推理的高效, 使用`DeepSeekMoE`来降低成本
2. V3新增的部分在于:
```text
Firstly, DeepSeek-V3 pioneers an auxiliary-loss-free strategy (Wang et al., 2024a) for load balancing, with the aim of
minimizing the adverse impact on model performance that arises from the effort to encourage
load balancing. Secondly, DeepSeek-V3 employs a multi-token prediction training objective,
which we have observed to enhance the overall performance on evaluation benchmarks.
```
即引入了两个新技术
  + Auxiliary-loss-free Strategy (无辅助损失策略), 替换掉传统MoE中使用`辅助损失函数`的方案, 保护了模型原生性能
  + Multi-token Prediction (MTP, 多 Token 预测), 同样提升了模型的整体性能


#### 关键名词的解释
+ MLA (Multi-head Latent Attention, 多头潜在注意力) 在DeepSeek-V2时期被提出
```text
MLA 是 DeepSeek 提出的一种创新的注意力机制，旨在解决大模型推理时 KV Cache（键值缓存） 占用显存过大的痛点。

核心原理：传统的 Multi-head Attention (MHA) 需要存储所有层的 Key 和 Value 向量。
MLA 通过低秩自适应压缩（Low-rank Compression）技术，将 KV 向量压缩成一个极小的“潜在向量”（Latent Vector）。

技术优势：
大幅降低显存占用：在推理时，MLA 仅需缓存极压缩后的潜在向量，其 KV Cache 消耗仅为传统 MHA 的几分之一。
性能无损：尽管进行了大幅压缩，其模型表现甚至优于标准的 Multi-head Attention，实现了效率与效果的平衡。
```

+ DeepSeekMoE (DeepSeek Mixture-of-Experts, 混合专家架构)
```text
DeepSeekMoE 是对传统 MoE（如 GShard 架构）的改进版，核心在于更精细的专家分工策略。
核心特征：
细粒度专家切分 (Fine-Grained Expert Specialization)：将传统的“大专家”拆分为多个更小的“细粒度专家”，
使得模型在激活相同参数量的情况下，能够更灵活地组合不同的专业知识。

共享专家 (Shared Experts)：始终激活一部分特定的专家处理所有输入。
这些共享专家负责捕获公共的、通用的知识，从而减少其他特定专家之间的冗余信息。

技术优势：
更高的参数效率：在保持计算量（FLOPs）不变的情况下，显著提升了模型的总参数量和性能。
缓解知识覆盖问题：通过共享专家设计，有效解决了 MoE 模型中专家之间知识冲突或遗忘的问题。
```

+ Load Balancing (负载均衡)
```text
在混合专家模型（MoE, Mixture-of-Experts）架构中，负载均衡是指将输入的 Token 均匀地分配给不同的专家（Expert）处理。
如果分配不均（某些专家过载，某些专家闲置），会降低计算效率并导致硬件利用率低下。
```

+ Auxiliary-loss-free Strategy (无辅助损失策略)
```text
传统的 MoE 模型通常引入“辅助损失”函数来强制专家间的负载均衡，但这往往会偏离主训练目标，导致模型表征能力下降。
DeepSeek-V3 采用的方法在不引入额外损失项的情况下实现均衡，从而保护了模型的原生性能。
```

+ Multi-token Prediction (MTP, 多 Token 预测)
```text
传统的语言模型训练目标是“下一个 Token 预测”（Next-token prediction）。
而 MTP 允许模型在每个位置同时预测未来的多个 Token。
这不仅能提供更强的监督信号，加快训练收敛，还能显著提升模型在推理时的思维链（CoT）能力和生成效率。
```

+ Evaluation Benchmarks (评估基准)
```text
指用于衡量模型能力的标准化测试集或评估框架（如 MMLU, GSM8K, HumanEval 等），涵盖了语言理解、数学推理、代码生成等多个维度。
```

#### 衍生术语
+ MHA (Multi-head Attention, 多头注意力)
```text
MHA 是 Transformer 架构中的标准注意力机制，也是 MLA 改进前的基准。

工作原理：
将输入的查询（Query）、键（Key）和值（Value）分别通过不同的线性变换，拆分成多个独立的“头”（Heads）。
每个头独立计算注意力，最后将结果拼接。

主要特点：
允许模型同时关注序列中不同位置、不同子空间的信息（例如一个头关注语法，另一个头关注语义）。

痛点：
在推理（Inference）过程中，每个头都需要缓存完整的 K 和 V 向量，随着序列变长，显存占用（KV Cache）会呈线性增长，成为大规模部署的瓶颈。
```

+ 传统 MoE (Traditional Mixture-of-Experts, 混合专家架构)
```text
传统 MoE（如 GShard 或 Switch Transformer）通过增加参数总量但不增加单次计算量来提升模型容量。

工作原理：将模型中的全连接层（FFN）替换为多个并行的“专家”层，并引入一个路由网络（Router/Gating Network）。
对于每个输入 Token，路由器只选择得分最高的 1 到 2 个专家进行计算。

主要特点：
稀疏激活：虽然总参数量巨大，但每次推理仅激活一小部分参数，计算开销较低。
粗粒度分工：专家通常体量较大且相互独立，容易出现专家利用率不均或知识冗余的问题。

痛点：难以平衡“负载均衡”与“模型性能”，且由于缺乏共享专家，容易导致通用知识在不同专家间重复学习，降低了参数效率。
```

---

### 模型的Training

原文如下:

<img width="1439" height="707" alt="image" src="https://github.com/user-attachments/assets/5909cc04-8189-45e6-a400-ba1e884e7cf8" />

#### 训练框架

此处作者指出, 低精度训练已经成为一种更具前景的高效训练方案, DS给出的即: FP8 混合精度训练

> FP8 混合精度训练是一种利用 8 位浮点数格式来大幅提升深度学习训练效率的技术。它是继 FP16/BF16 之后，大模型训练演进的关键步骤。
> 即, 在训练过程中，不再全程使用高精度的 32 位（FP32）或 16 位（BF16）浮点数，而是在计算密集型的操作（如矩阵乘法）中切换为 8 位（FP8） 格式。

DS-V3应用FP8混合精度训练得到的成果:
1. 超大规模模型上依旧有效
2. 训练加速
3. 降低了GPU显存

#### 框架技术

1. 开发设计`DualPipe 算法` , 减少GPU的闲置等待时间
2. 使用`计算与通信重叠（Computation-communication overlap）`, hide训练过程中绝大部分通信开销,
即使扩大规模后, 只要满足恒定的计通比（Computation-to-communication ratio）,依然可以实现近乎0通信开销
3. 开发`跨节点 All-to-all 通信算子（Kernels）`, 充分利用了带宽
4. 优化`显存占用（Memory footprint）`, 进一步降低了成本

#### 术语解释

+ DualPipe (双向流水线)
```text
DeepSeek 专门为 V3 开发的流水线并行优化算法。
它通过精细调度前向和后向计算任务，极大地减少了硬件在等待数据时的空闲时间（即“pipeline bubbles”）。
```

+ Pipeline Bubbles (流水线气泡)
```text
在分布式训练中，由于各阶段计算存在依赖关系，导致 GPU 在某些时刻处于闲置等待状态。
这些空闲时间被称为“气泡”，优化目标就是让气泡尽可能小。
```

+ Computation-Communication Overlap (计算通信重叠)
```text
一种优化策略，让 GPU 在进行矩阵运算（计算）的同时，通过独立通道进行数据传输（通信）。
如果能实现完美重叠，通信开销就会被计算时间“掩盖”，从而提升整体效率。
```

+ All-to-all Communication (全对全通信)
```text
在 MoE 架构中，不同 Token 需要被发送到不同的专家（可能在不同节点上）。
All-to-all 指的是所有参与计算的节点相互交换数据的通信模式，它是 MoE 规模化扩展的主要瓶颈。
```

+ InfiniBand (IB) & NVLink
```text
NVLink：用于服务器内部（GPU 与 GPU 之间）的高速互联技术。

InfiniBand：用于服务器集群之间（节点与节点之间）的高速网络协议，常用于高性能计算（HPC）。
```

+ Tensor Parallelism (TP, 张量并行)
```text
将单个参数矩阵拆分到多个 GPU 上计算。虽然能处理超大模型，但 TP 在节点间会产生巨大的通信压力。
DeepSeek-V3 成功避免了使用 TP，证明了其在显存优化和模型切分策略上的领先。
```

+ Memory Footprint (显存占用/显存足迹)
```text
指程序运行过程中实际消耗的 GPU 显存总量，包括模型参数、梯度、优化器状态以及中间激活值（Activations）。
```

---

### 预训练与后训练

<img width="1397" height="435" alt="image" src="https://github.com/user-attachments/assets/61000874-8b45-4d1b-9ebc-c92d2d08fa2a" />

还有下一页3个单词
```text
and generation length
```

#### PreTraining

论文中提到了, 预训练分为了两个Stage, 第一个阶段扩展到32K长度, 第二阶段扩展到128K长度

#### PostTraining

后训练采用了有监督微调(SFT)以及强化学习, 并且从R1中蒸馏(Distill)得到了推理能力, 并尽可能在准确度和生成长度之间保持平衡

#### 术语解释

+ 知识蒸馏 (Knowledge Distillation)
```text
一句话概括:
通过“教师-学生”机制实现能力迁移

核心逻辑：
将一个性能强大但体量巨大或计算昂贵的模型（教师模型，如 DeepSeek-R1）所学到的知识，通过训练迁移到一个更轻量或更高效的模型（学生模型，如 DeepSeek-V3）中。

具体做法：
在训练学生模型时，不仅让它学习原始的标注数据，还让它去模仿教师模型的“输出表现”（如概率分布、推理逻辑或思维链）。

目的：
让学生模型在保持较快推理速度和较低成本的同时，尽可能获得教师模型的高级智能（如复杂的逻辑推理能力）。在不大幅增加参数量的情况下提升智能水平。
```

+ SFT (Supervised Fine-Tuning, 有监督微调)
```text
后训练的第一阶段，使用人工标注的“指令-回答”对来训练基座模型，使其学会遵循人类指令并以正确的对话格式输出。
```

+ RL (Reinforcement Learning, 强化学习)
```text
通过奖励模型（Reward Model）或规则对模型的回答进行评分，引导模型学习哪些回答更符合人类偏好（如有用性、诚实性、安全性）。
```

> 以上两个后训练过程都是在align人类的preference


---

### Summary

往后论文分别总结了DS的低成本, 核心贡献, 以及成果展示, 这里不再给出

总而言之, V3是DS系列的里程碑版本, 论文给出了许多突破性的贡献, 尤其是降本增效这一点, DS做的真的无可挑剔.






