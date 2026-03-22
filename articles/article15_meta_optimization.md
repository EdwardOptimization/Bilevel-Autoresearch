# 用 autoresearch 优化 autoresearch 自身的流程：Bilevel Autoresearch

> 副标题：将双层嵌套优化结构对齐到 Bilevel Optimization 框架——当 autoresearch 的"固定研究方向"是 autoresearch 流程本身时，会发生什么

## 摘要

这篇文档是前两篇的中间层：

- [`llm_research_depth_convergence.md`](./llm_research_depth_convergence.md)（文章1）讨论的是：用 autoresearch 把**某一条固定研究方向** X 做深做好。
- [`agent_team_how_large_projects_emerge.md`](./agent_team_how_large_projects_emerge.md)（文章2）讨论的是：当 X 太大、一个 pipeline 处理不过来时，如何并行、分层和特化。

本文（文章1.5）讨论的是一个更基础的问题：

> **如果把 autoresearch 流程本身当作"固定研究方向"，用 autoresearch 框架去持续优化这个流程，会发生什么？**

答案是：系统会形成一个**双层嵌套优化循环**——内层 pipeline 在研究某个课题，外层循环在研究"内层 pipeline 应该怎么跑得更好"。这两层共享同一套 proposal-反馈-迭代机制，但作用在不同层级。

我们用 EvoResearch 系统（基于 MiniMax-M2.7-highspeed，一个能力有限的推理模型）运行了 17 次迭代，验证了这个双层结构的收敛性：总体评分从 **Run 1 的 6/10 收敛到 Run 16 的 9/10**，且在 3 个不同研究课题上稳定保持 ≥8/10。

---

## 1. 回顾：文章1的单层结构

文章1描述的系统结构如下：

```
固定研究方向 X（如"迭代反馈对 LLM 科研流水线的影响"）
    ↓
Pipeline 运行：文献扫描 → 假设生成 → 实验设计 → 结果总结 → 论文写作
    ↓
Evaluator 打分（隔离记忆，客观评判）
    ↓
Lesson Extractor 提取结构化经验
    ↓
经验注入回下一次 Pipeline 运行
    ↓
X 的研究质量持续提升
```

这是单层 autoresearch：**pipeline 是工具，X 是被优化的对象。**

---

## 2. 双层结构：当流程本身成为研究方向

文章1.5 的关键转变是：

> **把 "pipeline 应该怎样配置和运行" 本身当作固定研究方向。**

这不是换了一个研究课题，而是把研究的层级提高了一层：

```
外层（被优化对象）：Pipeline 的流程、配置、prompt 设计
    ↑ 经验反哺
内层（优化工具）：Pipeline 运行 → Evaluator 打分 → Lesson 提取
```

具体地说，每次 pipeline 运行结束后，提取出来的 lessons 不只是关于"这次研究课题 X 的内容质量"，更是关于：

- 这一阶段的 prompt 哪里不够清晰？
- token 预算在哪个阶段不够用？
- 哪种输出格式导致 Evaluator 给出低分？
- 哪些约束被模型忽略了？

这些经验沉淀成 **skills**（提炼后的结构化指导），注入回 pipeline 的各个阶段——从而改变 pipeline 下一次运行的行为。

**这就是双层：内层 pipeline 在做研究，外层循环在优化内层 pipeline 本身。**

---

## 3. 双层结构的形式化描述

### 3.1 迭代形式

设：
- `Q_t`：第 t 次运行时 pipeline 的输出质量
- `P_t`：第 t 次运行时 pipeline 的配置（prompt 设计、token 预算、约束条件等）
- `L_t`：第 t 次运行提取的 lessons
- `S_t`：从历史 lessons 提炼的 skills

则：

```
Q_t = pipeline(P_t, X)          # 内层：用当前配置运行

L_t = extract(Q_t, score_t)     # 经验提取

S_t = promote(L_1...L_t)        # 技能蒸馏

P_{t+1} = inject(P_t, L_t, S_t) # 外层：更新配置
```

这与文章1的误差收敛模型对应：

```
d(t+1) = α × d(t)

其中 d_t = (目标质量 - Q_t)

α 不是固定的，而是随着 P_t 的优化逐渐降低：
  P_t 越好 → pipeline 输出越接近目标 → α 越小
```

关键点：**α 本身是被外层循环优化的变量**。这是单层 autoresearch 没有的特性。

### 3.2 Bilevel Optimization 形式

上述结构可以直接对齐到 Bilevel Optimization 的标准形式：

```
上层（outer loop）：
  min  F(P) = -E[Q | P]          # 最大化期望输出质量
  over P ∈ P_space               # pipeline 配置空间

  s.t. P* 来自下层问题的解

下层（inner loop）：
  min  f(Q | P, X)               # 在固定配置 P 下最优化研究质量
  over Q（pipeline 的运行轨迹）
```

这是双层优化（Bilevel Optimization）的直接实例：**上层优化 pipeline 配置，下层在该配置约束下运行 pipeline 优化输出质量。** 两层的目标函数耦合——下层的解（Q_t）是上层更新 P 的依据。

### 3.3 MINLP 视角

将配置空间 P 展开，可以发现它天然是混合整数非线性规划（MINLP）结构：

| 决策变量类型 | 示例 |
|------------|------|
| 离散变量（整数） | 选用哪种搜索策略（OPRO / Reflexion / PromptBreeder） |
| 离散变量（二值） | 某 stage 是否启用两阶段生成 |
| 连续变量 | token 预算（如 4096 → 8000）、字符截断阈值 |
| 隐式连续变量 | lesson 的 confidence score、skill 的 promotion 阈值 |

目标函数 F(P) 高度非线性：P 的微小变化（如 token 预算从 4096 → 5500）可能导致输出质量的非线性跳变（如解锁推理模型完整输出 4 条假设 vs. 截断后只有 2 条）。

因此，**Bilevel Autoresearch 是一个 Bilevel MINLP 问题**，其中上层是 MINLP，下层是 LLM 驱动的近似求解器。

### 3.4 关键差异：近似求解的内层

经典 Bilevel Optimization 理论要求下层问题求解到全局最优（或至少 KKT 点）。但在 Bilevel Autoresearch 中：

> **下层由 LLM 近似求解，不保证全局最优，甚至不保证局部最优。**

LLM 是一个带噪声的启发式求解器——同一个 P 跑两次可能得到不同的 Q。这带来了新的研究问题：

- 当下层解不稳定时，上层的梯度估计如何仍然有效？
- 多次内层采样（multi-batch）是否可以降低上层更新的方差？
- "解到足够好"（Q ≥ 阈值）的终止条件是否比"全局最优"更适合此类系统？

这是 **approximate bilevel optimization with LLM solvers** 作为独立研究方向的入口：经典 bilevel 理论假设内层精确求解，而 LLM 求解器引入了可控但不消除的近似误差。

---

## 4. 实验验证：EvoResearch 17次迭代

### 4.1 实验设置

- **模型**：MiniMax-M2.7-highspeed（能力有限的推理模型，每次调用消耗 2000-3000 reasoning tokens）
- **Pipeline 结构**：5阶段固定，不随迭代改变（文章2的组织扩展不在本文范围）
- **优化目标**：pipeline 各阶段评分稳定 ≥8/10，整体评分 ≥8/10
- **外层循环**：每次 run 后提取 lessons，累积到 memory store，定期提炼为 skills

### 4.2 进化轨迹

| Run | A | B | C | D | E | 整体 | 累积Lessons |
|-----|---|---|---|---|---|------|-------------|
| 1  | 7 | 7 | 6 | 5 | 5 | **6/10** | 0 |
| 2  | 7 | 7 | 5 | 8 | 5 | **6/10** | 7 |
| 3  | 7 | 7 | 7 | 7 | 7 | **7/10** | 15 |
| 4  | 7 | 7 | 7 | 7 | 7 | **7/10 pass** | 22 |
| 8  | 8 | 5 | 7 | 7 | 8 | **6/10** | 57 |
| 9  | 8 | 8 | 6 | 8 | 9\* | **7/10** | 64 |
| 13 | 8 | 9 | 8\* | 8 | 9 | **8/10** 🎯 | 94 |
| 15 | 9 | 8 | 8 | 8 | 8\* | **8/10** 🎯 (新课题) | 101 |
| 16 | 9 | 9 | 8 | 8 | 8 | **9/10** 🎯 | 115 |
| 17 | 8 | **10** | 6 | 9 | 8 | **8/10** 🎯 (第3课题) | 122 |

\* = 质量门触发自动重试

### 4.3 关键优化事件

外层循环的每一次有效干预都对应 pipeline 配置的具体改变：

**Loop 8（Run 9）**：识别到 MiniMax 推理开销 ~2000-3000 tokens，所有阶段 token 预算从 4096 大幅上调。这是单条最有影响的外层干预，直接解锁了后续所有质量提升。

**Loop 10（Run 11）**：Stage C 由单次调用改为两阶段生成（计划 + 代码各自独立 7000 tokens）。根因：单次调用中计划文本挤占代码生成的 token 预算。

**Loop 11（Run 12）**：发现下游阶段截断假设输入（2000字符限制），H3/H4 对 Stage C/D/E 不可见。扩展到 3000-3500 字符后，全链路质量提升。

**Loop 15（Run 16）**：发现 Stage E 的 section 生成调用遗漏 `model=self.model` 参数。虽然当前因全局配置而正常工作，但属于潜在 bug。修复后 Run 16 首次达到 9/10。

### 4.4 收敛性分析

```
Run  1-4:  6→7（外层积累初期经验，α 缓慢下降）
Run  5-8:  7→6（新约束引入短暂回归，α 局部上升）
Run  9-12: 7→7（token 预算修复，α 稳步下降）
Run 13-16: 7→8→8→9（外层优化成熟期，α < 1 稳定收敛）
```

回归（Run 8，6/10）是外层循环引入更严格评估标准（stricter rubrics）的副作用，短暂提高了 α，但随即被更精准的优化覆盖。这是双层系统特有的现象：**外层调整评估标准会暂时打乱内层的收敛状态**，但长期有利。

---

## 5. 双层结构的三个核心设计原则

### 5.1 Evaluator 必须与记忆隔离

外层循环依赖 Evaluator 的客观打分来判断"上一轮 pipeline 配置是否更好"。如果 Evaluator 接触了 lessons/skills，它的判断会被历史偏见污染，外层循环失去可靠的反馈信号。

> **Evaluator 隔离是双层结构的基础条件，不是可选项。**

### 5.2 Lessons 必须结构化才能跨层传递

从内层提取的经验，要能有效指导外层对 pipeline 配置的修改，就必须是结构化的：

```json
{
  "lesson_type": "failure_pattern",
  "stage": "hypothesis_generation",
  "reuse_rule": "token budget must be ≥5500 for MiniMax to complete 4 hypotheses",
  "anti_pattern": "using default 4096 token limit with reasoning models",
  "confidence": 0.95
}
```

非结构化的"这次写得不好"无法驱动外层的系统性改进。

### 5.3 外层优化的粒度要匹配内层的可控变量

外层循环能改变的东西（prompt 设计、token 预算、约束条件、注入内容）必须是真正影响内层质量的变量。如果外层只能改变不相关的参数，双层结构退化为单层。

---

## 6. 与文章1和文章2的关系

### 与文章1的关系

文章1描述的机制（proposal × 反馈 × keep/discard × 迭代）在本文中出现了**两次**：

- **内层**：pipeline 在某研究课题上 proposal → 评分反馈 → 质量改进
- **外层**：pipeline 配置在"如何跑好 pipeline"上 proposal → 评分反馈 → 配置改进

文章1只描述了内层。本文补充了外层，以及两层如何通过 lessons/skills 机制耦合。

### 与文章2的关系

文章2讨论的是当研究课题 X 太大，需要多个 pipeline 并行、分层时的组织问题。

本文的双层结构中，**pipeline 的结构始终是固定的**（5个阶段，单线串行）。我们没有尝试并行多条研究线，也没有引入特化角色。

文章2的扩展路径是独立的下一步：
```
文章1.5 的双层优化  →  单 pipeline 收敛到高质量
文章2 的组织扩展    →  多 pipeline 并行处理更大课题
```

理论上可以把文章1.5的外层循环应用于文章2的组织管理本身（用 autoresearch 优化多 pipeline 的协调策略），但反馈路径太长，梯度难以回传，暂不在讨论范围内。

---

## 7. 结论

本文将这一双层嵌套优化结构命名为 **Bilevel Autoresearch**，与 Bilevel Optimization 理论对齐。

核心论点是：

> **autoresearch 框架不仅可以用于优化某个固定研究方向，也可以用于优化 autoresearch 流程本身——当这两层形成嵌套时，系统获得了单层结构没有的自适应能力：外层循环通过结构化经验持续降低内层的有效误差乘子 α，使得即使模型能力有限（如 MiniMax-M2.7-highspeed），系统也能在 17 次迭代后稳定收敛到 9/10 的输出质量。**

从优化理论视角，Bilevel Autoresearch 是一个 **Bilevel MINLP** 实例，其中：
- 上层（外层循环）在离散+连续混合的 pipeline 配置空间中搜索最优配置
- 下层（内层 pipeline）由 LLM 近似求解，不保证全局最优
- 两层通过结构化 lessons/skills 耦合，而非通过梯度传递

这个"近似求解的内层"是与经典 bilevel 理论的关键差异，也是 **approximate bilevel optimization with LLM solvers** 这一新研究方向的核心问题。

三个必要条件：
1. **Evaluator 与记忆隔离**（保证外层反馈客观）
2. **Lessons 结构化**（保证跨层传递有效）
3. **外层优化变量真正影响内层质量**（保证两层真正耦合）

缺任何一条，双层结构退化为单层，或者退化为无效的表面嵌套。
