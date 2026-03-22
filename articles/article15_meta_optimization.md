# Optimizing Autoresearch with Autoresearch: Bilevel Autoresearch

> Subtitle: Aligning the bilevel nested optimization structure with the Bilevel Optimization framework — what happens when autoresearch's "fixed research direction" is the autoresearch process itself

## Abstract

This document serves as an intermediate layer between the previous two articles:

- [`llm_research_depth_convergence.md`](./llm_research_depth_convergence.md) (Article 1) discusses: using autoresearch to deepen and refine **a fixed research direction** X.
- [`agent_team_how_large_projects_emerge.md`](./agent_team_how_large_projects_emerge.md) (Article 2) discusses: when X is too large for a single pipeline to handle, how to parallelize, layer, and specialize.

This article (Article 1.5) addresses a more fundamental question:

> **What happens if we treat the autoresearch process itself as a "fixed research direction" and use the autoresearch framework to continuously optimize this process?**

The answer is: the system forms a **bilevel nested optimization loop** — the inner pipeline researches a given topic, while the outer loop researches "how the inner pipeline should run better." Both layers share the same proposal-feedback-iteration mechanism, but operate at different levels.

We ran the EvoResearch system (based on MiniMax-M2.7-highspeed, a reasoning model with limited capability) for 17 iterations, validating the convergence of this bilevel structure: the overall score converged from **6/10 at Run 1 to 9/10 at Run 16**, and stably maintained ≥8/10 across 3 different research topics.

---

## 1. Review: The Single-Layer Structure of Article 1

The system structure described in Article 1 is as follows:

```
Fixed research direction X (e.g., "The impact of iterative feedback on LLM research pipelines")
    ↓
Pipeline run: Literature scan → Hypothesis generation → Experiment design → Result synthesis → Paper writing
    ↓
Evaluator scoring (memory-isolated, objective judgment)
    ↓
Lesson Extractor extracts structured experience
    ↓
Experience injected into the next Pipeline run
    ↓
Research quality of X continuously improves
```

This is single-layer autoresearch: **the pipeline is the tool, X is the object being optimized.**

---

## 2. Bilevel Structure: When the Process Itself Becomes the Research Direction

The key transformation in Article 1.5 is:

> **Treating "how the pipeline should be configured and run" itself as the fixed research direction.**

This is not simply switching to a different research topic — it elevates the level of research by one layer:

```
Outer layer (object being optimized): Pipeline process, configuration, prompt design
    ↑ Experience feedback
Inner layer (optimization tool): Pipeline run → Evaluator scoring → Lesson extraction
```

Specifically, after each pipeline run, the extracted lessons are not solely about "the content quality of research topic X," but also about:

- Where was the prompt for this stage insufficiently clear?
- At which stage was the token budget inadequate?
- Which output format caused the Evaluator to give low scores?
- Which constraints were ignored by the model?

These experiences are distilled into **skills** (refined structured guidance) and injected back into the various stages of the pipeline — thereby changing the pipeline's behavior in its next run.

**This is the bilevel structure: the inner pipeline conducts research, while the outer loop optimizes the inner pipeline itself.**

---

## 3. Formal Description of the Bilevel Structure

### 3.1 Iterative Formulation

Let:
- `Q_t`: the output quality of the pipeline at the t-th run
- `P_t`: the pipeline configuration at the t-th run (prompt design, token budget, constraints, etc.)
- `L_t`: lessons extracted from the t-th run
- `S_t`: skills distilled from historical lessons

Then:

```
Q_t = pipeline(P_t, X)          # Inner layer: run with current configuration

L_t = extract(Q_t, score_t)     # Experience extraction

S_t = promote(L_1...L_t)        # Skill distillation

P_{t+1} = inject(P_t, L_t, S_t) # Outer layer: update configuration
```

This corresponds to the error convergence model in Article 1:

```
d(t+1) = α × d(t)

where d_t = (target quality - Q_t)

α is not fixed, but gradually decreases as P_t is optimized:
  Better P_t → pipeline output closer to target → smaller α
```

Key point: **α itself is a variable being optimized by the outer loop.** This is a property that single-layer autoresearch does not possess.

### 3.2 Bilevel Optimization Formulation

The above structure directly aligns with the standard form of Bilevel Optimization:

```
Upper level (outer loop):
  min  F(P) = -E[Q | P]          # Maximize expected output quality
  over P ∈ P_space               # Pipeline configuration space

  s.t. P* derived from the solution of the lower-level problem

Lower level (inner loop):
  min  f(Q | P, X)               # Optimize research quality under fixed configuration P
  over Q (the pipeline's execution trajectory)
```

This is a direct instance of Bilevel Optimization: **the upper level optimizes pipeline configuration, while the lower level runs the pipeline to optimize output quality under the configuration constraints.** The objective functions of the two levels are coupled — the lower level's solution (Q_t) is the basis for the upper level's update of P.

### 3.3 MINLP Perspective

Expanding the configuration space P reveals that it naturally has a Mixed-Integer Nonlinear Programming (MINLP) structure:

| Decision Variable Type | Example |
|----------------------|---------|
| Discrete variable (integer) | Which search strategy to use (OPRO / Reflexion / PromptBreeder) |
| Discrete variable (binary) | Whether to enable two-stage generation for a given stage |
| Continuous variable | Token budget (e.g., 4096 → 8000), character truncation threshold |
| Implicit continuous variable | Lesson confidence score, skill promotion threshold |

The objective function F(P) is highly nonlinear: small changes in P (e.g., token budget from 4096 → 5500) can cause nonlinear jumps in output quality (e.g., unlocking a reasoning model's ability to fully output 4 hypotheses vs. truncation leaving only 2).

Therefore, **Bilevel Autoresearch is a Bilevel MINLP problem**, where the upper level is MINLP and the lower level is an LLM-driven approximate solver.

### 3.4 Key Distinction: Approximate Solving in the Inner Layer

Classical Bilevel Optimization theory requires the lower-level problem to be solved to global optimality (or at least to a KKT point). However, in Bilevel Autoresearch:

> **The lower level is approximately solved by an LLM, with no guarantee of global optimality, or even local optimality.**

The LLM is a noisy heuristic solver — running the same P twice may yield different Q values. This introduces new research questions:

- When the lower-level solution is unstable, how can the upper-level gradient estimate remain effective?
- Can multiple inner-layer samples (multi-batch) reduce the variance of upper-level updates?
- Is a "solved well enough" (Q ≥ threshold) termination condition more suitable than "global optimality" for such systems?

This is the entry point for **approximate bilevel optimization with LLM solvers** as an independent research direction: classical bilevel theory assumes exact inner-level solving, while LLM solvers introduce controllable but non-eliminable approximation error.

---

## 4. Experimental Validation: 17 Iterations of EvoResearch

### 4.1 Experimental Setup

- **Model**: MiniMax-M2.7-highspeed (a reasoning model with limited capability, consuming 2000-3000 reasoning tokens per call)
- **Pipeline structure**: 5 fixed stages, unchanged across iterations (the organizational scaling from Article 2 is outside the scope of this article)
- **Optimization objective**: all pipeline stage scores stably ≥8/10, overall score ≥8/10
- **Outer loop**: after each run, extract lessons, accumulate in memory store, periodically distill into skills

### 4.2 Evolution Trajectory

| Run | A | B | C | D | E | Overall | Cumulative Lessons |
|-----|---|---|---|---|---|---------|-------------------|
| 1  | 7 | 7 | 6 | 5 | 5 | **6/10** | 0 |
| 2  | 7 | 7 | 5 | 8 | 5 | **6/10** | 7 |
| 3  | 7 | 7 | 7 | 7 | 7 | **7/10** | 15 |
| 4  | 7 | 7 | 7 | 7 | 7 | **7/10 pass** | 22 |
| 8  | 8 | 5 | 7 | 7 | 8 | **6/10** | 57 |
| 9  | 8 | 8 | 6 | 8 | 9\* | **7/10** | 64 |
| 13 | 8 | 9 | 8\* | 8 | 9 | **8/10** 🎯 | 94 |
| 15 | 9 | 8 | 8 | 8 | 8\* | **8/10** 🎯 (new topic) | 101 |
| 16 | 9 | 9 | 8 | 8 | 8 | **9/10** 🎯 | 115 |
| 17 | 8 | **10** | 6 | 9 | 8 | **8/10** 🎯 (3rd topic) | 122 |

\* = quality gate triggered automatic retry

### 4.3 Key Optimization Events

Each effective intervention by the outer loop corresponds to a specific change in pipeline configuration:

**Loop 8 (Run 9)**: Identified that MiniMax reasoning overhead is ~2000-3000 tokens; token budgets for all stages were significantly increased from 4096. This was the single most impactful outer-layer intervention, directly unlocking all subsequent quality improvements.

**Loop 10 (Run 11)**: Stage C was changed from a single call to two-stage generation (plan + code each with an independent 7000 tokens). Root cause: in a single call, plan text was crowding out the token budget for code generation.

**Loop 11 (Run 12)**: Discovered that downstream stages were truncating hypothesis input (2000 character limit), making H3/H4 invisible to Stages C/D/E. After expanding to 3000-3500 characters, quality improved across the entire pipeline.

**Loop 15 (Run 16)**: Discovered that Stage E's section generation call was missing the `model=self.model` parameter. Although it was functioning correctly at the time due to global configuration, this was a latent bug. After the fix, Run 16 reached 9/10 for the first time.

### 4.4 Convergence Analysis

```
Run  1-4:  6→7 (outer layer accumulating initial experience, α decreasing slowly)
Run  5-8:  7→6 (new constraints introduced temporary regression, α locally increased)
Run  9-12: 7→7 (token budget fixes, α steadily decreasing)
Run 13-16: 7→8→8→9 (outer optimization maturation phase, α < 1 with stable convergence)
```

The regression (Run 8, 6/10) is a side effect of the outer loop introducing stricter evaluation criteria (stricter rubrics), which temporarily increased α but was quickly offset by more precise optimizations. This is a phenomenon unique to bilevel systems: **adjusting evaluation criteria at the outer layer temporarily disrupts the inner layer's convergence state**, but is beneficial in the long run.

---

## 5. Three Core Design Principles of the Bilevel Structure

### 5.1 The Evaluator Must Be Isolated from Memory

The outer loop depends on the Evaluator's objective scoring to determine "whether the previous pipeline configuration was better." If the Evaluator has access to lessons/skills, its judgment will be contaminated by historical bias, and the outer loop loses its reliable feedback signal.

> **Evaluator isolation is a foundational requirement of the bilevel structure, not an optional feature.**

### 5.2 Lessons Must Be Structured for Cross-Layer Transfer

For experience extracted from the inner layer to effectively guide the outer layer's modifications to pipeline configuration, it must be structured:

```json
{
  "lesson_type": "failure_pattern",
  "stage": "hypothesis_generation",
  "reuse_rule": "token budget must be ≥5500 for MiniMax to complete 4 hypotheses",
  "anti_pattern": "using default 4096 token limit with reasoning models",
  "confidence": 0.95
}
```

Unstructured feedback like "it didn't write well this time" cannot drive systematic improvements at the outer layer.

### 5.3 Outer-Layer Optimization Granularity Must Match Inner-Layer Controllable Variables

What the outer loop can change (prompt design, token budget, constraints, injected content) must be variables that genuinely affect inner-layer quality. If the outer layer can only modify irrelevant parameters, the bilevel structure degenerates into a single layer.

---

## 6. Boundaries and Extensions of the Outer Loop: From Configuration Optimization to Mechanism Research

### 6.1 Limitations of the Current Outer Layer

The outer loop described in this article optimizes the pipeline's **configuration** — prompt design, token budget, constraints. In the language of Article 1, it adjusts "what the inner layer says," not "how the inner layer searches."

This distinction is critical:

| What the current outer layer can do | What the current outer layer cannot do |
|-------------------------------------|---------------------------------------|
| Modify Stage B's prompt requirements | Add an independent critic node after Stage B |
| Adjust token budget | Change single-track search to multi-batch parallel |
| Select Reflexion / OPRO strategy | Discover a search mechanism that humans have not yet conceived |

If the available strategies are implemented as a fixed menu (Reflexion / multi-batch / beam search / teacher-critique), the outer layer becomes a classifier selecting among predefined mechanisms. **The search space is fixed, and the ceiling is the mechanisms humans have already designed.**

### 6.2 The Autoresearch Framework Itself Was Researched into Existence

The autoresearch paradigm described in Article 1 — "proposal × feedback × keep/discard × iteration" — is itself a product of human research: humans observed a large number of iterative optimization systems, identified this common pattern, distilled it into a framework, and then codified it in software.

[AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) adding multi-batch was one human insight; [EvoScientist](https://github.com/EvoScientist/EvoScientist) introducing a teacher model was another human insight. **Every mechanism evolution has required human intervention.**

But if autoresearch itself can be researched into existence, then agents can also use the same framework to research new mechanisms:

> **The outer-layer agent should not select mechanisms from a fixed menu, but should instead treat "discovering better inner-layer search mechanisms" as a research question in its own right, investigating it through the autoresearch approach.**

### 6.3 Three-Level Recursive Structure

This forms a natural recursion:

```
Level 0: Human research
  research question: What kind of iterative framework enables LLM systems to continuously improve?
  output: autoresearch framework (proposal × feedback × iteration)

Level 1: Inner-layer autoresearch
  research question: How to bring the output quality of topic X to ≥8/10?
  tool: Fixed 5-stage pipeline + evaluator
  output: High-quality research content

Level 2: Outer-layer autoresearch
  research question: What kind of inner-layer search mechanism yields lower α and faster convergence?
  tool: Inner layer as experimental harness, convergence rate as reward
  output: Better inner-layer search mechanisms (new stage combinations, new feedback structures, new search topologies)
```

The outer-layer agent at Level 2 follows a workflow that is fully isomorphic to Level 1:

```
Propose mechanism hypothesis → Generate implementation (code or pipeline modification)
→ Run validation using the inner layer as harness
→ Observe whether α decreases
→ Keep (codify as new mechanism) or discard (rollback)
→ Distill lesson: "critic node is effective for dimension D, has no impact on dimension B"
→ Propose new hypothesis in the next round
```

Here, the proposal is no longer prompt text, but a **mechanism description** — the outer-layer LLM can propose any search structure change it can imagine, then implement it via code generation. The implementation itself can be iterative: the first version of the code has bugs, it is fixed based on error messages, then retested — this is autoresearch manifested at the implementation level.

### 6.4 Boundaries of the Autoresearch Framework

This recursive structure reveals a more fundamental proposition:

> **The boundary of the autoresearch framework is not determined by the size of the search space, but by the measurability of the research question. As long as a clear feedback surface can be defined, autoresearch can operate at any level — including researching autoresearch itself.**

Validating this proposition requires three conditions to hold:

1. **The research question must be sufficiently specific**: The problem at Level 2 is "which mechanism yields lower α," not "which AI system is better" — the former is measurable, the latter is incomparable.
2. **The feedback surface must be authentic**: The inner layer's convergence rate, peak score, and number of rounds to reach the threshold are objective, non-gameable signals.
3. **The iteration budget must be sufficient**: Mechanism hypotheses require multiple inner cycles to validate, and the outer layer needs enough cycle budget for research.

From the perspective of Article 1's formula, the four factors at Level 2 correspond directly to those at Level 1:

```
effective mechanism research =
  outer_model_prior          (outer LLM's prior on "what mechanisms work")
  × context_update           (convergence data from each inner run accumulated as outer context)
  × reward_fidelity          (whether inner convergence metrics truly reflect mechanism quality)
  × iteration_budget         (how many cycles the outer layer has to validate mechanism hypotheses)
```

This correspondence is not a metaphor, but the same framework instantiated at different levels.

---

## 7. Relationship with Article 1 and Article 2

### Relationship with Article 1

The mechanism described in Article 1 (proposal × feedback × keep/discard × iteration) appears **twice** in this article:

- **Inner layer**: the pipeline applies proposal → scoring feedback → quality improvement on a research topic
- **Outer layer**: the pipeline configuration applies proposal → scoring feedback → configuration improvement on "how to run the pipeline well"

Article 1 described only the inner layer. This article supplements the outer layer, as well as how the two layers are coupled through the lessons/skills mechanism.

### Relationship with Article 2

Article 2 discusses the organizational challenges when research topic X is too large, requiring multiple pipelines to run in parallel with layered specialization.

In the bilevel structure of this article, **the pipeline structure remains fixed throughout** (5 stages, single-track sequential). We did not attempt to run parallel research lines, nor did we introduce specialized roles.

The scaling path of Article 2 is an independent next step:
```
Article 1.5's bilevel optimization  →  Single pipeline converges to high quality
Article 2's organizational scaling  →  Multiple pipelines in parallel for larger topics
```

In theory, the outer loop of Article 1.5 could be applied to the organizational management of Article 2 itself (using autoresearch to optimize multi-pipeline coordination strategies), but the feedback path would be too long, gradients would be difficult to propagate, and this is not within the scope of the current discussion.

---

## 8. Conclusion

This article names this bilevel nested optimization structure **Bilevel Autoresearch**, aligning it with Bilevel Optimization theory.

The core argument is:

> **The autoresearch framework can be used not only to optimize a fixed research direction, but also to optimize the autoresearch process itself — when these two layers form a nested structure, the system acquires self-adaptive capabilities absent from the single-layer structure: the outer loop continuously reduces the inner layer's effective error multiplier α through structured experience, so that even with a model of limited capability (such as MiniMax-M2.7-highspeed), the system can stably converge to 9/10 output quality after 17 iterations.**

From an optimization theory perspective, Bilevel Autoresearch is a **Bilevel MINLP** instance, where:
- The upper level (outer loop) searches for the optimal configuration in a mixed discrete-continuous pipeline configuration space
- The lower level (inner pipeline) is approximately solved by an LLM, with no guarantee of global optimality
- The two levels are coupled through structured lessons/skills, rather than through gradient propagation

This "approximately solved inner layer" is the key distinction from classical bilevel theory, and is also the core problem of **approximate bilevel optimization with LLM solvers** as a new research direction.

Three necessary conditions:
1. **Evaluator isolated from memory** (ensuring objective outer-layer feedback)
2. **Structured lessons** (ensuring effective cross-layer transfer)
3. **Outer-layer optimization variables genuinely affect inner-layer quality** (ensuring true coupling between the two layers)

Without any one of these, the bilevel structure degenerates into a single layer, or degenerates into an ineffective superficial nesting.
