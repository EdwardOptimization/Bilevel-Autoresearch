# Discussion Post for karpathy/autoresearch

**Title:** Bilevel Autoresearch: an outer loop that discovers new pipeline mechanisms via code generation

**Category:** Show and tell

**Posted:** https://github.com/karpathy/autoresearch/discussions/375

---

## What we built

We extended the autoresearch pattern with a second optimization loop. The inner loop runs the standard propose → evaluate → iterate cycle on a task. The outer loop treats the inner loop's *configuration* as its own optimization target — analyzing traces, diagnosing bottlenecks, and updating the pipeline.

The key question: autoresearch treats the pipeline structure as fixed. What happens when the pipeline itself becomes the research subject?

**Repo:** [EdwardOptimization/Bilevel-Autoresearch](https://github.com/EdwardOptimization/Bilevel-Autoresearch)

---

## Architecture

```
Inner loop: Task Input → Pipeline → Evaluator → Lessons → Better Output
                                ↑ improved prompts / config / mechanisms
Outer loop: Trace Analysis → Meta-Optimizer → Pipeline Config Updates
```

The framework is domain-agnostic. Our demo optimizes research articles against a 5-dimension rubric (Argumentative Rigor / Conceptual Clarity / Cross-Article Consistency / Insight Novelty / Actionability), but the inner loop can be anything with a measurable objective.

---

## Results

**Single-layer** (inner loop only, 17 runs):

```
Run 1:  A:7 B:7 C:6 D:5 E:5 → 6/10  (0 lessons)
Run 4:  A:7 B:7 C:7 D:7 E:7 → 7/10  (skills promoted)
Run 13: A:8 B:9 C:8 D:8 E:9 → 8/10  (122 lessons, 6 skills)
Run 16: A:9 B:9 C:8 D:8 E:8 → 9/10  (3rd topic)
```

**Dual-layer** (outer loop automated, 4 cycles × 5 runs):

```
Cycle 1: 7.2, 6.6, 6.6, 6.4, 6.4  (baseline)
Cycle 2: 6.6, 7.0, 7.0, 7.0, 7.0  ← outer loop intervention stabilized scores
```

Cycle 2 stability (4/5 runs at 7.0 vs declining in Cycle 1) is direct evidence of the outer loop working.

---

## Level 2: Mechanism Research via Code Generation

This is where it gets interesting. Prompt-level optimization has a ceiling — you can't discover a fundamentally new search mechanism by rewording a prompt. So we asked: **can the outer loop research new mechanisms the same way autoresearch researches any topic?**

The outer LLM (DeepSeek) runs a multi-round research session:
1. **Explore** — freely choose domains (optimization, cognitive science, formal logic, etc.) and generate hypotheses
2. **Critique** — score each hypothesis on impact × feasibility ÷ complexity
3. **Specify** — write a detailed implementation spec
4. **Generate** — produce Python code implementing a new pipeline stage
5. **Validate** — dynamically load the stage via `importlib`, inject it into the pipeline, run the inner loop, measure improvement

In our first successful run, DeepSeek drew from **Behavioral Psychology / Curriculum Learning** and autonomously generated a `SubskillFeedbackLoopStage` — a stage that decomposes "argumentative rigor" into sub-skills (premise clarity, transition logic, jargon usage, conclusion support), scores each, and provides targeted revision directives for weak areas.

The code was generated on the first attempt (0 retries), dynamically loaded, and injected after the edit planning stage. Result:

```
Baseline:     [6, 6, 7, 7, 6]  peak = 6
With stage:   [6, 6, 6, 7, 6]  peak = 7  (+1)
```

Modest improvement, but the point is: **the outer loop autonomously wrote working Python code that modified the inner pipeline's behavior** — no human specified what mechanism to try or which domain to draw from.

---

## The recursive structure

This creates three levels of the same pattern:

| Level | Who researches | What gets optimized |
|-------|---------------|-------------------|
| Level 0 | Human researchers | → Invented the autoresearch framework |
| Level 1 | Inner loop (LLM) | → Improves task output quality |
| Level 2 | Outer loop (LLM) | → Discovers new mechanisms for Level 1 |

Each level uses the same core loop: propose × evaluate → iterate. The boundary isn't the search space — it's whether the research question is **measurable**. As long as we can score the inner loop's output, the outer loop can research how to improve it.

---

## Honest limitations

- The +1 score improvement from the generated stage is modest. MiniMax M2.7 (the inner loop model) has a capability ceiling around 6-7 on reasoning dimensions — mechanism innovation can't overcome raw model limitations.
- The generated code sometimes misunderstands the interface contract (e.g., treating string outputs as dicts). We fixed this by improving the code generation prompt.
- We've only run 2 research sessions so far. More iterations needed to see if the outer loop can discover mechanisms that compound.

---

## Connection to autoresearch

This project was directly inspired by autoresearch. The core observation: autoresearch, AutoResearchClaw, and EvoScientist each represent a **human-designed mechanism change** to the base loop. We asked whether an outer optimization loop could discover such improvements autonomously.

The theoretical framing maps to bilevel optimization:
```
Upper level (outer):  min  F(P) = -E[Q | P]   over pipeline config P
Lower level (inner):  min  f(Q | P, X)         over output quality Q
```

The inner problem is solved approximately by LLM, making this an instance of *approximate bilevel optimization with LLM solvers*.

---

MIT licensed. Feedback and ideas welcome.
