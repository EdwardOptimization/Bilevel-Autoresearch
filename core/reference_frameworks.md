# Reference Frameworks for the Outer Optimization Loop

This document is injected into the outer loop's context as background knowledge.
Without it, the outer loop only knows to adjust token budgets and prompt phrasing.
With it, the outer loop can try proven optimization strategies from the literature.

---

## Why This Document Exists

The outer loop's job is to improve the inner loop's pipeline configuration.
To do that intelligently, it needs a "menu" of optimization strategies to try.
Each framework below represents a different lever for improving a pipeline that
calls LLMs in sequence and accumulates experience across runs.

---

## Framework Reference

### 1. Reflexion (Shinn et al., 2023)
**Core idea**: After a failure, generate verbal feedback about *why* it failed,
store that feedback in memory, and inject it into the next attempt.

**What the outer loop can try**:
- When a stage scores <7, generate a structured failure postmortem
- Store postmortems as a separate memory class (distinct from lessons)
- Inject the most recent postmortem at the top of the stage prompt

**When to use**: Stage quality is inconsistent across runs on the same topic
(scores like 6, 8, 6, 9 — high variance). Reflexion reduces variance by
making each failure generate targeted corrective signal.

**Reference**: Shinn et al., "Reflexion: Language Agents with Verbal
Reinforcement Learning", NeurIPS 2023.

---

### 2. Self-Refine (Madaan et al., 2023)
**Core idea**: Same LLM generates output, then critiques its own output, then
refines it — in a loop within a single stage call.

**What the outer loop can try**:
- Add an internal critique-refine loop inside Stage C or Stage E
- The critique prompt asks: "what is missing, inconsistent, or unclear?"
- The refine prompt says: "fix exactly the issues the critique identified"
- This is distinct from quality gates (which retry the whole stage); self-refine
  stays within one call and is cheaper

**When to use**: Stages that produce structurally correct but shallow output
(e.g., Stage B hypotheses that are syntactically complete but lack specificity).

**Reference**: Madaan et al., "Self-Refine: Iterative Refinement with
Self-Feedback", NeurIPS 2023.

---

### 3. OPRO — Optimization by PROmpting (Yang et al., 2023)
**Core idea**: Treat the prompt itself as the optimization target. At each
iteration, show the optimizer the (prompt, score) history and ask it to
generate a better prompt.

**What the outer loop can try**:
- Collect (stage_prompt, stage_score) pairs across inner runs
- At each outer iteration, feed this history to a meta-LLM and ask:
  "Given these prompts and their scores, write a better Stage X prompt"
- This is the most direct form of outer-loop prompt optimization

**When to use**: A stage has plateaued at 7/10 for 5+ inner runs. Token
budget changes have no effect. The prompt itself needs structural revision.

**Reference**: Yang et al., "Large Language Models as Optimizers", ICLR 2024.

---

### 4. DSPy (Khattab et al., 2023)
**Core idea**: Separate program structure (which LLM calls happen in what order)
from prompt content. Prompts are compiled automatically from labeled examples
using a teleprompter.

**What the outer loop can try**:
- Treat high-scoring (input, output) pairs from inner runs as labeled training examples
- Use these to bootstrap few-shot demonstrations injected into stage prompts
- When 3+ high-quality outputs exist for a stage, inject the best 2 as
  demonstrations ("here is an example of a good Stage B output")

**When to use**: The pipeline has accumulated ≥10 high-confidence lessons and
≥3 high-quality stage outputs but quality is still inconsistent. Few-shot
demonstrations often stabilize performance better than additional instructions.

**Reference**: Khattab et al., "DSPy: Compiling Declarative Language Model
Calls into Self-Improving Pipelines", ICLR 2024.

---

### 5. AI Scientist / EvoScientist (Lu et al., 2024; Yamada et al., 2024)
**Core idea**: Fully automated research pipeline — idea generation, literature
review, experiment, writeup, review — with the review step providing feedback
for the next cycle.

**What the outer loop can try**:
- The AI Scientist uses a "review" stage where the same model reads the full
  paper and scores it against NeurIPS criteria. Adopt this pattern: after
  Stage E, run a separate "paper review" call that mimics a peer reviewer.
- EvoScientist adds evolutionary operators: crossover (combine good elements
  from two runs) and mutation (random perturbation of prompts). The outer loop
  can maintain a population of prompt variants and evolve them.

**When to use**:
- Peer-reviewer stage: when Stage E writeup scores are inconsistent (7, 9, 6, 8)
- Evolution operators: when the outer loop has run 3+ iterations without
  convergence improvement and needs to escape local optima

**Reference**: Lu et al., "The AI Scientist: Towards Fully Automated Open-Ended
Scientific Discovery", arXiv 2024. Yamada et al., "EvoScientist", arXiv 2024.

---

### 6. TextGrad (Yuksekgonul et al., 2024)
**Core idea**: Automatic differentiation for text. Instead of gradient descent
on numerical parameters, TextGrad propagates "textual gradients" — natural
language descriptions of how to change each component to reduce a loss.

**What the outer loop can try**:
- After an inner run, for each stage that scored below target, generate a
  "textual gradient": "Stage C scored 6/10 because... The prompt should be
  changed to..."
- Propagate this backward: if Stage E scored poorly because Stage D gave bad
  results, and Stage D scored poorly because Stage C truncated, the gradient
  propagates to Stage C's prompt
- This enables multi-stage blame attribution, not just single-stage fixes

**When to use**: Quality issues are cross-stage (one stage's bad output causes
downstream failures). Standard lesson extraction only captures the failing
stage; textual gradients capture the root cause stage.

**Reference**: Yuksekgonul et al., "TextGrad: Automatic Differentiation via
Text", arXiv 2024.

---

### 7. Voyager / Skill Library Pattern (Wang et al., 2023)
**Core idea**: As an agent solves problems, it accumulates a library of
reusable "skills" (code functions or prompts). When facing a new problem,
it retrieves relevant skills and composes them.

**What the outer loop can try**:
- This is the existing skill promotion mechanism (lessons → skills). The outer
  loop should ensure skill retrieval is stage-specific AND cross-stage.
- Add a "skill composition" step: when two skills address adjacent weaknesses,
  generate a combined skill that handles both together.
- Prune skills that haven't been accessed in 5+ runs (stale skills add noise).

**When to use**: Skills file has grown >5KB and stage quality is not improving.
Large skill files can dilute attention; pruning and composition help.

**Reference**: Wang et al., "Voyager: An Open-Ended Embodied Agent with Large
Language Models", NeurIPS 2023.

---

## Outer Loop Decision Tree

```
Stage score plateau (same score ±1 for 5+ runs)?
    ├─ Variance high (6,8,6,9)?  → Try Reflexion (failure postmortems)
    ├─ Variance low (7,7,7,7)?   → Try OPRO (meta-prompt optimization)
    └─ Few-shot examples exist?  → Try DSPy (inject demonstrations)

Cross-stage failures (downstream blames upstream)?
    └─ Try TextGrad (multi-stage textual gradient)

Inner loop convergence too slow (>15 runs to reach 8/10)?
    └─ Try Self-Refine within stage (reduce within-stage variance)

Outer loop stuck after 3 iterations?
    └─ Try EvoScientist evolution operators (crossover/mutation of prompts)

Skills file large and quality not improving?
    └─ Try Voyager pruning + composition
```

---

## How to Use This Document in the Outer Loop

At the start of each outer iteration:
1. Read the inner run trace (stage scores, lesson types, failure patterns)
2. Apply the decision tree above to select 1–2 strategies to try
3. Document the strategy choice and rationale in the outer-loop lesson
4. After the inner cycle, record whether the strategy improved convergence speed

The outer loop's accumulated lessons should reference these framework names
(e.g., `strategy: "reflexion_postmortem"`) so patterns can be tracked across
outer iterations.
