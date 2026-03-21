# Quality Definitions for Dual-Layer Autoresearch

This document defines "good" for both the inner and outer loops of the
dual-layer autoresearch system targeting article optimization.

---

## Inner Loop Quality — "Is the article good?"

The inner loop runs the pipeline directly against article content. Each run
produces an improved version of the article. Quality is measured by a
5-dimension rubric, each dimension scored 0–10.

### Inner Loop Rubric

| Dimension | What is measured | Threshold |
|-----------|-----------------|-----------|
| **A: Argumentative Rigor** | Every core claim has a clear derivation chain or empirical support; no logical jumps | ≥8 |
| **B: Conceptual Clarity** | Key terms are defined; reader cannot misinterpret the central concept | ≥8 |
| **C: Cross-Article Consistency** | No contradictions between Article 1, 1.5, and 2; cross-references are accurate and bidirectional | ≥8 |
| **D: Insight Novelty** | The article says something non-obvious that the reader could not derive from first principles in 5 minutes | ≥7 |
| **E: Actionability** | Reader knows what to do or how to update their thinking after reading | ≥7 |

**Overall score** = average of A–E, rounded to nearest integer.

### Inner Loop Convergence Criteria

- **Converged**: Overall ≥8/10, sustained for 3 consecutive runs without regression
- **Max budget**: 20 inner iterations per outer cycle (hard limit)
- **Quality gate**: Any stage scoring <6 triggers automatic retry with evaluator feedback
- **Evaluator isolation**: The evaluator never sees lessons or skills from previous runs

### Inner Loop Stage Mapping (Direct Path)

The pipeline stages are repurposed from scientific research to article optimization:

| Original Stage | Repurposed Role |
|---------------|-----------------|
| A: Literature Scan | Analyze current article state; identify all weaknesses against the rubric |
| B: Hypothesis Generation | Generate improvement hypotheses (H1–H4); each must name a specific weakness and proposed fix |
| C: Experiment Plan | Draft a concrete edit plan: which section, what change, why it addresses the hypothesis |
| D: Result Summary | Simulate post-edit evaluation: predict how each hypothesis fix improves the rubric score |
| E: Draft Writeup | Output the revised article sections implementing the approved plan |

---

## Outer Loop Quality — "Is the optimization process good?"

The outer loop optimizes the inner loop's pipeline configuration (prompts,
rubrics, injection strategy, lesson extraction). Each outer iteration consists
of one full inner run cycle (up to 20 inner iterations).

### Outer Loop Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Convergence Speed** | How many inner iterations to reach overall ≥8/10 | ≤10 runs |
| **Peak Quality** | Best overall score achieved within the budget | ≥9/10 |
| **Generalization** | Does the pipeline reach ≥8/10 on a previously unseen article without reconfiguration | Yes, within 15 runs |
| **Lesson Quality** | Fraction of extracted lessons with confidence ≥0.85 | ≥70% |

**Primary outer metric**: Convergence speed. A better outer configuration makes
the inner loop converge faster, independent of article content.

### Outer Loop Convergence Criteria

- **Converged**: Inner loop consistently reaches ≥8/10 in ≤10 iterations across
  2 different articles
- **Max outer budget**: 5 outer iterations (each inner cycle costs ~20 runs)
- **Outer evaluator**: Scores the *process trace*, not the article quality directly

### Reset Policy

Each outer iteration **resets**:
- Article content → restored to original version
- Inner loop lesson memory → cleared (start from zero inner lessons)

Each outer iteration **preserves**:
- Pipeline configuration (prompts, rubric design, token budgets, injection strategy)
- Outer-loop lessons and skills (accumulated cross-iteration knowledge about process quality)

**Rationale**: This isolates what the outer loop is measuring — the speed and
quality of the optimization process itself, starting from scratch each time.
Analogous to MAML: the outer loop learns a good optimizer initialization;
the inner loop demonstrates how well that optimizer works from a cold start.

---

## Relationship Between Layers

```
Outer Loop (optimizes pipeline config)
    │
    │  "what configuration makes inner converge fastest?"
    ▼
Inner Loop × N runs (optimizes article content)
    │
    │  "what edits make this article score ≥8/10?"
    ▼
Article Quality Score (A/B/C/D/E rubric)
```

The outer loop's α (error multiplier) is itself being minimized — the exact
dual-layer structure described in `autoresearch_meta_optimization.md`.

---

## What the Outer Loop Cannot Change

To keep the two layers cleanly separated, the outer loop is **not** allowed to:

- Modify the article content directly
- Change the evaluation rubric's 5 dimensions (only the sub-criteria within each)
- Skip inner iterations to save time
- Access inner loop lesson memory during its own optimization step

These constraints prevent the outer loop from "cheating" by directly editing
the article rather than improving the pipeline that edits it.
