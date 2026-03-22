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

| Stage | Role |
|-------|------|
| A: Article Analysis | Analyze current article state; identify all weaknesses against the rubric |
| B: Improvement Hypotheses | Generate H1–H4; each must name a specific weakness and a concrete proposed fix |
| C: Edit Planning | Draft a concrete edit plan: which section, what change, why it addresses the hypothesis |
| D: Impact Assessment | Simulate post-edit evaluation: predict how each fix improves rubric scores |
| E: Revised Output | Write the revised article sections implementing the approved plan |

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

- **Converged**: Inner loop consistently reaches ≥8/10 in ≤10 iterations across 2 different articles
- **Max outer budget**: 5 outer iterations (each inner cycle costs ~20 runs)
- **Outer evaluator**: Scores the *process trace*, not the article quality directly

---

## State Boundary: What Gets Reset and What Persists

This section defines the exact state boundary between the inner and outer loops.
Getting this boundary wrong collapses the two layers into one.

### Complete Inner Loop State Inventory

| State Item | Description | On Outer Reset |
|------------|-------------|----------------|
| Article working copy | The version currently being optimized | ✅ Restore to original |
| Inner lessons | Structured JSON extracted after each run | ✅ Clear |
| Inner skills | Markdown files distilled from inner lessons | ✅ Clear (article-specific, not process-general) |
| Stage output artifacts | A/B/C/D/E outputs from every run | ✅ Clear (archive best first — see below) |
| Run trace | `(run_n, score_A, score_B, ..., score_overall)` sequence | ✅ Clear (extract to outer first) |
| Evaluator feedback text | Textual explanation behind each score | ✅ Clear (extract patterns to outer first) |
| Quality gate retry log | Which stage retried, how many times, retry context | ✅ Clear |

### Outer Loop State (Persists Across All Outer Iterations)

| State Item | Description | Never Reset |
|------------|-------------|-------------|
| Pipeline config | Stage prompts, token budgets, rubric sub-criteria, injection strategy | ✅ Persists |
| Outer lessons | Process-level lessons: "Reflexion reduced Stage B variance in outer cycle 2" | ✅ Persists |
| Outer skills | Distilled process strategies, referenced by name from `reference_frameworks.md` | ✅ Persists |
| Strategy history | Which frameworks were tried per outer iteration and their measured effect | ✅ Persists |
| Iteration summaries | Per outer iteration: convergence speed, peak quality, generalization result | ✅ Persists |
| Calibration article | A fixed held-out article used only for rubric calibration checks | ✅ Persists, never modified |

### Archive Before Reset

Before clearing inner state, the outer loop saves:
- Best-scoring article version → `examples/outer_cycle_{N}_best_article.md`
- Full run trace → `examples/outer_cycle_{N}_trace.json`
- Evaluator feedback patterns → extracted into outer lesson (not re-injected as content)

These are **archival only** — the next inner cycle does NOT receive them as input.

---

## Information Flow: What Crosses the Layer Boundary

### Inner → Outer (Extraction, after inner cycle ends)

**Allowed (process-level signals):**
- Convergence trace: `[(run_n, overall_score), ...]`
- Stage failure pattern: which stage was most volatile, highest retry rate
- Lesson quality distribution: confidence histogram, lesson_type breakdown
- Evaluator dimension patterns: which rubric dimension was repeatedly cited as weak
- Strategy effectiveness: did the injected strategy (e.g., Reflexion) reduce variance?
- Token truncation events: which stage, how often

**Archived but not re-injected (content-level):**
- Best article version → saved to `examples/`, not fed back as starting content

### Outer → Inner (Injection, at start of next inner cycle)

**Allowed (process-level):**
- Updated stage prompts (rewritten based on outer analysis)
- Updated rubric sub-criteria (more precise scoring guidance within each dimension)
- Updated token budgets
- Strategy activation: "use Self-Refine loop inside Stage C this cycle"
- Few-shot stage examples from prior high-scoring outputs (DSPy pattern)
- Reference framework selection: which strategy to emphasize from `reference_frameworks.md`
- Pipeline structure changes: e.g., add intra-stage critique-refine sub-loop

**Forbidden (content-level):**
- Previous cycle's improved article content (bypasses inner loop's job)
- Inner lessons or inner skills from any previous cycle (article-specific knowledge)
- Specific textual edits to the article ("add this sentence to paragraph 3")
- Evaluator score history (evaluator must score fresh every run)

---

## Rubric Stability Constraint

The outer loop may refine rubric **sub-criteria** (how to score within each dimension),
but must never shift the scoring baseline.

**Check required after any rubric change:**
Score the calibration article with old and new rubric. If overall score shifts by
more than ±0.5 points, the change is rejected — it distorts the convergence metric.

---

## Outer Loop's Own Memory System

The outer loop requires its own lesson/skill system, separate from the inner loop's:

```
Inner loop memory  →  article-specific patterns  →  cleared on reset
Outer loop memory  →  process-level patterns     →  never cleared
```

**Outer lesson schema:**
```json
{
  "outer_cycle": 2,
  "lesson_type": "strategy_effectiveness",
  "strategy_used": "reflexion_postmortem",
  "inner_convergence_before": 16,
  "inner_convergence_after": 11,
  "stage_affected": "hypothesis_generation",
  "summary": "Reflexion-style failure postmortems reduced Stage B variance from ±2.1 to ±0.8",
  "reuse_rule": "Apply Reflexion postmortems when Stage B score variance across 3 consecutive runs exceeds ±1.5",
  "confidence": 0.88
}
```

---

## Relationship Between Layers

```
Outer Loop (optimizes pipeline config P_t)
    │  extracts: convergence trace, failure patterns, strategy effectiveness
    │  injects:  updated prompts, token budgets, strategy activation, few-shot examples
    ▼
Inner Loop × N runs (optimizes article content)
    │  article resets to original each outer iteration
    │  inner memory clears each outer iteration
    ▼
Article Quality Score (A/B/C/D/E rubric)
```

The outer loop's α (error multiplier from `autoresearch_meta_optimization.md`) is
itself being minimized across outer iterations. This is the dual-layer structure:
inner loop reduces article quality error; outer loop reduces inner loop's α.
