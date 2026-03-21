# Example: Evolution Trace

This file shows a real evolution trace from EvoResearch, demonstrating how
the pipeline improves with each run.

**Topic**: "The role of iterative feedback in improving LLM-based research pipelines"
**Provider**: MiniMax-M2.7-highspeed (reasoning model)

---

## Run Progression

| Run | A: Lit Scan | B: Hypotheses | C: Experiment | D: Results | E: Writeup | Overall | Lessons |
|-----|-------------|---------------|---------------|------------|------------|---------|---------|
| 1 | 7 | 7 | 6 | 5 | 5 | **6/10 weak** | 7 |
| 2 | 7 | 7 | 5 | 8 | 5 | **6/10 weak** | 8 |
| 3 | 7 | 7 | 7 | 7 | 7 | **7/10 weak** | 7 |
| 4 | 7 | 7 | 7 | 7 | 7 | **7/10 pass** | 7 |
| 5 | 7 | 8 | 7 | 8 | 7 | **7/10 weak** | 8 |
| 6 | 7 | 8 | 7 | 8 | 7 | **7/10 pass** | 7 |
| 7 | 7 | 8 | 7 | 7 | 7 | **7/10 pass** | 7 |
| 8 | 8 | 5 | 7 | 7 | 8 | **6/10 weak** | 10 |
| 9 | 8 | 8 | 6 | 8 | 9\* | **7/10 weak** | 7 |
| 10 | 6\* | 9 | 5\* | 9 | 9\* | **7/10 weak** | 7 |
| 11 | 6 | 8 | 6\* | 8 | 8 | **7/10 pass** | 6 |
| 12 | 7 | 9 | 6\* | 8 | 8 | **7/10 pass** | 7 |
| **13** | **8** | **9** | **8\*** | **8** | **9** | **8/10 pass** 🎯 | 7 |
| 14 | 8 | 9 | 6\* | 9 | 8 | 7/10 pass | 7 |
| **15** | **9** | **8** | **8** | **8** | **8\*** | **8/10 pass** 🎯 (new topic) | 6 |
| **16** | **9** | **9** | **8** | **8** | **8** | **9/10 pass** 🎯 | 7 |
| **17** | **8** | **10!** | **6** | **9** | **8** | **8/10 pass** 🎯 (3rd topic) | 7 |

\* = stage retried via quality gate (score < 6 triggered automatic retry with evaluator feedback)

**Topics verified ≥8/10:**
- Run 13: "Iterative feedback in LLM-based research pipelines" → 8/10
- Run 15: "Chain-of-thought reasoning vs. direct prompting" → 8/10 (new domain, zero retraining)
- Run 16: "Iterative feedback in LLM-based research pipelines" → 9/10 (115 lessons accumulated)
- Run 17: "Model scale and in-context learning: emergent abilities" → 8/10 (B scored 10/10!)

---

## Key Inflection Points

### Run 2 → Run 3 (+1 overall)
**What changed**: Skill Promoter first activated; 15 high-confidence lessons promoted to 5 skill templates.

**Why it improved**: Skills provided structural guidance (synthesis tables, hypothesis format) that the LLM
hadn't been following. Stage C went from 5/10 to 7/10.

### Run 4 → Run 5 (Stage B +1, Stage D +1)
**What changed**: 29 lessons accumulated. hypothesis_generation skill required explicit statistical criteria.

**Lesson injected**: "Hypotheses without measurable evaluation criteria propagate downstream as untestable"

**Why it improved**: Hypotheses now include named statistical tests, power analyses, Cohen's κ thresholds.
The downstream result summary (D) improved because it could map results to well-defined criteria.

### Run 8 → Run 9 (+Stage B: 5→8, +Stage D: 7→8, −Stage E: 7→5→9)
**What changed**: Fixed hypothesis generation max_tokens (4096→5500), constrained to exactly 4 hypotheses.
Root cause identified: MiniMax-M2.7-highspeed reasoning overhead ~2000-3000 tokens per call.

**Why Stage B improved**: H5 was truncated because 5 hypotheses with full operationalization exceeded
the default 4096 token budget minus reasoning overhead. With 5500 tokens and exactly 4 hypotheses,
all hypotheses complete.

### Run 10 (rubric tightening + quality gate calibration)
**What changed**: Stage rubrics rewritten to explicitly require completeness and synthesis tables.
Quality gate min_score raised 5→6. Score integer coercion added (LLM returned "7" as string).

**Effect**: More retries triggered (catching real quality issues) but also A/C regressions due to
stricter scoring. Net: overall still 7/10 but with higher peak stage scores (B:9, D:9, E:9).

### Run 11 → Run 12 (Stage A: 6→7)
**What changed**: Stage C split into two-phase generation (Plan: separate LLM call, Code: separate LLM call).
Synthesis table column alignment fixed (prompt said "Pipeline Role" but rubric expected "Gap/Limitation").

**Why it improved**: Stage C no longer truncates code due to plan text consuming the token budget.
Stage A improved because synthesis table column match caused evaluator to give higher coverage score.

### Run 12 → Run 13 (**7/10 → 8/10** 🎯)
**What changed**:
- Stage A max_tokens: 6500 → 8000 (last knowledge gap section completing without truncation)
- Stage C Phase 1 (plan): max_tokens 4500 → 7000 (H4 metrics table completing without truncation)
- Stage C Phase 2 (code): max_tokens 5500 → 7000 (all 4 hypothesis functions + orchestration complete)
- Plan context passed to code prompt expanded from 1500 → 2500 chars (better alignment)
- Citation format standardized in Stage A prompt

**Why the breakthrough**: With enough token budget for reasoning overhead (~2000-3000 per call for
MiniMax), all stages complete without truncation. Stage C reached 8/10 (from retry). Stage E reached 9/10.
RunReviewer scored overall 8/10 pass for first time in 13 runs.

---

## Sample Lesson Extracted (Run 3)

```json
{
  "id": "lesson_40df6bb1",
  "lesson_type": "failure_pattern",
  "stage": "hypothesis_generation",
  "confidence": 0.95,
  "summary": "H5 was generated with missing evaluation criteria, which propagated as untestable through all downstream stages.",
  "reuse_rule": "Implement mandatory hypothesis completeness validator: each hypothesis must include (1) clear statement, (2) operationalized variables, (3) explicit evaluation criteria with statistical tests, (4) expected outcome.",
  "anti_pattern": "Generating hypotheses in vague conceptual terms without operationalized variables or named statistical tests."
}
```

## Sample Promoted Skill (after Run 13)

Distilled from 15 high-confidence lessons about hypothesis generation:

> **MANDATORY completeness validator**: Every hypothesis must include all four elements before stage completion:
> 1. clear hypothesis statement
> 2. operationalized variables with specific measurement methods
> 3. explicit evaluation criteria naming statistical tests with expected effect sizes
> 4. expected outcome statement with quantified predictions
>
> **Anti-pattern**: Surface-level metric selection (BLEU-4, BERTScore) to assess research quality — they measure
> surface similarity, not scientific validity. For simulated results: use "our simulation suggests..." not
> "our results demonstrate..." — and never report F-statistics or p-values for simulated data.

---

## Memory Stats (end of Run 17)

```
Total lessons: 122
Average confidence: 90%
High-confidence (≥85%): 114 eligible for promotion
Promoted skills: 6 stages (literature_scan, hypothesis_generation, experiment_plan_or_code,
                            experiment_result_summary, draft_writeup, global)
Skill sizes: 2.7KB–5.9KB per stage (growing as lessons accumulate)
Test suite: 59 passing (unit tests for LLM client, memory store, retrieval, evaluator, pipeline base)
```
