# Article Optimizer — Direct Path

This subdirectory implements the **direct path** of the dual-layer autoresearch
experiment: the pipeline stages are repurposed from scientific research to
directly optimize article content.

## What This Is

The root-level `src/` (V1) demonstrates inner-loop autoresearch on *scientific topics*
(e.g., "iterative feedback in LLM pipelines"). Articles are written as a byproduct.

This directory (V2) applies the same inner-outer loop structure *directly* to the
three articles in `articles/`:
- `article1_llm_research_depth.md` — single-direction autoresearch depth
- `article15_meta_optimization.md` — dual-layer autoresearch (this system itself)
- `article2_agent_team_scale.md` — multi-agent scale and project management

## Layer Structure

```
Outer Loop (5 iterations max)
    optimizes: pipeline prompts, rubrics, injection strategy, lesson extraction
    metric: how fast does inner loop converge to ≥8/10?
    reset: article content + inner memory at each outer iteration

Inner Loop (20 iterations max per outer cycle)
    optimizes: article content (analysis → hypotheses → edits → revised output)
    metric: 5-dimension rubric score (A-E), target ≥8/10
    quality gate: <6 triggers auto-retry
```

## Stage Mapping

| Stage | Role |
|-------|------|
| A: Article Analysis | Identify all weaknesses against the 5-dimension rubric |
| B: Improvement Hypotheses | Generate H1–H4 specific improvement plans |
| C: Edit Planning | Concrete edit plan: which section, what change |
| D: Impact Assessment | Predict post-edit rubric scores |
| E: Revised Output | Write the actual revised article sections |

## Quality Definitions

See `../docs/quality_definitions.md` for complete rubric and convergence criteria.

## Reference Frameworks

See `../docs/reference_frameworks.md` for the outer loop's strategy menu
(Reflexion, OPRO, DSPy, TextGrad, EvoScientist, etc.).

## Status

- [ ] Stage implementations (adapt from `../src/pipeline/`)
- [ ] Outer loop controller
- [ ] Inner loop with article reset logic
- [ ] Evaluator with 5-dimension rubric
- [ ] Lesson extractor (article-domain)
- [ ] First inner run (baseline article scores)
