# Bilevel Autoresearch

**A dual-layer self-improving article optimizer. The inner loop improves articles. The outer loop improves how the inner loop does it.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CI](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Providers](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20MiniMax%20%7C%20OpenAI%20%7C%20Anthropic-purple.svg)](#providers)
[![Tests](https://img.shields.io/badge/tests-59%20passing-brightgreen.svg)](tests/)

---

## What is this?

Most AI pipelines run once and stop. EvoResearch runs repeatedly, learning from each run. **Bilevel Autoresearch** takes this one step further: when the pipeline itself becomes the subject of optimization, a two-level structure emerges.

```
Inner loop: Article → 5-Stage Pipeline → Evaluator → Lessons → Better Article
                                  ↑ improved prompts / config
Outer loop: Trace Analysis → DeepSeek Meta-Optimizer → Pipeline Config Updates
```

- **Inner loop** (MiniMax): edits one article, accumulates lessons, improves quality
- **Outer loop** (DeepSeek): analyzes the inner loop's trace, updates pipeline config, improves *how* the inner loop runs

This maps directly to **Bilevel Optimization**: upper level minimizes pipeline config loss, lower level (approximately) minimizes article quality loss under that config.

---

## Evolution Trace

**Single-layer experiment** — inner loop only, manual outer optimization (17 loops):

```
Run 1:  A:7 B:7 C:6 D:5 E:5 → 6/10  (0 lessons)
Run 4:  A:7 B:7 C:7 D:7 E:7 → 7/10  (skills v2 promoted)
Run 9:  A:8 B:8 C:6 D:8 E:9 → 7/10  (token budget fix for MiniMax reasoning overhead)
Run 13: A:8 B:9 C:8 D:8 E:9 → 8/10 🎯 (122 lessons, 6 skills)
Run 15: A:9 B:8 C:8 D:8 E:8 → 8/10 🎯 (2nd topic: CoT reasoning)
Run 16: A:9 B:9 C:8 D:8 E:8 → 9/10 🎯 (3rd topic: in-context learning)
```

*A–E are rubric dimensions: Argumentative Rigor / Conceptual Clarity / Cross-Article Consistency / Insight Novelty / Actionability*

**Dual-layer experiment** — outer loop automated (4 cycles × 5 runs):

```
Cycle 1: 7.2, 6.6, 6.6, 6.4, 6.4  (baseline, no outer intervention yet)
Cycle 2: 6.6, 7.0, 7.0, 7.0, 7.0  ← outer Reflexion injection → C dimension locked at 8/10
Cycle 3: 6.4, 6.2, 6.2, 7.0, 6.2  (outer searching for better strategy)
Cycle 4: 6.4, 6.4, 7.0, 6.8, 6.6
```

Cycle 2 stability (4/5 runs at 7.0 vs. declining in Cycle 1) is direct evidence of the outer loop working.

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Set API keys
export MINIMAX_API_KEY="sk-..."     # inner loop (article editing)
export DEEPSEEK_API_KEY="sk-..."    # outer loop (meta-optimization)

# 3. Smoke test — one pass on article1
python article_optimizer/cli.py once --article article1

# 4. Inner loop only — 5 runs, no outer optimization
python article_optimizer/cli.py inner --article article1 --max-inner 5

# 5. Full dual-layer experiment
python article_optimizer/cli.py run --articles article1 --max-inner 5 --max-outer 4

# 6. Run all three articles
python article_optimizer/cli.py run --max-inner 5 --max-outer 4
```

---

## Pipeline Stages

Each inner loop run passes the article through 5 stages:

| Stage | Role | Lesson Injection |
|-------|------|-----------------|
| **A: Article Analysis** | Identify weaknesses per rubric dimension | Skills only (bias-free) |
| **B: Improvement Hypotheses** | Generate H1–H4 concrete fixes | ✅ lessons + skills |
| **C: Edit Planning** | Triage hypotheses → ranked edit plan | ✅ lessons + skills |
| **D: Impact Assessment** | Predict post-edit rubric scores, flag regressions | Skills only |
| **E: Revised Output** | Apply edits, produce new article | ✅ lessons + skills |

After each run:
- **Evaluator** scores the revised article on 5 rubric dimensions (A–E) — *never reads memory*
- **Lesson Extractor** pulls 2–4 structured lessons from the run
- Lessons with confidence ≥ 0.85 are promoted to **skills** (verified rules, shown as Tier 1 in next run)

---

## Rubric Dimensions

| Dim | Name | Target |
|-----|------|--------|
| A | Argumentative Rigor — every claim has a support chain | ≥8 |
| B | Conceptual Clarity — key terms defined, no ambiguity | ≥8 |
| C | Cross-Article Consistency — no contradictions with companion articles | ≥8 |
| D | Insight Novelty — non-obvious, explicitly stated | ≥7 |
| E | Actionability — clear implication for practice or future work | ≥7 |

---

## Outer Loop

The outer loop (DeepSeek) analyzes the inner cycle trace after each outer iteration and produces:

```json
{
  "root_cause": "...",
  "strategy": "reflexion | opro | promptbreeder",
  "prompt_overrides": {
    "article_analysis": "Pay special attention to cross-article consistency..."
  },
  "outer_lessons": [...]
}
```

Prompt overrides are injected into the inner pipeline's stages at the start of the next cycle. The inner loop never knows about the outer loop — it just sees better prompts.

---

## Architecture

```
article_optimizer/
├── cli.py                        # Entry point: once / inner / run commands
├── articles/                     # Baseline article files (ground truth)
│   ├── article1_llm_research_depth.md
│   ├── article15_meta_optimization.md
│   └── article2_agent_team_scale.md
└── src/
    ├── runner.py                 # InnerRunner: one full A→B→C→D→E pass
    ├── inner_loop.py             # InnerLoopController: runs until convergence
    ├── outer_loop.py             # OuterAnalyzer + OuterLoopController
    ├── state.py                  # InnerLoopState / OuterLoopState / RunResult
    ├── pipeline/
    │   ├── base.py               # BaseStage
    │   ├── article_analysis.py   # Stage A
    │   ├── improvement_hypotheses.py  # Stage B
    │   ├── edit_planning.py      # Stage C
    │   ├── impact_assessment.py  # Stage D
    │   └── revised_output.py     # Stage E
    └── evaluator/
        └── article_evaluator.py  # Rubric scoring — never reads memory

src/
└── llm_client.py                 # Multi-provider client (LLMClient + module-level API)
                                  # LLMClient is instance-scoped — prevents outer loop
                                  # from clobbering inner loop's provider config
```

---

## Artifact Structure

```
article_optimizer/artifacts/
└── cycle_01/
    ├── run_001/
    │   ├── evaluation.json          ← rubric scores A-E + overall
    │   └── stages/
    │       ├── article_analysis/analysis.md
    │       ├── improvement_hypotheses/hypotheses.md
    │       ├── edit_planning/edit_plan.md
    │       ├── impact_assessment/impact_assessment.md
    │       └── revised_output/revised_article.md
    ├── run_002/ ...
    └── run_005/
└── cycle_02/ ...
```

---

## Providers

| Provider | Env Var | Role in System |
|----------|---------|----------------|
| `minimax` | `MINIMAX_API_KEY` | Inner loop (article editing) — reasoning model, allocate ~3000 extra tokens for think blocks |
| `deepseek` | `DEEPSEEK_API_KEY` | Outer loop (meta-optimization) — fast, reliable JSON output |
| `openai` | `OPENAI_API_KEY` | Either loop |
| `anthropic` | `ANTHROPIC_API_KEY` | Either loop (requires `pip install anthropic`) |
| `glm` | `GLM_API_KEY` | Either loop |

---

## The Theory

This system is the implementation described in the article series in `article_optimizer/articles/`:

- **Article 1** (`article1_llm_research_depth.md`): Single-loop autoresearch — proposal × feedback × iteration
- **Article 1.5** (`article15_meta_optimization.md`): **Bilevel Autoresearch** — when the pipeline itself becomes the research target
- **Article 2** (`article2_agent_team_scale.md`): Multi-pipeline scaling for large research tasks

The key formalization from Article 1.5:

```
Upper level (outer loop):   min  F(P) = -E[Q | P]      over pipeline config P
Lower level (inner loop):   min  f(Q | P, X)            over article quality Q
```

The inner problem is solved **approximately** by LLM — not to global optimality. This makes it an instance of *approximate bilevel optimization with LLM solvers*, distinct from classical bilevel theory which assumes exact inner solutions.

The config space P is mixed-integer: discrete (strategy selection, two-phase generation on/off) + continuous (token budgets, truncation thresholds). The objective is highly nonlinear — a token budget change from 4096 to 5500 can cause a non-smooth quality jump by unlocking full reasoning output.

---

## Key Design Decisions

**Evaluator isolation**: The evaluator never reads lesson memory. Lessons only influence proposals, never judgments. Without this, outer loop feedback degrades as the evaluator starts echoing what it "knows" should be good.

**Two-tier lesson injection**: Raw lessons (any confidence) always injected as Tier 2. Promoted skills (≥0.85 confidence) shown as Tier 1 "Verified Rules". This prevents the empty-injection failure mode when the LLM returns low confidence scores.

**Instance-scoped LLMClient for outer loop**: The outer loop uses `LLMClient("deepseek", ...)` instead of the module-level `configure()`. This prevents the DeepSeek config from clobbering the MiniMax config that inner loop stages depend on — a subtle global-state race condition in a concurrent two-model system.

---

## Roadmap

- [ ] Outer loop strategy diversity (beyond Reflexion — try PromptBreeder, OPRO on separate cycles)
- [ ] Embedding-based lesson retrieval (vs. keyword)
- [ ] Multi-article parallel inner loops with shared outer signal
- [ ] Real literature search integration (Semantic Scholar / arXiv)
- [ ] Code execution sandbox for real experiments

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built on the principle that a system which optimizes its own optimization process is strictly more powerful than one that doesn't.*
