# Bilevel Autoresearch

**Autoresearch that researches itself.** An outer loop autonomously discovers new mechanisms for the inner loop — not by tuning prompts, but by inventing and code-generating structural changes to the search process.

**Paper:** [Bilevel Autoresearch: Meta-Autoresearching Itself](https://arxiv.org/abs/2603.23420) (arXiv:2603.23420 | [AISC 2026](https://aixiv.science/abs/aixiv.260323.000006))

[![arXiv](https://img.shields.io/badge/arXiv-2603.23420-b31b1b.svg)](https://arxiv.org/abs/2603.23420)
[![AISC2026](https://img.shields.io/badge/AISC2026-aixiv.260323.000006-blue.svg)](https://aixiv.science/abs/aixiv.260323.000006)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CI](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Core Idea

```
Inner loop:  Optimizes the task output         (propose → execute → evaluate → keep/discard)
Outer loop:  Optimizes how the inner loop works (analyze trace → modify mechanisms → re-run)
```

Both levels use the same pattern: **propose × evaluate × iterate**. The inner loop improves the task. The outer loop improves the inner loop — not by tuning prompts, but by structurally changing how it searches.

**Autoresearch anything with a measurable objective.** The same bilevel principle applies to project management, multi-agent coordination, experiment scheduling, and the research process itself.

**Why 3 layers in the experiment?** The design is bilevel (inner + outer), but in practice we split the outer loop into two responsibilities for cleaner separation of concerns:

| Layer | Role | What it changes |
|-------|------|-----------------|
| **Level 1** | Inner loop | Task output (hyperparameters) |
| **Level 1.5** | Outer loop — config | Runtime search parameters (freeze/unfreeze, strategy shift) |
| **Level 2** | Outer loop — mechanism | Inner loop code structure (generate new Python mechanisms) |

Level 1.5 handles tactical adjustments ("stop searching WEIGHT_DECAY, focus on LR"). Level 2 handles strategic innovation ("invent a Tabu Search mechanism to prevent repetitive proposals"). Separating them lets Level 2 focus purely on mechanism discovery without being distracted by parameter tuning.

### The Ultimate Goal: Recursive Bootstrapping

Not only does Level 2 generate mechanisms to accelerate Level 1, but the architecture inherently supports **recursive mechanism feedback**. 

If Level 2 discovers that a mechanism (e.g., parallel multi-agent debate, persistent memory, or Tabu search) significantly improves Level 1's search efficiency, **this same mechanism can be abstracted and applied inversely to Level 2 itself.** The system learns how to learn, and then applies those lessons to its own meta-learning process. This moves the framework beyond automated optimization and towards a truly self-improving digital ecosystem.

## Key Result: Controlled Ablation Experiment

On Karpathy's GPT pretraining benchmark (val_bpb, 300s budget, RTX 5090), we ran a controlled ablation with **3 groups × 3 independent repeats × 30 iterations**, using the **same LLM (DeepSeek)** for all levels:

| Group | What it does | Mean Δval_bpb | vs Group A |
|-------|-------------|--------------|------------|
| **A** — Level 1 | Standard autoresearch (propose → train → keep/discard) | -0.009 ± 0.001 | 1× |
| **B** — Level 1 + 1.5 | + Outer loop adjusts search config | -0.007 ± 0.006 | 0.8× |
| **C** — Level 1 + 1.5 + 2 | + Outer loop generates new mechanisms as code | **-0.045 ± 0.030** | **5×** |

*Baseline val_bpb ≈ 1.10. More negative = better. 3 independent repeats × 30 iterations each.* The outer loop autonomously generated Python code for new search mechanisms, dynamically loaded them via `importlib`, and injected them into the running inner loop.

### Mechanisms discovered autonomously by Level 2

Each repeat independently discovered different mechanisms from different domains — no human specified which domains to explore:

| Mechanism | Domain | What It Does |
|-----------|--------|-------------|
| **Tabu Search Manager** | Combinatorial optimization | Prevents revisiting recently explored parameter regions |
| **Multi-Scale Bandit** | Online learning / MAB | Balances exploration and exploitation across parameters |
| **Orthogonal Exploration** | Design of experiments | Forces search across orthogonal parameter dimensions |
| **GP Regressor** (reverted) | Bayesian optimization | Surrogate model for val_bpb prediction (sklearn not installed) |

### Why Level 2 wins

Group A (no outer loop) follows a **deterministic search path** — it tries WEIGHT_DECAY, then WINDOW_PATTERN, then gets stuck repeating the same proposals for 20+ iterations. Level 2's mechanisms (Tabu Search, Orthogonal Exploration) break this loop and guide the LLM to discover that **reducing TOTAL_BATCH_SIZE** dramatically improves val_bpb — a direction A and B never explored.

```python
# Agent-generated mechanism: Tabu Search prevents the LLM from repeating failed proposals
class TabuSearchManager:
    def __init__(self, max_tabu_size=10):
        self._tabu_list = []  # recently visited parameter regions

    def is_tabu(self, changes: dict) -> bool:
        """Check if proposed changes are too similar to recent attempts."""
        for tabu_entry in self._tabu_list:
            if self._similarity(changes, tabu_entry) > 0.8:
                return True  # block this proposal, force the LLM to try something new
        return False
```

Full ablation report: [`experiments/ablations/paper_ablation/run2_results/REPORT.md`](experiments/ablations/paper_ablation/run2_results/REPORT.md)

---

## Quick Start

**Prerequisites:** Python 3.10+, an LLM API key (DeepSeek, OpenAI, or Anthropic). Training demo also needs a GPU and [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) cloned.

```bash
pip install -e .
cp .env.example .env  # fill in your API keys

# Article optimization — lightweight demo (no GPU needed)
python -m domains.article_opt.cli --provider openai once --article article1  # smoke test
python -m domains.article_opt.cli run --articles article1 --max-inner 5 --max-outer 4   # full bilevel

# Training optimization — reproduce the paper result (needs GPU)
git clone https://github.com/karpathy/autoresearch.git ~/karpathy_autoresearch
python -m domains.train_opt.cli --provider deepseek bilevel --inner-budget 5 --outer-cycles 2
```

---

## Architecture

```
core/                             # Bilevel framework (shared, domain-agnostic)
├── inner_loop.py                 # InnerLoopController
├── state.py                      # State management with isolation boundaries
└── llm_client.py                 # Multi-provider LLM client
domains/
├── article_opt/                  # Article optimization demo
│   ├── cli.py                    # Entry point: python -m domains.article_opt.cli
│   ├── runner.py                 # InnerRunner + inject_stage()
│   ├── outer.py                  # OuterAnalyzer + OuterLoopController
│   ├── mechanism_research.py     # Level 2: generate new pipeline stages as code
│   ├── pipeline/                 # Article stages (A→E)
│   ├── evaluator/                # Article rubric evaluator
│   └── reference_frameworks.md  # Optimization strategy reference doc
└── train_opt/                    # Training demo: GPT pretraining optimization
    ├── runner.py                 # Inner loop with 12 agent-invented mechanisms
    ├── outer.py                  # Outer loop: trace analysis → config updates
    └── config.py                 # SearchConfig (outer loop's control surface)
articles/                         # Article demo input data (root-level, referenced by path)
experiments/                      # Ablation results and experiment records
paper/                            # LaTeX paper (submitted to AISC2026)
tests/                            # 110 unit tests
```

---

## How It Works

The **inner loop** runs a task repeatedly, learning from each run. The **outer loop** analyzes the inner loop's trace and modifies its configuration. **Level 2** goes further — an autonomous agent reads the inner loop's code, identifies bottlenecks, and writes new Python code to fix them.

Two demo domains are included:

**Article optimization** — 5-stage pipeline (Analysis → Hypotheses → Planning → Assessment → Revision) evaluated against a rubric. Outer loop injects prompt overrides. Level 2 generates new pipeline stages via `importlib`.

**Training optimization** — LLM proposes hyperparameter changes to Karpathy's `train.py`, trains for 5 minutes, measures `val_bpb`. Outer loop freezes ineffective params and shifts strategy. Level 2 generates new Python mechanisms (Tabu Search, Multi-Scale Bandit, Orthogonal Exploration) and injects them via `importlib`. Validated with 3×3 ablation on RTX 5090 — Level 2 achieves 5× improvement over baseline autoresearch.

---

## Related Work

| Project | Contribution | Link |
|---------|-------------|------|
| **AutoResearch** (Karpathy) | The single-track autoresearch loop | [GitHub](https://github.com/karpathy/autoresearch) |
| **AutoResearchClaw** (AIMing Lab) | Multi-batch parallel search | [GitHub](https://github.com/aiming-lab/AutoResearchClaw) |
| **EvoScientist** | Persistent experience memory | [GitHub](https://github.com/EvoScientist/EvoScientist) |

Each of the above is a **human-designed** mechanism change. This project asks: can an outer loop discover such improvements **autonomously**?

---

## Contributing

```bash
git clone https://github.com/EdwardOptimization/Bilevel-Autoresearch.git
cd Bilevel-Autoresearch && pip install -e ".[dev]"
python -m pytest tests/ -v          # run tests (offline, no API key)
ruff check core/ domains/ tests/    # lint
```

**Add a new domain:** create `domains/your_domain/` with `runner.py`, `outer.py`, `config.py`, `cli.py`. See [`domains/README.md`](domains/README.md) and `train_opt/` as a template. Domains are self-contained — import `core.llm_client` for LLM access, but implement your own runner and outer loop.

**Add a new LLM provider:** add an entry to `PROVIDERS` in `core/llm_client.py`. All OpenAI-compatible providers work with `native_sdk: False`.

**Key constraint:** the evaluator must NEVER receive lesson memory. Lessons influence proposals, never judgments.

## License

MIT — see [LICENSE](LICENSE)
