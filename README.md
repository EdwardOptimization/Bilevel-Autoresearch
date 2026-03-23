# Bilevel Autoresearch

**Autoresearch that researches itself.** An outer loop autonomously discovers new mechanisms for the inner loop — not by tuning prompts, but by inventing and code-generating structural changes to the search process.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CI](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen.svg)](tests/)

---

## Core Idea

```
Inner loop:  Optimizes the task output         (propose → execute → evaluate → keep/discard)
Outer loop:  Optimizes how the inner loop works (analyze trace → modify mechanisms → re-run)
```

Both levels use the same pattern: **propose × evaluate × iterate**. The inner loop improves the task. The outer loop improves the inner loop — not by tuning prompts, but by structurally changing how it searches.

## Key Result: Controlled Ablation Experiment

On Karpathy's GPT pretraining benchmark (val_bpb, 300s budget, RTX 5090), we ran a controlled ablation with **3 groups × 3 independent repeats × 30 iterations**, using the **same LLM (DeepSeek)** for all levels:

| Group | What it does | Mean Improvement |
|-------|-------------|-----------------|
| **A** — Level 1 | Standard autoresearch (propose → train → keep/discard) | -0.009 ± 0.001 |
| **B** — Level 1 + 1.5 | + Outer loop adjusts search config | -0.007 ± 0.006 |
| **C** — Level 1 + 1.5 + 2 | + Outer loop generates new mechanisms as code | **-0.045 ± 0.030** |

**Level 2 improves 5× over Level 1.** The outer loop autonomously generated Python code for new search mechanisms, dynamically loaded them via `importlib`, and injected them into the running inner loop.

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

Full ablation report: [`experiments/paper_ablation/run2_results/REPORT.md`](experiments/paper_ablation/run2_results/REPORT.md) |
Paper: [arxiv.260323.000006](https://aixiv.science/papers/aixiv.260323.000006)

---

## Quick Start

```bash
pip install -e .

# Article optimization demo (needs MINIMAX_API_KEY + DEEPSEEK_API_KEY)
python cli.py once --article article1                    # smoke test
python cli.py run --articles article1 --max-inner 5 --max-outer 4   # full bilevel

# Training optimization demo (needs GPU + DEEPSEEK_API_KEY + Karpathy's autoresearch)
python -m domains.train_opt.cli --provider deepseek bilevel --inner-budget 5 --outer-cycles 2

# Use any LLM provider
python cli.py --provider openai --outer-provider openai run --max-inner 5
```

See [`.env.example`](.env.example) for required API keys.

---

## Architecture

```
cli.py                            # Entry point (article domain)
core/                             # Bilevel framework + article optimization demo
├── runner.py                     # InnerRunner + inject_stage()
├── inner_loop.py                 # InnerLoopController
├── outer_loop.py                 # OuterAnalyzer + OuterLoopController
├── mechanism_research.py         # Level 2: generate new pipeline stages as code
├── state.py                      # State management with isolation boundaries
├── llm_client.py                 # Multi-provider LLM client
├── pipeline/                     # Article demo stages (A→E) — bundled, not the framework
└── evaluator/                    # Article demo evaluator — bundled, not the framework
domains/
└── train_opt/                    # Training demo: GPT pretraining optimization
    ├── runner.py                 # Inner loop with 12 agent-invented mechanisms
    ├── outer.py                  # Outer loop: trace analysis → config updates
    └── config.py                 # SearchConfig (outer loop's control surface)
articles/                         # Article demo input data
experiments/                      # Timestamped experiment records
tests/                            # 44 unit tests
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
ruff check core/ tests/             # lint
```

**Add a new domain:** create `domains/your_domain/` with `runner.py`, `outer.py`, `config.py`, `cli.py`. See [`domains/README.md`](domains/README.md) and `train_opt/` as a template. Domains are self-contained — import `core.llm_client` for LLM access, but implement your own runner and outer loop.

**Add a new LLM provider:** add an entry to `PROVIDERS` in `core/llm_client.py`. All OpenAI-compatible providers work with `native_sdk: False`.

**Key constraint:** the evaluator must NEVER receive lesson memory. Lessons influence proposals, never judgments.

## License

MIT — see [LICENSE](LICENSE)
