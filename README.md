# Bilevel Autoresearch

**Autoresearch that researches itself.** An outer loop autonomously discovers new mechanisms for the inner loop — not by tuning prompts, but by inventing and code-generating structural changes to the search process.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CI](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/EdwardOptimization/Bilevel-Autoresearch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen.svg)](tests/)

---

## Core Idea

```
Level 1:    Inner loop optimizes the task      (propose → execute → evaluate → keep/discard)
Level 1.5:  Outer loop adjusts inner config    (freeze params, shift strategy, inject guidance)
Level 2:    Agent modifies inner loop CODE      (invent new mechanisms, change loop structure)
```

Each level uses the same pattern: **propose × evaluate × iterate**. The difference is what gets optimized — task output, search configuration, or search mechanism itself.

## Key Result

A Level 2 agent ran autonomously for 7 rounds on Karpathy's GPT pretraining benchmark. It invented **12 mechanisms** without human specification:

| Mechanism | Inspired By | What It Changed |
|-----------|------------|-----------------|
| ElitePool + Crossover | Evolutionary algorithms | Proposal generation: LLM + population-based interpolation |
| Simulated Annealing | Statistical mechanics | Keep/discard: probabilistic acceptance of regressions |
| Plateau Detector | Signal processing | Loop behavior: force diversification when stuck |
| Momentum Tracker | Gradient optimization | Proposal context: directional signals from history |
| Crash Memory | Software engineering | Proposal context: warn about crash-causing params |
| Freeze Limit | Control theory | Outer loop: prevent over-constraining the search space |

```
val_bpb: 1.393 → 1.219    Search efficiency: 36% → 91%    Crash rate: 27% → 0%
```

Full report: [`experiments/train_opt_20260322/REPORT.md`](experiments/train_opt_20260322/REPORT.md)

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
core/                             # Bilevel framework
├── runner.py                     # InnerRunner + inject_stage()
├── inner_loop.py                 # InnerLoopController
├── outer_loop.py                 # OuterAnalyzer + OuterLoopController
├── mechanism_research.py         # Level 2: generate new pipeline stages as code
├── state.py                      # State management with isolation boundaries
├── llm_client.py                 # Multi-provider LLM client
├── pipeline/                     # Article demo: 5 stages (A→E)
└── evaluator/                    # Rubric scoring (isolated from lessons)
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

**Training optimization** — LLM proposes hyperparameter changes to Karpathy's `train.py`, trains for 2 minutes, measures `val_bpb`. Outer loop freezes ineffective params and shifts strategy. Level 2 agent invented CrashMemory, MultiCandidate, ElitePool+Crossover, SimulatedAnnealing, PlateauDetector, and more.

---

## Related Work

| Project | Contribution | Link |
|---------|-------------|------|
| **AutoResearch** (Karpathy) | The single-track autoresearch loop | [GitHub](https://github.com/karpathy/autoresearch) |
| **AutoResearchClaw** (AIMing Lab) | Multi-batch parallel search | [GitHub](https://github.com/aiming-lab/AutoResearchClaw) |
| **EvoScientist** | Persistent experience memory | [GitHub](https://github.com/EvoScientist/EvoScientist) |

Each of the above is a **human-designed** mechanism change. This project asks: can an outer loop discover such improvements **autonomously**?

---

## Extending

Add a new domain in `domains/your_domain/` — see [`domains/README.md`](domains/README.md) and `train_opt/` as a template. You need: a runner (inner loop), an outer loop, a config, and a measurable metric.

## License

MIT — see [LICENSE](LICENSE)
