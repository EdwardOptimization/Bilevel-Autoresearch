# Training Optimization Experiment — 2026-03-22

Bilevel autoresearch on Karpathy's GPT pretraining benchmark (val_bpb metric).
RTX 5080 (16GB), 2-minute training budget per run.

## Setup

- Inner loop LLM: DeepSeek (deepseek-chat)
- Outer loop LLM: DeepSeek (deepseek-chat)
- Benchmark: Karpathy autoresearch (train.py with SDPA fallback for Blackwell GPU)
- Training budget: 120 seconds per run

## Experiment Phases

### Phase 1: Manual bilevel (no Level 2 agent)

50M parameter model (DEPTH=8, ASPECT_RATIO=64).

| Run | Baseline | Best | Improvement |
|-----|----------|------|-------------|
| First bilevel | 1.393 | 1.245 | -0.148 (-10.6%) |

3/11 crashes (DEPTH changes), outer loop correctly froze DEPTH.

### Phase 2: Level 2 agent — 50M model

| Round | Best | Keeps | New Mechanism |
|-------|------|-------|--------------|
| 1 | 1.229 (no improvement) | 1/11 | CrashMemory, MultiCandidate, QuickTest (broken) |
| 2 | 1.226 | 4/11 | QuickTest fix, MomentumTracker |
| 3 | (data lost — log overwritten) | — | ElitePool + Crossover |
| 4 | 1.220 (data lost — log overwritten) | — | — |
| 5 | 1.223 | 5/11 | ElitePool active |

### Phase 3: Level 2 agent — 13M model (agent changed model size)

Agent autonomously reduced model from 50M to 13M (DEPTH=4, ASPECT_RATIO=32).
This made cross-phase comparisons invalid but is itself a Level 2 decision.

| Round | Baseline | Best | Improvement | Keeps | New Mechanism |
|-------|----------|------|-------------|-------|--------------|
| 3 (rewritten) | 1.273 | 1.220 | -0.053 | 6/11 | StepSizeCalibrator, crossover validation |
| 4 (rewritten) | 1.223 | 1.221 | -0.002 | 5/11 | PlateauDetector |
| 5 | 1.224 | 1.224 | 0 | 3/11 | Crossover frequency limiter (counterproductive) |
| 6 | 1.226 | 1.223 | -0.003 | 4/11 | Risk-aware SA, confidence-aware momentum |
| 7 | 1.224 | **1.219** | -0.006 | **10/11** | Outer loop freeze limit (most impactful) |

## Mechanisms Invented by Level 2 Agent

All invented autonomously — no human specification.

| # | Mechanism | Source Domain | What it does |
|---|-----------|--------------|-------------|
| 1 | CrashMemory | Software engineering | Track crash-causing params, warn LLM |
| 2 | MultiCandidate | Decision theory | Generate 3 candidates, pick best before GPU time |
| 3 | QuickTest | Software testing | 15s smoke test before full training |
| 4 | MomentumTracker | Optimization | Track param change directions and effects |
| 5 | SimulatedAnnealing | Statistical mechanics | Probabilistic accept of regressions |
| 6 | Revert-to-Best | Optimization | Rollback when stuck |
| 7 | ElitePool + Crossover | Evolutionary algorithms | Top-K configs + interpolation |
| 8 | StepSizeCalibrator | Adaptive optimization | Learn per-param step sizes |
| 9 | PlateauDetector | Optimization | Detect stagnation, force diversification |
| 10 | Freeze Limit | Control theory | Prevent outer loop from freezing all params |
| 11 | Confidence-aware Momentum | Statistics | Label early signals as tentative |
| 12 | Risk-aware SA | Risk management | Auto-reject large regressions |

## Known Issues

1. **Model size change**: Level 2 agent changed DEPTH/ASPECT_RATIO (50M→13M), making
   cross-phase results incomparable. Level 2 should only modify search mechanisms
   (runner.py), not the search space (train.py initial config).

2. **Accidental 3-layer nesting**: Level 2 agent called `cli.py bilevel` which includes
   outer.py (Level 1.5). Intended design was Level 2 → Level 1 (two layers), actual
   execution was Level 2 → Level 1.5 → Level 1 (three layers).

3. **Log overwrite**: Agent reused round3.log and round4.log filenames, losing 50M
   experiment data from those rounds.

## Files

- `logs/` — experiment logs for all rounds (round1-7)
- `report.json` — final round's structured report
- `outer/` — outer loop analysis artifacts
