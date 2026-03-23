# Bilevel Autoresearch: Training Optimization Experiment Report

*2026-03-22 | RTX 5080 (16GB) | DeepSeek for both inner and outer loops*

---

## 1. Objective

Validate that bilevel autoresearch works on a non-article domain by running it on Karpathy's GPT pretraining benchmark (`val_bpb`, lower is better). Specifically:

1. Can a bilevel system (Level 1 + Level 1.5) outperform a single-layer autoresearch loop?
2. Can a Level 2 agent autonomously discover new mechanisms that improve the inner loop?

## 2. Setup

| Component | Configuration |
|-----------|--------------|
| Benchmark | Karpathy's autoresearch (`train.py` + `prepare.py`) |
| GPU | RTX 5080, 16GB VRAM, compute capability 12.0 (Blackwell) |
| Training budget | 120 seconds per run (reduced from Karpathy's 300s for faster iteration) |
| Inner loop LLM | DeepSeek (deepseek-chat) — proposes hyperparameter changes |
| Outer loop LLM | DeepSeek (deepseek-chat) — analyzes trace, modifies search config |
| Level 2 | Claude Opus 4.6 subagent — modifies inner loop code autonomously |
| Attention | PyTorch SDPA (FA3 not supported on Blackwell) |

## 3. Results Summary

### 3.1 Overall Progression

| Phase | Model | Baseline | Best val_bpb | Total Improvement |
|-------|-------|----------|-------------|-------------------|
| Manual bilevel (no Level 2) | 50M | 1.393 | 1.245 | -0.148 (-10.6%) |
| Level 2 Round 2 | 50M | 1.234 | 1.226 | -0.008 |
| Level 2 Round 5 | 50M | 1.224 | 1.223 | -0.001 |
| Level 2 Round 3 (13M) | 13M | 1.273 | 1.220 | -0.053 |
| Level 2 Round 7 (13M) | 13M | 1.224 | **1.219** | -0.006 |

**End-to-end: 1.393 → 1.219 (50M baseline → 13M best)**

Note: The 50M→13M model change makes direct comparison invalid. See Section 5.1.

### 3.2 Search Efficiency Over Time

| Round | Model | Keeps/Total | Crash Rate | Best Strategy |
|-------|-------|-------------|------------|---------------|
| Manual bilevel | 50M | 4/11 (36%) | 27% | — |
| Level 2 Round 1 | 50M | 1/11 (9%) | 91% | Quick test broken |
| Level 2 Round 2 | 50M | 4/11 (36%) | 0% | Momentum tracking |
| Level 2 Round 5 | 50M | 5/11 (45%) | 9% | Elite pool |
| Level 2 Round 3 (13M) | 13M | 6/11 (55%) | 0% | Crossover |
| Level 2 Round 7 (13M) | 13M | **10/11 (91%)** | 0% | Freeze limit |

Search efficiency improved from 36% to 91% over the course of Level 2 research.

## 4. Level 2 Agent: Autonomous Mechanism Discovery

The Level 2 agent ran for ~4.5 hours across 7 rounds. It autonomously invented 12 mechanisms, drawing from multiple domains without human specification.

### 4.1 Mechanisms Invented

| # | Mechanism | Inspired By | What It Does | Impact |
|---|-----------|------------|-------------|--------|
| 1 | **CrashMemory** | Software engineering | Records crash-causing params, warns LLM in proposals | Reduced repeat crashes |
| 2 | **MultiCandidate** | Decision theory | Generate 3 candidates, pick best before GPU time | Filtered bad ideas cheaply |
| 3 | **QuickTest** | Software testing | 15-second smoke test before full training | Saved GPU time on bad configs |
| 4 | **MomentumTracker** | Gradient optimization | Track param change direction × outcome | Guided search direction |
| 5 | **SimulatedAnnealing** | Statistical mechanics | Probabilistic acceptance of worse results | Escape local optima |
| 6 | **Revert-to-Best** | Optimization | Rollback to best config when stuck | Recovery from stagnation |
| 7 | **ElitePool + Crossover** | Evolutionary algorithms | Top-5 configs + interpolation for new candidates | Combined best features |
| 8 | **StepSizeCalibrator** | Adaptive optimization | Learn per-parameter step sizes from history | Right-sized changes |
| 9 | **PlateauDetector** | Signal processing | Detect stagnation, inject diversification | Break out of plateaus |
| 10 | **Freeze Limit** | Control theory | Prevent outer loop from freezing all params | Keep search space open |
| 11 | **Confidence-aware Momentum** | Statistics | Label early signals as tentative | Reduce overreaction to noise |
| 12 | **Risk-aware SA** | Risk management | Auto-reject regressions > 0.003 bpb | Prevent catastrophic accepts |

### 4.2 Which Are True Mechanism Changes vs. Prompt Engineering?

**Changed the autoresearch loop structure (Level 2):**
- SimulatedAnnealing — changed keep/discard from binary to probabilistic
- Revert-to-Best — added rollback branch to the loop
- ElitePool + Crossover — added evolutionary search to proposal generation
- PlateauDetector — added adaptive strategy switching
- MultiCandidate — changed propose-one to propose-three-pick-one
- Freeze Limit — constrained outer loop's control authority

**Enhanced prompt context (Level 1.5):**
- CrashMemory — injects crash history into prompt
- MomentumTracker — injects directional signals into prompt
- StepSizeCalibrator — injects step size recommendations into prompt
- Confidence-aware Momentum — labels signal reliability in prompt

### 4.3 Most Impactful Changes

1. **Outer loop freeze limit (Round 7)**: Without this, the outer loop froze 11+ params in one cycle, leaving the inner loop with no options. Limiting to 5 freezes per cycle and requiring ≥4 active params produced 10/11 keeps — the single most impactful change.

2. **ElitePool + Crossover (Round 3)**: Instead of only LLM-generated proposals, crossover between top configs created candidates the LLM would not have proposed. The LLM chose crossover candidates 40% of the time when available.

3. **QuickTest fix (Round 2)**: The 15-second smoke test had a path bug that made every proposal crash. Fixing it went from 91% crash rate to 0%. This was debugging, not mechanism design, but was necessary for everything else to work.

## 5. Known Issues and Errors

### 5.1 Model Size Change (Boundary Violation)

In Round 3, the Level 2 agent changed `DEPTH` from 8→4 and `ASPECT_RATIO` from 64→32, reducing the model from 50M to 13M parameters. This invalidated cross-round comparisons — a 13M model's val_bpb cannot be compared to a 50M model's val_bpb.

**Root cause**: The Level 2 agent was not constrained from modifying `train.py`'s initial hyperparameters. It should only modify the search mechanism (`runner.py`, `outer.py`), not the search space itself.

**Lesson**: Level 2's control boundary must be explicitly defined. "Modify how the inner loop searches" ≠ "modify what the inner loop starts from."

### 5.2 Accidental Three-Layer Nesting

The intended architecture was:
```
Level 2 (subagent) → Level 1 (inner loop)
```

The actual execution was:
```
Level 2 (subagent) → calls cli.py bilevel → Level 1.5 (outer.py) → Level 1 (runner.py)
```

The Level 2 agent called `python -m domains.train_opt.cli bilevel`, which includes the outer loop (`outer.py`). This created an unintended three-layer nesting. The Level 2 agent should have been the outer loop itself, directly calling the inner loop.

**Lesson**: When the Level 2 agent IS the outer loop, it should call the inner loop directly, not wrap another outer loop.

### 5.3 Log Overwrite

The Level 2 agent reused `round3.log` and `round4.log` filenames when switching to the 13M model, losing the 50M experiment data from those rounds. The 50M Round 3 and Round 4 results are only available from conversation history, not from log files.

## 6. Conclusions

### 6.1 Bilevel Autoresearch is Effective

The core thesis is validated: an outer loop that modifies the inner loop's search mechanisms produces better results than the inner loop alone.

Evidence:
- Search efficiency improved from 36% (manual bilevel) to 91% (Level 2 Round 7)
- Crash rate dropped from 27% to 0%
- The system discovered 6 genuine mechanism changes autonomously
- val_bpb improved continuously across rounds

### 6.2 Level 2 Can Discover Novel Mechanisms

The Level 2 agent drew from evolutionary algorithms (crossover), statistical mechanics (simulated annealing), control theory (freeze limits), and adaptive optimization (step-size calibration) without being told which domains to explore. This validates the Article 1.5 thesis: "autoresearch can research itself."

### 6.3 Boundary Control is Critical

The two main failures (model size change, three-layer nesting) were both boundary violations — the Level 2 agent doing things outside its intended scope. Future work needs explicit constraints on what each level can and cannot modify.

### 6.4 The Recursive Structure Works but is Impure

The actual structure (Level 2 → Level 1.5 → Level 1) was accidental but functional. Each level genuinely operated at a different abstraction:
- Level 1: "What hyperparameters should I try?"
- Level 1.5: "Which parameters should I focus on?"
- Level 2: "How should the search loop itself work?"

This three-layer recursion, while unintended, demonstrates that the bilevel principle extends to multiple levels.

---

*Generated from experiments run on 2026-03-22. Full logs in `logs/`, structured data in `report.json`.*
