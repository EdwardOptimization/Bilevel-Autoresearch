# Ablation Experiment Report — Run 2 (Final)

*2026-03-23 | RTX 5090 32GB × 3 servers | DeepSeek for all levels | 30 iters × 3 repeats × 3 groups*

## Setup

- **Benchmark**: Karpathy autoresearch (train.py, val_bpb metric, TIME_BUDGET=300s)
- **Model**: 50.3M params (DEPTH=8, ASPECT_RATIO=64, HEAD_DIM=128, locked)
- **GPU**: RTX 5090 32GB, DEVICE_BATCH_SIZE=64, SDPA fallback (Blackwell)
- **LLM**: DeepSeek (deepseek-chat) for all levels — same model eliminates capability confound
- **Repeats**: 3 independent repeats per group (train.py reset to baseline between repeats)

## Results

| Group | R1 Imp | R2 Imp | R3 Imp | **Mean ± Std** |
|-------|--------|--------|--------|---------------|
| **A** (Level 1) | -0.009 | -0.008 | -0.011 | **-0.009 ± 0.001** |
| **B** (Level 1+1.5) | -0.000 | -0.010 | -0.009 | **-0.007 ± 0.006** |
| **C** (Level 1+1.5+2) | **-0.065** | -0.011 | **-0.058** | **-0.045 ± 0.030** |

**C's mean improvement is 5× A and 6.4× B.**

### Per-Repeat Detail

| Group | Repeat | Baseline | Best | Improvement | Keeps/30 |
|-------|--------|----------|------|-------------|----------|
| A | 1 | 1.105 | 1.096 | -0.009 | ~3 |
| A | 2 | 1.104 | 1.096 | -0.008 | ~2 |
| A | 3 | 1.105 | 1.095 | -0.011 | ~3 |
| B | 1 | 1.094 | 1.094 | -0.000 | ~1 |
| B | 2 | 1.103 | 1.093 | -0.010 | ~3 |
| B | 3 | 1.103 | 1.093 | -0.009 | ~4 |
| C | 1 | 1.113 | **1.048** | **-0.065** | — |
| C | 2 | 1.114 | 1.103 | -0.011 | — |
| C | 3 | 1.113 | **1.055** | **-0.058** | — |

## Level 2 Mechanisms Generated

| Repeat | Round | Mechanism | Inspired By | Patch | Import | Status |
|--------|-------|-----------|------------|-------|--------|--------|
| R1 | 1 | TabuSearch Manager | Combinatorial optimization | ✅ | ✅ | **Active** |
| R1 | 2 | (unnamed helper) | — | ✅ | ✅ | **Active** |
| R2 | 1 | MultiScaleBandit Proposer | Online learning / MAB | ✅ | ✅ | **Active** |
| R2 | 2 | (unnamed helper) | — | ✅ | ✅ | **Active** |
| R3 | 1 | GP Regressor | Bayesian optimization | ✅ | ❌ sklearn | **Reverted** |
| R3 | 2 | Systematic Orthogonal Exploration | DOE / Orthogonal arrays | ✅ | ✅ | **Active** |

**6 research sessions, 5 successful injections, 1 revert (sklearn dependency).**
Each repeat independently discovered different mechanisms from different domains.

## Key Findings

### 1. Level 2 produces dramatically larger improvements
C's mean improvement (-0.045) is 5× larger than A (-0.009). The effect is driven by
R1 and R3 where Level 2 mechanisms successfully guided the search to discover
TOTAL_BATCH_SIZE reduction (2**19 → 2**17/2**18) as a major improvement lever.

### 2. Level 1 (Group A) gets stuck in repetitive loops
A consistently shows the same pattern across all 3 repeats:
- Iter 1-3: find WEIGHT_DECAY and WINDOW_PATTERN improvements (~3 keeps)
- Iter 4-30: repeat the same failed proposals (WINDOW_PATTERN="SSSS" up to 22 times consecutively)

Without outer loop or Level 2, the LLM cannot escape its fixed search path.

### 3. Level 1.5 (Group B) provides modest but inconsistent benefit
B's outer loop correctly freezes ineffective params and shifts strategy, but
the improvement over A is not statistically significant (p > 0.05 with n=3).
B's R1 had almost no improvement (-0.000) which inflates variance.

### 4. Level 2 mechanisms are reproducible but variable
Different repeats independently discover different mechanisms (TabuSearch, Bandit,
OrthogonalExploration) from different domains. This demonstrates that the mechanism
discovery capability is robust, though the specific mechanisms vary.

### 5. TOTAL_BATCH_SIZE reduction is the single biggest improvement
Across C's best repeats, the largest val_bpb drops came from reducing TOTAL_BATCH_SIZE
from 2**19 to 2**17/2**18. A and B never discovered this because:
- A repeated the same proposals without exploring BATCH_SIZE direction
- B's outer loop froze BATCH_SIZE early based on BATCH_SIZE increase failures

## Known Issues

1. **Baseline variance**: Different repeats get different baselines (~1.094-1.114)
   due to training randomness, making per-repeat comparisons noisy.

2. **C's R2 underperformed**: -0.011 improvement, similar to A/B. The Level 2
   mechanisms in R2 (MultiScaleBandit) may not have been as effective as R1's
   TabuSearch or R3's OrthogonalExploration.

3. **sklearn dependency failure**: R3 Round 1 generated code using sklearn
   (not installed). The validate+revert mechanism correctly handled this,
   but it wasted one Level 2 round.

4. **Dynamic load fix was critical**: Without the sys.modules registration fix,
   no Level 2 mechanisms would have worked (as demonstrated in Run 1).

## Files

- `experiment_{A,B,C}.log` — full experiment logs
- `results_{A,B,C}/` — per-repeat artifacts (proposals, reports, runner snapshots, mechanism sessions)
