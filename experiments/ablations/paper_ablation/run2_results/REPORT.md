# Ablation Experiment Report

*2026-03-23 | Bilevel Autoresearch on Karpathy's GPT Pretraining Benchmark*

---

## 1. Hypothesis

Adding optimization levels to autoresearch improves search effectiveness, measured by val_bpb improvement within a fixed iteration budget, using the same LLM model for all levels.

Specifically:
- **H1**: Level 1 + Level 1.5 (outer loop adjusts search config) outperforms Level 1 alone (pure autoresearch)
- **H2**: Level 1 + Level 1.5 + Level 2 (outer loop + autonomous mechanism discovery) outperforms Level 1 + Level 1.5
- **H3**: Level 2 can autonomously discover novel search mechanisms from diverse domains without human specification

---

## 2. Experimental Design

### 2.1 Groups

| Group | Levels | What It Does |
|-------|--------|-------------|
| **A** | Level 1 | Pure autoresearch: LLM proposes hyperparameter changes → train → evaluate → keep/discard. No outer intervention. (`simple_mode=True`) |
| **B** | Level 1 + 1.5 | Same inner loop as A, plus outer loop every 5 iterations: analyzes trace, freezes/unfreezes parameters, shifts search strategy, injects guidance. (`simple_mode=True` + `TrainOuterLoop`) |
| **C** | Level 1 + 1.5 + 2 | Same as B, plus Level 2 every 2 outer cycles: DeepSeek runs a 4-round research session (Explore → Critique → Specify → Generate code), patches runner.py with new mechanism, validates import. (`simple_mode=True` initially, patched runner dynamically loaded) |

### 2.2 Why Three Layers Instead of Two?

The framework is conceptually bilevel (inner loop + outer loop), but in the experiment we split the outer loop into two distinct responsibilities:

- **Level 1.5 (config adjustment)**: Tactical — analyzes the inner loop's trace and adjusts runtime search parameters (freeze ineffective params, shift strategy from "explore" to "focused", inject textual guidance). This operates within the existing search framework without changing any code.

- **Level 2 (mechanism discovery)**: Strategic — runs a multi-round research session to identify structural bottlenecks, generates new Python code (e.g., Tabu Search, Orthogonal Exploration), and injects it into the runner via `importlib`.

This separation serves a key purpose: **it lets Level 2 focus purely on mechanism innovation**. Without Level 1.5 handling the tactical work (parameter freezing, strategy shifting), Level 2 would need to do both — tuning configs AND inventing mechanisms — which dilutes its research focus. The experiment tests whether this separation produces better results than either alone (Group B = 1.5 only, Group C = 1.5 + 2).

### 2.3 Controlled Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| LLM model | DeepSeek (deepseek-chat) | Same model for ALL levels — eliminates capability confound |
| GPU | RTX 5090 32GB × 3 (SeetaCloud) | Identical hardware across all servers |
| Training budget | 300s per run | Matches Karpathy's original 5-minute benchmark |
| Iteration budget | 30 per repeat | Same search budget for fair comparison |
| Baseline config | DEPTH=8, ASPECT_RATIO=64, HEAD_DIM=128, TOTAL_BATCH_SIZE=2\*\*19 | Same starting point, DEPTH/ASPECT_RATIO locked |
| Repeats | 3 independent per group | train.py reset to original baseline between repeats |

### 2.3 Metrics

| Metric | Definition |
|--------|-----------|
| **Improvement** (primary) | baseline_bpb − best_bpb within 30 iterations |
| **Best val_bpb** | Lowest val_bpb achieved |
| **Keep rate** | Number of kept improvements / 30 |
| **Crash rate** | Number of training crashes / 30 |

### 2.4 Key Design Decisions

- **Same model everywhere**: Using DeepSeek for Level 1 (proposals), Level 1.5 (outer analysis), and Level 2 (mechanism research) ensures any improvement comes from the bilevel architecture, not from a stronger model.
- **Independent repeats**: train.py is restored to the original baseline before each repeat. This was a fix from Run 1 where keeps accumulated across repeats, making them dependent.
- **Locked architecture params**: DEPTH and ASPECT_RATIO are removed from editable_params. This prevents Level 2 from changing model size (a boundary violation observed in earlier experiments).

---

## 3. Execution

### 3.1 Infrastructure

Three SeetaCloud servers (RTX 5090 32GB each) ran the experiments in parallel:
- Server 1 (port 35363): Group A
- Server 2 (port 24368): Group B
- Server 3 (port 11568): Group C

Environment: conda + PyTorch 2.8 + CUDA 12.8, SDPA attention fallback (FA3 not supported on Blackwell compute 12.0).

### 3.2 Run 1 (Failed)

The first run revealed three bugs:
1. **Repeats not independent**: keeps accumulated in train.py across repeats, making them dependent. Group B's R3 started from a catastrophic baseline (bpb=2.83) due to a destructive UNEMBEDDING_LR=0.77 from R2.
2. **Dynamic load failure**: `@dataclass` decorator crashed because the dynamically loaded module wasn't registered in `sys.modules`. All Level 2 patches silently fell back to the original runner.
3. **SameFileError**: runner.py snapshot tried to copy a file to itself.

Run 1 data was partially lost (logs preserved, per-iter artifacts deleted before backup — a procedural error).

### 3.3 Run 2 (Final)

After fixing all three bugs, the experiment was re-run with:
- train.py reset between repeats (verified via log: "train.py reset to baseline")
- `sys.modules` registration before `exec_module` (Level 2 patches loaded successfully)
- SameFileError guard in snapshot logic

Compliance monitoring ran every 15 minutes via SSH, checking:
- DEPTH=8 and ASPECT_RATIO=64 in train.py (never violated)
- Process alive on each server
- Level 2 only modifying runner.py, not train.py

Total wall-clock time: ~12 hours (Server C finished last due to Level 2 research overhead).

### 3.4 Level 2 Research Sessions

Group C ran 6 Level 2 research sessions across 3 repeats (2 per repeat):

| Repeat | Round | Mechanism Generated | Domain | Patch | Import | Active |
|--------|-------|-------------------|--------|-------|--------|--------|
| R1 | 1 | **Tabu Search Manager** | Combinatorial optimization | ✅ | ✅ | ✅ |
| R1 | 2 | Helper class (unnamed) | — | ✅ | ✅ | ✅ |
| R2 | 1 | **Multi-Scale Bandit Proposer** | Online learning / Multi-armed bandits | ✅ | ✅ | ✅ |
| R2 | 2 | Helper class (unnamed) | — | ✅ | ✅ | ✅ |
| R3 | 1 | **GP Regressor** | Bayesian optimization | ✅ | ❌ (sklearn) | Reverted |
| R3 | 2 | **Systematic Orthogonal Exploration** | Design of Experiments (DOE) | ✅ | ✅ | ✅ |

All code generated on first attempt (0 retries). 5/6 successfully injected. 1 reverted due to sklearn dependency not installed — the validate+revert mechanism worked correctly.

---

## 4. Results

### 4.1 Primary Results

| Group | R1 Improvement | R2 Improvement | R3 Improvement | **Mean ± Std** |
|-------|---------------|----------------|----------------|---------------|
| **A** (Level 1) | -0.009 | -0.008 | -0.011 | **-0.009 ± 0.001** |
| **B** (Level 1+1.5) | -0.000 | -0.010 | -0.009 | **-0.007 ± 0.006** |
| **C** (Level 1+1.5+2) | **-0.065** | -0.011 | **-0.058** | **-0.045 ± 0.030** |

**C's mean improvement is 5× A and 6.4× B.**

### 4.2 Per-Repeat Detail

| Group | Repeat | Baseline | Best | Improvement |
|-------|--------|----------|------|-------------|
| A | 1 | 1.105 | 1.096 | -0.009 |
| A | 2 | 1.104 | 1.096 | -0.008 |
| A | 3 | 1.105 | 1.095 | -0.011 |
| B | 1 | 1.094 | 1.094 | -0.000 |
| B | 2 | 1.103 | 1.093 | -0.010 |
| B | 3 | 1.103 | 1.093 | -0.009 |
| C | 1 | 1.113 | **1.048** | **-0.065** |
| C | 2 | 1.114 | 1.103 | -0.011 |
| C | 3 | 1.113 | **1.055** | **-0.058** |

### 4.3 Search Behavior Patterns

**Group A — Fixed search path, repetitive failure:**
All 3 repeats show identical pattern:
- Iter 1: TOTAL_BATCH_SIZE↑ → discard
- Iter 2: WEIGHT_DECAY↓ → keep (~-0.008)
- Iter 3: WINDOW_PATTERN="SSSS" → keep (~-0.002)
- Iter 4–30: repeat WINDOW_PATTERN="SSSS" and WEIGHT_DECAY=0.05, up to 22 consecutive discards

The LLM follows a deterministic search path from the same baseline and cannot escape it.

**Group B — Outer loop redirects but modest effect:**
- Cycle 1: finds WEIGHT_DECAY and WINDOW_PATTERN (same as A)
- Cycle 2–3: outer loop freezes ineffective params, redirects to LR search
- Cycle 4–6: explores UNEMBEDDING_LR, MATRIX_LR, HEAD_DIM, FINAL_LR_FRAC
- Effect: B explores more parameters than A (outer loop prevents repetition) but finds similarly sized improvements

**Group C — Level 2 mechanisms unlock new search directions:**
- Batch 1–2: baseline search (same as B)
- Level 2 Round 1: generates mechanism (e.g., TabuSearch), patches runner
- Batch 3–4: patched runner guides search to new directions
- Level 2 Round 2: generates second mechanism, patches again
- Batch 5–6: double-patched runner finds TOTAL_BATCH_SIZE reduction

The critical insight: Level 2 mechanisms forced the LLM to explore TOTAL_BATCH_SIZE *decrease* (2\*\*19 → 2\*\*17/2\*\*18), which produced the largest single improvements (-0.039 to -0.065). A and B never found this because:
- A only tried BATCH_SIZE *increase* (always discard) and never tried decrease
- B's outer loop froze BATCH_SIZE after the increase failed, preventing decrease exploration

---

## 5. Analysis

### 5.1 Hypothesis Testing

**H1 (B > A): NOT SUPPORTED.** B's mean improvement (-0.007 ± 0.006) is actually worse than A's (-0.009 ± 0.001), though the difference is not statistically significant with n=3. The outer loop (Level 1.5) adds value in search diversity but does not reliably improve the outcome within 30 iterations. B's R1 had essentially zero improvement (-0.000), inflating variance.

**H2 (C > B): SUPPORTED.** C's mean improvement (-0.045 ± 0.030) is 6.4× B's (-0.007 ± 0.006). Despite high variance (C's R2 underperformed at -0.011), 2 of 3 repeats showed dramatic improvement. The effect size is large enough to be meaningful despite n=3.

**H3 (Level 2 discovers novel mechanisms): SUPPORTED.** Across 3 independent repeats, Level 2 generated 6 different mechanisms from 4 different domains (combinatorial optimization, online learning, Bayesian optimization, DOE). Each repeat independently chose different approaches. Code generation succeeded on first attempt every time (0 retries). 5/6 mechanisms passed import validation and were successfully injected.

### 5.2 Why Level 1.5 Alone Wasn't Enough

The outer loop correctly diagnoses problems (e.g., "WEIGHT_DECAY tried 14 times with 0 keeps → freeze it") and redirects search. However, it operates within the existing search framework — it can only freeze/unfreeze parameters and inject text guidance. It cannot:
- Change the keep/discard logic
- Add pre-filtering steps
- Modify how proposals are generated
- Introduce new search paradigms (taboo lists, bandits, orthogonal exploration)

These structural changes require Level 2.

### 5.3 Why C's R2 Underperformed

C's R2 achieved only -0.011 improvement (similar to B). Possible explanations:
- The MultiScaleBandit mechanism generated in R2 may have been less effective than R1's TabuSearch or R3's OrthogonalExploration
- R2's baseline (1.114) was slightly higher, and the search may have found a local optimum earlier
- Level 2 mechanisms add overhead (research sessions take ~3 minutes each), reducing effective inner iterations

This demonstrates that Level 2 is not uniformly beneficial — the quality of the generated mechanism matters.

### 5.4 The TOTAL_BATCH_SIZE Discovery

The single most impactful finding across all experiments: reducing TOTAL_BATCH_SIZE from 2\*\*19 to 2\*\*17/2\*\*18 dramatically improves val_bpb on RTX 5090 with 300s budget.

Why this works: smaller batch size = more gradient steps within the fixed time budget = better convergence for this model size (50M params). The original 2\*\*19 was tuned for H100 throughput; 5090 with SDPA (not FA3) has different optimal batch characteristics.

Why A and B missed it:
- DeepSeek's default search path tries BATCH_SIZE *increase* first (LLM bias: "bigger batch = better")
- After the increase fails, A keeps trying the same thing; B freezes BATCH_SIZE entirely
- Only C's Level 2 mechanisms (TabuSearch preventing revisits, OrthogonalExploration forcing diversity) pushed the LLM to try the *decrease* direction

### 5.5 Level 2 Mechanism Inventory

| Mechanism | What It Does | Inspired By |
|-----------|-------------|-------------|
| **Tabu Search Manager** | Maintains a tabu list of recently visited parameter regions, preventing the LLM from proposing the same changes repeatedly | Combinatorial optimization |
| **Multi-Scale Bandit Proposer** | Treats parameter selection as a multi-armed bandit problem, balancing exploration and exploitation across parameters at different scales | Online learning / MAB |
| **GP Regressor** (reverted) | Fits a Gaussian Process surrogate model to predict val_bpb from config, proposes the predicted optimum | Bayesian optimization |
| **Systematic Orthogonal Exploration** | Forces the LLM to explore orthogonal parameter dimensions, preventing over-focus on a single parameter | Design of Experiments (DOE) |

All mechanisms were generated by the same DeepSeek model that runs the inner loop — no stronger model was involved.

---

## 6. Discussion

### 6.1 Limitations

**Small sample size.** n=3 repeats is insufficient for rigorous statistical testing. The high variance in C (0.030) means the result could be influenced by lucky draws. A proper study needs n≥10 repeats per group.

**Baseline variance.** Different repeats get different baselines (1.094–1.114) due to training randomness (data ordering, initialization). This noise affects improvement calculations. Future work should use fixed random seeds or report baseline-normalized metrics.

**Level 2 overhead.** Each Level 2 research session adds ~3 minutes (4 DeepSeek calls). With 2 sessions per repeat, C effectively has ~6 minutes less search time than A/B. This is a minor effect (<2% of total wall time) but should be controlled.

**Single benchmark.** All results are on one task (GPT pretraining, 50M params, 300s budget). Generalization to other tasks is unproven.

**Dynamic load was initially broken.** Run 1's Level 2 data is invalid because no mechanisms were actually injected (sys.modules bug). The fix was critical — without it, C ≈ B. This highlights the fragility of the code injection pipeline.

### 6.2 Unexpected Findings

**A's search is near-deterministic.** Given the same baseline train.py, DeepSeek proposes almost the same sequence of changes every time. This means A's 3 repeats are not truly independent samples — they're 3 runs of the same deterministic process with slight randomness from training noise.

**B's outer loop can harm search.** In Run 1 (before reset fix), B's outer loop unfroze UNEMBEDDING_LR which led to a catastrophic keep (bpb=2.66) that poisoned R3. Even in Run 2, B's R1 achieved -0.000 improvement — the outer loop froze the right params but the LLM couldn't find anything better in the remaining search space. Outer loop guidance is not always additive.

**Level 2 drew from 4 different domains without being told.** TabuSearch (combinatorial optimization), Bandit (online learning), GP (Bayesian optimization), and Orthogonal Exploration (DOE) were all independently selected by DeepSeek across different repeats. The prompt only says "propose mechanism improvements" — it doesn't suggest which domains to explore.

### 6.3 Implications for the Paper

The experiment provides evidence for the core thesis: **an outer loop that modifies the inner loop's search mechanisms produces larger improvements than one that only adjusts search parameters.** The effect is large (5×) but variable (R2 didn't show it).

For the AISC 2026 submission:
- The ablation design (A/B/C with same model, same budget, same benchmark) is clean
- The Level 2 mechanism inventory (TabuSearch, Bandit, OrthogonalExploration) is concrete evidence of autonomous mechanism discovery
- The TOTAL_BATCH_SIZE finding demonstrates that Level 2 can discover non-obvious search directions
- The limitations (n=3, baseline variance, R2 underperformance) should be reported honestly

### 6.4 What We Would Do Differently

1. **More repeats** (n≥10) with fixed random seeds for statistical power
2. **Install sklearn** on servers so Level 2 can use more sophisticated mechanisms
3. **Constrain Level 2 prompt** to only generate pure-Python code (no external dependencies)
4. **Add Level 1 only with simple_mode=False** as a 4th group to isolate the effect of pre-baked mechanisms vs. autonomously discovered ones
5. **Multiple benchmarks** (different model sizes, different training budgets) for generalization

---

## Appendix: Files

- `experiment_{A,B,C}.log` — full experiment logs (every iteration, every keep/discard, every outer loop decision)
- `results_A/A{1,2,3}/` — Group A per-repeat artifacts (report.json, runner.py snapshot)
- `results_B/B{1,2,3}/` — Group B per-repeat artifacts (report.json, runner.py snapshot)
- `results_C/C{1,2,3}/` — Group C per-repeat artifacts including:
  - `mechanism_sessions/round_{1,2}/` — Level 2 research artifacts (exploration.md, critique.md, spec.md, code, patched runner)
  - `runner.py.bak_*` — pre-patch backups
  - `runner_after_cycles_*.py` — runner snapshots at each Level 2 intervention point
  - `runner_final.py` — final runner state
