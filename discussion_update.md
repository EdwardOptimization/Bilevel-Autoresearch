## Update: Controlled ablation on Karpathy's benchmark (3×3 repeats)

Since the original post, we ran a proper controlled experiment directly on the autoresearch benchmark (train.py, val_bpb, 300s budget) using **3 RTX 5090 servers in parallel**.

### Setup

Same LLM (DeepSeek) for all levels — eliminates model capability as a confound.

| Group | What it does | Mean Δval_bpb | vs Group A |
|-------|-------------|--------------|------------|
| **A** — Level 1 | Standard autoresearch (propose → train → keep/discard) | -0.009 ± 0.001 | 1× |
| **B** — Level 1 + outer loop | + Outer loop freezes/unfreezes params, shifts strategy | -0.007 ± 0.006 | 0.8× |
| **C** — Level 1 + outer loop + Level 2 | + DeepSeek generates new Python mechanisms, injected via importlib | **-0.045 ± 0.030** | **5×** |

3 independent repeats × 30 iterations each. train.py reset to baseline between repeats.

### What Level 2 discovered

Each repeat independently generated different mechanisms from different domains — no human specified which domains to explore:

- **Tabu Search Manager** (combinatorial optimization) — prevents revisiting failed parameter regions
- **Multi-Scale Bandit** (online learning) — balances exploration/exploitation across parameters
- **Systematic Orthogonal Exploration** (DOE) — forces search across orthogonal dimensions
- **GP Regressor** (Bayesian optimization) — reverted because sklearn wasn't installed

### Why Level 2 wins

Group A follows a near-deterministic search path: it tries WEIGHT_DECAY, then WINDOW_PATTERN, then **repeats the same failed proposal up to 22 times consecutively**. Level 2's mechanisms (Tabu Search, Orthogonal Exploration) break this loop and guided the LLM to discover that *reducing* TOTAL_BATCH_SIZE dramatically improves val_bpb — a direction Groups A and B never explored.

### Paper

We submitted to AISC 2026: [Bilevel Autoresearch: When the Optimization Loop Optimizes Itself](https://aixiv.science/abs/aixiv.260323.000006)

Full ablation report, experiment logs, and all Level 2 generated code: [GitHub](https://github.com/EdwardOptimization/Bilevel-Autoresearch)
