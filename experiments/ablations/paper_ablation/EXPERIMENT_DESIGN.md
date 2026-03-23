# Paper Ablation Experiment Design

## Research Question

Does adding optimization levels (outer loop, mechanism discovery) improve autoresearch performance, controlling for model capability?

## Experimental Setup

### Independent Variable: Number of Optimization Levels

| Group | Levels | Description |
|-------|--------|-------------|
| A | Level 1 | Pure autoresearch: propose → train → evaluate → keep/discard |
| B | Level 1 + 1.5 | + Outer loop adjusts search config every 5 iterations |
| C | Level 1 + 1.5 + 2 | + DeepSeek Level 2 invents new mechanisms between rounds |

### Controlled Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| LLM model | deepseek-chat | Same model for ALL levels — eliminates capability confound |
| GPU | RTX 5090 32GB | Same hardware across all servers |
| Training budget | 300s per run | Matches Karpathy's original benchmark |
| Total iterations | 30 per repeat | Same search budget for fair comparison |
| Baseline config | DEPTH=8, ASPECT_RATIO=64 (50M model) | Same starting point |
| Locked params | DEPTH, ASPECT_RATIO | Prevents model size changes |
| Repeats | 3 per group | For statistical significance |

### Dependent Variables (Metrics)

| Metric | Definition | Lower/Higher = Better |
|--------|-----------|----------------------|
| **val_bpb** | Validation bits per byte | Lower = better |
| **Best val_bpb** | Lowest val_bpb achieved in 30 iterations | Lower = better |
| **Improvement** | baseline_bpb - best_bpb | Higher = better |
| **Keep rate** | keeps / total iterations | Higher = better |
| **Crash rate** | crashes / total iterations | Lower = better |
| **Iterations to threshold** | First iteration reaching a target bpb | Lower = better |

## Group Details

### Group A — Level 1 (Baseline Autoresearch)

```
for i in range(30):
    proposal = LLM.propose(current_config, trace)
    new_config = apply(proposal)
    val_bpb = train(new_config)
    if val_bpb < best:
        keep(new_config)
    else:
        discard()
```

- `simple_mode=True` in TrainRunner
- Single proposal per iteration (no multi-candidate)
- Binary keep/discard (no simulated annealing)
- No quick-test pre-filter
- No crash memory, momentum, elite pool
- Crash warnings still present in prompt (from trace history)

### Group B — Level 1 + Level 1.5

Same inner loop as Group A, plus:

```
for outer_cycle in range(6):
    for i in range(5):
        # Same as Group A
        ...

    # Outer loop intervention (Level 1.5)
    analysis = LLM.analyze(trace, search_config)
    search_config.frozen_params = analysis.freeze_params
    search_config.strategy = analysis.strategy
    search_config.guidance = analysis.guidance
```

- Outer loop (DeepSeek) analyzes trace every 5 iterations
- Freezes ineffective params, shifts strategy, injects guidance
- Still `simple_mode=True` — no Level 2 mechanisms in the runner

### Group C — Level 1 + Level 1.5 + Level 2

Same as Group B, plus Level 2 mechanism discovery:

```
for round in range(3):  # 2 outer cycles per round
    for outer_cycle in range(2):
        for i in range(5):
            # Inner loop (starts simple, gets mechanisms added)
            ...
        # Level 1.5 intervention
        ...

    # Level 2 intervention (every 2 outer cycles)
    mechanism = LLM.research(trace, runner_code)  # 4-round research session
    runner = apply_mechanism(mechanism, runner)     # patch runner.py
    # Next round uses improved runner
```

- Starts with `simple_mode=True` (same as A and B)
- Every 2 outer cycles, DeepSeek runs a 4-round research session:
  1. Explore: propose mechanism hypotheses from diverse domains
  2. Critique: score impact × feasibility ÷ complexity
  3. Specify: write implementation spec
  4. Generate: produce Python code, syntax-check, apply
- Mechanisms are cumulative — each round builds on previous discoveries

## Analysis Plan

### Primary Analysis

1. **Bar chart**: Mean best val_bpb ± std for each group (3 bars)
2. **Line plot**: val_bpb over iterations for each group (showing convergence curves)
3. **Statistical test**: One-way ANOVA on best val_bpb across groups, followed by pairwise t-tests (A vs B, B vs C, A vs C) with Bonferroni correction

### Secondary Analysis

4. **Keep rate comparison**: Are higher levels more efficient per iteration?
5. **Crash rate comparison**: Do higher levels avoid more crashes?
6. **Mechanism inventory**: What mechanisms did Group C's Level 2 discover? Were they repeated across repeats?
7. **Convergence speed**: At what iteration does each group reach 90% of its final improvement?

### Expected Outcomes

- **A < B**: Outer loop should improve search focus (freeze ineffective params)
- **B < C**: Level 2 should improve search mechanisms (multi-candidate, crash memory, etc.)
- **A has highest crash rate**: No mechanism to avoid known-bad configs
- **C discovers similar mechanisms across repeats**: If the mechanism discovery is robust, similar ideas should emerge independently

### Potential Confounds

1. **LLM non-determinism**: Different runs may get different proposals. Mitigated by 3 repeats.
2. **Outer loop overcorrection**: Freezing too many params could hurt Group B/C. Mitigated by freeze limits.
3. **Level 2 code bugs**: Generated code may crash. Mitigated by syntax check + validation + backup restore.
4. **Time overhead**: Level 2 research sessions add ~2 min per round (4 LLM calls). Not counted in the 30-iteration budget but adds wall-clock time.

## Ethical Considerations

- No human subjects involved
- API costs: ~$5-10 per group (DeepSeek is cost-effective)
- GPU energy: ~10 hours × 3 servers × 300W = ~9 kWh total
