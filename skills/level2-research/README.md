# Level 2 Research — Meta-Optimize Any Search Loop

> Your optimization loop is stuck. Level 2 reads your code, diagnoses why, and generates new mechanisms as injectable Python code.

## Install

```bash
npx skills add EdwardOptimization/Bilevel-Autoresearch@level2-research
```

Or manually copy `SKILL.md` to `~/.claude/skills/level2-research/`.

## Usage

```
> /level2-research
> "My optimization loop keeps proposing the same changes"
> "Why is my search stuck at the same score?"
```

## How It Works

```
Round 1: EXPLORE  — Survey mechanisms from adjacent algorithmic domains
Round 2: CRITIQUE — Evaluate each against the specific failure mode
Round 3: SPECIFY  — Write precise interface spec
Round 4: GENERATE — Produce runnable Python code + integration patch
```

## Example

```
User: "My research loop keeps tweaking LR and ignoring batch size."

→ Diagnosis: parameter fixation
→ Round 1: Tabu Search, Bandit selector, Orthogonal sampling
→ Round 2: Select Tabu Search (directly prevents re-proposing failed regions)
→ Round 3: TabuSearchManager interface spec
→ Round 4: 30-line Python class + integration diff
→ Result: proposal diversity 3x, found better config in 5 iterations
```

## Mechanism Catalog

This skill draws from a catalog of 156 mechanism categories (distilled from 9,270 autonomously generated mechanisms). Common ones:

| Symptom | Mechanism | What it does |
|---------|-----------|-------------|
| Repeating proposals | Tabu Search | Blocks proposals too similar to recent attempts |
| Plateau | Simulated Annealing | Accepts slightly worse results to escape local optima |
| Noisy evaluations | Noise Floor Estimator | Filters real improvements from measurement noise |
| Timid exploration | Adaptive Step Size | Calibrates change magnitude from history |
| Repeating failures | Crash Memory | Records which changes caused crashes |
| Not learning | Elite Pool | Maintains top-K configs, enables crossover |

Full catalog: [`artifacts/level2_10k_mechanisms/`](../../artifacts/level2_10k_mechanisms/)

## Related

- [Bilevel Autoresearch paper](https://arxiv.org/abs/2603.23420)
- Part of [Bilevel-Autoresearch](https://github.com/EdwardOptimization/Bilevel-Autoresearch)
