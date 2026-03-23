# Domains

Each subdirectory is a self-contained optimization domain that uses the bilevel autoresearch framework.

## Available Domains

### train_opt — Training Optimization

Bilevel autoresearch on Karpathy's GPT pretraining benchmark.

- **Metric**: val_bpb (validation bits per byte, lower is better)
- **Inner loop**: LLM proposes hyperparameter changes → train → evaluate → keep/discard
- **Outer loop**: Analyzes trace, freezes/unfreezes parameters, adjusts search strategy
- **Level 2**: Autonomous agent modifies inner loop mechanisms

```bash
# Full bilevel experiment
python -m domains.train_opt.cli --provider deepseek bilevel --inner-budget 5 --outer-cycles 2

# Inner loop only
python -m domains.train_opt.cli --provider deepseek inner --iterations 10
```

Requires [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) cloned at `~/karpathy_autoresearch`.

> **Note:** The article optimization demo lives in `domains/article_opt/`. The training domain
> (`domains/train_opt/`) is the recommended pattern for new domains — fully self-contained with
> its own runner, outer loop, and config.

## Adding a New Domain

See `train_opt/` for the pattern. Each domain needs:
- `runner.py` — inner loop implementation
- `outer.py` — outer loop (trace analysis → config updates)
- `config.py` — SearchConfig with editable parameters
- `cli.py` — entry point
