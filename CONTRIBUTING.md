# Contributing to Bilevel Autoresearch

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/EdwardOptimization/Bilevel-Autoresearch.git
cd Bilevel-Autoresearch
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: LLM client (think-tag stripping, JSON parsing), pipeline stages, runner (inject_stage), state management.
They run offline — no API key required.

## Key Architecture Constraints

Before contributing, understand these design invariants:

1. **Evaluator isolation**: The evaluator must NEVER receive lesson memory.
   Lessons only influence *proposals* (stages that produce research content), never *judgments*.

2. **Skills vs Lessons**: Promoted skills are distilled from many lessons. Skills are structural
   (how to do something); lessons are empirical (what went wrong/right in specific runs).

3. **Skill injection order**: Skills are prepended before lessons in the injection block.
   The order matters — skills are higher-authority guidance.

4. **No hardcoded models**: Stage implementations use `self.model` (inherits from config).
   Never hardcode a model name in pipeline stages.

## Adding a New Pipeline Stage

1. Create `core/pipeline/your_stage.py` extending `BaseStage`
2. Add `name = "your_stage"` class attribute
3. Register in `core/runner.py` — add to `self.stages` list in `InnerRunner.__init__`
4. Decide if this stage receives lesson injection (add to relevant stages if yes)
5. Write tests if the stage has non-trivial logic

## Adding a New Domain

The framework supports multiple optimization domains beyond article editing.
See `domains/train_opt/` for a complete example.

New domains should be self-contained in `domains/your_domain/`. They may import `core.llm_client` for LLM access but should implement their own runner and outer loop — do not subclass or depend on `core.runner` or `core.outer_loop`.

To add a new domain:
1. Create `domains/your_domain/` with these files:
   - `runner.py` — inner loop (propose → execute → evaluate → keep/discard)
   - `outer.py` — outer loop (analyze trace → modify search config)
   - `config.py` — SearchConfig defining editable parameters
   - `cli.py` — entry point
2. Define your evaluation metric (must be objective and automated)
3. Implement a runner class with `run_iteration()` and `run_baseline()` methods
4. Wire outer loop to modify `SearchConfig` based on trace analysis

Or use **Level 2 mechanism research** to let the outer LLM generate stages automatically:
```bash
python cli.py mechresearch --article article2
```

## Adding a New LLM Provider

Edit `core/llm_client.py` — add an entry to `PROVIDERS`:

```python
"yourprovider": {
    "base_url": "https://api.example.com/v1",
    "default_model": "model-name",
    "api_key_env": "YOURPROVIDER_API_KEY",
    "native_sdk": False,  # True only for Anthropic
},
```

All OpenAI-compatible providers work with `native_sdk: False`.

## Code Style

```bash
ruff check core/ tests/    # lint
ruff format core/ tests/   # format
```

## Pull Request Guidelines

- Keep PRs focused — one concern per PR
- Tests must pass: `python -m pytest tests/ -v`
- Lint must pass: `ruff check core/ tests/`
- Do not commit `artifacts/` or `memory/` directories

## Reporting Issues

Please include:
- Python version
- Provider being used
- Full error traceback
- Whether the issue is reproducible
