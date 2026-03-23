# Changelog

## [0.3.0] - 2026-03-23

### Added
- Training optimization domain on Karpathy's GPT benchmark
- Level 2 mechanism research via DeepSeek code generation
- Controlled ablation experiment (3×3 repeats, Level 2 improves 5× over Level 1)
- Paper submitted to AISC2026 (aixiv.260323.000006)
- BaseMechanismResearcher shared base class
- 110 unit tests (llm_client, pipeline, runner, state, inner_loop, train_opt, mechanism_research, skills)

### Changed
- Restructured: article demo moved from core/ to domains/article_opt/
- core/ now contains only framework code (llm_client, state, inner_loop)
- eval() replaced with ast.literal_eval() for safety

### Changed (cont.)
- train_opt/runner.py split into mechanisms/ subpackage (1543 → 1012 lines)
- Both mechanism_research.py files migrated to subclass BaseMechanismResearcher
- Auto-detect .venv or sys.executable python for training subprocess

### Fixed
- Dynamic module loading (sys.modules registration for @dataclass)
- train.py reset between independent experiment repeats
- SameFileError in runner snapshot
- eval() → ast.literal_eval() for LLM-supplied values
