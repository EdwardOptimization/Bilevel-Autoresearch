# Changelog

## [0.3.0] - 2026-03-23

### Added
- Training optimization domain on Karpathy's GPT benchmark
- Level 2 mechanism research via DeepSeek code generation
- Controlled ablation experiment (3×3 repeats, Level 2 improves 5× over Level 1)
- Paper submitted to AISC2026 (aixiv.260323.000006)
- BaseMechanismResearcher shared base class
- 49 unit tests

### Changed
- Restructured: article demo moved from core/ to domains/article_opt/
- core/ now contains only framework code (llm_client, state, inner_loop)
- eval() replaced with ast.literal_eval() for safety

### Fixed
- Dynamic module loading (sys.modules registration for @dataclass)
- train.py reset between independent experiment repeats
- SameFileError in runner snapshot
