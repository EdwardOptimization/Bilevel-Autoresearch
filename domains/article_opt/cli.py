"""CLI entry point for the dual-layer article optimizer.

Usage:
  # Run full dual-layer experiment (all articles, up to 5 outer cycles):
  python cli.py run

  # Run inner loop only on one article (debug/test):
  python cli.py inner --article article15

  # Run a single pipeline pass on an article (smoke test):
  python cli.py once --article article1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load project .env before anything else
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)

from core.inner_loop import InnerLoopController
from core.llm_client import configure
from core.state import InnerLoopState, OuterLoopState

from .outer import OuterAnalyzer, OuterLoopController
from .runner import InnerRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent  # project root
ARTICLES_DIR = BASE_DIR / "articles"
REFERENCE_DOC = Path(__file__).parent / "reference_frameworks.md"

ARTICLE_FILES = {
    "article1": "article1_llm_research_depth.md",
    "article15": "article15_meta_optimization.md",
    "article2": "article2_agent_team_scale.md",
}


def load_articles(ids: list[str]) -> dict[str, str]:
    articles = {}
    for aid in ids:
        fname = ARTICLE_FILES.get(aid)
        if not fname:
            raise ValueError(f"Unknown article id: {aid}. Choose from: {list(ARTICLE_FILES)}")
        path = ARTICLES_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Article not found: {path}")
        articles[aid] = path.read_text(encoding="utf-8")
        logger.info(f"Loaded {aid}: {len(articles[aid])} chars")
    return articles


def make_runner(provider: str = "minimax", model: str = "") -> InnerRunner:
    from core.llm_client import PROVIDERS
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        sys.exit(f"ERROR: Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        sys.exit(f"ERROR: {pinfo['api_key_env']} not set")
    use_model = model or pinfo["default_model"]
    configure(provider, api_key, use_model)
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return InnerRunner(model=use_model, eval_model=use_model, min_score=6, max_retries=2, artifacts_base=artifacts_dir)


def cmd_run(args):
    """Full dual-layer experiment."""
    from core.llm_client import PROVIDERS
    inner_provider = args.provider or "minimax"
    inner_model = args.model or ""
    outer_provider = args.outer_provider or "deepseek"
    outer_model = args.outer_model or ""

    outer_pinfo = PROVIDERS.get(outer_provider)
    if not outer_pinfo:
        sys.exit(f"ERROR: Unknown outer provider '{outer_provider}'. Choose from: {list(PROVIDERS)}")
    outer_key = os.environ.get(outer_pinfo["api_key_env"], "")
    if not outer_key:
        sys.exit(f"ERROR: {outer_pinfo['api_key_env']} not set")
    outer_use_model = outer_model or outer_pinfo["default_model"]

    article_ids = args.articles or list(ARTICLE_FILES.keys())
    articles = load_articles(article_ids)

    outer_state = OuterLoopState(
        base_dir=BASE_DIR,
        original_articles=articles,
    )
    if args.resume and outer_state.load_checkpoint():
        logger.info(f"Resumed from checkpoint at cycle {outer_state.current_cycle}")
    runner = make_runner(inner_provider, inner_model)
    inner_ctrl = InnerLoopController(
        runner=runner,
        max_iterations=args.max_inner,
        convergence_threshold=8,
        convergence_consecutive=3,
    )
    analyzer = OuterAnalyzer(model=outer_use_model, api_key=outer_key)
    outer_ctrl = OuterLoopController(
        outer_state=outer_state,
        inner_controller=inner_ctrl,
        analyzer=analyzer,
        article_ids=article_ids,
        max_outer_iterations=args.max_outer,
        reference_doc_path=REFERENCE_DOC,
    )

    logger.info("Starting dual-layer experiment")
    logger.info(f"Articles: {article_ids}")
    logger.info(f"Inner budget: {args.max_inner} runs/cycle | Outer budget: {args.max_outer} cycles")

    result = outer_ctrl.run()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Outer cycles run: {result['total_outer_cycles']}")
    print(f"Outer lessons accumulated: {result['total_outer_lessons']}")
    print(f"Active prompt overrides: {list(result['final_prompt_overrides'].keys())}")
    print("\nPer-cycle results:")
    for cycle in result["cycle_results"]:
        print(f"\n  Cycle {cycle['cycle']}:")
        for art in cycle["articles"]:
            runs_to_8 = art["runs_to_threshold_8"] or "never"
            print(f"    {art['article_id']}: peak={art['peak_score']}/10, "
                  f"runs_to_8={runs_to_8}, strategy={art['strategy']}")
    print(f"\nArtifacts saved to: {BASE_DIR / 'artifacts'}")
    print(f"Examples/traces:   {BASE_DIR / 'examples'}")


def cmd_inner(args):
    """Inner loop only — useful for debugging a single article."""
    inner_provider = args.provider or "minimax"
    inner_model = args.model or ""

    article_id = args.article or "article15"
    articles = load_articles([article_id])

    outer_state = OuterLoopState(base_dir=BASE_DIR, original_articles=articles)
    outer_state.begin_cycle()

    runner = make_runner(inner_provider, inner_model)
    inner_ctrl = InnerLoopController(
        runner=runner,
        max_iterations=args.max_inner,
    )

    logger.info(f"Running inner loop on {article_id} (max {args.max_inner} runs)")
    inner = inner_ctrl.run_cycle(article_id, outer_state)

    print("\nInner loop complete:")
    print(f"  Total runs: {len(inner.run_trace)}")
    print(f"  Peak score: {inner.peak_score()}/10")
    print(f"  Converged: {inner.is_converged()}")
    print(f"  Runs to 8/10: {inner.runs_to_threshold(8) or 'never'}")
    print(f"  Lessons extracted: {len(inner.inner_lessons)}")
    print(f"\n  Score trace: {[r.overall for r in inner.run_trace]}")


def cmd_mechresearch(args):
    """Level 2 mechanism research — outer LLM generates a new pipeline stage."""
    from core.llm_client import PROVIDERS

    from .mechanism_research import MechanismResearcher

    inner_provider = args.provider or "minimax"
    inner_model = args.model or ""
    outer_provider = args.outer_provider or "deepseek"
    outer_model = args.outer_model or ""

    outer_pinfo = PROVIDERS.get(outer_provider)
    if not outer_pinfo:
        sys.exit(f"ERROR: Unknown outer provider '{outer_provider}'. Choose from: {list(PROVIDERS)}")
    outer_key = os.environ.get(outer_pinfo["api_key_env"], "")
    if not outer_key:
        sys.exit(f"ERROR: {outer_pinfo['api_key_env']} not set")
    outer_use_model = outer_model or outer_pinfo["default_model"]

    article_id = args.article or "article2"
    articles = load_articles([article_id])
    article_content = articles[article_id]

    # --- Build baseline inner states ---
    # Run args.baseline_cycles inner cycles to give the researcher trace data.
    # Each cycle is a fresh InnerLoopController run on the original article.
    inner_states: list[InnerLoopState] = []
    runner = make_runner(inner_provider, inner_model)

    if args.baseline_cycles > 0:
        logger.info(f"Running {args.baseline_cycles} baseline cycle(s) to build trace data...")
        for cycle_idx in range(args.baseline_cycles):
            runner.outer_cycle = cycle_idx + 1
            outer_state = OuterLoopState(base_dir=BASE_DIR, original_articles=articles)
            outer_state.begin_cycle()
            ctrl = InnerLoopController(
                runner=runner,
                max_iterations=args.max_inner,
                convergence_threshold=8,
                convergence_consecutive=3,
            )
            state = ctrl.run_cycle(article_id, outer_state)
            inner_states.append(state)
            scores = [r.overall for r in state.run_trace]
            logger.info(f"  Cycle {cycle_idx+1}: scores={scores}, peak={state.peak_score()}")
    else:
        # No baseline: pass a single empty state so researcher has minimal context
        logger.warning("No baseline cycles — researcher will have minimal trace data.")
        inner_states = [InnerLoopState(original_article=article_content, article_id=article_id)]

    outer_lessons: list[dict] = []

    # --- Run mechanism research ---
    researcher = MechanismResearcher(
        model=outer_use_model,
        api_key=outer_key,
        provider=outer_provider,
        max_code_retries=3,
        artifacts_base=BASE_DIR / "artifacts" / "mechanism_research",
    )

    logger.info("Starting Level 2 mechanism research session...")
    result = researcher.research(article_id, inner_states, outer_lessons)

    print("\n" + "=" * 60)
    print("MECHANISM RESEARCH COMPLETE")
    print("=" * 60)
    print(f"Session ID:    {result.session_id}")
    print(f"Domain source: {result.domain_source[:80]}")
    print(f"Stage name:    {result.stage_name}")
    print(f"Inject after:  {result.inject_after}")
    print(f"Code retries:  {result.code_retries}")
    print(f"Stage file:    {result.stage_path}")

    if not args.skip_validate:
        # --- Validate: run inner loop with injected stage ---
        print("\nValidating generated stage against inner loop...")
        # Fresh runner so baseline stages are clean (no residual state)
        val_runner = make_runner(inner_provider, inner_model)
        val_runner.outer_cycle = 99  # separate artifact namespace
        validation = researcher.validate(result, article_content, val_runner, max_inner=args.max_inner)

        print("\n" + "-" * 40)
        print("VALIDATION RESULTS")
        print("-" * 40)
        print(f"Score trace:   {validation['score_trace']}")
        print(f"Peak score:    {validation['peak_score']}/10")
        print(f"Runs to 7/10:  {validation['runs_to_7'] or 'never'}")
        print(f"Runs to 8/10:  {validation['runs_to_8'] or 'never'}")
        print(f"Converged:     {validation['converged']}")
        print(f"Lessons:       {validation['lessons_extracted']}")

        # Compare against baseline if we have it
        if inner_states and inner_states[0].run_trace:
            baseline_peak = max(s.peak_score() for s in inner_states)
            delta = validation["peak_score"] - baseline_peak
            sign = "+" if delta >= 0 else ""
            print(f"\nBaseline peak: {baseline_peak}/10")
            print(f"New peak:      {validation['peak_score']}/10  ({sign}{delta})")

    print(f"\nArtifacts: {BASE_DIR / 'artifacts' / 'mechanism_research' / ('session_' + result.session_id)}")


def cmd_once(args):
    """Single pipeline pass — smoke test."""
    inner_provider = args.provider or "minimax"
    inner_model = args.model or ""

    article_id = args.article or "article15"
    articles = load_articles([article_id])

    outer_state = OuterLoopState(base_dir=BASE_DIR, original_articles=articles)
    outer_state.begin_cycle()

    runner = make_runner(inner_provider, inner_model)

    inner = InnerLoopState(
        original_article=articles[article_id],
        article_id=article_id,
    )

    logger.info(f"Running single pass on {article_id}")
    result = runner.run_once(inner)

    print("\nSingle run complete:")
    print(f"  Overall: {result.overall}/10")
    print(f"  Scores: {result.stage_map}")
    print(f"  Lessons: {len(inner.inner_lessons)}")
    print(f"  Artifacts: {BASE_DIR / 'artifacts' / 'run_001'}")


def main():
    parser = argparse.ArgumentParser(description="Dual-layer article optimizer")
    parser.add_argument("--provider", default=None,
                        help="Inner loop LLM provider (default: minimax). Options: minimax, openai, deepseek, anthropic, glm")
    parser.add_argument("--model", default=None,
                        help="Inner loop model name (default: provider's default)")
    parser.add_argument("--outer-provider", default=None,
                        help="Outer loop LLM provider (default: deepseek)")
    parser.add_argument("--outer-model", default=None,
                        help="Outer loop model name (default: provider's default)")
    sub = parser.add_subparsers(dest="cmd")

    # run
    p_run = sub.add_parser("run", help="Full dual-layer experiment")
    p_run.add_argument("--articles", nargs="+", choices=list(ARTICLE_FILES),
                       help="Articles to optimize (default: all)")
    p_run.add_argument("--max-inner", type=int, default=20, help="Max inner runs per cycle")
    p_run.add_argument("--max-outer", type=int, default=5, help="Max outer cycles")
    p_run.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint if available")

    # inner
    p_inner = sub.add_parser("inner", help="Inner loop only (one article)")
    p_inner.add_argument("--article", choices=list(ARTICLE_FILES), default="article15")
    p_inner.add_argument("--max-inner", type=int, default=20)

    # once
    p_once = sub.add_parser("once", help="Single pipeline pass (smoke test)")
    p_once.add_argument("--article", choices=list(ARTICLE_FILES), default="article15")

    # mechresearch
    p_mech = sub.add_parser(
        "mechresearch",
        help="Level 2: outer LLM researches + generates a new pipeline stage",
    )
    p_mech.add_argument("--article", choices=list(ARTICLE_FILES), default="article2",
                        help="Article to use as the research target (default: article2)")
    p_mech.add_argument("--baseline-cycles", type=int, default=2,
                        help="Inner cycles to run for baseline trace data (default: 2)")
    p_mech.add_argument("--max-inner", type=int, default=5,
                        help="Max inner runs per cycle (default: 5)")
    p_mech.add_argument("--skip-validate", action="store_true",
                        help="Skip validation run (research only, no inner loop execution)")

    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "inner":
        cmd_inner(args)
    elif args.cmd == "once":
        cmd_once(args)
    elif args.cmd == "mechresearch":
        cmd_mechresearch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
