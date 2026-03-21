"""CLI entry point for the dual-layer article optimizer.

Usage:
  # Run full dual-layer experiment (all articles, up to 5 outer cycles):
  python article_optimizer/cli.py run

  # Run inner loop only on one article (debug/test):
  python article_optimizer/cli.py inner --article article15

  # Run a single pipeline pass on an article (smoke test):
  python article_optimizer/cli.py once --article article1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load project .env before anything else
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Add root to path for shared src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.llm_client import configure

from article_optimizer.src.inner_loop import InnerLoopController
from article_optimizer.src.outer_loop import OuterAnalyzer, OuterLoopController
from article_optimizer.src.runner import InnerRunner
from article_optimizer.src.state import OuterLoopState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
ARTICLES_DIR = BASE_DIR / "articles"
REFERENCE_DOC = BASE_DIR.parent / "docs" / "reference_frameworks.md"

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


def make_runner(minimax_key: str, model: str = "MiniMax-M2.7-highspeed") -> InnerRunner:
    configure("minimax", minimax_key, model)
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return InnerRunner(
        model=model,
        eval_model=model,
        min_score=6,
        max_retries=2,
        artifacts_base=artifacts_dir,
    )


def cmd_run(args):
    """Full dual-layer experiment."""
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not minimax_key:
        sys.exit("ERROR: MINIMAX_API_KEY not set")
    if not deepseek_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set")

    article_ids = args.articles or list(ARTICLE_FILES.keys())
    articles = load_articles(article_ids)

    outer_state = OuterLoopState(
        base_dir=BASE_DIR,
        original_articles=articles,
    )
    runner = make_runner(minimax_key)
    inner_ctrl = InnerLoopController(
        runner=runner,
        max_iterations=args.max_inner,
        convergence_threshold=8,
        convergence_consecutive=3,
    )
    analyzer = OuterAnalyzer(model="deepseek-chat", api_key=deepseek_key)
    outer_ctrl = OuterLoopController(
        outer_state=outer_state,
        inner_controller=inner_ctrl,
        analyzer=analyzer,
        article_ids=article_ids,
        max_outer_iterations=args.max_outer,
        reference_doc_path=REFERENCE_DOC,
    )

    logger.info(f"Starting dual-layer experiment")
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
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    if not minimax_key:
        sys.exit("ERROR: MINIMAX_API_KEY not set")

    article_id = args.article or "article15"
    articles = load_articles([article_id])

    outer_state = OuterLoopState(base_dir=BASE_DIR, original_articles=articles)
    outer_state.begin_cycle()

    runner = make_runner(minimax_key)
    inner_ctrl = InnerLoopController(
        runner=runner,
        max_iterations=args.max_inner,
    )

    logger.info(f"Running inner loop on {article_id} (max {args.max_inner} runs)")
    inner = inner_ctrl.run_cycle(article_id, outer_state)

    print(f"\nInner loop complete:")
    print(f"  Total runs: {len(inner.run_trace)}")
    print(f"  Peak score: {inner.peak_score()}/10")
    print(f"  Converged: {inner.is_converged()}")
    print(f"  Runs to 8/10: {inner.runs_to_threshold(8) or 'never'}")
    print(f"  Lessons extracted: {len(inner.inner_lessons)}")
    print(f"\n  Score trace: {[r.overall for r in inner.run_trace]}")


def cmd_once(args):
    """Single pipeline pass — smoke test."""
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    if not minimax_key:
        sys.exit("ERROR: MINIMAX_API_KEY not set")

    article_id = args.article or "article15"
    articles = load_articles([article_id])

    outer_state = OuterLoopState(base_dir=BASE_DIR, original_articles=articles)
    outer_state.begin_cycle()

    runner = make_runner(minimax_key)

    from article_optimizer.src.state import InnerLoopState
    inner = InnerLoopState(
        original_article=articles[article_id],
        article_id=article_id,
    )

    logger.info(f"Running single pass on {article_id}")
    result = runner.run_once(inner)

    print(f"\nSingle run complete:")
    print(f"  Overall: {result.overall}/10")
    print(f"  Scores: {result.stage_map}")
    print(f"  Lessons: {len(inner.inner_lessons)}")
    print(f"  Artifacts: {BASE_DIR / 'artifacts' / 'run_001'}")


def main():
    parser = argparse.ArgumentParser(description="Dual-layer article optimizer")
    sub = parser.add_subparsers(dest="cmd")

    # run
    p_run = sub.add_parser("run", help="Full dual-layer experiment")
    p_run.add_argument("--articles", nargs="+", choices=list(ARTICLE_FILES),
                       help="Articles to optimize (default: all)")
    p_run.add_argument("--max-inner", type=int, default=20, help="Max inner runs per cycle")
    p_run.add_argument("--max-outer", type=int, default=5, help="Max outer cycles")

    # inner
    p_inner = sub.add_parser("inner", help="Inner loop only (one article)")
    p_inner.add_argument("--article", choices=list(ARTICLE_FILES), default="article15")
    p_inner.add_argument("--max-inner", type=int, default=20)

    # once
    p_once = sub.add_parser("once", help="Single pipeline pass (smoke test)")
    p_once.add_argument("--article", choices=list(ARTICLE_FILES), default="article15")

    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "inner":
        cmd_inner(args)
    elif args.cmd == "once":
        cmd_once(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
