"""CLI entry point for training optimization experiments.

Usage:
  # Full bilevel experiment (Level 1 + Level 1.5):
  python -m domains.train_opt.cli bilevel --inner-budget 5 --outer-cycles 3

  # Inner loop only (Level 1, like Karpathy's autoresearch):
  python -m domains.train_opt.cli inner --iterations 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

from core.llm_client import LLMClient

from .config import SearchConfig
from .outer import TrainOuterLoop
from .runner import TrainRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Karpathy's autoresearch directory
AUTORESEARCH_DIR = Path(os.environ.get(
    "AUTORESEARCH_DIR",
    str(Path(__file__).parent.parent.parent.parent / "karpathy_autoresearch"),
))


def get_llm_client(provider: str = "deepseek", model: str = "") -> LLMClient:
    """Create LLM client from environment."""
    from core.llm_client import PROVIDERS
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        sys.exit(f"ERROR: Unknown provider '{provider}'")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        sys.exit(f"ERROR: {pinfo['api_key_env']} not set")
    return LLMClient(provider, api_key, model or pinfo["default_model"])


def cmd_inner(args):
    """Inner loop only — equivalent to Karpathy's autoresearch (Level 1)."""
    client = get_llm_client(args.provider, args.model)

    train_py = AUTORESEARCH_DIR / "train.py"
    if not train_py.exists():
        sys.exit(f"ERROR: train.py not found at {train_py}")

    config = SearchConfig(inner_budget=args.iterations, time_budget=args.time_budget)
    runner = TrainRunner(
        train_py=train_py,
        work_dir=AUTORESEARCH_DIR,
        llm_client=client,
        search_config=config,
        artifacts_dir=PROJECT_ROOT / "artifacts" / "train_opt" / "inner_only",
    )

    # Baseline
    baseline = runner.run_baseline()

    # Run inner iterations
    for i in range(args.iterations):
        result = runner.run_iteration(i + 1)
        logger.info(
            f"[Iter {i+1}/{args.iterations}] bpb={result.val_bpb:.6f} "
            f"[{result.status}] {result.description[:60]}"
        )

    # Report
    trace = runner.trace
    print("\n" + "=" * 60)
    print("INNER LOOP COMPLETE (Level 1)")
    print("=" * 60)
    print(f"Baseline:     {baseline.val_bpb:.6f}")
    print(f"Best:         {trace.best_bpb:.6f} (iter {trace.best_iteration})")
    print(f"Improvement:  {baseline.val_bpb - trace.best_bpb:.6f}")
    print(f"Keeps:        {sum(1 for r in trace.results if r.status == 'keep')}")
    print(f"Discards:     {sum(1 for r in trace.results if r.status == 'discard')}")
    print(f"Crashes:      {sum(1 for r in trace.results if r.status == 'crash')}")
    print("\nTrace:")
    for r in trace.results:
        print(f"  {r.iteration:3d}: {r.val_bpb:.6f} [{r.status:7s}] {r.description[:50]}")


def cmd_bilevel(args):
    """Full bilevel experiment (Level 1 + Level 1.5)."""
    inner_client = get_llm_client(args.provider, args.model)
    outer_client = get_llm_client(args.outer_provider, args.outer_model)

    train_py = AUTORESEARCH_DIR / "train.py"
    if not train_py.exists():
        sys.exit(f"ERROR: train.py not found at {train_py}")

    config = SearchConfig(inner_budget=args.inner_budget, time_budget=args.time_budget)
    runner = TrainRunner(
        train_py=train_py,
        work_dir=AUTORESEARCH_DIR,
        llm_client=inner_client,
        search_config=config,
        artifacts_dir=PROJECT_ROOT / "artifacts" / "train_opt" / "bilevel",
    )

    outer = TrainOuterLoop(
        runner=runner,
        llm_client=outer_client,
        max_outer_cycles=args.outer_cycles,
    )

    report = outer.run()

    # Print report
    print("\n" + "=" * 60)
    print("BILEVEL EXPERIMENT COMPLETE (Level 1 + Level 1.5)")
    print("=" * 60)
    print(f"Baseline:      {report['baseline_bpb']:.6f}")
    print(f"Best:          {report['best_val_bpb']:.6f} (iter {report['best_iteration']})")
    print(f"Improvement:   {report['improvement']:.6f}")
    print(f"Total iters:   {report['total_iterations']}")
    print(f"Outer cycles:  {report['outer_cycles']}")
    print(f"Keeps/Discards/Crashes: {report['keeps']}/{report['discards']}/{report['crashes']}")

    print("\nOuter cycle decisions:")
    for ot in report["outer_trace"]:
        print(f"  Cycle {ot['cycle']+1}: {ot['analysis'].get('diagnosis', '')[:80]}")
        print(f"    → strategy={ot['config_after']['strategy']}, frozen={ot['config_after']['frozen']}")

    print("\nFull trace:")
    for r in report["trace"]:
        print(f"  {r['iter']:3d}: {r['bpb']:.6f} [{r['status']:7s}] {r['desc'][:50]}")

    # Save report
    report_path = runner.artifacts_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Training optimization with bilevel autoresearch")
    parser.add_argument("--provider", default="deepseek", help="LLM provider (default: deepseek)")
    parser.add_argument("--model", default="", help="LLM model")
    parser.add_argument("--outer-provider", default="deepseek", help="Outer loop LLM provider")
    parser.add_argument("--outer-model", default="", help="Outer loop LLM model")
    parser.add_argument("--time-budget", type=int, default=300, help="Training time per run in seconds")

    sub = parser.add_subparsers(dest="cmd")

    p_inner = sub.add_parser("inner", help="Inner loop only (Level 1)")
    p_inner.add_argument("--iterations", type=int, default=10, help="Number of inner iterations")

    p_bilevel = sub.add_parser("bilevel", help="Full bilevel experiment (Level 1 + 1.5)")
    p_bilevel.add_argument("--inner-budget", type=int, default=5, help="Inner iterations per outer cycle")
    p_bilevel.add_argument("--outer-cycles", type=int, default=3, help="Number of outer cycles")

    args = parser.parse_args()
    if args.cmd == "inner":
        cmd_inner(args)
    elif args.cmd == "bilevel":
        cmd_bilevel(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
