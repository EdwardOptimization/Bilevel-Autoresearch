"""Ablation study script for the EvoResearch paper.

Three experimental groups:
  A: Level 1 only   — inner loop (simple_mode=True, no outer loop)
  B: Level 1 + 1.5  — inner loop + outer loop (simple_mode=True, WITH outer cycles)
  C: Level 1 + 1.5 + Level 2 — same as B but with DeepSeek Level-2 researcher
                                that modifies runner.py between outer cycle pairs

Usage:
  cd /home/quyaonan/research_evo_mvp
  python -m experiments.paper_ablation.run_ablation --group A --repeats 3
  python -m experiments.paper_ablation.run_ablation --group B --repeats 3
  python -m experiments.paper_ablation.run_ablation --group C --repeats 3
  python -m experiments.paper_ablation.run_ablation --group all --repeats 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import traceback as tb
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup + .env loading (must happen before any project imports)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.strip().strip('"').strip("'")
            os.environ.setdefault(_k.strip(), _v)

from core.llm_client import PROVIDERS, LLMClient
from domains.train_opt.config import SearchConfig
from domains.train_opt.outer import TrainOuterLoop
from domains.train_opt.runner import TrainRunner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_DIR = Path(__file__).parent
RESULTS_DIR = ABLATION_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters that must not be editable — prevent model size changes
FORBIDDEN_PARAMS = {"DEPTH", "ASPECT_RATIO"}

# Time budget for all groups
TIME_BUDGET = 300

# Level-2 intervention schedule for Group C:
# Level-2 fires after every 2 outer cycles (cycles 2, 4, 6, ...)
LEVEL2_CYCLE_INTERVAL = 2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_client(provider: str, model: str) -> LLMClient:
    """Create an LLM client, resolving API key from env."""
    pinfo = PROVIDERS.get(provider)
    if not pinfo:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    api_key = os.environ.get(pinfo["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(
            f"API key not set. Expected env var: {pinfo['api_key_env']}"
        )
    resolved_model = model or pinfo["default_model"]
    return LLMClient(provider, api_key, resolved_model)


def _make_search_config(iterations: int, time_budget: int) -> SearchConfig:
    """Build SearchConfig with DEPTH and ASPECT_RATIO removed from editable_params."""
    cfg = SearchConfig(inner_budget=iterations, time_budget=time_budget)
    cfg.editable_params = [
        p for p in cfg.editable_params if p not in FORBIDDEN_PARAMS
    ]
    return cfg


def _get_train_py(autoresearch_dir: Path) -> Path:
    """Return path to the baseline train.py, raising if not found."""
    train_py = autoresearch_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(
            f"train.py not found at {train_py}. "
            f"Set --autoresearch-dir to the karpathy_autoresearch directory."
        )
    return train_py


def _build_run_dir(group: str, repeat: int) -> Path:
    """Return the directory for a single run, creating it if necessary."""
    run_dir = RESULTS_DIR / f"{group}{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_run_logging(run_dir: Path) -> logging.FileHandler:
    """Attach a file handler for the run's experiment.log."""
    log_path = run_dir / "experiment.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    # Attach to root logger so all project loggers write to this file too
    logging.getLogger().addHandler(fh)
    return fh


def _teardown_run_logging(fh: logging.FileHandler) -> None:
    """Remove and close the per-run file handler."""
    logging.getLogger().removeHandler(fh)
    fh.close()


def _save_report(run_dir: Path, report: dict) -> None:
    """Save a structured report.json to the run directory."""
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _snapshot_runner(run_dir: Path, runner_py: Path) -> None:
    """Copy the runner.py used for this run to the results directory."""
    dest = run_dir / "runner.py"
    shutil.copy2(runner_py, dest)
    logger.info(f"[Snapshot] runner.py → {dest}")


def _print_group_summary(group: str, results: list[dict]) -> None:
    """Print a formatted summary table for a group after all repeats."""
    print(f"\nGroup {group} Summary:")
    improvements = []
    for r in results:
        bl = r.get("baseline_bpb", float("nan"))
        best = r.get("best_val_bpb", float("nan"))
        imp = r.get("improvement", float("nan"))
        repeat = r.get("repeat", "?")
        status = r.get("status", "ok")
        if status == "error":
            print(f"  Repeat {repeat}: ERROR — {r.get('error', '')[:80]}")
        else:
            print(f"  Repeat {repeat}: baseline={bl:.6f}  best={best:.6f}  improvement={imp:.6f}")
            if not (imp != imp):  # not NaN
                improvements.append(imp)

    if len(improvements) >= 2:
        mean = statistics.mean(improvements)
        std = statistics.stdev(improvements)
        print(f"  Mean improvement: {mean:.6f} ± {std:.6f}")
    elif len(improvements) == 1:
        print(f"  Mean improvement: {improvements[0]:.6f} (n=1)")
    else:
        print("  No successful repeats to summarize.")


# ---------------------------------------------------------------------------
# Group A — Level 1 only (inner loop, simple_mode=True, no outer loop)
# ---------------------------------------------------------------------------

def run_group_a(
    repeat: int,
    iterations: int,
    time_budget: int,
    provider: str,
    model: str,
    autoresearch_dir: Path,
) -> dict:
    """Run one repeat of Group A: inner loop only."""
    run_dir = _build_run_dir("A", repeat)
    fh = _setup_run_logging(run_dir)
    logger.info(f"=== Group A | Repeat {repeat} | {iterations} iterations ===")

    try:
        train_py = _get_train_py(autoresearch_dir)
        client = _make_llm_client(provider, model)
        config = _make_search_config(iterations, time_budget)

        runner = TrainRunner(
            train_py=train_py,
            work_dir=autoresearch_dir,
            llm_client=client,
            search_config=config,
            artifacts_dir=run_dir / "artifacts",
            simple_mode=True,
        )

        # Baseline
        baseline_result = runner.run_baseline()
        logger.info(f"[A{repeat}] Baseline: {baseline_result.val_bpb:.6f}")

        # Run inner loop iterations
        for i in range(1, iterations + 1):
            result = runner.run_iteration(i)
            logger.info(
                f"[A{repeat}] Iter {i:3d}/{iterations}: bpb={result.val_bpb:.6f} "
                f"[{result.status}] {result.description[:55]}"
            )

        trace = runner.trace
        report = {
            "group": "A",
            "repeat": repeat,
            "status": "ok",
            "baseline_bpb": baseline_result.val_bpb,
            "best_val_bpb": trace.best_bpb,
            "best_iteration": trace.best_iteration,
            "improvement": baseline_result.val_bpb - trace.best_bpb,
            "total_iterations": len(trace.results),
            "keeps": sum(1 for r in trace.results if r.status == "keep"),
            "discards": sum(1 for r in trace.results if r.status == "discard"),
            "crashes": sum(1 for r in trace.results if r.status == "crash"),
            "trace": [
                {"iter": r.iteration, "bpb": r.val_bpb, "status": r.status,
                 "desc": r.description}
                for r in trace.results
            ],
        }
        _save_report(run_dir, report)
        _snapshot_runner(run_dir, Path(__file__).parent.parent.parent / "domains" / "train_opt" / "runner.py")

        logger.info(
            f"[A{repeat}] Done. baseline={baseline_result.val_bpb:.6f}  "
            f"best={trace.best_bpb:.6f}  improvement={report['improvement']:.6f}"
        )
        return report

    except Exception as exc:
        err_text = tb.format_exc()
        logger.error(f"[A{repeat}] FAILED:\n{err_text}")
        error_report = {
            "group": "A", "repeat": repeat, "status": "error",
            "error": str(exc), "traceback": err_text,
        }
        _save_report(run_dir, error_report)
        return error_report

    finally:
        _teardown_run_logging(fh)


# ---------------------------------------------------------------------------
# Group B — Level 1 + Level 1.5 (inner loop + outer loop)
# ---------------------------------------------------------------------------

def run_group_b(
    repeat: int,
    iterations: int,
    outer_cycles: int,
    time_budget: int,
    provider: str,
    model: str,
    autoresearch_dir: Path,
) -> dict:
    """Run one repeat of Group B: inner + outer loop."""
    run_dir = _build_run_dir("B", repeat)
    fh = _setup_run_logging(run_dir)

    inner_per_cycle = max(1, iterations // outer_cycles)
    logger.info(
        f"=== Group B | Repeat {repeat} | {outer_cycles} outer cycles × "
        f"{inner_per_cycle} inner iters/cycle ==="
    )

    try:
        train_py = _get_train_py(autoresearch_dir)
        client = _make_llm_client(provider, model)
        config = _make_search_config(inner_per_cycle, time_budget)

        runner = TrainRunner(
            train_py=train_py,
            work_dir=autoresearch_dir,
            llm_client=client,
            search_config=config,
            artifacts_dir=run_dir / "artifacts",
            simple_mode=True,
        )

        outer = TrainOuterLoop(
            runner=runner,
            llm_client=client,       # same DeepSeek client for all levels
            max_outer_cycles=outer_cycles,
            artifacts_dir=run_dir / "artifacts" / "outer",
        )

        report_raw = outer.run()

        report = {
            "group": "B",
            "repeat": repeat,
            "status": "ok",
            "baseline_bpb": report_raw.get("baseline_bpb"),
            "best_val_bpb": report_raw.get("best_val_bpb"),
            "best_iteration": report_raw.get("best_iteration"),
            "improvement": report_raw.get("improvement", 0.0),
            "total_iterations": report_raw.get("total_iterations"),
            "outer_cycles": report_raw.get("outer_cycles"),
            "keeps": report_raw.get("keeps"),
            "discards": report_raw.get("discards"),
            "crashes": report_raw.get("crashes"),
            "trace": report_raw.get("trace", []),
            "outer_trace": report_raw.get("outer_trace", []),
        }
        _save_report(run_dir, report)
        _snapshot_runner(run_dir, Path(__file__).parent.parent.parent / "domains" / "train_opt" / "runner.py")

        logger.info(
            f"[B{repeat}] Done. baseline={report['baseline_bpb']:.6f}  "
            f"best={report['best_val_bpb']:.6f}  improvement={report['improvement']:.6f}"
        )
        return report

    except Exception as exc:
        err_text = tb.format_exc()
        logger.error(f"[B{repeat}] FAILED:\n{err_text}")
        error_report = {
            "group": "B", "repeat": repeat, "status": "error",
            "error": str(exc), "traceback": err_text,
        }
        _save_report(run_dir, error_report)
        return error_report

    finally:
        _teardown_run_logging(fh)


# ---------------------------------------------------------------------------
# Group C — Level 1 + Level 1.5 + Level 2 (with mechanism researcher)
# ---------------------------------------------------------------------------

def run_group_c(
    repeat: int,
    iterations: int,
    outer_cycles: int,
    time_budget: int,
    provider: str,
    model: str,
    autoresearch_dir: Path,
) -> dict:
    """Run one repeat of Group C: inner + outer + Level-2 mechanism researcher.

    Level-2 fires every LEVEL2_CYCLE_INTERVAL outer cycles:
      Cycles 1-2: run → Level-2 analyzes and patches runner.py
      Cycles 3-4: run with patched runner → Level-2 patches again
      Cycles 5-6: run with latest patched runner
    """
    run_dir = _build_run_dir("C", repeat)
    mech_dir = run_dir / "mechanism_sessions"
    mech_dir.mkdir(parents=True, exist_ok=True)
    fh = _setup_run_logging(run_dir)

    inner_per_cycle = max(1, iterations // outer_cycles)
    logger.info(
        f"=== Group C | Repeat {repeat} | {outer_cycles} outer cycles × "
        f"{inner_per_cycle} inner iters/cycle | Level-2 every {LEVEL2_CYCLE_INTERVAL} cycles ==="
    )

    try:
        from domains.train_opt.mechanism_research import TrainMechanismResearcher

        train_py = _get_train_py(autoresearch_dir)
        client = _make_llm_client(provider, model)

        # The canonical runner.py source (never overwritten by Level-2)
        canonical_runner_py = PROJECT_ROOT / "domains" / "train_opt" / "runner.py"

        # Working copy of runner.py for this run — Level-2 patches THIS copy
        run_runner_py = run_dir / "runner.py"
        shutil.copy2(canonical_runner_py, run_runner_py)
        logger.info(f"[C{repeat}] Copied baseline runner.py to {run_runner_py}")

        # Level-2 researcher (uses the same LLM client as all other levels)
        researcher = TrainMechanismResearcher(
            model=client.model,
            api_key=client.api_key,
            provider=provider,
        )

        # Results accumulator
        all_trace: list[dict] = []
        outer_trace_all: list[dict] = []
        baseline_bpb: float | None = None
        best_bpb_overall: float = float("inf")
        best_iter_overall: int = 0
        level2_sessions: list[dict] = []

        # Run outer cycles in pairs, applying Level-2 between each pair
        completed_cycles = 0
        level2_round = 0

        while completed_cycles < outer_cycles:
            # How many cycles in this batch?
            remaining = outer_cycles - completed_cycles
            batch_size = min(LEVEL2_CYCLE_INTERVAL, remaining)

            batch_start = completed_cycles + 1
            batch_end = completed_cycles + batch_size
            logger.info(
                f"[C{repeat}] Starting cycle batch {batch_start}–{batch_end} "
                f"(using runner.py from {run_runner_py})"
            )

            # Dynamically load the (possibly patched) TrainRunner from run_runner_py
            runner_module = _load_runner_module(run_runner_py)
            PatchedTrainRunner = runner_module.TrainRunner

            config = _make_search_config(inner_per_cycle, time_budget)

            runner = PatchedTrainRunner(
                train_py=train_py,
                work_dir=autoresearch_dir,
                llm_client=client,
                search_config=config,
                artifacts_dir=run_dir / "artifacts" / f"batch_{batch_start}_{batch_end}",
                simple_mode=True,
            )

            # Run this batch of outer cycles
            outer = TrainOuterLoop(
                runner=runner,
                llm_client=client,
                max_outer_cycles=batch_size,
                artifacts_dir=run_dir / "artifacts" / f"batch_{batch_start}_{batch_end}" / "outer",
            )

            batch_report = outer.run()

            # Accumulate results
            if baseline_bpb is None:
                baseline_bpb = batch_report.get("baseline_bpb")

            batch_best = batch_report.get("best_val_bpb", float("inf"))
            if batch_best < best_bpb_overall:
                best_bpb_overall = batch_best
                best_iter_overall = batch_report.get("best_iteration", 0)

            all_trace.extend(batch_report.get("trace", []))
            outer_trace_all.extend(batch_report.get("outer_trace", []))
            completed_cycles += batch_size

            # Snapshot runner.py after each batch
            snapshot_path = run_dir / f"runner_after_cycles_{batch_start}_{batch_end}.py"
            shutil.copy2(run_runner_py, snapshot_path)

            # --- Level-2 intervention ---
            # Fire after every LEVEL2_CYCLE_INTERVAL cycles, but not after the last batch
            if completed_cycles < outer_cycles:
                level2_round += 1
                logger.info(
                    f"[C{repeat}] Level-2 Round {level2_round} — "
                    f"analyzing trace and patching runner.py..."
                )

                trace_text = runner.trace.summary(last_n=inner_per_cycle * batch_size + 5)
                runner_code = run_runner_py.read_text(encoding="utf-8")

                session_subdir = mech_dir / f"round_{level2_round}"
                session_subdir.mkdir(parents=True, exist_ok=True)

                try:
                    result = researcher.research(
                        trace_summary=trace_text,
                        runner_code=runner_code,
                        session_dir=session_subdir,
                        bottleneck="",  # auto-inferred from trace
                    )

                    applied = researcher.apply(run_runner_py, result)
                    if applied:
                        # Validate import
                        valid = researcher.validate(run_runner_py)
                        if not valid:
                            logger.warning(
                                f"[C{repeat}] Level-2 Round {level2_round}: "
                                f"patched runner.py fails import check — reverting"
                            )
                            # Restore pre-patch backup (researcher.apply made a .bak)
                            backup = run_runner_py.with_suffix(
                                f".py.bak_{result.session_id}"
                            )
                            if backup.exists():
                                shutil.copy2(backup, run_runner_py)
                                logger.info(f"[C{repeat}] Restored from {backup}")
                            applied = False

                    level2_sessions.append({
                        "round": level2_round,
                        "session_id": result.session_id,
                        "mechanism_name": result.mechanism_name,
                        "strategy": result.implementation_strategy,
                        "target": result.target,
                        "code_retries": result.code_retries,
                        "applied": applied,
                        "validated": result.validated,
                        "error": result.validation_error,
                    })

                    if applied:
                        logger.info(
                            f"[C{repeat}] Level-2 Round {level2_round}: "
                            f"patched runner.py with '{result.mechanism_name}'"
                        )
                    else:
                        logger.warning(
                            f"[C{repeat}] Level-2 Round {level2_round}: "
                            f"patch NOT applied — continuing with unchanged runner.py"
                        )

                except Exception as exc:
                    err_text = tb.format_exc()
                    logger.error(
                        f"[C{repeat}] Level-2 Round {level2_round} FAILED:\n{err_text}"
                    )
                    level2_sessions.append({
                        "round": level2_round,
                        "status": "error",
                        "error": str(exc),
                        "applied": False,
                    })

        # Snapshot final runner.py for the run
        shutil.copy2(run_runner_py, run_dir / "runner.py")

        improvement = (baseline_bpb - best_bpb_overall) if baseline_bpb is not None else 0.0
        report = {
            "group": "C",
            "repeat": repeat,
            "status": "ok",
            "baseline_bpb": baseline_bpb,
            "best_val_bpb": best_bpb_overall,
            "best_iteration": best_iter_overall,
            "improvement": improvement,
            "total_iterations": len(all_trace),
            "outer_cycles": outer_cycles,
            "level2_rounds": level2_round,
            "level2_sessions": level2_sessions,
            "trace": all_trace,
            "outer_trace": outer_trace_all,
        }
        _save_report(run_dir, report)

        logger.info(
            f"[C{repeat}] Done. baseline={baseline_bpb:.6f}  "
            f"best={best_bpb_overall:.6f}  improvement={improvement:.6f}  "
            f"level2_rounds={level2_round}"
        )
        return report

    except Exception as exc:
        err_text = tb.format_exc()
        logger.error(f"[C{repeat}] FAILED:\n{err_text}")
        error_report = {
            "group": "C", "repeat": repeat, "status": "error",
            "error": str(exc), "traceback": err_text,
        }
        _save_report(run_dir, error_report)
        return error_report

    finally:
        _teardown_run_logging(fh)


# ---------------------------------------------------------------------------
# Dynamic runner module loader (for Group C patched runners)
# ---------------------------------------------------------------------------

def _load_runner_module(runner_py: Path):
    """Dynamically import a runner.py from an arbitrary path.

    Returns the module so callers can access module.TrainRunner.
    Each call creates a fresh module instance to avoid cache contamination.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"_ablation_runner_{runner_py.stat().st_mtime_ns}", runner_py
    )
    module = importlib.util.module_from_spec(spec)

    # Pre-populate sys.modules with project root so relative imports work
    sys.path_hooks  # ensure sys.path is active
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study: Groups A (L1), B (L1+L1.5), C (L1+L1.5+L2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--group",
        default="all",
        choices=["A", "B", "C", "all"],
        help="Which ablation group to run",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of independent repeats per group",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Total inner iterations per repeat (Group A) or total across all outer cycles (B/C)",
    )
    parser.add_argument(
        "--outer-cycles",
        type=int,
        default=6,
        help="Number of outer cycles for Groups B and C (inner_per_cycle = iterations/outer_cycles)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=TIME_BUDGET,
        help="Training time budget per individual run in seconds",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        help="LLM provider used for ALL levels (inner, outer, Level-2)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model override (empty = provider default)",
    )
    parser.add_argument(
        "--autoresearch-dir",
        type=Path,
        default=Path.home() / "karpathy_autoresearch",
        help="Path to karpathy_autoresearch directory (must contain train.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    groups_to_run: list[str] = (
        ["A", "B", "C"] if args.group == "all" else [args.group]
    )

    logger.info(
        f"Ablation study starting: groups={groups_to_run}  repeats={args.repeats}  "
        f"iterations={args.iterations}  outer_cycles={args.outer_cycles}  "
        f"time_budget={args.time_budget}s  provider={args.provider}  "
        f"model={args.model or '(provider default)'}  "
        f"autoresearch_dir={args.autoresearch_dir}"
    )

    all_group_results: dict[str, list[dict]] = {}

    for group in groups_to_run:
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Group {group} ({args.repeats} repeats)")
        logger.info(f"{'='*70}")

        group_results: list[dict] = []

        for repeat in range(1, args.repeats + 1):
            logger.info(f"\n--- Group {group} | Repeat {repeat}/{args.repeats} ---")

            try:
                if group == "A":
                    result = run_group_a(
                        repeat=repeat,
                        iterations=args.iterations,
                        time_budget=args.time_budget,
                        provider=args.provider,
                        model=args.model,
                        autoresearch_dir=args.autoresearch_dir,
                    )
                elif group == "B":
                    result = run_group_b(
                        repeat=repeat,
                        iterations=args.iterations,
                        outer_cycles=args.outer_cycles,
                        time_budget=args.time_budget,
                        provider=args.provider,
                        model=args.model,
                        autoresearch_dir=args.autoresearch_dir,
                    )
                elif group == "C":
                    result = run_group_c(
                        repeat=repeat,
                        iterations=args.iterations,
                        outer_cycles=args.outer_cycles,
                        time_budget=args.time_budget,
                        provider=args.provider,
                        model=args.model,
                        autoresearch_dir=args.autoresearch_dir,
                    )
                else:
                    raise ValueError(f"Unknown group: {group}")

            except Exception as exc:
                # Outer safety net — individual run functions already catch internally,
                # but be extra safe here
                err_text = tb.format_exc()
                logger.error(
                    f"Unexpected error in Group {group} Repeat {repeat}:\n{err_text}"
                )
                result = {
                    "group": group, "repeat": repeat, "status": "error",
                    "error": str(exc), "traceback": err_text,
                }

            group_results.append(result)

        all_group_results[group] = group_results
        _print_group_summary(group, group_results)

    # Final cross-group summary
    if len(groups_to_run) > 1:
        print(f"\n{'='*70}")
        print("All Groups — Final Summary")
        print(f"{'='*70}")
        for grp in groups_to_run:
            _print_group_summary(grp, all_group_results[grp])

    logger.info("Ablation study complete.")


if __name__ == "__main__":
    main()
