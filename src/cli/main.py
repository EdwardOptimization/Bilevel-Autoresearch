"""EvoResearch CLI — research-evo command."""
import json
import sys
import webbrowser
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config() -> dict:
    """Load config from default.yaml, then merge local.yaml overrides."""
    cfg = {}
    for fname in ["default.yaml", "local.yaml"]:
        p = PROJECT_ROOT / "config" / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _deep_merge(cfg, data)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


@click.group()
@click.version_option("0.2.0", prog_name="research-evo")
def cli():
    """EvoResearch — A self-improving AI research pipeline.

    Each run learns from the last. Lessons are stored in memory and injected
    into the next run to improve hypothesis quality, experimental rigor, and
    writeup completeness.
    """
    pass


@cli.command()
@click.argument("topic")
@click.option("--provider", default=None, help="LLM provider (deepseek/openai/glm/minimax/anthropic)")
@click.option("--model", default=None, help="Model name override")
@click.option("--show-lessons", is_flag=True, help="Show retrieved lessons before running")
@click.option("--no-quality-gate", is_flag=True, help="Disable quality gate auto-retry")
def run(topic: str, provider, model, show_lessons: bool, no_quality_gate: bool):
    """Run the full research pipeline on TOPIC."""
    config = load_config()

    if provider:
        config.setdefault("provider", {})["name"] = provider
    if model:
        config.setdefault("provider", {})["model"] = model
    if no_quality_gate:
        config.setdefault("quality_gate", {})["enabled"] = False

    from ..orchestrator.run_manager import RunManager
    manager = RunManager(config, PROJECT_ROOT)

    if show_lessons:
        all_lessons = manager.memory_store.load_all_lessons()
        tags = manager._extract_tags(topic)
        relevant = manager.retriever.retrieve(all_lessons, topic_tags=tags, max_results=5)
        from ..evolution.injection import format_lessons_for_injection
        if relevant:
            console.print(Panel(
                format_lessons_for_injection(relevant),
                title="[bold cyan]Retrieved Lessons from Memory[/bold cyan]",
                border_style="cyan",
            ))
        else:
            console.print("[dim]No relevant lessons in memory yet.[/dim]\n")

    from ..llm_client import get_provider_info
    info = get_provider_info()
    console.print(Panel(
        f"[bold]{topic}[/bold]\n\n"
        f"[dim]Provider: {info['provider']} · Model: {info['model']}[/dim]",
        title="[bold green]Starting Research Run[/bold green]",
        border_style="green",
    ))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("Initializing...", total=None)

        def on_progress(stage, event, data):
            labels = {
                "literature_scan": "Stage A: Literature Scan",
                "hypothesis_generation": "Stage B: Hypothesis Generation",
                "experiment_plan_or_code": "Stage C: Experiment Plan",
                "experiment_result_summary": "Stage D: Result Summary",
                "draft_writeup": "Stage E: Draft Writeup",
                "run_review": "Run Review",
                "lesson_extraction": "Lesson Extraction",
            }
            label = labels.get(stage, stage)
            if event == "start":
                progress.update(task, description=f"[cyan]{label}[/cyan]...")
            elif event == "evaluating":
                atmp = data.get("attempt", 1)
                progress.update(task, description=f"[yellow]Evaluating {label}[/yellow]{'  (attempt '+str(atmp)+')' if atmp > 1 else ''}...")
            elif event == "retrying":
                progress.print(f"  ↺ {label}: [yellow]score {data.get('score')}/10 — retrying with evaluator feedback[/yellow]")
            elif event == "done":
                if "verdict" in data:
                    vc = {"pass": "green", "weak": "yellow", "fail": "red"}.get(data.get("verdict", ""), "white")
                    retry_tag = " [dim](retried)[/dim]" if data.get("retried") else ""
                    progress.print(
                        f"  ✓ {label}: [{vc}]{data.get('verdict', '?')}[/{vc}] "
                        f"({data.get('score', '?')}/10){retry_tag}"
                    )
                elif "count" in data:
                    progress.print(f"  ✓ {label}: [green]{data['count']} lessons extracted[/green]")

        result = manager.start_run(topic, on_progress=on_progress)

    rv = result["run_review"]
    vc = {"pass": "green", "weak": "yellow", "fail": "red"}.get(rv.get("overall_verdict", ""), "white")
    console.print()
    console.print(Panel(
        f"[bold]Run ID[/bold]: {result['run_id']}\n"
        f"[bold]Verdict[/bold]: [{vc}]{rv.get('overall_verdict', 'N/A')}[/{vc}] ({rv.get('score', '?')}/10)\n"
        f"[bold]Lessons extracted[/bold]: {result['lessons_extracted']}\n"
        f"[bold]Artifacts[/bold]: {result['run_dir']}\n"
        f"[bold]Dashboard[/bold]: [link]http://localhost:8080/run/{result['run_id']}[/link]",
        title="[bold green]Run Complete[/bold green]",
        border_style="green",
    ))
    console.print(f"\n[bold]Summary:[/bold] {rv.get('summary', '')}")
    if rv.get("recommendations"):
        console.print("\n[bold]Recommendations for next run:[/bold]")
        for rec in rv["recommendations"][:4]:
            console.print(f"  • {rec}")


@cli.command("serve")
@click.option("--port", default=8080, help="Port to serve on")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def serve(port: int, no_browser: bool):
    """Start the web dashboard."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")
        sys.exit(1)

    url = f"http://localhost:{port}"
    console.print(Panel(
        f"Dashboard running at [bold cyan]{url}[/bold cyan]\n"
        f"Press [bold]Ctrl+C[/bold] to stop.",
        title="[bold green]EvoResearch Dashboard[/bold green]",
        border_style="green",
    ))
    if not no_browser:
        webbrowser.open(url)

    uvicorn.run(
        "src.dashboard.app:app",
        host="0.0.0.0",
        port=port,
        app_dir=str(PROJECT_ROOT),
        log_level="warning",
    )


@cli.command("list-runs")
@click.option("--limit", default=20, help="Max runs to show")
def list_runs(limit: int):
    """List all runs."""
    config = load_config()
    artifacts_dir = PROJECT_ROOT / config["paths"]["artifacts_dir"]
    if not artifacts_dir.exists():
        console.print("[dim]No runs found.[/dim]")
        return

    runs = sorted(artifacts_dir.iterdir(), reverse=True)[:limit]
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Research Runs", show_header=True, border_style="slate_blue1")
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Topic", max_width=50)
    table.add_column("Verdict", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Lessons", justify="center")
    table.add_column("Model", style="dim")

    for run_dir in runs:
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        review = meta.get("run_review", {})
        verdict = review.get("overall_verdict", "-")
        vc = {"pass": "green", "weak": "yellow", "fail": "red"}.get(verdict, "white")
        table.add_row(
            meta.get("run_id", run_dir.name)[-13:],
            meta.get("topic", "-")[:50],
            f"[{vc}]{verdict}[/{vc}]",
            str(review.get("score", "-")),
            str(meta.get("lessons_extracted", "-")),
            meta.get("model", "-"),
        )
    console.print(table)


@cli.command("compare")
@click.argument("run_id_1")
@click.argument("run_id_2")
def compare(run_id_1: str, run_id_2: str):
    """Compare two runs side by side."""
    config = load_config()
    artifacts_dir = PROJECT_ROOT / config["paths"]["artifacts_dir"]

    def load(run_id):
        # Support short IDs (last 6 chars)
        for d in artifacts_dir.iterdir():
            if d.name == run_id or d.name.endswith(run_id):
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    r1 = load(run_id_1)
    r2 = load(run_id_2)
    if not r1 or not r2:
        console.print("[red]One or both run IDs not found.[/red]")
        sys.exit(1)

    table = Table(title="Run Comparison", show_header=True, border_style="slate_blue1")
    table.add_column("Metric", style="dim")
    table.add_column(r1.get("run_id", "")[-13:], justify="center")
    table.add_column(r2.get("run_id", "")[-13:], justify="center")

    def _fmt_verdict(meta):
        v = meta.get("run_review", {}).get("overall_verdict", "-")
        vc = {"pass": "green", "weak": "yellow", "fail": "red"}.get(v, "white")
        return f"[{vc}]{v}[/{vc}]"

    def _fmt_score(meta):
        s = meta.get("run_review", {}).get("score", "-")
        return str(s)

    table.add_row("Verdict", _fmt_verdict(r1), _fmt_verdict(r2))
    table.add_row("Score", _fmt_score(r1), _fmt_score(r2))
    table.add_row("Lessons extracted",
                  str(r1.get("lessons_extracted", "-")),
                  str(r2.get("lessons_extracted", "-")))
    table.add_row("Model", r1.get("model", "-"), r2.get("model", "-"))

    for stage in ["literature_scan", "hypothesis_generation", "experiment_plan_or_code",
                  "experiment_result_summary", "draft_writeup"]:
        s1 = r1.get("stages", {}).get(stage, {})
        s2 = r2.get("stages", {}).get(stage, {})
        def _fmt_stage(s):
            v = s.get("verdict", "-")
            sc = s.get("score", "-")
            vc = {"pass": "green", "weak": "yellow", "fail": "red"}.get(v, "white")
            return f"[{vc}]{v}[/{vc}] ({sc}/10)"
        table.add_row(stage[:25], _fmt_stage(s1), _fmt_stage(s2))

    console.print(table)


@cli.command("list-lessons")
@click.option("--stage", default=None, help="Filter by stage")
@click.option("--type", "lesson_type", default=None, help="Filter by lesson type")
@click.option("--min-confidence", default=0.0, type=float)
@click.option("--limit", default=50)
def list_lessons(stage, lesson_type, min_confidence, limit):
    """List all lessons in memory."""
    config = load_config()
    from ..evolution.memory_store import MemoryStore
    store = MemoryStore(PROJECT_ROOT / config["paths"]["memory_dir"])
    lessons = store.load_all_lessons()
    if stage:
        lessons = [l for l in lessons if l.stage == stage]
    if lesson_type:
        lessons = [l for l in lessons if l.lesson_type == lesson_type]
    if min_confidence:
        lessons = [l for l in lessons if l.confidence >= min_confidence]
    lessons = lessons[:limit]

    if not lessons:
        console.print("[dim]No lessons found.[/dim]")
        return

    table = Table(title=f"Lessons ({len(lessons)})", show_header=True, border_style="slate_blue1")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Type")
    table.add_column("Stage")
    table.add_column("Conf.", justify="center")
    table.add_column("Summary", max_width=60)

    for l in lessons:
        tc = {"failure_pattern": "red", "successful_pattern": "green",
              "guardrail": "yellow", "warning": "orange3", "decision": "blue"}.get(l.lesson_type, "white")
        table.add_row(
            l.id, f"[{tc}]{l.lesson_type}[/{tc}]", l.stage,
            f"{l.confidence:.0%}", l.summary,
        )
    console.print(table)


@cli.command("memory-stats")
def memory_stats():
    """Show memory statistics."""
    config = load_config()
    from ..evolution.lesson_schema import LESSON_TYPES
    from ..evolution.memory_store import MemoryStore
    store = MemoryStore(PROJECT_ROOT / config["paths"]["memory_dir"])
    lessons = store.load_all_lessons()
    console.print(f"\n[bold]Total lessons in memory:[/bold] {len(lessons)}")
    if lessons:
        by_type = {}
        for l in lessons:
            by_type[l.lesson_type] = by_type.get(l.lesson_type, 0) + 1
        table = Table(title="By Type", border_style="slate_blue1")
        table.add_column("Type")
        table.add_column("Count", justify="right")
        for t in LESSON_TYPES:
            if t in by_type:
                table.add_row(t, str(by_type[t]))
        console.print(table)
        avg_conf = sum(l.confidence for l in lessons) / len(lessons)
        console.print(f"[bold]Average confidence:[/bold] {avg_conf:.0%}")


@cli.command("export")
@click.argument("run_id")
@click.option("--output", "-o", default=None, help="Output zip file path")
def export(run_id: str, output: str):
    """Export a run's artifacts as a zip file."""
    import zipfile
    config = load_config()
    artifacts_dir = PROJECT_ROOT / config["paths"]["artifacts_dir"]

    run_dir = None
    for d in artifacts_dir.iterdir():
        if d.name == run_id or d.name.endswith(run_id):
            run_dir = d
            break

    if not run_dir:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        sys.exit(1)

    out_path = Path(output) if output else PROJECT_ROOT / f"{run_dir.name}.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in run_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(run_dir))

    console.print(f"[green]Exported to[/green] {out_path}  ({out_path.stat().st_size // 1024} KB)")


@cli.command("promote-lessons")
@click.option("--min-confidence", default=0.8, type=float, help="Min confidence for promotion")
def promote_lessons(min_confidence: float):
    """Promote high-confidence lessons to reusable agent skills.

    Distills empirical patterns into structured skill templates that are
    automatically injected into future pipeline runs and exposed via MCP.
    """
    config = load_config()
    from .. import llm_client
    from ..evolution.memory_store import MemoryStore
    from ..evolution.skill_promoter import SkillPromoter

    provider_cfg = config.get("provider", {})
    llm_client.configure(
        provider=provider_cfg.get("name", "deepseek"),
        api_key=provider_cfg.get("api_key", ""),
        model=provider_cfg.get("model", ""),
    )

    memory_dir = PROJECT_ROOT / config["paths"]["memory_dir"]
    store = MemoryStore(memory_dir)
    lessons = store.load_all_lessons()

    eligible = [l for l in lessons if l.confidence >= min_confidence]
    console.print(f"\n[bold]Lessons eligible for promotion[/bold] (≥{min_confidence:.0%} confidence): {len(eligible)}")

    if not eligible:
        console.print("[dim]No lessons meet the confidence threshold.[/dim]")
        return

    promoter = SkillPromoter(memory_dir, min_confidence=min_confidence)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        progress.add_task("Promoting lessons to skills...", total=None)
        promoted = promoter.promote_all(lessons)

    table = Table(title=f"Promoted Skills ({len(promoted)})", border_style="slate_blue1")
    table.add_column("Stage")
    table.add_column("Skill File")
    table.add_column("Size")

    for stage, path in promoted.items():
        table.add_row(stage, path.name, f"{path.stat().st_size} bytes")

    console.print(table)
    console.print(f"\n[green]Skills saved to:[/green] {promoter.skills_dir}")
    console.print("[dim]These will be automatically injected into future pipeline runs.[/dim]")


@cli.command("list-skills")
def list_skills():
    """List all promoted skills in memory."""
    config = load_config()
    from ..evolution.skill_promoter import SkillPromoter

    memory_dir = PROJECT_ROOT / config["paths"]["memory_dir"]
    promoter = SkillPromoter(memory_dir)
    skills = promoter.list_skills()

    if not skills:
        console.print("[dim]No promoted skills yet. Run: research-evo promote-lessons[/dim]")
        return

    table = Table(title=f"Promoted Skills ({len(skills)})", border_style="slate_blue1")
    table.add_column("Stage")
    table.add_column("Lessons", justify="center")
    table.add_column("Avg Confidence", justify="center")

    for s in skills:
        table.add_row(s["stage"], str(s.get("lesson_count", "?")), s.get("avg_confidence", "?"))
    console.print(table)

    for s in skills:
        path = Path(s["path"])
        content = path.read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if not l.startswith("<!--")]
        preview = "\n".join(lines[:8])
        console.print(Panel(preview, title=f"[cyan]{s['stage']}[/cyan]", border_style="slate_blue1"))


@cli.command("mcp-server")
def mcp_server():
    """Start the MCP server (stdio mode) for agent integration.

    Add to Claude Code settings.json:
    \b
    {
      "mcpServers": {
        "evo-research": {
          "command": "research-evo",
          "args": ["mcp-server"]
        }
      }
    }
    """
    from ..mcp_server import run_mcp_server
    run_mcp_server()


def main():
    cli()


if __name__ == "__main__":
    main()
