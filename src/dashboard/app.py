"""EvoResearch web dashboard — FastAPI + Jinja2."""
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="EvoResearch Dashboard", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _load_config() -> dict:
    import yaml
    cfg = {}
    for fname in ["default.yaml", "local.yaml"]:
        p = PROJECT_ROOT / "config" / fname
        if p.exists():
            with open(p) as f:
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


def _get_artifacts_dir() -> Path:
    cfg = _load_config()
    return PROJECT_ROOT / cfg.get("paths", {}).get("artifacts_dir", "artifacts")


def _get_memory_dir() -> Path:
    cfg = _load_config()
    return PROJECT_ROOT / cfg.get("paths", {}).get("memory_dir", "memory")


def _load_runs() -> list[dict]:
    artifacts_dir = _get_artifacts_dir()
    if not artifacts_dir.exists():
        return []
    runs = []
    for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(meta)
        except Exception:
            continue
    return runs


def _load_run(run_id: str) -> Optional[dict]:
    artifacts_dir = _get_artifacts_dir()
    run_dir = artifacts_dir / run_id
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Attach stage outputs and review
    review_path = run_dir / "run_review.json"
    if review_path.exists():
        meta["full_review"] = json.loads(review_path.read_text(encoding="utf-8"))

    lessons_path = run_dir / "lessons.json"
    if lessons_path.exists():
        meta["run_lessons"] = json.loads(lessons_path.read_text(encoding="utf-8"))

    # Load stage outputs
    stage_contents = {}
    for stage_name in meta.get("stages", {}):
        stage_dir = run_dir / "stages" / stage_name
        for candidate in ["draft.md", "hypotheses.md", "literature_notes.md",
                          "experiment_plan.md", "results.md"]:
            p = stage_dir / candidate
            if p.exists():
                stage_contents[stage_name] = p.read_text(encoding="utf-8")
                break
    meta["stage_contents"] = stage_contents

    return meta


def _load_lessons() -> list[dict]:
    lessons_file = _get_memory_dir() / "lessons.jsonl"
    if not lessons_file.exists():
        return []
    lessons = []
    for line in lessons_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                lessons.append(json.loads(line))
            except Exception:
                continue
    return lessons


def _verdict_class(verdict: str) -> str:
    return {"pass": "verdict-pass", "weak": "verdict-weak", "fail": "verdict-fail"}.get(verdict or "", "verdict-unknown")


def _score_class(score) -> str:
    try:
        s = int(score)
        if s >= 7:
            return "score-high"
        if s >= 5:
            return "score-mid"
        return "score-low"
    except (TypeError, ValueError):
        return "score-mid"


# Register template filters
templates.env.filters["verdict_class"] = _verdict_class
templates.env.filters["score_class"] = _score_class


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    runs = _load_runs()
    lessons = _load_lessons()

    # Stats
    total_runs = len(runs)
    total_lessons = len(lessons)
    completed = [r for r in runs if r.get("state") == "completed"]
    avg_score = (
        sum(r.get("run_review", {}).get("score", 0) for r in completed) / len(completed)
        if completed else 0
    )

    # Chart data: score evolution
    chart_labels = []
    chart_scores = []
    for r in reversed(runs[:20]):
        chart_labels.append(r.get("run_id", "?")[-6:])
        chart_scores.append(r.get("run_review", {}).get("score", 0))

    lesson_type_counts = {}
    for l in lessons:
        t = l.get("lesson_type", "unknown")
        lesson_type_counts[t] = lesson_type_counts.get(t, 0) + 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "runs": runs[:20],
        "total_runs": total_runs,
        "total_lessons": total_lessons,
        "avg_score": round(avg_score, 1),
        "chart_labels": json.dumps(chart_labels),
        "chart_scores": json.dumps(chart_scores),
        "lesson_type_counts": lesson_type_counts,
    })


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    meta = _load_run(run_id)
    if not meta:
        return HTMLResponse("<h1>Run not found</h1>", status_code=404)
    return templates.TemplateResponse("run_detail.html", {
        "request": request,
        "meta": meta,
        "run_id": run_id,
    })


@app.get("/memory", response_class=HTMLResponse)
async def memory_view(
    request: Request,
    lesson_type: str = "",
    stage: str = "",
    min_confidence: float = 0.0,
):
    lessons = _load_lessons()
    if lesson_type:
        lessons = [l for l in lessons if l.get("lesson_type") == lesson_type]
    if stage:
        lessons = [l for l in lessons if l.get("stage") == stage]
    if min_confidence > 0:
        lessons = [l for l in lessons if l.get("confidence", 0) >= min_confidence]

    all_types = sorted({l.get("lesson_type", "") for l in _load_lessons()})
    all_stages = sorted({l.get("stage", "") for l in _load_lessons()})

    return templates.TemplateResponse("memory.html", {
        "request": request,
        "lessons": lessons,
        "all_types": all_types,
        "all_stages": all_stages,
        "filter_type": lesson_type,
        "filter_stage": stage,
        "filter_confidence": min_confidence,
    })


@app.get("/api/runs")
async def api_runs():
    return JSONResponse(_load_runs())


@app.get("/api/memory")
async def api_memory():
    return JSONResponse(_load_lessons())


@app.get("/api/run/{run_id}")
async def api_run(run_id: str):
    meta = _load_run(run_id)
    if not meta:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(meta)
