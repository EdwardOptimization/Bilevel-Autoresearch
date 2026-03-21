"""EvoResearch MCP Server — expose research pipeline as agent tools.

Agents connect via Model Context Protocol (stdio or HTTP).
Tools exposed:
  - research_run(topic)          → run full pipeline, return summary
  - memory_query(topic, stage)   → query evolution memory for lessons
  - run_list(limit)              → list recent runs
  - run_detail(run_id)           → get run details and review
  - lesson_list(...)             → list lessons with filters
  - skill_list()                 → list promoted skills
  - promote_lessons()            → trigger skill promotion from current memory

Usage:
  python -m src.mcp_server              # stdio mode (default)
  research-evo mcp-server               # via CLI

Add to Claude Code settings.json:
  {
    "mcpServers": {
      "evo-research": {
        "command": "python",
        "args": ["-m", "src.mcp_server"],
        "cwd": "/path/to/research_evo_mvp"
      }
    }
  }
"""
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _load_config() -> dict:
    import yaml
    cfg = {}
    for fname in ["default.yaml", "local.yaml"]:
        p = PROJECT_ROOT / "config" / fname
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    return cfg


def _get_manager():
    from src.orchestrator.run_manager import RunManager
    config = _load_config()
    return RunManager(config, PROJECT_ROOT)


def _get_memory_store():
    from src.evolution.memory_store import MemoryStore
    config = _load_config()
    return MemoryStore(PROJECT_ROOT / config["paths"]["memory_dir"])


def _get_skill_promoter():
    from src.evolution.skill_promoter import SkillPromoter
    config = _load_config()
    return SkillPromoter(PROJECT_ROOT / config["paths"]["memory_dir"])


def _get_artifacts_dir() -> Path:
    config = _load_config()
    return PROJECT_ROOT / config["paths"]["artifacts_dir"]


if MCP_AVAILABLE:
    mcp = FastMCP("EvoResearch")

    @mcp.tool()
    def research_run(topic: str, show_progress: bool = False) -> str:
        """
        Run a full AI research pipeline on the given topic.

        The pipeline runs 5 stages: literature scan → hypothesis generation →
        experiment plan → result summary → draft writeup. Each stage is evaluated
        and lessons are extracted for future runs.

        Args:
            topic: The research topic or question to investigate
            show_progress: Whether to include progress info in output

        Returns:
            JSON string with run summary: run_id, verdict, score, lessons_extracted,
            recommendations, and artifact path.
        """
        manager = _get_manager()
        result = manager.start_run(topic)
        # Return a clean summary
        return json.dumps({
            "run_id": result["run_id"],
            "topic": result["topic"],
            "verdict": result["run_review"].get("overall_verdict"),
            "score": result["run_review"].get("score"),
            "summary": result["run_review"].get("summary"),
            "lessons_extracted": result["lessons_extracted"],
            "recommendations": result["run_review"].get("recommendations", [])[:5],
            "artifact_path": result["run_dir"],
            "stage_scores": {
                s: v.get("score") for s, v in result["stage_verdicts"].items()
            },
        }, indent=2, ensure_ascii=False)

    @mcp.tool()
    def memory_query(
        topic: str,
        stage: str = "",
        lesson_type: str = "",
        min_confidence: float = 0.5,
        limit: int = 5,
    ) -> str:
        """
        Query the evolution memory for lessons relevant to a topic.

        Use this BEFORE starting any research task to check if EvoResearch has
        already learned relevant patterns, failures, or guardrails.

        Args:
            topic: Topic or task description to match against
            stage: Filter by pipeline stage (hypothesis_generation, experiment_plan_or_code, etc.)
            lesson_type: Filter by type (failure_pattern, successful_pattern, guardrail, warning, decision)
            min_confidence: Minimum confidence threshold (0.0–1.0)
            limit: Maximum number of lessons to return

        Returns:
            JSON array of relevant lessons with summary, reuse_rule, and confidence.
        """
        import re

        from src.evolution.retrieval import LessonRetriever

        store = _get_memory_store()
        lessons = store.load_all_lessons()

        # Extract tags from topic
        words = re.findall(r"\b[a-zA-Z]{4,}\b", topic.lower())
        stop = {"with", "from", "that", "this", "have", "will", "been"}
        topic_tags = list({w for w in words if w not in stop})[:10]

        retriever = LessonRetriever()
        relevant = retriever.retrieve(
            lessons,
            topic_tags=topic_tags,
            stage=stage or None,
            lesson_types=[lesson_type] if lesson_type else None,
            min_confidence=min_confidence,
            max_results=limit,
        )

        return json.dumps([{
            "id": l.id,
            "lesson_type": l.lesson_type,
            "stage": l.stage,
            "confidence": l.confidence,
            "summary": l.summary,
            "reuse_rule": l.reuse_rule,
            "anti_pattern": l.anti_pattern,
            "from_run": l.created_from_run,
        } for l in relevant], indent=2, ensure_ascii=False)

    @mcp.tool()
    def run_list(limit: int = 10) -> str:
        """
        List recent research pipeline runs with their scores and verdicts.

        Returns:
            JSON array of run summaries sorted by date (newest first).
        """
        artifacts_dir = _get_artifacts_dir()
        if not artifacts_dir.exists():
            return json.dumps([])

        runs = []
        for run_dir in sorted(artifacts_dir.iterdir(), reverse=True)[:limit]:
            meta_path = run_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append({
                "run_id": meta.get("run_id"),
                "topic": meta.get("topic"),
                "state": meta.get("state"),
                "verdict": meta.get("run_review", {}).get("overall_verdict"),
                "score": meta.get("run_review", {}).get("score"),
                "lessons_extracted": meta.get("lessons_extracted"),
                "model": meta.get("model"),
                "created_at": meta.get("created_at"),
            })
        return json.dumps(runs, indent=2, ensure_ascii=False)

    @mcp.tool()
    def run_detail(run_id: str) -> str:
        """
        Get detailed information about a specific research run.

        Args:
            run_id: The run ID (full or last 6 chars)

        Returns:
            JSON with full run metadata, review, and lessons extracted.
        """
        artifacts_dir = _get_artifacts_dir()
        run_dir = None
        for d in artifacts_dir.iterdir():
            if d.name == run_id or d.name.endswith(run_id):
                run_dir = d
                break

        if not run_dir:
            return json.dumps({"error": f"Run '{run_id}' not found"})

        result = {}
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            result["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))

        review_path = run_dir / "run_review.json"
        if review_path.exists():
            result["review"] = json.loads(review_path.read_text(encoding="utf-8"))

        lessons_path = run_dir / "lessons.json"
        if lessons_path.exists():
            result["lessons"] = json.loads(lessons_path.read_text(encoding="utf-8"))

        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def lesson_list(
        stage: str = "",
        lesson_type: str = "",
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> str:
        """
        List lessons from the evolution memory store.

        Unlike memory_query (which does semantic search), this returns all lessons
        matching the filters — useful for inspecting the full memory state.

        Args:
            stage: Filter by pipeline stage (hypothesis_generation, etc.)
            lesson_type: Filter by type (failure_pattern, successful_pattern, guardrail, etc.)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of lessons to return (sorted by confidence desc)

        Returns:
            JSON array of lessons with all fields.
        """
        store = _get_memory_store()
        lessons = store.load_all_lessons()

        filtered = [
            l for l in lessons
            if l.confidence >= min_confidence
            and (not stage or l.stage == stage)
            and (not lesson_type or l.lesson_type == lesson_type)
        ]
        # Sort by confidence descending
        filtered.sort(key=lambda l: -l.confidence)

        return json.dumps([{
            "id": l.id,
            "lesson_type": l.lesson_type,
            "stage": l.stage,
            "confidence": l.confidence,
            "summary": l.summary,
            "reuse_rule": l.reuse_rule,
            "anti_pattern": l.anti_pattern,
            "topic_tags": l.topic_tags,
            "from_run": l.created_from_run,
        } for l in filtered[:limit]], indent=2, ensure_ascii=False)

    @mcp.tool()
    def skill_list() -> str:
        """
        List all promoted skills — distilled from high-confidence lessons.

        Promoted skills are generalized, reusable prompt templates extracted
        from patterns that repeated across multiple runs. They are automatically
        injected into the pipeline and can be used by agents directly.

        Returns:
            JSON array of promoted skills with stage, confidence, and content preview.
        """
        promoter = _get_skill_promoter()
        skills = promoter.list_skills()
        # Add content preview
        for skill in skills:
            path = Path(skill["path"])
            if path.exists():
                lines = [l for l in path.read_text(encoding="utf-8").splitlines()
                         if not l.startswith("<!--")]
                skill["preview"] = "\n".join(lines[:15])
        return json.dumps(skills, indent=2, ensure_ascii=False)

    @mcp.tool()
    def promote_lessons(min_confidence: float = 0.8) -> str:
        """
        Trigger skill promotion — distill high-confidence lessons into reusable skills.

        This converts empirically-discovered patterns into structured skill templates
        that get automatically injected into future pipeline runs.

        Args:
            min_confidence: Minimum confidence threshold for promotion (default: 0.8)

        Returns:
            JSON summary of promoted skills per stage.
        """
        from src.evolution.skill_promoter import SkillPromoter
        config = _load_config()
        memory_dir = PROJECT_ROOT / config["paths"]["memory_dir"]

        store = _get_memory_store()
        lessons = store.load_all_lessons()

        promoter = SkillPromoter(memory_dir, min_confidence=min_confidence)
        promoted = promoter.promote_all(lessons)

        return json.dumps({
            "promoted_stages": list(promoted.keys()),
            "skill_files": {stage: str(path) for stage, path in promoted.items()},
            "total_lessons_used": len([l for l in lessons if l.confidence >= min_confidence]),
        }, indent=2, ensure_ascii=False)

    def run_mcp_server():
        """Start the MCP server in stdio mode."""
        mcp.run(transport="stdio")

else:
    def run_mcp_server():
        print("MCP package not installed. Run: pip install mcp", file=sys.stderr)
        print("Falling back to HTTP API — use 'research-evo serve' instead.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_mcp_server()
