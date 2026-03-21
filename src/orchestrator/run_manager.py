"""Main orchestrator: manages the full run lifecycle with quality gates."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import llm_client
from ..evaluator.run_reviewer import RunReviewer
from ..evaluator.stage_evaluator import StageEvaluator
from ..evolution.injection import format_lessons_for_injection
from ..evolution.lesson_extractor import LessonExtractor
from ..evolution.memory_store import MemoryStore
from ..evolution.retrieval import LessonRetriever
from ..evolution.skill_promoter import SkillPromoter
from ..pipeline.draft_writeup import DraftWriteupStage
from ..pipeline.experiment_plan_or_code import ExperimentPlanStage
from ..pipeline.experiment_result_summary import ExperimentResultSummaryStage
from ..pipeline.hypothesis_generation import HypothesisGenerationStage
from ..pipeline.literature_scan import LiteratureScanStage
from .stage_runner import StageRunner
from .state import RunState, StageState

STAGE_ORDER = [
    "literature_scan",
    "hypothesis_generation",
    "experiment_plan_or_code",
    "experiment_result_summary",
    "draft_writeup",
]

# Stages that receive lesson injection
INJECTION_STAGES = {"hypothesis_generation", "experiment_plan_or_code", "draft_writeup"}


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{ts}-{uuid.uuid4().hex[:6]}"


class RunManager:
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.artifacts_dir = project_root / config["paths"]["artifacts_dir"]
        self.memory_dir = project_root / config["paths"]["memory_dir"]

        # Configure LLM provider
        provider_cfg = config.get("provider", {})
        llm_client.configure(
            provider=provider_cfg.get("name", "deepseek"),
            api_key=provider_cfg.get("api_key", ""),
            model=provider_cfg.get("model", ""),
        )

        self.max_tokens = config["model"]["max_tokens"]
        self.max_lessons = config["memory"]["max_retrieved_lessons"]
        self.min_confidence = config["memory"]["min_confidence"]

        # Quality gate config
        qg = config.get("quality_gate", {})
        self.qg_enabled = qg.get("enabled", True)
        self.qg_min_score = qg.get("min_score", 5)
        self.qg_max_retries = qg.get("max_retries", 1)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.stage_runner = StageRunner()
        self.stage_evaluator = StageEvaluator()
        self.run_reviewer = RunReviewer()
        self.lesson_extractor = LessonExtractor()
        self.memory_store = MemoryStore(self.memory_dir)
        self.retriever = LessonRetriever()
        self.skill_promoter = SkillPromoter(
            self.memory_dir,
            min_confidence=self.min_confidence,
        )

        self._stages = {
            "literature_scan":          LiteratureScanStage(),
            "hypothesis_generation":    HypothesisGenerationStage(),
            "experiment_plan_or_code":  ExperimentPlanStage(),
            "experiment_result_summary": ExperimentResultSummaryStage(),
            "draft_writeup":            DraftWriteupStage(),
        }

    def start_run(self, topic: str, on_progress=None) -> dict:
        """Execute a full research pipeline run. Returns run summary dict."""
        run_id = _make_run_id()
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stages").mkdir(exist_ok=True)

        def emit(stage, event, data=None):
            if on_progress:
                on_progress(stage, event, data or {})

        # ── Retrieve lessons ──────────────────────────────────────────────────
        all_lessons = self.memory_store.load_all_lessons()
        topic_tags = self._extract_tags(topic)

        retrieved_by_stage: dict[str, str] = {}
        retrieval_debug: dict[str, list] = {}

        # Load global skill once (structural guidance for all stages)
        global_skill = self.skill_promoter.load_skill("global")

        # Lesson injection: only for stages that benefit from prior-run lessons
        for stage_name in INJECTION_STAGES:
            relevant = self.retriever.retrieve(
                all_lessons,
                topic_tags=topic_tags,
                stage=stage_name,
                min_confidence=self.min_confidence,
                max_results=self.max_lessons,
            )
            retrieval_debug[stage_name] = [lesson.id for lesson in relevant]
            lesson_text = format_lessons_for_injection(relevant, context=f"stage: {stage_name}")
            # Inject stage-specific promoted skill (distilled from many runs)
            promoted_skill = self.skill_promoter.load_skill(stage_name)
            skill_parts = []
            if global_skill:
                skill_parts.append(f"## Global Research Standards\n{global_skill}")
            if promoted_skill:
                skill_parts.append(f"## Stage Skill: {stage_name}\n{promoted_skill}")
            if skill_parts:
                skill_section = "\n## Empirically Distilled Skills\n" + "\n\n".join(skill_parts) + "\n"
                lesson_text = skill_section + ("\n" + lesson_text if lesson_text else "")
            retrieved_by_stage[stage_name] = lesson_text

        # Skill-only injection for non-lesson stages (no lesson bias, but skills are structural)
        for stage_name in STAGE_ORDER:
            if stage_name not in INJECTION_STAGES:
                stage_skill = self.skill_promoter.load_skill(stage_name)
                skill_parts = []
                if global_skill:
                    skill_parts.append(f"## Global Research Standards\n{global_skill}")
                if stage_skill:
                    skill_parts.append(f"## Stage Skill: {stage_name}\n{stage_skill}")
                if skill_parts:
                    retrieved_by_stage[stage_name] = "\n## Empirically Distilled Skills\n" + "\n\n".join(skill_parts) + "\n"

        debug_path = self.memory_dir / "retrieval_debug" / f"{run_id}.json"
        debug_path.parent.mkdir(exist_ok=True)
        debug_path.write_text(json.dumps(retrieval_debug, indent=2), encoding="utf-8")

        # ── Init metadata ─────────────────────────────────────────────────────
        provider_info = llm_client.get_provider_info()
        metadata = {
            "run_id": run_id,
            "topic": topic,
            "topic_tags": topic_tags,
            "state": RunState.RUNNING,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider_info["provider"],
            "model": provider_info["model"],
            "stages": {},
            "retrieved_lesson_ids": retrieval_debug,
        }
        self._save_metadata(run_dir, metadata)

        # ── Run pipeline ──────────────────────────────────────────────────────
        previous_outputs: dict[str, str] = {}
        stage_outputs_raw: dict[str, str] = {}
        stage_verdicts: dict[str, dict] = {}

        for stage_name in STAGE_ORDER:
            emit(stage_name, "start")
            stage = self._stages[stage_name]
            lessons_text = retrieved_by_stage.get(stage_name, "")

            # ── Quality gate loop ─────────────────────────────────────────────
            # Stage can override max_retries via class attribute
            stage_max_retries = (
                stage.max_retries if stage.max_retries is not None else self.qg_max_retries
            )
            attempt = 0
            evaluator_feedback = ""
            verdict = {}

            while True:
                context = {
                    "topic": topic,
                    "previous_outputs": previous_outputs.copy(),
                    "retrieved_lessons": lessons_text,
                    "evaluator_feedback": evaluator_feedback,
                    "run_dir": run_dir,
                }
                output, run_meta = self.stage_runner.run_stage(stage, context)

                emit(stage_name, "evaluating", {"attempt": attempt + 1})
                verdict = self.stage_evaluator.evaluate(stage_name, output["content"], topic)
                verdict["state"] = run_meta["state"]
                verdict["duration_s"] = run_meta["duration_s"]
                if run_meta.get("error"):
                    verdict["error"] = run_meta["error"]
                if attempt > 0:
                    verdict["retried"] = True

                score = int(verdict.get("score", 5))  # coerce: LLM may return "7" as str
                verdict["score"] = score  # normalize in-place for downstream use
                retry_eligible = (
                    self.qg_enabled
                    and attempt < stage_max_retries
                    and score < self.qg_min_score
                    and run_meta["state"] == StageState.COMPLETED
                )

                if retry_eligible:
                    weaknesses = verdict.get("weaknesses", [])
                    evaluator_feedback = (
                        f"Your previous attempt scored {score}/10 (verdict: {verdict.get('verdict', 'weak')}).\n"
                        f"Evaluator feedback: {verdict.get('feedback', '')}\n\n"
                        f"Specific weaknesses to address:\n"
                        + "\n".join(f"- {w}" for w in weaknesses)
                        + "\n\nPlease revise your response to directly address these issues."
                    )
                    emit(stage_name, "retrying", {"attempt": attempt + 1, "score": score})
                    attempt += 1
                    continue
                break
            # ── End quality gate loop ─────────────────────────────────────────

            stage_verdicts[stage_name] = verdict
            previous_outputs[stage_name] = output["content"]
            stage_outputs_raw[stage_name] = output["content"]

            metadata["stages"][stage_name] = {
                "state": run_meta["state"],
                "duration_s": run_meta["duration_s"],
                "artifacts": output["artifacts"],
                "verdict": verdict.get("verdict"),
                "score": verdict.get("score"),
                "retried": verdict.get("retried", False),
            }
            self._save_metadata(run_dir, metadata)
            emit(stage_name, "done", {
                "verdict": verdict.get("verdict"), "score": verdict.get("score"),
                "retried": verdict.get("retried", False),
            })

        # ── Run review ────────────────────────────────────────────────────────
        emit("run_review", "start")
        run_review = self.run_reviewer.review(topic, stage_verdicts, stage_outputs_raw)

        review_md = self._format_review_md(run_id, topic, run_review, stage_verdicts)
        (run_dir / "review.md").write_text(review_md, encoding="utf-8")
        (run_dir / "run_review.json").write_text(
            json.dumps(run_review, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metadata["state"] = RunState.REVIEWED
        metadata["run_review"] = {
            "overall_verdict": run_review.get("overall_verdict"),
            "score": run_review.get("score"),
        }
        self._save_metadata(run_dir, metadata)
        emit("run_review", "done", run_review)

        # ── Lesson extraction ─────────────────────────────────────────────────
        emit("lesson_extraction", "start")
        lessons = self.lesson_extractor.extract(
            run_id=run_id,
            topic=topic,
            run_review=run_review,
            stage_verdicts=stage_verdicts,
            stage_outputs={k: {"content": v} for k, v in stage_outputs_raw.items()},
        )
        lessons_json = json.dumps([lesson.to_dict() for lesson in lessons], indent=2, ensure_ascii=False)
        (run_dir / "lessons.json").write_text(lessons_json, encoding="utf-8")
        self.memory_store.save_lessons(lessons)

        metadata["state"] = RunState.COMPLETED
        metadata["lessons_extracted"] = len(lessons)
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata(run_dir, metadata)
        emit("lesson_extraction", "done", {"count": len(lessons)})

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "topic": topic,
            "state": RunState.COMPLETED,
            "run_review": run_review,
            "lessons_extracted": len(lessons),
            "stage_verdicts": stage_verdicts,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_metadata(self, run_dir: Path, metadata: dict) -> None:
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def _extract_tags(self, topic: str) -> list[str]:
        import re
        words = re.findall(r"\b[a-zA-Z]{4,}\b", topic.lower())
        stop = {"with", "from", "that", "this", "have", "will", "been", "into",
                "than", "more", "each", "they", "their", "when", "what", "which"}
        return list({w for w in words if w not in stop})[:10]

    def _format_review_md(self, run_id, topic, review, verdicts) -> str:
        v_color = {"pass": "✅", "weak": "⚠️", "fail": "❌"}
        lines = [
            f"# Run Review: {run_id}",
            f"**Topic**: {topic}",
            f"**Overall**: {v_color.get(review.get('overall_verdict',''), '❓')} "
            f"{review.get('overall_verdict','N/A')} ({review.get('score','?')}/10)",
            "",
            "## Summary",
            review.get("summary", ""),
            "",
            "## Stage Results",
        ]
        for stage, v in verdicts.items():
            icon = v_color.get(v.get("verdict", ""), "❓")
            retry_tag = " *(retried)*" if v.get("retried") else ""
            lines.append(f"- {icon} **{stage}**: {v.get('verdict','?')} ({v.get('score','?')}/10){retry_tag}")
        for section, key in [("Successes", "success_points"), ("Failures", "failure_points"),
                              ("Weak Points", "weak_points"), ("Recommendations", "recommendations")]:
            items = review.get(key, [])
            if items:
                lines += ["", f"## {section}"]
                lines += [f"- {item}" for item in items]
        return "\n".join(lines)
