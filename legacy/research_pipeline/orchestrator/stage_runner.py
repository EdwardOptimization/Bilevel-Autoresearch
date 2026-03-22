"""Execute individual pipeline stages with timing and error handling."""
import time

from ..pipeline.base import BaseStage
from .state import StageState


class StageRunner:
    def run_stage(
        self,
        stage: BaseStage,
        context: dict,
    ) -> tuple[dict, dict]:
        """
        Run a single stage.

        Returns (output_dict, metadata_dict).
        output_dict: {content, artifacts}
        metadata_dict: {state, duration_s, error}
        """
        meta = {"state": StageState.RUNNING, "duration_s": 0.0, "error": None}
        t0 = time.time()

        try:
            output = stage.run(context)
            meta["state"] = StageState.COMPLETED
        except Exception as exc:
            meta["state"] = StageState.FAILED
            meta["error"] = str(exc)
            output = {"content": f"[STAGE FAILED: {exc}]", "artifacts": []}

        meta["duration_s"] = round(time.time() - t0, 2)
        return output, meta
