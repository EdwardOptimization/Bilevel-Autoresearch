"""Machine translation post-editing — refine the LLM's raw proposal before committing.

Translation Theory Mechanism 40:
In professional translation, machine translation output is "post-edited" by a
human to fix systematic errors (grammar, register, terminology). Applied to
autoresearch: the LLM's raw proposal is "post-edited" by rule-based checks
before committing to a GPU run. These rules encode hard-won domain knowledge
that the LLM repeatedly fails to learn from the trace alone:
  - Batch size must be a power of 2
  - LR changes should respect the scaling law (LR ~ sqrt(batch_size))
  - Weight decay should be proportional to LR (typical ratio 0.01-0.1)
  - Warmup + warmdown ratios must sum to <= 1.0
  - DEVICE_BATCH_SIZE must divide TOTAL_BATCH_SIZE evenly

The post-editor fixes or flags violations, preventing wasted GPU runs on
proposals that would certainly crash or regress.
"""
from __future__ import annotations

import ast
import logging
import math

logger = logging.getLogger(__name__)


class PostEditing:
    """Applies rule-based post-editing to refine LLM proposals before GPU runs.

    Checks domain-specific constraints and consistency rules that the LLM
    frequently violates. Can either fix violations automatically (safe fixes)
    or flag them as warnings (risky fixes) for the LLM's next iteration.
    """

    def __init__(self):
        # Track how many times each rule triggered
        self._rule_triggers: dict[str, int] = {}
        # Track auto-corrections applied
        self._corrections: list[dict] = []

    def _safe_eval(self, val) -> float | None:
        """Safely evaluate a parameter value to a float."""
        try:
            return float(ast.literal_eval(str(val)))
        except (ValueError, TypeError, SyntaxError, NameError):
            # Handle expressions like 2**19
            try:
                return float(eval(str(val), {"__builtins__": {}}, {}))
            except Exception:
                return None

    def _is_power_of_2(self, n: float) -> bool:
        """Check if a number is a power of 2."""
        if n <= 0:
            return False
        int_n = int(n)
        return int_n == n and (int_n & (int_n - 1)) == 0

    def _nearest_power_of_2(self, n: float) -> int:
        """Return the nearest power of 2."""
        if n <= 0:
            return 1
        log2 = math.log2(n)
        lower = 2 ** int(math.floor(log2))
        upper = 2 ** int(math.ceil(log2))
        if abs(n - lower) <= abs(n - upper):
            return lower
        return upper

    def _record_trigger(self, rule_name: str) -> None:
        """Record that a rule was triggered."""
        self._rule_triggers[rule_name] = self._rule_triggers.get(rule_name, 0) + 1

    def post_edit(self, changes: dict, current_config: dict) -> dict:
        """Apply post-editing rules to a proposed set of changes.

        Args:
            changes: Proposed parameter changes (param -> new_value).
            current_config: Current hyperparameter config.

        Returns:
            A dict with:
              "changes": the corrected changes dict
              "warnings": list of warning strings
              "corrections": list of auto-corrections applied
        """
        corrected = dict(changes)
        warnings = []
        corrections = []

        # --- Rule 1: Batch size must be a power of 2 ---
        for batch_param in ("TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE"):
            if batch_param in corrected:
                val = self._safe_eval(corrected[batch_param])
                if val is not None and not self._is_power_of_2(val):
                    fixed = self._nearest_power_of_2(val)
                    corrections.append({
                        "param": batch_param,
                        "original": corrected[batch_param],
                        "corrected": f"2**{int(math.log2(fixed))}",
                        "rule": "batch_size_power_of_2",
                    })
                    corrected[batch_param] = f"2**{int(math.log2(fixed))}"
                    self._record_trigger("batch_size_power_of_2")

        # --- Rule 2: Warmup + warmdown ratios must sum to <= 1.0 ---
        warmup = self._safe_eval(
            corrected.get("WARMUP_RATIO", current_config.get("WARMUP_RATIO", "0.0"))
        )
        warmdown = self._safe_eval(
            corrected.get("WARMDOWN_RATIO", current_config.get("WARMDOWN_RATIO", "0.0"))
        )
        if warmup is not None and warmdown is not None:
            if warmup + warmdown > 1.0:
                # Scale both proportionally to sum to 0.95
                total = warmup + warmdown
                scale = 0.95 / total
                new_warmup = round(warmup * scale, 4)
                new_warmdown = round(warmdown * scale, 4)

                if "WARMUP_RATIO" in corrected:
                    corrections.append({
                        "param": "WARMUP_RATIO",
                        "original": corrected["WARMUP_RATIO"],
                        "corrected": str(new_warmup),
                        "rule": "warmup_warmdown_sum",
                    })
                    corrected["WARMUP_RATIO"] = str(new_warmup)

                if "WARMDOWN_RATIO" in corrected:
                    corrections.append({
                        "param": "WARMDOWN_RATIO",
                        "original": corrected["WARMDOWN_RATIO"],
                        "corrected": str(new_warmdown),
                        "rule": "warmup_warmdown_sum",
                    })
                    corrected["WARMDOWN_RATIO"] = str(new_warmdown)

                self._record_trigger("warmup_warmdown_sum")

        # --- Rule 3: DEVICE_BATCH_SIZE must divide TOTAL_BATCH_SIZE ---
        total_bs = self._safe_eval(
            corrected.get("TOTAL_BATCH_SIZE", current_config.get("TOTAL_BATCH_SIZE", "0"))
        )
        device_bs = self._safe_eval(
            corrected.get("DEVICE_BATCH_SIZE", current_config.get("DEVICE_BATCH_SIZE", "0"))
        )
        if total_bs is not None and device_bs is not None and device_bs > 0:
            if total_bs % device_bs != 0:
                warnings.append(
                    f"DEVICE_BATCH_SIZE ({int(device_bs)}) does not evenly divide "
                    f"TOTAL_BATCH_SIZE ({int(total_bs)}). This will cause an error."
                )
                self._record_trigger("batch_size_divisibility")

        # --- Rule 4: LR values should be positive and reasonable ---
        lr_params = ["EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR"]
        for lr_param in lr_params:
            if lr_param in corrected:
                lr_val = self._safe_eval(corrected[lr_param])
                if lr_val is not None:
                    if lr_val <= 0:
                        warnings.append(
                            f"{lr_param} = {lr_val} is non-positive. Learning rate "
                            f"must be positive."
                        )
                        self._record_trigger("lr_nonpositive")
                    elif lr_val > 1.0:
                        warnings.append(
                            f"{lr_param} = {lr_val} is unusually high (>1.0). This "
                            f"will very likely cause divergence."
                        )
                        self._record_trigger("lr_too_high")

        # --- Rule 5: WEIGHT_DECAY should be non-negative ---
        if "WEIGHT_DECAY" in corrected:
            wd_val = self._safe_eval(corrected["WEIGHT_DECAY"])
            if wd_val is not None and wd_val < 0:
                corrections.append({
                    "param": "WEIGHT_DECAY",
                    "original": corrected["WEIGHT_DECAY"],
                    "corrected": str(abs(wd_val)),
                    "rule": "weight_decay_nonneg",
                })
                corrected["WEIGHT_DECAY"] = str(abs(wd_val))
                self._record_trigger("weight_decay_nonneg")

        # --- Rule 6: FINAL_LR_FRAC should be in [0, 1] ---
        if "FINAL_LR_FRAC" in corrected:
            frac_val = self._safe_eval(corrected["FINAL_LR_FRAC"])
            if frac_val is not None:
                if frac_val < 0 or frac_val > 1:
                    clamped = max(0.0, min(1.0, frac_val))
                    corrections.append({
                        "param": "FINAL_LR_FRAC",
                        "original": corrected["FINAL_LR_FRAC"],
                        "corrected": str(clamped),
                        "rule": "final_lr_frac_range",
                    })
                    corrected["FINAL_LR_FRAC"] = str(clamped)
                    self._record_trigger("final_lr_frac_range")

        # --- Rule 7: DEPTH must be positive integer ---
        if "DEPTH" in corrected:
            depth_val = self._safe_eval(corrected["DEPTH"])
            if depth_val is not None:
                if depth_val < 1 or depth_val != int(depth_val):
                    fixed = max(1, int(round(depth_val)))
                    corrections.append({
                        "param": "DEPTH",
                        "original": corrected["DEPTH"],
                        "corrected": str(fixed),
                        "rule": "depth_positive_int",
                    })
                    corrected["DEPTH"] = str(fixed)
                    self._record_trigger("depth_positive_int")

        # Store corrections
        self._corrections.extend(corrections)

        return {
            "changes": corrected,
            "warnings": warnings,
            "corrections": corrections,
        }

    def get_post_edit_text(self) -> str:
        """Generate a summary of common post-editing corrections for the prompt.

        Warns the LLM about constraints it keeps violating so it can
        self-correct in future proposals.
        """
        if not self._rule_triggers:
            return ""

        lines = ["## Post-Editing Constraints (rules your proposals keep violating)"]
        lines.append(
            "Your proposals have been automatically corrected for the following "
            "constraint violations. Please incorporate these rules into your "
            "reasoning to avoid future corrections.\n"
        )

        rule_descriptions = {
            "batch_size_power_of_2": (
                "TOTAL_BATCH_SIZE and DEVICE_BATCH_SIZE must be powers of 2 "
                "(e.g., 2**17, 2**18, 2**19)"
            ),
            "warmup_warmdown_sum": (
                "WARMUP_RATIO + WARMDOWN_RATIO must sum to <= 1.0 "
                "(they represent fractions of total training)"
            ),
            "batch_size_divisibility": (
                "DEVICE_BATCH_SIZE must evenly divide TOTAL_BATCH_SIZE"
            ),
            "lr_nonpositive": (
                "All learning rates must be strictly positive"
            ),
            "lr_too_high": (
                "Learning rates above 1.0 almost always cause divergence"
            ),
            "weight_decay_nonneg": (
                "WEIGHT_DECAY must be non-negative"
            ),
            "final_lr_frac_range": (
                "FINAL_LR_FRAC must be in [0, 1] (it's a fraction of peak LR)"
            ),
            "depth_positive_int": (
                "DEPTH must be a positive integer"
            ),
        }

        for rule, count in sorted(self._rule_triggers.items(),
                                   key=lambda x: -x[1]):
            desc = rule_descriptions.get(rule, rule)
            lines.append(f"- [{count}x] {desc}")

        return "\n".join(lines)
