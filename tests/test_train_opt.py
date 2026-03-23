"""Unit tests for the train_opt domain — SearchConfig, TrainResult, TrainTrace."""
import pytest

from domains.train_opt.config import SearchConfig
from domains.train_opt.runner import TrainResult, TrainTrace


class TestSearchConfig:
    def test_default_editable_params_contains_expected(self):
        cfg = SearchConfig()
        expected = [
            "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN",
            "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR",
            "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
            "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
            "DEPTH", "DEVICE_BATCH_SIZE",
        ]
        for param in expected:
            assert param in cfg.editable_params

    def test_active_params_excludes_frozen(self):
        cfg = SearchConfig(frozen_params=["DEPTH", "HEAD_DIM"])
        active = cfg.active_params
        assert "DEPTH" not in active
        assert "HEAD_DIM" not in active
        for p in cfg.editable_params:
            if p not in cfg.frozen_params:
                assert p in active

    def test_active_params_full_when_nothing_frozen(self):
        cfg = SearchConfig()
        assert cfg.active_params == cfg.editable_params

    def test_freeze_a_param(self):
        cfg = SearchConfig()
        assert "EMBEDDING_LR" in cfg.active_params
        cfg.frozen_params.append("EMBEDDING_LR")
        assert "EMBEDDING_LR" not in cfg.active_params

    def test_freeze_all_params(self):
        cfg = SearchConfig()
        cfg.frozen_params = list(cfg.editable_params)
        assert cfg.active_params == []

    def test_default_strategy_and_guidance(self):
        cfg = SearchConfig()
        assert cfg.strategy == "explore"
        assert cfg.guidance == ""

    def test_custom_strategy(self):
        cfg = SearchConfig(strategy="exploit", guidance="focus on LR")
        assert cfg.strategy == "exploit"
        assert cfg.guidance == "focus on LR"

    def test_default_budgets(self):
        cfg = SearchConfig()
        assert cfg.inner_budget == 5
        assert cfg.time_budget == 300


class TestTrainResult:
    def _make_result(self, **kwargs):
        defaults = dict(
            iteration=1,
            val_bpb=3.14,
            peak_vram_mb=8192.0,
            training_seconds=120.5,
            num_params_m=85.0,
            status="keep",
            changes={"EMBEDDING_LR": 0.001},
            description="lowered LR",
        )
        defaults.update(kwargs)
        return TrainResult(**defaults)

    def test_basic_creation(self):
        r = self._make_result()
        assert r.iteration == 1
        assert r.val_bpb == pytest.approx(3.14)
        assert r.status == "keep"
        assert r.changes == {"EMBEDDING_LR": 0.001}

    def test_default_depth_is_zero(self):
        r = self._make_result()
        assert r.depth == 0

    def test_custom_depth(self):
        r = self._make_result(depth=4)
        assert r.depth == 4

    def test_discard_status(self):
        r = self._make_result(status="discard")
        assert r.status == "discard"

    def test_crash_status(self):
        r = self._make_result(status="crash", val_bpb=float("inf"))
        assert r.status == "crash"
        assert r.val_bpb == float("inf")

    def test_empty_changes(self):
        r = self._make_result(changes={})
        assert r.changes == {}


class TestTrainTrace:
    def _make_result(self, iteration, val_bpb, status="keep"):
        return TrainResult(
            iteration=iteration,
            val_bpb=val_bpb,
            peak_vram_mb=0.0,
            training_seconds=0.0,
            num_params_m=0.0,
            status=status,
            changes={},
            description="test",
        )

    def test_empty_trace_initial_state(self):
        trace = TrainTrace()
        assert trace.results == []
        assert trace.best_bpb == float("inf")
        assert trace.best_iteration == 0

    def test_add_keep_updates_best(self):
        trace = TrainTrace()
        trace.add(self._make_result(1, 3.5, "keep"))
        assert trace.best_bpb == pytest.approx(3.5)
        assert trace.best_iteration == 1

    def test_add_better_keep_updates_best(self):
        trace = TrainTrace()
        trace.add(self._make_result(1, 3.5, "keep"))
        trace.add(self._make_result(2, 3.2, "keep"))
        assert trace.best_bpb == pytest.approx(3.2)
        assert trace.best_iteration == 2

    def test_add_worse_keep_does_not_update_best(self):
        trace = TrainTrace()
        trace.add(self._make_result(1, 3.2, "keep"))
        trace.add(self._make_result(2, 3.5, "keep"))
        assert trace.best_bpb == pytest.approx(3.2)
        assert trace.best_iteration == 1

    def test_add_discard_does_not_update_best(self):
        trace = TrainTrace()
        trace.add(self._make_result(1, 3.5, "keep"))
        trace.add(self._make_result(2, 2.0, "discard"))
        assert trace.best_bpb == pytest.approx(3.5)
        assert trace.best_iteration == 1

    def test_add_crash_does_not_update_best(self):
        trace = TrainTrace()
        trace.add(self._make_result(1, 3.5, "keep"))
        trace.add(self._make_result(2, 1.0, "crash"))
        assert trace.best_bpb == pytest.approx(3.5)

    def test_results_list_grows(self):
        trace = TrainTrace()
        for i in range(5):
            trace.add(self._make_result(i, 3.0 - i * 0.1, "keep"))
        assert len(trace.results) == 5

    def test_summary_contains_best_line(self):
        trace = TrainTrace()
        trace.add(self._make_result(1, 3.14, "keep"))
        summary = trace.summary()
        assert "3.140000" in summary
        assert "iter 1" in summary

    def test_summary_last_n_limits_output(self):
        trace = TrainTrace()
        for i in range(20):
            trace.add(self._make_result(i + 1, 3.0, "keep"))
        summary = trace.summary(last_n=3)
        # Body lines start with two spaces; the "Best:" header line does not
        body_lines = [l for l in summary.splitlines() if l.startswith("  iter")]
        assert len(body_lines) == 3

    def test_summary_empty_trace(self):
        trace = TrainTrace()
        summary = trace.summary()
        assert "Best" in summary
