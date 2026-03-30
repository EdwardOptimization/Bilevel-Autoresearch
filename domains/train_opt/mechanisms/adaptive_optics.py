"""Adaptive optics — real-time noise correction for val_bpb measurements.

In astronomy, atmospheric turbulence distorts starlight, blurring images. Adaptive
optics (AO) systems measure this distortion using a guide star and correct it in
real-time with a deformable mirror, recovering near-diffraction-limited resolution.

In this system, each val_bpb measurement is "distorted" by training noise: random
seed effects, GPU non-determinism, and the stochastic nature of short training runs.
A 0.001 bpb improvement might be real signal or just noise. Without correction, the
search wastes iterations chasing noise.

This mechanism estimates measurement noise from the history of similar configs (the
"guide star") and computes a signal-to-noise ratio (SNR) for each improvement. It
then warns the LLM when an improvement has low SNR (likely noise) and recommends
larger changes when the noise floor is high relative to recent deltas.

Key insight: the system treats every bpb delta as equally trustworthy. A 0.0001
improvement after changing LR by 0.1% is almost certainly noise. Adaptive optics
helps distinguish real signal from atmospheric distortion.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AdaptiveOptics:
    """Estimates measurement noise and computes signal-to-noise ratios.

    Tracks the variance of val_bpb across iterations and uses it to estimate
    the noise floor. When the LLM proposes a change that produces a delta
    smaller than the noise floor, it flags the result as "below the seeing limit."

    The noise estimate is built from pairs of consecutive iterations where the
    same parameter set was active — these should have similar bpb if the search
    is near an optimum, and the variance between them estimates the noise.
    """

    def __init__(self, min_samples: int = 3):
        self._min_samples = min_samples
        # List of (val_bpb, status, changes_set) for all iterations
        self._history: list[tuple[float, str, frozenset]] = []
        # Estimated noise std (in bpb units)
        self._noise_std: float = 0.0
        # Recent deltas for SNR computation
        self._recent_deltas: list[float] = []

    def record(self, val_bpb: float, status: str, changes: dict) -> None:
        """Record an observation for noise estimation.

        Args:
            val_bpb: the measured validation bpb.
            status: "keep", "discard", or "crash".
            changes: param -> value mapping of what was changed.
        """
        if status == "crash":
            return

        change_set = frozenset(changes.keys())
        self._history.append((val_bpb, status, change_set))

        # Track deltas between consecutive non-crash results
        if len(self._history) >= 2:
            prev_bpb = self._history[-2][0]
            delta = abs(val_bpb - prev_bpb)
            self._recent_deltas.append(delta)
            # Keep bounded
            if len(self._recent_deltas) > 20:
                self._recent_deltas = self._recent_deltas[-20:]

        self._update_noise_estimate()

    def _update_noise_estimate(self) -> None:
        """Update the noise floor estimate using robust statistics.

        Uses the median absolute deviation (MAD) of recent deltas as a robust
        estimator of noise std. MAD is preferred over std because it's resistant
        to outliers (genuine improvements are outliers from the noise perspective).
        """
        if len(self._recent_deltas) < self._min_samples:
            return

        # Median absolute deviation (MAD) — robust noise estimator
        sorted_deltas = sorted(self._recent_deltas)
        n = len(sorted_deltas)
        median = sorted_deltas[n // 2]

        abs_devs = sorted(abs(d - median) for d in sorted_deltas)
        mad = abs_devs[len(abs_devs) // 2]

        # Convert MAD to std: std ~= 1.4826 * MAD for normal distributions
        self._noise_std = 1.4826 * mad

        # Floor: noise can't be zero (there's always some training stochasticity)
        if self._noise_std < 1e-6:
            self._noise_std = 0.0005  # conservative minimum noise floor

    def compute_snr(self, delta_bpb: float) -> float:
        """Compute signal-to-noise ratio for a bpb improvement.

        Args:
            delta_bpb: the improvement (negative = better). Uses absolute value.

        Returns:
            SNR as a float. SNR > 2.0 is "likely real signal", SNR < 1.0 is
            "likely noise", 1.0-2.0 is "marginal."
        """
        if self._noise_std < 1e-9:
            return float("inf")  # No noise estimate yet
        return abs(delta_bpb) / self._noise_std

    def get_seeing_text(self) -> str:
        """Generate adaptive optics report for injection into the proposal prompt.

        Reports the estimated noise floor ("seeing conditions"), recent SNRs,
        and recommendations for minimum change magnitude to beat the noise.
        """
        if self._noise_std < 1e-9 or len(self._recent_deltas) < self._min_samples:
            return ""

        lines = ["## Adaptive Optics (measurement noise correction)"]
        lines.append(
            "The val_bpb measurement has inherent noise from training stochasticity. "
            "Changes smaller than the noise floor cannot be distinguished from random "
            "fluctuation. Aim for improvements LARGER than the noise floor."
        )

        lines.append(f"\n- **Noise floor (1-sigma)**: {self._noise_std:.6f} bpb")
        lines.append(f"- **Reliable detection threshold (2-sigma)**: {2 * self._noise_std:.6f} bpb")

        # Classify recent results by SNR
        if len(self._history) >= 2:
            recent_snrs = []
            for i in range(max(0, len(self._history) - 5), len(self._history)):
                if i == 0:
                    continue
                delta = abs(self._history[i][0] - self._history[i - 1][0])
                snr = self.compute_snr(delta)
                status = self._history[i][1]
                if snr != float("inf"):
                    recent_snrs.append((snr, status))

            if recent_snrs:
                lines.append("\n### Recent observation quality")
                for snr, status in recent_snrs[-5:]:
                    if snr > 2.0:
                        quality = "CLEAR (real signal)"
                    elif snr > 1.0:
                        quality = "MARGINAL (may be noise)"
                    else:
                        quality = "BELOW SEEING (likely noise)"
                    lines.append(f"  - SNR={snr:.1f} [{status}]: {quality}")

        # Recommendation
        lines.append(
            f"\n**Recommendation**: Make changes large enough to produce a delta "
            f"> {2 * self._noise_std:.4f} bpb. Tiny tweaks (delta < {self._noise_std:.4f}) "
            f"are indistinguishable from noise and waste iterations."
        )

        return "\n".join(lines)
