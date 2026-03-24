class StaleProposalDetector:
    """
    Detects when the proposal stream is stuck in a discard loop.
    
    The detector tracks consecutive discards and triggers a stale alert when
    the count exceeds a threshold. It can be reset by accepted proposals
    or by changes in the proposed hyperparameters.
    """
    
    def __init__(
        self,
        consecutive_discards_threshold: int = 3,
        reset_on_any_accept: bool = True,
        reset_on_parameter_change: bool = True,
    ):
        """
        Args:
            consecutive_discards_threshold: Number of consecutive discards after which
                the detector triggers a stale proposal alert.
            reset_on_any_accept: If True, reset the discard counter when any proposal
                is accepted (even if it's a regression within SA acceptance).
            reset_on_parameter_change: If True, reset the discard counter when the
                proposed hyperparameter set differs from the previous proposal's set.
        """
        self.consecutive_discards_threshold = consecutive_discards_threshold
        self.reset_on_any_accept = reset_on_any_accept
        self.reset_on_parameter_change = reset_on_parameter_change
        
        self._discard_count = 0
        self._last_proposed_params: dict[str, float] | None = None
        self._is_stale = False

    def record_proposal(self, proposed_params: dict[str, float]) -> None:
        """
        Store the hyperparameters of the newly generated proposal.
        Called before training/evaluation.
        """
        if self.reset_on_parameter_change and self._last_proposed_params is not None:
            # Compare keys, ignoring floating-point rounding
            keys = set(proposed_params.keys()) | set(self._last_proposed_params.keys())
            for key in keys:
                val1 = proposed_params.get(key)
                val2 = self._last_proposed_params.get(key)
                if val1 is None or val2 is None:
                    # Different keys present
                    self.reset()
                    break
                if abs(val1 - val2) > 1e-9:
                    self.reset()
                    break
        
        self._last_proposed_params = proposed_params.copy()

    def record_outcome(self, accepted: bool) -> None:
        """
        Update state based on whether the proposal was accepted or discarded.
        Called after training/evaluation.
        """
        if accepted:
            if self.reset_on_any_accept:
                self.reset()
            return
        
        # Proposal was discarded
        self._discard_count += 1
        if self._discard_count >= self.consecutive_discards_threshold:
            self._is_stale = True
        else:
            self._is_stale = False

    def is_stale(self) -> bool:
        """
        Returns True if the detector believes the proposal stream is stale
        (i.e., stuck in a discard loop).
        """
        return self._is_stale

    def reset(self) -> None:
        """
        Manually reset the detector (e.g., after a forced exploration move).
        """
        self._discard_count = 0
        self._last_proposed_params = None
        self._is_stale = False