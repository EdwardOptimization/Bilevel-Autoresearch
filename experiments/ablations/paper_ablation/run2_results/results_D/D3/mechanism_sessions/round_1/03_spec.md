**1. Mechanism name**  
`stale_proposal_detector`

**2. Implementation strategy**  
`new_helper_class`

**3. Target**  
New class `StaleProposalDetector` to be instantiated in `TrainRunner.__init__`.

**4. Interface**  

```python
class StaleProposalDetector:
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
        # Implementation logic below

    def record_outcome(self, accepted: bool) -> None:
        """
        Update state based on whether the proposal was accepted or discarded.
        Called after training/evaluation.
        """
        # Implementation logic below

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
```

**5. Step-by-step logic**  

**`record_proposal(proposed_params)`**  
1. If `reset_on_parameter_change` is True and `_last_proposed_params` exists:  
   a. Compare `proposed_params` with `_last_proposed_params` ignoring floating‑point rounding (e.g., `abs(a‑b) < 1e‑9`).  
   b. If any key’s value differs, call `self.reset()`.  
2. Store `proposed_params` in `_last_proposed_params`.

**`record_outcome(accepted)`**  
1. If `accepted` is True:  
   a. If `reset_on_any_accept` is True, call `self.reset()`.  
   b. Return.  
2. If `accepted` is False:  
   a. Increment `_discard_count` by 1.  
   b. If `_discard_count >= consecutive_discards_threshold`:  
      - Set `_is_stale = True`.  
   c. Else:  
      - Set `_is_stale = False`.

**`is_stale()`**  
1. Return `_is_stale`.

**`reset()`**  
1. Set `_discard_count = 0`.  
2. Set `_last_proposed_params = None`.  
3. Set `_is_stale = False`.

**6. Integration points**  

**In `TrainRunner.__init__`:**  
Add after other helper‑class initializations:  
```python
# Improvement 17: Stale proposal detector
self.stale_detector = StaleProposalDetector(
    consecutive_discards_threshold=3,
    reset_on_any_accept=True,
    reset_on_parameter_change=True,
)
```

**In `TrainRunner.run_iteration`:**  
1. **After generating a proposal** (step 1 in the loop), extract the proposed hyperparameters (using `_extract_hyperparams` on the proposed code) and call:  
   ```python
   proposed_params = self._extract_hyperparams(proposed_code)
   self.stale_detector.record_proposal(proposed_params)
   ```
2. **After the keep/discard decision** (step 5 in the loop), call:  
   ```python
   accepted = (result.status == "kept")  # or however the final accept/reject is determined
   self.stale_detector.record_outcome(accepted)
   ```
3. **In the proposal‑generation logic** (step 1), check for staleness before asking the LLM for a new proposal. If stale:  
   - Log a warning: “Stale proposal stream detected — forcing exploration move.”  
   - Inject a forced‑exploration prompt (e.g., “Ignore recent history and try a completely different hyperparameter set.”)  
   - After injecting the forced exploration, call `self.stale_detector.reset()` to clear the stale flag.

**Rationale for default parameters**  
- `consecutive_discards_threshold=3`: Matches the observed pathology (4+ consecutive discards in the trace).  
- `reset_on_any_accept=True`: Prevents the detector from triggering after a successful accept‑regress‑accept pattern.  
- `reset_on_parameter_change=True`: Allows the LLM to try a different hyperparameter set without being flagged as stale, even if the previous set was discarded.  

This design directly interrupts the “stuck on reduce WD” failure mode with minimal tuning surface and no new dependencies.