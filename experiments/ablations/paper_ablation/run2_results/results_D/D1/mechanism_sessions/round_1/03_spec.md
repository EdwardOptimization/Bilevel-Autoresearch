**Mechanism name**: `diversity_enforcer`

**Implementation strategy**: `new_helper_class`

**Target**: Create a new class `DiversityEnforcer` that will be instantiated in `TrainRunner.__init__` and called during the proposal generation phase.

**Interface**:
```python
class DiversityEnforcer:
    def __init__(
        self,
        history_window: int = 5,
        min_distance_threshold: float = 0.3,
        max_retries: int = 3
    ):
        """
        Args:
            history_window: Number of recent iterations to consider for diversity checking
            min_distance_threshold: Minimum normalized distance (0-1) required from recent configs
            max_retries: Maximum attempts to generate a diverse proposal before accepting the best
        """
        
    def enforce_diversity(
        self,
        proposed_config: Dict[str, Any],
        recent_configs: List[Dict[str, Any]],
        iteration: int
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Check if proposed config is sufficiently diverse from recent history.
        
        Args:
            proposed_config: Hyperparameter configuration from LLM proposal
            recent_configs: List of recent configurations (most recent last)
            iteration: Current iteration number
            
        Returns:
            Tuple of (final_config, was_modified, reason)
            - final_config: Either original or modified configuration
            - was_modified: True if config was modified for diversity
            - reason: Human-readable explanation of what happened
        """
```

**Step-by-step logic**:

1. **Normalize hyperparameters**:
   - For each hyperparameter in `proposed_config`:
     - If numeric (int/float), map to [0,1] range using predefined min/max bounds
     - For learning rate: log10 scale then normalize between 1e-5 and 1.0
     - For categorical (optimizer, scheduler): one-hot encode
     - For boolean: map to 0/1

2. **Calculate distance metric**:
   - If `recent_configs` is empty or shorter than 2 items, return `(proposed_config, False, "insufficient history")`
   - For each config in `recent_configs[-history_window:]`:
     - Compute Euclidean distance between normalized vectors
     - Track minimum distance to any recent config

3. **Diversity check**:
   - If `min_distance >= min_distance_threshold`:
     - Return `(proposed_config, False, f"sufficiently diverse (distance={min_distance:.3f})")`
   - Else:
     - Mark as insufficiently diverse

4. **Diversification procedure** (if insufficiently diverse):
   - Identify which hyperparameters contribute most to similarity:
     - For each parameter, compute its contribution to the small distance
     - Sort parameters by contribution (highest first)
   
   - Apply targeted perturbations:
     - For top 2-3 most similar parameters:
       - If numeric: apply random shift ±20-50% of parameter range
         - Bias toward exploration: 70% chance to move away from recent values
       - If categorical: switch to different category with 60% probability
   
   - Recalculate distance with modified config
   - Repeat up to `max_retries` times until diversity threshold met
   
   - If still insufficient after retries:
     - Accept the most diverse version found
     - Log warning about search space saturation

5. **Return final configuration**:
   - Denormalize modified parameters back to original ranges
   - Return tuple with final config, modification flag, and explanation

**Integration points**:

1. **In `TrainRunner.__init__`**:
```python
# Add after other helper class initializations
self.diversity_enforcer = DiversityEnforcer(
    history_window=4,
    min_distance_threshold=0.35,
    max_retries=2
)
```

2. **In `run_iteration` method** (after LLM generates proposal):
```python
# After extracting current_config and before any training

# Get recent configurations from trace
recent_configs = []
for i in range(max(0, iteration - 5), iteration):
    if i < len(self.trace.results):
        recent_configs.append(self._extract_hyperparams_from_trace(i))

# Apply diversity enforcement
diversified_config, was_modified, reason = self.diversity_enforcer.enforce_diversity(
    proposed_config=proposed_config_dict,
    recent_configs=recent_configs,
    iteration=iteration
)

if was_modified:
    logger.info(f"[Iter {iteration}] Diversity enforcement applied: {reason}")
    # Convert diversified config back to code modification
    diversified_code = self._apply_hyperparams_to_code(
        self.current_code, 
        diversified_config
    )
    # Use diversified code instead of original proposal
    # (Need to integrate with existing proposal processing)
```

3. **Additional helper method needed in TrainRunner**:
```python
def _extract_hyperparams_from_trace(self, iteration: int) -> Dict[str, Any]:
    """Extract hyperparameters from a past iteration's code in trace."""
    if iteration < len(self.trace.results):
        result = self.trace.results[iteration]
        if result.code:
            return self._extract_hyperparams(result.code)
    return {}
```

**Key design decisions**:
1. Operates on normalized hyperparameter space for fair distance calculation
2. Targeted perturbations rather than random noise
3. Progressive fallback: try to diversify, but accept best effort if stuck
4. Configurable thresholds to balance exploration/exploitation
5. Integration preserves existing proposal pipeline, just modifies the config before training