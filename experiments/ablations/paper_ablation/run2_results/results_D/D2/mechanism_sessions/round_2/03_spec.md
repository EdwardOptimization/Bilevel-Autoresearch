## **Implementation Specification: Trust Region Constraint**

### 1. **Mechanism name**
`trust_region_constraint`

### 2. **Implementation strategy**
`new_helper_class` + `modify_init` + `replace_method`

### 3. **Target**
- New class: `TrustRegionConstraint`
- Modified: `TrainRunner.__init__` (add attribute)
- Replaced: `TrainRunner._propose_changes` (or the proposal generation logic)

### 4. **Interface**

#### **TrustRegionConstraint Class**
```python
class TrustRegionConstraint:
    """Manages trust regions for hyperparameter exploration."""
    
    def __init__(
        self,
        initial_radii: dict[str, float],
        contraction_factor: float = 0.8,
        expansion_factor: float = 1.2,
        min_radius: dict[str, float] | None = None,
        max_radius: dict[str, float] | None = None,
        success_threshold: float = 0.0,  # bpb improvement to count as success
        window_size: int = 3,  # number of iterations to evaluate success rate
    ):
        """
        Args:
            initial_radii: Dict mapping hyperparameter names to initial trust radii.
                          Example: {"EMBEDDING_LR": 0.1, "BATCH_SIZE": 32}
            contraction_factor: Multiply radius by this when region fails (0-1)
            expansion_factor: Multiply radius by this when region succeeds (>1)
            min_radius: Minimum allowed radii (prevents collapse to zero)
            max_radius: Maximum allowed radii (prevents unbounded expansion)
            success_threshold: Minimum bpb improvement to count as "successful" step
            window_size: How many recent iterations to consider for success rate
        """
        # Implementation details below
```

#### **Key Methods**
```python
def constrain_proposal(
    self,
    current_params: dict[str, float],
    proposed_params: dict[str, float],
) -> dict[str, float]:
    """Clip proposed parameters to stay within trust region of current best."""
    # Returns constrained parameters
    
def update_region(
    self,
    current_params: dict[str, float],
    new_params: dict[str, float],
    improvement: float,
    iteration: int,
) -> None:
    """Update trust radii based on success/failure of last step."""
    
def get_region_status(self) -> dict:
    """Return current radii and success statistics for logging."""
```

### 5. **Step-by-step Logic**

#### **A. TrustRegionConstraint.constrain_proposal()**
```
1. For each hyperparameter in proposed_params:
   a. Get current value from current_params
   b. Get current radius for this parameter
   c. Calculate allowed range: [current - radius, current + radius]
   d. For bounded parameters (like dropout 0-1), clip range further
   e. Clip proposed value to allowed range
   f. For log-scale parameters (learning rates):
      - Convert to log space: log_current = log10(current)
      - Apply radius in log space: [log_current - radius, log_current + radius]
      - Convert back: 10^clipped_log_value
2. Return constrained dictionary
```

#### **B. TrustRegionConstraint.update_region()**
```
1. Store (improvement, iteration) in history buffer (max size = window_size)
2. Calculate recent success rate: % of improvements > success_threshold
3. For each parameter:
   a. If success_rate > 0.66:  # Region is working well
      - new_radius = current_radius * expansion_factor
      - Apply max_radius ceiling if defined
   b. Else if success_rate < 0.33:  # Region is failing
      - new_radius = current_radius * contraction_factor
      - Apply min_radius floor if defined
   c. Else:  # Mixed results, keep radius unchanged
4. Log radius changes if they occurred
```

#### **C. Integration with Proposal Generation**
```
In TrainRunner._generate_proposals() or equivalent:
1. Get current best hyperparameters from elite_pool or current_code
2. LLM generates raw proposals as before
3. For each proposal:
   a. Extract proposed hyperparameters
   b. Pass through trust_region_constraint.constrain_proposal()
   c. Use constrained values in final proposal
4. After training/evaluation:
   a. Calculate improvement vs previous best
   b. Call trust_region_constraint.update_region() with improvement
```

### 6. **Integration Points**

#### **Modified TrainRunner.__init__()**
```python
def __init__(self, ...):
    # Existing code...
    
    # Improvement 17: Trust region constraint
    self.trust_region = TrustRegionConstraint(
        initial_radii={
            # Conservative initial radii based on observed sensible ranges
            "EMBEDDING_LR": 0.15,      # ~50% of typical range (0.001-1.0 in log space)
            "BATCH_SIZE": 64,          # ~25% of typical range (32-512)
            "DROPOUT": 0.1,            # 20% of range (0.0-0.5)
            "LR_DECAY_GAMMA": 0.1,     # 20% of range (0.5-1.0)
            "WEIGHT_DECAY": 1e-5,      # Log scale: 1e-6 to 1e-3
        },
        contraction_factor=0.7,
        expansion_factor=1.3,
        min_radius={
            "EMBEDDING_LR": 0.02,      # Minimum meaningful change in log space
            "BATCH_SIZE": 8,           # Minimum batch size change
            "DROPOUT": 0.02,           # Minimum dropout change
        },
        success_threshold=-0.0005,     # Need at least 0.0005 bpb improvement
        window_size=4,
    )
```

#### **Modified Proposal Handling (in run_iteration or helper)**
```python
def _constrain_proposals(self, proposals: list[Proposal], current_best: dict) -> list[Proposal]:
    """Apply trust region constraints to all proposals."""
    constrained_proposals = []
    for prop in proposals:
        # Extract hyperparameters from proposal text
        proposed_params = self._extract_hyperparams_from_proposal(prop.text)
        
        # Apply trust region constraint
        constrained_params = self.trust_region.constrain_proposal(
            current_params=current_best,
            proposed_params=proposed_params,
        )
        
        # Update proposal text with constrained values
        constrained_text = self._apply_params_to_proposal(
            prop.text, 
            constrained_params
        )
        constrained_proposals.append(
            Proposal(text=constrained_text, metadata=prop.metadata)
        )
    
    return constrained_proposals
```

#### **Updated in run_iteration()**
```python
def run_iteration(self, iteration: int) -> TrainResult:
    # ... existing code ...
    
    # Get current best parameters (from elite pool or current code)
    current_best = self._get_best_hyperparams()  # New helper method
    
    # Generate proposals as before
    raw_proposals = self._generate_proposals(current_config, iteration)
    
    # Apply trust region constraints
    constrained_proposals = self._constrain_proposals(raw_proposals, current_best)
    
    # Continue with constrained_proposals instead of raw_proposals
    # ... rest of existing code ...
    
    # After evaluation:
    improvement = previous_best_bpb - result.bpb if result.bpb else 0
    
    # Update trust region
    self.trust_region.update_region(
        current_params=current_best,
        new_params=self._extract_hyperparams(result.code),
        improvement=improvement,
        iteration=iteration,
    )
    
    # Log region status
    if iteration % 2 == 0:  # Every other iteration
        status = self.trust_region.get_region_status()
        logger.info(f"Trust region status: {status}")
```

#### **New Helper Method**
```python
def _get_best_hyperparams(self) -> dict[str, float]:
    """Extract hyperparameters from the best known configuration."""
    if self.elite_pool.best():
        best_code = self.elite_pool.best().code
    else:
        best_code = self.current_code
    
    return self._extract_hyperparams(best_code)
```

### **Critical Implementation Notes**

1. **Parameter Scaling**: 
   - Log-scale params (LR, weight decay): radii are in log10 space
   - Linear params (batch size): radii are absolute values
   - Bounded params (dropout): radii are clipped to valid range

2. **Initial Radius Calibration**:
   - Start conservatively (20-30% of plausible range)
   - Allow expansion if region is successful
   - Contract quickly if failing

3. **Fallback Behavior**:
   - If trust region collapses too much (all radii < min), trigger exploration phase
   - Consider occasional "region reset" every N iterations (tie to exploration budget)

4. **Logging**:
   - Log radius changes and success rates
   - Visualize trust region evolution over iterations

This implementation directly addresses the "wild swings" pathology while maintaining the existing proposal-generation infrastructure. The trust region acts as a stabilizing filter on LLM proposals, preventing erratic jumps while allowing calibrated local exploration.