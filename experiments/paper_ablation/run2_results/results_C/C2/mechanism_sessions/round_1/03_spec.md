# Implementation Specification: Multi-Scale Bandit Proposal Generator

## 1. **Mechanism name**: `multi_scale_bandit_proposer`

## 2. **Implementation strategy**: `new_helper_class`

## 3. **Target**: New class `MultiScaleBanditProposer` + modifications to `TrainRunner.__init__` and `run_iteration`

## 4. **Interface**:

### Class Definition:
```python
class MultiScaleBanditProposer:
    """Bandit system for selecting proposal scales based on information gain."""
    
    def __init__(
        self,
        scale_levels: list[str] = None,
        initial_exploration_bias: float = 0.7,
        exploration_decay: float = 0.95,
        min_exploration: float = 0.1,
        optimistic_init_value: float = 2.0
    ):
        """
        Args:
            scale_levels: List of scale identifiers ['tiny', 'small', 'medium', 'large', 'xlarge']
            initial_exploration_bias: Initial probability of choosing exploration over exploitation
            exploration_decay: Multiplicative decay per iteration for exploration bias
            min_exploration: Minimum exploration probability
            optimistic_init_value: Initial Q-value for untried actions (optimistic initialization)
        """
        
    def select_scale(
        self,
        current_config: dict,
        iteration: int,
        recent_performance: list[float] = None
    ) -> tuple[str, dict]:
        """
        Select a scale level for the next proposal.
        
        Returns:
            Tuple of (scale_level, scale_parameters)
        """
        
    def update_reward(
        self,
        scale_level: str,
        was_accepted: bool,
        information_gain: float,
        performance_change: float = None
    ):
        """
        Update bandit Q-values based on outcome.
        
        Args:
            scale_level: The scale that was used
            was_accepted: Whether proposal was accepted
            information_gain: Metric for how much we learned (0-1)
            performance_change: Change in bpb (positive = improvement)
        """
        
    def get_scale_parameters(self, scale_level: str) -> dict:
        """Get mutation parameters for a given scale level."""
        
    def _calculate_information_gain(
        self,
        scale_level: str,
        was_accepted: bool,
        performance_change: float = None
    ) -> float:
        """Calculate information gain metric (0-1)."""
```

## 5. **Step-by-step logic**:

### 5.1 `__init__` method:
```
1. Set default scale_levels: ['tiny', 'small', 'medium', 'large', 'xlarge']
2. Initialize Q-values dict: each scale gets optimistic_init_value
3. Initialize visit_counts dict: all scales start at 0
4. Store exploration parameters
5. Define scale_parameters mapping:
   - tiny:    max_delta=0.01,  num_changes=1-2,   allow_new_params=False
   - small:   max_delta=0.05,  num_changes=2-3,   allow_new_params=False  
   - medium:  max_delta=0.15,  num_changes=3-4,   allow_new_params=True
   - large:   max_delta=0.3,   num_changes=4-5,   allow_new_params=True
   - xlarge:  max_delta=0.5,   num_changes=5-7,   allow_new_params=True
```

### 5.2 `select_scale` method:
```
1. Calculate current exploration probability:
   p_explore = max(min_exploration, 
                   initial_exploration_bias * (exploration_decay ** iteration))
   
2. With probability p_explore (exploration phase):
   a. If any scale has visit_count == 0:
      - Return the least-tried scale
   b. Else:
      - Use UCB1 formula: argmax[ Q(s) + sqrt(2 * ln(total_visits) / visits(s)) ]
      - Return scale with highest UCB score
   
3. Else (exploitation phase):
   a. Return scale with highest Q-value
   
4. Return (selected_scale, scale_parameters[selected_scale])
```

### 5.3 `update_reward` method:
```
1. Increment visit_count for this scale
2. Calculate base_reward:
   - If accepted: base_reward = 1.0
   - Else: base_reward = 0.2 (partial credit for exploration)
   
3. Calculate information_gain_bonus:
   - If scale_level in ['large', 'xlarge']: bonus_multiplier = 1.5
   - Else if not accepted: bonus_multiplier = 1.2
   - Else: bonus_multiplier = 1.0
   
4. If performance_change is not None:
   - Add normalized performance change: performance_bonus = performance_change * 0.5
   - Clamp to [-0.5, 0.5]
   
5. total_reward = base_reward * bonus_multiplier + performance_bonus
   Clamp to [0, 2.0]
   
6. Update Q-value using moving average:
   Q(s) = Q(s) + (1 / visits(s)) * (total_reward - Q(s))
```

### 5.4 `_calculate_information_gain` method:
```
1. If was_accepted:
   - gain = 0.3 (we learned this scale can produce good changes)
   
2. Else (discarded):
   - If scale in ['large', 'xlarge']: gain = 0.8 (learned about boundaries)
   - Else: gain = 0.5 (learned this scale doesn't work here)
   
3. Adjust based on performance_change if available:
   - Large negative change (divergence): gain += 0.2
   - Small change: gain += 0.1
   
4. Return min(gain, 1.0)
```

## 6. **Integration points**:

### 6.1 Modify `TrainRunner.__init__`:
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Improvement 17: Multi-scale bandit proposer
    self.bandit_proposer = MultiScaleBanditProposer(
        initial_exploration_bias=0.8,  # Start with heavy exploration
        exploration_decay=0.97,         # Slow decay
        min_exploration=0.15           # Always keep some exploration
    )
    
    # Track recent performance for bandit context
    self._recent_bpb_changes = []      # Store last 5 bpb changes
    self._current_scale = None
```

### 6.2 Modify `run_iteration` (proposal phase):
```python
def run_iteration(self, iteration: int) -> TrainResult:
    # ... existing code until proposal generation ...
    
    # Get scale from bandit
    scale_level, scale_params = self.bandit_proposer.select_scale(
        current_config=current_config,
        iteration=iteration,
        recent_performance=self._recent_bpb_changes[-5:] if self._recent_bpb_changes else None
    )
    self._current_scale = scale_level
    
    logger.info(f"[Iter {iteration}] Using {scale_level} scale: {scale_params}")
    
    # Pass scale parameters to LLM prompt
    prompt = self._build_proposal_prompt(
        current_code=self.current_code,
        current_config=current_config,
        trace=self.trace,
        scale_params=scale_params,  # NEW: Include scale guidance
        # ... other prompt components ...
    )
    
    # ... rest of proposal and training ...
```

### 6.3 Modify acceptance logic in `run_iteration`:
```python
# After evaluating the proposal (in run_iteration):

# Calculate information gain
info_gain = self.bandit_proposer._calculate_information_gain(
    scale_level=self._current_scale,
    was_accepted=accepted,
    performance_change=(prev_bpb - new_bpb) if has_prev_bpb else None
)

# Update bandit with outcome
self.bandit_proposer.update_reward(
    scale_level=self._current_scale,
    was_accepted=accepted,
    information_gain=info_gain,
    performance_change=(prev_bpb - new_bpb) if has_prev_bpb else None
)

# Track performance for context
if has_prev_bpb:
    self._recent_bpb_changes.append(prev_bpb - new_bpb)
    if len(self._recent_bpb_changes) > 5:
        self._recent_bpb_changes.pop(0)
```

### 6.4 Add to `_build_proposal_prompt` method:
```python
def _build_proposal_prompt(self, scale_params: dict, ...) -> str:
    """Build prompt with scale guidance."""
    prompt = f"""... existing prompt ...
    
    SCALE GUIDANCE (CRITICAL):
    - Make approximately {scale_params['num_changes']} changes
    - Each parameter change should be up to {scale_params['max_delta'] * 100}% different
    - {'You may introduce NEW parameters if helpful' if scale_params['allow_new_params'] else 'Only modify existing parameters'}
    
    For this {'LARGE' if scale_params['max_delta'] > 0.2 else 'MODERATE' if scale_params['max_delta'] > 0.1 else 'SMALL'} scale exploration:
    {"- Be bold and explore significantly different values" if scale_params['max_delta'] > 0.2 else "- Make meaningful but reasonable changes"}
    """
    return prompt
```

### 6.5 Add debugging/logging:
```python
def _log_bandit_stats(self):
    """Log current bandit state for debugging."""
    stats = self.bandit_proposer.get_stats()  # Add this method to return Q-values and counts
    logger.debug(f"Bandit stats: {stats}")
    # Log every 5 iterations or when scale changes dramatically
```

## Key Design Decisions:

1. **Optimistic Initialization**: All scales start with Q=2.0 (above max reward of 2.0) to ensure all get tried
2. **Information Gain Reward**: Large scales get bonus even when discarded (learning boundaries)
3. **Slow Exploration Decay**: Maintain 15% minimum exploration to avoid premature convergence
4. **Scale-Specific Parameters**: Each scale defines concrete mutation bounds for the LLM
5. **Integration with Existing**: Works alongside step_calibrator and momentum tracker

This implementation directly addresses the "tiny tweaks" problem by forcing diversity in proposal scales while rewarding exploration, not just immediate success.