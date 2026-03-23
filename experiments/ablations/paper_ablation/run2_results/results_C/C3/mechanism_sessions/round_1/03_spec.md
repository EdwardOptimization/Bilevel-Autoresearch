## Implementation Specification

### 1. **Mechanism name**
`bayesian_optimizer_guided_search`

### 2. **Implementation strategy**
`new_helper_class` - Create a BayesianOptimizer class that handles the optimization logic, plus modifications to TrainRunner to integrate it.

### 3. **Target**
- New class: `BayesianOptimizer`
- Modified in TrainRunner: `__init__`, `run_iteration`, `_propose_changes` (to be replaced)

### 4. **Interface**

#### BayesianOptimizer class:
```python
class BayesianOptimizer:
    """Bayesian optimization with constrained action space for hyperparameter tuning."""
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        discrete_params: Dict[str, List[float]],
        init_samples: int = 5,
        exploration_weight: float = 0.1,
        max_consecutive_failures: int = 3
    ):
        """
        Args:
            param_bounds: Continuous parameter bounds, e.g., {"lr": (1e-5, 1e-2)}
            discrete_params: Discrete parameter options, e.g., {"optimizer": ["adam", "sgd"]}
            init_samples: Number of random samples before using GP
            exploration_weight: Weight for acquisition function (higher = more exploration)
            max_consecutive_failures: Reset to random search after this many failures
        """
        
    def propose_next(
        self,
        history: List[Dict[str, Any]],
        current_best: float,
        llm_insights: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Propose next hyperparameter configuration.
        
        Args:
            history: List of dicts with keys: "params", "score", "iteration"
            current_best: Best score seen so far
            llm_insights: Optional LLM interpretation of patterns
            
        Returns:
            Dict with parameter changes to apply
        """
        
    def register_result(
        self,
        params: Dict[str, Any],
        score: float,
        success: bool
    ) -> None:
        """Register the outcome of a proposed configuration."""
```

#### Modified TrainRunner methods:
```python
# In __init__:
self.optimizer = BayesianOptimizer(
    param_bounds=self.search_config.param_bounds,
    discrete_params=self.search_config.discrete_params,
    init_samples=3,
    exploration_weight=0.15,
    max_consecutive_failures=2
)

# New method signature:
def _generate_optimizer_proposal(
    self,
    iteration: int,
    current_config: Dict[str, Any],
    llm_insights: str
) -> Dict[str, Any]:
    """Generate proposal using Bayesian optimizer with LLM insights."""
```

### 5. **Step-by-step logic**

#### BayesianOptimizer.propose_next():
1. **Input validation**:
   - Check if history has at least `init_samples` successful samples
   - Validate current_best is a float
   - Parse llm_insights for any constraints (e.g., "avoid learning rates > 0.001")

2. **If insufficient data**:
   - Generate random sample within bounds
   - For discrete params, choose uniformly at random
   - Ensure sample differs from recent failures (last 3 attempts)

3. **Gaussian Process proposal** (when enough data):
   - Extract features: normalize continuous params to [0, 1]
   - One-hot encode discrete params
   - Fit GP with RBF kernel + white noise
   - Use Upper Confidence Bound (UCB) acquisition:
     ```
     acquisition = mean_prediction + exploration_weight * std_prediction
     ```
   - Add LLM-derived constraints as penalty terms:
     - If LLM says "lower LR helps", penalize high LR proposals
     - If LLM identifies failure pattern, penalize similar configurations

4. **Generate candidate**:
   - Sample 100 random points within bounds
   - Select top 5 by acquisition score
   - Apply diversity filter: reject too-similar to recent attempts
   - Return best candidate

5. **Failure recovery**:
   - Track consecutive failures
   - After `max_consecutive_failures`, revert to random search for 2 iterations
   - Gradually reintroduce GP with increased exploration weight

#### TrainRunner._generate_optimizer_proposal():
1. **Extract LLM insights**:
   - Call LLM with prompt: "Summarize patterns from last 5 runs in one sentence"
   - Parse response for actionable constraints

2. **Get optimizer proposal**:
   - Call `self.optimizer.propose_next()` with:
     - History from `self.trace.get_recent(10)`
     - Current best score from `self.trace.best_score`
     - LLM insights string

3. **Convert to code changes**:
   - Map parameter dict to AST modifications
   - Only change 1-2 parameters per iteration (prioritize by expected improvement)
   - Use predefined step sizes: ±10%, ±25%, ±50% of current value

4. **Validate changes**:
   - Check against crash memory
   - Ensure within search config bounds
   - Apply step size constraints

### 6. **Integration points**

#### In TrainRunner.__init__:
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Bayesian Optimizer
    self.optimizer = BayesianOptimizer(
        param_bounds={
            "lr": (1e-5, 1e-2),
            "batch_size": (8, 256),
            "weight_decay": (0.0, 0.1)
        },
        discrete_params={
            "optimizer": ["adam", "sgd", "adamw"],
            "scheduler": ["none", "cosine", "reduce_on_plateau"]
        },
        init_samples=3,
        exploration_weight=0.15,
        max_consecutive_failures=2
    )
    
    # Remove LLM proposal generation flags
    self.use_llm_for_proposals = False  # LLM only interprets, doesn't propose
```

#### In TrainRunner.run_iteration():
```python
def run_iteration(self, iteration: int) -> TrainResult:
    # ... existing setup code ...
    
    # REPLACE LLM proposal generation with:
    
    # 1. Get LLM insights on recent patterns
    llm_insights = self._get_llm_insights(iteration)
    
    # 2. Generate optimizer proposal
    proposal = self._generate_optimizer_proposal(
        iteration=iteration,
        current_config=current_config,
        llm_insights=llm_insights
    )
    
    # 3. Apply constrained changes (max 2 params)
    modified_code = self._apply_constrained_changes(
        current_code=self.current_code,
        proposal=proposal,
        max_changes=2
    )
    
    # 4. Quick test (existing logic)
    quick_test_passed = self._quick_test(modified_code, iter_dir)
    
    # 5. If passes, run full training
    if quick_test_passed:
        result = self._train_full(modified_code, iter_dir)
        
        # Register result with optimizer
        self.optimizer.register_result(
            params=proposal,
            score=result.bpb,
            success=result.bpb < current_best * 1.1  # Allow 10% regression
        )
    
    # ... rest of existing logic ...
```

#### New helper method:
```python
def _get_llm_insights(self, iteration: int) -> str:
    """Get LLM interpretation of recent patterns."""
    if iteration < 3:
        return "Initial exploration phase"
    
    recent = self.trace.get_last_n(5)
    prompt = f"""
    Analyze these recent training results and identify ONE key pattern:
    {recent}
    
    Respond with a single sentence like "Lower learning rates perform better" 
    or "No clear pattern yet".
    """
    
    response = self.client.complete(prompt, max_tokens=50)
    return response.strip()
```

#### New constraint application:
```python
def _apply_constrained_changes(
    self,
    current_code: str,
    proposal: Dict[str, Any],
    max_changes: int = 2
) -> str:
    """Apply changes with constraints."""
    # Sort by expected improvement (from optimizer)
    sorted_params = sorted(
        proposal.items(),
        key=lambda x: abs(x[1] - current_value) / current_value,
        reverse=True
    )
    
    # Take top max_changes
    changes_to_apply = dict(sorted_params[:max_changes])
    
    # Apply with AST modification
    return self._modify_hyperparams(current_code, changes_to_apply)
```

This specification replaces the LLM's proposal generation with a Bayesian optimizer while keeping the LLM in the loop for pattern interpretation, addressing the core misalignment while maintaining interpretability.