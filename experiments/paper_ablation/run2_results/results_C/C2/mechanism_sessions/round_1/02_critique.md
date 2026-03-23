## Critical Review of Proposed Mechanisms

### **Hypothesis 1: Gradient-Informed Proposal via Surrogate Model**

1. **Most likely failure mode**: The surrogate model will overfit to the tiny elite pool (typically 5-10 points in high-dimensional space), producing hallucinated gradients that point toward random noise rather than true improvement directions. In high dimensions, quadratic models require O(d²) samples for reliable fits—we have O(d) at best.

2. **Implementation trap**: Computing meaningful gradients in a mixed discrete-continuous space with categorical parameters (attention patterns, optimizer choices). The gradient "increasing MATRIX_LR while decreasing EMBEDDING_LR" is meaningless when those parameters have different scales, constraints, and interactions with categorical variables.

3. **Evidence from trace**: The trace shows diminishing returns, not necessarily gradient-obvious regions. The elite pool likely contains non-monotonic, rugged relationships—exactly where local gradient estimates fail. No evidence that smooth quadratic approximations match this space.

4. **Score**: Impact (4) × Feasibility (2) ÷ Complexity (4) = **2.0**  
   *High risk of misleading signals with current data scales.*

---

### **Hypothesis 2: Multi-Scale Step Sizes with Bandit Allocation**

1. **Most likely failure mode**: The bandit will rapidly converge to "small" step sizes because they have higher immediate success rates (producing viable configurations), starving exploration of larger jumps needed to escape local optima. This recreates the exact problem it aims to solve.

2. **Implementation trap**: Defining "success" for bandit rewards. Is it acceptance into elite pool? Performance improvement? The bandit's exploration-exploitation tradeoff is highly sensitive to this definition and will collapse without careful tuning.

3. **Evidence from trace**: Clear pattern of small tweaks (iterations 6-8), but forcing large steps could simply produce invalid configurations that waste evaluations. The trace doesn't show whether large steps were tried and failed, or never tried.

4. **Score**: Impact (3) × Feasibility (4) ÷ Complexity (2) = **6.0**  
   *Simple but risks reinforcing existing biases.*

---

### **Hypothesis 3: Constraint-Aware Crossover with Directed Mutation**

1. **Most likely failure mode**: Directed mutation reduces genetic diversity, causing premature convergence. If all crossovers are biased toward the current "promising direction," the population collapses to a single region of parameter space, eliminating the exploration benefits of crossover.

2. **Implementation trap**: Handling constraints during mutation—parameters have bounds, dependencies (e.g., LR schedules must be monotonic), and discrete categories. A "small mutation biased toward promising directions" often produces invalid configurations when applied naively.

3. **Evidence from trace**: Current crossover is interpolative, but the trace doesn't show whether crossover candidates are being discarded for being too similar or for other reasons. No evidence that directed mutation would help more than random mutation.

4. **Score**: Impact (3) × Feasibility (3) ÷ Complexity (3) = **3.0**  
   *Moderate impact but risks reducing diversity.*

---

### **Hypothesis 4: Failure-Aware Proposal Filtering via Discard Reason Embedding**

1. **Most likely failure mode**: The LLM will learn to avoid entire regions of parameter space based on sparse, noisy failure signals. One "instability from high embedding LR" discard could cause the system to avoid all high embedding LRs, even though the true cause might have been a specific interaction with another parameter.

2. **Implementation trap**: Creating meaningful embeddings from LLM discard reasons, which are often vague, inconsistent, or wrong. "This configuration seems unstable" doesn't cluster meaningfully with "Training loss diverged." Keyword matching will fail on semantic variations.

3. **Evidence from trace**: Repeated discards for similar reasons suggest this could help, but the trace doesn't show whether these are *causal* reasons or post-hoc justifications. LLMs are notoriously bad at explaining why configurations fail.

4. **Score**: Impact (2) × Feasibility (2) ÷ Complexity (5) = **0.8**  
   *High complexity for questionable benefit.*

---

## **Selected**: Hypothesis 2 — with critical modifications

**Why**: It directly addresses the observable "tiny tweaks" pattern with lowest implementation risk, but requires fixing the bandit reward definition to favor *exploration* over immediate success.

**Essential modification**: The bandit should reward *information gain*, not just acceptance. Large steps that get discarded but reveal new landscape characteristics should receive partial credit. Implement an "optimistic initialization" that initially favors large steps, and decay the exploration parameter slowly.

**Alternative consideration**: Before implementing any new mechanism, first analyze whether the problem is truly "proposal quality" or "evaluation noise." The trace shows diminishing returns—this could indicate approaching a local optimum where *no* proposals help, not that proposals are poorly guided. A simpler fix: increase the mutation rate in the automatic crossover candidates, or occasionally inject completely random configurations to test landscape ruggedness.