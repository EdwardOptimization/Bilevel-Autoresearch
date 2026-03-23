Based on the trace, the optimizer is stuck in a local search pattern around `EMBEDDING_LR` (0.15-0.18) and `BATCH_SIZE`, discarding most proposals. The existing mechanisms (elite pool, crossover, plateau detection) are not preventing this. The core failure is **inefficient exploration and poor credit assignment**—the LLM lacks a clear model of the loss landscape to propose high-value changes.

Here are 4 mechanism improvements to address this.

---

### **Hypothesis 1: Guided Random Search with Adaptive Resource Allocation**
1.  **Domain**: Hyperparameter optimization (Successive Halving, Hyperband), Multi-armed Bandits.
2.  **Core idea**: Replace one LLM candidate with a **randomly sampled configuration from a promising subspace** (defined by elite pool ranges), and allocate more training steps to promising candidates early via a **low-fidelity quick test**.
3.  **Implementation target**: Modify `_generate_candidates` in `SearchState` to include a random candidate. Extend the `quick-test` to run for a variable, short duration (e.g., 5% of total steps) and use its loss trend to decide whether to commit to a full run.
4.  **Why it addresses the bottleneck**: The LLM is myopically tweaking the last changed parameter. A random candidate within proven-good ranges (e.g., `LR ∈ [0.1, 0.2]`, `BATCH_SIZE ∈ [2**18, 2**20]`) provides systematic exploration without LLM bias. The adaptive quick test prevents wasting full runs on clearly bad directions (like the disastrous batch size increases in iter 8-9) by giving early feedback.
5.  **Implementation complexity**: 3 (Moderate). Requires extending the quick-test logic to be a proper low-fidelity evaluation and integrating a random sampler that respects validated parameter ranges.
6.  **Risk of regressions**: Medium. The random candidate could be worse than an LLM candidate, and the quick test adds overhead. Mitigated by only replacing one candidate slot and making the quick test short.

---

### **Hypothesis 2: Surrogate Model for Proposal Pruning**
1.  **Domain**: Bayesian Optimization (BO), Surrogate Modeling.
2.  **Core idea**: Train a simple, lightweight surrogate model (e.g., linear regression or Gaussian Process on key parameters) on all historical `(config, loss)` data to **predict the outcome of proposed candidates**, and prune those predicted to be poor before any training.
3.  **Implementation target**: Add a `SurrogateModel` class updated after each iteration. Integrate it into `_propose` to score and filter the LLM's 3 candidates (or the random+crossover set), presenting only the top 1-2 predicted performers for the final selection.
4.  **Why it addresses the bottleneck**: The LLM's reasoning is heuristic and often wrong (e.g., insisting on larger batch size). A data-driven model learns the actual response surface from all attempts, identifying that `BATCH_SIZE > 2**19` consistently hurts, regardless of LR. This prevents the LLM from repeatedly proposing known bad changes.
5.  **Implementation complexity**: 4 (High). Requires implementing and maintaining a surrogate model, feature extraction from configs, and robust integration into the proposal loop. Start simple (linear model on normalized parameters).
6.  **Risk of regressions**: High. A poorly trained model could prune good candidates. Mitigate by using it only as a ranking filter (not an absolute gate), keeping at least one candidate, and having a fallback to random if model confidence is low.

---

### **Hypothesis 3: Directional Momentum with Gradient Estimation**
1.  **Domain**: Optimization Theory (Gradient Descent), Derivative-Free Optimization.
2.  **Core idea**: Explicitly estimate a **pseudo-gradient** for each hyperparameter by comparing losses between similar configs in the elite pool, and **bias proposals toward moving opposite to the estimated gradient** (toward lower loss).
3.  **Implementation target**: Enhance the existing "Momentum tracking" (Improvement #6) in `SearchState`. Add a method `_estimate_gradient(elite_pool)` that computes finite differences for each parameter. Inject the estimated descent direction (e.g., "EMBEDDING_LR gradient: +0.05 → suggests decreasing LR") into the LLM proposal prompt.
4.  **Why it addresses the bottleneck**: The LLM identifies monotonic patterns (e.g., "lower LR helps") but inefficiently performs coordinate search. A pseudo-gradient points directly toward the estimated steepest descent, accelerating convergence. It would clearly signal that increasing `BATCH_SIZE` has a positive gradient (increases loss), discouraging further attempts.
5.  **Implementation complexity**: 2 (Low). Builds on existing momentum and elite pool infrastructure. The core addition is a gradient calculation and a prompt template update.
6.  **Risk of regressions**: Low. It's an informative signal, not a hard constraint. The LLM can ignore it if evidence contradicts.

---

### **Hypothesis 4: Automated Multi-Parameter Sensitivity Analysis**
1.  **Domain**: Design of Experiments (DOE), Active Learning.
2.  **Core idea**: Every `N` iterations (or upon stagnation), **automatically run a tiny, designed experiment** (e.g., a 2-level fractional factorial design) that varies multiple parameters simultaneously over a short training horizon to identify significant interactions and main effects.
3.  **Implementation target**: Add a `SensitivityAnalyzer` that, when triggered by plateau detection, generates a small set (e.g., 4) of configs forming a DOE, runs quick tests on them, analyzes results, and summarizes findings (e.g., "LR-BATCH interaction strong") for the LLM.
4.  **Why it addresses the bottleneck**: The LLM changes one parameter at a time, missing interactions (e.g., the optimal `LR` may depend on `BATCH_SIZE`). The automated DOE efficiently probes the joint space, providing the LLM with causal insights it cannot glean from sequential changes, breaking unproductive single-parameter tuning loops.
5.  **Implementation complexity**: 4 (High). Requires designing experiments, managing multiple quick-test runs concurrently, and a statistical analysis module. High upfront cost.
6.  **Risk of regressions**: Medium. The short evaluations may be noisy, leading to incorrect conclusions. Mitigate by using larger step differences for the DOE and triggering it only during clear stagnation.

---

### **Recommended Implementation Order**
1.  **Start with #3 (Directional Momentum)**: Low complexity, builds on existing systems, directly addresses the credit assignment problem for single parameters. Likely to reduce obvious bad proposals quickly.
2.  **Then implement #1 (Guided Random Search)**: Provides a robust exploration baseline, reducing reliance on the LLM's sometimes flawed reasoning. The adaptive quick test is a powerful complement.
3.  **Later, consider #2 (Surrogate Model)**: Once sufficient data is collected (~30+ iterations), this can become the most intelligent filter, but requires careful implementation to avoid regressions.
4.  **Reserve #4 (Sensitivity Analysis)**: A powerful but complex tool for breaking out of deep stagnation. Implement if the simpler mechanisms prove insufficient for escaping strong local optima.

This combination shifts the inner loop from **reactive, heuristic prompting** toward a **hybrid system** that uses the LLM for creative proposal generation, but constrains and guides it with data-driven signals and systematic exploration algorithms.