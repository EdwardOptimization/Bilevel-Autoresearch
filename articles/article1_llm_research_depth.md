# How Automated Research Systems Achieve Depth: An Optimization Perspective

> Subtitle: LLM research capability resembles iterative convergence rather than one-shot correctness -- An internal methodology paper

## Abstract

This document takes an **optimization perspective** to discuss a non-obvious but critically important claim:

> **The recent wave of automated research projects (AutoResearch, AutoResearchClaw, EvoScientist, etc.) all share a common core component: using agents to pursue depth and quality along a single research thread. This component closely corresponds to black-box search, iterative convergence, and proposal quality in optimization theory -- although truly effective systems also depend on evaluators, memory, tool use, problem representation, and prior knowledge injection, the main thread of "high-quality proposal x feedback x iteration" is their shared underlying mechanism.**

If this claim holds, then many conventional understandings of "automated research" need to be revised:

- It is not about making AI "smarter at getting things right on the first try";
- Rather, it is about building a convergence system based on: **high-quality proposal x real feedback x keep/discard x extensive iteration**;
- The differences between projects lie not in who is "more automated," but in who achieves greater depth at which layer.

In other words, the research capability of LLMs is often not reflected in breadth (the ability to get things right in one shot across an infinitely wide task space), but rather in depth (the ability to iteratively refine, correct, and converge toward the objective within a sufficiently specific problem space, following high-quality directions).

**Scope of this article**: This article only discusses depth optimization along a single research thread -- how to achieve depth and quality on a single task. It does not discuss multi-direction parallelism, team layering, or role specialization; those are reserved for the companion article [`agent_team_how_large_projects_emerge.md`](./agent_team_how_large_projects_emerge.md).

In other words, the strength of LLMs often lies not in "getting it perfectly right on the first try," but rather in:

1. Generating a proposal significantly better than random;
2. Reading external feedback;
3. Updating the next-round proposal;
4. Continuously compressing error through keep / discard / rollback;
5. Achieving convergence after sufficient iterations.

This process resembles control theory, black-box optimization, RL / policy search with strong priors, and a form of engineering-level "in-context posterior updating."

The core thesis of this article is:

> **LLM research depth = On a sufficiently narrow problem surface with verifiable feedback, using proposal quality far above random, combined with context update, keep/discard mechanisms, and sufficient iterations, compressing the error multiplier from above 1 to below 1, and continuously driving it toward 0.**

The positioning of this article is:

> **An internal methodology document: it attempts to propose a sufficiently strong and explanatory engineering framework, without disguising itself as a rigorous theoretical proof.**

That is to say, the control theory, RL, Bayesian, and alpha convergence models mentioned in this article are primarily used to help understand system behavior, compare different system designs, and distill actionable research intuitions, rather than claiming the existence of rigorous general theorems.

---

## 1. Two Dimensions of Research Capability: Breadth and Depth

### 1.1 What Is Breadth

Breadth refers to the model's coverage capability across a wide task space, for example:

- Handling many completely different types of problems simultaneously;
- Producing a decent first version when the task definition is still vague;
- Quickly providing a usable framework for unfamiliar domains;
- Offering "seemingly reasonable" action suggestions even without a clear reward signal.

This is "breadth intelligence."

### 1.2 What Is Depth

Depth refers to the model's ability to continuously converge toward the objective along a clear feedback surface on a **specific enough** target, for example:

- Repeatedly optimizing against a fixed metric;
- Continuously narrowing down the localization of a specific bug;
- Iteratively fixing a specific pipeline until convergence;
- Significantly improving results after many rounds of trial and error on a fixed experimental harness.

This is "depth convergence capability."

### 1.3 The Position of This Article

This article argues that the most reliable and engineerable strength of LLMs in research is not breadth, but depth.

That is to say:

- They may not always make the optimal judgment on an infinitely wide problem in one shot;
- But they can often continuously optimize a problem that has been compressed to be sufficiently specific;
- Once there is a real feedback surface and iteration opportunities, they can exhibit clear convergence behavior.

---

## 2. Why This Is Not Pure Random Search

If the system were pure random search, it would encounter classic problems in high-dimensional action spaces:

- Most directions are ineffective;
- Most changes interfere with each other;
- Reward is sparse and noisy;
- An extremely large number of trials are needed to find stable improvement directions.

The key difference with LLMs is that they are not uniform random samplers.

### 2.1 LLM Proposals Have Strong Priors

LLM proposals naturally tend to satisfy:

- **Semantically reasonable**: Actions typically fall near common human engineering/research operations;
- **Locally self-consistent**: The changes they make tend to be internally coherent at the semantic level;
- **Consistent with context**: They reference existing logs, historical failures, and current objectives;
- **Closer to "experience-dense regions"**: They land in higher-value areas compared to random points.

Therefore, it is essentially not:

> random search

But rather more like:

> **prior-guided search**

Or more plainly:

> **high-quality proposal distribution**

### 2.2 Why Being "Slightly Better Than Random" Is Already Profoundly Important

Even if the LLM is only slightly better than random each round, the advantage compounds over multiple iterations:

- Fewer crashes;
- Fewer meaningless actions;
- More hits on effective directions;
- More utilization of historical experience;
- More exploration within the correct local space.

If the edge is slightly positive each round, the cumulative effect becomes very large.

This follows the same logic as in quantitative finance: "weak edge x high-frequency repetition = strong outcome."

---

## 3. Why This Process Simultaneously Resembles Control Theory, RL, and Bayesian Inference

### 3.1 Control Theory Perspective: This Is a Feedback Controller

Described in the simplest terms, such systems often behave as:

- Not enough? Push harder;
- Too much? Pull back;
- Crashed? Roll back;
- Working? Keep pushing in this direction.

This is clearly a feedback control structure.

#### Control Theory Mapping

- **Plant**: The task itself (code, experimental system, research pipeline, prompt surface)
- **Observed variable**: Evaluation results, metric changes, errors, logs, latency, cost
- **Control input**: Next-round proposal (modify prompt, modify code, tune hyperparameters, modify pipeline)
- **Error**: Gap between current state and target state
- **Safety loop**: Rollback, discard, freeze, guard

In this sense, it resembles a controller with memory and language priors.

### 3.2 RL / Policy Search Perspective: This Is a Black-Box Policy Optimization Process

Another view is:

- The system proposes an action proposal;
- The environment executes it;
- The environment returns reward / penalty;
- The system adjusts the next-round action based on the reward.

This is already very close to:

- black-box optimization
- bandit
- policy search
- REINFORCE-style trial-and-error

It is not strictly PPO, but the engineering feel is very much like "an extremely thin layer of RL":

- policy = current research strategy / prompt / context state
- action = this round's changes
- reward = evaluation result quality
- update = keep / discard / revert / next proposal

### 3.3 Bayesian Perspective: Context Functionally Resembles an Engineering-Level Posterior Update

From an engineering standpoint, LLM inference **can be analogized as** a form of amortized Bayesian update:

- Pre-training weights = broad prior;
- Current context = new observation;
- Current output distribution = posterior approximation.

Strictly speaking, the Transformer is not explicitly computing Bayes' formula, but it can be understood as:

> **Each new piece of evidence re-weights the existing prior.**

This is why:

- It does not guess from scratch each time;
- It absorbs failure experience;
- It rapidly changes proposal distribution based on recent evidence.

---

## 4. A Simple but Useful Convergence Model

To discuss the relationship between "model capability + number of iterations," we can introduce a minimalist model.

First, an important caveat:

> **This section is not formal convergence theory, but only a coarse-grained engineering model to aid thinking.**

Its purpose is to help determine whether the system is more likely converging, stagnating, or diverging, rather than precisely predicting real system dynamics.

Let:

- `d_t` = error distance from the target at round `t`
- `alpha` = scaling factor for remaining error after this round

Then:

```text
d_(t+1) = alpha * d_t
```

### 4.1 The Meaning of Alpha

- `alpha < 1`: Convergence
- `alpha = 1`: Stagnation
- `alpha > 1`: Divergence

This `alpha` can be understood as:

> **The effective error compression capability per iteration, given the current task complexity, model capability, feedback quality, and harness design.**

### 4.2 An Intuitive Version

This is a very useful engineering analogy:

- Very weak model: Each round might be like multiplying by `1.05`
- Slightly weak model: Might be like multiplying by `1.01`
- Decent model: Might be like multiplying by `0.99`
- Strong model: Might be like multiplying by `0.95`
- Very strong model: Might be like multiplying by `0.80`
- Approaching `0`: Indicates convergence

This judgment, while coarse, captures the essence very well.

### 4.3 Why This Model Is Useful

Because it directly explains two facts:

1. **Strong model capability means far fewer iterations are needed**;
2. **Weak model capability requires massive iteration, and may never grind through.**

For example:

- If `alpha = 0.99`, convergence is very slow;
- If `alpha = 0.95`, significant compression can occur within dozens of rounds;
- If `alpha = 0.80`, a small number of iterations can rapidly converge;
- If `alpha = 1.01`, the more you iterate, the more you drift;
- If `alpha = 1.05`, the system will visibly destabilize.

---

## 5. Where Model Capability Actually Manifests

A common misconception is:

> Model capability = how correct the answer is on a single shot.

But in automated research or automated optimization systems, what actually matters more is:

> **Whether the model can compress the effective error multiplier below 1.**

### 5.1 Proposal Quality Is the Core

A strong model does not necessarily mean it is always correct on the first round;
More importantly, a strong model:

- Proposes fewer obviously bad proposals;
- Better utilizes historical evidence;
- Better compresses broad problems into narrow subproblems;
- Overfits less on noise;
- Exhibits less meaningless drift;
- More easily retains truly effective changes under keep/discard mechanisms.

### 5.2 The Advantage of Strong Models Is Not Absolute Correctness, but a Better Chance at Lower Effective Alpha

More precisely:

> **Under the same task decomposition, same harness, and same feedback conditions, stronger models typically have a better chance of compressing the per-round effective error multiplier lower.**

Important caveats:

- `alpha` is not a property of the model alone;
- It is simultaneously influenced by task difficulty, problem decomposition approach, feedback quality, action surface design, and rollback mechanisms;
- Therefore, "stronger model = lower alpha" is a conditional claim, not a universal law.

Consequently:

- Strong model + few iterations = can still converge quickly;
- Weak model + many iterations = may barely converge;
- Very weak model + no matter how many iterations = still diverges.

---

## 6. Why "Specific Enough" Is a Necessary Condition

This does not hold for all tasks.

The depth capability of LLMs typically only truly activates when the problem has been narrowed to "specific enough."

### 6.1 What Does Specific Enough Mean

Typical characteristics include:

- A clear objective function;
- Comparable feedback;
- A well-defined action surface;
- Rollback / keep / discard mechanisms;
- A relatively stable evaluation protocol;
- A reasonable trial-and-error budget.

### 6.2 Why Things Tend to Fail When the Problem Is Too Broad

If the problem is too broad, the following issues arise:

- Reward is ambiguous;
- Action space is infinitely large;
- Each round's proposals are not comparable;
- Local successes cannot accumulate;
- Error localization is difficult;
- The model gets dragged around by noise.

In such cases, even if the model itself is very strong, it is prone to:

- Appearing smart but not converging;
- Each round "making sense" but producing no net progress;
- Strong breadth, insufficient depth.

### 6.3 Truly Convergent Problems Are Often "Compressed Problems"

Therefore, one of the most important engineering actions is not "directly letting the model solve the problem," but rather:

> **First compressing the problem to a narrow enough, measurable enough, comparable enough subspace.**

The better this step is done, the more likely the system is to converge.

### 6.4 Why AI Must Have Concrete Objectives to Get Things Done

This point is not just engineering experience; it also has strong multi-agent theoretical intuition behind it.

If a system has no concrete objective, no unified reward, and no clear verifier, then multi-agent collaboration easily degenerates into a form of "weak consensus problem":

- Every agent can propose seemingly reasonable opinions;
- But these opinions may not be comparable;
- Without a unified evaluation surface, the system cannot determine who is closer to the objective;
- Even if agents "reach agreement" at the conversation level, this does not mean they have achieved effective coordination toward the same real objective.

This is closely related to the **Byzantine Generals Problem**, which is often invoked as an analogy in multi-agent collaboration:

> **If the system lacks a sufficiently strong shared objective and shared adjudication mechanism, then "everyone is communicating" does not equal "everyone can reliably reach useful consensus."**

Two papers can serve as positive and negative references here.

#### Negative Boundary: *Can AI Agents Agree?*
This work serves as an excellent reminder:
- It focuses not on "whether agents sound alike," but on "whether multiple agents can reliably achieve stable agreement";
- The results show that even in relatively benign settings, valid agreement is not reliable, and further degrades as group size increases;
- Many failures are not due to values being silently tampered with, but rather more mundane **liveness** problems -- timeouts, deadlocks, inability to converge in time.

It provides a negative boundary:

> **Agreement is not a natural emergent capability of multi-agent groups.**

#### Engineering Direction: *Reaching Agreement Among Reasoning LLM Agents*
This other work is more like an engineering prescription. Its core message is not "multiple agents will naturally discuss their way to an answer," but rather:
- Many existing multi-agent orchestration systems rely on fixed round counts, barrier synchronization, and other ad-hoc heuristics;
- These methods waste compute, get slowed down by stragglers, and may terminate prematurely when there is only temporary agreement;
- If reasoning agents are to achieve reliable agreement, it must be treated as a **protocol design problem**, not as a capability that naturally emerges from "chatting a few more rounds."

In other words, it emphasizes:

> **For multi-agent systems to upgrade from "group chat" to "collaborative system," there must first be a shared objective, then a shared adjudication protocol.**

So, why must AI have concrete objectives to succeed? Because concrete objectives provide three critical things:

#### 1. A Shared Coordinate System
Without concrete objectives, each agent may be optimizing different implicit objectives; with concrete objectives, the system at least knows "which direction counts as closer."

#### 2. Comparable Verdicts
Without a unified adjudication surface, agents can only try to persuade each other; with a verifier / benchmark / gate, the system can shift from "debate" to "adjudication."

#### 3. A True Collaboration Foundation
True multi-agent collaboration is not chatting with each other, but rather:
- Dividing labor toward the same objective;
- Using the same standards for acceptance;
- Updating state on the same control plane.

That is to say:

> **Concrete objectives are not an accessory, but a prerequisite for multi-agent systems to upgrade from "group chat" to "collaborative system."**

In other words, the reason LLMs / agents find it easier to achieve depth on specific problems is not only because the problem is narrower, but also because:

- Narrow objectives more easily form a shared objective;
- A shared objective more easily forms a shared verifier;
- A shared verifier is what pulls multi-agent systems back from "Byzantine disagreement risk" to "a convergent closed loop."

---

## 7. Why the Harness Is the Decisive Variable

Model capability is not the only determining factor. Many systems fail not because the model is too weak, but because the harness is too poor.

### 7.1 Convergence Requires Four Foundational Conditions

#### 1. Real Feedback (Reward Fidelity)
Evaluation must be truthful and must not drift.

#### 2. Keep/Discard Must Be Hard
Without hard rollback, the system will mistake noise for progress.

#### 3. The Action Surface Must Be Narrow
If one round makes changes to 20 things simultaneously, it becomes very difficult to identify causality.

#### 4. Sufficient Trial-and-Error Must Be Allowed
Without iteration budget, even the best priors cannot be leveraged.

### 7.2 The Essential Role of the Harness

The true role of the harness is to:

- Compress the problem space;
- Fix the feedback surface;
- Constrain the action surface;
- Provide rollback mechanisms;
- Ensure evaluation consistency;
- Allow small edges to compound.

Therefore, a more accurate expression for final performance should be:

> **effective research performance = model prior x context update x reward fidelity x action narrowing x iteration budget**

---

## 8. Why This Resembles "PID + Random Direction Optimization + Thin RL"

This is not a rigorous mathematical definition, nor does it claim theoretical equivalence with these frameworks; the usage here is primarily:

> **Borrowing the engineering intuitions of these frameworks to help understand what the system resembles at the feedback, search, and update layers respectively.**

### 8.1 Where It Resembles PID

- Current error too large -> increase effort
- Overshot -> reduce effort
- Trend is bad -> change direction
- Crashed -> roll back

#### Intuitive Mapping

- P: Current error drives current action
- I: Historical results accumulate as experiential memory
- D: Trend changes influence the next-round proposal

Of course, it is not classical PID, because:

- The state is not continuous and low-dimensional;
- The action is not a single continuous variable;
- The controller itself is a language model;
- The system is highly nonlinear.

But "the feel of a feedback controller" absolutely holds.

### 8.2 Where It Resembles Random Direction Optimization

It has no true gradient.

What it does is:

- Guess a direction;
- Try it;
- Look at the reward;
- Keep it if it's good;
- Discard it if it's bad.

This is highly similar to:

- hill climbing
- direct search
- bandit exploration
- stochastic search

The key difference is:

> **The proposal is not random noise, but a high-quality direction with semantic priors.**

### 8.3 Where It Resembles "1-Batch PPO / Extremely Thin RL"

The intuition is:

- Each round is like having only one or very few interactions;
- The policy adjusts based on results;
- Reward is sparse but clear;
- Continuous iteration leads to overall convergence.

Strictly speaking, it is not standard PPO, because it lacks:

- clipping objective
- old/new policy ratio
- value function
- advantage estimator
- mini-batch gradient update

But as engineering intuition, it does very much resemble:

> **LLM proposal policy + single-trial reward + keep/discard update**

---

## 9. Why Context Closely Resembles Bayesian Inference, and Why This Matters

A key observation:

> **The LLM's context itself is essentially Bayesian.**

This statement is worth unpacking.

### 9.1 Why This Analogy Holds

Because the role of context is to re-weight the directions the model is most likely to output:

- Which recent attempts failed;
- Which directions were effective;
- Which constraints must not be violated;
- What the current task is actually optimizing;
- Which explanations are now more credible.

With each additional piece of evidence, the model's current proposal distribution shifts.

### 9.2 This Explains Why LLMs Are Much Stronger Than Random

Random search has no in-context posterior updating.

But LLMs have:

- Strong priors;
- Immediate evidence injection;
- Contextual conditioning;
- The ability to immediately correct proposals in the next round.

That is to say, their strength is not "always being correct," but rather:

> **Being able to rapidly change what they try next based on evidence.**

And this is precisely the capability that iterative systems need most.

---

## 10. Why the Number of Iterations Is So Critical

No matter how strong the model is, it cannot finish in one shot.

The real effect comes from:

> **proposal quality x number of iterations**

### 10.1 The Significance of Iteration Is Not Repetition, but Compounding

Each iteration can bring:

- A smaller error space;
- More stable local structure;
- Clearer failure boundaries;
- Higher-quality next-round proposals;
- Less wasted exploration.

If each round compresses the error just a little, the long-term effect is very powerful.

### 10.2 Why Strong Models Can Converge with Fewer Iterations

If the model is strong, `alpha` is lower:

- Proposals are more accurate;
- Evidence utilization is better;
- Rollback is more timely;
- Wasted exploration is less.

This means:

> **Strong models can achieve the same or even stronger convergence with fewer trials.**

### 10.3 Why Weak Models Heavily Depend on Massive Iteration

If the model is weak:

- Proposal marginal advantage is small;
- Drift is larger;
- More likely to get stuck on noise;
- Harder to narrow the problem;
- `alpha` is close to 1, or even greater than 1.

In such cases, iteration becomes "grinding":

- With luck, some convergence can be ground out;
- Without luck, the more you grind, the more you drift.

---

## 11. When Things Fail: Typical Divergence Patterns

If the system exhibits the following conditions, it tends to diverge:

### 11.1 Reward Drift
The model is optimizing a false objective, not the actual objective.

### 11.2 Action Surface Too Wide
One round changes too many things, making it impossible to identify causality.

### 11.3 Keep/Discard Too Soft
Everything can be explained as "somewhat helpful," leading to noise accumulation.

### 11.4 Evidence Cannot Be Written Back to the System
Context does not accumulate stably, causing posterior update failure.

### 11.5 Task Not Specific Enough
Vague objectives cause proposals that look smart but are not comparable.

### 11.6 Evaluation Cost Too High, Too Few Iterations
Even the best proposals cannot compound.

Under these conditions, even if the model appears "very smart," it may still:

- Output many reasonable explanations;
- Produce a great deal of content;
- Yet never truly compress the error.

That is to say:

> **Looking like it's working does not mean it's converging.**

---

## 12. Design Implications for Automated Research Systems

If we translate these insights into system design principles, we arrive at the following conclusions.

### 12.1 Do Not Treat the Agent as a One-Shot Answering Machine

It should be designed as:

- proposal generator
- evidence consumer
- iterative controller
- keep/discard participant

### 12.2 Do Not Pursue "Comprehensive Coverage" First

Priority should be given to:

- very specific target
- stable feedback surface
- narrow action surface
- high iteration frequency

### 12.3 Do Not Let Raw Signals Directly Become Boss-Level Conclusions

Because raw signals are only part of the proposal, not a verified verdict.

### 12.4 What Truly Matters Is the Closed Loop, Not Single-Step Brilliance

Whether a system works depends not primarily on "how brilliant a single model output is," but on:

- Whether proposals can enter the real environment;
- Whether results can be compared;
- Whether errors can be rolled back;
- Whether good directions can be preserved;
- Whether evidence can enter the next round.

### 12.5 A Paradigm Map Using Existing Automated Research Projects

If we look at the recent wave of automated research projects together, most of them are still different instantiations of the same overarching paradigm:

```text
proposal -> run -> evaluate -> keep/discard -> iterate
```

The real differences lie not in whether they are "automated," but in which layer they respectively strengthen.

#### [AutoResearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026): Single-Track Local Optimizer
- Compresses the problem to an extremely narrow experimental surface;
- Uses a fixed time budget for small-step trial and error;
- Continuously optimizes metrics along a single main thread;
- Excels at fast convergence on specific objectives.

Its strength is convergence speed; its weakness is a higher tendency to get stuck in local optima.

#### [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) (AIMing Lab): End-to-End Pipeline + Multi-Batch Search
- Covers the full chain from literature discovery, hypothesis generation, to experiments and paper writing;
- Uses multi-agent debate, multiple candidate hypotheses, and multi-round review to reduce single-path bias;
- Essentially, it is not just about "moving forward," but about preventing the system from prematurely getting stuck in local optima.

Its strength is full-pipeline coverage and multi-branch exploration; its weakness is greater system weight and more components, thus higher dependence on end-to-end feedback quality.

#### [EvoScientist](https://github.com/EvoScientist/EvoScientist): Experimental Notebook + Experience Abstractor
- Not just trial and error, but also writes failed strategies, ineffective directions, and reusable lessons into a memory bank;
- Then feeds this experience back into the next-round proposals;
- Essentially equips the AI with an experimental notebook, and lets it read its own notes and build institutional memory.

Its strength is cross-round, cross-project experience accumulation; its weakness is that if memory abstraction is done poorly, noise can be propagated as experience.

#### ARIS / Orchestra / Research-Claw: Skill Layer and Tool Layer
- They do not necessarily handle the entire pipeline themselves;
- They are more like augmenting agents with Skills, MCP, Tooling, literature, and experimental capabilities;
- They address the question of "what hands does the agent have, how many tools, and can they plug and play."

Their value lies in raising the infrastructure ceiling, but skill packages alone do not automatically equal depth convergence.

#### Dr. Claw: Research IDE / Workbench
- Places greater emphasis on human-AI collaboration, multi-project management, and visual supervision;
- Not extreme unattended autonomy, but rather letting AI handle the grunt work while humans maintain research taste and directional oversight.

Its strength is product experience and multi-project management; its weakness is that autonomous depth may not be the strongest.

So if we compress these trajectories one more level, we arrive at a clearer conclusion:

> **The automated research landscape is not about "who is most automated," but about "who makes which layer the strongest": some strengthen single-track convergence, some strengthen multi-branch search, some strengthen memory evolution, some strengthen skill infrastructure, and some strengthen the human-AI collaboration workbench.**

### 12.6 The Real Question: How to Achieve Depth on a Specific Project or Research Problem

If the projects above have already demonstrated the same overarching paradigm many times, then the next differentiator is not "extending the chain further," but rather:

> **How to achieve depth, stability, and net gains on a specific project, a specific research surface.**

I believe there are at least six key conditions here.

#### 1. First, Narrow the Problem to a Convergent Surface
Do not directly pursue "automated research" or "fully automated from idea to paper"; instead, first compress to:
- A comparable objective;
- A verifiable task surface;
- A local subproblem where single-round actions can be attributed.

The prerequisite for depth is not "big," but "narrow enough to converge."

#### 2. Make Feedback Hard
Without hard feedback, there is no real depth.
The system must face, as much as possible:
- Numerical metrics;
- Structured verifier verdicts;
- Clear keep/discard gates;
- Reproducible, rollback-capable evaluation protocols.

Otherwise, the system is merely outputting a large amount of seemingly reasonable content, without necessarily making real progress.

#### 3. Maximize Proposal Quality
Proposal quality comes from three parts:
- A stronger model;
- Cleaner context organization;
- A better proposal contract.

That is to say, depth is not achieved by model capability alone, but by:

> **model x context x proposal template**

working together.

#### 4. Keep State Management Clean
True depth depends on the separation of three types of state:
- working state: what is currently being tried;
- best-known state: what the current best solution is;
- knowledge state: what the system has truly learned.

If these three are mixed together, the system easily writes noisy experiments as long-term ground truth, or lets working state pollute the best-known state.

#### 5. Use Breadth to Find the Slope, Use Depth to Drill Down
Real research is not a binary choice between breadth and depth.
The more common pattern is:
- In early stages, explore multiple directions in parallel to avoid optimizing on the wrong slope from the start;
- In mid-to-late stages, lock onto a promising branch and drill deep along a single main thread for convergence.

That is to say:

> **Breadth is for finding the surface; depth is for breaking through.**

#### 6. Verifiers Are More Important Than Generators
Many systems focus on stacking generators, but what truly determines whether a system can achieve depth is often the verifier, gate, reviewer, and rollback.

Because the real moat is not "whether more things can be generated," but rather:
- What counts as progress;
- What is noise;
- What must be discarded;
- What can enter the next round;
- What is merely a proxy objective, not the real objective.

So what should ultimately be pursued is not "a model that talks better," but rather:

> **Turning the system into a machine that continuously compresses error on a narrow problem surface.**

### 12.7 A One-Line Summary

If this entire section is compressed into one sentence, it is:

> **The next phase of automated research is not about who makes the chain longer, but about who can achieve true depth convergence on a specific research surface.**

---

## 13. A More Complete Summary Formula

If the entire article is compressed into a single engineering expression, it can be written as:

> **LLM research depth ~ Proposal quality x Contextual posterior update x Reward fidelity x Action narrowing x Iteration budget**

And whether it can truly converge can be assessed using the error multiplier:

```text
d_(t+1) = alpha * d_t
```

Where `alpha` is not determined by the model alone, but is jointly determined by:

- Model capability
- Whether the problem is specific enough
- Whether the harness is reliable
- Whether the feedback is truthful
- Whether keep/discard is hard
- Whether the iteration count is sufficient

The final assessment can be coarsely understood as:

- `alpha < 1`: The system is more likely converging
- `alpha = 1`: The system is more likely stagnating
- `alpha > 1`: The system is more likely diverging

---

## 14. Limitations and Counterexamples

To avoid overstating the framework above, here are several explicit boundary conditions.

### 14.1 This Is Not a Universal Convergence Theorem

The `alpha` model in this article is merely an engineering analogy to aid thinking. It does not imply that real systems necessarily possess a single, stable, observable error scalar, nor that complex research processes can necessarily be accurately described by a single multiplicative model.

### 14.2 "Strong Model" Does Not Guarantee Convergence

A stronger model typically only has a better chance of producing a lower effective error multiplier **under the same task decomposition, same harness, and same feedback quality**; if the reward drifts, the action surface is too wide, or the evaluation protocol is unstable, then even the strongest model may "converge" faster toward the wrong objective.

### 14.3 A Typical Counterexample: Optimizing a Proxy Objective Instead of the Real Objective

For example, an agent in a code repair task continuously optimizes "unit test pass rate," but the test suite's coverage is incomplete; in this case, the system may be better than random each round and appear to be making continuous progress, yet what it converges to is a **proxy objective**, not the real objective. That is to say:

> **Better than random + ability to iterate does not automatically equal useful convergence.**

### 14.4 Breadth and Depth Often Alternate in Real Research

This article emphasizes that LLM capability in the **depth phase** tends to be underestimated.
This does not mean breadth is unimportant, nor that real research relies solely on depth. Real workflows often look more like:

- First, broad exploration to find directions
- Then, deep exploitation to achieve convergence
- When necessary, return to broad exploration to find new surfaces

So this article is not trying to prove "depth > breadth," but rather to demonstrate:

> **On a problem surface that has been narrowed sufficiently and where feedback is sufficiently hard, the depth capability of LLMs can be remarkably strong.**

---

## 15. Final Conclusions

What this article ultimately aims to convey is:

1. **The true strength of LLMs in research is often depth, not breadth.**
2. **This depth capability depends on "the problem being specific enough" rather than "the world being wide enough."**
3. **LLMs are not random search engines, but proposal engines with strong priors and in-context posterior updating.**
4. **When proposal quality is significantly above random, and the harness simultaneously provides keep/discard, rollback, stable feedback, and sufficient iteration budget, the system is more likely to exhibit significant convergence.**
5. **The engineering significance of model capability is not just smarter answers, but rather, under appropriate problem decomposition and appropriate harness, a better chance of compressing the per-round effective error scaling factor `alpha` lower.**
6. **The stronger the model, the fewer iterations typically required; the weaker the model, the more it depends on massive iteration, and it may never enter the true convergence zone.**
7. **Truly usable automated research is not "letting the LLM think about it," but placing the LLM inside a closed loop that can continuously compress error.**

Therefore, the entire conclusion can be compressed into one sentence:

> **Automated research capability is not one-shot brilliance, but the ability of high-quality proposals to continuously converge toward the objective through real feedback and extensive iteration on a specific task.**
