---
name: level2-research
description: >
  Meta-optimize any optimization loop. Reads your search/optimization code,
  diagnoses why it's stuck, and generates new mechanisms as injectable Python
  code — drawn from adjacent fields (combinatorial optimization, bandits,
  evolutionary algorithms, DOE, etc.).

  Works with any loop that does propose → evaluate → keep/discard.

  TRIGGER: "improve the loop", "meta-optimize", "why is my search stuck",
  "generate a new mechanism", "level 2", "bilevel", "break out of plateau",
  "my optimization keeps repeating", "add exploration to my loop",
  "inject a new strategy", "improve search diversity"
---

# Level 2: Meta-Optimize Your Search Loop

Your optimization loop proposes changes and keeps/discards them.
This skill reads your loop's **code** and **trace**, diagnoses structural
bottlenecks, and generates new Python mechanisms to fix them — drawn from
algorithmic domains your loop would never consider on its own.

```
Your loop (Level 1):  propose → evaluate → keep/discard
This skill (Level 2): read code + trace → diagnose → generate mechanism → inject
```

**Core principle:** The same LLM that runs the loop can improve the loop's
structure, if given the right protocol.

---

## When to Use

- Your loop keeps **proposing the same things** (parameter fixation)
- **Discard rate is high** — most proposals make things worse
- You're in a **plateau** — metric stopped improving for many iterations
- You suspect the loop has **blind spots** it can't see on its own
- You want ideas from **other fields** (optimization, evolution, bandits, DOE)
- Any loop that does propose → evaluate → keep/discard, in any language/domain

---

## Phase 1: Understand the Target

### Step 1.1 — Identify the Inner Loop

Ask the user (if not already clear):
1. **Where is the inner loop code?** (the file that runs propose -> evaluate -> keep/discard)
2. **Is there a search trace / experiment log?** (history of proposals and outcomes)
3. **What is the metric?** (what the inner loop optimizes)
4. **What is the current bottleneck?** (repetitive proposals? parameter fixation? crashes?)

If the user doesn't know the bottleneck, that's fine — Round 1 will diagnose it.

### Step 1.2 — Read the Code and Trace

Read the inner loop source code completely. Then read the search trace if available.

Look for:
- **Proposal generation logic** — how are changes proposed?
- **Acceptance criterion** — keep/discard rules
- **State management** — what history is maintained?
- **Structural limitations** — what *can't* the current code do?

Summarize findings before proceeding. The user should confirm the diagnosis.

---

## Phase 2: 4-Round Research Session

Each round is a distinct thinking step. Do NOT skip rounds or combine them.

### Round 1: Explore

Survey mechanisms from adjacent algorithmic domains that could address the
identified bottleneck. Consider at minimum:

| Domain | Example Mechanisms |
|--------|-------------------|
| Combinatorial optimization | Tabu search, simulated annealing, variable neighborhood search |
| Online learning / bandits | UCB, Thompson sampling, EXP3, contextual bandits |
| Design of experiments | Latin hypercube, orthogonal arrays, space-filling designs |
| Evolutionary algorithms | mutation operators, crossover, novelty search, MAP-Elites |
| Bayesian optimization | acquisition functions, surrogate models, expected improvement |
| Reinforcement learning | epsilon-greedy, curiosity-driven exploration, reward shaping |

**Output:** A ranked list of 3-5 candidate mechanisms with:
- Name and source domain
- How it addresses the specific bottleneck
- Rough implementation complexity (low/medium/high)
- Risk assessment (what could go wrong)

### Round 2: Critique

Evaluate each candidate against the observed failure mode:

For each candidate, answer:
1. Does it directly address the diagnosed bottleneck?
2. Can it be implemented with only stdlib dependencies?
3. Will it integrate cleanly with the existing code?
4. What is the worst-case behavior if it fails?

**Output:** Select the single most promising mechanism. Justify the choice.
Explicitly state why the alternatives were rejected.

### Round 3: Specify

Write a precise interface specification for the selected mechanism:

```
Class name: [e.g., TabuSearchManager]
Constructor args: [e.g., tenure: int = 5, distance_thresholds: dict = None]
Key methods:
  - method_name(args) -> return_type: one-line description
  - ...
Integration points:
  - Where in the existing code does this get called?
  - What data does it need from the existing loop?
  - What does it return/modify?
State:
  - What internal state does it maintain?
  - When is state reset?
```

**Output:** The complete interface spec. Get user confirmation before proceeding.

### Round 4: Generate

Write complete, runnable Python code implementing the specified mechanism.

**Requirements:**
- stdlib only (no sklearn, no scipy, no external deps unless user confirms availability)
- Include type hints and a module-level docstring
- The code must be copy-pasteable into the target codebase
- Include the modifications needed in the existing code to call the new mechanism
- Add inline comments explaining non-obvious logic

**Output:**
1. The new mechanism module (complete Python file)
2. A diff or patch showing how to integrate it into the existing inner loop code
3. A one-paragraph summary of what changed and why

---

## Phase 3: Validate and Inject

### Step 3.1 — Validate

Before activating the generated code:

1. **Syntax check**: Run `python -c "import ast; ast.parse(open('mechanism.py').read())"` or equivalent
2. **Import check**: Try importing the module in an isolated context
3. **Backup**: Copy the original inner loop code to a `.bak` file

### Step 3.2 — Inject

Apply the generated code:
1. Write the mechanism module to the appropriate location
2. Apply the integration patch to the inner loop code
3. Verify the patched code still imports cleanly

### Step 3.3 — Revert Plan

If anything fails:
1. Restore from the `.bak` backup
2. Log what went wrong
3. Return to Round 2 and select the next candidate

---

## Phase 4: Observe (Optional)

If the inner loop runs after injection, observe the first few iterations:
- Did proposal diversity increase?
- Did the acceptance rate change?
- Did the metric improve?

Report findings to the user. If the mechanism is not helping, offer to
run another research session with a different candidate from Round 1.

---

## Tips

- **Don't over-engineer.** A 30-line Tabu list beats a 300-line framework.
- **One mechanism per session.** Don't try to inject multiple mechanisms at once.
- **Respect the inner loop's structure.** The mechanism should *guide* the inner
  loop, not replace it. The LLM still makes the proposals; the mechanism
  shapes which proposals get through.
- **stdlib only by default.** External dependencies are a reliability risk.
  If the user confirms a library is available, it's fine to use it.
- **Log everything.** The mechanism should log its decisions so the next
  Level 2 session can read what happened.

---

## Examples

**ML hyperparameter tuning stuck on learning rate:**
```
User: "My loop keeps tweaking LR and ignoring batch size."
→ Diagnosis: parameter fixation
→ Mechanism: Tabu Search (prevents re-proposing recently visited param regions)
→ Result: proposal diversity 3x, found better config in 5 iterations
```

**Code optimization hitting plateau:**
```
User: "My kernel optimizer has been at 2,160 cycles for 6 sessions."
→ Diagnosis: hand-crafted scheduling at ceiling
→ Mechanism: strategy escalation rule (force architectural change after N stalls)
→ Result: broke through to 1,500 cycles with graph-based scheduler
```

**Prompt engineering loop not converging:**
```
User: "My prompt optimizer keeps generating similar variants."
→ Diagnosis: low diversity, no spatial memory
→ Mechanism: orthogonal exploration (force search along unused dimensions)
→ Result: discovered a prompt structure the loop had never tried
```

---

## Design Principles

- **Separate generation from evaluation.** The loop can't see its own blind spots; Level 2 can.
- **Log everything.** The mechanism should record its decisions so the next session can learn from them.
- **Learn from ALL results, not just the best.** Extract what worked from failures too.
