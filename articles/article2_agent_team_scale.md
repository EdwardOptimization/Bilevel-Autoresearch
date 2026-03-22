# Beyond Depth: How Agent Teams Scale Through Parallelization, Layering, and Specialization

> Subtitle: From another perspective, "making a project work well" is itself a research thread, and scaling the project is the organizational unfolding of that thread

## Abstract

A common misconception is:

> Multi-agent teams can only take on large projects after first "going deep on a single thread."

This is incomplete.

More precisely, there are at least two viable paths for Agent Teams to scale up to large projects:

1. **Depth-first path**: First go deep on a specific main thread, stabilize it, and achieve net positive gains, then gradually parallelize, layer, and specialize along real bottlenecks;
2. **Breadth-first / top-down path**: When experience is sufficiently rich and the problem structure is clear from the outset, the team can design a division of labor, multi-threaded collaboration, and organizational structure from the very beginning.

In other words, a project does not necessarily have to start from "depth" --- it can also start from "breadth." However, breadth-first typically requires:
- The organizer already has strong experience;
- Core modules can be identified early;
- Shared objectives, shared protocols, and shared verifiers can be defined in advance.

From another perspective, what it means to "make a project work well" is not simply execution, but rather:
- Defining objectives;
- Designing verification;
- Iterating through trial and error;
- Accumulating experience;
- Forming a division of labor;
- Eventually crystallizing into a stable structure.

This is itself a research thread.

**Scope of this article**: The previous article [`llm_research_depth_convergence.md`](./llm_research_depth_convergence.md) discussed single-thread depth --- how to use agents to go deep and do well on a single research thread. This article turns to multi-thread expansion --- when a project is no longer a single thread, how to parallelize, layer, and specialize.

Therefore, the core thesis of this article is:

> **Agent Teams can go not only deep but also broad. The real question is not "depth first or breadth first," but rather: as a project grows deeper and broader, how can the system parallelize, layer, and specialize without losing its shared objectives and convergence capability.**

### First, a definition: What does "going broad" mean here?

What this article means by "going broad" is not "touching on everything a little," but rather the following specific forms of expansion:

- **Exploring more hypotheses / sub-problems / research threads**: Advancing multiple research branches simultaneously;
- **Multi-role division of labor**: Different agents are responsible for different modules or stages;
- **Multi-thread parallel execution**: Multiple work streams run concurrently rather than sequentially;
- **Organizational scaling**: From monolith to small group to multiple groups to layered structure.

This differs from "expanding more proposals within a single thread" as described in Article 1. The multiple proposals in Article 1 involve parallel search within a single thread; the "breadth" in this article involves resource allocation and coordination across multiple different threads.

---

## 1. Two Starting Paths for Scaling Up Projects: Starting from Depth, or Starting from Breadth

When many people encounter multi-agent systems, their first intuition is to think about:

- Multi-role division of labor
- Multi-threaded parallelism
- Multi-module collaboration
- Multi-tool integration

These things are certainly important, but they do not automatically produce results by themselves.

If the system does not yet have a **main thread that can reliably produce net positive gains**, then:
- Division of labor merely becomes role confusion;
- Parallelism merely becomes noise amplification;
- Collaboration merely becomes mutual paraphrasing;
- Tools merely become more elaborate surface prosperity.

Therefore, for most teams, the safer path remains:

> **First prove that a single thread can converge reliably.**

Here, a "thread" can be:
- An experimental optimization loop;
- A literature-to-hypothesis-to-verification research chain;
- A bug triage-to-fix-to-verification engineering chain;
- A data-to-evaluation-to-decision analysis chain.

Only when one of these threads can sustain depth, sustain error correction, and sustain verifiable progress does it deserve to be expanded into a larger system.

But this does not mean breadth-first is invalid.

When the organizer is already sufficiently familiar with the type of problem and can see from the outset:
- Which modules naturally should be separated;
- Which work streams should proceed in parallel;
- Which roles should be specialized from day 1;
- Which verifiers and handoff protocols must exist in advance;

then the system can absolutely start from "breadth-first organization."

In other words:

> **Not all projects must start from depth, but if you want to start from breadth, the prerequisite is that you already have enough experience with the project structure to see in advance the necessity of division of labor, collaboration, and multi-thread advancement.**

More specifically, the "experience" that breadth-first depends on includes at least several layers:

- **Task decomposition experience**: Knowing roughly what pieces this type of project needs to be broken into;
- **Routing experience**: Knowing what type of sub-task should be assigned to what type of agent / role;
- **Evaluation infrastructure experience**: Knowing what kind of verifier and gate each thread needs;
- **Domain priors**: Knowing which directions in this domain are worth exploring in parallel, and which are likely dead ends.

Missing any one of these layers, breadth-first easily degenerates into "looks like there's a division of labor, but nobody is actually converging."

---

## 2. Regardless of Starting from Depth or Breadth, Local Threads Must Be Made Deep

Going deep does not mean endlessly polishing details, but rather:

> **On a sufficiently specific problem surface, establishing a stable closed loop of proposal, feedback, verification, rollback, and experience accumulation.**

A thread being truly "made deep" means at least:

1. **Specific objectives**
   - Knowing what this thread is optimizing;
   - The objective is not an abstract slogan, but a comparable, verifiable target.

2. **Hard feedback**
   - Knowing what counts as progress and what counts as noise;
   - Having clear gates, not relying on gut feeling.

3. **Sustained iteration capability**
   - A failed round does not cause the system to lose its memory;
   - A successful round is not immediately overwritten by noise.

4. **State crystallization**
   - Knowing what the current working state, best state, and knowledge state are respectively;
   - Knowing which results are proposals and which are verified results.

5. **Accumulable experience**
   - Failures are not lost, but become priors for the next round of search;
   - Successes are not accidental, but can be reused and verified.

In other words, "going deep" is essentially turning a thread into:

> **A system that can continuously compress error.**

If even this is absent, then "scaling up" is merely replicating an unstable structure to a larger scope.

So a more precise statement is not "depth must always precede breadth," but rather:

> **Regardless of whether a project starts from depth or from breadth, it must eventually make several local threads deep and solid; otherwise, breadth only becomes a management and communication burden.**

---

## 3. From Another Perspective: Making a Project Work Well Is Itself a Research Thread

This observation is important:

> **Project management is not merely execution; making a project work well is itself a form of research activity.**

Why? Because to truly make a project work well, one must continuously answer the following questions:

- What is the real objective?
- Which paths are worth investing in?
- Which attempts are merely local optima?
- Which modules should be factored out?
- Which collaborations are genuine, and which are just paraphrasing?
- What constitutes reusable experience, and what is just a one-off coincidence?

This is not fundamentally different from a research thread.

A mature research thread typically goes through:
- Posing questions
- Proposing hypotheses
- Running experiments
- Examining feedback
- Refining models
- Updating methodology

And a project that is truly made to work well also goes through:
- Clarifying objectives
- Defining division of labor
- Establishing verification
- Iterating through trial and error
- Crystallizing processes
- Forming a stable organizational structure

So from a systems perspective:

> **"Making a project work well" is itself continuous research on the question of "how to effectively achieve objectives."**

This is also why truly strong project leads / CTOs / architects are often not merely executors, but more like researchers:

- Researching where bottlenecks lie;
- Researching which organizational approaches are more effective;
- Researching when to split and when to merge;
- Researching how to turn local successes into long-term stable successes.

---

## 4. The Real Growth Trajectory of Agent Teams Scaling Up Projects

If we abstract the growth process of large projects, it typically looks more like the following trajectory.

### Phase 1: Monolithic MVP
The beginning is usually not a team, but a monolith:
- A single agent or a small group of tightly coupled agents;
- Wearing multiple hats;
- The goal is to first run through the minimum viable loop.

At this point, what matters most is not elegant division of labor, but rather:

> **Proving that "this thread can work."**

### Phase 2: Parallel Compensation
When the monolith's capability approaches its current ceiling, the system naturally shifts toward:
- Multiple parallel proposals
- Multiple parallel candidates
- Multi-perspective review
- Multiple cheap workers trying simultaneously

The essence of this step is not "professional specialization," but rather:

> **Using multiple decent individuals to compensate for the diminishing marginal returns of a single individual.**

### Phase 3: Coordination Bottleneck Emerges
Once the scale of parallelism grows, new problems immediately appear:
- Who should do what?
- Whose results count?
- Which proposal gets to enter the main thread?
- How to prevent messages and states from contaminating each other?

At this point, the system is no longer primarily bottlenecked on "insufficient capability," but rather on:

> **Agreement, protocol, handoff, verifier, and state management.**

### Phase 4: Layering and Specialization
At this stage, the system can no longer rely on "everyone chatting with everyone," and must develop:
- supervisor
- worker
- reviewer
- router
- verifier
- control plane
- shared state

In other words, the system evolves from "agent group chat" to "agent organization."

### Phase 5: Specialized Modules Become New Monoliths
Looking further, one sees:
- research team
- coding team
- audit team
- delivery team

Each of these modules internally repeats the same cycle:
- First monolith
- Then parallelization
- Then layering
- Then specialization

So this is not a one-time process, but a recursive one.

This also explains why:

> **Large projects are not built by a single super-agent, but by an agent organization capable of continuous fission, collaboration, adjudication, and convergence.**

---

## 5. Why Failure to Go Deep on Any Single Thread Makes True Scaling Impossible

If no single thread has been made deep, large projects typically degenerate into three illusions.

### Illusion One: The Division-of-Labor Illusion
It looks like there are many roles:
- planner
- coder
- reviewer
- PM
- analyst

But in reality, nobody is actually driving any thread toward convergence --- the system is merely paraphrasing endlessly.

### Illusion Two: The Parallelism Illusion
It looks like many agents are running simultaneously:
- multiple sessions
- multiple tasks
- multiple logs
- multiple outputs

But without a unified objective and unified gate, the end result is merely generating more noise in parallel.

### Illusion Three: The Scale Illusion
It looks like the system is large:
- many modules
- many workflows
- many tools
- many pages

But the core main thread has not formed positive accumulation, so the larger the scale, the faster the loss of control.

So ultimately:

> **You cannot prove value through scale first; you can only prove value first through the stable convergence of a single thread.**

---

## 6. The Real Goal Is Not "More Agents," but "A Deeper Main Thread"

If we ground this insight in practice, the truly high-leverage questions are not:

- Should we add another agent?
- Should we integrate another tool?
- Should we add another UI?

But rather:

### 6.1 What exactly is the current main thread?
Which single thread is the system's most critical one that most deserves to be made deep?

### 6.2 Who is this thread's verifier?
Who decides:
- What counts as progress
- What counts as failure
- What must be rolled back

### 6.3 Where is this thread's state?
- Where is the current working state?
- Where is the best state?
- Where is the genuinely learned knowledge?

### 6.4 At what stage should this thread be split?
- When is parallelism more cost-effective?
- When are specialized roles needed?
- When has the communication cost already exceeded the benefit?

### 6.5 How does this thread's experience transfer across iterations?
- Which failures are worth writing into memory?
- Which successes can be turned into protocols?
- Which rules should be crystallized into organizational structure?

Only when these questions are answered clearly is an Agent Team truly working on a large project, rather than merely performing the appearance of one.

---

## 7. The Depth Methodology from Article 1 Can Be Applied to Team and Project Management Itself

This point is easily overlooked but extremely important:

> **"How to organize an agent team" can itself be treated as a research thread to go deep on.**

That is to say, the methodology proposed in Article 1 --- proposal, feedback, keep/discard, iterative convergence --- applies not only to "scientific experiments" or "code optimization," but equally to:

- **How to divide labor more effectively**: Try one division-of-labor approach, observe the results, and switch if it doesn't work;
- **How to define verifiers and handoff protocols**: Start with one set, observe where things frequently get stuck, then iteratively improve;
- **How to design parallel structures**: Start by parallelizing two threads, see if the coordination cost is acceptable, then decide whether to add more;
- **How to implement layering and specialization**: First factor out a module, see if things improved or got messier, then decide whether to continue factoring or merge it back.

In other words:

> **Organizational design is not a one-time architectural blueprint, but a continuously optimized research thread.**

This means that all the tools from Article 1 can be directly reused:

- **Proposal**: Propose a new division-of-labor approach / coordination protocol / role definition
- **Feedback**: Observe the actual results of this round of division of labor (throughput, error rate, communication cost, delivery quality)
- **Keep/discard**: Keep if effective, roll back to the previous version of organizational structure if not
- **Experience accumulation**: Write "why this division-of-labor approach failed" into organizational memory

So the final insight is:

> **If you are working on a project that is growing ever deeper and ever broader, then "how to manage this project" should itself be treated as a research thread requiring iterative convergence --- using the same proposal-feedback-iterate mechanism to continuously optimize the organizational structure itself.**

This is also why the strongest agent teams are often not the ones that "got the design right from the start," but the ones that "can continuously adjust their own organizational approach based on feedback."

---

## 8. Relationship with "LLM Research Depth Convergence"

This article and the previous article:

- [`llm_research_depth_convergence.md`](./llm_research_depth_convergence.md)

are two perspectives on the same problem, but with different emphases.

The previous article emphasizes:
- Automated research projects are essentially exploring: **how to use agents to go deep and do well on a single task**;
- Why this relates to the optimization perspective, proposal quality, feedback, iteration, and shared verifiers;
- Why specific targets, protocols, and agreement mechanisms are prerequisites.

This article further emphasizes:

> **Agents can go not only deep but also broad; what truly deserves discussion is: as a project grows deeper and broader, how does the system parallelize, layer, and specialize.**

In other words:
- The previous article answers "how to go deep and do well on a single research thread";
- This article answers "after a project has grown deeper and broader, how to organize it into a team and scale it into a larger structure."

Together, the two articles form a more complete thesis:

> **One article discusses depth convergence, the other discusses organizational scaling. The former explains "how to make things deep," the latter explains "once things have grown both deep and broad, how to organize them to scale up."**

---

## 9. Conclusion

What this article ultimately aims to express is:

1. **Agent Teams can start scaling large projects either from "depth" or from "breadth"; breadth-first is entirely valid for experienced teams.**
2. **But regardless of the path, after a project has grown both deeper and broader, it must answer the same question: how to parallelize, layer, and specialize without losing control.**
3. **Making a project work well is not pure execution, but a research thread about "how to effectively achieve objectives."**
4. **The growth trajectory of Agent Teams has two typical forms:**
   - **Depth-first**: Monolithic MVP -> parallel compensation -> coordination bottleneck -> layering and specialization -> recursive unfolding
   - **Breadth-first**: Experience-driven multi-thread advancement from the start -> coordination bottleneck encountered -> layering and specialization -> each layer then deepens independently
5. **Both paths encounter the same challenges in the middle and later stages: communication cost, state synchronization, verifiers, and protocols.**
6. **The real goal is not more agents, but rather: maintaining the convergence capability of every local main thread as the system grows.**

Therefore, the entire article can be compressed into a single sentence:

> **Agent Teams can go both deep and broad; scaling up large projects is not a matter of which starting approach to choose, but rather, as a project grows ever deeper and ever broader, how to maintain the stable convergence of every local main thread while organizationally expanding through parallelization, layering, and specialization.**
