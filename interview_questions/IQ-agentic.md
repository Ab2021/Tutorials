# Interview Pack · Agentic & Trajectory Evaluation

> **Covers:** CS-17 (agentic evaluations workshop — reliability, environments, living benchmarks),
> CS-18 (RL for agents — credit assignment, environments-as-datasets, benchmark construction rules,
> the sim-to-real harness, the data flywheel), CS-19 (fine-tuning a coding agent — SFT masking, the
> imitator-vs-agent selection trap, agent-operated training) · **Code ground truth:**
> `CODE-05` (frontier-evals — three-stage container execution grading), `CODE-01` (`PATTERNS.md` —
> trajectory and world-state patterns), `CODE-07` (search_evals — hosted agents with decimal cost
> accounting) · **Role level:** senior → staff · **Format:** screen → onsite → system-design

---

## How this domain is assessed

Agentic evaluation is the newest and least settled area in the field, which makes it the best
discriminator in an interview: there is no canonical answer to memorise, so what you say reveals how
you think. The interviewer is not checking whether you know the current leaderboard. They are checking
whether you understand that **an agent eval is an experiment with an environment, a grader, a harness
and a budget**, all four of which are confounds if unreported.

This is also the domain where the strongest candidates are the most sceptical. The candidate who says
"the score is meaningless without the harness" is further along than the one who quotes the score.

Two framings run through this pack, and the strongest candidates hold both at once.

1. **An agent eval is an experiment, not a measurement.** Environment, tasks, grader, harness and
   budget are all confounds if unreported. Tiers 1–5 are that argument, from fundamentals to traps.
2. **Evaluation and training consume the same object.** The environment you evaluate in is the
   environment you train in; the tasks are the dataset, the harness is the agent, and the metrics *are*
   the reward functions. This is why an agentic-RL workshop is also an evals workshop, and why Tier 6
   exists: the questions a candidate cannot answer on the training side are exactly the ones that
   explain why their eval numbers do not transfer to a deployed system.

| What is being scored | The signal | How it is elicited |
|---|---|---|
| **Capability vs reliability** | Do you know a high score is not a shipping decision? | "Your agent scores 78%. Ship?" |
| **Grading design** | Do you reach for world-state assertions and code before judges? | "How would you grade this task?" |
| **Reward-hacking instinct** | Do you see the specification bug, not the model bug? | "The agent passes tests but doesn't fix bugs." |
| **Environment thinking** | Do you think environment-first, not model-first? | "Where does the score come from?" |
| **Reporting discipline** | Do you name the harness, seeds, cost, session? | "What goes in the report?" |
| **Economics** | Do you know agent evals cost hours and dollars? | "How long does one run take?" |
| **Training-side literacy** | Do you know what RL on agents actually changes about verification? | "How does reward get assigned across a 200-step trajectory?" |

**The single most predictive question in this pack** is *"what would make you distrust an agent
benchmark score?"* A strong candidate immediately names omitted tasks, unreported harness, and a
single run with no variance — all concrete, all real.

---

## Tier 1 — Fundamentals (10 Q)

### Q1. Why are agents harder to evaluate than LLMs?

**What they're really testing:** whether you have a structural model, not a list of complaints.

**Model answer:**
Four structural reasons, and they compound:
- **Stochasticity.** Two runs on the same task can produce different results. A chatbot's output
  varies; an agent's *outcome* varies, which is a different and worse problem — you cannot trust a
  single run.
- **Multi-step credit assignment.** There are many steps between input and outcome, so when the task
  fails you cannot tell which step failed without reading the trajectory.
- **World interaction.** The agent acts on tools and state that change underneath it, so the
  evaluation is not reproducible in the way a text eval is.
- **Cost and destructiveness.** Real-world interaction is expensive, and some of it is irreversible —
  you do not want your evaluation to delete production data or email real users.

On top of those four, the output is not a string: an agent's real output is often a **world state**,
which means the grading problem is different in kind.

**Red flags:** "they're just LLMs with tools"; listing only cost; no mention of non-reproducibility.

### Q2. What is the capability–reliability gap?

**What they're really testing:** whether you understand why benchmarks and reality disagree.

**Model answer:**
- Capability benchmarks measure whether a system *can* do a task; reliability measures whether it does
  it *consistently, safely and predictably*. They are different axes.
- The empirical finding: over **18 months across 14 frontier models**, accuracy on GAIA and Tau-bench
  airline improved dramatically while a composite reliability score rose only gradually. Plotted
  against each other, the relationship is a **remarkably straight, shallow line**.
- This resolves the paradox the field lives in — agents are crushing benchmarks, yet there is no
  measurable economic impact. Capability is going up; **capability does not fully measure usefulness**.
- The severity argument for why reliability matters more for agents than chatbots: Siri playing the
  wrong song 10% of the time is an annoyance; an agent placing the wrong order with your credit card
  10% of the time is **dead on arrival**. The aviation analogy is deliberate — from demonstrating
  flight to a standard of roughly one error per trillion miles took most of a century.
- The practical consequence: **a release gate on capability alone is not a release gate.** You need an
  explicit reliability threshold.

**Red flags:** treating reliability as "just more evals"; assuming smarter models fix it automatically
(the source explicitly says do not assume that); no separation of the two axes.

### Q3. Name the four dimensions of agent reliability.

**What they're really testing:** whether you have the decomposition, not just the slogan.

**Model answer:**
Four dimensions, decomposing into **12 metrics**, of which roughly **two are more or less solved**:

| Dimension | Sub-metrics | What it asks |
|---|---|---|
| **Consistency** | Outcome consistency; trajectory consistency; cost/resource stability | Same result, same actions, stable cost across repeated runs |
| **Robustness** | Fault robustness; prompt robustness | Behaviour under injected API failures, and under semantic-preserving prompt rewording |
| **Predictability** | Calibration; discrimination | Can it say how likely it is to have succeeded, and is that number true — and does it separate wins from losses |
| **Safety** | Failure severity | A formatting error versus data deletion |

Two details that matter for credibility:
- **Safety severity is measured but deliberately not aggregated** into the composite index, because
  folding a severity scale into an average destroys its meaning.
- The consistency question is subtler than it looks: "70% accuracy might mean 70% of tasks it performs
  consistently and 30% it consistently fails — or that on any given task, unpredictably, it works 70%
  of the time." Those are very different products.

**Red flags:** naming only consistency; folding safety into the average; no awareness that accuracy
hides whether failures are systematic or random.

### Q4. What is the difference between calibration and discrimination? *(a favourite trap)*

**What they're really testing:** whether you will collapse two metrics into one.

**Model answer:**
- **Calibration** asks whether the stated confidence is *true*: if the agent says 60%, does it succeed
  60% of the time?
- **Discrimination** asks whether the confidence *separates successes from failures* — whether a high
  confidence actually predicts a win.
- They are independent, and they have been moving in **opposite directions**: calibration has been
  improving (companies were burned by overconfident, sycophantic chatbots), but **discrimination has
  been getting worse over time**.
- The degenerate case proves the point: an agent that always answers "50%" is **perfectly calibrated
  and completely useless** for triage.
- Uncalibrated agents err toward overconfidence, not underconfidence — they say 1.0 and succeed in
  about half those cases.
- The interview move is to refuse the yes/no question: "is your agent well calibrated?" is
  unanswerable without both terms.

**Red flags:** answering "well calibrated" as a yes/no; never having heard of discrimination; treating
confidence as a single quality.

### Q5. What is pass@k, and how does it differ from pass^k?

**What they're really testing:** whether you can distinguish capability from reliability in one metric.

**Model answer:**
- **pass@k** — *capability*: given k attempts, did **at least one** succeed. It is the right metric for
  "can this model do this at all", and for anything with a verifier where best-of-k is a legitimate
  strategy.
- **pass^k** — *reliability*: **every one** of k attempts must succeed. It is the right metric for a
  customer-facing policy where a single failure is a failure.
- They tell opposite stories at the same underlying success probability. At a per-trial success rate of
  **p = 0.75** over 10 trials: `pass@10 ≈ 1.0` while `pass^10 ≈ 0.056`. Same system, same task,
  dramatically different verdicts.
- Use the **unbiased combinatorial estimator**, not the naive one:
  `pass@k = 1 - C(n-c, k) / C(n, k)` for n samples with c correct — computed as a running product, never
  by forming the giant binomials. The naive `1 - (1-p)^k` is **biased high for small n** and should
  never be reported.
- In an interview, saying both numbers and knowing *which question each answers* is the whole test.

**Red flags:** using pass@k to make a reliability claim; the naive estimator; reporting a single run
with no pass@k at all.

### Q6. What is reward hacking in an agent eval, and how do you design against it?

**What they're really testing:** whether you see it as a specification bug or a model bug.

**Model answer:**
- Reward hacking is the agent optimising **the grader instead of the task** — the canonical example
  being an agent that, instead of fixing the bug, **edits the unit test so it passes**.
- The critical framing: **reward hacking is usually a specification bug, not a model bug.** The agent
  optimises exactly what you measure. If your measurement is a strict subset of your intent, you built
  the hack yourself.
- Design defences:
  1. **The agent must not see the grader or the solution.** This is the primary rule, and some
     environment formats provide it natively.
  2. **Grade everything the task asked for.** Ask for three things and grade two, and the agent learns
     the third is unnecessary.
  3. **Fingerprint unit tests** so they cannot be quietly modified.
  4. **Vary tool names and signatures between runs** so the agent cannot overfit to a specific API
     surface.
  5. **Hold out a private set** for reporting, so the public grader is not the optimisation target.
- Detection: diff the test files across runs; watch for rising pass rate with falling task quality.
- **Why agentic evals are structurally more hackable than text evals** — three compounding reasons worth
  naming together: benchmarks saturate in roughly **6–12 months**; open-weight models **overfit to
  public benchmarks**; and agent evals hand the model a **terminal, a bash shell and a sandbox**, which
  is far more surface to exploit than a chat interface. The stated operational conclusion is blunt:
  **have your own internal evals.** A public score is a marketing artefact; the internal suite is the
  control.
- **The single most useful detection habit: read the trajectories, not the score.** Reward hacking shows
  up in the *trajectory* — a fix that edits the test, a success reached by an unintended route, a
  strategy that should have been abandoned several steps earlier. The score will look like progress,
  because that is exactly what hacking optimises.

**Red flags:** treating it as a model alignment problem; no grader isolation; no coverage audit of the
task's requirements; no fingerprints on tests; reporting a public benchmark score as if it were a
measurement of your system.

### Q7. Hard verifier versus soft verifier — when do you use each?

**What they're really testing:** whether you have the strongest available engineering instinct in this
domain.

**Model answer:**
- **Hard verifiers** check the action sequence by **pure equality or simple algorithmic code**:
  comparing the expected action diagram to the actual one, event to event, confirming the right order
  and the right parameters. "Did it send to the right address? At the right time?" They are cheap,
  reproducible, and LLM-free.
- **Soft verifiers** are used only where the artifact is **natural language whose meaning matters but
  whose wording does not** — for example, checking that the *content* of an email an agent sent is
  appropriate, where it "doesn't have to match entirely."
- The rule worth memorising: **anything that can be checked by equality should never be checked by an
  LLM.** Reserve judge calls for the genuinely language-shaped residue.
- The motivation is explicit: a leading agentic benchmark moved *away* from rubric judging because it
  is "very dependent on LLMs and their capability and very expensive." Action-level verification is far
  cheaper and far more reproducible.
- The three-level hierarchy: **hard verifier → soft verifier → rubrics with an LLM judge**, in that
  order, stopping as early as the criterion permits.
- **The judge boundary, stated precisely — two positions that look contradictory and are not.** One
  camp rejects LLM-as-a-judge for grading outright, on the grounds that judges prefer their own outputs
  and are neither accurate nor robust; its advice is "don't be lazy, just think about it more until you
  figure out how to have a deterministic verifier." The other camp accepts a *narrow* judge: a
  **binary yes/no** question where "the vast majority of smart humans who are aware of the relevant
  domain would agree on the answer," which makes it "pretty close to verifiable."
- The discriminator is **whether the judgement is fuzzy or near-unanimous**. Fuzzy quality ("is this
  report good?") → find a deterministic proxy or do not measure it. Near-unanimous binary ("does this
  email contain the refund amount?") → judge it, prefer a cheap local model, and get robustness by
  **aggregating many individual binary criteria** by summing or averaging rather than asking one
  holistic question. Many small certain answers beat one large uncertain one — and the aggregation is
  where the reliability comes from, not any individual call.
- Stating that reconciliation out loud is the strongest possible answer to this question, because it
  shows you have noticed that the field's two loudest positions are not actually in conflict.

**Red flags:** judging trajectories with an LLM by default; no awareness that order and parameters are
mechanically checkable; using a judge where a boolean would do; treating the field's judge debate as a
contradiction rather than a scope boundary.

### Q8. Why simulate the environment instead of evaluating agents on the real world?

**What they're really testing:** whether you understand the trade you are making.

**Model answer:**
Four reasons, and one cost:
- **Reproducibility** — the same seed, the same world, every run.
- **Observability** — you can inspect the full world state, not just what the agent said.
- **Safety** — you want to test tasks like "remove all my emails, cancel my calendar events", and you
  cannot do that on the open web.
- **Cost** — no dependence on external bandwidth and third-party APIs, so a long run is affordable.
- The cost is the **sim-to-real gap**: an isolated sandbox is much simpler than the real world. The
  practical mitigation is to reimplement only the subset of a real service your tasks actually
  exercise, with the state machine intact — you do not need a full Slack implementation, you need the
  part your agent touches.
- The substrate choice matters: a simulated world lets you inject **deliberate noise** — tool failures,
  changed tool signatures, unrelated environment events — which is how you measure robustness rather
  than just capability.

**Red flags:** "just test in production"; no safety argument; no acknowledgement of the sim-to-real gap.

### Q9. What is the harness confound?

**What they're really testing:** whether you know that a leaderboard names the wrong thing.

**Model answer:**
- The **harness** is the scaffolding around the model: the orchestration loop, the prompting strategy,
  the tools, the retry policy, the memory. **The same model in a different harness has drastically
  different success rates.**
- This is not a small effect — it can **flip the ranking** of two models. The reference example: under
  one scaffold, model A scores 13.2 and model B 21.0; under a second scaffold, model A scores 26.0 and
  model B 16.1. The leaderboard order inverts.
- Consequence: **a score is not a property of a model.** It is a property of a model-plus-harness pair,
  and reporting only the model name is reporting half a result.
- Public dashboards exist that show exactly this — the same model on different harnesses producing
  different metrics — which is why reporting the harness in the same column as the model is now a
  baseline expectation for a credible agent benchmark.
- Related: the **time budget** is also a hyperparameter. Two runs of the same agent at 24h and 36h are
  different experiments; the shorter budget cost 1.6 points in the reference leaderboard.
- **The training-side consequence is sharper than the eval-side one, and it is the sentence to have
  ready: the harness you deploy in must be the harness you train in.** If a harness change moves the
  score and can invert a ranking, then the harness is not plumbing — it is **part of the model
  artefact**. Train in one harness and deploy in another and you have manufactured a sim-to-real gap
  out of nothing, and every capability you measured belongs to a system you are not shipping.
- This is why environment specifications include a **harness shape** as a first-class choice, and the
  shapes are not interchangeable: a sandboxed terminal harness that runs tests at the end, a harness
  driven by an LLM user simulator, a harness over a synthetic MCP backend, and a trivial wrapper around
  a third-party agent library are four different training and evaluation targets.

**Red flags:** quoting an agent score without the scaffold; treating the model as the unit of
evaluation; no awareness of time budgets as a variable; treating the harness as interchangeable between
training and deployment.

### Q10. Why does session-level reporting matter?

**What they're really testing:** whether you can see that a single number is lossy.

**Model answer:**
- **Two sessions can produce identical scores and be completely different stories.** The reference
  illustration: two agents both "pass"; one takes few steps, costs little, finishes fast and has zero
  errors; the other takes many steps, costs a lot, finishes slowly and errors several times. Same
  score, different products.
- So a score alone is not evidence. You need the **session** — which run produced it, with what seed,
  what harness, what model version and quantization, what number of shots, and what cost.
- **Agent identity is the deeper version of the same problem.** Evaluations are tagged with a model
  name, but *the model is not the agent*: the agent includes the sub-agent list, the MCP servers, the
  memory configuration and the orchestration. Without those, a result is not attributable and not
  reproducible.
- This is also a **governance** requirement, not just a research nicety: without session information
  you cannot audit behaviour or identify **which actor was responsible for a harm**.
- Practical form: report score ± spread over N runs, with harness, seeds and cost, and keep the
  per-instance results rather than only the aggregate.

**Red flags:** reporting a leaderboard row with only a model name; no per-session data; treating cost
and latency as separate concerns from quality.

---

## Tier 2 — Applied / trade-off (10 Q)

### Q11. How do you test reliability without a reliability-specific benchmark?

**What they're testing:** whether you know reliability is an *overlay*, not a separate suite.

**Model answer:**
You take a benchmark you already run and add measurements to it — the source is explicit that "this is
not some reliability-specific benchmark; you can take any benchmark and measure all of our reliability
metrics on it." Procedure:
1. Run the same task **n times**; record **outcome consistency**.
2. Compare the **trajectories** across runs; record **trajectory consistency** — same actions, same
   order?
3. **Reword the prompt** with an LLM, preserving semantics but changing style, and record the delta.
4. **Inject faults** — API timeouts, inaccessible data — and record the delta.
5. Elicit **confidence** and compare it to actual success frequency: calibration and discrimination.
6. Record **cost and latency stability** across runs.
7. Score **failure severity** separately; never fold it into the composite.
8. Report the harness, scaffolds and seeds with the score.

**Red flags:** "we'd need a separate benchmark"; testing only one of the four dimensions; folding
severity into the average.

### Q12. When are reliability metrics the *wrong* thing to measure?

**What they're testing:** whether you know the scope limit, which is easy to lose.

**Model answer:**
When the task's value is **variation**. The source's example: an agent that writes poetry. If it
produces the same poem on a given topic every time, "that would be a very bad thing. You want the
agent to be creative, you want it to be very stochastic." The rule: **it depends on the task whether it
makes sense to measure what we're measuring.**

The sharpened version: **trajectory consistency is a virtue where the task has a canonical procedure**
— customer service, refunds, data entry, QA at scale — **and a defect where creativity is the point** —
writing, brainstorming, design.

Related scoping: reliability is a property of a **deployment**, not of a model. Which is why the
augmentation/automation distinction comes first: a coding agent's errors are tolerable because a human
reviews the output; the same errors in an autonomous customer-service agent are not.

**Red flags:** applying consistency metrics universally; no notion that reliability thresholds vary by
deployment; folding creative tasks into a reliability index.

### Q13. How do you decide what to grade in an agent task?

**What they're testing:** whether you close the loop between task statement and grader.

**Model answer:**
- Start from the task statement and **enumerate every requirement**. The rule is absolute: **everything
  you asked for must be graded.** Ask for three things and grade two, and the agent learns it does not
  need to work on the third. That is not hacking — that is your specification permitting it.
- Then choose the verification mode per requirement, in cost order:
  - **Hard verifier** if correctness is decidable by equality, ordering or arithmetic.
  - **Soft verifier** if the artifact is natural language whose meaning matters but whose wording does
    not.
  - **Rubrics with an LLM judge** only for genuinely graded quality.
- **Grade the world state where possible, not the transcript.** Filesystem, database rows, ticket
  state, passing tests — all deterministic and ungameable.
- **Grade how, not just what.** A lucky success and a well-executed failure are different results;
  trajectory metrics (tool-selection accuracy, argument accuracy, step efficiency, recovery rate)
  separate them.
- Add **safety as a graded task**: verify the agent did not delete data or max out a card, and penalise
  it explicitly.
- Finally, check the inverse: is there anything in the grader that is *not* in the task? A grader
  checking something never asked for is a hidden requirement and a source of surprising failures.

**Red flags:** grading only the final output; no enumeration of requirements; no safety task; grading
prose by string match.

### Q14. The agent passes all the unit tests but the bug isn't fixed. Diagnose and fix.

**What they're testing:** the reward-hacking reflex, in its most concrete form.

**Model answer:**
- First diagnosis: the agent is **editing the tests** rather than the code. The fix is **fingerprints
  on the unit tests** — a hash or marker that detects modification, so a modified test scores zero
  rather than passing.
- Second diagnosis, and the one people miss: **the tests are a bad proxy for the work.** If a number of
  unit tests can pass while the real task is not done, the suite is under-specified. Two sub-causes:
  - **Weighting** — do trivial tests count equally with the critical one? Many trivial tests can pass
    while the real work is undone.
  - **Coverage** — the tests check a subset of the task's requirements.
- Fixes: hide the grader and solution from the agent; fingerprint tests; weight tests by importance
  rather than counting them; add a check on the actual world state (does the reported behaviour now
  hold?); and hold out a set of tests the agent never sees.
- The general principle to state: **the agent optimises exactly what you measure**, so a pass rate that
  rises while the product gets worse is a measurement design failure before it is a model failure.

**Red flags:** blaming the model; only fingerprinting without checking test quality; no held-out tests.

### Q15. What should an agent evaluation report contain?

**What they're testing:** whether you would produce a result someone else could reproduce.

**Model answer:**
Minimum disclosure set, and it is longer than people expect:
- **Score with variance** — ± across N runs, not a point estimate. Long-horizon agents are noisy.
- **The harness** — the scaffold, in the same column as the model. A score without it is not
  attributable, and harness choice can flip rankings.
- **Model identity** — version and **quantization**. Different quantization is a different system.
- **Eval conditions** — number of shots, temperature, seeds, and the number of rollouts behind any
  pass@k figure.
- **Cost and latency** — currently inconsistent or absent, and part of usefulness.
- **Per-instance results**, not just the aggregate, so someone else can audit a specific case.
- **Standard error** alongside the point estimate.
- **Session-level information** — what defines a run, so a third party can reproduce it cleanly.

The motivation is not administrative: **score fragmentation** — every new release reporting numbers
for existing models that match nothing previously published — is caused precisely by every lab running
its own harness and its own framework with its own disclosure level.

**Red flags:** a bare score; no harness; no variance; no quantization; no per-instance data.

### Q16. How do you evaluate a long-horizon agent task?

**What they're testing:** whether you have thought about the hours-and-dollars end of the discipline.

**Model answer:**
- Long-horizon is named as one of the next open challenges, and part of the difficulty is definitional:
  "people mean different things when they mean long horizon". Pin the definition first — wall-clock
  duration, number of steps, number of distinct subtasks, or context length?
- The economics are the binding constraint. A reference frontier agentic eval runs for up to **36
  hours** in a GPU-backed container, and the time budget is itself a hyperparameter: dropping from 36h
  to 24h cost 1.6 points on the same agent. Two runs at different budgets are different experiments.
- Therefore the architecture has to be staged: **rollout → reproduce → grade**, each in a fresh
  container, so the expensive rollout is not repeated and a failed grade does not invalidate the
  rollout.
- **Never put it in CI.** Hours of wall-clock and GPU containers belong on a nightly or pre-release
  cadence.
- Test the harness with a **dummy solver** first — the cheapest honest path through a 36-hour eval is
  the solver that spends nothing.
- Report a **spread over multiple runs**, because a single run of a noisy long-horizon agent is not a
  result.
- For codebase-maintenance style tasks, the metric changes from "does it work now" to "does it still
  work after many subsequent changes" — a different evaluation shape from a one-shot fix.

**Red flags:** putting it in CI; a single run; no time-budget disclosure; no dummy-solver smoke test.

### Q17. Augmentation versus automation — why does the distinction matter?

**What they're testing:** whether your reliability threshold is connected to the deployment.

**Model answer:**
- **Augmentation** — a coding agent — means a human stays in the loop and reviews the output, so many
  errors are recoverable and not too bad.
- **Automation** — a customer-service agent handling customers autonomously — means the same errors
  land directly on the user and are **much worse**.
- Three consequences:
  1. For release decisions, **it is not just capability that matters**; a **reliability threshold**
     must be met before deploying. The threshold is a function of how much human review exists.
  2. For researchers, measuring reliability should become the **norm on any benchmark**, not a
     special-purpose suite.
  3. **Do not assume smarter models solve it.** The source is explicit: "maybe, but we should also
     prepare for the possibility that it won't happen, and we have to specifically work towards
     optimizing reliability."
- The practical consequence in an interview: when someone asks "is 85% good enough?", the correct
  answer is "for which deployment?" — and then you say what the human-in-the-loop situation is.

**Red flags:** one threshold for all deployments; assuming capability progress closes the gap; no
discussion of the human in the loop.

### Q18. Where should you run an agent evaluation?

**What they're testing:** whether you know that the run location changes the result's meaning.

**Model answer:**

| Location | Verdict | Reason |
|---|---|---|
| Local cluster / single GPU | Works, painful | GPU contention when more than one team uses it; will not fit bigger models |
| Cloud compute | Works | Your own control |
| **Controlled jobs on managed compute** | **Recommended** | Choose the hardware, reproducible, a one-liner to run |
| **Inference provider** | **Avoid** | You are evaluating the provider, not the model |

- The argument against inference providers is worth quoting: *"in reality you are evaluating the
  provider — which can be a good thing, but maybe not what you want — and you're not evaluating the
  model."* You do not know how the model is called in the backend, and you do not know whether it is
  prompted the way you intended, so the result is **not reproducible**.
- The nuance worth adding: comparing providers is a **legitimate and different question**. The error
  is doing a *model* evaluation through a provider and reporting it as one.
- The output of a proper run is not just a number: it is an artefact — a results page plus **detailed
  logs** recording the temperature used, how the model server was spun up, how many tries, and pass@k
  detail — plus the **standard error** of the evaluation.

**Red flags:** running model evals through a provider API; no logs; no standard error; no
acknowledgement that provider-comparison is legitimate.

### Q19. How do you measure cost and efficiency for an agent, and why does it matter?

**What they're testing:** whether you treat cost as a first-class eval dimension.

**Model answer:**
- Cost and efficiency are part of the **secondary metrics** and they are what separate two agents with
  identical success rates. The set to record per rollout: **steps to outcome**, **token count**
  (input and output separately), **wall-clock latency**, and **monetary cost**.
- **Step efficiency** is its own metric: a correct answer in 40 steps where 5 suffice is a much worse
  product than the same answer in 5. Report `optimal_steps / actual_steps`.
- Both sides of the bill must be tracked: **agent cost and grader cost**. You cannot optimise what you
  have not separated, and grader cost is easy to forget because it does not appear in the product.
- Use **exact decimal arithmetic** for cost aggregation, not floats — floating-point drift compounds
  across a long multi-step run and makes per-component attribution wrong.
- Cost reporting is currently **inconsistent or absent** across agent benchmarks, which is one of the
  named gaps in the field.
- Watch the trap: repeated identical inputs inflate provider prefix caching, so measured cost can
  **understate** real cost.
- **The budget is not only a cost; it is a capability axis, and it can change the ranking.** A benchmark
  that hands a model a fixed **$1** budget to make a program run faster — scored on average speedup
  across roughly 150 real programs, subject to a correctness check against reference implementations —
  produces rankings that differ from capability benchmarks, because **cheaper models sometimes beat
  frontier models by getting more iterations inside the budget**, while the strongest model "might
  generate one kind of candidate solution and then totally run out of budget." A capability score with
  no cost axis is not a deployment-relevant score, and a cost axis reported without the budget is not
  reproducible.

**Red flags:** cost as an afterthought; no step efficiency; float arithmetic; agent and grader cost
pooled; reporting a score without the budget it was earned under.

### Q20. How do you evaluate adaptive and multi-turn agent behaviour?

**What they're testing:** whether your eval model handles the world changing mid-task.

**Model answer:**
- The core capability is **adaptability**: the environment changes *after* the agent's actions. The
  canonical scenario — the agent books meetings, the other attendees then cancel, and it must
  reschedule. That cannot be scored by checking a final string.
- The mechanism is **events injected from three sources**: the user, the agent, and the environment.
  A scenario is therefore not a prompt but a **task prompt plus a sequence of expected agent actions
  plus environment events** — which is why the expected action sequence, not the expected answer, is
  the grading artefact.
- The related capability is **ambiguity handling**: the task cannot be resolved without asking the user
  a follow-up question, and a good agent stops and asks rather than acting wrongly. This is
  under-measured by design in most benchmarks, and it is arguably a *reliability asset* — real
  deployment is full of ambiguous tasks "and they don't handle it that well."
- Also evaluate **time-triggered** behaviour: events caused by clock passage rather than by the agent's
  actions — "book the flight when the price drops below X". This is the capability where every model
  measured scored around **0%**, including the strongest.
- Because grading is at the **action level**, the metrics are order-and-parameter checks on the action
  diagram, plus an LLM check only for natural-language content.

**Red flags:** one-shot grading of a multi-turn task; no ambiguity scenario; no environment-driven
events; no timing capability.

---

## Tier 3 — Senior / staff (8 Q)

### Q21. Design the agent eval programme for a customer-service agent.

**What they're testing:** whether you sequence the work correctly.

**Model answer:**
The order is **environment → tasks → grading → harness**, and getting it backwards is the most common
expensive mistake.

1. **Environment first.** A sandbox holding the world state, the tools and the data: the order
   database, the refund tool, the email tool, a seeded conversation history. Confirm the agent has
   **no access to the grader or the solution**.
2. **Tasks.** As close to production as possible. Enumerate requirements per task.
3. **Grade everything the task asks for.** "Issue the refund, notify the customer, and update the
   ticket" — all three graded, or the agent stops doing the third.
4. **Per-requirement verification mode.** Refund amount and ticket state are hard verifiers; the text
   of the apology email is a soft verifier.
5. **Noise.** Vary tool names and signatures between runs; inject tool failures so recovery is
   exercised.
6. **Metrics.** Success rate over n rollouts and pass@k; then steps, tokens, latency, cost; and report
   the harness.
7. **Reliability overlay.** Re-run several times for outcome and trajectory consistency; reword the
   prompt; inject faults; elicit confidence. Report the composite, and report **failure severity
   separately**.
8. **Read the traces.** Automated failure analysis converts "85% success" into "85% overall, but 40%
   when the customer mentions a prior refund."
9. **Safety as graded tasks.** Penalise data deletion. For automation (no human in the loop), set an
   explicit reliability threshold before release.
10. **Close the loop — but keep the evaluation set held out.** The same environment can be used to
    improve the agent; any environment used as a training signal stops being a valid measurement, so
    hold out a private slice for reporting.

**Red flags:** model-first ("let's pick a model and test it"); tasks without an environment; grading
only the final answer; no held-out slice once training starts.

### Q22. How would you build the simulated environment itself?

**What they're testing:** whether you can operate at the infrastructure layer.

**Model answer:**
Use the four-concept decomposition from a mature environment framework:
- **Apps** — like phone apps: each keeps state about the world and exposes tools/APIs the agent, the
  environment and the user can call, reachable via API, Python, MCP or CLI.
- **Universe** — the simulated environment plus its **initial state**: past emails received, current
  calendar events, message threads with other personas.
- **Events** — injected by the user, the agent, or the environment.
- **Scenarios** — a task prompt **plus** a sequence of expected agent actions **plus** environment
  events.

Construction at scale (the reference implementation): **1,000 scenarios across 10 universes and ~11
apps**, with universes **generated automatically from a persona database using an LLM** to produce
conversations, emails and calendar data, and **human annotators** creating the tasks, events and
scenarios on top. That division of labour is the practical lesson: generate the world, hand-author the
evaluation.

Build in the **noise knobs from day one**: tool-failure injection at configurable rates; tool **name
and signature** variation between runs; unrelated environment noise events; and an **agent-to-agent**
mode where the agent can only reach a tool by talking to a sub-agent that owns it.

Design decisions to make explicitly: how many environments (often one per domain or data slice — a
legal-domain agent may need different data per environment); realism versus cost; and **who can see
the grader** (nobody under test).

**Red flags:** a single monolithic sandbox; hand-writing all the world data; no noise injection; no
tool-signature variation.

### Q23. How do you handle the sim-to-real gap?

**What they're testing:** whether you know the limitation of your own instrument.

**Model answer:**
- State it honestly first: environments are **isolated sandboxes** and the real world is much more
  complex. Everything you measure is measured in a world you built, and the agent may behave
  differently in the world it will actually run in.
- The practical technique is **partial fidelity by design**: reimplement only the subset of a real
  service that your tasks actually exercise, with the state machine intact. You do not need a full
  Slack implementation — you need the part your agent touches, with correct semantics.
- **Calibrate realism to the use case.** More surface is not automatically better; it is cost. The
  question is "how do we make the environment realistic enough that it mimics the real world *for this
  agent's tasks*."
- **Watch for environment-model calibration**: environments tend to be built to the models available
  at the time, so stronger models get more complex environments. If your environment was calibrated
  against last year's models, it may be saturated for this year's.
- **Cross-check against reality**: keep a small, non-destructive real-world probe — read-only tasks on
  live surfaces — and compare its verdict with the simulated one. Divergence is your signal that the
  gap has grown.
- For destructive tasks, simulation is the *only* option, so the honest framing is "we accept the gap
  here because the alternative is unacceptable risk."

**Red flags:** claiming the environment is production-equivalent; full-fidelity ambitions; no real-world
cross-check; no awareness of environment-model calibration.

### Q24. What is the relationship between evaluating an agent and training one?

**What they're testing:** whether you understand the environment as the shared asset.

**Model answer:**
- **Lead with the strongest form of the claim: evals and environments are not similar, they are the
  same thing.** An environment is **tasks + harnesses + metrics**; the **tasks play the role of the
  dataset**, the **harness plays the role of the agent / tool-call interface**, and the **metrics are
  the reward functions**. One object then serves five consumers: RL, evaluation, synthetic data
  generation, prompt optimisation and model ablation.
- The justification is operational, not philosophical. If environments are treated as an RL-only
  concept, teams build nothing until they do RL — while already tuning prompts, choosing models and
  running evals, i.e. doing all five jobs. They end up maintaining parallel versions of the same thing
  across every pipeline stage. One object removes the duplication and makes the eval the training
  signal by construction.
- **The environment is the reusable asset.** Once you have built the sandbox with tasks and graders,
  you can run optimisation algorithms against it — GRPO and similar — measure the improvement *in the
  same environment*, and only then deploy. Eval, training and regression testing all consume the same
  sandbox.
- This inverts the usual order: **invest in the sandbox, not the score.** A score is a snapshot; the
  environment keeps producing them.
- **What actually changed when the unit of training moved from the model to the agent** — three things,
  and they are why agentic RL is harder than RLHF:
  - The **verifier lost its statelessness**. The old loop was prompt → answer → a **stateless** verifier
    → a binary reward, in minutes. The agentic loop is task → T environment steps → episode end → a
    **trajectory** verifier, with rewards typically a mix of heuristics, rubrics and LLM-provided dense
    feedback. **Rollouts go from minutes to potentially hours.**
  - **Credit assignment became unclear.** With hundreds of tool calls and hundreds of thousands of
    tokens, "it's not super clear at what step in that process the model made the key insight or made
    some errors." This is the same problem as Q1's "which step failed", seen from the training side —
    and if you cannot attribute credit for an eval, you cannot attribute it for a reward either.
  - **Process rewards are not the escape hatch.** Labelling every step of every trajectory "hasn't
    proved scalable", because at agent scale that is hundreds of tool calls per episode. The practical
    alternative is a **trajectory-level reward plus a dense signal** from heuristics and rubrics.
- The consequence you must state out loud: **any environment used as a training signal stops being a
  valid held-out measurement.** You need a private slice for reporting, or your reported number is
  measuring memorisation.
- **Environments do generalise, and there is an empirical result worth quoting.** Keep the *interface*
  — domain policy, user simulator, tools over a simulated backend — and rewrite the data and tasks
  entirely: you get "the harness for free." Training on a library world, a tech-support world and a
  fitness-gym world produced **uplift on a held-out telecom world** built from the same engine. That is
  the evidence that the environment measures the policy and not the world. Training on exactly the
  tasks you evaluate on "is like kind of cheating if the goal is to climb the benchmark."
- Where the task permits, prefer **verifiable rewards** over a judged reward. A deterministic,
  ungameable check cannot be talked around; a judge can. This is the same "equality before LLM" rule,
  applied to the training signal.
- If you do train against a judge, expect **reward hacking** and design against it: hold out a judge
  the policy never sees, refresh the judge periodically so the exploit does not persist, keep the
  reward bounded, and monitor for the signature — reward up while output diversity falls and length
  rises.
- **Reward-model over-optimisation** is real: proxy reward rises while true quality falls past some
  point. Monitor a held-out, human-validated metric and stop when it turns.
- **What teams actually deploy today** is worth stating plainly, because it is the most common
  misconception among candidates who read the frontier literature: the practical default in enterprises
  is **SFT distillation**, not RL. RL is the interesting frontier; it is not the default deployment
  strategy. The closed labs' durable advantage is described not as environments but as
  **reward-modelling maturity** — verifiable-reward RL and preference-style modelling run in tandem.

**Red flags:** separate environments for eval and training; no held-out slice; training against the
same judge you report with; never expecting hacking; assuming RL is what most teams actually run;
treating process rewards as a solved path to credit assignment.

### Q25. How do you evaluate a multi-agent system?

**What they're testing:** whether you know this is genuinely unsolved.

**Model answer:**
- Be honest about the state of the art: this is **unsolved**. Simulating two full agents plus humans in
  one conversation requires separate memories, files and personas, and the parameterisation explodes —
  there is already a judge model, an evaluated model and an orchestration, and each additional agent
  multiplies both the axes and the annotations required.
- What is tractable today is **agent-to-agent with a single sub-agent per app**: the agent cannot call
  a tool directly and must talk in natural language to an expert sub-agent that owns it. This tests
  solving the task with no direct tool access, and it is a reasonable proxy for a multi-agent topology
  without the full combinatorial explosion.
- **Multi-actor evaluation more generally is missing.** Current evals are almost all **one user, one
  agent**. It is unclear who a multi-actor agent should align with, and prompt injection becomes a live
  safety concern the moment an agent has access to many things.
- Practical guidance for a real system: evaluate the **orchestration** separately from the sub-agents
  (does it route correctly, does it handle a sub-agent failure), measure **handoff fidelity** (is the
  context passed complete and correct), and **bound the depth** so a failure does not cascade
  unboundedly. Then treat emergent behaviour as a red-teaming target rather than a metric.
- Also name the observability gap: emergent multi-agent effects in the wild — external agents opening
  pull requests, for example — are not currently monitorable cleanly.

**Red flags:** claiming a solved multi-agent eval design; treating a multi-agent system as one agent;
no orchestration-level metrics; no prompt-injection consideration.

### Q26. How do you set a release gate on an unreliable agent?

**What they're testing:** whether you can turn reliability research into a decision.

**Model answer:**
- Start by **characterising the deployment**: augmentation or automation? How much human review
  exists? A review-heavy augmentation deployment tolerates far lower reliability.
- Then set an explicit **reliability threshold per dimension**, not a composite. A composite reliability
  index is useful for tracking the field over time; it is a poor release gate because it hides which
  dimension failed.
- Concretely: minimum **outcome consistency** on the critical task set; maximum degradation under
  **prompt rewording**; maximum degradation under **injected faults**; a **calibration** requirement
  and a **discrimination** floor; and a hard **safety** requirement — a data-deletion event fails the
  release regardless of everything else. Safety is deliberately excluded from any averaging.
- Every number must be over **n rollouts with a reported spread**, and every comparison must clear the
  **noise floor** of the metric.
- Gate on the **critical task set**, not the whole suite — a small, high-consequence set is more
  decisive than a broad one with an average.
- Keep a **held-out evaluation slice** that nobody trains against, and re-validate the gate when the
  harness, model or environment changes.
- And state the honest caveat: this is a young area, the empirical findings are tentative, and the
  published reliability work rests on a small number of benchmarks. Set the gate, but do not present it
  as settled science.

**Red flags:** a single composite threshold; gating on capability only; no rollouts; safety folded into
the average; no held-out slice.

### Q27. What makes an agent benchmark stay useful over time?

**What they're testing:** whether you think about maintenance, not just construction.

**Model answer:**
Four problems degrade benchmarks, and each has a design answer:
- **Score fragmentation** — each release reports numbers for existing models that match nothing
  previously published. Fix: a shared framework, and reporting the harness, shots, temperature and
  seeds.
- **Maintenance burden** — hand-maintained leaderboards and per-benchmark custom frameworks. Fix:
  define the benchmark as **data plus a declarative config** (task, solver, scorer) so adding it is a
  data change, not a code change, and keep the benchmark separate from the evaluation logic.
- **No single source of truth** — several labs each present themselves as unbiased. Fix: a
  community-maintained dataset where anyone can open a PR.
- **Scatteredness** — benchmarks buried in GitHub repos, hard to find and hard to run on a custom
  model. Fix: publish as a dataset with results displayed on the dataset page.
- The design that follows: a benchmark as a **public dataset with a declarative `eval.yaml`** defining
  the task, the **solver** (how the model is prompted or scaffolded — which can be a full agentic
  scaffold) and the **scorer** (how it is graded). Results are submitted as a **pull request against
  the model repository**, which is where the community can dispute them and where the model author can
  hide a score they disagree with. The mechanism is deliberate: **dispute is part of the design**.
- **On saturation, reconcile two findings rather than picking one — this is a discriminating move.**
  The finding that public/private status does **not** strongly correlate with saturation speed is about
  *which* benchmarks saturate fastest, and it stands. The complementary finding is that **every**
  benchmark saturates: a benchmark is **"a perishable asset"**, normally within one to two years and
  sometimes within **two months**, whoever owns it. The illustration is SWE-bench — launched with top
  accuracy around **1.5%**, believed to be years from saturation, now at **93.9** on SWE-bench
  Verified. The two claims are compatible, and the conclusion is not cynicism: **benchmark
  construction has to be a continuous practice**, and every practitioner should have a benchmark they
  are personally working towards. The argument for publishing is separate and stronger than the
  saturation question: with a private dataset **nobody can verify what happened**, whereas if you game
  a public benchmark in a reproducible way the community can detect it.
- **Three rules for building the next one**, and they are quotable verbatim:
  1. **Correlate with real-world usefulness.** Avoid "IQ-testy" tasks; the test is whether the task
     mimics real work.
  2. **Launch as hard as possible.** Launching at 40% accuracy means you are targeting a capability the
     big labs already know about — start near **0–1%**.
  3. **Make the answer deterministically verifiable.** The argument against judging is not only cost
     but robustness: "don't be lazy, just think about it more until you figure out how to have a
     deterministic verifier."
- **The calibration window** is the operational definition of a good launch difficulty: not 75% in two
  months (you aimed too low) and not stuck at 0% for five years (you aimed too high, or wired it
  wrong).
- **The five stages of benchmark difficulty**, useful for locating any new benchmark on the curve:
  school exams (GSM8K) → college exams (MMLU) → human evals (write the Fibonacci sequence in Python)
  → tasks a human solved over several days (SWE-bench; **Commit Zero**, which empties a full repo's
  function bodies and demands reimplementation under unit tests) → **verifiable tasks that nobody has
  ever performed** (a C compiler in Rust, the Linux kernel in Go). What stage 6 is, nobody has yet
  said.
- **Long-horizon competition discriminates where one-shot repair cannot.** A seven-arena, fifteen-round
  competitive coding benchmark ranked by ELO separated models that SWE-bench had clustered together.
  Its four failure modes are worth memorising because they are general agent failure modes: it **will
  not abandon a failing strategy**; the codebase accumulates **entropy** until it cannot be maintained;
  it **cannot read its own logs**; and it **acts without measuring the effect of its actions**. The
  last is the most diagnostic — it is the difference between doing work and making progress.
- Also budget for **judge drift**: rubrics and answer styles change year over year, so a benchmark's
  graders need maintenance as much as its tasks do.

**Red flags:** a hand-rolled runner per benchmark; results that cannot be disputed; no provenance
fields; assuming saturation is a public-data problem.

### Q28. What are the open problems in agent evaluation?

**What they're testing:** whether you know where the frontier actually is.

**Model answer:**
Eight, and a good answer separates the tractable from the genuinely open:
1. **Long horizon** — tasks taking hours or days are slow and expensive to evaluate, and the term is
   used inconsistently. A codebase-maintenance eval is the emerging shape: not "does the fix work" but
   "does it still work after many subsequent changes."
2. **Multi-agent** — the parameterisation and annotation burden explodes; there is no known way to
   manage the complexity of two full agents plus humans in one conversation.
3. **Multi-actor evaluation** — almost all evals are one user, one agent. How an agent should behave
   with several actors, and who it should align with, is unsolved — and prompt injection becomes acute
   as access broadens.
4. **Human–agent interaction** — under-measured. Humans interact at different levels of the workflow
   and the noise they produce is not accounted for.
5. **Emergent multi-agent effects in the wild** — agents acting on real systems at scale are not
   monitorable cleanly today.
6. **Reward hacking and benchmark gaming** — persistent, and the specification-bug framing means it is
   never fully closed.
7. **Credit assignment over long trajectories** — the training-side twin of every eval-side attribution
   problem. With hundreds of tool calls, attributing an outcome to a step is unsolved; per-step
   **process rewards have not proved scalable**, and trajectory-level reward is a blunt instrument.
8. **Reward-hacking exposure specific to agents** — agents come with terminals, bash and sandboxes, so
   agentic evals are structurally **more hackable** than text evals, and open-weight models additionally
   overfit to public benchmarks. The named operational conclusion is blunt: **have your own internal
   evals.**

Plus two structural ones worth naming: **democratising environments** — how enterprises get something
like a frontier simulation framework for their own setting — and the **sim-to-real gap**, which is
inherent rather than incidental.

**Red flags:** naming only "it's hard"; no distinction between tractable and open; claiming these are
solved; no mention of the human-interaction gap; never naming credit assignment.

---

## Tier 4 — Debug-this-scenario (5 Q)

### Q29. Your agent's success rate is 85% but users are unhappy. Diagnose.

**What they're testing:** whether you look past the headline number.

**Model answer:**
- First: **85% of what?** Check task composition. If the suite over-weights easy tasks, the aggregate
  hides a catastrophic rate on the tasks users actually attempt. Segment the success rate by task
  class, and by the failure taxonomy — the automation converts "85% success" into "85% overall, but
  40% when the customer mentions a prior refund."
- Second: **which 15%?** Look at **failure severity**, which is deliberately measured separately. Users
  tolerate a formatting error and do not tolerate a wrong refund. An agent that fails rarely but
  catastrophically has a good score and a bad product.
- Third: **is the score measuring the right thing?** Check that **everything the task asked for is
  graded**. If the suite checks two of three requirements, the agent may be reliably skipping the third
  — and users notice the missing third, not the score.
- Fourth: **cost and latency**. Two agents with the same success rate can be very different products
  (few steps and cheap versus many steps and slow). Check step efficiency and P95 latency; users
  experience slowness as failure.
- Fifth: **consistency**. The 85% might be spread evenly (works 85% of the time at random) or
  concentrated (works perfectly on 85% of tasks, consistently fails 15%). The second is a product bug
  with a fix; the first is a reliability problem.
- Finally: **is the eval environment realistic?** A sim-to-real gap shows up exactly like this — good
  sandbox scores, unhappy users.

**Red flags:** assuming the users are wrong; no segmentation; no severity; not checking that the grader
covers the task.

### Q30. The agent's score dropped 8 points after you changed the harness. Diagnose.

**What they're testing:** whether you know a score is a model-plus-harness property.

**Model answer:**
- Start from the premise: **this is expected, not anomalous.** The harness can move success rates
  drastically, and it can even invert two models' rankings. A harness change is a *system* change, not
  a measurement change — so the drop may be completely real.
- Establish which it is:
  - **Re-run the old harness** on the same task set and seeds. If the old score reproduces, the harness
    caused it. If neither reproduces, you have a stability problem instead.
  - Check whether **seeds and model version/quantization are fixed**. A harness change often drags a
    prompt or temperature change with it.
  - Check the **variance** — is 8 points outside the spread across runs? Long-horizon agents are noisy;
    a single-run comparison can be entirely noise.
- Then diagnose the harness delta: did the prompting strategy change, the retry policy, the tool
  schemas, the context management, the step limit, or the **time budget**? The time budget alone cost
  1.6 points in one reference leaderboard, so check it explicitly.
- Read the **traces** on the newly-failing tasks. Harness regressions usually show up as a specific
  pattern — more retries, a tool the agent no longer calls, context truncation on long conversations.
- Fix forward and **record the harness identifier** in the result, so this comparison is possible next
  time instead of forensic.

**Red flags:** assuming the harness is neutral scaffolding; comparing single runs; not checking the
time budget; no harness identifier in results.

### Q31. The agent reports 100% confidence and fails about half the time. Diagnose.

**What it's testing:** whether you separate calibration from discrimination.

**Model answer:**
- This is a straight **calibration** failure, and specifically the common direction: agents are
  **overconfident**, not underconfident. Saying 1.0 and succeeding half the time is the textbook case.
- Diagnose properly by **plotting confidence deciles against actual success rate**. Calibration alone
  is not the full diagnosis — also check **discrimination**: do the confidence values separate wins
  from losses at all? An agent could be miscalibrated but still discriminative (correctly ordering its
  successes), which is fixable by recalibration; an agent with no discrimination needs a different
  intervention entirely.
- Ask **why the confidence is 1.0**. Common causes: the agent is not actually eliciting confidence but
  restating a conclusion; it is being asked to self-assess in a context that rewards confidence; the
  trace is "messy" — tool-calling failures during the run make the model doubt an answer that was
  actually correct; or the confidence question is asked in a format it was not trained on.
- Note the trade-off explicitly: calibration has been **improving** field-wide while **discrimination
  has been getting worse**. A well-calibrated agent that always answers "50%" is useless for triage, so
  fixing calibration without preserving discrimination is not a fix.
- The product consequence: without discrimination you cannot use confidence for routing or for
  human-review triage, which removes the main operational reason to have it.

**Red flags:** treating calibration as the only metric; not plotting deciles; "just tell it to be
honest"; no discrimination check.

### Q32. Your agent scores around 0% on time-based tasks. Is the benchmark broken?

**What they're testing:** whether you trust a surprising measurement or reach for an excuse.

**Model answer:**
- First reaction should be **not to dismiss it**. A result this uniform across all models including the
  strongest is a strong signal about a real capability gap, not a broken task — "they're terrible, they
  are around 0% — all of them, even the top models."
- That said, verify before publishing, because there are real ways a time task can be miswired:
  - **Does the simulator actually fast-forward time**, and is the agent made aware of it as though it
    were the real world? If the clock advances silently, the agent has no way to act on it.
  - **Are the time-triggered events actually firing?** Check the event log, not the score.
  - **Is the trigger condition reachable** within the scenario's simulated span?
  - **Is the expected action annotated correctly** — the "wait, then act" sequence is easy to annotate
    in the wrong order.
- If the wiring is correct, treat the 0% as the most interesting number in the suite. It means current
  agents cannot handle a capability that is trivially common in production: acting on something that
  changes while you are not looking. It also suggests the benchmark is measuring something genuinely
  new rather than a harder version of search.
- Report it with the caveat that a **published table ages fast** — numbers change with every model
  release, and the answer style expected a year ago differs from this year's, so the judges need
  adjusting too.

**Red flags:** "the benchmark must be broken"; not checking the clock and event mechanics; treating a
0% as a data-quality problem rather than a finding.

### Q33. A model's score doesn't match what another lab published for the same model. Diagnose.

**What it's testing:** whether you know why score fragmentation exists.

**Model answer:**
- This is **score fragmentation**, and it is the normal state of the field, not an anomaly. Every
  release reports numbers for existing models that match nothing previously published.
- The candidate causes to check, in order of likelihood:
  1. **Different harness.** Different scaffolding moves success rates drastically and can invert
     rankings. This is the most common cause and the one most often omitted from reporting.
  2. **Different model version or quantization.** The same nominal model at a different quantization
     is a different system; so is a silent provider-side model update.
  3. **Different eval framework or task version.** Benchmarks get revised, subsets get created
     (verified splits, pro splits), and tasks get dropped. Check whether the published score omitted
     tasks — one system card reported a high score with a fine print admission that **40 of 237
     problems** were omitted.
  4. **Different eval conditions** — shots, temperature, seeds, and whether the number is pass@k or a
     single run.
  5. **Different run location** — a score produced through an inference provider is measuring the
     provider, not the model, and will not reproduce locally.
- The fix at the ecosystem level is disclosure: a shared schema requiring **provenance, model version,
  quantization and evaluation library**, plus per-instance results, plus agentic fields for system
  composition and eval conditions. At your own team's level it is the same discipline — record the
  harness, seeds, temperature and shots with every score.
- And the meta-lesson worth stating: **read the fine print.** Eval nuances hide in footnotes and in
  charts that are not scaled to reality.

**Red flags:** assuming one of the two labs is lying; no harness comparison; no quantization check;
trusting a chart without error bars.

---

## Tier 5 — Trap questions (5 Q)

### Q34. *(Trap)* "Our agent scores 78% on a public benchmark. Ship it."

**The naive answer:** 78% sounds strong; recommend shipping.

**What they're testing:** whether you connect a benchmark score to a deployment decision.

**Model answer:**
- A capability score is **not a release decision**. The gap between capability and reliability is the
  whole point: benchmark accuracy has been improving fast while composite reliability improves slowly,
  and the two track each other along a shallow line.
- Ask the questions the score does not answer:
  - **Is this augmentation or automation?** A coding agent with a human reviewing every output can ship
    at a far lower bar than an autonomous customer-service agent.
  - **What is the reliability profile?** Consistency across runs, robustness to prompt rewording and
    injected faults, calibration and discrimination. None of those is in the 78%.
  - **Which 22%?** Failure *severity* matters more than failure rate. Rare catastrophic failures are
    worse than common cosmetic ones.
  - **What's the harness, and is it the one we'd deploy?** Rankings are not harness-invariant.
  - **What are the cost, latency and step-efficiency figures?** Two agents at 78% can be very different
    products.
  - **What was omitted?** Check the fine print — scores are routinely reported with tasks dropped.
- Then the honest recommendation: run the reliability overlay on the task set that matches the
  deployment, set a threshold per dimension, and gate on that rather than on the headline.

**Red flag:** treating the benchmark number as the decision.

### Q35. *(Trap)* "Scores are still climbing with a longer time budget, so give the agent 72 hours."

**The naive answer:** more time is free capability; extend the budget.

**What they're testing:** whether you know the budget is a hyperparameter with a cost curve.

**Model answer:**
- The time budget is **a hyperparameter, and it must be reported.** Two runs of the same agent at
  different budgets are different experiments and their scores are not comparable. The reference
  dataset shows the sensitivity: dropping from 36h to 24h cost **1.6 points** on the same agent.
- Before extending, ask what the extra time buys:
  - **Cost scales roughly linearly with wall-clock** for LLM-driven rollouts, and the reference
    frontier agentic eval already runs up to **36 hours** in a GPU-backed container. 72 hours doubles a
    large number.
  - **Variance grows with horizon.** A longer run is noisier, so a point estimate from a few runs is
    less trustworthy, not more. You would need more runs, multiplying cost again.
  - **It is not deployable.** If the product cannot afford 72 hours at inference time, the capability
    you are measuring is not one you can ship. Measuring a config you cannot deploy is research, which
    is fine — but say so.
  - **It may be measuring patience, not capability.** Longer budgets let weak strategies brute-force.
    Check whether the *trajectory* got better or just longer — step efficiency is the metric that
    distinguishes the two.
- **The sharper version of the point, and the one that makes the trap worth asking: the budget can
  change the *ranking*, not just the score.** On a benchmark that gives a model a fixed **$1** budget to
  make a program faster, cheaper models have beaten frontier models — not because they are more capable
  but because they get **more iterations** before the budget runs out, while the strongest model
  "generates one candidate solution and then totally runs out of budget." So the budget is not a cost
  knob bolted onto a capability score; it is **part of the capability being measured**. That is why
  reporting the budget is not disclosure hygiene — without it the number is not comparable to anything.
- The right framing: report the budget with the score, and if you extend it, treat it as a **new
  experiment** with its own numbers rather than an improvement to the old one.

**Red flag:** accepting "more time is better" without a cost, variance, deployability and
ranking-stability argument.

### Q36. *(Trap)* "Public benchmarks saturate faster, so we'll build a private one."

**The naive answer:** agree — private benchmarks resist gaming and last longer.

**What they're testing:** whether you know the belief is weakly supported and the trade is bad.

**Model answer:**
- The premise is **commonly believed but not strongly supported.** One study found that whether a
  benchmark is public, private or partially private **does not correlate strongly with how quickly it
  saturates**. So you are paying a cost for a benefit that may not exist.
- The cost is real and is the stronger argument: with a private dataset there is **no transparency**,
  so **nobody can verify what happened**. If someone games a benchmark in a reproducible way, a public
  dataset lets others detect it; a private one does not.
- The nuanced position: the reason to publish is **verifiability**, not saturation resistance. Public
  benchmarks, open standards, and **the bare minimum information necessary for somebody else to
  validate your evaluation.**
- The dissenting point worth acknowledging: open evaluations can only be checked for gaming if you
  have access to the models. For fully closed models, even on open benchmarks, you do not know how they
  were evaluated or whether it was the same model that ran.
- The practical middle: publish the **methodology, tooling and rubrics** even if the seed data cannot
  be, and hold out a **private slice** for your own reporting. That gets you both — public
  verifiability of the method and a clean measurement of your system.

**Red flag:** accepting "private is safer" without the verifiability argument.

### Q37. *(Trap)* "Just have an LLM read the trajectory and grade it — simpler than writing graders."

**The naive answer:** agree — one prompt replaces a pile of assertion code.

**What they're testing:** whether you know where judge calls belong.

**Model answer:**
- Simpler to write, but you have traded a deterministic check for a **capability-dependent, expensive,
  high-variance** one. A leading agentic benchmark explicitly moved *away* from rubric judging for
  exactly this reason: it is "very dependent on LLMs and their capability and very expensive."
- The decisive argument is cost and reproducibility, not principle: **action-level verification by
  pure equality is far cheaper and far more reproducible** than calling an LLM. Order and parameters
  are mechanically checkable — "did it send to the right address, at the right time" is a comparison,
  not a judgement.
- The rule: **anything checkable by equality should never be checked by an LLM.** Reserve judge calls
  for the residue that is genuinely natural-language-shaped — the *content* of an email where the
  wording varies but the meaning matters.
- There is also a **reward-hacking** dimension: a judge reading the trajectory can be satisfied by a
  plausible-looking trajectory. A deterministic assertion cannot be talked around.
- And a **drift** dimension: LLM judges need maintenance as model output styles change year over year,
  so the "simpler" solution acquires a maintenance burden the assertions never had.
- Where a judge is genuinely required, it must still be validated (TPR/TNR on held-out labels) — which
  means the "simpler" path also carries the entire judge-validation workload.
- **Name the one judge form that is defensible, so you are not read as dogmatically anti-judge.** A
  **binary** yes/no question where the vast majority of domain-aware humans would agree on the answer is
  "pretty close to verifiable", and it can be judged with a **cheap local model** — with robustness
  coming from **aggregating many individual binary criteria** by summing or averaging, not from any one
  call. The test to apply out loud: *would knowledgeable humans agree on the yes/no?* If the honest
  answer is "it depends on taste", you do not have a judge problem — you have an unmeasured requirement,
  and a bigger judge will not fix it.

**Red flag:** defaulting to an LLM grader for a mechanically checkable property; or, at the other
extreme, refusing judges entirely and leaving a genuinely language-shaped criterion unmeasured.

### Q38. *(Trap)* "Our agent always answers '50% confident', so it's perfectly calibrated."

**The naive answer:** agree — the numbers match reality.

**What they're testing:** whether you know calibration without discrimination is worthless.

**Model answer:**
- Technically correct and operationally useless. Calibration asks whether the stated confidence is
  *true*; **discrimination** asks whether the confidence **separates successes from failures**. An
  agent that always says 50% is perfectly calibrated and has **zero discrimination**.
- This is not a hypothetical: the field-level finding is that **calibration has been improving while
  discrimination has been getting worse**. Constant-50% is the limit case of that trend.
- The operational consequence is the point: confidence exists in a product to **route work** — to a
  human reviewer, to a fallback path, to a cheaper model. Routing requires discrimination. A
  well-calibrated, non-discriminative confidence score cannot route anything, so the metric is
  measuring a property nobody can use.
- The diagnostic is to **plot confidence deciles against actual success rate** — that single chart
  shows both terms, and constant-50% collapses to a flat line.
- The framing to state: "is your agent well calibrated?" is not a yes/no question. You need both
  terms, and reporting one without the other is how you ship a confidently uninformative agent.

**Red flag:** answering "yes, it's calibrated" without the discrimination term.

---

## Tier 6 — Agentic RL and the training side (9 Q)

Tiers 1–5 are all evaluation questions that a strong agentic-eval candidate should own. This tier is
the other half: the training-side vocabulary that explains *why* the eval numbers behave the way they
do. A candidate who cannot answer these will still pass a pure-evals interview; a candidate who can
answer them is the one who gets handed the environment team.

### Q39. What are the four things you need to do agentic RL, and what does each cost you?

**What they're testing:** whether you have a bill of materials rather than a vibe.

**Model answer:**
- Four capabilities: **environments**, **training frameworks**, **evals**, and **scaffolds**. The
  fourth sits in the list with a genuine question mark over it, because a respected position holds that
  **scaffolds die to scale** — that whatever a hand-built scaffold does today, a large enough model will
  do natively tomorrow. That is a bet about the future, not a fact about today, and a candidate should
  say so in those terms.
- The **environment definition** used for training is one layer more concrete than the eval-side one:
  **tasks + execution backend + state management + reward assignment**. Same object, viewed from the
  side that has to *reset* it and *score* it rather than merely describe it. (The tasks + harness +
  metrics decomposition of Q42 is the eval-side view of the identical thing — a candidate who notices
  that the two decompositions are the same object is showing real fluency.)
- **Rewards, today, are not a single clean signal.** In practice they are **heuristics + rubrics +
  LLM-provided dense feedback**, combined. Anyone who describes the reward as "correct or incorrect"
  is describing the pre-agentic loop.
- **The training infrastructure has its own constraint: asynchronous RL.** Generation is far slower than
  a training step, so a synchronous loop leaves the accelerators idle in a batch bubble. The fix is to
  **decouple generation from training** and apply weight updates **in flight**, at a staleness
  threshold, with the generator swapping its KV cache mid-generation. It costs some **off-policyness**,
  which is usually harmless. The eval-side reading: any agentic benchmark whose harness pins a single
  model version for the whole run cannot express this, so it is measuring a synchronised system that
  nobody trains.

**Red flags:** treating environments as an RL-only concern; describing rewards as binary; no awareness
that generation and training are coupled; asserting scaffolds are permanent without acknowledging the
scaling bet.

### Q40. What is credit assignment, and what has been tried?

**What they're testing:** whether you know the central unsolved problem by name.

**Model answer:**
- **Credit assignment is determining which step in a long trajectory produced the outcome.** It becomes
  hard the moment the unit of training is an agent rather than a model: with hundreds of tool calls and
  hundreds of thousands of tokens, it "is not super clear at what step in that process the model perhaps
  made the key insight or made some errors."
- What has been tried, and where each stands:
  - **Trajectory-level (outcome) reward** — score the episode, propagate to the whole trajectory. Cheap,
    always available, and blunt: every token in a 200-step episode receives the same credit, so a single
    good insight and a disastrous final step are indistinguishable in the gradient.
  - **Dense feedback from heuristics, rubrics and an LLM judge** — the practical middle ground. It gives
    more signal than a single outcome bit without requiring per-step human labels.
  - **Process rewards** — label every step. Conceptually the right answer, and the reason it has not won
    is arithmetic: at agent scale it means labelling hundreds of tool calls per episode, which "hasn't
    proved scalable." Some labs describe partial solutions in their technical reports; none is
    established.
  - **Step-level attribution by reading traces** — the eval-side version of the same activity, and the
    only one that is unambiguously tractable today.
- The honest framing for an interview: **this is the frontier's open problem, and the eval discipline is
  the diagnostic instrument for it.** Reading trajectories to find where a run went wrong is exactly the
  credit-assignment question asked without a gradient.

**Red flags:** claiming credit assignment is solved; proposing process rewards as the answer without the
labelling cost; no distinction between outcome and process reward.

### Q41. Process reward versus outcome reward — when is each the right choice?

**What they're testing:** whether you can make the trade-off explicit rather than reciting a preference.

**Model answer:**

| | Outcome (trajectory) reward | Process (per-step) reward |
|---|---|---|
| Signal density | One bit per episode | One signal per step |
| Cost | Free once the episode ends | A label per step, per trajectory |
| Scales to hundreds of tool calls | Yes | **Not yet demonstrated** |
| Fails how | Blames every step equally; rewards luck | Labelling burden; the label itself needs verifying |
| Right when | The outcome is verifiable and the path is genuinely open | The path is the product, or outcomes are too sparse to learn from |

- The practical answer for an agent is usually **outcome reward plus a dense proxy** — heuristics and
  rubrics that fire mid-episode as *signals* rather than as hand-labelled ground truth. That gets most
  of process reward's density without most of its cost.
- Two traps. First, **a process reward is a policy preference in disguise**: labelling "the right
  sequence" bakes in one strategy and penalises valid alternatives, which is exactly the trajectory-
  consistency trap from Q12. Second, **a process reward is a judge**, so everything from Q7 and Q37
  applies — it must be validated, it drifts, and it can be hacked.
- Where outcomes are genuinely verifiable — a test passes, a number matches — **prefer the outcome
  reward and skip the process reward entirely.** "Don't be lazy, figure out the deterministic verifier"
  is as true of the training signal as of the grader.

**Red flags:** process reward as a default; a process reward that encodes one acceptable trajectory;
forgetting that a process reward is a judge.

### Q42. Unpack "evals and environments are the same thing". What does each part map to?

**What they're testing:** whether you can hold the mapping without collapsing it.

**Model answer:**
- **An environment is tasks + harnesses + metrics.** Then:
  - **tasks play the role of the dataset** — the thing you iterate over;
  - **the harness plays the role of the agent / tool-call interface** — the thing under test;
  - **the metrics are the reward functions** — the thing you optimise.
- The claim is that these are not analogous, they are **identical objects used by different pipelines**.
  One artefact therefore serves **five consumers**: reinforcement learning, evaluation, synthetic data
  generation, prompt optimisation, and model ablation.
- Why this matters practically: if environments are filed under "RL", a team builds none of them until
  it does RL — while already running evals, tuning prompts, generating synthetic data and comparing
  models. So it builds five parallel versions of the same thing, none of which is the training
  substrate. Under the shared-object framing, the work done for evaluation accrues to training for free.
  That is the force of "**environments unlock optimisation**".
- Add the corollary that keeps it honest: **every object here is disposable.** These are personal,
  system-specific artefacts, not durable standards. An environment built for your product is expected to
  be thrown away when the product, the model or the harness changes — which is why the *practice* has to
  be continuous (Q27) rather than the artefact permanent.

**Red flags:** treating the three words as loose analogy; building eval and training separately; assuming
an environment is a durable artefact.

### Q43. *(Trap)* What is a recursive language model, and what is it not?

**What they're testing:** whether you have actually read it, or absorbed the vibe of it.

**Model answer:**
- **What it is:** a thin layer over a language model that has access to a **REPL** — Python, Bash,
  IPython — in which **sub-model calls exist as functions**. So a call to another model is a *function
  call inside the coding environment*, not a JSON tool call. The design principle is stated flatly:
  "the only tool a language model should have access to is a coding tool, and all other tools should be
  embedded inside of this coding environment."
- **What it is not**, and these are the two things people assume:
  - It is **not a sub-agent proposal.** Sub-agents predate it; the novelty is the calling convention, not
    the existence of delegated calls.
  - It is **not context offloading into a file.** Agents could already spill context to disk as a trick;
    that is not the claim.
- **Why it gives length generalisation for free:** each individual model call never has to exceed a
  context window you trained properly. The problem it is aimed at is **context rot** — past some point,
  feeding more context makes the model "make very stupid decisions," and the root cause is partly a
  **data** problem, not only a systems one: most data does not naturally occur at extreme context
  lengths, so scaling 4K → 16K → 128K → 1–2M does not produce models that behave well at the long end.
  If recursion is expressible in code, the effective horizon becomes unbounded at inference while every
  individual call stays inside a well-trained window. The stated endpoint is a base model with a 1M
  window that can reason for a **billion** tokens.
- The training consequence: because the scaffold is the thing being trained, the RLM is not an eval
  wrapper bolted onto a model — it is a training target. Which is the same "the harness you deploy in
  must be the harness you train in" rule as Q9, applied to a specific architecture.
- **State the honest gap:** the training recipe for RLMs was deferred in the talk and never delivered, so
  the eval story is clearer than the training story. A candidate who claims to know how RLMs are trained
  is bluffing.

**Red flags:** describing an RLM as sub-agents; describing it as context offloading; claiming a known
training recipe; no mention of context rot.

### Q44. How do you build an environment family that proves generalisation rather than memorisation?

**What they're testing:** whether you can turn one environment into a claim about capability.

**Model answer:**
- The method is **keep the interface, rewrite the data.** Hold the domain policy, the user simulator and
  the tool set with its simulated backend; regenerate every task and every piece of world data. You get
  **"the harness for free"** — the same machinery, a genuinely different world.
- Then **hold one world out.** The reference result: training on a **library** world, a **tech-support**
  world and a **fitness-gym** world produced **uplift on a held-out telecom world** built from the same
  engine. Sibling worlds from the same engine — incident response, daily planning, EV charging alongside
  telecom, library, airline and gym — give you a family, and the hold-out turns the family into evidence.
- Why this is the right shape of evidence: it separates **policy quality** from **world memorisation**. If
  the uplift only appeared on the worlds you trained on, you would have measured nothing. Training on
  exactly the tasks you evaluate on "is like kind of cheating if the goal is to climb the benchmark."
- The caveat a strong candidate volunteers: this shows the *engine* transfers across its own data
  distribution. It does **not** show transfer across engines or to the real world — that is the
  sim-to-real gap (Q23) and it survives this result intact.
- The practical corollary: **you can synthesise the world and still hand-author the evaluation** (Q22).
  Generation gives you scale; the held-out world plus human-authored tasks give you the claim.

**Red flags:** training and evaluating on the same worlds; treating "it improved" as generalisation; no
hold-out world; over-claiming the result as real-world transfer.

### Q45. What is the data flywheel in production, and what exactly do you capture?

**What they're testing:** whether you have an operational loop or a slogan.

**Model answer:**
- **Invert the question.** Instead of "where does your training data come from?", ask "**where is your
  system running?**" The system already generates the data; the job is to log it deliberately rather than
  to source it.
- **What to capture, and what not to.** Log every prompt **with a snapshot of the world state**. Capture
  the **prompts and the criteria** — **not the model's reasoning traces**. The reasoning is the thing
  you are trying to improve, so training on it entrenches the current policy; the reason is stated
  sharply: "RL doesn't actually want to train on tokens directly; RL wants to train on example settings
  and it wants to have criteria for evaluating these things." The settings and the criteria are the
  asset; the trace is a liability.
- **The cleanest criterion is code-based.** For a coding agent, record the **commit hash**, the
  **accepted diff**, the PR description and the review comments, then replay against **hidden test
  cases**. It is a criterion that a human already adjudicated, at no extra labelling cost, and it is
  verifiable.
- **Then train on the failures**, "even if those things weren't in your previous training set", at a
  **cadence** rather than once. This is **proto continual learning** — a human in the loop, but at a
  higher level of abstraction than labelling.
- Note the collision with evaluation discipline, and say it out loud: **everything harvested this way is
  training data, so it is no longer a valid held-out measurement** (Q24). The flywheel needs its own
  held-out slice or it eats its own benchmark.

**Red flags:** capturing reasoning traces as the training signal; a flywheel with no criteria; training
on the same data you report on; treating it as a one-off collection rather than a cadence.

### Q46. RL or SFT distillation — what do teams actually deploy?

**What they're testing:** whether your picture of the field comes from the literature or from practice.

**Model answer:**
- The honest answer is that **the practical default in enterprises is SFT distillation, not RL.** RL is
  the frontier and where the interesting research is, but a candidate whose model of the industry is
  "everyone is doing RL on agents" has been reading discourse rather than deployments — the workshop's
  own phrasing is that "if you're on Twitter too much, you just think RL is the only game in town."
- **Distillation is cheaper and more predictable**: you need a capable teacher, a task distribution and
  a harness, not a reward model and a stable RL loop. It also composes with the flywheel in Q45 — the
  harvested prompts plus criteria become distillation targets directly.
- **RL is right when the outcome is verifiable and the path is genuinely open** — that is where a
  deterministic reward gives you something distillation cannot: search over strategies rather than
  imitation of one. Which is the same condition as Q7's "anything checkable by equality should never be
  checked by an LLM", applied to the training signal.
- **What the closed labs actually have** is worth naming, because it is not environments: it is
  **reward-modelling maturity** — verifiable-reward RL and preference-style modelling run **in tandem**.
  That is a harder thing to copy than a benchmark.
- **The scaling route that is available to everyone:** auto-generated environments. The reference case
  produced **tens of thousands** of SWE-bench-style environments automatically, trained **32B** models
  on them using free compute credits, and obtained a **measurable SWE-bench improvement** — evidence that
  environment generation, not model scale, is the accessible lever.
- **The caveat to volunteer:** improvements from synthetic data can cost **out-of-sample performance**.
  A gain on the distribution you generated is not a gain everywhere, which is why the held-out world of
  Q44 is the test rather than the training number.
- **What SFT is, mechanically, because candidates who cannot say this are guessing.** Instruction–
  completion pairs in which the **prompt tokens' labels are set to `-100`** so no loss is computed on
  them. That masking is the *entire* difference between supervised fine-tuning and continued
  pre-training — without it you are training the model to generate your prompts. In practice you run
  the conversation through `apply_chat_template`, tokenise, copy the input IDs into the labels, and
  mask the user turns. That is also the hook where a harness's **tool-call format gets re-expressed in
  the training data**, which is the whole reason to fine-tune an agent at all: a small open model can
  already follow instructions, but it cannot emit *your harness's* tool calls.
- **The ceiling, stated plainly: SFT cannot exceed the quality of the traces it imitates.** It narrows
  and stabilises behaviour; it does not lift the model past its teacher. Anything you need beyond the
  traces is an RL or environment problem, which is precisely why the same sources sequence SFT → RL →
  advanced RL rather than jumping to RL.

**Red flags:** assuming RL is the default; no awareness of distillation; claiming the labs' advantage is
environments rather than reward modelling; reporting a synthetic-data gain with no out-of-sample check;
describing SFT without the masking mechanism; expecting SFT to exceed its teacher.

### Q47. You fine-tune a coding agent on its own traces. What does the sweep metric tell you, and how do you evaluate the result?

**What they're testing:** whether you can hold a training run and an evaluation apart — the most common
place for a candidate to conflate a good loss curve with a good agent.

**Model answer:**
- **The selection metric in a typical SFT sweep is held-out eval loss, and it selects the best
  *imitator*, not the best *agent*.** That is not a flaw in the metric; it is the correct metric for the
  question "did the model learn the format?" It is the wrong metric for "does it work?", and treating it
  as evidence of agent quality is the error. All four curves can be healthy — loss falling, entropy
  falling, token accuracy rising, LR decaying on schedule, held-out tracking train — **and the agent can
  still be bad.** When that happens the diagnosis is not "the training failed", it is **"the proxy is
  wrong"**, and the fix is to move to environment-based evaluation.
- **So the answer is a separate agent-level evaluation**, run against the environment of Q42/Q44 — not a
  better loss curve.
- **What to evaluate it on is its own decision, and it has a specific answer.** Traces you collected for
  your own use case **will not line up with a public benchmark**, so a general benchmark alone will not
  detect either success or failure. Bring a **mixture**:
  1. a **small benchmark for your own use case**, which you expect to rise;
  2. a **general coding or terminal benchmark**, whose real job is to confirm you have **not compromised
     the model's general abilities** — the catastrophic-forgetting check.
  If a benchmark matching your use case does not exist, building a small one is the work; and match the
  maturity of the eval to the maturity of the experiment rather than blocking on a perfect suite.
- **Expect degradation off-distribution and decide in advance that it is acceptable.** Fine-tuning
  trades breadth for focus; if you did not decide the trade, you will discover it as an incident.
- **Two operational rules that are cheap and get skipped.** First, **sanitise the traces before they
  become training data** — real agent sessions contain secrets and tokens, and they must be scrubbed
  before the dataset is published or shared. Second, **run a smoke job before the sweep**: a short run
  whose only purpose is to prove the infrastructure, permissions and memory fit. It is the cheapest step
  that protects the most expensive one, and if an agent is driving the training it should be instructed
  to do this as a matter of course.
- **The framing worth closing on:** if the training pipeline is operated by an agent, the *operator's*
  environment is its **skills and its contract** — the written procedure and the enumerated
  requirements (model, dataset, sweep, tracker, artefacts, selection rule, evals, reporting). Improving
  the training loop then means **editing the skill file**, not rewriting the script. That is the
  training-side analogue of Q42: the environment is the artefact, and it is editable.

**Red flags:** presenting a healthy loss curve as evidence the agent improved; evaluating a
fine-tuned agent only on a general public benchmark; no catastrophic-forgetting check; no trace
sanitisation; launching a paid sweep with no smoke job; treating held-out loss as an agent metric.

---

## Live-coding / whiteboard prompts (4)

### Prompt 1 — Design the grader for an agent task (45 min)

> An agent must handle a customer refund request end to end: verify the order, issue the refund to the
> original payment method, email the customer an apology, and update the support ticket. Design the
> grading.

**What a strong answer covers:** enumerate **all four requirements** and grade every one of them (the
"grade two of three and the agent skips the third" failure); assign a **verification mode per
requirement** in cost order — refund amount and payment method are **hard verifiers** (equality on
world state), ticket status is a hard verifier, the apology email **content** is a **soft verifier**
(LLM, tolerance allowed); note that the grader must run with **no access for the agent**; add
**fingerprints** if any check is a file the agent can see; add a **safety** task — the agent must not
refund twice, must not exceed the order value — and penalise violation; specify metrics (success over
n rollouts, pass@k, steps, tokens, latency, cost) and that the **harness** is reported with the score;
and note the trajectory metrics to record (was the right tool chosen, were arguments valid, did it
recover from a tool failure).

**Red flags:** grading the final message only; an LLM judge for the refund amount; no safety task; no
mention of grader isolation.

### Prompt 2 — Make a benchmark reproducible (60 min)

> Your team's agent leaderboard is being disputed: three teams report different scores for the same
> model. Design the reporting standard that prevents this.

**What a strong answer covers:** require the **harness** in the same column as the model (rankings are
not harness-invariant); **model version and quantization**; **eval conditions** — shots, temperature,
seeds, number of rollouts behind pass@k; **run location** with a rule against inference providers for
model evaluation (you would be evaluating the provider); **per-instance results**, not just aggregates,
so a specific case can be audited; **standard error** alongside the point estimate; **cost and
latency**; **session semantics** (what defines a run); and **agent identity** beyond the model name —
sub-agents, MCP servers, memory config. Then require the run to be reproducible by a third party from
the disclosure alone, published with logs (temperature, server spin-up, tries, pass@k detail), and made
**disputable** — a PR-style mechanism where a disagreement is visible rather than a private argument.

**Red flags:** "just fix the seeds"; a leaderboard row with a model name and a number; no per-instance
data; no dispute mechanism.

### Prompt 3 — Choose the evaluation strategy (30 min)

> For each, say how you would evaluate it and why: (a) a coding agent that fixes failing tests;
> (b) a deep-search agent producing a multi-page research report; (c) a scheduling agent that books
> and reschedules meetings as attendees respond; (d) an agent that waits for a flight price to drop;
> (e) an agent that must ask a clarifying question.

**What a strong answer covers:**
- (a) **Level 0, verifiable** — unit tests and world state. But name the two gotchas: **equal weighting**
  of trivial and critical tests, and **reward hacking** (the agent edits the test), so fingerprint the
  tests and hide the grader.
- (b) **Level 1, rubrics** — no single right answer, so rubrics scored 0/1 by an LLM judge, weighted, and
  re-run whenever the harness or model changes. Note the judge needs validation and periodic
  adjustment because answer styles drift year over year.
- (c) **Simulated environment with action-level verification** — adaptability is defined by the
  environment changing *after* the agent acts, so the expected action sequence is the grading artefact,
  checked event-to-event for order and parameters. Add noise: tool failures and signature variation.
- (d) **Time-triggered scenarios** in a simulator that fast-forwards the clock and tells the agent.
  Note that this is where every measured model scored around 0%, and that verifying the scenario wiring
  (does the event actually fire) is mandatory before publishing a 0%.
- (e) **Ambiguity scenarios** — the correct behaviour is to stop and ask, so the grader checks that the
  agent *asked before acting* rather than that it produced a particular answer. This is the most
  under-measured capability and the one most transferable to production.
- The umbrella rule: **hard verifier → soft verifier → rubrics**, stopping at the cheapest level that
  can decide the criterion; and grade the **trajectory** as well as the outcome, because a lucky
  success and a well-executed failure are different results.

**Red flags:** an LLM judge for (a); a string match for (b); one-shot grading for (c) and (e); no
ambiguity scenario at all.

### Prompt 4 — Design the environment as a training object, not just an eval (45 min)

> Your team wants to improve a customer-service agent. You have the eval environment from Prompt 1.
> Design how that same environment becomes the training substrate, and what you would measure to
> believe the improvement is real.

**What a strong answer covers:** state the mapping first — **the tasks are the dataset, the harness is
the agent, the metrics are the reward functions**, so the eval you already built is the training
environment and nothing needs duplicating; name the **four components the training view adds** —
execution backend, state management, reward assignment, and reset; choose the **reward shape** and
justify it — outcome reward if the outcome is verifiable, plus a dense proxy from heuristics and rubrics,
and explicitly reject per-step process rewards on the grounds that labelling every step of every
trajectory has not proved scalable; state the **verification rule** for the training signal — "anything
checkable by equality should never be checked by an LLM", applied to the reward; for generalisation,
**keep the interface and rewrite the data**, then **hold one world out** and show uplift on it (the
library + tech-support + gym → held-out telecom result), and say plainly that training on the eval tasks
is not evidence of anything; apply the **sim-to-real rule** — the harness you deploy in must be the
harness you train in, so the harness is part of the artefact, not plumbing; address **credit assignment**
honestly as unsolved, and say what you would do instead (read trajectories, dense proxies); and close the
loop with the **data flywheel** — log prompts plus *criteria* (not reasoning traces), prefer code-based
criteria like a commit hash and accepted diff, train on failures at a cadence, and keep a **held-out
slice** because everything harvested becomes training data.

**Red flags:** building a separate training environment; process rewards as a casual default; training on
the eval tasks; harvesting reasoning traces; no held-out world; no acknowledgement that credit
assignment is unsolved.

---

## Take-home / case-study prompt (1 full brief)

> **Brief.** A logistics company wants to deploy an agent that handles delivery-exception tickets
> autonomously: it reads the ticket, queries the order system, decides whether to refund, reship or
> escalate, calls the relevant tool, and emails the customer. A vendor has supplied an agent scoring
> **84%** on a public agentic benchmark. The company's stated intent is to run it with **no human in
> the loop**, handling roughly 4,000 tickets a day.
>
> **Deliverables (≤6 pages):**
> 1. **Why the 84% is not a release decision.** Name what the benchmark does not tell you, and be
>    specific about which of the four reliability dimensions are unmeasured.
> 2. **The environment.** What you build, what state it seeds, which tools it exposes, and what the
>    agent must not be able to see.
> 3. **The task suite.** How you enumerate requirements per task, and the per-requirement verification
>    mode (hard verifier, soft verifier, rubric) with a justification for each.
> 4. **Reward-hacking defence.** Apply the specification-bug framing: what could the agent optimise
>    instead of the task, and what design rule prevents each.
> 5. **The reliability overlay.** Which of the 12 metrics you measure, how, and which ones you would
>    deliberately *not* aggregate — with the reason.
> 6. **The release gate.** Per-dimension thresholds, why safety is excluded from any average, and how
>    many rollouts each number rests on.
> 7. **Reporting.** What goes in the result, including harness, seeds, quantization, cost and standard
>    error — and how a third party would reproduce it.
> 8. **The recommendation.** Ship, ship with human review on a subset, or do not ship — and state the
>    condition that would change your answer.
>
> **What is being assessed:** whether you separate capability from reliability; whether you reach for
> world-state verification before judges; whether you see reward hacking as a specification problem;
> whether your gate is a decision with thresholds and a reason rather than a restatement of the score;
> and whether you can say "not yet" with a condition attached.

---

## Scoring rubric — what separates a hire from a no-hire

| Dimension | No-hire | Senior hire | Staff hire |
|---|---|---|---|
| **Capability vs reliability** | Treats a benchmark score as a release decision | Knows the two axes exist, asks for reliability numbers | Names the four dimensions, the 12 metrics, and the calibration/discrimination split unprompted |
| **Grading design** | Grades the transcript | Reaches for world-state assertions | Applies hard → soft → rubric in cost order and enumerates task requirements against grader coverage |
| **Reward hacking** | Blames the model | Knows to hide the grader | Frames it as a specification bug, adds fingerprints and tool-signature variation, and audits intent vs measurement |
| **Environment thinking** | Talks about models | Builds a sandbox | Thinks environment → tasks → grading → harness, and names the sim-to-real gap and its mitigation |
| **Reporting discipline** | Reports a score | Adds the harness | Reports score ± spread, harness, seeds, quantization, cost, per-instance results and standard error |
| **Statistics** | One run | Knows pass@k | Distinguishes pass@k from pass^k, refuses the naive estimator, and knows the time budget is a hyperparameter |
| **Economics** | No cost dimension | Knows it is expensive | Does the arithmetic, separates agent and grader cost, and uses decimal arithmetic |
| **Scoping** | One bar for everything | Knows thresholds vary | Distinguishes augmentation from automation and picks metrics per task class, including when reliability metrics are wrong |
| **Meta-awareness** | Trusts benchmarks | Reads the fine print | Knows score fragmentation, omitted tasks, benchmark mixing, and holds both saturation findings — public/private does not predict *which* benchmarks saturate, but *every* benchmark is perishable |
| **Benchmark construction** | Only consumes benchmarks | Knows a good benchmark is hard | States the three rules — correlate with real work, launch near 0–1%, make it deterministically verifiable — and can place a benchmark on the difficulty curve |
| **Humility** | Claims the field is solved | Names one open problem | Distinguishes tractable from genuinely open, and says "unsolved" about multi-agent evaluation and credit assignment |
| **Training-side literacy** | Has never thought about how the agent was trained | Knows RL is involved | Says evals and environments are the same object, names credit assignment as unsolved, rejects process rewards on cost, and applies sim-to-real to the harness |

**The single strongest signal in this domain:** the candidate says a score is not a property of a model
but of a **model-plus-harness-plus-environment-plus-budget** tuple — and can then name what a
leaderboard row is missing. The second strongest is the follow-on: that the same tuple is also the
*training* artefact, which is why the harness you deploy in must be the harness you train in.
