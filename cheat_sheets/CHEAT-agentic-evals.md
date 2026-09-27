# Cheat Sheet · Agentic & Trajectory Evaluation

> **Covers:** CS-17 (agentic evaluations workshop), CS-18 (RL for agents — credit assignment,
> environments-as-datasets, benchmark construction, the data flywheel), CS-19 (fine-tuning a coding
> agent — SFT masking, the emulation ceiling, agent-operated training) ·
> **Code ground truth:** `CODE-05` (frontier-evals — execution grading in containers), `CODE-01`
> (`PATTERNS.md` — trajectory and world-state patterns), `CODE-07` (search_evals — hosted agents with
> cost accounting) · **Use when:** the thing you are evaluating takes actions, calls tools, or runs
> for minutes to hours — or when you are about to train one.

---

## The 60-second version

Agents break the assumption that an eval reads a string. An agent produces a **trajectory** — a
sequence of decisions, tool calls and observations — and its real output is often not text but a
**world state**: a database row, a file, a passing test suite.

Three ideas carry the whole field:

1. **Grade the world state, not the transcript.** Where the outcome is checkable, assert on it. This
   is deterministic, cheap, and ungameable.
2. **Isolate the grader.** The thing being graded must not be able to touch its own grader. The
   strongest available design puts each stage in a **fresh container**.
3. **Report the harness with the score.** The same model under different scaffolding can *flip the
   ranking*. A leaderboard that names only the model is reporting half a result.

And one economic fact: agent evals are the expensive end of the discipline. The reference
implementation of a frontier agentic eval runs for up to **36 hours** in a GPU-backed container.

---

## Core concepts

| Concept | Meaning | Why it matters |
|---|---|---|
| **Trajectory** | The full sequence: reasoning, tool selection, arguments, observations, recovery | You can succeed by luck and fail correctly |
| **Outcome / world-state grading** | Assert on the environment *after* the run | The escape hatch from unjudgeable prose |
| **Verifiable reward** | A deterministic, cheap, ungameable check | "Verifiable beats judgeable" |
| **Reward hacking** | The system games the proxy instead of doing the task | Any trainable reward will be exploited |
| **Container isolation** | Fresh environment per stage | The strongest anti-hacking defence without a separate grader model |
| **Harness confound** | Same model + different scaffolding → different ranking | Report the scaffold in the same column as the model |
| **pass@k** | Capability: one success in k is enough | Best-of-k, anything with a verifier |
| **pass^k** | Reliability: every one of k trials must succeed | Customer-facing policy adherence |
| **Step efficiency** | optimal steps / actual steps | A correct answer in 40 steps where 5 suffice is a failure |
| **Rubric tree** | Rubric decomposed into graded nodes | Partial credit becomes structural, not a judge's gestalt |
| **Dummy / human solver** | A solver that always fails, and a human baseline | Brackets the score; detects a drifted floor or ceiling |
| **Agent vs grader split** | Separate cost lines for the system under test and the judge | You cannot optimise what you have not separated |
| **Capability–reliability gap** | Capability improves fast; reliability improves slowly | The reason benchmark scores and real-world impact disagree |
| **Calibration vs discrimination** | Is the confidence *true* vs does it *separate* wins from losses | An agent that always says "50%" is perfectly calibrated and useless |
| **Hard vs soft verifier** | Equality/ordering/arithmetic vs an LLM reading natural language | Anything checkable by equality must never call an LLM |
| **Augmentation vs automation** | Is a human still in the loop? | Sets how high the reliability bar must be |
| **Credit assignment** | Which step of a long trajectory produced the outcome? | **Unsolved.** With hundreds of tool calls, "it's not clear at what step the model made the key insight or the errors" |
| **Process vs outcome reward** | Label every step vs score the episode | Per-step labelling "hasn't proved scalable" at agent scale — dense proxies are the practical middle |
| **Environment = tasks + harness + metrics** | tasks → the dataset; harness → the agent; metrics → the reward functions | Evals and environments are **the same object**, not analogues. One artefact serves RL, evals, synthetic data, prompt optimisation and ablation |
| **Sim-to-real (training rule)** | Train in the harness you deploy in | The harness is **part of the model artefact** — training in one and deploying in another manufactures a sim-to-real gap for free |
| **Data flywheel** | Log prompts + **criteria**, not reasoning traces | Invert "where does your data come from?" into "where is your system running?" |
| **Budget as a capability axis** | A fixed spend the agent must earn capability within | Cheaper models can **outrank** frontier ones by getting more iterations before the budget runs out |
| **Emulation ceiling** | SFT cannot exceed the quality of the traces it imitates | SFT narrows a model's behaviour; it does not lift it past its teacher |
| **Held-out eval loss** | The model-selection proxy in an SFT sweep | Selects the best **imitator**, not the best **agent** — you must eval the agent separately |
| **Masking (`-100`)** | Prompt-token labels set to `-100` so they contribute no loss | The one mechanism that makes SFT *not* continued pre-training |

---

## Formulas & metrics

**Pass-rate (the two questions people conflate)**

```
pass@k = 1 - C(n-c, k) / C(n, k)      n samples drawn, c correct
        compute as a RUNNING PRODUCT — never form the giant binomials
pass^k = estimated (per-trial success)^k

p = 0.75  ->   pass@10 ~ 1.0    vs    pass^10 ~ 0.056
```
Naive `1 - (1-p)^k` is **biased high for small n**. Never report it.

**Trajectory metrics**

```
tool-selection accuracy = correct tool chosen / tool calls made
argument accuracy       = tool calls with valid AND correct args / tool calls
step efficiency         = optimal_steps / actual_steps
recovery rate           = (failures followed by a successful retry) / total failures
loop detection          = repeated (state, action) pairs / total steps
```

**Execution grading (three stages, the reusable pattern)**

```
Stage 1  rollout        agent works in container A -> produces an artefact
Stage 2  reproduction   artefact EXECUTED in a fresh container B (with GPU) -> "executed submission"
Stage 3  grading        judge scores the executed submission in a third container C against a rubric
```
"Does it run" is a **deterministic gate ahead of** the judged rubric. A plausible submission that
does not execute scores zero.

**Cost**

```
task_cost = agent_cost + grader_cost          # both, always
cost_per_query = (in_tok/1e6 * in_rate) + (out_tok/1e6 * out_rate)
```
Use exact decimal arithmetic — floating-point drift compounds across a long run. Track cost **per
component** so you can attribute it.

**Report format**

```
score ± spread over N runs, with the harness named
```
The reference leaderboard reports **3 runs** with ± on every entry, and names the agent scaffolding
in the same column as the model.

---

## The numbers worth knowing

| Item | Value |
|---|---|
| Top PaperBench score | **26.0 ± 0.3** — IterativeAgent o1-high, **36 h** limit, 3 runs |
| Same agent, 24 h limit | 24.4 ± 0.7 — **the time budget is a hyperparameter**: 12 h cost 1.6 points |
| PaperBench **Code-Dev** variant | **43.4 ± 0.8** — writing the code is far easier than making it reproduce the result |
| Best BasicAgent | claude-3.5-sonnet, **21.0 ± 0.8** |
| The harness flip | BasicAgent o1-high **13.2** < claude-3.5-sonnet **21.0** — but IterativeAgent o1-high **26.0** > IterativeAgent claude-3.5-sonnet **16.1** |
| Runners-up | BasicAgent gpt-4o 4.1; gemini-2.0-flash 3.2; o3-mini-high 2.6; deepseek-r1 6.0 |
| p = 0.75 | pass@10 ≈ **1.0** vs pass^10 ≈ **0.056** |
| Non-retryable HTTP statuses | `{400, 401, 403, 404, 422}` — retrying these wastes budget |
| Benchmark half-life | **1–2 years**, sometimes **2 months** — "benchmarks are a perishable asset" |
| Launch difficulty | **0–1%**, never 40% — 40% targets a capability the labs already have |
| Calibration window | not 75% in 2 months (too easy), not 0% for 5 years (too hard or miswired) |
| SWE-bench trajectory | **1.5%** at launch → **93.9** on SWE-bench Verified |
| RL rollout length | **minutes → hours** when the unit of training moved from model to agent |
| AlgoTune budget | **$1** per task, ~150 real programs, avg speedup subject to a correctness check |
| Code Clash | **7 arenas**, **15 rounds**, ranked by ELO |
| RL base model scale | Qwen **30B** MoE (RL) · **7–8B** (Dr. Tulu) · **32B** (SWE-smith runs) |
| Context windows cited | 4K → 16K → 128K → ~**1–2M** |

**The Code-Dev gap is the most informative number in the table.** 43.4 versus 26.0 for the *same
agent* isolates two different capabilities: writing code that looks right, and writing code that
actually reproduces a result. If your eval only has one number, you cannot tell which one you are
measuring.

---

## The reliability axis — the second thing to measure

Capability answers *can it*. Reliability answers *does it, every time, under pressure*. Over **18
months across 14 frontier models** on GAIA and Tau-bench airline, accuracy rose steeply while a
composite reliability score rose only gradually — plotted against each other, a **remarkably straight,
shallow line** `[28:57]`. Capability does not fully measure usefulness.

**Four dimensions → 12 metrics** `[26:45]` `[35:57]`:

| Dimension | Sub-metrics | The trap |
|---|---|---|
| **Consistency** | Outcome; trajectory; cost/resource stability | 70% accuracy may mean 70% of tasks always work, or any task works 70% of the time — different products |
| **Robustness** | Fault (inject API errors); prompt (LLM reword, same semantics) | Style changes the answer when semantics did not |
| **Predictability** | Calibration **and** discrimination | They move in **opposite** directions: calibration improving, discrimination worsening `[31:30]` |
| **Safety** | Failure **severity** (formatting error vs data deletion) | **Measured but deliberately NOT aggregated** `[32:00]` |

Reliability is an **overlay**, not a separate suite: "you can take any benchmark and measure all of
our reliability metrics on it" `[34:32]`. But it is **not always desirable** — a poetry agent that
returns the same poem every time is broken, not consistent `[38:49]`. Trajectory consistency is a
virtue where a procedure is canonical and a defect where variation is the point.

**Order of work** `[1:12:34]`: **environment → tasks → grading → harness** — then optimise (GRPO) in
the same environment. The sandbox is the reusable asset; a score is just a snapshot of it.

**Environment design rules** `[1:05:41]` `[1:07:32]`: the agent must not see the grader or the
solution; **everything the task asked for must be graded** (ask for three, grade two, and the agent
learns to skip the third); fingerprint unit tests; vary tool **names and signatures** between runs.
The first two are the same insight: **reward hacking is usually a specification bug, not a model
bug.**

**Never evaluate through an inference provider** `[1:27:15]`: "you are evaluating the provider, and
you're not evaluating the model." No control over how it was called or prompted → not reproducible.

---

## The training side — why your eval numbers don't transfer

Everything above is the eval view. The training view is the same object, and the gap between them is
where most agent programmes fail.

**The claim to hold: evals and environments are the same thing.** An environment is **tasks + harnesses
+ metrics**; the **tasks play the dataset role**, the **harness plays the agent role**, and the
**metrics are the reward functions**. One object then serves five consumers — **RL, evals, synthetic
data, prompt optimisation and model ablation.** If environments are filed under "RL", a team builds
none until it does RL, while already running evals, tuning prompts and comparing models — and
maintains five parallel versions of the same thing.

**What changed when the unit of training moved from model to agent:**

```
OLD  prompt -> answer -> STATELESS verifier -> binary reward        . minutes
NEW  task -> T env steps -> episode ends -> TRAJECTORY verifier     . HOURS
       rewards = heuristics + rubrics + LLM dense feedback
```

Three consequences, and only one of them is solved:
- The verifier is no longer stateless — it grades a trajectory.
- **Credit assignment is unclear** — with hundreds of tool calls, which step mattered? **Unsolved.**
- **Process rewards are not the escape hatch** — labelling every step of every trajectory "hasn't
  proved scalable." The practical answer is a trajectory-level reward plus dense heuristic/rubric
  proxies.

**The four rules that follow:**

1. **Train in the harness you deploy in.** The harness is part of the model artefact, not plumbing.
2. **Keep the interface, rewrite the data, hold one world out.** Sibling worlds from one engine are
   how you prove generalisation — train on library + tech-support + gym, and look for **uplift on a
   held-out telecom world**. Training on the tasks you evaluate on "is like cheating if the goal is to
   climb the benchmark."
3. **Capture criteria, not traces.** The flywheel inverts "where does your data come from?" into
   "**where is your system running?**" Log prompts plus a world-state snapshot and the **criteria** —
   "RL doesn't want to train on tokens directly; it wants example settings and criteria." The cleanest
   criterion is code-based: **commit hash + accepted diff**, replayed against hidden tests. Then train
   on the failures at a cadence = **proto continual learning**.
4. **What teams actually run is SFT distillation, not RL.** RL is the frontier and the research; SFT is
   the enterprise default. The labs' durable moat is **reward-modelling maturity**, not environments.

**SFT mechanics worth knowing (CS-19):** mask the prompt tokens with **`-100`** so no loss is computed
on them — that is the whole difference between SFT and continued pre-training. `apply_chat_template`
is the hook where the harness's tool-call format is re-expressed in training data. **Held-out eval loss
selects the best imitator, not the best agent** — so a clean sweep of the loss curves proves the model
learned the *format*, and tells you nothing about whether it *works*. The **emulation ceiling** caps
you at the teacher's quality.

**The eval-set rule after fine-tuning on your own traces:** your traces will **not** line up with a
public benchmark, so bring a **mixture** — a small benchmark for your own use case *plus* a general
coding/terminal benchmark, to confirm you have not **compromised general ability**. Expect
off-distribution degradation and decide in advance whether it is acceptable.

**Benchmark construction, if you have to build one (three rules):** correlate with **real-world
usefulness** (no IQ tests); launch at **0–1%**, not 40%; make the answer **deterministically
verifiable** — "don't be lazy, just think about it more until you figure out how to have a
deterministic verifier." The difficulty curve to place it on: school exams → college exams → human
evals → days-of-human-work → **verifiable tasks nobody has ever done** (a C compiler in Rust, the
Linux kernel in Go).

**When a judge is acceptable at all:** a **binary** question where the vast majority of domain-aware
humans would agree on the answer — "pretty close to verifiable" — aggregated across **many** binary
criteria rather than asked once, holistically. Fuzzy quality is not a judge problem; it is an
unmeasured requirement.

**Training-side mistakes** (the mirror of the Top 10):

1. Training in one harness and deploying in another, then being surprised by the gap.
2. Training on the tasks you evaluate on, then calling the gain generalisation.
3. Harvesting **reasoning traces** instead of prompts-plus-criteria.
4. Using **held-out loss** as evidence the agent got better.
5. Expecting **process rewards** to solve credit assignment at agent scale.
6. Assuming RL is what most teams deploy.
7. Claiming a synthetic-data gain with no **out-of-sample** check.
8. Treating a benchmark as a durable asset rather than a perishable one.

---

## Decision rules

1. **If the outcome is checkable, assert on the world state.** Database, filesystem, passing tests.
   Do not grade the transcript.
2. **Run the artefact before grading it.** A deterministic "does it run" gate in front of the judged
   rubric removes an entire class of false positives.
3. **Put each stage in a fresh container.** The grader must not share filesystem, network or state
   with the system under test.
4. **Grade *how* as well as *what*.** A lucky success and a well-executed failure are different
   results; trajectory metrics separate them.
5. **Use pass@k for capability, pass^k for reliability** — and never confuse them in a report.
6. **Always report the harness alongside the model.** Rankings are not harness-invariant.
7. **Bracket the score with a dummy and a human baseline.** A dummy that always fails and a human
   ceiling detect floor/ceiling drift in your own eval.
8. **Decompose rubrics into trees** so partial credit is structural rather than impressionistic.
9. **Separate agent cost from grader cost**, in exact arithmetic, per run.
10. **Test the harness with a dummy before committing to a long rollout.** The cheapest honest path
    through a 36-hour eval is the solver that spends nothing.
11. **Treat the time limit as a hyperparameter** and report it. Two runs of the same agent at
    different budgets are different experiments.
12. **Expect high variance and report ±.** Long-horizon agents are noisy; a single run is not a
    result.
13. **Hold out a private set.** Any eval used as a training signal stops being a valid measurement.
14. **Do not run execution grading in CI.** Hours of wall-clock and GPU containers belong on a
    nightly or pre-release cadence.
15. **Measure calibration AND discrimination.** Reporting one without the other is how you ship a
    confidently uninformative agent.
16. **Set a reliability threshold per dimension**, never on a composite. A composite index tracks the
    field; it is a poor release gate because it hides which dimension failed.
17. **Keep safety out of every average** — a data-deletion event fails the release regardless of the
    other numbers.
18. **Reimplement only the real service your tasks actually touch**, with the state machine intact.
    Partial fidelity by design beats a full clone you cannot afford to maintain.

**And the training-side rules, if you are also the one training it:**

19. **Train in the harness you deploy in.** A harness change can invert a ranking, so it is part of
    the artefact, not configuration.
20. **Build the environment once and consume it everywhere** — the same object is your eval, your
    training ground, your synthetic-data generator and your ablation harness.
21. **Keep the interface, rewrite the data, hold one world out** — and report the uplift on the
    held-out world, not the training worlds.
22. **Prefer verifiable rewards to judged ones**; if you must judge, use a binary criterion whose
    answer domain experts would agree on, and aggregate many of them.
23. **Harvest prompts and criteria, never reasoning traces** — and hold out a slice, because everything
    you harvest is training data from that moment on.
24. **Read the trajectories, not the score.** Reward hacking lives in the trajectory, and so does the
    insight about which step actually mattered.

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Runs reported per leaderboard entry | **3**, with ± |
| Longest observed rollout budget | **36 hours** (GPU container) |
| CI suitability | **None** — execution grading is nightly/pre-release |
| Isolation stages | **3** — rollout, reproduction, grading, each fresh |
| Solver bracketing | always include a `dummy` (floor) and `human` (ceiling) |
| Pass-rate estimator | unbiased combinatorial, **never** `1-(1-p)^k` |
| Retry policy | do **not** retry `{400, 401, 403, 404, 422}` |
| Cost precision | **decimal**, not float |
| Reliability overlay | run the task set **n times**; no reliability-specific benchmark needed |
| Severity scale | keep **out** of the composite index |
| GAIA 2 capability floor | **time-triggered** tasks ≈ **0%** for *all* models incl. top ones `[52:07]` |
| GAIA 2 scale | 1,000 scenarios · 10 universes · ~11 apps |

---

## Tool commands

```bash
# --- frontier-evals (CODE-05): start with the dummy solver ---
cd project/paperbench
uv sync
git clone <repo> --filter=blob:none && cd <repo>
git lfs fetch --include "project/paperbench/data/**" --exclude ""
git lfs checkout project/paperbench/data
export PAPERBENCH_DATA_DIR="$(pwd)/project/paperbench/data"

cp .env.example .env      # GRADER_OPENAI_API_KEY defaults to OPENAI_API_KEY

uv run python -m paperbench.reproduce --help   # stage 2 — execute the artefact
uv run python -m paperbench.grade --help       # stage 3 — rubric grading
#   read solvers/dummy FIRST: it exercises all three stages with NO model spend
```

```python
# --- the shape of a hosted-agent eval (CODE-07) ---
# BaseGrader / BaseSuite; GraderResult carries grade_text AND cost
# GraderError            -> retryable
# NonRetryableGraderError-> do not retry (permanent HTTP statuses)
# GraderPreflightError   -> fail fast before spending
# judge emits STRICT JSON: {extracted_final_answer, reasoning, correct: yes|no, confidence}
# failed-as-zero vs failed-excluded is a DELIBERATE reporting choice — state which you use
```

**Registry note:** in every mature runner here, a suite name resolves to a **loader + grader pair**
through a registry. That is what makes adding a task a data change rather than a core change.

---

## Top 10 mistakes

1. **Grading the transcript** when the world state was checkable.
2. **Running the grader in the same container** as the agent — the system can influence its own score.
3. **Reporting a leaderboard that names only the model**, hiding the harness confound.
4. **Confusing pass@k with pass^k** — they tell opposite stories at p = 0.75.
5. **Using the naive `1-(1-p)^k`** estimator.
6. **Reporting a single run** of a long-horizon agent with no ± and no harness.
7. **Ignoring step efficiency** — counting a 40-step success as equal to a 5-step one.
8. **Retrying permanent HTTP failures**, burning budget on calls that can never succeed.
9. **Floating-point cost accounting** across a long multi-step run.
10. **No dummy or human baseline**, so a drifted floor or ceiling goes unnoticed.

---

## If you only remember three things

1. **Grade the world state, not the transcript — and run the artefact before grading it.** A
   deterministic "does it run" gate in front of a judged rubric removes a whole class of false
   positives.
2. **Isolate the grader and report the harness.** Fresh container per stage; the score is meaningless
   without the scaffolding that produced it.
3. **pass@k is capability, pass^k is reliability** — p = 0.75 gives pass@10 ≈ 1.0 against pass^10 ≈
   0.056, and long-horizon agents need ± over multiple runs because a single run is not a result.

**And one for the training side:** the environment is not only where you evaluate — it is also where
you train, **which is why the harness you deploy in must be the harness you train in.** Everything
else about agent training follows from taking that sentence seriously.
