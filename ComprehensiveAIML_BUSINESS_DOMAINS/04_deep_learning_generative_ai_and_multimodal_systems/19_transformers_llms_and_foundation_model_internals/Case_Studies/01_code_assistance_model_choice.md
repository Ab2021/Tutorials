---
case_id: "CASE-19-01"
case_title: "Choosing a code-assistance model with authorised repository context"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Developer tooling"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["authorization-scope", "cost-latency-tradeoff"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Developer productivity owner"
source_index: "INDEX.md"
review_trigger: "Model, repository permission, benchmark, toolchain, or cost contract change"
---

# Choosing a Code-Assistance Model Under Authorised Repository Context

**Evidence disclaimer.** Nothing in this manuscript describes a system that was built, a model that was
served, or a developer outcome that was measured. The decision structure, the data contract, and the
evaluation design are the deliverable; every token price, acceptance rate, minute estimate, and
retrieval hit rate below is a constructed number marked `ILLUSTRATIVE`. Arithmetic on those inputs is
recomputed by hand and shown term by term, so a reader can substitute their own inputs and rebuild the
conclusion. Where a real mechanism is asserted (attention cost growth, permission revocation, test
oracles), the assertion rests on documented model behaviour or on the repository's own contract, not on
an observation of this design. Where evidence is absent the text says `not-measured` rather than
implying a pilot happened.

## 1. The One-Line Framing

Given an authorised repository task pinned to one commit, choose among four configurations — manual
only, model without retrieval, model with authorised pinned retrieval, and retrieval plus a sandboxed
test loop — or abstain, where the deciding quantity is minutes saved per task after rejected attempts
are paid for, not benchmark accuracy, and where no configuration may read a path the requester is not
entitled to see at the pinned revision.

## 2. Context & Background

A code-assistance request looks small and is not. The developer has an issue, a repository at some
commit, a set of files they are allowed to read, and an expectation about what "fixed" means. A
language model can produce a diff in seconds, but the diff is only useful if it targets the same
revision the reviewer will merge into, references dependencies that actually exist there, and survives
the tests that encode the intended behaviour. Each of those is a separate contract, and each can fail
independently of the model's ability to write plausible code.

The reason this case exists is that the usual proposal — pick the model with the best public benchmark
score — answers a different question. Public code benchmarks score a single function body against a
test harness, with no repository, no permission boundary, no reviewer, and no retry budget. The
workflow being decided here has all four. Between the benchmark and the workflow sits a gap wide enough
to reverse a ranking, and this manuscript spends most of its length on that gap.

**The workflow under decision.** A developer opens a task. The tool resolves the pinned commit hash and
asks a permission service which repository paths the requester may read for this purpose. Retrieval
selects candidate files, packs them into a token budget, and the model emits a unified diff plus a short
rationale. Under the strongest configuration the diff is applied inside a sandbox with the declared
test command, and failing output is fed back for a bounded number of retries. A maintainer reviews
whatever survives and merges or rejects it. Nothing in this chain is allowed to skip the maintainer.

**Why the manual baseline is not a straw man.** Manual implementation already has an access boundary,
already has a reviewer, and already produces a mergeable diff with a known cost of roughly one working
hour per small task. The assistant only earns its place by reducing the expected minutes per accepted
task below that baseline, which means the rejected attempts must be cheap to detect and cheap to
discard. An assistant with a high hit rate and an expensive rejection path can lose to no assistant at
all, and the arithmetic in section 6 shows exactly how steep that trade is.

**The four configurations.** They differ in what the model is allowed to see and what it is allowed to
do, and those differences are what the numbers track. The tokens column is the mean input size per
attempt under each configuration, `ILLUSTRATIVE`.

| Config | Retrieval | Sandbox tools | Mean input tokens/attempt | Mean attempts per task | First-attempt acceptance |
|---|---|---|---:|---:|---:|
| C0 | none (human only) | none | 0 | 1.0 | 1.000 by definition |
| C1 | developer pastes 3 files | none | 9,400 | 2.9 | 0.31 |
| C2 | authorised pinned retrieval, k=12 | none | 26,178 | 2.4 | 0.42 |
| C3 | authorised pinned retrieval, k=12 | test command, 3 retries | 31,600 | 2.0 | 0.55 |

C1 is in the comparison deliberately. It is the configuration most teams actually start with, because
it needs no retrieval service and no permission integration: the developer selects the files. It has
the smallest token bill and the worst acceptance rate, because file selection is the part of the task
humans are worst at when they are already tired enough to want help.

**What the repository contributes to difficulty.** Small bug-fix tasks in a service module are not
self-contained. A patch that changes the signature of one function has to update its call sites, its
type declarations, and sometimes a serialisation schema. The number of files a correct patch must
respect has a long tail: most tasks touch three, some touch six, a few touch twelve. A retrieval
strategy that returns twelve files therefore has two separate failure surfaces — missing a required
file, and drowning the required file in twelve plausible neighbours.

**Why a token-price comparison misleads.** Model spend for this workload is on the order of cents per
attempt. Reviewer attention is on the order of ten dollars per attempt. Any decision that ranks
configurations by token spend is optimising the small term. The case therefore computes three
different rankings — cost per accepted patch, minutes per task, and first-attempt acceptance — and
shows they disagree, which is the analytical core of the finding.

**The abstain path is an option, not a formality.** Some tasks cannot be served: the requester is not
entitled to the file that must change, the task has no reproducible test so "correct" is undefined, or
the change touches a path under a release freeze. In those cases the only compliant action is to refuse
the assist and route the task to normal development. A decision framework that has no refusal branch
will eventually serve a request it should have declined, and that is a policy failure rather than a
quality failure.

**Where the model internals matter.** Three properties of transformer inference show up directly in
this decision. Context is a hard budget: the window is finite, the system prompt and tool schemas
consume part of it before any code is packed, and truncation silently removes the file that mattered.
Prefill cost scales with input tokens while decode cost scales with output tokens, so a retrieval-heavy
configuration pays latency at the start of every attempt. And attention cost grows superlinearly with
sequence length, so doubling packed context is not free even when the window allows it. Those three
facts are why section 6 starts with a budget derivation instead of a benchmark table.

## 3. The Business Question

**Decision statement.** For the task family "small bug-fix patches in the payments-service module", at
a granularity of one task at one pinned commit, which configuration should be adopted for a six-week
pilot: C0 manual, C1 paste-only, C2 authorised retrieval, C3 retrieval plus sandbox test loop, or defer
the whole programme and keep the current manual process.

**Decision owner.** The developer-productivity owner holds the decision. Repository owners hold the
permission boundary and can veto any configuration that widens it. Maintainers hold merge authority and
supply the acceptance label. The productivity owner may not overrule a maintainer rejection, and may
not grant repository access that the permission service denies.

**Decision grain.** One row is one task at one commit: task text, commit hash, requester identity,
authorised path set, configuration, attempts consumed, maintainer verdict, minutes recorded. The
decision is made once per configuration for the pilot, not once per task; per-task rows exist only to
produce the aggregate the decision reads.

**Cut-off timestamp.** `2026-10-31T00:00:00Z`. Tasks created after that instant are out of the
evaluation population, and any file content committed after it is invisible to the pilot. The cut-off
exists so that a model trained later cannot have seen the answer.

**Alternatives on the table.** Five, and each is a real option with a different cost shape. C0 costs
46 minutes per task and nothing else. C1 costs a token bill near zero and a small integration effort.
C2 costs a retrieval service, a permission integration, and a token bill. C3 costs all of C2 plus a
sandbox runner and its operational surface. Deferral costs nothing this quarter and forgoes any saving.
The framework must be able to select deferral, which happens whenever the best configuration's expected
saving falls inside the uncertainty interval computed in section 9.

**Hard constraints.** Least-privilege retrieval evaluated per request against requester, purpose, and
path. Pinned source revision carried through retrieval, prompt, and diff application. Sandbox-only tool
execution with no network egress and no write access outside a scratch copy. Explicit test contract: a
diff is not "passing" unless the declared command ran and its output is attached. Human merge approval
with no auto-merge path. Bounded budget: token spend per accepted patch under ten cents and p95
generation latency under ninety seconds.

**Out of scope.** Fine-tuning or continued pretraining on the repository, IDE interaction design,
licence scanning of generated code, and any change to the review policy itself. Those are separate
decisions with separate owners and are listed in section 14 as handoffs rather than quietly assumed.

## 4. Assumptions & Constraints

Each assumption carries the check that would falsify it. An assumption with no check is a wish, and
this table is the list of things the decision is allowed to depend on.

| ID | Assumption | Falsifying check | If false |
|---|---|---|---|
| A-01 | The task names one commit and the repository has that commit. | hash resolves; tree hash matches index | abstain, no assist |
| A-02 | Every retrieved path is authorised for this requester and purpose. | permission decision logged per path | drop path, re-pack |
| A-03 | Required-file count is at most twelve for 90% of tasks. | import-graph closure audit over 60 tasks | raise k or refuse complex tasks |
| A-04 | A reproducible failing test exists for the task. | test command exits non-zero before the patch | no test loop; fall back to C2 |
| A-05 | Maintainer acceptance is stable across reviewers. | two-reviewer agreement on 40 sampled tasks | widen label or add adjudication |
| A-06 | Manual baseline is 46 minutes median for this task family. | timestamps on 40 manual tasks | recompute break-even |
| A-07 | Token prices hold for the pilot's six weeks. | contract price at invoice time | recompute cost ranking |
| A-08 | Repository code is outside the model's training corpus cut-off. | probe novel internal APIs; ask for a name it cannot know | benchmark scores are inflated |
| A-09 | Sandbox test runtime is under 90 seconds per run. | timed runs on the declared command | cap retries at one |
| A-10 | Reviewers read the diff rather than the rationale. | audit 20 reviews for rationale-first approval | label is contaminated |
| A-11 | Retry attempts are independent enough to multiply. | compare attempt 1 and 2 acceptance on same task | treat attempts as correlated |
| A-12 | Task families are exchangeable across weeks. | per-week acceptance spread | cluster the interval |

| ID | Constraint | Enforcement point | Violation response |
|---|---|---|---|
| C-01 | No merge without a human maintainer action. | branch protection rule | pipeline fails closed |
| C-02 | No path outside the authorised set enters the prompt. | retrieval filter before packing | deny the request |
| C-03 | No network egress from the sandbox. | runner network policy | kill the run, flag it |
| C-04 | No write outside the scratch copy. | mount policy, read-only source | kill the run |
| C-05 | Diff must be applied to the pinned commit, not to HEAD. | apply step asserts hash | reject the attempt |
| C-06 | Token spend per accepted patch at most $0.10. | per-request metering | throttle to C1 |
| C-07 | p95 generation latency at most 90 seconds. | client-side timer | fall back to manual |
| C-08 | Revert rate within 14 days at most 3%. | release tracker join | suspend the config |
| C-09 | Cache lifetime must not exceed revocation latency. | cache key includes permission epoch | invalidate, re-retrieve |
| C-10 | No secret value may appear in a diff or a rationale. | secret scanner on output | block, notify owner |

Two constraints deserve a note because they are usually written as aspirations. C-05 exists because a
model asked to patch a file it saw at commit A will happily produce a diff that applies only to commit
B if the packing step drifted; asserting the hash at apply time converts a silent wrong-revision merge
into a failed attempt, which is the cheap outcome. C-09 exists because retrieval caches are an
authorisation bypass if they outlive a permission change; binding the cache key to a permission epoch
makes revocation take effect at the next request rather than at cache expiry.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per task attempt; one task may produce up to three attempt rows plus one verdict row.
The decision reads aggregates over task rows, so attempt rows must never be counted as tasks.

**Keys.** `task_id` (stable issue-derived string), `commit_hash` (40 hex characters), `attempt_index`
(1..3), `config_id` (C0..C3), `requester_id`, `purpose_code`. The pair `(task_id, commit_hash)` is the
unit of correctness; `(task_id, config_id, attempt_index)` is the unit of cost.

**Population.** 400 tasks drawn from eight weeks of the issue tracker, filtered to: label indicates a
bug fix, the described change touches at most two source files, a failing test exists in the declared
test manifest, and the task was created before the cut-off. Tasks failing the filter are counted in an
exclusion table, not deleted silently.

**Cut-off and windows.** Task creation cut-off `2026-10-31T00:00:00Z`. Evaluation window
`2026-09-01` through `2026-10-31` for task selection; a 14-day revert window opens at each merge and
may extend past the task window, which is why outcome rows carry their own timestamps.

**Units.** Minutes as decimal minutes from developer-recorded timestamps, not rounded hours. Tokens as
tokenizer counts of the exact packed string. Money as United States dollars at the contracted rate.
Latency as milliseconds at the client for time-to-first-token and total generation.

**Labels.** Primary label is `first_pass_accepted`: a maintainer merged the first attempt with no
substantive change to the diff. Substantive means any edit to control flow, signatures, or test
expectations; comment and formatting edits do not disqualify. Secondary label is `solved_within_three`,
and a third label `reverted_14d` joins the release tracker to catch silent wrong merges.

**What is not available at decision time.** The maintainer verdict, the retry count actually consumed,
the revert outcome, and any post-merge incident are unknown when the request is served. They are
available only after the fact. Any feature computed from them is a label, never an input; a design that
feeds back the verdict into its own retrieval choice is measuring itself.

**Splits.** By repository and by time. Tasks from the last two weeks form the holdout; the earlier six
weeks supply the development sample used to tune k, packing order, and retry cap. A change family —
several tasks touching the same function — stays entirely on one side, because otherwise a tuned
configuration memorises the family rather than the task.

**Exclusions and their accounting.** Tasks without a reproducible test (estimated 18% of the filtered
set, `ILLUSTRATIVE`), tasks whose owner is also the evaluator, tasks touching a frozen path, and tasks
whose required file the requester cannot read. Each exclusion carries a reason code, and the exclusion
rate is reported next to every acceptance number so a reader can see what population remains.

### 5b Tools & Stack

The stack is described as interfaces because the decision depends on the contracts, not on vendors.

- **Permission service.** Answers `(requester, purpose, path, commit) -> allow|deny` and returns a
  permission epoch that any cache must embed. Denials are logged with the reason code; the log is the
  evidence for the zero-disclosure guardrail.
- **Retrieval service.** Ranks candidate paths for a task, expands one hop through the import graph,
  and returns for each chunk: path, commit hash, byte range, content hash, and access decision. Chunks
  without an access decision never reach the packer.
- **Packer.** Orders chunks by a deterministic policy (definition before use, failing test last but
  present, headers before bodies), counts tokens with the target tokenizer, and records what it
  dropped. The dropped list is a first-class output; a silent drop is how truncation becomes a quality
  regression.
- **Model service.** Records model identifier, tokenizer identifier, prompt template hash, decoding
  parameters, and the exact packed input hash for each attempt. Two attempts with different templates
  are different configurations and must not be pooled.
- **Sandbox runner.** Applies the diff to a scratch copy of the pinned commit, runs the declared test
  command with network disabled, and returns exit code, duration, and a bounded output excerpt.
- **Review console.** Shows the diff, the test output, and the list of files the model saw. The
  file-list display is a control, not a feature: a reviewer who cannot see what context was packed
  cannot judge whether the patch was lucky.
- **Metering.** Per-request input tokens, output tokens, cache-hit tokens, wall-clock latency, and
  retry count, written to the same store as the labels so cost and quality join on `task_id`.

### 5c Method Summary

The protocol is five steps and is deliberately boring, because the interesting risk is in the labels
rather than in the estimator.

1. Fix the task population and the exclusions before running any configuration. Changing the
   population after seeing results is the most common way an evaluation becomes a sales document.
2. Run C0 first on a 40-task subsample to calibrate the manual baseline in minutes, since every
   comparison is stated relative to it. If the baseline moves, every saving number moves with it.
3. Run C1, C2, C3 on the same 400 tasks in a paired design: same task, same commit, same reviewer
   pool, configuration assigned by a fixed rotation so no configuration sees systematically easier
   tasks. Paired data supports the discordant-pair test in section 9.
4. Score three ways: first-attempt acceptance, minutes per accepted task, and token cost per accepted
   patch. Report all three, because the rankings disagree and the disagreement is the decision.
5. Apply the decision rule: adopt the configuration with the best minutes-per-accepted-task point
   estimate whose lower confidence bound still beats the manual baseline by more than the cost of
   operating it; otherwise defer.

**Why not measure only accuracy.** Acceptance alone cannot distinguish a configuration that saves time
from one that produces more attempts at the same total cost. Ten attempts per task at 60% acceptance and
two attempts at 55% acceptance have similar task-level solve rates and completely different review
load. Minutes and attempts are therefore primary, and accuracy is the constraint that keeps them honest.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Task text comes from the issue tracker; code comes from git at an explicit hash;
permission decisions come from the permission service; verdicts come from the review console; minutes
come from the developer's own timestamps. Every row is traceable to a commit and a reviewer action.
Nothing in the evaluation is reconstructed from memory, and no number in this manuscript is presented as
measured.

**Leakage vectors, ranked by how often they ruin this kind of study.**

1. **Test-oracle leakage.** The failing test usually contains the expected value. If the test file is
   retrieved and packed, the task collapses to copying a literal. Remedy: retrieve tests only as a
   signature-level stub, or exclude test bodies and score separately on tasks whose tests are hidden.
2. **Future-commit leakage.** A retrieval index built over all branches can return the commit that
   eventually fixed the bug. Remedy: index only ancestors of the pinned commit and assert that every
   chunk's commit is an ancestor.
3. **Change-family leakage.** Sibling tasks in the same function teach the answer to each other.
   Remedy: cluster by changed symbol and split by cluster.
4. **Reviewer contamination.** A reviewer who has seen the rationale may accept a diff they would have
   rejected cold. Remedy: rationale hidden until the verdict is recorded, then audited.
5. **Benchmark contamination.** A public-suite score for a model trained after the suite was published
   measures memorisation as much as capability. Remedy: probe internal APIs that cannot be in any
   corpus, and treat a large public-to-internal gap as evidence for the gap, not for the model.
6. **Prompt-template drift.** Editing the template mid-run changes the configuration without changing
   its label. Remedy: template hash in the row key; pooled analysis refuses mixed hashes.

**Quality controls.** Duplicate tasks are removed by normalised task text plus changed-symbol match. A
10% sample of verdicts is double-reviewed to estimate label noise. Minutes include waiting time, because
a developer blocked on a slow generation is not productive elsewhere; excluding wait time would flatter
the slower configuration.

**What the evidence level permits.** `not-measured`. The design can be executed as written; the numbers
are constructed to make the arithmetic inspectable. A reader must not cite the acceptance rates as
findings about any real model.

### 5e Reproducibility & Method-Selection Ledger

Every analysis choice below was made against a rejected alternative. Recording the rejected alternative
is what makes the method reproducible, because a later reader can see which conclusions depend on which
choice.

| Choice made | Rejected alternative | Why rejected | Consequence if reversed |
|---|---|---|---|
| Minimum expected minutes per accepted task | Public benchmark score | Benchmark has no repository, reviewer, or retry budget | Ranking can invert; C1 would look best |
| First-attempt acceptance | Any-acceptance-within-three | Hides attempt inflation and review load | C1's 2.9 attempts become invisible |
| Paired design, fixed rotation | Independent arms | Task difficulty dominates variance | Needs several times the sample |
| Commit-pinned retrieval | Latest-branch retrieval | Wrong-revision merges are silent | Guardrail C-05 becomes meaningless |
| Token-exact packing with dropped-file log | Character-count approximation | 3.6 chars/token varies by content | Budget off by tens of percent on code |
| Wilson interval with cluster adjustment | Normal approximation | Small counts and clustering both bite | Interval too narrow, decision too confident |
| Manual baseline measured, not assumed | Industry-average minutes | Baseline scale sets every saving | All savings claims become unfalsifiable |
| Abstain as a first-class outcome | Best-effort always | Serves unauthorised or untestable tasks | Policy failure, not quality failure |

**Reproduction steps.** Rebuild the task table from the issue export with the stated filter; resolve
each commit; run the permission filter; pack with the recorded tokenizer; replay configurations in the
recorded order; recompute the section 6 formulas from the raw table. Two analysts following those steps
should obtain identical aggregates, because ties are broken by `task_id` sort rather than by map
iteration order.

**Version pinning.** Model identifier, tokenizer identifier, prompt hash, retrieval index hash, packing
policy version, sandbox image digest, and test-manifest hash are all recorded per attempt. A change to
any one of them opens a new configuration identity, because a weight-only rollback does not restore
behaviour when the prompt or the index also moved.

## 6. Analysis Walkthrough

Six derivations follow. Each is stated as a formula, the symbols are defined, the constraint that makes
it valid is named, the arithmetic is carried out with intermediate values, and the reading is stated.
All inputs are `ILLUSTRATIVE`.

### 6.1 Usable context budget and file packing

```text
FORMULA
  U = (1 - m) * (W - S - T - L - O)
  N_files = floor(U / (b + h))

WHERE
  W = nominal context window, tokens                  = 131,072
  S = system prompt and policy block, tokens          = 1,480
  T = task statement plus failing-test stub, tokens   = 720
  L = tool schema block for four tools, tokens        = 2,150
  O = output reserve for diff plus rationale, tokens  = 4,096
  m = truncation safety margin as a fraction          = 0.06
  b = mean file body size, tokens                     = 300
  h = per-file header and separator, tokens           = 34

CONSTRAINT
  Valid only when W, S, T, L, O are counted with the same tokenizer that serves the request,
  and when O is reserved before packing. If the service reserves output lazily, the last
  packed file is silently truncated instead of the packer dropping it, which is the failure
  the margin exists to prevent.

NUMERICAL EXAMPLE
  Fixed overhead      = 1,480 + 720 + 2,150 + 4,096 = 8,446 tokens
  Packable space      = 131,072 - 8,446             = 122,626 tokens
  After margin        = 122,626 * 0.94              = 115,268.44 -> 115,268 tokens
  Per-file footprint  = 300 + 34                    = 334 tokens
  File slots          = floor(115,268 / 334)        = floor(345.11) = 345 files

INTERPRETATION
  The window allows 345 average files. The pilot uses k = 12. Context capacity is therefore
  not the binding constraint on this task family; retrieval precision is. That reading matters
  because it removes "use a bigger window" from the option list: a configuration that packs
  345 files would spend 28.8 times the prefill compute to add mostly irrelevant context, and
  attention cost grows faster than linearly with sequence length.
```

The number 345 is a count, not a recommendation, and it is worth seeing what happens when the
distribution is not average. A generated lockfile of 412,000 tokens exceeds the entire window, so the
packer must have an exclusion rule for generated files; without one, one such file evicts every source
file and the attempt is guaranteed to fail while looking like a model failure. That is the boundary
case below.

### 6.2 Per-attempt token cost and cost per accepted patch

```text
FORMULA
  c_attempt = (T_in * p_in + T_out * p_out) / 1e6
  c_accepted = (tasks * attempts_per_task * c_attempt) / accepted_tasks
  accepted_tasks = tasks * (1 - (1 - a)^r)

WHERE
  T_in  = mean input tokens per attempt, tokens       = 26,178
  T_out = mean output tokens per attempt, tokens      = 880
  p_in  = contracted input price, USD per million     = 0.45
  p_out = contracted output price, USD per million    = 1.80
  a     = first-attempt acceptance probability        = 0.42
  r     = mean attempts per task                      = 2.4
  tasks = evaluation population                       = 400

CONSTRAINT
  Valid when attempts are approximately independent and when the retry cap is not binding.
  With r = 2.4 and a cap of 3, the geometric model is a stand-in, not the true attempt
  distribution; the constraint check is to compare predicted task-level solve against the
  observed solved_within_three count and to distrust the ratio if they diverge by over
  5 percentage points.

NUMERICAL EXAMPLE
  Input cost     = 26,178 * 0.45 / 1e6 = 0.0117801 USD
  Output cost    =    880 * 1.80 / 1e6 = 0.0015840 USD
  Per attempt    = 0.0117801 + 0.0015840 = 0.0133641 USD
  Attempts       = 400 * 2.4 = 960
  Total spend    = 960 * 0.0133641 = 12.8295 USD
  Task solve     = 1 - (1 - 0.42)^2.4
                 = 1 - 0.58^2.4
                 ln 0.58 = -0.544727; 2.4 * -0.544727 = -1.307345
                 exp(-1.307345) = 0.270584 -> solve = 0.729416
  Accepted tasks = 400 * 0.729416 = 291.77
  Cost/accepted  = 12.8295 / 291.77 = 0.043972 USD

INTERPRETATION
  Token spend is 12.83 USD for 400 tasks and 4.4 cents per accepted patch, against a
  guardrail of 10 cents. Cost is not close to binding. The same computation for C3 gives
  31,600 * 0.45/1e6 = 0.01422 plus 1,050 * 1.80/1e6 = 0.00189, so 0.01611 per attempt over
  800 attempts, 12.888 USD, against 400 * 0.7975 = 319.0 accepted tasks, or 4.04 cents.
  C3 is cheaper per accepted patch than C2 despite costing 20% more per attempt, because
  its acceptance rate is higher. Ranking by per-attempt price would have picked C1.
```

Prefix caching changes the input term and is worth computing because a retrieval-heavy configuration
repeats the same repository context across attempts far more than it repeats the task. With a hit
fraction `hh` and cached tokens priced at 10% of the normal rate, the effective input price is
`p_in * ((1 - hh) + 0.10 * hh)`. At `hh = 0.55` that multiplier is `0.45 + 0.055 = 0.505`, so the input
term falls from 0.01178 to 0.00595 and the per-attempt cost drops to 0.00753 USD — a 43.6% reduction
that does not change any ranking in this case, which is a useful thing to know before spending
engineering time on it.

### 6.3 Break-even first-attempt acceptance against the manual baseline

```text
FORMULA
  E[minutes] = A * a + (R + W + M) * (1 - a)
  a* = (R + W + M - M_manual) / (R + W + M - A)

WHERE
  A        = minutes for an accepted first attempt (setup + wait + review)  = 7.5
  R        = minutes to review and reject a wrong patch                     = 4.0
  W        = minutes to discard, revert, and clean up                       = 6.5
  M        = minutes to implement manually after a rejection                = 42.0
  M_manual = minutes for the manual baseline with no assist                 = 46.0
  a        = first-attempt acceptance probability

CONSTRAINT
  Valid when a rejected attempt still ends in a manual implementation of the same task, which
  is the normal workflow. It is invalid when a rejection is followed by a second model attempt
  instead, because then the rejection cost is paid again without the manual fallback.

NUMERICAL EXAMPLE
  Rejection path total = 4.0 + 6.5 + 42.0 = 52.5 minutes
  52.5 - 46.0 = 6.5 ;  52.5 - 7.5 = 45.0
  a* = 6.5 / 45.0 = 0.14444 -> 14.4% is the formal break-even
  Recheck at the formal point: 7.5*0.14444 + 52.5*0.85556 = 1.0833 + 44.9167 = 46.0 minutes
  At C2's a = 0.42: 7.5*0.42 + 52.5*0.58 = 3.15 + 30.45 = 33.60 minutes
  Saving = 46.0 - 33.60 = 12.40 minutes, or 27.0% of baseline

INTERPRETATION
  Break-even is far below any plausible acceptance rate, which is the strongest argument for
  the programme and also the easiest argument to misuse. The reason the threshold is low is
  that a rejected patch costs only 10.5 minutes more than doing the task manually, since the
  developer still has to implement it. Change the rejection path and the conclusion moves a
  lot: if reviewers only discover a wrong patch after 22 minutes of reading (R = 22), the
  rejection path becomes 70.5 minutes and a* = (70.5 - 46.0) / (70.5 - 7.5) = 24.5 / 63.0
  = 38.9%, which C2's 0.42 barely clears and C1's 0.31 does not.
```

That sensitivity is the point. The decision does not hinge on acceptance in the abstract; it hinges on
how fast a wrong patch is detected. A configuration that surfaces its own failure — running the test,
showing the diff against the failing assertion — is worth more than a configuration that is slightly
more accurate but leaves the reviewer to find the error. This is why C3's rejection path
(`R = 2.5`, `W = 3.0`) produces 25.2 minutes even though its acceptance rate is only 13 points above C2.

Recomputing C1 and C3 with their own rejection paths, using `A` of 8.0 and 7.0 and rejection totals of
54.5 and 47.5 minutes: C1 gives `8.0*0.31 + 54.5*0.69 = 2.48 + 37.605 = 40.09` minutes, a 12.8% saving;
C3 gives `7.0*0.55 + 47.5*0.45 = 3.85 + 21.375 = 25.23` minutes, a 45.2% saving. The ordering on
minutes is C3, C2, C1, which is also the ordering on acceptance — but not the ordering on cost per
accepted patch, where C1 leads at 2.5 cents. Section 7 reconciles them.

### 6.4 Dependency closure and the probability a patch sees every file it needs

```text
FORMULA
  P(complete | d) = q^d
  P(complete) = exp(d_mean * ln q)

WHERE
  q      = per-required-file probability that the file appears in the retrieved set  = 0.88
  d      = number of required files for the task
  d_mean = mean required files per task                                            = 3.6

CONSTRAINT
  Valid only under independence: each required file's retrieval is treated as a separate
  Bernoulli trial with the same q. Required files in the same package share vocabulary and
  import structure, so misses are positively correlated and the true completeness is lower
  than q^d for clustered tasks. The formula is an optimistic bound, not an estimate.

NUMERICAL EXAMPLE
  d = 3 : 0.88^3 = 0.88 * 0.7744 = 0.681472
  d = 4 : 0.7744 * 0.7744     = 0.599695
  d = 6 : 0.599695 * 0.7744   = 0.464404
  d_mean: ln 0.88 = -0.127833 ; 3.6 * -0.127833 = -0.460200
          exp(-0.460200) = 0.631120
  Task-level solve is then bounded by 0.6311 at best, before any model error is counted.

INTERPRETATION
  At a 0.88 per-file hit rate, more than a third of tasks never see all their files, so the
  ceiling on correctness is 0.63 regardless of model quality. This is the single most useful
  number in the case, because it explains why a better model on the same retrieval produces
  small gains: the model is being asked to reason about code it was not shown.
```

The boundary is clustering. If a task needs two packages and each package is retrieved completely with
probability 0.68, completeness is `0.68^2 = 0.4624`, and with three packages it is `0.68^3 = 0.314432`.
The independence model at `d = 6` predicted 0.4644, which is right for two packages and 32% too
optimistic for three. Since the observed population has a long tail of multi-package tasks, a plan that
budgets retrieval capacity from `q^d` alone will under-reserve second-hop expansion. The check is an
import-graph audit that counts packages per task, not files.

Retrieval recall also interacts with k in the other direction. Raising `k` from 12 to 24 lifts `q`
slightly, to perhaps 0.91, so completeness at `d = 6` goes to `0.91^6 = 0.5679` — but it also doubles
prefill tokens and adds distracting context. Whether the distraction cost exceeds the recall gain is an
empirical question; the design answers it by measuring acceptance at both values on the holdout rather
than assuming monotonic improvement.

### 6.5 Wilson interval, and what clustering does to it

```text
FORMULA
  centre = (p + z^2/(2n)) / (1 + z^2/n)
  half   = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
  n_eff  = n / DEFF

WHERE
  p    = observed proportion                                     = 0.7294
  n    = tasks in the arm                                        = 400
  z    = 1.96 for a two-sided 95% interval
  DEFF = design effect from clustering by change family          = 1.6

CONSTRAINT
  Valid when tasks are independent draws from one population. Tasks sharing a changed symbol
  are not independent, so the unadjusted interval understates uncertainty and must be widened
  by the design effect before it is used in a decision rule.

NUMERICAL EXAMPLE
  z^2 = 3.8416 ; z^2/n = 0.009604 ; z^2/(2n) = 0.004802
  centre = (0.7294 + 0.004802) / 1.009604 = 0.734202 / 1.009604 = 0.727218
  p(1-p) = 0.7294 * 0.2706 = 0.197376 ; /400 = 0.00049344
  z^2/(4n^2) = 3.8416 / 640,000 = 0.0000060025
  sqrt(0.00049344 + 0.0000060025) = sqrt(0.00049944) = 0.022348
  half = 1.96 * 0.022348 / 1.009604 = 0.043802 / 1.009604 = 0.043385
  Interval = [0.683833, 0.770603]
  With DEFF: n_eff = 400 / 1.6 = 250
  p(1-p)/250 = 0.00078950 ; z^2/(4n^2) = 3.8416/250,000 = 0.0000153664
  sqrt(0.00080487) = 0.028370 ; half = 1.96*0.028370/(1+0.0153664) = 0.055605/1.0153664 = 0.054763
  Adjusted interval = [0.672455, 0.781981]

INTERPRETATION
  The clustered interval spans 11 percentage points. Any two configurations whose acceptance
  rates differ by less than about 8 points are indistinguishable at this sample size, which is
  exactly the gap between C2 at 0.42 and a hypothetical C2.5 at 0.49. The decision rule in
  section 9 therefore refuses to choose on acceptance alone and falls back on the minutes
  metric, whose interval is narrower because minutes are measured per task rather than as a
  binary event.
```

The interval is also the reason the pilot is sized the way it is. A 400-task arm detects a difference of
about 9.7 percentage points between two acceptance rates near 0.59 at 80% power, computed from
`(1.96 + 0.8416) * sqrt(2 * 0.59 * 0.41 / 400) = 2.8016 * 0.034778 = 0.0974`. Detecting a 3-point
difference needs roughly 593 tasks per arm by the same formula, which is a larger pilot than the budget
allows. The honest response is not a bigger claim but a narrower question: decide between configurations
that differ by more than the detectable difference, and treat close calls as deferrals.

### 6.6 Attempt independence, checked rather than assumed

```text
FORMULA
  r_eff = 1 + (r - 1) * rho
  P(solve) = 1 - (1 - a)^r

WHERE
  r   = nominal attempts per task                     = 2.4
  rho = intra-task correlation of attempt success     = 0.35 (check value)
  a   = first-attempt acceptance                      = 0.42

CONSTRAINT
  Valid when rho is estimated from repeat attempts on the same task. Assuming rho = 0 makes
  the attempt count look more productive than it is, because a second attempt by the same
  model on the same packed context usually repeats the same mistake.

NUMERICAL EXAMPLE
  Nominal  : 1 - 0.58^2.4 = 0.729416
  Effective: r_eff = 1 + 1.4 * 0.35 = 1.49
             1 - 0.58^1.49 ; ln 0.58 * 1.49 = -0.811643 ; exp = 0.444095
             solve = 0.555905
  Overstatement = 0.729416 - 0.555905 = 0.173511, i.e. 17.4 percentage points

INTERPRETATION
  If attempts are correlated, the nominal attempt budget overstates task-level solve by 17
  points, and the cost per accepted patch in section 6.2 is understated by the same factor
  direction. The discriminating check is the pair (attempt 1 failed, attempt 2 accepted)
  counted per task: under independence that count should be near a*(1-a) of tasks, and a much
  lower count indicates the retry loop is repeating work rather than exploring.
```

This is where the test loop earns its keep in a way the acceptance table does not show. Feeding failing
test output back into the next attempt changes the model's input, which is a genuine perturbation rather
than a resample, so `rho` falls. A configuration with a slightly lower `a` but a much lower `rho` can
win on task-level solve; the design measures both.

## 7. The Finding

Six findings, in the order a decision-maker needs them.

**F-1. Cost is not the constraint; reviewer attention is.** Token spend is 12.83 USD across 400 tasks
and 4.4 cents per accepted patch against a 10-cent guardrail. Reviewer minutes at 33.6 minutes per task
over 400 tasks come to 224 hours. A programme justified by token savings is justified by the wrong
term, and a programme rejected because "the API is expensive" is rejected on a rounding error.

**F-2. Retrieval completeness caps quality before the model does.** At a 0.88 per-file hit rate with a
mean of 3.6 required files, the probability that a patch sees every file it must respect is 0.631.
No model upgrade moves that ceiling. The first engineering investment is second-hop import-graph
expansion and an explicit dropped-file counter, not a larger model.

**F-3. The three plausible rankings disagree, and the disagreement is informative.** On cost per
accepted patch the order is C1 (2.5 cents), C2 (4.4), C3 (4.0). On first-attempt acceptance it is C3
(0.55), C2 (0.42), C1 (0.31). On expected minutes it is C3 (25.2), C2 (33.6), C1 (40.1). Only the
minutes ranking reflects the workflow, and only it is stable under the sensitivity checks.

**F-4. Detection speed dominates accuracy in the break-even.** With a fast rejection path the threshold
is 14.4%; with a slow one it is 38.9%. The same model is worth deploying or not worth deploying
depending on how quickly a wrong patch is caught. This makes review tooling part of the model decision,
not a downstream nicety.

**F-5. The sample cannot resolve small differences.** The clustered 95% interval on 0.7294 is
[0.672, 0.782]. Two configurations inside that band are tied for decision purposes. A pilot that
declares a 2-point winner is reporting noise; the correct output is a deferral or a tie-break on cost.

**F-6. Refusal is load-bearing.** Roughly 18% of filtered tasks lack a reproducible test and some
fraction of the rest need a file the requester cannot read. Serving those tasks produces patches with
undefined correctness or unauthorised disclosure, so abstention is a component of the design rather
than a fallback error path.

Stated as one sentence for the decision memo: **adopt C3 for the pilot because it wins on expected
minutes and on task-level solve, keep the abstain branch, and spend the next engineering cycle on
retrieval closure rather than on a larger model.**

## 8. System/Solution Design

The design is a pipeline with five gates, and each gate fails closed. The gates exist because each of
the six findings above corresponds to a place where an unchecked assumption becomes a silent defect.

**Gate 1 — identity and authority.** Resolve task, commit, requester, and purpose. Ask the permission
service for a decision on each candidate path. Attach the permission epoch. Any path without an allow
decision is removed before ranking, so an unauthorised file cannot be ranked highly and cannot be
packed. This gate implements C-02 and C-09 together: the epoch is what makes a cache safe.

**Gate 2 — ranking and closure.** Rank paths by lexical and semantic relevance to the task, take the top
`k`, then expand one hop through the import graph restricted to paths the requester may read. Record the
closure additions separately, because a closure addition that is denied is an authorisation event worth
counting, and because closure-expanded files tend to be the ones the model actually needs.

**Gate 3 — packing.** Order chunks deterministically: type definitions and interfaces before
implementations, the failing test as a signature stub, the target file's full body, then neighbours by
rank. Count tokens with the target tokenizer. Drop from the tail, and log every drop with reason
(`budget`, `superseded-by-closure`, `generated-file`, `too-large`). Emit the packed-input hash.

**Gate 4 — generation and repair.** Request a unified diff plus a rationale. Apply to a scratch copy of
the pinned commit; assert the hash before applying. Run the declared test command. On failure, feed a
bounded output excerpt back and retry, up to three attempts total. Stop early on a pass. Record the
attempt index and the correlation-relevant history.

**Gate 5 — review and release.** Present diff, test output, packed file list, and the dropped list to a
maintainer. The verdict is recorded before the rationale is shown. Merge requires the maintainer action;
there is no auto-merge path. Metering closes the row and joins the 14-day revert tracker.

**Interfaces and their invariants.**

| Interface | Input | Output | Invariant enforced |
|---|---|---|---|
| `authorise` | requester, purpose, path, commit | allow/deny, epoch | deny is final for the request |
| `retrieve` | task, commit, epoch, k | ranked chunks with content hashes | every chunk has an allow decision |
| `pack` | chunks, tokenizer, budgets | packed string, drop list, hash | token count ≤ usable budget |
| `generate` | packed string, template hash | diff, rationale, usage | template hash recorded per attempt |
| `apply` | diff, pinned commit | applied tree or conflict | applied commit equals pinned commit |
| `test` | tree, command | exit code, duration, excerpt | network disabled, source read-only |
| `review` | diff, test output, file list | verdict, edits, minutes | verdict recorded before rationale |
| `meter` | attempt rows | cost, latency, retries | one row per attempt, never per task |

**Ordering that matters.** The permission decision precedes ranking, not the other way round. Ranking
first then filtering is the common implementation and it leaks through side channels: timing, rank
position, and error text all reveal that a file exists. Filtering first makes the deny invisible.

**Determinism.** Packing is deterministic given the same chunks and tokenizer, so a replayed request
produces the same input hash. Generation is not deterministic unless decoding is fixed; the design pins
temperature and seed for the evaluation runs and lets the serving configuration relax them afterwards,
which is why the evaluation row carries the decoding parameters.

**Degradation path.** If generation exceeds the p95 latency guardrail, the request falls back to C1-style
paste-only help, not to a longer wait. If the permission service is unavailable, the request is denied
rather than served from cache. If the sandbox is unavailable, C3 degrades to C2 and the attempt is
labelled as such so the acceptance numbers are not attributed to a tool loop that never ran.

**What the design deliberately does not do.** No cross-request context sharing beyond the permission-keyed
cache; no learning from maintainer verdicts inside the pilot; no automatic test generation, because a
model-written test that passes is not evidence of correctness and would corrupt the label. Each of these
is a plausible product feature and each one destroys the measurement.

## 9. Evaluation & Validation

**Primary metric.** `first_pass_accepted` rate. Numerator: tasks whose first attempt was merged with no
substantive diff change. Denominator: tasks attempted under that configuration, excluding infrastructure
failures but including content failures. Population: the 400 filtered tasks created before the cut-off.
Window: the eight-week evaluation window, with the 14-day revert window extending past it. Comparator:
C0 manual, whose acceptance is 1.000 by construction and whose cost is 46 minutes. Guardrails: zero
unauthorised disclosures, p95 latency ≤ 90 s, token spend ≤ 10 cents per accepted patch, revert rate
≤ 3%. Uncertainty: Wilson interval with the design-effect adjustment from 6.5. Status: **not executed**.

**Secondary metrics.** `solved_within_three` with numerator = tasks solved by any attempt and
denominator = tasks attempted; `minutes_per_accepted_task` from developer timestamps, including
generation wait; `attempts_per_task` as the review-load proxy; `dropped_files_per_request` as the
truncation proxy; `closure_denials` as the authorisation-event count.

**Discrimination suite.** Four checks, each aimed at a specific confound.

1. **Revision check.** Replay 30 attempts with the packing step pointed at HEAD instead of the pinned
   commit. Acceptance must fall; if it does not, the revision constraint is decorative and the whole
   case is measuring something else.
2. **Oracle check.** Run the same tasks with test bodies excluded from retrieval. Acceptance should fall
   by a large margin on tasks whose tests contain expected literals. If it does not, either the tests
   carry no oracle or the model is not using them, and the finding about leakage needs rewriting.
3. **Attempt-independence check.** Count the `(attempt 1 failed, attempt 2 accepted)` pairs per task.
   Under independence this is near `a * (1 - a)` of tasks.
4. **Tie-break check.** Compare two configurations whose acceptance differs by under 8 points. The test's
   job is to confirm the framework refuses to declare a winner.

**What is executed and what is not.** The arithmetic in section 6 is executed and rechecked. The
evaluation is not executed: no model was served, no developer was timed, no maintainer verdict exists.
Every acceptance rate, minute count, and price in this document is `ILLUSTRATIVE`. The claim the
manuscript makes is structural — that the decision hinges on minutes and closure rather than on token
price or benchmark score — and that claim is supported by the algebra, not by a pilot.

**Decision rule.** Adopt the configuration whose `minutes_per_accepted_task` point estimate is lowest
and whose clustered lower bound still beats the manual baseline by at least 5 minutes, provided all
guardrails are met in the dry-run harness. If no configuration clears that bar, defer. If two clear it
within the interval described in F-5, choose the one with the cheaper rejection path, because that is
the term the arithmetic shows to be most fragile.

## 10. Action Taken

The recommendation is a staged adoption of C3 with an explicit abstain branch, and four gates that must
each be demonstrated before the next stage. This is a design recommendation, not a report of an action
that was carried out; the stages are written so a future owner can execute them in order.

**Stage 0 — harness before assistance.** Build the permission-filtered retrieval path, the packer with
its drop log, the pinned-commit apply step, and the metering join. Accept no assistance request until
the permission log shows deny decisions being recorded for paths that exist. A harness that has never
denied a request has never been exercised.

**Stage 1 — shadow mode.** Run C2 and C3 on the 400-task population without showing output to
developers. Compare acceptance against the harness's own test runs. The purpose is to validate the
label pipeline and the drop log, not to produce a quality claim. Exit criterion: drop-log completeness
at 100% against a manual audit of 40 requests, and a stable packed-input hash on replay.

**Stage 2 — paired pilot.** Run the paired design with a fixed rotation over 400 tasks, verdicts
recorded before rationales are shown, and a 10% double-review sample. Exit criterion: the metrics in
section 9 computed with intervals, and the discrimination suite run at least once per confound.

**Stage 3 — bounded production.** Enable for one module and one team, with the abstain branch live,
throttling at the cost guardrail, and a weekly review of reverts. Reversal conditions: revert rate above
3% for two consecutive weeks, any unauthorised disclosure, or p95 latency above 90 seconds for a day.

**What would make the owner choose deferral instead.** A manual baseline materially below 46 minutes,
a rejection path longer than about 20 minutes, a retrieval hit rate below roughly 0.75, or a pilot
sample that cannot resolve the candidate gap. Any one of those turns the expected saving inside the
interval and the honest action is to keep the current process and revisit when one input changes.

**Owner decisions recorded, not inferred.** The productivity owner sets the configuration; repository
owners can veto a path set; maintainers can reject any diff without appeal; security owns the revert-rate
suspension. The design records these as separate authority so that no quality metric can be traded
against an authorisation guarantee.

## 11. Outcome & Results

**Measured outcomes: none.** `not-measured` applies to every column below. The table states what the
arithmetic implies under `ILLUSTRATIVE` inputs, and what would have to be true for the implication to
hold in a real deployment.

| Quantity | Illustrative value | Basis | What would falsify it |
|---|---:|---|---|
| Manual baseline minutes | 46.0 | assumed per task family | timed manual subsample above 55 |
| C2 first-attempt acceptance | 0.42 | assumed | paired run below 0.30 |
| C3 first-attempt acceptance | 0.55 | assumed | tool loop adds no measurable gain |
| C3 minutes per accepted task | 25.2 | derived from 7.0/2.5/3.0/42.0 | rejection path above 20 minutes |
| Token spend, 400 tasks | 12.83 USD | derived at 0.45/1.80 per million | price change above 2x |
| Per-request recall hit rate | 0.88 | assumed audit value | closure audit below 0.75 |
| Task-level completeness | 0.631 | derived, optimistic | three-package clustering at 0.68 each |
| Revert rate within 14 days | not-measured | guardrail only | any value above 0.03 suspends |

**Sensitivity.** Three inputs dominate. Raising the manual baseline to 55 minutes raises every saving
proportionally and makes C1 viable at 27.1% (`1 - 40.09/55.0`). Raising the rejection path `R` to 20
minutes makes C2's break-even 31.6% and puts C1 below threshold. Lowering `q` to 0.78 caps completeness
at `0.78^3.6 = exp(-0.8944) = 0.4089`, at which point no configuration can exceed about 41% solve and
the programme should be re-scoped to retrieval work first.

**Cost of being wrong.** If C3 is adopted and its acceptance is really 0.35 rather than 0.55, expected
minutes become `7.0*0.35 + 47.5*0.65 = 2.45 + 30.875 = 33.33`, still a 27.5% saving, and the operating
cost is a sandbox runner. The asymmetry is favourable: the downside is a smaller saving rather than a
loss, because the manual fallback is retained inside the rejection path. The one genuine downside risk
is a silent wrong merge, which is why the revert guardrail exists and why it is a suspend condition
rather than a monitored metric.

**What a later reviewer should demand before citing this.** A real acceptance rate with its interval, a
real closure audit, a real rejection-path measurement, and a revert join. Until those exist, the file is
a decision scaffold with checked arithmetic — useful for reasoning, unusable as evidence of a result.

## 12. Failure Modes & What Went Wrong

Six mechanisms, each with competing causes. The table names the mechanism and the check that separates it
from its lookalike; the paragraphs explain why the lookalike is plausible.

| ID | Mechanism | Competing cause that looks identical | Discriminating check |
|---|---|---|---|
| FM-1 | Patch targets a stale or wrong revision | Model lacks capability | Apply against pinned hash; count conflicts per config |
| FM-2 | Test oracle leaked through retrieved test body | Model is genuinely good at the task | Rerun with test bodies excluded; compare delta |
| FM-3 | Silent wrong merge accepted without test | Test suite is weak in general | 14-day revert join; mutation probe on guardrail tests |
| FM-4 | Retrieval missed a required dependency | Model cannot reason across files | Import-graph closure audit; error-class histogram |
| FM-5 | Review capacity saturated by attempt inflation | Reviewers are slow or under-resourced | Attempts per task vs queue latency over weeks |
| FM-6 | Cached context outlived a permission change | Model regurgitated old code | Permission epoch in cache key; revocation replay |

**FM-1, stale revision.** The failure looks like a bad patch and is actually a bookkeeping failure.
Retrieval served file content from a branch tip, the apply step used the working tree, and the diff
hunk offsets shifted. The patch is rejected, acceptance falls, and the team concludes the model is weak
on this codebase. The discriminating check is mechanical: count apply conflicts per configuration. A
configuration with a high conflict count has an identity bug, not a quality bug, and no amount of model
upgrading fixes it.

**FM-2, oracle leakage.** Because the failing test usually contains the expected value, a retrieved test
body turns a reasoning task into a copying task. Acceptance looks excellent and generalisation is zero.
The lookalike cause is a genuinely strong model, and the only way to separate them is the exclusion
rerun: if acceptance collapses when test bodies are withheld, the original number was leakage. A second
signal is the shape of the patch — a leaked patch tends to touch only the literal and not the
surrounding control flow.

**FM-3, silent wrong merge.** The patch passes the declared test and breaks something the test did not
cover. Acceptance is 1.0 by definition on merge, so the primary metric cannot see it. The check is the
14-day revert join plus a mutation probe: inject a small defect into a guardrail-covered path and
confirm the suite fails. If the probe survives, the test contract is too weak for the acceptance number
to mean anything, and the configuration's quality is unmeasured regardless of its score.

**FM-4, closure miss.** The model writes a patch that calls a method that does not exist at that
revision, or imports from a module the retrieval never surfaced. This presents as a reasoning failure
but is an information failure. The error-class histogram separates them: a closure miss concentrates in
`NameError`, `ImportError`, undefined-attribute, and type-mismatch classes, while a genuine reasoning
failure spreads across logic and boundary errors. The audit is the import graph, not the model.

**FM-5, review saturation.** A configuration that raises attempts per task from 2.0 to 2.9 increases
review queue depth even when per-attempt quality is unchanged. The symptom is a rising review latency
that looks like reviewer under-staffing. The check is attempts per task plotted against queue latency:
if the two move together, the assistant is generating the backlog. The correction is a cap on attempts
per task, not more reviewers.

**FM-6, authorisation drift.** A developer's access is revoked at 10:00; the retrieval cache was keyed
by path and commit only, so a request at 10:05 is served context the requester can no longer read. This
is the highest-severity mechanism in the list because it is invisible in quality metrics and appears
only in an audit. The check is a revocation replay: revoke a permission, issue a request immediately,
and assert a deny. If the cache key does not embed the permission epoch, the replay passes only by
accident of expiry timing.

**Two further mechanisms worth naming.** Truncation drift (FM-7): the packer drops the highest-value
file because the ordering policy is unstable, which looks like a model failure on exactly the tasks
where context was largest; the check is the drop log joined to rejection reason. Benchmark contamination
(FM-8): a public score inflates the candidate during selection, so the chosen configuration
underperforms its own validation; the check is a novel-internal-API probe plus a post-cut-off task
holdout. Neither is listed in the table above because six rows already cover the mechanisms that change
the decision, and padding the table would dilute the discriminating checks that matter.

## 13. Transferable Lesson

**The lesson in one sentence.** When an assistant's output must be reviewed by a human anyway, the unit
of comparison is minutes per accepted artefact, and the terms that dominate it are detection speed and
input completeness — not model accuracy and not token price.

Four transfers follow, each stated as a rule a practitioner can apply to a different domain.

**Transfer 1 — price the human term explicitly.** Any workflow with a mandatory human gate has a review
cost that dwarfs inference cost. Writing the review minutes into the expected-value expression converts
an argument about model quality into an argument about detection latency, which is something engineering
can actually improve. Teams that skip this step optimise the cheapest term and are surprised when the
programme does not pay back.

**Transfer 2 — treat retrieval completeness as a ceiling, then measure it.** A model cannot respect code
it was not shown. Before comparing models, measure the probability that the context is complete for the
task. If it is 0.63, the model comparison is bounded at 0.63 and the next investment is retrieval, not
weights. The same logic applies to document question answering, incident triage, and any task whose
evidence is selected rather than given.

**Transfer 3 — make the failure cheap to detect.** The break-even acceptance rate moved from 14% to 39%
purely by slowing detection. That is a larger effect than most model upgrades. Designing output so a
wrong answer is visibly wrong — showing the diff against the failing assertion, running the test,
displaying the packed inputs — is a quality intervention, not a UX nicety.

**Transfer 4 — refuse to resolve differences your sample cannot see.** With a clustered interval 11
points wide, a 2-point win is noise. Publishing it as a win teaches the organisation to trust a metric
that has no resolving power, and the next decision inherits that false confidence. A stated tie-break
rule is a stronger deliverable than a spurious ranking.

**Anti-lesson.** Do not conclude "the model is not good enough" from a rejection-heavy pilot. In this
case the plausible causes of rejection are revision drift, oracle leakage in reverse (a test that
encodes the wrong behaviour), closure misses, and review saturation, and only one of them is about model
capability. The diagnostic order in section 12 exists because the default explanation is usually wrong.

## 14. Connections & Cross-References

- Token budgeting and sequence accounting: [`../Cheat_Sheets/`](../Cheat_Sheets/INDEX.md) sheet family
  for context budgets; the derivation in 6.1 is the same arithmetic with a repository packing policy
  layered on top.
- Attention and transformer block cost: [`../03_Attention_Mechanism.md`](../03_Attention_Mechanism.md)
  for why packing 345 files is not merely wasteful but superlinearly expensive.
- Scaling and parameter counting: [`../05_Scaling_and_Parameter_Counting.md`](../05_Scaling_and_Parameter_Counting.md)
  for the parameter/FLOP vocabulary used when arguing that a bigger model does not fix a closure miss.
- Training objectives: [`../3.1_Training_Objectives.md`](../3.1_Training_Objectives.md) for why a
  next-token objective produces plausible-but-unverified code and why the test contract is external.
- Context length: [`../11.1_Why_Context_Length_Matters.md`](../11.1_Why_Context_Length_Matters.md) for
  the nominal-versus-usable distinction that 6.1 quantifies.
- Sibling case: model selection within this unit is a planned commitment; see `INDEX.md` section 5 rather than this draft.
  takes the same decision family from the pilot-selection side, with candidate ranking and benchmark
  transfer as its subject.
- Handoffs: fine-tuning and adapters belong to module 21; serving admission and KV policy belong to the
  inference-behaviour thread of this module; review-policy change belongs to the engineering-process
  owner and is out of scope here.

### Interview pack

**Q1. A vendor reports 92% on a public code benchmark. Your pilot shows 42% first-attempt acceptance.
Is the model misrepresented?**
- Answer: Not necessarily, and the gap alone proves nothing. The benchmark scores a function body
  against a provided harness with the relevant code in the prompt; the pilot adds commit pinning, path
  authorisation, retrieval, and a reviewer. The right response is to decompose the gap before judging.
- Reasoning: The two measurements differ in four ways at once, so a single ratio is unidentifiable. The
  decomposition that matters is retrieval completeness (0.63 ceiling), revision correctness (conflict
  count), oracle availability (test bodies excluded), and label strictness (no substantive edit).
- Follow-up: If you exclude test bodies from retrieval and acceptance falls from 42% to 18%, what have
  you learned about the original 42%?
- Red flag: Accepting the benchmark as ground truth and concluding the vendor lied, or accepting the
  pilot number without checking whether the patch ever saw the required files.
- Scoring: 2 = names at least three of the four gap sources and proposes a decomposition; 1 = names one
  source; 0 = asserts deception or dismisses the pilot.

**Q2. Compute the break-even first-attempt acceptance rate when the manual baseline is 46 minutes, an
accepted attempt costs 7.5 minutes, and a rejection costs 4.0 review + 6.5 cleanup + 42.0 manual
implementation.**
- Answer: 14.4%. The rejection path totals 52.5 minutes; the expression `a* = (52.5 - 46.0)/(52.5 - 7.5)`
  gives `6.5/45.0 = 0.1444`.
- Reasoning: Break-even is where the expected minutes equal the baseline, so the surplus cost of a
  rejection over the baseline is spread across the range between the two path costs.
- Follow-up: Recompute with `R = 22` minutes and state which of acceptance rates 0.31, 0.42, 0.55 clear it.
- Red flag: Dividing by the baseline instead of the cost difference, or ignoring cleanup.
- Scoring: 2 = correct 14.4% with the intermediate 52.5 and 45.0 visible and the recomputation at 38.9%;
  1 = correct formula, arithmetic slip; 0 = compares acceptance to a cost ratio.

**Q3. Your context window is 131,072 tokens. System prompt 1,480, task 720, tool schemas 2,150, output
reserve 4,096, safety margin 6%, mean file 300 tokens plus a 34-token header. How many files fit?**
- Answer: 345 files. Overhead is 8,446, packable space 122,626, after margin 115,268, per-file 334,
  and `floor(115268/334) = 345`.
- Reasoning: The output reserve and the margin must be subtracted before dividing, because the packer
  cannot use space it has promised to the response.
- Follow-up: A 412,000-token generated lockfile enters the candidate set. What must the packer do, and
  what happens if it does nothing?
- Red flag: Forgetting the margin or the output reserve, or answering "as many as you like".
- Scoring: 2 = 345 with all intermediate values and the lockfile exclusion rule; 1 = correct method,
  arithmetic error; 0 = divides the raw window by 334.

**Q4. Why is packing 345 files into a 131k context a bad idea when 12 files suffice?**
- Answer: Prefill and attention cost grow with sequence length, and irrelevant context distracts
  selection. The 345-slot capacity is an upper bound, not a target; k is set by retrieval precision.
- Reasoning: Prefill work scales with packed tokens, and attention work grows faster than linearly, so
  the cost per attempt rises steeply while acceptance does not.
- Follow-up: What measurement would tell you that raising k from 12 to 24 was net positive?
- Red flag: Claiming a larger context is always better, or ignoring that the decision metric is
  acceptance per minute.
- Scoring: 2 = names both compute growth and distraction, plus a measurement that could settle it;
  1 = names one; 0 = says more context is strictly better.

**Q5. At a 0.88 per-file retrieval hit rate and a mean of 3.6 required files, what is the probability
that a patch sees every file it must respect?**
- Answer: 0.631. Using `exp(3.6 * ln 0.88) = exp(-0.4602) = 0.6311`; the discrete checks are 0.681 at
  d = 3 and 0.464 at d = 6.
- Reasoning: Under independence, completeness is a product of hit probabilities, so it decays
  exponentially in the number of required files.
- Follow-up: Three packages are required and each is retrieved completely with probability 0.68. What is
  completeness, and why does the independent formula mislead?
- Red flag: Treating the hit rate as the task success probability, or ignoring the clustering caveat.
- Scoring: 2 = 0.631 with the log form or the product chain plus the clustering correction at 0.314;
  1 = correct product for one d but no general form; 0 = answers 0.88.

**Q6. Your observed task-level solve is 0.729 with 2.4 attempts and per-attempt acceptance 0.42. Attempts
turn out to correlate at rho = 0.35. What is the corrected solve rate?**
- Answer: About 0.556. Effective attempts are `1 + 1.4 * 0.35 = 1.49`, and `1 - 0.58^1.49 = 0.5559`,
  so the nominal figure overstates by 17.4 points.
- Reasoning: Correlated retries resample the same mistake instead of exploring, so the attempt budget
  buys less than the geometric model assumes.
- Follow-up: Which observable pair per task would you count to estimate rho empirically?
- Red flag: Assuming independence because retries are "separate calls", or inventing a correction
  without recomputing the exponent.
- Scoring: 2 = 0.556 with r_eff and the overstatement quantified; 1 = right idea, no number;
  0 = restates the nominal 0.729.

**Q7. Three metrics rank three configurations differently. Which ranking do you adopt, and what do you
do with the other two?**
- Answer: Adopt the minutes-per-accepted-task ranking, because it is the only one that includes the
  mandatory human gate. Report the cost and acceptance rankings as constraints and diagnostics: cost
  must stay under the guardrail, and acceptance must stay above its break-even.
- Reasoning: A ranking is only meaningful relative to the objective; token cost is a constraint below
  10 cents per accepted patch and acceptance is a threshold, not the objective.
- Follow-up: C1 is cheapest per accepted patch at 2.5 cents and worst on minutes at 40.1. Under what
  change to the inputs would C1 become the right choice?
- Red flag: Averaging the three ranks, or choosing on cost because it is the only number with a dollar
  sign.
- Scoring: 2 = picks minutes with the others as constraints and states a condition that flips the
  choice; 1 = picks minutes with weak justification; 0 = averages ranks.

**Q8. Design the check that distinguishes "the model is weak on this codebase" from "the retrieval
missed a file".**
- Answer: Join rejection reasons to an import-graph closure audit and to the error-class histogram.
  Closure misses concentrate in `ImportError`, `NameError`, undefined-attribute, and type-mismatch
  errors, while capability failures spread across logic and boundary errors. Cross-check by forcing
  closure expansion and seeing whether those error classes drop.
- Reasoning: The two hypotheses predict different error distributions, so a distribution test
  discriminates where an aggregate acceptance number cannot.
- Follow-up: Closure expansion lifts acceptance by 6 points but doubles prefill tokens. Is that a win?
- Red flag: Proposing to "look at some examples", or upgrading the model first.
- Scoring: 2 = names the error-class split plus the intervention test; 1 = names the audit only;
  0 = suggests retraining or manual review.

**Q9. Your clustered 95% interval on 0.7294 is [0.672, 0.782]. Candidate B measures 0.75. What is your
decision, and what is the risk of the obvious alternative?**
- Answer: The two are statistically indistinguishable, so the decision rule falls back to the rejection
  path or to deferral. Declaring B the winner on 2 points teaches the organisation to trust a metric
  that cannot resolve that difference, and the error compounds into later decisions.
- Reasoning: The design effect reduces the effective sample to 250, widening the interval to 11 points;
  differences smaller than the interval width are noise for decision purposes.
- Follow-up: What sample size would detect 3 points at 80% power near p = 0.59, and is that affordable?
- Red flag: Quoting the unadjusted interval, or comparing point estimates as if intervals did not exist.
- Scoring: 2 = correct deferral with the design-effect reasoning and the ~593-per-arm figure; 1 =
  correct deferral without the mechanism; 0 = picks B.

**Q10. A permission is revoked at 10:00. A request at 10:05 is served from a cache keyed on path and
commit. What went wrong, and what is the minimal fix?**
- Answer: The cache outlived the authorisation decision. The minimal fix is to include the permission
  epoch in the cache key so revocation invalidates at the next request rather than at expiry.
- Reasoning: Authorisation is a property of requester, purpose, path, and time; a key missing any of
  those dimensions can return content the requester is no longer entitled to read.
- Follow-up: Write the replay test that would have caught this before deployment.
- Red flag: Proposing a shorter TTL as the fix, which reduces but does not remove the window, or
  treating it as a model problem.
- Scoring: 2 = epoch-keyed cache plus the revocation replay test; 1 = shorter TTL with the residual
  risk acknowledged; 0 = blames the model or ignores the issue.

**Q11. Acceptance rises from 42% to 55% when a sandbox test loop is added, but attempts per task stay at
2.0 while review latency worsens. What is happening?**
- Answer: The loop is converting rejections into additional automated attempts, so the reviewer sees
  fewer wrong patches but the queue is fed by longer-running requests. Check queue latency against
  attempts per task; if they move together, cap attempts rather than adding reviewers.
- Reasoning: The metric that changed is the rejection path, not the attempt count; a reviewer-facing
  latency increase alongside stable attempts points at request duration, not volume.
- Follow-up: If queue latency is flat and only p95 duration rose, what would you look at next?
- Red flag: Assuming more acceptances always mean less work, or hiring reviewers without checking the
  attempt series.
- Scoring: 2 = names the duration-versus-volume distinction and the cap; 1 = names saturation without
  the diagnostic; 0 = concludes the loop is useless.

**Q12. Revert rate at 14 days is 4.5% against a 3% guardrail. What do you check before suspending the
configuration?**
- Answer: First check whether the test contract covers the reverted paths — run a mutation probe. If
  the suite misses the injected defect, the acceptances were never evidence and the guardrail breach is
  a measurement failure as well as a quality one. Then check whether reverts cluster on one file or
  configuration stage.
- Reasoning: A revert can mean a bad patch or a weak suite; separating them determines whether the fix
  is model-side, retrieval-side, or test-side.
- Follow-up: Suppose the probe survives. Is suspending still correct, and what replaces the acceptance
  metric?
- Red flag: Immediately blaming the model, or waiving the guardrail because acceptance looked good.
- Scoring: 2 = mutation probe plus clustering check plus a stated suspension decision; 1 = clustering
  only; 0 = restarts the pilot unchanged.

**Q13. Why is 4.4 cents per accepted patch a weak argument for the programme, and what is the strong
one?**
- Answer: Because the total token spend, 12.83 USD for 400 tasks, is trivial next to 224 reviewer hours.
  The strong argument is 20.8 minutes saved per task, or 45.2% of the manual baseline, driven mostly by
  a cheap rejection path.
- Reasoning: Decisions should be justified by the dominant term; a cost claim that is 1% of the value
  is rhetorically convenient and analytically irrelevant.
- Follow-up: At what reviewer hourly rate does a 20.8-minute saving per task over 400 tasks pay for a
  sandbox runner?
- Red flag: Leading with vendor pricing, or quoting savings without the rejection-path assumption.
- Scoring: 2 = identifies the dominant term and quantifies the saving per task; 1 = identifies the term
  without numbers; 0 = repeats the cents figure as the case.

**Q14. Which single measurement, if it came back bad, would kill this programme?**
- Answer: The manual baseline, because every saving is expressed relative to it. If manual is really 32
  minutes rather than 46, C3's 25.2 minutes is a 21% saving and the adoption case weakens sharply; if it
  is 58 minutes, even C1 clears. A close second is the rejection path length.
- Reasoning: The baseline is the denominator of every saving proportion and the largest unverified
  input; the rejection path is the largest unverified term inside the assistant's own cost.
- Follow-up: Design the 40-task measurement that would pin the baseline within a stated interval.
- Red flag: Naming model quality, which the case deliberately does not rank first.
- Scoring: 2 = baseline with a quantified consequence and the runner-up; 1 = baseline alone; 0 = names a
  metric the case treats as a constraint.

**Q15. Give the operational reason a refusal branch is required rather than optional.**
- Answer: About 18% of filtered tasks have no reproducible test, so correctness is undefined; other
  tasks need a file the requester cannot read. Serving them yields either unverifiable patches or
  unauthorised disclosure, and both are worse than no assist.
- Reasoning: A missing refusal branch converts a policy boundary into a quality metric, and quality
  metrics can be traded away under delivery pressure.
- Follow-up: How would you report a 20% refusal rate to a stakeholder without it reading as failure?
- Red flag: Treating refusals as an error to be optimised down, or letting the model decide access.
- Scoring: 2 = both refusal reasons plus the policy-versus-quality argument; 1 = one reason; 0 = calls
  refusal a fallback bug.

## 15. Quick Reference Card

| Item | Value / rule |
|---|---|
| Decision | Choose among C0 manual, C1 paste-only, C2 pinned retrieval, C3 retrieval + test loop, or defer |
| Owner | Developer-productivity owner; repository owners hold access; maintainers hold merge |
| Grain | One task at one pinned commit; attempts are rows, not tasks |
| Primary metric | First-attempt acceptance, no substantive diff change |
| Decision metric | Expected minutes per accepted task, including generation wait |
| Break-even acceptance | 14.4% at `R = 4.0`; 38.9% at `R = 22.0` |
| Context budget | 115,268 usable tokens, 345 file slots at 334 tokens each |
| Completeness ceiling | 0.631 at `q = 0.88`, `d = 3.6`; 0.314 for three packages at 0.68 |
| Interval rule | Clustered interval 11 points wide; differences under 8 points are ties |
| Cost guardrail | 10 cents per accepted patch; observed illustrative value 4.0 to 4.4 cents |
| Revert guardrail | 3% at 14 days; breach suspends the configuration |
| Always-on checks | Pinned-hash apply, test-body exclusion, drop log, permission epoch in cache key |

Three rules that survive a change of model, vendor, or repository: price the human gate before the
machine, measure context completeness before comparing capability, and refuse to declare a winner inside
the interval.
