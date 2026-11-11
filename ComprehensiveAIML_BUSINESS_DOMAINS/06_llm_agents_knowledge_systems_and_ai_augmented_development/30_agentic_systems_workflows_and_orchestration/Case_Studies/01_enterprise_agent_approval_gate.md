---
case_id: "CASE-2026-030"
case_title: "Approval gate for an enterprise agent facing untrusted instructions"
case_version: "2.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Enterprise operations and agent execution security"
primary_pattern: "exact-action-approval"
secondary_patterns: ["least-privilege-execution", "idempotent-side-effects"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Enterprise agent security lead"
source_index: "INDEX.md"
review_trigger: "Change to action semantics, sink guarantees, policy, model, or adversarial evidence"
---

# Approval gate for an enterprise agent facing untrusted instructions

**Evidence declaration.** Every quantity in this manuscript is constructed for teaching and is labelled
ILLUSTRATIVE. The enterprise, the 4,300 daily requests, the injection prevalence, the service times, the
reviewer cost, and the loss severities were chosen so the arithmetic closes, not measured in a live system.
All derived figures were recomputed by hand and cross-checked with a second route where one existed.
The implementation stage is `proposed`; the document state is `draft`; no independent editorial review is claimed.
General security principles referenced here (least privilege, complete mediation, idempotency) are public doctrine;
they do not report this design or its effectiveness.

## 1. The One-Line Framing

Execute a ticket update only when the exact bytes about to be written match a digest that a named reviewer
approved against a policy-checked plan diff, and treat the human gate as a *selective* control bought only
where its expected avoided loss per proposal exceeds its review cost.

The decision is not "should a human review agent actions". The decision is "for which of 2,072 daily
proposals is a reviewed effect digest worth 22 to 140 seconds of a $92,000-loaded reviewer, given that a
cheap non-human filter already removes 91% of the modelled harm".

Three things are being decided together, and they interact:

1. the routing rule that sends a proposal to auto-execution, to digest review, or to full review;
2. the artifact a reviewer is asked to compare, which determines whether an approval means anything;
3. the size and staffing of the review pool, which decides whether the gate is a control or a queue.

## 2. Context & Background

A service-desk agent answers employee requests about tickets: reassignment, priority change, SLA clock
adjustment, customer notification, refund initiation, and attachment handling. The agent reads the ticket,
retrieves similar resolved tickets, and proposes one concrete write per request. There are 46 tools in the
registry, grouped into four risk classes by the size of the loss a wrong write can cause.

The threat that motivates the whole design is not a malicious employee. It is text arriving from outside
the trust boundary: customer-written ticket bodies, pasted email threads, attachment OCR, and vendor
status pages. That text enters the retrieval index, so an instruction embedded in a customer message can
arrive in the agent's context with the same apparent authority as the operating procedure it sits next to.

Three facts make this hard, and each one is a design constraint rather than a complaint:

- **The reviewer sees a story, the system executes bytes.** A model can write a confident, correct-sounding
  justification while a different tool call executes. Approving the story is not approving the action.
- **The write happens after the approval.** Between the reviewer's click and the tool call, the underlying
  ticket can change, so an approval of "set priority to P2" can silently become "set priority to P2 on a
  record that has since been merged into another".
- **Most proposals are harmless.** Gating everything spends review capacity on proposals whose expected
  loss is a few cents, and the queue that creates is itself a failure mode: a backlogged reviewer approves
  faster, and approval latency becomes the customer-visible injury.

The unit's other cases cover session isolation and durable idempotent execution. This case deliberately
stays on the authorization question: who may cause which side effect, on what evidence, at what price.

## 3. The Business Question

Given one request, one retrieved evidence set, and one proposed write, which of the following do we choose,
and what does each choice cost per day at the ILLUSTRATIVE volume?

- **Auto-execute** when the write is inside a pre-approved action envelope and the effect digest recomputes.
- **Digest review** when a human must compare the plan diff against the request and the cited policy clause.
- **Full review** when the tool has no deterministic digest encoder, so a reviewer must construct the effect.
- **Quarantine** when the retrieved evidence contains instruction-shaped text, in which case the proposal is
  never shown to a reviewer as an actionable item; it is held, sanitized, and re-derived.

The operational question that follows is narrower and harder: *is the human gate worth its cost at all, and
if so on which classes?* A gate that costs more than the harm it prevents is not a control; it is a ritual
with a headcount attached. Answering this requires four numbers per class - proposal volume, review seconds,
harm per proposal, and reviewer catch rate - and one number globally, the prevalence of injected text.

## 4. Assumptions & Constraints

All values ILLUSTRATIVE, chosen to be internally consistent and round enough to audit by hand.

| Assumption | Value | Unit | Why this value |
|---|---|---|---|
| Daily requests | 4,300 | requests/day | sets the injection base rate |
| Ticket updates proposed | 2,072 | proposals/day | 48.2% of requests produce a write |
| Injected-text prevalence | 1.8% | of requests | constructed input for the teaching loss chain |
| Injection reaches retrieved context | 62% | conditional | retrieval top-k includes the poisoned chunk |
| Injection yields an action proposal | 18% | conditional | agent acts on the embedded instruction |
| Reviewer loaded cost | 92,000 | USD/year | salary plus overhead |
| Productive hours | 7.8 | hours/day | 255 working days/year |
| Reviewer cost rate | 0.012848 | USD/second | 92,000 / 255 / 7.8 / 3,600 |
| Digest-gate latency p95 | 44.6 | seconds | queue wait plus digest plus policy check |
| Horizon | 255 | days/year | working-day annualization |

Constraints that bound the design space:

- **No reviewer may see the raw tool call as the decision surface.** Reviewers compare a plan diff; the
  executed artifact is the digest of that diff. This is a hard rule because the failure it prevents is
  silent and invisible in review volume statistics.
- **Execution requires digest equality, not reviewer intent.** If the recomputed digest differs from the
  approved one, the action fails closed and re-queues. There is no override path with a shorter check.
- **Idempotency keys are minted before approval, not at execution.** A retried execution after a timeout
  must not produce a second write; the key is bound to the approved digest.
- **Three of 46 tools have no digest encoder.** Their 124 daily proposals cannot be digest-reviewed and
  must take the full-review path at a longer service time. This is a capacity fact, not a policy choice.
- **The quarantine layer may not be tuned on the same data used to report its recall.** In an implementation,
  recall would be estimated on an audited sample drawn after deployment.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per proposed action (`action_id`), not per ticket and not per conversation. A single
request can produce zero or more proposals; a single proposal produces exactly one decision record.

**Cut-off.** The decision timestamp is `proposal_ts`, the moment the agent emitted a tool-shaped proposal.
Every feature used by the router must be available strictly before `proposal_ts`. The effect digest is
computed at `proposal_ts` and re-computed at `execute_ts`; the pair is stored.

**Population.** Ticket-update proposals on the 46 registered tools, for the service-desk tenant, on
business days. Exclusions: proposals generated during incident freezes (about 1.1% of volume) because
the freeze changes routing anyway, and proposals whose tool was retired mid-window.

**Keys.** `action_id` (primary), `ticket_id`, `tool_id`, `policy_version`, `digest_sha256`, `approval_id`,
`idempotency_key`. The `approval_id` links a reviewer decision to exactly one digest, so a single approval
cannot authorize two executions.

**Units.** Volumes in proposals/day; service times in seconds; losses in USD per event; rates as proportions.
Currency is USD, undiscounted, and treated as a recurring daily loss annualized at 255 days.

**Fields.**

| Column | Type | Meaning | Available at proposal_ts |
|---|---|---|---|
| action_id | string | unique proposal identifier | yes (generated) |
| tool_id | enum | which of 46 tools would be called | yes |
| risk_class | enum A/B/C/D | loss band of the tool | yes (registry) |
| digest_sha256 | hex | hash of the normalized effect | yes (computed) |
| digest_bound | bool | encoder exists for this tool | yes |
| retrieved_ids | list | evidence chunks in context | yes |
| injected_flag | bool | quarantine classifier output | yes |
| reviewer_seconds | int | time on the plan diff | no (post-hoc) |
| execute_ts | timestamp | actual write time | no |
| realized_loss | USD | cost of an accepted wrong write | no |
| quarantined | bool | held rather than routed | yes |

**What is not available at decision time.** The realized loss of the action, the reviewer's eventual catch
or miss, whether the ticket changed between approval and execution, and whether a later reversal succeeds.
These are labels, and any model that uses them is leaking the future. The router, the quarantine classifier,
and the staffing model are all fitted or sized on features that exist before the cutoff.

### 5b Tools & Stack

- **Policy engine.** Deterministic, versioned rules over `(tool_id, risk_class, request_type, bound_flag)`.
  It returns one of four routes and a policy version string that is hashed into the digest.
- **Effect digest.** A canonical serializer over `(tool_id, object_id, field paths, new values, idempotency
  key, policy version)`, hashed with SHA-256. Canonicalization is the hard part: field ordering, number
  formatting, and null handling must be identical on both sides or the comparison is noise.
- **Plan diff renderer.** Turns the digest pre-image into a reviewer-readable field-by-field change list,
  quoting the cited policy clause. The renderer reads the same pre-image, so diff and digest cannot drift.
- **Quarantine classifier.** Instruction-shaped-text detector over retrieved chunks; emits a flag with a
  calibrated score. Recall 0.91 at a 4% false-quarantine rate (ILLUSTRATIVE).
- **Queue model.** Erlang C with class-dependent service times, used for capacity and latency, not for
  headcount arithmetic alone - the two disagree once utilization passes about 0.8.
- **Statistics.** Sensitivity analysis for constructed injection prevalence, Cohen's kappa for reviewer
  agreement, and a two-proportion sample-size formula for audit sizing.

### 5c Method Summary

1. Build the harm chain from request volume to harmful proposals, multiplying three conditional rates.
2. Attribute harmful proposals to risk classes and multiply by class loss severities to get an unmitigated
   daily loss.
3. Cost each review policy from service seconds, then convert to staffed heads with a ceiling, because
   humans are not divisible.
4. Compare policies on total daily cost (residual loss plus labour plus tooling), not on catch rate.
5. Compute the per-class break-even loss severity that makes a gate worth buying, and compare it to the
   actual severity of that class.
6. Size the audit that would detect a degradation of quarantine recall, and check it is affordable.
7. Stress the two assumptions that carry the conclusion: prevalence and the integrity of the digest binding.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** The 1.8% injection prevalence is a constructed input, not an annotation result. The
lower-prevalence sensitivity later uses 1.11%, which is 0.615 of the base value; it is a scenario rather
than a confidence interval. The case therefore treats prevalence as a parameter, not an observed fact.

**Quality.** Service times are constructed ILLUSTRATIVE inputs. The scenario uses mean 41.2 s and variance
674.2 s^2, hence SCV `674.2 / 41.2^2 = 0.397`; the Erlang C wait is adjusted by
`(1 + SCV) / 2 = 0.6985` to show the effect of non-exponential service.

**Exclusions.** Freeze windows and retired tools, as stated. Also excluded: proposals that never reached
the router because the agent refused at generation time. That exclusion biases the harmful-proposal rate
downward if refusals are correlated with injected text, which is exactly the correlation a safety team
would hope for and therefore cannot assume.

**Leakage.** Three leaks are easy to commit and were avoided: (i) using post-hoc reviewer time as a router
feature; (ii) sizing the audit on the same sample used to estimate recall; (iii) computing the digest after
execution, which would make every action appear bound.

### 5e Reproducibility & Method-Selection Ledger

| Decision | Method chosen | Alternative rejected | Why |
|---|---|---|---|
| Harm chain | product of three conditional rates | direct count of harmful proposals | no labelled harmful set exists at proposal time |
| Gate economics | expected loss per proposal vs review cost | catch-rate maximization | catch rate ignores volume and severity |
| Staffing | ceiling of hours/7.8 | fractional FTE | a 1.07 FTE demand still needs two people |
| Queue latency | Erlang C with SCV correction | M/M/1 | class mix makes SCV 0.397, not 1.0 |
| Audit sizing | two-proportion power at 80% | fixed 100-sample rule | 100 samples cannot separate 0.91 from 0.75 |
| Comparison | total daily cost | reduction percentage | percentages hide that P5 beats P4 while both cut loss |

## 6. Analysis Walkthrough

**Block 1 - the harm chain.**

```text
FORMULA
  H = R * p_inj * p_reach * p_yield
WHERE
  R        = requests per day                       (requests/day)
  p_inj    = share of requests carrying injected instruction text
  p_reach  = P(injected text enters retrieved context | injected)
  p_yield  = P(agent emits an action proposal | reached)
  H        = harmful proposals per day
CONSTRAINT
  0 <= each p <= 1; p_reach and p_yield assumed independent of each other and of R.
NUMERICAL EXAMPLE
  R = 4300, p_inj = 0.018, p_reach = 0.62, p_yield = 0.18
  step 1: 4300 * 0.018  = 77.400   injected requests/day
  step 2: 77.400 * 0.62 = 47.9878  reached context/day
  step 3: 47.9878 *0.18 = 8.6378   harmful proposals/day
  per 1000 requests: 8.6378 / 4.3 = 2.0088 harmful proposals
INTERPRETATION
  About 8.6 proposals a day carry an instruction that did not come from the operator.
  That is 0.42% of the 2,072 daily proposals - small enough that gating everything
  spends most review capacity on proposals that were never at risk.
```

Boundary where the formula breaks: if the attacker targets high-volume tickets, `p_reach` and `p_yield`
rise together, and the product understates harm. A targeted campaign that lifts both to 0.85 and 0.45
gives 4300 x 0.018 x 0.85 x 0.45 = 29.6 proposals/day, 3.4x the base case while `p_inj` is unchanged.
The formula is a prevalence model, not an adversarial model; it must be re-run with campaign parameters.

**Block 2 - attributed loss by risk class.**

```text
FORMULA
  L_none = sum over classes c of ( H * s_c * ell_c )
WHERE
  s_c    = share of harmful proposals landing in class c   (sum s_c = 1)
  ell_c  = expected loss of one accepted wrong write in c   (USD/event)
  L_none = daily loss with no control at all               (USD/day)
CONSTRAINT
  Loss must be additive across classes; no class loss may depend on another's.
NUMERICAL EXAMPLE
  class  A: 8.6378*0.30 = 2.5914  x 60   =   155.48
  class  B: 8.6378*0.25 = 2.1595  x 640  = 1,382.05
  class  C: 8.6378*0.40 = 3.4551  x 3100 = 10,710.92
  class  D: 8.6378*0.05 = 0.4319  x 9400 =  4,059.78
  total                                         = 16,308.24 USD/day
  annualised at 255 days                        = 4,158,601 USD/year
INTERPRETATION
  Class C is 40.0% of harmful proposals but 65.7% of the loss.
  Class D is 5.0% of harmful proposals and 24.9% of the loss.
  Classes A and B together are 9.4% of loss - the gate cannot be justified on them.
```

**Block 3 - review minutes and staffed heads.**

```text
FORMULA
  M     = sum over classes ( n_c * t_c ) / 60          (reviewer-minutes/day)
  FTE   = M / (H_shift * 60)                           (person-equivalents)
  staff = ceil(FTE)
WHERE
  n_c   = proposals routed to human review in class c  (proposals/day)
  t_c   = mean service seconds for one review in c     (seconds)
  H_shift = productive hours per reviewer-day = 7.8
CONSTRAINT
  t_c represents the review task only; no context switching or queue time is included.
NUMERICAL EXAMPLE
  full review of all proposals:
    1180*22 + 660*55 + 208*95 + 24*140 = 85,380 seconds
    85,380 / 60   = 1,423.0 minutes
    1,423.0 / 60  = 23.717 hours
    23.717 / 7.8  = 3.041 FTE  -> ceil = 4 reviewers
  digest review of the 1,948 bound proposals plus 124 full reviews at 55 s:
    1180*22 + 660*38 + 208*70 + 24*110 + 124*55 = 72,332 seconds
    72,332 / 3600 = 20.092 hours
    20.092 / 7.8  = 2.576 FTE  -> ceil = 3 reviewers
INTERPRETATION
  The digest path saves 6.3 seconds per review on average but 13,048 seconds a day
  in total, and it saves a whole head because 2.576 ceilings to 3 while 3.041
  ceilings to 4. The staffing step, not the seconds, is the real saving.
```

Boundary where ceiling logic misleads: at 23.4 hours the same policy needs 3 reviewers; a 1.4% dip in
volume moves the answer by one salary. Conversely a demand of 3.99 FTE is served by 4 people at 99.7%
utilisation, which is not a plan - it is a queue. The ceiling is therefore a floor on cost, not a
capacity check; Section 6 Block 5 supplies the capacity check.

**Block 4 - when is a gate worth buying at all.**

```text
FORMULA
  value_c = q_c * m_c * (1 - rho) / (t_c * c_r)
  ell*_c  = t_c * c_r / ( q_c * m_c * (1 - rho) )
WHERE
  q_c   = harm per proposal in class c = H * s_c / n_c        (events/proposal)
  m_c   = reviewer catch probability for class c              (0..1)
  rho   = residual leakage of the cheaper quarantine layer     (0..1)
  c_r   = reviewer cost per second = 0.012848                 (USD/s)
  value_c = avoided loss per dollar of review, dimensionless
  ell*_c  = break-even loss severity for class c              (USD/event)
CONSTRAINT
  value_c > 1 means the gate pays; q_c must be defined at proposal grain.
NUMERICAL EXAMPLE
  class A: q = 8.6378*0.30/1180 = 0.002196, m = 0.90, rho = 0.09
           avoided = 0.002196*60*0.90*0.91 = 0.1079 USD/proposal
           cost    = 22*0.012848          = 0.2827 USD/proposal
           value   = 0.38 -> the A gate destroys value
           ell*    = 0.2827/(0.002196*0.90*0.91) = 157 USD
  class C: q = 8.6378*0.40/208  = 0.016611, m = 0.9667, cost = 95*0.012848 = 1.221
           avoided = 0.016611*3100*0.9667*0.91 = 45.32 USD/proposal
           value   = 45.32/1.221 = 37.1
  class D: q = 8.6378*0.05/24   = 0.017995, m = 0.50,   cost = 110*0.012848 = 1.413
           avoided = 0.017995*9400*0.50*0.91 = 76.99 USD/proposal
           value   = 76.99/1.413 = 54.5
INTERPRETATION
  Classes C and D repay review 37 and 54 times over; classes A and B return 0.38
  and 0.35. A single routing rule for all proposals is therefore wrong in both
  directions: it over-reviews A and B and under-invests in the 24 class-D cases.
```

Boundary: the formula assumes catch probability is independent of severity. In practice reviewers catch
more on severe classes because they slow down, which raises `m_D` and makes D even more attractive; the
opposite happens under queue pressure, when `m_D` falls first because D reviews are the longest. This is
why Section 12 treats queue collapse as a security mechanism and not merely an operations metric.

**Block 5 - queue wait and the capacity check.**

```text
FORMULA
  a   = lambda / mu                     (offered load, erlangs)
  rho = a / c                           (utilisation with c reviewers)
  C   = ErlangC(a, c)                   (probability all c are busy)
  Wq  = C / (c*mu - lambda)             (mean wait, same time unit as lambda)
  Wq' = Wq * (1 + SCV) / 2              (SCV correction for non-exponential service)
WHERE
  lambda = proposals/day / open seconds        (proposals/s)
  mu     = 1 / mean service time               (1/s)
  c      = number of reviewers on the queue
CONSTRAINT
  Requires c*mu > lambda; otherwise the queue is unstable and Wq is meaningless.
NUMERICAL EXAMPLE
  all proposals to a 4-reviewer digest pool:
    lambda = 2072 / (8*3600) = 0.07194 proposals/s
    Ts     = 72,332/2072     = 34.91 s  -> mu = 0.02865
    a      = 0.07194/0.02865 = 2.5115 erlangs
    3 reviewers: rho = 0.837, C = 0.7087, Wq = 50.6 s, corrected 35.3 s
    4 reviewers: rho = 0.628, C = 0.3238, Wq =  7.6 s, corrected  5.3 s
  full review pool, Ts = 41.21 s:
    a = 2.9646; 4 reviewers rho = 0.741, Wq = 19.7 s; 5 reviewers Wq = 4.6 s
INTERPRETATION
  Cutting from four reviewers to three raises mean wait from 7.6 s to 50.6 s
  and the 95th percentile from 43.8 s to 189.5 s. Reviewers under that pressure
  shorten reviews, which is exactly the mechanism that lowers catch probability.
```

Boundary: at `c*mu = lambda` the formula diverges; a pool at 98.8% utilisation (three reviewers on the
full-review workload) has a computed Wq of 1,137.6 s, which no reviewer survives, so the model's answer is
"not staffable" rather than "slow". The digest path is what moves the pool from unstaffable to comfortable.

**Block 6 - the approval-to-execution collision.**

```text
FORMULA
  p_win   = 1 - exp( - e_d * w / D )
  collide = p_win * N_prop
WHERE
  e_d     = edits per ticket per day          (edits/ticket/day)
  w       = approval validity window          (seconds)
  D       = open seconds per day              (28,800 s)
  N_prop  = proposals per day
CONSTRAINT
  Edit arrivals per ticket are Poisson; a ticket edited twice inside w counts once.
NUMERICAL EXAMPLE
  edits/day = 1180*0.06 + 660*0.05 + 208*0.04 + 24*0.02 = 112.6
  e_d       = 112.6/4300 = 0.02619 edits/ticket/day
  p_win     = 1 - exp(-0.02619 * 120/28800) = 1.091e-4
  collide   = 1.091e-4 * 2072 = 0.2261 events/day
  material  = 0.2261 * 0.10 (material share) = 0.0226/day
  annual at ell = 3100: 0.0226*3100*255 = 17,870 USD
  annual if 94.02% are bound to a digest: 17,870*(124/2072) = 1,069 USD
INTERPRETATION
  Without binding, stale approvals leak about 1.1e-4 of proposals into a wrong
  write. With digest equality at execute time, the window collapses to the 124
  unbound proposals a day, and the annual figure falls by 94%.
```

Counterexample: the collision model assumes the reviewer approves the version they read. If the reviewer
approves a *summary* produced before the ticket edit, the collision probability is not a window property but
a documentation property, and no window length fixes it. That is the failure Section 12 calls "approving
the story", and it is why the digest, not the window, is the load-bearing control.

**Block 7 - where to put the quarantine operating point.**

The classifier has a threshold, and moving it trades recall against false quarantine. Five constructed
ILLUSTRATIVE operating points give the frontier below. The question is not "maximise recall"; it is
"minimise total cost, given what else is in the stack".

```text
FORMULA
  total_q(r)   = L_none * (1 - r) + R * fp(r) * w * c_r + P_plat
  total_P6(r)  = g(r) + labour + R * fp(r) * w * c_r + P_plat
  g(r)         = gate_residual * (1 - r) / (1 - r_0)
WHERE
  r      = quarantine recall at the chosen threshold
  fp(r)  = false-quarantine rate at that threshold
  w      = rework seconds per false quarantine = 90
  P_plat = platform amortisation = 86.27 USD/day
  g(r)   = C/D-gate residual, rescaled from the base point r_0 = 0.91
CONSTRAINT
  0 <= fp <= 1; the frontier must be monotone (higher recall costs more false positives).
NUMERICAL EXAMPLE
   r     fp     quarantines   q-cost   P3 total   P6 total
  0.85  0.020        86      185.71   2,631.95   1,323.35
  0.91  0.040       172      285.17   1,752.90   1,187.36
  0.94  0.075       322.5    459.18   1,437.68   1,243.65
  0.96  0.110       473      633.21   1,285.54   1,339.19
  0.98  0.190       817    1,030.98   1,357.15   1,658.49
  arithmetic for r = 0.94: 4300*0.075 = 322.5 quarantines
                          322.5*90*0.012848 = 372.91; +86.27 = 459.18
                          16308.24*0.06 = 978.49 residual; total 1,437.67
INTERPRETATION
  Quarantine alone is cheapest at r = 0.96 (1,285.54), because it pays for
  false positives directly and buys residual loss one-for-one.
  In the composite the optimum moves back to r = 0.91 (1,187.36), because the
  C/D gate already removes a fixed fraction of whatever leaks, so the marginal
  dollar of recall is worth less than it was in isolation.
```

Boundary: past r = 0.96 the frontier turns up - each further point of recall costs about 4.3 points of
false quarantine, and at 19% the rework queue itself becomes the bottleneck (817 cases a day against a
400-a-day rework capacity, so the layer backlogs and effectively stops working). A classifier operated
beyond its frontier does not fail loudly; it fails by accumulating a hold queue that someone eventually
releases in bulk.

**Block 8 - the marginal head as the unit of decision.**

Because staffing is a ceiling, almost every policy change should be priced in heads, not seconds.

```text
FORMULA
  one_head = (salary + overhead) / working_days
  worth_it  = delta_residual > one_head
WHERE
  one_head = 92,000 / 255 = 360.78 USD/day  (the marginal cost of the next step)
CONSTRAINT
  Applies only when the change crosses an integer boundary; inside a step the
  marginal cost of a second is zero and the rule must not be applied.
NUMERICAL EXAMPLE
  base load 24,020 s = 6.672 h  -> FTE 0.855  -> 1 head
  class-D escalation to 360 s: 24,020 + 24*250 = 30,020 s
                                = 8.339 h -> FTE 1.069 -> 2 heads
  delta heads = 1, delta cost = 360.78/day
  delta residual = -182.69/day  (class-D misses fall to zero)
  net = -182.69 + 360.78 = +178.09 -> reject
  alternative: external review at 18 USD each, 24 a day = 432 USD
  net = -182.69 + 432 = +249.31    -> reject by a wider margin
INTERPRETATION
  Both routes to full class-D coverage cost more than the loss they remove.
  The gate's 110-second version captures the cheap half of the class and the
  remaining half is bought by quarantine, not by more humans.
```

Boundary: this rule reverses if class-D volume grows without the severity falling. If class D rises to
60 proposals a day, the escalated version needs 24,020 + 60*250 = 39,020 s = 10.84 h = 1.39 FTE, which is
still two heads, while the half-caught residual grows to 8.6378*0.125*9,400*0.50*0.09 = 456.72 USD/day.
Removing that residual costs 360.78, so escalation now returns 95.94 a day and the decision flips. The
break-even share of harmful proposals landing in class D is
360.78 / (8.6378 * 9,400 * 0.50 * 0.09) = 0.0987; below that, do not escalate, above it, do. The
decision is a function of volume, so it must be recomputed when volume changes rather than fixed once.

### 6b Cross-checks on the walkthrough

| Quantity | First route | Second route | Agreement |
|---|---|---|---|
| harmful proposals/day | 4300 x 0.018 x 0.62 x 0.18 = 8.6378 | 47.9878 x 0.18 with 47.9878 from step 2 | exact |
| review hours, full | seconds/3600 = 23.717 | 1423.0 min / 60 | exact |
| reviewer cost/second | 92,000/255/7.8/3,600 | 0.7709 per minute / 60 | 0.012848 both |
| annual unmitigated loss | 16,308.24 x 255 | 155.48+1382.05+10710.92+4059.78 then x255 | 4,158,601 both |
| queue utilisation | a/c = 2.5115/4 | lambda/(c*mu) = 0.07194/(4x0.02865) | 0.628 both |

## 7. The Finding

The gate is not the control that carries this system. Three findings, in order of how much they change
the decision:

1. **A cheap non-human filter dominates.** Quarantining proposals whose retrieved evidence contains
   instruction-shaped text, with recall 0.91 and a 4% false-quarantine rate, removes 91% of modelled harm
   for 285.17 USD/day. That is 1,467.74 residual loss plus tooling and rework, against 16,308.24 with no
   control. No review staffing at any affordable level matches that ratio: the best review-only design
   reaches 3,810.91 USD/day, and it is 13 times more expensive than quarantine alone for a worse outcome.
2. **Where the gate does pay, it pays enormously.** Classes C and D return 37 and 54 dollars of avoided
   loss per dollar of review, because a single wrong write costs 3,100 and 9,400 USD while the review
   takes 95 and 110 seconds. Classes A and B return 0.38 and 0.35 - they destroy value, and gating them is
   the most expensive mistake in the whole design: it consumes three reviewers to protect a class whose
   break-even severity is 157 USD against an actual severity of 60.
3. **The binding constraint on class D is the staffing step, not the minutes.** Twenty-four proposals a day
   at 360 seconds would catch all of them, but it needs 1.069 FTE, which ceilings to two reviewers and
   costs 232,000 USD/year. The 110-second version catches half of them for one reviewer. The design that
   ships is therefore deliberately half-covered on the most severe class, and that gap is documented as a
   known residual rather than presented as a solved problem.

Composite: quarantine plus a gate restricted to classes C and D costs 1,187.36 USD/day all-in, a 92.7%
reduction against no control, and its annual figure is 302,777 USD against 4,158,601. The full gate on all
classes costs 4,148.61 USD/day, or 3.5 times more, for a *worse* residual because it drains the reviewers
who are the only defence on C and D.

## 8. System/Solution Design

**Component 1 - intent binding.** The request is reduced to an intent record before any retrieval happens:
`(requester, ticket, requested_change, allowed_tools, policy_version)`. The policy engine returns the route
from the intent alone. Retrieved text may inform the *content* of a proposal but may not widen the route,
which is what stops an injected instruction from promoting a class-B action into an auto-executed class-A one.

**Component 2 - effect digest.** A canonical pre-image is serialized with fixed field order, decimal
formatting, and null policy, then hashed. The digest is stored with the proposal and recomputed at execute
time; inequality is a hard failure. Coverage is 43 of 46 tools, 1,948 of 2,072 proposals, 94.02% of volume.
The remaining 124 proposals come from three tools with free-text bodies whose canonical form is not
decidable, and they route to full review at 55 seconds mean.

**Component 3 - plan-diff review surface.** The reviewer sees field-by-field changes, the diff, the cited
policy clause, and the digest fingerprint in short form. The approval binds the full digest, not the
fingerprint; the fingerprint exists so a human can compare two screens, not so the system can compare them.
This separation matters because a shortened hash is a display convenience and must never be the execution key.

**Component 4 - quarantine layer.** Retrieved chunks are scored; any proposal whose evidence contains an
instruction-shaped passage is held. Held proposals are re-derived on sanitized evidence, or escalated if
re-derivation fails. The layer costs 285.17 USD/day: 172 false quarantines x 90 s of rework at the reviewer
rate (198.89 USD) plus 22,000 USD/year of platform amortized over 255 days (86.27 USD).

**Component 5 - audit sampler.** Two independent audits would run continuously. One would estimate
quarantine recall on held proposals, because recall is the number the whole design rests on. The other would
estimate reviewer agreement on approvals, because a reviewer who always approves is a rubber stamp with a
salary. In an ILLUSTRATIVE 240-decision agreement exercise, 204 decisions agree; marginal totals of 180/240
and 172/240 yield expected agreement 0.6083 and kappa 0.617, with standard error 0.0588 and a 95% interval
of 0.502 to 0.733. This illustrates a substantial but not excellent agreement result, not a measurement.

**Component 6 - idempotent sink.** Each approved digest carries one idempotency key minted before approval.
A retry after a timeout re-sends the same key, so the sink collapses duplicates. The key is scoped to the
digest, so a re-proposal after a failed execution gets a fresh key and cannot be mistaken for a duplicate.

| Component | Control it provides | Failure it does not cover |
|---|---|---|
| Intent binding | injected text cannot widen the route | a legitimately routed but wrong change |
| Effect digest | approved bytes equal executed bytes | the approval was for the wrong thing |
| Plan-diff surface | reviewer can see the actual change | reviewer approves without reading |
| Quarantine | 91% of instruction-carrying proposals held | paraphrased instructions the classifier misses |
| Audit sampler | detects recall and agreement drift | drift faster than the sampling interval |
| Idempotent sink | no duplicate side effects on retry | a wrong side effect executed once, correctly |

## 9. Evaluation & Validation

**Primary metric - executed-write authorization defect rate.**

- Numerator: executions whose recomputed digest differed from the approved digest, plus executions with no
  matching `approval_id`, plus executions on a ticket whose version changed after approval.
- Denominator: executions in the window. Population: all executions, including auto-executed ones, because
  an auto-executed action with no approval record is a defect of the same family.
- Window: 30 consecutive days, reported weekly with a rolling 7-day view for early warning.
- Comparator: an ILLUSTRATIVE pre-gate counterfactual in which agent proposals have no digest binding.
  The constructed scenario uses a 0.4% silent-mismatch rate to motivate testing the binding mechanism.
- Guardrails: (i) p95 gate latency under 90 s; (ii) reviewer queue utilisation under 0.80; (iii) quarantine
  false-positive rate under 6%; (iv) approved-without-diff-view rate under 1%, to be computed from surface
  telemetry if the design is executed.
- Uncertainty: the defect rate is rare, so an implementation would report an exact one-sided upper bound
  when zero defects are observed. With zero in 2,072 executions, the illustrative bound is
  `1 - 0.05^(1/2072) = 0.144%`; that would describe the bound, not safety.
- Status: **not executed.** No measurement exists; every number in this case is ILLUSTRATIVE arithmetic.

**Secondary metric - avoided loss per review dollar, by class.** Computed weekly from the ledger in Block 4.
The guardrail is that no class may be gated for two consecutive weeks while its `value_c` is below 1.0.
This is the mechanism that would have caught the A/B overspend automatically rather than after a review.

**Tertiary metric - quarantine recall.** The number that decides everything else. Detecting a drop from
0.91 to 0.75 at 80% power and 5% one-sided alpha needs about 615 audited held proposals; at 43.67 held
proposals a day that is 14.1 days of audit. A 100-sample weekly audit cannot distinguish those rates and
must not be used to certify the layer.

**Validation of the arithmetic.** Every block was recomputed through a second route where one existed
(Section 6b). The two routes that matter most: staffed heads from seconds and from minutes, and annual
loss from the class table and from the total. Both agree exactly.

## 10. Action Taken

The decision record adopts the composite policy and rejects three alternatives with arithmetic rather than
opinion.

**Adopted.** Quarantine plus a digest-review gate restricted to classes C and D, with 124 unbound proposals
routed to full review, one reviewer, and both audits funded. All-in 1,187.36 USD/day: 353.17 residual loss,
549.02 labour, 285.17 quarantine.

**Rejected: full gate on all classes.** 4,148.61 USD/day and a worse residual (2,517.24) than the adopted
policy, because four reviewers at 74% utilisation shorten reviews and the class-D catch probability falls
from 0.50 toward 0.35. Cost is 3.5 times higher for an outcome 7 times worse. The mechanism is capacity
competition between classes, not a modelling artifact.

**Rejected: class-D escalation to 360-second full manual review.** Catching all 24 class-D proposals raises
labour from 24,020 to 30,020 seconds; 30,020 seconds is 8.339 hours, or 1.069 FTE, which ceilings to two
reviewers. Labour becomes 909.80 and the total becomes 1,365.45 - worse than 1,187.36 despite zero class-D
residual loss. The staffing step eats the benefit. This is the single most useful negative result in the case.

**Rejected: buying class-D reviews from an external pool at 18 USD each.** 24 proposals x 18 = 432 USD/day
against a residual class-D loss after the 110-second gate of 4,059.78 x 0.50 x 0.09 = 182.69 USD/day.
The purchase costs 2.4 times what it saves. The gate already captures the cheap half of the class.

**Funded.** Two audits, not one. The recall audit at 615 held proposals per cycle, and the agreement audit at
240 sampled decisions per cycle, which yields a kappa interval of roughly 0.50 to 0.73 - wide, and reported
as wide. A tighter interval would need about 900 decisions and is deferred until the review pool grows.

## 11. Outcome & Results

All values ILLUSTRATIVE, in USD/day unless stated, annualized at 255 days.

| Policy | Residual loss | Daily cost | Total/day | Annual | Reduction vs none |
|---|---|---|---|---|---|
| P0 no control | 16,308.24 | 0.00 | 16,308.24 | 4,158,602 | baseline |
| P1 full gate, all classes | 2,517.24 | 1,631.37 | 4,148.61 | 1,057,896 | 74.6% |
| P2 digest gate, all classes | 2,540.32 | 1,270.59 | 3,810.91 | 971,782 | 76.6% |
| P3 quarantine only | 1,467.74 | 285.17 | 1,752.91 | 446,992 | 89.3% |
| P4 quarantine plus full gate | 226.55 | 1,916.54 | 2,143.09 | 546,489 | 86.9% |
| P5 quarantine plus digest gate | 228.63 | 1,555.76 | 1,784.39 | 455,018 | 89.1% |
| P6 adopted: quarantine plus C/D gate | 353.17 | 834.19 | 1,187.36 | 302,777 | 92.7% |

Three results deserve comment because they invert the intuitive ranking.

**Adding the full gate to quarantine makes things worse, not better.** P3 is 1,752.91 and P4 is 2,143.09.
The gate removes 1,241.19 of residual loss and costs 1,631.37 in labour, so it is a net loss of 390.18 a day.
P5 improves on P4 only because the digest path costs one fewer head, not because it catches more.

**P6 is not a compromise between P3 and P5; it is a different shape.** It gives up 182.69 of class-D
residual to save 721.57 of labour by reviewing only 232 proposals a day instead of 1,948. The exchange rate
is 3.95 dollars saved per dollar of residual given up, which is why it wins.

**The 124 unbound proposals carry a disproportionate share of the audit risk.** They are 6.0% of volume but
100% of the approvals that cannot be digest-bound, and their annual stale-approval exposure is 17,870 USD
if they were mis-bound, falling to 1,069 USD under binding. Closing the digest gap on three tools is
therefore worth more than any further review-rule tuning, and is the top backlog item.

**Sensitivity - prevalence.** At a lower ILLUSTRATIVE prevalence of 1.11% instead of 1.8%, unmitigated
loss scales by `1.11 / 1.8 = 0.615`, from 16,308.24 to 10,029 USD/day, and P6's total falls to about
1,024 USD/day with residual loss nearly unchanged in relative terms. Breakeven for quarantine is a
prevalence of 0.035% and for the full gate 0.186%, so both remain above the lower scenario. The conclusion
is directionally robust across this constructed sensitivity range, not a sample-based confidence claim.

**Sensitivity - a missed adversarial path.** Suppose quarantine is effective on text channels but blind to
indirect instructions arriving through tool outputs, and class D harm is entirely of that kind. Class-D
residual becomes 4,059.78 instead of 182.69, P6's loss becomes 4,230.26, its total 5,064.45, and the annual
figure 1,291,435. The policy still beats P0, but 4.25 times less impressively. One unmitigated channel
dominates the composite, which is the argument for auditing per channel rather than in aggregate.

## 12. Failure Modes & What Went Wrong

Seven distinct mechanisms, each with what you would observe and what you would not.

**F1 - Approving the story, not the bytes.** The reviewer reads a fluent justification and clicks approve;
the tool call that executes was generated from a different plan. *Competing causes:* (a) the diff renderer
and the executor read different pre-images; (b) the reviewer never opened the diff; (c) a post-approval
regeneration replaced the plan but reused the approval. *Discriminating check:* log the digest at approval
and at execution for every action; a nonzero mismatch count with low diff-view telemetry points to (b), a
mismatch count with high diff-view telemetry points to (a) or (c). *Correction:* execution fails closed on
digest inequality, approvals bind digests not fingerprints.

**F2 - Gating a class that cannot repay it.** The gate runs on class A for a quarter. Review capacity is
consumed, class-D reviews get shorter, and the class-D catch probability falls. *Competing causes:* (a) the
routing rule is uniform by policy inheritance; (b) the per-class economics were computed at launch and never
refreshed; (c) class-A volume grew. *Discriminating check:* recompute `value_c` weekly; if `value_A` is 0.38
while `value_D` is 54 and both are gated, the defect is the rule, not the volume. *Correction:* route by
`value_c > 1`, and put a two-week consecutive-breach guardrail on any gated class.

**F3 - Quarantine recall silently decaying.** New paraphrase styles lower recall from 0.91 to about 0.75.
Residual loss rises from 1,467.74 to 4,077 as leakage goes from 9% to 25%. *Competing causes:* (a) the
classifier is stale against new instruction phrasings; (b) false positives rose, so the threshold was relaxed
to reduce rework; (c) held proposals are being released manually without re-derivation. *Discriminating
check:* the recall audit on 615 held proposals separates 0.91 from 0.75 at 80% power; a rework-versus-recall
plot identifies (b) as a threshold move rather than a drift. *Correction:* audit-driven threshold control,
with rework cost as an explicit term rather than an implicit pressure.

**F4 - Stale approval executing on a changed record.** 0.2261 collisions a day, 0.0226 material, 17,870 USD a
year if unbound. *Competing causes:* (a) the approval window is too long; (b) the record version is not
checked at execution; (c) concurrent edits are common on a small set of tickets rather than spread evenly.
*Discriminating check:* plot collisions per ticket; heavy concentration on a few tickets means (c) and points
to serializing edits per ticket, while uniform collisions mean (a). *Correction:* version token in the digest
plus per-ticket execution leases.

**F5 - Unbound proposals auto-approved by omission.** A tool without a digest encoder is added to the
auto-execute list because "the digest is not applicable". The 124 daily proposals become 400 with no review.
*Competing causes:* (a) the route defaults to auto when `digest_bound` is false; (b) the encoder exists but
throws, and the error path is treated as success; (c) the tool registry marks the tool as low risk, and low
risk is conflated with digestible. *Discriminating check:* assert that the count of executions with a null
digest is exactly zero; any nonzero value identifies the default-branch defect. *Correction:* fail closed on
absent digest, with a registry test that every tool either has an encoder or is explicitly full-review only.

**F6 - Review pool collapse under queue pressure.** Three reviewers on the digest workload run at 0.837
utilisation, 50.6 s mean wait, 189.5 s p95; reviewers compensate by shortening reviews. *Competing causes:*
(a) volume rose; (b) a class was added to the pool without adding heads; (c) service times rose because the
diff renderer changed. *Discriminating check:* compare mean reviewer seconds before and after the volume
change; stable seconds with rising wait is (a) or (b), falling seconds with rising wait is (c) plus coping.
*Correction:* utilisation guardrail at 0.80, with the pool sized by Erlang C rather than by hours.

**F7 - Severity estimate wrong for the extreme class.** Class D is modelled at 9,400 USD per event from a
small number of precedents. If the true mean is 40,000, the class-D gate's advantage grows and the case for
escalating review strengthens; if the true mean is 900, the 110-second review is over-investment. *Competing
causes:* (a) the estimate is a mean over a heavy tail and the mean is unstable; (b) the class is misdefined
and includes two populations with different severities. *Discriminating check:* plot the loss distribution
inside class D; a single dominant precedent means (a) and calls for a median-plus-tail treatment rather than
a mean. *Correction:* report `ell_D` as a range with the tail mass explicit, and re-run the routing rule at
both ends.

## 13. Transferable Lesson

**Bind the artifact, not the narrative.** Every design that lets a human approve a description while a
machine executes a different object has the same defect, whether the object is a database row, a payment, a
deployment manifest, or a surgical order. The fix is structural: hash the exact effect, show the diff
derived from the same pre-image, and refuse to execute on inequality. Narratives are for humans; digests
are for execution; never let one stand in for the other.

**Buy controls where the exchange rate is good, and say so in dollars.** The instinct to gate everything is
a refusal to do the arithmetic. Here the arithmetic says a 285-dollar filter beats a 1,600-dollar human
gate, and that a gate on the two cheap classes destroys value while starving the two expensive ones. The
same reasoning transfers to any human-in-the-loop design: compute avoided loss per proposal, divide by the
review cost, and route on the ratio.

**Integer staffing is a design constraint, not an accounting detail.** A policy that needs 1.069 FTE costs
two heads. That single fact reversed the decision on the most severe class, and it will reverse decisions in
every organisation that sizes review pools by dividing minutes by minutes.

**A control you cannot audit is a belief.** Quarantine recall is the number the whole design rests on, and
615 held proposals - 14 days - is the smallest sample that can tell 0.91 from 0.75. Anything cheaper is
reassurance.

## 14. Connections & Cross-References

- **Session isolation and lifecycle** (`04_agent_sessions_state_isolation_and_lifecycle.md`): the intent
  record here is per-session state; a session that leaks state across requests defeats route binding.
- **Durable workflows and idempotent execution** (`05_durable_workflows_and_idempotent_execution.md`): the
  idempotency key minted before approval is the join point between this case and replay-safe execution.
- **Agent boundaries and request ownership** (`01_agent_boundaries_and_request_ownership.md`): ownership of
  the request is what makes "who may authorize this" answerable at all.
- **Claims operations release case** (`02_claims_operations_agent.md`): the same economics appear there as
  canary length versus guardrail power; both cases conclude that sample size, not intent, sets the control.
- **External doctrine.** Complete mediation and least privilege explain why the route is fixed before
  retrieval; they do not supply any number in this manuscript, and none is attributed to them.

### Interview pack

**Q1. Why is a reviewer approving a natural-language summary not an authorization?**

- Answer: Because the executed object is the tool call, not the summary. Approval must bind the bytes that
  will be written; a summary is a rendering that can drift from the pre-image.
- Reasoning: Two independent renderings of the same plan can differ in field ordering or number formatting,
  and a regenerated plan can replace the approved one. Digest equality at execute time is the only check
  that survives both.
- Follow-up: Where would you store the approval so a retry cannot reuse it for a different payload?
- Red flag: "The reviewer is experienced, so the summary is enough."
- Scoring: 0 summary is fine; 1 mentions hashing but not execute-time recomputation; 2 requires digest
  equality at execution and binds approval to the digest.

**Q2. Compute the daily harmful-proposal count from 4,300 requests, 1.8% injection, 62% reach, 18% yield.**

- Answer: 4300 x 0.018 = 77.4; x 0.62 = 47.9878; x 0.18 = 8.6378 proposals/day.
- Reasoning: The three conditionals multiply because each is a filter on the previous population; the
  result is 0.42% of the 2,072 daily proposals.
- Follow-up: What changes if reach and yield are positively correlated?
- Red flag: multiplying 4,300 by 0.018 x 0.62 x 0.18 in a different order but reporting 8.6 without units.
- Scoring: 0 wrong product; 1 correct number, no units or no per-1000 framing; 2 correct with the
  independence caveat.

**Q3. Full review of 2,072 proposals needs 85,380 reviewer-seconds. How many people?**

- Answer: 85,380/60 = 1,423.0 minutes; /60 = 23.717 hours; /7.8 = 3.041 FTE; ceil = 4 reviewers.
- Reasoning: Humans are not divisible, so the ceiling is the cost. A 3.041 demand cannot be served by
  three people without overtime that is really a queue.
- Follow-up: What volume dip would drop it to three heads, and is that a plan?
- Red flag: answering 3.04 people.
- Scoring: 0 wrong arithmetic; 1 correct FTE but no ceiling; 2 correct heads plus the observation that
  the ceiling dominates.

**Q4. Class A returns 0.38 dollars of avoided loss per review dollar. What do you do, and why is the
number not 1.0?**

- Answer: Stop gating class A. The number is below 1 because harm per proposal is 0.002196 events and loss
  per event is 60 USD, so expected harm is 0.13 USD while the 22-second review costs 0.28 USD.
- Reasoning: The gate is a purchase; below 1.0 it is a purchase at a loss. The break-even severity is
  157 USD against an actual 60 USD, a 2.6x gap.
- Follow-up: What would have to change for class A to become worth gating?
- Red flag: "Review everything, cheap classes too, for safety culture."
- Scoring: 0 no decision; 1 drops A without the arithmetic; 2 drops A and states break-even severity.

**Q5. Quarantine costs 285 USD/day and removes 14,840 of 16,308 daily loss. The full gate costs 1,631 and
removes 13,791. Which do you buy first?**

- Answer: Quarantine, unambiguously - 52 dollars of avoided loss per dollar versus 8.5.
- Reasoning: Compare ratios, not absolute removals. Quarantine also frees the reviewers that the gate
  would consume, so the two are not independent purchases.
- Follow-up: When would you buy the gate second? Only on classes whose per-proposal ratio exceeds 1.0.
- Red flag: choosing the gate because it is "the human control".
- Scoring: 0 picks the gate; 1 picks quarantine without ratios; 2 gives both ratios and the sequencing.

**Q6. Why does adding the full gate to quarantine make the total worse?**

- Answer: P3 is 1,752.91 and P4 is 2,143.09. The gate removes 1,241.19 of residual and costs 1,631.37.
- Reasoning: At 91% quarantine recall, only 9% of harm remains, so the gate's marginal market has shrunk
  by a factor of 11 while its price is unchanged.
- Follow-up: What recall would make the gate break even on top of quarantine?
- Red flag: assuming controls are additive in benefit.
- Scoring: 0 "more controls are better"; 1 spots the residual shrink; 2 quantifies the marginal exchange.

**Q7. Three reviewers at 0.837 utilisation show 50.6 s mean wait. Add a fourth and wait falls to 7.6 s.
Explain the non-linearity.**

- Answer: Erlang C is convex in utilisation: at rho 0.837 the all-busy probability is 0.7087, at 0.628 it
  is 0.3238, and wait is that probability divided by spare capacity (c*mu - lambda).
- Reasoning: A 25% utilisation drop cuts the all-busy probability by more than half and doubles spare
  capacity, so wait falls by roughly 6.7x.
- Follow-up: What does the p95 do, and why does that matter for review quality?
- Red flag: "wait scales linearly with load."
- Scoring: 0 linear reasoning; 1 quotes the numbers; 2 explains both terms in the denominator.

**Q8. Your kappa is 0.617 with a 95% interval of 0.502 to 0.733. Is the review reliable?**

- Answer: Substantial but not excellent, and the interval excludes the "moderate" band at the low end only
  barely. It is not evidence of a rubber stamp, and it is not evidence of consistency either.
- Reasoning: Raw agreement is 0.85, which flatters; expected agreement at the marginals is 0.6083, so the
  chance-corrected figure is 0.617. With 240 decisions the standard error is 0.0588.
- Follow-up: How many decisions to halve the interval width?
- Red flag: citing 0.85 as reliability.
- Scoring: 0 cites agreement; 1 gives kappa; 2 gives kappa, interval, and the sample-size implication.

**Q9. A tool has no digest encoder. What must the router do, and what breaks if it does not?**

- Answer: Route it to full review and never auto-execute; assert that executions with a null digest are
  exactly zero. If the default branch is auto, 124 proposals a day become unbound, and unbound approvals
  carry 17,870 USD/year of stale-execution exposure.
- Reasoning: Absence of a control is not evidence that the control is unnecessary; the safe default for a
  missing digest is the longest review path, not the shortest.
- Follow-up: How would you detect the defect before an incident?
- Red flag: "the tool is low risk so the digest does not matter."
- Scoring: 0 auto-executes; 1 routes to review without the assertion; 2 states the fail-closed test.

**Q10. Estimate the stale-approval collision rate for a 120-second window with 112.6 edits/day over
28,800 open seconds and 4,300 tickets.**

- Answer: 112.6/4300 = 0.02619 edits per ticket per day; 1 - exp(-0.02619 x 120/28800) = 1.091e-4;
  times 2,072 proposals = 0.2261 collisions/day, 0.0226 material at a 10% material share.
- Reasoning: The Poisson window formula is the right model because edit arrivals are independent and rare
  per ticket; a linear approximation gives 1.091e-4 too, so the exponential is a convenience here.
- Follow-up: Why does a version token beat a shorter window?
- Red flag: using 112.6/2072 as the per-ticket rate.
- Scoring: 0 wrong denominator; 1 correct probability without the proposal volume; 2 full chain.

**Q11. A reviewer approves 240 decisions and you audit 240. Design the agreement study.**

- Answer: Sample decisions across classes and reviewers, have a second reviewer independently label
  approve/decline, then compute kappa with marginals. 240 gives a standard error near 0.059, so the
  interval spans about 0.23 - enough to separate 0.6 from 0.4, not enough to certify 0.75.
- Reasoning: Kappa corrects for chance agreement, which raw agreement inflates when one label dominates;
  here 158 both-approve and 46 both-decline give 0.85 raw but 0.617 corrected.
- Follow-up: What confound does reviewer-level clustering introduce?
- Red flag: reporting raw agreement and calling it reliability.
- Scoring: 0 raw agreement; 1 kappa without interval; 2 kappa, interval, and the clustering caveat.

**Q12. Quarantine recall is 0.91 on the launch sample. How many held proposals to detect 0.75?**

- Answer: About 615 at 80% power and 5% one-sided alpha; at 43.67 held proposals/day that is 14.1 days.
- Reasoning: The two rates are far apart in relative terms (0.91 to 0.75) but the absolute counts are large,
  so the sample is hundreds, not tens. A 100-sample weekly audit cannot separate them.
- Follow-up: What would you do while the audit runs?
- Red flag: "100 samples a week is plenty."
- Scoring: 0 no sample size; 1 order of magnitude only; 2 correct n, days, and the interim stance.

**Q13. Diagnose: mismatch count between approved and executed digests is 0, diff-view telemetry is 4%,
and reviewers approve 99.4% of proposals.**

- Answer: The binding works and the review does not. A 0.6% decline rate with 4% diff views means the
  reviewers are approving the justification, not the change; the digest cannot catch a wrong-but-approved
  action.
- Reasoning: Digest equality certifies fidelity, not authorization. The relevant defect class is wrongful
  approval, measured by decline rate and diff-view rate, not by mismatch.
- Follow-up: What experiment would distinguish laziness from genuine high precision?
- Red flag: "mismatch is zero, so the gate is working."
- Scoring: 0 declares success; 1 notices diff views; 2 names the wrongful-approval gap and a test.

**Q14. The class-D loss estimate is a mean of 9,400 from four precedents. What is the risk to the routing
decision?**

- Answer: The mean is unstable over a heavy tail, so the decision should be checked at both ends. At 900
  per event the class-D gate returns 5.4 dollars per review dollar rather than 54; at 40,000 it returns 232.
  The sign of the decision survives, the magnitude does not.
- Reasoning: Expected loss is linear in severity, so the ratio scales with it; the routing threshold is
  therefore insensitive in direction and very sensitive in investment level.
- Follow-up: How would you present the estimate to a risk committee?
- Red flag: quoting 9,400 as a point estimate.
- Scoring: 0 ignores instability; 1 acknowledges the tail; 2 recomputes the ratio at both ends.

**Q15. Which single change would most reduce residual risk, and what does it cost?**

- Answer: Close the digest gap on the three undigestable tools, moving 124 proposals a day from unbound
  full review to bound digest review. Residual risk falls mainly through the stale-approval channel, worth
  about 16,800 USD/year, and the class-D escalation that then becomes affordable removes 182.69 of residual
  for one additional head.
- Reasoning: The unbound proposals are 6% of volume and 100% of the approvals that cannot be bound, so they
  carry the concentration. Everything else in the design is a tuning exercise by comparison.
- Follow-up: What would you do if the canonicalization were provably undecidable?
- Red flag: proposing more reviewers instead of closing the structural gap.
- Scoring: 0 names a tuning change; 1 names the gap without the economics; 2 names the gap, the channel
  it closes, and the cost.

**Q16. Why route on `value_c > 1` rather than on risk class?**

- Answer: Risk class encodes severity but not volume or review time. Class D is the most severe and the
  smallest; class C is slightly less severe and eight times more frequent. The ratio folds severity, volume,
  catch probability, leakage, and review cost into one comparison against the reviewer's price.
- Reasoning: A routing rule is an allocation of scarce reviewer seconds, so it must compare value per second,
  not value per event.
- Follow-up: What breaks if catch probability depends on the queue state?
- Red flag: "gate everything above class B."
- Scoring: 0 class-based rule; 1 ratio without the queue caveat; 2 ratio plus the capacity interaction.

### Authority, escalation, and recovery drill

**Authority is a state machine, not a job title.** The requester owns the business intent, the policy
engine owns route selection, the reviewer owns a bounded approve-or-decline decision, and the executor owns
only a byte-identical write. None may substitute for another. In particular, an on-call engineer may restore
availability but cannot turn a declined approval into an approved one; recovery authority is deliberately
narrower than change authority.

| State | Permitted actor | Transition evidence | Forbidden shortcut |
|---|---|---|---|
| Proposed | agent service | signed intent record and policy version | direct sink call |
| Quarantined | sanitization worker | detector score and retrieved chunk IDs | reviewer approval of raw instructions |
| Awaiting review | assigned reviewer | canonical diff, digest, record version | approval by ticket title alone |
| Approved | executor | approval ID, digest equality, unexpired window | payload regeneration |
| Declined | requester or agent | decline reason code | reuse of the approval ID |
| Expired | policy engine | clock exceeds validity window | delayed execution |
| Executed | sink | idempotency receipt and new record version | a second side effect on retry |

The authority tuple is `(principal, action_id, digest_sha256, policy_version, expires_at, record_version)`.
The executor validates every member. A valid reviewer signature over the wrong policy version is rejected,
and a valid policy result over a changed record version is rejected. This is stricter than role-based access
control because it answers not only *who* signed but also *what exact transition* that signature can cause.

**ILLUSTRATIVE escalation clock.** A class-C proposal may wait 15 minutes before reassignment; class D may
wait 5 minutes. Reassignment changes the reviewer, never the required artifact. At 15 minutes, C becomes
`expired`, not `auto-approved`; the system creates a fresh proposal using the current record version. This
avoids converting an availability alert into an authorization bypass.

The following worked recovery trace distinguishes a safe retry from a compensating action:

```text
PRECONDITION
  approval_id = A-91, digest = d1, idempotency_key = k1, record_version = v17
TRACE
  1. executor submits (d1, k1) to the sink; network reply times out.
  2. executor queries the receipt ledger by k1.
  3a. receipt exists with d1: mark Executed; do not submit again.
  3b. receipt absent and record remains v17: resubmit exactly (d1, k1).
  3c. receipt absent and record is v18: mark Stale; require a new proposal.
INVARIANT
  k1 can produce at most one sink effect, and no path changes d1 after approval.
FAILURE BOUNDARY
  A downstream sink that cannot store or query k1 cannot provide exactly-once behaviour;
  its route must be full review plus a reconciled compensating workflow, not optimistic retry.
```

A compensating action is not a retry. If the approved update incorrectly changed a ticket owner from Elena to
Ravi, the correction `owner=Elena` is a new write with a new digest, a new policy evaluation, and normally a
new approval. Treating it as rollback under the old approval would let any later operator smuggle a change
through the old authorization.

**ILLUSTRATIVE recovery objective.** Suppose the receipt ledger is unavailable for 12 minutes while 36
approved proposals arrive at 3 proposals/minute. The executor pauses rather than guessing whether a timeout
committed. When the ledger returns, it resolves all 36 keys. With a 20-second median receipt lookup, the
reconciliation work is `36 x 20 = 720` seconds, or 12 reviewer-equivalent minutes; the business delay is
accepted because the alternative is 36 opportunities for duplicate or stale effects. This is a constructed
capacity calculation, not an observed outage.

The recovery runbook has four stop conditions. First, any approval/digest mismatch stops the individual
execution and writes an immutable mismatch event. Second, two mismatches for the same tool in 30 minutes
freeze that tool's route because the canonicalizer may be inconsistent. Third, receipt-ledger unavailability
past the maximum approval window expires queued approvals. Fourth, a policy-signature verification failure
freezes every affected policy version, since selectively bypassing one failed verification destroys the audit
boundary.

**Audit chain.** Each transition appends an event containing `event_id`, event time, processing time,
actor class, action ID, prior state, new state, digest, policy version, record version, and correlation ID.
The log is append-only from the application perspective; access to query it is separated from access to
execute actions. A reviewer may inspect the audit chain but cannot rewrite it, while an auditor may sample
executed actions without gaining sink credentials.

The audit question is temporal: can an investigator reconstruct what was known *before* approval and what
was verified *at* execution? Storing only the final ticket is insufficient because it loses the retrieved
chunk IDs, the original version, and the route decision. Storing only the model transcript is insufficient
because it cannot prove the bytes sent to the sink. The minimal evidentiary join is `action_id -> approval_id
-> digest -> idempotency_key -> sink receipt`.

**ILLUSTRATIVE audit sampling calculation.** If an investigator samples 80 executed class-C actions from a
week of 208 and finds zero missing joins, the simple one-sided 95% upper bound on a missing-join rate within
that sampled population is `1 - 0.05^(1/80) = 3.68%`. That is far too wide to certify an exact-binding
control. The operational response is not to declare success; it is to run a complete join check over all 208
executions and reserve sampling for semantic review of whether the cited policy actually applied.

A complete join check should return four disjoint counts: executed with valid bound approval, executed with
permitted auto-execution evidence, rejected before sink submission, and impossible states. The fourth count
must be zero. Examples include an execution after expiration, a receipt with no action ID, or an approval
linked to two distinct digests. A dashboard that sums the first three but omits impossible states reports
throughput, not control integrity.

**Escalation failure modes.** A reviewer can be unavailable, a policy clause can conflict, a requested action
can exceed the registered envelope, or the sink can make the state ambiguous. These have different owners:
availability goes to the queue manager, policy conflict to the policy owner, envelope expansion to the
business authority, and ambiguous sink state to the recovery operator. Sending all four to the on-call
reviewer encourages authority drift because the reviewer can see the customer urgency but may not alter
policy or infer a missing receipt.

An emergency route remains an approval gate. It requires two named emergency approvers, a shorter maximum
validity window, a mandatory reason code, and post-event audit within one business day. It does not accept a
single administrator's free-form note as a substitute for a digest. Emergency authority changes the quorum;
it does not erase the object being authorized.

**Decision test.** When asked to unblock a stuck action, apply this sequence: (1) can the prior effect be
proved from the receipt ledger; (2) does the present record still match the approved version; (3) does the
policy version still authorize the exact digest; (4) is the signer still within the required quorum; and
(5) has the approval expired? A single `no` yields a new proposal or an explicit incident, never a manual
sink call. The sequence is intentionally conservative: urgency can choose an emergency quorum, but cannot
manufacture evidence that does not exist.

This control-plane detail changes the economics in the main case. A 20-second receipt lookup is cheaper than
a 55-second human re-review and avoids the much larger uncertainty cost of duplicate execution. Conversely,
a permanent receipt ledger is part of the cost of claiming idempotency; omitting it would make the labour
comparison artificially favourable. Any real deployment must price ledger retention, key management, audit
storage, and emergency exercises alongside reviewer seconds. Those costs are unknown here and are therefore
not folded into the ILLUSTRATIVE policy totals.

**Practice prompt.** A class-D action is approved under policy version 18, the policy is revoked before
execution, and the record is unchanged. What happens? The executor rejects it because the authoritative
policy version is part of the approval tuple and revocation changes the authorization context. The action is
then re-proposed under the current policy; it is not grandfathered merely because a reviewer acted earlier.
This rule can delay a legitimate change, but it makes revocation meaningful.

**Practice prompt.** A sink receipt appears for `k1` but carries a different digest `d2`. What happens? Treat
it as a high-severity integrity incident: freeze the tool, preserve the receipt and canonical pre-image, and
do not compensate automatically. The mismatch could be a sink serialization defect, a key collision, or a
corrupted audit join. Each explanation requires different remediation, but all invalidate the assumption that
an idempotency key identifies one approved effect.

## 15. Quick Reference Card

| Question | Answer | Number |
|---|---|---|
| Harmful proposals per day | product of three conditionals | 8.6378 |
| Unmitigated daily loss | sum of class loss | 16,308.24 USD |
| Worst class by loss share | class C at 65.7% | 10,710.92 USD |
| Review cost per second | 92,000/255/7.8/3,600 | 0.012848 USD |
| Full-review heads | ceil(23.717/7.8) | 4 |
| Digest-review heads | ceil(20.092/7.8) | 3 |
| Digest coverage | 1,948 of 2,072 proposals | 94.02% |
| Break-even severity, class A | 22 s x rate / (q x m x 0.91) | 157 USD |
| Break-even severity, class C | 95 s x rate / (q x m x 0.91) | 622 USD |
| Quarantine residual leakage | 1 - recall | 9% |
| Digest gate wait at 4 heads | Erlang C, rho 0.628 | 7.6 s mean, 43.8 s p95 |
| Stale-approval collisions | Poisson window, 120 s | 0.2261/day |
| Recall audit size | 0.91 vs 0.75 at 80% power | 615 held proposals, 14.1 days |
| Adopted policy total | quarantine plus C/D gate | 1,187.36 USD/day |
| Adopted annual saving | 4,158,602 - 302,777 | 3,855,825 USD |
