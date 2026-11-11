## 7. The Finding

The finding has a primary claim, a mechanism, and a scope. Stating only the claim invites
over-generalisation; stating the mechanism makes it transferable.

**Primary claim.** For this decision, a ranking built on `m * (p1 - p0) - c` dominates a ranking
built on `p1` at the same contact budget, by 21,600 dollars per cycle in the constructed
population — a 19.1% improvement in value (134,400 versus 112,800) achieved with no additional
contacts, no new data source, and no change to the offer. The improvement comes entirely from
changing the target of the ranking.

The magnitude deserves a sanity check rather than awe. 19.1% is the gap between the two policies on
the constructed three-group population, where the "sure thing" group is 24% of all rows and the
"sleeping dog" group is 40%. In a population with fewer sure things, the gap shrinks; in a
population with more, it grows. The finding is directional and its size is population-dependent,
which is why the case reports the direction as the claim and the size as an illustration.

**Mechanism.** Propensity ranking allocates budget by the level of `p1`. Uplift ranking allocates by
the slope `p1 - p0`. Level and slope are uncorrelated in general and negatively correlated in
practice, because members with high purchase probability have less room to improve. Any programme
that spends a fixed budget on the highest-level members is therefore systematically funding the
members it does not need to fund. This is not a modelling defect that a better model fixes; it is a
consequence of the objective the model was trained to hit.

The mechanism has an observable signature, which is what makes it a finding rather than an opinion:
the selected population's *realised* response rate rises while the selected population's *realised
incremental* margin falls. Both metrics move in opposite directions, and the vanity metric is the
one that appears first on the dashboard.

**Scope.** The claim holds when all of the following are true, and each condition is checkable:

1. The budget binds, so a truncation must happen somewhere (at 120,000 of 250,000 it does).
2. Contact cost is positive and uniform, so a break-even threshold exists and is the same for all
   rows (true within a channel, false across channels).
3. The uplift estimate is at least ordinally accurate across the selected band; it need not be
   calibrated in level, but a badly miscalibrated estimate that inverts the top decile would
   destroy the advantage.
4. The holdout used to train `p0` is a genuine random sample of the eligible band, not a sample of
   previously contacted members.
5. No interference strong enough to make one member's treatment change another's outcome at a scale
   that reorders the top deciles.

Condition 5 is the one the case cannot verify, and it is stated as a limit rather than resolved. At
a 48% contact rate, interference is plausible; the direction of its effect on the ranking is toward
compressing differences between members in the same store catchment, which would flatten the value
curve and reduce the advantage of uplift ranking without eliminating it.

**What the finding is not.** It is not a claim that the uplift model is accurate in dollars. The
estimated 109,200-cycle value in section 6.2 depends on `delta_p` values that are constructed. It is
not a claim that the policy was deployed. Nothing was deployed. It is not a claim that the retailer
improved margin; the retailer does not exist.

**The secondary finding** is about the evaluation gate itself and is arguably more useful than the
primary one. The original acceptance criterion for the programme was "response rate at least 3
points above the rules queue." That criterion is satisfiable by the worse policy: the propensity
policy delivers 38.5% against the rules queue's constructed 26.4%, a 12.1-point improvement, and
destroys 21,600 dollars per cycle while doing it. A gate that a dominated policy can pass is not a
gate.

```text
FORMULA
  Gate pass rate for a dominated policy
    pass(S_propensity) = 1 iff  response_rate(S_propensity) - response_rate(baseline) >= 3 points

WHERE
  response_rate(S) = sum over i in S of p1_i / |S|
  baseline rules queue response rate = 26.4% (ILLUSTRATIVE)

CONSTRAINT
  The comparison is only meaningful at equal budget. A policy that contacts fewer members
  can beat a larger policy on rate while losing on total value.

NUMERICAL EXAMPLE
  propensity policy rate = (60,000*0.55 + 60,000*0.22) / 120,000 = 46,200/120,000 = 0.3850
  baseline rate          = 0.2640 (constructed: the rules queue concentrates on mid-recency rows)
  margin over baseline   = 38.50 - 26.40 = 12.10 points, which passes a 3-point gate
  value                  = 112,800 versus uplift policy 134,400, a 21,600 shortfall
  The gate passes the policy that loses 1,123,200 dollars per year.

INTERPRETATION
  A response-rate gate is not a weak proxy for margin; it is a proxy with the opposite sign in
  the region that matters. The corrected gate is a margin-per-contact comparison at equal
  budget, with a holdout, and the response rate demoted to a diagnostic.
```

That arithmetic is the case's real lesson and the reason it belongs in an `evaluation-mismatch`
collection rather than a modelling collection. Changing the model would not have helped; changing
the acceptance criterion would have.

**A third, quieter finding** concerns the frequency control. The cooldown of fourteen days permits
six contacts per quarter, and at the constructed decay of 0.62 the third through sixth contacts are
margin-negative. The programme's own policy documents treated the cooldown as the frequency policy,
which meant the operating rule permitted 4.4 contacts per member per quarter on average among
contacted members while the economically correct cap was 2. The gap between the permitted maximum
and the profitable maximum is a policy defect that no model metric would surface.

Takeaway: the finding is a target change plus a gate change plus a frequency cap, and each of the
three is independently checkable against the same arithmetic.

## 8. System/Solution Design

The design is a batch decision pipeline with four gates, an append-only log, and a measurement path
that runs on the same snapshot as the decision.

```text
[event tables] -> [as-of feature build] -> [uplift scoring] -> [value sort]
                          |                     |                 |
                          v                     v                 v
                   [quality gate Q1-Q6]   [calibration gate]  [eligibility gate]
                                                                      |
                                                                      v
                                                            [review partition]
                                                              |            |
                                                     [auto-send]      [human queue]
                                                              |            |
                                                              +-----+------+
                                                                    v
                                                          [append-only decision log]
                                                                    |
                                                                    v
                                                     [holdout + outcome join at T+42d]
```

Each box has a contract, and the contracts are what make the diagram more than a picture.

**Stage 1 — As-of feature build.** Inputs are read with `effective_ts <= cut_off - 15 min`. Output
is a feature table keyed by `member_id, decision_cycle`, plus a snapshot identifier. Contract: the
same snapshot identifier must reproduce the same feature table byte-for-byte, and the build must
fail closed if any feature's definition window overlaps the outcome window.

**Stage 2 — Scoring.** Two models produce `p1` and `p0`; the difference is the estimate. Output adds
`p1`, `p0`, `delta_p_raw`, `delta_p_clipped`, and a `model_version`. Contract: no row may be scored
twice with different artefacts in the same cycle, which is enforced by a unique constraint on
`(member_id, decision_cycle, model_version)`.

**Stage 3 — Value sort.** `V = m * delta_p_clipped - c`, sorted descending with a deterministic
tie-break on `member_id`. Contract: the sort must be total, which requires the tie-break; a partial
order produces non-reproducible selections at the budget boundary and did, in the constructed
replay.

**Stage 4 — Eligibility gate.** Consent, suppression, cooldown, conflicting-experiment, and
offer-funding checks. Contract: this gate can only remove rows, never add them, and every removal
writes a reason code. A gate that can add rows is not a gate.

**Stage 5 — Review partition.** Rows above the policy envelope (estimated value above $2.00 per
contact, or any offer with a face value above $10.00, or any member with an open service case) are
routed to the human queue, capped at 2,400. Rows within the envelope auto-send. Contract: the
partition does not reorder the selected set; a reviewed row that the reviewer rejects becomes an
abstention with a reason code, and the freed budget is *not* refilled in the same cycle, because
refilling requires a re-sort and would break the published decision.

That last contract is a real design decision with a cost. Not refilling wastes budget — in the
constructed cycle, 180 of 2,400 reviews are rejected, which leaves 180 contacts unused, worth about
$0.82 × 180 = $147.60 of forgone value. Refilling would recover $147.60 and would make the
published decision mutable after the fact, which breaks the audit trail. The $147.60 is the price of
immutability, and it is cheap.

**Stage 6 — Decision log.** Append-only, one row per decision, including abstentions. Fields:
`decision_id`, `member_id`, `decision_cycle`, `snapshot_id`, `feature_hash`, `model_version`,
`p1`, `p0`, `delta_p_clipped`, `V`, `rank`, `action`, `reason_code`, `reviewer_id`, `review_ts`,
`send_ts`. Contract: no update, no delete; a correction is a new row with a `supersedes` pointer.

**Stage 7 — Measurement.** At T+42 days the outcome join produces gross margin, net margin, return
rate, and contact cost per row, joined to the holdout flag. Contract: the outcome join may not read
the decision log's `action` field for any row where the holdout flag is set, because a holdout by
definition received no contact and a join that conditions on action would silently drop them.

The failure and recovery paths are part of the design, and each has a specific behaviour:

| Failure | Detection | Behaviour | Recovery |
|---|---|---|---|
| Snapshot missing | Snapshot id not resolvable | Cycle falls back to rules queue | Rebuild from source; re-run |
| Freshness breach (Q2) | Max feature age > 26h | Degrade to baseline, log degraded flag | Backfill; next cycle normal |
| Score range violation (Q4) | Any `p1` outside [0,1] | Quarantine those rows only | Investigate model artifact |
| Offer withdrawn mid-cycle | `offer_id` inactive at send | Exclude offer, keep content-only | Re-issue with replacement offer |
| Review backlog > capacity | Queue depth > 2,400 at 04:00 | Auto-approve within envelope only | Raise cap or narrow envelope |
| Send job partial failure | Sent count < selected count | Log per-row send status | Retry idempotently by `decision_id` |
| Late correction to source | Correction after cut-off | Do not rewrite; next cycle uses it | Note in cycle close-out |

The recovery path for a partial send failure deserves emphasis because it is the one place where
idempotency is safety-critical. The send job is keyed by `decision_id`, so a retry after a broker
failure cannot double-send; without that key, a retry storm sends the same offer two or three times
to the members in the failed segment, which is both a cost event and a complaint event.

The design also has an explicit "do less" mode, which is what a mature pipeline looks like. If any
of the following hold, the cycle publishes the rules queue instead of the ranked selection, and the
decision log records `mode = degraded` with the triggering condition:

- calibration drift beyond the pre-registered band in two consecutive cycles,
- more than 2% of rows quarantined by Q4,
- review capacity below 1,500 rows,
- a live pricing or supply event that invalidates the offer's margin assumption.

The degraded mode exists because the alternative — shipping a ranking that the team no longer
believes — is worse than shipping an older policy that everyone can explain. It is also what allows
the programme to survive a bad month without a public reversal.

Two design options were rejected, and the reasons are worth recording because they will be proposed
again.

*Rejected: online learning.* The decision runs 52 times per year and the outcome matures in 42 days,
which gives at most 8 fully observed cycles per year. An online loop with 8 updates per year is a
batch retrain with extra failure surface. The specific test that would reopen this: if the cycle
cadence moved to daily with a 7-day outcome window, the update count rises to 52 per year and the
trade-off changes.

*Rejected: per-member threshold optimisation.* Solving a separate threshold per member is
mathematically equivalent to sorting by value when costs are equal, and it adds a parameter per
member that cannot be estimated. The specific test: if contact costs varied by member (for example
by channel preference), the problem becomes a genuine knapsack and the per-member formulation
becomes necessary.

Takeaway: the design's value is in its gates and its log, not in its model, and every gate can only
subtract rows.

## 9. Evaluation & Validation

The evaluation contract is stated before the result, because a metric defined after seeing outcomes
is a description, not a test.

| Element | Definition |
|---|---|
| Numerator | Gross margin from purchases in [cut-off, +28d], net of returns processed to +42d, minus contact cost |
| Denominator | Eligible member-weeks in the decision cycle (250,000), not contacted rows (120,000) |
| Population | Members active in the trailing 90 days, consenting, not suppressed, outside cooldown |
| Window | Decision cycle plus 28-day purchase window plus 42-day return tail |
| Comparator | Rules queue at the same budget on the same population, and a 10% household holdout |
| Guardrails | Complaint rate, unsubscribe rate, review rejection rate, contact-cost variance, frequency cap violations |
| Uncertainty | Holdout difference with a cluster-robust interval at household level; decile bands coarsened to 3 |
| Status | Designed and specified; **not executed**. No production cycle was run. |

The denominator choice is the one most often got wrong, and the arithmetic shows why it matters.
Using contacted rows as the denominator measures the value of a contact; using eligible rows
measures the value of the *policy*, which is what the owner is deciding. The two differ by the
factor 250,000/120,000 = 2.083, so the uplift policy's $1.12 per contact is $0.5376 per eligible
member-week, while the propensity policy's $0.94 per contact is $0.4512 — and only the second pair
of numbers answers the owner's question.

**Guardrails with thresholds.** A guardrail without a threshold is a wish. Each guardrail below has
a limit and a consequence.

| Guardrail | Constructed baseline | Limit | Consequence on breach |
|---|---|---|---|
| Complaint rate per 10,000 contacts | 3.1 | 6.0 | Pause channel, review creative |
| Unsubscribe rate per cycle | 0.42% | 0.60% | Reduce frequency cap to 1 |
| Review rejection rate | 7.5% | 15.0% | Narrow policy envelope |
| Contact cost per send | 0.62 | 0.75 | Re-solve threshold at new cost |
| Frequency cap violations | 0 | 0 | Block cycle; fix filter |
| Offer honour failures | 0.9% | 2.0% | Suspend offer at affected stores |

The complaint-rate guardrail is the one that has actually bound in comparable programmes, and it is
worth understanding why it binds before the margin metric does. Complaints rise with contact volume
and with offer relevance failure; a policy that maximises margin on a poorly calibrated uplift
estimate will concentrate contacts on members with high estimated lift and low realised lift, which
is precisely the population that complains about irrelevant offers. Margin and complaints are
therefore positively correlated in the failure direction, not opposed.

**Validation design.** Three stages, in increasing cost, each with a decision it can support.

1. *Offline replay.* Score historical cycles with the current artefact, select under the current
   policy, and compare to the rules queue's actual selection on the same rows. This detects
   implementation error and policy infeasibility, and it cannot detect target error, because the
   replay uses the same `delta_p` estimates that the policy was built on. It is a regression test,
   not evidence of value.
2. *Shadow cycle.* Run the pipeline live, log decisions, send the rules-queue selection, and compare
   the two on realised outcome at T+42. This detects target error partially: the shadow policy's
   selected set differs from the sent set, so the comparison is confounded by selection. Its real
   value is operational: it exercises every gate with live data and finds the failures that only
   occur on real inputs.
3. *Randomised holdout.* Within the contacted band, withhold contact from a 10% household-level
   random sample. This is the only stage that identifies the effect, and it is the only stage that
   can support a margin claim.

The holdout has a cost that must be stated honestly. Withholding 10% of 120,000 contacts forgoes
12,000 contacts, worth 12,000 × 0.82 = $9,840 per cycle at the decile-3 value, or $511,680 per
year. That is the price of identification, and it is 45.6% of the annual value the propensity
policy destroys (1,123,200). The holdout is expensive and it is still correct, because
without it the programme cannot distinguish a working policy from a lucky one, and the decision to
continue spending $3.87 million per year on contacts would rest on a response-rate dashboard.

**Power.** Section 6.5 established that a single-cycle decile readout resolves 2.57 points while the
policy's claimed effect in the marginal decile is around 3.2 points. The validation plan therefore
promises less than the policy claims: the holdout can confirm the effect in the top three bands
pooled over four cycles, and it cannot confirm it decile by decile. That asymmetry is stated in the
pre-registration so that a later reader does not mistake an unpowered null for a refutation.

**Not executed.** Nothing in this section ran. The status of every element is `designed`. The
honest summary is that the case specifies what a credible evaluation would look like, gives the
arithmetic that determines its resolution and its cost, and makes no claim about what such an
evaluation would find.

## 10. Action Taken

No production action was taken. The work products are a specification, a rejected prototype, a
design, and a measurement plan. This section distinguishes what was decided from what was done,
because a case that blurs the two is the failure mode the collection exists to prevent.

| # | Action | Owner | Status | Evidence produced |
|---|---|---|---|---|
| 1 | Audit the propensity prototype for leakage | Analytics | completed (constructed) | Timestamp audit; one post-purchase feature removed |
| 2 | Rebuild the target as a two-model difference | Analytics | completed (constructed) | Feature ledger; snapshot ids |
| 3 | Add the six quality gates | Data engineering | completed (constructed) | Gate thresholds and block/flag behaviour |
| 4 | Add the append-only decision log | Platform | designed | Field list and contract |
| 5 | Define the review envelope and capacity | Loyalty operations | designed | 2,400 rows, 2.4 minutes per review |
| 6 | Pre-register the holdout and its analysis | Analytics + finance | designed | 10% household holdout, 3 bands, 4 cycles |
| 7 | Recompute the margin vector by category | Finance | not started | — |
| 8 | Run the shadow cycle | Platform | not started | — |

The completed items are constructed: this case describes what a programme would have to do, and the
"completed (constructed)" status means the work is described rather than evidenced. Items 7 and 8
are the two that would actually change the decision, and neither was done.

Item 7 matters more than it looks. The whole selection rests on a single blended margin rate of
30%, and section 6.6 shows that a 22% rate in the promoted category raises the break-even threshold
by 36.4% and removes a decile from the selected set. Until the margin vector exists, the selected
set has an uncertainty band that is wider than the difference between the top two policy candidates.

Item 8 is the gate to everything else. A shadow cycle costs one cycle of analyst time and no
customer contact, and it is the cheapest way to discover that the feature build cannot run inside
the six-hour window between cut-off and send.

**Decisions that were made, with their rationale**, because these are the parts a successor
inherits:

- Keep the rules queue as the comparator and the degraded-mode policy, rather than retiring it. A
  comparator that is deleted cannot be used to detect regression.
- Clamp `delta_p` at zero for selection but report it unclipped. The clamp prevents negative-value
  rows entering a tie-break; the unclipped report preserves the sleeping-dog signal, which is a
  finding in its own right.
- Cap frequency at two per quarter rather than relying on the cooldown. The cooldown is a spacing
  rule; the cap is an economics rule, and only the second one is derived from the decay.
- Do not refill budget after review rejections. Immutability of the published decision is worth more
  than $147.60 per cycle.
- Keep the response rate on the dashboard as a diagnostic and remove it from the acceptance gate.

The last decision is the one that generated the most internal resistance, and the objection is
worth recording because it is reasonable: response rate is the metric the programme has reported for
three years, it is understood by every stakeholder, and replacing it with margin per eligible
member-week makes the weekly report harder to read. The counter-argument is that a metric which
rewards the dominated policy is not a simplification, it is a decision to lose money legibly.

## 11. Outcome & Results

**Status: `not-measured`.** No cycle was executed, no holdout was drawn, and no margin was realised.
The only results in this case are arithmetic identities that hold under stated assumptions.

What is established, and how firmly:

| Claim | Basis | Firmness |
|---|---|---|
| Break-even lift is 2.583 points at m = 24.00, c = 0.62 | Arithmetic | Exact under the stated inputs |
| Uplift ranking beats propensity ranking by 21,600 per cycle | Constructed group means | Illustrative; sign is the claim |
| Response-rate gate passes the dominated policy | Arithmetic on the same group means | Exact given the constructed rates |
| Two contacts per quarter is optimal at r = 0.62 | Geometric decay arithmetic | Illustrative; depends on r |
| A 10% single-cycle holdout resolves 2.57 points | Standard two-proportion formula | Exact given alpha, power, p |
| The selected set is feasible at budget and capacity | Counts | Exact given the counts |

What is not established, and would require execution:

- whether measured uplift in this population is positive at the margin,
- whether the two-model estimate ranks correctly out of sample,
- whether the review gate's rejection rate is 7.5% or materially different,
- whether complaints rise with volume at the modelled slope,
- whether interference compresses the value curve enough to reorder the top decile.

**The counterfactual question.** A reader is entitled to ask what the programme would have earned
had it run the propensity policy. The answer is the constructed $112,800 per cycle, which is
$5,865,600 per year, and it is not a forecast. The reason to report it is comparative: the
difference between the two policies, $1,123,200 per year, is the annual cost of using the wrong
target, and that number is what justifies the work.

A second counterfactual is more useful for a practitioner: what would the programme have earned by
doing nothing? At zero contacts, the value is zero by construction, but that is an artefact of
measuring only incremental margin. The programme also produces retention, data, and redemption
behaviour that the rules queue maintained, so "do nothing" is not a neutral baseline; it is a
different programme with unmeasured effects. The case keeps the rules queue as the baseline for
exactly this reason.

**What the results would look like if the policy were wrong.** The value of a case is partly in its
falsifiability, so the expected signature of failure is specified:

- If the holdout shows lift indistinguishable from zero in the top band across four cycles, the
  estimate is noise and the correct policy is the rules queue with the budget cut by half.
- If lift is positive but falls below 2.583 points in every band, the offer is under-funded and the
  correct response is a larger face value or a different incentive, not a better model.
- If lift is positive and large in the top band but the complaint rate breaches 6.0 per 10,000, the
  policy is buying margin with relationship damage and the frequency cap must fall to one.
- If lift is positive in the middle bands and negative in the top band, the ranking is inverted at
  the head — the signature of a leakage defect or a `p0` model trained on contacted rows only.

Takeaway: the case's output is a testable operating rule with a stated invalidation signature, and
its results are arithmetic rather than measured, which the status line must continue to say.
