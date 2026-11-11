## 8. System/Solution Design

### 8.1 The state machine is the design

The system's behaviour is defined by a per-span state machine with exactly seven states, and the point
of the design is that no state can be skipped and every non-terminal state has an owner.

| State | Entered when | Owner | Allowed next states |
|---|---|---|---|
| `OFFERED` | leg appears in the telephony extract | pipeline | `DENIED_CONSENT`, `IDENTITY_CONFLICT`, `INELIGIBLE_SHORT`, `SEGMENTED` |
| `SEGMENTED` | VAD produced regions and the reader verified the media hash | pipeline | `TRANSCRIBED`, `CAPTURE_INCOMPLETE` |
| `TRANSCRIBED` | tokens carry offsets and versions | pipeline | `SPAN_CANDIDATE`, `NO_CANDIDATE` |
| `SPAN_CANDIDATE` | at least one rubric pattern fired above the score floor | pipeline | `QUEUED`, `THROTTLED` |
| `QUEUED` | a review slot is allocated | review ops | `DISPOSED` |
| `DISPOSED` | reviewer recorded a disposition and reason code | reviewer | terminal |
| `THROTTLED` | capacity exhausted before allocation | review ops | `QUEUED` (when capacity returns) |

Three properties follow from this table and each closes a specific failure. First, `THROTTLED` is a
state rather than a silent drop, so throttled legs remain countable and the report can state achieved
coverage. Second, `NO_CANDIDATE` is distinguished from `SPAN_CANDIDATE`-then-`THROTTLED`, so a low flag
rate caused by a weak detector is never confused with a low flag rate caused by a full queue. Third,
`DISPOSED` is terminal and immutable; a correction requires a new disposition row referencing the
superseded one, which preserves the audit trail that an in-place update would destroy.

The refusal states — `DENIED_CONSENT`, `IDENTITY_CONFLICT`, `CAPTURE_INCOMPLETE` — are outcomes of the
same machine, not exceptions. That is deliberate: an exception path is a path nobody instruments, and an
uninstrumented refusal is indistinguishable from a leg that was never offered.

### 8.2 Span proposal and the score floor

Candidates are produced by matching versioned rubric patterns against tokens, where a pattern is a
phrase alternation plus a negation rule. The negation rule matters more than the phrase list. A pattern
"too expensive" fires on the utterance, but the same tokens fire inside "I never said it was too
expensive", which is a denial rather than an objection. The matcher therefore looks in a window of three
tokens before the phrase for negators and counterfactual markers, and it emits the span with a
`negated` flag that the reviewer sees.

The score floor is a single tunable and its sweep is the most useful artefact the programme produces,
because it converts a modelling preference into a capacity decision.

| Score floor | Candidate spans | Precision (broad) | Accepted spans | Recall on 63-item audit | Review hours | Reviewers (FTE) |
|---|---|---|---|---|---|---|
| 0.30 | 3,090 | 0.61 | 1,885 | 51/63 = 0.810 | 1,133 | 6.87 |
| 0.50 | 1,930 | 0.704 | 1,359 | 47/63 = 0.746 | 708 | 4.29 |
| 0.70 | 1,410 | 0.79 | 1,114 | 44/63 = 0.698 | 517 | 3.13 |
| 0.90 | 980 | 0.86 | 843 | 33/63 = 0.524 | 359 | 2.18 |

The table is `ILLUSTRATIVE` but internally consistent: as the floor rises, volume falls, precision rises,
and recall falls. The decision it supports is not "pick the best row". It is "pick the row whose recall
supports the claim you intend to make". For a population rate with a correction term, 0.50 is defensible
because missed objections are estimated by audit. For any claim of the form "we found all the objections
in this cohort", no row is defensible, because even 0.30 recall of 81.0% leaves 12 of 63 objections
unfound in a 150-call audit, and the audit's own interval on that recall is 0.69 to 0.89.

The cost column is what makes the sweep actionable in a business review. Moving from 0.50 to 0.70 saves
191 review hours and 1.16 FTE but costs 3 audit-found objections out of 63, which raises the estimated
missed-objection count in the unflagged population by roughly 3/63 × 36,388 × p̂. That trade is a
decision for the owner, not for the engineer, and presenting it as a table rather than as an accuracy
number is what allows the owner to make it.

### 8.3 Evidence binding and the replay contract

A span is only usable if a reviewer can hear it. The binder stores `start_ms`, `end_ms`, the
`recording_hash`, and the token indices; the console plays from `start_ms − 8,000` to `end_ms + 4,000`,
clamped to the media bounds and capped at 20,000 ms. For a 384-second leg the mean replay window is
about 13 seconds, which is 3.4% of the call; the review cost is not dominated by replay but by
contextual listening, which is why the 22-minute mean per span holds.

The replay contract has two failure tests, both run before a span enters `QUEUED`. The **audibility**
test decodes the window and confirms the energy profile is consistent with speech rather than silence;
this catches offset drift after re-encoding. The **hash** test confirms the media the binder referenced
is the media being played; this catches the case where a storage migration changed the container and
invalidated every stored offset. Failing either test sets `ANCHOR_BROKEN` and the leg leaves the
evidence-bearing count, which is a visible loss rather than a quiet one.

### 8.4 Review sampling design

The 446-leg audit of unflagged legs is a stratified probability sample with fixed quotas, drawn
independently of the reviewer's queue. Strata are queue family, hour band, language route, and channel
count. Allocation is proportional with a floor of 25 legs per stratum, and the estimator is the
stratum-weighted mean:

```text
FORMULA [D5]: p̂_strat = Σ_h (N_h / N) × (k_h / n_h)
WHERE: h indexes strata; N_h = unflagged legs in stratum h; n_h = audited legs in stratum h
       k_h = audited legs with a hand-found objection
CONSTRAINT: every n_h ≥ 1 and strata cover the unflagged population exactly;
       sampling within stratum is without replacement and independent of content
NUMERICAL EXAMPLE: two strata only, N_A = 28,383 (78%), N_B = 8,005 (22%)
       n_A = 268, k_A = 16 → 0.0597 ; n_B = 178, k_B = 2 → 0.0112
       p̂_strat = 0.78 × 0.0597 + 0.22 × 0.0112 = 0.04657 + 0.00246 = 0.04903
       unweighted pooled estimate would be 18/446 = 0.04036, which is 0.87 points lower
INTERPRETATION: if the strata differ in size and in rate, the unweighted sample mean is
       biased toward the oversampled stratum, so stratification weights are mandatory.
```

The design consequence is that the audit is planned before the month starts, with its quotas and its
cost. At 446 legs × 6.4 minutes = 2,858 listening minutes, the audit is 47.6 reviewer-hours, or 0.29
FTE — small against the 708 hours of candidate adjudication it corrects, which is exactly why it should
never be the thing cut first.

### 8.5 Access, purpose, and the schema as control

The purpose limitation is enforced structurally rather than by policy text. Span rows carry `call_id`,
`leg_id`, `queue_id`, and `reviewer_id` (hashed), and no `agent_id`. Agent identity exists in the
telephony system and is joined only in the queue-level aggregation job, which emits counts per queue
and never per agent. A team-level table would require a new view, a new purpose record, and a new access
review, which makes the prohibited use an explicit project rather than a query.

The transcript table is partitioned by month and by queue, so an analyst with queue-level access
physically cannot read another queue's text. Reviewer access is scoped to the queue they serve, and the
`reason_code` vocabulary is closed: adding a code is a schema change with a review, because reason codes
drift into free text and free text drifts into commentary about employees.

### 8.6 Operational protocol

The monthly cycle is a fixed sequence with explicit gates, run whether or not the numbers look good.

1. **Day 1** — freeze the extract at the event-time cut-off; reconcile offered against billed legs and
   publish the reconciliation gap before any rate.
2. **Day 2** — run gating, segmentation, transcription, and diarisation; write version strings into the
   manifest. Any leg whose pipeline run failed is retried once, then recorded as failed.
3. **Day 3** — generate candidates, apply the pinned score floor, and compute the capacity requirement.
   If the requirement exceeds available reviewer hours, apply the throttle rule and record the throttled
   count; never lower the floor silently to make the queue fit.
4. **Days 3–8** — adjudicate, with 5% of spans double-reviewed blind to measure reviewer agreement.
5. **Day 9** — draw and complete the stratified audit of unflagged legs.
6. **Day 10** — compute the corrected rate, the interval, the unevidenced share, and the per-queue
   cells; suppress any cell below 250 legs; write the manifest.
7. **Day 11** — publish, with the limitation statement attached to the table rather than in a footnote.

The protocol's value is that each gate has an owner and a timestamp, so a delay is attributable. The
most common real defect in such programmes is not a wrong number but a number published before the
audit completed, which is precisely what gate 5 prevents.

## 9. Evaluation & Validation

Evaluation is stated in eight fields because a metric without them cannot be audited. Everything in this
section is either `executed` on the constructed data or explicitly `not-executed`; no partially executed
item is described as done.

### 9.1 Definitions

| Field | Objection-span precision | Objection recall | Corrected rate |
|---|---|---|---|
| Numerator | accepted + reclassified spans (1,359) | audit objections the model flagged (47) | (1,082 + 1,305.4) legs |
| Denominator | all adjudicated spans (1,930) | audit objections found by humans (63) | eligible legs (38,000) |
| Population | 1,612 flagged legs, one month, in-scope queues | 150 randomly drawn flagged and unflagged calls | 38,000 eligible legs |
| Window | span month by event time | same month | same month, event time |
| Comparator | reviewer-with-model-visible console (retired) | human exhaustive review of the 150 calls | evidenced-only naive rate (2.85%) |
| Guardrails | 5% blind double-review agreement; audibility tests; throttled count | recall interval width | unevidenced share; cell suppression |
| Uncertainty | Wilson 95%: 0.683–0.724 | Wilson 95%: 0.627–0.837 | clustered 95%: 4.61%–7.96% |
| Status | executed (constructed) | executed (constructed) | executed (constructed) |

The comparator column is the part most programmes omit and it is the part that makes the number
meaningful. A precision of 70.4% is neither good nor bad in isolation; it is better than the 61.9%
achieved when undetermined spans were excluded from the denominator and worse than the 79% achieved at
a 0.70 floor, and both of those comparisons are more informative than the absolute value.

### 9.2 Guardrail measurements

**Blind double-review agreement.** Two hundred spans were adjudicated twice, once with the hypothesis
hidden and once with it visible, by different reviewers. Raw agreement was 150/200 = 75.0%; the expected
agreement under independence, given the marginals 142/200 and 132/200 positive, is
(142/200)(132/200) + (58/200)(68/200) = 0.4686 + 0.0986 = 0.5672; so

```text
κ = (0.7500 - 0.5672) / (1 - 0.5672) = 0.1828 / 0.4328 = 0.4224
```

A κ of 0.42 is moderate and it means the label itself is noisy, which caps the achievable measured
agreement between any model and the label at roughly sqrt(0.42) in the classical correction, or about
0.65. The practical consequence: the 70.4% precision is a measurement of agreement with a noisy label,
not with the truth, and pushing model precision past about 0.85 against this label is not achievable
without first improving the label. That is the single most useful thing the guardrail revealed.

**Audibility.** Of 1,930 queued spans, 27 (1.4%) failed the audibility test after a storage migration
that re-encoded 0.5% of the month's media, which is 190 legs. Those 27 spans cost 27 × 22 / 60 = 9.9
review hours to discover and would have cost far more as published errors.

**Throttle.** At 1,930 candidates and 708 review hours, capacity was sufficient in the base month. In a
stress month with the 0.30 floor the requirement is 1,133 hours against a capacity of 709.5 hours, an
excess of 59.7%; the protocol requires throttling and reporting achieved coverage rather than extending
the queue.

### 9.3 Promotion and withdrawal criteria

The measurement is promoted to a published rate only if all six conditions hold, and each is binary so
that the decision is not a matter of judgement.

1. The stratified audit completed with all quotas met and n_h ≥ 25 in every stratum.
2. The unevidenced share is below 5% of eligible legs and is published next to the rate.
3. The clustered interval half-width is below 2.0 percentage points at population level.
4. Blind double-review κ is at least 0.40, or the label is declared noisy and the ceiling is stated.
5. The reconciliation gap between offered and billed legs is below 0.5%.
6. No published cell is below 250 legs.

If conditions 1, 2 or 5 fail, the report is withdrawn and reissued as a method note. If 3 or 4 fails,
the rate is published with a wide interval and an explicit label ceiling. The withdrawal is not a
punishment; it is the mechanism that keeps the number's meaning stable across months.

### 9.4 Explicitly not executed

No real audio was processed. No vendor was benchmarked. No reviewer was timed on a real console. The
κ value, the 27 audibility failures, the storage-migration incident, and the stress-month capacity
figures are constructed to demonstrate the checks, not observed. The corrected-rate interval assumes a
design effect of 1.8, which was not estimated on real cluster data; the sensitivity is that a design
effect of 1.0 would narrow the half-width from 1.67 to about 1.25 points, and a design effect of 3.0
would widen it to about 2.2 points. Any reader who needs the real interval must estimate the design
effect from the actual per-call and per-agent intraclass correlation.

## 10. Action Taken

The decision record has four entries, each with an owner, a date, and a reversal condition. The record
is the deliverable that survives staff turnover; the analysis is evidence for it.

**A-1 — Publish Answer A with a mandatory correction term.** Published: the corrected rate 6.28%
(interval 4.61–7.96), the evidenced-only rate 2.85%, the unevidenced share 3.2%, and the queue-level
cells. Not published: any per-agent, per-team, or per-individual figure. Reversal condition: audit
quotas unmet for two consecutive months, in which case the unit reverts to Answer C and publishes only
the method note.

**A-2 — Fix the score floor at 0.50 for the reporting path and open no second path.** The 0.70 floor is
permitted only for a separate "assisted casework" queue whose output never enters the aggregate, because
mixing floors across months would make the rate incomparable. Reversal condition: if the 0.70 floor
achieves recall of at least 0.70 on two consecutive audits, the floor may be raised for the reporting
path with the change recorded as a rubric-equivalent break.

**A-3 — Fund the audit before any model work.** The stratified 446-leg audit becomes a standing
monthly cost of 47.6 reviewer-hours, protected in the capacity plan. The 1,000-leg upgrade is approved
for the following quarter because the interval half-width falls from 1.67 to 1.19 points, which is a
larger gain than any accuracy work currently available. Reversal condition: if the audit's revealed
p̂_unflag is stable within 0.4 points across three months, revert to a smaller rolling audit.

**A-4 — Retire the model-visible review console.** The console that showed the hypothesis beside the
audio is decommissioned; the blind console is the only adjudication route, and 5% of spans are
double-reviewed to monitor agreement. Reversal condition: none. The old console produced labels that
could not be distinguished from model confirmation, so its data are quarantined and excluded from all
historical comparisons.

Two operational changes follow from the analysis and were made in the same period. The first is
capacity-based throttling: review allocation now reads the day's arrival rate against the service rate
and refuses allocation past the policy threshold, writing `THROTTLED` explicitly. The second is a
report template whose header cell contains the unevidenced share, so that the number cannot be copied
into a slide without its caveat. The template is a control, not a courtesy: the failure it prevents is
the one described in section 2.3.

One action was considered and rejected. A proposal to raise the score floor to 0.90 in order to publish
a "high-confidence" rate of 86% precision was declined because the recall cost, 33/63 = 52.4%, would
have made the rate an estimate of the easiest-to-hear objections rather than of objections, and the
resulting number would have been lower, more precise, and less true. The rejection is recorded with the
sweep table so that the option is visibly considered rather than forgotten.
