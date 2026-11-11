
## 7. The Finding

The finding is a conditional operating rule with a measured-sized but unmeasured-value gain, and it is stated that way deliberately.

**Supported.** On the constructed population, at the store's real capacity of forty checks per store-day and at a fully loaded cost of 0.50 USD per check, the candidate ranking reaches an expected 10.116 real gaps per store-day against 5.140 for the incumbent's equal-minutes list. The difference, 4.976 gaps per store-day, is worth 20.18 USD/store-day at a value of 4.05504 USD per prevented stockout-day, and it survives every substitution assumption tested up to 30% of gap demand absorbed by neighbouring SKUs. The break-even precision of 12.33% sits below both the candidate's 25.29% and the incumbent's 12.85%, and the incumbent's margin over break-even is only 0.52 points.

**Not supported.** No statement about OSA percentage points, sales uplift, or return on investment is supported by this case. The 4.976-gap difference is arithmetic on constructed bin hit rates; converting it to OSA requires knowing that a check fixes 88% of the gaps it finds (`phi`), that the fixed gap is not immediately re-emptied, and that the audit's own error is smaller than the effect. §9 shows the store-day-level OSA effect is 0.031 percentage points per store, well inside a pilot's resolving power at twelve stores.

**Why the finding is small in OSA terms.** The store has 2,300 sellable location-days and a 3.1% failure rate, so about 71 gap-location-days exist per trading day. Reaching 10 instead of 5 of them and fixing 88% of those reached prevents 8.90 against 4.52 gaps — an absolute OSA movement of 0.064 percentage points against 0.032. OSA is dominated by the 61 gaps nobody checked, not by the ranking of the forty that were checked. A programme that reports only OSA and not the check-level precision will conclude the model did nothing, and it will be right about OSA and wrong about the decision.

**What would reverse the finding.** Four conditions, each with a test. If the audit sensitivity falls below 0.85 or specificity below 0.97, the observed precision gap between arms shrinks toward the audit error and the comparison is not resolvable at pilot scale (§9). If `t` rises to 150 s, capacity falls to 24 and the candidate's last sixteen ranks are negative-value; the rule must shorten, and the reported 20.18 USD becomes 10.01 USD at `K` = 24. If mid-day planogram resets exceed 2% of locations, the location key is not stable within the day and the task must drop to SKU level, which costs the walk-order benefit in §12. If the incumbent's nomination list is maintained more actively than assumed — say its hit rate is 22% rather than 30% on the nominated twelve — the incumbent's precision falls to (12×0.22 + 28×0.055)/40 = 0.1045, it destroys 1.76 USD/day, and the candidate's advantage widens to 25.6 USD/day rather than narrowing. That asymmetry is the useful part: the finding is robust to the incumbent being *worse* than assumed and fragile to the incumbent being much *better*.

**The framing that must not be lost.** The deliverable is not "a model with 25% precision". It is "a work list of forty locations per store-day, publishable by 05:45, capped by shift seconds, with a reason code per task and a replayable log", plus the arithmetic showing that this list beats the equal-minutes incumbent. The model is a component of the list. The list is the decision.

## 8. System/Solution Design

### 8a. Offline / Batch Path

```text
[per-store nightly extract]  ->  [availability gate]  ->  [as-of feature build]
        (02:00, run_id)              (fails closed)         (declared available_by)
                                                                     |
                                                                     v
[coefficient table]  ->  [score & rank]  ->  [capacity gate]  ->  [list header + tasks]
   (versioned)              (tie-break fixed)     (B/t truncation)      (published 05:45)
                                                     |                       |
                                                     v                       v
                                            [truncation record]     [append-only task log]
```

The batch path is the whole decision surface, and each box has a named failure that routes to the incumbent rule rather than to a degraded list. The availability gate runs a single query: for each feature, the maximum source modification timestamp in the snapshot partition, compared against the cut-off. If any feature exceeds it, the store-day publishes the incumbent list with reason code `FEATURE_LATE` and the offending feature name. This is deliberately blunt. A per-row repair would let a mostly-clean build through and reintroduce the leakage the gate exists to catch.

The capacity gate computes `K = floor(B/t)`, where `B` comes from the labour plan for that store-day and `t` from a per-store measured value that defaults to the district median. The truncation record stores the score of the last published task and the score of the first suppressed task, which is what makes "was the list cut at a sensible place" a queryable question rather than a matter of opinion. If the two scores are close, the cut is arbitrary and the reader should not read meaning into the fortieth task.

The reason code is a contractual part of the payload. Each task carries exactly one code from a closed set — `RECORD_ZERO`, `LONG_NO_RECEIPT`, `HIGH_VELOCITY_LOW_FACE`, `RECENT_AUDIT_GAP`, `PROMO_TROUGH`, `LOW_COVER` — chosen by the largest single contribution to the score, with ties resolved by the order the codes are listed. A closed set is a constraint, not a nicety: it keeps the handheld view to one line and it makes the override analysis in §9 a group-by instead of a text-mining exercise.

### 8b. Online / Serving Path

The store-day list is the only online artefact, and it is immutable once published. A mid-day change of circumstances — a late delivery, a planogram reset, a district push — does not re-score the list; it triggers a **revocation** for named tasks with a reason code. The distinction matters because re-scoring mid-day would use fields that were post-cut-off for the morning publication, and the log would then contain two lists produced under two different information sets with one published timestamp.

The handheld serves a cached copy of the list, so a connectivity drop at 09:00 does not stop work. Completion is recorded locally with a monotonic sequence number and synced when the connection returns; the server rejects any completion whose sequence precedes the revocation timestamp, and the conflict is surfaced to the duty manager rather than silently dropped. This is the one place where the design has real distributed-systems content, and it is where an early prototype lost three days of completions in the constructed walkthrough of §12.

Latency is not a meaningful constraint here. The list is computed once per store-day in a batch window of hours; a scoring job that takes four minutes rather than forty seconds changes nothing operationally. Saying so is useful because it removes a class of optimisation that would trade auditability for speed the decision cannot use.

### 8c. Data Contracts and Interfaces

| Interface | Producer | Consumer | Contract | Failure behaviour |
|---|---|---|---|---|
| Snapshot partition | extract batch | feature build | one directory per `store_id/business_date/run_id`; never mutated after write | build refuses to run on a mutated partition |
| Feature set | feature build | scorer | six named features, no nulls, declared `available_by` ≤ cut-off | store-day falls to incumbent with reason code |
| Score table | scorer | capacity gate | `(store, sku, loc, date, score, reason_code)`, score in [0,1] | gate refuses; incumbent published |
| List header | capacity gate | handheld | `K`, `B`, `t`, `cutoff_ts`, `model_version`, truncation scores | handheld shows incumbent label if header absent |
| Task log | handheld | metrics | append-only, idempotent on `(task_id, seq)` | replay from log; no reconstruction from cache |
| Audit feed | audit walk | metrics | `(store, sku, loc, date, gap_flag, auditor_id, walk_ts)` | precision suppressed, completion still reported |

The weakest contract in the table is the audit feed, and it is weak for a reason that no schema fixes: the audit and the check are both physical walks of the same shelf, and if the auditor and the associate walk at the same time they see the same thing. The protocol therefore fixes the audit window to start at 11:00 and forbids auditing a location that is on the day's published list, which converts the audit into an independent sample of *non-selected* locations — useful for estimating the base rate, useless for measuring precision. Precision must be measured by the associate's own finding, with the auditor's double-walk as a calibration of that finding rather than as an independent truth.

### 8d. Rollout, Reversibility, and Authority

Rollout is a shadow publication first: the list is generated, logged, and withheld from the handheld for ten store-days, during which the incumbent list is what the store actually works. Shadow mode answers two questions that no offline evaluation can: whether the list is producible every day without operator intervention, and whether the associate would have found the same gaps while doing something else. The second question is measured by asking the associate at end of shift whether any of the shadow tasks happened to be worked, which is imperfect and cheap.

Authority sits with the store operations manager for publication and with the duty manager for revocation. Neither can change the scoring model, the capacity formula, or the cut-off; those are owned by the analyst and change only with a version bump and a replay test. This split is the practical form of the auditability constraint: the people accountable for the store's labour can control what work is asked for, and cannot silently change the reason the work was chosen.

Reversibility has one hard requirement that is easy to under-design: revoking a task must not revoke the *record* that it was published. The log keeps both the publication and the revocation, with timestamps, so that a later analysis of "what did we ask for and what did we get" is not corrupted by the fact that some tasks were withdrawn. A design that deletes revoked tasks makes the completion-rate denominator unknowable, and completion rate is the guardrail that catches an unusable list before precision does.

## 9. Evaluation & Validation

### 9a. Metric definitions

| Metric | Numerator | Denominator | Population | Window | Comparator | Status |
|---|---|---|---|---|---|---|
| Precision@K | tasks on the list where the associate recorded a real gap | published tasks worked | eligible location-days, both arms, same capacity | 20 store-days per arm | incumbent equal-minutes list | executed (constructed) |
| Gaps reached/day | real gaps reached by worked tasks | store-days in window | same | same | incumbent | executed (constructed) |
| OSA (audited) | audited locations with a sellable unit present | audited locations | independent audit sample, excluding listed locations | same | incumbent | not-executed |
| Completion rate | tasks marked done | tasks published | all published lists | same | incumbent | executed (shadow) |
| Minutes per check | worked minutes logged | tasks completed | completed tasks only | same | design value 90 s | executed (shadow) |
| Override rate | tasks revoked or skipped with reason | tasks published | all published lists | same | none (new) | executed (shadow) |
| OSA percentage points | OSA(candidate) − OSA(comparator) | not applicable | store-days | 40 trading days | incumbent | not-executed |

**Guardrails.** Missing-source rate (share of store-days that fell to the incumbent rule, target below 2%), late-publication count (lists published after 05:45, target zero), replay mismatch count (target zero), and associate-reported unusable-task rate (target below 5%, measured by a two-question end-of-shift prompt).

**Uncertainty.** Every reported precision carries a store-clustered interval. With forty tasks per store-day and a within-store day-to-day intraclass correlation of 0.02, the design effect at twenty days per store is 1 + 19 × 0.02 = 1.38, so the naive binomial interval must be widened by √1.38 = 1.175. Concretely, a single-arm precision of 0.25 measured on 24 stores × 20 days = 480 store-days, 19,200 tasks, has a naive standard error of √(0.25 × 0.75 / 19200) = 0.00313 → wait, the correct grain for the interval is the store-day, not the task: 480 store-days at 40 tasks each give a per-store-day precision with standard error √(0.25×0.75/40) = 0.0685, and the arm-level standard error is 0.0685/√480 × √1.38 = 0.00368. The two-arm difference has standard error 0.00368 × √2 = 0.00520, and the minimum detectable difference at 80% power is 2.802 × 0.00520 = 0.01457, or 1.46 percentage points. The constructed difference is 25.29 − 12.85 = 12.44 points, so precision@K is resolvable with an enormous margin at this design size. The 262 store-days per arm needed for a 1.24-point effect confirms the same conclusion from the other direction.

### 9b. What the pilot can and cannot resolve

The OSA endpoint is a different story. Each store contributes one audited OSA estimate per trading day from a 40-location audit sample, so the per-store-day sampling standard error is √(0.031 × 0.969 / 40) = √0.000751 = 0.0274, or 2.74 percentage points. Twenty days per store reduce that to 2.74/√20 = 0.613 points per store. Adding a between-store heterogeneity term of 0.9 points gives a per-store standard deviation of √(0.613² + 0.9²) = √(0.376 + 0.810) = 1.089 points. With twelve stores per arm the difference has standard error 1.089 × √(2/12) = 0.445 points and a minimum detectable effect of 2.802 × 0.445 = 1.25 percentage points. The constructed treatment effect is 0.031 percentage points — forty times smaller than the pilot's resolving power. A twelve-store, twenty-day pilot cannot detect this intervention on OSA, and reporting "no significant OSA effect" from it would be a statement about the pilot's size, not about the intervention.

This is the evaluation-mismatch pattern in its purest form. The metric the organisation wants (OSA points) has a signal-to-noise ratio that requires roughly (1.25/0.031)² ≈ 1,626 times more store-days than the metric the decision actually moves (gaps reached per list). The right response is not to run a bigger pilot; it is to make the primary endpoint the quantity the decision controls, and to report OSA as a directional, underpowered secondary with its interval shown rather than its p-value.

### 9c. Executed versus not executed

Executed in the constructed walkthrough: the shadow publication for ten store-days, the completion and override instrumentation, the replay test, and the offline precision and gaps-reached comparison on the 480-store-day evaluation window. Not executed: any live OSA measurement, any randomised assignment to candidate versus incumbent at store level, any measurement of `phi`, any measurement of lost units on a gap day, and any labour-productivity effect of the extra walking. The `phi` of 0.88 and the 3.2 lost units/day are assumptions with plausible magnitudes, not estimates from this setting, and §6.1's break-even moves linearly in both.

**Validation design if it were to proceed.** Randomise at the store level, not the task level, because a task-level split lets the same associate work both arms in the same aisle and contaminates the walk time. Twelve stores per arm for twenty trading days, pre-register precision@K as primary and OSA as secondary, freeze the eligible population definition and the audit protocol before unblinding, and analyse on the intention-to-publish basis including store-days that fell back to the incumbent rule. The last point is the one that most often gets lost: excluding fallback store-days biases the estimate toward whichever arm had a cleaner pipeline.

Takeaway: measure the endpoint the decision moves, size the pilot to that endpoint, and report the organisation's preferred endpoint with its interval rather than its verdict.

## 10. Action Taken

No live action was taken; the case is illustrative and the store is constructed. The action that the analysis supports, and that the rest of this section specifies as a plan with owners and exit criteria, is a **shadow-then-limited pilot** with a pre-registered primary endpoint.

| Step | Action | Owner | Cadence | Exit criterion |
|---|---|---|---|---|
| 1 | Freeze eligible population, cut-off, capacity formula, and audit protocol in a one-page decision contract | analyst with store operations manager | once, before any build | all four parties sign; no later change without version bump |
| 2 | Build the availability gate and prove it catches the known-late field | data engineer | one sprint | gate rejects a seeded late-feature build with reason code |
| 3 | Reproduce the incumbent list mechanically, including the planogram-order extension to forty | analyst | one sprint | the reproduced list matches what the store actually worked for five consecutive days |
| 4 | Shadow-publish candidate lists for ten store-days, withholding from the handheld | store operations manager | ten trading days | zero late publications, replay matches on all ten, missing-source rate below 2% |
| 5 | Measure `t` per store by timing forty checks across the four aisle types | store operations manager | one week | median and interquartile time per aisle; `K` recomputed from measurement |
| 6 | Enable the list on the handheld for four stores, incumbent on the other eight | analyst | twenty trading days | completion above 85%, unusable-task reports below 5% |
| 7 | Analyse precision@K on the pre-registered window | analyst | once, at the end | interval reported; if the lower bound exceeds the incumbent's point estimate, proceed |

**What was explicitly not done.** No threshold was set on the score. No production scoring service was stood up. No supplier, planogram, or labour-budget change was requested. No associate-facing change was made before step 4, and step 4 by construction makes no associate-facing change at all.

**Stopping rules.** The pilot stops, and the incumbent rule becomes permanent, if any of the following hold: the missing-source rate exceeds 2% for five consecutive store-days; completion falls below 85% in the enabled stores after two coaching cycles; the measured `t` exceeds 130 s so that `K` falls below 28 and the candidate's marginal rank is no longer positive-value at the measured cost; or the audit's double-walk agreement implies sensitivity below 0.85, which would make the label too noisy to rank against.

The stopping rules are written before the pilot for the same reason the endpoint is: a rule invented after seeing the data is a rationalisation. The `t` > 130 s rule deserves a note, because it is the one that most plausibly fires. At 130 s, `K` = floor(3600/130) = 27 and `c` = 0.7222, so break-even precision is 0.7222/4.05504 = 17.81%; the candidate's rank-27 precision is between the 0.195 of ranks 25–28 and the 0.215 of ranks 21–24, so roughly 0.201, which still clears 17.81% but with only 2.3 points of margin. The decision is then genuinely close and the honest answer is to re-measure the time rather than to re-argue the model.

Takeaway: the plan is structured so that every step has an observable exit criterion and the two conditions that would end the programme are stated before the data arrives.

## 11. Outcome & Results

**Status: not-measured.** No live store ran this list, so there is no OSA outcome, no sales effect, and no labour-productivity result. What follows separates the arithmetic that was computed from the outcome that was not.

**Computed, constructed inputs.** At `K` = 40, `t` = 90 s, `c` = 0.50 USD, `V` = 4.05504 USD: candidate expected gaps reached 10.116/store-day against incumbent 5.140; candidate net 21.02 USD/store-day against incumbent 0.84 USD; incremental 20.18 USD/store-day, or 5,044 USD per store-year at 250 trading days, or about 302,700 USD across sixty stores at the same cadence. Every one of those figures inherits the uncertainty of `V`, and `V` is the weakest input: at 20% substitution the incremental figure falls to 12.14 USD/store-day, and at 30% to 8.04 USD.

| Scenario | `V` (USD) | break-even | candidate net (USD/day) | incumbent net (USD/day) | incremental (USD/day) |
|---|---|---|---|---|---|
| No substitution | 4.0550 | 12.33% | 21.02 | 0.84 | 20.18 |
| 10% substitution | 3.6495 | 13.70% | 16.92 | −1.24 | 18.16 |
| 20% substitution | 3.2440 | 15.41% | 12.82 | −3.33 | 16.14 |
| 30% substitution | 2.8385 | 17.62% | 8.72 | −5.41 | 14.13 |

The table's shape is the finding: substitution makes the incumbent unprofitable before it makes the candidate unprofitable, because the incumbent's precision is only 0.52 points above the no-substitution break-even. That is a fragile baseline, and a programme that measures only the candidate's value will mis-state the decision.

**Computed, but only as a power statement.** The OSA endpoint at twelve stores per arm and twenty days has a minimum detectable effect of 1.25 percentage points against a constructed effect of 0.031 points. This is a computed statement about the design, not an outcome, and it is the reason OSA is not the primary endpoint.

**Not measured, and not to be implied.** The 88% fix probability; the 3.2 lost units per gap day; the 1.44 USD unit margin; the 30% and 5.5% incumbent hit rates; the decile hit-rate profile; the 90-second check time; the 0.92/0.985 audit error rates. All are constructed inputs chosen to be plausible, and all are published in §5 and §6 so that a reader who disagrees can substitute their own and recompute. The arithmetic is verified; the magnitudes are not.

**What a reader should take as real.** Two things survive the illustrative label. First, the structural result that break-even precision `c/V` for a shelf check is low — around one in eight at plausible grocery economics — which means almost any ranker with signal clears it and the binding question is where to stop. Second, the structural result that OSA is an insensitive endpoint for a forty-check intervention in a two-thousand-location store, because the intervention reaches 10 of 71 daily gaps. Both are consequences of the arithmetic, not of the particular numbers, and both are checkable in a reader's own setting with three inputs.

Takeaway: report the arithmetic with its inputs visible and the outcome as not-measured; the reader can then decide whether the inputs resemble their store.
