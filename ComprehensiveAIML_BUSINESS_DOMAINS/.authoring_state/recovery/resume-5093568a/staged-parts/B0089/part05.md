## 9. Complexity & Cost Analysis

Diagnosis is not free, and the cost has three components that are easy to conflate: compute, human attention, and the value of the decision being delayed.

### 9.1 Computational cost

The reconciliation in D-01 is a single pass with a keyed hash, so it is linear in the number of records and constant in the number of keys held in memory at once, provided the extract is sorted or partitioned by key.

```text
keyed reconciliation        O(N) time,  O(K) memory
sort-merge comparison       O(N log N) time, O(N) memory in the worst case
pairwise cross-join check   O(N^2) time, unacceptable beyond ~10^5 records
clustered variance estimate  O(G) after one pass of per-cluster sums
WHERE: N = records in the window, K = distinct keys, G = clusters (stores)
```

The numbers that matter are smaller than they look. A nine-store estate at 120 SKUs over 14 days is 15,120 cells, or roughly 15,000 rows of about 400 bytes — 6.0 MB before indexes. The sort-merge alternative on that volume is indistinguishable from the keyed pass in wall time, so the cost argument does not decide the method. What decides it is that the keyed pass can be written to emit the four D-01 terms separately, while a sort-merge written for equality testing cannot without a second pass.

Cost only becomes a design constraint at two points. The first is a full-history re-reconciliation, where N grows by a factor of the retention period; there, partitioning by month keeps memory bounded and is worth the extra code. The second is per-event streaming reconciliation, where the keyed state must be evicted on a maturity horizon or the memory grows without bound.

### 9.2 Human attention

Manual review is the dominant cost in every entry that routes work to a person, and it is routinely underestimated because the estimate counts first touch and forgets rework.

```text
review_minutes = n_tasks x (first_touch + rework_rate x rework_minutes)
review_cost    = review_minutes/60 x hourly_rate
WHERE:
  n_tasks        tasks routed to human review            [tasks]
  first_touch    minutes to decide one task              [min]
  rework_rate    share of tasks that need a second look  [fraction]
  rework_minutes minutes for the second look             [min]
  hourly_rate    fully loaded cost per reviewer hour     [currency/h]
NUMERICAL EXAMPLE:
  84 cells audited at 1.5 min each = 126 min = 2.1 h
  at $17.40/h -> $36.54 for the audit itself
  with 20% rework at 3 min: 84 x (1.5 + 0.2 x 3) = 176.4 min = $51.16
INTERPRETATION: rework adds 40% to the apparent cost and is invisible in a
first-touch estimate.
```

The 40% figure is the useful output. Any business case for an automated check that quotes only first-touch minutes will understate the manual baseline it replaces, and therefore understate its own saving — a rare error direction that still produces bad decisions, because it makes the automation look unnecessary.

Queue age is a second, separate cost. A task that waits three days has usually been overtaken by events, so the reviewer re-derives context that the original analyst already had. The correct comparison for a capacity decision is not tasks per hour but cost per correctly closed task, which includes the share of tasks that expire before anyone touches them.

### 9.3 Cost of delay

The third component is the value lost while the decision is unresolved, and it is the only one that justifies rushing.

```text
cost_of_delay_per_day = affected_units x margin_per_unit x carry_days
WHERE:
  affected_units  cells or units subject to the defect   [units]
  margin_per_unit gross margin at risk per unit          [currency/unit]
  carry_days      days the defect persists               [days]
NUMERICAL EXAMPLE:
  318 phantom cells x 1.4 units each x $1.10 margin = $489.72 per day
  over a 14-day detection lag -> $6,856 of margin at risk in one store-window
  against an audit cost of $51 -> the audit is cheap, the lag is expensive
INTERPRETATION: detection latency, not detection accuracy, is the dominant term.
```

This is the arithmetic behind the ordering rule in §2. A crude check applied within a day beats an exact check applied within a fortnight, and the comparison is not close.

### 9.4 Sampling economics

When the check cannot cover the population, the sample size is set by the precision needed, and the precision is set by the decision.

```text
n = z^2 x p x (1 - p) / e^2                 simple random sample
n_cluster = n x DEFF                        store-clustered design
WHERE:
  p  expected rate        [fraction]
  e  desired half-width    [fraction]
  z  1.96 for 95% confidence
NUMERICAL EXAMPLE:
  p = 0.93, e = 0.01  -> n = 3.8416 x 0.93 x 0.07 / 0.0001 = 2,501 cells
  p = 0.93, e = 0.005 -> n = 10,004 cells
  with DEFF 2.428 (120 cells per store, rho 0.012) -> 6,073 cells, ~51 stores
INTERPRETATION: halving the target precision quadruples the sample, and
clustering multiplies it again, so precision targets must be negotiated.
```

The clustered figure is the one that usually kills a proposed audit. Fifty-one stores is not an audit; it is a programme. The practical alternative is to accept a wider half-width, or to stratify by store and audit only the stores whose own rate is furthest from the estate, which trades unbiasedness for decision relevance.

### 9.5 Cost of the wrong correction

A correction applied to the wrong cause has a cost that must be counted against the correction's apparent benefit.

```text
net_value = avoided_loss - correction_cost - regression_risk x blast_radius
WHERE:
  avoided_loss     margin protected by the fix        [currency]
  correction_cost  engineering plus review            [currency]
  regression_risk  probability the fix breaks another flow [fraction]
  blast_radius     cost if it does                     [currency]
NUMERICAL EXAMPLE:
  avoided_loss $6,856; correction_cost $900
  regression_risk 0.10; blast_radius $40,000
  net = 6,856 - 900 - 0.10 x 40,000 = $1,956
  at regression_risk 0.25 -> net = -$4,044, i.e. a loss
INTERPRETATION: a 15 pp rise in regression risk flips the decision, so the
evidence requirement scales with blast radius, not with the benefit.
```

The blast radius here is not the same as the affected population. A shared time-zone normalisation touches every store, while a planogram fix touches one aisle. That is why §7.4 pairs a financial posting with a 100% reconciliation and a coaching conversation with a per-store partition.

Takeaway: spend the diagnostic effort where detection latency and blast radius are large, and accept coarse checks everywhere else.

## 10. Key Theorems / Results Reference

The results below are the load-bearing statements of the file. Each is stated as an invariant or a bound with the condition under which it holds, because a result quoted outside its condition is the most common source of confident error in this domain.

### 10.1 Partition invariant

**Statement.** For a frozen extract, the four terms of D-01 partition the discrepancy exactly: `delta = d_dup + d_late + d_key + d_scope`, and the residual is zero.

**Why it holds.** Every record in the symmetric difference is excluded from one side for exactly one of four reasons. The reasons are exhaustive by construction of the extract (a record is either duplicated, outside the time rule, keyed differently, or scoped differently) and mutually exclusive as long as the classification is applied in a fixed order.

**What breaks it.** Classification order. A record that is both a duplicate and late will be assigned to whichever rule runs first, and if the two passes use different extracts the counts will not sum. The invariant is therefore a property of a *procedure*, not of the data alone.

**Use.** A non-zero residual is a contradiction: it proves that two causes were assigned to one record. That makes the residual the cheapest available test of whether the attribution can be trusted at all.

### 10.2 Idempotency invariant

**Statement.** A replay is safe if and only if applying it twice produces the same set of effective actions as applying it once.

**Why it holds.** The decision key is the unit of effect. If the writer is keyed on the decision key and the write is an upsert, a second application is a no-op by construction.

**What breaks it.** Any writer that appends instead of upserting, or that keys on a surrogate identifier minted per attempt. This is the mechanism behind CI-25, where offline replay duplicates observations because the key includes a device-local attempt number.

**Use.** The verification is a count comparison: effective actions before, plus the replay, must equal effective actions after. An inequality of one means the writer is not idempotent, regardless of what the code review said.

### 10.3 Censoring bound

**Statement.** For a right-censored duration with cut-off `t_cut`, the naive mean over resolved cases is a lower bound on the true mean, and `(sum_observed + k x t_cut)/n_total` is an upper bound whenever the true maximum duration does not exceed `t_cut`.

**Why it holds.** Every censored case has true duration at least equal to its observed duration up to the cut-off, which is at least zero and at most `t_cut`. Substituting the extremes gives the two bounds.

**What breaks it.** `t_cut` shorter than some true durations, which makes the upper bound invalid rather than merely loose. The bound is therefore only usable once the analyst has confirmed that nothing in the population can exceed the cut-off — usually by looking at the maximum observed duration in a prior complete window.

### 10.4 Cluster variance bound

**Statement.** For a cluster-randomised design, the variance of an arm mean is at least `sigma^2/n` and at most `sigma^2/n x DEFF`, with `DEFF = 1 + (m-1) rho`. Equality at the lower end requires `rho = 0`.

**Why it holds.** Positive intra-cluster correlation means observations inside a cluster carry redundant information; the design effect is exactly the redundancy multiplier under an exchangeable correlation structure.

**What breaks it.** Negative correlation, which can occur when a fixed-capacity store compensates across SKUs, and non-exchangeable structures such as a planogram change applied mid-window to half the stores. In both cases the scalar `rho` is the wrong model and the bound does not apply.

### 10.5 Little's law as a diagnostic

**Statement.** In a stable queue, the average number in system equals arrival rate times average time in system: `L = lambda x W`.

**Why it holds.** It is a conservation identity on counts, independent of the arrival and service distributions.

**What breaks it.** Instability. Once `rho >= 1` the averages do not exist, and a reported queue length is simply the length at the moment of measurement. This is why the 61-arrivals case in CI-15 has no finite answer and why quoting one is a category error.

**Use.** Little's law is the cheapest consistency check available on queue data. If arrivals, measured wait, and measured queue length do not satisfy the identity within sampling error, at least one of the three is measured over a different population.

### 10.6 Definition stability

**Statement.** A metric series is comparable across periods only if the key, the population, the time rule, and the unit are identical in both periods.

**Why it holds.** A rate is a ratio over a population. Changing any element of the ratio's definition changes the quantity being measured, and no amount of arithmetic can restore comparability.

**What breaks it.** Every silent redefinition in the catalogue: CI-12, CI-21, CI-27, CI-28. The dangerous property is that the defective series is internally consistent — the arithmetic is right and the comparison is meaningless.

**Use.** The check is a registry lookup, not a data check. That is the point: some defects cannot be found in the data at all, and the only control is a versioned definition register that consumers are required to read.

### 10.7 Result selection guide

| Question | Result | Fails when |
|---|---|---|
| Do my counts reconcile? | 10.1 partition | Two causes assigned to one record |
| Can I safely replay? | 10.2 idempotency | Writer appends instead of upserts |
| Is my lead time biased? | 10.3 censoring bound | Cut-off shorter than true maximum |
| Is my significance real? | 10.4 cluster bound | Negative or non-exchangeable correlation |
| Do my queue metrics agree? | 10.5 Little's law | System unstable (rho >= 1) |
| Can I compare periods? | 10.6 definition stability | Any definition element changed |

Takeaway: state the result's validity condition in the same sentence as the result, or the next reader will apply it outside it.

## 11. Interview / Exam Questions

Thirteen questions, each with its own answer. Questions 1, 2, 4, 5, 7, 9, 11, and 13 require computation or diagnosis; the rest test selection and judgement. Answers are written to be marked, not memorised.

### Q1. A store reports 95.3% system availability and 93.2% shelf availability. How large is the phantom rate, and why is it not 2.1%?

**Answer.** The phantom rate is `(14,410 - 14,092)/14,410 = 2.207%`. The 2.1 pp figure is `318/15,120 = 2.103 pp`, which divides by all cells including those with no stock in either system. The rates differ because the denominators differ: one is believed-stock cells, the other is all cells. Quote 2.21% when the question is "how much of my believed stock is unreachable" and 2.10 pp when the question is "how many points of availability do I lose".

**Reasoning.** Percentage and percentage-point differences over different bases are the most frequent numeric confusion in retail reporting, and the two figures here differ by enough to matter at the third significant figure.

**Follow-up.** If the store has 8% of cells out of stock in both systems, which figure moves and by how much? The phantom rate is unaffected; the pp gap is diluted because the common-stock-out cells stay in the denominator.

**Red flag.** Reporting 2.1% by dividing 318 by 15,120 and calling it a rate — that is a share of cells, not a share of stock.

**Scoring.** Full marks require both numbers, the two denominators, and a statement of when to use each.

### Q2. A daily report shows a 20% swing between adjacent days that vanishes at the weekly level. What is the most likely mechanism and what single check confirms it?

**Answer.** A total-preserving time-shift, most often a time-zone offset applied at the extractor rather than at capture. The check is to sum a seven-day block spanning the boundary: if the block reconciles to the control total while the days do not, mass has moved rather than disappeared. A five-hour offset on a day transacting $18,400 moves $3,833 of value, or 20.8% of the day.

**Reasoning.** Data loss reduces totals, duplication raises them, and only a time shift conserves them. The conservation property is the discriminating signal.

**Follow-up.** How would the answer change if the weekly block also failed to reconcile? Then mass is not conserved and the mechanism is a loss or a duplication defect, so the check moves to the identity pass.

**Red flag.** Recommending a backfill of the prior day, which destroys the evidence of the swap without correcting the extractor.

**Scoring.** Award full marks only for the conservation argument; naming "time zones" without the weekly-block test is a guess.

### Q3. Two teams quote different availability numbers, each correct within its own definition. What is the diagnostic order?

**Answer.** Do not reconcile arithmetic; reconcile populations. First compare the SKU set, then the store set, then the time window, then the location filter (shelf versus in-building), then the substitution treatment. In the catalogue, a 214-cell substitution credit moves a 93.20% rate to 94.62%, and a 38-of-120 assortment change moves the population by 31.7% while leaving the count unchanged.

**Reasoning.** When both numbers are internally consistent, the defect is definitional by construction, and record-level checks will find nothing because there is nothing wrong at record level.

**Follow-up.** Which of the five elements is hardest to audit? The substitution treatment, because it is usually applied inside a query rather than stored, so there is no artifact to compare.

**Red flag.** Averaging the two figures, which produces a number that matches neither definition.

**Scoring.** The five elements in order, plus one quantitative example of the divergence.

### Q4. A queue accepts 480 tasks into a 400-task capacity. What happens, and what must be logged?

**Answer.** Eighty tasks are dropped, 16.7% of arrivals, and the visible backlog pins at exactly 400. The drop count must be logged with the priority tier, because an unlogged drop is indistinguishable from a data-quality failure downstream. At an arrival rate of 46/h against 52/h service the queue is stable at rho 0.885 with an 8.85-minute wait; the capacity problem is a separate axis from the stability problem.

**Reasoning.** Capacity truncation and queue instability both produce backlog, but only truncation produces a plateau at a round number. The round number is the clue.

**Follow-up.** What alert threshold? Warn at 85% utilisation for two consecutive intervals, because the queue length hyperbola starts to steepen well before saturation.

**Red flag.** Adding reviewers without a priority rule, which changes throughput but leaves the drop tier undefined.

**Scoring.** Both the 80-task arithmetic and the requirement to log the dropped tier.

### Q5. Thirty of thirty-four orders resolve with a mean of 2.9 days and four remain open at a 6-day cut-off. What is the defensible statement about lead time?

**Answer.** The mean is between 2.56 and 3.27 days. The naive figure treating open orders as zero is `(30 x 2.9)/34 = 2.559`; treating them at the cut-off gives `(87.0 + 24)/34 = 3.265`. Report the bound with the open count: "mean 2.6–3.3 days, 4 of 34 unresolved". The naive figure understates by 27.6% and always in that direction.

**Reasoning.** Right-censoring biases downward because the cases that remain open are open precisely because they are slow.

**Follow-up.** When is the naive mean acceptable? When the open count is zero by construction, i.e. the window exceeds the maximum possible lead time.

**Red flag.** Reporting 2.9 days, which uses only the resolved cases and silently drops four orders from the denominator.

**Scoring.** Both bounds, the direction of bias, and the open count in the reported statement.

### Q6. Why is a store-level pilot with nine stores and 15,120 cells not a 15,120-observation experiment?

**Answer.** Because treatment is assigned at store level, so the store is the independent unit. With 1,680 cells per store and an assumed intra-cluster correlation of 0.00125, `DEFF = 1 + 1,679 x 0.00125 = 3.099` and the effective sample is 4,879 cells. A naive z of 2.73 becomes 1.55 after adjustment, which is not significant.

**Reasoning.** Cells within a store share unobserved influences — the same planogram, the same staff, the same delivery schedule — so they carry redundant information.

**Follow-up.** What raises power? More stores, not more days. Adding days raises `n` without raising the cluster count, and under D-06 the information gain is small once within-store correlation dominates.

**Red flag.** Quoting a cell-level p-value for a store-level intervention.

**Scoring.** The design effect computation and the correct remedy.

### Q7. A category manager's audit says 95.1% and the pipeline says 92.98%. You find a 2.38% false-negative rate and a 3.57% false-positive rate in the audit. What do you conclude?

**Answer.** The audit overstates by `(3 - 2)/84 = 1.19%`, so the bias-adjusted shelf count is `1,562 x 83/84 = 1,543.4` cells and adjusted OSA is `1,543.4/1,680 = 91.87%`. Correcting the audit moves the estimate *down* by 1.11 pp, widening the gap. The 95.1% figure still needs a population explanation, and the remaining candidate is the scope difference — 16 cells inside the manager's list and outside the 120-SKU set.

**Reasoning.** Observation error must be corrected before any comparison, and the direction of the correction is determined by the net of the two error rates, not by which error sounds worse.

**Follow-up.** What sample size would detect a 2 pp residual? About 2,501 cells at 1 pp precision, or more once clustering is accounted for.

**Red flag.** Adjusting only for false negatives, which is the direction that flatters the pipeline.

**Scoring.** Net bias, adjusted rate, and the recognition that the correction widens rather than closes the gap.

### Q8. When is it correct to keep a defective metric in production?

**Answer.** When the correction changes the population rather than the arithmetic, and consumers have a contractual or regulatory reliance on the current number. In that case run both definitions in parallel for one full cycle, publish both with version stamps, and switch consumers at a stated date. The catalogue treats a redefinition as irreversible in the sense that consumers cannot silently revert.

**Reasoning.** A metric that changes definition without a version bump converts a data problem into a trust problem, and the second is more expensive to repair.

**Follow-up.** What is the minimum dual-run length? One complete cycle of the longest downstream process, so that every consumer sees at least one full period under both definitions.

**Red flag.** Shipping the new definition and explaining it afterwards.

**Scoring.** The parallel-run requirement and the explicit version stamp.

### Q9. A store shows a 7.5 pp spread across nine stores with a standard deviation of 2.51 pp. Can you rank the stores?

**Answer.** No, not usefully. A three-versus-six store comparison has a standard error of `2.51 x sqrt(1/3 + 1/6) = 1.78 pp`, so the minimum detectable difference at 80% power is `(1.96 + 0.84) x 1.78 = 4.98 pp`. A two-versus-seven split is worse at 5.64 pp. The spread of 7.5 pp is real but individual rankings inside it are not resolvable with nine clusters.

**Reasoning.** Between-store variance, not cell count, sets the resolution, and with nine stores the standard error of any contrast is a large fraction of the spread.

**Follow-up.** What is the right chart? Paired change against each store's own prior period, which removes the between-store level entirely.

**Red flag.** Publishing a league table and acting on the bottom three.

**Scoring.** The standard error computation and the paired-change alternative.

### Q10. Give a case where zero discrepancy between two systems is not evidence of correctness.

**Answer.** When both systems read the same upstream snapshot. In the worked example the ledger and the dashboard both report 14,410 cells with zero discrepancy between them, while the shelf audit finds 14,092 — a 318-cell gap. Agreement between two descendants of one source measures the source's self-consistency, not its accuracy.

**Reasoning.** Independence of the control total is a structural property, not a statistical one. Two correlated measures cannot bound a shared error.

**Follow-up.** What would make the control valid? A physical count or a second, independently operated system — anything whose errors are not produced by the same transformation.

**Red flag.** Citing a zero reconciliation as proof that the pipeline is correct.

**Scoring.** The shared-ancestor mechanism plus a named independent alternative.

### Q11. A shelf camera's observation rate falls 12% over six weeks on one 41-SKU aisle and then resets after maintenance. Genuine decline or drift?

**Answer.** Drift, because the fall is present in cells that were never out of stock and recovers without a replenishment event. The effect is `41 x 0.12 = 4.9` cells continuously. The discriminator against lighting (CI-17) is time-of-day invariance: drift affects morning and afternoon equally, glare affects only the afternoon.

**Reasoning.** A ramp that resets at a maintenance event is a property of the instrument; a decline that resets is a property of the stock, and stock does not reset without a delivery.

**Follow-up.** What is the weekly control? A fixed reference target in frame, checked for detection rate; a fall in target detection precedes any shelf effect.

**Red flag.** Re-calibrating without first capturing a frame from the drifted period, which destroys the evidence needed to quantify the affected window.

**Scoring.** The ramp-versus-step signature and the time-of-day discriminator against glare.

### Q12. What makes a discriminating check discriminating?

**Answer.** It must predict a *different* observable under each competing cause, and the prediction must be checkable in the frozen extract. "Segment by store and look" is not a check; "the weekly total reconciles while the daily totals do not" is, because it is false under loss and true under a time shift.

**Reasoning.** A check that both causes would satisfy reduces confidence in nothing and costs an hour.

**Follow-up.** What is the failure mode of a check that predicts the same signal under two causes? It confirms the pair without separating them, and the analyst then picks the more familiar cause — a bias the catalogue's competing-cause lists exist to counter.

**Red flag.** Proposing to "monitor the model" or "add logging" as a discriminating check.

**Scoring.** The requirement of distinct predicted signals, with one concrete example.

### Q13. You have 900 of 1,240 cells failing tolerance. What is the first question, and why is it not "which cells"?

**Answer.** The first question is whether the tolerance itself is right for the SKU class. `900/1,240 = 72.6%` is too high a failure rate for a process defect, which typically hits a minority of cells, and is characteristic of a tolerance set in the wrong unit — cases against eaches, or a percentage applied to a count of one. A cell holding a single unit has a 0% or 100% error and can never meet a 5% tolerance.

**Reasoning.** Failure-rate magnitude is itself diagnostic. A defect concentrated in a mechanism affects few cells; a defect in the rule affects most of them.

**Follow-up.** What tolerance is defensible for single-unit cells? An absolute one, typically zero for high-value items and one unit for low-value ones, which is why tolerance must be specified per class rather than per estate.

**Red flag.** Opening a store-level investigation on a 72.6% failure rate before checking the rule.

**Scoring.** The failure-rate-as-diagnostic argument and the per-class tolerance conclusion.

## 12. Quick Reference Card

| Element | Rule | Number to remember |
|---|---|---|
| First action | Freeze the window, preserve raw evidence, stop only the affected action | Never backfill a past decision |
| Diagnostic order | Identity, then time, then capacity, then definition | Residual must be zero before attribution |
| Discrepancy identity | delta = duplicates + late + key + scope | 54 = 11 + 0 + 9 + 34 in the worked case |
| Phantom rate | (system - shelf)/system, never over all cells | 318/14,410 = 2.21%, not 2.10% |
| Substitution | Additive credit, capped at one per cell | 214 credits = +1.42 pp, three treatments span 2.83 pp |
| Queue stability | rho = lambda/mu; design at or below 0.85 | rho 0.885 gives 8.85 min; rho 1.173 has no finite wait |
| Capacity overflow | dropped = max(0, A - K), and it must be logged | 480 into 400 = 80 dropped, 16.7% |
| Clustering | DEFF = 1 + (m-1) rho; report n_eff | DEFF 3.099 turns z 2.73 into 1.55 |
| Censoring | Report the bound and the open count | 2.56 to 3.27 days with 4 of 34 open |
| Verification | Independent control total, stated precision | 2,501 cells for 1 pp |
| Escalate when | A customer promise, an authorisation boundary, or a financial posting cannot be reconciled | Do not resolve by judgement |
| Definition change | Version stamp plus one full dual-run cycle | No silent redefinition, ever |
