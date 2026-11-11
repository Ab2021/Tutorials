---
title: "25 Common Issues"
topic: "Common Issues"
collection: "retail-01"
level: "advanced"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
file_id: "04"
status: "draft"
version: "2.0"
review_trigger: "Any change to POS, WMS, planogram, footfall, or labour-export interfaces"
notation_style: "plain-text"
prerequisites: ["collection sheets 01-03"]
pattern_tags: ["diagnostics", "retail-operations", "point-in-time"]
source_status: "OPEN"
owner_role: "retail operations analyst"
source_index: "INDEX.md"
---

# 25 Common Issues

> **Purpose:** Give a store-operations analyst a diagnosis catalogue for the failures that recur when shelf, labour, traffic, replenishment, and shrink data are joined into one decision.
> **Evidence:** Every entry in this file is HYPOTHETICAL teaching guidance. No entry describes an observed incident at a named retailer; every count in this file is constructed and labelled ILLUSTRATIVE.
> **Reader task:** identify a competing cause, run one bounded discriminating check, apply the narrowest correction that preserves evidence, then verify against an independent control total.
> **Scope:** store-level operating data only. Corporate finance, legal determination, and employment-law questions are out of scope and are handed off rather than answered.
> **Key outcome:** after working through §8 and §11 you should be able to name the failing boundary, the check that separates it from its nearest lookalike, and the verification that proves recovery.

## 1. File Header

This is the troubleshooting artifact for the Store Operations and Merchandising unit. Sheets `01_Store_KPI_Tree.md`, `02_Shelf_Availability_Metric_Card.md`, and `03_Field_Test_Checklist.md` teach how to compute a valid store metric. This file assumes those skills and teaches the inverse operation: given a number that looks wrong, decide which part of the computation broke.

The unit of diagnosis is the **decision contract**, not the dashboard. A decision contract is the tuple of key, effective-time rule, eligible population, action, capacity, and owner that an automated or manual action was taken under. Almost every issue below is a statement that one element of that tuple silently changed while the others stayed fixed.

Three properties make store data unusually prone to this class of failure, and they explain why the catalogue is heavy on time and identity rather than on modelling:

- Store events are produced by devices (tills, hand-helds, shelf cameras, door counters) whose clocks and firmware versions are not centrally controlled at the moment of capture.
- Physical state changes (a case moved from backroom to shelf) are often recorded only at the next scan, so the observation lag is a property of the process and not of the network.
- The same physical quantity appears in several systems at different granularities — cases, eaches, pallets, weight — and the join between them is where unit errors enter.

Each issue entry follows one shape so that a reader can scan the catalogue quickly: identifier and title, category, the environment where it appears, the symptom as it reaches a dashboard or queue, at least two competing causes, one discriminating check with an expected signal for each branch, a correction, a prevention control, and a verification test. Entries never propose a root cause without naming what the competing explanation would have looked like.

The category vocabulary is fixed to four values and is deliberate:

| Category | Meaning | Typical giveaway |
|---|---|---|
| VIOLATED_ASSUMPTION | A stated precondition of the metric no longer holds | Metric is arithmetically correct but answers a different question |
| NUMERICAL_EDGE | Rounding, conversion, censoring, or boundary handling | Discrepancy scales with magnitude or appears at a cut-off |
| SCALE_FAILURE | Volume, capacity, or cluster structure exceeds the design | Error grows when load or store count grows |
| CONCEPT_CONFUSION | Two different quantities share a name or a column | Two teams disagree while both are internally consistent |

The catalogue is not a severity ranking. A CONCEPT_CONFUSION entry such as treating substitution as availability can move a headline number by more than a percentage point, while a NUMERICAL_EDGE entry such as half-hour shift rounding can look trivial and still distort every productivity ratio derived from it. Severity is decided per store per period by the reconciliation in §5, not by the category label.

Finally, a warning about the word "issue". The entries describe *mechanisms*, not blame. A planogram ambiguity is usually a publication-timing design choice, not an individual's mistake. Diagnosis that ends in a person's name is unrepeatable and therefore useless as a control; diagnosis that ends in a contract clause can be automated.

## 2. Concept Map / Mental Model

```text
symptom observed
      |
      v
[ freeze window + preserve raw evidence ]      <- always first, always reversible
      |
      v
[ locate the boundary that was crossed ]
      |            |             |            |
   identity      time         capacity     definition
   (key)      (effective)     (queue)      (semantics)
      |            |             |            |
      v            v             v            v
  duplicate     lag > eps     dropped      two metrics
  /reuse        /wrong tz     lowest       one name
      |            |             |            |
      +------------+-------------+------------+
                   |
                   v
     [ one discriminating check with a predicted signal ]
                   |
                   v
     [ containment: baseline or authorised manual review ]
                   |
                   v
     [ correction + version stamp + eligible-only replay ]
                   |
                   v
     [ verification against an independent control total ]
```

The left-to-right order in the lower half of the diagram is not arbitrary. Identity checks are cheapest and must run first because a duplicate key will also produce a time symptom and a capacity symptom, and repairing time or capacity first will hide the duplicate. A practitioner who starts with the most sophisticated explanation almost always ends by re-running the identity check anyway.

The four boundaries have distinct signatures, and the signature is what makes the check discriminating rather than merely suggestive:

- An **identity** break produces a discrepancy whose size equals a whole group of records and whose duplicates carry the same business attributes but different surrogate keys.
- A **time** break produces a discrepancy confined to the edges of the window; the interior of the window reconciles, the first and last hours do not.
- A **capacity** break produces a discrepancy that depends on load, so it disappears on quiet days and reappears on promotion days with the same underlying data quality.
- A **definition** break produces no discrepancy at the record level at all: both systems agree on every row and still report different totals, because the totals were computed over different populations.

The mental model also fixes the order of containment. Containment is deliberately narrower than correction: it stops the automated action, preserves the evidence, and routes affected work to a documented fallback. It never rewrites history. The reason is evidential, not procedural — a backfilled decision destroys the only record of what the operator could have known at the time, and every later question about the incident becomes unanswerable.

Two invariants hold throughout the catalogue and are restated in §10 in testable form. First, the reconciliation partition must be mutually exclusive and exhaustive over the declared eligible population, so that no record can be counted in two causes and no record can fall out of all of them. Second, any replay must be idempotent on the decision key, so that recovery cannot create a second effective action for the same key and window.

Takeaway: classify the boundary before choosing a tool, because the boundary determines which check can produce a contradiction.

## 3. Foundational Definitions

The definitions below are the vocabulary the rest of the file uses. Each is stated so that a reader can decide whether a given dataset satisfies it, not merely recognise the word.

| Term | Working definition | Test that it holds |
|---|---|---|
| Decision key | Minimal tuple that identifies one action: store, SKU, date, and action type in this unit | Two rows with the same key must never both be effective |
| Event time | When the physical event happened, in the store's local clock | Reconstructable from a device log, not from a batch timestamp |
| Processing time | When the platform learned about the event | Always later than or equal to event time; a negative lag is a defect |
| Effective time | Earliest moment a fact is permitted to change a decision | Set by policy, usually the next planning cycle boundary |
| Maturity horizon | Elapsed time after which an outcome is treated as complete | Stated in days; outcomes past it are not re-opened |
| Cell | One store-SKU-day combination; the atomic unit of availability | Count equals stores x SKUs x days in the window |
| On-shelf availability (OSA) | Share of cells where at least one sellable unit was physically on the shelf | Requires a shelf observation, not a system on-hand |
| System availability | Share of cells where the inventory record showed positive on-hand | Computed from the stock ledger alone |
| Phantom inventory | Cells where the ledger says stock exists and the shelf says otherwise | Difference of the two rates above, in cells |
| Substitution | A different sellable item satisfying the same basket need | Counted in an adjusted numerator, never in the base OSA |
| Tolerance | Allowed absolute difference between counted and recorded units | SKU-class specific; usually a percentage of units |
| Censoring | An observation that is incomplete because the window closed first | Detected when the open count is non-zero at cut-off |
| Fill rate | Delivered order lines divided by ordered lines in the window | Denominator must include lines still open at cut-off |
| Replay | Re-executing a decision for records that were wrongly excluded | Permitted only when the key is not already effective |
| Idempotency | Property that repeating an operation produces no additional effect | Verified by comparing effective-action counts before and after |
| Cluster | A group whose members share unobserved influences, here a store | Analysis unit when treatment is assigned at store level |
| Design effect | Multiplier on variance caused by clustering | At least 1; a value of 84 means naive intervals are meaningless |
| Control total | An independent count used to verify reconciliation | Must not be derived from the same upstream extract |
| Containment | Reversible action that limits impact and preserves evidence | Never includes deleting or overwriting source records |
| Provenance | Whether a claim is OBSERVED, REPRODUCED, or HYPOTHETICAL | Separate axis from the failure category |

Two of these definitions carry most of the diagnostic weight and deserve expansion.

**Cell.** The cell count is the denominator of every availability statistic in this file, and it is also the first thing that silently changes. A trial that begins with 120 SKUs and ends with 120 SKUs has not necessarily kept a stable denominator, because the SKU set itself may have rotated. When 38 of 120 listed SKUs are newly introduced and 38 others are delisted, the cell count is unchanged while 31.7% of the population is different. Any comparison across that boundary is a comparison of two different questions.

**Effective time.** Practitioners often conflate effective time with event time and then wonder why a correct backfill looks like a wrong decision. The distinction is operational: a delivery scanned at 21:40 on Monday has an event time of Monday evening, a processing time of Monday night, and an effective time of Tuesday's replenishment run. A decision built on Tuesday morning may lawfully use it; a decision built on Monday evening may not, no matter how accurate the record later became. The lag between event and effective time is a policy parameter, and the defect is not that the lag exists but that it is undocumented.

The remaining definitions are covered where they are used. Tolerance is specified per SKU class in §4.3, censoring is derived in §4.4, and design effect is derived in §4.3 because it changes the interpretation of every experiment-based entry in §8.

Takeaway: a term is only defined usefully when the sheet also states the check that shows the term applies.

## 4. Core Formulas with Full Derivation

This section builds the arithmetic that the catalogue relies on. Each result is presented in five parts: the formula, the symbols and where they come from, the constraint that must hold for the result to be valid, a numerical example with intermediate values, and the interpretation that follows. Each subsection closes with an explicit validity condition, because a formula printed without one is a trap for the next reader.

### 4.1 Base case

The base case is the discrepancy identity. Every reconciliation in this file is a special case of it, and every competing cause is a term in its expansion.

```text
FORMULA [D-01]: delta = R - C = d_dup + d_late + d_key + d_scope
WHERE:
  R  reported count           [records]   from the consuming system
  C  reconciled count         [records]   from the frozen extract
  d_dup    duplicates counted twice          [records]
  d_late   records outside the effective-time rule   [records]
  d_key    records whose key changed (reuse, recode) [records]
  d_scope  records inside one population and outside the other [records]
CONSTRAINT: the four terms partition delta; every record belongs to exactly one term.
NUMERICAL EXAMPLE:
  R = 1,240 counted cells, C = 1,186 cells within tolerance
  delta = 1,240 - 1,186 = 54 cells
  observed split: d_dup = 11, d_late = 0, d_key = 9, d_scope = 34
  check: 11 + 0 + 9 + 34 = 54  (residual 0)
INTERPRETATION: 34 of the 54 cells are a scope difference, not a data-quality
defect. Correcting them as if they were duplicates would double-count the fix.
```

The derivation is a partition argument rather than algebra. Let U be the union of records seen by either system and D the symmetric difference. Every record in D is absent from one side for exactly one reason among four: it was counted twice on one side, it arrived after the effective-time cut-off, its business key was rewritten, or it was legitimately inside one definition of the population and outside the other. Hence the sum of the four terms equals the size of D, which equals the absolute discrepancy when one side is a subset of the other.

The constraint matters more than the identity. If the four terms are computed by four different teams at four different times they will overlap, and the residual will be negative. A negative residual is diagnostic in itself: it means at least one record was assigned to two causes, and the true split is unknown until the assignment is redone against a single frozen extract.

Two derived quantities follow from D-01 and are used repeatedly below.

```text
FORMULA [D-01a]: delta_pp = 100 * delta / N        (percentage-point view)
FORMULA [D-01b]: share_i  = d_i / delta            (attribution view)
WHERE: N is the eligible cell count for the same window.
NUMERICAL EXAMPLE: delta = 54, N = 1,240 -> delta_pp = 4.35 pp
  and share_scope = 34/54 = 63.0%, share_dup = 11/54 = 20.4%
INTERPRETATION: the largest share belongs to the cheapest fix (a denominator
statement), which is typical; duplicates are usually the most expensive.
```

Validity condition for §4.1: the identity holds only when both counts use the same key, the same window, and the same effective-time rule. If any of those three is different between the sides, the residual is not a measure of data quality but a measure of the definition gap, and D-01 must be replaced by an explicit two-population comparison before any attribution is attempted.

### 4.2 General case

The general case extends the identity from counts to rates, because most store decisions are taken on a rate. Two rates that share a denominator are comparable; two rates that do not are not, and the phantom-inventory family of issues lives exactly at that seam.

```text
FORMULA [D-02]: g = (S_sys - S_shelf) / S_sys
WHERE:
  S_sys    cells with positive system on-hand   [cells]
  S_shelf  cells with a confirmed shelf unit    [cells]
  g        phantom rate                          [dimensionless, 0..1]
CONSTRAINT: S_shelf must be a subset of S_sys; otherwise g measures a different
            defect (negative on-hand, or a shelf unit with no ledger record).
NUMERICAL EXAMPLE:
  window: 9 stores x 120 SKUs x 14 days = 15,120 cells
  S_sys   = 14,410 cells  -> system availability 14,410/15,120 = 95.30%
  S_shelf = 14,092 cells  -> on-shelf availability 14,092/15,120 = 93.20%
  g = (14,410 - 14,092)/14,410 = 318/14,410 = 2.207%
  gap in points: (14,410 - 14,092)/15,120 = 2.103 pp
INTERPRETATION: about 2.2% of believed stock is not reachable by a shopper.
The 2.10 pp figure and the 2.21% figure answer different questions and must
never be quoted interchangeably.
```

The 0.1 pp difference between 2.103 and 2.207 is not rounding noise. The first divides by all cells, including the 710 cells with no stock in either system, and therefore dilutes the phantom effect by the out-of-stock share. The second divides only by cells where the ledger claimed stock. When a category has a high genuine out-of-stock rate, the two numbers diverge sharply, and a report that switches between them mid-year will show a trend that does not exist.

The general adjustment for substitution follows the same structure. Substitution is a real commercial outcome but it is not availability, so it belongs in a separate numerator rather than in the base rate.

```text
FORMULA [D-03]: OSA_adj = (S_shelf + D_sub) / N_cells
WHERE:
  D_sub   cells where a designated substitute was on shelf and the primary was not
  N_cells stores x SKUs x days, same window and same SKU set
CONSTRAINT: each cell contributes at most one substitution credit, and the
            substitute must be listed in the planogram's approved set.
NUMERICAL EXAMPLE:
  base   = 14,092/15,120 = 93.20%
  D_sub  = 214 cells -> (14,092 + 214)/15,120 = 14,306/15,120 = 94.62%
  if substitutes were instead subtracted from the failure count:
  (14,092 - 214)/15,120 = 91.79%
INTERPRETATION: the two treatments differ by 2.83 pp, which is larger than the
annual movement most availability programmes are trying to detect.
```

The direction of the substitution is a policy decision, and the arithmetic only makes the consequence visible. Adding a credit of 214 cells raises the rate by 1.42 pp; removing the failed cells from the denominator entirely would raise it further and is almost always wrong, because it deletes the evidence that the primary was missing. The defensible convention is the additive credit with a capped count of one per cell, disclosed in the metric definition, and never mixed across periods.

Validity condition for §4.2: the general case is valid when both rates are computed on the identical cell set, when the substitution credit is capped at one per cell, and when the SKU set is held fixed. Violating the third condition is the most common and the least visible; it is the mechanism behind the seasonal-assortment entry in §8.

### 4.3 Variants

Three variants recur across the catalogue: the effective-time lag, the capacity overflow, and the clustered comparison. Each changes one assumption of the base case while leaving the rest intact.

**Variant A — effective-time lag.** A record may only influence a decision if the lag from event to effective time is inside a stated window.

```text
FORMULA [D-04]: L = t_effective - t_event  [hours]
                admissible iff 0 <= L <= eps
WHERE:
  t_event     device-local timestamp of the physical event
  t_effective first planning-cycle boundary at which the record may be used
  eps         policy lag allowance, typically 24 h for daily replenishment
CONSTRAINT: L < 0 is never admissible; a negative lag means a clock or a
            time-zone defect, not an early record.
NUMERICAL EXAMPLE:
  a till in a UTC-5 store stamps 21:40 local; the extractor reads it as 21:40 UTC
  apparent event time = 02:40 next day; true event time 21:40
  L = 5 h of manufactured lag on 5/24 = 20.8% of the day's records
  if that day's transacted value is $18,400, the misassigned share is
  $18,400 x 5/24 = $3,833.33
INTERPRETATION: the store loses the record from the correct day and gains a
fictitious one on the following day; both daily totals look plausible.
```

The discriminating property of a time-zone defect is that it moves a fixed fraction of the day between two adjacent buckets and leaves the total unchanged. A data-loss defect reduces the total. A duplicate defect raises it. Because a time-zone defect is total-preserving, weekly reports reconcile while daily reports do not, and that asymmetry is the cheapest test available.

**Variant B — capacity overflow.** A queue that exceeds its cap does not fail loudly; it drops work.

```text
FORMULA [D-05]: dropped = max(0, A - K)
                W_q = L_q / lambda,  L_q = rho^2 / (1 - rho),  rho = lambda/mu
WHERE:
  A       arrivals eligible for the queue in the window   [tasks]
  K       review capacity in the window                   [tasks]
  lambda  arrival rate      [tasks/hour]
  mu      service rate of one reviewer [tasks/hour]
  W_q     mean wait in queue [hours]
CONSTRAINT: the queue is stable only while rho < 1; at rho >= 1 the wait is
            unbounded and no prioritisation rule can recover lost service.
NUMERICAL EXAMPLE:
  lambda = 46 tasks/h, mu = 52 tasks/h -> rho = 46/52 = 0.885
  L_q = 0.885^2/(1 - 0.885) = 0.7832/0.1154 = 6.78 tasks
  W_q = 6.78/46 = 0.1474 h = 8.85 min
  at lambda = 61: rho = 61/52 = 1.173 -> L_q undefined (queue grows without bound)
  overflow illustration: A = 480, K = 400 -> dropped = 80 tasks = 16.7%
INTERPRETATION: a 33% rise in arrivals does not raise the wait by 33%; it turns
a finite queue into an unbounded one and starts discarding the lowest tier.
```

The practical consequence is that capacity must be sized against the peak eligible arrival rate, not the mean, and that the drop rule must be logged. An unlogged drop is indistinguishable from a data-quality failure, which is why the exception-overflow entry sits in the SCALE_FAILURE category while the operator-override entry sits in NUMERICAL_EDGE: the first loses work by design, the second loses work by omission.

**Variant C — clustered comparison.** Store-level interventions are assigned to stores, so the analysis unit is the store even when the measurement unit is the cell.

```text
FORMULA [D-06]: DEFF = 1 + (m - 1) * rho
                n_eff = n / DEFF
                half_width = z * sqrt(p(1-p)/n) * sqrt(DEFF)
WHERE:
  m    cells per cluster (store-window)      [cells]
  rho  intra-cluster correlation             [0..1]
  n    total cells across clusters           [cells]
  z    1.96 for a 95% interval
CONSTRAINT: rho > 0 always for a store cluster; DEFF = 1 is an assumption that
            must be justified, never a default.
NUMERICAL EXAMPLE:
  n = 15,120 cells, p = 0.932, m = 1,680 cells per store
  naive se = sqrt(0.932 x 0.068 / 15,120) = 0.002047 -> +/- 0.40 pp
  assumed rho = 0.00125 -> DEFF = 1 + 1,679 x 0.00125 = 3.099
  clustered half-width = 1.96 x 0.002047 x sqrt(3.099) = 0.71 pp
  effective sample size = 15,120/3.099 = 4,879 cells
INTERPRETATION: the naive interval understates the true uncertainty by 1.76x,
which is enough to turn a flat result into a false 0.5 pp win.
```

The variant matters for diagnosis as well as for inference. When a store-level pilot moves availability by 0.8 pp and the clustered half-width is 5.0 pp, the correct statement is not "no effect" but "the design cannot see an effect of this size", and the appropriate follow-up is more stores, not more days. Increasing days inside the same nine stores raises n without raising the number of clusters, and under D-06 it buys almost nothing once the within-store correlation dominates.

Validity condition for §4.3: each variant is valid only inside its own stated bound — a lag of exactly zero is admissible for real-time decisions, a queue at rho of exactly 1 has no finite steady state, and a design effect computed with an assumed rho is an assumption and must be labelled as one. Where rho is unknown, the honest output is a range, not a point.

### 4.4 Limits and counterexamples

The formulas above fail in specific, predictable ways. This subsection states each failure and gives the counterexample that exposes it.

**Limit 1 — censoring.** A mean computed over resolved cases only is biased downward whenever unresolved cases would have been slower.

```text
FORMULA [D-07]: mu_naive = sum(observed) / n_total
                mu_bound = (sum(observed) + k * t_cut) / n_total
WHERE:
  k        cases still open at the cut-off        [orders]
  t_cut    cut-off length                         [days]
  n_total  all cases including open ones          [orders]
CONSTRAINT: t_cut must not exceed the true maximum service time; using the
            cut-off as the value for open cases is a lower bound, not an estimate.
NUMERICAL EXAMPLE:
  34 orders; 30 resolved with mean 2.9 days; 4 still open at a 6-day cut-off
  mu_naive = (30 x 2.9)/34 = 87.0/34 = 2.559 days
  mu_bound = (87.0 + 4 x 6)/34 = 111.0/34 = 3.265 days
  bias = 3.265/2.559 - 1 = +27.6% understatement of the naive figure
INTERPRETATION: the naive mean says service is 12.6% better than the bound;
the bound is the defensible number, and the gap is the censoring cost.
```

Note the asymmetry: D-07 bounds the bias but does not remove it, because the open cases could be slower than the cut-off. The correct reporting form is "mean between 2.56 and 3.27 days, with 4 of 34 unresolved", which is honest and still actionable. A single number with no open-case count is not.

**Limit 2 — unit conversion.** A rate built on mixed units is wrong by the conversion factor, and the error is invisible while the mix is stable.

```text
FORMULA [D-08]: U = Q_case * P_pack + Q_each
WHERE:
  Q_case  cases received        [cases]
  P_pack  units per case        [each/case]
  Q_each  loose units received  [each]
CONSTRAINT: P_pack must be read from the pack record effective at the receipt
            date, not from the current item master.
NUMERICAL EXAMPLE:
  need 12 units; pack size 24; minimum order 1 case
  units delivered = 1 x 24 = 24; surplus = 24 - 12 = 12 units = 100% over need
  three cases = 3 x 24 = 72 units for a need of 12 -> 84.6% over
INTERPRETATION: the surplus is a real cash and shelf-space cost, and it appears
as over-delivery in the ledger even though the order quantity was correct.
```

**Limit 3 — definition shift.** A name that survives a redefinition is the most expensive defect in the catalogue, because every consumer keeps working and only the number changes.

The counterexample that matters most is the substitution case in §4.2. Adding 214 credits to 14,092 cells gives 94.62%, while the unadjusted rate is 93.20% and the strict-failure rate is 91.79%. Three defensible-sounding numbers, one dataset, a spread of 2.83 pp. If the metric definition changes without a version bump, the trend line will show a 1.42 pp improvement on a day when no shelf changed. Nothing in the data can detect this; only the definition register can.

**Limit 4 — shared upstream.** A zero discrepancy on two counts does not prove correctness, because both counts may descend from one defective extract.

```text
FORMULA [D-09]: independent iff source(A) ∩ source(B) = ∅
COUNTEREXAMPLE: the ledger and the dashboard both read the same nightly
inventory snapshot; both report 14,410 cells; the shelf audit finds 14,092.
  agreement between A and B: 0 cells discrepancy
  agreement with the shelf: 318 cells discrepancy
INTERPRETATION: agreement is evidence of a shared ancestor, not of truth.
```

Validity condition for §4.4: each limit is stated as a bound rather than a correction, and a bound is only usable when its direction is stated. Censoring understates, conversion error is a multiple, definition shift is a level jump with no local signal, and shared ancestry produces false agreement. A diagnostic that cannot state its direction cannot be used to decide which of two conflicting numbers to trust.
