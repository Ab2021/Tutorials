---
collection: "pricing_underwriting_and_reserving"
file_id: "05"
topic: "Claims reserving methods"
level: 2
status: "draft"
version: "1.2"
created: "2026-11-07"
reviewed: null
review_trigger: "Change in claims handling, valuation basis, source or actuarial guidance"
notation_style: "plain-text"
prerequisites: ["Basic insurance terminology, ratios and weighted averages"]
pattern_tags: ["reserving", "claims", "development"]
source_status: "OPEN"
owner_role: "Actuarial education reviewer"
source_index: "INDEX.md"
---

# 1. Claims Reserving Methods — File Header

> **Purpose:** Develop a paid claims triangle into an illustrative unpaid estimate and diagnose when its assumptions fail.
> **Prerequisites:** Ratios, sums, accident-year cohorts and basic claims terminology; [pricing sheet](03_NonLife_Pricing_Ratemaking.md) is optional.
> **Key outcome:** Reproduce the 148-currency-unit paid-based outstanding estimate and explain why it is neither a confidence interval nor an incurred-based IBNR estimate.
> **This file does NOT cover:** Booking advice, jurisdiction-specific valuation rules, a complete stochastic reserve model, or reinsurance contract interpretation.
> **Level:** 2 — Core Methods.
> **Version / reviewed:** 1.2 / not independently reviewed.
> **Evidence:** DERIVED and VERIFIED arithmetic for an ILLUSTRATIVE triangle; external reference SOURCED; original inventory contents OPEN.
> **Retrieval cue:** For unexpectedly high development, start with Section 8; for the numerical trace, use Section 5.

A reserving analyst estimates future claim obligations at a valuation date and communicates assumptions, ranges and limitations to decision owners.
The paid chain-ladder calculation is a projection under selected development assumptions, not an automatic bookable reserve recommendation.
**OPEN:** The source inventory names `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/05_Claims_Reserving_Methods.md`; its original contents were unavailable and are not represented as inspected or reproduced.
**SOURCED:** Jacqueline Friedland, *Estimating Unpaid Claims Using Basic Techniques*, Casualty Actuarial Society, 2010, is the primary educational reference for claim development and Bornhuetter-Ferguson methods: https://www.casact.org/sites/default/files/database/studynotes_friedland_estimating.pdf.
No insurer's production data, deployment, accuracy, or performance is asserted here.
**Takeaway:** A mathematically reproducible point estimate still requires informed actuarial judgment and an explicitly matched liability basis.

# 2. Concept Map / Mental Model

```text
Claim transactions + case snapshots -> consistent valuation cutoff and scope
    -> reconcile paid / incurred to source controls
    -> build incremental cells -> accumulate by accident year
    -> inspect age-to-age and calendar diagonals
       -> assumptions unsuitable: segment, adjust, use an alternative, or escalate
       -> assumptions defensible: select factors and explicit tail
    -> project ultimate -> subtract matching observed paid or incurred
    -> compare methods, quantify uncertainty, document and obtain review
```
The arrows describe an analytical dataflow, not proof that one observed variable causes another.
The central invariant is that the cumulative value equals the sum of its included increments through that age, on the same cohort and financial basis.
Paid and incurred are distinct inputs; they do not become interchangeable just because both can be arranged as triangles.
**Takeaway:** Reconciliation and cohort consistency precede factor selection, not the reverse.

# 3. Foundational Definitions

| Term | Definition and units | Interpretation / nonexample | Operational check |
|---|---|---|---|
| Accident year (AY), i | Cohort indexed by year of accident | Not payment year | Match occurrence-date mapping to the policy and claims extracts. |
| Development age, j | Ordered elapsed time bucket since cohort origin | Not the calendar year by itself | Use identical age cutoffs and valuation dates for all rows. |
| Incremental paid, P(i,j) | Currency paid in age bucket j for cohort i | A payment movement, not a balance | Sum included transactions, including corrections under the stated convention. |
| Cumulative paid, C(i,j) | Currency paid for cohort i through age j | A running sum, not incurred loss | Recompute from incremental paid and compare to snapshot. |
| Case reserve | Claim-specific estimate of future payments on known claims at valuation | Not an extra payment already made | Reconcile the snapshot to open claim records and adjuster system. |
| Incurred | Paid plus outstanding case reserves on a consistent basis | Not automatically ultimate | Match payment, case, expense and recovery definitions. |
| IBNR | Incurred-but-not-reported claims; some operational usages include development on known claims | Terminology varies: specify whether IBNER is included | Record the actuarial definition and formula alongside the number. |
| IBNER | Incurred but not enough reported: development on known claims beyond booked case estimates | Not synonymous with strictly unreported claims | Explain whether IBNER is part of the reported IBNR amount. |
| Development factor f(j) | Dimensionless ratio selected for cumulative development from age j to j+1 | Not a claim count or a premium | Check the matched-pair denominator and excluded cohorts. |
| Tail factor T | Dimensionless selected development after the final modeled age | Not automatically one in real portfolios | Document external evidence, extrapolation and sensitivity. |
| Ultimate U(i) | Estimated total paid by final settlement for cohort i, on the stated basis | Not the same as current incurred | Preserve cohort, gross/net, expenses, currency and discount basis. |
| Paid-based outstanding | Ultimate less paid to date, if bases match | Includes amounts represented by case reserves and development beyond them | Do not add case reserves a second time to this total. |
| Incurred-based IBNR | Ultimate less incurred to date, when incurred equals paid plus case | Often includes IBNER in broad usage; not necessarily pure unreported loss | State included categories and reconcile to paid outstanding. |
| Calendar period | Common transaction or valuation period across accident years | A diagonal in a standard evenly spaced triangle, not an accident-year row | Compare changes in process and inflation across diagonals. |
| Valuation cutoff | Last included transaction and snapshot date, with inclusion rules | Not necessarily the date a file was exported | Preserve as-of extraction logic and late-arriving adjustments. |

All amounts in the worked example are currency units; factors are dimensionless scalars.
A negative payment increment may represent recoveries or corrections; cumulative paid need not be strictly increasing on every real dataset.
A missing future cell is unobserved, not zero; an observed zero is a real value under the extract convention.
**Takeaway:** The economic category, timing, and financial basis of each cell determine what subtraction means.

# 4. Core Formulas with Full Derivation

## 4.1 Simplest / Base Case

Validity: incremental amounts for one cohort are mutually exclusive age buckets, consistently defined, with all earlier buckets included.
Derivation: a running total through age j is the preceding total plus the payment movement during the new age bucket.
[1] FORMULA — C(i,j) = sum over k=1..j of P(i,k).
[2] WHERE — C and P are currency units; i identifies an accident-year cohort and j an ordered development bucket.
[3] CONSTRAINT — Apply the same coverage, financial scope, cutoff and sign convention to every bucket.
[4] NUMERICAL EXAMPLE — AY1 increments 100, 50, 30 yield cumulative 100, 150, 180.
[5] INTERPRETATION — The 30 is a new movement; 180 is the total paid through age 3.
Boundary: an omitted correction or a duplicated payment invalidates the cumulative total even if it appears smooth.

Validity: matched cohorts have observable cumulative paid at both adjacent ages and their total prior-age paid is nonzero.
Derivation: the volume-weighted mean of each individual ratio C(i,j+1)/C(i,j), weighted by C(i,j), cancels its denominators, giving the ratio of matched sums.
[1] FORMULA — f(j) = sum over matched i C(i,j+1) / sum over matched i C(i,j).
[2] WHERE — f(j) is dimensionless; the same eligible matched set is used in both sums.
[3] CONSTRAINT — For the ordinary weighted-mean interpretation, individual prior cumulative values are positive; a nonzero aggregate denominator alone permits the ratio but not necessarily a convex weighted mean.
[4] NUMERICAL EXAMPLE — f(1) = (150+180)/(100+120) = 330/220 = 1.5.
[5] INTERPRETATION — Each paid currency unit at age 1 is associated with 1.5 paid currency units at age 2 in this matched sample.
Boundary: do not include AY3's 140 in the denominator because its age-2 value is unobserved.

## 4.2 General Case

Validity: the selected development factors represent the relevant cohort sufficiently well for the intended projection; all selected ages and the tail use the same paid basis.
Derivation: sequential multiplication converts the latest observed cumulative paid into the final modeled age, then multiplication by the tail extends beyond it.
[1] FORMULA — U(i) = C(i,a(i)) × product of selected f(j) for j=a(i)..J-1 × T.
[2] WHERE — a(i) is cohort i's latest observed age, J the last modeled age, and T the selected tail factor from age J to ultimate.
[3] CONSTRAINT — The latest cell must be observed, factors and tail explicitly selected, and multiplication meaningful for the signed data and method.
[4] NUMERICAL EXAMPLE — AY3: 140 × 1.5 × 1.2 × 1 = 252 currency units.
[5] INTERPRETATION — The selected assumptions project AY3 to 252 total paid at ultimate, not 252 of additional payments.
Boundary: no paid amount at an early age makes a multiplicative paid projection potentially uninformative, even when future payments are plausible.

Validity: ultimate and observed paid cover the same obligations, currency, gross/net, expense and discount basis.
Derivation: partition ultimate into already-paid and future-paid portions; rearrange the equality.
[1] FORMULA — paid outstanding O(i) = U(i) − C(i,a(i)).
[2] WHERE — O, U and C are currency units for the same cohort and valuation date.
[3] CONSTRAINT — No further deductions or discounting are applied in this example.
[4] NUMERICAL EXAMPLE — AY3: 252 − 140 = 112 currency units.
[5] INTERPRETATION — 112 is projected additional paid, including amounts already represented by case reserves where relevant.
Boundary: negative results are possible in signed or revised data and require investigation rather than an automatic floor.

Validity: incurred I equals paid C plus case reserves K on the same basis and U estimates the same ultimate.
Derivation: U − C = (U − I) + (I − C) = (U − I) + K.
[1] FORMULA — broad incurred-based IBNR B = U − I; paid outstanding O = K + B.
[2] WHERE — I, K, U, B and O are same-cohort currency amounts; B here includes IBNER when present.
[3] CONSTRAINT — Terminology is explicitly broad, and no additional reserve category is silently omitted.
[4] NUMERICAL EXAMPLE — if U=252, C=140 and K=70, then I=210, B=42 and O=70+42=112.
[5] INTERPRETATION — 42 is development beyond the case estimate under this definition; the case reserve is already inside the 112 outstanding.
Boundary: the 70 case amount is a separate hypothetical illustration, not observed in the paid triangle.

## 4.3 Key Special Cases / Variants

| Variant | Change and assumption | Use when | Avoid when |
|---|---|---|---|
| Paid chain ladder | Project reported paid with selected historical factors | Payment patterns are informative and sufficiently stable | Payment speed changed materially without adjustment. |
| Incurred chain ladder | Apply factors to paid plus case | Case practices are consistent and informative | Case strengthening or weakening distorts the pattern. |
| Bornhuetter-Ferguson (BF) | Add expected unpaid share of an independently selected prior ultimate to observed amount | Immature development makes direct extrapolation fragile and credible prior information exists | Prior expected ultimate or development proportions are unsupported. |
| Tail sensitivity | Replace T=1 with a stated alternative | Claims may develop beyond last observed age | Presenting an assumed tail as observed maturity. |

Validity for a paid BF estimate: prior expected ultimate E and expected paid proportion q are on the same cohort and loss basis; q is between zero and one under an ordinary nonnegative payment pattern.
Derivation: expected future paid equals E times the expected unpaid proportion 1−q; add observed paid C to obtain projected ultimate.
[1] FORMULA — U_BF = C + E × (1−q); paid outstanding under BF is E × (1−q).
[2] WHERE — E and C are currency units, q is dimensionless, and U_BF is a currency-unit estimate.
[3] CONSTRAINT — E is selected separately; q must correspond to the same age, tail and scope.
[4] NUMERICAL EXAMPLE — C=140, E=240 and q=0.5 imply U_BF=140+240×0.5=260 and outstanding=120.
[5] INTERPRETATION — The prior contributes the expected unpaid portion, while actual paid replaces its expected paid portion.
Boundary: BF reduces sensitivity to current immature paid, but does not remove prior, pattern, or process risk; this separate hypothetical is not the triangle's chain-ladder estimate.

## 4.4 Asymptotic / Limiting Behaviour and Counterexample

At the final modeled age with T=1, the selected method assigns no further development, so estimated paid outstanding is zero.
That is an assumption about the remaining tail, not empirical proof that claims are fully settled.
If the matched denominator is zero, the factor is undefined and a numerical fallback must not be presented as a calculated factor.
If every matched row has the same positive age-to-age ratio, the volume-weighted ratio equals that common ratio.
A calendar-period change affecting all cohorts at different development ages can violate the stable age-pattern assumption.
**Takeaway:** Every projected amount is conditional on a matched set, a selected pattern and a tail, not a certainty about future settlement.

# 5. Worked Numerical Example

**ILLUSTRATIVE / VERIFIED arithmetic:** The following is constructed solely for teaching and is not an insurer's experience.
Workflow: estimate paid-based outstanding at one common valuation date using a cumulative paid triangle.
No deductions, discounting, expenses outside the stated paid basis, or reinsurance adjustments are applied; tail factor T=1.
The case reserve and incurred triangles are not provided, so incurred-based IBNR cannot be computed from this paid triangle alone.

| Accident year | Age 1 cumulative | Age 2 cumulative | Age 3 cumulative |
|---|---:|---:|---:|
| AY1 | 100 | 150 | 180 |
| AY2 | 120 | 180 | unobserved |
| AY3 | 140 | unobserved | unobserved |

The implied incremental triangle is AY1 [100, 50, 30], AY2 [120, 60, unobserved], AY3 [140, unobserved, unobserved].
Age 1→2 uses AY1 and AY2: (150+180)/(100+120)=330/220=1.5.
Age 2→3 uses AY1 only: 180/150=1.2; this single observation is especially fragile.
The missing cells are excluded rather than assigned zero; the tail of 1 is a stated simplification.

| Cohort | Latest paid | Remaining factor product | Ultimate | Paid outstanding |
|---|---:|---:|---:|---:|
| AY1 | 180 | 1 | 180 | 0 |
| AY2 | 180 | 1.2 | 216 | 36 |
| AY3 | 140 | 1.5 × 1.2 = 1.8 | 252 | 112 |
| Total | 500 | Not applicable | 648 | 148 |

Independent arithmetic check: projected future increments are AY2 216−180=36 and AY3 (210−140)+(252−210)=70+42=112; 36+70+42=148.
A second reconciliation is total ultimate 180+216+252=648 less current total paid 180+180+140=500, yielding 148.
A wrong path is to divide 330 by 360, including AY3's unmatched 140; 330/360 is not a matched age-to-age factor.
Another wrong path is to add case reserves to 148; paid-based outstanding already includes the future payments those case amounts anticipate.
Production-scale interpretation: estimate multiple segments only after adequate reconciliation and stability checks; sparse cohorts and heterogeneous claims may demand alternative methods.
The 148 is a deterministic point projection, not a confidence interval, a reserve distribution, or a recommendation to book 148.
**Takeaway:** The arithmetic reconciles exactly, while the assumptions and uncertainty remain open to actuarial assessment.

# 6. Visual / Geometric Intuition

```text
                    Development age ->
Accident cohort       1        2        3
AY1                 100      150      180  <- latest observed
AY2                 120      180       ?   <- project 216
AY3                 140       ?        ?   <- project 210, then 252
```
Caption: rows are accident-year cohorts; columns are elapsed development ages; question marks are future, not zero, cells.
For aligned equal-length periods, a diagonal through the latest cells corresponds to the common valuation period.
Calendar shocks can run along diagonals and therefore appear at different ages for different cohorts.
The triangle shape depicts data availability, not a probability interval or a guarantee that projected cells will materialize.
**Takeaway:** Read along rows for cohort development and along aligned diagonals for shared calendar conditions.

# 7. Comparison / Selection Table

| Method | Preconditions | Guarantee / cost | Suitable use | Avoid / boundary | Evidence |
|---|---|---|---|---|---|
| Paid chain ladder | Stable relevant paid pattern; reconciled transactions | Reproduces selected factors; low arithmetic cost, no accuracy guarantee | Mature, credible paid development | Structural payment-speed shifts | DERIVED; Friedland 2010 |
| Incurred chain ladder | Consistent case reserving and paid basis | Uses case information; same triangle order of computation | Case estimates add stable information | Changing case adequacy | SOURCED; Friedland 2010 |
| Paid BF | Defensible prior ultimate and payment proportion | Less directly driven by immature paid; requires prior selection | Early cohorts with a credible prior | Unsupported or stale prior | DERIVED; Friedland 2010 |
| Expert scenario | Documented assumptions and independent review | Makes structural risks visible; no automatic statistical coverage | Breaks or sparse data | Undocumented overrides | HEURISTIC |

First disqualify a method whose basis cannot be reconciled or whose assumptions are demonstrably inappropriate.
If paid patterns are stable and credible, use paid development as a transparent baseline.
If paid maturity is low but an independently supported prior is available, compare BF against the baseline.
If case practices are stable, incurred development offers a complementary view; otherwise investigate the change first.
If methods conflict, reconcile definitions and investigate assumptions before weighting or escalating the difference.
Comparative predictive performance is NOT MEASURED here; use rolling as-of backtests with unchanged claim scope and document revisions.
**Takeaway:** Selection depends on data validity and information quality, not a universal ranking of methods.

# 8. Failure Modes & Edge Cases

| ID | Symptom and competing causes | Discriminating check | Bounded action and verification | Provenance |
|---|---|---|---|---|
| F-01 | Factor rises: inflation or delayed payments | Compare calendar diagonals, counts and payment timing | Investigate process change; backtest adjusted assumptions | HYPOTHETICAL |
| F-02 | Incurred development shifts: case adequacy or severity mix | Compare case per open claim and closure rates with consistent segmentation | Review adjuster practices and case history; reconcile snapshots | HYPOTHETICAL |
| F-03 | Latest diagonal does not reconcile: late transactions or duplicates | Compare source controls by period and transaction ID | Correct source lineage and rerun controlled as-of extraction | HYPOTHETICAL |
| F-04 | Factor undefined: zero denominator or missing pairs | Inspect pair eligibility and zero versus null values | Use a justified alternative; never silently divide by zero | HYPOTHETICAL |
| F-05 | Negative increment: recovery or error | Trace signed transactions to original claim and recovery classification | Retain valid corrections; repair misclassification and reconcile | HYPOTHETICAL |
| F-06 | Paid outstanding exceeds incurred-based IBNR | Case reserves may explain the difference | Reconcile U−paid = case + U−incurred; prevent double count | HYPOTHETICAL |
| F-07 | Young AY projection implausibly low | Sparse early payments or portfolio shift | Compare BF and exposure-based prior against paid development | HYPOTHETICAL |
| F-08 | Aged cohort still pays after modeled horizon | Tail factor of one may be invalid | Extend evidence or use justified tail scenarios | HYPOTHETICAL |
| F-09 | Factor seems unusually low | Unmatched denominator or recoveries may be responsible | Inspect matched cohorts and gross/net definitions | Correct pairing and rerun reconciliation | HYPOTHETICAL |
| F-10 | Backtest looks perfect | Future data or revised snapshots may leak into as-of model | Freeze historical extracts and factor-selection dates | Repeat genuine as-of backtest and preserve lineage | HYPOTHETICAL |

A negative diagnostic does not prove stability: small samples can conceal a change, and absence of a documented process change does not establish absence of change.
Escalate unresolved financial reconciliation, material basis ambiguity or unstable factors to the responsible actuary before use.
Common Issues collection artifact is planned, not authored; these local F-IDs are not claims of observed incidents.
**Takeaway:** Diagnose the mechanism behind a symptom before altering factors or interpreting the reserve movement.


## Operational controls and evidence

| Control | Evidence and action | Risk if omitted |
|---|---|---|
| Source reconciliation | Bridge paid and case snapshots to independent claim-system totals at the valuation cutoff. | An incomplete extract can produce a plausible but understated triangle. |
| Transaction lineage | Retain claim identifier, transaction identifier, event date, posting date, currency and correction history. | Duplicates, late arrivals and reversals cannot be distinguished reliably. |
| Cohort mapping | Preserve occurrence-date rules and effective-dated coverage and segment mappings. | Reclassification can look like claim development. |
| Financial basis | Document gross/net, expenses, recoveries, currency conversion and discount conventions. | Inconsistent bases invalidate subtraction and comparison. |
| As-of reproducibility | Retain immutable historical extracts, code versions, assumptions and review decisions. | A rebuilt current snapshot leaks later information into historical evaluation. |
| Change management | Compare payment speed, closure, case adequacy and calendar diagonals with documented process changes. | Handling changes can masquerade as loss development. |
| Review and approval | Record independent review, exceptions, sensitivity, rationale and decision ownership. | A point projection may be mistaken for an approved liability. |

# 9. Complexity & Cost Analysis

## 9a. Algorithmic Complexity

Let N be transaction records, A the number of accident-year cohorts, and D the number of development ages.
A single-pass aggregation of normalized transactions is O(N) time plus O(A×D) dense triangle storage, assuming constant-time keyed aggregation.
Factor calculation and projection each take O(A×D) time in a straightforward dense implementation; sparse representations may reduce work when most cells are absent.
Input sorting, joins, deduplication, source extraction and multiple dimensions may dominate the simple aggregation bound.
Historical reruns require retained as-of inputs and selected assumptions; recomputation is not a substitute for source lineage.

## 9b. Back-of-Envelope Sizing

ILLUSTRATIVE sizing: 100 cohorts × 40 development ages × 8 bytes per numeric cell = 32,000 bytes for one dense raw numeric triangle.
Four such arrays use 128,000 bytes of numeric payload before metadata, null masks, object overhead, versioning, replicas or input transactions.
Doubling both dimensions quadruples the payload to 512,000 bytes for four arrays; it does not predict runtime.
Environment, latency budget, currency costs and real workload are NOT MEASURED; no deployment or benchmark is claimed.
Reconciliation, expert review, data extraction, reruns and governance often matter operationally beyond the small arithmetic kernel; validate the actual bottleneck by measurement.
**Takeaway:** Triangle arithmetic is usually simple, but reproducible source preparation and expert review are separate costs.

# 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Practical implication | Does not imply | Source |
|---|---|---|---|---|
| Matched ratio identity | With positive prior values, a prior-value-weighted average of individual ratios equals the ratio of their matched sums | Large prior cells have proportionately more weight | Stable future development | Algebra, DERIVED; Friedland 2010 development reference |
| Reserve decomposition | When incurred = paid + case on a matching basis, U−paid = case + (U−incurred) | Reconciles paid outstanding and broad incurred-based IBNR | Strict unreported-only IBNR is separately identified | Algebra, DERIVED |
| Deterministic projection | Selected factors produce one point outcome conditional on assumptions | Supports transparent sensitivity analysis | Confidence coverage or a probability distribution | Definition and scope, DERIVED |

**Takeaway:** These are algebraic identities or definitions, not statistical coverage guarantees.

# 11. Interview / Exam Questions

## Level 1 — Definition / Recall

1. **What is a cumulative paid triangle?** A cohort-by-age table of paid-to-date amounts; AY1 150 at age 2 includes both the initial 100 and a 50 increment.
2. **How does case reserve differ from paid?** Case reserve estimates future payments on known claims; paid represents payments already made, subject to the chosen transaction basis.
3. **Is paid outstanding the same as incurred-based IBNR?** No: with matching bases, paid outstanding includes case reserve plus broad IBNR; strict unreported-only terminology may differ.
4. **What is a tail factor?** A selected multiplier extending final modeled-age cumulative amounts to ultimate; one assumes no further development beyond that age.

## Level 2 — Derivation / Apply

5. **Derive the first age factor for the example.** Sum matched age-2 values 150+180=330 and matched age-1 values 100+120=220; the ratio is 1.5.
6. **Calculate AY3 ultimate and outstanding.** 140×1.5×1.2×1=252 ultimate; 252−140=112 paid outstanding, in currency units.
7. **Reconcile all cohorts.** Total ultimate is 648 and current paid is 500, giving 148 outstanding; 0+36+112 independently gives 148.
8. **If AY3 case reserve is 70, what is broad incurred-based IBNR?** Incurred is 140+70=210 and 252−210=42; 70+42 reconciles to 112.
9. **Give a BF calculation.** With prior ultimate 240, paid proportion 0.5 and actual paid 140, BF ultimate is 140+240×0.5=260; outstanding is 120.

## Level 3 — Judgement / Design / Debug

10. **Why not include AY3 in the first factor denominator?** Its age-2 observation is absent; including only its age-1 amount produces an unmatched and biased ratio, not the desired pairwise statistic.
11. **Payments slow after a workflow change; what do you investigate?** Compare payment timing, closure rates and diagonal patterns, then review the operational change and historical as-of backtests before selecting an adjustment.
12. **Case estimates are strengthened; what happens to incurred factors?** Higher case estimates can shift incurred development even without a matching change in final claim costs; compare case adequacy and paid development before interpreting it as severity.
13. **Does 148 give a 95% confidence interval?** No; it is one deterministic projection with no stated stochastic model or calibrated interval.
14. **When can BF help?** When early paid is immature and a credible prior and payment pattern exist; it can fail if that prior is stale or inconsistent with the cohort.
15. **How would you make a production result auditable?** Preserve as-of extracts, transaction and snapshot definitions, reconciliation controls, factor selections, tail assumptions, overrides, reviewer approvals and historical reruns.
16. **Can a low last-age factor prove no tail remains?** No; sparse older development and late-emerging claims may be unobserved; assess external evidence and tail sensitivity.

**Takeaway:** A strong answer reconciles the arithmetic while naming the data, method, uncertainty and decision boundaries.

## Applied reserving laboratory

The following cases are **ILLUSTRATIVE** and all numerical results are derived from the stated inputs.
They are not reports of real insurers, benchmarks, or recommended actuarial selections.
Amounts are currency units unless another unit is explicitly stated.
Numbers are exact where shown; rounded values are marked approximately.
Each case can be used as an interview exercise: cover the answer, solve it, then explain the control and limitation.
The examples are independent unless a connection is explicitly stated.

### 1. Posting date versus economic date

A claim payment of 80 has an economic effective date of December 29.
It is posted to the claims ledger on January 4.
The valuation date is December 31.
The extraction is performed on January 10.
An economic-date retrospective view includes the 80.
A contemporaneous posting-date view excludes it.
These are different questions, not interchangeable implementations.
Suppose the contemporaneous ledger total was 500.
The retrospective economic-date total becomes 580 before other changes.
The bridge is 500 + 80 = 580.
This does not imply the financial statements must use either view.
The accounting close policy determines whether an accrual is required.
Preserve both effective and posting dates in the event record.
Also preserve the extraction timestamp and source cutoff.
Do not overwrite a frozen historical view with a later restatement.
For a genuine backtest, retain the information actually available at that valuation.
For restatement, report the bridge separately from forecast error.
If the 80 was known before year-end but posted late, record the evidence.
If it was unknown, including it in the original forecast leaks future information.
An audit should reconcile both views to their respective independent controls.
**Interview:** Is a later extract automatically a better historical test?
**Answer:** No; it can be a better restatement but an invalid as-known forecast input.
Escalate an unresolved difference before selecting development factors.

### 2. Reversal and replacement

An original payment is 100.
A later reversal is −100.
The corrected replacement payment is 90.
All three events refer to one claim but have different transaction identifiers.
If all are within the cutoff, net paid is 100 − 100 + 90 = 90.
If the cutoff precedes the reversal, the as-known paid amount is 100.
The next period then contains −100 + 90 = −10 of net movement.
A negative increment is therefore not automatically an error.
Dropping the reversal would incorrectly produce 190.
Dropping the original while retaining the reversal produces −10.
Deduplication must not collapse distinct financial events by claim identifier alone.
Link a reversal to the original event using a stable reference where available.
Retain the signed original currency amount and transaction type.
Check the replacement's coverage, payee and expense classification separately.
The source ledger control should agree on the sum of signed movements.
Trace exceptions to the source before adjusting triangle cells.
The corrected history may change an earlier cohort rather than the current cohort.
An as-of replay must respect the time at which each event became available.
Counts of reversal events also help diagnose a migration or interface problem.
Large reversal volumes warrant investigation even when the net total is small.
**Interview:** Why is filtering negative paid dangerous?
**Answer:** It removes valid reversals and biases aggregate paid upward.
No inference about future settlement follows merely from a negative correction.

### 3. Duplicate interface delivery

A payment message with source event key A carries 75.
The interface delivers A twice after a retry.
A second legitimate event B also carries 75 for the same claim.
Raw ingestion therefore contains three rows totaling 225.
The correct event-key total is 75 + 75 = 150.
Deduplicating by claim and amount would incorrectly discard B.
Deduplicating by event key removes only the repeated delivery of A.
The required key must be stable across retries and unique in its source namespace.
If IDs are only unique per source, include source system in the key.
Conflicting payloads under one ID must enter an exception queue.
Do not silently choose the latest payload without a source correction rule.
Record payload hashes, first-seen time and subsequent deliveries.
Compare row counts before and after deduplication by source and day.
Reconcile the deduplicated sum to an independently controlled ledger.
Replay a delivery in a nonproduction environment to test idempotency.
Ensure the replay does not create a second financial effect.
A sum reconciliation alone can miss offsetting duplicated transactions.
An event-level check can also miss an omitted source partition.
Use both completeness and uniqueness controls.
If the ledger itself has duplicate postings, feed-level deduplication cannot fix the ledger.
The financial owner must decide whether source correction is necessary.
**Interview:** Can identical amounts prove duplicate payments?
**Answer:** No; two distinct legitimate events can have the same amount.
Record evidence of the retry and the result of the replay.

### 4. Case reserve snapshots versus transactions

At the end of March a claim has paid 40 and case reserve 60.
At the end of April paid is 55 and case reserve is 50.
March incurred is 40 + 60 = 100.
April incurred is 55 + 50 = 105.
The incurred movement is 105 − 100 = 5.
It consists of 15 additional paid and a −10 case movement.
Adding both case snapshots would incorrectly count 110 as outstanding case.
The March and April snapshots describe states at different dates.
Snapshots should be selected as of each valuation, not summed through time.
Case-reserve changes may also be stored as movements.
If so, accumulating movements must reproduce the snapshot under the same scope.
The claim-level bridge is 60 − 10 = 50.
If payments are booked without a corresponding case reduction, incurred may increase.
That increase may be legitimate or a workflow issue; the numbers alone do not decide.
Separate a change in claim estimate from a change in snapshot extraction.
Record whether closed claims can retain a nonzero case reserve.
Investigate stale open cases and late claim closure messages.
Preserve the time at which an adjuster entered the case estimate.
Also preserve the effective date if backdating is possible.
A current-state table alone cannot reconstruct an as-known historical snapshot.
The finance bridge must include all relevant claims, not just those still open.
**Interview:** Why does paid 15 not imply incurred increased 15?
**Answer:** The case reserve decreased by 10, offsetting part of the payment.
This example does not determine the claim's ultimate cost.

### 5. Cohort reassignment after accident-date correction

Initially a claim has accident date December 31 in AY1.
The recorded date is corrected to January 1 in AY2.
The claim has cumulative paid 120 on the valuation date.
Total portfolio paid is unchanged by the reassignment.
AY1 decreases by 120 and AY2 increases by 120.
The cross-cohort bridge is −120 + 120 = 0.
If AY1 paid had been 600 and AY2 paid 400, revised values are 480 and 520.
The portfolio total remains 1,000.
Reassignment can nevertheless change age-to-age ratios and projected reserves.
It can also alter segment and attachment-year allocation.
Preserve the original and corrected occurrence dates and correction timestamp.
Define whether the report shows as-known or retrospectively corrected cohorts.
Use the same cohort definition in both members of a factor pair.
An effective-dated claim master is necessary for a historical as-known view.
Join cardinality controls should confirm one valid cohort assignment per claim.
Check boundaries around midnight, timezone conversion and incomplete dates.
Large claim corrections deserve separate review before aggregate projection.
If a correction changes coverage eligibility, total included paid might also change.
That is a scope effect, not merely a cohort reallocation.
Tie any adjustment back to the claim-system evidence.
**Interview:** Why can a zero portfolio bridge affect the reserve?
**Answer:** The same dollars move between cohorts with different remaining factors.
Recalculate the affected cells, factor selections and downstream sensitivity.
Do not silently restate a published historical triangle.

### 6. Policy segment migration

A product was coded as segment A when written.
The claims system now classifies it as segment B.
There are 200 of paid claims on that product.
An as-written segmentation assigns 200 to A.
A current-product segmentation assigns 200 to B.
Both may be useful but answer different management questions.
Assume the remaining multipliers are 1.1 for A and 1.4 for B.
As-written projected future paid is 200 × 0.1 = 20.
Current-segment projection is 200 × 0.4 = 80.
The difference is 60 solely under the stated multipliers.
It does not prove either segmentation is more predictive.
The mapping table needs effective dates and a documented business definition.
Avoid joining a historical claim to today's product label without a stated intention.
Backtesting should reconstruct the mapping available at each valuation.
Compare stable like-for-like product cohorts before declaring a payment-pattern change.
Determine whether a portfolio transfer changed claims handling as well as labels.
Reconcile the segment totals back to the unchanged portfolio total.
If a claim maps to multiple segments, define allocation weights summing to one.
Keep unallocated claims in an explicit exception category.
Material unallocated amounts require review rather than a default segment.
**Interview:** Is a portfolio-total reconciliation sufficient?
**Answer:** No; misallocation can leave the portfolio total intact while distorting factors.
Treat the multiplier comparison as sensitivity, not a measured impact.
An authorized owner must approve any historical restatement.

### 7. Foreign currency and payment timing

A claim has two payments of 100 units of foreign currency.
The first occurs when the reporting-currency rate is 1.0 per foreign unit.
The second occurs when the rate is 1.2 per foreign unit.
Transaction-date conversion gives 100 × 1.0 + 100 × 1.2 = 220.
Applying the final rate to both gives 200 × 1.2 = 240.
The 20 difference is a conversion-basis difference.
Neither number is universally correct for every financial reporting purpose.
The chosen valuation and reporting policy determines the relevant basis.
Do not combine a transaction-date paid triangle with an ultimate expressed at a different rate without a bridge.
Store original currency, original amount, conversion date and rate source.
Handle missing or corrected rates explicitly.
Exchange-rate changes can create apparent calendar-period development.
Analyze the local-currency triangle as a complementary diagnostic.
Separate claims inflation from currency translation where evidence permits.
For future payments, the assumed exchange-rate path is an additional uncertainty.
A fixed-rate sensitivity is not a forecast distribution.
Reconciliation should distinguish local-currency totals from reporting-currency totals.
The conversion policy must cover recoveries and reversals consistently.
Test that a reversal uses the correct accounting rate convention.
If not, a residual foreign-exchange gain or loss can remain.
**Interview:** Did nominal claims payments grow by 20?
**Answer:** Not in local currency; the displayed difference comes from translation.
Consult the applicable accounting policy rather than inferring it from the triangle.
Retain the precise rate series used for reproduction.

### 8. Recovery grossing and signs

An insurer pays 200 to a claimant.
It later receives a salvage recovery of 30.
Gross paid before recovery is 200.
Net paid after recovery is 200 − 30 = 170.
If gross ultimate is 300, gross outstanding is 300 − 200 = 100.
Suppose net ultimate is separately estimated as 250.
Net outstanding is 250 − 170 = 80.
Gross ultimate minus net paid would be 300 − 170 = 130.
That mixed-basis 130 is neither of the defined outstanding amounts.
Recovery timing may differ materially from indemnity payment timing.
In a net triangle, a recovery can make cumulative paid decrease.
A factor below one can therefore be legitimate on a signed net basis.
Trace recoveries by type rather than assuming every negative entry is salvage.
Deductibles, subrogation and reinsurance have different economic meanings.
Confirm whether the ledger records recovery as a negative payment or a separate balance.
Use gross-to-net reconciliations at each valuation and development age.
Do not infer net ultimate by subtracting current recoveries from gross ultimate.
Future recoveries also need a consistent estimate.
Recoverability and counterparty risk require separate investigation where material.
Source totals and case estimates must use the same recovery definition.
**Interview:** Why might a negative increment be expected?
**Answer:** A recovery or reversal may exceed payments in that period.
Review the underlying event before applying a data-cleaning rule.
This example does not provide a reinsurance valuation.

### 9. Changing expense inclusion

One paid triangle contains indemnity only.
Another includes allocated claim adjustment expense, abbreviated ALAE.
Suppose indemnity paid is 300 and ALAE paid is 30.
The combined paid amount is 330.
An indemnity-only ultimate of 450 implies outstanding of 150.
Subtracting the combined 330 from 450 produces 120.
That subtraction wrongly omits 30 of indemnity outstanding.
If a combined ultimate is 500, combined outstanding is 500 − 330 = 170.
The two estimates answer different liability-scope questions.
An expense may have a different reporting and settlement pattern from indemnity.
Build separate triangles where data quality and decision needs justify them.
Document whether defense cost is inside or outside policy limits.
Do not assume a particular legal or contractual treatment.
Review historical changes in expense coding and vendor invoices.
A coding migration can move expense across categories without changing total cash.
Trace the change at transaction and coverage level.
Reconcile the combined view to the sum of its components.
Unallocated expenses may require a separate methodology and are not inferred here.
If historical scope is inconsistent, reconstruct a comparable series where possible.
Otherwise disclose the break and assess alternative methods.
**Interview:** Is a lower residual always evidence of reserve release?
**Answer:** No; subtracting extra expense from an indemnity-only ultimate creates a false reduction.
Obtain the financial owner's definitions before using the calculation.
Preserve gross, recovery and expense dimensions in the output.

### 10. Unequal factor weighting

Two matched cohorts have cumulative paid 10 and 90 at age one.
At age two they have 20 and 108 respectively.
Their individual age-to-age ratios are 2.0 and 1.2.
The simple average is (2.0 + 1.2) / 2 = 1.6.
The volume-weighted ratio is (20 + 108) / (10 + 90) = 1.28.
The volume-weighted result gives the larger cohort nine times the weight.
Both statistics are calculated correctly but encode different weighting choices.
The ratio of sums is the standard matched volume-weighted construction used here.
It is not automatically the best predictor for the next cohort.
If the 90 cohort has unusually fast settlements, its dominance matters.
If the 10 cohort is noisy, equal weighting can also be unstable.
Examine the claim counts and composition underlying both cells.
Assess whether large losses or one-off recoveries drive either ratio.
Compare alternative windows without choosing the best-looking outcome retrospectively.
Selection criteria and exclusions must be recorded before final approval.
A single outlier can dominate both a small sample and its volume-weighted factor.
The denominator must be matched to the same eligible cohorts as the numerator.
Zero or negative starting values invalidate the usual convex-weight interpretation.
The result is dimensionless.
Multiplying a currency amount by a factor returns currency.
**Interview:** Why is 1.6 not the weighted factor?
**Answer:** It assigns equal weights to cohorts with different prior paid volumes.
Neither calculation gives a confidence interval or a tail estimate.

### 11. Missing versus zero

At a valuation date, AY1 has age-one paid 50 and age-two paid 75.
AY2 has age-one paid 40 and has not yet reached age two.
The matched factor is 75 / 50 = 1.5.
The AY2 age-two cell is unobserved, not a payment of zero.
Including AY2 only in the denominator gives 75 / 90 approximately 0.833.
That quantity is not the matched age-to-age factor.
Now consider an older AY3 with observed age-one paid 10 and age-two paid 0.
If valid on the signed basis, AY3 is an observed pair.
Including it gives (75 + 0) / (50 + 10) = 1.25.
Investigate why cumulative paid declined before accepting that observation.
It could reflect a reversal or recovery; it could also reflect bad data.
A genuine zero starting value creates an undefined individual ratio.
The aggregate ratio may still be computable with a nonzero combined denominator.
However, an individual-ratio weighted-mean explanation then needs care.
Store a separate observation mask rather than encoding missing cells as zero.
Define the observation mask from the valuation date and age boundaries.
Check partial periods rather than treating an incomplete year as fully observed.
An absent source partition must not masquerade as a genuine zero.
Reconcile the number of eligible cohort-age pairs as well as the money.
The data contract should distinguish null, zero and rejected records.
**Interview:** What if a missing future cell is filled with zero before factor fitting?
**Answer:** The eligible sample is contaminated and the development factor can be biased downward.
The correct remedy is to reconstruct observability, not merely floor the factor.

### 12. Partial development periods

Suppose an annual age-one cell represents twelve months of development.
A new cohort has only six months of observed payments.
Its six-month paid amount is 60.
Treating 60 as a full age-one cell makes it incomparable with twelve-month cells.
If a selected six-to-twelve-month factor is 1.5, the implied twelve-month paid is 90.
That 90 is a projection, not an observed twelve-month cell.
It must not enter a fitted historical factor as if it were observed.
Monthly or quarterly triangles can represent partial periods more faithfully.
They require consistent period boundaries and adequate data at each age.
Alternatively evaluate only fully matured periods and disclose the shorter sample.
If accident dates are spread through the year, age labels may represent approximations.
The cohort convention must specify whether age is measured from period start, end or individual occurrence.
Using policy year instead of accident year introduces another time definition.
Do not mix these conventions in a single selection without a mapping.
Check a sample of claims near the cohort boundary.
Measure elapsed time in the same calendar and timezone as the source.
Calendar-month arithmetic may not equal fixed-day arithmetic.
Reconcile finer-grained cells back to the annual totals.
Compare an annual and quarterly fit on the same frozen historical information.
Additional granularity does not automatically improve accuracy.
Sparse cells can make factor estimates less stable.
**Interview:** Can projected age-one values be treated as observed pairs?
**Answer:** No; doing so feeds the model's own assumption back into its fitted evidence.
Label derived values and keep them outside the observed triangle.

### 13. Factor selection after an outlier

Three eligible starting values are 100, 100 and 100.
Their next-age values are 120, 130 and 300.
Individual ratios are 1.2, 1.3 and 3.0.
The all-cohort volume-weighted factor is 550 / 300 approximately 1.8333.
Excluding the third cohort gives 250 / 200 = 1.25.
For a new cohort with 200 paid, these factors imply 366.67 and 250 ultimate respectively.
The implied difference in ultimate is approximately 116.67.
The exclusion is justified only if the third observation is not representative for the decision.
First identify whether a catastrophe, coding error or genuine large claim explains it.
Check whether similar events can recur in the future cohort.
Removing a genuine recurring exposure can understate the reserve.
Correct a source error in the source data rather than hiding it with an exclusion.
If the outlier is valid but nonrecurring, document the evidence and treatment.
A separate large-loss component may be preferable to unexamined deletion.
Reconcile the modeled base and the separately treated claim to the whole portfolio.
Check sensitivity to the threshold used to define a large claim.
Factor-window choices should be tested using frozen historical selections.
One extreme observation does not establish a stable trend.
The all-cohort estimate is also not automatically conservative in every future state.
Do not assign probabilities to the two outcomes without a justified model.
**Interview:** Should the 3.0 factor always be removed?
**Answer:** No; its cause and relevance to future claims determine its treatment.
Escalate unsupported material exclusions to the reviewer.

### 14. Large-loss carve-out

The observed cohort has 500 paid in total.
Of that amount, one large claim contributes 300.
The remaining attritional claims contribute 200.
Suppose the attritional selected remaining multiplier is 1.2.
Attritional ultimate is 200 × 1.2 = 240.
An independent claim-specific estimate sets the large claim's ultimate at 450.
Combined ultimate is 240 + 450 = 690.
Total outstanding is 690 − 500 = 190.
It decomposes into attritional outstanding 40 and large-claim outstanding 150.
Applying 1.2 to the entire 500 would give ultimate 600.
The 90 difference reflects different assumptions, not an observed error.
The large claim must be removed consistently from the historical attritional triangle.
The same claim must not be counted again in the aggregate modeled base.
Preserve the mapping between claim identifiers and carve-out selections.
If thresholds change over time, compare amounts on a consistent basis.
Consider whether claim-specific estimates include expenses and recoveries.
The claim-specific estimate may itself be uncertain and needs independent review.
The attritional pattern may change if the portfolio composition changes.
New large claims not yet reported are not captured by the known-claim estimate.
An additional approach may be needed for their future emergence.
Do not infer that a large-claim estimate of 450 is a cap.
**Interview:** What is the most common reconciliation mistake?
**Answer:** Adding the 450 claim ultimate to a triangle that already includes that claim.
Record the bridge from original total to carved and recombined estimates.

### 15. Tail factor sensitivity

Use the Section 5 projected modeled-age ultimates 180, 216 and 252.
Their sum before tail development is 648.
The latest observed paid total is 500.
With tail 1.00, total ultimate is 648 and outstanding is 148.
With tail 1.05, total ultimate is 648 × 1.05 = 680.4.
The resulting outstanding is 680.4 − 500 = 180.4.
The difference is 32.4.
This comparison applies the same selected tail to every cohort's modeled-age amount.
It assumes all three cohorts share that tail and financial basis.
It does not establish that 1.05 is the correct tail.
The observed triangle contains no direct information beyond age three.
Older claims, external data and portfolio-specific evidence may help assess the tail.
Any external pattern needs a compatibility assessment.
Coverage, jurisdiction, limit and claims handling can all differ.
Check whether the last observed age is genuinely mature for each segment.
A tail multiplier can reflect additional development after the final modeled age.
It should not be used to hide unmodeled expenses or a scope mismatch.
Separate tail uncertainty from observed age-to-age variability.
If the tail is estimated from the same sparse data, acknowledge dependence.
The range between 148 and 180.4 is a scenario range, not a confidence interval.
**Interview:** Why does AY1 gain outstanding in the 1.05 scenario?
**Answer:** The new assumption allows development beyond its last observed modeled age.
Reconcile the scenario at the cohort and portfolio levels.

### 16. Prior selection in paid Bornhuetter–Ferguson

A cohort has actual paid 60.
Its independently selected expected ultimate is 200.
The expected paid proportion at its age is 0.30.
The expected unpaid proportion is 0.70.
Paid BF outstanding is 200 × 0.70 = 140.
Paid BF ultimate is 60 + 140 = 200.
If the prior ultimate is 240 instead, outstanding is 168.
The prior change increases outstanding by 28.
Alternatively, keeping the prior 200 and using q = 0.40 gives outstanding 120.
The payment-pattern change reduces outstanding by 20.
The two changes affect different assumptions and require different evidence.
If q comes from the same selected age-to-ultimate factor, check its tail basis.
For a positive factor of 2.5, the implied paid proportion is 1 / 2.5 = 0.4.
This reciprocal interpretation assumes the usual nonnegative development setting.
A prior can be based on exposure and an expected loss cost.
For example, 100 exposure units at 2 per unit gives 200 prior ultimate.
The exposure and expected loss cost must match the cohort and coverage.
Do not call a prior independent merely because it appears in another spreadsheet.
Its inputs may share the same historical claims information as the triangle.
When actual paid is zero, BF can still produce a positive reserve from the prior.
That does not establish the prior's accuracy.
**Interview:** Does BF eliminate development-pattern risk?
**Answer:** No; its unpaid proportion still depends on the selected pattern.
Validate both prior and pattern using information available at the valuation.

### 17. Paid and incurred comparison

For a cohort, cumulative paid is 100 and case reserve is 80.
Current incurred is therefore 180.
Suppose paid development gives ultimate 250.
Paid-method outstanding is 250 − 100 = 150.
Its broad incurred-based residual on that same ultimate is 250 − 180 = 70.
The reconciliation is 80 + 70 = 150.
Now suppose an independent incurred development method gives ultimate 230.
On that different ultimate, broad residual is 230 − 180 = 50.
The paid-method and incurred-method ultimate estimates differ by 20.
Do not reconcile 150 to 80 + 50: those numbers use different ultimates.
The difference may arise from payment timing, case adequacy or factor selection.
It may also arise from a gross/net or expense mismatch.
First align the cohort, valuation and financial scope.
Then compare selected development patterns and case practices.
Check claim counts and average case per open claim.
Review large claims separately if they dominate the divergence.
An increase in case reserves can make incurred development look faster.
That observation alone does not establish that ultimate claims cost changed.
Compare subsequent settlement evidence where available.
Neither method is inherently superior without validation against the use case.
**Interview:** Why does case plus incurred residual fail to equal paid outstanding here?
**Answer:** The residual was calculated from 230 while the paid outstanding used 250.
Use one ultimate for the accounting decomposition.

### 18. Calendar shock and diagonals

Consider three successive accident cohorts with observed increments in one calendar year.
At ages three, two and one, respectively, their payments are 30, 50 and 80.
The total on that calendar diagonal is 160.
Suppose comparable expected increments were 20, 40 and 60.
The actual-to-expected ratios are 1.5, 1.25 and approximately 1.333.
The aggregate ratio is 160 / 120 approximately 1.333.
The common calendar timing suggests a shared event is worth investigating.
It does not establish that a calendar shock caused the increase.
The cohorts could differ in severity, claims mix or operational workload.
Compare paid counts, average payment size and settlement speed.
Review contemporaneous claims-handling and legal environment records.
Check whether inflation indices are relevant to the claims being settled.
A system migration can also cause delayed postings to cluster in one period.
Inspect subsequent periods for compensating reductions.
Map calendar diagonals only after verifying consistent period alignment.
For unequal periods, use actual dates rather than a visual diagonal alone.
Removing a diagonal without a defensible causal explanation may discard valid experience.
If adjusting, record an unadjusted and adjusted triangle.
Reconcile the aggregate amount of the adjustment.
Backtest the adjustment on earlier known shocks where data permit.
A purely visual pattern has no quantified significance in this example.
**Interview:** What does a diagonal pattern prove?
**Answer:** It identifies shared timing, not the cause or persistence of the effect.
Document competing explanations and the evidence that discriminates them.

### 19. Claims workflow slows payments

In an earlier period, 80 of 100 comparable claims are paid within six months.
After a new approval step, only 50 of 100 are paid within six months.
The observed six-month completion rate falls from 80% to 50%.
The difference is 30 percentage points, not 30%.
Suppose average eventual payment is unchanged at 10 per claim.
Expected six-month paid would fall from 800 to 500 under these simplified assumptions.
The 300 reduction in early paid need not reduce ultimate loss.
A paid chain ladder trained on the faster old pattern may underproject.
Its latest paid input is lower because of delayed settlement.
That direction is conditional; other simultaneous changes can reverse the observed result.
Check whether the claims are comparable by age, coverage and severity.
The 100-claim samples are illustrative and no uncertainty estimate is supplied.
Compare open inventory, approval queue age and payment-date distributions.
Use operational release notes to identify the actual change date.
Inspect whether payments later catch up in subsequent development.
Check whether case reserves increased while payment speed fell.
BF or an incurred-based method can provide a complementary estimate.
Each alternative has its own prior or case-reserve assumptions.
Do not mechanically scale paid without a defensible model of the delay.
Backtest with historical as-of data around the workflow transition.
Escalate if the break is material and cannot be quantified.
**Interview:** Does lower paid prove lower ultimate claims cost?
**Answer:** No; it may reflect slower payments on the same obligations.
Record the operational evidence and any selected adjustment separately.

### 20. Case strengthening

At one valuation a cohort has paid 100 and case reserve 50.
Its incurred amount is 150.
An adjuster review raises case reserves by 30 without any payment.
Incurred rises to 180.
The incurred increase is 30, although paid remains 100.
Suppose ultimate is independently held at 250 for illustration.
Broad IBNR falls from 250 − 150 = 100 to 250 − 180 = 70.
Paid outstanding remains 250 − 100 = 150.
Before strengthening, 50 + 100 = 150.
After strengthening, 80 + 70 = 150.
The reserve categories moved while total paid outstanding stayed constant.
In reality the independent ultimate may also change after new information.
Do not assume case strengthening is merely a classification shift.
Analyze which claims received revisions and why.
Compare case adequacy by development age and claim type.
Look for changes in adjuster instructions, authority limits and system defaults.
A one-time case uplift may create an incurred development break.
Paid development is not immune to simultaneous payment process changes.
Reconcile snapshots to case movement transactions where available.
Backtests should preserve the earlier case estimate rather than overwrite it.
Separate the date the information became known from a backdated effective date.
**Interview:** Why can broad IBNR fall without reducing total outstanding?
**Answer:** More of the same projected future payment is allocated to case reserves.
The result depends on holding ultimate fixed and using a common basis.

### 21. Claim closure and reopening

A cohort starts with 100 open claims.
During the period 30 close and 5 previously closed claims reopen.
If no new claims are reported, the ending open count is 100 − 30 + 5 = 75.
A naive calculation that ignores reopenings gives 70.
If ten new claims are reported, the ending count becomes 85.
Closed-without-payment and closed-with-payment claims should be distinguished.
A reopening can reflect an administrative correction or a substantive new development.
Claim-status history must retain each transition and effective date.
Current status alone cannot reconstruct closure rates at past valuations.
Count claims consistently when multiple coverages belong to one occurrence.
Check whether a reopened claim receives a new identifier.
If identifiers change, entity linkage can create a false new report.
Compare reopening rates by closure cohort and elapsed time.
A surge may reflect premature closure or a change in coding.
It may also arise from genuine legal or medical development.
Follow reopened claims to paid and case movements.
Investigate whether case reserves are reset on reopening.
Claims closed at the valuation can still generate future payments.
Thus open-claim count alone does not capture all future exposure.
For counts-based methods, preserve the treatment of zero-payment closures.
**Interview:** Why is a closed claim not always fully developed?
**Answer:** It can reopen or receive a later adjustment under the defined process.
The numerical count bridge does not estimate the probability of reopening.

### 22. Reporting delay and strict IBNR

At valuation, 80 claims from a cohort are reported.
A separate reporting-delay analysis expects 20 more claims.
Suppose the expected ultimate cost per not-yet-reported claim is 5.
The illustrative strict unreported component is 20 × 5 = 100.
Known claims may also need 60 of development beyond their case estimates.
Broad IBNR under the stated convention is 100 + 60 = 160.
If current case reserves are 140, total paid outstanding is 140 + 160 = 300.
These components require matching definitions and no overlapping claims.
The 20 expected reports and cost of 5 are assumptions, not observations.
Reporting and payment development are different processes.
Changes in reporting channels can change report delays before payment delays change.
Compare notification date to occurrence date at claim level.
Late-entered notifications can masquerade as genuine late reporting.
Record the system entry timestamp as well as the reported date.
Severity of unreported claims may differ from already reported claims.
Using the observed average cost without adjustment can be misleading.
Counts may include zero-payment claims depending on the definition.
Use a compatible severity denominator and expense basis.
The aggregate paid triangle alone does not identify the strict unreported component.
Individual claim histories or other justified information are needed.
**Interview:** Can the Section 5 total of 148 be split into pure IBNR and IBNER?
**Answer:** No; paid data alone do not identify that decomposition.
Disclose the chosen terminology in every management report.

### 23. Claim counts and severity

A cohort has 50 reported claims and 400 of incurred loss.
Average incurred per reported claim is 400 / 50 = 8.
Suppose 10 of those claims are closed without payment.
Average per claim excluding those ten is 400 / 40 = 10.
Both averages use the same loss numerator but different denominators.
Neither is directly comparable with an average using a different inclusion rule.
Suppose ultimate claim count is expected to be 60.
If ultimate average cost per claim is 12, implied ultimate is 60 × 12 = 720.
Subtracting paid, not incurred, gives a paid outstanding estimate.
If paid is 250, that outstanding is 720 − 250 = 470.
Subtracting incurred 400 instead gives broad incurred residual 320.
The difference is the 150 case reserve implied by incurred minus paid.
Projected claim count and severity must have consistent claim definitions.
They can be dependent, especially when reporting differs by claim size.
Closure rates, zero claims and reopenings affect the count projection.
Large claims can dominate severity averages.
Review robust summaries and claim-level distributions in addition to the mean.
Do not multiply projected counts by a severity measure on an incompatible maturity basis.
Segmenting can improve homogeneity but reduce sample size.
Use backtesting to evaluate the combined forecast, not just its components.
**Interview:** Why can count times severity fail to reconcile to loss?
**Answer:** The count and severity numerator may cover different claim populations or periods.
The example's 720 depends entirely on the selected count and severity.

### 24. Exposure-based expected ultimate

An immature accident cohort has 1,000 earned risk-years.
The selected expected ultimate loss cost is 0.8 per risk-year.
The expected ultimate is 1,000 × 0.8 = 800.
Suppose observed paid is 100.
An expected paid fraction of 0.25 gives expected unpaid fraction 0.75.
Paid BF outstanding is 800 × 0.75 = 600.
The BF ultimate is 100 + 600 = 700.
That is lower than the prior ultimate because actual paid is below expected paid of 200.
The prior's loss cost must be on the same ultimate and coverage basis.
Earned exposure should match the accident period and insured population.
Policy counts are not automatically equivalent to earned risk-years.
Changes in deductibles, limits and coverage may require prior adjustment.
Rate changes affect premium but do not automatically change exposure.
Inflation and changes in insured mix may affect expected loss cost.
If exposure data arrive late, preserve the valuation-date estimate and later revision.
An uncertain exposure count contributes uncertainty to the prior.
The assumed paid fraction must incorporate the selected tail.
A low actual paid amount is not sufficient evidence that the prior is excessive.
Conversely, BF can preserve an obsolete prior for too long.
Compare predicted and observed emergence across multiple historical cohorts.
**Interview:** Why is BF ultimate 700 rather than 800?
**Answer:** Actual paid of 100 replaces the prior's expected paid component of 200.
All estimates remain hypothetical and conditional.

### 25. Average versus volume weighting under trend

Three matched pairs have starting amounts 50, 100 and 200.
Their later amounts are 75, 140 and 260.
Their individual ratios are 1.5, 1.4 and 1.3.
The simple average is 4.2 / 3 = 1.4.
The volume-weighted factor is 475 / 350 approximately 1.3571.
The last-two-cohort volume-weighted factor is 400 / 300 approximately 1.3333.
The declining ratios may indicate a trend toward faster development.
They may also reflect cohort mix or random variation.
Larger recent cohorts have more weight in the all-year volume-weighted selection.
Dropping older cohorts further emphasizes recency.
Recency reduces historical sample size and can increase sensitivity to one period.
Inspect claim-level mix and operational changes before selecting a window.
Use a fixed, documented window rule for retrospective evaluation.
If an exceptional year is excluded, document its reason separately from the window.
Do not interpret a falling factor as lower ultimate loss without considering paid-to-date.
For a latest paid amount of 100, factors 1.4 and 1.3333 imply about 140 and 133.33 ultimate.
The approximately 6.67 difference is a selection sensitivity.
It is not the empirical standard error of either estimate.
Rounding individual ratios before averaging can introduce avoidable error.
Keep full precision internally and round outputs at presentation.
**Interview:** Why can the all-year weighted factor be below the simple average?
**Answer:** The largest starting amount has the smallest individual ratio.
Review the reason for that relationship before assuming it will continue.

### 26. Gross and net reinsurance

Suppose gross ultimate is 1,000 and gross paid is 600.
Gross outstanding is 400.
An independent reinsurance analysis estimates ultimate recoveries of 300.
Recoveries already collected are 180.
Outstanding reinsurance recoveries are 300 − 180 = 120.
Net ultimate is 1,000 − 300 = 700.
Net paid is 600 − 180 = 420.
Net outstanding is 700 − 420 = 280.
The bridge is gross outstanding 400 minus recovery outstanding 120 = 280.
The calculation assumes matching cohorts, currency and valuation dates.
The recoveries here are reinsurance recoveries, not salvage.
Actual contracts may contain attachments, limits, exclusions and reinstatements.
Those features cannot be inferred from gross aggregate loss alone.
Claims may attach differently across coverage or treaty years.
Counterparty collectability can introduce another financial question.
A net triangle can hide a change in treaty terms over time.
Compare gross and recovery triangles as separate diagnostics where feasible.
Check whether a recovery is booked when billed, collected or estimated.
Those statuses are not interchangeable.
Document the boundary between actuarial gross estimates and contract-specific recoverables.
**Interview:** Why is gross outstanding minus ultimate recoveries wrong?
**Answer:** It subtracts recoveries already collected as well as future recoveries.
Use outstanding recovery, 120, for the outstanding-to-outstanding bridge.

### 27. Discounting future cash flows

Suppose undiscounted future payments are 100 in one year and 100 in two years.
Use an illustrative annual effective discount rate of 10%.
The first present value is 100 / 1.1 approximately 90.91.
The second is 100 / 1.1² approximately 82.64.
Total present value is approximately 173.55.
Undiscounted outstanding is 200.
The difference is approximately 26.45.
This is a calculation under a specified cash-flow schedule and rate.
It is not an assertion about permitted reporting or regulatory discount rates.
Discounting total outstanding once at the two-year rate would give approximately 165.29.
That would incorrectly treat the first payment as if it occurred in year two.
The schedule needs payment dates, not just an ultimate amount.
Cash-flow timing and discount rate both contribute uncertainty.
If payment timing changes, the present value changes even with the same undiscounted total.
Currency and discount-rate denomination must be consistent.
Avoid mixing nominal cash flows with a real discount rate.
Recoveries and expenses require a matching timing and basis.
Retain the undiscounted estimate as a reconciliation control.
Identify whether a discount convention uses beginning, middle or end of period.
Actual accounting and actuarial requirements depend on the assignment.
**Interview:** Does a 10% annual rate mean subtracting 10% from 200?
**Answer:** No; each payment is discounted for its own time to payment.
The applicable framework and yield curve require separate professional assessment.

### 28. Roll-forward of unpaid claims

At the start of a year, estimated unpaid for old accident years is 300.
During the year, 120 is paid on those old years.
Before assumption changes, the mechanical remaining amount is 180.
At year-end the selected old-year unpaid estimate is 210.
The adverse old-year re-estimation is 210 − 180 = 30.
Equivalently, the old-year ultimate estimate increased by 30.
Suppose the new accident year's selected unpaid is 90.
Total ending unpaid is 210 + 90 = 300.
The bridge is 300 − 120 + 30 + 90 = 300.
This is a simplified paid-based bridge without acquisitions or scope changes.
Separate prior-year development from new-year claims.
Do not interpret unchanged total unpaid as zero development.
The 30 adverse movement and 90 new-year addition are offset by 120 paid.
If currency rates change, include a separate translation component.
If portfolio transfers occur, include a separate transfer component.
If historical case classification changes, bridge its effect independently.
Use the same gross/net and expense scope at both dates.
The estimate of 210 remains subject to new assumptions.
Keep actual payment records separate from management overlays.
Require an owner and rationale for each nonmechanical bridge item.
Compare the bridge with finance movements and investigate differences.
**Interview:** Can an unchanged closing reserve hide adverse development?
**Answer:** Yes; 30 of adverse development is offset by other movements here.
The bridge does not itself identify why the estimate changed.

### 29. One-year forecast backtest

At the start of a year a model forecasts 100 of payments during that year.
Actual payments are 120.
Define error as actual minus forecast.
The error is 120 − 100 = 20.
The absolute error is 20.
The signed percentage error relative to the forecast is 20 / 100 = 20%.
The signed percentage error relative to actual is 20 / 120 approximately 16.67%.
The denominator must be named because these percentages differ.
Forecast accuracy for one-year payments is not the same as ultimate reserve accuracy.
A model can forecast next year's cash flow well and still miss the tail.
Preserve the forecast and input data available at the start of the year.
Do not update its factor selection using year-end outcomes.
Use a consistent treatment of recoveries and currency.
Separate actual transactions from retrospective corrections.
Repeat over multiple historical origins and segments where available.
Compare signed bias, absolute error and material financial impact.
Do not average percentage errors blindly when denominators are near zero.
Measure uncertainty in evaluation metrics when the sample is small.
The single 20 error does not establish a persistent bias.
Record any changes to the observed outcome definition.
If the model was not deployed historically, distinguish a reconstructed backtest from live performance.
**Interview:** Is this a test of the final unpaid estimate?
**Answer:** Only indirectly; it tests a one-year payment forecast on the stated scope.
For ultimate validation, choose a mature outcome and a suitable evaluation horizon.

### 30. Immature outcomes in a backtest

An old forecast predicts ultimate of 500 for an accident cohort.
Three years later, cumulative paid is 400.
The remaining case and tail are not yet settled.
Calling 500 − 400 = 100 an overestimate assumes no future payments.
That assumption is not supported by the stated information.
Suppose a later independent estimate of remaining paid is 70.
A provisional evaluated ultimate is 400 + 70 = 470.
The provisional forecast error is 470 − 500 = −30.
It is not a final realized error because 70 is itself estimated.
If the eventual remaining payments total 130, realized ultimate becomes 530.
The final error would then be 530 − 500 = 30.
The sign of the provisional error has changed.
Maintain a maturity label for each evaluation target.
Use fully settled cohorts when available, but check selection bias.
Settled claims may differ systematically from open long-tail claims.
Track successive restatements rather than treating an intermediate estimate as truth.
Reserve deterioration after the evaluation date must be attributed to the later period.
Use a common valuation basis for both forecast and outcome.
For long-tail business, one-year claims development results may be a useful complementary target.
They answer a different question from ultimate settlement.
The appropriate horizon depends on the decision being evaluated.
**Interview:** Why is a three-year-old ultimate estimate not necessarily ground truth?
**Answer:** Future payments and case revisions can still materially change ultimate.
Disclose the evaluation target's maturity and any remaining estimation.

### 31. Historical leakage through latest master data

A model was notionally run at December 31 of year one.
In March of year two, a claim is recoded from routine to severe.
The claim ultimately costs 100 rather than the 20 anticipated at year-end.
A backtest extracts the current master table after the March recoding.
It assigns the claim to the severe category at the earlier valuation.
If the severe category selects ultimate 100, the backtest appears exact.
An as-known model using the routine category would have estimated 20.
Its retrospective error against the 100 outcome is 80.
The apparent accuracy improvement is created by future information.
The issue persists even if the transaction amounts are correctly cutoff.
Each joined feature must have an as-known time boundary.
An effective-dated record may still leak if entered later and backdated.
Track both valid time and system-availability time.
An immutable valuation snapshot is a practical control.
Alternatively use a bitemporal history with explicit query rules.
Check join cardinality and the date predicates on the feature table.
Reconstruct the feature available to the original decision maker.
Do not assume a data warehouse's load date equals claim-system availability.
Record the source interface latency when deciding the as-known timestamp.
Test a claim deliberately corrected after the historical valuation.
The reconstructed earlier view must retain its earlier category.
**Interview:** Are cutoff transaction dates alone enough to prevent leakage?
**Answer:** No; current master attributes can still contain information learned later.
Restated histories are useful for different purposes but must be labeled.

### 32. As-of data contract and reproducible run

A reserving run should identify its cohort origin and development convention.
It should name the valuation date and the extraction availability cutoff.
It should declare whether the view is as-known or restated.
The contract should identify gross or net treatment.
It should identify included expense and recovery categories.
It should state currency conversion and discount conventions.
Each source should have a named authoritative owner.
Each source extract should carry a snapshot identifier and checksum.
Claim and transaction keys must include their source namespaces.
Duplicate rules should distinguish retries from financial corrections.
The mapping from claims to cohort and segment must be effective-dated.
Observation masks should distinguish unavailable cells from observed zeros.
Independent control totals should be captured by source and financial category.
The input contract should specify acceptable discrepancies and escalation.
Thresholds are governance decisions, not universal actuarial constants.
An exception report should include amount, count, cause and owner.
Material exceptions must block release or receive authorized disposition.
The run should record code version and dependency versions.
Selected factors, exclusions and tail assumptions should be versioned.
Manual overrides should identify author, reason, approver and timestamp.
Outputs should include both aggregate and cohort-level reconciliation.
Retain artifacts under the organization's security and retention rules.
Replay from the retained inputs should reproduce the same selected output.
Differences caused by code or dependency changes require an explained bridge.
**Interview:** Why is a versioned model file insufficient?
**Answer:** Inputs, mappings, assumptions and manual selections also determine the result.

### 33. Reconciliation with a tolerance

The independent ledger reports paid of 10,000.
The triangle extract reports 9,970 on the same scope.
The signed difference, extract minus ledger, is −30.
The absolute difference is 30.
Relative to the ledger, the absolute difference is 30 / 10,000 = 0.3%.
An illustrative tolerance is 0.2% and 10 currency units.
The difference exceeds both of those illustrative limits.
The tolerance is an example, not a regulatory or actuarial standard.
Suppose investigation identifies a missing source batch of 25.
The remaining unexplained difference is 5.
The 25 is a completeness failure, not a modeling assumption.
Restoring the batch brings extracted paid to 9,995.
The residual still needs a documented disposition.
If an unrelated overstatement of 25 and omission of 25 offset, total reconciliation can appear exact.
Therefore compare by coverage, source, transaction type and period.
Also compare counts and key-level exceptions.
Zero ledger totals need an absolute rather than percentage threshold.
The correct denominator must be specified for nonzero totals.
A small aggregate difference can be material for a small segment.
Escalation should reflect both financial materiality and systemic risk.
The person approving a threshold breach should not silently change the data.
**Interview:** Does passing an aggregate tolerance prove the triangle is complete?
**Answer:** No; offsetting errors can pass an aggregate test.
Retain the original control, corrected amount and disposition.

### 34. Model change bridge

Last quarter's selected total outstanding was 400.
New payments of 50 reduce the mechanical outstanding to 350.
Updated data and factors increase the estimate by 30.
A revised tail increases it by 20.
A change in expense scope adds 40.
The new estimate is 350 + 30 + 20 + 40 = 440.
The movement from 400 is +40.
The bridge is −50 + 30 + 20 + 40 = +40.
These components depend on the order in which changes are applied.
Factor and tail effects can interact multiplicatively.
For example, a factor increase of 10% and tail increase of 5% together multiply by 1.155.
Their combined effect is 15.5%, not 15%.
An order-based bridge allocates that interaction differently depending on ordering.
Document the sequence or use an agreed interaction allocation.
Do not represent the individual bridge components as uniquely causal.
The scope increase is not model deterioration.
New payments are cash movements, not a model error by themselves.
Reconcile the bridge to the published opening and closing estimates.
Freeze inputs while measuring each change to avoid unintended confounding.
Retain intermediate versions for reviewer reproduction.
If the bridge cannot be reproduced, hold the release pending investigation.
**Interview:** Why might two valid bridge orders yield different factor effects?
**Answer:** The factor and tail changes interact, so attribution depends on sequence.
The closing total should nevertheless reconcile.

### 35. Process and parameter uncertainty

Consider a conditional model with expected future payment of 100.
Suppose the future process has conditional standard deviation of 20.
If parameters were known, that process variability would remain.
Now suppose the estimate of the conditional mean is also uncertain.
Illustratively its standard deviation is 10.
If these two sources are independent under the chosen model, variances add.
The combined variance is 20² + 10² = 500.
The combined standard deviation is approximately 22.36.
Adding the standard deviations would incorrectly give 30.
Independence is an illustrative assumption, not guaranteed in reserving.
If covariance is nonzero, include twice the covariance.
Model selection and structural break risk may not be included in either component.
Tail risk and data quality may also be omitted.
A standard deviation alone does not specify a distribution.
Do not convert 100 ± 1.96 × 22.36 into a 95% interval without further assumptions.
Claims payments can be skewed and bounded in ways that make normal approximations poor.
Clarify whether uncertainty concerns the mean, future outcome or reserve change.
The forecast horizon and conditioning information must be explicit.
Explain whether uncertainty in selected factors is reflected.
Also explain whether model and expert judgment uncertainty is reflected.
**Interview:** Why is a deterministic range from two tail assumptions not a 95% interval?
**Answer:** It does not assign a validated probability distribution or coverage.
The arithmetic example is not an estimate of the Section 5 reserve variability.

### 36. Dependence between cohorts

Suppose two future cohort payments each have variance 100.
If their covariance is zero, variance of their sum is 200.
The aggregate standard deviation is approximately 14.14.
If covariance is 50, aggregate variance is 100 + 100 + 2 × 50 = 300.
The aggregate standard deviation is approximately 17.32.
Positive covariance raises aggregate uncertainty relative to the independence case.
Common inflation or a legal change can affect several cohorts.
Common model parameters can also induce dependence in estimates.
The covariance of 50 is a hypothetical input, not an empirical measurement.
Adding individual standard deviations gives 20 in either case.
That addition is not the general variance rule.
Conversely, adding variances without covariance assumes zero covariance.
An aggregate range must specify how shared shocks are modeled.
Segmenting a portfolio does not make its segments independent.
Check whether reinsurance layers introduce nonlinear dependencies.
Correlation can change in extreme events.
Historical correlations from a short sample may be unstable.
Scenario analysis can complement, but not automatically calibrate, a stochastic model.
Avoid summing segment percentile estimates and calling the result an aggregate percentile.
Joint simulations need coherent common shocks and basis definitions.
Document which dependencies are included and excluded.
**Interview:** What extra term is required when summing two dependent variances?
**Answer:** Twice their covariance.
This calculation concerns the sum, not the correlation of point estimates alone.

### 37. Scenario weights versus confidence intervals

Suppose three internally constructed reserve scenarios are 100, 150 and 250.
Assign explicit subjective weights of 0.2, 0.5 and 0.3.
The weights sum to one.
The weighted mean is 0.2 × 100 + 0.5 × 150 + 0.3 × 250 = 170.
This is a mean under the stated discrete scenario distribution.
It is not the same as the middle scenario of 150.
The range of scenarios is 100 to 250.
That range is not automatically a 95% confidence interval.
The probabilities may omit scenarios outside the three selected points.
The weights can encode judgment rather than measured frequencies.
Document who selected them and on what evidence.
Changing weights to 0.4, 0.5 and 0.1 gives a mean of 140.
The 30 difference shows weight sensitivity.
Scenarios should be internally coherent across inflation, payment timing and severity.
Do not mix an undiscounted scenario with a discounted one.
If an economic shock affects multiple segments, apply it consistently.
Record the decision the scenario distribution is intended to support.
A regulatory capital quantile requires its own applicable framework.
A forecast distribution requires validation and calibration evidence.
This simple discrete example provides neither.
**Interview:** Does a weighted mean of 170 establish a best estimate?
**Answer:** Only relative to the specified scenarios and weights; their adequacy remains open.
Keep the scenario assumptions visible beside the summary.

### 38. Interval coverage and width

Ten historical forecasts each publish an 80% prediction interval.
Seven realized outcomes lie inside their respective intervals.
The observed coverage is 7 / 10 = 70%.
The observed shortfall from the nominal 80% is 10 percentage points.
Ten cases are too few by themselves to diagnose the interval model reliably.
The intervals must target comparable outcomes and horizons.
Check whether outcomes are fully mature and genuinely out of sample.
A very wide interval may achieve high coverage but have little decision value.
Report interval width alongside coverage.
For example, intervals [0, 300] and [90, 110] have widths 300 and 20.
Their usefulness depends on outcomes, decision costs and calibration.
Coverage across the entire portfolio can conceal poor coverage in large-loss segments.
Examine relevant slices where sample size supports meaningful inference.
Avoid presenting unstable slice estimates as conclusive.
Check whether overlapping forecast origins create dependent outcomes.
Dependence affects uncertainty in the observed coverage rate.
An 80% interval does not mean 80% of a particular reserve amount is certain.
It describes the procedure's stated predictive or inferential interpretation.
Distinguish a prediction interval for future outcomes from an interval for a parameter.
Investigate systematic misses above versus below the interval.
Use frozen intervals rather than intervals revised after seeing the outcomes.
**Interview:** Does 7 of 10 prove the model is undercalibrated?
**Answer:** No; sampling variation and dependence must be considered.
The example is a diagnostic trace, not measured insurer performance.

### 39. Mack model boundary

**SOURCED:** Mack's 1993 paper derives a distribution-free standard-error calculation for chain-ladder reserve estimates under its stated assumptions.
Primary source: [Mack, *ASTIN Bulletin* 23(2), 1993](https://www.casact.org/sites/default/files/database/astin_vol23no2_213.pdf).
Distribution-free does not mean assumption-free.
The model assumes a specified conditional mean development relationship.
It also imposes a specified conditional variance structure and independence across origin periods.
Consult the paper for the exact conditions and estimator.
A structural calendar shock affecting several cohorts challenges independence.
Changing settlement practice can challenge the conditional mean relationship.
Sparse late-age pairs limit the information available for variance estimation.
The Section 5 age-two-to-three factor has only one observed pair.
That single pair cannot by itself support a conventional sample variance estimate at that age.
Additional assumptions or external information would be needed for a complete stochastic calculation.
Mack's 1999 paper discusses recursive calculations and tail factors.
Primary source: [Mack, *ASTIN Bulletin* 29(2), 1999](https://www.casact.org/sites/default/files/database/astin_vol29no2_361.pdf).
Tail parameter uncertainty should not be assumed absent merely because a point tail was selected.
An estimated standard error is not automatically a percentile.
Converting it to an interval requires an appropriate predictive distribution or other justified procedure.
The reported variability is conditional on the model and its included risk sources.
It need not include data defects, structural breaks or all model uncertainty.
Compare predicted uncertainty against rolling historical outcomes where feasible.
Examine residuals by age, cohort and calendar period.
Residual diagnostics are limited when the triangle contains few observations.
**Interview:** Does “distribution-free” guarantee a valid 95% interval?
**Answer:** No; assumptions remain, and a standard error alone does not define a 95% predictive interval.

### 40. Bootstrap residuals and limitations

A residual bootstrap begins with a fitted model and its residuals.
The residual construction must match the model's variance assumptions.
For each simulated dataset, a fit can be repeated to reflect parameter estimation.
Future process variation can then be simulated under an explicit process model.
Reserve outcomes across simulations form an empirical distribution conditional on those choices.
The number of simulations controls Monte Carlo noise, not model misspecification.
For an illustrative probability of 0.5 estimated from 10,000 independent draws,
the Monte Carlo standard error is approximately sqrt(0.5 × 0.5 / 10,000) = 0.005.
That is 0.5 percentage points for the estimated probability.
It says nothing about whether the model matches real claims.
Residual resampling can be invalid under strong calendar dependence.
Resampling individual cells may destroy a common shock affecting a diagonal.
Negative increments and recoveries can conflict with some process distributions.
Sparse triangles may provide too few residuals for stable resampling.
Tail uncertainty must be explicitly included if it is to appear in the output.
Check whether the method preserves exposure, financial scope and signs.
Store the random seed and software versions for reproducibility.
Record the number of simulations and convergence diagnostics.
Compare repeated runs to assess Monte Carlo variability.
Do not select a process distribution solely because software supplies it.
Assess backtest calibration and operational plausibility.
**Interview:** Does increasing simulations from 10,000 to 1,000,000 fix a structural break?
**Answer:** No; it reduces simulation noise around a potentially misspecified model.
This is a conceptual procedure, not a fitted bootstrap of the example triangle.

### 41. High-severity concentration

A cohort has five reported claims with paid amounts 10, 10, 10, 10 and 60.
Total paid is 100.
The largest claim contributes 60% of observed paid.
The mean paid per claim is 20.
The median paid per claim is 10.
The difference indicates concentration in this constructed sample.
Neither mean nor median alone describes future ultimate cost.
The largest claim could still have substantial case reserves.
A factor estimated from this cohort can be dominated by its development.
Trace its next-age movement before attributing the factor to the whole portfolio.
Separate analysis may be justified if it remains reconcilable to the total.
The treatment of not-yet-reported large claims also matters.
Review policy limits and deductibles on the same basis.
An observed maximum is not a maximum possible loss.
Small claim counts make distribution estimates fragile.
Check whether the same event generated multiple related claims.
Those claims may be dependent even when their identifiers differ.
Capture catastrophe and event identifiers where available.
The threshold for separate treatment should be documented in advance when possible.
Test sensitivity to plausible thresholds.
Avoid reusing a threshold from another portfolio without checking comparability.
**Interview:** Why can the median be misleading for reserve totals?
**Answer:** Aggregate cash depends on the tail, and the median ignores its size.
The example is not a fitted severity distribution.

### 42. Legal limit change

An old cohort has 100 claims each subject to a limit of 10.
Its maximum aggregate before expenses and other contract effects is 1,000.
A new cohort has 100 otherwise similar claims with limits of 20.
Its analogous maximum aggregate is 2,000.
This arithmetic is an upper-bound illustration under simplified per-claim assumptions.
It does not imply either cohort will reach its limit.
An old paid pattern may not describe the new limit structure.
Claims near the old limit may develop differently under the new contract.
Record effective dates of coverage and limit changes.
Review whether limits apply per occurrence, per claim or in aggregate.
Defense costs and deductibles can change the effective amount.
The actual contract language controls; this sheet does not interpret it.
Segmenting by limit can reduce comparability problems.
However, smaller cells can make development factors noisy.
External or claim-level information may be required.
The exposure base should identify the distribution of limits.
Check whether old cohorts were restated to current contract terms.
Do not mechanically scale all claims by the ratio of limits.
Many claims may never reach either limit.
Assess the share of payments and case reserves near each attachment or limit.
Use a severity-sensitive scenario rather than a blanket assumption.
**Interview:** Does doubling the limit double expected loss?
**Answer:** No; it changes the cap, while expected loss depends on the severity distribution and contract.
The 1,000 and 2,000 bounds omit explicitly excluded contract features.

### 43. Inflation and payment pattern interaction

Suppose an outstanding payment would be 100 at today's price level.
If claim-cost inflation is 5% per year for two years, its nominal amount is 100 × 1.05² = 110.25.
At a 10% annual discount rate for two years, its present value is 110.25 / 1.1² approximately 91.12.
If inflation were omitted, present value would be 100 / 1.1² approximately 82.64.
The difference is approximately 8.47.
These are hypothetical fixed rates and a single payment date.
Claims inflation may differ from a general consumer price index.
Different expense categories can have different inflation sensitivities.
Payment delays expose future cash flows to inflation for longer.
Discounting also depends on that timing.
Therefore timing, inflation and discount assumptions interact.
A historical nominal triangle may already embed historical inflation.
Applying an extra historical inflation adjustment without a clear model can double count.
Separate the adjustment to past data from assumptions about future inflation.
Record the price index, coverage and period if an external index is used.
Test whether the index is relevant to the underlying claim costs.
For a mixed portfolio, one index may be inadequate.
Check the consistency of inflation and discount-rate currency.
A scenario is not a probability distribution without weights or a stochastic model.
The calculation does not establish an accounting discount policy.
**Interview:** Why can slower payment increase nominal ultimate without more claims?
**Answer:** Costs may inflate over the longer settlement period.
The effect is conditional on the assumed escalation and cash-flow mechanism.

### 44. Segment mix shift

An old portfolio has 80 low-severity claims at average cost 1.
It also has 20 high-severity claims at average cost 5.
Its overall average cost is (80 × 1 + 20 × 5) / 100 = 1.8.
A new portfolio has 50 claims in each segment.
At unchanged segment averages its overall average is (50 × 1 + 50 × 5) / 100 = 3.
The overall increase is 3 − 1.8 = 1.2 per claim.
Within-segment average cost has not changed in the example.
The aggregate increase comes entirely from mix.
An aggregate triangle may similarly change when segment weights change.
Compare age-to-age factors within stable, meaningful segments.
Reweight old experience to the new mix when the assumptions support it.
Check that the segment definitions did not change during the period.
The example assumes the same maturity and claim-count convention.
Different reporting delays can otherwise distort the comparison.
Some segments may be too small for standalone factor estimation.
Pooling or credibility methods require explicit assumptions.
Do not assume more segmentation always improves predictive performance.
Compare out-of-time results and practical stability.
Reconcile segment estimates to the total portfolio.
Investigate missing segment assignments before interpreting mix movement.
**Interview:** Does an aggregate average increase prove severity inflation within segments?
**Answer:** No; a larger high-severity share can produce the increase alone.
This example does not measure an actual insurer's mix.

### 45. Handling a zero denominator

A matched age-one cohort has cumulative paid zero.
Its next-age cumulative paid is 20.
The individual ratio 20 / 0 is undefined.
Replacing the denominator with one produces 20, an arbitrary number.
Replacing the ratio with one asserts no development without evidence.
Suppose another matched cohort has 100 at age one and 150 at age two.
The aggregate ratio of sums is (20 + 150) / (0 + 100) = 1.7.
This aggregate calculation exists because its denominator is nonzero.
It is not a convex weighted average of both individual ratios.
The zero-start cohort can contribute to the numerator without an individual finite ratio.
Investigate whether zero means no payment, missing data or a coding issue.
If zero is genuine, multiplicative projection from zero produces zero.
That is uninformative if future payments are plausible.
Exposure-based methods or BF may be useful alternatives.
They require a defensible prior and development proportion.
An additive model may also be considered with suitable assumptions.
The selected alternative must handle cohort scale and heterogeneity.
Do not apply a silent software fallback.
Raise an explicit exception and preserve the selected treatment.
Backtest the alternative against comparable immature cohorts.
**Interview:** Can an aggregate ratio exist when an individual ratio does not?
**Answer:** Yes, if the aggregate denominator is nonzero; the usual weighted-average interpretation fails.
Recheck the observation mask before making the calculation.

### 46. Negative cumulative paid

An observed cohort has cumulative paid of −10 after net recoveries.
At the next age cumulative paid is 20.
The numerical ratio is 20 / −10 = −2.
Multiplying another cohort's positive paid by −2 would produce a negative projected ultimate.
That may be economically meaningless for the intended scope.
The sign reversal makes the standard multiplicative interpretation unstable.
The issue is not solved by replacing −10 with its absolute value.
Nor should a legitimate recovery be silently deleted.
Trace original payments, recovery events and the netting convention.
Consider modeling gross paid and recoveries separately.
The separate components must be recombined on the same valuation basis.
Each component may have different timing and uncertainty.
Some recovery amounts can exceed previous payments under particular accounting scopes.
Confirm the actual contract and ledger treatment before deciding.
An additive or claim-level model may be preferable for signed values.
Any alternative requires its own validation.
Record the claim-level transactions underlying the negative cell.
Examine whether a currency translation or reversal timing contributes.
A factor below one alone does not establish a data error.
The selected method must support the sign and scale of the observed data.
**Interview:** Is −2 a usable standard development factor merely because it is computable?
**Answer:** No; the multiplicative interpretation and intended liability scope must make sense.
Escalate if no defensible basis for projection is available.

### 47. Reserve floor and recoveries

A cohort has cumulative paid of 200.
A selected net ultimate is 190.
The arithmetic net outstanding is 190 − 200 = −10.
If net future recoveries of 10 are expected with no future payments, −10 can be coherent.
If the basis excludes recoveries, the negative estimate requires another explanation.
Applying a zero floor changes the estimate from −10 to 0.
That is a 10 upward adjustment.
Across ten similar cohorts, the mechanical total adjustment would be 100.
A floor can therefore materially change aggregate results.
Contractual, accounting or governance requirements may prescribe a treatment.
This example does not determine those requirements.
The floor should be a separately disclosed adjustment, not hidden in the calculation.
Investigate whether the negative result arises from valid signed transactions.
Check whether a factor was estimated using a mismatched basis.
Review whether cash outflows and recoveries are presented separately.
If a negative reserve is operationally prohibited, route the issue to the financial owner.
Avoid calling every negative projection impossible.
Also avoid assuming every negative projection is a valid recovery.
Retain both the raw model output and approved adjustment.
Monitor the count and amount of floored cohorts over time.
**Interview:** Why can automatic flooring bias an aggregate?
**Answer:** It removes negative contributions while retaining positive ones.
The magnitude and appropriateness depend on the liability and reporting basis.

### 48. Release gates and incident response

A model run produces a reserve of 148 in a test triangle.
The arithmetic reconciles, but a production source extract fails its independent control.
The correct release state is blocked or explicitly escalated under the governed policy.
The numerical result must not be presented as a validated production estimate.
First identify the failing source, scope, amount and cutoff.
Preserve the failing artifact and the original control totals.
Do not overwrite the evidence during remediation.
Check whether the failure affects gross, net, expense or only a specific segment.
Determine whether a prior approved run remains available.
Using an older run is a governed decision, not an automatic fallback.
Its valuation date and limitations must be visible.
Correct the source or mapping under change control.
Rebuild the affected triangle from the corrected immutable inputs.
Reconcile cells, control totals and the reserve movement.
Obtain independent review of the incident and the correction.
Record the owner, approval, release time and any remaining limitations.
If an estimate was already communicated, assess whether correction notification is needed.
Access to claim-level details should follow least-privilege policy.
Keep sensitive identifiers out of broad management reports.
Monitor recurrence and assign a preventive action owner.
Escalate unresolved material exceptions to the responsible actuary and financial owner.
**Interview:** Does passing the arithmetic check justify releasing the run?
**Answer:** No; a correct formula can operate on incomplete or incorrect inputs.
No actual incident or production release is claimed by this scenario.

### 49. Professional standard and scope

**SOURCED:** The Actuarial Standards Board publishes [ASOP No. 43, Property/Casualty Unpaid Claim Estimates](http://www.actuarialstandardsboard.org/asops/propertycasualty-unpaid-claim-estimates/).
Its scope concerns actuarial services involving specified property/casualty unpaid claim estimates.
The practitioner must consult the applicable version and full text for the assignment.
The standard is not a universal rule for every country or every insurance liability.
Identify the principal, intended users and intended purpose of the estimate.
Document the type of unpaid claim estimate and its financial scope.
Identify relevant data and material limitations.
Consider assumptions and methods in the context of the assignment.
Explain material uncertainty and significant judgments in the communication.
The detailed requirements and any permitted departures are governed by the standard itself.
This teaching note is not a substitute for reading the source.
It does not declare any individual valuation compliant.
Consult local law, professional requirements and accounting guidance as applicable.
The [ASOP No. 20 source](http://www.actuarialstandardsboard.org/asops/discounting-of-property-casualty-claim-estimates/) addresses discounting within its scope.
Its applicability and current version should be checked for the specific work.
Neither source validates the numerical assumptions in the illustrative cases.
Neither source authorizes an analyst to book the example reserve.
The assigned actuary retains professional responsibility for the work.
Record the edition, access date and any source changes in the workpaper.
Separate compliance requirements from internal management preferences.
Clarify whether the requested output is an estimate, range, opinion or scenario.
**Interview:** Is citing ASOP 43 enough to make a reserve compliant?
**Answer:** No; the standard must be applied to the actual assignment and documented work.
The sheet's own independent professional review remains open.

### 50. Integrated reserving interview

You receive a paid triangle whose factors are 1.5 and 1.2.
Its paid-based outstanding is 148 under a tail factor of one.
The claims system was migrated in the latest calendar year.
The latest paid diagonal is 20 below the old-system control.
The case-reserving team also reports a new strengthening policy.
Management asks whether the 148 can be booked immediately.
First reproduce the matched factor and outstanding calculations in Section 5.
The arithmetic is 0 + 36 + 112 = 148.
Next reconcile the 20 discrepancy at transaction and segment level.
If the 20 is missing paid, determine its cohort and age before recalculating.
Do not add 20 directly to the reserve: the impact depends on its location and factors.
Review migration records for delayed postings, duplicates and changed payment codes.
Preserve the as-known and restated views.
Separately investigate whether case strengthening changes incurred development.
The case change does not automatically invalidate the paid triangle.
The system migration can nevertheless change payment timing and paid development.
Compare paid, incurred and prior-based methods only on consistent bases.
Assess tail sensitivity rather than treating one as proven.
Run an as-of backtest around the migration if adequate historical data exist.
The original 148 has no stated predictive interval.
Summarize the data discrepancy, assumption risk and alternatives.
Escalate the release to the responsible actuary and financial owner.
**Interview:** What is the tempting wrong answer?
**Answer:** “The arithmetic is right, so book 148”; it ignores unresolved data and structural-change risk.
The scenario supplies no evidence that any actual system migrated or that 148 is bookable.

**Takeaway:** Distinct data, process, model and uncertainty failures require different evidence; a coherent audit trail connects the selected estimate to each decision.

# 12. Quick Reference Card

| Field | Reference |
|---|---|
| **Topic** | Paid claims development and actuarial reserving on a reconciled financial basis. |
| **Core formula** | Matched age factor = sum later cumulative / sum earlier cumulative; ultimate = latest paid × remaining selected factors × tail; paid outstanding = ultimate − latest paid. |
| **Key assumption** | Matched nonzero denominators, consistent cohort and financial definitions, and defensible development and tail selections. |
| **Complexity** | Expected O(N + A×D) time with constant-time keyed aggregation and O(A×D) dense storage for N transactions, A cohorts and D ages. |
| **Use when** | A reconciled paid triangle and a sufficiently credible development pattern support the intended estimate. |
| **Avoid when** | Structural change, sparse pairs, a mismatched basis or an unsupported tail make historical development unsuitable without adjustment. |
| **Go deeper** | Friedland and Mack references are linked above; the dedicated reserving validation sheet remains planned. |
| **Worked result** | Factors 1.5 and 1.2 with tail 1 give ultimates 180, 216 and 252; paid outstanding totals 0 + 36 + 112 = 148 currency units. |
| **Boundary** | Paid outstanding includes case reserves; broad incurred-based IBNR is ultimate minus paid-plus-case. The point estimate is not a confidence interval or booking recommendation. |
| **Action** | Reconcile inputs, compare alternatives, document uncertainty and obtain qualified actuarial review. |
