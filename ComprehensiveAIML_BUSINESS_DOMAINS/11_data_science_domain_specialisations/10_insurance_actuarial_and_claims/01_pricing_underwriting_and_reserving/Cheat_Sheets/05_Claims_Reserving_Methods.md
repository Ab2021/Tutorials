---
collection: "pricing_underwriting_and_reserving"
file_id: "05"
topic: "Claims reserving methods"
level: 2
status: "draft"
version: "1.1"
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
> **Version / reviewed:** 1.1 / not independently reviewed.
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

## Applied diagnostic and numerical practice

The following 80 scenarios are ILLUSTRATIVE exercises, not observed insurer results.
Each example uses currency units on one consistent, undiscounted paid basis unless explicitly noted.
The arithmetic illustrates a conditional projection; it does not validate the scenario assumptions.
The diagnostic interpretation is an educational application of the Friedland reference cited in Section 1.

### Data lineage and valuation cutoff

#### Practice 01: Late-posted payment
- **Situation:** An analyst evaluates late-posted payment while projecting a cohort with 102 paid to date.
- **Diagnostic:** Separate occurrence, payment and posting dates before deciding whether the transaction belongs in the as-of extract.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 102 × 1.15 = 117.30.
- **Outstanding calculation:** 117.30 − 102 = 15.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 122.40, or 5.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Separate occurrence, payment and posting dates before deciding whether the transaction belongs in the as-of extract. The calculated 15.30 is not a confidence interval.

#### Practice 02: Reversed payment
- **Situation:** An analyst evaluates reversed payment while projecting a cohort with 104 paid to date.
- **Diagnostic:** Link a reversal to its original transaction and preserve both signed movements.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 104 × 1.20 = 124.80.
- **Outstanding calculation:** 124.80 − 104 = 20.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 130.00, or 5.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Link a reversal to its original transaction and preserve both signed movements. The calculated 20.80 is not a confidence interval.

#### Practice 03: Duplicate feed
- **Situation:** An analyst evaluates duplicate feed while projecting a cohort with 106 paid to date.
- **Diagnostic:** Compare transaction identifiers and source-system totals before removing a suspected duplicate.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 106 × 1.25 = 132.50.
- **Outstanding calculation:** 132.50 − 106 = 26.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 137.80, or 5.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare transaction identifiers and source-system totals before removing a suspected duplicate. The calculated 26.50 is not a confidence interval.

#### Practice 04: Missing claim identifier
- **Situation:** An analyst evaluates missing claim identifier while projecting a cohort with 108 paid to date.
- **Diagnostic:** Quarantine unmatched records and reconcile their value before assigning them to cohorts.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 108 × 1.30 = 140.40.
- **Outstanding calculation:** 140.40 − 108 = 32.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 145.80, or 5.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Quarantine unmatched records and reconcile their value before assigning them to cohorts. The calculated 32.40 is not a confidence interval.

#### Practice 05: Changed accident date
- **Situation:** An analyst evaluates changed accident date while projecting a cohort with 110 paid to date.
- **Diagnostic:** Retain effective-dated cohort assignments and explain transfers between accident years.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 110 × 1.10 = 121.00.
- **Outstanding calculation:** 121.00 − 110 = 11.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 126.50, or 5.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Retain effective-dated cohort assignments and explain transfers between accident years. The calculated 11.00 is not a confidence interval.

#### Practice 06: Snapshot taken after cutoff
- **Situation:** An analyst evaluates snapshot taken after cutoff while projecting a cohort with 112 paid to date.
- **Diagnostic:** Use the historical snapshot available at valuation rather than a later restatement.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 112 × 1.15 = 128.80.
- **Outstanding calculation:** 128.80 − 112 = 16.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 134.40, or 5.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Use the historical snapshot available at valuation rather than a later restatement. The calculated 16.80 is not a confidence interval.

#### Practice 07: Currency conversion
- **Situation:** An analyst evaluates currency conversion while projecting a cohort with 114 paid to date.
- **Diagnostic:** Use a documented conversion convention consistently for historical and projected amounts.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 114 × 1.20 = 136.80.
- **Outstanding calculation:** 136.80 − 114 = 22.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 142.50, or 5.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Use a documented conversion convention consistently for historical and projected amounts. The calculated 22.80 is not a confidence interval.

#### Practice 08: Mixed expense scope
- **Situation:** An analyst evaluates mixed expense scope while projecting a cohort with 116 paid to date.
- **Diagnostic:** Separate allocated expense from indemnity if the ultimate and paid extracts do not share that scope.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 116 × 1.25 = 145.00.
- **Outstanding calculation:** 145.00 − 116 = 29.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 150.80, or 5.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Separate allocated expense from indemnity if the ultimate and paid extracts do not share that scope. The calculated 29.00 is not a confidence interval.

### Triangle construction and pairing

#### Practice 09: Missing future cell
- **Situation:** An analyst evaluates missing future cell while projecting a cohort with 118 paid to date.
- **Diagnostic:** Exclude the unobserved pair rather than treating the later cumulative amount as zero.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 118 × 1.30 = 153.40.
- **Outstanding calculation:** 153.40 − 118 = 35.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 159.30, or 5.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Exclude the unobserved pair rather than treating the later cumulative amount as zero. The calculated 35.40 is not a confidence interval.

#### Practice 10: Observed zero denominator
- **Situation:** An analyst evaluates observed zero denominator while projecting a cohort with 120 paid to date.
- **Diagnostic:** A zero prior cell cannot support an individual development ratio.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 120 × 1.10 = 132.00.
- **Outstanding calculation:** 132.00 − 120 = 12.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 138.00, or 6.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** A zero prior cell cannot support an individual development ratio. The calculated 12.00 is not a confidence interval.

#### Practice 11: Unequal development periods
- **Situation:** An analyst evaluates unequal development periods while projecting a cohort with 122 paid to date.
- **Diagnostic:** Align elapsed-time buckets before interpreting adjacent development factors.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 122 × 1.15 = 140.30.
- **Outstanding calculation:** 140.30 − 122 = 18.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 146.40, or 6.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Align elapsed-time buckets before interpreting adjacent development factors. The calculated 18.30 is not a confidence interval.

#### Practice 12: Partial calendar period
- **Situation:** An analyst evaluates partial calendar period while projecting a cohort with 124 paid to date.
- **Diagnostic:** Do not compare a partial age interval directly with a complete historical interval.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 124 × 1.20 = 148.80.
- **Outstanding calculation:** 148.80 − 124 = 24.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 155.00, or 6.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Do not compare a partial age interval directly with a complete historical interval. The calculated 24.80 is not a confidence interval.

#### Practice 13: Cumulative versus incremental
- **Situation:** An analyst evaluates cumulative versus incremental while projecting a cohort with 126 paid to date.
- **Diagnostic:** Reconstruct cumulative amounts by summing each included increment once.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 126 × 1.25 = 157.50.
- **Outstanding calculation:** 157.50 − 126 = 31.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 163.80, or 6.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Reconstruct cumulative amounts by summing each included increment once. The calculated 31.50 is not a confidence interval.

#### Practice 14: Cohort reclassification
- **Situation:** An analyst evaluates cohort reclassification while projecting a cohort with 128 paid to date.
- **Diagnostic:** Rebuild comparable histories if a segmentation change moves claims across rows.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 128 × 1.30 = 166.40.
- **Outstanding calculation:** 166.40 − 128 = 38.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 172.80, or 6.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Rebuild comparable histories if a segmentation change moves claims across rows. The calculated 38.40 is not a confidence interval.

#### Practice 15: Negative movement
- **Situation:** An analyst evaluates negative movement while projecting a cohort with 130 paid to date.
- **Diagnostic:** Trace valid recoveries and corrections rather than forcing cumulative values to increase.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 130 × 1.10 = 143.00.
- **Outstanding calculation:** 143.00 − 130 = 13.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 149.50, or 6.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Trace valid recoveries and corrections rather than forcing cumulative values to increase. The calculated 13.00 is not a confidence interval.

#### Practice 16: Inconsistent matched populations
- **Situation:** An analyst evaluates inconsistent matched populations while projecting a cohort with 132 paid to date.
- **Diagnostic:** Apply exactly the same cohort eligibility rule to numerator and denominator.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 132 × 1.15 = 151.80.
- **Outstanding calculation:** 151.80 − 132 = 19.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 158.40, or 6.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Apply exactly the same cohort eligibility rule to numerator and denominator. The calculated 19.80 is not a confidence interval.

### Development factor selection

#### Practice 17: Volume weighting
- **Situation:** An analyst evaluates volume weighting while projecting a cohort with 134 paid to date.
- **Diagnostic:** Explain how large starting amounts receive more weight in the ratio of sums.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 134 × 1.20 = 160.80.
- **Outstanding calculation:** 160.80 − 134 = 26.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 167.50, or 6.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Explain how large starting amounts receive more weight in the ratio of sums. The calculated 26.80 is not a confidence interval.

#### Practice 18: Simple averaging
- **Situation:** An analyst evaluates simple averaging while projecting a cohort with 136 paid to date.
- **Diagnostic:** Compare a simple average with volume weighting when individual ratios are meaningful.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 136 × 1.25 = 170.00.
- **Outstanding calculation:** 170.00 − 136 = 34.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 176.80, or 6.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare a simple average with volume weighting when individual ratios are meaningful. The calculated 34.00 is not a confidence interval.

#### Practice 19: Single observed pair
- **Situation:** An analyst evaluates single observed pair while projecting a cohort with 138 paid to date.
- **Diagnostic:** Treat an age factor based on one matched pair as particularly fragile.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 138 × 1.30 = 179.40.
- **Outstanding calculation:** 179.40 − 138 = 41.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 186.30, or 6.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Treat an age factor based on one matched pair as particularly fragile. The calculated 41.40 is not a confidence interval.

#### Practice 20: Large individual claim
- **Situation:** An analyst evaluates large individual claim while projecting a cohort with 140 paid to date.
- **Diagnostic:** Inspect a large claim separately before extrapolating its effect to other cohorts.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 140 × 1.10 = 154.00.
- **Outstanding calculation:** 154.00 − 140 = 14.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 161.00, or 7.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Inspect a large claim separately before extrapolating its effect to other cohorts. The calculated 14.00 is not a confidence interval.

#### Practice 21: Calendar shock
- **Situation:** An analyst evaluates calendar shock while projecting a cohort with 142 paid to date.
- **Diagnostic:** Investigate whether a diagonal shift affects several development ages at once.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 142 × 1.15 = 163.30.
- **Outstanding calculation:** 163.30 − 142 = 21.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 170.40, or 7.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Investigate whether a diagonal shift affects several development ages at once. The calculated 21.30 is not a confidence interval.

#### Practice 22: Changing settlement speed
- **Situation:** An analyst evaluates changing settlement speed while projecting a cohort with 144 paid to date.
- **Diagnostic:** Compare payment timing and claim closures before attributing a factor change to severity.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 144 × 1.20 = 172.80.
- **Outstanding calculation:** 172.80 − 144 = 28.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 180.00, or 7.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare payment timing and claim closures before attributing a factor change to severity. The calculated 28.80 is not a confidence interval.

#### Practice 23: Selection window
- **Situation:** An analyst evaluates selection window while projecting a cohort with 146 paid to date.
- **Diagnostic:** Compare plausible historical windows and record exclusions and their reasons.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 146 × 1.25 = 182.50.
- **Outstanding calculation:** 182.50 − 146 = 36.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 189.80, or 7.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare plausible historical windows and record exclusions and their reasons. The calculated 36.50 is not a confidence interval.

#### Practice 24: Factor below one
- **Situation:** An analyst evaluates factor below one while projecting a cohort with 148 paid to date.
- **Diagnostic:** Check recoveries, reversals and data basis before treating a declining cumulative series as invalid.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 148 × 1.30 = 192.40.
- **Outstanding calculation:** 192.40 − 148 = 44.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 199.80, or 7.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Check recoveries, reversals and data basis before treating a declining cumulative series as invalid. The calculated 44.40 is not a confidence interval.

### Tail and maturity

#### Practice 25: Tail of one
- **Situation:** An analyst evaluates tail of one while projecting a cohort with 150 paid to date.
- **Diagnostic:** Make explicit that a unit tail assumes no further development after the final modeled age.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 150 × 1.10 = 165.00.
- **Outstanding calculation:** 165.00 − 150 = 15.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 172.50, or 7.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Make explicit that a unit tail assumes no further development after the final modeled age. The calculated 15.00 is not a confidence interval.

#### Practice 26: Long reporting lag
- **Situation:** An analyst evaluates long reporting lag while projecting a cohort with 152 paid to date.
- **Diagnostic:** Assess whether late-reported claims are represented in the available history.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 152 × 1.15 = 174.80.
- **Outstanding calculation:** 174.80 − 152 = 22.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 182.40, or 7.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Assess whether late-reported claims are represented in the available history. The calculated 22.80 is not a confidence interval.

#### Practice 27: Late settlement
- **Situation:** An analyst evaluates late settlement while projecting a cohort with 154 paid to date.
- **Diagnostic:** Consider whether mature open claims can still produce payments beyond the triangle horizon.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 154 × 1.20 = 184.80.
- **Outstanding calculation:** 184.80 − 154 = 30.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 192.50, or 7.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Consider whether mature open claims can still produce payments beyond the triangle horizon. The calculated 30.80 is not a confidence interval.

#### Practice 28: Sparse older cohorts
- **Situation:** An analyst evaluates sparse older cohorts while projecting a cohort with 156 paid to date.
- **Diagnostic:** Use external evidence cautiously when internal late-age observations are thin.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 156 × 1.25 = 195.00.
- **Outstanding calculation:** 195.00 − 156 = 39.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 202.80, or 7.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Use external evidence cautiously when internal late-age observations are thin. The calculated 39.00 is not a confidence interval.

#### Practice 29: Coverage change
- **Situation:** An analyst evaluates coverage change while projecting a cohort with 158 paid to date.
- **Diagnostic:** Avoid applying a historical tail uncritically after changes in coverage or claim limits.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 158 × 1.30 = 205.40.
- **Outstanding calculation:** 205.40 − 158 = 47.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 213.30, or 7.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Avoid applying a historical tail uncritically after changes in coverage or claim limits. The calculated 47.40 is not a confidence interval.

#### Practice 30: Inflation in the tail
- **Situation:** An analyst evaluates inflation in the tail while projecting a cohort with 160 paid to date.
- **Diagnostic:** Consider whether late settlement amounts are exposed to a different inflation environment.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 160 × 1.10 = 176.00.
- **Outstanding calculation:** 176.00 − 160 = 16.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 184.00, or 8.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Consider whether late settlement amounts are exposed to a different inflation environment. The calculated 16.00 is not a confidence interval.

#### Practice 31: Tail selection review
- **Situation:** An analyst evaluates tail selection review while projecting a cohort with 162 paid to date.
- **Diagnostic:** Record the rationale, evidence period and range of selected tail assumptions.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 162 × 1.15 = 186.30.
- **Outstanding calculation:** 186.30 − 162 = 24.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 194.40, or 8.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Record the rationale, evidence period and range of selected tail assumptions. The calculated 24.30 is not a confidence interval.

#### Practice 32: Tail and discounting
- **Situation:** An analyst evaluates tail and discounting while projecting a cohort with 164 paid to date.
- **Diagnostic:** Distinguish undiscounted tail development from the timing assumptions required for discounting.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 164 × 1.20 = 196.80.
- **Outstanding calculation:** 196.80 − 164 = 32.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 205.00, or 8.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Distinguish undiscounted tail development from the timing assumptions required for discounting. The calculated 32.80 is not a confidence interval.

### Prior-based alternatives

#### Practice 33: Prior expected ultimate
- **Situation:** An analyst evaluates prior expected ultimate while projecting a cohort with 166 paid to date.
- **Diagnostic:** Derive the prior from an exposure or premium base consistent with the covered claims.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 166 × 1.25 = 207.50.
- **Outstanding calculation:** 207.50 − 166 = 41.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 215.80, or 8.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Derive the prior from an exposure or premium base consistent with the covered claims. The calculated 41.50 is not a confidence interval.

#### Practice 34: Paid proportion
- **Situation:** An analyst evaluates paid proportion while projecting a cohort with 168 paid to date.
- **Diagnostic:** Use the paid proportion for the same development age and selected tail as the prior.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 168 × 1.30 = 218.40.
- **Outstanding calculation:** 218.40 − 168 = 50.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 226.80, or 8.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Use the paid proportion for the same development age and selected tail as the prior. The calculated 50.40 is not a confidence interval.

#### Practice 35: Immature cohort
- **Situation:** An analyst evaluates immature cohort while projecting a cohort with 170 paid to date.
- **Diagnostic:** Compare a prior-based method with direct paid development when little has been paid.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 170 × 1.10 = 187.00.
- **Outstanding calculation:** 187.00 − 170 = 17.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 195.50, or 8.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare a prior-based method with direct paid development when little has been paid. The calculated 17.00 is not a confidence interval.

#### Practice 36: Stale prior
- **Situation:** An analyst evaluates stale prior while projecting a cohort with 172 paid to date.
- **Diagnostic:** Refresh exposure mix and price-level assumptions before using a historical expected loss ratio.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 172 × 1.15 = 197.80.
- **Outstanding calculation:** 197.80 − 172 = 25.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 206.40, or 8.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Refresh exposure mix and price-level assumptions before using a historical expected loss ratio. The calculated 25.80 is not a confidence interval.

#### Practice 37: Prior contamination
- **Situation:** An analyst evaluates prior contamination while projecting a cohort with 174 paid to date.
- **Diagnostic:** Avoid calling a prior independent if it already incorporates the same observed claims without disclosure.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 174 × 1.20 = 208.80.
- **Outstanding calculation:** 208.80 − 174 = 34.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 217.50, or 8.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Avoid calling a prior independent if it already incorporates the same observed claims without disclosure. The calculated 34.80 is not a confidence interval.

#### Practice 38: Exposure growth
- **Situation:** An analyst evaluates exposure growth while projecting a cohort with 176 paid to date.
- **Diagnostic:** Check whether growth reflects homogeneous risk units rather than just a change in insured values.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 176 × 1.25 = 220.00.
- **Outstanding calculation:** 220.00 − 176 = 44.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 228.80, or 8.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Check whether growth reflects homogeneous risk units rather than just a change in insured values. The calculated 44.00 is not a confidence interval.

#### Practice 39: Incurred alternative
- **Situation:** An analyst evaluates incurred alternative while projecting a cohort with 178 paid to date.
- **Diagnostic:** Compare paid and incurred methods only after reconciling case-reserve definitions.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 178 × 1.30 = 231.40.
- **Outstanding calculation:** 231.40 − 178 = 53.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 240.30, or 8.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare paid and incurred methods only after reconciling case-reserve definitions. The calculated 53.40 is not a confidence interval.

#### Practice 40: Method weighting
- **Situation:** An analyst evaluates method weighting while projecting a cohort with 180 paid to date.
- **Diagnostic:** Document why a blend is selected; do not treat an arbitrary blend as statistical credibility.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 180 × 1.10 = 198.00.
- **Outstanding calculation:** 198.00 − 180 = 18.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 207.00, or 9.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Document why a blend is selected; do not treat an arbitrary blend as statistical credibility. The calculated 18.00 is not a confidence interval.

### Case-reserve diagnostics

#### Practice 41: Case strengthening
- **Situation:** An analyst evaluates case strengthening while projecting a cohort with 182 paid to date.
- **Diagnostic:** An increase in case adequacy can change incurred development without an equal change in ultimate loss.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 182 × 1.15 = 209.30.
- **Outstanding calculation:** 209.30 − 182 = 27.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 218.40, or 9.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** An increase in case adequacy can change incurred development without an equal change in ultimate loss. The calculated 27.30 is not a confidence interval.

#### Practice 42: Case weakening
- **Situation:** An analyst evaluates case weakening while projecting a cohort with 184 paid to date.
- **Diagnostic:** Lower initial case estimates can create apparent subsequent adverse incurred development.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 184 × 1.20 = 220.80.
- **Outstanding calculation:** 220.80 − 184 = 36.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 230.00, or 9.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Lower initial case estimates can create apparent subsequent adverse incurred development. The calculated 36.80 is not a confidence interval.

#### Practice 43: Closure initiative
- **Situation:** An analyst evaluates closure initiative while projecting a cohort with 186 paid to date.
- **Diagnostic:** Faster closures can alter paid and case patterns independently of underlying claim frequency.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 186 × 1.25 = 232.50.
- **Outstanding calculation:** 232.50 − 186 = 46.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 241.80, or 9.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Faster closures can alter paid and case patterns independently of underlying claim frequency. The calculated 46.50 is not a confidence interval.

#### Practice 44: Reopened claims
- **Situation:** An analyst evaluates reopened claims while projecting a cohort with 188 paid to date.
- **Diagnostic:** Track reopened status consistently to avoid understating open inventory and future work.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 188 × 1.30 = 244.40.
- **Outstanding calculation:** 244.40 − 188 = 56.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 253.80, or 9.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Track reopened status consistently to avoid understating open inventory and future work. The calculated 56.40 is not a confidence interval.

#### Practice 45: Large claim protocol
- **Situation:** An analyst evaluates large claim protocol while projecting a cohort with 190 paid to date.
- **Diagnostic:** Review whether large claims are reserved using a distinct operational process.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 190 × 1.10 = 209.00.
- **Outstanding calculation:** 209.00 − 190 = 19.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 218.50, or 9.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Review whether large claims are reserved using a distinct operational process. The calculated 19.00 is not a confidence interval.

#### Practice 46: Adjuster turnover
- **Situation:** An analyst evaluates adjuster turnover while projecting a cohort with 192 paid to date.
- **Diagnostic:** Check whether staffing changes coincide with altered case estimates or closure rates.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 192 × 1.15 = 220.80.
- **Outstanding calculation:** 220.80 − 192 = 28.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 230.40, or 9.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Check whether staffing changes coincide with altered case estimates or closure rates. The calculated 28.80 is not a confidence interval.

#### Practice 47: Reported claim mix
- **Situation:** An analyst evaluates reported claim mix while projecting a cohort with 194 paid to date.
- **Diagnostic:** Changes in the mix of reported claim types can move average case amounts without a practice change.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 194 × 1.20 = 232.80.
- **Outstanding calculation:** 232.80 − 194 = 38.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 242.50, or 9.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Changes in the mix of reported claim types can move average case amounts without a practice change. The calculated 38.80 is not a confidence interval.

#### Practice 48: Case adequacy metric
- **Situation:** An analyst evaluates case adequacy metric while projecting a cohort with 196 paid to date.
- **Diagnostic:** Compare case estimates against later development by comparable claim maturity and status.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 196 × 1.25 = 245.00.
- **Outstanding calculation:** 245.00 − 196 = 49.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 254.80, or 9.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare case estimates against later development by comparable claim maturity and status. The calculated 49.00 is not a confidence interval.

### Uncertainty and sensitivity

#### Practice 49: Parameter sensitivity
- **Situation:** An analyst evaluates parameter sensitivity while projecting a cohort with 198 paid to date.
- **Diagnostic:** Change a selected factor while holding the remaining assumptions fixed to isolate its effect.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 198 × 1.30 = 257.40.
- **Outstanding calculation:** 257.40 − 198 = 59.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 267.30, or 9.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Change a selected factor while holding the remaining assumptions fixed to isolate its effect. The calculated 59.40 is not a confidence interval.

#### Practice 50: Tail sensitivity
- **Situation:** An analyst evaluates tail sensitivity while projecting a cohort with 200 paid to date.
- **Diagnostic:** Vary the tail separately from early development to avoid conflating sources of change.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 200 × 1.10 = 220.00.
- **Outstanding calculation:** 220.00 − 200 = 20.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 230.00, or 10.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Vary the tail separately from early development to avoid conflating sources of change. The calculated 20.00 is not a confidence interval.

#### Practice 51: Model uncertainty
- **Situation:** An analyst evaluates model uncertainty while projecting a cohort with 202 paid to date.
- **Diagnostic:** Compare methods that encode different assumptions rather than treating one model as complete.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 202 × 1.15 = 232.30.
- **Outstanding calculation:** 232.30 − 202 = 30.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 242.40, or 10.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare methods that encode different assumptions rather than treating one model as complete. The calculated 30.30 is not a confidence interval.

#### Practice 52: Process uncertainty
- **Situation:** An analyst evaluates process uncertainty while projecting a cohort with 204 paid to date.
- **Diagnostic:** Recognize that actual future payments fluctuate even if selected assumptions are appropriate.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 204 × 1.20 = 244.80.
- **Outstanding calculation:** 244.80 − 204 = 40.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 255.00, or 10.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Recognize that actual future payments fluctuate even if selected assumptions are appropriate. The calculated 40.80 is not a confidence interval.

#### Practice 53: Parameter uncertainty
- **Situation:** An analyst evaluates parameter uncertainty while projecting a cohort with 206 paid to date.
- **Diagnostic:** Estimated factors are uncertain because the historical sample is finite.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 206 × 1.25 = 257.50.
- **Outstanding calculation:** 257.50 − 206 = 51.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 267.80, or 10.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Estimated factors are uncertain because the historical sample is finite. The calculated 51.50 is not a confidence interval.

#### Practice 54: Scenario dependence
- **Situation:** An analyst evaluates scenario dependence while projecting a cohort with 208 paid to date.
- **Diagnostic:** Avoid adding separate scenario impacts if the assumptions move together economically.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 208 × 1.30 = 270.40.
- **Outstanding calculation:** 270.40 − 208 = 62.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 280.80, or 10.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Avoid adding separate scenario impacts if the assumptions move together economically. The calculated 62.40 is not a confidence interval.

#### Practice 55: Prediction interval
- **Situation:** An analyst evaluates prediction interval while projecting a cohort with 210 paid to date.
- **Diagnostic:** A deterministic range of scenarios is not automatically a calibrated prediction interval.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 210 × 1.10 = 231.00.
- **Outstanding calculation:** 231.00 − 210 = 21.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 241.50, or 10.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** A deterministic range of scenarios is not automatically a calibrated prediction interval. The calculated 21.00 is not a confidence interval.

#### Practice 56: Reserve risk communication
- **Situation:** An analyst evaluates reserve risk communication while projecting a cohort with 212 paid to date.
- **Diagnostic:** Report assumptions, potential direction of error and decision relevance alongside a point estimate.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 212 × 1.15 = 243.80.
- **Outstanding calculation:** 243.80 − 212 = 31.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 254.40, or 10.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Report assumptions, potential direction of error and decision relevance alongside a point estimate. The calculated 31.80 is not a confidence interval.

### Production controls

#### Practice 57: Source reconciliation
- **Situation:** An analyst evaluates source reconciliation while projecting a cohort with 214 paid to date.
- **Diagnostic:** Compare aggregated triangle movements to independently controlled claims-system totals.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 214 × 1.20 = 256.80.
- **Outstanding calculation:** 256.80 − 214 = 42.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 267.50, or 10.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare aggregated triangle movements to independently controlled claims-system totals. The calculated 42.80 is not a confidence interval.

#### Practice 58: Reproducible run
- **Situation:** An analyst evaluates reproducible run while projecting a cohort with 216 paid to date.
- **Diagnostic:** Freeze input versions, software version, valuation date and selected assumptions.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 216 × 1.25 = 270.00.
- **Outstanding calculation:** 270.00 − 216 = 54.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 280.80, or 10.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Freeze input versions, software version, valuation date and selected assumptions. The calculated 54.00 is not a confidence interval.

#### Practice 59: Access control
- **Situation:** An analyst evaluates access control while projecting a cohort with 218 paid to date.
- **Diagnostic:** Limit reserve changes to authorized roles and preserve an audit trail.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 218 × 1.30 = 283.40.
- **Outstanding calculation:** 283.40 − 218 = 65.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 294.30, or 10.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Limit reserve changes to authorized roles and preserve an audit trail. The calculated 65.40 is not a confidence interval.

#### Practice 60: Override governance
- **Situation:** An analyst evaluates override governance while projecting a cohort with 220 paid to date.
- **Diagnostic:** Require a documented reason and independent approval for material overrides.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 220 × 1.10 = 242.00.
- **Outstanding calculation:** 242.00 − 220 = 22.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 253.00, or 11.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Require a documented reason and independent approval for material overrides. The calculated 22.00 is not a confidence interval.

#### Practice 61: Exception handling
- **Situation:** An analyst evaluates exception handling while projecting a cohort with 222 paid to date.
- **Diagnostic:** Assign ownership and resolution deadlines for unresolved reconciliation differences.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 222 × 1.15 = 255.30.
- **Outstanding calculation:** 255.30 − 222 = 33.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 266.40, or 11.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Assign ownership and resolution deadlines for unresolved reconciliation differences. The calculated 33.30 is not a confidence interval.

#### Practice 62: Run comparison
- **Situation:** An analyst evaluates run comparison while projecting a cohort with 224 paid to date.
- **Diagnostic:** Bridge changes from prior valuation into data, assumption and methodological components.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 224 × 1.20 = 268.80.
- **Outstanding calculation:** 268.80 − 224 = 44.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 280.00, or 11.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Bridge changes from prior valuation into data, assumption and methodological components. The calculated 44.80 is not a confidence interval.

#### Practice 63: Reporting boundary
- **Situation:** An analyst evaluates reporting boundary while projecting a cohort with 226 paid to date.
- **Diagnostic:** Label analytical estimates separately from booked reserves and management decisions.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 226 × 1.25 = 282.50.
- **Outstanding calculation:** 282.50 − 226 = 56.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 293.80, or 11.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Label analytical estimates separately from booked reserves and management decisions. The calculated 56.50 is not a confidence interval.

#### Practice 64: Change control
- **Situation:** An analyst evaluates change control while projecting a cohort with 228 paid to date.
- **Diagnostic:** Review material changes in claim handling or source systems before applying historical factors.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 228 × 1.30 = 296.40.
- **Outstanding calculation:** 296.40 − 228 = 68.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 307.80, or 11.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Review material changes in claim handling or source systems before applying historical factors. The calculated 68.40 is not a confidence interval.

### Backtesting and monitoring

#### Practice 65: As-of backtest
- **Situation:** An analyst evaluates as-of backtest while projecting a cohort with 230 paid to date.
- **Diagnostic:** Use only information available at the historical valuation date.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 230 × 1.10 = 253.00.
- **Outstanding calculation:** 253.00 − 230 = 23.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 264.50, or 11.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Use only information available at the historical valuation date. The calculated 23.00 is not a confidence interval.

#### Practice 66: Maturity comparison
- **Situation:** An analyst evaluates maturity comparison while projecting a cohort with 232 paid to date.
- **Diagnostic:** Compare outcomes at matching development ages rather than mixing immature and mature cohorts.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 232 × 1.15 = 266.80.
- **Outstanding calculation:** 266.80 − 232 = 34.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 278.40, or 11.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Compare outcomes at matching development ages rather than mixing immature and mature cohorts. The calculated 34.80 is not a confidence interval.

#### Practice 67: Actual versus expected
- **Situation:** An analyst evaluates actual versus expected while projecting a cohort with 234 paid to date.
- **Diagnostic:** Bridge realized payments against the prior forecast using the same financial scope.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 234 × 1.20 = 280.80.
- **Outstanding calculation:** 280.80 − 234 = 46.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 292.50, or 11.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Bridge realized payments against the prior forecast using the same financial scope. The calculated 46.80 is not a confidence interval.

#### Practice 68: Revision tracking
- **Situation:** An analyst evaluates revision tracking while projecting a cohort with 236 paid to date.
- **Diagnostic:** Distinguish later source-data corrections from genuine forecast error.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 236 × 1.25 = 295.00.
- **Outstanding calculation:** 295.00 − 236 = 59.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 306.80, or 11.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Distinguish later source-data corrections from genuine forecast error. The calculated 59.00 is not a confidence interval.

#### Practice 69: Rolling valuations
- **Situation:** An analyst evaluates rolling valuations while projecting a cohort with 238 paid to date.
- **Diagnostic:** Repeat historical evaluations across multiple dates when retained snapshots permit.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 238 × 1.30 = 309.40.
- **Outstanding calculation:** 309.40 − 238 = 71.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 321.30, or 11.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Repeat historical evaluations across multiple dates when retained snapshots permit. The calculated 71.40 is not a confidence interval.

#### Practice 70: Segment drift
- **Situation:** An analyst evaluates segment drift while projecting a cohort with 240 paid to date.
- **Diagnostic:** Monitor mix shifts that make historical development less representative.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 240 × 1.10 = 264.00.
- **Outstanding calculation:** 264.00 − 240 = 24.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 276.00, or 12.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Monitor mix shifts that make historical development less representative. The calculated 24.00 is not a confidence interval.

#### Practice 71: Small-sample noise
- **Situation:** An analyst evaluates small-sample noise while projecting a cohort with 242 paid to date.
- **Diagnostic:** Do not infer a persistent trend from a single volatile observation.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 242 × 1.15 = 278.30.
- **Outstanding calculation:** 278.30 − 242 = 36.30 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 290.40, or 12.10 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Do not infer a persistent trend from a single volatile observation. The calculated 36.30 is not a confidence interval.

#### Practice 72: Threshold escalation
- **Situation:** An analyst evaluates threshold escalation while projecting a cohort with 244 paid to date.
- **Diagnostic:** Define who investigates a material variance and how the investigation is documented.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 244 × 1.20 = 292.80.
- **Outstanding calculation:** 292.80 − 244 = 48.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 305.00, or 12.20 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Define who investigates a material variance and how the investigation is documented. The calculated 48.80 is not a confidence interval.

### Communication and interview judgment

#### Practice 73: Outstanding versus ultimate
- **Situation:** An analyst evaluates outstanding versus ultimate while projecting a cohort with 246 paid to date.
- **Diagnostic:** Explain that ultimate includes paid-to-date, whereas outstanding excludes it.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 246 × 1.25 = 307.50.
- **Outstanding calculation:** 307.50 − 246 = 61.50 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 319.80, or 12.30 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Explain that ultimate includes paid-to-date, whereas outstanding excludes it. The calculated 61.50 is not a confidence interval.

#### Practice 74: Broad versus strict IBNR
- **Situation:** An analyst evaluates broad versus strict ibnr while projecting a cohort with 248 paid to date.
- **Diagnostic:** State whether development on reported claims is included in the IBNR definition.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 248 × 1.30 = 322.40.
- **Outstanding calculation:** 322.40 − 248 = 74.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 334.80, or 12.40 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** State whether development on reported claims is included in the IBNR definition. The calculated 74.40 is not a confidence interval.

#### Practice 75: Basis mismatch
- **Situation:** An analyst evaluates basis mismatch while projecting a cohort with 250 paid to date.
- **Diagnostic:** Explain why gross ultimate minus net paid does not produce a coherent net reserve.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 250 × 1.10 = 275.00.
- **Outstanding calculation:** 275.00 − 250 = 25.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 287.50, or 12.50 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Explain why gross ultimate minus net paid does not produce a coherent net reserve. The calculated 25.00 is not a confidence interval.

#### Practice 76: Double-counting case
- **Situation:** An analyst evaluates double-counting case while projecting a cohort with 252 paid to date.
- **Diagnostic:** Do not add case reserves to paid outstanding when both refer to the same future payments.
- **Baseline calculation:** With a selected remaining factor of 1.15, ultimate is 252 × 1.15 = 289.80.
- **Outstanding calculation:** 289.80 − 252 = 37.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.20 gives ultimate 302.40, or 12.60 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Do not add case reserves to paid outstanding when both refer to the same future payments. The calculated 37.80 is not a confidence interval.

#### Practice 77: Method disagreement
- **Situation:** An analyst evaluates method disagreement while projecting a cohort with 254 paid to date.
- **Diagnostic:** Investigate differences in assumptions and data before choosing or averaging estimates.
- **Baseline calculation:** With a selected remaining factor of 1.20, ultimate is 254 × 1.20 = 304.80.
- **Outstanding calculation:** 304.80 − 254 = 50.80 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.25 gives ultimate 317.50, or 12.70 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Investigate differences in assumptions and data before choosing or averaging estimates. The calculated 50.80 is not a confidence interval.

#### Practice 78: One-factor certainty
- **Situation:** An analyst evaluates one-factor certainty while projecting a cohort with 256 paid to date.
- **Diagnostic:** A precise arithmetic factor does not demonstrate stable future development.
- **Baseline calculation:** With a selected remaining factor of 1.25, ultimate is 256 × 1.25 = 320.00.
- **Outstanding calculation:** 320.00 − 256 = 64.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.30 gives ultimate 332.80, or 12.80 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** A precise arithmetic factor does not demonstrate stable future development. The calculated 64.00 is not a confidence interval.

#### Practice 79: Management decision
- **Situation:** An analyst evaluates management decision while projecting a cohort with 258 paid to date.
- **Diagnostic:** Separate the actuary’s estimate and uncertainty from the authorized booking decision.
- **Baseline calculation:** With a selected remaining factor of 1.30, ultimate is 258 × 1.30 = 335.40.
- **Outstanding calculation:** 335.40 − 258 = 77.40 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.35 gives ultimate 348.30, or 12.90 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Separate the actuary’s estimate and uncertainty from the authorized booking decision. The calculated 77.40 is not a confidence interval.

#### Practice 80: Scope limitation
- **Situation:** An analyst evaluates scope limitation while projecting a cohort with 260 paid to date.
- **Diagnostic:** Identify missing data and disclose which liability components the estimate excludes.
- **Baseline calculation:** With a selected remaining factor of 1.10, ultimate is 260 × 1.10 = 286.00.
- **Outstanding calculation:** 286.00 − 260 = 26.00 of future paid, conditional on that factor.
- **Alternative:** A factor of 1.15 gives ultimate 299.00, or 13.00 more than the baseline.
- **Decision boundary:** The factor change is a sensitivity, not evidence that either factor is calibrated.
- **Review evidence:** Preserve the relevant source records, cohort scope and the rationale for this particular diagnostic.
- **Interview answer:** Identify missing data and disclose which liability components the estimate excludes. The calculated 26.00 is not a confidence interval.

# 12. Quick Reference Card

| Field | Reference |
|---|---|
| **Trigger** | Estimate future paid claims from a reconciled, sufficiently credible paid development triangle. |
| **Inputs** | Latest cumulative paid by accident year, matched cumulative age pairs, selected tail, valuation date and consistent basis. |
| **Rule** | Matched volume-weighted factor = sum later cumulative / sum earlier cumulative; ultimate = latest paid × remaining selected factors × tail; outstanding = ultimate − latest paid. |
| **Conditions** | Matched nonzero denominators, coherent cohort and financial definitions, and defensible selected development and tail. |
| **Example** | 1.5 and 1.2 with tail 1 give ultimates 180, 216, 252 and paid outstanding 0, 36, 112; total 148 currency units. |
| **Failure / boundary** | Paid outstanding includes case reserves; incurred-based broad IBNR equals ultimate minus paid-plus-case. Structural change, sparse pairs and an unsupported tail can invalidate projection. |
| **Action / escalation** | Reconcile, compare alternatives, document uncertainty and seek actuarial review; this point estimate is not a confidence interval or a booking recommendation. |
