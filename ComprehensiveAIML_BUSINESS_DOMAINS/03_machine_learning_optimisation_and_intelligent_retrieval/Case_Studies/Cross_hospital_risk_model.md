---
case_id: "CASE-03-16-001"
source_plan_id: "PLAN-M16-01"
case_title: "Cross-hospital risk model: federated training under a fixed privacy budget"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "draft"
content_status: "draft"
source_status: "OPEN"
case_type: "hybrid"
domain: "Healthcare — hospital operations, readmission risk, multi-site data collaboration"
primary_pattern: "privacy-safe collaboration"
secondary_patterns: ["federated learning", "validation"]
evidence_level: "illustrative"
source_basis: "composite"
as_of_date: "2026-11-09"
owner_role: "consortium data governance board (decision); clinical analytics lead (delivery)"
source_index: "Train/ComprehensiveAIML_BUSINESS_DOMAINS/03_machine_learning_optimisation_and_intelligent_retrieval/Case_Studies/INDEX.md"
review_trigger: "Change in privacy budget, site count, label definition, program cost, or a regulator ruling on cross-border model training"
---

# Cross-hospital risk model: federated training under a fixed privacy budget

> **Evidence declaration.** This is a composite teaching case. The six-hospital consortium, its encounter volumes, prevalences, model scores, privacy parameters, prices, and program capacity are **constructed illustrative values**. They are internally consistent and the arithmetic is independently checked in the text, but they are not measurements of any named hospital, vendor, or consortium. Mechanisms attributed to published work (federated averaging, DP-SGD, zCDP accounting, secure aggregation, group privacy) are **sourced** to the papers named in Sections 5e and 14a; the numeric instantiations here are **derived** or **illustrative**. Where no source exists for a claim, the text says so. Section 12 marks every failure row as observed, published, simulated, or hypothetical, and in this file all rows are hypothetical by construction.

## 1. The One-Line Framing

> Given **one discharge-level encounter record per hospital admission, available in six hospitals' own warehouses and never pooled**, decide **whether the consortium should train one shared 30-day-readmission risk model, and under what privacy mechanism and budget**, for the **consortium data governance board acting on behalf of six care-management programs**, to improve **the identification and equitable treatment of patients who will be readmitted**, subject to **a hard prohibition on moving patient-level records across organizational boundaries and a declared differential-privacy budget of ε ≤ 4 at δ = 10⁻⁶ per model release**, compared with **six independently trained local models and the current rule-based LACE-style discharge screen**.

The task category is **binary risk classification used for a ranked allocation decision**, not autonomous triage. The model produces a score; a care-management team consumes the top slice of the ranking each week against a fixed enrollment capacity. The decision recurs **monthly** at the governance layer (retrain, recalibrate, re-approve, suspend) and **daily** at the operational layer (who gets called). The unit of analysis is the **index admission**, and the prediction horizon is **30 days after discharge**, measured from the discharge timestamp, not from admission.

One primary success condition, stated in plain language: *the shared model must rank patients better than each hospital's own model by an amount that survives a pre-declared minimum detectable difference, without any hospital being able to reconstruct another hospital's patient records from what crosses the boundary.* The complete metric definitions, denominators, windows, and uncertainty method are in Section 9; the plain-language version is deliberately weaker than the operational definition so that it cannot be read as a benchmark claim.

The reusable pattern is **privacy-safe collaboration**: a decision that could technically be served by pooling data, but where pooling is blocked by law, contract, or institutional risk appetite, so the engineering question becomes *what can be learned without centralizing the records, and what does that cost in statistical power*. Secondary patterns are **federated learning** (the training topology that replaces pooling) and **validation** (the discipline that decides whether the resulting model is better than the local alternative, rather than merely different).

The decisive tradeoff, and the reason the naive framing fails: the improvement in discrimination that federation buys is **real but small and concentrated in the smallest sites**, while the privacy mechanism needed to make federation defensible **costs discrimination of the same order of magnitude**. A framing that asks only "can we train a shared model?" produces a yes, a shipped model, and no defensible claim of benefit. The framing that survives is a three-way comparison — local-only, federated, federated-under-budget — evaluated against an economic decision rule, where the answer may legitimately be "federate for three sites and share an evaluation harness, not a model for all six."

A second surprise, which the worked example in Section 6 makes concrete: the binding constraint on this project is not bandwidth, compute, or even model quality. Training a 122,625-parameter tabular network over 400 federated rounds moves roughly **1.1 GiB** in total across all six sites — less than a single disk image. The binding constraints are the **privacy budget arithmetic** (a noise multiplier that looks small yields an ε in the hundreds), the **statistical power of the evaluation set** (8,400 held-out encounters cannot resolve a 1.5-point AUROC difference), and the **economics of the downstream program** (which do not break even at the assumed effect size). A team that sizes only the network will believe it has finished the problem.

**30-second statement a reader should be able to give:** six hospitals cannot pool records, so they train locally and exchange only model updates under secure aggregation and differential privacy; the question is whether the shared model beats each hospital's local model by enough to justify the privacy cost, and the answer here is that it does for the three smallest sites and does not clear the economic bar for the three largest at the assumed program parameters.

**What this case is not.** It is not evidence that any real consortium ran this design, not a claim that federated learning is superior to pooling, and not a claim about any vendor's federated platform. The pooled-data model in Section 6 is an **oracle upper bound computed for teaching purposes**, standing in for "what centralization would have bought"; it was not trained, because the premise of the case is that centralization is prohibited.

## 2. Context & Background

### 2.1 The workflow the model interrupts

A patient is admitted, treated, and discharged with a discharge disposition and a follow-up plan. Between discharge and day 30, three things can happen that the hospital cares about: the patient is readmitted (to any hospital in the network, ideally), the patient dies, or the patient stays out. The care-management program intervenes *before* discharge or in the first few days after it: a nurse navigator calls, medication reconciliation happens, a follow-up appointment is booked, a remote monitoring kit may be issued.

The intervention is scarce. Each hospital funds a fixed number of navigator slots per month, and enrolling a patient who would have done fine anyway consumes a slot that could have gone to someone else. So the operational question is a **ranking question under a capacity constraint**, not a threshold question under an error-rate constraint. This distinction drives almost every subsequent methodological choice: the metric that matters is not accuracy at 0.5, it is the precision of the top 1.3% of the ranking, and the calibration of scores near that cut.

Before the model, each hospital used a rule-based screen — a points-based instrument built from a handful of administrative flags (prior admissions, length of stay, discharge destination, comorbidity count). Rules of this kind are cheap, auditable, and stable, and their documented weakness is that they compress heterogeneous patients into coarse buckets: a patient with two short admissions and one with one long admission can score identically, and the instrument cannot express interactions between social and clinical risk. The baseline in this case is therefore **not** "no model"; it is a specific instrument with known coarse granularity, which is a much harder baseline to beat than a coin flip.

### 2.2 The six sites and why they cannot pool

| Site | Role | Index discharges (12 months) | 30-day readmission prevalence | Prior screening instrument | Data posture |
|---|---|---:|---:|---|---|
| H1 | Academic medical centre | 62,400 | 9.8% | LACE-style points | On-prem warehouse; legal review required for any external transfer |
| H2 | Regional referral centre | 41,200 | 11.4% | LACE-style points | On-prem; contractual bar on record export |
| H3 | Community hospital | 28,900 | 13.2% | LACE-style points | Vendor-hosted EHR; export requires vendor cooperation |
| H4 | Community hospital | 17,600 | 15.1% | LACE-style points | On-prem; no analyst dedicated to modelling |
| H5 | Critical access hospital | 11,300 | 16.9% | Ad hoc clinical judgement | Spreadsheet-based reporting |
| H6 | Rural critical access | 6,800 | 18.4% | Ad hoc clinical judgement | No analytics function; one BI analyst shared across three facilities |

Totals: **168,200** index discharges and, at the stated prevalences, **20,446** readmissions, for a consortium-wide prevalence of 20,446 / 168,200 = **12.16%**. The gradient in prevalence from 9.8% at the academic centre to 18.4% at the smallest rural site is the central fact of the case. It is not noise; it reflects case mix, catchment deprivation, and post-discharge support availability. Any model trained across these sites will face a **label-shift** problem: the same feature vector implies materially different absolute risk depending on where the patient is discharged from.

The prevalence gradient also creates a temptation that must be named early. Because H6's prevalence (18.4%) is nearly twice H1's (9.8%), a model that simply learned "site identity" would appear to discriminate well on a pooled test set while learning nothing clinically transferable. This is a form of **shortcut learning** that federated training makes *harder* to detect, because the pooled test set does not exist inside any single site and nobody can eyeball it.

Why the sites cannot pool is worth stating precisely, because "privacy" is often used as a vague blocker and vague blockers produce vague designs. Three independent barriers apply here, and they have different remedies:

- **Legal and contractual.** Record-level transfer of identifiable patient data across organizational boundaries requires legal basis that the consortium's counsel has not established. This is a constraint on *data*, not on *parameters*: a gradient is not a record, though the boundary is thinner than the naive view suggests (Section 8b, Section 12 F-05).
- **Institutional risk appetite.** Even where a legal basis could be constructed, each hospital's board bears reputational and liability exposure if a shared repository leaks. The governance board has stated that no central store of patient-level records will be created, at any encryption standard. This is a policy constraint, and policy constraints are legitimate design inputs.
- **Operational.** H4, H5, and H6 have no analytics staff who could produce a compliant extract in a reasonable time, and H3's EHR vendor charges per export. Centralization would therefore also be a *capacity* failure, not only a privacy one.

The third barrier is the one most often omitted from privacy narratives, and it changes the recommendation: a design that requires each site to run a nontrivial local training job imposes an **engineering** burden proportional to the site's technical maturity, which is inversely proportional to its data volume. The sites that benefit most from federation are precisely the sites least able to participate in a complex federation.

### 2.3 Stakeholders and their actual interests

| Stakeholder | Interest | What they will not accept | Decision authority |
|---|---|---|---|
| Consortium data governance board | Defensible, documented, revocable collaboration | A design whose privacy claim depends on trusting one participant's infrastructure | Approves or rejects the training topology and the privacy budget |
| Hospital CIOs (6) | No new audit findings | Any outbound flow they cannot enumerate and shut off | Can veto their site's participation |
| Care-management directors (6) | Higher precision in the enrollment queue | A model that reorders their queue without warning or an override | Owns the operational threshold |
| Clinical informatics / quality | Safety, equity, no gaming | A score whose inputs include post-discharge events | Signs off on feature admissibility |
| Patients (via patient advisory) | No surprise use of their data | Being scored by a model they cannot have explained | Consultation, not veto |
| Analytics engineering | Something maintainable by 2.5 FTE | A bespoke security protocol with no independent review | Implements and operates |

The governance board's stated non-negotiable — *"we will not trust any single participant's server with the union of our data, and we will not accept a privacy claim we cannot state as a number"* — is the constraint that forces both secure aggregation and differential privacy. Either alone leaves a gap: secure aggregation without DP leaves a model that memorizes; DP without secure aggregation requires a trusted aggregator that the board has already refused to create.

### 2.4 Scale, timing, and seasonality

The modelling window is a rolling 12 months of discharges; the case's snapshot is the period ending 2025-06-30, with the test window 2025-07-01 to 2025-12-31. Two seasonal effects matter and are easy to miss. First, readmission prevalence rises in winter months in temperate climates (respiratory admissions), so a test window that runs July–December spans a prevalence shift inside itself; a model calibrated on a July–June training period will be systematically miscalibrated by December. Second, care-management capacity is itself seasonal (holiday staffing), so the *capacity* against which the threshold is set is not constant across the test window. The evaluation in Section 9 therefore reports calibration by month as well as by site, and the threshold is re-derived monthly rather than fixed for the half-year.

Volume per month is 168,200 / 12 = **14,016.7 discharges**, and the program's consortium-wide capacity is **180 enrollments per month**, i.e. **1.28%** of monthly discharges. That ratio is the single most consequential number in the case: the model is being asked to discriminate within the top 1.28% of a ranking, where calibration error of a few percentage points translates directly into misallocated slots.

### 2.5 Prior attempts, and what the board learned from them

Two prior attempts frame the current decision. Both are **hypothetical** scenario elements, included because the decision's shape depends on them.

The first was a **centralized extract** proposed eighteen months earlier: each site would send a de-identified extract to a trusted third party that would build one model. It failed at the contract stage for three reasons — the de-identification standard could not be agreed (H1 wanted expert determination, H3's vendor would only support a fixed field list), re-identification risk through rare-diagnosis combinations was raised by the patient advisory group and never answered, and two sites would have needed to build extract pipelines they did not have staff to build. The lesson the board drew was *"centralization fails on governance, not on technology"*, which is directionally right but incomplete: it also failed on **capacity at the small sites**, which federation does not automatically fix.

The second was a **shared rule set**: harmonize the LACE-style instrument across all six sites, tune the point weights on each site's own data, and compare. This is a genuinely strong baseline and it is the one the case actually competes against. It requires no data movement at all, it is explainable to a clinician in one page, and it degrades gracefully when a feature is missing. Any argument for federated learning has to be an argument that beats this, not an argument that beats "nothing".

### 2.6 What "the model" is expected to change

It is important to be concrete about the causal chain the consortium believes in, because the evaluation in Section 9 will test each link separately and one link is where the case fails.

```text
[Better ranking] -> [Different patients enrolled] -> [More readmissions prevented]
      ^                      ^                                ^
   measurable            measurable                    NOT measurable here
   (AUROC, precision@k)  (overlap analysis)            (needs an experiment)
```

The first arrow is what the modelling work can establish. The second is arithmetic: given two rankings, the set difference at capacity is computable exactly. The third requires a randomized or quasi-experimental design that the consortium has not run, and the case is explicit that the third arrow is **unmeasured**. Section 6 shows that the second arrow is smaller than intuition suggests — the top-180 sets of the local and federated models overlap by roughly three quarters — and Section 6's economic arithmetic shows the third arrow is where the value actually lives, and where the program's cost structure decides the answer.

## 3. The Business Question

### 3.1 The question in the owner's words

The governance board chair's question, reconstructed for this composite case and therefore labeled an **illustrative paraphrase**, not a quotation:

> "We have six hospitals that cannot share records and will not accept a vendor holding all of them. We have a privacy budget our counsel will defend — epsilon four, delta minus six, per model release — and we have three months before the readmission program's contract renewal. Do we build one model together, do we each build our own, or do we keep the rules? And whatever we choose, I need to be able to say what it bought us in dollars and in patients, and what it cost us in privacy, in one page."

### 3.2 The decision contract

| Decision field | Content |
|---|---|
| Question | Should the consortium train a shared readmission model using federated learning with differential privacy, and if so for which sites and under which privacy budget — or should each site continue with a locally fitted model behind the shared rule set? |
| Decision owner | Consortium data governance board (six CIOs plus a chair with tie-breaking authority); individual CIOs retain a per-site veto |
| Trigger and deadline | Readmission program contract renewal in three months; H3's vendor export pricing changes at the same date; the current rule set has been in place for four years without revalidation |
| Available actions | (a) Local-only models, one per site, no cross-site learning; (b) federated model for all six sites, no DP; (c) federated model for all six sites with DP at ε ≤ 4, δ = 10⁻⁶; (d) federated model for a subset of sites that opt in, with DP; (e) retain the harmonized rule set and invest the difference in program operations |
| Value at stake | Program spend of 180 slots × $1,650 = **$297,000 per month** consortium-wide; avoided readmissions valued at $14,200 each on an illustrative basis; the difference between a good and a mediocre ranking at 1.28% capacity is worth roughly one order of magnitude less than the program's own effectiveness uncertainty |
| Objective | Maximize net expected value of the enrollment decision at fixed capacity, subject to no site's calibration error exceeding a stated bound; secondary objective, equalize discrimination across sites (the smallest site must not be the worst served) |
| Hard constraints | No record-level data leaves a site; ε ≤ 4 at δ = 10⁻⁶ per release; model must be explainable at the feature level to a clinician; inference must run inside each site's environment; no feature may encode post-discharge events; total engineering capacity 2.5 FTE |
| Scope | Six hospitals; adult inpatient discharges with a live discharge disposition; 30-day all-cause readmission to any consortium facility; rolling 12-month training window; monthly retrain |
| Out of scope | Pediatric readmissions; mortality prediction; ED revisits without admission; any use of the score for denial of coverage or for pricing; individual clinician profiling |

### 3.3 Objective versus proxy, and where they diverge

The business objective is **net value of the enrollment decision**. The modelling proxy is **AUROC of the 30-day readmission label**. These can diverge in four identifiable ways, and each divergence is a guardrail in Section 9:

1. **Discrimination versus calibration.** AUROC is invariant to any monotone transform of the score, so it is blind to the miscalibration that destroys a capacity-constrained queue. A model can gain 0.01 AUROC and lose 30% of queue precision if its calibration slope drifts at the site where capacity is scarcest. Guardrail: per-site calibration slope and ECE.
2. **Ranking versus set overlap.** Two models differing by 0.03 AUROC may select almost the same 180 patients. The value-relevant quantity is the **symmetric difference** of the top-k sets, not the AUROC gap. Guardrail: overlap at k.
3. **Risk versus uplift.** Predicted readmission risk is not the causal effect of enrollment. Enrolling the highest-risk patients can produce less benefit than enrolling patients with moderate risk and high responsiveness, because the highest-risk patients may be high-risk for reasons no navigator can change. Guardrail: the economic model must carry an explicit, separately justified relative risk reduction, and the case must not silently equate risk with benefit.
4. **Average versus worst site.** A pooled metric can improve while the smallest site degrades. Guardrail: the decision rule includes a per-site floor, not only a pooled mean.

Naming these four divergences before any modelling begins is what separates a decision-grade evaluation from a leaderboard. It also pre-commits the team to publishing a result it may not like: if the federated model wins on AUROC but loses on worst-site calibration, the decision rule says no.

### 3.4 What "no intervention" looks like, and what an expensive error looks like

The counterfactual is not idleness. If the board chooses (e) — keep the rules — the program continues to enroll by points score, the six sites continue to disagree about what the points mean, and H5 and H6 continue to enroll by clinical judgement with no audit trail. That is a stable, low-cost, low-benefit state. The case must be honest that this is a *reasonable* outcome and that the burden of proof sits with the modelling proposal.

An expensive false positive in this decision has two distinct forms, and the board cares about them differently:

- **A wasted program slot.** Cost $1,650 of program resource plus the opportunity cost of the patient who was not enrolled. Bounded, recurring, and visible in the queue.
- **A wrong privacy claim.** If the board approves a design and later cannot defend the ε figure, every model trained under it becomes a liability, the program is suspended, and the consortium's credibility with its own patient advisory group is damaged in a way that no model quality can repair. This is the asymmetric risk, and it is why Section 9 treats the privacy accounting as a *primary* evaluation criterion with a pass/fail gate rather than a compliance checkbox.

### 3.5 Why the question cannot be answered by building the thing

A team that starts by implementing federated averaging will produce a model in weeks and will then be unable to answer the chair's question, because the comparison that matters — federated-under-budget versus each site's local alternative, judged by net value at capacity with a per-site floor — requires three artifacts the implementation does not produce by itself: a **reproducible local baseline per site**, a **valid privacy accountant output tied to the actual training schedule**, and a **held-out evaluation set large enough to resolve the difference the decision depends on**. Section 6 shows that the third artifact is the binding one, and that a naive evaluation with 8,400 test encounters cannot resolve a 1.5-point difference at 80% power. Discovering that after the model is built is the classic sequencing failure of this class of project.

## 4. Assumptions & Constraints

### 4.1 Assumptions to test

| ID | Assumption | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | A 30-day readmission label can be observed for all six sites, including readmissions to a *different* consortium hospital | Data | Compare HIE-style admission feed against each site's internal admissions for a 10,000-encounter sample | Unverified | Labels under-count cross-site readmissions, biasing prevalences and penalizing sites that refer out | Restrict label to same-site readmission and declare the narrower outcome | Any change in health-information-exchange coverage / data steward |
| A-02 | The feature set available at discharge is identical in meaning across sites | Data | Field-level semantic audit of 40 candidate features against each site's data dictionary | Unverified; three features known to differ in definition (Section 5d) | A feature that means different things at different sites becomes a site-identity proxy | Drop or harmonize the discordant features; retrain | Any EHR upgrade / informatics lead |
| A-03 | Secure aggregation can be deployed with at least five of six sites holding the required key material | Infrastructure | Pilot round with dummy updates and one forced dropout | Unverified | Federation degenerates to a trusted aggregator, which the board has refused | Two-party or three-party sub-federation; reconsider centralization | Certificate rotation / security architect |
| A-04 | The privacy budget ε ≤ 4 at δ = 10⁻⁶ is achievable at a noise level that preserves usable discrimination | Model | Accountant output plus a utility curve across noise multipliers | Unverified; Section 6 shows σ = 30 gives ε ≈ 3.73 | Either the budget is breached or the model loses its advantage | Raise ε with board approval, reduce rounds, or restrict to a subset of sites | Accountant version change / privacy lead |
| A-05 | Program effect (relative risk reduction) is at least 28.3% among enrollees | Business | Prior program evaluation, or a stepped-wedge pilot | Unverified; the break-even figure is *derived*, the achieved figure is unknown | The program does not break even at any ranking quality | Re-price the program, target differently, or stop | Program contract renewal / care-management director |
| A-06 | Monthly retraining is sufficient; drift within a month is immaterial | Model | Score-distribution monitoring plus a mid-month shadow evaluation | Unverified | Mid-month calibration drift misallocates slots late in the month | Weekly recalibration of the intercept only | Any drift alert / MLOps owner |
| A-07 | A held-out evaluation set of ~8,400 encounters can resolve the decision-relevant difference | Statistical | Power calculation (Section 6, F-05) | **Violated** — MDE ≈ 2.5 AUROC points, larger than the effect the board hopes to see | The evaluation cannot distinguish the preferred option from the baseline | Extend the test window, pool across sites, or use a paired design with correlation correction | Any change in test-window size / analytics lead |

A-07 is the assumption most likely to reverse the recommendation, and it is already known to be violated at the design stage. That is deliberate: a case that only discovers its power problem at the end is a case about project management, not about method. Discovering it before training changes the design — the paired comparison in Section 6 exploits within-patient correlation to recover most of the lost power, and the recommendation in Section 10 is contingent on that correction.

The second assumption capable of reversing the decision is **A-05**. If the program's true relative risk reduction is 15% rather than 28%, no ranking improvement in the plausible range rescues the business case, and the correct action is (e) — keep the rules, re-price the program. The case therefore presents the economic arithmetic as a *sensitivity surface*, not a single number.

### 4.2 Constraints

| ID | Constraint | Type | Source | Status | Consequence | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| C-01 | No patient-level record may leave the site's boundary | Legal/policy | Board resolution; counsel opinion referenced but not reproduced here | Confirmed (scenario) | Rules out pooled training and any central feature store | Federation, or local-only | Any change in the legal basis / CIO |
| C-02 | ε ≤ 4 at δ = 10⁻⁶ per model release | Privacy | Board resolution | Confirmed (scenario) | Fixes the noise multiplier floor given the round count | Reduce rounds or restrict participation | Regulator guidance / privacy lead |
| C-03 | Inference inside each site's environment; no central scoring service | Security | CIO constraint | Confirmed (scenario) | Each site must host a model runtime; model artifacts must be portable | Batch scoring from a nightly extract | Any central runtime approval / CIO |
| C-04 | Score must be explainable at feature level | Clinical governance | Informatics sign-off requirement | Confirmed (scenario) | Rules out unconstrained deep models with uninterpretable interactions as the primary artifact | Monotone-constrained or additive model; post-hoc attribution with caveats | Clinical safety review / informatics |
| C-05 | Engineering capacity 2.5 FTE for 6 months | Resourcing | Program plan | Confirmed (scenario) | No bespoke cryptographic protocol development; use reviewed implementations only | Defer, or use a managed service (rejected: C-01 trust model) | Staffing change / analytics lead |
| C-06 | Enrollment capacity 180 per month consortium-wide | Operational | Program contract | Confirmed (scenario) | Threshold is a quantile, not a probability | Re-derive monthly | Contract renewal / care-management director |

C-01 and C-02 interact in a way that is easy to miss. C-01 forbids centralization, which forces federation; C-02 forbids an unbounded privacy loss, which forces noise. Federation multiplies the number of observable updates (one per site per round), and noise is added per round, so the two constraints jointly determine the achievable utility. Neither constraint alone would be difficult. Together they define the case's central arithmetic (Section 6, F-06 through F-08).

C-03 has a downstream consequence that shows up in Section 8: because scoring is local, each site must run a model runtime, which means six deployment targets with three different operating systems and one vendor-hosted environment with no shell access. The deployment surface, not the training algorithm, is what consumes most of the 2.5 FTE.

### 4.3 Assumptions explicitly *not* made

Stating non-assumptions prevents a class of silent overreach:

- **Not assumed:** that federation is privacy-preserving by itself. A federated model without DP leaks through the model weights; membership inference and property inference against federated updates are documented attack classes (Section 12, F-05).
- **Not assumed:** that differential privacy at the record level protects a patient who appears at three hospitals. Group privacy degrades the guarantee multiplicatively (F-08), and the case quantifies the degradation rather than waving at it.
- **Not assumed:** that a federated model matches a pooled model. Under label shift it can be worse than the best local model for the largest site, because it optimizes an average that no single site inhabits.
- **Not assumed:** that AUROC improvement implies program benefit. Section 3.3 lists four divergence mechanisms, and Section 6 shows the economic arithmetic can be negative at a positive AUROC gain.
- **Not assumed:** that the six sites' data are mutually independent samples. Patients move between sites; the effective sample size is smaller than the row count (F-15).

### 4.4 The three assumptions that would reverse the design

For interview and review purposes it helps to have the reversal conditions written down explicitly:

1. **If the legal basis for pooling were established** (A-01/C-01 reversed), the entire federated apparatus becomes unnecessary; centralized training with a clean holdout is simpler, cheaper, and strictly better statistically. The case's design is only justified by the prohibition, and a team that keeps the federated design after the prohibition lifts is optimizing for its own architecture.
2. **If ε ≤ 4 cannot be met at usable utility** (A-04 falsified with σ required above ~40), the correct output is not a weaker model but a different collaboration: share *evaluation* and *feature definitions*, not gradients. Section 10's recommendation is already shaped by this fallback.
3. **If the program's relative risk reduction is below ~15%** (A-05 falsified downward), ranking quality is not the constraining factor and the correct action is to re-price or retire the program. No model improvement fixes a negative-value intervention.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | One row per **index inpatient discharge** with a live discharge disposition. Readmissions are labels on the index row, not separate rows, to prevent a readmission from being both a label and a new index case within the same window |
| Grain and keys | `(site_id, encounter_id)` unique; `patient_key` is a site-local salted hash used only for within-site deduplication and for the overlap audit; it is never transmitted |
| Volume and window | 168,200 index discharges, 2024-07-01 through 2025-06-30 inclusive; test window 2025-07-01 through 2025-12-31 (8,400 encounters, 1,022 positives at 12.17%) |
| Time zone | Each site's local time for care events; all window arithmetic is performed in UTC with the site's offset applied at the boundary. DST transitions are handled by converting timestamps to UTC before differencing, never by adding 24-hour multiples to local times |
| Inputs | 220 candidate features in eight families: demographics (age band, sex, insurance class), prior utilization (admissions and ED visits in trailing 12 months, computed strictly before index admission), comorbidity flags from the discharge coding, labs at admission and discharge, medications on the discharge list, discharge disposition and follow-up plan fields, and a small set of social-context fields captured at registration |
| Outcome/label | `readmitted_30d = 1` if any inpatient admission to a consortium facility occurs with admission timestamp in `(discharge_ts, discharge_ts + 30 days]`. Death within 30 days without readmission is coded 0 for the primary label and separately flagged; competing-risk handling is discussed in 5d |
| Label maturity | A row is mature only when 30 days have elapsed since discharge *and* the last consortium admission feed has been reconciled through that date. Immature rows are excluded from training and evaluation rather than treated as negatives |
| Population and exclusions | Adults (≥ 18 at admission); inpatient discharges only; excludes transfers to another acute facility within 24 hours (the receiving facility's discharge becomes the index event), deaths during the index stay, discharges against medical advice with no forwarding address (2.1% at H5), and obstetric-only stays. Exclusions applied before any feature computation |
| Join semantics | Features join to the index encounter on `encounter_id` (1:1). Prior-utilization joins use a half-open window `[t-365d, t)` where `t` is the index **admission** timestamp. Discharge-medication joins are many-to-one and are aggregated to a count plus a high-risk-drug-class flag, never row-expanded |
| Splits | Temporal split by discharge date (train ≤ 2025-06-30, test ≥ 2025-07-01). Within training, each site holds out its own most recent 15% for local early stopping. No patient appears in both train and test (enforced by `patient_key` within site and by the overlap audit across sites) |
| Federation boundary | Only model updates (and, for evaluation, aggregate statistics with a minimum cell size) cross the boundary. Feature data, labels, row counts below 11, and any per-patient score vector stay local |

The decision to model **index discharges** rather than patients deserves a note, because it is the single most consequential grain choice. A patient with four admissions contributes four rows and four labels; the rows are not independent. An evaluation that treats them as independent overstates precision, and a training procedure that shuffles them into minibatches effectively samples the same patient four times, which interacts badly with per-example gradient clipping (a frequently visiting patient has their influence clipped per row but multiplied by visit count). The case keeps the discharge grain — it matches the operational decision, which is made at discharge — but attaches a cluster-robust evaluation (F-15) and reports the design effect.

The second consequential grain choice is the **label's cross-site scope**. Restricting the label to same-site readmission would be simpler and would avoid the health-information-exchange dependency in A-01, but it systematically under-counts readmissions for sites that refer complex patients elsewhere — H3 and H4 refer a substantial share of cardiac and oncology follow-up to H1 and H2. Under a same-site label, H3's measured prevalence would fall and its model would learn to predict *local* return, which is not the outcome the program is trying to prevent. The case therefore uses the cross-site label and treats A-01 as its most important data-validity assumption.

### 5b. Tools & Stack

The stack is chosen for reviewability by a small team, not for novelty. Every component is either a widely used open-source library or a protocol with a public specification, because C-05 forbids bespoke cryptography and the board's trust model forbids a managed service that would hold updates in the clear.

| Component | Role | Why this choice | Version/ownership note |
|---|---|---|---|
| Python 3.11 + PyTorch 2.x | Local model definition, training loop, per-example gradient computation | Per-example gradients are required for DP-SGD clipping; the library's `vmap`-based gradient computation makes this tractable without writing custom CUDA | Version pinned per site; H3's vendor environment runs a constrained image (see 8d) |
| Opacus (or an equivalent DP-SGD implementation) | Clipping, noising, and per-step accounting hooks | Reviewed implementation of a documented mechanism; avoids hand-rolled noise injection | Accountant version recorded per run; the accountant's output is an artifact |
| A vetted RDP/zCDP accountant | Converting the training schedule into (ε, δ) | Accounting is where privacy claims are usually wrong; use a published, versioned implementation | The zCDP closed form in F-07 is used as an independent cross-check, not as the primary accountant |
| Secure aggregation protocol (Bonawitz-style) | Masked summation of updates so the server never sees an individual update | Published protocol with dropout recovery by secret sharing | Requires key material per site; H5 has no PKI — see 8d constraint |
| gRPC over mutual TLS | Transport between sites and the aggregation service | Boring, mutually authenticated, testable with a packet capture | Aggregation service hosted by H2 under a neutral-operations agreement |
| Git + artifact registry | Versioned model, config, and run manifests | Reproducibility requirement (5e) | Every round's aggregate checksum logged |
| Parquet on each site's warehouse | Feature materialization | No data movement; each site owns its extract | Site-local schema mapped to a consortium canonical schema |
| Jupyter-free evaluation harness | Reproducible metric computation from saved score vectors | Prevents the "notebook result" failure mode | Scores stay local; only aggregates are returned |

Two choices deserve justification because a reasonable reviewer might choose otherwise.

**Why an aggregation service at all, rather than peer-to-peer.** A peer-to-peer design would remove the single aggregation endpoint, but it also removes the ability to enforce a round protocol (who submitted, in which order, with which model version) and makes dropout recovery combinatorially harder. The board's objection was to a *trusted* aggregator, not to a *coordinating* one; secure aggregation reduces the coordinator to a message router that sees only sums. This is the standard cross-silo tradeoff and it is the right one at n = 6.

**Why not secure multi-party computation of the whole training.** MPC would remove the need for DP's noise, but it is 10²–10⁴× slower per operation for the nonlinearities in a neural network and requires all parties online simultaneously for the whole training run. At 2.5 FTE and a three-month deadline this is not implementable, and it is worth stating plainly that "we could use MPC" is not a design unless the cost is carried.

### 5c. Method Summary

The pipeline is: local feature materialization → local minibatch training with clipped, noised gradients → secure aggregation of the update → global model update → broadcast → repeat for R rounds → local recalibration on each site's holdout → threshold selection against capacity → local deployment.

```text
input:  per-site feature tables F_k (never leave site), labels y_k
step 1  each site trains E local epochs on its own data from the current
        global weights w_t, with per-example gradient clipping at C and
        Gaussian noise sigma*C added to the summed gradient
step 2  each site computes delta_k = w_k_local - w_t  (the "update")
step 3  updates are masked, submitted, and summed under secure aggregation;
        the server observes only sum_k delta_k
step 4  w_{t+1} = w_t + sum_k (n_k / n) * delta_k        [F-01]
step 5  repeat for R rounds; log epsilon after every round
output: global weights w_R, plus a per-site calibration map and threshold
```

The baseline and alternatives, with an honest statement of what was tested:

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Harmonized rule set (LACE-style points, locally tuned weights) | Four years of operational use; locally tuned weights on each site's data | Explainable in one page; no data movement; degrades gracefully with missing fields; no privacy cost | Coarse buckets; cannot express interactions; weight tuning is itself a small modelling exercise per site | **Chosen as the baseline** | Never — it remains the fallback |
| Local-only gradient-boosted model per site | Each site's own 12-month history | No privacy cost; adapts to local case mix; simplest deployment | H6 has 6,800 rows and ~1,251 positives — too few for a stable 220-feature model | **Tested (simulated evaluation)** | If a site's volume grows 3× |
| Federated model, all sites, no DP | Simulated federation over the same data | Closest to pooled performance; largest gains at small sites | Model weights memorize; board has refused a no-DP design; membership inference documented | **Rejected on governance, not performance** | Only if C-02 were removed, which the board will not do |
| Federated model, all sites, DP-SGD at σ chosen to meet ε ≤ 4 | Accountant output + utility curve | Meets C-02; defends the privacy claim as a number | Utility loss at high σ; calibration drift; six deployment targets | **Chosen, then narrowed** — see Section 10 | If ε budget changes or site count changes |
| Federated model, three smallest sites only, DP | Same as above, restricted membership | Gains concentrate where local data is thin; simpler governance; fewer deployment targets | Large sites get no benefit and may resent funding it | **Chosen in the final recommendation** | If a large site's local model degrades |
| Pooled-data model (centralized) | None — prohibited | Statistical upper bound | Violates C-01 | **Not tested; used only as an oracle bound in Section 6** | If the legal basis changes |

The selection logic is worth making explicit because it is the part most often left implicit: the DP-federated option was not chosen because it scored best. It was the only option that satisfied C-01 and C-02 simultaneously, and the modelling work then asked a narrower question — *at which membership does it beat the local alternative by enough to matter?* That reframing is what produced the three-site recommendation, and it is transferable (Section 13).

#### The aggregation rule

```text
FORMULA [F-01]: w_{t+1} = sum_{k=1..n} (n_k / n) * w_k^{(t)}
WHERE: w_{t+1} global weights after round t [dimensionless parameter vector, P entries];
       w_k^{(t)} weights after client k's local training in round t [P entries];
       n_k local training examples [rows]; n = sum_k n_k [rows];
       n_k/n aggregation weight [dimensionless, sums to 1].
CONSTRAINT: identical model architecture at every client; equal local epoch count E;
            the average is over weights, not over gradients, so it is only equal to a
            single large-batch gradient step when E = 1 and clients start from w_t;
            requires every participating client to report (or the server to know) n_k.
NUMERICAL EXAMPLE: n = 168,200 with n_k = 62,400 / 41,200 / 28,900 / 17,600 /
                   11,300 / 6,800. For H6, n_k/n = 6,800/168,200 = 0.04043 (4.04%).
                   Under equal weighting each client would receive 1/6 = 0.16667,
                   an over-weight factor of 0.16667/0.04043 = 4.122.
INTERPRETATION: proportional weighting makes the global objective an unbiased estimate
                of the pooled empirical risk; equal weighting silently redefines the
                objective as "average per-site performance", which is a different
                decision and must be argued for explicitly.
```

The over-weight factor of **4.12** is the number to carry into any discussion of fairness in this design. Under equal weighting H6's 6,800 encounters would carry the same influence as H1's 62,400 — a site with 4% of the data would steer 17% of the update. Some teams argue for this deliberately, on the grounds that small sites are underrepresented and their patients are highest-risk. That is a legitimate *policy* argument, but it must be made as policy; presented as a neutral default it is a silent reweighting of the objective.

#### The local objective

```text
FORMULA [F-02]: F_k(w) = (1/n_k) * sum_{i in D_k} l(f_w(x_i), y_i) + (mu/2) * ||w - w_t||^2
WHERE: F_k local objective at client k [nats or log-loss units];
       l binary cross-entropy [nats per example]; f_w model output in (0,1);
       mu proximal coefficient [dimensionless]; w_t global weights at round start;
       the second term is present only in the FedProx variant (mu = 0 recovers FedAvg).
CONSTRAINT: mu >= 0; the proximal term is anchored to w_t, so it changes every round;
            with mu > 0 the client's optimum is biased toward w_t, which is the point.
NUMERICAL EXAMPLE: with mu = 0.01 and ||w - w_t||^2 = 41.7 after E = 5 epochs, the
                   penalty contributes 0.5 * 0.01 * 41.7 = 0.2085 nats to F_k, against
                   a local cross-entropy of about 0.41 nats per example — a 51% relative
                   penalty, i.e. mu = 0.01 is far from a small perturbation at this
                   parameter scale and must be tuned against the update norm, not chosen
                   by folklore.
INTERPRETATION: the proximal term is a trust region in parameter space; its strength has
                to be expressed relative to the squared distance clients actually travel.
```

#### Candidate generation of features and the admissibility rule

Feature admissibility follows a single mechanical rule: a feature is admissible if and only if its **latest contributing event timestamp is ≤ the discharge timestamp**, verified per feature by inspecting the underlying event table, not by inspecting the extract. In practice this rule eliminated 9 of the 220 candidates and forced the redefinition of 4 more. The eliminations are listed in 5d because they are the most instructive part of the data work.

### 5d. Data provenance, quality, exclusions and leakage controls

| Source ID | Content | Owner | Extraction | Version | Known limitation |
|---|---|---|---|---|---|
| `SRC-D1` | Site admission/discharge/transfer feeds, six sites | Each site's data steward | Site-local SQL, monthly snapshot | Snapshot 2025-07-02 | H3's feed includes only the vendor's "encounter" object, which merges observation stays; observation stays were split out by a rule that is itself a modelling assumption |
| `SRC-D2` | Discharge coding (diagnoses, procedures) | Each site's coding unit | Site-local extract | Snapshot 2025-07-02 | Coding depth varies with site size; H6 codes at a coarser granularity, producing systematically fewer comorbidity flags |
| `SRC-D3` | Laboratory results | Each site's LIS | Site-local extract | Snapshot 2025-07-02 | Unit conventions differ (mg/dL vs mmol/L for two analytes); conversion applied at materialization with a recorded mapping |
| `SRC-D4` | Discharge medication list | Each site's pharmacy system | Site-local extract | Snapshot 2025-07-02 | Free-text sig fields are not machine-readable at H4 and H5; only structured fields used |
| `SRC-D5` | Cross-site admission feed (health information exchange) | Regional HIE operator | Query API | Snapshot 2025-07-02 | Coverage unknown for out-of-region readmissions; A-01 not yet verified |
| `SRC-D6` | Program enrollment and cost records | Care-management program office | Finance extract | FY2025 | Cost per enrollee is a blended figure; marginal cost differs by site |

Provenance and lineage are recorded per feature family: the canonical feature name, the source system, the transformation expression, the site-specific overrides, and the person who signed off. This is heavier than a typical modelling project's documentation, and it is justified here because the same extract feeds six organizations with different auditors.

**Quality profile.** Missingness is concentrated in three places: discharge labs at H5 (31% missing), social-context fields at H6 (64% missing), and structured medication fields at H4/H5 (18% missing). Missingness is **not** imputed across sites — a site-level mean imputation would leak site identity into the feature and make the model learn "H6 patients are missing social context" as a risk factor. Instead, missingness indicators are created and the imputation values are computed **within each site** from that site's training partition. The cost is that the imputation model itself is a small federated-learning problem; the case treats it as out of scope and uses site-local median imputation with an indicator, acknowledging that this makes the local preprocessing non-identical across sites and therefore weakens the "one global model" claim (see F-05 in Section 12).

**Exclusions, in order, with counts (illustrative but internally consistent):**

| Step | Excluded | Remaining | Reason |
|---|---:|---:|---|
| Raw discharge rows | — | 183,900 | All sites, all ages, all dispositions |
| Non-adult (age < 18) | 9,240 | 174,660 | Different readmission mechanism and different program |
| In-hospital death | 3,110 | 171,550 | No discharge to follow |
| Transfer to acute facility < 24 h | 1,890 | 169,660 | The receiving facility's discharge is the index event; keeping both double-counts one episode |
| Obstetric-only stay | 1,020 | 168,640 | Program does not serve this population |
| Discharge against advice, no contactable address | 440 | 168,200 | Cannot be enrolled even if scored; retaining them would inflate apparent capacity |

The last exclusion is a **selection effect** and must be reported: patients who leave against advice are plausibly at higher risk, so removing them lowers measured prevalence and removes exactly the patients the program cannot help. The correct interpretation is that the model is trained on an *enrollable* population, which is the right population for the decision, but it is not the right population for a clinical claim about all discharges.

**Leakage controls, and the four features that failed.**

1. **`post_discharge_ed_visit_7d`** — an ED visit within 7 days of discharge is strongly predictive of readmission and is *caused* by the same deterioration process. Its latest contributing timestamp is after discharge, so it is inadmissible. The failure mode it would have created is the classic one: excellent offline AUROC, zero operational value, because the feature is unavailable at the moment of scoring. Removed.
2. **`followup_appointment_booked`** — booked *after* discharge at some sites and *at* discharge at others. The site-level inconsistency means the feature would function as a proxy for "this site has a discharge-planning workflow", i.e. a site-identity shortcut. Removed pending harmonization, which the informatics group has not completed.
3. **`readmission_flag_prior_30d`** — computed from an admission that may itself fall inside the label window when a patient has closely spaced admissions. Redefined as "admission in `[t-365d, t)` relative to the *index admission* timestamp", which is strictly prior. Renamed to `prior_admission_count_365d` to make the window explicit in the name.
4. **`length_of_stay_bucket`** — admissible on timing, but computed from the *discharge* timestamp minus admission, which is available only at discharge. This is a genuine at-decision-time feature and was retained, but noted because it is frequently miscomputed using a midnight-to-midnight difference, which produces a one-day error for evening admissions.

The general control is mechanical: every feature carries `max_event_ts`; the materialization job asserts `max_event_ts <= discharge_ts` for every row and fails the run otherwise. This assertion caught feature 2 during the third dry run, after it had already been included in a preliminary model — a concrete instance of the principle that leakage is found by checking timestamps, not by inspecting the model.

**Overlap and duplication.** `patient_key` hashes are exchanged as **salted, truncated** values (site-specific salt, 64-bit truncation) solely to estimate overlap, using a private set intersection rather than transmitting hashes. Estimated overlap: **4.8%** of patients appear at two or more sites within the year, **0.7%** at three or more. Two consequences follow. First, a patient in site A's training set may appear in site B's test set, which is cross-site leakage: the global model has seen that patient. The evaluation therefore reports a **cross-site-clean** variant in which test rows whose patient appears in any site's training window are excluded, costing 4.8% of test rows. Second, DP's guarantee is per-record; a patient contributing rows at three sites is a group of size 3, and the group-privacy degradation is quantified in F-08.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-01` | McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 | 2017 | FedAvg aggregation and the E-rounds communication reduction argument | Public | The paper's benchmarks are image/text tasks; not evidence about this tabular workload |
| `SRC-02` | Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016 | 2016 | DP-SGD clipping + Gaussian noise; moments accounting | Public | Original accountant is superseded; the closed-form cross-check in F-07 is used here instead |
| `SRC-03` | Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*, CCS 2017 | 2017 | Masked summation and dropout recovery by secret sharing | Public | Exact threshold parameterization must be checked against the deployed implementation |
| `SRC-04` | Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* | 2014 | Definition, Laplace/Gaussian mechanisms, composition, group privacy | Public | Tutorial text; the specific bounds used are re-derived in F-06..F-08 |
| `SRC-05` | Bun & Steinke, *Concentrated Differential Privacy* | 2016 | zCDP composition (additive ρ) and conversion to (ε, δ) | Public | Conversion is conservative relative to a full numerical accountant |
| `SRC-06` | Mironov, *Rényi Differential Privacy*, CSF 2017 | 2017 | RDP composition used by production accountants | Public | Not computed numerically in this file |
| `SRC-07` | Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020 | 2020 | Proximal term and its convergence behaviour under heterogeneity | Public | Convergence constants are workload-dependent; not transferred as claims |
| `SRC-08` | Karimireddy et al., *SCAFFOLD*, ICML 2020 | 2020 | Client-drift diagnosis and control-variate correction | Public | Requires per-client control state; not implemented here |
| `SRC-09` | Kairouz et al., *Advances and Open Problems in Federated Learning* | 2021 | Taxonomy of threats, non-IID effects, open problems | Public | Survey; used for framing, not for numeric claims |
| `SRC-10` | Hanley & McNeil, *The meaning and use of the area under a ROC curve*, Radiology 1982 | 1982 | SE of AUROC approximation used in F-04 | Public | Approximation degrades for AUROC near 1 or with very small positive counts |
| `RUN-01` | Simulated federation over the six-site snapshot | Not executed as a live federation; arithmetic performed from stated per-site metrics | Round-by-round aggregation and accounting chain | Illustrative | No live training occurred; metrics are constructed to be internally consistent |
| `RUN-02` | Accountant cross-check (F-07 closed form) | Recomputed by hand in Section 6 | Independent check on the reported ε | Derived | Closed-form zCDP bound is conservative; a numerical accountant would report a lower ε |
| `RUN-03` | Power calculation (F-04, F-05) | Recomputed by hand in Section 6 | Minimum detectable AUROC difference | Derived | Uses the Hanley–McNeil approximation and an assumed within-model rank correlation |

**What can be reproduced and what cannot.** The accounting chain, the communication volume, the power calculation, the calibration aggregation, and the net-benefit arithmetic are all closed-form and can be recomputed from the numbers printed in this file — every intermediate quantity is shown. The model metrics cannot be reproduced because no model was trained: they are constructed values chosen to be internally consistent with the mechanism (gains larger at small sites, degradation monotone in σ). A reader who wants to reproduce them empirically must obtain six real datasets under governance, which is precisely the constraint the case is about.

**Configuration that would be recorded for a real run** (listed so the reproducibility gap is explicit rather than implied): code revision per site, PyTorch and Opacus versions, accountant version, `R`, `E`, batch size, learning rate schedule, `C`, `σ`, `δ`, random seeds per client, model architecture hash, per-round aggregate checksums, and the identity of the aggregator's deployment. None of these exist in this file because no run occurred; the list is the *requirement*, and Section 12 F-08 shows what happens when a team records the seed but not the accountant version.
