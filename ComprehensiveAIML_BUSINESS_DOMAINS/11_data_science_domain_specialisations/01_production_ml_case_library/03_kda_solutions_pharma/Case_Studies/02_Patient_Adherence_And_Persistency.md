---
case_id: "CASE-01-03-002"
case_title: "Patient adherence and persistency in a fictional support programme"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Pharma commercial and patient-services analytics"
primary_pattern: "metric-definition-drift"
secondary_patterns: ["immortal-time-bias", "informative-censoring"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Patient support programme lead"
source_index: "INDEX.md"
review_trigger: "A change to the metric specification, the refill-gap rule, the observation window, the pack-size mix, or the consent and data-minimisation rules"
---

# Patient adherence and persistency in a fictional support programme

Evidence statement. No real programme, product, cohort, or patient was observed for this file. Every
count, proportion, interval, and cost below is arithmetic on constructed inputs and is labelled
ILLUSTRATIVE where it is used. The statistical machinery — the medication possession ratio, the
proportion of days covered, the Kaplan-Meier product-limit estimator with Greenwood variance, a
two-proportion power calculation, and the E-value — is standard, and each result is hand-recomputed
in §6 with intermediate values shown so a reader can reproduce it without a computer. The decision
narrative is a simulated decision record. Where a real-world quantity is unknown this file says
unknown rather than asserting a number.

## 1. The One-Line Framing

Given a consented cohort of patients enrolled in a fictional refill-support programme, decide whether
to fund an expansion of the structured reminder component at the next quarter boundary, or to
continue the current manual outreach unchanged.
The decision is programme-level: it allocates operating budget to a support service, and it never
selects a patient for a treatment, changes a prescription, or conditions care on participation.
The accountable owner is the fictional patient support programme lead, who may approve, reject, or
defer, and whose name goes on the record either way.
The decision grain is one programme in one market over one twelve-month observation window, with
patients as the measurement unit but not the decision unit.
The cut-off is the enrolment freeze at the quarter start; the index date is the first dispensing
after enrolment is verified, and everything measured before that instant is fair game while every
later record is excluded from the feature set.
The baseline is the current manual outreach, which the programme already runs and which has never
been measured against a structured alternative in this scenario.
The primary measure is twelve-month persistency, defined as the product-limit estimate of remaining
on therapy under a pack-scaled refill-gap rule that must be locked before the analysis runs.
The secondary measure is the proportion of days covered over a fixed 180-day window, which is
reported alongside persistency because the two disagree in informative ways.
The material threat is that persistency is not one number: it is a family of numbers indexed by a
gap rule, a denominator convention, an observation window, and a censoring assumption, and the
choice among them can move the headline by more than thirty percentage points.
The second threat is that censoring in a support programme is rarely random, because the patients who
drift away are the ones the programme is meant to help and the ones whose records go quiet first.
The success condition is a prespecified persistency difference whose lower confidence bound clears a
cost-per-additional-persistent-patient ceiling the owner states in advance.
The failure condition is any metric-definition change after the read, any disclosure breach, or a
result that cannot be separated from the definition that produced it.
The honest default when the definition is unsettled is to lock the specification and defer, not to
pick the definition that flatters the programme.

## 2. Context & Background

### 2.1 The programme as constructed

A fictional manufacturer runs a support programme for patients starting a chronic oral therapy.
Enrolment is voluntary and requires documented consent, including consent for the programme to use
dispensing records for service improvement.
The programme offers nurse-led onboarding calls, an optional reminder service, and a helpline.
A patient may withdraw at any time, and withdrawal removes the patient from active outreach.
The programme does not prescribe, does not adjust doses, and does not communicate clinical advice
outside approved materials reviewed by a medical reviewer.
A treating clinician remains the only party who decides therapy, and the programme never contacts a
clinician to argue for continuation.
A privacy lead approves the data flows, the retention period, and the minimum-cell rule for any
report that leaves the programme.
A health-economics reviewer owns the cost inputs and the affordability ceiling.

### 2.2 Why adherence is a hard measurement problem rather than a hard modelling problem

Adherence is usually defined as the degree to which a patient takes medication as prescribed, and it
is not directly observable in dispensing data: what is observable is a sequence of fills with days
supply, from which coverage is inferred.
Persistency is a duration concept — how long a patient remains on therapy before a clinically
meaningful interruption — and it requires a rule for what counts as an interruption, which is where
most disputes originate.
A patient who fills every thirty days is unambiguously persistent; a patient who fills every
forty-five days on a thirty-day supply is persistent under some rules and non-persistent under
others, and both readings are defensible.
Because the measure is constructed, the analysis cannot be separated from its specification, and a
specification changed after seeing results is a degree of freedom rather than a methodological
refinement.
The modelling is comparatively easy: a product-limit curve, a two-proportion contrast, and a
sensitivity table. The judgement is in the definition and in the interpretation of censoring.

### 2.3 The organisational pressure

Programme budgets are renewed annually, and renewal arguments are written in the language of
persistency improvement.
A vendor offering a reminder platform will quote a persistency lift, usually from a single-arm
before-and-after comparison with a favourable gap rule.
The owner therefore faces a specific hazard: the number that justifies the budget and the number that
would survive review are different numbers, and the difference is often larger than the effect.
This case is about holding both numbers in view and making the decision on the defensible one.

### 2.4 What the source material provided

The catalogue entry for this topic is a title only: patient adherence and persistency.
No programme documents, cohort extracts, vendor decks, or results were available, and none are
invented here as fact.
Every organisational element above is a scenario construct, and every number is labelled as
illustrative arithmetic.

## 3. The Business Question

### 3.1 The decision in operational form

Illustrative question: for the next four quarters, should the programme fund the structured reminder
component at a stated per-patient cost, or continue the manual outreach at its current level?
The question is asked once per budget cycle, before the first quarter of that cycle, and it cannot be
reopened mid-cycle without a written amendment.
The answer must be executable: a budget line, a per-patient cost, an enrolment target, a locked
metric specification, and a named reviewer who certifies the read.
An answer that only says "the programme works" is not a decision, because it does not allocate
anything.

### 3.2 The four available actions

Fund the expansion: commit the budget, enrol to target, and read the primary measure at twelve months
under the locked specification.
Continue the manual programme unchanged: no incremental spend, no new evidence, and the same
unmeasured claim persists for another year.
Run a bounded parallel-cohort comparison first: a smaller spend, a shorter horizon, and a narrower
question — whether the reminder component changes refill timing at all.
Defer with a named trigger: no spend now, revisit when the metric specification is signed, when
consent coverage reaches a stated threshold, or when the dispensing feed's revision behaviour is
characterised.
Deferral is a real option here because the definitional work is a precondition, not a delay tactic.

### 3.3 The objective, stated so it can fail

The primary objective is the difference in twelve-month persistency between the structured-reminder
cohort and the usual-care cohort, estimated under a specification locked before the first read.
The specification fixes four things that could otherwise move the answer: the gap rule, the
observation window, the denominator convention, and the handling of censoring.
The commercial objective is the cost per additional persistent patient, computed from the incremental
programme cost and the absolute persistency difference, and compared with a ceiling the owner states
before seeing data.
The equity check is that the persistency difference must be reported by the prespecified subgroups —
age band, pack size, and region — because a programme that helps only the easiest patients is a
different proposition from one that helps the ones at risk.
The safety check is that no outreach may occur to a withdrawn patient, and no clinical advice may be
given outside approved materials.

### 3.4 What the decision does not authorise

It does not authorise contacting patients who withdrew consent, or re-contacting them through another
channel.
It does not authorise using adherence data to restrict access, to price, or to rank patients for any
commercial purpose.
It does not authorise a claim that the programme improves clinical outcomes, because clinical
outcomes are not measured here.
It does not authorise publishing patient-level or small-cell results outside the approved disclosure
policy.

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption or constraint | Verification method | Owner | Consequence if false |
|---|---|---|---|---|
| A-01 | Dispensing records capture essentially all fills for enrolled patients | Reconcile a sample against pharmacy-reported totals | Data steward | Coverage is understated and persistency is biased downward |
| A-02 | Days supply is recorded accurately per fill | Compare recorded days supply against pack size and quantity dispensed | Data steward | Coverage arithmetic inherits the error multiplicatively |
| A-03 | The dispensing feed settles within four weeks of the event date | Re-extract the same historical period at +1, +2 and +4 weeks | Outcome steward | The read date moves; early reads mix mature and immature events |
| A-04 | Enrolled and usual-care cohorts are comparable before enrolment | Standardised differences on prior utilisation and prior therapy | Analytics lead | The contrast measures who enrolled, not what the programme did |
| A-05 | Censoring is unrelated to the hazard of discontinuation | Compare censoring rates and reasons across cohorts and time | Analytics lead | Persistency is overstated; a bounding analysis is mandatory |
| A-06 | Pack-size mix is stable across cohorts and over the window | Tabulate pack size by cohort and quarter | Analytics lead | A fixed gap rule silently misclassifies one pack size |
| A-07 | Consent covers the analysis use and the retention period | Privacy review of the consent text and the data flow | Privacy lead | The cohort cannot be used; the analysis is not performed |
| C-01 | Withdrawal removes a patient from outreach immediately | Nightly reconciliation of withdrawal list against the outreach queue | Operations | A contact incident occurs; the read is suspended pending review |
| C-02 | Only approved content may be sent | Content registry version check on every template | Medical reviewer | Delivery stops for the affected template |
| C-03 | No patient identifiers leave the approved environment | Architecture review plus export-list audit | Privacy lead | Export blocked; analysis re-run inside the environment |
| C-04 | Small cells are suppressed before release | Minimum-cell rule applied before human review | Privacy lead | Report withheld and re-issued at a coarser grain |
| C-05 | No clinical recommendation is made to a patient or clinician | Medical reviewer audits a sample of outreach records | Medical reviewer | Programme content is suspended |

### 4.2 Which assumptions can reverse the decision

Three assumptions dominate. If censoring is informative (A-05), the persistency estimate is biased
upward by an unknown amount and the true effect may be smaller or absent; the bounding analysis in
§6.10 is what keeps the decision honest rather than the point estimate.
If pack-size mix differs between cohorts (A-06), a fixed gap rule compares two different measurements
and the observed difference is an artefact; this is the mechanism behind the thirty-three-point swing
in §6.4.
If the enrolled and usual-care cohorts were already different before enrolment (A-04), the contrast
describes who opted in, which is a real finding about the programme's reach but not evidence that the
programme worked.
The remaining assumptions affect precision and disclosure, not direction.

### 4.3 Constraints that bound the design

Enrolment is voluntary, so the comparison population is self-selected no matter how the analysis is
framed; randomisation of a support service is possible in principle but constrained by consent and by
the ethics of withholding a service that is already offered.
The observation window is bounded by the budget cycle at twelve months, which is shorter than the
natural history of a chronic therapy.
The dispensing feed cannot be enriched with clinical outcomes without a separate approval, so the
analysis speaks to refilling behaviour and not to health.
The privacy regime forbids reporting cells below a minimum size, which forces some subgroup results
to be combined rather than published.
The programme cannot pay for medications or reduce copayments, so the mechanism studied is
information and reminder behaviour only — a substantial limitation on the size of any plausible
effect.

## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** Two grains are used and must not be confused. The dispensing table has one row per fill per
patient, keyed by pseudonymous patient identifier plus dispensing date plus product code. The analysis
table has one row per patient, keyed by the same pseudonymous identifier, carrying the index date,
the coverage intervals, the censoring date, and the outcome.

**Cut-off timestamp.** The enrolment freeze occurs at 23:59 local time on the last day of the prior
quarter. The index date is the first dispensing date on or after the freeze for a patient whose
enrolment was verified before the freeze. Any fill dated before the index date contributes to prior
utilisation covariates only and never to the coverage intervals.

**Population.** Consented enrolled patients in the single constructed market, aged eighteen or over,
with at least one dispensing of the study product on or after the freeze, no prior dispensing of the
same product in the preceding 180 days, and no record of therapy discontinuation in the product
record before enrolment. The usual-care comparator cohort is drawn from the same market under the same
inclusion rules but without programme enrolment.

**Keys and joins.** The pseudonymous identifier is stable within the market and is the only join key
permitted. No name, date of birth, address, or clinician identifier enters the analysis table. The
mapping between the pseudonymous identifier and any identifiable record is held by a separate team
and is not accessible to the analytics role.

**Units.** Days supply is an integer count of days. Coverage is a union of date intervals expressed in
days. Persistency is measured in days from index to the discontinuation event or censoring.
Proportions are dimensionless and reported to three decimal places. Costs are currency units per
patient per year.

**Fields present at the decision time.** Enrolment date, index date, product code, pack size, days
supply per fill, dispensing dates, region, age band, prior 180-day utilisation count, and the
programme component the patient received.

**Fields deliberately excluded from the feature and outcome logic.** Any clinical result, any
patient-reported outcome, any clinician note, any post-freeze change to enrolment status used as a
covariate, and any record whose event date falls after the patient's censoring date. The last
exclusion is the leakage boundary and is restated in §5d.

**Constructed sizes.** 400 enrolled patients in the persistency cohort, 1,055 per arm in the planned
confirmation test, and a 180-day window for the coverage measure alongside the 365-day persistency
window. None of these were observed.

### 5b. Tools & Stack

A restricted analysis environment holds the pseudonymised dispensing extract and the enrolment
register, with no outbound network path.
A version-controlled analysis script computes coverage intervals, the product-limit curve, and the
contrasts; the script hash is recorded in the specification register and any change requires a new
version identifier.
A specification register holds the locked metric definition, its version, its approver, and the date
of approval — this register, not the analysis script, is the authority on what the metric means.
A reporting layer applies the minimum-cell rule before any table is released.
No vendor platform, runtime version, or production pipeline is asserted as implemented; the stack is
a design proposal for a programme that does not exist in this scenario.

### 5c. Method Summary

Coverage is computed per patient by merging the date intervals implied by fills and days supply, then
summing the merged length inside the observation window.
The proportion of days covered divides that merged length by the fixed window length; the medication
possession ratio divides the sum of days supply by the same denominator, which is why the two diverge
whenever fills overlap.
Persistency is estimated by the product-limit method from index to the discontinuation event, with
patients censored at withdrawal, at the administrative end of the window, or at the last observable
fill when the feed cannot confirm a later event.
The discontinuation event is a refill gap exceeding a grace period measured from the end of the last
covered day; the grace period is expressed in multiples of the pack's days supply, not as a fixed
number of days.
The cohort contrast is a difference in twelve-month persistency, with a risk-ratio counterpart and a
cluster check if patients are clustered by prescriber.
Sensitivity analyses are prespecified: the gap rule at two settings, the denominator at two settings,
and a bounding analysis that reclassifies all censored patients as events.
A before-and-after comparison within the enrolled cohort is explicitly rejected as the primary
analysis, because enrolment timing correlates with the decision to seek support.
A per-protocol restriction to patients who received at least one reminder call is reported as a
secondary, labelled analysis only, because it conditions on a post-enrolment behaviour.

### 5d. Provenance, Quality, Exclusions, Leakage

Each extract carries an event date, an arrival date, an extraction date, and a version identifier; a
correction to a historical fill creates a new version rather than mutating the old row.
Days supply is validated against quantity dispensed and pack size; a mismatch above a stated
tolerance is flagged and either corrected from the pack record or excluded with a reason code.
Overlapping fills are retained and merged rather than dropped, because dropping the later fill
silently shortens coverage and biases the ratio downward.
Negative-value or reversed-date rows are quarantined and counted, never repaired silently.
The enrolled cohort is reconciled against the outreach queue so that patients who withdrew are
excluded from outreach on the same day, not on the next reporting cycle.
Missing coverage is not zero: a patient with no observable fills for a period is flagged as
unobserved and handled under the censoring rules rather than assumed non-adherent.
Leakage controls: the specification is frozen and hashed before the first read; the outcome logic is
implemented once in a shared function used by every table; and a negative-control cohort of patients
on an unrelated chronic therapy is run through the identical pipeline, since a large apparent effect
there would indicate a pipeline or cohort-assembly defect rather than a programme effect.
Identity quality is checked by confirming that no pseudonymous identifier appears with two different
index dates in the same window.

### 5e. Reproducibility & Method-Selection Ledger

| Decision | Chosen | Rejected alternative | Reason for rejection | Reversal trigger |
|---|---|---|---|---|
| Primary metric | Twelve-month persistency by product-limit estimate | Proportion of days covered | Coverage averages over a window and hides the timing of stopping | Coverage becomes the contractual measure |
| Gap rule | Grace period scaled to days supply, one pack | Fixed 60-day gap | A fixed gap allows two missed fills on a 30-day pack and zero on a 90-day pack | Pack mix becomes uniform across the market |
| Observation window | 365 days from index | 180 days | Shorter window truncates the tail where differences concentrate | Budget cycle shortens to two quarters |
| Denominator for coverage | Fixed 180 days from index | Days to last fill plus supply | The latter gives a one-fill patient a coverage of 1.00 | Not reversible; the bias is structural |
| Censoring handling | Product-limit with a bounding sensitivity analysis | Complete-case analysis | Complete-case discards exactly the patients at highest risk | Censoring proven independent with adequate power |
| Cohort contrast | Parallel consented cohorts with standardised comparison | Single-arm before-and-after | Enrolment timing correlates with the outcome | A randomised or stepped-wedge design is approved |
| Per-protocol analysis | Secondary and labelled | Primary | Conditions on post-enrolment behaviour | Receipt becomes near-universal |
| Subgroup reporting | Prespecified four subgroups with a multiplicity note | Exploratory scan | Scan results would be unactionable and inflated | Subgroups preregistered with separate budgets |

Every row can be changed only by an amendment recorded in the specification register, and each
amendment must name the tables it invalidates.

## 6. Analysis Walkthrough

### 6.1 The first computation is not an effect

The first quantity computed in this analysis is a coverage table per patient, because every later
number is a function of it and an error here propagates silently.
The coverage table records, per patient, the merged intervals, the total covered days, the sum of days
supply, the number of fills, the last covered day, and the largest gap.
Only after that table is reconciled against the dispensing counts does any contrast get computed.

### 6.2 Derivation 1: proportion of days covered versus medication possession ratio

```text
FORMULA
  covered_days = length( union of [fill_date, fill_date + days_supply - 1] ) inside the window
  PDC = covered_days / window_days
  MPR = (sum of days_supply for fills inside the window) / window_days

WHERE
  fill_date     = date the dispensing occurred (date, no time component)
  days_supply   = integer days the fill is expected to last, >= 1
  window_days   = fixed observation length in days (180 here)
  PDC           = proportion of days covered, dimensionless, 0..1 by construction
  MPR           = possession ratio, dimensionless, can exceed 1 when fills overlap

CONSTRAINT
  Both use the same fixed denominator. MPR equals PDC only when no two fills overlap and no
  fill extends beyond the window. PDC is capped at 1.00; MPR is not, which is the diagnostic
  signal that a patient is stockpiling rather than adhering.

NUMERICAL EXAMPLE      (ILLUSTRATIVE patient, 180-day window, days 1..180)
  Fill 1 at day 1,   30 days -> covers   1..30
  Fill 2 at day 29,  30 days -> covers  29..58     (overlap of 2 days: 29 and 30)
  Fill 3 at day 63,  90 days -> covers  63..152
  Union: 1..58 (58 days) plus 63..152 (90 days) = 148 covered days
  Uncovered: 59..62 (4 days) + 153..180 (28 days) = 32 days, and 148 + 32 = 180
  PDC = 148 / 180 = 0.82222  -> 82.2%
  MPR = (30 + 30 + 90) / 180 = 150 / 180 = 0.83333 -> 83.3%
  Difference = 2 days = 1.1 percentage points, exactly the overlap

INTERPRETATION
  The two metrics differ by the double-counted overlap. On a single patient this is cosmetic; across
  a cohort, early refillers accumulate overlap and the MPR mean drifts above the PDC mean, which
  makes the metric choice a function of refill behaviour rather than of adherence.
```

Boundary case for this formula: take a patient who refills every 25 days, at days 1, 26, 51, 76, 101,
126 and 151. That is seven fills of 30 days each, so the sum of days supply is 210 while the union of
intervals covers days 1 to 180 exactly, giving MPR = 210/180 = 1.1667 and PDC = 180/180 = 1.0000. A
metric that can print 117% is a supply ratio, not a coverage proportion; reporting it as "116%
adherent" is a unit error, and capping it at 1 destroys the stockpiling signal that the uncapped value
carried.

### 6.3 Derivation 2: the fixed-horizon denominator and the one-fill patient

The denominator convention is the single largest source of disagreement between programme reports and
reviewer calculations, so it is demonstrated on the same twelve patients used later.

```text
FORMULA
  PDC_fixed(p)    = covered_days(p) / 180
  PDC_observed(p) = covered_days(p) / min(180, last_covered_day(p))

WHERE
  last_covered_day(p) = fill_date of the final fill + days_supply - 1, capped at 180
  PDC_observed uses the patient's own observable span as the denominator
  both are dimensionless proportions in 0..1

CONSTRAINT
  PDC_observed is only interpretable as a coverage measure if the observation span is unrelated
  to the outcome — which it is not, because the span ends precisely when the patient stops
  filling. The metric is therefore a within-episode coverage measure, not a cohort adherence
  measure.

NUMERICAL EXAMPLE      (ILLUSTRATIVE twelve-patient enumeration, 30-day packs)
  Patient 5 fills once at day 1 and never again.
    covered days = 30; last covered day = 30
    PDC_fixed    = 30 / 180 = 0.1667
    PDC_observed = 30 / 30  = 1.0000
  Patient 4 fills at days 1, 31 and 130: covered 90 days over three blocks, last covered day 159.
    PDC_fixed    = 90 / 180 = 0.5000
    PDC_observed = 90 / 159 = 0.5660
  Cohort sums over the twelve patients:
    sum of covered days    = 1565  -> mean 1565/12 = 130.4167 days
    PDC_fixed mean         = 130.4167/180 = 0.724537
    sum of observed ratios = 10.0938 -> mean 10.0938/12 = 0.841151
  Inflation from the denominator change alone = 0.841151 - 0.724537 = 0.116614 -> 11.7 points

INTERPRETATION
  Switching the denominator moves the cohort mean from 72.45% to 84.12% with no change in any
  patient's behaviour. The three patients who filled once or stopped early (patients 5, 7, 8)
  each score 1.000 under the observed denominator — the worst adherents in the cohort become the
  best-scoring ones, which is the signature of immortal-time bias expressed as a ratio.
```

### 6.4 Derivation 3: the gap rule, enumerated patient by patient

Persistency depends on what counts as an interruption. Twelve constructed patients make the size of
that dependence exact rather than rhetorical.

```text
FORMULA
  gap_k = fill_date_{k+1} - (fill_date_k + days_supply_k)
  discontinuation at fill_k if gap_k > G, occurring on day fill_date_k + days_supply_k + G
  persistent at horizon H if no gap before H exceeds G and (H - last_covered_day) <= G

WHERE
  gap_k   = integer days of no supply between consecutive fills, may be negative for early refills
  G       = grace period in days
  H       = 180 for this enumeration
  a patient with a negative final gap is persistent by definition

CONSTRAINT
  The rule is "strictly greater than G", so a gap of exactly G is tolerated. This choice must be
  stated, because flipping it to "greater than or equal to" reclassifies every boundary patient.

NUMERICAL EXAMPLE      (ILLUSTRATIVE enumeration, all packs 30 days, H = 180)
  P1  fills 1,31,61,91,121,151      gaps 0,0,0,0,0     covered to 180   G=30: P    G=60: P
  P2  fills 1,31,61,121,151         gap 30             covered to 180   G=30: P    G=60: P
  P3  fills 1,31,101,131,161        gap 40             covered to 180   G=30: NP   G=60: P
  P4  fills 1,31,130                gap 69             last cover 159   G=30: NP   G=60: NP
  P5  fills 1                       no refill          last cover 30    G=30: NP   G=60: NP
  P6  fills 1,31,61,91,165          gap 44             covered to 180   G=30: NP   G=60: P
  P7  fills 1,31,61,91,121          uncovered 30       last cover 150   G=30: P    G=60: P
  P8  fills 1,31,61,91              uncovered 60       last cover 120   G=30: NP   G=60: P
  P9  fills 1,32,63,94,125,156      gaps of 1          covered to 180   G=30: P    G=60: P
  P10 fills 1,31,61,91,121,176      gap 25             covered to 180   G=30: P    G=60: P
  P11 fills 1,31,61,151             gap 60             last cover 180   G=30: NP   G=60: P
  P12 fills 1,31,61,152             gap 61             last cover 180   G=30: NP   G=60: NP
  Counts: G=30 persistent = 5/12 = 0.41667 ; G=60 persistent = 9/12 = 0.75000
  Swing = 0.75000 - 0.41667 = 0.33333 -> 33.3 percentage points

INTERPRETATION
  Two defensible rules, applied to identical data, differ by a third of the cohort. P11 and P8
  are the instructive cases: both sit exactly on a boundary, so the strict-versus-inclusive
  choice flips them, and a report that does not state the convention is not reproducible. Any
  programme claim quoted without its gap rule is uninterpretable, not merely imprecise.
```

### 6.5 Derivation 4: twelve-month persistency with censoring, by the product-limit method

Censoring is unavoidable: patients withdraw, the window ends, and some records simply stop. Ignoring
it by computing a naive proportion wastes the partial information; ignoring it by treating censored
patients as persistent overstates the result.

```text
FORMULA
  S(t) = product over intervals of ( 1 - d_i / n_i )
  Var( S(t) ) = S(t)^2 * sum over intervals of d_i / ( n_i * (n_i - d_i) )
  CI95 = S(t) +/- 1.96 * sqrt( Var( S(t) ) )

WHERE
  n_i = patients at risk at the start of interval i
  d_i = discontinuation events in interval i
  c_i = patients censored in interval i, removed after the interval
  S(t) = probability of remaining persistent to time t, dimensionless, 1.0 at index
  the variance formula is the Greenwood estimator

CONSTRAINT
  Non-informative censoring: censoring times must not carry information about the hazard.
  Independence of patients (or a cluster-robust counterpart when they share a prescriber), and
  no competing event that removes a patient while still on therapy.

NUMERICAL EXAMPLE      (ILLUSTRATIVE cohort of 400, 365-day window)
  Interval   at risk  events  censored   factor (1 - d/n)     S
  1..30        400      20       5        1 - 20/400 = 0.950000     0.950000
  31..60       375      25       8        1 - 25/375 = 0.933333     0.886667
  61..90       342      22      10        1 - 22/342 = 0.935673     0.829630
  91..180      310      40      25        1 - 40/310 = 0.870968     0.722581
  181..365     245      55      60        1 - 55/245 = 0.775510     0.560369
  Arithmetic check on the risk set: 400-20-5=375 ; 375-25-8=342 ; 342-22-10=310 ;
  310-40-25=245 ; 245-55-60=130 event-free at the end.
  Events total 162, censored total 108, and 162+108+130 = 400.
  Greenwood sum = 20/(400*380)=0.000131579 ; 25/(375*350)=0.000190476 ;
                  22/(342*320)=0.000201023 ; 40/(310*270)=0.000477901 ;
                  55/(245*190)=0.001181524 ; total = 0.002182503
  Var = 0.560369^2 * 0.002182503 = 0.314013 * 0.002182503 = 0.000685334
  SE  = sqrt(0.000685334) = 0.0261789
  CI95 = 0.560369 +/- 1.96*0.0261789 = 0.560369 +/- 0.051311 = [0.509058, 0.611680]

INTERPRETATION
  Twelve-month persistency is 56.0% with an interval of 50.9% to 61.2%. Note the naive
  proportion, 238/400 = 59.5%, is higher than the product-limit estimate, which is the expected
  direction when censoring concentrates among patients who stay: the naive figure credits those
  patients with a full year they were not observed to complete.
```

### 6.6 Derivation 5: the grace period expressed in missed fills

A fixed grace period is not a fixed amount of leniency, because leniency depends on how long a pack
lasts. The conversion is one line but it changes which patients are called persistent.

```text
FORMULA
  missed_fills_tolerated = floor( G / days_supply )

WHERE
  G           = grace period in days
  days_supply = pack duration in days, >= 1
  floor(x)    = greatest integer <= x
  the result is a count of complete pack-durations with no supply

CONSTRAINT
  Valid when the patient refills a full pack at a time and the gap is measured from the end of
  the last covered day. With partial fills or split dispensing, the count is approximate and the
  rule must be applied on the actual days supply of each fill.

NUMERICAL EXAMPLE
  G = 60: days_supply 30 -> floor(60/30) = 2 missed fills tolerated
          days_supply 90 -> floor(60/90) = 0 missed fills tolerated
          days_supply 7  -> floor(60/7)  = 8 missed fills tolerated
  G scaled to one pack (G = days_supply):
          30 -> floor(1) = 1 ;  90 -> floor(1) = 1 ;  7 -> floor(1) = 1
  Reclassification size: with 35% of the cohort on a 90-day pack (140 of 400) and a median gap
  after a missed fill of 95 days among them, a fixed 60-day rule classifies most of those 140 as
  discontinued at a point where they had missed exactly one pack.

INTERPRETATION
  A fixed 60-day rule is twice as strict for a 30-day pack as for a 120-day pack, and eight times
  more lenient for a 7-day pack than for a 30-day pack. Scaling the grace period to one pack makes
  the measure comparable across the market; leaving it fixed makes the pack mix, not the patient,
  the driver of the persistency number.
```

### 6.7 Derivation 6: sizing the confirmation test

The programme's decision needs a sample size that can detect the difference the owner would act on,
not the difference the vendor quoted.

```text
FORMULA
  n_per_arm = 2 * (z_{1-alpha/2} + z_{1-beta})^2 * pbar * (1 - pbar) / delta^2
  n_enrol   = n_per_arm / (1 - loss)
  MDE       = (z_{1-alpha/2} + z_{1-beta}) * sqrt( 2 * pbar*(1-pbar) / n_per_arm )

WHERE
  pbar  = pooled twelve-month persistency, dimensionless
  delta = target absolute difference in persistency, dimensionless
  loss  = share of enrolled patients with unusable or unobservable records
  alpha = 0.05 two-sided (z = 1.959964), power 0.80 (z = 0.841621)

CONSTRAINT
  Equal allocation, a single analysis at a fixed horizon, normal approximation valid because
  n*p and n*(1-p) both exceed 5. Any interim look requires an alpha-spending rule and a larger
  sample, which this calculation does not include.

NUMERICAL EXAMPLE
  pbar = (0.560 + 0.620)/2 = 0.590 ; delta = 0.620 - 0.560 = 0.060
  z_sum   = 1.959964 + 0.841621 = 2.801585 ; z_sum^2 = 7.848878
  pbar*(1-pbar) = 0.590 * 0.410 = 0.241900
  numerator = 2 * 7.848878 * 0.241900 = 3.797288
  n_per_arm = 3.797288 / 0.0036 = 1054.80 -> 1,055 per arm
  loss = 0.20 -> n_enrol = 1055 / 0.80 = 1318.50 -> 1,319 per arm
  MDE check at n = 1055: 2.801585 * sqrt(2*0.2419/1055)
                       = 2.801585 * sqrt(0.00045858) = 2.801585 * 0.0214145 = 0.059994

INTERPRETATION
  The design detects a 6.0-point absolute difference (0.06/0.56 = 10.7% relative) at 80% power, which
  requires 1,055 analysable patients per arm and 1,319 enrolled once a 20% loss is allowed for.
  Anything smaller than six points would need a materially larger cohort, so the owner must state
  whether a four-point difference would even change the funding decision before the test is sized.
```

### 6.8 Derivation 7: the observed contrast and its commercial conversion

The constructed read gives 649 of 1,055 persistent in the reminder cohort and 590 of 1,055 in usual
care. The arithmetic below is the entire decision input.

```text
FORMULA
  delta_hat = p1 - p0
  SE = sqrt( p1(1-p1)/n1 + p0(1-p0)/n0 )
  cost_per_additional_persistent = incremental_cost_per_patient / delta_hat
  RR = p1 / p0 ;  E = RR + sqrt( RR*(RR-1) )

WHERE
  p1, p0 = persistency proportions in reminder and usual-care cohorts
  n1, n0 = analysable patients per cohort
  incremental_cost_per_patient = programme cost above usual care, currency per patient per year
  E = E-value, the minimum confounder strength on the risk-ratio scale needed to nullify the result

CONSTRAINT
  Both cohorts measured on the same 365-day window with the same locked gap rule; patients who
  withdrew counted as censored, not as events; cost excludes any medication subsidy, because the
  programme does not provide one.

NUMERICAL EXAMPLE
  p1 = 649/1055 = 0.615166 ; p0 = 590/1055 = 0.559242
  delta_hat = 0.055924 -> 5.59 percentage points
  var1 = 0.615166*0.384834/1055 = 0.236737/1055 = 0.00022439
  var0 = 0.559242*0.440758/1055 = 0.246490/1055 = 0.00023364
  sum  = 0.00045803 ; SE = 0.021402 ; z = 0.055924/0.021402 = 2.613
  CI95 = 0.055924 +/- 1.96*0.021402 = [0.013977, 0.097872] -> [1.40 pp, 9.79 pp]
  incremental cost = 85 currency units per patient per year
  cost per additional persistent patient = 85/0.055924 = 1,519.92
  at the interval's lower bound          = 85/0.013977 = 6,081.42
  RR = 0.615166/0.559242 = 1.1000 ; E_point = 1.1000 + sqrt(1.1000*0.1000) = 1.4317
  RR lower = 0.573219/0.559242 = 1.0250 ; E_ci = 1.0250 + sqrt(1.0250*0.0250) = 1.1851

INTERPRETATION
  The difference is 5.59 points with a lower bound of 1.40 points, so the cost per additional
  persistent patient ranges from about 1,520 to about 6,081 currency units across the interval —
  a four-fold spread that the point estimate conceals. The E-values say a confounder that raises
  both enrolment and persistence by 43% would nullify the estimate, while one of 18.5% would nullify
  the interval's claim of any benefit; in a voluntary programme, the first is entirely credible.
```

### 6.9 Where the formulas break: the censoring bound

The product-limit estimate assumes that censoring carries no information about the hazard. In a
support programme that assumption is weak, because withdrawal and record silence both correlate with
disengagement. The honest response is a bound rather than a defensible-sounding claim.

Treating all 108 censored patients as if they had discontinued at their censoring times gives the
extreme lower case: 162 + 108 = 270 events out of 400, so at most 130 of 400 remain persistent, which
is 32.5%. Treating all censored patients as persistent to the end of the window gives the extreme
upper case: 238 of 400, which is 59.5%. The product-limit estimate of 56.0% sits near the upper end of
that bracket, which is informative: it means the estimate is only defensible if censoring is closer to
non-informative than to informative, and the programme has no evidence either way in this scenario.

A second break is subtler. If censoring rates differ between cohorts — the reminder cohort may retain
contact longer, so fewer of its patients go silent — then the two product-limit curves are estimated
under different censoring regimes, and the difference between them partly measures retention in the
programme rather than persistence on therapy. The check is to compare censoring rates by cohort and by
quarter; a difference of more than a few points invalidates the head-to-head read.

A third break is the single-read horizon. At exactly 365 days, patients who enrolled late have less
than a full window of observation, so their contribution to S(365) is an extrapolation through the
tail of the curve, where the risk set is smallest and the variance is largest. The remedy is to report
the risk set size at each plotted point, because a curve that looks stable at 300 patients at risk can
be visibly unstable at 40.

### 6.10 The definition-sensitivity table

Four defensible specifications, applied to the same constructed data, produce four headline numbers.
This table is the case's central exhibit because it shows why a single quoted persistency figure
without its specification is not a measurement. Rows A and B are computed from the risk table in
§6.5; rows C and D are ILLUSTRATIVE counterparts constructed to the same cohort size, because only
the twelve-patient enumeration in §6.4 was computed event by event.

| Specification | Gap rule | Denominator | Censoring | Persistency at 12 months |
|---|---|---|---|---|
| A — programme report | fixed 60 days | fixed 365 days | all censored treated as persistent | 0.595 (238/400) |
| B — locked primary | one pack (30 days) | fixed 365 days | product-limit | 0.560 (product-limit, §6.5) |
| C — lenient gap, ILLUSTRATIVE | one pack, inclusive boundary | fixed 365 days | product-limit | 0.592 |
| D — pack-blind lenient, ILLUSTRATIVE | fixed 90 days | fixed 365 days | product-limit | 0.631 |

The spread between the tightest and loosest defensible specifications is 0.071 in this construction —
seven percentage points, larger than the 5.59-point effect the programme wants to claim. That single
comparison is what forces the specification to be locked before the read rather than chosen after it.

Note that specification A is not dishonest in isolation; it is a different question answered with a
different estimator, and its 59.5% is a correct answer to "what share of patients had any observable
coverage in the last period we tracked". It becomes misleading only when it is presented as
persistency alongside a 56.0% figure computed another way.

## 7. The Finding

First finding, definitional. The persistency measure is not stable: the gap rule alone moves the
twelve-patient enumeration from 41.7% to 75.0%, and the denominator convention alone moves the
twelve-patient coverage mean from 72.45% to 84.12%. Any programme claim that does not carry its
specification is uninterpretable.

Second finding, comparative. Under the locked specification the reminder cohort persists at 61.5%
against 55.9% for usual care, a difference of 5.59 points with an interval of 1.40 to 9.79 points.
The direction is consistent with a real effect and the interval excludes zero.

Third finding, commercial. The cost per additional persistent patient is about 1,520 currency units at
the point estimate and about 6,081 at the interval's lower bound. Whether that is acceptable depends
on a ceiling the owner has not yet stated, which is itself a finding: the decision cannot be made on
the statistics alone.

Fourth finding, fragility. An unmeasured factor that raises both enrolment and persistence by about
43% would nullify the estimate, and one of about 18.5% would nullify the interval's claim of benefit.
Voluntary enrolment makes such a factor plausible, so the estimate is best read as supporting a
properly designed confirmation rather than a rollout claim.

Fifth finding, censoring. The product-limit estimate sits near the top of its own extreme-case bracket
(32.5% to 59.5%), which means it depends on an assumption the programme has not tested. Reporting the
bracket alongside the estimate is the only honest presentation available at this stage: if censoring
is even mildly informative, the true figure could be several points lower, and the effect against the
comparator narrows with it.

The finding as a sentence: the programme probably helps, the size of the help is smaller than the
headline, and the headline is not stable enough to be renewed on.

## 8. System/Solution Design

### 8.1 Components required for a repeatable measurement

| Component | Responsibility | Input | Output | Failure behaviour |
|---|---|---|---|---|
| Enrolment register | Record consent, enrolment date, and withdrawal | Consent record plus enrolment form | Versioned cohort membership with consent scope | Refuses to output a patient whose consent scope excludes the analysis |
| Dispensing reconciler | Merge fills into coverage intervals | Fill rows plus days supply plus pack record | Coverage table with merged intervals and gap statistics | Quarantines fills with unresolvable days supply rather than guessing |
| Specification register | Hold the locked metric definition and its hash | Approved specification document | Version, approver, effective date | Blocks any analysis whose script hash does not match the registered version |
| Persistency engine | Compute product-limit curves and variances | Coverage table plus censoring dates | Curve, risk sets, confidence bands | Refuses to emit a curve without risk-set sizes at each point |
| Cohort comparator | Standardise and contrast cohorts | Two cohort tables plus covariates | Difference, ratio, and balance table | Reports imbalance above a threshold instead of adjusting silently |
| Sensitivity runner | Execute the prespecified alternative specifications | Locked specification set | Definition-sensitivity table | Fails loudly if any specification is missing from the register |
| Disclosure filter | Apply minimum-cell rules | Result tables | Releasable aggregates | Suppresses and reports which cells were withheld |
| Withdrawal propagator | Remove withdrawn patients from outreach same-day | Withdrawal events | Updated outreach queue | Halts the queue rather than risk a contact |

### 8.2 Design choices worth defending

The specification register is the authority, not the analysis script, because a script can be edited
by anyone with commit access while a register entry requires an approver and a date.
The persistency engine refuses to draw a curve without risk sets, because the commonest misreading of
a survival curve is at the tail where the risk set is invisible.
The reconciler quarantines rather than imputes, because an imputed days supply propagates into every
downstream ratio with no trace.
The withdrawal propagator runs before outreach and on the same day, because a late withdrawal is not a
reporting defect, it is a contact incident.
The sensitivity runner executes all registered specifications every time, so that the definitional
spread is a routine output rather than an occasional forensic exercise.

### 8.3 Operating cadence

Daily: withdrawal propagation and outreach-queue reconciliation.
Weekly: fill reconciliation, quarantine review, and a duplicate-identifier check.
Monthly: coverage-table refresh, censoring-rate monitoring by cohort, and a specification-integrity
check confirming the script hash matches the register.
Quarterly: a definition-sensitivity run and a review of whether the primary specification still
answers the question the owner is asking.
Annually: a persistence-curve refresh with the full window, a disclosure review of every released
table, and a written continue-or-retire decision on the programme.

### 8.4 What is deliberately not built

No patient-level risk score, because a score that ranks patients for outreach on predicted
non-adherence would require a justification the programme does not have and would invite uses outside
the programme's remit.
No real-time adherence dashboard, because the feed settles in weeks and a live dashboard would display
immature counts as if they were facts.
No automated content escalation, because content approval is a human gate and automating it would
convert a governance control into throughput.

## 9. Evaluation & Validation

### 9.1 The measurement contract

| Element | Definition |
|---|---|
| Numerator (persistency) | Discontinuation events in the interval, where an event is a refill gap exceeding one pack-duration after the last covered day |
| Denominator (persistency) | Patients at risk at the start of each interval, after removing prior events and censoring |
| Numerator (coverage) | Merged covered days inside the window |
| Denominator (coverage) | 180 fixed days from index |
| Population | Consented enrolled patients meeting all inclusion rules, plus a parallel usual-care cohort under the same rules |
| Window | 365 days for persistency; 180 days for coverage; both from the index dispensing |
| Comparator | Usual-care cohort in the same market, standardised on prior utilisation and pack size |
| Guardrails | Zero contacts to withdrawn patients; zero unapproved content deliveries; zero clinical recommendations; suppression applied before release |
| Uncertainty | Greenwood interval for persistency; two-proportion interval for the cohort difference; E-values for fragility; extreme-case bounds for censoring |
| Status | Not executed. No data were collected and no patient was involved. Every figure above is computed from constructed inputs |

### 9.2 Validations performed while constructing this case

Arithmetic re-check: every result in §6 was recomputed by hand from its printed inputs, and the
intermediate lines shown are those hand computations rather than software output.
Internal consistency: the risk-set accounting in §6.5 satisfies 400 − 20 − 5 = 375 through to
245 − 55 − 60 = 130, and events plus censored plus surviving equals 400, so no patient is
double-counted or lost.
Cross-check between metrics: the summed covered days, 1,565, divided by 12 and by 180 gives 0.724537,
which equals the mean of the twelve individually computed PDC_fixed values (1.0000, 0.8333, 0.7778,
0.5000, 0.1667, 0.7556, 0.8333, 0.6667, 0.9722, 0.8611, 0.6667, 0.6611); the observed-span mean of
0.841151 differs by construction, not by an arithmetic error.
Limit check: the product-limit estimator returns exactly 1.000 at time zero and is non-increasing,
and the Greenwood variance returns zero when there are no events, both as required.
Boundary check: the twelve-patient enumeration reproduces P8 and P11 as flip cases under the
strict-versus-inclusive boundary, confirming the rule statement in §6.4 is load-bearing rather than
decorative.

### 9.3 What a live deployment would require

An independent reimplementation of the coverage and persistency code by a second analyst, reconciled
to the last decimal, with any difference investigated before release.
A back-test on a historical cohort with a known, externally audited persistency figure, to confirm the
pipeline reproduces it.
A negative-control cohort on an unrelated therapy run through the identical pipeline; a large apparent
effect there would indicate a cohort-assembly defect.
A specification-integrity test that deliberately edits the script and confirms the register blocks it.
A withdrawal-propagation drill that inserts a synthetic withdrawal and confirms the outreach queue
drops the patient before the next cycle.
A disclosure rehearsal with synthetic small cells, confirming suppression fires before human review.

### 9.4 Guardrails that veto a passing result

A statistically convincing difference does not override a contact to a withdrawn patient; that is an
incident, and the read is suspended pending review.
An unapproved content delivery voids the read for the affected period, because the intervention
actually delivered differs from the one specified.
A clinical recommendation outside approved materials voids the read and triggers a medical review,
regardless of the metric.
A disclosure failure voids the release even when the arithmetic is correct, because the harm occurs at
the moment of release, not at the moment of review.

### 9.5 Uncertainty the interval does not capture

The Greenwood interval captures sampling variability given the model. It does not capture informative
censoring, cohort self-selection, pack-mix drift, a specification change after the read, or a feed
revision that reclassifies fills mid-window. Each of those lives in the assumption register and the
sensitivity table, and reporting an interval without them is a claim of precision the design has not
earned.

## 10. Action Taken

The constructed decision record contains one decision and three conditions.

Decision: fund a confirmation cohort at 1,319 enrolled patients per arm, using the locked
specification (one-pack grace period, fixed 365-day window, product-limit estimation), and do not renew
the expansion on the current single-cohort read.

Condition one, specification primacy. The locked definition is registered and hashed; any future
report must display the gap rule, the denominator, and the censoring treatment alongside the number.
A report missing any of the three is returned without review.

Condition two, censoring evidence. Before the confirmation read is interpreted, the team must report
the censoring rate by cohort and quarter, and the extreme-case bracket. If the bracket's lower edge
falls below the owner's minimum effect, the read is reported as inconclusive rather than positive.

Condition three, commercial ceiling. The owner stated in advance that a cost above 3,000 currency
units per additional persistent patient would not be funded. At the observed point estimate the cost
is about 1,520, which clears the ceiling; at the interval's lower bound it is about 6,081, which does
not. The confirmation test is therefore designed to narrow the interval enough to decide, not to
re-litigate whether a difference exists.

Actions explicitly not taken: no patient-level targeting score, no change to the reminder content, no
change to consent scope, no expansion of the observation window, and no adoption of the vendor's
before-and-after headline as a programme claim.

The current manual outreach continues unchanged while the confirmation runs, so the comparator stays
the same policy rather than becoming a partly-treated hybrid.

## 11. Outcome & Results

The constructed analysis produced five results, each stated with its status.

Persistency under the locked specification is 0.560 with a Greenwood interval of 0.509 to 0.612 on 400
patients and 162 events. This is computed arithmetic on constructed inputs, not an observed programme
result, and no claim about a real therapy, market, or patient population is implied.

The definitional spread across four defensible specifications is 0.071, from 0.560 to 0.631, which
exceeds the 0.056 cohort difference the programme wanted to attribute to its own service. That
comparison is the case's most transferable single number.

The cohort contrast is 0.0559 with an interval of 0.0140 to 0.0979, computed on 1,055 analysable
patients per arm — the planned size, not a fielded one.

The commercial conversion gives 1,519.92 currency units per additional persistent patient at the point
estimate and 6,081.4 at the lower bound, against a ceiling of 3,000, which the lower bound fails.

The censoring bracket is 0.325 to 0.595, with the estimate near the top edge, which is why the read
was labelled supportive rather than confirmatory.

The decision outcome was a funded confirmation with three binding conditions. Under this collection's
governance vocabulary that is a partial positive: money moved, but not on the strength of the headline
number, and the conditions are the reason the decision is auditable.

## 12. Failure Modes & What Went Wrong

Six mechanisms are recorded, each with the symptom an analyst would see, the competing causes that
produce the same symptom, and a check that separates them. "Monitor the metric" is not a mechanism and
is not listed.

### 12.1 Denominator drift disguised as improvement

Symptom: the coverage mean rises quarter over quarter with no change in fill patterns, and the rise
disappears when an outside analyst recomputes from raw fills.
Competing causes: a genuine improvement in refilling; a switch from fixed to observed follow-up
denominators; a change in the window length; a change in the inclusion rule that removes short
follow-up patients.
Discriminating check: recompute the metric on a frozen historical extract using both denominators. If
the discrepancy equals the reported improvement, the denominator moved. A second check is the
distribution of observation spans; a spike at short spans points to the truncation that produces
the one-fill patient scoring 1.000.
Correction used here: the fixed 180-day denominator is mandatory for the coverage measure, and the
observed-span variant is reported only as a within-episode diagnostic.

### 12.2 Gap-rule switching after the read

Symptom: the persistency figure changes between drafts while the underlying data table hash is
unchanged.
Competing causes: a legitimate specification amendment; a boundary convention flip from strict to
inclusive; a change in the grace period from one pack to a fixed number of days; a change in which
fill counts as the last fill before censoring.
Discriminating check: diff the specification register entries behind the two drafts. A boundary flip
alone reclassifies every patient sitting exactly on the gap threshold, which in the twelve-patient
enumeration is 2 of 12 — enough to move a cohort-level figure by more than a point.
Correction used here: the register hash is checked by the runner, and a mismatch blocks execution.

### 12.3 Informative censoring in a voluntary programme

Symptom: the curve's tail flattens unexpectedly, and persistency rises when patients are censored
without explanation.
Competing causes: patients moving to a different product; genuine therapy completion; loss of
pharmacy coverage in the feed; withdrawal that correlates with disengagement.
Discriminating check: tabulate censoring reasons and rates by cohort and by elapsed time. Censoring
that concentrates in the first ninety days, or that differs by cohort, is unlikely to be independent
of the hazard. The extreme-case bracket quantifies what independence is worth: here it spans 0.325 to
0.595.
Correction used here: the bracket is reported with every persistency figure, and the read is labelled
supportive when the estimate sits near the bracket's upper edge.

### 12.4 Pack-size mix confounded with cohort

Symptom: the reminder cohort's persistency advantage vanishes when the analysis is restricted to the
30-day pack.
Competing causes: a real effect that is concentrated in one pack; differential enrolment by pack; a
change in dispensing practice that moved patients to 90-day packs mid-window; a fixed gap rule
treating the two packs differently.
Discriminating check: the missed-fill conversion, floor(G/days_supply), applied per pack. If a fixed
gap allows two missed fills for one pack and zero for another, the comparison is between measurements,
not patients. Standardising on one pack and re-running separates the artefact from the effect.
Correction used here: the grace period is scaled to one pack, and pack mix is a prespecified balance
covariate.

### 12.5 Enrolment self-selection read as programme effect

Symptom: the enrolled cohort differs from usual care on prior utilisation before the first
programme contact, and the difference persists after adjustment.
Competing causes: a real effect of the programme; selection of already-motivated patients into
enrolment; a difference in prescriber mix between the cohorts; eligibility rules that exclude
patients who would have done badly.
Discriminating check: standardised differences on pre-enrolment covariates, plus a pre-period outcome
comparison — if the cohorts already differed on refill behaviour before enrolment, the contrast
measures who opted in. A further check is the enrolment route: patients referred by a nurse differ
from self-referred patients, and the mixture can shift over time.
Correction used here: cohort standardisation is reported, the pre-period comparison is published
alongside the result, and the claim is limited to the enrolled population.

### 12.6 Coverage inflation through unrecorded fills

Symptom: coverage exceeds the plausible maximum for a subset of patients, and the excess correlates
with a particular pharmacy chain.
Competing causes: genuine early refilling and stockpiling; duplicate fill records from a feed
re-send; cash-pay fills that arrive through a second channel; a days-supply field populated with pack
size rather than dispensed quantity.
Discriminating check: compare the fill count per patient against the pharmacy-reported total for a
sample, and inspect the days-supply distribution by chain. A spike at a suspiciously round value, or a
chain-specific duplicate pattern, identifies the mechanism.
Correction used here: the reconciler deduplicates on patient, product, and dispensing date before
merging intervals, and quarantines fills whose days supply disagrees with the pack record.

## 13. Transferable Lesson

The transferable lesson is that in adherence analytics the measurement is the analysis, and everything
downstream inherits the definition.

A persistency number is a four-tuple: a gap rule, a window, a denominator, and a censoring
assumption. Three of the four are choices, and in this construction the choice alone moved the
headline by more than the programme effect. Any team that reports one number without the tuple has
reported a decision, not a measurement.

The second lesson is that the flattering specification is usually a real specification. The
observed-span denominator is a correct measure of within-episode coverage; the naive proportion is a
correct answer to a different question. Definitional disagreement in this domain is rarely dishonesty
and almost always a failure to state the question, which is why a register with an approver matters
more than an argument about which number is right.

The third lesson is that censoring deserves a bracket, not a caveat. A caveat says the assumption may
not hold; a bracket says what the answer would be if it did not. The bracket here spans 27 points, and
no reviewer can weigh a caveat of that size, but they can weigh a bracket.

The fourth lesson is that voluntary enrolment limits the claim before any data arrive. A programme
that people choose to join measures the effect on people who chose to join, and no amount of
standardisation converts that into a population effect. Naming the limitation in the objective is
cheaper than discovering it in review.

The fifth lesson is that a stated commercial ceiling is what turns a statistic into a decision. Without
a ceiling, a 5.6-point difference with a 1.4-point lower bound can be argued either way forever; with
one, the answer is arithmetic, and the sample size follows from the width of the interval rather than
from a vendor's slide.

## 14. Connections & Cross-References

### 14.1 Patterns in this collection

`metric-definition-drift` is the primary pattern: the reported quantity changes because its
specification changes, not because behaviour changes.
`immortal-time-bias` appears as the observed-span denominator, where patients are credited with full
coverage over a span that begins at their first fill and ends at their last.
`informative-censoring` appears in the withdrawal and record-silence mechanisms and is handled by
bounding rather than by assumption.

### 14.2 Contrast with sibling cases

Compared with the promotional-response case in this unit, the difficulty here is not confounding by
assignment but confounding by measurement: the exposure contrast is comparatively well defined, while
the outcome is a construct with several defensible versions.
Compared with a forecasting case, the deliverable is a curve and a definition, not a point prediction,
and the model's accuracy is not the binding constraint.
Compared with a deployment case, the guardrails are consent and disclosure rather than latency and
throughput, and the failure that matters most is a contact to someone who withdrew.

### 14.3 Prerequisites and handoffs

Prerequisite: interval arithmetic over dates, the product-limit estimator and its variance, and the
distinction between a ratio of supplies and a proportion of covered time.
Handoff to payer and access work: formulary changes move both pack size and switching behaviour, so an
access event during the window is a specification hazard as well as a commercial one.
Handoff to omnichannel attribution: a reminder sent through two channels is one exposure for the
patient and two rows for the analyst, which reintroduces the overlap problem at the contact level.
Handoff to brand-performance decomposition: any decomposition that uses persistency as a component
inherits the definitional spread documented here, and must publish its tuple.

### Interview pack

**Q1. A patient fills 30 days at day 1, 30 days at day 29, and 90 days at day 63, in a 180-day window.
Compute PDC and MPR and explain the gap between them.**
Answer: PDC 148/180 = 0.822; MPR 150/180 = 0.833; the difference is the two-day overlap at days 29–30.
Reasoning: The union of intervals is 1–58 and 63–152, totalling 148 days, while the sum of days supply
is 150; the two days of double supply are counted once by coverage and twice by possession.
Follow-up: What would the patient have to do to make MPR exceed 1.00, and why does that matter for a
cohort mean?
Red flag: Reporting 83% as adherence without saying which of the two ratios produced it.
Scoring: 2 for the covered-day union, 2 for both ratios, 1 for naming the overlap as the mechanism.

**Q2. Twelve patients under a 30-day pack yield 5 persistent at G = 30 and 9 at G = 60. Compute both
proportions and state what the difference is evidence of.**
Answer: 0.417 and 0.750, a 33.3-point swing, which is evidence about the rule rather than about the
patients.
Reasoning: The same fills are classified twice; patients whose only irregularity is a gap between 31
and 60 days move from event to persistent, which is a definitional change with no behavioural content.
Follow-up: Which two patients in the enumeration flip purely because of the strict-versus-inclusive
boundary, and why does that make the convention load-bearing?
Red flag: Concluding that the patients "improved" or that the 60-day rule is more accurate without
saying what clinical meaning the grace period is meant to encode.
Scoring: 2 for both proportions, 2 for the interpretation, 1 for identifying the boundary cases.

**Q3. Using the interval table — 400 at risk, 20/25/22/40/55 events against risk sets
400/375/342/310/245 — compute twelve-month persistency.**
Answer: 0.560, from 0.95 × 0.933333 × 0.935673 × 0.870968 × 0.775510.
Reasoning: Each factor is 1 − d/n for that interval, and the product compounds the conditional
survival; the risk set shrinks by events plus censoring each time.
Follow-up: What would the naive proportion be, and why is it higher than 0.560?
Red flag: Averaging the five interval factors instead of multiplying them, which gives 0.893 and is
meaningless.
Scoring: 2 for the factor list, 2 for the product, 1 for the naive-comparison explanation.

**Q4. Compute the Greenwood standard error for that curve and state what the interval is used for.**
Answer: 0.02618, giving 0.5091 to 0.6117; the interval bounds sampling variability only.
Reasoning: The Greenwood sum is 0.002182503, multiplied by S² = 0.314013 to give 0.000685334, whose
square root is 0.0261789; it says nothing about informative censoring or self-selection.
Follow-up: Why does a log-log transformed interval behave better near the extremes of the curve?
Red flag: Quoting the interval as a range of plausible programme effects.
Scoring: 2 for the sum, 2 for the SE and interval, 1 for the scope limitation.

**Q5. A report shows mean coverage rising from 72.45% to 84.12% with unchanged fills. What happened, and
how do you prove it?**
Answer: The denominator changed from a fixed 180 days to each patient's observed span.
Reasoning: Dividing by a span that ends at the last fill makes one-fill patients score 1.000; three
such patients in the enumeration rise from 0.167, 0.833 and 0.667 to 1.000 each, which alone moves the
mean by more than eleven points.
Follow-up: Under what circumstance is the observed-span denominator the right choice?
Red flag: Accepting the improvement because the numerator did not change.
Scoring: 2 for the diagnosis, 2 for the arithmetic mechanism, 1 for the legitimate use case.

**Q6. Size a two-arm test for 56% versus 62% persistency at 80% power, and allow for 20% loss.**
Answer: 1,055 analysable per arm, 1,319 enrolled.
Reasoning: p̄ = 0.59 gives p̄(1−p̄) = 0.2419; 2 × 7.848878 × 0.2419 = 3.7973, divided by 0.0036 gives
1,054.8; dividing that by 0.80 gives 1,318.5, rounded up to 1,319.
Follow-up: What sample would detect a four-point difference, and what does that do to the budget?
Red flag: Forgetting the attrition inflation and then wondering why the achieved interval is wide.
Scoring: 2 for the formula and pooled variance, 2 for the two sample figures, 1 for the rounding
direction and justification.

**Q7. The observed contrast is 649/1055 versus 590/1055 with an incremental cost of 85. Compute the
cost per additional persistent patient at the point estimate and at the interval bound.**
Answer: 1,519.92 and 6,081.42.
Reasoning: The difference is 0.055924 with SE 0.021402, so the interval is 0.013977 to 0.097872;
dividing 85 by each bound gives a four-fold spread that the point estimate hides.
Follow-up: At a ceiling of 3,000, what does the interval say about funding?
Red flag: Dividing by the relative difference, which produces a currency figure in the wrong units.
Scoring: 2 for the difference and interval, 2 for both costs, 1 for the ceiling interpretation.

**Q8. A fixed 60-day grace period is applied across a market with 30-day, 90-day and 7-day packs. What
does the rule actually mean for each?**
Answer: Two missed fills, zero missed fills, and eight missed fills respectively, from floor(60/DS).
Reasoning: Leniency is a ratio of grace period to pack duration, so one fixed number encodes three
different clinical meanings and makes pack mix a driver of the persistency figure.
Follow-up: What grace period would make the measure comparable, and what does that assume?
Red flag: Treating 60 days as equally lenient for every pack because the number is the same.
Scoring: 2 for all three conversions, 2 for the comparability argument, 1 for the assumption behind
scaling.

**Q9. Censoring is 108 of 400. Give the extreme-case bracket for twelve-month persistency and say what
it implies about the point estimate.**
Answer: 0.325 to 0.595, with the estimate of 0.560 near the top, so it depends on near-independent
censoring.
Reasoning: Treating all censored as events leaves 130 of 400 persistent; treating all as persistent
leaves 238 of 400; the product-limit estimate falls between them and close to the upper edge.
Follow-up: Which single diagnostic would most change your confidence in the estimate?
Red flag: Reporting 56.0% as though the assumption were verified.
Scoring: 2 for both bounds, 2 for the location of the estimate, 1 for naming a discriminating
diagnostic such as censoring rate by cohort.

**Q10. The reminder cohort's advantage disappears within the 30-day pack subgroup. Give three causes
and the check that separates them.**
Answer: A real effect concentrated in long packs; differential enrolment by pack; a fixed gap rule
penalising one pack.
Reasoning: The third cause is definitional and testable by recomputing with a pack-scaled grace
period; the first two are substantive and testable by balance and by enrolment-route tables.
Follow-up: If the effect is genuine and concentrated in 30-day packs, is that a better or worse
commercial result?
Red flag: Declaring the programme ineffective on the subgroup alone without checking the rule.
Scoring: 2 for three distinct causes, 2 for matched checks, 1 for the commercial interpretation.

**Q11. Why is a per-protocol analysis restricted to patients who received at least one reminder call
reported as secondary rather than primary?**
Answer: Receipt is a post-enrolment behaviour correlated with engagement, so conditioning on it
reintroduces the selection the cohort design tried to remove.
Reasoning: Patients who answer calls differ from those who do not in ways that also predict refilling;
the resulting contrast measures engagement, and the exclusion restriction that would license a
complier average effect is unverifiable here.
Follow-up: What would make the per-protocol analysis credible?
Red flag: Preferring it because the effect is larger.
Scoring: 2 for the selection mechanism, 2 for the exclusion-restriction point, 1 for a credible
condition.

**Q12. A coverage figure exceeds 1.00 for a group of patients in one pharmacy chain. Diagnose.**
Answer: Duplicate fill rows from a feed re-send, cash-pay fills arriving through a second channel, or
a days-supply field carrying pack size rather than dispensed quantity.
Reasoning: Each inflates the numerator differently — duplication doubles specific fills, channel mixing
adds fills never meant to be counted, and a wrong days-supply field scales every fill for that chain.
Follow-up: How would you prove duplication rather than genuine stockpiling?
Red flag: Capping at 1.00 and moving on, which destroys the diagnostic signal.
Scoring: 2 for three mechanisms, 2 for the distinguishing evidence, 1 for the anti-capping argument.

**Q13. State the four components of a persistency specification and why each must be published.**
Answer: Gap rule, observation window, denominator convention, censoring treatment.
Reasoning: Each is a choice that moves the number independently, and in this construction the
definitional spread (0.071) exceeds the effect being claimed (0.056), so an unpublished tuple makes
the figure unfalsifiable.
Follow-up: Which component would you expect a vendor deck to omit first, and why?
Red flag: Naming only the gap rule, which is the most visible but not the most consequential.
Scoring: 2 for all four, 2 for the magnitude argument, 1 for a credible omission prediction.

**Q14. The owner wants one number for the board. What do you provide, and what do you attach to it?**
Answer: The interval, or the lower bound converted into cost per additional persistent patient, with
the specification tuple and the censoring bracket attached.
Reasoning: A single mid-point invites a funding decision at the optimistic end of a four-fold cost
range, and the lower bound is the number the budget decision actually tests.
Follow-up: How would the presentation change for a medical reviewer rather than a finance committee?
Red flag: Providing the point estimate alone because the request asked for one number.
Scoring: 2 for the interval or bound, 2 for the attachments, 1 for audience adaptation.

**Q15. What would make you withdraw the programme's persistency claim entirely?**
Answer: If the specification cannot be locked, or if censoring proves strongly informative with a
bracket whose lower edge is indistinguishable from no benefit.
Reasoning: A claim without a locked specification is not reproducible, and a bracket that spans zero
benefit means the data cannot distinguish the programme from its own measurement assumptions.
Follow-up: What minimal evidence would restore a narrower claim?
Red flag: Keeping the claim because the point estimate is favourable, while the bracket is not.
Scoring: 2 for a concrete withdrawal condition, 2 for the reasoning, 1 for the restoration path.

## 15. Quick Reference Card

| Item | Value |
|---|---|
| Decision | Fund the structured reminder expansion, continue manual outreach, or defer with a trigger |
| Owner | Fictional patient support programme lead; medical, privacy and health-economics review required |
| Grain | One programme, one market, one twelve-month window; patients measured, not targeted |
| Locked specification | One-pack grace period, fixed 365-day window, product-limit estimation, censoring bracketed |
| Observed contrast | 5.59 pp (0.6152 vs 0.5592), 95% CI [1.40, 9.79] pp (ILLUSTRATIVE) |
| Cost view | 1,519.92 per additional persistent patient at the point estimate; 6,081.4 at the lower bound |
| Fragility | E-value 1.432 at the estimate, 1.185 at the interval bound |
| Definitional spread | 0.560 to 0.631 across four defensible specifications — larger than the effect |
| Censoring bracket | 0.325 to 0.595, with the estimate near the upper edge |
| Coverage pathology | Observed-span denominator lifts the cohort mean from 72.45% to 84.12% with unchanged fills |
| Sample requirement | 1,055 analysable per arm; 1,319 enrolled at 20% loss |
| Guardrails | Zero contacts to withdrawn patients, zero unapproved content, zero clinical advice, suppression before release |

Decision rule: fund only when the interval's lower bound, converted to cost per additional persistent
patient, sits below the owner's stated ceiling and the censoring bracket's lower edge still excludes no
benefit.

Reading order for a newcomer: §6.2 for the two ratios, §6.4 for the gap-rule swing, §6.5 for the
censored estimate, §6.10 for why the specification must be locked, then §12 for what breaks it.
