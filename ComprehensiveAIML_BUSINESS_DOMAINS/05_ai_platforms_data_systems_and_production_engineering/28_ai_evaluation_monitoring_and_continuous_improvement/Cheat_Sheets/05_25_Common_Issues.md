---
title: "25 Common Issues — AI Evaluation, Monitoring and Continuous Improvement"
topic: "Diagnostic catalogue for measurement, labelling, judging and monitoring failures"
collection: "ai-evaluation-monitoring"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — AI Evaluation, Monitoring and Continuous Improvement

> **Evidence declaration.** Every formula here is derived in place and every arithmetic result was
> recomputed at the precision printed. The *situations* are constructed teaching scenarios: a support
> assistant processing four thousand runs a week, a six-hundred-item human audit, a rubric-judged
> regression suite. No organisation, deployment, outage, or measured product result is described.
> Quantities labelled ILLUSTRATIVE exist so that a diagnostic has an expected magnitude to check
> against. None of them may be cited as an observed benchmark.

## 1. File Header

### 1.1 Scope of the catalogue

This file collects twenty-seven recurring defects in the machinery that decides whether an AI system is
working. The scope runs from the moment a run is recorded to the moment a promotion, rollback, or
retraining decision is taken on the basis of what was recorded. Inside that band sit the label join, the
metric definition, the interval, the sampling mechanism, the judge, the drift statistic, the alert
threshold, the review queue, and the regression gate.

Outside the band sit the model's architecture, the training algorithm, the retrieval index design, and
the product's pricing. Those have their own defects, documented elsewhere. The dividing line is
deliberate: a defect in this catalogue is one where the *claim* is wrong while the *system* may be
behaving exactly as designed. That is the class of defect that survives a green dashboard, a passing
test suite, and a confident reviewer.

### 1.2 Why evaluation defects are different

A serving defect announces itself. Latency rises, errors appear, a page fires. An evaluation defect is
silent by construction, because the instrument that would detect it is the instrument that is broken. A
service that computes precision over a biased label set will not report a problem with the label set; it
will report a precision. A regression suite whose judge drifted will not report judge drift; it will
report that the candidate improved.

This asymmetry has an operational consequence that shapes every entry below. Detection cannot come from
the metric itself, because the metric is the suspect. It must come from an *orthogonal* observation:
coverage next to precision, a re-score of frozen outputs, a hand reconstruction of one record, a second
instrument applied to the same items. Every discriminating check in section 8 was chosen for that
property — it observes something the broken instrument does not control.

### 1.3 The four defect families

| Category | Definition | Where it bites | Typical discovery |
|---|---|---|---|
| VIOLATED_ASSUMPTION | The computation silently depends on a precondition that did not hold | at promotion, months later | a live result never reproduces offline |
| NUMERICAL_EDGE | Arithmetic at a boundary produces a defined but meaningless value | at low counts, small slices | a claim about zero, or about a tiny difference |
| SCALE_FAILURE | A cost or rate grows faster than the design assumed | after a green pilot | backlog, cost overrun, skipped suite |
| CONCEPT_CONFUSION | Two distinct quantities are treated as interchangeable | in every document that cites the number | an argument between two correct people |

The families differ in how they evade review. A VIOLATED_ASSUMPTION is invisible to the person who
holds the assumption, because their tests encode it. A NUMERICAL_EDGE is invisible at typical values,
because the arithmetic is correct in the middle of the range. A SCALE_FAILURE is invisible at pilot
volume, because the non-linear term is small there. A CONCEPT_CONFUSION is invisible to anyone who
measures the wrong quantity precisely, because the measurement is precise.

### 1.4 The eight required fields

Every entry in section 8 carries the same eight fields, and the field definitions are stated once here
rather than repeated twenty-seven times.

**Identifier.** Stable, of the form `CI-nn`, and never renumbered. Later revisions may sharpen an entry
but cases and exercises cite these identifiers, so renumbering silently invalidates every citation.

**Category.** One of the four families above. The category is assigned from the mechanism, not from the
severity, and the same symptom can appear under two categories when two mechanisms produce it.

**Symptom.** What a person can observe without knowing the cause. Symptoms are deliberately coarse and
usually phrased as a mismatch between two artefacts: a dashboard and a reviewer's experience, a number
and its own past, a claim and a replay.

**Competing causes.** At least two mechanisms each consistent with the symptom. An entry with one cause
is an entry where the author stopped thinking at the first plausible story, and the first plausible story
is usually the second most likely.

**Discriminating check.** An observation whose result differs across the competing causes, runnable
before any correction. A check that requires the fix first is not a check; it is a confirmation.

**Correction.** The change that removes the mechanism. Corrections are stated as edits to the pipeline or
the contract, never as "investigate further" or "monitor closely".

**Prevention.** The change that makes the mechanism unable to recur. The distinction from correction
matters: a correction repairs one cohort, a prevention changes what a future cohort can contain.

**Verification.** The replay that demonstrates the correction works. Always two records at minimum — the
triggering record and a boundary record — because a fix validated only on the trigger has usually been
narrowed until it fits.

### 1.5 Severity and response windows

Severity is assigned from the decision the defect corrupts, not from how alarming the symptom looks.
A defect that makes a promotion decision wrong is severe even when the numbers are small, because the
promotion is not reversible for the users who receive the version.

| Level | Effect | Response window |
|---|---|---|
| S1 | Irreversible user effect, or an unauditable promotion | stop the path before further traffic |
| S2 | Wrong but reversible decision, recorded | correct within the current reporting window |
| S3 | Misleading number that a human still reviews | correct before the number is published |
| S4 | Rework with no decision effect | schedule |

The label is less important than the window. A defect labelled S2 that nobody corrects inside the window
has become S1 by neglect, and the neglect is usually caused by the defect being silent.

### 1.6 Conventions used in this file

Formulas appear in fenced `text` blocks with the anatomy FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE,
INTERPRETATION. Symbols carry units at first use. Proportions are dimensionless in `[0,1]`. Counts are
integers. Time is in days unless suffixed. Currency is a stylised unit written `cu`. Intervals are 95%
two-sided Wilson intervals unless another construction is named.

One rule governs every number: a derived figure is labelled with the status of its inputs. A number
computed from constructed inputs is `derived from illustrative`, never `measured`. The distinction is not
pedantry. A reader who copies "precision fell from 0.68 to 0.60" out of this file into a review document
has created a measurement that nobody performed, and the fault lies with the file if the label was
absent.

### 1.7 Relationship to the rest of the unit

The sibling sheet in this unit, `04_Model_Evaluation_and_Production_Monitoring.md`, establishes the
protocol: population, cut-off, outcome definition, maturity, comparator, decision rule. This file is the
inverse view. It assumes a protocol exists and catalogues the ways a protocol is quietly violated in
practice, so a reader who has worked through that sheet will recognise several entries as the failure
modes of its sections. CI-01, CI-05, CI-11 and CI-23 are protocol violations of its coverage and
interval material; CI-06, CI-15 and CI-22 are its operational arithmetic failing under load.

No entry here duplicates the sibling sheet. Where an overlap exists, this file states the failure and the
sibling sheet states the correct construction.

**Takeaway:** the catalogue exists because evaluation failures are silent; every entry is built around an
observation that the failing instrument does not control.

## 2. Concept Map / Mental Model

### 2.1 The evidence chain

Every monitoring claim is the end of a chain, and a defect anywhere upstream propagates through every
downstream ratio without announcement.

```text
 [1] run recorded          key, version, timestamp, inputs
        |
 [2] label attached        outcome at a horizon, adjudicated independently
        |
 [3] population selected   eligibility predicate, cut-off, exclusion list
        |
 [4] partition built       TP / FP / FN / TN / pending
        |
 [5] ratio computed        precision, recall, cost, calibration
        |
 [6] uncertainty stated    interval, denominator, coverage
        |
 [7] claim made            "candidate is better", "quality held", "drift detected"
        |
 [8] action taken          promote, roll back, retrain, adjust threshold
```

Each arrow is a place where the meaning changes and nothing warns. The chain is linear but the failures
are not: a defect at step 2 surfaces at step 5 as a plausible number, and a defect at step 6 surfaces at
step 7 as a confident sentence. The diagnostic implication is that a challenge to a claim must walk the
chain backwards, because the claim itself contains no trace of where it was corrupted.

### 2.2 The three clocks

Monitoring arithmetic is ambiguous until three clocks are named, and most disagreements about "the same
period" are disagreements about which clock.

```text
event clock      when the user's thing happened
                 |---- decision ---- horizon ---- outcome ----|
processing clock when the record arrived in the warehouse
                              |-- lag --|----- lag -----|
evaluation clock when the query ran and froze a snapshot
                                                  |
                                                  v
                                            as-of view
```

A window over the event clock and a window over the processing clock select different rows. A comparison
between two versions must hold the event clock fixed, or it compares traffic mixes rather than models.
A query that filters on the evaluation clock is not reproducible, because the same query run tomorrow
returns a different label set.

### 2.3 The instrument metaphor

Treat every measurement component as an instrument with a calibration state: the judge, the rubric, the
label pipeline, the sampler, the threshold, the dashboard query. An instrument is trustworthy only
within a stated calibration, and calibration is a claim that must be re-established after any change to
the instrument.

The metaphor earns its place because it predicts a specific failure that appears repeatedly below. When
an instrument changes and the measured value changes, the observer cannot tell whether the underlying
quantity moved or the instrument did. The only defence is a frozen reference set measured under both
instrument versions, which is exactly the structure of CI-08, CI-09 and CI-17.

### 2.4 The four denominators

Ninety percent of metric arguments are denominator arguments. Four denominators appear in this file and
they are not interchangeable.

| Denominator | Meaning | Used by | Misread as |
|---|---|---|---|
| All eligible decisions | the managed population | coverage, cost per decision | precision |
| Decisions with a usable label | the observable subset | precision among labelled | precision on the population |
| Predicted positives | the action set | precision | recall |
| Actual positives | the outcome set | recall | precision |

A rate whose denominator is unstated is not a rate. It is a fraction with a story attached, and the
story is supplied by whoever reads it next.

### 2.5 Triage order

The order below is chosen so that each step is cheap and each eliminates a layer before statistics.

1. **Count the rows at every stage.** Recorded runs, labelled runs, eligible runs. A mismatch localises
   the defect to the record or join layer without any arithmetic.
2. **Check monotonicity.** Label timestamp after decision timestamp, outcome after decision, maturity
   after horizon. A violation is a clock defect and stops the investigation.
3. **Reconstruct one record by hand.** Pick a run, recompute its contribution to the metric from raw
   fields. Agreement means the arithmetic is fine; disagreement localises a compute defect immediately.
4. **Compare two versions of the same view.** As-of versus current, judge-v1 versus judge-v2. A
   difference localises an instrument defect.
5. **Say the metric in words.** "Correct alerts over all alerts" versus "correct alerts over labelled
   alerts". If the sentence and the formula disagree, stop at interpretation.
6. **Check the actor and the policy version.** For any decision already taken, is there a named owner and
   a version? A blank localises an authority defect.

**Takeaway:** walk the chain backwards from the claim, name all three clocks, and identify the denominator
before defending a number.

## 3. Foundational Definitions

### 3.1 Measurement terms

| Term | Definition | Unit or type | The confusion it attracts |
|---|---|---|---|
| Run | One invocation of the system under evaluation, with a stable identifier | event | treated as a user or a session |
| Attempt | One try within a run, including retries | event | counted as a separate run |
| Completion | The run returned without an infrastructure or timeout error | boolean | treated as success |
| Rubric pass | The run's output satisfies an independently applied checklist | boolean | treated as ground truth |
| First-attempt success | The first attempt of a run passes the rubric | boolean | confused with eventual success |
| Eventual success | Any attempt of the run passes the rubric | boolean | reported as user experience |
| Judge | An instrument, human or model, that applies a rubric | versioned instrument | treated as an oracle |
| Rubric | The written criteria the judge applies | document, versioned | assumed stable across revisions |
| Audit sample | A stratified subset scored by a second instrument | sampling design | treated as the population |
| Coverage | Runs with a usable label over eligible runs | fraction | reported as a quality metric |
| Maturity | Whether enough time passed for the outcome to be knowable | per-cohort boolean | treated as complete labelling |
| Comparator | The baseline the candidate is measured against | version reference | left implicit, then varied |
| Effect size | The difference the decision cares about | metric-specific | confused with statistical significance |
| Guardrail metric | A measure that must not degrade regardless of the primary | metric with bound | monitored only after the fact |
| Error budget | Permitted bad events over a window at a stated target | count | read as a percentage |
| Burn rate | Budget consumed divided by budget for the same window | ratio | computed against the wrong target |
| Regression gate | A pre-declared rule blocking promotion on suite failure | policy | rewritten after seeing the candidate |
| Slice | A predefined subpopulation evaluated separately | subset | searched after the fact |
| Multiplicity | The number of comparisons made, declared or not | count | ignored when reading a p-value |
| Reference set | Frozen inputs with stored scores used to detect instrument drift | artefact | regenerated, destroying its purpose |
| Trace | The linked record of steps inside one run | event graph | treated as proof of correctness |
| Threshold | The score cut that converts a score into an action | policy parameter | treated as a model property |
| Prevalence | Outcome-positive share of the population | fraction | assumed stable across windows |

### 3.2 The status vocabulary

Five statuses attach to every printed number, and they are not decorative.

- **measured** — observed on a stated population under a stated protocol, with the observation stored.
- **derived** — computed arithmetically from stated inputs; any reader can reproduce it.
- **illustrative** — a constructed input chosen to make an arithmetic check possible.
- **heuristic** — a rule of thumb with no derivation; admissible for triage, not for a threshold.
- **not-measured** — named but unobserved.

The commonest violation in this file's own subject matter is a `derived` number quoted without its
`illustrative` input label. The arithmetic survives; the provenance does not. A number that has lost its
provenance is indistinguishable from a measurement to every subsequent reader, and that indistinguishability
is how a teaching example becomes a fabricated benchmark inside an organisation.

### 3.3 What counts as a label

A label must be able to disagree with the system's own action for reasons unrelated to the system's
policy. That single test excludes most convenient proxies.

| Candidate | Can disagree with the action independently? | Verdict |
|---|---|---|
| Analyst closed the alert as uninteresting | No: the analyst reacts to the alert | triage policy, not a label |
| Issuer confirmed fraud | Yes | usable label |
| User clicked the suggestion | Partly: exposure depends on the suggestion | engagement proxy, not correctness |
| Human rubric applied to a frozen output | Yes, if blind to version | usable label |
| Judge model applied to the same output | Yes, if calibrated against humans | instrument, not ground truth |
| Downstream conversion | Partly: confounded by the system's own choice | causal claim needs a design |
| Ticket reopened within seven days | Yes, if the reopen is not caused by the system's advice | usable with a stated horizon |

The third and sixth rows are the interesting ones: the observation is real, it is simply not an answer
to the question being asked. Those become CI-03 and CI-24.

### 3.4 The claim ladder

| Rung | Claim | Minimum evidence | Cannot support |
|---|---|---|---|
| 1 | The system responded | request success indicator | correctness |
| 2 | The system produced the intended output shape | schema and contract checks | quality |
| 3 | Outputs are associated with the intended outcome | labelled cohort, defined metric | causation |
| 4 | Changing the system changed the outcome | randomised or credible quasi-experiment | generalisation beyond the studied population |

Most review arguments are two people on different rungs, each correct. The sentence that is wrong is
always the one that omits its rung.

**Takeaway:** a label must be able to disagree with the action, a number must carry its status, and a
claim must name its rung.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

All monitoring claims begin as a proportion, and the proportion's only real content is the pair of sets
it divides.

```text
FORMULA:  rate = k / n
WHERE: k = events satisfying the numerator predicate [count]; n = events satisfying the denominator
       predicate [count]; both integers; the predicates are written down before computing.
CONSTRAINT: numerator events must be a subset of denominator events, otherwise the ratio can exceed 1
            and the predicates are inconsistent.
NUMERICAL EXAMPLE: 2,730 rubric passes over 4,000 eligible runs gives 2,730/4,000 = 0.68250.
INTERPRETATION: 31.75% of runs did not satisfy the pass rubric under this protocol.
```

The subset constraint is not a formality. A rate computed as "passes in the labelled set over all
eligible runs" has a numerator drawn from one population and a denominator from another, and its value is
a hybrid that answers no question. This is the arithmetic behind CI-04 and CI-18.

Interval estimation completes the base case, because a proportion from a sample has a sampling
distribution and near-boundary counts break the naive interval.

```text
FORMULA (Wilson):  centre = (p + z^2/(2n)) / (1 + z^2/n)
                   half   = z * sqrt( p(1-p)/n + z^2/(4n^2) ) / (1 + z^2/n)
WHERE: p = k/n [fraction]; n = denominator [count]; z = 1.959964 for a two-sided 95% interval.
CONSTRAINT: approximately independent draws from one population; clustering reduces effective n.
NUMERICAL EXAMPLE: k = 2,730, n = 4,000 gives p = 0.68250, half 0.01446,
                   interval [0.66804, 0.69696].
INTERPRETATION: the plausible range spans 2.9 percentage points, so a two-point "improvement" is
                not separated from sampling noise at this volume.
```

Derivation: the score statistic is approximately standard normal; inverting `|p_hat - p| <= z sqrt(p(1-p)/n)`
for `p` yields a quadratic whose two roots are the printed bounds. Inversion is what gives the interval
non-zero width at `p = 0`, which the Wald form does not.

### 4.2 General case

The general case is conditional: the rate of interest holds only inside a stratum, and the population
rate is an average of strata weighted by their share. When the sample's stratum shares differ from the
population's, the sample average is the wrong estimator.

```text
FORMULA (standardised rate):
    R = sum_s w_s * r_s      with  sum_s w_s = 1
WHERE: s indexes strata; w_s = population share of stratum s [fraction];
       r_s = observed rate within stratum s [fraction]; R = standardised rate [fraction].
CONSTRAINT: the same stratum definition must apply to both the weights and the rates; strata must be
            exhaustive and mutually exclusive.
NUMERICAL EXAMPLE: strata with population shares (0.30, 0.375, 0.325) and rates (0.34, 0.22, 0.11)
    give R = 0.30*0.34 + 0.375*0.22 + 0.325*0.11 = 0.10200 + 0.08250 + 0.03575 = 0.22025.
INTERPRETATION: the naive average of the three rates would be (0.34+0.22+0.11)/3 = 0.22333, so
                weighting moves the estimate by 0.3 points here — small, but the same mistake on a
                lopsided design moves it by several points.
```

The same identity in inverse-probability form repairs coverage bias, where each labelled observation
stands in for the unlabelled ones it resembles.

```text
FORMULA (inverse-probability rate):  R_w = sum_s (n_s r_s) / sum_s n_s
WHERE: n_s = eligible runs in stratum s [count]; r_s = observed rate among *labelled* runs in
       stratum s [fraction]. Expanding r_s = k_s / L_s with k_s labelled passes and L_s = n_s c_s
       labelled runs gives the equivalent form sum_s (k_s / c_s) / sum_s n_s:
       each labelled pass is weighted by the inverse of its selection probability.
CONSTRAINT: c_s > 0 in every stratum, and c_s must be known or estimable per stratum. The identity
            assumes labels are missing at random *within* stratum; it does not repair selection
            that operates on the outcome inside a stratum.
NUMERICAL EXAMPLE: see section 5.4, where the labelled-only rate is 0.24028 and this estimator,
    with the stratum rates applied to the full eligible counts, gives 0.22025.
INTERPRETATION: the two differ by 2.0 points, and the labelled-only figure is the pessimistic one
                because audits concentrate on the stratum where the system performs worst. The
                direction of the bias is a property of the design and must be derived, not assumed.
```

### 4.3 Variants

Five variants carry the rest of the catalogue: agreement, calibration, stability, k-of-n success, and
budget burn.

#### 4.3.1 Agreement corrected for chance

Raw agreement between two judges is inflated when one label dominates, so the corrected statistic is the
one that survives a change of class balance.

```text
FORMULA:  kappa = (p_o - p_e) / (1 - p_e)
WHERE: p_o = observed agreement [fraction]; p_e = agreement expected if both judges were independent
       with their own marginal rates = p_A*p_B + (1-p_A)*(1-p_B) [fraction].
CONSTRAINT: both judges classify every item into the same category set; p_e must be computed from the
            judges' own marginals, not from a reference distribution.
NUMERICAL EXAMPLE: p_o = 499/600 = 0.83167; judge A passes 0.680 of items, judge B passes 0.745, so
    p_e = 0.680*0.745 + 0.320*0.255 = 0.50660 + 0.08160 = 0.58820, and
    kappa = (0.83167 - 0.58820)/(1 - 0.58820) = 0.24347/0.41180 = 0.59123.
INTERPRETATION: 0.83 raw agreement is largely explained by both judges passing most items.
```

The paradox that makes this statistic necessary in review documents: with two judges who both pass 95%
of items and agree on 90%, kappa is `(0.90-0.905)/(1-0.905) = -0.0526` — negative, because agreement is
*below* what the marginals alone predict. Raw agreement would have called that 90% concordance.

#### 4.3.2 Expected calibration error

```text
FORMULA:  ECE = sum_b (n_b/n) * |o_b - pbar_b|
WHERE: b indexes probability bins; n_b = items in bin b [count]; n = total items [count];
       o_b = observed positive frequency in bin b [fraction]; pbar_b = mean predicted probability
       in bin b [fraction].
CONSTRAINT: every bin must contain enough items for o_b to be estimable; the binning must be reported
            with the value, because ECE is a function of the binning.
NUMERICAL EXAMPLE: five bins over 5,000 items give contributions
    0.36*0.02 + 0.28*0.01 + 0.18*0.02 + 0.12*0.03 + 0.06*0.02 = 0.01840.
INTERPRETATION: mean absolute calibration error of 1.84 percentage points under this binning.
```

Rebinning the same 5,000 items into two bins gives 0.00880 and into four bins gives 0.01280. A factor of
two in the reported statistic comes from the bin edges alone, which is the mechanism behind CI-09.

#### 4.3.3 Population stability index

```text
FORMULA:  PSI = sum_i (q_i - p_i) * ln(q_i/p_i)
WHERE: p_i = reference share of bin i [fraction]; q_i = current share of bin i [fraction];
       both distributions sum to one; every compared share strictly positive.
CONSTRAINT: identical bin edges, identical feature semantics, and a reference window that is itself
            representative; a zero share makes the term infinite.
NUMERICAL EXAMPLE: p = (0.45, 0.30, 0.15, 0.07, 0.03), q = (0.38, 0.31, 0.17, 0.09, 0.05) gives
    contributions (0.01184, 0.00033, 0.00250, 0.00503, 0.01022), total PSI = 0.02991.
INTERPRETATION: composition moved, mostly by a shift out of the largest bin; no harm is implied.
```

#### 4.3.4 k-of-n success

Reporting a success rate over attempts rather than over runs inflates the figure whenever retries occur.
The unbiased estimator for "at least one attempt passes" given `c` passes out of `n` attempts is
combinatorial, not `c/n`.

```text
FORMULA:  pass@k = 1 - C(n-c, k) / C(n, k)
WHERE: n = attempts for one task [count]; c = attempts that pass [count]; k = attempts allowed
       [count]; C is the binomial coefficient.
CONSTRAINT: attempts are exchangeable; selection of the k attempts is random, not chosen by a human
            looking at the outputs.
NUMERICAL EXAMPLE: n = 20, c = 6 gives pass@1 = 0.30000, pass@3 = 0.68070, pass@5 = 0.87087,
                  pass@10 = 0.99458.
INTERPRETATION: a 30% per-attempt rate becomes 87% within five attempts, which measures the retry
                budget as much as the capability.
```

#### 4.3.5 Budget burn

```text
FORMULA:  burn = F / (N_total * (1 - S))
WHERE: F = failures observed in the window [count]; N_total = eligible events in the *budget period*
       [count]; S = target [fraction].
CONSTRAINT: the numerator window and the budget period must be the same length, or the burn rate is
            a ratio of two different things and is unbounded in meaning.
NUMERICAL EXAMPLE: 600,000 requests per month, target S = 0.995 gives budget 3,000 bad events. A
    one-day window with 400 bad events out of 20,000 requests has burn 400/3,000 = 0.1333, but the
    comparable daily budget is 100, so the *pace* is 4.0x.
INTERPRETATION: 13% of the monthly budget consumed in 3% of the month is a fast burn, not a slow one.
```

### 4.4 Limits and counterexamples

**Limit 1: a zero count is not a zero rate.** Zero passes in 24 items is compatible with a 5% rate;
the exact one-sided 95% upper bound is `1 - 0.05^(1/24) = 0.1173`. For 60 items with zero failures the
bound is 0.0487, and for 600 items 0.00498. The practical reading is that a clean golden set of 24 items
supports a claim of "below about 12%", which is far weaker than the sentence "no failures observed"
implies.

**Limit 2: kappa is not comparable across different marginals.** Changing the class balance changes
`p_e` and therefore kappa, with no change in the judges' behaviour. Two teams reporting 0.59 and 0.76 may
be reporting the same discrimination under different base rates, which makes a kappa league table
meaningless (CI-17).

**Limit 3: ECE, PSI, and every binned statistic are functions of the bins.** Section 4.3.2 shows a
factor of two from rebinning 5,000 items; section 4.3.3 shows a tail bin contributing a third of a small
PSI. A statistic whose value changes materially under a defensible alternative binning is a statistic
that must be reported with its binning, and a threshold on it must be calibrated per implementation.

**Limit 4: the cost-optimal threshold requires calibrated probabilities.** With a false-positive cost of
1.2 cu and a false-negative cost of 9.5 cu, the rule `p* = C_FP/(C_FP + C_FN)` gives
`1.2/10.7 = 0.11215`. If the score is a raw logit or an uncalibrated rank, the number 0.112 does not
correspond to the probability 0.112, and the rule silently misplaces the threshold (CI-15).

**Limit 5: standardisation is only as good as the strata.** Weighting by population shares reproduces
the target population's composition, not its outcome relationships. If the outcome relationship differs
between the sample and the population inside a stratum, no weighting repairs the estimate, and the
residual error is an assumption rather than an interval.

**Takeaway:** each variant is valid inside a stated region; outside it the arithmetic still runs and the
number stops meaning anything.

## 5. Worked Numerical Example

All inputs in this section are ILLUSTRATIVE. Every result is computed from those inputs and can be
reproduced from the printed intermediate values.

### 5.1 The system and its protocol

A support assistant answers user questions with retrieved documentation and tool calls. The unit of
evaluation is a run. In one week the service records 4,000 eligible runs: distinct user sessions,
excluding internal test accounts, identified by run key, served by model version 11 and retrieval index
version 4.

Three outcome definitions coexist and each supports a different claim.

- **Completion:** the run returned without an infrastructure error or timeout. Fast, available at
  request time, silent about correctness.
- **Rubric pass:** an independently applied checklist — answer addresses the question, cites a source
  that supports the claim, performs no unsupported action — scored by a judge calibrated against humans.
- **First-attempt pass:** the rubric applied to the first attempt of the run, ignoring retries.

The audit population is a stratified sample of 600 runs per week, scored by trained reviewers blind to
model version. The judge scores all 4,000.

### 5.2 The four headline numbers

```text
completion             = 3,640 / 4,000 = 0.91000   Wilson [0.90073, 0.91848]
rubric pass per run    = 2,730 / 4,000 = 0.68250   Wilson [0.66804, 0.69696]
pass among completed   = 2,730 / 3,640 = 0.75000
first-attempt pass     = 2,100 / 4,000 = 0.52500   Wilson [0.50951, 0.54044]
retry uplift           = (2,730 - 2,100) / 2,100   = 0.30000, i.e. +30% relative
```

The headline 0.6825 and the first-attempt 0.5250 differ by 30% relative, and both are correct statements
about different quantities. The first answers "how often does the service eventually produce an
acceptable answer"; the second answers "how often is the first answer acceptable". A product review that
prints the first while discussing user experience has overstated experience by 15.75 percentage points
(CI-10).

### 5.3 The human audit and the judge

The 600-item stratified audit produced 408 passes, and the judge produced 447 passes on the same items.
Cross-tabulating the two instruments gives 377 items both pass, 70 judge-pass/human-fail, 31
human-pass/judge-fail, and 122 both fail.

```text
human rate = 408/600 = 0.68000   Wilson [0.64163, 0.71608]
judge rate = 447/600 = 0.74500   Wilson [0.70864, 0.77824]
difference = 0.06500 in favour of the judge
paired test on the 70 and 31 discordant items: McNemar chi-square = (70-31)^2/(70+31) = 1,521/101 = 15.06,
  so z = 3.881, which is far outside a two-sided 5% critical value of 1.96
naive independent-sample standard error = sqrt(0.68*0.32/600 + 0.745*0.255/600) = 0.02606, z = 2.49
```

The independent-sample form is wrong here because these are the same items, and it happens to be
conservative: it reports z = 2.49 where the paired test gives 3.88. The paired test uses the two
discordant counts, which is why a 6.5-point difference in totals can coexist with 83% item-level
agreement — most items agree, and the ones that disagree all lean the same way.

Agreement is 499 of 600 items, or 0.83167, and the chance-corrected value is 0.59123 from section
4.3.1. The interpretation that survives: the judge is a usable instrument with a known optimistic bias of
about 6.5 points, and any promotion decision that treats 0.745 as the pass rate is measuring the
instrument rather than the system.

### 5.4 Coverage bias across strata

Labelling is not uniform. The audit sample is drawn per stratum with different intensity, because review
capacity is allocated where the risk is highest.

| Stratum | Eligible runs | Labelling probability | Observed pass rate | Labelled passes |
|---|---:|---:|---:|---:|
| High-risk (actions taken) | 1,200 | 0.92 | 0.34 | 375.36 |
| Medium-risk (retrieval only) | 1,500 | 0.78 | 0.22 | 257.40 |
| Low-risk (informational) | 1,300 | 0.51 | 0.11 | 72.93 |

```text
labelled runs   = 1,200*0.92 + 1,500*0.78 + 1,300*0.51 = 1,104 + 1,170 + 663 = 2,937
labelled passes = 375.36 + 257.40 + 72.93 = 705.69
labelled-only rate        = 705.69 / 2,937 = 0.24028
MAR-standardised rate     = 0.30*0.34 + 0.375*0.22 + 0.325*0.11 = 0.10200 + 0.08250 + 0.03575 = 0.22025
all-unlabelled-fail bound = 705.69 / 4,000 = 0.17642
all-unlabelled-pass bound = (705.69 + 1,063) / 4,000 = 1,768.69/4,000 = 0.44217
labelled-only optimism    = 0.24028 - 0.22025 = 0.02003, about 2.0 percentage points
```

Three numbers, three meanings, and the whole point of the table is that they must not be confused.

The labelled-only rate 0.24028 divides by labelled runs, so it answers "among runs we audited, how many
passed". It is a correct answer to a narrow question.

The MAR-standardised rate 0.22025 assumes labels are missing at random *within stratum* — that the
unlabelled 8% of the high-risk stratum passes at the same 0.34 rate as the audited 92%. Under that
assumption the population rate is the observed stratum rates weighted by population shares, giving
0.22025. The gap of 2.0 points is entirely composition: the labelled set weights strata by
`n_s c_s / sum(n_s c_s)` = (0.3759, 0.3984, 0.2257), while the population weights are (0.30, 0.375,
0.325). The low-risk stratum, which has the highest pass rate, is under-weighted from 0.325 to 0.2257, so
the labelled-only figure is pulled *down* by about two points, not up.

That direction is worth stating plainly because it inverts the usual intuition. Selective labelling does
not always flatter a system. Here the audits concentrate on the high-risk stratum where the system
performs worst, so the audited set is harder than the population and the labelled-only rate is
*pessimistic*. A team that corrected in the intuitive direction — assuming labelled rates overstate
quality — would move this number the wrong way.

The two bounds bracket both estimates. If every unlabelled run failed, the population rate is 0.17642; if
every unlabelled run passed, it is 0.44217. Both estimates sit inside the bracket, and the bracket is
0.26575 wide — wider than every effect this gate can detect. That width, not the choice between 0.24028
and 0.22025, is the finding.

The lesson is that a weighting scheme requires its assumption to be stated. MAR within stratum gives
0.22025; a different missingness model — say, that unlabelled low-risk runs are those the system handled
so confidently that nobody checked, and therefore pass at 0.30 rather than 0.11 — gives a different
number again. The arithmetic is exact in each case; the assumption is doing the work.

### 5.5 Calibration of a confidence score

The assistant emits a confidence score with each answer. On 5,000 runs with a rubric label, binned by
predicted confidence:

| Bin centre | Items | Passes | Observed rate | Absolute gap | Weight |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 1,800 | 18 | 0.010 | 0.020 | 0.36 |
| 0.12 | 1,400 | 154 | 0.110 | 0.010 | 0.28 |
| 0.30 | 900 | 288 | 0.320 | 0.020 | 0.18 |
| 0.52 | 600 | 330 | 0.550 | 0.030 | 0.12 |
| 0.74 | 300 | 216 | 0.720 | 0.020 | 0.06 |

```text
ECE = 0.36*0.020 + 0.28*0.010 + 0.18*0.020 + 0.12*0.030 + 0.06*0.020
    = 0.00720 + 0.00280 + 0.00360 + 0.00360 + 0.00120 = 0.01840
```

The score is slightly over-confident in the top bin (0.720 observed against 0.740 predicted) and
under-confident in the bottom bin (0.010 against 0.030), which is the normal shape for a shrunk score.
Rebinning to two bins gives 0.00880; to four bins, 0.01280. The spread across defensible binnings is
0.0096, more than half the headline value, and that is the whole of CI-09.

### 5.6 The regression gate

The promotion gate has two components: a rubric-judged suite of 4,000 items and a frozen golden set of
240 items with 5 known critical failures.

```text
golden critical rate      = 5/240 = 0.02083
upper bound if 0/240      = 1 - 0.05^(1/240) = 0.01239
expected criticals at 0.01 rate over 240 items = 2.4; P(observe 0) = 0.99^240 = 0.08963
power to detect 0.81 -> 0.75 at 4,000-item suite = the suite is 4,000, needed 747 at 80% power
power to detect 0.81 -> 0.78 at the same suite = needed 2,841 at 80% power
```

Two facts follow. First, the golden set cannot detect a change in rare critical failures: with 240 items
and a true rate of 1%, the probability of observing zero is 0.0896, so "zero criticals" is an ordinary
outcome and not evidence of safety. Second, the 4,000-item suite is generously powered for a six-point
regression (needs 747) and adequately powered for a three-point one (needs 2,841), so the suite size is
not the constraint on detecting regressions — the judge's own variance is.

### 5.7 Re-checks

```text
CHECK 1 (partition sums):
  passes 2,730 + failures 1,270 = 4,000 ; completion 3,640 + non-completion 360 = 4,000
  first-attempt passes 2,100 <= eventual passes 2,730, as required by nesting.

CHECK 2 (coverage from components):
  1,104 + 1,170 + 663 = 2,937 labelled ; 4,000 - 2,937 = 1,063 unlabelled
  2,937/4,000 = 0.73425 coverage ; 1,063/4,000 = 0.26575 ; sum = 1.00000

CHECK 3 (cost of measurement):
  600 audits * 4.5 cu = 2,700 cu/week
  4,000 runs * 0.0008 cu (judge) = 3.2 cu/week
  judge is 0.12% of the audit cost and cannot replace it, only rank items for it.
```

CHECK 3 is the one that changes planning. The temptation after reading the cost line is to replace the
audit with the judge; the arithmetic says the judge costs a thousandth as much and carries a 6.5-point
bias against a human instrument whose own interval is ±3.7 points. The judge is a pre-filter, and the
audit is the instrument.

### 5.8 Sensitivity

```text
SENSITIVITY A (coverage assumption):
  all unlabelled fail        -> 0.17642
  MAR-standardised estimate  -> 0.22025
  labelled-only, as reported -> 0.24028
  all unlabelled pass        -> 0.44217
  the extreme-bracket spread is 0.26575, which exceeds every effect size this gate can detect;
  the choice between the two interior estimates is worth only 0.02003.

SENSITIVITY B (judge bias):
  if the judge is +6.5 points optimistic, a candidate measured at 0.745 is at 0.680 by the
  human instrument — exactly the incumbent's rate, so the apparent gain may be instrument noise.

SENSITIVITY C (binning):
  ECE 0.01840 (5 bins), 0.01280 (4 bins), 0.00880 (2 bins): the threshold must be fixed with the binning.
```

Sensitivity A is the uncomfortable one, and the discomfort is about which number to trust. Two defensible
interior estimates sit two points apart, and the extreme bracket is twenty-six points wide. Neither gap
can support a two-point promotion decision, however precise the interval on the labelled subset is. The
correct response is not a tighter interval; it is more labels, because no estimator can recover the
missing 1,063 outcomes from the 2,937 that were observed.

**Takeaway:** the four headline numbers are internally consistent, the audit exposes the judge's bias,
and the coverage spread is larger than any effect the gate can resolve — which is the finding, not a
caveat.

## 6. Visual / Geometric Intuition

### 6.1 The chain with its failure points

```text
run recorded ---> label attached ---> population ---> partition ---> ratio ---> claim ---> action
     |                  |                  |              |            |          |         |
   CI-26             CI-01             CI-04          CI-02        CI-25      CI-07    CI-26
   no actor          as-of join        coverage       zero denom   ratio avg  drift    no actor
   CI-19             CI-03             CI-18          CI-05        CI-09      CI-11
   refset contam.    action-as-label   labelled prev  Wald         binning    prevalence
                     CI-23
                     alerted-only
```

The diagram's purpose is locational. A symptom that appears as a ratio defect can originate three columns
to the left, and the repair belongs where the defect is, not where it was noticed.

### 6.2 Coverage as an interval, not a number

```text
0.0      0.176      0.220  0.240          0.442              1.0
 |----------|---------|------|--------------|
 all-fail  MAR       labelled-only       all-pass
           estimate  as reported

the extreme bracket is 0.266 wide; the two interior estimates differ by 0.020
the bracket is what the data cannot exclude, the point estimates are what the assumption adds
```

Reading the figure: the four marks are not four estimates of equal standing. The two interior marks are
estimates under two stated selection models; the outer two are bounds under two extreme missing-data
assumptions. A document that prints only one interior mark has replaced an interval with an assumption,
and a document that prints neither bound has hidden the size of what it does not know.

### 6.3 The alert-rate band and its width

```text
alerts/day
 110 |----------------------------------------- 3-sigma upper at n=571
     |
  68 |========== mean = 0.6825 * 571 = 39.0/day (pass-rate view) ==========
     |
  50 |----------------------------------------- 3-sigma lower at n=571
     +------|------|------|------|------|------> day
           Mon    Tue    Wed    Thu    Fri
```

At 571 runs per day the standard error of a 0.6825 rate is
`sqrt(0.6825*0.3175/571) = 0.01948`, so the 3-sigma band is `[0.6241, 0.7409]`. A day at 0.60 is outside
and worth a look; a day at 0.65 is inside and is not evidence of anything. The same computation at the
weekly volume gives `[0.6604, 0.7046]`, so the weekly figure is five times more informative than the daily
one — which is why daily alerting on this metric generates more noise than signal (CI-06, CI-12).

### 6.4 Geometry of the ratio-averaging error

```text
Stratum A:  100 runs, 60 pass  ->  0.60
Stratum B:  300 runs, 135 pass ->  0.45

pooled      = (60 + 135) / 400 = 195/400 = 0.4875
mean of two = (0.60 + 0.45) / 2 = 0.5250

the mean-of-ratios answer weights a 100-run stratum equally with a 300-run stratum
```

The gap is 3.75 points, larger than most effects under discussion, and it appears whenever a dashboard
averages per-slice rates instead of summing counts. The geometric statement is that averaging ratios
averages the *areas* as if they were equal, when the areas are the denominators.

### 6.5 The judge and the human on the same axis

```text
item-level agreement      |============================== 499/600 = 0.832
chance-expected agreement |==================== 0.588
                          |          |
                          |          +-- kappa = 0.591
                          +------------- observed
```

The distance between the two bars is the part of agreement not explained by the marginals, and it is what
a promotion decision should use. The distance is also the part that changes when the class balance
changes, which is why the same two judges can produce 0.59 and 0.76 (CI-17).

**Takeaway:** locate the defect on the chain, draw coverage as an interval, and never average ratios whose
denominators differ.

## 7. Comparison / Selection Table

### 7.1 Which instrument answers which question

| Question | Instrument | Cost per item | Blind spot |
|---|---|---|---|
| Did the run complete? | request telemetry | negligible | correctness |
| Was the output well-formed? | schema and contract check | negligible | content quality |
| Is the answer acceptable? | rubric judge | 0.0008 cu | judge bias, drift |
| Is the answer acceptable to a human? | trained reviewer | 4.5 cu | reviewer variance |
| Is a rare harmful output present? | targeted adversarial set | moderate | fixed attack surface |
| Did quality change this week? | labelled cohort trend | label cost | maturity |
| Did the instrument change? | frozen reference re-score | low | reference staleness |
| Did prevalence change? | outcome counts | negligible | nothing about ranking |
| Did the input distribution move? | PSI on features | negligible | harm |

### 7.2 Which interval at which count

| Situation | Construction | Why not Wald |
|---|---|---|
| n large, p near 0.5 | Wilson or Wald agree | either is fine |
| p near 0 | Wilson | Wald collapses to zero width |
| zero successes in n | Wilson or exact one-sided | Wald gives `[0,0]` |
| clustered items | design effect, or cluster bootstrap | independence violated |
| stratified sample | standardise, then interval per stratum | pooled interval misstates variance |
| ratio of two counts | delta method or bootstrap | interval on numerator is not interval on ratio |

### 7.3 Which response to which diagnosis

| Diagnosis | Response | Reversible | Never do |
|---|---|---|---|
| Label join uses current state | rebuild as-of view | yes | compare old and new metrics |
| Coverage too low for the claim | buy labels or publish bounds | yes | reweight and call it measured |
| Judge drifted | pin version, re-score reference set | yes | compare across judge versions |
| Prevalence shifted | recompute expected precision | yes | retrain for a balance change |
| Capacity exceeded | cut alert volume or staff up | yes | let the backlog label the model |
| Threshold stale after recalibration | re-derive `p*` from costs | yes | keep the numeric threshold |
| Gate too slow for release cadence | split into fast and full suites | yes | skip the full suite |

**Takeaway:** select the instrument by the question, the interval by the count, and the response by the
diagnosed mechanism.

## 8. Failure Modes & Edge Cases

### 8.0 Catalogue index

The twenty-seven entries below are ordered by position in the evidence chain, from record integrity
through claim authority. Identifiers are stable; a later revision may sharpen wording but must not
renumber.

| ID | Category | Symptom in one line |
|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | Retrospective lift that live traffic never reproduces |
| CI-02 | NUMERICAL_EDGE | A metric over an empty window printed as zero |
| CI-03 | CONCEPT_CONFUSION | The system's action used as its own label |
| CI-04 | NUMERICAL_EDGE | Unlabelled runs counted as failures in the denominator |
| CI-05 | NUMERICAL_EDGE | Zero-width interval from a zero-count cell |
| CI-06 | SCALE_FAILURE | Alert volume exceeds review capacity and the backlog flatters precision |
| CI-07 | CONCEPT_CONFUSION | Score drift reported as quality drift |
| CI-08 | VIOLATED_ASSUMPTION | Judge version changed between the two arms |
| CI-09 | NUMERICAL_EDGE | Calibration error changes by half under a different binning |
| CI-10 | CONCEPT_CONFUSION | Eventual success reported as first-attempt experience |
| CI-11 | VIOLATED_ASSUMPTION | Prevalence shift attributed to model degradation |
| CI-12 | SCALE_FAILURE | Per-slice monitors multiply false alarms past the response budget |
| CI-13 | NUMERICAL_EDGE | Stability index dominated by one sparse bin |
| CI-14 | CONCEPT_CONFUSION | Average precision compared across populations with different prevalence |
| CI-15 | VIOLATED_ASSUMPTION | Threshold retained after the score was recalibrated |
| CI-16 | SCALE_FAILURE | Trace retention cost grows with retries, not users |
| CI-17 | CONCEPT_CONFUSION | Raw judge agreement quoted as reliability |
| CI-18 | NUMERICAL_EDGE | Prevalence estimated from the labelled subset |
| CI-19 | VIOLATED_ASSUMPTION | Reference set contaminated by the candidate it tests |
| CI-20 | SCALE_FAILURE | Regression suite exceeds the release window and gets skipped |
| CI-21 | CONCEPT_CONFUSION | Error budget read as a percentage target |
| CI-22 | NUMERICAL_EDGE | Burn rate computed across mismatched windows |
| CI-23 | VIOLATED_ASSUMPTION | Recall estimated by sampling only alerted cases |
| CI-24 | CONCEPT_CONFUSION | Engagement signal read as correctness |
| CI-25 | NUMERICAL_EDGE | Mean of per-slice rates used as the pooled rate |
| CI-26 | VIOLATED_ASSUMPTION | No version or actor recorded, so rollback is undecidable |
| CI-27 | SCALE_FAILURE | Judge throughput and cost measured only on the pilot volume |

Two conventions hold for every detailed entry. The discriminating check is written so that the competing
causes produce *different observations*; where no such check exists the entry says so and recommends
escalation instead of a check. And every verification names a boundary item alongside the triggering item,
because a correction validated only on the trigger has usually been narrowed to fit it.

### 8.1 CI-01 — a decision-time field was joined at its latest value

| ID | Category | Symptom |
|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | An offline evaluation shows a gain that live traffic never reproduces |

**Competing causes.** (a) An outcome-derived field entered the feature set through a join that ignores
time. (b) The gain was real but decayed because the evaluation cohort was older than live traffic.
(c) The live path runs a different configuration from the replay.

**Discriminating check.** For every feature, compare its write timestamp against the decision timestamp;
any field written after the decision is leak. Then compare configuration hashes between the replay and
the serving path.

**Correction.** Remove the offending fields and rebuild the decision-time feature record from the
snapshot taken at scoring time.

**Prevention.** Fail the feature-set build when any field's write time exceeds the decision time. Fail,
not warn: a warning produces a comment, a failure produces a removal.

**Verification.** Replay one run whose features were all written before the decision, and one whose
last feature was written seconds before it. The second is where an as-of join error becomes visible.

The mechanism is a join that does not know about time. The most predictive table available to an analyst
is almost always the one written latest, because it contains more of the outcome. Reaching for it is
rational, and it converts the evaluation into a measurement of the pipeline's hindsight.

Cause (b) deserves separate mention because it produces an identical report with no leakage at all. The
discriminator is to re-run the same comparison on the oldest slice of the evaluation window: a real gain
that decayed is largest there, while a leak is uniform across the window. Cause (c) is rarer and is
excluded by a configuration hash comparison before any modelling discussion begins.

### 8.2 CI-02 — a rate over an empty window is printed as zero

| ID | Category | Symptom |
|---|---|---|
| CI-02 | NUMERICAL_EDGE | A quiet period reports a perfect score and no alert fires |

**Competing causes.** (a) A zero denominator coerced to zero. (b) A counter that was never initialised
for the window. (c) Errors genuinely absent because no traffic arrived.

**Discriminating check.** Print numerator, denominator, and the counter's own emission timestamp
separately. No emission means (b); an emission with numerator zero and denominator zero means (a); an
emission with a non-zero denominator means (c).

**Correction.** Represent the undefined case explicitly and exclude it from aggregates rather than
filling it.

**Prevention.** Make the rate type carry an undefined variant so arithmetic on it cannot compile without
a case split.

**Verification.** Replay an empty window, a one-event window, and a normal window. The one-event window
is the important one: it confirms the rate is computed rather than short-circuited.

The alerting consequence is worse than the arithmetic. An alert written `rate > 0.05` is false both for a
genuine zero and for an undefined value coerced to zero, so it is silent in both cases. The distinction
between "nothing happened" and "nothing was measured" disappears exactly when it matters most — during a
partial outage, when the service that measures failures may itself be down.

Filling undefined with the previous window's value is a third defect rather than a repair: it invents an
observation, and the invention is indistinguishable from data in every downstream query.

### 8.3 CI-03 — the system's action used as its own label

| ID | Category | Symptom |
|---|---|---|
| CI-03 | CONCEPT_CONFUSION | Precision looks healthy while users report irrelevant answers |

**Competing causes.** (a) The label is derived from the action (reviewer closure, user click), so it
cannot disagree with the action. (b) The label is genuine but defined over a horizon that truncates the
relevant failures. (c) The population excludes the users who complain.

**Discriminating check.** Correlate the label with the action at item level. A correlation near one
identifies (a) immediately. Then recompute on a holdout where the action was suppressed for a random
subset — a small deliberate sample where no suggestion was shown.

**Correction.** Introduce a blind rubric applied to frozen outputs, independent of what the system did.

**Prevention.** Reject at pipeline level any label whose generating event is caused by the prediction.

**Verification.** Score a set of outputs where the system abstained, and confirm the rubric produces a
mix of passes and failures. A label that cannot fail on abstentions is an action in disguise.

The intuition trap is strong here because the action is recorded at decision time and needs no waiting.
An analyst who measures "false positive rate" as the share of alerts closed as uninteresting has measured
the triage policy against the alerting policy. That is a useful operational quantity; it is not model
correctness, and it moves whenever analysts are told to clear queues faster.

### 8.4 CI-04 — unlabelled runs counted as failures

| ID | Category | Symptom |
|---|---|---|
| CI-04 | NUMERICAL_EDGE | Coverage is high, yet the reported rate is far below the labelled rate |

**Competing causes.** (a) The denominator is all eligible runs while the numerator is labelled passes.
(b) The labelled set genuinely has a higher pass rate because labelling favours successes. (c) A join
dropped a stratum from the numerator but not the denominator.

**Discriminating check.** Recompute with the labelled count as the denominator. If the two figures differ
by roughly the coverage factor, (a) is confirmed arithmetically. Then compare per-stratum labelled counts
against eligible counts to test (c).

**Correction.** Either report the labelled rate with coverage, or weight by inverse labelling
probability.

**Prevention.** Make the metric definition carry its numerator population and denominator population as
separate required fields.

**Verification.** Recompute the metric from raw counts for a stratum with 100% coverage and confirm the
two definitions agree there, which they must.

Section 5.4 is this defect worked in full. The labelled-only rate is 705.69/2,937 = 0.24028. Dividing
by the eligible 4,000 instead gives 705.69/4,000 = 0.17642, which is not a population rate at all: it is
a rate computed over a population in which 1,063 runs were silently assigned to the failure class. The
difference between the two figures, 6.4 percentage points, is the size of that silent assignment, and
neither number is the population rate — the honest bracket is [0.17642, 0.44217].

### 8.5 CI-05 — zero width from a zero-count cell

| ID | Category | Symptom |
|---|---|---|
| CI-05 | NUMERICAL_EDGE | A safety check reports "0% failure, interval [0, 0]" |

**Competing causes.** (a) Wald interval applied at `k = 0`. (b) The check genuinely has no failures and
the interval is honest. (c) The check never ran and reported its default.

**Discriminating check.** Inspect the number of executed checks against the planned count. A executed
count of zero identifies (c). For (a) versus (b), compute the exact one-sided bound: `1 - 0.05^(1/n)`.

**Correction.** Replace the interval with an exact or Wilson construction and print n.

**Prevention.** Ban zero-width intervals in the reporting layer by construction.

**Verification.** Replay a cell with zero failures out of 24 and one with zero out of 600; the bounds
must differ by an order of magnitude (0.1173 against 0.00498).

The practical harm is decision-theoretic. A reviewer told "0% failures, 240 items" cannot distinguish
"below 1.2%" from "below 12%" without the denominator, and 12% is often unacceptable while 1.2% is fine.
Two adjacent decisions get the same input and deserve different answers.

### 8.6 CI-06 — the backlog flatters precision

| ID | Category | Symptom |
|---|---|---|
| CI-06 | SCALE_FAILURE | Precision improves while queue age and unreviewed volume rise |

**Competing causes.** (a) Reviews are being skipped, and skipped items are the ones analysts judged least
interesting, so the reviewed subset is enriched for successes. (b) The model genuinely improved.
(c) The threshold moved, changing the alert mix.

**Discriminating check.** Compute precision by review latency bucket. Under (a) precision is highest for
the most-delayed reviews and lowest for the immediate ones. Under (b) it is flat across latency. Check
the threshold version for (c).

**Correction.** Cap the alert rate at capacity, or add reviewers, and treat unreviewed items as pending
rather than absent.

**Prevention.** Make coverage a required companion of every precision figure, and alert on queue age.

**Verification.** Replay a week where the queue was drained and confirm precision returns to the
pre-backlog value.

The mechanism is selection on the variable being measured, induced by a queue. It is the one failure in
this catalogue that makes the number move in the *opposite* direction from the truth, and it is therefore
the one most likely to be reported as good news.

### 8.7 CI-07 — score drift reported as quality drift

| ID | Category | Symptom |
|---|---|---|
| CI-07 | CONCEPT_CONFUSION | "Model drift detected" appears in a weekly report with no quality number |

**Competing causes.** (a) The input population changed and the model is correct on the new population.
(b) A unit or encoding change corrupted a feature. (c) The outcome prevalence changed, moving the score
mix. (d) The relationship between inputs and outcome changed.

**Discriminating check.** Compute the metric separately by score band. A uniform shift with stable
within-band performance is (a) or (c); a within-band degradation is (d); a discontinuity at a specific
feature value with correct values elsewhere is (b).

**Correction.** For (a) or (c), recompute expectations under the new mix; for (b), repair and replay; for
(d), evaluate a candidate model.

**Prevention.** Require every drift alert to name which of the four mechanisms was excluded and with what
observation.

**Verification.** Replay the alerting period with the reference population resampled to the new mix; if
the statistic falls silent, the alert was a population effect.

Only one of the four causes justifies retraining, and retraining on corrupted inputs — cause (b) — makes
the degradation worse in a way that is invisible until the corruption is repaired, at which point the
model has learned a false relationship and must be rebuilt.

### 8.8 CI-08 — judge version changed between arms

| ID | Category | Symptom |
|---|---|---|
| CI-08 | VIOLATED_ASSUMPTION | Candidate beats incumbent on every rubric dimension by a similar margin |

**Competing causes.** (a) The judge was upgraded, or its prompt edited, between scoring the two arms.
(b) Sampling temperature or decoding parameters differ. (c) The candidate is genuinely better uniformly.

**Discriminating check.** Re-score 300 frozen outputs from a common reference set with the current judge
and compare against their stored scores. Any change identifies (a) or (b) before the candidate is
considered. A uniform-margin pattern across dimensions is itself weak evidence for (a).

**Correction.** Re-score both arms under a single pinned judge version.

**Prevention.** Log judge model identifier, prompt hash, and decoding parameters with every score, and
re-score the reference set on every judge change.

**Verification.** Run the same arm twice under identical judge conditions and confirm the scores agree
within the judge's test-retest variation measured on the reference set.

A uniform margin across unrelated rubric dimensions is the tell. A genuine capability change is rarely
uniform; an instrument change is uniform by construction because it shifts every score the same way.

### 8.9 CI-09 — calibration error changes under rebinning

| ID | Category | Symptom |
|---|---|---|
| CI-09 | NUMERICAL_EDGE | Two teams report 0.019 and 0.009 for the same model and dataset |

**Competing causes.** (a) Different bin edges or bin counts. (b) Different item sets. (c) One team
weighted by bin count and the other by equal weight per bin.

**Discriminating check.** Exchange bin definitions and recompute. If the two figures swap, (a) is
confirmed. Then compare item counts; a difference identifies (b).

**Correction.** Publish the binning with the value, and standardise on equal-count bins with a stated
minimum per bin.

**Prevention.** Store the bin edges and the item count alongside every calibration figure.

**Verification.** Recompute the published figure from stored edges and confirm exact agreement, then
recompute under the alternative binning and confirm the reported spread matches the stated sensitivity.

Section 5.5 shows the magnitude: 0.01840 with five bins, 0.01280 with four, 0.00880 with two. Any
threshold on "ECE below 0.02" is therefore a threshold on a binning choice unless the binning is fixed.

### 8.10 CI-10 — eventual success reported as experience

| ID | Category | Symptom |
|---|---|---|
| CI-10 | CONCEPT_CONFUSION | Product metrics look strong while users report frequent bad first answers |

**Competing causes.** (a) The headline rate counts any attempt, so retries inflate it. (b) Users do not
retry, so the retry-inclusive rate describes a path few take. (c) The rubric is satisfied by a retry that
the user never saw.

**Discriminating check.** Compute first-attempt and any-attempt rates on the same runs, plus the retry
distribution per user. A large gap with a low per-user retry count identifies (a) and (b) together.

**Correction.** Report both, and attach the retry rate that links them.

**Prevention.** Make attempt index a required dimension of every success metric.

**Verification.** Replay a day and confirm the first-attempt rate on replayed data matches the reported
one, then confirm the eventual rate exceeds it by the expected retry uplift.

In the worked example the gap is 0.68250 against 0.52500, a 30% relative uplift produced entirely by the
retry budget. Cost per success inherits the same distortion: `1680/2730 = 0.61538` cu per eventual
success against `1680/2100 = 0.80000` cu per first-attempt success.

### 8.11 CI-11 — prevalence shift attributed to the model

| ID | Category | Symptom |
|---|---|---|
| CI-11 | VIOLATED_ASSUMPTION | Precision falls 8 points in three weeks with no version change |

**Competing causes.** (a) Prevalence fell, which lowers positive predictive value at constant
sensitivity and specificity. (b) A slice's share of traffic grew. (c) A feature degraded.

**Discriminating check.** Recompute recall and false-positive rate. If both are stable and only
prevalence moved, the precision change is arithmetically predicted and requires no model explanation.

**Correction.** Restate targets in terms of recall and false-positive rate, which are prevalence-free.

**Prevention.** Derive precision from its two inputs rather than monitoring it as a primitive.

**Verification.** Back-test the historical series: reconstruct precision from recall, false-positive rate,
and observed prevalence, and confirm the reconstruction matches the dashboard throughout.

The identity is worth writing in full because it settles the argument in one line:
`PPV = se*pi / (se*pi + (1-sp)(1-pi))`. With `se` and `sp` fixed, PPV is a monotone function of `pi`
alone, so a precision drop with stable `se` and `sp` is a prevalence event by definition.

### 8.12 CI-12 — slice monitors multiply false alarms

| ID | Category | Symptom |
|---|---|---|
| CI-12 | SCALE_FAILURE | Two or three slice alerts fire every day and none is actionable |

**Competing causes.** (a) Forty independent 3-sigma tests per day produce about 0.11 expected false
alarms each, or roughly one every nine days per test. (b) The slices are correlated, so the effective
number of tests is smaller. (c) A real defect affects one slice persistently.

**Discriminating check.** Compute the number of tests and the family-wise error rate: `1-(1-0.0027)^40`.
Then check persistence — a real defect recurs in the same slice, a false alarm wanders.

**Correction.** Predeclare the slice set, allocate alpha across it, and require persistence before
escalation.

**Prevention.** Report a multiplicity-adjusted threshold with every slice table.

**Verification.** Replay a month of history and confirm the alert count under the adjusted procedure
falls near its nominal expectation.

The arithmetic: with 24 hourly looks at 3-sigma, the probability of at least one false alarm is
`1-(1-0.0027)^24 = 0.0628`. At 720 looks it is `0.8572`. A monitoring system that inspects more often
than it corrects will spend its credibility on noise, and the response to noise is eventually to ignore
alerts entirely — which is how a real defect goes unnoticed.

### 8.13 CI-13 — one sparse bin dominates the stability index

| ID | Category | Symptom |
|---|---|---|
| CI-13 | NUMERICAL_EDGE | Stability index crosses a warning line overnight on an unchanged model |

**Competing causes.** (a) A tail bin's share moved, and log-weighted contributions amplify small tail
changes. (b) The reference window or bin edges changed with a dashboard release. (c) A genuine
population shift.

**Discriminating check.** Print per-bin contributions. In the section 4.3.3 example, the two outer bins
contribute 0.01184 and 0.01022 of a 0.02991 total, or 74% between them. Then diff the dashboard's stored
edges and reference dates.

**Correction.** Use quantile bins with a minimum count, and report the tail's contribution share.

**Prevention.** Version bin edges and reference windows as artefacts and refuse cross-version comparisons.

**Verification.** Recompute from raw data for the alert night and a quiet night, and confirm the quiet
night reproduces its stored value exactly.

### 8.14 CI-14 — average precision compared across prevalence

| ID | Category | Symptom |
|---|---|---|
| CI-14 | CONCEPT_CONFUSION | A model looks much worse in a new market on the same metric |

**Competing causes.** (a) Prevalence in the new market is lower, which depresses every precision term.
(b) The model genuinely discriminates worse. (c) The label definition differs between markets.

**Discriminating check.** Compare the ROC curves, which are prevalence-free, and the precision-recall
curves at matched prevalence by resampling. Curves that coincide after resampling identify (a).

**Correction.** Report AUC or a prevalence-standardised AP, and state the prevalence with any AP figure.

**Prevention.** Store prevalence with every AP value in the results table.

**Verification.** Resample one market's negatives to the other's prevalence and confirm the recomputed AP
approaches the reported difference; the residual is the genuine gap.

### 8.15 CI-15 — threshold retained after recalibration

| ID | Category | Symptom |
|---|---|---|
| CI-15 | VIOLATED_ASSUMPTION | After a calibration fix, action volume changes by 30% with no policy decision |

**Competing causes.** (a) Recalibration changed the mapping from score to probability, so the old numeric
threshold selects a different set. (b) Traffic changed. (c) The threshold was altered in configuration
without a change record.

**Discriminating check.** Compare the empirical positive rate among items scoring near the threshold
before and after. If the rate moved from, say, 0.30 to 0.20 while the threshold stayed 0.16, (a) is
confirmed. Check configuration history for (c).

**Correction.** Re-derive the threshold from `p* = C_FP/(C_FP+C_FN)` on the recalibrated score.

**Prevention.** Treat threshold and calibration as a single versioned artefact.

**Verification.** Replay a day under both artefacts and confirm action volume matches the pre-change
volume at the re-derived threshold.

### 8.16 CI-16 — trace cost grows with retries, not users

| ID | Category | Symptom |
|---|---|---|
| CI-16 | SCALE_FAILURE | Storage cost doubles while user volume is flat |

**Competing causes.** (a) Retry and tool-call counts rose, multiplying payload per run. (b) Prompt and
retrieved-context length grew with a model change. (c) Retention policy changed.

**Discriminating check.** Plot bytes per run against attempts per run. A linear relation with a steep
slope identifies (a); a step change at the model release identifies (b).

**Correction.** Cap trace payload per attempt, sample verbose traces, and keep aggregated metrics longer
than raw traces.

**Prevention.** Budget storage per run rather than per user, and alarm on bytes per run.

**Verification.** Recompute storage from a replayed day's run records and confirm it matches the measured
daily volume within sampling error.

In the worked example, 4,000 runs at 22 KB is 88 MB per day; the same runs with 40 label pairs at 55
bytes each is another 8.8 MB, and a doubled retry rate takes the first figure to about 176 MB with no
change in users.

### 8.17 CI-17 — raw agreement quoted as reliability

| ID | Category | Symptom |
|---|---|---|
| CI-17 | CONCEPT_CONFUSION | Two reviewers agree 83% of the time and the rubric is called reliable |

**Competing causes.** (a) Raw agreement is inflated because both reviewers pass most items. (b) The
sample is unrepresentative and contains only easy items. (c) The rubric genuinely has high reliability.

**Discriminating check.** Compute chance-expected agreement from the two marginals and the corrected
statistic. Section 4.3.1 gives 0.58820 expected and kappa 0.59123 for the same 0.832 agreement.

**Correction.** Report raw agreement alongside kappa and the marginals.

**Prevention.** Require marginals in any reliability table.

**Verification.** Recompute on a set with balanced classes; raw agreement should fall while kappa stays
approximately constant if the rubric is reliable.

The pathological case is worth memorising: two judges who both pass 95% and agree on 90% have kappa
-0.0526, because their agreement is *below* chance. Any process that reports 90% concordance there is
reporting agreement that is worse than random.

### 8.18 CI-18 — prevalence estimated from the labelled subset

| ID | Category | Symptom |
|---|---|---|
| CI-18 | NUMERICAL_EDGE | Expected precision recomputed from observed positives disagrees with the dashboard |

**Competing causes.** (a) Prevalence was computed on labelled items, whose selection favours one class.
(b) The dashboard uses a different population window. (c) One figure uses event time and the other uses
processing time.

**Discriminating check.** Compute prevalence three ways — labelled, eligible with pending excluded,
eligible with pending included as positives — and compare. If the dashboard matches none, the window is
different (b).

**Correction.** Declare the prevalence source and window; recompute affected metrics.

**Prevention.** Make prevalence a first-class stored field, not an inline derivation.

**Verification.** Recompute one month from raw counts and confirm the stored prevalence matches the
declared definition exactly.

### 8.19 CI-19 — the reference set is contaminated

| ID | Category | Symptom |
|---|---|---|
| CI-19 | VIOLATED_ASSUMPTION | The frozen set reports a new high pass rate after a release |

**Competing causes.** (a) The reference items entered the candidate's training or prompt-tuning data.
(b) The reference set was regenerated and now contains easier items. (c) The candidate genuinely
improved.

**Discriminating check.** Hash the reference items and compare against the training and tuning manifests.
Then score the incumbent on the current reference set; if the incumbent's score also rose, (b) is live.

**Correction.** Rebuild the reference set from items never used in training, and record provenance.

**Prevention.** Version the reference set by content hash and compare hashes against every training
manifest at build time.

**Verification.** Score both arms on a freshly constructed holdout and confirm the ordering, not just the
levels.

### 8.20 CI-20 — the suite exceeds the release window

| ID | Category | Symptom |
|---|---|---|
| CI-20 | SCALE_FAILURE | Full suite is "temporarily" skipped for the third consecutive release |

**Competing causes.** (a) Suite runtime exceeds the release cadence, so the gate is bypassed. (b) Cost
per run exceeds budget. (c) The suite is flaky and its failures are discounted.

**Discriminating check.** Measure runtime distribution and flake rate. A p95 runtime above the cadence
identifies (a); a flake rate above a few percent identifies (c) because the team has learned to ignore it.

**Correction.** Split into a fast blocking suite and a full nightly suite, with the nightly result blocking
the next release rather than the current one.

**Prevention.** Treat suite runtime and flake rate as service indicators with their own budgets.

**Verification.** Confirm the fast suite catches a deliberately injected defect of the class it claims to
cover, and that the nightly catches one only it covers.

A gate that is skipped is worse than no gate, because the release record shows a green tick that nobody
earned.

### 8.21 CI-21 — error budget read as a percentage

| ID | Category | Symptom |
|---|---|---|
| CI-21 | CONCEPT_CONFUSION | "We are at 99.4%, so we have 0.6% to spare" |

**Competing causes.** (a) The target is 99.5% and the budget is expressed as a count, so 0.6% of a
20,000-request day is 120 events while the monthly budget is 3,000. (b) The target applies to a different
window. (c) The 99.4% is a mean of daily rates rather than a pooled rate.

**Discriminating check.** Convert the budget to a count for the stated window and compare with observed
failures. Then check whether the reported rate is a mean of ratios (c) by recomputing from totals.

**Correction.** Express budgets as counts with a window, and burn as a dimensionless pace over that window.

**Prevention.** Store the budget count, not only the target percentage.

**Verification.** Recompute a month's burn from raw counts and confirm it equals observed failures divided
by the monthly budget.

### 8.22 CI-22 — burn rate across mismatched windows

| ID | Category | Symptom |
|---|---|---|
| CI-22 | NUMERICAL_EDGE | A one-day spike consumes "13% of budget" but pages nobody |

**Competing causes.** (a) The numerator is a one-day count and the denominator is the monthly budget, so
the ratio understates pace by 30. (b) The target changed mid-month. (c) The failures were retried
successes and should not count.

**Discriminating check.** Divide the numerator's window by the denominator's window and scale. In the
worked example, 400 failures against a 3,000 monthly budget reads 0.1333, but 400 against the 100-event
daily budget reads 4.0.

**Correction.** Compute burn over matched windows, or scale explicitly and label the scaling.

**Prevention.** Make the window a required argument of the burn function.

**Verification.** Recompute a historical incident's burn under both conventions and confirm the paging
decision matches the matched-window value.

### 8.23 CI-23 — recall estimated from alerted cases only

| ID | Category | Symptom |
|---|---|---|
| CI-23 | VIOLATED_ASSUMPTION | Recall looks strong, yet users report harms that were never flagged |

**Competing causes.** (a) The sample was drawn from alerts, so unflagged positives are absent by
construction. (b) The model genuinely captures most positives. (c) The outcome definition excludes the
missed harm type.

**Discriminating check.** Draw a random sample from the non-alerted population and adjudicate it. The
recall estimate requires the unflagged positive count, which only that sample can supply.

**Correction.** Estimate recall from a population sample, or report precision alone and state that recall
is unidentified.

**Prevention.** Make the sampling frame a required field of every recall estimate.

**Verification.** Compare the sampled recall with the alerted-only estimate on the same period; the gap is
the size of the selection effect.

An alerted-only sample cannot identify recall, because the denominator includes positives the system did
not find, and none of them can appear in the sample. This is not a precision problem; it is an
identification failure.

### 8.24 CI-24 — engagement read as correctness

| ID | Category | Symptom |
|---|---|---|
| CI-24 | CONCEPT_CONFUSION | Suggestion acceptance is 62% and the team calls it accuracy |

**Competing causes.** (a) Acceptance is caused by visibility, not correctness, and the system controls
visibility. (b) The population is self-selected by who reads suggestions. (c) Acceptance genuinely tracks
quality.

**Discriminating check.** Compare acceptance on items where the system's ordering was randomised. If
acceptance equalises across rank positions, (a) is confirmed. Compare accepted versus rejected items on
an independent rubric for (c).

**Correction.** Pair acceptance with a rubric judged on a sample, and report both.

**Prevention.** Classify engagement signals as exposure metrics in the metric registry.

**Verification.** Recompute the acceptance rate within a randomised subset and confirm it agrees with the
observed rate only if rank effects are absent.

### 8.25 CI-25 — mean of per-slice rates

| ID | Category | Symptom |
|---|---|---|
| CI-25 | NUMERICAL_EDGE | Dashboard shows 0.525 while the pooled figure is 0.4875 |

**Competing causes.** (a) The dashboard averages per-slice rates unweighted. (b) Slices are defined
differently between views. (c) A slice was dropped for low volume.

**Discriminating check.** Recompute unweighted and pooled from the same slice counts. A match with the
unweighted figure identifies (a); a missing slice identifies (c).

**Correction.** Compute the pooled rate from summed counts and publish slice rates separately.

**Prevention.** Remove unweighted averaging of ratios from the reporting layer.

**Verification.** Recompute a period where all slices have equal volume; there the two methods must agree,
which validates the implementation rather than the practice.

### 8.26 CI-26 — no version or actor, so rollback is undecidable

| ID | Category | Symptom |
|---|---|---|
| CI-26 | VIOLATED_ASSUMPTION | Nobody can say which version produced last week's outputs |

**Competing causes.** (a) The version was logged at the service level, not per run. (b) The version was
logged but the field is nullable and unset for a batch of runs. (c) The version changed mid-window and
the record was overwritten.

**Discriminating check.** Count runs with a null version field and compare the service-level deployment
timeline against per-run values. A null count identifies (b); a timeline mismatch identifies (a).

**Correction.** Backfill from deployment history where possible, and mark the rest unknown.

**Prevention.** Make the version and the acting principal non-nullable in the run record.

**Verification.** Attempt a rollback drill: pick a run from last week and reconstruct its exact version,
configuration, and model artefact. Failure of the drill is the defect.

A rollback decision requires knowing what to roll back to. Without per-run provenance the organisation
can only roll back to "yesterday", which is a different operation with a different risk profile.

### 8.27 CI-27 — judge capacity measured at pilot volume

| ID | Category | Symptom |
|---|---|---|
| CI-27 | SCALE_FAILURE | Judge scoring takes eleven hours at full traffic and blocks the daily gate |

**Competing causes.** (a) Throughput was measured on a sample and the queue is linear in volume. (b)
Per-item latency grew with context length. (c) Rate limits, not compute, are binding.

**Discriminating check.** Measure items per second at two volumes. Linear scaling identifies (a); rising
latency at constant volume identifies (b); throughput plateaus at a fixed ceiling identify (c).

**Correction.** Sample rather than score everything, batch where the provider allows, and move the full
run off the critical path.

**Prevention.** Capacity-test the judge at target volume before it becomes a gate.

**Verification.** Run the gate at production volume once and confirm completion inside the release window,
including the retry path.

**Takeaway:** the catalogue's entries are separated by observation rather than argument; each check is
chosen because the competing mechanisms move it differently.

## 9. Complexity & Cost Analysis

### 9.1 Cost of the instruments

| Instrument | Unit cost | Coverage | Cost at 4,000 runs/week | Bias to control |
|---|---|---|---|---|
| Request telemetry | negligible | 100% | ~0 | measures availability only |
| Schema check | negligible | 100% | ~0 | measures shape only |
| Judge | 0.0008 cu/item | 100% | 3.2 cu | 6.5 points optimistic here |
| Human audit | 4.5 cu/item | 15% | 2,700 cu | reviewer variance |
| Adversarial set | moderate per item | fixed | depends on size | fixed attack surface |
| Reference re-score | 0.0008 cu/item | 300 items | 0.24 cu | reference staleness |

The ratio between the judge and the audit is roughly 840:1 per item at this volume. That ratio explains
the whole architecture of a practical evaluation: the judge ranks and pre-filters, humans calibrate the
judge and adjudicate the disagreements, and the ratio is stable enough that a budget can be planned from
it.

### 9.2 Labels are the binding cost

```text
FORMULA: cost per decision = (n_audit * c_item) / n_decisions
NUMERICAL EXAMPLE: (600 * 4.5)/4,000 = 2,700/4,000 = 0.67500 cu per run
                  a 240-item audit costs (240*4.5)/4,000 = 0.27000 cu per run
INTERPRETATION: the audit costs two thirds of a currency unit per run, which is more than the run
                itself; label design is a budget decision, not an afterthought.
```

Sample size drives this directly. A ±4-point interval on a 0.81 rate needs 370 items; ±3 points needs
about 657; ±2 points needs 1,478. Because cost is linear in `n` and precision improves as `1/sqrt(n)`,
each halving of the interval costs about four times the labels. That is the curve a reviewer should
visualise before asking for "a tighter number".

### 9.3 Cost of late detection

A gate that runs nightly detects a regression one release late. If the regression affects 4,000 runs at
0.6825 pass and the cost of a failing run is 9.5 cu of downstream rework, one release of exposure is
`4,000 * 0.03 * 9.5 = 1,140` cu for a three-point regression. The judge costs 3.2 cu per week. The
asymmetry is about 350:1 in favour of running the suite, and it is the entire economic argument for
measurement investment — an argument that is forgotten after a quiet quarter.

### 9.4 Cost of a false alarm

A slice alert that pages an engineer costs attention even when nothing is wrong. At 40 slices and 24
looks per day with independent 3-sigma rules, expected false alarms per day are
`40*24*0.0027 = 2.59`. If each costs 15 minutes of attention, that is 39 minutes per day, or about 3.2
hours per week, spent on noise. Adjusting to a family-wise 5% threshold reduces the expected count to
about 2 per day across the whole slice set, and requiring persistence reduces it further at the cost of
detection latency.

The trade is explicit and should be made explicitly: latency against attention. A catalogue entry that
says "reduce false alarms" without naming the latency it buys has not made the trade, only deferred it.

### 9.5 Cost of the evaluation pipeline

The pipeline cost is dominated by the label join and by per-slice recomputation. Materialising the
partition once per cohort and computing every ratio from it keeps the join single-pass; re-scanning the
raw log per metric makes cost linear in the number of metrics and guarantees disagreement the first time
retention changes.

At the scale in this example — 4,000 runs and 40 features — none of this is a capacity problem. At four
million it is the difference between an hourly job and a nightly one, and between a gate that can block a
release and one that reports after the fact.

**Takeaway:** the judge is cheap and biased, the audit is expensive and calibrated, and the cost of
skipping the audit is invisible until a promotion is wrong.

## 10. Key Theorems / Results Reference

### 10.1 Positive predictive value identity

`PPV = se*pi / (se*pi + (1-sp)(1-pi))`. PPV is not a property of the model alone; it is a property of the
model applied to a population. The corollary that settles many arguments: with `se = 0.75` and
`sp = 0.93`, PPV at 0.5% prevalence is about 0.05 and at 20% prevalence about 0.72, from the same
artefact.

### 10.2 Rank statistics versus prevalence

AUC is the probability a random positive outranks a random negative, so it is invariant to monotone
score transforms and to resampling within class. Average precision is a sum of precisions whose
denominators contain negative counts, so it is not. Cross-population comparison of AP without
prevalence adjustment compares two different questions.

### 10.3 Dvoretzky–Kiefer–Wolfowitz

The empirical CDF lies within `epsilon` of the true CDF with probability at least `1 - 2exp(-2 n
epsilon^2)` uniformly over thresholds. At 95%, `epsilon = sqrt(ln(40)/(2n))`, giving 0.0679 at n = 400,
0.0392 at n = 1,200, and 0.0192 at n = 5,000. The practical reading is that no calibration or
discrimination curve from a few hundred items can be trusted to better than a few points at any
threshold, which is a stronger claim than a single ECE figure suggests.

### 10.4 Wilson versus Wald

Wilson inverts the score test and therefore has non-zero width at the boundary; Wald uses the estimated
variance and degenerates. Every zero-count cell in a monitoring report is a place where the difference
changes a decision.

### 10.5 Chance correction

Cohen's kappa subtracts chance agreement computed from the judges' own marginals. Its dependence on those
marginals is a property, not a bug, and it is why kappa cannot be compared across populations with
different class balances without also reporting the marginals.

### 10.6 Little's law

`L = lambda * W` in a stable system. Applied to a review queue it converts an arrival rate and a mean
dwell time into a depth, and it fails only when records are duplicated or dropped. Its diagnostic use is
negative: a queue whose depth disagrees with `lambda * W` has a record-integrity defect, not a capacity
problem.

### 10.7 CUSUM average run length

With reference value `k` and decision interval `h`, the in-control run length grows approximately
exponentially in `h`, while the out-of-control run length falls as the shift grows. The design
consequence is that cumulative sums pay for themselves only when the shift of interest is small relative
to noise; for large shifts a fixed band reacts in a single window and is simpler to explain.

### 10.8 Multiplicity

For `m` independent looks at level `alpha`, the family-wise error rate is `1 - (1-alpha)^m`. At
`alpha = 0.0027` that is 0.0628 over a day of hourly looks, 0.3651 over a week, and 0.8572 over a month.
No monitoring design survives an undeclared multiplicity, which is why the declaration precedes the
threshold.

**Takeaway:** eight results decide whether a printed interval, drift flag, or comparison is admissible;
cite the assumption with the number.

## 11. Interview / Exam Questions

### Q1. Coverage is 0.734 with a labelled rate of 0.240. What can and cannot be said about the population?

**Answer.** The labelled rate 0.240 is a statement about 2,937 runs. Extending it to 4,000 requires an
assumption about 1,063 unlabelled runs. The bounds under all-fail and all-pass assumptions are 0.176 and
0.442, a range of 0.266. Standardising by stratum gives 0.240 as well, which is why the weighting must be
checked rather than trusted — the printed weighted figure and the labelled figure coincide here only
because the weights happen to be uniform within class.

**Reasoning.** Coverage enters through the denominator, and the missing 26.6% of runs can move the
estimate more than any effect under discussion.

**Follow-up.** Which is cheaper: 400 more audits, or publishing the bounds?

**Red flag.** Quoting 0.240 as "the pass rate" and adding "approximately" to cover the difference.

**Scoring.** Full credit requires the labelled denominator, both bounds, and a statement that the interval
width exceeds the decision-relevant effect.

### Q2. A golden set of 240 items shows zero critical failures. What is the strongest claim?

**Answer.** That the critical-failure rate is below 1.24% at 95% confidence, from `1 - 0.05^(1/240)`.
It is not a claim of zero risk, and it is not a claim that the rate has not increased: with a true rate of
1%, observing zero has probability `0.99^240 = 0.0896`, so zero is an ordinary outcome under the old rate
too.

**Reasoning.** The rule-of-three bound converts a zero count into a rate bound whose width depends only on
n, and the second calculation shows the test's power to detect a change is negligible at this size.

**Follow-up.** How many items would give a 0.3% bound?

**Red flag.** "Zero failures observed, therefore no critical failures."

**Scoring.** Full credit gives the numeric bound, the power calculation, and the required n.

### Q3. Judge passes 0.745, humans pass 0.680 on the same 600 items. Is the judge usable?

**Answer.** Usable with a stated bias of about 6.5 points, provided the bias is stable and the reference
set is re-scored at each judge change. Raw agreement is 0.832 and kappa 0.591, so item-level reliability is
moderate rather than high. For promotion decisions the judge can rank, but the pass rate must come from the
human instrument. The paired test on the discordant counts gives McNemar chi-square 15.06, or z = 3.88,
so the 6.5-point gap is not sampling noise — it is instrument bias and must be corrected, not averaged.

**Reasoning.** Bias and reliability are separate properties: a judge can agree well on ranking while
shifting the level, and a level shift is fatal only when the judge's level is reported as the system's.

**Follow-up.** Which single number would you put in the release note?

**Red flag.** Reporting 0.745 as the pass rate because the judge scored everything.

**Scoring.** Full credit separates bias from reliability, cites both statistics, and assigns each
instrument its role.

### Q4. The same two judges produce 0.83 raw agreement. Why is kappa 0.59, and when would it be negative?

**Answer.** Chance agreement from the marginals is 0.588, so `(0.83-0.588)/(1-0.588) = 0.587`. It goes
negative when agreement is below what the marginals predict — two judges who both pass 95% and agree on
90% give `(0.90-0.905)/(1-0.905) = -0.0526`, because their disagreements exceed chance.

**Reasoning.** Kappa measures agreement in excess of the agreement implied by the judges' own rates, so a
skewed rubric depresses it.

**Follow-up.** What would you report instead if kappa is unstable across slices?

**Red flag.** Presenting 0.83 as reliability without the marginals.

**Scoring.** Full credit gives the arithmetic, the negative case, and the reason marginals matter.

### Q5. ECE is 0.018 with five bins. Is the model calibrated?

**Answer.** Not on that alone. Rebinning the same 5,000 items gives 0.0128 with four bins and 0.0088 with
two, so the value is partly a property of the binning. The defensible statement is "ECE 0.018 under five
equal-count bins, n = 5,000", plus a reliability curve and the maximum bin gap of 0.03.

**Reasoning.** Binned statistics average over bins, so both the weights and the within-bin dispersion
change with edges.

**Follow-up.** Which binning would you standardise on, and why?

**Red flag.** Comparing 0.018 against a 0.02 threshold as though both were binning-free.

**Scoring.** Full credit cites the spread, states the binning, and proposes a standard.

### Q6. Alert volume is 68 per day against capacity 90, yet the queue grows. What is happening?

**Answer.** The mean is comfortable and the peak is not: at a 1.4x peak the arrival rate is about 95
against capacity 90, so utilisation exceeds one and the queue is unstable. The backlog then suppresses
precision, because skipped reviews produce no labels and the skipped items are the least promising.

**Reasoning.** A queue is governed by its peak-hour utilisation, not its daily mean, and the labelling
consequence feeds back into the metric.

**Follow-up.** What would you monitor to catch this before the queue grows?

**Red flag.** Adding capacity equal to the mean and expecting the queue to clear.

**Scoring.** Full credit identifies peak utilisation, the feedback into precision, and a leading
indicator.

### Q7. A candidate scores 0.82 against 0.74 on a judge suite. What is the first action?

**Answer.** Re-score 300 frozen outputs with the current judge and compare against stored scores. A change
in stored-versus-rescored agreement proves instrument drift and invalidates the comparison until both arms
run under one judge version. Only then consider human adjudication of the residual gap.

**Reasoning.** The two arms were scored at different times, so the instrument is a confound, and checking
it is cheaper than more labels.

**Follow-up.** If the judge is stable, what sample size separates a three-point gain at 0.81?

**Red flag.** Commissioning a 1,000-item human study before checking the instrument.

**Scoring.** Full credit orders the checks correctly and names a concrete re-scoring design.

### Q8. Daily volume is 571 runs at a 0.6825 rate. Give the 3-sigma band and interpret 0.65.

**Answer.** `sqrt(0.6825*0.3175/571) = 0.01948`, so the band is `[0.6241, 0.7409]`. A day at 0.65 is
inside the band twice over and is not evidence of a change. At weekly volume the band narrows to
`[0.6604, 0.7046]`, which is why weekly reporting is more informative.

**Reasoning.** The band is the sampling distribution of a rate at that volume, so it narrows as volume
grows.

**Follow-up.** What sample is needed to detect a two-point drop at 80% power?

**Red flag.** Reacting to the daily number, or comparing a daily rate to a weekly mean.

**Scoring.** Full credit gives the standard error, both bounds, and the correct null verdict.

### Q9. First-attempt pass is 0.525 and eventual pass is 0.6825. Which goes in the product review?

**Answer.** 0.525, because that is what a user experiences on their first request. The 0.6825 measures
eventual capability including retries, a 30% relative uplift produced by the retry budget. Reporting the
higher figure in a product context overstates experience by 15.75 percentage points.

**Reasoning.** The gap is a system property, not a capability, and mixing them makes retry budget look
like quality.

**Follow-up.** What is cost per successful run under each definition?

**Red flag.** Averaging them, or reporting the higher one with the label "success rate".

**Scoring.** Full credit assigns each to its decision and quantifies the gap.

### Q10. Precision falls from 0.36 to 0.28 in three weeks with stable recall and false-positive rate. Diagnose.

**Answer.** Prevalence fell. With sensitivity and specificity fixed, PPV is a monotone function of
prevalence, so a fall at stable `se` and `sp` is arithmetically predicted. The check is to plug the new
prevalence into `PPV = se*pi/(se*pi + (1-sp)(1-pi))` and confirm the dashboard value.

**Reasoning.** Two of the three precision inputs were observed stable, so the third must have moved; the
identity makes that inference exact rather than suggestive.

**Follow-up.** Which metrics would you put on the dashboard instead?

**Red flag.** Retraining to fix a prevalence change.

**Scoring.** Full credit names prevalence, uses the identity, and proposes prevalence-free monitoring.

### Q11. Forty slices are monitored hourly at 3-sigma. How many false alarms per day, and what would you change?

**Answer.** `40*24*0.0027 = 2.59` expected false alarms per day, or about 18 per week. Change: predeclare
the slice set, allocate alpha across it (a Bonferroni 5% threshold at `z = 2.807` for 40 tests), and
require persistence over two or more windows before escalation.

**Reasoning.** Each test carries a per-look false-alarm rate, and the family-wise rate grows linearly in
the number of independent tests at small alpha.

**Follow-up.** What does requiring persistence cost in detection latency?

**Red flag.** Answering 0.27% per test and stopping there.

**Scoring.** Full credit gives the expected count, the adjustment, and the latency trade.

### Q12. A one-day spike shows 400 failures. Monthly budget is 3,000. What is the burn rate?

**Answer.** Two answers, and the difference is the point. Against the monthly budget, `400/3,000 =
0.1333`, which reads as modest. Against the daily budget of `3,000/30 = 100`, it is 4.0x pace, which is a
fast-burn page. The matched-window figure is the decision-relevant one.

**Reasoning.** A ratio of a daily count to a monthly budget understates pace by the ratio of the windows,
here 30.

**Follow-up.** What would you change in the alert rule so the mismatch cannot recur?

**Red flag.** Reporting 13% and concluding the day was unremarkable.

**Scoring.** Full credit gives both figures, names the window mismatch, and specifies the fix.

### Q13. An alerted-only sample gives recall 0.71. What is wrong?

**Answer.** Recall is not identified from that sample. The denominator includes positives the system never
flagged, and none can appear in an alerted-only sample, so the estimate is conditional on being alerted.
Estimating recall requires adjudicating a random sample of the non-alerted population.

**Reasoning.** The sampling frame excludes the event class that determines the denominator, which is an
identification failure rather than a bias to be corrected by weighting.

**Follow-up.** What sample size would give a ±5-point recall estimate at 0.71?

**Red flag.** Treating 0.71 as conservative and reporting it.

**Scoring.** Full credit names identification rather than bias, and specifies the required frame.

### Q14. A dashboard shows 0.525 and the pooled rate is 0.4875. Explain and fix.

**Answer.** The dashboard averages per-slice rates unweighted: `(0.60+0.45)/2 = 0.525`. The pooled rate is
`(60+135)/400 = 0.4875`. Averaging ratios weights a 100-run slice equally with a 300-run slice. Fix: sum
counts and divide once, and publish slice rates as separate columns.

**Reasoning.** The denominator of each slice rate is its own area, and averaging ratios averages areas as
if they were equal.

**Follow-up.** When is the unweighted mean the right quantity?

**Red flag.** Explaining the gap as "different slice definitions" without recomputing.

**Scoring.** Full credit reproduces both figures and names the mechanism.

**Takeaway:** the questions interlock on denominators, instruments, and windows — the three places where a
correct computation answers the wrong question.

## 12. Quick Reference Card

| Concern | Rule | Anchor number |
|---|---|---|
| Denominator | Name all four populations before computing any ratio | 4,000 eligible, 2,937 labelled, coverage 0.734 |
| Coverage | Publish it beside every rate, or weight by labelling probability | 0.17642 low bound to 0.44217 high bound |
| Interval | Wilson at boundaries, exact for zero counts | `1-0.05^(1/240) = 0.01239` |
| Success | Report first-attempt and eventual separately | 0.52500 against 0.68250, +30% |
| Instrument | Re-score a frozen set before comparing arms | 300-item reference, judge bias 6.5 points |
| Agreement | Raw agreement plus kappa plus marginals | 0.83167 raw, 0.58820 chance, 0.59123 kappa |
| Calibration | State the binning with the value | 0.01840, 0.01280, 0.00880 across binnings |
| Drift | Print per-bin contributions and name the excluded mechanism | PSI 0.02991, 74% from two outer bins |
| Capacity | Convert to reviews per day and check the peak, not the mean | 68 mean against 90 capacity |
| Multiplicity | Declare the number of looks before the threshold | `40*24*0.0027 = 2.59` false alarms/day |
| Windows | Match numerator and denominator windows, or scale explicitly | 400/3,000 = 0.1333 against 400/100 = 4.0 |
| Verification | Replay the trigger and a boundary item | interior plus boundary plus zero-label case |
