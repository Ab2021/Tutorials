# Public Health

> Module 23 — Specialty Electives
> Topic 06 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a legal opinion on
> public health authority, and not a substitute for the case definitions and reporting
> rules in force in your jurisdiction.

## 1. Why this topic exists

Public health analytics is the branch of the field where the denominator is the product.
A case count without a population at risk is a rumour. A rate without a stated
denominator is a comparison that cannot be reproduced. An estimate without a vintage is
a claim that will change next week, silently, after the data are revised.

The decisions are consequential and the feedback is slow. A health department decides
whether to open an outbreak investigation, a state decides how to allocate a limited
vaccine supply, a national programme decides whether to continue a screening
intervention, and an emergency operations centre decides where to send staff. Each
decision has a cost of error in both directions. A false alarm consumes investigator
time that is already scarce. A missed signal extends transmission for the weeks it takes
for the signal to become undeniable.

What makes public health distinct as a data discipline is not the modelling. It is that
the data-generating process is the intervention. Reporting is mandatory in some
jurisdictions and voluntary in others, laboratories differ in test panels and turnaround,
case definitions change between years, and the most recent weeks are always incomplete.
Every one of those features is a measurement property, not a nuisance, and an analyst
who treats them as nuisance produces a confident chart that is wrong at exactly the
point where decisions are made.

The signature failures are:

- **The last-point fallacy.** The most recent week always looks like a decline because
  reports have not arrived. A naive time series ends with a downward hook that is an
  artefact of the reporting delay distribution.
- **Denominator substitution.** Cases per 100,000 population are compared across
  jurisdictions with different testing access, or hospital-onset infection rates are
  divided by admissions rather than patient-days.
- **Case-definition drift.** A definition changes mid-series, and the resulting step
  change is interpreted as an epidemic change.
- **Vintage mixing.** Provisional counts from one download are compared with final
  counts from another, and the difference is attributed to epidemiology.
- **Ecological inference.** A tract-level correlation is read as an individual-level
  risk, which is the ecologic fallacy and is almost always wrong in the direction that
  flatters the analyst's hypothesis.
- **Suppression artefacts.** Small counts are suppressed for privacy, and the suppressed
  cells are treated as zero, which manufactures declines in rural and small
  jurisdictions.

Each of these is a measurement problem with a standard remedy. None of the remedies is
a better model.

### 1.1 Who is blocked when the measurement is wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient or resident | A warning that arrived late or a service that was aimed at the wrong area | Cannot see the denominators |
| Local health officer | Credibility when a published rate is revised downward | Depends on state and federal systems |
| Epidemiologist | Time reconstructing a series that changed definition | No control over the case definition |
| Laboratory | Blame for a delay caused by reporting rules | Reports into a system it does not own |
| Hospital | A penalty based on a rate with the wrong denominator | Cannot change the measure specification |
| Elected official | A decision made on a signal that later disappears | Receives charts, not reporting triangles |
| Federal agency | Public trust when provisional data are treated as final | Cannot control local reporting |
| Analyst | A model that cannot be validated because the target moves | Inherited a series without metadata |

The last row is the professional risk. A public health model is a claim about a
population, and the population changes every time a case definition or a reporting rule
changes. If the analysis cannot state the definition version and the data vintage, the
claim cannot be checked.

### 1.2 The industry question

Which decision does this work change, who pays for it, and which specialty-specific bias
will invalidate the evidence first? For public health the third clause has a specific
answer: the denominator and the reporting delay. A model that predicts outbreaks but
uses a denominator that changed mid-series is not measuring outbreaks. It is measuring
the denominator.

### 1.3 What this topic covers

This topic covers the measurement chain from a single case report to a published
indicator: case definitions, event dates and report dates, the reporting triangle,
nowcasting, denominators, age standardisation, revision and backfill, and the privacy
rules that shape what can be published. It uses notifiable disease surveillance as the
running example because it exposes every problem at once. Environmental surveillance,
syndromic surveillance, and screening programme evaluation appear as variations.

It does not teach infectious disease epidemiology, outbreak investigation, or health
policy. It assumes you can read a rate and will not re-derive the standard error of a
directly standardised rate, though it does insist that you state the standard
population.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Build a reporting triangle from event dates and report dates and estimate a delay distribution | Exercise 1; self-check 1 |
| 2 | Nowcast the most recent weeks and compare the result with a persistence baseline | Exercise 2; self-check 2 |
| 3 | State and defend the denominator for incidence, hospital-onset infection, and maternal mortality | Exercise 3; interview question 2 |
| 4 | Age-standardise a rate and explain why the crude comparison was misleading | Exercise 4; self-check 4 |
| 5 | Detect a case-definition change in a time series and separate it from a real change in transmission | Exercise 5; interview question 5 |
| 6 | Quantify revision and backfill and explain why a provisional count is not a final count | Exercise 6; self-check 6 |
| 7 | Apply small-cell suppression and explain how treating suppressed cells as zero biases a map | Exercise 7; interview question 7 |
| 8 | Choose between an ecologic and an individual-level design for a stated question | Exercise 8; interview question 8 |

Outcomes 1, 2, and 6 are the technical core and the ones that require arithmetic.
Outcomes 3 and 4 are the denominator skills that distinguish a public health analyst
from a generic data scientist. Outcomes 5 and 7 are the audit skills that catch the
failures in Section 1. Outcome 8 is the design skill that prevents an entire class of
invalid claims.

Outcome 1 is assessed by construction, not by description. A learner who can build a
triangle from a case-level extract has understood the difference between a missing cell
and a zero. Outcome 2 is assessed by backtest, because a nowcast that has never been
scored has no claim to being better than the last observed value.

Outcome 3 is assessed by a table of indicators and denominators, and the assessment is
strict about direction of bias. Saying that the wrong denominator "affects the rate" is
not sufficient; the answer must say whether the rate is too high or too low and why.

Outcomes 5 and 7 are assessed by critique. The learner is given a chart and must find
the artefact. This is the closest assessment to the actual job, where the artefact is
rarely labelled and the chart is usually accompanied by a confident narrative.

### 2.1 How the outcomes map to the module

Outcome 1 is shared with critical care (time zero) and oncology (event dating); the
mechanics differ but the principle is identical. Outcome 3 is the public health version
of the cohort definition in topic 01. Outcome 8 is developed further in topic 07, where
the comparison across specialties makes the ecologic problem explicit.

If you retain only one outcome, retain outcome 3. Every other error produces a number
that can be corrected; a wrong denominator produces a number that looks right.

## 3. The core mental model

A public health indicator is produced by a chain that runs from a person to a published
number, and every link adds delay, revision, and a definition.

```text
  Person with illness
        |
        v
  [1] Care or test seeking ..... access, insurance, language, distance
        |
        v
  [2] Test ..................... panel, sensitivity, turnaround
        |
        v
  [3] Diagnosis and reporting .. case definition version, mandatory vs voluntary
        |
        v
  [4] Report received .......... delay distribution, duplicate, residence vs report site
        |
        v
  [5] Case classification ...... confirmed / probable / suspect; counted or not
        |
        v
  [6] Aggregation .............. week assignment, revision, backfill, suppression
        |
        v
  [7] Denominator .............. population, patient-days, live births, sewershed
        |
        v
  [8] Indicator and decision ... rate, Rt, positivity, trend, alert
```

Read the chain as a set of filters that are correlated with each other. People with less
access are less likely to reach [1], which means they are less likely to appear at [2]
and [3]. The indicator at [8] therefore undercounts exactly the populations with the
highest burden, and the undercount is not random.

### 3.1 The reporting triangle

The single most useful object in public health analytics is the reporting triangle. Rows
are event weeks; columns are delays between the event and the report; cells are counts
observed at a fixed data vintage.

```text
                 delay 0   delay 1   delay 2   delay 3   total observed
  week 40           90        0         0         0          90
  week 41           70       20         7         3         100
  week 42           77       22         8         .         107
  week 43           84        .         .         .          84
```

The dots are not zero. They are unreported, and the difference between a dot and a zero
is the entire discipline. A triangle read as a table of zeros produces the last-point
fallacy; a triangle read as a table of censored observations produces a nowcast.

### 3.2 Two clocks and one vintage

Every public health record has at least two dates and every dataset has a vintage.

| Clock | What it marks | Why it matters |
|---|---|---|
| Event date | Symptom onset, specimen collection, or diagnosis | The epidemiological clock; the one that answers "when did transmission happen" |
| Report date | When the jurisdiction received the record | The operational clock; the one that determines completeness |
| Vintage | When the dataset was extracted | The only way to know which cells were observable |

An analysis that mixes clocks produces a series that is neither epidemiological nor
operational. An analysis that omits the vintage cannot be reproduced, because the same
query run next week returns different numbers.

### 3.3 The denominator ladder

Public health has a ladder of denominators, and choosing the wrong rung is the most
common analytic error in the field.

```text
  resident population  -> incidence, prevalence, mortality per 100,000
  person-time          -> incidence rate, hazard
  patient-days         -> hospital-onset infection
  live births          -> maternal mortality, infant mortality
  pregnancies          -> pregnancy-related outcomes
  tests performed      -> test positivity
  sewershed population -> wastewater signal per capita
  eligible population  -> vaccination coverage
```

Each rung answers a different question. "Cases per 100,000 residents" and "cases per
100,000 tests" can move in opposite directions during the same week, and both can be
correct for their own question.

A concrete version of the ladder matters because the error is easy to make and hard to
see. Suppose a jurisdiction performs 20,000 tests in week 1 and 40,000 in week 2, with
200 and 320 positive results. Cases rose by 60 percent, but tests rose by 100 percent, so
positivity fell from 1.0 percent to 0.8 percent. Per 100,000 residents with a population
of 500,000, the rates are 40 and 64 per 100,000, which is the same 60 percent rise as the
count. The three indicators tell three stories: cases up 60 percent, positivity down
from 1.0 to 0.8 percent, and the population rate up 60 percent. None is wrong, and a
claim that does not say which denominator it used cannot be checked.

### 3.4 What the model is allowed to see

The decision wall in public health is the data vintage. At the moment a decision is
made, only the reports available at that vintage exist. Backtesting a nowcast with the
final revised data is the public health form of leakage: it uses information that the
decision-maker did not have.

```text
  available at vintage t .......... | decision at t | ........... future revisions
  reports received by t,            |               |  reports arriving after t,
  case definition in force,         |               |  reclassifications, renumbers
  denominators published at t       |               |
```

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Notifiable disease | A condition that law or regulation requires clinicians or laboratories to report | Assumed to be a complete census of cases | Jurisdiction-specific list |
| Case definition | The criteria that determine whether a person counts as a case | Treated as stable across years | CSTE position statement identifier |
| Confirmed case | Meets laboratory criteria in the case definition | Pooled with probable cases without a flag | Classification category |
| Probable case | Meets clinical and epidemiological criteria without laboratory confirmation | Counted as confirmed in dashboards | Classification category |
| Suspect case | Meets preliminary criteria pending confirmation | Counted once and never updated | Classification category |
| Event date | The date the event occurred, usually onset or specimen collection | Replaced by report date in extracts | ISO 8601 date |
| Report date | The date the jurisdiction received the report | Used as if it were the event date | ISO 8601 date |
| MMWR week | The CDC epidemiological week used for aggregation | Confused with ISO week, which starts Monday | Sunday-start week numbering |
| Vintage | The date and time a dataset was extracted | Not recorded, making the analysis irreproducible | Timestamp |
| Reporting triangle | Counts by event date and reporting delay | Read as a complete table with zeros | Counts |
| Backfill | Reports that arrive after the first publication and raise historical counts | Mistaken for a new outbreak | Counts |
| Revision | Any change to a previously published count, including decreases | Assumed to be rare and small | Counts |
| Nowcast | An estimate of the current value after correcting for reporting delay | Treated as observed data | Count |
| Delay distribution | The probability that a case is reported at each delay | Assumed constant across seasons and systems | Proportion per delay |
| Persistence baseline | Forecast equal to the last observed value | Treated as a straw man rather than a required benchmark | Count |
| WIS | Weighted interval score, a proper scoring rule for probabilistic forecasts | Replaced by point error alone | Score, lower is better |
| Incidence | New cases divided by person-time at risk | Computed with a denominator that includes prevalent cases | Per 100,000 person-years |
| Prevalence | Existing cases divided by the population at a point or period | Confused with incidence | Proportion |
| Case fatality ratio | Deaths divided by diagnosed cases | Interpreted as infection fatality ratio | Proportion |
| Infection fatality ratio | Deaths divided by all infections, including undiagnosed | Estimated from case counts | Proportion |
| Test positivity | Positive tests divided by tests performed | Used as a prevalence estimate without adjustment | Proportion |
| Rt | Effective reproduction number, average secondary infections per case | Reported without the generation interval | Dimensionless |
| Excess mortality | Observed minus expected deaths | Computed against a baseline chosen after seeing the data | Count or rate |
| Age standardisation | Adjusting rates to a common age structure | Reported without naming the standard population | Rate per 100,000 |
| Direct standardisation | Applying a standard age structure to age-specific rates | Used when age-specific rates are unstable | Rate |
| Indirect standardisation | Comparing observed with expected events using a reference | Reported as a directly standardised rate | SMR |
| SMR | Standardised mortality ratio, observed divided by expected | Interpreted as a rate | Ratio |
| Ecologic fallacy | Inferring individual risk from group-level association | Used to target individuals | Not a unit |
| Capture-recapture | Estimating an unobserved total from overlap between incomplete sources | Applied without checking source independence | Count estimate |
| Syndromic surveillance | Monitoring pre-diagnostic signals such as chief complaints | Treated as diagnosis data | Syndrome category |
| Sewershed | The catchment area contributing to a wastewater sampling site | Treated as a political boundary | Geographic area |
| WVAL | Wastewater viral activity level, a standardised signal used by CDC NWSS | Compared across pathogens | Ordinal level |
| Suppression | Withholding small counts to protect privacy | Treated as zero | Rule-based |
| SVI | CDC/ATSDR Social Vulnerability Index | Treated as individual risk | Index 0 to 1 |
| ACS | American Community Survey, source of population estimates | Used without margins of error | Estimate with MOE |

### 4.1 Three terms that carry the most confusion

**Incidence versus prevalence.** Incidence counts new events over a period and requires
person-time. Prevalence counts existing cases at a moment and requires only a population.
A screening programme that detects more early cases raises prevalence and incidence
simultaneously while changing neither transmission nor mortality. Reporting "cases are
rising" without saying which measure is being used is not a finding.

**Case fatality versus infection fatality.** The case fatality ratio divides deaths by
diagnosed cases, so it rises when testing narrows to severe cases and falls when testing
expands. The infection fatality ratio divides deaths by all infections and requires
serology or modelling. During a testing expansion, the case fatality ratio can fall
while the infection fatality ratio is unchanged, and the fall is an artefact of who got
tested.

**Provisional versus final.** A provisional count is a statement about reports received,
not about cases that exist. The NNDSS notice to data users states that weekly data are
provisional and subject to change until reconciled annually, and that cumulative counts
can decrease as well as increase when duplicates are removed or classifications are
revised. Every chart of provisional data should carry that sentence.

## 5. The mechanics, step by step

This section is the procedure for producing a defensible surveillance indicator. It is
written as ten steps, each with its input, output, rationale, branch points, and edge
cases. The running example is a weekly notifiable disease series, because it exposes
every failure mode in Section 1.

### Step 1 — Name the decision and the indicator

**Input.** A decision, a decision-maker, and a cadence.

**Output.** One indicator specification with a numerator, a denominator, a time unit, a
geography, and a threshold for action.

**Rationale.** Surveillance produces many numbers and few decisions. An indicator that
no one acts on is a reporting obligation, not an analytic product. Naming the decision
first determines whether the right measure is a count, a rate, a trend, or a forecast.

**Procedure.**

1. Write the decision as an action with an actor: "the health officer will open an
   outbreak investigation if the weekly count exceeds the seasonal baseline by more than
   two standard deviations for two consecutive weeks."
2. Write the indicator to match: weekly confirmed cases by event week, compared with a
   seasonal baseline, with an alert rule.
3. State the geography: residence of the case, not location of the reporting laboratory.
4. State the cadence: how often the indicator is refreshed and who signs it off.
5. State what would change the decision in the other direction, so that the indicator
   has both a trigger and a stand-down rule.

**Branch point.** If the decision is about resource allocation across regions, the
indicator needs a denominator and an uncertainty interval, not a count. If the decision
is about whether to investigate a single cluster, the indicator needs case-level data
and a space-time scan, not a weekly rate.

**Edge case.** Decisions that are legally mandated on a fixed schedule, such as a
weekly publication, do not adapt to the indicator's uncertainty. Publish the interval
anyway and state the revision policy.

### Step 2 — Fix the case definition and its version

**Input.** The jurisdiction's notifiable condition list and the current CSTE position
statement or national equivalent.

**Output.** A definition record with the condition, the classification levels, the
laboratory criteria, the version identifier, and the effective date.

**Rationale.** The case definition is the measurement instrument. When it changes, the
series changes, and the change is not epidemiology.

**Procedure.**

1. Record the definition identifier and effective date, not just the condition name.
2. Record which classification levels are counted: confirmed only, confirmed plus
   probable, or all including suspect.
3. Record the laboratory criteria and whether they changed. A switch from a
   culture-based to a PCR-based criterion increases detected cases without increasing
   disease.
4. Record any jurisdiction-specific modifications. States may adopt a CSTE definition
   with local amendments.
5. Version the definition in the analytic table, one row per case, so that a series can
   be stratified by definition version.

**Branch point.** If the definition changed mid-series, either restrict the analysis to
one version or produce a bridged series with both versions computed for an overlap
period. Bridging requires case-level data with the components of both definitions.

**Edge case.** A definition can change in a way that removes cases. When a condition is
removed from the notifiable list, or a classification is retired, the series drops to
zero. The drop is administrative, and a chart that does not annotate it will be read as
the disappearance of a disease.

### Step 3 — Assign the event date

**Input.** Case-level records with onset date, specimen collection date, diagnosis date,
and report date.

**Output.** One event date per case, with a field recording which date was used and
whether it was imputed.

**Rationale.** The event date defines the epidemic clock. Report date defines the
operational clock. Using report date makes the series lag by the reporting delay and
makes the recent period look artificially low.

**Procedure.**

1. Prefer onset date. If absent, use specimen collection date. If absent, use diagnosis
   date. Record the hierarchy and the share of cases at each level.
2. Do not impute onset date from report date without a flag. The imputation compresses
   the epidemic curve toward the reporting weeks and flattens peaks.
3. Assign the aggregation week using the correct week-numbering convention. CDC uses
   MMWR weeks, which start on Sunday; ISO weeks start on Monday. A one-day mismatch
   moves cases between weeks at every boundary.
4. Check for impossible dates: onset after specimen collection, report before onset,
   dates in the future.
5. Record the share of records with an imputed event date by jurisdiction.

**Decision table for event-date assignment.** The choice has more than two outputs, and
each has a different consequence.

| Available dates | Event date used | Flag | Consequence |
|---|---|---|---|
| Onset and specimen | Onset | None | Correct epidemic clock |
| Specimen only | Specimen collection | `date_type=specimen` | Slight lag relative to onset |
| Diagnosis only | Diagnosis | `date_type=diagnosis` | Lag depends on testing turnaround |
| Report only | Report date, flagged | `date_type=report` | Series shifted later; recent weeks low |
| Onset after specimen | Specimen collection | `date_conflict=1` | Indicates a data-entry error to review |
| Onset missing for a large share in one jurisdiction | Report date for that jurisdiction, flagged | `date_type=report`, `jurisdiction` | Cross-jurisdiction comparison biased |
| Date in the future | Excluded from the current vintage | `date_invalid=1` | Prevents a spurious spike |

**Edge case.** Retrospective dating. A case reported in week 10 with an onset in week 4
moves the week 4 count up and the week 10 count down relative to a real-time chart. The
revision is correct and the original chart was not wrong at the time; it was provisional.

### Step 4 — Build the reporting triangle

**Input.** Case-level records with event week, report date, and vintage.

**Output.** A matrix of counts by event week and delay, plus the observed total at each
vintage.

**Rationale.** The triangle is the only structure that separates "no cases" from "no
reports yet". Every nowcast, revision analysis, and backfill estimate is computed from
it.

**Procedure.**

1. Set the delay as the difference in weeks between the report date and the event date.
   Negative delays indicate a date error, not a fast laboratory.
2. Tabulate counts by event week and delay, using one vintage for the whole table.
3. Mark cells that were not observable at the vintage as missing, not zero.
4. Repeat the tabulation at several vintages to measure revision.
5. Estimate the delay distribution from complete weeks only. A week is complete when the
   maximum observed delay exceeds the plausible maximum, or when the cumulative
   proportion has plateaued.
6. Check whether the delay distribution varies by jurisdiction, season, day of week, and
   holiday. It usually does.

**Branch point.** If the maximum delay is long relative to the reporting cadence, a
simple distribution is inadequate and a hierarchical delay model is needed. If the delay
is short, a simple empirical distribution is defensible and easier to audit.

**Edge case.** Duplicate reports. A case reported twice, or a case transferred between
jurisdictions, appears as a new report at a later delay. Deduplicate on a stable case
identifier plus a rule for the reporting jurisdiction, and record the rule.

### Step 5 — Estimate and validate the delay distribution

**Input.** The triangle for complete weeks.

**Output.** A delay distribution with uncertainty, plus diagnostics for non-constancy.

**Rationale.** The nowcast is only as good as the delay distribution. A distribution
estimated during a period of stable testing will misestimate the current week if testing
has expanded or a laboratory has changed its reporting system.

**Procedure.**

1. Compute the empirical distribution over delays using complete weeks.
2. Stratify by jurisdiction and by laboratory type if the counts allow.
3. Test for a trend in the distribution over the last 12 weeks.
4. Compare the distribution in holiday weeks with ordinary weeks.
5. Report the distribution and its variability, not only its mean.

**Branch point.** If the distribution is non-stationary, use a time-varying or
hierarchical delay model, or restrict the nowcast to a shorter horizon and report wider
intervals.

**Edge case.** A laboratory onboarding. When a large laboratory begins electronic
reporting, the delay distribution shifts abruptly and the recent triangle is not
comparable with the historical one. The nowcast will over- or under-estimate by the size
of the shift, and the shift is detectable as a change in the delay-0 share.

### Step 6 — Nowcast the incomplete weeks

**Input.** The triangle at the current vintage and the delay distribution.

**Output.** A point estimate and an interval for each incomplete week.

**Rationale.** The most recent weeks are the ones decisions are made on, and they are
the least complete. A nowcast converts "reports received so far" into "estimated cases
to date".

**Procedure.**

1. For each incomplete week, compute the cumulative proportion of the delay distribution
   observed so far.
2. Divide the observed count by that proportion. This is the delay-adjustment estimator.
3. Propagate uncertainty from the delay distribution and from the observed count. A
   simple interval can be obtained by resampling the delay distribution, or by using a
   negative binomial observation model.
4. Do not nowcast more than two or three weeks beyond the last complete week unless the
   delay distribution is stable and the intervals are honest about it.
5. Report the nowcast as an estimate, visually distinct from observed counts.

**Branch point.** If the observed count for the current week is zero, the estimator
returns zero. That is wrong whenever the delay-0 share is below one. Report a
probability of at least one case rather than a point estimate of zero.

**Edge case.** Exponential growth. During rapid growth, the current week contains a
disproportionate share of cases that will be reported late, so a constant-delay
nowcast under-estimates. The bias is proportional to the growth rate and the mean delay,
and it can be corrected analytically or by simulation.

### Step 7 — Choose and compute the denominator

**Input.** The indicator definition and the available population or exposure data.

**Output.** A denominator with a vintage, a geography, and a stated definition.

**Rationale.** A rate is a ratio, and the ratio is only as good as its denominator. The
wrong rung of the denominator ladder produces a number that looks precise and answers a
different question.

**Procedure.**

1. Match the denominator to the question: resident population for incidence, person-time
   for incidence rate, patient-days for hospital-onset infection, live births for
   maternal mortality, tests for positivity, eligible population for coverage.
2. Record the population vintage. Census estimates are revised; ACS estimates carry
   margins of error; intercensal estimates change.
3. Match the geography. Cases by residence and population by residence; not cases by
   reporting site and population by residence.
4. For small areas, report the margin of error or suppress the rate.
5. For coverage, define the eligible population explicitly, including age bands and
   contraindications if they are excluded.

**Branch point.** If the numerator and denominator come from different geographies or
periods, the rate is not interpretable. Rebuild one of them rather than adjusting.

**Edge case.** Testing-dependent indicators. Test positivity depends on who is tested.
When testing expands to asymptomatic people, positivity falls even if incidence is
flat. Report testing volume alongside positivity, and never interpret positivity alone
as a prevalence estimate.

### Step 8 — Standardise and compare

**Input.** Rates by age group or another confounder, and a standard population.

**Output.** A standardised rate or ratio with the standard population named.

**Rationale.** Crude rates differ when age structures differ. A state with an older
population will have a higher crude mortality rate for many conditions without any
difference in age-specific risk.

**Procedure.**

1. Choose direct or indirect standardisation. Direct requires stable age-specific rates;
   indirect is more stable when counts are small.
2. Name the standard population and its year. Common choices include the 2000 US
   standard population and the WHO world standard.
3. Compute the standardised rate as the weighted sum of age-specific rates, with weights
   from the standard population.
4. Report the crude and standardised rates side by side, and the direction of the
   difference.
5. For comparisons over time, hold the standard population fixed.

**Branch point.** If the age-specific counts are small, direct standardisation produces
unstable rates. Use indirect standardisation and report the SMR with its interval.

**Edge case.** Small-area suppression. A standardised rate for a small county may be
based on three deaths. Report the count, suppress the rate, or use a Bayesian
small-area model with explicit shrinkage. Do not publish a rate that implies precision
the data do not support.

### Step 9 — Publish with vintage, uncertainty, and revision policy

**Input.** The indicator, its interval, the definitions used, and the revision history.

**Output.** A release with metadata that lets a reader reproduce and reinterpret it.

**Rationale.** Public health data are revised. A release without a vintage and a
revision policy cannot be compared with the next release, and the comparison will
produce spurious findings.

**Procedure.**

1. State the data vintage to the day.
2. State the case definition version and any changes since the last release.
3. State the denominator and its vintage.
4. Show uncertainty for every estimate, including counts near zero.
5. Annotate definition changes and laboratory changes on the chart.
6. Publish a revision policy: which weeks are considered complete, when final data
   replace provisional data, and how errata are issued.
7. Archive each release so that a later analysis can reproduce what was known at the
   time.

**Branch point.** If a revision changes a previously published conclusion, issue an
explicit erratum rather than silently updating the chart.

**Edge case.** Suppression-driven artefacts. When a cell is suppressed, publish the
suppression rule and avoid treating the cell as zero in derived totals. A map that
renders suppressed counties as zero will show a false rural decline.

### Step 10 — Make the release reproducible

**Input.** The code, the extract, the definitions, and the release note.

**Output.** A reproducible release package.

**Rationale.** A surveillance release is used in decisions and later cited in
evaluations. If it cannot be reproduced, the later evaluation cannot tell whether a
difference is epidemiology or a change in method.

**Procedure.**

1. Store the query and the extract identifier with the release.
2. Store the case definition versions used for each jurisdiction.
3. Store the delay distribution and the nowcast code, with the vintage used for each
   week.
4. Record every manual adjustment and who approved it.
5. Re-run the release from the stored artifacts before publication and compare the
   output with the draft.

**Branch point.** If the release is produced by a vendor, require the vendor to supply
the same artifacts. A black-box indicator cannot be revised transparently.

**Edge case.** A mid-cycle system migration. When a surveillance system is replaced, the
old and new systems overlap for a period. Publish both series in the overlap and state
the mapping; do not splice them without showing the join.

## 6. Worked example

A state health department publishes weekly confirmed cases of a enteric notifiable
disease. The analyst is asked whether the most recent week shows the start of an
outbreak. The data vintage is the end of week 43.

### 6.1 The reporting triangle

The delay distribution, estimated from the previous 52 complete weeks, is 0.70 reported
in the event week, 0.20 one week later, 0.07 two weeks later, and 0.03 three weeks
later. It sums to 1.00.

| Event week | Delay 0 | Delay 1 | Delay 2 | Delay 3 | Observed at vintage |
|---|---:|---:|---:|---:|---:|
| 40 | 90 | 0 | 0 | 0 | 90 |
| 41 | 70 | 20 | 7 | 3 | 100 |
| 42 | 77 | 22 | 8 | not yet | 107 |
| 43 | 84 | not yet | not yet | not yet | 84 |

Weeks 40 and 41 are complete. Week 42 is missing the delay-3 cell. Week 43 is missing
delays 1 to 3.

### 6.2 The naive chart and its error

Plotting the observed column gives 90, 100, 107, 84. The last point is a 21.5 percent
fall from the previous week:

```text
(84 - 107) / 107 = -0.2150 -> -21.5%
```

That is the chart a dashboard would show, and it is the chart that produces a stand-down
decision. It is wrong, because week 43 has only received 70 percent of its eventual
reports.

### 6.3 The nowcast

For each incomplete week, divide the observed count by the cumulative delay proportion
observed so far.

```text
Week 42: delays 0-2 observed -> 0.70 + 0.20 + 0.07 = 0.97
         107 / 0.97 = 110.3

Week 43: delay 0 observed    -> 0.70
         84 / 0.70 = 120.0
```

The nowcast series is 90, 100, 110.3, 120.0. The change from week 42 to week 43 is:

```text
(120.0 - 110.3) / 110.3 = +0.0880 -> +8.8%
```

The naive chart says a 21.5 percent decline. The nowcast says an 8.8 percent increase.
The direction of the conclusion reverses, and the decision reverses with it.

### 6.4 Two diagnostics that reveal a broken analysis

**Diagnostic 1: plot the observed and estimated series on the same axis and mark the
vintage.** If the observed series always declines in the final one to three points,
across many weeks and many jurisdictions, the decline is the reporting delay and not the
epidemic. The diagnostic is structural: compute the mean observed change in the final
week across the last 26 vintages. If it is reliably negative while the nowcast is not,
the pipeline is plotting incomplete data. In this example the final-week change is
negative in most vintages by construction, which is the signature.

**Diagnostic 2: compare the delay distribution in the current period with the historical
one.** The nowcast assumes 0.70 of cases are reported in the event week. Suppose a
laboratory switched to electronic reporting in week 20 and the delay-0 share rose to
0.85. Applying 0.70 to the current data divides by too small a number and over-estimates:

```text
Correct:   84 / 0.85 = 98.8
Applied:   84 / 0.70 = 120.0
Over-estimate: 21.2 cases, or 21.5%
```

The diagnostic is to recompute the delay distribution on the most recent eight complete
weeks and compare it with the historical distribution. A difference larger than the
sampling variability is a signal that the nowcast is mis-specified.

### 6.5 Vary one assumption

The nowcast above assumes a constant delay distribution. Suppose the relevant
distribution is not 0.70 but 0.60, because a holiday week reduced laboratory reporting
in the event week. The estimate becomes:

```text
84 / 0.60 = 140.0
```

The conclusion moves from "cases are rising by 8.8 percent" to "cases are rising by 27
percent", and the implied alert level changes. The direction does not change here, but
the magnitude changes enough to alter a threshold-based decision.

A second variation is more consequential. Suppose the analyst nowcasts week 43 and
compares it with the nowcast for week 42 rather than the observed value. The comparison
is then 120.0 against 110.3, which is consistent. If instead the analyst compares the
nowcast for week 43 with the observed value for week 42, the comparison is 120.0 against
107, which overstates the increase:

```text
(120.0 - 107) / 107 = +12.1%   vs the correct +8.8%
```

Mixing observed and estimated values in a single trend line is a common and subtle
error. The rule is to compare like with like: nowcast with nowcast, or observed with
observed at the same vintage.

### 6.6 Checking the nowcast against a seasonal baseline

A rising nowcast is not by itself an outbreak signal. The question the health officer
actually asks is whether the current value is unusual for the season. Suppose the
previous five years give counts for the same event week of 78, 84, 91, 88, and 95. The
mean is 87.2 and the sample standard deviation is 6.6.

```text
mean = (78 + 84 + 91 + 88 + 95) / 5 = 436 / 5 = 87.2
deviations: -9.2, -3.2, 3.8, 0.8, 7.8
squared:    84.64, 10.24, 14.44, 0.64, 60.84  -> sum 170.80
variance = 170.80 / 4 = 42.70; sd = sqrt(42.70) = 6.53
```

Using 6.53 rather than the rounded 6.6, the nowcast of 120.0 is:

```text
z = (120.0 - 87.2) / 6.53 = 5.02
```

That is five standard deviations above the historical mean, and it is above the largest
historical value of 95. A threshold rule of two consecutive weeks above two standard
deviations would trigger.

The baseline has two weaknesses that must be reported. First, five points give a very
uncertain standard deviation; a 95 percent prediction interval using a t distribution
with four degrees of freedom is approximately:

```text
87.2 +/- 2.776 * 6.53 * sqrt(1 + 1/5)
= 87.2 +/- 2.776 * 6.53 * 1.0954
= 87.2 +/- 19.86
= [67.3, 107.1]
```

The nowcast of 120.0 lies above that interval, so the signal survives the small-sample
correction. Second, the historical years may have used different case definitions or
testing volumes. If the definition widened in the current year, part of the elevation is
administrative. The baseline is evidence, not proof, and it is only as good as the
comparability of its years.

| Series | Week 41 | Week 42 | Week 43 | Week 42 to 43 change |
|---|---:|---:|---:|---:|
| Observed at vintage | 100 | 107 | 84 | −21.5% |
| Nowcast | 100.0 | 110.3 | 120.0 | +8.8% |
| Seasonal mean | 87.2 | 87.2 | 87.2 | 0.0% |
| Above-baseline z | 1.96 | 3.54 | 5.02 | — |

The table is the artifact a decision-maker needs: three series, one change calculation,
and a standardised distance from the baseline. The observed series alone would have
produced the opposite decision.

### 6.7 What the case does not establish

The case does not show that an outbreak is occurring. It shows that the observed decline
is an artefact and that the adjusted trend is upward. Whether the upward trend exceeds
the seasonal baseline requires a baseline model with several years of data, and whether
it warrants an investigation requires case-level review. The arithmetic establishes the
direction and the size of the correction, not the public health conclusion.

## 7. Code, query, or calculation

The block below implements the delay-adjustment nowcast, the weighted interval score,
and the comparison with a persistence baseline. It is standard-library Python and runs
as written.

### 7.1 Nowcast and scoring

```python
"""Reporting-triangle nowcast and weighted interval score. Standard library only."""

DELAY = [0.70, 0.20, 0.07, 0.03]   # share reported at delay 0,1,2,3 weeks

# counts[week] = [reported at delay 0, 1, 2, 3]; trailing zeros are not yet reported
TRIANGLE = {
    40: [90, 0, 0, 0],
    41: [70, 20, 7, 3],
    42: [77, 22, 8, 0],
    43: [84, 0, 0, 0],
}


def nowcast(counts, delay=DELAY):
    observed = sum(counts)
    seen = 0.0
    for i, c in enumerate(counts):
        if c > 0 or i == 0:
            seen = sum(delay[: i + 1])
    return observed, observed / seen, seen


for week in (41, 42, 43):
    obs, est, frac = nowcast(TRIANGLE[week])
    print(f"week {week}: observed={obs} delay_covered={frac:.2f} nowcast={est:.1f}")


def interval_score(y, lo, hi, alpha):
    s = (hi - lo)
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    if y > hi:
        s += (2.0 / alpha) * (y - hi)
    return s


def wis(y, median, intervals):
    """intervals: list of (lo, hi, alpha). K = len(intervals)."""
    k = len(intervals)
    total = 0.5 * abs(y - median)
    for lo, hi, alpha in intervals:
        total += (alpha / 2.0) * interval_score(y, lo, hi, alpha)
    return total / (k + 0.5)


actual = {42: 110, 43: 120}
fc = {42: (110.3, 104.0, 117.0), 43: (120.0, 108.0, 132.0)}
intervals = [(104.0, 117.0, 0.20), (104.0, 117.0, 0.05)]
for w in (42, 43):
    med = fc[w][0]
    print(f"week {w}: WIS(nowcast)={wis(actual[w], med, intervals):.2f} "
          f"WIS(persistence)={wis(actual[w], 84, intervals):.2f}")

print("naive change 42->43:", round(100 * (84 - 107) / 107, 1), "%")
print("nowcast change 42->43:", round(100 * (120 - 110.3) / 110.3, 1), "%")
print("delay 0.60 nowcast w43:", round(84 / 0.60, 1))
```

Expected output:

```text
week 41: observed=100 delay_covered=1.00 nowcast=100.0
week 42: observed=107 delay_covered=0.97 nowcast=110.3
week 43: observed=84 delay_covered=0.70 nowcast=120.0
week 42: WIS(nowcast)=0.71 WIS(persistence)=5.85
week 43: WIS(nowcast)=3.05 WIS(persistence)=10.25
naive change 42->43: -21.5 %
nowcast change 42->43: 8.8 %
delay 0.60 nowcast w43: 140.0
```

The WIS values are lower for the nowcast at both horizons, which is the minimum
requirement for a nowcast to be preferred over carrying the last value forward. Note
that the persistence forecast in this comparison uses the last observed value, 84, which
is itself incomplete; a fairer baseline would use the last nowcast. The comparison shown
is the one most dashboards implicitly make.

### 7.2 What the code does not do

The estimator assumes a constant delay distribution. It does not model day-of-week or
holiday effects, does not propagate uncertainty from the delay distribution into the
interval, and does not adjust for exponential growth. The interval score is computed
against fixed intervals supplied by the analyst, not derived from the model. These are
teaching simplifications, and each one is a source of error in a real nowcast.

A production nowcast would estimate the delay distribution hierarchically, propagate
uncertainty by simulation or Bayesian inference, and be backtested over at least 12
weeks using only the data available at each vintage.

## 8. Data, standards, and regulatory context

### 8.1 Surveillance systems and what they measure

The National Notifiable Diseases Surveillance System (NNDSS) aggregates weekly reports
from state and territorial health departments. The CDC notice to data users, updated in
May 2026, states that weekly tables are provisional and subject to change until
reconciled annually, that cumulative counts can decrease as well as increase, and that
provisional counts rarely match final counts. The 2026 publication cycle added
COVID-19-associated pediatric mortality and Oropouche as nationally notifiable
conditions, retired the dengue-like illness event code, and restricted syphilis
publication to confirmed and probable cases under CSTE position statement 25-ID-08.

The National Healthcare Safety Network (NHSN) collects healthcare-associated infection
data from participating facilities. Its denominator is patient-days or device-days, not
admissions, and its measures are specified in the NHSN surveillance manual rather than
in statute.

The National Wastewater Surveillance System (NWSS) publishes pathogen concentrations from
sampling sites. CDC's data methodology, updated in August 2025, states that the
wastewater viral activity level is calculated from non-normalised concentrations against
a 24-month baseline, that sites need at least eight weeks of same-method data before
public posting, and that aggregation proceeds by median from site to state to region to
nation. The methodology notes that median activity may not represent every community
because site density is uneven. Earlier CDC guidance described flow and population
normalisation; the shift is itself a reminder that the method is versioned.

### 8.2 Case definitions and coding

CSTE position statements define each notifiable condition, with classification levels
and laboratory criteria. The definitions are revised annually, and a revision can change
both the numerator and the trend. ICD-10-CM codes are used for diagnosis and cause of
death; ICD-11 is used in some jurisdictions and is not directly mappable for all
conditions. LOINC codes identify laboratory tests, and a change in the LOINC mapping can
change which tests count toward a definition.

What this constrains: the case definition and the code set are part of the measurement,
and any multi-year series must record them. What it does not constrain: the choice of
indicator, which is a decision problem.

### 8.3 Privacy and legal context

In the United States, HIPAA permits disclosures to public health authorities for
specified purposes under 45 CFR 164.512(b). The permission is narrower than it is often
assumed to be: it covers public health activities, not every research or commercial use
of the same data. State laws add reporting requirements and, in some cases, stronger
confidentiality protections, for example for HIV, substance use, and mental health.

Small-cell suppression is a privacy control, and it interacts with analysis. A suppressed
cell is not zero. Maps and tables that render suppression as zero create false declines
in rural areas, where cells are small precisely because the population is small.

### 8.4 Forecasting and evaluation standards

The COVID-19 Forecast Hub established a standard evaluation practice: probabilistic
forecasts, scored with the weighted interval score, compared against a persistence
baseline, with coverage reported alongside the score. CDC's FluSight programme uses a
similar framework. The standard matters because a point forecast can look accurate while
its intervals are meaningless, and because a model that cannot beat persistence has no
operational value.

### 8.5 Regulatory and authority boundaries

Public health powers are largely state-based in the United States. Mandatory reporting,
isolation, quarantine, and data collection authority vary by jurisdiction and must be
grounded in statute or regulation. An analytic product that informs a compulsory action
carries a different evidentiary burden from one that informs a voluntary advisory. The
distinction is legal, not statistical, and it should be settled with counsel before a
product is deployed.

## 9. Industry practice and case evidence

### 9.1 Ensemble forecasting as operational practice

The COVID-19 Forecast Hub (Cramer et al., PNAS, 2022) combined independent models into
an ensemble that outperformed most individual models for US COVID-19 mortality
forecasts. Documented fact: the ensemble had lower average scores than the median
contributor. Inference: ensembles are a robust operational strategy under model
uncertainty, because they do not depend on identifying the single best model in advance.
What did not transfer: the ensemble's performance depended on contributor diversity and
on continued investment in data and evaluation, and a forecast is not a causal claim.

### 9.2 Wastewater as a testing-independent signal

CDC launched NWSS in 2020 to detect SARS-CoV-2 in wastewater, and later extended it to
other pathogens. The signal is independent of clinical testing, which is its value when
testing declines. Documented fact: the programme publishes site-level concentrations and
a standardised activity level with a documented methodology. Inference: wastewater can
fill a gap when clinical surveillance is thin. What did not transfer: wastewater does not
identify individuals, does not measure prevalence directly, and its population mapping
depends on sewer infrastructure and site coverage.

### 9.3 Probability sampling as the denominator

The UK Office for National Statistics COVID-19 Infection Survey sampled households
repeatedly to estimate infection prevalence, providing a denominator that case counts
could not. Documented fact: repeated probability sampling produced prevalence estimates
with design-based uncertainty. Inference: when the testing system is biased, a dedicated
sample can correct the denominator. What did not transfer: the survey was expensive,
had response bias, and lagged real-time decisions.

### 9.4 National linked data and rapid evidence

Israel's nationwide vaccination study (Dagan et al., New England Journal of Medicine,
2021) linked national records to compare vaccinated and unvaccinated members during a
mass campaign. Documented fact: the study reported substantial protection against
infection and severe disease. Inference: a national linked dataset can produce rapid,
high-impact evidence. What did not transfer: observational confounding, healthy-vaccinee
effects, and a specific context limit generalisation.

### 9.5 Environmental detection of poliovirus

In 2022, vaccine-derived poliovirus was detected in sewage in London and in New York
State, triggering booster campaigns and investigation. Documented fact: environmental
sampling detected the virus before clinical cases were identified. Inference:
wastewater can extend sensitivity beyond clinical surveillance. What did not transfer:
detection does not prove ongoing transmission, and the response depended on vaccination
coverage and public trust.

### 9.6 What the cases have in common

Each case is a measurement intervention, not a model. The forecast hub changed how
forecasts were scored and combined; NWSS changed what was sampled; the infection survey
changed who was sampled; the Israeli study changed which records were linked; the polio
response changed where surveillance looked. In every case, the analytic contribution was
to make a denominator or a comparison credible, and the modelling was secondary.

A practical test for a vendor or internal claim in this space is to ask which of the
five measurement interventions the product performs. If the answer is none, and the
product is a model applied to an unchanged surveillance stream, then its value depends
entirely on discrimination that the stream already limits. That is not a reason to
reject it; it is a reason to demand an evaluation against the persistence baseline and a
statement of the denominator the product uses.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Last-point fallacy | Final one to three points decline in every vintage | Incomplete reporting plotted as observed | False stand-down decision | Nowcast or truncate the series |
| Case-definition drift | Step change at a publication date | Definition revised mid-series | Trend attributed to transmission | Version the definition; annotate |
| Vintage mixing | Two charts of the same week disagree | Different extraction dates compared | Spurious change | Record and match vintages |
| Wrong denominator | Rates differ from neighbouring jurisdictions by a constant factor | Population versus patient-days versus tests | Comparison is invalid | State and match the denominator |
| Crude rate comparison | A state with older population looks worse on every cause | Age structure differs | Misallocated resources | Age-standardise; name the standard |
| Suppression treated as zero | Rural areas show implausibly low rates | Small cells withheld | False declines in small areas | Publish suppression rule; model small areas |
| Testing-dependent positivity | Positivity falls while cases rise | Testing expanded to asymptomatic people | Misread as declining transmission | Report testing volume with positivity |
| Event-date imputation | Epidemic curve has artificial weekly spikes | Onset imputed from report date | Distorted peak timing | Flag imputed dates; report share |
| Duplicate reports | Counts fall at reconciliation | Same case reported twice or transferred | Overcount then revision | Deduplicate on stable identifier |
| Delay distribution drift | Nowcast errors are one-sided | Laboratory or process change | Biased current estimate | Re-estimate on recent weeks |
| Ecologic inference | Tract-level correlation presented as individual risk | Group and individual levels conflated | Mis-targeted intervention | Use individual data or state the limitation |
| Rt without generation interval | Rt changes when the interval assumption changes | Model-dependent estimate | False precision | Report the assumption and interval |
| Baseline chosen post hoc | Excess mortality appears only under one baseline | Analyst selected the baseline after seeing data | Manufactured signal | Pre-specify the baseline |
| Coverage denominator mismatch | Coverage exceeds 100 percent | Eligible population understated | Implausible estimates | Reconcile numerator and denominator |
| Alert threshold with no stand-down | Alert remains on after the wave | One-sided rule | Alert fatigue | Define de-escalation criteria |
| Unmarked laboratory change | Delay-0 share jumps | New reporting system | Nowcast bias | Annotate and re-estimate delays |

The two most reliable red flags are structural. First, a series whose final point always
falls is plotting incomplete data, and the fix is a nowcast or a truncation, never a
smoother. Second, a rate that differs from a neighbour by a suspiciously round factor is
usually a denominator mismatch, not a real difference.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The case definition is constant over the series | Trend interpretation requires a fixed instrument | Annual CSTE revisions; local amendments | Stratify by definition version |
| Reporting delay distribution is stable | Nowcast divides by a fixed proportion | Laboratory onboarding; holiday weeks | Compare recent and historical distributions |
| Report date is recorded consistently | Delay is measured from it | Batch uploads; weekend processing | Inspect delay histogram by weekday |
| Event dates are reported, not imputed | The epidemic clock requires onset | Missing onset fields | Report the imputed share |
| Cases are deduplicated | Counts represent people | Transfers and repeat reports | Audit a sample of identifiers |
| The denominator matches the numerator geography | Rates require the same population | Residence versus reporting site | Reconcile geographies |
| Population estimates are current | Rates use the right denominator | Census revisions; ACS margins | Record the vintage |
| Testing access is unrelated to risk given covariates | Positivity and case counts are interpretable | Targeted testing of contacts | Report testing rates by group |
| Small-cell suppression is not informative | Totals and maps are unbiased | Suppression correlated with rurality | Model suppressed cells explicitly |
| Forecast errors are scored against a baseline | Model value is relative | Baseline omitted | Report WIS versus persistence |
| Case counts are not a census of infections | IFR and prevalence require adjustment | Under-ascertainment varies | Use serology or capture-recapture |

The largest limit is under-ascertainment. Surveillance counts people who sought care,
were tested, and were reported. The ratio of counted to true infections varies by
pathogen, access, and phase of the epidemic, and it is not identifiable from case data
alone. Any statement about burden must either state the ascertainment assumption or use
a design that estimates it.

## 12. Exercises

### Exercise 1 — Build a reporting triangle (application, 40 minutes)

**Deliverable.** A triangle from a supplied case-level extract, with delay distribution
and a note on incomplete cells.

**Acceptance criteria.** Cells not observable at the vintage are marked missing, not
zero; the distribution is estimated from complete weeks only; the maximum delay is
justified.

### Exercise 2 — Nowcast and backtest (application, 50 minutes)

**Deliverable.** A nowcast for the last three weeks and a backtest over 12 vintages
against a persistence baseline, with WIS.

**Acceptance criteria.** Only data available at each vintage are used; the baseline is
reported; interval coverage is reported; the failure mode is explained if the nowcast
does not beat persistence.

### Exercise 3 — Denominator audit (critique, 35 minutes)

**Deliverable.** A table of five indicators with the correct denominator, the wrong
denominator commonly used, and the direction of bias.

**Acceptance criteria.** At least one indicator uses patient-days, one uses live births,
one uses tests, and one uses person-time; each bias direction is justified.

### Exercise 4 — Age standardisation (application, 40 minutes)

**Deliverable.** Crude and directly standardised rates for two populations with different
age structures, using a named standard population.

**Acceptance criteria.** The arithmetic is shown; the standard population is named; the
write-up explains why the crude comparison reversed.

### Exercise 5 — Detect a definition change (critique, 35 minutes)

**Deliverable.** A short memo identifying a step change, the likely definition or coding
cause, and two analyses that would distinguish it from a real change.

**Acceptance criteria.** The memo names the definition version and date; it proposes a
bridged series; it states what data would be required.

### Exercise 6 — Revision analysis (application, 40 minutes)

**Deliverable.** A revision table showing how the estimate for week t changed at vintages
t+1, t+2, and t+4, with the percentage revision.

**Acceptance criteria.** Revisions are computed on the same definition; decreases are
included; the write-up explains why a provisional count is not a final count.

### Exercise 7 — Suppression and small areas (design, 35 minutes)

**Deliverable.** A map specification and a suppression policy for a county-level
indicator, with a plan for handling suppressed cells.

**Acceptance criteria.** Suppressed cells are not rendered as zero; the policy states the
threshold; the plan includes a small-area model or an explicit "insufficient data"
category.

### Exercise 8 — Ecologic versus individual design (design, 40 minutes)

**Deliverable.** A one-page design memo for a question about an environmental exposure
and a health outcome, comparing an ecologic and an individual-level design.

**Acceptance criteria.** The memo states the ecologic fallacy risk; it names the
confounders that differ between designs; it recommends one design with a justification.

## 13. Self-check answers

**1. Delay distribution.** From complete weeks only, the share reported at each delay is
the count at that delay divided by the total for the week. In the worked example the
distribution is 0.70, 0.20, 0.07, 0.03. Estimating it from incomplete weeks
under-estimates the long delays, because those cells have not been observed.

**2. Nowcast arithmetic.** Week 43 observed 84 and delay 0 covers 0.70 of eventual
reports, so the estimate is 84 / 0.70 = 120.0. Week 42 observed 107 with delays 0 to 2
covering 0.97, so the estimate is 107 / 0.97 = 110.3.

**3. Denominator for hospital-onset infection.** Patient-days, not admissions and not
population. The risk period is the days spent in hospital with a device or exposure, and
admissions vary in length.

**4. Age standardisation.** Direct standardisation applies the age-specific rates of each
population to a common standard age structure. If the crude rate is higher in the older
population but the age-specific rates are similar, the standardised rates converge. The
standard population must be named, and it must be held fixed across comparisons.

**5. Definition change detection.** Look for a step change at a publication date that is
not accompanied by a change in other indicators, and check the definition version and
laboratory criteria. A real change usually appears in related indicators and in
neighbouring jurisdictions at the same time; an administrative change appears only where
the definition changed.

**6. Why provisional is not final.** Provisional counts reflect reports received by the
vintage. Reports arrive late, duplicates are removed, and classifications are revised.
The CDC NNDSS notice states that cumulative counts can decrease as well as increase and
that provisional counts rarely match final counts.

**7. Suppression bias.** Treating a suppressed cell as zero understates the count and the
rate in small areas. Because suppression is more common in rural and small-population
areas, the bias is systematic and produces a false urban-rural gradient.

**8. Ecologic design.** Use an ecologic design for hypothesis generation or when
individual exposure data do not exist, and state that it cannot support individual-level
inference. Use an individual-level design when the decision concerns individuals and the
data can be linked.

**Rubric for Exercises 5, 7, and 8.** A strong answer names the mechanism, shows the
direction of bias, and proposes a test that would distinguish the competing
explanations. A weak answer asserts that the data are "adjusted" without saying what was
adjusted or which assumption the adjustment relies on.

## 14. Interview questions

**1. Why do the last two weeks of a case curve often decline?**
Strong answer: reporting delay and backfill, with a nowcast or truncation as the fix.
Weak answer: "reporting lag" with no implication for the chart.

**2. What is the denominator for a hospital-onset infection rate?**
Strong answer: patient-days or device-days, matched to the numerator. Weak answer:
admissions, which varies with length of stay.

**3. Cases rose but testing also rose. What can you conclude?**
Strong answer: little without positivity and testing access; incidence may be flat.
Weak answer: "cases are up, so transmission is up".

**4. What is the ecologic fallacy?**
Strong answer: inferring individual risk from group-level associations, with an example
and a design fix. Weak answer: "correlation is not causation", which is a different
problem.

**5. How do you compare mortality across states with different age structures?**
Strong answer: age-standardise, name the standard population, show crude and standardised
side by side. Weak answer: "adjust for age" without the method.

**6. What does a wastewater signal measure?**
Strong answer: pathogen shed in a sewershed, not diagnosed infections, with normalisation
and coverage caveats. Weak answer: "cases in the sewer".

**7. How would you evaluate a vaccine's real-world effectiveness?**
Strong answer: test-negative design or cohort with confounding control, reported by
variant, age, and prior infection. Weak answer: compare vaccinated and unvaccinated case
counts.

**8. Why is Rt model-dependent?**
Strong answer: it depends on the generation interval, reporting delay, and immunity
assumptions, so it should be reported with those assumptions. Weak answer: "it is
estimated from cases".

**9. When is a forecast useful even if it is wrong?**
Strong answer: when it quantifies uncertainty and supports a decision with asymmetric
costs, scored against a baseline. Weak answer: "when it is close".

**10. What are the privacy risks of small-area public health data?**
Strong answer: re-identification of neighbourhoods or facilities, with suppression and
aggregation as controls. Weak answer: "HIPAA covers it", without the suppression rule.

## 15. Key terms

- **Notifiable disease** — condition that law or regulation requires to be reported.
- **Case definition** — criteria that determine whether a person counts as a case.
- **CSTE** — Council of State and Territorial Epidemiologists, publisher of US case definitions.
- **Event date** — date the event occurred, usually onset or specimen collection.
- **Report date** — date the jurisdiction received the report.
- **Vintage** — date and time a dataset was extracted.
- **Reporting triangle** — counts by event date and reporting delay.
- **Backfill** — late reports that raise historical counts.
- **Nowcast** — estimate of the current value corrected for reporting delay.
- **Delay distribution** — share of cases reported at each delay.
- **Persistence baseline** — forecast equal to the last observed value.
- **WIS** — weighted interval score for probabilistic forecasts.
- **Incidence** — new cases per person-time at risk.
- **Prevalence** — existing cases per population at a point or period.
- **Case fatality ratio** — deaths divided by diagnosed cases.
- **Infection fatality ratio** — deaths divided by all infections.
- **Test positivity** — positive tests divided by tests performed.
- **Rt** — effective reproduction number.
- **Excess mortality** — observed minus expected deaths.
- **Age standardisation** — adjusting rates to a common age structure.
- **SMR** — standardised mortality ratio, observed divided by expected.
- **Ecologic fallacy** — inferring individual risk from group-level associations.
- **Capture-recapture** — estimating an unobserved total from overlapping sources.
- **Syndromic surveillance** — monitoring pre-diagnostic signals.
- **Sewershed** — catchment area of a wastewater sampling site.
- **WVAL** — wastewater viral activity level.
- **Suppression** — withholding small counts to protect privacy.
- **SVI** — CDC/ATSDR Social Vulnerability Index.

## 16. Sources

- CDC. National Notifiable Diseases Surveillance System, notice to data users and publication criteria (updated 14 May 2026). https://www.cdc.gov/nndss/infectious-disease/notice-to-data-users.html (accessed 26 October 2026).
- CDC. Notifiable infectious disease data tables. https://www.cdc.gov/nndss/infectious-disease/ (accessed 26 October 2026).
- CDC. About notifiable infectious diseases and conditions data. https://www.cdc.gov/nndss/notifiable-infectious-disease-tables/about-data.html (accessed 26 October 2026).
- CDC. 2026 state epidemiology letter, NNDSS changes. https://ndc.services.cdc.gov/wp-content/uploads/2026_StateEpiLetter_NNDSS_Changes.pdf (accessed 26 October 2026).
- CSTE. Surveillance case definitions for current and historical conditions. https://ndc.services.cdc.gov/ (accessed 26 October 2026).
- CDC. National Wastewater Surveillance System data methodology (updated 15 August 2025). https://www.cdc.gov/nwss/data-methods.html (accessed 26 October 2026).
- CDC. About wastewater data. https://www.cdc.gov/nwss/about-data.html (accessed 26 October 2026).
- CDC. National Healthcare Safety Network. https://www.cdc.gov/nhsn/ (accessed 26 October 2026).
- CDC WONDER. https://wonder.cdc.gov/ (accessed 26 October 2026).
- HHS. HIPAA public health exception, 45 CFR 164.512(b). https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/public-health/index.html (accessed 26 October 2026).
- Cramer EY, et al. Evaluation of individual and ensemble probabilistic forecasts of COVID-19 mortality in the United States. PNAS. 2022. https://doi.org/10.1073/pnas.2112532119 (accessed 26 October 2026).
- Dagan N, et al. BNT162b2 mRNA Covid-19 vaccine in a nationwide mass vaccination setting. N Engl J Med. 2021. https://doi.org/10.1056/NEJMoa2101765 (accessed 26 October 2026).
- UK Office for National Statistics. COVID-19 Infection Survey. https://www.ons.gov.uk/surveys/informationforhouseholdsandindividuals/householdandindividualsurveys/covid19infectionsurvey (accessed 26 October 2026).
- CDC/ATSDR. Social Vulnerability Index. https://www.atsdr.cdc.gov/place-health/php/svi/ (accessed 26 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Previous topic: [`05_pediatrics-and-obstetrics.md`](05_pediatrics-and-obstetrics.md)
- Next topic: [`07_cross-specialty-comparison.md`](07_cross-specialty-comparison.md)
- Related topic: [`04_critical-care.md`](04_critical-care.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Study design: [`../05_study_design/`](../05_study_design/)
- Epidemiology foundations: [`../08_ehr_claims/`](../08_ehr_claims/)
