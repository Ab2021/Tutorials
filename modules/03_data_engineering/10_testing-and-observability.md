# Testing and Observability

> Module 03 — Data Engineering and Interoperability
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02, 04, 07
> Version 2026.3

## 1. Why this topic exists

A pipeline that completes successfully has proven that the code ran. It has not proven that the data are
correct. The distinction is the source of most silent data incidents: the job is green, the dashboard
refreshes, and the number is wrong.

The failures are quiet.

- A laboratory feed starts sending null units. The loader accepts them, the range check skips nulls, and
  the cohort's baseline lab rate falls by 6%. Nobody notices until a clinician asks why.
- A source system changes a date field from `admit_date` to `admission_date`. The loader maps the missing
  column to null, the pipeline succeeds, and every length-of-stay calculation returns null.
- A code set update adds 1,200 patients to the cohort. The row count rises by 16%, which is within the
  "expected growth" band because the band was set at 20%.
- A partition is written twice after a retry. The row count rises by 0.8%, which passes the volume check,
  and every rate is slightly inflated.
- A model's input distribution shifts because a new site joined the network. The model's calibration
  degrades, but no monitor compares the input distribution to the training distribution.
- An alert fires every night for a check that has been failing for months. The on-call engineer mutes it,
  and the next real failure is also muted.

Testing and observability are the practices that make these failures visible. Testing checks the code and
the data before release; observability checks the running system and the data after release. They are
different mechanisms with the same purpose: to convert a silent failure into a signal with an owner and an
action.

This topic closes the module. It assumes the cohort logic from Topic 02, the orchestration semantics from
Topic 04, and the versioning and lineage from Topic 07. A test without a versioned input is not
reproducible; an alert without lineage cannot be scoped.

> **Note:** A test suite is not a quality guarantee. It encodes the checks the team thought of. The
> purpose of this topic is to make the checks systematic, measurable, and owned, not to claim that a
> passing suite proves correctness.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** data tests by level (column, row, table, cross-table, distribution) and choose the right
   level for a given risk. Assessed in Exercise 1.
2. **Write** schema, uniqueness, referential, temporal, and range tests for a cohort table. Assessed in
   Exercise 2 and self-check item 3.
3. **Compute** freshness and volume metrics and set warn and error thresholds. Assessed in Exercise 3.
4. **Detect** distribution shift with a robust z-score and explain why the mean and the median can
   disagree. Assessed in Exercise 4 and self-check item 5.
5. **Calibrate** alert thresholds from a target false-positive rate and compute the annual alert volume.
   Assessed in Exercise 5.
6. **Define** a service-level objective and an error budget for a data product. Assessed in Exercise 6.
7. **Design** a runbook that maps each alert to an owner, a diagnostic query, and an action. Assessed in
   Exercise 7.
8. **Diagnose** a silent failure from monitoring signals and trace it to a root cause. Assessed in
   Exercise 8.

## 3. The core mental model

Observability for data has five signals and three levels of testing.

```text
SIGNALS                                  TESTS
+----------------------------+           +-----------------------------+
| 1. Freshness               |           | Level 4: distribution       |
|    when did data arrive?   |           |   shift, anomaly            |
+----------------------------+           +-----------------------------+
| 2. Volume                  |           | Level 3: cross-table        |
|    how many rows/bytes?    |           |   referential, reconciliation|
+----------------------------+           +-----------------------------+
| 3. Schema                  |           | Level 2: table              |
|    did the shape change?   |           |   uniqueness, row count     |
+----------------------------+           +-----------------------------+
| 4. Distribution            |           | Level 1: column/row         |
|    did the values change?  |           |   null, range, accepted     |
+----------------------------+           +-----------------------------+
| 5. Lineage                 |           |                             |
|    what depends on this?   |           |                             |
+----------------------------+           +-----------------------------+
```

The signals answer "is the data product healthy?" The tests answer "does this dataset meet its contract?"
A mature pipeline uses both: tests at build time, signals at run time.

### 3.1 The test pyramid for data

```text
                 /\
                /  \        end-to-end: 2-5
               /----\       full pipeline on a fixture, compare output
              /      \
             /--------\     cross-table: 10-30
            /          \    referential integrity, reconciliation, temporal
           /------------\
          /              \  column and row: 100+
         /                \ nulls, ranges, accepted values, formats
        /------------------\
```

Most tests should be cheap and specific: a null check on a critical column, a range check on a vital sign,
an accepted-values check on a status field. Cross-table tests are fewer and catch integration errors.
End-to-end tests are few and expensive.

The common mistake is to rely on a small number of end-to-end tests that compare a full output to a
golden file. When they fail, the diagnosis is slow, and the golden file is often updated to make the test
pass, which erases the signal.

### 3.2 Tests versus monitors

A test runs at build time against a known input and has a pass/fail result. A monitor runs against
production data and produces a metric over time. The same logic can appear in both: a null-rate check can
be a test with a threshold and a monitor with a time series.

Tests are deterministic and block a release. Monitors are statistical and trigger an investigation. A team
needs both because some failures are introduced by code (caught by tests) and some by data (caught by
monitors).

### 3.3 Freshness and completeness are different

Freshness asks when the data arrived. Completeness asks whether all the expected data arrived. A partition
can be fresh and incomplete: a feed delivers 60% of its rows on time and the rest two days later. A
freshness check passes; a volume check may pass if the daily variation is wide; a completeness check
against the source count fails.

The three checks are complementary. A pipeline that monitors only freshness will miss missing rows; a
pipeline that monitors only volume will miss stale data.

### 3.4 Alerting is a design problem

An alert is a signal plus an owner plus an action. Without an owner, it is a notification. Without an
action, it is noise. The design questions are:

- What is the threshold, and what false-positive rate does it produce?
- Who is paged, and at what hour?
- What is the first diagnostic query?
- What is the containment action?
- When is the alert resolved, and by whom?

A team that cannot answer these for an alert should not enable it. A muted alert is worse than no alert
because it trains the team to ignore the channel.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Data test | Assertion about data with a pass/fail result | Confused with a unit test | — |
| Monitor | Continuous metric with thresholds | Confused with a test | — |
| Freshness | Age of the most recent data | Confused with completeness | Minutes or hours |
| Completeness | Share of expected records present | Confused with freshness | Percent |
| Volume | Row or byte count | Confused with completeness | Rows, bytes |
| Schema drift | Unplanned structural change | Confused with a value change | — |
| Distribution shift | Change in the value distribution | Confused with a schema change | — |
| Anomaly | Value outside the expected pattern | Confused with a bug | — |
| Robust z-score | Deviation scaled by median absolute deviation | Confused with a standard z-score | — |
| MAD | Median absolute deviation | Confused with standard deviation | — |
| Threshold | Boundary for pass, warn, or error | Chosen without calibration | — |
| False positive | Alert with no real issue | Confused with a failed test | Count or rate |
| False negative | Missed real issue | Ignored in calibration | Count or rate |
| SLO | Service-level objective | Confused with an SLA | Percent |
| SLI | Service-level indicator | Confused with an SLO | Metric |
| SLA | Contractual service level | Confused with an internal SLO | Contract |
| Error budget | Allowed unreliability | Confused with a target | Percent or time |
| Alert fatigue | Desensitization from too many alerts | Confused with high volume | — |
| Runbook | Procedure for an alert | Confused with documentation | — |
| Ownership | Named person or team responsible | Confused with a team name | — |
| Data contract | Executable schema and quality agreement | Confused with a document | — |
| dbt test | SQL assertion in a dbt project | Confused with a unit test | — |
| Source freshness | dbt check on source `loaded_at` | Confused with model freshness | — |
| Expectation | Great Expectations assertion | Confused with a dbt test | — |
| Checkpoint | Great Expectations validation run | Confused with a test suite | — |
| Reconciliation | Comparison of two counts | Confused with a test | Count |
| Golden dataset | Known-good fixture for regression | Confused with production data | — |
| Canary | Small-scale run before full release | Confused with a test | — |
| Lineage | Dependency graph | Confused with a DAG diagram | — |
| Root cause | Underlying cause of an incident | Confused with the first symptom | — |
| Blast radius | Scope of impact | Confused with severity | — |
| Observability | Ability to infer internal state from outputs | Confused with monitoring | — |

Three terms need expansion.

**Robust z-score.** The standard z-score uses the mean and standard deviation, both of which are sensitive
to outliers. The robust z-score uses the median and the MAD, scaled by 1.4826 to approximate a standard
deviation for normal data. It is the right tool for daily metrics that occasionally spike.

**Error budget.** If an SLO is 99.5% over 30 days, the error budget is 0.5% of 30 days, or 3.6 hours. The
budget converts reliability into a quantity that can be spent on changes. A team that exhausts the budget
pauses feature work and fixes reliability; a team that never tracks it has no basis for that decision.

**Data contract.** In dbt, a contract enforces column names and types at build time. In a broader sense, a
contract includes freshness, nullability, accepted values, and ownership. A contract that enforces types
but not freshness leaves the consumer exposed to stale data.

## 5. The mechanics, step by step

The procedure builds a test and observability layer for a clinical data product. It assumes a daily
pipeline and a cohort table consumed by a dashboard and a model.

### Step 1 — List the data products and their consumers

Input: the pipeline inventory.
Output: a table of data products, consumers, and decision deadlines.

Example: `gold.hf_cohort` feeds a quality dashboard at 06:00 and a model training job weekly. The dashboard
needs freshness and completeness; the model needs distribution stability and schema stability.

Reason: the checks follow the consumer. A dataset with no consumer does not need an alert; a dataset that
feeds a clinical decision needs one with an owner and a deadline.

### Step 2 — Write the data contract

Input: the consumer requirements.
Output: a contract with schema, grain, primary key, nullability, accepted values, freshness, and owner.

The contract is the specification that tests enforce. It should be versioned and reviewed by the consumer,
not only by the producer.

Reason: a test without a contract encodes the producer's assumptions. A contract makes the assumptions
explicit and gives the consumer a basis for complaint.

### Step 3 — Declare the grain and primary key

Input: the data model.
Output: a documented grain and a uniqueness test.

For a cohort table, the grain is one row per person per index admission, and the key is
`(person_id, index_encounter_id)`. The uniqueness test fails if the key is duplicated.

Reason: grain errors are the most common cause of inflated counts, and a uniqueness test is the cheapest
way to catch them.

### Step 4 — Test nullability on critical columns

Input: the contract.
Output: not-null tests on the columns that cannot be null.

Critical columns: identifiers, index date, outcome flag, and any column used in a join or a denominator.
Non-critical columns may be nullable with a documented maximum null rate.

Reason: a null in a join key silently drops rows; a null in a denominator produces a wrong rate. Not-null
tests catch both at build time.

### Step 5 — Test accepted values and ranges

Input: the clinical definitions.
Output: accepted-values tests for categorical fields and range tests for numeric fields.

Vitals: heart rate 20-300 per minute, systolic blood pressure 40-300 mmHg, temperature 25-45 Celsius.
Labs: creatinine 1-4,000 µmol/L. Categories: sex, encounter type, discharge disposition.

Reason: range checks catch unit errors, sign errors, and placeholder values such as -1 or 9999. The bounds
must come from clinical knowledge, not from the observed minimum and maximum, or the test learns the
errors.

### Step 6 — Test referential integrity

Input: the relationships between tables.
Output: relationship tests from child to parent.

Every encounter has a person; every cohort row has an encounter; every lab has a person. The test counts
orphans and fails above zero, or above a documented tolerance if the source is known to have orphans.

Reason: an orphan row is a row that cannot be attributed. It may still be clinically valid, but it cannot
be joined, and it often indicates an extraction or mapping error.

### Step 7 — Test temporal ordering

Input: the time columns.
Output: tests for impossible sequences.

Discharge after admission; death after birth; lab after encounter start minus a window; index date inside
the study period; follow-up time non-negative. Report the count and the affected persons.

Reason: temporal errors are common in interface data and produce impossible features. A negative length of
stay is both a data error and a clinical impossibility.

### Step 8 — Test cross-table reconciliation

Input: the source and target counts.
Output: a reconciliation test with an expected tolerance.

The number of cohort persons should equal the number of distinct persons in the index events. The number
of loaded observations should equal the extracted count minus duplicates and quarantines. The test
compares and explains the difference.

Reason: reconciliation catches extraction and join errors that row-level tests cannot see. A pipeline can
pass every column test and still lose 4% of rows in a join.

### Step 9 — Test distributions and drift

Input: a baseline distribution.
Output: a distribution test with a threshold and a metric.

For a numeric feature, compare the current mean, median, and 99th percentile to the baseline. For a
categorical feature, compare the share of each category. Use a robust statistic and a threshold calibrated
to the natural variation.

Reason: a value can be in range and still be wrong for the population. A creatinine of 300 µmol/L is
plausible for one patient and implausible as a population median.

### Step 10 — Choose the test execution point

Input: the pipeline stages.
Output: a mapping from test to stage: source, staging, mart, or release.

Source tests check schema and freshness. Staging tests check types, ranges, and referential integrity. Mart
tests check grain, business rules, and reconciliation. Release tests compare to the previous version.

Reason: a test at the wrong stage either fails too late or blocks on data that are expected to be dirty at
that point. Source data are often dirty; the staging layer is where cleaning is verified.

### Step 11 — Fail the build on error, warn on drift

Input: the test results.
Output: a severity policy.

Structural errors (missing column, duplicate key, null identifier) fail the build. Statistical warnings
(distribution shift, row-count change within tolerance) warn and notify. The policy must state which is
which.

Reason: treating every warning as an error causes alert fatigue; treating every error as a warning lets
bad data through. The severity is a design decision.

### Step 12 — Measure freshness

Input: the source's load timestamp.
Output: a freshness metric and thresholds.

Compute the age of the most recent record at the time of the check. Set a warning threshold and an error
threshold based on the consumer's deadline and the source's normal cadence.

Reason: freshness is the simplest signal and the most actionable. A stale source is usually a broken
interface or a failed extraction, and the fix is known.

### Step 13 — Measure volume

Input: the row count per partition.
Output: a volume metric and a comparison to a baseline.

Compare to the previous run, the same day last week, and a rolling median. Use both absolute and relative
differences. A 0.8% increase is noise on a normal day and a signal after a retry.

Reason: volume catches missing and duplicated partitions. The comparison window matters: a Monday is
different from a Sunday, and a holiday is different from both.

### Step 14 — Detect schema drift

Input: the table schema.
Output: a diff against the contract and an alert on unexpected change.

New columns, removed columns, type changes, and nullability changes are all drift. A new column is usually
benign but should be reviewed; a removed or retyped column is a breaking change.

Reason: schema drift breaks downstream queries silently when the query uses `SELECT *` or a positional
load. A diff catches it before the consumer does.

### Step 15 — Compute a robust anomaly score

Input: a metric time series.
Output: a robust z-score per observation.

Use the median and MAD over a trailing window (for example, 28 days), excluding the current point. Flag
when the absolute robust z exceeds a threshold, typically 3 to 4.

Reason: the mean and standard deviation are distorted by the very outliers being detected. The median and
MAD are resistant, and the trailing window adapts to gradual change.

### Step 16 — Calibrate thresholds from data

Input: historical metric values and the target false-positive rate.
Output: thresholds with a measured false-positive rate.

Run the detector on history, count how many days would have alerted, and adjust the threshold until the
rate matches the target. Do not choose a threshold because it looks round.

Reason: a threshold is a statistical decision. Without calibration, the team discovers the false-positive
rate in production, at 03:00, one alert at a time.

### Step 17 — Define SLOs and error budgets

Input: the consumer deadline and the historical reliability.
Output: an SLO per data product and an error budget.

Example: 99.5% of daily partitions are complete and fresh by 06:00, measured over 30 days. The error budget
is 0.5% of 30 days, or 3.6 hours of allowed breach.

Reason: an SLO turns reliability into a measurable commitment. The error budget gives the team a basis for
prioritizing fixes over features.

### Step 18 — Route alerts to owners with runbooks

Input: the alerts and the team structure.
Output: an alert catalog with owner, severity, channel, runbook, and escalation.

Each alert names the data product, the check, the threshold, the first diagnostic query, the containment
action, and the escalation path. A page goes to a person, not a distribution list.

Reason: an alert without an owner is a notification; an alert without a runbook is an investigation that
starts from zero every time.

### Step 19 — Test the tests

Input: the test suite.
Output: mutation tests or injected faults that prove the tests fail when they should.

Inject a duplicate key, a null in a critical column, a unit change, and a missing partition. Confirm that
the corresponding test fails. A test that never fails is not a test.

Reason: test suites rot. A check that was disabled, misconfigured, or scoped to an empty table passes
forever. Fault injection is the evidence that the suite still works.

### Step 20 — Review alerts and retire noise

Input: the alert history.
Output: a monthly review with false positives, missed incidents, and retired alerts.

Track precision (share of alerts that led to an action) and time to acknowledge. Retire or retune alerts
below a precision threshold. Add alerts for incidents that were missed.

Reason: alert quality decays. Without review, the channel fills with noise and the team stops responding.

### Step 21 — Record lineage and link alerts to impact

Input: the lineage graph and the alert.
Output: an alert that names the downstream outputs in scope.

When a check fails, the alert includes the affected dashboards, models, and reports from the lineage
graph. The responder knows the blast radius before diagnosing.

Reason: an alert that says "row count anomaly" without impact cannot be prioritized against another
alert. Lineage supplies the impact.

### Step 22 — Run an incident review

Input: a confirmed data incident.
Output: a review with timeline, root cause, blast radius, and preventive actions.

The review separates the trigger (what alerted) from the cause (why the data were wrong) and from the
escape (why the test did not catch it). Each preventive action has an owner and a date.

Reason: the review is where the observability system improves. Without it, the same class of failure
recurs with a different trigger.

### Step 23 — Handle nulls and sentinels in checks

Input: the check definitions.
Output: an explicit rule for each non-standard value.

A null should fail a not-null check, not pass a range check. A sentinel such as `-1` or `9999` should be
mapped to null before the range check, or the check will evaluate a fake value. A check that treats null
as zero produces a false pass on missing data.

| Value | Not-null check | Range check | Rule |
|---|---|---|---|
| Null | Fail | Skip, do not pass | Null is missing, not zero |
| Empty string | Fail if required | Skip | Normalize to null |
| `-1` | Pass | Map to null first | Sentinel |
| `9999` | Pass | Map to null first | Sentinel |
| `1900-01-01` | Pass | Map to null first | Sentinel date |
| `0` | Pass | Evaluate against range | Zero can be real |

Reason: range checks are the most common source of false passes because they silently skip nulls or
accept sentinels as values.

### Step 24 — Handle clock and timezone boundaries in freshness

Input: source timestamps and check time.
Output: a freshness computation that is timezone-correct.

Compute the age from two timezone-aware instants. If the source timestamp has no offset, apply the
documented site offset or exclude it from freshness. A check that subtracts a naive local time from a UTC
time is wrong by the offset, which can be hours.

| Boundary | Risk | Rule |
|---|---|---|
| DST transition | One hour of apparent staleness | Use UTC |
| Source in another zone | Wrong age | Apply offset explicitly |
| Future timestamp | Negative age | Reject as invalid |
| Missing timestamp | No age | Fail the check, do not skip |
| Clock skew | False staleness | NTP; tolerance |

Reason: freshness alerts at 03:00 are often timezone bugs, not source failures. The rule prevents the
false page.

### Step 25 — Handle partial failure and alert suppression

Input: a pipeline failure or a run with incomplete data.
Output: a rule for which alerts fire and which are suppressed.

If the pipeline fails, the downstream quality checks should not fire independently; one root alert is
better than twelve downstream ones. Use a dependency-aware alert rule that suppresses dependent checks
for the duration of the failure, and re-enable them after recovery.

| Event | Alerts | Suppression |
|---|---|---|
| Extract fails | One pipeline alert | Suppress dependent checks |
| Load fails | One pipeline alert | Suppress dependent checks |
| Test fails after load | One data alert plus lineage impact | Do not suppress |
| Partial partition | One completeness alert | Suppress distribution checks |
| Source late | One freshness alert | Suppress volume if within tolerance |

Reason: alert storms obscure the root cause and train the team to mute the channel. The dependency rule
keeps one signal per failure.

The suppression rule must be time-bounded. A dependent check that stays suppressed for a week is not
suppressed; it is disabled, and the disablement will outlast the incident that justified it. Every
suppression carries an expiry, and the expiry is checked by the job that re-enables the check.

### Branch point A — The pipeline is batch and daily

Daily checks and a morning SLA are appropriate. The freshness threshold can be hours. The volume baseline
should account for day-of-week effects.

### Branch point B — The pipeline is streaming

Freshness becomes lag, volume becomes throughput, and distribution checks run on windows. Alerts must be
rate-limited, and the runbook must distinguish a processing delay from a data error.

### Branch point C — The data product is a model input

Add training-serving consistency checks: compare the feature distribution at serving time to the training
distribution, and monitor the model's calibration. A data test that passes can still produce a model
input that is out of distribution.

## 6. Worked example

A heart-failure cohort table feeds a quality dashboard at 06:00 and a model training job weekly. The team
has 12 checks and a set of daily metrics.

### 6.1 The checks

| # | Check | Level | Threshold | Severity |
|---:|---|---|---|---|
| 1 | `person_id` not null | Column | 0 nulls | Error |
| 2 | `(person_id, index_encounter_id)` unique | Table | 0 duplicates | Error |
| 3 | `age_at_index` between 18 and 120 | Column | 0 violations | Error |
| 4 | `index_date` within 2024 | Column | 0 violations | Error |
| 5 | `discharge_at` not null | Column | ≤0.1% null | Error |
| 6 | `discharge_at >= admit_at` | Row | 0 violations | Error |
| 7 | `index_encounter_id` exists in encounter | Cross-table | 0 orphans | Error |
| 8 | Row count vs trailing median | Table | ±5% | Warn |
| 9 | Freshness: max `loaded_at` age | Signal | ≤6 h | Error |
| 10 | Creatinine p99 vs baseline | Distribution | ≤1.5× | Warn |
| 11 | Admission source share vs baseline | Distribution | ±10 pp | Warn |
| 12 | Duplicate `(person_id, code, effective_time)` in labs | Table | ≤0.5% | Warn |

### 6.2 The run

The 06:00 run produces:

| # | Observed | Result |
|---:|---|---|
| 1 | 0 nulls | pass |
| 2 | 3 duplicate keys | fail |
| 3 | 2 rows with age 121 | fail |
| 4 | 0 outside 2024 | pass |
| 5 | 41 nulls of 7,468 = 0.55% | fail |
| 6 | 5 rows with discharge before admit | fail |
| 7 | 12 orphans | fail |
| 8 | 7,468 vs median 7,405 = +0.85% | pass |
| 9 | age 226 minutes | pass |
| 10 | p99 1,240 vs baseline 420 = 2.95× | fail |
| 11 | admission source share 38% vs 31% = +7 pp | pass |
| 12 | 0.9% duplicates | fail |

Six errors and one warning. The pipeline status is "success" because the checks run after the load; the
data product status is "failed." That distinction is the point: the orchestrator's green status is not the
data product's status.

### 6.3 The diagnosis

The failures cluster into two causes.

**Cause one: a duplicate delivery.** Checks 2, 7, and 12 all involve duplicated or orphaned records. The
source sent the 04:00 batch twice, and the loader appended instead of replacing. The three duplicate
cohort keys, 12 orphan encounters, and 0.9% duplicate labs are the same event.

**Cause two: a unit change at one sender.** Check 10 shows the creatinine 99th percentile at 1,240 µmol/L
against a baseline of 420, a ratio of 2.95. The median is 89 vs a baseline of 88, a change of 1.1%, which
does not trigger a median-based check.

The direction matters. A sender reporting 1.2 mg/dL into a field expected in µmol/L would produce values
far below the range, and the minimum check would fire. A 99th percentile above 420 µmol/L indicates the
opposite: values larger than expected, consistent with a sender reporting in µmol/L into a field that
another process later multiplies by 88.4, or with a different assay. The p99 identifies the tail; the
median does not.

The arithmetic: 420 × 1.5 = 630. The observed 1,240 exceeds 630, so the check fires. The ratio 1,240 / 420
= 2.952. If the contamination is 4% of values at 20,000 µmol/L, the mean rises by 4% × 20,000 = 800, while
the median is unchanged. The mean is not a robust statistic here; the p99 is the right monitor.

### 6.4 Alert calibration

Twelve checks run daily: 12 × 365 = 4,380 check-runs per year.

If each check has a 1% false-positive rate, the expected false alerts are 4,380 × 0.01 = 43.8 per year, or
3.65 per month. If the team wants at most one false alert per month across all checks, the per-check rate
must be at most 1 / (12 × 30) = 0.00278, or 0.28%.

For a normal distribution, a two-sided 3-sigma threshold has a false-positive rate of about 0.27%, which
is close to the target. Clinical metrics are not normal, so the threshold must be calibrated empirically:
run the detector on 365 days of history, count the alerts, and adjust until the rate matches 0.28%.

The arithmetic also shows why 12 daily checks at 1% each produce more noise than a team can absorb. The
number of checks is part of the alert budget.

### 6.5 SLO and error budget

The SLO is 99.5% of daily partitions complete and fresh by 06:00 over 30 days.

- Allowed breach: 0.5% × 30 days = 0.15 days = 3.6 hours.
- One missed deadline by 4 hours exhausts the budget for the month.
- Two missed deadlines by 2 hours each also exhaust it (4 hours > 3.6 hours).

The error budget is small because the SLO is high. A team that wants more change capacity should lower the
SLO to 99.0%, which allows 7.2 hours per 30 days, and state the trade-off to the consumer.

### 6.6 The incident review

The duplicate delivery was not caught by check 8 (row count +0.85%, within ±5%). It was caught by checks 2,
7, and 12. The review records:

- Trigger: uniqueness check failure at 06:04.
- Cause: source retransmission, loader append.
- Escape: the volume check tolerance was too wide to catch a 0.85% duplication.
- Blast radius: the cohort and every downstream output, 5 tables.
- Preventive action: make the load idempotent on the partition key; narrow the volume check to ±1.5% and
  add a duplicate-rate check at the staging layer.

The review also notes that the unit issue was caught by the distribution check, not by the range check,
because the values were within the broad range. The range check is necessary but not sufficient.

### 6.7 Check-run ledger

The 06:00 run records each check with its observed value and status.

| # | Check | Observed | Threshold | Status |
|---:|---|---:|---|---|
| 1 | `person_id` not null | 0 | 0 | pass |
| 2 | Key uniqueness | 3 | 0 | fail |
| 3 | Age range | 2 | 0 | fail |
| 4 | Index date in 2024 | 0 | 0 | pass |
| 5 | Discharge not null | 0.55% | ≤0.1% | fail |
| 6 | Temporal order | 5 | 0 | fail |
| 7 | Encounter reference | 12 | 0 | fail |
| 8 | Row count vs median | +0.85% | ±5% | pass |
| 9 | Freshness | 226 min | ≤360 | pass |
| 10 | Creatinine p99 | 2.95× | ≤1.5× | fail |
| 11 | Admission source share | +7 pp | ±10 pp | pass |
| 12 | Lab duplicate rate | 0.9% | ≤0.5% | fail |

Six failures and one warning. The ledger is machine-readable and is written to the quality schema with
the run ID, so a consumer can query the status of the partition they are reading.

### 6.8 Alert reconciliation

Twelve checks per day produce 4,380 check-runs per year.

| Metric | Value | Calculation |
|---|---:|---|
| Check-runs per year | 4,380 | 12 × 365 |
| False positives at 1% per check | 43.8 | 4,380 × 0.01 |
| False positives at 0.28% per check | 12.3 | 4,380 × 0.0028 |
| Alerts per year (all causes) | 120 | Observed |
| Alerts that led to an action | 44 | Review |
| Alert precision | 36.7% | 44 / 120 |
| Median time to acknowledge | 18 min | Observed |

A precision of 36.7% means roughly two of every three alerts are noise. The target is above 70%, which
requires either fewer checks or better thresholds. The arithmetic shows the trade-off: reducing the
per-check false-positive rate to 0.28% brings the expected false alerts to about one per month.

### 6.9 Second run and stability

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Checks executed | 12 | 12 | Equal |
| Failures | 6 | 6 | Equal |
| Freshness | 226 min | 226 min | Equal |
| Robust z | 6.07 | 6.07 | Equal |
| Quality digest | `e4a1...0c7b` | `e4a1...0c7b` | Equal |

A different result on the second run means a check depends on wall-clock time, iteration order, or a
random sample. Checks must be deterministic on a frozen partition, or the status cannot be compared
across runs.

### 6.10 Diagnostic checks

**Check A: partition completeness.** Loaded rows / source rows = 7,468 / 7,468 = 100%. Pass. A value
below 100% means the load dropped rows, and the next check should be the referential one.

**Check B: duplicate rate.** Duplicate keys / rows = 3 / 7,468 = 0.040%. Threshold 0%. Fail. The rate is
small in percentage terms and still corrupts every rate that uses the row as the denominator.

**Check C: freshness.** Age = 226 minutes. Threshold 360. Pass, but above the 180-minute warning, so the
dashboard shows a "delayed" badge. The badge is part of the product, not a console message.

**Check D: distribution tail.** p99 ratio = 1,240 / 420 = 2.95. Threshold 1.5. Fail. The median moved by
1.1%, which is why a median-only monitor would miss the tail contamination.

### 6.11 Decision table for a failed check

| Symptom | First diagnostic | Likely cause | Action |
|---|---|---|---|
| Duplicate key | Duplicate rate by source | Retransmission | Idempotent reload |
| Null in critical column | Null rate by sender | Interface change | Quarantine; contact sender |
| Range violation | Value distribution | Unit or sentinel | Map unit; fix source |
| Referential orphan | Missing parent count | Extract time mismatch | Re-extract; quarantine |
| Temporal violation | Count by facility | Clock or interface | Correct source; flag |
| Distribution shift | p50, p99, share by site | Unit change or new site | Segment; investigate |
| Freshness warn | Source `lastUpdated` | Source lag | Notify owner |
| Row count jump | Partition manifest | Duplicate or backfill | Compare partition hashes |

### 6.12 What the check suite does not cover

| Gap | Why it matters | Compensating control |
|---|---|---|
| Clinical plausibility beyond ranges | A value in range can be wrong | Clinician review of a sample |
| Cross-field consistency | Systolic below diastolic passes both ranges | Paired-field checks |
| Temporal patterns | A plausible date can be in the wrong sequence | Sequence checks per patient |
| Free-text content | Notes are not checked | NLP quality monitor |
| Source completeness | Loaded rows can be complete and the source partial | Source reconciliation |
| Model input drift | Features can pass and still be out of distribution | Training-serving monitor |
| Semantics of a code change | A new mapping can be valid and different | Cohort count diff by code set |
| Late-arriving correction | A complete partition can later change | Replay and `data_through` marker |
| Consumer-side filters | A correct table can be misread | Publish a data dictionary and examples |
| Access-path errors | A correct table can be joined incorrectly | Reconciliation to the source count |
| Denominator drift | A rate can change without the numerator moving | Track both numerator and denominator |
| Unit assumptions in the consumer | A correct SI value can be plotted as conventional | Publish units in the column name |

The suite covers structure, simple constraints, referential integrity, and a small number of
distributions. It does not cover clinical meaning. A passing suite is a floor, and the floor is useful
precisely because it catches the cheap failures before they consume clinical review time.

A second limit is that the checks run on the table, not on the consumer's query. A correct cohort table
joined to the wrong encounter table produces a wrong dashboard with no check failing. Consumer-side
validation, such as a dashboard reconciliation to the source cohort count, is a separate control and
should be part of the release.

A third limit is that a check can be right and the threshold wrong. The p99 check fired at 2.95× for a
newly onboarded scanner protocol; the values were correct and the threshold was not. Threshold changes
are versioned and reviewed like code changes, including their false-positive cost.

## 7. Code, query, or calculation

### 7.1 Data quality checks and robust anomaly detection (standard library, executable)

The script creates a small cohort table, runs six checks, computes a robust z-score for the daily row
count, and calculates freshness. It was executed on CPython 3.11.15 and SQLite 3.50.4.

```python
"""Data quality checks and robust anomaly detection. Standard library only."""
from __future__ import annotations

import datetime as dt
import sqlite3
from statistics import median

con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE encounter (
    encounter_id INTEGER PRIMARY KEY,
    admit_at TEXT NOT NULL,
    discharge_at TEXT
);
CREATE TABLE cohort (
    person_id INTEGER NOT NULL,
    index_encounter_id INTEGER NOT NULL,
    age_at_index INTEGER NOT NULL,
    index_date TEXT NOT NULL
);
INSERT INTO encounter VALUES
 (1, '2024-01-05T08:00:00Z', '2024-01-10T12:00:00Z'),
 (2, '2024-01-05T08:00:00Z', '2024-01-10T12:00:00Z'),
 (3, '2024-02-01T08:00:00Z', NULL),
 (4, '2024-03-01T08:00:00Z', '2024-02-28T08:00:00Z'),
 (5, '2024-04-01T08:00:00Z', '2024-04-05T08:00:00Z');
INSERT INTO cohort VALUES
 (1, 1, 72, '2024-01-05'),
 (1, 1, 72, '2024-01-05'),
 (2, 3, 45, '2024-02-01'),
 (3, 4, 121, '2024-03-01'),
 (4, 99, 60, '2024-04-01');
""")

checks = []


def check(name: str, observed, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "observed": observed, "passed": passed, "detail": detail})


dupes = con.execute("""
SELECT COUNT(*) FROM (
    SELECT person_id, index_encounter_id, COUNT(*) AS n
    FROM cohort GROUP BY person_id, index_encounter_id HAVING n > 1
)""").fetchone()[0]
check("unique_key", dupes, dupes == 0)

nulls = con.execute("SELECT COUNT(*) FROM cohort c JOIN encounter e "
                    "ON e.encounter_id = c.index_encounter_id "
                    "WHERE e.discharge_at IS NULL").fetchone()[0]
check("discharge_not_null", nulls, nulls == 0)

bad_age = con.execute("SELECT COUNT(*) FROM cohort WHERE age_at_index < 18 OR age_at_index > 120").fetchone()[0]
check("age_range", bad_age, bad_age == 0)

orphans = con.execute("""
SELECT COUNT(*) FROM cohort c
LEFT JOIN encounter e ON e.encounter_id = c.index_encounter_id
WHERE e.encounter_id IS NULL""").fetchone()[0]
check("referential", orphans, orphans == 0)

temporal = con.execute("""
SELECT COUNT(*) FROM encounter
WHERE discharge_at IS NOT NULL AND discharge_at < admit_at""").fetchone()[0]
check("temporal", temporal, temporal == 0)

row_count = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
check("row_count_vs_baseline", row_count, abs(row_count - 5) / 5 <= 0.05, "baseline 5")

for c in checks:
    print(c["name"], c["observed"], "PASS" if c["passed"] else "FAIL", c["detail"])

# robust z-score for a daily metric
history = [7400, 7412, 7398, 7420, 7405, 7390, 7418]
current = 7468
med = median(history)
mad = median([abs(x - med) for x in history])
sigma = 1.4826 * mad
z = (current - med) / sigma if sigma else float("inf")
print("median", med)
print("mad", mad)
print("robust_z", round(z, 2))
print("anomaly", abs(z) > 3.5)

# freshness
loaded = dt.datetime.fromisoformat("2026-10-24T02:14:00+00:00")
checked = dt.datetime.fromisoformat("2026-10-24T06:00:00+00:00")
age_minutes = int((checked - loaded).total_seconds() / 60)
print("freshness_minutes", age_minutes)
print("freshness_status", "error" if age_minutes > 360 else "warn" if age_minutes > 180 else "ok")
```

Expected output:

```text
unique_key 1 FAIL
discharge_not_null 1 FAIL
age_range 1 FAIL
referential 1 FAIL
temporal 1 FAIL
row_count_vs_baseline 5 PASS baseline 5
median 7405
mad 7
robust_z 6.07
anomaly True
freshness_minutes 226
freshness_status warn
```

Five checks fail and one passes. The row count check passes because 5 equals the baseline, which shows why
a volume check alone is insufficient: the same count can contain duplicates, orphans, and impossible
values.

### 7.2 dbt schema tests (configuration, not executable here)

The following is a dbt `schema.yml` for a dbt 1.12 project. It is configuration for dbt and requires a dbt
installation and a warehouse connection.

```yaml
version: 2

models:
  - name: hf_cohort
    description: "One row per person per first heart-failure admission in 2024."
    config:
      contract:
        enforced: true
    columns:
      - name: person_id
        data_type: bigint
        tests:
          - not_null
          - unique
      - name: index_encounter_id
        data_type: bigint
        tests:
          - not_null
          - relationships:
              to: ref('encounter')
              field: encounter_id
      - name: age_at_index
        data_type: integer
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 18
              max_value: 120
      - name: index_date
        data_type: date
        tests:
          - not_null
```

The `contract: enforced: true` block requires every column to declare a `data_type` and makes dbt fail the
build on a mismatch. The `relationships` test is the referential integrity check. The
`dbt_utils.accepted_range` test requires the `dbt-utils` package.

### 7.3 Source freshness configuration

```yaml
sources:
  - name: ehr
    config:
      freshness:
        warn_after: {count: 6, period: hour}
        error_after: {count: 12, period: hour}
      loaded_at_field: _loaded_at
    tables:
      - name: encounter
      - name: observation
```

`dbt source freshness` (or `dbt freshness --resource-type source` in newer versions) compares
`max(_loaded_at)` to the current time and returns pass, warn, or error. The thresholds are per source and
can be overridden per table.

## 8. Data, standards, and regulatory context

### 8.1 Testing tools and their scope

| Tool | Scope | Strength | Limitation |
|---|---|---|---|
| dbt tests | SQL models | Native to transformation; contracts; docs | SQL-centric; generic tests are structural |
| dbt source freshness | Sources | Simple warn/error thresholds | Timestamp field must be reliable |
| Great Expectations | Data frames and databases | Rich expectation library; data docs | Operational overhead; version API changes |
| Soda | SQL and data frames | YAML checks; alerting | Commercial features |
| Pandera | Data frames | Schema and statistical checks in Python | Python-only |
| Custom SQL | Any warehouse | Exact business rules | Maintenance and ownership |

The choice is less important than the placement. Tests should run where the data are transformed, and the
results should be stored where the team can see them.

### 8.2 Lineage and quality metadata

OpenLineage includes dataset facets for `dataQualityMetrics` and `dataQualityAssertions`, and a
`TestRunFacet` for test outcomes. Emitting quality results as lineage facets connects a failed check to the
datasets and jobs it affects. The current release line in 2026 is 1.x (1.54.0 in October 2025).

A quality result without lineage is a row in a table. A quality result with lineage is an impact
assessment.

### 8.3 SLOs for data

A data SLO is a measurable commitment about a data product: freshness, completeness, or accuracy. It
should be defined with the consumer, measured automatically, and reviewed on a cadence. The error budget
is the allowed breach.

An SLO is not an SLA. An SLA is a contract with consequences; an SLO is an internal target. A team can
have an SLO without an SLA, but an SLA without an SLO is a promise without a measurement.

### 8.4 Regulatory context

- **HIPAA Security Rule.** Audit controls require recording and examining activity. Quality monitors are
  not audit controls, but the logs of who ran a check and what it found can support an investigation.
- **21 CFR Part 11.** If a quality result supports a regulated submission, the test definition, the
  execution record, and the result must be attributable and retained.
- **Model risk management.** A model's input data quality is part of its validation. Distribution
  monitoring and training-serving consistency checks are evidence of ongoing monitoring.
- **Data contracts.** A contract is not a regulation, but it is the mechanism by which a producer and a
  consumer agree on quality. In a hospital, that agreement may be reviewed by the data governance
  committee.

### 8.5 What the tools do not provide

- The correct thresholds.
- The clinical plausibility rules.
- The owner of an alert.
- The decision about what to do when a check fails.
- The definition of the data product.

Those are design decisions. A team can install every tool in the table and still have no observability if
no one owns the alerts.

## 9. Industry practice and case evidence

### 9.1 dbt tests and contracts

dbt documents four generic tests: `not_null`, `unique`, `accepted_values`, and `relationships`. Model
contracts enforce column names and data types at build time, with platform-specific constraint enforcement.
Source freshness uses `warn_after` and `error_after` with a `loaded_at_field`. The combination covers
structure, simple constraints, referential integrity, and freshness, which is a large share of common
defects. It does not cover clinical plausibility or distribution shift.

### 9.2 Great Expectations

Great Expectations provides an expectation library and validation checkpoints with data docs. The API
changed between the 0.18.x line and the 1.x core, which is a reminder that test code is code and must be
versioned and maintained. The value is the breadth of expectations; the cost is the operational surface.

### 9.3 OpenLineage quality facets

OpenLineage's `DataQualityAssertionsDatasetFacet` and `TestRunFacet` record assertion results alongside
lineage. This is the mechanism that connects a failed check to the datasets and jobs downstream. Adoption
requires instrumentation in the test runner, not only in the orchestrator.

### 9.4 What is inference

- That a passing dbt suite means the data are correct. It means the declared structural and referential
  checks passed.
- That a robust z-score of 3.5 is the right threshold. It is a starting point; the correct threshold comes
  from calibration on the metric's history.
- That more checks improve quality. Beyond a point, more checks produce more false positives and less
  attention.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Green pipeline, bad data | Checks run after publish; no data status | Success means the job ran | Silent wrong numbers | Separate data product status |
| Test with no threshold | Check always passes | Placeholder or disabled | False assurance | Review and fault injection |
| Volume-only monitoring | Duplicate passes ±5% | Count ignores content | Inflated rates | Uniqueness and duplicate-rate checks |
| Mean-only drift | Tail contamination invisible | Mean sensitive to outliers | Missed unit error | Median, p99, robust z |
| Alert storm | Channel muted | Too many low-value alerts | Missed real incident | Calibrate and retire |
| No owner | Alert to a group alias | Organizational gap | No response | Named owner and escalation |
| Stale threshold | Alert fires daily | Baseline not updated | Desensitization | Recalibrate quarterly |
| Freshness without completeness | Data fresh but partial | Feed delivers in waves | Understated counts | Volume and reconciliation |
| Schema drift undetected | New column appears | No contract | Downstream break | Schema diff and contract |
| Distribution check on raw values | Unit change passes | Range too wide | Wrong feature | Per-unit and per-site checks |
| No test of the tests | Suite never fails | Disabled or empty scope | False confidence | Fault injection |
| Alert without lineage | Cannot scope impact | No downstream map | Slow triage | Lineage-linked alerts |

Two red flags deserve emphasis.

**A dashboard with no data status.** If the consumer cannot see whether the underlying partition is
complete and fresh, they will treat the number as current. The status must be visible where the number is
consumed, not only in the orchestrator.

**A test suite with no failing test in the last quarter.** Either the data are perfect or the tests are
not running. The second is more likely. Fault injection resolves the ambiguity.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The baseline represents normal variation | Anomaly detection | Seasonality, site onboarding | Compare across seasons and sites |
| The metric is stationary over the window | Robust z-score | Trend or level shift | Detrend or use a shorter window |
| Thresholds are calibrated | Alert quality | Chosen by intuition | Backtest on history |
| The source timestamp is reliable | Freshness | Clock skew or null | Compare to ingestion time |
| The contract matches the consumer's need | Test relevance | Producer-only contract | Consumer review |
| Alerts reach an owner | Response | Stale rotation | Quarterly alert test |
| Tests run on every build | Coverage | CI skip or scope | Check test count and failures |
| Lineage is complete | Impact analysis | Uninstrumented jobs | Coverage report |
| The team has capacity to respond | Reliability | Understaffing | Track time to acknowledge |

The largest limit is attention. A team can maintain a small number of high-signal checks and respond to
them, or a large number of low-signal checks and ignore them. The design should optimize for the first.

A second limit is that observability detects anomalies, not causes. A robust z-score tells you that
something changed; it does not tell you whether the change is a data error, a population change, or a
legitimate new site. The diagnosis requires lineage, domain knowledge, and the source.

## 12. Exercises

### Exercise 1 — Test inventory (recall, 25 minutes)

Given a cohort table, deliverable: a test inventory with at least 15 checks classified by level and
severity.

Acceptance criteria: all five levels represented; at least three cross-table checks; each check has a
threshold and a severity.

### Exercise 2 — Write the tests (application, 50 minutes)

Deliverable: SQL or dbt tests for uniqueness, not-null, range, referential integrity, and temporal order,
with a fixture that fails each one.

Acceptance criteria: each test fails on the fixture and passes on valid data; the fixture is synthetic; the
tests state their engine.

### Exercise 3 — Freshness and volume (application, 40 minutes)

Deliverable: freshness and volume monitors with warn and error thresholds, and the arithmetic for the
expected alert volume.

Acceptance criteria: thresholds derive from the consumer deadline; day-of-week effects are addressed; the
false-positive estimate is shown.

### Exercise 4 — Distribution shift (calculation, 45 minutes)

Given a baseline median of 88 and MAD of 18, and a new median of 92, deliverable: the robust z-score and a
decision at a threshold of 3.5.

Acceptance criteria: sigma = 1.4826 × MAD; the arithmetic is shown; the decision is stated; the limitation
of the median is discussed.

### Exercise 5 — Alert calibration (calculation, 40 minutes)

Deliverable: the per-check false-positive rate needed to keep 12 daily checks under one false alert per
month, and the annual alert volume at 1% per check.

Acceptance criteria: the arithmetic is correct; the difference between the two rates is explained; a
recommendation is made.

### Exercise 6 — SLO (design, 35 minutes)

Deliverable: an SLO and error budget for a daily cohort, with the measurement method and the consequences
of exhausting the budget.

Acceptance criteria: the SLO is measurable; the budget is in hours per 30 days; the consequence is stated;
the consumer agrees.

### Exercise 7 — Runbook (design, 45 minutes)

Deliverable: a runbook for the top three alerts, each with owner, first query, containment, escalation, and
resolution criteria.

Acceptance criteria: each alert has a named owner; the first query is copy-ready; containment is specific;
resolution is defined.

### Exercise 8 — Silent failure (critique, 60 minutes)

A dashboard shows a 6% drop in the baseline lab rate; the pipeline is green. Deliverable: a diagnostic
plan using the five signals and the lineage graph, with the first three queries.

Acceptance criteria: considers freshness, volume, schema, distribution, and lineage; the queries are
specific; the plan distinguishes data error from population change.

## 13. Self-check answers

**1. What is the difference between a test and a monitor?**

A test runs at build time against a known input and blocks a release on failure. A monitor runs against
production data and produces a metric over time, triggering an investigation. Both are needed because
failures can be introduced by code or by data.

**2. Why is a green pipeline not a quality signal?**

Because success means the job completed, not that the data are correct. The load can append duplicates,
map a missing column to null, or accept a unit error and still return success.

**3. What are the five levels of data tests?**

Column and row (null, range, accepted values), table (uniqueness, row count), cross-table (referential,
reconciliation, temporal), distribution (drift, anomaly), and end-to-end (full pipeline on a fixture).

**4. Why is a volume check alone insufficient?**

Because the count can be unchanged while the content changes. Duplicates, orphans, and impossible values
can all preserve the row count. Uniqueness, referential, and temporal checks are needed.

**5. Compute the robust z-score.**

Median 88, MAD 18, new value 92. Sigma = 1.4826 × 18 = 26.6868. z = (92 − 88) / 26.6868 = 0.1499, or
0.15. At a threshold of 3.5, the change is not flagged. A median-based check is robust to outliers but
insensitive to small shifts; pair it with a mean or a quantile check.

**6. Compute the per-check false-positive rate.**

Target: one false alert per month across 12 daily checks. Total check-runs per month = 12 × 30 = 360. Rate
= 1 / 360 = 0.00278, or 0.28%. At 1% per check, the expected false alerts are 360 × 0.01 = 3.6 per month,
or 43.2 per year.

**7. What is an error budget?**

The allowed amount of unreliability implied by an SLO. For 99.5% over 30 days, the budget is 0.5% × 30 days
= 0.15 days = 3.6 hours. It converts reliability into a quantity that can be spent on change.

**8. Why calibrate a threshold?**

Because a threshold is a statistical decision with a false-positive cost. Without calibration, the team
discovers the cost in production. Backtesting on history gives the rate before the alert is enabled.

**9. What is the first query for a freshness alert?**

Check the source's maximum load timestamp and the ingestion job's last successful run. If the source is
stale, the problem is upstream; if the source is current and the table is not, the problem is the
extraction or the load.

**10. Why link alerts to lineage?**

Because impact determines priority. An alert on a dataset with no consumers is lower priority than one on
a dataset feeding a clinical dashboard. Lineage supplies the downstream scope.

## 14. Interview questions

**1. "How do you know your data pipeline is working?"**

Strong: separates job success from data quality, names tests and the five signals, and describes the
status the consumer sees. Weak: "the job is green."

**2. "What would you test in a cohort table?"**

Strong: grain and uniqueness, critical nulls, ranges, referential integrity, temporal order,
reconciliation, and distribution. Weak: "row count."

**3. "How do you detect a broken feed?"**

Strong: freshness, volume versus baseline, schema diff, duplicate rate, and source reconciliation. Weak:
"check if the table has rows."

**4. "How do you set an alert threshold?"**

Strong: calibrate on history to a target false-positive rate, account for seasonality, and review
quarterly. Weak: "three sigma" without checking the distribution.

**5. "What is alert fatigue, and how do you prevent it?"**

Strong: measure precision, retire noisy alerts, name owners, and keep the channel high-signal. Weak: "send
fewer alerts."

**6. "How do you monitor a model's input data?"**

Strong: compare feature distributions to training, check schema and freshness, monitor calibration, and
alert on training-serving skew. Weak: "monitor accuracy," which may be unavailable.

**7. "What is an SLO for data?"**

Strong: a measurable commitment about freshness, completeness, or accuracy, with an error budget and a
consumer. Weak: "99.9% uptime."

**8. "How do you test the tests?"**

Strong: inject faults and confirm the expected test fails; review disabled or empty-scope tests. Weak: "we
have coverage."

**9. "A dashboard number dropped 6%. What do you check first?"**

Strong: freshness and completeness, then lineage for the affected partitions, then distribution and
source reconciliation. Weak: "check the SQL."

**10. "When would you not add a check?"**

Strong: when the signal is low-value, the false-positive cost is high, and no owner will act. Weak: "add
everything."

## 15. Key terms

- **Alert fatigue** — desensitization from too many alerts.
- **Anomaly** — value outside the expected pattern.
- **Canary** — small-scale run before full release.
- **Completeness** — share of expected records present.
- **Data contract** — executable schema and quality agreement.
- **Data test** — assertion with pass/fail.
- **dbt test** — SQL assertion in a dbt project.
- **Distribution shift** — change in value distribution.
- **Error budget** — allowed unreliability.
- **Expectation** — Great Expectations assertion.
- **False negative** — missed real issue.
- **False positive** — alert with no real issue.
- **Freshness** — age of the most recent data.
- **Golden dataset** — known-good fixture.
- **Lineage** — dependency graph.
- **MAD** — median absolute deviation.
- **Monitor** — continuous metric with thresholds.
- **Observability** — inferring state from outputs.
- **Owner** — named person or team responsible.
- **Reconciliation** — comparison of two counts.
- **Robust z-score** — deviation scaled by MAD.
- **Root cause** — underlying cause.
- **Runbook** — procedure for an alert.
- **Schema drift** — unplanned structural change.
- **SLA** — contractual service level.
- **SLI** — service-level indicator.
- **SLO** — service-level objective.
- **Source freshness** — dbt check on source timestamp.
- **Threshold** — pass, warn, or error boundary.
- **Volume** — row or byte count.

## 16. Sources

All URLs accessed 25 October 2026.

1. dbt Labs. "Add tests to your DAG." https://docs.getdbt.com/docs/build/data-tests
2. dbt Labs. "Model contracts." https://docs.getdbt.com/docs/mesh/govern/model-contracts
3. dbt Labs. "Source freshness." https://docs.getdbt.com/docs/build/sources
4. dbt Labs. "freshness configuration." https://docs.getdbt.com/reference/resource-configs/freshness
5. dbt Labs. "About dbt build." https://docs.getdbt.com/reference/commands/build
6. Great Expectations. "Define Expectations." https://docs.greatexpectations.io/docs/core/define_expectations/
7. Great Expectations. "Validate Data." https://docs.greatexpectations.io/docs/oss/guides/validation/validate_data_lp/
8. OpenLineage. "Specification." https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md
9. OpenLineage. "Releases." https://github.com/OpenLineage/OpenLineage/releases
10. Apache Airflow. "Best Practices." https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
11. eCFR. "45 CFR 164.312." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
12. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
13. NIST. "SP 800-66 Rev. 2." https://csrc.nist.gov/pubs/sp/800/66/r2/final
14. Pandera. https://pandera.readthedocs.io/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_security-engineering.md`](09_security-engineering.md)
- Related sibling: [`04_orchestration.md`](04_orchestration.md)
- Related sibling: [`07_data-versioning-and-lineage.md`](07_data-versioning-and-lineage.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.4 implements six quality checks and an alert; Lab 3.6 diagnoses a broken pipeline.
