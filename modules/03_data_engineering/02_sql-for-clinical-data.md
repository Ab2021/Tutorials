# SQL for Clinical Data

> Module 03 — Data Engineering and Interoperability
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: 01; basic SQL
> Version 2026.3

## 1. Why this topic exists

Every cohort is a query. A clinician says "adults with a first heart-failure admission in 2024, excluding
anyone already known to have heart failure." An analyst writes SQL. The query runs, the number appears,
and a steering committee treats the number as a fact. If the query's grain, time window, or deduplication
is wrong, the number is still precise, still reproducible, and still wrong. Nothing in the result set
flags it.

The failure pattern is consistent across organizations.

- Two analysts define "first admission" differently. One takes the first row returned by the database;
  the other takes the earliest admission date. The cohort sizes differ by 11%, and both queries are
  defensible readings of the request.
- A diagnosis appears twice because the interface sent it twice. A `COUNT(*)` doubles those patients. A
  `COUNT(DISTINCT person_id)` fixes the count but hides the duplication, which still affects every
  per-encounter rate.
- A laboratory result is joined to an encounter by `person_id` alone. The join matches every lab the
  patient ever had, and the "admission lactate" becomes an average of values spanning three years.
- A date range uses `BETWEEN '2024-01-01' AND '2024-12-31'`. Timestamps on 31 December after midnight
  fall outside the range, and the last day of the year disappears.
- A window function partitions by patient but orders by a timestamp with ties. The row chosen is
  arbitrary, and the cohort changes between runs on the same data.
- A left join is used to find patients "without a follow-up visit," then a predicate on the right table
  turns it into an inner join. The exclusion silently includes patients who should have been excluded.

None of these requires a database expert to catch. They require knowing the grain of each table, the
meaning of each timestamp, and the semantics of the SQL construct being used. That is the content of this
topic.

SQL is also the interface between engineering and clinical reasoning. A pipeline is reproducible only if
its cohort logic is a versioned, testable artifact rather than a sequence of manual filters in a
notebook. A query that another analyst can read and check is a clinical definition. A query that only
its author understands is an opinion with a result set.

> **Note:** The examples use an OMOP-shaped schema because OMOP CDM is the most common common data model
> in observational health research. The SQL patterns transfer to any event table: encounters, diagnoses,
> labs, medications, claims, or device readings.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the grain of a clinical table before writing a query, and detect grain errors from duplicate
   counts. Assessed in Exercise 1 and self-check item 1.
2. **Deduplicate** event rows with `ROW_NUMBER()` using an explicit partition, ordering, and tie-breaker,
   and explain what the choice hides. Assessed in Exercise 2 and self-check item 3.
3. **Write** a temporal join with half-open intervals and a pre-admission window, and compute the match
   rate. Assessed in Exercise 3 and self-check item 4.
4. **Build** a cohort with an index date, lookback period, exclusions, and a documented definition
   version. Assessed in Exercise 4.
5. **Collapse** overlapping or adjacent intervals with a gaps-and-islands query. Assessed in Exercise 5.
6. **Diagnose** a query that returns a plausible but wrong count by tracing grain, join cardinality, and
   null semantics. Assessed in Exercise 6.
7. **Translate** a query between PostgreSQL and at least one other engine, naming the semantic
   differences that can change results. Assessed in Exercise 7.
8. **Write** tests for cohort uniqueness, referential integrity, and temporal ordering. Assessed in
   Exercise 8.

## 3. The core mental model

Clinical SQL is four operations applied to event tables, in a fixed order of reasoning.

```text
             +------------------------------------------------------+
             | 4. AGGREGATE / WINDOW                                |
             |    per patient, per encounter, per period            |
             +--------------------------^---------------------------+
                                        |
             +--------------------------+---------------------------+
             | 3. JOIN ON TIME AND IDENTITY                         |
             |    same patient, event inside interval, nearest prior|
             +--------------------------^---------------------------+
                                        |
             +--------------------------+---------------------------+
             | 2. FILTER TO THE POPULATION AND PERIOD               |
             |    age, enrollment, index window, code set           |
             +--------------------------^---------------------------+
                                        |
             +--------------------------+---------------------------+
             | 1. DECLARE THE GRAIN                                 |
             |    one row = one what?                               |
             +------------------------------------------------------+
```

Read it upward. Most wrong answers enter at layer 1 and become invisible at layer 4. A query that
aggregates a table with the wrong grain produces a number that looks like an answer.

### 3.1 Grain is the first question

For every table, ask: what does one row represent? In an OMOP-shaped model:

- `person`: one row per person.
- `visit_occurrence`: one row per visit, which may include transfers if the source merges them.
- `condition_occurrence`: one row per condition record, not one per patient and not one per admission.
- `measurement`: one row per measurement, including repeats at the same timestamp.
- `drug_exposure`: one row per exposure period, which may overlap another exposure of the same drug.

If you cannot state the grain in one sentence, you cannot write a correct `COUNT`, `JOIN`, or window
function. The grain determines the primary key, the deduplication rule, and the meaning of every
aggregate.

### 3.2 Time is an interval, not a point

Clinical events have at least three times:

- **Effective time**: when the event occurred or the measurement was taken.
- **Recorded time**: when it was entered into the source system.
- **Loaded time**: when the platform received it.

A query that uses the wrong one answers a different question. "Labs during admission" uses effective
time. "Labs available to the clinician at discharge" uses recorded time. "Records received this week"
uses loaded time.

Intervals should be half-open, `[start, end)`. A lab at exactly the discharge timestamp belongs to the
next interval, not to both. Closed intervals `[start, end]` double-count boundary events when two
intervals are adjacent.

### 3.3 Joins multiply rows

A join is a cartesian product followed by a filter. If the left side has 1,000 patients and the right
side has 20 labs per patient, the result has 20,000 rows. That is correct if the question is
lab-level. It is wrong if the question is patient-level, because the patient now appears 20 times.

Before every join, write the expected cardinality: one-to-one, one-to-many, many-to-one, or
many-to-many. If the answer is many-to-many, the query needs a deduplication or an aggregation before
the join. Most "the numbers are too high" bugs are an undeclared many-to-many join.

### 3.4 Null is not a value

SQL uses three-valued logic. `NULL = NULL` is unknown, not true. `NOT IN` with a null in the subquery
returns no rows for the whole predicate. `COUNT(column)` skips nulls; `COUNT(*)` does not. A left join
that finds no match produces nulls, and a `WHERE` predicate on those nulls converts the join back to an
inner join.

These are not trivia. A cohort exclusion written as `WHERE dx.person_id IS NULL` after a left join is
correct; the same intent written as `WHERE dx.condition_concept_id NOT IN (...)` is wrong when the
subquery contains a null.

### 3.5 Window functions compute without collapsing

`ROW_NUMBER()`, `RANK()`, `LAG()`, `LEAD()`, and aggregate windows add a value to each row while keeping
all rows. They are the correct tool for first-event, prior-event, and running-total questions. They are
also the most common source of nondeterminism because the `ORDER BY` inside the window may not be unique.
A window ordering by `visit_start_date` alone, when a patient has two visits on the same day, produces a
different winner on different runs or engines.

The rule: every window ordering needs a tie-breaker that makes the order total. `ORDER BY
visit_start_date, visit_occurrence_id` is deterministic. `ORDER BY visit_start_date` is not.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Grain | What one row represents | Confused with primary key | — |
| Cardinality | Number of rows a join produces per input row | Assumed to be one-to-one | — |
| Cohort | A defined set of persons meeting inclusion and exclusion criteria | Any query result | — |
| Index date | The anchor date from which follow-up and features are measured | Confused with the first row returned | Date |
| Lookback | The period before the index date used to establish prior history | Confused with the study period | Days |
| Washout | A period during which prior events are counted but not included as outcomes | Used interchangeably with lookback | Days |
| Observation period | The span during which a person's data are considered complete | Assumed to equal enrollment | Start and end date |
| Prevalent case | A case that existed before the index date | Confused with an incident case | — |
| Incident case | A new case after a clean lookback period | Assumed to be the first record | — |
| Half-open interval | An interval including its start and excluding its end, `[start, end)` | Written with `BETWEEN` | — |
| Temporal join | A join that matches on identity and on time | Any join with a date filter | — |
| As-of join | A join that takes the most recent prior row | Confused with an equi-join | — |
| Deduplication | Selecting one row from a set of duplicates by a stated rule | `SELECT DISTINCT` on all columns | — |
| Tie-breaker | A deterministic secondary sort key | Omitted from window ordering | — |
| Gaps and islands | Collapsing adjacent intervals into continuous episodes | Summing durations without merging | — |
| SCD type 2 | A dimension that preserves history with valid-from and valid-to dates | Overwriting the attribute | — |
| Fact table | A table of events at a declared grain | Any large table | — |
| Dimension | A table describing entities | A lookup with duplicates | — |
| Surrogate key | A generated integer key | Confused with a natural key | Integer |
| Natural key | A key meaningful in the source system | Assumed unique across sources | Source-specific |
| Three-valued logic | True, false, unknown | Treated as two-valued | — |
| NULL | Absence of a value, not zero and not empty string | Treated as zero | — |
| `COUNT(*)` | Counts rows, including rows with nulls | Used when distinct entities are meant | — |
| `COUNT(DISTINCT)` | Counts distinct non-null values | Assumed to fix duplication in other columns | — |
| Window frame | The subset of rows a window function sees | Left at the default unintentionally | Rows or range |
| `ROW_NUMBER` | Arbitrary but deterministic rank within a partition | Used without a tie-breaker | — |
| `RANK` / `DENSE_RANK` | Rank with gaps / without gaps | Used to deduplicate | — |
| `LAG` / `LEAD` | Previous / next row in a window | Assumed to respect a filter | — |
| CTE | Common table expression, a named subquery | Assumed to be materialized | — |
| Correlated subquery | A subquery referencing the outer row | Used at scale without checking the plan | — |
| Anti-join | Returning left rows with no right match | Written as `NOT IN` with nulls | — |
| Semi-join | Returning left rows with at least one right match | Written as `IN` with duplicates | — |
| Idempotent query | A query whose result does not change when re-run on the same snapshot | Confused with a fast query | — |
| Dialect | Engine-specific SQL syntax and semantics | Assumed portable | PostgreSQL, BigQuery, Snowflake, DuckDB |
| Predicate pushdown | Evaluating filters before the join | Assumed automatic | — |
| Data skew | Uneven distribution of join keys | Ignored until a job times out | — |

Two terms need emphasis because they drive most cohort disputes.

**Observation period.** A patient with an encounter in 2024 may have no data for 2022 because they moved
or because the health system did not capture those visits. Requiring a 365-day observation period before
the index date removes patients whose prior history is unknown, which changes both the cohort size and
the prevalence estimate. The requirement must be explicit, because it is a clinical assumption, not a
SQL detail.

**Index date.** "First heart-failure admission in 2024" can mean the first admission whose primary
diagnosis is heart failure, the first admission with any heart-failure diagnosis, the first heart-failure
diagnosis date, or the first treatment date. Each produces a different cohort and a different follow-up
window. The definition must state which.

## 5. The mechanics, step by step

This procedure builds a cohort query that another analyst can reproduce. Each step names its input,
output, and the failure it prevents.

### Step 1 — Write the clinical question in one sentence

Input: a stakeholder request.
Output: a sentence with population, exposure or condition, time window, and outcome.

Example: "Among adults with a first heart-failure admission between 1 January and 31 December 2024, what
proportion is readmitted within 30 days of discharge?"

Reason: the sentence exposes ambiguities before SQL hides them. "Adults" needs an age threshold and an
age date. "First" needs a lookback. "Admission" needs a visit type. "Readmitted" needs a definition of
the same hospital versus any hospital.

### Step 2 — Declare the unit of analysis

Input: the question.
Output: one row per person, per admission, or per admission-day.

The readmission question is one row per index admission, with a person identifier attached. A
patient-level cohort that collapses multiple admissions answers a different question and cannot compute
readmission correctly.

Reason: the unit of analysis determines the primary key of the result and the grain of every join. It is
the most common omitted statement in cohort documentation.

### Step 3 — Identify the tables and their grain

Input: the data model.
Output: a table inventory with grain, key, and time columns.

For each table, write one row: name, grain, primary key, effective time, recorded time. If a table has no
declared primary key, infer one from the source and test uniqueness before using it.

Reason: joins are only correct when the grain on both sides is known. An encounter table with duplicate
`encounter_id` values will multiply every lab joined to it.

### Step 4 — Establish the observation period

Input: enrollment or coverage data.
Output: a per-person `[observation_start, observation_end)` interval.

Use the intersection of enrollment and data availability. A patient enrolled for the full year but with
no claims for six months may still be observable; a patient who moved away is not. Document the rule:
continuous enrollment with a maximum gap, or first-to-last event, or a fixed window.

Reason: without an observation period, "no prior heart failure" may mean "no data," and the cohort
silently includes patients whose history is unknown.

### Step 5 — Define the code set and version it

Input: clinical concept definitions.
Output: a versioned code list with source vocabulary, standard concept, and inclusion rationale.

Example: heart failure as SNOMED CT concepts mapped to OMOP standard concepts, excluding codes that
represent pulmonary edema secondary to renal failure if the clinical definition excludes them. Record
the vocabulary release date.

Reason: code sets change. A query that reads "heart failure" from a table populated by a different
vocabulary version is not reproducible. The code list is part of the cohort definition, not a
configuration detail.

### Step 6 — Filter to the eligible population

Input: person and enrollment tables.
Output: persons meeting age, sex, and observation criteria at the index date.

Age must be computed at a stated date. `year_of_birth` gives an integer year; age at index is
`index_year - year_of_birth`, which is off by one for birthdays later in the year. If month and day are
available, compute exact age; if not, state the approximation.

Reason: age is a common source of off-by-one cohort changes. A threshold of 18 that uses year only
includes some 17-year-olds and excludes some 18-year-olds.

### Step 7 — Find candidate index events

Input: the event table and code set.
Output: candidate rows at the declared grain, with person, event, and effective time.

For an admission-based cohort, join the condition to the visit and keep the visit start as the candidate
index date. For a diagnosis-based cohort, use the diagnosis date. Do not mix the two.

Reason: the index date anchors every downstream window. Mixing diagnosis dates and admission dates
creates follow-up windows that start before the patient was in the hospital.

### Step 8 — Deduplicate candidate events

Input: candidate rows, possibly duplicated by interfaces or multiple diagnosis rows.
Output: one row per person per index event.

Use `ROW_NUMBER() OVER (PARTITION BY person_id, encounter_id ORDER BY diagnosed_ts, diagnosis_id)` and
keep `rn = 1`. The tie-breaker matters. If two diagnosis rows have the same timestamp, the lower
`diagnosis_id` is arbitrary but deterministic; document that choice.

Reason: `DISTINCT` on all columns retains rows that differ only in an identifier, and `DISTINCT` on
selected columns collapses legitimate repeats. An explicit window makes the rule visible.

### Step 9 — Apply the lookback for prevalent cases

Input: deduplicated candidates and prior history.
Output: candidates with no qualifying event in the lookback window.

Use a half-open window: prior event time `>= index_date - 365 days` and `< index_date`. A prior event on
the index date is ambiguous; decide whether same-day counts as prior and state it.

Reason: "first ever" and "first in period" are different cohorts. A 365-day lookback approximates
incident cases without requiring lifetime history, which is rarely available.

### Step 10 — Select the first qualifying event per person

Input: candidates after exclusions.
Output: one index event per person.

Order by index date and a deterministic tie-breaker. If two admissions on the same day qualify, choose
the lower encounter identifier and record the rule.

Reason: a patient-level cohort needs one anchor. Selecting arbitrarily makes the cohort unstable across
runs and engines.

### Step 11 — Apply clinical exclusions

Input: the index cohort and exclusion criteria.
Output: the final cohort with an exclusion reason per removed person.

Common exclusions: age below threshold, prior heart failure, end-stage renal disease, hospice, transfer
to another acute hospital, death during the index admission, missing discharge date. Each exclusion
should be a separate flag, not a chain of `WHERE` clauses, so the attrition can be reported.

Reason: a single final count cannot explain why the cohort changed. An attrition table with counts per
criterion is the evidence that the definition was applied as intended.

### Step 12 — Build the encounter spine

Input: the cohort and all encounters.
Output: one row per encounter for cohort persons, with admission and discharge times.

Use the index encounter plus subsequent encounters. Keep transfers separate if the source does, or merge
them with a documented rule. A transfer that is counted as a readmission inflates the outcome rate.

Reason: readmission and length-of-stay calculations depend on whether transfers are separate episodes.
The rule must match the clinical definition.

### Step 13 — Join events on identity and time

Input: the spine and event tables such as labs or medications.
Output: event rows attributed to the correct encounter.

Join on `person_id` and on `effective_time >= admit_time - pre_window` and `< discharge_time`. Use a
half-open interval. Decide the pre-admission window from the clinical question, not from convenience.

Reason: joining on person alone attributes years of history to one admission. The pre-admission window
is a clinical choice: a lactate drawn two hours before admission may be the triage value that triggered
the admission.

### Step 14 — Handle multiple events per encounter

Input: event rows after the temporal join.
Output: one value per encounter per concept, using a stated aggregation.

Options: first value, last value, worst value, mean, or all values in a separate table. "First lactate"
needs an ordering by effective time and a tie-breaker. "Worst lactate" needs a defined direction
(highest). Document the choice; it changes the feature.

Reason: an encounter with six lactate values cannot be represented by one row unless the aggregation is
stated. Averaging a series that rises and falls produces a value that never occurred.

### Step 15 — Compute the outcome with a fixed window

Input: the cohort and subsequent encounters.
Output: a binary outcome per index admission.

For 30-day readmission: find the first subsequent admission with `admit_time >= index_discharge` and
`admit_time < index_discharge + 30 days`. Exclude transfers and planned admissions if the definition
does. If the patient dies before 30 days, decide whether death is a competing event, a censoring event,
or an outcome.

Reason: the denominator and the competing-risk rule change the rate. A 30-day readmission rate that
counts deaths as non-readmissions is not comparable to one that censors them.

### Step 16 — Aggregate at the declared grain

Input: the joined and filtered rows.
Output: the final result set.

Use `COUNT(DISTINCT person_id)` for persons and `COUNT(*)` for events, and label each column with its
unit. A result column named `count` without stating whether it counts persons, admissions, or rows is
not a deliverable.

Reason: the most common reporting error is a count whose unit is unstated. The number is then reused in
a rate with the wrong denominator.

### Step 17 — Test uniqueness and referential integrity

Input: the final and intermediate tables.
Output: pass or fail per test.

Tests: `person_id` unique in the person-level cohort; every `encounter_id` in the cohort exists in the
encounter table; no discharge before admission; no index date outside the study window; no negative
follow-up time.

Reason: these tests catch the errors that produce plausible numbers. A cohort with duplicate persons
passes a row-count check and fails a uniqueness check.

### Step 18 — Test temporal ordering and coverage

Input: the cohort and event tables.
Output: counts of impossible or missing sequences.

Checks: labs before birth, encounters after death, medications ending before they start, observation
periods shorter than the lookback, index dates outside enrollment. Report the count and the affected
persons, not just a pass or fail.

Reason: a small number of impossible rows may be legitimate source errors; a large number indicates a
systematic join problem. The count distinguishes the two.

### Step 19 — Compare against a known baseline

Input: the cohort and a previously validated definition or published estimate.
Output: a difference and an explanation.

If the new cohort is 20% larger than the prior definition, find which criterion accounts for the
difference. Run the query with each criterion disabled in turn and report the attrition. A difference is
not necessarily an error, but an unexplained difference is not a result.

Reason: regression testing is how cohort definitions earn trust. The comparison also catches code-set
changes that arrived with a vocabulary update.

### Step 20 — Package the query as a versioned artifact

Input: the final SQL and its parameters.
Output: a file in version control with a definition document, a parameter block, and a result hash.

Parameters: study start, study end, lookback days, age threshold, code-set version, minimum observation
days. The query should read these from a configuration table or templating layer, not from hard-coded
literals scattered through the SQL.

Reason: a cohort definition that cannot be re-run with a different date range is not reusable. The
parameter block is also the interface that lets a reviewer see what changed between versions.

### Step 21 — Handle late-arriving and superseded records

Input: event rows with effective time and recorded time.
Output: one current row per clinical fact, plus a history table.

When a record arrives late, its effective time places it in a past partition. When a record is
superseded, a later `recorded_time` carries a corrected value. The rule is: select the row with the
greatest `recorded_time` per clinical key, and retain the superseded rows in a history table. Never
delete the prior value; a cohort that was published last month must remain explainable.

Reason: a query that uses `recorded_time <= run_time` produces the state as known at that time; a query
that uses the latest row produces the current state. Both are valid, but they answer different questions
and must be labeled.

### Step 22 — Handle nulls, sentinels, and identifier collisions

Input: the source values.
Output: a documented mapping for each non-standard value.

Sentinel values (`-1`, `9999`, `1900-01-01`, `"NULL"`, empty string) must be mapped to null before any
aggregation, or they become real measurements. Identifiers must be joined on `(system, value)`, not on
value alone, because two facilities can issue the same medical record number. A null in a join key must
be quarantined, not silently dropped by the join.

Reason: sentinels and collisions produce correct-looking results with wrong denominators. They are
invisible in a row count and visible only in a value audit.

### Step 23 — Decide partial-failure behavior

Input: the load and transform steps.
Output: a transaction boundary and a recovery rule for each.

A transform that writes in place must run in a transaction or write to a staging table and swap. A load
that fails halfway must leave the prior partition intact. A query that reads while a load is in progress
must read the prior snapshot, not a partially written table.

| Failure point | State after failure | Recovery |
|---|---|---|
| Extract | No new partition | Retry the extract |
| Transform | Staging table partial, target untouched | Truncate staging, retry |
| Load | Transaction rolled back | Retry the load |
| Post-load test | Target replaced, test failed | Restore prior snapshot, fix, reload |
| Publish | Consumer not switched | Republish after validation |

Reason: a partial commit is worse than a failure because it looks like success. The transaction boundary
is the design decision that prevents it.

### Branch point A — The question is a rate, not a cohort

If the deliverable is a rate, define the numerator, denominator, and time at risk before writing SQL.
A rate with a person-time denominator needs one row per person-interval, not one row per person. The
grain step changes accordingly.

### Branch point B — The data are claims, not EHR

Claims have enrollment spans, claim lag, and diagnosis codes that are billing artifacts. Add a claim-lag
window, use the claim through date rather than the service date for completeness, and expect diagnosis
codes to be less specific. The temporal logic is the same; the interpretation is not.

### Branch point C — The cohort spans multiple sites

Run the same query against each site's conformed model and compare attrition at every step. A site with
a much larger drop at the code-set step has a mapping problem, not a different patient population. Do
not pool without a site-level attrition report.

## 6. Worked example

A health system wants the 30-day readmission rate for adults with a first heart-failure admission in
2024. The analytic table is OMOP-shaped. The definition is versioned `HF_READMIT_2024_v3`.

### 6.1 Schema

```sql
-- PostgreSQL 18 / OMOP-shaped teaching schema (abridged)
CREATE TABLE person (
    person_id           bigint PRIMARY KEY,
    year_of_birth       integer NOT NULL,
    gender_concept_id   integer NOT NULL,
    race_concept_id     integer
);

CREATE TABLE observation_period (
    person_id           bigint NOT NULL,
    period_start        date   NOT NULL,
    period_end          date   NOT NULL,
    PRIMARY KEY (person_id, period_start)
);

CREATE TABLE visit_occurrence (
    visit_occurrence_id bigint PRIMARY KEY,
    person_id           bigint NOT NULL REFERENCES person (person_id),
    visit_start_datetime timestamptz NOT NULL,
    visit_end_datetime   timestamptz,
    visit_type_concept_id integer NOT NULL,
    discharge_disposition_concept_id integer
);

CREATE TABLE condition_occurrence (
    condition_occurrence_id bigint PRIMARY KEY,
    person_id               bigint NOT NULL REFERENCES person (person_id),
    visit_occurrence_id     bigint REFERENCES visit_occurrence (visit_occurrence_id),
    condition_concept_id    integer NOT NULL,
    condition_start_datetime timestamptz NOT NULL,
    condition_source_value  text
);

CREATE TABLE measurement (
    measurement_id      bigint PRIMARY KEY,
    person_id           bigint NOT NULL REFERENCES person (person_id),
    visit_occurrence_id bigint REFERENCES visit_occurrence (visit_occurrence_id),
    measurement_concept_id integer NOT NULL,
    measurement_datetime timestamptz NOT NULL,
    value_as_number     numeric,
    unit_source_value   text
);

CREATE TABLE drug_exposure (
    drug_exposure_id    bigint PRIMARY KEY,
    person_id           bigint NOT NULL REFERENCES person (person_id),
    visit_occurrence_id bigint REFERENCES visit_occurrence (visit_occurrence_id),
    drug_concept_id     integer NOT NULL,
    drug_exposure_start_datetime timestamptz NOT NULL,
    drug_exposure_end_datetime   timestamptz
);

CREATE TABLE death (
    person_id  bigint PRIMARY KEY REFERENCES person (person_id),
    death_date date NOT NULL
);
```

### 6.2 Population counts

| Step | Criterion | Persons remaining | Removed |
|---|---|---:|---:|
| 0 | All persons with any 2024 encounter | 120,000 | — |
| 1 | At least one heart-failure diagnosis in 2024 | 9,100 | 110,900 |
| 2 | Age 18 or older at index | 8,968 | 132 |
| 3 | No heart-failure diagnosis in the 365 days before index | 7,764 | 1,204 |
| 4 | Index encounter has a discharge date and is not a transfer | 7,668 | 96 |
| 5 | Alive at index discharge | 7,627 | 41 |
| 6 | At least 365 days of observation before index | 7,412 | 215 |

The final cohort is 7,412 persons and 7,412 index admissions, because the definition selects the first
qualifying admission per person.

The diagnosis table contains 14,820 heart-failure diagnosis rows in 2024. After deduplication to one row
per person per encounter, there are 11,932 person-encounter pairs. The first qualifying encounter per
person yields 9,100 persons. That difference, 14,820 to 9,100, is why counting diagnosis rows overstates
the cohort by 63%.

Calculation: 14,820 / 9,100 = 1.6286, so diagnosis rows are 62.9% more numerous than persons. Reporting
14,820 as "patients with heart failure" overstates the population by 5,720 persons.

### 6.3 Readmission outcome

Of the 7,412 index admissions, 1,118 have a subsequent unplanned admission within 30 days of discharge.
The rate is 1,118 / 7,412 = 0.15084, or 15.1%.

If transfers are counted as separate admissions, 212 additional "readmissions" appear, raising the count
to 1,330 and the rate to 17.9%. The difference, 2.9 percentage points, is a definitional artifact. The
v3 definition merges transfers that occur within 24 hours and share a discharge disposition indicating
transfer.

If deaths within 30 days are counted as non-readmissions, the denominator stays 7,412. If they are
censored, 96 patients leave the denominator and the rate becomes 1,118 / 7,316 = 15.28%. The definition
censures deaths and reports both the rate and the censoring count; a rate without the censoring rule is
not comparable across systems.

### 6.4 Temporal join check

The admission lactate is defined as the first lactate measurement in `[admit - 6 hours, discharge)`.
There are 412,000 lactate measurements for cohort persons in 2024. The temporal join matches 386,540,
a match rate of 386,540 / 412,000 = 93.82%. The 25,460 unmatched measurements are either outside any
encounter window or belong to encounters not in the spine. The pipeline reports both counts; it does not
drop the unmatched rows silently.

### 6.5 What the query must state

The final definition document records: code set `HF_SNOMED_2024-09`, vocabulary release September 2024,
lookback 365 days, observation requirement 365 days, transfer merge window 24 hours, death censoring at
30 days, index date = admission date, and age computed at index admission using year of birth when month
and day are unavailable. Each of these is a parameter, not a comment.

### 6.6 Row-level reconciliation

The person-level attrition in Section 6.2 hides the row-level work. The diagnosis table is the largest
source of duplication.

| Stage | Rows or persons | Calculation |
|---|---:|---|
| HF diagnosis rows in 2024 | 14,820 | Source rows with a qualifying concept |
| Distinct `(person_id, encounter_id)` | 11,932 | Deduplicate diagnosis rows per encounter |
| Distinct persons with any HF diagnosis | 9,100 | Deduplicate to person |
| First qualifying admission per person | 9,100 | `ROW_NUMBER() = 1` |
| Age 18 or older | 8,968 | −132 |
| No prior HF in 365 days | 7,764 | −1,204 |
| Discharge present, not a transfer | 7,668 | −96 |
| Alive at discharge | 7,627 | −41 |
| 365 days of observation | 7,412 | −215 |

The ratio 14,820 / 9,100 = 1.6286 means there are 62.9% more diagnosis rows than persons. The ratio
14,820 / 11,932 = 1.2420 means there are 24.2% more diagnosis rows than person-encounter pairs. A report
that quotes 14,820 as the cohort size is wrong by 5,720 persons.

### 6.7 Second run and determinism

The cohort query is run twice on the same frozen snapshot with the same parameters.

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Diagnosis rows scanned | 14,820 | 14,820 | Equal |
| Distinct persons | 9,100 | 9,100 | Equal |
| Final cohort | 7,412 | 7,412 | Equal |
| Readmissions | 1,118 | 1,118 | Equal |
| Cohort hash | `4c8e...1a72` | `4c8e...1a72` | Equal |

A controlled experiment removes the `visit_occurrence_id` tie-breaker from the window ordering. The
cohort then differs by 41 persons between runs, 41 / 7,412 = 0.55%. That is the size of the
nondeterminism that a missing tie-breaker introduces: small enough to pass a casual review, large enough
to change a secondary analysis.

### 6.8 Two diagnostic checks

**Check A: index date inside the study window.** Expected 0 rows outside 2024. Observed 0. Pass. A
non-zero count means the interval predicate is wrong or a timezone conversion moved a boundary event.

**Check B: readmission chart audit.** A reviewer samples 50 readmissions and finds 7 that the definition
should have excluded: 4 transfers, 2 planned admissions, and 1 admission at an outside hospital that the
discharge disposition did not flag. The discrepancy rate is 7 / 50 = 14.0%. Threshold 5%. Fail. The
numerator is overstated by approximately 14%, which means the reported 15.1% rate is closer to 13.0% if
the audit finding generalizes: 1,118 × 0.86 = 961; 961 / 7,412 = 12.97%.

**Check C: duplicate clinical key rate.** Observed 0.9% of lab rows with the same
`(person_id, code, effective_time)`. Threshold 0.5%. Fail. The duplicates come from two senders and
inflate the per-encounter lab count without changing the person-level cohort.

### 6.9 Decision table for a changed cohort count

| Symptom | First diagnostic | Likely cause | Action |
|---|---|---|---|
| Count up 15-20% after refresh | Attrition by criterion | New code mappings or a new site | Compare code-set versions; review mapping |
| Count up <2% | Duplicate rate by source | Retransmission | Make load idempotent; deduplicate |
| Count down 5-10% | Observation-period attrition | Enrollment file changed | Reconcile enrollment dates |
| Count stable, rate changed | Numerator audit | Transfer or planned admission definition | Reclassify; document |
| Count changes between identical runs | Window ordering | Missing tie-breaker | Add unique sort key |
| Count differs by site | Site-level attrition | Coding or coverage difference | Report per site; do not pool blindly |

## 7. Code, query, or calculation

### 7.1 Executable SQLite cohort query

SQLite 3.50.4 supports window functions and CTEs. The script creates an OMOP-shaped mini-schema, inserts
deliberate edge cases, and runs the cohort definition. Expected output is stated after the code.

```python
import sqlite3

con = sqlite3.connect(":memory:")
con.executescript("""
CREATE TABLE person (
    person_id INTEGER PRIMARY KEY,
    year_of_birth INTEGER NOT NULL
);
CREATE TABLE visit_occurrence (
    visit_occurrence_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    visit_start TEXT NOT NULL,
    visit_end TEXT,
    visit_type TEXT NOT NULL
);
CREATE TABLE condition_occurrence (
    condition_occurrence_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    visit_occurrence_id INTEGER NOT NULL,
    condition_concept_id INTEGER NOT NULL,
    condition_start TEXT NOT NULL
);
CREATE TABLE measurement (
    measurement_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    visit_occurrence_id INTEGER,
    measurement_concept_id INTEGER NOT NULL,
    measurement_time TEXT NOT NULL,
    value_as_number REAL
);

INSERT INTO person VALUES
 (1, 1950), (2, 1960), (3, 2010), (4, 1970), (5, 1980);

INSERT INTO visit_occurrence VALUES
 (1, 1, '2024-01-05 08:00:00', '2024-01-10 12:00:00', 'inpatient'),
 (2, 1, '2024-03-01 09:00:00', '2024-03-04 09:00:00', 'inpatient'),
 (3, 2, '2023-12-20 10:00:00', '2023-12-28 10:00:00', 'inpatient'),
 (4, 2, '2024-02-10 08:00:00', '2024-02-15 08:00:00', 'inpatient'),
 (5, 3, '2024-04-01 08:00:00', '2024-04-03 08:00:00', 'inpatient'),
 (6, 4, '2024-05-01 08:00:00', '2024-05-02 08:00:00', 'inpatient'),
 (7, 5, '2024-06-01 08:00:00', '2024-06-05 08:00:00', 'inpatient');

-- p1 has a duplicate diagnosis row for visit 1
INSERT INTO condition_occurrence VALUES
 (1, 1, 1, 100, '2024-01-06 10:00:00'),
 (2, 1, 1, 100, '2024-01-06 10:00:00'),
 (3, 1, 2, 100, '2024-03-02 10:00:00'),
 (4, 2, 3, 100, '2023-12-22 10:00:00'),
 (5, 2, 4, 100, '2024-02-11 10:00:00'),
 (6, 3, 5, 100, '2024-04-02 10:00:00'),
 (7, 4, 6, 100, '2024-05-01 10:00:00'),
 (8, 5, 7, 100, '2024-06-02 10:00:00');

-- one lab 2 hours before admission, one 8 hours before, one during
INSERT INTO measurement VALUES
 (1, 1, 1, 200, '2024-01-05 06:00:00', 2.4),
 (2, 1, 1, 200, '2024-01-05 00:00:00', 1.9),
 (3, 1, 1, 200, '2024-01-06 12:00:00', 2.8);
""")

cohort_sql = """
WITH hf AS (
    SELECT
        c.person_id,
        c.visit_occurrence_id,
        v.visit_start,
        c.condition_start,
        ROW_NUMBER() OVER (
            PARTITION BY c.person_id, c.visit_occurrence_id
            ORDER BY c.condition_start, c.condition_occurrence_id
        ) AS dx_rn
    FROM condition_occurrence c
    JOIN visit_occurrence v
      ON v.visit_occurrence_id = c.visit_occurrence_id
    WHERE c.condition_concept_id = 100
      AND v.visit_start >= '2024-01-01'
      AND v.visit_start < '2025-01-01'
),
first_dx AS (
    SELECT
        person_id,
        visit_occurrence_id,
        visit_start,
        ROW_NUMBER() OVER (
            PARTITION BY person_id
            ORDER BY visit_start, visit_occurrence_id
        ) AS rn
    FROM hf
    WHERE dx_rn = 1
),
with_age AS (
    SELECT
        f.*,
        CAST(substr(f.visit_start, 1, 4) AS INTEGER) - p.year_of_birth AS age
    FROM first_dx f
    JOIN person p ON p.person_id = f.person_id
    WHERE f.rn = 1
)
SELECT a.person_id, a.visit_occurrence_id, a.visit_start, a.age
FROM with_age a
WHERE a.age >= 18
  AND NOT EXISTS (
      SELECT 1
      FROM condition_occurrence prior
      WHERE prior.person_id = a.person_id
        AND prior.condition_concept_id = 100
        AND prior.condition_start >= date(a.visit_start, '-365 days')
        AND prior.condition_start < a.visit_start
  )
ORDER BY a.person_id;
"""

rows = con.execute(cohort_sql).fetchall()
print("cohort")
for row in rows:
    print(row)

lab_sql = """
SELECT m.measurement_id, m.measurement_time, m.value_as_number
FROM measurement m
JOIN visit_occurrence v
  ON v.visit_occurrence_id = m.visit_occurrence_id
WHERE m.measurement_concept_id = 200
  AND m.measurement_time >= datetime(v.visit_start, '-6 hours')
  AND m.measurement_time < v.visit_end
ORDER BY m.measurement_id;
"""
print("labs_in_window")
for row in con.execute(lab_sql):
    print(row)
```

Expected output:

```text
cohort
(1, 1, '2024-01-05 08:00:00', 74)
(4, 6, '2024-05-01 08:00:00', 54)
(5, 7, '2024-06-01 08:00:00', 44)
labs_in_window
(1, '2024-01-05 06:00:00', 2.4)
(3, '2024-01-06 12:00:00', 2.8)
```

Person 1 is included once despite the duplicate diagnosis and the second admission. Person 2 is excluded
because the December 2023 admission is within the 365-day lookback. Person 3 is excluded by age. The lab
at `2024-01-05 00:00:00` is 8 hours before admission and falls outside the 6-hour window.

### 7.2 PostgreSQL 18 gaps-and-islands

PostgreSQL 18 is the target engine. The query collapses overlapping infusion intervals per person into
continuous episodes with a tolerance of one hour.

```sql
-- PostgreSQL 18
WITH ordered AS (
    SELECT
        person_id,
        start_ts,
        end_ts,
        start_ts - LAG(end_ts) OVER (
            PARTITION BY person_id ORDER BY start_ts, end_ts
        ) AS gap
    FROM infusion
),
marked AS (
    SELECT
        person_id, start_ts, end_ts,
        SUM(CASE WHEN gap IS NULL OR gap > INTERVAL '1 hour' THEN 1 ELSE 0 END)
            OVER (PARTITION BY person_id ORDER BY start_ts, end_ts
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp
    FROM ordered
)
SELECT
    person_id,
    MIN(start_ts) AS episode_start,
    MAX(end_ts)   AS episode_end,
    COUNT(*)      AS source_rows,
    MAX(end_ts) - MIN(start_ts) AS episode_duration
FROM marked
GROUP BY person_id, grp
ORDER BY person_id, episode_start;
```

The `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` frame is explicit. Without it, the default frame
for an aggregate window with `ORDER BY` is `RANGE UNBOUNDED PRECEDING`, which includes peers with the
same `start_ts` and can assign two rows to the same group incorrectly.

### 7.3 Dialect notes

| Construct | PostgreSQL 18 | BigQuery | Snowflake | DuckDB 1.4 |
|---|---|---|---|---|
| Date literal | `DATE '2024-01-01'` | `DATE '2024-01-01'` | `DATE '2024-01-01'` | `DATE '2024-01-01'` |
| Subtract interval | `ts - INTERVAL '6 hours'` | `TIMESTAMP_SUB(ts, INTERVAL 6 HOUR)` | `DATEADD(hour, -6, ts)` | `ts - INTERVAL 6 HOUR` |
| Safe cast | `CAST(x AS integer)` | `SAFE_CAST(x AS INT64)` | `TRY_TO_NUMBER(x)` | `TRY_CAST(x AS INTEGER)` |
| Null ordering | `NULLS LAST` supported | `NULLS LAST` supported | `NULLS LAST` supported | `NULLS LAST` supported |
| As-of join | `LATERAL` or window | `LATERAL` or window | `ASOF JOIN` (limited) | `ASOF JOIN` native |
| Qualify window | subquery | `QUALIFY` | `QUALIFY` | `QUALIFY` |

The semantic differences that change results are null ordering, timestamp precision, and implicit casts.
A query that relies on the default null order is not portable. A query that compares a `date` to a
`timestamp` may silently truncate in one engine and error in another.

## 8. Data, standards, and regulatory context

### 8.1 OMOP CDM

The OMOP Common Data Model, maintained by the Observational Health Data Sciences and Informatics
(OHDSI) community, is at version 5.4 (release 5.4.3, August 2025), with version 5.5 released. Version 5.4
is the version most deployed tools target. The tables used in this topic map as follows:

| Clinical concept | OMOP table | Key fields |
|---|---|---|
| Person | `PERSON` | `person_id`, `year_of_birth`, `gender_concept_id` |
| Enrollment | `OBSERVATION_PERIOD` | `observation_period_start_date`, `observation_period_end_date` |
| Visit | `VISIT_OCCURRENCE` | `visit_start_date`, `visit_end_date`, `visit_concept_id` |
| Diagnosis | `CONDITION_OCCURRENCE` | `condition_concept_id`, `condition_start_date` |
| Lab | `MEASUREMENT` | `measurement_concept_id`, `value_as_number`, `unit_concept_id` |
| Drug | `DRUG_EXPOSURE` | `drug_concept_id`, `drug_exposure_start_date` |
| Death | `DEATH` | `death_date` |

OMOP uses `concept_id = 0` for "no matching concept." A query that filters `concept_id IN (...)` excludes
unmapped rows, which is correct for a standard-concept cohort but must be reported as mapping loss. The
`*_source_value` fields retain the original code and are the audit trail for the mapping.

OMOP is a community specification, not a regulation. Its authority comes from the tools that read it:
ATLAS for cohort definition, Achilles and DataQualityDashboard for characterization, and the OHDSI
methods library for estimation. A query written against OMOP is portable to the extent that the local
ETL followed the conventions, which varies by site.

### 8.2 SQL standards and dialect reality

SQL is standardized by ISO/IEC 9075, with the current edition being SQL:2023. Window functions entered
the standard in SQL:2003; `MERGE` in SQL:2003; `JSON_TABLE` in SQL:2016; property graph queries in
SQL:2023. No clinical warehouse implements the whole standard, and every engine has extensions.

The practical rule is to write ANSI-shaped SQL and isolate dialect-specific constructs in a small number
of models. dbt, which is covered in Topic 10, supports this through adapter macros and cross-database
macros, but it does not make semantics portable. A `DATEADD` macro can translate syntax; it cannot
translate the difference between a timestamp with time zone and one without.

### 8.3 What the standards do not constrain

Neither OMOP nor SQL specifies:

- The cohort definition itself.
- The lookback length.
- Whether transfers are separate visits.
- How to handle deaths in a readmission rate.
- The observation-period rule.
- The tie-breaking rule for duplicate events.

Those are clinical and methodological choices. They belong in a definition document, and they are the
part a reviewer should challenge. A query can be syntactically perfect and methodologically indefensible.

### 8.4 Regulatory touchpoints

A cohort query that feeds a regulatory submission or a certified product becomes evidence. Under 21 CFR
Part 11, electronic records must have secure, computer-generated, time-stamped audit trails, and changes
must not obscure prior values. That means the query version, the parameter values, and the input snapshot
must be retained alongside the result. A SQL file in a personal folder is not an auditable record.

Under HIPAA, a query that returns identifiable data is a use or disclosure of PHI and must be authorized,
minimum necessary, and logged. A query that returns only aggregate counts may still be identifiable if
the cells are small; many organizations apply a minimum cell size of 11 before releasing counts. That
threshold is organizational policy, not a HIPAA requirement, and should be labeled as such.

## 9. Industry practice and case evidence

### 9.1 OHDSI cohort definitions

OHDSI's ATLAS tool represents a cohort as a structured JSON definition with inclusion rules, index
events, and censoring. The definition can be exported, versioned, and executed against any OMOP instance.
This is the most mature example of cohort logic as a portable artifact rather than a query. The
documented limitation is that the same definition can produce different counts across sites because of
differences in source coding and ETL, which is why OHDSI recommends running diagnostics and comparing
attrition rather than pooling counts directly.

### 9.2 MIMIC-IV and reproducible cohort code

MIMIC-IV (version 3.1, released 11 October 2024) provides an openly documented schema and a companion
code repository with derived concepts. The documentation and the code are part of the dataset's value:
without them, an "open dataset" is a collection of tables with no agreed definitions. The MIMIC code
repository's derived schema is versioned and records the dataset version in a metadata table, which is
the practice this topic recommends.

### 9.3 dbt and SQL as a tested artifact

dbt's model contracts allow a model to declare column names and data types and to fail the build when the
output does not match. dbt tests cover uniqueness, not-null, accepted values, and relationships. The
combination turns a SQL model into an interface with executable guarantees. The limitation is that dbt
tests check structure and simple constraints, not clinical plausibility; a cohort can pass every generic
test and still include the wrong patients.

### 9.4 What is inference

- That a 365-day lookback is appropriate for heart failure. It is a common convention, not a validated
  standard; the correct lookback depends on the condition and the data.
- That 30-day readmission is a good quality measure. It is widely used and widely criticized for
  attributing social and outpatient factors to hospital care.
- That OMOP mapping preserves enough clinical detail for a given study. That is an empirical question
  answered by a mapping-loss report, not by the existence of a mapping.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong grain | Counts differ from the clinical expectation by a round factor | Aggregating event rows as if they were persons | Inflated cohort and rates | State grain; test uniqueness |
| Many-to-many join | Row count grows after a join that should be one-to-many | Joining on person only | Duplicated outcomes | Declare cardinality before joining |
| Closed interval | Boundary events appear twice | `BETWEEN` on timestamps | Double counting at midnight | Half-open intervals |
| Nondeterministic window | Cohort changes between runs | `ORDER BY` without a tie-breaker | Unreproducible results | Add a unique tie-breaker |
| `NOT IN` with nulls | Exclusion returns zero rows | Three-valued logic | Silent over-exclusion | Anti-join with `NOT EXISTS` |
| Left join turned inner | Exclusion count equals the full cohort | Predicate on the right table in `WHERE` | Wrong denominator | Put predicates in the `ON` clause or test null explicitly |
| Age from year only | Borderline patients shift by one year | Integer subtraction | Small but systematic bias | Compute exact age when possible |
| Code set drift | Cohort size jumps after a vocabulary update | New concept mappings | Definition changed without notice | Version the code set |
| Timezone mixing | Events cluster at odd hours or shift by a day | Local time stored without offset | Temporal misordering | Store UTC with offset |
| Unstated aggregation | Feature values cannot be reproduced from source | Mean of a series | Value never observed | Document first/last/worst |
| Transfer counted as readmission | Readmission rate higher than peer institutions | Separate visit rows | Inflated outcome | Merge transfers by rule |
| Death handling unstated | Rates not comparable across studies | Competing risk ignored | Biased estimate | State censoring rule |
| Query in a notebook | No version, no tests, no owner | Manual filters | Cannot reproduce or audit | Versioned model with tests |

Two red flags are worth naming separately.

**A count with no unit.** If the result column is called `patients` but the query counts diagnosis rows,
the error will propagate into every ratio. Ask for the `COUNT(DISTINCT person_id)` and the `COUNT(*)`
side by side; the ratio between them is the duplication factor.

**A cohort that cannot be rebuilt.** If the analyst cannot re-run the query on the same snapshot and get
the same number, the definition is not a definition. The usual causes are `now()` in a predicate, a
window without a tie-breaker, or a dependency on row order without `ORDER BY`.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Source records are complete for the observation window | Prior history and outcomes depend on it | Patient moves or uses an outside provider | Compare enrollment gaps to encounter dates |
| Effective timestamps are accurate | Temporal joins depend on them | Interface sets all times to midnight | Check time-of-day distribution |
| Code mappings preserve clinical meaning | Cohort code sets rely on them | Local codes map to `concept_id = 0` | Mapping-loss report by domain |
| Transfers are identifiable | Readmission definition depends on it | Source does not record disposition | Audit a sample of 50 stays |
| Death data are complete | Censoring depends on it | Deaths outside the system are missing | Link to state death index if available |
| One row per encounter after deduplication | Rates use encounter denominators | Duplicate interface messages | Uniqueness test on encounter key |
| The lookback is long enough | Incidence depends on it | Prior diagnosis outside the window | Sensitivity analysis at 180/365/730 days |
| Age is known precisely enough | Eligibility depends on it | Only birth year is available | Report the approximation and its effect |
| The study period is stable | Reproducibility depends on it | Data refresh adds retroactive records | Freeze the snapshot and record its hash |

The largest limit is that SQL answers the question it is given, not the question that was meant. A
perfectly written query for "first admission with any heart-failure diagnosis" is still the wrong query
if the clinical team meant "principal diagnosis." The definition document is the only defense, and it
must be reviewed by someone who knows the clinical domain.

A second limit is that cohort definitions are local. A definition validated at one site may not transfer
because coding practices, discharge dispositions, and observation rules differ. Multi-site work requires
site-level diagnostics, not just a shared query.

## 12. Exercises

### Exercise 1 — Grain inventory (recall, 20 minutes)

Given the six tables in Section 6.1, deliverable: a table with grain, primary key, effective time, and
recorded time for each.

Acceptance criteria: each grain is one sentence; at least one table has a composite natural key; the
difference between effective and recorded time is stated for the measurement table.

### Exercise 2 — Deduplicate with a rule (application, 30 minutes)

The condition table has three rows for the same person, encounter, and code, with timestamps one second
apart and different identifiers. Deliverable: a query that returns one row and a note explaining what the
tie-breaker chooses.

Acceptance criteria: the query uses `ROW_NUMBER()` with a unique ordering; the result is stable across two
runs; the note states what information is lost.

### Exercise 3 — Temporal join (application, 45 minutes)

Deliverable: a query that attributes the first lactate in `[admit - 6 hours, discharge)` to each index
admission, plus a count of admissions with no qualifying lab.

Acceptance criteria: half-open interval; deterministic first-value rule; unmatched admissions counted, not
dropped; result includes the pre-admission lab when present.

### Exercise 4 — Build the cohort (design, 90 minutes)

Implement `HF_READMIT_2024_v3` against the teaching schema, including the attrition table. Deliverable:
SQL file, parameter block, and attrition counts.

Acceptance criteria: every exclusion is a separate flag; the final count reconciles to the attrition
table; the code-set version is a parameter; no hard-coded dates outside the parameter block.

### Exercise 5 — Gaps and islands (application, 45 minutes)

Given infusion rows with overlapping intervals, deliverable: a query that returns one row per continuous
episode with a one-hour tolerance, plus the number of source rows per episode.

Acceptance criteria: adjacent intervals merge; a gap of 61 minutes splits; the frame is explicit; the
result is ordered deterministically.

### Exercise 6 — Wrong but plausible (critique, 40 minutes)

A query returns 14,820 "patients" with heart failure. Deliverable: a 300-word diagnosis of the grain
error and a corrected query.

Acceptance criteria: identifies the table grain; computes the duplication factor; states the corrected
person count; explains why `DISTINCT` on all columns does not fix it.

### Exercise 7 — Dialect translation (application, 40 minutes)

Translate the temporal join in Section 7.3 to BigQuery and Snowflake. Deliverable: two queries and a list
of semantic differences that could change results.

Acceptance criteria: interval arithmetic is correct in each dialect; at least two semantic differences
are named; the queries state their engine and version.

### Exercise 8 — Write the tests (design, 60 minutes)

Deliverable: six SQL tests for the cohort: person uniqueness, encounter referential integrity, temporal
ordering, index-date range, observation length, and readmission window non-negativity.

Acceptance criteria: each test returns a count of violations and a pass threshold; each has an owner and
an action; the tests run against a snapshot, not a live table.

## 13. Self-check answers

**1. What is grain, and why does it come first?**

Grain is what one row represents. It determines the meaning of every count, join, and window. A query
that aggregates at the wrong grain returns a number that is arithmetically correct and semantically
wrong; no later clause can repair it.

**2. Why is `BETWEEN` on timestamps dangerous?**

`BETWEEN` is inclusive on both ends. For a date range, it includes the whole final day only if the column
is a date; for a timestamp, it includes midnight at the start of the final day and excludes the rest of
that day, or includes both endpoints and double-counts boundary events when intervals are adjacent. Use
`>= start AND < end`.

**3. What does `ROW_NUMBER()` hide?**

It hides the duplicates it discards and the arbitrariness of the ordering. If the `ORDER BY` is not
unique, the selected row can differ between runs. If the partition is wrong, it can collapse legitimate
distinct events. The function is safe only when both partition and order are stated and justified.

**4. Compute the match rate.**

386,540 matched of 412,000 measurements: 386,540 / 412,000 = 0.9382039, or 93.82%. Unmatched = 412,000 −
386,540 = 25,460. If the pre-admission window were 12 hours instead of 6, the match rate would rise, but
the additional matches would include values less likely to have triggered the admission.

**5. Why is `NOT IN` risky for exclusions?**

If the subquery returns a null, `x NOT IN (1, 2, NULL)` evaluates to unknown for every x, and the
predicate returns no rows. `NOT EXISTS` is null-safe and expresses the anti-join directly.

**6. How do you handle a patient with two qualifying admissions on the same day?**

Add a deterministic tie-breaker, such as the lower encounter identifier, and document it. Alternatively,
define the index event at the person-day level and aggregate the two admissions. The important property
is that the rule is stated and stable, not that one particular admission is chosen.

**7. What is the difference between a semi-join and an anti-join?**

A semi-join returns left rows with at least one match and does not duplicate them; an anti-join returns
left rows with no match. Both are usually written with `EXISTS` and `NOT EXISTS`. `IN` can express a
semi-join but duplicates and nulls change the result.

**8. Compute the readmission rate under transfer merging.**

Without merging: 1,330 / 7,412 = 0.17944, or 17.9%. With merging: 1,118 / 7,412 = 0.15084, or 15.1%.
The absolute difference is 2.9 percentage points, or 212 admissions. The correct rate depends on the
clinical definition, not on which number is larger.

**9. Why version the code set?**

Because vocabulary releases add, deprecate, and remap concepts. A query that reads a code list from a
live table changes when the vocabulary changes, without any code change. Versioning makes the change
visible and allows the old cohort to be reproduced.

**10. What makes a window ordering deterministic?**

A total order on the partition. If the business key can tie, add a unique identifier as the final sort
key. Determinism is a property of the ordering, not of the engine.

## 14. Interview questions

**1. "Walk me through how you would build a cohort."**

Strong: starts with the clinical question and unit of analysis, declares grain, builds the spine, applies
lookback and exclusions as separate flags, tests uniqueness and temporal order, and reports attrition.
Weak: jumps to a `SELECT` with a list of `WHERE` clauses and no definition of first or prevalent.

**2. "How do you deduplicate clinical records?"**

Strong: names the partition, the ordering, the tie-breaker, and what is lost; distinguishes exact
duplicates from legitimate repeats; tests the result. Weak: `SELECT DISTINCT`, which either keeps
identifier-level duplicates or collapses real events.

**3. "What is the difference between a visit and an encounter?"**

Strong: explains that the terms are source-dependent, that a transfer may be a separate visit, and that
the choice affects readmission and length of stay. Weak: treats them as synonyms.

**4. "How do you handle late-arriving data in a cohort?"**

Strong: separates event time from load time, freezes a snapshot for the analysis, and defines a cutoff;
if the cohort is refreshed, versions the result and explains the delta. Weak: filters on load date and
assumes completeness.

**5. "Why can two correct queries return different cohort sizes?"**

Strong: different index definitions, lookback windows, observation requirements, transfer handling, or
code-set versions. Weak: "one of them is wrong," without a diagnostic method.

**6. "How do you test a cohort query?"**

Strong: uniqueness, referential integrity, temporal ordering, range checks, attrition reconciliation,
and comparison to a prior definition. Weak: "it runs without error."

**7. "What is the risk of `LEFT JOIN` plus a `WHERE` on the right table?"**

Strong: it becomes an inner join and silently removes the unmatched rows that the exclusion was meant to
find. Weak: knows the syntax but not the semantics.

**8. "How do you make SQL portable across engines?"**

Strong: keeps ANSI-shaped models, isolates dialect functions, tests semantics not just syntax, and
documents timestamp and null behavior. Weak: assumes a translation macro guarantees identical results.

**9. "What would you do if the cohort count changed by 15% after a data refresh?"**

Strong: compares attrition at each step, checks code-set and vocabulary versions, checks for duplicate
feeds and retroactive records, and quantifies each contribution before concluding. Weak: reruns the query
and accepts the new number.

**10. "When would you not use SQL?"**

Strong: when the computation is iterative or graph-based, such as probabilistic linkage or a sequence
model over long histories; SQL can prepare the inputs but not express the algorithm cleanly. Weak: "SQL
can do everything."

## 15. Key terms

- **Anti-join** — left rows with no match on the right.
- **As-of join** — join taking the most recent prior row.
- **Attrition table** — counts remaining after each cohort criterion.
- **Cardinality** — rows produced per input row by a join.
- **Cohort** — defined set of persons meeting criteria.
- **CTE** — named subquery used to structure a query.
- **Dialect** — engine-specific SQL syntax and semantics.
- **Effective time** — when the clinical event occurred.
- **Gaps and islands** — collapsing adjacent intervals into episodes.
- **Grain** — what one row represents.
- **Half-open interval** — `[start, end)`.
- **Index date** — anchor date for follow-up.
- **Incident / prevalent case** — new after lookback / existing before index.
- **Lookback** — prior period used to establish history.
- **Observation period** — span of complete data for a person.
- **OMOP CDM** — common data model for observational health data.
- **Predicate pushdown** — evaluating filters before joins.
- **Readmission** — subsequent admission within a defined window.
- **Recorded time** — when the event was entered into the source.
- **Semi-join** — left rows with at least one match.
- **Slowly changing dimension** — dimension preserving history.
- **Surrogate key** — generated identifier.
- **Temporal join** — join on identity and time.
- **Three-valued logic** — true, false, unknown.
- **Tie-breaker** — unique secondary sort key.
- **Transfer** — movement between units or facilities.
- **Washout** — period excluded from outcome ascertainment.
- **Window frame** — rows visible to a window function.
- **Window function** — computation across related rows without collapsing them.

## 16. Sources

All URLs accessed 25 October 2026.

1. OHDSI. "OMOP CDM v5.4." https://ohdsi.github.io/CommonDataModel/cdm54.html
2. OHDSI. "Changes v5.3 to v5.4." https://ohdsi.github.io/CommonDataModel/cdm54Changes.html
3. OHDSI. "CommonDataModel releases." https://github.com/OHDSI/CommonDataModel/releases
4. OHDSI. "ATLAS." https://ohdsi.github.io/Atlas/
5. PhysioNet. "MIMIC-IV v3.1." https://www.physionet.org/content/mimiciv/3.1/
6. Johnson, A. E. W. et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific
   Data, 2023. https://doi.org/10.1038/s41597-022-01899-x
7. PostgreSQL Global Development Group. "PostgreSQL 18 Documentation." https://www.postgresql.org/docs/18/
8. PostgreSQL Global Development Group. "Release 18." https://www.postgresql.org/docs/current/release-18.html
9. DuckDB. "AsOf Join." https://duckdb.org/docs/stable/guides/sql_features/asof_join
10. SQLite. "Window Functions." https://www.sqlite.org/windowfunctions.html
11. dbt Labs. "Model contracts." https://docs.getdbt.com/docs/mesh/govern/model-contracts
12. dbt Labs. "Add sources to your DAG." https://docs.getdbt.com/docs/build/sources
13. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
14. HHS. "Guidance Regarding Methods for De-identification of PHI." https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
15. ISO/IEC. "ISO/IEC 9075-2:2023 Database languages SQL." https://www.iso.org/standard/76584.html

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_architecture-patterns.md`](01_architecture-patterns.md)
- Next topic: [`03_python-engineering.md`](03_python-engineering.md)
- Related sibling: [`05_fhir-ingestion.md`](05_fhir-ingestion.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../course/03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.1 builds a cohort in DuckDB; Lab 3.6 injects a duplicate feed.
