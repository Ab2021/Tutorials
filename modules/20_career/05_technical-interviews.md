# Technical Interviews: SQL, Statistics, and Code Under Time Pressure

> Module 20 — Career, Portfolio, and Interviews
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 04
> Version 2026.3

## 1. Why this topic exists

A technical interview is a sample, not an examination. The employer has 45 to 60 minutes and needs
to infer whether you can produce correct work under constraints, explain your reasoning to a
non-specialist, and notice when you are wrong. Those three inferences are not the same as knowing
statistics or SQL, and candidates who prepare only knowledge fail on the other two.

The format has changed in the last few years. Employers report moving away from memorisable
syntax questions, which generative assistants now trivialise, toward open-ended problems that
probe context gathering, trade-off reasoning, and collaboration. The Chan Zuckerberg Initiative's
public data science interview guide (February 2025) and Square's engineering blog on revamping
data science interviews (May 2025) both describe loops with a SQL or coding screen, a
statistics or machine learning discussion, a case, and behavioural rounds, with the emphasis on
how the candidate reasons rather than on recall.

That shift helps candidates who understand their work and hurts candidates who memorised
question banks. It also raises the value of a specific skill that is rarely taught: narrating your
reasoning while producing an artifact, without letting the narration consume the time needed to
produce it.

### 1.1 The four question families

Almost every technical question in a biomedical data loop falls into one of four families.

| Family | Typical question | What it infers |
|---|---|---|
| SQL and data manipulation | "Compute the 30-day readmission rate by service line" | Can you define grain, handle duplicates, and write correct logic |
| Statistics and study design | "How would you analyse this experiment?" | Do you know what the estimate means and what would invalidate it |
| Machine learning | "How would you validate this model?" | Do you know leakage, calibration, and the difference between discrimination and usefulness |
| Coding and algorithms | "Implement a function to group encounters" | Can you write working code and reason about edge cases |

The families overlap. A SQL question about readmissions is also a study-design question about the
index event and the denominator. An ML question about validation is also a statistics question
about dependence. The interviewer usually cares more about the overlap than about the family.

### 1.2 Why candidates fail

Three failure patterns account for most rejections, and none of them is a knowledge gap.

**Silent production.** The candidate writes correct code for 20 minutes without speaking, then
explains it in the last five. The interviewer cannot score the reasoning and marks communication
low, even when the code is right.

**Narrated production.** The opposite: the candidate explains every keystroke and never finishes.
The artifact is incomplete, which is scored as failure regardless of the quality of the
explanation.

**Unverified production.** The candidate finishes and says "done" without testing the result. The
interviewer asks for a row count or an edge case, and the answer is wrong. Verification is the
cheapest signal in the interview and the most commonly skipped.

A fourth pattern is becoming common as assistants enter the process: **unexamined generation.**
The candidate produces plausible code quickly and cannot explain why it is correct. It looks like
fluency and is scored as its absence.

### 1.3 What this topic produces

A preparation protocol that covers all four families, an execution protocol for the interview
itself, a worked example with two diagnostics and a revised plan, and a Python implementation of
window functions that you can run to check your mental model of SQL semantics. The aim is not to
teach SQL or statistics from scratch; the methods modules do that. The aim is to make your
existing knowledge legible under time pressure.

### 1.4 What this topic does not cover

It does not cover clinical case interviews (Topic 06), system design (Topic 07), or behavioural
interviews (Topic 08). It does not cover the statistical methods in depth; Modules 04 and 05 do
that. It does not provide a question bank, because question banks age quickly and because the
2025 shift described above makes memorised answers less useful than reasoning aloud.

It also does not cover take-home exercises in detail; Topic 04 covers the decision to accept and
the scope negotiation, and the portfolio topic covers the artifact quality that a take-home is
judged against.

### 1.5 What a strong answer sounds like

A strong technical answer has a recognisable shape, regardless of family:

> "The decision is whether to enrol this patient in the transitional care programme, so the unit
> is the patient at discharge, not the encounter. I will build one row per index discharge,
> exclude planned readmissions, and require 30 days of enrolment. The denominator is index
> discharges meeting those criteria. I will write it as three CTEs, check the row count against
> the admission count, and then compute the rate by service line. The limitation is that the
> planned-readmission flag is incomplete for one payer, so the rate will be slightly overstated
> for that group."

It is about 40 seconds, and it contains the decision, the grain, the exclusions, the denominator,
the plan, the check, and the limitation. Everything after that is execution. A candidate who can
produce this paragraph before writing has already passed the part of the interview that most
candidates fail.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Classify a technical question into one of four families and name what it tests | Exercise 1; self-check 1 |
| 2 | Apply the clarify, solve, verify protocol within a fixed time box | Exercise 2; self-check 2 |
| 3 | Write a correct SQL query involving a join, a window function, and a denominator | Exercise 3; self-check 3 |
| 4 | Explain calibration and decision curves to a non-statistician in 90 seconds | Exercise 4; self-check 4 |
| 5 | Diagnose leakage in a described validation design | Exercise 5; self-check 5 |
| 6 | Implement a window function in Python and verify it against a hand calculation | Exercise 6; self-check 6 |
| 7 | Recover from a wrong answer without losing the stage | Exercise 7; self-check 7 |
| 8 | Prepare a six-week drill plan with timed reps and a feedback loop | Exercise 8; interview question 9 |

Outcomes 2 and 7 are the ones that distinguish candidates with equal knowledge. Outcome 3 is the
most frequently tested and the easiest to verify.

## 3. The core mental model

A technical interview is a loop with three phases, and each phase has a different failure mode.

```text
  CLARIFY (2-4 min)          SOLVE (15-35 min)           VERIFY (3-6 min)
  ------------------         -----------------           ----------------
  What is the grain?         Write the logic in          Row count and
  What is the unit?          small steps.                expected magnitude.
  What is the denominator?   State assumptions           Edge cases: NULLs,
  What does "done" mean?     as you go.                  duplicates, ties,
  What are the constraints?  Keep the artifact           boundaries.
                             runnable.                   Explain the result.
       |                          |                            |
       v                          v                            v
  Fails by asking             Fails by silence            Fails by saying
  too much or too little      or by over-narration        "done" without a test
```

The phases are not equal. Clarify is short and high-leverage; solve is where most of the time
goes; verify is short and is the most common omission. A candidate who spends four minutes
clarifying, twenty solving, and four verifying outperforms a candidate who spends twenty-eight
minutes solving.

### 3.1 The signal-per-minute principle

Every action in a technical interview produces signal at a different rate.

| Action | Signal per minute | Notes |
|---|---|---|
| Restating the question and the grain | High | Prevents the most common error |
| Writing a query skeleton with a comment per step | High | Lets the interviewer follow without narration |
| Running a small test query | High | Catches grain errors early |
| Explaining a concept you were not asked about | Low | Consumes time, adds little |
| Silent debugging for more than two minutes | Negative | Interviewer cannot distinguish stuck from thinking |
| Adding unrequested optimisation | Low | Often introduces bugs |
| Verifying an edge case | High | Demonstrates judgment |

The practical rule: narrate decisions, not keystrokes. "I am joining encounters to admissions on
patient ID and admission date because the encounter table has multiple rows per admission" is a
decision. "Now I am typing SELECT" is not.

### 3.2 What each family is really testing

| Family | Surface question | Underlying construct |
|---|---|---|
| SQL | Can you write the query | Can you define the unit of analysis and the denominator |
| Statistics | Do you know the test | Do you know what the estimate means and when it is invalid |
| Machine learning | Do you know the algorithm | Do you know how it fails and how you would detect that |
| Coding | Can you implement it | Can you decompose a problem and handle edges |

This is why a candidate can write a perfect query and still fail: the query computed a rate per
encounter when the decision needed a rate per patient, and the candidate never asked. The
interviewer's note is usually "did not clarify the grain," not "SQL weak."

### 3.3 The recovery model

Mistakes are expected. The interviewer is testing whether you can detect and correct one.

```text
  Detect  ->  Acknowledge  ->  Localise  ->  Correct  ->  Confirm
```

- **Detect:** run a check, or notice an inconsistency in the output.
- **Acknowledge:** "That count is wrong; it is counting encounters, not patients."
- **Localise:** "The join fan-out is duplicating admissions."
- **Correct:** change the join or pre-aggregate.
- **Confirm:** rerun the check and state the new number.

A candidate who completes this sequence in under two minutes often scores higher than one who
never made the error, because the recovery demonstrates exactly the skill the employer needs.

### 3.4 The knowledge you actually need

The interview samples a small subset of a large field. The subset is predictable enough to
prepare:

- **SQL:** joins and fan-out, GROUP BY and HAVING, window functions (ROW_NUMBER, RANK, LAG,
  running sums), CTEs, NULL semantics, date arithmetic, deduplication.
- **Statistics:** confidence intervals, p-values and what they do not mean, power and minimum
  detectable effect, multiple comparisons, confounding, missing data, survival basics.
- **Machine learning:** leakage, temporal and grouped splits, discrimination versus calibration,
  decision curves, class imbalance, regularization, tree ensembles, evaluation under prevalence
  shift.
- **Coding:** data structures (dict, list, set, heap), string and array manipulation, complexity,
  testing, and reading someone else's code.

You do not need to know all of deep learning, all of causal inference, or every SQL dialect. You
need to be fluent in the subset and honest about the boundary.

> **Note:** The boundary is tested more often than the depth. "I have not used that in
> production; here is how I would approach it" scores better than a confident wrong answer,
> because the interviewer is inferring what you will do when you are out of your depth on the
> job.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Grain | The entity represented by one row of a table | Left implicit, causing fan-out | Statement |
| Fan-out | Row multiplication from a one-to-many join | Discovered after the numbers look wrong | Row count ratio |
| CTE | Common table expression; a named subquery | Used for everything, including trivial filters | Query block |
| Window function | A function computed over a partition without collapsing rows | Confused with GROUP BY | SQL construct |
| Partition | The subset of rows a window function operates on | Confused with a table partition | Clause |
| Frame | The rows within a partition used by a window function | Left to the default without knowing it | Clause |
| Deduplication | Reducing multiple rows per entity to one | Done with DISTINCT instead of a rule | Operation |
| NULL | An unknown or absent value | Treated as zero or empty string | SQL semantics |
| Three-valued logic | SQL's TRUE, FALSE, UNKNOWN evaluation | Assumed to be two-valued | Semantics |
| Index event | The event that starts follow-up | Confused with the first observed record | Definition |
| Denominator | The population at risk | Chosen to flatter the rate | Definition |
| Attributable risk | The difference in outcome rates between exposed and unexposed | Confused with relative risk | Proportion |
| Confidence interval | A range of plausible values for an estimate | Interpreted as a probability the truth is inside | Interval |
| p-value | The probability of data at least as extreme under the null | Interpreted as the probability the null is true | Probability |
| Power | The probability of detecting an effect of a given size | Computed after the study | Proportion |
| Minimum detectable effect | The smallest effect a design can detect with stated power | Confused with the observed effect | Effect size |
| Multiple comparisons | Testing many hypotheses and inflating false positives | Ignored in subgroup analyses | Count |
| Leakage | Information in training unavailable at prediction time | Reduced to "I used a pipeline" | Failure mode |
| Temporal split | Splitting by time rather than at random | Confused with a random split | Design |
| Grouped split | Splitting so that a group (patient, site) appears in one set only | Omitted when patients repeat | Design |
| Calibration | Agreement between predicted probabilities and observed frequencies | Confused with discrimination | Slope, intercept |
| AUROC | Rank discrimination across thresholds | Reported as the only metric | 0–1 |
| AUPRC | Precision-recall area, sensitive to prevalence | Compared across different prevalences | 0–1 |
| Brier score | Mean squared error of predicted probabilities | Confused with log loss | Score |
| Decision curve | Net benefit across threshold probabilities | Omitted because AUROC is easier | Net benefit |
| Regularisation | Penalising model complexity (L1, L2) | Treated as a universal fix for overfitting | Penalty |
| Bias-variance trade-off | The tension between underfitting and overfitting | Used as a slogan | Concept |
| Boosting | Sequential ensemble that corrects prior errors | Confused with bagging | Method |
| Hash map | Key-value structure with average constant lookup | Implemented as a list scan | Structure |
| Complexity | Growth of runtime or memory with input size | Quoted without a reason | Big-O |


## 5. The mechanics, step by step

The procedure builds a six-week drill programme and a protocol for the interview itself. It
assumes 6 to 8 hours per week, which is realistic alongside a full-time job, and it assumes you
already know the methods at the level taught in Modules 04 to 06.

### Step 1. Build the drill inventory before drilling

**Input:** three postings from your target family and the four-family table in Section 1.1.
**Output:** a list of 20 to 30 specific question types, tagged by family and by confidence.
**Rationale:** drilling without an inventory produces practice in the areas you already like.
The inventory forces coverage and makes the weak family visible.

For each posting, extract the technical requirements and convert them into question types:

| Posting phrase | Question type | Family |
|---|---|---|
| "Advanced SQL" | Deduplicate encounters, compute a rate by service line | SQL |
| "Experience with claims data" | Define a 30-day readmission cohort with continuous enrolment | SQL and design |
| "Strong statistical foundations" | Explain a confidence interval to a clinician | Statistics |
| "Model validation" | Design a temporal validation and name the leakage risks | ML |
| "Python and pandas" | Group and reshape a visit table | Coding |

Score each type 1 to 5 for confidence. The drill allocation is proportional to (6 minus
confidence), not to the number of postings that mention it.

Score twice: once before any drilling and once after the first timed rep. The first score measures
familiarity; the second measures retrieval under pressure. The gap between them is usually larger
for SQL than for statistics, because SQL is recognised more easily than it is recalled.

### Step 2. Practise in the environment the employer will use

**Input:** the inventory.
**Output:** a sandbox with a realistic schema and a shared-editor setup.
**Rationale:** fluency in your own IDE does not transfer to a browser-based editor with no
autocomplete. The first time you meet the environment should not be the interview.

Set up:

1. A local PostgreSQL instance or a free cloud warehouse with a small synthetic schema:
   patients, encounters, diagnoses, claims, labs.
2. A browser-based notebook or a plain text editor with no extensions, to simulate the shared
   environment.
3. A timer visible on screen.
4. A one-page schema card with table names, keys, and row counts, which is what an interviewer
   would give you.

### Step 3. Learn the grain rule before anything else

**Input:** the sandbox schema.
**Output:** a written grain statement for each table.
**Rationale:** most SQL interview errors are grain errors. The candidate writes a correct query
against the wrong unit, and the number is plausible enough to pass unnoticed until the interviewer
asks.

For each table, write one sentence: "One row per [entity] at [time]." Examples:

- `encounters`: one row per encounter, where an encounter may have multiple rows per admission.
- `admissions`: one row per admission, with admission and discharge timestamps.
- `claims`: one row per claim line, not per claim.
- `labs`: one row per lab result, with a collection time and a result time.

Then write the join cardinality for every pair you plan to use: one-to-one, one-to-many, or
many-to-many. A many-to-many join without a deduplication rule is the most common cause of
inflated denominators.

### Step 4. Drill seven SQL patterns until they are automatic

**Input:** the sandbox.
**Output:** seven query templates you can write from memory and adapt.
**Rationale:** interview SQL reuses a small set of patterns. Automating them frees attention for
the grain and the definition, which is where the interview is actually decided.

| Pattern | Template | Common trap |
|---|---|---|
| Deduplicate to one row per entity | `ROW_NUMBER() OVER (PARTITION BY id ORDER BY time DESC)` then filter to 1 | Forgetting the filter, or ordering by the wrong timestamp |
| Latest value per patient | `ROW_NUMBER` or `DISTINCT ON` | Ties on timestamp |
| First event after an index date | Join to a subquery with `MIN(event_time) WHERE event_time >= index` | Including events before the index |
| Running total | `SUM(x) OVER (PARTITION BY id ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` | Default frame including peers |
| Previous value | `LAG(x) OVER (PARTITION BY id ORDER BY time)` | NULL for the first row |
| Rate with correct denominator | `COUNT(DISTINCT patient_id) FILTER (WHERE event)::numeric / COUNT(DISTINCT patient_id)` | Integer division; counting events instead of patients |
| Cohort retention | Index date per patient, then count activity in each period | Survivorship bias from patients lost to follow-up |

Write each template by hand three times, then from memory twice, then under a 10-minute timer.
The timer matters: the interview tests retrieval under pressure, not recognition.

Five traps recur in almost every screen:

| Trap | Symptom | Fix |
|---|---|---|
| Integer division | A rate is exactly 0 or 1 | Cast to numeric before dividing |
| NULL comparison | Filters silently drop rows | Use `IS NULL`, `COALESCE`, or an explicit anti-join |
| Fan-out | The denominator exceeds the population | Pre-aggregate or deduplicate before joining |
| Date boundaries | Off-by-one at day 30 | State whether the window is inclusive and test both ends |
| Time zones | Events land on the wrong day | Convert to a single zone before truncating |

### Step 5. Drill the 90-second explanation

**Input:** a list of ten statistical concepts.
**Output:** a 90-second explanation of each, with one clinical analogy and one limitation.
**Rationale:** the statistics round usually tests whether you can explain, not whether you can
derive. A correct but jargon-heavy answer scores low because the interviewer is imagining you in
a meeting with a clinician.

The ten concepts: confidence interval, p-value, power, confounding, missing data, calibration,
AUROC versus AUPRC, decision curve, survival and censoring, and multiple comparisons.

Structure for each:

```text
What it is (one sentence).
What it is not (one sentence).
Why it matters for a decision (one sentence).
What would change my mind (one sentence).
```

Example for calibration: "Calibration is whether a model that says 20 percent actually corresponds
to about 20 events per 100 patients. It is not the same as ranking, which is what AUROC measures.
It matters because a threshold-based decision uses the probability, not the rank. If the
calibration slope were far from 1, I would recalibrate before using any threshold."

### Step 6. Drill validation design, not algorithms

**Input:** three model scenarios.
**Output:** a validation design and a leakage list for each.
**Rationale:** ML interviews in biomedical settings test whether you can stop a model from being
deployed, not whether you can recite the bias-variance decomposition.

For each scenario, write:

1. Time zero and the prediction moment.
2. The label definition and the observation window.
3. The split (temporal, grouped, external) and why.
4. The leakage risks: target leakage, temporal leakage, patient leakage, preprocessing leakage.
5. The metrics: discrimination, calibration, decision curve, subgroups.
6. The stop criterion: what result would make you recommend against deployment.

A candidate who can produce this in ten minutes is stronger than one who can derive the gradient
of logistic loss, because the employer is hiring for the first skill.

The leakage list is worth memorising, because it is asked in almost every biomedical ML interview:

| Leakage type | Example | Check |
|---|---|---|
| Target leakage | A feature recorded after the outcome, such as a discharge diagnosis used to predict readmission | Timestamp every feature against time zero |
| Temporal leakage | Random split across time, so the model sees the future | Split by time and inspect the feature availability |
| Patient leakage | The same patient in train and test | Group the split by patient identifier |
| Preprocessing leakage | Imputing or scaling before the split | Fit the transformer on the training fold only |
| Site leakage | The same hospital in train and test when the claim is transportability | Split by site, or report both |

### Step 7. Drill coding with tests

**Input:** five small problems drawn from data manipulation, not algorithm puzzles.
**Output:** working code with at least three tests per problem.
**Rationale:** the coding round in a data role usually involves grouping, joining, deduplication,
or string handling. Writing tests is the cheapest way to demonstrate verification.

Problems to practise: group and aggregate records; deduplicate by a key with a tie-break rule;
parse and normalise dates; compute a rolling window; implement a hash-map count.

For each: state the input and output types, write the function, write three tests (normal, empty,
edge), and run them. In the interview, say that you are writing a test before you write it. The
signal is the habit, not the test framework.

### Step 8. Rehearse the full protocol under time

**Input:** the drills.
**Output:** three recorded mock interviews, one per family.
**Rationale:** the component skills do not automatically compose. The mock is where the clarify,
solve, verify sequence becomes automatic.

Mock protocol: 45 minutes, a stranger as interviewer, a question neither of you has seen, a timer,
and a recording if permitted. Score yourself on five dimensions: clarified the grain, stated
assumptions, produced a working artifact, verified it, explained it in plain language.

### Step 9. Run a deliberate-error drill

**Input:** a solved problem.
**Output:** a list of three plausible errors and the check that catches each.
**Rationale:** recovery is a scored skill and is rarely practised. Deliberately introducing an
error and narrating the recovery builds the habit.

Examples: join without deduplication, integer division, NULL comparison, off-by-one in a date
window, training on the test period. For each, write the symptom and the check: "the row count
exceeds the patient count" catches fan-out; "the rate is exactly 0 or 1" catches integer division.

### Step 10. Prepare the environment and the rules

**Input:** the interview invitation.
**Output:** a logistics card with links, credentials, and the AI policy.
**Rationale:** environment failures consume the time box and are scored as performance.

Checklist: link tested, editor permissions checked, schema available, calculator allowed, notes
policy confirmed, AI assistance policy confirmed in writing, backup connection, and a clock.

The AI policy deserves a direct question. Some employers allow an assistant and evaluate how you
verify its output; others prohibit it. Using one when prohibited ends the process. Using one when
allowed, without verifying the output, demonstrates the opposite of the skill being tested.

### Step 11. Execute: clarify, solve, verify

**Input:** the question.
**Output:** a correct artifact and a clear explanation.
**Rationale:** the protocol is the difference between a candidate who knows and a candidate who
can be trusted.

Timing for a 45-minute question:

| Phase | Minutes | Actions |
|---|---:|---|
| Clarify | 3 to 5 | Restate the decision, the grain, the denominator, the time window, the definition of done |
| Plan | 2 | State the steps aloud, in order, before writing |
| Solve | 25 to 30 | Write in small steps, run intermediate checks, narrate decisions only |
| Verify | 4 to 6 | Row counts, edge cases, a sanity check against a known magnitude |
| Explain | 3 to 5 | Plain-language summary, one limitation, one thing you would do with more time |

If you finish early, do not add features. Verify more, or explain the trade-offs you did not take.
Finishing early is a chance to demonstrate judgment, not a chance to demonstrate extra
functionality that nobody asked for and that may introduce a bug.

### Step 12. Debrief and update the drill inventory

**Input:** the interview.
**Output:** an updated confidence score per question type and one new drill.
**Rationale:** each interview is a sample of what the market asks. The debrief converts it into
preparation for the next one.

Record the question, your approach, the error (if any), the interviewer's reaction, and the one
drill you will add. Update the confidence score. If a family produced two failures, reallocate the
next week's hours toward it.

### Step 13. Use the last five minutes deliberately

**Input:** an almost-finished stage.
**Output:** a summary that the interviewer can write down.
**Rationale:** interviewers score from notes taken during and immediately after the stage. A clear
closing summary gives them the sentence they will use in the debrief.

Close with four sentences: what you produced, the key assumption, the limitation, and what you
would do next. Then ask one question about how the team would use the result. Do not add new
analysis in the last five minutes; the risk of breaking a working artifact exceeds the value of
one more insight.

### Branch points and edge cases

**No computer science degree.** You do not need one for a data role, but you do need the
vocabulary of complexity and data structures. Learn enough to say why a hash map is faster than a
nested loop, and do not apologise for the background. The domain knowledge is the scarce input.

**A statistics-heavy role.** The SQL round may be shorter and the study-design discussion longer.
Prepare to defend an estimand, a confounding strategy, and a sensitivity analysis. The 90-second
explanation drill still applies, because the interviewer is testing communication as well as
knowledge.

**A SQL-only screen.** This is common at payers and health systems. The screen is usually a single
query with a business framing and a time limit of 20 to 30 minutes. Practise one query end to end,
including the row-count check, in 15 minutes.

**A whiteboard or shared-document interview.** You cannot run the code. Write pseudocode with
explicit column names and say what you would check. The verification phase becomes a verbal list
of checks rather than an execution.

**AI assistance is allowed.** The test shifts from production to review. Practise reading a
generated query and finding the fan-out, the missing deduplication, or the wrong date boundary.
Say what you are checking and why.

**The take-home is the technical stage.** Some employers replace the live screen with a take-home
and use the interview to discuss it. Prepare to defend every choice in the artifact, including the
ones you made quickly. The most common failure is being unable to explain why a particular join or
model was chosen, which reads as borrowed work.

**A take-home overlaps the technical screen.** Do not reuse the take-home verbatim; the
interviewer may have read it. Use the same schema and a different question, and mention the
connection if it is relevant.

**English is not your first language.** Slow down by about 15 percent and use shorter sentences.
The interviewer is scoring clarity, not speed, and a slight pause reads as thought.

**You need an accommodation.** Request extra time or a different format through the recruiter.
Untimed technical exercises are a common and reasonable adjustment.

### Decision table: when you are stuck

| Situation | Ask a clarifying question | Simplify and continue | Stop and reset |
|---|---|---|---|
| The question is ambiguous | Yes, immediately | Not yet | No |
| You do not know the function name | Describe the logic in words | Yes, using a workaround | No |
| The query returns implausible numbers | No | Check grain and counts | If two checks fail, reset the approach aloud |
| You have spent four minutes silent | No | Yes, narrate the current hypothesis | If still stuck after two more minutes, say so |
| You realise the approach is wrong | No | No | Yes, state the error and restart with a simpler design |

### Decision table: AI assistance in a technical screen

| Policy | Your behaviour | What is being tested |
|---|---|---|
| Prohibited | Do not use it; close the tab if it is open | Compliance and honesty |
| Allowed, unmonitored | Use it for boilerplate only; verify every output | Judgment about when to rely on it |
| Allowed and monitored | Use it openly, narrate what you ask and what you check | Review skill and verification |
| Unclear | Ask the recruiter in writing before the stage | Diligence |

## 6. Worked example

Marcus is 33, a biostatistician with five years at an academic medical centre. He has strong
statistics, moderate R, weak SQL, and no production Python. He is applying to a clinical data
scientist role at a payer. The loop has a 30-minute SQL screen, a 45-minute statistics and ML
discussion, and a 60-minute case with a technical component. He has 36 hours over six weeks.

### 6.1 The drill inventory

| Family | Question types | Confidence before (1–5) | Hours allocated |
|---|---|---:|---:|
| SQL | Deduplicate, rate with denominator, window functions, cohort | 2 | 14 |
| Statistics | CI, p-value, power, confounding, missing data | 4 | 6 |
| ML | Validation, leakage, calibration, class imbalance | 3 | 8 |
| Coding | Grouping, deduplication, date parsing | 2 | 4 |
| Mock interviews | Full protocol | — | 4 |
| **Total** | | | **36** |

The allocation is proportional to (6 minus confidence), with a floor for the mock interviews. SQL
and coding receive 18 of 36 hours despite being two of four families, because they are the
weakest and because they are the most frequently screened.

### 6.2 Diagnostic 1: SQL fails under time, and the cause is grain

In the first timed drill, Marcus answered 6 of 10 window-function questions correctly in 15
minutes. The errors were not syntax errors. Three of the four wrong answers had the right
function and the wrong partition:

| Question | His answer | Error | Correct approach |
|---|---|---|---|
| Latest lab per patient | `ROW_NUMBER() OVER (PARTITION BY lab_id ORDER BY collected_at DESC)` | Partitioned by lab, not patient | Partition by patient, order by collection time |
| 30-day readmission rate | Counted readmissions / count of encounters | Denominator is encounters, not index admissions | Denominator is index admissions meeting inclusion |
| First claim after diagnosis | Joined on `claim_date >= diagnosis_date` and took `MIN` | Included claims on the same day from a prior episode | Use a 30-day clean period or an explicit index rule |
| Running total of spend | `SUM(amount) OVER (ORDER BY date)` | No partition by patient | Partition by patient |

The pattern is consistent: he knew the functions and did not define the unit. The fix was not
more SQL practice; it was the grain rule from Step 3. He wrote grain statements for every table
and a cardinality note for every join, and then re-drilled.

After two weeks: 10 of 10 correct in 12 minutes, with the row-count check performed every time.

### 6.3 Diagnostic 2: the statistics explanation is technically correct and unusable

In a mock statistics round, Marcus was asked to explain a confidence interval to a clinician. His
answer ran four minutes and included the phrase "the interval is the set of values not rejected by
a two-sided test at the alpha level." The mock interviewer, a product manager, could not restate
it.

The problem is not knowledge. It is audience calibration. The interviewer is imagining a meeting
in which a clinician asks whether the result is real, and Marcus's answer would not help.

The rewrite, 80 seconds:

> "The study estimates a 12 percent reduction, but the data are consistent with anything from a 3
> percent reduction to a 20 percent reduction. The 3 percent end is small enough that the
> programme might not be worth the cost, so I would not call this definitive. What would change my
> mind is a larger sample or a second site with the same direction."

He drilled the ten concepts in the four-sentence structure, recorded himself, and had a
non-specialist restate each one. The pass criterion was the restatement, not his own satisfaction.

### 6.4 The revised plan

| Activity | Original hours | Revised hours | Reason |
|---|---:|---:|---|
| SQL drills | 14 | 12 | Grain rule fixed the errors faster than volume |
| Statistics explanations | 6 | 8 | Audience calibration was the gap, not knowledge |
| ML validation | 8 | 8 | Unchanged |
| Coding | 4 | 4 | Unchanged |
| Mock interviews | 4 | 4 | Unchanged |
| **Total** | **36** | **36** | |

The reallocation is small because the diagnosis changed the content of the SQL hours, not their
number. Two of the SQL hours were spent writing grain statements and cardinality notes rather than
writing queries.

### 6.5 The screen

The SQL screen asked for the 30-day readmission rate by service line for a calendar year, with
the payer's definition: an admission within 30 days of discharge, excluding planned readmissions,
with continuous enrolment for the 30 days after discharge.

His protocol:

1. **Clarify (4 minutes).** Confirmed the index event (discharge), the exclusion (planned
   readmission flag), the enrolment requirement, and the output grain (service line, one row
   each).
2. **Plan (2 minutes).** Stated four steps: build index admissions, join subsequent admissions
   within 30 days, exclude planned and unenrolled, aggregate by service line.
3. **Solve (20 minutes).** Wrote a CTE for index admissions, a `LEAD`/self-join for the next
   admission, a flag for the 30-day window, and an aggregate with `COUNT(DISTINCT patient_id)`.
4. **Verify (5 minutes).** Ran a count of index admissions and compared it with the encounter
   count; checked that the readmission count was lower than the index count; checked one service
   line by hand.
5. **Explain (4 minutes).** Stated the rate, the denominator, one limitation (planned readmissions
   flagged by a claims field that is incomplete for one payer), and what he would check with more
   time.

He passed. The interviewer's feedback, relayed by the recruiter, was "clear about the denominator
and checked his work."

### 6.6 The query skeleton he wrote

The screen allowed notes, and he wrote this skeleton before typing:

```sql
-- grain: one row per index admission (discharge)
WITH index_adm AS (
    SELECT patient_id, admission_id, discharge_date, service_line
    FROM admissions
    WHERE discharge_date BETWEEN '2025-01-01' AND '2025-12-31'
      AND planned_readmission_flag = 0
),
next_adm AS (
    SELECT i.patient_id, i.admission_id, i.service_line,
           MIN(a.admission_date) AS next_admission_date
    FROM index_adm i
    JOIN admissions a
      ON a.patient_id = i.patient_id
     AND a.admission_date > i.discharge_date
     AND a.admission_date <= i.discharge_date + INTERVAL '30 days'
    GROUP BY i.patient_id, i.admission_id, i.service_line
)
SELECT service_line,
       COUNT(DISTINCT i.patient_id) AS index_patients,
       COUNT(DISTINCT n.patient_id) AS readmitted_patients,
       COUNT(DISTINCT n.patient_id)::numeric
         / COUNT(DISTINCT i.patient_id) AS readmission_rate
FROM index_adm i
LEFT JOIN next_adm n USING (patient_id, admission_id)
GROUP BY service_line;
```

The skeleton is not the final query: it omits the continuous-enrolment join and the exclusion of
planned readmissions on the second admission. It is the structure he narrated in the plan phase,
and writing it first let him spend the solve time on the definitions rather than on the shape.

### 6.7 What he would do with ten hours instead of thirty-six

| Activity | Hours | Reason |
|---|---:|---|
| SQL grain and dedup drills | 5 | The highest-frequency failure mode |
| Statistics explanation drills | 2 | Cheap, and the existing knowledge is strong |
| Mock interview | 2 | Protocol integration |
| Logistics and review | 1 | Environment and notes |
| **Total** | **10** | |

With ten hours he would drop ML validation practice entirely and rely on the portfolio case study
to carry that discussion. The trade is defensible because the SQL screen is a gate and the ML
discussion is a differentiator, and a gate failure ends the process before the differentiator is
reached.

### 6.7b What the ML round tested that he had not prepared

The ML discussion began with "tell me about a model you built" and moved to "what would make you
stop trusting it." Marcus had prepared validation design but not the stop criterion, and his first
answer was a list of metrics. The interviewer pressed: "Which result would make you recommend
against deployment?"

He recovered with a specific answer: if the calibration slope on the most recent year fell below
0.8 or above 1.2, he would not deploy without recalibration, because the threshold the care team
uses assumes the probability is roughly right. That answer was stronger than the metrics list,
because it connected a numerical condition to an operational decision.

The lesson is that ML rounds in biomedical settings are usually about the boundary between usable
and unusable, not about the model. Prepare one stop criterion per artifact you plan to discuss.

### 6.8 Vary one assumption

Change one input: the employer allows an AI assistant during the screen and says it will evaluate
how candidates use it.

The knowledge does not change. The task changes from production to review, and the practice plan
changes in three ways:

1. **Six of the twelve SQL hours move to review drills.** He practises reading generated queries
   and finding the fan-out, the missing deduplication, the wrong date boundary, and the integer
   division. Each drill is 10 minutes: generate, review, correct, verify.
2. **The narration changes.** He says what he is asking the assistant for and what he will check
   before accepting the output. "I will ask for a deduplication query; I will check that the
   partition is by patient and that the tie-break is deterministic."
3. **The verification phase grows from 5 to 8 minutes**, because generated code can be plausible
   and wrong in ways that are harder to spot than a hand-written error.

The general lesson: the tool changes the skill that is scarce. When generation is free, review and
verification become the differentiator, and a candidate who can explain why an output is wrong is
more valuable than one who can type it faster.

## 7. Code, query, or calculation

The script implements window-function semantics in pure Python so that the partition, order, and
frame rules can be executed and inspected. It is the drill from Step 3 in executable form: if you
can predict the output before running it, you understand the SQL.

Assumptions:

- The data are six synthetic claims for two patients, with a deliberate tie on date.
- `ROW_NUMBER` is computed in descending date order, because the common use is "latest row per
  patient." `RANK`, `LAG`, and the running sum are computed in ascending order, which is the
  chronological reading.
- Ties are broken by amount, which is an arbitrary but deterministic choice. SQL does not
  guarantee a tie-break unless the `ORDER BY` includes one.

```python
"""Window-function semantics in pure Python, standard library only.

Run: python window_functions.py
Computes ROW_NUMBER, RANK, LAG, and a running SUM over partitions of a small
claims table. The point is to make the SQL semantics executable so that the
partition and frame rules can be checked.
"""

from collections import defaultdict

CLAIMS = [
    {"patient": "P1", "claim_date": "2026-01-04", "amount": 120.0},
    {"patient": "P1", "claim_date": "2026-02-11", "amount": 80.0},
    {"patient": "P1", "claim_date": "2026-02-11", "amount": 45.0},
    {"patient": "P1", "claim_date": "2026-03-02", "amount": 300.0},
    {"patient": "P2", "claim_date": "2026-01-19", "amount": 500.0},
    {"patient": "P2", "claim_date": "2026-04-07", "amount": 75.0},
]


def window(rows, partition_key, order_key, func, reverse=False):
    """Apply func to each partition, ordered by order_key.

    func receives (position, row, previous_row, running_total) and returns a
    dict of computed columns.
    """
    partitions = defaultdict(list)
    for row in rows:
        partitions[row[partition_key]].append(row)

    out = []
    for key in sorted(partitions):
        ordered = sorted(partitions[key], key=lambda r: (r[order_key], r["amount"]), reverse=reverse)
        running = 0.0
        previous = None
        for position, row in enumerate(ordered, start=1):
            running += row["amount"]
            computed = func(position, row, previous, running)
            out.append({**row, **computed})
            previous = row
    return out


def row_number(position, row, previous, running):
    return {"row_number": position}


def rank_with_ties(position, row, previous, running):
    # RANK leaves gaps after ties; DENSE_RANK does not. Ties here are on date.
    if previous is not None and previous["claim_date"] == row["claim_date"]:
        rank = previous["_rank"]
    else:
        rank = position
    row["_rank"] = rank
    return {"rank": rank}


def lag_amount(position, row, previous, running):
    return {"prev_amount": None if previous is None else previous["amount"]}


def running_sum(position, row, previous, running):
    return {"running_amount": round(running, 2)}


def main():
    rows = window(CLAIMS, "patient", "claim_date", row_number, reverse=True)
    rows = window(rows, "patient", "claim_date", rank_with_ties)
    rows = window(rows, "patient", "claim_date", lag_amount)
    rows = window(rows, "patient", "claim_date", running_sum)

    print("CLAIMS WITH WINDOW COLUMNS")
    header = f"  {'patient':8s} {'date':12s} {'amount':>7s} {'rn_desc':>8s} {'rank':>4s} {'prev':>7s} {'cum':>8s}"
    print(header)
    for r in rows:
        prev = "" if r["prev_amount"] is None else f"{r['prev_amount']:.1f}"
        print(f"  {r['patient']:8s} {r['claim_date']:12s} {r['amount']:7.1f} "
              f"{r['row_number']:8d} {r['rank']:4d} {prev:>7s} {r['running_amount']:8.1f}")

    latest = [r for r in rows if r["row_number"] == 1]
    print()
    print("LATEST CLAIM PER PATIENT (row_number = 1)")
    for r in latest:
        print(f"  {r['patient']}: {r['claim_date']} amount {r['amount']:.1f}")

    total = sum(r["amount"] for r in latest)
    print()
    print(f"  total of latest claims: {total:.1f}")
    print(f"  total of all claims:    {sum(r['amount'] for r in CLAIMS):.1f}")
    print("  the difference is the fan-out you get if you forget the deduplication filter.")


if __name__ == "__main__":
    main()
```

Expected output, confirmed by running the script:

```text
CLAIMS WITH WINDOW COLUMNS
  patient  date          amount  rn_desc rank    prev      cum
  P1       2026-01-04     120.0        4    1            120.0
  P1       2026-02-11      45.0        3    2   120.0    165.0
  P1       2026-02-11      80.0        2    2    45.0    245.0
  P1       2026-03-02     300.0        1    4    80.0    545.0
  P2       2026-01-19     500.0        2    1            500.0
  P2       2026-04-07      75.0        1    2   500.0    575.0

LATEST CLAIM PER PATIENT (row_number = 1)
  P1: 2026-03-02 amount 300.0
  P2: 2026-04-07 amount 75.0

  total of latest claims: 375.0
  total of all claims:    1120.0
  the difference is the fan-out you get if you forget the deduplication filter.
```

Reading the output:

1. **Partition by patient, order by date.** `rn_desc` resets to 1 for each patient. This is the
   partition rule: a window function never crosses a partition boundary. If you partition by
   claim instead of patient, every row gets 1 and the deduplication does nothing.
2. **Ties are ambiguous.** The two rows on 2026-02-11 receive ranks 2 and 2, and `rn_desc` 3 and
   2 depending on the amount tie-break. In SQL, `ROW_NUMBER` on a tie is nondeterministic unless
   the `ORDER BY` includes a unique tie-break. In an interview, say this out loud; it is a common
   probe.
3. **RANK versus ROW_NUMBER.** `RANK` gives 1, 2, 2, 4 after a tie; `ROW_NUMBER` gives 1, 2, 3, 4.
   The choice matters when the question says "the latest row" (use `ROW_NUMBER`) versus "all rows
   tied for latest" (use `RANK`).
4. **LAG is scoped to the partition.** `prev_amount` is empty for the first row of each patient
   and never borrows from the other patient. This is the check that catches a missing
   `PARTITION BY`.
5. **The running sum uses the default frame.** It accumulates within each patient in date order.
   In SQL, the default frame with `ORDER BY` is `RANGE UNBOUNDED PRECEDING TO CURRENT ROW`, which
   includes peers; with ties, a running sum can therefore include both rows at the same date.
   Add an explicit `ROWS BETWEEN ...` clause when you need row-by-row accumulation.

The final comparison is the point of the exercise. Summing the latest claim per patient gives
375.0; summing all claims gives 1,120.0. The second number is what a dashboard shows when the
query joins claims to patients and forgets the deduplication. The interview question is usually a
version of "why is this total too high," and the answer is fan-out.

To use the script as a drill, cover the expected output, predict each column, then run it. Repeat
until the predictions are correct, then translate the same logic into SQL in your sandbox and
compare the row counts.

## 8. Data, standards, and regulatory context

A technical interview is not a regulated activity, but the content of the answers is. A query that
computes a clinical rate, a model that predicts an outcome, and a statistic quoted to a clinician
are all governed by definitions that come from standards bodies and payers. Getting the syntax
right and the definition wrong is the most common technical failure in biomedical interviews.

### 8.1 Definitions that are standards, not choices

| Definition | Source | Why it matters in an interview |
|---|---|---|
| 30-day readmission | Payer and regulator specifications, such as the CMS Hospital Readmissions Reduction Program measure definitions | The index event, planned-readmission exclusions, and enrolment rules are specified; inventing your own definition changes the denominator |
| Mortality | Usually in-hospital, 30-day, or 90-day, and each is different | "Mortality" without a window is not a metric |
| Length of stay | Admission to discharge, with observation status handled explicitly | Observation encounters are a common trap |
| Lab result | LOINC code plus specimen and unit | A lab name is not a concept |
| Diagnosis | ICD-10-CM code plus position (primary or secondary) | Primary-diagnosis cohorts differ from any-position cohorts |
| Procedure | ICD-10-PCS or CPT, depending on the setting | Inpatient and outpatient coding systems differ |
| Drug exposure | RxNorm or NDC, with days supply and fills | Adherence inferred from fills is an assumption, not a measurement |

The interview version of this is a clarifying question: "Which readmission definition are we
using, and does it exclude planned readmissions?" A candidate who asks that is signalling that
they have worked with the measure, not just with SQL.

### 8.2 SQL dialects and what transfers

| Dialect | Common in | Differences that matter |
|---|---|---|
| PostgreSQL | Startups, analytics teams, open-source stacks | `DISTINCT ON`, strict typing, `FILTER` |
| BigQuery (GoogleSQL) | Google Cloud analytics | `QUALIFY`, `SAFE_CAST`, `STRUCT` and `ARRAY` |
| Snowflake | Health systems and payers with cloud warehouses | `QUALIFY`, `FLATTEN`, variant types |
| T-SQL (SQL Server) | Many hospitals and Epic-adjacent environments | `TOP`, `ISNULL`, different date functions |
| Oracle SQL | Legacy payer and hospital systems | `ROWNUM`, `NVL`, older syntax |

Core SQL transfers: joins, GROUP BY, window functions, CTEs, CASE, NULL handling. Dialect-specific
functions do not. In an interview, say which dialect you are writing and ask whether the employer
uses another. A candidate who writes PostgreSQL and says so is more credible than one who writes
vague pseudocode and hopes.

Three semantics are worth knowing because they are dialect-independent and frequently tested:

- **NULL is not a value.** `NULL = NULL` evaluates to UNKNOWN, not TRUE. Use `IS NULL`. Aggregates
  ignore NULLs, which changes averages when the denominator is implicit.
- **Integer division truncates.** `1/2` is 0 in many dialects. Cast to numeric before dividing.
- **Date arithmetic is not portable.** `DATE_ADD`, `DATEADD`, and `+ INTERVAL` differ. State the
  boundary rule in words before writing the function.

### 8.3 What is binding, and what is convention

Binding: the data use agreement that governs the dataset, the privacy rules that govern PHI, and
the measure specifications that a regulator or payer uses to compute a rate that affects payment.
If you are asked to compute a readmission rate for a payer, the specification is not a suggestion.

Convention: SQL style, naming conventions, whether to use CTEs or subqueries, and the choice of
window function for a tie. These are judged on readability, not correctness.

The distinction matters in an interview because candidates sometimes defend a stylistic choice as
if it were a correctness issue, and sometimes treat a specified measure as if it were a design
choice. The first wastes time; the second produces a wrong number.

### 8.4 The standards do not cover

They do not tell you which model to use, how to validate it, or what threshold to choose. They do
not define fairness, and they do not resolve the trade-off between sensitivity and alert burden.
Those are judgment calls, and the interview is designed to test them.

### 8.5 Working with data you do not own

Technical exercises often use a sample extract under a data use agreement. Three rules apply even
in a sandbox: do not copy the extract to a personal machine unless permitted, do not paste
record-level rows into a public tool or an AI assistant, and delete local copies when the exercise
ends. If the exercise uses synthetic data, say so when you describe your approach; it is a small
signal of governance awareness.

The same rules apply to your portfolio. A candidate who demonstrates a query on data they were
not authorised to share has demonstrated a governance failure, and biomedical employers weight
that more heavily than a missing optimisation.

### 8.6 When you have not used the tool

Interviews often include a tool you have not used: a warehouse, a framework, a coding system. The
credible answer has three parts: what you have used that is analogous, what transfers, and what you
would need to learn. "I have used PostgreSQL and Snowflake, not BigQuery; the joins and window
functions transfer, and I would need a week to learn the QUALIFY clause and the partitioning
model" is a strong answer. "I can learn anything" is not, because it gives the interviewer nothing
to assess.

## 9. Industry practice and case evidence

### 9.1 Documented: employers publish their loops

Square's developer blog published a description of its revised data science interview process in
May 2025, describing a structure of a recruiter screen, a technical screen, a case or take-home,
and an onsite loop of four to five sessions covering coding, analytics, and behavioural rounds.
The Chan Zuckerberg Initiative published a data science interview guide in February 2025 with a
similar structure and explicit guidance on what each round assesses.

These are employer-published descriptions, not independent audits. They are useful because they
are specific about format and because they show that large employers standardise loops to make
decisions comparable across candidates.

### 9.2 Documented: the shift away from memorisable syntax

The same 2025 sources describe a move away from syntax-recall questions toward open-ended
problems that test context gathering, trade-off reasoning, and collaboration, with employers
citing generative assistants as a reason that memorised syntax is less informative. The CZI guide
frames technical rounds around how a candidate works through a problem, and Square's post
describes evaluating collaboration and communication alongside technical skill.

This is a stated rationale from employers, not a measured trend across the market. It is
consistent with the structured-interview evidence in Topic 04, which favours job-related,
criterion-based assessment over trivia.

### 9.3 Documented: SQL is the most common first filter

Across the postings reviewed for Topic 01, SQL appears in the majority of clinical data science
and analytics roles, and it is the most common first technical screen at payers and health
systems. Window functions, cohort definitions, and rate calculations appear repeatedly in public
interview accounts. The frequency is an observation from a convenience sample of postings, not a
survey.

### 9.4 What employers say they assess

The public descriptions from Square and CZI name the constructs rather than the question types.
Square's 2025 post describes evaluating collaboration and communication alongside technical
skill, and CZI's guide describes rounds that assess problem structuring, technical depth, and
communication with cross-functional partners. Both describe standardised rubrics and interviewer
training, which is consistent with the structured-interview literature in Topic 04.

The practical translation: the interviewer is likely filling in a rubric with dimensions such as
"problem framing," "technical correctness," "verification," and "communication." A candidate who
is technically correct but silent loses points on two of the four. A candidate who is slightly
less optimal but explains the trade-off and checks the result can score higher.

### 9.5 Inference, labelled

- **Inference:** the SQL screen is usually a gate, and the statistics and ML discussion is usually
  a differentiator. A candidate who fails the gate never reaches the discussion. This is why the
  worked example allocates half the preparation to SQL despite a strong statistics background.
- **Inference:** interviewers weight verification more heavily than candidates expect, because
  verification is the behaviour that prevents silent errors in production.
- **Inference:** a candidate who states the dialect and the definition is read as experienced,
  because both require having worked in a real warehouse with real specifications.
- **Inference:** AI-allowed screens are shifting the scarce skill from writing to reviewing, and
  candidates who narrate their review will outperform those who narrate their typing.

> **Note:** No pass rate, question frequency, or rejection statistic is quoted here. Public
> accounts of interviews are self-selected and unverifiable, and employer blogs describe intent
> rather than outcomes.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong grain | The denominator exceeds the population | Query written before the unit was defined | Plausible but wrong number | State the grain and cardinality before writing |
| Silent production | More than two minutes without speaking | Focus on syntax, not reasoning | Communication scored low | Narrate decisions, not keystrokes |
| Over-narration | The artifact is incomplete at time | Explaining every step | Scored as failure to deliver | Time box the explanation to decisions |
| No verification | "Done" with no row count | Verification feels optional under time pressure | Errors pass unnoticed | Run one count and one edge case every time |
| Integer division | A rate is exactly 0 or 1 | Type semantics | Wrong metric | Cast to numeric |
| Fan-out | Totals are inflated | One-to-many join without deduplication | Wrong denominator and wrong cost | Pre-aggregate before joining |
| Jargon answer | The interviewer asks "what does that mean" | Speaking to a statistician when the panel is mixed | Inferred inability to communicate | 90-second explanation with a clinical analogy |
| Algorithm recital | You describe the model before the validation | Memorised ML answers | Inferred weak judgment | Lead with time zero, label, split, leakage |
| Ignoring the definition | You compute your own readmission rate | Assumed definitions are universal | The number does not match the payer's | Ask which specification applies |
| Defending a wrong answer | You repeat the approach with more confidence | Ego under observation | Low coachability score | Acknowledge, localise, correct |
| Not asking about the dialect | You write pseudocode and hope | Avoidance of commitment | Appears inexperienced | Name the dialect you are writing |
| Environment failure | The editor or link does not load | No pre-check | Time lost and composure damaged | Test 24 hours before and 10 minutes before |
| Using AI when prohibited | The tab is visible | Convenience or panic | Process ends | Confirm the policy in writing |
| Over-preparing one family | All practice is SQL | Comfort | A different family decides | Allocate by confidence and weight |
| Ignoring the clock | The mock runs 15 minutes over | No timed rehearsal | Incomplete artifact | Timer visible in every rep |
| Copying a memorised answer | The answer does not fit the question | Question banks reward recall | Interviewer hears a mismatch | Adapt the structure, not the content |
| No stop criterion | You list metrics but not a threshold for rejecting the model | Validation taught as reporting, not decision | Inferred weak judgment | Prepare one numerical stop rule per artifact |

Two red flags are worth naming as employer signals. A screen that asks only trivia and gives no
business context is usually a process that has not been updated, and the same team may evaluate
your work by trivia after hiring. A screen that asks you to solve a real problem with no schema
or definition and then penalises a different assumption is a process with no rubric, and your
performance will be judged by impression.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The four families cover most technical stages | It bounds preparation | Some roles test only one family, or add a domain quiz | Ask the recruiter what the stage assesses |
| Confidence scores predict failure | They drive allocation | Confidence is poorly calibrated | Compare with timed drill results |
| Timed drills transfer to the interview | They justify practice | The interview has a stranger and a different stakes level | Run a mock with a stranger |
| SQL is a gate in most processes | It justifies the allocation | Some roles screen on statistics first | Check three postings for order |
| Verification is scored | It justifies the time spent | Some interviewers only check the final answer | Observe reactions when you test |
| The environment resembles a shared editor | It justifies practising without autocomplete | Some employers allow your own IDE | Ask what environment is used |
| The measure definitions are specified | It justifies asking | Some exploratory questions have no specification | Ask whether the number is for payment or exploration |
| AI policy is knowable in advance | It prevents disqualification | Policies may be unstated | Ask in writing before the stage |
| Six weeks is enough to move one family by two points | It sets the plan horizon | Some gaps need months | Measure after three weeks and adjust |
| The interviewer shares your definition of done | It justifies the clarifying phase | Some interviewers want the simplest query, not the production one | Ask what done looks like in the first two minutes |

The main limit is that an interview measures performance on the day, not capability. Sleep,
nerves, and the interviewer's style all affect the sample. The protocol in Section 5 is designed to
reduce variance, not to eliminate it. A bad stage is not proof of a bad candidate, and the debrief
log is how you distinguish a fixable error from noise.

## 12. Exercises

**Exercise 1. Family classification (recall, 20 minutes).**
Deliverable: ten interview questions from public sources, each classified by family and by the
construct it tests.
Acceptance criteria: every question has a family and a construct; at least two questions are
classified as spanning two families, with a reason.

**Exercise 2. Protocol drill (application, 45 minutes).**
Deliverable: a recorded attempt at one SQL question using clarify, solve, verify, with timestamps
for each phase.
Acceptance criteria: clarify takes 3 to 5 minutes; the artifact runs; at least one check is run;
the total is within the time box.

**Exercise 3. SQL patterns (application, 2 hours).**
Deliverable: seven queries in a sandbox, one per pattern in Section 5, Step 4, each with a row-count
check.
Acceptance criteria: every query runs; each has a comment stating the grain; the rate query uses
`COUNT(DISTINCT ...)` with a numeric cast; the deduplication query has a deterministic tie-break.

**Exercise 4. Ninety-second explanations (application, 45 minutes).**
Deliverable: recordings of five concepts from Section 5, Step 5, each under 90 seconds.
Acceptance criteria: a non-specialist can restate each; no unexplained jargon; each includes one
limitation and one "what would change my mind."

**Exercise 5. Leakage audit (critique, 45 minutes).**
Deliverable: a written leakage audit of a model description from a paper or a portfolio project.
Acceptance criteria: all five leakage types from Section 5, Step 6 are considered; each finding
names the check that would catch it; at least one finding is actionable before deployment.

**Exercise 6. Window functions (application, 60 minutes).**
Deliverable: predictions for each column of the Section 7 output before running it, then a
comparison with the actual output, then the equivalent SQL.
Acceptance criteria: predictions are written down first; differences are explained; the SQL
produces the same `rn_desc` values; the tie-break is documented in both versions.

**Exercise 7. Deliberate error (application, 30 minutes).**
Deliverable: a recording of a solved problem in which you introduce one error, detect it, and
recover using the five-step model.
Acceptance criteria: detection is triggered by a check, not by the interviewer; the recovery is
under two minutes; the final answer is correct.

**Exercise 8. Six-week plan (design, 45 minutes).**
Deliverable: a plan with weekly hours, drill counts, mock dates, and a stopping criterion.
Acceptance criteria: hours total what you actually have; allocation follows (6 minus confidence);
at least two mocks are scheduled with a stranger; the plan names what will be dropped if a week
slips.

## 13. Self-check answers

**1. What does each family test?**
SQL tests whether you can define the unit and the denominator; statistics tests whether you know
what an estimate means and when it is invalid; ML tests whether you know how a model fails and how
you would detect it; coding tests decomposition and edge cases. The overlap is usually more
important than the family label.

**2. What are the three phases and their time budget?**
Clarify 3 to 5 minutes, solve 25 to 30, verify 4 to 6, explain 3 to 5. The budget assumes a
45-minute question. Clarify is short and high-leverage; verify is short and most often omitted.

**3. Write the readmission rate query in words.**
Build one row per index admission meeting inclusion, excluding planned readmissions; join to the
next admission within 30 days; require continuous enrolment; count distinct patients with a
readmission divided by distinct index patients, cast to numeric. The grain is one row per service
line in the output.

**4. Explain calibration in 90 seconds.**
Calibration is whether predicted probabilities match observed frequencies; a model that says 20
percent should see about 20 events per 100 similar patients. It is not ranking, which is AUROC. It
matters because a threshold decision uses the probability. If the slope were far from 1, I would
recalibrate before choosing a threshold.

**5. Name the five leakage types and one check each.**
Target leakage: timestamp every feature against time zero. Temporal: split by time. Patient:
group the split by patient. Preprocessing: fit transformers on the training fold only. Site: split
by site when the claim is transportability.

**6. Why does the Python output give 375 for latest claims and 1,120 for all claims?**
Because the latest-claim query keeps one row per patient, while the all-claims sum counts every
claim row. The difference is the fan-out that appears when a claim-level table is joined to a
patient-level table without deduplication.

**7. What do you do when you realise your approach is wrong?**
Stop, state the error and its cause in one sentence, propose a simpler approach, and restart within
the time box. Do not continue a wrong approach to avoid appearing wrong; the interviewer is
scoring recovery, and a silent wrong answer is worse than an acknowledged correction.

**8. How do you decide what to drop when a week slips?**
Drop practice volume, not mocks and not verification drills. The mock is the integration test; the
verification habit is the scored behaviour. Reducing from ten SQL reps to six is acceptable;
cancelling the mock is not.

If two weeks slip, cut the scope of the target family rather than the protocol: apply to fewer
roles in the same cycle, and keep the mock and the verification drills.

**Rubric for Exercise 3.** Full marks: seven queries, all runnable; grain comment on each;
deterministic tie-break; numeric cast; row-count check; at least one query tested against a hand
calculation.

**Rubric for Exercise 6.** Full marks: predictions written before running; every difference
explained by a partition, order, or frame rule; the SQL equivalent produces the same `rn_desc`
values; the tie-break is stated explicitly; the fan-out comparison is explained in one sentence.

**Rubric for Exercise 8.** Full marks: realistic hours; allocation justified by confidence; two
stranger mocks scheduled; a stopping criterion that is a test result rather than a date; a stated
drop rule.

## 14. Interview questions

**1. "Walk me through how you would compute 30-day readmissions."**
Strong: index event, exclusions, enrolment, denominator, grain, and a check. Weak: starts writing
a join.

**2. "What is the difference between ROW_NUMBER and RANK?"**
Strong: ties, gaps, and a use case for each. Weak: says both number rows.

**3. "Why might your total be higher than expected?"**
Strong: fan-out, duplicates, NULL handling, or a wrong denominator, with a check for each. Weak:
blames the data.

**4. "How would you validate this model?"**
Strong: time zero, label window, temporal or external split, leakage audit, calibration,
subgroups, stop criterion. Weak: "cross-validation."

**5. "What is the difference between AUROC and AUPRC, and when do you prefer one?"**
Strong: prevalence sensitivity, imbalanced outcomes, and the decision context. Weak: says AUPRC is
better in general.

**6. "Explain a confidence interval to a clinician."**
Strong: a range of plausible effects, the decision implication, and what would change their mind.
Weak: the definition of a confidence procedure.

**7. "How would you handle missing data?"**
Strong: mechanism, proportion, imputation strategy, sensitivity analysis, and the decision
consequence. Weak: "impute the mean."

**8. "You have 15 minutes left and the query is wrong. What do you do?"**
Strong: isolate with a count, simplify to a working version, state the limitation. Weak: keeps
debugging silently.

**9. "How do you use an AI assistant in your work?"**
Strong: names the tasks, the verification step, and the limits. Weak: claims never to use one, or
claims it does the work.

**10. "What would you do with more time?"**
Strong: names a specific check or sensitivity analysis that would change confidence. Weak: says
"more features."

## 15. Key terms

- **AUPRC** — area under the precision-recall curve.
- **AUROC** — area under the receiver operating characteristic curve.
- **Brier score** — mean squared error of predicted probabilities.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Cardinality** — the number of rows a join produces per key.
- **CTE** — common table expression; a named subquery.
- **Decision curve** — net benefit across threshold probabilities.
- **Deduplication** — reducing multiple rows per entity to one.
- **Denominator** — the population at risk.
- **Fan-out** — row multiplication from a one-to-many join.
- **Frame** — the rows within a partition used by a window function.
- **Grain** — the entity represented by one row.
- **Grouped split** — a split that keeps a group in one set only.
- **Index event** — the event that starts follow-up.
- **Leakage** — information available in training but not at prediction time.
- **Minimum detectable effect** — the smallest effect a design can detect with stated power.
- **NULL** — an unknown or absent value.
- **Partition** — the subset of rows a window function operates on.
- **Power** — the probability of detecting an effect of a given size.
- **p-value** — the probability of data at least as extreme under the null.
- **Regularisation** — penalising model complexity.
- **Temporal split** — splitting by time rather than at random.
- **Three-valued logic** — TRUE, FALSE, UNKNOWN in SQL.
- **Window function** — a function computed over a partition without collapsing rows.

## 16. Sources

All URLs accessed 26 October 2026.

1. PostgreSQL Global Development Group. *Documentation 18: Window Functions.*
   https://www.postgresql.org/docs/current/functions-window.html
2. PostgreSQL Global Development Group. *Documentation 18: Window Functions tutorial.*
   https://www.postgresql.org/docs/current/tutorial-window.html
3. scikit-learn developers. *Probability calibration*, user guide 1.16.
   https://scikit-learn.org/stable/modules/calibration.html
4. scikit-learn developers. *calibration_curve.*
   https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
5. Square Developer Blog. *Data Science Interview Best Practices*, May 2025.
   https://developer.squareup.com/blog/revamping-data-science-interviews/
6. Chan Zuckerberg Initiative. *Interview Guide: Data Science*, February 2025.
   https://chanzuckerberg.com/wp-content/uploads/sites/8/2025/02/Data-Science-Interview-Guide.pdf
7. Centers for Medicare & Medicaid Services. *Hospital Readmissions Reduction Program measure
   specifications.* https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
8. Observational Health Data Sciences and Informatics. *OMOP Common Data Model.*
   https://ohdsi.github.io/CommonDataModel/
9. HL7. *FHIR R4 specification.* https://hl7.org/fhir/R4/
10. LOINC. *Laboratory observation identifiers.* https://loinc.org/
11. ANSI. *SQL standard (ISO/IEC 9075).* https://www.iso.org/standard/76583.html
12. Google Cloud. *BigQuery GoogleSQL query syntax.* https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax
13. Snowflake. *SQL command reference.* https://docs.snowflake.com/en/sql-reference

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_interview-types.md`](04-interview-types.md)
- Next topic: [`06_clinical-case-interviews.md`](06_clinical-case-interviews.md)
- Related topic: [`02_portfolio.md`](02_portfolio.md)
- Survey file: [`../../course/20_career_portfolio_interviews.md`](../../course/20_career_portfolio_interviews.md)
- Labs: [`../../labs/`](../../labs/)
