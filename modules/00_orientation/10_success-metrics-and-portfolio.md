# Success Metrics and Portfolio

> Module 00 — Orientation and Industry Map
> Topic 10 of 10
> Estimated study time: 3 hours
> Prerequisites: 04, 09
> Version 2026.3

## 1. Why this topic exists

Every program claims to measure success. Most measure activity: modules completed, hours logged,
videos watched, certificates issued. Activity metrics are easy to collect and nearly useless for
answering the question that matters, which is whether a learner can do something they could not do
before.

The same problem appears at the individual level. A data scientist lists courses and tools on a
CV and wonders why interviews do not follow. The list describes exposure, not capability. What
employers assess is evidence: an artifact that shows a decision, a method, a result, and a
limitation.

This topic covers two related subjects. The first is how to measure a program honestly, with
denominators, baselines, and uncertainty. The second is how to build and present a portfolio that
demonstrates competence. The two connect because a portfolio is the individual-level version of a
program metric: it is the evidence that the learning produced something.

### 1.1 The vanity metric problem

Three metrics are common and misleading.

**Completion rate.** High completion can mean the content was easy, the assessment was weak, or
the learners were already skilled. It says nothing about capability.

**Satisfaction score.** Learners reliably prefer content that feels good, which is not the same as
content that changes practice. Satisfaction is a leading indicator at best.

**Certificate count.** Certificates attest to attendance or assessment, depending on the program.
A stack of certificates from short courses is weaker evidence than one substantial artifact.

None of these is worthless. Each is a proxy that becomes misleading when treated as the outcome.

### 1.2 What employers actually use

**Documented.** A 2025 CodePath survey of more than 200 engineering leaders, reported in December
2025, found that the top hiring signals outside interviews were side projects and portfolios
(38%), internship experience (35%), and public code portfolios such as GitHub (34%). Degree and
academic focus were cited by 23%, school prestige by 17%, and credentialing programs by 4%.

**Documented.** The CoderPad and CodinGame State of Tech Hiring 2025 report, covering more than
5,000 developers and recruiters, found that portfolio review was used by about 20.5% of employers
as a skills-assessment method, while only 21% used work-sample or take-home tests. Data scientist
roles were targeted by 17.5% of recruiters.

**Documented.** The Codio 2025 Data Science and Analytics Talent Survey of 111 senior executives
reported that 57% of employers said new hires lacked industry best practices and 56% said they
lacked current technical knowledge.

**Inference.** Portfolios are not the only signal, but they are among the strongest non-interview
signals and they are directly under your control. The gap employers report, best practices and
current technical knowledge, is exactly what an artifact-driven portfolio demonstrates.

### 1.3 Why measurement needs denominators

A placement rate of 78.6% sounds strong. If 60 graduates were surveyed and 42 responded, the rate
among respondents is 78.6% but the rate among all graduates is 55.0%. Both are true. Reporting
only the first is a denominator error, and it is common.

This topic treats denominators as a first-class concern. Every metric in the examples is reported
with its numerator, denominator, and uncertainty.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish activity, output, outcome, and impact metrics | Exercise 1; self-check 1 |
| 2 | Compute a rate with its denominator and a confidence interval | Exercise 2; self-check 4 |
| 3 | Design a portfolio with artifacts mapped to competencies | Exercise 3 |
| 4 | Score a portfolio against a weighted rubric and identify the largest gap | Exercise 4; self-check 6 |
| 5 | Write a portfolio narrative that a hiring manager can read in ten minutes | Exercise 5 |
| 6 | Evaluate a placement or success claim for denominator and response bias | Exercise 6 |
| 7 | Present an artifact orally in five minutes with limitations | Exercise 7 |
| 8 | Build a 90-day maintenance plan for the portfolio | Exercise 8 |

## 3. The core mental model

Measurement operates at three levels: the program, the artifact, and the person. Each level has
leading and lagging indicators. Confusing the levels produces metrics that look precise and mean
nothing.

### 3.1 Three levels

```text
   PROGRAM LEVEL
   Did the cohort acquire the capability?
   metrics: reproducibility rate, evidence quality, regulatory literacy,
            communication, equity reporting, placement
        |
        v
   ARTIFACT LEVEL
   Does the work demonstrate the capability?
   metrics: rubric score, external review, reproducibility, disclosure quality
        |
        v
   PERSON LEVEL
   Can the individual do the work unsupervised?
   metrics: capstone defense, interview performance, on-the-job output
```

The program level is an aggregate of artifacts. The person level is what the artifact predicts.
A program can report strong artifact scores and weak person-level outcomes if the artifacts were
heavily scaffolded.

### 3.2 Leading and lagging indicators

| Type | Definition | Examples | Use |
|---|---|---|---|
| Leading | Predicts a later outcome | Labs completed reproducibly, exercises submitted, reviews scheduled | Early correction |
| Lagging | The outcome itself | Placement, promotion, capstone pass, published artifact | Accountability |
| Activity | Effort expended | Hours, videos, logins | Operations only |
| Output | Thing produced | Artifacts, reports, models | Intermediate |
| Outcome | Change in capability or behavior | Independent project delivery | Program goal |
| Impact | Change in the world | Better decisions, reduced harm | Long-term, hard to attribute |

The program's stated metrics in the survey file are mostly outcome metrics: reproducibility,
evidence quality, regulatory literacy, communication, equity reporting, placement, and external
review. That is a good sign. Activity metrics are not listed as goals.

### 3.3 The evidence ladder for competence

```text
  Level 4  Independent delivery in a real setting, with external review
  Level 3  Artifact produced under supervision, with documented limitations
  Level 2  Exercise completed against acceptance criteria
  Level 1  Worked example reproduced
  Level 0  Content consumed
```

Most courses stop at Level 0 or 1. A portfolio should reach Level 3, and the capstone should reach
Level 4.

### 3.4 The portfolio as an argument

A portfolio is not a folder. It is an argument with evidence. The argument has four claims:

1. I can find the decision and define the population.
2. I can build and validate a method that fits the decision.
3. I can state the limitations and the conditions under which the work fails.
4. I can communicate the result to the people who must act on it.

Each artifact supports one or more claims. An artifact that supports none is a demonstration, not
evidence.

### 3.5 The metric contract

Every metric needs a contract with five fields.

| Field | Question | Example |
|---|---|---|
| Definition | What exactly is counted? | A lab is reproducible if it runs end to end in a clean environment |
| Numerator | What is the count? | Labs that ran successfully |
| Denominator | Out of what? | Labs attempted in the cohort |
| Source | Where does the data come from? | Automated run logs |
| Threshold | What counts as success? | 90% of attempted labs |

Without the contract, two people can compute the same metric and get different answers.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Metric | A defined measurement | Used for any number | Measure |
| KPI | Key performance indicator tied to a goal | Applied to every metric | Measure |
| Activity metric | Measure of effort | Treated as outcome | Measure |
| Output metric | Measure of things produced | Treated as impact | Measure |
| Outcome metric | Measure of capability or behavior change | Confused with impact | Measure |
| Impact metric | Measure of change in the world | Attributed without design | Measure |
| Leading indicator | Predicts a later outcome | Confused with a target | Measure |
| Lagging indicator | The outcome itself | Ignored in favor of activity | Measure |
| Baseline | Starting value before intervention | Omitted | Value with unit |
| Target | Intended value at a stated time | Set without a baseline | Value with unit |
| Numerator | Count of events meeting the definition | Confused with the rate | Count |
| Denominator | Population at risk or eligible | Omitted or inconsistent | Count |
| Rate | Numerator divided by denominator | Reported without denominator | Proportion |
| Response rate | Respondents divided by invited | Confused with completion rate | Proportion |
| Confidence interval | Range of plausible values | Treated as a probability | Interval |
| Wilson interval | Confidence interval for a proportion, better at extremes | Confused with normal approximation | Interval |
| Statistical significance | Compatibility with a null model | Confused with importance | Judgement |
| Effect size | Magnitude of a difference | Confused with p-value | Standardized or raw |
| Rubric | Criteria and performance levels | Confused with a checklist | Assessment tool |
| Inter-rater reliability | Agreement between raters | Assumed without measurement | Kappa or ICC |
| Artifact | Tangible output demonstrating work | Confused with a certificate | Deliverable |
| Portfolio | Curated set of artifacts with a narrative | Confused with a folder | Artifact set |
| Narrative | Short argument connecting artifacts to a role | Confused with a CV | Document |
| Competency | Ability to perform a defined task to a standard | Confused with knowledge | Capability |
| Placement rate | Share of graduates employed in the field | Denominator often omitted | Proportion |
| Response bias | Difference between respondents and non-respondents | Ignored | Bias |
| Selection bias | Non-random inclusion in a sample | Confused with response bias | Bias |
| Attrition | Loss of participants over time | Ignored in cohort metrics | Proportion |
| Capstone defense | Oral examination of individual contribution | Confused with team presentation | Assessment |
| Model card | Structured model documentation | Treated as a portfolio | Documentation |
| Datasheet | Structured dataset documentation | Treated as a license | Documentation |
| Public-safe | Containing no proprietary or identifiable data | Assumed to be anonymized | Property |
| Reproducibility | Ability to regenerate results from inputs | Confused with repeatability | Property |
| Provenance | Record of origin and transformation | Assumed to be the EHR | Metadata |
| Maintenance | Keeping artifacts current | Omitted from portfolio plans | Practice |
| Vanity metric | Metric that looks good but does not inform decisions | Treated as a KPI | Measure |
| Goodhart's law | When a measure becomes a target, it ceases to be a good measure | Quoted without application | Principle |
| Kirkpatrick model | Four-level training evaluation framework | Treated as a causal theory | Framework |

Three definitions deserve expansion.

**Denominator.** The denominator is the population that could have been counted. A placement rate
among respondents is not a placement rate among graduates. A reproducibility rate among submitted
labs is not a rate among attempted labs. Always state the denominator.

**Wilson interval.** The normal approximation for a proportion fails when the proportion is near 0
or 1, or when the sample is small. The Wilson score interval behaves better. For 24 out of 27, the
normal interval is roughly 0.83 to 0.95, while the Wilson interval is roughly 0.72 to 0.96. The
difference matters when the sample is small.

**Goodhart's law.** When a measure becomes a target, people optimize the measure rather than the
goal. A program that targets "90% of labs reproducible" may achieve it by making labs easier. The
defence is to pair the metric with a quality rubric and external review.

## 5. The mechanics, step by step

This section is a twelve-step procedure for building a portfolio and measuring it honestly. It is
written for a learner, and it applies equally to a program director designing metrics.

### Step 1: Define the target role and its evidence

**Input:** career direction.
**Output:** a list of five to eight capabilities the target role requires, each with an artifact
that would evidence it.

**Reasoning.** A portfolio is built backwards from the role. Without the role, the portfolio
becomes a collection of interests. With it, each artifact has a purpose.

**Branch point.** If the role is unfamiliar, interview three people in it and ask what they would
want to see from a candidate. Their answers are the requirements.

### Step 2: Inventory existing artifacts

**Input:** your work.
**Output:** a table of every artifact, its status (complete, partial, idea), and the capability it
evidences.

**Reasoning.** Most learners have more raw material than they think, and less finished material
than they need. The inventory distinguishes the two.

**Branch point.** If you have fewer than four complete artifacts, the next quarter is a production
quarter, not an application quarter.

### Step 3: Score each artifact against a rubric

**Input:** artifact inventory.
**Output:** a score from 0 to 4 on each of five criteria.

**Reasoning.** Scoring converts a vague sense of quality into a gap. The five criteria below are
the minimum for a credible artifact.

| Criterion | 0 | 2 | 4 |
|---|---|---|---|
| Decision and user | No decision named | Decision named but no user | Decision, user, moment, and action named |
| Data and provenance | Undescribed | Sources named | Sources, versions, access basis, and quality documented |
| Validation | None | Random split | Temporal or external validation with calibration |
| Limitation and failure | None | Generic caveats | Specific failure modes and conditions |
| Communication | Unreadable | Clear to a specialist | Clear to a non-specialist with a decision summary |

**Branch point.** An artifact scoring 2 or below on validation is not ready for a portfolio,
regardless of how impressive the model is.

### Step 4: Identify the largest gap

**Input:** scores.
**Output:** one gap that, if closed, would most improve the portfolio.

**Reasoning.** Improving a 1 to a 3 is worth more than improving a 3 to a 4. The largest gap is
usually in validation or limitations, not in modelling.

**Branch point.** If the largest gap is in the decision and user criterion, the problem is
framing, not technique. Return to topic 05.

### Step 5: Build the missing artifact

**Input:** gap.
**Output:** one complete artifact that closes the gap.

**Reasoning.** A portfolio improves by addition and by completion, not by editing. Building the
missing piece is usually faster than perfecting an existing one.

**Branch point.** If the artifact needs restricted data, build a public-safe version with open or
synthetic data and document the substitution.

### Step 6: Write the portfolio narrative

**Input:** completed artifacts.
**Output:** a one-page narrative with four claims and links to evidence.

**Reasoning.** Reviewers do not read everything. The narrative tells them what to look at and why.
It should take ten minutes to read and should make the argument in section 3.4 explicit.

**Branch point.** If the narrative cannot be written without jargon, the artifacts are not yet
understood well enough to present.

### Step 7: Run an external review

**Input:** portfolio and narrative.
**Output:** written feedback from at least two people, one technical and one clinical or
operational.

**Reasoning.** Self-assessment is unreliable. An external reviewer sees the gaps you have learned
to ignore. The clinical reviewer tests whether the work is intelligible outside the technical
frame.

**Branch point.** If both reviewers are from the same background, the review is incomplete. Find
the missing perspective.

### Step 8: Publish a public-safe version

**Input:** portfolio.
**Output:** a public version with no proprietary data, no PHI, and no confidential methods.

**Reasoning.** A portfolio that cannot be shown does not function as evidence. The public version
can use synthetic data, open datasets, or aggregated results.

**Branch point.** If the employer forbids publication, publish the method and the evaluation
design with fabricated or public data, and describe the restricted work in an interview.

### Step 9: Rehearse the five-minute presentation

**Input:** one artifact.
**Output:** a five-minute talk with decision, method, result, limitation, and next step.

**Reasoning.** The presentation is the person-level test. A reviewer learns more in five minutes
of questions than in an hour of reading.

**Branch point.** If you cannot state the limitation in one sentence, you do not yet own the work.

### Step 10: Track leading indicators weekly

**Input:** portfolio plan.
**Output:** a weekly log of artifacts completed, reviews received, and exercises passed.

**Reasoning.** Leading indicators allow correction before the deadline. A weekly log of artifacts
is more useful than a monthly count of hours.

**Branch point.** If the log shows no artifact completed in three weeks, the plan is too
ambitious or the scope is wrong. Adjust.

### Step 11: Measure outcomes with denominators

**Input:** program or personal outcomes.
**Output:** rates with numerator, denominator, source, and interval.

**Reasoning.** A rate without a denominator is not a measurement. The discipline applies to
personal metrics too: "applied to 12 roles, 4 interviews" is a rate; "got interviews" is not.

**Branch point.** If the denominator is unknown, report the count and say so. Do not compute a
rate from an unknown base.

### Step 12: Review and maintain quarterly

**Input:** portfolio and metrics.
**Output:** a quarterly review with updates, removals, and one new artifact.

**Reasoning.** A portfolio decays. Tools change, data expire, and the narrative drifts from the
target role. A quarterly review keeps it current with modest effort.

**Branch point.** If an artifact is more than two years old and no longer represents your work,
remove it. A weaker current artifact is better than a stronger obsolete one.

### 5.1 The program metrics

The survey file states seven program metrics. Each needs a definition, a denominator, and a
source.

| Metric | Target | Definition | Denominator | Source |
|---|---:|---|---|---|
| Reproducibility | 90% | Labs that run end to end in a clean environment | Labs attempted | Automated run logs |
| Evidence quality | 80% | Capstones with temporal or external validation | Capstones submitted | Rubric |
| Regulatory literacy | 80% | Pass a regulatory triage quiz | Learners taking the quiz | Quiz results |
| Communication | 80% | Explain a model to a clinician in 5 minutes | Learners assessed | Oral assessment |
| Equity | 100% | Capstones reporting at least one subgroup analysis | Capstones submitted | Rubric |
| Placement | Tracked | Employed in field at 6 and 12 months | Graduates invited to survey | Alumni survey |
| Industry review | 4/5 or higher | External reviewers rate capstone relevance | Reviewers | Review panel |

Two of these are unusual and worth noting. Equity at 100% is a reporting requirement, not a
performance requirement: every capstone must report a subgroup analysis, whatever the result.
Industry review at 4/5 is a relevance judgement, not a quality judgement.

### 5.2 Designing a metric that cannot be gamed

Four rules.

1. **Pair the metric with a quality rubric.** A completion metric paired with a rubric score is
   harder to game than completion alone.
2. **Use an external source where possible.** Automated logs are harder to manipulate than
   self-reports.
3. **Report the denominator.** A rate without a denominator invites inflation.
4. **Publish the failures.** A program that reports only successes is not measuring.

### 5.3 Portfolio structure

A portfolio has five parts. Each has a length limit.

| Part | Length | Purpose |
|---|---|---|
| Narrative | 1 page | The argument and the target role |
| Artifact summaries | 1 page each | Decision, method, result, limitation |
| Full artifacts | Linked | The evidence itself |
| Reflection | Half page | What you would do differently |
| Contact and availability | A few lines | How to reach you |

The reflection is not optional. It demonstrates the judgement that reviewers are looking for, and
it is the part most candidates omit.

### 5.4 Presenting an artifact in five minutes

| Time | Content | What the reviewer learns |
|---:|---|---|
| 0:00-0:30 | The decision and who makes it | Whether you understand the problem |
| 0:30-1:30 | The data and the population | Whether the inputs are credible |
| 1:30-3:00 | The method and validation | Whether the result is trustworthy |
| 3:00-4:00 | The result in decision terms | Whether you can translate |
| 4:00-4:30 | The limitation and failure mode | Whether you are honest |
| 4:30-5:00 | What you would do next | Whether you can prioritize |

A presentation that spends four minutes on the model and thirty seconds on the decision has the
proportions wrong.

### 5.5 Common portfolio mistakes

| Mistake | Why it fails | Fix |
|---|---|---|
| Titanic or Iris projects | Signals tutorial completion | Use a domain problem with a decision |
| Notebooks without narrative | Reviewer cannot find the point | Add a one-page summary |
| Accuracy without calibration | Shows statistical naivety | Add calibration and decision curve |
| No limitations section | Reads as overclaiming | Add specific failure modes |
| Screenshots instead of code | Not verifiable | Link to a repository |
| Proprietary data with no public version | Cannot be assessed | Build a public-safe version |
| Ten shallow projects | Dilutes the signal | Three deep artifacts |
| No reflection | Missing judgement | Add what you would change |

### 5.6 Reporting the metrics

A metric report has one row per metric and six columns. It is short enough to read in a meeting
and complete enough to reproduce.

| Metric | Result | Denominator | 95% interval | Source | Note |
|---|---|---:|---|---|---|
| Reproducibility | 24 | 27 labs | 71.9% to 96.2% | Run logs | One lab short of target |
| Evidence quality | 18 | 20 capstones | 68.9% to 96.0% | Rubric | Two lacked external validation |
| Regulatory literacy | 41 | 45 learners | 79.6% to 95.9% | Quiz | Below 80% target at point estimate |
| Communication | 36 | 45 learners | 66.8% to 89.4% | Oral assessment | Wide interval |
| Equity reporting | 20 | 20 capstones | 83.9% to 100% | Rubric | Meets 100% target |

Three conventions make the report useful.

First, report the point estimate and the interval. A rate of 80% from 20 observations is not the
same as a rate of 80% from 2,000 observations, and the interval shows the difference.

Second, report the denominator in the table, not in a footnote. Readers scan the result column and
miss the footnote.

Third, add a note that states the implication. "One lab short of target" is more useful than a
bare percentage.

The interval also determines the claim. A result of 41 out of 45, with an interval from 79.6% to
95.9%, does not establish that the 80% target was met. The correct statement is that the point
estimate is just below target and the data are compatible with a range that includes it. That is
a less satisfying sentence and a more accurate one.

Where a target is a hard requirement, such as 100% equity reporting, report the count rather than
the rate. A rate of 100% from 20 items is a count, and the count is what the requirement is
about.

## 6. Worked example

This example computes two program metrics and scores a portfolio. The numbers are constructed; the
arithmetic is exact.

### 6.1 Reproducibility

A cohort attempts 27 labs. Automated runs show 24 complete end to end in a clean environment.

```text
Rate = 24 / 27 = 0.8889 (88.9%)
Target = 90%
Shortfall = 90% - 88.9% = 1.1 percentage points
Additional labs needed to reach 90%: 0.90 x 27 = 24.3, so 25 of 27 = 92.6%
```

The program is one lab short of the target. That is a more useful statement than "approximately
90%".

The 95% Wilson interval for 24 out of 27:

```text
p = 0.8889, n = 27, z = 1.96
denominator = 1 + z^2/n = 1 + 3.8416/27 = 1.14228
center = (p + z^2/(2n)) / denominator
       = (0.8889 + 0.07114) / 1.14228
       = 0.96004 / 1.14228
       = 0.84046
margin = z x sqrt(p(1-p)/n + z^2/(4n^2)) / denominator
       = 1.96 x sqrt(0.09877/27 + 3.8416/2916) / 1.14228
       = 1.96 x sqrt(0.003658 + 0.001317) / 1.14228
       = 1.96 x 0.07053 / 1.14228
       = 0.12102
interval = 0.84046 +/- 0.12102 = 0.7194 to 0.9615
```

The point estimate meets nothing confidently: the interval runs from 71.9% to 96.2%. With 27 labs,
the program cannot distinguish 90% from 80% or 95%. That is a measurement limitation, not a
performance failure.

### 6.2 Placement

Sixty graduates are invited to a survey. Forty-two respond. Thirty-three of the respondents are
employed in the field within six months.

```text
Response rate = 42 / 60 = 0.70 (70%)
Placement among respondents = 33 / 42 = 0.7857 (78.6%)
Placement among all graduates = 33 / 60 = 0.55 (55.0%)
```

Both numbers are correct. The first answers "among those who replied, how many are employed". The
second answers "what share of graduates are employed, assuming non-respondents are not employed".
Neither is the true placement rate, because non-respondents may be employed at a different rate.

A sensitivity analysis:

```text
If non-respondents are employed at the same rate as respondents:
  estimated employed = 0.7857 x 60 = 47.1
  rate = 47.1 / 60 = 78.6%

If non-respondents are employed at half the respondent rate:
  estimated employed = 33 + 18 x 0.3929 = 33 + 7.07 = 40.07
  rate = 40.07 / 60 = 66.8%

If no non-respondent is employed:
  rate = 33 / 60 = 55.0%
```

The plausible range is 55.0% to 78.6%. Reporting a single number without that range is a
denominator and response-bias error.

### 6.3 Portfolio scoring

Weights and scores for eight artifacts, on a 0 to 4 scale:

| Artifact | Weight | Score | Weighted points |
|---|---:|---:|---:|
| Project charter | 15 | 3 | 45 |
| Data quality report | 10 | 3 | 30 |
| Risk model with calibration | 20 | 4 | 80 |
| Causal protocol | 10 | 2 | 20 |
| NLP evaluation | 10 | 2 | 20 |
| Regulatory memo | 10 | 3 | 30 |
| Model card and monitoring plan | 15 | 2 | 30 |
| Value case | 10 | 1 | 10 |
| **Total** | **100** | | **265** |

```text
Weighted score = 265 / 100 = 2.65 out of 4
Percentage of maximum = 2.65 / 4 = 66.25%
Target = 3.00 out of 4
Gap = 0.35 points
```

### 6.4 Closing the gap

The two weakest artifacts are the value case (score 1, weight 10) and the model card (score 2,
weight 15).

```text
If the value case improves from 1 to 3:
  gain = 10 x 2 = 20 points -> total 285 -> 2.85

If the model card also improves from 2 to 3:
  gain = 15 x 1 = 15 points -> total 300 -> 3.00
```

Improving two artifacts closes the gap. Improving the risk model from 4 to 5 is not possible, and
improving the charter from 3 to 4 would add only 15 points. The arithmetic points to the same
conclusion as judgement: build the missing value case and complete the model card.

### 6.5 What the score does not measure

The score does not measure whether the artifacts form a coherent argument, whether the narrative is
readable, or whether the candidate can defend the work. A portfolio can score 3.0 and fail an
interview, or score 2.6 and succeed because the candidate explains the limitations well.

The score is a diagnostic, not a verdict.

## 7. Code, query, or calculation

The block computes rates with Wilson intervals, does a response-bias sensitivity analysis, and
scores a portfolio. Standard library only.

```python
"""Program metrics and portfolio scoring."""

from dataclasses import dataclass
from math import sqrt


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - margin, center + margin)


def rate(k: int, n: int) -> dict:
    low, high = wilson(k, n)
    return {
        "numerator": k, "denominator": n,
        "rate": k / n if n else float("nan"),
        "low": low, "high": high,
    }


print("Reproducibility")
r = rate(24, 27)
print(f"  {r['numerator']}/{r['denominator']} = {r['rate']:.4f} "
      f"(95% CI {r['low']:.4f} to {r['high']:.4f})")
print(f"  labs needed for 90%: {int(-(-0.90 * 27 // 1))} of 27")

print("\nPlacement")
p = rate(33, 42)
print(f"  among respondents {p['numerator']}/{p['denominator']} = {p['rate']:.4f} "
      f"(95% CI {p['low']:.4f} to {p['high']:.4f})")
print(f"  among graduates 33/60 = {33 / 60:.4f}")

print("\nNon-response sensitivity")
respondent_rate = 33 / 42
for label, factor in (("same rate", 1.0), ("half rate", 0.5), ("zero", 0.0)):
    employed = 33 + 18 * respondent_rate * factor
    print(f"  {label:10s} estimated employed {employed:5.2f}, rate {employed / 60:.4f}")

print("\nPortfolio score")
ARTIFACTS = [
    ("Project charter", 15, 3),
    ("Data quality report", 10, 3),
    ("Risk model with calibration", 20, 4),
    ("Causal protocol", 10, 2),
    ("NLP evaluation", 10, 2),
    ("Regulatory memo", 10, 3),
    ("Model card and monitoring", 15, 2),
    ("Value case", 10, 1),
]
total_weight = sum(w for _, w, _ in ARTIFACTS)
points = sum(w * s for _, w, s in ARTIFACTS)
print(f"  weighted points {points}/{total_weight} = {points / total_weight:.2f} of 4")
print(f"  percentage {points / total_weight / 4:.2%}")

improved = [(n, w, 3 if n == "Value case" else (3 if n == "Model card and monitoring" else s))
            for n, w, s in ARTIFACTS]
points2 = sum(w * s for _, w, s in improved)
print(f"  after two fixes {points2}/{total_weight} = {points2 / total_weight:.2f} of 4")
```

Expected output:

```text
Reproducibility
  24/27 = 0.8889 (95% CI 0.7194 to 0.9615)
  labs needed for 90%: 25 of 27

Placement
  among respondents 33/42 = 0.7857 (95% CI 0.6406 to 0.8829)
  among graduates 33/60 = 0.5500

Non-response sensitivity
  same rate  estimated employed 47.14, rate 0.7857
  half rate  estimated employed 40.07, rate 0.6679
  zero       estimated employed 33.00, rate 0.5500

Portfolio score
  weighted points 265/100 = 2.65 of 4
  percentage 66.25%
  after two fixes 300/100 = 3.00 of 4
```

Four checks. First, 24 out of 27 is 88.9%, and 25 out of 27 would be 92.6%, so the program is one
lab short. Second, the Wilson interval for 24 out of 27 runs from 71.9% to 96.2%, which means the
cohort is too small to establish the target precisely. Third, the placement rate depends on the
denominator, and the range across plausible non-response assumptions is 55.0% to 78.6%. Fourth,
the portfolio score of 2.65 reaches 3.00 by improving two artifacts, not by improving the
strongest one.

> **Note:** Wilson intervals assume independent observations. Labs within one cohort are not fully
> independent, because the same learners and the same environment affect all of them. Treat the
> interval as a lower bound on uncertainty, not an exact statement.

## 8. Data, standards, and regulatory context

This topic sits mostly outside device and privacy regulation, but three bodies of standards apply:
evaluation frameworks, competency frameworks, and the rules that govern what may be published.

### 8.1 Evaluation frameworks

| Framework | Origin | Levels or domains | Use in this program |
|---|---|---|---|
| Kirkpatrick model | 1959, revised | Reaction, learning, behavior, results | Maps to satisfaction, assessment, practice, and placement |
| Phillips ROI model | 1990s | Adds a fifth level, return on investment | Used for corporate cohorts |
| Competency frameworks | NHS England, AIDH, DECODE | Domains and proficiency levels | Maps modules to role expectations |
| Logic model | Program evaluation | Inputs, activities, outputs, outcomes, impact | Used to design the metric chain |

The Kirkpatrick model is a descriptive taxonomy, not a causal theory. A 2025 retrospective in
Human Resource Development Quarterly, thirty years after Holton's critique, describes it as a
taxonomy that has been modernized rather than replaced, and warns against treating the four levels
as a causal hierarchy. The practical implication is that satisfaction does not cause learning, and
learning does not automatically cause behavior change.

| Kirkpatrick level | Question | Program metric |
|---|---|---|
| 1. Reaction | Did learners find it relevant? | Satisfaction, not a goal |
| 2. Learning | Did they acquire knowledge and skill? | Quiz, exercises, rubric scores |
| 3. Behavior | Do they do it in practice? | Capstone defense, on-the-job artifacts |
| 4. Results | Did outcomes improve? | Placement, promotion, deployed work |

Most programs measure levels 1 and 2 and claim level 4. This program's metrics are deliberately
weighted toward levels 2 and 3, with level 4 tracked but attributed cautiously.

### 8.2 Competency frameworks

Three frameworks are useful for mapping the portfolio to employer language.

**NHS England, National Competency Framework for Data Professionals in Health and Care.** Five
practitioner levels, from associate to leading practitioner, covering data analysis, data
engineering, and data science, with optional specialisms such as machine learning.

**Australasian Institute of Digital Health, Australian Health Informatics Competency Framework,
2nd edition, March 2025.** Six domains: health sciences, information science, information
technology, leadership and management, social and behavioural sciences, and core health
informatics. Competencies are levelled using Bloom's taxonomy.

**DECODE framework, JAMA Network Open, January 2025.** Four domains, 19 competencies, and 33
mandatory plus 145 discretionary learning outcomes, developed by a Delphi consensus of 211 experts
from 79 countries. Domain 4 is health data science.

A portfolio that names the competency each artifact evidences is easier for an employer to assess
than one that lists technologies.

### 8.3 What may be published

A portfolio is a publication. The rules that apply to clinical data also apply to it.

| Data type | Rule | Practical action |
|---|---|---|
| PHI | HIPAA Privacy Rule; institutional policy | Do not publish; use synthetic or aggregate data |
| De-identified data | Check the de-identification standard and the DUA | Confirm publication is permitted |
| Proprietary data | Employment contract; DUA | Publish method only, with approval |
| Third-party data | License terms | Check redistribution rights |
| Images | Consent and license | Use open datasets or synthetic images |
| Model weights | Contract and IP policy | Check before release |

A public-safe portfolio is not a weakened portfolio. It is a portfolio that can be shown, which is
the point.

### 8.4 Alumni surveys and data protection

Placement tracking uses alumni contact data. In the EU, that processing requires a lawful basis
under GDPR, usually consent or legitimate interest, and a retention schedule. In the US, alumni
survey data are generally not PHI, but institutional policy and state privacy law may apply.

The practical requirements are modest: state the purpose, allow withdrawal, report in aggregate,
and suppress small cells. A cell with fewer than five people should not be reported separately,
because it may permit identification.

> **Note:** Small-cell suppression is a convention, not a universal legal requirement. The
> threshold should be set by the institution's privacy office.

### 8.5 Reporting standards for artifacts

If a portfolio artifact reports a prediction model, the reporting expectations are set by
TRIPOD+AI (2024). If it reports an early clinical evaluation, DECIDE-AI (2022) applies. If it
reports an observational study, STROBE is the common reference. Following the relevant checklist
makes the artifact legible to reviewers and is itself evidence of professional competence.

## 9. Industry practice and case evidence

### 9.1 What hiring managers say they use

**Documented.** The 2025 CodePath survey of more than 200 engineering leaders, reported in
December 2025, found that side projects and portfolios (38%), internship experience (35%), and
public code portfolios (34%) were the top hiring signals outside interviews. Degree and academic
focus were cited by 23%, school prestige by 17%, and credentialing programs by 4%.

**Documented.** The CoderPad and CodinGame State of Tech Hiring 2025 report, covering more than
5,000 developers and recruiters, found that about 20.5% of employers used portfolio review as a
skills-assessment method, and 21% used work-sample or take-home tests. Data scientist roles were
targeted by 17.5% of recruiters.

**Inference.** Portfolio review is not universal, but it is common enough to matter and it is the
signal a learner controls most directly. The low weight given to credentialing programs (4%) is a
warning against optimizing for certificates.

### 9.2 The skills gap employers report

**Documented.** The Codio 2025 Data Science and Analytics Talent Survey of 111 senior executives
reported that 57% of employers said new hires lacked industry best practices and 56% said they
lacked current technical knowledge. Statistical analysis, programming, and data manipulation were
simultaneously the most lacking, hardest to recruit, and most valued skills.

**Inference.** "Industry best practices" is the language of process: version control, validation,
documentation, reproducibility. Those are exactly the criteria in the portfolio rubric in section
5.3. A portfolio that demonstrates them addresses the gap employers name.

### 9.3 The tutorial project problem

**Documented.** A 2025 analysis of 1,200 data science portfolios by a hiring manager, published on
Medium, reported that tutorial projects correlated with rejection, citing Titanic, Iris, stock
prediction, MovieLens, and housing price projects. The analysis is anecdotal and not peer
reviewed.

**Inference.** The mechanism is plausible: tutorial projects signal that the candidate has
completed instruction but not tackled an ambiguous problem. The response is not to avoid common
datasets; it is to use them for a decision that matters, with validation and limitations.

### 9.4 Evaluating training outcomes

**Documented.** The 2025 literature on the Kirkpatrick model treats it as a design and measurement
backbone rather than a causal hierarchy. A 2025 structural equation modelling study in the Journal
of Agricultural Education and Extension (n = 300) empirically tested links across the four levels
and found support for the sequence in that context. A 2025 retrospective in Human Resource
Development Quarterly warned against algorithmic reductionism in AI-enabled evaluation.

**Inference.** The evidence supports using the levels as a checklist and does not support claiming
that level 1 causes level 4. A program that reports placement should also report the response rate
and the attribution limits.

### 9.5 What a strong portfolio looks like

**Inference, from public hiring practice and the sources above.** A strong portfolio has three to
five artifacts, each with a decision, a validation, a limitation, and a short narrative. It is
public-safe, reproducible, and accompanied by a five-minute explanation. It is not a list of
technologies and it is not twenty notebooks.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Activity metrics as outcomes | Reporting hours and completions | Easy to collect | False confidence | Use outcome metrics |
| Missing denominator | "78% placement" | Rate computed on respondents only | Misleading claim | State numerator and denominator |
| Response bias ignored | Low survey response | Non-respondents differ | Overstated placement | Sensitivity analysis |
| Vanity portfolio | Many tutorials, no decision | Optimizing for volume | Weak signal | Three deep artifacts |
| No limitations section | Confident tone throughout | Overclaiming | Credibility loss | State failure modes |
| Unreproducible artifact | "Works on my machine" | Environment drift | Cannot be assessed | Container and pinned versions |
| Proprietary data only | Nothing can be shown | Confidentiality | No evidence | Public-safe version |
| Certificate stacking | Many short courses | Credential chasing | Low weight in hiring | One substantial project |
| Metric gaming | Target met by weakening the task | Goodhart's law | Illusory quality | Pair with rubric and review |
| No external review | Self-assessment only | Blind spots | Polished but weak | Two external reviewers |
| Stale portfolio | Last update two years ago | Neglect | Appears inactive | Quarterly review |
| Overweighting the strongest artifact | Perfecting the model | Comfort | Ignores gaps | Improve the weakest weighted item |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Outcomes can be attributed to the program | Reporting placement | Selection and market effects | Compare with base rates |
| Survey respondents represent graduates | Placement estimate | Response bias | Sensitivity analysis |
| Rubric scores are comparable across raters | Portfolio scoring | Rater variability | Calibration and kappa |
| Artifacts predict job performance | Portfolio as signal | Weak correlation | Follow-up studies |
| Public-safe artifacts are equivalent evidence | Portfolio | Loss of detail | Reviewer feedback |
| Metrics are stable over cohorts | Trend analysis | Curriculum changes | Track by cohort |
| Employers value portfolios | Strategy | Segment differences | Job description review |
| Maintenance is feasible quarterly | Currency | Time pressure | Calendar the review |

### 11.1 Limits

The hiring evidence in section 9 comes from surveys with different samples and definitions. The
CodePath survey covers engineering leaders, not specifically biomedical employers. The CoderPad
report covers developers and recruiters broadly. Neither is a study of this program's graduates.

The rubric in section 5.3 is a teaching device. It has not been validated for inter-rater
reliability, and two reviewers may score the same artifact differently.

The Wilson interval assumes independent observations, which is not fully true within a cohort.

## 12. Exercises

### Exercise 1: Classify metrics (recall, 20 minutes)

**Deliverable.** Twelve metrics classified as activity, output, outcome, or impact.

**Acceptance criteria.** Each classification justified in one line. At least three are ambiguous
and the ambiguity is explained.

**Time.** 20 minutes.

### Exercise 2: Rate with a denominator (calculation, 30 minutes)

**Deliverable.** For 18 of 22 capstones meeting a criterion, compute the rate, the Wilson interval,
and the count needed to reach 90%.

**Acceptance criteria.** Arithmetic shown. Interval interpreted. Count to target computed.

**Time.** 30 minutes.

### Exercise 3: Portfolio plan (design, 40 minutes)

**Deliverable.** A portfolio plan with five artifacts, the competency each evidences, and the
audience for each.

**Acceptance criteria.** Each artifact maps to a role capability. At least one is public-safe by
design. No artifact is a tutorial reproduction.

**Time.** 40 minutes.

### Exercise 4: Score a portfolio (application, 35 minutes)

**Deliverable.** Score your current portfolio against the rubric, compute the weighted score, and
identify the largest gap.

**Acceptance criteria.** Scores justified with evidence. Weights sum to 100. The gap analysis names
one action.

**Time.** 35 minutes.

### Exercise 5: Narrative (design, 30 minutes)

**Deliverable.** A one-page portfolio narrative with four claims and links to evidence.

**Acceptance criteria.** Readable in ten minutes. No jargon without definition. Each claim points
to a specific artifact.

**Time.** 30 minutes.

### Exercise 6: Critique a success claim (critique, 25 minutes)

**Deliverable.** A 300-word critique of a program's claim that "90% of graduates are employed in
data roles".

**Acceptance criteria.** Identifies the denominator, the response rate, the time window, and the
definition of "data role". Proposes a corrected claim.

**Time.** 25 minutes.

### Exercise 7: Five-minute presentation (application, 30 minutes)

**Deliverable.** A five-minute presentation of one artifact, timed, with the five-part structure.

**Acceptance criteria.** Decision in the first 30 seconds. Limitation stated. No more than five
minutes.

**Time.** 30 minutes.

### Exercise 8: Maintenance plan (design, 20 minutes)

**Deliverable.** A 90-day portfolio maintenance plan with dates and actions.

**Acceptance criteria.** At least one artifact added or updated. One artifact removed or archived.
A review date in the calendar.

**Time.** 20 minutes.

## 13. Self-check answers

**1. What is the difference between activity and outcome metrics?**
Activity measures effort, such as hours or completions. Outcome measures a change in capability or
behavior, such as independent delivery or placement.

**2. Why is completion rate a weak metric?**
Because it can be high for easy content, weak assessment, or already-skilled learners. It does not
measure capability.

**3. What are the four Kirkpatrick levels?**
Reaction, learning, behavior, and results.

**4. Compute the rate and Wilson interval for 24 of 27.**
Rate 88.9%; Wilson 95% interval 71.9% to 96.2%.

**5. Why does the placement denominator matter?**
Because 33 of 42 respondents is 78.6%, while 33 of 60 graduates is 55.0%. The two answer different
questions.

**6. A portfolio scores 265 weighted points out of a possible 400. What is the score out of 4?**
265/100 = 2.65 out of 4, or 66.25% of the maximum.

**7. Which artifact should you improve first?**
The one with the largest weighted gap, usually the weakest artifact with a high weight. In the
example, the value case and the model card.

**8. Why is a public-safe portfolio important?**
Because a portfolio that cannot be shown cannot function as evidence. A public-safe version with
open or synthetic data can be assessed.

**9. What does Goodhart's law warn about?**
When a measure becomes a target, people optimize the measure rather than the goal. Pair metrics
with rubrics and external review.

**10. What makes a five-minute presentation effective?**
It states the decision first, translates the result into decision terms, and states the limitation
before the audience asks.

### 13.1 Rubric for exercises 3 and 5

Exercise 3 is scored on mapping to role capabilities (0 to 3), depth over volume (0 to 3), and
public safety (0 to 2).

Exercise 5 is scored on the clarity of the four claims, the specificity of the evidence links, and
readability for a non-specialist.

## 14. Interview questions

**1. "What is in your portfolio?"**
Strong: three artifacts with decisions, validation, and limitations. Weak: a list of courses.

**2. "Which artifact are you least proud of, and why?"**
Strong: names a specific weakness and what was learned. Weak: deflects.

**3. "How do you know your model works?"**
Strong: temporal or external validation, calibration, and a decision threshold. Weak: "high
accuracy".

**4. "What would you do differently on that project?"**
Strong: names a design or data choice and its consequence. Weak: "nothing".

**5. "How do you measure the success of a program?"**
Strong: outcomes with denominators, intervals, and attribution limits. Weak: completion rates.

**6. "Why is placement rate hard to measure?"**
Strong: response bias, definition of employment, time window, denominator. Weak: "people do not
respond".

**7. "What makes a portfolio credible?"**
Strong: real decisions, reproducible methods, honest limitations. Weak: impressive metrics.

**8. "How do you present to a non-technical audience?"**
Strong: decision, result in their units, limitation, next step. Weak: explains the architecture.

**9. "How do you keep your portfolio current?"**
Strong: quarterly review, one new artifact, remove stale work. Weak: "I update my CV".

**10. "What is a vanity metric?"**
Strong: a metric that looks good but does not inform a decision; gives an example. Weak: "any
metric I do not like".

**11. "How would you evaluate this program?"**
Strong: Kirkpatrick levels, artifact rubric, placement with response analysis, external review.
Weak: satisfaction survey.

**12. "What evidence would convince you that a candidate is competent?"**
Strong: an artifact they can explain, defend, and criticize. Weak: a credential.

## 15. Key terms

- **Activity metric** — measure of effort.
- **Artifact** — tangible output demonstrating work.
- **Attrition** — loss of participants over time.
- **Baseline** — starting value.
- **Capstone defense** — oral examination of individual contribution.
- **Competency** — ability to perform to a standard.
- **Confidence interval** — range of plausible values.
- **Denominator** — population that could be counted.
- **Effect size** — magnitude of a difference.
- **Goodhart's law** — measures become targets and cease to be good measures.
- **Impact metric** — change in the world.
- **Inter-rater reliability** — agreement between raters.
- **Kirkpatrick model** — four-level training evaluation framework.
- **KPI** — key performance indicator.
- **Lagging indicator** — outcome itself.
- **Leading indicator** — predictor of a later outcome.
- **Metric contract** — definition, numerator, denominator, source, threshold.
- **Narrative** — short argument connecting artifacts to a role.
- **Numerator** — count meeting the definition.
- **Outcome metric** — change in capability or behavior.
- **Output metric** — things produced.
- **Placement rate** — share employed in the field.
- **Portfolio** — curated artifacts with a narrative.
- **Public-safe** — no proprietary or identifiable data.
- **Response bias** — difference between respondents and non-respondents.
- **Rubric** — criteria and performance levels.
- **Vanity metric** — looks good, does not inform decisions.
- **Wilson interval** — confidence interval for a proportion.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- CodePath, survey of more than 200 engineering leaders, reported December 2025 (via Fortune).
  https://tech.yahoo.com/business/articles/want-job-ai-era-tech-130000289.html
- CoderPad and CodinGame, State of Tech Hiring 2025 (more than 5,000 developers and recruiters).
  https://coderpad.io/survey-reports/coderpad-and-codingame-state-of-tech-hiring-2025/
- Codio, 2025 Data Science and Analytics Talent Survey (111 senior executives).
  https://www.codio.com/blog/2025-industry-survey-data-skills-gap
- Egan, T., Kim, J., Ghosh, R., "Assessing Training Outcomes From Levels to Learning Ecosystems in
  the Age of AI: Thirty Years After Critiquing 'The Flawed Four-Level Evaluation Model',"
  *Human Resource Development Quarterly*, 2025. https://doi.org/10.1002/hrdq.70007
- Sathishwaran et al., structural equation modelling analysis of the Kirkpatrick model, *Journal
  of Agricultural Education and Extension*, 2025.
  https://doi.org/10.1080/1389224x.2025.2573498
- NHS England, National Competency Framework for Data Professionals in Health and Care.
  https://transform.england.nhs.uk/media/documents/NCF_Framework_Booklet.pdf
- Australasian Institute of Digital Health, Australian Health Informatics Competency Framework,
  2nd edition, March 2025.
  https://digitalhealth.org.au/chia/wp-content/uploads/sites/37/2025/03/Second-Edition_AHICF_March-2025.pdf
- Digital Health Competencies in Medical Education (DECODE) framework, *JAMA Network Open*,
  January 2025.
- Collins, G. S., Moons, K. G. M., Dhiman, P., et al., "TRIPOD+AI statement," *BMJ*
  2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Vasey, B., Nagendran, M., Campbell, B., et al., "DECIDE-AI," *Nature Medicine*
  2022;28:924-933. https://doi.org/10.1038/s41591-022-01772-9
- STROBE Initiative, Strengthening the Reporting of Observational Studies in Epidemiology.
  https://www.strobe-statement.org/
- US Department of Health and Human Services, HIPAA Privacy Rule.
  https://www.hhs.gov/hipaa/
- European Parliament and Council, Regulation (EU) 2016/679 (GDPR), Articles 6, 9, 17, 35.

> **Note:** The hiring surveys cited here are secondary summaries of survey results and cover
> technology employers broadly, not biomedical employers specifically. Treat the percentages as
> directional.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_how-to-use-this-package.md`](09_how-to-use-this-package.md)
- Related topic: [`01_why-this-course-exists.md`](01_why-this-course-exists.md)
- Related topic: [`04_career-tracks-and-role-definitions.md`](04_career-tracks-and-role-definitions.md)
- Related topic: [`06_evidence-hierarchy-for-industry-claims.md`](06_evidence-hierarchy-for-industry-claims.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision-curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
