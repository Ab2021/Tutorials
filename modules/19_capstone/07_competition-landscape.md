# Competition Landscape 2025-2026

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 07 of 11
> Estimated study time: 4 hours
> Prerequisites: Topics 04 and 05
> Version 2026.3

## 1. Why this topic exists

Public challenges are the closest thing the field has to a shared benchmark.
They publish a task, a dataset, a metric, and a leaderboard, and they produce a record of what
worked under constraints that no single lab controls.
For a student, they are a calibration: they show what a strong result looks like on a problem that
hundreds of teams have attempted.
For an employer, they are a signal: a candidate who can explain why a winning solution won, and
what it did not prove, is more useful than one who quotes a rank.

The landscape is also a trap.
Challenge results are frequently cited as if they were clinical evidence, and winning methods are
described as if they were validated products.
Neither is true.
A leaderboard measures performance on a specific hidden split under a specific metric, and a
winning solution is a description of what scored well, not a controlled comparison of methods.

This topic teaches three things: how to read the landscape, how to evaluate a competition before
committing a team, and how to extract transferable lessons without importing the format's
blind spots.
It covers the 2025 and 2026 cycles where they can be verified, and it is explicit about what
cannot be verified.
A 2026 event that has not concluded is described as a category and a status, not as a result.

The status vocabulary matters, and it is used consistently in this topic.
**Concluded** means the event took place and results were reported.
**Closed, results pending** means the submission window closed but outcomes were not published at
the source cutoff.
**Open** means submissions are being accepted.
**Announced only** means the event is scheduled but not held.
**Recurring** means a cadence exists but the specific edition must be checked.
**Unverified** means the source could not confirm details.

The reason for the vocabulary is that a course which invents a winner teaches students to invent
evidence.

There is also a practical use for the landscape that has nothing to do with entering.
A team designing a capstone can use a challenge's contract as a template: the task definition
forces precision about the prediction, the data page forces a provenance record, and the metric
page forces a decision about the operating point. Even a team that never submits benefits from
reading three or four contracts carefully, because the format makes explicit what a capstone
dossier often leaves implicit.
A statement such as "the 2026 challenge was won by a transformer ensemble" is unverifiable at the
time of writing unless the organisers have published it, and repeating it would model exactly the
behaviour the programme penalises in capstones.

The same discipline applies to scores. A score is reported with its metric, split, and edition, or
it is not reported. A number without those three attributes cannot be checked, and an unchecked
number in a dossier is a liability rather than an asset.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a public challenge by status, task type, metric family, and data regime, using the
   official page rather than a secondary summary.
2. **Explain** how a capacity-constrained metric differs from AUROC, and compute the challenge
   score for a small ranking by hand and in code.
3. **Evaluate** whether a competition is suitable for a given team, using data access, compute,
   metric fit, skill, and time as criteria.
4. **Distinguish** a leaderboard result from clinical evidence, and state what each supports.
5. **Identify** the recurring technical patterns in winning solutions across imaging, signals,
   omics, and language tasks.
6. **Detect** the common failure modes of challenge evidence: contamination, split leakage,
   leaderboard probing, and post-hoc metric selection.
7. **Extract** transferable lessons for a capstone without importing the competition's incentive
   to overfit a metric.
8. **Report** a competition result honestly, with the metric, the split, the rank, and the
   limitations in one paragraph.

Outcomes 1 to 3 are assessed in Exercises 1 to 3 and tested at the midpoint.
Outcomes 4 and 5 are assessed in Exercises 4 and 7 and appear in the defense as the
"what does that number mean" question.
Outcomes 6 and 8 are assessed in Exercises 5 and 6 and are the two items most often missing from a
competition section in a dossier.
Outcome 7 is assessed in the capstone's method section, where the team must say which pattern it
adopted and why.

A team that can perform all eight can read a challenge page in twenty minutes and decide whether it
is relevant, which is the practical skill this topic is for.

## 3. The core mental model

A challenge is an evaluation contract with five clauses: task, data, split, metric, and rules.
The leaderboard is the output of the contract, not a property of the method.
Change any clause and the ranking can change, which is why comparing across challenges is usually
meaningless.

```text
   TASK  ---- what is predicted, from what inputs, at what granularity
     |
   DATA  ---- sources, sites, languages, label provenance, licence
     |
   SPLIT ---- public leaderboard / private holdout / hidden site
     |
   METRIC --- accuracy, AUROC, capacity-constrained TPR, Dice, kappa, LLM rubric
     |
   RULES ---- external data, pretrained weights, team merging, code release
     |
     v
   LEADERBOARD  (a ranking under this contract, not a clinical claim)
     |
     +--> what transfers: method patterns, data handling, validation discipline
     +--> what does not: rank as value, score as benefit, winner as product
```

Read the diagram from the top.
Each clause constrains the next.
A task at the level of a whole study requires a different split from a task at the level of a
lesion, and a metric that ignores capacity produces a different winner from one that respects it.

Three properties of the model matter.

**The contract is fixed before the results.** That is what makes a challenge informative: every
team is scored on the same task, data, split, and metric. It is also what makes a challenge
narrow: the result says nothing about a different task, a different population, or a different
operating point.

**The hidden split is the evidence.** The public leaderboard is a development signal. A team that
optimises against it is fitting a sample of the test distribution, and the private split is where
that shows. This is the same logic as a locked holdout in a capstone, at a larger scale.

**Rules are part of the contract.** External data, pretrained weights, and team merging change
what a score means. A score achieved with external data is not comparable to one achieved without
it, even on the same leaderboard, unless the rules separate the categories.

A fourth property is sociological.
Challenges are also recruiting and marketing events. Team counts reflect prize money, novelty, and
the organisers' reach as much as difficulty. A 1,800-team challenge is not necessarily harder than
a 40-team one; it may simply be better publicised.

### The three questions a result must answer

Before any competition result enters a dossier, it should answer three questions.

What was measured? The metric, the split, and the population.
Against whom? The other teams, the organisers' baseline, or the team's own simpler model.
What does it not show? Calibration, subgroups, external sites, workflow impact, and maintenance.

A result that answers all three is a benchmark finding. A result that answers only the first is a
score. The difference is what a panel is testing when it asks "what does that number mean?"

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Challenge | A time-boxed competition with a fixed task, data, split, metric, and rules | Any benchmark | Event and year |
| Track | A sub-competition with its own task or metric | A team category | Track name |
| Public leaderboard | Scores on a visible subset during the competition | The result | Ranking |
| Private leaderboard | Scores on a held-out subset released at the end | A formality | Ranking |
| Hidden site | A data source withheld entirely from participants | A held-out row set | Site identifier |
| Shake-up | Change in ranking between public and private leaderboards | A scandal | Rank correlation |
| Capacity-constrained metric | A metric computed at a fixed operating capacity, such as the top 5 percent | A threshold-free metric | TPR at k, precision at k |
| TPR at k | Fraction of positives captured in the top k ranked cases | Precision at k | Proportion |
| Precision at k | Fraction of the top k cases that are positive | TPR at k | Proportion |
| Contamination | Test data present in pretraining or model selection | Any overlap | Screen result |
| Leaderboard probing | Submitting many variants to infer the test set | Normal iteration | Submission count |
| External data | Data not provided by the organisers | Any public dataset | Rule category |
| Pretrained weights | A model trained before the competition | Fine-tuning | Rule category |
| Code competition | A format requiring notebook submission with runtime limits | Any Kaggle competition | Rule set |
| Efficiency track | A sub-track scoring performance per unit of compute or runtime | A speed bonus | Metric and cap |
| Benchmark harness | The code that runs and scores submissions | The leaderboard website | Version |
| Winning solution write-up | A team's description of its approach after the event | A peer-reviewed method | Document |
| Post-challenge analysis | Organiser or community analysis of what the results mean | The leaderboard | Paper or report |
| Label provenance | How the target was ascertained | The label column | Source and adjudication |
| Prevalence shift | Difference in outcome frequency between development and deployment | Covariate shift | Proportion |
| Status | Concluded, closed, open, announced, recurring, unverified | Assumed | Label and date |
| Baseline entry | The organisers' or a simple model's score on the same split | The leaderboard median | Score and model |
| Score distribution | Spread of scores across ranked entries | The top score only | Median and interquartile range |
| Ranked entries | Teams with a valid submission | Registered teams | Count |
| Rule category | The division a score belongs to under the rules | The overall leaderboard | Category label |
| Method pattern | A transferable design choice, conditional on task and resources | A recipe | Pattern and condition |
| Gap analysis | Comparison of a team's score with a stronger entry's approach | A rank | Difference and cause |

The distinction between a ranked entry and a registered team matters more than it appears.
Challenges often report thousands of registrations and hundreds of valid submissions, and the
denominator chosen changes the apparent achievement. A rank of 40 out of 41 is different from a
rank of 40 out of 1,100, and only the ranked count is meaningful.

The most important definition is the capacity-constrained metric.
A model that ranks well across the whole cohort and a model that finds the sickest 5 percent are
different products. A challenge that scores the second is testing a decision, and a challenge that
scores AUROC is testing a ranking. The two can produce different winners, and a team that reports
the wrong one has misread the contract.

A useful discipline is to write the metric definition in the team's own words before reading any
results. If the paraphrase disagrees with the official definition, the team has found a
misunderstanding while it is still cheap to correct.

## 5. The mechanics, step by step

The procedure below evaluates a competition for a team and extracts its lessons. It takes about
three hours for a full evaluation, and less if the competition has concluded and published a
post-challenge analysis.

### Step 1. Verify the status on the official page

**Input:** the competition name and year.

**Output:** a status label, a source URL, and an access date.

**Rationale:** Secondary summaries confuse announced events with concluded ones, and a team that
plans around an unheld event wastes weeks. The official page is the only reliable source for
dates, rules, and results.

**Branch point:** if the official page is unavailable or the status is ambiguous, label it
unverified and do not plan around it. A category description is safer than a fabricated date.

### Step 2. Map the task to a decision

**Input:** the task description.

**Output:** a sentence describing what action the output would support in a real setting, or a
statement that the task is a methods benchmark.

**Rationale:** Most challenges are methods benchmarks, not clinical decisions. That is legitimate
and useful, but it changes what the result means. A segmentation challenge tests a method's Dice
score, not whether a surgeon would use the segmentation.

**Branch point:** if the task has no plausible decision, treat it as a benchmark and evaluate it
on method-transfer grounds rather than clinical relevance.

### Step 3. Inspect the data regime and licence

**Input:** the data page and the rules.

**Output:** the sources, the number of sites, the languages, the label provenance, the licence,
and the access route.

**Rationale:** The data regime determines both the evidence and the schedule. A credentialed
dataset needs training and an agreement; a Kaggle competition needs an account and acceptance of
rules; a Synapse challenge may need registration and a data-use agreement.

**Branch point:** if the licence prohibits the intended post-competition use, the competition is a
learning exercise, not a route to a product. Record that before the team invests.

### Step 4. Read the metric definition precisely

**Input:** the evaluation page and the scoring code if available.

**Output:** the metric, the operating point, the tie handling, and the aggregation across tasks.

**Rationale:** Metric definitions differ in details that change strategy. A macro average across
findings weights rare findings equally with common ones. A capacity-constrained score depends on
the cohort size and the tie rule. A weighted kappa depends on the ordinal distance definition.

**Branch point:** if the scoring code is available, read it. If it is not, write down the
interpretation the team will use and check it against the leaderboard's behaviour on a public
submission.

A concrete example of why the detail matters: a metric described as "macro AUC across 12 findings"
gives each finding equal weight, so a rare finding with 40 positives contributes as much as a
common one with 4,000. A model that is excellent on the common findings and poor on the rare ones
will score lower than its overall accuracy suggests, and a team that optimises overall accuracy
will be surprised by the leaderboard.

### Step 5. Identify the split structure

**Input:** the data page and the leaderboard description.

**Output:** the public and private split sizes, whether sites are held out, and whether the test
set is released after the event.

**Rationale:** The split determines what the leaderboard measures. A random row split measures
generalisation to new rows; a site-held-out split measures generalisation to a new environment,
which is a much stronger claim.

**Branch point:** if the test set is released after the event, the team can evaluate its own model
on it, which is useful for a post-challenge analysis but must be labelled as such.

### Step 6. Estimate the compute requirement

**Input:** the data size and the task.

**Output:** an estimate of training hours, memory, and storage, and whether the team's resources
are sufficient.

**Rationale:** Imaging and video challenges are compute-bound in a way that tabular challenges are
not. A 3D segmentation task on thousands of volumes may need more GPU hours than a course team
has, and a code competition with a nine-hour runtime cap constrains the inference pipeline as well
as training.

**Branch point:** if compute is insufficient, the competition is a spectator exercise. Reading the
winning write-ups is still valuable, but entering is not.

### Step 7. Check eligibility and team rules

**Input:** the rules page.

**Output:** eligibility, team size, merging policy, and disclosure requirements.

**Rationale:** Eligibility rules exclude some institutions and countries; team rules affect
strategy; disclosure rules affect what can be published. A team that discovers an eligibility
problem after six weeks has wasted the effort.

**Branch point:** if the team is ineligible, use the competition as a benchmark rather than an
entry. The data may still be accessible for research.

### Step 8. Read the previous edition's results and write-ups

**Input:** the prior year's leaderboard, winner write-ups, and any post-challenge paper.

**Output:** a summary of the winning approaches, the score distribution, and the reported
limitations.

**Rationale:** Previous editions reveal the task's difficulty and the patterns that work. They
also reveal the failure modes the organisers later analysed, which is often more instructive than
the winning method.

**Branch point:** if the write-ups describe methods that depend on resources the team lacks, the
competition is not a realistic entry. If they describe simple methods near the top, it may be.

### Step 9. Assess whether the metric matches the team's decision

**Input:** the metric and the team's intended use.

**Output:** a statement of fit, or a statement that the challenge metric is not the decision metric.

**Rationale:** A competition optimises its metric; a capstone optimises a decision. If the two
diverge, the competition is a source of methods, not a source of evidence. A team that reports a
leaderboard score as evidence for a decision has conflated the two.

**Branch point:** if the metric diverges, use the competition to select a method and evaluate it
on the capstone's own metric.

### Step 10. Estimate the effort and the opportunity cost

**Input:** the task, the data, and the team's hours.

**Output:** an estimate of hours to a competitive entry, and the capstone work it would displace.

**Rationale:** A competitive entry in a large imaging challenge typically takes 150 to 400 hours,
which is most of a twelve-week capstone. The opportunity cost is the capstone evidence that does
not get built.

**Branch point:** if the effort exceeds 40 percent of the capstone capacity, enter only if the
challenge is the capstone's data source and the write-up is an acceptable artefact.

### Step 11. Screen for contamination and leakage

**Input:** the task, the data, and the model.

**Output:** a contamination assessment for pretrained models and a leakage assessment for features.

**Rationale:** Public benchmarks can be in pretraining corpora, and medical images can appear in
web-scraped datasets. A model that has seen the test set will score well and generalise poorly.
The screen is a check, not a guarantee.

**Branch point:** if contamination cannot be ruled out, report it as a limitation and prefer a
model trained from scratch or on a documented corpus.

### Step 12. Decide: enter, benchmark, or observe

**Input:** steps 1 to 11.

**Output:** one of three decisions with a written rationale.

**Rationale:** Entering is appropriate when the data are accessible, the compute is sufficient,
the metric matches the decision, and the effort fits. Benchmarking is appropriate when the data
are accessible but the effort does not fit. Observing is appropriate when neither holds, and it
is still useful for learning the method patterns.

**Branch point:** if the decision is to enter, the competition becomes the capstone's data plan
and must be recorded in the charter. If it is to benchmark, the competition becomes a comparison
baseline.

### Step 13. Extract the transferable lessons

**Input:** the winning write-ups and the post-challenge analysis.

**Output:** three to five method patterns, each with the condition under which it applies.

**Rationale:** The lessons that transfer are patterns, not recipes: localisation before
classification, vessel-first pipelines, label-noise handling, simple baselines, augmentation, and
ensembling. Each is conditional on the task and the data.

**Branch point:** if a pattern depends on resources the team lacks, record the dependency rather
than the pattern. "Train three nnU-Net variants and ensemble" is not transferable to a team with
one GPU.

### Step 14. Report the result honestly

**Input:** the score, the rank, the split, and the limitations.

**Output:** one paragraph with the metric, the split, the rank or score, the team count if known,
and what the result does not show.

**Rationale:** The paragraph is the artefact that prevents overclaiming. It is also what an
interviewer will ask for, and a candidate who can produce it is more credible than one who quotes
a rank.

**Branch point:** if the result is a private-leaderboard rank, say so; if it is a public
leaderboard rank, say that too, because the difference matters.

### Step 15. Read the organiser's analysis, if one exists

**Input:** the post-challenge paper, the special issue, or the organiser's report.

**Output:** three findings about what the scores mean, which may contradict the leaderboard.

**Rationale:** Organisers see the test distribution, the subgroup performance, and the failure
patterns that participants cannot. Their analysis is often the most valuable artefact the
challenge produces, and it is frequently published a year after the leaderboard.

**Branch point:** if the analysis identifies a leakage or shortcut that affected the ranking,
treat the affected scores as method demonstrations rather than as evidence. The Sea-AD metadata
finding is the clearest example.

### Common metric families

Five families cover most biomedical challenges, and each implies a different failure mode.

Threshold-free ranking metrics, such as AUROC and AUPRC. They measure ordering across all
thresholds and say nothing about calibration or capacity.
Capacity-constrained metrics, such as TPR at k or precision at k. They match a fixed review
capacity and depend on the cohort size and the tie rule.
Threshold metrics, such as sensitivity at a fixed specificity. They fix one error rate and measure
the other, which matches a rule-out or rule-in decision.
Agreement metrics, such as Cohen's kappa, quadratic weighted kappa, or Dice. They measure
agreement with a reference, and they are sensitive to class imbalance and ordinal distance.
Composite or rubric metrics, such as a clinician-graded score or a weighted average across tasks.
They measure a judgement, and their validity depends on the rubric and the graders.

A team should identify the family before reading any score, because a score from one family cannot
be compared with a score from another.

### Reading a leaderboard honestly

Four numbers are worth recording, and none of them is the rank alone.

The score itself, with the metric definition and the split. A score without a metric and a split
is not a number; it is a claim.
The score distribution, or at least the median and the interquartile range. A score in the top
decile of a tight distribution means something different from the same score in a wide one.
The team count and the number of ranked entries. Many teams register and never submit, so the
ranked count is the relevant denominator.
The rule category. If external data or pretrained weights were permitted, the score belongs to a
category, not to the task in general.

A team that records these four can compare its result with another team's honestly, and can say
what the comparison does not establish.

### What a leaderboard cannot show

A leaderboard cannot show calibration, because most metrics do not measure it.
It cannot show subgroup performance, because the private split is rarely stratified in a way that
participants can see.
It cannot show performance at a different prevalence, because the test prevalence is fixed.
It cannot show workflow impact, because no clinician is using the output.
It cannot show maintenance cost, because the model is evaluated once.
It cannot show failure behaviour, because the test set contains no outages, missing fields, or
adversarial inputs.

A capstone can show some of these. That is the division of labour between the two formats, and a
dossier that uses a leaderboard score as evidence for any of the six has overclaimed.

### Decision table: enter, benchmark, or observe

| Condition | Enter | Benchmark | Observe |
|---|---|---|---|
| Data accessible within two weeks | Yes | Yes | Yes |
| Compute sufficient for a competitive entry | Required | Not required | Not required |
| Metric matches the capstone decision | Preferred | Not required | Not required |
| Effort under 40 percent of capstone capacity | Required | Yes | Yes |
| Team eligible and rules permit publication | Required | Yes | Yes |
| Contamination screen passes | Required | Preferred | Not required |

### Edge cases and forced decisions

**The competition closes before the capstone starts.** Use the released data and the published
results as a benchmark. A post-challenge analysis on the released test set is legitimate if it is
labelled as post-hoc.

**The metric changes between editions.** Treat the editions as different competitions. A score
from one is not comparable to a score from the other, and a team that combines them produces an
invalid comparison.

**The organisers publish a post-challenge analysis that contradicts the leaderboard.** The
analysis is usually more informative, because it examines what the scores mean. A team should cite
the analysis, not only the rank.

**The team cannot access the data in time.** Switch to benchmark mode with a public equivalent, or
observe. Do not enter a competition whose data you cannot legally use.

**A team member wants to keep probing the leaderboard.** Cap submissions per week and log them.
Probing is a form of test-set fitting, and the cap is the control.

**The winning solution used external data that the rules permitted.** That is legitimate, and it
means the result is not comparable to a no-external-data entry. Report the rule category with the
score.

## 6. Worked example

A four-person team is choosing between three options for a twelve-week capstone: the concluded
2025 PhysioNet Chagas challenge, the closed 2026 RSNA knee challenge, and a local one-week
datathon followed by an eight-week extension.

### The three contracts

The 2025 PhysioNet challenge asked teams to identify Chagas disease from 12-lead ECGs to
prioritise scarce serological testing. The data were multi-source: CODE-15%, SaMi-Trop, and
PTB-XL for training, with hidden validation and test sets from other sources. The metric was the
fraction of true cases captured in the top 5 percent of the ranking. The organisers report 111
teams, 630 participants, and 41 ranked entries. Results were released on 24 September 2025.

The 2026 RSNA knee challenge asked for detection of 12 knee abnormalities from MRI plus report
text, with more than 5,000 exams from 19 sites and reports in about 12 languages. The metric was a
macro-averaged ROC AUC across the 12 findings, with a separate efficiency track. The submission
window closed on 22 October 2026, and results were pending at the source cutoff.

The local datathon provides a de-identified extract from one hospital, a week of structured
mentoring, and no fixed metric beyond the organiser's rubric.

### The feasibility arithmetic

The team scores each option on data access, compute, metric fit, skill, and time, with weights of
0.25, 0.20, 0.20, 0.15, and 0.20.

| Option | Data | Compute | Metric fit | Skill | Time | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| PhysioNet 2025 | 5 | 4 | 5 | 3 | 4 | 4.30 |
| RSNA 2026 knee | 3 | 1 | 4 | 2 | 2 | 2.45 |
| Local datathon | 5 | 5 | 4 | 4 | 5 | 4.65 |

The datathon wins on feasibility. The PhysioNet option is second and has the advantage of a
concluded, well-documented challenge with a metric that matches a capacity-constrained decision.
The RSNA option fails on compute and skill: 3D MRI across 12 findings needs GPU hours and
musculoskeletal expertise the team does not have.

### Diagnostic one: the metric is not the decision

The team's intended decision is which patients a community clinic prioritises for confirmatory
testing at a fixed capacity. The PhysioNet metric, TPR at the top 5 percent, matches that decision
closely. The datathon has no such metric; its rubric rewards insight and reproducibility.

A team that enters the datathon and reports a strong AUC has not measured the decision. That is
the first diagnostic: if the available metric does not match the decision, the competition is a
source of methods, not evidence.

### Diagnostic two: the data cannot support the claim

For the local datathon, the extract contains 1,900 admissions over two years and 58 positive
cases. For a half-width of 0.10 around a sensitivity of 0.70, the team needs about 81 positives,
so the extract is short by 23. The team can report a wider interval, or pool with a public
dataset, or narrow the claim.

The RSNA option fails a different version of the same test: 5,000 exams sounds large, but the
private split and the 12-finding macro average mean that some findings will have few positives,
and the team cannot check because the labels are not public.

### The revision, and one assumption varied

The team chooses the PhysioNet route, using the released 2025 data as a benchmark and the published
results as a calibration. It does not attempt to reproduce the winning score; it tests whether a
simple model on ECG intervals is competitive with a foundation-model embedding, and reports the
gap.

### The effort estimate behind the choice

The PhysioNet route is estimated at 190 hours: 50 for data handling across three sources, 40 for
the interval-feature baseline, 55 for the embedding model and its fine-tuning, 25 for the
comparison and intervals, and 20 for the write-up.

The RSNA route is estimated at 340 hours: 80 for DICOM handling and resampling, 100 for the
3D pipeline, 60 for the multi-finding aggregation, 60 for compute management and debugging, and
40 for the write-up. The difference is 150 hours, which is more than a quarter of the team's
540-hour capacity.

At a blended 60 dollars per hour, the difference is 9,000 dollars of team time, before compute.
The PhysioNet route is not merely easier; it is the only route that leaves room for the
governance and validation artefacts the dossier requires.

### The decision meeting

The team records the decision in four lines: chosen option, rejected options, the constraint that
decided it, and the review date.

Chosen: PhysioNet 2025 released data, benchmarked against a published score, with a comparison of
a simple baseline and an embedding.
Rejected: RSNA 2026 knee, because compute and musculoskeletal expertise were insufficient and the
competition had closed; local datathon, because its extract had too few positives for the planned
interval.
Constraint: 58 positives in the local extract against a requirement of 81.
Review date: week 4, after the data audit.

The meeting takes forty minutes, and it produces the paragraph that appears in the charter.

Now vary one assumption: the team gains access to a GPU cluster with 2,000 GPU hours.
The RSNA option's compute score rises from 1 to 4, and its weighted total rises to 3.05. It is
still below the PhysioNet option, because skill and time remain low, and because the competition
has closed. The variation shows that compute alone does not make a competition feasible.

### What the team would do with a ranking

Suppose the team had entered the 2025 challenge and finished mid-table. The useful response is not
to describe the rank but to analyse the gap.

It would compute the score on the public validation split, compare it with the top score's
published approach, and identify whether the gap comes from the model, the features, or the
preprocessing. It would then report the gap and its likely cause, which is a methods finding
rather than a competitive one.

A ranking without a gap analysis teaches nothing. A gap analysis without a ranking is a
benchmarking exercise, which is often the more valuable of the two.

### What the team reports

The team writes the paragraph before it writes the slides, because the paragraph constrains what
the slides can claim. If the paragraph says "we do not claim a difference", the slide cannot say
"our model outperforms".

One paragraph in the dossier:

"We benchmarked a logistic model on standard ECG intervals against a foundation-model embedding
on the released PhysioNet 2025 Chagas data. On the public validation split, the logistic model
captured 0.24 of true cases in the top 5 percent, against 0.31 for the embedding. The gap is
within the interval we can estimate from 37 positives, so we do not claim a difference. The
result does not establish performance on the hidden test sites, and it does not support a clinical
claim."

That paragraph is more useful than a rank, and it is the kind of statement a panel rewards.

### The data audit that followed

In week 2 the team audited the released data and found three things that changed the plan.

The three training sources had different sampling rates and different lead configurations. The
team had assumed a common 500 Hz sampling rate; CODE-15% and PTB-XL differed, and two records in
the validation split had a missing lead. The preprocessing pipeline was rebuilt to resample and to
flag missing leads, which took eleven hours and would have invalidated the comparison if it had
been discovered after the results.

The label definitions differed by source. CODE-15% labels were weak, self-reported, and
serology-untested; SaMi-Trop labels were serologically confirmed. The team had planned to pool
them; after the audit it trained on the confirmed subset and used the weak labels only for
pretraining, which reduced the effective sample but made the evaluation interpretable.

The hidden test sites were not described in detail, which meant the team could not stratify its
own validation by site. It reported a random-split estimate and stated that it could not estimate
site generalisation. That statement is a limitation, and it is more valuable than a number that
implies transportability the data cannot support.

### What the panel asked

Three questions at the midpoint.

"Why this competition rather than a local dataset?" The team cited the capacity-constrained metric,
the multi-source data, and the published results that allowed a benchmark comparison.

"What does your score mean for a clinic?" The team said it does not mean anything for a clinic,
because the cohort is not the clinic's population and the operating capacity is assumed. It means
that a simple model captures a fraction of cases at a fixed capacity on this cohort.

"What would you need to make a clinical claim?" The team listed a local cohort with confirmed
labels, a temporal split, calibration, and a reader-workflow study. The panel accepted the answer
because it was specific and because the team had not claimed otherwise.

## 7. Code, query, or calculation

The first block computes the capacity-constrained score that the PhysioNet 2025 challenge used,
alongside precision at the same capacity, to show why the two are different. The second scores
three competition options against team constraints.

```python
import random

# A synthetic cohort: 1,000 cases, 37 positive (3.7% prevalence)
random.seed(3)
labels = [1] * 37 + [0] * (1000 - 37)
scores = [random.gauss(1.2 if y else 0.0, 1.0) for y in labels]

ranked = sorted(zip(scores, labels), reverse=True)
k = int(0.05 * len(ranked))
top = ranked[:k]
positives = sum(labels)
tpr = sum(y for _, y in top) / positives
precision = sum(y for _, y in top) / k

print(f"cohort {len(ranked)}, positives {positives}, k {k}")
print(f"TPR in top 5%: {tpr:.3f}")
print(f"precision in top 5%: {precision:.3f}")
print(f"random TPR baseline: {0.05:.3f}")
```

Expected output:

```text
cohort 1000, positives 37, k 50
TPR in top 5%: 0.243
precision in top 5%: 0.180
random TPR baseline: 0.050
```

The TPR is the challenge score; the precision is a different quantity that some team papers
report. With 37 positives, a TPR of 0.243 means 9 cases captured, and the 95 percent interval for
9 of 37 is wide, roughly 0.12 to 0.41 by the Wilson method. A leaderboard built on so few
positives ranks teams noisily, which is why the organisers used a larger hidden cohort.

```python
criteria = {"data": 0.25, "compute": 0.20, "metric": 0.20,
            "skill": 0.15, "time": 0.20}
options = {
    "PhysioNet 2025": (5, 4, 5, 3, 4),
    "RSNA 2026 knee": (3, 1, 4, 2, 2),
    "Local datathon": (5, 5, 4, 4, 5),
}
for name, scores in options.items():
    total = sum(s * w for s, w in zip(scores, criteria.values()))
    print(f"{name}: {total:.2f}")
```

Expected output:

```text
PhysioNet 2025: 4.30
RSNA 2026 knee: 2.45
Local datathon: 4.65
```

The scores are judgements. The useful part is the argument in the justification column, not the
second decimal place.

## 8. Data, standards, and regulatory context

Competitions impose their own governance, layered on top of the data licences.
The layers are: the challenge rules, the dataset licence, the platform terms, and the publication
policy. A team must satisfy all four, and they can conflict.

### Challenge rules

Challenge rules govern eligibility, team composition, external data, pretrained weights, code
release, and the number of submissions. They are contractual: accepting the rules is a condition
of participation.
The rules also define what a score means. A challenge that permits external data produces scores
that are not comparable to one that prohibits it, even if the leaderboard does not separate them.

### Platform terms

Kaggle, Synapse, and Codabench each impose platform terms in addition to the challenge rules.
Kaggle code competitions require notebook submission, may disable internet access, and cap
runtime. Synapse requires registration and, for some data, a data-use agreement. Codabench is
often used for academic challenges with open submission.

### Dataset licences

The underlying dataset may have a licence that differs from the challenge rules. RSNA challenge
data are distributed under the competition's terms, and post-competition use is governed by the
dataset licence, which must be read separately. PhysioNet data are open or credentialed, and the
credentialed sets require CITI training and a data-use agreement per dataset.
CAGI data are provided for assessment, with held-out truth that is not distributed.

### Reporting and publication

Many challenges publish a proceedings paper or a special issue, and some require a code release or
a method description. The reporting norms are those of the field: STARD-AI and TRIPOD+AI for
diagnostic and prediction studies, MICCAI-style method papers for segmentation, and benchmark
reporting for evaluation studies.
A team that publishes a challenge result without the split, metric, and rank is not reporting
enough for someone else to interpret the number.

### Regulatory status

A challenge result is not a regulatory artefact. It does not establish intended use, clinical
validity, or safety, and it cannot support a device claim on its own.
A team that later pursues a device pathway must generate the evidence that the pathway requires:
analytical validation, clinical validation, and a risk analysis. The challenge may contribute a
method and a benchmark, and nothing more.

| Layer | Example | Binding | What it constrains |
|---|---|---|---|
| Challenge rules | RSNA, PhysioNet, CAGI | Contractual | Eligibility, data, code, submissions |
| Platform terms | Kaggle, Synapse, Codabench | Contractual | Submission format, access |
| Dataset licence | CC-BY, research-only, DUA | Legal or contractual | Redistribution, commercial use |
| Publication policy | Proceedings, special issue | Contractual | Code and method disclosure |
| Device regulation | FDA, EU MDR | Legal, if marketing | Claims and evidence |

### What a competition cannot supply

Four artefacts that a dossier requires cannot come from a challenge.

A data provenance record with the licence and the access date. The challenge provides data under
its own terms, and the team must record them separately.
An intended-use statement. A challenge task is not an intended use, and the user is not specified.
A calibration and subgroup analysis. The challenge metric does not require either, and the private
split is not available for stratified analysis.
A monitoring plan. A challenge evaluates once, and a deployed model is evaluated continuously.

A team that uses a challenge as its data source must produce these four artefacts itself. That is
the difference between a competition entry and a capstone, and it is the difference a panel scores.

## 9. Industry practice and case evidence

### Imaging: RSNA

The RSNA 2024 Lumbar Spine Degenerative Classification challenge ran from May to October 2024 and
was announced in November 2024. Organisers reported 1,874 teams, 14,660 entrants, and 23,203
submissions, with a 50,000 dollar prize pool shared by nine teams. The first-place solution used a
two-stage pipeline: vertebral level localisation, then region-of-interest classification with a
bi-LSTM and attention-based multiple-instance model.

The RSNA 2025 Intracranial Aneurysm Detection challenge ran from July to October 2025 and was
announced in November 2025. Organisers reported 1,147 teams, a reference dataset from 18 sites on
five continents annotated by more than 60 radiologists, and nine teams sharing 50,000 dollars. The
first-place solution used three nnU-Net vessel-segmentation models followed by a location-aware
3D ROI classifier, reporting an AUC of 0.915 for aneurysm presence and 0.916 across 13 locations.

The RSNA 2026 Knee Abnormality Detection challenge was open through October 2026, with more than
5,000 MRI exams from 19 sites, reports in about 12 languages, 12 findings, a macro ROC AUC metric,
and a 77,000 dollar prize pool including an 18,000 dollar efficiency track. The submission window
closed on 22 October 2026; results were pending at the source cutoff, and winners were scheduled
to be recognised at RSNA 2026.

What the three share: a clinically meaningful task, a large multi-site dataset, a metric that
aggregates across findings, and a solution pattern that localises before classifying.
What they do not share: clinical validation. A leaderboard AUC is not a reader study.

### Signals: PhysioNet

The George B. Moody PhysioNet Challenge 2025 asked for Chagas disease detection from 12-lead ECGs.
The organisers report 111 teams, 630 participants, 1,317 code submissions, and 41 ranked teams. The
score is the fraction of Chagas patients captured in the top 5 percent of the ranking, a
capacity-constrained metric. The top test score was 0.323; the same models scored about 0.100 on
an external cohort from a largely asymptomatic population.

The 2026 Challenge moves to cognitive impairment screening from polysomnography, using recordings
from the Human Sleep Project and an age-conditioned AUROC with a two-year window. The data were
released and the task is open; results are not asserted here. The design is notable for removing a
demographic shortcut by conditioning the metric on age.

The 2025 result is the most instructive single number in this topic. The gap between 0.323 and
0.100 is not a modelling failure; it is a population difference, and it is the reason a
capacity-constrained metric on a hidden test set is calibrating evidence rather than clinical
evidence.

### What the 2026 edition changes

The 2026 PhysioNet Challenge asks teams to predict a later diagnosis of cognitive impairment from
polysomnography, with basic demographics and automated sleep annotations as inputs. The label
requires at least two cognitive-impairment diagnoses in a defined window after the sleep study,
and the metric is an age-conditioned AUROC with a two-year window.

Two design choices are worth studying.

Age conditioning removes a shortcut. Older patients are more likely to receive a diagnosis and
more likely to have a sleep study, so an unconditioned model could rank well by age alone. The
conditioned metric compares patients of similar age, which forces the model to use signal.

The hidden sites include deliberate missingness, such as absent channels and absent annotation
files. That tests robustness to the conditions a deployed model actually meets, rather than the
clean conditions of a curated dataset.

Both choices are architectural decisions that a capstone can copy without copying the task. A
team designing a temporal validation can condition its metric on a confounder, and a team
designing a robustness analysis can remove channels and measure the degradation.

### MICCAI and the challenge ecosystem

MICCAI 2026 lists 45 challenges, running around the conference in Strasbourg from 27 September to
1 October 2026. The list spans segmentation, registration, foundation models, vision-language
tasks, and agentic systems, with 25 recurring and 20 new challenges.

The size of the list is itself a finding. A team cannot survey 45 challenges; it must filter by
task, data regime, and metric, which is why the evaluation procedure in Section 5 starts with the
contract rather than the topic. The same filter applies to the recurring challenges, whose rules
and metrics change between editions often enough that a prior year's approach may not transfer.

### Inference, labelled

### Omics and single-cell

CAGI7 ran fourteen challenges in 2025, with a conference in Boston on 6 to 8 December 2025. The
PRS challenge provided summary statistics and simulated genotypes, held out validation genotypes,
and accepted containers that ran locally because the training genotypes were not shared. The
challenge acknowledged a European-ancestry limitation in the real data.

The SEA-AD DREAM Challenge, reported in 2026, asked teams to predict Alzheimer's disease
neuropathology from single-nucleus transcriptomics. The organisers report 17 eligible teams from
15 countries, a best quadratic weighted kappa of 1.0 for overall AD neuropathological change, and
a best concordance correlation coefficient of 0.48 for quantitative amyloid and tau burden. The
post-challenge analysis found that top categorical predictions leaned on donor-level metadata
rather than transcriptomic signal, which is a leakage finding of exactly the kind a leaderboard
cannot show.

The finding deserves emphasis because it is the strongest argument for reading analyses rather than
leaderboards. A model that predicts a donor-level attribute from metadata can achieve near-perfect
categorical staging without using the transcriptomics the task claims to test. The leaderboard
rewards it; only an analysis that asks what the model used can detect it. A capstone team should
run the same check on its own model: which features carry the performance, and are they available
at decision time?

The Open Problems in Single-Cell Analysis benchmarks, reported in Nature Biotechnology in 2025,
found that a logistic regression on log-normalised principal components was the best method in
five of eight label-projection datasets. The result is a reminder that a simple baseline is a
serious competitor in a well-controlled benchmark.

### Language and agents

The NeurIPS 2025 competition track included CURE-Bench on drug decision-making, SLC-PFM on
pathology foundation models with roughly 300 million images across 39 cancer types, and the EEG
Foundation Challenge on cross-subject decoding from the HBN-EEG dataset. The EEG challenge
reported 1,183 teams and more than 8,000 submissions on Codabench; results were published on the
challenge site and at the conference.

The MedGemma Impact Challenge, launched in January 2026 and concluded with winners in March 2026,
asked for prototypes on open-weight health models in low-resource settings. The winning projects
were narrow prototypes, not validated products, and the competition did not test clinical
outcomes.

### The landscape at a glance

The table records what the official sources support at the access date. Blank cells are
deliberate: a fact that could not be verified is not asserted.

| Event | Edition | Task | Status | Verified fact |
|---|---|---|---|---|
| PhysioNet/CinC | 2025 | Chagas disease from ECG | Concluded | 111 teams; 41 ranked; TPR-at-5% metric |
| PhysioNet/CinC | 2026 | Cognitive impairment from PSG | Open, results not asserted | Age-conditioned AUROC; hidden sites |
| RSNA | 2024 | Lumbar spine degeneration | Concluded | 1,874 teams; two-stage winning pipeline |
| RSNA | 2025 | Intracranial aneurysm | Concluded | 1,147 teams; 18 sites; winner AUC 0.915 |
| RSNA | 2026 | Knee abnormalities, 12 findings | Closed 22 Oct 2026, results pending | 19 sites; macro ROC AUC; 77,000 USD pool |
| MICCAI | 2026 | 45 challenges, Strasbourg | Scheduled 27 Sep to 1 Oct 2026 | Challenge list published |
| NeurIPS | 2025 | CURE-Bench, SLC-PFM, EEG Foundation | Concluded | EEG: 1,183 teams on Codabench |
| DREAM | 2025 | SEA-AD and others | Concluded; analysis 2026 | 17 teams from 15 countries; metadata finding |
| CAGI | 7 | 14 challenges | Concluded Dec 2025 | PRS challenge European-ancestry limitation |
| Open Problems | Ongoing | Single-cell benchmarks | Recurring | Logistic regression best in 5 of 8 datasets |
| MedGemma Impact | 2026 | Prototypes on open-weight models | Concluded March 2026 | Winners reported as prototypes |

The table is a reading aid, not a substitute for the official pages. Statuses change, and a team
that plans around a table rather than a page will eventually be wrong.

### What winning solutions share

Across imaging, signals, and omics, five patterns recur in the published write-ups.

Localisation before classification. Three-dimensional tasks are easier when the anatomical
structure is found first and the classifier sees a cropped region.
Preprocessing that matches the acquisition. Resampling, intensity normalisation, and montage
handling matter more than architecture in many challenges.
Label-noise handling. Weak labels, text-derived labels, and multi-rater disagreement are handled
with soft labels, loss functions that tolerate noise, or adjudication subsets.
Simple baselines. A logistic or gradient-boosted model on well-chosen features is often within a
few points of the winner, and it is the sanity check that the complex model is worth its cost.
Ensembling and test-time augmentation. The top of most leaderboards is an ensemble, and the gain
is usually a fraction of a point for a large increase in compute.

What the write-ups rarely show is calibration, subgroup performance, or external validation. Those
are the capstone's job, and a team that imports the winning method without importing the missing
evidence has imported the wrong half.

### The pattern that is not in the write-ups

Across every challenge described here, the winning team had more compute, more attempts, and more
people than the median entrant. That is not a criticism; it is a description of the competition's
economics. The pattern is rarely stated because it is not a method, but it explains why a small
team should benchmark rather than attempt to win, and why the useful transfer is the preprocessing
and validation discipline rather than the ensemble.

A team that accepts this can use challenges well: as a source of tasks, data, and method patterns,
and as a calibration of what strong performance looks like, without treating the leaderboard as a
target it must hit.

The calibration is the part that transfers to a capstone defense. When a panel asks whether a
sensitivity of 0.72 is good, the honest answer is "against what comparator, on what population, at
what capacity". A person who has read three challenge contracts can answer that question with
numbers, because the contracts make the comparator and the capacity explicit.

### Inference, labelled

It is an inference that reading a post-challenge analysis is more useful than reading the
leaderboard. The Sea-AD metadata finding supports the claim for that challenge; generalising it to
all challenges is plausible and untested.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Announced treated as concluded | No results page, only a save-the-date | Status not verified | Plans built on nothing | Check official page and date |
| Metric misread | Team reports AUC for a capacity metric | Familiarity | Wrong model selection | Read the scoring code |
| Rank treated as clinical evidence | Slide says "state of the art" | Leaderboard framing | Overclaim | State what the score does not show |
| Leaderboard probing | Daily submissions near the cap | Test-set fitting | Private-split collapse | Cap and log submissions |
| Contamination | Unusually high public score | Test data in pretraining | Invalid result | Contamination screen |
| External data undisclosed | Score jump with no method change | Rule breach or category error | Disqualification or invalid comparison | Read and record the rule category |
| Compute underestimated | Training queue exceeds plan | 3D data and ensembles | No competitive entry | Estimate GPU hours first |
| Team count as difficulty | "1,800 teams, so it is hard" | Marketing proxy | Wrong calibration | Read the score distribution |
| Winner's method copied wholesale | Ensemble of five models on one GPU | Resource mismatch | Unreproducible entry | Extract the pattern, not the recipe |
| Post-hoc metric selection | Metric changed after results | Rationalisation | Invalid comparison | Pre-specify the metric |
| Test set reused | Released test set tuned against | Silent overfitting | Inflated result | Hold out a local split |
| Licence ignored | Data in a public repository | Convenience | Breach | Record licence and obligations |
| Unverified 2026 claims | Specific winner named without source | Fabrication | Credibility loss | Describe the category, not the result |

The last row is the one this topic is most exposed to. A landscape that is described a year after
the events will contain gaps, and the correct response to a gap is to describe the category: a
recurring challenge with an annual cycle, a task type, and a metric family, without asserting a
winner or a score that the source does not support.

The first row is the most common. A team that plans around an announced event and discovers that
it was postponed has lost the planning time, and the loss is avoidable with one page check.

The table is used prospectively, not retrospectively. A team reads the rows before it commits, and
each row it cannot answer is a question for the organisers or a reason to choose a different
competition.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Official pages are current | Status depends on them | Pages lag announcements | Check date and results section |
| Team counts indicate participation | Calibration | Bots and duplicate accounts | Treat as approximate |
| Scores are comparable within an edition | Ranking logic | Rule categories differ | Read rule categories |
| The hidden split is uncontaminated | Evidence depends on it | Public data in pretraining | Contamination screen |
| A winning write-up describes the method | Learning depends on it | Write-ups omit details | Check code release |
| Compute estimates are achievable | Entry feasibility | Shared cluster queues | Measure a pilot run |
| The metric matches a decision | Transfer depends on it | Benchmark task only | Map task to action |
| Challenge data can be used afterwards | Capstone depends on it | Licence restricts use | Read the licence |

The largest limit is that this topic describes a moving landscape. Events open, close, and publish
results on their own schedule, and any statement here is a snapshot at 26 October 2026. The
categories and the evaluation procedure are stable; the specific statuses are not.

A second limit is that published winning solutions are self-reports. They describe what the team
believes it did, often after the fact, and they omit the failed experiments that consumed most of
the time. Reading them as a recipe overstates the method's contribution and understates the
engineering and the luck.

A third limit is that the transferable patterns are conditional. Localisation before
classification helps when the label is attached to an anatomical object; it does not help a
tabular prediction task. Ensembling helps at the top of a leaderboard; it rarely helps a capstone
that needs a reproducible pipeline and an interpretable model.

A fourth limit is that challenge data are not a population. A team that builds a capstone on a
challenge dataset has a cohort chosen by organisers, filtered by inclusion criteria the team did
not write, and labelled by a process the team did not audit. The dataset may be excellent and it
is still not the deployment population, which is why the intended-use statement must describe the
dataset's population rather than the clinic's.

## 12. Exercises

### Exercise 1. Classify five events (recall, 30 minutes)

Take five events from this topic and classify each by status, task type, metric family, and data
regime using the official pages.
**Deliverable:** a five-row table with URLs and access dates.
**Criteria:** every status verified; no event labelled concluded without a results page.
**Time:** 30 minutes.

### Exercise 2. Compute a capacity metric (calculation, 35 minutes)

Using the synthetic cohort in Section 7, compute TPR and precision at the top 5 percent for a
model that scores positives with mean 0.8 and negatives with mean 0.0, both standard deviation 1.
**Deliverable:** the numbers and a one-sentence interpretation.
**Criteria:** correct ranking and division; the difference between the two metrics explained.
**Time:** 35 minutes.

### Exercise 3. Evaluate an option (application, 45 minutes)

Score three competitions against your team's constraints using the criteria in Section 7. Write a
justification for each score.
**Deliverable:** the table and a decision memo.
**Criteria:** scores tied to evidence; the weakest criterion named; the decision follows.
**Time:** 45 minutes.

### Exercise 4. Read a winning write-up (critique, 40 minutes)

Choose a published winning solution and identify the method pattern, the resource dependency, and
one thing the write-up does not report.
**Deliverable:** a one-page analysis.
**Criteria:** pattern stated conditionally; resource dependency quantified; missing evidence named
(calibration, subgroups, external validation).
**Time:** 40 minutes.

### Exercise 5. Design a contamination screen (design, 30 minutes)

For a model that uses a pretrained image backbone, list the checks that would detect test-set
contamination and the evidence each would produce.
**Deliverable:** a checklist.
**Criteria:** at least four checks; each has an observable outcome; the limits of the screen stated.
**Time:** 30 minutes.

### Exercise 6. Write the honest paragraph (application, 25 minutes)

Write the one-paragraph report for a competition result, including metric, split, score, and
limitations.
**Deliverable:** the paragraph.
**Criteria:** no claim beyond the metric; the split named; the limitation specific.
**Time:** 25 minutes.

### Exercise 7. Compare two editions (critique, 35 minutes)

Compare the 2024 and 2025 RSNA challenges on task, data, metric, and solution pattern. Explain why
a score from one is not comparable to a score from the other.
**Deliverable:** a comparison table and 200 words.
**Criteria:** differences are specific; the comparability argument is correct; no invented scores.
**Time:** 35 minutes.

### Exercise 8. Decide whether to enter (design, 30 minutes)

For a competition of your choice, apply the decision table and justify enter, benchmark, or
observe.
**Deliverable:** a decision record.
**Criteria:** each condition addressed; the opportunity cost stated in hours; the decision is
consistent with the evidence.
**Time:** 30 minutes.

**Assessment note.** Exercises 1 and 2 have checkable answers. Exercises 3 to 8 are assessed on
the quality of the record: whether the evidence is cited, whether the constraint is named, and
whether the decision follows. A team that chooses a modest competition with a clear argument
should score higher than one that chooses a prestigious competition it cannot enter.

The exercise set is also a rehearsal for the dossier's competition section. A team that has
completed Exercise 6 has already written the paragraph it will use, and a team that has completed
Exercise 4 has the method analysis that the related-work section needs.

## 13. Self-check answers

**1. Why is a capacity-constrained metric different from AUROC?**
AUROC measures ranking across all thresholds. A capacity-constrained metric measures how many true
cases are captured when only a fixed fraction can be acted on. The second depends on the cohort
size and the operating capacity; the first does not.

**2. Compute TPR at the top 5 percent for 37 positives and 9 captured.**
9 / 37 = 0.243. The precision at the same capacity is 9 / 50 = 0.18, because the top 5 percent of
1,000 cases is 50 cases.

**3. Why did the PhysioNet 2025 top score fall from 0.323 to about 0.100 on an external cohort?**
Because the external cohort was a different population, largely asymptomatic, with a different
prevalence and case mix. The model ranked that population less well, which is a transportability
finding rather than a coding error.

**4. What does a private-leaderboard rank measure?**
Performance on a held-out subset under the same task, metric, and rules. It does not measure
clinical utility, calibration, subgroup performance, or performance at another site.

**5. Why is a team count a poor difficulty measure?**
Because it reflects prize money, publicity, and platform defaults as much as task difficulty. A
large challenge may have a wide score distribution with many casual entries.

**6. What is the difference between external data and pretrained weights under competition rules?**
External data are additional training examples; pretrained weights are a model trained elsewhere.
Rules often treat them differently, and a score achieved with either may be in a separate category.

**7. Why did the SEA-AD analysis matter more than the leaderboard?**
Because it showed that top categorical predictions relied on donor-level metadata rather than
transcriptomic signal. A leaderboard cannot reveal that; a post-challenge analysis can.

**8. What are three patterns that recur in winning solutions?**
Localisation before classification, preprocessing matched to acquisition, and ensembling with
test-time augmentation. Label-noise handling is a fourth.

**9. When should a team benchmark rather than enter?**
When the data are accessible but the compute or the time is insufficient for a competitive entry.
Benchmarking still produces a comparison and a learning artefact.

**10. Rubric for Exercise 7.** A strong answer identifies the task difference (localisation and
grading versus detection across locations), the data difference (sites, annotations), and the
metric difference, and explains that scores on different tasks and metrics are not comparable. A
weak answer compares the numbers directly.

**11. Rubric for Exercise 3.** A strong evaluation names the binding constraint for each option and
shows the arithmetic behind the weakest score. A weak evaluation assigns scores without
justification and picks the option the team already preferred.

**12. Why should a team record the ranked-entry count rather than the registration count?**
Because registrations include teams that never submit, and a rank against a denominator of
registrations overstates the achievement. The ranked count is the population the team actually
competed against.

**13. What is the difference between a method pattern and a recipe?**
A pattern is a design choice with a condition, such as "localise before classifying when the label
belongs to an anatomical object". A recipe is a sequence of steps tied to a resource level, such as
"train three nnU-Net variants and ensemble". The pattern transfers; the recipe usually does not.

## 14. Interview questions

**Q1. "Which competition results do you follow?"**
Strong answer names two or three with year and task, and says what the results show and do not.
Weak answer names a leaderboard rank.

**Q2. "What did you learn from a winning solution?"**
Strong answer names a conditional pattern and its resource dependency. Weak answer lists
architectures.

**Q3. "Why is a leaderboard not clinical evidence?"**
Strong answer explains the hidden split, the metric, and the missing calibration and subgroup
evidence. Weak answer says the data are different.

**Q4. "How would you evaluate whether to enter a challenge?"**
Strong answer uses data access, compute, metric fit, skill, and time, and names the opportunity
cost. Weak answer says whether the task is interesting.

**Q5. "What is contamination, and how do you check for it?"**
Strong answer describes pretraining overlap, deduplication, and a documented-corpus baseline. Weak
answer says it is when the test set leaks.

**Q6. "How do you report a competition result honestly?"**
Strong answer gives metric, split, score, rank, and limitations in one paragraph. Weak answer
quotes a percentile.

**Q7. "What does the Chagas result tell you about deployment?"**
Strong answer explains prevalence shift and transportability, citing the drop on the external
cohort. Weak answer says the model did not generalise, without a mechanism.

**Q8. "Why do winning solutions ensemble?"**
Strong answer explains variance reduction and the small marginal gain relative to compute. Weak
answer says ensembles are better.

**Q9. "What would you do differently from the winning team?"**
Strong answer names calibration, subgroup analysis, or external validation. Weak answer says use a
bigger model.

**Q10. "How do you keep up with the landscape?"**
Strong answer describes following organiser pages and post-challenge analyses, and re-checking
status. Weak answer says Twitter or newsletters only.

**Q11. "What would make you distrust a leaderboard result?"**
Strong answer names contamination, a metric that does not match the decision, a private split with
few positives, and a rule category that permits external data. Weak answer says a low score.

A strong answer to this set is always specific about the edition, the metric, and the split. A
candidate who says "the 2025 challenge" without the year, the task, and the metric is reciting a
headline rather than reading a result.

## 15. Key terms

- **Benchmark harness** — code that runs and scores submissions.
- **Capacity-constrained metric** — metric at a fixed operating capacity.
- **Challenge** — time-boxed competition with a fixed contract.
- **Code competition** — format with notebook submission and runtime caps.
- **Contamination** — test data in pretraining or selection.
- **Efficiency track** — sub-track scoring performance per compute.
- **External data** — data not provided by organisers.
- **Hidden site** — data source withheld entirely.
- **Leaderboard probing** — many submissions to infer the test set.
- **Post-challenge analysis** — organiser or community analysis of results.
- **Precision at k** — fraction of top k that are positive.
- **Pretrained weights** — model trained before the competition.
- **Prevalence shift** — different outcome frequency in deployment.
- **Private leaderboard** — scores on held-out data.
- **Public leaderboard** — scores on visible data.
- **Shake-up** — ranking change between public and private.
- **Status** — concluded, closed, open, announced, recurring, unverified.
- **TPR at k** — fraction of positives captured in top k.
- **Track** — sub-competition with its own task.
- **Winning solution write-up** — team's description after the event.

## 16. Sources

All URLs accessed 26 October 2026.

- PhysioNet/CinC Challenge 2025, Chagas disease. https://moody-challenge.physionet.org/2025/
- Reyna MA, Koscova Z, Pavlus J, et al. Detection of Chagas disease from the ECG: the George B.
  Moody PhysioNet Challenge 2025. arXiv:2510.02202. https://arxiv.org/abs/2510.02202
- PhysioNet/CinC Challenge 2026, cognitive impairment from polysomnography.
  https://moody-challenge.physionet.org/2026/
- RSNA, Lumbar Spine Degenerative Classification AI Challenge, 2024.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge
- RSNA, Intracranial Aneurysm Detection AI Challenge, 2025.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/intracranial-aneurysm-detection-ai-challenge
- RSNA, Knee MRI AI Challenge, 2026.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- MICCAI 2026 challenges. https://conferences.miccai.org/2026/en/challenges.asp
- MICCAI 2025 FLARE challenge.
  https://conferences.miccai.org/2025/en/FLARE-2025-Challenge.html
- CAGI7 challenges. https://genomeinterpretation.org/challenges.html
- CAGI7 PRS challenge. https://genomeinterpretation.org/cagi7-prs.html
- SEA-AD DREAM Challenge. https://www.synapse.org/Synapse:syn66496696
- Lai et al. The SEA-AD DREAM Challenge. bioRxiv 2026.
  https://www.biorxiv.org/content/10.64898/2026.07.02.736180v1
- Luecken MD, Gigante S, Burkhardt DB, et al. Defining and benchmarking open problems in
  single-cell analysis. *Nature Biotechnology* 2025.
  https://doi.org/10.1038/s41587-025-02694-w
- NeurIPS 2025 competitions. https://neurips.cc/virtual/2025/loc/san-diego/events/Competition
- EEG Foundation Challenge. https://arxiv.org/abs/2506.19141
- Google Research, MedGemma Impact Challenge winners, March 2026.
  https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- RSNA, Intracranial Aneurysm Detection results announcement, November 2025.
  https://www.rsna.org/media/press/2025/2629

## 17. Links

The competition statuses in this topic are a snapshot at the access date. Re-check the official
page before assigning an event to a team; the survey file is the module map, not a substitute for
the organiser's site. Where a status could not be verified, this topic describes the category and
says so, which is the only defensible option for a teaching text.

A final note on the relationship between this topic and the capstone: the landscape is a reference,
not a curriculum. No team should choose a capstone because a competition exists; a competition is
useful when its data and metric happen to fit a decision the team can own.

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_hackathon-formats.md`](06_hackathon-formats.md)
- Next topic: [`08_scoping-playbook.md`](08_scoping-playbook.md)
- Related topic: [`04_capstone-archetypes.md`](04_capstone-archetypes.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
