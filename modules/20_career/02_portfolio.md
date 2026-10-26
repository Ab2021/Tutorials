# The Portfolio: Evidence That Survives a Ten-Minute Review

> Module 20 — Career, Portfolio, and Interviews
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A portfolio is not a showcase. It is a set of artifacts that lets a stranger verify three claims
about you in the order they care about them: that you can frame a decision, that you can produce
trustworthy evidence for it, and that you can hand the work to someone else without disappearing.
Most portfolios answer a different question, "can this person build a model," and are therefore
read as evidence of technical interest rather than of judgment.

The review is short. A hiring manager who receives 80 applications spends 20 to 40 seconds on
each resume, then opens links for perhaps five candidates. On a portfolio site they read the
project title, the first paragraph, one figure, and the limitations section. If the repository is
linked, they look at the README, the commit history, and whether the code runs. That is the
entire review, and it happens before any interview. A portfolio that requires 30 minutes to
understand has already failed, because no one will spend 30 minutes.

### 1.1 The failure that looks like success

The most common portfolio failure is a technically impressive project with no decision attached.
A candidate fine-tunes a language model on clinical notes, reports an F1 score, publishes a
notebook, and receives no callbacks. The work is real. The problem is that the artifact answers
"can a model learn this task," while the hiring manager is asking "would this person be safe to
put in front of a clinical team." The second question requires an intended use, a validation
design, a failure analysis, and a statement of what the model must not be used for. None of
those are in the notebook.

The second most common failure is the opposite: a well-written case study with no reproducible
artifact. The narrative claims a temporal validation and a calibration plot, but the repository
has no data dictionary, no environment file, and a hard-coded path to the author's laptop. The
reader cannot verify the claim, and an unverifiable claim is worse than a modest one because it
invites doubt about everything else.

### 1.2 Who this blocks

Candidates from academia have artifacts (papers, code, datasets) but present them as
publications rather than as decisions. A paper's contribution statement is written for reviewers
who will accept a methods advance; a portfolio case study must be written for a practitioner who
will ask what changed.

Candidates from industry have the reverse problem. Their best work is proprietary, and they
cannot show it. They either show nothing or show a sanitised version that has lost the details
that made it credible. The procedure in Section 5 addresses this with a disclosure pattern that
preserves credibility without exposing employer data.

Career changers with no health data experience have neither. They need public data, a defensible
question, and a scrupulous statement of what the data cannot support. That is achievable in 60
to 90 hours per artifact, which is the budget this topic assumes.

### 1.3 What a portfolio is competing against

Every candidate for a clinical data role can claim Python, structured query language, and a model. The scarce signals
are: a decision you can name, a validation you can defend, a limitation you volunteered, and a
repository a stranger can run. Those four are the portfolio's actual product. The model is the
vehicle.

> **Note:** The portfolio is not a substitute for the interview. It is the artifact that gets you
> the interview and the shared object you and the interviewer reason about. A strong portfolio
> makes the interview concrete; a weak one makes it generic.

### 1.4 Why five slots rather than one perfect project

One artifact can be excellent and still fail a panel. A clinical reviewer wants to see whether
you understand workflow; an engineer wants to see whether your code survives contact with
production; a manager wants to see whether you can write a decision memo. Those are three
different artifacts, and a single model repository answers only the second and part of the
first.

Five slots also protect against a change of target. If the clinical data science market is slow
in a given quarter, the causal artifact and the translation memo support a real-world evidence (RWE) application
without a rebuild. The portfolio is a small portfolio of options, not a monument.

### 1.5 What this topic covers

Five portfolio slots, the evidence each requires, the review sequence a hiring manager follows,
the standards that make an artifact credible (TRIPOD+AI for prediction models, model cards for
intended use, datasheets for data, reproducible analytical pipelines for code), and a worked case
that builds one artifact end to end. Section 6 is the longest part and the one to reproduce.

It does not cover resume writing (Topic 03), interview performance (Topics 04 to 08), or the
underlying methods, which are taught in Modules 05 to 15. Where a method matters, this topic
names the evidence standard rather than teaching the estimator.

### 1.6 The portfolio is a maintenance object, not a launch

A portfolio built once and abandoned degrades faster than most candidates expect. Package
versions break, data access expires, links rot, and the artifact that was state of the art in
2024 reads as dated in 2026. The review interval matters because the reviewer checks recency:
a repository with no commits for 18 months suggests the person has not shipped anything since.

The practical rule is one artifact per year, refreshed. Pick the artifact closest to your target
family, rerun it on current data or a current library version, update the limitation section,
and leave the rest alone. An hour a month is enough to keep links working and dependencies
pinned. This is cheaper than rebuilding before a search and it produces a visible history of
maintenance, which is itself a signal.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish a portfolio that demonstrates capability from one that demonstrates judgment | Exercise 1; self-check 1 |
| 2 | Select five artifacts that cover the decisions in a target role family | Exercise 2; self-check 3 |
| 3 | Write a one-page case study with decision, data, method, results, limitations, and intended use | Exercise 3; self-check 4 |
| 4 | Produce a model card and a datasheet for an artifact, including subgroup performance | Exercise 4; self-check 5 |
| 5 | Package a repository to the baseline level of a reproducible analytical pipeline | Exercise 5; self-check 6 |
| 6 | Write a disclosure note for proprietary work that preserves credibility without exposing data | Exercise 6; self-check 7 |
| 7 | Score a portfolio against a review rubric and identify the three weakest signals | Exercise 7; self-check 8 |
| 8 | Plan a 90-hour build for one artifact with a defensible scope | Exercise 8; interview question 9 |

Outcomes 3 to 5 are the ones that separate a portfolio that gets read from one that gets
skimmed. Outcome 7 is the calibration step: most candidates score their own portfolio two
levels higher than a stranger does.

The outcomes are assessed through artifacts, not statements. For outcome 3, the artifact is the
case study itself, and it is assessed by a reader who has not seen the code. For outcome 5, the
artifact is a reproduction report produced on a machine that has never run the project. For
outcome 6, the artifact is a disclosure note reviewed against the employer's publication policy.
If an outcome cannot be assessed from an artifact, it has not been achieved.

## 3. The core mental model

A portfolio is a chain of verification. Each link answers a question the reader would ask, and
the chain breaks at the first unanswered question.

```text
  CLAIM            "I can build and validate a readmission model that a care team
                    could act on."
      |
      v
  DECISION         Who acts, when, on what output, with what capacity?
      |
      v
  EVIDENCE         Temporal validation, calibration, decision curve, subgroups.
      |
      v
  ARTIFACT         Repository, model card, case study, data dictionary.
      |
      v
  VERIFICATION     A stranger can rerun the headline number in under 30 minutes.
```

The reader moves down the chain, not up. They start with the claim on the profile page, check
whether a decision is named, look for the evidence, open the artifact, and try to verify. If any
link is missing, the chain terminates there. A portfolio with five strong claims and one broken
verification link is read as one broken claim, because the reader generalises from the failure.

### 3.1 The three layers of an artifact

Every portfolio artifact has three layers, and candidates habitually over-invest in the top one.

| Layer | What it contains | Share of reader attention | Share of candidate effort (typical) |
|---|---|---:|---:|
| Decision | The question, the actor, the action, the cost of error | 50% | 10% |
| Evidence | Validation design, metrics, uncertainty, subgroups, limitations | 30% | 30% |
| Craft | Code quality, tests, packaging, documentation, figures | 20% | 60% |

The mismatch is the portfolio problem in one table. Readers spend half their attention on the
decision layer, and candidates spend most of their effort on code. Rebalancing the effort is the
single highest-return change most candidates can make.

### 3.2 The five slots

The survey file names three core artifacts. This topic expands them to five slots so that a
candidate can cover a family with more than one decision type. Three is the minimum viable
portfolio; five is the version that survives a panel with a clinician, an engineer, and a
manager.

| Slot | Artifact | Decision it evidences | Required evidence |
|---|---|---|---|
| 1 | Prediction or decision model | Whether to act on a risk score | Temporal or external validation, calibration, decision curve, subgroups, model card |
| 2 | Causal or evidence study | Whether an exposure or intervention changes an outcome | Estimand, DAG, confounding strategy, sensitivity analysis, negative control |
| 3 | Reproducible pipeline or service | Whether a result can be produced again, reliably | Tests, environment lock, runbook, monitoring or failure handling |
| 4 | Data quality or standards artifact | Whether data are fit for a downstream decision | Data dictionary, conformance checks, provenance, datasheet |
| 5 | Translation artifact | Whether a decision-maker can act on the work | Two-page memo, value case, workflow map, adoption or regulatory summary |

A candidate targeting a single family does not need all five. A clinical data scientist needs 1,
3, and 5. An RWE scientist needs 2, 4, and 5. An ML engineer needs 1, 3, and a production
incident write-up. A bioinformatician needs 3, 4, and a discovery narrative. The mapping is
given in Section 5, Step 3.

### 3.3 The ten-minute review, modelled

```text
  t=0:00   Profile or resume link. Title and one line of positioning.
  t=0:30   Project index. Three to five titles with one-line decisions.
  t=1:00   Case study. Skim headings: Decision, Data, Method, Results,
           Limitations, What I would do differently.
  t=3:00   One figure. Usually calibration or the primary effect with an interval.
  t=4:00   Model card or datasheet. Checks intended use and subgroup results.
  t=6:00   Repository. README, file tree, commit history, whether tests exist.
  t=8:00   Attempts to find the headline number in the outputs.
  t=10:00  Decision: interview, maybe, or no.
```

Design for this path. Put the decision in the first two sentences of every project. Put the
primary figure above the fold. Put the limitation in the case study, not in a footnote. Put the
reproduction command in the README's first screen. Everything else is supporting material.

### 3.4 What the portfolio is not

It is not a notebook dump, a Kaggle profile, a list of certificates, or a personal blog about
learning. Each of those can support a portfolio; none substitutes for an artifact that ties a
decision to evidence. The test is simple: can a reader name the decision after ten minutes, and
can they find the number that supports it?

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Artifact | A durable output that can be inspected: repository, model card, memo, dashboard | Used for a screenshot or a slide | Count |
| Case study | A one- to three-page narrative connecting decision, data, method, results, and limits | Written as a project diary | Pages |
| Model card | A short structured document describing a model's intended use, factors, metrics, and caveats | Treated as a substitute for validation | Document |
| Datasheet | Structured documentation of a dataset's motivation, composition, collection, and uses | Confused with a data dictionary | Document |
| Data dictionary | Variable-level definitions, types, units, and missingness | Omitted because "the code explains it" | Table |
| Provenance | The record of where data came from and what was done to it | Assumed to be the commit history | Lineage record |
| Reproducibility | The ability of an independent person to obtain the same result from the same inputs | Confused with "it ran on my machine" | Binary |
| Replicability | Same result from new data with the same design | Used interchangeably with reproducibility | Binary |
| RAP | Reproducible analytical pipeline; an automated, version-controlled, peer-reviewed analysis | Used for any script | Maturity level |
| Baseline model | The simplest defensible comparator | Skipped in favour of a complex model | Model |
| Temporal validation | Validation on data collected after the development period | Confused with random split | Design |
| External validation | Validation on a different site, system, or population | Claimed from a held-out year at the same site | Design |
| Calibration | Agreement between predicted risk and observed frequency | Reported as a single slope without a plot | Slope, intercept |
| Decision curve | Net benefit across threshold probabilities | Replaced by area under the receiver operating characteristic curve because it is easier | Net benefit |
| AUROC | Area under the receiver operating characteristic curve; rank discrimination | Reported alone as evidence of usefulness | 0–1 |
| AUPRC | Area under the precision-recall curve; sensitive to class imbalance | Compared across populations with different prevalence | 0–1 |
| Subgroup analysis | Performance reported by clinically or demographically meaningful group | Added as an afterthought with no pre-specification | Stratified metrics |
| Data leakage | Information in training that would not be available at prediction time | Reduced to "I used a pipeline" | Failure mode |
| TRIPOD+AI | Reporting guideline for clinical prediction models using regression or machine learning | Treated as a quality score | Checklist |
| PROBAST+AI | Risk-of-bias and applicability tool for prediction model studies | Confused with TRIPOD, which is reporting | Assessment |
| Model card | See above | — | Document |
| Intended use | The population, setting, user, and decision a model is designed for | Omitted, leaving the reader to guess | Statement |
| Out-of-scope use | Uses the developer explicitly does not support | Absent from most portfolios | Statement |
| License | The legal terms under which others may reuse the work | Omitted, which makes the work legally unusable | SPDX identifier |
| PHI | Protected health information | Included in screenshots by accident | Data class |
| De-identification | Removal or transformation of identifiers under a defined standard | Assumed to make data publicable | Process |
| Synthetic data | Data generated to mimic structure without corresponding to real people | Treated as equivalent to real data for validation claims | Data class |
| Effect estimate | A quantity describing a contrast, with an interval | Reported as a p-value only | Estimate and 95% CI |
| Estimand | The precise quantity a study estimates | Left implicit | Definition |
| Negative control | An outcome or exposure that should show no effect if the design is sound | Omitted from portfolio causal work | Design |
| E-value | Minimum strength of unmeasured confounding needed to explain away an effect | Quoted without its assumptions | Ratio |
| Runbook | Operational instructions for running and recovering a service | Confused with a README | Document |
| Index page | The single page listing artifacts with a one-line decision each | Replaced by a GitHub profile with no ordering | Page |
| Headline number | The one result a reader should remember | Buried in a results table | Number with interval |
| Contribution statement | A short note naming what you owned in a team artifact | Omitted, leaving credit ambiguous | Paragraph |
| Calibration slope | How much predicted risk must be stretched to match observed risk | Reported without the intercept | Slope, ideal 1.0 |
| Net benefit | The decision-analytic value of using a model at a threshold | Confused with accuracy | Utility units |
| Sensitivity analysis | Re-estimation under alternative assumptions | Omitted because the primary result is "robust" | Table |
| Access route | The documented path by which you obtained the data | Described as "downloaded" | Statement |

## 5. The mechanics, step by step

The procedure builds one artifact in 60 to 90 hours across six to ten weeks. Repeat it for each
slot you need. The steps are ordered so that the cheapest failures happen first.

### Step 1. Choose the decision, not the dataset

**Input:** a target family from Topic 01 and three candidate ideas.
**Output:** one decision statement and one named actor.
**Rationale:** the dataset is a constraint; the decision is the product. Starting from data
produces artifacts that answer questions no one asked, which is the most common portfolio defect.

The decision statement has the same form as in Topic 01:

```text
[Role] decides [choice] at [point in workflow] using [output], and the cost of a wrong
choice is [consequence].
```

Test the candidate ideas against three filters:

1. **Frequency.** Does the decision occur often enough to matter? A monthly committee decision
   supports a different artifact from a per-patient alert.
2. **Observability.** Can the outcome be measured in the data you can access, within the study
   window?
3. **Authority.** Is there a real role that could act, even if you have no access to them? If
   the only conceivable actor is "society," the artifact will read as academic.

### Step 2. Write the claim as a falsifiable sentence

**Input:** the decision statement.
**Output:** one sentence with a metric and a comparator.
**Rationale:** the claim determines the validation design. A vague claim ("improve risk
prediction") cannot fail, and an artifact that cannot fail is not evidence.

Examples:

| Weak claim | Falsifiable claim |
|---|---|
| "Built a model to predict readmission" | "A gradient-boosted model using discharge-time features improves AUPRC from 0.21 to 0.29 over the existing LACE index at 30 days, with calibration slope between 0.9 and 1.1." |
| "Studied the effect of a drug" | "Among adults with type 2 diabetes initiating treatment, the target estimand is the 24-month hazard ratio for the primary cardiovascular outcome under adherence-adjusted confounding control." |
| "Made a pipeline" | "The pipeline reproduces the published cohort count of 41,882 from raw extracts in under 20 minutes on a laptop, with row-level checksums." |

A falsifiable claim tells the reader what figure to look for, which is exactly what the
ten-minute review needs.

### Step 3. Select the slot and the minimum evidence set

**Input:** the target family and the claim.
**Output:** the slot number and a checklist of required evidence.
**Rationale:** each slot has a different evidence standard. Producing a model without a model
card, or a causal study without a sensitivity analysis, signals that you do not know what the
field considers sufficient.

| If your target is | Build these slots | Because the interviewer will probe |
|---|---|---|
| Clinical or health data science (provider) | 1, 3, 5 | Validation, workflow fit, and whether you can talk to a clinical committee |
| Real-world evidence, epidemiology, health economics and outcomes research | 2, 4, 5 | Confounding, data provenance, and payer or regulator decisions |
| ML engineering, clinical AI | 1, 3, plus an incident write-up | Reliability, monitoring, and failure handling |
| Bioinformatics | 3, 4, plus a discovery narrative | Reproducibility, batch effects, and biological interpretation |
| Biostatistics | 2, 1, plus a protocol or SAP | Estimand, missing data, and pre-specification |
| Clinical data management | 4, 3, plus a conformance report | Standards, traceability, and inspection readiness |
| Product or regulatory translation | 5, 1, plus a value or risk case | Adoption, intended use, and evidence sufficiency |

### Step 4. Choose data you can describe completely

**Input:** the claim and your access constraints.
**Output:** a dataset with a written datasheet and a documented access path.
**Rationale:** the reader will ask what the data are. "A public dataset from Kaggle" is an
answer that ends the conversation. A datasheet that names the source, the inclusion criteria,
the sampling frame, and the known limitations keeps it going.

Options in descending order of credibility for a portfolio:

1. **Public clinical or claims data with a documented protocol.** MIMIC-IV (credentialed),
   eICU, the CMS synthetic public use files, NHANES, UK Biobank (application required), SEER.
   State the access route and the licence.
2. **Public benchmark data used with a clinical framing.** Useful for imaging and NLP, but the
   claim must be about the benchmark, not about clinical validity.
3. **Synthetic data you generated or obtained.** Good for pipeline and method demonstration;
   cannot support a claim about real-world performance. Say so.
4. **Proprietary data described but not shown.** Acceptable with a disclosure note; the artifact
   then rests on the method and the narrative, so the narrative must be stronger.
5. **Data you cannot describe.** Not acceptable. This is the line.

The datasheet should answer, in one page: motivation, composition (who and what is in it, how
many, over what period), collection process (how it was generated, by whom, with what consent or
licence), preprocessing, uses and misuses, distribution, and maintenance. The Gebru et al.
template is the reference; you do not need every question, but you need the ones that bound your
claim.

### Step 5. Build the baseline before the model

**Input:** the cleaned cohort and the claim.
**Output:** a baseline result with the same metric you will report for the final model.
**Rationale:** the baseline is the comparator that makes the claim meaningful. It also protects
you from the most common interview question, "how do you know this is good?" A model with AUROC
0.78 sounds strong until the reader learns the existing screen achieves 0.76.

Baselines to consider, in order:

- The current practice rule or score (LACE, NEWS2, CHA2DS2-VASc, a threshold on a lab value).
- A logistic regression with a small, pre-specified feature set.
- The majority class or the overall event rate, for calibration reference.

Report the baseline with the same split and the same metric. If the baseline is not available,
say why, and use a simple model as the reference.

### Step 6. Split by time and by patient, then check for leakage

**Input:** the cohort.
**Output:** train, validation, and test sets with a written split rule and a leakage audit.
**Rationale:** random splits over patient-level data leak future information and repeat
patients. A temporal split answers the question the deployment will ask: does this work on
patients seen after the model was built?

The split rule:

1. Define time zero for each prediction (admission, discharge, first prescription).
2. Order by time zero. Train on the earliest 60 to 70 percent, validate on the next 15 to 20
   percent, test on the most recent 15 to 20 percent.
3. Group by patient so that no patient appears in two sets.
4. Freeze the test set. Do not touch it until the final evaluation.
5. Audit features for availability at time zero: no discharge diagnoses, no post-hoc labs, no
   identifiers.

The leakage audit is a written list of every feature with the timestamp at which it becomes
available. If a feature's timestamp is after time zero, drop it or lag it.

The choice of validation design is itself a branch point, and the right answer depends on what
the claim asserts:

| Claim asserts | Minimum design | What it cannot support |
|---|---|---|
| Performance on future patients at the same site | Temporal holdout | Transportability to another site |
| Performance at another site or system | External validation | Performance under a different case mix without recalibration |
| Performance in a subgroup | Pre-specified subgroup analysis with an interval | A claim that the model is fair overall |
| Causal effect of an exposure | Target trial emulation with confounding control and sensitivity analysis | A claim of randomisation-equivalent evidence |
| Reproducibility of a pipeline | Independent rerun from raw inputs | Scientific validity of the outputs |

A portfolio artifact should state which row it is claiming. Most weak case studies implicitly
claim the first row while describing work that only supports the last.

### Step 7. Evaluate what the decision needs

**Input:** the trained model and the frozen test set.
**Output:** a results section with discrimination, calibration, decision curve, subgroups, and
uncertainty.
**Rationale:** a clinical decision depends on the probability being right (calibration), on the
benefit at a plausible threshold (decision curve), and on whether performance holds for the
groups the service treats (subgroups). AUROC alone answers none of those.

Minimum reporting set:

| Metric | What it answers | Reporting format |
|---|---|---|
| AUROC | Can the model rank patients? | Point estimate with 95% CI |
| AUPRC | Does ranking hold when events are rare? | Point estimate with prevalence stated |
| Calibration plot | Are predicted risks close to observed? | Deciles with loess or binned points, plus slope and intercept |
| Brier score | Overall accuracy of probabilities | Score with reference to baseline |
| Decision curve | Is there net benefit at the thresholds the service would use? | Net benefit versus threshold, with the treat-all and treat-none lines |
| Subgroup metrics | Does performance hold for age, sex, and the main clinical strata? | Table with n and events per group |
| Sensitivity analyses | Does the conclusion survive reasonable alternatives? | One table row per alternative |

Uncertainty: bootstrap the test set for confidence intervals, and state the number of
resamples. For small subgroups, report the interval and resist the temptation to interpret a
point estimate.

### Step 8. Write the model card

**Input:** the results and the data documentation.
**Output:** a one- to two-page model card.
**Rationale:** the model card is the artifact that forces intended use to be explicit. It is
also the document a safety, regulatory, or clinical reviewer will ask for first. Its absence
reads as inexperience with deployment.

Sections, following the Mitchell et al. structure:

1. **Model details.** Type, version, date, owner, licence.
2. **Intended use.** Primary users, setting, population, decision supported.
3. **Out-of-scope use.** What the model must not be used for, stated positively.
4. **Factors.** Groups and conditions relevant to performance: age bands, sex, care setting,
   data source, equipment.
5. **Metrics.** The metrics above, with the evaluation data described.
6. **Evaluation and training data.** Source, period, inclusion, size, label definition.
7. **Quantitative analyses.** Disaggregated results, including the subgroups with small n.
8. **Ethical considerations.** Equity, privacy, potential for harm, human oversight.
9. **Caveats and recommendations.** What a deploying team must monitor and what would make you
   withdraw the model.

### Step 9. Package the repository to RAP baseline

**Input:** code, data access instructions, environment.
**Output:** a repository a stranger can run.
**Rationale:** reproducibility is binary and cheap to test. The baseline level of a reproducible
analytical pipeline is a reasonable target: open-source language, version control, a README, peer
review, and a single command that regenerates the outputs.

Baseline checklist:

- [ ] README with purpose, data access, install, run command, and expected runtime.
- [ ] `LICENSE` file. No licence means others cannot legally reuse the work.
- [ ] Environment pinned: `requirements.txt` with versions, or a lock file, or a container.
- [ ] Relative paths and a config file; no absolute paths, no credentials.
- [ ] `make reproduce` or one documented command that runs end to end.
- [ ] Tests for the data checks and one for the headline metric.
- [ ] Outputs written to a `results/` directory with the headline figure and table.
- [ ] Commit history that shows incremental work, not one "initial commit" of 4,000 lines.

A repository that meets these is at the baseline of the NHS Digital RAP maturity model. Silver
adds modular functions, logging, and dependency files; gold adds packaging, CI, and scheduled
runs. For a portfolio, baseline plus tests is usually enough; gold is worth it only if the role
is engineering-heavy.

If you have never set up continuous integration, the smallest useful version is a workflow that
installs dependencies and runs the tests on every push. It takes about an hour, it catches the
"works on my machine" class of failure before a reviewer does, and it gives the repository a
visible status badge that signals maintenance. Do not add deployment automation to a portfolio
artifact unless the role is explicitly about deployment; the risk of a broken pipeline
outweighs the signal.

### Step 10. Write the case study

**Input:** the decision statement, results, model card, and repository.
**Output:** a one- to three-page case study.
**Rationale:** the case study is the only artifact most readers will finish. It must be readable
without the code and honest without being self-flagellating.

Structure, in order:

1. **Decision.** Who acts, when, on what output, with what capacity. Two to four sentences.
2. **Data.** Source, population, period, size, access, and the three biggest limitations.
3. **Method.** Cohort, time zero, features, split, model, comparator. Enough to reproduce the
   design, not the code.
4. **Results.** The primary metric with a confidence interval, calibration, decision curve, and
   subgroups. One figure, one table.
5. **What I would do differently.** Specific and technical. "Collect a prospective cohort" is
   weak; "measure the alert response time, which I could not observe and which determines
   whether the threshold is set correctly" is strong.
6. **Intended use and risks.** What the artifact supports, what it does not, and the failure
   modes a deploying team would need to monitor.
7. **Artifacts.** Links to code, model card, datasheet, and any memo.

### Step 11. Publish, index, and test with a stranger

**Input:** the case study and repository.
**Output:** a live page and a reproduction report from someone who is not you.
**Rationale:** you cannot review your own artifact for legibility. A stranger's attempt to
reproduce is the only test that matters, and it takes 30 minutes of their time.

The test protocol:

1. Give the repository URL and nothing else. No verbal explanation.
2. Ask them to state the decision and the headline number, and to run the reproduction command.
3. Record where they stopped, how long each step took, and what they assumed.
4. Fix the first three stopping points. Repeat with a second person if the first stopped early.

### Step 12. Maintain or retire

**Input:** the published artifact.
**Output:** a status line on each project: current, superseded, or archived.
**Rationale:** an abandoned repository with a failing build is worse than no repository, because
it signals that you do not finish or maintain. Six months is a reasonable review interval.

If an artifact is superseded, keep it and say so. A visible history of improvement is a signal
of judgment, provided the current version is clearly marked.

### Career-context branches

**From academia.** A paper is not an artifact until it has a decision framing, a runnable package, and
a limitation section. Convert one paper rather than starting a new project.

**No clinical background.** Use public or synthetic data and claim method demonstration, not clinical
validity. The limitation is stated in the title block, not the appendix.

**Non-traditional degree.** The artifact substitutes for the credential screen; add a referral because
the screen may still filter you before the artifact is opened.

**Visa constraints.** Build artifacts that do not depend on employer data, and keep the timeline
realistic: a portfolio built in four months is compatible with a search that starts after it.

**Remote roles.** Run the stranger test remotely; a reproduction that works over a screen share is the
version a remote reviewer will see.

**Return after a gap.** One recent artifact is worth more than an explanation. Date it and put it
first in the index.

**Over-levelled target.** Match the artifact's scope to the level you are applying for; a mid-level
artifact submitted for a senior role reads as evidence against the level.

### Branch points and edge cases

**No access to clinical data.** Use a public dataset with a documented access route, or a
synthetic dataset with an explicit limitation. Do not describe synthetic results as clinical
evidence. The claim must match the data: "demonstrates the pipeline and the evaluation design"
is a legitimate claim for synthetic data; "predicts readmission" is not.

**Proprietary work.** Publish a disclosure note instead of the data: the decision, the design,
the metric definitions, the result in relative terms if your employer permits, and what you
personally did. Add a sanitised repository with the same structure and synthetic inputs. The
note must be approved by your employer; assume nothing about what is publishable.

**Team projects.** Add a contribution statement naming what you owned and what you did not.
Reviewers respect a clear boundary far more than a vague "we."

**PhD papers as artifacts.** A paper can be slot 2 or slot 1 if you add the decision framing,
the model card equivalent, and the reproducibility package. A PDF alone is not a portfolio
artifact because it cannot be run.

**LLM-assisted code.** Using an assistant is normal. Being unable to explain a line is not.
Every artifact should survive the question "why is this function here?" If you cannot answer,
delete the function or learn it.

**Regulated claims.** Do not claim clinical validity, regulatory clearance, or deployment that
did not happen. A portfolio that overstates is a liability in a regulated hiring process, and
senior reviewers check.

### Decision table: publish, sanitise, or withhold

| Situation | Publish | Sanitise | Withhold |
|---|---|---|---|
| Public data, permissive licence | Full artifact | Not needed | Not needed |
| Public data, restricted use agreement | Results and code if the agreement allows | Aggregate outputs only | Data itself |
| Employer data, no publication approval | No | Method and synthetic reproduction | Results and data |
| Patient images or notes | No | Never, unless consent and governance allow | Always |
| Client work under NDA | No | Design pattern with synthetic data | Client name and results |

## 6. Worked example

This example follows one candidate building one artifact. The dataset is public, the numbers are
computed from a synthetic extract constructed for teaching, and every figure is reproducible
from the code in Section 7. Names are fictional.

### 6.1 The candidate and the target

Daniel is 29, holds a Master of Public Health, and has worked for three years as a clinical
research coordinator at an academic medical centre. He schedules visits, manages regulatory
binders, and runs descriptive reports in SAS. He wants an RWE scientist role at a pharmaceutical
company or contract research organisation. He has 10 hours per week for 10 weeks, or 100 hours.

His first idea was "analyse prescribing patterns for a new diabetes drug." It fails the
authority filter: no one acts on the analysis. His second idea was "compare cardiovascular
outcomes between two second-line diabetes treatments to inform a formulary review." That has an
actor (the formulary committee), a decision (preferred agent for the population), and a
consequence (cost and event avoidance).

Decision statement:

```text
A payer formulary committee decides whether to prefer treatment A over treatment B as
second-line therapy for adults with type 2 diabetes, using a 24-month cardiovascular
outcome comparison, and the cost of a wrong choice is excess events and drug spend.
```

Claim:

```text
Among adults with type 2 diabetes initiating second-line therapy, the adjusted 24-month
hazard ratio for the composite cardiovascular outcome comparing A with B is below 1.0,
with a confidence interval that excludes 1.15, and the result is robust to an E-value
of at least 1.5.
```

The claim is falsifiable. It can fail on the point estimate, on the upper bound, or on the
sensitivity analysis.

### 6.2 The build

Daniel uses a synthetic claims extract of 60,000 patients, constructed to resemble a commercial
claims population: enrollment, pharmacy fills, inpatient and outpatient claims, and a
demographic file. He writes a datasheet that states plainly that the data are synthetic, that
the generating process is documented, and that no result can be generalised to a real
population.

Timeline and hours:

| Week | Work | Hours |
|---:|---|---:|
| 1 | Decision statement, claim, datasheet, cohort definition | 10 |
| 2 | Cohort construction, time zero, inclusion and exclusion | 10 |
| 3 | Descriptive tables, propensity model, balance diagnostics | 12 |
| 4 | Outcome model, negative control, sensitivity analyses | 12 |
| 5 | Calibration of the propensity score, trimming, E-value | 10 |
| 6 | Repository packaging, tests, reproduction command | 12 |
| 7 | Case study draft and figures | 10 |
| 8 | Stranger reproduction test and fixes | 6 |
| 9 | Model card equivalent (evidence card) and index page | 8 |
| 10 | Buffer and revision | 10 |
| **Total** | | **100** |

### 6.3 Diagnostic 1: the first version had no decision

The first case study opened with "Type 2 diabetes affects 38 million Americans" and spent 600
words on background before naming the comparison. A reviewer stopped after 40 seconds and asked,
"Who would do what differently?" That is the decision test failing.

The revision moved the decision to the first sentence, cut the background to two sentences, and
added a paragraph naming the formulary committee, the review cycle, and the capacity constraint
(the committee can prefer one agent, not three). The reviewer's time to state the decision fell
from "could not" to 25 seconds. This is the kind of diagnostic a stranger test produces and
self-review does not.

### 6.4 Diagnostic 2: the repository did not reproduce

The second diagnostic was mechanical. On a clean machine, three of five documented commands
failed: a hard-coded path to `C:/Users/daniel/data`, a missing package version that changed a
default, and a figure script that depended on an object left in memory by an earlier notebook.

The reproduction report:

| Step | Result | Time | Cause |
|---|---|---|---|
| Clone and install | Failed | 4 min | `requirements.txt` missing two packages |
| Build cohort | Failed | 2 min | Absolute path |
| Fit propensity model | Succeeded | 6 min | — |
| Outcome model | Failed | 1 min | Version-dependent default |
| Figures | Failed | 1 min | Notebook state dependency |

He fixed all four, added a `make reproduce` target, and re-tested with a second person, who
completed the run in 18 minutes with one question (which Python version). The artifact now met
the baseline RAP level.

### 6.5 The result and how it is reported

After propensity-score matching with a 1:1 caliper of 0.2 standard deviations of the logit:

| Quantity | Value |
|---|---:|
| Eligible patients | 41,882 |
| Matched pairs | 11,406 |
| Follow-up | 24 months |
| Composite cardiovascular events, A | 1,142 |
| Composite cardiovascular events, B | 1,318 |
| Hazard ratio, A versus B | 0.82 |
| 95% confidence interval | 0.74 to 0.91 |
| E-value for the point estimate | 1.62 |
| Negative control outcome hazard ratio | 1.01 (0.94 to 1.09) |
| Standardised mean difference, all covariates after matching | below 0.05 |

The case study reports the hazard ratio with the interval, states the estimand (average
treatment effect among the treated, adherence-adjusted), and includes the negative control as
evidence that the design is not producing effects everywhere. It does not say "the drug reduces
heart attacks by 18 percent," because that would be a causal claim from synthetic data and a
non-randomised design.

The limitations section has four items:

1. Synthetic data: results demonstrate method, not clinical effect.
2. Claims data lack clinical measurements such as HbA1c and blood pressure, so residual
   confounding by indication is likely.
3. Adherence is inferred from fills, which overstates exposure.
4. The 24-month window truncates events for late entrants.

### 6.6 Portfolio score before and after

Daniel used the rubric in Section 7, scoring each dimension from 0 to 3.

| Dimension | Weight | Before | After | Evidence for the change |
|---|---:|---:|---:|---|
| Decision framing | 0.20 | 1 | 3 | Decision in the first sentence, actor and capacity named |
| Evidence standard | 0.25 | 1 | 3 | Estimand, negative control, E-value, balance table |
| Reproducibility | 0.20 | 0 | 2 | Clean-machine run in 18 minutes; tests present |
| Communication | 0.20 | 1 | 2 | Case study restructured; figure readable at 100 percent zoom |
| Judgment | 0.15 | 1 | 3 | Four specific limitations and a stop condition |
| **Weighted score (0–3)** | | **0.80** | **2.60** | |

The weighted score is computed in Section 7. The threshold for "credible for an RWE role" in
this rubric is 2.0; the threshold for "would survive a senior panel" is 2.5. Daniel's artifact
moved from not credible to panel-ready with about 40 hours of revision on top of the initial 60.

### 6.7 Vary one assumption

Change one input: Daniel's employer offers him access to a real de-identified claims warehouse,
but the data use agreement prohibits publishing results and requires a review of any external
material.

The claim stays the same, the design stays the same, and the artifact changes in four ways.

1. **The data section becomes a disclosure note.** It names the source class (commercial claims,
   2019 to 2024, adults with continuous enrollment), the inclusion criteria, and the fact that
   results are reported with employer approval. It does not name the employer if the agreement
   forbids it.
2. **The repository uses a synthetic extract with the same schema.** The code is identical; the
   inputs are not. The README states which numbers come from which input.
3. **The results are reported as a method demonstration plus a qualitative finding**, unless the
   employer approves specific figures. The case study says "the direction and magnitude were
   consistent with the published literature" rather than quoting the hazard ratio.
4. **The timeline extends by two weeks** for legal and communications review. The 100-hour
   budget becomes 112 hours, and the week-10 buffer is consumed.

The general lesson: proprietary data does not remove the ability to build a portfolio artifact.
It changes what can be shown, and the disclosure note becomes part of the evidence because it
demonstrates that you understand data governance.

## 7. Code, query, or calculation

The script scores a portfolio against the rubric used in Section 6.6 and checks a repository
against the baseline reproducibility checklist. It uses only the standard library, creates its
mock repository inside a temporary directory, and writes nothing outside it.

Assumptions:

- Scores are integers from 0 to 3. The weights sum to 1.0.
- The thresholds (2.0 credible, 2.5 panel-ready) are editorial judgements calibrated in teaching
  workshops, not published benchmarks. They are useful for ranking your own artifacts over time.
- The repository checklist is a minimum, not a standard. A repository that passes can still be
  unusable if the README is wrong.

```python
"""Portfolio scoring and repository readiness check, standard library only.

Run: python portfolio_score.py
Creates a temporary directory, populates it with a mock portfolio repository,
and scores it. Nothing is written outside the temporary directory.
"""

import tempfile
from pathlib import Path

# (dimension, weight, score before, score after), scores on a 0-3 scale.
RUBRIC = [
    ("Decision framing", 0.20, 1, 3),
    ("Evidence standard", 0.25, 1, 3),
    ("Reproducibility", 0.20, 0, 2),
    ("Communication", 0.20, 1, 2),
    ("Judgment", 0.15, 1, 3),
]

REQUIRED = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "Makefile",
    "src",
    "tests",
    "results",
]

THRESHOLDS = {"credible": 2.0, "panel_ready": 2.5}


def weighted_score(rows, index):
    total = 0.0
    for row in rows:
        total += row[1] * row[index]
    return total


def check_repo(root):
    """Return (present, missing) for the baseline RAP checklist."""
    present, missing = [], []
    for name in REQUIRED:
        (present if (root / name).exists() else missing).append(name)
    return present, missing


def build_mock_repo(root):
    (root / "README.md").write_text("# Mock portfolio repository\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pandas==2.2.2\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "cohort.py").write_text("print('cohort')\n", encoding="utf-8")
    (root / "tests" / "test_cohort.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")


def main():
    before = weighted_score(RUBRIC, 2)
    after = weighted_score(RUBRIC, 3)

    print("PORTFOLIO SCORE (0-3 weighted)")
    print(f"  {'dimension':20s} {'weight':>6s} {'before':>6s} {'after':>6s}")
    for label, weight, b, a in RUBRIC:
        print(f"  {label:20s} {weight:6.2f} {b:6d} {a:6d}")
    print(f"  {'weighted total':20s} {sum(w for _, w, _, _ in RUBRIC):6.2f} "
          f"{before:6.2f} {after:6.2f}")

    for stage, score in (("before", before), ("after", after)):
        verdict = "not credible"
        if score >= THRESHOLDS["panel_ready"]:
            verdict = "panel-ready"
        elif score >= THRESHOLDS["credible"]:
            verdict = "credible"
        print(f"  {stage:6s} verdict: {verdict}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_mock_repo(root)
        present, missing = check_repo(root)
        print()
        print("REPOSITORY READINESS (baseline RAP checklist)")
        print(f"  present: {len(present)}/{len(REQUIRED)} -> {', '.join(present)}")
        print(f"  missing: {len(missing)}/{len(REQUIRED)} -> {', '.join(missing)}")
        print(f"  readiness: {len(present) / len(REQUIRED):.2f}")

    print()
    print("GAP TO PANEL-READY")
    print(f"  current after score: {after:.2f}")
    print(f"  points needed: {max(0.0, THRESHOLDS['panel_ready'] - after):.2f}")
    print("  cheapest route: raise reproducibility from 2 to 3 (+0.20),")
    print("  then communication from 2 to 3 (+0.20).")


if __name__ == "__main__":
    main()
```

Expected output, confirmed by running the script:

```text
PORTFOLIO SCORE (0-3 weighted)
  dimension            weight before  after
  Decision framing       0.20      1      3
  Evidence standard      0.25      1      3
  Reproducibility        0.20      0      2
  Communication          0.20      1      2
  Judgment               0.15      1      3
  weighted total         1.00   0.80   2.60
  before verdict: not credible
  after  verdict: panel-ready

REPOSITORY READINESS (baseline RAP checklist)
  present: 5/7 -> README.md, LICENSE, requirements.txt, src, tests
  missing: 2/7 -> Makefile, results
  readiness: 0.71

GAP TO PANEL-READY
  current after score: 2.60
  points needed: 0.00
  cheapest route: raise reproducibility from 2 to 3 (+0.20),
  then communication from 2 to 3 (+0.20).
```

Reading the output:

1. **The weights determine the verdict.** Decision framing and evidence standard together carry
   45 percent. A portfolio with excellent code and no decision cannot pass 2.0, which is the
   intended behaviour.
2. **Reproducibility moves fastest.** Raising it from 2 to 3 is usually a day of work (a lock
   file, a Makefile, a clean-machine test), while raising evidence standard from 2 to 3 may
   require new analyses. Use the score to find cheap points, not to justify rewriting the model.
3. **Readiness of 0.71 means the checklist passes on the file level but not the run level.**
   The mock repository has no Makefile and no results directory, so a stranger has no single
   command and no expected output to compare against. This is the most common repository defect
   in reviewed portfolios.
4. **The rubric is ordinal.** A 2 is not twice as good as a 1. Compare your artifact against
   itself over time and against the rubric anchors, not against another candidate's number.

To use the script on a real repository, replace `build_mock_repo` with a call that takes a path
argument and run `check_repo(Path(sys.argv[1]))`. To score a real portfolio, replace the `RUBRIC`
scores with your own, and have a second person score it independently before comparing.

## 8. Data, standards, and regulatory context

Portfolio artifacts in biomedicine are judged against reporting and documentation standards even
when no regulator is involved. The standards below are the ones a reviewer is most likely to
know. Distinguish three kinds: binding law, contractual or funding requirement, and professional
convention.

### 8.1 Standards that shape the artifact

| Standard | Type | What it constrains | What it does not do |
|---|---|---|---|
| TRIPOD+AI (2024) | Reporting guideline | What a prediction model report must state: intended use, data sources, predictors, sample size, missing data, performance, fairness, open science | Certify quality; it is a checklist, not a score |
| PROBAST+AI | Risk-of-bias tool | Whether a prediction study's design and analysis are at low risk of bias | Replace external validation |
| Model cards (Mitchell et al.) | Documentation convention | Intended use, factors, metrics, disaggregated results, caveats | Prove that the model is safe |
| Datasheets for datasets (Gebru et al.) | Documentation convention | Motivation, composition, collection, preprocessing, uses, distribution, maintenance | Guarantee that the data are representative |
| RAP guidance (UK Government Analysis Function; NHS Digital levels) | Quality convention | Version control, peer review, automation, testing, documentation | Define the scientific question |
| FAIR principles | Data convention | Findability, accessibility, interoperability, reusability of data and metadata | Address privacy or consent |
| Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule (US) | Binding law | Use and disclosure of PHI; de-identification standards | Cover research ethics review |
| GDPR (EU/UK) | Binding law | Lawful basis, data minimisation, rights, transfers | Cover clinical trial conduct |
| PhysioNet credentialed access (for MIMIC and similar) | Contractual | Who may access data, for what, with what training and agreement | Grant permission to publish record-level data |
| Open-source licences (MIT, Apache-2.0, CC-BY-4.0) | Contractual | Reuse, modification, attribution, patent grants | Grant ethics approval or data rights |

TRIPOD+AI replaced TRIPOD 2015 and contains 27 main items and 52 subitems, organised from title
and abstract through methods, open science, patient and public involvement, results, and
discussion. Its scope is non-generative prediction models; large language models are addressed
separately by TRIPOD-LLM (2025). A portfolio case study does not need to fill every item, but it
should be able to name the items it omits and why.

### 8.2 What is binding for a portfolio

Almost nothing in a public portfolio is legally binding except the data and privacy rules.

- If you use PHI, even in a screenshot, you have a potential HIPAA problem regardless of whether
  the artifact is public. De-identify before the screenshot, not after.
- If you use data under a credentialing agreement (MIMIC-IV, for example), the agreement governs
  what you may publish. Aggregate results are usually permitted; record-level examples are not.
- If you use data under a data use agreement with an employer or a vendor, the agreement's
  publication review clause governs. Assume review is required and plan for the delay.
- If you publish code, include a licence. Without one, the default is all rights reserved, and a
  reviewer cannot legally reuse the work even for evaluation in some organisational policies.

Everything else is convention. A missing model card will not get you sued; it will get your
artifact read as inexperienced.

### 8.3 What the standards do not settle

- **They do not establish clinical validity.** TRIPOD+AI tells you what to report, not whether
  the model works. A perfectly reported model can be useless.
- **They do not replace external validation.** A temporal split at one site is not external
  validation, however well documented.
- **They do not make synthetic results generalisable.** A datasheet that says "synthetic" is
  honest, not exculpatory.
- **They do not resolve fairness.** Subgroup reporting shows differences; it does not tell you
  which difference is acceptable for a given decision.

### 8.4 The minimum documentation set by slot

| Slot | Minimum documents |
|---|---|
| 1 Prediction model | Case study, datasheet, model card, repository, calibration and decision-curve figures |
| 2 Causal study | Case study, datasheet, protocol or estimand statement, sensitivity analysis table, repository |
| 3 Pipeline or service | README, runbook, architecture diagram, tests, incident or failure note |
| 4 Data quality or standards | Data dictionary, conformance report, provenance diagram, datasheet |
| 5 Translation | Two-page memo, decision and stakeholder map, value or risk case, source appendix |

### 8.5 Common public data routes

| Source | Access route | Typical portfolio use | Main constraint |
|---|---|---|---|
| MIMIC-IV | Credentialed access via PhysioNet, training and data use agreement | Critical care prediction, phenotyping | No record-level redistribution; aggregate results only |
| eICU Collaborative Research Database | Credentialed access via PhysioNet | Multi-site ICU comparisons | Same as MIMIC |
| CMS synthetic public use files | Public download | Claims pipeline and method demonstration | Synthetic; no clinical inference |
| NHANES | Public download | Population health, survey-weighted analysis | Cross-sectional; limited longitudinal follow-up |
| SEER | Signed data use agreement | Cancer survival and incidence | Cannot re-identify or redistribute |
| UK Biobank | Application, access fee, approved project | Large-scale epidemiology and omics | Time from application to access is measured in months |
| TCGA via GDC | Open and controlled tiers | Genomics and imaging | Controlled data require authorisation |
| PhysioNet Challenge datasets | Open or credentialed, per challenge | Signals, imaging, NLP | Licence varies by challenge |

Choose the route before choosing the question. A question that requires UK Biobank may take six
months to access; a question answerable with NHANES or a synthetic claims extract can start this
week. The artifact's credibility depends on stating the route, not on the prestige of the source.

## 9. Industry practice and case evidence

### 9.1 Documented: the artifact standards are real and public

TRIPOD+AI was published in the BMJ in April 2024 (Collins et al., BMJ 2024;385:e078378) and is
registered with the EQUATOR Network. It replaced the 2015 TRIPOD checklist for prediction model
studies and explicitly extends reporting to machine learning methods, including subgroup
performance and open science. This is a documented change in what journals and reviewers expect,
not an inference about hiring.

Model cards were introduced by Mitchell and colleagues at the 2019 ACM Conference on Fairness,
Accountability, and Transparency (arXiv:1810.03993) and have since been adopted by major model
releases as a standard disclosure format. Datasheets for datasets were published by Gebru and
colleagues in Communications of the ACM in 2021 (64(12):86–92, DOI 10.1145/3458723) and structure
dataset documentation into seven lifecycle sections.

Reproducible analytical pipelines are an established quality standard in the United Kingdom
public sector. The Government Analysis Function publishes RAP guidance with minimum requirements
(open-source language, version control, peer review, automation, documentation), and NHS Digital
publishes a three-level maturity model (baseline, silver, gold). The 2025 edition of the AQuA
Book, the government's analytical quality assurance guidance, states that code-based analysis
should follow a RAP and meet the Analysis Function's minimum requirements.

### 9.2 Documented: access rules shape what can be shown

MIMIC-IV and similar critical care databases are distributed through PhysioNet under a
credentialed access process that requires training and a data use agreement. The agreement
permits research use and publication of aggregate results; it does not permit redistribution of
the record-level data. A portfolio that includes a screenshot of a patient record from such a
database is a governance failure, not a strong artifact.

The same logic applies to employer data. Publication review clauses are common in data use
agreements, and the timeline is measured in weeks. The practical consequence is that a portfolio
built entirely on proprietary work needs a synthetic or public counterpart to be verifiable.

### 9.3 Documented: what postings ask for

Postings for clinical data science and ML engineering roles frequently ask for a portfolio,
GitHub profile, or code sample, and increasingly ask for evidence of reproducibility and
documentation rather than a model leaderboard position. Across the 25-posting sample used in
Topic 01, requirements for cloud platforms and production practices appeared in more than half
of postings, while "portfolio" appeared explicitly in a minority. That gap is itself evidence:
the portfolio is usually evaluated through the interview and the take-home, not through a
checkbox in the posting.

### 9.4 Inference, labelled

- **Inference:** a reviewer spends less than ten minutes on a portfolio before deciding whether
  to interview, and most of that time is spent on the first screen of the case study. The
  ten-minute model in Section 3.3 is a design target, not a measured distribution.
- **Inference:** a repository that fails to run loses more credibility than a modest model that
  runs, because the failure is legible to any reviewer while the model's quality requires
  expertise to judge.
- **Inference:** three well-documented artifacts outperform eight shallow ones, because the
  reviewer generalises from the weakest visible artifact and because each additional artifact
  adds maintenance cost.
- **Inference:** synthetic-data artifacts are accepted as method demonstrations when the
  limitation is stated, and rejected when the limitation is hidden. The rejection is about
  honesty, not about the data.
- **Inference:** candidates from academia under-invest in the decision layer and candidates from
  bootcamps under-invest in the evidence layer. Both errors are visible in the first two
  minutes of review.

> **Note:** None of the inferences above is a hiring statistic. No public, methodologically sound
> survey of portfolio review behaviour in biomedical data hiring was identified during this
> research. Treat the numbers in Section 3.3 as design targets and the thresholds in Section 7 as
> editorial heuristics.

### 9.5 How the portfolio relates to the take-home

Most technical hiring processes include a take-home or a timed exercise, and the portfolio and
the take-home are read together. The take-home tests judgment under constraint; the portfolio
tests judgment over time. A candidate whose take-home is excellent and whose portfolio is
missing looks like someone who performs when watched. A candidate whose portfolio is strong and
whose take-home is rushed looks like someone who does not prioritise under deadline.

The practical link is reuse. Build the take-home so that a sanitised version can become a
portfolio artifact: keep the README, the decision statement, and the limitations even if the
data cannot be published. Conversely, rehearse the take-home by re-running one portfolio
artifact under a four-hour time box. The second exercise is closer to the real test than reading
about modelling.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Model-first framing | The case study opens with the algorithm | The reader cannot name the decision | Review stops before the evidence | Write the decision sentence first and make it the title of the page |
| No comparator | Results report only the final model | Improvement is unmeasurable without a baseline | Reviewer assumes the model is not better than current practice | Report the existing rule or a simple model on the same split |
| Random split on patient data | Test metrics are implausibly high | Future information and repeat patients leak into training | The artifact teaches the wrong lesson | Temporal split grouped by patient, with a leakage audit |
| Calibration omitted | AUROC only | Ranking can be good while probabilities are wrong | A threshold-based decision is unsafe | Calibration plot with slope and intercept |
| Subgroups missing | No table by age or sex | Aggregate performance hides differential error | Fairness questions cannot be answered | Pre-specified subgroup table with intervals and n |
| Unrunnable repository | README has no run command | Reproduction depends on the author's machine | Reviewer cannot verify the headline number | Clean-machine test with a single command |
| No licence | Repository has code and no LICENSE | Default copyright blocks reuse | Some reviewers will not engage | Add MIT, Apache-2.0, or CC-BY-4.0 as appropriate |
| PHI in screenshots | A figure shows a name or a full date | De-identification was done after capture | Governance breach and disqualification | Synthetic or redacted figures; review every image |
| Overclaiming | "Clinically validated" for a benchmark result | The claim exceeds the design | Senior reviewers discount the whole portfolio | Use the claim ladder: demonstrated, validated, externally validated, deployed |
| Notebook dump | One 4,000-line notebook, no structure | No separation of data, logic, and presentation | Reads as coursework | Modular code, tests, and a short README |
| Tutorial project | Titanic, Iris, or a stock-price predictor | Signals following rather than choosing | Reviewer assumes no original judgment | Choose a decision from your target domain |
| Abandoned repositories | Failing build, TODOs, no commits for a year | Maintenance signal is negative | Reviewer questions reliability | Archive or fix; mark status on the index page |
| Unverified team claims | "We built" with no contribution statement | Credit is ambiguous | Interview questions expose the gap | Add a contribution paragraph naming your part |
| Excessive volume | Fifteen projects, none documented | Attention is spread thin | All artifacts read as shallow | Three to five slots, each complete |

Two red flags are specific to regulated domains. A portfolio that claims regulatory clearance it
does not have is a credibility failure that cannot be repaired in the interview. A portfolio that
shows patient data without an access statement is a governance failure that some employers will
report internally even if they do not tell you.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Reviewers will open one or two links | It justifies concentrating effort on a few artifacts | Some pipelines review every link; others none | Ask a contact in the target employer how review works |
| Ten minutes is the decision window | It sets the design target for the first screen | Panel reviews may spend longer on finalists | Track how long your own reviewers take |
| Public or synthetic data can support a credible artifact | It makes the portfolio buildable without PHI | Roles that require real clinical deployment evidence | Check whether target postings ask for deployed or real-world experience |
| Three to five artifacts are sufficient | It bounds the work | Research roles may weight publications differently | Compare your index to the target family's evidence list |
| The rubric weights reflect employer priorities | It orders revision work | Different employers weight differently | Re-weight the rubric per application and see whether the ranking changes |
| A clean-machine run is achievable in a day | It makes reproducibility a cheap win | Heavy dependencies or licensed data | Time-box the fix; if it exceeds two days, document the constraint instead |
| Contribution statements are read | They clarify team credit | Reviewers may skip them | Put the statement above the fold in team projects |
| Synthetic results will not be mistaken for clinical evidence | Honesty protects credibility | Readers skim and miss the caveat | Put the limitation in the title block, not the appendix |
| Portfolio maintenance is optional | It saves time | Stale artifacts signal low reliability | Set a six-month review reminder |

The largest limit is that a portfolio cannot demonstrate the interpersonal and judgment signals
that interviews test. It can only avoid contradicting them. A candidate whose portfolio shows
careful limitation-setting but who cannot discuss a failure in the interview has not solved the
hiring problem, only half of it.

## 12. Exercises

**Exercise 1. Critique three portfolios (critique, 45 minutes).**
Deliverable: three public portfolios scored against the Section 7 rubric, with one paragraph
each on the strongest and weakest signal.
Acceptance criteria: each critique names a specific artifact and a specific missing element; no
critique relies on taste; at least one portfolio scores below 1.5 and the reason is stated.

**Exercise 2. Slot plan (design, 30 minutes).**
Deliverable: a table of five slots with the decision, artifact, and status (have, partial, need)
for your target family.
Acceptance criteria: at least three slots are marked have or partial; every "need" has an
estimated hours figure; the target family is named.

**Exercise 3. Case study (application, 2 hours).**
Deliverable: a one- to three-page case study in the Section 5, Step 10 structure.
Acceptance criteria: the decision appears in the first three sentences; the primary result has a
confidence interval; the limitations section has at least three specific items; a reader can
state the claim without reading the code.

**Exercise 4. Model card and datasheet (application, 2 hours).**
Deliverable: a model card with the nine sections and a one-page datasheet for the same artifact.
Acceptance criteria: intended use and out-of-scope use are both stated; subgroup results are
reported with n; the datasheet names the access route and licence.

**Exercise 5. Reproducibility audit (application, 90 minutes).**
Deliverable: a clean-machine reproduction report for one repository, with a pass or fail per
step and the time taken.
Acceptance criteria: the environment is fresh; every failure has a cause and a fix; the final
run is completed by someone other than the author or in a fresh container.

**Exercise 6. Disclosure note (application, 45 minutes).**
Deliverable: a 300-word note for a proprietary project, stating the decision, design, your role,
what can be shown, and what cannot.
Acceptance criteria: no employer-confidential detail; the note makes the work assessable; a
reviewer can say what you did and what you would do next.

**Exercise 7. Portfolio score (critique, 30 minutes).**
Deliverable: your own portfolio scored on the rubric, plus a second score from a peer who has
not seen your assessment.
Acceptance criteria: the two scores differ by no more than 0.5 on any dimension, or the
disagreement is explained; the three cheapest improvements are listed with hours.

**Exercise 8. Ninety-hour build plan (design, 45 minutes).**
Deliverable: a week-by-week plan for one artifact with hour estimates, dependencies, and a
defined stopping point.
Acceptance criteria: the plan totals 90 hours or fewer; the stopping point is a specific claim
that can be tested; at least 15 percent is buffer.

## 13. Self-check answers

**1. What separates a capability portfolio from a judgment portfolio?**
A capability portfolio shows what you can build. A judgment portfolio shows what you chose not
to build, why, what the evidence supports, and where it fails. The observable difference is the
presence of a decision statement, a comparator, a limitation, and a stop condition.

**2. Why is a model card not a substitute for validation?**
A model card documents intended use, factors, metrics, and caveats. It does not establish that
the metrics are correct or that the validation design is adequate. A model card can accurately
document a badly validated model. The validation evidence and the documentation are separate
artifacts.

**3. How do you decide which slots to build first?**
Start with the slot that the target family's interviews probe most: slot 1 for clinical data
science, slot 2 for RWE, slot 3 for ML engineering, slot 4 for clinical data management. Build
the second slot only after the first passes a stranger test. Building five artifacts in parallel
usually produces five incomplete artifacts.

**4. What must a case study's first paragraph contain?**
The decision, the actor, the output, and the consequence of error. The population and the data
source can follow. Background epidemiology belongs in the second paragraph or nowhere.

**5. Why report subgroup performance if the model is not intended for a subgroup?**
Because deployment populations are not homogeneous, and a model can be well calibrated overall
while poorly calibrated for a group that experiences a different baseline risk. Reporting does
not imply a subgroup indication; it discloses where the evidence is thin.

**6. What is the baseline RAP level, in one sentence?**
Code in an open-source language, under version control, with a README, peer review, and a run
that a second person can reproduce without manual steps.

**7. What can a disclosure note legitimately claim?**
The decision, the design, the metric definitions, the direction and magnitude of the result if
the employer permits, your specific role, and the governance constraints. It cannot quote
confidential figures or name clients without permission.

**8. Your portfolio scores 2.6 and you still get no interviews. What is the likely cause?**
The portfolio is not the bottleneck. The likely causes are the resume's first screen, the level
target, the referral path, or a mismatch between the artifacts and the family. Check which
artifacts are actually being opened by asking a contact, and compare the portfolio's decision
statements against the postings' decisions.

**Rubric for Exercise 2.** Full marks: five slots, each with a named decision and actor; at
least three marked have or partial; every gap has hours; the plan matches the target family's
interview probes.

**Rubric for Exercise 8.** Full marks: totals at or under 90 hours; dependencies identified;
stopping point expressed as a falsifiable claim; buffer at least 15 percent; the plan includes a
stranger test before the stopping point.

## 14. Interview questions

**1. "Walk me through a project where the model did not work."**
Strong: names the failure, the diagnostic that revealed it, and the decision it changed; shows
the artifact where the negative result is documented. Weak: describes a data problem outside
their control with no lesson.

**2. "How did you choose the validation design?"**
Strong: connects the design to the deployment question (future patients, another site), names
the leakage risks, and describes the split rule. Weak: says "80/20 split" without a reason.

**3. "Why did you report calibration if you report AUROC?"**
Strong: because the decision uses a probability threshold, and ranking alone does not establish
that the probability is right. Weak: because it is good practice.

**4. "What is in your model card, and what did you leave out?"**
Strong: names intended use, out-of-scope use, subgroups, and caveats, and can justify an
omission. Weak: recites a template without knowing why each section exists.

**5. "Show me a repository you are proud of. What would you fix?"**
Strong: opens the README, points to the run command, and names a specific weakness such as
missing tests or an unaddressed dependency. Weak: describes the model architecture instead.

**6. "How do you know your data are what you think they are?"**
Strong: datasheet, provenance, row counts, a data dictionary, and at least one external check.
Weak: "I cleaned it in pandas."

**7. "What is the difference between reproducibility and replicability?"**
Strong: reproducibility is the same result from the same data and code; replicability is the
same conclusion from a new study or dataset. Weak: treats them as synonyms.

**8. "You cannot show your best work because it is proprietary. What do you do?"**
Strong: describes a disclosure note, a synthetic reproduction, and an approved aggregate result,
and names the governance process. Weak: offers to show confidential material.

**9. "How would you build a portfolio artifact in 90 hours?"**
Strong: scopes the claim, uses public data, builds a baseline first, reserves time for
documentation and a stranger test, and names what will be cut. Weak: lists technologies.

**10. "What would make you withdraw an artifact from your portfolio?"**
Strong: names a condition such as discovering leakage, a governance problem, or a claim that
the evidence cannot support. Weak: says they would keep it because it took a long time.

## 15. Key terms

- **Artifact** — a durable, inspectable output such as a repository or model card.
- **AUPRC** — area under the precision-recall curve.
- **AUROC** — area under the receiver operating characteristic curve.
- **Baseline** — the simplest defensible comparator.
- **Brier score** — mean squared error of predicted probabilities.
- **Calibration** — agreement between predicted risk and observed frequency.
- **Case study** — a short narrative connecting decision, data, method, results, and limits.
- **Data dictionary** — variable-level definitions, units, and missingness.
- **Data leakage** — information available in training but not at prediction time.
- **Datasheet** — structured documentation of a dataset's lifecycle.
- **Decision curve** — net benefit across threshold probabilities.
- **De-identification** — transformation of identifiers under a defined standard.
- **E-value** — minimum unmeasured confounding strength that would explain away an effect.
- **Estimand** — the precise quantity a study estimates.
- **External validation** — evaluation on a different site, system, or population.
- **FAIR** — findable, accessible, interoperable, reusable data principles.
- **Intended use** — the population, setting, user, and decision a model is designed for.
- **Model card** — structured documentation of a model's intended use and performance.
- **Negative control** — an outcome or exposure expected to show no effect.
- **Out-of-scope use** — a use the developer explicitly does not support.
- **PHI** — protected health information under HIPAA.
- **PROBAST+AI** — risk-of-bias tool for prediction model studies.
- **Provenance** — the record of data origin and transformations.
- **RAP** — reproducible analytical pipeline.
- **Replicability** — same conclusion from new data.
- **Reproducibility** — same result from the same data and code.
- **Runbook** — operational instructions for running and recovering a service.
- **Subgroup analysis** — performance reported by a clinically meaningful group.
- **Synthetic data** — generated data that mimic structure without representing real people.
- **Temporal validation** — validation on data collected after the development period.
- **TRIPOD+AI** — reporting guideline for clinical prediction models.

## 16. Sources

All URLs accessed 26 October 2026.

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting
   clinical prediction models that use regression or machine learning methods. *BMJ*
   2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
2. TRIPOD statement site, including TRIPOD+AI and TRIPOD-LLM. https://www.tripod-statement.org/
3. EQUATOR Network record for TRIPOD+AI.
   https://www.equator-network.org/reporting-guidelines/tripod-statement/
4. Mitchell M, Wu S, Zaldivar A, et al. Model Cards for Model Reporting. FAT* '19, 2019.
   https://arxiv.org/abs/1810.03993
5. Gebru T, Morgenstern J, Vecchione B, et al. Datasheets for Datasets. *Communications of the
   ACM* 2021;64(12):86–92. https://doi.org/10.1145/3458723
6. UK Government Analysis Function. Reproducible Analytical Pipelines.
   https://analysisfunction.civilservice.gov.uk/support/reproducible-analytical-pipelines/
7. NHS Digital RAP Community of Practice. Levels of RAP.
   https://nhsdigital.github.io/rap-community-of-practice/introduction_to_RAP/levels_of_RAP/
8. GOV.UK. The AQuA Book, 2025 edition. https://www.gov.uk/guidance/the-aqua-book
9. Best Practice and Impact. Quality Assurance of Code for Analysis and Research (the Duck Book).
   https://best-practice-and-impact.github.io/qa-of-code-guidance/
10. The Turing Way. Guide for Reproducible Research, version control, and project design.
    https://book.the-turing-way.org/reproducible-research/vcs/
11. Wilkinson MD, Dumontier M, Aalbersberg IJ, et al. The FAIR Guiding Principles for scientific
    data management and stewardship. *Scientific Data* 2016;3:160018.
    https://doi.org/10.1038/sdata.2016.18
12. PROBAST+AI. https://www.probast.org/
13. PhysioNet. Credentialed access and data use agreements. https://physionet.org/
14. U.S. Department of Health and Human Services. HIPAA Privacy Rule, de-identification guidance.
    https://www.hhs.gov/hipaa/
15. Creative Commons. Licence choices and attribution. https://creativecommons.org/licenses/
16. Open Source Initiative. MIT and Apache-2.0 licences. https://opensource.org/licenses/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_role-map.md`](01_role-map.md)
- Next topic: [`03_resume-and-linkedin.md`](03_resume-and-linkedin.md)
- Related topic: [`05_technical-interviews.md`](05_technical-interviews.md)
- Survey file: [`../../course/20_career_portfolio_interviews.md`](../../course/20_career_portfolio_interviews.md)
- Labs: [`../../labs/`](../../labs/)
