# Benchmark Pitfalls

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 12 of 12
> Estimated study time: 3 hours
> Prerequisites: Topic 01, Topic 08, Topic 11
> Version 2026.3

## 1. Why this topic exists

A benchmark is a measurement instrument, and like any instrument it can be miscalibrated, contaminated, or used for a purpose it was not designed for. The failures are common enough that a benchmark result should be treated as a hypothesis until it is audited. The audit is not a formality; it changes decisions.

The failures are specific and recurring:

- Test items appear in pretraining data, and the score measures memorisation as well as capability.
- Two papers report numbers on different splits, and the comparison is meaningless.
- A leaderboard is tuned against for months, and the winner is the team that overfitted most.
- The task drifts from the clinical decision, and a higher score means less.
- A benchmark saturates, and a one-point difference is reported as progress.
- A benchmark is retired, and the old number is still cited.
- A vendor presents a public leaderboard rank as deployment evidence.

This topic closes the module by treating benchmark validity as an engineering problem. It covers contamination, non-comparable splits, leaderboard overfitting, task drift, saturation, and the claims a benchmark cannot support. It gives a contamination calculation, a selection-inflation calculation, and a rule for retiring a benchmark.

The central idea is that a benchmark result is a claim about a benchmark, not a claim about the world. The gap between the two is the subject of this topic.

The practical consequence is that benchmark literacy is a form of evidence literacy. The question is not "what is the score" but "what does the score measure, on whom, under what conditions, and what would change it." A team that asks those questions will use benchmarks well; a team that does not will be managed by them.

A second consequence is that benchmarks need maintenance. A benchmark is a public good with a maintainer, a version, and a retirement. Treating it as a permanent fact is how stale numbers survive in slide decks for years.

> **Note:** Benchmark versions, leaderboards, and scores change. This topic does not report 2026 scores or winners. It describes audit methods and decision rules. Verify the benchmark version and terms at the official page before citing a result.

## 2. Learning outcomes

By the end of this topic you can:

1. **Audit** a benchmark result for contamination, split comparability, and task drift. Assessed by Exercise 1 and self-check item 1.
2. **Compute** the inflation from a contaminated subset and correct the reported estimate. Assessed by Exercise 2.
3. **Detect** non-comparable splits and explain why a comparison is invalid. Assessed by Exercise 3.
4. **Quantify** leaderboard selection inflation and decide whether a ranking is supported. Assessed by Exercise 4.
5. **Recognise** task drift and saturation, and state when a benchmark should be retired. Assessed by Exercise 5.
6. **Map** a benchmark claim to the evidence it actually supports. Assessed by Exercise 6.
7. **Design** a clean evaluation with a private holdout and a pre-specified metric. Assessed by Exercise 7.
8. **Write** a benchmark audit memo with a retire-or-keep recommendation. Assessed by Exercise 8.

## 3. The core mental model

A benchmark claim passes through five filters. A failure at any filter weakens or invalidates the claim.

```text
   BENCHMARK
   task | items | labels | split | metric
        |
        v
   CONTAMINATION
   items or paraphrases in training?
        |
        v
   COMPARABILITY
   same task, split, and metric?
        |
        v
   SELECTION
   how many submissions, how much tuning?
        |
        v
   DRIFT
   does the task still match the decision?
        |
        v
   CLAIM
   capability | safety | utility | deployment
```

**The construct rule.** A benchmark measures the construct its task and metric operationalise. A knowledge exam measures exam performance. A leaderboard measures leaderboard performance. Neither measures clinical utility.

**The contamination rule.** If the test items, the reference answers, or close paraphrases were in training, the score includes memorisation. The size of the effect depends on the contaminated fraction and the difficulty gap between contaminated and clean items.

**The comparability rule.** A comparison is valid only if the task, the split, the preprocessing, and the metric are the same. A different split is a different benchmark.

**The selection rule.** A leaderboard selects the maximum of many noisy estimates. The winner's score overstates the winner's skill, and the effect grows with the number of submissions and the metric variance.

**The drift rule.** A benchmark can stop measuring the decision when the population, the workflow, the labels, or the technology changes. A high score on a drifted benchmark is not evidence.

**The claim rule.** A benchmark supports a claim about the benchmark. Clinical utility requires a prospective or decision-analytic study.

| Filter | Question | Evidence | Failure |
|---|---|---|---|
| Contamination | Were the items in training? | Overlap and probes | Inflated score |
| Comparability | Is this the same benchmark? | Split and preprocessing | Invalid comparison |
| Selection | How many tries? | Submission count | Winner's curse |
| Drift | Does it match the decision? | Task audit | Irrelevant score |
| Claim | What does it support? | Claim map | Overclaim |

## 4. Definitions and vocabulary

### 4.1 Benchmark terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Benchmark | A fixed task, dataset, and metric. | Treated as clinical validation | Specification |
| Construct | The capability intended to be measured. | Left implicit | Definition |
| Item | One task instance. | Counted as a patient | Item |
| Split | The partition into train, validation, and test. | Changed without notice | Specification |
| Test set | Data held out from all model selection. | Reused for tuning | Split |
| Metric | The scoring function. | Treated as the decision | Function |
| Leaderboard | A public ranking. | Treated as deployment evidence | Ranking |
| Saturation | Top models cluster within noise. | Reported as a record | Interval overlap |
| Task drift | The task no longer matches the decision. | Ignored | Drift |
| Benchmark retirement | Withdrawing a benchmark from use. | Ignored | Status |

### 4.2 Contamination terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Contamination | Test items in training data. | Checked only by exact match | Proportion |
| Near-duplicate | A paraphrase or close variant. | Ignored | Similarity |
| Canary | A marker inserted to detect leakage. | Omitted | Probe |
| Temporal holdout | Items published after the cutoff. | Not constructed | Split |
| Private holdout | Items never released. | Replaced by public test | Split |
| Memorisation | Reproduction of training text. | Confused with reasoning | Behaviour |
| Contaminated fraction | The share of test items in training. | Estimated loosely | Proportion |
| Clean estimate | The estimate on uncontaminated items. | Not reported | Metric |

### 4.3 Selection and statistics terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Winner's curse | The maximum of noisy estimates overstates skill. | Ignored | Bias |
| Selection inflation | The gap between the winner and the true mean. | Not quantified | Metric points |
| Multiple comparisons | Testing many hypotheses. | Uncorrected | Count |
| Confidence interval | Uncertainty around an estimate. | Omitted | Interval |
| Seed variance | Variation across random seeds. | Ignored | Standard deviation |
| Submission count | The number of leaderboard entries. | Not reported | Count |
| Public-private gap | Difference between public and private scores. | Ignored | Metric points |
| Overfitting | Tuning to the test set. | Confused with model capacity | Behaviour |

### 4.4 Drift and governance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Task drift | The task no longer matches the decision. | Ignored | Status |
| Population drift | The case mix changed. | Treated as model decay | Distribution |
| Label drift | The label definition changed. | Treated as improvement | Definition |
| Benchmark governance | Who maintains and retires a benchmark. | Absent | Role |
| Retirement note | A record of withdrawal and replacement. | Omitted | Document |
| Version hash | A fingerprint of the benchmark. | Not recorded | Hash |
| Audit trail | The evidence behind a decision. | Informal | Record |
| Claim boundary | What the evidence supports. | Implicit | Statement |

### 4.5 Claim terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Capability | What a model can do on a task. | Treated as utility | Claim |
| Safety | Whether a model avoids harm. | Inferred from accuracy | Claim |
| Utility | Whether a model improves a decision. | Inferred from a benchmark | Claim |
| Deployment readiness | Whether a model can be used. | Inferred from a leaderboard | Claim |
| External validity | Whether a result generalizes. | Assumed from a public set | Claim |
| Decision curve | Net benefit across thresholds. | Replaced by AUROC | Analysis |
| Prospective study | Evaluation in real use. | Replaced by a benchmark | Design |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: reconstruct the benchmark, audit contamination and comparability, audit selection and drift, and decide whether to keep or retire it.

### Phase A — Reconstruct the benchmark

**Step 1. Identify the benchmark and version.**
Input: the paper, leaderboard, or vendor claim.
Output: name, version, release date, and item count.
Rationale: a result without a version is not reproducible.
Branch: if the version is unknown, treat the result as unverified.
Edge case: a leaderboard can change without a version number.

**Step 2. Reconstruct the task and the metric.**
Input: the benchmark documentation.
Output: task definition, metric, and operating point.
Rationale: the task and metric define the construct.
Branch: if the metric is an average across tasks, report the weighting.
Edge case: a metric can be computed differently across implementations.

**Step 3. Reconstruct the split.**
Input: the split specification and the item IDs.
Output: train, validation, and test counts and group keys.
Rationale: the split determines what the test set measures.
Branch: if the split is random by item, check for patient or group leakage.
Edge case: a public test set can be reused across papers.

**Step 4. Reconstruct the claim.**
Input: the paper's or vendor's statement.
Output: the claim labelled as capability, safety, utility, or deployment.
Rationale: the claim determines the evidence needed.
Branch: if the claim is utility or deployment, a benchmark is insufficient.
Edge case: a headline can claim more than the results section supports.

### Contamination probes

| Probe | What it detects | Strength | Limitation |
|---|---|---|---|
| Exact n-gram | Copied text | Simple | Misses paraphrases |
| Near-duplicate search | Paraphrases | Broad | False positives |
| Date holdout | Older training data | Clean | Requires dates |
| Canary insertion | Marker leakage | Direct | Only the marker |
| Paraphrase rewrite | Fragility and memorisation | Informative | Changes difficulty |
| Option reorder | Position bias | Cheap | Not contamination |
| Private holdout | Tuning overfit | Strong | Requires new items |

A probe is evidence, not proof. The audit reports the probe, the result, and the residual uncertainty.

### Split comparability checklist

| Item | Must match | Common difference |
|---|---|---|
| Task definition | Same labels and classes | Label set changed |
| Item set | Same version | Items added or removed |
| Split | Same grouping | Random versus patient-level |
| Preprocessing | Same pipeline | Normalisation fitted differently |
| Metric | Same implementation | Averaging differs |
| Operating point | Same threshold rule | Threshold tuned on test |
| External data | Same allowance | Pretraining allowed in one |

If any row differs, the comparison is a contrast between two benchmarks, not a ranking.

### Selection arithmetic

| Teams | Metric sd | Expected best | Inflation |
|---:|---:|---:|---:|
| 10 | 0.02 | 0.725 | +0.025 |
| 50 | 0.02 | 0.745 | +0.045 |
| 200 | 0.02 | 0.765 | +0.065 |
| 50 | 0.05 | 0.812 | +0.112 |

The expected maximum rises with the number of tries and with the metric variance. A leaderboard with many teams and a noisy metric produces a large inflation.

### Drift audit

| Drift type | What changes | Detection | Response |
|---|---|---|---|
| Population | Case mix | Subgroup distributions | Re-weight or re-collect |
| Workflow | Decision timing | Process audit | Re-define the task |
| Label | Definition or adjudication | Label audit | Re-adjudicate |
| Technology | Device or model | Version audit | Re-baseline |
| Guideline | Clinical standard | Guideline review | Re-specify |
| Data source | Sampling frame | Source audit | Re-weight |

A benchmark can be valid and drifted at the same time. The audit asks whether it still measures the decision, not whether it is internally consistent.

### Saturation check

| Signal | Interpretation | Action |
|---|---|---|
| Top five within the CI | Ranking is noise | Stop ranking; use a harder task |
| Ceiling above 95% | Task too easy | Add hard items |
| No improvement across editions | Task exhausted | Retire for ranking |
| Improvements only on a subset | Subgroup saturation | Report by subgroup |
| Gains vanish on a private set | Overfitting | Trust the private set |

Saturation is not a failure of the models; it is a failure of the benchmark to discriminate.

### Reporting checklist

| Item | What to report | Why |
|---|---|---|
| Benchmark and version | Name and date | Reproducibility |
| Task and metric | Definition | Construct |
| Split | Grouping and counts | Comparability |
| Contamination | Probes and results | Validity |
| Selection | Teams and submissions | Inflation |
| Interval | Confidence interval | Uncertainty |
| Subgroups | Performance by group | Hidden failure |
| Claim | What the result supports | Boundary |

### Benchmark lifecycle

| Stage | Question | Artefact |
|---|---|---|
| Design | What construct and decision? | Specification |
| Release | What is public and what is private? | Split and terms |
| Adoption | Who uses it and for what claim? | Claim map |
| Monitoring | Is it saturating or drifting? | Audit report |
| Retirement | When should it stop being used? | Retirement note |

A benchmark is not permanent infrastructure. It has a lifecycle, and the retirement decision is part of responsible use.

### Benchmark audit worksheet

| Check | Pass | Fail | Evidence |
|---|---|---|---|
| Version pinned | Hash recorded | Unknown version | Release note |
| Split grouped | Patient-level | Image-level | ID overlap |
| Contamination bounded | Below threshold | Above threshold | Probe report |
| Metric matches decision | Decision metric | Proxy only | Metric memo |
| Selection reported | Team count | Not reported | Leaderboard |
| Interval reported | CI present | Point only | Results table |
| Subgroups reported | Stratified | Aggregate only | Results table |
| Governance current | Maintained | Abandoned | Project page |

### Common audit findings

| Finding | Frequency | Severity | Action |
|---|---|---|---|
| Test set reused | High | Medium | Discount the ranking |
| Image-level split | High | High | Re-split by patient |
| Report-derived labels | High | High | Audit the label |
| Near-duplicate contamination | Medium | High | Bound the fraction |
| Public-private gap | Medium | Medium | Trust the private score |
| Saturated benchmark | Medium | Medium | Stop ranking |
| Undisclosed training data | Medium | High | Treat as unknown |
| Metric mismatch | High | High | Add a decision metric |

The frequency column is an impression from published audits, not a measured rate. The severity and the action are the useful parts.

A finding is not a verdict. The audit combines findings into a claim boundary: which claims survive, which are weakened, and which are prohibited. That boundary is the deliverable.

A benchmark audit is not a criticism of the benchmark's authors. It is a check that the instrument is being used within its range, the way a thermometer is not used to measure blood pressure.

The audit should be repeatable: the same evidence and criteria should produce the same boundary. A judgement that cannot be repeated is an opinion.

The audit is a service to the decision, not an academic exercise.

The audit is repeatable and the boundary is stated.

### Phase B — Audit contamination and comparability

**Step 5. Collect the contamination evidence.**
Input: the model's training data description and cutoff.
Output: a list of sources and dates.
Rationale: contamination is possible if the test items predate the cutoff and are public.
Branch: if the training data are undisclosed, the contamination risk is unknown.
Edge case: a model can be trained on a derivative of the benchmark.

**Step 6. Run exact overlap checks.**
Input: the test items and a pretraining proxy.
Output: exact matches and their sources.
Rationale: exact overlap is the easiest contamination to detect.
Branch: if matches are found, exclude or flag the items.
Edge case: a match in a public corpus does not prove the model saw it.

**Step 7. Run near-duplicate checks.**
Input: the items and an embedding or n-gram similarity.
Output: near-duplicate candidates.
Rationale: paraphrases can be memorised.
Branch: if near-duplicates are common, the benchmark is contaminated for that model family.
Edge case: clinical text has standard phrases; high similarity is not always contamination.

**Step 8. Compute the contaminated fraction and the clean estimate.**
Input: the flagged items and the scores on flagged and clean subsets.
Output: the contaminated share and the score on clean items.
Rationale: the clean estimate is the generalizing number.
Branch: if the clean subset is small, report a wide interval.
Edge case: contamination can be concentrated in easy items, which inflates more.

**Step 9. Audit split comparability.**
Input: two results and their split specifications.
Output: a comparability verdict.
Rationale: different splits make the comparison invalid.
Branch: if the splits differ, the comparison is a contrast, not a ranking.
Edge case: the same split name can hide a different preprocessing.

### Phase C — Audit selection and drift

**Step 10. Count the submissions.**
Input: the leaderboard metadata.
Output: the number of entries and teams.
Rationale: selection inflation grows with the number of tries.
Branch: if the count is unavailable, assume many and discount the ranking.
Edge case: a team can submit under multiple accounts.

**Step 11. Estimate selection inflation.**
Input: the number of teams, the metric variance, and the true mean.
Output: the expected maximum and the inflation.
Rationale: the winner's score is a maximum, not a mean.
Branch: if the inflation is larger than the gap, the ranking is noise.
Edge case: correlated submissions reduce the effective number of tries.

**Step 12. Compute the public-private gap.**
Input: the public and private scores.
Output: the gap and its interpretation.
Rationale: a large gap indicates overfitting to the public set.
Branch: if the gap is large, trust the private score and the interval.
Edge case: the private set can be small, which widens its interval.

**Step 13. Audit task drift.**
Input: the deployment decision and the benchmark task.
Output: a drift verdict.
Rationale: a benchmark can stop measuring the decision.
Branch: if the population, workflow, or label changed, the benchmark is drifted.
Edge case: a benchmark can be updated while the claim stays the same.

### Phase D — Decide keep or retire

**Step 14. Apply the retirement criteria.**
Input: the audit evidence.
Output: a keep, caveat, or retire recommendation.
Rationale: a benchmark that cannot support a claim should not be used for one.
Branch: use the retirement table below.
Edge case: a benchmark can be kept for a narrow capability claim and retired for a product claim.

**Step 15. State the supported claims.**
Input: the audit.
Output: a list of claims the benchmark can and cannot support.
Rationale: the boundary is the deliverable.
Branch: if no claim survives, retire the benchmark for that purpose.
Edge case: a capability claim can survive while a safety claim does not.

**Step 16. Design the replacement evaluation.**
Input: the claim and the available data.
Output: a private holdout, a temporal split, or a prospective study.
Rationale: retirement requires a replacement.
Branch: if no replacement exists, the claim is deferred.
Edge case: a replacement can be a decision-analytic analysis on the same data.

**Step 17. Write the audit memo.**
Input: the evidence and the recommendation.
Output: a memo with the version, the contamination, the comparability, the selection, the drift, and the claims.
Rationale: the memo is the decision artefact.
Branch: if the evidence is incomplete, state the uncertainty.
Edge case: a regulator may require a specific format.

**Step 18. Re-audit on version, model, or claim change.**
Input: the new release and the new claim.
Output: an updated audit.
Rationale: benchmarks and models change.
Branch: re-audit when the benchmark version, the model checkpoint, or the claim changes.
Edge case: a leaderboard can change without notice.

### Retirement criteria

| Criterion | Threshold | Action | Exception |
|---|---|---|---|
| Contamination | Cannot bound below 10 percent | Retire for capability claims | Keep for a narrow, audited subset |
| Saturation | Top five within the confidence interval for two editions | Retire for ranking | Keep for regression testing |
| Task drift | Decision or population changed | Retire for the product claim | Keep for historical comparison |
| Label validity | Material error found | Retire until re-adjudicated | Use a corrected subset |
| Access or licence | No longer permits the use | Retire | None |
| Split integrity | Patient or group leakage found | Retire the split | Rebuild with grouping |

The thresholds are policy choices. The important discipline is to set them before the audit, not after seeing the result.

### Failure severity

| Finding | Severity | Claim affected | Action |
|---|---|---|---|
| Exact contamination | High | Capability | Exclude items |
| Near-duplicate contamination | High | Capability | Bound and report |
| Patient leakage | High | Generalization | Re-split |
| Different metric | High | Comparability | Recompute |
| Leaderboard overfitting | Medium | Ranking | Discount |
| Saturation | Medium | Ranking | Stop ranking |
| Task drift | High | Utility | Re-specify |
| Small test set | Medium | Any | Report interval |

Severity depends on the claim. A finding that is fatal for a product claim can be acceptable for a methods claim.

### Claim-to-evidence table

| Claim | Benchmark evidence | Still needed |
|---|---|---|
| State-of-the-art capability | Rank on a clean benchmark | External replication |
| Clinical safety | Safety-specific evaluation | Red-teaming and review |
| Clinical utility | None | Decision analysis or trial |
| Deployment readiness | None | Prospective workflow study |
| Fairness | Subgroup metrics | Access and label audit |
| Robustness | Stress tests | Prospective drift monitoring |

### Contamination arithmetic

| Contaminated share | Contaminated accuracy | Clean accuracy | Reported | Inflation |
|---:|---:|---:|---:|---:|
| 10% | 95% | 70% | 72.5% | +2.5 |
| 20% | 95% | 70% | 75.0% | +5.0 |
| 40% | 95% | 70% | 80.0% | +10.0 |

The reported score is a weighted average. The clean accuracy is the estimate that generalizes, and the inflation is the price of contamination.

### Edge cases that change the procedure

**Licence forbids commercial use.** A benchmark licensed for research cannot be the primary evidence for a product. Use it for methods and find a commercial-permissive or private evaluation.

**Benchmark withdrawn.** A withdrawn benchmark should not be cited as current. Record the retirement and use the replacement.

**Credentialing takes months.** A private holdout or a prospective study takes time. Plan it before the claim is needed.

**Version drift.** A new benchmark version changes items or scoring. A score from one version is not comparable with another.

**Label errors discovered.** Re-adjudicate a sample and report the error. Do not relabel only the test set.

**Benchmark contamination.** If the items are in training, the score is not a clean capability estimate. Replace the test set or bound the contamination.

## 6. Worked example

### The case

A hospital system is evaluating a vendor's chest radiograph triage product. The vendor reports a 0.93 area under the receiver operating characteristic curve (AUROC) on a well-known public benchmark and cites a top leaderboard position. The hospital's evaluation committee has 10 weeks and a limited budget.

**Decision statement.** "Decide whether the vendor's evidence supports a pilot in our emergency departments; a pilot costs about $120,000 and six months, and a wrong choice exposes patients to a tool that does not work in our population."

**Claim.** The vendor claims benchmark superiority. The hospital needs evidence of local performance.

### Candidate evidence

| Evidence | Source | What it measures | Access | Why it is on the list |
|---|---|---|---|---|
| Public benchmark | Vendor paper | Benchmark AUROC | Public | The headline claim |
| Leaderboard | Challenge site | Rank on a hidden test | Public | The competitive claim |
| Vendor validation | Vendor report | Internal test | Under NDA | The vendor's own evidence |
| Private holdout | Hospital data | Local performance | Internal | The decision-relevant evidence |
| Partner evaluation | Two hospitals | Prospective silent mode | Contract | The strongest evidence |

### Gate scores for the pilot decision

| Gate | Public benchmark | Leaderboard | Vendor validation | Private holdout | Partner |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 1 | 1 | 1 | 2 | 2 |
| 2 Population fit | 1 | 1 | 1 | 2 | 2 |
| 3 Access fit | 2 | 2 | 1 | 2 | 0 |
| 4 Licence fit | 1 | 1 | 1 | 2 | 2 |
| 5 Benchmark fit | 0 | 0 | 1 | 2 | 2 |
| Veto | benchmark | benchmark | none | none | access |

**Public benchmark, Gate 5.** The benchmark's labels are report-derived, the test set has been public for years, and the split is image-level. The benchmark cannot support a local performance claim. Score 0, veto.

**Leaderboard, Gate 5.** A leaderboard rank is a selection maximum on a reused test set. It cannot support a deployment claim. Score 0, veto.

**Vendor validation, Gate 5.** The vendor's internal test may be clean and is not independently audited. Score 1. It is supporting evidence, not the decision.

**Private holdout, Gate 5.** A local patient-level holdout with adjudicated labels can estimate local performance. Score 2, admissible.

**Partner, Gate 3.** A prospective silent-mode evaluation is the strongest evidence and cannot be completed in 10 weeks. Score 0 for the timeline, retained for the pilot.

### The two diagnostics

**Diagnostic 1: contamination.** The benchmark's items have been public since before the vendor model's training cutoff. An n-gram and near-duplicate audit flags about 20 percent of the test items as overlapping with public corpora. If contaminated items are answered at 95 percent and clean items at 70 percent, the reported 75 percent overstates clean performance by 5 points. The audit cannot prove the model saw the items, and it bounds the risk.

**Diagnostic 2: split comparability.** The vendor's split is image-level, and about 30 percent of test patients also appear in training. A patient-level re-split reduces the vendor's reported AUROC and changes the rank. The comparison with other vendors, who may use a different split, is invalid.

A third diagnostic is task drift. The benchmark's label is "pneumothorax mentioned in the report," and the product's decision is "prioritise a radiograph for review." The label is a proxy, and the benchmark does not measure the triage decision.

### The selected route

The admissible set is the private holdout and the vendor validation. The plan uses the private holdout for the local estimate, the vendor validation for supporting evidence, and the partner for the pilot's prospective evaluation.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Contamination audit | Overlap and probe | Evaluation team | 2 weeks | Staff |
| Split audit | Patient overlap and re-split | Evaluation team | 2 weeks | Staff |
| Local holdout | 1,000 studies, adjudicated labels | Radiology | 4 weeks | About $20,000 |
| Vendor validation review | Under NDA | Procurement | 2 weeks | Legal |
| Pilot design | Silent mode and metrics | Clinical | 2 weeks | Staff |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Audits | Contamination and split | 4 weeks | Staff |
| Local holdout | Labels and evaluation | 6 weeks | About $20,000 |
| Pilot | Prospective silent mode | 6 months | About $120,000 |
| Total to a decision | Audits plus holdout | 10 weeks | About $25,000 |

The audits are cheap relative to the pilot and change the decision. That is the argument for doing them first.

### Substitute plan if the local holdout is not ready

If the labels are not ready in 10 weeks, the committee can defer the pilot, run a smaller holdout, or require the vendor to provide a patient-level external validation. Each option has a cost; none justifies using the public benchmark as the decision evidence.

### Vary one assumption: contamination is 40 percent

If the audit flags 40 percent of items, the inflation rises from 5 to 10 points at the same per-subset accuracies. The benchmark's 75 percent becomes 80 percent, and the clean estimate remains 70 percent. The vendor's apparent advantage over a 72 percent baseline disappears. The decision changes from "promising" to "unproven."

If the variation is the split, a patient-level re-split can reduce the vendor's AUROC by several points and reorder the leaderboard. If the variation is the label, an adjudicated label set can change the prevalence and the operating point.

### Conditions under which the case would fail

- **The local holdout is too small.** The interval is wide, and the pilot decision is deferred.
- **The vendor refuses patient-level validation.** The claim remains unverified.
- **The benchmark is retired before the audit.** The comparison loses its reference point.
- **The label audit finds material error.** The local estimate becomes uncertain.
- **The pilot is funded before the audit.** The audit becomes a justification, not a decision input.
- **The private holdout leaks into vendor tuning.** The estimate inflates.

The last condition is the most common in practice: the vendor is given the holdout for "validation," tunes to it, and returns an optimistic number. The holdout must stay private.

### Gate evidence log

| Gate | Public benchmark | Vendor validation | Private holdout |
|---|---|---|---|
| Decision fit | Proxy label | Proxy label | Local label |
| Population fit | Not local | Vendor population | Local population |
| Access fit | Public | NDA | Internal |
| Licence fit | Research terms | NDA | Internal |
| Benchmark fit | Contaminated, reused | Audited | Clean |

### Decision timeline

| Week | Audit | Local data | Vendor | Deliverable |
|---:|---|---|---|---|
| 1 | Overlap and probes | Label sampling | NDA review | Audit plan |
| 2 | Split re-analysis | Reader training | Validation review | Findings |
| 4 | Contamination bound | Labeling | Clarifications | Interim memo |
| 6 | Selection analysis | Holdout ready | Final answers | Local estimate |
| 8 | Drift audit | Evaluation | — | Draft decision |
| 10 | Retirement decision | — | — | Committee memo |

### What the case demonstrates

- The headline benchmark was contaminated and reused, and the reported number overstated clean performance by 5 points.
- The split was image-level, and half the test patients appeared in training. The reported AUROC was not a patient-level estimate.
- The local holdout was cheap relative to the pilot and changed the decision.
- The leaderboard rank was a selection maximum, not a measure of skill.
- The label was a proxy for the triage decision, and the benchmark did not measure the decision.
- The audit produced a claim boundary, not a single verdict.

### Sensitivity of the decision

| Change | Effect | Plan response |
|---|---|---|
| Contamination rises to 40% | Inflation doubles | Treat the claim as unproven |
| Local holdout confirms performance | Evidence strengthens | Proceed to pilot |
| Local holdout fails | Vendor claim does not transport | Do not pilot |
| A cleaner benchmark is released | Re-audit | Update the memo |
| The vendor provides patient-level validation | Evidence improves | Review and verify |

### Decision timeline

| Week | Audit | Local data | Vendor | Deliverable |
|---:|---|---|---|---|
| 1 | Overlap and probes | Label sampling | NDA review | Audit plan |
| 2 | Split re-analysis | Reader training | Validation review | Findings |
| 4 | Contamination bound | Labeling | Clarifications | Interim memo |
| 6 | Selection analysis | Holdout ready | Final answers | Local estimate |
| 8 | Drift audit | Evaluation | — | Draft decision |
| 10 | Retirement decision | — | — | Committee memo |

### What the case demonstrates

- The public benchmark was contaminated and reused, and the reported number overstated clean performance by 5 points.
- The image-level split leaked half the test patients, so the reported AUROC was not a patient-level estimate.
- The leaderboard rank was a selection maximum, not a measure of skill.
- The local holdout was cheap relative to the pilot and changed the decision.
- The label was a proxy for the triage decision, and the benchmark did not measure the decision.
- The audit produced a claim boundary, not a single verdict.

### Sensitivity of the decision

| Change | Effect | Plan response |
|---|---|---|
| Contamination rises to 40% | Inflation doubles | Treat the claim as unproven |
| Local holdout confirms performance | Evidence strengthens | Proceed to pilot |
| Local holdout fails | Claim does not transport | Do not pilot |
| A cleaner benchmark is released | Re-audit | Update the memo |
| The vendor provides patient-level validation | Evidence improves | Review and verify |

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement the contamination correction, the selection-inflation estimate, and a split-overlap check.

### 7.1 Contamination correction

```python
N = 1000
for contaminated_share in (0.10, 0.20, 0.40):
    c = int(N * contaminated_share)
    clean = N - c
    acc_c, acc_clean = 0.95, 0.70
    overall = (c * acc_c + clean * acc_clean) / N
    inflation = overall - acc_clean
    print(f"Contaminated {contaminated_share:.0%}: {c} items, reported {overall:.1%}, clean {acc_clean:.1%}, inflation {inflation:+.1%}")
print()
print("The clean estimate is the number that generalizes; the reported number includes memorisation.")
```

Expected output:

```text
Contaminated 10%: 100 items, reported 72.5%, clean 70.0%, inflation +2.5%
Contaminated 20%: 200 items, reported 75.0%, clean 70.0%, inflation +5.0%
Contaminated 40%: 400 items, reported 80.0%, clean 70.0%, inflation +10.0%

The clean estimate is the number that generalizes; the reported number includes memorisation.
```

The arithmetic assumes the contaminated and clean accuracies are known. In practice they are estimated from the flagged and unflagged subsets, and the estimate carries uncertainty.

### 7.2 Selection inflation

```python
import random
random.seed(7)
teams = 50
trials = 2000
true_mean, sd = 0.70, 0.02
maxima = []
for _ in range(trials):
    scores = [random.gauss(true_mean, sd) for _ in range(teams)]
    maxima.append(max(scores))
expected_max = sum(maxima) / len(maxima)
print(f"Teams: {teams}")
print(f"True mean: {true_mean:.2f}, sd: {sd:.2f}")
print(f"Expected best of {teams}: {expected_max:.3f}")
print(f"Selection inflation: {expected_max - true_mean:+.3f}")
print()
print("With many teams and a noisy metric, the winner's score overstates the true skill.")
```

Expected output:

```text
Teams: 50
True mean: 0.70, sd: 0.02
Expected best of 50: 0.745
Selection inflation: +0.045

With many teams and a noisy metric, the winner's score overstates the true skill.
```

The simulation assumes independent scores. Correlated submissions reduce the effective number of tries and the inflation.

### 7.3 Split overlap check

```python
records = [
    ("p01", "train"), ("p01", "test"), ("p02", "train"),
    ("p03", "test"), ("p04", "train"), ("p04", "test"),
    ("p05", "test"),
]
train = {p for p, s in records if s == "train"}
test = {p for p, s in records if s == "test"}
print(f"Train patients: {sorted(train)}")
print(f"Test patients:  {sorted(test)}")
print(f"Overlap:        {sorted(train & test)}")
print(f"Share of test patients leaked: {len(train & test)/len(test):.0%}")
print()
print("A patient-level split moves every record for a leaked patient to one side.")
```

Expected output:

```text
Train patients: ['p01', 'p02', 'p04']
Test patients:  ['p01', 'p03', 'p04', 'p05']
Overlap:        ['p01', 'p04']
Share of test patients leaked: 50%
```

Half the test patients appear in training. The reported test performance is not a patient-level generalization estimate.

## 8. Data, standards, and regulatory context

Benchmark validity is addressed by reporting guidelines, risk-of-bias tools, and regulatory expectations. The table separates binding requirements from conventions.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting guideline | Prediction-model reporting | Cover generative models explicitly |
| PROBAST+AI | Appraisal tool | Risk of bias and applicability | Produce a score that replaces judgement |
| CONSORT-AI | Reporting guideline | AI intervention trials | Validate a benchmark |
| DECIDE-AI | Reporting guideline | Early clinical evaluation | Replace a trial |
| STARD-AI | Reporting guideline | Diagnostic accuracy studies | Certify a model |
| FDA device guidance | Binding when applicable | Validation evidence and labeling | Certify fairness |
| EU AI Act | Binding, phased | Risk management and documentation | Define a benchmark |
| HIPAA and GDPR | Binding | Data protection | Define a metric |

**A reporting guideline is not an audit.** It tells you what to report; it does not tell you whether the benchmark is contaminated. The audit is a separate activity.

**A regulatory clearance is not a benchmark result.** A cleared device has met a regulatory standard for its intended use. It has not necessarily been compared with a benchmark, and its benchmark result is not the basis for the clearance.

### Benchmark governance questions

| Question | Why it matters | Owner |
|---|---|---|
| Who maintains the benchmark? | Abandonment risk | Benchmark owner |
| How are items added or removed? | Comparability | Maintainer |
| How is contamination monitored? | Validity | Maintainer |
| When is it retired? | Relevance | Governance group |
| How are labels adjudicated? | Ceiling | Annotation lead |
| How are disputes handled? | Trust | Governance group |

A benchmark without governance is a file, not an instrument.

### Reporting items for a benchmark result

| Item | What to report | Why |
|---|---|---|
| Benchmark and version | Name, release, hash | Reproducibility |
| Task and metric | Definition and implementation | Construct |
| Split | Grouping, counts, and preprocessing | Comparability |
| Contamination | Probes and results | Validity |
| Selection | Teams and submissions | Inflation |
| Interval | Confidence interval | Uncertainty |
| Subgroups | Performance by group | Hidden failure |
| Claim | What the result supports | Boundary |

## 9. Industry practice and case evidence

### What organisations actually do

- **Audit before citing.** Mature teams treat a public benchmark as a hypothesis.
- **Prefer a private holdout.** The internal test is the one that matters for a local claim.
- **Report the split.** Patient-level splits are the default in clinical work.
- **Discount the leaderboard.** A rank is a selection maximum, not a skill estimate.
- **Retire benchmarks.** A saturated or drifted benchmark is removed from the decision.
- **Publish the audit.** The contamination and split checks are part of the evidence.
- **Separate capability from utility.** A benchmark score does not become a clinical claim.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| RSNA 2026 knee challenge | The edition closed in October 2026 with disputes over labels generated at scale and possible leakage, according to the module survey; the organiser report should be verified. | When labels are generated at scale, the label pipeline is part of the benchmark. |
| HealthBench | A clinician-written benchmark with rubrics; scores depend on the judge and the rubric. | A benchmark score is a property of the model, the rubric, and the judge. |
| MedQA | A public exam-style question set; public items can appear in pretraining. | A high score is not a clean capability estimate without a contamination audit. |
| GRAIL NHS-Galleri | A screening study reported a primary endpoint with stage shift rather than a reduction in advanced-stage incidence, as summarised in the module survey. | A proxy can improve while the endpoint does not; the endpoint decides adoption. |
| MIMIC benchmarks | Several tasks use report-derived labels and single-center data. | A benchmark can be useful for methods and weak for population claims. |

The first and fourth rows are recent and should be verified against the primary report before citation. The structural lessons are stable.

### What good looks like

- The benchmark version and hash are recorded.
- The split is patient-level and the counts are reported.
- Contamination probes are run and the clean subset is reported.
- The submission count and the public-private gap are reported.
- The metric is tied to the decision.
- Subgroup intervals are reported.
- The claim boundary is written.
- A retirement date and replacement are named when needed.

### Procurement questions

- Which benchmark version and split were used?
- Was the test set public before the model's training cutoff?
- What is the patient overlap between train and test?
- How many leaderboard submissions were made?
- What is the private score and its interval?
- What is the subgroup performance?
- What claim does the benchmark support?
- When will the benchmark be retired?

A vendor that cannot answer these has a demo, not evidence.

### Benchmark audit in practice

| Step | Time | Output | Owner |
|---|---|---|---|
| Version and split review | 1 day | Split memo | Evaluation lead |
| Contamination probes | 3 days | Probe report | Data scientist |
| Selection analysis | 1 day | Inflation estimate | Statistician |
| Drift review | 2 days | Task memo | Clinical lead |
| Claim boundary | 1 day | Decision memo | Product lead |

A week of audit can change a six-figure pilot decision. The cost is small relative to the decision.

### Why leaderboards mislead procurement

A buyer asks whether the product works locally. A leaderboard answers whether a model ranks well on a fixed test under a fixed metric. The gap is a difference in estimand, and it is closed only by a local evaluation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Contamination | High score on easy items | Test items in training | Inflated capability | Temporal or private holdout |
| Image-level split | Unusually high performance | Patient on both sides | Memorisation | Patient-level split |
| Leaderboard overfitting | Public-private gap | Repeated tuning | Winner's curse | Trust the private score |
| Task drift | Metric no longer matches decision | Population or workflow change | Irrelevant score | Re-specify the task |
| Saturation | Top models within CI | Task too easy | Ranking noise | Harder tasks |
| Metric mismatch | AUROC for a capacity decision | Convenience | Wrong operating point | Decision metric |
| Test reuse | Same test set across papers | Publication pressure | Inflated and stale | Private holdout |
| Label error | Adjudication absent | Proxy labels | Ceiling | Re-adjudicate |
| Benchmark withdrawal | Old number still cited | No retirement process | Misleading comparison | Retirement note |
| Undisclosed training data | Vendor cannot describe data | Secrecy | Unknown contamination | Treat as unknown |
| Small test set | Wide interval | Sampling | Over-ranked models | Report interval |
| Subgroup silence | No subgroup results | Aggregate score | Hidden failure | Stratified reporting |
| Selection on the test set | Threshold tuned on test | Leakage | Optimistic performance | Pre-specify threshold |
| Benchmark shopping | Many benchmarks, one reported | Selective reporting | False positive | Pre-register the benchmark |

Red flags that should stop a benchmark claim:

- The test set is public and older than the model.
- The split unit is not stated.
- The leaderboard submission count is not reported.
- The metric is not the decision metric.
- The benchmark is saturated.
- The claim is clinical utility.
- The licence is research-only and the use is commercial.
- The benchmark has been retired and is still cited.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The test set is clean | Capability estimate depends on it | Contamination | Probes |
| The split is patient-level | Generalization depends on it | Image split | ID overlap |
| The metric matches the decision | Utility depends on it | Proxy metric | Decision analysis |
| The benchmark is current | Relevance depends on it | Drift | Task audit |
| The leaderboard is not tuned | Ranking depends on it | Many submissions | Inflation estimate |
| The labels are valid | The ceiling depends on it | Label error | Re-adjudication |
| The test set is large enough | Interval depends on it | Small set | Confidence interval |
| The subgroups are reported | Equity depends on it | Aggregate only | Stratified analysis |
| The licence covers the use | Compliance depends on it | Research-only | Clause review |
| The benchmark is not retired | Comparability depends on it | Retirement | Version check |
| The training data are disclosed | Contamination depends on it | Undisclosed | Treat as unknown |
| The claim matches the evidence | Interpretation depends on it | Overclaim | Claim map |

The deepest limit is that a benchmark measures a benchmark. It does not measure the world, the workflow, or the patient. Those require a different study.

## 12. Exercises

**Exercise 1. Audit a benchmark result.**
Deliverable: a one-page audit with version, split, contamination, selection, and claim.
Acceptance criteria: every element present; one quantified issue; a claim boundary.
Time: 45 minutes.

**Exercise 2. Compute contamination inflation.**
Deliverable: a table of reported and clean estimates at three contaminated fractions.
Acceptance criteria: arithmetic correct; assumptions stated; a decision implication.
Time: 35 minutes.

**Exercise 3. Detect a non-comparable split.**
Deliverable: a split comparison and a verdict on comparability.
Acceptance criteria: grouping, counts, and preprocessing compared; a clear verdict.
Time: 40 minutes.

**Exercise 4. Quantify selection inflation.**
Deliverable: an expected-maximum estimate for a stated team count and metric variance.
Acceptance criteria: method stated; inflation compared with the reported gap.
Time: 35 minutes.

**Exercise 5. Decide to retire a benchmark.**
Deliverable: a retirement memo applying the criteria and naming the replacement.
Acceptance criteria: criteria met; replacement named; date and owner.
Time: 45 minutes.

**Exercise 6. Map a claim to evidence.**
Deliverable: a claim-to-evidence table for a vendor's benchmark claim.
Acceptance criteria: claim labelled; missing evidence named; a prohibited claim.
Time: 35 minutes.

**Exercise 7. Design a clean evaluation.**
Deliverable: a private holdout, a pre-specified metric, and a contamination plan.
Acceptance criteria: patient-level split; metric matches decision; probes named.
Time: 50 minutes.

**Assessment note.** Exercises 2, 3, 4, and 6 produce checkable artefacts: an inflation table, a split comparison, a selection estimate, and a claim map. Exercises 1, 5, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write an audit memo.**
Deliverable: a two-page memo with findings, claim boundary, and recommendation.
Acceptance criteria: specific evidence; one quantified issue; a keep/retire recommendation.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why is a benchmark result a claim about the benchmark?**
Because the score is computed on the benchmark's items, split, and metric. It does not automatically transfer to another population, workflow, or label definition.

**Item 2. What is contamination?**
Test items or close paraphrases present in training data. It inflates the score by an amount that depends on the contaminated fraction and the difficulty gap.

**Item 3. How do you compute the clean estimate?**
Score the contaminated and clean subsets separately, then report the clean-subset score with an interval. The overall score is a weighted average and is not the generalizing estimate.

**Item 4. When is a comparison invalid?**
When the task, item set, split, preprocessing, metric, or external-data allowance differs. A different split is a different benchmark.

**Item 5. What is selection inflation?**
The gap between the best of many noisy estimates and the true mean. It grows with the number of tries and the metric variance.

**Item 6. What is task drift?**
A change in the population, workflow, label, or technology that makes the benchmark stop measuring the decision.

**Item 7. When is a benchmark saturated?**
When the top models cluster within the confidence interval for successive editions, so the ranking is noise.

**Item 8. What are the retirement criteria?**
Unbounded contamination, saturation, task drift, invalid labels, lost access or licence, or a broken split.

**Item 9. Why is a private holdout valuable?**
It limits tuning against the test set and provides a cleaner estimate of generalization. It does not repair label error or population shift.

**Item 10. What claim can a benchmark support?**
A capability claim on the benchmark's task and population. It cannot support clinical utility or deployment readiness without a separate study.

**Item 11. Why is test-set reuse a validity problem even without contamination?**
Repeated evaluation on one set turns it into a tuning signal. The reported score reflects selection on that set, not generalization to new data.

**Item 12. What is the difference between construct and criterion validity?**
Construct validity asks whether the benchmark measures the intended concept; criterion validity asks whether it predicts a relevant outcome. A benchmark can have one and not the other.

**Item 13. Why is a benchmark with 100 items unable to support a fine ranking?**
The confidence interval is wide, often several percentage points, so small differences are not resolvable. Ranking requires more items or a paired design.

**Item 14. What is the difference between a public and a private leaderboard?**
A public leaderboard is visible during the competition; a private one is hidden until the end. The private score is less exposed to tuning.

**Item 15. Why is pre-registration useful in benchmark evaluation?**
It fixes the benchmark, metric, and analysis before results, which prevents benchmark shopping and post hoc selection.

**Item 16. What is the difference between robustness and generalization?**
Robustness is performance under perturbation; generalization is performance on new data from the target distribution. A robust model can fail to generalize and vice versa.

**Item 17. Why can a higher benchmark score mean a worse product?**
If the metric does not match the decision, optimizing it can worsen the operating point, calibration, or subgroup performance. A strong answer maps the metric to the decision.

**Rubric for the design exercises (2, 3, 7, 8).** Full marks require a quantified contamination bound, a split audit, a pre-specified metric, and a claim boundary. Partial marks for a complete structure without quantification. No marks for a leaderboard rank presented as clinical evidence.

## 14. Interview questions

**1. A vendor reports a top leaderboard rank. What do you ask?**
The benchmark version, the split, the submission count, the contamination checks, the private score, and the claim. A strong answer asks for the audit. A weak answer accepts the rank.

**2. How do you detect contamination?**
Exact and near-duplicate checks, date holdouts, canary strings, and paraphrase probes. A strong answer reports the contaminated fraction. A weak answer checks exact matches only.

**3. Why can two papers on the same benchmark be incomparable?**
Different splits, preprocessing, metrics, or external-data allowances. A strong answer names the difference. A weak answer compares the headline numbers.

**4. What is winner's curse in a leaderboard?**
The maximum of many noisy estimates overstates the winner's skill. A strong answer quantifies the inflation. A weak answer treats the rank as skill.

**5. When should a benchmark be retired?**
When contamination cannot be bounded, it saturates, it drifts, its labels are invalid, or its access or licence changes. A strong answer names the replacement. A weak answer keeps using it.

**6. Why is AUROC often the wrong metric?**
It is insensitive to the operating point and to prevalence. A fixed-capacity decision needs precision or sensitivity at capacity. A strong answer maps the metric to the decision. A weak answer uses AUROC because it is standard.

**7. How do you evaluate a benchmark's claim of clinical utility?**
You do not, from the benchmark. You need a decision analysis or a prospective study. A strong answer separates capability from utility. A weak answer infers utility from accuracy.

**8. What is a private holdout?**
A test set never released, used to limit tuning and contamination. A strong answer notes its cost. A weak answer assumes the public test set is enough.

**9. Why is an image-level split a problem?**
The same patient can appear on both sides, so the test measures memorisation. A strong answer proposes a patient-level split. A weak answer trusts the official split.

**10. What is the difference between a benchmark and a quality measure?**
A benchmark compares methods on a task; a quality measure assesses care against a standard. A strong answer names the different purposes. A weak answer treats them as interchangeable.

**11. How do you report a negative benchmark result?**
Report the benchmark, the split, the interval, the contamination, and the claim boundary. A strong answer distinguishes absence of evidence from evidence of absence. A weak answer reports only the mean.

**12. When would you stop using a benchmark?**
When it can no longer discriminate or support the claim, and a replacement exists. A strong answer has a retirement rule. A weak answer keeps it because it is familiar.

**18. When is a benchmark result still useful despite contamination?**
When the contaminated fraction is bounded, the clean subset is reported, and the claim is limited to that subset. The result is useful with a caveat, not as a headline.

**19. Why is a temporal holdout often the best practical option?**
It uses items published after the model cutoff, which reduces contamination without requiring a new data collection. It requires reliable dates and enough items.

**20. What is the first question about a benchmark claim?**
What is the claim: capability, safety, utility, or deployment. A strong answer then asks whether the benchmark measures that. A weak answer starts with the score.

**21. How do you report a retired benchmark?**
State the retirement, the reason, the date, and the replacement. A strong answer removes it from current comparisons. A weak answer keeps citing the old number.

**22. What is the difference between a benchmark and a leaderboard?**
A benchmark is the task, data, and metric; a leaderboard is a public ranking on a benchmark. A leaderboard adds selection and tuning effects.

**23. Why is a subgroup analysis important in a benchmark audit?**
An average can hide a subgroup failure. A strong answer reports subgroup intervals. A weak answer reports the aggregate.

**24. What is the difference between internal and external validity?**
Internal validity is whether the comparison is sound within the study; external validity is whether it generalizes. A benchmark can be internally valid and externally limited.

**25. Why is a benchmark with a hidden test set still vulnerable to overfitting?**
Teams can tune against the public leaderboard or the validation split. The hidden test reduces, and does not eliminate, selection effects.

**26. What is the difference between label drift and population drift?**
Label drift is a change in the definition or adjudication of the target; population drift is a change in the people being scored. Both degrade a benchmark, and they require different responses.

## 15. Key terms

- **Benchmark** — a fixed task, dataset, and metric.
- **Benchmark retirement** — withdrawing a benchmark from use.
- **Canary** — a marker inserted to detect leakage.
- **Capability** — what a model can do on a task.
- **Clean estimate** — the score on uncontaminated items.
- **Comparability** — whether two results use the same benchmark.
- **Confidence interval** — uncertainty around an estimate.
- **Construct** — the capability intended to be measured.
- **Contamination** — test items present in training.
- **Contaminated fraction** — the share of test items in training.
- **Decision curve** — net benefit across thresholds.
- **Deployment readiness** — whether a model can be used.
- **Drift** — change that makes a benchmark irrelevant.
- **External validity** — whether a result generalizes.
- **Item** — one task instance.
- **Label error** — incorrect reference labels.
- **Leaderboard** — a public ranking.
- **Metric** — the scoring function.
- **Near-duplicate** — a paraphrase or close variant.
- **Overfitting** — tuning to the test set.
- **Private holdout** — items never released.
- **Public-private gap** — difference between public and private scores.
- **Saturation** — top models cluster within noise.
- **Selection inflation** — the winner's advantage over the true mean.
- **Split** — the partition into train and test.
- **Task drift** — the task no longer matches the decision.
- **Temporal holdout** — items published after the cutoff.
- **Test set** — data held out from model selection.
- **Utility** — whether a model improves a decision.
- **Winner's curse** — the maximum of noisy estimates overstates skill.
- **Benchmark shopping** — reporting the benchmark that looks best.
- **Ceiling effect** — scores near the maximum.
- **Data leakage** — test information in training.
- **Evaluation harness** — the code that runs a benchmark.
- **External replication** — an independent repetition.
- **Floor effect** — scores near the minimum.
- **Generalization gap** — the difference between test and deployment.
- **Held-out set** — data not used in development.
- **Item exposure** — how widely items are known.
- **Leaderboard hacking** — exploiting the evaluation.
- **Metric gaming** — optimizing the metric without the task.
- **Model selection** — choosing a model using validation data.
- **Multiple testing** — repeated comparisons.
- **Out-of-distribution** — data unlike the training set.
- **Overfitting to the benchmark** — tuning to the test set.
- **Pre-registration** — specifying the analysis before results.
- **Public leaderboard** — a visible ranking.
- **Private leaderboard** — a hidden ranking.
- **Reproducibility** — the ability to rerun a result.
- **Robustness** — performance under perturbation.
- **Selection bias** — distortion from the selection process.
- **Statistical power** — the ability to detect an effect.
- **Stress test** — an adversarial evaluation.
- **Test-set reuse** — repeated evaluation on one set.
- **Training cutoff** — the date training data end.
- **Validation set** — data used for tuning.
- **Variance** — variability of an estimate.
- **Benchmark validity** — whether the benchmark measures the construct.
- **Construct validity** — whether the measure matches the concept.
- **Criterion validity** — whether the measure predicts an outcome.
- **Ecological validity** — whether the setting matches reality.
- **Adversarial evaluation** — testing with deliberately hard inputs.
- **Benchmark card** — documentation of a benchmark.
- **Blind evaluation** — scoring without knowing the model.
- **Calibration** — agreement between predicted and observed risk.
- **Cross-validation** — repeated splits for estimation.
- **Data shift** — change in the input distribution.
- **Evaluation protocol** — the pre-specified procedure.
- **Ground truth** — the reference label.
- **Holdout leakage** — information from the holdout entering development.
- **Inference cost** — compute per prediction.
- **Label noise** — error in the reference.
- **Model card** — documentation of a model.
- **Operating point** — the selected threshold.
- **Paired test** — a comparison on the same items.
- **Performance ceiling** — the best attainable score.
- **Ranking instability** — the order changes under resampling.
- **Reference standard** — the best available truth.
- **Sampling error** — error from observing a sample.
- **Score variance** — variability of a score.
- **Selection pressure** — the incentive to tune.
- **Statistical significance** — a result unlikely under the null.
- **Test harness** — the code that runs the evaluation.
- **Threshold tuning** — selecting a threshold.
- **Training data** — data used to fit the model.
- **Validation strategy** — the plan for model selection.
- **Benchmark drift** — change in the benchmark over time.
- **Contamination audit** — the check for test leakage.
- **Deployment gap** — the difference between test and use.
- **Evaluation bias** — distortion in the evaluation.
- **Generalization** — performance on new data.
- **Replication crisis** — widespread failure to reproduce.
- **Score inflation** — a score higher than the true capability.
- **Test fairness** — comparable performance across groups.
- **Uncertainty quantification** — reporting intervals.
- **Benchmark governance** — who maintains and retires a benchmark.
- **Data provenance** — the origin of the data.
- **Evaluation transparency** — disclosing the method.
- **Model selection bias** — bias from choosing among models.
- **Outcome measure** — the variable being predicted.
- **Benchmark fatigue** — declining use after saturation.
- **Blinded scoring** — scoring without model identity.
- **Ceiling saturation** — scores too close to the maximum.
- **Confirmation bias** — preferring results that fit.
- **Data availability** — whether data can be obtained.
- **Evaluation cost** — the cost of running the benchmark.
- **External benchmark** — a benchmark from another domain.
- **False discovery** — a spurious significant result.
- **Holdout contamination** — the holdout entering development.
- **Internal benchmark** — a private benchmark.
- **Metric leakage** — the metric using test information.
- **Model family** — a group of related models.
- **Negative control** — a test expected to show no effect.
- **Overfitting gap** — train-test performance difference.
- **Positive control** — a test expected to show an effect.
- **Publication bias** — positive results more likely published.
- **Reference implementation** — the official code.
- **Reproducibility package** — code, data, and environment.
- **Score distribution** — the spread of scores.
- **Selective reporting** — reporting only favourable results.
- **Test collection** — the process of building a test set.
- **Test leakage** — test information in training.
- **Training-serving skew** — different pipelines in training and serving.
- **Validation leakage** — validation information in training.
- **Benchmark inflation** — rising scores without capability gains.
- **Benchmark monoculture** — everyone using one benchmark.
- **Task ambiguity** — an underspecified task.
- **Annotation drift** — changing annotation rules.
- **Label leakage** — the label present in features.
- **Evaluation suite** — a set of benchmarks.
- **Scoreboard** — a summary of results.
- **Benchmark audit** — a structured validity review.
- **Claim mapping** — linking a claim to its evidence.
- **Contamination bound** — an upper estimate of contaminated share.
- **Data versioning** — identifying a dataset release.
- **Evaluation plan** — the pre-specified evaluation.
- **False negative** — a missed positive.
- **False positive** — a wrongly flagged negative.
- **Generalization error** — error on new data.
- **Holdout set** — data reserved for final evaluation.
- **In-distribution** — data like the training set.
- **Label quality** — accuracy of the reference labels.
- **Metric sensitivity** — how the metric responds to change.
- **Model selection bias** — bias from choosing among models.
- **Operating characteristic** — performance across thresholds.
- **Positive predictive value** — the probability a positive is true.
- **Reference label** — the label used as truth.
- **Sampling bias** — distortion from the sample.
- **Score ceiling** — the maximum attainable score.
- **Selection effect** — distortion from choosing the best.
- **Statistical power** — the ability to detect an effect.
- **Test coverage** — the range of cases in the test set.
- **Training set** — data used to fit.
- **Validation set** — data used for tuning.
- **Benchmark report** — the documented result.
- **Evaluation error** — error in the evaluation process.
- **Metric validity** — whether the metric measures the goal.
- **Outcome validity** — whether the outcome is correct.
- **Population validity** — whether the population matches.
- **Temporal validity** — whether the time period matches.
- **Setting validity** — whether the setting matches.
- **Benchmark transparency** — disclosure of method.
- **Community benchmark** — a benchmark maintained by a community.
- **Private benchmark** — a benchmark held by one organisation.
- **Public benchmark** — a benchmark released publicly.
- **Challenge benchmark** — a competition benchmark.
- **Leaderboard metric** — the metric used for ranking.
- **Submission limit** — a cap on entries.
- **Test server** — the system that scores submissions.
- **Score variance** — variability of scores.
- **Benchmark adoption** — how widely a benchmark is used.
- **Accuracy** — the share of correct predictions.
- **Area under the curve** — discrimination across thresholds.
- **Calibration slope** — agreement across the risk range.
- **Class imbalance** — unequal class frequencies.
- **Confusion matrix** — counts of predicted and actual classes.
- **Cross-validation split** — repeated train-validation partitions.
- **Data dictionary** — field definitions.
- **Evaluation bias** — distortion in scoring.
- **External validation** — evaluation on independent data.
- **Feature leakage** — target information in features.
- **Internal validation** — evaluation on held-out development data.
- **Label distribution** — the frequency of each class.
- **Metric definition** — the exact formula.
- **Model checkpoint** — a saved model state.
- **Negative predictive value** — the probability a negative is true.
- **Prevalence** — the proportion positive.
- **Sensitivity** — the true positive rate.
- **Specificity** — the true negative rate.
- **Test-retest reliability** — stability across runs.
- **Threshold** — the decision cut-off.
- **Training cutoff** — the end date of training data.
- **Validation curve** — performance across a parameter.
- **Benchmark suite** — a collection of benchmarks.
- **Benchmark registry** — a list of benchmarks and versions.
- **Evaluation report** — the documented evaluation.
- **Model comparison** — a structured comparison of models.
- **Reference implementation** — the official evaluation code.
- **Score normalization** — rescaling scores for comparison.
- **Task specification** — the formal task definition.
- **Test protocol** — the procedure for running the test.
- **Validation protocol** — the procedure for model selection.
- **Benchmark audit trail** — the evidence behind the audit.

## 16. Sources

All URLs accessed 26 October 2026. Benchmark versions and scores change; verify at the source.

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
2. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
3. CONSORT-AI and SPIRIT-AI guidance. https://www.equator-network.org/
4. DECIDE-AI reporting guideline. https://www.equator-network.org/
5. US Food and Drug Administration. AI-enabled device software functions guidance, draft January 2025. https://www.fda.gov/media/184856/download
6. European Parliament and Council. Regulation (EU) 2024/1689 (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
7. Radiological Society of North America. AI challenges. https://www.rsna.org/artificial-intelligence
8. OpenAI. HealthBench. https://openai.com/index/healthbench/
9. MedQA repository. https://github.com/jind11/MedQA
10. PhysioNet. MIMIC-IV. https://physionet.org/content/mimiciv/
11. Module 21 survey file, section 17. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, tool stack: [`11_tool-stack.md`](11_tool-stack.md)
- Text and LLM benchmarks: [`08_text-and-llm-benchmarks.md`](08_text-and-llm-benchmarks.md)
- The five-gate decision: [`01_five-gate-decision.md`](01_five-gate-decision.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the benchmark validity audit: [`../../labs/`](../../labs/)

