# Reproducibility

> Module 18 — Quality, Evaluation, and Reporting
> Topic 05 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A result is reproducible when a different team, using the same data and code, obtains the same
answer. It is repeatable when the same team, on the same setup, obtains the same answer. The
distance between those two definitions is where most industrial quality problems live, because a
team can repeat its own analysis indefinitely while no one else can reproduce it.

The problem is measurable. McDermott and colleagues analysed 511 machine learning papers and
found that about 20 percent of machine learning for health papers released code publicly,
compared with about 38 percent in computer vision and 48 percent in natural language processing.
A 2026 AI-assisted scoping review of 3,967 articles citing TRIPOD or TRIPOD+AI found code-sharing
statements in 12.2 percent, rising from about 4.5 percent in 2016 to about 15 percent in
2024-2025. Among the repositories that were accessible, only about 52 percent had adequate
READMEs, 38 percent specified dependencies, 35 percent had licences, and 3.9 percent had tests.
Availability is not reproducibility, and reproducibility is not replication.

The cost is not academic. A payer asks for the analysis behind a submitted performance claim and
the team cannot rerun it because the environment was not pinned. A regulator asks why the
validation table changed between the submission and the supplement and no one can reconstruct
which run produced which number. A hospital adopts a model, the vendor retrains it, and no one
can demonstrate that the new version performs like the validated one. A journal reviewer requests
the code, receives a notebook with absolute paths and a missing data file, and the authors spend
three weeks reconstructing work they had already done.

The cost also compounds. An analysis that takes three weeks to reconstruct in year one takes
longer in year three, when the analyst has left and the library versions are two major releases
behind. Organisations that treat reproducibility as a project-end task pay this cost repeatedly
for the same analysis, and the payments are invisible because they are charged to the people who
inherit the work rather than to the people who created it. A submission that requires a
reconstruction is not a one-time delay; it is a signal that every future question about that
analysis will require the same reconstruction.

There is a second, subtler cost. An analysis that cannot be reproduced cannot be audited, and an
analysis that cannot be audited cannot be trusted at the level required for clinical use. The
trust that a regulator, a buyer, or a clinician extends is not trust in the team's competence; it
is trust in the artifact's verifiability. Reproducibility is the property that makes that trust
possible.

This topic teaches the four levels of reproducibility, the stack of artifacts that determines the
level, the procedure for auditing someone else's analysis, and the packaging practices that make
an analysis reproducible by default. It builds on the evidence chain: reproducibility is the test
of link D, the link between the analysis and the reported number.

One clarification before the definitions. Reproducibility is not the same as honesty, and a
failure to reproduce is not an accusation of misconduct. Most irreproducible analyses were
performed carefully by competent people who did not know that the environment, the seed, or the
data snapshot would matter to someone else later. The topic is written in that spirit: the
failures are systemic, the fixes are procedural, and the goal is an artifact that survives its
authors.

Three audiences need different things from this topic. A data scientist needs the packaging
practices so that their own work survives review. A quality or regulatory professional needs the
audit procedure and the level vocabulary so that a claim of reproducibility can be tested rather
than accepted. A buyer or manager needs the three questions that separate a package from an
archive, and the language to make them a contract condition. The topic is written so that all
three can use the same procedure with different stopping points.

A note on scope. This topic does not teach software engineering. It teaches the minimum
engineering needed for a biomedical analysis to survive review, audit, and staff turnover, and it
teaches how to audit an analysis you did not write.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** repeatable, reproducible, replicable, and generalisable, and place a given
   study at the correct level.
2. **Inventory** the artifacts that determine reproducibility: data, code, environment,
   configuration, seeds, hardware, and outputs.
3. **Build** a pinned environment and a data manifest, and explain why each is necessary.
4. **Audit** a repository against a reproducibility checklist and assign a level with evidence.
5. **Compare** two runs and decide whether a difference is within tolerance, a defect, or a
   different result.
6. **Diagnose** a discrepancy to its cause: dependency version, seed, data version, hardware
   nondeterminism, or code change.
7. **Package** an analysis so that a stranger can run it on a clean machine.
8. **Write** a reproducibility statement that does not overclaim.

## 3. The core mental model

Reproducibility is a ladder with four rungs, and each rung requires more than the one below it.

```text
  generalisable   different population, same methods, consistent conclusion
        ^
  replicable      different data, same methods, consistent conclusion
        ^
  reproducible    same data, same code, different team, same result
        ^
  repeatable      same team, same setup, same result
```

The rungs are not a quality ranking of the science. A study can be highly generalisable and not
reproducible, if the methods are simple and the populations varied but the code is lost. A study
can be perfectly reproducible and not replicable, if the data are idiosyncratic. The ladder
describes what a claim has been shown to survive, not how good the claim is.

Beneath the ladder is a stack. Every analysis is a function of seven layers, and a change in any
layer can change the output.

```text
  7  outputs        tables, figures, metrics, model weights
  6  code           scripts, notebooks, pipeline definitions
  5  configuration  hyperparameters, thresholds, paths, flags
  4  runtime        language version, libraries, system libraries
  3  operating system and hardware   CPU, GPU, drivers, math libraries
  2  data           version, snapshot, preprocessing, splits
  1  question       cohort definition, outcome definition, estimand
```

Two properties of the stack matter.

**The bottom layers dominate.** A change in the cohort definition (layer 1) or the data snapshot
(layer 2) usually changes the result more than any change above it. Teams often focus on code and
environment because those are the layers they control, while the largest reproducibility risk
sits in the data and the question.

**Determinism is layered, not binary.** A pipeline can be bitwise deterministic on one machine
and nondeterministic across machines, because floating-point reductions, GPU kernels, thread
scheduling, and library implementations differ. The practical target is not bitwise identity
everywhere; it is a stated tolerance within which the result is considered the same, combined
with a check that the conclusion does not depend on the difference. A model whose AUROC moves
from 0.842 to 0.837 across runs is reproducible at the level of the conclusion and not at the
level of the digit. That distinction should be stated, not hidden.

A third model is useful for audits: the "clean room" test.

```text
  artifacts in  ->  clean machine  ->  documented commands  ->  outputs  ->  comparison
       |                  |                    |                  |             |
   manifest           no cached state      no undocumented      hashes        tolerance
   and access         no analyst memory     manual steps         and diffs     and conclusion
```

The clean room is the operational definition of reproducible. If the analysis needs the person
who wrote it to answer a question during the run, it is repeatable at best.

A fourth model explains why teams underinvest. The benefit of reproducibility accrues to whoever
audits the analysis later, while the cost is paid by whoever builds it now. The builder and the
auditor are usually different people, often in different years. This is a classic externality,
and it is why reproducibility does not improve through exhortation. It improves through a gate:
a release condition that makes the cost unavoidable at the point where the benefit is realised,
which is the submission, the sale, or the deployment. The gate does not need to be elaborate. It
needs to be enforced. A workable gate is binary, checkable by someone other than the author, and
cheap enough that teams do not route around it. A gate that takes a week will be waived; a gate
that takes an hour will be used.

A fifth model is the failure taxonomy, which is more useful than a checklist because it points to
the remedy. Reproducibility failures fall into five families: missing artifacts (the code or data
does not exist), inaccessible artifacts (they exist but cannot be obtained), unstable artifacts
(the same artifact produces different results), undocumented artifacts (the run cannot be
reconstructed), and misdescribed artifacts (the documentation does not match the artifact). Each
family has a different fix: create, negotiate access, control determinism, document, or reconcile.
A team that treats every failure as "we need better documentation" will not fix the first three.

The taxonomy also explains why the ladder in section 3 has four rungs rather than two. Missing and
inaccessible artifacts cap a study at repeatable. Unstable artifacts cap it at statistically
reproducible. Undocumented and misdescribed artifacts prevent any level above repeatable for a
stranger, even when the code runs on the author's machine. The rung is determined by the worst
family present, which is the same minimum rule used for the evidence chain in topic 01.

Finally, the ladder is not a maturity model for organisations. A team can be excellent at
reproducibility and weak at replication, because the two require different investments: the first
is engineering discipline, the second is access to diverse data. A quality programme that
measures only one will optimise for the one it measures.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Level or unit |
|---|---|---|---|
| Repeatability | Same team, same setup, same result | Called reproducibility | Level 1 |
| Reproducibility | Different team, same data and code, same result | Called replication | Level 2 |
| Replicability | Different data, same methods, consistent conclusion | Called reproducibility | Level 3 |
| Generalisability | Different population or setting, consistent conclusion | Assumed from replication | Level 4 |
| Computational reproducibility | The pipeline reruns and produces the reported outputs | Confused with scientific validity | Property of an artifact |
| Bitwise reproducibility | Identical bytes in outputs | Treated as the only acceptable standard | Strictest form |
| Statistical reproducibility | Same conclusion within a stated tolerance | Treated as weaker and therefore invalid | Practical standard |
| Environment lock | Exact record of language and library versions | A requirements file with unpinned versions | File |
| Container | Packaged runtime including system libraries | Assumed to solve data access | Image |
| Manifest | List of files with hashes and sizes | A directory listing | File |
| Hash | Fixed-length digest of content | Assumed to prove correctness | SHA-256 |
| Seed | Value initialising a pseudo-random generator | Assumed to guarantee determinism | Integer |
| Nondeterminism | Variation across runs from uncontrolled sources | Blamed on "the GPU" without diagnosis | Property |
| Data snapshot | Immutable copy of data at a point in time | A live database connection | Artifact |
| Data version | Identifier for a snapshot | A filename with a date | Identifier |
| Pipeline | Ordered set of steps from raw data to outputs | A notebook run top to bottom | Artifact |
| DAG | Directed acyclic graph of pipeline steps | Used loosely for any script | Structure |
| Provenance | Record of how an output was produced | A log file | Metadata |
| Artifact | Any input or output of the pipeline | Only the model weights | Object |
| Tolerance | Acceptable numerical difference between runs | Absence of a difference | Threshold |
| Code review | Independent reading of code before use | Assumed from testing | Process |
| Regression test | A test that a known input yields a known output | A unit test of a function | Test |
| Data use agreement | Contract governing data access | Assumed to permit redistribution | Legal |
| Synthetic data | Artificially generated data preserving some structure | Treated as equivalent to real data | Artifact |

Four distinctions carry the topic.

**A note on the word "validation".** In machine learning, "validation" often means the held-out
split used for tuning. In clinical evidence, "validation" means an independent evaluation of
performance. The two uses collide in reproducibility discussions, because a pipeline can
faithfully reproduce a "validation" that was in fact a tuning set. The auditor should ask what
the split was used for, not only how it was drawn.

**Reproducible versus replicable.** Reproducibility holds the data fixed and varies the team.
Replicability holds the methods fixed and varies the data. A study can be reproducible and not
replicable: the code runs, the numbers match, and the finding does not hold in a new cohort. The
two failures have different remedies. A reproducibility failure is an engineering problem; a
replication failure is a scientific finding.

**Bitwise versus statistical.** Bitwise identity is achievable for deterministic code on a fixed
platform and is often impossible across platforms. Requiring it everywhere produces two bad
outcomes: teams spend effort on irrelevant determinism, or they give up and claim nothing can be
reproduced. The practical standard is a pre-specified tolerance on the reported quantities, plus
a check that the conclusion is stable within it.

**Environment versus data.** A container pins the runtime but not the data. Many "reproducibility
packages" ship a Dockerfile and a script that expects a database the reader cannot access. The
package is reproducible in principle and unusable in practice. The data access path is part of
the artifact.

**Availability versus usability.** A repository that exists but cannot be run is available, not
reproducible. The scoping review's finding that only 52 percent of repositories had adequate
READMEs is the measurement of this gap. A README that lists commands, inputs, expected outputs,
and runtime is not documentation overhead; it is the difference between a repository and an
archive.

## 5. The mechanics, step by step

This procedure audits an analysis for reproducibility and produces a level assignment with
evidence. It takes one to three days for a small pipeline and one to three weeks for a full
model development. The same steps, run in authoring mode, produce the reproducibility package.

### Step 1. State the claim to be reproduced

**Input:** the paper, report, or submission, and the specific number or conclusion under test.

**Output:** a claim record: the quantity, its value, the table or figure, and the population.

**Rationale:** "Reproduce the study" is not a testable objective. "Reproduce the AUROC of 0.842
in Table 2 for the held-out cohort" is. The claim determines which outputs must match and which
can vary.

**Branch point:** if the claim is a conclusion rather than a number ("the model is clinically
useful"), the audit must first operationalise it into quantities. Record the operationalisation
and have the author confirm it, because a mismatch here invalidates the audit.

### Step 2. Inventory the artifacts

**Input:** the repository, supplement, data access statement, and any correspondence.

**Output:** an artifact inventory with seven layers: question definition, data, environment,
code, configuration, outputs, and documentation.

**Rationale:** The inventory determines what can be tested. A missing data layer means the audit
can test code reproducibility only, which is a weaker claim, and the report must say so rather
than implying a full test.

**Branch point:** three outcomes.

| Inventory state | What can be tested | Level assignable |
|---|---|---|
| Data, code, environment, config present | Full pipeline | Reproducible |
| Code and config present, data restricted | Code runs on synthetic or substitute data | Partially reproducible |
| Code absent, method described | Nothing computational | Repeatable at best |

### Step 3. Obtain access and record the terms

**Input:** the data use agreement, the repository licence, and the access request.

**Output:** an access record: what was obtained, under what terms, on what date, and what cannot
be shared onward.

**Rationale:** Access is the most common failure point and the least discussed. A data use
agreement that permits analysis but not redistribution makes reproducibility possible for a
licensed auditor and impossible for the public. That distinction must be stated, because
"available on request" is not the same as "available".

**Branch point:** if access is denied, record the denial and its reason. The audit then assesses
the code on substitute data and states the limit.

### Step 4. Build the environment from the lock file

**Input:** the environment specification.

**Output:** a running environment on a clean machine, with versions recorded.

**Rationale:** The environment is the layer most often unpinned. A requirements file that says
`scikit-learn` without a version is not a specification. The build should be performed on a
machine that has no cached packages and no prior installation, because that is the condition a
stranger faces.

**Branch point:** if the environment cannot be built, record the error and the missing
dependency. Do not patch silently; a patch changes the artifact under test.

### Step 5. Freeze the data and verify the snapshot

**Input:** the data source and the manifest.

**Output:** a verified snapshot with hashes matching the author's manifest, or a documented
difference.

**Rationale:** Data change. A live database queried in 2026 is not the database queried in 2021.
The snapshot is what makes the analysis a fixed object. Hash verification is the only cheap way
to know that the data used are the data described.

**Branch point:** if the hash does not match, stop and diagnose before running. A data mismatch
makes every downstream comparison uninterpretable, and a run on the wrong data produces a
discrepancy that cannot be attributed to any other layer.

**Edge case, conflicting reference standards:** if the pipeline can compute the outcome under two
definitions, record which one the reported number used. A reproducibility audit that reproduces
the wrong definition produces a false discrepancy.

### Step 6. Run the pipeline as documented

**Input:** the README, the configuration, and the snapshot.

**Output:** a run log with commands, durations, warnings, and outputs.

**Rationale:** The run must follow the documentation, not the auditor's knowledge. Every manual
step the auditor has to invent is a documentation defect, and it should be recorded as such
rather than absorbed.

**Branch point:** if the pipeline requires an undocumented step, record it and continue. The
number of undocumented steps is a reproducibility metric.

**Edge case, post-hoc changes:** if the configuration contains values that differ from the paper
(for example, a threshold), record the difference and run both. The paper's number may correspond
to one of them; which one is a finding.

### Step 7. Compare outputs against the reported values

**Input:** the run outputs and the published tables.

**Output:** a comparison table with the reported value, the reproduced value, the difference, and
the tolerance.

**Rationale:** Comparison requires a stated tolerance before the run. Choosing the tolerance after
seeing the difference is a way of guaranteeing a pass. The tolerance should be justified by the
precision of the reported number and by the decision it informs.

**Branch point:** four outcomes, and the fourth is the one that requires judgement.

| Difference | Interpretation | Action |
|---|---|---|
| Within rounding tolerance | Reproduced | Record tolerance and environment |
| Small, conclusion unchanged | Statistically reproducible | Record and explain source |
| Large, conclusion unchanged | Reproduced in direction only | Investigate before accepting |
| Changes the conclusion | Not reproduced | Diagnose; do not report as a pass |

**What counts as a match.** The comparison needs a rule for each output type. A table of metrics
can be compared numerically with a tolerance. A figure can be compared by the underlying data,
not by the image. A model can be compared by predictions on a fixed test set, which is the most
informative comparison because it captures every upstream difference. A hash comparison is
appropriate only for files that should be byte-identical, such as a configuration or a
preprocessing output; applying it to a metrics table with a timestamp will always fail.

A practical hierarchy of comparisons, from strongest to weakest:

| Comparison | Detects | Cost | Limitation |
|---|---|---|---|
| Prediction-level equality | Any upstream change | High: requires saved predictions | Needs the test set |
| Metric equality within tolerance | Changes that move the metric | Low | Misses offsetting changes |
| Table equality | Most reporting differences | Medium | Formatting noise |
| Figure inspection | Gross differences | Very low | Not reproducible |
| Narrative comparison | Almost nothing | None | Impressionistic |

The audit should use the strongest comparison the artifacts permit, and state which one it used.
An audit that compares figures and declares a match has tested almost nothing.

### Step 8. Diagnose any discrepancy

**Input:** the comparison table and the run logs.

**Output:** a cause attribution: data, code, environment, configuration, seed, or hardware.

**Rationale:** A discrepancy without a cause is not an audit finding; it is an observation. The
diagnosis is what makes the finding actionable. The common causes are a different library
version, a different data snapshot, a seed that was not fixed, a preprocessing step applied in a
different order, and a threshold that was tuned interactively.

**Branch point:** if the cause cannot be identified, record "unresolved" and the experiments
attempted. An unresolved discrepancy is itself a finding about the artifact.

### Step 9. Test determinism

**Input:** the pipeline and a fixed environment.

**Output:** two or more runs with the same inputs, and a measure of variation.

**Rationale:** Some pipelines are nondeterministic by design (GPU reductions, multithreaded
aggregation). The audit must measure the variation rather than assume it away. If two runs on the
same machine differ, the tolerance must be widened and the conclusion checked.

**Branch point:** if variation exceeds the tolerance, the reported number is not a stable
quantity. That is a finding regardless of whether the mean matches.

**The order of diagnosis.** Test the layers from the bottom up, because a change in a lower layer
masks the effect of changes above it. The order that works in practice:

1. Data: confirm the snapshot hash and the row counts.
2. Question: confirm the cohort definition and the outcome definition.
3. Split: confirm the seed and the split unit.
4. Configuration: compare every parameter with the reported value.
5. Code: diff the version against the released tag.
6. Runtime: pin the library versions to the paper's year and rerun.
7. Hardware: rerun on the same hardware class, then on a different one.

Testing in this order isolates one layer at a time. Testing the environment first, which is the
common instinct, produces a pass or a fail that is confounded by the split, because the split
affects the metric far more than the library version usually does.

### Step 10. Write the audit report

**Input:** the inventory, the run log, the comparison, the diagnosis, and the determinism test.

**Output:** a report with the claim, the level, the evidence, the discrepancies, and the
recommendation.

**Rationale:** The report is the artifact that a decision-maker uses. It should state the level
plainly: "reproducible at the level of the code and the conclusion, not at the level of the
digit, because the data are restricted and the GPU reduction is nondeterministic."

**Branch point:** if the audit cannot assign a level, say so and state what would be required.

### Step 11. Package for the next auditor

**Input:** the audit experience.

**Output:** a reproducibility package: manifest, lock file, container or build script, README with
commands, expected outputs, and a data access path.

**Rationale:** The auditor's experience is the best specification for the package, because every
undocumented step the auditor had to invent is a step the package must document.

**Branch point:** if data cannot be redistributed, include a synthetic dataset with the same
schema and a script that runs end to end on it. The synthetic run demonstrates that the code
works; it does not demonstrate that the result reproduces, and the README must say so.

### Step 12. Re-test on a clean machine and assign the level

**Input:** the package.

**Output:** a second clean-room run and a final level assignment.

**Rationale:** The package is not reproducible until someone other than its author has run it
from scratch. The second run is the acceptance test.

**Branch point:** if the second run fails, the package is not complete, regardless of how well it
worked on the author's machine.

### Edge cases

The edge cases below are the ones that change the audit's conclusion rather than its effort.
Each is recorded as a finding with a link to the layer it affects.

**Incomplete reporting.** A missing method detail is a documentation defect. Record it and, where
possible, test the plausible alternatives. Do not fill the gap with the author's likely intent
and report a pass.

**Selective outcome reporting.** If the pipeline can produce several metrics and the paper
reports one, check whether the reported one is the pre-specified primary. A pipeline that
reproduces the reported metric but not the others is reproducible for that metric, not for the
study.

**Subgroup claims.** A subgroup result is reproducible only if the subgroup definition and the
analysis code are present. A hand-computed subgroup in a spreadsheet is not reproducible from the
pipeline.

**Protocol deviations.** If the pipeline differs from the registered analysis plan, record the
difference. The audit should reproduce both the plan and the reported analysis if they differ,
because the discrepancy is the finding.

**Post-hoc changes.** An interactively tuned threshold is a post-hoc change. The pipeline may
reproduce the final threshold but not the selection process. Record the distinction.

**Conflicting reference standards.** If the outcome can be computed two ways, run both and report
which matches the paper. A match under one definition and not the other is informative.

**Retracted studies.** If the study is retracted, stop the audit and record the status. There is
no point reproducing a withdrawn result, and continuing wastes the auditor's time.

A note on what the edge cases have in common. Each one is a place where the auditor is tempted to
supply information the artifact does not contain. Supplying it feels helpful and produces a pass
that the next auditor cannot repeat. The discipline is to record the gap, run the plausible
alternative, and report both the result under the alternative and the fact that the reported
configuration is unknown. That is slower and more honest, and it is the only way the audit
remains reproducible itself.

**What the audit is not.** It is not a code review for correctness. A pipeline can be perfectly
reproducible and compute the wrong estimand, because reproducibility tests whether the same
inputs produce the same outputs, not whether the outputs answer the question. The audit should
state this limit in the report, because a reader who sees "reproducible" may infer "correct". The
appraisal in topic 03 is the tool for correctness; this topic is the tool for verifiability.

### Recording the audit

The report has six parts, and the order matters because a reader should be able to stop after
part three if they only need the verdict.

| Part | Content | Length |
|---|---|---|
| Claim | The number under test, its table, its population | Two sentences |
| Inventory | Seven layers, present or absent | One table |
| Level | The rung assigned and why | One paragraph |
| Comparison | Reported, reproduced, difference, tolerance | One table |
| Diagnosis | Cause of each discrepancy, or "unresolved" | Bullets |
| Recommendation | What to fix, and what the reader may rely on | One paragraph |

The diagnosis section is where most reports fail. A report that lists discrepancies without
causes gives the reader no way to judge whether the artifact is nearly reproducible or
fundamentally broken. An unresolved cause is acceptable if the report states the experiments
attempted and what would be needed to resolve it.

### Decision table for the final level

| Data | Code | Environment | Output match | Level |
|---|---|---|---|---|
| Same, verified | Present, runs | Pinned | Within tolerance | Reproducible |
| Same, verified | Present, runs | Pinned | Conclusion only | Reproducible in conclusion |
| Restricted | Present, runs on substitute | Pinned | Not testable | Partially reproducible |
| Different | Present | Pinned | Conclusion consistent | Replicable |
| Different population | Present | Pinned | Conclusion consistent | Generalisable |
| Unavailable | Absent | Absent | Not testable | Repeatable only |

## 6. Worked example

The case is a composite constructed for teaching: a 30-day readmission model developed on
electronic health record data at three hospitals, published with a public repository, and audited
by an external team. The numbers are constructed so that the arithmetic is checkable; the pattern
is drawn from the reproducibility literature cited in section 9. It is not a description of a
specific published study, and it should not be cited as one.

### The claim and the artifact set

The claim under test: the model achieves an area under the receiver operating characteristic curve
(AUROC) of 0.842 on a held-out test set of 9,180 admissions, at a threshold of 0.18, with
sensitivity 0.71 and specificity 0.78.

The repository contains a training script, a preprocessing script, an evaluation notebook, a
requirements file with unpinned versions, a README with three commands, and no data. The data are
available under a data use agreement that permits analysis but not redistribution. The paper
reports the cohort definition and the split, but not the random seed.

### The audit, step by step

**Inventory.** Code present and readable. Environment specified but not pinned. Data restricted.
Configuration partly embedded in the training script. Outputs: one figure and one table, no saved
predictions. Documentation: a README with commands that assume the data are in a local directory
named `data/`.

**Access.** The data use agreement was signed, and the auditor obtained the same extract with a
documented query date. The extract is 41,206 admissions before exclusions.

**Environment.** The requirements file lists `pandas`, `scikit-learn`, `xgboost`, and `numpy`
without versions. The auditor built an environment with the versions current at the time of the
audit and recorded them. The build succeeded.

**Run.** The README's three commands ran preprocessing, training, and evaluation. The
preprocessing command failed because it expected a file named `admissions.csv` that the extract
named `admissions_2021.csv`. The auditor renamed the file and recorded the defect. The training
run completed in 41 minutes on CPU.

**Comparison.** The reproduced AUROC was 0.837 against the reported 0.842, a difference of 0.005.
Sensitivity was 0.70 against 0.71, and specificity 0.78 against 0.78. The threshold in the script
was 0.18, matching the paper.

**Diagnosis.** Three candidate causes were tested in order. First, the split: the script used a
random split with a seed read from an environment variable that was not set, so the split
differed from the reported one. Setting the seed to the value found in a comment in the
evaluation notebook (`SEED = 20210417`) produced an AUROC of 0.841. Second, the library version:
pinning `scikit-learn` to the version released in the paper's year changed the AUROC from 0.841
to 0.842. Third, the remaining difference of 0.000 was within rounding.

**Determinism.** Two runs with the same seed and the same environment produced AUROC 0.842 and
0.842 on CPU. A GPU run produced 0.841, because of nondeterministic reduction order. The pipeline
is deterministic on CPU and nondeterministic at the third decimal on GPU.

**Level assignment.** Reproducible on CPU with a stated seed and pinned versions, at the level of
the reported digit; reproducible in conclusion on GPU. The data are restricted, so a member of the
public cannot reproduce the result without an access agreement. The correct level is
"reproducible for a licensed auditor", not "publicly reproducible".

### Scoring the repository against a checklist

| Item | Status | Evidence |
|---|---|---|
| Code present and licensed | Partial | Code present; no licence file |
| Environment pinned | No | Unpinned requirements |
| Data access path documented | Partial | Agreement mentioned; no request process |
| Preprocessing code present | Yes | `preprocess.py` |
| Random seed recorded | No | Seed in an environment variable, unset |
| Splits reproducible | No | Split depends on the seed |
| Hyperparameters recorded | Partial | In script, not in config |
| Evaluation code present | Yes | Notebook |
| Expected outputs documented | No | No reference values |
| Runtime documented | No | Not stated |
| Known nondeterminism documented | No | Not mentioned |
| Model weights or access | No | Not provided |

Score: 3 yes, 4 partial, 5 no, out of 12 items. A weighted score that counts yes as 1 and partial
as 0.5 gives 5 of 12, or 41.7 percent.

### Recommendation

The repository is conditionally reproducible. A licensed auditor with the same data can reproduce
the reported AUROC to the third decimal on CPU, provided the seed and library versions are
supplied by the authors. A member of the public cannot, because the data are restricted and the
repository contains no synthetic substitute. The recommendation to the authors is to pin the
environment, move the seed into a configuration file, add a licence, add expected outputs, and
publish a synthetic dataset with the same schema and a script that runs end to end on it.

The recommendation to a buyer is different. For a diligence decision, the analysis is
reproducible enough to rely on the reported performance, because an independent auditor with data
access reproduced it. For a regulatory submission, the absence of a seed and pinned versions is a
documentation gap that should be closed before the analysis is relied upon in a controlled
submission.

### The cost of the audit

The audit took two auditor-days. The time was distributed as follows: access and data extract,
half a day; environment build, two hours; first run and failure, one hour; comparison and
diagnosis, half a day; determinism test, two hours; report, three hours.

The authors' cost to fix the defects is smaller than the auditor's cost to find them, which is
the usual asymmetry. Pinning four versions, moving a seed into a configuration file, adding a
licence, and writing a README with expected outputs is perhaps half a day for someone who knows
the repository. The audit found the defects because no one had done that half-day of work.

This asymmetry is the business case for the release gate in Exercise 8. A gate that requires the
half-day before release converts a two-day external audit into a one-hour verification.

### The arithmetic of the comparison

Three quantities determine whether the audit passes.

The first is the difference: 0.842 - 0.837 = 0.005.

The second is the tolerance. It should be derived from the precision of the reported number,
which is three decimal places, and from the sampling variability of the metric. With 9,180 test
admissions and a 12 percent event rate, the standard error of an AUROC near 0.84 is roughly 0.008;
a 95 percent interval is about plus or minus 0.016. A difference of 0.005 is well inside the
sampling noise, which means the reproduction cannot distinguish the two runs on statistical
grounds even if the pipeline is identical.

The third is the decision threshold. If the deployment decision changes at an AUROC of 0.80, both
0.842 and 0.837 support the same decision, and the difference is irrelevant to the conclusion. If
the decision changes at 0.84, the difference matters and the audit must resolve it.

The audit's verdict therefore has three parts: the difference is within the stated tolerance; the
difference is smaller than the sampling error; and the conclusion is unchanged at the decision
threshold. Any one of those alone would be a weaker claim.

### Two diagnostics that would change the verdict

**Diagnostic 1: the data were publicly available under an open licence.** The level would move
from "reproducible for a licensed auditor" to "publicly reproducible", and the recommendation to
the authors would drop the synthetic-data requirement. The recommendation to the buyer would not
change, because the audit already reproduced the number. This diagnostic shows that data access
changes the level without changing the trustworthiness of the result for someone who has access.

**Diagnostic 2: the reproduced AUROC had been 0.79 instead of 0.837.** A difference of 0.05
changes the conclusion about the model's discrimination, because it crosses the range in which
the deployment decision was made. The audit would assign "not reproduced", the diagnosis would
become the central finding, and the recommendation would be to suspend reliance on the reported
performance until the cause is resolved. The level would remain "reproducible in principle" and
the result would be "does not reproduce", which is the most damaging combination: the artifact is
sufficient to test the claim, and the test fails.

### What the level does not say

The assigned level says nothing about whether the model works. A reproducible analysis of a
poorly designed study is reproducible and uninformative. The audit report should separate the two
explicitly, in one sentence: "The analysis is reproducible for a licensed auditor; whether the
reported performance is clinically meaningful is outside the scope of this audit." Without that
sentence, a reader will read "reproducible" as an endorsement.

The level also says nothing about the future. A repository that reproduces today may not
reproduce next year, because dependencies disappear, data agreements expire, and cloud services
change. A reproducibility package has a shelf life, and the README should state the date on which
it was last tested. A package that has never been tested is a hypothesis about reproducibility,
not a demonstration of it.

### Vary one assumption: the validation design

Suppose the paper had reported a temporal validation (train on 2018-2020, test on 2021) rather
than a random split. The reproducibility audit would change in two ways. First, the split would
no longer depend on a seed, so the largest source of discrepancy in this case would disappear and
the audit would be simpler. Second, the test set would be fixed by the date boundary, which makes
the result reproducible by anyone with the same extract, even without the seed. The level would
rise to "reproducible with data access" without the seed requirement, and the checklist score
would rise by two items. The lesson is that a design choice made for scientific reasons (temporal
validation) also makes the analysis easier to reproduce, because it removes an uncontrolled
random choice. Reproducibility is often a design property, not a packaging property.

## 7. Code, query, or calculation

The script builds a manifest of a directory, compares two runs, checks determinism across seeds,
and computes whether a difference in AUROC is within a stated tolerance. It uses only the
standard library.

Two implementation notes before the code. First, the comparison function reports only added,
removed, and changed files; it does not say whether a change is material. Materiality is a
judgement about the output, and the auditor makes it after inspecting the diff. Second, the
bootstrap is included to show that a difference between two metrics can be smaller than the
sampling noise, which is the quantitative reason a tolerance must be tied to precision rather
than chosen for convenience. The scores in that block are synthetic and are labelled as such.

```python
"""Reproducibility audit: manifest, run comparison, and tolerance check."""

import hashlib
import json
import os
import random


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root, exts=(".py", ".json", ".csv")):
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(exts):
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                out[rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    return out


def compare(a, b):
    keys = sorted(set(a) | set(b))
    rows = []
    for k in keys:
        if k not in a:
            rows.append((k, "added", ""))
        elif k not in b:
            rows.append((k, "removed", ""))
        elif a[k]["sha256"] != b[k]["sha256"]:
            rows.append((k, "changed", f"{a[k]['bytes']}->{b[k]['bytes']} bytes"))
    return rows


def bootstrap_auc_diff(y, s1, s2, n_boot=2000, seed=7):
    """Percentile bootstrap for the difference in AUC between two score vectors."""
    rng = random.Random(seed)
    idx = list(range(len(y)))
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in idx]
        ys = [y[i] for i in sample]
        a = [s1[i] for i in sample]
        b = [s2[i] for i in sample]
        if len(set(ys)) < 2:
            continue
        diffs.append(auc(ys, a) - auc(ys, b))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return lo, hi


def auc(y, s):
    pos = [s[i] for i in range(len(y)) if y[i] == 1]
    neg = [s[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg:
        return float("nan")
    # Rank-based AUC with ties handled by average ranks.
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


os.makedirs("run_a", exist_ok=True)
os.makedirs("run_b", exist_ok=True)
open("run_a/pipeline.py", "w").write("SEED = 20210417\nTHRESHOLD = 0.18\n")
open("run_b/pipeline.py", "w").write("SEED = 20210417\nTHRESHOLD = 0.19\n")
open("run_a/metrics.csv", "w").write("auroc,sensitivity,specificity\n0.842,0.71,0.78\n")
open("run_b/metrics.csv", "w").write("auroc,sensitivity,specificity\n0.837,0.70,0.78\n")

ma, mb = manifest("run_a"), manifest("run_b")
print("files compared:", len(ma), "vs", len(mb))
print(f"{'file':<18}{'status':<10}detail")
print("-" * 46)
for name, status, detail in compare(ma, mb):
    print(f"{name:<18}{status:<10}{detail}")

# Determinism: same seed, two runs, one metric; different seed, third run.
same_seed = [0.8412, 0.8412]
other_seed = 0.8371
spread = max(same_seed + [other_seed]) - min(same_seed + [other_seed])
print()
print(f"same-seed spread: {max(same_seed)-min(same_seed):.4f}")
print(f"seed sensitivity: {abs(same_seed[0]-other_seed):.4f}")
print(f"tolerance 0.005 exceeded by seed change: {abs(same_seed[0]-other_seed) > 0.005}")

# Is the observed difference within tolerance, and does it change the conclusion?
reported, reproduced, tol = 0.842, 0.837, 0.005
diff = round(abs(reported - reproduced), 3)
print()
print(f"reported {reported:.3f}, reproduced {reproduced:.3f}, diff {diff:.3f}")
print(f"within tolerance: {diff <= tol}")
print(f"conclusion stable at threshold 0.80: {reproduced >= 0.80 and reported >= 0.80}")

# Bootstrap CI for a difference, using synthetic scores (illustrative only).
rng = random.Random(11)
n = 400
y = [1 if rng.random() < 0.2 else 0 for _ in range(n)]
s1 = [min(1.0, max(0.0, rng.gauss(0.62 if t == 1 else 0.38, 0.15))) for t in y]
s2 = [min(1.0, max(0.0, rng.gauss(0.60 if t == 1 else 0.40, 0.15))) for t in y]
lo, hi = bootstrap_auc_diff(y, s1, s2)
print()
print(f"synthetic AUC A: {auc(y, s1):.3f}")
print(f"synthetic AUC B: {auc(y, s2):.3f}")
print(f"bootstrap 95% CI for difference: {lo:.3f} to {hi:.3f}")
print(f"CI excludes zero: {lo > 0 or hi < 0}")

with open("manifest_a.json", "w") as f:
    json.dump(ma, f, indent=2, sort_keys=True)
print()
print("manifest written:", os.path.getsize("manifest_a.json"), "bytes")
```

Expected output:

```text
files compared: 2 vs 2
file              status    detail
----------------------------------------------
metrics.csv       changed   48->48 bytes
pipeline.py       changed   35->35 bytes

same-seed spread: 0.0000
seed sensitivity: 0.0041
tolerance 0.005 exceeded by seed change: False

reported 0.842, reproduced 0.837, diff 0.005
within tolerance: True
conclusion stable at threshold 0.80: True

synthetic AUC A: 0.881
synthetic AUC B: 0.855
bootstrap 95% CI for difference: -0.032 to 0.082
CI excludes zero: False

manifest written: 255 bytes
```

Reading the output. The manifest comparison shows two changed files and no added or removed
files, which localises the difference to the pipeline configuration and the metrics. The seed
sensitivity of 0.0041 is below the tolerance of 0.005, so the reported difference of 0.005 is at
the edge of what a seed change alone can explain. That is exactly the situation in which an
auditor should not declare a pass on the number alone: the conclusion is stable, the digit is
not.

The bootstrap illustrates a statistical point. The synthetic scores produce a difference of 0.026
whose interval runs from -0.032 to 0.082, so the gap is not distinguishable from noise at this
sample size. The same logic applies to a reproducibility comparison: a difference smaller than
the sampling variability of the metric is not evidence of a reproducibility defect. An auditor
who demands an exact match on a metric with sampling error will report false failures.

The manifest size is small because the directory contains two files. In a real audit the manifest
is the artifact that lets a second auditor confirm that the inputs are the same, and it should be
committed with the outputs.

## 8. Data, standards, and regulatory context

**Reporting guidelines.** TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378) includes open science
items covering protocol availability, registration, data access, and code access. STARD-AI
includes dataset and code availability items. These are reporting requirements, not engineering
standards, but they create the disclosure obligation that makes reproducibility auditable.

**Regulatory context.** The FDA's Quality Management System Regulation, effective 2 February 2026,
incorporates ISO 13485:2016 and requires design controls, document control, and records that
demonstrate that the device was developed as described. For software, this includes configuration
management and the ability to reconstruct a released version. The FDA's draft guidance on
AI-enabled device software functions (January 2025) discusses the total product life cycle,
including data management, model development, validation, and monitoring, and it expects
documentation that supports reconstruction.

**21 CFR Part 11.** For electronic records used in regulatory decisions, the regulation requires
secure, computer-generated, time-stamped audit trails that record creation, modification, and
deletion, with old and new values retained. An analysis pipeline that overwrites outputs without
a record is not compliant with the record-keeping expectations of a regulated submission, even if
it is technically reproducible.

**ISO 13485 and configuration management.** The standard requires documented procedures for
control of documents and records, and for identifying the configuration of the product. In
practice, a model version should be identifiable by a hash, the training data by a snapshot
identifier, and the environment by a lock file. These are the same artifacts that a
reproducibility audit requests, which is why reproducibility and quality management converge.

**Data protection.** The GDPR and equivalent regimes constrain redistribution of clinical data.
A reproducible package cannot assume that the reader can obtain the data. The standard practice
is to publish the code, the schema, a synthetic dataset, and a documented access process, and to
state that the result was reproduced by a licensed auditor.

**Data standards and provenance.** Two health data standards affect reproducibility directly.
FHIR (Fast Healthcare Interoperability Resources) defines resources and versions, and a FHIR
release changes the shape of the data even when the underlying records are unchanged. OMOP
(Observational Medical Outcomes Partnership) Common Data Model defines a stable schema with
concept identifiers, and an OMOP extract is reproducible only if the vocabulary version is
recorded alongside the data. In both cases, the standard is part of the environment: a pipeline
pinned to a FHIR release or an OMOP vocabulary version is reproducible, and one that reads
whatever the server returns is not.

**Records and retention.** A regulated submission must be able to reconstruct the analysis after
the staff who performed it have left. That means retaining the code, the configuration, the
environment specification, the data snapshot or its identifier, and the outputs, for the
retention period required by the applicable regime. Retention is where reproducibility meets
records management: a pipeline that can be rebuilt only by rerunning a query against a live
system is not a record, because the system will have changed.

**Model updates under a predetermined change control plan.** A predetermined change control plan
(PCCP) describes modifications that a manufacturer intends to make after authorisation, with the
methodology for developing and validating them. Reproducibility is the property that lets a
reviewer confirm that an updated model was produced by the described process. Without a pinned
environment and a data snapshot, the PCCP's verification step cannot be performed, and the plan's
premise fails.

**What the standards do not require.** None requires bitwise reproducibility, and none requires
public data. They require that the analysis be reconstructable by someone with the appropriate
access. That is a weaker and more achievable standard, and it is the one this topic teaches.

## 9. Industry practice and case evidence

**Documented fact: code sharing is low.** The 2026 scoping review of 3,967 TRIPOD-citing articles
found code-sharing statements in 12.2 percent, with wide variation by journal (from 0 percent at
many journals to about 71 percent at *Nature Communications*). Among accessible repositories,
about 52 percent had adequate READMEs, 38 percent specified dependencies, 35 percent had
licences, and 3.9 percent had tests. The inference for practice: a repository is a starting
point, not a reproducibility package.

**Documented fact: ML-for-health lags other fields.** McDermott et al. (*Science Translational
Medicine* 2021) found about 20 percent code release in ML-for-health, against 38 percent in
computer vision and 48 percent in NLP, with about 55 percent of papers using public datasets.
The comparison is useful because it rules out the explanation that the field is young; the gap is
specific to health, where data access and regulation raise the cost.

**Documented fact: rerunning notebooks is hard.** A 2024 study in *GigaScience* reported that
only 5.3 percent of biomedical Jupyter notebooks could be rerun end to end. The finding is about
notebooks specifically, and it illustrates the difference between sharing code and sharing a
runnable analysis.

**Inference: why teams fail at reproducibility.** Three mechanisms. First, the analysis is
performed interactively and the notebook is cleaned up afterwards, so the recorded steps are not
the steps that produced the result. Second, the environment is whatever the analyst had, and no
one records it until a reviewer asks. Third, the data are accessed through a live system, so
there is no snapshot and the analysis cannot be rerun on the same records. Each is a process
problem with a cheap fix at the start and an expensive fix at the end.

**Inference: the audit as a service.** Some health systems and regulators now commission
independent reproducibility audits before accepting a model for procurement or review. The audit
is cheap relative to the decision: two days of an analyst's time against a multi-year contract.
The output is not a verdict on the vendor but a statement of what a second team could and could
not verify. That framing is important, because it keeps the audit factual and lets the buyer make
the commercial judgement with better information.

**Inference: what mature teams do.** They containerise the runtime, snapshot the data, put
configuration in version control, fix seeds, save predictions as artifacts, and run a
clean-machine test before release. They treat the reproducibility package as a deliverable with
an owner, not as an afterthought for the code appendix. The cost is a few days per project; the
benefit is that every later audit, submission, and handover starts from a working artifact.

**Inference: reproducibility as a purchasable property.** A buyer can require a clean-machine run
as a condition of contract. The requirement is cheap to state and cheap to verify: the vendor
supplies a container and a synthetic dataset, the buyer's analyst runs it on a laptop, and the
expected output matches. A vendor that cannot pass this test has an analysis that only its own
team can operate, which is a maintenance risk independent of the model's performance. The
requirement also has a second effect: it forces the vendor to document the pipeline, which makes
every later audit cheaper.

**Inference: the procurement use.** A buyer can ask three questions that separate a real package
from an archive. Can you run the pipeline from a clean machine using only the README? What is the
expected output, and does it match? What is the seed, and is it in the configuration? A vendor
that answers all three has a reproducible analysis. A vendor that answers none has a slide.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unpinned environment | `requirements.txt` without versions | Analyst used local packages | Build differs or breaks | Lock file and container |
| Missing seed | "Results may vary" | Random split or initialisation | Numbers drift across runs | Seed in config, committed |
| Live data source | Query against a production database | No snapshot | Analysis cannot be rerun | Immutable snapshot and hash |
| Notebook-only pipeline | Manual cell execution order | Interactive development | Recorded steps differ from actual | Scripted pipeline with DAG |
| Undocumented manual step | Auditor has to rename a file | Local convention | Stranger cannot run | README with exact commands |
| No expected outputs | "Run and see" | No reference values | Cannot detect silent change | Save reference outputs |
| Overwritten outputs | Single `results.csv` | No run history | Cannot compare versions | Timestamped artifacts |
| Hidden configuration | Constants in code | No config layer | Parameters untraceable | Config file in version control |
| Nondeterminism unmeasured | Different numbers on GPU | Parallel reduction | Tolerance unknown | Measure and document |
| Data redistribution assumed | "Data available on request" | Access not granted | Public cannot reproduce | Synthetic data and access process |
| No licence | Repository without terms | Legal default | Cannot reuse legally | Add a licence |
| Stale manifest | Hash mismatch after update | Data changed silently | Audit compares different inputs | Re-hash at run time |
| Metric without interval | Single number compared exactly | Sampling error ignored | False failures | Tolerance tied to precision |
| Retracted dependency | Citing a withdrawn result | No status check | Invalid analysis | Check at date of use |
| Schema drift | FHIR or OMOP version changed | Standard updated | Pipeline reads different fields | Pin the version |
| Access expires | Data agreement lapsed | Time-limited licence | Package no longer runnable | Record expiry; renew |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The claim can be operationalised | Audit needs a target | Vague conclusion | Write the quantity and confirm with author |
| The data can be obtained | Full audit needs inputs | Access denied | Audit code on synthetic data; state limit |
| The pipeline is scriptable | Clean room requires commands | Interactive analysis | Run from README only |
| A tolerance can be pre-specified | Comparison needs a rule | Metric precision unknown | Derive from sampling error |
| Determinism can be measured | Variation must be bounded | Hidden nondeterminism | Repeat runs on same and different hardware |
| The environment can be rebuilt | Stranger needs the same runtime | Proprietary system libraries | Container or documented build |
| Outputs are saved | Comparison needs artifacts | Only figures in a paper | Save predictions and metrics |
| The manifest is current | Hash check is meaningful | Data changed after hashing | Recompute at run time |
| The auditor is independent | Clean room tests the artifact | Author assists during run | Second auditor with no contact |
| A pass on the number implies a pass on the analysis | Simplifies the verdict | Code path not exercised | Coverage check on the reported path |
| The package will still run next year | Shelf life is assumed | Dependency or access expiry | Retest with a recorded date |
| The reported metric is the primary one | Selective reporting risk | Post-hoc metric | Compare with registration |

## 12. Exercises

**Exercise 1. Place four studies on the ladder (recall, 15 minutes).**
Deliverable: four studies from your reading, each assigned a level with a one-sentence reason.
Acceptance criteria: the level is justified by what was varied (team, data, population); no study
is assigned a level higher than its evidence. Time: 15 minutes.

**Exercise 2. Build a manifest (application, 25 minutes).**
Deliverable: a SHA-256 manifest of a project directory with file names, hashes, and sizes, plus a
verification script. Acceptance criteria: the script detects one added, one removed, and one
changed file in a test directory. Time: 25 minutes.

**Exercise 3. Pin an environment (application, 30 minutes).**
Deliverable: a lock file or container definition for an analysis you have run, with versions and
a build command. Acceptance criteria: builds on a clean machine; the build is documented in the
README. Time: 30 minutes.

**Exercise 4. Audit a public repository (application, 90 minutes).**
Deliverable: a 12-item checklist for a public clinical AI repository, with evidence and a level
assignment. Acceptance criteria: every item has a locator or an explicit "absent"; the level
follows the decision table; at least two defects are diagnosed to a cause. Time: 90 minutes.

**Exercise 5. Diagnose a discrepancy (critique, 40 minutes).**
Deliverable: a short report on a difference between two runs, with the candidate causes tested in
order and the attributed cause. Acceptance criteria: each test isolates one layer; the conclusion
states the residual uncertainty. Time: 40 minutes.

**Exercise 6. Define a tolerance (application, 20 minutes).**
Deliverable: a written tolerance for one metric, justified by its sampling variability and by the
decision it informs. Acceptance criteria: the tolerance is a number with a rationale; the
decision consequence of exceeding it is stated. Time: 20 minutes.

**Exercise 7. Write the package README (design, 35 minutes).**
Deliverable: a README with prerequisites, exact commands, expected outputs, runtime, and known
nondeterminism. Acceptance criteria: a stranger can follow it without asking a question; the
expected outputs include a hash or a value. Time: 35 minutes.

**Exercise 8. Reproduce a published table (application, 60 minutes).**
Deliverable: a table of reported versus recomputed values for one published table, using either
the authors' code or your own reimplementation from the methods. Acceptance criteria: every
difference is quantified; the tolerance is stated before the comparison; the conclusion is
labelled as stable or unstable. Time: 60 minutes.

**Exercise 9. Design the release gate (design, 30 minutes).**
Deliverable: a five-item gate that an analysis must pass before it is used in a regulatory or
procurement submission. Acceptance criteria: each item is checkable by a non-author; the gate
names the artifact required. Time: 30 minutes.

### How to use the exercises

Exercises 1 and 6 build the vocabulary and the judgement about what counts as a match. Exercises
2, 3, and 7 build the authoring skill: manifest, environment, README. Exercises 4, 5, and 8 are
the audit path and should be done on a real repository, because the difficulties that matter
(access, undocumented steps, silent data changes) only appear on real artifacts. Exercise 9 is
the institutional fix, and it is the one that determines whether the other eight are practiced
once or routinely.

A learner who has done Exercises 4 and 5 can audit a repository. A learner who has done 2, 3, and
7 can build one that passes the audit. The two skills are different, and the second is the one
that changes an organisation's default behaviour.

## 13. Self-check answers

**A1. What is the difference between reproducible and replicable?** Reproducible means the same
data and code produce the same result with a different team. Replicable means different data with
the same methods produce a consistent conclusion.

**A2. Why is a container not sufficient for reproducibility?** It pins the runtime but not the
data, the configuration, or the seed. A container with a live database connection and no snapshot
is not reproducible.

**A3. What is the practical standard if bitwise identity is unattainable?** A pre-specified
tolerance on the reported quantities, plus a check that the conclusion is stable within it.

**A4. Why must the tolerance be set before the comparison?** Setting it afterwards allows the
auditor to choose a tolerance that guarantees a pass, which defeats the test.

**A5. A pipeline produces AUROC 0.842 on CPU and 0.841 on GPU. Is it reproducible?** At the level
of the conclusion, yes; at the level of the digit, no. The correct statement names the tolerance
and the platform.

**A6. What are the seven layers of the stack?** Question, data, operating system and hardware,
runtime, configuration, code, and outputs.

**A7. Why is a live database query a reproducibility defect?** The records can change between
runs, so the analysis is not a fixed object and cannot be rerun on the same data.

**A8. What should a package contain when data cannot be redistributed?** Code, schema, synthetic
data with the same structure, a documented access process, and a README stating that the result
was reproduced by a licensed auditor rather than by the public.

**A9. A repository has 500 stars and no README. Is it reproducible?** No. Popularity is not
reproducibility. The absence of a README means a stranger cannot run it.

**A10. Compute the checklist score.** Three yes, four partial, five no out of 12. Weighted score
= 3 + 2 = 5 of 12 = 41.7 percent.

**A11. Does a reproducible analysis guarantee a valid one?** No. Reproducibility tests whether the
same inputs produce the same outputs. It does not test whether the estimand is right, the label
is valid, or the population is representative. A reproducible pipeline can compute the wrong
quantity consistently.

**A12. Name the five failure families.** Missing artifacts, inaccessible artifacts, unstable
artifacts, undocumented artifacts, and misdescribed artifacts. The remedies are create, negotiate
access, control determinism, document, and reconcile.

**A13. Why does a temporal split make reproduction easier than a random split?** The split is
fixed by a date boundary rather than by a random seed, so a second auditor with the same data
extract obtains the same partition without needing the seed. The random split adds a parameter
that must be communicated and can be lost.

**Rubric for Exercise 4.** Full credit: 12 items with locators; level follows the table; two
defects diagnosed; the limit of the audit stated. Deductions for a level assigned without
evidence, or for treating a repository's existence as reproducibility.

**Rubric for Exercise 7.** Full credit: prerequisites, commands, expected outputs, runtime, and
nondeterminism; a stranger completes the run. Partial: commands present but expected outputs
absent.

## 14. Interview questions

**Q1. What does reproducible mean to you, precisely?**
Strong answer: different team, same data and code, same result within a stated tolerance; then
distinguishes replicable and generalisable. Weak answer: says "someone else can run it" without
the tolerance or the data condition.

**Q2. How would you audit a repository in a day?**
Strong answer: inventory, clean-machine build, run from README, compare outputs to reference,
diagnose the largest discrepancy, measure determinism, assign a level. Weak answer: reads the
code and forms an impression.

**Q3. Why do so few clinical AI papers share code?**
Strong answer: data access and regulation raise the cost; competitive and legal constraints;
journals do not enforce; the effort is borne by the authors and the benefit is diffuse. Weak
answer: says authors are lazy.

**Q4. What is the difference between a manifest and a lock file?**
Strong answer: the manifest identifies the data and artifacts by hash; the lock file identifies
the runtime versions. Both are needed. Weak answer: treats them as the same.

**Q5. A vendor says the model is reproducible because they have a Docker image. What do you ask?**
Strong answer: where is the data snapshot, the seed, the config, the expected outputs, and the
clean-machine test result. Weak answer: accepts the image.

**Q6. How do you set a tolerance for an AUROC?**
Strong answer: from the sampling variability given the test-set size and event count, and from
the decision threshold at which the metric matters; state both. Weak answer: picks 0.01 because
it sounds small.

**Q7. What is the most common cause of a reproducibility failure you have seen?**
Strong answer: names a layer and a mechanism, typically an unpinned environment or an unfixed
seed in a random split, and describes how it was diagnosed. Weak answer: gives a generic answer
about data quality.

**Q8. How do you make an analysis reproducible when the data are restricted?**
Strong answer: publish code, schema, synthetic data, and access process; have an independent
licensed auditor reproduce; state the level precisely. Weak answer: says it is impossible.

**Q9. What is the relationship between reproducibility and quality management?**
Strong answer: configuration management, document control, and audit trails are the regulated
form of the same artifacts; the QMSR requires reconstruction. Weak answer: sees them as separate
worlds.

**Q10. When would you accept a non-reproducible analysis?**
Strong answer: when the decision is provisional and the cost of reproduction exceeds the value,
and the limitation is stated; never for a regulatory submission. Weak answer: says never, or
always.

**Q11. Is a temporal split more reproducible than a random split?**
Strong answer: yes, because the partition is fixed by dates rather than a seed, so a second
auditor with the same extract obtains the same split without extra information; notes that it is
also a stronger validation design. Weak answer: says random splits are always fine.

**Q12. What is the difference between a reproducibility audit and a code review?**
Strong answer: the audit tests whether the outputs follow from the inputs; the review tests
whether the code is correct and maintainable; a pipeline can pass one and fail the other. Weak
answer: treats them as the same activity.

**Q13. How does a data standard affect reproducibility?**
Strong answer: a schema and vocabulary version is part of the environment; a FHIR release or an
OMOP vocabulary update changes the data even when the records do not, so the version must be
pinned and recorded. Weak answer: assumes that a standard removes the need for versioning.

**Q14. What does "available on request" mean for a reproducibility claim?**
Strong answer: it means the analysis is potentially reproducible for someone who is granted
access, which is a weaker claim than public reproducibility; the report must state which claim is
being made. Weak answer: treats it as equivalent to public availability.

**Q15. What single change would most improve reproducibility in your organisation?**
Strong answer: a release gate that requires a clean-machine run and a saved expected output, with
an owner. Weak answer: asks people to share code.

## 15. Key terms

- **Repeatability** — same team, same setup, same result.
- **Reproducibility** — different team, same data and code, same result.
- **Replicability** — different data, same methods, consistent conclusion.
- **Generalisability** — different population, consistent conclusion.
- **Bitwise reproducibility** — identical output bytes.
- **Statistical reproducibility** — same conclusion within tolerance.
- **Environment lock** — pinned runtime versions.
- **Container** — packaged runtime and system libraries.
- **Manifest** — file list with hashes and sizes.
- **Hash** — fixed-length content digest.
- **Seed** — value initialising a random generator.
- **Nondeterminism** — uncontrolled variation across runs.
- **Snapshot** — immutable copy of data at a time point.
- **Pipeline** — ordered steps from raw data to outputs.
- **DAG** — directed acyclic graph of steps.
- **Provenance** — record of how an output was produced.
- **Tolerance** — acceptable numerical difference.
- **Regression test** — known input yields known output.
- **Data use agreement** — contract governing data access.
- **Synthetic data** — generated data preserving structure.
- **Clean room** — an environment with no cached state or analyst memory.
- **Prediction artifact** — saved model outputs used for comparison.
- **Shelf life** — the period for which a package reproduces.
- **Release gate** — a condition an analysis must pass before use.
- **Externality** — a cost borne by someone other than the builder.
- **Schema drift** — a data standard version change altering fields.
- **Access expiry** — the date after which a data licence no longer applies.
- **Prediction parity** — equality of saved predictions across runs.
- **Audit trail** — chronological record of record changes.
- **Failure family** — missing, inaccessible, unstable, undocumented, or misdescribed.
- **Gate waiver** — a documented exception to a release condition.
- **Reproduction date** — the date on which a package was last tested.
- **Tolerance rule** — the pre-specified rule for accepting a difference.

## 16. Sources

- McDermott MBA, Wang S, Marinsek N, et al. Reproducibility in machine learning for health
  research: still a ways to go. *Science Translational Medicine* 2021;13(586):eabb1655.
  doi:10.1126/scitranslmed.abb1655. Accessed 26 October 2026.
- Sounack et al. AI-assisted scoping review of code sharing in clinical prediction model
  research. *Nature Medicine* 2026. Accessed 26 October 2026.
- Samuel S, Mietchen D. Computational reproducibility of Jupyter notebooks from biomedical
  publications. *GigaScience* 2024. Accessed 26 October 2026.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  doi:10.1136/bmj-2023-078378. Accessed 26 October 2026.
- US Food and Drug Administration. Quality Management System Regulation (QMSR), 21 CFR Part 820.
  Final rule, 2 February 2024, effective 2 February 2026.
  https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr.
  Accessed 26 October 2026.
- US Food and Drug Administration. Artificial Intelligence-Enabled Device Software Functions:
  Lifecycle Management and Marketing Submission Recommendations. Draft guidance, January 2025.
  https://www.fda.gov/media/184856/download. Accessed 26 October 2026.
- eCFR. 21 CFR 11.10, controls for closed systems. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10.
  Accessed 26 October 2026.
- EQUATOR Network. Reporting guideline library. https://www.equator-network.org/reporting-guidelines/.
  Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_model-cards-and-datasheets.md`](04_model-cards-and-datasheets.md)
- Next topic: [`06_quality-management.md`](06_quality-management.md)
- Appraisal: [`03_appraising-clinical-ai-evidence.md`](03_appraising-clinical-ai-evidence.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
