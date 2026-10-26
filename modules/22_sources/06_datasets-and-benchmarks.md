# Datasets and Benchmarks

> Module 22 — Sources and Further Reading
> Topic 06 of 09
> Estimated study time: 3 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

A dataset citation looks like the simplest citation in a bibliography. It has a name, a repository, a version, and a DOI. That simplicity is deceptive. The citation has to identify a moving object with a licence, an access route, a population, and a benchmark role, and a failure in any of those makes the result unreproducible or unusable.

The failures are common and they cluster around four words:

- **Version.** A paper reports counts from one release, and a reader downloads another. The numbers differ, and neither is wrong.
- **Licence.** A dataset is open for research and restricted for commercial use, or open for use but not for redistribution of derived data. The model is trained, the paper is written, and the licence question arrives at diligence.
- **Access.** A credentialed dataset is treated as open because a colleague shared a copy. The result cannot be reproduced by anyone who follows the citation.
- **Benchmark.** A leaderboard score is compared across papers that used different splits, different preprocessing, or different label sets.

The people affected are the data scientist who cannot reproduce a result, the reviewer who cannot obtain the data, the legal team that discovers the licence limits after the fact, and the product team that builds on a benchmark whose test set has been used for model selection by everyone in the field.

Dataset sources also have a property that journal articles do not: the object changes while the citation stays the same. A DOI can resolve to a landing page whose files have been replaced, and a version string can be reused by a vendor after a silent update. The citation discipline for datasets is therefore stricter than for papers, not looser.

This topic covers how to identify, evaluate, cite, and maintain dataset and benchmark sources. It overlaps with Module 21, which covers dataset selection as a decision; this topic covers the source work that makes the selection defensible and the citation reproducible.

The distinction between selection and citation is worth stating. Selection asks whether a dataset can support a claim. Citation asks whether a reader can obtain the same data and repeat the analysis. A dataset can be the right selection and a poor citation, and the second failure is more common because citation is treated as formatting.

A reader who masters this topic will find that most "reproducibility" problems are citation problems: the version was not recorded, the licence was not checked, or the split was not specified. Those are fixable with a register and a habit, not with more compute.

The habit has two parts: record the version at download, and read the licence before training. Both take minutes and both prevent failures that are expensive to reverse.

A third habit is to treat a benchmark score as a claim with conditions, rather than as a number that can be compared with any other number. The conditions are the split, the metric, the version, and the preprocessing, and without them the score is a result without a meaning.

> **Note:** A dataset DOI identifies a landing page, not a file. Reproducibility requires the version, the access date, the licence, and, where possible, a hash of the files actually used.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** a dataset by creator, title, repository, version, year, and persistent identifier, and distinguish a landing page from a file. Assessed by Exercise 1 and self-check item 1.
2. **Determine** the access tier and licence conditions, and map them to a planned use. Assessed by Exercise 2.
3. **Detect** a version mismatch between a paper's reported counts and the files in hand. Assessed by Exercise 3 and the worked example.
4. **Evaluate** a benchmark for contamination, split integrity, and comparability across papers. Assessed by Exercise 4.
5. **Assess** whether a dataset's population and setting match a stated intended use. Assessed by Exercise 5.
6. **Write** a dataset citation that a reader can resolve to the same version and files. Assessed by Exercise 6.
7. **Apply** the citation-checking script to a data register and interpret the completeness scores. Assessed by Exercise 7 and section 7.
8. **Design** a version and licence register for a project with review triggers. Assessed by Exercise 8.

## 3. The core mental model

A dataset is an object with six attributes that the citation must carry: who made it, what it contains, which version, under what terms, at what access tier, and for what benchmark role.

```text
                    DATASET NEED
        population | variables | outcome | setting | era
                              |
                              v
                    CANDIDATE SET
        primary | comparator | fallback
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
     VERSION               LICENCE               ACCESS
   release, counts,      permitted use,        tier, route,
   changes, hash         derivatives,          latency, cost
                         redistribution
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                    BENCHMARK ROLE
        task, split, metric, contamination, leaderboard
                              |
                              v
                    CITATION + REGISTER ENTRY
        creator | title | repository | version | DOI | access date
```

Read the diagram downwards. The version, licence, and access attributes are checked before the benchmark role, because a dataset that cannot be used legally or obtained reliably is not a candidate regardless of its benchmark quality.

**A dataset is a measurement instrument, not a file.** The same repository can release a corrected version that changes counts, labels, or variables. The version is part of the instrument's identity, and a result computed on a different version is a different result.

**The licence is about use, not access.** Open access to download does not imply permission to use in a product, to redistribute, or to publish derived data. The four questions are: may I train, may I evaluate, may I publish results, and may I deploy.

**Benchmarks are social objects.** A benchmark's meaning depends on how the community uses it: which split is standard, whether the test set has been used for tuning, and whether the metric rewards the clinically relevant behaviour. A leaderboard is a record of a competition, not a measure of utility.

**The citation is a reproducibility claim.** A citation that resolves to a landing page but not to a file is a pointer, not a reproduction package. The version, access date, and file hash are what make the claim checkable.

| Attribute | Question | Failure mode | Artefact |
|---|---|---|---|
| Provenance | Who made it, from what source? | Unknown origin | Datasheet |
| Version | Which release, with what changes? | Silent update | Version and hash |
| Licence | What use is permitted? | Commercial use prohibited | Clause matrix |
| Access | Who can obtain it, how? | Credentialing gap | Access route |
| Population | Who is in it? | Cohort mismatch | Coverage table |
| Benchmark | What task and split? | Contamination | Split specification |
| Licence version | Which terms apply? | Terms revised silently | Licence copy |
| Subset | Which portion is used? | Parent cited for a subset | Subset definition |
| Evaluation protocol | How results are produced | Metric gaming | Protocol document |

The six attributes are not independent. A licence can determine the access tier; the access tier can determine the benchmark role; the version can change the population. The procedure checks them in order because an early attribute can disqualify a dataset before the later ones matter.

A second reading of the diagram is as a record structure. Each box becomes a field in the register, and a register with all six fields answers the questions that arise at review, at audit, and at the next project.

The model also explains why a dataset can be right for a project and wrong for a paper. The project needs data that support a claim; the paper needs data that a reader can obtain. A credentialed dataset can satisfy the first and complicate the second, and the citation has to address both.

## 4. Definitions and vocabulary

### 4.1 Dataset and provenance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Dataset | A structured collection of records with documentation. | Used for a single file. | Object |
| Release | A published version of a dataset. | Confused with a patch. | Version string |
| Version | An identifier for a specific release. | Reused after a silent update. | Semantic or date-based |
| Patch | A correction that does not change the schema. | Assumed to leave counts unchanged. | Version suffix |
| Landing page | The repository page for a dataset. | Treated as the data. | URL or DOI |
| File hash | A digest of the file contents. | Omitted from the record. | Hash string |
| Datasheet | Documentation of a dataset's motivation, composition, and use. | Confused with a data dictionary. | Document |
| Data dictionary | Definitions of fields and values. | Confused with a datasheet. | Document |
| Provenance | The origin and processing history of the data. | Recorded as "from the EHR". | Provenance record |
| Synthetic data | Data generated to resemble real data without containing it. | Assumed to preserve all statistical properties. | Generation method |

### 4.2 Access tiers

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Open access | Direct download without identity verification. | Assumed to permit any use. | Access tier |
| Registered access | Account creation and click-through terms. | Treated as credentialed. | Access tier |
| Credentialed access | Identity verification, training, and a signed agreement. | Shared through a colleague's account. | Access tier |
| Controlled access | Institutional approval and output review. | Planned as a short step. | Access tier |
| Licensed access | Contractual access with fees and field-of-use limits. | Treated as ownership. | Access tier |
| Query-only access | Aggregate results without row-level data. | Assumed sufficient for modelling. | Access tier |

### 4.3 Licence terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Commercial use | Use in a product or revenue-generating activity. | Assumed to exclude internal research at a company. | Clause |
| Derivative work | A new artefact substantially derived from the data. | Assumed to exclude model weights. | Clause |
| Redistribution | Providing data or close derivatives to a third party. | Assumed to permit sharing features. | Clause |
| Model training | Using data to fit or fine-tune a model. | Assumed covered by "research use". | Clause |
| Publication | Releasing results, figures, or derived data. | Assumed permitted by access. | Clause |
| Attribution | A requirement to credit the source. | Omitted in derived tables. | Clause |
| Share-alike | A requirement to license derivatives similarly. | Ignored for model weights. | Clause |
| Field of use | The permitted purpose or therapeutic area. | Ignored because access was free. | Clause |

### 4.4 Benchmark terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Benchmark | A fixed task, dataset, and metric. | Treated as clinical utility. | Specification |
| Split | The partition into train, validation, and test. | Reused across papers. | Split key |
| Patient-level split | A split that keeps all records from one patient together. | Replaced by a row split. | Split key |
| Temporal split | A split by time, with training before testing. | Replaced by a random split. | Split key |
| Site-level split | A split that holds out entire sites. | Absent from single-site data. | Split key |
| Contamination | Test items present in training data. | Checked only for text benchmarks. | Proportion |
| Leaderboard | A ranked list of submissions. | Treated as a measure of progress. | Ranking |
| Saturation | Top methods within noise of each other. | Reported as a new state of the art. | Interval overlap |
| Metric gaming | Optimising the metric without the task. | Invisible in the score. | Behaviour |

### 4.5 Citation elements

| Element | Why it is needed | Common omission |
|---|---|---|
| Creator | Attribution and provenance | Institutional author |
| Title | Identification | Subset name |
| Repository | Location and persistence | Project website |
| Version | Reproducibility | Version string |
| Year | Temporal context | Release year |
| Identifier | Resolution | DOI |
| Access date | Verification | Date |
| Licence | Reuse permission | Licence name |
| Access date | Verification | Date |
| Version hash | File identity | Hash |
| Subset | The portion actually used | Definition |
| Split | The evaluation partition | Split key |
| Preprocessing | Transformations applied | Pipeline version |

## 5. The mechanics, step by step

The procedure has 24 steps in five phases. It assumes the dataset has already been selected as scientifically suitable, and it checks the source properties that make the selection defensible.

The order matters. Version, licence, and access are checked before benchmark quality, because a dataset that cannot be used or obtained is not a candidate. Benchmark quality is checked before citation, because a citation of a contaminated benchmark is a citation of a misleading result.

### Phase A — Frame the need

**Step 1. Write the data need as a specification.**
Input: the question, the intended use, and the deployment setting.
Output: a specification with population, variables, outcome, setting, era, and unit of analysis.
Rationale: a dataset can only be evaluated against a need, and "we need EHR data" is not a need.
Branch: if the specification cannot be written, the project is not ready for dataset selection.
Edge case: a research question and a product question can have different specifications from the same data.
Check: does the specification name the unit of analysis.
Record: the specification.

**Step 2. Identify the access tier the project can accept.**
Input: the institution, the timeline, and the legal entity.
Output: acceptable tiers and the maximum latency.
Rationale: the access route is a schedule and legal constraint, not an afterthought.
Branch: if the project needs commercial use, exclude datasets whose licence forbids it before evaluating quality.
Edge case: an academic collaboration can have different terms from a commercial one.
Check: is the tier stated in the plan.
Record: the tier and latency.

**Step 3. Identify the licence uses the project requires.**
Input: the product plan and the publication plan.
Output: four yes/no requirements: train, evaluate, publish, deploy.
Rationale: the four uses are governed by different clauses, and a dataset can permit one and forbid another.
Branch: if deploy is required and the licence is research-only, the dataset is a research comparator, not a product input.
Edge case: model weights can be a derivative work under some licences and not others.
Check: are the four uses recorded.
Record: the use matrix.

**Step 4. Identify the benchmark role.**
Input: the project's evaluation plan.
Output: whether the dataset is for development, internal evaluation, external comparison, or a public leaderboard.
Rationale: the role determines the split discipline and the contamination risk.
Branch: if the dataset is a public leaderboard, assume some contamination and plan a private holdout.
Edge case: the same dataset can be both a development set and a benchmark, which is a conflict.
Check: is the role stated.
Record: the role.

### Phase B — Identify the object and version

**Step 5. Find the dataset at the repository.**
Input: the name and the field.
Output: the repository record, DOI, and landing page.
Rationale: the repository record is the authoritative source for version and terms; a project website is not.
Branch: if the dataset exists only on a project website, record the absence of a persistent identifier as a risk.
Edge case: mirrors and redistributions can be outdated or altered.
Check: is the record at a repository.
Record: the repository and DOI.

**Step 6. Identify the version and release date.**
Input: the repository record and the files.
Output: a version string and a release date.
Rationale: the version is part of the object's identity and the citation.
Branch: if the version is not stated, use the release date and record the ambiguity.
Edge case: a version string can be reused after a patch, which is why the release date and hash matter.
Check: is the version in the citation.
Record: version and date.

**Step 7. Check what changed between versions.**
Input: the release notes or changelog.
Output: a list of changes that affect the planned analysis.
Rationale: a patch can change counts, labels, or variables, and the effect on a result is not always small.
Branch: if the change affects the outcome or a key feature, re-run rather than caveat.
Edge case: a changelog can be incomplete, which is itself a risk signal.
Check: are the release notes read.
Record: the changes.

**Step 8. Record the files and a hash.**
Input: the downloaded files.
Output: file names, sizes, and a hash of each.
Rationale: a DOI resolves to a landing page, and the files can be replaced; the hash is what makes the record reproducible.
Branch: if the repository provides a versioned archive, use its identifier rather than individual files.
Edge case: large datasets can be distributed in parts, so the hash should cover the assembled set.
Check: is a hash recorded.
Record: the files and hashes.

### Phase C — Check licence and access

**Step 9. Read the licence, not the summary.**
Input: the licence text or data use agreement.
Output: a clause-by-clause matrix for train, evaluate, publish, deploy, redistribute, and retain.
Rationale: summaries omit the clause that matters, and the clause is usually about derivatives or commercial use.
Branch: if a clause is silent, treat it as unresolved and ask the owner.
Edge case: a licence can change between versions, so the version and licence must be recorded together.
Check: is the licence version recorded.
Record: the matrix.

**Step 10. Check the access route and latency.**
Input: the repository's access page.
Output: the steps, the required training, the institutional roles, and the expected time.
Rationale: access latency is a schedule risk, and the route determines who can reproduce the work.
Branch: if the latency exceeds the project window, the dataset fails regardless of quality.
Edge case: a demo subset is not a substitute for the credentialed data.
Check: is the route documented.
Record: the route and estimate.

**Step 11. Check the granularity and fields available.**
Input: the data dictionary and the access specification.
Output: a statement of whether the required fields are available at the required granularity.
Rationale: a query-only product cannot support row-level modelling, and a de-identified dataset can lack the dates a temporal split needs.
Branch: if a required field is unavailable, the dataset fails for this specification.
Edge case: small-cell suppression can remove the subgroup cells that motivated the study.
Check: are the fields listed and mapped.
Record: the field map.

**Step 12. Check the de-identification and governance constraints.**
Input: the data use agreement and the privacy documentation.
Output: a statement of what cannot be done, such as re-identification attempts or linking to external data.
Rationale: governance constraints can prohibit analyses that are technically possible.
Branch: if linkage is prohibited, a planned enrichment is not available.
Edge case: output review can delay publication or require suppression of small cells.
Check: are the constraints recorded.
Record: the constraints.

### Phase D — Check population and benchmark

**Step 13. Build a coverage table.**
Input: the dataset's demographic and site fields.
Output: counts by site, year, age, sex, and relevant subgroups, with outcome counts.
Rationale: the population is the basis for any claim about applicability, and the table exposes underpowered subgroups.
Branch: if a deployment-relevant subgroup has few events, any subgroup metric is uninformative.
Edge case: harmonised categories can hide differences in source coding.
Check: does the table include outcome counts.
Record: the coverage table.

**Step 14. Check the era and the definition versions.**
Input: the dataset documentation and the clinical definitions used.
Output: a statement of the era and any definition changes inside it.
Rationale: a dataset spanning a definition change contains two measurements under one label.
Branch: if the change is material, restrict or stratify.
Edge case: a dataset can document the coding system but not the clinical definition.
Check: are the definition versions recorded.
Record: the era and versions.

**Step 15. Check the benchmark's task and metric.**
Input: the benchmark specification and the papers that use it.
Output: a statement of the task, the metric, and whether they match the intended decision.
Rationale: a benchmark metric can reward behaviour that is irrelevant to the clinical decision.
Branch: if the metric is not decision-relevant, use the benchmark for development and a decision metric for evaluation.
Edge case: a benchmark can have several official metrics, and papers can report different ones.
Check: is the metric named.
Record: the task and metric.

**Step 16. Check the split integrity.**
Input: the split definition and the identifiers.
Output: a statement of the split unit and whether it respects patients, sites, and time.
Rationale: a row-level split leaks information and inflates performance.
Branch: if the split is row-level, the reported performance is optimistic.
Edge case: a patient can appear under different identifiers across sites.
Check: is the split unit stated.
Record: the split.

**Step 17. Check for contamination.**
Input: the benchmark's history and the model's training data.
Output: a contamination assessment with a method and a residual risk.
Rationale: a test set that has been public for years is likely to have influenced model selection, even if no explicit training occurred.
Branch: if contamination is likely, create a private holdout or a temporal test set.
Edge case: near-duplicates and derived images are a form of contamination that string matching misses.
Check: is the contamination risk stated.
Record: the assessment.

**Step 18. Check comparability across papers.**
Input: the papers that report benchmark results.
Output: a table of preprocessing, splits, and metrics, and a statement of comparability.
Rationale: a leaderboard compares numbers that may have been produced under different conditions.
Branch: if preprocessing differs materially, the ranking is not established.
Edge case: a paper can use a subset of the benchmark without saying so.
Check: are the conditions aligned.
Record: the comparison table.

### Phase E — Cite and maintain

**Step 19. Write the citation with all elements.**
Input: creator, title, repository, version, year, DOI, access date, licence.
Output: a citation that resolves to the version used.
Rationale: the citation is the reproducibility claim, and each missing element weakens it.
Branch: if the dataset has no creator listed, cite the institution and record the ambiguity.
Edge case: a subset should be cited as a subset, not as the parent dataset.
Check: does the citation include the version and access date.
Record: the citation.

**Step 20. Record the licence in the project documentation.**
Input: the licence matrix.
Output: a statement of permitted uses and restrictions in the model card or protocol.
Rationale: the licence constrains what the project may do with the result, not only the data.
Branch: if the licence forbids commercial deployment, the model card should say so.
Edge case: a licence can permit research and forbid a product even when the model is a derivative.
Check: is the licence visible to the product team.
Record: the statement.

**Step 21. Record the population limits.**
Input: the coverage table.
Output: a limitations paragraph naming the sites, years, and subgroups.
Rationale: a model inherits the population of its data, and the documentation should say so.
Branch: if a subgroup is absent, the model's performance for that group is unknown.
Edge case: a dataset that excludes a group can still be useful for a narrow intended use.
Check: are the limits in the model card.
Record: the limits.

**Step 22. Record the benchmark result with its conditions.**
Input: the evaluation.
Output: a result table with split, metric, preprocessing, and version.
Rationale: a benchmark number without its conditions cannot be compared with anything.
Branch: if the conditions differ from the published baseline, say so.
Edge case: a reimplementation can differ from the original baseline for reasons unrelated to the model.
Check: are the conditions reported.
Record: the result table.

**Step 23. Set the version and licence review trigger.**
Input: the repository's release cadence and the licence.
Output: a review date and event triggers.
Rationale: datasets change and licences can be revised, and a model built on one version needs a trigger when the version changes.
Branch: the triggers should include a new release, a licence change, and a repository deprecation.
Edge case: a repository can remove a dataset, which makes the citation unresolvable.
Check: is the trigger recorded.
Record: the trigger.

**Step 24. Record the residual uncertainty.**
Input: the full record.
Output: a sentence naming what is not known about the dataset or benchmark.
Rationale: every dataset has unmeasured properties, and the uncertainty should be stated rather than assumed away.
Branch: if the uncertainty is material to the claim, downgrade the claim.
Edge case: an undocumented dataset is not necessarily unusable, and the documentation gap should be stated.
Check: is the uncertainty visible.
Record: the uncertainty.

The procedure produces a citable dataset and a record that a reader can use to obtain the same version. It does not produce a good dataset; it produces a defensible use of one.

### Version changes and their effects

| Change type | Example | Effect on a result | Action |
|---|---|---|---|
| Patch | Corrected item identifiers | Small or none | Re-run if the field is used |
| Added records | New years appended | Counts and prevalence change | Re-report and re-split |
| Removed records | De-identification fix | Cohort shrinks | Re-run and document |
| Label correction | Outcome misclassification fixed | Performance changes | Re-train |
| Schema change | Field renamed or split | Pipeline breaks | Update and re-validate |
| Licence change | New restriction on derivatives | Use may become non-compliant | Re-review with legal |

The patch row is the one teams underrate. A patch that corrects identifiers can change a join and therefore a cohort, even though the version number looks minor.

### Split types and what they test

| Split | What it tests | What it does not test |
|---|---|---|
| Random row | Nothing useful in patient data | Anything |
| Patient-level random | Generalisation to new patients, same site | New site or new era |
| Temporal | Generalisation forward in time | New site |
| Site-level | Transportability across institutions | Temporal drift |
| Prospective | Real deployment performance | Subgroup performance without size |

A benchmark that uses a random row split is measuring memorisation as much as prediction. The split type should be reported with every result, and a result without it is not comparable.

### Access routes compared

| Route | Typical latency | Who can reproduce | Main risk |
|---|---|---|---|
| Open download | Minutes | Anyone | Licence misunderstood |
| Registered | Hours to days | Anyone who registers | Terms clicked without reading |
| Credentialed | Weeks | Anyone who completes training | Account sharing |
| Controlled | Weeks to months | Approved researchers | Output review delays |
| Licensed | Contract-dependent | Licensees only | Field-of-use limits |
| Query-only | Immediate | Anyone with access | No row-level reproduction |

The route determines who can check the work. A paper based on credentialed data can still be reproducible if the route is described and the code is shared, even though the reader must apply for access.

### Benchmark contamination checks

- Is the test set public, and for how long.
- Was the test set used for model selection in the paper.
- Are there near-duplicate images or documents across splits.
- Did pretraining include the test set.
- Is the leaderboard still accepting submissions.
- Are the top results within the confidence interval of each other.
- Does the paper report a private holdout.
- Is the official split the one used.

A benchmark with a public test set and a long leaderboard history should be treated as a development set, and the evaluation claim should rest on a private or prospective holdout.

### A branch worked in miniature: the same benchmark, two splits

Two papers report a chest radiograph benchmark. The first uses a random row split and reports an AUC of 0.91. The second uses a patient-level split and reports 0.85. Both use the same images and the same label set.

The difference is not a modelling improvement. The first paper's split allows images from the same patient to appear in training and test, so the model can recognise the patient rather than the finding. The second paper's split measures generalisation to new patients.

The correct comparison is between the two patient-level results, and the row-split result should be reported as an upper bound or not at all. A leaderboard that mixes the two is comparing different tasks.

### What a dataset register should contain

- Name, creator, repository, DOI, and version.
- Release date and file hash.
- Licence and the four-use matrix.
- Access tier and route.
- Population coverage summary.
- Benchmark role and split type.
- Review trigger.

A register with these fields answers the questions that a reviewer, a lawyer, and a new analyst will ask, and it answers them once rather than in three separate conversations.

## 6. Worked example

A grant application states: "We will use MIMIC-IV, which contains 364,627 ICU patients." The sentence will be read by a methods reviewer who knows the dataset.

### Framing the claim

The sentence has one number and two possible meanings:

1. **Total patients.** The number of unique patients in the dataset, including those who only had an emergency department visit.
2. **ICU patients.** The number of unique patients with at least one intensive care stay.

The number 364,627 is the first. The phrase "ICU patients" asserts the second. The claim is a unit error, and unit errors in dataset descriptions are common because the headline count is the one that gets quoted.

| Sub-claim | Type | What could support it |
|---|---|---|
| The dataset contains 364,627 patients | Count | The repository page, with the version |
| Those patients are ICU patients | Definition | The ICU module's patient count |
| The dataset is suitable for the study | Applicability | Population and variable comparison |

### Candidate sources

| # | Candidate | Type | Role |
|---|---|---|---|
| C1 | PhysioNet dataset page for MIMIC-IV v3.1 | Repository record | The authoritative counts and version |
| C2 | The MIMIC-IV data paper | Peer-reviewed description | The design and contents |
| C3 | A tutorial that quotes the headline count | Secondary | The likely origin of the sentence |
| C4 | A paper that used the ICU module | Empirical use | What the ICU cohort looks like in practice |
| C5 | The course survey file | Course material | The counts as presented to learners |

### Scoring the candidates

**C1, the repository page.**
- Authority: highest for the counts and the version.
- Currency: current, since it is the versioned record.
- Applicability: the whole dataset, not the study question.
- Accessibility: public.
- Provenance: the maintainers describe the release.
- Verdict: primary support for the counts.

**C2, the data paper.**
- Authority: high for the design and the content description.
- Currency: depends on the version it describes.
- Applicability: the dataset as a whole.
- Accessibility: open access.
- Provenance: methods and schema are described.
- Verdict: primary support for the design, secondary for the current counts.

**C3, the tutorial.**
- Authority: low to moderate.
- Currency: unknown.
- Applicability: general.
- Accessibility: public.
- Provenance: often no methods.
- Verdict: background only; it is a pointer to C1.

**C4, the empirical paper.**
- Authority: high for its own cohort.
- Currency: depends on the version used.
- Applicability: its cohort may be a subset, such as a first-ICU-stay cohort.
- Accessibility: varies.
- Provenance: methods described.
- Verdict: evidence about a use, not about the dataset's total.

**C5, the course survey.**
- Authority: course material.
- Currency: reflects the version it cites.
- Applicability: general.
- Accessibility: public.
- Provenance: it cites the dataset.
- Verdict: orientation, with the repository as the primary source.

### The counts

The v3.1 repository page and the associated documentation give several different counts, and each answers a different question:

| Quantity | Count | What it counts |
|---|---:|---|
| Unique patients, all | 364,627 | Anyone in the dataset, including ED-only |
| Hospital admissions | 546,028 | Admissions in the hospital module |
| Patients with at least one admission | 223,452 | Patients in the hospital module |
| ICU stays | 94,458 | ICU episodes |
| Unique ICU patients | 65,366 | Patients with at least one ICU stay |
| ED-only patients | 141,175 | Patients with no admission |

The sentence's number is the first row and its noun is the fifth. The gap between them is 299,261 patients, which is not a rounding difference.

### Verdict

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| Contains 364,627 patients | Primary support | "MIMIC-IV v3.1 contains 364,627 unique patients." |
| Those are ICU patients | Reject | "Of these, 65,366 patients have at least one ICU stay, across 94,458 stays." |
| Suitable for the study | Separate question | Requires a population and variable check |

The corrected sentence is longer and it tells the reviewer that the team knows what the dataset contains. In a grant application, that is worth more than the larger number.

### Two diagnostics that would change the verdict

**Diagnostic 1: the study uses the hospital module, not the ICU module.** If the analysis uses ward and ICU data together, the relevant count is 223,452 patients with at least one hospital admission, and the sentence should say so. The verdict would change from "wrong unit" to "different unit, correctly described", and the number would change again.

**Diagnostic 2: a later version changes the counts.** If a new release adds years or corrects records, the counts change and the citation must name the version. The verdict on the sentence's structure would remain, and the number would change; the diagnostic is a version check, not a re-reading.

A third diagnostic, weaker, is a definition of "ICU patient" that includes patients admitted to an intermediate care unit. The dataset's ICU module has its own inclusion rules, and a study that defines ICU differently needs to say so.

### What the team should record

The data register records the dataset, version 3.1, the DOI, the release date, the counts used, and the module. The grant text is corrected to name the module and the count that matches it.

The case also produces a rule for the team: never quote a dataset count without the noun that the count belongs to. A count without its unit is a number, not a fact.

### Why the error is common

The headline count is the one that appears in the abstract, the README, and the press coverage. It is also the largest number, which makes it attractive in a grant application. The ICU count is buried in a table or a module description.

The error propagates because each document copies the previous one. A tutorial quotes the headline count, a paper cites the tutorial, a grant cites the paper, and by the fourth document the noun has changed from "patients" to "ICU patients" without anyone checking.

| Document | What it says | Where the number came from |
|---|---|---|
| Repository README | 364,627 unique patients | The maintainers |
| Tutorial | 364,627 patients | The README |
| Paper introduction | 364,627 patients | The tutorial |
| Grant | 364,627 ICU patients | The paper, with a changed noun |
| Review | "the ICU cohort of 364,627" | The grant |

The table is a reconstruction of a common pattern, not a record of a specific case. The mechanism is the same in every version: the number survives and the noun drifts.

### What the reviewer sees

- A team that knows the dataset has three or four counts and uses the right one.
- A team that knows the version, because the counts change between releases.
- A team that knows the module, because the hospital and ICU modules differ.
- A team that can name the outcome and the population it will study.

A reviewer who sees the wrong count will ask a question that the team cannot answer, and the rest of the application becomes harder to defend.

### How to check a count

- Open the repository page for the version you will use.
- Find the count table or the module description.
- Match the noun to the number.
- Check whether the count is patients, admissions, stays, or records.
- Record the version and the access date.
- Recheck when the version changes.

The check takes about five minutes and prevents a class of error that is embarrassing in proportion to its triviality.

A useful discipline is to read the dataset's README as a claim, not as documentation. Every count in it belongs to a noun, and every noun can be checked against the schema.

### Counts that are easy to confuse

| Quantity | Common label in papers | Confusion with |
|---|---|---|
| Unique patients | "patients" | Admissions or stays |
| Hospital admissions | "admissions" | Patients |
| ICU stays | "ICU admissions" | Patients |
| Unique ICU patients | "ICU patients" | ICU stays |
| Records | "samples" | Patients |
| Encounters | "visits" | Patients |

The pattern is that each count has a near-synonym that a reader will assume. Naming the unit explicitly is the only reliable fix, and it costs one word.

A second habit is to record the count and the version together, so that a later release does not silently invalidate the sentence.

## 7. Code, query, or calculation

The script checks a data register for citation completeness. It validates the DOI syntax, flags missing version, licence, access tier, and update date, and produces a score that a team can use as a review gate.

```python
"""Dataset citation completeness and DOI sanity check."""

import re
from dataclasses import dataclass

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")


@dataclass
class Dataset:
    name: str
    version: str
    doi: str
    licence: str
    access: str
    updated: str
    creator: bool


def check(d):
    issues = []
    if not d.version:
        issues.append("no version")
    if not d.doi:
        issues.append("no DOI")
    elif not DOI_RE.match(d.doi):
        issues.append("malformed DOI")
    if not d.licence:
        issues.append("no licence")
    if not d.access:
        issues.append("no access tier")
    if not d.updated:
        issues.append("no update date")
    if not d.creator:
        issues.append("no creator")
    return max(100 - 15 * len(issues), 0), issues


DATASETS = [
    Dataset("MIMIC-IV", "3.1", "10.13026/kpb9-mt58", "ODbL", "credentialed", "2024-10-11", True),
    Dataset("PTB-XL", "1.0.1", "10.13026/6sec-a640", "CC-BY-4.0", "open", "2020-05-25", True),
    Dataset("CheXpert", "", "", "research use", "registered", "", True),
    Dataset("BraTS 2024", "2024", "10.5281/zenodo.10978907", "", "registration", "2024-04", True),
    Dataset("EHRSHOT", "", "", "Stanford DUA", "credentialed", "2023-07", True),
    Dataset("All of Us", "v9", "10.0000/example", "DUA", "controlled", "2026-06-26", False),
    Dataset("Synthea", "3.3.0", "", "Apache-2.0", "open", "", True),
]

rows = []
for d in DATASETS:
    score, issues = check(d)
    rows.append((d.name, d.version or "-", score, issues))

rows.sort(key=lambda r: r[2])
print(f"{'dataset':14s} {'version':8s} {'score':>5s}  issues")
print("-" * 78)
for name, version, score, issues in rows:
    print(f"{name:14s} {version:8s} {score:5d}  {', '.join(issues) if issues else 'complete'}")
print("-" * 78)
print("complete:", sum(1 for r in rows if not r[3]),
      " incomplete:", sum(1 for r in rows if r[3]))
print("mean score:", round(sum(r[2] for r in rows) / len(rows), 1))
```

Expected output:

```text
dataset        version  score  issues
------------------------------------------------------------------------------
CheXpert       -           55  no version, no DOI, no update date
EHRSHOT        -           70  no version, no DOI
Synthea        3.3.0       70  no DOI, no update date
BraTS 2024     2024        85  no licence
All of Us      v9          85  no creator
MIMIC-IV       3.1       100  complete
PTB-XL         1.0.1     100  complete
------------------------------------------------------------------------------
complete: 2  incomplete: 5
mean score: 80.7
```

Four readings matter. The two complete records are the ones with a version and a DOI, which is not a coincidence. BraTS loses points for a missing licence even though the record is otherwise strong, which is the right weighting for a product context. All of Us loses points for a missing creator, which is a fixable documentation problem. And the mean score of 80.7 looks healthy while five of seven records are incomplete, which is why the count of complete records matters more than the mean.

The DOI check is syntactic. It confirms that the identifier has the right shape and does not confirm that it resolves, that the landing page is the right dataset, or that the licence on the page matches the record. Those require a network check and a human read.

To adapt the script, add your own register entries and adjust the weights. A product team might weight licence twice; a research team might weight the version more heavily. Keep the count of complete records as the headline, because it is harder to game than a mean.

## 8. Data, standards, and regulatory context

Dataset citation practice is governed by a mixture of community principles, repository policy, and, for health data, privacy law.

### 8.1 Citation and metadata standards

| Standard or practice | Issuer | What it constrains | Binding status |
|---|---|---|---|
| FAIR Guiding Principles | Community, published 2016 | Findability, accessibility, interoperability, reuse | Voluntary |
| DataCite metadata schema | DataCite | DOI metadata for datasets | Repository policy |
| Datasheets for Datasets | Research community | Documentation of motivation, composition, and use | Voluntary |
| Dataset Nutrition Label | Research community | Structured dataset description | Voluntary |
| Croissant | ML Commons | Dataset metadata for machine learning | Emerging convention |
| Repository certification | CoreTrustSeal and others | Repository trustworthiness | Repository policy |

The FAIR principles are frequently cited as a compliance requirement and are not one. They describe properties that make data reusable, and they require local interpretation. A dataset can be FAIR and unusable for a specific question, and a dataset can be non-FAIR and the only source for a question.

### 8.2 Privacy and governance

| Regime | Scope | What it constrains | Effect on datasets |
|---|---|---|---|
| HIPAA Privacy Rule | US covered entities | Use and disclosure of protected health information | De-identification standards |
| GDPR | EU personal data | Lawful basis, purpose limitation, data minimisation | Pseudonymisation and transfer rules |
| EHDS Regulation 2025/327 | EU health data space | Secondary use of health data | Permitted purposes and secure environments |
| Common Rule | US federally funded research | Human subjects protections | Consent and IRB review |
| Data use agreements | Repository or owner | Specific permitted uses | Contractual restrictions |

A dataset can satisfy a repository's access policy and still be subject to a data use agreement that adds restrictions. The agreement is the binding document for the researcher, and it should be read alongside the licence.

### 8.3 What the standards do not decide

- Whether a dataset is suitable for a specific clinical question.
- Whether a benchmark measures clinical utility.
- Whether a model trained on the data may be deployed.
- Whether derived outputs are personal data.
- Whether a licence permits model weights as a derivative.

Those are legal and scientific judgements, and the standards provide vocabulary rather than answers.

### 8.4 FAIR in practice

| Principle | What it asks | What it does not ask |
|---|---|---|
| Findable | A persistent identifier and rich metadata | That the data are free |
| Accessible | A defined access protocol, even if authentication is required | That access is open |
| Interoperable | Shared vocabularies and formats | That the data fit your model |
| Reusable | Licence, provenance, and documentation | That the data are accurate |

The accessibility principle is often misread as "open". A credentialed dataset can be FAIR if the access protocol is clear and the metadata are public, which is why the principle is compatible with privacy law.

### 8.5 A data use agreement checklist

- What is the permitted purpose, and does it cover the planned use.
- Is commercial use permitted, and how is commercial defined.
- Are derivatives permitted, and are model weights a derivative.
- Is redistribution permitted, and under what conditions.
- Is publication of results permitted, and is review required.
- Is linkage to other data permitted.
- What are the retention and deletion requirements.
- What are the security requirements.
- Who is liable for a breach.
- What is the term, and what happens at expiry.

The checklist is not a legal opinion. It is a set of questions that a legal reviewer will ask, and having the answers shortens the review.

### 8.6 What a datasheet should contain

| Section | Content | Why it matters |
|---|---|---|
| Motivation | Why the dataset was created | Frames the intended use |
| Composition | What each instance represents | Determines the unit |
| Collection | How the data were gathered | Reveals selection |
| Preprocessing | Cleaning and filtering | Affects comparability |
| Uses | Intended and discouraged uses | Prevents misuse |
| Distribution | Licence and access | Governs reuse |
| Maintenance | Who supports it | Determines longevity |

A dataset without a datasheet is not unusable, and its limitations are unknown. The correct response is to state that the documentation is absent, not to assume the limitations are absent.

## 9. Industry practice and case evidence

**MIMIC-IV (documented).** Version 3.1 was released on 11 October 2024 and is distributed through PhysioNet under a credentialed access agreement, with the DOI 10.13026/kpb9-mt58. The release contains 364,627 unique patients, 546,028 hospital admissions, 94,458 ICU stays, and 65,366 unique ICU patients. The counts are the basis of the worked example, and the case shows why the noun matters as much as the number.

**PTB-XL (documented).** The dataset contains 21,837 twelve-lead ECG records from 18,885 patients, collected between 1989 and 1996, and is distributed under a CC BY 4.0 licence with a DOI and a PhysioNet record. The case shows what an open dataset citation can look like when the licence, version, and DOI are all present, and it also shows the limit: the data are old, so an era claim must be stated.

**CheXpert (documented).** The dataset contains 224,316 chest radiographs from 65,240 patients, labelled for 14 observations with uncertainty labels, and was released with a competition framework. The case shows how a benchmark and a dataset can be bundled, and how the split and label conventions become part of the benchmark's meaning.

**BraTS 2024 (documented).** The 2024 challenge focused on post-treatment glioma segmentation, with roughly 2,200 cases across seven contributing centres and 1,350 cases in the initial training release. The dataset is distributed through Synapse with registration. The case shows a benchmark where the test labels are hidden, which changes the contamination calculus.

**All of Us (documented).** The Curated Data Repository version 9 was released on 26 June 2026, with data from more than 747,000 participants, a data cutoff of 1 January 2025, and both registered and controlled tiers. The case shows a dataset whose value is in its population diversity and whose access is tiered, so the citation must state the tier.

**EHRSHOT (documented).** The benchmark contains 6,739 patients and 41.6 million clinical events from Stanford Medicine, released with a research data use agreement, and was published in the NeurIPS 2023 Datasets and Benchmarks track. The case shows a benchmark designed for few-shot evaluation, where the number of examples is itself part of the task definition.

### Documented facts and inferences

| Statement | Status |
|---|---|
| MIMIC-IV v3.1 counts | Documented on PhysioNet |
| PTB-XL size and licence | Documented in the data paper and repository |
| CheXpert size and labels | Documented in the AAAI paper |
| BraTS 2024 case count | Documented in the challenge paper and Synapse |
| All of Us v9 release and size | Documented by NIH and the program |
| EHRSHOT size and venue | Documented in the benchmark paper |
| Most teams check the licence after training | Inference from practice |

### What mature teams do (inference)

- They record the version and a file hash at download.
- They read the licence before training, not before publishing.
- They keep the benchmark split specification with the results.
- They maintain a private holdout for claims that matter.
- They re-check the dataset when a new release appears.

The practices are not exotic. They are the difference between a result that can be defended and one that has to be explained.

### What the cases have in common

- Each dataset has a version and a DOI, and the version is part of the citation.
- Each has an access tier that determines who can reproduce the work.
- Each has a population with known limits, stated in the documentation.
- Each has been used in papers that do not always state the version.

The last point is the practical problem. A dataset can be well documented and still be cited without the version, because the citation style does not require it. The register in section 7 exists to make the version impossible to omit.

### A comparison of access models

| Dataset | Access | Who can reproduce | Citation burden |
|---|---|---|---|
| PTB-XL | Open, CC BY 4.0 | Anyone | Version and DOI |
| CheXpert | Registration | Anyone who registers | Terms and version |
| MIMIC-IV | Credentialed | Anyone who completes training | DUA and version |
| All of Us | Registered and controlled | Approved researchers | Tier and version |
| EHRSHOT | Research DUA | Approved researchers | DUA and version |
| BraTS | Registration via Synapse | Registered participants | Challenge rules and version |

The citation burden rises with the access restriction. A paper using credentialed data has to describe the route, the agreement, and the version, or the reader cannot tell what was used.

### Three citations compared

A weak citation: "MIMIC-IV (Johnson et al.)." It names a dataset and a paper. It does not name a version, a DOI, an access route, or a subset, and a reader cannot tell which release was used.

A better citation: "Johnson, A., et al. MIMIC-IV (version 3.1). PhysioNet, 2024. https://doi.org/10.13026/kpb9-mt58. Accessed 26 October 2026 under the PhysioNet credentialed data use agreement." It names the version, the repository, the DOI, and the access route. A reader can apply for the same data.

A complete citation adds the module, the count used, and the file hash: "…ICU module, 65,366 unique ICU patients; files hashed on download." The addition is what makes the result reproducible rather than merely citable.

The progression costs a few minutes and moves the citation from a name to a record.

### What the access tier implies for the paper

| Tier | What the paper must describe | What the reader can do |
|---|---|---|
| Open | Version and DOI | Download and rerun |
| Registered | Registration process and terms | Register and rerun |
| Credentialed | Training and agreement | Apply and rerun |
| Controlled | Approval and output review | Apply and wait |
| Licensed | Licence terms and restrictions | Negotiate access |

A paper that does not describe the route leaves the reader unable to tell which row applies, and the reproducibility claim becomes ambiguous.

- The route is part of the methods, not the acknowledgements.
- The version is part of the citation, not the appendix.
- The licence is part of the project record, not the legal file.
- The split is part of the result, not the supplement.
- The hash is part of the reproduction package, not an afterthought.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Version omitted | Citation has a name and a DOI only | Object under-specified | Counts cannot be reproduced | Record version and date |
| Count without a unit | "364,627 patients" with no module | Patient and stay counts conflated | Inflated population claim | Name the unit |
| Licence read late | Access was free, use unclear | Access confused with permission | Product cannot use the data | Read the licence first |
| Silent licence change | Terms differ from the saved copy | Repository updated terms | Past use may be non-compliant | Record the licence version |
| Account sharing | Colleague provided a copy | Credentialing bypassed | Governance breach | Use the official route |
| Row-level split | Performance much higher than other papers | Patient leakage | Optimistic benchmark | Patient-level split |
| Public test set reuse | Leaderboard has hundreds of entries | Test used for selection | Saturated and contaminated | Private holdout |
| Benchmark metric mismatch | High score, poor clinical utility | Metric optimised without the task | Misleading progress | Decision-relevant metric |
| Dataset deprecation | Repository marks the record superseded | New version released | Citation no longer resolves | Update the register |
| Field unavailable | Required variable missing | Access specification not checked | Analysis cannot be run | Check the dictionary |
| Small-cell suppression | Subgroup counts missing | Privacy protection | Underpowered analysis | Plan for suppression |
| Cohort mismatch | Population differs from deployment | Selection not checked | Performance drop | Coverage table |
| Unvalidated subset | Subset citation without definition | Subset treated as the dataset | Results not comparable | Cite the subset explicitly |
| Hash not recorded | Files changed after download | Repository update | Result not reproducible | Record the hash |
| Documentation absent | No datasheet or dictionary | Dataset undocumented | Unknown limitations | Treat as a risk |
| Hash mismatch | Files differ from the record | Silent replacement | Result not reproducible | Re-download and re-run |
| Deprecated split | Split differs from the official one | Custom split unstated | Results not comparable | State the split |
| Metric subset | Only favourable metrics reported | Selective reporting | Misleading comparison | Report all official metrics |
| Population leakage | Test patients in training | Split not patient-level | Inflated performance | Patient-level split |
| Licence expiry | Term ended | Time-limited agreement | Use becomes non-compliant | Track the term |

### A dataset audit in ten checks

- Does the citation include creator, title, repository, version, year, DOI, and access date?
- Is the licence recorded and mapped to the four uses?
- Is the access tier stated?
- Is the file hash recorded?
- Does the coverage table show the deployment-relevant subgroups?
- Is the benchmark split unit stated?
- Is the contamination risk assessed?
- Is the metric decision-relevant?
- Is the version pinned in the pipeline?
- Is there a trigger for a new release?

A dataset used for a claim that leaves the team should pass at least eight of these checks.

- A citation without a version cannot be reproduced.
- A citation without a licence cannot be reused.
- A benchmark without a split cannot be compared.
- A benchmark without a contamination check cannot be trusted.
- A coverage table without outcome counts cannot support a subgroup claim.

Each check corresponds to a field in the register, and the register is the artefact that survives the project.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The repository record is authoritative | Version and licence are checked there | Mirror or redistribution | Resolve the DOI |
| The version label is stable | Reproducibility depends on it | Silent update | Record hash and date |
| The licence covers the planned use | The product may use the data | Research-only clause | Read the licence |
| The access route is available to reviewers | Reproducibility depends on it | Credentialing barrier | Describe the route |
| The population matches the claim | Applicability depends on it | Different setting or era | Coverage table |
| The split is patient-level | Performance is honest | Row-level split | Read the split spec |
| The benchmark is uncontaminated | Scores are comparable | Public test reuse | Check history |
| The metric reflects the decision | Progress is meaningful | Accuracy for imbalanced data | Map metric to decision |
| The subset is representative | Subset results generalise | Convenience subset | Compare to the parent |
| The documentation is accurate | Limitations are known | Datasheet absent | Read the dictionary |
| The benchmark version is fixed | Results are comparable | Benchmark updated | Record the version |
| The split is the official one | Comparison is valid | Custom split | State the split |
| The access term has not expired | Use is lawful | Time-limited agreement | Track the expiry |
| The derived data are permitted | Publication is lawful | Derivative clause | Read the licence |
| The population is stable across releases | Trends are interpretable | New sites added | Compare release notes |

Two limits are structural. First, a dataset citation cannot establish that the data are suitable; it establishes which data were used. Second, a benchmark cannot establish clinical utility; it establishes relative performance on a task.

A third limit is legal. The licence and the data use agreement bind the researcher, and the interpretation of a clause can require counsel. The procedure records the clause; it does not resolve it.

## 12. Exercises

**Exercise 1 — Citation anatomy (recall, 20 minutes).**
Take three dataset citations from your own work and mark the creator, title, repository, version, year, identifier, and access date.
Deliverable: three annotated citations with missing elements flagged.
Acceptance criteria: every citation is assessed against all eight elements; at least one missing element is found.
Time: 20 minutes.

**Exercise 2 — Licence matrix (application, 30 minutes).**
For one dataset, map the licence to train, evaluate, publish, and deploy.
Deliverable: a four-cell matrix with clause references.
Acceptance criteria: each cell cites the clause or says "silent"; silent is treated as unresolved, not permitted.
Time: 30 minutes.

**Exercise 3 — Count reconciliation (calculation, 30 minutes).**
Take a dataset description and reconcile every count with its unit against the repository.
Deliverable: a table of quantity, count, unit, and source.
Acceptance criteria: every count has a unit; at least one ambiguity is resolved; the version is recorded.
Time: 30 minutes.

**Exercise 4 — Benchmark integrity (critique, 35 minutes).**
Assess a benchmark you use for split unit, contamination, and metric relevance.
Deliverable: a one-page assessment with three findings and a recommendation.
Acceptance criteria: the split unit is stated; a contamination risk is assessed; the metric is compared with the decision.
Time: 35 minutes.

**Exercise 5 — Coverage table (application, 35 minutes).**
Build a coverage table for a dataset by site, year, and key subgroups, with outcome counts.
Deliverable: a table with counts and a note on underpowered cells.
Acceptance criteria: outcome counts are included; at least one underpowered subgroup is identified; the source of each field is recorded.
Time: 35 minutes.

**Exercise 6 — Citation rewrite (application, 25 minutes).**
Rewrite three dataset citations so that a reader can obtain the same version and files.
Deliverable: three citations with version, DOI, access date, and licence.
Acceptance criteria: each citation resolves to a version; the licence is named; the access route is stated.
Time: 25 minutes.

**Exercise 7 — Register scoring (calculation, 40 minutes).**
Run the script on a register of at least eight datasets and interpret the results.
Deliverable: the script output and a paragraph on the three weakest records.
Acceptance criteria: the output is reproducible; the interpretation names the missing fields; a remediation is proposed.
Time: 40 minutes.

**Exercise 8 — Maintenance design (design, 30 minutes).**
Design a dataset version and licence register with review triggers.
Deliverable: a table with dataset, version, licence, owner, review date, and trigger.
Acceptance criteria: every entry has a trigger; licence changes are included as a trigger; the register fits a monthly maintenance budget.
Time: 30 minutes.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Anatomy | All eight elements assessed | Only name and DOI |
| 2 Licence | Clause references | "Open access" assumed |
| 3 Counts | Unit for every count | Headline number only |
| 4 Benchmark | Split and contamination assessed | Metric only |
| 5 Coverage | Outcome counts | Totals only |
| 6 Citation | Resolves to a version | No version |
| 7 Register | Reproducible output | Scores without interpretation |
| 8 Maintenance | Triggers present | Annual review only |

The exercises are designed to be done on a live register. A learner who uses a public dataset will still find missing fields, because public repositories vary in what they require.

Exercise 7 is the one that most often changes behaviour, because the score makes the gaps visible in a way that a checklist does not.

A learner who completes Exercise 2 usually finds that at least one clause they assumed was permissive is silent, and silence is the most common answer in licence work.

The exercises assume access to at least one dataset's documentation. A learner without access can use the repository pages cited in section 16, which publish enough to complete most of the work.

## 13. Self-check answers

**1. What are the elements of a dataset citation?**
Creator, title, repository, version, year, persistent identifier, access date, and licence. The version and access date are the elements most often omitted and the ones that make the citation reproducible.

**2. Why is a DOI not enough to reproduce a dataset result?**
Because a DOI resolves to a landing page, and the files behind it can be replaced. Reproduction requires the version, the access date, and ideally a hash of the files used.

**3. What is the difference between access and licence?**
Access is the ability to obtain the data; licence is the permission to use it in particular ways. A dataset can be openly downloadable and restricted to non-commercial research.

**4. Why does a row-level split inflate performance?**
Because records from the same patient can appear in both training and test, so the model can learn patient-specific features rather than the target pattern. The inflation is invisible in the reported metric.

**5. What is benchmark contamination?**
The presence of test items, or close variants, in the data used to develop or select a model. It inflates scores and is hard to detect without checking the training data and the benchmark history.

**6. Why is a metric mismatch a source problem?**
Because the benchmark's metric defines what the field treats as progress, and a metric that does not reflect the decision rewards models that are not clinically useful. The source is the benchmark specification, not the model.

**7. What does a coverage table add to a dataset description?**
It shows who is in the data, in what numbers, with how many outcome events. It converts a population claim into a set of denominators that a reviewer can check.

**8. Why should a subset be cited separately?**
Because a subset has its own inclusion criteria and its own counts, and citing the parent dataset implies a population the analysis did not use.

**Rubric for Exercises 4 and 7.** A strong benchmark assessment states the split unit, assesses contamination with a method, and compares the metric with the decision. A weak assessment reports the score. A strong register interpretation names the missing fields and proposes a fix. A weak interpretation reports the mean score.

**9. Why is a file hash useful when the repository already has a version?**
Because a version label can be reused or a file can be replaced without a version change. The hash identifies the exact bytes used, which is the strongest available evidence of reproduction.

**10. What does a licence that is silent on model weights mean?**
It means the question is unresolved, not permitted. Models can be considered derivative works depending on the jurisdiction and the licence, and the safe course is to ask the owner and record the answer.

**11. Why is a temporal split a stronger test than a random split?**
Because it tests generalisation forward in time, which is what deployment requires. A random split tests performance on the same period and can hide drift.

**12. What is the risk of a benchmark whose test labels are public?**
The test set can be used for model selection across the community, which saturates the leaderboard and makes the score a measure of familiarity rather than generalisation.

**13. Why should a dataset subset be defined in the citation?**
Because a subset has its own inclusion criteria and counts, and a reader who downloads the parent will not obtain the same cohort.

**14. What does a coverage table reveal that a total count hides?**
The distribution of the population and the number of outcome events in each subgroup, which determines whether a subgroup claim is supportable.

**15. How does a data use agreement differ from a licence?**
A licence is a permission attached to the data; a data use agreement is a contract between the provider and the user that can add obligations, restrictions, and liability. Both can bind, and the stricter one governs.

## 14. Interview questions

**1. How do you cite a dataset so that someone can reproduce your result?**
Strong answer: creator, title, repository, version, DOI, access date, licence, and a file hash where possible. Weak answer: name and DOI.

**2. What do you check before training on a new dataset?**
Strong answer: licence, access, version, population, and the field availability for the intended features. Weak answer: checks that it downloads.

**3. How do you know a benchmark result is comparable to another paper's?**
Strong answer: checks the split, preprocessing, metric, and version, and only compares when they align. Weak answer: compares the headline number.

**4. What is the risk of a public test set?**
Strong answer: it can be used for model selection across the community, which saturates and contaminates the benchmark; a private holdout is needed for a real claim. Weak answer: says public data are good for reproducibility.

**5. How would you evaluate a dataset's licence for a product?**
Strong answer: maps the clauses to train, evaluate, publish, and deploy, treats silence as unresolved, and gets legal review for derivatives and weights. Weak answer: reads the summary on the repository page.

**6. What does a dataset version change mean for a model?**
Strong answer: counts, labels, or fields can change, so the model may need retraining and the reported numbers need re-checking. Weak answer: assumes a patch is harmless.

**7. How do you handle a dataset that is credentialed?**
Strong answer: uses the official route, records the agreement, describes the access in the paper, and shares code so the analysis can be checked. Weak answer: uses a colleague's copy.

**8. What makes a benchmark decision-relevant?**
Strong answer: the metric reflects the clinical decision, the population matches the intended use, and the split tests the relevant generalisation. Weak answer: says accuracy is enough.

**9. How do you detect contamination in an image benchmark?**
Strong answer: checks for near-duplicate images, patient overlap, pretraining corpora, and the benchmark's submission history. Weak answer: checks for exact duplicates only.

**10. What should a dataset register contain?**
Strong answer: name, version, DOI, licence, access tier, owner, review date, and triggers. Weak answer: a list of links.

**11. How would you decide between two datasets with similar performance?**
Strong answer: compares population coverage, licence, access latency, version stability, and benchmark integrity, not only the score. Weak answer: picks the higher score.

**12. What is the risk of using a benchmark as both development and evaluation data?**
Strong answer: the evaluation is no longer independent, and the reported performance reflects selection on the same data. Weak answer: says it is fine if the split is respected.

**13. How do you handle a dataset that is no longer maintained?**
Strong answer: records the version and hash, checks the licence for continued use, and treats the lack of maintenance as a risk for future reproduction. Weak answer: keeps using it without recording the risk.

**14. Why is "publicly available" not a licence statement?**
Because availability describes access, not permission. The licence determines what may be done with the data, and a public dataset can still restrict commercial use or redistribution.

**15. What would you check before publishing a derived dataset?**
Strong answer: the licence's derivative and redistribution clauses, the privacy risk of the derivative, and whether the derivative could allow re-identification. Weak answer: checks only whether the source was public.

**16. How do you make a credentialed dataset result reproducible?**
Strong answer: describes the access route, records the version and agreement, shares the analysis code, and reports the exact cohort definition. Weak answer: says the data cannot be shared and stops.

**17. What is the difference between a benchmark and a dataset?**
A dataset is a collection of records; a benchmark is a task specification built on a dataset, with splits, metrics, and rules. The same dataset can support several benchmarks with different meanings.

**18. Why should a benchmark's split be part of the citation?**
Because the split determines what the result measures. A result on a row-level split and a result on a patient-level split are not comparable even when the dataset and metric are identical.

**19. What does a repository deprecation mean for a citation?**
It means the citation may stop resolving or the files may be removed. The register should record the successor version and the date of migration.

**20. How would you check a dataset's population against an intended use?**
Build a coverage table, compare each dimension to the intended use, and identify the subgroups that are absent or underpowered.

**21. What is the risk of a custom split that is not reported?**
The result is not comparable with any other result on the benchmark, and the reader cannot tell whether the difference is due to the model or the split.

**22. Why should a licence be recorded at the version level?**
Because terms can change between releases, and a project that began under one set of terms may be governed by another after an update.

## 15. Key terms

- **Dataset** — a structured collection of records with documentation.
- **Release** — a published version of a dataset.
- **Version** — an identifier for a specific release.
- **Patch** — a correction that may not change the schema.
- **Landing page** — the repository page a DOI resolves to.
- **File hash** — a digest of file contents.
- **Datasheet** — documentation of a dataset's motivation and composition.
- **Data dictionary** — definitions of fields and values.
- **Provenance** — the origin and processing history of data.
- **Access tier** — the route by which data are obtained.
- **Credentialed access** — access after identity verification and training.
- **Controlled access** — access with institutional approval and output review.
- **Licence** — the permission to use data in specified ways.
- **Derivative work** — an artefact substantially derived from the data.
- **Redistribution** — providing data or derivatives to a third party.
- **Field of use** — the permitted purpose of a licence.
- **Benchmark** — a fixed task, dataset, and metric.
- **Split** — the partition into training, validation, and test sets.
- **Patient-level split** — a split keeping a patient's records together.
- **Temporal split** — a split by time.
- **Site-level split** — a split holding out entire sites.
- **Contamination** — test data present in development.
- **Leaderboard** — a ranked list of benchmark submissions.
- **Saturation** — top results within noise of each other.
- **Metric gaming** — optimising the metric without the task.
- **Coverage table** — counts of a population by key subgroups.
- **Subset** — a defined portion of a dataset.
- **Data use agreement** — a contract governing access and use.
- **De-identification** — removal of identifying information.
- **Small-cell suppression** — hiding counts below a threshold.
- **Repository** — the service that hosts and versions a dataset.
- **Persistent identifier** — an identifier that survives location changes.
- **Provenance record** — documentation of origin and processing.
- **Data sheet** — a structured description of a dataset's composition.
- **Benchmark suite** — a collection of tasks sharing a dataset.
- **Holdout** — data reserved for final evaluation.
- **Private holdout** — a holdout not publicly available.
- **Prospective evaluation** — evaluation on data collected after deployment.
- **Near-duplicate** — a record that closely resembles another across splits.
- **Benchmark version** — a specific release of a benchmark's data and rules.
- **Evaluation protocol** — the rules for running and reporting an evaluation.
- **Reproduction package** — the code and instructions needed to repeat an analysis.
- **Data availability statement** — the paper section describing access.
- **Embargo** — a period during which data are available only to specific users.
- **Linked data** — records joined across datasets.
- **Record linkage** — the process of joining records across sources.
- **Synthetic cohort** — generated data resembling a real cohort.
- **Utility metric** — a measure of how useful synthetic data are for a task.
- **Data availability statement** — the paper section describing access.
- **Reproduction package** — the code and instructions needed to repeat an analysis.
- **Benchmark leaderboard** — the public ranking of submissions.
- **Holdout set** — data reserved for final evaluation.
- **Data provenance** — the documented origin of a dataset.
- **Dataset deprecation** — the retirement of a dataset version.
- **Provenance metadata** — structured fields describing a dataset's origin.
- **Data availability** — whether and how data can be obtained.

## 16. Sources

All URLs were checked on 26 October 2026. Counts are those published for the version named.

**Johnson, A., Bulgarelli, L., Pollard, T., et al.** MIMIC-IV (version 3.1). PhysioNet, 2024. https://doi.org/10.13026/kpb9-mt58
Good for: the dataset, its counts, its schema, and its access terms. Not for: a validated cohort definition for a specific study.

**Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al.** "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7:154, 2020. https://doi.org/10.1038/s41597-020-0495-6
Good for: the dataset description, labels, and recommended splits. Not for: current clinical ECG criteria.

**Irvin, J., Rajpurkar, P., Ko, M., et al.** "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison." *AAAI* 2019. https://doi.org/10.1609/aaai.v33i01.3301590
Good for: the dataset, the uncertainty labels, and the benchmark design. Not for: clinical reporting standards.

**BraTS 2024 challenge.** "The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI." arXiv:2405.18368. https://arxiv.org/abs/2405.18368
Good for: the task definition, case counts, and evaluation. Not for: clinical deployment guidance.

**National Institutes of Health.** All of Us Research Program, Curated Data Repository version 9. https://support.researchallofus.org/
Good for: the release contents, tiers, and participant counts. Not for: a specific cohort definition; the workbench documentation is required.

**Wornow, M., Thapa, R., Steinberg, E., Fries, J. A., and Shah, N. H.** "EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models." *NeurIPS 2023 Datasets and Benchmarks*. https://arxiv.org/abs/2307.02028
Good for: the benchmark design and few-shot task definitions. Not for: clinical validation of a model.

**Wilkinson, M. D., et al.** "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data* 3:160018, 2016. https://doi.org/10.1038/sdata.2016.18
Good for: the vocabulary of findability, accessibility, interoperability, and reuse. Not for: a compliance checklist.

**Gebru, T., et al.** "Datasheets for Datasets." https://arxiv.org/abs/1803.09010
Good for: the questions a dataset should document. Not for: a guarantee that the documentation is accurate.

**DataCite.** Metadata schema and DOI services. https://datacite.org/
Good for: dataset DOI metadata and version information. Not for: licence interpretation.

**PhysioNet.** Repository and credentialed access. https://physionet.org/
Good for: access to clinical datasets and their documentation. Not for: a substitute for reading the data use agreement.

**Synapse.** Platform for challenge datasets. https://www.synapse.org/
Good for: challenge data access and submission. Not for: long-term citation stability without a DOI.

**ML Commons.** Croissant dataset metadata format. https://mlcommons.org/
Good for: machine-readable dataset metadata for ML. Not for: clinical or legal documentation.

**CoreTrustSeal.** Repository certification. https://www.coretrustseal.org/
Good for: evidence that a repository has trustworthy stewardship practices. Not for: the quality of a specific dataset.

**European Commission.** European Health Data Space. https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en
Good for: the EU framework for secondary use of health data. Not for: the terms of a specific dataset.

**U.S. Department of Health and Human Services.** HIPAA de-identification guidance. https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
Good for: the two de-identification approaches under HIPAA. Not for: GDPR or other jurisdictions.

**PhysioNet.** MIMIC-IV documentation and schema. https://physionet.org/content/mimiciv/3.1/
Good for: the schema, module descriptions, and counts. Not for: clinical interpretation of the variables.

**Synapse.** BraTS challenge documentation. https://www.synapse.org/
Good for: challenge data access, rules, and submission. Not for: a stable citation without a DOI or version.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_clinical-background.md`](05_clinical-background.md)
- Next topic: [`07_industry-and-market.md`](07_industry-and-market.md)
- Methods: [`04_methods.md`](04_methods.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 21, dataset selection: [`../21_datasets/01_five-gate-decision.md`](../21_datasets/01_five-gate-decision.md)
- Module 21, access and licensing: [`../21_datasets/02_access-and-licensing.md`](../21_datasets/02_access-and-licensing.md)
- Module 02, data landscape: [`../02_data_landscape/`](../02_data_landscape/)
- Module 16, privacy and security: [`../16_privacy_security/`](../16_privacy_security/)
- Module 10, imaging benchmarks: [`../10_imaging/`](../10_imaging/)
- Module 11, signal benchmarks: [`../11_signals/`](../11_signals/)
- Module 15, model registries: [`../15_mlops/`](../15_mlops/)
- Module 19, capstone datasets: [`../19_capstone/`](../19_capstone/)
- Survey dataset table: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md) section 6

The next topic moves from datasets to industry evidence, where the source problem is method disclosure rather than version control.

A reader who wants a single habit from this file should take this one: every dataset count needs a unit, and every dataset citation needs a version.
