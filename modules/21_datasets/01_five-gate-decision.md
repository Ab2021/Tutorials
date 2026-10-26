# The Five-Gate Dataset Decision

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 01 of 12
> Estimated study time: 3 hours
> Prerequisites: Module 02
> Version 2026.3

## 1. Why this topic exists

Dataset selection is usually run as procurement: someone finds a file, checks that it downloads, and the project starts. That sequence produces most of the expensive data failures in biomedical product work, because a dataset is not a neutral input. It is an argument about what can be measured, in whom, at what time, under what legal terms, and against what evaluation. If the argument fails, no amount of modeling recovers it.

The failures are predictable and they cluster at the start of projects:

- A readmission model is built on claims, then the team learns that the outcome flag includes transfers to other hospitals, so "readmission" means "billing event" rather than "return to care."
- A sepsis alert is trained on features that are recorded after the alert would fire, so offline AUROC is high and prospective performance collapses.
- A model is trained on a non-commercial research dataset, and at commercial diligence the company discovers that the licence restricts product development, not only redistribution.
- A benchmark delta of two points is reported as progress, then disappears when the split is changed from image-level to patient-level.
- A single academic ICU is used to claim national generalization, and the first community-hospital pilot shows a drop the team cannot explain.

Each of these is a decision failure, not a modeling failure. The model did what it was asked. The dataset could not support the claim, and nobody checked before the claim was made.

The people blocked by this are specific. The data lead cannot defend a choice in a design review. The regulatory lead cannot write a validation section that survives review. The product manager cannot price the access latency or the licence risk. The clinical lead cannot say whether the model will work in the intended setting. A five-gate screen gives each of them a shared artefact and a shared vocabulary.

Why five gates and not a weighted score. A weighted score lets a strong result on one dimension compensate for a zero on another. That is the wrong arithmetic for admission decisions. A dataset that cannot measure the outcome at the decision time is not "70 percent suitable"; it is unusable for that claim. A dataset whose licence forbids commercial training is not "mostly fine"; it is disqualifying for a commercial product. The gates are conjunctive. Scores of 0 to 2 exist to structure the evidence and expose disagreement, not to be averaged into a verdict.

Why the order matters. The gates are ordered by cost and irreversibility. Gate 1 and Gate 2 are desk work that can be done in a day. Gate 3 and Gate 4 involve institutions, signatures, fees, and weeks of latency. Gate 5 depends on the metric and the split, which are only meaningful once the first four are settled. Running the gates out of order is how teams spend six weeks on credentialing before discovering that the dataset has no outcome at the prediction time.

This topic is the framework. It is not a catalogue: topics 03 to 10 cover modalities. It is not legal advice: topic 02 covers access and licensing mechanics in detail. It is a repeatable decision procedure with evidence standards, a worked case, and executable checks.

> **Note:** A gate score is a record of evidence, not a measurement. Two reviewers can score the same dataset differently and both be right, because they are describing different claims. Always score a candidate against a named claim and a named deployment setting. A dataset has no fitness in the abstract.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a decision statement that names the actor, the action, the timing, the population, and the cost of each error type, and use it as the reference against which every gate is scored. Assessed by Exercise 1 and self-check item 1.
2. **Score** at least three candidate datasets on all five gates, attaching a specific piece of evidence to every score and identifying every veto. Assessed by Exercise 2 and the worked example.
3. **Conduct** a temporal audit of a feature table, classify each feature as available, boundary, or leaking, and state the consequence for the claimed prediction time. Assessed by Exercise 3 and self-check item 4.
4. **Build** a population coverage table by site, age, sex, race or ethnicity, payer, and year, and identify the subgroup that is too small to support a reported metric. Assessed by Exercise 4.
5. **Map** a licence to a product plan clause by clause, distinguishing permitted use, permitted with conditions, prohibited, and silent. Assessed by Exercise 5.
6. **Choose** an evaluation metric from the decision's cost structure rather than from convention, and justify the choice in one paragraph. Assessed by Exercise 6.
7. **Defend** a rejection in writing with evidence, including what would have to change for the candidate to become admissible. Assessed by Exercise 7.
8. **Recognise** when no available dataset can support the claim, and convert the deliverable from a model to a data-collection or study-design plan. Assessed by Exercise 8.

Each outcome is observable in a document a reviewer can mark. "Understands dataset quality" is not an outcome; "produces a coverage table with a named underpowered subgroup" is.

## 3. The core mental model

The model is a funnel with five conjunctive filters. Candidates enter; evidence is attached at each filter; zeros exit. What leaves the funnel is not "the best dataset" but the admissible set, which may be empty.

```text
                    DECISION STATEMENT
        actor | action | timing | population | error costs
                              |
                              v
                 CANDIDATE LONG LIST (>= 3)
             primary candidate + comparator + fallback
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
   GATE 1                GATE 2                GATE 3
   Decision fit          Population fit        Access fit
   outcome, exposure,    site, era, setting,   route, latency,
   confounders at        subgroup coverage     cost, granularity
   decision time              |                     |
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                         GATE 4
                         License fit
                  commercial use, training,
                  redistribution, jurisdiction
                              |
                              v
                         GATE 5
                         Benchmark fit
                  task, metric, clean test set,
                  contamination status
                              |
                              v
                    ADMISSIBLE SET
              (possibly empty -> change the plan)
                              |
                              v
                  BENCHMARK DESIGN + EVIDENCE LEDGER
```

Read the diagram in one direction only. Evidence flows forward; a failure at any gate sends the candidate back to the long list, not forward to a weighted score. The reverse arrow that teams actually use, "we already have the data, so let us find a claim it supports," is the anti-pattern this topic exists to prevent.

**Why the gates are conjunctive.** Each gate answers a different question, and the questions are not substitutes. Decision fit asks whether the variable you need exists at the moment you need it. Population fit asks whether the people in the data resemble the people who will be affected. Access fit asks whether you can obtain the data at the required granularity before the deadline. License fit asks whether the owner permits the use you plan. Benchmark fit asks whether the evaluation measures the decision. A dataset can pass four and fail the fifth, and the fifth still determines whether the result means anything.

**Why the scores are ordinal.** A score of 2 means "direct evidence, no material gap." A score of 1 means "usable with a stated limitation or a design change." A score of 0 means "cannot support the claim as written." The numbers force a reviewer to separate "I am uncomfortable" from "this is disqualifying." They do not license arithmetic across gates, and they do not survive a change in the claim.

**Gate interaction.** The gates are not statistically independent, even though they are evaluated separately:

- Access can cure a population problem. A single-site dataset fails population fit for a national claim, but a credentialed multi-site dataset with the same variables passes.
- License can constrain benchmark design. If the licence forbids releasing derived data, an external replication package may be impossible, which weakens the benchmark's credibility even if the metric is right.
- Label provenance sits between Gate 1 and Gate 5. A label that is defined by a billing code is a decision-fit problem when the decision is clinical, and a benchmark-fit problem when the metric is treated as ground truth.
- Version drift touches every gate. A new release can change the cohort, the variable definitions, the licence, and the published split at once.

**The evidence ledger.** For each candidate and each gate, record four fields: the claim being tested, the evidence inspected, the verdict, and the residual uncertainty. A score without an evidence line is an opinion. The ledger is the artefact that survives staff turnover and makes the decision auditable a year later.

**The empty admissible set is a result.** If no candidate passes, the correct deliverable is a data-collection plan, a partnership, a narrower claim, or a stop decision. Teams treat an empty set as a failure of the search and keep looking until they find a dataset that passes by lowering the claim. That is how unsupportable claims enter the pipeline.

| Gate | Core question | Primary artefact | Typical owner |
|---|---|---|---|
| 1 Decision fit | Does the data contain the outcome, exposure, and confounders at the decision time? | Temporal audit table and label definition | Clinical or scientific lead |
| 2 Population fit | Do site, era, setting, and subgroups match the deployment population? | Coverage table with denominators | Data lead |
| 3 Access fit | Can we obtain the required granularity before the deadline? | Access route, latency, and cost estimate | Data or legal lead |
| 4 License fit | Does the licence permit training, evaluation, release, and deployment? | Clause-by-clause matrix | Legal or privacy lead |
| 5 Benchmark fit | Does the task and metric correspond to the decision on a clean test set? | Metric-to-decision memo and split specification | Evaluation lead |

## 4. Definitions and vocabulary

The vocabulary is split into six families because the words are used loosely in practice and the looseness causes most disagreements. Each entry gives the definition used in this module, the common misuse, and the unit or coding system where one applies.

### 4.1 Decision and estimand terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Decision statement | A sentence naming the actor, the action, the timing, the population, and the cost of false positives and false negatives. | Replaced by a model objective such as "predict sepsis." | Free text, one sentence |
| Prediction time | The moment at which all inputs must be available and the output is acted on. | Confused with the outcome time or the data-extraction date. | Timestamp, often hours from arrival |
| Index date | The anchor date from which follow-up or feature windows are measured. | Set to the outcome date, which leaks future information. | Date or timestamp |
| Estimand | The precise quantity the analysis targets, including population, outcome, and intervention. | Used as a synonym for "question." | Structured definition |
| Counterfactual | What happens to the same population under current practice without the model. | Omitted, making the value of the model unmeasurable. | Scenario description |
| Label | The target variable used for supervision or evaluation. | Treated as ground truth rather than as a measurement with error. | Variable with provenance |
| Label provenance | The origin and process that produced the label, including coding system, abstraction rules, and adjudication. | Recorded as "from EHR." | Provenance record |
| Outcome ceiling | The best performance attainable given label noise and irreducible uncertainty. | Assumed to be 100 percent. | Proportion or metric value |

The decision statement is the anchor for every later judgement. A claim such as "predict sepsis" is not scoreable, because it does not say when, in whom, or what happens if the alert is wrong. A decision statement is scoreable: "At hour 6 after emergency department arrival, flag adults admitted with suspected infection who will meet Sepsis-3 criteria within the next 6 hours, so that a rapid-response nurse can review them; a false alert costs 10 minutes of nurse time, a missed case costs a delayed antibiotic." Every gate is then evaluated against those words.

The estimand deserves separate attention because machine-learning practice often skips it. A model that predicts "sepsis at any time during the admission" and a model that predicts "sepsis onset in the next 6 hours" can be built from the same table and have very different clinical meaning. The first can use discharge diagnoses; the second cannot. Gate 1 fails for the first if the deployment decision is prospective, even though the dataset "contains sepsis."

### 4.2 Population terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Target population | The people who will be affected by the deployed decision. | Confused with the source population. | Eligibility criteria |
| Source population | The people actually captured in the dataset, after inclusion and exclusion. | Assumed to equal the target population. | Cohort definition |
| Coverage table | Counts of the source population by site, year, age, sex, race or ethnicity, payer, and setting. | Replaced by a total row count. | Counts and proportions |
| Transportability | Whether a model trained in one setting retains performance in another. | Asserted from a single external test. | Performance delta by site |
| Healthy-volunteer bias | Selection of people healthier than the target population because participation requires effort. | Ignored when the cohort is large. | Standardized morbidity ratio |
| Era effect | Change in coding, treatment, or measurement over calendar time. | Treated as random noise. | Calendar year |
| Setting | The care context: emergency department, ICU, ward, ambulatory, home. | Recorded as "hospital." | Controlled vocabulary |
| Unit of analysis | Whether a row is a patient, an encounter, a stay, a sample, or a cell. | Left implicit, producing double counting. | Entity key |

A coverage table is not a demographic summary. It is a set of denominators attached to the subgroups for which performance will later be reported. If the smallest site contributes 210 stays and the outcome prevalence is 8 percent, that site has about 17 events. Any per-site metric computed on 17 events is a wide interval, not a site-level result. The table should make that visible before modeling, not after a reviewer asks.

Unit of analysis is the quietest failure in this family. MIMIC-IV, for example, is organized around patients, hospitalizations, and ICU stays, and a table joined at the wrong level will duplicate rows or drop them. The survey file gives the 3.1 release as 364,627 patients, 546,028 hospitalizations, and 94,458 ICU stays; those three numbers describe three different units, and a claim about "patients" cannot be computed from a stay-level table without deduplication.

### 4.3 Access terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Open access | Direct download without identity verification. | Assumed to mean unrestricted use. | Access tier |
| Registered access | Account creation and click-through terms. | Treated as equivalent to credentialed access. | Access tier |
| Credentialed access | Identity verification, training, and a signed data use agreement. | Delegated to a colleague's account. | Access tier |
| Controlled access | Institutional sign-off, approved analysis plan, and output review. | Planned as a two-week step. | Access tier |
| Licensed access | Contractual access with fees and field-of-use limits. | Treated as a purchase of the data. | Access tier |
| Contributor-only | Access tied to membership or data contribution. | Assumed to permit row-level export. | Access tier |
| Data use agreement (DUA) | Contract governing access, use, retention, and redistribution. | Read as a licence to publish anything derived. | Contract |
| Credentialing latency | Calendar time from application to first data access. | Estimated from the download time of a demo. | Weeks |
| Granularity | The finest unit available: row-level, aggregate, query-only. | Assumed from the dataset description. | Access specification |

The access tier determines what questions are even askable. A query-only product can return cohort counts and summary statistics; it cannot support training a row-level model or reproducing a split. A contributor-only network can be scientifically excellent and still fail Gate 3 for a team that is not a contributor. Topic 02 covers the mechanics of each tier, including the training requirements and the institutional roles.

### 4.4 License terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Commercial use | Use of data or outputs in a product, service, or revenue-generating activity. | Assumed to exclude internal research at a company. | Clause category |
| Derivative work | A new artefact substantially derived from the data, potentially including model weights. | Assumed to exclude trained models. | Clause category |
| Redistribution | Providing data or close derivatives to a third party. | Assumed to permit sharing features or synthetic samples. | Clause category |
| Model training | Use of data to fit, fine-tune, or distil a model. | Assumed to be covered by "research use." | Clause category |
| Publication review | A requirement that the owner review manuscripts or figures before release. | Discovered at submission. | Clause category |
| Field of use | The permitted purpose, therapeutic area, or population. | Ignored because the data were cheap. | Clause category |
| Jurisdiction | The legal territory where processing is permitted. | Broken by a default cloud region. | Clause category |
| Retention | The period for which data or derivatives may be kept. | Derivatives kept indefinitely after project end. | Clause category |

Licence fit is evaluated against the product plan, not against the research protocol. A clause that permits "research" may still forbid "commercial product development," and a clause that permits internal evaluation may forbid releasing model weights. The four outcomes used in this module are permitted, permitted with conditions, prohibited, and silent. Silent is not permission; it is a question for the owner.

### 4.5 Benchmark terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Benchmark | A fixed task, dataset, and metric used to compare methods. | Treated as a measure of clinical utility. | Task specification |
| Test set | Data held out from all model selection and tuning. | Reused across dozens of submissions. | Split specification |
| Patient-level split | A split that keeps all records from one patient on one side. | Replaced by a random row split. | Split key |
| Temporal split | A split by calendar time, with training before testing. | Replaced by a random split that leaks the future. | Split key |
| Site-level split | A split that holds out entire hospitals or scanners. | Absent from single-site datasets. | Split key |
| Contamination | Test items or near-duplicates present in training data. | Checked only for text benchmarks. | Proportion |
| Metric-to-decision mapping | An argument that the reported metric orders options the way the decision costs do. | Assumed because the metric is standard. | Memo |
| Precision at capacity | The fraction of true positives among the top-k ranked when capacity is fixed. | Replaced by AUROC because AUROC is familiar. | Proportion at k |
| Calibration | Agreement between predicted probabilities and observed frequencies. | Omitted when discrimination is good. | Calibration slope and intercept |
| Benchmark saturation | Top methods cluster within noise, so ranking is unstable. | Reported as a new state of the art. | Confidence interval overlap |

The benchmark gate is where "the dataset is fine" and "the evaluation is meaningful" separate. A dataset can support the task and still fail Gate 5 because the only public split is contaminated, the metric does not encode the decision, or the test set has been reused so often that the leaderboard order is noise. Topic 12 covers contamination and leaderboard pathology in depth.

### 4.6 Evidence and governance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Evidence ledger | A table recording claim, evidence, verdict, and residual uncertainty for each gate. | Replaced by a slide with scores. | Table |
| Veto | A zero on any gate, disqualifying the candidate for the named claim. | Averaged into a weighted score. | Boolean |
| Admissible set | Candidates with no veto, ranked only within the set. | Treated as a single winner. | Set |
| Data card | Structured documentation of composition, provenance, preprocessing, uses, and limitations. | Written after the model, from memory. | Document |
| Datasheet | Documentation following the Datasheets for Datasets structure: motivation, composition, collection, preprocessing, uses, distribution, maintenance. | Confused with a data dictionary. | Document |
| Model card | Documentation of a model's intended use, training data, evaluation, and limitations. | Treated as independent evidence. | Document |
| Risk-of-bias assessment | Structured judgement of whether a study's results are trustworthy. | Replaced by a checklist score. | PROBAST+AI domains |
| Applicability | Whether a study's population and setting match the review question. | Assumed from a similar disease area. | Domain judgement |

Documentation is not a substitute for evidence. A data card tells you what the custodian says about the data; it does not tell you whether the labels are correct, whether the split leaks, or whether the licence covers your product. Read the card, then inspect the data.

## 5. The mechanics, step by step

The procedure has 18 steps in five phases. Each step states its input, its output, the reason it exists, and the branch or edge case that changes the plan. Steps 1 to 4 are framing; 5 to 9 are Gates 1 and 2; 10 to 12 are Gates 3 and 4; 13 to 15 are Gate 5; 16 to 18 are decision, documentation, and re-entry.

Do not compress the framing steps. A team that starts at step 5 with a vague claim will score the gates against different claims and argue past each other.

### Phase A — Framing

**Step 1. Write the decision statement.**
Input: a product or research proposal, the clinical or operational workflow, and the names of the people who act on the output.
Output: one sentence with five slots filled: actor, action, timing, population, and error costs. A second sentence states the counterfactual.
Rationale: every later gate is scored against these words. Without them, "fit" is undefined and the review becomes a taste debate.
Branch: if the team cannot name the actor and the action, the project is not ready for dataset selection. Stop and return to problem framing (Module 06).
Edge case: the actor is a model rather than a person, as in an automated prior-authorization decision. Then name the human or system that consumes the output and the appeal path.

**Step 2. Fix the prediction time and the outcome window.**
Input: the workflow, the decision statement, and the clinical guideline that defines the event.
Output: an index time, a feature window, and an outcome window, all with units.
Rationale: the same dataset can pass or fail Gate 1 depending on these three numbers. Sepsis at hour 6 is a different task from sepsis at any time in the admission.
Branch: if the outcome window extends beyond the dataset's follow-up, Gate 1 fails for that outcome. Choose a shorter window or a different outcome.
Edge case: recurrent events. A patient can have more than one sepsis episode; decide whether the unit is the episode, the stay, or the patient before counting.

**Step 3. Define the target population and the deployment setting.**
Input: the intended customers, the care settings, and the eligibility rules.
Output: a written inclusion and exclusion list, plus the settings in scope: emergency department, ICU, ward, ambulatory, home.
Rationale: population fit is a comparison between this list and the coverage table. Without the list there is nothing to compare.
Branch: if the product will be sold to community hospitals but the only candidate is a tertiary academic center, Gate 2 is at most 1 and may be 0 for a national claim.
Edge case: the deployment population includes children, pregnant patients, or people with impaired decision-making capacity. Most adult ICU datasets exclude them; that is a scope limit, not a small gap.

**Step 4. Assemble a candidate long list of at least three datasets, including one fallback.**
Input: the survey file, domain knowledge, a literature search, and custodian catalogues.
Output: a table with name, custodian, version, unit of analysis, access tier, and the reason each candidate is plausible.
Rationale: a single candidate produces a justification, not a decision. A comparator makes the trade-offs visible; a fallback keeps the project alive if access fails.
Branch: if only one candidate exists, the deliverable is a feasibility memo that says so, plus a plan to create or buy a second source.
Edge case: two candidates from the same custodian may share the same failure. MIMIC-IV, MIMIC-CXR, and MIMIC-IV-Note are not independent for population fit; they are one institution.

### Phase B — Gates 1 and 2

**Step 5. Audit every feature for availability at the prediction time.**
Input: a feature list with source table, timestamp column, and clinical meaning.
Output: a three-way classification: available (recorded before prediction time), boundary (recorded at the boundary and only usable if complete), leaking (recorded after).
Rationale: temporal leakage is the most common reason a strong offline result fails in deployment. The audit is cheap desk work and catches the failure before compute is spent.
Branch: if more than a small fraction of features leak, the model is not a prospective model. Either move the prediction time later or rebuild the feature set.
Edge case: result time versus collection time. A lab collected at hour 5 but resulted at hour 9 is unavailable at hour 6 if the feature uses the result timestamp; it is available if the feature uses collection time and the value was known to the clinician. Decide which timestamp the deployment system will have.
See the executable audit in section 7.

**Step 6. Trace the label to its source and measure its error.**
Input: the label definition, the coding system or abstraction protocol, and a sample of records.
Output: a provenance statement, an estimated positive predictive value or agreement statistic, and a list of known false-positive and false-negative patterns.
Rationale: the label is a measurement. If it is built from billing codes, it measures billing; if it is built from reports, it inherits reporting style; if it is built by a language model, it inherits that model's errors.
Branch: if the label's positive predictive value is low relative to the decision's cost of a missed case, Gate 1 fails or the outcome must be redefined.
Edge case: the label is unavailable for part of the cohort because the patient transferred out or died before the diagnosis could be recorded. That is informative missingness, not random missingness.

**Step 7. Build the population coverage table.**
Input: the cohort, the site and year fields, and the demographic fields.
Output: counts and proportions by site, year, age band, sex, race or ethnicity, payer, and setting, with the outcome count in each cell.
Rationale: Gate 2 is about who is present in sufficient numbers to support the claim, including the subgroup analyses the reviewer will ask for.
Branch: if a deployment-relevant subgroup has fewer than about 10 outcome events per predictor, any subgroup metric is uninformative. Mark it as underpowered rather than reporting it.
Edge case: race and ethnicity are coded differently across sites and eras. A harmonized table can hide a category that changed meaning; report the source coding alongside the harmonized value.

**Step 8. Compare era and practice.**
Input: the years covered, the treatment guidelines in force, and the measurement technology.
Output: a statement of the era gap and its likely direction of effect.
Rationale: a model trained on 2008 to 2012 data may learn sepsis definitions, antibiotic protocols, and lab ordering patterns that no longer hold.
Branch: if the era gap spans a definition change, such as Sepsis-2 to Sepsis-3, the label is not comparable across the boundary. Restrict the cohort or model the change explicitly.
Edge case: COVID-19 disrupted care patterns and coding in 2020 and 2021. A model trained across that period may learn pandemic-specific practice.

**Step 9. Score Gates 1 and 2 with evidence and record vetoes.**
Input: the outputs of steps 5 to 8.
Output: a score of 0, 1, or 2 per gate with one evidence line and one residual-uncertainty line.
Rationale: the score forces a verdict, and the evidence line makes the verdict reviewable.
Branch: a zero at either gate ends the candidate for this claim. Do not proceed to access until the zero is resolved by a design change or a substitute.
Edge case: a candidate can be retained for a different claim. Record the claim for which it is admissible rather than discarding the candidate entirely.

### Phase C — Gates 3 and 4

**Step 10. Identify the access route and its latency.**
Input: the custodian's access page, the current application requirements, and the institutional roles needed.
Output: a route with named steps, responsible people, expected calendar time, and a critical-path estimate.
Rationale: access latency is a schedule risk and a cost. A six-month credentialing process can exceed the value of a marginal accuracy gain.
Branch: if the latency exceeds the decision window, the candidate fails Gate 3 even if the science is perfect.
Edge case: the demo or public subset is not a substitute for the credentialed data. It can validate the pipeline, not the claim.
Decision table for access outcomes is given below.

**Step 11. Verify the granularity the analysis requires.**
Input: the analysis plan and the access specification.
Output: a statement of whether row-level, aggregate, or query-only access is sufficient, and whether the required fields are exported.
Rationale: a query product can answer cohort-count questions and cannot train a row-level model. Granularity is part of access fit.
Branch: if the required fields are not exported, the dataset fails Gate 3 for this analysis regardless of tier.
Edge case: small-cell suppression can remove the exact subgroup cells that motivated the study.

**Step 12. Map the licence to the product plan, clause by clause.**
Input: the terms of use or DUA, the product plan, and the deployment jurisdictions.
Output: a matrix with rows for training, evaluation, internal demonstration, model release, publication, redistribution, retention, and jurisdiction, and columns for permitted, conditional, prohibited, and silent.
Rationale: licence fit is not a yes or no. Different clauses bind different product activities, and the blocking clause is often not the one the team read first.
Branch: a prohibited clause on training or deployment is a veto for a commercial product. A silent clause is a question to the owner, not a permission.
Edge case: the licence is silent on model weights. Treat that as unresolved, and plan for the possibility that weights are a derivative work.
See topic 02 for the clause vocabulary and the negotiation sequence.

### Phase D — Gate 5

**Step 13. Specify the task and the split before looking at results.**
Input: the decision statement, the unit of analysis, and the patient, site, and time keys.
Output: a written split specification with group keys, fractions, and a hash of the assigned IDs.
Rationale: a split chosen after seeing results is a tuning parameter. Pre-specification is what makes the test set a test.
Branch: if the dataset has no patient identifier, a patient-level split is impossible, and the benchmark gate is at most 1.
Edge case: the same patient can appear under different identifiers across sites or eras. Probabilistic linkage introduces its own error; document it.

**Step 14. Choose the metric from the decision's cost structure.**
Input: the error costs from step 1, the capacity constraint, and the calibration requirement.
Output: a primary metric, a secondary metric, and a calibration plan, each with a rationale.
Rationale: AUROC is a ranking metric that is insensitive to the threshold and to prevalence. A fixed-capacity screening program needs precision at capacity; a cost-sensitive decision needs net benefit.
Branch: if the decision has a fixed capacity, AUROC alone is insufficient and the primary metric should be precision or recall at k.
Edge case: the metric is required by a regulator or a journal. Report it, but do not let it replace the decision metric; report both and explain the difference.

**Step 15. Test for contamination and split leakage.**
Input: the dataset, the model's training cutoff, and the split keys.
Output: a contamination assessment with the method used and the residual risk.
Rationale: a benchmark result is only interpretable if the test items were unseen. Contamination inflates performance and is invisible in the metric.
Branch: if contamination is detected, the test set is not a test set. Replace it with a fresh or private holdout before reporting.
Edge case: near-duplicates, not exact duplicates, are the common case in text and imaging. Use embedding or perceptual-hash search, not string matching alone.

### Phase E — Decision and upkeep

**Step 16. Convene the veto review.**
Input: the scored ledger.
Output: for each vetoed candidate, one of four decisions: cure by design change, substitute dataset, change the claim, or stop.
Rationale: a veto is a decision point, not an ending. The review asks whether the claim can be narrowed without losing the product's value.
Branch: see the veto response table below.
Edge case: a cure that changes the claim must be re-scored from step 1, because population and benchmark fit change with the claim.

**Step 17. Select the admissible set and write the fallback plan.**
Input: the veto review and the admissible candidates.
Output: a primary dataset, a comparator, a fallback, and a trigger for switching.
Rationale: access can fail late. A fallback without a trigger is a hope.
Branch: if the admissible set is empty, the deliverable is a data-collection plan or a stop memo.
Edge case: two candidates may be combined, one for development and one for external validation, provided they are independent enough to count as external.

**Step 18. Register the decision and its kill criteria, then re-run on change.**
Input: the ledger, the access plan, and the benchmark specification.
Output: a dated decision record with the evidence, the owner, and the conditions that would reopen it.
Rationale: datasets change. A decision without a re-run trigger silently becomes false.
Branch: re-run all five gates when the dataset version changes, the licence changes, the access route changes, the deployment population changes, or a label audit finds a material error.
Edge case: a dataset withdrawal can happen with little notice. Keep the fallback warm and the derived artefacts documented.

### Access decision table

| Observation at step 10 | Gate 3 score | Action |
|---|---|---|
| Open download, required fields present, no identity check | 2 | Proceed; record the version and download date. |
| Registered account, click-through terms, required fields present | 2 | Proceed; calendar the terms acceptance and the account owner. |
| Credentialed, training and DUA, latency under the decision window | 1 | Proceed in parallel with Gate 4; start the application now. |
| Credentialed, latency longer than the decision window | 0 | Veto for this timeline; switch to fallback or reschedule. |
| Controlled, institutional sign-off, output review, unknown latency | 1 or 0 | Score 0 unless a signed path and a named signing official exist. |
| Licensed, fee and field-of-use limits, negotiation required | 1 | Proceed only with a budget line and a legal owner. |
| Contributor-only, no row-level export | 0 | Veto for row-level modeling; consider aggregate use only. |
| Query-only product with no export | 0 | Veto for training or reproduction; use for feasibility counts. |

### Licence decision table

| Clause outcome | Effect on research use | Effect on commercial product | Action |
|---|---|---|---|
| Permitted | No condition | No condition | Record the clause and proceed. |
| Permitted with conditions | Conditions apply | Conditions may block release | Map each condition to a product activity; assign an owner. |
| Prohibited | Cannot proceed | Veto | Substitute the dataset or drop the activity. |
| Silent | Ambiguous | Ambiguous | Ask the owner in writing; do not infer permission. |

### Veto response table

| Veto gate | Cure by design | Substitute | Change the claim | Stop |
|---|---|---|---|---|
| Decision fit | Move prediction time; drop leaking features; redefine outcome | Dataset with the outcome at the decision time | Predict a proxy outcome and label it as such | No measurement exists |
| Population fit | Stratify and report; restrict the claim to the covered setting | Multi-site or registry data | Narrow to the source population | Target population absent |
| Access fit | Use aggregate access for a feasibility claim | Faster-access dataset | Methods claim on public data | No access inside the window |
| License fit | Remove the prohibited activity from the plan | Commercially licensed data | Non-commercial research only | Product depends on the restricted use |
| Benchmark fit | Pre-specify a clean split; change the metric | Fresh or private test set | Report as internal validation | No clean evaluation possible |

### Edge cases that change the procedure

**Licence forbids commercial use.** The candidate can still support a methods paper or an internal research question, but not product development. Re-score Gate 4 as 0 for the product claim, record the permitted claim separately, and start a commercial-access track in parallel. Do not let the research result create an expectation that the product can use the same data.

**Dataset withdrawn.** Custodians withdraw datasets for consent, governance, or security reasons. If the dataset is withdrawn after download, the licence usually requires deletion, and derived artefacts may also be affected. Keep the fallback warm, document the deletion, and re-run Gate 3 for the replacement.

**Credentialing takes months.** Re-plan the schedule around the critical path rather than around the modeling work. If the credentialing exceeds the decision window, the candidate fails Gate 3 for that decision even if it would pass later. A common error is to start modeling on a demo and assume the credentialed data will arrive in time.

**Version drift.** A new release can change cohort counts, variable definitions, and even the licence. Pin the version, record the release date, and re-run the gates when the version changes. Do not mix versions across development and evaluation.

**Label errors discovered.** If an audit finds that a material fraction of positives are false, the outcome ceiling drops and the benchmark gate weakens. Re-estimate the ceiling, consider a probabilistic label, and report a sensitivity analysis. Do not simply relabel the test set, which would invalidate the pre-specified evaluation.

**Benchmark contamination.** If test items appear in pretraining or in the model's training data, the test set is no longer a test. Replace it with a fresh or private holdout, document the probe used, and report the contaminated result only as a historical comparison.

### Evidence standards per gate

A score is only as good as the evidence behind it. The table below states the minimum evidence for each score. If the evidence is not in the ledger, the score is not reviewable.

| Gate | Score 2 evidence | Score 1 evidence | Score 0 evidence |
|---|---|---|---|
| Decision fit | Timestamped feature audit shows all inputs available before prediction time; outcome definition matches the decision; label positive predictive value estimated and acceptable | One or two boundary features; outcome is a proxy with a stated direction of error | Any core feature is recorded after prediction time; outcome is unavailable at the decision or is defined by a downstream action |
| Population fit | Coverage table matches the target on site type, era, setting, and deployment-relevant subgroups with adequate events | Coverage matches on setting but one subgroup is thin or one era is missing | Target setting absent; cohort is a selected population with no overlap |
| Access fit | Signed or immediate access at the required granularity, inside the decision window | Route identified, application not yet complete, latency estimated inside the window | No route, latency beyond the window, or granularity insufficient |
| License fit | Clause-by-clause review permits training, evaluation, release, and deployment in the target jurisdictions | Permits research and internal evaluation; product use needs negotiation | Prohibits a core activity; silent on model weights for a product plan |
| Benchmark fit | Pre-specified patient or site-level split, decision metric, calibration plan, contamination check | Split exists but is not pre-specified, or the metric is a proxy | No patient key, test set reused, or metric does not order the decision |

The evidence standard is deliberately asymmetric. A score of 2 requires positive evidence, not the absence of a known problem. "We have not found leakage" is not a temporal audit. "The custodian says the data are diverse" is not a coverage table.

### Scoring anchors

Use the same anchors across candidates. The anchors below are written for a clinical prediction product; adapt the wording, not the structure.

| Gate | 2 | 1 | 0 |
|---|---|---|---|
| Decision fit | Outcome, exposure, and confounders measured at the decision time in the same unit as the claim | Outcome measured with a proxy; one confounder missing | Outcome occurs after the decision or is defined by the action being guided |
| Population fit | Multi-site, target setting represented, era overlaps deployment, subgroups have adequate events | Single site of the right type, or multi-site with uneven coverage | Wrong setting, wrong era, or a selected cohort with no target overlap |
| Access fit | Immediate or short-latency access at row level | Credentialed or licensed access with a path and a schedule | No path, no export, or latency beyond the window |
| License fit | All planned activities permitted, including model release and commercial deployment | Research permitted; product use conditional | A core activity prohibited, or silence on a core activity |
| Benchmark fit | Clean pre-specified split, decision metric, calibration and subgroup plan | Split or metric has a known weakness that can be reported | No clean test set, or the metric cannot support the decision |

Two reviewers should score independently, then reconcile the evidence lines rather than the numbers. A disagreement about a number is usually a disagreement about the claim. Re-read the decision statement before reconciling.

### Critical path and parallelisation

Access and licence work is the long pole. Run it in parallel with the desk audits, not after them.

| Week | Desk work | Institutional work | Compute work |
|---:|---|---|---|
| 1 | Decision statement, long list, feature audit | Identify custodians and routes | None; do not train yet |
| 2 | Coverage table, label audit, Gate 5 draft | Start DUA or application; legal review | Pipeline on demo data only |
| 3-4 | Score Gates 1, 2, 5; veto review | Credentialing, training, signatures | Schema tests; no model selection |
| 5-8 | Refine benchmark specification | Access granted; data transfer | Development and internal validation |
| 9-12 | External validation plan | Output review, publication approval | External test if a second site exists |

The table is a planning heuristic, not a promise. Credentialing can take days or months depending on the custodian, the institution, and whether the application is complete on the first submission. A missing training certificate can restart the clock.

### Branch errors to avoid

- **Scoring Gate 3 before Gate 1.** A team spends a month on a DUA and then finds that the dataset has no outcome at the prediction time.
- **Scoring the licence against the research protocol.** The protocol permits research; the product does not. Score against the product plan.
- **Treating a demo as a candidate.** A 100-patient demo can test a pipeline, not a claim. It should not appear in the admissible set.
- **Averaging a zero.** A weighted total can make a vetoed dataset look competitive. Report totals only within the admissible set, or do not report them at all.
- **Scoring without a claim.** The same dataset can score 2 for a methods claim and 0 for a national incidence claim. Write the claim in the column header.
- **Forgetting re-entry.** A candidate that fails now may pass after a version change or a licence negotiation. Record the condition that would reopen it.

### Timestamp semantics for the Gate 1 audit

The audit is only as good as the timestamp chosen. The same lab value can be available at four different times, and the deployment system sees only the last one.

| Timestamp | Meaning | Available to a real-time alert | Common error |
|---|---|---|---|
| Order time | When the clinician requested the test | Yes, but the result is not known | Treating an order as a measurement |
| Collection time | When the specimen or reading was obtained | Usually, if the interface is fast | Assuming zero transport delay |
| Result time | When the laboratory verified and released the value | Yes, this is the safe timestamp | Using collection time and overstating availability |
| Chart time | When the value appears in the clinical chart | Sometimes later than result time | Assuming the chart is real time |
| Interface time | When the value reaches the alerting system | The true deployment timestamp | Ignored in offline audits |
| Extract time | When the research extract was built | Not a clinical timestamp | Used by mistake in feature windows |

For a research extract, the safe rule is to use result time for labs and chart time for flowsheet observations, then subtract a latency margin that matches the deployment interface. Document the margin. If the interface latency is unknown, run the audit at two margins and report both.

### Coverage workbook fields

A coverage table is a workbook, not a paragraph. The minimum fields are below; add fields until a reviewer can see who is missing.

| Field | Type | Purpose |
|---|---|---|
| site_id | Categorical | Supports leave-site-out and per-site reporting |
| year | Integer | Exposes era effects and definition changes |
| setting | Categorical | Emergency, ICU, ward, ambulatory, home |
| age_band | Ordered categorical | Paediatric and older-adult coverage |
| sex | Categorical | Subgroup reporting |
| race | Categorical with source coding | Equity analysis and coding-shift detection |
| ethnicity | Categorical with source coding | Separate from race; do not collapse silently |
| payer | Categorical | Access bias and transfer patterns |
| denominator | Integer | Stays, patients, or samples in the cell |
| outcome_events | Integer | Events in the cell, not prevalence alone |
| events_per_predictor | Float | Planning check for subgroup models |
| include_flag | Boolean | Whether the cell supports a reported metric |

The `include_flag` column is the discipline. A cell with 17 events and 28 predictors is present in the data and absent from the evidence. Marking it explicitly prevents a later analyst from computing a site metric because the rows exist.

### Veto review record

For each veto, record the four-field decision below. The record is reviewed by the same people who scored the gates, and it is dated.

| Field | Example entry |
|---|---|
| Candidate and claim | Epic Cosmos, row-level sepsis model for community hospitals |
| Veto gate and evidence | Access: contributor-only, no row-level export |
| Cure considered | Become a contributor; not feasible in 12 months |
| Substitute considered | eICU-CRD; admitted, with lower setting precision |
| Claim change considered | Aggregate feasibility counts only |
| Decision | Substitute, retain Cosmos for market sizing |
| Re-entry condition | Contributor agreement signed, or aggregate export added |
| Owner and date | Data lead, 14 November 2026 |

### Resolving scoring disagreements

Two reviewers rarely disagree about the evidence. They disagree about the claim. Use the sequence below before escalating.

1. Re-read the decision statement aloud. If either reviewer would change a word, the claim is the problem.
2. Compare evidence lines, not scores. "I found three leaking features" and "I found one" is a factual disagreement that can be settled by opening the table.
3. Ask what evidence would move the score by one point. If no evidence could, the reviewer is expressing a preference about the product, and that belongs in the decision record, not the gate score.
4. If the disagreement survives, score conservatively and record both positions. A documented 1 is better than an undocumented argument for 2.
5. Escalate only when the disagreement changes a veto. A one-point difference inside the admissible set rarely changes the plan; a one-point difference at zero changes everything.

## 6. Worked example

### The case

A 40-person clinical software company sells an early-warning module to US community hospitals of 100 to 400 beds. The module is meant to run inside the hospital's electronic health record (EHR) and page a rapid-response nurse when a ward patient is deteriorating from suspected infection.

The team has 12 months to a pilot and a commercial claim. The data lead is asked whether an available dataset can support development and evaluation.

**Step 1 output, decision statement.** "At hour 6 after emergency department arrival, for adults admitted to a ward with suspected infection, flag patients who will meet Sepsis-3 criteria within the next 6 hours so that a rapid-response nurse can review them before clinical deterioration; a false alert costs about 10 minutes of nurse time, and a missed case delays antibiotics and increases the risk of organ dysfunction."

**Step 2 output.** Index time: emergency department arrival. Feature window: 0 to 6 hours. Outcome window: 6 to 12 hours. Unit of analysis: hospital admission, with one alert per admission.

**Step 3 output, target population.** Adults aged 18 years or older, admitted through the emergency department to a ward, at a US community hospital, in the current era of Sepsis-3 practice. Excluded: direct ICU admissions, transfers from another hospital, obstetric admissions, and patients with comfort-care-only status.

### Candidate long list

| Candidate | Custodian | Version inspected | Unit | Access tier | Why it is on the list |
|---|---|---|---|---|---|
| MIMIC-IV | Beth Israel Deaconess and MIT | v3.1 | Patient, hospitalization, ICU stay | Credentialed | Deep physiology and labs; methods development |
| eICU-CRD | Philips and MIT | v2.0 | ICU stay | Credentialed | Multi-site US coverage; transportability |
| Epic Cosmos | Epic | 2026 vendor release | Query-level cohort | Contributor-only | Large hospital network; feasibility counts |
| National commercial claims | Licensed vendor | Current contract year | Claim | Licensed | National coverage and cost outcomes |
| Prospective partner site | Three community hospitals | Not yet collected | Admission | Contract | Exact target setting and prospective evaluation |

### Gate scores

| Gate | MIMIC-IV | eICU-CRD | Epic Cosmos | Claims | Prospective site |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 2 | 2 | 0 | 2 |
| 2 Population fit | 1 | 2 | 2 | 2 | 2 |
| 3 Access fit | 2 | 1 | 0 | 1 | 0 |
| 4 License fit | 2 | 2 | 1 | 1 | 2 |
| 5 Benchmark fit | 1 | 1 | 1 | 1 | 2 |
| Veto | none | none | access | decision | access |

**MIMIC-IV, Gate 1.** Sepsis-3 is computable from vitals, labs, antibiotics, and cultures. The feature audit at hour 6 found three candidate features that are recorded later: repeat creatinine, repeat lactate, and vasopressor start. They are excluded, leaving 25 features. The label is built from the Sepsis-3 clinical criteria rather than from discharge codes, and a 200-record audit estimates positive predictive value at 0.81 with a wide interval. Score 2, with the label error carried into the ceiling estimate.

**MIMIC-IV, Gate 2.** One tertiary academic medical center in Boston, intensive care selected, with a patient mix and practice pattern that does not match a 200-bed community hospital. The era overlaps the deployment period, and the physiology is rich, but the setting gap is material. Score 1. A national claim would score 0.

**MIMIC-IV, Gate 3.** The company already holds PhysioNet credentials from an earlier project. Completion, training, and the project-specific data use agreement took 19 days for the earlier project. Score 2 for this team, 1 for a team starting fresh.

**MIMIC-IV, Gate 4.** The data use agreement permits research use under the approved project. Commercial product development and model release require separate terms from the custodian. For a development and methods claim, permitted. Score 2 for research; the product track is unresolved.

**MIMIC-IV, Gate 5.** No frozen patient-level split is published with the release. The team must create and hash one. The metric for a fixed-capacity nurse review is precision at the top-k alerts per shift, with calibration reported alongside. The dataset can support this, but the split is ours, so external comparability is limited. Score 1.

**eICU-CRD, Gate 1.** Sepsis-3 is partially computable. Vitals and many labs are present, but coverage varies by hospital; some sites have sparse lactate or culture data. A site-level data dictionary audit is required before pooling. Score 2 for the computable subset, with a coverage restriction recorded.

**eICU-CRD, Gate 2.** About 200 US hospitals with a mix of community and academic sites, which is closer to the target than MIMIC-IV. Participation is voluntary, so the network is not a probability sample, and the hospitals that contribute are not a random draw. Score 2 for setting coverage, with the participation caveat written into the claim.

**eICU-CRD, Gate 3.** PhysioNet credentialed. The application requires a project description and a data use agreement; the earlier project took 19 days, but a new project can take longer if the description is incomplete. Score 1 because the timeline is not yet proven for this project.

**eICU-CRD, Gate 4.** Research use under the data use agreement. Commercial use requires separate terms. Same position as MIMIC-IV. Score 2 for research.

**eICU-CRD, Gate 5.** No frozen split. Site identifiers are present, which makes a leave-site-out evaluation possible, and that is the strongest benchmark available without a prospective site. Score 1, with a note that leave-site-out is the primary evaluation.

**Epic Cosmos, Gate 3.** Contributor-only. The company is not a contributing organization, and the product returns query-level results rather than row-level records. No path to row-level training data inside 12 months. Veto.

**Claims, Gate 1.** No vitals, no labs at 6-hour granularity, and the diagnosis is a billing concept recorded after discharge. The outcome cannot be measured at the decision time. Veto.

**Prospective site, Gate 3.** The three hospitals are interested but have no signed agreement, no institutional review board (IRB) approval, and no data-sharing contract. Twelve months is possible but not credible as a plan. Veto for the current timeline, retained as the external-validation path.

### The two diagnostics that decide the case

**Diagnostic 1: temporal audit.** The team listed 28 candidate features and their first availability times. Three were recorded after hour 6. In an illustrative re-run, a gradient-boosted model using all 28 features reached an area under the receiver operating characteristic curve (AUROC) of 0.86 on a patient-level holdout; the same model restricted to the 25 available features reached 0.78. The 0.08 difference is the value of information the deployment will not have. The audit is reproducible from the timestamp table and does not require model training to interpret. The full script is in section 7.

**Diagnostic 2: coverage and events per site.** The pooled eICU cohort has about 18,400 stays with a Sepsis-3 prevalence of 11 percent, which is 2,024 events and 72 events per predictor for a 28-predictor model. Adequate. But the median site has 780 stays and 9 percent prevalence, which is 70 events and 2.5 events per predictor. The smallest reporting site has 210 stays and 8 percent prevalence, which is 17 events and 0.6 events per predictor. A per-site metric at the median site is already underpowered; at the smallest site it is noise. The product claim is about community hospitals, and the sites that look most like community hospitals are the ones with the fewest events. Either the claim is pooled performance with site-level heterogeneity reported, or the claim is not supportable.

A third diagnostic, applied because the licence is unresolved, is the clause review. It does not reveal a data problem; it reveals a product problem, described under the variation below.

### Access route, time, and cost

| Item | Detail | Calendar time | Illustrative cost |
|---|---|---|---|
| PhysioNet account and CITI training | "Data or Specimens Only Research" module; individual accounts | 3 to 5 days | Staff time, 8 hours |
| Project application and data use agreement | Project description, custodian review, signature | 2 to 6 weeks | Staff and legal, 20 hours |
| Data transfer and storage | Credentialed download; encrypted storage; access controls | 1 week | Cloud storage, about $400 per month |
| Development compute | Model training and internal validation | 8 weeks | About $3,800 |
| Label audit | 200-record chart review by a clinical abstractor | 3 weeks, part-time | About $6,000 |
| Total to first result | Critical path is the agreement, not the modeling | 8 to 12 weeks | About $12,000 cash plus 0.5 FTE |

The cash figures are planning estimates for this case, not quotes. The important number is the calendar time: the agreement is on the critical path, and modeling cannot start on credentialed data until it is signed.

### Selection and substitute plan

The admissible set is MIMIC-IV and eICU-CRD. The plan uses eICU-CRD for development and leave-site-out evaluation because its setting coverage is closer to community hospitals, and MIMIC-IV for physiological depth and a second, independent development environment. The prospective site is retained as the external-validation path if the pilot proceeds.

Substitute plan if eICU access fails: use MIMIC-IV for development, test transport on HiRID or AmsterdamUMCdb (both credentialed, both non-US, so the transport test is harder to interpret), and convert the deliverable to a methods and transportability claim rather than a product claim. Trigger: if the eICU agreement is not executed by week 6, switch. The fallback is worse for the product claim because it removes the multi-site US evidence, and the switch must be recorded in the decision ledger.

### Vary one assumption: the licence forbids commercial use

Suppose the clause review finds that the data use agreement permits research use only and that the custodian will not negotiate commercial terms inside the timeline. Gate 4 becomes 0 for the product claim, and both admissible candidates are vetoed for that claim.

The plan changes in four ways:

1. The claim splits. A research claim ("multi-site development and leave-site-out evaluation of a sepsis deterioration model") proceeds on eICU-CRD and MIMIC-IV. The product claim is deferred.
2. A commercial data track starts in parallel. The options are a licensed multi-site EHR or claims network for evaluation, or a prospective site contract. The licensed route adds roughly $150,000 and 8 to 12 weeks in this planning scenario; the prospective route adds 6 to 9 months.
3. The benchmark weakens. Without row-level commercial data, the product evaluation cannot be reproduced by a customer or a regulator, so the evidence package must rely on the prospective pilot.
4. The decision record changes. The gate score for licence is 0 for the product claim and 2 for the research claim, recorded side by side. Reporting a single number would hide the split.

If instead the assumption that changes is size, the plan fails earlier: a 100-patient demo cannot support any claim, and the team must find a credentialed cohort before modeling. If the assumption is label quality, a positive predictive value of 0.62 instead of 0.81 lowers the outcome ceiling, forces a sensitivity analysis with a probabilistic label, and makes the precision-at-capacity target harder to defend. The structure of the decision does not change; the numbers and the claim wording do.

### Gate evidence log

The scores in the table above are only defensible if each has an evidence line. The excerpt below shows the format for the two admissible candidates.

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | MIMIC-IV | 28-feature timestamp audit; 200-record label audit | 2 | Label positive predictive value 0.81 with a wide interval |
| 1 | eICU-CRD | Variable coverage report by site; Sepsis-3 computability | 2 | Sparse lactate at some sites; pooled estimate may not apply |
| 2 | MIMIC-IV | Site description, case mix, era | 1 | Cannot test community-hospital transport |
| 2 | eICU-CRD | Hospital count, bed mix, participation rules | 2 | Voluntary participation, not a probability sample |
| 3 | MIMIC-IV | Prior credentialing record, 19 days | 2 | None for this team |
| 3 | eICU-CRD | Access page, application requirements | 1 | New project review may take longer |
| 4 | Both | Data use agreement text | 2 | Silent on model weights; product track unresolved |
| 5 | MIMIC-IV | Release contents; no frozen split | 1 | Split is ours; no external comparability |
| 5 | eICU-CRD | Site identifiers present; leave-site-out possible | 1 | Site-level event counts thin |

The log is the artefact that a reviewer reads. The score table is the summary; the log is the argument.

### Access plan by week

| Week | Action | Owner | Done when |
|---:|---|---|---|
| 1 | Confirm project scope and named users | Data lead | Project description drafted |
| 1 | Verify CITI training status | Each analyst | Certificates on file |
| 2 | Submit PhysioNet project application | Data lead | Application acknowledged |
| 2-4 | Respond to custodian questions | Data lead | No open queries |
| 4-6 | Execute data use agreement | Institutional signatory | Signed copy stored |
| 5 | Prepare secure storage and access controls | Engineering | Encryption and logging verified |
| 6 | Confirm fallback trigger | Product lead | Decision recorded |
| 6-7 | Transfer and checksum | Data lead | Row counts match the release |
| 7-8 | Rebuild the feature audit on the real extract | Analyst | Timestamp audit signed off |

### What the case demonstrates

- The highest-scoring candidate on depth was not the best candidate on population fit. The decision used two datasets for two jobs rather than pretending one was sufficient.
- The veto on the prospective site was a schedule veto, not a scientific one. It remained the external-validation path, which is why the project did not discard it.
- The licence question did not appear in the scientific scoring and still changed the product plan. That is the normal pattern: the blocking constraint is often commercial, not technical.
- The two diagnostics were cheap. A timestamp table and a count table decided more than a model comparison would have.
- The admissible set had two members, not one. A single selected dataset would have hidden the trade-off the reviewer will ask about.

### Counterfactual and value

The decision statement includes a counterfactual for a reason. If the current pathway already reviews every at-risk patient within the window, the model adds nothing, and no dataset can fix that. In this case the counterfactual is a nurse who reviews patients when the ward calls, often after vital-sign changes are noticed at routine observation intervals. The model's value depends on moving review earlier, not on ranking patients who would have been reviewed anyway.

That has a data consequence. The evaluation must compare the model-assisted pathway with the current pathway on the same patients, or estimate the difference under explicit assumptions. A stand-alone AUROC does not answer the counterfactual question. Gate 5 should therefore include a decision-curve or net-benefit analysis alongside discrimination, because the decision is whether to add a review, not whether to rank all admissions.

### Sensitivity of the decision

| Change | Gate affected | Does the selected dataset change? | Plan response |
|---|---|---|---|
| Prediction time moves from 6 to 12 hours | 1 | No, but the feature set grows | Re-run the timestamp audit; re-score Gate 5 |
| Target widens to ICU admissions | 2 | Yes, MIMIC-IV becomes a stronger fit | Add MIMIC-IV as a co-primary development set |
| Product becomes research-only | 4 | No | Licence veto lifts; product track deferred |
| Access to eICU slips past week 6 | 3 | Yes | Trigger the fallback; narrow the claim |
| Label audit finds positive predictive value below 0.65 | 1 and 5 | No, but the ceiling falls | Add a probabilistic label and a sensitivity analysis |
| A public leaderboard is added as a secondary benchmark | 5 | No | Assume contamination until tested; keep the private split primary |

### Conditions under which the case would fail

The case passes because two candidates survive and the claim can be split if the licence blocks the product. Remove any one of the following and the recommendation changes:

- **No second site.** If eICU were unavailable and only MIMIC-IV remained, the population-fit gap would be a 1 with no comparator, and the product claim would rest on a single tertiary center. The recommendation would narrow to a methods claim with a prospective validation requirement.
- **No existing credentials.** If the team were starting from zero, Gate 3 for both candidates would fall to 1 and the critical path would extend by four to six weeks. The pilot date would move, not the science.
- **A stricter label.** If the chart audit found a positive predictive value of 0.65, the outcome ceiling would fall and the precision target would need renegotiation with the clinical sponsor. The dataset would still pass Gate 1, with a weaker claim.
- **A larger population gap.** If the product were aimed at academic medical centers rather than community hospitals, MIMIC-IV would score 2 on population fit and become the primary development set.
- **A shorter timeline.** If the pilot were six months away, both credentialed candidates would fail Gate 3, and the only honest deliverable would be a feasibility study on public data.
- **A regulated claim.** If the output were intended to inform diagnosis rather than prioritise review, the evidence standard would rise to a prospective study, and no retrospective dataset would be sufficient.

The pattern is that the gates do not produce a single answer. They produce a defensible plan with named conditions, and the plan changes when a condition changes.

## 7. Code, query, or calculation

The three blocks below run on the Python standard library only. They implement the scorecard, the temporal audit, and the events-per-predictor check used in the worked example. Run them in order; each prints a deterministic table.

### 7.1 Scorecard with vetoes

```python
from dataclasses import dataclass

GATES = ("decision", "population", "access", "license", "benchmark")
LABELS = {
    "decision": "Decision",
    "population": "Population",
    "access": "Access",
    "license": "License",
    "benchmark": "Benchmark",
}

@dataclass(frozen=True)
class Candidate:
    name: str
    scores: dict
    evidence: dict

    def vetoes(self):
        return tuple(g for g in GATES if self.scores[g] == 0)

    def total(self):
        return sum(self.scores.values())

CANDIDATES = [
    Candidate(
        "MIMIC-IV v3.1",
        {"decision": 2, "population": 1, "access": 2, "license": 2, "benchmark": 1},
        {
            "decision": "Sepsis-3 computable from vitals, labs, antibiotics, cultures",
            "population": "Single tertiary academic ICU; no community-hospital test",
            "access": "PhysioNet credentialed; CITI; DUA",
            "license": "Research use under DUA; commercial use needs separate terms",
            "benchmark": "No frozen patient-level split published",
        },
    ),
    Candidate(
        "eICU-CRD v2.0",
        {"decision": 2, "population": 2, "access": 1, "license": 2, "benchmark": 1},
        {
            "decision": "Sepsis-3 partially computable; lab coverage varies by site",
            "population": "About 200 US hospitals; voluntary participation",
            "access": "PhysioNet credentialed; application review",
            "license": "Research use under DUA; commercial use needs separate terms",
            "benchmark": "No frozen split; site identifiers present",
        },
    ),
    Candidate(
        "Epic Cosmos",
        {"decision": 2, "population": 2, "access": 0, "license": 1, "benchmark": 1},
        {
            "decision": "Phenotypes vendor-mediated; raw labs not exported",
            "population": "About 2,358 hospitals (vendor figure)",
            "access": "Contributing organizations only; no row-level export",
            "license": "Contributor agreement; query outputs only",
            "benchmark": "No reproducible test set",
        },
    ),
    Candidate(
        "National claims",
        {"decision": 0, "population": 2, "access": 1, "license": 1, "benchmark": 1},
        {
            "decision": "No vitals or labs at 6-hour granularity",
            "population": "Commercially insured or Medicare coverage",
            "access": "Licensed contract",
            "license": "Field-of-use limits; no model release",
            "benchmark": "Billing labels only",
        },
    ),
    Candidate(
        "Prospective partner site",
        {"decision": 2, "population": 2, "access": 0, "license": 2, "benchmark": 2},
        {
            "decision": "Full EHR at bedside; prospective collection",
            "population": "Three community hospitals in target setting",
            "access": "No agreement inside 12-month window",
            "license": "Contract negotiable",
            "benchmark": "Prospective silent-mode evaluation possible",
        },
    ),
]

def report(rows):
    header = f"{'Candidate':<24}" + "".join(f"{LABELS[g]:>11}" for g in GATES) + f"{'Veto':>16}"
    print(header)
    print("-" * len(header))
    for c in rows:
        cells = "".join(f"{c.scores[g]:>11}" for g in GATES)
        veto = ",".join(c.vetoes()) if c.vetoes() else "none"
        print(f"{c.name:<24}{cells}{veto:>16}")

print("ALL CANDIDATES")
report(CANDIDATES)
print()
print("ADMISSIBLE SET")
for c in CANDIDATES:
    if not c.vetoes():
        print(f"  {c.name}: total {c.total()}/10")
print()
print("VETOED")
for c in CANDIDATES:
    if c.vetoes():
        first = c.vetoes()[0]
        print(f"  {c.name}: {first} -> {c.evidence[first]}")
```

Expected output:

```text
ALL CANDIDATES
Candidate                  Decision Population     Access    License  Benchmark            Veto
-----------------------------------------------------------------------------------------------
MIMIC-IV v3.1                     2          1          2          2          1            none
eICU-CRD v2.0                     2          2          1          2          1            none
Epic Cosmos                       2          2          0          1          1          access
National claims                   0          2          1          1          1        decision
Prospective partner site          2          2          0          2          2          access

ADMISSIBLE SET
  MIMIC-IV v3.1: total 8/10
  eICU-CRD v2.0: total 8/10

VETOED
  Epic Cosmos: access -> Contributing organizations only; no row-level export
  National claims: decision -> No vitals or labs at 6-hour granularity
  Prospective partner site: access -> No agreement inside 12-month window
```

The total is printed only inside the admissible set. Printing a total for a vetoed candidate invites the reader to compare it with admissible totals and reopen a settled veto.

### 7.2 Temporal audit

```python
FEATURES = [
    ("age", -87600.0, "registration"),
    ("triage_vitals", 0.1, "triage"),
    ("initial_lactate", 1.2, "lab"),
    ("wbc", 1.4, "lab"),
    ("culture_collected", 2.0, "micro"),
    ("antibiotic_first_dose", 2.3, "eMAR"),
    ("urine_output_6h", 6.0, "flowsheet"),
    ("creatinine_repeat", 7.5, "lab"),
    ("lactate_repeat", 8.2, "lab"),
    ("vasopressor_start", 9.4, "eMAR"),
]
PREDICTION_HOUR = 6.0
late = [f for f in FEATURES if f[1] > PREDICTION_HOUR]
print(f"Prediction time: hour {PREDICTION_HOUR:.0f}")
print(f"Features audited: {len(FEATURES)}")
print(f"Available at or before prediction time: {len(FEATURES) - len(late)}")
print(f"Late features that would leak: {len(late)}")
for name, t, src in late:
    print(f"  LEAK {name:<24} first at {t:5.1f} h ({src})")
print()
print("Boundary case: urine_output_6h is timestamped exactly at the prediction")
print("time. Include only if the value is complete before the alert fires.")
```

Expected output:

```text
Prediction time: hour 6
Features audited: 10
Available at or before prediction time: 7
Late features that would leak: 3
  LEAK creatinine_repeat        first at   7.5 h (lab)
  LEAK lactate_repeat           first at   8.2 h (lab)
  LEAK vasopressor_start        first at   9.4 h (eMAR)

Boundary case: urine_output_6h is timestamped exactly at the prediction
time. Include only if the value is complete before the alert fires.
```

The audit takes a feature table and a prediction hour. In a real project the timestamp column comes from the source system, and the hard part is deciding which timestamp the deployment will have: order time, collection time, result time, or chart time. Document the choice.

### 7.3 Events per predictor by site

```python
def epv(events, predictors):
    return events / predictors

scenarios = [
    ("All sites pooled", 18400, 0.11, 28),
    ("Largest site", 3100, 0.14, 28),
    ("Median site", 780, 0.09, 28),
    ("Smallest reporting site", 210, 0.08, 28),
]
print(f"{'Stratum':<26}{'Stays':>8}{'Prev':>7}{'Events':>8}{'EPV':>7}  Verdict")
for name, stays, prev, p in scenarios:
    events = round(stays * prev)
    value = epv(events, p)
    if value >= 10:
        verdict = "adequate"
    elif value >= 5:
        verdict = "marginal"
    else:
        verdict = "inadequate"
    print(f"{name:<26}{stays:>8}{prev:>7.0%}{events:>8}{value:>7.1f}  {verdict}")
```

Expected output:

```text
Stratum                      Stays   Prev  Events    EPV  Verdict
All sites pooled             18400    11%    2024   72.3  adequate
Largest site                  3100    14%     434   15.5  adequate
Median site                    780     9%      70    2.5  inadequate
Smallest reporting site        210     8%      17    0.6  inadequate
```

The ten-events-per-predictor rule is a planning heuristic for logistic regression, not a law. It is used here as a screening check: if a site is below it by a wide margin, a per-site metric will be uninformative regardless of the model. For a nonlinear model the effective number of parameters is larger, so the check is optimistic.

## 8. Data, standards, and regulatory context

The five gates are a local decision procedure. They sit inside a wider set of reporting standards, risk-of-bias tools, and regulations that determine what evidence a reviewer will accept. The table separates binding requirements from voluntary conventions, because teams often treat a checklist as a law and a regulation as a suggestion.

| Instrument | Issuer | Status | What it constrains | What it does not do |
|---|---|---|---|---|
| TRIPOD+AI | EQUATOR network and international group | Reporting guideline, voluntary | What a prediction-model report must state: data sources, participants, predictors, outcome, sample size, missing data, model, evaluation, fairness, open science | Does not judge whether the model is good; does not cover generative models or large language models explicitly |
| PROBAST+AI | International group, published 2025 | Appraisal tool, voluntary | Risk of bias and applicability across participants, predictors, outcome, and analysis, for development and evaluation | Does not produce a score that can substitute for judgement |
| Datasheets for Datasets | Academic, widely adopted | Documentation convention | Motivation, composition, collection, preprocessing, uses, distribution, maintenance | Does not verify the claims; a datasheet can be inaccurate |
| Model Cards | Academic, widely adopted | Documentation convention | Intended use, factors, metrics, evaluation data, training data, ethical considerations | Does not establish external validity |
| FDA AI-enabled device software guidance | US Food and Drug Administration | Draft, January 2025, not for implementation | Content of marketing submissions for devices with AI-enabled software functions, including data management, reference standard, representativeness, validation, monitoring | Does not apply to research tools or to products that make no medical claim |
| Predetermined change control plan guidance | FDA | Final, August 2025 | How a developer pre-specifies anticipated model modifications and their verification | Does not authorise unanticipated changes |
| HIPAA Privacy and Security Rules | US Department of Health and Human Services | Binding, 45 CFR Parts 160 and 164 | Acquisition, de-identification, minimum necessary use, business associate agreements, safeguards | Does not govern research use by non-covered entities in all cases; does not replace IRB review |
| Common Rule | US federal agencies | Binding for federally funded human-subjects research | Consent, IRB review, secondary use, identifiable private information | Does not cover all commercial research; institutional policies vary |
| GDPR | European Union | Binding, Regulation (EU) 2016/679 | Lawful basis, minimisation, transfer, data-subject rights, records, impact assessment | Does not by itself permit or forbid a model; national implementations differ |
| EU AI Act and MDR/IVDR | European Union | Binding, phased | Risk classification, technical documentation, post-market monitoring for high-risk AI and medical devices | Timelines and harmonised standards are still settling; legal review is required |
| ISO 14971 | International Organization for Standardization | Voluntary standard, often contractual | Risk management process for medical devices | Does not prescribe model-specific methods |
| IEC 62304 | International Electrotechnical Commission | Voluntary standard, often contractual | Medical device software lifecycle processes | Does not cover the statistical validity of a model |
| HTI-1 final rule | US Office of the National Coordinator | Binding for certified health information technology | Source attributes, intended use, validation, and fairness information for predictive decision support | Does not certify the model's clinical performance |

Three distinctions matter in practice.

**Binding versus voluntary.** HIPAA, GDPR, the Common Rule, the EU AI Act, and HTI-1 create legal obligations. TRIPOD+AI, PROBAST+AI, Datasheets, and Model Cards are conventions that reviewers, journals, and procurement teams expect. A voluntary standard becomes effectively binding when a contract, a journal, or a regulator incorporates it by reference. Read the contract, not only the standard.

**Documentation versus evidence.** A datasheet, a data card, and a model card are claims by an author. They are useful for orientation and dangerous as proof. The gate audit inspects the data and the evaluation, not only the documentation.

**Product regulation versus research ethics.** An IRB approval is not a licence to commercialise. A commercial licence is not evidence that the cohort consented to a commercial product. A regulatory clearance is not evidence that the model is fair or well calibrated. The instruments overlap but none substitutes for another.

### Data standards that affect gate evidence

| Standard | Domain | What it gives the gate audit | What it does not give |
|---|---|---|---|
| FHIR | Health data exchange | Resource structure for observations, conditions, medications, and encounters | A guarantee that the source data are complete or correct |
| OMOP CDM | Observational data harmonisation | Common tables and vocabularies for multi-site analysis | Preservation of every source field; mapping loss is real |
| LOINC | Laboratory and observation codes | Comparable test identities across sites | Comparable units, reference ranges, or assay methods |
| SNOMED CT | Clinical terminology | Granular clinical concepts and relationships | Consistent use by clinicians or coders |
| ICD-10-CM and ICD-10-PCS | Billing and administrative coding | Cohort definitions from claims and discharge data | Clinical timing, severity, or confirmation |
| DICOM | Medical imaging | Image metadata, acquisition parameters, and pixel data | Label truth or patient-level split keys |
| HL7 v2 | Legacy interface messaging | Timestamps and order/result structure | Semantic consistency across vendors |

A gate audit that ignores the standard behind a field will misread it. A LOINC code tells you which test was ordered, not whether the result is comparable across two laboratories with different assays. An ICD code tells you what was billed, not when the condition began. A DICOM series tells you the scanner and acquisition, which is exactly what a site-level split needs, but it does not tell you whether the radiologist's label is correct.

### How the gates map to a regulatory submission

- Gate 1 evidence becomes the intended-use and reference-standard section: what the device measures, when, and against what truth.
- Gate 2 evidence becomes the representativeness and subgroup analysis section: who was in the development and validation data, and who was not.
- Gate 3 evidence becomes the data provenance and integrity section: where the data came from, under what agreement, and how they were protected.
- Gate 4 evidence becomes the contract and data-rights section: what the sponsor is permitted to do with the data and the resulting model.
- Gate 5 evidence becomes the validation section: prespecified endpoints, split strategy, statistical analysis plan, calibration, and subgroup performance.

A submission that reports a metric without these five evidence lines is incomplete even when the metric is excellent.

### Where standards mislead a gate audit

- **Same LOINC code, different assay.** Two laboratories can report the same test code with different platforms, units, and reference ranges. The code gives identity, not comparability.
- **ICD code timing.** A discharge diagnosis is assigned at the end of the encounter. Using it as a feature for an early prediction leaks the future; using it as an outcome measures coding, not onset.
- **OMOP mapping loss.** A source field with no standard concept may be dropped or mapped to a broad concept. The harmonised table looks complete and is not.
- **FHIR resource completeness.** A resource can exist with missing or null elements. The presence of a resource type is not evidence that the clinical fact was recorded.
- **DICOM de-identification.** Burned-in text and private tags can retain identifiers. Metadata that looks clean can still carry a site or device fingerprint.
- **SNOMED granularity.** A concept may be more specific than the source data support, or so general that it merges distinct conditions. The terminology does not repair the clinical record.

The audit rule is to inspect values, not only codes. A table of concept identifiers tells you what the site intended to record; a sample of raw values tells you what it recorded.

## 9. Industry practice and case evidence

### What teams actually do

- **Gate reviews before modeling.** Mature teams hold a short, documented review with data, clinical, legal, and evaluation representatives. The output is a decision record, not a slide.
- **Two datasets by default.** One for development and one for external or transport testing. Teams that rely on a single dataset usually discover the need for a second one after the first external question.
- **Licence review at the start.** Legal is involved before the first model is trained, because a licence veto after training is a sunk cost.
- **Version pinning.** Dataset version, release date, and download hash are recorded with the split and the model. Without them, a result cannot be reproduced after a release change.
- **Data cards before model cards.** The data card is written while the data are being audited, not reconstructed at the end.
- **Kill criteria.** The team agrees in advance what result would stop the project, so the stop decision is not a political one.

These are practices, not universal norms. Many organisations still select data by availability and discover the constraints later. The gap between the two is a common source of failed diligence.

### Case evidence

| Case | Documented fact | Inference for this topic |
|---|---|---|
| GRAIL NHS-Galleri | A large screening study reported its primary endpoint in 2026; the advanced-stage incidence rate ratio was close to 1, with stage shift rather than a reduction in advanced-stage incidence, as summarised in the module survey. | A test can improve a proxy, such as early-stage detection, without improving the endpoint that decides adoption. Gate 5 must name the endpoint, not the proxy. |
| Apple hypertension notification | A consumer feature received FDA clearance in September 2025 with high specificity and lower sensitivity, as reported in the survey and the FDA clearance database. | A high-specificity, low-sensitivity notification can be a legitimate product for a defined intended use and a poor screening test for another. Match the claim to the intended use. |
| RSNA 2026 knee MRI challenge | The competition closed in October 2026 with thousands of teams and disputes over labels generated at scale and possible leakage, according to the survey; the organiser report should be verified before citing outcomes. | When labels are generated at scale, the label pipeline is part of the benchmark. A leaderboard without a published label pipeline is not interpretable. |
| MIMIC-IV in national claims | The dataset is a single-center resource, as documented by its custodians and the peer-reviewed data descriptor. | National incidence or equity claims are outside its scope. The failure is a population-fit veto, not a modeling error. |

The first three rows are recent and changing. Treat the survey summary as a pointer and verify the primary report before using any number in a submission. The fourth row is stable: the custodians describe the data as a single-center resource, and the descriptor is peer reviewed.

### The economics of access latency

Access latency is a real cost, and it is routinely underestimated because it is invisible in a spreadsheet. A six-week credentialing delay on a 12-month programme is roughly 12 percent of the calendar, and it delays every downstream activity: feature engineering, label audit, model selection, and external validation. If the delay pushes the evaluation past a customer's procurement window, the cost is not six weeks; it is the deal.

The same logic applies to licence negotiation. A clause that requires six months of redlines is a schedule risk even if the fee is zero. Teams should price the latency alongside the licence fee and the compute, and should start the access track on day one rather than after the first model.

### What buyers ask, and what benchmarks do not answer

Procurement teams ask about workflow fit, integration effort, liability, support, validation in a setting like theirs, and total cost. A leaderboard rank answers none of these. A benchmark result is necessary evidence for a technical reviewer and insufficient evidence for a buyer. The five gates produce the technical evidence; the access and licence tracks produce the commercial evidence; neither replaces a prospective pilot.

### Documented fact versus inference

The case table mixes two kinds of statement, and the difference matters when the material is reused in a submission.

- **Documented fact:** a custodian's release note, a peer-reviewed descriptor, a regulator's clearance database entry, or an organisers' report. These can be cited with a date and a version.
- **Inference:** a lesson drawn from the fact for a different context. The inference is useful for teaching and must be labelled when it appears in a decision memo.
- **Unverified summary:** a secondary summary of a recent event, including the module survey's account of a 2026 result. Treat it as a pointer to the primary report, not as the citation.

A practical rule: every row in a case table gets a source column and a status column with one of the three labels. If the status is unverified, the row cannot support a claim in a regulatory or commercial document until the primary source is read.

### How practice differs by organisation size

| Organisation | Typical behaviour | Consequence for the gates |
|---|---|---|
| Academic medical centre | Deep phenotyping, single-site data, IRB-led governance | Strong Gate 1, weak Gate 2; access is slow but well documented |
| Health system | Owns longitudinal EHR and claims; can link | Strong Gates 1 and 2 inside its catchment; external validity unknown |
| Payer | Claims plus limited clinical data | Strong population coverage, weak clinical granularity |
| Medical device company | Purchases or partners for data; regulatory evidence culture | Strong Gates 4 and 5; access cost is priced into the programme |
| Early-stage software company | Uses public or academic data to move fast | Weak Gates 3 and 4; licence and latency discovered late |
| Contract research organisation | Runs prospective studies for sponsors | Strong Gate 5 by design; cost and time are the constraints |

No row is a criticism. Each is a starting position, and the gate review exists to make the starting position explicit before it becomes a plan.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Dataset-first selection | The project starts with "we have access to X" | The claim is reverse-engineered from the data | The claim drifts until it matches whatever the data can show | Write the decision statement before the long list |
| Decision-time leakage | Offline performance is much higher than clinical intuition | Features are recorded after the prediction time | Prospective performance collapses | Timestamp audit at the claimed prediction time |
| Label laundering | The label is described only as "from the EHR" | A billing code, report, or model output is treated as truth | The metric measures the label pipeline | Trace provenance and estimate agreement |
| Population mismatch | The dataset is one site; the claim is national | Convenience replaces coverage | External validation fails | Coverage table and a narrowed claim |
| Access optimism | The plan assumes a DUA in two weeks | Download time is confused with credentialing time | Schedule slips; modeling starts on a demo | Route, latency, and owner recorded |
| Licence blindness | Nobody can quote the commercial-use clause | Legal review happens after training | The product cannot use the model | Clause matrix before training |
| Silent licence treated as permission | The terms do not mention model weights | Absence of a clause is read as freedom | Release or deployment is challenged | Written question to the owner |
| Benchmark reuse | The same test set has been used for months | Repeated tuning against the test | Reported gains are noise | Pre-specified split and a private holdout |
| Metric substitution | AUROC is reported for a fixed-capacity programme | Familiar metric replaces the decision metric | The model is ranked well and operated badly | Precision at capacity and net benefit |
| Single-site generalization | One hospital is used to claim transportability | Site and workflow shift are ignored | The first external site underperforms | Leave-site-out or a second site |
| Synthetic validation | Synthetic data are used to claim clinical performance | Synthetic distributions are convenient | The claim is unsupported | Synthetic data only for pipelines |
| Credential sprawl | Analysts share accounts or copy data locally | Access controls are bypassed for speed | DUA breach and possible termination | Named accounts and governed compute |
| Version drift | Results cannot be reproduced after an update | The release changed without a record | The evidence base is unstable | Pin version and hash; re-run gates on change |
| Empty-set denial | The team keeps searching until a dataset passes | The claim is lowered silently | An unsupportable product proceeds | Treat an empty set as a valid result |

Red flags that warrant a pause, not a note:

- The word "sepsis" or "readmission" appears in the claim without a definition and a timestamp.
- The dataset description gives a patient count but not an encounter or event count.
- The access section says "we will apply" and names no applicant, institution, or date.
- The licence section quotes a website FAQ rather than the agreement.
- The benchmark section reports a single number without a split description.
- The evaluation section reports no subgroup and no calibration.
- The data card was written after the model card.
- The team cannot state what result would stop the project.

### Pre-modeling checklist

Use this list at the end of the gate review. Each item points to a file, not to a memory.

| Gate | Check | Evidence file |
|---|---|---|
| 1 | Decision statement is versioned and dated | `decision_statement.md` |
| 1 | Prediction time and outcome window have units | `decision_statement.md` |
| 1 | Every feature has a source and timestamp | `feature_timestamp_audit.csv` |
| 1 | Leaking features are removed or the claim moves | `feature_timestamp_audit.csv` |
| 1 | Label provenance is documented | `label_provenance.md` |
| 1 | Label error is estimated on a sample | `label_audit.csv` |
| 2 | Coverage table includes denominators and events | `coverage.csv` |
| 2 | Underpowered subgroups are flagged, not hidden | `coverage.csv` |
| 2 | Era and definition changes are dated | `era_notes.md` |
| 3 | Access tier and route are named | `access_plan.md` |
| 3 | Applicant, institution, and owner are named | `access_plan.md` |
| 3 | Latency estimate has a source and a date | `access_plan.md` |
| 3 | Fallback trigger and date are recorded | `access_plan.md` |
| 4 | Clause matrix covers all planned activities | `licence_matrix.md` |
| 4 | Silence is recorded as a question, not a permission | `licence_matrix.md` |
| 4 | Jurisdictions and processing locations are checked | `licence_matrix.md` |
| 5 | Split specification is written before results | `split_spec.md` |
| 5 | Split hash is stored with the IDs | `split_hash.txt` |
| 5 | Metric follows from the decision costs | `metric_memo.md` |
| 5 | Contamination probes are named and run | `contamination_probes.md` |
| All | Veto review record is complete | `veto_review.md` |
| All | Decision owner and date are recorded | `decision_record.md` |

A checklist item with no file is an assertion. The review should reject assertions.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The decision statement is stable for the project | Gates are scored against it | The product scope changes mid-project | Re-read it at every gate review; version it |
| The claimed prediction time matches deployment | Feature availability depends on it | The workflow changes the alert timing | Shadow-mode timing study |
| The label measures the clinical event | The metric is interpreted through it | Coding changes, rule-out diagnoses, abstraction errors | Chart audit and agreement estimate |
| The source population overlaps the target | Population fit is a comparison | Referral patterns, payer mix, or setting differ | Coverage table against target criteria |
| Access latency is estimable | The schedule depends on it | Institution or custodian backlog | Ask the custodian for current median time |
| The licence text is the operative agreement | Gate 4 is scored from it | A master agreement or institutional rider overrides it | Legal review of the full contract set |
| The public split is clean | Comparability depends on it | Contamination or patient overlap | Contamination probes and ID overlap check |
| The metric orders decisions correctly | Gate 5 depends on it | Costs are asymmetric or capacity is fixed | Decision-curve analysis |
| One external site is enough for transport | Resources are finite | Site heterogeneity is large | Two external sites or leave-site-out |
| Documentation reflects the data | Orientation depends on it | Cards are stale or marketing-written | Inspect the data; do not rely on the card |
| The dataset will remain available | Continuity depends on it | Withdrawal, embargo, or licence change | Keep a fallback and a deletion plan |
| A small subgroup result is informative | Equity analysis depends on it | Too few events | Report events per subgroup and interval width |

The deepest limit is that none of this establishes clinical utility. The gates establish whether a dataset can support a claim about model performance. Whether the model improves outcomes is a separate question answered by a prospective study or trial, and no retrospective dataset settles it.

## 12. Exercises

**Exercise 1. Write a decision statement.**
Deliverable: one paragraph containing the actor, action, timing, population, and the cost of a false positive and a false negative, plus a counterfactual sentence.
Acceptance criteria: all five slots filled; costs stated in a unit (minutes, dollars, cases); the counterfactual describes current practice; no model name appears.
Time: 30 minutes.

**Exercise 2. Score three candidates.**
Deliverable: a five-gate table with a score and an evidence line per cell, a veto column, and a one-paragraph selection.
Acceptance criteria: at least three candidates; every score has evidence; every zero is named as a veto; totals are reported only inside the admissible set.
Time: 90 minutes.

**Exercise 3. Run a temporal audit.**
Deliverable: a feature table with source, timestamp, first-availability hour, and classification; a list of leaking features; a revised feature set.
Acceptance criteria: at least 15 features; the prediction time is stated; boundary cases are discussed; the consequence for the claim is stated.
Time: 60 minutes.

**Exercise 4. Build a coverage table.**
Deliverable: counts and outcome events by site, year, age band, sex, race or ethnicity, and payer, with an include flag.
Acceptance criteria: denominators and event counts both present; at least one underpowered cell identified; the coding source for race and ethnicity is stated.
Time: 75 minutes.

**Exercise 5. Map a licence.**
Deliverable: a clause-by-clause matrix for training, evaluation, internal demonstration, model release, publication, redistribution, retention, and jurisdiction.
Acceptance criteria: each cell marked permitted, conditional, prohibited, or silent; quotations located by section; no legal conclusion stated as certain.
Time: 60 minutes.

**Exercise 6. Choose a metric.**
Deliverable: a one-page memo naming the primary metric, a secondary metric, and a calibration plan, with the decision costs that justify each.
Acceptance criteria: the metric follows from the cost structure; AUROC is not the primary metric for a fixed-capacity decision; the memo states what the metric does not capture.
Time: 45 minutes.

**Exercise 7. Defend a rejection.**
Deliverable: a 400-word rejection memo for one candidate, with the veto evidence and the conditions that would make it admissible.
Acceptance criteria: specific evidence; no vague quality language; at least one condition that could be tested; a named alternative.
Time: 45 minutes.

**Exercise 8. Design the empty-set response.**
Deliverable: a two-page plan for the case where no candidate passes, including the data-collection route, the narrowed claim, the interim deliverable, and the stop criteria.
Acceptance criteria: the plan does not lower the claim silently; it names the decision that would stop the project; it estimates time and cost.
Time: 90 minutes.

## 13. Self-check answers

**Item 1. What are the five gates, and why is a weighted average inappropriate?**
The gates are decision fit, population fit, access fit, licence fit, and benchmark fit. A weighted average is inappropriate because the gates are conjunctive: a zero on any gate means the dataset cannot support the claim as written, and no strength on another gate compensates. Averaging also hides which constraint is binding, so the team cannot design a fix.

**Item 2. Why does the order of the gates matter?**
The order follows cost and irreversibility. Decision and population fit are desk audits that can be completed in days. Access and licence fit involve institutions, signatures, and weeks of latency. Benchmark fit depends on the metric and split, which are only meaningful once the first four are settled. Running access first can consume the schedule before a fatal decision-fit problem is found.

**Item 3. What is the difference between a target population and a source population?**
The target population is who will be affected by the deployed decision. The source population is who is actually captured after inclusion and exclusion. Population fit is the comparison between them. A dataset can be large and still have a source population that does not overlap the target.

**Item 4. Classify a lab collected at hour 5 and resulted at hour 9 for a prediction at hour 6.**
It leaks if the feature uses result time, because the value is not available to the deployment system until hour 9. It is available only if the feature uses collection time and the clinician could act on a value that had not been verified, which is usually false for a lab. The safe classification is leaking. The audit should record both timestamps and state which one the deployment interface exposes.

**Item 5. Why is AUROC insufficient for a fixed-capacity screening programme?**
AUROC measures ranking across all thresholds and is insensitive to prevalence and to the capacity constraint. A programme that can review 20 patients per shift needs precision or recall at the top 20, and it needs calibration to set the threshold. Two models can have the same AUROC and very different performance at the operating point.

**Item 6. What does a score of 1 mean, and what does it require?**
A score of 1 means the candidate is usable with a stated limitation or a design change. It requires the limitation to be written down, the design change or reporting plan to be named, and the residual uncertainty to be recorded. A 1 without a mitigation is a 0 in disguise.

**Item 7. When is an empty admissible set the correct result?**
When no candidate can measure the outcome at the decision time, in the target population, with access inside the window, under a licence that permits the plan, and against a clean benchmark. The correct deliverable is then a data-collection plan, a partnership, a narrowed claim, or a stop decision, not a search for a dataset that passes a weakened claim.

**Item 8. What belongs in a veto review record?**
The candidate and claim, the veto gate and evidence, the cures considered, the substitutes considered, the claim changes considered, the decision, the re-entry condition, and the owner and date. The record makes the rejection auditable and prevents the same candidate from being re-proposed without new evidence.

**Item 9. Why does a score of 2 not mean the dataset has no limitations?**
A 2 means the evidence supports the claim without a material gap on that gate. It does not mean the dataset is unlimited. A single-site dataset can score 2 on decision fit and still fail population fit; a licensed dataset can score 2 on licence fit for research and 0 for a product. The score is relative to the claim and the gate, not a global quality rating.

**Item 10. What is the difference between external validation and transportability?**
External validation is an evaluation on data from a different site, time, or population. Transportability is the property of retaining performance when the setting changes. One external test is evidence of transportability in one direction, not proof of it. A model can pass external validation at one site and fail at the next because the shift is heterogeneous.

**Rubric for the design exercises (2, 5, 8).** Full marks require: a named claim, evidence per cell, an explicit treatment of uncertainty, a fallback or re-entry condition, and a document a reviewer could act on without asking a clarifying question. Partial marks for a complete structure with thin evidence. No marks for a score table without evidence lines.

## 14. Interview questions

**1. Walk me through how you would choose a dataset for a readmission model.**
A strong answer defines the decision and the prediction time first, then compares EHR, claims, and health-system data on post-discharge capture and label definition, then checks access and licence, then specifies a patient-level split and a metric tied to the intervention. A weak answer starts with a dataset name and works backwards.

**2. What can claims data never tell you?**
Severity at the bedside, laboratory values, vital signs, adherence, out-of-network care, and the reason a therapy stopped. A strong answer also notes that the diagnosis is a billing concept recorded after the encounter. A weak answer says claims are "less granular" without naming the consequence.

**3. MIMIC-IV or eICU for a sepsis early-warning model?**
A strong answer separates the methods claim from the transportability claim: MIMIC-IV for depth and phenotyping, eICU for multi-site coverage, with external validation either way. A weak answer picks the larger dataset without discussing setting.

**4. How would you detect benchmark contamination?**
Date-based holdout against the model cutoff, canary strings, paraphrase and option-order perturbations, nearest-neighbour search in public corpora, and a private holdout. A strong answer says what each probe can and cannot detect. A weak answer says "check if the data were in training" without a method.

**5. A vendor reports 0.95 AUROC on a public dataset. What do you ask?**
Split strategy, label source, external validation, calibration, subgroup results, test-set reuse, preprocessing fit scope, and whether the model saw the test set. A strong answer asks for the data card and the evaluation code. A weak answer asks only for the sample size.

**6. What does a non-commercial licence prevent in a product company?**
It may restrict training, evaluation, internal demonstration, model release, and derived weights, depending on the clause. A strong answer says each activity must be mapped separately and that silence is not permission. A weak answer says "we cannot sell the data," which misses the model.

**7. How do you choose between UK Biobank and All of Us?**
By estimand, phenotype depth, linkage, diversity, consent, latency, and return of results. A strong answer notes that neither is a probability sample and that the choice follows the claim. A weak answer chooses on size alone.

**8. Why is a random split dangerous in imaging?**
The same patient, study, scanner, and acquisition can appear on both sides, so the model can memorise anatomy or acquisition rather than pathology. A strong answer specifies patient-level and site-level grouping. A weak answer says "random splits are fine if the dataset is large."

**9. What is the difference between a data card and a datasheet?**
They overlap. A datasheet emphasises motivation, composition, collection, preprocessing, uses, distribution, and maintenance. A data card is a shorter internal summary often used for review. A strong answer notes that both are claims, not evidence.

**10. When is synthetic data acceptable?**
For pipeline development, schema testing, privacy-safe demos, and augmentation with validation. Never for clinical performance or fairness claims. A strong answer names the validation that would be required before any clinical claim.

**11. What would make you tell a team not to proceed?**
No dataset contains the outcome at the decision time; the licence forbids the product; access latency exceeds the window; the only evidence is a contaminated leaderboard; or the counterfactual shows no decision changes. A strong answer treats this as a valid deliverable. A weak answer treats stopping as a failure of effort.

**12. How would you report a benchmark result to a regulator?**
Versioned data, prespecified analysis, patient-level splits, confidence intervals, subgroup and calibration results, contamination checks, and reproducible artefacts. A strong answer separates internal from external validation. A weak answer reports a single metric and a confusion matrix.

**13. How do you price access latency?**
Convert it into calendar time on the critical path, then into the downstream activities it delays, then into the commercial event it threatens. A six-week delay is not six weeks of analyst time; it is a delayed pilot, a delayed procurement window, and a delayed revenue line. A strong answer names the fallback and the trigger date.

**14. What goes in a one-page gate memo?**
The claim, the candidates, the score table with evidence lines, the vetoes and their resolutions, the selected set with a fallback, the access route and latency, the blocking licence clause, the benchmark specification, and the decision owner with a date. A weak answer includes a long narrative and no decision.

## 15. Key terms

- **Admissible set** — candidates with no veto, ranked only within the set.
- **Benchmark** — a fixed task, dataset, and metric used to compare methods.
- **Benchmark fit** — the gate asking whether the task and metric correspond to the decision on a clean test set.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Contamination** — test items or near-duplicates present in training data.
- **Coverage table** — counts and outcome events by site, year, and subgroup.
- **Credentialed access** — identity verification, training, and a signed data use agreement.
- **Data use agreement (DUA)** — contract governing access, use, retention, and redistribution.
- **Datasheet** — documentation of motivation, composition, collection, preprocessing, uses, distribution, and maintenance.
- **Decision fit** — the gate asking whether the outcome, exposure, and confounders are measured at the decision time.
- **Decision statement** — actor, action, timing, population, and error costs in one sentence.
- **Estimand** — the precise quantity an analysis targets.
- **Evidence ledger** — claim, evidence, verdict, and residual uncertainty per gate.
- **External validation** — evaluation on data from a different site, time, or population.
- **Gate** — one of five conjunctive admission tests for a dataset.
- **Label provenance** — the origin and process that produced the target label.
- **Leakage** — information unavailable at prediction time, or overlap between splits.
- **Licence fit** — the gate asking whether the licence permits the planned activities.
- **Open access** — direct download without identity verification.
- **Population fit** — the gate asking whether the source population matches the deployment population.
- **Prediction time** — the moment at which all inputs must be available and the output is acted on.
- **Precision at capacity** — true positives among the top-k ranked when capacity is fixed.
- **PROBAST+AI** — risk-of-bias and applicability tool for prediction models.
- **Registered access** — account creation and click-through terms.
- **TRIPOD+AI** — reporting guideline for prediction models using regression or machine learning.
- **Veto** — a zero on any gate, disqualifying the candidate for the named claim.
- **Access fit** — the gate asking whether the data can be obtained at the required granularity before the deadline.
- **Decision curve analysis** — net benefit across threshold probabilities, incorporating the harms of false positives and false negatives.
- **Field of use** — the permitted purpose, therapeutic area, or population in a licence.
- **Model card** — documentation of a model's intended use, training data, evaluation, and limitations.
- **Risk of bias** — the likelihood that a study's results are distorted by its design or analysis.
- **Test set** — data held out from all model selection and tuning.
- **Temporal audit** — a feature-by-feature check of availability at the prediction time.
- **Transportability** — whether a model trained in one setting retains performance in another.
- **Unit of analysis** — whether a row is a patient, encounter, stay, sample, or cell.
- **Silent clause** — a licence term that does not address a planned activity; a question, not a permission.
- **Data provenance** — the origin, custody, and processing history of a dataset.
- **Admissible claim** — a claim that at least one candidate can support without a veto.

## 16. Sources

All URLs accessed 26 October 2026. Versions, licences, and access conditions change; verify at the primary source before use in a contract or submission.

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
2. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. *BMJ* 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
3. Gebru T, Morgenstern J, Vecchione B, et al. Datasheets for Datasets. *Communications of the ACM* 2021;64(12):86-92. Preprint: https://arxiv.org/abs/1803.09010
4. Mitchell M, Wu S, Zaldivar A, et al. Model Cards for Model Reporting. *Proceedings of FAT* 2019. https://arxiv.org/abs/1810.03993
5. US Food and Drug Administration. Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations. Draft guidance, January 2025. https://www.fda.gov/media/184856/download
6. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions. Final guidance, August 2025. https://www.fda.gov/media/166704/download
7. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data* 2023;10:1. https://doi.org/10.1038/s41597-022-01899-x
8. MIMIC-IV documentation and release notes. https://mimic.mit.edu/
9. PhysioNet. MIMIC-IV project page. https://physionet.org/content/mimiciv/
10. PhysioNet. eICU Collaborative Research Database. https://physionet.org/content/eicu-crd/
11. US Department of Health and Human Services. HIPAA Privacy Rule, 45 CFR Part 160 and Part 164, Subparts A and E. https://www.hhs.gov/hipaa/
12. European Parliament and Council. Regulation (EU) 2016/679 (General Data Protection Regulation). https://eur-lex.europa.eu/eli/reg/2016/679/oj
13. European Parliament and Council. Regulation (EU) 2024/1689 (Artificial Intelligence Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
14. International Organization for Standardization. ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
15. International Electrotechnical Commission. IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes. https://webstore.iec.ch/publication/22794
16. Office of the National Coordinator for Health Information Technology. HTI-1 final rule, 2024. https://www.healthit.gov/rule/hti-1
17. Module 21 survey file, sections 3, 15, and 21. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Next topic, access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- EHR, claims, and ICU data: [`03_ehr-claims-and-icu.md`](03_ehr-claims-and-icu.md)
- Imaging datasets: [`04_imaging-datasets.md`](04_imaging-datasets.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the dataset audit and license audit: [`../../labs/`](../../labs/)

