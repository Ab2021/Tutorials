# Brief Catalogue

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 05 of 11
> Estimated study time: 4 hours
> Prerequisites: Topics 01 and 04
> Version 2026.3

## 1. Why this topic exists

A capstone brief is a compressed statement of a decision, a dataset, a metric, and a constraint.
The programme publishes forty-four of them because teams otherwise choose a project by
enthusiasm, and enthusiasm is a poor predictor of feasibility.
The catalogue is not a menu of ideas. It is a set of pre-scoped problems, each with a known
evidence bar, a known data route, and a known characteristic failure, so that a team can compare
options on the dimensions that determine whether the project can be finished.

The failure this topic prevents is specific.
A team reads a brief, recognises the dataset, and starts work. Three weeks later it discovers
that the dataset requires credentialed access the institution does not hold, that the metric
cannot be computed at the capacity the owner has, or that the brief's difficulty rating assumed a
clinician who is not available.
All three facts were in the brief's constraints, and none were read.

There is a second failure, less visible: a team chooses a brief because it is prestigious and
then quietly narrows it until it resembles a different brief.
The dossier then describes a project that is neither the one proposed nor the one completed.
Panels detect this by comparing the charter with the results, and the comparison is unkind.

The catalogue has an industry analogue.
Companies maintain opportunity backlogs with a standard anatomy: the decision, the user, the data
required, the estimated value, the effort, the dependencies, and the reason the item might be
rejected. The anatomy matters more than the ideas, because it makes comparison possible.
A team that can read a brief and extract its constraints in ten minutes has learned a skill that
transfers directly to evaluating a product request or a research proposal.

A backlog also has a queue discipline that a catalogue lacks.
In a company, an item enters the backlog with an owner, is scored against others, and is either
funded or rejected with a reason. The rejection is recorded because the next person will ask the
same question.
A capstone team that keeps only its chosen brief loses that record, and the panel's question
"why not the other one?" becomes unanswerable.
The rejection note is therefore not paperwork; it is the artefact that demonstrates a decision
process rather than a preference.

A catalogue also solves a problem that a list of ideas cannot: it makes the constraints
comparable. Two briefs with the same difficulty rating may have completely different binding
constraints, one a licence and one an owner, and the comparison is only meaningful when both are
written in the same fields. The programme's briefs are written that way on purpose, and the first
skill this topic teaches is to read the fields rather than the title.

Finally, the catalogue is a teaching device for rejection. Most professional work consists of
deciding what not to build, and the reasoning is rarely written down. A team that practises the
reasoning on forty-four briefs in week 1 will find the same reasoning easier when a sponsor
proposes a new question in week 9.

This topic teaches that extraction, then applies it to the catalogue.
It assumes the archetype taxonomy from Topic 04 and the capacity arithmetic from Topics 02 and 03.
The output is a shortlist of three briefs, a written adaptation of the chosen one, and a record
of why the others were rejected.

## 2. Learning outcomes

By the end of this topic you can:

1. **Extract** the decision, data route, metric, regulatory angle, and difficulty from a brief in
   under ten minutes.
2. **Test** a brief against five constraints: owner availability, data licence, validation level,
   metric computability, and team capacity.
3. **Distinguish** a brief that is infeasible from one that is feasible only after adaptation, and
   write the adaptation explicitly.
4. **Compare** three shortlisted briefs on a common scoring frame and justify the ranking.
5. **Identify** the data-access tier of each catalogue entry and the obligation it creates.
6. **Adapt** a brief to a team's constraints without changing the decision it supports.
7. **Reject** a brief on evidence grounds and record the reason, rather than abandoning it
   silently.
8. **Defend** the chosen brief to a panel in two minutes, including what was rejected and why.

## 3. The core mental model

A brief has five fields and one hidden field.
The five are the decision, the data, the metric, the regulatory angle, and the difficulty.
The hidden field is the owner, which the brief implies but rarely names.
A team that reads only the five visible fields will choose a brief it cannot execute.

```text
   BRIEF
   +-------------------------------------------------------------+
   | decision  -> who acts, how often, with what capacity        |
   | data      -> source, tier, licence, schema, coverage        |
   | metric    -> computable at the operating point?              |
   | regulator -> device, CDS, research-only, programme          |
   | difficulty-> 1 (lab scale) to 5 (multi-site, high risk)     |
   | OWNER     -> implied; must be found, not assumed            |
   +-------------------------------------------------------------+
              |            |             |            |
              v            v             v            v
        feasible?     admissible?    measurable?   finishable?
              \            |             |            /
               \           |             |           /
                +----------+-------------+----------+
                           |
                    shortlist of three
                           |
                    adaptation memo
```

Read the diagram as a funnel with four independent gates.
Feasibility asks whether the data can be obtained and the team can build the artefact.
Admissibility asks whether the claim is one the evidence can support.
Measurability asks whether the metric can be computed at the operating point.
Finishability asks whether the work fits the calendar and the team's hours.

A brief must pass all four. Passing three is not a near miss; it is a different project.

The funnel is also a reading order for the catalogue.
Read the decision first, because a brief whose decision you cannot state is not a candidate.
Read the data second, because the access route determines whether the next six weeks are
possible.
Read the metric and the level third, together, because a metric without a level is a number
without a claim.
Read the difficulty last, because it is the only field that is an average rather than a
specific.

### The four gates in order

The gates are applied in a fixed order because each one is cheaper than the next.

Feasibility is checked first, because a brief with inaccessible data cannot be evaluated on any
other dimension. It takes one licence lookup and one access request.

Admissibility is checked second, because a claim with no owner cannot be defended even if the
data are perfect. It takes one meeting request.

Measurability is checked third, because a metric that cannot be computed at deployment prevalence
produces a number the decision cannot use. It takes a prevalence count and a capacity number.

Finishability is checked last, because it is the most expensive to estimate and the easiest to
revise. It takes the effort arithmetic from Topic 02.

A team that reverses the order, starting with finishability, will spend two days estimating hours
for a brief whose data it cannot obtain.

Three properties of the model matter.

**The fields interact.** A difficulty rating of 4 with open data may be more feasible than a
rating of 3 with credentialed data, because the binding constraint is access rather than
modelling. The rating is an average, and averages hide the constraint that binds.

**The owner is not in the brief.** Briefs describe decisions, not the people who make them. The
team's first task after shortlisting is to find the owner, and a brief without an owner is a
methods study regardless of how clinical its title sounds.

**Adaptation is legitimate; substitution is not.** Changing the population, the validation level,
or the metric while keeping the decision is adaptation. Changing the decision while keeping the
title is substitution, and it produces a dossier that cannot be defended.

A fourth property is about the catalogue's origin.
The forty-four briefs were written to span difficulty and domain, not to be equally feasible.
Several are deliberately at difficulty 5 because they represent multi-site, high-risk work that a
course team should attempt only with an external partner. Choosing one of those without the
partner is the most common catalogue error.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Brief | A pre-scoped problem statement with decision, data, metric, angle, and difficulty | A project idea | Catalogue number |
| Difficulty rating | 1 to 5 scale from lab-scale to multi-site high-risk | A prediction of grade | Integer |
| Data option | The named dataset or class of dataset a brief expects | A suggestion | Dataset name and tier |
| Data tier | Open, credentialed, restricted, or synthetic | "Public" for anything downloadable after login | Tier label |
| Constraint | A condition that must hold for the brief to be feasible | A risk | Named condition |
| Adaptation | A change to population, level, or metric that preserves the decision | Any change | Written memo |
| Substitution | A change to the decision while keeping the brief title | Undisclosed narrowing | Detected at G4 |
| Shortlist | Three briefs compared on a common frame | A long list of interests | Three entries |
| Owner test | Whether a named person will act on the output | A letter of support | Named role and hours |
| Licence check | Verification that the dataset may be used for the intended purpose | Assuming open means unrestricted | Licence identifier |
| Metric computability | Whether the metric can be calculated at the deployment prevalence and capacity | Reporting AUROC because it is easy | Metric and operating point |
| Minimum validation level | The lowest level that supports the brief's claim | The level the team can reach | L2, L3, or L4 |
| Operating point | Threshold plus sensitivity and specificity | The default 0.5 | Threshold and rates |
| Coverage | The proportion of the target population represented in the data | Dataset size | Percentage and groups |
| Label provenance | How the outcome was ascertained and by whom | The column name | Source and adjudication |
| Contamination risk | Whether test data could be in pretraining or tuning | A concern without a check | Screen result |
| External partner | An organisation that supplies data or a workflow | A data file | Named organisation and agreement |
| Adaptation memo | The document recording changes to a brief | A verbal agreement | One page |
| Pre-registration | Writing the analysis plan before results | Registering a trial | Plan and timestamp |
| Synthetic data | Data generated to resemble a real cohort, with known limits | Pretending it is real | Generator and calibration source |
| Public use file | A de-identified dataset released for general research | Any downloadable file | File version and terms |
| Data dictionary | The definition, type, and coding of each field | The column headers | Field, type, code |
| Cohort definition | The rules that select the analysis population | The SQL query alone | Inclusion, exclusion, time zero |
| Refresh cadence | How often a source updates | The extraction date | Hours or days |
| Sandbox | An isolated environment for testing with mock data | A production system | Environment name |
| Benchmark version | The dated release of a benchmark and its harness | The website | Version identifier |
| Rule set | Competition-specific constraints on data, code, and submissions | The dataset licence | Rule section |

The last three definitions matter more than they appear.
A sandbox is the only safe place to test an agentic brief such as Brief 21, because a write to a
real record is an incident, not a bug.
A benchmark version determines whether two reported numbers are comparable; a score from a 2025
harness and a score from a 2026 harness may not be.
A competition rule set is not a licence, and a team that satisfies one may still breach the other.

Two definitions do most of the work.
A constraint is not a risk. A risk may happen; a constraint either holds or the brief is
infeasible. Treating a constraint as a risk produces a plan that assumes the constraint away.
An adaptation is not a rejection. A team that changes the population to fit the data has not
failed; it has made a decision that must be written down and signed.

## 5. The mechanics, step by step

The procedure has two parts: reading the catalogue, and selecting from it.
Reading takes about ninety minutes for the full set; selection takes a further two hours with the
team and the clinical mentor.

### The catalogue, by domain

The tables condense the programme's forty-four briefs.
The metric column names the primary metric the brief expects, not the only metric a team may
report. The level column is the minimum validation level that supports the brief's claim.
The access column names the route and the obligation; every licence must be re-checked before
use, and the word "check" is deliberate.

#### Imaging

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 1 | Aneurysm triage on head CTA | RSNA 2025 aneurysm data; open CTA | Macro AUC; sensitivity at 2 FP per scan | L3 | Kaggle rules; competition data licence | 5 |
| 2 | Knee MRI multi-finding classifier | RSNA 2026 knee MRI | Macro ROC AUC across 12 findings | L3 | Kaggle rules; report text in about 12 languages | 5 |
| 3 | Lumbar spine severity grading | RSNA 2024 lumbar spine | Weighted multi-class log loss | L3 | Kaggle rules; imaging plus labels | 4 |
| 4 | Pneumothorax triage under site shift | CheXpert; MIMIC-CXR | AUROC plus subgroup gap | L2 | CheXpert licence; MIMIC-CXR credentialed | 3 |
| 5 | Tuberculosis screening on device | CODA-TB; HeAR embeddings | Sensitivity at fixed specificity; model size | L2 | Check dataset terms; on-device constraint | 4 |
| 6 | BI-RADS extraction from reports | Institutional reports; MIMIC-style corpus | Macro-F1; calibration | L2 | PHI risk; de-identification required | 2 |
| 7 | Federated glioma segmentation | BraTS; FeTS | Dice; HD95; communication cost | L3 | Synapse registration; multi-site terms | 5 |
| 8 | Diabetic retinopathy grading across cameras | APTOS; DeepDR | Quadratic weighted kappa | L2 | Open licence; camera metadata | 3 |

#### ECG, sleep, and physiological signals

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 9 | Chagas triage at 5% capacity | CODE-15%; SaMi-Trop; PTB-XL | Precision at top 5% | L2 | PhysioNet terms; some sets credentialed | 4 |
| 10 | Age-conditioned dementia risk from PSG | Human Sleep Project | Age-conditioned AUROC within 2 years | L3 | PhysioNet 2026 terms; hidden sites | 5 |
| 11 | Atrial fibrillation from a wearable | PhysioNet AF/PPG-style | F1; false alarms per hour | L2 | Check device and consent terms | 3 |
| 12 | Sepsis early warning | MIMIC-IV; eICU | AUROC; utility; lead time | L2 | Credentialed; CITI and DUA | 4 |
| 13 | EEG seizure classification | TUH EEG | Weighted kappa | L2 | Registration; montage heterogeneity | 4 |
| 14 | Sleep staging from a wearable | Sleep-EDF | Cohen's kappa | L2 | Open licence; label noise | 3 |
| 15 | ICU deterioration with fairness audit | MIMIC-IV | AUROC; equalized-odds gap | L2 | Credentialed; subgroup fields | 3 |

#### EHR, claims, and operations

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 16 | Emergency department boarding | MIMIC-IV-ED | MAE; calibration | L2 | Credentialed; ED schema | 2 |
| 17 | Readmission with social determinants | MIMIC-IV plus area deprivation index | AUROC; net benefit | L2 | Credentialed plus public SDOH | 3 |
| 18 | Post-transplant survival with censoring | CIBMTR-style registry | Stratified concordance | L3 | Restricted; registry governance | 4 |
| 19 | Operating-room scheduling | Public OR logs | Utilisation; overtime hours | L2 | Check institutional terms | 3 |
| 20 | Claims cost anomaly detection | CMS SynPUF | Precision at k | L2 | Public use file terms | 2 |
| 21 | FHIR medication reconciliation agent | MedAgentBench-style FHIR environment | Pass@1 | L3 | Open sandbox; no real PHI | 4 |

#### Clinical NLP, LLMs, and agents

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 22 | Discharge summary drafting | MIMIC-IV-Note | Factuality; omission rate | L2 | Credentialed; PHI handling | 3 |
| 23 | Prior-authorization letter drafting | Synthetic policies and notes | Rubric score; human preference | L1 | Synthetic only | 2 |
| 24 | Trial eligibility matching | TrialGPT-style; ClinicalTrials.gov | nDCG; precision | L2 | Public trials; note source | 3 |
| 25 | HealthBench-style evaluator | Public evaluation subset | Correlation with physician grades | L2 | Check benchmark licence | 3 |
| 26 | Multilingual triage | Ilera-Bench-style | Accuracy; safety recall | L2 | Check benchmark terms; language equity | 3 |
| 27 | Radiology report to structured findings | MIMIC-CXR | F1 per finding | L2 | Credentialed; PHI in prompts prohibited | 2 |

#### Genomics, pathology, and public health

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 28 | Missense variant effect prediction | CAGI7 deep mutational scanning | Spearman; ROC | L3 | CAGI terms; held-out truth | 4 |
| 29 | Polygenic risk score portability | CAGI7 PRS challenges | AUC across ancestries | L3 | CAGI/Synapse terms; European-ancestry limitation | 4 |
| 30 | Pathology foundation-model probe | SLC-PFM; TCGA | Balanced accuracy | L3 | Check model and data licences | 5 |
| 31 | Single-cell label projection | Open Problems in Single-Cell Analysis | Macro-F1 versus logistic baseline | L1 | Open; benchmark harness | 3 |
| 32 | Suicide-narrative abstraction | NVDRS; synthetic narratives | F1 | L2 | Restricted; mandated-reporter ethics | 3 |
| 33 | Wastewater and syndromic nowcasting | CDC NWSS plus ED visits | MAE; weighted interval score | L2 | Public; small-area privacy | 3 |
| 34 | FHIR quality-measure gap | Synthea; US Core; DEQM | Compliance rate | L2 | Synthetic; specification terms | 2 |

#### Hardware, sensors, and robotics

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 35 | ECG front-end hardware-in-the-loop | Synthetic ECG plus noise; public ECG | Sensitivity at fixed false alarms; latency | L2 | Build or simulate the front end | 3 |
| 36 | PPG skin-tone and motion audit | Public or approved PPG; device metadata | Stratified MAE; bias; coverage | L2 | Check consent and device terms | 3 |
| 37 | TinyML arrhythmia on an MCU | PTB-XL or MIT-BIH; target board | Accuracy after quantization; latency; RAM | L2 | Open data; hardware cost | 4 |
| 38 | Surgical phase recognition | Cholec80; public surgical video | Macro-F1; segmental edit distance | L3 | Consent and privacy review | 4 |
| 39 | Robot kinematics and safety monitor | Simulation; synthetic forces | Safety violations; stop latency | L2 | Simulation only | 3 |
| 40 | CGM compression-low detection | OhioT1DM; public CGM | Detection sensitivity; false alarms per day | L2 | Data-use terms; safety claim | 4 |
| 41 | Exoskeleton gait intent detection | Public IMU/EMG; open gait data | Intent accuracy; assistance latency | L2 | Check consent; fall risk | 4 |

#### Efficient AI and on-device inference

| # | Brief | Data option | Primary metric | Min level | Access note | Diff. |
|---:|---|---|---|---|---|---:|
| 42 | Quantized ECG on a wearable MCU | PTB-XL or MIT-BIH; target board | Sensitivity at fixed false alarms; p99 latency | L2 | Open data; energy measurement | 4 |
| 43 | On-device LLM triage | HealthBench subset; synthetic notes | Factuality; safety recall; latency; memory | L2 | Check model licence; no PHI | 5 |
| 44 | Surgical perception on edge hardware | Cholec80; dVRK simulation | p99 latency; frame-drop rate; macro-F1 | L3 | Consent; fail-safe fallback | 4 |

Two rules apply to the hardware and efficient-AI groups.
The deliverable must include a signal-chain diagram, a device or firmware version, calibration
assumptions, a hardware-in-the-loop or simulation test, and a failure-mode register.
A notebook score is not a device claim, and a desktop benchmark is not evidence of on-device
performance.

### Step 1. Read the brief three times

**Input:** a catalogue entry.

**Output:** the five visible fields separated into a table, plus the implied owner written as a
question.

**Rationale:** The first read is for the decision, the second for the data, the third for the
constraint that will bind. Reading once produces the illusion of understanding, because the
prose is fluent and the domain is familiar.

**Branch point:** if the decision cannot be stated in one sentence after three reads, the brief is
underspecified for this team. Choose another or write the decision and have it reviewed.

### Step 2. Locate the owner

**Input:** the decision and the clinical or operational setting.

**Output:** a named role, a commitment of hours, and a first meeting date.

**Rationale:** The owner is the hidden field. A brief about triage implies an on-call clinician; a
brief about outreach implies a programme manager; a brief about scheduling implies an operations
lead. Without that person, the claim is not admissible and the error review cannot happen.

**Branch point:** if no owner can be found within two weeks, the brief is a methods study. Record
the change of archetype and proceed only if the methods question is worth twelve weeks.

### Step 3. Determine the data tier and licence

**Input:** the data option column.

**Output:** a tier label, a licence identifier, and the obligations the licence imposes.

**Rationale:** Access determines the schedule and the artefacts. A credentialed dataset requires
CITI training and a data-use agreement, which take weeks; a Kaggle competition dataset has rules
about external data and code release; a restricted registry requires a named protocol.

**Branch point:** if the licence prohibits the intended use, for example a research-only licence
and a commercial pitch, the brief cannot be used as written. Record the prohibition in the
regulatory memo even if the capstone is academic.

### Step 4. Check the metric is computable at the operating point

**Input:** the metric column and the owner's capacity.

**Output:** the metric plus the threshold and the prevalence at which it will be computed.

**Rationale:** Precision at the top 5 percent is computable only if the cohort is large enough and
the ranking is well defined with ties. A macro AUC across twelve findings is computable only if
every finding has enough positives. A metric that cannot be computed at the deployment prevalence
is a metric for a different project.

**Branch point:** if the brief's metric assumes a capacity the owner does not have, adapt the
metric and record the adaptation. Do not report a metric the decision cannot use.

### Step 5. Establish the minimum validation level

**Input:** the claim implied by the brief.

**Output:** L2, L3, or L4, with the dataset's sites and dates checked against it.

**Rationale:** The level is a property of the claim. A brief about a new site needs L3; a brief
about next year needs L2; a brief about a method needs a held-out split and a contamination
check. The data must be able to support the level, not merely the model.

**Branch point:** if the data cannot support the level, the brief must be adapted to a narrower
claim. Adaptation is legitimate and must be written; pretending the level is met is not.

### Step 6. Estimate the evidence burden and compare with capacity

**Input:** the level, the artefacts, and the team's hours.

**Output:** hours for validation, governance, and documentation, and the fraction of total
capacity they consume.

**Rationale:** A difficulty-5 brief typically consumes more than half of a four-person team's
capacity in non-modelling work. The estimate prevents a team from choosing a brief whose
governance alone exceeds its calendar.

**Branch point:** if the burden exceeds 40 percent of capacity, choose a lower-difficulty brief or
adapt the claim. A team that proceeds will discover the overrun in week 8.

### Step 7. Check the team's capability against the brief's core artefact

**Input:** the role map and the brief's method.

**Output:** a capability rating for the artefact that carries the claim: can the team build it?

**Rationale:** A team can learn a supporting skill in twelve weeks; it cannot learn a core
discipline and produce defensible evidence in the same period. The distinction is between
learning to use a segmentation library and learning to design a reader study.

**Branch point:** if the capability gap is in the claim-carrying artefact, choose another brief.
If it is in a supporting artefact, budget the learning time explicitly.

### Step 8. Identify the characteristic failure

**Input:** the brief and its archetype.

**Output:** the failure the brief is most likely to produce, and the early signal.

**Rationale:** Each brief inherits its archetype's failure. Brief 1 inherits the research-AUC-as-
clinical-claim failure; Brief 12 inherits the alert-burden failure; Brief 29 inherits the
ancestry-portability failure. Naming it makes the risk register specific.

**Branch point:** if the team cannot describe a mitigation, the brief is not understood well
enough to select.

### Step 9. Shortlist three

**Input:** the eligible briefs.

**Output:** three entries, each with a one-paragraph rationale and one named rejection risk.

**Rationale:** Three is enough to compare and few enough to evaluate properly. A shortlist of one
is a decision already made; a shortlist of ten is a reading exercise.

**Branch point:** if fewer than three briefs are eligible, the constraints are too tight. Relax
the difficulty ceiling or accept a narrower claim rather than forcing a shortlist.

### Step 10. Score the shortlist on a common frame

**Input:** the three entries.

**Output:** a table with decision clarity, owner access, data fitness, evidence feasibility,
regulatory tractability, and capability, each scored 1 to 5 with a one-line justification.

**Rationale:** A common frame prevents the comparison from becoming a preference contest. The
justifications matter more than the numbers, because they are what the panel will question.

**Branch point:** if two entries are within 0.3 points, choose on the characteristic failure the
team can best tolerate, and say so.

The scoring frame is deliberately blunt. It cannot capture whether a question is interesting, and
it should not, because interest is the variable that leads teams to overcommit. It captures
whether the team can finish, whether the claim is admissible, and whether the data can carry the
metric. Those are the variables that determine whether the dossier exists in week 12.

A team that finds the frame uncomfortable is usually protecting a preference. The discomfort is
useful evidence, and it should be written in the justification column rather than used to
override the score.

### Step 11. Write the adaptation memo

**Input:** the chosen brief and the constraints that require change.

**Output:** a one-page memo listing each change, its reason, its effect on the claim, and the
artefact affected.

**Rationale:** The memo is the difference between adaptation and substitution. It is reviewed at
G0 and appended to the charter, and it gives the panel a single place to see what changed.

**Branch point:** if an adaptation changes the decision, the brief has been substituted. Return
to Step 9 and choose a brief whose decision survives the team's constraints.

### Step 12. Confirm at G0 and record the rejection reasons

**Input:** the memo, the scores, and the owner's confirmation.

**Output:** a signed charter and a short rejection note for the two other shortlisted briefs.

**Rationale:** Rejection reasons are evidence of judgement. A team that can say why it did not
choose the prestigious brief is more credible than one that never considered alternatives.

**Branch point:** if the owner withdraws between shortlisting and G0, the decision is not
confirmed. Return to Step 2 with the second-ranked brief rather than proceeding without an owner.

### Step 13. Re-verify the constraint that binds

**Input:** the chosen brief and the evidence gathered in steps 2 to 6.

**Output:** a dated record of the licence, the access status, and the owner's commitment, checked
again within seven days of the start of work.

**Rationale:** Catalogue entries age. A dataset may be withdrawn, a competition may close, a
licence may change, and an owner may be reassigned. The re-verification is cheap and prevents the
most embarrassing form of failure: a project that begins on a fact that was true last term.

**Branch point:** if a binding constraint has changed, return to the decision table and choose
between adaptation and a different brief. Do not proceed on the assumption that the change is
temporary.

### Decision table: what to do when a brief fails a constraint

| Constraint failed | Adaptation available | When to adapt | When to reject |
|---|---|---|---|
| No owner | Narrow to methods claim; recruit reviewer | Methods question is valuable | Claim requires clinical action |
| Data licence prohibits use | Switch to an equivalent open dataset | Schema and population comparable | Only dataset with the needed labels |
| Metric not computable | Change metric to one the capacity supports | Decision unchanged | Metric is the decision |
| Validation level unreachable | Narrow claim to the reachable level | Claim remains useful | Claim is about the unreachable setting |
| Difficulty above capacity | Reduce scope; drop stretch artefacts | Core evidence still deliverable | Core evidence is the difficulty |
| Capability gap in core artefact | Mentor support; simplified method | Method is replaceable | Claim depends on the method |

### Edge cases and forced decisions

**The brief's dataset disappears.** Public datasets are sometimes withdrawn or relicensed. If the
brief has no alternative, the team must adapt to a comparable dataset and re-check the label
definition, because a different dataset usually means a different outcome ascertainment.

**Two teams want the same brief.** The programme may allow it, but the teams should differentiate
the claim or the population. Two identical projects produce two identical dossiers and a
comparison that neither team controls.

**The owner wants a different question.** This is common and usually healthy. The owner's question
may be more valuable and less feasible. Write both, compare them at G0, and choose one in
writing. Do not run two decisions in one capstone.

**The brief is easier than expected.** A team that finishes the core evidence in week 7 should add
a pre-specified secondary analysis, not a new decision. The secondary analysis is planned and
reported with the same discipline, and it does not change the primary claim.

**The brief is harder than expected.** Apply the cut table from Topic 02. The first cut is the
stretch tier, the second is error-analysis breadth, and the last is the validation level, which
requires a panel decision.

## 6. Worked example

A four-person team has 45 hours per week across twelve weeks, no credentialed data access, and a
clinical mentor who can give four hours per week. It shortlists three briefs from the catalogue:
Brief 2, knee MRI multi-finding classification; Brief 12, sepsis early warning; and Brief 29,
polygenic risk score portability.

### Why the shortlist shrinks

Brief 2 is difficulty 5, requires 12-finding macro AUC on a multimodal dataset, and assumes a
musculoskeletal radiologist who can adjudicate label provenance. The mentor is an internist. The
brief fails the capability check for the claim-carrying artefact: the team has no musculoskeletal
imaging expertise, and the label disputes described in the brief require a specialist.

Brief 12 requires MIMIC-IV, which is credentialed. The team has no CITI training and no executed
data-use agreement, and the institutional process takes four to six weeks. The brief fails the
feasibility gate at week 1. It could be adapted to a public sepsis dataset, but the public
alternatives have different label definitions and smaller cohorts, which changes the claim.

Brief 29 uses CAGI7 polygenic risk score challenges, open data, and a metric the team can compute:
AUC across ancestries. The difficulty is 4, the primary limitation is that the real-data component
is European-ancestry only, and the team has a statistician who can implement the methods. The
mentor can review the clinical framing at four hours per week. The brief passes all four gates
with an adapted claim: not "PRS should be offered", but "the portability gap is measurable, and
here is its size in the available cohorts".

The shortlist of three has become a shortlist of one plus two documented rejections. That is a
legitimate outcome, and the rejection reasons belong in the charter.

### The arithmetic for Brief 29

The CAGI7 PRS challenge provides real phenotypes and simulated phenotypes. The team plans to use
the simulated phenotypes to test the method across genetic architectures and the real phenotypes
to estimate the portability gap.

For a target half-width of 0.05 on an AUC near 0.70, the number of cases needed is approximately
1.96² × 0.70 × 0.30 / 0.05² = 323 cases per ancestry group. With three ancestry groups and a
case fraction of 0.10 in the available data, the team needs about 3,230 individuals per group,
or 9,690 in total, before exclusions.

The available simulated validation set is 50,000 individuals, which is sufficient. The real-data
validation set is a single ancestry subset, which is not. The team therefore computes the
portability gap on simulated architectures and reports the real-data result as a single-ancestry
estimate with an explicit limitation. That is an adaptation, and it is written in the memo.

### Diagnostic one: the licence-to-claim check

The first diagnostic is a single question: does the dataset's licence permit the claim the brief
implies?

For Brief 29, the CAGI data are provided for challenge assessment, and the model licences the
team might use have their own terms. If the team intended a commercial PRS product, the licences
would prohibit it. If the team intends a methods evaluation, they permit it. The check takes ten
minutes and prevents a regulatory memo that contradicts the data terms.

### Diagnostic two: the positive-case count

The second diagnostic is arithmetic. For every group in which the team plans to report a metric,
count the positives. If any group has fewer than about 30 events, the metric will have an
interval too wide to support a claim about that group, and the report must say so.

For Brief 29, the real-data subset has enough positives for a single-ancestry estimate and not
enough for a three-ancestry comparison. The team discovers this in week 1 rather than week 9.

### Varying one assumption

Suppose the institution completes a data-use agreement in week 2 and the team gains credentialed
access. Brief 12 becomes feasible.

The team re-scores it. Owner access rises because the mentor is an internist who works on sepsis;
data fitness rises to 4; evidence feasibility rises to 3; regulatory tractability stays at 2
because a real-time alert is likely a device function. The weighted score moves from 3.1 to 3.7,
which is close to Brief 29's 3.9.

The decision then turns on the characteristic failure the team can tolerate. Brief 12's failure is
alert burden and automation bias, which the team can study with a simulation but cannot observe
in a live workflow. Brief 29's failure is ancestry portability, which the team can measure
directly with the available data.

The team stays with Brief 29 and records the alternative. The record shows that the choice was
made on evidence, not on novelty.

### Effort and budget for the chosen brief

The team estimates 214 hours of work: 70 for data preparation and the benchmark harness, 60 for
model implementation and runs, 34 for statistical comparison, 20 for subgroup analysis, 15 for
the model card, and 15 for the dossier and rehearsal.
At 45 hours per week across twelve weeks, capacity is 540 hours, so the plan uses about 40
percent of capacity and leaves room for review and rework.

At a blended 60 dollars per hour, the labour is 12,840 dollars. Compute for the simulated
genotypes and the model runs is estimated at 700 dollars. The total is 13,540 dollars, which is
within a typical course budget and well below the cost of a credentialed data route, where the
hidden cost is legal review time rather than compute.

The budget arithmetic is not the point. The point is that the chosen brief has no external
dependency that can consume weeks, which is why its effective cost is lower than a brief with a
similar price and a data-use agreement.

### The G0 defense

At G0 the team presents three slides.

The first names the decision: whether the informatics group adopts a polygenic score pipeline for
research use, based on measured portability across simulated genetic architectures.

The second names the constraint that eliminated the alternatives: no credentialed access within
the project window, and no musculoskeletal specialist for the imaging brief.

The third names the limitation the team accepts: the real-data component is European-ancestry
only, so the portability claim is demonstrated on simulated architectures and reported as a
single-ancestry estimate on real data.

The panel approves with one condition: the team must add a sensitivity analysis showing how the
portability gap changes when the causal fraction varies, because that is the parameter most
likely to drive the result.

### Diagnostics in one line

The licence-to-claim check passed because the use is a methods evaluation.
The positive-case count passed for the simulated validation set and failed for the real-data
three-group comparison, which is why the claim was narrowed.
Both checks took under an hour in week 1 and determined the shape of the project.

### The decision record

The team writes a four-line record and stores it with the charter.

Chosen: Brief 29, PRS portability, adapted to simulated architectures plus a single-ancestry
real-data estimate.
Rejected: Brief 2, because the claim-carrying artefact needs musculoskeletal expertise the team
does not have; Brief 12, because credentialed access cannot be obtained within the project window.
Constraint that binds: no multi-ancestry real data, which limits the portability claim.
Review date: G2, when the simulated analysis is complete.

The record takes ten minutes and answers the panel's first three questions. It also prevents the
team from re-litigating the choice in week 7, when Brief 12 looks attractive again because the
seeded results are hard.

The record also names the review date. A decision without a review date becomes either permanent
or arbitrary, and neither is a decision. Naming G2 gives the team a scheduled moment to ask
whether the constraint has changed, and it gives the panel a signal that the choice was
conditional rather than defensive.

The review date should fall after the feasibility evidence is in and before the claim is locked.
For a twelve-week project that is week 5 or 6, which is early enough to change course and late
enough to know whether the simulated results are workable.

A record that is not written down will be reconstructed differently by each member. In the
worked example, one member remembers the rejection of Brief 12 as a timing problem, another as a
licensing problem, and a third as a decision by the mentor. All three are partly true, and none
is the recorded reason. The panel will ask one member and get one version, then ask another and
get a different one, and the inconsistency will be read as a weak decision process.

### What the exercise produces

Three artefacts: a shortlist table with scores and justifications, an adaptation memo for the
chosen brief, and a rejection note for the other two. Together they are about four pages, and
they answer the panel's first question before it is asked.

## 7. Code, query, or calculation

The block filters a subset of the catalogue against team constraints, ranks the eligible briefs,
and reports why each exclusion occurred. It uses only the standard library.

```python
briefs = [
    # id, name, difficulty, owner_hours, min_level, data_tier, licence_ok
    (1, "Aneurysm triage CTA", 5, 8, 3, "open", True),
    (2, "Knee MRI multi-finding", 5, 6, 3, "open", True),
    (4, "Pneumothorax triage", 3, 4, 2, "open", True),
    (9, "Chagas triage 5%", 4, 3, 2, "open", True),
    (12, "Sepsis early warning", 4, 6, 2, "credentialed", False),
    (17, "Readmission + SDOH", 3, 4, 2, "credentialed", False),
    (21, "FHIR med reconciliation", 4, 6, 3, "open", True),
    (25, "HealthBench evaluator", 3, 2, 2, "open", True),
    (29, "PRS portability", 4, 4, 3, "open", True),
    (31, "Single-cell label projection", 3, 1, 1, "open", True),
    (33, "Wastewater nowcasting", 3, 5, 2, "open", True),
    (37, "TinyML arrhythmia", 4, 2, 2, "open", True),
]
team = {"hours_per_week": 45, "weeks": 12, "max_difficulty": 4,
        "owner_hours": 4, "credentials": False}

def eligible(b):
    _, _, diff, owner, _, tier, _ = b
    if diff > team["max_difficulty"]:
        return False, "difficulty"
    if owner > team["owner_hours"]:
        return False, "owner hours"
    if tier == "credentialed" and not team["credentials"]:
        return False, "credentials"
    return True, "ok"

def score(b):
    _, _, diff, owner, level, _, _ = b
    fit = (team["max_difficulty"] - diff + 1) / team["max_difficulty"]
    own = min(owner / team["owner_hours"], 1.0)
    lvl = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.0}[level]
    return 0.45 * fit + 0.35 * own + 0.20 * lvl

kept = []
for b in briefs:
    ok, why = eligible(b)
    if ok:
        kept.append(b)
    else:
        print(f"excluded #{b[0]:2d} {b[1]:28s} {why}")
print(f"eligible {len(kept)} of {len(briefs)}")
for b in sorted(kept, key=score, reverse=True):
    print(f"#{b[0]:2d} {b[1]:28s} score {score(b):.2f} L{b[4]} diff {b[2]}")
print("capacity hours", team["hours_per_week"] * team["weeks"])
```

Expected output:

```text
excluded # 1 Aneurysm triage CTA          difficulty
excluded # 2 Knee MRI multi-finding       difficulty
excluded #12 Sepsis early warning         owner hours
excluded #17 Readmission + SDOH           credentials
excluded #21 FHIR med reconciliation      owner hours
excluded #33 Wastewater nowcasting        owner hours
eligible 6 of 12
# 4 Pneumothorax triage          score 0.71 L2 diff 3
#29 PRS portability              score 0.66 L3 diff 4
#25 HealthBench evaluator        score 0.54 L2 diff 3
# 9 Chagas triage 5%             score 0.51 L2 diff 4
#37 TinyML arrhythmia            score 0.43 L2 diff 4
#31 Single-cell label projection score 0.39 L1 diff 3
capacity hours 540
```

The scoring function rewards fit to the difficulty ceiling, owner availability, and the strength
of the claim the brief can support. It does not reward novelty, and it should not: novelty is not
a criterion a panel scores.

Two limits are worth stating. The data are a twelve-brief subset, not the full catalogue. And the
scores are judgements encoded as numbers; changing the weights by 0.1 can reorder briefs whose
scores are within 0.05 of each other, which is why the shortlist comparison in Section 5 uses a
justification column.

## 8. Data, standards, and regulatory context

The catalogue spans four access regimes, and the regime determines the artefacts as much as the
data do.

| Regime | Examples in the catalogue | Obligation | What it constrains |
|---|---|---|---|
| Open licence | CheXpert, PTB-XL, Sleep-EDF, CMS SynPUF, Synthea | Attribution and licence terms | Redistribution and commercial use |
| Competition rules | RSNA via Kaggle, PhysioNet Challenge sets | Competition agreement; code and data rules | External data, team merging, code release |
| Credentialed | MIMIC-IV, eICU, MIMIC-CXR, MIMIC-IV-Note | CITI training, DUA, no redistribution | Storage, access lists, artefacts |
| Restricted | CIBMTR-style registry, institutional EHR, NVDRS | IRB approval, named protocol | Named uses only, retention |

### Competition datasets

Kaggle-hosted challenges, including the RSNA series, impose competition-specific rules: a code
competition may require notebook submission, prohibit internet access, cap runtime, and limit
final submissions. Those rules bind during the competition and shape what a team can build.
Using the data after the competition is governed by the dataset licence, which is separate from
the competition rules and must be read individually.

The RSNA 2026 Knee Abnormality Detection challenge is an example of a rule set that changes the
engineering: more than 5,000 exams from 19 sites, reports in about 12 languages, notebook-only
submission with a runtime cap, and a separate efficiency track. A capstone that uses the dataset
outside the competition is not bound by the runtime cap, but it is bound by the licence, and it
inherits the label provenance issues that the multimodal design creates.

### PhysioNet datasets

PhysioNet hosts open and credentialed data under different terms. Credentialed access requires an
individual account, completion of the CITI "Data or Specimens Only Research" course, and a
signed data-use agreement per dataset. The agreement prohibits redistribution and requires that
access not be shared, which means a course cannot circulate the files.

The PhysioNet Challenge data add a competition layer. The 2025 Challenge used CODE-15%,
SaMi-Trop, and PTB-XL for training, with hidden validation and test sets from other sources. The
2026 Challenge uses polysomnography from the Human Sleep Project with hidden sites. A team using
these data must respect both the challenge terms and the underlying dataset terms.

### Genomics and benchmark data

CAGI7, the seventh Critical Assessment of Genome Interpretation, released fourteen challenges in
2025, including variant effect prediction and polygenic risk scores. The PRS challenge provided
summary statistics and simulated genotypes, held out validation genotypes, and accepted
containers that ran locally, because the training genotypes were not shared. That structure is a
data-governance design as much as a scientific one: the assessors kept control of the genotypes
and evaluated code instead.

The Open Problems in Single-Cell Analysis benchmarks publish data and a harness, with control
methods that anchor the metric. Using them requires citing the benchmark version and respecting
the licence of each dataset in the collection, which varies.

### Standards by domain

Imaging briefs inherit DICOM and, for exchange, FHIR ImagingStudy. EHR briefs inherit FHIR or
OMOP, and the choice affects cohort definitions. Genomics briefs inherit VCF, HGVS, and GA4GH
conventions. Trial briefs inherit CDISC SDTM and ADaM. Documentation briefs inherit FHIR
DocumentReference and the audit requirements that go with writing to a record.

The standards constrain the pipeline, not the claim. A team that uses OMOP concepts without
recording the concept IDs cannot reproduce its cohort, and a team that reports variant effects
without HGVS notation cannot be checked.

### Regulatory regimes represented in the catalogue

Briefs 1, 2, 4, 5, 10, and 11 imply device functions under the FDA clinical decision support
guidance or the medical device regulations, because they analyse images or signals or provide
time-critical alerts. Briefs 12, 15, 17, and 21 imply clinical decision support, with the
classification depending on the output and the user. Briefs 22 to 27 imply documentation or
evaluation, with device exposure only if the output enters a clinical workflow. Briefs 28 to 31
imply in vitro diagnostic exposure only if the output becomes a clinical test.

The catalogue's regulatory column is a prompt, not a determination. The determination requires the
intended use, the user, and the output, and it belongs in the week-2 memo.

### Synthetic data as a brief option

Several briefs offer a synthetic route: Synthea for FHIR quality measures, synthetic notes for
prior-authorization drafting, simulated forces for the robot safety monitor.
Synthetic data solve an access problem and create an inference problem. A model evaluated on
Synthea has been evaluated on a generator's assumptions, not on a hospital's patients, and the
result cannot be presented as clinical evidence.

The briefs that use synthetic data are written for archetypes where the question is about
plumbing or workflow rather than patient outcomes. Brief 34 asks whether a clinic can produce a
compliant quality measure from FHIR resources; the answer depends on the specification, not on
the patients. Brief 23 asks whether a drafting workflow produces acceptable letters; the answer
depends on the rubric and the reviewer.

A team that uses synthetic data must state the generator, the version, the calibration source,
and the parts of the question the data cannot answer. The statement belongs in the datasheet, not
in a footnote.

Synthetic data also affect the reproducibility artefact. A dossier that ships a generator seed and
a configuration can be reproduced without the real data, which is an advantage; a dossier that
ships only the synthetic CSV cannot be checked against the generator's assumptions.

### Data standards in operational briefs

Briefs 16, 19, and 33 use operational data, which rarely arrives in a clinical standard. Bed
management systems export CSV, scheduling systems export proprietary formats, and wastewater
surveillance data arrive as CSV with site identifiers that must be protected.
The standards that apply are therefore local: a data dictionary, a refresh cadence, and a
definition of the operational event. A team that assumes FHIR for an operating-room schedule will
spend a week discovering that the scheduler has never heard of it.

## 9. Industry practice and case evidence

### What the challenge record shows about brief difficulty

The RSNA 2024 Lumbar Spine challenge, with 1,874 teams reported, asked for localisation and
grading of three conditions across spinal levels. The first-place solution used a two-stage
pipeline: vertebral level detection followed by region-of-interest classification with a
bi-LSTM and attention-based multiple-instance model. The architecture is a direct response to the
brief's structure, not a generic model choice, which is what a well-specified brief produces.

The RSNA 2025 aneurysm challenge, with 1,147 teams, asked for detection across 13 anatomical
locations. The first-place solution used vessel segmentation followed by location-aware
classification, reporting an AUC of 0.915 for presence and 0.916 across locations. Again, the
brief's structure, not the leaderboard, determined the method.

The lesson for brief selection is that difficulty ratings compress a lot of structure. A brief
that requires localisation is harder than one that requires a single label, because the label
must be attached to an anatomical object and the object must be found first. A team that reads
only the metric will miss that.

### What the evidence says about data limitations

The PhysioNet 2025 Chagas Challenge scored the fraction of true cases in the top 5 percent of a
ranking. The top test score was 0.323, and performance on an external, largely asymptomatic
cohort fell to about 0.100. The organisers report 111 teams and 41 ranked entries.

The gap is the most useful fact in the challenge for brief selection. A brief that names a
development dataset and a metric does not guarantee that the metric transports. The catalogue's
access column and level column exist to make that risk visible before the team commits.

The 2026 PhysioNet Challenge, on cognitive impairment from polysomnography, uses an age-conditioned
AUROC with a two-year window and hidden sites with deliberate missingness. The design removes a
shortcut and tests robustness, and it shows what a well-constructed brief looks like: the metric
encodes the clinical constraint.

### What benchmark studies show about simple methods

The Open Problems in Single-Cell Analysis benchmarks reported in 2025 that logistic regression on
log-normalised principal components was the best method in five of eight label-projection
datasets. The result matters for Brief 31, which is explicitly framed as a comparison against a
logistic baseline. A team choosing that brief should expect the baseline to be strong, and should
plan to explain why rather than to assume a neural method will win.

### What the competition landscape shows about scope

The MedGemma Impact Challenge, announced in January 2026 and concluded with winners in March
2026, asked for prototypes on open-weight models in low-resource settings. The winning projects
were narrow: offline surveillance, on-device tuberculosis screening, medication-error
reconciliation. None attempted a full clinical validation.

That is the scope calibration a capstone needs. A twelve-week project can produce one narrow
prototype plus an evaluation, or one evaluation without a prototype, and rarely both at a
standard a panel will accept.

### What the rules of a live competition add

The RSNA 2026 Knee Abnormality Detection challenge, open through October 2026, illustrates how a
rule set changes the engineering as well as the science. The task is multimodal: MRI exams paired
with reports in about twelve languages, scored by a macro-averaged AUC across twelve findings.
The submission format is a notebook with a runtime cap, and a separate efficiency track rewards
performance per unit of runtime.

A team using the same dataset outside the competition is not bound by the runtime cap, but it is
bound by the data licence and by the label-provenance problem the brief describes. Reports in
twelve languages mean that the labels are derived from text, and text-derived labels carry
negation, uncertainty, and translation artefacts. The catalogue entry says "label provenance and
leakage disputes at scale" for that reason.

The general lesson for brief selection is that the rule set is part of the brief's constraints.
A team that reads the task but not the rules will build something that cannot be submitted, or
will submit something that breaks a rule it never read.

The same applies to the PhysioNet Challenges, whose rules govern team composition, external data,
and code release, and to any benchmark with a licence that differs from the competition terms.
A team should keep a one-page rule summary with the charter, because the rules are read once at
the start and needed again at submission.

### Inference, labelled

It is an inference that teams which write a rejection note for the briefs they did not choose
produce more coherent charters. The mechanism is that the note forces the comparison to be
explicit. No controlled evidence from capstone cohorts is cited.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Brief chosen by dataset | The first sentence names the data | Familiarity beats fit | Claim unsupported | State the decision first |
| Constraint read as risk | "Access should be fine" | Optimism about external processes | Late switch, lost weeks | Convert each constraint to a yes/no test |
| Difficulty rating trusted alone | No check of the binding constraint | Averages hide access and owner | Overrun at week 8 | Check owner, licence, level separately |
| Licence ignored | No licence identifier in the charter | Open assumed unrestricted | Unusable artefacts | Record licence and obligations |
| Metric not computable | No prevalence or capacity stated | Metric chosen from the brief | Unusable operating point | Compute at deployment prevalence |
| Owner assumed | "The clinical team will review" | No named person | Inadmissible claim | Owner test in week 2 |
| Silent adaptation | Charter differs from the brief | Changes made verbally | Panel detects inconsistency | Adaptation memo at G0 |
| Substitution | Title unchanged, decision changed | Prestige pressure | Dossier defends nothing | Re-sign the charter |
| Shortlist of one | No rejection notes | Decision already made | No evidence of judgement | Require three entries |
| Two decisions in one project | Two charters | Enthusiasm | Both incomplete | Choose one primary |
| Dataset withdrawn | No fallback named | Single-source dependency | Project stops | Fallback in the charter |
| Competition rules forgotten | External data used in a closed competition | Rules read once | Disqualification | Re-read rules at submission |
| PHI in prompts | Notes pasted into an external model | Convenience | Governance breach | De-identified or synthetic text only |
| Prototype presented as product | Demo works, no evaluation | Hackathon habits | Credibility loss | Label the artefact |

Two of these failures can be detected from the charter alone, before any work begins: choosing by
dataset, and a shortlist of one. Both are visible as an absence, which is why the charter template
requires the decision sentence and the two rejection notes.

The others are detected at a gate. A licence ignored surfaces at G1, an owner assumed at G3 when
the error review cannot be scheduled, and a metric that cannot be computed at G2 when the first
results are reported at the wrong prevalence.

The table's organising idea is that every failure is a gate that was skipped.
Choosing by dataset skips feasibility and admissibility. Trusting the difficulty rating skips the
binding constraint. A shortlist of one skips the comparison. A silent adaptation skips the memo.
A team that can name which gate its project is closest to failing has a better risk register than
one that lists generic risks.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The catalogue's data options are accessible | Selection depends on it | Dataset withdrawn or relicensed | Check the licence page in week 1 |
| Difficulty ratings are comparable | Ranking depends on it | Ratings mix access and modelling | Decompose into constraints |
| An owner can be found for clinical briefs | Admissibility depends on it | No clinician available | Ask for a meeting within two weeks |
| Team capacity is 45 hours per week | Filtering depends on it | Other modules, placements | Use actual availability |
| A metric can be computed at deployment prevalence | Decision relevance depends on it | Cohort too small | Count positives by group |
| Adaptation preserves the decision | The memo's logic depends on it | Adaptation changes the action | Read the decision sentence aloud |
| Public benchmarks are uncontaminated | Methods claims depend on it | Pretraining on test data | Contamination screen |
| The brief's regulatory angle is a prompt | Triage depends on it | It is read as a determination | Regulatory mentor review |
| One brief per team is enough | Scope discipline depends on it | Sponsor wants two questions | Choose one, park the other |

The largest limit is that the catalogue is a snapshot. Datasets are withdrawn, competitions
close, and licences change. Every entry must be re-verified before a team commits, and the
catalogue's access notes are prompts for that verification rather than substitutes for it.

A second limit is that the briefs were written for a programme, not for a specific team. The
difficulty rating is an average across teams, and a team with an unusual skill set may find a
difficulty-4 brief easy and a difficulty-2 brief hard. The scoring frame in Section 5 is the
correction: it scores the team against the brief rather than the brief against the field.

A third limit is that the catalogue cannot tell a team whether a question is worth twelve weeks.
That judgement belongs to the owner and the panel, and it is the one judgement that no amount of
constraint checking can replace.

A fourth limit is that the briefs were written at a particular moment in the field. A brief that
was cutting-edge when written may be routine by the time a team starts, and a routine brief may
become urgent after a policy change. The catalogue date matters, and a team should ask whether
the question has already been answered by a public challenge or a paper.

The practical test is a ten-minute literature search before commitment. If three papers have
already answered the brief's question on the same data, the capstone should either replicate one
with a new population or choose a different brief.

## 12. Exercises

### Exercise 1. Extract five fields (application, 30 minutes)

Choose three catalogue entries and extract decision, data, metric, regulatory angle, and the
implied owner for each.
**Deliverable:** a three-row table.
**Criteria:** decisions are one sentence; owners are roles with hours; the regulatory angle is
stated as a question, not a conclusion.
**Time:** 30 minutes.

### Exercise 2. Run the constraint tests (calculation, 45 minutes)

For your top three briefs, test owner availability, licence, validation level, metric
computability, and capacity. Record pass or fail with evidence.
**Deliverable:** a pass/fail matrix with evidence links.
**Criteria:** every test has a verifiable source; failures are specific; no test is marked
"probably".
**Time:** 45 minutes.

### Exercise 3. Compute a positive-case requirement (calculation, 30 minutes)

For a brief with a binary outcome, compute the positives needed for a half-width of 0.07 around
an expected sensitivity of 0.75, then compute the cohort size at the brief's prevalence.
**Deliverable:** the arithmetic and a one-sentence feasibility conclusion.
**Criteria:** correct formula; prevalence stated; conclusion references the available data.
**Time:** 30 minutes.

### Exercise 4. Write an adaptation memo (design, 40 minutes)

Adapt a brief to your team's constraints without changing the decision. State each change and
its effect on the claim.
**Deliverable:** a one-page memo.
**Criteria:** the decision sentence is unchanged; each change is traceable to a constraint; the
effect on the claim is explicit.
**Time:** 40 minutes.

### Exercise 5. Write a rejection note (critique, 25 minutes)

Write a 200-word note explaining why your team rejected the most prestigious brief on its
shortlist.
**Deliverable:** the note.
**Criteria:** the reason is evidential, not motivational; it names the constraint; it does not
disparage the brief.
**Time:** 25 minutes.

### Exercise 6. Score a shortlist (application, 40 minutes)

Score three briefs on the six criteria and run the sensitivity test with owner weight at 0.35.
**Deliverable:** the table, the code output, and a decision.
**Criteria:** justifications for every score; sensitivity interpreted; tie resolved on
characteristic failure.
**Time:** 40 minutes.

### Exercise 7. Check a licence (application, 25 minutes)

Take one dataset from the catalogue and find its actual licence. Record the obligations and one
use it prohibits.
**Deliverable:** a licence record with a URL and access date.
**Criteria:** the licence is the current one; the prohibited use is specific; the record names
the obligations that affect the dossier.
**Time:** 25 minutes.

### Exercise 8. Defend the choice (design, 20 minutes)

Prepare a two-minute defense of your chosen brief that names the two rejected alternatives and
why.
**Deliverable:** a spoken script.
**Criteria:** the decision, the constraint, and the rejection reason are each stated in one
sentence; no claim exceeds the evidence.
**Time:** 20 minutes.

**Assessment note for the exercise set.** Exercises 1 to 3 test reading and arithmetic and have
single defensible answers. Exercises 4 to 8 test judgement and are assessed on the quality of the
record, not on whether the chosen brief is the one an instructor would have chosen. A team that
chooses a modest brief with a clear constraint argument should score higher than a team that
chooses an ambitious brief with an unexamined dependency.

## 13. Self-check answers

**1. Why is the owner the hidden field?**
Because briefs describe decisions, not the people who make them. The owner determines capacity,
workflow, error review, and whether the claim is admissible. A team that assumes an owner will
discover in week 8 that no one will sign the intended use.

**2. What is the difference between adaptation and substitution?**
Adaptation changes the population, level, or metric while preserving the decision. Substitution
changes the decision while keeping the title. The adaptation memo records the first; the panel
detects the second by comparing the charter with the results.

**3. Compute the positives needed for a half-width of 0.07 around 0.75.**
n = 1.96² × 0.75 × 0.25 / 0.07² = 3.8416 × 0.1875 / 0.0049 = 147.0, so about 147 positive cases.
At 10 percent prevalence, that is 1,470 records.

**4. Why does the difficulty rating not determine feasibility?**
Because it averages across constraints, and feasibility is set by the binding one. A difficulty-4
brief with open data and a committed owner can be more feasible than a difficulty-3 brief that
requires credentialed access the team does not have.

**5. What does the licence-to-claim check test?**
Whether the dataset's terms permit the use the claim implies. A research-only licence permits a
methods evaluation and prohibits a commercial product; a competition agreement may prohibit
external data during the competition. The check takes minutes and prevents a contradiction
between the dossier and the terms.

**6. Why count positives by group rather than in total?**
Because the claim is usually about a group, and a subgroup with fewer than about 30 events
produces an interval too wide to support a group-specific statement. A large total can hide a
small subgroup.

**7. Why keep rejection notes?**
Because they are evidence of judgement, and because they protect the team from revisiting the
decision every time the chosen brief becomes difficult. A note written in week 1 is more
persuasive than a recollection in week 10.

**8. What should a team do if the brief's dataset is withdrawn?**
Switch to the named fallback if one exists; otherwise adapt to a comparable dataset and re-check
the label definition. A different dataset usually means different outcome ascertainment, so the
claim may narrow.

**9. Why is a shortlist of one a problem?**
Because it provides no evidence that the choice was compared. A panel cannot tell whether the
team chose well or simply chose first, and the rejection notes are the only artefact that
distinguishes the two.

**10. Rubric for Exercise 4.** A strong memo keeps the decision sentence identical, traces each
change to a named constraint, and states the effect on the claim in the claim's own vocabulary. A
weak memo lists changes without saying what they do to the claim, which is substitution with
paperwork.

**11. Rubric for Exercise 2.** A strong pass/fail matrix cites a licence page, an access status,
and a named person for every test, with a date. A weak matrix marks items "likely" or "should be
fine", which converts constraints back into risks.

**12. Why does the catalogue include briefs that no course team can complete?**
Because the catalogue is a map of the field, not a menu of guaranteed projects. Difficulty-5
briefs show what multi-site, high-risk work requires, and a team that reads them learns why a
narrower brief is a legitimate choice rather than a failure of ambition.

**13. What does a rejection note protect against?**
It protects against re-litigation and against the panel's inference that no comparison occurred.
It also protects the team's confidence in week 7, when the chosen brief is difficult and the
rejected one looks easier in memory than it was in the assessment.

## 14. Interview questions

**Q1. "Why did you choose this project?"**
Strong answer names the decision, the owner, and the constraint that eliminated the alternatives.
Weak answer says the topic was interesting.

**Q2. "What did you reject, and why?"**
Strong answer names a brief and the specific constraint that failed. Weak answer says the others
were not feasible, without detail.

**Q3. "How did you check the data were usable?"**
Strong answer names the licence, the tier, the access process, and the date checked. Weak answer
says the data were public.

**Q4. "What is your metric, and why that one?"**
Strong answer ties it to capacity and error costs. Weak answer says it is standard in the
literature.

**Q5. "How did you handle a dataset limitation?"**
Strong answer describes the adaptation, the memo, and the narrowed claim. Weak answer says the
limitation was acknowledged in the limitations section.

**Q6. "What would you have done with credentialed data?"**
Strong answer names the brief that would have become feasible and the claim it would support.
Weak answer says the results would be better.

**Q7. "How do you know the owner would use this?"**
Strong answer describes the workflow, the capacity number, and the review sessions. Weak answer
says the clinician was interested.

**Q8. "What is the regulatory status of your output?"**
Strong answer names the regime and the criteria applied. Weak answer says it is research.

**Q9. "What was the hardest constraint?"**
Strong answer names it and shows the arithmetic. Weak answer says time.

**Q10. "How would you brief a new team on this catalogue?"**
Strong answer describes the five fields, the hidden owner, and the four gates. Weak answer says
read the descriptions carefully.

**Q11. "What would you do if the dataset licence changed tomorrow?"**
Strong answer returns to the decision table, names the fallback, and describes the adaptation
memo. Weak answer says the project would be in trouble.

**Q12. "Which brief would you choose if you had a clinician and credentialed access?"**
Strong answer names one and explains the constraint that changes, without claiming the project
would be easy. Weak answer lists several prestigious briefs.

The strongest answers to this set share a property: they treat the brief as a set of constraints
rather than a title. A candidate who can say "this brief is infeasible for us because of the
licence, and here is the adaptation that preserves the decision" has demonstrated the judgement
the topic is teaching.

## 15. Key terms

- **Adaptation** — change that preserves the decision.
- **Adaptation memo** — one-page record of changes and their effect on the claim.
- **Brief** — a pre-scoped problem with decision, data, metric, angle, and difficulty.
- **CAGI** — Critical Assessment of Genome Interpretation.
- **CDISC** — standards body for clinical trial data.
- **Constraint** — a condition that must hold for feasibility.
- **Contamination risk** — test data present in training.
- **Coverage** — proportion of the target population represented.
- **Credentialed access** — access requiring training and a data-use agreement.
- **Data option** — the named dataset a brief expects.
- **Data tier** — open, credentialed, restricted, or synthetic.
- **Difficulty rating** — 1 to 5 scale of scope and risk.
- **DUA** — data-use agreement.
- **External partner** — an organisation supplying data or workflow.
- **Label provenance** — how the outcome was ascertained.
- **Licence check** — verification that use is permitted.
- **Metric computability** — whether the metric works at deployment prevalence.
- **Minimum validation level** — lowest level supporting the claim.
- **OMOP** — common data model for observational data.
- **Open Problems** — single-cell benchmarking initiative.
- **Operating point** — threshold with sensitivity and specificity.
- **Owner test** — whether a named person will act.
- **PHI** — protected health information.
- **Pre-registration** — analysis plan written before results.
- **Shortlist** — three compared briefs.
- **Substitution** — change of decision behind an unchanged title.
- **Rule set** — competition constraints on data, code, and submission.
- **Sandbox** — isolated environment for testing without real records.
- **Benchmark version** — dated release of a benchmark and harness.
- **Public use file** — de-identified dataset for general research.
- **Refresh cadence** — how often a source updates.
- **Cohort definition** — rules selecting the analysis population.

A final note on units: difficulty is unitless, hours are hours, and access dates are dates. The
catalogue mixes all three, and a team that reads "difficulty 4" as "four weeks" will plan badly.

The list is deliberately operational. Every term in it appears in a brief's constraint column,
and a team that cannot define them will read a brief as a description rather than a specification.

## 16. Sources

All URLs accessed 26 October 2026.

- RSNA, Knee MRI AI Challenge, 2026.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- RSNA, Intracranial Aneurysm Detection AI Challenge, 2025.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/intracranial-aneurysm-detection-ai-challenge
- RSNA, Lumbar Spine Degenerative Classification AI Challenge, 2024.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge
- Reyna MA, Koscova Z, Pavlus J, et al. Detection of Chagas disease from the ECG: the George B.
  Moody PhysioNet Challenge 2025. arXiv:2510.02202. https://arxiv.org/abs/2510.02202
- PhysioNet Challenge 2026. https://moody-challenge.physionet.org/2026/
- PhysioNet, required training for MIMIC-IV.
  https://physionet.org/content/mimiciv/view-required-training/3.0/
- CAGI7 PRS challenge. https://genomeinterpretation.org/cagi7-prs.html
- CAGI7 challenges. https://genomeinterpretation.org/challenges.html
- Luecken MD, Gigante S, Burkhardt DB, et al. Defining and benchmarking open problems in
  single-cell analysis. *Nature Biotechnology* 2025.
  https://doi.org/10.1038/s41587-025-02694-w
- Google Research, MedGemma Impact Challenge winners, March 2026.
  https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- FDA, *Clinical Decision Support Software*, guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- CheXpert dataset page. https://stanfordmlgroup.github.io/competitions/chexpert/
- Synthea. https://synthetichealth.github.io/synthea/
- CDC National Wastewater Surveillance System. https://www.cdc.gov/nwss/
- HL7 FHIR US Core. https://hl7.org/fhir/us/core/

Source note: the catalogue's competition facts, team counts, scores, and described approaches
come from the organisers' pages and reports listed above, at the access date. Dataset licences
change; every entry must be re-checked before a team commits, and this list is a starting point
rather than a legal record.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_capstone-archetypes.md`](04_capstone-archetypes.md)
- Next topic: [`06_hackathon-formats.md`](06_hackathon-formats.md)
- Related topic: [`08_scoping-playbook.md`](08_scoping-playbook.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
