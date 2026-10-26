# Oncology

> Module 23 — Specialty Electives
> Topic 01 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08; helpful: 05, 12
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a staging
> authority, and not a substitute for the current AJCC, NCCN, or institutional
> protocol. Every threshold below names its source and version; where a source
> has a rolling release, the version used is stated explicitly.

## 1. Why this topic exists

Oncology is the specialty where a data error is most likely to be invisible and most
expensive to correct. The reason is structural: almost every oncology decision is
downstream of a stage, a histology, a biomarker, and a line of therapy, and each of
those four variables is produced by a different system, at a different time, under a
different coding standard, by a different person. When they disagree, nothing crashes.
The pipeline runs, the model trains, the dashboard renders, and the cohort is quietly
wrong.

Consider the ordinary path of a lung cancer case. A radiologist measures a lesion and
writes a report. A pathologist signs a diagnosis with a morphology code. A surgeon
removes a lobe and a second pathologist issues a synoptic report with a pathologic
stage. A tumour board assigns a clinical stage and a plan. A pharmacist dispenses a
regimen. A research coordinator abstracts the case into a registry. A claims system
records the administered drug under a billing code. Six systems now hold six partial
versions of the same patient, and the analytic table that joins them is the artifact on
which every subsequent claim depends.

The signature failures of oncology data science are not modelling failures. They are
definition failures:

- **Time zero drift.** The cohort is anchored at diagnosis for one patient, at first
  treatment for another, and at randomization for a third. Survival is then compared
  across groups that do not share a starting line.
- **Stage edition mixing.** A 2016 case was staged under the seventh edition, a 2020
  case under the eighth, and a 2026 lung case under the ninth. Pooling them into one
  "stage III" variable creates a category that never existed in any single rulebook.
- **Clinical versus pathologic stage confusion.** cTNM and pTNM answer different
  questions. A model that uses whichever is present learns to predict treatment
  selection, not biology.
- **Response criteria substitution.** RECIST 1.1 is used in trials; real-world
  progression is inferred from radiology text and scan intervals. The two are not
  interchangeable, and treating them as interchangeable manufactures effect.
- **Immortal time.** Patients must survive long enough to receive the treatment that
  defines the exposure group. A naive treated-versus-untreated comparison therefore
  credits the treatment with survival that occurred before it started.

Each of these is cheap to detect at design time and nearly impossible to repair after a
result has been presented to a tumour board, a regulator, or an investor.

### 1.1 Who is blocked when the definition is wrong

The cost is distributed unevenly, and the people who bear it are rarely the people who
could fix it.

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | A therapy that was withheld or given on a misclassified stage | Cannot see the staging logic or the source documents |
| Oncologist | Confidence in a risk score that disagrees with their clinical impression | Cannot audit the cohort definition |
| Tumour board | Time spent reconciling three stage values for one case | No single accountable staging record |
| Pharma sponsor | A trial that fails on a population the drug would have helped | Eligibility is encoded in the protocol, not the data |
| Payer | Spend on a biomarker-guided therapy with a mis-specified eligible population | Sees claims, not stage or histology |
| Registry | Published survival that shifts when the edition changes | Abstractors follow the manual in force at abstraction |
| Data scientist | A model that validates internally and fails on the next site | Inherited definitions were never documented |

The last row is the professional risk. An oncology model that cannot state its stage
edition, its time zero, and its response rule is not a weak model; it is an
unfalsifiable one, and a reviewer who knows the domain will say so in the first
meeting.

### 1.2 What this topic covers and what it does not

This topic covers the measurement layer: how a cancer case becomes structured data, how
stage is assigned and versioned, how response is computed, how endpoints and censoring
are defined, and how those choices propagate into an analysis. It does not teach
oncology practice, drug mechanisms, or trial design. It assumes you have met survival
analysis and will not re-derive the Kaplan-Meier estimator, though it does insist on the
censoring assumptions that estimator requires.

The emphasis is deliberate. Most oncology data-science curricula spend their time on
genomic feature engineering and almost none on the fact that the same patient can carry
three different stages in three different tables. The second problem destroys more
projects.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Assign a TNM stage group from T, N, and M categories and state which edition's rules you applied | Exercise 1; self-check 1 |
| 2 | Distinguish clinical, pathologic, post-neoadjuvant, and recurrence stage, and choose the right one for a stated question | Exercise 2; interview question 2 |
| 3 | Compute RECIST 1.1 target-lesion selection, sum of diameters, and timepoint response with correct reference points | Exercise 3; self-check 3 |
| 4 | Define time zero, index date, and censoring for overall survival, progression-free survival, and real-world progression-free survival | Exercise 4; self-check 4 |
| 5 | Detect immortal time bias and repair it with a landmark, a time-dependent covariate, or clone-censor-weight | Exercise 5; interview question 4 |
| 6 | Explain stage migration and demonstrate its effect on stage-specific survival with a constructed example | Exercise 6; self-check 6 |
| 7 | Audit biomarker missingness and distinguish missing at random from missing because the test was never ordered | Exercise 7; interview question 7 |
| 8 | Map an oncology data product to its regulatory path: device, in-vitro diagnostic, companion diagnostic, or label change | Exercise 8; interview question 9 |

Outcomes 1 and 2 are about the staging record. They are separated because a learner can
often derive a group from a table and still be unable to say which prefix and edition
the group belongs to. The second is the skill that survives a guideline update.

Outcomes 3 and 4 are about the event. They are the quantitative core of the topic and
the only outcomes that require arithmetic. A learner who can compute RECIST but cannot
state a censoring rule has learned a formula, not a measurement system.

Outcomes 5 and 6 are about bias. They are assessed by constructing a demonstration, not
by naming a bias. "I would adjust for immortal time" is not evidence; a landmark
analysis that changes the hazard ratio from 0.62 to 0.94 is.

Outcomes 7 and 8 are about context. They connect the measurement layer to the
regulatory and equity questions that decide whether the work can be used.

### 2.1 How the outcomes map to the module

Outcomes 1 to 3 are the measurement core and are prerequisites for the cross-specialty
comparison in topic 07. Outcome 4 is shared with critical care and public health, where
time zero has different names but the same consequences. Outcomes 5 and 6 are the
signature-bias audit that the module assessment rewards. Outcomes 7 and 8 connect to
module 12 (omics and precision medicine) and module 14 (device regulation).

If you retain only one outcome, retain outcome 4. Every other error is visible in a
table; a time-zero error is invisible because it produces a plausible number.

## 3. The core mental model

An oncology analysis is a chain of translations. Each arrow loses information, and each
loss is a place where a definition can be wrong.

```text
  Patient with symptoms or screen detection
        |
        v
  [1] Diagnosis ............. histology (ICD-O-3), grade, site
        |                     source: pathology report
        v
  [2] Stage ................. T, N, M -> stage group
        |                     edition, prefix (c / p / yp / r), date
        v
  [3] Prognosis / fitness ... stage, performance status, comorbidity, biomarkers
        |
        v
  [4] Treatment decision .... regimen, intent (curative / palliative), line
        |
        v
  [5] Response .............. RECIST 1.1 (trial) or inferred (real world)
        |
        v
  [6] Endpoint .............. OS, PFS, DFS, EFS, ORR, DoR, pCR, MRD
        |
        v
  [7] Payment / regulation .. drug label, companion diagnostic, device clearance
```

Read the diagram as a set of joins, not a set of stages. The analytic table is built by
joining [1] to [2] to [4] to [5] to [6]. Every join has a key and a date, and the most
common defect is that the join is done on the patient identifier without regard to the
date, so a stage assigned after progression is attached to a baseline feature.

### 3.1 The three clocks

Oncology has three clocks running at once, and confusing them is the root of most
temporal leakage.

| Clock | Origin | Used for | Typical error |
|---|---|---|---|
| Disease clock | Date of diagnosis (or first evidence of disease) | Staging, incidence, survival from diagnosis | Using treatment start as if it were diagnosis |
| Treatment clock | Date of first dose of the line under study | PFS, response, line-of-therapy duration | Back-dating to a decision date that follows a scan |
| Study clock | Randomization or enrolment | Trial endpoints, landmark analyses | Mixing trial and real-world clocks in one cohort |

A defensible analysis names one clock as primary and reports the others as sensitivity
analyses. A common and defensible choice is: diagnosis clock for prognosis questions,
treatment clock for comparative effectiveness questions, and study clock for trials.
What is not defensible is a cohort in which the clock varies by patient without a column
that records which clock was used.

### 3.2 The staging triangle

Stage is not one variable. It is a triple of (categories, edition, prefix), plus a date.
Any analysis that stores only "stage" has thrown away the information needed to check
it.

```text
           edition (7th / 8th / 9th)
                  /\
                 /  \
                /    \
   prefix -----/------\----- categories
 (c, p, yp, r)          (T, N, M, grade, biomarkers)
```

AJCC moved from numbered editions to a rolling Version 9 model: individual disease-site
protocols are released in batches and take effect on 1 January of the year after
release. The lung protocol was released in 2025 and took effect 1 January 2026; the
salivary gland and HPV-associated oropharyngeal protocols took effect the same date.
Sites not yet replaced remain on the eighth edition. The UICC published its ninth
edition in July 2025, effective 1 January 2026, and it incorporates the corresponding
AJCC changes. The practical consequence for a data team is that "stage III" is
versioned data, not a constant, and a multi-year cohort will contain at least two
rulebooks unless the institution re-abstracted historical cases, which most did not.

### 3.3 Response is a state machine, not a measurement

RECIST 1.1 defines a sequence of timepoint assessments relative to two different
reference points: baseline for shrinkage, and the on-study nadir for growth. The same
sum of diameters can be a partial response, stable disease, or progressive disease
depending on which reference is used and what happened earlier.

```text
  baseline SLD ──(-30%)──> PR threshold
       │
       └── running nadir ──(+20% AND +5 mm)──> PD threshold
```

The "AND" matters. A 22 percent increase from a 10 mm nadir is 2.2 mm, which does not
meet the 5 mm absolute criterion and is therefore not progressive disease by target
lesions alone. Analyses that implement only the percentage branch over-call
progression on small lesions, which is exactly the situation in which a patient is
taken off an effective drug.

### 3.4 What the model is allowed to see

The final mental model is a wall at the decision time. Features to the left of the wall
are admissible; features to the right are leakage. In oncology the wall is often drawn
at the treatment decision, and the most common violations are stage recorded after
surgery, biomarker results returned after the line started, and response assessments
used to decide the treatment being evaluated.

```text
  admissible ........................ | decision | ........... inadmissible
  demographics, baseline labs,        |          |  response, later stage,
  pre-treatment imaging, stage at     |          |  subsequent therapy, survival
  diagnosis, performance status       |          |
```

## 4. Definitions and vocabulary

Every term below is used later. The "common misuse" column is the part worth memorizing,
because the misuse is what appears in real analytic tables.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| TNM | Classification of malignant tumours by primary tumour (T), regional nodes (N), and distant metastasis (M) | Treating the stage group as the primary datum and discarding T, N, M | AJCC / UICC categories |
| Stage group | Alphanumeric summary (IA1, IIB, IIIA, IVB) derived from T, N, M and, in some sites, grade or biomarkers | Pooling groups across editions | AJCC stage group values |
| Clinical stage (cTNM) | Stage assigned from examination, imaging, and biopsy before treatment | Used as if it were pathologic stage in surgical cohorts | Prefix `c` |
| Pathologic stage (pTNM) | Stage assigned after resection and microscopic examination | Applied to patients who never had surgery | Prefix `p` |
| Post-neoadjuvant stage (ypTNM) | Pathologic stage after preoperative therapy | Pooled with pTNM, which mixes treated and untreated biology | Prefix `yp` |
| Recurrence stage (rTNM) | Stage assigned at recurrence | Used as baseline stage | Prefix `r` |
| Histology | Tissue type and subtype, coded by morphology | Treated as a free-text field | ICD-O-3.2 morphology code |
| Grade | Differentiation, site-specific scheme | Assumed comparable across tumour types | Site-specific (e.g., Gleason, Nottingham, WHO grade) |
| Performance status | Functional capacity scored by clinician | Recorded once and carried forward across lines | ECOG 0–5; Karnofsky 0–100 |
| Line of therapy | Ordered sequence of systemic regimens in advanced disease | Inferred from drug dates without regimen logic | Institution- or vendor-defined |
| Measurable disease | Lesion meeting RECIST 1.1 size criteria at baseline | Counting any visible lesion | mm, longest diameter; short axis for nodes |
| Target lesion | Up to 5 measurable lesions, at most 2 per organ, chosen at baseline | Re-selecting targets at follow-up | Count and organ |
| SLD | Sum of diameters of target lesions | Recomputing the baseline sum after a lesion disappears | mm |
| Nadir | Smallest SLD observed on study, including baseline | Using the previous scan instead of the nadir | mm |
| CR | Disappearance of all target lesions; nodes <10 mm short axis | Calling CR with residual nodes of 12 mm | RECIST 1.1 category |
| PR | At least 30 percent decrease in SLD from baseline | Comparing to nadir | Percent |
| SD | Neither PR nor PD | Calling any small increase SD when it meets PD criteria | RECIST 1.1 category |
| PD | At least 20 percent increase from nadir and at least 5 mm absolute increase, or new lesion | Applying the percentage without the 5 mm rule | Percent and mm |
| iRECIST | Immune-response criteria requiring confirmation of progression | Used interchangeably with RECIST 1.1 | iUPD, iCPD, iPR, iCR |
| ORR | Proportion with confirmed CR or PR | Counting unconfirmed responses | Proportion |
| DCR | CR plus PR plus SD at a defined timepoint | Mixing timepoints across patients | Proportion |
| DoR | Time from first response to progression or death among responders | Reported as a population effect | Months |
| OS | Time from time zero to death from any cause | Censoring at last contact when a death registry exists | Months |
| PFS | Time from time zero to progression or death | Treating scan date as progression date without interval censoring | Months |
| DFS | Time from curative treatment to recurrence or death | Pooled with PFS | Months |
| EFS | Composite of progression, recurrence, death, or protocol-defined event | Assuming the composite is comparable across trials | Months |
| rwPFS | PFS inferred from real-world data | Treated as trial PFS | Months |
| pCR | No invasive tumour at surgery after neoadjuvant therapy | Assumed to be a validated surrogate in every site | Proportion |
| MRD | Residual disease below conventional detection | Assumed comparable across assays | Assay-specific |
| ctDNA | Tumour-derived DNA in plasma | Treated as equivalent to tissue genotype | Variant allele fraction |
| ECOG PS | Eastern Cooperative Oncology Group performance status, 0 (fully active) to 5 (dead) | Recorded as a baseline constant | Ordinal 0–5 |
| Karnofsky | Functional status 0–100 in deciles | Converted to ECOG with an unvalidated mapping | Ordinal 0–100 |
| Immortal time | Person-time during which the exposure could not yet have occurred | Credited to the exposed group | Person-time |
| Lead-time bias | Earlier detection lengthens measured survival without delaying death | Interpreted as benefit | Months |
| Length-time bias | Screening preferentially detects slow-growing disease | Interpreted as benefit | Months |
| Stage migration | More sensitive staging reassigns patients upward, improving stage-specific survival without changing prognosis | Interpreted as treatment progress | Stage distribution |
| Overdiagnosis | Detection of disease that would never have caused harm | Counted as incidence reduction failure | Proportion |
| Surrogate endpoint | A measure substituted for the outcome of interest | Assumed validated | Endpoint-specific |
| Tumour board | Multidisciplinary meeting that assigns stage and plan | Treated as a data source of record | Workflow |
| mCODE | HL7 FHIR minimal Common Oncology Data Elements | Assumed to be a complete oncology model | FHIR profiles, STU 4 |
| NAACCR | North American Association of Central Cancer Registries, registry data standards | Assumed to match EHR fields | Registry data dictionary |

### 4.1 Categories that look like labels but are not

Three distinctions cause more rework than any others.

First, **histology is not a string**. "Adenocarcinoma" appears in lung, colon, prostate,
and pancreas with different ICD-O-3 codes, different staging rules, and different
systemic therapies. A model keyed on the string pools diseases that share a word.

Second, **stage is not ordinal in the way analysts assume**. The jump from IIIA to IIIB
is not a unit of progression; the groups are defined by different combinations of T, N,
and M, and the survival ordering is empirical, not built into the labels. Treating stage
as a continuous predictor and fitting a single coefficient imposes an assumption that
the data may not support.

Third, **line of therapy is an event, not an attribute**. A patient is on line 1 until
the regimen changes for a defined reason. Vendor definitions differ on whether a
treatment holiday, a dose reduction, or a maintenance switch starts a new line. Two
datasets that both report "second line" may not mean the same thing.

### 4.2 What counts as an event

The single most consequential coding decision in an oncology analysis is what counts as
an event and when it occurred. The table below is the minimum you should record for
each endpoint.

| Endpoint | Event | Censoring rule | Ascertainment source |
|---|---|---|---|
| OS | Death, any cause | Last known alive | Death registry, EHR, claims |
| PFS | Progression or death | Last adequate assessment | Imaging plus clinical notes |
| DFS | Recurrence or death | Last adequate assessment | Imaging, pathology, notes |
| EFS | Protocol-defined event set | Protocol-defined | Trial records |
| ORR | Confirmed CR or PR | Not applicable | Imaging reads |
| DoR | Progression or death after response | Last assessment for responders | Imaging reads |
| rwPFS | Inferred progression or death | Last activity plus a washout window | EHR, claims, imaging |

> **Note:** "Last adequate assessment" is not "last contact". Censoring at last contact
> when the last scan was eight months earlier assumes no progression in the gap, which
> biases PFS upward and biases comparative estimates whenever scan intervals differ
> between arms.

## 5. The mechanics, step by step

This section is the procedure for turning raw oncology records into an analysable
cohort. It is written as ten steps. Each step states its input, its output, the reason
it exists, its branch points, and its edge cases. A team that can execute these ten
steps in order, and document the choices, can defend the resulting dataset.

The procedure assumes the question is comparative or prognostic in a defined cancer
population. Screening programme evaluation has additional requirements (lead time,
overdiagnosis) that are noted but not developed here.

### Step 1 — Fix the decision and the cohort

**Input.** A written question naming a decision, a decision-maker, and an action.

**Output.** An eligibility specification: cancer site, histology set, stage range,
treatment setting (curative or advanced), age range, data availability requirements, and
the observation window.

**Rationale.** The cohort is a claim about who the result applies to. A cohort defined
by "patients with lung cancer in the EHR" includes small-cell and non-small-cell
disease, early and metastatic disease, treated and untreated patients, and prevalent
cases diagnosed before the data window. Any of those inclusions can dominate the
estimate.

**Procedure.**

1. Write the decision as a sentence with an actor and a verb: "the tumour board will
   choose adjuvant chemotherapy for stage II colon cancer using this recurrence score."
2. Translate the decision into an eligibility statement. If the decision applies only
   to resected stage II disease, the cohort is resected stage II disease, not all colon
   cancer.
3. Specify the diagnostic code set by ICD-O-3 morphology, not by ICD-10 site code
   alone. Site codes cannot separate histologies.
4. Specify the data availability rule: a patient must have a stage, a treatment record,
   and at least one follow-up contact, or the patient is excluded and the exclusion is
   counted.
5. Specify prevalent versus incident disease. For prognosis, use incident cases with a
   diagnosis date inside the window. For prevalence or burden questions, the rule
   differs.

**Branch point.** If the decision is about a therapy, the cohort should be defined by
eligibility for that therapy at the decision date, which may be narrower than the
diagnosis cohort. If the decision is about a diagnostic pathway, the cohort should be
defined by the population that presents for testing, not the population that tests
positive.

**Edge case.** Patients diagnosed outside the data window but treated inside it. These
are prevalent cases with left-truncated follow-up. Including them without accounting
for the truncation makes long survivors over-represented. Either exclude them or use
delayed-entry methods.

### Step 2 — Establish the diagnosis and histology

**Input.** Pathology reports, registry abstracts, or coded diagnosis tables.

**Output.** One diagnosis record per patient per primary cancer, with date, site,
morphology code, grade, and the source document.

**Rationale.** Histology determines staging rules, eligible biomarkers, and therapy.
Multiple primaries are common; a patient with breast and lung cancer is two records,
not one.

**Procedure.**

1. Identify the primary cancer date as the date of the first positive pathology
   specimen, or the date of the first clinical evidence if no specimen exists. Record
   which definition was used.
2. Code morphology to ICD-O-3.2. Do not map from ICD-10-CM malignant neoplasm codes,
   which lack the morphology dimension.
3. Apply the surveillance rules for multiple primaries in force for the registry. Rules
   differ between registries and across years, and the differences are large enough to
   change incidence trends.
4. Record grade using the site-specific scheme. Do not pool Nottingham grade for breast
   with WHO grade for neuroendocrine tumours.
5. Record the source and the confidence: histologically confirmed, cytologically
   confirmed, clinically diagnosed, or unknown.

**Branch point.** If a patient has a biopsy showing metastasis and no primary site, the
record is a cancer of unknown primary. Assigning the metastatic site as the primary is
a common and consequential error.

**Edge case.** Reclassification. A 2015 diagnosis of "bronchioloalveolar carcinoma" does
not map cleanly onto the 2015 WHO classification, and a 2021 diagnosis of a molecularly
defined entity may not have existed in 2015. Molecular reclassification (for example,
the separation of entities by driver alteration) changes the denominator of every
histology-specific analysis. Version the histology field with the WHO classification
edition and the report date.

### Step 3 — Collect T, N, and M separately

**Input.** Imaging reports, operative notes, pathology synoptic reports, tumour board
notes.

**Output.** Four category fields (T, N, M, plus modifiers), a prefix, an edition, and
the date of assignment.

**Rationale.** The stage group is derived. Storing only the group makes it impossible to
recompute under a new edition, to check internal consistency, or to distinguish a T3N0
tumour from a T1N2 tumour that happens to share a group.

**Procedure.**

1. Extract T from the imaging and pathology descriptions using the site-specific rules.
   Record the size in millimetres and the invasion details that determine the category.
2. Extract N from imaging and pathology. Record the nodal stations or levels examined,
   the number positive, and the number examined.
3. Extract M from imaging. Record the modality and the date. A PET scan and a CT scan
   performed a week apart can disagree; record which established M.
4. Assign the prefix: `c` for clinical, `p` for pathologic, `yp` after neoadjuvant
   therapy, `r` at recurrence.
5. Record the edition in force at the date of assignment.
6. Derive the stage group using the edition's algorithm and store both the inputs and
   the output.

**Branch point.** When clinical and pathologic stages disagree, do not overwrite one
with the other. Store both, and choose by question. For a surgical cohort, pTNM is the
appropriate staging variable. For a cohort that includes inoperable patients, cTNM is
the only variable available to everyone, and using pTNM where available creates a
selective cohort of surgical patients.

**Edge case.** The staging discrepancy. A patient staged cT3N1 by imaging is found at
surgery to have pT2N0 disease. This is not a data error; it is the expected
consequence of imaging resolution. If the analysis uses pTNM, the patient moves from
stage IIIA to stage IB, and the cohort definition changes. A protocol that says "stage
III" without specifying the prefix will include this patient in one site and exclude
her in another.

### Step 4 — Assign the stage group under a named edition

**Input.** T, N, M, prefix, edition, and any site-specific prognostic factors the
edition includes.

**Output.** A stage group plus a derivation record.

**Rationale.** The group is the variable clinicians and payers use. It must be
reproducible from the stored categories.

**Procedure.**

1. Select the edition by the date of assignment, not the date of analysis. A 2019 case
   is staged under the eighth edition even if analysed in 2026.
2. Apply the site-specific grouping table. Do not interpolate between sites.
3. For editions that include prognostic stage (breast cancer in the eighth edition adds
   biological factors), decide whether the analysis uses anatomic or prognostic stage
   and record the choice. They are different variables with different performance.
4. Validate internal consistency: T, N, M must be valid categories; the group must be
   in the edition's value set; the group must be consistent with the categories.
5. Where the edition changed mid-cohort, either restrict the cohort to one edition or
   map to a common collapsed scheme (for example, I, II, III, IV) and report the
   mapping.

**Decision table for stage assignment when the inputs are incomplete.** More than two
outputs exist, so the decision must be explicit.

| Situation | Action | Output value | Consequence if mishandled |
|---|---|---|---|
| T, N, M all present, edition known | Apply grouping table | Exact group | None |
| M missing but imaging shows no distant disease and follow-up exceeds 12 months | Assume M0 with a documented rule | Group with `M0 assumed` flag | Understaging if occult metastases exist |
| M missing and follow-up short | Do not assume; assign unknown | `unknown` | Complete-case analysis biases the cohort |
| T missing but N and M present | Cannot derive group | `unknown` with reason | Imputation invents stage |
| Clinical and pathologic both present, question is prognosis after surgery | Use pTNM | Pathologic group | Mixing prefixes biases survival |
| Clinical and pathologic both present, question is eligibility for neoadjuvant therapy | Use cTNM | Clinical group | Using pTNM leaks the future |
| Edition changed mid-cohort, both editions available | Restrict or collapse | Group plus edition field | Pooling creates a category that never existed |
| Post-neoadjuvant case | Use ypTNM | yp group | Pooling with pTNM mixes treated biology |

**Edge case.** Unknown stage is not missing at random. It is more common in older
patients, in patients who did not have surgery, and in patients diagnosed at death.
Complete-case analysis on stage therefore selects a healthier, more treated population.
Report the unknown fraction by subgroup and use multiple imputation with a sensitivity
analysis, or model the stage-unknown process explicitly.

### Step 5 — Derive performance status and fitness

**Input.** Clinic notes, oncology assessments, anaesthesia records.

**Output.** ECOG performance status at the decision date, comorbidity count, and organ
function.

**Rationale.** Performance status is the strongest single clinical predictor of
treatment tolerance and survival in most advanced-disease settings, and it is a
eligibility criterion for many trials.

**Procedure.**

1. Prefer a structured field. If absent, extract from text with an explicit rule and
   record the source sentence.
2. Take the value closest before the decision date, not the value closest to diagnosis.
3. Do not carry forward across lines. Performance status changes with progression and
   toxicity.
4. Record missingness. A patient with no recorded performance status is not ECOG 0.

**Branch point.** If performance status is missing for more than a small fraction,
consider a sensitivity analysis using a proxy (for example, recent hospitalisation or
oxygen use). Label the proxy as a proxy.

**Edge case.** Discordance between clinician-recorded performance status and functional
data. Wearable step counts and claim-derived activity often disagree with the clinic
note; neither is the truth, but the disagreement should be reported rather than
silently resolved.

### Step 6 — Reconstruct lines of therapy

**Input.** Medication administration records, prescription fills, procedure codes,
clinic notes.

**Output.** An ordered table of regimens with start date, end date, reason for
discontinuation, and line number.

**Rationale.** Comparative effectiveness in advanced cancer is almost always
line-specific. A second-line comparison that accidentally includes third-line patients
is not the comparison the label describes.

**Procedure.**

1. Define a regimen as a set of agents started within a window (commonly 28 days,
   institution-defined). Record the window.
2. Merge administration records into regimen episodes. Handle dose delays and
   reductions without starting a new line.
3. Define a line change: addition of a new agent, switch of a core agent, or a
   treatment holiday beyond a threshold. Record each rule.
4. Distinguish maintenance from a new line. Maintenance after platinum doublet is
   usually the same line.
5. Record discontinuation reason: progression, toxicity, patient choice, death, or
   administrative.
6. Validate against the clinical narrative for a random sample of 50 patients.

**Branch point.** Claims-only data cannot see oral agents reliably and cannot see
inpatient administration in all settings. If the data are claims, state the coverage
assumption and test sensitivity to oral versus intravenous regimens.

**Edge case.** Coding shifts between years. A regimen that changes its billing code, or
a biosimilar that appears under a different code, can look like a line change. Mapping
must be to ingredient, not to billing code.

### Step 7 — Define and compute response

**Input.** Baseline and follow-up imaging measurements, target lesion selection, and
assessment dates.

**Output.** Timepoint response, best overall response, and date of progression.

**Rationale.** Response is the earliest endpoint and the one most often computed
incorrectly. RECIST 1.1 is explicit about which lesions count, which reference point
applies, and what makes progression.

**Procedure.**

1. At baseline, select measurable lesions. Non-nodal lesions must be at least 10 mm in
   longest diameter on CT with slice thickness 5 mm or less; nodal lesions must be at
   least 15 mm in short axis. Record the measurement axis.
2. Select target lesions: up to 5 total and at most 2 per organ. Nodes count as one
   organ for this limit. Choose the largest, most reproducible lesions.
3. Compute the baseline SLD as the sum of the selected measurements.
4. At each follow-up, measure the same target lesions with the same method and compute
   the SLD.
5. Classify the timepoint using the rules in order: new lesion means PD; otherwise
   compare to nadir for PD; otherwise compare to baseline for PR; otherwise SD.
6. Track the running nadir, including baseline.
7. For trials using immune checkpoint inhibitors, apply iRECIST separately: an initial
   iUPD requires confirmation at least 4 weeks later before iCPD.
8. Record the assessment date and the imaging modality. For PFS, the progression date
   is the date of the assessment that first met criteria, not the date the report was
   signed.

**Branch point.** If a target lesion becomes too small to measure, RECIST 1.1 assigns a
default of 5 mm (0 mm if the lesion is believed to have disappeared). This default is a
convention, not a measurement, and it should be flagged.

**Edge case.** New lesions. A new lesion is PD regardless of SLD. But a new lesion
identified on a scan that also shows shrinkage is still PD under RECIST 1.1, and
analysts who "wait for confirmation" in a non-iRECIST trial are applying the wrong
rule. Conversely, a lesion that appears and then disappears may be artefact; the
iRECIST confirmatory rule exists for this reason.

### Step 8 — Define endpoints and censoring

**Input.** Time zero, event dates, last assessment dates, death dates.

**Output.** One row per patient per endpoint with time, event indicator, and the rule
used.

**Rationale.** The endpoint is the contract between the analysis and the decision. A
PFS estimate whose censoring rule differs between arms is not a PFS estimate.

**Procedure.**

1. Write the endpoint definition in one sentence, including time zero, event, censoring,
   and the handling of new anticancer therapy.
2. For OS, use death from any cause and censor at last known alive. Prefer a death
   registry over EHR status.
3. For PFS, use progression or death and censor at the last adequate assessment. Define
   "adequate" as a scan within a stated interval.
4. Decide how subsequent therapy is handled. Common conventions censor at the start of
   a new therapy or count it as an event; the choice must be pre-specified and applied
   to both arms.
5. For rwPFS, define progression from structured data and note the inference rule:
   treatment discontinuation plus a subsequent line, a new lesion in the radiology
   report, or a coded progression. Validate against chart review.
6. Record the number censored and the reasons.

**Decision table for ambiguous progression dates.**

| Evidence pattern | rwPFS treatment | Rationale | Risk |
|---|---|---|---|
| Radiographic progression dated in a report | Event at scan date | Direct evidence | Report date may lag |
| Treatment discontinuation for "progression" with no scan | Event at discontinuation date | Clinical decision implies progression | Circular if discontinuation was toxicity |
| New therapy started without documented progression | Event at start of new therapy | Proxy for progression | Administrative switch misclassified |
| Death without prior progression | Event at death | Standard PFS convention | None if death date reliable |
| No scan for 9 months, then death | Event at death, or censor at last scan | Choice changes estimate | Censoring assumes no progression |
| Scan shows new lesion but clinician continues therapy | Event at scan date | RECIST rule is objective | Real-world practice may differ |

**Edge case.** Informative censoring. Patients who stop scanning because they are doing
well are censored early and look event-free; patients who stop because they moved to
hospice are also censored. The two processes point in opposite directions and do not
cancel.

### Step 9 — Audit the assembled dataset

**Input.** The analytic table.

**Output.** An audit report with counts, cross-tabs, and a list of resolved and
unresolved discrepancies.

**Rationale.** Every step above can fail silently. The audit is the only defence.

**Procedure.**

1. Recompute the stage group from T, N, M and compare with the stored group. Report the
   disagreement rate.
2. Cross-tabulate diagnosis year against stage edition. Any cell with two editions is a
   pooling risk.
3. Cross-tabulate cTNM against pTNM. A high disagreement rate is expected; an
   implausibly low rate suggests one field was copied into the other.
4. Verify that treatment start dates follow diagnosis dates. Negative intervals are
   data errors or prevalent cases.
5. Verify that no feature timestamp is after the decision date. Sample 100 rows and
   inspect.
6. Recompute RECIST for 30 patients by hand and compare with the pipeline output.
7. Count missingness for stage, histology, biomarker, performance status, and follow-up
   by subgroup.
8. Produce a cohort flow diagram with every exclusion and its count.

**Branch point.** If the recomputation disagreement rate exceeds a pre-set threshold
(for example, 5 percent), stop and fix the derivation before any modelling. A model
built on a 20 percent stage disagreement is measuring the disagreement.

**Edge case.** Duplicate records from multiple source systems. A patient treated at two
hospitals in the same system can have two diagnosis records and two stage records.
Deduplicate on a stable identifier and a date rule, and record which record won.

### Step 10 — Freeze, version, and document

**Input.** The audited dataset and the code that built it.

**Output.** A versioned extract with a data dictionary, a changelog, and a
reproducibility statement.

**Rationale.** Oncology definitions change. A dataset without a version cannot be
compared with the next extract, and a result without a version cannot be reproduced
after the next guideline release.

**Procedure.**

1. Pin the source extract date, the staging edition, the WHO classification edition, the
   response criteria version, and the code commit.
2. Write a data dictionary with every derived field, its rule, and its source field.
3. Record every exclusion with a count and a reason.
4. Store the audit outputs with the dataset.
5. State the intended use and the known limits in a model card or dataset card.

**Edge case.** A guideline update arrives mid-project. Freeze the analysis on the
edition in force at the extract date and run a sensitivity analysis under the new
edition if the site was replaced. Do not silently restage.

## 6. Worked example

A 58-year-old woman presents with a persistent cough. CT shows a 32 mm mass in the right
upper lobe, an 18 mm lesion in the right lower lobe, and enlarged mediastinal nodes.
PET-CT shows uptake in the right upper lobe mass and in stations 4R and 7, with no
distant uptake. EBUS confirms adenocarcinoma in station 4R. She is a former smoker with
ECOG performance status 1.

### 6.1 Staging

The clinical categories are cT2a (tumour 32 mm, no chest wall or mediastinal invasion),
cN2 (ipsilateral mediastinal nodal disease, station 4R), and cM0. Under the AJCC eighth
edition lung table, T2a N2 M0 groups to stage IIIA. Under the ninth edition, the lung
protocol released in 2025 and effective 1 January 2026 reorganized some categories; the
case would need re-staging under that protocol for a 2026 diagnosis. The analysis below
uses the eighth edition because the case is a 2024 diagnosis, and it records the edition
explicitly, which is the point.

The baseline sum of diameters for RECIST requires lesion selection first. The two lung
lesions are in one organ, so at most two of them may be targets. The station 4R node
measures 22 mm in short axis and is measurable. The station 7 node measures 14 mm in
short axis, which is below the 15 mm threshold; it is pathological but non-target.

| Lesion | Organ | Axis | Baseline (mm) | Target? |
|---|---|---|---:|---|
| Right upper lobe mass | Lung | Longest diameter | 32 | Yes |
| Right lower lobe lesion | Lung | Longest diameter | 18 | Yes |
| Station 4R node | Node | Short axis | 22 | Yes |
| Station 7 node | Node | Short axis | 14 | No, non-target |

Targets selected: three lesions, within the limit of five total and two per organ
(nodes count as one organ). Baseline SLD is:

```text
32 + 18 + 22 = 72 mm
```

### 6.2 Treatment and response series

She receives concurrent chemoradiotherapy, then durvalumab consolidation. Imaging is
performed every eight weeks.

| Timepoint | RUL mass | RLL lesion | 4R node | SLD | Change vs baseline | Change vs nadir | Response |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 32 | 18 | 22 | 72 | — | — | — |
| Week 8 | 26 | 16 | 19 | 61 | −15.3% | −15.3% | SD |
| Week 16 | 21 | 14 | 15 | 50 | −30.6% | −18.0% | PR |
| Week 24 | 25 | 16 | 17 | 58 | −19.4% | +16.0% | SD |
| Week 32 | 30 | 19 | 20 | 69 | −4.2% | +38.0% | PD |

Arithmetic for the two reference points:

```text
Week 16 vs baseline: (50 - 72) / 72 = -0.3056 -> -30.6%, meets PR (<= -30%)
Week 16 vs nadir 61: (50 - 61) / 61 = -0.1803 -> -18.0%, not PD
Week 24 vs nadir 50: (58 - 50) / 50 = +0.1600 -> +16.0%, below 20%, so SD
Week 32 vs nadir 50: (69 - 50) / 50 = +0.3800 -> +38.0% and +19 mm -> PD
Week 32 vs baseline: (69 - 72) / 72 = -0.0417 -> -4.2%, still below baseline
```

Best overall response is PR, first observed at week 16. PFS is measured from the start
of treatment to the week 32 assessment, which is 224 days, or 32 weeks. If the analysis
instead dates progression at the report signature, which was four days after the scan,
PFS becomes 228 days. The four-day difference is small here and can be large when
reporting is delayed.

### 6.3 Two diagnostics that reveal a broken analysis

The first diagnostic asks which reference point the pipeline used for progression. A
common implementation compares each scan with baseline, because that is the simpler
join. Under that implementation, the week 32 SLD of 69 mm is 4.2 percent below the
baseline of 72 mm, so the pipeline returns stable disease and the patient continues
therapy. The correct comparison with the on-study nadir of 50 mm returns a 38 percent
increase and 19 mm absolute increase, which is progressive disease. The broken
analysis does not merely mislabel one timepoint; it delays discontinuation and biases
PFS upward by an unknown amount. The diagnostic is to recompute every timepoint under
both reference points and count the discordant calls. Any discordance is a bug, not a
clinical nuance.

The second diagnostic asks which lesions entered the sum. RECIST 1.1 excludes the 14 mm
station 7 node because it is below the 15 mm short-axis threshold for measurability.
Suppose the pipeline includes every lesion with a recorded measurement, a natural
choice when the data arrive as a measurement table. The baseline sum becomes 86 mm and
the week 16 sum becomes 64 mm:

```text
Including the non-target node: (64 - 86) / 86 = -0.2558 -> -25.6%
Correct target-only sum:       (50 - 72) / 72 = -0.3056 -> -30.6%
```

The broken analysis fails to call a partial response at week 16. In a trial with ORR as
the primary endpoint, that single inclusion rule can move the reported response rate by
several percentage points, and the direction depends on how many non-target lesions
carry measurements.

Both diagnostics are cheap: recompute under an alternative implementation and count the
discordance. Neither requires new data. Neither is a modelling question. This is the
pattern the module rewards: find the definition that, if changed, changes the answer,
then show the size of the change.

### 6.4 Vary one assumption

The station 4R node measured 22 mm in short axis. Nodal measurement is the least
reproducible part of RECIST, and the 15 mm measurability threshold is a cliff. Suppose
the node had measured 14 mm rather than 22 mm, which is within inter-reader variation
for a borderline node. It would no longer be a target lesion, and the baseline sum
would be 50 mm rather than 72 mm.

| Timepoint | Target-only SLD without 4R | Change vs baseline 50 | Response |
|---|---:|---:|---|
| Baseline | 50 | — | — |
| Week 8 | 42 | −16.0% | SD |
| Week 16 | 35 | −30.0% | PR, exactly at threshold |
| Week 24 | 44 | −12.0% | SD |
| Week 32 | 54 | +8.0% | not PD by SLD, but see below |

The week 16 call becomes a partial response exactly at the −30.0 percent boundary. A
one-millimetre change in a single nodal measurement moves the baseline by 22 mm and
changes whether the patient is a responder at all. The clinical conclusion, "this
patient had a partial response and then progressed", survives, but the quantitative
claim, "ORR was 44 percent in this cohort", does not survive a measurement rule that
excludes borderline nodes.

This is the general lesson for oncology analytics. The endpoint is not a property of the
patient alone; it is a property of the patient, the criteria version, the lesion
selection, and the measurement convention. Report all four or the number is not
interpretable.

### 6.5 What the case does not establish

The case is a measurement worked example, not a treatment claim. It does not show that
durvalumab consolidation was effective, because there is no comparator and no
randomisation. It does not establish that the patient responded to any particular agent,
because the response assessment follows multimodality therapy. It does not generalise
to the ninth edition, to small-cell histology, or to a different response criteria
version.

What it does establish is narrower and more useful: the pipeline's output depends on
choices that are not in the data. The table below separates the choices.

| Choice | Value used here | Alternative that changes the answer | How to disclose it |
|---|---|---|---|
| Staging edition | AJCC eighth | Ninth edition lung protocol | Edition field per case |
| Stage prefix | Clinical | Pathologic after resection | Prefix field |
| Nodal measurability | 15 mm short axis | Any visible node | Measurement axis and threshold |
| Target limit | 5 total, 2 per organ | All measurable lesions | Target selection log |
| Progression reference | On-study nadir | Baseline | Criteria version plus code |
| Progression date | Scan date | Report signature date | Date source field |
| Response confirmation | None required | Confirm at 4 weeks | Confirmation flag |

An analysis that discloses these seven choices can be reproduced and criticised. An
analysis that discloses none of them produces a number that cannot be checked, which is
the definition of an unfalsifiable claim.

A second limitation is that the case has one reader. Inter-reader variability in
oncology imaging is substantial, and a single measurement set overstates the precision
of every threshold call. The honest presentation of a borderline call is a
range, not a point: "SLD 50 mm, with a reader-to-reader range of 46 to 54 mm, which
straddles the partial response threshold." That sentence is more useful to a tumour
board than a binary label, and it is the form a reviewer will ask for.

## 7. Code, query, or calculation

The two blocks below are standard-library Python. The first implements RECIST 1.1
timepoint evaluation, including target selection and the two reference points. The
second implements AJCC eighth edition lung stage grouping and checks that every T/N
combination maps to a group. Both are runnable as written.

### 7.1 RECIST 1.1 timepoint evaluator

```python
"""RECIST 1.1 timepoint evaluator. Standard library only."""

from dataclasses import dataclass


@dataclass
class Lesion:
    lid: str
    organ: str
    nodal: bool
    baseline: float          # mm; longest diameter, or short axis if nodal
    current: float
    new: bool = False


def measurable(l: Lesion) -> bool:
    return l.baseline >= (15.0 if l.nodal else 10.0)


def select_targets(lesions):
    """Up to 5 targets, at most 2 per organ, largest first, measurable only."""
    chosen, per_organ = [], {}
    for l in sorted((x for x in lesions if measurable(x)),
                    key=lambda x: -x.baseline):
        if len(chosen) == 5:
            break
        if per_organ.get(l.organ, 0) == 2:
            continue
        chosen.append(l)
        per_organ[l.organ] = per_organ.get(l.organ, 0) + 1
    return chosen


def timepoint_response(targets, nadir, new_lesion=False):
    sld = sum(t.current for t in targets)
    base = sum(t.baseline for t in targets)
    pct_base = 100.0 * (sld - base) / base
    pct_nadir = 100.0 * (sld - nadir) / nadir
    delta_nadir = sld - nadir
    if new_lesion:
        return "PD", sld, pct_base, pct_nadir, "new lesion"
    if all(t.current < (10.0 if t.nodal else 0.1) for t in targets):
        return "CR", sld, pct_base, pct_nadir, "all targets resolved"
    if pct_nadir >= 20.0 and delta_nadir >= 5.0:
        return "PD", sld, pct_base, pct_nadir, ">=20% and >=5 mm vs nadir"
    if pct_base <= -30.0:
        return "PR", sld, pct_base, pct_nadir, ">=30% decrease vs baseline"
    return "SD", sld, pct_base, pct_nadir, "neither PR nor PD"


base = [Lesion("L1", "lung", False, 32, 32), Lesion("L2", "lung", False, 18, 18),
        Lesion("N1", "node", True, 22, 22), Lesion("N2", "node", True, 14, 14)]
targets = select_targets(base)
print("targets:", [(t.lid, t.baseline) for t in targets])
print("baseline SLD:", sum(t.baseline for t in targets))

scans = {
    "week 8":  {"L1": 26, "L2": 16, "N1": 19, "N2": 12},
    "week 16": {"L1": 21, "L2": 14, "N1": 15, "N2": 10},
    "week 24": {"L1": 25, "L2": 16, "N1": 17, "N2": 11},
    "week 32": {"L1": 30, "L2": 19, "N1": 20, "N2": 12},
}
nadir = sum(t.baseline for t in targets)
for tp, m in scans.items():
    for t in targets:
        t.current = m[t.lid]
    r, sld, pb, pn, why = timepoint_response(targets, nadir)
    print(f"{tp}: SLD={sld:.0f} vs baseline={pb:+.1f}% "
          f"vs nadir={pn:+.1f}% -> {r} ({why})")
    nadir = min(nadir, sld)
```

Expected output:

```text
targets: [('L1', 32), ('N1', 22), ('L2', 18)]
baseline SLD: 72
week 8: SLD=61 vs baseline=-15.3% vs nadir=-15.3% -> SD (neither PR nor PD)
week 16: SLD=50 vs baseline=-30.6% vs nadir=-18.0% -> PR (>=30% decrease vs baseline)
week 24: SLD=58 vs baseline=-19.4% vs nadir=+16.0% -> SD (neither PR nor PD)
week 32: SLD=69 vs baseline=-4.2% vs nadir=+38.0% -> PD (>=20% and >=5 mm vs nadir)
```

Note that the nadir is updated after each assessment, so the week 16 comparison uses
the week 8 value of 61 mm. That ordering is the part most implementations get wrong.
If the nadir is updated before the comparison, the week 16 call is still PR but the
reported nadir change is wrong, and the error compounds at later timepoints.

### 7.2 AJCC eighth edition lung stage grouping

```python
"""Lung cancer stage grouping, AJCC 8th edition (NSCLC)."""

from itertools import product

T = ["T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]
N = ["N0", "N1", "N2", "N3"]

GROUPS = {
    ("T1a", "N0"): "IA1", ("T1b", "N0"): "IA2", ("T1c", "N0"): "IA3",
    ("T2a", "N0"): "IB",  ("T2b", "N0"): "IIA",
    ("T1a", "N1"): "IIB", ("T1b", "N1"): "IIB", ("T1c", "N1"): "IIB",
    ("T2a", "N1"): "IIB", ("T2b", "N1"): "IIB", ("T3", "N0"): "IIB",
    ("T1a", "N2"): "IIIA", ("T1b", "N2"): "IIIA", ("T1c", "N2"): "IIIA",
    ("T2a", "N2"): "IIIA", ("T2b", "N2"): "IIIA",
    ("T3", "N1"): "IIIA", ("T4", "N0"): "IIIA",
    ("T1a", "N3"): "IIIB", ("T1b", "N3"): "IIIB", ("T1c", "N3"): "IIIB",
    ("T2a", "N3"): "IIIB", ("T2b", "N3"): "IIIB",
    ("T3", "N2"): "IIIB", ("T4", "N1"): "IIIB", ("T4", "N2"): "IIIB",
    ("T3", "N3"): "IIIC", ("T4", "N3"): "IIIC",
}
M = {"M0": "I-IV", "M1a": "IVA", "M1b": "IVA", "M1c": "IVB"}


def stage_group(t, n, m):
    if m != "M0":
        return M[m]
    return GROUPS.get((t, n), "unmapped")


print("T2a N2 M0 ->", stage_group("T2a", "N2", "M0"))
print("T4 N3 M0 ->", stage_group("T4", "N3", "M0"))
print("mapped:", sum(1 for t, n in product(T, N) if (t, n) in GROUPS),
      "of", len(T) * len(N))

# staged discrepancy: clinical vs post-neoadjuvant pathologic
print("cT2aN2M0:", stage_group("T2a", "N2", "M0"),
      "| ypT1bN0M0:", stage_group("T1b", "N0", "M0"))
```

Expected output:

```text
T2a N2 M0 -> IIIA
T4 N3 M0 -> IIIC
mapped: 28 of 28
cT2aN2M0: IIIA | ypT1bN0M0: IA2
```

### 7.3 What the code does not do

The RECIST function does not handle non-target lesion progression, which requires the
"unequivocal progression" judgement and is not computable from a diameter table. It does
not handle the 5 mm default for "too small to measure" or the 0 mm convention for
disappeared lesions, both of which must be applied before the sum is taken. It does not
implement iRECIST confirmation. It assumes the target set has already been validated
against the baseline imaging report.

The staging function covers only the T/N/M combinations of the eighth edition lung
table. It does not cover small-cell lung cancer, the ninth edition, or stage 0. It is a
teaching implementation, not a staging authority.

## 8. Data, standards, and regulatory context

### 8.1 Staging standards

AJCC and UICC publish the TNM classification jointly in content but separately in
edition mechanics. AJCC moved to a rolling Version 9 model in which disease-site
protocols are released in batches and take effect on 1 January of the following year.
The practical release history relevant to a 2024–2026 cohort is: cervix (2021), anus,
appendix, brain and spinal cord (2023), neuroendocrine sites and vulva (2024), lung,
thymus, pleural mesothelioma and nasopharynx (2025, effective 1 January 2026), and
salivary gland and HPV-associated oropharynx (effective 1 January 2026). All eighth
edition sites not yet replaced remain current. UICC published TNM ninth edition in July
2025, effective 1 January 2026, and incorporates the corresponding AJCC changes.

What this constrains: any analysis spanning a release date must record the edition per
case and must not pool stage groups across editions without a mapping. What it does not
constrain: the choice of clinical versus pathologic stage, which is a study-design
decision.

### 8.2 Response standards

RECIST 1.1 (Eisenhauer et al., European Journal of Cancer, 2009) is the reference for
solid tumour response in most adult trials. It defines measurability, target selection,
the sum of diameters, the four timepoint categories, and confirmation requirements. The
RECIST committee published an update and clarification in 2016 covering lymph node
counting, the 5 mm default, and PET-CT caveats. iRECIST (Seymour et al., 2017) adapts
the framework for immune checkpoint inhibitors by requiring confirmation of progression.

What this constrains: response is a rule-based classification, and any pipeline that
reports response must implement the rule, not a proxy. What it does not constrain:
real-world progression, which usually has no protocol-specified assessment schedule and
must be inferred.

### 8.3 Histology and molecular standards

The WHO Classification of Tumours series defines entity boundaries; the fifth edition
volumes have been released site by site since 2019. ICD-O-3.2 provides the morphology
codes used by registries. Molecular reclassification is now embedded in WHO entities
for several sites, which means the histology field and the molecular field are no longer
independent.

What this constrains: entity definitions change, and a cohort spanning a WHO edition
change may contain cases that would be classified differently today. What it does not
constrain: the decision to analyse by molecular subgroup, which is a design choice.

### 8.4 Interoperability standards

HL7 FHIR mCODE (minimal Common Oncology Data Elements) Implementation Guide, STU 4,
version 4.0.0, published 16 February 2025, defines profiles for cancer patient, primary
cancer condition, TNM categories, stage group, cancer staging, tumour marker, genomics,
and treatment. The TNM Stage Group profile is an Observation with LOINC code 21908-9,
21902-2, or 21914-7 depending on staging type, a required `focus` on the primary cancer
condition, a required `method` naming the staging system, and up to three `hasMember`
references to the T, N, and M category observations. That structure is exactly the
"store the inputs, not only the group" rule from Step 3, expressed in FHIR.

OMOP has an oncology extension with episodes, tumour episodes, and cancer modifiers.
CDISC SDTM has tumour identification, tumour assessment, and response domains (TR, RS),
and ADaM has time-to-event structures. NAACCR defines registry fields. None of these is
a legal requirement in the United States; they are conventions that purchasers,
registries, and sponsors impose contractually.

### 8.5 Regulatory context

A model that selects therapy or reports a biomarker used for selection may be a medical
device, an in-vitro diagnostic, or a companion diagnostic. In the United States, the
FDA's device pathways are 510(k), De Novo, and premarket approval; laboratory-developed
tests have been offered under CLIA, and the FDA's 2024 LDT rule was vacated in
litigation in 2025, so the enforcement position must be confirmed rather than assumed.
In the European Union, an in-vitro diagnostic falls under Regulation (EU) 2017/746
(IVDR), and an AI component may also engage Regulation (EU) 2024/1689 (the AI Act).

Real-world evidence for oncology is addressed by FDA guidance on registries (final,
December 2023), externally controlled trials (draft, February 2023), and the use of
real-world data and evidence (final, August 2023). The Oncology QCARD initiative defines
minimum data elements for oncology real-world study proposals. These documents
constrain what a sponsor must demonstrate; they do not validate a particular dataset.

> **Note:** Regulatory status changes. The LDT rule, the AI Act application dates, and
> the IVDR transition schedule have all moved. Confirm the current position with
> counsel before a submission or a commercial claim. Standards make data exchangeable,
> not correct: mCODE can carry a stage group with the wrong edition, and no conformance
> statement substitutes for the audit in Step 9.

## 9. Industry practice and case evidence

The cases below are documented public events, not composites. Each is followed by what
transferred and what did not, because the transferable part is usually the evaluation
design rather than the result.

### 9.1 Tissue-agnostic approval and the biomarker-defined cohort

In May 2017 the FDA granted accelerated approval to pembrolizumab for adult and
pediatric patients with unresectable or metastatic, microsatellite instability-high
(MSI-H) or mismatch repair deficient (dMMR) solid tumours, regardless of tumour site.
The evidence base was a basket of small single-arm trials with objective response rate
and duration of response as endpoints, not overall survival.

Documented fact: the approval crossed histology on the basis of a biomarker. Inference:
the case demonstrated that a biomarker-defined eligibility rule can be regulated as a
label rather than as a site-specific indication. What transferred is the idea that the
cohort definition, not the tumour site, can be the product. What did not transfer is
the evidentiary standard: accelerated approval carries a confirmatory obligation, and
response in one MSI-H tumour does not imply response in every MSI-H tumour.

### 9.2 A threshold, an assay, and a label as one package

In June 2020 the FDA approved pembrolizumab for adults and children with
unresectable or metastatic tumour mutational burden-high (TMB-H, defined as at least
10 mutations per megabase) solid tumours, using FoundationOne CDx as the companion
diagnostic. The threshold is assay-specific: TMB measured on one panel is not
numerically comparable with TMB measured on another, because panels cover different
genomic territory and use different filtering.

Documented fact: the approval named both a cut point and a device. Inference: the
commercial unit is the triplet of drug, assay, and threshold, and a data product that
computes TMB without binding itself to a validated assay is not the same product. The
analytic consequence is that a cohort assembled from multiple panels cannot pool TMB
values without harmonisation, and harmonisation is an approximation.

### 9.3 De-escalation with a circulating tumour DNA assay

The DYNAMIC trial (Tie et al., New England Journal of Medicine, 2022) randomised
patients with resected stage II colon cancer to circulating tumour DNA (ctDNA)-guided
adjuvant management or standard clinicopathological management. The ctDNA-guided arm
received less adjuvant chemotherapy, and recurrence-free survival in the two arms was
not significantly different.

Documented fact: a biomarker-guided strategy reduced treatment exposure without a
detected recurrence-free survival penalty in that population. Inference: the result
supports de-escalation under a non-inferiority logic, not escalation. The trial does not
establish that treating ctDNA-positive patients improves outcomes; that is a different
question requiring a different design. This distinction is the most commonly misread
result in the oncology literature, and it is a definition problem before it is a
statistical one.

### 9.4 Screening accuracy is not mortality benefit

The MASAI trial (Lång et al., Lancet Oncology, 2023) randomised about 80,000 Swedish
women to artificial-intelligence-supported screen reading or standard double reading in
a population-based mammography programme. AI-supported reading increased cancer
detection without increasing recall.

Documented fact: detection rate rose at acceptable recall. Inference: the trial measured
screening performance, not mortality. A detection increase can reflect earlier
diagnosis, overdiagnosis, or genuine benefit, and separating them requires long
follow-up with mortality as the endpoint. Any vendor claim that cites MASAI as evidence
of mortality reduction has substituted an endpoint.

### 9.5 What regulators now expect from real-world oncology evidence

The FDA's registry guidance (final, December 2023), externally controlled trial guidance
(draft, February 2023), and real-world evidence guidance (final, August 2023) set out
relevance and reliability criteria: whether the data contain the variables that matter,
whether the patients are representative, and whether the data are accurate, complete,
and traceable. The Oncology QCARD initiative defines minimum data elements for oncology
real-world study proposals. The practical implication for a data team is that a
real-world oncology dataset must be able to document lineage for every variable used in
an endpoint, not just for the exposure.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed staging editions | Stage distribution shifts abruptly at a calendar year | Cohorts pool cases staged under different rulebooks | Stage-specific survival is not comparable | Store edition per case; restrict or map |
| cTNM and pTNM conflated | cTNM equals pTNM for nearly every surgical case | One field copied into the other during abstraction | Cohort silently restricted to surgical patients | Store both; choose by question |
| Time zero drift | Median survival differs from the published value by a wide margin | Index date varies by patient | Survival and PFS are not comparable across groups | One time-zero rule; audit timestamps |
| Immortal time | Exposed group has zero deaths in the first months | Treatment must be survived to be received | Spuriously protective hazard ratio | Landmark or time-dependent exposure |
| PD computed against baseline | Fewer progressions than the imaging narrative implies | Wrong reference point in code | Patients stay on ineffective therapy; PFS inflated | Compare to running nadir; unit-test |
| Non-target lesions summed | Baseline SLD higher than the sum of reported targets | Measurement table joined without the target flag | Response misclassified near thresholds | Enforce target selection before summation |
| Node measured on long axis | Nodal measurements larger than the radiology report | Axis confusion in extraction | Measurability and SLD both wrong | Record axis; validate against report |
| Biomarker missing treated as negative | Biomarker-negative fraction equals the untested fraction | Missing coded as a category | Eligibility and effect estimates biased | Model testing; never impute negative |
| rwPFS unvalidated | rwPFS longer than PFS in the same regimen | Discontinuation used as progression proxy | Comparative claim not reproducible | Chart-validate a sample; publish the rule |
| Lead-time bias in screening | Survival improves with no mortality change | Earlier diagnosis moves the start date | Screening appears effective when it is not | Report mortality and interval cancers |
| Stage migration | Stage-specific survival improves in every stage at once | More sensitive imaging reassigns patients | Progress attributed to treatment | Report stage distribution and overall survival |
| Overdiagnosis | Incidence rises faster than mortality falls | Detection of indolent disease | Harm and cost without benefit | Report incidence-to-mortality ratio over time |
| Informative censoring | Censoring rate differs by arm | Scan schedules and follow-up differ | PFS comparison biased | Pre-specify assessment windows; sensitivity analysis |
| Line-of-therapy drift | Second-line cohort contains third-line patients | Regimen reconstruction rules differ | Comparison is against the wrong label population | Validate regimen logic against notes |

Two red flags deserve emphasis because they are cheap to check and almost always
indicate a real problem.

The first is a stage distribution that is too clean. Real cohorts have a substantial
unknown-stage fraction, and it is concentrated in older and less-treated patients. A
cohort with 1 percent unknown stage has either excluded those patients without saying so
or imputed them without a flag.

The second is a response rate that matches the trial exactly. Real-world response
ascertainment is noisier than trial assessment, and an exact match usually means the
real-world variable was copied from a publication or derived from a rule calibrated to
reproduce it. That is not validation; it is circularity.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The diagnosis date is the first evidence of disease | Survival and incidence need a common origin | Incidental findings and screen-detected cases have earlier true onset | Compare diagnosis date with first abnormal test date |
| Histology is stable over the cohort window | Pooling requires a consistent entity definition | WHO reclassification and molecular redefinition | Tabulate morphology by year; review boundary cases |
| Stage was assigned under the edition in force at diagnosis | Edition mixing invalidates pooling | Late abstraction under a newer manual | Cross-tab year by edition |
| Clinical stage is available for all included patients | Otherwise the cohort is surgical-only | Inoperable patients lack pTNM, not cTNM | Count missing by treatment group |
| RECIST target selection is reproducible | SLD depends on which lesions are chosen | Reader variation at the 10 mm and 15 mm thresholds | Re-read a sample; measure disagreement |
| Progression date equals assessment date | PFS needs a date | Reporting lag and retrospective dating | Compare scan date with report date |
| Censoring is non-informative | Survival estimators require it | Scan stopping related to prognosis | Sensitivity analysis under worst-case censoring |
| Subsequent therapy is comparable across groups | Otherwise the comparison mixes sequences | Access differences and crossover | Report time to next treatment by group |
| Biomarker missingness is unrelated to prognosis given covariates | Complete-case analysis requires it | Testing depends on access and stage | Model the testing process; compare tested and untested |
| Death ascertainment is complete | OS otherwise underestimates events | No registry linkage; loss to follow-up | Compare EHR vital status with registry |
| Line of therapy is reconstructable from available data | Comparative claims are line-specific | Oral agents and outside care missing | Chart-review 50 patients |
| Performance status is measured at the decision | It is a time-varying predictor | Single baseline value carried forward | Compare values across lines |

The largest limit is not in the table. Oncology data are generated by care, not by
measurement. A patient who is not scanned has no response assessment, a patient who is
not tested has no biomarker, and a patient who is not treated has no line of therapy.
Every one of those absences is a function of the care pathway, and the care pathway is
a function of access. An oncology model therefore encodes access as if it were biology
unless the analysis explicitly separates the two.

A second limit is that the edition problem has no clean fix. Collapsing to I to IV loses
information, restricting to one edition loses sample, and restaging historical cases
requires the original source documents, which often no longer exist. The honest
approach is to report the restriction and its cost, not to pretend the pooled variable
is stable.

## 12. Exercises

### Exercise 1 — Stage a case and defend the edition (recall and application, 30 minutes)

**Deliverable.** A one-page staging record for a supplied case with T, N, M, prefix,
edition, group, and the derivation.

**Acceptance criteria.** Categories are justified from the source text; the edition is
named with its effective date; the group follows from the categories; the record states
which edition would apply if the diagnosis date moved by two years.

### Exercise 2 — Choose the stage for the question (application, 25 minutes)

**Deliverable.** A table of four questions (neoadjuvant eligibility, prognosis after
resection, trial eligibility, recurrence analysis) with the correct prefix and a
one-sentence justification for each.

**Acceptance criteria.** At least one answer uses cTNM, one uses pTNM, one uses ypTNM,
and one uses rTNM; each justification names the information available at the decision
point.

### Exercise 3 — Compute a RECIST series (application, 45 minutes)

**Deliverable.** A completed timepoint table for a five-lesion case, including target
selection, baseline SLD, per-timepoint SLD, percentage change against both reference
points, and response category.

**Acceptance criteria.** Target selection respects the 5-total and 2-per-organ limits;
the nodal lesion is measured on the short axis; PD uses both the 20 percent and 5 mm
criteria; the nadir update order is stated.

### Exercise 4 — Write three endpoint definitions (application, 35 minutes)

**Deliverable.** One-paragraph definitions of OS, PFS, and rwPFS for a stated cohort,
each naming time zero, event, censoring, and the handling of subsequent therapy.

**Acceptance criteria.** The three definitions share one time zero where appropriate;
the rwPFS definition names the inference rule and the validation sample; no definition
uses the phrase "as recorded".

### Exercise 5 — Demonstrate immortal time (critique, 50 minutes)

**Deliverable.** A short analysis on a supplied dataset comparing a naive
treated-versus-untreated hazard ratio with a 90-day landmark and a time-dependent
exposure, plus a paragraph explaining the direction and size of the change.

**Acceptance criteria.** All three estimates are reported with confidence intervals; the
landmark population is described; the write-up states which estimate answers which
question.

### Exercise 6 — Construct a stage migration example (critique, 40 minutes)

**Deliverable.** A worked example with a pre- and post-imaging-improvement stage
distribution showing stage-specific survival improving in every stage while overall
survival is unchanged.

**Acceptance criteria.** The arithmetic is shown; the example states the assumption
about how many patients move; the write-up explains why the Will Rogers phenomenon is
not evidence of treatment benefit.

### Exercise 7 — Audit biomarker missingness (design, 45 minutes)

**Deliverable.** A missingness audit plan and a table of testing rates by stage, age,
insurance, and site, with a proposed sensitivity analysis.

**Acceptance criteria.** The plan distinguishes missing at random from missing because
untested; the sensitivity analysis specifies the assumption and the reporting; no step
imputes "negative".

### Exercise 8 — Map the regulatory path (design, 30 minutes)

**Deliverable.** A one-page memo mapping a described oncology product to a regulatory
pathway, naming the intended use, the evidence required, and the precedent product.

**Acceptance criteria.** The memo distinguishes device, in-vitro diagnostic, companion
diagnostic, and label change; it names at least one binding guidance document; it states
what the product does not claim.

## 13. Self-check answers

**1. T3 N1 M0 under the AJCC eighth edition lung table.** Stage IIIA. T3 with N1 groups
to IIIA, as does T1–T2 with N2. The answer is only complete with the edition named,
because the ninth edition lung protocol took effect on 1 January 2026 and may assign a
different group.

**2. Clinical versus pathologic stage.** Clinical stage uses examination, imaging, and
biopsy before treatment. Pathologic stage uses the resected specimen. They answer
different questions: cTNM supports the decision to operate or to give neoadjuvant
therapy; pTNM supports prognosis after resection. Using pTNM in a cohort that includes
inoperable patients excludes exactly the patients with the worst prognosis.

**3. RECIST arithmetic.** Baseline targets 32, 18, and 22 mm give an SLD of 72 mm. A
follow-up sum of 50 mm is a 30.6 percent decrease from baseline, which meets the
partial response threshold of at least 30 percent. The same 50 mm is an 18.0 percent
decrease from the week 8 nadir of 61 mm, which is not progressive disease. A later sum
of 69 mm is 38.0 percent above the nadir of 50 mm with a 19 mm absolute increase, which
meets both progression criteria.

**4. Time zero for rwPFS.** Time zero is the date of initiation of the line of therapy
under study, not the date of diagnosis and not the date the regimen was decided. The
event is inferred progression or death. Censoring is at the last adequate assessment,
not the last contact. The inference rule must be stated and chart-validated.

**5. Why the naive comparison is biased.** Treatment cannot start before diagnosis, so
every treated patient has survived from diagnosis to treatment. That period is immortal
time credited to the exposure. The bias is largest when the interval from diagnosis to
treatment is long and when early deaths are common.

**6. Stage migration.** More sensitive imaging moves patients with occult metastases
from a lower stage to a higher stage. Those patients had worse prognosis than the
average patient in their original stage, so removing them improves survival in the
lower stage. They join a higher stage where they are healthier than the average, so
survival improves there too. Every stage improves while overall survival is unchanged.

**7. Why "missing equals negative" is wrong.** A patient without a biomarker result was
not tested, and testing is ordered for reasons that correlate with stage, histology,
performance status, and access. Coding those patients as negative mixes a clinical
category with an access category and biases both eligibility counts and effect
estimates.

**8. Regulatory mapping.** A model that reports a HER2 score used to select therapy is
an in-vitro diagnostic and, if the label ties the drug to the test, a companion
diagnostic. A model that flags a slide for pathologist review without making a
diagnostic claim is more likely a device with a detection-aid intended use. The
distinction is intended use, not algorithm class.

**Rubric for Exercise 6 and Exercise 8.** A strong answer states the assumption, shows
the arithmetic or the regulatory citation, names the counterfactual, and says what the
result does not establish. A weak answer asserts a conclusion without the arithmetic or
cites a regulation without its scope.

## 14. Interview questions

**1. Why is a treated-versus-untreated comparison at diagnosis usually biased?**
Strong answer: immortal time, with a concrete mechanism and at least two fixes. Weak
answer: "confounding", with no mechanism and no fix.

**2. A cohort has both cTNM and pTNM. Which do you use?**
Strong answer: it depends on the question, and the choice changes the cohort. Weak
answer: "use the pathologic one, it is more accurate", which silently restricts the
cohort to surgical patients.

**3. Your pipeline reports a 62 percent ORR. What do you check first?**
Strong answer: target selection, reference point, confirmation rule, and whether
non-target lesions entered the sum. Weak answer: "the model", which is not where the
error usually is.

**4. How would you detect immortal time in an existing analysis?**
Strong answer: plot events over time in the exposed group, check for zero events before
the median time to treatment, and compare naive with landmark estimates. Weak answer:
"adjust for time", without saying how.

**5. PFS improved but OS did not. Give three explanations.**
Strong answer: surrogate failure, crossover or subsequent therapy, and improved
post-progression survival. Weak answer: "the trial was underpowered", which is possible
but not the first explanation to test.

**6. What is stage migration and how do you detect it?**
Strong answer: the Will Rogers mechanism, detected by simultaneous improvement in all
stage-specific survival with a shift in stage distribution and no change in overall
survival. Weak answer: "patients move stage", with no consequence.

**7. How do you handle missing biomarker status?**
Strong answer: model the testing process, report testing rates by subgroup, impute with
uncertainty, and never code untested as negative. Weak answer: drop missing rows,
which selects the tested population.

**8. What is the difference between RECIST 1.1 and iRECIST?**
Strong answer: iRECIST requires confirmation of progression at least four weeks later
to account for pseudoprogression on immunotherapy, and the two systems must not be
mixed within one analysis. Weak answer: "iRECIST is for immunotherapy", without the
confirmation rule.

**9. When is a real-world external control arm credible?**
Strong answer: rarely, and only with a target trial framework, comparable eligibility,
measured confounding, and pre-specified positivity checks. Weak answer: "when the
propensity score balances", which addresses one assumption and not the design.

**10. Which patients is an oncology triage model most likely to harm?**
Strong answer: those with less testing, atypical presentation, or limited access, with
a plan to report subgroup sensitivity and alert rates. Weak answer: "no one, the model
is validated", which ignores transportability.

**11. What single field would you add to an oncology warehouse first?**
Strong answer: the staging edition and prefix, because without them the stage variable
is not interpretable. Weak answer: "more genomic features", which does not fix the
denominator.

## 15. Key terms

- **AJCC** — American Joint Committee on Cancer, publisher of the TNM staging system.
- **TNM** — tumour, node, metastasis classification.
- **Stage group** — alphanumeric summary derived from T, N, M and site-specific factors.
- **cTNM / pTNM / ypTNM / rTNM** — clinical, pathologic, post-neoadjuvant, and recurrence stage prefixes.
- **WHO classification** — entity definitions for tumours, released by site.
- **RECIST 1.1** — response evaluation criteria in solid tumours, version 1.1.
- **iRECIST** — immune-response RECIST with confirmation of progression.
- **SLD** — sum of diameters of target lesions.
- **Nadir** — smallest SLD observed on study.
- **Target lesion** — measurable lesion selected at baseline, up to five total.
- **CR / PR / SD / PD** — complete response, partial response, stable disease, progressive disease.
- **ORR** — objective response rate.
- **DoR** — duration of response.
- **OS** — overall survival.
- **PFS** — progression-free survival.
- **DFS** — disease-free survival.
- **rwPFS** — real-world progression-free survival.
- **pCR** — pathologic complete response.
- **MRD** — minimal residual disease.
- **ctDNA** — circulating tumour DNA.
- **ECOG PS** — performance status scale from 0 to 5.
- **Line of therapy** — ordered systemic regimen sequence.
- **Immortal time bias** — person-time before exposure credited to the exposed group.
- **Lead-time bias** — earlier detection lengthens measured survival without delaying death.
- **Length-time bias** — screening favours detection of slower-growing disease.
- **Stage migration** — reassignment across stages from more sensitive staging.
- **Overdiagnosis** — detection of disease that would not have caused harm.

## 16. Sources

- American Joint Committee on Cancer. AJCC Version 9 cancer staging system. https://www.facs.org/quality-programs/cancer-programs/american-joint-committee-on-cancer/version-9/ (accessed 26 October 2026).
- Union for International Cancer Control. TNM Classification of Malignant Tumours, 9th edition, updates April 2026. https://www.uicc.org/sites/default/files/2026-04/tnm-9th-updates-april-2026.pdf (accessed 26 October 2026).
- American College of Surgeons. AJCC staging: insights into new 2026 changes. https://www.facs.org/media/wlmmwiz0/ajcc-staging-insights-into-new-2026-changes-and-clarification-of-staging-issues.pdf (accessed 26 October 2026).
- Eisenhauer EA, et al. New response evaluation criteria in solid tumours: revised RECIST guideline (version 1.1). Eur J Cancer. 2009;45:228–247. https://project.eortc.org/recist/wp-content/uploads/sites/4/2015/03/RECISTGuidelines.pdf (accessed 26 October 2026).
- Schwartz LH, et al. RECIST 1.1 — update and clarification: from the RECIST Committee. Eur J Cancer. 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC5737828/ (accessed 26 October 2026).
- HL7 International. mCODE Implementation Guide v4.0.0, TNM Stage Group profile. https://www.hl7.org/fhir/us/mcode/StructureDefinition-mcode-tnm-stage-group.html (accessed 26 October 2026).
- US Food and Drug Administration. Real-world data: assessing registries to support regulatory decision-making for drug and biological products (final guidance, December 2023). https://www.fda.gov/media/154449/download (accessed 26 October 2026).
- US Food and Drug Administration. Considerations for the design and conduct of externally controlled trials for drug and biological products (draft guidance, February 2023). https://www.fda.gov/media/164960/download (accessed 26 October 2026).
- US Food and Drug Administration. Oncology QCARD initiative. https://www.fda.gov/news-events/fda-voices/facilitating-review-real-world-data-studies-oncology-qcard-initiative (accessed 26 October 2026).
- US Food and Drug Administration. FDA approves first cancer treatment for any solid tumor with a specific genetic feature (2017). https://www.fda.gov/news-events/press-announcements/fda-approves-first-cancer-treatment-any-solid-tumor-specific-genetic-feature (accessed 26 October 2026).
- US Food and Drug Administration. Pembrolizumab for TMB-H solid tumors (2020). https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-pembrolizumab-adults-and-children-tmb-h-solid-tumors (accessed 26 October 2026).
- Tie J, et al. Circulating tumor DNA analysis guiding adjuvant therapy in stage II colon cancer. N Engl J Med. 2022. https://doi.org/10.1056/NEJMoa2200075 (accessed 26 October 2026).
- Lång K, et al. Artificial intelligence-supported screen reading versus standard double reading in the MASAI trial. Lancet Oncol. 2023. https://doi.org/10.1016/S1470-2045(23)00298-X (accessed 26 October 2026).
- National Cancer Institute. Surveillance, Epidemiology, and End Results Program. https://seer.cancer.gov/ (accessed 26 October 2026).
- AACR Project GENIE. https://aacrprojectgenie.org/ (accessed 26 October 2026).
- National Comprehensive Cancer Network. NCCN clinical practice guidelines in oncology. https://www.nccn.org/guidelines (accessed 26 October 2026).

> **Access note:** All sources were checked on 26 October 2026. Guideline editions
> change; confirm the version in force at the date of your analysis before citing a
> threshold.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Next topic: [`02_cardiology.md`](02_cardiology.md)
- Related topic: [`03_pathology.md`](03_pathology.md)
- Cross-specialty comparison: [`07_cross-specialty-comparison.md`](07_cross-specialty-comparison.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Foundation material: [`../01_clinical_foundations/`](../01_clinical_foundations/)
- Omics and biomarkers: [`../12_omics/`](../12_omics/)
