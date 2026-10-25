# Clinical Genomics: Classification, Reporting, and the Counselling Interface

> Module 12 — Omics and Precision Medicine
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, 05, topics 02 and 08
> Version 2026.3

## 1. Why this topic exists

A clinical genomic test is not a data product. It is a medical service with a report, a
patient, a clinician, and a family. The technical work — sequencing, calling, annotation — is the
part that can be validated, automated, and scaled. The part that determines whether the test
helps is the interpretation and the communication, and both are labour-intensive, variable, and
hard to regulate through performance metrics.

Three problems dominate. The first is uncertainty: most rare variants are classified as variants
of uncertain significance (VUS), and a VUS is not a diagnosis. A report that lists a VUS without
explaining what it does not mean transfers the uncertainty to a clinician who has neither the
time nor the training to resolve it. The second is scope: a genome contains actionable findings
unrelated to the indication, and whether to report them is a policy decision that varies by
country and by consent. The third is equity: variant interpretation depends on reference
databases whose ancestry composition is skewed, so a variant that is rare in Europeans may be
common in a population that is under-represented, and the classification inherits that error.

The industry consequence is that clinical genomics is where the abstract problems of the previous
nine topics become individual. A batch effect is a statistical inconvenience in a discovery
study; in a clinical report it is a misdiagnosis. A VUS reported as pathogenic is a patient who
undergoes unnecessary surgery. A variant missed because of low coverage is a family that does not
get surveillance.

This topic covers the clinical genomics workflow from indication to report to counselling, the
regulatory framework, and the failure modes that matter to patients. It assumes variant
classification from topic 02 and the evidence standards from topic 08.

### 1.1 The failure pattern

1. A panel is ordered without a clear clinical question, and the report returns a VUS.
2. The VUS is communicated as "a mutation was found," and the patient interprets it as a
   diagnosis.
3. A secondary finding is returned without a consent conversation, and the patient learns about
   a cardiac risk in a results letter.
4. A negative result is reported without the coverage limitations, and the clinician interprets
   "no variant detected" as "no genetic cause."
5. A variant is classified using a frequency database that under-represents the patient's
   ancestry, and a common variant is reported as rare.
6. The report is issued, the patient moves, and no one is responsible for reanalysis when the
   classification changes.

Each of these is a communication failure as much as a technical one, and each has a documented
mitigation: a clear indication, a consent conversation, a limitations section, ancestry-aware
interpretation, and a reanalysis policy.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | A correct diagnosis, or peace of mind | Depends on the report and the conversation |
| Family members | Access to cascade testing | Depend on the proband's report |
| Clinician | Confidence in the result | Cannot see the evidence table |
| Genetic counsellor | A clear message to deliver | Must explain a classification they did not make |
| Laboratory director | Accreditation and trust | Inherits variant-level ambiguity |
| Payer | Appropriate use of testing | Sees utilization, not outcomes |
| Regulator | Assurable performance | Sees submissions, not reports |

A fourth pressure is volume. Testing has moved from single-gene assays to panels, exomes, and
genomes, and the number of variants per report has risen accordingly. A laboratory that could
review 20 variants per case now reviews hundreds, and the time per variant has not fallen
proportionally. Automation handles annotation and filtering; it does not handle the judgment that
distinguishes a reportable variant from a VUS, which is why interpretation remains the rate-
limiting step.

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

In clinical genomics, the question has a fourth part: is the result communicated in a way that
supports the decision it is meant to inform? A correct classification in an unreadable report
does not help the patient.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** analytical, interpretive, and communication uncertainty in a genomic test.
2. **Write** a report that separates finding, interpretation, action, and limitation.
3. **Apply** a policy for secondary findings consistent with consent and jurisdiction.
4. **Compute** a post-test probability from a pre-test probability and a likelihood ratio, and
   explain why classification categories are not probabilities.
5. **Explain** the VUS to a non-specialist without overstating or dismissing it.
6. **Design** a reanalysis policy with triggers, responsibilities, and a communication path.
7. **Assess** the equity implications of ancestry representation in reference data.
8. **Map** the regulatory obligations for a clinical genomic test in the US and the EU.

## 3. The core mental model

A clinical genomic result passes through four transformations, and each can fail independently.

```text
  Indication and consent
        |
        v
  Test selection (gene set, variant classes, sample)
        |
        v
  Laboratory analysis (sequencing, calling, classification)
        |
        +--> analytical validity: was the variant called correctly?
        +--> interpretive validity: does the evidence support the category?
        v
  Report (finding, interpretation, action, limitation)
        |
        +--> communication validity: can the reader act correctly?
        v
  Clinician and patient
        |
        +--> decision (treatment, surveillance, cascade testing, nothing)
        v
  Family and follow-up
        |
        +--> reanalysis when knowledge changes
```

### 3.1 Three uncertainties

**Analytical uncertainty.** The variant may not be real, or may be missed. Depth, mapping,
sample identity, and orthogonal confirmation determine this. It is measurable and should be
stated as a limit of detection.

**Interpretive uncertainty.** The variant may be real and its meaning unknown. The classification
categories encode this, imperfectly: pathogenic and benign at the extremes, VUS in the middle.

**Communication uncertainty.** The report may be correct and misunderstood. This is the least
measured and the most consequential, because it determines whether the correct interpretation
reaches the decision.

### 3.2 Categories are not probabilities

"Pathogenic" is a category defined by evidence criteria, not a posterior probability that the
patient has disease. The ClinGen Bayesian framing assigns odds to evidence strengths, but the
prior is a modelling choice and the posterior is not a clinical risk. The clinical risk depends
on penetrance, age, sex, family history, and modifiers, and it is a separate statement.

A report that says "pathogenic variant, 80% lifetime risk" is making two claims with different
evidence. The first is a classification; the second is a risk estimate that requires
penetrance data specific to the gene, the variant, and the family.

### 3.3 The report as an interface

The report is read by at least three audiences: the ordering clinician, the patient (often via a
portal), and the next clinician who inherits care. Each needs different information at different
levels of detail. A report written only for the geneticist fails the oncologist; a report written
only for the oncologist fails the family.

The practical structure is a short summary with the finding and the action, followed by the
evidence table and the limitations. The summary must be readable without the evidence table; the
evidence table must support the summary for anyone who checks.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Germline variant | Variant present in all cells | Confused with hereditary disease | VAF near 0.5 |
| Somatic variant | Variant acquired in a tissue | Confused with germline | VAF variable |
| VUS | Variant of uncertain significance | Reported as a diagnosis | Category |
| Pathogenic | Variant category indicating causal role | Treated as a probability | Category |
| Likely pathogenic | Category above 90% certainty | Treated as pathogenic | Category |
| Secondary finding | Actionable variant unrelated to indication | Confused with incidental finding | ACMG SF list |
| Incidental finding | Finding unrelated to the test's purpose | Used loosely | Finding |
| Actionability | Whether an intervention changes outcome | Confused with pathogenicity | Binary with evidence |
| Penetrance | Probability of phenotype given genotype | Confused with classification | Proportion by age |
| Expressivity | Variation in phenotype severity | Ignored | Description |
| Genetic counselling | Communication and support process | Confused with consent | Service |
| Informed consent | Agreement after understanding risks and limits | Treated as a signature | Process |
| Cascade testing | Testing relatives of a proband | Confused with population screening | Strategy |
| Reanalysis | Reinterpretation with new knowledge | Confused with re-sequencing | Process |
| Reclassification | Change in variant category | Confused with reanalysis | Event |
| Pharmacogenomics | Variant effects on drug response | Confused with dosing advice | Field |
| CPIC | Consortium producing PGx guidelines | Confused with a regulator | Guideline body |
| DPWG | Dutch Pharmacogenetics Working Group | Guideline body | Guideline body |
| Clinical validity | Association with the condition | Confused with utility | Evidence |
| Clinical utility | Evidence that use improves outcomes | Confused with validity | Evidence |
| Analytical validity | Accuracy of the measurement | Confused with interpretation | Evidence |
| CLIA | US laboratory regulation | Confused with device regulation | Regulation |
| CAP | Laboratory accreditation | Confused with CLIA | Accreditation |
| ISO 15189 | International laboratory standard | Confused with a device standard | Standard |
| IVDR | EU in vitro diagnostic regulation | Confused with FDA | Regulation |
| GINA | US genetic information nondiscrimination law | Assumed to cover all insurance | Law |
| HIPAA | US health information privacy law | Confused with GINA | Law |
| GDPR | EU data protection regulation | Assumed to be the same as HIPAA | Law |
| Tumor mutational burden | Mutations per megabase | Compared across panels | Mutations/Mb |
| MSI | Microsatellite instability | Confused with TMB | Score |
| ctDNA | Circulating tumour DNA | Confused with germline | Fraction |
| MRD | Minimal residual disease | Confused with ctDNA detection | Status |
| Molecular tumour board | Multidisciplinary review of genomic results | Confused with a report | Process |

Three distinctions carry the most weight in practice.

**Secondary and incidental findings are not synonyms.** A secondary finding is deliberately
sought in a defined gene list (for example, the ACMG SF list) and reported because it is
actionable. An incidental finding is discovered without being sought. The distinction determines
the consent conversation and the reporting policy.

**Clinical validity and clinical utility are different claims.** A variant can be convincingly
associated with a condition and still have no intervention that changes the outcome. Utility is a
separate evidence standard, and it is the one payers and guideline committees apply.

**Reanalysis is not re-sequencing.** Reanalysis reinterprets existing data with new knowledge: new
frequency data, new functional studies, new classification guidance. It is cheaper than
re-sequencing and is the main mechanism by which negative results become diagnoses.

## 5. The mechanics, step by step

### Step 1. Establish the indication and the pretest probability

**Input:** phenotype, family history, prior testing.

**Output:** a clinical question and a pretest probability, with the reasoning.

**Rationale:** The pretest probability determines whether a positive result is believable and
whether a negative result is informative. A patient with a strong family history and early onset
has a higher pretest probability than a population screen, and the same variant carries different
post-test meaning. The indication also determines the gene set and the variant classes.

### Step 2. Obtain informed consent with the right scope

**Input:** the test plan.

**Output:** a documented consent covering secondary findings, data use, and reanalysis.

**Rationale:** Consent is not a signature on a form. It is a conversation about what the test can
find, what it cannot find, what will be reported, what will not, and what happens to the data.
The secondary findings decision must be explicit and recorded, because the default varies by
laboratory and jurisdiction.

### Step 3. Select the test and the sample

**Input:** the indication and the variant classes.

**Output:** a test specification with sample type and coverage requirements.

**Rationale:** A panel, an exome, a genome, and a targeted assay cover different variant classes.
The sample must be appropriate: blood for germline, tumour with a matched normal for somatic,
saliva with awareness of chimerism, and a second tissue for suspected mosaicism. Sample type is
part of the test specification, not an operational detail.

### Step 4. Perform the laboratory analysis with documented QC

**Input:** sample and test specification.

**Output:** a VCF and a QC record.

**Rationale:** The QC record must include coverage at the reported sites, the fraction of the
target covered at the validated depth, contamination estimates, and sample identity checks. A
result reported without coverage at the site is not interpretable, and a negative result without
a coverage statement is not a negative.

### Step 5. Classify variants with a documented evidence table

**Input:** annotated variants.

**Output:** classifications with criteria and evidence.

**Rationale:** The classification must be reproducible by a second reviewer. That requires the
criteria applied, the evidence behind each, the transcript, and the frequency dataset with
version. A classification without an evidence table cannot be audited, and cannot be updated
systematically when knowledge changes.

### Step 6. Apply a secondary findings policy

**Input:** consent and the laboratory policy.

**Output:** a decision to report or not, with the gene list and the rationale.

**Rationale:** The ACMG SF list is a policy statement, not a mandate: it recommends reporting a
defined set of actionable genes when patients opt in, and it is updated as evidence changes. The
2023 v3.2 list contains 81 genes, after adding CALM1, CALM2, and CALM3. Laboratories differ in
whether they report, and the difference must be disclosed in consent.

### Step 7. Annotate pharmacogenomic variants

**Input:** the gene panel and the patient's variants.

**Output:** metabolizer status and drug-specific implications.

**Rationale:** Pharmacogenomic reporting uses a different evidence standard: drug response rather
than Mendelian causality. Guidelines from CPIC and DPWG define the phenotype categories and the
drug-specific recommendations. The report must state the guideline version, because
recommendations change.

### Step 8. Report somatic findings with tiered evidence

**Input:** tumour variants.

**Output:** tiered interpretations with therapy, prognosis, and diagnostic relevance.

**Rationale:** Somatic reporting uses tiers (for example, strong clinical significance, potential
clinical significance, unknown significance, benign). The tier depends on the evidence for the
specific tumour type and the specific drug, not on the variant's frequency alone. A variant that
is a known driver in one cancer may be a variant of unknown significance in another.

### Step 9. Write the report in four parts

**Input:** the classifications and the clinical context.

**Output:** a report with finding, interpretation, action, and limitation.

**Rationale:** The four-part structure forces the writer to separate what was found from what it
means from what should be done from what could not be excluded. Most report defects are
conflations of these parts: a finding presented as an action, or an interpretation presented
without its limitation.

### Step 10. Communicate with the clinician before the report lands

**Input:** the report and the clinical context.

**Output:** a conversation or a call for urgent or unexpected findings.

**Rationale:** A secondary finding in a cardiac gene, or a pathogenic variant in a patient not
expecting one, should not arrive first as a portal notification. The clinician needs the context
to plan the conversation. This is a workflow requirement, not a courtesy.

### Step 11. Support counselling and cascade testing

**Input:** the result and the family structure.

**Output:** a counselling referral and a cascade testing plan.

**Rationale:** A germline pathogenic variant has implications for relatives, and cascade testing
is the mechanism by which the diagnosis produces value beyond the proband. The report should name
the implication, and the service should have a pathway for relatives who are not yet patients.

### Step 12. Define a reanalysis policy

**Input:** the test and the knowledge landscape.

**Output:** triggers, responsibilities, and a communication plan.

**Rationale:** Classifications change. A VUS may be reclassified as pathogenic when new evidence
appears, and a negative exome may become diagnostic when a new gene is discovered. Without a
policy, reanalysis happens only when a clinician remembers to ask. The policy should specify
triggers (time-based, event-based, or request-based), who is responsible, and how patients are
contacted.

### Step 13. Monitor quality and audit reports

**Input:** reports and outcomes.

**Output:** audit findings and corrective actions.

**Rationale:** Quality assurance for clinical genomics includes proficiency testing, concordance
checks with reference materials, audit of report completeness, and review of classification
concordance with external experts. The audit should sample reports across clinicians and
indications, because report quality varies by author.

### Step 14. Address equity and data governance

**Input:** the patient population and the data resources.

**Output:** an equity assessment and a data governance plan.

**Rationale:** Reference databases under-represent some ancestries, and the resulting
misclassification is not random. A laboratory that serves a diverse population must state how it
handles frequency evidence for under-represented groups, and must not treat absence from a
database as rarity without a coverage check. Data sharing improves the databases but requires
consent and governance.

### Step 15. Record the result in a structured, reusable form

**Input:** the report.

**Output:** a structured result linked to the report.

**Rationale:** A PDF report cannot be queried, compared across patients, or fed into a registry.
A structured result allows the laboratory to audit concordance, the health system to track
outcomes, and a research program to reuse data under consent. The structured form must carry the
same evidence as the report, not a summary of it.

### Step 16. Plan the handoff for transitions of care

**Input:** the patient's care pathway.

**Output:** a documented handoff to the next clinician and to the family.

**Rationale:** Genomic results are often relevant years after the test, when the patient has moved
to a new clinician. A result that lives only in one institution's record is lost at the
transition. The handoff should include the report, the reanalysis status, and the family
implications, in a form that travels.

### Step 17. Audit equity of access and outcome

**Input:** testing volumes and outcomes by population group.

**Output:** an equity report with identified gaps.

**Rationale:** A laboratory can measure who is tested, who receives a reportable finding, who is
referred for counselling, and who completes cascade testing. Differences by ancestry, language,
insurance, or geography are actionable. Without measurement, inequity is invisible.

### Branch points

**Branch A — the variant is a VUS in a high-risk gene.** Report the VUS, state what it does not
mean, and describe the reanalysis pathway. Do not use it for management.

**Branch B — a secondary finding is detected in a patient who declined secondary findings.**
Follow the consent and the jurisdiction's policy. In most frameworks, the patient's choice is
respected unless the finding is immediately actionable and disclosure is legally required.

**Branch C — the result is negative but coverage was low at a relevant gene.** Report the gene as
not adequately covered, not as negative. This is the most common reporting defect in panel
testing.

### Decision table: finding to report action

| Finding | Report | Action | Caveat |
|---|---|---|---|
| Pathogenic in indication gene | Yes | Management per guideline | State penetrance and age-dependence |
| Likely pathogenic | Yes | Often management, with caveat | State the evidence gap |
| VUS | Yes, with explicit statement | None based on the variant | Reanalysis pathway |
| Secondary finding, opted in | Yes | Referral per gene | Consent recorded |
| Secondary finding, opted out | Per policy | Usually not reported | Document the decision |
| Pharmacogenomic variant | Yes, if panel includes it | Drug-specific recommendation | Guideline version |
| Somatic tier I | Yes | Therapy or trial | Tumour type and evidence level |
| Negative with adequate coverage | Yes | Continue clinical workup | State coverage |
| Negative with inadequate coverage | Yes, as inconclusive | Consider re-testing | Name the gene and coverage |

### Edge cases

**VUS in a high-risk gene.** Diagnostic: classification in the 0-5 point range with no
informative evidence. Consequence: patient anxiety and possible overtreatment. Mitigation: clear
language, counselling, and a reanalysis plan.

**Low-level mosaicism.** Diagnostic: VAF well below 0.5 in blood. Consequence: a germline
classification applied incorrectly. Mitigation: second tissue and a stated limitation.

**Tumor-only testing.** Diagnostic: no matched normal. Consequence: germline variants reported as
somatic or filtered out. Mitigation: paired normal, or flag and confirm.

**Inadequate coverage.** Diagnostic: the gene of interest is below the validated depth.
Consequence: a false negative reported as negative. Mitigation: report coverage per gene.

**Consent mismatch.** Diagnostic: the report includes secondary findings the patient declined.
Consequence: a serious ethical and legal breach. Mitigation: a report-level consent flag.

**Ancestry under-representation.** Diagnostic: the variant is absent from gnomAD but the site has
no coverage in the patient's ancestry group. Consequence: a common variant classified as rare.
Mitigation: ancestry-aware frequency review and coverage checks.

**Pediatric testing.** Diagnostic: a test for an adult-onset condition in a child. Consequence:
information that may harm the child and violate the child's future autonomy. Mitigation: policy
restricting predictive testing in minors.

**Family disclosure.** Diagnostic: a pathogenic variant with no cascade pathway. Consequence:
relatives miss surveillance. Mitigation: a family letter and a testing pathway.

**Reclassification after reporting.** Diagnostic: a variant's category changes after the report.
Consequence: patients are managed on outdated information. Mitigation: a reanalysis policy and a
contact registry.

**Result in a patient who cannot consent.** Diagnostic: testing performed on a sample from a
patient without capacity. Consequence: a result the patient did not agree to. Mitigation:
substituted judgement, best-interest review, and a documented process.

**Result that implies a relative's risk.** Diagnostic: a pathogenic variant with implications for
a relative who has not been tested. Consequence: a duty to warn that conflicts with
confidentiality. Mitigation: a family communication plan and a policy consistent with local law.

**Incidental consanguinity or non-paternity.** Diagnostic: genotype data inconsistent with the
stated relationship. Consequence: unintended disclosure with family consequences. Mitigation: a
policy for unexpected findings, agreed before testing.

**Variant in a gene with disputed validity.** Diagnostic: the gene-disease relationship is
reported as limited or disputed by ClinGen. Consequence: a pathogenic classification in a gene
that may not cause disease. Mitigation: check gene-disease validity before classifying, and state
it in the report.

**Report issued before confirmation.** Diagnostic: the reportable variant is not orthogonally
confirmed. Consequence: a false report and a recall. Mitigation: hold the report until
confirmation, except for urgent situations with an explicit caveat.

**Template drift.** Diagnostic: reports from different authors use different structures.
Consequence: inequitable communication. Mitigation: a controlled template with required fields
and audit.

**Turnaround pressure.** Diagnostic: reports are issued with "pending confirmation" and never
updated. Consequence: an unconfirmed result becomes the record. Mitigation: a status field and an
automated follow-up.

**Test ordered for the wrong indication.** Diagnostic: the phenotype does not match the panel.
Consequence: a negative result that does not exclude the suspected condition. Mitigation: a
pre-test review by the laboratory or a genetic counsellor.

**Follow-up test not ordered.** Diagnostic: a report recommends a confirmatory or family test with
no order. Consequence: the recommendation is never acted on. Mitigation: a closed-loop order or a
documented referral.

The order of the steps is deliberate. Consent precedes testing because the findings determine the
conversation; classification precedes reporting because the report cannot be clearer than the
evidence; and the reanalysis policy precedes sign-out because a policy written after a
reclassification request is a policy written under pressure.

### What the procedure produces

The output is a report and a record: the indication, consent scope, test and coverage, the
classification with evidence, the secondary findings decision, the pharmacogenomic annotation,
the action, the limitations, the counselling referral, and the reanalysis plan. The report is the
product; the record is what makes it defensible.

A useful test of a finished report: give it to a clinician who did not order the test and ask
what they would do. If the answer is not the intended action, the report has failed, regardless
of the classification's correctness.

## 6. Worked example

A 35-year-old is diagnosed with breast cancer. A multigene panel is ordered. The report returns
three findings: a pathogenic variant in a high-penetrance gene, a VUS in a second gene, and a
pharmacogenomic variant.

> **Note:** the genes and the disease associations used here are real and well established. The
> specific variants and the case details are constructed for teaching.

### 6.1 The pretest and post-test probability

The patient's pretest probability of carrying a pathogenic variant, given early onset and a
first-degree relative with breast cancer, is estimated at 10%. The pathogenic variant found has a
likelihood ratio of 8 for the associated syndrome, based on segregation and penetrance data.

```text
prior odds = 0.10 / 0.90 = 0.111
posterior odds = 0.111 x 8 = 0.889
posterior probability = 0.889 / (1 + 0.889) = 0.471
```

The post-test probability is 47%. That is the probability that the patient has the syndrome given
the test result and the prior. It is not the probability that the variant is pathogenic, which is
already established by classification, and it is not the probability of developing cancer, which
depends on penetrance and age.

If the likelihood ratio were 20, the posterior would be 69%; if it were 50, 85%. The same variant
produces different clinical meaning in different patients, which is why the report's
interpretation must be tied to the clinical context rather than to the classification alone.

### 6.2 The VUS

The second finding is a missense VUS in a moderate-risk gene. The evidence is: PM2 supporting
(absent from gnomAD, with coverage checked), PP3 supporting (two computational predictors), and
no functional or segregation data. The point total is 2, which is in the uncertain range.

```text
points = 1 (PM2) + 1 (PP3) = 2  ->  VUS
```

The report must state three things: what the VUS is, what it does not mean, and what will happen
next. "A variant of uncertain significance was found in gene X. This result does not establish
an increased risk and should not be used to change your management. The variant will be
re-reviewed annually, and you will be contacted if the classification changes."

That paragraph is more useful than a classification alone, and it takes two minutes to write.

### 6.3 The reclassification arithmetic

Published reclassification rates for VUS vary by gene and by follow-up time. Suppose the annual
probability of reclassification to pathogenic or likely pathogenic is 3% for this gene, and the
annual probability of reclassification to benign or likely benign is 5%.

After five years, the probability that the variant is still a VUS is:

```text
(1 - 0.03 - 0.05)^5 = 0.92^5 = 0.659
```

So about 66% of VUS in this gene remain uncertain after five years, 17% become actionable
pathogenic or likely pathogenic, and 23% become benign or likely benign. The arithmetic is
illustrative, not a published estimate, but it shows why a reanalysis policy matters: roughly one
in six VUS becomes actionable within five years in this scenario.

### 6.4 The pharmacogenomic finding

The panel includes a pharmacogenomic variant that predicts intermediate metabolizer status for a
drug the patient may receive. The report cites the CPIC guideline version and gives the
drug-specific recommendation.

The evidence standard here is different from the germline classification: it is drug-response
evidence, often from pharmacokinetic studies, not segregation. The report must not present the
pharmacogenomic finding as a disease risk, and must not present the germline finding as a dosing
recommendation.

### 6.5 Diagnostic 1: the coverage gap

The negative result for a third gene is reported as "no pathogenic variant detected." The QC
record shows the gene had a mean coverage of 12x, below the laboratory's validated threshold of
20x, and two exons had no coverage.

The correct report states: "Gene Z was not adequately covered (mean 12x; exons 4 and 7 at 0x).
A pathogenic variant in this gene cannot be excluded." The difference between the two statements
is the difference between a false negative and an informative negative.

### 6.6 Diagnostic 2: the ancestry check

The VUS includes PM2 based on absence from gnomAD. A review of the ancestry-specific subset shows
the patient's population has 0x coverage at the site, and a small cohort study reports the
variant at 0.8% in that population.

The variant is not rare in the patient's population. PM2 should not have been applied.
Reclassifying without PM2 gives one point (PP3), which remains a VUS, but the evidence statement
changes from "absent from controls" to "frequency unknown in the patient's population, likely
common." That change matters for reanalysis and for any future evidence.

### 6.7 Vary one assumption: the variant is reclassified

Suppose that two years later, a functional study shows the VUS abolishes protein function, and
the classification becomes likely pathogenic (PS3 strong plus PP3 supporting, 5 points, plus
PM2 if still applicable). The patient, who had been managed as average risk, now qualifies for
enhanced surveillance.

The change is only useful if someone contacts her. A report without a reanalysis policy produces
a classification that is correct at the time and obsolete later, and the patient bears the cost of
the obsolescence.

### 6.8 The family arithmetic

The patient has two siblings and two adult children. For an autosomal dominant condition with a
50% transmission probability per child, the expected number of carriers among four first-degree
relatives is:

```text
4 x 0.50 = 2.0 expected carriers
```

The probability that at least one relative is a carrier is:

```text
1 - (1 - 0.5)^4 = 1 - 0.0625 = 0.9375
```

There is a 94% chance that at least one relative carries the variant, and cascade testing is
likely to find someone. That arithmetic is why cascade testing is the highest-yield part of a
genomics service: the pretest probability in relatives is 50%, not the population prevalence.

If each carrier needs surveillance costing USD 1,500 per year and the alternative is late
detection, the case depends on the number needed to test. Four relatives tested, two carriers
detected, at a testing cost of about USD 2,000 each:

```text
cost per carrier detected = (4 x 2,000) / 2 = USD 4,000
```

Compare that with the cost of one late-stage diagnosis. The arithmetic is simple and it is the
argument that sustains cascade programs.

### 6.9 What the report says

> Indication: breast cancer at 35, family history. Consent: secondary findings declined.
> Findings: (1) pathogenic variant in gene A, heterozygous, confirmed by Sanger, coverage 142x;
> (2) VUS in gene B, PM2 and PP3 only, no functional data; (3) intermediate metabolizer genotype
> for drug D, per CPIC version X. Interpretation: gene A result confers a 47% post-test
> probability of the associated syndrome given the pretest probability; the VUS does not
> establish risk. Actions: referral to genetics for counselling and cascade testing; no
> management change based on the VUS; drug D dosing per guideline. Limitations: gene Z not
> adequately covered (exons 4 and 7 at 0x); the panel does not detect structural variants or
> repeat expansions. Reanalysis: annual review; patient contactable via the record.

The report is four paragraphs, each answering a different question. It is readable by the
oncologist, checkable by the geneticist, and explainable by the counsellor.

The limitations section is doing the most work in this report. It converts two potential false
negatives (gene Z, structural variants) into stated scope limits, which is the difference between
a report that says "nothing else was found" and one that says "these things were not looked for."
The first is a claim; the second is a fact about the assay.

That distinction is what a clinical genomics report is for. The classification is the science; the
scope statement is the medicine.

## 7. Code, query, or calculation

The script computes a Bayesian post-test probability, applies the ACMG point system, and scores a
report against a completeness checklist.

### 7.1 Post-test probability

```python
def posterior(prior, lr):
    odds = prior / (1 - prior)
    post_odds = odds * lr
    return post_odds / (1 + post_odds)

for lr in (2, 8, 20, 50):
    print(f"LR {lr}: posterior {posterior(0.10, lr):.3f}")
```

Expected output:

```text
LR 2: posterior 0.182
LR 8: posterior 0.471
LR 20: posterior 0.690
LR 50: posterior 0.847
```

The likelihood ratio is the multiplicative weight of the test result. A classification is not a
likelihood ratio, but evidence categories can be translated into one; the translation is a
modelling choice and should be stated.

### 7.2 ACMG points and category

```python
POINTS = {"very_strong": 8, "strong": 4, "moderate": 2, "supporting": 1}

def classify(codes):
    total = sum(POINTS[s] * (1 if d == "p" else -1) for _, s, d in codes)
    if total >= 10:
        cat = "Pathogenic"
    elif total >= 6:
        cat = "Likely pathogenic"
    elif total >= 0:
        cat = "VUS"
    elif total >= -3:
        cat = "Likely benign"
    else:
        cat = "Benign"
    return total, cat

cases = {
    "PVS1 + PM2 + PP3": [("PVS1", "very_strong", "p"), ("PM2", "supporting", "p"),
                          ("PP3", "supporting", "p")],
    "PM2 + PP3": [("PM2", "supporting", "p"), ("PP3", "supporting", "p")],
    "PS3 + PP3": [("PS3", "strong", "p"), ("PP3", "supporting", "p")],
    "BA1 alone": [("BA1", "very_strong", "b")],
}
for name, codes in cases.items():
    print(name, "->", classify(codes))
```

Expected output:

```text
PVS1 + PM2 + PP3 -> (10, 'Pathogenic')
PM2 + PP3 -> (2, 'VUS')
PS3 + PP3 -> (5, 'VUS')
BA1 alone -> (-8, 'Benign')
```

The third case is the important one: a strong functional result plus supporting computational
evidence is not enough for likely pathogenic, because the point total is 5. The threshold
structure is unforgiving by design.

### 7.3 Report completeness score

```python
REQUIRED = ["indication", "consent", "specimen", "coverage", "variant",
            "transcript", "classification", "evidence", "action",
            "limitations", "reanalysis"]

def completeness(report):
    present = [k for k in REQUIRED if report.get(k)]
    missing = [k for k in REQUIRED if not report.get(k)]
    return len(present) / len(REQUIRED), missing

report = {
    "indication": "breast cancer at 35",
    "consent": "secondary findings declined",
    "specimen": "blood",
    "coverage": "142x at reported site; gene Z 12x",
    "variant": "c. example",
    "transcript": "MANE Select",
    "classification": "Pathogenic",
    "evidence": "PVS1, PM2, PP3",
    "action": "genetics referral",
    "limitations": "gene Z not covered",
    "reanalysis": "annual",
}
score, missing = completeness(report)
print("completeness:", round(score, 2))
print("missing:", missing)

partial = dict(report)
del partial["limitations"]
del partial["transcript"]
print("partial:", completeness(partial))
```

Expected output:

```text
completeness: 1.0
missing: []
partial: (0.8181818181818182, ['transcript', 'limitations'])
```

A checklist is not a quality measure, but it catches the omissions that account for most report
defects: the transcript, the coverage, and the limitations.

### 7.4 Reclassification over time

```python
p_path, p_benign = 0.03, 0.05
stay = (1 - p_path - p_benign) ** 5
to_path = 1 - (1 - p_path) ** 5
to_benign = 1 - (1 - p_benign) ** 5
print("still VUS at 5 years:", round(stay, 3))
print("became pathogenic:", round(to_path, 3))
print("became benign:", round(to_benign, 3))
```

Expected output:

```text
still VUS at 5 years: 0.659
became pathogenic: 0.141
became benign: 0.226
```

The three numbers do not sum to one because a variant can move in either direction, and the
independent-probability model allows a variant to be counted in both transitions over five years.
The model is illustrative; a real analysis would use competing risks.

### 7.5 Assumptions and limits of this code

- The Bayesian model assumes a single likelihood ratio and a well-calibrated prior.
- The point system assumes independent evidence; correlated codes inflate the total.
- The completeness checklist measures presence, not quality.
- The reclassification model assumes constant annual probabilities, which is unrealistic.
- No code here performs classification, which requires evidence review by a qualified
  interpreter.

## 8. Data, standards, and regulatory context

### 8.1 Classification and reporting standards

- **ACMG/AMP 2015** defines the criteria and categories; ClinGen specifications define how each
  criterion is applied. The Bayesian reformulation (Tavtigian 2018) and the point system
  (Tavtigian 2020) provide the arithmetic used in section 7.
- **ACMG SF v3.2** (2023) lists 81 genes for secondary findings reporting, after adding CALM1,
  CALM2, and CALM3. The list is a policy recommendation with an opt-in model; it is not a
  regulatory requirement.
- **ClinGen** publishes gene-disease validity, dosage sensitivity, and variant-specific
  specifications.
- **HGVS** nomenclature is required for variant description.
- **ClinVar** is an archive of interpretations with review status, not a truth set.

### 8.2 Laboratory regulation

- **CLIA** requires performance specifications for accuracy, precision, analytical sensitivity and
  specificity, reportable range, and reference intervals, plus proficiency testing where
  available.
- **CAP** accreditation adds document control, competency assessment, and inspection.
- **ISO 15189:2022** provides the international quality framework, including point-of-care
  testing merged from ISO 22870.
- **IVDR 2017/746** applies to in vitro diagnostic devices in the EU, including companion
  diagnostics.

The laboratory-developed test pathway in the US is unsettled. The 2024 FDA final rule was vacated
on 31 March 2025 by the Eastern District of Texas, and the FDA's September 2025 "Implementation of
Vacatur" rule restored the prior text. LDTs are currently under enforcement discretion, with the
matter remanded. A clinical genomics program should plan for either outcome.

### 8.3 Privacy and nondiscrimination

- **GINA** (US, 2008) prohibits discrimination in employment and health insurance based on
  genetic information. It does not cover life insurance, disability insurance, or long-term care
  insurance.
- **HIPAA** governs the privacy and security of protected health information, including genomic
  data held by covered entities.
- **GDPR** treats genetic data as a special category under Article 9, requiring an explicit legal
  basis and additional safeguards.
- **Common Rule** and institutional review boards govern research use of human data.
- **Data use agreements** constrain secondary use of controlled-access datasets.

A report that says "your data are protected by GINA" without the life-insurance caveat is
inaccurate, and the inaccuracy matters to patients making decisions about testing.

### 8.4 Professional and accreditation expectations

Beyond the statutory instruments, professional bodies set expectations:

- **ACMG** policy statements on secondary findings, incidental findings, and reanalysis.
- **CAP** checklists for molecular pathology, which specify report content.
- **NSGC** practice guidelines for genetic counselling.
- **AMP** guidance on somatic variant interpretation and reporting.

These are not law, but accreditation inspections and malpractice review use them as the standard
of practice. A laboratory that departs from them should document why.

### 8.5 What the standards do not settle

- Whether to report secondary findings, and which genes.
- How to communicate a VUS.
- How much coverage is adequate for a given gene.
- When to reanalyze, and who pays for it.
- How to handle variants of uncertain significance in under-represented populations.

These are policy and practice decisions, and they vary by laboratory, country, and payer. A
program should document its choices rather than assume a standard exists.

## 9. Industry practice and case evidence

### 9.1 The databases and their scale

ClinVar, as of the mid-2026 snapshot, contains roughly 6.9 million submitted records and 4.5
million unique variants from more than 3,400 submitters, with a small fraction carrying expert-
panel review. gnomAD v4 contains 807,162 individuals, with 730,947 exomes and 76,215 genomes, and
about 138,000 individuals of non-European ancestry.

The scale is impressive and the coverage is uneven. A variant absent from gnomAD is not
necessarily rare in a population that is under-represented in the database, and a ClinVar record
without expert review is an opinion rather than a consensus. Both facts are reportable
limitations, not footnotes.

### 9.2 What laboratories do

Documented practice:

- Clinical laboratories use ACMG/AMP with ClinGen specifications and MANE Select transcripts.
- Most laboratories confirm reportable variants orthogonally and state the method.
- Secondary findings policies vary; some report the ACMG SF list by default with opt-out, others
  require opt-in.
- Reanalysis is increasingly programmatic, with time-based triggers and patient contact
  registries.
- Pharmacogenomic results are reported with CPIC or DPWG guidance and the guideline version.

Inference: the largest source of variation across laboratories is not the classification
framework but the evidence assembly and the reporting language. Two laboratories can agree on a
category and disagree on what the report says, and the report is what the patient experiences.

### 9.3 The equity evidence

Polygenic scores and variant classification share the same underlying problem: reference data are
skewed toward European ancestry. For classification, the consequence is that a variant may be
absent from a database because the population is absent, not because the variant is rare. The
mitigation is ancestry-aware frequency review, coverage checks, and a stated limitation.

For polygenic scores, the consequence is larger: a score developed in one ancestry has
substantially lower performance in another, and clinical use without ancestry-specific validation
can widen disparities. The evidence for this is covered in the survey file and in topic 08.

### 9.4 What direct-to-consumer testing changed

Direct-to-consumer genetic testing moved variant interpretation outside the clinical laboratory,
often with a different evidence standard and a different consent model. The clinical consequence
is that patients arrive with raw data and third-party interpretations, some of which are
classified differently from a clinical laboratory's interpretation.

The practical response is not to dismiss the data but to reanalyze it: confirm the variant in a
clinical laboratory, apply the clinical classification framework, and explain any discrepancy. A
discrepancy is itself informative, because it usually reflects a difference in evidence
inclusion rather than an error.

### 9.5 What health systems changed

Several national health systems have moved genomic testing into centralized services with
standardized consent, reporting, and reanalysis. The documented motivation is consistency and
equity: a single service can apply the same classification standards, the same report template,
and the same reanalysis policy across a population.

The trade-off is centralization itself: a single laboratory serving a whole country creates
capacity constraints and a single point of failure. The design question is not centralization
versus local testing but where the standards live, and whether the service can be audited against
them.

The public evidence on reanalysis yield is heterogeneous, and the heterogeneity is informative:
cohorts differ in gene set, follow-up interval, and the definition of a diagnostic finding. A
program planning a reanalysis service should estimate its own yield from its own backlog rather
than adopt a published rate, and should report the yield so that others can compare.

### 9.7 The reanalysis evidence

Reanalysis of negative exomes and VUS yields diagnoses at rates reported in the literature that
vary widely by cohort and follow-up time. The consistent finding is that a meaningful fraction of
previously uncertain or negative cases become diagnostic, and that the yield depends on the
interval since the original test and the gene-disease discovery rate.

The operational lesson is that a genomic test is a service over time, not a one-time result. The
laboratory that issued the report is not always the party that reanalyzes it, and the handoff is
where patients are lost.

> **Note:** reanalysis yield estimates vary by cohort, gene set, and follow-up. Cite the specific
> study rather than a general rate.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| VUS reported as diagnosis | Report language implies action | Category conflation | Overtreatment, anxiety | Explicit VUS paragraph |
| No coverage statement | Negative report without depth | Assay limit unstated | False reassurance | Per-gene coverage |
| Secondary finding without consent | Report includes SF in opt-out patient | Consent mismatch | Ethical and legal breach | Consent flag in report |
| Ancestry-blind frequency | PM2 applied without coverage check | Database bias | Misclassification | Ancestry-aware review |
| Wrong transcript | Consequence differs across annotations | Annotation choice | Wrong HGVS and classification | MANE Select |
| Tumor-only germline confusion | VAF near 0.5 in tumour | No matched normal | Misattributed result | Paired normal |
| No orthogonal confirmation | Single-platform call | Artifact | False report | Confirm reportable variants |
| Report unreadable | Clinician calls for clarification | Jargon, length | Decision delayed | Four-part structure |
| No reanalysis policy | VUS never revisited | Process gap | Obsolete result | Triggers and registry |
| Pediatric predictive testing | Adult-onset gene in a child | Policy gap | Harm to child's autonomy | Age policy |
| Family disclosure gap | No cascade pathway | Workflow gap | Relatives miss surveillance | Family letter and pathway |
| Misreported penetrance | "80% risk" without source | Risk estimate conflated with classification | Wrong decisions | Cite penetrance source |
| Unexpected non-paternity | Genotype inconsistency | Incidental finding | Family disruption | Pre-agreed policy |
| Data breach | Unencrypted transfer | Governance gap | Privacy harm | Encryption, access control |
| Report author variation | Same variant, different language | No template | Inequitable care | Template and audit |
| Gene validity ignored | Disputed gene-disease link | Evidence not checked | Classification of a non-disease gene | ClinGen validity review |
| Premature report | Variant not confirmed | Workflow pressure | False report, recall | Hold until confirmed |

Four red flags:

**If the report does not state coverage at the reported gene, stop.** The negative may be
uninformative.

**If a VUS appears without an explicit statement of what it does not mean, stop.** The patient
will read it as a diagnosis.

**If a secondary finding appears in a patient who declined them, stop.** This is a consent
breach, not a reporting preference.

**If a frequency criterion was applied without checking coverage in the patient's ancestry,
stop.** The absence may be an artifact of the database.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The pretest probability is estimated | Post-test interpretation | Unknown baseline | Clinical model |
| The likelihood ratio is valid | Bayesian update | Classification not calibrated | Published estimates |
| The reference database represents the patient | Frequency evidence | Ancestry under-representation | Coverage check |
| Coverage is adequate at the gene | Negative result | Low depth | Per-base QC |
| The transcript is clinically relevant | HGVS and classification | Tissue-specific isoform | MANE review |
| The consent covers the findings | Ethical reporting | Scope mismatch | Report flag |
| The classification is current | Clinical action | Knowledge change | Reanalysis policy |
| The clinician can act on the report | Utility | Unclear language | Readability audit |
| Cascade testing is available | Family benefit | No pathway | Service design |
| Data are protected | Trust | Breach | Governance audit |

The assumption that the report is understood is the least tested. A report can be accurate,
complete, and still fail because the reader cannot tell what to do.

## 12. Exercises

**Exercise 1 — Report anatomy (recall, 25 minutes).**
Take a published example report. Identify the finding, interpretation, action, and limitation.
State which part is missing or conflated.
*Deliverable:* a four-part annotation.
*Criteria:* all four parts located or their absence justified.
*Time:* 25 minutes.

**Exercise 2 — Post-test probability (calculation, 30 minutes).**
Pretest probability 5%. Compute the post-test probability for likelihood ratios of 5, 10, and 25.
State how the interpretation changes across the three.
*Deliverable:* three calculations and a paragraph.
*Criteria:* arithmetic correct; interpretation tied to the numbers.
*Time:* 30 minutes.

**Exercise 3 — VUS language (application, 30 minutes).**
Write the VUS paragraph for a report, in language a patient can understand, including what it does
not mean and what happens next.
*Deliverable:* 150-200 words.
*Criteria:* no jargon without definition; explicit non-diagnosis statement; reanalysis pathway.
*Time:* 30 minutes.

**Exercise 4 — Secondary findings policy (design, 40 minutes).**
Write a secondary findings policy for a laboratory serving a mixed adult and pediatric
population. Specify the gene list, the opt-in or opt-out model, and the pediatric rule.
*Deliverable:* a one-page policy.
*Criteria:* gene list named with version; consent model explicit; pediatric rule justified.
*Time:* 40 minutes.

**Exercise 5 — Ancestry audit (critique, 35 minutes).**
A report applies PM2 to a variant absent from gnomAD. The patient's ancestry group has 0x coverage
at the site. Write the corrected evidence statement and the classification impact.
*Deliverable:* a corrected paragraph.
*Criteria:* coverage issue identified; PM2 removed; impact on category stated.
*Time:* 35 minutes.

**Exercise 6 — Reanalysis policy (design, 45 minutes).**
Design a reanalysis policy for a clinical genomics laboratory. Specify triggers, responsibilities,
patient contact, and resource estimate.
*Deliverable:* a two-page policy.
*Criteria:* at least two trigger types; named responsible role; contact pathway; resource logic.
*Time:* 45 minutes.

**Exercise 7 — Coverage language (application, 25 minutes).**
A gene has a mean coverage of 15x with two exons at 0x, against a validated threshold of 20x.
Write the report sentence and the recommended next step.
*Deliverable:* two sentences.
*Criteria:* states inadequacy, names the exons, recommends an action.
*Time:* 25 minutes.

**Exercise 8 — Regulatory map (design, 40 minutes).**
Map the regulatory obligations for a germline panel in the US and the EU: laboratory regulation,
device regulation, privacy, and nondiscrimination.
*Deliverable:* a four-row table by jurisdiction.
*Criteria:* correct instruments; binding versus voluntary distinguished; one gap per row.
*Time:* 40 minutes.

## 13. Self-check answers

**Answer 1 (report anatomy).** The finding is the variant with its HGVS and classification. The
interpretation is what the variant means for the patient, including penetrance and context. The
action is the recommendation, with the guideline or rationale. The limitation is what the assay
cannot exclude. The common defect is a report that merges interpretation into the finding
("pathogenic variant causing syndrome X") and omits the limitation, so the reader cannot tell
what was measured from what was inferred.

**Answer 2 (post-test).** Prior 0.05, odds 0.0526.

```text
LR 5:  odds 0.263 -> posterior 0.208
LR 10: odds 0.526 -> posterior 0.345
LR 25: odds 1.316 -> posterior 0.568
```

At LR 5 the result raises probability from 5% to 21%, which may not cross a decision threshold. At
LR 25 it reaches 57%, which likely does. The same test result means different things at different
likelihood ratios, and the report should not present a single number as if it were universal.

**Answer 3 (VUS).** "We found a change in gene B that we cannot yet classify. It is called a
variant of uncertain significance. This does not mean you have the condition associated with gene
B, and it should not change your screening or treatment. Most such variants are later found to be
harmless, but some are reclassified as important. We review variants every year and will contact
you if this one changes." The paragraph avoids "mutation," states the non-diagnosis explicitly,
and names the next step.

**Answer 4 (policy).** Gene list: ACMG SF v3.2, 81 genes, reviewed annually. Consent model:
opt-out for adults, with a documented conversation; opt-in for patients tested for adult-onset
conditions in childhood is not offered. Pediatric rule: secondary findings are reported only for
childhood-actionable conditions; adult-onset secondary findings are deferred until the patient can
consent. The rule protects the child's future autonomy and is consistent with professional
guidance.

**Answer 4 note.** A strong policy also states what happens when a patient changes their mind
after testing, and whether results already issued can be withdrawn. They usually cannot, which is
why the consent conversation precedes the test.

**Answer 5 (ancestry).** "The variant is absent from gnomAD v4 overall; however, the site has 0x
coverage in the patient's ancestry group, so absence is uninformative for this population. PM2 was
not applied." Without PM2, the variant retains only PP3 (1 point), which remains a VUS, but the
evidence statement changes and the variant is flagged for re-review if population data become
available.

**Answer 6 (reanalysis).** Triggers: annual time-based review of all VUS; event-based review when
a gene-disease association is established or a classification guideline changes; request-based
review from a clinician. Responsibility: the variant curation team, with a laboratory geneticist
sign-off. Contact: the original ordering clinician, with a patient letter template and a
documented attempt. Resource: estimate the number of VUS per year, the minutes per review, and the
curator FTE required; report the backlog separately.

**Answer 7 (coverage).** "Gene Z was not adequately covered: mean depth 15x against a validated
threshold of 20x, with exons 4 and 7 at 0x. A pathogenic variant in gene Z cannot be excluded;"
recommend re-testing with a method that covers those exons, or an orthogonal assay.

**Answer 6 note.** A strong policy also specifies who pays for reanalysis and what happens when
the patient cannot be contacted. Both are resource questions, and both are usually left implicit
in institutional policies.

**Answer 8 (regulatory map).** US: CLIA and CAP for the laboratory; FDA device regulation for a
marketed test, with LDTs under enforcement discretion after the 2025 vacatur; HIPAA for privacy;
GINA for employment and health insurance, not life insurance. EU: IVDR for the device; ISO 15189
accreditation where applicable; GDPR Article 9 for genetic data; national implementing law for
insurance. Each row has a gap: the US LDT pathway is unsettled; the EU IVDR transition and
notified-body capacity are constraints; GINA does not cover life insurance; GDPR requires a legal
basis that clinical consent may not supply for secondary research use.

The questions test whether a candidate can hold two ideas at once: the classification is a
technical judgment, and the report is a communication act with a different failure mode.

## 14. Interview questions

**1. What is a VUS and how do you communicate it?**
Strong answer: a category for insufficient evidence; explicitly not a diagnosis; state what it
does not mean and the reanalysis path.
Weak answer: "a variant we are not sure about," without the communication consequence.

**2. Why is a pathogenic classification not a probability of disease?**
Strong answer: the category is evidence-based; disease risk depends on penetrance, age, family
history, and modifiers.
Weak answer: treats pathogenic as "you will get the disease."

**3. What is a secondary finding?**
Strong answer: an actionable variant deliberately sought in a defined gene list, reported under a
consent policy; distinct from an incidental finding.
Weak answer: "an unexpected finding."

**4. How do you handle a negative result with poor coverage?**
Strong answer: report the gene as not adequately covered, name the exons, recommend re-testing;
never report it as negative.
Weak answer: "report negative with a caveat."

**5. What does GINA cover?**
Strong answer: employment and health insurance in the US; not life, disability, or long-term care
insurance.
Weak answer: "all genetic discrimination."

**6. How do you decide whether to report a VUS in a high-risk gene?**
Strong answer: report it with explicit limitation language and a reanalysis plan; do not use it
for management.
Weak answer: "do not report VUS" or "report as pathogenic if the gene is important."

**7. What triggers reanalysis?**
Strong answer: time-based, event-based (new gene-disease evidence, guideline change), and
request-based; with a named owner and a contact pathway.
Weak answer: "when the clinician asks."

**8. What is the difference between clinical validity and utility?**
Strong answer: validity is association with the condition; utility is evidence that using the
result improves outcomes.
Weak answer: conflates them.

**9. How does ancestry affect variant interpretation?**
Strong answer: frequency databases under-represent some populations, so absence may reflect
missing data; use coverage checks and state the limitation.
Weak answer: "genetics is the same in everyone."

**10. What should a report contain?**
Strong answer: finding, interpretation, action, limitation, and the technical evidence including
transcript and coverage.
Weak answer: "the variant and its classification."

**11. How would you handle a secondary finding in a patient who opted out?**
Strong answer: follow consent and jurisdiction; document the decision; disclose only if legally
required and immediately actionable, with clinical guidance.
Weak answer: "always report it because it is important."

**12. How would you audit report quality?**
Strong answer: sample reports across authors, score against a checklist, review with a clinician,
and track corrections; report the distribution, not the average.
Weak answer: "read a few reports."

**13. What is the regulatory status of laboratory-developed tests in the US?**
Strong answer: the 2024 rule was vacated in March 2025 and the FDA restored the prior text in
September 2025; LDTs are under enforcement discretion with the matter remanded.
Weak answer: "LDTs are unregulated" or "LDTs are devices."

## 15. Key terms

- **ACMG SF** — secondary findings gene list.
- **Actionability** — whether an intervention changes outcome.
- **Analytical validity** — accuracy of the measurement.
- **Cascade testing** — testing relatives of a proband.
- **CLIA** — US laboratory regulation.
- **Clinical utility** — evidence that use improves outcomes.
- **Clinical validity** — association with the condition.
- **ClinGen** — variant and gene curation consortium.
- **ClinVar** — public variant interpretation archive.
- **CPIC** — pharmacogenomics guideline consortium.
- **ctDNA** — circulating tumour DNA.
- **GDPR** — EU data protection regulation.
- **Genetic counselling** — communication and support service.
- **GINA** — US genetic nondiscrimination law.
- **HIPAA** — US health privacy law.
- **Incidental finding** — finding not deliberately sought.
- **Informed consent** — agreement after understanding risks and limits.
- **IVDR** — EU in vitro diagnostic regulation.
- **Likely pathogenic** — category above 90% certainty.
- **Molecular tumour board** — multidisciplinary genomic review.
- **MRD** — minimal residual disease.
- **MSI** — microsatellite instability.
- **Penetrance** — probability of phenotype given genotype.
- **Pharmacogenomics** — variant effects on drug response.
- **Reanalysis** — reinterpretation with new knowledge.
- **Reclassification** — change in variant category.
- **Secondary finding** — actionable variant unrelated to indication.
- **Somatic variant** — variant acquired in a tissue.
- **TMB** — tumour mutational burden.
- **VUS** — variant of uncertain significance.
- **Variant curation** — the evidence review behind a classification.
- **Report template** — controlled structure for clinical reports.

## 16. Sources

Classification and guidance:

- Richards, S. et al., "Standards and guidelines for the interpretation of sequence variants," *Genetics in Medicine* 17:405-424, 2015. https://doi.org/10.1038/gim.2015.30 (accessed 25 October 2026).
- Tavtigian, S. V. et al., "Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework," *Genetics in Medicine* 20:1054-1060, 2018. https://doi.org/10.1038/gim.2017.210 (accessed 25 October 2026).
- Miller, D. T. et al., "ACMG SF v3.2 list for reporting of secondary findings," *Genetics in Medicine* 25:100866, 2023. https://doi.org/10.1016/j.gim.2023.100866 (accessed 25 October 2026).
- ClinGen, Variant Classification Guidance. https://www.clinicalgenome.org/tools/clingen-variant-classification-guidance (accessed 25 October 2026).
- CPIC guidelines. https://cpicpgx.org/ (accessed 25 October 2026).

Data:

- ClinVar statistics. https://www.ncbi.nlm.nih.gov/clinvar/docs/statistics/ (accessed 25 October 2026).
- gnomAD v4.0 release. https://gnomad.broadinstitute.org/news/2023-11-gnomad-v4-0/ (accessed 25 October 2026).

Regulation:

- FDA, Laboratory Developed Tests. https://www.fda.gov/medical-devices/in-vitro-diagnostics/laboratory-developed-tests (accessed 25 October 2026).
- Federal Register, "Medical Devices; Laboratory Developed Tests; Implementation of Vacatur," 90 FR 45134, 19 September 2025. https://www.federalregister.gov/documents/2025/09/19/2025-18239/ (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).
- GINA, Genetic Information Nondiscrimination Act of 2008. https://www.eeoc.gov/laws/statutes/gina.cfm (accessed 25 October 2026).
- GDPR, Regulation (EU) 2016/679. https://eur-lex.europa.eu/eli/reg/2016/679/oj (accessed 25 October 2026).

The module ends here, but the clinical interface does not: every other topic in this folder
eventually reaches a report like the one in section 6, and the quality of that report determines
whether the science becomes care.

A final note on scope: this topic is educational and is not medical or legal advice. The
regulatory positions described are dated to October 2026 and can change, and the consent and
reporting policies described are examples, not recommendations for a specific jurisdiction.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_drug-discovery.md`](09_drug-discovery.md)
- Related topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Related topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 12.1, variant annotation, in the survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
