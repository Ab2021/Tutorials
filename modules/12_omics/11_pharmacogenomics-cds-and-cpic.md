# Pharmacogenomics Clinical Decision Support and CPIC Integration

> Module 12 — Omics and Precision Medicine
> Topic 11 of 16
> Estimated study time: 4 hours
> Prerequisites: topics 01, 02, and 10
> Version 2026.3

## 1. Why this topic exists

A prescription is written in about forty seconds. The genetic result that should change that
prescription may have been produced years earlier, in a different laboratory, using a different
nomenclature version, and stored as a PDF. Pharmacogenomics (PGx) is the study of how germline
variation changes drug exposure or drug response. Its clinical value is realized only if the
result reaches the prescriber at the moment of ordering, in a form the prescriber can act on
without leaving the electronic health record (EHR).

That gap between result and action is the subject of this topic. The biology is comparatively
well understood. The hard part is a data pipeline with four translation steps, each of which can
silently produce a plausible but wrong answer:

1. A variant call must become a star allele (`*` allele), which is a haplotype definition, not a
   single variant.
2. A pair of star alleles must become a diplotype, which requires knowing phase and copy number.
3. A diplotype must become a phenotype, using a gene-specific rule that is not the same for every
   gene.
4. A phenotype plus a drug plus an indication must become a recommendation with an evidence
   level and a recommendation strength.

Every step has a defensible-looking default that is wrong for some genes. A pipeline that sums
allele activity values works for cytochrome P450 2D6 (CYP2D6) and fails for cytochrome P450 2C19
(CYP2C19), where the Clinical Pharmacogenetics Implementation Consortium (CPIC) assigns phenotype
from allele-function pairs. A pipeline that treats the star allele as a single nucleotide
polymorphism (SNP) will call a deletion allele as wild type. A pipeline that looks up a drug
without the indication will give a cardiology recommendation to a stroke patient.

The consequences are not abstract. Three documented examples set the scale:

- **Clopidogrel and CYP2C19.** Clopidogrel is a prodrug that requires CYP2C19 activation. In a
  2021 meta-analysis cited by the 2022 CPIC guideline, patients with intermediate or poor
  metabolizer phenotypes had a relative risk of 0.70 (95% confidence interval 0.59 to 0.83) for
  major adverse cardiovascular events when treated with prasugrel or ticagrelor instead of
  clopidogrel. The genotype-treatment interaction p-value was 0.013.
- **Abacavir and HLA-B\*57:01.** In the PREDICT-1 randomized trial, prospective screening removed
  immunologically confirmed hypersensitivity reactions (0% versus 2.7% in the control arm). The
  allele is necessary but not sufficient: most carriers tolerate the drug, and the reaction is
  immune-mediated, so re-exposure after a reaction can be fatal.
- **Codeine and CYP2D6.** Codeine is a prodrug for morphine. Ultrarapid metabolizers convert it
  faster; the United States Food and Drug Administration (FDA) has a boxed warning against use in
  children younger than 12 years and after tonsillectomy or adenoidectomy in patients younger
  than 18 years, following deaths in that population.

None of these facts is new. What changed between 2010 and 2026 is the infrastructure: CPIC
publishes machine-readable tables, the CPIC database exposes a public API, PharmVar curates star
allele definitions, PharmCAT performs diplotype calling from a variant call format (VCF) file, and
Health Level Seven (HL7) publishes both a genomics reporting implementation guide and a clinical
decision support (CDS) hook specification. The data scientist's job is to wire these together
without introducing a translation error that a clinician cannot see.

### 1.1 The failure pattern

The pattern below repeats across health systems. Each item is a default in at least one deployed
pipeline.

1. The laboratory reports a genotype ("CYP2C19 \*2 heterozygote") but not a phenotype, and the EHR
   stores the string rather than a coded observation.
2. The interface maps the string to a phenotype using a local table that was correct when the
   guideline was published in 2013 and has not been updated since.
3. The recommendation is keyed on drug only, not on drug and indication, so a neurovascular
   patient receives the acute coronary syndrome recommendation.
4. The alert fires on every order, including orders for drugs the patient is already taking
   safely, and clinicians learn to dismiss it.
5. No one records whether the alert was accepted, so the program cannot show that it changed
   anything.
6. The next guideline update arrives, and no one knows which patients were classified under the
   old rules.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Prescriber | A recommendation they cannot trust or act on | Depends on phenotype and evidence fields they did not build |
| Patient | Exposure to an avoidable adverse reaction | Cannot see the genotype or the alert |
| Laboratory | Credibility when a result is misread downstream | Does not control EHR display or terminology mapping |
| Clinical informaticist | Weeks reconciling laboratory strings to coded values | Depends on laboratory vocabulary and EHR build |
| Pharmacist | Time adjudicating alerts that fire without context | Depends on rule specificity and override reasons |
| Program sponsor | An unmeasurable return on genotyping cost | Depends on outcome capture that no one owns |

### 1.3 The industry question

> Given a genotype, a drug, and a clinical context, what should the system do now, and how strong
> is the evidence for doing it?

Everything in this topic serves that question. The genotype is an input. The recommendation is an
output. The evidence strength is a property of the output that must travel with it. The clinical
context, particularly the indication and the alternative drugs already tried or contraindicated,
determines which output applies.

A second question sits underneath: what should the system do when the evidence does not cover the
case? CPIC guidelines routinely say "no recommendation" for a phenotype-indication pair. A
pipeline that treats "no recommendation" as "no alert" and a pipeline that treats it as "normal
metabolizer" produce the same silence and different clinical meanings.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a variant, a star allele, a diplotype, a phenotype, and a recommendation, and
   identify which of the five a given data field represents.
2. **Compute** a CYP2D6 activity score from a diplotype with copy number, and **assign** a CYP2C19
   phenotype from allele-function pairs, explaining why the two genes need different rules.
3. **Build** a recommendation lookup keyed on gene, phenotype, drug, and indication, and return an
   explicit "no recommendation" rather than a default.
4. **Query** the CPIC/ClinPGx application programming interface (API) and pin the guideline
   version used in a report.
5. **Design** a CDS Hooks card for an order-sign hook, including indicator severity, source
   attribution, and a suggested alternative order.
6. **Audit** a deployed pipeline for genotype quality, phenotype distribution, alert burden, and
   override reasons, and state which diagnostic detects which failure.
7. **Explain** the difference between an FDA table section and a CPIC evidence level, and why
   inclusion in the table is not a testing mandate.
8. **Evaluate** a pre-emptive testing program with a number-needed-to-genotype calculation and a
   stated set of assumptions.

## 3. The core mental model

Pharmacogenomic decision support is a translation chain with four lossy conversions and one
delivery step. Each conversion has a different owner, a different file format, and a different
failure mode.

```text
  ORDER CONTEXT                     PATIENT DATA                      KNOWLEDGE BASE
  -------------                     ------------                      --------------
  drug, dose, indication,     -->   VCF / array calls           -->   PharmVar allele definitions
  alternatives, allergies,          depth, VAF, CNV                   CPIC guideline tables
  problem list, prior therapy       sample identity                   FDA label text
        |                                 |                                 |
        |                                 v                                 |
        |                     +-------------------------+                   |
        |                     | 1. Variant -> star allele| <----------------+
        |                     +-------------------------+
        |                                 |
        |                                 v
        |                     +-------------------------+
        |                     | 2. Star pair -> diplotype|  phase, copy number
        |                     +-------------------------+
        |                                 |
        |                                 v
        |                     +-------------------------+
        |                     | 3. Diplotype -> phenotype|  gene-specific rule
        |                     +-------------------------+
        |                                 |
        |                                 v
        |                     +-------------------------+
        +-------------------> | 4. Phenotype + drug +    |
                              |    indication -> action  |
                              +-------------------------+
                                              |
                                              v
                              +-------------------------+
                              | 5. Deliver in workflow   |  CDS Hooks / SMART on FHIR
                              |    card, order, silence  |
                              +-------------------------+
                                              |
                                              v
                              +-------------------------+
                              | 6. Capture outcome and   |
                              |    override reason       |
                              +-------------------------+
```

### 3.1 The unit of translation is the diplotype

A variant is a position, a reference allele, and an alternate allele. A star allele is a named
haplotype: a specified combination of variants, sometimes including a structural change, on one
copy of the gene. A diplotype is an unordered pair of star alleles, one per copy. The phenotype
is a function of the diplotype, not of any single variant.

This matters because the same variant can belong to several star alleles, and a star allele can
require several variants to define. CYP2D6\*2 is defined by a set of variants, one of which is also
present in other alleles. A pipeline that maps each variant to a star allele independently will
produce impossible diplotypes such as \*2/\*2 in a patient who is actually \*1/\*2.

### 3.2 Two genes, two translation mechanisms

CPIC does not use one algorithm for all genes.

**CYP2D6 uses an activity score.** Each allele carries an activity value (1.0 for normal function,
0.5 for decreased, 0.0 for no function, and 1.0 per copy for a duplication of a functional allele).
The activity score is the sum across the two copies, including copy number. Phenotype is assigned
by ranges: 0 is poor metabolizer, greater than 0 and less than 1.25 is intermediate, 1.25 to 2.25
is normal, and greater than 2.25 is ultrarapid.

**CYP2C19 uses allele-function pairs.** The 2022 clopidogrel guideline assigns phenotype from the
pair of allele functions, with categories for "likely intermediate", "likely poor", and
"indeterminate". Two increased-function alleles give ultrarapid; one increased and one normal give
rapid; two no-function alleles give poor. A sum of activity values cannot express the difference
between \*1/\*17 and \*17/\*17, which are both 2.0 under a naive sum and are rapid and ultrarapid
respectively.

The general lesson: the phenotype rule is a property of the gene and the guideline version. It is
data, not code. Store it in a table with a version identifier.

### 3.3 Recommendation, evidence, and strength are three different things

A CPIC recommendation has an action, an evidence level for the gene-drug pair, and a strength for
the specific phenotype-indication combination. The evidence level describes how good the
underlying evidence is (levels A through D). The strength describes how confident the committee
is that the action does more good than harm for that group. A strong recommendation can rest on
moderate evidence, and a moderate recommendation can rest on strong evidence, because strength
also accounts for harms, burden, and patient values.

Alert severity in the EHR is a fourth thing, chosen by the implementing organization. A strong
recommendation to avoid a drug that causes anaphylaxis is usually rendered as an interruptive
card. A strong recommendation about a dose cap for a statin may be rendered non-interruptively.
Conflating severity with strength makes every alert look the same.

### 3.4 Four quantities that are constantly confused

| Quantity | What it is | What it is not |
|---|---|---|
| Star allele | A named haplotype on one gene copy | A single variant, or a phenotype |
| Diplotype | An unordered pair of star alleles | A genotype at one position |
| Phenotype | A predicted metabolic capacity category | A measured enzyme activity |
| Recommendation | An action for a phenotype, drug, and indication | A property of the genotype alone |

A fifth confusion is common in vendor material: a "PGx risk score". There is no such score in
CPIC. Each gene-drug pair has its own evidence base, and combining them into one number destroys
the information a prescriber needs.

### 3.5 The information that must travel with a result

A genotype result without provenance is not actionable. The minimum set is: gene, reference
genome build, star allele nomenclature version (PharmVar release), diplotype, phenotype with the
rule version, guideline identifier and version, and the date the knowledge base was last
refreshed. If any of these is missing, a downstream consumer cannot tell whether the result is
current.

This is why the model below treats provenance as a field on the recommendation, not as metadata
in a document store.

```text
  genotype ---> diplotype ---> phenotype ---> recommendation
      |             |              |                  |
      |             |              |                  +-- guideline id, version, strength
      |             |              +--------------------- rule version, gene
      |             +------------------------------------ PharmVar release, phase source
      +-------------------------------------------------- build, assay, depth, VAF
```

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Pharmacogenomics (PGx) | Study of how germline genetic variation affects drug response | Used interchangeably with pharmacogenetics; the two are effectively synonyms in clinical use | None |
| Pharmacogene | A gene whose variation changes drug exposure or response | Assumed to be a drug target; most pharmacogenes are metabolizing enzymes or transporters | Gene symbol (HUGO) |
| Star allele | Named haplotype of a pharmacogene, written with an asterisk and a number | Treated as a single variant | PharmVar identifier, for example CYP2D6\*4 |
| Reference allele | The allele against which others are named, usually \*1 | Assumed to be the ancestral allele; it is a naming convention | Star allele |
| Diplotype | Unordered pair of star alleles | Written with phase when phase is unknown | Two star alleles, for example CYP2C19\*1/\*2 |
| Activity score (AS) | Sum of allele activity values across both copies, including copy number | Applied to genes that CPIC translates by allele-function pair | Dimensionless number, typically 0 to 3 or more |
| Phenotype | Standardized metabolic category | Treated as a measured enzyme activity | CPIC term: ultrarapid, rapid, normal, intermediate, poor, indeterminate |
| Ultrarapid metabolizer (UM) | Increased metabolic capacity | Equated with "needs a higher dose" for every prodrug | CPIC phenotype term |
| Poor metabolizer (PM) | Absent or near-absent metabolic capacity | Equated with "needs a lower dose" for every substrate | CPIC phenotype term |
| Phenoconversion | Change in effective phenotype caused by a drug interaction or disease | Ignored when the genotype and the observed response disagree | Phenotype category |
| Clinical Pharmacogenetics Implementation Consortium (CPIC) | International consortium that publishes peer-reviewed gene-drug guidelines | Treated as a regulator; it is not | Organization |
| Dutch Pharmacogenetics Working Group (DPWG) | European group that publishes national formulary annotations | Assumed identical to CPIC; recommendations can differ | Organization |
| ClinPGx | Integrated resource combining PharmGKB, CPIC, and PharmCAT content, launched July 2025 | Cited by its former name, PharmGKB | Web resource |
| PharmGKB | Former name of the pharmacogenomics knowledge base now part of ClinPGx | Cited as if still separate | Retired name |
| PharmVar | Curated repository of star allele definitions and nomenclature | Assumed to be a genotype caller; it is a definition source | Web resource, release version |
| PharmCAT | Open-source tool that calls star alleles and diplotypes from a VCF | Assumed to validate the sequencing data | Software |
| Variant call format (VCF) | Text format for storing variant calls | Assumed to contain phased genotypes | File format, with specification version |
| Copy number variant (CNV) | Structural change altering the number of gene copies | Missed by pipelines that read only SNPs | Copy number, integer |
| Structural variant (SV) | Rearrangement, deletion, duplication, or inversion larger than a threshold | Used loosely for CNV | Base pairs |
| rsID | Reference SNP cluster identifier | Treated as a stable definition of an allele | NCBI dbSNP identifier |
| Genotype quality (GQ) | Phred-scaled confidence in a genotype call | Confused with depth | Phred score |
| Variant allele fraction (VAF) | Fraction of reads carrying the alternate allele | Treated as fixed at 0.5 for heterozygotes in all assays | Proportion, 0 to 1 |
| Pre-emptive testing | Genotyping before a drug is needed, with results stored for later use | Confused with population screening | Program design |
| Reactive testing | Genotyping triggered by a specific prescription | Assumed equivalent in outcome to pre-emptive testing | Program design |
| Clinical decision support (CDS) | Software that presents patient-specific information at the point of care | Used to mean "alert" only; CDS includes non-interruptive display | System |
| CDS Hooks | HL7 specification for workflow-triggered synchronous CDS over HTTP | Confused with SMART on FHIR, which launches an app | Specification, version 3.0.0 |
| SMART on FHIR | Standard for launching apps against FHIR data | Assumed to include CDS triggering | Specification |
| Fast Healthcare Interoperability Resources (FHIR) | HL7 standard for exchanging health data | Assumed to be a database schema | Standard, release 4 or 5 |
| Genomics Reporting Implementation Guide | HL7 guide for representing genomic findings as FHIR Observations | Assumed to define CDS behavior | Implementation guide |
| Logical Observation Identifiers Names and Codes (LOINC) | Terminology for laboratory observations | Used for gene names, which are HGNC symbols | Code system |
| Systematized Nomenclature of Medicine Clinical Terms (SNOMED CT) | Clinical terminology | Used for pharmacogene names | Code system |
| Evidence level | CPIC grade for the gene-drug evidence base, A to D | Confused with recommendation strength | Ordinal grade |
| Recommendation strength | CPIC grade for a specific recommendation: strong, moderate, optional | Confused with evidence level or alert severity | Ordinal grade |
| Actionability | Whether a result changes a decision now | Treated as a property of the gene rather than the context | Boolean with context |
| Clinical Laboratory Improvement Amendments (CLIA) | United States laboratory certification statute | Treated as a test validation standard by itself | Regulation |
| College of American Pathologists (CAP) | Accreditation body with molecular pathology checklists | Assumed to be a government agency | Accreditation |
| Laboratory developed test (LDT) | Test designed, manufactured, and used within one laboratory | Assumed to be exempt from all oversight | Regulatory category |
| Genetic Information Nondiscrimination Act (GINA) | United States law limiting genetic discrimination in health insurance and employment | Assumed to cover life, disability, and long-term care insurance; it does not | Statute |

## 5. The mechanics, step by step

The procedure below runs from a clinical decision to a delivered intervention. It assumes the
genotype already exists; the assay design that produced it is out of scope except where it
constrains interpretation.

Each step states its input, its output, and the reason it exists. The reason matters more than
the mechanics, because every step has a shortcut that produces output and loses meaning. A step
that cannot state its failure is usually the step that fails.

| Step | Question answered | Primary artifact | Owner |
|---:|---|---|---|
| 1 | What decision are we changing? | Decision specification | Clinical informatics |
| 2 | What can this assay see? | Callable-variant manifest | Laboratory |
| 3 | What variants are present, and how confident? | Quality-controlled VCF | Bioinformatics |
| 4 | Which variants sit on the same copy? | Phased diplotype and copy number | Bioinformatics |
| 5 | Which named haplotype does each copy match? | Star allele calls | Bioinformatics |
| 6 | What is the diplotype, and how certain? | Diplotype with confidence | Bioinformatics |
| 7 | What phenotype does the diplotype imply? | CPIC phenotype term | Knowledge base |
| 8 | What is the evidence, and which version? | Evidence package | Knowledge base |
| 9 | What should be done for this drug and indication? | Action and alternatives | Clinical lead |
| 10 | How should it be presented? | Display decision | Clinical informatics |
| 11 | How does it reach the prescriber in time? | CDS response | Engineering |
| 12 | Did it work? | Audit dataset | Program owner |

### Step 1. Define the decision, the trigger, and the outcome

**Input.** A clinical workflow: for example, a cardiologist signing a clopidogrel order for a
patient after percutaneous coronary intervention (PCI).

**Output.** A written decision specification: the hook (`order-sign`), the data required, the
action the system may take, the outcome that will be measured, and the owner of each element.

**Rationale.** CDS projects fail when the decision is described as "surface PGx results" rather
than as a specific order and a specific change. The specification forces the team to name the
indication, because the same genotype produces different recommendations for acute coronary
syndrome and for stroke.

**If skipped.** The alert fires in workflows where it cannot change anything, the acceptance rate
is low, and the program concludes that pharmacogenomics has no value.

### Step 2. Establish the assay scope and what it cannot call

**Input.** Laboratory test description: targeted genotyping panel, exome, genome, or long-read
sequencing; list of interrogated variants; ability to detect copy number and structural variants.

**Output.** A callable-variant manifest per gene, plus a list of star alleles that the assay cannot
distinguish.

**Rationale.** A targeted panel that assays five CYP2D6 variants cannot distinguish \*1 from \*2,
or detect the \*5 deletion. Reporting a diplotype from such a panel requires an explicit
"limited resolution" flag. The flag is a property of the test, not of the patient.

**If skipped.** Deletion alleles are reported as wild type, ultrarapid metabolizers are reported as
normal, and the error is invisible because the output looks well formed.

### Step 3. Call variants and apply assay-specific quality control

**Input.** Reads or array intensities; reference genome; variant caller.

**Output.** A VCF with genotype, depth, genotype quality, and variant allele fraction per site.

**Rationale.** Pharmacogene loci include pseudogenes and homologous regions. CYP2D6 is adjacent to
two pseudogenes, CYP2D7 and CYP2D8, and short reads mismap between them. Quality control must be
locus-aware: a depth threshold that is adequate for a unique region is not adequate for a
paralogous region.

The blind spots differ by assay, and the manifest in step 2 exists to make them explicit.

| Assay | Typical blind spot | Consequence if unmodeled |
|---|---|---|
| Targeted genotyping panel, 5 to 20 variants | Cannot distinguish alleles that share the assayed variants; cannot detect deletions | \*1 reported instead of \*5; \*1 reported instead of \*2 |
| Short-read exome or genome | Poor mapping in paralogous regions; phase often unresolved | CYP2D6 hybrid alleles missed; cis/trans ambiguity |
| Long-read genome | Higher raw error in homopolymers unless polished | Indel-defined alleles called inconsistently |
| Targeted long-read pharmacogene panel | Limited to the captured interval | Enhancer or promoter variants outside the interval missed |
| Whole-genome with structural variant calling | Copy number resolution depends on coverage model | Duplications below the model's resolution missed |

**If skipped.** A paralogous sequence variant is called as a CYP2D6 variant, shifting the diplotype
by one functional copy.

### Step 4. Resolve phase and copy number

**Input.** VCF; read-backed phasing where available; copy number estimates from depth, probe
intensity, or long reads.

**Output.** A phased or unphased diplotype plus a copy number per allele where the gene is subject
to deletion or duplication.

**Rationale.** Phase matters when two variants in the same gene are in cis on one copy or in trans
across both. Copy number matters because \*1xN and \*2xN change the activity score. Short-read
sequencing often cannot resolve phase, in which case the result is a set of possible diplotypes,
and the pipeline must choose a reporting convention rather than invent certainty.

**If skipped.** A \*1/\*4 patient with a duplication on the \*1 copy is reported as \*1/\*4, missing
the ultrarapid phenotype.

### Step 5. Map variants to star alleles

**Input.** Quality-controlled VCF; PharmVar definitions for the gene and release.

**Output.** For each gene copy, the best matching star allele, with a match quality and a list of
unexplained variants.

**Rationale.** Star allele calling is a matching problem against a curated definition set, not a
per-variant lookup. Definitions include required variants, forbidden variants, and sometimes
structural features. A variant that is not in any definition is either a novel allele or an
error, and both require a different downstream action.

**If skipped.** The pipeline assigns the nearest known allele, which silently converts uncertainty
into a confident phenotype.

### Step 6. Assemble the diplotype and handle uncertainty

**Input.** Per-copy allele assignments; match quality; phase information.

**Output.** A diplotype string, a confidence category, and a list of unresolved features.

**Rationale.** The diplotype is the unit of translation. Its confidence must be explicit, because
a diplotype with one uncertain-function allele produces the phenotype "indeterminate", which has
its own handling rules in the 2022 CYP2C19 guideline.

The step has more than two outcomes, and each outcome routes differently.

| Diplotype outcome | Condition | Route |
|---|---|---|
| Fully resolved | Both alleles match a PharmVar definition with high confidence | Continue to step 7 |
| Resolved with likely phenotype | One allele is a decreased-function or uncertain-function allele with a known likely category | Continue to step 7 with a "likely" qualifier flag |
| Indeterminate | One or both alleles have uncertain function, or phase is unresolved in a way that changes the call | Report indeterminate; do not issue a dose recommendation; consider reflex testing |
| Novel allele | Variant combination matches no definition | Report the variant set; do not assign a star allele; refer for curation |
| Assay limit | The assay cannot interrogate a defining variant for the suspected allele | Report the limit; do not report a diplotype that the assay cannot support |

### Step 7. Translate the diplotype to a phenotype

**Input.** Diplotype; gene; guideline version.

**Output.** A CPIC phenotype term.

**Rationale.** The translation rule differs by gene. For CYP2D6, sum activity values including
copy number and apply the ranges. For CYP2C19, use the allele-function pair table from the 2022
guideline, including the "likely" and "indeterminate" categories.

CYP2D6 activity score to phenotype:

| Activity score | Phenotype | Example diplotype |
|---|---|---|
| 0 | Poor metabolizer | \*4/\*4 |
| Greater than 0 and less than 1.25 | Intermediate metabolizer | \*1/\*4 (score 1.0), \*10/\*10 (score 1.0), \*4/\*10 (score 0.5) |
| 1.25 to 2.25 | Normal metabolizer | \*1/\*1 (score 2.0), \*1/\*2 (score 2.0), \*1/\*41 (score 1.5) |
| Greater than 2.25 | Ultrarapid metabolizer | \*1/\*1xN (score 3.0), \*2x2/\*2 (score 3.0) |

CYP2C19 allele-function pairs, 2022 guideline:

| Allele functions | Phenotype | Example diplotype |
|---|---|---|
| Increased and increased | Ultrarapid metabolizer | \*17/\*17 |
| Increased and normal | Rapid metabolizer | \*1/\*17 |
| Normal and normal | Normal metabolizer | \*1/\*1 |
| Decreased and decreased, decreased and increased, or decreased and normal | Likely intermediate metabolizer | \*1/\*9, \*9/\*17 |
| Normal and no function, or increased and no function | Intermediate metabolizer | \*1/\*2, \*2/\*17 |
| Decreased and no function | Likely poor metabolizer | \*2/\*9 |
| No function and no function | Poor metabolizer | \*2/\*2, \*2/\*3 |
| Any uncertain-function allele | Indeterminate | \*1/\*12 |

### Step 8. Attach the evidence package

**Input.** Gene, drug, phenotype, guideline identifier.

**Output.** Evidence level, recommendation strength, citation, guideline version, and the date the
knowledge base was refreshed.

**Rationale.** A recommendation without provenance cannot be audited. When a guideline is updated,
the program needs to know which historical decisions were made under the previous version. Two
CPIC guidelines for the same gene can coexist for different drugs, and one can be updated while
the other is not.

**If skipped.** The organization cannot answer a regulator's or a patient's question about which
evidence supported a specific decision.

### Step 9. Resolve the recommendation using gene, phenotype, drug, and indication

**Input.** Phenotype, drug, indication, comorbidities, contraindications, current medications.

**Output.** An action, a strength, and a set of alternatives with their own contraindications.

**Rationale.** This is the step most often collapsed to a drug-only lookup. The 2022 clopidogrel
guideline differs by indication: acute coronary syndrome or PCI, non-acute-coronary cardiovascular
disease, and neurovascular disease all have separate rows. Prasugrel is an option after PCI and is
contraindicated after stroke or transient ischemic attack, so the alternative depends on the
indication as well as the phenotype.

Decision table for clopidogrel, condensed from the 2022 CPIC guideline:

| Phenotype | ACS or PCI | Non-ACS cardiovascular | Neurovascular |
|---|---|---|---|
| Ultrarapid | Standard dose, strong | No recommendation | No recommendation |
| Rapid | Standard dose, strong | No recommendation | No recommendation |
| Normal | Standard dose, strong | Standard dose, strong | Standard dose, strong |
| Likely intermediate | Avoid standard dose; prasugrel or ticagrelor, strong | No recommendation | Consider ticagrelor or ticlopidine, moderate |
| Intermediate | Avoid standard dose; prasugrel or ticagrelor, strong | No recommendation | Consider ticagrelor or ticlopidine, moderate |
| Likely poor | Avoid clopidogrel; prasugrel or ticagrelor, strong | Avoid if feasible, moderate | Avoid if feasible; another P2Y12 inhibitor, moderate |
| Poor | Avoid clopidogrel; prasugrel or ticagrelor, strong | Avoid if feasible, moderate | Avoid if feasible; another P2Y12 inhibitor, moderate |
| Indeterminate | No recommendation; treat by default and document | No recommendation | No recommendation |

The "no recommendation" cells are outputs, not gaps. They mean the evidence does not support a
genotype-specific action, so the clinician proceeds with standard care.

### Step 10. Choose the intervention type

**Input.** Recommendation, strength, urgency, workflow, and local policy.

**Output.** One of: interruptive card with a suggested order, non-interruptive information
display, silent documentation in the problem list, or no display.

**Rationale.** Interruptive alerts consume attention and are subject to alert fatigue. The choice
depends on the consequence of the error, the availability of an alternative, and whether the
prescriber can act at that moment. A strong recommendation to avoid abacavir in an HLA-B\*57:01
carrier is a candidate for an interruptive card because the harm is severe and the alternative
exists. A recommendation to cap a statin dose can be non-interruptive because the harm is
reversible and the order can be modified later.

| Consequence of error | Alternative available now | Recommended display |
|---|---|---|
| Severe, irreversible, or life-threatening | Yes | Interruptive card with one-click alternative |
| Severe, irreversible | No | Interruptive card with stop-and-consult guidance |
| Moderate, reversible | Yes | Non-interruptive inline note at order entry |
| Moderate, reversible | No | Non-interruptive note plus monitoring suggestion |
| Low, theoretical | Either | Silent documentation; no display |
| Evidence does not cover the case | Either | No display; log the gap for curation |

### Step 11. Deliver through the workflow standard

**Input.** Card content; FHIR server; hook context; prefetch templates.

**Output.** An HTTP response containing cards, each with summary, indicator, source, detail,
suggestions, and override reasons.

**Rationale.** CDS Hooks defines a discovery endpoint and an invocation contract. The service must
respond within the workflow's latency budget, typically under two seconds, which means the
genotype and phenotype must be prefetched or cached rather than assembled at order time. A card
that arrives after the order is signed has no effect.

Latency decomposes into four budgets, and a program should measure each rather than the total.

| Segment | Typical budget | What makes it exceed budget |
|---|---|---|
| Hook delivery to the CDS service | 100 ms | Network path through a proxy or a cross-region call |
| Patient and medication prefetch | 400 ms | FHIR server under load; queries without a search parameter index |
| Knowledge lookup and rule evaluation | 50 ms | Reloading the guideline table on every request |
| Card rendering and response | 100 ms | Server-side templating that calls a remote terminology service |
| Prescriber's browser rendering the card | 300 ms | A card with a large embedded app or image |

A total budget of one second leaves headroom. A service that assembles the phenotype from raw
sequence at order time will not meet it, which is why the precomputed phenotype is the artifact
that sits between the laboratory and the order.

**If skipped.** The recommendation exists in a database and never reaches a prescriber.

### Step 12. Capture the outcome and close the loop

**Input.** Alert identifier, action taken, override reason, subsequent orders, adverse events.

**Output.** An audit dataset that supports three questions: did the alert fire correctly, did the
clinician act, and did the patient benefit.

**Rationale.** Without outcome capture, the program cannot distinguish a bad recommendation from a
good recommendation that was ignored. The override reason is the most valuable field, because it
is where clinicians record the context that the rule did not model.

**If skipped.** The program cannot justify its cost, and the rule set cannot improve.

### 5.13 Testing the chain

The chain has four independent translations, and a test that only exercises the whole chain from
a VCF to a card cannot tell which translation failed. Test each boundary.

| Test | Input | Expected output | What it catches |
|---|---|---|---|
| Golden diplotype set | A table of 200 diplotypes with curated phenotypes | Exact phenotype match | Rule-table errors, boundary errors at activity score 1.25 and 2.25 |
| PharmCAT concordance | 50 real VCFs | Identical star allele and diplotype calls | Calling-logic drift from the reference implementation |
| Synthetic phase set | Variants placed in cis and in trans | Different diplotypes | Phase handling that ignores read-backed phasing |
| Copy number set | \*1/\*1, \*1/\*1xN, \*2x2/\*1 | Activity scores 2.0, 3.0, 3.0 | Copy number dropped from the score |
| Guideline regression | A frozen expected-output file per guideline version | Only intended changes appear in the diff | Silent behavior change on a knowledge base refresh |
| Negative control | A sample with no template | No call, no phenotype | Sample swap or contamination producing a confident result |

The golden diplotype set is the highest-value artifact in the whole program. It takes a day to
build, it runs in under a second, and it is the only test that will catch a rule-table edit made
by someone who has never seen the guideline.

### 5.14 Where the procedure stops

The procedure ends at a recommendation and an audit record. It does not decide whether to test,
how to consent the patient, whether to return secondary findings, or how to bill. Those are
program decisions that sit above the pipeline, and they differ by jurisdiction. The pipeline's
obligation is to make its own assumptions visible: which assay produced the call, which PharmVar
release defined the allele, which guideline version produced the recommendation, and when the
knowledge base was last refreshed.

If any of those four facts is missing from the audit record, the recommendation cannot be
reproduced, and a recommendation that cannot be reproduced cannot be defended.

## 6. Worked example

### 6.1 The case

A 68-year-old woman presents with a non-ST-elevation myocardial infarction and undergoes PCI with
a drug-eluting stent. She has no history of stroke or transient ischemic attack, weight 72 kg, and
no active bleeding. The cardiologist is about to sign an order for clopidogrel 75 mg daily.

A pre-emptive pharmacogenomic panel was performed 14 months earlier. The laboratory report states:
"CYP2C19 \*1/\*2 detected. Interpretation: intermediate metabolizer." The report was filed as a PDF
attachment. The EHR has no coded phenotype.

### 6.2 Step 1: check the genotype call before trusting the phenotype

The pipeline re-reads the VCF. At rs4244285, the defining variant for CYP2C19\*2 (a splice-site
G to A change at position 681 of the coding sequence), the call is:

| Field | Value | Expected for a true heterozygote |
|---|---|---|
| Depth | 214 reads | At least 30 reads at this locus |
| Genotype | 0/1 | 0/1 for heterozygous |
| Variant allele fraction | 0.47 | Approximately 0.5, typically 0.4 to 0.6 |
| Genotype quality | 99 | At least 20 on the Phred scale |
| Second call at the same site | None | None |

The call passes. The diplotype is \*1/\*2, with no copy number change and no uncertain allele.

### 6.3 Step 2: translate the diplotype

Using the 2022 CPIC allele-function table: \*1 is normal function, \*2 is no function, normal plus
no function is intermediate metabolizer. Note that this is an allele-function lookup, not a sum.

A pipeline using the older 2013 convention could also produce intermediate here, which is why
this case does not expose the translation bug. Section 6.5 shows a case that does.

### 6.4 Step 3: resolve the recommendation

The keys are: gene CYP2C19, phenotype intermediate metabolizer, drug clopidogrel, indication acute
coronary syndrome with PCI.

The 2022 guideline row is: avoid standard-dose clopidogrel if feasible; use prasugrel or
ticagrelor at the usual dose if not contraindicated. Strength: strong.

The alternative check:

| Alternative | Contraindication check | Result |
|---|---|---|
| Prasugrel 10 mg daily | Prior stroke or TIA is a contraindication; age 75 years or older and weight under 60 kg require caution | No prior stroke or TIA, age 68 years, weight 72 kg: acceptable |
| Ticagrelor 90 mg twice daily | Prior intracranial hemorrhage; severe hepatic impairment; strong CYP3A4 inhibitors | None documented: acceptable |
| Clopidogrel 150 mg daily | The guideline states that dose escalation is not supported; 225 mg may be considered only if clopidogrel cannot be avoided | Not preferred |

The system returns prasugrel as the suggested alternative with ticagrelor as a second option, and
records the reason: intermediate metabolizer, ACS/PCI indication, strong recommendation.

### 6.5 The arithmetic of benefit

The relative risk of 0.70 for prasugrel or ticagrelor versus clopidogrel in intermediate and poor
metabolizers comes from the meta-analysis cited in the 2022 guideline. To convert it into a
clinical effect, an absolute baseline risk is needed. Take 8.0% as the 12-month risk of
cardiovascular death, myocardial infarction, or stroke on clopidogrel for a post-PCI population
with this phenotype. This is an assumption for the calculation, not a trial result for this
patient.

Risk with the alternative:

```text
  8.0% x 0.70 = 5.6%
```

Absolute risk reduction:

```text
  8.0% - 5.6% = 2.4 percentage points
```

Number needed to treat for 12 months:

```text
  1 / 0.024 = 41.7, rounded to 42 patients
```

Now account for harm. Major bleeding on ticagrelor in a comparable post-PCI population is about
0.4 percentage points higher than on clopidogrel. Net absolute benefit:

```text
  2.4 - 0.4 = 2.0 percentage points
  1 / 0.020 = 50 patients
```

The net number needed to treat is 50, not 42. The recommendation is unchanged, but the framing
changes: this is a meaningful benefit at the population level and a modest one for an individual.

### 6.6 Diagnostic 1: genotype-level quality control

The first diagnostic that would reveal a broken analysis is the joint distribution of depth and
variant allele fraction at every pharmacogene locus, plotted per sample and per site.

A working pipeline shows heterozygous calls clustered at a variant allele fraction near 0.5 and
homozygous alternate calls near 1.0, with depth above the assay's validated minimum. A broken
pipeline shows one of three patterns:

- **Low variant allele fraction at high depth.** A heterozygous call with a fraction of 0.03 at
  1,000 reads is not a heterozygote. It is contamination, index hopping, or a mismapped paralog.
  The phenotype derived from it is wrong in the direction of the contaminating sample.
- **Adequate fraction at low depth.** A fraction of 0.5 at 6 reads is consistent with a
  heterozygote but not evidence of one. The correct output is "insufficient data", not a diplotype.
- **Site-specific collapse.** A cluster of samples with allele fractions near 0.33 or 0.67 at a
  triploid or duplicated locus suggests copy number was not modeled.

For this case the diagnostic passes: fraction 0.47, depth 214, quality 99.

### 6.7 Diagnostic 2: cohort-level phenotype distribution against expectation

The second diagnostic is the distribution of phenotypes in the tested cohort compared with the
distribution expected from published allele frequencies under Hardy-Weinberg equilibrium.

For a European-ancestry cohort, take the CYP2C19\*2 allele frequency as 0.15 and ignore other
alleles for the check. Expected genotype frequencies:

```text
  *1/*1 = (0.85)^2 = 0.7225
  *1/*2 = 2 x 0.85 x 0.15 = 0.2550
  *2/*2 = (0.15)^2 = 0.0225
```

In 1,000 patients, the expected counts are 722.5, 255.0, and 22.5. A pipeline that reports 520
\*1/\*1, 360 \*1/\*2, and 120 \*2/\*2 has a genotype distribution that is impossible under the
stated allele frequency. The likely cause is a calling error that converts one genotype class
into another, for example treating \*2/\*17 as \*2/\*2 because \*17 is not in the panel. The
phenotype distribution is then skewed toward poor metabolizers, and every one of those patients
receives an avoidable alternative-drug recommendation.

A second cohort check is the indeterminate rate. If the pipeline reports zero indeterminate
results in a cohort of 5,000 patients, uncertain-function alleles are being silently assigned a
function value. The correct rate is not zero unless the assay truly interrogates every known
uncertain allele, which is rare.

### 6.8 The varied assumption

Change one assumption: the patient had a lacunar stroke two years earlier, and the current
presentation is an acute coronary syndrome. The genotype and phenotype are unchanged. The
indication set now includes a history that makes prasugrel contraindicated.

Under the 2022 guideline, the ACS/PCI row still says to avoid standard-dose clopidogrel and use
prasugrel or ticagrelor if not contraindicated. Prasugrel is now contraindicated by history, so
the remaining option is ticagrelor. If the treating indication were neurovascular rather than
ACS, the row would change to "consider ticagrelor or ticlopidine", strength moderate, and
prasugrel would remain contraindicated.

The conclusion changes in two ways: the recommended alternative changes from a choice of two to
one, and the strength of the recommendation for the neurovascular indication drops from strong to
moderate. A drug-only lookup would have returned the ACS recommendation and offered prasugrel,
which is the wrong alternative for this patient.

### 6.9 What the prescriber sees

The card that reaches the cardiologist contains five elements, and each one carries a decision
that the data team made earlier.

| Field | Value in this case | Design decision behind it |
|---|---|---|
| Summary | Clopidogrel: intermediate metabolizer, avoid standard dose | One line, names the drug first so it is readable in a list |
| Indicator | Critical | Set because the recommendation is strong and the harm is thrombotic |
| Source | CPIC 2022, CYP2C19 and clopidogrel | Names the guideline and version, not the vendor |
| Detail | Avoid standard-dose clopidogrel; use prasugrel or ticagrelor at usual dose if not contraindicated | The action, not the mechanism |
| Suggestion | Order prasugrel 10 mg daily | One click, pre-populated, still editable |

Two fields that are deliberately absent are worth naming. The card does not display the diplotype
in the summary, because a prescriber under time pressure does not need to parse \*1/\*2 to act. It
also does not display the relative risk of 0.70, because a relative risk without a baseline is
misleading when the absolute benefit depends on a baseline risk the rule does not know.

The audit record stores what the card displayed, when it displayed, whether it was overridden, and
the override reason if one was given. If the prescriber overrode the card and ordered clopidogrel
anyway, the reason is the most informative data point the program will collect that day.

## 7. Code, query, or calculation

The script below implements the translation chain for two genes and produces a CDS Hooks card. It
uses only the Python standard library, so it runs anywhere Python 3.9 or later is installed. The
allele tables are a teaching subset, not a clinical knowledge base. The output was produced by
running the script as written.

```python
"""pgx_cds.py - CPIC-style diplotype translation and CDS Hooks card generation.

Standard library only. Run: python pgx_cds.py
Data model: a tiny subset of the ClinPGx/PharmVar tables, used to show the
control flow, not to be used for patient care.
"""
import json

# ---------------------------------------------------------------- allele tables
CYP2C19_FUNCTION = {
    "*1": "normal", "*2": "no_function", "*3": "no_function", "*4": "no_function",
    "*17": "increased", "*9": "decreased", "*12": "uncertain",
}

CYP2D6_ACTIVITY = {
    "*1": 1.0, "*2": 1.0, "*35": 1.0,
    "*9": 0.5, "*10": 0.5, "*17": 0.5, "*29": 0.5, "*41": 0.5,
    "*3": 0.0, "*4": 0.0, "*5": 0.0, "*6": 0.0,
}

# CPIC 2022 assigns CYP2C19 phenotype from allele-function pairs, not from a sum.
CYP2C19_PHENOTYPE = {
    ("increased", "increased"): "Ultrarapid metabolizer",
    ("increased", "normal"): "Rapid metabolizer",
    ("normal", "normal"): "Normal metabolizer",
    ("decreased", "decreased"): "Likely intermediate metabolizer",
    ("decreased", "increased"): "Likely intermediate metabolizer",
    ("decreased", "normal"): "Likely intermediate metabolizer",
    ("increased", "no_function"): "Intermediate metabolizer",
    ("no_function", "normal"): "Intermediate metabolizer",
    ("decreased", "no_function"): "Likely poor metabolizer",
    ("no_function", "no_function"): "Poor metabolizer",
}


def cyp2c19_phenotype(allele_a, allele_b):
    f = tuple(sorted((CYP2C19_FUNCTION[allele_a], CYP2C19_FUNCTION[allele_b])))
    if "uncertain" in f:
        return "Indeterminate"
    return CYP2C19_PHENOTYPE[f]


def cyp2d6_activity(allele_a, allele_b, copies_a=1, copies_b=1):
    return CYP2D6_ACTIVITY[allele_a] * copies_a + CYP2D6_ACTIVITY[allele_b] * copies_b


def cyp2d6_phenotype(score):
    if score == 0.0:
        return "Poor metabolizer"
    if score < 1.25:
        return "Intermediate metabolizer"
    if score <= 2.25:
        return "Normal metabolizer"
    return "Ultrarapid metabolizer"


# ------------------------------------------------------- recommendation tables
# (phenotype, indication) -> (action, strength)
CLOPIDOGREL = {
    ("Normal metabolizer", "acs_pci"): ("Clopidogrel 75 mg/day if selected", "Strong"),
    ("Rapid metabolizer", "acs_pci"): ("Clopidogrel 75 mg/day if selected", "Strong"),
    ("Ultrarapid metabolizer", "acs_pci"): ("Clopidogrel 75 mg/day if selected", "Strong"),
    ("Intermediate metabolizer", "acs_pci"):
        ("Avoid standard-dose clopidogrel; use prasugrel or ticagrelor", "Strong"),
    ("Likely intermediate metabolizer", "acs_pci"):
        ("Avoid standard-dose clopidogrel; use prasugrel or ticagrelor", "Strong"),
    ("Poor metabolizer", "acs_pci"):
        ("Avoid clopidogrel; use prasugrel or ticagrelor", "Strong"),
    ("Likely poor metabolizer", "acs_pci"):
        ("Avoid clopidogrel; use prasugrel or ticagrelor", "Strong"),
    ("Normal metabolizer", "neurovascular"): ("Clopidogrel 75 mg/day if selected", "Strong"),
    ("Intermediate metabolizer", "neurovascular"):
        ("Consider ticagrelor or ticlopidine; prasugrel contraindicated", "Moderate"),
    ("Likely intermediate metabolizer", "neurovascular"):
        ("Consider ticagrelor or ticlopidine; prasugrel contraindicated", "Moderate"),
    ("Poor metabolizer", "neurovascular"):
        ("If feasible avoid clopidogrel; consider another P2Y12 inhibitor", "Moderate"),
    ("Likely poor metabolizer", "neurovascular"):
        ("If feasible avoid clopidogrel; consider another P2Y12 inhibitor", "Moderate"),
    ("Normal metabolizer", "non_acs"): ("Clopidogrel 75 mg/day if selected", "Strong"),
    ("Poor metabolizer", "non_acs"):
        ("If feasible avoid clopidogrel; use prasugrel or ticagrelor", "Moderate"),
}

CODEINE = {
    "Ultrarapid metabolizer": ("Avoid codeine; consider a non-tramadol opioid", "Strong"),
    "Poor metabolizer": ("Avoid codeine because of likely lack of analgesia", "Strong"),
    "Normal metabolizer": ("Use label-recommended age- or weight-specific dosing", "Strong"),
    "Intermediate metabolizer": ("Use label dosing; monitor response", "Moderate"),
}


def cds_card(drug, phenotype, action, strength, source):
    """Build a CDS Hooks 3.0.0 order-sign card."""
    indicator = "critical" if strength == "Strong" and action.startswith("Avoid") else "warning"
    return {
        "hook": "order-sign",
        "summary": f"{drug}: {phenotype} - {action}",
        "indicator": indicator,
        "source": {"label": source},
        "detail": f"CPIC recommendation strength: {strength}.",
        "suggestions": [{"label": action, "actions": []}],
    }


# ------------------------------------------------------------------- test cases
CASES = [
    ("P-001", "CYP2C19", "*1", "*2", "clopidogrel", "acs_pci"),
    ("P-002", "CYP2C19", "*17", "*17", "clopidogrel", "acs_pci"),
    ("P-003", "CYP2C19", "*2", "*2", "clopidogrel", "neurovascular"),
    ("P-004", "CYP2C19", "*1", "*12", "clopidogrel", "acs_pci"),
    ("P-005", "CYP2D6", "*1", "*1", "codeine", None),
    ("P-006", "CYP2D6", "*1", "*4", "codeine", None),
    ("P-007", "CYP2D6", "*4", "*4", "codeine", None),
]


def run_case(case):
    pid, gene, a1, a2, drug, indication = case
    if gene == "CYP2C19":
        phenotype = cyp2c19_phenotype(a1, a2)
        key = (phenotype, indication)
        action, strength = CLOPIDOGREL.get(key, ("No CPIC recommendation for this pair", "n/a"))
    else:
        score = cyp2d6_activity(a1, a2)
        phenotype = cyp2d6_phenotype(score)
        action, strength = CODEINE.get(phenotype, ("No CPIC recommendation", "n/a"))
    return pid, gene, f"{a1}/{a2}", phenotype, drug, action, strength


def main():
    print("PATIENT  GENE      DIPLOTYPE  PHENOTYPE                      DRUG         STRENGTH")
    for case in CASES:
        pid, gene, diplo, pheno, drug, action, strength = run_case(case)
        print(f"{pid:<8} {gene:<9} {diplo:<10} {pheno:<30} {drug:<12} {strength}")

    print("\nCDS HOOKS CARD (case P-001)")
    _, _, _, pheno, _, action, strength = run_case(CASES[0])
    print(json.dumps(cds_card("clopidogrel", pheno, action, strength, "CPIC 2022"), indent=2))

    # Number needed to genotype to prevent one confirmed abacavir reaction.
    # Assumptions: carrier prevalence 5.0%, confirmed HSR 2.7% of unscreened
    # abacavir users (PREDICT-1 control arm), screening removes the risk in carriers.
    n_patients = 1000
    prevalence = 0.05
    baseline_hsr = 0.027
    carriers = n_patients * prevalence
    prevented = n_patients * baseline_hsr
    nng = n_patients / prevented
    print(f"\nABACAVIR SCREENING ARITHMETIC")
    print(f"  patients genotyped            : {n_patients:d}")
    print(f"  HLA-B*57:01 carriers          : {carriers:.1f} (prevalence {prevalence:.1%})")
    print(f"  confirmed HSR avoided         : {prevented:.1f} per {n_patients:d} screened")
    print(f"  number needed to genotype     : {nng:.1f}")

    # Alert burden: share of orders that fire an interruptive card.
    orders = 4000
    firing = 312
    print(f"\nALERT BURDEN")
    print(f"  orders signed                 : {orders:d}")
    print(f"  interruptive cards fired      : {firing:d} ({firing / orders:.1%})")


if __name__ == "__main__":
    main()
```

Output of the run:

```text
PATIENT  GENE      DIPLOTYPE  PHENOTYPE                      DRUG         STRENGTH
P-001    CYP2C19   *1/*2      Intermediate metabolizer       clopidogrel  Strong
P-002    CYP2C19   *17/*17    Ultrarapid metabolizer         clopidogrel  Strong
P-003    CYP2C19   *2/*2      Poor metabolizer               clopidogrel  Moderate
P-004    CYP2C19   *1/*12     Indeterminate                  clopidogrel  n/a
P-005    CYP2D6    *1/*1      Normal metabolizer             codeine      Strong
P-006    CYP2D6    *1/*4      Intermediate metabolizer       codeine      Moderate
P-007    CYP2D6    *4/*4      Poor metabolizer               codeine      Strong

CDS HOOKS CARD (case P-001)
{
  "hook": "order-sign",
  "summary": "clopidogrel: Intermediate metabolizer - Avoid standard-dose clopidogrel; use prasugrel or ticagrelor",
  "indicator": "critical",
  "source": {
    "label": "CPIC 2022"
  },
  "detail": "CPIC recommendation strength: Strong.",
  "suggestions": [
    {
      "label": "Avoid standard-dose clopidogrel; use prasugrel or ticagrelor",
      "actions": []
    }
  ]
}

ABACAVIR SCREENING ARITHMETIC
  patients genotyped            : 1000
  HLA-B*57:01 carriers          : 50.0 (prevalence 5.0%)
  confirmed HSR avoided         : 27.0 per 1000 screened
  number needed to genotype     : 37.0

ALERT BURDEN
  orders signed                 : 4000
  interruptive cards fired      : 312 (7.8%)
```

What the output demonstrates, and what it does not:

- P-002 shows why a summed activity score fails for CYP2C19: \*17/\*17 is ultrarapid and \*1/\*17
  would be rapid, both with the same summed value.
- P-004 returns an explicit "no CPIC recommendation" for an indeterminate result rather than a
  default. In production this should route to curation or reflex testing, not to a card.
- P-006 shows \*1/\*4 at activity score 1.0, below the 1.25 boundary and therefore intermediate. A
  boundary error here would misclassify a large share of European patients.
- The abacavir arithmetic uses a 5.0% carrier prevalence and a 2.7% confirmed reaction rate. Both
  are stated assumptions; the true values depend on ancestry and on the definition of a confirmed
  reaction.

The script does not model drug interactions, renal or hepatic function, age, weight, or
contraindications. Those are separate inputs to the recommendation step in a production system.

## 8. Data, standards, and regulatory context

### 8.1 CPIC: what it is and what it is not

CPIC is a consortium, not a regulator. It publishes peer-reviewed guidelines that translate
genotype to phenotype to a prescribing action. Each guideline carries an evidence level for the
gene-drug pair and a strength for each recommendation.

| CPIC evidence level | Meaning | Binding status |
|---|---|---|
| A | Genetic information should be used to change prescribing; high-quality evidence | Not legally binding |
| B | Genetic information could be used to change prescribing; moderate evidence | Not legally binding |
| C | Genetic information could be used to change prescribing; low evidence | Not legally binding |
| D | Genetic information should not be used to change prescribing | Not legally binding |

Recommendation strength is a separate axis: strong, moderate, or optional. CPIC strength is
defined in its guideline methodology and is not the same as the GRADE (Grading of Recommendations
Assessment, Development and Evaluation) strength used in other clinical guidelines, although the
concepts are related.

What CPIC constrains: the content of a genotype-guided recommendation, when a program chooses to
follow CPIC. What it does not constrain: laboratory validation, reimbursement, EHR design, or
whether testing should be offered at all.

### 8.2 ClinPGx, PharmVar, and PharmCAT

PharmGKB launched as ClinPGx on 29 July 2025, and PharmGKB links began redirecting on 30 July
2025. CPIC and PharmCAT content are being integrated into the same resource. This matters for data
pipelines in three ways. First, field names in the CPIC database changed: `guideline.pharmgkbid`
became `clinpgxid`, `pair.pgkbcalevel` became `clinpgxlevel`, and the drug and gene identifier
columns were renamed, so a pipeline that hard-codes the old names will break at the next release.
Second, the CPIC API at `https://api.cpicpgx.org` remains a PostgREST service over the CPIC
database, so endpoints correspond to tables such as `/gene`, `/drug`, `/pair`, `/recommendation`,
`/guideline`, `/allele`, and `/diplotype`; there is no version prefix in the URL, and the release is
identified by a data version such as 1.55.0. Third, PharmVar remains the nomenclature authority for
star alleles, and ClinPGx links to it rather than duplicating definitions, so an allele name in a
report should be checked against a PharmVar release.

PharmCAT is the reference implementation that turns a VCF into star alleles, diplotypes, and
phenotypes. It is useful both as a production component and as a test oracle: if a custom pipeline
disagrees with PharmCAT on a sample, one of them is wrong, and the disagreement is the fastest way
to find the bug.

### 8.3 The FDA Table of Pharmacogenetic Associations

The FDA maintains a table of gene-drug associations in three sections. It was launched in February
2020 and has been updated repeatedly, with the most recent listed update on 10 September 2026.

| Section | What the data show | What it does not mean |
|---|---|---|
| Section 1: data support therapeutic management recommendations | Evidence that genotype alters the therapeutic approach, often reflected in labeling | Does not by itself require testing before prescribing |
| Section 2: data indicate a potential impact on safety or response | Evidence of an effect that may inform monitoring or choice | Does not establish that genotype-guided care improves outcomes |
| Section 3: data demonstrate a potential impact on pharmacokinetic properties only | Exposure changes without established clinical outcome data | Does not support a dose recommendation |

Two regulatory facts are frequently misread:

- **Inclusion is not a testing mandate.** A test is mandatory only when the FDA has approved it as
  a companion diagnostic for a specific drug, or when a label states that testing is required.
- **Section assignment does not track CPIC strength.** The FDA table and CPIC guidelines are built
  for different purposes, and a 2022 comparison found that among 39 drugs covered by both, only a
  minority had concordant gene-drug and dosing recommendations.

### 8.4 Laboratory standards

Clinical pharmacogenomic testing in the United States is performed in laboratories certified under
CLIA and usually accredited by CAP. The relevant constraints:

- **Validated variants.** Each variant in the report must be validated for the specimen type,
  including the minimum depth and the limit of detection.
- **Star allele reporting.** A laboratory that reports a star allele is asserting a haplotype
  call. If the assay cannot distinguish two alleles, the report must say so.
- **Reference material.** External quality assessment schemes, such as the College of American
  Pathologists proficiency testing program for pharmacogenomics, provide sample-level checks.
- **LDT oversight.** The FDA finalized a rule in May 2024 intended to phase out enforcement
  discretion for laboratory developed tests. Litigation left the effective status uncertain, so any
  statement about LDT oversight should be treated as time-stamped and checked before reliance.

What these standards do not do: they do not validate the downstream EHR rule, the phenotype
mapping, or the alert logic. A laboratory can be fully compliant and still hand a health system a
diplotype that the health system then misreads.

### 8.5 Interoperability standards

**FHIR Genomics Reporting Implementation Guide.** The guide represents a genomic report as a
`DiagnosticReport` grouping `Observation` resources for variants, haplotypes, genotypes,
therapeutic implications, and recommended actions. It is Observation-centric, with `hasMember` and
`derivedFrom` links. The historical `MolecularSequence` resource has been de-emphasized in later
FHIR versions, so a new build should follow the implementation guide rather than the older
resource.

**CDS Hooks 3.0.0.** The specification defines a discovery endpoint at `/cds-services`, an
invocation with a hook name, hook instance, FHIR server, and context, and a response of cards. Each
card has a summary, an indicator (`info`, `warning`, or `critical`), a source, optional detail,
suggestions, and links. The `order-sign` hook is the usual trigger for a prescribing decision
because the order is complete but not yet committed.

**SMART on FHIR.** Used when the card needs to launch an app for a richer workflow, such as a
pharmacist review screen. It is not a substitute for CDS Hooks; the two solve different problems.

### 8.6 Privacy, discrimination, and what the standards leave open

In the United States, GINA prohibits genetic discrimination in health insurance and employment.
It does not cover life insurance, disability insurance, or long-term care insurance. A
pharmacogenomic result is also protected health information under HIPAA, but downstream uses such
as underwriting are governed by other rules that vary by state.

The standards also leave open how to display "no recommendation", how to version a rule, what
alert burden is acceptable, and how to resolve CPIC and DPWG conflicts. Those are local decisions
that belong in governance documents, not in a vendor's default configuration.

## 9. Industry practice and case evidence

### 9.1 What health systems actually deploy

Three delivery models dominate, differing in when the genotype is produced rather than in the
interpretation logic.

| Model | When the sample is collected | Strength | Weakness |
|---|---|---|---|
| Reactive, single gene | After the drug is chosen | Simple consent, clear indication, payer may cover | Result arrives after the decision; the patient is exposed in the interval |
| Pre-emptive panel, 8 to 20 genes | Before a drug is needed, often at a routine visit | Result is available at every future order | Cost sits with the first payer; variants outside the panel are absent |
| Opportunistic reuse of research or clinical genome | When a genome is sequenced for another reason | No extra assay cost | Consent, validation, and return-of-results obligations are unresolved |

The published implementations cluster in academic medical centers. St. Jude Children's Research
Hospital genotypes patients pre-emptively and returns results into the EHR before drugs are
ordered, with a focus on pediatric oncology. The IGNITE network funded multiple health systems to
implement pharmacogenomic CDS and published shared lessons on alert design and workflow fit. Mayo
Clinic's RIGHT protocol, Geisinger's program, and the European U-PGx program are further
implementations that document pre-emptive panel testing with a common report format.

The consistent finding across these programs is that the genotype is the easy part. The work is
terminology mapping, alert governance, provider education, and outcome measurement.

### 9.2 Trial evidence that changed recommendations

| Trial | Population | Design | Documented result | What it supports |
|---|---|---|---|---|
| PREDICT-1 | HIV patients starting abacavir | Screening versus standard care | Confirmed hypersensitivity 0% with screening versus 2.7% in controls | HLA-B\*57:01 screening before abacavir |
| TAILOR-PCI | Patients after PCI | Genotype-guided versus conventional antiplatelet therapy | Primary cardiovascular endpoint 4.0% versus 5.9%, hazard ratio 0.66, 95% CI 0.43 to 1.02, not statistically significant | Directional benefit; underpowered for the primary endpoint |
| POPular Genetics | ST-elevation myocardial infarction after PCI | CYP2C19-guided choice versus standard ticagrelor or prasugrel | Primary thrombotic endpoint 2.7% versus 3.9%, noninferior; bleeding lower in the guided arm | Genotype-guided de-escalation is a defensible strategy |
| CHANCE-2 | Minor stroke or high-risk transient ischemic attack in CYP2C19 loss-of-function carriers | Ticagrelor plus aspirin versus clopidogrel plus aspirin | Stroke 6.0% versus 7.6%, hazard ratio 0.77 | The neurovascular indication row in the 2022 guideline |

Two inferences are often drawn from this table, and both should be labeled as inference rather
than fact: that genotype-guided therapy reduces mortality, although the trials were powered for
composite cardiovascular endpoints, and that the benefit generalizes to all ancestries, although
the trial populations were predominantly East Asian or European.

### 9.3 The commercial landscape

Pharmacogenomic testing is sold through hospital laboratories, national reference laboratories,
and direct-to-consumer genomics companies. The channels differ in what they report. Hospital
laboratories typically report star alleles with an interpretation tied to a guideline version.
Reference laboratories often report a phenotype with the star-allele detail in an appendix.
Direct-to-consumer companies may report a genotype and an educational interpretation without a
CLIA-certified result, which means the output cannot support a prescribing decision in most
jurisdictions.

The data scientist's concern is the interface between these outputs and the EHR. A report that
arrives as a PDF forces a human to retype a phenotype, which is the single most reliable way to
introduce a transcription error into a clinical system.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Star allele called from a single variant | Diplotype distribution has too few distinct pairs | One variant is treated as sufficient to define a haplotype | Wrong diplotype for every allele sharing that variant | Call against full PharmVar definitions; compare with PharmCAT |
| Deletion allele reported as wild type | No \*5 or \*5 carriers in a large cohort | Targeted panel does not assay copy number | Ultrarapid and poor metabolizers misclassified | Add a copy number assay; flag the limitation |
| Activity score applied to CYP2C19 | \*1/\*17 and \*17/\*17 receive the same phenotype | Gene-specific rule replaced by one generic sum | Rapid and ultrarapid confused; dose advice wrong at both ends | Store the phenotype rule per gene and version |
| Activity score boundary error | Excess of normal metabolizers near score 1.25 | Comparing with greater-than instead of greater-than-or-equal | Intermediate patients treated as normal | Golden diplotype test at every boundary |
| Drug-only recommendation lookup | Same card for stroke and myocardial infarction | Indication dropped from the key | Prasugrel offered to a patient with prior stroke | Key on gene, phenotype, drug, and indication |
| "No recommendation" rendered as normal | Indeterminate results produce no card and no note | Missing key returns an empty dictionary, and the caller treats empty as normal | Uncertain patients receive standard dosing with false reassurance | Return an explicit status; test for it |
| Guideline version not pinned | Two reports for the same patient disagree | Knowledge base updated between reports | Irreconcilable record; audit failure | Store guideline identifier and version on every recommendation |
| Alert fires on already-active therapy | Override rate above 70% | Rule does not check active medications or order type | Alert fatigue; clinicians dismiss the one that matters | Suppress when the patient is stable on the drug; measure override rate |
| Phenotype stored as free text | Inconsistent strings across sites | No coded observation | Cannot compute, audit, or exchange | Use coded observations and a controlled value set |
| Sample swap | Genotype inconsistent with prior results or with sex markers | Pre-analytical error | Entire result is wrong for the patient | Compare with prior genotypes; check identity markers |
| Ancestry-specific allele absent from panel | Ultrarapid rate far below expectation in one group | Panel designed on European alleles | Systematic misclassification of one population | Include alleles at appreciable frequency in all served ancestries |
| Alert latency above budget | Cards appear after the order is signed | Phenotype assembled at order time | Zero clinical effect | Precompute and cache; measure p95 latency |
| Recommendation ignores contraindication | Suggested alternative is contraindicated | Alternatives chosen without checking the problem list | Harm from the recommendation itself | Evaluate alternatives against active problems and medications |
| No outcome capture | Cannot state acceptance or harm rates | Audit fields not implemented | Program cannot improve or justify cost | Log card identifier, action, override reason |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The assay interrogates every variant that defines the reported alleles | Star allele calling assumes the defining variants are observable | A panel that omits a defining variant or cannot detect copy number | Callable-variant manifest; concordance with a second assay |
| The reference genome build matches the allele definitions | Positions and definitions are build-specific | A pipeline mixing GRCh37 and GRCh38 coordinates | Assert build in the VCF header; reject mismatches |
| Phase can be resolved or safely assumed | Diplotype requires knowing which variants are on the same copy | Unresolved cis/trans in a compound heterozygote | Read-backed phasing rate; report ambiguity |
| The phenotype rule matches the guideline version in use | The rule is data, not a constant | Guideline update without a rule update | Version pinning; regression test on update |
| The patient's ancestry is represented in the allele frequency data | Expected distributions and some recommendations depend on it | A population with alleles absent from the reference panels | Stratify audits by ancestry; include all common alleles |
| The patient is not taking a drug that changes the effective phenotype | Genotype is interpreted as baseline capacity | Strong inhibitor or inducer, or a biologic that suppresses enzyme activity | Medication review at order time; phenoconversion flag |
| The indication is known and coded | Recommendation rows are indication-specific | Free-text indication or missing problem list entry | Require an indication code before showing the card |
| The clinician can act on the recommendation at that moment | CDS is only useful if an alternative can be ordered | No alternative on formulary; prior authorization required | Check formulary and restriction data at build time |
| Guideline recommendations are internally consistent across genes | Program logic assumes one lookup interface | Two guidelines use different phenotype vocabularies | Validate vocabulary per gene |
| A strong recommendation is worth an interruptive alert | Severity is derived from strength | High override rate shows the alert is not valued | Measure acceptance and override reasons |

## 12. Exercises

### Exercise 1. Translate five diplotypes

**Task.** For each diplotype, state the gene, the activity score where the gene uses one, the CPIC
phenotype, and the step in section 5 where a pipeline could go wrong: CYP2D6 \*1/\*4; CYP2D6
\*1/\*1xN; CYP2D6 \*10/\*10; CYP2C19 \*2/\*17; CYP2C19 \*1/\*12.

**Deliverable.** A five-row table with a one-sentence justification per row.

**Acceptance criteria.** CYP2D6 scores are 1.0, 3.0, and 1.0 with phenotypes intermediate,
ultrarapid, and intermediate. CYP2C19 \*2/\*17 is intermediate, not poor. \*1/\*12 is
indeterminate. Each justification names the rule, not just the answer.

**Time.** 25 minutes.

### Exercise 2. Build the recommendation key

**Task.** Extend the code in section 7 with at least four rows for a gene-drug pair of your choice
(for example SLCO1B1 and simvastatin, or HLA-B and abacavir). Include one row whose action is "no
recommendation" and one whose strength differs by indication.

**Deliverable.** A Python module plus a table of the rows and their sources.

**Acceptance criteria.** Every row cites a guideline and version. The "no recommendation" row
returns a distinct status, not an empty dictionary. A test demonstrates that an unknown key
produces an explicit error rather than a default action.

**Time.** 60 minutes.

### Exercise 3. Design the card for a contraindication

**Task.** Write the JSON for an `order-sign` card for a patient who is HLA-B\*57:01 positive and
has an abacavir order. Decide the indicator, the summary, the suggestion, and whether the card
should be interruptive.

**Deliverable.** A JSON document and a 200-word design note.

**Acceptance criteria.** The card validates against the CDS Hooks card schema. The source field
names the guideline and version. The design note justifies the indicator against the harm and the
availability of an alternative.

**Time.** 40 minutes.

### Exercise 4. Compute a number needed to genotype

**Task.** For a pre-emptive panel costing 300 United States dollars per patient, with a carrier
prevalence of 4.0% for a variant that causes a reaction in 3.0% of carriers, and an assumed 100%
efficacy of avoidance, compute the number needed to genotype to prevent one reaction and the cost
per reaction prevented. Then recompute with 1.5% prevalence.

**Deliverable.** The arithmetic, with units, and one sentence on how the result changes.

**Acceptance criteria.** Number needed to genotype is 833 at 4.0% prevalence and 2,222 at 1.5%.
Cost per reaction prevented is 250,000 dollars and 666,667 dollars respectively. The answer states
that the calculation ignores the benefit of avoiding non-reactive exposure and the cost of the
reaction.

**Time.** 30 minutes.

### Exercise 5. Audit an alert

**Task.** A program reports 4,000 clopidogrel orders per quarter, 312 interruptive cards, 41
accepted suggestions, and 271 overrides with the reason "patient already stable". Diagnose the
most likely design defect and propose two changes with a metric for each.

**Deliverable.** A one-page audit memo.

**Acceptance criteria.** The memo identifies the missing active-medication check, proposes
suppression for stable therapy and a separate path for new starts, and names a metric and a target
for each change, such as override rate below 40% and suggestion acceptance above 30%.

**Time.** 45 minutes.

### Exercise 6. Reconcile two guidelines

**Task.** Compare the CPIC recommendation and the FDA label text for one gene-drug pair where they
differ, and write the rule your system would implement.

**Deliverable.** A two-column comparison plus a decision paragraph.

**Acceptance criteria.** The comparison quotes or closely paraphrases both sources with dates. The
decision paragraph states which source the system follows, why, and how the other source is
surfaced to the clinician.

**Time.** 60 minutes.

### Exercise 7. Design the provenance record

**Task.** Specify the fields your system stores for every recommendation, sufficient to reproduce
it two years later after two guideline updates.

**Deliverable.** A field table with type, source, and retention period.

**Acceptance criteria.** The record includes assay, genome build, PharmVar release, diplotype,
phenotype rule version, guideline identifier and version, knowledge base refresh date, card
content, and the action taken. The retention period is justified against a stated policy.

**Time.** 45 minutes.

### Exercise 8. Critique a vendor claim

**Task.** A vendor states that its panel covers "all clinically relevant pharmacogenomic variants"
and reports a single "PGx risk category". Write a critique.

**Deliverable.** A 400-word critique with at least four specific questions the buyer should ask.

**Acceptance criteria.** The critique addresses allele coverage by ancestry, copy number
detection, the absence of a composite risk score in CPIC, guideline versioning, and the difference
between a research result and a CLIA-certified result.

**Time.** 40 minutes.

## 13. Self-check answers

**Exercise 1.** CYP2D6 \*1/\*4: activity 1.0 plus 0.0 equals 1.0, which is greater than 0 and less
than 1.25, so intermediate metabolizer. The risk point is the boundary at 1.25. CYP2D6 \*1/\*1xN:
the duplicated allele contributes 2.0, so the score is 1.0 plus 2.0 equals 3.0, which is greater
than 2.25, so ultrarapid. The risk point is copy number, which a SNP-only pipeline cannot see.
CYP2D6 \*10/\*10: 0.5 plus 0.5 equals 1.0, intermediate. The risk point is the decreased-function
value, which is often mis-set to 0.0. CYP2C19 \*2/\*17: no function plus increased function is
intermediate, not poor, because the increased-function allele partly compensates. The risk point
is a naive sum that would give 2.0 and call it normal or rapid. CYP2C19 \*1/\*12: the uncertain
allele makes the phenotype indeterminate, and the correct action is no recommendation plus a
curation or reflex path.

**Exercise 2.** A correct answer has the shape below. The important properties are the citation,
the explicit no-recommendation status, and a test for unknown keys.

```python
SLCO1B1_SIMVASTATIN = {
    "Normal function": ("Simvastatin 40 mg/day or per label", "Strong"),
    "Decreased function": ("Cap simvastatin at 20 mg/day or use an alternative statin", "Strong"),
    "Poor function": ("Avoid simvastatin; use an alternative statin", "Strong"),
}

def lookup(gene, phenotype, drug, indication=None):
    table = TABLES[(gene, drug)]
    key = (phenotype, indication) if indication else phenotype
    if key not in table:
        return {"status": "no_recommendation", "action": None, "strength": None}
    action, strength = table[key]
    return {"status": "ok", "action": action, "strength": strength}
```

**Exercise 3.** A strong answer sets the indicator to `critical`, the summary to "Abacavir:
HLA-B\*57:01 positive, do not use", and the suggestion to a documented alternative regimen. It
makes the card interruptive because the harm is severe, immune-mediated, and not reversible by
stopping the drug after a reaction begins. A weak answer uses `info` because the genotype is "just
information", which misreads the purpose of the alert.

**Exercise 4.** At 4.0% prevalence and 3.0% reaction risk, expected reactions per 1,000 patients
are 1,000 x 0.04 x 0.03 = 1.2, so the number needed to genotype is 1,000 / 1.2 = 833.3, rounded to
834. Cost per reaction prevented is 300 x 833.3 = 250,000 dollars. At 1.5% prevalence, expected
reactions are 1,000 x 0.015 x 0.03 = 0.45, number needed to genotype is 2,222.2, and cost per
reaction prevented is 666,667 dollars. The calculation is sensitive to prevalence, which is why a
program should compute it for its own population rather than adopt a vendor figure.

**Exercise 5.** The dominant defect is that the rule does not check whether the patient is already
taking clopidogrel and is stable. The 271 overrides with that reason are 87% of all overrides and
67.8% of all cards (271 / 400 = 0.6775). Suppress the card for continuation orders and route only
new starts and dose changes through the rule. Metrics: override rate, with a target below 40%,
and suggestion acceptance among firing cards, with a target above 30%. A second change is to
separate the "already stable" override reason from clinical disagreement so the two can be
analyzed apart.

**Exercise 6.** A strong answer names a real divergence. For example, CPIC recommends avoiding
codeine in CYP2D6 ultrarapid metabolizers with a strong rating, while the FDA approach is a boxed
warning and a contraindication in specific pediatric settings rather than a genotype-guided
alternative for all adults. The decision paragraph states that the system surfaces the CPIC
recommendation, labels the FDA labeling context in the detail, and does not silently merge the
two.

**Exercise 7.** Rubric, not a single answer. Full credit requires: every field has a type and a
source; the version fields are immutable identifiers rather than free text; the card content is
stored as rendered; the action taken and override reason are stored; retention is justified
against a named policy; and the record can be reconstructed without access to the original
laboratory system.

**Exercise 8.** Rubric. Full credit requires at least four of: which star alleles and which
variants define them, and whether the panel can detect copy number; the ancestry-specific allele
frequencies of the served population; the PharmVar release and the guideline version used; the
fact that CPIC does not define a composite PGx risk score; whether the result is CLIA-certified;
and what the vendor does when the evidence says "no recommendation".

## 14. Interview questions

**1. What is the difference between a star allele and a variant?**
A strong answer defines the star allele as a named haplotype that may require several variants,
and notes that a variant can belong to multiple alleles. A weak answer describes a star allele as
"a variant with an asterisk".

**2. Why does CYP2D6 use an activity score and CYP2C19 not?**
A strong answer explains that the activity score is a CPIC convention for CYP2D6 that sums allele
activity values including copy number, while the 2022 CYP2C19 guideline assigns phenotype from
allele-function pairs with categories for likely and indeterminate. A weak answer assumes one
algorithm fits all genes.

**3. A patient is CYP2C19 \*1/\*2 and is scheduled for PCI. What do you return, and with what
strength?**
A strong answer returns intermediate metabolizer, avoid standard-dose clopidogrel, use prasugrel
or ticagrelor if not contraindicated, strength strong, and checks the alternatives against the
problem list. A weak answer returns "reduce the dose", which is not the CPIC recommendation.

**4. How would you detect that a genotype-calling pipeline is broken before a clinician does?**
A strong answer names the variant allele fraction and depth distribution, the cohort phenotype
distribution against Hardy-Weinberg expectation, the indeterminate rate, and concordance with
PharmCAT. A weak answer says "validate the pipeline", without naming a diagnostic.

**5. What is the difference between CPIC evidence level and recommendation strength?**
A strong answer separates the gene-drug evidence base from the confidence that a specific action
does more good than harm, and notes that strength also reflects harms and burden. A weak answer
treats them as synonyms.

**6. Why is inclusion in the FDA table not a testing mandate?**
A strong answer explains that the table summarizes associations and labeling, and that a mandate
arises from a companion diagnostic or explicit label requirement. A weak answer says the FDA
requires testing for all listed drugs.

**7. How do you handle a "no recommendation" row?**
A strong answer returns a distinct status, suppresses the genotype-specific action, documents the
gap, and routes it to curation. A weak answer falls back to the normal-metabolizer row.

**8. What would make you choose a non-interruptive display?**
A strong answer weighs the consequence of the error, the reversibility of the harm, the
availability of an alternative at that moment, and the alert burden. A weak answer says
"interruptive alerts are more effective".

**9. How do you keep the knowledge base current?**
A strong answer describes a pinned version, a scheduled refresh with a regression test against
frozen expected outputs, a change review by a clinical owner, and a record of which patients were
classified under which version. A weak answer says "we update the JSON when CPIC publishes".

## 15. Key terms

- **Activity score** — sum of allele activity values across both gene copies, including copy
  number; used by CPIC for CYP2D6.
- **Allele function** — the categorical effect of a star allele: normal, decreased, no function,
  increased, or uncertain.
- **CDS Hooks** — HL7 specification for workflow-triggered synchronous clinical decision support.
- **ClinPGx** — integrated pharmacogenomics resource combining PharmGKB, CPIC, and PharmCAT.
- **CLIA** — United States Clinical Laboratory Improvement Amendments; laboratory certification.
- **Copy number variant** — structural change altering the number of gene copies.
- **CPIC** — Clinical Pharmacogenetics Implementation Consortium.
- **Diplotype** — unordered pair of star alleles.
- **DPWG** — Dutch Pharmacogenetics Working Group.
- **Evidence level** — CPIC grade A to D for a gene-drug evidence base.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **GINA** — Genetic Information Nondiscrimination Act.
- **Indeterminate** — phenotype assigned when allele function is uncertain.
- **Metabolizer phenotype** — standardized category from ultrarapid to poor.
- **PharmCAT** — open-source star allele and diplotype caller.
- **PharmVar** — star allele nomenclature authority.
- **Pre-emptive testing** — genotyping before a drug is needed.
- **Recommendation strength** — CPIC grade of strong, moderate, or optional for a specific action.
- **Reactive testing** — genotyping triggered by a prescription.
- **Star allele** — named haplotype of a pharmacogene.
- **Ultrarapid metabolizer** — phenotype with increased metabolic capacity.
- **Variant allele fraction** — fraction of sequencing reads carrying an alternate allele.

## 16. Sources

Accessed 26 October 2026 unless otherwise stated.

- CPIC. *CYP2C19 and Clopidogrel Therapy: 2022 Update*. https://doi.org/10.1002/cpt.2526;
  full text https://pmc.ncbi.nlm.nih.gov/articles/PMC9287492/
- CPIC. *CYP2D6, OPRM1, and COMT Genotypes and Select Opioid Therapy*, 2021.
  https://doi.org/10.1002/cpt.2149; full text https://pmc.ncbi.nlm.nih.gov/articles/PMC8249478/
- Clinical Pharmacogenetics Implementation Consortium. *SLCO1B1, ABCG2, and CYP2C9 Genotypes and
  Statin-Associated Musculoskeletal Symptoms*, 2022. https://doi.org/10.1002/cpt.2557. Full text:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9035072/
- ClinPGx. *Announcing ClinPGx*, July 2025. https://blog.clinpgx.org/pharmgkb-is-now-clinpgx/
- ClinPGx. *CPIC gene page for CYP2C19*. https://www.clinpgx.org/cpic/gene/cyp2c19
- ClinPGx. *Updates to the CPIC Database and API*. https://blog.clinpgx.org/updates-to-the-cpic-database-and-api/
- CPIC database and API source. https://github.com/cpicpgx/cpic-data
- CPIC API root. https://api.cpicpgx.org/
- PharmVar. https://www.pharmvar.org/
- PharmCAT. https://pharmcat.org/
- United States Food and Drug Administration. *Table of Pharmacogenetic Associations*, updated
  10 September 2026. https://www.fda.gov/medical-devices/precision-medicine/table-pharmacogenetic-associations
- HL7. *CDS Hooks, version 3.0.0*. https://build.fhir.org/ig/HL7/cds-hooks/en/index.html
- HL7. *Genomics Reporting Implementation Guide*. https://build.fhir.org/ig/HL7/genomics-reporting/
- Mallal, S., et al. *HLA-B\*5701 Screening for Hypersensitivity to Abacavir*. New England Journal
  of Medicine, 2008. https://doi.org/10.1056/NEJMoa0706135
- Claassens, D. M. F., et al. *A Genotype-Guided Strategy for Oral P2Y12 Inhibitors in Primary
  PCI*. New England Journal of Medicine, 2019. https://doi.org/10.1056/NEJMoa1907096
- Wang, Y., et al. *Ticagrelor in CYP2C19 Loss-of-Function Carriers with Stroke or TIA*. New
  England Journal of Medicine, 2021. https://doi.org/10.1056/NEJMoa2111749
- Pereira, N. L., et al. *Effect of Genotype-Guided Oral P2Y12 Inhibitor Selection vs Conventional
  Clopidogrel Therapy on Ischemic Outcomes After PCI* (TAILOR-PCI). JAMA, 2020.
  https://doi.org/10.1001/jama.2020.17108
- Pritchard, D., et al. *Comparison of FDA Table of Pharmacogenetic Associations and Clinical
  Pharmacogenetics Implementation Consortium Guidelines*. American Journal of Health-System
  Pharmacy, 2022. https://doi.org/10.1093/ajhp/zxac064

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`10_clinical-genomics.md`](10_clinical-genomics.md)
- Next topic: [`12_polygenic-risk-scores.md`](12_polygenic-risk-scores.md)
- Related topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Related topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Module survey: [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Laboratory exercises and datasets: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
