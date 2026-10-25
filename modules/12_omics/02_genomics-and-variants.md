# Genomics and Variants: From Reads to Clinical Interpretation

> Module 12 — Omics and Precision Medicine
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, topic 01
> Version 2026.3

## 1. Why this topic exists

A genome sequence is not a diagnosis. Between a blood sample and a clinical report sit seven
transformations — alignment, calling, filtering, annotation, evidence assembly, classification,
and sign-out — and each one can turn a true variant into a false negative or a benign
polymorphism into a pathogenic call. The industry bottleneck is no longer sequencing cost. A
human genome can be sequenced for a few hundred US dollars. The bottleneck is interpretation:
the labour, variability, and error rate of deciding what a variant means.

The scale of the problem is set by the ratio of variants to actionable findings. A typical
whole-genome sequence contains roughly 4 to 5 million single-nucleotide variants and several
hundred thousand insertions and deletions relative to the reference, of which a small number
fall in coding regions and a much smaller number are rare, damaging, and relevant to the
patient's phenotype. A diagnostic exome may return 20,000 to 50,000 coding variants after
filtering. The interpreter must reduce that to a handful of reportable findings without
discarding the causal one.

The second problem is disagreement. The same variant can be classified Pathogenic by one
laboratory and as a Variant of Uncertain Significance (VUS) by another, because the two
laboratories weighted the same evidence differently or had access to different evidence. The
American College of Medical Genetics and Genomics (ACMG) and the Association for Molecular
Pathology (AMP) published a framework in 2015 that was intended to reduce this variability,
and the Clinical Genome Resource (ClinGen) has since published specification after
specification for individual criteria. Variability has fallen but has not disappeared, and it
falls unevenly: well-studied genes have calibrated criteria; under-studied genes do not.

The third problem is class imbalance. Pathogenic variants are rare among all variants tested,
so even a highly specific classifier produces mostly false positives when applied broadly.
This is not a machine-learning curiosity. It determines whether a screening program does more
good than harm, and it is the arithmetic that most variant-interpretation software hides.

This topic covers the germline and somatic variant pipeline from raw reads to a classified and
reported variant, with the arithmetic of classification and the evidence standards behind it.
It is the technical foundation for topic 10 (clinical genomics), which covers reporting,
counselling, and the regulatory interface.

### 1.1 The failure pattern

1. A sample is sequenced without a paired normal, and a somatic caller is run in tumor-only
   mode.
2. Germline variants are reported as somatic, or somatic variants are reported as germline,
   because the mode was not recorded in the output.
3. Frequency filtering uses a database whose ancestry composition does not match the patient,
   so a common variant in the patient's population looks rare.
4. A predicted loss-of-function variant is classified Very Strong without checking whether the
   affected exon is expressed in the disease-relevant transcript.
5. An in-silico predictor is treated as strong evidence rather than supporting evidence.
6. A VUS is reported in a way that a clinician reads as actionable.
7. Nobody performs orthogonal confirmation, and the variant turns out to be a alignment
   artifact in a segmental duplication.

Steps 3 through 6 are interpretive, not technical, which is why they survive pipeline
validation. A pipeline can be 99.9% accurate at calling variants and still produce a report
that is wrong.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | A correct diagnosis, or unnecessary surveillance | Depends on the laboratory's interpretation |
| Clinician | Confidence in the report; time adjudicating a VUS | Cannot see the evidence table |
| Laboratory director | Accreditation status and turnaround time | Inherits variant-level ambiguity |
| Genetic counsellor | A clear message to deliver | Must explain a classification they did not make |
| Payer | Spend on confirmatory testing and surveillance | Sees codes, not evidence |
| Test vendor | Reimbursement and market access | Must defend a classifier across populations |
| Regulator | A device whose performance can be assessed | Receives summary performance, not per-variant logic |

### 1.3 The industry question

The module question is sharpest here:

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

For a variant, the analogue is: is this call a true variant, and does the evidence support the
clinical action it implies? Those are two questions, and the second cannot be answered if the
first is uncertain. A classification built on an artifact is worse than no classification,
because it carries the authority of a report.

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a variant from FASTQ to VCF to classification, naming the failure modes at each
   stage.
2. **Distinguish** germline, somatic, and mosaic variant calling, and state what each requires
   in sample design.
3. **Compute** variant allele fraction with a confidence interval and interpret it against
   purity, ploidy, and mosaicism.
4. **Apply** the ACMG/AMP 2015 criteria and the ClinGen point system to a variant, showing the
   arithmetic.
5. **Calculate** the positive predictive value of a variant classifier under class imbalance.
6. **Critique** the use of population frequency evidence when the patient's ancestry is under-
   represented in the reference database.
7. **Choose** an assay for a variant class: single-nucleotide variant, copy-number change,
   structural variant, repeat expansion, or mitochondrial variant.
8. **Write** a variant report that separates analytical finding, interpretation, and clinical
   action, with uncertainty stated.

## 3. The core mental model

Variant interpretation is an evidence-weighing problem with a fixed decision rule. The pipeline
produces candidates; the evidence determines direction and strength; the rule converts evidence
into a category.

```text
  Specimen (blood, tumour, saliva)
      |
      v
  Library -> Reads (FASTQ) -> Alignment (BAM/CRAM)
      |                            |
      |                            +--> coverage, mapping quality, duplicates
      v
  Variant calling (single, joint, trio, tumour-normal)
      |
      v
  Candidate variants (VCF) -- normalization, splitting, filtering
      |
      v
  Annotation: consequence, transcript, frequency, in silico, literature, databases
      |
      v
  Evidence codes (PVS1, PS1-4, PM1-6, PP1-5; BA1, BS1-4, BP1-7)
      |
      v
  Points (8, 4, 2, 1, and negatives) -> sum -> category
      |
      v
  Report: analytical finding + interpretation + action + uncertainty
```

### 3.1 Two axes: call confidence and interpretive confidence

These are constantly conflated. Call confidence is a property of the data: depth, mapping
quality, strand balance, and the probability that the alternate allele is real. Interpretive
confidence is a property of the evidence: population frequency, functional data, segregation,
and prior knowledge. A variant can be called with near-certainty and interpreted with near-
total uncertainty (a well-covered VUS), or called with low confidence and interpreted
decisively (a known pathogenic variant at 12x depth that needs confirmation).

A report that gives one confidence number for both is misleading. The correct structure is two
statements: "the variant is present, with this evidence" and "the variant means this, with
this evidence."

### 3.2 The evidence hierarchy

The ACMG/AMP framework groups evidence into pathogenic and benign, each at four strengths.

| Direction | Very strong | Strong | Moderate | Supporting |
|---|---|---|---|---|
| Pathogenic | PVS1 null variant in a LOF-mechanism gene | PS1 same amino acid change; PS2 de novo; PS3 functional; PS4 prevalence | PM1 hotspot; PM2 absent from controls; PM3 in trans; PM4 length change; PM5 novel missense; PM6 assumed de novo | PP1 segregation; PP2 missense in constrained gene; PP3 computational; PP4 phenotype; PP5 reputable source |
| Benign | BA1 population frequency above 5% | BS1 frequency above disease prevalence; BS2 healthy adult; BS3 no functional effect; BS4 lack of segregation | — | BP1 missense in truncating gene; BP2 in trans with pathogenic; BP3 in-frame in repeat; BP4 computational; BP5 alternative cause; BP6 reputable source; BP7 synonymous |

The table is a simplification of the 2015 criteria and does not include the ClinGen
modifications that changed how several codes are used. Two modifications matter enough to
state here. PM2 was downgraded in ClinGen guidance from moderate to supporting, and is
intended only as supporting evidence of pathogenicity in the absence of frequency data; it is
not a stand-alone criterion. PP3 and BP4, the computational codes, are supporting evidence by
construction and should not be promoted without calibration against functional data.

### 3.3 The decision rule

The 2015 framework is a set of combining rules. ClinGen's Bayesian reformulation gives the
same structure a probabilistic interpretation: each strength level carries an odds of
pathogenicity, and the log-odds add.

| Strength | Points (pathogenic) | Points (benign) |
|---|---:|---:|
| Very strong | 8 | -8 |
| Strong | 4 | -4 |
| Moderate | 2 | -2 |
| Supporting | 1 | -1 |

| Total points | Category |
|---|---|
| >= 10 | Pathogenic |
| 6 to 9 | Likely pathogenic |
| 0 to 5 | Uncertain significance |
| -1 to -3 | Likely benign |
| <= -4 | Benign |

The point system is a convenience, not a license to add evidence uncritically. Two pieces of
supporting evidence are not the same as one moderate piece if they derive from the same
underlying data. Correlated evidence — two computational predictors trained on the same
features, or two segregation reports from the same family — should not be counted twice.

### 3.4 Why the categories are not probabilities of disease

"Pathogenic" is a statement about the variant's role in a Mendelian condition under a
specified inheritance model. It is not the probability that this patient has disease. A
pathogenic variant in a low-penetrance gene may confer a 20% lifetime risk; the same variant
in a high-penetrance gene may confer 80%. The report states the variant classification; the
clinical interpretation states the risk, which depends on age, sex, family history, and
modifiers. Confusing the two produces both over-treatment and false reassurance.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Reference genome | The assembly to which reads are aligned | Assumed to be universal | GRCh38 (hg38); T2T-CHM13 |
| Alignment | Placement of a read on the reference | Treated as a solved problem | BAM/CRAM; MAPQ 0-60 |
| Coverage / depth | Reads covering a position | Confused with read count per sample | x (fold) |
| Mapping quality (MAPQ) | Confidence that an alignment is correct | Ignored in repeat regions | Phred scale |
| Base quality | Confidence in a called base | Confused with mapping quality | Phred (Q30 = 99.9%) |
| SNV | Single-nucleotide variant | Used loosely for any small variant | VCF record |
| Indel | Insertion or deletion, typically 1-50 bp | Compared across callers without normalization | VCF record |
| MNV | Multiple-nucleotide variant | Treated as several SNVs | VCF record |
| Structural variant (SV) | Variant >= 50 bp, including deletions, duplications, inversions, translocations | Assumed detectable by short-read WGS | VCF symbolic alleles |
| Copy-number variant (CNV) | A segment with altered copy number | Used interchangeably with SV | Copy ratio, log2 |
| Variant allele fraction (VAF) | Alternate reads divided by total reads at a locus | Read as zygosity | Proportion 0-1 |
| Genotype (GT) | 0/0, 0/1, 1/1, or missing | Assumed to encode phase | VCF GT field |
| Depth (DP) | Reads at a locus in a sample | Confused with genotype quality | Integer |
| Genotype quality (GQ) | Confidence in the genotype call | Treated as calibrated | Phred scale |
| Phasing | Assignment of variants to parental haplotypes | Assumed from read-backed phasing alone | `|` in GT; phase set |
| Joint calling | Calling variants across many samples simultaneously | Assumed equivalent to single-sample calling | gVCF + GenomicsDB |
| gVCF | Variant file with reference-confidence blocks | Assumed to be a variant-only file | VCF with `<*>` |
| Multi-nucleotide polymorphism | Adjacent SNVs inherited together | Called as separate SNVs | VCF record |
| Normalization | Left-alignment and trimming of indels | Omitted before comparison | VCF normalization |
| Consequence | Predicted effect on transcript | Treated as certain for splice variants | Sequence Ontology terms |
| Transcript | An annotated RNA product of a gene | All transcripts treated as equivalent | RefSeq/Ensembl accession |
| HGVS | Nomenclature for variants | Written from genomic coordinates only | `c.`, `p.`, `g.` prefixes |
| Population frequency | Allele frequency in a reference population | Treated as ancestry-neutral | Allele frequency; FAF |
| Filtering allele frequency (FAF) | Upper confidence bound on frequency | Confused with point estimate | Proportion |
| ClinVar | Public archive of variant interpretations | Treated as ground truth | Variation ID; review status |
| ClinGen | Consortium producing gene and variant guidance | Confused with ClinVar | Expert panel assertions |
| ACMG/AMP | 2015 variant classification framework | Treated as a regulation | Criteria codes |
| VUS | Variant of uncertain significance | Reported as if actionable | Category |
| Penetrance | Probability of phenotype given genotype | Confused with classification | Proportion by age |
| Actionability | Whether an intervention changes outcome | Confused with pathogenicity | Binary with evidence |
| Secondary finding | Medically actionable variant unrelated to the indication | Confused with incidental finding | ACMG SF list |
| Germline variant | Variant present in all cells, inherited or de novo | Assumed from blood alone | VAF near 0.5 or 1.0 |
| Somatic variant | Variant acquired in a tissue | Assumed to imply cancer | VAF depends on purity |
| Mosaic variant | Variant present in a subset of cells | Confused with low-level somatic | VAF < 0.3 in bulk |
| Tumor mutational burden (TMB) | Mutations per megabase in tumour | Compared across panels without calibration | Mutations/Mb |
| Microsatellite instability (MSI) | Length variation in repeat loci | Confused with TMB | Score or proportion |
| Driver mutation | Mutation conferring growth advantage | Inferred from frequency alone | Evidence-based |
| Passenger mutation | Mutation without growth effect | Called driver by proximity | Evidence-based |
| Oncogenicity | Somatic variant classification (AMP/ASCO/CAP tiers) | Confused with germline pathogenicity | Tier I-IV |

Three vocabulary distinctions are load-bearing.

**VAF is not zygosity.** A germline heterozygous variant has an expected VAF near 0.5, but
observed VAF depends on depth, mapping, copy number, and purity. A VAF of 0.30 in blood can be
a true heterozygote with allelic imbalance, or mosaicism, or contamination. Only orthogonal
data resolve it.

**Pathogenicity and oncogenicity are different frameworks.** Germline classification asks
whether a variant causes a Mendelian condition. Somatic classification asks whether a variant
confers a therapeutic, diagnostic, or prognostic significance in a tumour. A variant can be
oncogenic in a tumour and benign in the germline, and the same word "pathogenic" is used
loosely for both.

**Actionability is not pathogenicity.** A pathogenic variant in a gene with no intervention is
a diagnosis without a management change. A variant of uncertain significance in a highly
actionable gene may still warrant surveillance based on family history. The report should say
which of the two it is claiming.

## 5. The mechanics, step by step

The procedure below runs from specimen to signed report. It assumes short-read sequencing; the
long-read differences are noted where they change a step.

### Step 1. Fix the clinical question and the test scope

**Input:** phenotype, family history, prior testing, consent.

**Output:** a test description naming genes or regions, variant classes, and inheritance
models.

**Rationale:** The scope determines the assay, the analysis, and the reportable set. A
hereditary cancer panel, a diagnostic exome, a tumour-normal pair, and a carrier screen have
different sensitivity requirements and different reporting rules. Without a scope statement,
the analysis cannot decide what to filter out, and the report cannot decide what to include.

Consent is part of the input, not an administrative afterthought. Consent determines whether
secondary findings are returned, whether research use is permitted, and whether reanalysis is
allowed.

### Step 2. Choose the assay for the variant class

**Input:** variant classes implied by the phenotype.

**Output:** assay or assay combination.

**Rationale:** No single assay detects all variant classes. Short-read exome sequencing detects
SNVs and small indels well, CNVs moderately, and repeat expansions poorly. Long-read sequencing
detects SVs and repeat expansions but has different error modes. A negative exome does not
exclude a structural variant, a repeat expansion, or a methylation defect.

### Step 3. Library preparation and sequencing QC

**Input:** DNA quantity and quality.

**Output:** reads with quality metrics.

**Rationale:** Library complexity limits what can be called. Low input produces duplicate-heavy
libraries and uneven coverage, which causes false negatives in GC-rich regions. Report the
metrics: Q30 fraction, duplication rate, mean coverage, coverage at >= 20x, and the fraction of
the target covered at >= 20x. A mean coverage of 100x with 5% of the target at 0x is worse than
a mean of 60x with 99% at >= 20x.

### Step 4. Align to a declared reference

**Input:** FASTQ and a reference build.

**Output:** BAM/CRAM with a recorded reference version and aligner version.

**Rationale:** The reference build is a hard dependency. A VCF with GRCh37 coordinates
annotated against GRCh38 produces wrong frequencies and wrong consequences, often without an
error. Record the build in the VCF header and in the report, and convert coordinates explicitly
with a liftover tool when needed, checking the conversion rate.

### Step 5. Mark duplicates and recalibrate base qualities

**Input:** aligned reads.

**Output:** analysis-ready BAM.

**Rationale:** Duplicate marking prevents PCR duplicates from inflating VAF. Base quality score
recalibration (BQSR) reduces systematic quality errors that bias calling. Skipping these steps
is defensible for some modern callers, but the choice must be recorded, because it changes the
error profile.

### Step 6. Call variants with a design-matched strategy

**Input:** analysis-ready reads.

**Output:** raw VCF.

**Rationale:** Single-sample calling, joint calling across a cohort, trio calling, and
tumour-normal calling answer different questions. Joint calling improves sensitivity for
low-coverage sites because information is shared across samples. Trio calling enables de novo
detection and phasing. Tumour-normal calling separates somatic from germline. Tumor-only
calling cannot, and its output must never be described as germline.

### Step 7. Filter and normalize

**Input:** raw VCF.

**Output:** filtered, normalized VCF.

**Rationale:** Filters remove low-quality calls; normalization makes variants comparable across
callers and samples. Left-align and trim indels, split multi-allelic records, and apply
site-level filters for depth, quality, strand bias, and mapping quality. A hard filter on
depth below 10x will remove true variants in low-coverage regions; a soft filter preserves them
with a flag.

### Step 8. Annotate consequence and transcript

**Input:** normalized VCF.

**Output:** annotated VCF with gene, transcript, consequence, and HGVS.

**Rationale:** Consequence depends on the transcript. The same variant can be a missense in one
transcript, a synonymous change in another, and intronic in a third. Clinical annotation must
name the transcript used, usually the MANE Select transcript for the gene. Without that, HGVS
nomenclature is ambiguous and classification is unstable.

### Step 9. Annotate population frequency by ancestry

**Input:** annotated VCF.

**Output:** frequency estimates overall and by genetic ancestry, with the dataset version.

**Rationale:** Frequency evidence is only as good as the reference population's match to the
patient. gnomAD v4 contains 807,162 individuals, but its ancestry composition is not uniform,
and some populations remain under-represented. A variant absent from gnomAD is not necessarily
rare in every population; it may be absent because that population is not in the database.

### Step 10. Assemble evidence codes

**Input:** annotated variant and clinical context.

**Output:** a list of codes with the evidence behind each.

**Rationale:** Each code has a definition and, in ClinGen specifications, a required evidence
format. PVS1 requires a null variant and a gene where loss of function is the mechanism. PS3
requires a functional assay with validated performance. PP1 requires segregation data with a
stated pedigree and model. Recording the code without the evidence makes the classification
unauditable.

### Step 11. Apply the combining rule

**Input:** evidence codes.

**Output:** classification with the point total and the rule applied.

**Rationale:** The rule must be applied mechanically and then reviewed for correlated evidence.
A total of 10 points assembled from three supporting computational predictions is not the same
as 10 points from one very strong and one moderate criterion, even though the arithmetic is
identical. The point system assumes evidence independence, which is often false.

### Step 12. Confirm orthogonally when the call is uncertain or actionable

**Input:** the candidate variant.

**Output:** confirmation status, method, and result.

**Rationale:** Orthogonal confirmation catches alignment artifacts, sample swaps, and
contamination. Sanger sequencing for SNVs and indels, multiplex ligation-dependent probe
amplification (MLPA) or chromosomal microarray for CNVs, and repeat-primed PCR for expansions
are standard. For tumour-only calls, confirmation in a second sample or a second assay is
often required before a therapeutic decision.

### Step 13. Classify clinical significance and actionability separately

**Input:** classification.

**Output:** two statements: what the variant is, and what should be done.

**Rationale:** Classification is evidence-based and relatively stable; actionability depends on
guidelines, age, and local practice and changes more often. Separating them lets the report be
updated when guidelines change without reinterpreting the variant.

### Step 14. Report with uncertainty and limitations

**Input:** everything above.

**Output:** a report with the variant, HGVS, classification, evidence summary, limitations, and
recommendations.

**Rationale:** The report is the product. It must state the assay's limits (what it cannot
detect), the classification's basis (which codes), and the uncertainty (VUS, low coverage,
unconfirmed call). A report that states only "pathogenic" transfers all the uncertainty to a
clinician who cannot resolve it.

### Branch points

**Branch A — the phenotype suggests a CNV or SV.** Exome-derived CNV calling has limited
resolution and higher false-positive rates; use a chromosomal microarray for large CNVs, or
genome sequencing for balanced rearrangements.

**Branch B — the variant is in a segmental duplication or pseudogene.** Short reads map
ambiguously, and apparent variants may be paralogous sequence variants. Use long reads,
droplet digital PCR, or a locus-specific assay.

**Branch C — the patient's ancestry is absent from the frequency database.** Frequency evidence
must be downgraded or omitted, and the classification should be reported with an explicit
ancestry limitation.

**Branch D — the variant is heterozygous and phase matters.** When two variants in the same
gene could be in cis or in trans, phase determines whether the second hit exists. Read-backed
phasing works only when the variants are close enough to be covered by the same reads; trio
data or long reads resolve the rest. Reporting "compound heterozygous" without phase evidence
is an inference, not an observation.

**Branch E — the call is at the edge of the assay's validated range.** Low depth, low VAF, or a
repeat context means the analytical sensitivity is below the assay's validated limit. The
correct output is "not determined at validated sensitivity," not "absent." This distinction
determines whether a negative result can be used to rule out a diagnosis.

### QC thresholds and what they mean

No threshold is universal, but the direction of each metric is. The table records the values a
reviewer will ask about and why they matter, not a rule to apply blindly.

| Metric | Typical acceptable range | What a failure indicates | Action |
|---|---|---|---|
| Mean target coverage | >= 100x for germline exome | Low input, uneven capture | Re-sequence or report limit |
| Fraction of target at >= 20x | >= 95% | Capture bias, GC-rich dropout | Flag genes with low coverage |
| Duplication rate | < 20% for exome | Low library complexity | Exclude or model complexity |
| Q30 fraction | >= 85% | Instrument or chemistry problem | Re-run if below 75% |
| Contamination estimate | < 2% | Cross-sample carryover | Re-extract |
| Sex genotype concordance | Matches recorded sex | Sample swap or metadata error | Halt and investigate |
| Ti/Tv ratio (WGS) | 1.9-2.1 for known sites | Calling or filtering artifact | Review filters |
| Het/hom ratio | Roughly 1.5-2.0 for WGS | Contamination, CNV, inbreeding | Investigate before reporting |

### Decision table: variant class to assay

| Variant class | First-line assay | Confirmatory assay | Cannot be excluded by |
|---|---|---|---|
| SNV, coding | Exome or panel | Sanger | Nothing; exome covers coding SNVs well |
| Small indel, coding | Exome or panel | Sanger | Exome in homopolymer regions |
| CNV, multi-exon | Microarray or exome CNV | MLPA or digital PCR | Karyotype; short-read exome alone |
| Balanced translocation | Genome sequencing or karyotype | FISH | Exome; microarray |
| Repeat expansion | Repeat-primed PCR or long read | Southern blot | Exome; standard short-read genome |
| Mitochondrial variant | mtDNA sequencing at high depth | PCR-RFLP or second assay | Exome off-target coverage |
| Deep intronic variant | Genome sequencing | Minigene or RNA study | Exome |
| Methylation defect | Methylation-specific assay | Second locus | Sequence-based panel |
| Mosaic variant | Deep targeted sequencing | Second tissue | Blood-only testing |

### Edge cases

**Low VAF in blood.** A VAF of 0.15 in a blood sample may indicate mosaicism, a haematological
clone, or contamination. Diagnostic: check the VAF distribution across the batch and the
sample's sex-genotype concordance. Consequence: a germline classification applied to a mosaic
variant, with wrong recurrence risk. Mitigation: test a second tissue, state the mosaicism
possibility, and avoid Mendelian recurrence arithmetic without evidence.

**Contamination.** A low-level alternate allele present across many loci at similar VAF suggests
cross-sample contamination rather than a variant. Diagnostic: estimate contamination from
homozygous sites and from a panel of common SNPs. Consequence: false-positive variants and
false somatic calls. Mitigation: exclude or re-extract; report the contamination estimate.

**Sample swap.** Diagnostic: genotype concordance with a prior sample, sex check, and
relatedness check in a trio. Consequence: a variant attributed to the wrong patient.
Mitigation: fingerprint at accessioning and at reporting.

**Segmental duplication.** Reads from a paralog map onto the gene, producing apparent variants
at non-integer VAFs. Diagnostic: mapping quality near zero, excess heterozygous calls in one
region, and VAF clusters that do not fit 0.5 or 1.0. Consequence: false-positive pathogenic
calls in genes with pseudogenes. Mitigation: long-read or locus-specific assays.

**Homopolymer indels.** Polymerase slippage causes systematic errors at homopolymers.
Diagnostic: indels concentrated in homopolymer runs, with strand-biased support. Consequence:
false-positive frameshifts. Mitigation: orthogonal confirmation and stricter filters in these
contexts.

**Absent from gnomAD.** Absence is evidence of rarity only if the database covers the
population. Diagnostic: check the ancestry-specific subset and the coverage at the site. A site
with no coverage in gnomAD is not "absent"; it is unobserved. Consequence: PM2 applied to a
common variant in an under-represented population. Mitigation: use coverage-aware frequency
evidence and state the limitation.

**Tumor-only calling.** Without a normal, germline variants appear at VAF near 0.5 and somatic
variants at variable VAF, and the two overlap. Diagnostic: the VAF distribution is bimodal with
a germline peak; the variant is present in a matched normal if one exists. Consequence:
reporting a germline predisposition as somatic, or vice versa. Mitigation: paired normal where
possible; otherwise flag and confirm in a germline sample.

**Missing annotation.** A variant falls in a region with no transcript annotation, or the
transcript used is not expressed in the relevant tissue. Diagnostic: consequence is "intron
variant" or "non-coding transcript" for a variant expected to be coding. Consequence:
misclassification as intronic or synonymous. Mitigation: check MANE Select and tissue
expression before assigning consequence.

**Low coverage at the site of interest.** Mean coverage is adequate but the specific exon is at
5x. Diagnostic: per-base coverage at the variant and at known challenging regions. Consequence:
false negative reported as "no variant detected." Mitigation: report coverage at the gene
level, and state that absence of a call is not evidence of absence below the validated depth.

**Multi-allelic site.** Two alternate alleles at one position are split inconsistently across
callers, so the same genotype is represented as one record or two. Diagnostic: compare
normalized records from two pipelines at the same position. Consequence: double counting in
burden analyses and inconsistent VAFs. Mitigation: normalize and split before annotation.

**Phase ambiguity.** Two heterozygous variants in the same gene are reported as compound
heterozygous without phasing. Diagnostic: no phase set in the VCF and no trio data.
Consequence: a false second hit, or a missed one, with direct consequences for carrier status
and recurrence risk. Mitigation: trio sequencing, long reads, or clone-based phasing; otherwise
state the inference.

**Annotation version drift.** The same variant has different HGVS and consequence in two
annotation releases. Diagnostic: reannotation of a previously reported variant changes the
protein change. Consequence: report inconsistency across time. Mitigation: record the
annotation version and reannotate deliberately, with a change log.

### What the procedure produces

Three artifacts should exist at the end of the procedure, and each answers a different
question: the VCF answers "what was called," the evidence table answers "why this category,"
and the report answers "what should be done." A laboratory that retains only the first cannot
reproduce the second or defend the third.

The output is a variant-level evidence record: assay, coverage at the site, call quality,
frequency by ancestry, transcript, consequence, each evidence code with its source, the point
total, the classification, the confirmation status, and the limitations. That record is what
allows a second laboratory to reproduce the classification, and it is what a regulator or
accreditor will ask for.

## 6. Worked example

A 42-year-old is diagnosed with colorectal cancer at age 41, with a family history of
colorectal and endometrial cancer in two first-degree relatives. A germline panel of 84 genes
is ordered. The laboratory reports a frameshift variant in a mismatch-repair gene.

> **Note:** the gene (MLH1) and the disease association (Lynch syndrome) are real and well
> established. The specific variant and the read counts in this example are constructed for
> teaching. The arithmetic is the arithmetic a laboratory would perform; the variant is not a
> published case.

### 6.1 The call

The variant is a single-base deletion in exon 12, predicted to cause a frameshift and a
premature termination codon. Sequencing depth at the locus is 142 reads, with 61 alternate
reads.

```text
VAF = 61 / 142 = 0.430
```

The Wilson 95% confidence interval for 61 of 142 is:

```text
[0.351, 0.512]
```

The interval includes 0.5, so the observation is compatible with a heterozygous germline
variant. It also includes values as low as 0.35, which would be unusual for a clean
heterozygote and would prompt a check for allelic imbalance or mosaicism. At this depth, the
interval is wide enough that VAF alone cannot distinguish those explanations.

### 6.2 Evidence assembly

| Code | Evidence | Strength | Points |
|---|---|---|---:|
| PVS1 | Frameshift in a gene where loss of function is the established mechanism; premature termination in a well-expressed exon | Very strong | 8 |
| PM2 | Absent from gnomAD v4, with coverage confirmed at the site | Supporting (ClinGen downgrade) | 1 |
| PP3 | SpliceAI and a conservation score both suggest an effect | Supporting | 1 |
| PP1 | Segregation in one affected relative, likelihood ratio below the strong threshold | Supporting | 0 (not counted) |

```text
Total = 8 + 1 + 1 = 10  ->  Pathogenic (>= 10)
```

The PP1 evidence is present but not counted because one affected relative does not meet the
supporting threshold under the ClinGen specification. Counting it would not change the
category; recording why it was not counted is what makes the classification auditable.

### 6.3 The positive predictive value arithmetic

Suppose the laboratory's classifier is applied to every rare coding variant in the panel, not
just to variants that pass manual review, and suppose the prior probability that a randomly
chosen rare variant in this patient is truly pathogenic is 1 in 4,000. Suppose the classifier
has 95% sensitivity and 98.5% specificity.

```text
prevalence          = 0.00025
true positives      = 0.95 x 0.00025          = 0.0002375
false positives     = 0.015 x 0.99975         = 0.0149963
PPV                 = 0.0002375 / (0.0002375 + 0.0149963) = 0.0156
```

The positive predictive value is 1.6%. Even with 95% sensitivity and 98.5% specificity, 98 of
every 100 positive calls would be wrong if the classifier were applied to all rare variants
without review. This is why variant interpretation is not a classification problem to be
automated end to end: the base rate makes specificity the binding constraint, and manual
evidence review is what raises specificity to a clinically usable level.

### 6.4 Multiple-testing arithmetic at genome scale

A genome-wide association analysis of 3.2 million common variants tests each variant for
association with a trait. The Bonferroni threshold at alpha 0.05 is:

```text
0.05 / 3,200,000 = 1.56 x 10^-8
```

This is close to the conventional genome-wide significance threshold of 5 x 10^-8, which
corresponds to a slightly larger effective number of independent tests. The point is not the
exact number but the scale: at 3.2 million tests, a p-value of 10^-6 is unremarkable. Any
claim of a novel association at p = 10^-6 without replication is a claim about the tail of a
null distribution.

### 6.5 Diagnostic 1: the transcript problem

The variant is annotated against a transcript that includes exon 12. A review of the MANE
Select transcript shows that exon 12 is present, but an alternative transcript that skips exon
12 is the dominant isoform in colorectal mucosa. If the skipping isoform is the relevant one,
the frameshift does not produce a truncated protein in the disease tissue, and PVS1 does not
apply.

Removing PVS1:

```text
Total = 0 + 1 + 1 = 2  ->  Variant of uncertain significance (0-5)
```

The classification moves from Pathogenic to VUS on the basis of one annotation decision. This
is not a hypothetical edge case; transcript selection is one of the most common sources of
classification disagreement between laboratories. The diagnostic is to check, for every
LOF-based classification, that the transcript is expressed in the relevant tissue and that the
exon is not subject to physiological skipping.

### 6.6 Diagnostic 2: the frequency database mismatch

The variant was reported absent from gnomAD. A second check against the ancestry-specific
subset shows the site has a coverage of 0x in the patient's ancestral population, because that
population is under-represented in the database and the region is GC-rich.

Absence of the variant is therefore not evidence of rarity in this patient's population. PM2
should not be applied at all, and the report must state that the frequency evidence is
uninformative. Combined with the transcript problem, the classification is a VUS with two
specific evidence gaps, not a VUS of unknown reason.

### 6.7 Vary one assumption: the variant is present at 0.5% in the population

Suppose a new reference dataset shows the variant at an allele frequency of 0.005 (0.5%) in the
patient's ancestry group, with 40 heterozygotes among 4,000 individuals. The disease prevalence
of the associated syndrome is roughly 1 in 300, which is 0.0033.

BS1 applies when the allele frequency is greater than the disease prevalence and greater than
the expected maximum allele frequency for the condition. Here 0.005 exceeds 0.0033, so BS1
applies at strong strength:

```text
Total = 2 - 4 = -2  ->  Likely benign (-1 to -3)
```

The conclusion changes from Pathogenic (with PVS1) or VUS (without it) to Likely benign. One
number — a population frequency from an ancestry-matched dataset — moved the classification two
categories. This is why frequency evidence is checked before functional evidence, and why the
absence of a variant from a database is a weak statement, not a strong one.

### 6.8 Throughput arithmetic

A laboratory receives 1,200 germline panel samples per month. Each sample yields about 40 rare
coding variants after frequency filtering, of which roughly 3 require manual evidence review.
The review workload is:

```text
1,200 x 3 = 3,600 variant reviews per month
```

At 45 minutes per review, that is 2,700 hours per month, or about 17 full-time reviewers. Most
laboratories automate the 37 non-reviewed variants with a rules engine and reserve humans for
the 3. The automation is not a convenience; it is what makes the economics work.

The arithmetic also shows where investment pays. Reducing the rare-variant count from 40 to 30
by better frequency filtering cuts review volume by 25%, while reducing the review-required set
from 3 to 2 cuts it by 33%. The second lever is worth more, and it is a classification
decision, not a pipeline decision.

### 6.9 What the report says

> MLH1 c.[example] frameshift, heterozygous, VAF 0.43 (95% CI 0.35-0.51), depth 142x.
> Classification: Variant of uncertain significance. PVS1 was considered and not applied
> because the exon is skipped in the dominant mucosal isoform. PM2 was not applied because
> coverage in the patient's ancestral population is absent from the reference dataset. PP3
> supporting. Segregation data from one affected relative are insufficient to count. The assay
> does not detect structural variants or repeat expansions. Confirmatory Sanger sequencing was
> performed and the call was confirmed.

Every claim in that report has an evidence statement behind it, and every evidence statement
has a reason for its strength. That is what distinguishes a clinical report from a VCF
annotation.

## 7. Code, query, or calculation

The script below parses VCF-style records, computes VAF with a Wilson interval, applies the
ClinGen point system, and computes the positive predictive value under class imbalance. It uses
only the standard library.

### 7.1 Parse a variant record and compute VAF

```python
import math

record = {
    "chrom": "3", "pos": 37025756, "ref": "C", "alt": "CT",
    "gene": "MLH1", "consequence": "frameshift_variant",
    "depth": 142, "alt_depth": 61, "gq": 99,
}

def vaf(alt_depth, depth):
    return alt_depth / depth if depth else 0.0

def wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

v = vaf(record["alt_depth"], record["depth"])
lo, hi = wilson(record["alt_depth"], record["depth"])
print(record["gene"], f"VAF={v:.3f}", f"95% CI [{lo:.3f}, {hi:.3f}]")
print("heterozygous-compatible:", lo <= 0.5 <= hi)
```

Expected output:

```text
MLH1 VAF=0.430 95% CI [0.351, 0.512]
heterozygous-compatible: True
```

The interval check is a screening step, not a genotype test. It can rule out a clean
heterozygote when the interval excludes 0.5, but it cannot confirm one.

### 7.2 Apply the point system

```python
POINTS = {"very_strong": 8, "strong": 4, "moderate": 2, "supporting": 1}

def classify(codes):
    total = 0
    for code, strength, direction in codes:
        pts = POINTS[strength]
        total += pts if direction == "pathogenic" else -pts
    if total >= 10:
        cat = "Pathogenic"
    elif total >= 6:
        cat = "Likely pathogenic"
    elif total >= 0:
        cat = "Uncertain significance"
    elif total >= -3:
        cat = "Likely benign"
    else:
        cat = "Benign"
    return total, cat

with_pvs1 = [("PVS1", "very_strong", "pathogenic"),
             ("PM2", "supporting", "pathogenic"),
             ("PP3", "supporting", "pathogenic")]
without_pvs1 = [("PM2", "supporting", "pathogenic"),
                ("PP3", "supporting", "pathogenic")]
with_frequency = [("PM2", "supporting", "pathogenic"),
                  ("PP3", "supporting", "pathogenic"),
                  ("BS1", "strong", "benign")]

for name, codes in [("with PVS1", with_pvs1), ("without PVS1", without_pvs1),
                    ("with BS1", with_frequency)]:
    print(name, classify(codes))
```

Expected output:

```text
with PVS1 (10, 'Pathogenic')
without PVS1 (2, 'Uncertain significance')
with BS1 (-2, 'Likely benign')
```

The classifier is deterministic, which is the point: two laboratories with the same codes
should reach the same category. The hard part is assembling the codes consistently, not adding
the points.

### 7.3 Positive predictive value under class imbalance

```python
def ppv(prevalence, sensitivity, specificity):
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp)

print("prevalence 1/4000:", round(ppv(0.00025, 0.95, 0.985), 4))
print("prevalence 1/100: ", round(ppv(0.01, 0.95, 0.985), 4))
print("prevalence 1/4000, specificity 0.999:",
      round(ppv(0.00025, 0.95, 0.999), 4))
```

Expected output:

```text
prevalence 1/4000: 0.0156
prevalence 1/100:  0.3905
prevalence 1/4000, specificity 0.999: 0.1919
```

Three numbers that explain why variant interpretation is hard. At a prevalence of 1 in 4,000,
a classifier with 98.5% specificity has a PPV of 1.6%. Raising specificity to 99.9% raises PPV
to 19.2%, still below 50%. Raising prevalence to 1 in 100 raises PPV to 39.1% at the original
specificity. Specificity and prevalence dominate; sensitivity barely moves the result at low
prevalence.

### 7.4 Contamination estimate from homozygous sites

```python
# At sites where the sample is homozygous reference, contamination shows as low-level alt
homozygous_ref_sites = [
    {"depth": 120, "alt": 2},
    {"depth": 98, "alt": 1},
    {"depth": 140, "alt": 3},
    {"depth": 110, "alt": 2},
]
rates = [s["alt"] / s["depth"] for s in homozygous_ref_sites]
mean_rate = sum(rates) / len(rates)
print("per-site rates:", [round(r, 4) for r in rates])
print("mean contamination estimate:", round(mean_rate, 4),
      "(", round(mean_rate * 100, 2), "% )")
```

Expected output:

```text
per-site rates: [0.0167, 0.0102, 0.0214, 0.0182]
mean contamination estimate: 0.0166 ( 1.66 % )
```

A contamination estimate near 1.7% is within the range where a true heterozygous call is still
safe, but a somatic call at VAF 2% could be entirely contamination. The estimate should be
reported alongside low-VAF somatic calls, not buried in a QC file.

### 7.5 Assumptions and limits of this code

- The VCF parser is a dictionary, not a parser; real files need header handling, multi-allelic
  splitting, and normalization.
- The Wilson interval assumes independent reads. Reads are not independent when they derive
  from the same fragment, so the interval is optimistic.
- The point system assumes independent evidence. Correlated codes must be collapsed before
  summing, and this code cannot detect correlation.
- The PPV calculation assumes the classifier's sensitivity and specificity are calibrated in
  the population to which it is applied. They usually are not.
- Nothing here validates an assay. Analytical validation requires replicate samples, known
  truth sets, and a stated limit of detection.

## 8. Data, standards, and regulatory context

Variant interpretation sits at the intersection of a scientific framework, a nomenclature
standard, public databases, and a regulatory regime. The framework is not law; the databases
are not curated truth; the regulation does not tell you how to classify.

### 8.1 The classification framework

The ACMG/AMP 2015 guideline (Richards et al., *Genetics in Medicine*) defines 28 criteria and
five categories, with "likely" categories intended to convey greater than 90% certainty. It is
a professional consensus recommendation, not a regulation, and it explicitly states that it
requires clinical judgment.

ClinGen's Sequence Variant Interpretation working group has published two things that changed
practice:

- A Bayesian interpretation of the framework (Tavtigian et al., 2018), which showed that the
  criteria correspond to odds of pathogenicity of roughly 350 (very strong), 18.7 (strong),
  4.33 (moderate), and 2.08 (supporting), with a prior probability of 0.10.
- A naturally scaled point system (Tavtigian et al., 2020), assigning 8, 4, 2, and 1 points,
  with category thresholds of 10, 6, 0, -1, and -4.

Criterion-specific specifications followed: PVS1 decision tree, PS2/PM6, PS3/BS3, PM2, PM3,
PP1/BS4, PP3/BP4, PP5/BP6, BA1, and splicing. These are the documents a laboratory actually
applies. The 2015 paper alone is not sufficient.

### 8.2 Nomenclature and file formats

- **HGVS** nomenclature is maintained by the Human Genome Variation Society. It defines
  `g.` (genomic), `c.` (coding DNA), `p.` (protein), and `m.`/`n.` (mitochondrial) descriptions.
  A report that gives only genomic coordinates is not following HGVS for a clinical variant.
- **VCF 4.4/4.5** (GA4GH hts-specs) defines the record structure, symbolic alleles for
  structural variants, and the gVCF block representation. Version drift changes SV semantics.
- **MANE Select** provides a single transcript per gene agreed by NCBI and Ensembl, which
  removes a major source of HGVS ambiguity.
- **Sequence Ontology** provides consequence terms; **HPO** provides phenotype terms.

### 8.3 Databases and their limits

| Resource | Content | Size or scope | Limit |
|---|---|---|---|
| gnomAD v4 | Population allele frequencies | 807,162 individuals; 730,947 exomes, 76,215 genomes | Ancestry composition uneven; not a control cohort for all diseases |
| ClinVar | Submitted interpretations | About 6.9 million submitted records, 4.5 million unique variants (2026 snapshot) | Submissions vary in quality; conflicting interpretations common |
| ClinGen | Expert-panel gene and variant assertions | Gene-disease validity, dosage sensitivity, variant specifications | Coverage concentrated in well-studied genes |
| OMIM | Mendelian phenotype-gene relationships | Curated literature | Not a variant database; no frequency data |
| COSMIC | Somatic variants in cancer | Catalogue of somatic mutations | Ascertainment bias toward studied tumours |
| Human Gene Mutation Database | Published mutations | Legacy collection | Historical inclusion of non-pathogenic variants |
| gnomAD-SV | Structural variant frequencies | Companion SV callset | Short-read ascertainment limits |

The ClinVar review-status field is the most important metadata in the database and the most
often ignored. A one-star submission from a single laboratory is not equivalent to a three-star
expert-panel assertion. A report that cites "ClinVar says pathogenic" without the review status
is citing an opinion, not a consensus.

### 8.4 Regulatory context: germline

In the United States, germline testing performed in a clinical laboratory falls under CLIA.
The laboratory must validate the test's accuracy, precision, analytical sensitivity and
specificity, reportable range, and reference intervals, and must participate in proficiency
testing where available. CAP accreditation adds requirements that CLIA inspectors expect.

ISO 15189:2022 provides the international laboratory quality framework, including requirements
for point-of-care testing merged from ISO 22870. Accreditation is a laboratory-level
obligation.

Laboratory-developed tests have a contested US regulatory history. The FDA's May 2024 final
rule would have ended enforcement discretion for LDTs; the Eastern District of Texas vacated
the rule on 31 March 2025, and the FDA's September 2025 "Implementation of Vacatur" rule
restored the prior text. The current position is enforcement discretion, with the matter
remanded. Programs should not assume either device regulation or permanent discretion.

### 8.5 Regulatory context: somatic and companion diagnostics

When a somatic variant test selects patients for a therapy, it is often a companion diagnostic
and requires a device pathway. The FDA maintains a list of authorized companion diagnostics with
the associated therapeutic, biomarker, and marketing authorization. Examples authorized in 2025
include FoundationOne CDx for tovorafenib in pediatric low-grade glioma and Guardant360 CDx for
imlunestrant in ESR1-mutant breast cancer. Each authorization specifies the sample type, the
variant class, and the therapeutic label.

A laboratory that develops an in-house tumour sequencing test and uses it to select therapy is
outside that device pathway in the US under current enforcement discretion, but it still must
meet CLIA validation requirements. In the EU, IVDR 2017/746 applies to companion diagnostics
with a conformity assessment route.

### 8.6 Pharmacogenomics

ICH E15 defines terms for genomic biomarker nomenclature and ICH E16 defines the context,
structure, and format of qualification submissions for genomic biomarkers. Neither is a
variant-classification guideline. They govern how a genomic biomarker is described and
qualified for drug development, which is a different question from whether a variant is
pathogenic.

A pharmacogenomic variant is typically classified by function (poor, intermediate, normal,
rapid metabolizer) using a consensus system such as those from the Pharmacogenomics Knowledge
Base (now ClinPGx) or the Association for Molecular Pathology. The evidence standard is
drug-response evidence, not Mendelian segregation.

### 8.7 What the standards do not settle

- Which transcript to use when MANE is not expressed in the tissue of interest.
- How to weight conflicting functional assays.
- How to classify variants in genes with no established disease mechanism.
- What frequency threshold applies to a rare, low-penetrance condition.
- How to report a VUS without implying action.

These are the judgments that produce laboratory disagreement, and no standard removes them.

## 9. Industry practice and case evidence

### 9.1 What the databases show

**gnomAD v4** was released in November 2023 with 807,162 individuals: 730,947 exomes and
76,215 genomes, on GRCh38. The exome callset contains about 62.9 million SNVs and 6.2 million
indels passing all filters. About 138,000 individuals are of non-European genetic ancestry, and
the inclusion of UK Biobank increased the European proportion relative to earlier releases.
The practical consequence is that absence from gnomAD is stronger evidence for European-
ancestry patients than for others.

**ClinVar** as of the mid-2026 snapshot contains roughly 6.9 million submitted records and 4.5
million unique variants, from more than 3,400 submitters. Only a small fraction carry expert-
panel review. The database is an archive of interpretations, not a curated truth set.

**ACMG secondary findings** list v3.2 (2023) contains 81 genes, after adding CALM1, CALM2, and
CALM3 for long QT and catecholaminergic polymorphic ventricular tachycardia. The list is a
policy statement about what to report when a patient has opted in, not a statement about
pathogenicity of any particular variant.

### 9.2 What laboratories do

Documented practice from publications and guidance:

- Clinical laboratories use the ACMG/AMP framework with ClinGen specifications and report the
  criteria applied.
- Most laboratories use MANE Select transcripts for clinical annotation.
- Orthogonal confirmation is standard for reportable variants; the method depends on variant
  class.
- Tumor-only pipelines are used where a matched normal is unavailable, with explicit flagging
  and confirmatory testing before therapy selection.
- Reanalysis of negative exomes is increasingly routine, on a schedule, because annotation and
  knowledge change.

Inference: the largest source of inter-laboratory disagreement is not the combining rule but the
evidence assembly, especially PVS1 application, frequency thresholds, and the treatment of
computational predictors. Two laboratories applying the same point system to different evidence
will disagree, and the point system will not reveal why.

### 9.3 The benchmarking evidence

The Critical Assessment of Genome Interpretation (CAGI) runs blinded challenges in which
computational methods predict experimental or clinical outcomes. Across editions, methods
perform well on some variant classes and poorly on others, and performance on one gene or
assay type does not transfer automatically. The lesson is that a predictor's published
accuracy is a property of its benchmark, and a laboratory adopting it must validate it on its
own variant spectrum.

> **Note:** CAGI results are reported per challenge and per edition. A claim that "CAGI showed
> method X is best" requires naming the challenge, the edition, and the metric.

### 9.4 The class-imbalance case

Newborn screening and population carrier screening are the clearest examples of the PPV
arithmetic in practice. When a test is applied to a population with low prior probability, the
positive predictive value is low even at high specificity, and confirmatory testing is
mandatory. Programs that report screen-positive results without confirmation generate anxiety
and unnecessary intervention. The arithmetic in section 6.3 is the reason confirmatory
protocols exist.

### 9.5 What changed with paired tumour-normal testing

The shift from tumor-only to paired tumour-normal sequencing in oncology changed the false-
positive profile. Germline variants that were previously reported as somatic (or filtered out
entirely) became identifiable, and the incidental germline findings required a consent and
reporting pathway. The ACMG secondary findings list and the ESMO recommendations on germline
testing in cancer patients are responses to that shift. The data-science consequence is that
the same variant record now needs two classification fields, germline and somatic, and a rule
for which one the report is asserting.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reference build mismatch | Coordinates off by megabases; frequency lookup empty | GRCh37 VCF annotated as GRCh38 | Wrong gene, wrong frequency | Record build in header; liftover with check |
| Transcript mismatch | Consequence differs between annotations | Non-MANE transcript used | Wrong HGVS; PVS1 misapplied | Pin MANE Select; state tissue expression |
| PVS1 overuse | Every frameshift called very strong | Exon skipping or last-exon escape ignored | Over-classification to pathogenic | Apply ClinGen PVS1 decision tree |
| PM2 misuse | PM2 used as stand-alone moderate | Absence from gnomAD treated as rarity | VUS shifted toward pathogenic | Downgrade to supporting; coverage-aware |
| PP3 over-reliance | Several predictors counted separately | Correlated computational evidence | Double counting | Count PP3 once; calibrate against function |
| Tumor-only germline confusion | VAF near 0.5 in tumour | No matched normal | Germline reported as somatic | Paired normal; flag; confirm |
| Sample swap | Sex or genotype discordance | Accessioning or plating error | Variant attributed to wrong patient | Fingerprint at accession and report |
| Contamination | Low-level alt at many homozygous sites | Cross-sample carryover | False somatic calls | Estimate contamination; re-extract |
| Segmental duplication | MAPQ near 0; VAF not near 0.5 or 1.0 | Paralog mapping | False pathogenic call | Long read or locus-specific assay |
| Homopolymer error | Indels clustered in repeats | Polymerase slippage | False frameshift | Orthogonal confirmation; stricter filters |
| Low coverage site | Gene mean adequate, exon at 5x | Uneven capture | False negative | Per-base coverage; report limit |
| Ancestry mismatch | Frequency absent in one population | Database under-representation | PM2 applied wrongly | Ancestry-specific lookup; state limit |
| VUS reported as actionable | Report language implies management | Category conflation | Overtreatment or anxiety | Separate classification and action |
| Missing orthogonal confirmation | Single-platform call | Artifact not excluded | False report | Confirm reportable variants |
| Annotation drift | Reannotated variant changes protein | Release change | Inconsistent reports over time | Version annotation; change log |

Standing rules:

**If the VCF header lacks a reference build, stop.** Every coordinate downstream is unverified.

**If a classification has no evidence table, stop.** The category is not reproducible, and the
laboratory cannot defend it.

**If the positive predictive value is below 50%, stop and ask what confirmatory step exists.**
A test whose positives are mostly wrong needs a second test, not a better threshold.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The reference build is correct and declared | Coordinates and frequencies depend on it | Silent build mismatch | Header check; liftover concordance |
| The transcript is the clinically relevant one | Consequence and HGVS depend on it | Tissue-specific isoform | Expression data; MANE review |
| gnomAD represents the patient's ancestry | Frequency filtering | Under-represented populations | Ancestry-specific subset; coverage check |
| VAF reflects zygosity | Genotype and mosaicism calls | Allelic imbalance, contamination, CNV | Depth, second tissue, orthogonal assay |
| Reads map uniquely | Variant calling in repeats | Segmental duplication, pseudogene | MAPQ distribution; long reads |
| Evidence codes are independent | Point system additivity | Correlated computational or segregation data | Collapse correlated codes |
| Functional assays are validated | PS3/BS3 strength | Assay not calibrated for the gene | Assay validation data; replicate |
| Classification equals clinical risk | Clinical action | Penetrance, modifiers, age | Risk models; family data |
| The assay detects the variant class | Negative results | CNV, SV, repeat expansion missed | State assay scope; orthogonal testing |
| Databases are current | Frequency and interpretation | Version lag | Record access date and version |
| Orthogonal confirmation is available | False-positive control | No second sample or assay | Confirm before reporting |

The deepest limit is the last one: a classification is a statement about evidence available at
a point in time. Reanalysis changes classifications, and a report that does not carry a version
and a date cannot be reconciled with a later one.

## 12. Exercises

**Exercise 1 — Pipeline trace (recall, 25 minutes).**
Draw the path from a blood sample to a classified germline variant. At each of the seven stages,
name one failure that would produce a false-positive report and one that would produce a false
negative.
*Deliverable:* a seven-row table.
*Criteria:* one failure per direction per stage; each failure has a detection method.
*Time:* 25 minutes.

**Exercise 2 — VAF and zygosity (calculation, 30 minutes).**
A variant has 38 alternate reads at a depth of 96 in blood. Compute the VAF and the Wilson 95%
confidence interval. State whether the observation is compatible with a heterozygous germline
variant, and name two alternative explanations that the interval admits.
*Deliverable:* arithmetic and a three-sentence interpretation.
*Criteria:* VAF to three decimals; interval correct; two alternatives named with a test for
each.
*Time:* 30 minutes.

**Exercise 3 — Classification arithmetic (application, 60 minutes).**
A missense variant in a gene with an established LOF mechanism has: PM2 (supporting), PP3
(supporting), PS3 (strong, from a validated functional assay), and PP1 (supporting, from two
affected relatives). Apply the point system and state the category. Then recompute under the
assumption that the functional assay was performed at a non-physiological temperature and
should be downgraded to supporting.
*Deliverable:* two point totals, two categories, and a one-paragraph justification.
*Criteria:* arithmetic correct; the downgrade rationale is explicit; the final category follows
the threshold table.
*Time:* 60 minutes.

**Exercise 4 — PPV calculation (calculation, 30 minutes).**
A carrier screen has 99% sensitivity and 99.5% specificity, and the carrier prevalence in the
tested population is 1 in 250. Compute the PPV. Then compute it for a population with
prevalence 1 in 10,000. State the operational implication for each.
*Deliverable:* two calculations and a two-sentence implication.
*Criteria:* both PPVs correct to three decimals; the implication names a confirmatory step.
*Time:* 30 minutes.

**Exercise 5 — Critique a report (critique, 45 minutes).**
You are given a report that states: "Pathogenic variant detected. Patient should undergo
surveillance." The report lists no transcript, no evidence codes, no depth, and no confirmation
method. Write a critique naming four defects and the consequence of each.
*Deliverable:* 400-word critique.
*Criteria:* four defects, each tied to a specific consequence; at least one is a
classification defect and one is an analytical defect.
*Time:* 45 minutes.

**Exercise 6 — Assay selection (design, 50 minutes).**
A child has developmental delay and a normal exome. The differential includes a repeat
expansion disorder and a large deletion. Design the next two tests and state what each can and
cannot detect.
*Deliverable:* a two-test plan with a decision rule for ordering.
*Criteria:* variant class matched to assay; limitations stated for each; no test justified by
"it is comprehensive."
*Time:* 50 minutes.

**Exercise 7 — Frequency evidence (application, 40 minutes).**
A variant is absent from gnomAD overall and from the ancestry-matched subset, but the site has
0x coverage in that subset. Write the frequency evidence statement for the report, including
whether PM2 applies.
*Deliverable:* a three-sentence evidence statement.
*Criteria:* distinguishes absence from unobserved; states the PM2 decision; names the
limitation.
*Time:* 40 minutes.

**Exercise 8 — Design a validation study (design, 75 minutes).**
Design an analytical validation for a 100-gene germline panel. Specify samples, truth sets,
metrics, acceptance criteria, and the limit of detection for VAF.
*Deliverable:* a validation protocol outline.
*Criteria:* known-truth samples named; sensitivity and specificity defined per variant class;
LoD stated with depth; reproducibility and contamination addressed.
*Time:* 75 minutes.

## 13. Self-check answers

**Answer 1 (pipeline trace).** Alignment: a mismatch of reference build produces false positives
(wrong-coordinate calls) and false negatives (true variants not called); detection is the
liftover concordance check. Calling: low depth produces false negatives; a noisy region
produces false positives; detection is per-base coverage and orthogonal confirmation.
Annotation: a wrong transcript produces misclassified variants; detection is MANE review.
Frequency: an under-represented database produces false PM2 evidence; detection is the
ancestry-specific coverage check. Classification: correlated evidence produces
over-classification; detection is an evidence audit. Confirmation: no orthogonal assay leaves
artifacts undetected; detection is a second method. Reporting: missing limitations produces
over-interpretation; detection is report review against a checklist.

**Answer 2 (VAF).**

```text
VAF = 38 / 96 = 0.396
Wilson 95% CI = [0.304, 0.495]
```

The interval includes 0.5 but also includes 0.30, so it is compatible with a heterozygous
germline variant and also with mosaicism at roughly 30-40%, or with contamination at that
level, or with allelic imbalance from a copy-number change. Distinguishing them requires a
second tissue for mosaicism, a contamination estimate from homozygous sites, and copy-number
data for allelic imbalance.

**Answer 3 (classification).**

```text
PM2 1 + PP3 1 + PS3 4 + PP1 1 = 7  ->  Likely pathogenic
```

After downgrading PS3 to supporting (1 point): 1 + 1 + 1 + 1 = 4, which is in the 0-5 range and
therefore Uncertain significance. The change is not a rounding issue; it is a statement that a
functional assay performed outside physiological conditions cannot carry strong weight. If the
assay is the only functional evidence, the variant cannot be classified above VUS without
additional data.

**Answer 4 (PPV).**

```text
prevalence 1/250 = 0.004
TP = 0.99 x 0.004 = 0.00396
FP = 0.005 x 0.996 = 0.00498
PPV = 0.00396 / (0.00396 + 0.00498) = 0.443

prevalence 1/10,000 = 0.0001
TP = 0.99 x 0.0001 = 0.000099
FP = 0.005 x 0.9999 = 0.0049995
PPV = 0.000099 / (0.000099 + 0.0049995) = 0.0194
```

At 1 in 250, 44% of positives are true; at 1 in 10,000, 1.9% are. The operational implication
is that confirmatory testing is mandatory in the low-prevalence setting and still advisable in
the higher-prevalence setting.

**Answer 5 (report critique).** Four defects: (1) no transcript is named, so the HGVS and
consequence are ambiguous and the classification is not reproducible; (2) no evidence codes are
given, so the classification cannot be audited or challenged; (3) no depth or quality metrics
are reported, so the analytical confidence is unknown and a low-VAF artifact cannot be excluded;
(4) no confirmation method is stated, so an alignment artifact in a duplicated region could be
the entire finding. Consequence of each: non-reproducible classification, unverifiable
category, unknown false-positive risk, and possible wrong patient management. A fifth defect is
the leap from classification to management without an actionability statement, which conflates
two different evidence standards.

**Answer 6 (assay selection).** For a repeat expansion: repeat-primed PCR or long-read
sequencing, which detects expansion size; it cannot detect a point mutation causing the same
phenotype. For a large deletion: chromosomal microarray or MLPA, which detects exon-level copy
number; it cannot detect a balanced rearrangement or a point mutation. Order both if the
phenotype fits both classes, because a negative result from one does not exclude the other.
The decision rule should be phenotype-first: expansion disorder features (for example,
progressive cerebellar signs) justify the expansion test first.

**Answer 7 (frequency statement).** "The variant is absent from gnomAD v4 overall and from the
ancestry-matched subset; however, the site has 0x coverage in that subset, so absence is
uninformative for this population. PM2 was not applied. The frequency evidence cannot
distinguish a rare variant from an unobserved one, and this limitation is carried into the
classification." The statement distinguishes absence from unobserved, which is the whole point.

**Answer 8 (validation outline).** Samples: at least 20 known-positive samples per variant
class (SNV, indel, CNV), 20 negative samples, and a reference material such as a Genome in a
Bottle sample. Metrics: per-variant-class sensitivity, specificity, positive predictive value,
reproducibility across runs and operators, and limit of detection. Acceptance criteria:
sensitivity >= 95% for SNVs and indels at >= 20x, specificity >= 99.9%, and no unreported
pathogenic variant in the negative set. LoD: state the VAF at which detection is reliable at a
stated depth, for example 20% at 100x with three replicates. Contamination and sample-swap
checks are part of the protocol, not an afterthought.

## 14. Interview questions

**1. Walk me through a germline variant from FASTQ to report.**
Strong answer: names alignment, calling, filtering, normalization, annotation with transcript,
frequency by ancestry, evidence codes, combining rule, orthogonal confirmation, and report
structure, with a failure mode at each stage.
Weak answer: lists tools without the decisions or the failure modes.

**2. Why is VAF not zygosity?**
Strong answer: VAF depends on depth, purity, copy number, mapping, and mosaicism; the expected
0.5 is a prior, not a measurement. Names the tests that resolve the ambiguity.
Weak answer: "0.5 means heterozygous."

**3. What is the difference between pathogenic and likely pathogenic?**
Strong answer: the ACMG certainty threshold (greater than 90% for likely), operationalized as
point totals of 10 or more versus 6-9, with the caveat that correlated evidence can inflate the
total.
Weak answer: "likely is less certain" with no threshold.

**4. How do you handle a VUS in a high-risk gene?**
Strong answer: report as VUS, do not use it for management, consider segregation or functional
follow-up, manage by family history, and state that reclassification is possible.
Weak answer: treats the VUS as actionable because the gene is important.

**5. Why does PM2 need care?**
Strong answer: absence from a database is not rarity in every population; ClinGen downgraded
PM2 to supporting, and coverage at the site must be checked before applying it.
Weak answer: "if it is not in gnomAD, it is rare."

**6. What is the PPV problem in variant screening?**
Strong answer: at low prevalence, even high specificity yields low PPV; gives the arithmetic and
the implication for confirmatory testing.
Weak answer: confuses PPV with sensitivity or says "specificity is high, so it is fine."

**7. When can you not use a short-read exome?**
Strong answer: repeat expansions, balanced rearrangements, deep intronic variants, methylation
defects, and some CNVs; names the alternative assay for each.
Weak answer: "exome covers everything."

**8. How do you decide a variant is somatic?**
Strong answer: paired normal, VAF relative to purity, presence in the normal at low level,
and orthogonal confirmation; tumor-only is a flag, not a conclusion.
Weak answer: "it is not in the blood, so it is somatic."

**9. What does a ClinVar star rating mean?**
Strong answer: review status, from single submitter to practice guideline; a one-star record is
an opinion, not a consensus.
Weak answer: treats ClinVar as ground truth.

**10. How would you validate a variant caller?**
Strong answer: known-truth samples, per-class sensitivity and specificity, truth-set limitations
in repeats, reproducibility, and LoD by VAF and depth.
Weak answer: "compare to another caller."

**11. What is the difference between classification and actionability?**
Strong answer: classification is evidence about the variant; actionability is evidence that an
intervention changes outcome; they are separate statements with separate evidence.
Weak answer: conflates them.

**12. What would make you reclassify a variant?**
Strong answer: new frequency data, new functional data, updated transcript annotation, revised
guidelines, or new segregation; states the need for a versioned report and a reanalysis policy.
Weak answer: "if a paper comes out."

## 15. Key terms

- **ACMG/AMP** — 2015 framework for germline variant classification.
- **Actionability** — whether an intervention changes outcome.
- **Allele frequency** — proportion of chromosomes carrying an allele in a population.
- **BAM/CRAM** — binary alignment formats.
- **BA1** — stand-alone benign frequency criterion.
- **CNV** — copy-number variant.
- **ClinGen** — consortium producing gene and variant specifications.
- **ClinVar** — public archive of variant interpretations.
- **Coverage** — reads per position, in fold depth.
- **FAF** — filtering allele frequency, an upper confidence bound.
- **gVCF** — VCF with reference-confidence blocks for joint calling.
- **gnomAD** — Genome Aggregation Database of population frequencies.
- **HGVS** — variant nomenclature standard.
- **Indel** — insertion or deletion of a small number of bases.
- **Joint calling** — multi-sample calling that shares information.
- **MANE Select** — agreed representative transcript per gene.
- **Mosaic** — variant present in a subset of cells.
- **MSI** — microsatellite instability.
- **Oncogenicity** — somatic variant classification framework.
- **Pathogenic** — variant category indicating causal role in a Mendelian condition.
- **Penetrance** — probability of phenotype given genotype.
- **PP3/BP4** — computational evidence codes, supporting strength.
- **PVS1** — very strong null-variant criterion.
- **Secondary finding** — actionable variant unrelated to the test indication.
- **SNV** — single-nucleotide variant.
- **Somatic variant** — variant acquired in a tissue, not inherited.
- **Structural variant** — variant of 50 bp or more.
- **TMB** — tumour mutational burden, mutations per megabase.
- **Transcript** — annotated RNA product of a gene.
- **VAF** — variant allele fraction.
- **VCF** — variant call format.
- **VUS** — variant of uncertain significance.

## 16. Sources

Framework and guidance:

- Richards, S. et al., "Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology," *Genetics in Medicine* 17:405-424, 2015. https://doi.org/10.1038/gim.2015.30 (accessed 25 October 2026).
- Tavtigian, S. V. et al., "Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework," *Genetics in Medicine* 20:1054-1060, 2018. https://doi.org/10.1038/gim.2017.210 (accessed 25 October 2026).
- ClinGen, Variant Classification Guidance. https://www.clinicalgenome.org/tools/clingen-variant-classification-guidance (accessed 25 October 2026).
- Miller, D. T. et al., "ACMG SF v3.2 list for reporting of secondary findings," *Genetics in Medicine* 25:100866, 2023. https://doi.org/10.1016/j.gim.2023.100866 (accessed 25 October 2026).

Data resources:

- gnomAD v4.0 release announcement, November 2023. https://gnomad.broadinstitute.org/news/2023-11-gnomad-v4-0/ (accessed 25 October 2026).
- ClinVar statistics. https://www.ncbi.nlm.nih.gov/clinvar/docs/statistics/ (accessed 25 October 2026).
- ClinVar. https://www.ncbi.nlm.nih.gov/clinvar/ (accessed 25 October 2026).
- CAGI, Critical Assessment of Genome Interpretation. https://genomeinterpretation.org/ (accessed 25 October 2026).

Formats and nomenclature:

- GA4GH hts-specs, VCF 4.4 and 4.5. https://samtools.github.io/hts-specs/ (accessed 25 October 2026).
- HGVS nomenclature. https://hgvs-nomenclature.org/ (accessed 25 October 2026).
- MANE Select. https://www.ncbi.nlm.nih.gov/refseq/MANE/ (accessed 25 October 2026).

Regulatory:

- FDA, List of FDA-Authorized Companion Diagnostic Devices. https://www.fda.gov/medical-devices/in-vitro-diagnostics/list-fda-authorized-companion-diagnostic-devices-in-vitro-and-imaging-tools (accessed 25 October 2026).
- FDA, Laboratory Developed Tests. https://www.fda.gov/medical-devices/in-vitro-diagnostics/laboratory-developed-tests (accessed 25 October 2026).
- ICH E16, Biomarkers Related to Drug or Biotechnology Product Development. https://database.ich.org/sites/default/files/E16_Guideline.pdf (accessed 25 October 2026).
- Regulation (EU) 2017/746 on in vitro diagnostic medical devices. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_data-types.md`](01_data-types.md)
- Next topic: [`03_transcriptomics.md`](03_transcriptomics.md)
- Related topic: [`10_clinical-genomics.md`](10_clinical-genomics.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 12.1, variant annotation, in the survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
