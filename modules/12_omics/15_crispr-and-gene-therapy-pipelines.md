# CRISPR and Gene Therapy Data Pipelines

> Module 12 — Omics and Precision Medicine
> Topic 15 of 16
> Estimated study time: 4 hours
> Prerequisites: topics 01, 02, and 09
> Version 2026.3

## 1. Why this topic exists

In December 2023 the United States Food and Drug Administration (FDA) approved two cell-based
gene therapies for sickle cell disease on the same day: exagamglogene autotemcel, trade name
Casgevy, which uses CRISPR-Cas9 to disrupt a regulatory element, and lovotibeglogene autotemcel,
trade name Lyfgenia, which uses a lentiviral vector to add a functional globin gene. Casgevy was
extended to transfusion-dependent beta-thalassemia on 30 January 2024. These approvals moved gene
editing from a research technique to a regulated manufacturing process.

The data pipeline behind such a product is not a single analysis. It is a chain of assays, each
with its own sequencing data, statistics, and acceptance criteria:

1. **Guide design.** Choose a 20-nucleotide guide RNA (sgRNA) that cuts the intended site and
   minimizes cutting elsewhere in the genome.
2. **Off-target assessment.** Predict candidate off-target sites in silico, then measure editing at
   those sites with targeted sequencing.
3. **On-target characterization.** Measure the spectrum of insertions and deletions at the intended
   site, because a one-base deletion and a hundred-base deletion have different functional
   consequences.
4. **Potency.** Show that the edited cells or the vector produce the intended biological effect.
5. **Vector manufacturing.** For viral vectors, measure full versus empty capsids, genome integrity,
   and residual host DNA.
6. **Nonclinical safety.** Assess off-target editing, translocations, and tumorigenicity in relevant
   models.
7. **Long-term follow-up.** Follow patients for years, because the consequences of an edit can
   appear late.

Each step produces data that a regulator will inspect. A bioinformatician in this field spends
more time on assay design, limits of detection, and acceptance criteria than on the editing
reaction itself.

Three technical facts shape the work.

**Editing is not binary.** A CRISPR-Cas9 cut is repaired by non-homologous end joining or by
homology-directed repair. The first produces a distribution of insertions and deletions; the second
produces a precise change at a low efficiency in most cell types. Base editors change one base
without a double-strand break, and prime editors write a new sequence using a reverse
transcriptase. Each modality has a different error spectrum, and the pipeline must measure the
spectrum rather than a single "editing efficiency".

**Off-target detection has a limit.** An amplicon sequencing assay at a given depth cannot exclude
editing below a calculable fraction. Reporting "no off-target editing detected" without the limit
of detection is not a result. The limit is a function of depth, and the depth required grows as
the acceptable fraction falls.

**Delivery determines the biology.** Ex vivo editing of hematopoietic stem cells requires
myeloablative conditioning and produces a one-time product. In vivo delivery with a lipid
nanoparticle is transient, repeatable, and limited by the biodistribution of the particle.
Adeno-associated virus (AAV) delivers a persistent episome and is limited by packaging capacity and
pre-existing immunity. The same edit has different risks depending on how it is delivered.

### 1.1 The failure pattern

1. Guides are ranked by an on-target score from a model trained on a different cell type.
2. Off-target sites are predicted with a mismatch-count heuristic and never measured.
3. Off-target sequencing is run at 1,000 reads per site and reported as "no editing detected".
4. Editing efficiency is reported as the fraction of reads with any indel, without the spectrum.
5. A dominant single indel is treated as a clean knockout when it may be a passenger artifact.
6. Vector genome integrity is assessed by a method that cannot see truncations.
7. Potency is inferred from a surrogate marker that has not been qualified against the clinical
   mechanism.
8. Long-term follow-up is planned but not resourced.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | A therapy with an uncharacterized risk | Cannot inspect the assay |
| Manufacturing team | A batch that fails release | Depends on assay acceptance criteria |
| Regulatory reviewer | A submission without a defensible off-target package | Depends on the sponsor's data |
| Clinical team | A potency claim that does not predict effect | Depends on assay qualification |
| Bioinformatician | Weeks on data that cannot answer the question | Inherited the assay design |
| Program | A clinical hold | Depends on the whole chain |

### 1.3 The industry question

> Is the intended edit present at the intended site, absent at unintended sites to a stated limit
> of detection, and produced by a process that can do it reproducibly?

The question has three clauses, and each maps to a different dataset. The first is on-target
characterization, the second is off-target assessment, and the third is chemistry, manufacturing,
and controls. A program that answers one and not the others does not have a product.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** nuclease editing, base editing, and prime editing by mechanism, outcome
   spectrum, and the assays each requires.
2. **Choose** a delivery vector for a given edit, and state the constraints that rule out the
   alternatives.
3. **Design** a guide selection workflow that balances on-target efficiency against off-target
   risk, and justify the balance.
4. **Compute** the limit of detection of an off-target sequencing assay and state what a negative
   result excludes.
5. **Classify** amplicon reads into unmodified, in-frame, and frameshift outcomes, and explain why
   the spectrum matters.
6. **Evaluate** vector manufacturing data, including full versus empty capsids and genome
   integrity.
7. **Map** a program onto the FDA gene therapy pathway, including the genome editing guidance and
   long-term follow-up expectations.
8. **Diagnose** a batch failure or an ambiguous off-target result from the underlying data.

## 3. The core mental model

A gene therapy program is a chain from an edit to a patient, and every link has a measurement.

```text
  THERAPEUTIC HYPOTHESIS
      |
      v
  EDIT DESIGN: which base, which gene, which cell
      |
      +--> nuclease (cut and repair)
      +--> base editor (chemical change, no cut)
      +--> prime editor (write new sequence)
      |
      v
  DELIVERY
      |
      +--> ex vivo: electroporation or lentiviral vector
      +--> in vivo: AAV or lipid nanoparticle
      |
      v
  PRODUCT
      |
      +--> cell product: viability, dose, identity, potency
      +--> vector product: titer, full/empty, integrity, residuals
      |
      v
  SAFETY PACKAGE
      |
      +--> on-target spectrum
      +--> off-target editing to a stated LOD
      +--> translocations and large rearrangements
      +--> tumorigenicity and long-term follow-up
      |
      v
  CLINICAL
      |
      +--> conditioning, infusion, engraftment
      +--> efficacy endpoint
      +--> durability and late adverse events
```

### 3.1 Three editing modalities, three error spectra

| Modality | Mechanism | Typical outcome | Dominant risk | Primary assay |
|---|---|---|---|---|
| Nuclease (Cas9) | Double-strand break, repaired by non-homologous end joining or homology-directed repair | Insertions and deletions of variable size | Large deletions, translocations, chromothripsis | Amplicon sequencing, long-read sequencing, translocation assays |
| Base editor | Deamination of a target base without a double-strand break | A specific C to T or A to G change, with bystander edits | Bystander editing in the window, off-target deamination | Amplicon sequencing, RNA off-target assays |
| Prime editor | Nick plus reverse transcriptase writing a new sequence | A precise insertion, deletion, or substitution | PegRNA stability, unintended edits at the nick | Amplicon sequencing, long-read sequencing |

The distinction matters because a nuclease program that reports "80% editing" has said nothing
about whether the edits are frameshift knockouts or in-frame deletions that preserve function.
A base editor program that reports "80% editing" has said nothing about bystander edits in the
same window.

### 3.2 Delivery determines persistence and risk

| Delivery | Duration | Redosable | Capacity | Dominant constraint |
|---|---|---|---|---|
| Ex vivo electroporation | One-time cell product | No, without a second collection | Not applicable | Conditioning toxicity |
| Lentiviral vector, ex vivo | Stable integration | No | About 8 to 10 kilobases | Insertional mutagenesis, replication-competent virus |
| AAV, in vivo | Episomal, long-lived in non-dividing cells | Limited by immunity | About 4.7 kilobases | Packaging limit, pre-existing neutralizing antibodies |
| Lipid nanoparticle, in vivo | Transient | Yes | Nucleic acid cargo | Biodistribution, repeat dosing, innate immune response |

The vector is not a delivery detail. It determines whether the edit is permanent, whether the
patient can be re-dosed, and which safety questions apply.

### 3.3 The limit of detection is a property of the assay

If a site has a true editing fraction \(f\) and the assay observes \(N\) reads, the probability of
seeing zero edited reads is \((1-f)^N\). Setting that equal to 0.05 gives the depth needed to
exclude \(f\) at 95% confidence:

```text
  N = ln(0.05) / ln(1 - f)
```

For \(f = 0.001\), \(N = 2,994\) reads. For \(f = 0.0001\), \(N = 29,956\) reads. A report that
says "no off-target editing" without a depth and a limit of detection is not interpretable.

### 3.4 What "editing efficiency" should mean

Four different quantities are called editing efficiency:

| Quantity | Definition | What it answers |
|---|---|---|
| Read-level indel fraction | Fraction of sequencing reads with any indel | How much of the population was cut |
| Cell-level editing fraction | Fraction of cells with at least one edited allele | How many cells carry the modification |
| Allele-level fraction | Fraction of alleles edited | Dosage of the edit |
| Functional fraction | Fraction of cells with the intended functional change | Whether the biology changed |

For a biallelic knockout, the cell-level fraction under independent allele editing is
\(1 - (1-p)^2\), where \(p\) is the allele-level fraction. At \(p = 0.8\), the cell fraction is
0.96. At \(p = 0.5\), it is 0.75. The difference matters when the therapeutic effect requires both
alleles to be disrupted.

### 3.5 The three regulators of the pipeline

| Regulator | What it constrains | What it does not |
|---|---|---|
| Science | Whether the edit produces the intended biology | Whether the process is reproducible |
| Manufacturing | Whether each batch meets specification | Whether the specification predicts efficacy |
| Regulation | Whether the evidence supports an investigational or marketing application | Whether the therapy is affordable or accessible |

A program needs all three, and a failure in one is not fixed by strength in another.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| CRISPR | Clustered regularly interspaced short palindromic repeats, a bacterial adaptive immune system repurposed for editing | Used as a synonym for Cas9 | System |
| Cas9 | RNA-guided endonuclease that creates a double-strand break | Assumed to be the only editing nuclease | Protein |
| Single guide RNA (sgRNA) | 20-nucleotide guide plus scaffold that directs Cas9 | Confused with the target sequence alone | RNA sequence |
| Protospacer adjacent motif (PAM) | Short sequence required next to the target, such as NGG for SpCas9 | Assumed to be part of the guide | Nucleotide motif |
| Double-strand break (DSB) | Break in both DNA strands | Treated as the only way to edit | Lesion |
| Non-homologous end joining (NHEJ) | Error-prone repair that produces indels | Assumed to be uniform | Repair pathway |
| Homology-directed repair (HDR) | Templated repair that can produce a precise change | Assumed efficient in all cell types | Repair pathway |
| Insertion or deletion (indel) | Small length change at a cut site | Reported without the size distribution | Base pairs |
| Frameshift | Indel whose length is not a multiple of three | Confused with any indel | Boolean |
| Base editor | Fusion protein that converts one base to another without a DSB | Assumed to have no off-target effects | Protein |
| Bystander edit | Unintended conversion of another base in the editing window | Ignored in efficiency reporting | Position |
| Prime editor | Cas9 nickase fused to reverse transcriptase, guided by a prime editing guide RNA | Assumed to be free of indels | Protein |
| Prime editing guide RNA (pegRNA) | Guide that encodes both the target and the desired edit | Confused with an sgRNA | RNA sequence |
| Off-target editing | Editing at a site other than the intended one | Reported without a limit of detection | Site and fraction |
| On-target spectrum | Distribution of edit outcomes at the intended site | Reported as a single efficiency number | Distribution |
| GUIDE-seq | Experimental method that tags DSBs and sequences the sites | Assumed to find all off-targets | Assay |
| CIRCLE-seq | In vitro method for unbiased off-target detection | Assumed to reflect cellular editing | Assay |
| DISCOVER-seq | Method that maps repair-protein binding | Assumed to measure editing directly | Assay |
| CHANGE-seq | In vitro, cell-free off-target profiling method | Confused with cell-based assays | Assay |
| Cutting frequency determination (CFD) score | Position-weighted score for a mismatched guide | Treated as a measured probability | 0 to 1 |
| Limit of detection (LOD) | Smallest fraction distinguishable from zero at a stated confidence | Omitted from negative results | Fraction |
| Vector copy number (VCN) | Average number of vector genomes per cell | Reported as a single mean without distribution | Copies per cell |
| Full capsid | AAV particle containing the vector genome | Assumed to be the majority | Fraction |
| Empty capsid | AAV particle without the vector genome | Assumed harmless | Fraction |
| Inverted terminal repeat (ITR) | AAV sequence element required for packaging | Assumed to be always intact | Sequence |
| Replication-competent virus | Virus able to replicate after manufacturing | Assumed absent without testing | Assay |
| Transgene | Gene or sequence delivered by a vector | Confused with the edit | Sequence |
| Conditioning | Preparative regimen before cell infusion | Assumed to be risk-free | Regimen |
| Engraftment | Establishment of infused cells in the bone marrow | Confused with infusion | Clinical outcome |
| Potency | Ability of the product to produce the intended effect | Substituted with a surrogate marker | Assay-specific |
| Chemistry, manufacturing, and controls (CMC) | The manufacturing and testing package | Treated as paperwork | Regulatory section |
| Investigational new drug (IND) | Application to test an unapproved product in humans | Confused with approval | Regulatory filing |
| Long-term follow-up | Extended monitoring after gene therapy | Assumed optional | Protocol |
| Insertional oncogenesis | Cancer caused by vector integration near an oncogene | Assumed impossible for non-integrating vectors | Adverse event |

## 5. The mechanics, step by step

The procedure runs from a therapeutic hypothesis to a follow-up plan. It assumes a candidate edit
and a target tissue.

| Step | Question answered | Artifact | Failure if skipped |
|---:|---|---|---|
| 1 | What is the edit and the mechanism? | Edit specification | A cut with no functional rationale |
| 2 | Which editing modality? | Modality choice | Wrong error spectrum measured |
| 3 | Which delivery? | Vector or ex vivo route | Persistence and risk mischaracterized |
| 4 | Which guide? | Ranked guides | Poor efficiency or high off-target risk |
| 5 | What are the predicted off-targets? | Candidate site list | Unmeasured risk |
| 6 | What are the measured off-targets, to what LOD? | Amplicon data and LOD | A negative result with no meaning |
| 7 | What is the on-target spectrum? | Edit distribution | An in-frame deletion called a knockout |
| 8 | Is the product potent? | Potency assay | A batch that passes but does not work |
| 9 | Is the vector or cell product consistent? | CMC release data | Batch failure or unexplained variability |
| 10 | Is the nonclinical safety package adequate? | Toxicology and tumorigenicity | Clinical hold |
| 11 | Is the clinical design adequate? | Protocol | Uninterpretable efficacy |
| 12 | Is the follow-up adequate? | Long-term plan | Late events missed |

### Step 1. Define the edit and the mechanism

**Input.** A disease mechanism and a candidate target.

**Output.** An edit specification: the gene, the coordinate, the intended change, the cell type,
and the functional consequence.

**Rationale.** The specification must state what the edit does, not only where it cuts. Disrupting
an enhancer to raise fetal hemoglobin is a different specification from correcting a point
mutation in a coding exon. The first tolerates a range of indels; the second requires a precise
sequence change.

**If skipped.** The program optimizes cutting efficiency for an edit whose functional consequence
was never defined.

### Step 2. Choose the editing modality

**Input.** The edit specification; the cell type; the delivery constraints.

**Output.** A modality with a stated rationale.

**Rationale.** Modality determines the outcome spectrum, the assays, and the regulatory questions.
The choice has more than two outcomes.

| Situation | Modality | Reason | Main assay risk |
|---|---|---|---|
| Disrupt a gene or enhancer | Nuclease | Indels abolish function | Large deletions and translocations |
| Correct a point mutation in a dividing cell | Nuclease plus HDR template | Templated repair can write the correct base | Low HDR efficiency; indels dominate |
| Convert a single base | Base editor | No DSB; higher precision | Bystander edits and off-target deamination |
| Insert a short sequence precisely | Prime editor | Writes new sequence without a DSB | Efficiency varies by cell type; pegRNA design |
| Replace a large sequence | Nuclease plus large template, or vector-mediated addition | HDR is inefficient for large cargo | Integration and delivery |
| Add a functional gene | Lentiviral or AAV vector | The gene is added, not corrected | Insertional risk or episomal loss |

**If skipped.** A nuclease is used for a correction that requires precision, and the resulting
indel mixture is reported as "editing" without the correction.

### Step 3. Choose the delivery route

**Input.** Cell type, tissue, edit durability requirement, dosing strategy.

**Output.** A delivery route with a stated persistence and redosing plan.

**Rationale.** Delivery determines whether the edit is permanent, whether the patient can be
treated again, and which safety questions apply.

| Target | Route | Persistence | Key constraint |
|---|---|---|---|
| Hematopoietic stem cells | Ex vivo electroporation | Permanent in the graft | Myeloablative conditioning |
| Hematopoietic stem cells with gene addition | Ex vivo lentiviral | Permanent, integrating | Insertional risk; VCN control |
| Liver, transient effect | Lipid nanoparticle | Days to weeks | Repeat dosing; biodistribution |
| Liver, long-term effect | AAV | Years in hepatocytes | Packaging limit; immunity; episomal loss with division |
| Retina | Subretinal AAV | Long-term in post-mitotic cells | Surgical delivery |
| Muscle or central nervous system | AAV | Long-term | Dose-limiting toxicity |

**If skipped.** The program plans a repeat dose for a vector that induces neutralizing antibodies
after the first administration.

### Step 4. Select the guide

**Input.** Target sequence; PAM availability; cell type; nuclease.

**Output.** Ranked guides with on-target scores and predicted off-target profiles.

**Rationale.** On-target efficiency models are trained on specific cell types and delivery
methods, and their predictions transfer imperfectly. Off-target risk depends on the genome and on
the guide sequence, and the two objectives can conflict: the most efficient guide may have the
worst off-target profile.

| Guide property | Favorable | Unfavorable |
|---|---|---|
| PAM | Canonical, close to the edit | Non-canonical, distant |
| GC content | 40% to 60% | Extremes |
| Poly-T stretch | Absent | Present, can terminate transcription |
| Self-complementarity | Low | High, can form secondary structure |
| Position of the edit | Close to the cut | Far from the cut |
| Predicted off-targets | Few with high mismatch | Many with few mismatches |

**If skipped.** A guide chosen for efficiency alone carries an off-target profile that the program
then has to defend.

### Step 5. Predict off-target sites in silico

**Input.** Guide sequence; reference genome; variant information.

**Output.** A ranked list of candidate sites with mismatch positions and scores.

**Rationale.** Prediction narrows the search but cannot prove absence. The number of possible
mismatch combinations is large: for a 20-nucleotide guide, allowing up to three mismatches gives
1 + 60 + 1,710 + 30,780 = 32,551 sequence variants, and the genome contains about 387.5 million
PAM sites on both strands. Exhaustive enumeration is feasible with efficient data structures, but
the scoring model is the weak point.

**If skipped.** Only the top few predicted sites are tested, and the rest of the genome is
unexamined.

### Step 6. Measure off-target editing and state the limit of detection

**Input.** Candidate sites; amplicon panel or unbiased assay; sequencing depth.

**Output.** Editing fraction per site, depth per site, and limit of detection.

**Rationale.** A negative result is only meaningful with a limit. The depth determines the limit,
and the limit must be reported alongside the result.

| Depth per site | LOD at 95% confidence | What a negative result excludes |
|---:|---:|---|
| 300 reads | About 1.0% | Editing above 1% |
| 3,000 reads | About 0.1% | Editing above 0.1% |
| 30,000 reads | About 0.01% | Editing above 0.01% |
| 100,000 reads | About 0.003% | Editing above 0.003% |

The discovery method determines which sites enter the panel, and the methods differ in what they
can see.

| Method | Setting | Strength | Limitation |
|---|---|---|---|
| GUIDE-seq | Cells | Captures breaks in the living cell | Requires double-strand breaks and tag integration |
| CIRCLE-seq | Cell-free | High sensitivity, unbiased | Does not reflect chromatin or cellular repair |
| DISCOVER-seq | Cells | Maps repair-protein binding | Detects repair sites, not edits directly |
| CHANGE-seq | Cell-free | High throughput | Same chromatin limitation as CIRCLE-seq |
| Amplicon panel | Cells | Quantitative, high depth | Only the sites in the panel |
| Whole-genome sequencing | Cells | Unbiased | Expensive; limited depth per site |

**If skipped.** The program reports "no off-target editing" from an assay that could not have
detected a clinically relevant event.

### Step 7. Characterize the on-target spectrum

**Input.** Amplicon sequencing of the target site.

**Output.** The distribution of edit sizes and types, the frameshift fraction, and the fraction
of alleles with the intended change.

**Rationale.** A single efficiency number hides the spectrum. A population with 80% editing might
be 80% one-base deletions, or 40% frameshifts and 40% large in-frame deletions. The functional
consequence differs.

| Spectrum observation | Interpretation | Action |
|---|---|---|
| Dominant single indel | A preferred repair outcome | Confirm the functional effect; check for a microhomology artifact |
| Broad distribution of small indels | Typical NHEJ | Report the frameshift fraction |
| Large deletions above the amplicon size | Amplicon assay cannot see them | Use long-read sequencing |
| Substitutions only | Base editing, or a nuclease without cutting | Confirm the modality and the on-target edit |
| Unexpected sequence from the template | Template integration | Check for plasmid or vector backbone insertion |

**If skipped.** An in-frame deletion that preserves protein function is counted as a knockout, and
the potency assay is interpreted against the wrong mechanism.

### Step 8. Qualify a potency assay

**Input.** Product mechanism; candidate assays.

**Output.** A potency assay with a defined range, precision, and a link to the clinical mechanism.

**Rationale.** Potency is the release test most likely to be disconnected from efficacy. A
surrogate marker, such as the fraction of edited alleles, is easy to measure and may not predict
the biological effect. The assay should be qualified against a relevant functional readout.

**If skipped.** A batch that passes release fails in the clinic, and the failure is attributed to
the patient rather than to the specification.

### Step 9. Control the manufacturing process

**Input.** Cell or vector manufacturing data.

**Output.** Release data against specifications, plus process trends.

**Rationale.** For AAV, the critical quality attributes include full versus empty capsids, vector
genome integrity, residual host DNA, and infectious titer. For cell products, they include
viability, cell dose, identity, vector copy number where applicable, and potency.

| Attribute | Why it matters | Typical method |
|---|---|---|
| Full capsid fraction | Empty capsids add antigen load without payload | Analytical ultracentrifugation, charge detection mass spectrometry |
| Genome integrity | Truncated genomes are non-functional | Long-read sequencing, droplet digital polymerase chain reaction |
| Residual host DNA | Oncogenic and immunogenic risk | Shotgun sequencing, quantitative polymerase chain reaction |
| Vector copy number | High copies increase insertional risk | Droplet digital polymerase chain reaction |
| Replication-competent virus | Potential for dissemination | Cell-based assay, sequencing |
| Potency | Batch-to-batch consistency | Mechanism-relevant assay |

**If skipped.** A batch is released on a specification that does not control the attribute that
determines clinical effect.

### Step 10. Assemble the nonclinical safety package

**Input.** On-target and off-target data; biodistribution; toxicology.

**Output.** A safety package covering off-target editing, large rearrangements, and
tumorigenicity.

**Rationale.** The FDA's genome editing guidance, finalized in January 2024, names three risks:
off-target editing, unintended consequences of on-target editing, and unknown long-term effects.
The package must address each with data appropriate to the modality and the delivery route.

**If skipped.** The IND is placed on hold for a missing safety study.

### Step 11. Design the clinical trial

**Input.** Safety package; potency assay; patient population.

**Output.** A protocol with an endpoint that reflects the mechanism, a conditioning regimen, and
a monitoring plan.

**Rationale.** For an ex vivo hematopoietic stem cell therapy, the endpoint is usually a
functional outcome such as transfusion independence or the absence of vaso-occlusive crises. For
an in vivo liver-directed therapy, it may be a biomarker such as a clotting factor level. The
endpoint must be measurable in the population and must not be a surrogate that the product
influences trivially.

**If skipped.** The trial measures a biomarker that moves for reasons unrelated to the edit.

### Step 12. Plan long-term follow-up

**Input.** Integration status; delivery route; patient population.

**Output.** A follow-up protocol with duration, endpoints, and data collection.

**Rationale.** Integrating vectors and editing nucleases carry late risks. The FDA's long-term
follow-up guidance, issued in January 2020, recommends follow-up for at least 15 years after
administration for integrating vectors and for genome editing products, with shorter periods
possible for non-integrating vectors in some cases. The exact expectation depends on the product.

The follow-up design must also specify what is collected: survival status, new malignancies, new
autoimmune or neurological diagnoses, and, where consent allows, samples for vector integration
site analysis. A follow-up that records only survival cannot attribute a late event to the
product.

**If skipped.** A late malignancy is detected by the health system rather than by the sponsor, and
the causal link cannot be assessed.

### 5.13 Approved products and what they establish

| Product | Approval | Vector or modality | Indication | What it established |
|---|---|---|---|---|
| Luxturna (voretigene neparvovec) | FDA, December 2017 | AAV2, subretinal | RPE65-mediated retinal dystrophy | In vivo AAV gene addition can restore function |
| Zolgensma (onasemnogene abeparvovec) | FDA, May 2019 | AAV9, intravenous | Spinal muscular atrophy | Systemic AAV delivery in infants |
| Hemgenix (etranacogene dezaparvovec) | FDA, November 2022 | AAV5, intravenous | Hemophilia B | In vivo factor IX expression |
| Roctavian (valoctocogene roxaparvovec) | FDA, June 2023 | AAV5, intravenous | Hemophilia A | In vivo factor VIII expression with variable durability |
| Casgevy (exagamglogene autotemcel) | FDA, 8 December 2023; thalassemia, 30 January 2024 | CRISPR-Cas9, ex vivo | Sickle cell disease and transfusion-dependent beta-thalassemia | A genome editing product can be approved |
| Lyfgenia (lovotibeglogene autotemcel) | FDA, 8 December 2023 | Lentiviral vector, ex vivo | Sickle cell disease | A integrating vector product can be approved alongside an editing product |

The dates and indications are documented in the FDA approvals. The interpretation, that each
product establishes a pathway for later products, is inference.

### 5.14 Base and prime editing in the clinic

Base editing entered human testing later than nuclease editing, and prime editing later still.

| Program | Modality | Setting | Documented status |
|---|---|---|---|
| BEAM-101 | Base editing of HBG1 and HBG2 promoters | Ex vivo, sickle cell disease | Phase 1/2 BEACON study reported engraftment and fetal hemoglobin above 60% in an interim update in 2025 |
| VERVE-101 | Base editing of PCSK9 | In vivo, lipid nanoparticle | First in vivo base editing; enrollment stopped after an adverse event |
| VERVE-102 | Base editing of PCSK9 with a revised particle | In vivo, lipid nanoparticle | Reported reductions in PCSK9 and low-density lipoprotein cholesterol in early cohorts |
| PM359 | Prime editing of NCF1 | Ex vivo, chronic granulomatous disease | First prime editing therapy with human data, reported in December 2025 |

The details change quickly, and any specific number should be checked against the current
publication before it is used in a program document. The structural point is stable: base and
prime editing have moved from concept to clinical data, and each brings a different error spectrum
that the pipeline must measure.

### 5.15 Tests that separate a defensible package from a plausible one

| Test | Procedure | Pass criterion | What it catches |
|---|---|---|---|
| Guide activity | Measure editing at the target in the intended cell type | Above the project threshold | Model predictions that do not transfer |
| Unbiased off-target discovery | GUIDE-seq, CIRCLE-seq, or CHANGE-seq | A candidate list, not a proof of absence | Sites missed by prediction |
| Targeted off-target sequencing | Amplicon at 30,000 reads or more | LOD stated and met | Negative results with no meaning |
| On-target spectrum | Amplicon plus long-read | Frameshift fraction and large deletions reported | In-frame edits counted as knockouts |
| Translocation assay | Dual-guide or chromosome-specific assay | Below the stated limit | Large rearrangements |
| Vector integrity | Long-read sequencing across the ITRs | Full-length genome fraction reported | Truncated genomes |
| Potency | Mechanism-relevant functional assay | Dose-response and precision | Surrogate release tests |
| Replication-competent virus | Cell-based assay | Negative at the stated sensitivity | Manufacturing contamination |

### 5.16 What the procedure does not decide

The procedure produces a data package. It does not decide whether the target is the right target,
whether the indication justifies the risk of myeloablative conditioning, or how to price the
product. Those are program and policy decisions, and they depend on evidence outside the editing
pipeline.

## 6. Worked example

### 6.1 The case

A 19-year-old patient with sickle cell disease is enrolled in an ex vivo CRISPR program targeting
the BCL11A erythroid enhancer to raise fetal hemoglobin. The manufacturing chain collects
CD34-positive hematopoietic stem and progenitor cells, edits them by electroporation with a
Cas9-guide RNA ribonucleoprotein, and infuses the edited product after myeloablative conditioning.

The patient weighs 70 kilograms. The collection target is 8 million CD34-positive cells per
kilogram.

### 6.2 Step 1: compute the cell dose

```text
  target collection = 8 x 10^6 cells/kg x 70 kg = 5.6 x 10^8 cells
```

After editing, the product loses cells to the electroporation and to cryopreservation. Assume 90%
viability after editing and 70% recovery through the process:

```text
  product = 5.6 x 10^8 x 0.90 x 0.70
          = 5.6 x 10^8 x 0.63
          = 3.528 x 10^8 cells
  dose    = 3.528 x 10^8 / 70 kg
          = 5.04 x 10^6 cells/kg
```

If the program's minimum effective dose is 3 million cells per kilogram, the product is adequate
with a margin of 5.04 / 3.00 = 1.68, or 68% above the minimum.

### 6.3 Step 2: translate allele editing into cell editing

Suppose the on-target assay reports that 80% of sequenced alleles carry an indel at the enhancer.

For a cell to have at least one edited allele, at least one of its two alleles must be edited.
Assuming independent editing:

```text
  P(at least one edited allele) = 1 - (1 - 0.80)^2
                                = 1 - 0.04
                                = 0.96
```

If the therapeutic effect requires both alleles to be disrupted:

```text
  P(both edited) = 0.80^2 = 0.64
```

The difference between 96% and 64% is the difference between a population-level effect and a
near-clonal one. Which number matters depends on the mechanism: raising fetal hemoglobin is a
per-cell effect that can be achieved with one disrupted allele, so the 96% figure is the relevant
one for this program.

### 6.4 Step 3: compute the limit of detection for off-target assays

The off-target panel includes 50 predicted sites. Each site is sequenced to 30,000 reads. The
limit of detection at 95% confidence for a true editing fraction \(f\) is:

```text
  N = ln(0.05) / ln(1 - f)
```

Solving for \(f\) at \(N = 30,000\):

```text
  ln(1 - f) = ln(0.05) / 30,000 = -2.9957 / 30,000 = -9.986 x 10^-5
  1 - f = exp(-9.986 x 10^-5) = 0.9999001
  f = 9.99 x 10^-5, or about 0.0100%
```

A negative result at a site therefore excludes editing above about 0.01%, and says nothing about
editing at 0.001%. If the program's acceptance criterion is 0.1%, 30,000 reads give a ten-fold
margin. If the criterion is 0.001%, the assay is not adequate, and roughly 300,000 reads would be
needed.

### 6.5 Step 4: interpret the on-target spectrum

The amplicon data for the product batch show:

| Class | Reads | Fraction |
|---|---:|---:|
| Unmodified | 1,400 | 14.0% |
| 1-base deletion | 4,600 | 46.0% |
| 2-base deletion | 900 | 9.0% |
| 3-base deletion, in frame | 700 | 7.0% |
| 4-base or larger deletion | 1,500 | 15.0% |
| Insertion | 900 | 9.0% |
| Total | 10,000 | 100.0% |

Frameshift fraction:

```text
  1-base deletion   : 4,600
  2-base deletion   :   900
  4-base or larger  : 1,500 (assumed frameshift; some may be in frame)
  insertions        :   900 (sizes not specified)
```

If all insertions and all deletions of 4 bases or more are frameshifts, the frameshift count is
4,600 + 900 + 1,500 + 900 = 7,900, or 79.0%. The in-frame fraction is 700, or 7.0%, and the
unmodified fraction is 14.0%.

For an enhancer disruption, both frameshift and in-frame deletions can abolish the element, so the
relevant number is the fraction of alleles with any disrupting change, which is 86.0%. For a
coding knockout, the frameshift fraction would be the relevant number, and a 79% figure would be
reported instead of the 86% "editing efficiency".

### 6.6 Diagnostic 1: the indel spectrum shape

The first diagnostic that would reveal a broken analysis is the shape of the indel spectrum at the
on-target site.

A working nuclease experiment produces a mixture of small deletions and insertions, with a modal
size of one to four bases and a tail of larger deletions. A spectrum with one dominant species
above about 70% of edited reads is a red flag for one of three causes:

- **A microhomology-mediated repair outcome.** The sequence around the cut shares a short repeat,
  and repair consistently uses it. The result is real, and the functional consequence must be
  checked because the dominant allele may be an in-frame deletion.
- **A polymerase chain reaction or alignment artifact.** A repetitive sequence or a long
  homopolymer can cause the aligner to collapse distinct indels into one apparent species. Long-read
  sequencing distinguishes them.
- **A clonal expansion.** A single edited clone dominates the product, which can happen if the
  culture favours one clone. The insertion site analysis or vector copy number distribution would
  show a narrow clone.

For this batch, the dominant one-base deletion at 46% is within the expected range and does not
trigger the flag. A batch with 85% one-base deletions and 3% other classes would.

### 6.7 Diagnostic 2: off-target depth and limit of detection

The second diagnostic is the relationship between sequencing depth and the reported limit of
detection across the off-target panel.

A working package reports, for every site, the depth, the observed editing fraction, and the LOD.
A broken package shows one of three patterns:

| Pattern | Interpretation | Action |
|---|---|---|
| Depth varies by more than ten-fold across sites | Uneven capture or amplicon efficiency | Rebalance the panel; report per-site LOD |
| LOD above the acceptance criterion at any site | The assay cannot exclude the event of concern | Increase depth or use a different method |
| Zero edited reads at every site with no LOD | The result is not interpretable | Report depth and compute the LOD |
| Editing detected at a site excluded from the panel | Prediction missed a real site | Expand the panel from unbiased discovery |

The third row is the most common failure in submissions. A table of zeros without depths is not a
negative result.

### 6.8 The varied assumption

Change one assumption: the edit is delivered in vivo to the liver with a lipid nanoparticle rather
than ex vivo to hematopoietic stem cells.

The arithmetic changes in four ways.

1. **Dose is per kilogram of body weight, not per kilogram of cells collected.** A 70-kilogram
   patient might receive 0.3 milligrams per kilogram of lipid nanoparticle, or 21 milligrams total.
2. **Editing is transient.** The edit persists only if the target cell survives and the edit is
   stable; hepatocytes turn over slowly, so the effect can last months, but the delivery is not a
   one-time permanent product.
3. **Redosing is possible.** A lipid nanoparticle does not induce a lasting neutralizing antibody
   response against a capsid, although repeat dosing can be limited by innate immunity.
4. **The safety question changes.** The relevant off-target tissue is the liver, and the concern
   is editing in hepatocytes and delivery to other tissues, not insertional mutagenesis in a stem
   cell graft.

The conclusion changes from "a one-time product requiring myeloablative conditioning, with
insertional and clonal risks" to "a repeatable transient therapy with a biodistribution and
hepatic safety question". The off-target LOD requirement is similar, but the tissue sampled and the
duration of follow-up differ.

## 7. Code, query, or calculation

The script below classifies amplicon reads by alignment, computes editing and frameshift fractions,
computes limits of detection, scores a mismatched guide with a simplified position-weight model,
and enumerates the off-target search space. It uses only the Python standard library. The output
was produced by running the script as written.

```python
"""crispr.py - indel classification, detection limit, and off-target arithmetic.

Standard library only. Run: python crispr.py
"""
import math

REF = "GTCAGTACCGATTCGAGCTAGCATTGACCT"   # 30 bp amplicon, non-repetitive
READS = [
    REF,
    REF,
    REF,
    REF,
    REF[:10] + REF[11:],                  # 1 bp deletion
    REF[:8] + REF[14:],                   # 6 bp deletion
    REF[:15] + "A" + REF[15:],            # 1 bp insertion
    REF[:12] + REF[15:],                  # 3 bp deletion, in frame
    REF[:20] + "TT" + REF[20:],           # 2 bp insertion, frameshift
    REF[:22] + "T" + REF[23:],            # single substitution
]


def align(ref, read, match=1, mismatch=-1, gap=-2):
    """Needleman-Wunsch global alignment. Returns aligned strings."""
    n, m = len(ref), len(read)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap
        trace[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (match if ref[i - 1] == read[j - 1] else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            best = max(diag, up, left)
            score[i][j] = best
            trace[i][j] = 0 if best == diag else (1 if best == up else 2)
    a, b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i][j]
        if i > 0 and j > 0 and t == 0:
            a.append(ref[i - 1]); b.append(read[j - 1]); i -= 1; j -= 1
        elif i > 0 and (j == 0 or t == 1):
            a.append(ref[i - 1]); b.append("-"); i -= 1
        else:
            a.append("-"); b.append(read[j - 1]); j -= 1
    return "".join(reversed(a)), "".join(reversed(b))


def classify(ref, read):
    ar, ad = align(ref, read)
    ins = sum(1 for x, y in zip(ar, ad) if x == "-")
    dele = sum(1 for x, y in zip(ar, ad) if y == "-")
    subs = sum(1 for x, y in zip(ar, ad) if x != y and x != "-" and y != "-")
    net = ins - dele
    edited = ins + dele > 0
    return ins, dele, subs, net, edited, (abs(net) % 3 != 0 if edited else False)


print("READ CLASSIFICATION")
print("  ins  del  sub  net  edited  frameshift")
counts = {"unmodified": 0, "edited": 0, "frameshift": 0}
for r in READS:
    ins, dele, subs, net, edited, fs = classify(REF, r)
    counts["edited"] += int(edited)
    counts["frameshift"] += int(fs)
    print(f"  {ins:3d} {dele:4d} {subs:4d} {net:4d}  {str(edited):6s}  {fs}")

n = len(READS)
print(f"\nEDITING SUMMARY (n={n})")
print(f"  edited reads      : {counts['edited']} ({counts['edited']/n:.1%})")
print(f"  frameshift reads  : {counts['frameshift']} ({counts['frameshift']/n:.1%})")
print(f"  in-frame or silent: {counts['edited'] - counts['frameshift']} "
      f"({(counts['edited'] - counts['frameshift'])/n:.1%})")


# ------------------------------------------------------------- detection limit
def reads_for_lod(fraction, confidence=0.95):
    return math.log(1 - confidence) / math.log(1 - fraction)


for f in (0.01, 0.001, 0.0001):
    print(f"  LOD {f:.4%}: {reads_for_lod(f):.0f} reads for 95% confidence")


# ------------------------------------------------------------------- CFD score
# Simplified position weights: 1.0 = mismatch abolishes cutting, 0.0 = tolerated.
PENALTY = {
    1: 0.10, 2: 0.15, 3: 0.20, 4: 0.25, 5: 0.30, 6: 0.35, 7: 0.40,
    8: 0.45, 9: 0.60, 10: 0.70, 11: 0.80, 12: 0.85, 13: 0.88, 14: 0.90,
    15: 0.92, 16: 0.94, 17: 0.96, 18: 0.98, 19: 0.99, 20: 1.00,
}
MISMATCHES = [3, 11, 18]


def cfd(mismatch_positions, penalty=PENALTY):
    score = 1.0
    for p in mismatch_positions:
        score *= (1.0 - penalty[p])
    return score


print("\nOFF-TARGET SCORE (simplified CFD)")
for pos_set in ([18], [11, 18], [3, 11, 18]):
    print(f"  mismatches at {pos_set}: score {cfd(pos_set):.5f}")


# ------------------------------------------------------------- search space
GENOME_BP = 3_100_000_000
PAM_SITES = GENOME_BP / 8          # NGG on both strands
print("\nSEARCH SPACE")
print(f"  genome            : {GENOME_BP:,} bp")
print(f"  NGG sites         : {PAM_SITES:,.0f} (1 per 8 bp, both strands)")
total = 0
for k in range(4):
    comb = math.comb(20, k) * (3 ** k)
    total += comb
    print(f"  up to {k} mismatches: {comb:,} guide variants")
print(f"  total guides      : {total:,}")
print(f"  guide x site pairs: {total * PAM_SITES:.3e}")
```

Output:

```text
READ CLASSIFICATION
  ins  del  sub  net  edited  frameshift
    0    0    0    0  False   False
    0    0    0    0  False   False
    0    0    0    0  False   False
    0    0    0    0  False   False
    0    1    0   -1  True    True
    0    6    0   -6  True    False
    1    0    0    1  True    True
    0    3    0   -3  True    False
    2    0    0    2  True    True
    0    0    1    0  False   False

EDITING SUMMARY (n=10)
  edited reads      : 5 (50.0%)
  frameshift reads  : 3 (30.0%)
  in-frame or silent: 2 (20.0%)
  LOD 1.0000%: 298 reads for 95% confidence
  LOD 0.1000%: 2994 reads for 95% confidence
  LOD 0.0100%: 29956 reads for 95% confidence

OFF-TARGET SCORE (simplified CFD)
  mismatches at [18]: score 0.02000
  mismatches at [11, 18]: score 0.00400
  mismatches at [3, 11, 18]: score 0.00320

SEARCH SPACE
  genome            : 3,100,000,000 bp
  NGG sites         : 387,500,000 (1 per 8 bp, both strands)
  up to 0 mismatches: 1 guide variants
  up to 1 mismatches: 60 guide variants
  up to 2 mismatches: 1,710 guide variants
  up to 3 mismatches: 30,780 guide variants
  total guides      : 32,551
  guide x site pairs: 1.261e+13
```

Reading the output:

- The six-base deletion is in frame because six is a multiple of three. This is correct and is the
  central reason a frameshift count differs from an editing count.
- The single substitution is not counted as edited, because the classifier counts indels. A
  substitution at the cut site is a real modification and must be counted in a separate class.
- The limit of detection scales approximately with the reciprocal of the depth: 1% needs about
  300 reads, 0.1% about 3,000, and 0.01% about 30,000.
- The simplified CFD score falls from 0.02 for a single PAM-proximal mismatch to 0.0032 for three
  mismatches, a six-fold change. The real CFD model uses measured penalty tables; the structure of
  the calculation is the same.
- The search space is about 1.26 x 10^13 guide-site pairs. This is why off-target search uses
  indexed genomes and bit-parallel matching rather than aligning each guide to each site.

## 8. Data, standards, and regulatory context

### 8.1 The FDA genome editing guidance

The FDA finalized "Human Gene Therapy Products Incorporating Human Genome Editing" in January
2024. The Federal Register notice was published on 30 January 2024 at 89 FR 5913, under docket
FDA-2021-D-0398. The guidance applies to somatic cell editing, ex vivo and in vivo, and to
nuclease-dependent and nuclease-independent approaches.

It names three categories of risk: off-target editing, unintended consequences of on-target
editing, and unknown long-term effects. For each, it describes the information an investigational
new drug application should contain, covering product design, manufacturing and testing,
nonclinical assessment, and clinical trial design.

The guidance is a recommendation, not a statute. Compliance is demonstrated through the IND
process, and the review division can ask for more than the guidance describes.

### 8.2 Related FDA guidance

| Guidance | Issued | What it covers |
|---|---|---|
| Long Term Follow-Up After Administration of Human Gene Therapy Products | January 2020 | Duration and design of follow-up, including 15 years for integrating vectors and editing products |
| Chemistry, Manufacturing, and Control Information for Human Gene Therapy INDs | January 2020 | CMC content of an IND |
| Testing of Retroviral Vector-Based Human Gene Therapy Products for Replication Competent Retrovirus | 2020 | RCR testing |
| Human Gene Therapy Products Incorporating Human Genome Editing | January 2024 | Genome editing-specific recommendations |
| NGS-based safety assessment of genome editing (draft) | 2026 | Sequencing-based off-target and on-target assessment |

The last row reflects the direction of travel: the agency is moving toward sequencing-based
evidence as the primary safety dataset, which makes the bioinformatics pipeline part of the
regulatory submission rather than a supporting analysis.

### 8.3 Data standards

| Standard | What it covers | Binding status |
|---|---|---|
| HIPAA | Protected health information | Statute in the United States |
| 21 CFR Part 312 | IND requirements | Regulation |
| 21 CFR Part 11 | Electronic records and signatures | Regulation |
| ICH Q5A | Viral safety of biologics | International guideline |
| ICH Q5B | Analysis of expression constructs | International guideline |
| HL7 FHIR | Exchange of clinical data | Standard, not specific to gene therapy |
| MIxS | Metadata for sequencing | Community convention |

The absence of a gene-therapy-specific data standard means that programs define their own
schemas, and cross-program comparison is difficult. A submission should document the schema.

### 8.4 What the standards do not cover

- No standard defines a universal off-target acceptance criterion. The criterion is
  product-specific and is negotiated with the review division.
- No standard defines the required depth for off-target sequencing. The depth follows from the
  limit of detection the program claims.
- No standard defines a potency assay for a given mechanism.
- No standard resolves the tension between a 15-year follow-up expectation and a patient
  population that may be lost to follow-up.

Those gaps are why the diagnostics in section 6 and the tests in section 5.15 matter.

## 9. Industry practice and case evidence

### 9.1 The operating model for an ex vivo therapy

An autologous ex vivo therapy is a logistics business as much as a biology business. The chain
has seven steps, and each has a data artifact.

| Step | Activity | Data artifact | Typical duration |
|---|---|---|---|
| 1 | Mobilization and apheresis | Cell count, viability, CD34-positive fraction | 1 to 2 days |
| 2 | Cryopreservation and shipment | Temperature log, chain of custody | 1 to 3 days |
| 3 | Manufacturing | Editing efficiency, cell dose, potency | 2 to 6 weeks |
| 4 | Release testing | Sterility, identity, vector copy number or edit fraction | 1 to 2 weeks |
| 5 | Shipment to site | Chain of custody, thaw instructions | 1 to 3 days |
| 6 | Conditioning and infusion | Engraftment markers | 2 to 6 weeks |
| 7 | Follow-up | Efficacy and safety endpoints | Years |

The vein-to-vein time, from apheresis to infusion, is a product specification. A manufacturing
delay can force a second collection, and a second collection is a clinical event, not a
logistics event.

### 9.2 What the approvals changed

The 2023 approvals of Casgevy and Lyfgenia established three precedents.

1. **A genome editing product can be approved.** The regulatory pathway does not require a
   traditional small molecule or protein characterization package; it requires a product-specific
   package covering the edit, the delivery, and the manufacturing.
2. **Two modalities can be approved for the same indication on the same day.** Casgevy edits a
   regulatory element; Lyfgenia adds a gene with a lentiviral vector. The review pathways are
   similar, and the risk profiles are different.
3. **Long-term follow-up is part of the product.** The approval includes a follow-up obligation
   that extends beyond the trial.

The inference that these precedents generalize to in vivo editing is not established. An in vivo
edit has different biodistribution, different redosing, and different off-target tissue
questions.

### 9.3 The base and prime editing frontier

Base editing and prime editing are earlier in the clinic, and the public data are thinner.

- Base editing has been tested ex vivo for sickle cell disease and in vivo for cholesterol
  lowering. The BEACON study of BEAM-101 reported engraftment and fetal hemoglobin above 60% in an
  interim update in 2025. The in vivo programs established that a lipid nanoparticle can deliver a
  base editor to the liver and change a circulating biomarker.
- Prime editing entered human testing with PM359 in chronic granulomatous disease, reported in
  December 2025. The reported engraftment and functional neutrophil recovery are the first human
  data for the modality.

The practical implication for a data pipeline is that the assays must be modality-specific. A
base editing program measures the converted fraction and the bystander fraction; a prime editing
program measures the intended edit, the indels at the nick, and the pegRNA-derived unintended
edits. A nuclease amplicon pipeline is not sufficient for either.

### 9.4 Manufacturing and the CDMO relationship

Most sponsors do not manufacture their own vector or cell product. They work with a contract
development and manufacturing organization (CDMO), and the data pipeline crosses an organizational
boundary.

Three consequences follow.

- **Assay transfer is a validation event.** The sponsor's acceptance criteria must be reproduced at
  the CDMO, and a method transfer that changes the limit of detection changes the release decision.
- **Data access is contractual.** The sponsor may not have access to raw sequencing reads, only to
  summary tables. A submission that needs to reanalyze the raw data must negotiate that access
  before the batch is made.
- **Change control is slow.** A change to the editing protocol or the vector genome requires
  comparability studies, which are themselves sequencing experiments.

### 9.5 What is not yet established

- No gene editing therapy has been approved for in vivo delivery.
- No base editing or prime editing therapy has been approved.
- The durability of in vivo AAV factor expression beyond a few years is variable across products.
- The reimbursement model for multi-million-dollar one-time therapies is unresolved in most
  health systems.
- The long-term follow-up completion rate for patients who move or change providers is unknown.

A program should treat these as open risks rather than as solved problems.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Guide efficiency does not transfer | Editing low in the intended cell type | Model trained on a different cell type or delivery | No therapeutic effect | Test the guide in the intended cell type |
| Off-target panel incomplete | Sites found by unbiased methods are missing from the panel | Prediction used as the only source | Unmeasured risk | Combine prediction with GUIDE-seq or CIRCLE-seq |
| Off-target LOD not reported | Table of zeros without depths | Negative result reported without a limit | Uninterpretable safety claim | Report depth and LOD per site |
| In-frame indels counted as knockouts | Editing efficiency exceeds functional effect | Frameshift fraction not computed | Potency overestimated | Report the spectrum and the frameshift fraction |
| Large deletions missed | Amplicon shorter than the deletion | Polymerase chain reaction cannot amplify the product | Underestimated genotoxicity | Long-read sequencing and droplet digital PCR |
| Translocations missed | Only single-site assays | Dual breaks on different chromosomes | Oncogenic risk | Chromosome-specific translocation assays |
| Vector genome truncated | Titer high but potency low | ITR or packaging constraints | Non-functional product | Long-read sequencing across the genome |
| Empty capsids dominate | High particle titer, low infectious titer | Packaging inefficiency | Immune load without payload | Full/empty measurement by AUC or charge detection |
| Residual host DNA high | Batch variability | Purification failure | Immunogenic and oncogenic risk | Shotgun sequencing and process control |
| Potency assay not linked to mechanism | Surrogate marker moves independently | Assay chosen for convenience | Batch release does not predict effect | Qualify against a functional readout |
| Conditioning toxicity understated | Early discontinuations | Myeloablative regimen | Morbidity and mortality unrelated to the edit | Include in the risk-benefit and consent |
| Insertional oncogenesis | Clonal expansion in culture | Vector integration near an oncogene | Malignancy | Integration site analysis and follow-up |
| Replication-competent virus | Unexpected replication in culture | Recombination during manufacturing | Dissemination | RCR testing at the stated sensitivity |
| Long-term follow-up lost | Patients lost after year two | Mobility and care transitions | Late events unassessed | Registry linkage and decentralized follow-up |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The guide cuts the intended site in the intended cells | The whole mechanism depends on it | Cell-type-specific chromatin and delivery | Edit the intended cell type and measure |
| The reference genome represents the patient | Off-target prediction uses a reference | Common variants create or remove sites | Personal genome or variant-aware search |
| The predicted off-target list is complete enough | The panel is built from it | Prediction misses sites | Pair prediction with an unbiased method |
| The amplicon assay sees the relevant edits | Editing is inferred from amplified reads | Large deletions and complex rearrangements | Long-read and droplet digital PCR |
| The limit of detection is below the acceptance criterion | A negative result must mean something | Insufficient depth | Compute LOD from depth |
| The editing fraction predicts the functional effect | Potency and dosing depend on it | In-frame edits, clonal selection | Functional assay |
| The vector genome is intact | The product must express the transgene | Truncation during packaging | Long-read sequencing |
| The manufacturing process is reproducible | Batch release requires it | Process drift, CDMO transfer | Comparability studies and trends |
| The nonclinical model predicts human risk | Toxicology supports the first dose | Species differences in immunity and biodistribution | Relevant model and dose scaling |
| Long-term risk is captured by follow-up | Late events must be detected | Loss to follow-up | Registry linkage and consented sampling |

## 12. Exercises

### Exercise 1. Classify an indel spectrum

**Task.** Given 10,000 amplicon reads with 1,400 unmodified, 4,600 one-base deletions, 900 two-base
deletions, 700 three-base deletions, 1,500 deletions of four bases or more, and 900 insertions,
compute the editing fraction, the frameshift fraction, and the in-frame fraction.

**Deliverable.** The arithmetic and a one-sentence interpretation.

**Acceptance criteria.** Editing 86.0%, frameshift 79.0% under the stated assumption that all
insertions and deletions of four bases or more are frameshifts, in-frame 7.0%. The answer states
that the relevant number depends on whether the mechanism requires a frameshift.

**Time.** 25 minutes.

### Exercise 2. Compute a limit of detection

**Task.** An off-target site is sequenced to 12,000 reads. Compute the LOD at 95% confidence and
state whether the assay can exclude editing at 0.05%.

**Deliverable.** The calculation and the conclusion.

**Acceptance criteria.** LOD = ln(0.05)/ln(1 - f) solved for f at N = 12,000 gives approximately
0.00025, or 0.025%. The assay can exclude editing above 0.025%, so it can exclude 0.05%.

**Time.** 20 minutes.

### Exercise 3. Compute cell-level editing

**Task.** Allele-level editing is 65%. Compute the fraction of cells with at least one edited
allele and with both alleles edited, assuming independence. State which number matters for a
biallelic knockout.

**Deliverable.** The arithmetic.

**Acceptance criteria.** At least one allele 1 - 0.35^2 = 0.8775, or 87.75%. Both alleles 0.4225,
or 42.25%. The answer states that a biallelic knockout requires the second number.

**Time.** 20 minutes.

### Exercise 4. Choose a delivery route

**Task.** For each case, choose a route and justify it: a liver enzyme deficiency requiring
long-term correction; a hematopoietic stem cell defect requiring a permanent edit; a transient
lipid-lowering edit in hepatocytes.

**Deliverable.** A three-row table with route, persistence, and main risk.

**Acceptance criteria.** AAV for long-term liver correction, with the packaging and immunity
caveats; ex vivo electroporation for the stem cell edit, with conditioning risk; lipid nanoparticle
for the transient edit, with redosing and biodistribution caveats.

**Time.** 30 minutes.

### Exercise 5. Design an off-target package

**Task.** Design the off-target assessment for a nuclease program: discovery method, panel size,
depth, LOD, controls, and acceptance criteria.

**Deliverable.** A two-page package outline.

**Acceptance criteria.** The package combines an unbiased discovery method with a targeted panel,
states a depth that supports the acceptance criterion, includes a positive control site and a
negative control, and defines what happens if a site is positive.

**Time.** 50 minutes.

### Exercise 6. Diagnose a batch failure

**Task.** A batch has a high particle titer, a low infectious titer, a full capsid fraction of 22%,
and a potency result below specification. Identify the most likely cause, the confirmatory test,
and two process changes.

**Deliverable.** A root-cause memo.

**Acceptance criteria.** The memo identifies empty capsid excess or genome truncation as the
candidates, proposes long-read genome sequencing and AUC or charge detection mass spectrometry, and
suggests changes to the packaging plasmid ratio or the purification step.

**Time.** 40 minutes.

### Exercise 7. Map the regulatory path

**Task.** For an in vivo base editing program targeting the liver, list the FDA guidance documents
that apply, the nonclinical studies likely required, and the long-term follow-up expectation.

**Deliverable.** A regulatory map with citations and dates.

**Acceptance criteria.** The map cites the January 2024 genome editing guidance, the January 2020
long-term follow-up guidance, and the CMC guidance, and notes that no in vivo editing product has
been approved as of this writing.

**Time.** 45 minutes.

### Exercise 8. Critique an editing claim

**Task.** A press release states that a therapy "achieved 90% gene correction with no off-target
editing". Write a critique with at least four questions.

**Deliverable.** A 350-word critique.

**Acceptance criteria.** The critique asks which cell type and assay, whether 90% is read-level or
allele-level, what the limit of detection was and at what depth, and whether an unbiased
off-target method was used. It notes that "no off-target editing" without an LOD is not a result.

**Time.** 30 minutes.

## 13. Self-check answers

**Exercise 1.** Total edited reads = 4,600 + 900 + 700 + 1,500 + 900 = 8,600, so editing is 86.0%.
Frameshift reads, under the stated assumption, are 4,600 + 900 + 1,500 + 900 = 7,900, or 79.0%.
The in-frame fraction is 700, or 7.0%. The unmodified fraction is 14.0%, and the classes sum to
100%. For an enhancer disruption, any deletion can abolish the element, so 86.0% is the relevant
number. For a coding knockout, 79.0% is the relevant number, and reporting 86.0% would overstate
the functional effect.

**Exercise 2.** Solve 12,000 = ln(0.05)/ln(1 - f). Then ln(1 - f) = -2.9957/12,000 = -0.00024964,
so 1 - f = exp(-0.00024964) = 0.9997504, and f = 0.0002496, or 0.025%. The assay can exclude
editing above 0.025% at 95% confidence, so it can exclude 0.05% with a two-fold margin. It cannot
exclude 0.01%.

**Exercise 3.** The probability that a cell has no edited allele is (1 - 0.65)^2 = 0.1225. The
probability of at least one edited allele is 1 - 0.1225 = 0.8775, or 87.75%. The probability of two
edited alleles is 0.65^2 = 0.4225, or 42.25%. A biallelic knockout requires the second number,
which is less than half the cells, even though the allele-level editing looks high.

**Exercise 4.** Long-term liver correction: AAV, episomal and long-lived in hepatocytes, limited
by a packaging capacity of about 4.7 kilobases and by pre-existing neutralizing antibodies.
Permanent stem cell edit: ex vivo electroporation of CD34-positive cells, permanent in the graft,
limited by myeloablative conditioning. Transient lipid-lowering edit: lipid nanoparticle,
transient and redosable, limited by biodistribution and innate immunity. A correct answer states
that the vector choice follows from the durability requirement, not from the efficiency alone.

**Exercise 5.** Rubric. Full credit requires an unbiased discovery method, a targeted panel that
includes the discovered sites, a depth justified by the acceptance criterion, a positive and a
negative control, and a pre-specified action for a positive site. A package built only on
prediction fails.

**Exercise 6.** Rubric. Full credit identifies empty capsids or truncated genomes, proposes the
right confirmatory assays, and suggests process changes with a rationale. A memo that blames the
potency assay without checking the capsid data fails.

**Exercise 7.** Rubric. Full credit cites the 2024 genome editing guidance, the 2020 long-term
follow-up guidance, and the CMC guidance, describes nonclinical off-target and toxicology studies,
and states the 15-year follow-up expectation for editing products. It also notes the absence of an
approved in vivo editing product.

**Exercise 8.** Rubric. Full credit asks about cell type, assay, read-level versus allele-level
fraction, depth, LOD, and the unbiased method. A critique that accepts the claim because the
number is high fails.

## 14. Interview questions

**1. What is the difference between nuclease editing, base editing, and prime editing?**
A strong answer describes the double-strand break and repair spectrum, the direct chemical
conversion without a break, and the reverse-transcriptase write, and names the different assays.
A weak answer treats all three as "CRISPR".

**2. Why is "80% editing" an incomplete result?**
A strong answer explains that the spectrum matters: frameshift versus in-frame, large deletions,
and the fraction with the intended change. A weak answer accepts the number as sufficient.

**3. How do you compute the limit of detection of an off-target assay?**
A strong answer uses the zero-observation probability and the depth, and states the confidence
level. A weak answer says "sequence deeply".

**4. Why is an AAV limited to about 4.7 kilobases?**
A strong answer describes the packaging capacity of the capsid and the consequence for large
genes or multiple cassettes. A weak answer says AAV is "small".

**5. What is the difference between full and empty capsids, and why does it matter?**
A strong answer explains that empty capsids add antigen load without delivering the genome, and
names the measurement methods. A weak answer treats the ratio as a manufacturing detail.

**6. What does the FDA genome editing guidance require?**
A strong answer names the January 2024 final guidance, the three risk categories, and the IND
content areas. A weak answer says "follow FDA guidance" without specifics.

**7. How would you detect a large deletion that an amplicon assay misses?**
A strong answer proposes long-read sequencing or droplet digital polymerase chain reaction, and
explains that the amplicon cannot amplify a product that no longer exists. A weak answer proposes
deeper amplicon sequencing.

**8. Why is an in vivo edit different from an ex vivo edit?**
A strong answer covers persistence, redosing, biodistribution, and the tissue-specific off-target
question. A weak answer says one is "inside the body".

**9. What is a potency assay, and what makes it good?**
A strong answer defines it as a mechanism-relevant measure with a defined range and precision, and
distinguishes it from a surrogate. A weak answer says it measures "how well the product works".

**10. What would make you halt a program?**
A strong answer names an off-target signal above the acceptance criterion at a site of concern, a
batch that cannot meet specification, a potency assay disconnected from the mechanism, or a
nonclinical safety signal. A weak answer says "if it does not work".

## 15. Key terms

- **Adeno-associated virus (AAV)** — small DNA virus used for in vivo gene delivery.
- **Base editor** — enzyme that converts one base to another without a double-strand break.
- **Bystander edit** — unintended base conversion in the editing window.
- **Cas9** — RNA-guided endonuclease.
- **CFD score** — position-weighted off-target score for a mismatched guide.
- **CHANGE-seq** — cell-free off-target profiling method.
- **CIRCLE-seq** — cell-free off-target detection method.
- **CMC** — chemistry, manufacturing, and controls.
- **Conditioning** — preparative regimen before cell infusion.
- **DISCOVER-seq** — repair-protein mapping method.
- **Double-strand break** — break in both DNA strands.
- **Empty capsid** — AAV particle without a vector genome.
- **Engraftment** — establishment of infused cells in the marrow.
- **Frameshift** — indel length not divisible by three.
- **Full capsid** — AAV particle containing the vector genome.
- **GUIDE-seq** — cellular off-target detection method.
- **Homology-directed repair** — templated repair producing a precise change.
- **IND** — investigational new drug application.
- **Indel** — insertion or deletion.
- **Insertional oncogenesis** — cancer caused by vector integration.
- **ITR** — inverted terminal repeat required for AAV packaging.
- **Limit of detection** — smallest fraction distinguishable from zero.
- **Lipid nanoparticle** — non-viral delivery particle.
- **Long-term follow-up** — extended post-administration monitoring.
- **Non-homologous end joining** — error-prone repair producing indels.
- **Off-target editing** — editing at an unintended site.
- **PAM** — protospacer adjacent motif.
- **pegRNA** — prime editing guide RNA.
- **Potency** — ability to produce the intended effect.
- **Prime editor** — nickase fused to reverse transcriptase.
- **Replication-competent virus** — virus able to replicate after manufacturing.
- **sgRNA** — single guide RNA.
- **Transgene** — delivered gene sequence.
- **Vector copy number** — vector genomes per cell.

## 16. Sources

Accessed 26 October 2026 unless otherwise stated.

- United States Food and Drug Administration. *Human Gene Therapy Products Incorporating Human
  Genome Editing; Guidance for Industry*, January 2024.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/human-gene-therapy-products-incorporating-human-genome-editing
- Federal Register. *Human Gene Therapy Products Incorporating Human Genome Editing; Guidance for
  Industry; Availability*, 89 FR 5913, 30 January 2024.
  https://www.federalregister.gov/documents/2024/01/30/2024-01788/
- United States Food and Drug Administration. *Long Term Follow-Up After Administration of Human
  Gene Therapy Products; Guidance for Industry*, January 2020.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/long-term-follow-after-administration-human-gene-therapy-products
- United States Food and Drug Administration. *Chemistry, Manufacturing, and Control (CMC)
  Information for Human Gene Therapy Investigational New Drug Applications (INDs)*, January 2020.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/chemistry-manufacturing-and-control-cmc-information-human-gene-therapy-investigational-new-drug
- United States Food and Drug Administration. *Cellular and Gene Therapy Guidances*.
  https://www.fda.gov/vaccines-blood-biologics/biologics-guidances/cellular-gene-therapy-guidances
- Doudna, J. A., and Charpentier, E. *The New Frontier of Genome Engineering with CRISPR-Cas9*.
  Science 346, 1258096, 2014. https://doi.org/10.1126/science.1258096
- Tsai, S. Q., et al. *GUIDE-seq Enables Genome-Wide Profiling of Off-Target Cleavage by CRISPR-Cas
  Nucleases*. Nature Biotechnology 33, 187-197, 2015. https://doi.org/10.1038/nbt.3117
- Clement, K., et al. *CRISPResso2 Provides Accurate and Rapid Genome Editing Sequence Analysis*.
  Nature Biotechnology 37, 224-226, 2019. https://doi.org/10.1038/s41587-019-0032-3
- Li, W., et al. *MAGeCK Enables Robust Identification of Essential Genes from Genome-Scale
  CRISPR/Cas9 Knockout Screens*. Genome Biology 15, 554, 2014.
  https://doi.org/10.1186/s13059-014-0554-4
- Frangoul, H., et al. *CRISPR-Cas9 Gene Editing for Sickle Cell Disease and Beta-Thalassemia*.
  New England Journal of Medicine 384, 252-260, 2021. https://doi.org/10.1056/NEJMoa2031054
- Beam Therapeutics. *BEACON Study Update*, 2025.
  https://beamtx.com/wp-content/uploads/2025/06/BEAM-101-EHA_Final_22May2025.pdf
- Prime Medicine. *PM359 Clinical Data Publication in the New England Journal of Medicine*, 2025.
  https://investors.primemedicine.com/news-releases/news-release-details/prime-medicine-announces-new-england-journal-medicine
- *Precision Gene Editing Medicine Makes History*. Nature Biotechnology, 2025.
  https://doi.org/10.1038/s41587-025-02741-6

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`14_ai-for-protein-engineering.md`](14_ai-for-protein-engineering.md)
- Next topic: [`16_microbiome-analytics.md`](16_microbiome-analytics.md)
- Related topic: [`10_clinical-genomics.md`](10_clinical-genomics.md)
- Related topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Module survey: [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Laboratory exercises and datasets: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

