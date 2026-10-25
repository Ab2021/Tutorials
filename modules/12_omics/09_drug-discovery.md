# Drug Discovery: Target Identification, Validation, and What Computation Actually Contributes

> Module 12 — Omics and Precision Medicine
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, 05, topics 02, 03, 06, 07
> Version 2026.3

## 1. Why this topic exists

Clinical development has a failure problem, and the failure is concentrated at the front. Most
programs that enter phase 1 do not reach approval, and the largest single cause is lack of
efficacy, which usually means the target was wrong or the biology did not translate. A target
decision made in the first six months of a program constrains the next eight years and several
hundred million dollars. It is the highest-leverage decision in biopharma, and it is made with
less evidence than the later, cheaper decisions that follow it.

Omics data are the main source of that evidence. Human genetics asks whether a target is causal
in people; expression data ask where and in which cell type it is active; perturbation screens
ask whether cells depend on it; pathway data ask whether it sits in a coherent mechanism. Each
source has a characteristic failure mode, and the failure modes are not interchangeable. A
genetic association can be a linkage artifact; an expression correlation can be reverse
causation; a CRISPR dependency can be a pan-essential gene that kills every cell; an animal model
can be a model of the model rather than of the disease.

The industry question is not "can we find a target?" There are thousands of candidates. It is
"which target has enough causal, safety, and tractability evidence to justify a program?" That
question is answered with a structured evidence assessment, not with a single score.

This topic covers target identification and validation, the evidence types and their limits,
tractability and safety assessment, and the honest role of computational methods including
structure prediction and machine learning. It follows biomarker discovery (topic 08) and precedes
clinical genomics (topic 10).

### 1.1 The failure pattern

1. A GWAS locus is associated with a disease at genome-wide significance.
2. The nearest gene is nominated as the target.
3. Expression data show the gene is higher in cases, which is presented as validation.
4. A CRISPR screen shows the gene is essential in a cell line, which is presented as functional
   validation.
5. A program is launched without checking whether the association is driven by a different gene
   in the locus, whether the expression difference is causal or consequential, or whether the
   gene is essential in all cells.
6. Two years later, the tool compound has no in vivo effect, and the program cannot tell whether
   the target or the chemistry failed.

Steps 2 through 4 are the standard evidence package in many target reviews, and each step
contains an inferential leap that is not stated.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Discovery lead | Two years and a program budget | Depends on the target evidence |
| Medicinal chemist | Time optimizing a molecule for the wrong target | Receives a target, not the evidence |
| Translational lead | A biomarker strategy built on a false mechanism | Inherited the target |
| Portfolio committee | Capital allocation across programs | Sees a summary score, not the evidence |
| Patient | A therapy that never arrives | Has no role in target selection |
| Investor | A pipeline whose lead asset fails for a known reason | Sees the narrative |

A third industry reality is that the evidence package is assembled by different teams with
different incentives. Discovery teams are rewarded for nominating targets; chemistry teams are
rewarded for molecules; translational teams are rewarded for biomarkers. Nobody is rewarded for
stopping a program early, which is why pre-specified criteria and an independent review are the
only reliable protection against escalation of commitment.

The pipeline view matters too. A target decision is not only a scientific judgment; it is a
resource allocation. The same evidence that justifies a program also justifies a competing
program, and the portfolio question is which combination of targets maximizes the probability of
at least one success within the available capital. That question is covered in more depth in the
pharma development module, but it shapes what a target summary must contain.

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

For a target, the question becomes: is the association causal in humans, is the gene's function
required for the disease phenotype, and can a molecule modulate it safely? Those are three
questions, and a program needs an answer to all three before it spends on chemistry.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** target identification, target validation, and tractability, and name the
   evidence that supports each.
2. **Evaluate** human genetic evidence, including the difference between a locus and a gene.
3. **Interpret** Mendelian randomization estimates and their assumptions.
4. **Use** expression, perturbation, and pathway evidence without overreading it.
5. **Assess** tractability and safety liabilities early enough to change a decision.
6. **Compute** the expected change in success probability from genetic support.
7. **Critique** an AI-driven target or molecule claim against the evidence it actually provides.
8. **Write** a target evidence summary with a go/no-go recommendation.
9. **Separate** a benchmark result from a clinical claim when evaluating a computational tool.
10. **Recognise** when a target is a biology program without a viable modality.

## 3. The core mental model

Target selection is an evidence-weighing problem under uncertainty, with three independent axes:
causality, tractability, and safety. A target must score acceptably on all three; strength on one
does not compensate for failure on another.

```text
  Human genetics (GWAS, rare variants, MR, colocalization)
        |
        +--> Is the gene causal in people?
        |
  Expression and localization (tissue, cell type, condition)
        |
        +--> Where and when is it active?
        |
  Perturbation (CRISPR, RNAi, chemical probes, animal models)
        |
        +--> Does the system depend on it?
        |
  Pathway and network
        |
        +--> Is there a coherent mechanism?
        v
  Target candidate
        |
        +--> Tractability: is there a binding site, a modality, a chemical start?
        +--> Safety: is it essential, broadly expressed, or in a risky pathway?
        +--> Biomarker: can we measure engagement and select patients?
        v
  Program decision
```

### 3.1 The three questions

**Causality.** Does modulating the target change the disease in humans? Genetic evidence is the
strongest available, because it is inherited, precedes disease, and is not affected by reverse
causation. But a genetic association is a property of a locus, not a gene, and the causal gene
must be established by fine-mapping, colocalization, or functional evidence.

**Tractability.** Can a molecule reach the target and change its function? A secreted protein
with no binding pocket is hard; a nuclear transcription factor is hard; a kinase with a conserved
ATP site is easy but selectivity is hard. Tractability is a property of the target class and the
modality, and it determines the chemistry timeline.

**Safety.** Is the target required for normal physiology? A gene that is essential in every cell
is a poor target unless the therapeutic window is achievable. Expression breadth, knockout
phenotypes, and pathway position are the evidence.

### 3.2 The evidence hierarchy

| Evidence type | Strength | Characteristic failure |
|---|---|---|
| Mendelian disease gene | Strong | Rare, may not generalize to common disease |
| Human genetics with colocalization | Strong | Locus complexity, ancestry limits |
| Mendelian randomization | Moderate to strong | Pleiotropy, weak instruments |
| Perturbation in disease-relevant cells | Moderate | Cell line artifacts, pan-essentiality |
| Animal model | Moderate | Model does not recapitulate human disease |
| Expression correlation | Weak | Reverse causation, composition |
| Pathway membership | Weak | Pathway databases are incomplete |
| Literature frequency | Very weak | Popularity bias |

The hierarchy is not a scoring system. A weak evidence type can be decisive if it is the only
human evidence, and a strong evidence type can be wrong if the locus is misassigned.

### 3.3 What genetic support buys

The strongest quantitative claim in the field is that targets with human genetic support have
roughly twice the success rate in clinical development. Nelson et al. (2015) estimated a ratio
of 2.0 (95% CI 1.6-2.4) for phase 1 to approval, and Minikel et al. (2024) updated the estimate
to about 2.6 with a decade more data.

The arithmetic matters for portfolio decisions:

```text
baseline phase 1 to approval success ~8%
with genetic support ~16% (2x) or ~21% (2.6x)
```

A doubling of success probability changes the expected value of a program substantially, which
is why genetic evidence is now a standard part of target review. It does not guarantee success:
most genetically supported targets still fail.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Target | Molecule whose modulation is intended to change disease | Confused with the gene | Protein or pathway |
| Target identification | Finding candidate targets | Confused with validation | Stage |
| Target validation | Evidence that modulation changes the phenotype | Confused with tractability | Stage |
| Tractability | Feasibility of modulating the target | Confused with validation | Assessment |
| Druggability | Presence of a binding site for a drug-like molecule | Confused with safety | Assessment |
| Ligandability | Ability to bind a small molecule | Used loosely | Assessment |
| GWAS | Genome-wide association study | Treated as gene discovery | Study |
| Locus | Genomic region associated with a trait | Confused with a gene | Region |
| Fine-mapping | Narrowing a locus to causal variants | Assumed to identify the gene | Method |
| Colocalization | Shared causal variant between two traits | Confused with correlation | Posterior probability |
| eQTL | Expression quantitative trait locus | Assumed to act in the relevant tissue | Association |
| pQTL | Protein quantitative trait locus | Assumed to equal function | Association |
| Mendelian randomization (MR) | Instrumental-variable estimate using genetic variants | Treated as a trial | Estimate |
| Instrument | Variant used as a proxy for exposure | Assumed valid | Variant |
| Pleiotropy | Instrument affects outcome through another path | Ignored | Bias |
| Wald ratio | MR estimate from one variant | Assumed precise | Ratio |
| F-statistic | Instrument strength | Ignored | Statistic |
| L2G | Locus-to-gene score | Treated as proof | Score |
| DepMap | Cancer cell line dependency map | Treated as in vivo truth | Dataset |
| Essentiality | Requirement for cell viability | Confused with disease relevance | Score |
| Pan-essential | Essential in most cell lines | Ignored | Class |
| Selectivity | Preference for target over related proteins | Confused with potency | Fold |
| Potency | Concentration for a given effect | Confused with efficacy | IC50, EC50 |
| ADMET | Absorption, distribution, metabolism, excretion, toxicity | Treated as a checklist | Properties |
| Hit | Molecule with confirmed activity | Confused with lead | Compound |
| Lead | Optimized starting point | Confused with candidate | Compound |
| Candidate | Molecule entering development | Confused with lead | Compound |
| Phenotypic screen | Screen for a phenotype without a target | Assumed to reveal mechanism | Method |
| Target-based screen | Screen against a specific target | Assumed to predict efficacy | Method |
| PROTAC | Degrader recruiting an E3 ligase | Treated as a small-molecule inhibitor | Modality |
| Molecular glue | Compound inducing protein interaction | Confused with inhibitor | Modality |
| Structure prediction | Computational protein structure | Treated as experimental structure | Model |
| Docking | Computational binding pose prediction | Treated as affinity prediction | Method |
| Biomarker | Measurable indicator of target engagement or effect | Confused with efficacy | Measurement |
| Proof of concept (PoC) | Evidence of biological effect in humans | Confused with approval | Stage |

Three distinctions matter most.

**A locus is not a gene.** A GWAS signal marks a region, often containing dozens of genes, and
the nearest gene is a guess. Fine-mapping identifies candidate causal variants; colocalization
with an eQTL or pQTL provides evidence for a specific gene; functional experiments provide more.
Without these, "the GWAS gene" is shorthand for "a gene near the signal."

**Druggability is not validation.** A beautifully tractable kinase that is not causal produces a
potent molecule with no efficacy. The two assessments are independent, and programs that
conflate them choose targets because they are easy to drug.

**A cell-line dependency is not a disease dependency.** DepMap measures whether a gene is
required for a cancer cell line to grow in culture. Many dependencies are pan-essential, and many
disease-relevant dependencies do not appear in cell lines. The screen is a filter, not a
validation.

## 5. The mechanics, step by step

### Step 1. Define the unmet need and the proof-of-concept

**Input:** the disease and the current standard of care.

**Output:** a target product profile and a PoC definition.

**Rationale:** The target must be judged against the effect size the market and the disease
require. A 20% reduction in a biomarker is not a PoC if the disease needs a 50% reduction in
events. Writing the PoC first prevents the post hoc selection of an endpoint that happened to
move.

### Step 2. Assemble human genetic evidence at the locus

**Input:** GWAS summary statistics, rare-variant data, sequencing cohorts.

**Output:** a locus map with credible sets, fine-mapped variants, and candidate genes.

**Rationale:** The locus is the unit of genetic evidence, and the gene is the inference. Record
the credible set size, the ancestry composition of the cohort, and the strength of the
association. A locus with a large credible set and no functional annotation is a lead, not a
target.

### Step 3. Establish the causal gene

**Input:** locus, eQTL and pQTL data, chromatin interaction data, colocalization.

**Output:** a ranked gene list with colocalization posteriors and direction of effect.

**Rationale:** Colocalization asks whether the same variant drives both the disease association
and the molecular trait. A high posterior for a shared variant is strong evidence for that gene;
a high posterior for two distinct variants is evidence against. The direction matters: a variant
that raises expression and raises disease risk supports an inhibitor, while the reverse supports
an agonist.

### Step 4. Test causality with Mendelian randomization

**Input:** instrument variants and outcome summary statistics.

**Output:** an MR estimate with an interval and diagnostics.

**Rationale:** MR uses genetic variants as instruments for an exposure. Its assumptions are
relevance (the variant affects the exposure), independence (no confounding), and exclusion
restriction (the variant affects the outcome only through the exposure). Pleiotropy violates the
third, and it is common. Report the instrument F-statistic, the number of variants, the
heterogeneity statistic, and a sensitivity analysis such as MR-Egger.

### Step 5. Map expression and localization

**Input:** tissue and single-cell expression data.

**Output:** a tissue and cell-type map with condition-specific changes.

**Rationale:** A target expressed only in the disease-relevant cell type is more attractive than
one expressed everywhere, both for efficacy and for safety. Expression correlation with disease
is weak evidence of causality but strong evidence of localization, which is what it should be
used for.

### Step 6. Test perturbation in disease-relevant systems

**Input:** CRISPR or RNAi screens, chemical probes, organoids, primary cells.

**Output:** dependency and phenotype data with specificity controls.

**Rationale:** Perturbation establishes that the system responds to modulation. The controls
matter: a pan-essential gene will score in every screen, and an off-target effect can mimic a
phenotype. Use multiple guides, rescue experiments, and disease-relevant cell models.

### Step 7. Assess tractability and modality

**Input:** protein structure, domain annotation, known ligands, localization.

**Output:** a modality recommendation and a chemistry risk assessment.

**Rationale:** The target class determines the modality: kinases and proteases favor small
molecules; secreted ligands and receptors favor antibodies; intracellular proteins without
pockets may require degradation or a genetic modality. A target with no tractable modality is a
biology program, not a drug program.

### Step 8. Assess safety liabilities

**Input:** expression breadth, knockout phenotypes, pathway position, related proteins.

**Output:** a safety risk list with mitigation options.

**Rationale:** The earliest safety assessment is expression and genetics: a gene with a severe
Mendelian loss-of-function phenotype in humans is a warning, and a gene with no human knockout
phenotype is more permissive. Selectivity against related proteins is part of safety, because
off-target activity in a paralog can produce the toxicity that the on-target effect avoids.

### Step 9. Build the biomarker strategy

**Input:** the mechanism and the expected effect.

**Output:** a target engagement biomarker and a pharmacodynamic biomarker.

**Rationale:** A program needs to know whether the drug hit the target before it can interpret a
negative efficacy result. Without an engagement biomarker, a failed trial cannot distinguish a
bad molecule from a bad target, and the program learns nothing.

### Step 10. Design the assay cascade

**Input:** the modality and the target.

**Output:** a screening cascade with potency, selectivity, and ADMET filters.

**Rationale:** The cascade encodes the program's quality thresholds. Primary assays measure
activity; secondary assays measure selectivity and mechanism; ADMET assays measure developability.
A cascade that optimizes potency before selectivity produces molecules that are potent and
non-selective.

### Step 11. Screen and triage

**Input:** compound libraries or design ideas.

**Output:** hits with confirmed activity and chemical matter assessment.

**Rationale:** Screening is a filtering problem with a very low prior probability. The hit rate
for a novel target is often well below 1%, and most confirmed hits fail later. The triage must be
designed to fail compounds early and cheaply.

### Step 12. Optimize leads for potency, selectivity, and ADMET

**Input:** hit series.

**Output:** a candidate molecule with a developability package.

**Rationale:** Optimization is multi-objective, and the objectives conflict. Potency gains often
cost selectivity or solubility. The program needs a target product profile to decide which
trade-offs are acceptable.

### Step 13. Demonstrate in vivo proof of concept

**Input:** candidate molecule and disease model.

**Output:** exposure, engagement, and efficacy data.

**Rationale:** In vivo PoC must show that the molecule reaches the target at a tolerated dose and
that engagement produces the expected biology. A model that responds only at doses with
toxicity is not a PoC. Species differences in target biology are a common failure.

### Step 14. Make the go/no-go decision against pre-specified criteria

**Input:** the evidence package.

**Output:** a decision with a documented rationale.

**Rationale:** The criteria should be written before the data arrive, because after the data
every program can construct a rationale for continuing. A pre-specified bar for causality,
tractability, safety, and biomarker is the only protection against escalation of commitment.

### Step 15. Map the competitive and standard-of-care landscape

**Input:** pipeline and literature data for the indication.

**Output:** a competitive map with differentiation criteria.

**Rationale:** A target that is well validated and already crowded may be a worse program than a
target with weaker evidence and no competition. The map should record which modalities are in
development, which trials are running, and what the new program would do differently.

### Step 16. Define the translational strategy

**Input:** the target, the biomarker, and the indication.

**Output:** a phase 1 and phase 2 plan with decision points.

**Rationale:** Phase 1 should measure engagement and safety; phase 2 should test the mechanism
in the right population with the right endpoint. A plan that goes directly to an efficacy
endpoint without an engagement readout wastes the trial's information.

### Step 17. Assess portfolio fit

**Input:** the program, the portfolio, and the strategy.

**Output:** a recommendation with a resource and risk assessment.

**Rationale:** A scientifically sound target may not fit the portfolio: wrong modality
capability, wrong therapeutic area, or a risk profile the company cannot carry. A target summary
that ignores portfolio fit will be rejected for reasons the team does not understand.

### Modality selection

The target class constrains the modality, and the modality constrains the timeline and the cost.

| Target class | Preferred modality | Why | Typical risk |
|---|---|---|---|
| Kinase, protease, GPCR | Small molecule | Pocket, oral dosing | Selectivity |
| Secreted ligand or receptor | Antibody or biologic | Accessibility, specificity | Tissue penetration |
| Intracellular protein without a pocket | Degrader or molecular glue | Removes the protein rather than inhibiting | E3 ligase dependence |
| Transcription factor | Degrader or genetic modality | No classical pocket | Delivery, specificity |
| Ion channel | Small molecule | Pocket in the pore or allosteric site | Cardiac safety |
| Protein-protein interaction | Small molecule, peptide, or biologic | Interface inhibition is hard | Potency, permeability |
| RNA | Oligonucleotide | Sequence specificity | Delivery, stability |

A target with no entry in this table is a biology program. That is not a failure, but it should
be labelled, because the timeline and the budget are different from a small-molecule program.

### Safety evidence and how to read it

| Evidence | Interpretation | Caveat |
|---|---|---|
| Human loss-of-function phenotype | Strong safety signal | Rare variants may be more severe than a drug |
| Knockout mouse | Organ or developmental phenotype | Species differences |
| Expression breadth | Ubiquitous expression suggests risk | Expression is not function |
| Paralogue selectivity | Off-target risk | Conserved pockets |
| Pathway position | Upstream nodes have broader effects | Redundancy |
| Clinical trial adverse events for a related drug | Empirical signal | Class-specific |

The strongest safety evidence is human genetics, because it reflects the consequence of
modulating the target in people over a lifetime. A gene with a severe Mendelian loss-of-function
phenotype is a target where partial inhibition may be tolerable but complete inhibition is not.

### Branch points

**Branch A — the locus has no colocalized molecular trait.** The causal gene is unknown. Either
invest in fine-mapping and functional work, or deprioritize.

**Branch B — the gene is pan-essential.** The therapeutic window is likely narrow. Consider
tissue-specific delivery, a partial inhibitor, or a different target.

**Branch C — the target has no tractable modality.** Reframe as a biology program or a genetic
medicine opportunity, not a small-molecule program.

### Decision table: evidence to interpretation

| Evidence | Supports | Does not support | Check |
|---|---|---|---|
| GWAS locus | A region is associated | A specific gene is causal | Credible set size |
| eQTL colocalization | Gene expression mediates the association | Protein function | Tissue relevance |
| MR estimate | Causal effect under assumptions | A drug will work | Pleiotropy diagnostics |
| DepMap dependency | Cell-line requirement | In vivo efficacy | Pan-essentiality |
| Animal knockout | Requirement in that species | Human requirement | Human genetics |
| Expression correlation | Localization | Causality | Composition adjustment |

### Edge cases

**LD confounding.** Diagnostic: the eQTL and the disease signal are in linkage disequilibrium but
not the same variant. Consequence: the wrong gene is nominated. Mitigation: colocalization and
fine-mapping.

**Pleiotropy.** Diagnostic: MR estimates differ across variants, or MR-Egger intercept differs
from zero. Consequence: a biased causal estimate. Mitigation: sensitivity analyses and
multivariable MR.

**Weak instruments.** Diagnostic: F-statistic below 10. Consequence: weak-instrument bias toward
the confounded estimate. Mitigation: stronger instruments, or a different design.

**Tissue mismatch.** Diagnostic: the eQTL is measured in a tissue unrelated to the disease.
Consequence: the wrong isoform or gene. Mitigation: disease-relevant tissue or cell type.

**Pan-essentiality.** Diagnostic: the gene scores as essential in most cell lines. Consequence:
toxicity. Mitigation: selectivity, partial inhibition, or deprioritization.

**CRISPR off-target.** Diagnostic: the phenotype persists with a rescue or differs across guides.
Consequence: a false dependency. Mitigation: multiple guides and rescue.

**Animal model mismatch.** Diagnostic: the model's transcriptome differs from human disease.
Consequence: efficacy that does not translate. Mitigation: human tissue validation and
human-relevant models.

**Chemical matter gap.** Diagnostic: no series with acceptable properties after a year.
Consequence: a biology program without a drug. Mitigation: modality change or stoppage.

**IP blockage.** Diagnostic: the target or modality is covered by blocking patents. Consequence:
commercial failure despite scientific success. Mitigation: freedom-to-operate review early.

**Paralogue compensation.** Diagnostic: inhibiting the target produces no phenotype because a
paralogue compensates. Consequence: a potent molecule with no effect. Mitigation: dual inhibition
or a different node.

**Target expression lost in disease.** Diagnostic: expression is high in healthy tissue and low in
diseased tissue. Consequence: the target is irrelevant in the population to be treated.
Mitigation: expression data from the disease state, not the healthy state.

**Biomarker absence.** Diagnostic: no measurable engagement marker. Consequence: an
uninterpretable trial. Mitigation: develop one before the trial.

**Assay cascade mismatch.** Diagnostic: the primary assay measures a proximal biochemical effect
that does not translate to the cellular phenotype. Consequence: potent molecules with no cellular
activity. Mitigation: include a cell-based confirmation early in the cascade.

**Unvalidated disease model.** Diagnostic: the model has no human validation and responds to
treatments that failed in patients. Consequence: a PoC that does not predict clinical results.
Mitigation: choose models with human concordance data, and state the model's limitations.

A target summary that reports only the evidence that supports the program is not a summary; it is
an advocacy document. The format should force the contrary evidence into the same table, because
the decision is made on the balance, not on the best line.

The summary should also record what was not measured, because the gaps determine the next
experiment. A target with no engagement biomarker and no selectivity data has two clear next
steps, and naming them converts a judgment into a plan.

### What the procedure produces

The output is a target evidence summary: the locus and causal gene evidence, the MR estimate
with diagnostics, the expression and perturbation data, the tractability and safety assessment,
the biomarker plan, and the go/no-go criteria with the decision. The summary is the artifact a
portfolio committee can audit.

## 6. Worked example

A program is considering a target for a common inflammatory disease. The evidence package is
assembled from a GWAS locus, an eQTL resource, a perturbation screen, and expression data.

### 6.1 The locus

The GWAS identifies a locus with a lead variant at p = 3 x 10^-9 and an odds ratio of 1.12 per
allele. The locus contains 40 genes, and the credible set contains 12 variants.

A naive reading nominates the nearest gene. A structured reading asks which gene in the locus has
a colocalized molecular trait. The eQTL data show a cis-eQTL for gene G at the same locus, with a
colocalization posterior of 0.92 for a shared causal variant. For the other 39 genes, the
posterior is below 0.10.

The interpretation: the association is likely mediated by G's expression, not by the nearest
gene. The posterior of 0.92 is strong but not conclusive; a posterior of 0.92 still leaves an 8%
chance that the two traits share a locus by coincidence.

### 6.2 The Mendelian randomization estimate

Two variants are available as instruments for G expression. The Wald ratio for the lead
instrument:

```text
beta_outcome (log OR per allele) = 0.009
beta_exposure (SD per allele)    = 0.050
ratio = 0.009 / 0.050 = 0.180 log OR per SD
```

The standard error by the delta method is approximately:

```text
SE = SE_outcome / |beta_exposure| = 0.003 / 0.050 = 0.060
95% CI = 0.180 +/- 1.96 x 0.060 = [0.062, 0.298]
OR = exp(0.180) = 1.20, 95% CI [1.06, 1.35]
```

The instrument F-statistic is:

```text
F = (beta_exposure / SE_exposure)^2 = (0.050 / 0.010)^2 = 25
```

An F of 25 is above the conventional weak-instrument threshold of 10, so the estimate is
reasonably stable. The estimate says that higher G expression is associated with higher disease
risk, which supports an inhibitor rather than an agonist.

### 6.3 The perturbation evidence

A CRISPR screen in a disease-relevant cell line shows that G knockout reduces the inflammatory
phenotype by 40%. The gene's DepMap effect score is -0.05, which indicates no general essentiality
(the scale runs from about 0 for no effect to -1 or lower for strong dependency).

That combination is favorable: the gene matters in the disease model and is not required for
general cell viability. The check that matters is that the effect is not pan-essential; a score
of -0.05 is reassuring.

### 6.4 The success-probability arithmetic

The program's baseline assumption for phase 1 to approval in this indication is 8%. With human
genetic support, the Nelson estimate of a 2.0 ratio gives:

```text
8% x 2.0 = 16%
```

The updated Minikel estimate of 2.6 gives:

```text
8% x 2.6 = 20.8%
```

The program uses 16% as a conservative planning value and 21% as an upside. Neither number is a
prediction for this target; they are population averages for targets with genetic support, and
this target's specific evidence could be better or worse.

### 6.5 Diagnostic 1: the locus was misassigned

A re-analysis with a larger reference panel changes the credible set. The lead variant is no
longer in the credible set, and the eQTL colocalization posterior for G falls from 0.92 to 0.31.
A second gene, G2, has a posterior of 0.55.

The evidence package was built on a variant that is not causal under the new reference. The
target nomination is no longer supported. This happens when the reference panel's linkage
disequilibrium structure does not match the study population, and it is more common in
ancestries under-represented in reference panels.

### 6.6 Diagnostic 2: pan-essentiality missed

The initial DepMap query used a single cell line and showed an effect score of -0.6, which was
read as "the gene matters." A query across the full panel shows a median effect of -0.75 across
more than 1,000 lines, which is pan-essential.

The perturbation result in one disease-relevant line was therefore not specific. A pan-essential
gene will show a phenotype in any knockout experiment, and the 40% reduction in inflammation is
a general toxicity signal rather than a disease-specific effect.

### 6.7 Vary one assumption: no genetic support

Suppose the colocalization evidence fails and the target has no human genetic support. The
success probability reverts to the baseline:

```text
with genetic support: 16%
without:               8%
```

The expected value of the program halves. If the program's net present value at 16% is positive
and at 8% is negative, the decision changes from go to no-go. That is the practical consequence
of a single evidence line, and it is why the genetic evidence is worth the investment to obtain.

### 6.8 What the target summary says

> Target G, inflammatory disease. Locus with lead variant p = 3 x 10^-9, OR 1.12 per allele;
> credible set 12 variants; eQTL colocalization posterior 0.92 for G, below 0.10 for other
> genes in the locus. MR: OR 1.20 per SD (95% CI 1.06-1.35), F = 25, consistent across two
> instruments. Perturbation: 40% phenotype reduction in a disease-relevant line; DepMap median
> effect -0.05, not pan-essential. Expression: restricted to myeloid cells in the disease tissue.
> Tractability: secreted protein with a known binding pocket; antibody modality feasible. Safety:
> no severe Mendelian loss-of-function phenotype reported. Biomarker: soluble G measurable in
> serum. Success probability: 16% (baseline 8% x 2.0). Recommendation: proceed to antibody
> discovery, with a go/no-go at candidate selection.

### 6.9 The cost of being wrong

A target decision has an asymmetric cost profile. Suppose the program costs USD 120 million to
reach a phase 2 readout, and the probability of success at the decision point is 16%. The
expected cost per success is:

```text
120 / 0.16 = USD 750 million
```

At 8%, the expected cost per success is:

```text
120 / 0.08 = USD 1.5 billion
```

The difference between 8% and 16% is USD 750 million per success, which is why a genetic evidence
package worth a few hundred thousand dollars is cheap. The arithmetic also shows why portfolio
decisions are not about individual programs: a portfolio of 20 programs at 8% expects 1.6
successes, and at 16% expects 3.2, for the same total spend.

The cost of a false positive is not only the wasted spend. It is also the opportunity cost of the
program that was not run, and the patients who were exposed in a trial that could not succeed.
Those costs are asymmetric and mostly invisible in a spreadsheet, which is why the decision
defaults to continuing: the spend is visible and the opportunity cost is not.

A useful counterfactual question at every target review is: if this program did not exist, would
we start it today with this evidence? If the answer is no, the program is being continued because
of sunk cost, not because of the evidence.

### 6.10 What would change the recommendation

Four findings would reverse it: a colocalization posterior below 0.5 after fine-mapping; a
pan-essential DepMap profile; a severe human loss-of-function phenotype; or no measurable
engagement biomarker. Each is a single experiment or query, and each is cheaper than a year of
chemistry.

## 7. Code, query, or calculation

The script computes a Wald ratio with a delta-method interval, an instrument F-statistic, a
simple colocalization posterior from likelihoods, and a success-probability update.

### 7.1 Wald ratio and interval

```python
import math

def wald(beta_out, se_out, beta_exp, se_exp):
    ratio = beta_out / beta_exp
    se = abs(se_out / beta_exp)
    lo = ratio - 1.959963985 * se
    hi = ratio + 1.959963985 * se
    return ratio, se, lo, hi

ratio, se, lo, hi = wald(0.009, 0.003, 0.050, 0.010)
print(f"log OR per SD: {ratio:.3f} (SE {se:.3f})")
print(f"95% CI: [{lo:.3f}, {hi:.3f}]")
print(f"OR: {math.exp(ratio):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}]")
print(f"F: {(0.050/0.010)**2:.1f}")
```

Expected output:

```text
log OR per SD: 0.180 (SE 0.060)
95% CI: [0.062, 0.298]
OR: 1.197 [1.064, 1.347]
F: 25.0
```

The delta-method interval ignores uncertainty in the exposure estimate, which is acceptable when
the instrument is strong. With an F below 10, the interval is too narrow and the estimate is
biased toward the confounded association.

### 7.2 Colocalization posterior from likelihoods

```python
def posteriors(prior, l1, l2, l3, l4):
    unnorm = [prior[i] * l for i, l in enumerate([l1, l2, l3, l4])]
    total = sum(unnorm)
    return [u / total for u in unnorm]

# Priors for H1-H4; likelihoods from the data
priors = [0.1, 0.1, 0.1, 0.7]
post = posteriors(priors, 0.2, 0.3, 0.1, 5.0)
labels = ["H1: trait 1 only", "H2: trait 2 only", "H3: distinct variants", "H4: shared variant"]
for lab, p in zip(labels, post):
    print(f"{lab}: {p:.3f}")
```

Expected output:

```text
H1: trait 1 only: 0.006
H2: trait 2 only: 0.008
H3: distinct variants: 0.003
H4: shared variant: 0.983
```

The posterior for a shared variant is 0.983. The result is sensitive to the priors, which is why
colocalization results should be reported with the prior set and with a sensitivity analysis
across priors.

### 7.3 Success probability update

```python
def with_genetic_support(baseline, ratio):
    return baseline * ratio

base = 0.08
for label, ratio in [("no support", 1.0), ("Nelson 2015", 2.0), ("Minikel 2024", 2.6)]:
    print(label, "->", round(with_genetic_support(base, ratio) * 100, 1), "%")
```

Expected output:

```text
no support -> 8.0 %
Nelson 2015 -> 16.0 %
Minikel 2024 -> 20.8 %
```

The multiplication is a planning heuristic, not a prediction. It assumes the target's genetic
evidence is of the kind and quality represented in the study populations, which is often false
for under-studied genes and ancestries.

### 7.4 A simple evidence score with explicit weights

```python
weights = {"genetic": 0.4, "perturbation": 0.25, "expression": 0.15,
           "tractability": 0.1, "safety": 0.1}

def score(evidence):
    return sum(weights[k] * evidence[k] for k in weights)

candidate_a = {"genetic": 1.0, "perturbation": 0.8, "expression": 0.9,
               "tractability": 0.7, "safety": 0.8}
candidate_b = {"genetic": 0.2, "perturbation": 0.9, "expression": 0.8,
               "tractability": 0.9, "safety": 0.9}
for name, ev in [("A", candidate_a), ("B", candidate_b)]:
    print(name, round(score(ev), 3))
```

Expected output:

```text
A 0.885
B 0.605
```

The weights decide the ranking. A program that weights tractability more heavily would rank B
first. The score is a communication device for a committee, not a substitute for judgment; its
weights should be stated and challenged.

### 7.5 Assumptions and limits of this code

- The delta-method interval assumes a strong instrument and no pleiotropy.
- The colocalization posterior depends on the priors and the likelihood model; the toy
  likelihoods are illustrative.
- The success-probability multiplier is a population average, not a target-specific prediction.
- The weighted score encodes arbitrary weights and should never be the sole basis for a decision.
- No code here performs fine-mapping, MR sensitivity analysis, or DepMap queries.

## 8. Data, standards, and regulatory context

### 8.1 Data resources

| Resource | Content | Scale | Limit |
|---|---|---|---|
| GWAS Catalog | Curated association studies | Hundreds of thousands of associations | Ascertainment and ancestry |
| Open Targets Platform | Integrated target-disease evidence | Release 25.09: 78,726 targets, 39,530 diseases, 30.4 million evidence strings | Evidence quality varies by source |
| gnomAD | Population variant frequencies | 807,162 individuals | Ancestry composition |
| GTEx | Tissue expression and eQTLs | 17,382 samples, 54 tissues | Post-mortem, bulk |
| Human Cell Atlas | Single-cell expression | About 70.9 million cells in the 2026 snapshot | Tissue coverage |
| DepMap | Cancer cell line dependencies | More than 1,000 lines | Cancer only, in vitro |
| ClinGen | Gene-disease validity | Curated assertions | Well-studied genes |
| PDB | Experimental structures | About 200,000 entries | Coverage gaps |
| AlphaFold DB | Predicted structures | Proteome-scale | Prediction, not experiment |

Open Targets integrates genetics, expression, pathways, literature, and chemical data, and its
release notes document the evidence sources and their versions. The 25.03 release merged the
Platform and Open Targets Genetics, adding variant and credible-set entities and a locus-to-gene
model. Using the platform means accepting its evidence weighting, which should be understood
rather than treated as ground truth.

### 8.2 Standards

There is no regulatory approval for a target. The applicable standards are about the evidence and
the studies:

- **Good Laboratory Practice (GLP)** and **Good Clinical Practice (GCP)** for the studies that
  generate regulatory evidence.
- **ICH E16** for the context, structure, and format of genomic biomarker qualification
  submissions, which applies when a target-linked biomarker is used in development.
- **FDA Biomarker Qualification Program** for a biomarker used to support a target-based claim.
- **ARRIVE** guidelines for reporting animal research.
- **Declaration of Helsinki** for human research ethics.
- **FAIR data principles** for the data resources.

### 8.3 Study standards that do apply

Even without a target standard, the studies that generate evidence are regulated:

- **GLP** for toxicology and safety studies intended for regulatory submission.
- **GCP** for clinical trials, including first-in-human studies.
- **ARRIVE** for reporting animal research, including randomisation, blinding, and sample size.
- **Declaration of Helsinki** for the ethics of human research.
- **Gene therapy and cell therapy guidance** where the modality is advanced therapy.

A target package that cites animal data without randomisation or blinding is citing evidence that
would not survive a regulatory review, even though the target itself is not regulated.

### 8.4 Data access and use agreements

Target evidence often comes from controlled-access resources: UK Biobank, dbGaP, and consortium
data with data use agreements. The agreements constrain what can be published, what can be shared
with commercial partners, and whether results can be used for target validation. A program that
discovers a target in controlled data and moves to chemistry without checking the agreement's
commercial-use terms is exposed to a legal risk that no amount of science resolves.

### 8.5 What the standards do not settle

- Which evidence types are sufficient for a target.
- How to weight genetic versus functional evidence.
- What constitutes a validated target.
- How much preclinical evidence justifies a first-in-human trial.

These are scientific and portfolio judgments. The absence of a standard is why target review is
often political, and why a written evidence summary with explicit criteria is valuable.

## 9. Industry practice and case evidence

### 9.1 The genetic-support evidence

Nelson et al. (2015) combined GWAS and OMIM data with drug pipeline data and found that approved
drug targets were enriched for genetic support. The estimated ratio for phase 1 to approval was
2.0 (95% CI 1.6-2.4), and the proportion of mechanisms with genetic support rose from 2.0% at
preclinical/phase 1 to 8.2% for approved drugs. The support varied by therapeutic area, with
musculoskeletal, metabolic, and haematology targets showing the most support and oncology the
least.

Minikel et al. (2024) updated the analysis with a decade more data and 29,476 target-indication
pairs, estimating a 2.6-fold relative success rate. They also noted that only about 1.1% of
genetically supported gene-indication pairs have been clinically explored, which is an argument
that the opportunity is under-used.

Both estimates are observational and subject to confounding: genetically supported targets may
differ in other ways that affect success, such as being better studied or having clearer
biology. The direction of the association is consistent across analyses; the magnitude is not
precisely known.

### 9.2 Structure prediction

AlphaFold2 achieved high accuracy in the 2020 CASP14 assessment and produced proteome-scale
predictions. The practical contribution to drug discovery is real but narrower than the
headlines: it accelerates structure determination, supports binding-site identification, and
enables docking where no experimental structure exists. It does not establish that a target is
causal, that a pocket is functional, or that a molecule will be selective.

The correct use is as a hypothesis-generation tool inside a structure-based design cycle, with
experimental validation by crystallography, cryo-electron microscopy, or mutagenesis.

### 9.3 Perturbation at scale

DepMap and similar resources made it possible to ask, for every gene, whether cancer cell lines
depend on it. The result is a powerful filter and a misleading validation. Most strong
dependencies are pan-essential or lineage-specific in ways that do not match the disease, and the
correlation between cell-line dependency and in vivo efficacy is modest.

The practical rule: use dependency data to exclude, not to confirm. A gene with no dependency in
any relevant model is a weaker target; a gene with a dependency in one model is a candidate, not
a validated one.

### 9.4 The portfolio arithmetic

A portfolio of 20 programs at an 8% success rate expects 1.6 approvals; at 16%, it expects 3.2.
The difference in expected approvals, multiplied by the value of an approval, is the value of
better target selection. That arithmetic explains why large companies invest in genetics at
scale, and why the same companies still fail on targets that looked well supported.

The limit of the arithmetic is that the probabilities are averages. A portfolio with 20 programs
in the same mechanism is not 20 independent draws; it is one draw with 20 attempts. Correlation
across programs is the risk that portfolio arithmetic hides.

### 9.5 What public-private consortia changed

Open Targets, the Structural Genomics Consortium, and similar pre-competitive collaborations
changed the evidence landscape by pooling target evidence across companies. A program can now
see, in one place, the genetic associations, expression, pathways, and chemical matter for a
target, with the evidence sources timestamped.

The benefit is speed and breadth. The cost is that the evidence is someone else's curation, with
its own quality thresholds and its own errors. A target summary that cites Open Targets should
still trace the key claims to the primary studies, at least for the evidence that drives the
decision.

### 9.6 What the AI claims do and do not show

Machine-learning models for target identification and molecule generation report benchmark gains
on tasks such as binding affinity prediction, retrosynthesis, and property prediction. Those
gains are real within their benchmarks and do not establish clinical success. The bottlenecks in
drug discovery are not only molecular design; they are target validation, safety, and clinical
translation, and a model that improves one step does not change the others.

A 2025 benchmark of single-cell foundation models found that zero-shot representations
underperformed simpler baselines on clustering and integration, which is a useful reminder that a
general model requires task-specific validation. The same standard applies to target-prediction
models.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Nearest-gene nomination | No colocalization | Locus complexity | Wrong target | Fine-mapping |
| LD confounding | Signal shifts with reference panel | Linkage, not causality | Wrong gene | Colocalization |
| Pleiotropy | MR estimates heterogeneous | Exclusion restriction violated | Biased effect | MR-Egger, multivariable MR |
| Weak instrument | F below 10 | Little exposure variation | Bias to confounded estimate | Stronger instruments |
| Tissue mismatch | eQTL from irrelevant tissue | Wrong isoform | Wrong target | Disease-relevant data |
| Pan-essentiality | DepMap median strongly negative | General requirement | Toxicity | Selectivity, partial inhibition |
| Single-cell-line evidence | One model only | Model artifact | False validation | Multiple models |
| Animal mismatch | Model transcriptome differs | Species difference | No translation | Human tissue validation |
| Tractability confusion | Target chosen because it is easy to drug | Druggability mistaken for causality | Potent, ineffective drug | Independent assessments |
| No engagement biomarker | No PD marker | Uninterpretable trial | No learning from failure | Biomarker development |
| IP blockage | Freedom-to-operate not checked | Patents | Commercial failure | Early FTO review |
| Escalation of commitment | Criteria change after data | Sunk cost | Continued investment | Pre-specified go/no-go |
| Benchmark overreach | AI claim from a single task | Task-specific gain | False confidence | External validation |
| Literature bias | Target "well studied" | Popularity | Crowded, not validated | Independent evidence |
| Assay cascade mismatch | Biochemical hit, no cellular effect | Proximal assay | Potent but inactive molecule | Cell-based confirmation |
| Irreversible modality | Permanent modification | Covalent chemistry | Irreversible toxicity | Reversible design |
| Crowded mechanism | Several phase 2 programs | Competition | Weak differentiation | Define differentiation early |

Four red flags:

**If the target is the nearest gene and nothing else, stop.** The locus is not a gene.

**If the dependency is pan-essential, stop.** You have found a toxicity, not a mechanism.

**If there is no engagement biomarker, stop.** A negative trial will teach nothing.

**If the go/no-go criteria were written after the data, stop.** They are a rationalization.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The locus contains the causal gene | Genetic evidence | LD, multiple genes | Colocalization, fine-mapping |
| The instrument affects only the exposure | MR validity | Pleiotropy | Sensitivity analyses |
| The eQTL tissue is relevant | Expression evidence | Tissue mismatch | Disease-relevant data |
| Cell-line dependency predicts disease | Perturbation evidence | Pan-essentiality, context | Multiple models, in vivo |
| Animal models recapitulate human disease | Preclinical PoC | Species differences | Human tissue validation |
| The modality can reach the target | Tractability | Localization, delivery | Chemical and PK data |
| Selectivity is achievable | Safety | Conserved binding sites | Selectivity screens |
| Genetic support generalizes | Portfolio planning | Ancestry, study quality | Target-specific assessment |
| The biomarker measures engagement | Trial interpretation | Indirect marker | Target engagement study |
| The program can be stopped | Governance | Sunk cost | Pre-specified criteria |

The assumption that genetic support generalizes is the one most often overstated. The 2x estimate
comes from targets that reached clinical development, which is a selected set; a new target with
genetic support is not a random draw from that population.

These exercises assume the evidence hierarchy in section 3.2. Where an exercise asks for a
judgment, the acceptance criteria require the reasoning to be explicit, because the judgment is
the skill and the answer is secondary.

## 12. Exercises

**Exercise 1 — Locus to gene (application, 30 minutes).**
A GWAS locus contains 25 genes and the nearest is gene X. Colocalization posteriors are 0.85 for
gene Y, 0.08 for gene X, and below 0.05 for the rest. Write the target nomination and the
remaining uncertainty.
*Deliverable:* a short memo.
*Criteria:* nomination follows the evidence, not proximity; uncertainty quantified.
*Time:* 30 minutes.

**Exercise 2 — MR interpretation (calculation, 35 minutes).**
Beta outcome 0.012 (SE 0.004), beta exposure 0.040 (SE 0.012). Compute the Wald ratio, the
delta-method interval, the OR, and the F-statistic. State whether the instrument is strong.
*Deliverable:* arithmetic and a judgment.
*Criteria:* all four values correct; judgment follows the F threshold.
*Time:* 35 minutes.

**Exercise 3 — Pan-essentiality (application, 25 minutes).**
A gene has a DepMap median effect of -0.72 across 1,100 lines and -0.80 in the disease-relevant
line. Interpret the result and state the program implication.
*Deliverable:* a 200-word assessment.
*Criteria:* pan-essentiality identified; implication for therapeutic window stated.
*Time:* 25 minutes.

**Exercise 4 — Evidence weighting (critique, 40 minutes).**
A target review scores genetic evidence 10/10, tractability 2/10, and safety 3/10, and
recommends proceeding because the total is 15/30. Critique the scoring and propose a decision
rule.
*Deliverable:* a 400-word critique.
*Criteria:* identifies non-compensatory dimensions; proposes a rule with thresholds.
*Time:* 40 minutes.

**Exercise 5 — Success probability (calculation, 25 minutes).**
Baseline success 6%. Compute the success probability with genetic support at 2.0x and 2.6x, and
the expected number of programs needed for one approval at each.
*Deliverable:* arithmetic and a two-sentence implication.
*Criteria:* both values correct; expected programs computed as the reciprocal.
*Time:* 25 minutes.

**Exercise 6 — Biomarker plan (design, 45 minutes).**
Design a target engagement and pharmacodynamic biomarker plan for an antibody against a secreted
protein. Specify the sample, the assay, the timing, and the decision each measurement supports.
*Deliverable:* a one-page plan.
*Criteria:* engagement and effect distinguished; timing tied to PK; decision rule per measurement.
*Time:* 45 minutes.

**Exercise 7 — AI claim critique (critique, 35 minutes).**
A vendor claims its model identifies targets with a 3x higher success rate based on a benchmark
of binding affinity prediction. Write a critique of the claim.
*Deliverable:* 300 words.
*Criteria:* distinguishes benchmark task from clinical outcome; names confounders; proposes a
validation.
*Time:* 35 minutes.

**Exercise 8 — Go/no-go (design, 40 minutes).**
Write pre-specified go/no-go criteria for advancing a target from validation to lead discovery,
covering causality, tractability, safety, and biomarker.
*Deliverable:* a criteria table.
*Criteria:* each criterion numeric or binary; a stop rule; a review date.
*Time:* 40 minutes.

## 13. Self-check answers

**Answer 1 (locus).** Nominate gene Y: its colocalization posterior of 0.85 is far above the
others, and proximity is not evidence. The remaining uncertainty is the 15% probability that the
shared signal is coincidental or driven by another gene in the locus, plus the possibility that
the molecular trait measured is not the relevant one. The next steps are fine-mapping to a
smaller credible set, a disease-relevant eQTL or pQTL, and functional perturbation of Y.

**Answer 2 (MR).**

```text
ratio = 0.012 / 0.040 = 0.300 log OR per SD
SE = 0.004 / 0.040 = 0.100
95% CI = [0.104, 0.496]
OR = 1.35 [1.11, 1.64]
F = (0.040 / 0.012)^2 = 11.1
```

The F of 11.1 is just above the conventional threshold of 10, so the instrument is adequate but
not strong. The estimate should be reported with a sensitivity analysis, because a single
instrument at this strength is vulnerable to pleiotropy.

**Answer 3 (pan-essentiality).** A median effect of -0.72 across 1,100 lines indicates that most
cancer cell lines require the gene, which is the signature of a pan-essential gene. The effect in
the disease-relevant line (-0.80) is not meaningfully different. A drug against this target would
likely have a narrow therapeutic window, and the program should either deprioritize or pursue a
tissue-sparing modality. The result does not prove toxicity in vivo, but it is a strong warning.

**Answer 4 (weighting).** The scoring is compensatory: high genetic evidence offsets low
tractability and safety, which is inappropriate because the dimensions are not substitutable. A
target that cannot be drugged or safely modulated fails regardless of causality. The decision
rule should be non-compensatory: require a minimum on each dimension (for example, genetic
evidence at least moderate, tractability at least one viable modality, safety with a plausible
window), then rank within the qualifying set. A total score can communicate, but it cannot
decide.

**Answer 4 note.** A strong critique also notes that the score's dimensions are correlated:
genetic evidence often comes with better-studied biology, which correlates with tractability. The
apparent independence of the dimensions is itself an assumption.

**Answer 5 (success).** At 6% baseline: 6% x 2.0 = 12%, and 6% x 2.6 = 15.6%. The expected
number of programs per approval is the reciprocal: 1/0.12 = 8.3 programs at 2.0x, and
1/0.156 = 6.4 at 2.6x. Without genetic support, 1/0.06 = 16.7 programs. The arithmetic explains
why portfolio size and target quality interact so strongly.

**Answer 6 (biomarker).** Target engagement: measure the soluble target in serum before and at
several times after dosing, with a validated immunoassay; a sustained reduction confirms binding
and clearance. Pharmacodynamic: measure a downstream marker in blood or tissue at the same time
points; a change confirms mechanism. Timing: sample at peak and trough after the first dose and
at steady state. Decision: no engagement at tolerated doses means the dose or molecule is wrong;
engagement without a pharmacodynamic change means the mechanism hypothesis is wrong.

**Answer 7 (critique).** Binding affinity prediction is a molecular property task; clinical
success depends on target validation, safety, and translation, none of which the benchmark
measures. A 3x claim requires a comparison of clinical success rates between targets selected by
the model and a matched comparison, not a benchmark gain. Confounders include selection of
well-studied targets, availability of structures, and the choice of benchmark dataset. The claim
should be reframed as "improves prioritization within a validated target set," and tested
prospectively.

**Answer 7 note.** A strong answer also distinguishes a model that ranks targets within a
validated set from one that discovers targets. The first is a prioritization tool with a
measurable gain; the second requires prospective evidence that the targets it discovers succeed
more often, which no benchmark of binding affinity can provide.

**Answer 8 (go/no-go).** Causality: colocalization posterior at least 0.8 for the nominated gene,
or MR with F above 10 and consistent sensitivity analyses. Tractability: at least one modality
with a credible path to a clinical candidate within 24 months. Safety: no severe human
loss-of-function phenotype; expression not ubiquitous at high levels. Biomarker: a measurable
engagement marker validated in the relevant matrix. Stop rule: failure on any dimension stops the
program, not a low total score. Review at month 12 with the criteria unchanged.

The questions below test whether a candidate can separate the three axes — causality,
tractability, and safety — and whether they know what each piece of evidence can and cannot
support.

## 14. Interview questions

**1. How do you decide a GWAS locus identifies a gene?**
Strong answer: fine-mapping, colocalization with eQTL or pQTL, functional evidence, and
direction of effect; proximity is not evidence.
Weak answer: "the nearest gene."

**2. What are the assumptions of Mendelian randomization?**
Strong answer: relevance, independence, exclusion restriction; diagnostics include F-statistic,
heterogeneity, and MR-Egger.
Weak answer: "it uses genes as instruments" without assumptions.

**3. What does DepMap tell you and what does it not?**
Strong answer: cell-line dependency; it does not establish in vivo efficacy, and pan-essential
genes are a toxicity signal.
Weak answer: "whether a gene is essential."

**4. Why is genetic support valuable?**
Strong answer: it is human, pre-disease, and less confounded; the estimated success ratio is about
2 to 2.6.
Weak answer: "it proves the target works."

**5. What makes a target tractable?**
Strong answer: a binding site or a modality path, localization, and a chemistry or biologics
route; tractability is independent of causality.
Weak answer: "it has a known inhibitor."

**6. What is the difference between target validation and tractability?**
Strong answer: validation is evidence that modulation changes disease; tractability is evidence
that modulation is feasible; a target can have one without the other.
Weak answer: conflates them.

**7. How would you use AlphaFold in a target program?**
Strong answer: structure hypothesis, binding-site identification, docking where no experimental
structure exists, with experimental validation.
Weak answer: "it solves the structure, so we can design drugs."

**8. Why do many genetically supported targets still fail?**
Strong answer: genetic support is a population average, effect sizes may be small, safety may be
limiting, and the modality may not achieve the required exposure.
Weak answer: "because drug discovery is hard."

**9. What is an engagement biomarker and why does it matter?**
Strong answer: a measurement that the drug hit the target, which allows a failed trial to
distinguish molecule failure from target failure.
Weak answer: "a biomarker for efficacy."

**10. How would you present a target to a portfolio committee?**
Strong answer: evidence by dimension, with explicit uncertainty, the success-probability
assumption, and pre-specified criteria.
Weak answer: a single score.

**11. What is the main limitation of expression correlation for target validation?**
Strong answer: reverse causation and composition; it supports localization, not causality.
Weak answer: "correlation is not causation" without specifics.

**12. What is the difference between a target and a mechanism?**
Strong answer: a target is a molecule; a mechanism is the causal process it participates in; a
program can hit a target and miss the mechanism, which is why pathway and biomarker evidence
matter.
Weak answer: uses the terms interchangeably.

**13. How do you evaluate a phenotypic screen hit?**
Strong answer: deconvolution with genetic evidence (resistance mutations, target-null), not a
plausible binding partner from chemical proteomics alone.
Weak answer: "find what it binds."

**14. What would stop a target program?**
Strong answer: no colocalized gene, pan-essentiality, a severe human phenotype, no tractable
modality, or no engagement biomarker.
Weak answer: "bad data."

## 15. Key terms

- **ADMET** — absorption, distribution, metabolism, excretion, toxicity.
- **AlphaFold** — protein structure prediction system.
- **Candidate** — molecule entering development.
- **Colocalization** — shared causal variant between traits.
- **CRISPR screen** — pooled perturbation screen.
- **DepMap** — cancer cell line dependency resource.
- **Druggability** — presence of a binding site for a drug-like molecule.
- **eQTL** — expression quantitative trait locus.
- **Essentiality** — requirement for cell viability.
- **Fine-mapping** — narrowing a locus to causal variants.
- **F-statistic** — instrument strength in MR.
- **GWAS** — genome-wide association study.
- **Hit** — molecule with confirmed activity.
- **Instrument** — variant used as an MR proxy.
- **L2G** — locus-to-gene score.
- **Lead** — optimized starting compound.
- **Ligandability** — ability to bind a small molecule.
- **Locus** — genomic region associated with a trait.
- **Mendelian randomization** — instrumental-variable analysis using genetic variants.
- **Molecular glue** — compound inducing a protein interaction.
- **Pan-essential** — essential in most cell lines.
- **Pleiotropy** — instrument effect through another pathway.
- **pQTL** — protein quantitative trait locus.
- **Proof of concept** — evidence of biological effect in humans.
- **PROTAC** — targeted protein degrader.
- **Selectivity** — preference for the target over related proteins.
- **Target** — molecule whose modulation is intended to change disease.
- **Target product profile** — the desired product characteristics.
- **Tractability** — feasibility of modulating the target.
- **Wald ratio** — single-variant MR estimate.
- **Target deconvolution** — identifying the target of a phenotypic screen hit.
- **Therapeutic window** — exposure range between efficacy and toxicity.
- **Freedom to operate** — confirmation that a program does not infringe blocking patents.

## 16. Sources

Evidence:

- Nelson, M. R. et al., "The support of human genetic evidence for approved drug indications," *Nature Genetics* 47:856-860, 2015. https://doi.org/10.1038/ng.3314 (accessed 25 October 2026).
- Minikel, E. V., Painter, J. L., Dong, C. C., Nelson, M. R., "Refining the impact of genetic evidence on clinical success," *Nature* 629:624-629, 2024. https://doi.org/10.1038/s41586-024-07316-0 (accessed 25 October 2026).
- Jumper, J. et al., "Highly accurate protein structure prediction with AlphaFold," *Nature* 596:583-589, 2021. https://doi.org/10.1038/s41586-021-03819-2 (accessed 25 October 2026).
- Tsherniak, A. et al., "Defining a cancer dependency map," *Cell* 170:564-576, 2017. https://doi.org/10.1016/j.cell.2017.06.010 (accessed 25 October 2026).
- "Benchmarking foundation cell models," *Genome Biology*, 2025. https://doi.org/10.1186/s13059-025-03574-x (accessed 25 October 2026).

Data resources:

- Open Targets Platform. https://www.opentargets.org/ (accessed 25 October 2026).
- Open Targets release notes. https://github.com/opentargets/platform-docs/blob/main/release-notes.md (accessed 25 October 2026).
- GWAS Catalog. https://www.ebi.ac.uk/gwas/ (accessed 25 October 2026).
- gnomAD. https://gnomad.broadinstitute.org/ (accessed 25 October 2026).
- GTEx Portal. https://gtexportal.org/ (accessed 25 October 2026).
- DepMap. https://depmap.org/ (accessed 25 October 2026).
- AlphaFold Protein Structure Database. https://alphafold.ebi.ac.uk/ (accessed 25 October 2026).

Standards:

- ICH E16. https://database.ich.org/sites/default/files/E16_Guideline.pdf (accessed 25 October 2026).
- FDA Biomarker Qualification Program. https://www.fda.gov/drugs/biomarker-qualification-program/about-biomarkers-and-qualification (accessed 25 October 2026).
- ARRIVE guidelines. https://arriveguidelines.org/ (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Next topic: [`10_clinical-genomics.md`](10_clinical-genomics.md)
- Related topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Related topic: [`07_multi-omics-integration.md`](07_multi-omics-integration.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
