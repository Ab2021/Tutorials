# AI for Protein Engineering and Structural Biology

> Module 12 — Omics and Precision Medicine
> Topic 14 of 16
> Estimated study time: 4 hours
> Prerequisites: topics 01, 02, and 09
> Version 2026.3

## 1. Why this topic exists

A protein's function is determined by its three-dimensional structure, and for fifty years the
structure was harder to obtain than the sequence. X-ray crystallography, nuclear magnetic
resonance, and cryogenic electron microscopy each cost months to years per structure, and many
proteins never yielded one at all. Drug discovery worked around the gap: screening millions of
compounds against an unknown pocket, or engineering antibodies without knowing the epitope.

Between 2020 and 2026 the gap narrowed sharply. AlphaFold2 predicted single-chain structures at
accuracy competitive with experiment for many targets. AlphaFold3 extended prediction to complexes
with nucleic acids, ligands, and ions, using a diffusion module over atom coordinates. Open
reproductions such as Boltz-1 and Boltz-2 made the architecture available under permissive
licenses, and Boltz-2 added binding affinity prediction. The AlphaFold Protein Structure Database
passed 200 million predicted structures. For monomeric, well-represented protein families, fold
prediction is close to a solved problem.

The frontier moved from prediction to design. If a structure can be predicted, it can be
inverted: generate a backbone that fits a target pocket, then design a sequence that folds into
that backbone. RFdiffusion generates backbones by denoising in three-dimensional coordinate space,
and ProteinMPNN solves the inverse folding problem by choosing a sequence for a given backbone.
The combination has produced de novo binders, symmetric assemblies, and enzyme scaffolds that did
not exist in nature.

The gap that remains is the one that matters commercially: in silico success is not experimental
success. A design pipeline can produce thousands of models with high confidence scores and no
measured binding. Published campaigns report hit rates that vary by orders of magnitude depending
on the target, the design method, and the assay. A program that treats a predicted structure as a
wet-lab result will spend its budget on proteins that do not fold, do not express, or do not bind.

Three specific risks drive the cost.

**Confidence scores are local.** The predicted local distance difference test (pLDDT) measures
confidence in local geometry. It does not measure whether two domains have the correct relative
orientation, and it does not measure whether a designed interface will bind. The predicted aligned
error (PAE) matrix answers the second question, and it is routinely ignored.

**Benchmarks measure the wrong thing.** The Critical Assessment of Structure Prediction (CASP)
measures blind prediction of structures that experimentalists chose to solve; those targets are
biased toward crystallizable, well-behaved proteins. ProteinGym measures fitness prediction across
deep mutational scanning assays. Neither measures whether a designed protein works as a
therapeutic.

**The design funnel is multiplicative.** If each design has a 2% chance of becoming a binder, 96
designs give an 86% chance of at least one binder, but 96 designs require synthesizing and
expressing 96 proteins. If the per-design probability falls to 0.2%, the same 90% confidence
requires about 1,150 designs. The arithmetic of the funnel, not the elegance of the model,
determines the budget.

### 1.1 The failure pattern

1. A target structure is taken from a prediction without checking its confidence in the relevant
   region.
2. A generative model produces backbones, and the top-scoring models are selected by the same
   model that generated them.
3. Sequences are designed and scored by predicted structure, with no interface-specific metric.
4. Designs are ranked by pLDDT, which is high for a well-folded monomer that does not bind.
5. Dozens of designs are ordered without an expression or solubility filter.
6. Binding is tested at a single concentration, and non-binders are discarded without a
   dose-response check.
7. The campaign reports a hit rate without reporting how many designs were tested, which makes the
   number uninterpretable.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Protein engineer | Months on designs that do not express | Depends on filters upstream |
| Discovery lead | A campaign with no hits and no diagnosis | Cannot tell whether design or assay failed |
| Computational biologist | A model that looks accurate and is not predictive | Benchmarks do not measure the campaign |
| Wet-lab scientist | Time on constructs with no chance | Received designs without QC |
| Program | Budget spent on synthesis and assay | Depends on the funnel arithmetic |
| Partner | A milestone missed | Depends on the same funnel |

### 1.3 The industry question

> Does this design have a credible chance of folding, expressing, and performing the intended
> function in a real assay, and what evidence supports each of those three claims?

The three claims are separate. Folding is supported by structure prediction confidence and
self-consistency. Expression is supported by sequence-level properties and by prior experience
with similar scaffolds. Function is supported only by experiment, or by a validated surrogate that
has been calibrated against experiment for this target class.

## 2. Learning outcomes

By the end of this topic you can:

1. **Interpret** pLDDT and PAE separately, and explain what each does and does not support.
2. **Describe** the AlphaFold3 architecture at the level of trunk, pair representation, and
   diffusion module, and explain why it generalizes to ligands and nucleic acids.
3. **Run** a design funnel from backbone generation to sequence design to in silico filtering, and
   compute the expected number of designs needed for a target hit probability.
4. **Compute** contact-map agreement between a designed model and a predicted model, and explain
   why high agreement within a monomer says nothing about binding.
5. **Apply** biophysical filters for expression and solubility before synthesis.
6. **Evaluate** benchmark evidence from CASP and ProteinGym, and state what each benchmark does
   not measure.
7. **Diagnose** the difference between a folding failure and a binding failure from in silico and
   experimental data.
8. **Design** a campaign with a pre-specified funnel, assay, and stopping rule.

## 3. The core mental model

Protein engineering inverts the prediction problem. Prediction maps sequence to structure. Design
maps a desired structure or function to a sequence.

```text
  PREDICTION                          DESIGN
  ----------                          ------
  sequence                            target structure or function
      |                                    |
      v                                    v
  fold prediction                     backbone generation
  (AlphaFold, ESMFold, Boltz)         (RFdiffusion, hallucination)
      |                                    |
      v                                    v
  3D coordinates                      backbone coordinates
      |                                    |
      +----------> confidence <-----------+
                   pLDDT, PAE
                        |
                        v
                  sequence design
                  (ProteinMPNN, inverse folding)
                        |
                        v
                  candidate sequences
                        |
                        v
                  in silico filters
                  folding, interface, solubility, charge
                        |
                        v
                  synthesis and assay
                        |
                        v
                  measured binding or function
                        |
                        v
                  learn and iterate
```

### 3.1 Three representations of the same protein

| Representation | What it captures | What it loses | Typical artifact |
|---|---|---|---|
| Sequence | Covalent composition | Geometry, dynamics | A sequence that cannot fold |
| Structure | Static coordinates | Dynamics, ensemble, environment | A single conformation treated as the truth |
| Function | Activity in an assay | Mechanism | A hit that does not generalize |

A model that predicts structure well has learned a map from sequence to a static conformation. It
has not learned the energy landscape, the folding pathway, or the binding affinity. Those are
different problems, and conflating them is the central error in the field.

### 3.2 What the confidence metrics mean

**pLDDT** is a per-residue score from 0 to 100 that estimates agreement with the true local
distance differences. The conventional bands:

| pLDDT | Interpretation | What it supports |
|---|---|---|
| Above 90 | Very high local confidence | Side-chain and backbone geometry in this region |
| 70 to 90 | Confident backbone | Fold of the region |
| 50 to 70 | Low confidence | Flexible or ambiguous region |
| Below 50 | Very low confidence | Often intrinsically disordered; do not assume a fixed structure |

**PAE** is a pairwise matrix in angstroms. Entry (i, j) estimates the expected positional error of
residue j when the prediction is aligned on residue i. It answers a different question: are the
relative orientations of two regions correct?

| PAE pattern | Interpretation |
|---|---|
| Low within a domain, low between domains | Rigid, confident assembly |
| Low within domains, high between domains | Each domain folds, relative orientation unknown |
| High everywhere | The model has no confident global structure |
| Low for one chain, high for the other in a complex | One partner is confidently placed, the other is not |

The practical rule: use pLDDT to judge whether a region folds, and PAE to judge whether an
interface or assembly is credible. A designed binder with high pLDDT and high interface PAE is a
well-folded protein that probably does not bind.

### 3.3 The design funnel is a product of conditional probabilities

```text
  P(success) = P(fold) x P(express | fold) x P(bind | express, fold) x P(assay detects | bind)
```

Each factor is less than one, and they multiply. A campaign with 0.7, 0.5, 0.05, and 0.9 gives
0.7 x 0.5 x 0.05 x 0.9 = 0.0158, or about 1.6% per design. That is the number that determines how
many designs to order.

The factors are not independent. A design that folds unusually well may be rigid and fail to
adapt to the target. A design that expresses at high yield may aggregate. The product is an
approximation, and the approximation is still more useful than a single "confidence score".

### 3.4 Self-consistency is necessary but not sufficient

A common filter is self-consistency: predict the structure of the designed sequence and compare
it with the designed backbone. A high agreement means the sequence encodes the intended fold. It
does not mean the fold binds the target.

Three checks are often conflated:

| Check | Question | Method |
|---|---|---|
| Sequence-structure consistency | Does the sequence fold into the designed backbone? | Predict the monomer and compare |
| Interface consistency | Does the complex place the binder on the intended epitope? | Predict the complex and inspect the interface |
| Functional plausibility | Is the interface compatible with binding? | Shape complementarity, hydrogen bonds, buried area |

Only the third begins to address binding, and none of the three replaces an assay.

### 3.5 The wet-lab gap

The distance between a model and a result has four components:

1. **Expression.** A designed sequence may not express in the chosen host, may be toxic, or may
   form inclusion bodies.
2. **Stability.** A protein that folds in a prediction may be marginally stable in solution.
3. **Assay sensitivity.** A weak binder may be below the detection limit of the assay.
4. **Biology.** A binder in vitro may not work in cells or in vivo.

Each component is a place where a campaign fails without the model being wrong. A program that
cannot distinguish them will misattribute the failure to the design method.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Protein language model (pLM) | Transformer trained on protein sequences | Assumed to model structure explicitly | Model class |
| ESM-2 | Protein language model family | Confused with ESMFold, the structure model built on it | Model |
| ESMFold | Structure prediction model using a protein language model | Assumed to match AlphaFold2 accuracy on all targets | Model |
| AlphaFold2 | Structure prediction model using multiple sequence alignments and attention | Assumed to predict complexes | Model |
| AlphaFold3 | Structure prediction model using a pairformer trunk and diffusion module | Assumed to be open source at release; it was server-only | Model |
| Boltz-1 | Open reproduction of AlphaFold3 | Assumed identical in accuracy | Model |
| Boltz-2 | Open model for structure and binding affinity | Assumed to replace experimental affinity measurement | Model |
| pLDDT | Predicted local distance difference test | Treated as a global quality score | 0 to 100 |
| PAE | Predicted aligned error | Ignored when judging complexes | Angstroms |
| Multiple sequence alignment (MSA) | Aligned homologous sequences | Assumed available for all targets | Alignment |
| Template | Experimental structure used as a model input | Assumed to be correct | PDB entry |
| RFdiffusion | Diffusion model that generates protein backbones | Assumed to design sequences | Model |
| ProteinMPNN | Message-passing network for inverse folding | Assumed to validate binding | Model |
| Inverse folding | Designing a sequence for a given backbone | Confused with folding prediction | Problem class |
| Backbone | Carbon-alpha trace without side chains | Treated as a final design | Structure |
| Self-consistency | Agreement between a designed backbone and a predicted structure of the designed sequence | Treated as evidence of binding | Root mean square deviation, contact F1 |
| Root mean square deviation (RMSD) | Average atomic distance after superposition | Reported without saying which atoms | Angstroms |
| Contact map | Set of residue pairs within a distance cutoff | Sensitive to the cutoff | Pair set |
| Interface | Residues in contact across a chain boundary | Confused with a binding site | Residue set |
| Shape complementarity | Geometric fit of two surfaces | Treated as a binding predictor | Score |
| Binder | Molecule that binds a target with measurable affinity | Confused with a functional agonist or antagonist | Molecule |
| Hit rate | Fraction of tested designs that meet the assay criterion | Reported without the number tested | Proportion |
| Enrichment | Over-representation of true hits in a ranked subset | Confused with precision | Ratio |
| Deep mutational scanning (DMS) | Systematic measurement of variant effects | Assumed to generalize across proteins | Assay |
| ProteinGym | Benchmark of fitness prediction across DMS assays | Assumed to measure design success | Benchmark |
| CASP | Blind assessment of structure prediction | Assumed to measure druggable targets | Benchmark |
| CAMEO | Continuous assessment of structure prediction servers | Assumed to cover designed proteins | Benchmark |
| AlphaFold Protein Structure Database | Repository of predicted structures | Assumed to contain experimental quality | Database |
| Protein Data Bank (PDB) | Repository of experimental structures | Assumed to contain all conformations | Database |
| mmCIF | File format for macromolecular structures | Confused with PDB format | File format |
| Solubility | Propensity of a sequence to remain in solution | Predicted poorly from sequence alone | Assay-dependent |
| Aggregation propensity | Tendency to form aggregates | Confused with insolubility | Score |
| Epitope | Target surface recognized by a binder | Confused with the entire target | Surface patch |

## 5. The mechanics, step by step

The procedure runs from a design objective to a measured result. It assumes access to structure
prediction and design tools; the computational steps can be run with open models, and the
experimental steps require a laboratory.

| Step | Question answered | Artifact | Failure if skipped |
|---:|---|---|---|
| 1 | What function, and against what target? | Design specification | Aesthetic designs with no use |
| 2 | Is the target structure credible? | Validated target model | Designs against a wrong surface |
| 3 | Which design modality? | Method choice | Wrong scaffold class |
| 4 | Generate backbones | Backbone models | No diversity to screen |
| 5 | Design sequences | Candidate sequences | Backbones with no encodable sequence |
| 6 | Predict the designed structures | Monomer and complex models | Designs that cannot fold |
| 7 | Check interface confidence | PAE and interface metrics | Well-folded non-binders |
| 8 | Filter for expression and solubility | Ranked shortlist | Proteins that never express |
| 9 | Diversify and rank | Final order list | Redundant designs |
| 10 | Synthesize and express | Protein samples | No material to assay |
| 11 | Measure binding or function | Dose-response data | Single-point false negatives |
| 12 | Learn and iterate | Updated design rules | No improvement across rounds |

### Step 1. Define the objective and the target

**Input.** A biological hypothesis: for example, a binder that blocks a receptor-ligand interface.

**Output.** A design specification: the target, the epitope, the desired affinity, the assay, the
expression host, and the stopping rule.

**Rationale.** "Design a binder" is not a specification. The epitope determines whether blocking
is possible. The affinity target determines the assay and the acceptable hit rate. The host
determines the solubility constraints and the codon optimization.

**If skipped.** A campaign produces binders to an irrelevant surface and cannot say whether the
hypothesis was tested.

### Step 2. Validate the target structure

**Input.** An experimental structure or a predicted model.

**Output.** A target model with per-residue confidence, a list of disordered regions, and a chosen
epitope.

**Rationale.** A predicted target with low confidence in the epitope region is not a reliable
docking surface. The design may fit the prediction and not the protein. Experimental structures
have their own problems: missing loops, alternate conformations, and ligands that displace the
intended site.

| Target structure situation | Design risk | Action |
|---|---|---|
| Experimental structure, complete epitope | Low | Proceed |
| Experimental structure with missing loops in the epitope | Moderate | Model the loops; flag uncertainty |
| Predicted structure, pLDDT above 90 in the epitope | Moderate | Proceed, validate with a second predictor |
| Predicted structure, pLDDT below 70 in the epitope | High | Do not design against this region |
| Multiple conformations of the epitope | High | Design against each, or choose a stable region |
| Target is a disordered region | Very high | Use a different modality or a peptide approach |

**If skipped.** Designs are optimized against an artifact of the prediction.

### Step 3. Choose the design modality

**Input.** The target, the desired function, and the constraints.

**Output.** A modality: de novo binder, scaffolded binder, enzyme, or peptide.

**Rationale.** Different modalities have different design pipelines and different success rates.
A de novo minibinder is small and can be designed from scratch. An enzyme requires a catalytic
site and a substrate channel, which is a harder design problem. A peptide is easier to synthesize
but less stable.

| Modality | Typical size | Design difficulty | Best evidence of success |
|---|---|---|---|
| De novo minibinder | 50 to 120 residues | Moderate | Binding measured by surface plasmon resonance or yeast display |
| Scaffolded binder | 80 to 200 residues | Moderate | Grafted loops on a stable scaffold |
| Enzyme | 200 to 400 residues | High | Catalytic rate measured, not just binding |
| Symmetric assembly | Variable | Moderate to high | Electron microscopy or size exclusion |
| Peptide | 10 to 30 residues | Low to moderate | Binding with weak affinity; stability limited |

**If skipped.** A binder pipeline is applied to an enzyme problem and produces a protein that binds
the substrate but does not catalyze.

### Step 4. Generate backbones

**Input.** The target structure, the epitope, and the modality.

**Output.** A set of backbone models with the binder placed at the epitope.

**Rationale.** RFdiffusion denoises random coordinates into a protein-like backbone, conditioned on
the target and the desired interface. The number of backbones determines the diversity available
to the sequence design step. Too few, and the campaign screens near-duplicates. Too many, and the
compute and synthesis cost grows without a proportional gain.

Three parameters dominate the output and should be recorded.

| Parameter | Effect | Typical starting point |
|---|---|---|
| Binder length | Shorter binders are easier to synthesize and less stable | 50 to 90 residues for a minibinder |
| Epitope specification | Constrains where the binder docks | A residue set on the target surface |
| Number of backbones | Sets diversity before sequence design | A few hundred to a few thousand |
| Symmetry | Produces assemblies rather than monomers | Off unless an assembly is the goal |
| Noise schedule | Affects diversity and realism | Default unless benchmarking |

**If skipped.** Sequence design is applied to a single backbone, and the campaign has no
diversity.

### Step 5. Design sequences

**Input.** Backbones; a sequence design model such as ProteinMPNN.

**Output.** Several sequences per backbone, with a model likelihood per sequence.

**Rationale.** Inverse folding is underdetermined: many sequences can fold into a given backbone.
Sampling several sequences per backbone increases the chance that one expresses and is stable.
The model likelihood is a guide, not a guarantee.

**If skipped.** One sequence per backbone is tested, and a single unfortunate choice removes a
good backbone from consideration.

### Step 6. Predict the designed structures

**Input.** Candidate sequences; structure prediction models.

**Output.** Predicted monomer structures and, where a complex model is available, predicted
complexes.

**Rationale.** The prediction is the first independent check on the design. Two properties matter:
whether the monomer matches the designed backbone, and whether the complex places the binder at
the intended epitope.

**If skipped.** A sequence that cannot fold is synthesized and assayed.

### Step 7. Check interface confidence

**Input.** Predicted complexes; PAE matrices; interface residue sets.

**Output.** Interface PAE, buried surface area, shape complementarity, and hydrogen-bond counts.

**Rationale.** This is the step that separates a folded protein from a binder. A high pLDDT on the
binder with a high inter-chain PAE means the model is confident about the binder's fold and not
confident about where it sits.

The decision has more than two outcomes.

| Monomer pLDDT | Interface PAE | Interpretation | Action |
|---|---|---|---|
| High | Low | Confident fold and confident placement | Advance to experimental filters |
| High | High | Folded but placement uncertain | Deprioritize; check alternative poses |
| Low | Low | Unusual; check for a small or disordered binder | Inspect manually |
| Low | High | Neither fold nor placement confident | Reject |
| High | Low but off-epitope | Confident placement on the wrong surface | Reject or redesign the epitope |
| High | Low but few interface contacts | Confident but weak interface | Advance only with a strong shape complementarity score |

**If skipped.** The shortlist is enriched for well-folded non-binders.

### Step 8. Filter for expression and solubility

**Input.** Sequences and predicted structures.

**Output.** A ranked shortlist with biophysical flags.

**Rationale.** Expression is the most common failure mode and the least modeled. Sequence-level
heuristics help: avoid long hydrophobic stretches, keep net charge near neutral to slightly
positive, avoid cysteine pairs that can form unintended disulfides, and avoid low-complexity
regions.

| Filter | Threshold | Rationale |
|---|---|---|
| Length | Within the validated range for the host | Very short and very long constructs behave differently |
| Net charge at pH 7.4 | Between -2 and +6 | Extreme charge impairs purification and folding |
| Hydrophobic patch | No stretch above the chosen window | Aggregation risk |
| Cysteine count | Even, with no unpaired exposed cysteine | Disulfide scrambling |
| Low-complexity content | Below a stated fraction | Expression and aggregation problems |
| Predicted solubility | Above the tool's threshold | Weak evidence, but better than none |
| Homology to host proteome | Low | Toxicity and interference |

**If skipped.** Designs that cannot express consume the synthesis budget and produce no assay
data.

### Step 9. Diversify and rank

**Input.** Filtered designs with scores.

**Output.** An ordered list with diversity constraints.

**Rationale.** Ranking by a single score selects near-duplicates. A campaign needs diversity
across backbones, epitopes, and interface geometries, because the model's ranking is imperfect.
Clustering by backbone and interface and selecting one or two per cluster is more robust than
taking the top N overall.

**If skipped.** The campaign tests twenty variants of one design and learns little.

### Step 10. Synthesize and express

**Input.** Final sequences; expression host; tags and purification strategy.

**Output.** Purified protein samples with yield and quality metrics.

**Rationale.** The expression system is part of the design. A binder designed for bacterial
expression may not fold in Escherichia coli, and a mammalian target may require a mammalian host
for correct glycosylation. Yield is a filter: a design that expresses at 0.1 milligrams per litre
may not be assayable.

| Host | Strength | Limitation | Use when |
|---|---|---|---|
| Escherichia coli | Fast, cheap, high yield | No glycosylation; inclusion bodies common | Disulfide-free, small binders |
| Pichia pastoris | Secretes protein; eukaryotic folding | Glycosylation differs from human | Eukaryotic domains without human-specific glycans |
| HEK293 | Human-like processing | Slow, expensive | Mammalian targets and glycoproteins |
| Insect cells | High yield, eukaryotic | Different glycosylation | Large complexes |
| Cell-free | Fast, flexible | Cost per milligram | Screening many designs at small scale |

**If skipped.** The assay has no material, and the campaign records a failure that is not a design
failure.

### Step 11. Measure binding or function

**Input.** Purified samples; target; assay.

**Output.** Dose-response curves, rate constants, or functional readouts.

**Rationale.** Single-point screening produces false negatives for weak binders and false positives
from nonspecific interactions. A dose-response measurement distinguishes a real interaction from
an artifact, and it produces the affinity that the program needs.

**If skipped.** The hit rate is uninterpretable, and the program cannot tell whether the design
method or the assay failed.

### Step 12. Learn and iterate

**Input.** Experimental results; design features; scores.

**Output.** An updated ranking model or design rule, and a decision to continue or stop.

**Rationale.** The first round is a calibration experiment. If the top-ranked designs do not
enrich for hits, the ranking is not predictive for this target, and the second round should change
the filter rather than generate more of the same.

**If skipped.** The campaign repeats the same mistake at larger scale.

### 5.13 Benchmarks and what they measure

| Benchmark | What it measures | Strength | Limitation |
|---|---|---|---|
| CASP | Blind prediction of structures chosen by experimentalists | Independent, blind, biennial | Targets are biased toward tractable proteins |
| CAMEO | Continuous server assessment on newly released structures | Frequent, automated | Similar target bias; no design |
| ProteinGym | Fitness prediction across deep mutational scanning assays | Many assays, standardized metrics | Measures variant effect, not design |
| FoldBench | All-atom assembly prediction across tasks | Broad task coverage | Ligand similarity to training data inflates scores |
| Experimental campaigns | Binding or function of designed proteins | The only direct evidence | Expensive, target-specific, often unpublished |

The CASP16 assessment in 2024 found that single-domain fold prediction is close to solved: no
target fold was predicted incorrectly. It also found that the AlphaFold3 server improved complex
prediction over AlphaFold2 (mean DockQ 0.53 versus 0.46 for first-ranked models), that correct
stoichiometry was predicted for only 34% of Phase 0 targets at rank 1, and that model ranking
remains weak, with most groups barely beating random selection among their own five models.

Those findings are the correct frame for a design program. The model is good at the fold and
unreliable at the assembly and the ranking. Design inherits both properties. A design pipeline
that uses a predicted complex should therefore report the ranking uncertainty explicitly, for
example by showing the top five models and their scores, rather than presenting a single best
model as the answer.

### 5.14 Tests before an order is placed

| Test | Procedure | Pass criterion | What it catches |
|---|---|---|---|
| Monomer self-consistency | Predict the monomer and compare with the design | Contact F1 above 0.8, RMSD below 2 angstroms | Sequences that do not encode the fold |
| Interface PAE | Average PAE across the interface | Below the project threshold, typically 5 to 10 angstroms | Folded non-binders |
| Epitope check | Compute interface residues and compare with the intended epitope | Overlap above 80% | Binders on the wrong surface |
| Solubility screen | Sequence heuristics plus a predictor | No hard flags | Expression failures |
| Diversity check | Cluster by backbone and interface | At least one design per cluster | Redundant orders |
| Positive control | Include a known binder in the same assay | Signal in the expected range | Assay failure |
| Negative control | Include a scrambled or non-cognate design | No signal | Nonspecific binding |

The positive and negative controls are not optional. Without them, a campaign with zero hits
cannot distinguish a design failure from an assay failure.

### 5.16 Stopping rules

A campaign needs a pre-specified rule for when to stop, because the funnel arithmetic can always
justify one more round.

| Observation after round 1 | Interpretation | Action |
|---|---|---|
| Hits at or above the predicted rate | The funnel assumptions held | Continue with the same filters at larger scale |
| Hits below the predicted rate, controls valid | The per-design probability was overestimated | Reassess the target, the epitope, or the modality before ordering more |
| Zero hits with a failed positive control | Assay failure | Repeat the assay; do not change the design |
| Zero hits with valid controls and 7.9% prior probability | Consistent with a low-probability campaign | One more round only if the expected value justifies it |
| Hits concentrated in a filter that was down-weighted | The filter was anti-predictive | Invert or remove the filter |
| Hits spread evenly across the ranking | The ranking is not informative | Replace the ranking model |
| Designs express poorly across the board | Expression problem, not design | Change the host or the construct before the next round |

The rule should state the maximum number of rounds and the maximum spend before the campaign
begins. A campaign that discovers its stopping rule after the first result has no stopping rule.

### 5.17 What the procedure does not decide

The procedure produces a ranked, filtered set of designs and a measured result. It does not decide
whether the target is the right target, whether the affinity is sufficient for the therapeutic
hypothesis, or whether the protein is developable as a drug. Those questions require
 pharmacology, manufacturing, and clinical input.

## 6. Worked example

### 6.1 The case

A team wants a de novo minibinder against a receptor tyrosine kinase. The target structure is
experimental, with a complete epitope and a pLDDT-equivalent quality that is high in the region of
interest. The team generates 400 backbones, designs four sequences per backbone, and predicts each
monomer and complex.

### 6.2 Step 1: interpret the confidence metrics

One candidate has the following profile:

```text
  residues              : 120
  mean pLDDT, domain 1  : 91.8
  mean pLDDT, domain 2  : 39.5
  residues below 50     : 60
```

The first 60 residues are a confidently folded domain. The second 60 are below 50, which is the
band where a fixed structure should not be assumed. The PAE blocks:

```text
  domain 1 to domain 1  : 3.8 angstroms
  domain 2 to domain 2  : 4.4 angstroms
  domain 1 to domain 2  : 21.7 angstroms
  domain 2 to domain 1  : 22.3 angstroms
```

The interpretation is that each domain is confident internally and the relative orientation is
not. The correct conclusion is that this candidate is a two-domain protein with a flexible linker,
not a rigid binder. If the design intended a rigid interface between the domains, the design has
failed, even though the mean pLDDT of the first domain is 91.8.

### 6.3 Step 2: check monomer self-consistency

The designed backbone and the predicted monomer are compared by contact map at an 8 angstrom
carbon-alpha cutoff, excluding pairs within two positions in sequence.

```text
  designed contacts   : 27
  predicted contacts  : 29
  shared contacts     : 27
  precision           : 0.931
  recall              : 1.000
  F1                  : 0.964
  Rg designed         : 5.66 angstroms
  Rg predicted        : 5.68 angstroms
```

The sequence encodes the designed fold. The radius of gyration agrees to 0.02 angstroms. This is
a pass on self-consistency, and it says nothing about binding.

### 6.4 Step 3: compute the design funnel

Assume the following per-design probabilities, from prior campaigns against similar targets:

```text
  P(fold)                 = 0.70
  P(express | fold)       = 0.50
  P(bind | express, fold) = 0.05
  P(assay detects | bind) = 0.90
```

The per-design success probability is:

```text
  P = 0.70 x 0.50 x 0.05 x 0.90
    = 0.01575
```

The probability that at least one of \(n\) independent designs succeeds is:

```text
  P(at least one) = 1 - (1 - 0.01575)^n
```

For 24, 96, and 384 designs:

```text
  n = 24  : 1 - 0.98425^24  = 1 - 0.6830 = 0.3170, or 31.7%
  n = 96  : 1 - 0.98425^96  = 1 - 0.2174 = 0.7826, or 78.3%
  n = 384 : 1 - 0.98425^384 = 1 - 0.0023 = 0.9977, or 99.8%
```

The number of designs for a 90% chance:

```text
  n = ln(0.10) / ln(0.98425)
    = -2.302585 / -0.015875
    = 145.0
```

With 400 backbones and four sequences per backbone, the team has 1,600 candidates before
filtering. If the filters retain 10%, the order list is 160 designs, which gives about a 92%
chance of at least one binder under these assumptions.

### 6.5 Step 4: cost per hit

Suppose synthesis and expression cost 400 United States dollars per design, and the binding assay
costs 50 dollars per design. The cost of testing 160 designs is:

```text
  160 x (400 + 50) = 160 x 450 = 72,000 dollars
```

At a 1.575% per-design success rate, the expected number of hits is:

```text
  160 x 0.01575 = 2.52 hits
```

The expected cost per hit is:

```text
  72,000 / 2.52 = 28,571 dollars per hit
```

The number is an expectation, not a guarantee. A campaign with zero hits is consistent with these
assumptions: the probability of zero hits in 160 designs is:

```text
  (1 - 0.01575)^160 = 0.98425^160 = 0.0794, or 7.9%
```

A 7.9% chance of zero hits is high enough that a zero-hit result should not, by itself, be read as
evidence that the design method failed.

### 6.6 Diagnostic 1: pLDDT and PAE disagreement

The first diagnostic that would reveal a broken analysis is the joint distribution of monomer
pLDDT and interface PAE across the shortlist.

A working campaign shows designs spread across the plane, with the best candidates in the
high-pLDDT, low-PAE corner. A broken campaign shows one of three patterns:

- **High pLDDT everywhere, interface PAE high everywhere.** The pipeline is measuring folding and
  calling it binding. This is the most common failure, and it produces a shortlist of well-folded
  non-binders.
- **High pLDDT and low PAE, but no hits in the assay.** The model is confident and wrong, which
  points to a target model error, a wrong epitope, or an assay problem rather than a design
  problem.
- **Low pLDDT across the shortlist.** The design model and the prediction model disagree
  systematically, often because the design was generated with a different tool version or a
  different MSA strategy.

Plotting the two metrics against each other, with the assay result colored, is the fastest way to
see whether either metric is predictive for this target.

### 6.7 Diagnostic 2: enrichment of hits in the ranked list

The second diagnostic is the enrichment of experimental hits in the top fraction of the ranked
designs.

If the ranking is informative, hits concentrate at the top. Suppose 20 of 1,000 tested designs are
true binders, and the top 100 by the in silico score contain 12 of them:

```text
  precision in top 10% = 12 / 100 = 12.0%
  baseline prevalence  = 20 / 1,000 = 2.0%
  enrichment           = 12.0 / 2.0 = 6.0-fold
```

A six-fold enrichment means the score is useful for triage. An enrichment near 1.0 means the
score is not predictive, and the campaign should change the filter rather than generate more
designs. A enrichment below 1.0 means the score is anti-predictive, which usually indicates a bug
in the sign of a score or a mismatch between the design model and the prediction model.

This diagnostic requires assaying a random sample of the shortlist, not only the top. A campaign
that tests only the top cannot compute enrichment, because it has no baseline.

### 6.8 The varied assumption

Change one assumption: the per-design binding probability is 0.2% rather than 5%, because the
target is a shallow, flat surface with no deep pocket.

```text
  P = 0.70 x 0.50 x 0.002 x 0.90 = 0.00063
```

The number of designs for a 90% chance:

```text
  n = ln(0.10) / ln(1 - 0.00063)
    = -2.302585 / -0.0006302
    = 3,654 designs
```

At 450 dollars per design, the cost is:

```text
  3,654 x 450 = 1,644,300 dollars
```

The conclusion changes from a feasible campaign of about 160 designs to an infeasible one of
about 3,650 designs. The correct response is not to order more designs; it is to change the
target, change the modality, or use an experimental display platform that tests millions of
variants rather than a designed shortlist.

This is the most important arithmetic in the topic. A design method that works well on deep
pockets can be economically useless on flat surfaces, and the difference is visible before any
wet-lab work.

## 7. Code, query, or calculation

The script below parses confidence metrics, compares contact maps, computes sequence recovery, and
performs the design-funnel arithmetic, using only the Python standard library. The structures are
synthetic, constructed to have a known geometry. The output was produced by running the script as
written.

```python
"""protein.py - confidence parsing, contact-map comparison, and design arithmetic.

Standard library only. Run: python protein.py
"""
import math


# ------------------------------------------------------------ confidence metrics
# Per-residue pLDDT for a 120-residue protein: a folded domain and a disordered tail.
PLDDT = [93.0 - (i % 5) * 0.6 for i in range(60)] + [36.0 + (i % 7) * 1.2 for i in range(60)]


def mean(xs):
    return sum(xs) / len(xs)


print("CONFIDENCE")
print(f"  residues             : {len(PLDDT)}")
print(f"  mean pLDDT domain 1  : {mean(PLDDT[:60]):.2f}")
print(f"  mean pLDDT domain 2  : {mean(PLDDT[60:]):.2f}")
print(f"  residues below 50    : {sum(1 for x in PLDDT if x < 50)}")

# PAE as four blocks: intra-domain low, inter-domain high.
PAE_BLOCKS = {(0, 0): 3.8, (1, 1): 4.4, (0, 1): 21.7, (1, 0): 22.3}
print("  PAE block means (A):")
for key, val in PAE_BLOCKS.items():
    print(f"    domain {key[0]} to domain {key[1]}: {val:.1f}")


# --------------------------------------------------------------- contact maps
def helix(n, phase=0.0, rise=1.5, radius=2.3, twist=100.0):
    pts = []
    for i in range(n):
        ang = math.radians(phase + i * twist)
        pts.append((radius * math.cos(ang), radius * math.sin(ang), rise * i))
    return pts


DESIGNED = helix(12)

# A predicted model: same fold with a small deterministic perturbation.
PREDICTED = [(x + 0.9 * math.sin(1.7 * i), y + 0.8 * math.cos(1.3 * i), z + 0.6 * math.sin(0.9 * i))
             for i, (x, y, z) in enumerate(DESIGNED)]


def contacts(coords, cutoff=8.0, min_sep=2):
    out = set()
    for i in range(len(coords)):
        for j in range(i + min_sep, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < cutoff:
                out.add((i, j))
    return out


def radius_of_gyration(coords):
    cx = mean([p[0] for p in coords])
    cy = mean([p[1] for p in coords])
    cz = mean([p[2] for p in coords])
    return math.sqrt(mean([(p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2 for p in coords]))


a = contacts(DESIGNED)
b = contacts(PREDICTED)
tp = len(a & b)
precision = tp / len(b) if b else 0.0
recall = tp / len(a) if a else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print("\nCONTACT MAP (C-alpha, 8.0 A, sequence separation >= 2)")
print(f"  designed contacts    : {len(a)}")
print(f"  predicted contacts   : {len(b)}")
print(f"  shared contacts      : {tp}")
print(f"  precision            : {precision:.3f}")
print(f"  recall               : {recall:.3f}")
print(f"  F1                   : {f1:.3f}")
print(f"  Rg designed          : {radius_of_gyration(DESIGNED):.2f} A")
print(f"  Rg predicted         : {radius_of_gyration(PREDICTED):.2f} A")


# ------------------------------------------------------------ sequence recovery
NATIVE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
DESIGNED_SEQ = "MKTAYIAKQREISFVKSHFARQLEERLGLIEVQ"


def sequence_recovery(native, designed):
    same = sum(1 for x, y in zip(native, designed) if x == y)
    return same, len(native), same / len(native)


same, n, rec = sequence_recovery(NATIVE, DESIGNED_SEQ)
print("\nSEQUENCE RECOVERY")
print(f"  identical positions  : {same} of {n}")
print(f"  recovery             : {rec:.1%}")


# --------------------------------------------------------------- design funnel
def prob_at_least_one(p, n):
    return 1.0 - (1.0 - p) ** n


def designs_for(target_prob, p):
    return math.log(1.0 - target_prob) / math.log(1.0 - p)


p_success = 0.02
for n_designs in (24, 96, 384):
    print(f"  test {n_designs:3d} designs at p={p_success:.2f}: "
          f"P(at least one binder) = {prob_at_least_one(p_success, n_designs):.2%}")

n90 = designs_for(0.90, p_success)
print("\nDESIGN FUNNEL")
print(f"  designs for 90% chance: {n90:.1f}")
print(f"  designs for 50% chance: {designs_for(0.50, p_success):.1f}")

# Enrichment: top 10% of 1,000 designs contains 12 of 20 true binders.
top_n, true_in_top, total_true, total = 100, 12, 20, 1000
enrichment = (true_in_top / top_n) / (total_true / total)
print(f"  enrichment of top 10%: {enrichment:.2f}x")
print(f"  precision in top 10% : {true_in_top / top_n:.1%}")
```

Output:

```text
CONFIDENCE
  residues             : 120
  mean pLDDT domain 1  : 91.80
  mean pLDDT domain 2  : 39.48
  residues below 50    : 60
  PAE block means (A):
    domain 0 to domain 0: 3.8
    domain 1 to domain 1: 4.4
    domain 0 to domain 1: 21.7
    domain 1 to domain 0: 22.3

CONTACT MAP (C-alpha, 8.0 A, sequence separation >= 2)
  designed contacts    : 27
  predicted contacts   : 29
  shared contacts      : 27
  precision            : 0.931
  recall               : 1.000
  F1                   : 0.964
  Rg designed          : 5.66 A
  Rg predicted         : 5.68 A

SEQUENCE RECOVERY
  identical positions  : 31 of 33
  recovery             : 93.9%
  test  24 designs at p=0.02: P(at least one binder) = 38.42%
  test  96 designs at p=0.02: P(at least one binder) = 85.62%
  test 384 designs at p=0.02: P(at least one binder) = 99.96%

DESIGN FUNNEL
  designs for 90% chance: 114.0
  designs for 50% chance: 34.3
  enrichment of top 10%: 6.00x
  precision in top 10% : 12.0%
```

Reading the output:

- The pLDDT split at residue 60 is visible in the two means, 91.80 and 39.48. A pipeline that
  reports the overall mean, about 65.6, hides the fact that half the protein is disordered.
- The PAE blocks show the same split at the level of relative orientation: 3.8 and 4.4 within
  domains, 21.7 and 22.3 between them. The matrix is the diagnostic, not the scalar.
- Contact F1 of 0.964 is a pass for monomer self-consistency. It is computed within one chain and
  contains no information about an interface.
- Sequence recovery of 93.9% would be suspiciously high for a de novo design against a natural
  protein; here it is a property of the synthetic example and is meant to show the computation,
  not a realistic value.
- The funnel numbers use p = 0.02, not the 0.01575 of the worked example, because the code is a
  general calculator. At p = 0.02, 96 designs give 85.62%, and 114 designs give 90%.
- The enrichment calculation uses a top 10% subset and a 2% prevalence to produce a six-fold
  enrichment. The same formula with the actual counts from a campaign is the diagnostic in
  section 6.7.

## 8. Data, standards, and regulatory context

### 8.1 Structure data

| Resource | Contents | Use | Limit |
|---|---|---|---|
| Protein Data Bank | Experimental structures | Ground truth for benchmarks | Biased toward tractable proteins |
| AlphaFold Protein Structure Database | More than 200 million predicted structures | Starting models for design | Predictions, not experiments |
| Model Archive | Models accompanying publications | Reproducibility | Variable quality |
| CASP target archive | Blind prediction targets | Benchmarking | Small, biennial |
| ProteinGym | Deep mutational scanning assays | Fitness prediction benchmarks | Variant effect, not design |

The file formats are the PDB format, which is legacy and limited to 99,999 atoms per model, and
mmCIF, which is the current standard for large structures. Validation reports from the
Worldwide Protein Data Bank describe geometry, clashes, and B-factor distributions, and they are
the right template for reporting a predicted model's quality.

### 8.2 Prediction standards and their limits

There is no binding standard for predicted structures. The conventions are:

- **Report the model, version, and date.** AlphaFold2, AlphaFold3, ESMFold, Boltz-1, and Boltz-2
  produce different results, and a version number is part of the result.
- **Report pLDDT and PAE separately.** A single confidence number is not sufficient for a complex.
- **Report the MSA depth and source.** A shallow MSA is a known cause of low accuracy.
- **Report the target's similarity to training data.** FoldBench found that ligand similarity to
  the training set was the decisive factor in protein-ligand accuracy, which is a memorization
  concern.

The AlphaFold3 code was not released at publication; it was available as a server with terms that
restricted commercial use. Open reproductions such as Boltz-1 and Boltz-2 changed the licensing
landscape, but a program should check the license of every model it uses, because the terms differ
and can change.

### 8.3 Benchmark standards

- **CASP** assesses blind prediction every two years. The 16th edition ran in 2024. Its value is
  independence: predictors do not see the answers.
- **CAMEO** assesses servers continuously on structures released after prediction.
- **ProteinGym** standardizes fitness prediction across more than 200 deep mutational scanning
  assays. It measures the ability to rank variants, not the ability to design a binder.

A benchmark result is a statement about the benchmark's distribution of tasks. It is not a
statement about the user's target.

### 8.4 Regulatory context

A designed protein intended as a therapeutic is regulated as a biologic, not as software. The
relevant framework in the United States is the investigational new drug process, with chemistry,
manufacturing, and controls requirements for the protein, and preclinical safety studies before
first-in-human trials. The design method does not have its own regulatory pathway; the product
does.

Two consequences follow:

- A model's accuracy claim is not a regulatory claim. It supports internal decisions, not a
  submission.
- Computational design does not reduce the need for analytical characterization of the expressed
  protein: identity, purity, aggregation, potency, and stability are measured, not predicted.

For research use, the relevant constraints are biosafety, dual-use research review, and
institutional oversight of synthetic DNA ordering. A program that designs a binding protein
against a pathogen target should expect a review that a small-molecule program does not.

### 8.5 What the standards do not cover

- No standard defines a minimum interface PAE for a design to advance.
- No standard defines how to report a design campaign's hit rate.
- No standard requires deposition of designed sequences and structures.
- No benchmark measures design success against clinically relevant targets.

Those gaps are why the funnel arithmetic and the diagnostics in section 6 matter: the program has
to supply the validity evidence that the field has not standardized.

## 9. Industry practice and case evidence

### 9.1 Who does what

The field divides into four groups, and they use the same models for different purposes.

| Group | Examples | Primary use | Evidence of output |
|---|---|---|---|
| Academic design labs | Baker laboratory and collaborators | De novo binders, assemblies, enzymes | Peer-reviewed designs with experimental validation |
| Structure prediction platforms | DeepMind, European Bioinformatics Institute | Databases and servers | AlphaFold Protein Structure Database, AlphaFold Server |
| Open model developers | Boltz, Chai, Protenix and others | Reproducible prediction and affinity models | Open weights and benchmarks |
| Therapeutic companies | Isomorphic Labs, Generate Biomedicines, Absci, and others | Internal pipelines for biologics and targets | Limited public disclosure; press releases are not evidence |

The last row deserves caution. A company that announces a design platform has not necessarily
published a validated design. The published evidence base is concentrated in academic
collaborations, and the commercial claims should be treated as claims until a structure, an assay,
or a clinical result is available.

### 9.2 Documented design campaigns

| Work | What was designed | Documented result | What it demonstrates |
|---|---|---|---|
| Chevalier et al., Nature, 2017 | De novo binders to a target using massively parallel design | Binding confirmed for a subset of designs | Large-scale design and screening is feasible |
| Cao et al., Nature, 2022 | Binders designed from the target structure alone | Nanomolar binders without experimental epitope mapping | Structure-based design can work without prior binders |
| Watson et al., Nature, 2023 | RFdiffusion backbones for binders, symmetric assemblies, and enzyme scaffolds | Experimental validation across several design tasks | Diffusion generates designable backbones |
| Dauparas et al., Science, 2022 | ProteinMPNN sequence design | High recovery and experimental success across scaffolds | Inverse folding is a solved-enough subproblem |
| Bennett et al., Nature Communications, 2023 | Deep-learning-filtered de novo binder design | Improved hit rates over prior filters | In silico filtering can raise the experimental success rate |

The consistent lesson across these campaigns is that hit rates depend heavily on the target. A
deep, hydrophobic pocket is easier than a flat, polar surface. A campaign plan that assumes a
published hit rate without matching the target class is planning against the wrong number.

### 9.3 What has changed in practice

Three shifts are visible between 2020 and 2026.

1. **From one model to many.** AlphaFold2 was a single reference point. By 2026 a program chooses
   among AlphaFold3-class models, open reproductions, and protein language model folders, and
   compares their outputs.
2. **From structure to affinity.** Boltz-2 added binding affinity prediction, which moves the
   model closer to the decision. The accuracy claim, a Pearson correlation of about 0.62 on the
   OpenFE benchmark, is promising and is not a substitute for measurement.
3. **From design to screening.** The most reliable campaigns combine design with a display
   platform that tests many variants, rather than ordering a small shortlist and hoping. The
   design model proposes; the display platform tests at scale.

### 9.4 The wet-lab bottleneck

The computational cost of generating 10,000 designs is small relative to the cost of synthesizing
and assaying 100. That asymmetry shapes practice: teams generate far more designs than they can
test, and the binding constraint is the experimental funnel, not the model.

Two operational consequences follow. First, the filter that decides which 100 of 10,000 designs
to order is the highest-leverage component of the pipeline, and it is usually the least validated.
Second, a program should measure the predictive value of its filter on a random sample, not only on
the designs it chose to test.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Target model wrong in the epitope | Low pLDDT in the region of interest | Prediction error or a disordered region | Designs fit an artifact | Check regional confidence; use a second predictor |
| pLDDT used as a binding score | Shortlist is all high-pLDDT monomers | Folding confidence conflated with interface confidence | Well-folded non-binders | Report interface PAE separately |
| Interface PAE ignored | Complexes predicted but never inspected | Relative orientation unmeasured | Binder placed on the wrong surface | Average PAE across the interface |
| Self-consistency treated as validation | RMSD reported as evidence of function | The check is circular: same model family predicts and validates | False confidence | Add an orthogonal metric and an assay |
| No expression filter | All designs are ordered at equal priority | Sequence properties ignored | Most samples fail to express | Charge, hydrophobicity, cysteine, low-complexity filters |
| Ranking not validated | Top designs are the only ones assayed | No baseline to compute enrichment | Ranking may be useless or anti-predictive | Assay a random sample; compute enrichment |
| Assay without controls | Zero hits with no positive control | Assay may have failed | Design method blamed incorrectly | Include positive and negative controls |
| Single-point binding assay | Weak binders scored as non-binders | Concentration below the detection limit | False negatives | Dose-response for all candidates |
| Hit rate reported without denominator | "We found binders" | Number tested omitted | Result is uninterpretable | Report tested, hit, and assay conditions |
| Redundant designs ordered | Top 20 are near-identical | Ranking collapses diversity | Budget spent on one hypothesis | Cluster by backbone and interface |
| Ligand similarity to training data ignored | Very high accuracy on a familiar ligand | Memorization | Overconfidence on novel chemistry | Check similarity; benchmark on novel scaffolds |
| Model version not recorded | Results cannot be reproduced | Weights and code change | Irreproducible campaign | Record model, version, date, and MSA source |
| Stoichiometry assumed | Complex prediction with the wrong chain count | The model predicts a plausible but wrong assembly | Wrong design target | Specify stoichiometry; check Phase 0-style predictions |
| Designed protein immunogenic | No immunogenicity assessment | Non-human sequences can be immunogenic | Program failure late | Assess in silico and experimentally early |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The target structure is accurate in the epitope | Design is only as good as the target model | Prediction error, disorder, conformational change | Regional pLDDT and PAE; a second structure |
| The designed sequence folds as intended | The backbone is only useful if encodable | Sequence-structure mismatch | Monomer self-consistency |
| The predicted complex reflects the binding pose | Interface design depends on the pose | Model predicts a plausible but wrong pose | Interface PAE; mutagenesis |
| In silico filters predict expression | Synthesis budget requires triage | Host-specific effects | Retrospective analysis of prior campaigns |
| The assay measures binding, not aggregation | A signal must reflect the intended interaction | Nonspecific binding | Negative control and dose-response |
| The hit rate from a prior campaign transfers | Planning requires a number | Different target class or scaffold | Pilot round with a random sample |
| The model's training data does not include the target | Novelty is assumed | Memorization inflates accuracy | Similarity check against the training set |
| Designs are independent | The funnel arithmetic assumes independence | Shared backbone or interface | Cluster and count independent designs |
| The therapeutic hypothesis is correct | A binder is not a drug | The target does not drive disease | Biology validation outside the design pipeline |
| Benchmarks generalize | Model selection uses benchmark rankings | Benchmark targets differ from the program's | Internal benchmark on the program's target class |

## 12. Exercises

### Exercise 1. Interpret a confidence profile

**Task.** A 200-residue prediction has a mean pLDDT of 88 for residues 1 to 120 and 32 for residues
121 to 200. The PAE between the two regions is 24 angstroms. State what can and cannot be
concluded.

**Deliverable.** A short interpretation with three claims labeled supported, unsupported, or
unknown.

**Acceptance criteria.** The folded domain is supported; the disordered region is not a failed
prediction but a low-confidence region; the relative orientation is unsupported. The answer does
not conclude that the protein is a single rigid unit.

**Time.** 20 minutes.

### Exercise 2. Compute the design funnel

**Task.** A campaign assumes P(fold) = 0.8, P(express) = 0.6, P(bind | express, fold) = 0.03, and
P(assay detects) = 0.95. Compute the per-design success probability and the number of designs
needed for a 90% chance of at least one hit.

**Deliverable.** The arithmetic with units.

**Acceptance criteria.** P = 0.8 x 0.6 x 0.03 x 0.95 = 0.01368. n = ln(0.10)/ln(0.98632) =
167.7, rounded to 168 designs. The answer notes that the factors are not independent.

**Time.** 25 minutes.

### Exercise 3. Compare contact maps

**Task.** Design a ten-residue helix and a predicted model with a small perturbation. Compute
contacts at an 8 angstrom cutoff with a sequence separation of at least two, and report precision,
recall, and F1.

**Deliverable.** The contact sets and the three metrics.

**Acceptance criteria.** The method is correct, the counts are reported, and the answer states
that the metric is a monomer self-consistency check and says nothing about binding.

**Time.** 40 minutes.

### Exercise 4. Design a filter set

**Task.** Specify six filters for a bacterial expression campaign of 100 minibinders, with a
threshold and a rationale for each.

**Deliverable.** A filter table.

**Acceptance criteria.** The filters cover length, net charge, hydrophobic patches, cysteine
content, low-complexity regions, and one solubility predictor. Each threshold has a stated reason
and a note on its evidence strength.

**Time.** 35 minutes.

### Exercise 5. Diagnose a zero-hit campaign

**Task.** A campaign tests 150 designs, finds no binders, and reports that the design method
failed. The positive control gave a signal, the designs expressed at 0.2 milligrams per litre, and
the assay was a single-point measurement at 1 micromolar. Identify at least three alternative
explanations and the evidence that would distinguish them.

**Deliverable.** A diagnostic memo.

**Acceptance criteria.** The memo considers expression yield, assay sensitivity, target model
error, and the prior probability of zero hits. It notes that with a 1.6% per-design rate, 150
designs have a 9.1% chance of zero hits, so the result is not conclusive on its own.

**Time.** 40 minutes.

### Exercise 6. Compute enrichment

**Task.** Of 500 tested designs, 15 bind. The top 50 by score contain 9 binders. Compute precision,
baseline prevalence, and enrichment, and interpret.

**Deliverable.** The arithmetic and a two-sentence interpretation.

**Acceptance criteria.** Precision 18%, baseline 3%, enrichment 6-fold. The answer states that the
score is useful for triage and that the estimate has a wide confidence interval with 15 events.

**Time.** 25 minutes.

### Exercise 7. Critique a benchmark claim

**Task.** A vendor states that its model "achieves 90% accuracy on CASP16 targets and therefore
predicts binding". Write a critique.

**Deliverable.** A 350-word critique.

**Acceptance criteria.** The critique separates fold accuracy from binding prediction, notes that
CASP targets are selected for tractability, asks whether the 90% is GDT or another metric, and asks
for interface-level evidence. It does not accept the inference from structure to binding.

**Time.** 30 minutes.

### Exercise 8. Design a campaign protocol

**Task.** Write a two-page protocol for a de novo binder campaign against a chosen target class,
including the funnel assumptions, the filters, the assay, the controls, the stopping rule, and the
budget.

**Deliverable.** The protocol.

**Acceptance criteria.** The protocol states the target structure source and its confidence, the
number of backbones and sequences, the interface filter, the expression host, the assay with
controls, the pre-specified stopping rule, and a budget derived from the funnel arithmetic.

**Time.** 60 minutes.

## 13. Self-check answers

**Exercise 1.** Supported: residues 1 to 120 form a confidently folded domain, with a mean pLDDT
of 88. Unsupported: the relative orientation of the two regions, because the inter-region PAE of
24 angstroms is high. Unknown: whether residues 121 to 200 have a fixed structure at all; a mean
pLDDT of 32 is in the band where intrinsic disorder is the most likely explanation. The correct
conclusion is a folded domain with a disordered tail, not a failed prediction of a rigid protein.

**Exercise 2.** P = 0.8 x 0.6 x 0.03 x 0.95. First, 0.8 x 0.6 = 0.48. Then 0.48 x 0.03 = 0.0144.
Then 0.0144 x 0.95 = 0.01368. The number of designs for a 90% chance is n = ln(0.10) /
ln(0.98632) = -2.302585 / -0.013774 = 167.2, rounded up to 168. The independence assumption is an
approximation: designs sharing a backbone are correlated, which increases the variance and means
the effective number is lower than the count.

**Exercise 3.** A correct answer builds the coordinate sets, computes all pairwise distances,
selects pairs below 8 angstroms with a sequence separation of at least two, and compares the sets.
The exact counts depend on the geometry. The key point is that precision and recall answer a
different question from binding, and a high F1 should never be reported as evidence that a design
works.

**Exercise 4.** Rubric. Full credit requires six filters with thresholds and rationales, and an
honest statement of which filters have weak evidence. A filter set that includes only length and
charge is incomplete. A filter set that claims a solubility predictor is highly accurate is
overclaiming.

**Exercise 5.** Alternative explanations: the designs expressed at 0.2 milligrams per litre, which
may be too low for the assay; the single-point assay at 1 micromolar may miss weak binders; the
target model may be wrong in the epitope; and the prior probability of zero hits in 150 designs at
a 1.6% rate is 0.984^150 = 0.089, or 8.9%, so zero hits is not by itself evidence of failure. The
distinguishing evidence is a dose-response assay, a target-model confidence check, and a
comparison of the designs' expression levels with the campaign's validated range.

**Exercise 6.** Precision in the top 50 is 9/50 = 18%. Baseline prevalence is 15/500 = 3%.
Enrichment is 18/3 = 6-fold. The score concentrates binders, which supports its use for triage.
With 15 events, the confidence interval on the enrichment is wide, and the estimate should be
reported with it.

**Exercise 7.** Rubric. Full credit separates fold accuracy from binding, questions the metric,
notes the target selection bias in CASP, and asks for interface and affinity evidence. A critique
that accepts the inference because the model is accurate on structures fails.

**Exercise 8.** Rubric. Full credit requires a target confidence statement, a funnel with numbers,
an interface filter, an expression host and filter, an assay with controls, a stopping rule, and a
budget consistent with the arithmetic. A protocol that omits the stopping rule or the negative
control fails.

## 14. Interview questions

**1. What is the difference between pLDDT and PAE?**
A strong answer separates local geometry confidence from pairwise relative orientation and gives
the case of two confident domains with an uncertain arrangement. A weak answer treats both as
"confidence".

**2. Why is a high pLDDT not evidence that a designed binder binds?**
A strong answer explains that pLDDT is computed per chain and says nothing about the interface,
and names interface PAE as the relevant check. A weak answer says the score is "high quality".

**3. What does AlphaFold3 add over AlphaFold2?**
A strong answer describes the pairformer trunk and diffusion module, the extension to ligands and
nucleic acids, and the fact that it was server-only at release. A weak answer says it is "more
accurate".

**4. How does RFdiffusion generate a backbone?**
A strong answer describes denoising in three-dimensional coordinate space conditioned on the
target and the epitope. A weak answer says it "uses AI to design proteins".

**5. What does ProteinMPNN do, and what does it not do?**
A strong answer describes inverse folding, sequence design for a fixed backbone, and states that
it does not predict binding. A weak answer says it "validates the design".

**6. How would you decide how many designs to order?**
A strong answer uses the product of conditional probabilities and the binomial formula for at
least one success, then checks the budget. A weak answer says "as many as possible".

**7. What is the wet-lab gap?**
A strong answer names expression, stability, assay sensitivity, and biology as separate failure
points. A weak answer says "models are not perfect".

**8. What does CASP measure, and what does it not measure?**
A strong answer describes blind structure prediction on experimentally chosen targets and notes
the selection bias and the absence of design or affinity assessment. A weak answer treats a CASP
ranking as a product benchmark.

**9. How would you detect that a ranking model is not predictive?**
A strong answer assays a random sample and computes enrichment against baseline prevalence. A
weak answer says "look at the top hits".

**10. When would you stop a design campaign?**
A strong answer uses a pre-specified stopping rule tied to the funnel arithmetic and the observed
hit rate, with controls to distinguish assay failure from design failure. A weak answer says
"when we run out of budget".

## 15. Key terms

- **AlphaFold2** — structure prediction model using MSAs and attention.
- **AlphaFold3** — structure prediction model using a pairformer and diffusion module.
- **Backbone** — carbon-alpha trace without side chains.
- **Boltz-1** — open reproduction of AlphaFold3.
- **Boltz-2** — open model for structure and binding affinity.
- **CAMEO** — continuous assessment of structure prediction servers.
- **CASP** — blind assessment of structure prediction.
- **Contact map** — set of residue pairs within a distance cutoff.
- **Deep mutational scanning** — systematic measurement of variant effects.
- **Enrichment** — over-representation of hits in a ranked subset.
- **Epitope** — target surface recognized by a binder.
- **ESMFold** — structure prediction model built on a protein language model.
- **Hit rate** — fraction of tested designs meeting the assay criterion.
- **Interface** — residues in contact across a chain boundary.
- **Inverse folding** — designing a sequence for a given backbone.
- **mmCIF** — file format for macromolecular structures.
- **Multiple sequence alignment** — aligned homologous sequences.
- **pLDDT** — per-residue local confidence score.
- **PAE** — predicted aligned error matrix.
- **Protein Data Bank** — repository of experimental structures.
- **Protein language model** — transformer trained on protein sequences.
- **ProteinGym** — benchmark of fitness prediction across DMS assays.
- **ProteinMPNN** — message-passing inverse folding model.
- **RFdiffusion** — diffusion model for protein backbone generation.
- **RMSD** — average atomic distance after superposition.
- **Self-consistency** — agreement between a design and a prediction of its sequence.
- **Shape complementarity** — geometric fit of two surfaces.
- **Solubility** — propensity to remain in solution.
- **Stoichiometry** — chain composition of a complex.
- **Template** — experimental structure used as a model input.

## 16. Sources

Accessed 26 October 2026 unless otherwise stated.

- Jumper, J., et al. *Highly Accurate Protein Structure Prediction with AlphaFold*. Nature 596,
  583-589, 2021. https://doi.org/10.1038/s41586-021-03819-2
- Abramson, J., et al. *Accurate Structure Prediction of Biomolecular Interactions with
  AlphaFold 3*. Nature 630, 493-500, 2024. https://doi.org/10.1038/s41586-024-07487-w
- Lin, Z., et al. *Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language
  Model*. Science 379, 1123-1130, 2023. https://doi.org/10.1126/science.ade2574
- Watson, J. L., et al. *De Novo Design of Protein Structure and Function with RFdiffusion*.
  Nature 620, 1089-1100, 2023. https://doi.org/10.1038/s41586-023-06415-8
- Dauparas, J., et al. *Robust Deep Learning-Based Protein Sequence Design Using ProteinMPNN*.
  Science 378, 49-56, 2022. https://doi.org/10.1126/science.add2187
- Bennett, N. R., et al. *Improving De Novo Protein Binder Design with Deep Learning*. Nature
  Communications 14, 2625, 2023. https://doi.org/10.1038/s41467-023-38328-5
- Cao, L., et al. *Design of Protein-Binding Proteins from the Target Structure Alone*. Nature 605,
  551-560, 2022. https://doi.org/10.1038/s41586-022-04654-9
- Chevalier, A., et al. *Massively Parallel De Novo Protein Design for Targeted Therapeutics*.
  Nature 550, 74-79, 2017. https://doi.org/10.1038/nature23912
- Notin, P., et al. *ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and
  Design*. Advances in Neural Information Processing Systems, 2023.
  https://doi.org/10.1101/2023.12.07.570727
- Wohlwend, J., et al. *Boltz-1: Democratizing Biomolecular Interaction Modeling*. bioRxiv, 2024.
  https://doi.org/10.1101/2024.11.19.624167
- Boltz. *Boltz-2: Joint Structure and Binding Affinity Prediction*. https://boltz.com/boltz2
- Xu, et al. *Benchmarking All-Atom Biomolecular Structure Prediction with FoldBench*. Nature
  Communications, 2025. https://doi.org/10.1038/s41467-025-67127-3
- Elofsson, A. *AlphaFold3 at CASP16*. Proteins, 2025. https://doi.org/10.1002/prot.70044
- CASP16 results. https://predictioncenter.org/casp16/results.cgi
- Varadi, M., et al. *AlphaFold Protein Structure Database in 2024*. Nucleic Acids Research, 2024.
  https://doi.org/10.1093/nar/gkad1011

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`13_spatial-omics-and-pathology-integration.md`](13_spatial-omics-and-pathology-integration.md)
- Next topic: [`15_crispr-and-gene-therapy-pipelines.md`](15_crispr-and-gene-therapy-pipelines.md)
- Related topic: [`09_drug-discovery.md`](09_drug-discovery.md)
- Related topic: [`06_proteomics-and-metabolomics.md`](06_proteomics-and-metabolomics.md)
- Module survey: [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Laboratory exercises and datasets: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

