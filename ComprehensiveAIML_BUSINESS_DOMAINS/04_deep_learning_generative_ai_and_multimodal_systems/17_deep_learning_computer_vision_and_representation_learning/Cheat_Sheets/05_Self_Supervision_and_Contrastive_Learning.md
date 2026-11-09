---
collection: "module-17-vision"
file_id: "05"
topic: "Self-supervision and contrastive learning"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Augmentation policy, negative population, encoder version, or downstream-label distribution change."
notation_style: "plain-text"
prerequisites: ["02_CNNs_Attention_and_Transfer_Learning.md"]
pattern_tags: ["self-supervision", "contrastive-learning", "representation"]
source_status: "OPEN — visible source name only"
owner_role: "Vision ML reviewer"
source_index: "INDEX.md"
---

# Self-Supervision and Contrastive Learning

> **Purpose:** Define positive-pair semantics, negative population, and augmentation invariances before claiming a representation is useful.
> **Prerequisites:** [CNNs, attention and transfer learning](02_CNNs_Attention_and_Transfer_Learning.md), cross-entropy, grouped splits.
> **Key outcome:** Given similarities and temperature, compute an InfoNCE term and diagnose a false-negative or destructive-augmentation risk.
> **This file does NOT cover:** generative reconstruction objectives, foundation-scale data governance, or a measured embedding benchmark.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED objective; VERIFIED hand calculation; SOURCED primary papers; illustrative workload.
> **Retrieval cue:** If representation quality is high on probes but fails a critical slice, start at Section 8.

## 1. File Header

Self-supervision creates a learning signal from data transformations, masked content, or paired modalities rather than an externally labelled class for every input. The visible inventory names `13_Self_Supervised_Contrastive_Learning.md` at [`directory.txt`](../../../../Templates/directory.txt#L561); unavailable source bodies are not reconstructed.

**Takeaway:** The augmentation policy is part of the label definition in contrastive learning.

## 2. Concept Map / Mental Model

```text
base image x -> view t1(x) -> encoder -> z1 -- positive pair -- z2 <- encoder <- view t2(x)
                         |                                  |
                         +-- must preserve task identity      +-- compare against batch/memory candidates
similarities / temperature -> contrastive loss -> representation -> frozen probe or adapted downstream task
augmentation destroys label feature OR negative is same entity -> objective teaches the wrong invariance
```

**Takeaway:** “Unlabelled” does not mean assumption-free.

## 3. Foundational Definitions

| Term | Meaning | Shape / units | Boundary |
|---|---|---|---|
| view | transformed observation of a base item | image tensor | semantics must be declared |
| positive pair | two views intended to represent same identity | pair IDs | may be invalid under crop/color changes |
| negative | comparison item intended to differ | batch/memory item | can be a false negative |
| embedding z | encoder output, commonly normalised | [D] | norm convention stated |
| temperature tau | softmax scale | positive scalar | changes gradient concentration |
| linear probe | frozen-representation downstream head | evaluation procedure | not production performance |

**Takeaway:** Record entity identity and transformation seeds so pair semantics can be audited.

## 4. Core Formulas with Full Derivation

Validity before use: one anchor i, a declared positive j, candidate set A containing j, cosine-like similarity s, and tau>0. Softmax assigns probability to the declared positive among candidates; negative log probability creates the loss.

```text
FORMULA [SSL-01]: L_i = -log( exp(s(i,j)/tau) / sum_{k in A} exp(s(i,k)/tau) )
WHERE: L_i dimensionless; s similarity; tau>0 dimensionless scale; A candidate indices; j positive index.
CONSTRAINT: positive relation must preserve target-relevant identity; candidates must follow the stated batch/memory policy.
NUMERICAL EXAMPLE: s_pos=0.8, two negatives 0.4 and 0.1, tau=1.
                   exp values are 2.2255,1.4918,1.1052; p_pos=2.2255/4.8225=0.4617;
                   L_i=-ln(0.4617)=0.7728.
INTERPRETATION: the positive is preferred but remains far from probability one among these candidates.
```

Arithmetic is hand-derived and rounded to four decimals. Lower tau accentuates relative similarity differences but can make optimisation sensitive to hard/false negatives.

For L2-normalised embeddings, cosine similarity is s(i,k)=z_i dot z_k. Validity requires nonzero vectors before normalisation; it is not an object identity proof.

Counterexample: two photographs of different products with the same SKU family can be semantically similar and thus a “hard negative”; two views of the same product may need separation if the downstream task is defect state, not product identity.

**Takeaway:** A low contrastive loss is evidence only that the specified pair game was solved.

## 5. Worked Numerical Example

**ILLUSTRATIVE, hand-checked arithmetic; not an executed experiment.** A warehouse image is augmented by crop, mild brightness jitter, and horizontal flip. The intended downstream task is package-type retrieval, so these transforms are proposed to preserve identity. If the task instead detects left/right orientation labels, flipping violates positive-pair semantics. In the SSL-01 trace, lowering tau from 1 to 0.5 changes exponents to exp(1.6)=4.9530, exp(0.8)=2.2255, exp(0.2)=1.2214; p_pos=4.9530/8.3999=0.5897 and L=0.5282. The objective becomes more confident without new evidence that the relation is correct.

Sensitivity: add a false negative with similarity 0.75 at tau=0.5. Its exp(1.5)=4.4817 reduces p_pos to 4.9530/(8.3999+4.4817)=0.3845 and raises loss to about 0.9558. That rise can indicate useful discrimination or a broken negative policy; inspect identity metadata.

**Takeaway:** Temperature and candidate composition can change loss without improving the downstream decision.

## 6. Visual / Geometric Intuition

```text
embedding space
  view A o----o view B     (declared positive: pull together)
              \
               o look-alike from another entity (negative: push apart, but inspect identity)

wrong crop removes defect -> two visually compatible views may carry conflicting downstream labels
```

The geometry requests relative ordering, not a universal map of visual meaning. Augmentations encode which variations should collapse. A retrieval application may want camera invariance; a safety inspection may need camera/device metadata preserved as an explicit input or slice.

**Takeaway:** Decide which variation is nuisance before training it away.

## 7. Comparison / Selection Table

| Method family | Signal | Use when | Avoid when | Validation |
|---|---|---|---|---|
| supervised transfer | adjudicated labels | label task is well-defined | label budget is tiny | grouped target metrics |
| contrastive | view/pair identity | invariances can be justified | false negatives common/untracked | probe plus slice/error audit |
| masked prediction | withheld content | reconstruction supports representation aim | pixels/tokens do not align with action | downstream task suite |
| clustering/prototype | assignment consistency | natural groups are meaningful | group collapse harms rare class | occupancy and downstream recall |

**Takeaway:** Evaluate representations through the downstream decision, not only their pretext loss.

## 8. Failure Modes & Edge Cases

| Failure | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|
| downstream critical feature erased | destructive augmentation or insufficient capacity | replay paired views with label-relevant annotation | remove/condition transform | target slice recall |
| loss stays high | false negatives or optimisation issue | inspect near-neighbour identity and gradients | repair candidate policy before tuning tau | false-negative audit |
| probe high, deployment low | split leakage or representation shift | entity/site/time grouped probe | freeze split; collect target data | grouped confidence interval |
| collapse/low diversity | objective imbalance or implementation bug | embedding norm, covariance, neighbour distribution | inspect normalisation/objective | fixed-input trace |
| rare-class failure | majority-defined invariance or labels missing | support/rate by critical class | targeted pairs/data and threshold policy | rare-slice gate |

**Takeaway:** Diagnose pair semantics before escalating model size.

## 9. Complexity & Cost Analysis

For batch B and embedding width D, constructing all pair dot products is O(B^2D) and stores O(B^2) similarities if materialised. With B=512 and D=256, pair products number 512^2*256=67,108,864. This is a derived workload count, not a device benchmark. Larger batches give more comparisons but increase memory; memory banks alter freshness and negative provenance. Record which candidate source was used.

**Takeaway:** Batch size changes both resource use and the statistical meaning of negatives.

## 10. Key Theorems / Results Reference

- InfoNCE is a contrastive objective with bounds interpreted under stated sampling assumptions; it does not guarantee semantic transfer.
- [Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*](https://arxiv.org/abs/2002.05709), 2020, documents SimCLR; reviewed 2026-11-09.
- [He et al., *Momentum Contrast*](https://arxiv.org/abs/1911.05722), 2019, documents a queue-based contrastive design; source claims must be scoped to its experiment.

**Takeaway:** Theorems and papers do not substitute for target-pair validity evidence.

## 11. Interview / Exam Questions

1. **Compute SSL-01 when all three similarities equal and tau is finite.** Positive probability is 1/3 and loss ln(3).
2. **Why can a flip be invalid?** It can alter orientation, text, anatomy side, or another label-defining signal.
3. **What is a false negative?** A candidate treated as different although it shares the intended identity/semantic class.
4. **Does lower loss prove better retrieval?** No; it may reflect tau, candidate policy, or destructive invariance.
5. **Why use grouped probes?** Entity/site leakage can inflate a representation's apparent transfer.
6. **What does normalisation change?** Dot product becomes cosine-like similarity, but zero-vector handling remains required.
7. **Changed constraint: memory budget halves.** Reduce batch or use documented queue strategy, then re-evaluate candidate semantics and quality.
8. **How audit positives?** Sample transformations with domain reviewers and annotate whether the downstream-relevant label survives.
9. **What is a wrong fix for false negatives?** Simply increase batch size; it can increase collisions.
10. **When prefer labels?** When the high-cost target action has a precise adjudicated label and sufficient coverage.

**Takeaway:** Strong answers expose the hidden pair and invariance assumptions.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Positive | State identity and transformations that preserve it |
| Negative | State batch/memory population and false-negative audit |
| Objective | SSL-01 with tau>0 and declared candidate set |
| Evaluation | Grouped probe plus target slices, calibration, error review |
| Capacity | Similarity work O(B^2D); record B and candidate source |
| Avoid | Equating pretext loss with production reliability |
| Evidence | Hand-checked illustrative values; no benchmark or source-body claim |
