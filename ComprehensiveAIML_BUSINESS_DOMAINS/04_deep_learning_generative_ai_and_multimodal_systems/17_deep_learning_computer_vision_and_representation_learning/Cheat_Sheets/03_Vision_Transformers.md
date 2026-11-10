---
collection: "module-17-vision"
file_id: "03"
topic: "Vision transformers"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Patch size, input resolution, positional-encoding scheme, attention pattern, or target-image distribution changes."
notation_style: "plain-text fenced"
prerequisites: ["02_CNNs_Attention_and_Transfer_Learning.md", "01_Vision_Evaluation_and_Error_Taxonomy.md"]
pattern_tags: ["vision-transformer", "patch-tokenisation", "attention-budget"]
source_status: "OPEN — visible title only; original source body not inspected"
owner_role: "Vision ML reviewer"
source_index: "INDEX.md"
---

# Vision Transformers: Patch Tokens, Position, and Attention Budgets

> **Purpose:** Convert an image size and patch size into a token, score-matrix and parameter budget, then choose global, windowed, hierarchical or hybrid attention on evidence rather than fashion.
> **Prerequisites:** [CNNs, attention and transfer learning](02_CNNs_Attention_and_Transfer_Learning.md) for convolution geometry and the scaling of dot-product attention; the task/split protocol in [Vision evaluation and error taxonomy](01_Vision_Evaluation_and_Error_Taxonomy.md); matrix multiplication.
> **Key outcome:** Given H, W, P, D and h, compute token count, attention-score count, per-layer parameter counts and the memory envelope, then identify which of four plausible causes explains an out-of-memory event or an accuracy regression.
> **This file does NOT cover:** detector heads and mask losses, video temporal attention, generative image models, or foundation-model pretraining recipes.
> **Level:** 2 — Core methods.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED geometry and cost algebra; VERIFIED illustrative arithmetic (independently recomputed in §5); SOURCED paper definitions named in §10; no benchmark, latency or accuracy claim.
> **Retrieval cue:** If a resolution change caused an out-of-memory failure or a metric regression, start at §4.1, §5 Step 4 and §8 entry F-03.

## 1. File Header

A vision transformer (ViT) cuts an image into a grid of patches, projects each patch to a vector, adds position information, and lets a stack of transformer blocks mix those vectors. The visible course inventory names a “Vision Transformers / ViT” source title; that source body was not available for inspection, so everything here is a from-first-principles teaching derivation of the visible topic, not a reconstruction of the original lesson.

The decision this sheet supports is not “should we use a transformer” but a joint feasibility question. Does this token budget, at this resolution and patch size, fit the accuracy requirement and the resource envelope? Which attention pattern is the cheapest one that still sees what the task needs? A ViT is a resource contract disguised as an architecture: resolution and patch size together determine a quadratic term that dominates memory long before it dominates arithmetic.

**The four artefacts this sheet produces.** Each later section serves one row; a project missing a row has an unfalsifiable design.

| Artefact | Question it freezes | Fails when |
|---|---|---|
| Token ledger | How many positions exist at each stage, at this resolution and patch size? | Token counts are recomputed from memory and disagree |
| Cost ledger | What are the per-block MACs, parameters and peak score-matrix bytes? | The model is sized by parameter count and OOMs at training step one |
| Position contract | Which positional scheme, at which grid, with which interpolation rule? | A resolution change is applied without touching positions |
| Pretraining contract | Which corpus, which augmentation, which licence, which label vocabulary? | A checkpoint is used whose preprocessing differs from the pipeline |

**The three claims worth holding from the start.** First, patch size is a representation decision and a compute decision simultaneously, so changing it is never a pure efficiency tweak. Second, token count rather than pixel count is the first-order driver of attention cost, and token count scales as the inverse square of patch size. Third, positional handling is what turns an unordered bag of patch vectors into an image, so any resolution adaptation that touches positions is a new model hypothesis needing its own evaluation.

**Contract vector.** Freeze the comparison as a tuple:

```text
V = (patch_size, input_resolution, width D, depth L, heads h,
     attention_pattern, positional_scheme, pretrain_corpus, decode_rule)
```

Two accuracy numbers are comparable only when their tuples agree on every component that moves the estimator. A patch-size change alters the token count, the receptive field of one block and the spatial resolution available to a dense head, so a single-field diff is a three-way change; treat it as a new hypothesis rather than a variant.

**Where the numbers come from.** Every workload, resolution, width, batch size, GPU hour and cost figure in this file is an ILLUSTRATIVE construction whose arithmetic the reader can recompute; §5 performs an independent re-check of the central ones. Definitions of patch embedding, class token, learned absolute position embeddings, scaled dot-product attention, window attention with shifting, and hierarchical token merging are SOURCED conventions from the named primary works in §10; an implementation may legally differ provided it says so. No value here is a measurement of any product, dataset or device.

**Ownership.** The vision reviewer owns the token and cost ledgers and signs the position contract. The data owner owns split eligibility and the evaluation cohort. The deployment owner owns the latency and memory ceiling and therefore holds a veto on resolution and patch size. A role may not silently change another's frozen value; a change to any tuple component requires a new comparison.

**What the reader must be able to do afterwards.** Compute N from H and P; compute per-block and whole-network MACs; compute the score-matrix size in bytes for a given batch, head count and precision; state the resolution at which the quadratic term overtakes the dense projections; choose between global, windowed and hierarchical attention with a stated reason; and reject a promotion whose position contract cannot be reconstructed.

**How the artefacts are used in a review.** The token ledger is checked first, because a wrong N makes every downstream number wrong and is the cheapest error to catch. The cost ledger is checked second, against the same N, so that arithmetic and memory are seen to be consistent. The position contract is checked third, because it is the component most often omitted from a run manifest and the one that silently invalidates an evaluation when the input grid changes between validation and serving. The pretraining contract is checked last, and it is the only one whose violation can be discovered after the model has been trained, which is why it is documented rather than tested.

**Scope boundaries against neighbouring sheets.** Detector heads, anchor assignment and mask losses belong to the detection sheet; this file supplies the token geometry and the attention budget those heads consume. Evaluation algebra — threshold selection, average precision, calibration error, queue sizing — belongs to the evaluation sheet, and this file refers to it rather than duplicating it. Convolutional geometry and the scaling of dot-product attention are established in the companion CNN sheet, and the derivations here assume them rather than restating them.

**What this sheet refuses to answer.** It will not say whether transformers beat convolutions on your task, because that is a measurement your split must supply. It will not predict latency from a MAC count, because at the token counts where these models are used the workload is frequently memory-bound and the operator implementations differ by more than the arithmetic does. And it will not recommend a patch size, because the right patch size is a property of the smallest discriminative structure in the images, which is a property of the task and not of the architecture.

**Maintenance trigger.** Re-derive the arithmetic whenever a named architecture's block definition changes — a change from a 4x MLP to a gated variant, or from one attention to grouped-query attention, changes the constant factors and sometimes the exponents. Re-read the failure table whenever a deployment incident is traced to a mechanism not listed in §8, because that table is a catalogue of observed mechanisms and the mechanism missing from it is the one that will surprise the next team.

**Takeaway:** Patch size and resolution are one joint decision with one quadratic consequence; treat a change to either as an architecture change.

## 2. Concept Map / Mental Model

The pipeline has five stages and three feedback arrows. Read it top to bottom; every arrow pointing left is a place where a result becomes invalid without any weight changing.

```text
image [H, W, 3]
  |  resize, normalise (statistics from the pretraining corpus)
  v
patchify  ->  grid (H/P) x (W/P)  ->  flatten each P x P x 3 patch
  |                                        |
  |  P is a hyperparameter that fixes      |  patch embedding: linear map
  |  both resolution and cost              |  P*P*3 -> D
  v                                        v
token sequence [N, D]  +  [CLS]?  +  position information
  |
  v
L transformer blocks:  LN -> attention -> residual -> LN -> MLP -> residual
  |                          |                        |
  |                          |                        +-- 4x expansion, GELU
  |                          +-- Q,K,V projections; softmax over N positions
  v
head: CLS token -> linear -> logits        (classification)
      reshaped tokens -> conv decoder      (dense prediction)
```

**Invariant 1 — token count closure.** N = (H/P)·(W/P) for a patch-aligned grid, plus one for a class token if one is used, plus register tokens if the design includes them. A token count that does not close against the stated grid means either the resize policy or the patch size is not what the design says.

**Invariant 2 — width closure.** Every block preserves [N, D]. Attention, MLP, residuals and normalisation all operate at the same N and D, so the shape is invariant through depth. This is why a transformer's memory profile is flat in depth and quadratic only in attention.

**Invariant 3 — position is not free.** Patch embeddings are permutation-equivariant: permuting tokens permutes outputs identically. Position information is the only thing that breaks that symmetry, so removing or misapplying it leaves a model that is invariant to image layout and therefore unable to distinguish a face from a scrambled face.

**Invariant 4 — the quadratic term is in N, not in D.** Attention cost is 2N²D per block; the dense projections are 12ND². Doubling width quadruples the linear term and leaves the quadratic term linear in D; halving patch size squares N and therefore multiplies the quadratic term by sixteen.

**Invariant 5 — score-matrix bytes scale with batch, heads, N² and precision.** B·h·N²·bytes. At N = 1,024, h = 12 and fp16 this is 25 MiB per layer; at N = 4,097 it is 384 MiB per layer. Nothing about the parameter count predicts these numbers.

**Invariant 6 — resolution change is a position-contract change.** Interpolating a learned position table is a defined operation with a defined result; failing to interpolate is a shape error at best and a silent misalignment at worst. Either way the model has changed, so its validation must be redone.

**Where the convolutional comparison enters.** A convolution mixes a fixed local neighbourhood with weights independent of content. Attention mixes all positions with weights computed from content. The transformer therefore has a weaker locality prior, a stronger global-mixing capability and a cost that grows with area squared. Every practical variant in §4.3 is an attempt to keep some of the locality prior or to bound the mixing set.

**The single diagram to draw first.** Draw the token ladder: N, D and the score-matrix bytes at each stage, in one column. Ninety percent of transformer sizing errors are visible in that column, because the quadratic term is the only term that can change by a factor of sixteen on a single hyperparameter edit.

```text
token ladder template (classification, P = 16)
stage        H x W     N      D     per-block MAC      scores per head
patch grid   224       197    768   1.454e9            197*197 = 38,809
block 1..12  224       197    768   1.454e9 x 12        same
head         -         -      1000  768*1000            -
```

**Takeaway:** Keep two ledgers — token count and score bytes — and recompute both whenever P, H, W, h or precision changes.

## 3. Foundational Definitions

For each term the “not the same as” column exists because the neighbouring concept is what a reviewer will actually be thinking of when they disagree with you.

| Symbol | Meaning | Units / shape | Valid range |
|---|---|---|---|
| H, W | input height and width | pixels | positive integers |
| P | patch side | pixels | divides H and W, or a resize policy is stated |
| N | token count | count | ≥1 |
| D | model width | channels | ≥1 |
| L | block depth | count | ≥1 |
| h | attention heads | count | divides D |
| d_k | per-head key dimension | channels | D/h |
| MAC | multiply-accumulate | operations | ≥0 |
| B | batch size | images | ≥1 |
| σ | peak score-matrix bytes | bytes | ≥0 |

**Patch.** A P x P spatial crop of the input, treated as one vector. It is *not* a receptive field: one patch is the smallest unit the model can distinguish, so a structure smaller than P is invisible to the first block regardless of capacity.

**Patch embedding.** The linear map from a flattened patch of size P·P·C to width D, usually implemented as a strided convolution with kernel and stride both equal to P. It is *not* a learned convolutional feature extractor with spatial structure; at kernel equal to stride it has no overlap and produces a partition rather than a filtering.

**Token.** One vector of length D corresponding to one patch, or to a special token such as the class token or a register. It is *not* a pixel, and it is *not* independent of its neighbours, because the first attention layer mixes across all tokens immediately.

**Class token.** An extra learned vector prepended to the sequence whose final state is used for classification. It is *not* the only way to classify: global average pooling over patch tokens is an alternative with no extra parameter and often similar behaviour. The choice is part of the tuple because it changes the head and the interpretation.

**Register token.** An extra learned token added to absorb global computation that would otherwise be stored in patch tokens. It is *not* a class token; it carries no label and is discarded at the head.

**Absolute position embedding.** A learned vector per grid position, added to the token before the first block. It is *not* translation-equivariant: shifting the content by one patch changes which position vector is added, so the model sees a different input.

**Relative position bias.** A learned or fixed bias added to attention logits as a function of the offset between query and key positions. It is *not* an absolute embedding, and it transfers across resolutions more gracefully because it depends on offsets rather than on absolute coordinates.

**Rotary position embedding.** A rotation of query and key vectors by an angle proportional to position, so that the dot product depends on relative offset. It is *not* a bias term, because it acts on the vectors before the product.

**Window attention.** Attention restricted to a partition of the grid into M x M windows, so each query sees only its window. It is *not* sparse attention in the sense of a fixed random pattern; the restriction is structured and tile-aligned.

**Shifted window attention.** Alternate blocks use a partition offset by half a window, so information crosses window boundaries every other block. It is *not* global attention: crossing a window boundary takes one block, and crossing the whole grid takes logarithmically many.

**Hierarchical token merging.** Stages that reduce the token count by merging or pooling, producing a pyramid analogous to a convolutional ladder. It is *not* token pruning at inference; merging is part of the architecture and is trained end to end.

**Inductive bias.** The set of solutions a model prefers before seeing data. A transformer's bias is weak: it prefers almost nothing about locality, which is why it needs more data to reach the same result as a convolutional model on small datasets. It is *not* a claim that transformers cannot learn locality — they learn it, at a data cost.

**Attention sink.** A phenomenon in which some tokens attract a large share of attention mass in every layer, acting as a no-op destination. It is *not* a bug in the softmax; it is a learned routing pattern whose presence changes how much capacity remains for content.

**Layer normalisation (pre-norm).** Normalisation applied to the input of each sub-layer, with the residual path carrying the unnormalised signal. It is *not* post-norm, where the normalisation follows the residual addition; the two differ in gradient scale at depth and cannot be swapped without re-tuning the learning rate.

**MLP expansion ratio.** The multiplier between the block width and the hidden width of the feed-forward sub-layer, conventionally 4. It is *not* a free hyperparameter when comparing parameter counts: at ratio r the block costs (4 + 2r)ND² MACs, so r enters the arithmetic linearly and is part of the tuple.

**GELU.** A smooth activation whose value is close to x·Φ(x). It is *not* ReLU with a different name; the smoothness changes the gradient near zero and is part of the definition of the block whose cost this file computes.

**Drop path.** Random skipping of a whole residual branch during training, scaled by the survival probability. It is *not* dropout on activations, and it changes the effective depth seen by the optimiser rather than the width.

**Masking.** Additive constants of large negative magnitude added to attention logits to forbid certain query-key pairs. It is *not* a multiplicative zero: the softmax of minus infinity is zero, but the arithmetic still runs unless the kernel skips the pair, which is why masked padding costs time.

**Patch merging.** A learned combination of neighbouring tokens that reduces the token count and increases the width, usually by concatenation followed by a linear map. It is *not* average pooling, because the combination weights are learned and the concatenation preserves within-window structure before mixing.

**Token pruning.** Removing tokens judged unimportant at inference. It is *not* the same as token merging, which combines rather than discards, and the distinction matters because discarding is irreversible while merging is not.

**Distillation.** Training a student to match a teacher's soft outputs. It is *not* a compression guarantee: the student's capacity and the teacher's confidence determine whether the transferred function is faithful on the inputs that matter.

**Takeaway:** Most disagreements here are about whether a term names a mechanism, a shape or a bias; pin all three before arguing about a number.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**Result 1 — token count.**

```text
FORMULA      N = (H/P) * (W/P) + c
WHERE        H, W input sides (pixels); P patch side; c = 1 if a class
             token is prepended, 0 otherwise; add register tokens if used
CONSTRAINT   H and W must be divisible by P, or a resize policy must be
             applied first; a ceiling on tokens implies a ceiling on H*W
NUMERICAL    H = W = 224, P = 16, c = 1
             N = 14 * 14 + 1 = 196 + 1 = 197
INTERPRETATION the grid is 14 x 14; a 2-pixel defect occupies 1/64 of a
             token's area, so it is below the resolution the first block
             can act on
```

The patch grid is a partition, not a sliding window: with kernel equal to stride, the patches tile the image exactly and share no pixels. That is why N is exactly the area ratio and not something like (H-P+1)². Any implementation that reports a token count inconsistent with the partition has an overlapping embedding, which changes both the cost and the effective receptive field.

**Result 2 — patch embedding parameters.**

```text
FORMULA      P_embed = P*P*C*D + D          (linear map plus bias)
WHERE        C input channels; D model width; P patch side
CONSTRAINT   implemented as a conv with kernel = stride = P, so the bias
             is present unless a normalisation follows and absorbs it
NUMERICAL    P = 16, C = 3, D = 768
             P_embed = 256*3*768 + 768 = 589,824 + 768 = 590,592
INTERPRETATION the embedding is a tiny fraction of an 86M-parameter model,
             so patch size is chosen for its cost consequences rather than
             for its parameter consequences
```

Doubling the patch side quadruples the embedding parameters and quarters the token count. Halving it does the reverse: at P = 8 the embedding costs 148,224 parameters and the token count becomes 784. The parameter change is negligible in both directions; the token change is decisive.

**Result 3 — per-block arithmetic.**

```text
FORMULA      MAC_linear = 4*N*D*D           (Q, K, V, output projections)
             MAC_attn   = 2*N*N*D           (QK^T and the AV product)
             MAC_mlp    = 8*N*D*D           (two-layer MLP at 4x expansion)
             MAC_block  = 12*N*D*D + 2*N*N*D
WHERE        N tokens, D width; the 4x MLP expansion is a convention, not
             a law; softmax, layer norms, residuals and GELU are excluded
CONSTRAINT   QK^T is [N, N] per head and the AV product is [N, D]; both
             carry the same N*N*D product, hence the factor 2
NUMERICAL    N = 197, D = 768
             D*D = 589,824
             MAC_linear = 4*197*589,824 = 464,781,312
             MAC_attn   = 2*197*197*768 = 59,610,624
             MAC_mlp    = 8*197*589,824 = 929,562,624
             MAC_block  = 1,453,954,560
INTERPRETATION attention is 4.1% of the block at this resolution; the
             dense projections are 96% of it, which is why “attention is
             quadratic” is not an explanation of transformer cost at
             224 input
```

Check the MLP term independently: 8·197 = 1,576; 1,576·589,824 = 929,562,624. And 464,781,312 + 59,610,624 + 929,562,624 = 1,453,954,560. The two routes agree.

**Result 4 — score-matrix memory.**

```text
FORMULA      sigma_layer = B*h*N*N*bytes / (heads share the same N)
WHERE        B batch; h heads; bytes per element (4 fp32, 2 fp16/bf16);
             the division hint is that each head has its own N x N matrix
CONSTRAINT   training stores the matrix for backward, so the peak is per
             layer times the number of layers that are not recomputed;
             inference may materialise one layer at a time
NUMERICAL    B = 8, h = 12, N = 197, fp16
             sigma = 8*12*197*197*2 = 8*12*38,809*2
                   = 96*77,618 = 7,451,328 B = 7.1 MiB per layer
INTERPRETATION small at 224, which is why a base-size model trains on a
             single accelerator; the same expression at N = 4,097 gives
             384 MiB per layer, which is the practical ceiling
```

**Result 5 — parameter count of the whole model.**

```text
FORMULA      P_total = L*(12*D*D) + P*P*C*D + D + N_pos*D + D*K + K
WHERE        L blocks at 12*D² each (attention 4D², MLP 8D²); position
             table N_pos rows; K output classes
CONSTRAINT   excludes the class token and register tokens (D each) and any
             per-layer normalisation parameters (2*D per norm, two norms
             per block); include them for an exact count
NUMERICAL    L = 12, D = 768, P = 16, C = 3, N_pos = 197, K = 1000
             12*D*D = 7,077,888
             blocks = 12 * 7,077,888 = 84,934,656
             embed  = 3*256*768 + 768 = 590,592
             pos    = 197*768 = 151,296
             head   = 768*1000 + 1000 = 769,000
             total  = 84,934,656 + 590,592 + 151,296 + 768 + 769,000
                    = 86,446,312
INTERPRETATION 98% of the parameters are in the blocks, 0.9% in the head
             and 0.7% in the embedding, so architecture edits that touch
             the embedding are capacity-neutral by construction
```

**Validity condition for §4.1.** These results hold for a pre-norm transformer with a 4x MLP, one class token, learned absolute positions, and no registers. They are wrong for a model with a convolutional stem, with merged hierarchical stages, with Mixture-of-Experts MLPs, or with shared block weights. In each of those cases the block count, the token count or both change per stage, and the arithmetic must be summed per stage.

### 4.2 General case

**Result 6 — resolution scaling of the two terms.**

```text
FORMULA      MAC(N) = 12*N*D*D + 2*N*N*D
             ratio to N0: linear term scales as N/N0, quadratic as (N/N0)^2
             N = (H/P)^2, so quadratic term scales as (H/H0)^4
WHERE        H0 the reference input side; all else fixed
CONSTRAINT   holds only if the position table is resized rather than
             retrained; a retrained table is a new model, not a scaled one
NUMERICAL    P = 16, D = 768, H0 = 224 (N0 = 197)
             H = 384 -> N = 577
             per block = 12*577*589,824 + 2*577*577*768
                       = 4,083,941,376 + 511,378,944
                       = 4,595,320,320   (3.16x the 224 value)
             H = 1024 -> N = 4097
             per block = 54,780,495,360   (37.7x)
INTERPRETATION a 1.7x increase in side costs 3.16x; a 4.6x increase costs
             37.7x, so resolution is the most expensive hyperparameter in
             the tuple and must be justified by object size, not by habit
```

**Result 7 — the crossover between the terms.**

```text
FORMULA      quadratic = linear  =>  2*N*N*D = 12*N*D*D  =>  N = 6D
             quadratic = 1/3 of block => N/(6D + N) = 1/3 => N = 3D
WHERE        block definition includes 4 projections and an 8ND² MLP
CONSTRAINT   against the projections alone (4ND²) the crossover is N = 2D;
             the number quoted must name the comparison it belongs to
NUMERICAL    D = 768: N = 2,304 at the one-third mark, N = 4,608 at parity
             at P = 16 those N correspond to input sides of 768 and 1,085
INTERPRETATION at ordinary classification resolutions the quadratic term
             is a minority of the arithmetic; it is the majority only at
             dense-prediction resolutions or very small patches
```

**Result 8 — windowed attention cost and its ratio.**

```text
FORMULA      MAC_win = 2*N*M*M*D
             ratio_global_to_window = N/M^2
WHERE        M window side in tokens; N total tokens; D width
CONSTRAINT   requires N to be divisible by M² for a clean partition, or a
             pad-and-mask scheme whose masked positions still cost arithmetic
NUMERICAL    N = 3,136 (56 x 56), M = 7, D = 96
             MAC_win = 2*3,136*49*96 = 29,503,488
             MAC_global = 2*3,136*3,136*96 = 1,888,223,232
             ratio = 3,136/49 = 64
INTERPRETATION the saving is exactly the number of windows, so the design
             question becomes how much cross-window mixing the task needs
             and how many shifted blocks pay for it
```

**Result 9 — hierarchical token ladder cost.**

```text
FORMULA      per stage: 2*N_s*M*M*D_s for windowed attention
             total = sum over stages s
WHERE        stage token counts N_s = (H/2^s / P_s)^2 with P_s the per-stage
             patch or merge factor; widths D_s typically double per stage
CONSTRAINT   a merge layer itself costs a linear projection whose MACs are
             N_out*D_in*D_out and must be added, not ignored
NUMERICAL    stages (56,28,14,7) with widths (96,192,384,768), M = 7
             s1: 2*3136*49*96  = 29,503,488
             s2: 2*784*49*192  = 14,751,744
             s3: 2*196*49*384  =  7,375,872
             s4: 2*49*49*768   =  3,687,936
             attention total   = 55,319,040 per block-set per stage
INTERPRETATION the per-stage attention cost halves as the token count
             quarters and the width doubles; a balanced pyramid therefore
             has a flat attention budget, the same regularity as a
             balanced convolutional ladder
```

**Result 10 — positional interpolation under a resolution change.**

```text
FORMULA      grid_new = (H_new/P, W_new/P); interpolate the N_pos x D table
             from grid_old to grid_new with bicubic or bilinear resampling
             in the two spatial axes, then flatten back to N_new rows
WHERE        the table is a 2-D grid, not a list, so the interpolation is
             geometric and must respect the axis order
CONSTRAINT   only meaningful for learned absolute positions; relative
             biases need a rule for unseen offsets, rotary needs none
NUMERICAL    old grid 14 x 14 = 196 rows at D = 768; new grid 24 x 24 = 576
             rows; the interpolation is 14->24 on each axis, and the class
             token row is excluded from the resampling
INTERPRETATION the operation is well defined but changes every position
             vector, so the model's behaviour after interpolation is a new
             measurement, not an expected identity
```

**Result 11 — MLP expansion ratio enters the arithmetic linearly.**

```text
FORMULA      MAC_block = (4 + 2r)*N*D*D + 2*N*N*D
WHERE        r hidden-to-width ratio of the feed-forward sub-layer, 4 by
             convention; the first projection is N*D*(r*D) and the second
             is N*(r*D)*D, so 2r*N*D*D together, plus 4N*D*D for Q,K,V,O
CONSTRAINT   some modern blocks use gated feed-forward layers, whose cost
             is 3r*N*D*D because the gate is an extra projection
NUMERICAL    r = 4, N = 197, D = 768
             MAC = 12*197*589,824 + 2*197*197*768 = 1,453,954,560
             r = 2: MAC = 8*197*589,824 + 59,610,624 = 989,173,248
             r = 8: MAC = 20*197*589,824 + 59,610,624 = 2,383,516,800
INTERPRETATION halving the expansion ratio saves 32% of the block, which is
             a larger lever than removing attention entirely — the
             quantitative reason MLP width is the usual efficiency dial
```

**Result 12 — depth sharing and its parameter arithmetic.**

```text
FORMULA      unique parameters = L_unique*12*D*D when weights are shared
             across L total blocks with L/L_unique reuse
WHERE        L_unique distinct block parameter sets; sharing is usually
             uniform in the simple case L_unique = 1
CONSTRAINT   sharing does not change MACs at all; every token still passes
             through L blocks of arithmetic
NUMERICAL    L = 12, D = 768, full: 84,934,656 block parameters
             L_unique = 4: 28,311,552, a 3x storage reduction, MAC unchanged
INTERPRETATION weight sharing trades capacity for storage at constant
             compute, so it is the wrong lever for a memory-bound
             deployment and the right one for a storage-bound edge device
```

**Result 13 — the number of tokens a merge layer removes and the arithmetic it saves.**

```text
FORMULA      N_next = N/4 for a 2x2 merge; saved attention MAC per block
             = 2*D*(N*M*M - N/4*M*M) for windowed attention
WHERE        M window side; the merged stage's width doubles, which is a
             separate parameter change
CONSTRAINT   the merge projection itself costs N_next*(4*D)*(2*D) MACs for
             a concatenation of four D-vectors into 2D; that cost must be
             included before claiming a saving
NUMERICAL    N = 3,136, M = 7, D = 96
             before: 2*3,136*49*96 = 29,503,488
             after:  N = 784, D = 192 -> 2*784*49*192 = 14,751,744
             merge:  784*384*192 = 57,802,752
INTERPRETATION the merge costs about twice what the attention saving
             recovers at this stage, so the case for merging is the
             pyramidal head, not the attention arithmetic
```

That last result is the one most often missed in design reviews: merging is sold as a cost saving and is, at the first stage, a cost increase. Its real justification is that a dense head needs multiple resolutions, and that later stages genuinely are cheaper.

**Result 14 — effective receptive field per token under a shifted window schedule.**

```text
FORMULA      windows spanned after k blocks = 2^(k/2) along each axis,
             so the covered side in tokens is M * 2^(k/2)
CONSTRAINT   holds for an alternating shift with M dividing the grid;
             without shifts the covered side stays M indefinitely
NUMERICAL    M = 7, k = 6: side = 7 * 2^3 = 56 tokens, which equals the
             full 56 x 56 grid of the stage
INTERPRETATION six shifted blocks are enough to span the stage grid, so a
             stage of eight blocks has global reach by its end without any
             block being global
```

**Validity condition for §4.2.** Each result is an arithmetic identity under the stated block and stage definition. None predicts wall-clock time, because at the sizes where the arithmetic terms are near parity the workload is memory-bound and the softmax plus normalisation traffic dominates. A second condition applies to every interpretive sentence: it describes a consequence of the algebra, not a measured property of a trained model, and any sentence that would change with the data belongs in §8 or §10 rather than here.


### 4.3 Variants

**Variant A — convolutional stem.** Replace the first patch embedding with two or three strided convolutions that reduce the side by 4 before tokenisation. At 224 input the token count falls from 197 to about 50, and the quadratic term falls by 16x while the stem's own cost stays linear in area. This is the standard remedy when a pure transformer overfits on a small target set, because it restores a locality prior without giving up global mixing in the body.

**Variant B — windowed attention with shifts.** Partition into M x M windows and alternate the partition offset by (M/2, M/2) every other block. Cross-window information then travels one window per two blocks, so reaching the full grid takes about 2·log₂(H/(P·M)) blocks. At 56 x 56 tokens with M = 7 that is 2·log₂(8) = 6 blocks, which is affordable; the shift costs a roll-and-mask operation that must be implemented without a copy if the latency matters.

**Variant C — hierarchical merging.** Merge 2 x 2 neighbouring tokens between stages and double the width. The token count quarters and the width doubles, so the per-stage MLP cost halves while the attention cost falls by a factor of eight per stage before the width is counted. Dense prediction heads then have a pyramid to work from, which is the reason this variant exists.

**Variant D — token pruning or merging at inference.** Drop tokens whose attention mass is below a threshold, or merge similar tokens, after a chosen block. The saving is real and the risk is that the dropped tokens are exactly the small objects; the discriminating evidence is per-object-size recall before and after pruning, not aggregate accuracy.

**Variant E — linear or kernelised attention.** Replace softmax(QKᵀ)V with a product of feature maps whose cost is O(N·D²) rather than O(N²·D). The saving is large at high N and the cost is an approximation whose error is dataset-dependent; the check is a direct comparison against the quadratic model at the target resolution, not a citation.

**Variant F — multi-query or grouped-query attention.** Share key and value projections across heads, keeping separate queries. The parameter and memory savings are proportional to the sharing factor, and the quadratic arithmetic is unchanged, so this is a memory-traffic optimisation rather than an arithmetic one.

**Variant G — registers.** Add a small number of extra learned tokens. They absorb the attention mass that would otherwise concentrate in patch tokens, which changes the internal routing without changing N materially (a handful of tokens against hundreds).

**Variant H — grouped-query attention with a shared key-value cache.** Keep h query heads but only g key-value heads, so the stored key and value tensors are g/h of the full size. The arithmetic is unchanged because every query still attends to every key; the win is in the memory traffic of the cache, which is why the variant matters for autoregressive decoding and matters little for single-pass image classification. The check is a measured bytes-moved figure per block, not a FLOP count.

**Variant I — patch embedding by convolution with overlap.** Use a stem whose kernel is larger than its stride, so patches share pixels at their borders. This is not the same model as a non-overlapping partition: the token count is the same but each token has already seen a neighbourhood, which partially restores locality. The cost is a slightly larger embedding and a change to the effective receptive field of block one, so the variant belongs to the stem family rather than to the attention family.

**Variant J — cross-shaped or axial attention.** Factor the N x N attention into a row-wise pass and a column-wise pass, each costing 2N·√N·D. The composite sees every position through two hops rather than one, which is a real restriction; the saving at N = 3,136 is a factor of √N = 56 per pass before the two-hop factor. Choose it only when the task's dependencies are axis-aligned or when the two-hop path is demonstrably sufficient.

**Variant K — conditional computation in the MLP.** Route each token to a subset of feed-forward experts. The arithmetic per token falls by the routing factor while the parameter count grows, so the trade is storage for compute. The failure it invites is routing collapse, where a few experts receive most tokens; the discriminating check is the per-expert token histogram over a held-out batch.

**How the variants compose.** A practical high-resolution model is usually three of these at once: a convolutional stem (A), a hierarchical ladder with merging (C), and windowed attention with shifts (B), often with grouped-query attention (H) for the memory traffic. The composition is legitimate provided the cost model is summed per stage and per pattern rather than borrowed from any single row. The error to avoid is quoting a variant's saving as if it were multiplicative with another's: a stem's 4x token reduction and a window's N/M² reduction do not multiply, because the window saving already depends on the reduced N.

**Validity condition for §4.3.** Each variant is a stated cost and bias transformation. None is claimed to improve accuracy, and each requires its own comparison under the tuple in §1, at the resolution it will actually be deployed at. Where two variants are combined, the cost model must be recomputed rather than scaled, because both the token count and the pattern change the exponent of the dominant term.

### 4.4 Limits and counterexamples

**Counterexample 1 — “attention is the expensive part” at classification resolution.** At N = 197 and D = 768 attention is 4.1% of the block; the projections and MLP are the cost. A design review that optimises the attention kernel at this scale optimises the wrong 4%.

**Counterexample 2 — the quadratic term is not the first constraint at high resolution.** At N = 4,097 the arithmetic is 37.7x the 224 value, but the score matrix at 384 MiB per layer per batch element-group is what fails first. Memory binds before arithmetic, so the mitigation order is precision, then windowing, then checkpointing, then more devices.

**Counterexample 3 — token count is not a proxy for information.** Two models with 197 tokens can carry very different information if one uses 16-pixel patches of a 224 image and the other uses 16-pixel patches of a 224 crop that was downsampled from 896. The tuple must state the crop-and-resize policy, not only the final size.

**Counterexample 4 — absolute positions do not transfer by interpolation alone.** Interpolating a 14 x 14 table to 24 x 24 produces a valid tensor and a model whose position statistics no longer match its training distribution. The interpolation is licensed only when a short adaptation on the new resolution follows.

**Counterexample 5 — window attention is not global in one layer.** A model with M = 7 needs several blocks for information to cross the grid. A claim that the model “has global context” must state the number of blocks and the shift pattern, otherwise it describes a property the architecture does not have.

**Counterexample 6 — more heads do not buy more capacity per FLOP.** Heads partition the width, so h heads at width D cost the same as one head at width D and can express h independent attention patterns over D/h-dimensional subspaces. Increasing h at fixed D reduces the rank per head, which can hurt when the needed pattern is high-rank.

**Counterexample 7 — the parameter count is misleading for capacity comparisons against convolutions.** An 86M-parameter transformer and a 25M-parameter convolutional network can have similar MAC counts at 224 input, because the convolutional ladder spends its parameters at low resolution where each parameter is reused many times. Comparing parameter counts across families compares storage, not compute and not capacity.

**Counterexample 8 — patch size below the object scale is not free accuracy.** Halving P quadruples the token count and squares the quadratic term, and if the objects are large the extra tokens carry redundant information while the cost grows sixteenfold in the attention term. The right patch size is set by the smallest discriminative structure, not by the largest available compute budget.

**Counterexample 9 — the MLP expansion ratio is the cheapest lever, and the least examined.** Halving r from 4 to 2 removes 32% of block arithmetic and 33% of block parameters. No published architecture change is reviewed as often as this one is left unexamined. The cost is a narrower nonlinear feature space per token, which matters most when labels are abundant and least when they are scarce; the discriminating check is a paired run at matched epochs, reported by slice rather than in aggregate.

**Counterexample 10 — merging is often a cost increase at the first stage.** The merge projection at N = 3,136 with widths 96 to 192 costs 57.8 MMAC while the attention saving it enables recovers 14.8 MMAC. A design justified by “merging reduces cost” is justified by a claim that is false at the stage where the merge happens. The true justification is the pyramid the head needs, and the cost is paid knowingly.

**Counterexample 11 — a larger batch is not free capacity for a memory-bound model.** At small N the workload is memory-bound, so doubling the batch doubles the memory footprint while leaving utilisation nearly unchanged. The symptom is that doubling the batch halves the step count without halving the wall-clock time, which contradicts the arithmetic expectation and confuses teams who measure in steps rather than in images per second.

**Counterexample 12 — an attention-entropy diagnostic can be misread.** High entropy is expected at initialisation and low entropy is expected after training; a model that keeps high entropy late is not necessarily broken, because a task whose label depends on a global average genuinely wants a broad mixture. The diagnostic identifies a collapse that coincides with a validation plateau, not any particular entropy value.

**Validity condition for §4.** Every formula is an arithmetic identity on a stated block; every counterexample is a boundary at which a commonly quoted summary stops being true. Where a block contains merging, sharing or sparsity, the arithmetic must be summed per stage and per pattern, and no shortcut from this section survives contact with a real implementation. The section as a whole is also valid only for a dense, non-recurrent model: a model with cross-layer token feedback or with a learned halting mechanism changes the depth term and therefore every total computed here.

## 5. Worked Numerical Example

**The task (ILLUSTRATIVE).** A 12-class defect classifier at 224 x 224, three candidate configurations, and a training cluster with 40 GiB of usable accelerator memory per device. The question is which configuration can be trained at batch 64, and which one must change before the first run.

**Step 1 — token ladder for each candidate.**

```text
candidate   P    N (incl. CLS)   D     L    h
A          16    197             768   12   12
B           8    785             768   12   12
C          16    577             768   12   12   (input 384)
```

N for A: 14·14 + 1 = 197. N for B: 28·28 + 1 = 785. N for C: 24·24 + 1 = 577. Note that C's token count (577) is between A's and B's, which is the first sign that resolution and patch size are not interchangeable levers: C has finer patches in input-pixel terms only if its patch is smaller, and here its patch is the same 16 pixels applied to a larger image.

**Step 2 — per-block MACs.**

```text
A: 12*197*589,824 + 2*197*197*768
   = 1,394,343,936 + 59,610,624 = 1,453,954,560
B: linear 12*785*589,824 = 12*463,011,840 = 5,556,142,080
   quad   2*785*785*768 = 2*473,260,800 = 946,521,600
   B block = 6,502,663,680
C: 12*577*589,824 + 2*577*577*768
   = 4,083,941,376 + 511,378,944 = 4,595,320,320
```

Check B's linear term by another route: 12·785 = 9,420; 9,420·589,824 = 5,556,142,080. And A's: 12·197 = 2,364; 2,364·589,824 = 1,394,343,936. Both agree.

**Step 3 — whole-model arithmetic.**

```text
A: 12 * 1,453,954,560 = 17,447,454,720 MAC  (34.9 GFLOP forward)
B: 12 * 6,502,663,680 = 78,031,964,160 MAC  (156.1 GFLOP forward)
C: 12 * 4,595,320,320 = 55,143,843,840 MAC  (110.3 GFLOP forward)
ratios to A: B = 4.47x, C = 3.16x
```

**Step 4 — score-matrix memory at batch 64, fp16.**

```text
formula: sigma = B*h*N*N*2 bytes per layer
A: 64*12*38,809*2 = 64*12*77,618 = 59,610,624 B  = 56.9 MiB
B: 64*12*616,225*2 = 64*12*1,232,450 = 946,521,600 B = 902.7 MiB
C: 64*12*332,929*2 = 64*12*665,858 = 511,378,944 B = 487.7 MiB
```

These are per layer, per training step, for the scores alone. A training step keeps the attention matrix of every layer whose backward pass still needs it, so the naive peak is 12 times the per-layer figure: A = 683 MiB, B = 10.6 GiB, C = 5.7 GiB. Flash-style recomputation removes most of the stored matrix and replaces it with recomputation, which is why the same configurations that are impossible naively become feasible, at roughly a third more attention arithmetic.

**Step 5 — parameter counts.**

```text
A: blocks 12*12*589,824 = 84,934,656 ; embed 590,592 ; pos 197*768 = 151,296
   head 769,000 ; CLS 768 ; total 86,446,312
B: blocks same 84,934,656 ; embed 3*64*768 + 768 = 148,224 ; pos 785*768 = 602,880
   head 769,000 ; total 86,455,528
C: blocks same 84,934,656 ; embed 590,592 ; pos 577*768 = 443,136
   head 769,000 ; total 86,738,152
```

The three totals differ by less than 0.4%, and their MAC counts differ by up to 4.5x. That sentence is the whole lesson of comparing transformer configurations: parameters are nearly constant under resolution and patch changes, while arithmetic is not. Any capacity argument based on parameter count cannot distinguish these three models.

**Step 6 — the decision.** At batch 64 with naive storage, candidate A trains comfortably, C trains only with recomputation, and B is not feasible without either recomputation or a batch reduction to 8. The cheapest change that preserves the token count is precision and recomputation; the cheapest change that preserves the accuracy hypothesis is to keep A and test whether the smaller patch actually helps by comparing A and B at a matched *effective* batch, which for B means 8. Since the comparison must be at matched batch to be interpretable, the decision is to run A first, then B at batch 8 if the small-defect slice fails.

**Independent arithmetic re-check.** Recompute C's block total from the identity 12·N·D² + 2·N²·D with N = 577 and D² = 589,824. The linear term: 577·589,824 = 340,328,448; times 12 = 4,083,941,376. The quadratic: 577² = 332,929; times 768 = 255,689,472; times 2 = 511,378,944. Sum: 4,595,320,320. Then 12 blocks: 4,595,320,320·12 = 55,143,843,840. The earlier figure matches exactly. Recompute B's memory from the identity B·h·N²·2 with B = 64, h = 12, N² = 616,225: 616,225·2 = 1,232,450; times 12 = 14,789,400; times 64 = 946,521,600 bytes. Matches.

**Sensitivity note.** The binding constraint is memory, and it scales linearly in batch, linearly in heads, quadratically in N and linearly in precision bytes. Halving precision from fp32 to fp16 halves the score memory and does not change the arithmetic; halving N by choosing P = 22 (which does not divide 224, so the resize policy changes first) would quarter it. The most sensitive lever is N, which is why patch size, not depth, is the first thing to reconsider when a run does not fit. Accuracy sensitivity is opposite in direction: N is what carries small-object information, so the memory saving is paid for in exactly the currency the task needs. The two sensitivities are therefore opposed, and the decision needs a per-object-size recall curve to price the trade.

**Step 7 — a fourth candidate with a convolutional stem, and what it does to the ledger.** Candidate D replaces the patch embedding with a stride-4 stem of two 3 x 3 convolutions, so the token grid at 224 input becomes 7 x 7 plus the class token, N = 50.

```text
D: N = 50, D = 768
   block = 12*50*589,824 + 2*50*50*768
         = 353,894,400 + 3,840,000 = 357,734,400
   12 blocks = 4,292,812,800 MAC  (8.6 GFLOP forward)
   ratio to A = 0.246
   score memory at B=64, h=12, fp16 = 64*12*2,500*2 = 3,840,000 B = 3.7 MiB per layer
```

The stem itself costs a forward pass over the full 224 x 224 image: two 3 x 3 convolutions at 32 and 64 channels cost 3·9·32·224² ≈ 4.33 MMAC and 9·32·64·112² ≈ 23.1 MMAC, which are negligible against the 4.29 GMAC body. So D achieves a 4.1x arithmetic reduction and a 15x score-memory reduction for a stem cost under 1% of the total. That is the quantitative form of the argument for a convolutional stem, and it is why the variant is the first thing to try when a transformer does not fit or does not generalise.

**Step 8 — the decision, restated with all four candidates.** A is the reference and fits comfortably. B is the accuracy hypothesis at 16x the quadratic cost and 4.5x the arithmetic, feasible only at reduced batch or with recomputation. C is the resolution hypothesis at 3.16x the arithmetic and 8.6x the score term. D is the sample-efficiency hypothesis at a quarter of the arithmetic. The four are not four points on one axis: D changes the prior, B changes the detail, C changes the coverage, and A is the control. The correct experiment design compares each of D, B, C against A at a matched effective batch and the same augmentation, and reports per-object-size recall rather than accuracy, because that is the slice each variant is hypothesised to move.

## 6. Visual / Geometric Intuition

**Patchify as a tiling.** Draw the 224 image as a 14 x 14 grid of 16-pixel tiles. Each tile becomes one row of the token matrix. The tiling is exact and non-overlapping, so the token grid carries 196 samples of the image and nothing else. When a reader pictures “attention over the image”, this is the resolution at which it happens: a 4-pixel defect is a quarter of one tile, and no amount of depth recovers the sub-tile position from a single token.

```text
224 x 224, P = 16  ->  14 x 14 grid
+----+----+----+ ... +----+
| t0 | t1 | t2 |     | t13|
+----+----+----+ ... +----+
| t14| t15| t16|     | t27|
+----+----+----+ ... +----+
   one tile = 16 x 16 x 3 = 768 numbers -> projected to D
```

**The attention matrix as a picture.** Each block produces one N x N matrix per head. At N = 197 that is 38,809 cells, a small image in its own right, and visualising a row shows which tiles the model consults for that tile. At N = 4,097 the matrix is 16.8 million cells per head, which is why the matrix, not the arithmetic, is the object that fails to fit.

**The quadratic term drawn as area.** Plot attention cost as a function of N on linear axes: it is a parabola. Plot the linear term on the same axes: a straight line. The crossing is at N = 6D against a full block. The useful intuition is that the parabola is flat and boring at 197 and becomes vertical around 2,000; the design question is always which side of that knee the deployment sits on.

**Position as a coordinate system painted on the grid.** An absolute position table is a D-dimensional vector per tile, arranged as a second grid. Interpolating it to a larger grid stretches that painted coordinate system; the mapping is smooth but the painted values were learned for a different spacing, so the model sees coordinates it has never seen. This is why a resolution change requires a short adaptation rather than only an interpolation call.

**Why locality must be learned.** A convolution's weight at offset (1,0) is the same for every position and is applied to a neighbour by construction. Attention has no such constraint: at initialisation its weights are near-uniform, and it must discover, from data, that nearby tiles are more informative for most tasks. With few labels the discovery is incomplete, which is the mechanism behind the sample-efficiency gap and the reason a convolutional stem helps on small datasets.

**Hierarchical stages as a pyramid.** Draw four stages with token grids 56, 28, 14 and 7 and widths 96, 192, 384, 768. The area of each stage halves relative to the previous one after accounting for width, which is the visual form of the flat per-stage budget in §4.2 result 9. A dense head then reads the finest stage for small objects and the coarsest for scene context.

```text
stage      grid    tokens   width   relative attention cost
1          56x56    3136      96          29.5 MMAC
2          28x28     784     192          14.8 MMAC
3          14x14     196     384           7.4 MMAC
4           7x7       49     768           3.7 MMAC
```

**Window partition as a quilt.** Window attention cuts the grid into 7 x 7 patches of tiles. Information inside one window mixes freely; information across windows does not move at all in that block. A shift moves the quilt by half a window, so tiles that were separated become neighbours in the next block. Picture information diffusing one window-width per two blocks, which is the geometric content of the logarithmic mixing depth.

**Token merging as pooling with learned weights.** A 2 x 2 merge takes four neighbouring tokens and produces one, so the grid halves and the width doubles. The geometry is identical to a strided convolution's shape map; what differs is that the merge is a dense linear map over concatenated tokens rather than a shared filter.

**The receptive field of one block, drawn honestly.** A single attention layer's receptive field is the whole image, because every query attends to every key. What is not global is the *effective* use of that field: with near-uniform weights at initialisation, a block's output is close to an average and therefore close to blind. Depth converts nominal reach into usable reach, which is why a one-block transformer is not a global model in any useful sense even though its support is global.

```text
block 1   nominal support: all N tokens; effective: nearly uniform mix
block 2..L  effective support grows as the routing sharpens
windowed   nominal support: M*M tokens; effective: same, plus the
           shifted blocks that carry information across boundaries
```

**Why patch boundaries show up as seams.** A structure straddling two patches is split between two tokens, and no later layer can rejoin the halves at full resolution because the boundary information is already compressed into two vectors. The visible symptom is a periodic error pattern aligned with the 16-pixel grid, most obvious in dense prediction. A convolutional stem with overlapping windows is the structural remedy; augmentation with shifts is the statistical one.

**The score matrix as a heat map of routing.** For one head and one image, the matrix's rows are distributions over tiles. A model that has learned to localise will show sparse rows concentrated near the diagonal for texture tasks and sparse rows off the diagonal for tasks where the label depends on a distant region. This is the cheapest interpretability diagnostic available and it needs no gradient computation: one forward pass, one matrix, one picture.

**Resolution as a magnifying glass over a fixed number of tiles.** Holding the patch size fixed and increasing the resolution increases the number of tiles, so the model sees more detail and pays quadratically. Holding the token count fixed and increasing the resolution means increasing the patch size, so the model sees the same number of tiles covering larger areas and loses detail. The two operations are different experiments and produce different results; a report that says only “we trained at higher resolution” has not said which one it did.

```text
fixed P, H: 224 -> 448   N: 197 -> 785     detail up, cost x4 (linear term)
fixed N, H: 224 -> 448   P: 16 -> 32       detail down, cost unchanged
```

**Depth as repeated mixing.** Each block mixes the current token set, so after k blocks a token's representation depends on the k-hop neighbourhood of the original token under the attention routing. In a windowed model with shifts, that neighbourhood grows by about one window per two blocks; in a global model it grows to the whole grid after one block. Depth therefore buys different things in the two designs, and the correct depth is the mixing depth the task needs, not the largest depth that fits.

**Takeaway:** Every transformer sizing question is either “how many tiles” or “how big is the matrix”; draw both and the arithmetic follows.

## 7. Comparison / Selection Table

The rows are attention patterns for the same token budget. The selection procedure is to find the row whose constraint you actually have — resolution, dense prediction, data volume, memory ceiling — rather than the row whose published result is strongest.

| Pattern | Cost per block | Global in one layer | Resolution flexibility | Failure it invites | Choose when |
|---|---|---|---|---|---|
| Global attention | 2N²D | yes | needs position interpolation | OOM at high N; weak locality prior | N is small (≤ ~600) and objects need long-range context |
| Windowed, no shift | 2NM²D | no | tile-aligned | seams at window borders | Rarely alone; use as a component |
| Windowed with shift | 2NM²D plus roll cost | no, but logarithmic depth | tile-aligned, shift needs even M | mis-implemented mask leaking across windows | High-resolution dense prediction |
| Hierarchical + windowed | sum over stages | no, via pyramid | high, natural pyramid | merge layer cost ignored in estimates | Detection and segmentation at high resolution |
| Convolutional stem + global body | conv linear + 2N²D at reduced N | yes at reduced N | high, stem absorbs resolution | stem dominates compute if too wide | Mid-size data where locality and global range both matter |
| Linear/kernelised attention | O(ND²) | yes, approximately | high | approximation error on long-range tasks | Very large N with tolerance for approximation |
| Multi-query / grouped-query | 2N²D unchanged | yes | unchanged | quality loss per head | Memory-traffic bound inference, not arithmetic bound |
| Token pruning at inference | decreasing with depth | yes until pruned | any | drops the smallest objects first | Throughput-critical classification, not dense prediction |

**The four configurations this table usually resolves to.** For classification at 224 with a few thousand labels, a convolutional stem plus global attention at about 200 tokens. For classification at 224 with hundreds of thousands of labels, global attention at 197 tokens with a learned absolute table. For dense prediction at 512 or above, a hierarchical windowed model with shifts. For streaming inference under a strict latency ceiling, a pruned or distilled convolutional student, because the transformer's advantage rarely survives the pruning budget at small scale.

| Decision axis | Question that decides it | Evidence required before committing |
|---|---|---|
| Patch size | What is the smallest discriminative structure in input pixels? | Per-object-size recall at two patch sizes |
| Resolution | What is the largest small object that must be detected? | Slice recall on the smallest object quartile |
| Pattern | Does the label depend on relationships between distant regions? | Ablation with the mixing set restricted |
| Positions | Will the model ever be evaluated off its training grid? | Interpolation run plus short adaptation |
| Stem | Are labels scarce relative to the locality prior? | Probe and fine-tune comparison against the pure model |

**Anti-patterns in this selection.** Comparing a global model at 224 with a windowed model at 512 and attributing the difference to the attention pattern. Choosing a hierarchical model for a classification task whose objects are all large, where the pyramid costs memory for no gain. Choosing a linear-attention variant to solve a memory problem that a batch-size reduction would have solved without changing the function. And choosing a patch size by dividing the input side by a round number rather than by the smallest structure that matters.

**A selection walk (ILLUSTRATIVE).** Suppose 6,000 labelled images, three cameras, 12 classes, objects between 10 and 200 pixels in a 448 x 448 frame, a dense mask required for one class, and 16 GiB of accelerator memory. The object floor of 10 pixels rules out patch 16 at the finest stage if the mask must resolve the boundary, so the first stage needs patch 8 or a convolutional stem with a stride of 2 followed by patch 8. The mask requirement rules out a pure classification model and points at the hierarchical row. The 448 frame at patch 8 gives N = 3,136 at stage one, whose score matrix at batch 16, 8 heads, fp16 is 16·8·3,136²·2 = 2,517,630,976 bytes ≈ 2.3 GiB per layer, so windowing is mandatory rather than optional. The shortlist is therefore: hierarchical with windowed shifted attention at patch 8 in stage one, and the first ablation is window size, not depth.

**What the table does not rank, and why.** It does not rank patterns by accuracy, because no ordering of that kind survives a change of dataset. It does not rank them by cost either, because cost depends on N, which depends on the resolution the task needs, which is a task property. What it ranks is the set of constraints each pattern satisfies and the failure it invites, which is the only ordering a reader can apply before running anything.

**The three questions that resolve most selections in practice.** First, what is the smallest structure the decision depends on, in input pixels? That fixes the patch size and therefore N. Second, does the decision depend on a relationship between regions further apart than one window? That fixes the pattern. Third, will the model be evaluated at a different grid than it was trained on? That fixes the position scheme. A selection that answers these three and states the answers is reviewable; one that cites a published benchmark is not.

| If this is true | Then this follows | Because |
|---|---|---|
| Smallest structure < 8 px | Windowed hierarchical at patch 4–8, or a stem | Global attention at that N will not fit |
| No dense output required | Classification head, global attention | The pyramid costs memory for nothing |
| Objects larger than 100 px | Larger patch, fewer tokens | Detail below patch size is unusable anyway |
| Labelling budget < 5k images | Convolutional stem, frozen positions | The locality prior must come from somewhere |
| Deployment grid differs from training | Pooling head, relative or rotary positions | Both are defined off the trained grid |
| Latency under 5 ms at batch 1 | Distilled student, quantised | Memory-bound at small N; arithmetic is not the limit |

## 8. Failure Modes & Edge Cases

**F-01 — Out of memory on the first training step at a resolution that fits inference.**
Competing causes: score-matrix storage for backward; activation retention across all blocks; a batch size chosen from an inference benchmark; an attention kernel that materialises the full matrix instead of a recomputed one.
Discriminating check: run one forward pass with no gradient and record peak memory, then one forward-plus-backward; if the second is several times the first, storage for backward is the cause. Then set the attention implementation to the recomputing variant and re-measure.
Correction: recomputation, lower precision for scores, gradient accumulation to keep the effective batch, or a smaller token count.
Prevention: a memory budget computed from B·h·N²·bytes per layer before the first run, recorded in the run manifest.

**F-02 — Accuracy collapses after changing the input resolution.**
Competing causes: position table not interpolated; interpolated but not adapted; patch size unchanged while the effective object scale changed; the resize policy switched from crop to squash.
Discriminating check: compare the parameter shapes and the position table's row count against the new token count. If they disagree the code path is broken; if they agree, evaluate recall binned by object size to see whether the scale change is the cause.
Correction: interpolate, then run a short adaptation at the new resolution; restate the resize policy in the tuple.
Prevention: the position contract names the grid and the interpolation rule, and a shape assertion runs at model construction.

**F-03 — Model is accurate on average and blind to small objects.**
Competing causes: patch size larger than the object; token count reduced by pruning or merging; the head reads only the coarsest stage; global pooling averaged away a localised response.
Discriminating check: per-object-size recall quartiles. If the smallest quartile is at chance while larger quartiles are fine, the token grid is the cause; if all quartiles degrade equally, the head is the cause.
Correction: smaller patches at the first stage, a finer pyramid level for the head, or a convolutional stem that preserves resolution before tokenisation.
Prevention: the acceptance gate reports recall by object-size quartile, not accuracy alone.

**F-04 — Training loss decreases but validation is flat from the first epoch.**
Competing causes: position table with far more parameters than the data supports; augmentation too weak for a weak locality prior; learning rate tuned for a convolutional model; a leak-free but very small validation set giving a noisy curve.
Discriminating check: freeze the position table or replace it with a relative bias and re-run. If validation starts moving, the position parameters were the overfitting surface. Separately, plot the validation curve with a bootstrap band to see whether “flat” is inside noise.
Correction: relative or rotary positions, stronger augmentation, a convolutional stem, or a smaller width at the same depth.
Prevention: parameter counts are reported per component so the position table's share is visible before training.

**F-05 — Attention weights are nearly uniform in every layer and every head.**
Competing causes: logits scale missing (no division by √d_k); inputs not normalised as the checkpoint expects; the model undertrained; a masking bug that adds a large constant everywhere.
Discriminating check: print the maximum attention weight per head per layer at step 0 and after convergence. Near-uniform at step 0 is expected; near-uniform after convergence means the model is not using attention, and a mask bug is the first suspect.
Correction: restore the scale factor, verify normalisation statistics against the checkpoint's, and check the mask's sign and dtype.
Prevention: a diagnostic that logs attention entropy per layer in the training run.

**F-06 — Windowed model shows a grid-aligned artefact in its dense predictions.**
Competing causes: shift mask leaking so tokens attend across the shifted partition; window size not dividing the grid, leaving padded regions that pollute the merge; merge layer mis-indexed after the roll.
Discriminating check: feed a constant image and inspect the output for tile-periodic structure. A constant input must produce a constant output; any structure is a mask or indexing bug, not a model property.
Correction: fix the mask's additive constant, pad the grid to a multiple of the window, and unit-test the roll-and-unroll pair for exactness.
Prevention: an invariance test on constant input in continuous integration.

**F-07 — Fine-tuning a pretrained transformer degrades it more than fine-tuning a convolutional checkpoint does.**
Competing causes: the pretraining augmentation differs strongly from the target pipeline; the learning rate is too high for a pre-norm architecture; the head is initialised at a scale that overwhelms the pretrained features; the position table is being updated while the token grid changed.
Discriminating check: probe first, then fine-tune with the position table frozen. If freezing positions recovers most of the gap, the position contract was violated; if not, the learning-rate map is the cause.
Correction: lower learning rate for the body, warm-up, discriminative rates, and a frozen position table when the grid is stable.
Prevention: the adaptation contract records which components are frozen, including positions and normalisation statistics.

**F-08 — Throughput on the target device is far below the arithmetic prediction.**
Competing causes: memory-bound layers at small N; the attention kernel falling back to a slow path for an unusual head dimension; per-block layer-normalisation traffic; a batch of one removing parallelism.
Discriminating check: profile per block and compare measured bytes moved with the model's arithmetic intensity. If the intensity is below the device balance point, the workload is memory-bound and no arithmetic optimisation will help.
Correction: fuse normalisation, choose head dimensions that the kernel supports, raise the batch, or replace the slow pattern with the windowed variant.
Prevention: a batch-1 latency benchmark on the target device before the architecture is frozen.

**F-09 — Two runs of the same configuration differ by several points.**
Competing causes: nondeterministic attention kernels; data order differing because the shuffling seed is set after the loader is constructed; augmentation randomness not seeded; dropout differing between the two runs.
Discriminating check: fix all seeds and disable nondeterministic kernels; if the gap persists, diff the loader configuration and the augmentation parameters. A gap that disappears was inside kernel-level noise.
Correction: report three seeds and compare intervals; never compare single runs.
Prevention: a seeded reference run archived with its exact environment.

**F-10 — The model improves on the benchmark and regresses in production.**
Competing causes: the benchmark resolution differs from the production resolution; the position contract was adapted on the benchmark grid; preprocessing in production uses a different interpolation kernel; the production crop policy differs.
Discriminating check: hash the model input tensor at benchmark and production paths for one image. Identical bytes rule out the pipeline; different bytes localise it to preprocessing.
Correction: version the preprocessing and the grid together, and re-validate at production resolution.
Prevention: the tuple includes the resize kernel and the crop policy, and a golden-image test asserts them.

**Edge case — grid not divisible by the window size.** A 56 x 56 token grid with M = 7 divides exactly; a 60 x 60 grid does not. Padding to 63 x 63 adds masked tokens whose masked scores still cost arithmetic unless the kernel skips them, so the saving is smaller than the ratio suggests.

**Edge case — head dimension not a multiple favoured by the kernel.** With D = 768 and h = 12 the head dimension is 64, which most kernels handle well; with D = 768 and h = 16 it is 48, which some kernels handle poorly. The arithmetic is identical and the latency is not.

**Edge case — a single-token batch after pruning.** Pruning can reduce a sparse image to a handful of tokens, and a model trained at N = 197 may behave unpredictably at N = 12. Guard the minimum token count or disable pruning below a floor.

**Edge case — class token versus pooling under a resolution change.** A class token learns to aggregate over the training grid's token statistics; global average pooling averages over whatever tokens exist. Under a grid change the pooled head is usually more stable, which is a cheap reason to prefer it when the deployment grid may vary.

**Edge case — mixed precision and the softmax.** Scores computed in half precision can overflow before the max-subtraction; the stable implementation subtracts the row maximum in the same precision and accumulates the exponential sum in a wider type. A model that trains in half precision and evaluates in full can behave differently at the tail, so evaluate in the deployment precision.

**F-11 — The model's attention entropy collapses in the first few layers and stays collapsed.**
Competing causes: a learning rate high enough to sharpen the routing before the representations are informative; layer-normalisation parameters drifting to a large scale; a pretrained checkpoint whose routing was already sharp and a target distribution that no longer matches.
Discriminating check: log the per-layer attention entropy each epoch. A monotone collapse from high entropy at step 0 is the normal pattern; a collapse in the first few hundred steps that never recovers, followed by flat validation, indicates the routing froze before features formed.
Correction: lower the learning rate for the attention projections, add or extend warm-up, and freeze the normalisation scale parameters for the first epoch.
Prevention: entropy-per-layer is part of the training diagnostic set, plotted alongside loss.

**F-12 — A distilled student matches the teacher on the benchmark and fails on the tail.**
Competing causes: the distillation temperature was too low, so the student learned the teacher's hard decisions and not its uncertainty; the teacher's confident errors dominate the high-temperature regime; the student's capacity is insufficient in the region where the teacher was uncertain.
Discriminating check: compare student and teacher agreement as a function of the teacher's confidence. If agreement is high where the teacher is confident and low where it is uncertain, distillation transferred the wrong part of the distribution.
Correction: raise the temperature, weight the distillation loss by the teacher's entropy, or keep the teacher's hard labels as a floor on the confidence band.
Prevention: the distillation gate reports agreement by teacher-confidence bin, not overall agreement.

**F-13 — Positional parameters dominate the trainable set after a resolution change.**
Competing causes: a learned absolute table interpolated to a much larger grid contributes N_new·D parameters that are all trainable; the table's learning rate was inherited from the body; the head was re-initialised while the table was not.
Discriminating check: print the trainable parameter count per component before and after the change. A jump of N·D in the position component that exceeds the head's own count means the table is the dominant surface.
Correction: freeze the interpolated table for the first phase, or switch to a relative scheme with far fewer parameters.
Prevention: the position contract states the parameter count and the frozen status of the table.

**F-14 — The same checkpoint performs differently in two services that load it identically.**
Competing causes: input normalisation statistics applied in one service and not the other; channel order differing between decoding libraries; the resize kernel differing; a mixed-precision path enabled in one service only.
Discriminating check: log the first block's input tensor statistics — mean, variance and per-channel extrema — at both services for the same image. A mismatch localises the pipeline before any model question is asked.
Correction: version preprocessing as an artefact, and assert tensor equality across paths in continuous integration.
Prevention: a golden-image test that fails on any byte difference in the model input.

**F-15 — A patch-size change breaks a downstream dense head.**
Competing causes: the head assumes a fixed token grid and reshapes without checking; the pyramid level it reads has a different stride; the merge layer count assumed a particular divisibility.
Discriminating check: instrument the head's input shape and compare with its assumed shape. The error is a shape assertion away and is usually hidden by a reshape that silently reorders elements.
Correction: derive the head's grid from the actual token count, and assert divisibility at construction.
Prevention: the token ledger is a test fixture, and each stage asserts its expected shape at model build.

## 9. Complexity & Cost Analysis

**Four cost axes.** Token count, arithmetic, score-matrix memory and parameter storage. They respond to different levers and bind in a fixed order as resolution grows: memory first, then arithmetic, then communication.

| Axis | Scaling | Binds | Lever |
|---|---|---|---|
| Token count | (H/P)² | Everything downstream | Patch size, stem downsampling |
| Block arithmetic | 12ND² + 2N²D | Compute-bound latency | Width, depth, pattern |
| Score memory | B·h·N²·bytes | Training feasibility | Precision, recomputation, windowing |
| Parameter storage | L·12D² | Model download, memory residency | Width, depth, sharing |

**Where the budget goes at 224.** For candidate A the forward pass is 17.45 GMAC, i.e. 34.9 GFLOP. On a stated 78 TFLOP/s effective device that is 0.447 ms of pure arithmetic, giving a nominal 2,236 images/s. A measured 400 images/s implies 13.96 TFLOP/s of sustained throughput, 17.9% of peak, which is a normal efficiency for a small model with limited parallelism. Training costs roughly three times the forward arithmetic per step, so 64 images per step is 64·3·34.9e9 = 6.70e12 FLOP, or 0.086 s per step at 78 TFLOP/s, i.e. about 11.6 steps/s and 745 images/s.

**The memory calculation that decides feasibility.** At batch 64 with naive score storage the per-layer figure is 56.9 MiB and the twelve-layer peak is 683 MiB, which is small; activations of the same order again are 1.3 GiB; optimiser state for 86.4M parameters at Adam is three copies of 4 bytes = 1.0 GiB. The total lands near 4 GiB, which fits comfortably. At 384 input the same calculation gives 5.7 GiB of scores, 3.2 GiB of activations and the same optimiser state, near 9 GiB, still feasible; at 1,024 input the score term alone is 47 GiB across layers and the run is impossible without windowing or recomputation. The lesson is that the feasible resolution is set by the score term's N² and not by the parameter count, which barely moved.

**Recomputation economics.** Storing no attention matrices and recomputing them in the backward pass costs one extra forward pass over the attention sub-layer, which is about a third of the attention arithmetic and therefore a small fraction of the block. It reduces the score-matrix peak from twelve times the per-layer figure to one, a factor of twelve, at roughly 2–3% extra total arithmetic. That trade is almost always correct, which is why the naive measurement is not the one to quote in a capacity plan.

**Annotation and pretraining economics (ILLUSTRATIVE).** A transformer trained from scratch on 12,000 labelled images will underperform a pretrained checkpoint, because the pretraining corpus supplies the locality and texture priors that the labels cannot. If a permissively licensed checkpoint exists, the effective label requirement falls by one to two orders of magnitude on typical vision benchmarks, which is the whole economic argument for the checkpoint and the reason the licence check is a first-class step rather than a legal afterthought.

**Cost of the wrong pattern.** Choosing global attention for a 1,024-input dense task multiplies the score memory by (1,024/224)⁴ ≈ 436 relative to the 224 classification case, and the recovery is a re-architecture plus re-validation. Choosing window attention for a task whose labels depend on two distant regions costs accuracy that no amount of training recovers, because the information path does not exist. Both errors are cheap to avoid with one ablation and expensive to discover after a release.

**Energy and edge deployment.** Energy per inference tracks bytes moved at least as closely as FLOPs. At small N the transformer is memory-bound and its energy is dominated by weight streaming, which favours smaller widths and quantisation; at large N the score matrix dominates and windows or pruning are the only real levers. A 4x weight quantisation reduces streaming energy proportionally, which is why edge deployments of these models are usually quantised before they are pruned.

**A worked throughput comparison across the three candidates.** At 78 TFLOP/s effective, the forward arithmetic per image is 34.9 GFLOP for A, 156.1 GFLOP for B and 110.3 GFLOP for C, giving arithmetic lower bounds of 0.447 ms, 2.00 ms and 1.41 ms respectively. Measured throughput will be several times worse at batch 1 because the score matrices at these sizes are small enough that kernel launch and normalisation traffic dominate. The practical consequence is that none of the three is disqualified by arithmetic on a single device; the choice is made by memory and by accuracy evidence, not by this table.

```text
candidate  forward GFLOP  arithmetic lower bound  ratio to A
A          34.9           0.447 ms                1.00
B         156.1           2.00 ms                 4.48
C         110.3           1.41 ms                 3.16
```

**Training-step time and its components.** A step at batch 64 costs about three times the forward arithmetic, so A is 6.70 TFLOP per step and 0.086 s at the stated throughput, i.e. 11.6 steps/s. B is 4.5x that at 0.386 s per step, 2.6 steps/s. Over 30 epochs at 12,000 images the totals are 16,900 steps for A (24 minutes) and the same step count for B (1.8 hours). The gap is large in ratio and small in absolute terms, which is the usual situation: the binding cost of a transformer experiment is rarely the GPU hours and almost always the evaluation and review time.

**Memory headroom as the real constraint on parallelism.** With 40 GiB per device and a 4 GiB requirement at 224 batch 64, the headroom permits eight-way model replicas per device only in principle; in practice the data-parallel factor is set by the device count, not by memory. At 384 the requirement rises to about 9 GiB and the same batch still fits. At 1,024 the score term alone exceeds the device, and the run requires windowing before it requires more hardware. This ordering — reduce the exponent, then add devices — is the one that survives a budget review.

**Cost of an evaluation grid change.** If the deployment resolution differs from the validation resolution, the evaluation must be repeated at the deployment grid, and any threshold or calibration fitted at the old grid is void. The cost is one full validation pass plus the recall-by-size analysis plus a review cycle. It is small compared with the cost of a release at an unvalidated grid, which is the failure mode F-10 describes.

**Takeaway:** Compute the score term first, because it is the term that turns a plausible configuration into an impossible one.

## 10. Key Theorems / Results Reference

Each entry states the result, its scope and the way it is commonly overstated. No accuracy claim about any dataset appears here.

**R1 — Self-attention is permutation-equivariant without position information.** Permuting the token sequence permutes the outputs identically. Scope: exact, for any block whose operations are token-wise plus attention. Overstatement: “positional encoding is a detail.” It is the only mechanism that breaks the symmetry, so its absence makes the model unable to distinguish layouts.

**R2 — Attention cost is quadratic in token count and linear in width.** Derived in §4.1: 2N²D for the two products. Scope: a dense attention matrix without sparsity or kernelisation. Overstatement: “transformers are quadratic and therefore impractical for images.” At 197 tokens the quadratic term is 4.1% of the block, so the claim is false at classification resolution and true at dense-prediction resolution.

**R3 — The crossover between attention and the dense projections is at N = 2D, and against the full block at N = 6D.** Derived in §4.2 result 7. Scope: a block with four width-D projections and an 8ND² MLP. Overstatement: quoting one number without naming the comparison; the two answers differ by a factor of three.

**R4 — Token count scales as the inverse square of patch size.** N = (H/P)². Scope: exact for a partition embedding. Overstatement: “smaller patches always help.” They help when the discriminative structure is smaller than the patch, and they cost sixteenfold more attention arithmetic.

**R5 — Score-matrix memory scales as B·h·N²·bytes.** Scope: exact for a materialised matrix; reduced by recomputation or kernel fusion without changing the arithmetic. Overstatement: “memory scales with parameters.” It does not; the parameter count is nearly constant across the three configurations in §5 while the score memory varies by 16x.

**R6 — Windowed attention reduces the quadratic term by exactly N/M².** Derived in §4.2 result 8. Scope: a partition with no overlaps and no padding waste. Overstatement: “windowed attention is as good as global.” It restricts the mixing set, and the loss is task-dependent.

**R7 — Shifted windows give logarithmic mixing depth.** Information crosses one window per shifted block, so spanning a grid of W windows takes about 2·log₂(W) blocks. Scope: an alternating shift pattern; a different pattern gives a different depth. Overstatement: “shifting makes it global.” Shifting makes the path finite and short, not one layer.

**R8 — Patch embedding is a convolution with kernel equal to stride.** Scope: exact when the projection is implemented as a strided convolution with no overlap and no padding. Overstatement: “a ViT has no convolutions.” The embedding is one, and many practical models add more in the stem.

**R9 — Interpolating a learned position table preserves the tensor shape but not the learned statistics.** Scope: exact as a statement about shapes; the behavioural claim is empirical. Overstatement: “interpolating positions makes the model work at a new resolution.” It makes it run; the model must be adapted and re-evaluated.

**R10 — Weak inductive bias trades sample efficiency for ceiling.** A model with a weaker locality prior needs more data to reach a given error but is not bounded by the prior's assumptions. Scope: empirical regularity across vision benchmarks. Overstatement: “transformers need more data, full stop.” They need more data at small scale; at very large scale the prior stops binding and the ceiling matters more.

**R11 — Multi-head attention partitions width, not compute.** Total quadratic arithmetic is unchanged by h at fixed D. Scope: exact. Overstatement: “more heads means more capacity per FLOP.” Each head operates at lower rank, so the effect on expressiveness is a matter of inductive bias, not of cost.

**R12 — Registers absorb attention mass that would otherwise concentrate in patch tokens.** Scope: empirical, architecture-dependent, and observed in models with many layers. Overstatement: “registers fix artefacts.” They change routing; whether the artefact disappears depends on the task and the layer count.

**R13 — Hierarchical merging gives a flat per-stage attention budget when width doubles as tokens quarter.** Derived in §4.2 result 9. Scope: a pyramid with the stated ratio; other ratios give non-flat budgets. Overstatement: “hierarchical models are cheaper.” They are cheaper in the attention term and more expensive in the merge and head terms.

**R14 — Linear attention replaces the N² term with an ND² term at the cost of an approximation.** Scope: exact for the kernelised form; the approximation error relative to softmax attention is empirical. Overstatement: “linear attention is a free speedup.” It is a different function class, and the difference shows up on tasks that need sharp, sparse routing.

**How to cite these.** Cite the result with its scope. “Attention is quadratic in token count, which is 4.1% of the block at 197 tokens and the majority at 4,097” is usable in a design review. “Transformers are quadratic, so use windows” is not, because it neither quantifies the term nor states the cost of restricting the mixing set.

**R15 — Pre-norm and post-norm differ in gradient scale at depth.** Placing the normalisation inside the residual branch keeps the identity path unnormalised, so gradients reach early layers without passing through a sequence of normalisation Jacobians. Scope: an architectural statement about gradient flow, not a claim of monotone superiority. Overstatement: “pre-norm always trains better.” It trains more stably at depth without warm-up fiddling; post-norm can match it with careful initialisation and warm-up, and the two are not interchangeable without re-tuning.

**R16 — A convolutional stem changes the token count and the priors together.** With a stride-4 stem at 224 input, N falls from 197 to about 50, so the quadratic term falls by 16x, and the first tokens already encode a neighbourhood rather than a disjoint patch. Scope: exact for the cost claim; the prior claim is empirical. Overstatement: “a stem makes it a CNN.” It restores some locality in the first few layers and leaves the body fully attentional.

**R17 — The class token and global pooling are near-equivalent at a fixed grid and diverge off it.** At the training grid both aggregate the same token set, and reports commonly find them comparable. Off the grid, pooling averages over whatever tokens exist while the class token's learned state was tuned for a specific count. Scope: empirical at fixed grid, structural off it. Overstatement: “they are interchangeable.” They are interchangeable only while the grid is frozen, which is exactly the condition a resolution change breaks.

**R18 — Parameter count is invariant to token count, and arithmetic is not.** For a fixed width and depth, P_total changes by the embedding and position terms, which are under 1% of the total, while MACs change linearly and quadratically with N. Scope: exact for the stated model class. Overstatement: “bigger models have more parameters, so more resolution costs capacity.” It costs compute and memory, and no capacity at all.

**R19 — Masked positions still cost arithmetic unless the kernel skips them.** A padded grid's masked tokens are excluded from the result by the softmax but are still multiplied in a dense kernel. Scope: implementation-dependent. Overstatement: “padding to a multiple of the window is free.” It is free in accuracy and not free in time, and the realised saving of windowing is therefore below N/M² whenever padding is involved.

**R20 — The feasible resolution is set by the score term, not the parameter term.** At fixed width and depth the parameter term is flat while the score term grows with N². Scope: exact given the block definition, and the reason the three configurations in §5 have nearly equal parameter counts and 4.5x different costs. Overstatement: “the model is too big.” The model is the same size in all three cases; the token grid is what changed.

## 11. Interview / Exam Questions

**Q1. An image of 384 x 384 with patch 16 and a class token: how many tokens, and how many per-block MACs at width 768?**
Answer: N = (384/16)² + 1 = 24² + 1 = 576 + 1 = 577. Per-block MACs = 12·577·589,824 + 2·577²·768 = 4,083,941,376 + 511,378,944 = 4,595,320,320.
Why asked: the two-term structure is the whole cost model and candidates often remember only one term.
Follow-up: what fraction is the quadratic term? 511,378,944/4,595,320,320 = 11.1%, which is above the 4.1% at N = 197 but still a minority.

**Q2. Your model at 224 input uses 34.9 GFLOP forward and achieves 400 images/s on a device with 78 TFLOP/s peak. Interpret the utilisation.**
Answer: 400 images/s requires 400·34.9e9 = 1.396e13 FLOP/s, which is 17.9% of peak. At that utilisation the model is not arithmetic-bound; memory traffic, kernel launch and the small matrix sizes in the projections are the likely limits, so the next step is a per-block profile rather than a bigger model.
Why asked: turns a throughput number into a utilisation figure, which is the only form in which it means anything.
Follow-up: what would raise utilisation without changing the model? Larger batches, fused normalisation, and a kernel whose head dimension is hardware-friendly.

**Q3. Derive the token count at which attention is one third of a block at width 768, and say what the number is against the projections alone.**
Answer: the block is 12ND² + 2N²D, so the attention fraction is N/(6D + N). Setting it to 1/3 gives N = 3D = 2,304. Against the projections alone the fraction is N/(2D + N), and one third gives N = D = 768.
Why asked: two different “crossovers” circulate, and the candidate must know which block definition produces which.
Follow-up: at patch 16, what input side is N = 2,304? (H/16)² ≈ 2,304 means H ≈ 768, so about 768 pixels.

**Q4. A run OOMs at 384 input with batch 64 although the 224 run at the same batch fits. Name the term that grew and by how much.**
Answer: the score-matrix term grows as N², and N grew from 197 to 577, so it grew by (577/197)² = 8.58x. The parameter term did not grow at all; it actually changed by a few tenths of a percent because the position table grew. At 64 batch, 12 heads, fp16, the per-layer figure went from 56.9 MiB to 487.7 MiB.
Why asked: separates the term that binds from the terms that do not.
Follow-up: what is the cheapest fix that preserves the hypothesis? Recompute attention in the backward pass, which cuts the stored score term by the layer count for a few percent extra arithmetic.

**Q5. Explain why a transformer needs more labels than a convolutional model of the same arithmetic on a small dataset.**
Answer: the convolutional model's weights encode translation equivariance and locality by construction, so its hypothesis space excludes most functions that treat neighbouring pixels as unrelated. The transformer's attention starts near-uniform and must learn locality from data. With few labels, the data is not enough to discover a prior the convolution gets for free, so the transformer's error is higher at that scale.
Why asked: tests whether the candidate understands inductive bias as a data cost rather than as a slogan.
Follow-up: what are two ways to pay less? A convolutional stem, which reinstates locality before tokenisation, and stronger augmentation, which multiplies the effective data.

**Q6. You change the input from 224 to 320 with patch 16 and patch the position table by interpolation. What must you re-measure, and why is interpolation not enough?**
Answer: N goes from 197 to (20·20)+1 = 401, so the grid changes from 14 x 14 to 20 x 20 and every position vector is resampled. Interpolation preserves the tensor shape but the painted coordinate system was learned at 16-pixel spacing and is now being evaluated at positions it never saw. Re-measure per-object-size recall and calibration at the new resolution, and run a short adaptation on the new grid before trusting the numbers.
Why asked: the mistake is to treat interpolation as a no-op configuration change.
Follow-up: which position scheme avoids most of this? Relative biases or rotary positions, which are defined on offsets rather than absolute coordinates.

**Q7. A windowed model with M = 7 at a 56 x 56 token grid: give the attention MAC saving against global, and the number of shifted blocks needed to span the grid.**
Answer: the ratio is N/M² = 3,136/49 = 64, so attention drops from 1.888 GMAC to 29.5 MMAC per block. The grid has 8 x 8 = 64 windows, so spanning it takes about 2·log₂(8) = 6 alternating blocks, which is affordable inside a stage of 8–12 blocks.
Why asked: the saving is trivial to compute; the mixing depth is the part that decides feasibility.
Follow-up: what breaks if M does not divide the grid? Padding adds masked tokens that still cost arithmetic in most kernels, so the realised saving is below the nominal ratio.

**Q8. Why does a per-block memory model need the batch, head count and precision, when the arithmetic model needs neither head count nor precision?**
Answer: the arithmetic model counts multiply-accumulates, and splitting the width into heads does not change the total product count; precision does not change the count either. The memory model counts stored elements: B·h·N² of them per layer, each of `bytes` size. So the two models have different inputs, and confusing them produces either a wrong memory estimate or a wrong FLOP estimate.
Why asked: the two models are usually conflated in review.
Follow-up: which of the two changes if you switch to grouped-query attention? The memory model, because keys and values are shared and fewer elements are stored per head.

**Q9. A constant image produces a tile-periodic pattern in a segmentation output. Diagnose it.**
Answer: a constant input must produce a constant output under any correct implementation, so the pattern is a bug: either the shift mask leaks attention across partitions, or the roll and unroll are not exact inverses, or the merge layer is mis-indexed after the shift. The discriminating check is a unit test asserting constant-in, constant-out before any training.
Why asked: distinguishes a debugging instinct from a modelling instinct; the answer is never “retrain with more augmentation”.
Follow-up: how would you localise which block introduces it? Feed the constant through the model block by block and find the first block whose output is no longer constant.

**Q10. Parameters barely change when you move from 224 to 384 with the same width and depth, but wall-clock training time more than triples. Explain both facts.**
Answer: parameters live in the block matrices (12D² per block) plus a small embedding and position table, so they depend on D and L and only marginally on N. Wall-clock depends on arithmetic, which is 12ND² + 2N²D, and N went from 197 to 577, a factor of 2.93 in the linear term and 8.58 in the quadratic, giving 3.16x overall. The two facts are consistent because capacity and cost are governed by different variables.
Why asked: the parameter count is the number people quote and the number that explains the least.
Follow-up: which configuration would keep parameters fixed and increase arithmetic most? The smallest patch at the largest resolution, because N enters quadratically.

**Q11. Your fine-tune of a pretrained transformer is worse than a linear probe. Give three candidate causes and one check that separates them.**
Answer: a learning rate too high for a pre-norm body; the position table or normalisation statistics being updated when the grid and domain changed; a head initialised at a scale that disrupts the pretrained features. The separating check is to freeze everything except the head for the first epoch and then unfreeze: if the model recovers, early drift is the cause; if it does not, compare the input tensors with the probe's pipeline.
Why asked: this is the most common transfer failure for transformers and the check is cheap.
Follow-up: what would you freeze first on a domain with a strong colour shift? Everything except the head and the normalisation statistics, then re-estimate statistics on unlabelled target images.

**Q12. Where does the class token's information come from, and what is the alternative?**
Answer: the class token starts as a learned vector and accumulates information through attention, so its final state is a learned aggregation of the patch tokens. The alternative is global average pooling over patch tokens, which has no extra parameter, is more stable under grid changes and is often equivalent in accuracy. The choice belongs in the tuple because it changes the head.
Why asked: reveals whether the candidate knows the class token is a convention rather than a necessity.
Follow-up: which is preferable when the deployment grid may vary? Pooling, because it averages over whatever tokens exist rather than over a state learned at one grid.

**Q13. Estimate the score-matrix peak for N = 4,097, batch 4, 12 heads, fp16, across 12 layers, and state the first mitigation.**
Answer: per layer 4·12·4,097²·2 = 4·12·16,785,409·2 = 1,611,399,264 bytes ≈ 1.5 GiB; across twelve layers ≈ 18 GiB. The first mitigation is recomputation, which reduces the stored term to one layer's worth at a few percent extra arithmetic; the second is windowing, which reduces N² to N·M²; the third is precision, which halves bytes but not arithmetic.
Why asked: forces the candidate to do the four-factor multiplication with the right exponent.
Follow-up: why is precision third rather than first? Because it is a 2x lever against a 12x and a 64x lever, and it changes numerics, so it is the weakest of the three.

**Q14. When is a convolutional stem the right choice over a smaller patch?**
Answer: when labels are scarce and the objects are large enough that the finer token grid adds cost without adding discriminative structure. A stem reduces N by 4–16x before attention, restoring locality and cutting the quadratic term, whereas a smaller patch multiplies N and therefore the quadratic term by 4–16x while adding detail the task may not need.
Why asked: the two moves sound similar and have opposite cost directions.
Follow-up: what evidence would change your mind? Slice recall showing the small-object quartile is the failing one, which means detail is exactly what the task needs.

**Q15. What does 4x weight quantisation do for an edge deployment of this model, and what must be re-measured?**
Answer: it reduces weight storage and weight-streaming bytes by about 4x, which matters because at small N the workload is memory-bound; it does not reduce the score-matrix term, which is activation-side. Re-measure per-class recall at the operating threshold and the calibration curve, because quantisation changes the score tails first.
Why asked: connects the cost axis that binds at the edge to the metric that must be revalidated.
Follow-up: which component would you keep in higher precision? The patch embedding's input path and the classification head, because they straddle the raw pixel range and the logit scale.

## 12. Quick Reference Card

| Item | Value or rule |
|---|---|
| Tokens | N = (H/P)² + class token; scales as P⁻² |
| Block MACs | 12ND² + 2N²D |
| Attention fraction | N/(6D + N) of a block |
| Score memory | B·h·N²·bytes per layer |
| Crossover | N = 2D vs projections; N = 6D vs full block |
| Window saving | exactly N/M² |
| Shift mixing depth | about 2·log₂(number of windows) blocks |
| Parameters | L·12D² + P²CD + N_pos·D + D·K |
| Cheapest OOM fix | recompute attention, then lower precision, then window |
| Position rule | grid change ⇒ interpolate, adapt, and re-evaluate |
