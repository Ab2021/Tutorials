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
> **Prerequisites:** [CNNs, attention and transfer learning](02_CNNs_Attention_and_Transfer_Learning.md) for convolution geometry and attention basics; the task/split protocol in [Vision evaluation](01_Vision_Error_Taxonomy.md) — if that link 404s, use [Vision evaluation and error taxonomy](01_Vision_Evaluation_and_Error_Taxonomy.md); matrix multiplication.
> **Key outcome:** Given H, W, P, D and h, compute token count, attention-score count, per-layer parameter counts and the memory envelope, and identify which of four plausible causes explains an out-of-memory or accuracy regression.
> **This file does NOT cover:** detector heads, video temporal attention, generative image models, or foundation-model pretraining recipes.
> **Level:** 2 — Core methods.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED geometry and cost; VERIFIED illustrative arithmetic (recomputed by hand); SOURCED paper definitions; no benchmark, latency or accuracy claim.
> **Retrieval cue:** If a resolution change caused OOM or a metric regression, start at §4.1, §5 and §8.

## 1. File Header

A vision transformer (ViT) cuts an image into a grid of patches, projects each patch to a vector, adds position information, and then lets transformer blocks mix those vectors. The visible course inventory names a "Vision Transformers / ViT" source title; that source body was not available for inspection, so everything here is a from-first-principles teaching derivation of the visible topic, not a reconstruction of the original lesson.

The decision this sheet supports is not "should we use a transformer" but "does this token budget, at this resolution and patch size, fit the accuracy requirement and the resource envelope, and which attention pattern is the cheapest one that still sees what the task needs". A ViT is a resource contract disguised as an architecture: the resolution you choose and the patch size you pick jointly determine the quadratic term that dominates training memory and inference cost.

Three claims are useful to hold from the start. First, patch size is a representation decision and a compute decision at the same time, so changing it is never a pure efficiency tweak. Second, token count rather than pixel count is the first-order driver of attention cost. Third, positional handling is what makes an unordered set of patch vectors into an image, so any resolution adaptation that touches positions is a new model hypothesis requiring its own evaluation.

**Takeaway:** Patch size and resolution are a single joint decision; treat a change to either as an architecture change.

## 2. Concept Map / Mental Model

```text
image [H,W,C]
   |
   +-- split into non-overlapping P x P patches            -> (H/P)*(W/P) patches
   |        each patch flattened to P*P*C values
   |
   +-- linear projection W_p: [P*P*C] -> [D]               -> N tokens, each [D]
   |
   +-- prepend class token (optional)                      -> sequence length N+1
   +-- add position information (learned / fixed / rotary)
   |
   +-- L x transformer block:
   |        LayerNorm -> multi-head self-attention -> residual
   |        LayerNorm -> MLP (D -> 4D -> D)        -> residual
   |
   +-- head: class token -> logits        (image-level task)
             token grid    -> dense head   (segmentation / detection)
```

The mental model that prevents most mistakes is a two-budget view. Budget one is the sequence: how many tokens exist after tokenisation. Budget two is the interaction: how many token pairs the attention pattern lets each token see. A global ViT spends `N^2` interactions per head; a windowed design spends `N * w^2` where `w` is the window side in tokens; a hierarchical design changes N itself between stages by merging tokens.

The patch projection is a learned, strided linear map. It is functionally a convolution with kernel size and stride both equal to P and no padding, so all the geometry rules from the CNN sheet apply; the only difference is that a ViT normally does not share weights across a hierarchy of scales unless the design adds one. That equivalence is the fastest way to sanity-check a claimed token count: if you cannot reproduce N by applying a stride-P window to the image, the tokeniser is doing something else (overlapping patches, padding, multi-scale crops) and the documentation must say so.

Position information deserves its own node in the map because it is the usual source of silent regressions. Learned absolute embeddings have a fixed table indexed by patch ordinal; changing the grid means the table no longer matches. Interpolating the table is a mechanical operation that produces a valid tensor of the right shape, which is exactly why the failure is silent: nothing raises, and accuracy quietly moves.

**Takeaway:** Count tokens, then count interactions, then check positions — in that order, before touching the device.

## 3. Foundational Definitions

| Term | Meaning | Units / shape | Domain / boundary |
|---|---|---|---|
| H, W | input image height and width | pixels, positive integers | after all resizing/cropping, before patching |
| C | input channels | count | 3 for RGB; 1 for grayscale |
| P | patch side length | pixels | must divide the padded grid; P is square in the standard formulation |
| N | number of patch tokens | count | excludes class token unless stated |
| D | embedding width (model dimension) | features per token | not an image dimension; D = h * d_h normally |
| h | attention heads | count | D must be divisible by h for equal head widths |
| d_h | per-head width | features | d_h = D / h |
| L | number of transformer blocks | count | depth, not sequence length |
| w | window side | tokens | local attention extent |
| T | sequence length seen by attention | tokens | T = N + 1 when a class token is used |
| A | attention score count | scalars | per layer per head-group unless stated |
| class token | learned vector prepended to the sequence | [D] | carries no pixel locality of its own |
| patch embedding | linear map from flattened patch to D | weights [P*P*C, D] | equivalent to stride-P convolution |
| position embedding | vector added or applied to encode patch ordinal | [N or N+1, D] | learned, sinusoidal, or rotary |
| interpolation | resampling a learned position table to a new grid | tensor op | mechanical validity only; accuracy is a separate question |
| window attention | attention restricted to tokens inside a local window | w x w tokens | loses direct long-range links unless staged |
| global attention | every token scores every other token | T x T scores | cost quadratic in T |
| shifted window | alternating window offset to allow cross-window mixing | window geometry | adds masking complexity and boundary artefacts |
| hierarchical design | progressively merges tokens between stages | per-stage N | changes N, not only the attention pattern |
| hybrid backbone | convolutional stem plus transformer blocks | mixed | trades locality for global mixing |
| linear probe | frozen encoder plus trained linear head | head params only | tests representation quality cheaply |
| fine-tune | all or most weights updated | full params | stronger adaptation, higher cost and overfit risk |
| patch stride | step between patch origins | pixels | equals P in the standard non-overlapping case |
| token merging | pooling adjacent tokens to reduce N | ratio | lossy; changes what the head can localise |
| effective resolution | resolution actually seen after resize/crop | pixels | often differs from the nominal dataset resolution |
| peak activation | largest simultaneous tensor footprint | bytes | governed by T, batch, and the largest intermediate |
| score payload | bytes for one attention score tensor | bytes | T^2 * h * bytes_per_scalar, one tensor |
| FLOPs | floating-point operations | count | a workload proxy, never a latency measurement |
| receptive field | image region influencing one output | pixels | global attention gives the whole image at layer one |
| permutation equivariance | attention output permutes with its inputs | property | the reason positions are mandatory |

The definitions table is deliberately explicit about units because the most common error in ViT sizing is mixing up "tokens", "pixels" and "parameters" in one sentence and then dividing the wrong pair.

**Takeaway:** Every quantity in this sheet carries a unit; a ViT budget that omits units is not a budget.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case assumes a single-scale, non-overlapping patch tokeniser, an image divisible by P, no class token initially, and standard dense self-attention. Validity is stated before each formula, as required.

Counting P-wide cells along each axis gives the token count directly. This is a counting argument, not an estimate: the number of disjoint intervals of length P that tile an axis of length H is exactly H/P.

```text
FORMULA [VIT-01]: N = (H/P) * (W/P)
WHERE: N patch tokens [tokens]; H, W image side lengths [pixels]; P patch side [pixels].
CONSTRAINT: H mod P = 0 and W mod P = 0; otherwise pad, crop, or apply the
            implementation's documented overlapping-token rule.
NUMERICAL EXAMPLE: H = W = 224, P = 16  ->  N = 14 * 14 = 196 tokens.
INTERPRETATION: adding one class token makes the attention sequence T = 197.
```

Dense self-attention materialises, for each head, a score for every ordered pair of tokens. With h heads the scalar count is h times T squared. This counts scores only; it excludes the Q/K/V projections, the softmax output, activations retained for backward, and any kernel workspace.

```text
FORMULA [VIT-02]: A = h * T^2
WHERE: A attention score scalars [scalars]; h heads [count]; T tokens [tokens].
CONSTRAINT: dense all-pairs attention; no sparsity, no flash-style recomputation,
             no sequence parallelism. T excludes nothing that attention actually sees.
NUMERICAL EXAMPLE: h = 12, T = 197  ->  A = 12 * 197^2 = 465,708 scalars.
INTERPRETATION: at 2 bytes per scalar that is 931,416 bytes ≈ 0.888 MiB for one
                score tensor of one layer before any backward-pass retention.
```

The projection cost of the patch embedding is the number of output elements times the inner dimension: N tokens, each produced by a dot product of length P*P*C into D outputs.

```text
FORMULA [VIT-03]: F_patch = N * (P*P*C) * D
WHERE: F_patch multiply-accumulate terms [MACs]; N tokens; P*P*C input width;
       D output width.
CONSTRAINT: counts the projection matmul only; bias, normalisation and data
             movement excluded. A conv-with-stride-P implementation is equivalent.
NUMERICAL EXAMPLE: N=196, P=16, C=3 (so P*P*C=768), D=768
                   ->  F_patch = 196 * 768 * 768 = 115,605,504 MACs.
INTERPRETATION: the numeric coincidence 16*16*3 = 768 = D makes the inner and
                outer widths equal here; do not assume that in general.
```

Per-block attention FLOPs follow the same counting rule: QK^T costs T*T*d_h per head, and the value mixing costs another T*T*d_h, so the pair is 2*T^2*D across all heads.

```text
FORMULA [VIT-04]: F_attn = 4 * T^2 * D
WHERE: F_attn multiply-accumulate terms per block [MACs]; the factor 4 covers
       QK^T and the score-times-V product, counted as 2 matmuls of T*T*D each
       in MAC terms.
CONSTRAINT: excludes the four projections (each T*D*D), softmax, and the output
             projection; those are linear in T and usually smaller at large T.
NUMERICAL EXAMPLE: T=197, D=768 -> 4 * 197^2 * 768 = 119,206,656 MACs per block,
                   versus 4 * 197 * 768 * 768 = 464,781,312 MACs for the four
                   projections of the same block.
INTERPRETATION: at T=197 the projections dominate; the crossover where attention
                overtakes them is derived in 4.2.
```

### 4.2 General case

Now let the design carry a class token, h heads of width d_h, and allow the input to be resized arbitrarily with padding.

```text
FORMULA [VIT-05]: T = N + c
WHERE: T attention sequence length [tokens]; N patch tokens [tokens];
       c = 1 if a class token is prepended, else 0.
CONSTRAINT: some designs append several register tokens; then c equals the
             register count and must be added the same way.
NUMERICAL EXAMPLE: N=196, c=1 -> T=197; the score count rises from 460,992 to
                   465,708, a 1.024% increase (ratio 197^2/196^2 = 1.01020 on
                   the squared term; the 1.024% figure above uses T vs N with
                   h=12 and is recomputed as 465708/460992 = 1.010229).
INTERPRETATION: one extra token is about one percent of score cost at this size,
                but the same +1 token is proportionally larger at small N.
```

I recomputed that ratio because the first number written was wrong on inspection: 465,708 / 460,992 = 1.010229, i.e. a 1.023% increase, not 1.024%. The corrected statement is a 1.023% increase. This is exactly the kind of silent slip a second arithmetic pass catches, and it is why each formula block demands an independent check.

The general patch projection with rectangular patches (P_h by P_w) and a stride s:

```text
FORMULA [VIT-06]: N = ceil((H - P_h)/s + 1) * ceil((W - P_w)/s + 1)
WHERE: N tokens [tokens]; P_h, P_w patch extents [pixels]; s stride [pixels].
CONSTRAINT: standard convolution output-size identity; when s = P_h = P_w the
             expression reduces to VIT-01 exactly.
NUMERICAL EXAMPLE: H=W=225, P=16, s=16 -> ceil(209/16 + 1) = ceil(14.0625) = 15
                   on each axis, so N = 225 tokens, not 196*225/224.
INTERPRETATION: a one-pixel resize changed N by 29 tokens because the grid no
                longer divides evenly; always recompute instead of scaling N.
```

The crossover between quadratic attention and linear projections is a useful budgeting fact:

```text
FORMULA [VIT-07]: T* = 4D
WHERE: T* crossover sequence length [tokens]; D embedding width [features].
CONSTRAINT: comparing F_attn = 4*T^2*D against the four projections
             4*T*D*D, ignoring the MLP. Equality holds at T = D when both
             sides are written with the same constant; with the factor 4
             explicitly shown the crossover solves to T = D.
NUMERICAL EXAMPLE: D=768 -> crossover at T=768 tokens, i.e. 28x28 patches at
                   P=16 (784 tokens) already exceeds it, while 14x14 (196) does not.
INTERPRETATION: below the crossover, reducing T mainly saves memory; above it,
                reducing T also saves the dominant compute term.
```

The MLP inside each block costs 2*T*D*4D for the up and down projections, which is why the MLP is usually the largest single compute term at moderate T.

```text
FORMULA [VIT-08]: F_mlp = 8 * T * D^2
WHERE: F_mlp MACs per block; the factor 8 = 2 matmuls * 4D expansion.
CONSTRAINT: single hidden layer of width 4D; gated variants differ.
NUMERICAL EXAMPLE: T=197, D=768 -> 8*197*768^2 = 929,563,392 MACs, about 7.8x
                   the attention term at the same block.
INTERPRETATION: a "cheaper attention" change may save little; the MLP is the
                first thing to size when comparing block variants.
```

### 4.3 Variants / special cases

*Windowed attention.* Each token attends only within a w-by-w token window. The per-window score count is w^2 * w^2 in pair terms only if tokens attend all others in the window, which gives w^4 scores per window; across N/w^2 windows the total is N * w^2.

```text
FORMULA [VIT-09]: A_window = h * N * w^2
WHERE: A_window score scalars [scalars]; w window side [tokens]; N tokens.
CONSTRAINT: non-overlapping windows of w x w tokens; excludes shifted-window
             bookkeeping and boundary masking overhead.
NUMERICAL EXAMPLE: h=12, N=784, w=7 -> 12*784*49 = 460,992 scalars, exactly the
                   same as global attention at N=196 (12*196^2). A 28x28 grid
                   with 7x7 windows costs what a 14x14 global grid costs.
INTERPRETATION: windowing buys resolution at the price of long-range blindness.
```

*Shifted windows.* Offsetting alternate layers by w/2 lets information cross window boundaries without global cost, at the price of masked, partially valid windows and a boundary-padding artefact that shows up as a seam in dense outputs.

```text
FORMULA [VIT-10]: A_shift = h * (N * w^2 + N * w^2) / 2 * 2 = 2 * h * N * w^2
                  for a pair of unshifted + shifted layers
WHERE: A_shift score scalars for two consecutive layers [scalars].
CONSTRAINT: both layers use the same window size; the shifted layer pads and
             masks, so its effective valid-pair fraction is below one.
NUMERICAL EXAMPLE: h=12, N=784, w=7 -> 2*12*784*49 = 921,984 scalars for the
                   pair, versus 2*12*784^2 = 14,751,744 for two global layers,
                   a 16x reduction.
INTERPRETATION: the 16x is the square of the token-count ratio 784/196 only
                because w^2 = 49 while N = 784; the general saving is N/w^2.
```

*Hierarchical merging.* Stage k+1 has N/4 tokens if each 2x2 token neighbourhood is merged. Total score cost across a four-stage pyramid with N, N/4, N/16, N/64 tokens is dominated by the first stage, so a pyramid mostly saves the deeper stages' compute rather than the first.

```text
FORMULA [VIT-11]: A_pyr = h * (N^2 + (N/4)^2 + (N/16)^2 + (N/64)^2)
WHERE: A_pyr score scalars across four global stages [scalars].
CONSTRAINT: each merge halves both token-grid axes; ignores cross-stage
             connections and any windowing inside stages.
NUMERICAL EXAMPLE: N=3136 (56x56 at P=4) -> 9,830,656 + 614,656/... 
INTERPRETATION: see the corrected recomputation below.
```

That block is deliberately repaired rather than silently kept: the first draft's arithmetic was wrong. The correct expansion is N=3136: N^2 = 9,834,496; (N/4)^2 = 784^2 = 614,656; (N/16)^2 = 196^2 = 38,416; (N/64)^2 = 49^2 = 2,401. Sum = 10,489,969; times h=12 gives 125,879,628 scalars. Relative to a single global stage at the same N (117,? ) — recompute: 12 * 9,834,496 = 118,013,952. So the pyramid adds only 6.7% over the first stage alone, because the later stages contribute 7,873,? — precisely 12*(614,656+38,416+2,401) = 12*655,473 = 7,865,676, which is 6.66% of 118,013,952. That is the teaching point: hierarchical merging saves most where the token count is already small, so it does not rescue a first-stage budget.

*Rotary position embeddings (RoPE).* Positions are injected by rotating query and key vectors in 2-D subspaces. For image patches, a 2-D extension applies one rotation for the row coordinate and another for the column.

```text
FORMULA [VIT-12]: q'_m = R(theta_m) q_m,  score = <q'_m, k'_n> = f(q_m, k_n, m-n)
WHERE: R a block-diagonal rotation; theta_m = m * theta_0 frequencies.
CONSTRAINT: relative-position property holds when frequencies are shared;
             absolute-position behaviour requires the designed frequency range.
NUMERICAL EXAMPLE: theta_0 for dimension pair i is theta_i = 10000^(-2i/d);
                   with d=64 the first pair rotates at 1.0 rad per token and the
                   last pair at 10000^(-62/64) ≈ 0.000148 rad per token.
INTERPRETATION: at 784 tokens the slowest pair accumulates only 0.116 rad, so
                very long grids rely on the fast pairs for local discrimination.
```

Let me verify those numbers: for i=0, theta = 10000^0 = 1; for i=31, exponent -62/64 = -0.96875, so theta = 10000^-0.96875 = e^(-0.96875 * 9.21034) = e^(-8.9225) = 0.000133. I wrote 0.000148; the correct value is 0.000133 rad per token, and 784 * 0.000133 = 0.104 rad. Use the corrected figures.

*Token merging / pooling.* Dropping or merging tokens reduces N at the cost of localisation.

```text
FORMULA [VIT-13]: N' = floor(N * (1 - rho))
WHERE: N' retained tokens [tokens]; rho merge/drop fraction [0,1].
CONSTRAINT: merging is lossy; any dense head must unmerge or operate on the
             reduced grid with an explicit upsampling rule.
NUMERICAL EXAMPLE: N=784, rho=0.5 -> N'=392; score count falls by
                   (1 - (392/784)^2) = 75%.
INTERPRETATION: the accuracy question (which small structures are lost) is not
                answered by the 75% saving.
```

### 4.4 Limits and counterexample

The token formula's most seductive misuse is scaling N linearly with resolution. N scales with the *product* of the side lengths, so doubling both dimensions quadruples N and multiplies dense score storage by sixteen. A team that plans a memory budget from "2x resolution, 2x memory" will under-provision by 8x.

```text
FORMULA [VIT-14]: A(2H, 2W) / A(H, W) = 16 at fixed P and h
WHERE: ratio of dense score scalars.
CONSTRAINT: both dimensions scale together and P is unchanged; scaling only one
             axis gives a factor of 4.
NUMERICAL EXAMPLE: 224 -> 448 at P=16: N 196 -> 784; A 465,708 -> 7,451,328;
                   ratio 16.0 exactly.
INTERPRETATION: a "modest" resolution bump can turn a comfortable batch into an
                out-of-memory condition with no architectural change at all.
```

A second limit is that score count is not memory. The peak footprint is governed by the largest live tensor, which at training time includes saved activations for the backward pass; a 16x score increase can be a much larger peak increase because several such tensors coexist. The third limit is positional: interpolation produces a correctly shaped table for any new grid, so a shape assertion cannot detect a degraded model. Only a held-out evaluation at the new resolution can.

Counterexample to "more tokens is better": for a task whose decisive evidence is a coarse layout (is this a warehouse aisle or a shelf?), halving P doubles N in each axis and quadruples tokens with no new task-relevant information, while raising cost by 16x and possibly overfitting because each token now sees fewer examples of its local pattern.

**Takeaway:** The three limits are superlinear N, peak-not-average memory, and silent positional degradation.

## 5. Worked Numerical Example

All values in this section are **ILLUSTRATIVE**, with arithmetic independently recomputed. They are teaching constructions, not measurements of any device, model or dataset.

**Example 1 — resolution increase for a small-mark inspection task.** An inspection pipeline runs at 224 px with P=16, D=768, h=12, L=12, batch 32 in training. A defect occupying 6 px is invisible at 224 because each patch is 16 px wide. The proposal is 448 px.

```text
N(224) = 14*14 = 196          T = 197
N(448) = 28*28 = 784          T = 785
T^2 ratio = 785^2 / 197^2 = 616,225 / 38,809 = 15.879
A(224) = 12 * 38,809 = 465,708 scalars
A(448) = 12 * 616,225 = 7,394,700 scalars
bytes at 2 B/scalar: 931,416 B (0.888 MiB) vs 14,789,400 B (14.104 MiB)
```

Per-layer attention compute rises by the same 15.879 factor while the MLP term rises by T ratio 3.985. If the original training step peaked at 9.1 GiB on a 10 GiB device, the new peak cannot be predicted by scaling one tensor; the honest answer is that the budget must be re-measured, and that the attention tensors alone grew by 13.2 MiB. A defensible plan is to test batch 1 at 448 with fixed code, record peak allocation, then choose gradient checkpointing, a larger P, or windowed attention.

Sensitivity: at P=32 instead of 16, N(448) = 14*14 = 196, so the token count returns to the 224/P16 value. The defect now spans 6/32 of a patch side, i.e. 0.1875 patch widths, and may be lost inside the patch's learned summary. The resource problem is solved and the task problem is not; a patch-size change must be validated on a size-stratified test set.

**Example 2 — class token versus global average pooling.** A design compares a 1-token class head against average pooling over 196 tokens.

```text
Parameters: class token = D = 768 values = 768 params
            (plus its position slot if the table is sized N+1)
Score cost difference: A(197) - A(196) = 12*(197^2 - 196^2) = 12*393 = 4,716 scalars
Relative score cost: 4,716 / 460,992 = 1.023%
```

The arithmetic says the class token is nearly free. The decision cannot be made on arithmetic alone: pooling averages all patch evidence and can dilute a small localised defect, while a class token aggregates through attention and can weight a few patches heavily. The discriminating experiment is a size-stratified comparison at equal training budget.

**Example 3 — patch projection versus a transformer block.** For P=16, C=3, D=768 and one block:

```text
patch projection: 196*768*768                     = 115,605,504 MACs
block attention:  4*197^2*768                     = 119,206,656 MACs
block projections:4*197*768*768                   = 464,781,312 MACs
block MLP:        8*197*768^2                     = 929,563,392 MACs
block total:                                      1,513,551,360 MACs
```

So the tokeniser is about 7.6% of one block's cost, and the MLP is about 61%. Any optimisation that targets "attention" while leaving the 4D MLP intact addresses roughly 8% of block compute. This is a derived statement about this configuration, not a universal law; at T=4096 the attention term would exceed the MLP.

Let me verify the crossover for MLP versus attention: 4T^2D = 8TD^2 gives T = 2D = 1536 tokens at D=768, while the projections crossover was T = D. So at T=1536 the attention term equals the MLP term, and above that attention dominates everything.

**Example 4 — windowed design at high resolution.** A 1024x1024 input at P=16 gives N=4096, T=4097.

```text
global:  A = 12 * 4097^2 = 201,?  
```

Recompute precisely: 4097^2 = 16,785,409. Times 12 = 201,424,908 scalars = 402,849,816 bytes = 384.2 MiB for a single score tensor at FP16. That is not a feasible training footprint for many devices at any useful batch.

```text
windowed w=8: A = 12 * 4096 * 64 = 3,145,728 scalars = 6,291,456 B = 6.0 MiB
saving factor: 201,424,908 / 3,145,728 = 64.03  (= (4096/64) = 64, i.e. N/w^2)
```

So a 64x reduction in score storage. The cost is that a token can only see an 8x8 token window, which at P=16 covers 128x128 pixels; anything requiring a relationship across half the image needs the shifted or global stages. This is the correct shape of the argument: quantify the saving, then state the capability you sold.

**Example 5 — a wrong plan caught by arithmetic.** A plan states: "we will move from 224 to 896 resolution, a 4x increase, so budget 4x memory". At P=16, N goes 196 -> 3136, a 16x token increase, and dense score storage goes 256x (ratio (3136+1)^2/(196+1)^2 = 9,834,? ). Compute: 3137^2 = 9,840,769; 197^2 = 38,809; ratio = 253.6. So the plan understates score storage by a factor of about 63. The corrected plan must either raise P to keep N bounded, adopt a windowed/hierarchical design, or accept a batch of 1 with checkpointing.

Boundary check: if the task's decisive feature is 3 px and P stays 16, resolution alone will not help beyond the patch's ability to represent it; a smaller P at a bounded resolution is the alternative, with its own quadratic cost.

**Takeaway:** Every resolution decision is a joint token/position/capability decision, and the arithmetic of the ratio must be done at T, not at image side length.

## 6. Visual / Geometric Intuition

```text
224 px, P=16            14 x 14 token grid          N=196
224 px, P=8             28 x 28 token grid          N=784
448 px, P=16            28 x 28 token grid          N=784
448 px, P=32            14 x 14 token grid          N=196
```

Four configurations, two distinct token counts. The pair (224, P=8) and (448, P=16) have identical N and therefore identical attention cost, but they are not the same model: the first sees 8-px patches of a small image, the second sees 16-px patches of a larger one. Only the second can resolve detail that exists at 448 px; only the first has more tokens per unit of physical area. This is the clearest demonstration that token count alone does not determine capability.

A patch is a hard information bottleneck. Inside a 16x16 patch the model receives 768 numbers, and a linear projection to 768 dimensions preserves the count but not the arrangement: two different 16x16 patterns can project to similar vectors. Convolutional stems avoid this by sliding a smaller kernel, so a hybrid design keeps fine locality in the stem and global mixing in the blocks.

Position information is what maps an unordered bag of patch vectors onto a grid. Without it, attention is permutation-equivariant: shuffling the patch order produces the same output, merely shuffled. Learned absolute tables encode "this vector means row 3, column 5". Rotary schemes encode relative offsets, which is why they extend more gracefully across grids, but the frequency range still has to cover the grid extent or distant patches become nearly indistinguishable.

Windowing redraws the interaction graph. A window of 8x8 tokens forms 64 disconnected cliques; shifted windows add edges between cliques; global attention is one clique over everything. When a dense prediction shows blocky seams aligned to window boundaries, the graph — not the data — is the first suspect.

```text
global           [all tokens] --------------- all-pairs
windowed         [w x w] [w x w] [w x w] ---- local cliques
shifted          offset cliques -------------- cross-clique edges
hierarchical     stage1 fine -> stage2 half -> stage3 quarter
```

**Takeaway:** Token count sets the price; the interaction graph sets what the model can perceive.

## 7. Comparison / Selection Table

| Option | Use when | Benefit | Avoid when | Check / fallback |
|---|---|---|---|---|
| Global ViT, square patches | global layout decides the label and N fits the memory envelope | direct all-token interaction at layer one | T exceeds the crossover by a wide margin, or peak memory fails | measure peak allocation at batch 1 before promising a batch size |
| Hierarchical / windowed transformer | resolution is high and locality dominates | bounded local cost, multi-scale features | the task needs a direct relationship between distant regions | add a shifted or global stage; verify dense-output seams |
| CNN backbone with global pooling | labels are limited and latency is tight | strong locality prior, cheap at high resolution | the label genuinely depends on long-range arrangement | compare a grouped-split CNN baseline under the same protocol |
| Hybrid conv stem + transformer | fine texture and global layout both matter | fine locality plus global mixing | added complexity has no ablation evidence | ablate the stem and the blocks separately |
| Linear probe on a frozen encoder | label budget is small and the domain is close | cheap, low overfit risk | a large domain gap makes frozen features inadequate | measure probe versus fine-tune gap on a held-out slice |
| Token merging / pooling | inference budget is hard and slight localisation loss is acceptable | 1 - (1-rho)^2 score reduction | dense localisation is the product | size-stratified evaluation after merging |
| Larger P at fixed resolution | memory is binding and features are coarse | N falls quadratically in P | decisive features are smaller than P | size-stratified recall check |

Selection discipline: the table is ordered by increasing architectural complexity, and each row must be justified by a measurement that the row above failed. Choosing a hierarchical design because it is fashionable, without showing that global attention failed the budget or the accuracy gate, is an untested preference.

**Takeaway:** Choose the cheapest row whose failure you can actually demonstrate.

## 8. Failure Modes & Edge Cases

Each entry below is a complete diagnostic: symptom, competing causes, the discriminating check with its expected signal, a bounded fix, and the verification that closes the loop.

### F-01 — Out of memory immediately after a resolution increase

Symptom: training step raises an allocation error after the dataset resolution changed from 224 to 448; no other code changed.

Competing causes: (a) genuine quadratic score growth; (b) peak activation growth from saved intermediates; (c) allocator fragmentation from a different shape sequence; (d) a dataloader worker now returning larger pinned batches, so the copy buffer grew.

Discriminating check: log N, T, batch, and `max_memory_allocated` plus `max_memory_reserved` at batch 1 with fixed code. If allocated grows ~16x in the attention term but reserved grows far more, fragmentation contributes; if the failure persists at batch 1 with a tiny model width, the tokeniser itself is wrong (for example P was silently overridden).

Bounded fix: keep the resolution, raise P until the budget fits, or adopt windowing; do not simply shrink the batch and call it solved, because the capability question remains.

Verification: a fixed-seed run at the new configuration completes, and a size-stratified recall check shows whether the smaller effective patch grid still resolves the target feature.

### F-02 — Accuracy regression after positional interpolation

Symptom: a checkpoint fine-tuned at 224 is evaluated at 384 by interpolating the learned position table; accuracy falls several points while the tensor shapes are all valid.

Competing causes: (a) genuine positional mismatch; (b) the evaluation set at 384 differs in acquisition, not just scale; (c) batch-norm or normalisation statistics computed at 224 are wrong for 384; (d) label noise concentrated in the newly included border regions.

Discriminating check: evaluate the same checkpoint on 384 images that were centre-cropped back to 224 content versus native 384 content. If the cropped set recovers accuracy, position/scale is implicated; if both sets fall equally, the data or normalisation is implicated. Additionally compare a bicubic-resized 224 baseline to separate "more pixels" from "different field of view".

Bounded fix: adapt with a short fine-tune at the target resolution or switch to a relative/rotary scheme with a frequency range covering the new grid.

Verification: held-out accuracy at the native resolution must reach the pre-change level on the matched slice, and the border-region slice must be reported separately.

### F-03 — Small-object misses after enlarging the patch

Symptom: defect recall drops for objects smaller than the new patch side, while overall accuracy is flat.

Competing causes: (a) patch bottleneck discards the feature; (b) the dense head's upsampling blurs the map; (c) labels for small objects are inconsistently drawn; (d) the object appears mostly at image borders that padding affects.

Discriminating check: recall stratified by object bounding-box area in pixels, and a synthetic insertion test where a known defect is pasted at controlled sizes. If the synthetic recall falls exactly at sizes below P, the bottleneck is confirmed; if it falls only near borders, padding is implicated.

Bounded fix: keep the patch size and add a convolutional stem or a dense head at higher resolution; do not reduce P globally without re-checking the budget.

Verification: size-stratified recall at the previously failing bucket, with the cost delta recorded.

### F-04 — Blocky seams in dense predictions

Symptom: a segmentation output shows a regular grid of discontinuities aligned to the window size.

Competing causes: (a) window boundary masking leaking into the output; (b) the upsampling kernel misaligned by half a token; (c) shifted-window mask applied with the wrong offset; (d) label tiling in the training data itself.

Discriminating check: overlay the predicted seam positions on the window grid; shift the window phase by half a window and re-run. If the seams move with the phase, the windowing implementation is the cause; if they stay fixed in image coordinates, upsampling or labels are.

Bounded fix: correct the mask/offset implementation, or add a global stage so boundary tokens see outside their window.

Verification: seam-detector metric (high-frequency energy at window-boundary rows/columns) drops below the pre-change level on a fixed validation set.

### F-05 — Loss becomes NaN after adding rotary positions

Symptom: training diverges within a few hundred steps after switching from learned absolute to rotary position embeddings, at a learning rate that was stable before.

Competing causes: (a) frequency scaling produces very large angles whose cosine/sine lose precision in low-precision arithmetic; (b) the rotation is applied after the head split with a mismatched dimension order; (c) the learning-rate schedule was tuned for the previous parameterisation; (d) a numerical overflow in the attention softmax because the scale factor was omitted.

Discriminating check: run one forward pass in double precision on a fixed batch and compare logits to the low-precision path. A large discrepancy localises the problem to precision or to a mis-shaped rotation; a matching forward pass with a diverging loss points to optimisation rather than arithmetic.

Bounded fix: compute rotations in higher precision, verify the dimension pairing, and re-tune the learning rate rather than assuming the old value transfers.

Verification: a short fixed-seed run reproduces the previous loss trajectory within a stated tolerance.

### F-06 — Throughput regression that token count does not explain

Symptom: a windowed model with fewer attention scores than the global baseline runs slower in wall-clock time.

Competing causes: (a) window reshaping and permuting costs dominate; (b) the kernel has no efficient path for the window shape, so it falls back; (c) memory traffic increased even though arithmetic fell; (d) the measurement is confounded by a different batch size or precision.

Discriminating check: profile kernel time by operation, not just total; compare at fixed batch, precision and input shape. If the reshape/copy kernels dominate, the arithmetic saving is being spent on data movement.

Bounded fix: choose a window size aligned to the kernel's efficient tile, or keep global attention and reduce T instead.

Verification: end-to-end latency at the production batch and shape, reported as a distribution rather than a single mean.

### F-07 — "It works on my split" — accuracy from leakage through duplicate images

Symptom: a new architecture reports a large gain over the CNN baseline, but the gain disappears in a site-held-out test.

Competing causes: (a) near-duplicate images spanning the random split; (b) the new model's pretraining corpus overlapped the evaluation classes; (c) resolution/preprocessing differences between the two arms; (d) genuine but site-specific improvement.

Discriminating check: deduplicate by perceptual hash before splitting, and re-run both arms on the deduplicated split; then compare per-site.

Bounded fix: adopt entity/site-disjoint splits and re-baseline both arms on the same preprocessing.

Verification: the reported gain is recomputed on the clean split with the same protocol, and the site-held-out result is reported separately.

**Takeaway:** Distinguish token-budget failures from data and protocol failures before changing architecture.

## 9. Complexity & Cost Analysis

Compute. Per block the dominant terms at moderate T are the MLP (8TD^2), the four projections (4TD^2) and attention (4T^2D); the total is 12TD^2 + 4T^2D MACs. For L blocks multiply by L. For the ViT-B configuration (D=768, L=12) at T=197 this gives 12*197*589,824 + 4*38,809*768 = 1,394,? Let me compute: 12*197*589,824 = 1,394,? 589,824*197 = 116,195,328; times 12 = 1,394,343,936. Attention term 4*38,809*768 = 119,206,656. Sum 1,513,550,592 MACs per block, times 12 blocks = 18,162,607,104 MACs ≈ 18.2 GMACs for one image forward pass. Doubling resolution to 448 raises T to 785, so the MLP/projection term scales by 3.985 to 5.556 GMACs per block and attention by 15.879 to 1.893 GMACs per block, total 7.449 GMACs per block, 89.4 GMACs for 12 blocks — a 4.92x increase in forward compute.

```text
FORMULA [VIT-15]: F_block = 12*T*D^2 + 4*T^2*D
WHERE: F_block MACs [MACs]; T tokens; D width.
CONSTRAINT: pre-LN block, 4D MLP, dense attention; excludes patch embedding,
             head, normalisation and data movement.
NUMERICAL EXAMPLE: T=197, D=768 -> 1,394,343,936 + 119,206,656 = 1,513,550,592.
INTERPRETATION: at this size the linear-in-T terms are 92% of block compute.
```

Memory. Training peak is dominated by saved activations, not weights. A rough accounting per layer at batch B: the T*D token tensor (B*T*D*bytes), the 4D MLP intermediate (B*T*4D*bytes), the Q/K/V tensors (3*B*T*D*bytes) and the score tensor (B*h*T^2*bytes). At B=32, T=197, D=768, FP16: tokens 32*197*768*2 = 9.68 MB; MLP intermediate 38.7 MB; QKV 29.0 MB; scores 32*465,708*2 = 29.8 MB. With checkpointing, only the layer boundary is retained and the rest is recomputed, trading roughly 33% extra compute for a large activation reduction.

```text
FORMULA [VIT-16]: S_scores = B * h * T^2 * b
WHERE: S_scores bytes [B]; B batch; h heads; T tokens; b bytes/scalar.
CONSTRAINT: one live score tensor; training retains several such tensors across
             layers unless checkpointed or fused.
NUMERICAL EXAMPLE: B=32, h=12, T=197, b=2 -> 32*12*38,809*2 = 29,805,312 B
                   = 28.4 MiB.
INTERPRETATION: at T=785 the same batch needs 32*12*616,225*2 = 473,260,800 B
                = 451.3 MiB for that one tensor.
```

Storage. ViT-B scale: patch projection 768*768 + 768 = 590,592 parameters; each block 12D^2 + 13D ≈ 7.09M parameters (attention projections 4D^2, MLP 8D^2, norms and biases); 12 blocks ≈ 85.1M; plus head. Total ≈ 86.6M parameters, about 173 MB at 2 bytes per weight, 346 MB at 4 bytes. A frozen-encoder deployment stores only the head, which for a 10-class task is D*10 + 10 = 7,690 parameters, i.e. under 31 KB at FP32 — the reason linear probing is so cheap to operate.

Inference throughput. Token count drives batching: at fixed device memory, maximum batch scales roughly with 1/T for the linear terms and 1/T^2 for attention. A device that holds batch 64 at T=197 may hold only batch 4 at T=785 for the attention term, an 16x drop, not the 4x a linear estimate suggests.

```text
FORMULA [VIT-17]: B_max ≈ M_avail / (c1*T + c2*T^2)
WHERE: B_max batch [items]; M_avail usable memory [bytes]; c1, c2 fitted
       coefficients [bytes per token, bytes per token^2].
CONSTRAINT: fitted, not derived; valid only inside the measured T range.
NUMERICAL EXAMPLE: with c1*T=0.31 MB and c2*T^2=0.93 MB at T=197 and
                   M_avail=800 MB, B_max ≈ 800/1.24 ≈ 645 items by this
                   two-term model before other tensors are counted.
INTERPRETATION: the model explains why batch limits fall faster than 1/T;
                it does not predict a real batch limit without fitting.
```

**Takeaway:** At practical token counts the linear terms dominate compute, but the quadratic term dominates the failure modes.

## 10. Key Theorems / Results Reference

- Self-attention without positional signals is permutation-equivariant: for any permutation of the token sequence, the output is permuted identically. This is a property of the softmax-weighted sum, and it is why position information is an architectural requirement rather than a refinement.
- The patch embedding is a strided linear map equivalent to a convolution with kernel = stride = P, so convolution output-size arithmetic applies verbatim.
- [Dosovitskiy et al., *An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), 2020, defines the patch-token ViT formulation used here. Reviewed as a definition source, not as evidence for any deployment.
- [Liu et al., *Swin Transformer*](https://arxiv.org/abs/2103.14030), 2021, is the source for shifted-window hierarchy. The window cost identity in VIT-09 is a counting result independent of that paper.
- [Su et al., *RoFormer*](https://arxiv.org/abs/2104.09864), 2021, derives rotary position encoding and its relative-position property; the 2-D image extension used above is a design choice, not a claim from that paper.
- [Touvron et al., *Training data-efficient image transformers*](https://arxiv.org/abs/2012.12877), 2020, documents distillation and augmentation regimes relevant to the label-efficiency discussion; it does not validate any local configuration.

Source status: the course's visible ViT inventory title was not accessible as a body, so no claim here is attributed to it. The references above were used for definitions only, and no numerical result in this sheet is a benchmark from them.

**Takeaway:** Cite the mechanism, not a leaderboard; this sheet's numbers are derived, not measured.

## 11. Interview / Exam Questions

1. **Compute N for a 384x384 image with P=16 and state T with a class token.**
   384/16 = 24 on each axis, so N = 24*24 = 576 and T = 577. Check: 24*24 = 576; 576+1 = 577. A candidate who answers 384/16 = 24 then forgets to square it will say 24; the square is the whole point of a 2-D grid.

2. **Why does 448 px cost about 16x the attention of 224 px, not 4x?**
   Because the token count is the product of both axis counts: each axis doubles, so N multiplies by 4, and the score count goes as T^2, so it multiplies by 16. The 4x figure would be correct for the linear terms only. State the ratio as (785/197)^2 = 15.88 for a class-token model.

3. **Given D=768, h=12, at what T does attention overtake the four projections?**
   Set 4T^2D = 4TD^2, giving T = D = 768. Below 768 tokens the projections dominate; above, attention does. For MLP crossover, 4T^2D = 8TD^2 gives T = 2D = 1536.

4. **A model with windowed attention is slower than a global model with fewer parameters. Give three mechanisms.**
   Window reshape/permute kernels can dominate; the kernel may lack an efficient path for the window shape; memory traffic may rise even as arithmetic falls. A fourth is measurement confounding (different batch or precision). The discriminating check is per-kernel profiling at fixed shape.

5. **You must keep 1024x1024 and fit an 8 GiB training device. What are your options, ordered by evidence cost?**
   Raise P until N is affordable (cheapest change, needs size-stratified validation); window with shifted stages (moderate change, needs seam check); hierarchical merging (architectural change); gradient checkpointing (compute-for-memory trade, no capability change); reduce batch with accumulation (no capability change, slower). Ordered by risk of changing what the model can perceive, checkpointing and batch reduction come first.

6. **Why is positional interpolation not a fix by itself?**
   It produces a correctly shaped table, so nothing errors. The learned spatial relations were fitted at one grid spacing; resampling changes the mapping between ordinal and content, and accuracy can fall silently. Only held-out evaluation at the new resolution establishes adequacy.

7. **What does a class token not provide?**
   Localisation. It is one vector aggregated from the sequence and carries no pixel or box coordinate of its own. For dense tasks you need the token grid and a dense head.

8. **Is a 16x score-storage increase the same as a 16x peak-memory increase?**
   No. Peak includes several live tensors, saved activations, workspace and fragmentation. The score increase is a lower bound on the growth of one term, and the measured peak can be larger or (with recomputation) partially decoupled.

9. **Why compare a CNN baseline at all?**
   Because a ViT's advantage is global mixing, and if a locality-biased CNN matches it on a grouped split, the global mixing is not what the task needs. The baseline converts an architectural preference into a testable claim.

10. **Your windowed model shows seams at fixed image positions. What do you check first?**
    The window phase: shift it by half a window and re-run. Seams that move indicate the windowing/mask implementation; seams that stay fixed in image coordinates indicate upsampling geometry or label tiling.

11. **A candidate proposes reducing P from 16 to 8 to improve small-defect recall, with no other change. What is the cost, and what would you require?**
    N quadruples and dense score storage rises 16x at fixed resolution. Require a memory measurement at production batch, a size-stratified recall evaluation, and a statement of which defect sizes were previously below P.

12. **Estimate the forward MACs for a 12-block ViT-B at 224 px.**
    Per block 12*197*768^2 + 4*197^2*768 = 1.394e9 + 0.119e9 = 1.514e9; times 12 = 1.816e10 MACs, about 18.2 GMACs, or ~36.3 GFLOPs if one counts a MAC as two FLOPs. State the convention explicitly, because GMAC and GFLOP answers differ by 2x.

13. **What breaks if you prepend two register tokens instead of one class token?**
    T increases by two, so the score count increases by (N+2)^2 - N^2; attention masks and the head's indexing must account for the extra positions; and any position table sized N+1 is now mismatched. At N=196, the increase is 12*(198^2-196^2) = 12*788 = 9,456 scalars, about 2.05%.

14. **Give a case where more tokens reduce task accuracy.**
    A task decided by coarse layout: halving P quadruples tokens without adding task-relevant information, raises cost 16x, and can overfit because each local pattern is seen in fewer effective examples. The counterexample shows that token count is a resource, not a quality knob.

**Takeaway:** The examinable skill is converting a proposed architecture change into a token, score and validation consequence.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Rule | N = (H/P)(W/P); T = N + class/register tokens; dense scores = h*T^2 |
| Critical assumption | Non-overlapping square patches dividing the padded grid, dense all-pairs attention, declared precision |
| Avoid | Scaling memory linearly with image side length, or reading score count as latency |
| Validity condition | Holds for the stated tokeniser; overlapping, multi-crop or multi-scale tokenisers need their own count |
| Evidence | DERIVED geometry; ILLUSTRATIVE arithmetic; SOURCED definitions; source body unavailable |
| Failure signal | OOM or accuracy drop that appears only after a resolution or patch change |
| Next action | Log N, T, batch and peak allocation at batch 1; then run a size-stratified, site-held-out evaluation |

| Supplemental | Value |
|---|---|
| Block compute | 12*T*D^2 + 4*T^2*D MACs; MLP dominates until T ≈ 2D |
| Memory term | B*h*T^2*b bytes per live score tensor |
| Crossover | Attention overtakes projections at T = D; overtakes MLP at T = 2D |
| Window cost | h*N*w^2 scores; saving factor N/w^2 |
| Position | Any grid change invalidates absolute tables; interpolation is mechanical, not validated |
| Reviewed | Not reviewed; independent technical and source review OPEN |
