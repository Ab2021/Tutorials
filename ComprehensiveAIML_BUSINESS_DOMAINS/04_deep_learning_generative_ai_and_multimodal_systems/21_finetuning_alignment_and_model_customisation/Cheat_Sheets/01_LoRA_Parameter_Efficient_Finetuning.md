---
title: "LoRA Parameter-Efficient LLM Fine-Tuning"
topic: "LoRA low-rank adaptation: parameter accounting, memory budgeting, merge and rollout"
collection: "finetuning-alignment-and-model-customisation"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# LoRA Parameter-Efficient LLM Fine-Tuning

## 1. File Header

**Purpose:** Given a decoder's projection shapes, a target-module list, a rank and a precision budget, derive the trainable
parameter count, the resident training state, the merge cost and the serving consequence, then design the quality gate
that decides whether the resulting adapter may be deployed.
**Prerequisites:** Matrix shapes and products, the transformer block's four attention projections and three MLP
projections, the chain rule for a two-layer composition, token-level cross-entropy, and binary byte units where
1 MiB = 1,048,576 bytes and 1 GiB = 1,073,741,824 bytes.
**Key outcome:** You can answer three questions with arithmetic rather than intuition: how many scalars will be trained,
how much device memory the training step will hold at its peak, and what part of the base behaviour the adapter is even
capable of changing.
**Exclusions:** Quantized base weights (see the QLoRA card), preference optimization, model merging algebra across
heterogeneous deltas, context extension, and multi-tenant request routing as a platform design. Those are separate
artifacts in this unit, and this card only states the interfaces to them.
**Level:** Core. The algebra is elementary; the judgement is in target selection, evaluation and lifecycle.
**Version / reviewed:** 1.0 / not independently reviewed. Status is draft, and every numeric result below is either an
identity you can recheck by hand or a labelled ILLUSTRATIVE construction.
**Evidence:** DERIVED for every identity and count; ILLUSTRATIVE for the 32-block worked model; OPEN for all runtime,
latency and quality claims. No production measurement, vendor benchmark or customer outcome is asserted anywhere.

The source inventory for this unit lists a PEFT submodule covering LoRA and related parameter-efficient methods, and an
implementation submodule that combines LoRA with quantized bases.
Neither original lesson body was available for inspection, so nothing in this card is attributed to it.
The technique itself is documented in the primary paper cited in Section 10, and the derivations here are independent.

A reader arriving from a "how many parameters does LoRA train" search should start at Section 4.2 and then read Section 5,
because the single most common error is counting one matrix pair and forgetting that a 32-block model repeats the
pattern thirty-two times, over several distinct projection shapes.
A reader arriving from an out-of-memory incident should start at Section 8 entries F2 through F5, because the adapter is
rarely the largest resident tensor and the peak usually belongs to activations or to the frozen base.

**Retrieval cue:** Sections 4.1–4.3 hold the counts, 4.4 the algebraic limits, 5 the full arithmetic trace, 7 the method
selection table, 8 the diagnostic ladder, 9 the cost model, 11 the interview drills.

**Takeaway:** A small trainable tensor does not imply a small training footprint, and it does not imply a strong
capability to change behaviour; both must be argued separately.

## 2. Concept Map / Mental Model

```text
 acceptance criteria + baseline measurement
        |
        v
 does prompting or retrieval already pass the gate?  -- yes --> stop, ship the cheaper control
        | no
        v
 enumerate candidate target modules with exact shapes
        |
        v
 choose rank r, scale convention, precision, optimizer
        |
        v
 derive P_adapter, resident state, merge cost, serving delta
        |
        v
 train on authorized, group-split data with fixed token budget
        |
        v
 paired evaluation: base vs adapted, identical prompts and decoding, per-slice denominators
        |
        v
 gate: quality, safety, cost, compatibility, rollback -- all pass?
        | no                                    | yes
        v                                       v
 investigate or revert to baseline        versioned artifact + pinned base hash
```

The frozen weight supplies the representation; the adapter supplies a low-rank additive correction to selected linear
maps. Everything else in this card follows from that one sentence plus accounting.

Four claims are often conflated and must be kept apart:
1. **Algebraic claim.** The update BA has rank at most r. This is provable and cheap to check.
2. **Capacity claim.** A rank-r update family can represent some useful task changes and cannot represent others exactly.
   This is a statement about the hypothesis class, not about any particular trained adapter.
3. **Empirical claim.** On this dataset, with this budget, the adapted model scores higher on this metric. This requires
   an experiment and a denominator.
4. **Operational claim.** The artifact can be served, versioned, rolled back and audited at acceptable cost. This
   requires a runtime and a lifecycle, neither of which the algebra provides.

A fifth common confusion treats the adapter as an isolation boundary. It is not. Two tenants' adapters on one base share
the base weights, the tokenizer, the runtime, and any prompt cache. Isolation is an authorization and routing property.

**When LoRA is the wrong tool, stated before the mechanics:**
- The missing capability is knowledge that changes weekly. Weights are not a freshness mechanism; retrieval is.
- The required change is a new modality or a new tokenizer. Adapters do not add input channels or vocabulary entries.
- The task requires a behaviour the base cannot produce at any sampling temperature, and the corpus is large enough to
  justify continued pretraining. A low-rank correction to a frozen representation has a ceiling.
- The base cannot legally be frozen in memory, or the license forbids derivative artifacts. Check before spending.
- Latency budget is already saturated and the adapter cannot be merged, so every token pays a second kernel launch.

**Mental model that survives contact with production:** a LoRA adapter is a signed, versioned patch to a specific base
checkpoint, expressed in a constrained function family, trained on a specific data snapshot, validated on a specific
population, and only meaningful together with all four of those identifiers.

**Takeaway:** Choose adaptation only after the cheaper control has been measured and failed, and record which of the
four claims your evidence actually supports.

## 3. Foundational Definitions

| Term | Definition and boundary |
|---|---|
| Base weight `W` | A frozen dense linear map with shape `d_out x d_in`; it is never updated during adapter training. |
| Down factor `A` | Trainable matrix of shape `r x d_in`; it maps an input into the low-rank coordinate space. |
| Up factor `B` | Trainable matrix of shape `d_out x r`; it maps low-rank coordinates back to output width. |
| Rank `r` | Positive integer bounding the rank of the update; an upper bound on expressiveness, not a achieved quantity. |
| Scale `alpha` | Dimensionless hyperparameter entering the update as `alpha/r` in the common convention; other conventions exist. |
| Target module | A named linear layer that receives a factor pair, e.g. `q_proj` or `down_proj`; identity is per tensor, not per name string. |
| Trainable count `P` | Number of scalar parameters that receive optimizer updates, including any deliberately unfrozen extras. |
| Frozen parameter | A parameter that receives no weight gradient; it may still receive input gradients during backpropagation. |
| Activation | Any forward tensor retained for the backward pass, or recomputed when checkpointing is enabled. |
| Optimizer state | Persistent per-parameter tensors such as AdamW's first and second moments; usually larger than the adapter. |
| Merge | Adding the scaled update into a compatible base weight to produce a single dense matrix. |
| Effective weight | `W + (alpha/r) B A`, the matrix actually applied at inference in the unmerged case. |
| Adapter artifact | Factors plus configuration plus provenance metadata; unusable without a compatible base and tokenizer. |
| Group | The independent unit of evaluation, such as a customer, document set, or conversation; not a random row. |
| Leakage | Any dependence between training and evaluation that a deployed model would not have. |
| Peak resident memory | Maximum simultaneous device allocation during the step, including allocator slack, not the sum of tensors. |

| Symbol | Meaning | Units | Shape / type | Valid range |
|---|---|---|---|---|
| `d_in` | Input width of a target projection | scalars | integer | ≥ 1 |
| `d_out` | Output width of a target projection | scalars | integer | ≥ 1 |
| `r` | Adapter rank | dimensionless | integer | 1 ≤ r ≤ min(d_in, d_out) |
| `alpha` | Scaling hyperparameter | dimensionless | real | > 0 typical |
| `s` | Effective scale, `alpha/r` by convention | dimensionless | real | > 0 |
| `P_adapter` | Trainable scalar count | scalars | integer | ≥ 0 |
| `N_base` | Frozen base parameter count | scalars | integer | ≥ P_adapter |
| `T` | Supervised target tokens in a step | tokens | integer | ≥ 1 |
| `b_w` | Bytes per stored weight element | bytes | 1, 2, 4 | per precision mode |
| `M_peak` | Peak resident device memory | bytes | integer | ≥ sum of live tensors |
| `F_fwd` | Forward multiply-accumulate count per token | MACs | integer | ≥ 0 |

Two conventions must be stated explicitly before any comparison, because silently mixing them invalidates every
conclusion drawn from the numbers:
- **Scaling.** Whether the update is `(alpha/r) B A`, `alpha B A`, or `(alpha/sqrt(r)) B A`. The third appears in the
  rank-stabilised variant and changes the training dynamics at large r.
- **Orientation.** Whether stored weights are `[d_out, d_in]` with `y = x W^T` or `[d_in, d_out]` with `y = W x`.
  A transposed target still produces a square adapter count when `d_in = d_out`, which is exactly why the bug hides in
  square projections and surfaces only on rectangular ones.

Parameter counts have units of scalar elements. Memory has units of bytes, MiB or GiB; a count multiplied by bytes per
element is storage only for the tensor named, never for the process.

**Takeaway:** Write the scale convention, the weight orientation and the target list into the run configuration before
the first training step; retrofitting them to a completed run is not possible.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**Anatomy of the effective map.**

```text
FORMULA [F1]: h = W x + s B A x,        s = alpha / r
WHERE: W in R^(d_out x d_in) frozen; B in R^(d_out x r); A in R^(r x d_in);
       x in R^(d_in); s is a positive scalar; r >= 1.
CONSTRAINT: The two terms must be computed in the same dtype and then summed, and the
            scale must follow the implementation's documented convention.
NUMERICAL EXAMPLE: d_in = d_out = 4096, r = 16, alpha = 32 gives s = 2. The update term
            has shape 4096 x 4096 and rank at most 16.
INTERPRETATION: The adapter does not replace the projection; it adds a residual path
            whose output direction set has at most r dimensions.
```

**Trainable count for one pair.**

```text
FORMULA [F2]: P_pair = r * d_in + d_out * r = r * (d_in + d_out)
WHERE: A contributes r*d_in entries; B contributes d_out*r entries; no bias, no extra
       trainable tensor is included.
CONSTRAINT: One independent pair per tensor. Tied embedding/output weights, if targeted,
           are one allocation and count once.
NUMERICAL EXAMPLE: r = 16, d_in = d_out = 4096: P_pair = 16 * 8192 = 131,072.
INTERPRETATION: 131,072 scalars versus 16,777,216 in the frozen matrix, a ratio of
           1/128, which is 0.78%.
```

**Derivation of the ratio, which is more useful than the count.**

```text
FORMULA [F3]: P_pair / (d_in * d_out) = r * (d_in + d_out) / (d_in * d_out)
                                        = r * (1/d_out + 1/d_in)
WHERE: all symbols as F2; the ratio is dimensionless.
CONSTRAINT: Exact for a single dense target with no bias and no shared storage.
NUMERICAL EXAMPLE: square 4096 at r = 16: 16 * (1/4096 + 1/4096) = 16/2048 = 0.0078125.
           Rectangular k_proj 4096 -> 1024 at r = 16: 16 * (1/1024 + 1/4096)
           = 16 * 0.001220703125 = 0.01953125, i.e. 1.95%.
INTERPRETATION: The relative cost of an adapter is larger on skinny matrices. The k and v
           projections of a grouped-query model are exactly where the adapter is least
           economical, which is a reason they are sometimes skipped.
```

**Gradients, which explain the initialisation rule.**

```text
FORMULA [F4]: dL/dB = s * g (A x)^T
              dL/dA = s * B^T g x^T
              dL/dx = W^T g + s * A^T B^T g
WHERE: g = dL/dh in R^(d_out) is the upstream gradient; L is the scalar loss; ^T is
       transpose; all products are ordinary matrix-vector products.
CONSTRAINT: Ignores bias terms, non-linearities between layers, and any dtype casting.
NUMERICAL EXAMPLE: With B initialised to zero, dL/dB = s * g (Ax)^T is generally
       non-zero while dL/dA = 0 exactly at the first step.
INTERPRETATION: Zero-initialising B makes the adapter a no-op at step zero, which keeps
       the initial loss equal to the base loss, but A receives no gradient until B has
       moved. The pair's effective learning is offset by one step; this is benign at
       thousands of steps and material only in very short runs.
```

**Validity condition for 4.1.** All four formulas assume a dense linear map, a single independent factor pair, exact
real arithmetic for the algebraic claims, and a scaling convention of `alpha/r`. If any of those fail, recompute rather
than reuse.

### 4.2 General case

**Several targets, several blocks.**

```text
FORMULA [F5]: P_total = sum over i of r_i * (d_in_i + d_out_i)  +  P_extra
WHERE: i indexes distinct target tensors across all blocks; P_extra counts additional
       trainable scalars such as biases, layer norms or a targeted lm_head.
CONSTRAINT: Each tensor counted once; do not multiply by layer count twice; do not count
       a shared/tied tensor as separate per-block allocations.
NUMERICAL EXAMPLE: 32 blocks, targets q, k, v, o, gate, up, down; d = 4096, kv = 1024,
       ffn = 11008, r = 16:
         q,o : 16 * (4096 + 4096) = 131,072 each
         k,v : 16 * (4096 + 1024) =  81,920 each
         gate,up : 16 * (4096 + 11008) = 241,664 each
         down : 16 * (11008 + 4096) = 241,664
       per block = 131,072*2 + 81,920*2 + 241,664*3 = 425,984 + 724,992 = 1,150,976
       P_total = 1,150,976 * 32 = 36,831,232
INTERPRETATION: About 36.8 million trainable scalars, or 0.635% of a 5.80 billion
       parameter base. The three MLP targets account for 23,199,744 of the total and
       the four attention targets for 13,631,488.
```

**Share of the update budget by module family.** Using the per-block numbers above: MLP targets hold 724,992 and
attention targets hold 425,984, so 63.0% of the adapter sits in the MLP and 37.0% in attention. Attention is only 23.7%
of the frozen parameters (41,943,040 of 177,209,344 per block), so a target list chosen "for coverage" spends most of
its budget where the frozen model has least of its weight. This is not an argument either way; it is a fact to state
when someone claims a target list is balanced.

**Including the language-model head.**

```text
FORMULA [F6]: P_head = r_head * (V + d),   d_out = V
WHERE: V is vocabulary size; the head is targeted only if output-side behaviour must
       change, which is rare for style or format adaptation.
NUMERICAL EXAMPLE: V = 32,000, d = 4096, r_head = 16: 16 * 36,096 = 577,536, which
       raises P_total from 36,831,232 to 37,408,768 (+1.57%).
INTERPRETATION: The head is cheap in count because it is rank-limited on the output side,
       but it changes token-level preferences directly, which is a behavioural change
       most adaptation tasks do not want.
```

**General derivation step.** F5 is F2 summed over an index set. There is no interaction term, because factor pairs are
allocated independently and no parameter is shared between targets. The only way F5 goes wrong is an index-set error:
duplicating a module that already appears under two names, omitting a block, or forgetting a tied tensor.

**Validity condition for 4.2.** Distinct allocations, one pair per tensor, no hidden trainable modules created by the
framework (some frameworks add trainable scaling vectors or bias terms silently), and a rank constant across targets.
Mixed ranks are legal; then the sum uses each target's own `r_i`.

### 4.3 Variants

**Variant 1: memory accounting, which is what actually constrains the run.**

```text
FORMULA [F7]: M_peak = M_base + M_adapter + M_grad + M_opt + M_act + M_temp + M_alloc
WHERE: each term is bytes resident on the same device at the same instant; M_act is the
       activation peak, M_temp the workspace for fused kernels and communication
       buffers, M_alloc the allocator's reservation above live tensors.
CONSTRAINT: Terms must be measured at the same moment. A sum of separately measured
       maxima overestimates; a sum of averages underestimates.
NUMERICAL EXAMPLE: M_adapter at 36,831,232 parameters in bf16 = 73,662,464 bytes
       = 70.25 MiB. Adding gradient in bf16 and two AdamW moments in fp32 gives
       2 + 2 + 4 + 4 = 12 bytes per parameter = 441,974,784 bytes = 421.5 MiB.
       Adding an fp32 master copy raises the factor to 16 bytes = 562.0 MiB.
INTERPRETATION: Trainable state is 3.81% (12 B/param) to 5.08% (16 B/param) of the
       11,072 MiB frozen base in bf16. The adapter is not the memory problem; the base and
       the activations are.
```

**Variant 2: parameter-free scale changes and their cost.**

```text
FORMULA [F8]: s = alpha / r          (original convention)
              s = alpha / sqrt(r)    (rank-stabilised convention)
WHERE: s multiplies B A in the forward pass; both are dimensionless.
NUMERICAL EXAMPLE: alpha = 32 held fixed. Original: r=8 -> 4.000, r=16 -> 2.000,
       r=32 -> 1.000. Rank-stabilised: r=8 -> 11.314, r=16 -> 8.000, r=32 -> 5.657.
INTERPRETATION: Under alpha/r, eight-folding the rank divides the explicit scale by
       eight, so a "rank sweep" with fixed alpha changes two things at once. Under
       alpha/sqrt(r) the scale changes by 2.83x across the same sweep, which is why the
       variant is used when ranks above roughly 64 are compared.
```

**Variant 3: magnitude decomposition, which adds a row-vector.**

```text
FORMULA [F9]: W' = m * (W + s B A) / || W + s B A ||_c
WHERE: m in R^(d_out) is a trainable per-row magnitude; || . ||_c is the column-wise
       norm, i.e. the norm of each output row; * is row-wise scaling.
CONSTRAINT: Magnitude vectors are extra trainable scalars counted in P_extra; the
       decomposition is not available in every runtime.
NUMERICAL EXAMPLE: over the seven target families above, the added magnitudes are
       4096 + 1024 + 1024 + 4096 + 4096 + 4096 + 4096 = 22,528 per block, so
       22,528 * 32 = 720,896 extra scalars, +1.96% on 36,831,232.
INTERPRETATION: The variant separates "how much" from "which direction", at a cost below
       two percent of the adapter. Whether that helps is an empirical question about a
       specific task, not a general improvement.
```

**Variant 4: scaled-vector methods, for contrast on cost.**

```text
FORMULA [F10]: P_ia3 = sum over targets of d_target,  e.g. d_kv for keys and values
WHERE: each target learns one multiplicative vector rather than a matrix pair.
NUMERICAL EXAMPLE: one d_kv = 1024 vector for keys, one for values, and one d_ffn = 11008
       vector for the MLP: 1024 + 1024 + 11008 = 13,056 per block, 417,792 total.
INTERPRETATION: 1.1% of the LoRA count and correspondingly narrower in what it can
       express: it rescales existing directions and cannot rotate them. Choose it when
       the expected change is a reweighting, not a new mapping.
```

**Variant 5: prompt-space methods, where cost moves to the context window.**

```text
FORMULA [F11]: P_soft = n_soft * d ;  KV_per_request = 2 * L * n_kv * n_soft * b_w
WHERE: n_soft is the number of soft prompt tokens; n_kv is the per-token key/value width
       summed over layers; b_w bytes per element.
NUMERICAL EXAMPLE: n_soft = 100, d = 4096: P_soft = 409,600 trainable scalars. With
       L = 32, n_kv = 1024, bf16: 2 * 32 * 1024 * 100 * 2 = 13,107,200 bytes
       = 12.5 MiB of extra cache per concurrent request.
INTERPRETATION: The parameter count looks excellent and the serving cost is per-request
       and per-token-of-context. At 64 concurrent requests that is 800 MiB of cache that
       buys no capability beyond the prompt, and it consumes context the task needed.
```

**Validity condition for 4.3.** F7 is an accounting identity only when the terms are co-resident. F8–F11 describe
alternatives under their own conventions; comparing their parameter counts without also comparing their serving cost,
their expressiveness and their evaluation burden is the most common false comparison in PEFT selection.

### 4.4 Limits and counterexamples

**Limit 1: rank is an upper bound, and the bound is on the update only.**

```text
FORMULA [F12]: rank(B A) <= min(r, d_in, d_out);  rank(W + s B A) can be min(d_in, d_out)
WHERE: rank is the algebraic rank over the reals.
CONSTRAINT: Exact statement about the additive update; says nothing about W.
NUMERICAL EXAMPLE: W = I_2, B A = diag(1, 0). The effective matrix diag(2, 1) has rank 2
       with r = 1. Conversely W = I_2, B A = diag(-1, 0) gives diag(0, 1), rank 1.
INTERPRETATION: A rank-one adapter can make a matrix full rank or make it singular. The
       phrase "low-rank model" is therefore wrong; the model stays full rank and only the
       correction is constrained.
```

**Limit 2: the crossover at which the adapter is larger than the matrix.**

```text
FORMULA [F13]: r * (d_in + d_out) >= d_in * d_out  =>  r >= d_in*d_out/(d_in+d_out)
WHERE: the right-hand side is half the harmonic mean of the two widths.
CONSTRAINT: Applies per target with that target's own widths.
NUMERICAL EXAMPLE: square 4096 -> 2048; k_proj 4096 x 1024 -> 819.2; down_proj
       11008 x 4096 -> 2985.22.
INTERPRETATION: Ranks below a few hundred are economical on every shape. A rank of 1024 on
       a 4096-to-1024 projection trains more scalars than the matrix contains and is a
       signal that full fine-tuning or a different method was the right choice.
```

**Limit 3: rotation invariance means factors are not interpretable.**

```text
FORMULA [F14]: for any invertible R in R^(r x r):  B A = (B R)(R^-1 A)
WHERE: both factors are transformed; the product is unchanged.
CONSTRAINT: Exact for real invertible R; requires the pair to be trained jointly.
NUMERICAL EXAMPLE: r = 2, R = [[0,1],[1,0]] swaps the two adapter columns of B and the two
       rows of A; the product and therefore every output is identical.
INTERPRETATION: Two runs with identical loss can have completely different factor values.
       Inspecting A or B to explain behaviour is invalid; inspect the product or the
       outputs. It also means factor averaging across runs is undefined.
```

**Limit 4: a low-rank approximation can be good in Frobenius norm and catastrophic on a direction.**

```text
FORMULA [F15]: for target D with SVD D = sum_i sigma_i u_i v_i^T, the best rank-k
               approximation in Frobenius norm keeps the k largest sigma_i, and
               ||D - D_k||_F^2 = sum of the discarded sigma_i^2.
WHERE: sigma_i are singular values in descending order; k >= 1.
CONSTRAINT: Optimality is with respect to unweighted Frobenius error over all inputs,
       not with respect to the task loss or the data distribution.
NUMERICAL EXAMPLE: D = diag(6, 3, 1). ||D||_F^2 = 36 + 9 + 1 = 46. Rank 1 keeps 36, i.e.
       78.26% of squared energy; residual norm sqrt(10) = 3.162. Rank 2 keeps 45, i.e.
       97.83%; residual norm 1.
INTERPRETATION: At rank 2, the input (0,0,1) is mapped to zero while D maps it to (0,0,1):
       a 97.83% energy retention with 100% error on that input. Aggregate matrix error is
       not evidence about a slice.
```

**Limit 5: the objective that training minimises is not the approximation error.**

Nothing in F15 says gradient descent on a language-modelling loss finds the truncated SVD of the "true" update. The
true update is itself unknown, it depends on the data, and it is not the target of the optimisation. F15 is an offline
diagnostic for the case where a full fine-tuning delta exists and you want to know what rank would have been needed to
match it; it is not a prediction of adapter quality.

**Counterexample to "adapter rank equals task complexity".** Two tasks of equal apparent difficulty can need different
ranks because the base representation already contains one and not the other. A formatting convention often lies in a
near rank-one direction because it is a token-distribution shift; a new decision rule over domain vocabulary may need a
much wider subspace. Rank is a property of the pair (task, base), not of the task alone, so a rank chosen from a paper
about a different base and a different task carries no information.

**Counterexample to "adapter parameters are free at inference".** Unmerged, every generated token executes the extra
rank-r products. For the worked model at r = 16 the per-token adapter cost is 36,831,232 MACs, i.e. 73.7 MFLOP against
11.60 GFLOP for the frozen forward path. That is 0.635% more arithmetic, but it arrives as 448 additional matrix
multiplications (seven targets, 32 blocks, two products per pair), each a thin kernel that cannot reach the arithmetic
intensity of the frozen products. The realised latency effect depends on batch shape and kernel fusion; the algebra is
exact, the latency is unmeasured here.

**Validity condition for 4.4.** F12 and F14 hold in exact arithmetic. F13 is an inequality on counts. F15 requires the
target matrix to be known and the error to be measured in a fixed norm. Any claim that crosses from these statements to
"so it will work on my task" is an empirical claim that needs its own evidence.

**Takeaway:** The algebra bounds the update, not the model; the count bounds the storage, not the memory; neither bounds
the task outcome.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE. The architecture below is a constructed 32-block decoder chosen so the arithmetic is
checkable by hand; it is not a claim about any shipped model. Every number is DERIVED from the stated assumptions.

**Assumptions.**
- 32 transformer blocks, hidden width `d = 4096`.
- Grouped-query attention with key/value width `kv = 1024` per block, query and output width 4096.
- MLP with `gate`, `up` of shape 4096 -> 11008 and `down` of shape 11008 -> 4096.
- Vocabulary 32,000, output head 4096 -> 32000, untargeted.
- No biases counted, no tied targets, no other trainable module.
- Adapter rank `r = 16`, `alpha = 32`, convention `s = alpha/r = 2.0`.
- Frozen base stored in bf16; factors stored in bf16 with fp32 AdamW moments.

**Step 1: frozen parameter count, for the denominator.**

```text
per block attention: 2*4096*4096 + 2*4096*1024
                   = 33,554,432 + 8,388,608 = 41,943,040
per block MLP:       3*4096*11008 = 135,266,304
per block total:     41,943,040 + 135,266,304 = 177,209,344
32 blocks:           177,209,344 * 32 = 5,670,699,008
head:                4096 * 32,000 = 131,072,000
N_base:              5,670,699,008 + 131,072,000 = 5,801,771,008
```

The MLP accounts for 76.33% of the per-block parameters and attention for 23.67%.

**Step 2: adapter count per block, by target.**

```text
q_proj    r*(4096 + 4096)  = 16 *  8,192 = 131,072
k_proj    r*(4096 + 1024)  = 16 *  5,120 =  81,920
v_proj    r*(4096 + 1024)  = 16 *  5,120 =  81,920
o_proj    r*(4096 + 4096)  = 16 *  8,192 = 131,072
gate_proj r*(4096 + 11008) = 16 * 15,104 = 241,664
up_proj   r*(4096 + 11008) = 16 * 15,104 = 241,664
down_proj r*(11008 + 4096) = 16 * 15,104 = 241,664
                              per block = 1,150,976
```

**Step 3: totals.**

```text
P_total   = 1,150,976 * 32 = 36,831,232
P / N_base= 36,831,232 / 5,801,771,008 = 0.0063483 -> 0.6348%
```

**Step 4: storage of each training artifact.**

```text
adapter weights, bf16 : 36,831,232 * 2 =  73,662,464 B =  70.25 MiB
adapter gradients, bf16: 36,831,232 * 2 =  73,662,464 B =  70.25 MiB
AdamW m, v, fp32 each : 36,831,232 * 8 = 294,649,856 B = 281.00 MiB
total trainable state : 36,831,232 * 12 = 441,974,784 B = 421.50 MiB
frozen base, bf16     : 5,801,771,008 * 2 = 11,603,542,016 B = 10.807 GiB
trainable share of base: 421.50 MiB / (10.807 * 1024 MiB) = 3.81%
```

**Step 5: what a single 8192-token sequence costs on top.**

```text
KV cache per token: 2 * 32 * 1024 * 2 B = 131,072 B = 128 KiB
KV cache, 8192 tokens, one sequence: 128 KiB * 8192 = 1,048,576 KiB = 1.000 GiB
activations, batch 2, bf16, no checkpointing, LOWER BOUND of one width-d tensor per block:
  2 * 32 * 4096 * 8192 * 2 B = 4,294,967,296 B = 4.000 GiB
a real step retains four to six such tensors per block, so 16 to 24 GiB, which is the term
that decides whether the run fits; with checkpointing only the network input is retained
  (1.000 GiB for the same batch) and each block's internals exist transiently on recompute
```

**Independent arithmetic re-check.** The per-block total is recomputed from the ratio form F3 applied to each
target with its own dense count, instead of from the per-target products:

```text
q: 16,777,216 * (16 * (1/4096 + 1/4096)) = 16,777,216 * 0.00781250 = 131,072.0
k:  4,194,304 * (16 * (1/1024 + 1/4096)) =  4,194,304 * 0.01953125 =  81,920.0
v:  4,194,304 * 0.01953125                                        =  81,920.0
o: 16,777,216 * 0.00781250                                        = 131,072.0
gate, up, down: 45,088,768 * (16 * (1/4096 + 1/11008))
   1/4096 = 0.000244140625 ; 1/11008 = 0.0000908430...
   sum = 0.0003349836... ; * 16 = 0.005359738... ; * 45,088,768 = 241,664.0
per block = 131,072.0*2 + 81,920.0*2 + 241,664.0*3 = 1,150,976  (matches F5)
total     = 1,150,976 * 32 = 36,831,232
```

The re-check agrees to the last scalar. It also exposes the error mode: a family-level ratio applied to a family total
is invalid, because the ratio depends on each target's own pair of widths. Multiplying the attention family's dense
count 41,943,040 by a blended ratio of 0.03125 would give 1,310,720 instead of the correct 425,984, an error of 3.1x,
and the mistake survives every dimensional check because both sides are scalar counts.

**A second, sharper counterexample to the ratio shortcut.** Suppose a target is 11008 wide in and 4096 wide out, and
someone uses `2/d_out` as an approximation to `(1/d_in + 1/d_out)`, reasoning that the widths are roughly comparable.
The approximation gives `2/4096 = 0.00048828125` against the true `0.00033498363`, an overestimate of 45.8%; at r = 16
that is 110,592 phantom parameters on each targeted MLP matrix, or 10,616,832 across the 96 such matrices in this model.
Approximating a ratio is safe only when `d_in` and `d_out` differ by less than about a factor of two.

**Step 6: merge cost, if the adapter is folded into the base.**

```text
merging one target costs about 2 * r * d_in * d_out FLOPs (one scaled product plus an add)
per block: 2 * 16 * 177,209,344 = 5,670,699,008 FLOPs
all blocks: 5,670,699,008 * 32 = 181,462,368,256 FLOPs = 181.46 GFLOP
```

181.46 GFLOP once per artifact is negligible against training, and it is not negligible against a request path: this is
an offline operation and must never run per request.

**Step 7: serving comparison.**

```text
merged artifact   : one dense matrix per target, no extra kernel, 10.807 GiB in bf16
unmerged artifact : base 10.807 GiB + adapter 70.25 MiB = 10.876 GiB
artifact ratio    : 11,603,542,016 / 73,662,464 = 157.5x
extra arithmetic  : 2 * 36,831,232 = 73,662,464 FLOP per token (adapter path, all layers)
base arithmetic   : 2 * 5,801,771,008 = 11,603,542,016 FLOP per token (ignoring attention scores)
share             : 0.635%
```

**Sensitivity note.** Holding the target list fixed and moving rank:

```text
r = 4  : P =  9,207,808   (0.1587% of base)   state at 12 B = 105.4 MiB   s = 8.000
r = 8  : P = 18,415,616   (0.3174%)           state at 12 B = 210.8 MiB   s = 4.000
r = 16 : P = 36,831,232   (0.6348%)           state at 12 B = 421.5 MiB   s = 2.000
r = 32 : P = 73,662,464   (1.2697%)           state at 12 B = 843.0 MiB   s = 1.000
```

Two facts from the table. First, doubling rank doubles the count exactly, because F5 is linear in `r`; there is no
economy of scale. Second, holding `alpha` fixed while doubling rank halves the effective scale, so a rank sweep with a
fixed `alpha` is a joint change of capacity and step size. Choose one of two protocols and state it: hold `s` constant
by scaling `alpha` with `r`, or hold `alpha` constant and accept that scale is part of the treatment.

**A second sensitivity that is easy to miss.** Restricting the target list from seven modules to `q` and `v` only, at
`r = 8`, gives `(8*8192 + 8*5120) * 32 = (65,536 + 40,960) * 32 = 3,407,872` trainable scalars, 9.3% of the
seven-target `r = 16` count. Whether that is 9.3% as useful is an evaluation question, and the only honest answer
without a run is "unknown".

**Decision implied by the example.** At this scale the memory argument does not decide anything: 421.5 MiB of trainable
state against a 10.807 GiB base means the base and the activations dominate. The decision must therefore be argued on
quality, on artifact management, and on whether the serving path can afford the unmerged kernel path.

**Takeaway:** Recompute the count per target, never per family; and state whether your rank sweep also changed the scale.

## 6. Visual / Geometric Intuition

```text
                    output space R^d_out
                            |
                 image(W)   |  .......... full-dimensional
                            |
   x ---> A ---> z in R^r --+--> B z  lies in a subspace of dimension <= r
     \                      |
      \--> W x ------------>+--> plus the frozen contribution
```

Read the diagram as two parallel paths that are summed. The upper path passes through a bottleneck of width `r`, so
everything it can contribute lives in the at-most-`r`-dimensional column space of `B`. The lower path is unrestricted.
The sum is what the layer computes.

**Geometric statements that follow directly.**
- The adapter's reachable outputs, with `x` fixed, form an affine set of dimension at most `r` inside the output space.
- The adapter cannot move an output in a direction orthogonal to `span(B)`. If the required correction on your task lies
  mostly outside that span, more training steps will not help; the direction is unreachable at that rank.
- `A` decides which input directions are distinguished inside the bottleneck; `B` decides where they are sent. Swapping
  which factor is initialised to zero changes the first-step behaviour, not the reachable set.
- The parameterisation is redundant: `r` columns of `B` and `r` rows of `A` can be combined with any invertible
  `r x r` matrix and the product is unchanged. The reachable set has dimension `r*(d_in+d_out) - r^2`, which is the
  number of free parameters after quotienting by that symmetry. For `r = 16` and a square 4096 target that is
  `131,072 - 256 = 130,816` effective degrees of freedom, and for the `k_proj` shape `81,920 - 256 = 81,664`.

**A useful mental picture for target selection.** The frozen base already contains many directions. An adapter on a
target can only reweight and rotate the directions that reach that target. Adapting `q` and `k` changes where attention
looks; adapting `v` and `o` changes what information is carried and how it is recombined; adapting the MLP changes the
per-token feature transform. These are different interventions, and a target list is a hypothesis about which of them
the task needs.

```text
   layer i-1 output
        |
        +--> q_proj (+LoRA) --> queries ---+
        +--> k_proj (+LoRA) --> keys -------+--> softmax --> weights
        +--> v_proj (+LoRA) --> values -----+
                                            |
        <-- o_proj (+LoRA) <----------------+
        |
        +--> gate/up (+LoRA) --> activation --> down (+LoRA) --> residual add
```

**Why the composition is not a single low-rank map.** A stack of adapted layers passes through softmax normalisation and
elementwise non-linearities. Those operations are not linear, so the end-to-end function of a LoRA-adapted model is not
`W + s B A` at the model level. Only the per-layer weights have that form. Any argument of the shape "the whole model is
low rank therefore it cannot learn X" is a category error; the correct statement is narrower and applies layerwise.

**An asymmetry worth remembering.** A rank-one update can raise a matrix's rank to full (add `diag(1,0)` to the
identity and the result has rank two) and can also lower it (add `diag(-1,0)` and the result has rank one). The adapter
is therefore not a "capacity reduction" of the base in any monotone sense; it is an additive perturbation of bounded
dimension.

**Takeaway:** Picture a narrow residual channel beside a wide frozen channel; the channel width limits what the adapter
can add, never what the base already does.

## 7. Comparison / Selection Table

| Option | Choose when | Parameter count for the worked model | Disqualifier | Fallback |
|---|---|---|---|---|
| Prompt engineering only | Task is controllable with instructions and the base already has the capability | 0 | Repeated instructions still fail on held-out groups | Adapter with a paired evaluation |
| Retrieval | Facts change, must be cited, or are tenant-specific | 0 trainable | Source quality or authorization cannot be assured | Fix retrieval; do not encode facts in weights |
| Full fine-tuning | The needed change is broad, data is plentiful, and optimizer memory is affordable | 5,801,771,008 | 86.45 GiB of fp32 state at 16 B/param exceeds the device budget | Adapter, or sharded full fine-tuning |
| LoRA on attention only (`q`,`v`) | The expected change is where attention looks, and budget is tight | 3,407,872 at r=8 | MLP-side behaviour is the actual deficit | Extend the target list |
| LoRA on all seven projections | The deficit is unknown or spans formatting and reasoning | 36,831,232 at r=16 | Rank and budget pushed until counts exceed the matrix | Reduce targets or rank, keep evaluation fixed |
| QLoRA (quantized base) | Base storage is the binding constraint | Same adapter count, base 4x smaller in weight bytes | Quantization degrades required precision or kernels are unavailable | bf16 base with smaller adapter |
| Adapter with magnitude vector | Direction and magnitude of the update need separate control | 37,552,128 at r=16 (+1.96%) | Runtime does not support the decomposition | Plain LoRA with tuned scale |
| Scaled-vector method (IA3-like) | The change is a reweighting of existing directions | 417,792 | Needed change requires rotation, not scaling | Rank-4 LoRA |
| Soft prompt | No weight artifact is permitted and context is not scarce | 409,600 at 100 tokens, plus 12.5 MiB cache per request | Context window is the scarce resource, or prompt is ignored at long lengths | Adapter |
| Merged adapter at serving time | One adapter per deployed artifact, stable traffic mix | 0 extra at inference | Per-request adapter switching is required | Unmerged with pinned routing |
| Unmerged adapter at serving time | Many adapters, one base, independent lifecycles | 0 extra weights per request | Runtime cannot pin an adapter for a whole generation | Separate processes per adapter |

**How to read the table.** The count column is present only to stop the argument from being settled by the count. Two
options with counts three orders of magnitude apart can be the right choice in different situations, because the count
is not the objective. The objective is the decision: does the adapted model pass the gate at acceptable cost.

**A selection rule that survives review.** State the failing baseline measurement first, then state the smallest
intervention that could plausibly move it, then state the measurement that would show it failed. If the third statement
cannot be written, the first was not a measurement.

**Takeaway:** Select by disqualifier, not by parameter count; every row above is cheap in some dimension and expensive
in another.

## 8. Failure Modes & Edge Cases

Each entry is a HYPOTHETICAL diagnostic exercise. No incident is asserted and no measured outcome is implied. The
structure is: symptom → candidate cause → competing explanation → discriminating check → bounded correction →
prevention that survives the next run.

### F1. Trainable count is lower than the arithmetic predicts

- Symptom: the framework reports a trainable parameter count that does not match F5 for the intended target list.
- Candidate cause: a target name matched zero modules, or matched more modules than intended.
- Competing explanation: the framework adds its own trainable scalars, so the count is slightly higher rather than lower.
- Discriminating check: print every parameter tensor whose `requires_grad` is true, with name, shape and element count; sum
  the shapes and compare with the hand calculation per target.
- Correction: attach factors to explicit module objects obtained by walking named modules, not to substring matches.
- Prevention: assert the total count in the training entry point and fail fast when it differs from F5.
- Verification: the printed inventory sums to the asserted value and every target appears exactly once.

### F2. Out of memory before the first backward pass completes

- Symptom: allocation failure during model load or during the first forward pass, before any optimizer step.
- Candidate cause: the frozen base plus the framework's temporary loading peak, e.g. loading in fp32 and casting.
- Competing explanation: a long first sequence inflating activations, which would fail slightly later.
- Discriminating check: measure device allocation after load, after the first forward, and after the backward, separately.
- Correction: load in the target dtype directly, shard or offload the frozen base, or reduce the sequence length.
- Prevention: record a load-phase allocation budget as a test, and gate on it before training starts.
- Verification: the load-phase measurement and the step-phase measurement both sit under the recorded budget.

### F3. Out of memory during backward only

- Symptom: the step succeeds in evaluation mode but fails with gradients enabled, at a specific sequence length.
- Candidate cause: retained activations scaling with sequence length and microbatch, not with adapter size.
- Competing explanation: optimizer state allocated lazily at the first step, which would fail after the backward.
- Discriminating check: sweep sequence length at fixed microbatch and record peak; then sweep microbatch at fixed length.
- Correction: gradient checkpointing, smaller microbatch with gradient accumulation, or shorter packing units.
- Prevention: derive the activation budget from the maximum training sequence length and test at that length.
- Verification: the maximum configured length completes a step, and throughput is recorded for that configuration.

### F4. Loss falls smoothly while the held-out score does not move

- Symptom: training loss decreases by a large factor, and the paired evaluation shows no improvement over the base.
- Candidate cause: the adapter learned the prompt format or the label distribution of the training set only.
- Competing explanation: leakage inflating training-set-like evaluation, which would show the opposite pattern.
- Discriminating check: score the base and the adapter on the same groups with identical prompts and decoding; inspect
  per-slice deltas rather than the aggregate; check whether the training loss drop is concentrated in the first tokens.
- Correction: re-balance the data toward target tokens, mask prompt tokens, and re-evaluate on untouched groups.
- Prevention: define the evaluation before training and require a pre-registered minimum detectable effect.
- Verification: the per-slice table shows a delta with an interval that excludes zero on the slices that matter.

### F5. Adapter appears to have no effect at inference

- Symptom: outputs from the served model are byte-identical to the base model's outputs.
- Candidate cause: the runtime loaded the base only; the adapter was silently skipped or scaled to zero.
- Competing explanation: the adapter was trained on a different base revision, so its effect is small but nonzero.
- Discriminating check: log the base commit hash, adapter hash and the resolved scale at request time; compare a fixed
  prompt's logits in both configurations rather than eyeballing text.
- Correction: pin the adapter to the request for the whole generation, and assert non-zero scale in the request path.
- Prevention: a startup self-test that asserts a deterministic logit difference from the base on a fixed input.
- Verification: the self-test fails loudly when the adapter is removed from the configuration.

### F6. Quality drops after merging the adapter

- Symptom: the merged artifact scores lower than the unmerged pair on the same evaluation suite.
- Candidate cause: the update was added in a lower precision than it was trained in, and rounding was material.
- Competing explanation: the wrong scale or the wrong target was merged, e.g. a transposed matrix that only shows on
  rectangular projections.
- Discriminating check: compare merged and unmerged logits in fp32 on a fixed batch before any requantization; then
  compare per-target merge residuals.
- Correction: merge in higher precision, then cast the merged weights once; verify orientation against the runtime.
- Prevention: a merge unit test on a rectangular target, since square targets hide transposition errors.
- Verification: merged-versus-unmerged maximum absolute logit difference is under the stated tolerance.

### F7. Adapter works in training but the served model degrades under concurrency

- Symptom: single-request evaluation passes; under load, some responses show behaviour from a different adapter.
- Candidate cause: the adapter selection is resolved per token or per batch rather than per request.
- Competing explanation: a prompt or KV cache keyed without the adapter identity, so a cached prefix crosses adapters.
- Discriminating check: trace one request end to end and record the adapter identity at every step that touches state;
  replay with two adapters in parallel and check for cross-contamination in the cache keys.
- Correction: bind the adapter identity to the request and include it in every cache key, including prefix caches.
- Prevention: a concurrency test with two adapters and deliberately overlapping prompts, run in CI.
- Verification: the overlapping-prompt test produces each adapter's own behaviour for every request.

### F8. Quality is good but latency regresses beyond the budget

- Symptom: the adapted model passes quality gates and fails the serving latency gate.
- Candidate cause: the unmerged adapter adds kernel launches that dominate at small batch sizes.
- Competing explanation: the adapter increased memory pressure, reducing batch size and therefore throughput.
- Discriminating check: measure time per output token at fixed batch for base, unmerged and merged, with identical
  prompts; separately record achieved batch size in each configuration.
- Correction: merge the adapter, or batch requests by adapter identity so the extra kernels are amortised.
- Prevention: include a served-latency budget in the acceptance criteria before training, not after.
- Verification: the measured time per token at the production batch shape is within the pre-agreed budget.

### F9. Resumed training diverges from the pre-interruption curve

- Symptom: after a restart, the loss jumps or the run fails on the second step.
- Candidate cause: the checkpoint contains factors but not optimizer moments, scheduler position or scaler state.
- Competing explanation: the data loader restarted from the beginning, so the model sees the first examples twice.
- Discriminating check: inspect the checkpoint manifest for each state tensor, and compare the first post-resume step's
  loss against the last pre-interruption loss on the same batch.
- Correction: save and restore the full training state, and record the data position.
- Prevention: a resume drill executed before the long run, comparing the next step under a stated tolerance.
- Verification: the resume drill's loss difference is within tolerance and the step counter advances correctly.

### F10. Behaviour degrades on long inputs but not on short ones

- Symptom: slice metrics by input length show the improvement vanishing or reversing beyond a token threshold.
- Candidate cause: truncation during training removed the region where the target behaviour occurs.
- Competing explanation: the evaluation prompt template differs from training, so the model sees an unfamiliar format.
- Discriminating check: histogram training token lengths, count examples truncated, and inspect the decoded training
  examples near the maximum length; then compare the actual templates as token sequences.
- Correction: re-budget the sequence length or filter examples that cannot fit; align templates exactly.
- Prevention: report the truncated fraction per slice as a data-quality metric before training.
- Verification: the length-bucketed evaluation shows a delta on the longest bucket that meets the pre-agreed effect.

### F11. Training is unstable in the first hundred steps

- Symptom: loss spikes, gradient norm explodes, or non-finite values appear early.
- Candidate cause: the effective scale is too large for the chosen learning rate, for example after an `alpha` change.
- Competing explanation: a small number of malformed examples with extreme token distributions.
- Discriminating check: log per-step gradient norm, the maximum absolute value in each factor, and the batch identity
  at the spike; rerun with a sorted batch to see whether the spike follows the data or the configuration.
- Correction: lower the effective scale or the learning rate after isolating which one moved, and filter the outliers.
- Prevention: a short warm-up run with gradient clipping and finite checks as a gate before the full run.
- Verification: the first thousand steps show finite values and a decreasing gradient-norm median.

### F12. Aggregate metrics improve while a protected slice regresses

- Symptom: the mean score rises and one subgroup falls below its previous value or below a floor.
- Candidate cause: the training mixture under-represents the slice, so the adapter trades it away for the majority.
- Competing explanation: the slice's evaluation items were affected by a data change unrelated to adaptation.
- Discriminating check: per-slice numerator, denominator and interval for base and adapter; compare against the same
  items scored before the data change.
- Correction: reweight or augment the mixture, or exclude the adapter from the routes that serve the slice.
- Prevention: pre-register protected slices with minimum deltas and a blocking rule, before training.
- Verification: the slice table shows every protected slice at or above its floor with an interval.

### F13. Refusal or safety behaviour changes after adaptation

- Symptom: the adapted model complies with requests the base refused, or over-refuses benign ones.
- Candidate cause: the fine-tuning distribution contains examples whose completions conflict with the safety policy.
- Competing explanation: decoding settings changed between the two evaluations, which alone can move refusal rates.
- Discriminating check: hold decoding fixed and run a fixed safety probe set through base and adapter with blinded
  review; then rerun the adapter probe at the base's original decoding settings.
- Correction: remove conflicting examples, add policy-consistent examples, or reject the release.
- Prevention: a safety probe suite executed on every candidate adapter as a release gate.
- Verification: the blinded review shows no material increase in harmful compliance and no material over-refusal.

### F14. The adapter cannot be reproduced from its artifact

- Symptom: a retrained adapter from the stored configuration does not reproduce the deployed behaviour.
- Candidate cause: the configuration omits the target list, the scale convention, or the base revision.
- Competing explanation: nondeterministic kernels, which change the numbers but not the behaviour materially.
- Discriminating check: retrain twice from the artifact's metadata only and compare held-out scores and a fixed-batch
  logit fingerprint; separately rerun with deterministic kernels to bound the nondeterminism.
- Correction: complete the artifact metadata and pin the base by hash, not by name.
- Prevention: treat the artifact as unusable unless a from-scratch reproduction passes the fingerprint check.
- Verification: the reproduction run matches the recorded fingerprint within the stated tolerance.

**Takeaway:** Every entry above is decided by a measurement that separates two causes; changing a hyperparameter before
that measurement is a guess wearing a lab coat.

## 9. Complexity & Cost Analysis

**Per-token arithmetic.** For one token, one target contributes `r*(d_in+d_out)` multiply-accumulate operations for the
adapter path, against `d_in*d_out` for the frozen path. Summing over the seven targets of the worked model gives
36,831,232 MACs, or 73.66 MFLOP at two operations per MAC, against 11.60 GFLOP for the frozen forward path. The ratio of
arithmetic is exactly the ratio of parameter counts, 0.635%, because both are linear in the same shapes.

**Training arithmetic is a different quantity, and this is the most misreported number in PEFT.** A training step costs
roughly six operations per parameter per token for the full model (two for the forward, four for the backward). LoRA
does not remove the forward or most of the backward: the frozen weights still need input gradients to reach earlier
adapters, so `W^T g` is computed at every layer. What LoRA removes is the weight gradient for frozen parameters and the
optimizer state for them.

```text
ILLUSTRATIVE: 50,000 examples * 512 tokens * 3 epochs = 76,800,000 target tokens
full fine-tuning, 6 FLOP per parameter-token:
   6 * 5,801,771,008 * 76,800,000 = 2.673e18 FLOP
adapter-only accounting, 6 FLOP per adapter parameter-token:
   6 * 36,831,232 * 76,800,000    = 1.697e16 FLOP
ratio of the "training-only" terms: 0.635%
```

The 0.635% figure is the size of the term LoRA changes; it is not the speed-up. Amdahl's law applies: the forward and
the input-gradient backward remain, so the wall-clock saving is bounded by the fraction of time those consume. A run
that spends 85% of its time in frozen forward and backward can improve by at most about 15% even if the adapter terms
vanish, and in practice much less.

```text
FORMULA [F16]: speedup <= 1 / (f_frozen + f_adapter),  with f_frozen + f_adapter = 1
WHERE: f_frozen is the fraction of step time spent in work LoRA does not remove.
CONSTRAINT: Bound only; communication, launch overhead and memory pressure can make
            the realised speed-up lower, and can in principle make it negative.
NUMERICAL EXAMPLE: f_frozen = 0.85 gives 1/0.85 = 1.176, i.e. at most +17.6% throughput.
INTERPRETATION: Quote this bound before promising a training-cost reduction, and measure
            the realised value on the target hardware.
```

**Memory is where the saving is real.** Full fine-tuning at 16 bytes per parameter for weights, master weights, momenta
and gradients needs 92.83 GB, or 86.45 GiB, for the worked base. Adapter training needs the bf16 base at 10.807 GiB
plus 562 MiB of trainable state, i.e. about 11.36 GiB, a 7.6x reduction in the parameter-related term. That is the
honest headline: LoRA buys memory, not arithmetic.

```text
full FT: 5,801,771,008 * 16 B = 92,828,336,128 B = 86.453 GiB
LoRA   : 11,603,542,016 B (base bf16) + 589,299,712 B (adapter at 16 B/param) = 12,192,841,728 B
         = 11.356 GiB
ratio  : 86.453 / 11.356 = 7.61x
```

**Storage across adapters is where the lifecycle argument sits.**

```text
one bf16 adapter artifact        : 70.25 MiB
250 tenant adapters              : 250 * 70.25 MiB = 17,562.5 MiB = 17.15 GiB
250 tenant full copies in bf16   : 250 * 10.807 GiB = 2,701.75 GiB
```

The adapter path stores every tenant in less space than two base copies, which is the entire economic reason
multi-adapter serving exists. The counterweight is that every tenant adapter must be evaluated, versioned and rolled
back individually, so the operational cost per tenant is not small even though the storage is.

**Communication and parallelism.** Adapter training still shards the frozen base across devices or replicates it. In data
parallelism the adapter gradients are all-reduced and are small, but the frozen parameters are usually not reduced at
all, so the all-reduce volume falls in proportion to the adapter count rather than to the model size. In tensor
parallelism the factors must be sharded consistently with the target's sharding: an `o_proj` row-sharded across devices
needs `B` split along its output rows, and `A` replicated. A mismatch produces plausible shapes and wrong results, and
only a fixed-input comparison against the unsharded merge catches it.

**Merge cost.** 181.46 GFLOP per artifact for the worked model, computed in Section 5. It is a one-off cost per
artifact version, and it must be repeated whenever the base or the adapter changes. Budget it as an artifact-build step
with its own timeout, not as a request-path operation.

**Evaluation cost is the term that decides projects.** A paired evaluation that produces per-slice intervals on a
protected slice needs enough items per slice for the interval to be narrower than the effect you intend to detect. With
binary per-item outcomes, the standard error of a proportion near 0.8 with n items per arm is about
`sqrt(0.8*0.2/n)`; at n = 400 that is 0.02, so the half-width of a 95% interval is roughly 0.039. Detecting a 3-point
improvement therefore needs more than 400 items per arm per slice, and typically several thousand for a stable answer.
Evaluation compute is frequently larger than the adapter training compute for small adapters, and it is not optional.

**Cost table for the worked example, all ILLUSTRATIVE.**

| Term | Quantity | Note |
|---|---|---|
| Trainable parameters | 36,831,232 | F5, seven targets, r = 16 |
| Adapter weights, bf16 | 70.25 MiB | storage per artifact |
| Trainable state, 16 B/param | 562.0 MiB | weights, grads, fp32 master, two moments |
| Frozen base, bf16 | 10.807 GiB | dominant resident term |
| Merge compute | 181.46 GFLOP | once per artifact |
| Extra inference FLOPs | 73.66 MFLOP per token | 0.635% of the frozen forward |
| Extra KV cache | 0 | adapters add no cache |
| Evaluation items | thousands per slice | decided by the required interval width |

**Takeaway:** LoRA reduces parameter state by roughly 7.6x and inference arithmetic by 0.635%; it barely reduces training
arithmetic, so any claim of proportional speed-up is wrong.

## 10. Key Theorems / Results Reference

**DERIVED — rank of a product.** For conformable real matrices, `rank(B A) <= min(rank B, rank A)`. With `B` of shape
`d_out x r` and `A` of `r x d_in`, the product has rank at most `r`. This is the algebraic content of the method and it
is checkable by computing singular values of the product.

**DERIVED — the effective matrix need not be low rank.** `rank(W + s B A)` can be any value up to the full rank of `W`,
as the two 2x2 examples in Section 4.4 show. Therefore "LoRA produces a low-rank model" is false; the update is
low-rank and the model is not.

**DERIVED — linearity of the count.** For fixed shapes, `P` is linear in `r`, so halving the rank halves the trainable
count exactly. Nothing else in the cost model is linear: memory has the same linearity only in the parameter terms,
merge cost is linear in `r` with a large constant, and quality is not linear in anything.

**DERIVED — redundancy of the factorisation.** The map `(B, A) -> B A` is invariant under `B -> B R`, `A -> R^-1 A` for
invertible `R`. The identifiable degrees of freedom are `r*(d_in+d_out) - r^2`, and this is why factor values from two
runs are not comparable and why averaging factors across runs is undefined.

**DERIVED — merge equality in exact arithmetic.** `(W + s B A) x = W x + s B (A x)` holds exactly in real arithmetic,
which is what licenses merge-at-deploy-time. In floating point the two evaluation orders round differently, so the
merged artifact must be validated as its own artifact rather than assumed equivalent.

**DERIVED — best low-rank approximation.** For a known target `D`, the Frobenius-optimal rank-`k` approximation is the
truncated SVD, and the squared error equals the sum of discarded squared singular values. This is a diagnostic for what
rank a full fine-tuning delta would have needed; it is not a prediction, because the target of training is not the
delta and the loss is not the Frobenius norm.

**SOURCED — the method itself.** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685,
introduces adaptation of frozen pretrained weights by a low-rank additive update, with the `alpha/r` scaling and the
observation that the update can be merged into the base weights for zero added inference latency after merging. The
paper's experimental settings are specific to the models and tasks it studied; nothing in this card reproduces them.

**SOURCED — quantized bases.** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314,
describes training adapters through a frozen quantized base with higher-precision compute. The numerical and memory
behaviour depends on the quantization scheme and the kernels used; see the QLoRA card in this collection.

**SOURCED — rank-stabilised scaling.** Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA",
arXiv:2312.03732, proposes `alpha/sqrt(r)` so that increasing rank does not monotonically shrink the effective scale.
This card uses that as the second convention in F8, without asserting a quality benefit for any particular task.

**SOURCED — magnitude decomposition.** Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", arXiv:2402.09353,
decomposes the adapted weight into a trainable magnitude and a direction updated by a low-rank term. The extra
parameters are counted in F9; the reported behaviour is that paper's, not a general result.

**OPEN — everything about realised speed.** Kernel fusion, batch shape, sequence length, allocator behaviour and
hardware generation determine served latency. No number in this card measures it.

**OPEN — the rank needed for a task.** No theorem predicts it. The only sound procedure is a bounded sweep with a fixed
evaluation, a fixed token budget and a stated scale convention.

**Review trigger.** A change to the base checkpoint, the target module names, the scaling convention, the runtime's
merge implementation, or the evaluation population invalidates the numbers and the conclusions of this card.

**Takeaway:** The algebraic results travel across implementations; the empirical ones do not, and the paper citations
here are pointers, not evidence about your workload.

## 11. Interview / Exam Questions

### Q1. Compute the trainable count for a rank-16 adapter on 32 blocks targeting q and o only, with d = 4096.
**Answer:** 8,388,608 scalars.
**Reasoning:** Each target is `16 * (4096 + 4096) = 131,072`. Two targets per block give 262,144; over 32 blocks,
`262,144 * 32 = 8,388,608`. The tempting wrong answer is 4,194,304, which comes from counting one pair per block because
q and o share the same shape — a shape coincidence is not a storage coincidence.
**Trap:** Reporting the per-block figure as the total, or treating `o_proj` as the same tensor as `q_proj` because the
shapes match.
**Follow-up:** How does the answer change if the model uses tied q/o weights? It becomes 4,194,304, because one
allocation serves both.

### Q2. Why does a rank-8 adapter on 64 targets not equal a rank-512 adapter on one target?
**Answer:** Because rank bounds each update separately; the update is a direct sum over independent targets.
**Reasoning:** Each pair contributes at most eight dimensions inside its own matrix, and the outputs of different layers
are composed through non-linearities rather than summed into one matrix. A single rank-512 update to one matrix is a
different function family with a different parameter count, `512*(d_in+d_out)`, and the two cannot be compared by
adding ranks.
**Trap:** Treating total parameter count as interchangeable capacity.
**Follow-up:** What is the equivalent question you can actually answer with arithmetic? Parameter counts per target and
the linear-in-`r` relationship within a fixed target list.

### Q3. A teammate says "alpha does not matter, only the ratio". Is that right?
**Answer:** Only partly, and only under one convention.
**Reasoning:** Under `s = alpha/r`, `alpha` alone is inert: doubling both `alpha` and `r` keeps `s` and doubles the
count. Under `alpha B A` or `alpha/sqrt(r)`, `alpha` is directly the scale and does change the update magnitude. The
statement is a property of the convention, not of LoRA.
**Trap:** Assuming the ratio form is universal across frameworks; some apply no division by `r` at all.
**Follow-up:** With `alpha = 32` and `r` going from 8 to 32, what happens to `s` under each of the three conventions?
`alpha/r`: 4.0 to 1.0. `alpha`: 32 throughout. `alpha/sqrt(r)`: 11.31 to 5.66.

### Q4. Derive why zero-initialising B yields a no-op at step zero but not at step one.
**Answer:** Because the forward contribution `s B A x` is zero when `B = 0`, while `dL/dB = s g (Ax)^T` is generally
non-zero.
**Reasoning:** At the first step the loss and logits equal the base model's, so training starts from the base's
behaviour. The gradient with respect to `A` is `s B^T g x^T`, which is exactly zero when `B = 0`, so `A` is unchanged
after the first update while `B` moves. From step two both factors receive non-zero gradients.
**Trap:** Claiming `A` also gets a gradient at step one, or claiming the no-op property requires both factors to be zero.
**Follow-up:** What does this imply for a run of 50 steps? The adapter effectively has one fewer update than the step
count, which is negligible at thousands of steps and material at tens.

### Q5. You must fit a training run in 24 GiB with a 10.8 GiB bf16 base. What dominates, and what do you cut?
**Answer:** Activations dominate, and the adapter is not the lever.
**Reasoning:** Adapter state at 36.8M parameters and 16 bytes per parameter is 562 MiB, about 2.3% of the budget. A
batch-2, 8192-token forward retains at least 4 GiB of block activations without checkpointing — four to six such
tensors per block makes it 16 to 24 GiB — against 1 GiB of retained input with checkpointing, plus 1 GiB of KV cache
per sequence and allocator slack. Cutting rank from 16 to 8 saves 281 MiB; halving the sequence length halves the
retained input and the cache, and halves the recompute transient.
**Trap:** Halving rank and expecting the out-of-memory error to disappear.
**Follow-up:** What is the smallest configuration change that buys the most memory? Shortening the training sequence or
enabling checkpointing, both of which act on the dominant term.

### Q6. Distinguish the three cases where two adapters give different outputs for the same input.
**Answer:** Different base revision, different scale convention, or an intervening numerical difference such as merge
precision.
**Reasoning:** The adapter is meaningless without its base hash. Two adapters trained on different base commits, or the
same adapter applied with `s` interpreted differently, produce different effective weights. Merge precision adds a third
case where the mathematics is identical but the arithmetic is not.
**Trap:** Attributing the difference to training randomness, which cannot change inference for a fixed artifact.
**Follow-up:** Which check discriminates them in one run? Fix the base hash and the scale, compare logits per target,
then compare merged versus unmerged in fp32.

### Q7. Compute the crossover rank at which an adapter on `down_proj` (11008 x 4096) exceeds the matrix in parameters.
**Answer:** 2,985.22, so the smallest integer rank that exceeds it is 2,986.
**Reasoning:** `r >= d_in*d_out/(d_in+d_out) = 45,088,768/15,104 = 2,985.22`. At `r = 2985` the pair holds
`2985*15,104 = 45,085,440` scalars, slightly fewer than 45,088,768; at 2,986 it holds 45,100,544.
**Trap:** Using `min(d_in, d_out)/2` as a universal threshold; that is only correct for square matrices.
**Follow-up:** Why would anyone choose such a rank? They would not; it is the signal that a dense method is the right
one for that target.

### Q8. What is the effective number of free parameters of a rank-16 adapter on a 4096-square target, and why does it differ from the count?
**Answer:** 130,816, which is 131,072 minus 16 squared.
**Reasoning:** The factorisation has a symmetry group of invertible `16 x 16` matrices, so `r^2 = 256` parameter
directions leave the product unchanged. The reachable set of updates is therefore parameterised by 130,816 degrees of
freedom, and gradient descent moves along the quotient.
**Trap:** Assuming the model has 131,072 independent coordinates that training can set arbitrarily.
**Follow-up:** What practical consequence does this have? Factor values are not interpretable across runs, and factor
averaging between two runs is not a meaningful operation.

### Q9. A full fine-tune delta has singular values 6, 3, 1 in one layer. What fraction of squared Frobenius energy does a rank-1 adapter capture, and what is the residual norm?
**Answer:** 78.26%, with residual norm 3.162.
**Reasoning:** Squared norm is `36 + 9 + 1 = 46`. Rank-1 keeps `36`, so `36/46 = 0.7826`. The discarded squared mass is
`9 + 1 = 10`, whose square root is `sqrt(10) = 3.1623`.
**Trap:** Reporting the ratio of singular values, `6/10 = 0.6`, instead of the ratio of squared energy.
**Follow-up:** What is the caveat? The rank-2 approximation still outputs zero for the input aligned with the discarded
third direction, so 97.83% energy retention can coexist with total error on a specific input.

### Q10. Diagnose: training loss falls 40%, held-out exact-match is unchanged, and the training set was split randomly by row.
**Answer:** Likely leakage or a degenerate split, with prompt-format memorisation as the competing cause.
**Reasoning:** A random row split places near-duplicate items and the same conversation on both sides, so the drop can
be memorisation that does not transfer. If instead the groups are clean, the loss drop concentrated in prompt tokens
indicates the adapter learned formatting rather than the target behaviour.
**Trap:** Concluding the adapter "did not learn" and raising the rank.
**Follow-up:** What single change makes the evaluation trustworthy? Re-split by group with a time cut-off, mask prompt
tokens, and re-evaluate on untouched groups with a pre-registered effect size.

### Q11. Two runs have identical loss but completely different factor values. Is one wrong?
**Answer:** No; the parameterisation is redundant by construction.
**Reasoning:** Any invertible `R` with `B -> B R` and `A -> R^-1 A` leaves `B A` and therefore every output unchanged.
Two runs can converge to different points of the same orbit with identical loss, identical logits and identical
downstream behaviour.
**Trap:** Comparing `A` matrices across runs, or comparing per-factor norms, and concluding a training bug.
**Follow-up:** What is a valid comparison? A fixed-batch logit fingerprint, or the product `B A`, or held-out scores.

### Q12. Estimate whether LoRA gives a large training speed-up for a 5.8B model. Show the bound.
**Answer:** No; the bound is small because frozen work dominates.
**Reasoning:** LoRA removes frozen weight gradients and their optimizer state, which is about 0.635% of parameter-token
work in the worked example. If 85% of step time is frozen forward and input-gradient backward, the Amdahl bound is
`1/0.85 = 1.176`, i.e. at most +17.6% throughput, and usually less after communication and launch overhead.
**Trap:** Dividing total parameters by adapter parameters and calling the quotient a speed-up.
**Follow-up:** Where is the real gain? In memory: 86.45 GiB of full fine-tuning state versus about 11.36 GiB with a
bf16 base and an adapter, a 7.6x reduction in the parameter-related term.

### Q13. You must support 250 tenant adapters on one base. Compute the storage and name two operational costs that do not shrink.
**Answer:** 17.15 GiB of adapters against 2,701.75 GiB of full copies, and evaluation plus rollback per tenant.
**Reasoning:** `250 * 70.25 MiB = 17,562.5 MiB = 17.15 GiB`. A full bf16 copy is 10.807 GiB, so 250 copies is
2,701.75 GiB. Every tenant still needs its own evaluation, version record, rollback path and access control, and the
base must be pinned by hash for all of them simultaneously.
**Trap:** Assuming the storage saving also reduces the validation burden; it increases it, because there are now 250
artifacts with independent lifecycles.
**Follow-up:** What failure appears first at this scale? Adapter routing and cache-key errors, because the base is shared
and the identity of the adapter becomes the only thing separating tenants.

### Q14. When is merging the wrong deployment choice even though it is numerically exact?
**Answer:** When adapters must be switched per request or per tenant, and when the base must remain bit-identical.
**Reasoning:** Merging produces one dense artifact per adapter, so 250 tenants become 250 base-sized copies. It also
overwrites the base weights in that artifact, so any process that must serve the unadapted base needs a second copy, and
any audit that compares against the base needs a third.
**Trap:** Treating merge as a pure optimisation without pricing the artifact multiplication.
**Follow-up:** What is the correct test? Merge only where the traffic mix is stable per artifact, and compare merged and
unmerged logits in fp32 before shipping either.

### Q15. Which single check would have caught a transposed target in a rectangular projection?
**Answer:** A merge equivalence test on a target where `d_in != d_out`.
**Reasoning:** Transposing a square target leaves every count correct, so the parameter arithmetic passes. On a
`k_proj` of 4096 x 1024 the transposed pair has the wrong shapes and fails immediately, and even if it were reshaped,
the merged-versus-unmerged logit comparison would differ.
**Trap:** Relying on a square `q_proj` as the smoke test, which is exactly where the error is invisible.
**Follow-up:** What tolerance is appropriate? Compare in fp32 with a stated maximum absolute logit difference, since
exact equality is not achievable across evaluation orders.

**Takeaway:** Every question above is decided by arithmetic or by a discriminating measurement; none is decided by
preference.

## 12. Quick Reference Card

| Item | Value or rule | Where |
|---|---|---|
| Adapter count, one target | `r * (d_in + d_out)`, plus any extra trainable tensor | F2 |
| Relative cost on a target | `r * (1/d_in + 1/d_out)`; larger on skinny matrices | F3 |
| Worked model, r = 16, seven targets | 36,831,232 scalars = 0.635% of 5.80B | §5 |
| Trainable state, 16 B/param | 562 MiB; base bf16 is 10.807 GiB | F7 |
| Scale conventions | `alpha/r` (4→2→1 for r=8,16,32 at alpha=32); `alpha/sqrt(r)` (11.31→8.00→5.66) | F8 |
| Crossover rank | `d_in*d_out/(d_in+d_out)`: 2048 square, 819.2 kv, 2985.2 mlp | F13 |
| Merge cost | about `2*r*` frozen parameter count; 181.46 GFLOP here, once per artifact | §5 |
| Training speed-up bound | `1/f_frozen`; at most +17.6% when 85% is frozen work | F16 |
| Deployment rule | pin base hash, adapter hash and scale; validate the merged artifact as its own artifact | §8 F5, F6 |

Nothing follows this card; use Section 8 for diagnosis and Section 11 for rehearsal.
