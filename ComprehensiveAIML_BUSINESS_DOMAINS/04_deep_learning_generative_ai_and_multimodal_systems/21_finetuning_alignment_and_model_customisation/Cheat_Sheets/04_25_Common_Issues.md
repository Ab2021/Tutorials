---
title: "Common Issues in Fine-Tuning, Alignment and Model Customisation"
topic: "Diagnostic catalogue: LoRA/QLoRA accounting, adaptation data, preference optimization, position extension, evaluation"
collection: "finetuning-alignment-and-model-customisation"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Common Issues in Fine-Tuning, Alignment and Model Customisation

## 1. File Header

**Purpose:** Turn the failure symptoms that actually occur in adapter training, preference optimization and
context extension into 32 stable, individually diagnosable records, each with a discriminating check that separates two
plausible causes before anyone changes a hyperparameter.

**Prerequisites:** The three companion cards in this collection — LoRA parameter accounting, quantized-base adapters,
and rotary position handling — plus token-level cross-entropy, the difference between a proxy metric and a target
metric, and comfort with bytes versus bits.

**Key outcome:** Given a symptom ("the adapter has no effect", "training loss falls but evaluation does not", "quality
drops after merging", "the model ignores the last 20k tokens"), you can name the two or three causes with the highest
base rate, run one check that separates them, and only then choose a bounded correction.

**Exclusions:** This file does not restate the derivations in the three topic cards; it references them by formula
number. It does not cover pretraining from scratch, inference-server architecture, or legal review of training data.
**Level:** Advanced. Each record assumes you can read a run configuration and a device allocation report.
**Version / reviewed:** 1.0 / not independently reviewed.
**Evidence:** Every count, byte figure and interval in this file is DERIVED arithmetic on a stated constructed model and
is labelled ILLUSTRATIVE where it depends on architecture choices. No incident, vendor benchmark, customer result or
production measurement is asserted. Where a claim depends on a paper, the paper is cited in Section 10 as a pointer, not
as evidence about your workload.

**The constructed reference model.** To keep numbers comparable across records, every quantitative example uses one
fictional architecture: 32 decoder blocks, hidden width 4096, grouped-query attention with key/value width 1024,
MLP width 11008, vocabulary 32,000, and a parameter count of 5,801,771,008 excluding biases. Its bf16 weight storage is
10.807 GiB. Adapting seven projections at rank 16 trains 36,831,232 scalars, whose fp32 training state is 562.0 MiB.
These four numbers recur because almost every diagnostic below is a disagreement about one of them.

**How to use the catalogue.** Section 8 holds the records. Read the symptom column until one matches, then jump to the
matching detail block, which states the mechanism and the arithmetic that distinguishes the causes. Section 11 rehearses
the same material as questions, which is the fastest way to find out whether you have actually absorbed it.

**Retrieval cue:** Numbers live in Section 4; the end-to-end budget trace is Section 5; the records are Section 8;
rehearsal is Section 11; the card is Section 12.

**Takeaway:** A catalogue is only worth its length if every entry can be *settled* by a check you can run today, so each
record below names the check before it names the fix.

## 2. Concept Map / Mental Model

```text
                     symptom observed
                            |
             +--------------+--------------+
             |              |              |
        a number is      a number is    behaviour is
        wrong            surprising     wrong but the
             |              |          numbers are fine
             v              v              v
      accounting defect  assumption    data / objective
      (counts, bytes,    violated      defect
       positions)           |              |
             +--------------+--------------+
                            |
                 run ONE discriminating check
                            |
              +-------------+-------------+
              |                           |
        separates causes              fails to separate
              |                           |
        bounded correction          instrumentation defect:
              |                     record it as the finding
        verification signal
              |
        prevention that survives
        the next run
```

Four categories are used throughout, and the category decides what kind of evidence settles the record.

**VIOLATED_ASSUMPTION** records occur when a formula was applied outside its validity condition: a count that assumed
independent allocations, a token budget that assumed every example fits, a scale that assumed one convention. The
arithmetic is usually right; the premise is wrong. These are settled by printing the premise (shapes, hashes, flags) and
comparing it with the formula's stated constraint.

**NUMERICAL_EDGE** records occur when the computation is correct but a boundary behaves differently from the interior: a
transposition hidden by a square matrix, a zero-variance group, a rounding step before requantization, a log-probability
computed under two tokenizations. These are settled by constructing the boundary case deliberately and asserting on it.

**SCALE_FAILURE** records occur when a quantity that was negligible at the tested size dominates at the deployed size:
adapter count against activation count, per-request cache against concurrency, evaluation items against the effect you
want to detect, one tenant against 250. These are settled by recomputing the term at the deployed size, not by
re-running the small experiment.

**CONCEPT_CONFUSION** records occur when two distinct objects share a name: "model rank" versus "update rank",
"parameter count" versus "memory", "reward" versus "quality", "context supported" versus "context trained",
"fine-tuning" versus "knowledge injection". These are settled by defining the two objects separately and showing which
one the metric measures.

**Why the category matters more than the fix.** A CONCEPT_CONFUSION corrected with an engineering change produces a
second failure later, because the misunderstanding survives. A NUMERICAL_EDGE corrected by "try a smaller learning
rate" hides until the boundary recurs. The category tells you what to change: a premise, a boundary test, a sizing
calculation, or a definition.

**The three-question triage.** Before opening any record, answer: (1) which number is being compared with which, and are
their units and populations the same; (2) was the last change to data, configuration, or runtime, and can it be reverted
in one step; (3) what observation would look different if my current explanation were false. If question three has no
answer, you are not diagnosing, you are guessing, and the guess will be recorded as a fix.

**Takeaway:** Classify first, then check, then correct; a record whose check you cannot run is an instrumentation gap,
and the honest response is to log the gap rather than to change a hyperparameter.

## 3. Foundational Definitions

| Term | Definition and boundary | Category it most often confuses |
|---|---|---|
| Trainable count | Scalars that receive optimizer updates; excludes frozen weights and any tensor the framework creates silently | Parameter count vs memory |
| Adapter artifact | Factors plus configuration plus provenance; unusable without a pinned base and tokenizer | Artifact vs model |
| Effective weight | `W + (alpha/r) B A`; can be full rank even when the update is rank one | Update rank vs model rank |
| Scale convention | How `alpha` and `r` combine; `alpha/r`, `alpha`, and `alpha/sqrt(r)` all exist | Hyperparameter vs its effect |
| Trainable state | Weights, gradients, master copies and moments for trainable scalars, in bytes | State vs peak memory |
| Peak resident memory | Maximum simultaneous device allocation, including allocator slack and fragmentation | Sum of tensors vs peak |
| Activation | Forward tensor retained for backward, or recomputed under checkpointing | Adapter cost vs step cost |
| Group split | Splitting by customer, document or conversation so no group spans train and test | Row split vs group split |
| Packing | Concatenating examples into fixed-length sequences to raise utilisation | Padding vs attention isolation |
| Target mask | Which tokens contribute to the loss; prompt tokens are usually excluded | Loss value vs learning signal |
| Preference pair | A chosen and a rejected completion for one prompt, with a known provenance | Preference vs correctness |
| Reference policy | The frozen model whose log-probabilities anchor a preference objective | Regulariser vs baseline |
| Implicit reward | `beta * (log pi(y|x) - log pi_ref(y|x))`, a scalar defined up to a prompt-constant | Reward vs probability |
| Reward model | A learned scalar proxy trained on preference labels | Proxy vs target quality |
| Reward hacking | Improvement in the proxy without improvement, or with degradation, in the target | Proxy vs target |
| KL penalty | A term penalising divergence from the reference policy | Constraint vs regulariser |
| Rotary base | The `theta` constant setting the frequency ladder of rotary embeddings | Configuration vs checkpoint identity |
| Position interpolation | Rescaling positions by `1/s` so a longer sequence fits the trained range | Supported vs trained length |
| Frequency ladder | The set of per-dimension angular velocities in rotary attention | Dimension index vs frequency |
| KV cache | Per-token key/value storage that grows with context and concurrency | Model size vs serving memory |
| Quantization group | A block of weights sharing one scale; group size trades accuracy for bytes | Bits per weight vs bytes per weight |
| Double quantization | Quantizing the scales themselves, adding a second level of scales | Weight bytes vs total bytes |
| Merge | Folding the scaled update into a dense base weight | Exactness vs bit-exactness |
| Rollout | A sampled completion used by a policy-gradient or group-relative method | Sampling vs scoring |
| Zero-variance group | A group of rollouts with identical reward, hence zero standardised advantage | Zero gradient vs zero learning |
| Verifier | A program or model that scores a completion mechanically | Verification vs preference |
| Gold metric | The measure the decision is actually about, measured independently of the proxy | Proxy vs target |
| Minimum detectable effect | The smallest difference an evaluation can distinguish from noise at a stated interval width | Significance vs importance |

**Units and conventions used in every record.**
- Parameter counts are scalar elements; storage is bytes; 1 MiB = 1,048,576 B, 1 GiB = 1,073,741,824 B.
- Log-probabilities are natural logarithms of per-token probabilities, summed over the completion unless stated.
- `beta` in preference objectives is a dimensionless inverse temperature; it is not a sampling temperature.
- Positions are integer token indices starting at zero; a "32k context" means 32,768 positions, not 32,000.
- A difference in percentage points is written "pp"; a relative change is written with a percent sign.

**On "distinct issue" in this file.** Two records are distinct if their discriminating checks differ. `CI-02` (a
transposed target hidden by a square shape) and `CI-09` (an accidentally targeted output head) can both present as "the
count is not what I expected", and their checks differ: the first compares merged logits on a rectangular target, the
second prints the set of tensor names with `requires_grad` set. Records that would share a check are merged here rather
than split to inflate the count.

**Takeaway:** Most of the catalogue is bookkeeping about which object a number describes; the arithmetic is rarely the
hard part.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**Adapters attached to a target list, and the count that must be asserted.**

```text
FORMULA [C1]: P = sum over targets i of r_i * (d_in_i + d_out_i) + P_extra
WHERE: i indexes distinct tensors; r_i is the rank for that target; d_in_i and d_out_i are
       that tensor's widths; P_extra counts every other trainable scalar.
CONSTRAINT: Each tensor once; tied tensors once; no double multiplication by block count.
NUMERICAL EXAMPLE: seven targets, r = 16, 32 blocks: attention 425,984 + MLP 724,992 =
       1,150,976 per block, times 32 = 36,831,232 scalars, which is 0.6348% of 5,801,771,008.
INTERPRETATION: Any reported count that differs from C1 by a small factor is a target-set
       error, not a rounding error; the factor usually equals the number of mis-identified
       modules.
```

**Peak memory, which is the quantity people intend when they say "the adapter is small".**

```text
FORMULA [C2]: M_peak = M_base + P*(b_w + b_g + b_m + b_v) + M_act + M_tmp + M_alloc
WHERE: b_w, b_g, b_m, b_v are bytes per trainable scalar for weights, gradients and the two
       optimizer moments; M_act is the activation peak; M_tmp and M_alloc are workspace and
       allocator slack; all terms are live at the same instant on the same device.
CONSTRAINT: Co-residency. Summing maxima measured at different times overstates the peak.
NUMERICAL EXAMPLE: P = 36,831,232, b = (4,4,4,4) gives 589,299,712 B = 562.0 MiB, against
       a bf16 base of 10.807 GiB and a 4.000 GiB lower bound on the activation peak at batch 2
       and 8192 tokens.
INTERPRETATION: The adapter is 5.08% of the bf16 base and 18.7% of the 3.000 GiB
       checkpointed activation term; a memory investigation that starts with rank is
       starting with the smallest lever.
```

**Four-bit storage including scales, because "4-bit means half of bf16" is false.**

```text
FORMULA [C3]: M_4bit = N * (4/8) + (N / g) * b_s
WHERE: N is the number of quantized weights; g is the group size in weights per scale; b_s is
       bytes per scale; the first term is packed weight bytes.
CONSTRAINT: Excludes double quantization of the scales, compute buffers and any non-quantized
       module such as embeddings or the output head.
NUMERICAL EXAMPLE: N = 5,801,771,008, g = 64, b_s = 4:
       weights 2,900,885,504 B; groups 90,652,672; scales 362,610,688 B;
       total 3,263,496,192 B = 3.039 GiB, which is 28.13% of the 10.807 GiB bf16 base.
INTERPRETATION: The payload is 0.500 bytes per weight, not 0.5 bits, and the scales are
       12.5% of the packed bytes and 11.1% of the total. A 4x claim is approximately right
       only at large group size with double quantization, and it is never exactly 4x.
```

**Preference objective, written so its cancellation is visible.**

```text
FORMULA [C4]: L_DPO = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                                            - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )
WHERE: y_w is chosen, y_l is rejected, pi is the policy, pi_ref the frozen reference, beta a
       positive inverse temperature; all log-probabilities are over the completion.
CONSTRAINT: Both models must score the identical token sequence with the identical tokenizer
       and the identical template.
NUMERICAL EXAMPLE: policy log-probabilities are 1.60 (chosen) and 1.00 (rejected), so
       m_pi = +0.60; reference ones are 1.00 and 0.55, so m_ref = +0.45. Then
       z = beta*(m_pi - m_ref) = 0.1*0.15 = 0.015; loss = -log sigmoid(0.015) = 0.68564;
       gradient factor = sigmoid(-z) = 0.49625.
INTERPRETATION: The loss depends only on the difference of margins, so any prompt-level
       constant shared by both completions cancels exactly. A bug that changes the constant
       is invisible in the loss and fatal in the gradient only if it changes unequally.
```

Recomputing the example exactly: `sigmoid(0.015) = 1/(1+e^-0.015) = 0.50375`, so `-log = 0.68564` and the gradient
factor is `1 - 0.50375 = 0.49625`. Both are close to the neutral values `log 2 = 0.69315` and `0.5`, which is the point: a pair
whose margins barely differ carries almost no learning signal, and a dataset of such pairs produces a flat loss curve
that looks like a learning-rate problem.

**Context extension: the position map and what it does to frequencies.**

```text
FORMULA [C5]: theta_i = base^(-2i/d_head),  wavelength_i = 2*pi/theta_i,
               PI: p' = p / s,  NTK-style base adjustment: base' = base * s^(d_head/(d_head-2))
WHERE: i in [0, d_head/2) indexes rotary dimension pairs; d_head is head width; s > 1 is the
       extension factor.
CONSTRAINT: Rotary embeddings encode relative offsets; any monotone position map preserves
       relative offsets within the trained window and changes them outside it.
NUMERICAL EXAMPLE: d_head = 128, base = 10,000: theta at i = 16 is 0.1, wavelength 62.83
       tokens; at i = 48, theta 0.001, wavelength 6,283.19 tokens. With s = 8, a position of
       32,767 maps to 4,095.875 under PI, and the base becomes 10,000 * 8^(128/126) = 82,684.6.
INTERPRETATION: PI keeps every learned frequency and compresses positions into the trained
       range; base adjustment stretches low frequencies while keeping short wavelengths
       nearly unchanged. They fail differently, which is why the check must name the method.
```

**Evaluation width, so a claimed win can be compared with noise.**

```text
FORMULA [C6]: half-width = z * sqrt( p*(1-p) / n ),   MDE ~ 2 * half-width
WHERE: p is the per-item success rate, n the items per arm, z = 1.96 for a 95% interval.
CONSTRAINT: Independent items; a group split means items within a group are correlated and
       the effective n is smaller than the item count.
NUMERICAL EXAMPLE: p = 0.8. n = 200 -> se 0.02828, half-width 5.54 pp. n = 400 -> 3.92 pp.
       n = 1,000 -> 2.48 pp. n = 4,000 -> 1.24 pp.
INTERPRETATION: A 2 pp improvement cannot be established by 400 items per arm; the claimed
       effect is inside the interval, so the honest conclusion is "not distinguished".
```

**Validity condition for 4.1.** C1 requires independent allocations and a complete target list. C2 requires
co-residency. C3 requires a single quantization group size and excludes non-quantized modules. C4 requires identical
tokenization and template across the two models. C5 requires head width and base to match the checkpoint. C6 requires
independent items.

### 4.2 General case

**Several issues interacting: the composed memory model.** In practice the failures cluster. A team reduces rank to fix
memory (C2), reduces it so far that quality collapses, then adds more targets to recover, then discovers the count
assertion in CI-01 was never true. The composed model is:

```text
FORMULA [C7]: M_peak = N*b_base + P*(b_state) + T*(act_per_token) + 2*L*n_kv*b_w*T_ctx + M_alloc
WHERE: T is the microbatch times sequence length in tokens; act_per_token is retained
       activation bytes per token per block; T_ctx is context length for the KV term;
       n_kv is key/value width per layer; L is layer count.
CONSTRAINT: The KV term applies to generation and to cached evaluation, not to a training
       step unless the step caches.
NUMERICAL EXAMPLE (Section 5's configuration): N*b_base = 3.039 GiB (4-bit with fp32
       scales); P*b_state = 0.549 GiB; retained network input 1.000 GiB plus 2.000 GiB of
       recompute transient under checkpointing; live total 6.588 GiB, or 7.247 GiB with 10%
       slack. Adding the KV term for 8192 cached tokens puts it at 7.247 + 1.000 = 8.247 GiB.
INTERPRETATION: An 8 GiB device fails and a 24 GiB device passes with room to spare; the
       same configuration on a bf16 base adds 7.768 GiB (10.807 minus 3.039), taking the
       8.247 GiB total to 16.015 GiB and consuming most of a 24 GiB device. Base precision
       and sequence length, not rank, decide feasibility.
```

**Generalised count with mixed ranks and an extra trainable module.**

```text
FORMULA [C8]: P = sum_i r_i*(d_in_i + d_out_i) + n_bias*d_bias + n_norm*d_norm + P_head
WHERE: the extra terms are examples of silently trainable modules; each must be enumerated
       rather than assumed zero.
NUMERICAL EXAMPLE: the worked model at mixed ranks (q,o at 32; k,v at 16; MLP at 8) gives
       2*32*8192 + 2*16*5120 + 3*8*15104 = 524,288 + 163,840 + 362,496 = 1,050,624 per
       block, 33,619,968 total.
INTERPRETATION: Mixed ranks are legitimate but make the "total rank" meaningless; assert the
       per-target vector, not a sum.
```

**Generalised preference objective with a KL term, to show where the reference enters.**

```text
FORMULA [C9]: L = -E[ log sigmoid(beta * (m_pi - m_ref)) ] + lambda * KL(pi || pi_ref)
WHERE: m_pi and m_ref are the chosen-minus-rejected margin differences defined in C4;
       lambda >= 0 controls drift.
CONSTRAINT: lambda = 0 recovers DPO; lambda > 0 changes both the optimum and the gradient.
NUMERICAL EXAMPLE: with m_pi - m_ref = 0.15 and beta = 0.1, the pairwise term is 0.68564;
       adding lambda * KL = 0.05 * 0.4 = 0.020 raises the loss to 0.70564 and, more
       importantly, changes the gradient direction rather than its scale.
INTERPRETATION: Two runs reporting "the same loss" can be optimising different objectives;
       the lambda and beta values belong in the comparison, not in the appendix.
```

**Validity condition for 4.2.** C7 assumes the KV term and the activation term do not overlap in a training step. C8
assumes each extra module is enumerated explicitly; any "and others" in a count is an unverified term. C9 assumes the
KL is measured on the same tokenisation as the pairwise term.

### 4.3 Variants

**Variant A: group-relative advantage, where zero variance is a boundary.**

```text
FORMULA [C10]: A_j = (r_j - mean(r)) / (std(r) + eps)
WHERE: j indexes the G rollouts of one prompt; r_j is the scalar reward; eps is a small
       constant for numerical stability.
CONSTRAINT: If all r_j are equal, the numerator is zero for every j; the advantages are all
       zero and the prompt contributes no gradient.
NUMERICAL EXAMPLE: G = 8 rollouts with rewards 1,1,1,1,1,1,1,1 -> std 0, all A_j = 0.
       With rewards 1,1,1,1,0,0,0,0 -> mean 0.500, std 0.500, so A = +1.000 for the four
       successes and -1.000 for the four failures.
INTERPRETATION: A task that a model solves every time, or never, teaches nothing under this
       objective; the fraction of zero-variance groups is a first-class training metric, and
       it is invisible in a mean-reward curve.
```

**Variant B: quantized base with double quantization of scales.**

```text
FORMULA [C11]: M = N/2 + (N/g)*b_s1 + (N/g)/g2 * b_s2
WHERE: b_s1 is bytes per first-level scale, g2 the number of first-level scales sharing a
       second-level scale, b_s2 bytes per second-level scale.
CONSTRAINT: Savings depend on g and g2; small group sizes leave little to quantize away.
NUMERICAL EXAMPLE: N = 5,801,771,008, g = 64, b_s1 = 1, g2 = 256, b_s2 = 4:
       2,900,885,504 + 90,652,672 + 1,416,448 = 2,992,954,624 B = 2.787 GiB, which is 8.29%
       below the 3.039 GiB single-level figure.
INTERPRETATION: Double quantization buys 8.3% at these settings, not 2x; quoting it as if it
       were a precision change confuses two independent knobs.
```

**Variant C: partial position interpolation with a preserved high-frequency band.**

```text
FORMULA [C12]: for dimension pairs below index k, keep theta_i unchanged; for pairs at or
               above k, divide theta_i by s (equivalently, scale those positions by 1/s).
WHERE: k selects how many of the d_head/2 pairs are stretched; k = d_head/2 is full PI.
CONSTRAINT: The choice must match the trained configuration; a mismatch at inference is
       silent because shapes are identical.
NUMERICAL EXAMPLE: d_head = 128, k = 32 keeps pair indices 0..31 unchanged, i.e.
       wavelengths from 6.28 to 544.10 tokens, and stretches only pairs whose wavelength
       is 628.32 tokens or longer.
INTERPRETATION: Short-wavelength pairs encode local order; stretching them degrades
       short-range behaviour while rarely helping long range, which is why partial schemes
       exist. The failure signature is a model that is fine at 4k and incoherent at 32k.
```

**Validity condition for 4.3.** C10 requires more than one distinct reward in the group. C11 requires the scale
metadata to be counted, not assumed. C12 requires the band split to be recorded in the checkpoint config, because two
models with identical tensor shapes can use different bands.

### 4.4 Limits and counterexamples

**Limit 1: the count is not the cost.** A 36,831,232-scalar adapter is 0.635% of the base in parameter count. In
storage it is 562.0 MiB of trainable state against 10.807 GiB of frozen bf16 weights, i.e. 5.08% of the base's storage —
an eightfold larger share — because each trainable scalar carries 16 bytes across weights, gradients, master copy and
two moments while each frozen scalar carries two. Dividing two parameter counts and calling the result a cost ratio
understates the storage share by 8x and says nothing at all about step time.

**Limit 2: an exact algebraic identity does not survive a dtype change.** `(W + sBA)x = Wx + sB(Ax)` holds in real
arithmetic. In bf16 the two orders round differently, and after dequantization and requantization the difference can
exceed the effect you were trying to measure. Merge equality tells you the operation is *licensed*, not that the merged
artifact is equivalent.

**Limit 3: intervals assume independence, and group structure breaks it.** C6 with n = 4,000 items drawn from 40
conversations has an effective sample size near 40 for between-conversation effects. A "1.24 pp half-width" computed on
correlated items is not a 1.24 pp half-width.

**Limit 4: the reward cancels a constant, not an error.** In C4, any term added equally to chosen and rejected
log-probabilities cancels. Therefore a systematic scoring bug, such as a template prefix present in both, is invisible
in the loss. This is a *feature* of the objective and a trap for validation: the loss cannot detect it, so a separate
test must.

**Counterexample to "higher reward means better model".** A reward model is a proxy fitted on preference labels. Beyond
the point where the policy exploits its errors, proxy reward keeps rising while the independently measured target
quality can fall. Two constructed curves make this concrete in Section 5; the mechanism is that the proxy's error is a
function the optimiser is free to maximise.

**Counterexample to "more rank is safer".** Raising rank while holding `alpha` fixed divides the effective scale,
so the run that "increased capacity" may have simply reduced its step size. The symptom is a flatter loss curve that is
attributed to overfitting, and the correction applied is early stopping, which addresses neither cause.

**Validity condition for 4.4.** Every limit above is a statement about a model of the system. The counterexamples are
constructed, not observed, and they establish possibility only.

**Takeaway:** Identities hold in the arithmetic they were derived in; the moment precision, sampling or population
changes, the identity is a hypothesis again.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE. A constructed adaptation project is budgeted end to end so that each catalogue record can be
tied to a concrete number. No run was performed.

**Setup.** The reference model of Section 1. Base frozen in 4-bit with group size 64 and fp32 scales. Adapter rank 16 on
seven projections. Dataset: 50,000 examples, mean 380 tokens, maximum 1,500 tokens, three epochs, micro-batch 4,
gradient accumulation 8, sequence length 4,096 with packing, one device with 24 GiB.

**Step 1: base storage.**

```text
packed weights      : 5,801,771,008 * 0.5 B   = 2,900,885,504 B
scale count         : 5,801,771,008 / 64      = 90,652,672
scale bytes (fp32)  : 90,652,672 * 4 B        = 362,610,688 B
total base          : 3,263,496,192 B         = 3.039 GiB
bf16 comparison     : 5,801,771,008 * 2 B     = 11,603,542,016 B = 10.807 GiB
ratio               : 3,263,496,192 / 11,603,542,016 = 0.28125
```

The ratio is exactly 9/32, not 1/4, because of the scales. This is the number CI-23 is about.

**Step 2: adapter and its training state.**

```text
P                   : 36,831,232 scalars (0.6348% of base)
weights in fp32     : 36,831,232 * 4 B = 147,324,928 B = 140.5 MiB
gradients in fp32   : 147,324,928 B   = 140.5 MiB
two moments, fp32   : 294,649,856 B   = 281.0 MiB
total trainable     : 589,299,712 B   = 562.0 MiB
```

**Step 3: activation peak at the configured step.**

```text
tokens per step     : 4 * 8 * 4,096 = 131,072 tokens
one width-4096 tensor: 131,072 * 4,096 * 2 B (bf16) = 1,073,741,824 B = 1.000 GiB
retained per block  : at least one such tensor per block, and in practice four to six
                      (projections, attention scores, MLP hidden), so 4.000 to 6.000 GiB
32 blocks           : 128.000 to 192.000 GiB, i.e. far beyond the device
```

The lower bound alone exceeds a 24 GiB card by more than 5x, which is why checkpointing is not optional here. With
every block recomputed, only the network input is retained and one block's internals exist at a time:

```text
retained (network input across segments): 1.000 GiB
transient per block during recompute    : about 2.000 GiB (two such tensors)
```

So the activation term falls from at least 128.000 GiB to roughly 3.000 GiB, at the cost of a second forward pass. The
1 GiB network input is not avoidable by checkpointing; only shorter sequences or smaller microbatches reduce it.

**Step 4: live total with checkpointing.**

```text
base                : 3.039 GiB
trainable state     : 0.549 GiB (562.0 MiB)
retained input      : 1.000 GiB
recompute transient : 2.000 GiB
live before slack   : 6.588 GiB
allocator slack 10% : 0.659 GiB
live total          : 7.247 GiB
headroom on 24 GiB  : 16.753 GiB
```

Recheck: `3.039 + 0.549 + 1.000 + 2.000 = 6.588`, and `6.588 * 1.10 = 7.247`. The headroom is what the run has left for
longer sequences, a larger microbatch, or fragmentation, and it is the number to quote when someone asks whether the
configuration is safe. It is not a licence to raise the sequence length by 2x: doubling the token count doubles both the
1.000 GiB retained term and the 2.000 GiB transient, consuming 3.000 GiB of the 16.753 GiB headroom.

**Step 5: optimisation schedule and what the token budget buys.**

```text
total tokens        : 50,000 * 512 * 3 = 76,800,000
tokens per step     : 131,072
steps               : 76,800,000 / 131,072 = 585.94, so 585 full steps plus a partial
tokens per parameter: 76,800,000 / 5,801,771,008 = 0.0132
```

0.0132 tokens per parameter is a fact worth stating plainly: it is 1,500x below even the low end of the 20-to-2,000
tokens-per-parameter range usually associated with continued pretraining or from-scratch training
(`20 / 0.0132 = 1,511`). Adapter tuning at this budget can change behaviour and cannot add much knowledge. A run at this budget can change behaviour;
it cannot add much knowledge. This is the arithmetic behind CI-30.

**Step 6: packing utilisation.**

```text
mean example length           : 200*0.40 + 300*0.35 + 600*0.20 + 1500*0.05
                              : 80 + 105 + 120 + 75 = 380 tokens
unpacked utilisation at 4,096 : 380 / 4,096 = 9.28%
padding waste                 : 90.72% of every sequence
packed utilisation            : approximately 100% modulo the boundary remainder
```

Unpacked, 90.72% of the compute is spent on padding. That is the reason packing exists, and also the reason CI-13
exists: packing without attention isolation trades wasted compute for cross-example leakage.

**Step 7: evaluation sizing.**

```text
per-item success rate p = 0.80, 95% interval
n = 400  : se 0.02000, half-width 3.92 pp
n = 1,000: se 0.01265, half-width 2.48 pp
n = 4,000: se 0.00632, half-width 1.24 pp
a 2 pp effect is distinguishable only if the half-width is <= 1.00 pp:
n >= z^2 * p(1-p) / 0.01^2 = 3.8416 * 0.16 / 0.0001 = 6,146.6 items per arm
```

Recompute: `1.96^2 = 3.8416`, times `0.8 * 0.2 = 0.16` gives 0.614656, divided by `0.01^2 = 0.0001` gives 6,146.6. A
team that evaluates 400 items and reports a 2 pp win has a half-width of 3.92 pp, i.e. an interval about twice as wide
as the effect it claims, which is CI-25.

**Step 8: the proxy-versus-target divergence, constructed.**

```text
proxy reward  P(k) = 1.00 + 0.18k - 0.006k^2
target quality G(k) = 1.00 + 0.14k - 0.020k^2        k = optimisation pressure, arbitrary units
k = 0 : P = 1.000, G = 1.000
k = 2 : P = 1.336, G = 1.200
k = 5 : P = 1.750, G = 1.200
k = 10: P = 2.200, G = 0.400
k = 15: P = 2.350, G = -1.400  (proxy maximum: dP/dk = 0.18 - 0.012k = 0)
k = 20: P = 2.200, G = -4.200  (the proxy declines, the target has already collapsed)
```

The proxy peaks at `k = 15` (`dP/dk = 0`), the target peaks at `k = 3.5` (`dG/dk = 0`). Beyond `k = 3.5` every unit of
additional proxy reward costs target quality, and the gap is invisible to anyone watching only the proxy. This is CI-17.

**Independent arithmetic re-check.** Step 1's ratio can be recomputed structurally: `4 bits = 0.5 B`, plus `4 B` of scale
per 64 weights `= 0.0625 B` per weight, giving `0.5625 B` per weight and `0.5625/2 = 0.28125`. The check agrees, and it
also shows that the ratio is independent of `N` at fixed group size — a useful invariant, since it means a 30B model has
the same 28.125% ratio.

**Sensitivity note.** Four levers, in order of effect on feasibility: sequence length (linear in activation and cache),
group size in the quantizer (0.0625 B per weight at 64, 0.03125 B at 128), checkpointing (about 42x on the total
activation term here: 128 to 3 GiB), and rank (linear in 0.635% of the parameter count). A team that reaches for rank first is pulling the lever
with the smallest mechanical advantage.

**Takeaway:** The budget is decided by activations and context, not by the adapter; and the evaluation plan, not the
training run, determines whether any of it can be measured.

## 6. Visual / Geometric Intuition

```text
     memory, GiB, one device, 24 GiB card marked by the dotted line
 0.14 |.                                        adapter weights
 0.55 |#                                        trainable state, fp32
 1.00 |##                                       retained input, checkpointed
 2.00 |####                                     recompute transient, per block
 3.04 |######                                   quantized base
 4.00 |########                                 KV cache, one 32k sequence
128.00 |######################################## activations, no checkpointing (lower bound)
    24 |---------- budget --------------------------------------------
```

Reading the chart: the two terms a team usually tunes (adapter weights, trainable state) are the two smallest, together
1.4% of the budget. The dotted line is where the card runs out, and only the bottom bar crosses it. Nothing about the
adapter's rank moves the bottom bar; only sequence length, microbatch and checkpointing do.

```text
      rank and effective scale, alpha = 32
  r = 8   |##########................  s = 4.000
  r = 16  |#####....................   s = 2.000
  r = 32  |##.......................    s = 1.000
  r = 64  |#........................    s = 0.500
```

A rank sweep with fixed `alpha` is not a capacity sweep; it is a diagonal walk through capacity and step size together.
The dots are capacity, the hashes are scale, and only one of them is what the experiment claims to vary.

```text
   position mapping under interpolation, s = 8
   trained positions  |---- 0 ...... 4095 ----|
   requested 32,768   |---- 0 .. 4095 .. 32767 ----|
   mapped             |-- 0 .. 512 .. 4096 --|
```

Every requested position above 4,096 is mapped into the trained range. The model never sees a position it did not train
on, which is why PI is attractive, and also why it loses resolution: 8 consecutive requested positions collapse into 1
mapped position, so the model cannot distinguish tokens closer than the compression factor at the far end.

**Geometric consequence of the rank bound, restated as a picture.** The adapter's contribution, for a fixed input, lies
in the column space of `B`, a subspace of dimension at most `r`. The frozen contribution lies wherever `W` maps the
input. The layer output is the sum of a point in that low-dimensional subspace and a point in a full-dimensional space.
Removing the adapter removes a thin slice; it does not halve the model.

**Where the failure categories live on this picture.**
- VIOLATED_ASSUMPTION: the bar you labelled "adapter" is actually the whole trainable state, or the bar you called
  "base" excludes the scales.
- NUMERICAL_EDGE: the rank/scale diagram holds exactly, but the runtime applies `alpha` without dividing by `r`, so the
  walk is a different walk.
- SCALE_FAILURE: the KV bar is drawn for one sequence and the deployment runs sixty-four.
- CONCEPT_CONFUSION: the capacity dots are read as model capability, when they bound only the correction.

**Takeaway:** Draw the memory bars before the architecture diagram; the bars decide which questions are worth asking.

## 7. Comparison / Selection Table

| Symptom class | First check (cost) | What it settles | What it cannot settle | Typical wrong first move |
|---|---|---|---|---|
| Count mismatch | Print `requires_grad` tensors with shapes and sum (seconds) | Whether the target set is what you intended | Whether the targets are the right ones | Changing rank |
| Silent no-effect | Fixed-prompt logit comparison against base (minutes) | Whether the artifact is applied at all | Whether the adapter is good | Re-running evaluation |
| OOM before backward | Phase-wise allocation: after load, after forward, after backward (minutes) | Which term is binding | Which fix is cheapest | Halving rank |
| OOM during backward | Sweep sequence length at fixed microbatch (minutes) | Whether activations scale with length | Whether checkpointing will meet the time budget | Reducing batch to 1 |
| Loss falls, eval flat | Per-token loss split by prompt versus target, plus split audit (hours) | Whether the signal is in the wrong tokens or from leakage | Whether the task is learnable at all | Raising the learning rate |
| Eval improves, slices regress | Per-slice numerator, denominator, interval (hours) | Which population paid for the gain | Whether the gain is worth the cost | Reporting the aggregate |
| Quality drops after merge | Merged versus unmerged logits in fp32 on a rectangular target (minutes) | Whether the merge or the artifact is wrong | Whether merging is the right deployment choice | Reverting to the unmerged path and never merging |
| Wrong tenant behaviour | Request-to-adapter trace with cache keys (hours) | Whether routing or caching crosses identities | Whether isolation controls are sufficient | Flushing caches |
| Long-context degradation | Per-length-bucket scores plus training length histogram (hours) | Whether the model was trained at that length | Whether extension training will fix it | Raising the serving limit |
| Proxy rises, target flat | Independent gold measurement on a held-out population (days) | Whether the proxy has decoupled from the target | How much of the proxy is error | Continuing to optimise |
| Interval overlaps zero | Recompute n from the claimed effect (minutes) | Whether the claim is even measurable | Whether the effect exists | Adding more slices |

**Choosing between two plausible checks.** Prefer the check that discriminates in one run over the check that requires a
new training run. Of the eleven rows above, nine are executable in under an hour; the two that are not are exactly the
two that people reach for first.

**Choosing the correction scale.** Every correction in this file is bounded: change one premise, re-run one check,
record one number. Corrections that cannot be expressed that way — "improve the data", "tune the hyperparameters" — are
not corrections, they are new projects, and they should be scoped as such.

**Selection rule for this catalogue itself.** If a symptom matches two records, run both checks; they are cheap and the
second one is often the real cause. If a symptom matches no record, the defect is probably in the instrumentation, not
in the model, and the first correction is to add the missing observation.

**Takeaway:** The cheapest check that could change your mind is always the next action; anything else is
procrastination with a dashboard.

## 8. Failure Modes & Edge Cases

**Catalogue.** Columns follow the contract: identifier, category, symptom, competing causes, discriminating check,
correction, prevention, verification. All 32 records are ILLUSTRATIVE constructions of commonly reported failure shapes;
no incident is claimed. Each record is expanded below with its mechanism and the arithmetic that separates its causes.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | Trainable count is a clean fraction of the expected value | Name substring matched fewer modules; framework added scalars; a tied tensor counted twice | Print every `requires_grad` tensor with shape, sum the shapes, diff against C1 per target | Attach factors to explicit module objects from a name walk | Assert the total count at start-up and fail closed | The printed inventory sums to the asserted value with each target once |
| CI-02 | NUMERICAL_EDGE | Count is right, merged quality is wrong, and it is wrong only on rectangular projections | Transposed A/B orientation; wrong scale; merge precision | Merge one `k_proj` (4096x1024) and compare merged versus unmerged logits in fp32 | Fix orientation against the runtime's linear convention | Unit test on a rectangular target, never a square one | Maximum absolute logit difference under the stated tolerance on the rectangular case |
| CI-03 | CONCEPT_CONFUSION | Team caps rank because "the model is low rank" and expects capability loss | Update rank bound applied to the model; scale confounded with capacity | Compute singular values of `W + sBA` for a trained layer and compare its rank with `W` | Restate the claim as a bound on the update only | Review the statement, not the config, in design docs | A test asserting the effective weight's rank exceeds `r` |
| CI-04 | NUMERICAL_EDGE | Rank sweep shows no benefit above 8 and the run is flatter at 32 | `alpha/r` shrinks scale with rank; learning rate not retuned; data-limited ceiling | Log `alpha`, `r`, `alpha/r` and its variant per run; compare loss curves at equal effective scale | Hold the effective scale or the variant fixed across the sweep | Store the convention and the derived scale in the run record | Sweep points share one scale within 1% and differ only in rank |
| CI-05 | SCALE_FAILURE | Peak memory grows when sequence length grows, though rank never changed | Activations; KV cache; allocator slack; quantized buffer growth | Sweep length at fixed microbatch, log peak per phase | Shorten sequences, checkpoint, or reduce microbatch with accumulation | Derive the activation budget from the maximum training length | Measured peak at maximum length sits under the recorded budget |
| CI-06 | NUMERICAL_EDGE | Merged artifact scores 0.5-2 pp below the unmerged pair | Requantization after merge; merge dtype; wrong scale | Compare in fp32 before requantization, then after, on the same batch | Merge in higher precision, cast once, requantize once | Treat the merged artifact as its own artifact with its own evaluation | Deployed artifact's score matches the pre-quantization score within the planned interval |
| CI-07 | CONCEPT_CONFUSION | A cross-tenant request returns another tenant's style, and the adapter is blamed for a security defect | Adapter treated as isolation; cache key omits tenant; authorization bug | Replay two tenants with overlapping prompts and inspect cache keys and adapter ids | Bind tenant and adapter identity into every key and into the request | Negative tests for cross-tenant access in the release gate | Overlapping-prompt test yields each tenant's own behaviour, with access logs |
| CI-08 | VIOLATED_ASSUMPTION | Adapter works in staging and is subtly wrong in production | Unpinned base revision; different tokenizer; different scale | Compare the base commit hash and tokenizer vocabulary hash at both sites | Pin base and tokenizer by hash in the artifact manifest | Startup self-test asserting a logit delta from base | The hash in the request log equals the hash in the manifest |
| CI-09 | NUMERICAL_EDGE | Count is 1.5-2% higher than planned and output style shifts more than expected | Tied output head targeted; embeddings matched by a name pattern; bias trained | List target names explicitly and check for head or embedding entries | Remove the head from the target list or justify it | Target lists are allow-lists, never patterns | The target list contains exactly the intended module names |
| CI-10 | SCALE_FAILURE | Single-request latency and accuracy are fine; tail latency and identity errors appear at load | Adapter resolved per token; adapter swap mid-generation; per-request load cost | Trace an adapter identity across all steps of one generation | Pin the adapter for the whole request; cache by immutable key | Load test at the production tenant mix | Tail latency and adapter-identity assertions hold under load |
| CI-11 | CONCEPT_CONFUSION | Training loss falls, exact-match is flat, and the drop is concentrated in early tokens | Prompt tokens unmasked; template mismatch; leakage | Split loss into prompt and target tokens for the same batches | Mask prompt tokens; align templates; re-evaluate | Mask assertion on a hand-labelled batch before training | Target-token loss falls and held-out score moves with it |
| CI-12 | VIOLATED_ASSUMPTION | Holdout score is far above production behaviour | Row-level split with duplicate conversations; near duplicates; time leakage | Compute group intersection counts between splits and near-duplicate rate | Re-split by group with a time cut-off, deduplicate first | Split before augmentation and record the split hash | Zero group and near-duplicate overlap between splits |
| CI-13 | NUMERICAL_EDGE | Packed training produces cross-example continuations after fine-tuning | Packing without attention isolation; wrong position ids; loss mask crossing boundaries | Decode a packed batch with boundaries marked and inspect what the model can attend to | Use packing with per-example attention and reset positions | Assert segment ids in the data loader | A canary example cannot influence its neighbour's loss |
| CI-14 | SCALE_FAILURE | Improvement vanishes for inputs longer than 2k tokens | Truncation removed the target behaviour; template mismatch at long length | Histogram training lengths and the truncated fraction per slice | Re-budget length or filter examples; align templates | Publish truncated fraction per slice as a data metric | Longest bucket shows the planned effect with an interval |
| CI-15 | CONCEPT_CONFUSION | DPO run changes nothing, and lowering beta does not help | Beta confused with sampling temperature; reference missing; flat pairs | Print beta, the reference identity, and the margin distribution | Fix the objective, confirm the reference, filter near-zero pairs | Margin histogram in the data audit | Loss starts near `log 2` and margins separate over training |
| CI-16 | NUMERICAL_EDGE | DPO loss is plausible but gradients are wrong and the model degrades | Policy and reference tokenized differently; different template; different EOS handling | Re-tokenize one pair with both pipelines and diff the ids | Score both models on one shared token sequence | Store the tokenized batch, not the raw text, in the debug artifact | Token id sequences are identical for both scorers |
| CI-17 | CONCEPT_CONFUSION | Proxy reward rises monotonically while human or gold scores fall | Reward hacking after over-optimisation; proxy distribution shift; label noise | Measure the gold metric on a held-out population at several checkpoints | Stop at the gold-optimal checkpoint; fix the proxy; add constraints | Gold metric measured at every checkpoint, never after the fact | The selected checkpoint is gold-optimal, with an interval |
| CI-18 | SCALE_FAILURE | Policy drifts into degenerate phrasing and length collapse | KL coefficient too small; reward scale too large; advantage normalisation | Log KL per step and completion length distribution alongside reward | Raise the KL coefficient or fix the reward scale; re-run from a checkpoint | KL and length tracked as first-class metrics | KL stays in a planned band and lengths remain in range |
| CI-19 | NUMERICAL_EDGE | Half the training prompts contribute nothing and the loss curve is flat | Zero-variance groups; all-correct or all-wrong prompts; clipping saturation | Compute the fraction of groups with zero reward standard deviation | Filter or rebalance prompts; use a reward with finer resolution | Report zero-variance fraction per batch | Zero-variance fraction below a stated threshold on the training mix |
| CI-20 | CONCEPT_CONFUSION | A checkpoint advertised at 128k is incoherent past 8k | Rotary base mismatch; extension never trained; serving limit raised without training | Read the base constant and the trained maximum length from the config | Use the matching base or extend with training; cap the served length | Config fields for base, trained length, served limit | Long-context evaluation at the served limit passes |
| CI-21 | NUMERICAL_EDGE | Extension looks fine at 8k and degrades at 32k after an inference-only change | PI scale applied at inference but not at training; partial band mismatch; position offset | Compare the scale and band recorded in the checkpoint with the runtime's | Make the runtime read the recorded configuration | Config-driven position handling with no defaults | Identical outputs for a fixed prompt across two runtime builds |
| CI-22 | SCALE_FAILURE | One request fits, sixty-four do not, though weights are unchanged | KV cache per sequence; concurrency; fragmentation; prefix cache replication | Measure per-request cache growth against the concurrency plan | Cap concurrency, quantize cache, or shorten context | Budget cache as concurrency times context | Measured memory at target concurrency under the plan |
| CI-23 | CONCEPT_CONFUSION | "4-bit should halve bf16" but measured saving is 28% of bf16 | Scales and metadata uncounted; non-quantized modules; compute buffers | Recompute C3 from the config's group size and dtype of scales | Quote bytes per weight including overhead | Publish the storage identity with the config | Predicted and measured storage agree within 2% |
| CI-24 | VIOLATED_ASSUMPTION | Training diverges when batch size is raised with linearly scaled learning rate | Linear scaling invalid at large batch; warm-up too short; adaptive optimiser interaction | Sweep learning rate at the new batch with short runs | Use square-root scaling or retune; lengthen warm-up | Learning-rate protocol recorded with batch changes | Short-run loss curves are stable at the target batch |
| CI-25 | NUMERICAL_EDGE | A 2 pp improvement is reported from 400 items per arm | Interval wider than the effect; correlated items; multiple comparisons | Recompute C6 and the effective sample size | Report the interval, not the point; increase n | Pre-register the minimum detectable effect | The reported interval excludes zero for the claimed effect |
| CI-26 | CONCEPT_CONFUSION | ROUGE improves while reviewers prefer the base model | Overlap metric uncorrelated with the decision; length bias | Blind pairwise review on a fresh sample | Replace the metric or weight it with the human measure | Metric-to-decision mapping documented | Human preference and the automated metric agree in direction |
| CI-27 | SCALE_FAILURE | Post-restart loss jump, then slow recovery | Optimizer state missing; data order reset; base revision changed | Read the checkpoint manifest and compare the first post-resume step | Save and restore full state including data position | Resume drill before the long run | Post-resume loss within tolerance of pre-interruption |
| CI-28 | VIOLATED_ASSUMPTION | Special tokens appear as literal text in production | Tokenizer updated for training only; vocabulary mismatch; template drift | Compare vocabulary size and special-token ids across train and serve | Ship the tokenizer as part of the artifact | Tokenizer hash in the manifest and a startup check | Production encodes a probe string to the training ids |
| CI-29 | NUMERICAL_EDGE | Loss becomes non-finite on long sequences only | fp16 overflow in attention or loss; no loss scaling; bf16 not used | Run the same batch in bf16 and in fp16 with scaling, compare maxima | Use bf16 or raise scaling with clipping | Precision policy per tensor recorded in config | Finite loss and gradients at the maximum training length |
| CI-30 | CONCEPT_CONFUSION | "Fine-tuning will add the missing product knowledge" and it does not | Knowledge versus behaviour; token budget too small; retrieval not tried | Compare a retrieval baseline on the same questions | Use retrieval or continue pretraining | Baseline comparison before authorising adaptation | The chosen method beats the retrieval baseline on the target population |
| CI-31 | SCALE_FAILURE | The 7.6x memory saving is quoted as a 7.6x training speed-up | Parameter state versus step time; Amdahl bound; frozen forward dominates | Time a step with and without the adapter terms at equal batch | Quote the Amdahl bound, then measure | Cost model reviewed with the capacity plan | Measured step throughput is stated with its bound |
| CI-32 | VIOLATED_ASSUMPTION | Safety regression appears only in production review | Probe run at different decoding settings; different prompt template; different judge | Re-run the probe at the base's original decoding settings | Re-run paired and blinded at fixed decoding | Probe suite pinned to decoding configuration | Blinded review shows no material change in either direction |

**Co-occurrence and masking.** Records do not arrive alone, and they can cancel. CI-25 (an interval wider than the
effect) can hide a genuine regression that CI-17 (proxy divergence) would have explained, because the measurement cannot
see it. CI-12 (leakage) inflates a holdout score and can mask CI-11 (prompt tokens unmasked), both of which inflate the
same number. When two records are suspected, run the *cheaper* check first even if it belongs to the less likely cause,
because a defect you can rule out in five minutes is worth more than a hypothesis you can argue about for a day.

**Severity, assigned after the check.** CI-07, CI-08 and CI-32 are Sev-1 because they are correctness or safety defects
that no metric in this file would surface. CI-01, CI-02, CI-05, CI-06, CI-12 and CI-16 are Sev-2 because they invalidate
a decision input. The rest are usually Sev-3 cost or operability issues, with the exception of CI-17 and CI-18, which
become Sev-1 the moment a release gate depends on the proxy metric.

### CI-01 — the count does not match, and the factor is suspiciously round

**Mechanism.** Frameworks attach by name pattern. A pattern that matches `q_proj` inside `q_proj` and also inside a
fused `qkv_proj` attaches to one of them; a pattern that matches nothing attaches to nothing and the run proceeds with a
frozen model that looks like it is training. The reported count is then a clean multiple of the intended one, which is
why "the number is a third of what I computed" is the signature.

**Arithmetic that localises it.** The intended count for the reference model at rank 16 is 36,831,232, which is
1,150,976 per block. Suppose the run reports 13,631,488. Divide by 32 blocks to get 425,984 per block and match that
against C1: the four attention targets are `2*131,072 + 2*81,920 = 425,984` while the three MLP targets are
`3*241,664 = 724,992`, so the run targeted attention only and the MLP pairs were never attached. The complement,
`1,150,976 - 425,984 = 724,992`, is the missing term, which is the arithmetic form of the diagnosis.

The quotient narrows the family but does not identify the tensor: over the seven target shapes there are 128 subsets
but only 36 distinct per-block sums, because q and o share a shape at 131,072 and k and v at 81,920. So a per-block
quotient of 212,992 is consistent with `q + v`, with `o + k`, and with other combinations that happen to sum alike.
The name list separates those; the count never will.

**Why the wrong fix recurs.** The obvious response is to lower the rank so that the reported count matches the plan.
That makes the numbers agree while the model stays unadapted, and it converts a configuration defect into an evaluation
defect, because the run now has a plausible count and a flat loss curve.

### CI-02 — a square target hides an orientation error

**Mechanism.** The correct pair is `A` of shape 16x4096 and `B` of shape 4096x16. The transposed pair is `A` of shape
4096x16 and `B` of shape 16x4096. Both contain `16*4096 + 4096*16 = 131,072` scalars, so a square target passes every
count assertion, and the merge then adds a product of the wrong shape or reinterprets a reshape silently.

**Why rectangular targets are the test, not the square ones.** A `k_proj` at 4096x1024 forces the shapes to differ, so a
transposition fails a shape assertion immediately. The check is: merge one `k_proj`, score a fixed batch, compare with
the unmerged pair in fp32. A transposition produces a large difference; a precision difference produces a difference at
the rounding scale.

**Precision of the verification.** The tolerance must be stated before the check. Logits differ between evaluation
orders even when both are correct, so "identical" is the wrong criterion; a maximum absolute difference compatible with
the dtype's rounding behaviour is the right one.

### CI-03 — "the model is low rank"

**Mechanism.** The rank bound applies to `BA`. The matrix the model applies is `W + sBA`, and adding a bounded-rank
perturbation to a full-rank matrix generally leaves it full rank. A team that reads the bound as a statement about model
capacity will under-provision rank, and a team that reads it as a compression guarantee will assume a merged adapter is
smaller than it is.

**Check.** Take one trained layer, compute `W + sBA`, and report its numerical rank alongside the rank of `W`. The
result is almost always full rank. This is a five-line check and it ends the argument.

**Second-order consequence.** Because the effective matrix is full rank, the artifact is not compressible by discarding
the update's smallest directions; the merged weights occupy exactly the dense storage of the base.

### CI-04 — the rank sweep that swept two things

**Mechanism.** Under `alpha/r`, holding `alpha` fixed while doubling `r` halves the explicit scale. The training run
therefore changed capacity upward and step size downward, and the flat curve at high rank is as consistent with a small
step size as with overfitting.

**Check.** Log `alpha`, `r`, the convention name and the derived scale for every run. Two runs are comparable on rank
only if their derived scales agree within a stated tolerance, or if the comparison reports both axes.

**Numbers.** With `alpha = 32`, `r = 8` gives 4.000, `r = 16` gives 2.000, `r = 32` gives 1.000. Restoring the original
scale at `r = 32` requires `alpha = 128`. A sweep that never restated this produced three runs whose effective step
sizes differed by 4x.

### CI-05 — the memory model that blamed the adapter

**Mechanism.** C2 has seven terms and only two of them involve the adapter. When a run fails after someone increases the
training sequence length, the adapter's contribution is unchanged, and the change is entirely in the activation and
cache terms.

**Check.** Measure allocation after load, after the first forward, and after the backward, at two sequence lengths with
everything else fixed. If peak grows with length and not with rank, the adapter is exonerated in one run.

**Numbers.** At 4,096 tokens with microbatch 4 and gradient accumulation 8, a step holds 131,072 token positions. One
retained tensor of width 4096 per block costs `131,072 * 4,096 * 2 B = 1,073,741,824 B = 1.000 GiB`, so 32 blocks
retain 32.000 GiB when nothing is checkpointed. Doubling the sequence length doubles that term. The adapter's 562 MiB
does not move.

### CI-06 — the merge that lost half a point

**Mechanism.** Merging in bf16 and then requantizing applies two rounding steps to a quantity whose effect on outputs is
smaller than the rounding error in the worst case. The mathematics is exact; the artifact is not.

**Check.** Compare merged and unmerged logits in fp32 before requantization, then again after, on the same fixed batch.
The difference between those two comparisons is the cost of the rounding, isolated from the correctness of the merge.

**Bounded correction.** Merge in the highest precision the build allows, cast the merged weights once, and evaluate the
exact bytes that will be served. Evaluating the pre-cast artifact is evaluating a different model.

### CI-07 — the adapter is not a security boundary

**Mechanism.** Two tenants share base weights, tokenizer, runtime, batch and any prefix cache. The only thing separating
them is the identity attached to the request and the keys derived from it. If the adapter is selected per token, or a
cached prefix is keyed without the adapter identity, the separation is nominal.

**Check.** Send two requests from different tenants with deliberately overlapping prompts, and record the adapter
identity at every step that touches state. The failure appears as a cache hit that crosses identities.

**Why the correction must be structural.** Clearing caches restores behaviour for a while and reintroduces the defect at
the next warm start. The durable fix is to include the identity in every key and to assert the identity at each stage.

### CI-08 — staging and production disagree

**Mechanism.** An adapter is a delta against a specific checkpoint. Serving it against a later revision of the same
model name produces plausible, slightly wrong behaviour, because the base has moved under a delta that was fitted to the
old one.

**Check.** Compare the base commit hash, tokenizer hash and derived scale recorded at both sites, and compare a fixed
prompt's logits. Any of the three differing is sufficient to explain the discrepancy.

**Prevention that survives a rename.** Pin by content hash, not by version label. Labels are mutable; hashes are not.

### CI-09 — the head that was targeted by accident

**Mechanism.** A pattern matching `proj` or a pattern matching every linear layer with a large output dimension can
catch the language-model head, which changes token-level preferences directly. The count increase is small (577,536 for
`r = 16` on a 32,000 vocabulary), so the arithmetic barely flags it, while the behavioural change is large.

**Check.** Print the target list as module names and look for the head or an embedding. The count is a weak signal here;
the name list is the strong one.

**Boundary.** Targeting the head is legitimate when the task is genuinely about output distribution, such as a
constrained output format. It is not legitimate as a side effect.

### CI-10 — the adapter that switches mid-sentence

**Mechanism.** If the adapter is resolved per token or per batch, a batched generation containing two tenants will
interleave them. The symptom is not an exception; it is a stylistic drift and an occasional identity error in the tail.

**Check.** Trace one generation and record the adapter identity at each decode step, plus the cache key used at each
step. The identity must be constant for the request by construction.

**Scale note.** The defect is invisible in single-request testing, which is why it belongs to the scale-failure category
even though the fix is a correctness fix.

### CI-11 — loss falls in the tokens that do not matter

**Mechanism.** Causal language-model loss includes every token that has a target. If prompt tokens are unmasked, most of
the gradient describes reproducing the input, which the model already does well, and the loss falls for that reason only.

**Check.** Split the loss by token role for a fixed batch. If the decline is concentrated in prompt positions, the
adapter is learning to echo.

**Numbers.** At a 4:1 prompt-to-completion ratio, prompt tokens carry 80% of the loss, so a 10% decline measured on
prompt tokens alone moves the total by `0.8 * 10% = 8%` while completions contribute nothing.

### CI-12 — the holdout that was never held out

**Mechanism.** Sampling rows splits conversations, documents and near duplicates across the boundary. The model then
scores well on material it has effectively seen.

**Check.** Compute the number of groups present in both splits, and the near-duplicate rate across the boundary under a
stated similarity threshold. Both should be zero.

**Correction order.** Deduplicate first, then split by group, then split by time if the task has temporal structure.
Splitting before deduplication reintroduces the overlap through the near-duplicate pairs.

### CI-13 — packing without isolation

**Mechanism.** Concatenating examples shares the attention window across boundaries: token one of example B can attend
to the tail of example A. Positions are also wrong unless reset per segment, so the model learns spurious long-range
continuations.

**Check.** Decode one packed batch with boundaries marked and inspect what each boundary token can attend to. A canary
token in example A that changes example B's loss proves contamination.

**Trade-off.** Packing raises utilisation from 9.28% to about 100% on the reference data, which is a 10.8x compute
saving per token of sequence (`4096/380 = 10.78`); that is why the isolation bug is worth fixing rather than abandoning
packing.

### CI-14 — truncation removed the behaviour

**Mechanism.** The examples where the target behaviour occurs are the long ones. Truncation at a fixed length keeps the
short examples and cuts exactly the region that distinguishes them, so the adapter learns the format and not the task.

**Check.** Histogram training lengths, compute the truncated fraction per slice, and score the adapter per length bucket
against the base. A large truncated fraction plus a flat long bucket is diagnostic.

**Correction.** Raise the budget, filter the impossible examples, or split long examples at a documented boundary rather
than silently truncating.

### CI-15 — the preference run that changed nothing

**Mechanism.** Three separate defects produce the same flat curve: `beta` treated as a sampling temperature and applied
to decoding; a reference model that is actually the policy (so the log-ratio is identically zero); and data whose chosen
and rejected completions differ so little that the margin is noise.

**Check.** Print `beta`, the reference identity, and the distribution of the chosen-minus-rejected margin at
initialisation. A loss pinned at `log 2 = 0.6931` from the first step indicates zero or near-zero margins.

**Numbers.** For the pair in C4 the initial loss is 0.6856 and the gradient factor is 0.4963, both within 1% of the
neutral values. That proximity is the signature.

### CI-16 — two tokenizations, one pair

**Mechanism.** DPO compares log-probabilities of the same completion under two models. If the two pipelines apply
different templates, add a different EOS, or tokenize whitespace differently, they are scoring different strings, and the
difference of log-probabilities is meaningless even though both numbers are finite and plausible.

**Check.** Tokenize one pair once and score both models on the identical id sequence. Compare with the product path; any
difference in ids explains the whole discrepancy.

**Prevention.** Store the tokenized batch in the debug artifact, so the diagnostic does not require re-running the
pipeline.

### CI-17 — the proxy that kept improving

**Mechanism.** A learned reward is a proxy with error. Optimisation finds the error and raises the proxy by exploiting
it, while the target quality depends on the true objective. Section 5's curve pair shows the proxy peaking at a pressure
of 15 and the target at 3.5.

**Check.** Measure the gold metric at several checkpoints on a held-out population, and select the checkpoint by the
gold metric. If the gold measurement is only available once, at the end, the optimisation cannot be steered and the run
is a lottery.

**Correction.** Stop at the gold optimum, fix the proxy with better labels or constraints, or add an explicit penalty on
the known failure mode.

### CI-18 — drift, length collapse and the missing anchor

**Mechanism.** A policy-gradient run with an under-weighted KL term moves far from the reference. The reward rises while
completions shorten, repeat, or adopt stock phrasing; each of those is a proxy exploit rather than an improvement.

**Check.** Plot KL, mean completion length and reward on one axis of steps. A reward rise coinciding with a length
collapse is the pattern.

**Bounded correction.** Raise the coefficient, fix the reward scale so that a one-unit reward is a plausible quality
difference, and restart from the last checkpoint whose gold metric was acceptable.

### CI-19 — the groups that teach nothing

**Mechanism.** Group-relative advantages are standardised within a group. Prompts that the model always solves, or never
solves, have zero reward variance and therefore zero advantage for every rollout.

**Check.** Compute the fraction of zero-variance groups per batch and per curriculum stage. A high fraction means most
of the rollout compute produced no gradient.

**Correction.** Filter or rebalance prompts toward the difficulty band where outcomes vary, or use a finer-grained
reward so that partial success is distinguishable.

### CI-20 — the context length on the label

**Mechanism.** Three different numbers are called "context length": the trained length, the length supported by the
rotary configuration, and the served limit. A checkpoint with a small rotary base and no extension training is
incoherent well before its configuration can represent the positions.

**Check.** Read the base constant, the trained maximum length and the served limit from the checkpoint and the runtime,
and evaluate quality by length bucket up to the served limit.

**Numbers.** At head width 128 and base 10,000, the longest wavelength is about 54,410 tokens; representing a position
does not mean the model learned to use it. Extension by interpolation with `s = 8` maps 32,767 to 4,095.875, which is
inside the trained range and outside the range the attention pattern was fitted on.

### CI-21 — inference-only extension

**Mechanism.** Applying an interpolation scale at inference without the matching training produces a model whose
positions are compressed relative to what it learned. Shapes are identical, so nothing fails loudly; quality degrades
with length.

**Check.** Compare the scale and band recorded in the checkpoint with those the runtime applies, then compare outputs
for a fixed prompt across two runtime builds that differ only in this setting.

**Prevention.** Configuration-driven position handling with no silent defaults; the runtime must read the recorded value
or refuse to start.

### CI-22 — concurrency, not weights

**Mechanism.** KV cache is per sequence and grows with context. The weights are shared and fixed; the cache is not.

**Numbers.** For the reference model, `2 * 32 * 1024 * 2 = 131,072` bytes per token, i.e. 128 KiB. At 32,768 tokens
that is 4.000 GiB per sequence. Sixty-four concurrent sequences need 256 GiB, which is why a service that fits one
request can fail at load without any model change.

**Check.** Measure per-request cache growth against the concurrency plan, and include fragmentation in the measurement
rather than assuming it away.

### CI-23 — "4-bit should be half of bf16"

**Mechanism.** The packed payload is 0.5 bytes per weight, and the scales add bytes the slogan omits. At group size 64
with fp32 scales the true figure is 0.5625 bytes per weight, 28.125% of bf16, not 25%.

**Check.** Recompute C3 from the configuration's group size and scale dtype and compare with the measured resident
storage after load. A gap larger than a few percent means an uncounted module or a compute buffer.

**Second-order.** Double quantization removes 8.29% of the total at these settings, which is a real but modest gain and a
frequent rounding-error in planning documents.

### CI-24 — batch scaling

**Mechanism.** Linear learning-rate scaling is a heuristic derived under assumptions about gradient noise that fail at
large batches and interact badly with adaptive optimisers and short warm-ups. Doubling the batch and doubling the rate
can produce divergence that the smaller configuration never showed.

**Check.** Run short learning-rate sweeps at the *new* batch size and compare loss curves over the same token budget.

**Numbers.** A base rate of `2e-4` linear-scaled by 8 gives `1.6e-3`; square-root scaling gives `5.66e-4`. If the run
diverges at the first, the second is the cheaper hypothesis to test before blaming the data.

### CI-25 — a claim inside the noise

**Mechanism.** A 400-item evaluation at a 0.8 success rate has a 95% half-width of 3.92 pp. A reported 2 pp improvement
is smaller than the interval, so the observation is compatible with no effect, a real effect, and an effect in the other
direction.

**Check.** Recompute C6 with the actual n and p, then recompute with the effective n implied by the group structure if
items are clustered.

**Correction.** Report the interval, pre-register the minimum detectable effect, and increase n before claiming. If n
cannot be increased, the honest conclusion is that the effect was not distinguished.

### CI-26 — the metric that liked the wrong answer

**Mechanism.** Overlap metrics reward reproducing source tokens. A summary that copies more of the source scores higher
and is less useful, so optimisation toward the metric moves away from the decision.

**Check.** Run a blind pairwise review on a fresh sample and compare its direction with the metric's. Disagreement in
direction invalidates the metric for this decision, whatever its historical correlation.

**Correction.** Replace or reweight the metric, and record the decision it is meant to support alongside the number.

### CI-27 — the checkpoint that was not a checkpoint

**Mechanism.** An adapter export contains the factors. Resuming also needs optimizer moments, scheduler position,
gradient scaler state, data position and the base revision. Restoring the factors alone produces a loss jump that is
then misattributed to the data.

**Check.** Read the manifest, list every state tensor and counter, and compare the first post-resume step's loss with
the last pre-interruption loss on the same batch.

**Prevention.** A resume drill executed on a short run before the long one, with a stated tolerance on the next step.

### CI-28 — the tokenizer that shipped separately

**Mechanism.** Training adds special tokens or changes the template. The artifact ships without the tokenizer, so
serving encodes with the original vocabulary and the special tokens become literal text.

**Check.** Compare vocabulary size, special-token ids and a probe encoding between the training environment and the
runtime.

**Prevention.** The tokenizer is part of the artifact, pinned by hash, and a startup check encodes a probe string and
compares ids.

### CI-29 — finite at 2k, non-finite at 8k

**Mechanism.** fp16 has a narrow exponent range; attention logits and squared activations at long length can exceed it.
Without loss scaling the backward produces infinities, and the first symptom is a NaN loss on a subset of batches.

**Check.** Run the same batch in bf16 and compare the maximum absolute activation and attention logit. A large gap
localises the overflow.

**Correction.** Use a wider exponent dtype, or add scaling with clipping, and record the precision policy per tensor
rather than per run.

### CI-30 — fine-tuning as a knowledge pump

**Mechanism.** A token budget of 76.8 million tokens is 0.0132 tokens per parameter for the reference model. That is a
behaviour-scale budget: it can teach format, tone and decision policy, and it cannot reliably add facts that the base
does not represent.

**Check.** Run the retrieval baseline on the same questions. If retrieval answers them, adaptation was the wrong
instrument regardless of how well it was executed.

**Correction.** Use retrieval for facts, adapters for behaviour, and continued pretraining when neither suffices.

### CI-31 — the saving that was not a speed-up

**Mechanism.** Parameter state falls by a large factor because frozen weights need no gradients or moments. Step time
falls by much less because the forward pass and the input-gradient backward still traverse every frozen matrix.

**Check.** Time a step at equal batch with and without the adapter's trainable terms, then compare with the Amdahl
bound computed from the frozen fraction.

**Numbers.** With 85% of step time in work LoRA does not remove, the bound is `1/0.85 = 1.176`, i.e. at most +17.6%
throughput. Quoting the memory factor as a time factor overstates by an order of magnitude.

### CI-32 — the probe that moved, but the decoding moved too

**Mechanism.** Safety probes are sensitive to sampling temperature, top-p, stop sequences and the template. If the
adapter is probed with different decoding from the base, the comparison measures the configuration as much as the model.

**Check.** Re-run both arms with identical decoding and template, blinded, and compare harmful-compliance and
over-refusal rates with intervals.

**Prevention.** The probe configuration is pinned and versioned with the artifact, and any change to it invalidates
previous results rather than being merged into them.

**Takeaway:** Each record is one check away from being settled; the cost of the check, not the plausibility of the
cause, decides the order in which they are run.

## 9. Complexity & Cost Analysis

**Cost of diagnosing versus cost of guessing.** A wrong correction costs a training run. For the reference model, one
epoch at 76.8M tokens on the configured step is about 586 steps, and a diagnostic check that prints tensor names or
compares logits costs seconds. The expected value calculation is not close: run the check.

**Where the run time actually goes.** For adapter training on a quantized base, the step is dominated by dequantization
and the frozen matrix products, then by the input-gradient backward through those matrices, then by the adapter's own
small products, then by optimizer work on 36.8M scalars. The adapter's arithmetic share is 0.635%; its optimizer share is
larger relative to its size but still small; its memory share is 562 MiB of a footprint near 7.2 GiB with
checkpointing.

**The arithmetic of the Amdahl bound, restated for cost planning.**

```text
FORMULA [C13]: T_step >= T_frozen,  T_step = T_frozen + T_adapter
               speed-up from removing T_adapter <= T_step / T_frozen
WHERE: T_frozen is time in the forward pass and input-gradient backward through frozen
       parameters; T_adapter is time in adapter products and optimizer updates.
CONSTRAINT: Bound on the arithmetic only; a real implementation also pays launch overhead
       for the extra kernels, which can make the adapter path cost more than its FLOPs.
NUMERICAL EXAMPLE: T_adapter / T_step = 0.05 gives a bound of 1.053, i.e. +5.3%.
INTERPRETATION: Any claim of a large training speed-up from adapters requires T_frozen to be a
       small fraction of the step, which contradicts the premise that most of the model is
       frozen.
```

**Evaluation is the largest hidden cost, and the one that decides projects.** At 400 items per arm and five slices the
evaluation is 2,000 generations per arm, doubled for base and adapter, plus a blinded review. That is small. At 6,146
items per arm, required to distinguish 2 pp at a 0.8 success rate, it is 61,460 generations, which is comparable to the
training run's token count. This is why minimum detectable effects must be chosen before data collection: the
evaluation size follows from the claim, and the claim should follow from the decision.

```text
FORMULA [C14]: n_required = z^2 * p*(1-p) / (MDE/2)^2
WHERE: MDE is the smallest difference worth detecting; z = 1.96 for a two-sided 95% interval.
NUMERICAL EXAMPLE: p = 0.8, MDE = 0.02 -> n = 3.8416 * 0.16 / 0.0001 = 6,146.6 per arm.
       MDE = 0.05 -> n = 983.4. MDE = 0.10 -> n = 245.9.
INTERPRETATION: Halving the effect to be detected quadruples the sample; this is the single
       most useful planning number in the unit.
```

**Storage and lifecycle cost.** Each adapter artifact is 70.25 MiB in bf16 and 140.5 MiB in fp32. At 250 tenants that is
17.15 GiB or 34.3 GiB respectively, both of which are trivial next to the base. The cost that scales with tenant count
is not storage: it is the evaluation, review and rollback labour per artifact, which is why a per-tenant adapter is an
organisational commitment as much as a technical one.

**Cost of a missed defect, by category.** A missed NUMERICAL_EDGE typically costs one re-run. A missed
VIOLATED_ASSUMPTION costs a decision made on an invalid input, which is worse because it is invisible. A missed
SCALE_FAILURE costs a production incident at the size where it appears. A missed CONCEPT_CONFUSION costs repeatedly,
because the same misunderstanding generates new instances, and it is the only one whose cost does not amortise.

**Throughput of the diagnostic loop itself.** A team that can print allocation per phase, diff tokenizer ids, and
compare logits in fp32 can settle most of this catalogue within a day. Building those three capabilities is cheaper than
one avoided re-run.

**Takeaway:** Diagnosis is cheap, re-runs are expensive, and decisions made on invalid inputs are the most expensive of
all; budget for the check, not for the guess.

## 10. Key Theorems / Results Reference

**DERIVED — rank of a product.** `rank(BA) <= min(rank B, rank A)`, so a rank-`r` update is constrained to an at-most
`r`-dimensional image. This bounds the update and never the model; it is the source of CI-03.

**DERIVED — Eckart-Young.** The Frobenius-optimal rank-`k` approximation of a known matrix keeps the `k` largest
singular values, and the squared error equals the sum of the discarded squared singular values. For a target with squared
energy 46 and retained 45, energy retention is 97.83% while the residual norm is 1. This is why a matrix-level metric
can look excellent while a specific input is mapped to zero.

**DERIVED — factorisation is not identifiable.** `BA = (BR)(R^-1 A)` for any invertible `R`, so `r^2` parameter
directions are unidentifiable and factor values are not comparable across runs. Any diagnostic that inspects `A` or `B`
rather than `BA` or the outputs is measuring a gauge artefact.

**DERIVED — preference loss depends on margin differences.** In C4, adding a constant to both completions' log-probability
differences cancels exactly. Therefore the loss cannot detect a systematic scoring error, and a separate token-identity
check is required; this is CI-16's necessity, not a stylistic preference.

**DERIVED — rotary embeddings are relative.** Attention scores depend on `q_m . k_n` as a function of `m - n`; a
monotone position map preserves relative offsets only within the range where the map is approximately linear. Position
interpolation is exactly linear, so it preserves relative structure at the cost of collapsing resolution by `s`; base
adjustment is not linear in the same way and changes the frequency ladder instead.

**DERIVED — standard error and minimum detectable effect.** For independent binary items, `se = sqrt(p(1-p)/n)`, so the
sample needed to detect a difference scales as the inverse square of that difference (C14). Group structure reduces the
effective `n`, which is why clustered evaluation needs more items than the formula suggests.

**DERIVED — Amdahl's bound.** With a fixed non-removable fraction `f`, the maximum speed-up from eliminating the rest is
`1/f`. Applied to adapters, it converts "0.635% of the parameters" into "at most about the reciprocal of the frozen
fraction of step time", which is far smaller than a naive ratio.

**DERIVED — zero-variance groups yield zero advantage.** Standardising rewards within a group maps a constant group to a
constant zero vector after subtracting the mean, regardless of the denominator used for scaling. The prompt contributes
no gradient.

**SOURCED — adapter method.** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685,
introduces the frozen-base low-rank update and its merge property. Cited as the origin of the technique, not as evidence
about any workload here.

**SOURCED — quantized-base adaptation.** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs",
arXiv:2305.14314, describes 4-bit base storage with higher-precision compute and the scale-quantization trick that C11
models. Quantization behaviour is implementation-specific and OPEN for any particular runtime.

**SOURCED — extension by interpolation.** Chen et al., "Extending Context Window of Large Language Models via Positional
Interpolation", arXiv:2306.15595, describes rescaling positions to fit a trained range. The frequency and resolution
consequences in C5 follow from the definition, not from the paper's experiments.

**SOURCED — preference optimization.** Rafailov et al., "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model", arXiv:2305.18290, derives the pairwise log-ratio objective whose margin structure C4 uses.

**SOURCED — rank-stabilised scaling.** Kalajdzievski, arXiv:2312.03732, proposes `alpha/sqrt(r)` to prevent the effective
scale from shrinking monotonically with rank; CI-04's third variant comes from this.

**OPEN — every realised number.** Wall-clock time, achieved throughput, latency, accuracy on any real dataset and the
severity of any real incident are not measured here. The file is a diagnostic structure, not a measurement report.

**Review trigger.** A change to the base checkpoint family, the tokenizer, the runtime's position handling, the adapter
scaling convention, or the evaluation population invalidates the numeric examples and may retire records whose
mechanism no longer exists.

**Takeaway:** The algebraic results are portable; the numbers are portable only with their stated model, precision and
population.

## 11. Interview / Exam Questions

### Q1. A run reports 12,277,077 trainable parameters where you computed 36,831,232. What is the ratio, and what does it suggest?
**Answer:** exactly 3 in round terms, which no integer rank can produce, so the run is not a rank error alone; dividing by
the block count is what shows a second discrepancy.
**Reasoning:** `36,831,232 / 12,277,077 = 3.0000001`, i.e. a factor of three. A rank change would have to be
`16 / 3 = 5.33`, which is not an integer, so a pure rank error is ruled out. But dividing by 32 blocks gives
383,658.66, which is not an integer either, so the run cannot be "a subset of the seven named shapes at a constant
rank" over 32 blocks. The nearest attainable per-block sums are 372,736 and 405,504, so the observed configuration
differs from the intended one in at least two places — a target-set omission and a block-count or rank discrepancy.
Two simultaneous defects are the normal case, and this is exactly why the count check must print per-block detail
rather than a single total.
**Trap:** Halving the rank to make the numbers agree.
**Follow-up:** Which single print statement settles it? The list of `requires_grad` tensors with names and shapes, grouped by block index.

### Q2. The count is right and quality after merging is 1.5 pp below the unmerged pair. Give the two leading causes and the check that separates them.
**Answer:** Requantization after the merge, or a transposed target on a rectangular projection.
**Reasoning:** A merge in bf16 followed by requantization applies rounding twice to a small update; comparing merged and
unmerged logits in fp32 before requantization isolates the rounding, and a large fp32 difference instead points at the
orientation. Square-only test suites pass in both cases, so the check must use a target where `d_in != d_out`.
**Trap:** Reverting to unmerged serving and declaring merge unsafe, which hides a genuine configuration bug.
**Follow-up:** What tolerance do you state before running the check? A maximum absolute logit difference compatible with
the dtype's rounding, recorded in the test.

### Q3. Compute the storage of a 5,801,771,008-parameter base at 4 bits with group size 64 and fp32 scales, and compare with bf16.
**Answer:** 3.039 GiB against 10.807 GiB, a ratio of 0.28125.
**Reasoning:** Packed weights are `5,801,771,008 * 0.5 = 2,900,885,504` bytes; there are `5,801,771,008 / 64 =
90,652,672` groups, whose fp32 scales occupy `362,610,688` bytes; the total is 3,263,496,192 bytes, i.e. 3.039 GiB.
bf16 is 11,603,542,016 bytes, i.e. 10.807 GiB. The ratio is 9/32 because the true cost is 0.5625 bytes per weight.
**Trap:** Using 0.5 bytes per weight and reporting a clean 4x.
**Follow-up:** What does double quantization change? It quantizes the scales themselves, saving 8.29% here.

### Q4. A 400-item evaluation shows a 2 pp improvement at a 0.8 success rate. What is your conclusion?
**Answer:** Not distinguished; the interval half-width is 3.92 pp, twice the claimed effect.
**Reasoning:** `se = sqrt(0.8*0.2/400) = 0.02`, so the 95% half-width is `1.96*0.02 = 0.0392`. To halve the half-width
to about 2 pp the sample must quadruple to roughly 1,600 items per arm; to establish a 2 pp effect at a 1 pp half-width
needs 6,146 items per arm.
**Trap:** Calling the improvement "significant" because the point estimate is positive, or because a t-test on the same
data returns a small p-value from a one-sided framing.
**Follow-up:** How does grouping change the answer? It reduces the effective n, so the required item count rises.

### Q5. Diagnosis: training loss falls 8%, held-out exact match does not move, and the prompt-to-completion ratio is 4:1.
**Answer:** The gradient is dominated by prompt tokens that were not masked.
**Reasoning:** At 4:1, prompt tokens carry 80% of the loss, so a total decline of 8% is exactly what a 10% decline on
prompt positions with zero change on completions produces (`0.8 * 10% = 8%`). That is the echo-learning signature. Split
the loss by token role to confirm before changing anything else.
**Trap:** Increasing rank, which adds capacity to a signal that is aimed at the wrong tokens.
**Follow-up:** What else could produce the same pattern? Leakage through a row split, which the group-overlap check rules
out cheaply.

### Q6. Two candidates explain an out-of-memory failure: rank too high, or sequences too long. Which check settles it, and what will the numbers look like?
**Answer:** Sweep sequence length at fixed rank and microbatch, then sweep rank at fixed length; the term that moves peak is the cause.
**Reasoning:** Adapter weights and state are 562 MiB and do not depend on sequence length. Activations at 4,096 tokens
across 32 blocks are about 32 GiB uncheckpointed, doubling with length. A peak that tracks length and not rank
exonerates the adapter in one run.
**Trap:** Halving rank, which changes the peak by 281 MiB and appears to help only because the run is near a boundary.
**Follow-up:** Which single configuration change has the largest effect? Checkpointing, which cuts retained activations
by roughly the block count.

### Q7. Why can a DPO loss be unable to detect a scoring bug, and what test does that imply?
**Answer:** Because the loss depends only on margin differences, and a bug that shifts both completions equally cancels.
**Reasoning:** Substituting `log pi + c` for both chosen and rejected leaves `(m_pi - m_ref)` unchanged, so the loss and
the gradient are identical while the model is being trained on the wrong token sequence. The implication is a test on
the identity of the scored token ids, not on the loss value.
**Trap:** Monitoring the loss for anomalies as the only DPO health check.
**Follow-up:** What does a loss pinned near 0.6931 indicate? Near-zero margins, which is CI-15's signature.

### Q8. Compute the KV cache for 32,768 tokens on the reference model and for 64 concurrent requests.
**Answer:** 4.000 GiB per sequence and 256 GiB for 64 sequences.
**Reasoning:** Per token, `2 * 32 layers * 1024 kv-width * 2 bytes = 131,072` bytes, i.e. 128 KiB. Multiply by 32,768
gives 4,294,967,296 bytes, i.e. 4.000 GiB; multiply by 64 for 274,877,906,944 bytes, i.e. 256 GiB.
**Trap:** Quoting per-token cache as if it were a fixed model cost, or assuming batch serving amortises it.
**Follow-up:** Which lever reduces it most? Context length, since the term is linear in tokens per sequence.

### Q9. A rank sweep shows no gain above 8 and a flatter curve at 32. What two things moved?
**Answer:** Capacity rose and the effective scale fell, because `alpha` was held fixed under the `alpha/r` convention.
**Reasoning:** At `alpha = 32`, `r = 8` gives 4.000 and `r = 32` gives 1.000, a 4x reduction in the explicit scale.
Restoring the scale requires `alpha = 128`. A flatter curve is equally consistent with a smaller step size and with
overfitting, so the sweep cannot support either conclusion.
**Trap:** Concluding that rank above 8 overfits, then applying early stopping.
**Follow-up:** What would make the sweep interpretable? Holding the derived scale constant or using the
`alpha/sqrt(r)` variant and reporting it.

### Q10. Diagnosis: proxy reward rises from 1.0 to 2.2 while the gold metric falls from 1.0 to 0.4. What happened, and what should have been measured?
**Answer:** Reward hacking; the gold metric should have been measured at every checkpoint.
**Reasoning:** The proxy in Section 5 peaks at a pressure of 15 while the target peaks at 3.5, so every unit of proxy
gain past 3.5 costs quality. Selecting the final checkpoint selects a model that is optimal for the proxy's error. With
per-checkpoint gold measurements the run could have stopped at 3.5.
**Trap:** Concluding the policy optimiser is broken, when the optimiser did exactly what it was asked.
**Follow-up:** What is the cheapest mitigation? A gold-metric gate on checkpoint promotion, plus a constraint or penalty
on the known exploit.

### Q11. A 32k advertised checkpoint is incoherent past 8k. What three numbers do you read, and in what order?
**Answer:** The rotary base constant, the trained maximum length, and the served limit.
**Reasoning:** The base constant determines the longest wavelength the configuration can represent; the trained length
determines what the model learned; the served limit determines what is permitted. A model with a small base and no
extension training has all three disagreeing, and the observable failure begins near the trained length, not the served
limit.
**Trap:** Raising the served limit further to "give it more room".
**Follow-up:** What does interpolation with `s = 8` do to resolution? It maps 32,767 to 4,095.875, collapsing eight
requested positions into one.

### Q12. Why is the parameter-state reduction of about 7.6x not a 7.6x training speed-up? Show the bound.
**Answer:** Because the frozen forward and input-gradient backward remain, and the bound is the reciprocal of that fraction.
**Reasoning:** Removing the adapter's arithmetic removes 0.635% of parameter-token work. If frozen work is 85% of step
time, the bound is `1/0.85 = 1.176`, i.e. +17.6%. The adapter's optimizer work and its extra kernel launches are the
only things that actually disappear.
**Trap:** Dividing total parameters by adapter parameters and calling the quotient a speed-up.
**Follow-up:** Where does the 7.6x apply? To parameter-related memory: 86.45 GiB of fp32 full-fine-tuning state against
about 11.36 GiB with a bf16 base and an adapter.

### Q13. How do you distinguish a zero-variance GRPO batch from a broken reward function?
**Answer:** Compare the reward distribution's variance within groups against its variance across groups.
**Reasoning:** A functioning reward that the model solves uniformly produces low within-group variance and high
between-group structure; identical rewards everywhere, including across prompts, points at the reward function. The
distinguishing check is per-group standard deviation alongside the across-group mean spread.
**Trap:** Watching mean reward, which is flat in both cases.
**Follow-up:** What is the operational consequence? The zero-variance fraction belongs on the training dashboard as a
first-class metric, because it quantifies wasted rollout compute.

### Q14. Someone proposes an adapter as the tenant isolation mechanism. What is wrong, and what is the correct control?
**Answer:** Adapters share base weights, tokenizer, runtime, batching and caches; isolation is an authorization and routing property.
**Reasoning:** The factors carry no tenant identity, and nothing prevents a request from being routed to the wrong
factors or a cached prefix from being reused across them. The correct control is to bind the tenant and adapter identity
into the request, into every cache key, and into the access log, with negative tests that attempt cross-tenant access.
**Trap:** Pointing at per-tenant artifacts as evidence of separation.
**Follow-up:** Which test would have caught CI-07? Two tenants with overlapping prompts, replayed concurrently, with an
assertion that each response uses its own adapter identity.

### Q15. You must choose between re-running training with a lower rank and printing per-phase memory. Which, and why?
**Answer:** Print per-phase memory, because it costs minutes and can eliminate the hypothesis entirely.
**Reasoning:** Rank reduces trainable state by 281 MiB per halving, which is 5% of the live footprint at the reference
configuration; if the failure is in the activation term, the re-run will fail again after consuming a training slot. A
phase-wise allocation trace either shows the binding term or shows that the adapter is irrelevant to the peak.
**Trap:** Treating the re-run as cheaper because it is a single command.
**Follow-up:** What if the trace shows the base load itself is the peak? Then the fix is dtype, sharding or offload, and
rank was never a lever.

**Takeaway:** Diagnostic questions are decided by which check discriminates, not by which explanation sounds most
familiar.

## 12. Quick Reference Card

| Signal | Record | First check | Bounded correction |
|---|---|---|---|
| Count is a clean fraction of plan | CI-01 | Print `requires_grad` names and shapes | Allow-list the target modules |
| Count right, merged quality wrong | CI-02 | Merge a rectangular target, compare fp32 logits | Fix orientation, test on a non-square shape |
| "The model is low rank" | CI-03 | Rank of `W + sBA` for one layer | Restate the claim as an update bound |
| Rank sweep uninformative | CI-04 | Log `alpha`, `r`, derived scale per run | Fix the scale, vary rank |
| OOM with no rank change | CI-05 | Phase-wise allocation at two lengths | Shorten length or checkpoint |
| Adapter invisible in production | CI-08 | Base hash and tokenizer hash at both sites | Pin by content hash, add a logit self-test |
| Proxy up, gold down | CI-17 | Gold metric per checkpoint | Select the checkpoint by gold metric |
| 2 pp claim from 400 items | CI-25 | Recompute `sqrt(p(1-p)/n)` | Report the interval, or raise n to 6,146 |
| 4-bit saving is 28%, not 50% | CI-23 | Recompute storage with group size and scale dtype | Publish bytes per weight including overhead |
| Tenant behaviour crosses over | CI-07 | Trace adapter identity through one request | Bind identity into every cache key |

The five numbers worth remembering: 0.635% (adapter share of parameters), 562 MiB (its fp32 training state), 3.039 GiB
(4-bit base with fp32 scales at group 64), 128 KiB per token (KV cache on the reference model), and 6,146 items per arm
(to detect 2 pp at a 0.8 success rate).

Nothing follows this card; use Section 8 for the full records and Section 11 for rehearsal.
