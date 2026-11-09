---
collection: "module-17-vision"
file_id: "03"
topic: "Vision transformers"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Tokenisation, resolution, positional encoding, or target-image distribution change."
notation_style: "plain-text"
prerequisites: ["02_CNNs_Attention_and_Transfer_Learning.md"]
pattern_tags: ["vision-transformer", "attention"]
source_status: "OPEN — source bodies unavailable"
owner_role: "Vision ML reviewer"
source_index: "INDEX.md"
---

# Vision Transformers

> **Purpose:** Turn image resolution and patch size into a token and attention-cost budget, then select global, windowed, or convolutional structure.
> **Prerequisites:** [CNNs, attention and transfer learning](02_CNNs_Attention_and_Transfer_Learning.md); matrix multiplication.
> **Key outcome:** Given H, W, patch size and embedding width, calculate token count and attention-score payload; identify an invalid resolution change.
> **This file does NOT cover:** detector heads, video temporal attention, or foundation-model pretraining recipes.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED geometry/cost; VERIFIED illustrative arithmetic; SOURCED paper definitions; no benchmark claim.
> **Retrieval cue:** If an image-resolution change causes OOM or a metric regression, start at Sections 4 and 8.

## 1. File Header

A vision transformer (ViT) divides an image into patches, embeds them as tokens, adds position information, and applies transformer blocks. The visible inventory explicitly names `09_Vision_Transformers_ViT.md` at [`directory.txt`](../../../../Templates/directory.txt#L557); the original content was not accessible. This sheet teaches the visible topic without claiming reconstruction.

**Takeaway:** Patch size is both a representation decision and a quadratic resource decision.

## 2. Concept Map / Mental Model

```text
image [H,W,C] -> non-overlapping P x P patches -> flatten/project -> N tokens [N,D]
                                                      + position information
                                                                  |
                                              Q,K,V -> attention scores [N,N] -> mixed tokens
                                                                  |
                                                     class/region head -> decision
resolution or smaller patches -> N grows -> score matrix grows as N^2
```

**Takeaway:** Inspect N before blaming the device for an attention-memory failure.

## 3. Foundational Definitions

| Term | Meaning | Shape / units | Boundary |
|---|---|---|---|
| patch | contiguous P-by-P pixels | [P,P,C] | P must match preprocessing policy |
| token | projected patch representation | [D] | token order requires position information |
| D | embedding width | scalar features/token | not image dimension |
| h | attention heads | positive integer | D normally divisible by h |
| global attention | every token can score every token | [N,N] scores/head | costly as N grows |
| window attention | tokens attend within local windows | window-dependent | loses direct global links unless staged |

**Takeaway:** Token count, not pixel count alone, controls the first-order attention matrix size.

## 4. Core Formulas with Full Derivation

Validity before use: H and W are divisible by square patch P, no class token included initially, and patches do not overlap. Counting P-wide cells along each axis gives H/P and W/P.

```text
FORMULA [VIT-01]: N = (H/P)*(W/P)
WHERE: N patch tokens [tokens]; H,W image dimensions [pixels]; P patch side [pixels].
CONSTRAINT: H mod P=W mod P=0; otherwise pad/crop or use the implemented tokenizer rule.
NUMERICAL EXAMPLE: H=W=224, P=16: N=14*14=196 tokens.
INTERPRETATION: adding one class token makes the transformer sequence length 197.
```

Validity before use: standard full self-attention with h heads and no sparse kernel. Each head compares every query token with every key token.

```text
FORMULA [VIT-02]: A = h*N^2
WHERE: A attention-score scalars; h heads; N sequence tokens.
CONSTRAINT: counts scores only, excluding Q/K/V projections, activations, gradients and kernel workspace.
NUMERICAL EXAMPLE: h=12, N=197 gives A=12*197^2=465,708 scores.
INTERPRETATION: at FP16, scores alone occupy 931,416 bytes (~0.888 MiB) per example before training overhead.
```

A 448x448 image at P=16 has N=784, exactly four times 196; N squared is 16 times larger. This is a derived ratio, not a measured memory multiplier.

Counterexample: interpolation of positional embeddings may permit a new grid mechanically, but it does not establish target accuracy or calibration.

**Takeaway:** Doubling both image dimensions can multiply dense score storage by sixteen.

## 5. Worked Numerical Example

**ILLUSTRATIVE, VERIFIED arithmetic.** An inspection image is increased from 224 to 448 pixels to preserve a small mark. With P=16, N rises 196 to 784. Twelve-head score count rises from 460,992 if no class token is counted (12*196^2) to 7,375,872 (12*784^2), exactly 16x. If a 10 GiB device had only 1 GiB genuinely free after model, batch, and framework overhead, the score scalars alone are not the whole budget; test batch=1 peak allocation with fixed code before promising resolution.

Sensitivity: change P from 16 to 8 at 224 resolution. N rises 196 to 784, the same 16x score ratio, but local patch content is finer. The choice is therefore a task-resolution hypothesis needing size-slice evidence, not a default improvement.

**Takeaway:** Use a single-token budget table across every candidate resolution and patch size.

## 6. Visual / Geometric Intuition

```text
224 image, P=16: 14 x 14 patch grid  -> coarse spatial tokens
224 image, P=8:  28 x 28 patch grid  -> finer tokens, much larger all-pairs map
windowed design: [local grid] [local grid] -> staged cross-window mixing
```

A patch projection discards within-patch arrangement unless later structure retains it. Smaller patches make more detail available but demand more computation and may reduce data per token. Position embeddings identify order; without them, token permutation is ambiguous to ordinary self-attention.

**Takeaway:** Patching is controlled information compression, not just an implementation detail.

## 7. Comparison / Selection Table

| Option | Use when | Benefit | Avoid when | Check / fallback |
|---|---|---|---|---|
| global ViT | global layout matters and N fits | direct all-token interaction | high-resolution budget fails | measure peak memory; use hierarchy |
| hierarchical/windowed transformer | resolution is high | bounded local cost | task requires direct distant interaction | add shifted/global stage |
| CNN backbone | locality/latency and labels dominate | strong inductive bias | global relation is decisive | compare grouped target baseline |
| hybrid CNN+attention | local texture and global layout both matter | staged representation | added complexity lacks evidence | ablate each component |

**Takeaway:** Select the cheapest architecture that preserves the task-relevant field of view.

## 8. Failure Modes & Edge Cases

| Symptom | Competing causes | Discriminating check | Fix | Verification |
|---|---|---|---|---|
| OOM after resize | N-squared scores or unrelated batch/workspace change | log N, sequence length, peak allocation | larger patch/window/batch reduction | fixed-batch memory test |
| positional regression | interpolation mismatch or shift | compare native and interpolated grids by slice | retrain/adapt with target resolution | held-out resolution suite |
| small-object miss | patch too coarse or labels incomplete | recall by object size/adjudication | reduce patch or use dense head | size-slice gate |
| high score, new-site failure | shortcut or split leakage | site-held-out and background perturbation | diversify/adapt after audit | site metric + calibration |
| unstable training | scale/LR/normalisation issue | loss, gradient and activation traces | controlled schedule change | fixed-seed short run |

**Takeaway:** Treat resolution changes as architecture changes that require a new validation gate.

## 9. Complexity & Cost Analysis

Projection cost is approximately N*(P*P*C)*D multiplications; full score construction is approximately h*N*N*d_h=N*N*D when D=h*d_h. These counts exclude softmax, output projection, data loading, and hardware effects. At 224/P16/C3/D768, patch projection has 196*768*768 = 115,605,504 multiply terms; this numeric coincidence arises because P*P*C=768. The value is an illustrative operation count, not latency.

**Takeaway:** Report both projection and all-pairs terms; neither is a benchmark.

## 10. Key Theorems / Results Reference

- Self-attention is permutation-equivariant without positional signals; position treatment is therefore an architectural requirement for images.
- [Dosovitskiy et al., *An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), 2020, defines the ViT patch-token formulation; reviewed 2026-11-09.
- [Liu et al., *Swin Transformer*](https://arxiv.org/abs/2103.14030), 2021, is a source for shifted-window hierarchy; it does not validate a particular deployment.

**Takeaway:** The cited design provides a mechanism, not a universal selection result.

## 11. Interview / Exam Questions

1. **Compute N for 384x384, P=16.** 24*24=576.
2. **Why is 448 vs 224 expensive?** Token count quadruples and dense score count grows sixteenfold at fixed patch.
3. **What does a class token not provide?** Pixel/box-level localization by itself.
4. **When use windowing?** When global attention's token budget fails and staged locality remains sufficient.
5. **Why can positional interpolation fail?** New grid geometry can change learned spatial relations; test it.
6. **How distinguish OOM causes?** Hold model/batch code fixed and log sequence length, allocation, workspace and fragmentation.
7. **Why compare CNN baseline?** It tests whether global mixing materially improves the target decision.
8. **A smaller patch improves random-split accuracy only. Next step?** Evaluate grouped deployment split before selecting it.
9. **Does token count establish latency?** No; it is a workload proxy, not observed hardware timing.
10. **Changed constraint: 50 ms limit.** Measure end-to-end target latency, then consider windowing, resolution or a CNN; do not infer from score count alone.

**Takeaway:** Defensible selection combines geometry, task slices, and measured operational limits.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Token count | VIT-01: N=(H/P)(W/P), with actual tokenizer exceptions stated |
| Score count | VIT-02: hN^2; include class tokens where used |
| Resolution change | Recompute N and rerun grouped target evaluation |
| Position | Treat interpolation as a testable adaptation, not evidence |
| Selection | Compare CNN, global, and hierarchical options under same split |
| Avoid | Calling operation counts latency or random-image scores generalisation |
| Evidence | Illustrative numbers only; source bodies unavailable |
