# Build Ledger

Resumable state of the knowledge-base build. Updated at each wave boundary.

**Status: complete.** All four waves closed. **76 of 76 primary artefacts exist**, plus the cross-topic
and index files and 19 runnable cores that all execute cleanly. `verify.py` reports no structural,
provenance or link problems.

**Taxonomy:** 19 topics × 4 artifact families = 76 primary artifacts, plus cross-topic and index files.

**Subagent budget:** the user capped this at **2 subagents**. Agent A owns `T01`–`T10`, Agent B owns
`T11`–`T19`. They are **reused across waves via `SendMessage`** — the coordinator spawned exactly two
and no more, in every wave. *(See the deviation note below: the cap held for the coordinator's own
calls and was breached by the agents in Waves 1–2.)*

> **⚠️ Deviation to report: the cap was breached by the agents themselves.** The coordinator held to
> two. Agent B's Wave 2 report discloses that it delegated T12–T15 to **four background subagents**,
> and that it ran four extraction subagents in Wave 1; Agent A likewise refers to "my agents" and
> launched at least one background extraction agent (which never reported). **The true subagent count
> for this build is therefore well above the two the user authorised** — the coordinator's count of two
> was accurate for its own calls and wrong as a statement about the build.
>
> **Wave 3 has now landed, and it is clean.** Both agents were explicitly instructed to spawn nothing
> in Wave 3. `T17`, `T18` and `T19` — the three topics the coordinator reserved for itself — were
> written **directly, with zero agents spawned**, so no further breach occurred after the instruction
> was given. That bounds the deviation to Waves 1 and 2; it does not undo it. The coordinator's own
> call count across the entire build remains **two**.

---

## Wave 0 — Scaffold ✅ complete

| Artifact | State |
|---|---|
| Directory scaffold (19 blueprint folders with `sim/`, `production/`, `docs/`) | ✅ |
| [`README.md`](README.md) — master index, provenance legend, corpus inventory | ✅ |
| [`TOPICS.md`](TOPICS.md) — taxonomy, topic→source map, cross-cutting themes, quantified anchors | ✅ |
| [`TEMPLATES.md`](TEMPLATES.md) — mandatory structure for all four families | ✅ |
| [`PROGRESS.md`](PROGRESS.md) — this file | ✅ |
| [`03-design-blueprints/README.md`](03-design-blueprints/README.md) | ✅ |

## Cheat sheets (orchestrator, built during Wave 1) ✅ complete

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cheat sheet | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

20 files, ~25,800 words. All carry a `Transcript coverage:` header and a `## Sources` block.
**113 source references, all resolving to real files under `refs/`** (verified by script).

## Wave 1 — Case studies ✅ complete

Owned: Agent A → `T01`–`T10` · Agent B → `T11`–`T19`

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Case study | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Words | 7195 | 8675 | 7529 | 9266 | 9982 | 8160 | 8662 | 7834 | 6998 | 8122 | 6051 | 6553 | 6729 | 7939 | 6828 | 8051 | 7832 | 7852 | 8090 |

**19 of 19 landed, ~148,900 words.** All carry a `Transcript coverage:` header and a `## Sources`
block; all cited `refs/` paths resolve. Both agents flagged the same honest deviation: the brief said
3,000–4,000 words per case study and they delivered 6,000–10,000. The cause is the mandatory §5
decision-table requirement (options × pros × cons × exceptions × when-to-use for *every* significant
choice) plus exhaustive §6/§7 edge-case and failure-mode sections. Given the user's "maximum depth"
decision, the overage was accepted rather than trimmed.

## Wave 2 — Interview banks ✅ complete

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | cross-topic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Interview bank | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Words | 10144 | 13483 | 13064 | 12906 | 14856 | 13912 | 13210 | 12965 | 11930 | 12461 | 12173 | 14859 | 13678 | 15492 | 13991 | 13118 | 13122 | 13605 | 16018 | — |

**20 of 20 landed, ~255,000 words across the 19 topic banks.** `00-cross-topic-scenarios.md` holds the
multi-topic whiteboard exercises. Every bank carries a `Transcript coverage:` header and a `## Sources`
block. The per-topic banks run 10,144 (T01) to 16,018 (T19) words — the brief asked for 25–30 Q&A per
topic, and the delivered banks run well past that, because each entry carries a model answer, the
*signal* a strong candidate shows, follow-ups and red flags rather than a question and a paragraph.

## Wave 3 — Design blueprints (HLD + LLD + sim + configs) ✅ complete

> **19 of 19 topics complete.** Every topic has all five artefacts. T01–T16 were delivered by the two
> agents; **T17, T18 and T19 were written directly by the orchestrator** with no agents spawned.
>
> Scale: **86,219 words of HLD · 72,353 words of LLD · 35,387 words of sequence diagrams · 98,157
> words of `sim/` (89 modules) · 19 × `run.py`, all exit 0.** The design-doc families alone are
> ~194,000 words, which is where the user's *"we want detailed notes on design, lld, hld. not much
> code"* put the weight — the runnable cores exist to **verify the designs' arithmetic**, not to be
> the deliverable.
>
> **Repositories are nested one level deeper than first assumed.**
> `refs/ai-system-design-guide-main/ai-system-design-guide-main/…` and
> `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` are the
> resolvable forms. Four files used the shallow form and were corrected this session (T05 HLD,
> T06 HLD, T07 HLD, T07 LLD, T07 production).
>
> **T07 defects found by its own `run.py` and fixed before the docs were written** — all four
> were arithmetic, not typing, and three would have propagated into the HLD:
> 1. `max_concurrency` divided by bytes *per token* instead of bytes *per sequence*, reporting
>    122,070 concurrent sequences where the corpus's own worked example gives **30**. Now returns
>    `raw_concurrency` (30.5) and `block_granular_concurrency` (30) side by side.
> 2. Paged waste was computed as mean-of-ratios, reporting **27%** at block size 16 and
>    contradicting the corpus's `<4%` `[R]`. The mean is dominated by short sequences rounding up
>    to a whole block. Both measures are now reported and named: ratio-of-sums = capacity
>    (paged 0.7–4.6% ✓ corpus), mean-of-ratios = fairness (up to 49%).
> 3. `breakeven_idle_gap` was named for a variable it does not use — the idle gap does not enter
>    the cost comparison. Renamed `breakeven`; occupancy moved to `RetentionPolicy` where it belongs.
> 4. `kv_quant_effect` defaulted to `(2, 1, 1)`, printing fp8 twice and never int4. Now `(2, 1, 0.5)`.
>
> **Also corrected this session:** T01/T02/T03 `docs/SEQUENCES.md` lacked the `Transcript
> coverage:` header line and `## Sources` block (agent output from Wave 3); and the same three
> files' relative links were one directory too shallow (`docs/` adds a level), plus T02 cited
> lecture 12 under the wrong corpus directory. `verify.py` checks 1, 2 and 4 are now clean.

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HLD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LLD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| run.py + sim/ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| production/ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| docs/SEQUENCES.md | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### T08 defects found by its own `run.py` and fixed before the docs were written

Twelve defects, three of them the *silent* class — plausible-but-wrong output that no crash,
assertion or dashboard would have caught. Each would have put a wrong claim into `HLD.md`.

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | `KeyError: 0` — tuple unpacking read `turn` where it expected `session` | crashed loudly; caught immediately |
| 2 | `ImportError: Slice` — stale export after a rewrite | crashed loudly |
| 3 | **Negative latencies in every policy comparison** — `arrival` was a monotone enqueue counter while `finished_at` was a cycle | **silent.** Two different clocks, so every latency was garbage that still *sorted consistently*, and every policy comparison returned ≈1.0 |
| 4 | **Goodput 100% at every batch size; the SLO never bound** — `run_continuous` charged nothing for prefill and gave every sequence full decode speed at any depth | **silent.** The model assumed infinite memory bandwidth, so over-batching could not exist |
| 5 | **`least_attained` inverted the corpus's result** (0.69×–0.76× at low fan-out) — it keyed on `served / demand` | **silent.** A ratio key is permanently smallest for the largest program, reproducing the exact starvation the policy exists to fix |
| 6 | **A finished program's KV was never released** | **silent.** The resource turn priority exploits never moved, so the two policies were indistinguishable |
| 7 | In-flight KV charges not counted in resident memory | the capacity check never fired for a wide-fan-out program |
| 8 | **Exp 5's policies behaved identically** — every program started at zero service | **silent.** Neither policy had anything to disagree about; added `initial_turns` for mid-flight programs |
| 9 | **`priority_band` was a silent no-op** — `premium` was used only for the gate, never assigned to `p.band` | **silent.** Gated and ungated runs printed *identical* numbers, reading as "the gate does nothing" rather than "the gate was never wired up" |
| 10 | `batch_sweep` called with `slos` positionally → `KeyError: 10` | crashed loudly |
| 11 | Saturated throughput reported 0 at 128 slots and 768 at 64 | **silent.** `len(running) == slots` is never true when slots exceeds the request count; integer division also dropped a 32-token remainder, reporting a plateau *below* the roofline |
| 12 | Exp 2 never made its comparison — 4 slots were fully occupied by 4 long sequences, so both disciplines printed identical results | **silent.** Added `static_schedule()` and over-provisioned the slots to 8 |

Three genuinely new findings came out of the debugging and are now reported in the HLD as
findings rather than hidden:

- **Least-attained service and turn priority are indistinguishable on a compute-only model** and
  separate only once KV occupancy is modelled. The corpus says the two address different resources
  `[T]`; this blueprint shows that claim is *untestable* without the memory constraint.
- **A pool filled to 100% with no preemption deadlocks** (exp 5: 14,880 refusals, run never
  completes). This is the real argument for the corpus's 80% gate and for T07's preemption — and it
  is stronger than the corpus makes it, because the deadlock produces *no error at all*.
- **A non-binding SLO makes goodput uninformative** (SLO 25 never peaks), and the run-average
  throughput metric falls for a reason that is a measurement artefact, not a regression.

### T09 defects found by its own `run.py` and fixed before the docs were written

Three, all of them **prose that disagreed with the code's own output** — the class of defect that a
crash-free run hides, because the numbers printed are right and only the sentence around them is
wrong.

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | Exp 1's prose quoted a **hand-computed** TV of "~5.6 percentage points" where the measured value was **0.07963 ≈ 8.0pp** | My hand arithmetic had used the wrong `min` for token 0. A hand-computed number sitting next to a measured one reads as verified. Fixed by printing `rep['tv_broken']` directly rather than any hard-coded figure |
| 2 | Exp 3's denominator spread was described as "1.00 to 2.94x"; the actual table ranges **1.000 to 3.944** (Medusa, 32 candidates) | The wrong figure understated the cost spread — the exact lever the experiment exists to demonstrate |
| 3 | Exp 6's tree model reached **4.984/5.000** tokens per step, making a tree look like free depth at high α | Not a bug in the formula but an undisclosed modelling assumption. Added an *"HONEST LIMITATION OF THIS MODEL"* paragraph: `1 − (1−α)^m` assumes independent candidate draws, whereas a real tree's candidates are correlated and vocabulary-limited, so the true curve is flatter and approaches the ceiling more slowly |

Four findings came out of the T09 build and are reported in the HLD as findings rather than
smoothed away:

- **The exactness claim is checkable, and its failure mode is directional.** Implementing the
  "obvious" variant — resample `p_target` on rejection — shifts the output distribution by **0.0796
  total variation (83× the correct implementation's 0.00096)** while the text stays fluent. Token 0
  moves 0.3000 → 0.2389, token 1 moves 0.2500 → 0.2823. No quality eval reliably catches this; only
  a distribution test against the target does.
- **Speculation's regime boundary is a hard switch at ~295 concurrent sequences** for 70B/fp16 on an
  H100-class part (`batch* = (B·W) / (2N − B·kv)`). That number is *below* what a throughput-oriented
  deployment runs at and *above* what a latency-oriented one runs at — which explains most of the
  disagreement in the literature about whether speculative decoding "works". Both sides are
  measuring correctly, on opposite sides of one line.
- **Trees pay most where the draft is mediocre**, the opposite of the intuitive reading. At α₁ = 0.95
  the m=4 tree gains **1.30×**; at α₁ = 0.45 it gains **2.27×**. A linear draft at high acceptance is
  already keeping nearly every branch, so extra candidates rescue nothing. This inverts the usual
  deployment logic.
- **The constant-α simplification over-predicts by 1.37×** at γ = 8 (4.329 vs 3.167 tokens/step),
  which is exactly why a tuned γ underperforms its own forecast and why a capacity plan built on the
  simplification over-provisions by a third.

Two silent-regression patterns are now drawn explicitly rather than described: the **p50-safe /
p99-regressing deployment** (32 / 400 batches against `batch* = 295` → 2.81× at p50 and **0.59×** at
p99, with the window mean *improving*), and the **ungated config** (`batch_gate: null` is treated as
a defect, not a default, because its only symptom is an absent benefit).

### T10 defects found by its own `run.py` and fixed before the docs were written

Eleven defects, and the most instructive one was **a prose claim that the code's own output
contradicted** — the same class as T09's, found by reading the table rather than trusting the
sentence above it.

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | `weight_group` defaulted to `128` for bf16/fp8/int8, charging a group scale to precisions that have none (bf16 printed **16.12 GB** instead of 16.00) | **silent.** A plausible-looking column, wrong by exactly the metadata term the blueprint exists to make visible |
| 2 | The crossover metric was **degenerate by construction** — `kv_per_seq × sequences ≡ usable − weights`, so KV printed as 55.88 GB at *every* context and "past 30k the KV cache is the majority" was unfalsifiable | **silent.** The metric was 100% at all context lengths and said nothing; replaced with the per-sequence ratio, which crosses 1.0 at ~122k tokens |
| 3 | Exp 3's per-tensor SNR is **non-monotone** (14.76 → 8.48 → 16.61 dB) while worst-channel error is monotone (0.23 → 1.00); the prose claimed error "moved by tens of dB" | reframed as **the finding**: SNR is signal/noise energy and outliers inflate both, so the ratio improves while small weights are destroyed |
| 4 | Exp 6's prose claimed the KV gain "grows with context" but the table was a **constant 4.00×** at every context | rewrote the experiment around the correct finding — KV is a *concurrency* lever (4.00×), weights is a *fitting* lever (1.21×), and switched the column from fp8 KV to int4 KV to match |
| 5 | **The cost ladder was non-monotone: 100 → 35.9 → 70.6 → 24.3** — "4-bit costs more than not quantizing" | each configuration was normalised by its **own** weight size, so shrinking `W` inflated the ratio it was meant to shrink. Fixed with `_reference_bytes()` as a fixed bf16 constant |
| 6 | Exp 1's crossover printed "~122070312k tokens" and "131.072 GB per concurrent sequence" — both multiplied by 1000 an extra time | a units error that made a 0.131 GB marginal cost look like 131 GB |
| 7 | `CORPUS_LADDER` key `"four_bit"` never matched the rung name `"four_bit_weights"`, so the residual column silently read 0 / +0.0 | **silent.** The rung with the largest residual was the one reporting a perfect match |
| 8 | Exp 8's fit had a **67.2% worst residual** and the gate demo was self-contradictory (both mean and worst breached at 2%) | reframed the bad fit *as the finding* (implied exponents 0.40 vs 1.07 are inconsistent ⇒ no single power law reconciles the corpus's own anchors), and moved the threshold to 6% so the mean passes and the worst fails |
| 9 | Missing `import math` in `experiments.py` | crashed loudly |
| 10 | Exp 8's exponent printout at 1dp showed "0.1x" twice | hid the 3.00× vs 8.33× spread the finding rests on |
| 11 | `run.py`'s one-paragraph summary was **stale** — "Quantization is three things", "error moves by tens of dB" — because a replacement was applied to `experiments.py` instead of `run.py` | the last thing a reader sees contradicted the four findings above it |

Four findings came out of the T10 build and are reported in the HLD as findings rather than
smoothed away:

- **Aggregate SNR can improve while the model gets worse.** Per-tensor 4-bit SNR *rises* from 8.48
  to 16.61 dB as outlier severity goes 12 → 48, while the worst channel's relative error reaches
  **1.0000** — an output that is pure error. The metric people quote improves; the feature is
  destroyed. Same tail-over-mean pattern as T08's goodput and T09's p99.
- **The two quantization levers answer different questions.** KV quantization is a *concurrency*
  lever worth a **constant 4.00×** at every context length; weight quantization is worth only
  **1.21×** on concurrency on an 80 GB part, but it is the **fitting** lever — 70B in bf16 on 80 GB
  gives **0.0** sequences, and int4 gives **66.9**. No KV setting substitutes.
- **Quality loss is not a smooth function of reconstruction error.** The corpus's own three anchors
  (fp8 <1%, 4-bit 1–2%, 2-bit 10–15%) imply exponents of **0.40** and **1.07** between consecutive
  rungs; no single power law fits, with a 67.2% worst residual. This is the quantitative form of
  "always rerun your evals".
- **The eval gate's value depends on the quantizer, and it must be on the worst slice.** Group-wise
  is flat across a 48× change in tensor severity (2.57% → 2.50%), so the gate is nearly a formality;
  per-tensor moves across the whole budget (3.89% → 6.27%), so the gate does real work. At a 6%
  threshold: mean **5.29% PASSES**, worst **6.88% REJECTS** — same numbers, opposite decisions.

### T17 defects found by its own `run.py` and fixed before the docs were written

Nine defects. **Seven of the nine produced a plausible number rather than an error**, which is why
T17 took longer to reach a trustworthy state than any prior topic — and two of them were *prose
claims contradicted by the table directly beneath them*, the same class as T09's and T10's.

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | **The gold set was degenerate: the better answer was always placed in slot `a`.** Every kappa printed **0.000**; `+ randomize order` appeared to make agreement *worse* (92.0% → 78.8%); "+ all three" was worse than "+ length normalized" alone | **silent and cascading.** With a constant truth, a position bias aimed at slot `a` scores as *accuracy*, and `po == pe` makes kappa identically 0. The root cause of five downstream misreadings. Fixed by randomising the slot, plus `label_balance()`/`is_degenerate()`/`truths()` on `GoldSet` and both reported unconditionally inside `evaluate()` |
| 2 | `attribution()` ranked by `total_ms` **including the root span** | **silent.** The root owns 100% by construction, so "request" always won; and `llm_turn` + its own `decode` child both appeared, double-counting at a rate proportional to nesting depth. Fixed: exclude the root, rank by `self_ms`, add `count_discriminates` |
| 3 | `agentic_request(n_tools=12)` produced **10** tool calls (`n_tools // llm_turns` truncation) | **silent.** The count ranking — the experiment's whole finding — was computed on a span count the caller never asked for. Fixed with `base, extra = divmod(...)` |
| 4 | `verbosity_effect` hardcoded `"truth_length_quality_r": 0.0` | **silent.** The *baseline* is the honesty check that makes the biased r interpretable; a hardcoded 0.0 makes the comparison meaningless. Fixed with a real `_pearson()` |
| 5 | Exp 2's storage column printed GB at 1 dp, so **every row read 0.0** | cosmetic but it hid the frontier the experiment exists to show. Changed to MB |
| 6 | `_hdr(n: int, ...)` could not take `"1b"` | crashed on the added experiment; annotation dropped |
| 7 | A botched edit left a **dangling f-string** in exp 2 — the corpus paragraph became loose module-level text | the paragraph vanished from the output without any error |
| 8 | **The randomised and fixed arms of the bias sweep consumed different positions in the RNG stream**, so they saw different noise | **silent, and it inverted the finding.** Randomisation appeared to *lose* accuracy at high β for a reason unrelated to bias. Fixed by drawing the order decision in both arms, making the two runs a **paired** comparison |
| 9 | Exp 4's prose said "randomising order removes the largest single effect" and exp 5's said the curves "diverge in slope … recovers all of it" — **both contradicted by the tables directly above them** (length normalization is +5.0 vs randomisation's +1.2; the gap *peaks* mid-sweep rather than at the end) | the exact defect class recorded for T09 and T10. Both rewritten to match the measurement, with the correction ranking now **computed and printed** rather than asserted |

Nine findings came out of the T17 build and are reported in the HLD as findings rather than smoothed
away:

- **The corpus's prescribed judge fix is the weak one.** Randomising the answer order buys **+0.0 to
  +1.7 points** across the whole β sweep, because it decorrelates the bias from the label without
  removing the bias — the bonus is still applied, at random, on every call, and on a close pair a
  random bonus the size of the margin is a coin flip. **It trades a systematic error for variance.**
- **Running both orders is the strong fix, and the tie rate is the deliverable.** Both-orders cancels
  the bonus *exactly* instead of in expectation. The tie rate rises monotonically with β (15.8% →
  79.2%) because the pairs the bias flips are exactly the pairs whose verdict changes with the order —
  so **the abstention set is a calibrated measurement of the judge's unreliability**, and it is the
  only one of the three strategies that reports one at all. Decided accuracy likewise rises with bias
  (84.6% → 98.8%) while raw accuracy with ties-as-errors *falls* — three numbers, none meaningful alone.
- **The three judge corrections are not additive.** Best single (length normalization) 80.5%, all
  three 80.5% — so the other two buy **+0.0** once the dominant bias is fixed. Two of the three biases
  pulled verdicts the same way on this gold set. A team that ships the bundle pays for a second vendor
  to discover it bought almost nothing.
- **A metric cannot attribute, and the agentic case proves it.** Ranking spans by **count** and by
  **self time** disagree on the agentic request: `tool_call` wins on count (12 spans) and loses by
  more than an order of magnitude on wall clock (96 ms vs 1450 ms). A metrics-only dashboard cannot
  form the question, because a metric has no name attached.
- **Head sampling's blind spot is rarity, not cost.** At a 0.01% incident rate with 5% sampling,
  P(no trace in an hour of traffic) = **0.9512** and one trace needs **55.5 hours** — versus 2.77
  hours at 100%. *"Always keep 100% of the errors"* is a correctness requirement for debuggability,
  not a storage optimisation. And `errors_only` is a trap: same error coverage as tail sampling at
  1/6 the bytes, with **0% slow coverage**.
- **Redaction is orthogonal to sampling, at every rate.** `redaction_is_orthogonal()` returns
  `pii_risk_reduced_by_sampling: False` for all five policies including `uniform_1pct` — a 1% sample
  of unredacted prompts is still an incident, just a smaller one.
- **Eval-set size is decided by the tail, not by power.** Three constraints improve at three rates:
  MDE, false-improvement rate, and tail representation. At n=20 a set contains an example of a 5%
  failure class only **64.2%** of the time, and **a mean cannot report a problem in a class the set
  does not contain**. n=200 satisfies all three — the corpus's "curated 200" is not a round number.
- **A mean gate ships the highest-mean bad release.** Over four releases the gates disagree on **3 of
  4**, always mean-passes-tail-fails. `v3-aggressive` has the **highest mean of the four (4.39)** and a
  safety slice that fell **4.28 → 4.20**. Same tail-over-mean signature as T08's goodput, T09's p99 and
  T10's SNR — a fourth independent topic, one pattern.
- **The feedback loop compounds, and it closes the head before the tail.** The open loop stores
  **1,152 GB over 24 releases** and prevents **zero** defects; the closed loop differs in one edge and
  its escape rate falls to zero (raw ratio 22.6×, and the last-release ratio is a *divide by zero*).
  But its residual is severity-skewed **upward** (2.66 vs 2.39 per escape) because rare classes are
  noticed last — so the ramp is not the steady state, and the fix is to seed the eval set from incident
  review rather than from sampling alone.

### T18 defects found by its own `run.py` and fixed before the docs were written

One defect is recorded in `LLD.md` §3.5 because the fix is a contract rather than an edit; T18's other
in-session bugs were caught by the run itself and did not reach prose.

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | **`cadence_for_trough` iterated the cadences ascending**, so it returned `continuous` for *every* target | **silent and answer-shaped.** The function answers "what is the **largest** gap that still holds the trough above target", so it must try the infeasible end first. Ascending returns a *correct* cadence that is a useless answer, for every input. Fixed to descending; solved outputs now 0.90 → continuous (93.4%), 0.70 → **quarterly** (73.6%), 0.60 → **semi-annual** (62.5%) |

Six findings came out of the T18 build and are reported in the HLD as findings rather than smoothed
away — three of them are the **same silent class** as T08's goodput, T09's p99, T10's SNR and T17's
mean gate, and one of them **breaks the pattern**, which is recorded as a disagreement rather than
quietly dropped:

- **Four rails are not `1-(1-r)^4`.** A rail's catch is conditional on technique *and* path, so the
  scalar arithmetic overstates protection by **17.7 points** on the full stack and **83.9 points** on
  the two rails teams actually build. **One rail with full coverage beats all four with partial
  coverage** — the corpus's own defence-in-depth picture does not survive being multiplied out.
- **The detection approach is within 1.1 points of its own ceiling; the capability gate has 22×.**
  This is a build-order instruction: with the detectors **off**, the gate still beats detectors at
  maximum with no gate, by **3.7×**.
- **The false-refusal optimum is a ratio someone must name** — 97.1% refusal for a regulated payer,
  3.6 SD the other way for a marketing bot, and the two disagree about which endpoint is safe. One
  threshold is an average over a non-uniform population; the fix is a **cascade**, not a better
  threshold.
- **Fail-open on a lost trust tag doubles the gate's failure rate (0.045 → 0.093) while every
  detection metric stays flat.** This is why T18 ships a boot check rather than an alert, and why
  §13.1 finds **ten of fifteen failure modes silent**.
- **A red-team schedule is the control, and its failure mode is the interval** — 45.5% trough catch at
  annual cadence against 93.4% continuous. The published mean falls 53.1 points over two years, so it
  is not hiding anything; it is a *level where a rate of change was needed*.
- **The novel-class reading is silent by construction.** It moves **0.0 points** across two years of
  accelerating decay, because it starts at its own floor (7.9%) and stays there — an instrument
  pointed at a population that is entirely severity-5 cannot cross any threshold. **T18 explicitly
  records that this disagrees with the tail-over-mean pattern** the other topics found; the divergence
  is stated in HLD §10.3 rather than reconciled.
- **Approval is a resource.** Its failure mode is *approval*, its capacity is measured in headcount,
  and the gate's strength swings **20×** on queue depth alone.

### T19 defects found by its own `run.py` and fixed before the docs were written

Eight defects. **Two were arithmetic that would have propagated into the HLD**, two were *prose that
the code's own output contradicted* (the T09/T10/T17 class), and one was a Python packaging bug that a
purely-read review could not have found.

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | **`AttributeError: 'function' object has no attribute 'layer_table'`** — `sim/levers.py` exported a function named `stack`, which shadowed the `stack` **module** in the package namespace once `sim/__init__.py` imported it | crashed loudly, but only at run time and only in the package-import path. Renamed to `apply_order`, with the reason written into the docstring so it is not renamed back |
| 2 | **The attribution growth scenario shrank the total bill** — 1,000,000 → 809,859 — because spend was normalised by the **mean** growth (2.84×), which shrank every below-average team | **silent, and it inverted the finding.** The scenario exists to show that coverage falls *because the business is growing*; a shrinking bill contradicts the premise the section is built on. Fixed by normalising to the **minimum** growth (1.2×) so nothing shrinks: total 1,000,000 → **1,916,667 (1.92×)**, untagged 95,400 → 336,800, untagged share **9.54% → 17.57%**. Coverage is invariant to the reference — 90.46% → **82.43%** — which is the check that the fix did not move the finding |
| 3 | **The HLD's worked capacity/cost model applied the sovereignty uplift to the wrong bill.** Steps 4–5 said "blended 10.0% → 100,000 added" and "sovereignty cost is 25% of the saving" | **silent.** The uplift applies to the **post**-optimisation bill, because optimisation and attestation hit the *same* tokens. Verified: `0.4070 × 0.40 × 0.25 = 0.0407` → **4.07 points**, final **44.8%**, sovereignty cost **6.9% of the saving** — not 25% |
| 4 | **The HLD's sensitivity row was wrong** — "cached share 90% → 70%: banked 23.2% → 17.2%" | verified by running it: the true figure is **15.7%**, a −7.5 point move. The wrong value understated the sensitivity of the largest lever in the ladder |
| 5 | The claim table printed the header word **"claim" twice** and truncated every claim at 34 characters | cosmetic, but the table *is* the finding (claim order vs banked order); truncation cut the distinguishing words off the two levers the finding rests on. Header corrected and width to 44 |
| 6 | **Finding 2 said "2.4 points at its stated reach" while the printed line directly above it read 2.02% of the ceiling** | the exact defect class recorded for T09, T10 and T17 — the prose and the table beside it disagreed. Both now read "**2.0 points of the ceiling**", and `run.py`'s decision 2 matches |
| 7 | Awkward print wrapping: a stray `[R]` starting a line in the write-premium block, and `bills\n2300` in the reasoning-token block | the `[R]` marker detached from its claim and read as a new provenance marker for the following line |
| 8 | `shadow_gap` printed **"and False is not a passing answer."** | a boolean interpolated into a sentence. Rewritten to state what the number means: a 12.0% residual is not a rounding error, it is traffic that left the gateway |

Six findings came out of the T19 build and are reported in the HLD as findings rather than smoothed
away. The first three are new instances of the tail-over-mean pattern recorded for T08, T09, T10, T17
and T18 — which makes T19 the **sixth independent topic** to find it:

- **A discount is a price, not a saving.** A lever's worth is `reachable_share × discount ×
  share_of_bill`, and the third factor is what converts the first two into money. Measured on this
  bill: **eight of ten levers change rank** between claim order and banked order. `prompt_caching`
  is #3 by claim (10×) and **#1 by banked (23.2%)**; `distillation` is **#1 by claim (40×) and #8 by
  banked (2.4%)**. The corpus's ladder is not wrong — it is a list of *prices*, and the plan that
  follows from it does not start at the top of the list.
- **The cache has an asymptote, and it is 26.7%.** Banked saving by read discount: 2× → 11.2%,
  10× → 23.2%, 50× → 26.0%, 100× → 26.4%, **infinite → 26.7%**. A free cache read cannot do better,
  because the uncached 10% of calls still pays full price and nothing outside the prefix is touched.
  Any business case projecting more from caching alone is spending a discount that cannot be spent.
- **The headline is 90%; the ladder's ceiling is 60.61%.** The corpus's *"roughly a tenth of the
  cost"* **[T]** is a **10.00×**, and the ceiling over this bill with disjoint shares is **2.54×** —
  a gap of **29.39 points**. With distillation removed the ceiling is **58.59%**, so the ladder's most
  aggressive claim is worth **2.02 points** of it. And the gap is not a bug: a target above the
  ceiling is how a programme starts inventing savings.
- **There is one pricing slot, and the ordering rule is worth 4.15 points.** `reserved_capacity`
  (15.75 pts) and `batch_lane` (10.00 pts) reprice the same tokens. Taken by value: **59.30%**.
  Taken by ease, batch first: **55.17%**. The difference is not effort or risk — it is **which lever
  got there first**, and without conflict enforcement the stack reports **66.98% against a 60.61%
  ceiling**, i.e. a saving larger than its own bound.
- **The unit decides the sign, and the deliverable shrank faster than the price.** The same two rows
  (30¢ → 16¢, 630 → 91 lines, 8.2 → 3.6 files) give **−46.7% per artifact**, **+21.5% per file** and
  **+269.2% per line delivered**. Lines fell to 14.4% of their February value while cost fell only to
  53.3% — so the work costs **3.69× as much**, and the metric that would have shown it was never
  computed. Same shape as the mean-derived cap: **mean task 0.42¢ vs p99 12.24¢**, so a $5 ceiling
  set from the tail kills nothing legitimate and one set from the mean terminates real work.
- **Sovereignty is a vector with prerequisites and a floor, and residency is not a score.**
  `compliance_only` has a mean of 0.438 and a **floor of 0.100** (gap 0.338); `api_with_dpa` 0.288 /
  0.050 (gap 0.238). `vendor_tee_lease` claims **0.85 trust** and can evidence **0.50** — the
  attestation is *true and about somebody else's machine*, which is why the prerequisite rule
  (`trust ≤ control`) exists and why it is **data**, not code. And a score-based residency router at
  gap 1.0 / noise 1.0 leaks **15.87%** — **1,586.6 per 10,000 requests** against a filter's zero,
  because **99.84% correct residency is non-compliant**. Three of four continuity examples score a
  mean of 0.75 and a **floor of 0.0**.

## Verification

| Check | State |
|---|---|
| 1. Structural — all 19 IDs present in each family, files non-trivial | ✅ **complete** — cheat sheets 20 · case studies 19/19 · interview banks 20 · blueprints **19/19 topics × 5 artefacts** |
| 2. Provenance — header line + `## Sources` present, paths resolve | ✅ **complete** — every artefact in all four families, verified by script |
| 3. Code runs — every `run.py` executes cleanly | ✅ **19/19 exit 0**, none empty, **all output ASCII-clean** — re-run 2026-09-27 |
| 4. Links — relative links and `Tnn` refs resolve | ✅ **complete** — `verify.py` reports *"No problems found in checks 1, 2 and 4"*; no pending targets remain |
| 5. No fabrication — spot-check quoted numbers against sources | 🔄 five passes done, **25 defects found and fixed**; blueprint prose is covered only where a `run.py` recomputes it |

`verify.py` now reports **no pending targets** — the forward-reference list it carried through Waves 1
and 2 is empty, which is the structural signal that all 76 primary artefacts exist.

**Check 3 detail (re-run 2026-09-27, after T18 and T19 landed).** Every blueprint executes offline
with no GPU and no network. Exit 0 for all 19; output ranges from 129 lines (T11) to 620 (T18). The
ASCII sweep caught **one** non-ASCII character — a `§` in a T12 `print()` — which is mojibake on a
cp1252 console; replaced with "section 9". It is the same class as the T17 `_hdr` annotation bug: a
cosmetic defect that no human reading the output would flag, and that only a machine check finds.

| Topic | Exit | Lines | Topic | Exit | Lines | Topic | Exit | Lines |
|---|---|---|---|---|---|---|---|---|
| T01 | 0 | 135 | T08 | 0 | 303 | T15 | 0 | 218 |
| T02 | 0 | 133 | T09 | 0 | 269 | T16 | 0 | 254 |
| T03 | 0 | 183 | T10 | 0 | 370 | T17 | 0 | 486 |
| T04 | 0 | 208 | T11 | 0 | 129 | T18 | 0 | 620 |
| T05 | 0 | 157 | T12 | 0 | 166 | T19 | 0 | 532 |
| T06 | 0 | 180 | T13 | 0 | 157 | | | |
| T07 | 0 | 260 | T14 | 0 | 198 | | | |

Checks 1, 2 and 4 are automated in [`verify.py`](verify.py) — `python verify.py` (add `-v` for
per-file detail). It knows which paths the build is *supposed* to produce, so forward references to
not-yet-written topics report as pending rather than broken.

### Check 5 — accuracy audit (five passes, 25 defects found and fixed)

Nine headline claims were pulled from the transcripts and compared word-for-word against what the
artefacts assert. **All nine verified**, with two refinements applied:

| Claim | Verdict |
|---|---|
| Prefill is ~98% of agentic tokens | ✅ verbatim — *"prefill is occupying like 98% of the tokens"* |
| Cost ladder 100 → 42 → 26 → 11 | ✅ verbatim — batching→42, quantize→26, cache→11 |
| Cache reads ~10× cheaper than fresh | ✅ verbatim |
| "always rerun your evals after quantizing" | ✅ verbatim |
| `log n − (n−1)/n` KL bound (best-of-N) | ✅ verbatim, incl. the Beirami et al. attribution |
| `n = 32` for best-of-N | ✅ verified **but an illustrative example, not a rule** — the lecturer says size `n` so it fits one batch. Master index softened accordingly |
| WideEP: "tensor parallelism is one" | ✅ verbatim — EP 72 GPUs, DP attention 72 GPUs, **TP = 1** |
| MoE layer: 2 all-to-all + 6 kernels → 3 kernels | ✅ verbatim — dispatch A2A + combine A2A + fused MoE |
| EP sizing: 256/8 = 32 experts/GPU; 256/32 = 8 experts/GPU | ✅ verbatim — incl. the "more VRAM for KV cache" consequence |
| Pooled memory < RDMA << TCP/IP | ✅ verbatim — *"this pooled memory based data sharing is faster than RDMA"* |
| 20k max concurrency on 1P1D; 28k ⇒ KV recomputation | ✅ verbatim |
| KV 80% full / >8 active requests as saturation gates | ✅ verbatim — both figures in one sentence |
| MTP ≈ 2× throughput | ✅ verbatim — *"gained about 2x improvement in throughput"* |
| 5× TTFT from KV offload on session return | ✅ verbatim |

**Two corrections made:**

1. **The 16-GPU parallelism configuration was over-specified.** The source ASR renders the speaker's
   config as *"two-way tensor parallel … two-way pipeline parallel … tensor parallel plus sequence
   parallelism … across 16 GPUs"*. The artefacts had hardened this into a precise `2-way TP × PP × SP`
   factorisation. Corrected in the `T11` cheat sheet, the `T11` case study and `TOPICS.md` to state
   the **four mechanisms (TP+PP+SP+EP) and the 16-GPU figure** as reliable and explicitly flag the
   exact factorisation as ASR-garbled. The speaker's load-bearing claim is the *contrast with
   single-host 8-way TP*, not the product.

2. **"TP stays 1" was over-generalised.** It is stated for the WideEP attention case, but the same
   team's own target config is **TP8 + 2P2D + EP8 (intra-node "shallow EP") + DP16** — TP *is* used,
   kept inside the node. The durable rule recorded instead: *TP stays inside the node; EP and DP
   carry the wide dimension.* Corrected in the `T11` cheat sheet.

### Check 5 — second pass (fairness, sovereignty, constrained decoding)

Three more claims audited; **two substantive errors found and fixed** — both in the orchestrator's
cheat sheets, not the agents' case studies.

| Claim | Verdict |
|---|---|
| JSON-schema constrained decoding uses **pushdown automata, not FSAs** | ✅ verbatim — *"anything that supports JSON schemas is actually writing push down automata to enforce its constraints, not FSAs"* |
| Search error vs model error | ✅ verbatim — CMU lecture 1, not lecture 5 where I first looked |
| Least-attained service: latencies down up to 50% / 2–3× | ✅ verbatim — *"by up to 50% … I mean by up to two 2x or sometimes 3x"* |
| Turn priority frees KV by finishing near-done agents | ✅ verbatim — the speaker's own example is *turn ~100* |
| Sovereignty = three axes (data/model/infrastructure) | ❌ **wrong** — see correction 3 |

**Correction 3 — the sovereignty framing was invented.** The `T19` cheat sheet asserted a tidy
`data · model · infrastructure` triple marked `[T]`. The talk says no such thing. It explicitly
**rejects the binary framing** (*"Think about sovereignty as something like binary … I think that's
wrong"*) and enumerates its own dimensions: **control, choice, trust, economics, continuity** — where
control asks *where inference runs*, trust asks *what code and artifacts execute*, economics asks
*who controls the token cost*, continuity asks *can you operate without a single vendor*. Crucially
the speaker's load-bearing point is that **the inference layer itself is a sovereignty component
people forget** (*"if inference leaves your control, can you really call that as sovereign AI?"*).
Rewritten from the transcript. *(The speaker lists "choice" as its own item with "and then the
choice…", so the count reads as four or five depending on how strictly you split it — the `T19` case
study read four; both readings are now noted rather than one being asserted.)*

**Correction 4 — the fairness starvation direction was backwards.** The `T08` and `T16` cheat sheets
claimed least-attained service *"prevents a long agent program from being starved by a stream of
short requests."* The transcript describes the **opposite**: *"there is a small session and a **large**
session that comes in and then takes all the dispatch cycles. So the **shorter sessions are
starving** now."* The mechanism stops one big session monopolising the scheduler, and the second-order
effect is the large win — *"short sessions finish much faster and leaving the room for larger sessions
to also finish much faster."* Rewritten in both cheat sheets. The `T08` case study took a separate
per-tenant-fairness angle and did not repeat the error.

**Also captured while auditing** (`T19`): the Token Raj study figures — unit cost for a 10k-token task
**30¢ → 16¢** (Feb → Apr), code lines drafted **630 → 91**, files touched **8.2 → 3.6**, with the
speaker's caution that **billing fell less than the workload did because thinking tokens rose**. The
lab name and model named in that passage are ASR-garbled and are marked approximate in the sheet.

### Check 5 — third pass (quote-fidelity sweep across all 20 cheat sheets)

An automated scanner pulled every quoted phrase from the 20 cheat sheets and searched the corpus for
it with `[m:ss]` timestamps stripped. **Eight further defects found and fixed** — all in the
orchestrator's own cheat-sheet layer, none in the agents' case studies.

| Defect | Fix |
|---|---|
| `"Whatever we can backprop through wins"` attributed to Tworek — **fabricated**; "backprop" appears nowhere in the transcript | Replaced with a real Tworek quote on gradient steps, and two untraceable entries in master-index §9 removed rather than reworded |
| `"Putting a layer of indirection between things"` attributed to DeSantis — **fabricated** | Section 9 entry demoted to an explicitly-labelled paraphrase |
| `"eviction rampant"` — **fabricated**; the word is not in any transcript | Replaced with the supported claim (router consumes per-request create *and* evict events; an offload tier and a retention API exist), with the inference labelled `[D]` |
| `"often halves total spend"` — **untraceable** | Replaced with the verbatim *"Prompt caching and difficulty-based routing usually move the bill more than switching models does."* |
| `"that is a larger disk, not memory"` — paraphrase in quote marks | Replaced with the verbatim *"it is not memory. It is just a larger disk."* |
| `"always rerun evals after quantizing"` — dropped word | Corrected to the verbatim *"always rerun **your** evals after quantizing"* |
| `"There is no universal winner."` — punctuation drift | Corrected to the verbatim *"there's no universal winner"* |
| Nested bold artifact in `T17` (`**"always rerun **your** evals…"**`) | Repaired |

Three phrases that were the author's own gloss but sat in quotation marks were **de-quoted to
italics** so they cannot be mistaken for corpus quotes: *the model's distribution minus its tail*
(`T01`), *needs two 80 GB GPUs* (`T10`), and the `T19` sovereignty triple.

**Resolved — Agent A's objection to the best-of-N KL bound.** Agent A reported that its table
*"shows the bound **increases** with n while the quantity it bounds **decreases**"* and suspected an
internal inconsistency. Checked against CMU lecture 12 directly: **there is no inconsistency, but my
cheat sheet had the two distributions wrong.** The lecture states the bound is on *"the KL divergence
between your best-of-N outputs distribution and your **target** distribution"* `[T]` — the preference
distribution, **not the base policy** as `T05` had it. The bound and the quantity it bounds both rise
with `n` (more samples buy more aggressive alignment), so the monotonicity is consistent. Corrected in
`T05` and the master index; the lecture's own caveat that the figure is *"often quoted as an exact
value"* but is **not tight** in edge cases (citing Beirami et al. and a tighter bound at eq. 25) was
already present and is retained.

### Check 5 — fourth pass (fabricated-quote sweep over all 19 case studies)

The cheat-sheet sweep covered 20 files; this pass covered the ~149,000 words of case studies the two
agents wrote, which is the highest-value unaudited surface. Method: extract every double-quoted
string adjacent to a `[T]` marker (i.e. every quote *claiming transcript provenance*), normalize both
it and the corpus (strip `[m:ss]` timestamps, markdown emphasis, bracketed editorial insertions and
ellipses), then require each ellipsis-separated fragment to appear in the corpus.

| Stage | Quotes flagged |
|---|---|
| Raw scan (every quoted string ≥ 5 words) | 279 |
| Restricted to quotes claiming `[T]` provenance | 48 |
| After normalizing bold/ellipsis/brackets | 11 |
| After manual verification of each survivor | **5 — all confirmed verbatim, all matcher false positives** |

**Seven real defects found and fixed**, all of them quotes that over-claimed verbatim status:

| File | Was | Now |
|---|---|---|
| `T03` | `"don't emit a code from the retirement-products group"` — the author's *invented* illustrative constraint, in quote marks beside a `[T]` | De-quoted to italics and explicitly labelled *"an illustrative constraint — the author's example, not a transcript quote"* |
| `T04` | `"use the smallest model that can reason at all, with a large token budget"` — the author's summarising inversion | De-quoted to italics, labelled "the author's formulation of the inversion" |
| `T05` | `"a well-defined probability distribution"` — gloss | De-quoted to bold |
| `T05` | `"play with smaller generator vs larger reward model and vice versa"` | De-quoted to plain prose |
| `T05` | `"still a version of reward modeling that is used in a lot of RLHF"` | De-quoted to plain prose |
| `T09` | `"enabled more interactivity, which gained about 2x improvement in throughput"` — silently de-disfluenced | Ellipsis restored and the ASR disfluency noted in-text |
| `T14` | `"Autoscaling models is touchy because you need available GPU capacity"` — **compressed paraphrase in quote marks** | Replaced with the verbatim: *"autoscaling when it comes to models is a little touchy subject … for you to autoscale you need to have available GPU capacity"* |

The last one is the instructive case: the quote was *substantively* right and would have survived any
fact-check of its claim, but it was not what the speaker said. Compressed paraphrases inside quotation
marks are the failure mode that survives every other check in this list, because the claim is true.

### Check 5 — fifth pass (defects raised by the Wave 2 agents)

Both agents audited their own banks and flagged items outside their scope. **All five were real and
are fixed.** Four were internal-consistency defects rather than provenance ones — the class the quote
scanners cannot catch.

| # | File | Defect | Fix |
|---|---|---|---|
| 1 | `01-case-studies/T05` | Repeated the superseded `P_base` orientation in the decision table and the interview walkthrough — **contradicting the bank that now teaches `P_target`**. Also asserted the bound and the divergence move in opposite directions | Corrected to `KL(P_bon \|\| P_target)`; the inverted-monotonicity claim removed; the real limitation (loose, not a dial) stated instead |
| 2 | `00-cheat-sheets/T02` | `"α ∈ [0.6, 1.0]"` presented unmarked as fact; **the corpus states no range for α** | Marked `[D]`, with the absence from the corpus stated explicitly |
| 3 | `01-case-studies/T09` | The §8 cost table **did not reproduce from its own stated model** at several cells (`K=2, α=0.5` gave 27.0 ms where the model gives 34.3; `K=8, α=0.9` gave 17.8 vs 14.7) | All 20 cells recomputed from `step / E[α,K]` with `step = 5K + 50`, plus a note that it was recomputed. The `K=4` column was already consistent. The corrected table also changes a reading: at high acceptance **longer** drafts do pay (`α=0.9, K=8` beats `K=4`) |
| 4 | `01-case-studies/T08` | Two sensitivity rows **held per-chunk time constant at 200 ms**, implying a 16k chunk costs the same as a 4k chunk — not physical, and the resulting "~1.6 s" was unattainable | Replaced with a two-term model stated in full (`0.05 ms/token` compute, fixed `N = 128k/chunk` interleaved rounds, per-round work `d` left as an explicit unmeasured coefficient). The "~25 s" figure is retained as `6.4 s + 128d` |
| 5 | `00-cheat-sheets/T09` | `min(1, p_target/p_draft)` attributed to `[T]` CMU lecture 2, **which contains no acceptance rule** | Re-attributed to `[D]` standard construction, with the wrong attribution named |

Item 3 is the instructive one: every number in that table was plausible, and only recomputing the
column exposed that the table had not been generated from the formula printed directly above it.

### Corrections made during verification

Filenames corrected after script-resolving every `refs/` path against the real tree:

| File | Referenced | Actual |
|---|---|---|
| `T14` | `Semantic_Router_for_LLM_Inference.txt` | `Inside_vLLM_Semantic_Router.txt` |
| `T13` | `Tim_Hockin_-_Running_Agents_on_Kubernetes.txt` | `Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt` |
| `T12`, `T13` | `04-inference-optimization/06-prefill-decode-disaggregation.md`, `07-inference-engines.md` | `06-serving-infrastructure.md` |
| `T13`,`T14`,`T15`,`T19` | `11-infrastructure-and-mlops/README.md` | `01-llm-infrastructure.md`, `03-ai-gateways-and-model-routing.md`, `04-finops-and-token-economics.md` |
| `T17` | CMU lecture 2 under `_transcripts/` | under `_transcripts_2/` |
| `T13` | CMU lecture 1, 3 under `_transcripts/` | under `_transcripts_2/`, longer names |
| `T05` | `..._Best_of_N.txt`, `..._Multi_Agent_...txt` | `..._Best-of-N.txt`, `..._Multi-Agent_...txt` |

**CMU corpus trap:** lectures 1–4 live under `..._Fall_2025_transcripts_2/`, lectures 5–12 under
`..._Fall_2025_transcripts/`. Both agents were sent the corrected mapping so Waves 2 and 3 do not
repeat it.

## Open risks and residual debt

All three original risks are now **closed or bounded**, and what remains is stated rather than
resolved:

- ~~**Cross-agent numbering.**~~ **Closed.** The shared 19-topic ID space held: interview IDs are
  `Tnn-Qk` and no collision was found by `verify.py`'s structural check.
- ~~**Blueprint `run.py` on Windows.**~~ **Closed.** All 19 execute offline, no GPU, no network,
  exit 0 — verified by running them (check 3), not by inspection, and re-run after T18/T19 landed.
- ~~**Fabrication risk on `[D]`-marked numbers.**~~ **Bounded, not closed.** Five audit passes found
  and fixed **25 defects**. The residual surface is **`[D]` arithmetic in blueprint prose that no
  `run.py` recomputes** — the T14 `"40% of P50 routing latency"` figure, the guide's KV arithmetic
  8× discrepancy and the FP8 `<1%` vs `<0.1%` conflict were identified in earlier passes and **have
  not been reconciled**. They are not known to be wrong; they are known to be *unverified*.

**Three things this ledger wants a reader to know rather than discover:**

1. **The subagent cap was breached, by the agents, in Waves 1–2.** The coordinator's own count is
   two; the build's count is higher. It is recorded at the top of this file rather than in a footnote.
2. **T18 records a genuine disagreement with the rest of the knowledge base.** The tail-over-mean
   pattern found in T08, T09, T10, T17, T18 and T19 is *not* universal — T18's decay experiment does
   not reproduce it, and says so in HLD §10.3 instead of being smoothed into agreement.
3. **A pattern found in six independent topics is still a pattern, not a law.** `[D]` findings are
   this build's own arithmetic over the corpus's inputs; they are recomputable from the stated
   inputs and are labelled as derived at every use site. A reader disagreeing with one should argue
   with the inputs, which are printed beside it.

