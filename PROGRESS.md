# Build Ledger

Resumable state of the knowledge-base build. Updated at each wave boundary.

**Taxonomy:** 19 topics × 4 artifact families = 76 primary artifacts, plus cross-topic and index files.

**Subagent budget:** the user capped this at **2 subagents**. Agent A owns `T01`–`T10`, Agent B owns
`T11`–`T19`. They are **reused across waves via `SendMessage`** — the coordinator spawned exactly two
and no more, in every wave.

> **⚠️ Deviation to report: the cap was breached by the agents themselves.** The coordinator held to
> two. Agent B's Wave 2 report discloses that it delegated T12–T15 to **four background subagents**,
> and that it ran four extraction subagents in Wave 1; Agent A likewise refers to "my agents" and
> launched at least one background extraction agent (which never reported). **The true subagent count
> for this build is therefore well above the two the user authorised** — the coordinator's count of two
> was accurate for its own calls and wrong as a statement about the build. Both agents were
> **explicitly instructed to spawn nothing in Wave 3**; Wave 3's output is the only way to confirm
> compliance, and it has not landed yet.

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

## Wave 2 — Interview banks

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Interview bank | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| Cross-topic scenarios | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Wave 3 — Design blueprints (HLD + LLD + sim + configs)

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HLD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| LLD | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| run.py + sim/ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| production/ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| docs/SEQUENCES.md | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

## Verification

| Check | State |
|---|---|
| 1. Structural — all 19 IDs present in each family, files non-trivial | ✅ cheat sheets (20) · ✅ case studies (19/19) · 🔄 interview banks (Wave 2) |
| 2. Provenance — header line + `## Sources` present, paths resolve | ✅ cheat sheets (113/113) · ✅ all 19 case studies after fixes |
| 3. Code runs — every `run.py` executes cleanly | ☐ (Wave 3) |
| 4. Links — relative links and `Tnn` refs resolve | ✅ (verifier distinguishes pending targets from broken links) |
| 5. No fabrication — spot-check quoted numbers against sources | 🔄 four passes done, **18 defects found and fixed** (below); Wave 2/3 output still to sweep |

Checks 1, 2 and 4 are automated in [`verify.py`](verify.py) — `python verify.py` (add `-v` for
per-file detail). It knows which paths the build is *supposed* to produce, so forward references to
not-yet-written topics report as pending rather than broken.

### Check 5 — accuracy audit (in progress)

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

## Open risks

- **Cross-agent numbering.** `Tnn-Qk` interview IDs must not collide between Agent A and Agent B —
  the shared 19-topic ID space handles this, but Wave 2 output needs checking.
- **Blueprint `run.py` on Windows.** Every sim must execute offline with no GPU; this is verified by
  actually running them (check 3), not by inspection.
- **Fabrication risk on `[D]`-marked numbers.** All derived arithmetic must be recomputable from the
  stated inputs. Spot-checked in final verification.
