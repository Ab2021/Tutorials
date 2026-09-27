# Inference & Infrastructure Knowledge Base

A working reference for **LLM inference and the infrastructure that operates it** — built from 44
conference and course transcripts, cross-checked against three supporting repositories.

Four interlocking artifact families, all keyed to one topic taxonomy:

| Family | Folder | What it is |
|---|---|---|
| 📊 **Cheat sheets** | [`00-cheat-sheets/`](00-cheat-sheets/) | One dense page per topic — numbers, decision matrix, formulas, config, failure signatures |
| 📖 **Case studies** | [`01-case-studies/`](01-case-studies/) | Worked system designs with decision tables, worked arithmetic, failure modes, runbooks |
| 🎯 **Interview questions** | [`02-interview-questions/`](02-interview-questions/) | Numbered banks with model answers, signals, follow-ups and red flags |
| 🏗️ **Design blueprints** | [`03-design-blueprints/`](03-design-blueprints/) | **HLD + LLD** per topic — component decomposition, interfaces, state machines, data structures |

Start at [`TOPICS.md`](TOPICS.md) for the taxonomy and the topic → source-transcript map.
Build state is in [`PROGRESS.md`](PROGRESS.md).

**Complete: 19 topics × 4 families = 76 primary artefacts, plus the indexes and cores — ~819,000 words.**

| Family | Files | Words | Notes |
|---|---|---|---|
| Cheat sheets | 20 | 27.6k | one page per topic, plus a master index of every number in the corpus |
| Case studies | 19 | 148.8k | decision tables, worked arithmetic, failure modes, runbooks |
| Interview banks | 20 | 266.4k | `Tnn-Qk` numbering, shared across all four families |
| Design blueprints | 19 × 5 | 227.2k design | HLD + LLD + sequence diagrams + `production/`, plus 131.8k words of `sim/` and configs |

Every `run.py` executes **offline, with no GPU and no network**, and all 19 were re-run as the final
check — **19/19 exit 0**. They exist to *verify the designs' arithmetic*, not as the deliverable: each
one recomputes its topic's headline findings from stated inputs, which is how the numbers quoted in
the HLDs are checked rather than asserted. Roughly **25 defects** were found this way across the
build, most of them the silent kind — a plausible number that no crash or dashboard would have
caught. They are listed per topic in [`PROGRESS.md`](PROGRESS.md).

---

## The 19 topics

**Layer A — Decoding Algorithms** *(what token comes next, and how the search over sequences runs)*
| ID | Topic |
|---|---|
| `T01` | [Sampling & Decoding Strategies](01-case-studies/T01-sampling-decoding.md) |
| `T02` | [Beam, A* and Best-First Search](01-case-studies/T02-search-decoding.md) |
| `T03` | [Constrained & Structured Generation](01-case-studies/T03-constrained-generation.md) |
| `T04` | [Chain-of-Thought, Self-Correction, Reasoning Models](01-case-studies/T04-test-time-compute.md) |
| `T05` | [Reward Models, Verifiers, Best-of-N](01-case-studies/T05-verifiers-best-of-n.md) |

**Layer B — Engine Internals** *(where algorithms meet hardware)*
| ID | Topic |
|---|---|
| `T06` | [Prefill/Decode, Roofline, Latency Metrics](01-case-studies/T06-inference-fundamentals.md) |
| `T07` | [KV Cache, Paged Attention, Tiering](01-case-studies/T07-kv-cache.md) |
| `T08` | [Continuous Batching, Scheduling, Fairness](01-case-studies/T08-batching-scheduling.md) |
| `T09` | [Speculative Decoding](01-case-studies/T09-speculative-decoding.md) |
| `T10` | [Quantization](01-case-studies/T10-quantization.md) |

**Layer C — Distributed & Serving Infrastructure**
| ID | Topic |
|---|---|
| `T11` | [Parallelism, MoE, WideEP, ROCm](01-case-studies/T11-parallelism-moe.md) |
| `T12` | [Prefill/Decode Disaggregation & KV Transfer](01-case-studies/T12-disaggregation-kv-transfer.md) |
| `T13` | [Serving Engines & Runtimes](01-case-studies/T13-serving-engines.md) |
| `T14` | [Routing & Gateways](01-case-studies/T14-routing-gateways.md) |
| `T15` | [Autoscaling, Capacity Planning, SLO](01-case-studies/T15-autoscaling-slo.md) |

**Layer D — Operations, Agents & Governance**
| ID | Topic |
|---|---|
| `T16` | [Agentic Inference](01-case-studies/T16-agentic-inference.md) |
| `T17` | [Observability, Tracing & Evaluation](01-case-studies/T17-observability-evals.md) |
| `T18` | [Guardrails, Security & Governance](01-case-studies/T18-guardrails-security.md) |
| `T19` | [FinOps, Token Economics & Sovereignty](01-case-studies/T19-finops-sovereignty.md) |

---

## Provenance legend

The source material is a mix of **video transcripts** and **supporting repositories**, and the
transcripts have real holes. Every file in this knowledge base therefore marks where its claims
come from.

| Marker | Meaning |
|---|---|
| **`[T]`** | Stated in a video transcript. Attributed inline to the speaker and talk. |
| **`[R]`** | Taken from a supporting repository. Path cited. |
| **`[D]`** | Derived or standard practice — **not** in the corpus. Reasoned from first principles. |

Every file also carries a header line:

```
Transcript coverage: primary | partial | none
```

- **primary** — the talks are the main source and go deep
- **partial** — the talks touch it, but the substance comes from supporting repos (marked `[R]`)
- **none** — no transcript covers this; entirely `[R]` and `[D]`

> **Why this matters.** The two corpus halves barely overlap: the CMU lectures are a pure
> *inference-algorithms* course with almost no serving vocabulary, while the infrastructure talks
> contain almost none of the algorithms. And several core serving topics — paged attention,
> continuous batching, NIXL, Ray, Dynamo, goodput — are absent from the talks entirely. Without
> provenance markers you cannot tell a speaker's measured claim from a recalled one. See
> [TOPICS.md § Holes in the transcripts](TOPICS.md#holes-in-the-transcripts).

### On numbers

Benchmark figures are quoted **as reported by the speaker**, attributed to them, and **never
invented**. Where a figure is a vendor or speaker claim rather than an independent measurement,
the file says so. Where arithmetic is mine (capacity models, cost break-evens, worked examples),
it is labelled and the assumptions are shown so you can re-derive it.

### On ASR quality

The transcripts contain phonetically mangled proper nouns — SGLang appears as "Ashlan"/"SLM",
harness as "honey", LiteLLM as "LightLLM", NIXL as "NVIDIA and Excel". A correction table is in
[TOPICS.md](TOPICS.md#asr-garbling-warning). Treat proper nouns in the Banghua Zhu, Jianfeng Gao
and Saurabh Tiwary talks as approximate.

---

## Source corpus

Extracted from `zip_file_references/` into `refs/`.

**Transcripts — 44 files**

| Corpus | Files | Words | Contributes |
|---|---|---|---|
| `CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts{,_2}/` | 12 | ~131k | Layers A and the algorithmic half of B |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/` | 7 | ~26k | Layers B and C — the serving core |
| `Agentic_AI_Infra_transcripts{,_2,_3}/` | 14 | ~30k | Layers C and D — agent-era infrastructure |
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts{,_2}/` | 12 | ~15k | Layer D — operations |

**Supporting repositories**
- `refs/ai-system-design-guide-main/` — 150 md, ~356k words. Chapters `04-inference-optimization`,
  `11-infrastructure-and-mlops`, `00-interview-prep`, `16-case-studies`.
- `refs/llm-inference-engineering-main/` — blog index covering KV cache, paged attention,
  continuous batching, speculative decoding, vLLM/SGLang/TensorRT-LLM/GGUF, MoE, routing.
- `refs/gpu-perf-engineering-resources-main/` — index of GPU kernel, profiling and perf resources.
- `refs/MasterDeck vLLM Inference Meetup.pptx` / `.pdf` — 97 slides, **image-only, no extractable
  text**. The meetup transcripts are the text source for this deck.

---

## How to read this

**Studying a single topic end to end** → cheat sheet → case study → design blueprint HLD then LLD.
**Preparing for interviews** → interview bank, then the case study's *Interview walkthrough*
section, then `02-interview-questions/00-cross-topic-scenarios.md`.
**About to build something** → design blueprint HLD for the architecture, LLD for the interfaces,
`production/` for reference configs, then the case study's runbook and failure modes.
**Debugging** → cheat sheet *failure signatures* table, then the case study's *Failure modes*.
**Looking for a number** → [`00-cheat-sheets/00-master-index.md`](00-cheat-sheets/00-master-index.md).
