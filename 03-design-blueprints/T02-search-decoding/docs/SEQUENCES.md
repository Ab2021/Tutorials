# Sequences: Beam, A* and Best-First Search

> `T02` · [HLD](../HLD.md) · [LLD](../LLD.md) · [Case study](../../../01-case-studies/T02-search-decoding.md)
> **Transcript coverage:** primary · [Cheat sheet](../../../00-cheat-sheets/T02-search-decoding.md) · [Interview bank](../../../02-interview-questions/T02-search-decoding.md) · [Runnable core](../run.py)

End-to-end flows for the search-decoding plane. Each step names where it can fail; the failure
taxonomy is in [LLD §7](../LLD.md#7-error-handling).

---

## 1. Cold start — first request after a policy release

```mermaid
sequenceDiagram
    autonumber
    participant GW as Gateway
    participant REG as Policy Registry
    participant DEC as Search Decoder
    participant ALPHA as alpha Table
    participant KV as KV Pool

    GW->>REG: fetch active bundle (version 2026.09.3)
    REG-->>GW: bundle + eval-run id
    GW->>DEC: decode(request, bundle)
    DEC->>ALPHA: alpha for target language
    alt language present in table
        ALPHA-->>DEC: alpha = 1.1
    else language missing
        ALPHA-->>DEC: NotFound
        DEC-->>GW: 422 policy_incomplete
        Note over DEC: Refused, not defaulted. A wrong alpha is a<br/>systematic quality change, not a graceful degradation.
    end
    DEC->>KV: reserve K forked sequences (worst case)
    KV-->>DEC: blocks
    DEC-->>GW: SearchResult
```

**Failure point:** a bundle that loads without a gate id. The registry rejects it at load; a bundle
without a gate is a policy nobody evaluated.

---

## 2. Steady state — one beam step

```mermaid
sequenceDiagram
    autonumber
    participant Q as Priority Queue
    participant EXP as Expander
    participant REC as Recombiner
    participant NORM as Normalizer
    participant KV as KV Pool

    Q->>Q: pop best by comparator (length, then score)
    Q->>EXP: expand top K
    EXP->>KV: allocate blocks for K forked sequences
    EXP->>KV: reuse prefix blocks where beams share a prefix
    EXP-->>Q: up to K^2 candidates (fewer if a beam hit EOS)
    Q->>REC: recombine (n-gram n=3)
    REC-->>Q: cluster representatives
    Q->>NORM: prune to K by normalized score
    NORM-->>Q: surviving K
    Note over Q: EOS beams are NOT expanded.<br/>Width-3 example: 9 candidates, then 7 total.
```

**Failure point:** the normaliser used at prune time differing from the one used at final rescore.
That is a silent objective change at the last moment and it is the mechanism behind "width up, BLEU
down."

---

## 3. The curse of beam search — width raised, BLEU falls

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Search Engineer
    participant SWEEP as Width Sweep
    participant NORM as alpha Calibrator
    participant GATE as Eval Gate
    participant HUMAN as Human Review

    ENG->>SWEEP: run widths 1,2,4,8 over held-out set
    SWEEP-->>ENG: BLEU curve rising to 4, falling at 8
    ENG->>NORM: re-tune alpha first
    NORM-->>SWEEP: re-run the sweep at the new alpha
    alt degradation removed
        SWEEP-->>GATE: BLEU monotone in width
        GATE-->>ENG: ship
    else degradation persists
        Note over GATE: The objective is wrong, not the search.
        GATE->>HUMAN: sample 500 outputs, compare to metric
        HUMAN-->>ENG: metric and judgement diverge
        ENG->>GATE: REFUSE the release; escalate to reranker (T05)
    end
```

The order is prescribed: tune length normalisation first, and only then conclude the objective is at
fault. Both branches are legitimate outcomes and the second one is not a failure of the search design.

---

## 4. Bounded A\* on a glossary segment

```mermaid
sequenceDiagram
    autonumber
    participant DEC as Decoder
    participant GLOSS as Glossary Service
    participant PQ as Priority Queue
    participant MON as Overestimate Monitor

    DEC->>GLOSS: compile(glossary, target_lang)
    alt compiles to an FSA
        GLOSS-->>DEC: automaton, bounded length L
        DEC->>PQ: seed with h = row_min
        loop until length L
            PQ->>PQ: pop min f = g + h
            PQ->>MON: assert h <= true remaining
            MON-->>PQ: ok
        end
        PQ-->>DEC: optimal path within the segment
    else compile fails
        GLOSS-->>DEC: error
        DEC-->>DEC: route to plain beam
        Note over DEC: The glossary is NOT silently dropped.<br/>A dropped glossary is a customer-visible term error.
    end
```

**Failure point:** applying the same row-minimum heuristic one line over, on the open-ended decoder.
There EOS is available and can be cheaper than the state's cheapest non-EOS arc, so the row minimum
overestimates and the guarantee is void. The monitor exists to make that a counted event, and the
`InadmissibleHeuristicLive` alert is a page.

---

## 5. Failure — best-first queue blow-up

```mermaid
sequenceDiagram
    autonumber
    participant DEC as Decoder
    participant Q as Priority Queue
    participant OBS as Telemetry
    participant POL as Policy

    DEC->>Q: expand at K=8, long document
    Q->>Q: frontier grows
    Q->>OBS: queue_depth
    OBS->>POL: depth > 100000
    POL->>DEC: degradation ladder step 1
    DEC->>DEC: best-first -> plain beam, same K
    Note over DEC: Output unchanged. Only the speed-up is lost.
    OBS->>POL: kv utilisation still > 0.85
    POL->>DEC: step 2: halve K
    DEC-->>DEC: emit at reduced width
```

Step 1 is chosen first because it is the only rung that loses *nothing* — best-first beam search
returns identical results to plain beam by construction, so the first response to pressure is to
give up the speed-up rather than the quality.

---

## 6. Recovery — a bad alpha after a model upgrade

```mermaid
sequenceDiagram
    autonumber
    participant MON as Length Monitor
    participant ONCALL as On-call
    participant REG as Policy Registry
    participant CAL as alpha Calibrator

    MON->>ONCALL: output-length histogram shifted short, all fr
    ONCALL->>REG: roll the bundle back to 2026.09.2
    REG-->>ONCALL: previous alpha table active
    ONCALL->>CAL: re-calibrate alpha for fr on the new model
    CAL-->>REG: new bundle, gated
    Note over ONCALL: Roll back BEFORE touching width.<br/>Width is the second suspect, alpha the first.
```

---

## 7. Scale-out

Horizontal scale is unremarkable — the decoder is stateless and the engine scales out. The
interesting axis is vertical, and it is the reason `K` is a function of work rather than a constant:

```mermaid
sequenceDiagram
    autonumber
    participant LB as Load Balancer
    participant E1 as Engine A (beam pool)
    participant E2 as Engine B (interactive pool)
    participant KV as KV Pool

    LB->>E1: width-4 prose requests
    LB->>E2: greedy interactive requests
    Note over E1,E2: Split by width class, not by tenant.<br/>A width-4 request occupies 4 forked sequences<br/>and will produce TTFT spikes the interactive<br/>tenant can see if they share a pool.
    E1->>KV: K forked sequences per request
    E1->>E1: cap K by document length
```

---

## 8. Replay for a contract dispute

```mermaid
sequenceDiagram
    autonumber
    participant CUST as Customer
    participant SUP as Support
    participant JRN as Replay Journal
    participant DEC as Decoder

    CUST->>SUP: "this translation was wrong"
    SUP->>JRN: request_id
    JRN-->>SUP: request + policy_version + stack_pin
    SUP->>DEC: replay at the pinned version
    alt stack matches the pin
        DEC-->>SUP: byte-identical output
        SUP-->>CUST: here is the exact output and the policy
    else stack drifted
        DEC-->>SUP: replay unavailable
        SUP-->>CUST: we can reproduce the policy, not the bytes
        Note over SUP: Cross-version replay is not promised.<br/>Same reason as T01: GPU reduction order,<br/>MoE routing and quantization change the bytes.
    end
```

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_5_A_and_Best_First_Search.txt` — weighted finite-state automata; `f = g + h`; the greedy/beam/uniform-cost/A\* step counts; admissibility as sufficient but not necessary; row-minimum heuristics; hypothesis recombination; the 16x16 KL matrix
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt` — beam mechanics; log-probs for numerical stability; EOS unfairness and the HuggingFace length penalty; diverse beam search; stochastic beam search and the Gumbel-max trick
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt` — the reranker route taken when the objective rather than the search is wrong
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — decode is memory-bandwidth-bound; the occupancy argument

**All sequence structure is `[D]`.** Corpus facts (`[T]`) are attributed inline; the failure taxonomy
is in [LLD §7](../LLD.md#7-error-handling).
