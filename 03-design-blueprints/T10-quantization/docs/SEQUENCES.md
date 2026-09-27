# T10 — Quantization: sequence diagrams

> `T10` · **Transcript coverage:** partial · [HLD](../HLD.md) · [LLD](../LLD.md) · [Runnable core](../run.py) · [Production](../production/README.md)

Six flows. Each one is a decision the HLD argues for, rendered as the order in which things happen
— because in this topic **the order is the design**. The common failure is not a wrong setting but a
right setting applied at the wrong stage (KV precision decided before the checkpoint, re-evaluation
skipped after a KV retune, the gate run on the mean).

---

## 1. Offline quantization, with the gate

The artefact path. Every step before the gate is reversible; the gate is where the deployment
decision is made.

```mermaid
sequenceDiagram
    autonumber
    participant MT as Model team
    participant Q as Quantizer<br/>(AWQ/GPTQ/NF4)
    participant CAL as Calibration set
    participant MAN as quantization-manifest.yaml
    participant EV as Eval harness
    participant GATE as eval_gate<br/>slice=worst
    participant SRV as Serving team

    MT->>Q: base checkpoint (bf16)
    MT->>Q: quantizer=awq, bits=4, group=128, salient_frac=0.01
    Q->>CAL: read activations (REQUIRED for awq/gptq [R])
    CAL-->>Q: 512 samples
    Q->>Q: fit activation-driven salience -> protect top 1% at fp16
    note over Q: effective bits = 4 + 16/128 = 4.125 [D]<br/>storage overhead of the 1% = 3% [D]
    Q-->>MT: quantized checkpoint + per-group scales

    MT->>MAN: write manifest (quantizer, group, salient_frac)
    MT->>EV: evaluate at context_lengths [2048, 8192, 32768]
    note over EV: the DEPLOYED context must be in the list --<br/>KV damage is invisible below it (HLD 7)
    EV-->>GATE: rows [{slice, precision, degradation_pct, source}]

    GATE->>GATE: gate on the WORST row, never the mean
    note over GATE: at a 6% threshold: mean 5.29% PASS,<br/>worst 6.88% REJECT -- same numbers [D]
    alt worst slice within threshold
        GATE-->>MAN: verdict: PASS
        MAN-->>SRV: checkpoint + manifest released
    else worst slice breaches
        GATE-->>MT: verdict: FAIL, with the breaching slice named
        note over MT: remediation is a config change, not a prose caveat:<br/>smaller group, or more salient channels protected
    end
```

**The gate's position is the design.** It sits between "quantized" and "released", not after
release — and it returns the *breaching slice*, because the remediation (a smaller group, or a
higher `salient_frac`) is chosen from which slice broke.

---

## 2. A gate rejection and the re-quantization loop

The loop that costs an eval cycle, and the reason HLD §12.1 says a flat quantizer does not need a
gate while a per-tensor one does.

```mermaid
sequenceDiagram
    autonumber
    participant MT as Model team
    participant Q as Quantizer
    participant EV as Eval harness
    participant GATE as eval_gate
    participant TR as Trade table

    MT->>Q: quantize(bits=4, group=128)
    Q-->>EV: checkpoint A (4.125 effective bits)
    EV-->>GATE: worst-slice delta = 2.02% on gsm8k
    GATE-->>MT: FAIL (threshold 2.0%)

    Note over MT,TR: the loop is over THREE knobs, and each has a price
    MT->>TR: which knob?
    TR-->>MT: group 128 -> 64 : +3% storage, better error (SNR 20.56 -> 21.97 dB [D])
    TR-->>MT: salient_frac 0.01 -> 0.02 : +3% storage, +1.35 dB [D]
    TR-->>MT: bits 4 -> 8 : 2x storage, near-lossless, loses the fit lever
    MT->>Q: re-quantize(group=64)
    Q-->>EV: checkpoint B (4.25 effective bits)
    EV-->>GATE: worst-slice delta = 1.31%
    GATE-->>MT: PASS

    note over MT,GATE: an eval cycle per attempt. At 10x scale this is per model,<br/>per context length and per GPU class (HLD 14)
```

**The loop has an exit that is not another quantization: the `no-quantization-control` route.** If
every configuration breaches on the worst slice, the answer is that this model does not tolerate the
bit width — and the honest outcome is bf16 with fewer sequences, not a fourth attempt.

---

## 3. A KV retune, and the re-evaluation that is usually skipped

The most dangerous flow in the topic, because the change is a config flag — it *feels* like tuning —
and its damage is invisible to the evals most teams run.

```mermaid
sequenceDiagram
    autonumber
    participant OPS as Serving team
    participant CFG as engine-quant.yaml
    participant ENG as vLLM
    participant MET as metrics.promql
    participant EV as Eval harness

    OPS->>MET: concurrency is below target (query 1)
    MET-->>OPS: 52 sequences, expected 208
    OPS->>OPS: which lever? weights are already 4-bit -> KV is the untaken 4x
    OPS->>CFG: kv_cache_dtype: fp8 (was auto)
    Note over CFG: a RESTART-level change, not an artefact change

    CFG->>ENG: restart with calculate_kv_scales=true
    ENG-->>MET: concurrency 208 -> 417 sequences
    note over MET: 2.00x here (fp8); 4.00x at int4 [D]<br/>BOTH are constant in context length

    rect rgb(255, 235, 235)
        Note over OPS,EV: THE STEP THAT IS SKIPPED
        OPS->>EV: RE-EVALUATE at max_model_len (32768)
        EV-->>OPS: worst-slice delta 2.4% at 32k, 0.1% at 2k
        Note over EV: a short-context suite reports this change as FREE.<br/>It is not -- the damage accumulates over positions (HLD 7).
    end

    alt worst slice at the deployed ctx within threshold
        OPS->>ENG: keep fp8 KV
    else breach
        OPS->>CFG: kv_cache_dtype: int8 (2x, gentler) or auto
        Note over OPS: the lever has a middle setting,<br/>which is why the eval is worth running rather than skipping
    end
```

**The red block is the whole diagram.** Without it, this flow is a successful optimization; with it,
it is a validated one. HLD §7's table is the argument: weight quantization is visible on any
benchmark, KV quantization only on long-context ones.

---

## 4. The per-request accounting path — where the two levers act

What actually happens to one request, with the two levers acting on different terms of the same
expression (HLD §8).

```mermaid
sequenceDiagram
    autonumber
    participant R as Request
    participant P as Prefill
    participant KV as KV cache
    participant D as Decode loop
    participant M as Cost model

    R->>P: prompt (4000 tokens)
    Note over P: COMPUTE-bound. cost = 2N per token,<br/>INDEPENDENT of weight precision [D]
    P->>KV: write K,V for 4000 positions

    rect rgb(235, 245, 255)
        Note over P: LEVER 4 -- prefix caching<br/>removes the PREFILL term entirely, but only if the prefix is STABLE [T]<br/>worth 57% on this ratio-40 workload, 1% on an output-heavy one [D]
    end

    loop decode (100 tokens)
        D->>KV: read W/batch + ctx*kv_per_token
        Note over D: LEVER 1 -- batching: divides W/batch (single-digit depth suffices [D])<br/>LEVER 2 -- weight quant: shrinks W (1.21x on concurrency, but the FIT lever)<br/>LEVER 3 -- KV quant: shrinks ctx*kv_per_token (4.00x, constant in ctx)
    end
    D-->>R: 100 output tokens

    M->>M: total = output*(W/batch + ctx*kv)/W_ref + prompt*(1-hit)*prefill_per_token
    Note over M: FOUR terms, FOUR levers, and each rung of the corpus's<br/>ladder touches exactly one. Which rung is worth climbing<br/>is a property of the workload's SHAPE, not of the code.
```

**The `W_ref` in the denominator is a constant, and it has to be.** The decode term is a ratio; if
the reference were recomputed per precision, shrinking `W` would *inflate* the result and the ladder
would read 100 → 42 → **71** — "4-bit costs more than not quantizing" (LLD §3.4).

---

## 5. The ladder's four rungs as one traversal

The corpus's ladder `[T]` walked in order, with each rung's mechanism and its measured residual
against the corpus's own figure.

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant C1 as Rung 1: naive
    participant C2 as Rung 2: batching
    participant C3 as Rung 3: 4-bit weights
    participant C4 as Rung 4: prefix cache
    participant CORP as Corpus ladder [T]

    U->>C1: 16-bit, batch 1, no cache
    C1-->>U: 100.0 units (the reference)
    CORP-->>U: 100  (delta 0.0)

    U->>C2: enable continuous batching
    C2-->>U: 35.9 units
    note over C2: divides the WEIGHT term only.<br/>solve_batch(0.42) -> effective batch ~4 [D]
    CORP-->>U: 42  (delta -6.1)

    U->>C3: quantize weights to 4-bit (group 128)
    C3-->>U: 20.0 units
    note over C3: shrinks W AND the reference proportionally.<br/>The SMALLEST single step (x0.62), not x0.25 --<br/>the KV term is untouched and 4.125 != 4 [D]
    CORP-->>U: 26  (delta -6.0, the largest residual -- REPORTED)

    U->>C4: enable prefix caching (95% hit)
    C4-->>U: 9.0 units
    note over C4: removes the PREFILL term entirely.<br/>The LARGEST step -- and entirely a property of the workload's shape
    CORP-->>U: 11  (delta -2.0)

    Note over U,CORP: residuals are printed on every run: a mechanism, not a fit.<br/>What it does NOT include is KV quantization -- a FIFTH lever,<br/>worth 4.00x on concurrency and larger than rung 3 for capacity [D]
```

**Rung 3's residual is the largest, and it is reported rather than tuned away.** The model credits
weight quantization slightly more than the corpus's composite does; claiming a perfect reproduction
would be the dishonest move.

---

## 6. The AWQ salience scan

The one flow in the topic that is a *policy* rather than a format: quantize everything, then restore
the channels the activations say matter.

```mermaid
sequenceDiagram
    autonumber
    participant CAL as Calibration set
    participant S as salience()
    participant CH as Channel ranking
    participant Q as Quantizer
    participant ST as Storage accounting
    participant EV as Eval

    CAL->>S: activations over 512 samples
    Note over S: real AWQ salience is ACTIVATION-derived [R].<br/>The blueprint's L1-norm proxy is an OFFLINE stand-in --<br/>correlated, not equal (LLD 3.2)
    S->>CH: per-channel importance
    CH->>CH: rank descending
    CH->>Q: top salient_frac (0.01 = 1% [R]) -> keep at fp16
    Q->>Q: quantize the remaining 99% at 4-bit, group 32

    Q->>ST: protected_storage_overhead(4, 0.01)
    ST-->>Q: 0.01 * (16-4)/4 = 3% more bytes [D]
    Note over ST: AWQ without its storage cost is half a trade

    Q-->>EV: quantized tensor
    EV-->>EV: SNR 20.56 -> 21.67 dB (+1.12) for +3% storage [D]
    Note over EV: the first 1% recovers a disproportionate share.<br/>0.10 gives +8.52 dB for 30%, 0.25 gives +9.34 for 75% --<br/>diminishing fast, which is the empirical case for 1% [D]

    alt the quantizer is already flat across severity
        EV-->>Q: no re-ranking needed -- group-wise is flat (20.22 -> 20.61 dB over 48x severity [D])
    else per-tensor
        EV-->>Q: re-rank per LAYER -- the benefit is large and model-specific
    end
```

**The branch at the end is HLD §12.1.** AWQ's benefit depends on the quantizer beneath it: with
group-wise scales across the tensor, protection adds less because the error is already localised.
**The same policy is worth different amounts under different quantizers**, which is why the
calibration is part of the artefact rather than a one-off.

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the ladder traversed in §5 (`[T]`); "always rerun your evals after quantizing" (§1); "caching only helps a stable prefix" (§4).
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/03-training-and-adaptation/07-quantization-deep-dive.md` — AWQ's activation-derived 1% `[R]` (§6); "allow 4x higher concurrency on the same GPU" `[R]` (§3).
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` — prefill's compute-bound character (§4), which makes the prefill term precision-independent.
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` — the native low-precision training path behind §1's quantizer list.

Blueprint: [`HLD.md`](../HLD.md) §§3, 7, 8, 12 · [`LLD.md`](../LLD.md) §§3.2, 3.4, 4, 5 · [`production/README.md`](../production/README.md) §§1–5.
