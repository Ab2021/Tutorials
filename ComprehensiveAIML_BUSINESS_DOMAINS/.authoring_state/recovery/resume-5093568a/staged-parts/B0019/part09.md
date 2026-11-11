
## 12. Quick Reference Card

**Card 1 — Contract before number.** State four lines and version them: population (which audio, which inclusion rules), transform (normalizer name and hash), statistic (corpus, per-utterance, slice, field), decision rule (threshold, slice, guardrail, confidence). A rate without a contract is not a measurement.

**Card 2 — The arithmetic, compactly.** `E = S + D + I`; `WER = E/N`; `C = N - S - D`; invariants `C + S + D = N`, `C + S + I = H`, `I - D = H - N`. Variants: `CER = E_c/N_c` (declare whether spaces count), `WIP = I/H`, `WIL = I/(C+I)`, `SER = #(E>0)/U`, `DER = (miss+FA+confusion)/T`, `RTF = compute_s/audio_s`.

**Card 3 — Aggregation choices move the number.** On the twelve-row corpus: micro 0.1429, macro 0.1532, unweighted slice mean 0.1565, clinical slice alone 0.3000, name slice 0.1111. Name the aggregation in the same sentence as the number.

**Card 4 — Uncertainty, clustered.** Resample utterances, not tokens; report a 95 percent interval. Twelve utterances at 0.1429 give [0.0815, 0.2213], half-width 0.0699; the binomial figure of 0.0611 understates by 14 percent. Precision scales as `n ∝ 1/hw²`: ±2 pp needs 147 utterances, ±1 pp needs 587.

**Card 5 — Coverage outranks everything.** `scored + failed + missing = planned`, asserted. One corpus with 58 percent coverage produced 0.1429 (skip), 0.0833 (count correct), 0.5000 (impute all), 0.3374 (impute speech-bearing). A spread of 0.357 dwarfs every normalization decision.

**Card 6 — Streaming has two clocks and one extra term.** User-visible delay `= L_final + tau + transport + render`. Twelve measurements gave p95 703.5 ms interpolated, 720 ms nearest-rank, with a bootstrap 95 percent band of [530, 720]. Name the estimator; include `tau`; assert submitted = completed + failed + timed_out.

**Card 7 — Capacity is a rate ratio.** `capacity = workers / RTF`; `utilisation = arrival_audio_h_per_h / capacity`. RTF 0.381455 gives 2.6215 audio-h per worker-hour, so 24 workers sustain 62.917 audio-h/h and 60 audio-h/h arrives at 0.9536 utilisation — about 123 s of queueing for a 30-second file. Alert on backlog seconds, not on RTF.

**Card 8 — Costs, in order of size.** For 1,000 audio hours: annotation 5,250 h ≈ $115,500; adjudication $11,500–$23,100; inference 381.46 GPU-h ≈ $801; storage 115.2 GB raw or 14.4 GB compressed. Annotation is 144 times inference, so spend compute to avoid annotation.

**Card 9 — Scorer cost and the bootstrap trap.** Full DP on `N = H = 4500` is 20.25M cells (162 MB); a 101-wide band is 2.24 percent of that; rolling rows need 70.3 KB. Re-aligning inside a 2,000-resample bootstrap costs 405 s per file — bootstrap the stored `(N_u, E_u)` pairs instead, which is what makes the interval affordable.

**Card 10 — Stop conditions.** Do not promote when: coverage is below the planned population without an imputation policy; the interval on the delta crosses zero; a guardrail slice (critical fields, silence insertions, worst language, latency p95) fails; the normalizer version changed between the two runs; or the reference kappa is below 0.6 for the slice being gated.
