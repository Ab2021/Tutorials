
## 9. Complexity & Cost Analysis

### 9.1 Algorithmic complexity of scoring

| Operation | Time | Space | Note |
|---|---|---|---|
| Full DP alignment | `O(N·H)` | `O(N·H)` | path recoverable |
| Rolling two-row DP | `O(N·H)` | `O(min(N,H))` | score only |
| Banded DP, half-width `k` | `O(k·N)` | `O(k)` | requires `k ≥ true distance` |
| Myers / wavefront | `O(N·D)` | `O(N)` bits | best when `D` is small |
| Cluster bootstrap, `B` resamples | `O(B·N·H)` naive | `O(N·H)` | the dominant cost, see below |
| Percentile of `n` values | `O(n log n)` | `O(n)` | or `O(n)` with selection |
| Slice aggregation | `O(U)` | `O(U)` | trivial next to alignment |

Numbers for a 30-minute call at 150 words per minute: `N = H ≈ 4500`. Full DP is `4500 × 4500 = 20.25` million cells; at one 8-byte integer each that is 162 MB per file. Banded with `k = 50` needs a 101-cell band, `101 × 4500 = 454,500` cells, or 2.24 percent of the full table. Myers costs `O(N·D)`; at a realised distance of 300 edits that is `4500 × 300 = 1.35` million operations, which is roughly the same order as the banded table but with much smaller constants in practice.

**Why the bootstrap dominates.** Re-aligning on every resample is `B` times the alignment cost. At `B = 2000`, a naive implementation performs `2000 × 20.25M = 40.5` billion cell updates per file; at a generous 100 million cell updates per second that is **405 seconds per file**, which is unusable. Three fixes, in order of value:

1. **Resample the error counts, not the text.** Store `(N_u, E_u)` per utterance; bootstrap the ratio `sum(E*)/sum(N*)`. Alignment happens once, not `B` times. This is what section 5.2.4 does and it reduces the cost from 405 s to milliseconds.
2. **Vectorise the aggregation.** A bootstrap over 12 to 5,000 utterances is a matrix product; a compiled array library does `B = 2000` resamples of 5,000 utterances in under a second.
3. **Reduce `B`.** 1,000 resamples gives stable 2.5 and 97.5 percentiles to about three decimals for rates around 0.1; more than 2,000 rarely changes a decision.

### 9.2 Corpus size: the cost of precision

Every point of precision costs utterances quadratically, and every utterance costs annotation. With a per-utterance annotation cost of 2.625 minutes (a 30-second clip at roughly 5.25 times real time, including review and a second check on a sample of 20 percent):

| Target half-width | Utterances needed | Annotation hours | Cost at $22/h |
|---|---:|---:|---:|
| ±10 pp | 12 | 0.53 | $11.55 |
| ±5 pp | 47 | 2.06 | $45.24 |
| ±2 pp | 147 | 6.43 | $141.53 |
| ±1 pp | 587 | 25.68 | $564.96 |
| ±0.5 pp | 2,347 | 102.68 | $2,258.96 |

The scaling is exact `n ∝ 1/hw²`, so a tenfold precision gain costs a hundredfold corpus. The right-hand columns use the per-clip cost arithmetic of 9.4; the absolute currency values are ILLUSTRATIVE and the ratios are what matter.

**A cheaper route to the same decision.** If the question is "is the candidate better than the incumbent", a paired design needs far fewer utterances than an absolute measurement, because the shared difficulty cancels. Section 5.2.5's paired interval had half-width 0.062 on 12 utterances; to reach 0.03 on the *delta* requires `12 × (0.06215/0.03)² = 12 × 4.29 = 51.5`, so **52 utterances** — three times fewer than the 147 needed to pin an absolute rate to ±2 pp. Teams that measure two systems independently and compare the points waste most of their annotation budget.

### 9.3 Compute cost of running the evaluation

```text
 FORMULA     compute_hours = audio_hours * RTF
 WHERE       RTF = compute seconds per audio second, measured on the target
             hardware with the target model, excluding queue wait
 CONSTRAINT  excludes re-decoding for ablations; multiply by the number of
             variants under test
 EXAMPLE     1,000 audio hours at RTF 0.381455 -> 381.46 GPU-hours
             at $2.10 per GPU-hour -> 1,000 * 0.381455 * 2.10 = $801.06
 INTERPRET   decoding a thousand hours costs about eight hundred dollars;
             annotating it costs two orders of magnitude more
```

**Annotation dominates.** A thousand hours at 5.25 annotation hours per audio hour is 5,250 hours, which at $22/h is $115,500 — about 144 times the compute cost (`115500 / 801.06 = 144.2`). This single ratio determines the whole evaluation strategy: it is rational to spend compute extravagantly to avoid annotation. Concretely, spending ten times the compute to select the 5 percent of audio most likely to be informative is a 15-fold saving on the annotation line.

### 9.4 Cost per scored unit

Human review of a sample, priced properly:

```text
 FORMULA     review_cost = audio_hours * sample_share * review_ratio * rate
 WHERE       review_ratio = real-time hours per audio hour (a 4x listener
             takes 1 hour to audit 4 hours of audio)
 EXAMPLE     30 audio hours, 8 percent sample, 4x, $34/h
             30 * 0.08 = 2.4 audio hours sampled
             2.4 / 4 = 0.6 listening hours
             0.6 * 34 = $20.40
 INTERPRET   a small audit is cheap; the discipline of doing it is what is
             scarce, not the money
```

Recompute the same figure with the common error of multiplying instead of dividing by the review ratio: `2.4 × 4 × 34 = $326.40`, sixteen times too high. A cost model with a units error of this kind will conclude that no audit is affordable, which is how quality programmes die.

### 9.5 Capacity and queueing

A worker pool of `W` parallel decoders at RTF `rho` sustains `W / rho` audio-hours per wall-hour.

```text
 FORMULA     capacity = W / RTF      utilisation = arrival_audio_h_per_h / capacity
 WHERE       arrival measured in audio hours, not in requests
 CONSTRAINT  independent per-worker throughput; no shared bottleneck (storage,
             model server, network) below the decoder throughput
 EXAMPLE     W = 24, RTF = 0.381455 -> capacity 24 / 0.381455 = 62.917
             arrival 24 audio-h/h -> utilisation 24 / 62.917 = 0.3815
             arrival 60 audio-h/h -> utilisation 0.9536 -> queue explodes
 INTERPRET   utilisation is the ratio of two audio-hour rates; comparing a
             request rate to a throughput figure without unit conversion is
             the classic capacity-planning error
```

At utilisation 0.9536 a single-queue model predicts a mean wait of `rho(2-rho)/(2(1-rho))` service times for a deterministic service process: `0.9536 × 1.0464/(2 × 0.0464) = 10.75` service times. With a 30-second file taking `0.381455 × 30 = 11.44` s of compute, that is `10.75 × 11.44 = 123` s of queueing — a four-minute lag on a system that "has spare capacity" by the naive `RTF < 1` test.

### 9.6 Storage and I/O

| Item | Per 1,000 audio hours | Note |
|---|---|---|
| Raw PCM, 16 kHz mono 16-bit | 115.2 GB | `16000 × 2 × 3600 × 1000 / 10^9` |
| Compressed speech, 32 kbps | 14.4 GB | adequate for most ASR front ends |
| Transcripts at 150 wpm | 9.0 million words | about 60 MB of text |
| Alignments (word-level JSON) | 5 to 15 GB | the largest derived artifact |
| Evaluation artifacts (scores, slices) | under 1 GB | keep forever |

The ratio to note: transcripts are three orders of magnitude smaller than audio, and alignments are two orders larger than transcripts. Retention policy should therefore treat audio and alignments as expensive and transcripts as cheap, which is the opposite of the usual "keep the audio, regenerate the rest" instinct.

### 9.7 Where the money actually goes

Ordered by typical magnitude for a 1,000-hour evaluation, all ILLUSTRATIVE:

1. **Annotation** — $115,500 at 5.25 h per audio hour and $22/h. Dominant by a factor of 144 over inference.
2. **Human adjudication on disagreements** — typically 10 to 20 percent of the annotation line, so $11,500 to $23,100.
3. **Ongoing human review sampling at 8 percent** — $680 for 1,000 hours at the 9.4 rates, but recurring every release.
4. **Inference** — $801 for a single full pass.
5. **Storage** — tens of dollars per month at commodity rates.
6. **Engineering time on the harness** — the largest *hidden* cost; a scorer with normalizer versioning, slice reporting and bootstrap intervals is a small software project, and teams routinely rebuild it three times.

The ordering explains a common planning failure: teams optimise the inference line because it is visible in a cloud bill, and under-invest in the annotation and harness lines that determine whether the number means anything.

**Takeaway.** Alignment is cheap, bootstrap over text is not, and annotation costs roughly a hundred times more than inference — so spend compute to avoid annotation, and never trade corpus coverage for a cheaper normalizer.

## 10. Key Theorems / Results Reference

### 10.1 Edit distance and the WER identity

**Result.** For sequences `R` and `H` with unit insertion, deletion and substitution costs, the minimum edit distance `E` satisfies `E = S + D + I` for some optimal alignment, and every optimal alignment yields the same `E`. With `C` matches, `C + S + D = N` and `C + S + I = H`.

**Why it matters.** The score is well defined even though the decomposition is not. Subtract the two identities to get `I - D = H - N`, a useful invariant: the excess of insertions over deletions must equal the length difference. On the section 5 corpus, `I - D = 8 - 3 = 5` and `H - N = 131 - 126 = 5`. Any reported `S/D/I` triple violating this is a scorer bug, not a model property.

**Validity.** Unit costs only; with weighted costs (for example, a substitution priced at 1 but a deletion at 0.7) the identity in this form does not hold and `E` must be recomputed from the weights.

### 10.2 Triangle inequality and the impossibility of an additive decomposition

Edit distance is a metric: `d(a,c) ≤ d(a,b) + d(b,c)`. This gives a **bound** on what a two-stage pipeline can cost. If stage 1 (raw decode) has distance `d1` from the reference and stage 2 (ITN/formatting) moves the text by `d2`, then the final distance is at most `d1 + d2`. It is not equal, and it can be smaller — formatting can repair an error — but it can never exceed the sum. The practical value is a diagnostic: a pipeline whose measured distance exceeds `d1 + d2` has a scorer or normalization bug, because no sequence of edits can violate the triangle inequality.

### 10.3 WER as a plug-in estimator of a population ratio

**Result.** `WER` computed on a sample is a ratio estimator of the population quantity `E[E_u]/E[N_u]`. It is consistent and asymptotically normal, but its variance is *not* the binomial variance of `E/N`, because the numerator and denominator are correlated and both random.

**Consequence.** The naive binomial standard error `sqrt(p(1-p)/n_tokens)` understates the true uncertainty whenever utterances vary in difficulty. On the section 5 corpus it gives 0.0312 against a clustered 0.0357-equivalent (`0.0699/1.96`), a 14 percent understatement. The practical rule: cluster by utterance, always.

### 10.4 The rule of three for zero-event bounds

**Result.** With `n` independent trials and zero observed events, the 95 percent upper confidence bound on the event probability is approximately `3/n`.

**Derivation in one line.** `(1-p)^n ≥ 0.05` implies `n ln(1-p) ≥ ln 0.05`, so `p ≥ 1 - 0.05^(1/n)`. For `p` small, `1 - 0.05^(1/n) ≈ 2.996/n`, giving the familiar `3/n`.

**Use.** 59 clean utterances bound the per-utterance failure rate at `3/59 = 0.0508`. If a claim needs the bound below 1 percent, `n` must exceed 300; below 0.1 percent, `n` must exceed 3,000. These are the numbers to quote when someone proposes to certify a rare failure class from a small clean sample.

### 10.5 Little's law and the backlog identity

**Result.** In a stationary system, `L = lambda * W`: the average number in system equals the arrival rate times the average time in system. Applied to audio, with `lambda` in audio-hours per wall-hour and `W` in wall-hours per audio-hour, `L` is the average backlog in audio-hours.

**Use.** If arrival is 24 audio-h/h and mean residence is `0.6167` wall-hours per audio-hour (from the utilisation formula in 9.5 at `rho = 0.3815`, `0.381455/(1-0.381455) = 0.61670`), the backlog is `24 × 0.6167 = 14.80` audio-hours. Any monitored backlog above that figure means arrivals rose or capacity fell; the identity converts a single number into an actionable pair.

**Validity.** Stationary arrivals and a stable service discipline. During a traffic ramp the identity does not hold instantaneously and backlog is the better signal.

### 10.6 Cohen's kappa as the reference-quality ceiling

**Result.** `kappa = (p_o - p_e)/(1 - p_e)` where `p_o` is observed agreement and `p_e` is agreement expected from the marginals.

**Use as a ceiling.** If two independent annotators agree on a binary correctness judgement with kappa 0.33, the measurement cannot resolve differences smaller than roughly the disagreement rate. The section 8.14 example gives `p_o = 0.79`, `p_e = 0.6856`, kappa 0.332, so a 3-point improvement claim is below the noise floor of the reference itself.

**Validity.** Two raters, nominal categories, independent judgement. Kappa penalises skewed marginals heavily, so a high raw agreement with a very skewed positive rate can still yield a low kappa; report both.

### 10.7 Percentile estimator disagreement

**Result.** For `n` samples, the interpolated `p`-quantile and the nearest-rank `p`-quantile differ by at most one order statistic. With small `n` that "one order statistic" can be a large absolute gap.

**Numbers.** On the twelve latencies, interpolated p90 is 682.0 ms and nearest-rank p90 is 690 ms (the 11th of 12 values), a 8.0 ms gap; but the interpolated p95 is 703.5 ms against a nearest-rank 720 ms, a 16.5 ms gap, and the bootstrap interval for the p95 spans 530 to 720 ms. The estimator choice is a **gate parameter** and belongs in the evaluation contract, not in a library default.

### 10.8 Sampling-noise floor for a paired design

**Result.** The standard error of a paired difference scales with the standard deviation of per-utterance *differences*, not of per-utterance rates. Utterances that are equally hard cancel.

**Numbers.** On the section 5 corpus the per-utterance improvement has a bootstrap standard error of `0.06215/1.96 = 0.0317` on the delta, which is smaller than the 0.0425 standard error of the macro rate itself. That is the entire argument for paired designs: the same twelve utterances give a tighter statement about a difference than about an absolute level.

**Takeaway.** Five results carry most of the practical weight: `I - D = H - N`, the triangle-inequality bound for pipelines, the clustered variance for ratio estimators, the rule of three for clean samples, and Little's law for backlog.
