
## 5. Worked Numerical Example

The corpus below is constructed. Twelve utterances, one reference and one hypothesis each, four slices, plus audio durations and compute times. Everything is ILLUSTRATIVE; the arithmetic is exact to the digits shown and can be reproduced with a calculator.

### 5.1 The data

| ID | Slice | Dur (s) | Ref words `N` | `S` | `D` | `I` | `E` | WER_utt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E01 | routine | 9.4 | 13 | 1 | 0 | 0 | 1 | 0.0769 |
| E02 | routine | 7.2 | 12 | 0 | 0 | 0 | 0 | 0.0000 |
| E03 | numbers | 6.8 | 10 | 0 | 1 | 0 | 1 | 0.1000 |
| E04 | routine | 8.1 | 11 | 0 | 0 | 2 | 2 | 0.1818 |
| E05 | clinical | 6.0 | 9 | 1 | 0 | 2 | 3 | 0.3333 |
| E06 | routine | 5.5 | 8 | 1 | 1 | 2 | 4 | 0.5000 |
| E07 | numbers | 7.9 | 11 | 1 | 0 | 0 | 1 | 0.0909 |
| E08 | names | 5.1 | 9 | 1 | 0 | 0 | 1 | 0.1111 |
| E09 | routine | 8.8 | 14 | 1 | 0 | 0 | 1 | 0.0714 |
| E10 | numbers | 6.3 | 8 | 0 | 0 | 0 | 0 | 0.0000 |
| E11 | clinical | 7.7 | 11 | 1 | 0 | 2 | 3 | 0.2727 |
| E12 | numbers | 6.4 | 10 | 0 | 1 | 0 | 1 | 0.1000 |
| **Sum** | | **85.2** | **126** | **7** | **3** | **8** | **18** | |

Row identity checks: `C = N - S - D` gives 12, 12, 9, 11, 8, 6, 10, 8, 13, 8, 10, 9. Adding `C + S + I` per row reproduces hypothesis lengths 13, 12, 9, 13, 11, 7, 11, 9, 14, 8, 13, 10, whose total is 131. Both column sums reconcile, and either of the two relations alone would catch a mis-keyed row.

### 5.2 Five aggregations of the same twelve rows

**5.2.1 Corpus (micro) rate.**

```text
 FORMULA     WER_micro = sum(E) / sum(N)
 WHERE       all 12 utterances, all 4 slices
 CONSTRAINT  sum(N) > 0
 EXAMPLE     WER = 18 / 126 = 0.142857
 INTERPRET   14.3 percent of all reference tokens would need editing
```

**5.2.2 Per-utterance average (macro).**

Sum the twelve per-utterance rates at full precision: 1/13 + 0/12 + 1/10 + 2/11 + 3/9 + 4/8 + 1/11 + 1/9 + 1/14 + 0/8 + 3/11 + 1/10 = 1.838251. Divide by 12: **0.153188**. The macro rate exceeds micro by 0.010331, which is 7.2 percent of the micro value — larger than many improvements a team would call a win.

**5.2.3 Slice rates and a weighting check.**

| Slice | `E` | `N` | Rate | Sample share |
|---|---:|---:|---:|---:|
| routine | 8 | 58 | 0.1379 | 0.4603 |
| numbers | 3 | 39 | 0.0769 | 0.3095 |
| clinical | 6 | 20 | 0.3000 | 0.1587 |
| names | 1 | 9 | 0.1111 | 0.0714 |

Reconstructing micro from the rounded parts: `0.4603 × 0.1379 = 0.06348`, `0.3095 × 0.0769 = 0.02380`, `0.1587 × 0.3000 = 0.04761`, `0.0714 × 0.1111 = 0.00793`. Sum `= 0.14282`, against the exact 0.142857 — a 0.00004 gap that exists only because the shares were rounded to four decimals. The unweighted mean of the four slice rates is `0.6260 / 4 = 0.1565`, which is 0.0136 above micro.

This is the argument for never publishing a bare slice average. The clinical slice holds 20 of 126 tokens, yet its 0.30 rate influences an unweighted slice mean as much as the routine slice holding 58.

**5.2.4 Uncertainty: a clustered bootstrap.**

The resampling unit is the utterance, not the token, because errors inside an utterance are correlated — one bad endpoint produces several edits at once. Resample twelve utterances with replacement 2,000 times, recompute `sum(E*)/sum(N*)` each time, and take the 2.5th and 97.5th percentiles:

```text
 FORMULA     WER_boot at percentile q of { sum(E*)/sum(N*) } over resamples
 WHERE       utterance-level resampling with replacement, B = 2000
 CONSTRAINT  utterances are the independent unit; B must be large enough
             that the 2.5th percentile is stable to the precision quoted
 EXAMPLE     point 0.1429 ; 2.5th 0.0815 ; 97.5th 0.2213
             half-width (0.2213 - 0.0815)/2 = 0.0699
 INTERPRET   the value is pinned to a band of roughly 0.08 to 0.22; a three
             point improvement is well inside this band
```

A naive binomial calculation that treats all 126 tokens as independent gives `SE = sqrt(0.142857 × 0.857143 / 126) = 0.031172` and a 95 percent half-width of `1.96 × 0.031172 = 0.06110`. The bootstrap half-width of 0.0699 is **14.4 percent wider** (`0.0699 / 0.06110 = 1.144`), which is the design effect of clustering. Quoting the binomial interval understates uncertainty, and because the understatement is modest it is easy for a reviewer to miss.

**5.2.5 How many utterances for a promised precision?**

```text
 FORMULA     n approx n0 * (halfwidth0 / halfwidth_target)^2
 WHERE       n0 = 12 utterances, halfwidth0 = 0.0699
 CONSTRAINT  the larger sample resembles the old in rate and in clustering;
             adding a new language or channel generally needs more
 EXAMPLE     +/-2 pp: 12 * (0.0699/0.02)^2 = 12 * 12.215 = 146.6 -> 147
             +/-1 pp: 12 * (0.0699/0.01)^2 = 586.3 -> 587 utterances
 INTERPRET   a claim precise to one point needs roughly fifty times the corpus
```

The scaling is quadratic in precision and linear in heterogeneity, which is why "just add a hundred files" rarely buys the precision a launch review assumes it bought.

### 5.3 Coverage: the number that outranks normalization

Take the same twelve scored utterances plus 24 more planned ones. Of the 24, twelve were transcribed normally and are already inside the twelve rows; the other twelve produced no hypothesis and carry 90 reference tokens. A listening pass establishes that 37 of those 90 tokens contain speech and 53 are silence or line noise.

```text
 FORMULA     four candidate rates over one situation
 WHERE       planned population 216 tokens / 36 utterances
 EXAMPLE
   A skip the missing twelve:        18 / 126 = 0.142857   coverage 58.3%
   B count them correct:             18 / 216 = 0.083333   coverage 58.3%
   C impute all missing as deletions:108 / 216 = 0.500000   coverage 100%
   D impute only speech-bearing:     55 / 163 = 0.337423   coverage 75.5%
 CONSTRAINT  D requires a human listening pass over the missing audio;
             without it the honest output is "coverage 58 percent, rate
             undefined on the remainder"
 INTERPRET   the spread between A and C is 0.357, twenty-four times larger
             than the normalization choices examined in 5.7
```

Option B is not a straw man; it appears whenever a pipeline joins on hypothesis rows and treats the absence of a row as a perfect match. Option A is what most notebooks do by default, because the evaluation set is simply "the files that produced output".

### 5.4 Streaming: percentiles, estimator choice and a budget breach

Finalisation latency for the same twelve utterances, in milliseconds measured from the last speech frame, sorted ascending: 380, 395, 405, 420, 440, 455, 470, 510, 530, 610, 690, 720.

```text
 FORMULA     sorted x[0..11]; linear interpolation at k = 11 * p
 WHERE       small-sample percentile, stated estimator
 CONSTRAINT  name the estimator; nearest-rank answers differ by up to 80 ms
 EXAMPLE
   p50: k = 5.50 -> 455 + 0.50*(470-455) = 462.5 ms
   p90: k = 9.90 -> 610 + 0.90*(690-610) = 682.0 ms
   p95: k = 10.45-> 690 + 0.45*(720-690) = 703.5 ms
   mean 502.08 ms ; max 720 ms
   nearest-rank on the same data: p50 455, p90 690, p95 720
 INTERPRET   a 700 ms p95 budget is breached by 3.5 ms under interpolation
             and by 20 ms under nearest-rank; the estimator decides the gate
```

**5.4.1 How much can twelve samples tell you about a p95?** Resampling the twelve values 4,000 times gives a 95 percent interval for the p95 of **[530, 720] ms**. The point estimate 703.5 sits near the top of that band, and the band itself is 190 ms wide. A latency gate built on twelve files is a gate that will pass or fail based on which file happened to be slowest. Section 9 prices the corpus size that fixes this; the honest use of this table is to illustrate the arithmetic, not to certify a service.

**5.4.2 Adding endpoint hangover.** Each segment waits `tau` seconds of trailing silence before it closes, so user-visible finalisation is `L_final + tau`. With `tau = 0.30` s the p95 becomes 1003.5 ms and the breach against a 700 ms budget is 303.5 ms, not 3.5 ms. Teams routinely measure the decoder's `L_final`, quote 703 ms, and ship a product that waits a full second — the largest single discrepancy between a speech metric and a user's experience.

**5.4.3 Compute real-time factor.**

```text
 FORMULA     RTF = total compute seconds / total audio seconds
 WHERE       compute times 3.61, 2.75, 2.58, 3.12, 2.31, 2.09, 3.02,
             1.94, 3.36, 2.40, 2.92, 2.40 s for the twelve files
 CONSTRAINT  compute excludes queue wait; audio seconds > 0
 EXAMPLE     sum compute = 32.50 s ; sum audio = 85.2 s
             RTF = 32.50 / 85.2 = 0.381455
             per-file RTF range 0.375 to 0.385, so the ratio is stable
 INTERPRET   one worker sustains 1 / 0.381455 = 2.6215 audio-hours per hour
```

The per-file stability (a 2.7 percent spread) is itself informative: it says compute scales with audio duration rather than with content, so capacity can be planned from minutes of audio alone. A system whose RTF varied threefold across files would need a per-file cost model instead.

### 5.5 Paired comparison: is the improvement real?

A candidate system B re-decoded the same twelve utterances with the same normalizer. Its error counts per utterance: 1, 0, 2, 1, 1, 2, 1, 0, 1, 0, 2, 3. Totals: `E_B = 14`, `WER_B = 14/126 = 0.111111`.

```text
 FORMULA     delta = WER_A - WER_B, with a paired cluster bootstrap on
             per-utterance error deltas
 WHERE       same utterances, same reference, same normalizer version
 CONSTRAINT  pairing must be exact; a changed normalizer makes delta invalid
 EXAMPLE     delta = (18 - 14)/126 = 4/126 = 0.031746
             bootstrap 95 percent interval [-0.0305, +0.0938]
             share of resamples with delta <= 0: 0.179
             per-utterance: B better on 5, worse on 2, tied on 5
             sign test on 5 wins vs 2 losses: p = 2*29/128 = 0.453
 INTERPRET   a 3.2 point point-estimate improvement whose interval spans
             zero and whose win/loss record is consistent with chance
```

Two independent checks agree that this is not evidence, which is the useful outcome: the interval and the sign test fail for different reasons (magnitude uncertainty versus win-count scarcity), so their agreement is stronger than either alone. The correct action is to enlarge the corpus, and section 11 asks the reader to compute how far.

### 5.6 Independent re-check of every headline number

A second pass recomputes each quantity from the column totals only, without reusing any alignment path:

| Quantity | First computation | Independent re-check | Agree |
|---|---|---|---|
| `sum E` | 1+0+1+2+3+4+1+1+1+0+3+1 | 18 from the table total | yes |
| `sum N` | 13+12+10+11+9+8+11+9+14+8+11+10 | 126 from the table total | yes |
| micro WER | 18/126 | 3/21 = 0.142857 after dividing both by six | yes |
| macro WER | 1.838251/12 | 0.153188, unchanged by recomputation | yes |
| `C` total | 116 | 126 − 7 − 3 | yes |
| `H` total | 131 | 116 + 7 + 8 | yes |
| WIL | 8/124 | 8/(116+8) | yes |
| RTF | 32.50/85.2 | 0.381455 | yes |
| p95 latency | index 10.45 | 690 + 0.45 × 30 = 703.5 | yes |
| slice-weighted micro | 0.14282 | 0.142857 exact with fractions | to rounding |

The last row is the instructive one. Using exact fractions, `(58/126)(8/58) = 8/126 = 0.142857`; using shares rounded to four decimals the same expression returns 0.14282. The discrepancy is 0.00004 and comes entirely from rounding the weights before multiplying. The procedural fix is to carry exact counts into every product and round only at display — a rule that costs nothing and removes a whole category of "the numbers do not add up" review comments.

### 5.7 Sensitivity: which assumption moves the answer most?

Hold everything fixed and move one assumption at a time. The middle column is the recomputed headline; the last is the signed change from the 0.142857 baseline.

| Assumption changed | New headline | Change |
|---|---|---|
| Accept `dollar` for `dollars` (E07 only) | 17/126 = 0.134921 | −0.0079 |
| Accept `m r i` expansion (E05 and E11) | 12/126 = 0.095238 | −0.0476 |
| Drop the worst utterance E06 from numerator and denominator | 14/118 = 0.118644 | −0.0242 |
| Weight slices equally instead of by token count | 0.1565 | +0.0136 |
| Impute all missing tokens as deletions | 108/216 = 0.500000 | +0.3571 |
| Use a binomial rather than clustered interval | half-width 0.0611 | −0.0088 |
| Score at character level instead of word level | see 4.3.2 (0.0364 vs 0.2727) | not comparable |

The largest single lever is coverage, followed by tokenization of an acronym, followed by deleting a hard utterance. That ordering is the practical output of this section: fix the denominator before arguing about spelling, because coverage moves the number by an order of magnitude more. Note also the second row — accepting the `m r i` expansion is not cheating, it is a normalizer decision, but it is worth 0.048 on this corpus, which is three times the paired improvement that section 5.5 failed to establish.

**Takeaway.** Five aggregations of one twelve-row corpus span 0.0769 to 0.50 by slice and 0.0815 to 0.2213 as an interval; publish the contract, not the point.
