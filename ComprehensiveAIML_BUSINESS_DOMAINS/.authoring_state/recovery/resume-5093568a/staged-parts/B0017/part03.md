
### 4.4 Limits and counterexamples

**Limit 1 — N → 0.** As the reference count of a segment shrinks, the per-segment rate explodes. A segment with N = 1 and one error scores 1.00 regardless of how the other 99 words went. Consequences: (a) never micro-average over segments without reporting the length distribution; (b) never let a 1-word segment into a slice without noting that its rate is bounded by its N.

**Limit 2 — insertions are unbounded, so WER has no ceiling.** A hypothesis that repeats a phrase ten times yields I ≈ 10N and WER > 10. Any monitoring rule of the form "alert if WER > 100%" is therefore wrong in the same way that "alert if accuracy < 0" is wrong — it encodes a belief about bounds that the metric does not have. The real failure is a hallucinating or looping decoder, and the diagnostic is the I/N ratio, not the level.

**Limit 3 — alignment cost ignores token importance.** Set two hypotheses for "take two tablets daily": (a) "take too tablets daily", (b) "take two tablets weakly". Both cost 1 substitution, WER 0.25. For a medication instruction, (b) is a safety event and (a) is a spelling nuisance. Unit-cost edit distance cannot distinguish them; a weighted or entity-level metric must be layered on top.

**Limit 4 — normalization can be gamed, intentionally or not.** Because normalization changes N and the edit set, a policy change can move WER without any model change. Compressing spellings to digits ("twenty five" → "25") reduces N by one and replaces a 0-cost match with either a 0-cost match (if the hypothesis also says "25") or a substitution. On a corpus where the system prefers digit output, this lowers WER; on a corpus where references use words and the system uses words, it can raise WER if the number-format canonicalizer disagrees. The policy version belongs in the metric name.

**Limit 5 — CER on short strings is unstable in both directions.** A 3-character reference with 1 substitution is CER 0.333; the same error rate on a 300-character paragraph is invisible. Conversely, CER can exceed 1.0 whenever insertions exceed reference length: reference "COVID-19" (8 characters) against hypothesis "covid neunzehn" (14 characters) gives 8 substitutions and 6 insertions = 14 edits over N=8, CER = 1.75. Report such cases as counts, not as a percentage, or the reader will assume a bound that does not exist.

**Limit 6 — the metric is not comparable across languages or scripts.** Character error rate is meaningful in scripts with bounded grapheme inventory and roughly constant information per character; in languages where a single character carries a morpheme, or where the writing system differs between reference and system output (traditional versus simplified), CER differences reflect the unit, not the recognizer. Cross-language comparison is a category error unless the unit is defined by information content, which no standard ASR metric does.

**Boundary where the formula breaks entirely.** If reference and hypothesis are not segment-aligned — for example the reference has utterance boundaries at pauses and the system emits one continuous stream for the whole call — then per-segment scoring measures segmentation. The correct move is to score the concatenation (sequence-level WER over the whole call) and score segmentation separately with boundary F1, or to re-segment both streams with the same forced aligner. Reporting both numbers without saying which is which produces irreconcilable team disagreements, which is the single most common cause of "two teams report different WER for the same model".

## 5. Worked Numerical Example

All corpora, counts and latencies in this section are **ILLUSTRATIVE**: constructed to exercise the arithmetic, with every division re-computed. Nothing here is a measurement of any system.

### 5.1 One segment, full alignment

Reference (5 words): `schedule the meeting for tuesday`
Hypothesis (4 words): `schedule a meeting tuesday`

Cost grid by prefix lengths (rows = reference prefix length i, columns = hypothesis prefix length j), unit costs, match cost 0:

```text
        ""   sched  a   meet  tues
  ""     0     1     2    3     4
  sched  1     0     1    2     3
  the    2     1     1    2     3
  meet   3     2     2    1     2
  for    4     3     3    2     2
  tues   5     4     4    3     2
```

Reading the bottom-right cell: distance 2 = 1 substitution + 1 deletion. Backtracking from (5,4): `tuesday` matches, `for` deletes, `meeting` matches, `the → a` substitutes, `schedule` matches. Check the arithmetic independently: |N − M| = |5 − 4| = 1, so at least one insertion or deletion is required; total edits 2 means exactly one additional substitution, which the backtrack found. WER = (1 + 1 + 0)/5 = 0.40.

Note the alternative path: aligning `the → a` as delete-then-insert would cost 2 instead of 1 and is therefore not on any minimum path. This is why insertion and deletion counts are aligner-dependent in tie cases and why the total is safe to report while the split is not.

### 5.2 The corpus, counted from the utterance table

ILLUSTRATIVE evaluation set v4: 480 utterances, 12,000 normalized reference words, 720 total word edits (S = 432, D = 144, I = 144, so the S/D/I split is 60/20/20).

```text
FORMULA
  WER_micro = (S + D + I) / N

WHERE
  S = 432, D = 144, I = 144, N = 12,000 reference words
  counts sum over 480 utterances; all streams normalized by policy v3

CONSTRAINT
  the five slice denominators must sum to N and the numerators to the total
  edit count, or the table has a double count or a dropped segment

NUMERICAL EXAMPLE
  numerator  = 432 + 144 + 144 = 720
  denominator = 3500 + 3000 + 2900 + 1800 + 800 = 12,000  (checks)
  WER_micro  = 720 / 12,000 = 0.0600  -> 6.00%

INTERPRETATION
  6.00% of reference words required an edit. Per 1,000 words the service
  costs about 60 edits; at an average of 25 words per utterance that is
  1.5 edits per utterance, but the distribution is far from uniform (§5.3).
```

Slice table (all ILLUSTRATIVE, counts verified to sum):

| Slice | Utterances | Ref words | Edits | WER | Share of N |
|---|---|---|---|---|---|
| studio close-talk 48 kHz | 140 | 3,500 | 84 | 2.40% | 29.2% |
| mobile handset wideband | 120 | 3,000 | 180 | 6.00% | 25.0% |
| contact centre 8 kHz | 110 | 2,900 | 232 | 8.00% | 24.2% |
| far-field meeting room | 70 | 1,800 | 126 | 7.00% | 15.0% |
| voicemail / IVR telephony | 40 | 800 | 98 | 12.25% | 6.7% |
| **total** | **480** | **12,000** | **720** | **6.00%** | **100%** |

The aggregate 6.00% is exactly 14.3% below the unweighted mean of the five slice rates (2.40 + 6.00 + 8.00 + 7.00 + 12.25)/5 = 7.13%. Check: 7.13% × (1 − 0.158) ≈ 6.00%. The gap is composition: 54.2% of the words come from the two cleanest slices. Any change to slice proportions changes the aggregate with zero change in per-slice quality — which is the mechanism behind most "the new model is better" claims that fail in production.

### 5.3 Aggregation and the length distribution

The 480 utterances average 25.0 words (12,000/480); the shortest has 4 words, the longest 39. Errors are concentrated: 271 of 480 utterances (56.5%) have zero errors, while the worst 48 utterances (top decile) carry 475 of 720 edits (66.0%). The maximum in a single utterance is 15 edits.

```text
FORMULA (macro on the same data, equal utterance weight)
  WER_macro = (1/K) * sum_k (e_k / N_k),  K = 480

WHERE
  e_k is the edit count of utterance k, N_k its reference word count
  both from the same frozen set; the mean is unweighted

CONSTRAINT
  utterances with N_k = 0 are excluded by rule, not silently; if any exist the
  macro denominator changes and must be stated

NUMERICAL EXAMPLE
  sum_k e_k = 720, sum_k N_k = 12,000, K = 480
  The unweighted mean of per-utterance rates is (1/K) * sum_k e_k/N_k =
  0.063707 (6.371%), i.e. 0.371 points ABOVE the 6.000% micro rate.
  It is NOT 720/12,000 unless every N_k were equal — here they range 4..39.

INTERPRETATION
  Macro is higher because short utterances are disproportionately error-prone:
  a 6-word utterance with 3 errors contributes 0.50 to the average while
  adding only 6 to N. Reporting only the micro rate hides that one in seven
  utterances is badly broken. The gap is modest here (0.37 pp) only because
  56% of utterances have zero errors, which pulls the mean back down; on a
  corpus with fewer perfect utterances the same length skew produces a gap of
  several points.
```

Per-utterance accuracy is therefore bimodal in a way the aggregate hides: 271 of 480 utterances (56.5%) have zero errors, the median utterance rate is 0.00%, the 90th percentile is 13.3% and the 95th is 41.2% with a maximum of 88.2%. A product decision that depends on "can the user read this transcript" is better served by an utterance-level success rate (fraction of utterances with WER ≤ 0.10) than by a corpus average. That success rate for this ILLUSTRATIVE set is 0.856 — 411 of 480 utterances at or below 10% — while the corpus average sits at 6.00%. The 69 utterances above 10% hold only 12.6% of the reference words, which is exactly why micro-averaging reports a comfortable 6% while 14% of user sessions see a transcript with more than one word in ten wrong.

### 5.4 Uncertainty: the naive interval is wrong by a factor of 7 in variance

```text
FORMULA (naive binomial standard error, treats words as independent trials)
  se_naive = sqrt( p*(1-p) / N )

WHERE
  p = 720/12000 = 0.06, N = 12,000 reference words
  assumes each word is an independent Bernoulli trial with the same error
  probability — an assumption contradicted by §5.3

CONSTRAINT
  valid only if errors are independent across words; within-utterance
  correlation inflates the true variance by the design effect

NUMERICAL EXAMPLE
  se_naive = sqrt(0.06*0.94/12000) = sqrt(4.700e-6) = 0.002168
  95% interval = 0.06 ± 1.96*0.002168 = [0.0558, 0.0642]
  implied width: 0.85 percentage points

INTERPRETATION
  This interval is too narrow. A 12,000-word corpus does not contain 12,000
  independent observations when the utterances cluster.
```

```text
FORMULA (cluster-robust standard error for a ratio, delta method, utterance clusters)
  se_cluster = sqrt( (n/(n-1)) * sum_i (e_i - p*N_i)^2 / N^2 )

WHERE
  i indexes utterances, n = 480 clusters, e_i edits in utterance i,
  N_i its reference words, N = sum N_i, p = sum e_i / N
  the residual (e_i - p*N_i) is the influence of cluster i on the numerator

CONSTRAINT
  needs n >= ~30 clusters; with fewer, use the bootstrap or a t-multiplier
  with n-1 degrees of freedom. Clusters must be the unit of independent
  sampling (utterance, speaker, or call — declare which).

NUMERICAL EXAMPLE
  the per-utterance residuals e_i - 0.06*N_i must be read off the table;
  for this corpus their sum of squares is 4,814.46 (mean 10.03 per utterance,
  in count^2 units). Note the residuals sum to zero by construction:
    sum_i e_i = 720 and sum_i 0.06*N_i = 0.06*12,000 = 720.
  se_cluster = sqrt( (480/479) * 4,814.46 / 12,000^2 )
             = sqrt( 1.002088 * 4,814.46 / 1.44e8 )
             = sqrt( 3.3506e-5 )
             = 0.00578822
  Cross-check by the design-effect route: deff = 7.1284 and se_naive = 0.00216795,
  so se = 0.00216795 * sqrt(7.1284) = 0.00216795 * 2.66993 = 0.00578822. Agrees.

INTERPRETATION
  The honest 95% interval is 0.06 ± 1.96*0.00578822 = [0.04866, 0.07134], about
  2.67x wider than the naive one and 7.13x larger in variance. A 6.00% score and
  a 5.50% competitor are indistinguishable on this set; a 6.00% score and a
  4.50% competitor are not.
```

**Sensitivity note.** The design effect is not a property of the metric; it is a property of the corpus. Rebuilding the same totals (12,000 words, 720 edits) with errors spread evenly — every utterance carrying its proportional share — yields se_cluster = 0.000596 and deff = 0.076, an interval of [5.88%, 6.12%] that is *narrower* than naive. Same N, same WER, same 480 utterances; the interval width changes by a factor of 9.7. Therefore: the interval must be computed from the observed per-utterance distribution, and a design effect borrowed from another corpus is worthless.

### 5.5 Character error rate on the same audio

Character-level scoring uses the same formula with characters as tokens. Three ILLUSTRATIVE pairs, aligned at character level with the same unit costs:

```text
pair A  ref "Straße 17, München"  (18 chars)   hyp "strasse siebzehn münchen" (24 chars)
        S=6 D=0 I=6  edits=12   CER = 12/18 = 0.6667
pair B  ref "COVID-19" (8)        hyp "covid neunzehn" (14)
        S=8 D=0 I=6  edits=14   CER = 14/8  = 1.7500
pair C  ref "Let's meet at 3pm, OK?" (22)  hyp "lets meet at three pm okay" (26)
        S=7 D=1 I=5  edits=13   CER = 13/22 = 0.5909
```

Pair A shows the normalization trap: with case folding, `Straße`→`strasse` still differs by the ß expansion (2 characters), and `17` versus `siebzehn` is a full lexical rewrite that character distance scores as 8 edits. Under a digit-normalizing policy the same pair would score near zero. Pair B is the >1.0 case; the number is arithmetically correct and rhetorically useless. Pair C, after a policy that expands contractions on the reference side and compresses numbers on both sides, becomes ref "let us meet at 3pm ok" (19 chars) against the same hypothesis: S=1, D=0, I=9, edits=10, CER = 10/19 = 0.5263. The policy change alone moved CER by 6.5 points with no model change — the single strongest argument for pinning the policy version into the metric name.

Independent re-check of pair C after normalization: 19 reference characters, 28 hypothesis characters, distance 10. Lower bound from length difference |28 − 19| = 9; distance 10 means exactly one substitution beyond the length gap, consistent with 1 substitution and 9 insertions. The counts are internally consistent.

### 5.6 Streaming latency: which percentile, on which sample

ILLUSTRATIVE 1,200 finalized segments from one streaming deployment, finalization latency measured server-emit minus endpointer-declared speech end:

```text
p50 = 256.4 ms   p80 = 363.7 ms   p85 = 417.4 ms   p88 = 1059.3 ms
p90 = 1088.9 ms  p95 = 1158.4 ms  p99 = 1253.9 ms  mean = 368.3 ms
fraction above 1000 ms = 0.1308
```

The distribution is bimodal: 87% of segments are a nominal cluster with p85 at 417 ms, and 13% carry an injected stall of about +900 ms. The p95 therefore lands inside the stall mode, and the mean (368 ms) sits near the p80 of the mixture — which is why "average latency 370 ms" and "95th percentile 1,158 ms" can both be true and neither describes the user experience.

Consequences that follow arithmetically:

1. **The p95 is fragile to the stall fraction.** With stall weight w, the p95 falls in the stall mode whenever w > 0.05, i.e. more than 5% of segments stall. Below 5%, the p95 is a nominal-mode value near 380 ms. A service whose stall rate crosses 5% shows a step change of roughly +700 ms in p95 with no gradual degradation to warn anyone.
2. **The sample size limits the resolution at the tail.** The standard error of an estimated 0.95 quantile is sqrt(0.95*0.05/1200) = 0.00629 in probability units, so the plausible quantile level ranges from 0.9377 to 0.9623. At n = 1,200 those correspond to order statistics 1,124 and 1,154, i.e. the 30 observations between them. Quoting p95 to the millisecond from 1,200 samples claims a precision the tail cannot support; quote the interval [1,135.6 ms, 1,183.8 ms] or round to 1.16 s.
3. **Per-window p95 is not a trend test.** Split the same 1,200 segments into three consecutive 400-segment windows: p95 = 1,161.8, 1,143.7, 1,159.4 ms. The apparent 18 ms dip in the middle window is order-statistic noise, not an improvement. A rollout gate that compares p95 across daily windows with n ≈ 400 per window will flip on noise alone.
4. **Server-emit versus client-render differs by a constant plus a distribution.** If render adds a median 40 ms with a heavy right tail (device-dependent), server p95 1,158 ms can be client p95 well above 1.3 s. The measurement point must be named in the SLO string, not in a footnote.

Coverage arithmetic for the same ILLUSTRATIVE window: 12,000 eligible requests, 11,712 valid outputs, 288 failures (timeouts, empty outputs, rejected by validation).

```text
FORMULA
  coverage = valid / eligible = 11,712 / 12,000 = 0.9760
  Wilson 95%: denom = 1 + 3.8416/12000 = 1.00032
              centre = (0.9760 + 3.8416/24000)/1.00032 = 0.97632
              half   = 1.96*sqrt(0.9760*0.0240/12000 + 3.8416/(4*12000^2))/1.00032
                     = 1.96*sqrt(1.952e-6 + 6.671e-9)/1.00032 = 0.002741
              interval = [0.9736, 0.9791]

WHERE
  eligible = every request the service accepted, including ones that failed
  valid    = output that passed length, encoding and sanity validation

CONSTRAINT
  failures must be attributed to a cause; a 97.6% coverage number computed
  after excluding the 288 failures is 100% by construction and meaningless

NUMERICAL EXAMPLE
  If the 288 failures were scored as full-error segments with their expected
  ~25 words each, the added numerator would be 288*25 = 7,200 edits against an
  added 7,200 reference words: a second corpus with WER 1.0. Blended into the
  main corpus the rate becomes (720 + 7,200)/(12,000 + 7,200) = 7,920/19,200
  = 0.4125 — a 41% error rate on the same system that reported 6.00%.

INTERPRETATION
  Coverage does not sit beside accuracy; it decides whether accuracy is defined
  on the population you care about. This is the arithmetic behind "our ASR is
  94% accurate" claims that collapse when the failure rate is included.
```

### 5.7 Paired comparison, and why pairing buys precision

Candidate B is scored on the same 480 ILLUSTRATIVE utterances with the same normalization policy. Totals: 625 edits, WER 5.21%.

```text
FORMULA (paired difference of ratios, cluster-robust)
  delta = p_B - p_A = (E_B - E_A) / N
  se_delta = sqrt( (n/(n-1)) * sum_i [ (e_Bi - e_Ai) - delta*N_i ]^2 / N^2 )

WHERE
  e_Bi - e_Ai is the per-utterance change in edit count (integer, may be
  negative); delta is the corpus-level change in rate; N and n as before

CONSTRAINT
  both systems must be scored on identical utterances with identical
  normalization; if either differs, the paired variance formula is invalid and
  an unpaired comparison is required (costing ~7x the sample for the same
  power, computed below)

NUMERICAL EXAMPLE
  E_A = 720, E_B = 625, N = 12,000
  delta = (625 - 720)/12,000 = -95/12,000 = -0.007917  -> -0.79 pp
  se_delta = 0.0011452  (from the per-utterance difference residuals)
  z = -0.007917/0.0011452 = -6.91
  95% interval = -0.007917 ± 1.96*0.0011452 = [-0.010161, -0.005672]
  i.e. [-1.02 pp, -0.57 pp]

INTERPRETATION
  The improvement is real on this set: the interval excludes zero by a wide
  margin and the upper bound is still an improvement of more than half a point.
  Note the paired se (0.001145) is 5.1x smaller than the unpaired cluster se of
  system A alone (0.005788) — pairing removes the between-utterance variance
  that both systems share.
```

The per-slice paired results are where the decision actually gets made:

| Slice | A WER | B WER | Δ (pp) | 95% CI on Δ | Verdict |
|---|---|---|---|---|---|
| studio close-talk | 2.40% | 2.63% | +0.23 | [+0.07, +0.38] | regression, significant |
| mobile wideband | 6.00% | 5.03% | −0.97 | [−1.42, −0.51] | improvement |
| contact centre 8 kHz | 8.00% | 6.45% | −1.55 | [−2.15, −0.95] | improvement |
| far-field meeting | 7.00% | 6.06% | −0.94 | [−1.49, −0.40] | improvement |
| voicemail / IVR | 12.25% | 10.75% | −1.50 | [−2.49, −0.51] | improvement |

The aggregate gain is −0.79 pp and every large slice improves, but the cleanest slice regresses by 0.23 pp with an interval excluding zero. That is a genuine trade: if studio recordings feed a broadcast captioning product that has an accuracy SLA, B is not a drop-in upgrade for it. If studio is only a monitoring reference channel, the trade is free. The number does not make the decision; the decision rule does — and it must be written before the table exists.

**Power check, same data.** With the observed paired se of 0.0011452, the minimum detectable difference at 80% power and two-sided α = 0.05 is (1.96 + 0.8416) × 0.0011452 = 0.003208, i.e. 0.32 pp. Unpaired at the same N, the MDE is (1.96 + 0.8416) × sqrt(2 × 0.06 × 0.94 / 12,000) = 2.8016 × 0.003067 = 0.008589, i.e. 0.86 pp. To resolve a 0.5 pp difference unpaired at this error base you would need n = 2.8016² × 2 × 0.0564 / 0.005² = 35,414 reference words per arm — 2.95x the corpus. This is the quantitative case for paired evaluation: same data, 2.7x finer resolution.
