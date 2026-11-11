
## 3. Foundational Definitions

**Reference (ground truth by convention).** A transcript of a declared audio interval produced by a human under a written annotation guide, then adjudicated. It is the best available measurement, not an intrinsic truth. Every reference has a provenance: annotator pool size, guide version, adjudication rule, and disagreement rate.

**Hypothesis.** Text emitted by the candidate system for the same interval. In streaming, a hypothesis may be revised: the partial at 400 ms and the final at 1,900 ms are two different outputs for one audio span and must be scored separately.

**Segment.** The unit pairing audio with text: an utterance, a caller turn, a fixed window, or a whole recording. Segmentation must be identical between reference and hypothesis or the difference measured is segmentation, not recognition. A 40 ms boundary shift can move a word across a boundary and appear as one deletion plus one insertion in two adjacent segments.

**Token and scoring unit.** The unit counted by N. Word-level WER counts whitespace-separated words after normalization; character-level CER counts characters or grapheme clusters; some languages need sub-word or morpheme units because whitespace is not a word delimiter. Changing the unit changes both numerator and denominator and therefore changes the rate non-linearly.

**Substitution (S).** A reference token aligned to a different hypothesis token. Includes near-misses (one letter off) and semantic reversals (a dropped negation appears as a substitution only if the aligner pairs the two positions; often it appears as one deletion plus one insertion, which is why deletion and insertion counts must be read together).

**Deletion (D).** A reference token with no hypothesis token. **Insertion (I).** A hypothesis token with no reference token. Insertions are unbounded above: a system that emits fluent hallucinated text can produce I far exceeding N, giving WER over 100%.

**Reference count (N).** Total normalized reference tokens. For word-level scoring, N = sum over segments of reference word count. N is a *denominator of opportunity*, and the metric's meaning changes if N changes — a normalization policy that collapses "twenty-five" into "25" reduces N and can raise or lower the rate.

**Alignment.** A monotone path through the (n+1)×(m+1) grid of prefix costs. The minimum-cost path gives the minimum edit distance. Individual edit counts are not unique when several minimum-cost paths exist: the total S+D+I is fixed by the distance, but the split between S and I can differ. Report the split with the aligner name and version, or omit the split and report only the distance.

**Normalization policy.** A versioned, symmetric transformation applied to both streams before alignment: Unicode form (NFC/NFKC), case folding, punctuation policy, digit expansion or compression, contraction splitting ("let's" → "let us"), filler-token policy ("um", "uh"), and disfluency policy (repeated word removal). It must be symmetric — the same function applied to reference and hypothesis — and idempotent, or the order of operations becomes a hidden parameter.

**Micro aggregation.** Sum S, D, I and N over segments, then divide. Long segments carry more weight because they contribute more N. This is the natural estimator of per-word error in the population of *words*, and it is what a service-level contract should use when downstream cost is per word.

**Macro aggregation.** Compute a rate per segment or per group, then average with stated weights. Gives each segment or group equal voice. A one-word segment with one error contributes 100% to a macro average, which is usually undesirable but occasionally correct — for example when the contract is per-request fairness rather than per-word cost.

**Multi-reference scoring.** When several valid transcripts exist (paraphrase, number formats, name spellings), the score is the minimum over references. Multi-reference lowers measured WER mechanically and must be declared, because it is not comparable to a single-reference number on the same audio.

**Slice.** A documented, non-exclusive subset: channel (8 kHz telephony versus 16 kHz wideband versus far-field), accent or dialect, age band, speaking rate, noise level, entity class (names, dosages, addresses), consent tier. Slices overlap; a slice rate is descriptive and not automatically causal, because slice membership correlates with other properties.

**Endpointer (VAD).** The component deciding when speech starts and ends. Endpoint decisions are upstream of recognition: a missed endpoint produces a deletion of everything after it, and a late endpoint adds trailing silence that invites hallucination.

**Partial versus final hypothesis.** A partial is interim output emitted while audio arrives; a final is declared complete by protocol. "Final" is a protocol label, not a correctness guarantee — systems do revise after final, and that behavior needs its own counter.

**First-token latency (FTL).** Time from a named audio event (end of the first phoneme, first voiced frame detected, request start) to the client receiving the first eligible token. **Finalization latency (FL).** Time from end-of-speech to client receipt of the final transcript. Both need the event definition, the measurement point (server emit versus client render), and the percentile basis.

**Real-time factor (RTF).** Processing duration divided by audio duration under a stated protocol (which stage, which hardware, warm or cold, batched or online). RTF < 1 is a throughput statement. It is not a latency statement: a batch pipeline with RTF 0.05 can have 40 s latency because it waits for the whole file.

**Coverage.** Valid outputs divided by eligible requests, where the denominator includes crashes, timeouts, abstentions, empty outputs and outputs rejected by validation. A coverage number that excludes failures is a survival rate presented as an accuracy rate.

**Revision rate (flicker).** Fraction of finalized utterances whose displayed partial text changed after first display. Distinct from WER: a system can be accurate at the end and unpleasant to watch.

**Critical-content miss.** A miss on a token class whose failure has non-linear cost — medication dose, negation, patient identifier, monetary amount, address number. Measured per entity class, not per word, and reported as a count of misses so small denominators are visible.

**Takeaway:** Every term here exists because a plausible-looking number becomes meaningless when one of these distinctions is dropped. When a metric is quoted without them, assume the weakest reading.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: word error rate on one segment

```text
FORMULA
  WER = (S + D + I) / N

WHERE
  S = substitutions, D = deletions, I = insertions, all nonnegative integers
  from a minimum-cost alignment of reference tokens R (length N) to
  hypothesis tokens H (length M)
  N = |R| >= 1, an integer count of normalized reference tokens
  WER is dimensionless: edits per reference token

CONSTRAINT
  valid only if R and H are produced by the SAME frozen normalization policy,
  the segment boundaries agree, and N > 0. If N = 0 the ratio is undefined and
  the segment must be excluded or handled by a declared rule, never silently
  dropped.

NUMERICAL EXAMPLE
  R = "my name is katherine o'brien"      N = 5
  H = "my name is kathryn obrien"         M = 4
  aligner pairs katherine->kathryn (1 substitution), deletes o'brien? No:
  the apostrophe token is part of "o'brien", so the aligner sees
  "o'brien" vs "obrien" -> 1 substitution.
  S = 2, D = 0, I = 0
  WER = (2 + 0 + 0) / 5 = 0.40

INTERPRETATION
  Two of five reference words differ. The rate is 0.40, not 40% "wrong
  understanding": the name was still recognizable to a human reader.
```

The identity S + D + I = edit distance (Levenshtein distance with unit costs) is the reason the formula is a ratio rather than a model. It also fixes the numerator's scale: every edit costs exactly one token. Weighted variants exist (Section 4.3) but change the unit from tokens to arbitrary cost points, which forfeits comparability with published numbers.

### 4.2 General case: corpus aggregation and what the denominator chooses

```text
FORMULA (micro)
  WER_micro = (sum_k S_k + sum_k D_k + sum_k I_k) / (sum_k N_k)

FORMULA (macro, unweighted mean of segment rates)
  WER_macro = (1/K) * sum_k [ (S_k + D_k + I_k) / N_k ]

WHERE
  k indexes segments 1..K, each with N_k >= 1
  micro weights by N_k implicitly; macro weights every segment equally

CONSTRAINT
  WER_macro >= WER_micro is NOT a general identity. Jensen: the rate function
  e/N is convex in N for fixed error count only in specific regimes; in general
  the two can order either way. The safe statement is by example, not by
  theorem.

NUMERICAL EXAMPLE (ILLUSTRATIVE, arithmetic verified)
  Segment 1: N=7  errors=0
  Segment 2: N=5  errors=2
  Segment 3: N=8  errors=1
  Segment 4: N=9  errors=1
  Segment 5: N=7  errors=1
  sum N = 36, sum errors = 5
  micro = 5/36 = 0.138889
  macro = (0/7 + 2/5 + 1/8 + 1/9 + 1/7)/5
        = (0 + 0.4000 + 0.1250 + 0.1111 + 0.1429)/5
        = 0.7790/5 = 0.155794
  macro - micro = +0.0169

INTERPRETATION
  Here macro exceeds micro by 1.7 points because the worst segment is short.
  Dropping segment 2 (the short bad one) moves micro to 3/31 = 0.0968 and
  macro to 0.0948 — the same data, two defensible aggregation rules, a
  4-point spread.
```

**Boundary / counterexample.** Set N_k = 1 for all K segments with exactly one error in each: micro = 1.0 and macro = 1.0. Now let one segment have N = 1 with one error and another have N = 100 with one error: micro = 2/101 = 0.0198, macro = (1.0 + 0.01)/2 = 0.505. The gap is 48 points on identical underlying behaviour. Any comparison across corpora with different length distributions is confounded unless aggregation is held fixed and N is reported.

### 4.3 Variants: weighted, entity-level and multi-reference

```text
FORMULA (weighted error rate)
  WER_w = sum_k sum_j c(t_kj) * edit(t_kj)  /  sum_k sum_j c(t_kj) * [token is reference]

WHERE
  c(.) >= 0 is a declared cost per token class (e.g. name 3.0, digit 2.0,
  filler 0.0), applied to reference tokens; denominator uses the same weights
  so the ratio stays dimensionless

CONSTRAINT
  weights must be fixed before looking at results; otherwise the weight vector
  is fitted to the outcome. Weighted rates are not comparable to unweighted
  published numbers, and the denominator change means a system can "improve"
  by shifting errors from heavy to light classes without changing distance.

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  Reference tokens: 10 of class filler (c=0), 6 of class digit (c=2), 4 name (c=3)
  Weighted reference mass = 0*10 + 2*6 + 3*4 = 24
  System A edits: 3 filler deleted, 1 digit substituted, 0 names
    weighted errors = 0*3 + 2*1 + 3*0 = 2  -> 2/24 = 0.0833
  System B edits: 0 fillers, 3 digits, 1 name
    weighted errors = 0 + 6 + 3 = 9  -> 9/24 = 0.3750
  Unweighted: A = 4/20 = 0.20, B = 4/20 = 0.20  (identical)

INTERPRETATION
  The unweighted rate says the systems are equal; the weighted rate says A is
  4.5x better for this contract. Neither is wrong. The weighted version is the
  right one only if the cost vector is defensible independently of this result.
```

```text
FORMULA (entity-level critical miss rate)
  CMR = (number of required entities with at least one error) / (number of required entities)

WHERE
  an entity is a declared span (name, dose, MRN, amount, address number) whose
  reference occurrence is extracted by a frozen rule; a hit requires the
  hypothesis to contain the entity's canonical value after normalization

CONSTRAINT
  entity counts are small; 9 misses out of 42 entities is 21.4% with a Wilson
  95% interval of 11.7%–35.9%. Never quote a bare percentage from a
  denominator under ~50 without the interval.

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  Required entities: 42. Correct after normalization: 33. Missed: 9.
  CMR = 9/42 = 0.2143
  Wilson 95%: p_hat = 0.2143, z = 1.96, n = 42
    denom  = 1 + z^2/n = 1 + 3.8416/42 = 1.09147
    centre = (p_hat + z^2/(2n)) / denom = (0.2143 + 0.04573)/1.09147 = 0.2382
    half   = z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2)) / denom
           = 1.96*sqrt(0.004009 + 0.000544)/1.09147
           = 1.96*0.06747/1.09147 = 0.12117
    interval = 0.2382 ± 0.1212 -> [0.1171, 0.3594]

INTERPRETATION
  The point estimate is compatible with anything from 1 in 9 to 1 in 3. A
  decision to block on "more than 15% CMR" is at the boundary of what 42
  entities can resolve; the fix is more entities, not a finer threshold.
```

```text
FORMULA (multi-reference score)
  WER_multi = min over references r in R of [ dist(H, r) / N_r ]

WHERE
  R is a set of human-accepted transcripts, possibly with different lengths
  N_r; the minimum is taken over the whole corpus or per segment (declare which)

CONSTRAINT
  per-segment minimums do not sum to a corpus minimum; per-segment min then
  micro-aggregate is the usual choice, but it is biased downward relative to
  single-reference by construction and must be reported as such

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  Reference set for one segment: r1 = "route four two one" (N=4),
  r2 = "route 421" (N=2)
  H = "route 421"  -> dist to r1 = 3 (sub + 2 del? no: "421" vs "four two one"
    aligns as 3 words vs 1, cost 2 deletes + 1 sub = 3), N1 = 4, rate 0.75
  dist to r2 = 0, N2 = 2, rate 0.0
  min = 0.0

INTERPRETATION
  The system is perfect against the reference that already agreed with it. If
  r2 was written by the system vendor, this segment measures agreement with the
  vendor, not accuracy. Provenance of each reference is part of the score.
```
