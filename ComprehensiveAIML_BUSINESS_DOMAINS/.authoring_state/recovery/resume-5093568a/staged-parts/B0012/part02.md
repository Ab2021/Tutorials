## 6. Analysis Walkthrough

Every number in this section is computed from the `ILLUSTRATIVE` intake defined in section 5a and is
recomputed here rather than quoted from a stored rate. The corpus is 38,000 eligible legs with a
constructed mean duration of 384 seconds, which is 6.4 minutes, so the total audio is
38,000 × 384 / 3,600 = 4,053.3 hours. At a constructed 672 reference words per leg the reference corpus
holds 25,536,000 words; the scored evaluation sample is a 500-leg draw from it, 336,000 reference words,
which is the only corpus on which a word error rate is claimed.

### 6.1 Derivation one: the edit decomposition hides what the rubric cares about

```text
FORMULA [D1]: WER = (S + D + I) / N ;  accuracy = 1 - WER
WHERE:  S = substitutions, D = deletions, I = insertions (token counts, dimensionless integers)
        N = reference words after normalisation (count)
CONSTRAINT: N > 0; identical normalisation on both sides; unit edit cost; one scoring path policy
NUMERICAL EXAMPLE: S = 32,256 ; D = 8,064 ; I = 12,096 ; N = 336,000
        edits = 32,256 + 8,064 + 12,096 = 52,416
        WER = 52,416 / 336,000 = 0.15600 = 15.60%
        decomposition: 9.60% substitution + 2.40% deletion + 3.60% insertion
INTERPRETATION: 15.60% of reference words were edited, and the three components have
        different downstream consequences: deletions lose customer speech, insertions
        invent agent or caller speech, substitutions change one word into another.
```

The decomposition is more useful than the total because the three categories fail differently for an
objection rubric. A deletion removes evidence: the customer said something and the transcript does not
contain it, so no span can be proposed and the leg looks negative. An insertion manufactures evidence:
the recogniser emits a plausible phrase that never existed, a span is proposed on words nobody spoke,
and a reviewer who replays the audio rejects it, consuming review capacity to discover nothing. A
substitution corrupts evidence in place: the span exists, the words are nearly right, and the rubric
may fire on the wrong side of a negation.

With 53.4% of the 15.60% attributable to disfluency handling, the largest single lever is
normalisation rather than acoustics. The
scored sample contains 33,000 filled pauses, false starts, and backchannels. Removing them from both
sides of the alignment removes 28,000 edits:

```text
removed tokens = 336,000 - 303,000 = 33,000
removed edits  =  52,416 -  24,416 = 28,000
share of all edits attributable to disfluency = 28,000 / 52,416 = 0.5342
raw WER         = 52,416 / 336,000 = 15.60%
normalised WER  = 24,416 / 303,000 =  8.06%
```

The interpretation must be stated narrowly, because this is exactly the number a programme is tempted
to publish. The 8.06% figure is correct for the declared normalisation and says nothing about whether
any decisive word survived. It is the rate a reader sees if the report quotes a normalised WER and
omits the normalisation policy, and the omission is what makes the number misleading rather than the
arithmetic.

### 6.2 Derivation two: span precision depends on how undetermined dispositions are counted

Flagged legs produced candidate spans, and each span received one of four dispositions from the
reviewer. The counts on the 4.24%-flag-rate design are:

| Disposition | Count | Meaning for evidence |
|---|---|---|
| accepted | 1,196 | the span is a real objection, role confirmed, replayable |
| rejected | 486 | the span is not an objection, or the audio does not support the words |
| reclassified | 163 | real speech, wrong category; usable for a different rubric, not this one |
| undetermined | 85 | reviewer could not decide; audio unclear, overlap, or rubric ambiguity |
| total | 1,930 | all spans that consumed reviewer time |

```text
FORMULA [D2]: precision_strict = A / (A + R + C + U)
              precision_broad  = (A + C) / (A + R + C + U)
WHERE: A = accepted, R = rejected, C = reclassified, U = undetermined (span counts)
CONSTRAINT: every span carries exactly one disposition; denominator is the full adjudicated set
NUMERICAL EXAMPLE: A = 1,196 ; R = 486 ; C = 163 ; U = 85 ; total = 1,930
        strict = 1,196 / 1,930 = 0.6197 = 61.97%
        broad  = 1,359 / 1,930 = 0.7041 = 70.41%
INTERPRETATION: the choice of numerator is worth 8.44 percentage points, which is larger
        than any model upgrade available at this stage, so the numerator must be declared.
```

The third variant is the one that reveals the policy question. Excluding the 85 undetermined spans
from the denominator gives 1,196 / 1,845 = 64.82%, which is defensible only if undetermined spans occur
at random with respect to being true objections. They do not: undetermined spans are concentrated in
overlapped speech and in transfers, which is exactly where a real objection is hardest to hear and
therefore where the true rate among undetermined spans is plausibly *higher* than among resolved ones.
Excluding them therefore flatters precision. Counting them as negative, which is the strict formula,
is the conservative choice and is the one adopted.

Recall was measured on a separate audit and cannot be recomputed from the same table, because the
denominator is human work rather than model output. In a 150-call audit with 63 hand-found objections,
the model had flagged 47 of them:

```text
recall = 47 / 63 = 0.7460 = 74.60%
F1     = 2 × (0.7041 × 0.7460) / (0.7041 + 0.7460) = 2 × 0.5253 / 1.4501 = 0.7245
```

Recall's denominator is the honest one and its cost is the reason it is measured rarely. Finding 63
objections required listening to 150 calls end to end, which at 6.4 minutes each is 960 listening
minutes, or 16 hours, to establish a recall estimate with a confidence interval wide enough that the
interval must itself be reported. A 63-item denominator cannot distinguish 70% recall from 80% recall.

### 6.3 Derivation three: the rate, the correction, and why the naive rate is wrong

The naive aggregate counts only evidenced objections and divides by all eligible legs. That mixes two
different populations: legs the model looked at and flagged, and legs the model looked at and passed
over. The correction re-weights:

```text
FORMULA [D3]: r̂ = (O_flag + (L - L_flag) × p̂_unflag) / L
WHERE: L = eligible legs (38,000)
       L_flag = legs carrying at least one flagged span (1,612)
       O_flag = legs with at least one accepted objection (1,082)
       p̂_unflag = observed objection rate in a random sample of unflagged legs (16/446)
CONSTRAINT: L_flag > 0; audit sample drawn without replacement from unflagged legs; sample
       representative of unflagged legs; a leg counts once regardless of span count
NUMERICAL EXAMPLE: L - L_flag = 36,388
       p̂_unflag = 16 / 446 = 0.035874
       estimated missed objections = 36,388 × 0.035874 = 1,305.4 legs
       r̂ = (1,082 + 1,305.4) / 38,000 = 2,387.4 / 38,000 = 0.06283 = 6.28%
       naive evidenced-only rate = 1,082 / 38,000 = 0.02847 = 2.85%
INTERPRETATION: the naive rate is 45.3% of the corrected rate, so reporting the evidenced
       count alone understates the phenomenon by a factor of 2.21, and it does so in the
       direction that makes the operation look better.
```

That factor is the case's central quantitative result. It is not a model-quality result at all: it
follows from coverage, not accuracy. Improving span precision from 70.4% to 90% would add 1,930 × 0.196
= 378 accepted spans, which at 1.197 spans per flagged leg is about 316 legs, whereas the unreviewed
population holds an estimated 1,305 objection legs. The programme's information about the phenomenon is
therefore limited primarily by what it never listens to, and any investment ordering that puts model
accuracy ahead of coverage is misallocated.

The interval must reflect two independent uncertainties, because r̂ is a sum of a census-like term and
an estimated term:

```text
sampling component: var(r̂) ≈ p(1-p)/L × deff + ((L - L_flag)/L)² × var(p̂_unflag)
deff = 1.8 (design effect from clustering spans within calls)
var(p̂_unflag) = p̂(1-p̂)/n × fpc,  fpc = (36,388 - 446)/(36,388 - 1) = 0.9878
             = 0.035874 × 0.964126 / 446 × 0.9878 = 7.6522e-5
se(p̂_unflag) = 0.008747 → contribution to r̂ = (0.95758) × 0.008747 = 0.008376
se(census term) = sqrt(0.06283 × 0.93717 / 38,000 × 1.8) = sqrt(2.7899e-6) = 0.001670
se(r̂) = sqrt(0.001670² + 0.008376²) = 0.008541
95% interval = 0.06283 ± 1.96 × 0.008541 = [0.04609, 0.07957] = [4.61%, 7.96%]
```

The audit term dominates: 0.008376 of a total standard error of 0.008541, which is 96.4% of the
variance. Enlarging the unflagged audit from 446 to 1,000 legs would roughly halve that component and
narrow the interval to about ±0.9 percentage points, whereas a better recogniser would barely move it.
This is a direct, computable statement about where the next unit of effort belongs.

### 6.4 Derivation four: reviewer capacity, and why the candidate set must be small

```text
FORMULA [D4]: H_review = n_span × m_span / 60 ;  FTE = H_review / h_month
WHERE: n_span = candidate spans adjudicated (count); m_span = mean minutes per span
       h_month = productive reviewer hours per month (165)
CONSTRAINT: m_span measured on the real console, including replay, rubric lookup and
       reason-code entry; excludes queue wait and training time
NUMERICAL EXAMPLE: 1,930 × 22 / 60 = 707.67 hours ; 707.67 / 165 = 4.29 FTE
       cost per evidenced objection = 707.67 / 1,082 = 0.654 hours = 39.2 minutes
       full-census alternative = 38,000 × 6.4 / 60 = 4,053.3 listening hours = 24.6 FTE
INTERPRETATION: reviewed flagging costs 4.3 FTE against 24.6 FTE for a census, so the
       machine-assisted route is 5.7× cheaper in labour, and each evidenced objection
       costs 39 minutes of human time.
```

The comparison is not "machine versus human" but "sampled human review of machine candidates versus
exhaustive human review". The exhaustive route has known and near-perfect recall and no model error; it
simply costs 5.7 times more and does not scale with volume. The machine route is only better if its
missed objections can be estimated, which is what the 446-leg audit does. Remove the audit and the
machine route has neither exhaustive recall nor a measurable one, which is the failure described in
section 12.

Compute cost is a rounding error against review cost and should be reported as such. At a
constructed real-time factor of 0.02 for recognition and the same again for diarisation, total
processing is 2 × 4,053.3 × 0.02 = 162.1 GPU-hours, against 707.7 reviewer-hours. The ratio is 4.4:1,
which means that any decision to reduce compute by sampling audio before transcription is a false
economy unless it also reduces candidate volume: transcribing more audio and reviewing fewer spans is
strictly better than the reverse.

### 6.5 A boundary where D3 breaks

The correction in D3 assumes the 446-leg audit is representative of the 36,388 unflagged legs. The
formula has no term that can detect a violation of that assumption, and the violation is not exotic.
Suppose the audit was drawn 60% from business hours and 40% out of hours, because reviewers work
business hours, while out-of-hours legs are actually 22% of the unflagged population. Suppose the true
objection rate is 6.00% in daytime and 1.20% out of hours, because out-of-hours callers are renewing
rather than disputing.

```text
audit-weighted p̂      = 0.60 × 0.0600 + 0.40 × 0.0120 = 0.04080
population-weighted p = 0.78 × 0.0600 + 0.22 × 0.0120 = 0.02256
r̂ from audit          = (1,082 + 36,388 × 0.04080) / 38,000 = 2,566.6 / 38,000 = 0.06754
true rate            = (1,082 + 36,388 × 0.02256) / 38,000 = 1,902.9 / 38,000 = 0.05008
bias                  = 0.06754 - 0.05008 = +0.01747, i.e. 1.75 percentage points too high
D3 interval           = [0.05084, 0.08424], which excludes the true 0.05008
```

The bias is slightly larger than the interval half-width of 1.67 points, so the published interval does
not merely fail to be tight, it excludes the truth. This is the failure mode that destroys trust in a
measurement programme: the arithmetic is impeccable, the interval is honestly computed, and the number
is still wrong in a direction that happens to be interesting. Nothing in the data reveals it; only the
sampling design does.

The validity condition of D3, stated plainly, is that the audit is a probability sample of the unflagged
population, stratified on every variable known to predict objections. The discriminating check is a
stratification comparison: compute p̂_unflag separately by hour band, queue, language route and channel,
then compare the observed spread against each cell's binomial standard error. At n = 446 split four
ways, cells of about 110 legs have a binomial standard error of sqrt(0.04 × 0.96 / 110) = 1.87 points,
which cannot resolve a 4.8-point design bias. That is why the check must be a design requirement — the
audit is drawn by stratified probability sampling with fixed cell quotas — rather than a diagnostic run
after the fact.

A second boundary arises at low denominators. A leg with 6 reference words and 3 edits has
WER = 3/6 = 50%, and a corpus of such legs can report an alarming micro WER driven by legs whose entire
content is "yes, speaking". The micro formula is arithmetically correct and substantively useless there,
because the reference denominator carries no information. The guard is a minimum reference length per
included leg, declared in advance, with excluded legs counted in their own bucket.

## 7. The Finding

**The binding constraint on this programme is coverage, not recognition accuracy, and the measured
quantity that exposes it is the ratio between the naive rate and the corrected rate.**

Four results support that sentence and each is falsifiable.

First, span precision of 70.4% at the broad numerator and recall of 74.6% on a 63-item audit is
adequate for population reporting and inadequate for individual casework. That asymmetry is structural:
a population rate tolerates per-span error because errors partially cancel, while a single customer's
dispute does not. The correct conclusion is not "improve precision to 90%" but "keep the evidence route
for population claims and keep humans on individual cases".

Second, the evidenced-only rate of 2.85% and the corrected rate of 6.28% differ by a factor of 2.21,
and the difference is entirely attributable to legs nobody reviewed. Any report published without the
correction is not a conservative estimate of the truth, it is an estimate of a different quantity, and
it will move in the wrong direction whenever review capacity tightens. That coupling between operations
pressure and a published business metric is the most dangerous property the design has, because the
metric degrades precisely when the team is asked to justify itself.

Third, the uncertainty budget is almost entirely in the audit: 96.4% of the variance of the corrected
rate comes from the 446-leg sample of unflagged legs. This inverts the usual investment logic. A model
upgrade that halves edits per word moves the published rate by less than a rounding step, while doubling
the audit sample moves the interval from ±1.7 points to ±0.9 points. The programme's measurement
priority follows from its variance decomposition, not from its tooling roadmap.

Fourth, the cost structure is review-dominated. At 39.2 reviewer-minutes per evidenced objection and
4.4:1 review-to-compute hours, the operating question is span precision per reviewer-minute, not word
error rate. A change that improves WER while adding candidates is a net loss, and a change that adds
candidates at equal precision but concentrates them in under-covered segments is a net gain.

The finding therefore resolves the business question in section 3 to **Answer A with a mandatory
correction term**, conditional on three gates that the evidence in section 9 either passes or fails:
the audit must be a stratified probability sample; the unevidenced share must be published alongside
the rate; and no cell below 250 legs may be published. If the audit cannot be drawn as a probability
sample, the correct output is Answer C, because a corrected rate resting on a convenience sample is
worse than no rate: it carries the authority of arithmetic without the validity of a sample.
