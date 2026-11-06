---
collection: module-18-cheat-sheets
file_id: "03"
topic: Speech automatic transcription evaluation
level: 3
status: draft-unreviewed
version: "1.0"
created: 2026-11-07
reviewed: null
review_trigger: Change to target population, normalization, latency contract, or source material
notation_style: plain-text
prerequisites: Basic probability, ratios, text tokenization, and digital audio timing
pattern_tags: [asr, evaluation, streaming, privacy]
source_status: OPEN
owner_role: Speech evaluation curriculum maintainer
source_index: INDEX.md
---
# Speech automatic transcription evaluation

## 1. File Header

**Purpose:** Choose and diagnose an evaluation for batch or streaming automatic speech recognition (ASR), also called speech-to-text (STT), without mistaking a low aggregate word error rate for a safe, useful service.
**Audience:** Speech engineers, evaluators, product owners, and interview candidates who can read simple algorithms and rates.
**Outcome:** Given aligned transcript pairs, timestamps, deployment constraints, and slice metadata, compute and interpret error and latency; identify uncertainty, privacy risks, and promotion blockers.
**Prerequisites:** Tokenization, ratios, event timing, and sampling; the module [index](../INDEX.md) names course prerequisites. No unauthored lesson is a mandatory dependency.
**Not covered:** Training a speech recognizer, speech synthesis quality, speaker verification, or legal advice. Separate diarization and downstream language-model evaluation are mentioned only as handoffs.
**Evidence:** DERIVED for definitions and calculations; ILLUSTRATIVE for examples and thresholds; OPEN for source coverage and all operational results. No deployment or benchmark result is asserted.
**Retrieval cue:** For a surprising score go to Sections 4–5; for a production incident go to Section 8; for a launch decision go to Section 7.
**Source limitation:** The source crosswalk maps the directory entries 20.11 to speech production and 20.4 to speech ASR. The two source files themselves were not found in the accessible tree. Their names establish scope only, not their contents or any benchmark claim. [Directory](../../../../Templates/directory.txt); [crosswalk](../../../00_SOURCE_TOPIC_CROSSWALK.md).
**Publication state:** Draft, not independently reviewed. All population sizes, timings, thresholds, texts, and operational scenarios below are teaching assumptions, not observations.
**Takeaway:** The evaluation contract, not one score, defines what a claim means.

## 2. Concept Map / Mental Model

```text
consented audio + reference + slice metadata
   | -> split by speaker/session/time -> frozen evaluation set
   | -> identical declared text normalization -> alignment
   | -> aggregate edits and references -> WER / CER
   | -> separately evaluate names, numbers, silence and slices
   | -> streaming event timestamps -> partial/final latency and stability
   | -> privacy controls and failure accounting -> release decision
   v
paired comparison + uncertainty + operational guardrails + human review
```
Audio is the input evidence; the reference transcript is a fallible annotation of that evidence.
The recognizer produces hypotheses, possibly several revisions per utterance.
Text normalization decides what differences are scored.
Edit alignment describes lexical disagreement rather than understanding or clinical correctness.
Streaming measurements add user-visible time and revision behavior, which offline accuracy does not capture.
Slices expose heterogeneous performance; their counts and overlap must accompany their rates.
A decision combines quality, latency, reliability, accessibility, and privacy; none is automatically a substitute for another.
A held-out score describes the sampled evaluation distribution, not every future caller.
**Takeaway:** Trace the chain from audio to a bounded decision before comparing models.

## 3. Foundational Definitions

**Reference:** Human-authored transcription of an audio segment, produced under a written convention. An adjudicated reference is still not an infallible ground truth.
**Hypothesis:** Text emitted by a candidate ASR system for the same audio. Streaming hypotheses can change as new audio arrives.
**Segment:** A declared audio interval paired with text; boundaries must be consistent between reference and candidate or explicitly reconciled.
**Token:** The scoring unit after normalization. A word, Unicode code point, grapheme cluster, or other language-specific unit can each produce a different metric.
**Substitution S:** One reference token replaced by a different hypothesis token in a minimum-cost alignment.
**Deletion D:** One reference token aligned with no hypothesis token.
**Insertion I:** One hypothesis token aligned with no reference token.
**Reference count N:** Number of normalized reference units, a nonnegative integer. Word error rate (WER) uses reference words; character error rate (CER) uses declared character units.
**Alignment:** A minimum-cost path of matches and edits through a dynamic-programming grid. Equal-cost paths can have different individual edit counts, even when total distance is identical.
**Normalization:** Versioned conversion applied identically to reference and hypothesis before scoring, such as Unicode normalization, case policy, punctuation policy, and number handling.
**Micro aggregation:** Sum errors and reference units over segments, then divide; long segments contribute more reference units.
**Macro aggregation:** Average a stated set of per-segment or per-slice rates; define how empty references and group weights are treated.
**Slice:** A documented subset such as language, channel, consent tier, or dialect, possibly overlapping other slices. A slice rate is not automatically a causal effect.
**Endpointer:** Component that declares speech or utterance completion; endpoint delay and recognition delay are different.
**Partial hypothesis:** An interim transcript emitted before finalization. **Final hypothesis:** An output designated final according to a specified protocol, not necessarily intrinsically correct.
**First-token latency:** Time from a declared audio or speech event to receipt of the first eligible text token; the start event must be named.
**Finalization latency:** Time between a defined end-of-speech event and the client receiving final transcript. The speech-end label itself has uncertainty.
**Real-time factor (RTF):** Processing duration divided by audio duration under a specified measurement protocol; it is not equivalent to interactive latency or capacity.
**Coverage:** Proportion of eligible requests with valid output, explicitly including failed, timed-out, and dropped requests in its denominator.
**Takeaway:** Units, segmentation, populations, and clocks must be specified before a percentage can be interpreted.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Validity: a nonempty reference and hypothesis are tokenized with the same frozen policy; edit operations cost one and matches cost zero. The shortest edit path makes the fewest unit edits needed to transform reference into hypothesis. Dividing by the original number of reference units expresses edits per reference unit.

```text
FORMULA [E1]: WER = (S + D + I) / N
WHERE: S, D, I are word edit counts; N is reference word count.
CONSTRAINT: N > 0; same normalization and segmentation; unit edit costs.
NUMERICAL EXAMPLE: S=1, D=1, I=0, N=8 -> WER=2/8=0.25=25%.
INTERPRETATION: Two minimum edits per eight reference words, not 25% of words necessarily wrong in every utterance.
```

DERIVED: A substitution consumes one reference and one hypothesis word; a deletion consumes one reference word; an insertion consumes one hypothesis word. Correct matches consume one of each. No ratio is defined by E1 if the reference is empty; preserve the segment for separate insertion reporting.
CER uses the same derivation with character units in all four counts. Decide whether a character is a code point or a grapheme cluster; a visible accented letter can comprise multiple code points. A byte count is not a character count.

### 4.2 General Case

Validity: reference and hypothesis are sequences of lengths m and n; ordinary Levenshtein edits are independent unit costs. Let C(i,j) be the smallest number of edits converting the first i reference tokens to the first j hypothesis tokens. Empty prefixes require i deletions or j insertions. For nonempty prefixes every possible final action is a deletion, insertion, or match/substitution. Taking the smallest predecessor cost is therefore optimal by induction.

```text
FORMULA [E2]: C(0,j)=j; C(i,0)=i
C(i,j)=min(C(i-1,j)+1, C(i,j-1)+1,
           C(i-1,j-1)+[r_i != h_j])
WHERE: i=1..m, j=1..n; r_i and h_j are tokens;
       bracket is 1 if unequal, otherwise 0; C is an integer count.
CONSTRAINT: Unit insertion, deletion and substitution cost;
            monotonic sequence alignment, no transposition operation.
NUMERICAL EXAMPLE: reference [a,b], hypothesis [a,c,b]; C(2,3)=1.
INTERPRETATION: One insertion is sufficient and no zero-edit path exists.
```

DERIVED: Backtrace any optimal path for edit categories. Multiple paths may tie; retain the scorer version and deterministic tie policy. The minimum total is stable for fixed sequences and weights, while the S/D/I breakdown can depend on the selected path.

For K segments, micro WER pools counts rather than averaging rates. It is E1 applied to the sum of the separately aligned segments, not necessarily to one concatenated text: a concatenated alignment can cross boundaries.

```text
FORMULA [E3]: micro WER = sum_k(S_k+D_k+I_k) / sum_k N_k
WHERE: k indexes K predeclared segments; N_k reference words.
CONSTRAINT: sum_k N_k > 0; same scoring policy;
            empty-reference segments may contribute insertions.
NUMERICAL EXAMPLE: segment A has 1 error / 2 words;
                   B has 1 error / 8 words; micro=2/10=20%.
INTERPRETATION: Longer segments receive more weight than in a segment macro average.
```

### 4.3 Key Special Cases / Variants

| Variant | What changes | Use | Boundary |
|---|---|---|---|
| CER | Units become declared characters | Scripts with unstable word segmentation or spelling analysis | Can hide semantic word changes; code points and graphemes differ |
| Macro WER | Mean of defined per-item rates | Equal utterance or group weight is the objective | Empty references need policy; tiny items have high influence |
| Term recall | Reference critical terms found after matching | Names, drug names, account numbers | Requires a curated inventory and matching rules |
| Semantic review | Human or task-specific adjudication | Meaning and consequence matter | More expensive and not a replacement for lexical reproducibility |
| Streaming revision | Compare successive partial outputs | Caption stability and readability | Requires event ordering and a declared stability metric |
| Silence insertion rate | Output on reference-empty audio | Hallucination and false activation testing | Use a time or segment denominator, not zero-word WER |
| Weighted edits | Different costs for categories or entities | Explicit business loss model | Not standard WER; must justify costs and report unweighted WER too |

For a weighted score, substitution, deletion, and insertion costs must be specified before running dynamic programming. It may select a different path than E2; rescaling counts from a unit-cost alignment is not generally equivalent to finding the weighted optimum.
For multilingual text, declare whether language identification is utterance-level or span-level, and whether transliteration and mixed-script forms are accepted. Character segmentation and normalization must be independently reviewed by language-competent annotators.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

WER has no upper bound: a one-word reference and four extra hypothesis words yield four insertions and 400% WER. Reporting a clipped 100% value loses information. A reference with no words makes per-segment WER undefined, even though a nonempty hypothesis has observable insertions. An all-empty corpus also has zero denominator: report undefined WER and a separate silence metric.
If the same mistaken normalization removes both clinically meaningful units, a perfect normalized score is possible while the transcript is unsafe. For example removing the word “not” from both sides would hide negation. Aggressive normalization is a policy choice, not evidence of acoustic accuracy.
**Takeaway:** An edit ratio is meaningful only for a declared tokenization, path, aggregation and nonzero denominator.

## 5. Worked Numerical Example

### 5.1 Word alignment

ILLUSTRATIVE input: the reference is “please call the nurse now” and the hypothesis is “please contact nurse now again”. Lowercase and spaces are the declared normalization; punctuation is absent. Each row below shows one alignment column.

```text
reference   please  call     the    nurse  now  —
hypothesis  please  contact  —      nurse  now  again
operation   match   S        D      match  match I
```

There are five reference words and three edits: one substitution, one deletion, one insertion. Thus WER is 3/5 = 0.60 = 60%. The length difference is zero; zero length difference does not imply zero errors because substitutions and offsetting insertions/deletions can occur.
Independent lower-bound check: “please”, “nurse” and “now” are common in order; converting “call the” into “contact” needs at least two unit edits, and “again” needs an insertion. The displayed three-edit path reaches that bound. This check does not claim every optimal alignment has the same category counts.
Decision: inspect whether the changed verb and omitted article matter for the target workflow. WER alone cannot answer whether a nurse was actually called. No action on a real patient is implied.

### 5.2 Character alignment

ILLUSTRATIVE reference “cat”, hypothesis “cut”: code-point sequences are c-a-t and c-u-t. Alignment is c/c match, a/u substitution, t/t match. One edit divided by three reference code points is CER = 1/3 = 33.33% rounded to two decimal places. Independently, the two strings have equal length and differ at one position, so zero edits cannot suffice and one substitution does.
Another pair, “café” in composed and decomposed Unicode, may differ at the code-point level despite looking identical. Apply a declared normalization such as NFC on both texts if that matches the evaluation contract; never assume a renderer makes the underlying sequences identical. Grapheme-based CER is a different metric from code-point CER.

### 5.3 Micro, macro and a sensitivity check

ILLUSTRATIVE set A has 1 error in 2 reference words and set B has 1 error in 8 reference words. Per-item WERs are 50% and 12.5%. Their equal-item macro average is (50% + 12.5%)/2 = 31.25%; micro is (1+1)/(2+8)=20%. Both are correct for different weighting questions.
If A gains one additional error while B stays unchanged, micro becomes 3/10 = 30% and macro becomes (100%+12.5%)/2 = 56.25%. This demonstrates why small segments can dominate macro averages. Neither example quantifies sampling uncertainty.

### 5.4 Streaming timing example

ILLUSTRATIVE client monotonic timestamps: audio capture begins at 0 ms; first speech begins at 200 ms; first partial text arrives at 650 ms; last speech ends at 2,400 ms; final text arrives at 2,900 ms. Speech-onset-to-first-partial latency is 650−200 = 450 ms. End-of-speech-to-final latency is 2,900−2,400 = 500 ms. Capture-start-to-final elapsed time is 2,900 ms; it includes speaking time and is not finalization delay.
The endpointing and network portions require separately recorded boundary events; the 500 ms total cannot identify them. If final delivery moves to 3,200 ms with unchanged endpoint, finalization delay rises to 800 ms. Percentiles cannot be inferred from one request.

### 5.5 Paired comparison

ILLUSTRATIVE two-system corpus has 1,000 reference words. A makes 100 total edits; B makes 80 on the exact same normalized, fixed segments. A micro WER is 10%; B is 8%. The absolute difference B−A is −2 percentage points; relative reduction is (100−80)/100 = 20%. These are different quantities.
Recompute the paired difference by resampling speakers or sessions as clusters if recordings are correlated. This toy example supplies no cluster data, so no confidence interval or significance claim is available. A change in the mix of speakers or recording quality would break the paired comparison.
**Takeaway:** Reproduce the alignment and denominator before trusting a reported improvement.

## 6. Visual / Geometric Intuition

In the E2 grid, moving down consumes a reference token and is a deletion; moving right consumes a hypothesis token and is an insertion; diagonal consumes both and is a match or substitution. The path starts at the empty prefixes and finishes at the full pair. Its cost, not geometric length, counts errors.
A path can run around an apparent word mismatch in more than one way. Equal-cost paths need not identify a unique cause. Human review should listen to the audio, not infer a diagnosis solely from one backtrace.

```text
Time ->  capture ---- speech begins -------- speech ends ------ final received
         0 ms         200 ms                 2400 ms            2900 ms
                           |--- first partial at 650 ms
                           |<-- 450 ms -->|
                                             |<--- 500 ms ----->|
```

Partial text may change after emission. A user reading a caption experiences both delay and corrections; a final transcript hides that history. Align timestamps to the same clock or synchronize and quantify clock error. A server timestamp and client timestamp cannot safely be subtracted merely because both use milliseconds.
**Takeaway:** Edit paths explain text difference; event timelines explain perceived delay.

## 7. Comparison / Selection Table

| Decision | Baseline or alternative | What it answers | Limit or guardrail |
|---|---|---|---|
| Lexical accuracy | Human-gold micro WER | Edits per reference word | Segment, normalize and disclose the denominator |
| Script sensitivity | CER plus WER | Character-level differences | Declare Unicode unit and language-specific tokenization |
| Short phrases | Per-utterance macro | Typical item under equal weighting | Show distribution and exclude or separately handle empty references |
| Critical content | Entity recall and human review | Whether names or numbers survive | Specify matching and account for false entity insertions |
| Live captioning | First-partial and final latency percentiles | User-visible timeliness | Include timeouts, clock source, and sample counts |
| Stability | Revision count or erased tokens | Partial-text churn | Declare whether formatting revisions count |
| Silence | Insertions per silent hour | Output without speech | Human-verify silence and distinguish faint speech |
| Performance across languages | Slice rates with counts and intervals | Where evaluation coverage differs | Avoid extrapolation from tiny or nonrepresentative slices |
| Privacy | Exposure audit and retention inventory | What is collected and who can access it | Obtain lawful authority and minimize raw-audio access |
| Promotion | Existing system versus candidate paired | Change on same eligible examples | Keep frozen holdout and predeclared guardrails |

A weak baseline is the current service evaluated using the same protocol, not a score copied from an unrelated benchmark. A text-only baseline cannot isolate acoustic recognition. A public dataset can help regression testing, but its recording, vocabulary, and language distribution may not represent the deployment. Do not claim a named dataset's license, composition, or performance without checking the current primary documentation.
**Takeaway:** Choose a metric for the decision and keep its complementary failure signals.

## 8. Failure Modes & Edge Cases

The following are illustrative diagnostic playbooks, not reports of observed incidents. Each includes competing explanations, a discriminating check, a bounded response, and a verification condition. Preserve privacy and human-review controls during investigation.

### 8.1 WER improved but users complain

**Competing causes:** A new normalizer may have removed meaningful punctuation or a small high-impact slice may have regressed.
**Discriminating check:** Re-score identical raw pairs with both frozen normalizers; stratify critical commands and language.
**Bounded response:** Restore meaningful tokens in a separately versioned strict view; do not silently rewrite the historical series.
**Verification:** The strict paired score and task review should explain the reported change; unresolved regressions block the affected route.

### 8.2 Insertions on quiet recordings

**Competing causes:** Voice activity detection may admit noise, or the decoder may hallucinate during correctly identified silence.
**Discriminating check:** Listen to consented sampled clips, inspect speech boundaries, and compare a no-speech baseline.
**Bounded response:** Adjust endpointing or decoder only after isolating the component; preserve a representative quiet holdout.
**Verification:** Silent-hour insertion rate falls without hiding low-volume speech or reducing speech coverage.

### 8.3 Deletion spike at beginning

**Competing causes:** Audio buffering may clip initial samples; alternatively the reference segment may start before the delivered audio.
**Discriminating check:** Compare client capture, server ingress, and reference timestamps for the same requests.
**Bounded response:** Repair transport buffering or align segmentation to actual delivered audio; do not delete reference words to improve the score.
**Verification:** Start-of-utterance deletion rate and audio completeness both recover on a held-out set.

### 8.4 Deletion spike at end

**Competing causes:** Endpointing can terminate early, while network loss can truncate the final packet.
**Discriminating check:** Inspect endpointer state alongside packet sequence and waveform tail.
**Bounded response:** Change endpoint or transport recovery based on evidence; cap extra delay using the agreed latency budget.
**Verification:** Final-word recall improves with no unacceptable increase in end-of-speech-to-final latency.

### 8.5 High WER only on numeric fields

**Competing causes:** Number formatting may differ between gold and hypothesis, or the recognizer may actually mishear a digit.
**Discriminating check:** Compare audio, verbatim reference, normalized text, and structured digit sequence.
**Bounded response:** Use a documented number equivalence policy and retain a strict digit accuracy measure for high-risk fields.
**Verification:** Manual adjudication confirms equivalences and critical digit error is reported separately.

### 8.6 One language has a large apparent regression

**Competing causes:** The sampled language may have changed, or the word tokenizer may segment the candidate output differently.
**Discriminating check:** Compare matched speakers and channels using a linguistically reviewed tokenizer and inspect counts.
**Bounded response:** Repair a faulty evaluation policy or restrict rollout; do not average the language away.
**Verification:** The same fixed slice is rescored, uncertainty is displayed, and affected owners approve the route decision.

### 8.7 Code-switching is systematically missed

**Competing causes:** Utterance-level language tags can hide short switches; the model may also genuinely fail on switches.
**Discriminating check:** Review span-level language labels against audio and compare switch boundaries with substitutions.
**Bounded response:** Adjudicate annotations and test multilingual context with a fixed code-switch holdout.
**Verification:** Report span coverage and switch-specific errors, including examples the annotator could not resolve.

### 8.8 CER and WER disagree

**Competing causes:** Word boundaries may be inconsistent or small spelling differences may create entire word errors.
**Discriminating check:** Inspect normalized token sequences and grapheme boundaries for a sample of disagreements.
**Bounded response:** Fix incorrect segmentation; otherwise retain both metrics and state why they answer different questions.
**Verification:** Independent language reviewers can reproduce tokenization and the discrepancy remains explainable.

### 8.9 Perfect score on a dangerous sentence

**Competing causes:** The normalizer may discard negation or the reference may share the same transcription error as the hypothesis.
**Discriminating check:** Listen to audio and compare raw and normalized versions with an independent annotator.
**Bounded response:** Preserve negation and re-adjudicate reference; report clinically important entity and negation errors.
**Verification:** A blinded audit detects the deliberately constructed negation contrast.

### 8.10 Model appears improved after data refresh

**Competing causes:** Population mix could be easier, or evaluation audio could overlap adaptation data.
**Discriminating check:** Compare stable cohort scores and fingerprint speaker, session, and audio overlap with training.
**Bounded response:** Quarantine suspected overlaps and repeat the paired holdout comparison.
**Verification:** No known leakage remains and cohort-specific estimates include their sample sizes.

### 8.11 Latency is low but captions feel slow

**Competing causes:** The dashboard may start its clock at server inference rather than client speech onset, or omit failed requests.
**Discriminating check:** Trace client capture, speech start, first receipt, and timeout events for the same request IDs.
**Bounded response:** Instrument an end-to-end clock and add success coverage plus censored outcomes.
**Verification:** A replayed trace reproduces the client-visible latency and missing outcomes are counted.

### 8.12 Final transcript is accurate but partial captions flicker

**Competing causes:** The decoder may repeatedly revise hypotheses, or the client may render stale messages out of order.
**Discriminating check:** Inspect sequence numbers and text revisions at the client alongside server emission order.
**Bounded response:** Discard stale messages or tune a commitment policy, preserving a latency-versus-stability comparison.
**Verification:** A deterministic replay shows bounded revision behavior without dropping final text.

### 8.13 P95 latency improves when traffic rises

**Competing causes:** Timeouts may be excluded from successful-request percentiles; alternatively traffic composition may have changed.
**Discriminating check:** Compare all eligible requests, timeout rate, and per-route latency distribution before and after.
**Bounded response:** Restore timeout accounting and use matched traffic strata for comparison.
**Verification:** Success coverage and latency are jointly reported; discarded observations are traceable.

### 8.14 Latency varies by device

**Competing causes:** Client capture scheduling or resampling may dominate; transport could also differ by network.
**Discriminating check:** Compare monotonic capture and send events by device and network strata.
**Bounded response:** Fix the measured bottleneck and keep device/network slices separate from model inference.
**Verification:** A held-out device sample improves without changes to the definition of start time.

### 8.15 RTF is below one but backlog grows

**Competing causes:** RTF may be a single-stream average, while aggregate arrival load exceeds available concurrency or burst capacity.
**Discriminating check:** Measure aggregate arrival and completion rates, concurrency, queue time, and service failures.
**Bounded response:** Add capacity or admission control according to the actual workload; do not infer p99 from RTF.
**Verification:** Backlog drains under a realistic recovery load and queue-time tails are within the agreed bounds.

### 8.16 Audio and reference segments do not line up

**Competing causes:** Segmentation may split a sentence differently or transcription may include speech outside the clip.
**Discriminating check:** Overlay segment boundaries on audio and inspect cross-boundary words.
**Bounded response:** Reconcile boundaries under a fixed rule before scoring; preserve raw source intervals.
**Verification:** A sample of boundary cases is independently adjudicated and segment coverage is unchanged.

### 8.17 Speaker labels improve while WER worsens

**Competing causes:** Diarization and transcription can fail independently; a changed scoring permutation can also make labels appear better.
**Discriminating check:** Evaluate speaker attribution and word errors against the same fixed audio and gold conventions.
**Bounded response:** Resolve the failing component separately; use a speaker-attributed metric only with a clear matching rule.
**Verification:** Both the lexical and speaker-attribution results are reproducible with their own denominators.

### 8.18 A quiet speaker performs poorly

**Competing causes:** Microphone distance, background noise, or a biased annotation convention may explain the gap.
**Discriminating check:** Compare audio level, channel, reference confidence, and matched recording context without inferring identity from voice.
**Bounded response:** Improve capture or model behavior based on evidence; avoid inferring protected attributes without a lawful purpose.
**Verification:** The consented slice improves on held-out sessions and quality does not regress elsewhere.

### 8.19 Reference disagreements dominate

**Competing causes:** The audio may be genuinely ambiguous or annotators may have different transcription conventions.
**Discriminating check:** Compute agreement on overlapping independently labeled audio and adjudicate sampled disagreements.
**Bounded response:** Clarify the convention and retain an uncertainty flag rather than forcing unsupported certainty.
**Verification:** Repeat annotation on new items and report unresolved coverage alongside the score.

### 8.20 Critical names are missed despite good overall WER

**Competing causes:** Names are rare, so aggregate WER can conceal their deletion; entity annotation can also be incomplete.
**Discriminating check:** Audit a representative set with independently verified names and inspect deletion and substitution types.
**Bounded response:** Add entity recall and error review; ensure any vocabulary adaptation is tested on unseen names.
**Verification:** Name coverage and error rates improve without revealing actual names in the report.

### 8.21 Privacy redaction looks successful

**Competing causes:** The redactor may mask only known formats, or logs may retain original interim hypotheses.
**Discriminating check:** Audit authorized examples across audio storage, partials, final output, and downstream logs.
**Bounded response:** Apply data minimization, short retention, access control, and redaction to every appropriate sink.
**Verification:** A controlled synthetic canary is absent from unauthorized sinks and deletion completion is evidenced.

### 8.22 A privacy slice has worse coverage

**Competing causes:** Consent restrictions may prevent evaluation on a cohort, or the cohort may receive a different processing route.
**Discriminating check:** Compare eligibility, missing labels, routing, and retention rules without bypassing consent.
**Bounded response:** Report the missingness and commission appropriately authorized evaluation; do not use restricted data anyway.
**Verification:** Coverage and route-specific uncertainty are disclosed, and no unsupported parity claim remains.

### 8.23 Aggregate WER improves but every route does not

**Competing causes:** A volume shift toward an easy route can reverse the pooled comparison; one route may also have a genuine regression.
**Discriminating check:** Hold the route mix fixed and report matched route-specific edits and denominators.
**Bounded response:** Make the launch conditional on route guardrails instead of the pooled number alone.
**Verification:** Each eligible route passes its own prespecified criterion or remains on the baseline.

### 8.24 Very large WER seems impossible

**Competing causes:** Insertion-heavy output can produce more edits than reference words; an accidental duplicate hypothesis can do the same.
**Discriminating check:** Inspect the alignment, output length, and duplicated transport messages.
**Bounded response:** Fix duplication if present; otherwise report the unbounded rate without clipping.
**Verification:** Recomputed counts match the stored sequences and the operational cause is documented.

### 8.25 Performance vanishes after deployment

**Competing causes:** A decoder configuration or normalization version may differ from the offline test; the production audio may differ too.
**Discriminating check:** Compare signed configuration, sample rates, decoding options, and paired shadow audio where authorized.
**Bounded response:** Restore the evaluated configuration or roll back; then separately assess distribution shift.
**Verification:** The same authorized fixture yields reproducible output across the tested paths.

### 8.26 Uncertainty interval is implausibly narrow

**Competing causes:** The calculation may treat repeated utterances from a speaker as independent, or ignore sampling design.
**Discriminating check:** Examine cluster sizes and resample speakers or sessions rather than individual words.
**Bounded response:** Recompute a paired cluster bootstrap and report its sampling unit and limitations.
**Verification:** The resampling procedure preserves each selected cluster and reproduces with a fixed seed.

### 8.27 Model selection keeps improving the holdout

**Competing causes:** Repeated tuning may have overfit the evaluation set, even without direct training overlap.
**Discriminating check:** Check experiment history and determine how often the holdout influenced model choice.
**Bounded response:** Use a newly sequestered test set and restrict future access; preserve the old set for development.
**Verification:** The final result is evaluated once under the declared selection protocol.

### 8.28 Missing transcripts disappear from WER

**Competing causes:** The scorer may drop failed requests, or join keys may exclude unmatched hypotheses.
**Discriminating check:** Reconcile eligible IDs against reference, hypothesis, error, and timeout records.
**Bounded response:** Report coverage and account for missing outputs separately under an explicit failure policy.
**Verification:** Counts reconcile to the original eligible population and quality cannot improve merely by dropping hard requests.

### 8.29 A benchmark improvement does not transfer

**Competing causes:** A public benchmark may have different acoustics or vocabulary; a local pipeline configuration may also differ.
**Discriminating check:** Run paired tests on authorized target audio and compare preprocessing and route composition.
**Bounded response:** Use target evidence for the promotion decision and label external results with their exact setting.
**Verification:** Target slice outcomes and benchmark outcomes are reported separately without conflating their populations.

### 8.30 Substitution categories change across scorer versions

**Competing causes:** Equal-cost alignment ties may be resolved differently or the tokenizer may have changed.
**Discriminating check:** Recompute total edit distance and inspect the versioned backtraces and normalized tokens.
**Bounded response:** Pin scorer and tie policy; compare totals first and do not interpret unstable breakdowns as model changes.
**Verification:** A small fixed fixture reproduces both the total and the expected chosen path for the pinned version.

### 8.31 Real-time text is fast but wrong

**Competing causes:** The decoder may trade future context for speed, or partial output may be incorrectly compared with final gold.
**Discriminating check:** Score partials at declared times and compare final text separately on matched audio.
**Bounded response:** Select a latency/accuracy operating point with explicit user and critical-content guardrails.
**Verification:** Both time-to-text and final and partial quality are reported for the chosen setting.

### 8.32 Out-of-order streaming events corrupt text

**Competing causes:** Retries can duplicate events, while asynchronous clients may apply updates by arrival order.
**Discriminating check:** Inspect utterance IDs, sequence numbers, and retry logs for an authorized trace.
**Bounded response:** Make updates idempotent and apply the protocol order; preserve finalization semantics.
**Verification:** A replay with reordered and duplicated events yields the same final state.

### 8.33 Hallucinated text occurs after speech ends

**Competing causes:** Trailing noise may be classified as speech, or a stale decoder state may bleed between utterances.
**Discriminating check:** Compare waveform and VAD boundaries, then test fresh-state and continuous-state decoding.
**Bounded response:** Reset state at the correct boundary or adjust endpoint policy based on the observed cause.
**Verification:** A held-out post-speech silence set reduces insertions without truncating real trailing words.

### 8.34 Different teams report different WER

**Competing causes:** They may use different segmentation, or punctuation and number policies may differ.
**Discriminating check:** Exchange hashed dataset manifests, scorer versions, normalized token sequences, and inclusion rules.
**Bounded response:** Publish a versioned evaluation contract and compute both legacy series where needed for transition.
**Verification:** Independent teams reproduce the same counts on a shared authorized fixture.

### 8.35 Anonymized transcript still exposes a person

**Competing causes:** Rare phrases and timing may identify a speaker, while replacement text may retain identifiers elsewhere.
**Discriminating check:** Conduct an authorized re-identification risk assessment across all shared fields and linked data.
**Bounded response:** Minimize fields, restrict access, and aggregate only when disclosure risk is acceptable.
**Verification:** An independent privacy review verifies controls; anonymization is not asserted solely from name removal.

### 8.36 High accuracy on repeated callers masks new-caller errors

**Competing causes:** Speaker overlap can leak acoustic traits, or deployment may disproportionately contain new devices.
**Discriminating check:** Compare speaker-disjoint and seen-speaker evaluations with device stratification.
**Bounded response:** Select the split matching the intended generalization and add the other as a separate diagnostic.
**Verification:** Held-out speaker performance and counts are visible before deployment.

### 8.37 Scoring a silent clip throws an exception

**Competing causes:** The reference denominator is zero, or the normalization pipeline removed every token.
**Discriminating check:** Inspect raw reference, normalized reference, and audio for each empty case.
**Bounded response:** Treat truly empty references with separate false-output metrics; correct accidental token removal.
**Verification:** All eligible silent clips remain in coverage counts and corpus WER is undefined only when its pooled denominator is zero.

### 8.38 A confidence threshold appears calibrated

**Competing causes:** Calibration could have been measured only on accepted predictions, or confidence could apply to tokens rather than utterances.
**Discriminating check:** Compare confidence with independent outcomes over all eligible requests and within important slices.
**Bounded response:** Calibrate on held-out data and make abstention and review capacity explicit.
**Verification:** Reliability and coverage are reported on a disjoint test population; low-confidence omissions remain visible.

### 8.39 Customer audio is missing from monitoring

**Competing causes:** Consent exclusions may be legitimate, or an ingestion defect may silently drop a route.
**Discriminating check:** Reconcile routing and consent records against aggregate traffic without retrieving unauthorized audio.
**Bounded response:** Repair missing metadata or ingestion while respecting retention and access constraints.
**Verification:** The missingness rate is tracked by route and consent status with no access policy bypass.

### 8.40 The endpoint seems accurate but final latency is high

**Competing causes:** Inference queueing or downstream network delivery can add delay after endpoint detection.
**Discriminating check:** Trace endpoint event, queue entry, decoder completion, and client receipt on synchronized clocks.
**Bounded response:** Address the measured stage and preserve a fallback for saturated capacity.
**Verification:** The same end-to-end latency definition improves, including failure and timeout rates.

**Takeaway:** Investigate a symptom with competing explanations before altering the scorer or deployment.

### 8.41 Reference and normalization field guide

#### Audio/reference identity
- Use a stable recording identifier independent of the displayed filename.
- Store clip start and end in the manifest.
- Reject duplicate IDs with conflicting durations.
- Join references and hypotheses by segment ID, not row position.
- Track recordings that cannot be matched.
- Protect the manifest because identifiers can be linkable.
- Reconcile eligible, scored, and excluded counts.

#### Overlapping speakers
- Declare whether simultaneous speech is transcribed as one stream or by speaker.
- Annotate both speakers when the objective requires complete speech.
- Do not assume a single-channel transcript captures both speakers equally.
- Mark time intervals where overlap begins and ends.
- Score overlap separately from clean single-speaker speech.
- Check whether the chosen metric allows speaker permutation.
- Send unresolved speech to adjudication instead of inventing words.

#### Unintelligible speech
- Define an explicit unintelligible marker in the reference convention.
- Record whether the marker is excluded from lexical scoring.
- Do not count a guessed word as an independently verified label.
- Track the fraction of audio that receives the marker.
- Compare marker rates across recording conditions.
- Use a second listener on an authorized sample.
- Explain that excluded content limits the measured population.

#### Filled pauses
- Specify whether “uh” and “um” are transcribed.
- Keep the same convention for both model outputs.
- Retain raw text if a normalized view drops fillers.
- Check if fillers are relevant to turn-taking applications.
- Do not confuse deliberate removal with acoustic recognition.
- Inspect systematic hallucination of fillers in silence.
- Version any policy change before comparing scores.

#### False starts
- Decide whether repetitions and abandoned words belong in the verbatim reference.
- Mark truncation consistently rather than completing the speaker’s intent.
- Evaluate verbatim and edited transcription separately when both matter.
- Inspect whether automatic punctuation masks partial words.
- Keep an audio-linked adjudication trail.
- Do not compare two conventions as if they were the same benchmark.
- Check that segment boundaries preserve the abandoned portion.

#### Acronyms
- State whether letter names are written separately or as an acronym.
- Build equivalence rules before viewing candidate scores.
- Audit ambiguous pronunciations with a fluent reviewer.
- Keep entity accuracy when exact acronym spelling matters.
- Test a held-out list that includes unseen abbreviations.
- Avoid rules that map unrelated spoken words to the same acronym.
- Report both lexical and task-specific effects.

#### Spoken numbers
- Specify representation of digits, number words, and decimal separators.
- Check negative signs, decimal positions, and units separately.
- Keep a verbatim representation for ambiguous utterances.
- Do not normalize distinct doses to the same value.
- Evaluate digit-sequence accuracy when exact values matter.
- Review language-specific counting conventions.
- Record whether equivalent formatting variants are accepted.

#### Dates
- Declare ordering conventions before converting a spoken date.
- Keep ambiguous dates unconverted when the locale is unknown.
- Distinguish day, month, and year field errors.
- Avoid treating a different date as a harmless formatting change.
- Sample ambiguous dates for bilingual or locale-aware review.
- Preserve the original spoken sequence as evidence.
- Use a separate field-level correctness measure for downstream actions.

#### Punctuation
- Specify whether reference punctuation is literal, editorial, or absent.
- Report an unpunctuated lexical score if the task permits it.
- Add punctuation accuracy when sentence boundaries affect the product.
- Audit question marks and negation-sensitive punctuation separately.
- Do not infer a speaker pause from punctuation alone.
- Check punctuation revisions in partial captions.
- Keep punctuation model changes out of claims about acoustic recognition.

#### Capitalization
- Decide whether capitalization is meaningful in the target workflow.
- Compare case-folded words for a case-insensitive lexical view.
- Retain raw casing when proper-name formatting matters.
- Account for language-specific case folding rules.
- Avoid assuming every script has upper and lower case.
- Check case-sensitive entity extraction downstream.
- Pin the normalization library version.

#### Unicode composition
- Choose a Unicode normalization form and record it.
- Apply it to both reference and hypothesis before character scoring.
- Different byte sequences may display as the same glyph.
- Count grapheme clusters only with a declared segmenter.
- Keep raw text to permit later audit.
- Test composed and decomposed accent examples.
- Do not call byte error rate a character error rate.

#### Whitespace segmentation
- Decide whether spaces define words in each evaluated language.
- Avoid applying English whitespace rules indiscriminately.
- Compare a language-aware tokenizer against reviewed examples.
- Score CER as a complement, not a universal replacement.
- Document how punctuation attaches to tokens.
- Include mixed-script examples in the test fixture.
- Report the segmentation policy with each language result.

#### Compound words
- Record whether compound splitting is permitted in the language.
- Review concatenated and separated variants with a native speaker.
- Avoid rewarding a segmentation rule tuned to one candidate.
- Compare character-level and word-level outcomes.
- Preserve domain terms whose internal boundary affects meaning.
- Check the same rule on unseen compound examples.
- Document residual differences after normalization.

#### Contractions
- Specify whether a contraction and its expansion are equivalent.
- Do not remove negation when expanding a contraction.
- Handle ambiguous apostrophes consistently.
- Review examples involving dialect and informal speech.
- Record whether the policy changes entity boundaries.
- Keep an unexpanded score if exact text matters.
- Make the normalization transformation deterministic.

#### Code-switch boundaries
- Mark language at the span level when feasible.
- Record uncertainty when the boundary is ambiguous.
- Separate within-utterance switching from monolingual examples.
- Avoid assigning the whole clip to only the dominant language.
- Inspect whether a switch causes errors on neighboring words.
- Include each relevant script in normalization tests.
- Disclose the number of switch spans and speakers.

#### Transliteration
- Decide whether output must use the source script or an accepted transliteration.
- Collect acceptable spellings before model evaluation.
- Avoid silently accepting arbitrary phonetic approximations.
- Separate script conversion errors from acoustic errors where possible.
- Seek language-competent review of alternative spellings.
- Report the proportion of outputs requiring transliteration.
- Keep original text for reproducible re-scoring.

#### Named entities
- Define the entity types and annotation spans before evaluation.
- Measure missed reference entities and spurious hypothesis entities separately.
- Distinguish a wrong name from a harmless casing difference.
- Use disjoint entity sets when testing generalization.
- Protect names in evaluation artifacts and examples.
- Review entities whose audio is inherently ambiguous.
- Report entity counts alongside recall or precision.

#### Speaker role
- Define roles from trusted metadata or an adjudicated protocol.
- Avoid inferring a role solely from accent or perceived identity.
- Measure attribution separately from lexical transcription.
- Treat unknown speaker assignments as a distinct outcome.
- Check how overlapping speech affects role assignment.
- Audit the end-to-end consequence of a role swap.
- Do not publish identifiable examples without appropriate authorization.

#### Reference revision
- Assign a version to every gold transcript.
- Record which segments were changed and why.
- Freeze the gold version for a paired comparison.
- Retain an authorized audit trail for adjudications.
- Recompute both systems after a legitimate reference correction.
- Avoid silently changing labels only for the weaker system.
- Report whether the change affects previously published figures.

#### Annotation agreement
- Use independent labels on a prespecified subset.
- Define whether agreement is token, span, or full-transcript based.
- Inspect disagreements rather than relying on one summary number.
- Separate ambiguous audio from inconsistent instructions.
- Provide adjudicators access to the applicable convention.
- Report the subset selection and unresolved cases.
- Do not interpret agreement as proof that every agreed label is correct.

**Takeaway:** Apply these checks to the specific evaluation contract; they do not establish a measured production result.

### 8.42 Streaming instrumentation field guide

#### Capture start
- Record when the client begins buffering audio.
- Distinguish capture start from the first acoustic speech event.
- Use a monotonic clock for elapsed client durations.
- Include leading silence only if the product definition requires it.
- Record the capture configuration and sample rate.
- Check for gaps between capture and send.
- State whether capture startup is included in user latency.

#### Speech onset
- Define speech onset through annotation or a documented detector.
- Separate detector-estimated onset from manually reviewed onset.
- Quantify sensitivity to uncertain onset labels.
- Avoid aligning onset from text receipt timestamps.
- Include quiet initial consonants in review samples.
- Use the same onset labels for paired systems.
- Report what proportion of requests lacks a usable onset.

#### Packetization
- Record the audio duration represented by each chunk.
- Measure buffering delay introduced before the first packet.
- Check whether chunks overlap or omit samples.
- Track sequence numbers and actual send time.
- Compare chunk configurations at matched recognition quality.
- Include packetization in the client-visible latency budget.
- Avoid inferring the whole latency distribution from chunk size alone.

#### Network ingress
- Capture client-send and server-receive events when allowed.
- Use synchronized clocks or estimate clock offset before subtraction.
- Separate transmission delay from server queue time.
- Count connection failures and retries.
- Inspect mobile and unstable-network strata separately.
- Avoid storing raw sensitive packet contents solely for timing.
- Verify correlation IDs survive reconnects.

#### Queue entry
- Record when a request enters the serving queue.
- Define the queue boundary consistently across versions.
- Measure wait separately from model execution.
- Include priority or admission class in diagnostics.
- Track rejection and timeout counts alongside delay.
- Inspect queue tails under representative concurrency.
- Do not derive queue time from a single total latency number.

#### First decode
- Identify the first inference invocation that produces a candidate.
- Distinguish model emission from client-visible text.
- Account for any decoder warm-up period.
- Record whether batching waits for additional streams.
- Measure early-token quality at the same latency point.
- Compare cold and warm states separately.
- Include decoding configuration in the experiment manifest.

#### First partial
- Define whether an empty or punctuation-only event counts.
- Use the first visible lexical output for a user-facing measure.
- Record client receipt rather than server send for end-to-end timing.
- Measure absence of any partial as a separate outcome.
- Compare first-partial delay with partial correctness.
- Exclude neither retries nor failed sessions without disclosure.
- Keep one event definition across candidate systems.

#### Partial updates
- Assign an utterance ID and monotonically increasing sequence number.
- Treat duplicate events idempotently in the client.
- Record when each update becomes visible.
- Check whether updates replace the full text or append a delta.
- Audit stale events arriving after newer ones.
- Measure how often visible content changes.
- Retain only the minimum text needed under the privacy policy.

#### Commit policy
- Define when a partial token becomes stable or committed.
- Record whether the system may revise committed text.
- Measure delay added by any stability window.
- Evaluate correctness of committed tokens separately from final text.
- Check what happens when a later correction is necessary.
- Keep the commitment definition stable across model comparisons.
- Use representative read speeds when interpreting usability studies.

#### Revision burden
- Specify whether punctuation-only changes count as revisions.
- Count removed or replaced visible words under a fixed rule.
- Report revisions per utterance together with duration.
- Inspect the tail of extreme flicker rather than only the mean.
- Distinguish textual edits from out-of-order delivery defects.
- Compare revision burden at matched latency and final accuracy.
- Protect interim transcripts because they may contain identifiers.

#### Speech offset
- Define end of speech separately from endpointer decision.
- Use reviewed audio labels for a representative evaluation subset.
- Record annotation uncertainty near trailing consonants.
- Track whether background speakers extend the reference offset.
- Distinguish end of the target speaker from end of the recording.
- Use the same offsets for both candidates.
- Do not treat an endpointer timestamp as independent ground truth.

#### Endpoint detection
- Record the moment the endpoint component fires.
- Measure offset-to-endpoint delay with compatible clocks.
- Inspect premature endpoint decisions as well as late ones.
- Test short pauses inside an utterance.
- Check low-volume endings and noisy tails.
- Measure how policy changes affect both truncation and delay.
- Preserve an explicit timeout for streams that never end normally.

#### Final emission
- Record the event that makes the server consider output final.
- Specify whether the server can revise after finalization.
- Separate final decoding time from transport to the client.
- Track final events missing after a successful partial.
- Inspect duplicate final events following retries.
- Use protocol state to reject conflicting finalizations.
- Do not equate a final marker with transcript correctness.

#### Client receipt
- Capture the time the final output reaches the application.
- Distinguish network receipt from actual rendering.
- Include client processing delays if users experience them.
- Use request IDs that do not expose identity unnecessarily.
- Handle offline clients and reconnects as explicit outcomes.
- Compare the same device and network populations across releases.
- Report the fraction of sessions with no final receipt.

#### Clock synchronization
- List each timestamp source and clock domain.
- Prefer same-device monotonic differences where possible.
- Estimate clock offset and uncertainty for cross-device traces.
- Detect clock steps and resets rather than accepting negative durations.
- Avoid using wall-clock time as a monotonic interval without safeguards.
- Retain synchronization metadata without exposing raw audio.
- Mark intervals invalid when clock uncertainty dominates the measurement.

#### Latency percentiles
- Define the eligible population and observation window.
- State the percentile algorithm when comparing implementations.
- Show sample counts and the fraction of timeouts.
- Report first-partial and finalization distributions separately.
- Do not average per-route p95 values to obtain a global p95.
- Investigate tails with route and load context.
- Avoid declaring a percentile reliable from a handful of observations.

#### Timeout censoring
- Keep timed-out requests in the eligibility count.
- Distinguish an observed timeout threshold from actual completion time.
- Report success probability at the threshold.
- State whether abandoned requests can ever complete later.
- Avoid substituting the threshold as an exact completion latency.
- Use a suitable censoring method only when its assumptions hold.
- Compare failure and latency jointly for release decisions.

#### Retries
- Identify whether a retry is a new request or an attempt within one request.
- Measure user-visible elapsed time across all attempts.
- Track duplicate outputs and duplicate billing risk.
- Keep a common correlation ID across attempts.
- Compare request success with attempt-level success.
- Inspect retry amplification during partial outages.
- Avoid counting a failed first attempt as a separate successful user session.

#### Backpressure
- Monitor queue depth and age under sustained load.
- Specify whether the client buffers, drops, or slows audio.
- Detect dropped chunks and surface incomplete transcripts.
- Separate admission rejection from later inference failure.
- Define a maximum acceptable buffer duration.
- Test recovery after load returns below capacity.
- Do not claim that larger buffers fix sustained overload.

#### Graceful degradation
- Define a safe behavior when live captions cannot meet the budget.
- Tell users when output is delayed or unavailable.
- Prevent stale partial text from appearing final.
- Keep a baseline or fallback route if authorized and validated.
- Measure fallback accuracy and latency separately.
- Log state transitions without unnecessary content.
- Practice recovery using a bounded synthetic or consented workload.

**Takeaway:** Apply these checks to the specific evaluation contract; they do not establish a measured production result.

### 8.43 Population and statistical field guide

#### Population definition
- Define who and what recordings are eligible before sampling.
- Separate intended use from actual observed traffic.
- Name the sampling frame and its blind spots.
- State the evaluation window and geographic scope.
- Document consent-dependent exclusions.
- Report counts before and after every selection.
- Do not generalize beyond the supported population without evidence.

#### Speaker split
- Keep a speaker in only one split for a new-speaker claim.
- Use stable privacy-preserving linkage where authorized.
- Check whether household or agent overlap creates additional dependence.
- Report any unlinked identities and their possible leakage risk.
- Evaluate known-speaker behavior separately if relevant.
- Do not split utterances randomly and call the result speaker-disjoint.
- Freeze split assignments before model selection.

#### Session split
- Group all segments from one call or session together.
- Prevent adjacent clips from crossing training and test boundaries.
- Account for shared channel and room conditions.
- Check for repeated calls by the same speaker across sessions.
- Use session clusters in uncertainty analysis when justified.
- Record the granularity of the grouping key.
- Disclose what dependence may remain across sessions.

#### Time split
- Reserve later recordings for a forward-looking evaluation.
- Ensure labels and corrections were available at the cutoff.
- Inspect changes in devices, vocabulary, and traffic mix.
- Avoid using future-derived features or metadata.
- Keep a separate fixed regression set for longitudinal comparisons.
- Report the exact date boundaries and inclusion rule.
- Do not interpret temporal degradation as a causal model effect without controls.

#### Paired evaluation
- Run candidates on identical eligible audio and segments.
- Apply the same reference version and normalization.
- Compare per-cluster errors before aggregating.
- Record failures for each candidate rather than dropping mismatches.
- Calculate absolute and relative differences separately.
- Use cluster-aware uncertainty where dependence exists.
- Reconcile any change in eligible population before declaring a win.

#### Unpaired evaluation
- Recognize that different samples can have different difficulty.
- Describe the sampling and weighting of both groups.
- Avoid attributing the entire difference to the recognizer.
- Use matched or randomized comparisons when feasible.
- Report uncertainty from both samples.
- Inspect language, channel, and recording distributions.
- Label any remaining comparison observational.

#### Cluster bootstrap
- Resample independent sampling units with replacement.
- Carry all segments of a selected cluster together.
- Compute paired candidate differences within each replicate.
- Choose the cluster level to match the data collection design.
- Record the random seed and number of replicates.
- Inspect instability when there are few independent clusters.
- Do not claim that bootstrapping repairs systematic label bias.

#### Slice size
- Report the number of speakers, segments, and reference words.
- Explain which count drives the displayed metric denominator.
- Avoid suppressing a poor result solely because the slice is small.
- Mark uncertainty and privacy-related disclosure limits.
- Use a prespecified minimum-evidence rule for decisions.
- Consider further authorized sampling where uncertainty is material.
- Do not call an underpowered comparison proof of equality.

#### Slice overlap
- Record whether one utterance can belong to multiple slices.
- Do not sum overlapping slice counts as unique users.
- Use intersections when a joint risk is important.
- Avoid creating tiny intersections that expose individuals.
- Report unknown or missing slice labels explicitly.
- Keep a separate overall denominator for reconciliation.
- Explain that slice membership may change over time.

#### Language metadata
- State whether language labels are self-reported, annotated, or inferred.
- Do not equate language with nationality or ethnicity.
- Support multiple labels for code-switched content.
- Track unknown and uncertain labels as outcomes.
- Check whether labels refer to audio, user preference, or model prediction.
- Avoid filtering out model failures through a predicted language gate.
- Review labels with competent speakers under appropriate consent.

#### Accent analysis
- Use respectful and purpose-limited descriptors.
- Do not infer protected identity from an acoustic signal.
- Prefer consented self-description or validated study metadata where appropriate.
- Avoid treating a broad regional label as a homogeneous accent.
- Report sample and speaker counts without exposing identities.
- Compare recording conditions as possible confounders.
- State the limitations of each category and its collection method.

#### Noise conditions
- Define noise categories from recording evidence or controlled setup.
- Distinguish background speech from stationary mechanical noise.
- Record signal quality metrics without assuming one metric explains recognition.
- Sample real authorized environments where possible.
- Compare noise slices with device and distance context.
- Check reference quality on difficult audio.
- Avoid asserting causality from a noisy-versus-clean aggregate alone.

#### Channel effects
- Separate phone, microphone, and compressed-audio routes.
- Record codec and sample rate when available.
- Check whether channels have different language mixes.
- Compare paired processing where the same source audio is available.
- Investigate clipping and packet loss independently.
- Do not assume resampling recreates missing acoustic frequencies.
- Report unobserved channel metadata as unknown.

#### Device effects
- Use device families rather than exposing unique device IDs.
- Include operating-system and client versions when relevant.
- Check capture gain, buffering, and routing differences.
- Compare performance within common network conditions.
- Track newly introduced device families separately.
- Avoid assigning a device problem to the acoustic model without traces.
- Apply retention limits to device metadata.

#### Speech rate
- Define how speech rate is measured and on which audio.
- Separate fast speech from long pauses when interpreting rates.
- Check whether reference word count biases the rate estimate.
- Report sample distributions rather than arbitrary bins alone.
- Inspect endpointing and omission errors at rate extremes.
- Account for language differences in word segmentation.
- Avoid using speaking rate as a proxy for identity.

#### Rare terminology
- Define domain terms without using test labels for model tuning.
- Include both seen and unseen terms in separate subsets.
- Evaluate critical terms alongside overall WER.
- Distinguish pronunciation ambiguity from spelling variants.
- Audit whether the reference list is representative.
- Report entity prevalence and denominator.
- Do not disclose sensitive terms in public error examples.

#### Reference uncertainty
- Flag spans with disputed or inaudible words.
- Keep uncertainty distinct from recognizer confidence.
- Use adjudication for a planned subset of difficult cases.
- Test sensitivity to plausible alternative references where justified.
- Report exclusions caused by ambiguity.
- Avoid scoring an invented reference as certain truth.
- Preserve the convention for alternate valid transcriptions.

#### Selection bias
- Compare characteristics of evaluated and eligible recordings.
- Identify consent, technical, and annotation exclusions.
- Check whether failure-prone audio is less likely to receive a label.
- Report missingness by route and relevant slices.
- Avoid claiming unbiased performance when missingness is unknown.
- Use weighting only when its assumptions and weights are defensible.
- Gather additional authorized evidence rather than bypassing restrictions.

#### Multiplicity
- Prespecify the most important slices and guardrails.
- Recognize that scanning many slices can produce chance extremes.
- Report all planned comparisons, not just favorable ones.
- Use an appropriate multiplicity strategy for formal claims.
- Treat exploratory patterns as hypotheses for later confirmation.
- Do not hide a safety-critical anomaly behind a significance filter.
- Describe how decisions handle both uncertainty and practical impact.

#### Distribution shift
- Monitor changes in language, device, channel, and traffic mix.
- Separate reference-free input drift from measured error drift.
- Sample newly emerging conditions for authorized labeling.
- Compare a stable cohort and current-traffic cohort separately.
- Inspect whether a scoring or instrumentation change explains the shift.
- Define who owns investigation and promotion decisions.
- Avoid asserting that a drift statistic alone proves accuracy changed.

**Takeaway:** Apply these checks to the specific evaluation contract; they do not establish a measured production result.

### 8.44 Privacy and operations field guide

#### Legal authority
- Identify the applicable authority for each intended data use.
- Distinguish permission to serve a request from permission to retain audio for evaluation.
- Record jurisdiction and policy ownership without asserting universal legal rules.
- Review any secondary use before collecting samples.
- Provide a process for changing or withdrawing authorization where applicable.
- Limit evaluation access to the authorized purpose.
- Escalate uncertain cases to qualified privacy and legal reviewers.

#### Data minimization
- Collect only fields needed to answer the evaluation question.
- Keep raw audio separate from aggregate metric outputs.
- Use short-lived identifiers where persistent linkage is unnecessary.
- Remove content from routine infrastructure logs when feasible.
- Record the owner of each retained data field.
- Check whether interim transcripts create a duplicate collection.
- Periodically review whether each field is still needed.

#### Retention inventory
- List audio, references, hypotheses, and partial outputs separately.
- Include backups, caches, derived datasets, and observability sinks.
- Assign a retention duration and owner to each store.
- Identify exceptions such as approved investigations.
- Document how deletion requests propagate across stores.
- Track completion or failure of deletion workflows.
- Avoid assuming that deleting the primary object removes all copies.

#### Access control
- Separate annotator, engineer, analyst, and administrator privileges.
- Grant only the access needed for an approved task.
- Use audited access to sensitive raw audio.
- Review group membership and access grants periodically.
- Protect service accounts and automated export paths.
- Remove access when assignments end.
- Investigate access patterns inconsistent with the approved purpose.

#### Transcript redaction
- Define the identifier categories the redactor should detect.
- Evaluate both missed identifiers and over-redaction.
- Test partial, final, and corrected transcripts separately.
- Use consented or synthetic canaries for controlled audits.
- Check downstream copies after the redaction boundary.
- Restrict raw-text access during manual evaluation.
- Do not describe redaction as guaranteed anonymization.

#### Audio de-identification
- Recognize that removing names from text does not modify the waveform.
- Consider voice characteristics and background conversations as potential identifiers.
- Limit authorized access to source audio even when transcripts are masked.
- Evaluate the effect of any audio transformation on recognition quality.
- Retain an appropriate link between transformed and source versions only when justified.
- Review residual disclosure risk with the responsible privacy team.
- Avoid promising irreversible anonymity without supporting evidence.

#### Aggregate disclosure
- Assess whether a small cell can reveal a sensitive fact.
- Set reporting rules appropriate to the actual privacy threat.
- Account for differencing across successive dashboards.
- Avoid publishing rare transcript phrases in examples.
- Restrict drill-down access when a combined slice is identifying.
- Record suppressed cells without pretending they contain zero errors.
- Balance disclosure protection with visibility into affected groups.

#### Privacy slices
- Define a consent or retention slice without exposing individual choices.
- Track whether restrictions change which recordings can be scored.
- Report coverage and unknown outcomes for each permitted route.
- Do not use restricted recordings merely to improve statistical precision.
- Investigate routing and selection differences as competing explanations.
- Seek additional authorized data for materially uncovered groups.
- Avoid drawing parity conclusions from missing evidence.

#### Sensitive incidents
- Use synthetic material to reproduce pipeline behavior where possible.
- Restrict real incident samples to an authorized response team.
- Document the reason for temporary access.
- Avoid forwarding raw snippets through general chat or ticket systems.
- Set an incident-specific retention and deletion plan.
- Capture aggregate lessons without reproducing identifiers.
- Verify remediation across all affected sinks.

#### Vendor evaluation
- Identify whether a vendor receives audio, text, metadata, or all three.
- Review permitted use, storage location, retention, and subprocessors.
- Check contractual and technical controls with the responsible owners.
- Benchmark candidates on authorized and comparable inputs.
- Inspect whether vendor normalization differs from the internal scorer.
- Assess export and deletion mechanisms before onboarding.
- Do not infer compliance from a favorable accuracy score.

#### Public benchmark
- Verify dataset license and access terms before use.
- Record dataset version, split, and reference convention.
- Inspect whether its language and acoustic mix match the intended use.
- Check for overlap with model training data where feasible.
- Report benchmark and target-population results separately.
- Avoid presenting a public score as a production guarantee.
- Recheck benchmark documentation when a new version is adopted.

#### Human review workload
- Define when a transcript is routed for review.
- Measure the number of cases and expected handling time.
- Include unclear audio and model abstentions in workload estimates.
- Provide reviewers with appropriate context and secure tools.
- Record reviewer corrections without treating them as automatically perfect.
- Audit whether review prioritization misses particular slices.
- Reassess staffing if the threshold or traffic mix changes.

#### Review selection
- Document whether review samples are random, risk-based, or both.
- Retain selection probabilities when later weighting is intended.
- Keep a representative audit sample separate from targeted debugging.
- Do not estimate overall error directly from a handpicked difficult set.
- Check whether human reviewers see model identity.
- Blind the candidate where practical to reduce expectation bias.
- Report coverage and limitations of reviewed samples.

#### Threshold selection
- Define the operational cost of missed errors and unnecessary reviews.
- Tune a threshold on development data rather than the final holdout.
- Measure both coverage and error among accepted outputs.
- Check calibration separately for important slices.
- Include reviewer capacity as a constraint.
- Reassess thresholds when the traffic distribution changes.
- Do not assume a model confidence score is a calibrated probability.

#### Confidence calibration
- Define the event that a confidence value predicts.
- Collect independent labels for a representative range of scores.
- Compare predicted confidence with observed correctness by bins or another stated method.
- Report bin counts and uncertainty rather than a single curve alone.
- Inspect calibration in low-resource languages and noisy conditions.
- Evaluate calibration after deployment changes.
- Do not calibrate on the same data used for the final performance claim.

#### Critical-content policy
- Identify the categories whose errors have high downstream impact.
- Define exact matching and acceptable alternatives before scoring.
- Measure missed, incorrect, and hallucinated entities separately.
- Link high-risk outcomes to an appropriate review or abstention path.
- Evaluate the complete human-machine workflow, not only the recognizer.
- Protect entity examples in reports.
- Update the policy when the downstream decision changes.

#### Promotion gate
- Specify candidate and baseline configurations before evaluation.
- Freeze the gold set, scoring rules, and decision thresholds.
- Require paired quality and operational evidence on supported routes.
- Include coverage, latency, failure, and privacy guardrails.
- Document who may approve an exception and on what evidence.
- Keep an accessible rollback route for eligible traffic.
- Record the final decision and unresolved uncertainty.

#### Shadow testing
- Run a candidate without using its output for the live decision.
- Confirm authorization before duplicating audio to another route.
- Control extra compute and vendor exposure.
- Record failures and latency as well as transcript quality.
- Ensure shadow traffic represents the intended deployment population.
- Avoid feedback from shadow outputs into the frozen holdout.
- Use a limited staged rollout only after the predefined review.

#### Rollback
- Keep the previous validated configuration and routing policy available.
- Define the triggers and decision owner before launch.
- Measure whether rollback itself interrupts active streams.
- Preserve event identifiers across routing changes.
- Avoid retaining sensitive data solely for a rollback drill.
- Verify that key quality and availability indicators return to the baseline range.
- Document residual impact and the conditions for resuming promotion.

#### Monitoring without labels
- Track traffic, latency, failures, audio characteristics, and output anomalies.
- Treat these signals as proxies rather than measured WER.
- Look for sudden changes in empty output and unusually long output rates.
- Compare metrics within stable routes and client versions.
- Audit whether instrumentation changes explain apparent shifts.
- Use alerts to trigger authorized labeling or investigation.
- Do not claim model accuracy improved solely from an input-drift dashboard.

#### Sampled production labels
- Define a consented sampling frame and collection window.
- Include failures and incomplete outputs in the sampled population.
- Use independent human references rather than model text as gold.
- Oversample critical rare cases only with a documented analysis plan.
- Preserve sampling probabilities when estimating population performance.
- Keep a protected holdout separate from iterative debugging.
- Report the lag between production occurrence and available labels.

#### Error taxonomy
- Separate acoustic, segmentation, normalization, and reference issues.
- Allow multiple causes when one error spans components.
- Use stable category definitions and reviewer instructions.
- Audit agreement between independent error reviewers.
- Keep unknown cause as a valid category.
- Link each category to a possible diagnostic or mitigation.
- Avoid treating a taxonomy label as proof of causality.

#### Model comparison report
- List candidate versions and decoding parameters.
- Describe dataset provenance, consent, and split policy.
- Show edit counts and denominators alongside rates.
- Report per-route latency and failure coverage.
- Include language and critical-content slices with uncertainty.
- Disclose normalization and scorer versions.
- State whether the result is illustrative, measured, or externally sourced.

#### Experiment integrity
- Maintain an immutable record of model and scorer versions.
- Separate candidate selection from final evaluation.
- Record all planned metrics and failure rules before seeing results.
- Track any post-hoc changes as exploratory.
- Reproduce the paired comparison with the same input manifest.
- Restrict access to sensitive evaluation artifacts.
- Preserve a decision log without asserting an unperformed deployment.

**Takeaway:** Apply these checks to the specific evaluation contract; they do not establish a measured production result.

### 8.45 Additional system and scoring field guide

#### Wake-word contamination
- Separate wake-word activation from the downstream ASR task.
- Record whether activation audio is included in the reference.
- Measure false activations on representative non-target speech.
- Count missed activations separately from downstream word deletions.
- Do not exclude failed activations from the user journey metric.
- Inspect accidental activation of private conversations under authorization.
- Use a separate privacy review for always-listening capture.

#### Barge-in
- Define interruption start as a user speech event while playback is active.
- Measure how quickly playback stops after interruption.
- Record whether the interrupted prompt leaks into the recognition input.
- Assess the first words spoken during acoustic overlap.
- Evaluate false interruption from non-user sounds.
- Include recovery when the interruption is rejected.
- Keep barge-in latency distinct from ordinary transcription latency.

#### Echo cancellation
- Capture whether the reference audio includes device playback leakage.
- Measure recognition with and without echo-control changes on paired recordings.
- Distinguish echo suppression artifacts from recognizer errors.
- Inspect double-talk where the user and device speak together.
- Check that cancellation does not suppress the user’s voice.
- Include acoustic and device-route slices.
- Avoid extrapolating a lab-only result to all rooms and devices.

#### Resampling
- Record the input and output sample rates and conversion method.
- Ensure both candidates receive equivalent audio bandwidth.
- Check duration preservation and clipping after conversion.
- Do not assume upsampling restores missing high-frequency content.
- Measure any latency introduced by buffering or conversion.
- Compare raw and converted waveforms on authorized fixtures.
- Treat a preprocessing change as part of the evaluated system version.

#### Audio clipping
- Identify saturation using waveform inspection or a defined numeric detector.
- Distinguish clipping from quiet speech and packet loss.
- Review clips near the detection threshold.
- Report clipped and unclipped performance separately.
- Check whether clipping originates at capture or later processing.
- Avoid repairing the evaluation by silently excluding difficult clips.
- Include capture remediation in the product-level action plan.

#### Channel mixing
- Declare whether stereo channels are scored separately or mixed.
- Check for opposite-polarity cancellation when mixing channels.
- Retain a channel assignment when speaker roles depend on it.
- Compare channel-level and mixed-audio quality on the same sessions.
- Inspect silence on one channel before assuming a speaker is absent.
- Record the mixing rule and source channel layout.
- Protect any speaker-linked channel metadata.

#### Diarization boundaries
- Define how speaker turns are assigned near overlapping speech.
- Separate speaker attribution mistakes from lexical mismatches.
- Determine whether speaker labels are arbitrary or tied to identity.
- Declare any permutation matching used for scoring.
- Handle unknown speaker labels explicitly.
- Inspect transitions where a correct word has the wrong speaker.
- Avoid equating low ordinary WER with correct attribution.

#### Language identification gating
- Record which language route was selected for every eligible request.
- Evaluate language identification errors independently of ASR.
- Count unsupported and unknown language outcomes.
- Check whether gating hides difficult code-switched clips.
- Compare a fallback route with a correctly routed baseline.
- Report end-to-end results as well as conditional route accuracy.
- Do not select evaluation clips using the candidate’s own prediction alone.

#### Abstention
- Define what the system emits when it is not sufficiently confident.
- Count abstentions in coverage rather than excluding them silently.
- Measure accuracy among accepted outputs and across all eligible requests.
- Check whether abstention rates differ materially across slices.
- Provide an appropriate human or alternative route when required.
- Evaluate the delay introduced by that fallback.
- Avoid treating absence of a transcript as zero WER.

#### Partial scoring policy
- Choose fixed elapsed-time checkpoints or audio-progress checkpoints.
- Apply the same checkpoint definition to all candidates.
- Decide whether to compare only words spoken by that point.
- Avoid penalizing a partial for words not yet spoken.
- Report how unstable partials are matched to time-aligned references.
- Treat missing partials as a distinct outcome.
- Compare checkpoint quality with final quality and display delay.

#### Audio duration accounting
- Measure source audio duration independently of processing wall time.
- State whether leading and trailing silence are included.
- Include all chunks when a stream reconnects.
- Avoid double-counting overlapped chunks.
- Record whether processing includes preprocessing and transport.
- Use a consistent denominator for real-time factor comparisons.
- Report failures separately when processing time is undefined.

#### Concurrent stream limits
- Define a stream slot in terms of the actual serving configuration.
- Measure performance at representative concurrency levels.
- Watch memory growth from per-stream decoder state.
- Include stream duration and arrival burstiness in capacity studies.
- Track queue delays and admission failures at saturation.
- Compare cold-start and steady-state behavior.
- Do not turn a mean active-stream estimate into a tail guarantee.

#### Sampling weights
- Define the target traffic distribution before choosing weights.
- Assign weights to the actual sampling units.
- Check whether some target groups have zero sampling probability.
- Inspect extreme weights that make estimates unstable.
- Report unweighted sample counts alongside weighted rates.
- Avoid using weights to correct an unmeasured missing population.
- Recompute uncertainty using a method consistent with the sampling design.

#### Entity precision
- Count correctly matched predicted entities in the numerator.
- Count all predicted entities in the precision denominator.
- Define entity boundary and value matching before evaluation.
- Treat hallucinated entities as false positives.
- Report recall as well because precision can hide missed entities.
- Inspect rare entity types separately where sample size permits.
- Protect sensitive entity values in the resulting audit.

#### Entity recall
- Count correctly matched reference entities in the numerator.
- Count all eligible reference entities in the denominator.
- Include deleted or substituted critical entities as misses under the policy.
- Define how repeated mentions are counted.
- Document whether partial matches receive credit.
- Report the number of reference entities and annotator uncertainty.
- Review consequences with the downstream task owner.

#### False-output rate
- Define a no-speech segment through an audio review protocol.
- Count segments that produce any eligible lexical output.
- Divide by the number of eligible no-speech segments when that is the chosen rate.
- Also report insertions per silent hour to account for duration.
- Distinguish noise-induced activation from recognition after true activation.
- Keep ambiguous distant speech out of a confidently silent set.
- Include quiet real speech in a separate false-negative check.

#### Word-level timestamps
- Define whether a timestamp marks onset, offset, or midpoint.
- Compare against time-aligned human references when available.
- Report alignment tolerance and annotation uncertainty.
- Check timestamp monotonicity and impossible overlaps.
- Distinguish recognition correctness from timing correctness.
- Evaluate hard cases at speaker changes and short pauses.
- Do not use word timing as an identity signal without authorization.

#### Failure accounting
- Enumerate capture, transport, service, timeout, and scoring failures.
- Give every eligible request one terminal accounting state.
- Maintain separate counts for missing audio and missing hypotheses.
- Avoid interpreting a failed recognizer as a perfect empty transcript.
- Define how failures affect the release gate.
- Keep errors observable without logging unnecessary content.
- Reconcile totals whenever eligibility rules change.

#### Model update provenance
- Record model weights, decoder, language model, and preprocessing versions.
- Include endpointing and text normalization in the evaluated configuration.
- Link the configuration to the evaluation manifest.
- Separate model changes from infrastructure changes when diagnosing effects.
- Record whether adaptation used any related speakers or sessions.
- Avoid treating a product name as a complete model specification.
- Require new evidence after a material configuration change.

#### Accessibility review
- Consider users who depend on captions rather than audio playback.
- Evaluate reading delay and revision burden with an appropriate study design.
- Check whether punctuation or speaker changes affect comprehensibility.
- Include relevant language and device conditions.
- Respect participant consent and accessibility needs.
- Distinguish subjective usability feedback from lexical scoring.
- Use findings to set product-specific guardrails rather than universal thresholds.

**Takeaway:** Evaluate adjacent components only where their behavior affects the transcription decision.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

For reference length m and hypothesis length n, the standard E2 dynamic program fills (m+1)(n+1) cells. Each cell examines three candidate predecessors, so time is O(mn) for a fixed-cost token comparison. Retaining the entire grid for a straightforward backtrace takes O(mn) cells. If only the minimum distance is required, two rows suffice and use O(min(m,n)) cells after choosing the shorter sequence as the row width. Recovering a full alignment with reduced memory requires a different backtrace strategy and additional computation; two rows alone do not preserve the full path.
This complexity is per segment. Summing segment costs is generally cheaper than aligning a whole corpus concatenation, and segmentation also prevents unintended cross-utterance matches. For long continuous recordings, segment deliberately and document the effects near boundaries. Text preprocessing, audio decoding, annotation, and aggregation have their own costs; this grid bound is not an estimate of model inference cost.
For K segment pairs with lengths m_k and n_k, total cell updates are proportional to the sum of m_k n_k, not necessarily the product of total reference and total hypothesis length. Score tables and paths must be protected as potentially sensitive text. Minimizing stored intermediates reduces exposure but can impair reproducibility, so choose an authorized retention policy and record non-sensitive versioned metadata.

### 9b. Back-of-Envelope Sizing

ILLUSTRATIVE assumptions: 1,000 segments have 100 reference and 100 hypothesis words each. A full grid has 101 by 101 = 10,201 cells per segment, so 1,000 separate grids entail 10,201,000 cell positions over the batch. At four bytes per stored numeric cell, one grid contains 40,804 bytes, approximately 39.85 KiB; a sequential worker can reuse this memory. Backtrace pointers, object overhead, strings, runtime representation, and concurrency are excluded. Four bytes per cell is an assumption, not a promise about a particular implementation.
Independent check: 10,201 × 4 = 40,804 bytes; 40,804 / 1,024 = 39.84765625 KiB. If ten segments run concurrently with separate full grids, the numeric grids alone require 408,040 bytes, approximately 398.48 KiB. This is neither total process memory nor measured throughput.

ILLUSTRATIVE streaming capacity: suppose requests are continuously active for 10 seconds on average, arrival rate is 6 requests per second, and each stream uses one equivalent stream slot throughout its lifetime. With a stable system and representative averages, Little's law gives mean active streams = 6 requests/s × 10 s/request = 60 streams. This count is not the number of GPUs, a safe capacity limit, or a p99 concurrency estimate. Bursts, admission control, session lengths, and headroom require measurement.
If mean active duration rises to 15 seconds at the same arrival rate, the corresponding mean is 90 streams. If arrivals exceed sustainable completions, the stable-system assumption fails and the equation cannot justify a steady-state claim. The accounting unit is requests, not audio seconds or tokens.

### 9c. Measurement and operating cost

Human annotation cost depends on recorded duration, playback speed, ambiguity, reviewer expertise, and adjudication rate. Budget double annotation for a sampled subset and expert review for difficult languages and safety-critical spans. These are planning considerations, not a claim about a universal annotation price.
Streaming compute cost depends on concurrent streams, decoder state, batching, input rate, hardware, and routing. Include idle reservations, autoscaling lag, logging, storage, network egress, retry amplification, and human review. Do not calculate deployment cost from WER or RTF alone.
Data governance adds access administration, retention enforcement, audit, and secure deletion costs. Deleting only a primary object may leave replicas, caches, transcripts, or monitoring extracts. A retention inventory must include those destinations, ownership, and completion evidence.
**Takeaway:** Characterize algorithmic memory separately from end-to-end latency, concurrency, annotation, and privacy cost.

## 10. Key Theorems / Results Reference

**Edit-distance recurrence (DERIVED):** Under E2's unit-cost edit model, an optimal path must end in an insertion, deletion, match, or substitution. The recurrence considers every possible final action, and its base cases cover empty prefixes. By induction over prefix lengths, each cell gives the minimum edit count. This is a correctness argument for the stated recurrence, not a claim that its alignment is semantically optimal.

**Weighted average identity (DERIVED):** For nonempty segments, micro WER is the weighted average of segment WERs with weights proportional to reference length. Expanding each weight cancels the segment denominator and produces E3. This identity requires a separate convention for empty-reference segments, whose insertions cannot be expressed as a finite per-segment WER.

**Little's law (DERIVED application of a named result):** In a stable system with appropriate long-run averages and a consistent system boundary, average number in the system equals average arrival rate times average time in that system. The illustrative example in Section 9 uses active stream lifetime; the result does not determine tail behavior or concurrency ceilings. Primary original source: John D. C. Little, “A Proof for the Queuing Formula: L = λW,” Operations Research 9(3), 1961, DOI [10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). Citation supplied for identification; source text not independently reviewed in this draft. No publication status is implied.

**Paired design principle (DERIVED):** Testing both systems on the same held-out audio controls differences in the sampled items. It does not remove annotation bias, training leakage, selection bias, or correlated outcomes. Use a cluster-aware uncertainty procedure matching the sampling design.

**Source ledger:** Directory names 20.4 and 20.11 are planning provenance only; the named source contents are unavailable. An external primary source for ASR scoring and Unicode segmentation still needs review before publication. References to algorithms above are derived locally; no benchmark number, model performance, or vendor guarantee is claimed. Recheck sources and versions whenever scoring units or timestamp definitions change.
**Takeaway:** A theorem supports only conclusions inside its stated assumptions; source and deployment claims remain open.

## 11. Interview / Exam Questions

### Level 1 — Definition / Recall

**Q1. What do WER and CER measure?**
**Answer:** WER is the unit-cost minimum word edit count divided by reference word count; CER applies the same construction to declared character units. Neither directly measures intent, safety, or meaning. For example, a one-character numeric substitution can be consequential despite a small CER.
**Reasoning:** The numerator comes from an alignment, while the denominator is reference length, not hypothesis length. Both require a positive pooled reference denominator.
**Follow-up:** Is WER bounded by 100%? No; insertions can exceed the reference length.
**Scoring:** Full credit states edits, reference denominator, and insertion boundary. Red flag: treating WER as a probability of a wrong utterance.

**Q2. Why normalize transcripts?**
**Answer:** Normalization defines which surface differences matter for a particular task, such as case or punctuation, and makes scoring reproducible across systems. Apply the same frozen version to both reference and hypothesis. Preserve a strict view when punctuation or formatting carries operational meaning.
**Reasoning:** An undocumented rule can change the metric without any acoustic change.
**Follow-up:** Should “not” be removed as a stop word? No; doing so can erase a safety-critical negation.
**Scoring:** Full credit identifies symmetry and information-loss risk. Red flag: changing rules after observing which model wins.

**Q3. Define first-partial and finalization latency.**
**Answer:** First-partial latency is elapsed time from a named start event, such as speech onset, to client receipt of the first eligible partial. Finalization latency can be end of speech to client receipt of the final output. Other definitions are possible but must name their endpoints and clocks.
**Reasoning:** Starting at server inference omits capture and network delay; starting at capture start instead of speech onset includes leading silence.
**Follow-up:** Is RTF the same thing? No; RTF is a processing-time/audio-duration ratio under a specified protocol.
**Scoring:** Full credit gives both endpoints and avoids clock-domain subtraction without synchronization.

### Level 2 — Derivation / Apply

**Q4. Calculate a word score.**
**Question:** Reference “a b c”; hypothesis “a x c d”. What is WER?
**Answer:** Align a/a, b/x, c/c, and gap/d. The path has one substitution and one insertion, for 2/3 = 66.67% rounded. At least one edit is needed for b versus x and one for the extra token, so two is minimal.
**Follow-up:** What changes if d is absent? One substitution over three reference words gives 33.33%.
**Scoring:** Full credit gives an alignment, denominator, and minimum argument. Red flag: dividing by four hypothesis words.

**Q5. Compare micro and macro.**
**Question:** Segment A has 1 error out of 2 words; B has 1 out of 8. Calculate both.
**Answer:** Micro pools 2/10 = 20%. Equal-segment macro averages 50% and 12.5% to obtain 31.25%. They answer different weighting questions.
**Follow-up:** Can an empty-reference segment have a finite per-segment WER? No; its insertions must be separately handled or included in a positive pooled denominator under a declared policy.
**Scoring:** Full credit shows both denominators and states the weighting difference.

**Q6. Explain the edit recurrence.**
**Answer:** For each prefix pair, the last action must be deletion, insertion, or diagonal match/substitution. Add its cost to the corresponding optimal predecessor and take the minimum. Boundary prefixes require only insertions or deletions, so induction establishes the optimum.
**Follow-up:** Can two optimal paths have different S/D/I breakdowns? Yes; a pinned tie rule is needed for reproducible category counts.
**Scoring:** Full credit includes bases, the final-action argument, and the tie limitation.

**Q7. Calculate a timing budget.**
**Question:** Speech ends at 4.2 seconds on a synchronized client clock and final output is received at 4.85 seconds. What is the finalization latency?
**Answer:** 4.85 − 4.2 = 0.65 seconds, or 650 milliseconds. This is one observation, not a p95. Without intermediate events it cannot be divided into endpoint, inference, and network portions.
**Follow-up:** If endpoint detection takes 300 ms and the remainder is serial, how much remains for later stages? 350 ms, assuming the same start and end boundaries and no overlap.
**Scoring:** Full credit converts units and refuses a percentile claim from one sample.

**Q8. Interpret a paired change.**
**Question:** On the same 1,000 words system A has 100 edits and B has 80. Quantify the change.
**Answer:** A has 10% and B has 8% micro WER. B improves by 2 percentage points absolute and 20% relative to A's error rate. An uncertainty interval cannot be computed from totals alone if the cluster-level outcomes are missing.
**Follow-up:** What resampling unit is appropriate for repeated speaker data? Usually the independent sampling cluster, often the speaker or session, with the choice justified by the design.
**Scoring:** Full credit distinguishes percentage points and relative percent, and asks for cluster data.

### Level 3 — Judgement / Design / Debug

**Q9. A candidate reduces WER but harms one language. Ship?**
**Answer:** Not on the aggregate alone. Check the paired slice, sample size, confidence interval, annotation and tokenizer policy, and prespecified language guardrail. Restrict or roll back the affected route if the guardrail fails; seek more authorized data if uncertainty is too high.
**Reasoning:** A large easy-language population can dominate the pooled denominator.
**Follow-up:** Is a small sample proof of parity? No; it can be inconclusive and may warrant a hold rather than a pass.
**Scoring:** Full credit combines slice reliability with a bounded rollout decision. Red flag: declaring fairness from aggregate improvement.

**Q10. A service has RTF 0.5 yet its queue grows. Diagnose.**
**Answer:** A single-stream RTF does not establish aggregate concurrency capacity. Compare arrival and service rates, burstiness, active sessions, admission policy, and queue delay. Misdefined RTF or insufficient stream slots are competing hypotheses.
**Follow-up:** What if arrivals exceed completions for a sustained period? Backlog grows until load changes, capacity rises, or requests are shed.
**Scoring:** Full credit identifies system-level load and avoids inferring a p99 from RTF.

**Q11. The p95 improved after a release, but customers report timeouts. What next?**
**Answer:** Reconcile all eligible requests, including timeout and failed outcomes, and compare success coverage with the conditional successful-request latency distribution. Investigate whether hard requests were removed from the percentile and compare matched routes.
**Reasoning:** Faster successful requests can coexist with more failures.
**Follow-up:** Should every timeout be assigned exactly the timeout threshold as its actual latency? No; the true completion time is unobserved and may be censored or nonexistent.
**Scoring:** Full credit explicitly accounts for missing outcomes and avoids presenting a biased percentile as end-to-end experience.

**Q12. How would you evaluate silence hallucinations?**
**Answer:** Curate authorized, human-checked no-speech intervals with representative noise; report insertion count per silent hour and proportion of silent clips producing output, with duration and clip counts. Include false VAD activation and separately check that the fix does not suppress quiet speech.
**Follow-up:** Why not per-clip WER? A truly empty reference has zero reference-word denominator.
**Scoring:** Full credit covers denominator, false-positive signal, and quiet-speech tradeoff.

**Q13. A privacy constraint forbids retaining raw production audio. How can evaluation continue?**
**Answer:** Use appropriately consented and retained evaluation audio, a synthetic or specially collected fixture where suitable, and privacy-preserving operational aggregates. Report population coverage and selection bias; do not imply that synthetic audio establishes real-world parity. Audit transcript and partial-output sinks as well as audio.
**Follow-up:** Can an aggregate dashboard prove compliance? No; access, retention, legal basis, and deletion need separate evidence and review.
**Scoring:** Full credit respects the constraint, describes coverage limits, and avoids unauthorized collection.

**Q14. A recognizer writes the correct drug name but changes the dosage. Is low WER enough?**
**Answer:** No. Evaluate critical entities and digit sequences against independently adjudicated references, and route uncertain or high-risk results to a qualified reviewer. A small lexical error may materially alter a decision.
**Follow-up:** Would heavily weighting dosage errors alone establish safety? No; the weights and reference quality require validation, and downstream use and human factors remain relevant.
**Scoring:** Full credit distinguishes lexical error from downstream harm and proposes a measurable guardrail.

**Q15. How would you diagnose a scoring disagreement across teams?**
**Answer:** Compare eligible IDs, segmentation, raw input hashes, normalization, tokenizer units, scorer version, tie policy, and pooled counts. Reproduce a small authorized fixture before comparing models. Check that failures and empty references receive the same treatment.
**Follow-up:** What if total edits agree but substitution and deletion counts differ? An optimal-path tie or differing backtrace policy may explain the discrepancy; inspect the fixed token sequences and paths.
**Scoring:** Full credit uses evidence to distinguish data, metric, and implementation differences. Red flag: selecting whichever score looks better.

**Takeaway:** Strong answers specify population, units, assumptions, checks, and a consequence under changed constraints.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Evaluate batch and streaming ASR using paired lexical, timing, slice and privacy evidence. |
| Core formula | WER = (S+D+I)/N for N>0; CER uses declared character units. |
| Key assumption | Frozen segmentation, normalization, alignment, population and reference conventions. |
| Complexity | Unit-cost edit distance takes O(mn) time; full grid O(mn) space for lengths m and n. |
| Use when | Comparing recognizers on the same eligible, independently annotated audio and declared routes. |
| Avoid when | Treating one aggregate as proof of meaning, safety, fairness, or performance on an unmeasured population. |
| Go deeper | [Module 18 index](../INDEX.md); source coverage remains open pending access and review. |
| Operational guardrail | Report failures, language/critical-content slices, client-visible timing and privacy coverage alongside WER. |
