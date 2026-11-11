
## 7. Comparison / Selection Table

No metric is universally right. The tables below map a decision to the measurement that can actually block it, then compare the metrics against each other on the properties a reviewer cares about.

### 7.1 Decision to measurement

| Decision | Primary measure | Mandatory guardrail | Why the primary alone fails |
|---|---|---|---|
| Promote a model to production | corpus WER on a frozen, stratified set, with a clustered interval | worst-slice WER, insertion rate, critical-field accuracy | an aggregate gain can hide a regression in one slice or a fabrication surge |
| Choose between two vendors | paired WER delta with a paired interval, plus cost per audio hour | latency p95, data-handling terms | a 1-point difference within the interval is not a difference |
| Ship real-time captions | finalisation p95 and partial stability rate | WER on the caption slice, insertion rate on silence | low WER with 1 s finalisation is unusable for live conversation |
| Automate a downstream step on the transcript | field accuracy and critical-error rate | confidence calibration, abstention coverage | token-level WER cannot express "the claim number was wrong" |
| Size the human-review team | error distribution by utterance, not the mean | per-reviewer throughput | a mean predicts total edits but not how many files need a reviewer |
| Diagnose a customer complaint | per-utterance alignment path plus audio inspection | slice rates for the affected cohort | aggregate rates cannot locate a single bad session |
| Certify a language launch | per-language WER with intervals, plus code-switch slice | reference agreement (kappa), native-speaker review | a language with 40 utterances cannot be certified at all |
| Measure privacy redaction | redaction recall and precision on annotated entities | WER, since redaction can damage text | redaction quality is a separate classifier with its own errors |
| Tune the segmenter | boundary error rate, missed-speech and false-alarm seconds | end-to-end WER and finalisation latency | VAD accuracy is neither necessary nor sufficient for good transcripts |
| Report to an executive | a decision-shaped statement, not a metric | the interval and the coverage number | a bare percentage invites over-reading |

### 7.2 Metric properties side by side

| Metric | Unit | Denominator | Bounded | Sensitive to length | Handles fabrication | Typical use |
|---|---|---|---|---|---|---|
| WER | word | reference words | no, can exceed 1 | yes, linearly | weakly, priced as one edit | default batch text fidelity |
| CER | character | reference characters | no | yes | weakly | no-whitespace languages, spelling fields |
| SER | utterance | utterances | yes, 0..1 | no, length-blind | weakly | "will this file need review" |
| WIP | word | hypothesis words | yes, 0..1 | inversely | yes, isolated | over-generation monitoring |
| WIL | word | correct + inserted | yes, 0..1 | weakly | yes, isolated | reader-facing fabrication rate |
| field exact | slot | slots | yes, 0..1 | no | no | structured extraction gates |
| DER | second | speech seconds | no | no | not applicable | speaker attribution |
| RTF | second | audio second | no | no | not applicable | capacity planning |
| p95 latency | millisecond | none (quantile) | no | no | not applicable | streaming UX |
| ECE | probability | tokens or segments | yes | no | not applicable | routing and abstention |

The "sensitive to length" column deserves emphasis. WER weights a 60-word utterance twelve times as heavily as a 5-word one, which is right for text-volume questions and wrong for request-level questions. SER is the length-blind complement, and the pair together answers "how much text is wrong" and "how often is a request wrong" without either pretending to answer both.

### 7.3 Scorer implementation options

| Approach | Complexity | Memory | Path recoverable | When to choose |
|---|---|---|---|---|
| Full dynamic-programming table | `O(N·H)` time | `O(N·H)` cells | yes | diagnostics, small files, teaching |
| Two-row rolling DP | `O(N·H)` time | `O(min(N,H))` | no | production scoring at scale |
| Banded DP, width `2k+1` | `O(k·N)` time | `O(k)` | yes | long files with known-good pre-alignment |
| Myers bit-parallel / wavefront | `O(N·D)` time | `O(N)` bits | yes with backtrace | low-error-rate, long sequences |
| Anchor-then-align | `O(N log N)` plus local DP | small | yes | very long recordings, diarized sessions |
| Tokenise-then-hash equality | `O(N)` | `O(N)` | no | regression smoke tests only |

Concrete costs for `N = H = 4500` (a 30-minute call at roughly 150 words per minute: `150 × 30 = 4500` words): full table 20.25 million cells, 162 MB at 8 bytes per cell; rolling rows `2 × 4501 × 8 = 70.3` KB; a band of width 101 cells costs `101 × 4500 = 454,500` cells, 2.2 percent of the full table. Banded alignment is the single largest practical saving for long recordings, and it is safe exactly when the true edit distance stays below the band half-width `k`.

### 7.4 Normalization policy choices

| Policy | Effect on rate | Auditability | Recommended when |
|---|---|---|---|
| Strict verbatim, nothing removed | highest, most faithful to audio | high: no transformations to explain | measuring a decoder for research |
| Filler removal only | −1 to −3 points on conversational audio | high | product displays fillers as deleted |
| Fillers plus casefold plus punctuation | −2 to −5 points | medium | general product reporting |
| Add number and date verbalisation | −1 to −4 points on numeric domains | medium: needs a published expansion table | downstream automation consumes values |
| Add spell-equivalence for acronyms | −0.5 to −5 points, domain dependent | low: each rule is a judgement | medical or technical jargon audio |
| Aggressive semantic normalisation | unbounded | very low | never for a promotion gate |

The ranges in the second column are HEURISTIC and ILLUSTRATIVE, offered to show that normalizer choices are the same order of magnitude as the improvements teams fight over. On the twelve-utterance corpus, accepting one acronym expansion moved the rate by 0.048 — larger than the 0.032 improvement of the candidate model in 5.5. Anyone who tunes the normalizer while comparing models has lost control of the comparison.

### 7.5 Reference production strategies

| Strategy | Cost per audio hour | Reference agreement | Best for | Main risk |
|---|---|---|---|---|
| Single professional pass | baseline | unmeasured | fast, low-stakes corpora | undetected systematic error |
| Single pass plus targeted audit | +15 to 25 percent | estimated on a sample | most production sets | audit sample too small to detect a slice defect |
| Independent double pass with adjudication | +80 to 110 percent | measurable via kappa | promotion gates, regulated content | cost; adjudication latency |
| Crowd transcription with gold injection | 40 to 60 percent of professional | measurable on injected items | volume, non-critical slices | quality drift hidden by gold-item inflation |
| Synthetic reference from a strong model plus human correction | 30 to 50 percent | circular if the same model is under test | bootstrapping a new domain | confirms the model's own biases |

The last row is the one that bites. Using model-assisted transcription to build a reference for evaluating that same model measures self-consistency, not accuracy; it is acceptable only when a human corrects every segment and an independent second source checks a sample. The cost column is ILLUSTRATIVE and expressed as a fraction of the single-pass baseline, since absolute rates vary by market and language.

### 7.6 When to abandon WER entirely

WER is the wrong instrument in four situations, each with a replacement:

1. **The consumer is an intent classifier or a summariser.** Judge the downstream task; report end-to-end task accuracy with WER as a diagnostic only. A 20 percent WER with intent accuracy unchanged is not a regression.
2. **The output is a set of key-value fields.** Use field exact accuracy plus a per-field error breakdown. Token error distributes across words the consumer never reads.
3. **The output is a search index.** Use retrieval quality — query-level recall of the true passage — because insertions that a human ignores can poison an index.
4. **The product is live captions.** Use readability, measured as the share of displayed words that are both correct and stable, plus latency. A caption that is corrected 400 ms later was still read wrong.

**Takeaway.** Pick the measurement that can block the decision, then add the guardrails that catch what it cannot see; a table of metrics with no decision attached is decoration.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom as an operator would observe it, at least two competing causes that produce that symptom, a discriminating check that separates them, the correction, and a prevention. The discriminator is the part that makes the entry usable, because the symptom alone never identifies the cause.

### 8.1 Aggregate WER improves while users complain more

**Symptom.** The evaluation dashboard shows WER down from 0.143 to 0.111 across a release, while the complaints channel shows more "wrong number" reports than before.

**Competing causes.** (a) The candidate trades deletions for insertions, so corpus error falls while fabricated content rises; (b) the evaluation set was re-normalised in the same release, so part of the gain is measurement; (c) the sample is dominated by easy slices, and the hard slice that generates complaints regressed.

**Discriminating check.** Recompute `S`, `D` and `I` separately for both versions on the frozen set. If `I` rose while `E` fell, cause (a). If the normalizer version differs between the two runs, cause (b). If `I` and `E` both fell overall but the numbers slice rose, cause (c) — and the slice table shows it in one row.

**Correction.** Gate on the worst-slice rate and on the insertion rate, not on corpus WER alone.

**Prevention.** Freeze the normalizer hash in the evaluation artifact; publish the `S/D/I` split beside every rate.

**Verification.** A re-run of both versions through the identical scorer and normalizer reproduces the deltas within the interval.

### 8.2 Insertions spike on quiet recordings

**Symptom.** Files below a threshold signal level carry 3 to 5 inserted tokens per minute of silence.

**Competing causes.** (a) An energy-based voice activity detector labels room tone as speech and the decoder hallucinates plausible text; (b) the language model is over-weighted relative to acoustic evidence on low-SNR audio; (c) the audio channel is wrong (a stereo file scored as mono, so the empty channel's hiss is decoded).

**Discriminating check.** Mute the silent regions of the same files and re-score. If insertions vanish, cause (a). If they persist on real speech at low SNR, cause (b). If the per-channel energy differs by more than 20 dB, cause (c).

**Correction.** Raise the VAD threshold with hysteresis, or add a no-speech token with a tuned prior; verify channel selection in the ingest step.

**Prevention.** Add a silence-only slice to the frozen set, where the correct reference is empty and any output is an error.

**Verification.** The silence slice reports insertion count per audio hour, which should be a tracked number, not a byproduct.

### 8.3 Deletions concentrate in the first 300 ms of each segment

**Symptom.** Alignments show the first word of a segment missing on 12 to 20 percent of utterances.

**Competing causes.** (a) The segmenter cuts at the energy threshold, so the onset consonant is truncated; (b) the decoder discards its first frames while the context window fills; (c) the reference timestamps begin before the audio file's actual speech start, an offset error in the alignment metadata.

**Discriminating check.** Cross-correlate the hypothesis onset times against reference onset times. A constant negative lag points to (a); a variable lag that grows with segment length points to (b); a constant offset present in the reference itself points to (c).

**Correction.** Add 200 to 300 ms of pre-roll before the detected onset; warm the decoder with left context; fix the offset in the manifest.

**Prevention.** Report boundary error rate — the distribution of onset offsets — as a first-class metric.

**Verification.** Re-scoring after pre-roll should reduce first-token deletions without increasing insertions at the boundary.

### 8.4 A single language regresses after a multilingual adapter

**Symptom.** Spanish WER rises from 0.18 to 0.24 while the aggregate over five languages improves.

**Competing causes.** (a) Genuine negative transfer from the adapter; (b) the Spanish evaluation set changed, for example new speakers or a new channel; (c) the tokenizer changed, so Spanish contractions now split differently and the denominator moved.

**Discriminating check.** Re-score the *old* Spanish set with the new system. If the regression persists on the unchanged set, cause (a). If it disappears, cause (b). If the token counts per utterance differ between runs, cause (c).

**Correction.** Add language-specific replay data, or keep a per-language adapter; do not ship a multilingual average as a language guarantee.

**Prevention.** Freeze one immutable evaluation set per language with its tokenizer hash recorded.

**Verification.** Per-language deltas with intervals; a language whose interval excludes zero blocks promotion regardless of the aggregate.

### 8.5 Code-switching is systematically mangled

**Symptom.** In utterances where speakers alternate languages mid-sentence, error rates double and the output contains plausible words from the wrong language.

**Competing causes.** (a) The language identification runs per segment and picks one language for a mixed segment; (b) the language model's vocabulary lacks the borrowed terms; (c) the reference itself transliterates inconsistently, so alignment penalises correct output.

**Discriminating check.** Score the same audio with a forced single-language decode and compare. If forcing the dominant language performs as well as the auto path, cause (a). If the errors are concentrated on the borrowed nouns, cause (b). If two independent references disagree on the same tokens, cause (c).

**Correction.** Decode with a joint or per-token language decision; add a borrowed-term lexicon; fix the transliteration convention in the reference.

**Prevention.** Maintain a dedicated code-switch slice, never merged into a language average.

**Verification.** Slice WER with its own interval, reported beside each language's monolingual rate.

### 8.6 CER and WER rank two systems in opposite orders

**Symptom.** System A has better WER, system B has better CER on the same pairs.

**Competing causes.** (a) B makes many small character errors inside few words, which WER counts once per word; (b) A drops whole words, which costs more in WER than in CER because the missing characters may be few and short; (c) the two systems use different tokenizers for the word-level metric, so WER is not measuring the same unit.

**Discriminating check.** Examine the error-type distribution: count words with one character error versus wholly wrong words. If B's errors are mostly one-character, cause (a). If A's errors are mostly deletions of short function words, cause (b). Compare the tokenizer version strings, cause (c).

**Correction.** Choose the metric by consumer: readers care about word identity, spell-check style consumers care about characters.

**Prevention.** Report both with the tokenizer declared; never present one as a refinement of the other.

**Verification.** Reproducing both metrics from the same aligned pairs and the same normalizer.

### 8.7 A perfect score on a sentence that must never be wrong

**Symptom.** The transcript is character-identical to the reference, and the product consequence is still wrong because the reference itself was recorded incorrectly.

**Competing causes.** (a) Reference error introduced at annotation; (b) reference error introduced by an automated transcription used as a starting point; (c) the audio contains a correction ("...no, make that Tuesday") that the reference silently resolved.

**Discriminating check.** Independent second annotation of the same segment; if two humans disagree with the stored reference, cause (a) or (b). Read the raw audio around the value for cause (c).

**Correction.** Adjudicate and version the reference; keep the audio span as evidence.

**Prevention.** Double-transcribe all utterances containing critical fields; store reference provenance per segment.

**Verification.** Reference agreement (kappa) tracked per slice; a slice below 0.6 agreement cannot gate a promotion.

### 8.8 The final transcript is accurate but partial captions flicker

**Symptom.** Final WER 0.11, yet the live view rewrites the last two words several times per second.

**Competing causes.** (a) The decoder emits unstable hypotheses from a small right-context window and revises as more audio arrives; (b) the UI appends rather than replaces, so a correction appears as a duplicate; (c) chunk boundaries fall inside words, so each chunk re-emits a fragment.

**Discriminating check.** Instrument the emitted-token log with timestamps. Count revision events per emitted token. If revisions cluster at chunk ends, cause (c). If revisions happen continuously, cause (a). If the log shows one emission per token but the screen shows more, cause (b).

**Correction.** Use a stable-prefix policy: commit tokens only after they survive `k` consecutive chunks; make the UI replace by token index; align chunks to zero-crossings or use overlap-add.

**Prevention.** Track a stability metric — the share of displayed tokens never revised — as a launch gate.

**Verification.** Replay of recorded streams through the UI harness, with the revision counter as an assertion.

### 8.9 Latency p95 improves when traffic rises

**Symptom.** A capacity experiment shows p95 finalisation latency falling as concurrent requests climb.

**Competing causes.** (a) A queue-time-based scheduler drops or defers slow files, so the surviving sample is biased; (b) the client's timeout truncates long jobs, and timeouts are recorded as failures rather than as slow latencies; (c) batching increases throughput and the extra delay is small relative to the measurement noise.

**Discriminating check.** Compare the count of submitted versus completed files at each traffic level. A shortfall identifies (a) or (b); distinguishing them requires checking whether the missing files appear in an error bucket.

**Correction.** Treat timeouts as censored observations and report them as their own rate; never compute a percentile over survivors only.

**Prevention.** Assert `submitted == completed + failed + timed_out` in every latency report.

**Verification.** The identity holds under a synthetic load test at three traffic levels.

### 8.10 RTF stays below 0.4 but the backlog grows

**Symptom.** Per-file compute time is stable at 0.38 of audio duration, yet queued audio grows by roughly 40 minutes per hour of operation.

**Competing causes.** (a) Arrival rate exceeds capacity, because worker count is fixed while traffic grew; (b) retries are counted as new work, so effective arrival is higher than the dashboard's rate; (c) a subset of files has far worse RTF (very long files with quadratic alignment) and the mean hides it.

**Discriminating check.** Compute utilisation as `arrival_audio_hours_per_hour × RTF / workers`. At 24 audio-hours per hour arriving with 24 workers at RTF 0.3815, utilisation is `24 × 0.3815/24 = 0.3815` and no growth is possible; growth therefore implies the arrival figure is understated, which points to (b) or (c).

**Correction.** Count retries in arrival; cap file length or switch to banded alignment for long files; add workers.

**Prevention.** Alert on backlog seconds, not on RTF. RTF is a property of the model; backlog is a property of the system.

**Verification.** A soak test at constant arrival reproduces a flat backlog; a step increase in arrival produces the predicted slope.

### 8.11 Audio and reference segments do not line up

**Symptom.** Scores are poor but listening shows the transcript is largely right.

**Competing causes.** (a) The reference offsets were produced from a different version of the file (re-encoded, trimmed); (b) time units differ (samples versus milliseconds) in the manifest; (c) the scoring harness concatenates segments without separators, shifting word indices.

**Discriminating check.** Align by audio cross-correlation to find the offset in seconds; convert to samples at the file's sample rate and see whether the discrepancy is an exact power-of-ten multiple, which indicates (b).

**Correction.** Re-derive reference offsets from the scored file's checksum; normalise units in the manifest schema.

**Prevention.** Store a content hash of the audio beside every reference segment and refuse to score a mismatch.

**Verification.** The hash check fails loudly on a deliberately re-encoded file.

### 8.12 Speaker labels improve while WER worsens

**Symptom.** Diarization DER falls, but the concatenated per-speaker transcripts are less accurate than the undiarized baseline.

**Competing causes.** (a) Diarization boundaries cut words, so each speaker's stream contains fragments; (b) the diarizer assigns short backchannels to the wrong speaker, moving tokens between streams where the language model has less context; (c) scoring compares concatenations that reorder tokens relative to the reference.

**Discriminating check.** Score per speaker without reordering versus with reordering. If the reordered score is much worse, cause (c). If boundary-adjacent tokens carry most of the new errors, cause (a).

**Correction.** Decode with speaker-aware context and overlap handling; score cpWER-style with speaker-attributed tokens.

**Prevention.** Report diarization and recognition jointly as a speaker-attributed error metric, never as two independent improvements.

**Verification.** Both metrics move together on a held-out multi-speaker set.

### 8.13 A quiet speaker performs far worse than the cohort

**Symptom.** One speaker's utterances average 0.31 WER against a cohort 0.14.

**Competing causes.** (a) Genuine level difference (10 to 15 dB below cohort); (b) accent or dialect outside the training distribution; (c) that speaker's audio was recorded on a different device with a different codec; (d) their reference was transcribed by a different annotator with a different convention.

**Discriminating check.** Compute per-speaker mean level and codec metadata. If level explains the gap, (a). If normalising level closes it, (a) confirmed. If not, and the speaker's phone differs, (c). Compare that speaker's reference against a second annotator for (d).

**Correction.** Apply level normalisation and a gain-invariant front end; collect in-domain data; re-annotate with the standard convention.

**Prevention.** Store level, device and annotator ID per segment, and report WER by each.

**Verification.** Re-scoring with level normalisation shows the predicted improvement only for the level-driven cases.

### 8.14 The reference disagreements dominate the score

**Symptom.** The measured error rate is close to the disagreement rate between two human transcriptions of the same audio.

**Competing causes.** (a) The annotation guidelines are ambiguous (fillers, numerals, spelling); (b) the model is genuinely near-human and the metric has hit its noise floor; (c) the double transcription was done by the same annotator twice, so agreement is inflated and the true floor is higher.

**Discriminating check.** Compute Cohen's kappa on a binary "token correct" judgement across two independent annotators. On a 100-segment sample with 70 both-accept, 9 both-reject, 12 A-only and 9 B-only: `p_o = 79/100 = 0.79`; marginals give `p_e = 0.82 × 0.79 + 0.18 × 0.21 = 0.6478 + 0.0378 = 0.6856`; `kappa = (0.79 − 0.6856)/(1 − 0.6856) = 0.332`. A kappa of 0.33 means the measurement cannot support differences smaller than about 10 points.

**Correction.** Tighten guidelines, then adjudicate a stratified sample; report the noise floor beside every result.

**Prevention.** Publish `kappa` and the ceiling WER — the error rate of one annotator against the other — as part of the evaluation contract.

**Verification.** A re-annotation round with revised guidelines and a recomputed kappa above 0.6.

### 8.15 Critical names are missed despite good overall WER

**Symptom.** Corpus WER 0.09, yet 3 of 12 critical slots wrong.

**Competing causes.** (a) Out-of-vocabulary names; (b) the names are homophones the language model resolves to common words; (c) the slot extractor, not the recogniser, fails.

**Discriminating check.** Read the raw hypothesis text at the failing slots. If the name is absent or mangled in the hypothesis, cause (a) or (b) — distinguish by testing whether adding the name to a biasing list fixes it. If the hypothesis contains the name correctly and the extracted value is wrong, cause (c).

**Correction.** Add contextual biasing from the customer record; correct the extractor's schema.

**Prevention.** Score field accuracy per slot type with its own threshold; never infer field quality from WER.

**Verification.** Slot recall on a held-out set of names not in the biasing list, to detect overfitting to the bias source.

### 8.16 Privacy redaction looks successful but leaks

**Symptom.** The redaction step reports 96 percent recall, and an audit finds identifying digits in 4 percent of transcripts.

**Competing causes.** (a) Recall was measured on annotated entities the model itself proposed, a circular denominator; (b) the redaction ran on the formatted text and missed entities split by punctuation; (c) the entity list excluded categories (dates of birth, rare diagnoses) that are identifying in combination.

**Discriminating check.** Recompute recall against an independently annotated entity list, including entities the model did not propose. If recall collapses, cause (a). Check whether failing spans cross punctuation or token boundaries, cause (b). Review the category list against the privacy requirement, cause (c).

**Correction.** Annotate independently; redact after normalization on token spans; expand categories and test re-identification risk on combinations.

**Prevention.** Track redaction recall per category with an independent gold set, and treat the transcript as sensitive until audited.

**Verification.** A held-out audit set with an agreed recall floor per category, plus a manual re-identification attempt as a red team exercise.

### 8.17 Missing transcripts silently vanish from the denominator

**Symptom.** Reported WER falls from 0.14 to 0.08 in a week when nothing shipped.

**Competing causes.** (a) Files that failed to decode were dropped instead of counted; (b) a longer audio format started timing out and those files are excluded; (c) the normalizer began removing fillers, which were frequent in the excluded hard files.

**Discriminating check.** Compare the planned file count to the scored count. On the numbers in 5.3, coverage of 58 percent explains the entire movement. A coverage column answers this in one glance and nothing else does.

**Correction.** Recompute over the full planned population with imputation or an explicit undefined status.

**Prevention.** Make coverage a first-class metric that gates the report; a rate without coverage is not publishable.

**Verification.** `scored + failed + missing = planned` asserted wherever the report is generated.

**Takeaway.** Every entry above pairs a visible symptom with at least two plausible causes and a check that separates them; the metric that moves is rarely the subsystem that broke.
