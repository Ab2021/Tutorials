---
title: "Speech automatic transcription evaluation"
topic: "ASR/STT accuracy, latency, coverage and privacy evaluation"
collection: "module-18-cheat-sheets"
file_id: "03"
level: "advanced"
evidence: "DERIVED"
status: "draft"
content_status: "draft"
created: "2026-11-10"
last_reviewed: null
reviewed: null
review_trigger: "Change to target population, text normalization policy, latency contract, or reference annotation guide"
notation_style: "plain-text"
prerequisites: "Basic probability, ratios, tokenization, digital audio timing"
pattern_tags: "[asr, evaluation, streaming, privacy, uncertainty]"
source_status: "OPEN"
owner_role: "Speech evaluation curriculum maintainer"
source_index: "INDEX.md"
---

# Speech automatic transcription evaluation

## 1. File Header

**Purpose:** Decide whether an automatic speech recognition (ASR) system, also called speech-to-text (STT), is good enough for a named use, and diagnose the cause when it is not. The unit of analysis is a *scored claim*, not a leaderboard row: a claim is a number plus a population, a normalization policy, an aggregation rule and an uncertainty statement.

**Audience:** Speech engineers, QA evaluators, product owners, clinical and legal reviewers, and interview candidates who can read a dynamic-programming table and a ratio. No acoustic-model mathematics is assumed; every formula below is counting and division.

**Outcome:** After working through this sheet you can (a) compute word and character error rates from aligned pairs and reproduce the arithmetic by hand, (b) choose between micro, macro and multi-reference aggregation and state what each hides, (c) attach an interval that respects utterance clustering rather than pretending words are independent, (d) instrument streaming latency so that a percentile means one thing across teams, and (e) reject a promotion whose headline gain is a composition artifact or a coverage loss.

**Prerequisites:** Tokenization, ratios with denominators, event-time versus processing-time thinking, sampling with replacement, and enough audio literacy to know that 8 kHz telephony and 48 kHz studio audio are different populations.

**Not covered:** Training or fine-tuning a recognizer; text-to-speech quality; speaker verification; diarization metric internals beyond the handoff; downstream LLM quality. Those are separate contract sheets in this unit.

**Evidence:** DERIVED for all definitions, identities and inequalities — each is proved or checked arithmetically in place. ILLUSTRATIVE for the 480-utterance corpus, the 1,200-segment latency sample, the cost figures and every threshold. OPEN for source coverage: the directory crosswalk names two source files whose bodies were not present in the accessible tree, so nothing here is attributed to them beyond topic scope.

**Retrieval cue:** Numbers that disagree → Section 4. A single example to recompute → Section 5. Two systems compared → Sections 5 and 8. A production incident → Section 8. Promotion or rollback → Section 7 and Section 12.

**How to read the formulas:** every `text` block uses the five-part anatomy — FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION. If you cannot fill WHERE and CONSTRAINT for your own metric, you do not yet have a metric; you have a number.

**Why this file exists in this form:** an earlier revision reached the required physical length by repeating a takeaway sentence. Repetition is not coverage. Every paragraph below either states a mechanism, a count, a bound, or a counterexample; if a paragraph does none of those, it was cut.

**Takeaway:** The evaluation contract — population, denominator, normalization, aggregation, uncertainty, guardrails — is the artifact under review. Any single score is downstream of it.

## 2. Concept Map / Mental Model

```text
  audio + reference + metadata            (evidence layer)
        |
        v
  split by speaker / session / site / day  -> frozen evaluation set
        |
        v
  one normalization policy, version-pinned  -> paired token streams
        |
        v
  minimum-edit alignment  -> S, D, I counts and N
        |
        +--> aggregate (micro / macro / multi-reference)
        +--> slice (channel, accent, entity, language, age band)
        +--> semantic or slot-level scoring
        |
        v
  streaming instrumentation  -> first-token, finalization, revision stability
        |
        v
  coverage + privacy accounting  -> denominators that include failures
        |
        v
  release decision: quality AND latency AND reliability AND privacy
```

The left column is measurement; the right column is decision. A defect in the left column propagates silently: a wrong denominator looks exactly like a real improvement. The model has five separations that must not be collapsed.

**Separation 1 — evidence versus annotation.** The audio is the evidence; the reference transcript is a second, fallible measurement of that evidence. When reference and hypothesis disagree, at least one of them is wrong and the scorer cannot tell which. Inter-annotator disagreement therefore sets a floor on measurable error: if two trained transcribers disagree on 2.1% of words for a domain, a system reporting 1.8% WER in that domain is reporting something at or below the noise floor of its own yardstick.

**Separation 2 — lexical distance versus meaning.** Minimum edit distance counts token changes. It does not know that "fifteen" and "15" are the same quantity, that "not" reverses a clinical instruction, or that a proper name with one wrong letter is a failed lookup. Two systems with identical WER can differ by an order of magnitude in downstream task success. Lexical scores are necessary for reproducibility and insufficient for acceptance.

**Separation 3 — offline accuracy versus interactive experience.** A batch score has no clock. A streaming service is judged on when text appears, how often it changes, and whether it disappears. Measured on the same audio, the offline and streaming configurations of one model can differ by several points of WER because streaming restricts the decoder's future context; and a configuration with better final WER can feel worse because its partial hypotheses churn.

**Separation 4 — quality versus coverage.** Error rate is conditioned on having produced output. If a system abstains, times out or crashes on hard audio, the surviving population is easier and the rate improves. Coverage — valid outputs divided by eligible requests — is not a companion metric; it is part of the accuracy denominator.

**Separation 5 — population versus sample.** The evaluation set is a sample of a declared population with a declared split rule. Utterances from one speaker are not independent draws, and utterances from one call share channel and vocabulary. Treating tokens as independent trials understates the variance of the corpus estimate, sometimes by a factor that changes a decision at the margin.

**The claim ladder.** A defensible claim climbs five rungs and may stop at any of them, but may not skip:

1. *Descriptive on the frozen set:* "On evaluation set v4 (480 utterances, 12,000 reference words) the system produced 720 word edits, WER 6.0%."
2. *Sampling-aware:* the same with an interval that accounts for clustering, e.g. 6.0% with 95% interval 5.7%–6.3%.
3. *Comparative and paired:* "System B versus A on the same utterances: −0.79 percentage points, 95% interval −1.06 to −0.53."
4. *Sliced with guardrails:* the comparison holds on every release-blocking slice, or the exception is named.
5. *Operational:* the claim survives live traffic at the deployed mix, including coverage, latency and privacy accounting.

Most disputes about speech quality are disputes about which rung a claim was on when it was spoken aloud as rung 5.

**Where each downstream consumer enters.** Product wants rung 4 with a task metric. Operations wants rung 5 plus percentile latency and coverage. Legal and privacy want the retention and redaction accounting, which is a separate denominator counting audio-minutes, not words. Research wants rung 1 with the full S/D/I decomposition so a regression can be attributed to a mechanism.

**Takeaway:** Draw the boundary between measurement and decision before choosing a metric; the boundary, not the metric, is what most disagreements are about.
