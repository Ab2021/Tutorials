---
title: "Speech automatic transcription evaluation"
topic: "ASR transcription evaluation: error, timing and decision contracts"
collection: "module-18-cheat-sheets"
file_id: "03"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
prerequisites: ["Ratios and percentages", "Edit distance / dynamic programming", "Sampling and interval estimates", "Digital audio timing"]
pattern_tags: ["asr", "evaluation", "streaming", "privacy", "uncertainty"]
source_status: "OPEN — source bodies for the speech entries were not present in the accessible tree"
owner_role: "Speech evaluation curriculum maintainer"
source_index: "INDEX.md"
review_trigger: "Change to tokenization, normalization, target population, latency contract, or source material"
---

# Speech automatic transcription evaluation

## 1. File Header

**Purpose.** Give an evaluator a complete, checkable procedure for deciding whether an automatic speech recognition (ASR, also speech-to-text or STT) system is good enough for a named decision, and for diagnosing a score that moved for the wrong reason. The unit of work is not "the WER" but a contract: population, tokenization, normalization, aggregation, uncertainty, timing and cost, each written down before the number is read.

**Audience.** Speech engineers, applied scientists, evaluation owners, product managers who must sign a launch gate, and candidates preparing for speech or ML-evaluation interviews. The reader is assumed to know what a ratio is and to have seen dynamic programming once; no phonetics background is required.

**Outcome.** After working through this sheet a reader can:

- align a reference and a hypothesis by hand and by formula, and state which of substitution, deletion and insertion drove a change;
- separate a corpus-level ratio from a per-utterance average and explain when each is the honest summary;
- attach an interval to a reported error rate and say how many utterances a promised precision would require;
- evaluate a streaming system on event time rather than file time, including partial-caption behaviour and backlog growth;
- distinguish reference defects, normalization defects and model defects, because the three have different remedies;
- state the conditions under which a reported improvement is not evidence of improvement.

**Prerequisites.** Tokenization and whitespace rules; edit distance and the dynamic-programming table; the difference between a sample and a population; percentiles versus means. Module [index](../INDEX.md) lists course-level prerequisites. No unauthored lesson is a mandatory dependency for reading this card.

**Not covered here.** Acoustic modelling, language-model training, speaker verification, speech synthesis quality, diarization algorithm design, and legal determinations about recorded conversations. Diarization appears only as an evaluation surface, and downstream language-model quality is a handoff to the document-understanding cards in this collection.

**Evidence policy.** Definitions, algebra and complexity statements are DERIVED. Every number in the worked examples is ILLUSTRATIVE: the twelve utterances, their timings, their hardware and their costs were constructed so the arithmetic can be checked by a reader with a calculator. No benchmark, vendor comparison or production result is asserted anywhere in this file. Source coverage for the speech entries is OPEN; the directory crosswalk maps entries 20.11 (speech production) and 20.4 (speech ASR) to this unit, but neither source body was found, so their names bound the scope only.

**Retrieval cues.** A surprising aggregate: sections 4 and 5. A production incident: section 8. A launch decision: section 7 and the card in section 12. An interview in two days: section 11.

**Reading time.** About ninety minutes for a first pass, twenty minutes for a refresh through sections 7, 8 and 12.

**Companion material.** [NLP task contracts](./01_NLP_Task_Contracts_Classification_NER_Embeddings.md) covers classification, span extraction and retrieval evaluation; the transcription cases in `../Case_Studies/` apply the contract here to a call-centre quality problem and to a claims-document pipeline.

**Takeaway.** A transcription number is a claim about a population under a stated normalizer; the contract, not the score, is what a reviewer can audit.

## 2. Concept Map / Mental Model

### 2.1 The pipeline the metric sits inside

Every reported error rate is the output of six stages, and each stage can change the number without changing the recogniser's acoustic quality:

```text
 audio capture (device, codec, sample rate, channel count)
      |
      v
 [1] segmenter   VAD, endpointing, chunk size, overlap policy
      |
      v
 [2] decoder     acoustic model, language model, contextual biasing, adapter
      |
      v
 [3] formatter   casing, punctuation, inverse text normalisation (ITN)
      |
      v
 [4] normaliser  casefold, filler removal, number style, contraction expansion
      |
      v
 [5] aligner     reference vs hypothesis -> S, D, I and an alignment path
      |
      v
 [6] aggregator  slices, weights, intervals, cost, guardrails
      |
      v
 decision gate   promote / hold / roll back / escalate to human review
```

A score that improves because stage 4 changed is a measurement change. A score that improves because stage 1 stopped emitting empty segments is a coverage change. Only a change at stage 2 is a model change. Reviewers who cannot say which stage moved cannot say what the score means, and this single discipline resolves most of the arguments described in section 8.

### 2.2 Three clocks, never interchangeable

Transcription evaluation is unusual in that it has to reconcile three different time bases:

| Clock | What it measures | Typical field | Failure if confused with another |
|---|---|---|---|
| Event time | when the words were spoken | utterance start and end offsets in the audio | partial-caption delay is computed from file duration instead of speech onset |
| Processing time | when compute produced the token | decode timestamp, chunk completion | throughput looks fine while perceived latency grows |
| Decision time | when a user or system acts on the text | transcript availability, ticket creation | a downstream SLA is measured against the wrong milestone |

A batch service can be judged almost entirely on error. A streaming service is judged on error *and* on the delay at which each increment of text arrives, because a perfect transcript delivered after the caller hangs up has zero product value. Section 4.2 treats latency as a formula with the same rigour as error; section 5.4 works a numeric example.

### 2.3 The four-part evaluation contract

Before any number is computed, write these four lines down. They are short enough to fit in a pull-request description and they settle most disputes:

1. **Population.** Which audio: language, channel, device, domain, speaker mix, date window, inclusion rules for silence and music.
2. **Transform.** Which normalizer: casefold, punctuation, fillers, number verbalisation, spelling variants, script conversion. One implementation, referenced by version.
3. **Statistic.** Which aggregation: corpus ratio, per-utterance mean, per-slice report, or a field-level measure; and which uncertainty interval method.
4. **Decision rule.** Which threshold, on which slice, with which guardrail, at which confidence, and what happens when a guardrail fails.

The contract has a version. When the normalizer changes, the contract changes, and scores before and after the change are not comparable even if nothing about the model moved. Storing the normalizer version beside every reported number costs one column and prevents an entire class of false regressions.

### 2.4 What the metric can and cannot see

```text
 WER sees:         token substitutions, deletions, insertions vs one reference
 WER does not see: whether the meaning survived; whether a critical field changed
 audio quality:    visible only through its effect on tokens, not as a separate axis
 speaker identity: invisible unless diarization error is scored separately
 latency:          invisible; a separate family of measurements
 confidence:       invisible; requires calibration measurement
 privacy:          invisible; redaction recall must be measured on its own
 cost:             invisible; requires a cost model, section 9
```

The practical consequence is that a promotion gate made of WER alone is an incomplete gate. Section 7 gives the selection table that pairs each decision with the measurement that can actually block it.

### 2.5 Chain reasoning: find the stage, not the scapegoat

When text is wrong, the error is in one of five places: the audio (too quiet, clipped, wrong channel), the segmentation (a word split across chunks, an endpoint cut mid-word), the decoder (genuine recognition error), the formatter (a correct word rendered in the wrong convention), or the reference (the gold transcript is wrong). The evaluation method that follows from this is deliberately boring: take twenty failing utterances, put audio, reference, raw hypothesis, formatted hypothesis and normalized pair side by side, and classify. A hundred classified failures tell you which stage dominates; an aggregate score cannot.

**Takeaway.** Locate the stage before comparing models, because four of the five stages are measurement or product defects that no model change can fix.
