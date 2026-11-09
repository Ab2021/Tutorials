---
collection: "module-18-nlp-speech-documents"
file_id: "01"
topic: "NLP task contracts: classification, NER, embeddings"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Label ontology, language mix, entity schema, retrieval corpus, or human-review capacity changes."
notation_style: "plain-text"
prerequisites: ["Grouped evaluation splits and calibration basics"]
pattern_tags: ["nlp", "classification", "ner", "embeddings"]
source_status: "OPEN — original source bodies unavailable"
owner_role: "NLP reviewer"
source_index: "INDEX.md"
---

# NLP Task Contracts: Classification, NER and Embeddings

> **Purpose:** Choose output grain and evaluation before treating text labels, spans, or embeddings as interchangeable.
> **Prerequisites:** Probability, grouped/time splits, token boundaries.
> **Key outcome:** Given a task and counts, compute precision/recall; identify a span, language, or retrieval evaluation that would be invalid.
> **This file does NOT cover:** ASR acoustics, OCR geometry, LLM generation, or an empirical model comparison.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED metrics; hand-checked illustrative arithmetic; no model outcome asserted.
> **Retrieval cue:** If an aggregate score hides a dangerous language or entity error, start at Section 8.

## 1. File Header

Classification assigns document/utterance labels. Named-entity recognition (NER) assigns typed spans. Embeddings map inputs into a vector space for similarity, clustering, or retrieval. The module source additions include multilingual/tokenizer and speech commitments; visible original bodies were not available, so only the visible topic is taught.

**Takeaway:** Output grain determines labels, loss, evaluation, review workflow, and failure cost.

## 2. Concept Map / Mental Model

```text
text + availability time -> normalise/tokenise -> classifier | span tagger | embedding encoder
                                        |                         |              |
label/annotation contract -> split by author/case/time/language -> metric + slice -> threshold/review
```

**Takeaway:** Preserve raw text, normalization version, language/script and annotation schema for audit.

## 3. Foundational Definitions

| Term | Meaning | Type | Boundary |
|---|---|---|---|
| document label | one or more categories for a declared unit | categorical | prevalence and hierarchy stated |
| entity span | start/end offsets plus type | token/character interval | tokenization/offset convention fixed |
| embedding | numerical representation | vector [D] | similarity metric and corpus version stated |
| false positive | predicted item absent from gold under protocol | count | review/action cost required |
| temporal split | train before evaluation time | partition | feature availability must match inference |

**Takeaway:** A label is only meaningful at its unit, timestamp, and annotation policy.

## 4. Core Formulas with Full Derivation

Validity: a frozen gold set and a matching rule have produced TP, FP, FN. Precision conditions on predictions; recall conditions on gold positives.

```text
FORMULA [NLP-01]: precision=TP/(TP+FP); recall=TP/(TP+FN)
WHERE: TP/FP/FN counts under declared class/span matching; results dimensionless.
CONSTRAINT: denominators nonzero; state micro/macro aggregation and slice support.
NUMERICAL EXAMPLE: TP=36, FP=12, FN=9 -> precision=.75, recall=.80.
INTERPRETATION: one quarter of alerts are false and one fifth of gold positives are missed.
```

For an embedding retrieval query q with unit-normalised vectors, cosine score is q dot d. Validity requires same encoder/version and nonzero vectors; high similarity is not evidence of factual support or permission.

Counterexample: exact-span NER can score zero when a prediction has the right text but wrong boundary; that may be appropriate for structured extraction and too harsh for another workflow. Declare the rule.

**Takeaway:** Metrics are consequences of matching and aggregation rules, not universal facts.

## 5. Worked Numerical Example

**ILLUSTRATIVE, hand-checked.** A claims-routing classifier on a held-out, time-separated set produces TP=36, FP=12, FN=9 for a high-cost class: precision 0.75 and recall 0.80. If 1,000 daily messages have the deployment prevalence and the threshold produces 100 alerts, 25 false alerts under this illustrative precision estimate can consume reviewer capacity; the actual alert rate must be measured on a representative shadow workload. A random split would be invalid if repeated customer cases appear across partitions.

Sensitivity: if one low-resource language has TP=2, FP=2, FN=6, its recall is .25 despite aggregate .80. Do not promote solely on aggregate results.

**Takeaway:** Slice support and workflow capacity change the decision implied by the same aggregate score.

## 6. Visual / Geometric Intuition

```text
classification: [document] -> [label]
NER:            [token token ENTITY ENTITY token] -> [span, type]
retrieval:      query vector ---> near vectors, then inspect source/provenance
```

**Takeaway:** Embedding proximity proposes candidates; it does not validate a claim or authorization.

## 7. Comparison / Selection Table

| Output | Use when | Avoid when | Validation |
|---|---|---|---|
| classification | one disposition changes workflow | evidence span is required | class/language/time slices |
| NER | fields/spans feed structured action | schema is ambiguous/unadjudicated | exact/relaxed span policy |
| embeddings | candidate retrieval/clustering needed | score is treated as answer | recall with corpus/version/ACL |
| human review | error cost or uncertainty is high | queue/cues undefined | workload and adjudication audit |

**Takeaway:** Use the simplest output that preserves the decision evidence.

## 8. Failure Modes & Edge Cases

| Failure | Competing causes | Check | Fix | Verify |
|---|---|---|---|---|
| aggregate success, language failure | sampling or tokenizer/label mismatch | per-language support/recall | collect/adjudicate, scoped route | language gate |
| NER offset mismatch | normalization or annotation error | raw-to-token round trip | canonical offsets | replay fixture |
| retrieval leak | corpus/version/permission contamination | query-time eligibility trace | filter before rank | negative ACL test |
| high FP review burden | threshold or prevalence shift | alert-volume simulation | capacity-aware threshold | shadow queue |

**Takeaway:** Diagnose data contract defects before model replacement.

## 9. Complexity & Cost Analysis

A linear classification head costs O(DK) per representation for D features and K classes; exact nearest-neighbour scoring against M vectors costs O(MD) per query before indexing. These are algorithmic counts, not latency. Index recall, filters, reranking, tokenization and I/O must be measured separately.

**Takeaway:** A faster vector lookup can still be wrong if it searches an ineligible corpus.

## 10. Key Theorems / Results Reference

- Precision/recall depend on a stated classification and matching rule.
- [Tjong Kim Sang and De Meulder, CoNLL-2003](https://aclanthology.org/W03-0419/) documents a NER evaluation context; reviewed 2026-11-09.
- [Reimers and Gurevych, Sentence-BERT](https://aclanthology.org/D19-1410/) documents a sentence embedding method, not local quality.

**Takeaway:** Primary sources explain methods; they do not certify a target workflow.

## 11. Interview / Exam Questions

1. **Why is micro F1 insufficient alone?** It can hide rare-class/language failure.
2. **Compute recall for TP=8,FN=2.** .80.
3. **Why preserve offsets?** A reviewer must locate the claimed evidence in the original text.
4. **What makes a temporal split valid?** All inference features precede the decision time.
5. **Why cannot cosine score prove support?** Similarity is not entailment, provenance, or authorization.
6. **Changed constraint: reviewers halve.** Recompute threshold/abstention and measure queue impact.
7. **First test for span drift?** Raw-text/token/offset round-trip fixtures.
8. **Wrong fix for low-resource language?** Hiding it inside an aggregate metric.
9. **When prefer classification to NER?** When evidence location does not change action.
10. **What should an embedding evaluation state?** Corpus, version, query set, relevance rule, filters and recall metric.

**Takeaway:** Answers should make the evaluation contract auditable.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Task grain | label vs typed span vs candidate vector |
| Metric | NLP-01 only after matching/aggregation rules are frozen |
| Split | group/time/language by deployment dependency |
| Retrieval | filter eligibility before interpreting similarity |
| Diagnostics | offsets, slices, corpus version, queue capacity |
| Avoid | aggregate-only promotion and score-as-evidence reasoning |
| Evidence | illustrative hand arithmetic; no measured system claim |
