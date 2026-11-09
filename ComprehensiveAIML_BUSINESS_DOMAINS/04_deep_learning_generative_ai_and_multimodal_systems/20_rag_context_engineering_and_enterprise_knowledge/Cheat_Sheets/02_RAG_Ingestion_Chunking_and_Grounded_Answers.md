---
collection: "module-20-rag"
file_id: "02"
topic: "RAG ingestion, chunking and grounded answers"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Policy corpus, ACL, tokenizer, chunking, embedding, effective-date, or citation policy change."
notation_style: "plain-text"
prerequisites: ["01_Permission_Aware_Retrieval.md"]
pattern_tags: ["rag", "chunking", "grounding"]
source_status: "OPEN — visible source names and planning commitments only"
owner_role: "Knowledge-systems reviewer"
source_index: "INDEX.md"
---

# RAG Ingestion, Chunking and Grounded Answers

> **Purpose:** Build an eligibility-first retrieval path in which an answer can cite an authorized, version-correct source span or abstain.
> **Prerequisites:** [Permission-aware retrieval](01_Permission_Aware_Retrieval.md), tokenization, relevance labels.
> **Key outcome:** Given source versions, query time, ACL, chunks and required spans, decide eligibility, calculate a context budget, and diagnose missing evidence.
> **This file does NOT cover:** legal interpretation, vendor-specific APIs, or measured answer quality.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED accounting; hand-checked illustrative trace; source-body and deployment evidence open.
> **Retrieval cue:** If a fluent answer cites the wrong policy version, start at Sections 2, 4 and 8.

## 1. File Header

Retrieval-augmented generation is a pipeline, not a permission bypass: ingest source documents with provenance; determine eligibility at the decision time; retrieve/filter/rerank; assemble bounded context; generate only claims supported by cited spans or abstain. Source plans include legal-contract analysis and retrieval pipelines, but no inaccessible source body is implied.

**Takeaway:** Authorization and temporal eligibility are preconditions to relevance.

## 2. Concept Map / Mental Model

```text
source -> parse/OCR -> chunk + source/version/effective interval/ACL -> index
query + principal + decision time -> eligibility filter -> retrieve -> rerank -> context budget
context + claim rules -> cited answer | abstain | human escalation
logs: source IDs, chunk IDs, filter decisions, model/prompt/index version
```

**Takeaway:** Log rejected as well as selected sources; absence of a result can be a correct security outcome.

## 3. Foundational Definitions

| Term | Meaning | Type | Boundary |
|---|---|---|---|
| chunk | retrievable source segment with provenance | text + metadata | boundaries can break required evidence |
| eligibility | authorization and version-time predicate | boolean | evaluated before rank |
| effective time | when policy applies | timestamp/interval | distinct from recorded/published time |
| citation | pointer to exact supporting source span | source ID + offsets | not merely a URL |
| abstention | refusal/escalation due to insufficient eligible support | action | must be evaluated, not hidden |

**Takeaway:** A vector has no authority; metadata and policy evaluation do.

## 4. Core Formulas with Full Derivation

Validity: context window C includes prompt/control tokens P and output reserve O, leaving retrieved-token budget R. This is conservation of the fixed context capacity.

```text
FORMULA [RAG-01]: R = C - P - O
WHERE: R retrieved-context budget [tokens]; C model context limit [tokens]; P prompt/control tokens; O reserved completion tokens.
CONSTRAINT: C,P,O use the actual tokenizer; R>=0; safety/citation instructions are included in P.
NUMERICAL EXAMPLE: C=8192, P=1250, O=1000 -> R=5942 tokens.
INTERPRETATION: chunks totaling 6,500 tokens cannot be inserted unchanged.
```

Validity: every required span for a question has a gold span and the evaluation declares success only if all required spans are retrieved and eligible.

```text
FORMULA [RAG-02]: all-span success = (# questions with every required eligible span retrieved)/(# evaluable questions)
WHERE: numerator/denominator are question counts; result dimensionless.
CONSTRAINT: gold spans, filters, k, corpus snapshot and time/principal are frozen.
NUMERICAL EXAMPLE: 18 of 24 questions retrieve every required span -> 18/24=.75.
INTERPRETATION: this is stricter than per-span recall and exposes multi-clause failure.
```

Counterexample: high top-k lexical/vector recall can still return a revoked or future policy. It must not count as usable support.

**Takeaway:** Context budgeting and all-span success force the correct unit of evaluation.

## 5. Worked Numerical Example

**ILLUSTRATIVE, hand-checked.** A principal asks on 2026-01-15 about a policy. The index holds source A effective through 2025-12-31 and source B effective from 2026-01-01; both share keywords. Filtering by principal and effective date must remove A before retrieval. With R=5,942 tokens, a proposed 8-chunk assembly of 900 tokens each totals 7,200, so it exceeds budget by 1,258 tokens. Truncating every chunk may remove a decisive exception; instead retrieve smaller provenance-preserving chunks or select evidence by claim coverage and then verify each claim.

Sensitivity: reserve O=2,000 rather than 1,000 and R falls to 4,942. A system that changes output reserve but keeps the same chunks has changed its retrieval condition; rerun all-span evaluation.

**Takeaway:** A context budget change can change evidence coverage even when the index is unchanged.

## 6. Visual / Geometric Intuition

```text
eligible set E(query principal,time) -> rank only E
not: retrieve all -> rank -> hide unauthorized result

answer claim ----citation----> exact chunk/span ----> source version/effective interval
```

**Takeaway:** Filter-first prevents an unauthorized document from influencing rank, context, or answer wording.

## 7. Comparison / Selection Table

| Choice | Use when | Benefit | Disqualifier | Fallback |
|---|---|---|---|---|
| fixed-size chunks | uniform prose and simple recall task | predictable budget | boundaries split clauses | structure-aware chunks |
| structure-aware chunks | headings/tables/clauses matter | preserves meaning/provenance | parser quality unknown | evaluate parse fixtures |
| dense retrieval | semantic paraphrase matters | semantic candidates | ACL/version filtering absent | filtered hybrid baseline |
| hybrid + rerank | exact terms and semantics both matter | complementary signals | latency budget fails | measure filtered exact baseline |
| abstain/escalate | no eligible complete support | prevents unsupported answer | escalation owner missing | define review queue |

**Takeaway:** Select chunking against all-required-span and authorization tests, not anecdotal fluent answers.

## 8. Failure Modes & Edge Cases

| Failure | Competing causes | Diagnostic | Correction | Verification |
|---|---|---|---|---|
| wrong-version citation | effective-time bug or metadata absence | boundary-date replay | versioned intervals + reject incomplete metadata | dated fixture suite |
| fluent unsupported answer | missing chunk or generation claim drift | claim-to-span audit | abstain/ask narrower question | unsupported-claim rate |
| ACL leak | filter after rank or stale cache | cross-principal negative test | eligibility before cache key/rank | no-leak test |
| chunk loses exception | boundary policy or tokenizer truncation | required-span trace | structural split/overlap with provenance | all-span success |
| migration regression | changed embedding/index/corpus | dual-index replay | pin versions and rollback | per-version evaluation |

**Takeaway:** A retrieval failure is diagnosable only with query-time provenance and filter logs.

## 9. Complexity & Cost Analysis

Exact scoring of M vectors of width D is O(MD) per query before filters; approximate indexes trade recall, memory and latency. Context assembly cost is bounded by R tokens but can include parsing/reranking. These are algorithmic descriptions, not benchmark claims. Store only approved chunks and measure index/storage/egress and reviewer workload on the target corpus.

**Takeaway:** Do not compare retrieval methods without identical filters, corpus snapshots and budget.

## 10. Key Theorems / Results Reference

- RAG is a design pattern; relevance does not entail groundedness or entitlement.
- [Lewis et al., *Retrieval-Augmented Generation*](https://arxiv.org/abs/2005.11401), 2020, is a primary research reference; reviewed 2026-11-09.
- Permission logic and effective-date semantics require the organization’s authoritative policy, which is OPEN here.

**Takeaway:** Citation is evidence plumbing, not a guarantee of correctness.

## 11. Interview / Exam Questions

1. **Calculate R for C=4096,P=800,O=600.** 2,696 tokens.
2. **Why filter before rank?** Unauthorized/wrong-version content must not influence candidates or caches.
3. **Why is per-span recall insufficient?** A multi-part answer can be unusable if one required span is absent.
4. **What is the first wrong-version test?** Boundary-date replay with frozen effective intervals.
5. **What does a citation need?** Exact span, source/version and query-time eligibility basis.
6. **Changed constraint: output reserve doubles.** Recompute R and rerun evidence-coverage tests.
7. **Why can chunk overlap be risky?** Duplicate context can consume budget and amplify one source.
8. **How detect ACL cache leakage?** Cross-principal negative tests using identical query keys.
9. **Wrong fix for unsupported answer?** Increase k without checking eligibility or required spans.
10. **When abstain?** Eligible evidence is missing, contradictory, incomplete, or policy requires human ownership.

**Takeaway:** The right answer connects a retrieval metric to a safe answer action.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Eligibility | principal + source ACL + effective/recorded time before rank |
| Budget | RAG-01 with actual tokenizer and output reserve |
| Evidence | citations point to exact eligible spans/version |
| Evaluation | RAG-02, filtered recall, wrong-version/ACL negative tests |
| Fallback | abstain/escalate when required support is incomplete |
| Avoid | post-rank filtering and relevance-as-grounding claims |
| Evidence status | illustrative hand arithmetic; policy data and deployment results OPEN |
