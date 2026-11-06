---
collection: nlp-llm-playbook
file_id: "13"
topic: "RAG retrieval, chunking, permissions and evidence"
level: 2
status: draft
version: "0.1"
created: "2026-11-07"
reviewed: null
review_trigger: "Changes to corpus, ACL semantics, retrieval models or evaluation population"
notation_style: plain-text
prerequisites: ["tokenization, vectors, probability, ranking, access control"]
pattern_tags: [retrieval, grounded-generation, authorization]
source_status: OPEN
owner_role: knowledge-platform maintainer
source_index: INDEX.md
---

# Retrieval-augmented generation: evidence before answers

## 1. File Header

**Reader:** search, ML, security and product engineers designing evidence-grounded question answering.
**Decision:** select an authorized, useful and appropriately sized context, then evaluate the complete answer workflow.
**Outcome:** trace a question through ingestion, chunking, retrieval, filtering, ranking, context assembly, generation and citations.
**Scope:** text documents; multimodal parsing and agentic tool execution are adjacent topics, not assumed capabilities.
**Teaching home:** [NLP and LLM Playbook](../INDEX.md); [source crosswalk](../../../../00_SOURCE_TOPIC_CROSSWALK.md).
**Provenance:** the directory listing names the source sheet, but its contents have not been inspected here.
**Status:** draft, unreviewed; this is not certification of implementation, security or production outcomes.
**Evidence vocabulary:** ILLUSTRATIVE means invented inputs; DERIVED means consequences of stated definitions; OPEN means verification still required.
**Prerequisites:** understand tokens, basic vector arithmetic, ranked lists, denominators and document permissions.
**Source registry:** Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020, version and access review OPEN: supports the conceptual retrieval-plus-generation model, not any benchmark here.
**Source registry:** Robertson and Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), 2009, source inspection OPEN: reference for lexical ranking, not a measured local implementation.
**Source registry:** NIST, [Zero Trust Architecture, SP 800-207](https://doi.org/10.6028/NIST.SP.800-207), 2020, source inspection OPEN: background on authorization boundaries, not proof that an implementation is secure.
**Source registry:** OWASP, [Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/), version and source inspection OPEN: consult current injection guidance before deployment.
**Evidence boundary:** all workloads, values, incidents and operating thresholds in this sheet are constructed teaching examples, not observed results.
**Takeaway:** a document appearing in a prompt is neither proof of authorization nor proof that the answer follows from it.

## 2. Concept Map / Mental Model

A user asks a question under a principal and a request time.
A policy decision identifies which source objects that principal may access.
The ingest path extracts source content, records provenance and versions, and forms chunks.
Indexes associate chunks with lexical terms, vectors, metadata and authorization identifiers.
The query path retrieves candidate chunks within an authorized scope.
A reranker orders eligible candidates using a more expensive relevance signal.
Context selection trades relevance, coverage, redundancy, source authority and token budget.
Generation treats retrieved text as untrusted evidence, not instructions.
A response associates material claims with resolvable, authorized citations.
An abstention or escalation is a valid outcome when evidence is missing or conflicting.
Evaluation separates corpus coverage, retrieval, selection, claim support and task usefulness.
The offline index and the online permission service may update at different speeds.
A secure design must account for that gap rather than relying on old metadata alone.
The index version, embedding version, policy decision and prompt construction need traceability.
The learner should locate the first stage where an expected eligible fact disappears.
That first disappearance directs a fix more reliably than changing the generator blindly.
**Takeaway:** inspect the pipeline as a chain of conditional decisions, not as one opaque model score.

## 3. Foundational Definitions

A **document** is a source object with an owner, identifier, version and lifecycle.
A **chunk** is an indexed evidence unit derived from a document, with source offsets or another stable locator.
A **token** is a unit produced by a particular tokenizer; token counts are not interchangeable across tokenizers.
**Chunk overlap** repeats source content between consecutive units to preserve boundary context.
A **candidate** is an item returned by an initial retrieval stage before final context selection.
**Sparse retrieval** matches token or term features, often using a lexical scoring function.
**Dense retrieval** compares vector representations of a query and chunks in an embedding space.
**Hybrid retrieval** combines outputs or scores from distinct retrieval mechanisms.
A **reranker** estimates relative relevance for candidates; it cannot recover an item absent from the candidate pool.
A **context window** is the model's total input and output capacity under its actual serving contract.
A **context budget** is the remaining allowance after system instructions, question, output reserve and other overhead.
A **citation** is a reference from a claim to an identifiable source span, not a guarantee of entailment.
**Entailment** asks whether a cited passage supports a specific claim under its scope and date.
**Relevance** asks whether a passage helps answer the question; relevant material can still be obsolete or unauthorized.
**Groundedness** concerns support for generated claims from supplied evidence; definitions vary by evaluation protocol.
**Authorization** is a policy decision for a principal, resource, action and relevant context at a specified time.
**Recall at k** measures the share of labeled relevant eligible items represented in the first k results.
**Precision at k** measures the share of returned items in the first k that are labeled relevant.
**Hit rate at k** records whether at least one relevant item appears in the first k; it is not recall.
**Candidate recall** is measured before reranking; final context recall is measured after packing.
**ACL** means access-control list; an ACL attached to a document is not necessarily the full effective policy.
**ANN** means approximate nearest neighbor search, which may miss true nearest vectors for speed.
**RRF** means reciprocal rank fusion, a method that combines ranks rather than uncalibrated raw scores.
**MRR** means mean reciprocal rank, emphasizing the rank of the first relevant result.
**Gold labels** are adjudicated evaluation judgments; they can be incomplete or stale.
**Takeaway:** define the unit, eligible population and denominator before comparing retrieval scores.

## 4. Core Formulas with Full Derivation

### 4.1 Base: relevance counts

Validity: binary relevance labels must be defined for the eligible evaluation corpus at the chosen time.
Let the retrieved set have no duplicate item identifiers and contain at most k items.
Let R be the number of relevant eligible items and H the number retrieved among the first k.
Counting relevant returned items and dividing by the total relevant eligible items yields recall.
Dividing the same hits by the returned count n yields observed precision.
```text
FORMULA [R1]: recall@k = H / R; precision@k = H / n
WHERE: H, R, n are integer item counts; 0 <= H <= min(R,n); 0 < n <= k.
CONSTRAINT: R > 0; label eligibility and item granularity are held fixed.
NUMERICAL EXAMPLE: R=4, n=3, H=2: recall=0.5; precision=2/3.
INTERPRETATION: two of four eligible relevant items were found; two of three returns are relevant.
```
The dimensional check is count divided by count, hence both metrics are dimensionless.
An independent count check enumerates two hits, two missed relevant items and one nonrelevant return.
If R is zero, recall is undefined; report a separate no-answer slice rather than manufacturing a zero.
If n is zero, this observed-return precision is undefined; declare any alternative fixed-k convention explicitly.
If labels omit relevant documents, measured precision can underestimate true precision.
If labels include forbidden evidence, measured recall unfairly rewards a permission violation.

### 4.2 General: chunk count and token budget

Validity: fixed length windows, fixed tokenization, no structural breaks and positive stride are assumed.
Let document length L and chunk length C be positive integer tokens, and overlap O satisfy 0 <= O < C.
Each new chunk after the first advances C-O tokens.
For L greater than C, cover the remaining L-C tokens with enough strides, rounding up.
```text
FORMULA [R2]: N = 1 + ceil(max(0,L-C)/(C-O))
WHERE: N chunk count; L,C,O integers in tokens; ceil rounds upward.
CONSTRAINT: 0 <= O < C and nonempty document; final chunk may be shorter.
NUMERICAL EXAMPLE: L=1000, C=300, O=50: N=1+ceil(700/250)=4.
INTERPRETATION: starts at token offsets 0, 250, 500 and 750 cover the document.
```
Independent interval check: half-open windows [0,300), [250,550), [500,800), [750,1000) cover all 1000 tokens.
Their lengths sum to 300+300+300+250=1150 tokens, including 150 repeated tokens.
A structure-aware chunker may choose different boundaries and therefore does not obey this count exactly.
With O=0 the stride equals C; as O approaches C, chunk count and repeated storage increase sharply.
O=C gives zero stride and is invalid, rather than an infinite executable loop.

Validity: a serving endpoint enforces a combined limit on input and output tokens.
Let W denote the window, and S, Q, A and G denote instructions, question, other overhead and output reserve.
Subtracting each non-evidence component from the window gives an upper bound for evidence.
```text
FORMULA [R3]: B = W - S - Q - A - G
WHERE: all quantities are nonnegative token counts for the deployed tokenizer.
CONSTRAINT: B >= 0; include message framing and tool schema overhead in A.
NUMERICAL EXAMPLE: W=8192, S=700, Q=150, A=342, G=1000: B=6000 tokens.
INTERPRETATION: selected evidence must fit in 6000 tokens, not the full model window.
```
Independent sum: 700+150+342+1000+6000=8192.
If the endpoint separately caps output or input, enforce those additional limits.
A larger context budget does not prove better evidence use or lower hallucination.
Reserve room for delimiters, citation labels and system-specific framing using actual tokenization.

### 4.3 Variants: fusion and vector similarity

Validity: each retriever contributes at most one rank per item and ranks start at one.
A rank contribution decays as rank grows; summing contributions yields RRF.
```text
FORMULA [R4]: score(d) = sum_j 1 / (K + rank_j(d))
WHERE: d is an item; j indexes rank lists containing d; K >= 0 is a chosen constant.
CONSTRAINT: ranks are positive integers; absent items contribute zero; K is not top-k.
NUMERICAL EXAMPLE: K=60, ranks of d are 1 and 3: 1/61+1/63 ≈ 0.032266.
INTERPRETATION: agreement across lists can outweigh a single high lexical or dense rank.
```
Check using a common denominator: (63+61)/(61*63)=124/3843 ≈ 0.032266.
An item appearing at rank 1 in just one list scores approximately 0.016393.
Adding more lists changes the score scale; compare only under a fixed fusion configuration.
A larger K flattens rank differences and increases the effect of list agreement.
RRF is not a calibrated probability of relevance and does not enforce authorization.

Validity: cosine similarity is defined only for nonzero vectors in the same embedding space.
The dot product divided by the product of vector norms removes positive magnitude scaling.
```text
FORMULA [R5]: cosine(q,d) = (q dot d) / (norm(q) * norm(d))
WHERE: q,d are nonzero real vectors of identical dimension; cosine is dimensionless.
CONSTRAINT: embeddings must use compatible model, preprocessing and dimensions.
NUMERICAL EXAMPLE: q=(1,0), d=(3,4): cosine=3/(1*5)=0.6.
INTERPRETATION: similarity reflects angular alignment, not factual correctness.
```
Check: norm(d)=sqrt(9+16)=5 and the dot product is 3.
A zero vector has no defined cosine; a change in embedding model can invalidate comparisons.
For unit-normalized vectors, dot product equals cosine; this is not true without normalization.
Cosine cannot establish whether a document is current, authoritative or accessible.

### 4.4 Limits, counterexamples and invariants

Validity: a permission decision must be based on effective policy and must be enforced before content leaves a trust boundary.
**Preconditions:** authenticated principal, identified resource, action, policy version and request context.
**State transition:** evaluate permission, retrieve only eligible IDs where possible, then recheck before content disclosure.
**Invariant:** every chunk sent to a reranker, model, log, cache or user is authorized for that receiving boundary.
**Justification:** a later response filter cannot undo disclosure to an external model or log sink.
**Worked trace:** Alice can read A but not B; a similarity result containing both must expose only A downstream.
**Revocation trace:** if access to A is revoked after indexing but before assembly, current authorization denies A.
**Failure boundary:** if policy state is unavailable or freshness cannot meet the required contract, fail closed or use an explicitly authorized safe fallback.
**Termination:** a denied or unavailable decision produces an abstention or access-safe error, not an unrestricted retry.
A post-filtered top-k search can also lose relevant eligible candidates because forbidden results consumed the candidate slots.
Overfetching can mitigate candidate loss but never substitutes for authoritative permission enforcement.
**Takeaway:** retrieval quality and data authorization are independent conditions that must both hold.

## 5. Worked Numerical Example

**ILLUSTRATIVE scenario:** a support assistant answers an employee's question about an expense policy.
The data are invented and do not describe an actual organization.
The employee may access documents A, B and C, but not private document D.
Document A has four 300-token chunks from the R2 interval example.
Chunk A1 describes eligibility, A2 contains an old threshold, A3 states the current threshold and A4 explains exceptions.
Document B contains a current approval workflow, and C is an unrelated but accessible FAQ.
Private D happens to contain a similar phrase, but must not be sent to a downstream processor.
Assume effective policy and source versions are checked at request time.
The gold set for this question contains A3 and B1, two eligible relevant chunks.
The lexical ranking within eligible scope is A2, A3, B1, C1.
The dense ranking within eligible scope is B1, A3, C1, A2.
With K=60, R4 gives A3 the score 2/62 ≈ 0.032258.
B1 has score 1/63+1/61 ≈ 0.032266.
A2 has score 1/61+1/64 ≈ 0.032018.
C1 has score 1/64+1/63 ≈ 0.031498.
The fused order is B1, A3, A2, C1.
The small difference between B1 and A3 is a rank-fusion artifact, not calibrated confidence.
At k=2, both gold chunks appear, so R1 recall is 2/2=1.
At k=2, precision is 2/2=1 under the stated binary labels.
At k=1, recall is 1/2=0.5 and precision is 1/1=1.
The k=1 result is precise but omits a second necessary piece of evidence.
An answer needing both the policy threshold and the approval workflow therefore cannot rely on B1 alone.
A2 is topically similar but obsolete; source version and effective date matter.
The reranker is given B1, A3, A2 and C1, not D.
Assume it assigns A3 and B1 the highest positions after applying a freshness-aware selection rule.
Suppose their source text uses 300 and 250 tokens respectively.
Citation labels and boundaries consume another 30 tokens.
The resulting 580-token evidence package fits within the 6000-token budget of R3.
The prompt can include both sources without assuming that unused capacity should be filled.
The answer should cite A3 for the threshold and B1 for the workflow.
The answer should not cite B1 for a threshold that B1 does not state.
If A3 has conflicting effective dates, the assistant should disclose the conflict or escalate.
If B1 is revoked between retrieval and assembly, the recheck removes it.
Removing B1 changes the answerability of the workflow component, not merely its ranking.
A safe response then answers only the supported threshold and says the approval process is unavailable.
Alternatively, a product may require complete answers and abstain from the whole response.
The correct behavior is specified by the product's decision policy, not by the cosine score.
**Arithmetic cross-check:** A3's two rank-2 terms total 0.032258; B1's ranks 1 and 3 total 0.032266.
**Sensitivity:** increasing k from 1 to 2 increases recall from 0.5 to 1 in this constructed query.
**Boundary:** another question may have only one relevant chunk, so this example does not prove k=2 is generally optimal.
**Takeaway:** good ranking, source freshness, permissions and claim-specific support must all survive context assembly.

## 6. Visual / Geometric Intuition

```text
Source -> extract -> version -> chunk -> index
Identity -> current policy -> eligible candidate scope
Question -> lexical and dense candidates -> fuse -> rerank
Eligible + current + useful spans -> budgeted context -> model
Claims -> source-span support checks -> answer or abstain
```
In a vector plane, direction expresses similarity when cosine is used.
Two near directions indicate similar embeddings, not identical factual content.
A relevant exception can lie farther away than a generic policy paragraph.
Similarity neighborhoods can contain both current and superseded policy versions.
Authorization is a separate partition imposed by policy, not a geometric distance.
A forbidden nearest neighbor remains forbidden even if its similarity is highest.
Think of selection as packing evidence units into a constrained container.
A large duplicate can crowd out a small passage covering an essential exception.
A citation edge connects an answer claim to a passage and its source version.
An edge without entailment is merely a pointer, not support.
A source can support one claim and fail to support a neighboring claim.
A missing edge should be visible to reviewers rather than hidden by a general bibliography.
**Takeaway:** distance, eligibility and entailment are three different relations.

## 7. Comparison / Selection Table

| Decision | Candidate | Use when | Disqualifier / boundary |
|---|---|---|---|
| Chunk boundary | fixed windows | stable token budget and simple baseline | tables and clauses split across windows |
| Chunk boundary | structure-aware | headings and paragraphs survive extraction | unreliable structure or huge sections |
| Chunk boundary | parent-child | short search units need larger readable context | parent expansion violates ACL or budget |
| Retrieval | lexical | exact identifiers, codes, named entities | synonyms and vocabulary mismatch |
| Retrieval | dense | semantic paraphrases matter | opaque similarity and incompatible embeddings |
| Retrieval | hybrid | exact and semantic queries both occur | fusion and candidate costs exceed budget |
| Reranking | cross-encoder | small candidate pool and latency headroom | forbidden content sent to unauthorized service |
| Context | relevance-only | simple baseline with low redundancy | misses diversity, freshness or coverage |
| Context | coverage-aware | multiple facts and limited context | value estimates are unreliable |
| Authorization | prefilter | engine enforces effective eligible scope | stale index policy or unsupported filtering |
| Authorization | final recheck | revocation or policy changes are possible | recheck alone cannot undo earlier disclosures |
| Evaluation | chunk labels | diagnosing retriever and chunker | labels invalidated by boundary changes |
| Evaluation | source-span labels | comparing chunkers at stable source grain | adjudication and mapping require effort |
| Missing evidence | partial answer | individual supported claims remain useful | product requires a complete decision |
| Missing evidence | abstain | unsupported answer has high cost | poor UX if a useful safe partial answer exists |

Prefer the simplest method that meets measured quality, permission and latency requirements.
The table gives options, not a universal model ranking.
A reranker cannot fix an extraction omission or inaccessible evidence.
A source-span test set is more stable across chunking changes than a fixed list of chunk IDs.
Exact term retrieval deserves a baseline even when dense vectors are available.
Hybrid gains must be demonstrated on the actual permission-aware query distribution.
Candidate pool sizes and context budgets are workload-dependent, not defaults established here.
**Takeaway:** choose methods by constraints and evaluated failure slices, not by fashion.

## 8. Failure Modes & Edge Cases

All entries below are hypothetical diagnostic exercises, not incident reports.
Each provides competing causes, a discriminating signal, a bounded correction and a verification step.
Do not paste sensitive passages into debugging logs merely to investigate them.

### Diagnostic 01: Missing PDF text

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Missing PDF text.
**Candidate cause:** image-only page.
**Competing explanation:** parser dropped a text layer.
**Discriminating check:** compare rendered page with extracted text.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** run OCR with page provenance.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** re-query an annotated page.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 02: Misordered paragraphs

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Misordered paragraphs.
**Candidate cause:** column order lost.
**Competing explanation:** document actually has unusual sequence.
**Discriminating check:** compare bounding boxes with reading order.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** use layout-aware extraction.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** inspect citations spanning columns.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 03: Table answer wrong

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Table answer wrong.
**Candidate cause:** headers separated from cells.
**Competing explanation:** source itself contains conflicting tables.
**Discriminating check:** compare extracted rows against rendered table.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** retain row and header together.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** check question on a row with shared heading.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 04: Repeated headers dominate

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Repeated headers dominate.
**Candidate cause:** boilerplate indexed in every chunk.
**Competing explanation:** legitimate repeated warnings carry meaning.
**Discriminating check:** compare term frequencies and page regions.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** strip only verified boilerplate.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** ensure warnings remain searchable.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 05: Footnote missing

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Footnote missing.
**Candidate cause:** footnote detached from claim.
**Competing explanation:** the source has no qualifying footnote.
**Discriminating check:** inspect page geometry and source markers.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** link footnotes to their anchors.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** evaluate a question whose answer changes with the note.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 06: Bad OCR identifier

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Bad OCR identifier.
**Candidate cause:** character confusion in scan.
**Competing explanation:** query contains a genuinely different code.
**Discriminating check:** compare pixels and text for ambiguous characters.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** correct extraction or index aliases with provenance.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** search both original and corrected identifier.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 07: Language mismatch

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Language mismatch.
**Candidate cause:** analyzer lacks language support.
**Competing explanation:** question uses a different language than source.
**Discriminating check:** compare token output and language metadata.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** choose language-appropriate analyzers.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure multilingual recall by language.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 08: Unexpected Unicode

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Unexpected Unicode.
**Candidate cause:** normalization differs at ingest and query.
**Competing explanation:** characters are intentionally distinct identifiers.
**Discriminating check:** inspect code points before normalization.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** apply consistent documented normalization.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** check exact-match cases including confusables.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 09: Version disappears

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Version disappears.
**Candidate cause:** incremental ingest omitted update.
**Competing explanation:** upstream document was intentionally removed.
**Discriminating check:** compare source event sequence and index manifests.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** repair replay from source of truth.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** confirm version and deletion events reconcile.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 10: Old content persists

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Old content persists.
**Candidate cause:** tombstone failed.
**Competing explanation:** policy intentionally retains historical material.
**Discriminating check:** compare effective date and deletion audit.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** propagate tombstones to all replicas.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** query after the permitted propagation window.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 11: Duplicate document

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Duplicate document.
**Candidate cause:** same source ingested through two paths.
**Competing explanation:** two separately authoritative copies exist.
**Discriminating check:** compare canonical IDs and content hashes.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** deduplicate with owner-approved canonicalization.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** confirm citations resolve to intended source.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 12: Chunk starts mid-clause

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Chunk starts mid-clause.
**Candidate cause:** fixed window cut a condition.
**Competing explanation:** source clause genuinely starts there.
**Discriminating check:** inspect adjacent offsets.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** use sentence-aware boundaries with limits.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test a query requiring the condition.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 13: Chunk too large

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Chunk too large.
**Candidate cause:** long sections dilute a key fact.
**Competing explanation:** question genuinely needs broad context.
**Discriminating check:** measure evidence location and token cost.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** split by meaningful subheading.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** compare span recall and answer support.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 14: Chunk too small

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Chunk too small.
**Candidate cause:** qualifier separated from answer.
**Competing explanation:** retriever scores the wrong topic.
**Discriminating check:** inspect neighboring source spans.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** expand around authorized spans.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** check that negation and exceptions survive.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 15: Overlap explosion

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Overlap explosion.
**Candidate cause:** stride nearly zero.
**Competing explanation:** the corpus really contains many separate documents.
**Discriminating check:** inspect C and O per parser.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** reduce overlap after boundary tests.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** compare index size and boundary recall.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 16: Chunk ID collision

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Chunk ID collision.
**Candidate cause:** ID excludes version or offset.
**Competing explanation:** the source system reused identifiers.
**Discriminating check:** trace ID construction across versions.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** include immutable version and span identity.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** rebuild and verify deterministic mappings.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 17: Parent expansion leak

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Parent expansion leak.
**Candidate cause:** child is public but parent includes private section.
**Competing explanation:** source ACL is actually uniform.
**Discriminating check:** compare permissions for every expanded span.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** recheck each span before assembly.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** attempt mixed-permission parent retrieval.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 18: Wrong heading attached

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Wrong heading attached.
**Candidate cause:** parser hierarchy shifted.
**Competing explanation:** document author placed misleading heading.
**Discriminating check:** compare extracted hierarchy to source structure.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** repair hierarchy parser.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** evaluate heading-sensitive query.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 19: Stale embedding

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Stale embedding.
**Candidate cause:** content updated without vector update.
**Competing explanation:** query model changed independently.
**Discriminating check:** compare content hash and embedding manifest.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** re-embed changed chunks atomically.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** compare retrieval before and after controlled update.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 20: Embedding mismatch

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Embedding mismatch.
**Candidate cause:** query and document model versions differ.
**Competing explanation:** query preprocessing changed language distribution.
**Discriminating check:** compare model and preprocessing fingerprints.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** route to matching index version.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** check fixed probe queries across migration.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 21: Zero vector

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Zero vector.
**Candidate cause:** empty text or invalid encoder output.
**Competing explanation:** legitimate vector is unexpectedly all zero.
**Discriminating check:** inspect input and vector norm.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** reject and quarantine invalid embeddings.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** verify no undefined cosine enters index.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 22: Dimension mismatch

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Dimension mismatch.
**Candidate cause:** wrong index selected.
**Competing explanation:** model upgrade changed output shape.
**Discriminating check:** compare vector schema and routing metadata.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** bind endpoint to compatible index.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test routing across all active versions.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 23: ANN misses item

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** ANN misses item.
**Candidate cause:** search approximation too aggressive.
**Competing explanation:** embedding itself ranks item poorly.
**Discriminating check:** compare ANN with exact search on sampled eligible pool.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** tune search effort or index construction.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure ANN recall and latency jointly.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 24: Filter leaves too few

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Filter leaves too few.
**Candidate cause:** post-filter removes ineligible candidates.
**Competing explanation:** authorized corpus lacks supporting material.
**Discriminating check:** compare scoped exact search and filtered ANN.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** use secure prefilter or adaptive eligible retrieval.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure eligible recall at fixed latency.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 25: Rare code not found

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Rare code not found.
**Candidate cause:** lexical tokenizer split the identifier.
**Competing explanation:** source contains another code.
**Discriminating check:** inspect analyzer output at index and query.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** add exact identifier field.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test punctuation and case variants.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 26: Synonym not found

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Synonym not found.
**Candidate cause:** lexical vocabulary mismatch.
**Competing explanation:** the query is genuinely ambiguous.
**Discriminating check:** compare semantic and lexical candidate sets.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** add dense retrieval or curated synonyms.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure paraphrase slice without harming exact IDs.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 27: Dense search misses exact ID

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Dense search misses exact ID.
**Candidate cause:** embedding underweights characters.
**Competing explanation:** the ID does not exist in the corpus.
**Discriminating check:** compare exact lexical lookup.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** add lexical branch to hybrid retrieval.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure ID-query recall.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 28: Hybrid gets worse

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Hybrid gets worse.
**Candidate cause:** fusion weights or list depths misconfigured.
**Competing explanation:** one branch adds mostly noise.
**Discriminating check:** compare per-branch and fused ranked lists.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** tune fusion on held-out relevance judgments.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** evaluate distinct exact and paraphrase slices.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 29: RRF tie surprises

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** RRF tie surprises.
**Candidate cause:** rank contributions are nearly equal.
**Competing explanation:** stable tie-breaker is absent.
**Discriminating check:** calculate individual contributions and tie policy.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** use deterministic documented tie-breaking.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** repeat the identical query across replicas.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 30: Reranker degrades

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Reranker degrades.
**Candidate cause:** model domain mismatch.
**Competing explanation:** correct candidate is absent before reranking.
**Discriminating check:** measure candidate recall and rank changes.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** train or select domain-appropriate reranker.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** compare paired queries with confidence intervals.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 31: Reranker sees secrets

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Reranker sees secrets.
**Candidate cause:** candidate filtering occurs after external call.
**Competing explanation:** vendor is authorized but logs are uncontrolled.
**Discriminating check:** trace raw data to every processing boundary.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** apply current policy before transfer.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test denied principals and external traces.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 32: Noisy high scores

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Noisy high scores.
**Candidate cause:** scores not calibrated across branches.
**Competing explanation:** query class changed over time.
**Discriminating check:** inspect score distributions per branch and query type.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** fuse ranks or calibrate on validation data.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** check held-out threshold reliability.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 33: Overaggressive threshold

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Overaggressive threshold.
**Candidate cause:** relevant low-score material excluded.
**Competing explanation:** there truly is no authorized answer.
**Discriminating check:** inspect labeled false negatives by slice.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** choose a threshold against explicit cost tradeoffs.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** remeasure abstention and unsupported-answer rates.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 34: Duplicate context

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Duplicate context.
**Candidate cause:** overlapping chunks all win ranking.
**Competing explanation:** several copies are independently needed.
**Discriminating check:** compare source offsets and coverage.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** deduplicate overlapping spans without dropping qualifiers.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test cross-boundary evidence coverage.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 35: Missing exception

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Missing exception.
**Candidate cause:** context packing prioritizes one high-score chunk.
**Competing explanation:** exception absent from source corpus.
**Discriminating check:** inspect candidate-to-context trace.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** reserve space for required exception evidence.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** evaluate exception-bearing questions.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 36: Context overflow

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Context overflow.
**Candidate cause:** token estimate uses wrong tokenizer.
**Competing explanation:** generation reserve was not allocated.
**Discriminating check:** compare actual request token accounting.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** count with serving tokenizer and include overhead.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test maximum-size prompts at boundary.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 37: Truncated evidence

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Truncated evidence.
**Candidate cause:** server silently drops tail.
**Competing explanation:** assembler packed beyond its own limit.
**Discriminating check:** inspect actual sent prompt and truncation telemetry.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** enforce budget before submission.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** verify last citation span is present.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 38: Irrelevant context

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Irrelevant context.
**Candidate cause:** candidate pool includes broad topical matches.
**Competing explanation:** the question is ambiguous.
**Discriminating check:** inspect query intent and ranked evidence.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** apply targeted reranking and clarification.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure precision and downstream support.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 39: Outdated answer

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Outdated answer.
**Candidate cause:** historical source ranks above current source.
**Competing explanation:** current document is unavailable.
**Discriminating check:** compare source effective dates and version lineage.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** prefer valid-time evidence under a documented rule.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test before and after a policy change.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 40: Unresolved conflict

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Unresolved conflict.
**Candidate cause:** sources apply to different jurisdictions.
**Competing explanation:** sources truly contradict for same scope.
**Discriminating check:** inspect dates jurisdiction and authority.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** surface conflict or escalate.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** evaluate competing-policy questions.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 41: Answer cites wrong span

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Answer cites wrong span.
**Candidate cause:** citation mapping shifted during preprocessing.
**Competing explanation:** generator attached a plausible but unsupported citation.
**Discriminating check:** resolve offsets against source version and claim.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** repair mapping and check entailment.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** audit claims individually after rechunking.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 42: Citation link denied

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Citation link denied.
**Candidate cause:** authorization changed since answer generation.
**Competing explanation:** citation service uses wrong principal.
**Discriminating check:** re-evaluate source permission at click time.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** return access-safe denial without exposing content.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test revocation between answer and click.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 43: Citation link stale

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Citation link stale.
**Candidate cause:** source was replaced without immutable version.
**Competing explanation:** source retention policy removed old version.
**Discriminating check:** resolve document ID version and locator.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** retain permitted immutable reference or record unavailability.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** check historical citation resolution.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 44: Unsupported synthesis

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Unsupported synthesis.
**Candidate cause:** model infers a conclusion beyond sources.
**Competing explanation:** sources omit an intermediate condition.
**Discriminating check:** map each claim to cited evidence.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** require explicit uncertainty or abstention.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** adjudicate unsupported-claim rate.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 45: Prompt injection

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Prompt injection.
**Candidate cause:** document instructs model to ignore system policy.
**Competing explanation:** legitimate quoted instructions are task data.
**Discriminating check:** inspect source provenance and instruction boundaries.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** treat retrieved text as untrusted and constrain tools.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** run adversarial retrieval tests.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 46: Instruction hidden in metadata

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Instruction hidden in metadata.
**Candidate cause:** title or filename is injected.
**Competing explanation:** display rendering concatenates fields without boundaries.
**Discriminating check:** inspect full prompt including metadata.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** delimit and sanitize untrusted metadata.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test title and filename injection cases.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 47: Cache leaks answer

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Cache leaks answer.
**Candidate cause:** cache key omits principal or policy scope.
**Competing explanation:** upstream service returned a wrong identity.
**Discriminating check:** inspect key components and identity propagation.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** partition cache by effective authorization context.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test two users with disjoint access.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 48: Cache survives revocation

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Cache survives revocation.
**Candidate cause:** invalidation lags policy changes.
**Competing explanation:** authorization recheck is skipped on cache hit.
**Discriminating check:** trace policy version on cache reuse.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** recheck access or invalidate by policy event.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** revoke access and repeat cached request.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 49: Log reveals private text

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Log reveals private text.
**Candidate cause:** raw chunks logged in shared telemetry.
**Competing explanation:** redaction is bypassed on error path.
**Discriminating check:** inspect access controls and sample schemas without exposing content.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** log IDs and protected metadata rather than raw passages.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** exercise failure paths under least privilege.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 50: Cross-tenant retrieval

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Cross-tenant retrieval.
**Candidate cause:** tenant filter omitted.
**Competing explanation:** tenant identity mapped incorrectly.
**Discriminating check:** trace authenticated tenant to index query.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** enforce tenant isolation at trusted boundary.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test adversarial cross-tenant requests.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 51: ACL stale

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** ACL stale.
**Candidate cause:** index policy snapshot lags revocation.
**Competing explanation:** source policy service itself is stale.
**Discriminating check:** compare authoritative policy timestamp and index metadata.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** use fresh policy decision and fail closed.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure revocation propagation and denied access.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 52: Group membership stale

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Group membership stale.
**Candidate cause:** identity token outlives group change.
**Competing explanation:** policy decision ignores token expiry.
**Discriminating check:** compare membership change and credential issuance times.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** refresh claims within policy freshness requirement.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test removal using existing sessions.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 53: Time-of-check race

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Time-of-check race.
**Candidate cause:** permission changes after candidate search.
**Competing explanation:** content version changes independently.
**Discriminating check:** trace decision time and disclosure time.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** recheck immediately before each disclosure.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** inject a revocation between stages.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 54: Policy service outage

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Policy service outage.
**Candidate cause:** dependency unavailable.
**Competing explanation:** policy denies user correctly.
**Discriminating check:** distinguish timeout from explicit denial safely.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** fail closed or use an approved scoped fallback.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** fault-inject outage without unrestricted fallback.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 55: Unanswerable question

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Unanswerable question.
**Candidate cause:** gold corpus lacks the fact.
**Competing explanation:** retrieval missed an existing fact.
**Discriminating check:** run expert source search within authorized corpus.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** improve abstention or retrieval based on finding.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure no-answer discrimination.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 56: Incomplete labels

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Incomplete labels.
**Candidate cause:** pooling missed relevant candidates.
**Competing explanation:** candidate is genuinely nonrelevant.
**Discriminating check:** blindly adjudicate newly surfaced passages.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** expand and version relevance judgments.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** remeasure metrics after adjudication.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 57: Chunk-label drift

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Chunk-label drift.
**Candidate cause:** chunk boundaries changed.
**Competing explanation:** source content changed too.
**Discriminating check:** compare stable source offsets across versions.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** label source spans and remap chunks.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** audit labels after chunker migration.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 58: Leakage in test set

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Leakage in test set.
**Candidate cause:** near-duplicate documents cross splits.
**Competing explanation:** questions are independently authored but similar.
**Discriminating check:** cluster source lineage and question templates.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** split by source family and time.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** audit overlap before reporting results.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 59: Judge overestimates

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Judge overestimates.
**Candidate cause:** LLM judge follows injected text.
**Competing explanation:** rubric treats topicality as entailment.
**Discriminating check:** compare blinded human adjudication and judge rationales.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** harden rubric and isolate evidence as data.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure disagreement on adversarial cases.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 60: Judge underestimates

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Judge underestimates.
**Candidate cause:** valid paraphrase not recognized.
**Competing explanation:** answer adds an unsupported assertion.
**Discriminating check:** compare claim decomposition and expert decisions.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** adjudicate and refine rubric.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** report per-slice agreement and uncertainty.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 61: Click metric misleads

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Click metric misleads.
**Candidate cause:** users click prominent citations regardless of support.
**Competing explanation:** users rarely need citations for easy queries.
**Discriminating check:** compare randomized placements and audited support.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** use click signals alongside claim-level judgments.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** monitor groundedness separately from engagement.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 62: Latency regression

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Latency regression.
**Candidate cause:** reranker queue saturates.
**Competing explanation:** retrieval index latency has increased.
**Discriminating check:** compare stage-level latency and queue time.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** cap candidate work or scale measured bottleneck.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** compare tail latency and answer quality.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 63: Embedding cost surge

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Embedding cost surge.
**Candidate cause:** overlap creates excess indexed tokens.
**Competing explanation:** reingestion repeats unchanged content.
**Discriminating check:** compare token counts by version and source.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** deduplicate and skip unchanged hashes.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure cost and boundary recall together.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 64: Slow update propagation

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Slow update propagation.
**Candidate cause:** batch job delays indexing.
**Competing explanation:** source event was never received.
**Discriminating check:** compare event times at each pipeline stage.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** repair event delivery or reduce batch interval.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** measure freshness lag distribution.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 65: Uneven query quality

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Uneven query quality.
**Candidate cause:** evaluation averages hide a minority language slice.
**Competing explanation:** the slice has different permission constraints.
**Discriminating check:** stratify by language and eligible corpus.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** improve language-specific analysis or retrieval.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** report slice counts and uncertainty.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 66: Sensitive answer in output

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Sensitive answer in output.
**Candidate cause:** evidence is authorized but answer violates purpose policy.
**Competing explanation:** model generated unsupported sensitive details.
**Discriminating check:** trace claims to sources and applicable output rules.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** enforce task-specific disclosure policy.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** test permitted retrieval with prohibited output.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.

### Diagnostic 67: Deleted source in backup

**Provenance:** HYPOTHETICAL; no incident or production observation is asserted.
**Symptom:** Deleted source in backup.
**Candidate cause:** backup retention differs from serving retention.
**Competing explanation:** deletion event never reached restoration process.
**Discriminating check:** inspect retention rules and restore workflow.
**Interpretation:** do not select a correction until the check distinguishes these explanations.
**Bounded correction:** apply deletion controls during restore.
**Safety boundary:** preserve effective authorization and source provenance during investigation and repair.
**Verification:** exercise a restore with tombstone replay.
**Rollback condition:** revert the change if the relevant protected slice worsens or the permission invariant fails.


### Evidence-based evaluation protocol

Define the end-user decision before selecting metrics.
Specify the authorized user population rather than evaluating only a service account with broad access.
Sample questions from a declared collection window and document the sampling mechanism.
Distinguish naturally occurring questions from authored challenge questions.
De-identify prompts where possible before evaluators receive them.
Retain the principal and policy snapshot needed to reconstruct the eligible corpus.
Record the source version and the time at which the question should be answered.
Identify whether the question concerns the current policy or a historical policy.
Create an answerability label at the same principal, time and task scope.
Mark an answer as unknown when an assessor cannot determine corpus coverage.
Do not convert unknown labels into negative labels merely for convenience.
Construct a no-answer sample deliberately because ordinary query logs may underrepresent it.
Include source-absent cases and access-denied cases as distinct categories.
Never reveal the existence of a restricted document solely to explain an abstention.
Define whether a partial answer is useful for each task category.
List mandatory facts for multi-part questions before judging an answer complete.
Distinguish a source that states a fact from one that merely mentions the same topic.
Record whether a relevant span is sufficient by itself or must be combined with another span.
Record contradictory and superseded spans separately from irrelevant spans.
Retain source-span locators independently of the current chunking algorithm.
If source offsets move across versions, maintain an auditable mapping or re-label.
Use adjudication for cases in which two qualified reviewers disagree.
Keep unresolved disagreements visible instead of fabricating a single certain label.
Document the reviewer rubric, access boundaries and conflict-of-interest procedure.
Protect sensitive source material throughout the labeling workflow.
Use an independent holdout for final selection after tuning on development queries.
Group related questions by document family when splitting the evaluation data.
Avoid putting near-identical policy versions into both tuning and holdout partitions.
Use a temporal holdout when expected deployment requires future-document generalization.
Do not assume a temporal split alone prevents document-family leakage.
Record model, index, chunker, policy and prompt versions for each evaluated run.
Fix the underlying source snapshot for paired comparisons where operationally possible.
Compare systems on the same questions and same eligible evidence population.
Measure retrieval at the candidate stage and after each subsequent filter.
Measure context recall after token packing, not just before it.
Track the proportion of questions for which all mandatory evidence is included.
For multi-hop questions, define whether both facts must appear to count as success.
Use question-level aggregation when each user question is the unit of interest.
Report source-level and span-level results separately when their meanings differ.
An overlap between two chunks must not turn one source fact into two independent successes.
Count duplicate chunks deliberately and document whether duplicates occupy rank slots.
Measure exact retrieval against relevance labels to assess the embedding representation.
Measure ANN against exact vector neighbors to assess approximation loss.
The two evaluations answer different questions and can move in opposite directions.
Evaluate lexical-only, dense-only and hybrid configurations under comparable resource budgets.
Include candidate depth, reranker depth and maximum context length in every comparison.
Measure context composition, including relevant, irrelevant, duplicate and obsolete content.
Track the source authority and applicable date of each selected passage.
Check whether a cited passage is present in the actual serialized model request.
Check whether a citation points to the exact version used to generate the answer.
Decompose each material answer into claims before judging support.
Treat a cited source as supporting only what follows from the cited span.
Mark a claim unsupported if its citation is topical but lacks the asserted detail.
Mark a claim contradicted separately from one for which evidence is merely absent.
Inspect quantities, units, conditions, negations and effective dates within each claim.
Check whether a correct fact is applied to the right jurisdiction and user.
Judge completeness independently from claim-level factual support.
A response containing only one correct fact may be grounded yet incomplete.
Judge task usefulness separately from verbosity and stylistic fluency.
Assess whether uncertainty and abstention are appropriate for the stakes.
Measure appropriate abstentions among genuinely unanswerable queries.
Measure incorrect abstentions among answerable queries.
Measure unsupported answers among unanswerable queries as a separate safety outcome.
Report raw counts as well as fractions for each small or high-risk slice.
Specify every metric's numerator, denominator, unit and evaluation window.
State how zero denominators, missing judgments and failed requests are handled.
Do not silently exclude policy-service failures from end-to-end reliability.
Track authorization violations as security incidents, not as ordinary relevance misses.
Report the number of attempted unauthorized disclosures, including blocked attempts.
Distinguish enforcement failures from adversarial requests correctly denied.
Include access revocation and group-membership-change scenarios in red-team evaluation.
Test injection through document body, metadata, titles and embedded quotations.
Evaluate whether the answer obeys trusted instructions despite adversarial evidence.
Assess whether the citation endpoint reauthorizes at the time of access.
Avoid using real secrets as test payloads when synthetic canaries can test the boundary.
Limit access to test outputs and clean up synthetic data under the evaluation policy.
When using a model as a judge, prevent source text from becoming judge instructions.
Blind the judge to treatment identity where the interface permits it.
Compare automated judgments with independently reviewed human examples.
Report disagreement by language, source type, answerability and query complexity.
Model-judge agreement is evidence about the judge, not independent proof of truth.
Reassess the judge when the generator, corpus or rubric changes.
Use paired per-query differences when comparing candidate systems.
Account for correlated questions from the same source document during uncertainty analysis.
For clustered samples, resample or analyze at a justified cluster level.
Provide an interval or another uncertainty summary appropriate to the sampling design.
Report slice sample sizes and avoid declaring a tiny observed difference conclusive.
Predefine practical relevance and security guardrails before choosing a winner.
Inspect failure examples even when an aggregate metric increases.
Use online experiments only after offline and security checks establish eligibility.
Randomize at the appropriate independent unit for the product and contamination risks.
Separate user adoption, task success and answer support as different outcomes.
Do not interpret clicks on citations as proof that those citations support claims.
Monitor failed requests, latency and cost alongside any online quality measure.
Retain a rollback path tied to an explicit guardrail breach.
Audit sensitive slices separately without publishing private query examples.
Re-run the evaluation after meaningful source, ACL, chunker or model changes.
Version the evaluation set and record why each label or metric definition changed.
An apparent regression can reflect newly discovered relevant labels rather than changed behavior.
Compare using the same label version before attributing a change to the system.
Publish the uncertainty, access constraints and unresolved evidence limitations with results.

**Takeaway:** diagnose with stage-level traces and a competing hypothesis before changing retrieval settings.

## 9. Complexity & Cost Analysis

Let D be the number of indexed chunks, V the average vector dimension and b the bytes per vector element.
A brute-force dense scan performs O(DV) arithmetic per query and stores O(DV) vector elements.
An ANN index changes the speed, memory and recall tradeoff; its complexity depends on the actual algorithm and configuration.
A lexical inverted index visits postings for the analyzed query terms, with cost dependent on term frequency and implementation.
A reranker evaluates a bounded candidate pool, but token length and model architecture control its actual cost.
Network transfers, ACL checks, metadata, storage replicas and serialization may dominate an individual component.
Treat asymptotic expressions as models rather than observed latency guarantees.

**ILLUSTRATIVE sizing assumptions:** 1,000,000 chunks; 768-dimensional float32 vectors; one unreplicated raw copy.
Validity: a float32 component occupies four bytes and vectors contain exactly 768 components.
From count times dimensions times bytes per component, raw vector storage follows.
```text
FORMULA [R6]: M = D * V * b
WHERE: D chunks, V components/chunk, b bytes/component, M raw vector bytes.
CONSTRAINT: excludes index overhead, metadata, replicas, compression and temporary memory.
NUMERICAL EXAMPLE: D=1000000, V=768, b=4: M=3072000000 bytes ≈ 2.861 GiB.
INTERPRETATION: this is a lower-level payload estimate, not provisioned index capacity.
```
Independent check: each vector occupies 3072 bytes and 1,000,000 vectors occupy 3,072,000,000 bytes.
Conversion uses 1 GiB = 1,073,741,824 bytes; 3,072,000,000 divided by that value is approximately 2.861.
Two complete replicas would require approximately 5.722 GiB of raw vector payload before other overhead.
Quantization changes b and may also change retrieval accuracy; evaluate both rather than assuming equivalence.
Deletion markers and reindexing can temporarily increase space beyond steady state.
Model calls add input and output token charges under the actual contract, not a timeless price in this sheet.
Measure full request latency from receipt through answer, including queue and policy checks.
Record p50, p95 and p99 with request counts, time window and concurrency, not only an average.
Measure quality at the same candidate count and latency budget when comparing configurations.
Monitor query volume, cache hits, embedding volume, stored bytes and reranker work separately.
Separate one-time backfill costs from recurring ingestion and per-query costs.
**Takeaway:** size the complete system using measured overhead and quality constraints, not vector payload alone.

## 10. Key Theorems / Results Reference

**DERIVED — candidate ceiling:** final context cannot contain a relevant item that no subsequent stage can access.
Precondition: the selector chooses only from a fixed candidate pool and performs no independent expansion.
Reason: the selected set is a subset of the candidate set.
Boundary: neighbor expansion, a second retrieval call or direct lookup changes the candidate set and the premise.
Consequence: improve candidate recall before expecting a reranker to recover missing evidence.

**DERIVED — budget conservation:** evidence tokens plus reserved non-evidence tokens cannot exceed the enforced window.
Precondition: all token counts use the deployed tokenizer and include actual framing.
Reason: the budget R3 subtracts each reservation from the total.
Boundary: hidden provider overhead and different tokenizers invalidate naive accounting.
Consequence: validate actual serialized prompts at deployment boundaries.

**DERIVED — subset authorization:** output from a correctly restricted selector cannot introduce an item outside its input set.
Precondition: every input item is eligible, expansion is disabled and the selector does not fetch other content.
Reason: a subset of an authorized set remains authorized under the same policy snapshot.
Boundary: policy revocation or mixed-permission expanded context invalidates the snapshot assumption.
Consequence: reauthorize before each disclosure and account for time-dependent policy.

**DERIVED — precision and recall tradeoff is not a theorem of every individual query.**
Adding a relevant result can increase both recall and precision; adding an irrelevant result can lower precision without changing recall.
Averaged curves depend on query population, ordering, labels and how empty cases are handled.
Consequence: report measured curves rather than claiming that every larger k must decrease precision.

**OPEN — primary references:** inspect and version the primary sources listed in Section 1 before publication.
No empirical retrieval performance or production incident is claimed by this sheet.
**Takeaway:** guarantees apply only under explicit scope, set membership and timing assumptions.

## 11. Interview / Exam Questions

Questions use the illustrative conventions and counts specified above.
Answers describe reasoning, alternatives and a common mistake; none reports an observed deployment.

### Question 01: What distinguishes RAG from a model answering from parameters alone?

**Answer:** RAG adds request-time external evidence to the generation input.
**Reasoning:** The retrieval corpus and permissions can change independently of model weights.
**Application:** A model can still ignore, misread or contradict the evidence.
**Boundary or alternative:** Alternatives include direct database lookup or a curated answer without generation.
**Common error:** Red flag: claiming retrieval guarantees factuality.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 02: How many chunks cover 1000 tokens with length 300 and overlap 50?

**Answer:** Four chunks, by R2: 1+ceil((1000-300)/(300-50))=4.
**Reasoning:** The starts are 0, 250, 500 and 750.
**Application:** The last chunk ends at token 1000 and may be shorter than 300.
**Boundary or alternative:** A structure-aware splitter can produce a different count.
**Common error:** Red flag: divide by 300 while ignoring overlap.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 03: Why can a high precision at one still be insufficient?

**Answer:** A question can require multiple supporting facts.
**Reasoning:** In Section 5, the first result is relevant so precision is 1.
**Application:** It finds one of two relevant chunks so recall is 0.5.
**Boundary or alternative:** A second relevant result raises recall to 1 for that example.
**Common error:** Red flag: treating one relevant hit as complete answer evidence.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 04: What should happen when an unauthorized passage is the closest vector?

**Answer:** Exclude the passage before any unauthorized downstream disclosure.
**Reasoning:** The policy decision depends on principal, resource, action and time.
**Application:** Retrieve within the authorized scope and recheck on disclosure.
**Boundary or alternative:** If policy state is unavailable, follow a fail-closed contract.
**Common error:** Red flag: let an external reranker read it and filter the final response.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 05: Can a cross-encoder repair a low-recall candidate set?

**Answer:** Not if it can only reorder the candidates already provided.
**Reasoning:** The final set is a subset of that pool under the stated premise.
**Application:** It can improve ordering when the needed source is already present.
**Boundary or alternative:** An additional retrieval or expansion stage changes the premise.
**Common error:** Red flag: increasing reranker compute without checking candidate recall.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 06: How do you compare two different chunkers fairly?

**Answer:** Use stable source spans and map each candidate chunk to the same evidence labels.
**Reasoning:** Split evaluation by document family or time to reduce leakage.
**Application:** Measure source-span recall, answer support, latency and context cost.
**Boundary or alternative:** Re-adjudicate uncertain mappings and report incomplete labels.
**Common error:** Red flag: compare recall on old chunk IDs after redefining boundaries.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 07: Compute the RRF score for ranks 1 and 3 with K=60.

**Answer:** The score is 1/61+1/63=124/3843, approximately 0.032266.
**Reasoning:** The rank contributions are approximately 0.016393 and 0.015873.
**Application:** A document appearing only at rank 1 scores approximately 0.016393.
**Boundary or alternative:** The score is not a relevance probability.
**Common error:** Red flag: treating K as the number of results rather than a smoothing constant.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 08: Why are source citations alone not enough?

**Answer:** A cited source may not entail the associated claim.
**Reasoning:** The cited span may be outdated, partial, irrelevant or inaccessible.
**Application:** Evaluate each material claim against the specific version and span.
**Boundary or alternative:** A response should qualify or abstain where evidence does not support the claim.
**Common error:** Red flag: counting links instead of checking support.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 09: A retrieval metric improves but users receive worse answers; diagnose.

**Answer:** First separate candidate recall, selection, generation and claim support.
**Reasoning:** An increase in candidates can add obsolete or distracting evidence.
**Application:** Compare paired traces on the same eligible queries and source snapshots.
**Boundary or alternative:** Audit whether required evidence survives context packing and whether citations entail claims.
**Common error:** Red flag: infer answer improvement from a single retrieval metric.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 10: How does revocation affect retrieval caches?

**Answer:** Previously cached content can become unauthorized after the policy changes.
**Reasoning:** Cache keys must capture relevant scope and policy state, but keys alone may not handle revocation.
**Application:** Reauthorize cached material before disclosure or invalidate on a trusted policy event.
**Boundary or alternative:** Evaluate multi-user isolation and revocation races, including citation access.
**Common error:** Red flag: assume a cache hit preserves the original permission indefinitely.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 11: How should a system answer when two current sources conflict?

**Answer:** Check scope, effective dates, jurisdiction and authority before treating them as contradictions.
**Reasoning:** If both apply and no rule resolves the conflict, disclose uncertainty or escalate.
**Application:** Do not arbitrarily select the passage with the higher cosine score.
**Boundary or alternative:** A product-specific policy may identify an authoritative source if that rule is supported.
**Common error:** Red flag: hide conflict because one source ranked first.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 12: How would you measure ANN approximation separately from embedding quality?

**Answer:** For sampled queries, run exact similarity search over the same authorized vector population.
**Reasoning:** Compare ANN neighbors with exact neighbors at the same k and metric.
**Application:** Then compare both lists to relevance labels, which answers a different question.
**Boundary or alternative:** Report latency, filter semantics and slice sizes with the approximation metric.
**Common error:** Red flag: interpret poor relevance as proof that ANN approximation is at fault.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 13: What changes if the token budget is cut in half?

**Answer:** Recompute available evidence using R3 and the actual tokenizer.
**Reasoning:** Preserve mandatory instructions and output reservation before packing evidence.
**Application:** Select complementary current passages and avoid redundant overlap.
**Boundary or alternative:** If necessary facts no longer fit, answer partially or abstain according to the product policy.
**Common error:** Red flag: silently truncate the last evidence passage and keep its citation.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.

### Question 14: Why is a no-answer query a separate evaluation slice?

**Answer:** Its relevant eligible count is zero, so the R1 recall denominator is zero.
**Reasoning:** Measure appropriate abstention and unsupported-answer rates on this slice.
**Application:** Verify whether the fact is genuinely absent or merely missed by retrieval.
**Boundary or alternative:** Report the policy population and time at which absence was judged.
**Common error:** Red flag: define undefined recall as zero without documenting a convention.
**Follow-up:** identify the input, policy or evidence change that would alter the answer.
**Strong response:** names the affected pipeline stage and a discriminating check.
**Scoring:** full credit requires the answer and the stated boundary; partial credit for an answer without its condition.
**Support:** use the relevant definition, formula or invariant in Sections 3–10, not an unsupported production claim.


**Takeaway:** defend each answer using a defined population, permission boundary and observable evidence.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Permission-aware retrieval-augmented generation and evidence assembly |
| Core formula | Recall@k = relevant authorized retrieved items / all relevant authorized items, when the denominator is positive |
| Key assumption | Ground-truth labels, access policy, source versions and evaluation population are defined |
| Complexity | Index build and query cost depend on corpus, index, embedding dimension and retrieval configuration; measure on the target system |
| Use when | An answer requires current, attributable evidence from an accessible corpus |
| Avoid when | Authorization cannot be enforced for downstream recipients or evidence freshness is unknown |
| Go deeper | [Playbook](../INDEX.md), [standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md), and Section 1 sources; source review remains open |
| Diagnostic | Separate retrieval misses, stale sources, unsupported claims, permission failures and citation errors |
| Evidence | Numerical retrieval examples are illustrative; no measured production performance is asserted |
