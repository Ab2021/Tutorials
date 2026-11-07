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
**Provenance:** [directory inventory](../../../../../Templates/directory.txt) lines 648–665 names this topic; it is an inventory, not evidence of an implementation.
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

**Symptom:** Missing PDF text.
**Candidate cause:** image-only page.
**Competing explanation:** parser dropped a text layer.
**Discriminating check:** compare rendered page with extracted text.
**Bounded correction:** run OCR with page provenance.
**Verification:** re-query an annotated page.

### Diagnostic 02: Misordered paragraphs

**Symptom:** Misordered paragraphs.
**Candidate cause:** column order lost.
**Competing explanation:** document actually has unusual sequence.
**Discriminating check:** compare bounding boxes with reading order.
**Bounded correction:** use layout-aware extraction.
**Verification:** inspect citations spanning columns.

### Diagnostic 03: Table answer wrong

**Symptom:** Table answer wrong.
**Candidate cause:** headers separated from cells.
**Competing explanation:** source itself contains conflicting tables.
**Discriminating check:** compare extracted rows against rendered table.
**Bounded correction:** retain row and header together.
**Verification:** check question on a row with shared heading.

### Diagnostic 04: Repeated headers dominate

**Symptom:** Repeated headers dominate.
**Candidate cause:** boilerplate indexed in every chunk.
**Competing explanation:** legitimate repeated warnings carry meaning.
**Discriminating check:** compare term frequencies and page regions.
**Bounded correction:** strip only verified boilerplate.
**Verification:** ensure warnings remain searchable.

### Diagnostic 05: Footnote missing

**Symptom:** Footnote missing.
**Candidate cause:** footnote detached from claim.
**Competing explanation:** the source has no qualifying footnote.
**Discriminating check:** inspect page geometry and source markers.
**Bounded correction:** link footnotes to their anchors.
**Verification:** evaluate a question whose answer changes with the note.

### Diagnostic 06: Bad OCR identifier

**Symptom:** Bad OCR identifier.
**Candidate cause:** character confusion in scan.
**Competing explanation:** query contains a genuinely different code.
**Discriminating check:** compare pixels and text for ambiguous characters.
**Bounded correction:** correct extraction or index aliases with provenance.
**Verification:** search both original and corrected identifier.

### Diagnostic 07: Language mismatch

**Symptom:** Language mismatch.
**Candidate cause:** analyzer lacks language support.
**Competing explanation:** question uses a different language than source.
**Discriminating check:** compare token output and language metadata.
**Bounded correction:** choose language-appropriate analyzers.
**Verification:** measure multilingual recall by language.

### Diagnostic 08: Unexpected Unicode

**Symptom:** Unexpected Unicode.
**Candidate cause:** normalization differs at ingest and query.
**Competing explanation:** characters are intentionally distinct identifiers.
**Discriminating check:** inspect code points before normalization.
**Bounded correction:** apply consistent documented normalization.
**Verification:** check exact-match cases including confusables.

### Diagnostic 09: Version disappears

**Symptom:** Version disappears.
**Candidate cause:** incremental ingest omitted update.
**Competing explanation:** upstream document was intentionally removed.
**Discriminating check:** compare source event sequence and index manifests.
**Bounded correction:** repair replay from source of truth.
**Verification:** confirm version and deletion events reconcile.

### Diagnostic 10: Old content persists

**Symptom:** Old content persists.
**Candidate cause:** tombstone failed.
**Competing explanation:** policy intentionally retains historical material.
**Discriminating check:** compare effective date and deletion audit.
**Bounded correction:** propagate tombstones to all replicas.
**Verification:** query after the permitted propagation window.

### Diagnostic 11: Duplicate document

**Symptom:** Duplicate document.
**Candidate cause:** same source ingested through two paths.
**Competing explanation:** two separately authoritative copies exist.
**Discriminating check:** compare canonical IDs and content hashes.
**Bounded correction:** deduplicate with owner-approved canonicalization.
**Verification:** confirm citations resolve to intended source.

### Diagnostic 12: Chunk starts mid-clause

**Symptom:** Chunk starts mid-clause.
**Candidate cause:** fixed window cut a condition.
**Competing explanation:** source clause genuinely starts there.
**Discriminating check:** inspect adjacent offsets.
**Bounded correction:** use sentence-aware boundaries with limits.
**Verification:** test a query requiring the condition.

### Diagnostic 13: Chunk too large

**Symptom:** Chunk too large.
**Candidate cause:** long sections dilute a key fact.
**Competing explanation:** question genuinely needs broad context.
**Discriminating check:** measure evidence location and token cost.
**Bounded correction:** split by meaningful subheading.
**Verification:** compare span recall and answer support.

### Diagnostic 14: Chunk too small

**Symptom:** Chunk too small.
**Candidate cause:** qualifier separated from answer.
**Competing explanation:** retriever scores the wrong topic.
**Discriminating check:** inspect neighboring source spans.
**Bounded correction:** expand around authorized spans.
**Verification:** check that negation and exceptions survive.

### Diagnostic 15: Overlap explosion

**Symptom:** Overlap explosion.
**Candidate cause:** stride nearly zero.
**Competing explanation:** the corpus really contains many separate documents.
**Discriminating check:** inspect C and O per parser.
**Bounded correction:** reduce overlap after boundary tests.
**Verification:** compare index size and boundary recall.

### Diagnostic 16: Chunk ID collision

**Symptom:** Chunk ID collision.
**Candidate cause:** ID excludes version or offset.
**Competing explanation:** the source system reused identifiers.
**Discriminating check:** trace ID construction across versions.
**Bounded correction:** include immutable version and span identity.
**Verification:** rebuild and verify deterministic mappings.

### Diagnostic 17: Parent expansion leak

**Symptom:** Parent expansion leak.
**Candidate cause:** child is public but parent includes private section.
**Competing explanation:** source ACL is actually uniform.
**Discriminating check:** compare permissions for every expanded span.
**Bounded correction:** recheck each span before assembly.
**Verification:** attempt mixed-permission parent retrieval.

### Diagnostic 18: Wrong heading attached

**Symptom:** Wrong heading attached.
**Candidate cause:** parser hierarchy shifted.
**Competing explanation:** document author placed misleading heading.
**Discriminating check:** compare extracted hierarchy to source structure.
**Bounded correction:** repair hierarchy parser.
**Verification:** evaluate heading-sensitive query.

### Diagnostic 19: Stale embedding

**Symptom:** Stale embedding.
**Candidate cause:** content updated without vector update.
**Competing explanation:** query model changed independently.
**Discriminating check:** compare content hash and embedding manifest.
**Bounded correction:** re-embed changed chunks atomically.
**Verification:** compare retrieval before and after controlled update.

### Diagnostic 20: Embedding mismatch

**Symptom:** Embedding mismatch.
**Candidate cause:** query and document model versions differ.
**Competing explanation:** query preprocessing changed language distribution.
**Discriminating check:** compare model and preprocessing fingerprints.
**Bounded correction:** route to matching index version.
**Verification:** check fixed probe queries across migration.

### Diagnostic 21: Zero vector

**Symptom:** Zero vector.
**Candidate cause:** empty text or invalid encoder output.
**Competing explanation:** legitimate vector is unexpectedly all zero.
**Discriminating check:** inspect input and vector norm.
**Bounded correction:** reject and quarantine invalid embeddings.
**Verification:** verify no undefined cosine enters index.

### Diagnostic 22: Dimension mismatch

**Symptom:** Dimension mismatch.
**Candidate cause:** wrong index selected.
**Competing explanation:** model upgrade changed output shape.
**Discriminating check:** compare vector schema and routing metadata.
**Bounded correction:** bind endpoint to compatible index.
**Verification:** test routing across all active versions.

### Diagnostic 23: ANN misses item

**Symptom:** ANN misses item.
**Candidate cause:** search approximation too aggressive.
**Competing explanation:** embedding itself ranks item poorly.
**Discriminating check:** compare ANN with exact search on sampled eligible pool.
**Bounded correction:** tune search effort or index construction.
**Verification:** measure ANN recall and latency jointly.

### Diagnostic 24: Filter leaves too few

**Symptom:** Filter leaves too few.
**Candidate cause:** post-filter removes ineligible candidates.
**Competing explanation:** authorized corpus lacks supporting material.
**Discriminating check:** compare scoped exact search and filtered ANN.
**Bounded correction:** use secure prefilter or adaptive eligible retrieval.
**Verification:** measure eligible recall at fixed latency.

### Diagnostic 25: Rare code not found

**Symptom:** Rare code not found.
**Candidate cause:** lexical tokenizer split the identifier.
**Competing explanation:** source contains another code.
**Discriminating check:** inspect analyzer output at index and query.
**Bounded correction:** add exact identifier field.
**Verification:** test punctuation and case variants.

### Diagnostic 26: Synonym not found

**Symptom:** Synonym not found.
**Candidate cause:** lexical vocabulary mismatch.
**Competing explanation:** the query is genuinely ambiguous.
**Discriminating check:** compare semantic and lexical candidate sets.
**Bounded correction:** add dense retrieval or curated synonyms.
**Verification:** measure paraphrase slice without harming exact IDs.

### Diagnostic 27: Dense search misses exact ID

**Symptom:** Dense search misses exact ID.
**Candidate cause:** embedding underweights characters.
**Competing explanation:** the ID does not exist in the corpus.
**Discriminating check:** compare exact lexical lookup.
**Bounded correction:** add lexical branch to hybrid retrieval.
**Verification:** measure ID-query recall.

### Diagnostic 28: Hybrid gets worse

**Symptom:** Hybrid gets worse.
**Candidate cause:** fusion weights or list depths misconfigured.
**Competing explanation:** one branch adds mostly noise.
**Discriminating check:** compare per-branch and fused ranked lists.
**Bounded correction:** tune fusion on held-out relevance judgments.
**Verification:** evaluate distinct exact and paraphrase slices.

### Diagnostic 29: RRF tie surprises

**Symptom:** RRF tie surprises.
**Candidate cause:** rank contributions are nearly equal.
**Competing explanation:** stable tie-breaker is absent.
**Discriminating check:** calculate individual contributions and tie policy.
**Bounded correction:** use deterministic documented tie-breaking.
**Verification:** repeat the identical query across replicas.

### Diagnostic 30: Reranker degrades

**Symptom:** Reranker degrades.
**Candidate cause:** model domain mismatch.
**Competing explanation:** correct candidate is absent before reranking.
**Discriminating check:** measure candidate recall and rank changes.
**Bounded correction:** train or select domain-appropriate reranker.
**Verification:** compare paired queries with confidence intervals.

### Diagnostic 31: Reranker sees secrets

**Symptom:** Reranker sees secrets.
**Candidate cause:** candidate filtering occurs after external call.
**Competing explanation:** vendor is authorized but logs are uncontrolled.
**Discriminating check:** trace raw data to every processing boundary.
**Bounded correction:** apply current policy before transfer.
**Verification:** test denied principals and external traces.

### Diagnostic 32: Noisy high scores

**Symptom:** Noisy high scores.
**Candidate cause:** scores not calibrated across branches.
**Competing explanation:** query class changed over time.
**Discriminating check:** inspect score distributions per branch and query type.
**Bounded correction:** fuse ranks or calibrate on validation data.
**Verification:** check held-out threshold reliability.

### Diagnostic 33: Overaggressive threshold

**Symptom:** Overaggressive threshold.
**Candidate cause:** relevant low-score material excluded.
**Competing explanation:** there truly is no authorized answer.
**Discriminating check:** inspect labeled false negatives by slice.
**Bounded correction:** choose a threshold against explicit cost tradeoffs.
**Verification:** remeasure abstention and unsupported-answer rates.

### Diagnostic 34: Duplicate context

**Symptom:** Duplicate context.
**Candidate cause:** overlapping chunks all win ranking.
**Competing explanation:** several copies are independently needed.
**Discriminating check:** compare source offsets and coverage.
**Bounded correction:** deduplicate overlapping spans without dropping qualifiers.
**Verification:** test cross-boundary evidence coverage.

### Diagnostic 35: Missing exception

**Symptom:** Missing exception.
**Candidate cause:** context packing prioritizes one high-score chunk.
**Competing explanation:** exception absent from source corpus.
**Discriminating check:** inspect candidate-to-context trace.
**Bounded correction:** reserve space for required exception evidence.
**Verification:** evaluate exception-bearing questions.

### Diagnostic 36: Context overflow

**Symptom:** Context overflow.
**Candidate cause:** token estimate uses wrong tokenizer.
**Competing explanation:** generation reserve was not allocated.
**Discriminating check:** compare actual request token accounting.
**Bounded correction:** count with serving tokenizer and include overhead.
**Verification:** test maximum-size prompts at boundary.

### Diagnostic 37: Truncated evidence

**Symptom:** Truncated evidence.
**Candidate cause:** server silently drops tail.
**Competing explanation:** assembler packed beyond its own limit.
**Discriminating check:** inspect actual sent prompt and truncation telemetry.
**Bounded correction:** enforce budget before submission.
**Verification:** verify last citation span is present.

### Diagnostic 38: Irrelevant context

**Symptom:** Irrelevant context.
**Candidate cause:** candidate pool includes broad topical matches.
**Competing explanation:** the question is ambiguous.
**Discriminating check:** inspect query intent and ranked evidence.
**Bounded correction:** apply targeted reranking and clarification.
**Verification:** measure precision and downstream support.

### Diagnostic 39: Outdated answer

**Symptom:** Outdated answer.
**Candidate cause:** historical source ranks above current source.
**Competing explanation:** current document is unavailable.
**Discriminating check:** compare source effective dates and version lineage.
**Bounded correction:** prefer valid-time evidence under a documented rule.
**Verification:** test before and after a policy change.

### Diagnostic 40: Unresolved conflict

**Symptom:** Unresolved conflict.
**Candidate cause:** sources apply to different jurisdictions.
**Competing explanation:** sources truly contradict for same scope.
**Discriminating check:** inspect dates jurisdiction and authority.
**Bounded correction:** surface conflict or escalate.
**Verification:** evaluate competing-policy questions.

### Diagnostic 41: Answer cites wrong span

**Symptom:** Answer cites wrong span.
**Candidate cause:** citation mapping shifted during preprocessing.
**Competing explanation:** generator attached a plausible but unsupported citation.
**Discriminating check:** resolve offsets against source version and claim.
**Bounded correction:** repair mapping and check entailment.
**Verification:** audit claims individually after rechunking.

### Diagnostic 42: Citation link denied

**Symptom:** Citation link denied.
**Candidate cause:** authorization changed since answer generation.
**Competing explanation:** citation service uses wrong principal.
**Discriminating check:** re-evaluate source permission at click time.
**Bounded correction:** return access-safe denial without exposing content.
**Verification:** test revocation between answer and click.

### Diagnostic 43: Citation link stale

**Symptom:** Citation link stale.
**Candidate cause:** source was replaced without immutable version.
**Competing explanation:** source retention policy removed old version.
**Discriminating check:** resolve document ID version and locator.
**Bounded correction:** retain permitted immutable reference or record unavailability.
**Verification:** check historical citation resolution.

### Diagnostic 44: Unsupported synthesis

**Symptom:** Unsupported synthesis.
**Candidate cause:** model infers a conclusion beyond sources.
**Competing explanation:** sources omit an intermediate condition.
**Discriminating check:** map each claim to cited evidence.
**Bounded correction:** require explicit uncertainty or abstention.
**Verification:** adjudicate unsupported-claim rate.

### Diagnostic 45: Prompt injection

**Symptom:** Prompt injection.
**Candidate cause:** document instructs model to ignore system policy.
**Competing explanation:** legitimate quoted instructions are task data.
**Discriminating check:** inspect source provenance and instruction boundaries.
**Bounded correction:** treat retrieved text as untrusted and constrain tools.
**Verification:** run adversarial retrieval tests.

### Diagnostic 46: Instruction hidden in metadata

**Symptom:** Instruction hidden in metadata.
**Candidate cause:** title or filename is injected.
**Competing explanation:** display rendering concatenates fields without boundaries.
**Discriminating check:** inspect full prompt including metadata.
**Bounded correction:** delimit and sanitize untrusted metadata.
**Verification:** test title and filename injection cases.

### Diagnostic 47: Cache leaks answer

**Symptom:** Cache leaks answer.
**Candidate cause:** cache key omits principal or policy scope.
**Competing explanation:** upstream service returned a wrong identity.
**Discriminating check:** inspect key components and identity propagation.
**Bounded correction:** partition cache by effective authorization context.
**Verification:** test two users with disjoint access.

### Diagnostic 48: Cache survives revocation

**Symptom:** Cache survives revocation.
**Candidate cause:** invalidation lags policy changes.
**Competing explanation:** authorization recheck is skipped on cache hit.
**Discriminating check:** trace policy version on cache reuse.
**Bounded correction:** recheck access or invalidate by policy event.
**Verification:** revoke access and repeat cached request.

### Diagnostic 49: Log reveals private text

**Symptom:** Log reveals private text.
**Candidate cause:** raw chunks logged in shared telemetry.
**Competing explanation:** redaction is bypassed on error path.
**Discriminating check:** inspect access controls and sample schemas without exposing content.
**Bounded correction:** log IDs and protected metadata rather than raw passages.
**Verification:** exercise failure paths under least privilege.

### Diagnostic 50: Cross-tenant retrieval

**Symptom:** Cross-tenant retrieval.
**Candidate cause:** tenant filter omitted.
**Competing explanation:** tenant identity mapped incorrectly.
**Discriminating check:** trace authenticated tenant to index query.
**Bounded correction:** enforce tenant isolation at trusted boundary.
**Verification:** test adversarial cross-tenant requests.

### Diagnostic 51: ACL stale

**Symptom:** ACL stale.
**Candidate cause:** index policy snapshot lags revocation.
**Competing explanation:** source policy service itself is stale.
**Discriminating check:** compare authoritative policy timestamp and index metadata.
**Bounded correction:** use fresh policy decision and fail closed.
**Verification:** measure revocation propagation and denied access.

### Diagnostic 52: Group membership stale

**Symptom:** Group membership stale.
**Candidate cause:** identity token outlives group change.
**Competing explanation:** policy decision ignores token expiry.
**Discriminating check:** compare membership change and credential issuance times.
**Bounded correction:** refresh claims within policy freshness requirement.
**Verification:** test removal using existing sessions.

### Diagnostic 53: Time-of-check race

**Symptom:** Time-of-check race.
**Candidate cause:** permission changes after candidate search.
**Competing explanation:** content version changes independently.
**Discriminating check:** trace decision time and disclosure time.
**Bounded correction:** recheck immediately before each disclosure.
**Verification:** inject a revocation between stages.

### Diagnostic 54: Policy service outage

**Symptom:** Policy service outage.
**Candidate cause:** dependency unavailable.
**Competing explanation:** policy denies user correctly.
**Discriminating check:** distinguish timeout from explicit denial safely.
**Bounded correction:** fail closed or use an approved scoped fallback.
**Verification:** fault-inject outage without unrestricted fallback.

### Diagnostic 55: Unanswerable question

**Symptom:** Unanswerable question.
**Candidate cause:** gold corpus lacks the fact.
**Competing explanation:** retrieval missed an existing fact.
**Discriminating check:** run expert source search within authorized corpus.
**Bounded correction:** improve abstention or retrieval based on finding.
**Verification:** measure no-answer discrimination.

### Diagnostic 56: Incomplete labels

**Symptom:** Incomplete labels.
**Candidate cause:** pooling missed relevant candidates.
**Competing explanation:** candidate is genuinely nonrelevant.
**Discriminating check:** blindly adjudicate newly surfaced passages.
**Bounded correction:** expand and version relevance judgments.
**Verification:** remeasure metrics after adjudication.

### Diagnostic 57: Chunk-label drift

**Symptom:** Chunk-label drift.
**Candidate cause:** chunk boundaries changed.
**Competing explanation:** source content changed too.
**Discriminating check:** compare stable source offsets across versions.
**Bounded correction:** label source spans and remap chunks.
**Verification:** audit labels after chunker migration.

### Diagnostic 58: Leakage in test set

**Symptom:** Leakage in test set.
**Candidate cause:** near-duplicate documents cross splits.
**Competing explanation:** questions are independently authored but similar.
**Discriminating check:** cluster source lineage and question templates.
**Bounded correction:** split by source family and time.
**Verification:** audit overlap before reporting results.

### Diagnostic 59: Judge overestimates

**Symptom:** Judge overestimates.
**Candidate cause:** LLM judge follows injected text.
**Competing explanation:** rubric treats topicality as entailment.
**Discriminating check:** compare blinded human adjudication and judge rationales.
**Bounded correction:** harden rubric and isolate evidence as data.
**Verification:** measure disagreement on adversarial cases.

### Diagnostic 60: Judge underestimates

**Symptom:** Judge underestimates.
**Candidate cause:** valid paraphrase not recognized.
**Competing explanation:** answer adds an unsupported assertion.
**Discriminating check:** compare claim decomposition and expert decisions.
**Bounded correction:** adjudicate and refine rubric.
**Verification:** report per-slice agreement and uncertainty.

### Diagnostic 61: Click metric misleads

**Symptom:** Click metric misleads.
**Candidate cause:** users click prominent citations regardless of support.
**Competing explanation:** users rarely need citations for easy queries.
**Discriminating check:** compare randomized placements and audited support.
**Bounded correction:** use click signals alongside claim-level judgments.
**Verification:** monitor groundedness separately from engagement.

### Diagnostic 62: Latency regression

**Symptom:** Latency regression.
**Candidate cause:** reranker queue saturates.
**Competing explanation:** retrieval index latency has increased.
**Discriminating check:** compare stage-level latency and queue time.
**Bounded correction:** cap candidate work or scale measured bottleneck.
**Verification:** compare tail latency and answer quality.

### Diagnostic 63: Embedding cost surge

**Symptom:** Embedding cost surge.
**Candidate cause:** overlap creates excess indexed tokens.
**Competing explanation:** reingestion repeats unchanged content.
**Discriminating check:** compare token counts by version and source.
**Bounded correction:** deduplicate and skip unchanged hashes.
**Verification:** measure cost and boundary recall together.

### Diagnostic 64: Slow update propagation

**Symptom:** Slow update propagation.
**Candidate cause:** batch job delays indexing.
**Competing explanation:** source event was never received.
**Discriminating check:** compare event times at each pipeline stage.
**Bounded correction:** repair event delivery or reduce batch interval.
**Verification:** measure freshness lag distribution.

### Diagnostic 65: Uneven query quality

**Symptom:** Uneven query quality.
**Candidate cause:** evaluation averages hide a minority language slice.
**Competing explanation:** the slice has different permission constraints.
**Discriminating check:** stratify by language and eligible corpus.
**Bounded correction:** improve language-specific analysis or retrieval.
**Verification:** report slice counts and uncertainty.

### Diagnostic 66: Sensitive answer in output

**Symptom:** Sensitive answer in output.
**Candidate cause:** evidence is authorized but answer violates purpose policy.
**Competing explanation:** model generated unsupported sensitive details.
**Discriminating check:** trace claims to sources and applicable output rules.
**Bounded correction:** enforce task-specific disclosure policy.
**Verification:** test permitted retrieval with prohibited output.

### Diagnostic 67: Deleted source in backup

**Symptom:** Deleted source in backup.
**Candidate cause:** backup retention differs from serving retention.
**Competing explanation:** deletion event never reached restoration process.
**Discriminating check:** inspect retention rules and restore workflow.
**Bounded correction:** apply deletion controls during restore.
**Verification:** exercise a restore with tombstone replay.


### Engineering contract: identities and trust boundaries

A request begins with an authenticated principal, not a display name supplied in a prompt.
The identity provider authenticates that principal; authorization still requires an independent policy decision.
Bind the subject identifier to a tenant and a session with a defined lifetime.
A delegated service identity must not silently inherit the service's full corpus privileges for end-user queries.
If delegation is used, pass the user's constrained authority or evaluate both user and service permissions.
The requested action may differ between search, reading content, exporting, and sharing a citation.
A source can allow discovery of a title while denying access to its full text.
Treat titles and snippets as data that can themselves require authorization.
Resource identifiers must resolve to the exact version or source state evaluated by the policy engine.
Policy input should include resource ownership, tenant, classification, purpose, and any required environmental attributes.
Only attributes actually supported by the policy contract should affect the decision.
Untrusted request text must not be allowed to set its own clearance, tenant, or group membership.
Group membership should come from an authoritative identity source with a documented freshness guarantee.
A policy decision is valid only within its explicit scope and lifetime.
Indefinite caching of an allow decision defeats later revocation.
If a policy decision point returns an error, distinguish unavailable policy from an affirmative allow.
An unavailable decision should stop disclosure unless an independently authorized fallback is defined.
A fallback cannot be an unrestricted search that merely hides final citations.
Index metadata can accelerate filtering but may lag authoritative permission changes.
Record the index snapshot and the policy decision version separately.
A vector store holding confidential embeddings is itself a protected processor and storage boundary.
Similarity scores, identifiers, titles, and result counts may reveal sensitive information even without source text.
Do not return raw candidate IDs or scores across a boundary without authorization for that metadata.
An internal search process may need broader index access than an end user, under a separate service policy.
That internal privilege is not authority to expose every candidate to downstream services.
A prefilter should operate on a trusted server-side eligible set, not a client-supplied filter alone.
When an engine supports only postfiltering, treat its execution environment as trusted for all scanned content.
Before candidates leave that environment, enforce effective policy on every returned item.
Overfetching eligible results can restore some candidate coverage after filtering, but has no security magic.
A query with no eligible matches must not reveal whether an inaccessible document matched.
Different response sizes, error messages, timing, and citation identifiers can create side channels.
Specify acceptable disclosure of existence and mitigate the channels appropriate to the threat model.
A multi-tenant index needs tenant isolation in both the search constraint and every direct object lookup.
A guessed citation identifier must not bypass tenant isolation at the source endpoint.
When a chunk combines multiple protected regions, either split it by policy or apply the most restrictive effective policy.
Do not infer that all pages of a parent document share the same permissions.
Parent expansion, neighboring chunks, and source previews each introduce additional resources to authorize.
Deduplication must preserve the policy of every contributing source, not copy the least restrictive ACL.
A copied passage in a public document and a private original may have different source rights.
Use the authorization of the actual cited source and verify that redistribution is permitted.
Do not expose a private source merely because equivalent public text is available elsewhere.
A cross-encoder hosted externally is a disclosure recipient, even if it returns only scores.
The user, retrieval backend, reranker, generator, observability sink, and citation viewer are distinct recipients.
For each recipient, evaluate data classification, contract, residency, and permitted processing purpose.
Do not send restricted text to a model vendor solely because the end user may read it.
An authorized user does not automatically authorize every subprocess or third-party service.
Minimize sent spans and strip unrelated sensitive fields before an allowed external call.
If redaction changes a material clause, mark the evidence incomplete rather than pretending it is intact.
Store a redaction policy version with the prompt trace where retention is permitted.
Use encrypted transport and storage according to the system's threat model and key-management policy.
Encryption protects confidentiality in transit and at rest; it does not replace authorization at read time.
A service account's logs need their own access policy and retention period.
Avoid putting retrieved source text, credentials, or full prompts in broad operational logs.
Prefer opaque request IDs and policy-approved, minimized metadata for routine telemetry.
Restrict detailed traces to a controlled debugging workflow with a documented purpose and expiration.
A model provider's retention and training terms are separate from the local application's log policy.
The model's output may restate a restricted input, so output storage also needs an access decision.
A later user opening a shared conversation is a new recipient, not a continuation of the original authorization.
Recheck access to supporting evidence before replaying a stored answer to that later user.
If a claim is derived from multiple sources, losing permission to any necessary source can invalidate the claim.
A response-time guard should suppress that claim or regenerate it from currently authorized evidence.
Merely deleting the revoked citation while retaining its fact still discloses the underlying information.
When whole-answer provenance cannot be reconstructed, fail closed on the affected stored answer.
If permissions change while tokens are streaming, define whether delivery must stop and how revocation is detected.
A per-request snapshot gives a simpler contract but permits access until that snapshot expires.
A stricter revocation requirement needs bounded rechecks or an interrupt mechanism during streaming.
Neither approach provides instantaneous revocation without a defined synchronization and delivery contract.
A citation link must authorize the viewer independently when opened, including on a different device.
A signed URL grants a capability to its holder; its expiration and sharing risks must match policy.
Do not put durable credentials or unrestricted source URLs inside generated text.
Cache keys should partition by tenant, principal or effective scope, source snapshot, and relevant policy version.
If contextual attributes affect policy, the cache contract must include them or force reauthorization.
A cached answer cannot be safely shared merely because two users issued the same question.
An event-driven invalidation mechanism needs handling for missed, delayed, duplicate, and reordered events.
A policy revision check or bounded expiry can limit exposure when invalidation is delayed.
The correct bound is a product and security requirement, not a universal number.
Check backups, replicas, search indexes, reranker caches, and analytics exports when a source is deleted.
Retention obligations may permit some protected records to remain even when ordinary retrieval is disabled.
Separate deletion of user-visible access from physical erasure and from legally required preservation.
Record which artifact was removed, where, and when, without leaking the removed content into the audit record.

### Production trace: from source to claim

Assign each source a canonical identifier that survives renaming without conflating independent documents.
Record source system, owner, document type, language, acquisition time, and source version.
Distinguish document creation time, effective time, index time, and request time.
A document published today can describe a policy effective next month.
An older document can remain authoritative for a historical question.
A freshness filter should therefore use the time meaning required by the question, not ingestion time alone.
Keep a content digest for integrity and change detection, not as proof of authorship.
A digest collision-resistant enough for engineering use still does not establish source authority.
Store a trusted acquisition record or signature where the source system offers one.
Extraction must preserve page numbers, section names, and stable locators when possible.
For tables, retain column names, row headings, units, and relevant footnotes with each extracted value.
A cell containing 12 is uninterpretable without its measure, unit, row, and effective period.
For scanned pages, retain the rendered page reference and confidence information where available.
OCR confidence is a triage signal, not a guarantee that a digit or negation is correct.
Preserve negation and exception clauses when splitting sentences into search units.
A chunk identifier should include or resolve to document version and extraction or chunking revision.
Maintain a mapping from each chunk back to exact source spans.
If normalization changes offsets, preserve a map to the original text or use a stable alternate locator.
A citation should identify the source version and the cited span, not just an indexed chunk number.
Chunk identifiers can become invalid when a document is reparsed or the chunk size changes.
A source-span locator allows comparisons across alternative chunkers when the source version is unchanged.
When a source changes, old offsets cannot be silently applied to the new text.
Maintain a versioned source snapshot or resolve the citation against the historical version.
If historical access is prohibited or the version has been deleted, show an access-safe unavailable state.
A source URL that points only to the latest version cannot substantiate a historical quotation by itself.
Record extractor, normalization, tokenizer, chunker, embedding model, and index build versions.
An embedding model change requires re-embedding the corpus or maintaining explicitly separated compatible indexes.
Query vectors and document vectors from incompatible spaces must not be compared merely because dimensions match.
During migration, route each query to an index paired with its query encoder and metric.
A model version alias that changes silently undermines reproducibility.
Record the resolved artifact version where the serving interface exposes it.
A hybrid index may update lexical and dense representations at different times.
Track the high-water mark for each index and expose a consistent freshness contract to clients.
If only one index has a new document, fusion can still return it, but the ranking reflects partial ingestion.
A snapshot identifier should bind chunk data, metadata, and the retrieval configuration used for a trace.
For reproducibility, also retain the query rewrite, filters, top-k settings, and rank fusion parameters when allowed.
Do not assume nondeterministic model output can be recreated exactly from a prompt alone.
Record generator version, decoding settings, and provider limitations separately from the evidence snapshot.
Represent the answer as atomic material claims wherever practical.
Associate each claim with one or more source spans and distinguish direct quotes from paraphrases.
A direct quote needs exact text, including omitted context marked under the applicable quotation convention.
A paraphrase requires semantic support without adding an unsupported condition or numerical precision.
Multiple citations may jointly support a conclusion even if neither source supports it alone.
If an inference joins sources, document the join key, time scope, and reasoning needed to connect them.
Two sources can disagree because they refer to different locations rather than because either is wrong.
Check jurisdiction, audience, entity, unit, and effective period before resolving apparent conflicts.
Source authority is a policy or editorial determination, not a similarity ranking.
Where no authoritative conflict rule exists, report both positions or escalate.
Avoid averaging conflicting rules or dates unless the task explicitly calls for a justified aggregation.
A citation coverage metric counts links; an entailment review asks whether each linked span supports the claim.
Both are needed because perfect link coverage can coexist with wholly unsupported claims.
Resolve citations before delivery and confirm that the displayed source is the version actually used.
If a link fails, do not replace it with a different version without rechecking claim support.
Record unresolved provenance as unknown, not as a source with invented authority.

### Adversarial retrieval and evidence handling

Retrieved documents are data, even if they contain role labels or imperative text.
A webpage can say that prior instructions are obsolete; the statement does not acquire system authority.
A document can contain legitimate procedural instructions that the user asks the assistant to summarize.
Summarizing those instructions differs from obeying them as instructions to the assistant.
Keep trusted system and developer instructions structurally separate from retrieved material.
Mark retrieved spans as untrusted with stable delimiters, but do not treat delimiters as a complete defense.
An adversary may include fake delimiters, forged tool results, or role-like tags in source text.
Serialize content so source characters cannot alter the surrounding message structure.
Apply source encoding or escaping appropriate to the actual transport and parser.
The generator must not be the sole enforcement point for access control or tool permissions.
A prompt asking the model not to leak secrets is not a substitute for withholding unauthorized secrets.
Restrict available tools, destinations, and operations using code outside the model.
This sheet does not assume that the RAG answerer can execute arbitrary tool calls.
If a product later adds tools, reauthorize each call and require approval where its contract demands it.
A retrieved instruction to send the prompt to an external URL is not an authorized user request.
A retrieved link may itself contain an attacker-controlled destination or tracking parameter.
Do not automatically fetch links or follow redirects solely because retrieved content requests it.
If link fetching is an allowed product feature, enforce network egress restrictions and destination validation.
A citation viewer should render untrusted HTML safely rather than executing source scripts.
File names, document titles, and metadata can also contain instruction-like attacks.
Apply the same data-versus-instruction distinction to metadata and retrieved text.
A poisoned document can be relevant to a query while deliberately supplying false facts.
Relevance ranking does not certify document integrity, ownership, or truth.
Restrict who can publish to indexed sources and review high-impact source changes.
Monitor suspicious corpus changes using metadata and approved samples, not unrestricted content dumps.
A repeated attacker phrase can dominate a lexical retriever without being authoritative.
A semantically optimized poisoned paragraph can also attract dense retrieval.
Evaluate poisoning resistance using controlled test documents in an isolated test corpus.
Test instructions that demand role changes, secret disclosure, citation laundering, and unauthorized actions.
Test the same attack in body text, tables, headings, metadata, and source excerpts.
Include benign quoted attack examples to measure whether the defense overblocks legitimate discussion.
A source may legitimately quote a malicious message for security analysis.
The correct answer can describe the quoted text without carrying out its instruction.
Use harmless synthetic secrets and callback endpoints controlled by the test operator.
Measure actual unauthorized disclosure or action rather than only whether the output contains an attack phrase.
A refusal that still includes a protected secret is not a successful security outcome.
Judge models can themselves follow instructions embedded in answers or retrieved evidence.
Use isolated evaluation prompts, constrained outputs, and human review of adversarial samples.
An automatic judge's explanation may reveal sensitive source text and needs the same data handling controls.
Treat prompt-injection resistance as an evaluated property with unresolved risk, not a blanket guarantee.

### Evaluation design and counterfactual tests

Define the end-user task and the decision that an answer is intended to support.
Construct evaluation questions from the permitted population without exposing private examples in public reports.
Use document-family or time-based splits when related sources or revisions could leak across sets.
Reserve a final test set until the retrieval and prompting decisions have been selected.
Training on a test question or repeatedly tuning against it invalidates its role as an independent holdout.
Label the principal, action, policy snapshot, and question time for each evaluation query.
Gold relevance must include only evidence accessible to that principal under the stated snapshot.
A source that was accessible last week may not belong to today's eligible set.
Label supporting spans, not only documents, when claims depend on specific clauses.
A relevant document can contain both a supporting passage and a contradictory or obsolete passage.
Label whether the answer requires all evidence units or any one alternative source.
For conjunctive questions, finding one required span is not complete evidence coverage.
For alternative sources, avoid double-counting equivalent copies as independently required facts.
Maintain a separate no-answer set with evidence that the eligible corpus was searched adequately.
Absence judgments are provisional when the corpus inventory or relevance labels are incomplete.
Include permission-denied cases without requiring the system to reveal that a forbidden source exists.
Record a third state for ambiguous or disputed labels instead of forcing every item into relevant or irrelevant.
Specify the adjudication process and preserve disagreement where experts cannot resolve it.
Measure retrieval on the same corpus snapshot, principal, query set, and relevance version when comparing systems.
Compare exact vector retrieval to approximate retrieval within the identical authorized population.
Exact vector neighbors are a reference for approximation fidelity, not ground truth for semantic relevance.
A perfect ANN match can still retrieve irrelevant documents if the embedding representation is poor.
A poor ANN match can coexist with good task relevance if substitute results answer the question.
Measure both neighbor overlap and gold relevance, with their different denominators stated.
Measure candidate recall before reranking and evidence coverage after packing.
If candidate recall is high and context recall falls, investigate reranking, deduplication, and token packing.
If context contains support but answers fail, inspect generation, claim mapping, and citation validation.
If the answer is supported but unhelpful, task usefulness and response clarity need separate assessment.
Distinguish unsupported material claims from minor stylistic differences that do not change the decision.
Define whether one unsupported critical claim makes the entire answer a failure.
Treat a cited but inaccessible source as an authorization failure, not merely a broken link.
Track false abstention on answerable questions and unsupported answering on unanswerable questions.
A system that always refuses may have zero unsupported answers but poor task completion.
A system that always answers may have high completion but unacceptable unsupported claims.
Select operating points using explicit harm and utility assumptions, not an unqualified single score.
Report micro averages when each claim should have equal weight and macro averages when each query should have equal weight.
These averages can differ substantially when some answers contain many more claims.
Report the distribution of number of claims per answer alongside claim-level support.
Use paired comparisons because the same query is evaluated under both configurations.
For correlated queries from one source family, account for the family as a sampling cluster.
A confidence interval reflects the specified sampling procedure, not unknown systematic labeling bias.
Report query counts and uncertainty for each subgroup before interpreting a difference.
A zero observed security failure on a small sample is not proof of zero risk.
Measure latency end to end, including authentication, policy evaluation, retries, queueing, and answer validation.
Report error rates and timeouts alongside latency; excluding failed slow requests can mislead.
Use production-like concurrency and document the workload before interpreting load-test numbers.
Check that the evaluated provider configuration and data-processing boundary match the proposed use.
Monitor the same protected quality slices after source, permissions, chunker, index, or model changes.
Separate deliberate policy changes from regressions by evaluating against the correct historical snapshot.
When new labels are added, recompute both systems against the same revised label set.
A dashboard cannot substitute for an auditable sample of actual claims and source spans.

### Worked counterexamples: independently checkable invented inputs

**ILLUSTRATIVE:** A query has four eligible relevant spans and a candidate list of six distinct spans.
The six candidates contain three relevant spans, so candidate recall is 3/4 = 0.75.
After reranking, two relevant spans appear in the three selected context items.
Context recall is 2/4 = 0.50 and observed precision is 2/3, approximately 0.667.
The reranker or packer lost one candidate hit; this does not prove which component caused the loss.
Compare the reranked list with the packed list to identify the first stage that dropped it.
If a previously unseen source is introduced by a second retrieval call, the fixed-candidate ceiling no longer applies.

**ILLUSTRATIVE:** A response contains eight material claims, six fully supported and two unsupported.
Under an all-claims-equal definition, claim support is 6/8 = 0.75.
If the two unsupported claims are the only safety-critical ones, the aggregate conceals the main risk.
Report the critical-claim slice as 0/2 = 0 and describe the predefined criticality criterion.
Suppose each of the eight claims has a citation but only six citations support their claims.
Citation coverage is 8/8 = 1, whereas supported-claim rate remains 6/8 = 0.75.
The difference demonstrates why link counting cannot establish factual support.

**ILLUSTRATIVE:** There are 40 answerable and 20 unanswerable questions in a held-out set.
A system answers 32 answerable questions correctly and abstains on the other eight.
It correctly abstains on 15 unanswerable questions and gives unsupported answers on five.
Answerable task success is 32/40 = 0.80 under this explicitly strict correctness definition.
No-answer correct-abstention rate is 15/20 = 0.75.
No-answer unsupported-answer rate is 5/20 = 0.25.
The overall count of correct actions is 32+15 = 47 out of 60, approximately 0.783.
The overall value hides the different error types, so retain the two slice metrics.
These example counts are constructed, not measured performance or evidence of deployment.

**ILLUSTRATIVE:** A prompt has a 4096-token combined limit and reserves 500 instruction tokens.
The question uses 100 tokens, serialized framing uses 96, and output reserve is 600.
The remaining evidence budget is 4096-500-100-96-600 = 2800 tokens by R3.
A selection of four 650-token passages plus 100 tokens of citation labels totals 2700.
The package fits with 100 tokens of remaining headroom under these assumptions.
Adding a fifth 650-token passage would exceed the budget by 550 tokens.
Do not silently truncate a source while presenting its citation as if the entire source was read.
A separate provider input limit or hidden framing can make the nominal calculation insufficient.

**ILLUSTRATIVE:** User U can read source A at retrieval time but loses access before response delivery.
A generated claim derived only from A must be removed or regenerated from another authorized source.
Retaining the fact while stripping the link is still an unauthorized disclosure.
If no safe reconstruction is available, suppress the entire affected response.
A response-time check cannot undo disclosure to a previously unauthorized model or logging service.
Every earlier processor therefore needs its own permitted-purpose and data-access check.

**ILLUSTRATIVE:** A cache contains a response derived from sources A and B.
Both were authorized for user U when generated, but B is later revoked for user U.
Even if the same user repeats the same query, the old answer cannot be replayed unchanged.
Reauthorize both contributing sources and regenerate any claims that depended on B.
If the cache stores only the answer text and no claim-to-source dependencies, selective removal is unreliable.
Under a fail-closed contract, invalidate that entire cached answer instead.

### Operational acceptance and change control

Define security invariants before setting relevance or latency targets.
Document which services may see each data class and what action each service may perform.
Test denied access at index filtering, reranking, model prompting, output replay, and citation opening.
Include access changes between retrieval and generation in the test schedule.
Include multi-tenant identifiers that deliberately collide in non-security display fields.
Record whether policy failure returns an access-safe error and whether retries preserve the same restriction.
Test policy cache expiration and invalidation with delayed and reordered events.
Test parent expansion where a neighboring chunk has a stricter policy than the matched chunk.
Test logs and traces for leakage using synthetic markers rather than real secrets.
Test that source deletions propagate to all indexes and caches within the defined contract.
Test stale and newly effective source versions using questions with explicit as-of dates.
Test that chunk revisions preserve citations or return an honest unavailable state.
Use versioned configuration for model, index, policy integration, and evaluation definitions.
Deploy only after the responsible owners accept unresolved risks and define operational guardrails.
A canary or shadow comparison must respect the same authorization boundaries as the normal path.
Do not send protected inputs to an experimental provider without the required processing authorization.
A shadow system's outputs and logs can leak data even when no answer is shown to the user.
Track quality, permission denials, policy errors, source freshness, and cost by supported slice.
Alert on a security invariant breach independently of whether average retrieval quality improves.
A rollback should restore a known compatible combination of query encoder, index, and chunk mapping.
Rolling back only the query encoder while retaining incompatible document vectors can corrupt retrieval.
Keep prior artifacts only for their approved retention period and access scope.
After rollback, recheck source and permission freshness; old index snapshots may be unsafe.
For an incident, preserve minimal authorized evidence and identify affected recipients and source versions.
Contain unauthorized processing first, then investigate retrieval and generation quality separately.
Do not describe a hypothetical procedure as an incident that actually occurred.

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

### Question 02: How many chunks cover 1000 tokens with length 300 and overlap 50?

**Answer:** Four chunks, by R2: 1+ceil((1000-300)/(300-50))=4.
**Reasoning:** The starts are 0, 250, 500 and 750.
**Application:** The last chunk ends at token 1000 and may be shorter than 300.
**Boundary or alternative:** A structure-aware splitter can produce a different count.
**Common error:** Red flag: divide by 300 while ignoring overlap.

### Question 03: Why can a high precision at one still be insufficient?

**Answer:** A question can require multiple supporting facts.
**Reasoning:** In Section 5, the first result is relevant so precision is 1.
**Application:** It finds one of two relevant chunks so recall is 0.5.
**Boundary or alternative:** A second relevant result raises recall to 1 for that example.
**Common error:** Red flag: treating one relevant hit as complete answer evidence.

### Question 04: What should happen when an unauthorized passage is the closest vector?

**Answer:** Exclude the passage before any unauthorized downstream disclosure.
**Reasoning:** The policy decision depends on principal, resource, action and time.
**Application:** Retrieve within the authorized scope and recheck on disclosure.
**Boundary or alternative:** If policy state is unavailable, follow a fail-closed contract.
**Common error:** Red flag: let an external reranker read it and filter the final response.

### Question 05: Can a cross-encoder repair a low-recall candidate set?

**Answer:** Not if it can only reorder the candidates already provided.
**Reasoning:** The final set is a subset of that pool under the stated premise.
**Application:** It can improve ordering when the needed source is already present.
**Boundary or alternative:** An additional retrieval or expansion stage changes the premise.
**Common error:** Red flag: increasing reranker compute without checking candidate recall.

### Question 06: How do you compare two different chunkers fairly?

**Answer:** Use stable source spans and map each candidate chunk to the same evidence labels.
**Reasoning:** Split evaluation by document family or time to reduce leakage.
**Application:** Measure source-span recall, answer support, latency and context cost.
**Boundary or alternative:** Re-adjudicate uncertain mappings and report incomplete labels.
**Common error:** Red flag: compare recall on old chunk IDs after redefining boundaries.

### Question 07: Compute the RRF score for ranks 1 and 3 with K=60.

**Answer:** The score is 1/61+1/63=124/3843, approximately 0.032266.
**Reasoning:** The rank contributions are approximately 0.016393 and 0.015873.
**Application:** A document appearing only at rank 1 scores approximately 0.016393.
**Boundary or alternative:** The score is not a relevance probability.
**Common error:** Red flag: treating K as the number of results rather than a smoothing constant.

### Question 08: Why are source citations alone not enough?

**Answer:** A cited source may not entail the associated claim.
**Reasoning:** The cited span may be outdated, partial, irrelevant or inaccessible.
**Application:** Evaluate each material claim against the specific version and span.
**Boundary or alternative:** A response should qualify or abstain where evidence does not support the claim.
**Common error:** Red flag: counting links instead of checking support.

### Question 09: A retrieval metric improves but users receive worse answers; diagnose.

**Answer:** First separate candidate recall, selection, generation and claim support.
**Reasoning:** An increase in candidates can add obsolete or distracting evidence.
**Application:** Compare paired traces on the same eligible queries and source snapshots.
**Boundary or alternative:** Audit whether required evidence survives context packing and whether citations entail claims.
**Common error:** Red flag: infer answer improvement from a single retrieval metric.

### Question 10: How does revocation affect retrieval caches?

**Answer:** Previously cached content can become unauthorized after the policy changes.
**Reasoning:** Cache keys must capture relevant scope and policy state, but keys alone may not handle revocation.
**Application:** Reauthorize cached material before disclosure or invalidate on a trusted policy event.
**Boundary or alternative:** Evaluate multi-user isolation and revocation races, including citation access.
**Common error:** Red flag: assume a cache hit preserves the original permission indefinitely.

### Question 11: How should a system answer when two current sources conflict?

**Answer:** Check scope, effective dates, jurisdiction and authority before treating them as contradictions.
**Reasoning:** If both apply and no rule resolves the conflict, disclose uncertainty or escalate.
**Application:** Do not arbitrarily select the passage with the higher cosine score.
**Boundary or alternative:** A product-specific policy may identify an authoritative source if that rule is supported.
**Common error:** Red flag: hide conflict because one source ranked first.

### Question 12: How would you measure ANN approximation separately from embedding quality?

**Answer:** For sampled queries, run exact similarity search over the same authorized vector population.
**Reasoning:** Compare ANN neighbors with exact neighbors at the same k and metric.
**Application:** Then compare both lists to relevance labels, which answers a different question.
**Boundary or alternative:** Report latency, filter semantics and slice sizes with the approximation metric.
**Common error:** Red flag: interpret poor relevance as proof that ANN approximation is at fault.

### Question 13: What changes if the token budget is cut in half?

**Answer:** Recompute available evidence using R3 and the actual tokenizer.
**Reasoning:** Preserve mandatory instructions and output reservation before packing evidence.
**Application:** Select complementary current passages and avoid redundant overlap.
**Boundary or alternative:** If necessary facts no longer fit, answer partially or abstain according to the product policy.
**Common error:** Red flag: silently truncate the last evidence passage and keep its citation.

### Question 14: Why is a no-answer query a separate evaluation slice?

**Answer:** Its relevant eligible count is zero, so the R1 recall denominator is zero.
**Reasoning:** Measure appropriate abstention and unsupported-answer rates on this slice.
**Application:** Verify whether the fact is genuinely absent or merely missed by retrieval.
**Boundary or alternative:** Report the policy population and time at which absence was judged.
**Common error:** Red flag: define undefined recall as zero without documenting a convention.


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
