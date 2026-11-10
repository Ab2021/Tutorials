---
case_id: "CASE-49-002"
case_title: "Secure RAG design review: deciding where the authorization boundary belongs"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "AI security / retrieval-augmented generation over access-controlled corpora"
primary_pattern: "authorization-scope"
secondary_patterns: ["data-leakage", "evaluation-mismatch"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Platform security architect chairing the retrieval design review"
source_index: "INDEX.md"
review_trigger: "Any change to the identity provider, document ACL model, chunking pipeline, index engine, cache key, or revocation service level."
---

# Secure RAG Design Review: Where the Authorization Boundary Belongs

## 1. The One-Line Framing

Given a natural-language question plus an authenticated principal (a person, a service identity, and an effective document-ACL set evaluated at query time), decide **which architectural boundary enforces retrieval authorization** — inside the vector index before candidate truncation, after retrieval but before generation, or after generation — for a security review board that must approve or reject the retrieval architecture, subject to a hard requirement that no unauthorized document text reaches a model context window and a p95 latency ceiling of 600 ms, compared with the shipped baseline that ranks globally and filters by ACL after the top-k list is formed.

The task category is **system operation and admission control**, not text generation: a per-request security decision that is made several hundred thousand times per day and that must fail closed. The primary success condition, stated plainly, is that for every response the set of chunks placed in the generation context is a subset of the chunks the authenticated principal is permitted to read *at that instant*. Section 9 gives the operational definition, numerator, denominator, window, and the guardrails that prevent the security metric from being won by refusing everything.

Three claims need separating before any number appears.

First, the **scenario is constructed**. No company, vendor, incident, benchmark run, or deployment is represented. The corpus size, query volume, selectivity fractions, latency components, and prices are teaching values chosen so the arithmetic is auditable by hand. They are labelled ILLUSTRATIVE wherever they enter a calculation.

Second, the **arithmetic is DERIVED**. Every probability, count, byte total, and cost below is a closed-form consequence of the stated inputs. A reader can recompute all of it with a calculator; Section 6.9 restates the inputs so the reconstruction does not depend on trusting the prose.

Third, **no outcome is measured**. The review described here produces a design decision and a validation protocol. It does not produce a deployed system, a penetration-test result, or a realised risk reduction. Section 11 is explicit about that gap, and Section 10 records what was and was not approved.

The decisive tradeoff, stated up front because it drives everything else: the two cheap-looking boundaries fail for *different* reasons and each failure is invisible in the metric the other one optimises. Post-retrieval filtering preserves relevance and destroys confidentiality, because the unauthorized chunks have already entered the service, its logs, and its caches. Post-generation filtering preserves nothing at all, because a model that has read a document can paraphrase it, and a filter cannot reliably detect paraphrase. Only a pre-truncation boundary keeps unauthorized text out of the context window, and it costs either latency (oversampling) or recall (aggressive filtering). The review's job is to price those two costs honestly instead of choosing the boundary that makes the demo look good.

Why the obvious simpler framing misses the point: a team that asks "does the assistant leak documents?" will build a test that shows the assistant refusing to answer questions about documents the tester cannot open. That test passes under all three architectures, because a well-behaved model will usually decline. The failure that matters is not a refusal or an answer — it is a *chunk entering the context window*, which is invisible in the transcript, cheap to trigger, and logged by infrastructure the security team does not own. The unit of analysis must therefore be the retrieved chunk, not the finished sentence.

The primary reusable pattern is **authorization-scope**: an effect is permitted only when an executor independent of the model checks an authenticated principal against an explicit scope, quota, and expiry. The secondary patterns are **data-leakage**, the mechanism by which a component with different visibility rules from the caller's leaks across that boundary, and **evaluation-mismatch**, the mechanism by which a passing test suite measures a proxy that the deployed system does not actually satisfy.

## 2. Context & Background

### 2.1 The system under review

An internal knowledge assistant answers questions over an access-controlled document corpus. The corpus is large enough that retrieval is mandatory rather than optional: answers must cite specific passages, and the model cannot be given the whole corpus. The retrieval layer is therefore a security boundary in the precise sense that it decides which bytes a language model is allowed to read, and the language model's output is then shown to a human who may not be allowed to read those bytes.

The architecture under review has six moving parts:

| Component | Responsibility | Trust position |
|---|---|---|
| Identity provider | Issues the session token; asserts principal ID, workspace, role set | Trusted for identity only |
| Document store | Holds source documents with owner, classification, and ACL entries | Trusted for content, not for authorization |
| Chunk index | Vector and lexical index over chunks derived from documents | Not trusted; must be constrained |
| Policy decision point (PDP) | Evaluates principal attributes against document ACL at query time | Trusted for the authorization decision |
| Answer generator | Language model that receives a context window and returns prose with citations | Untrusted; treats context as data, never as instructions |
| Effect executor | Any tool that acts on the world (ticket creation, export, email) | Must re-authorize independently |

The important structural fact is that the identity provider is trusted for *who you are*, the PDP is trusted for *what you may read*, and everything else — including the model, the retrieved text, and any document that contains adversarial instructions — is untrusted. Most real RAG incidents begin with a design that quietly promotes the model or the retrieved text into the trust column it does not occupy.

### 2.2 The workflow the decision changes

Before the decision, a request flows as follows. A user opens the assistant inside an internal workspace, types a question, and the front end forwards the question with the session token. The retrieval service embeds the question, runs an approximate nearest-neighbour search over the whole index, takes the top 40 candidates, filters out any chunk whose parent document is not readable by the principal, keeps the top 8 of what remains, and passes those 8 chunks to the generator. The generator writes an answer with inline citations. The user sees the answer and can click a citation, at which point the document store re-checks the principal and either serves the document or returns a permission error.

After the decision, the retrieval service does not rank over the whole index on behalf of the principal. It derives an eligibility set from the PDP, restricts candidate generation to that set before truncation, and passes only eligible chunks forward. The citation click still re-checks at the document store, because retrieval-time permission and display-time permission are different moments and a document can be reclassified between them.

The decision therefore changes three things: the shape of the index (partitioned or filtered), the shape of the request (an eligibility token that travels with the query), and the shape of the audit record (which must now show the eligibility decision, not just the returned chunks).

### 2.3 Stakeholders and their actual incentives

| Stakeholder | What they own | What they optimise | What they will trade away if unsupervised |
|---|---|---|---|
| Platform security architect | The retrieval trust boundary | No unauthorized text in a context window | Latency and relevance, without measuring the cost |
| Retrieval engineering lead | Index, ranking quality, p95 | Answer quality and speed | Security properties that are invisible in the relevance dashboard |
| Data governance lead | Classification, ACL correctness, retention | Correct labels and defensible audit | Runtime enforcement, which they assume engineering did |
| Workspace administrator | Role membership inside a workspace | Easy onboarding | Fine-grained ACL granularity |
| Model risk / compliance reviewer | Evidence for the approval file | A test campaign that can be described in a filing | Coverage of adversarial cases, which are expensive to build |
| Support operations | Ticket volume | Fewer "access denied" complaints | Security strictness, because denials look like bugs |

The conflict that makes this a real decision rather than a checklist: the metric the retrieval team watches (recall of relevant chunks, p95 latency) and the metric security watches (unauthorized-context rate) move in opposite directions when the boundary is chosen badly, and both metrics can be improved by degrading the other. A review that does not fix both metrics with pass rules at the same time will be resolved by whoever ships first.

### 2.4 Baseline and its documented weaknesses

The baseline is the shipped architecture described in 2.2: global ranking, then ACL filtering of the top 40, then truncation to 8. It has three weaknesses that are visible in design review without any measurement.

The first is that the filter is applied *after* the candidate list is materialised, so the service handles unauthorized text on every request from a narrow-scope principal. Whether that text is written to a trace, a debug log, a semantic cache, or a reranker's feature store depends on components the security team does not control, and each of those components is a place where the text can be retained for a period nobody has specified.

The second is arithmetic: to obtain 8 eligible chunks by filtering a list built without regard to eligibility, you must fetch more than 8. The shipped code fetches 40, which happens to be approximately 8 divided by a *global average* eligible fraction. That constant is a mean-matching rule, and Section 6.3 shows it under-fills roughly half the time for the principal population that matters most.

The third is that the filter's correctness depends on a chunk inheriting its parent document's ACL, and the chunker is owned by a team whose definition of "parent" changed when they moved from page-level to heading-level splitting. Nothing in the pipeline asserts that the inherited label is the maximum over ancestors.

### 2.5 Scale, timing, and boundaries

The constructed scale used throughout: 60,000 documents, 4.8 million chunks, average 420 tokens per chunk, 768-dimensional embeddings at 4 bytes per component, 1,000 workspaces, and 120,000 queries per day with a peak arrival rate of 25 queries per second. The corpus is re-indexed nightly and incrementally on document change. Retention of retrieval traces is a scenario parameter; the review treats any retention longer than zero as an exposure surface.

Geographic and regulatory boundaries are deliberately left unspecified, because the case does not need them and inventing them would manufacture legal conclusions the scenario cannot support. What matters for the decision is that the corpus contains documents whose access is *conditional on the reader*, which is a property of the data model, not of any jurisdiction.

## 3. The Business Question

The question is reconstructed from the review packet, not quoted from a real meeting. It is an illustrative stakeholder question and is labelled as such.

> "Our assistant is answering questions using passages from documents the person asking cannot open. We can keep the current filter, move the check earlier, or put a filter on the output. Which one do we ship, and what evidence do we need before you will sign it?"

| Decision contract | Detail |
|---|---|
| Question | Which boundary — pre-truncation index filtering, post-retrieval pre-generation filtering, or post-generation output filtering — enforces document authorization for the assistant? |
| Decision owner | Platform security architect, with the retrieval engineering lead as co-signer and the data governance lead as consulted owner of the ACL model |
| Trigger and deadline | An internal audit found assistant traces containing document titles from workspaces the reviewer could not open; the review board meets once, six weeks before the planned expansion from 200 to 1,000 workspaces |
| Available actions | (a) keep post-retrieval filtering and add logging controls; (b) move authorization before truncation with a partitioned index; (c) add an output filter and defer the indexing work; (d) restrict the assistant to a single trusted corpus and defer multi-workspace access |
| Value and downside | Expansion unblocks roughly 800 workspaces of knowledge work. A confirmed cross-workspace disclosure triggers a notification obligation whose scope is unknown, freezes the expansion, and forces a corpus-wide ACL re-audit |
| Objective | Drive the unauthorized-context rate to a demonstrated upper bound at or below 0.1 percent without dropping authorized-recall below the agreed floor or breaching the p95 latency ceiling |
| Hard constraints | No unauthorized chunk in a generation context; no cross-workspace cache reuse; revocation effective within 30 seconds; p95 end-to-end at or below 600 ms |
| Scope | One assistant, one corpus, 1,000 workspaces, read-only question answering with citations; excludes any tool that writes to an external system |
| Out of scope | Export of documents, model fine-tuning on corpus content, legal notification decisions, and the retention policy for the document store itself |

### 3.1 Objective versus proxy

The objective is a risk reduction: fewer requests in which a principal's question causes text they may not read to be placed in front of a model. The proxy the team can actually count is the unauthorized-context rate on a labelled evaluation set. Those two quantities diverge in a specific and predictable way, and the divergence is the reason this case exists.

The proxy is measured on queries someone thought to write. Real traffic contains queries about documents the evaluation set never mentions, principals whose role sets are unusual, and documents reclassified after the evaluation set was frozen. A measured zero on the proxy therefore bounds the risk only for the population the set represents, which is why Section 9 treats the zero-event sample size as a first-class deliverable rather than a footnote.

The reverse divergence matters too. A team can drive the proxy to zero by making the assistant refuse whenever eligibility is ambiguous. That is a real security posture and a business failure, so the review pairs the security metric with an over-blocking guardrail and a recall floor. Without the pair, the security metric is trivially winnable and therefore worthless as an acceptance criterion.

### 3.2 What an expensive false decision looks like

A false positive here is "we approved the architecture and it leaks". Its cost is a disclosure event whose blast radius is measured in documents, not requests, because a single leaked passage can be quoted, forwarded, and retained outside any system the team can revoke.

A false negative is "we rejected a sound architecture and delayed the expansion". Its cost is six weeks of engineering and a delayed productivity programme. Those costs are not symmetric, and the review should say so explicitly rather than pretending to a balanced trade: the architecture must be rejected on *demonstrated* leakage, not on the absence of proof that it does not leak, but the burden of evidence sits with the proposer.

## 4. Assumptions & Constraints

The table below separates things the team believes (assumptions, testable), things imposed on the team (constraints, with a source), and things the team chose (design parameters, revisable). Mixing these three is how design reviews produce decisions nobody can revisit.

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | The session token's principal and workspace claims cannot be influenced by model output or retrieved text | Identity | Attempt to inject a workspace claim through a retrieved document and through a tool argument; assert the PDP never sees it | Unverified | Every downstream check is void; the boundary can be steered by an attacker who can write one document | Disable multi-workspace retrieval | Identity platform |
| A-02 | Every chunk carries a label inherited from its parent document, and the label equals the maximum over all ancestors | Data model | Recompute labels bottom-up for a 1,000-chunk sample and diff against stored labels | Unverified | A reclassified document's old chunks remain readable at the old label | Re-derive labels on every index build and block the build on a non-zero diff | Data governance |
| A-03 | The PDP can answer an eligibility query in under 10 ms at p99 for a set of up to 5,000 document IDs | Infrastructure | Load test the PDP with the real principal-attribute distribution | Unverified | The pre-filter design cannot meet the latency ceiling and the review's recommendation collapses | Cache eligibility per (principal, ACL version) with a bounded TTL | Platform |
| A-04 | The index engine can restrict candidate generation to a supplied ID set without degrading recall beyond the agreed floor | Retrieval | Compare filtered results against exhaustive scoring on the same eligible set | Unverified | The pre-filter design silently loses recall and the team compensates by widening the filter | Partition the index physically instead of filtering logically | Retrieval engineering |
| A-05 | Chunk-level deduplication does not merge text from two documents with different ACLs | Data model | Search for chunks whose provenance list contains more than one document ID | Unverified | One chunk can carry two labels, and the weaker label wins in practice | Disable cross-document dedup | Data governance |
| A-06 | The semantic cache key includes workspace and principal-class | Infrastructure | Replay a cross-workspace query pair and assert a miss | Unverified | Answer reuse across tenants, which no retrieval filter can prevent | Disable the cache | Platform |
| A-07 | Revocation propagates to the PDP within the stated window | Infrastructure | Revoke a grant and poll eligibility until it flips; repeat 50 times and take the maximum | Unverified | A revoked reader keeps reading for the propagation window | Shorten TTL to the measured propagation time | Data governance |
| A-08 | The generator's citation rendering cannot reproduce more than a short quotation from a chunk it was not given | Model behaviour | Adversarial prompt asking the model to recite context; measure longest verbatim span against a source chunk | Unverified | Post-generation filtering becomes load-bearing, which the review rejects | Constrain decoding and citation format; treat as a residual risk | Model risk |
| C-01 | No unauthorized chunk may enter a generation context | Security | Labelled evaluation set plus adversarial suite | Confirmed requirement | The decision is void | None; this is the gate | Platform security |
| C-02 | p95 end-to-end latency at or below 600 ms at 25 queries per second | Performance | Replay load test with the production query mix | Scenario constraint | The pre-filter design must change shape, not just move | Return fewer chunks or precompute role indexes | Retrieval engineering |
| C-03 | Revocation effective within 30 seconds | Governance | Revocation propagation test | Scenario constraint | Stale-authorization exposure grows linearly with the window | Reduce TTL and accept more PDP load | Data governance |
| C-04 | No cross-workspace cache reuse of any kind | Security | Negative test on the cache with a synthetic collision | Confirmed requirement | Tenant isolation is lost regardless of index design | Partition the cache per workspace | Platform |
| C-05 | Retrieval traces may not retain chunk text | Privacy | Inspect the trace schema and a week of sampled traces | Scenario constraint | The trace store becomes a shadow copy of the corpus | Store chunk IDs and digests only | Observability |

### 4.1 The assumptions most likely to reverse the recommendation

Two assumptions can flip the decision on their own, and the review should name them in the first paragraph rather than burying them.

**A-04 is the load-bearing one.** The entire argument for moving authorization before truncation rests on the claim that an index can restrict candidate generation to an eligible set *without* destroying recall. If the engine's filtered search degrades badly at low selectivity, the team will face a choice between poor answers and a wide filter, and the wide filter is exactly the baseline that was rejected. The discriminating test is not a vendor claim; it is a comparison against exhaustive scoring over the same eligible set at three selectivity levels (0.5, 0.05, 0.0125). If filtered recall falls below the floor at the narrow end, the design must switch to physical partitioning for narrow roles rather than argue about it.

**A-02 is the quiet one.** If chunk labels do not equal the maximum over ancestors, then a perfectly implemented pre-filter returns chunks whose stored label says "internal" for a document that was reclassified to "restricted" yesterday. The security metrics will look excellent because the labels they trust are wrong. The discriminating test is a bottom-up recomputation over a sample with a deliberate reclassification injected; the review should not accept a sample that contains no reclassification, because the failure only appears there.

### 4.2 Constraints that shape the design rather than the decision

The latency ceiling (C-02) does not by itself select a boundary, but it selects an *implementation* of the chosen boundary. Section 8 shows that post-retrieval oversampling costs 320 ms at the mean-matching fetch width and 560 ms at the width that actually achieves the recall target, against a 210 ms pre-generation stage budget. The ceiling is therefore what converts a security argument into an engineering redesign rather than a configuration change.

The revocation window (C-03) interacts with query rate in a way that is easy to underestimate. A 30-second window at 25 queries per second means up to 750 requests are answered while a revocation is in flight, and the mean number is about half that. Whether that is acceptable depends on how many of those requests would have touched the revoked documents, which is a property of the query mix, not of the revocation system.

### 4.3 Constraints deliberately not asserted

The case does not assert a retention obligation, a notification duty, a data-residency rule, or a sector-specific control framework. Those determinations depend on facts the scenario does not contain, and inventing them would produce a case that looks authoritative and is not. Where a control is described as required, it is required by the scenario's own design contract (C-01 through C-05), not by an external authority.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is **one retrieval request**, keyed by a request ID, with a nested record per candidate chunk. That grain matters: a request-level pass/fail hides the fact that one request can place four unauthorized chunks in a context window, and a chunk-level metric without request grouping overstates independence because chunks from the same document are correlated. Every metric in this case is computed at request level and reported with the chunk count alongside.

| Field | Specification for this review |
|---|---|
| Unit of analysis | Retrieval request; nested array of candidate chunks in rank order; one context-window record listing the k chunks actually passed to the generator |
| Volume and window | 4.8 million chunks over 60,000 documents; 1,000 workspaces; 120,000 queries per day; evaluation window is a frozen synthetic traffic week plus a 4,000-request labelled sample |
| Inputs | Question text; principal ID; workspace ID; role set; session issuance time; ACL version observed at query time; query embedding; candidate chunk IDs in rank order; per-chunk parent document ID, stored label, and inherited label |
| Outcome and label | *Unauthorized context* = the count, per request, of distinct chunks in the generation context whose parent document is not readable by the principal under the ACL version observed at query time, adjudicated against an oracle that re-evaluates the ACL independently of the retrieval service |
| Population and exclusions | All requests from principals with at least one workspace role. Excluded: service-to-service requests without a human principal (no ACL semantics), requests served entirely from cache (no retrieval occurred), and requests that failed before the generation stage |
| Join semantics | Chunk → document is many-to-one and must be resolved through the provenance table, not by parsing the chunk ID; duplicates are possible and are collapsed by (request, chunk ID) before counting chunks but not before counting requests |
| Splits | Evaluation set is frozen per release; the adversarial suite is held out and never used for tuning the eligibility threshold; the recall gold set is sampled per role band so the narrow-selectivity band is not swamped by broad-selectivity requests |
| Time semantics | Eligibility is evaluated at query time, not at index-build time; the ACL version used for the oracle is the one effective at the request timestamp, resolved to the second |

The evaluation population is deliberately stratified by **selectivity band**, where selectivity `f` is the eligible fraction of the corpus for that principal. The constructed traffic mix used in every weighted calculation below is: 20 percent of requests at `f = 0.0125` (a narrow single-project role), 60 percent at `f = 0.06` (a department role), and 20 percent at `f = 0.50` (a broad reader role). This mix is an ILLUSTRATIVE planning input. It is not measured traffic, and Section 6.7 shows what happens to the conclusion when it is wrong — which it will be, because narrow roles typically grow after a launch as teams add project-level restrictions.

The gold set for recall is 400 questions with hand-adjudicated relevant chunks, drawn 160/160/80 across the three bands so the narrow band is adequately represented. At 400 items the 95 percent Wilson interval around a measured recall of 93.0 percent is [90.07, 95.11] — an eight-percentage-point span. Section 9 explains why that width is sufficient to choose between architectures but not sufficient to certify a recall target of 94 percent.

### 5b. Tools & Stack

| Component | Proposed tool class | Role in the decision | Version assertion |
|---|---|---|---|
| Identity | OIDC provider issuing short-lived session tokens with principal and workspace claims | Supplies the authenticated principal; must be the only source of tenant identity | Behaviour asserted, not version-pinned |
| Policy decision point | Purpose-built authorization service with document-level ACL evaluation and a bulk "can-read-any" endpoint | Answers eligibility for a principal–document set; must return within the stage budget | Latency budget is a requirement, not a measurement |
| Chunk index | Approximate nearest-neighbour index with metadata filtering plus a lexical index | Candidate generation; the filtering capability is the crux of A-04 | Capability assumed; quantified by the A-04 test |
| Reranker | Cross-encoder scoring top candidates | Restores ranking quality after oversampling | Throughput assumed at 2,000 pairs/s |
| Generator | Instruction-following language model with a fixed context budget | Produces the answer and citations | Model identity irrelevant to the decision; context budget is what matters |
| Trace store | Structured event log keyed by request ID | Audit and diagnosis | Schema must exclude chunk text (C-05) |
| Semantic cache | Key-value store for repeated questions | Latency and cost reduction | Key must include workspace and principal class (A-06) |
| Load harness | Replay tool driving the production query mix at 25 queries per second | Latency and capacity evidence | Not executed in this case |

No product version, price, or benchmark number for any of these components is asserted. The prices used in Section 9 are constructed teaching values with stated units, and the throughput figure for the reranker is an assumed parameter that a real review would replace with a measured one. The distinction matters because a design review that cites a vendor's published latency is citing a number measured on a workload nobody in the room controls.

### 5c. Method Summary

The method has four stages, and the decision is about where the second one sits relative to the third.

```text
question + token
   -> [1] authenticate and resolve principal, workspace, role set
   -> [2] derive eligibility (PDP) and express it as an index constraint
   -> [3] generate and truncate candidates (ANN + lexical, then rerank)
   -> [4] assemble context, generate, cite, audit
```

| Candidate | Mechanism | Strength | Limitation / avoid when | Status |
|---|---|---|---|---|
| Post-retrieval ACL filter (baseline) | Rank globally, take 40, drop ineligible, keep top 8 | Zero index work; preserves relevance exactly | Unauthorized text is materialised and handled on every narrow-scope request; constant fetch width under-fills | Rejected by reasoning and arithmetic |
| Pre-truncation metadata filter | Pass an eligible-ID set into the ANN search so filtering happens before the top-k cut | Unauthorized text never enters the pipeline; correct by construction when labels are correct | Costs oversampling at low selectivity or recall when the engine filters poorly; needs a fast PDP and a bounded ID set | Chosen |
| Physical partitioning | One index partition per workspace or per role, queried only when the principal is a member | Removes the large-ID-set problem entirely; strong isolation | Multiplies index count and nightly build work; cross-workspace queries must fan out and merge | Chosen as the narrow-band supplement |
| Post-generation output filter | Generate freely, then detect and redact unauthorized content | Requires no index change | Cannot reliably detect paraphrase; the model has already read the text | Rejected |
| Single-corpus restriction | Only index one broadly readable corpus | Simplest possible control | Discards the product's value across 800 workspaces | Deferred as a fallback |

The chosen method is a **hybrid**: metadata-filtered candidate generation for broad and department roles, plus a physically separate partition for narrow roles where the eligible set is small enough that filtered ANN degrades. The review rejected a pure-filter design and a pure-partition design for the same underlying reason — each is correct only on one side of the selectivity range — and Section 8.3 records that trade explicitly.

### 5d. Data provenance, quality, exclusions and leakage controls

| Concern | Handling in this review |
|---|---|
| Provenance of corpus metadata | Document ID, owner, classification, and ACL entries come from the document store's change feed; the review treats the store as authoritative for content and the governance team as authoritative for labels |
| ACL correctness | Not assumed. A-02 and A-05 exist because the chunker's notion of "parent" changed and cross-document dedup can merge labels |
| Missingness | Documents with no ACL entry are treated as readable by nobody, not as readable by everybody; the count of such documents is a required build-gate statistic |
| Duplicates | Cross-document dedup is disabled pending the A-05 probe; intra-document duplicates are collapsed by content digest before embedding |
| Freshness | Chunk labels are recomputed on every index build; a non-zero diff against the previous build blocks promotion |
| Permissions | The evaluation set is built from synthetic documents with synthetic principals; no real document text is used anywhere in this case |
| Retention | Traces store chunk IDs, digests, scores, and the eligibility-set digest — never chunk text (C-05) |
| Unavailable data | Real query logs, real ACL distributions, real revocation propagation times, and any real incident record are unavailable and are not simulated in a way that implies measurement |

Two leakage controls apply specifically to this evaluation.

The first is **label leakage into the retrieval features**. If the reranker or the ANN index is trained or tuned on a corpus that includes the evaluation questions' answers, the measured recall is inflated. The review requires that the gold-set questions be excluded from any index-build-time query log aggregation and that the reranker's training cutoff precede the gold-set construction date.

The second is **oracle independence**. The unauthorized-context oracle must evaluate ACLs directly against the document store, not call the retrieval service's own eligibility endpoint. An oracle that asks the system under test whether it was allowed to do what it did will agree with the system under test by construction, and this is the single most common way a RAG security evaluation produces a false pass.

### 5e. Reproducibility and method-selection ledger

| Artifact | Location in this document | Version or date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-01` | OWASP Top 10 for Large Language Model Applications, LLM01 prompt injection and LLM02 insecure output handling | Accessed 2026-11-10 | Threat vocabulary for untrusted content | Public | A taxonomy, not evidence about this design |
| `SRC-02` | NIST AI Risk Management Framework, NIST AI 100-1, 2023 | Accessed 2026-11-10 | Risk-management vocabulary for the approval file | Public | Not a control catalogue for retrieval |
| `SRC-03` | scikit-learn common pitfalls guide | Accessed 2026-11-10 | Leakage vocabulary applied to the evaluation split | Public | Method reference only |
| `SRC-04` | Wilson score interval, standard binomial proportion method | Standard result | Zero-event and recall intervals | Public | Assumes independent Bernoulli trials per item |
| `RUN-01` | Formula blocks SR-01 through SR-08 and the arithmetic in Section 6 | 2026-11-10 | Every quantity used in the decision | Recomputable by hand | Not an executed system run; no latency measured |
| `RUN-02` | Synthetic traffic mix (20/60/20 across selectivity bands) | 2026-11-10 | Weighted expected-value calculations | Stated in 5a | Constructed, not observed |

Reproduction of the *reasoning* is complete: every input is stated, every formula is written out, and every substitution is shown. Reproduction of the *measurement* is not possible and is not claimed, because no measurement occurred. A reader who wants to run the evaluation would need the four probes named in Section 9 — the eligibility-error probe, the filtered-recall probe, the revocation propagation probe, and the cache isolation probe — each of which is specified with its input, its oracle, and its pass rule.

The method-selection ledger below records which decisions were made on evidence, which on arithmetic, and which on judgement. The third column is the honest one: several choices in this review are engineering judgement about which failure is easier to detect, and pretending otherwise would misrepresent the epistemic status of the design.

| Decision | Basis | Confidence | What would change it |
|---|---|---|---|
| Reject post-generation filtering | Mechanism reasoning: paraphrase defeats lexical and semantic comparison | High | A demonstrated detector with measured recall and precision on paraphrased restricted content |
| Reject post-retrieval filtering | Arithmetic (expected 34.46 unauthorized chunks per request) plus exposure reasoning | High | An index design that filters without materialising unauthorized candidates |
| Choose pre-truncation filtering | Correctness by construction plus a quantified latency cost | Medium | A-04 failing at the narrow band |
| Add physical partitioning | Recall arithmetic at f = 0.0125 requires m* = 1,049 | Medium | A partition-aware ANN that filters without recall loss at that width |
| Set k = 8 | Context budget and cost, not relevance | Low | Evidence that k = 12 improves answer quality enough to justify a 44 percent wider residual-leak term |

## 6. Analysis Walkthrough

The decisive result, stated before the derivation: the shipped baseline handles an expected **34.46 unauthorized chunks per request**, weighted over the stated traffic mix, and no amount of tuning the fetch width changes that number in the right direction — widening the fetch increases it, narrowing it destroys recall. The pre-truncation design reduces the quantity to zero by construction and replaces it with a residual term governed by label error, which is measurable and can be driven to the level required by the security gate. The cost of that substitution is 320 ms of rerank latency at the narrow band and a physical partition.

The walkthrough proceeds in eight steps. Each step names what it tests, what it produces, and what would falsify it. The formulas are numbered SR-nn and are restated in Section 6.9 with all inputs so the arithmetic can be reconstructed.

| Step | Question being tested | Method | Intermediate output | What changes if it fails |
|---|---|---|---|---|
| 1 | How much unauthorized text does the baseline handle? | Linearity of expectation over candidates | 34.46 chunks per request, mix-weighted | Nothing yet; establishes the size of the problem |
| 2 | Does the baseline's fixed fetch width serve the narrow band? | Binomial fill probability | m = 40 gives near-zero fill probability at f = 0.0125 | The baseline's recall claim collapses |
| 3 | What fetch width would be needed? | Smallest m with 95 percent fill | 1,049 at f = 0.0125; 162 at f = 0.08 | Determines whether post-retrieval filtering is even viable |
| 4 | What does that width cost in latency? | Rerank throughput at 2,000 pairs/s | 524.5 ms at m = 1,049 | Exceeds the whole 600 ms budget before generation |
| 5 | Can a pre-filter achieve the security gate? | Residual contamination with per-chunk error | Requires eps <= 1.25e-4 at k = 8 for 0.1 percent | The gate becomes unreachable and the design must restrict scope |
| 6 | What does each error source contribute? | Decomposition of eps into label, bypass, cache | Label 5e-4 dominates; cache is unbounded without a key fix | Reprioritises the engineering backlog |
| 7 | What is the staleness exposure? | Revocation window times query rate | 375 mean, 750 worst in-flight requests | Forces a shorter TTL and more PDP load |
| 8 | What does it cost? | Per-query token and compute arithmetic | 0.003282 currency units per query; 393.84 per day | Decides whether partitioning is affordable at 1,000 workspaces |

### 6.1 Step 1 — how much unauthorized text the baseline handles

The baseline does not show unauthorized text to the user. It *handles* it: the candidates are produced by an unrestricted search, scored by an unrestricted reranker, and only then filtered. The quantity that matters is therefore the expected number of ineligible candidates materialised per request, and it has an exact expectation that needs no independence assumption.

```text
FORMULA [SR-01]: H = m * (1 - f)
WHERE: H is the expected number of ineligible (unauthorized) chunks materialised
       per request [chunks/request]; m is the number of candidates fetched
       before filtering [chunks]; f is the eligible fraction of the corpus for
       the requesting principal, 0 < f <= 1 [dimensionless].
CONSTRAINT: H is exact by linearity of expectation for any candidate sampling
       scheme in which each candidate is ineligible with probability 1 - f.
       It requires no independence between candidates and no particular rank
       distribution. It does NOT describe what the user sees, only what the
       retrieval pipeline touches.
NUMERICAL EXAMPLE: m = 40, f = 0.06 gives H = 40 * 0.94 = 37.6 chunks/request.
       Mix-weighted over (0.2, f=0.0125), (0.6, f=0.06), (0.2, f=0.50):
       H_bar = 40 * [0.2*0.9875 + 0.6*0.94 + 0.2*0.50]
             = 40 * [0.19750 + 0.56400 + 0.10000]
             = 40 * 0.86150 = 34.46 chunks/request.
INTERPRETATION: at 120,000 requests/day the pipeline materialises about
       4.135 million ineligible chunks per day. Because the baseline's fetch
       width is fixed, the harm is concentrated in the narrow band, where a
       single request touches 39.5 ineligible chunks on average.
```

The derivation is one line and worth stating because it exposes an assumption people often add without noticing. Write the indicator `I_j = 1` when candidate `j` is ineligible. Then `H = sum_j I_j`, and `E[H] = sum_j P(I_j = 1) = m(1 - f)`. Nothing about the joint distribution enters. The only assumption is that each candidate is drawn with the principal's eligibility distribution — which is exactly the assumption the baseline's unrestricted search makes.

That last point is the sharp end. The formula is *not* saying the baseline is worse than a random sampler; it is saying the baseline samples from the full corpus and the principal's eligibility is an afterthought. The 39.5 ineligible chunks at the narrow band are not a bug in the filter; they are the arithmetic of fetching 40 items from a corpus where 98.75 percent of items are off-limits.

### 6.2 Step 2 — whether the fixed fetch width serves the narrow band

With a fixed fetch width, the number of eligible chunks obtained is a binomial-like count. If it falls below `k`, the context window is under-filled — the assistant has fewer passages than the design assumes, answers degrade, and the user sees a confident answer built on thin evidence.

```text
FORMULA [SR-02]: P_fill(m) = P(X >= k),  X ~ Binomial(m, f)
WHERE: P_fill is the probability that at least k eligible chunks are obtained
       from m candidates [probability]; k is the required context size
       [chunks]; f is the eligible fraction [dimensionless]; m is the fetch
       width [chunks].
CONSTRAINT: X is binomial when candidates are drawn with replacement from an
       effectively infinite corpus; for a finite corpus use the hypergeometric
       form, which is slightly less variable. Validity also requires that
       eligibility be independent of the ranking score, which is false in
       practice: a principal's own documents are often lexically similar to
       their questions. The binomial is therefore a LOWER bound on fill quality.
NUMERICAL EXAMPLE: k = 8, f = 0.0125, m = 40.
       E[X] = 40 * 0.0125 = 0.5 eligible chunks.
       P(X >= 8) = 1 - sum_{i=0}^{7} C(40,i) 0.0125^i 0.9875^(40-i)
                 < 1e-7 (the dominant term at i = 7 is below 1e-6).
       So the context is essentially never filled for this principal.
INTERPRETATION: the baseline's fetch width is a mean-matching rule against the
       AVERAGE eligible fraction, and the average is dominated by broad roles.
       For the narrow band it fails on nearly every request, which is why the
       same principals generate both under-filled answers and the complaints
       that drive teams to widen the fetch — increasing exposure.
```

The counter-intuitive consequence is worth spelling out because it drives the whole design. The narrow-band principal is the one whose data is most sensitive and the one whose assistant works worst. The natural remedy — "fetch more" — increases `H` linearly. The team is therefore pushed toward a fix that makes the security metric worse every time it makes the quality metric better, and no single dashboard shows both.

### 6.3 Step 3 — the fetch width that would actually be needed

If post-retrieval filtering is to be viable at all, it must fetch enough candidates that the filter still leaves `k`. That width is the smallest `m` satisfying the fill constraint.

```text
FORMULA [SR-03]: m* = min { m : P(X >= k) >= 1 - beta },  X ~ Bin(m, f)
WHERE: m* is the required fetch width [chunks]; beta is the tolerated
       under-fill probability [probability]; k and f as in SR-02.
CONSTRAINT: m* grows roughly as k/f for small f, with a variance correction
       that adds about z * sqrt(k(1-f)) / f, where z is the standard normal
       quantile for 1 - beta. Both the binomial and the normal approximation
       assume independent candidates, so m* is a LOWER bound.
NUMERICAL EXAMPLE: k = 8, 1 - beta = 0.95:
       f = 0.50   -> m* = 23
       f = 0.08   -> m* = 162
       f = 0.06   -> m* = 217
       f = 0.02   -> m* = 655
       f = 0.0125 -> m* = 1,049
       Check at f = 0.08: E[X] = 162 * 0.08 = 12.96; the normal
       approximation gives (8 + 1.645*sqrt(8*0.92)) / 0.08 = 155.8, and the
       exact binomial minimum is 162 — the approximation under-states by
       3.8 percent, which is the expected direction for a skewed count.
INTERPRETATION: relative to k = 8, the narrow band needs a 131-fold oversample
       (1,049/8). Any architecture that keeps post-retrieval filtering must
       pay that factor in candidates, and every extra candidate is another
       ineligible chunk handled per SR-01.
```

The 3.8 percent gap between the normal approximation and the exact minimum is a small but instructive detail: the normal approximation is applied to a count whose mean is 12.96 with a right tail, so it under-states the width needed for a 95 percent fill. A design that used the closed-form approximation without checking would under-fetch by about six candidates per request at the department band — not catastrophic, but exactly the kind of quiet shortfall that shows up as "the assistant sometimes gives thin answers" and is never traced back to a formula.

### 6.4 Step 4 — what the required width costs

The reranker is the component whose cost scales with candidate count. At an assumed throughput of 2,000 query–passage pairs per second per instance, the stage cost is linear in `m`.

```text
FORMULA [SR-04]: t_rerank = m / r
WHERE: t_rerank is rerank wall-clock time for one request [s]; m is the
       candidate count [pairs]; r is rerank throughput [pairs/s].
CONSTRAINT: linear scaling holds only below the throughput knee; above it,
       queueing dominates and latency grows superlinearly. The value of r is
       an assumed parameter here, not a benchmark, and batching across requests
       changes it in both directions (better utilisation, worse tail).
NUMERICAL EXAMPLE: r = 2,000 pairs/s.
       m = 40    -> 20.0 ms
       m = 162   -> 81.0 ms
       m = 217   -> 108.5 ms
       m = 655   -> 327.5 ms
       m = 1,049 -> 524.5 ms
INTERPRETATION: the narrow-band requirement alone consumes 524.5 ms of a
       600 ms end-to-end budget, leaving 75.5 ms for authentication, policy,
       ANN search, generation and serialisation. Since the generation stage
       alone is budgeted at 300 ms p95, post-retrieval filtering cannot meet
       C-02 for the narrow band. The design fails on latency, not only on
       exposure.
```

Put differently: post-retrieval filtering is not merely insecure, it is *also* the slow option at the selectivity levels that matter. That is a useful finding for the review because it removes the framing where security is asking engineering to accept a performance regression. The security-preferred design is the faster one at the narrow band, since it never reranks candidates it cannot use.

### 6.5 Step 5 — the residual contamination of a pre-truncation boundary

Moving the boundary earlier does not make leakage impossible; it makes leakage a function of *label correctness* rather than of rank geometry. The residual is governed by a per-chunk error rate.

```text
FORMULA [SR-05]: UCR = 1 - (1 - eps)^k
WHERE: UCR is the probability that at least one ineligible chunk enters the
       generation context for a request [probability/request]; eps is the
       per-chunk probability that an ineligible chunk passes the boundary
       [probability/chunk]; k is the context size [chunks].
CONSTRAINT: requires that chunk-level errors be independent of one another,
       which fails when one mislabelled document contributes several chunks
       to the same request. With clustering, UCR is LARGER than this formula
       at fixed eps; the formula is therefore optimistic and must be paired
       with a document-level check. Validity also requires that the boundary
       be the only path into the context, which is what the bypass audit tests.
NUMERICAL EXAMPLE: k = 8.
       eps = 5e-4 -> UCR = 1 - 0.9995^8 = 4.0e-3 (0.40 percent)
       eps = 1e-3 -> UCR = 1 - 0.999^8  = 8.0e-3 (0.80 percent)
       eps = 1e-4 -> UCR = 1 - 0.9999^8 = 8.0e-4 (0.08 percent)
       Requirement UCR <= 1e-3 implies eps <= 1 - 0.999^(1/8) = 1.25e-4.
INTERPRETATION: the security gate translates into a label-error budget. At
       k = 8 the corpus may tolerate about 1.25 mislabelled chunks per 10,000,
       which is a testable property of the indexing pipeline rather than a
       property of the model.
```

This is the single most useful transformation in the case: it converts a vague requirement ("don't leak") into a number a data-engineering team can own. A chunk-labelling pipeline can be tested for `eps`; a language model cannot be tested for "don't paraphrase".

The formula is also where the case's optimism must be declared. If a document has 40 chunks and its label is stale, one mislabelled document can deliver up to 8 ineligible chunks into a *single* request, so the independence assumption is wrong precisely in the scenario that matters. Section 6.6 introduces the document-level correction and Section 8 uses the conservative form.

### 6.6 Step 6 — decomposing the per-chunk error into controllable sources

`eps` is not one number. It is a sum of failure paths, and treating it as a single tunable constant hides which team owns which part.

```text
FORMULA [SR-06]: eps = eps_label + eps_bypass + eps_cache + eps_stale
WHERE: all terms are per-chunk probabilities of an ineligible chunk reaching
       the context [probability/chunk]. eps_label is stale or incorrect chunk
       inheritance; eps_bypass is a code path that reaches the generator
       without the eligibility constraint; eps_cache is reuse of a context
       assembled for a different principal or ACL version; eps_stale is a
       chunk whose document was revoked after the eligibility set was resolved.
CONSTRAINT: the terms are treated as additive, which over-counts when two
       causes coincide on the same chunk; that over-count is acceptable
       because the resulting eps is conservative for a security gate.
NUMERICAL EXAMPLE (constructed):
       eps_label  = 5.0e-4  (one stale label per 2,000 chunks)
       eps_bypass = 5.0e-5  (one legacy path per 20,000 chunks)
       eps_cache  = 0       (cache keyed per workspace and principal class)
       eps_stale  = 1.0e-4  (30 s window, 0.004 of requests touch a revoked doc)
       eps_total  = 6.5e-4
       UCR = 1 - (1 - 6.5e-4)^8 = 5.2e-3 (0.52 percent)
INTERPRETATION: the label term is 77 percent of eps, so the roadmap is a
       labelling pipeline fix, not a retrieval fix. With eps_label driven to
       5e-5 the total falls to 2.0e-4 and UCR to 1.6e-3; reaching the 0.1
       percent gate then requires also removing the stale term.
```

The decomposition changes the answer to "who fixes this". If the team treats `eps` as a single number and assigns it to the retrieval team, the retrieval team will spend its effort on ANN correctness — which is already sound — while the label pipeline, owned by data governance, remains the dominant term. The discriminating check is a controlled mislabelling injection: corrupt the label on a known document and verify that (a) the boundary blocks it and (b) the monitoring counts it. A boundary that blocks it but is not counted is not measurable, and an unmeasurable `eps` cannot support a gate.

### 6.7 Step 7 — staleness exposure during revocation

Eligibility is evaluated at query time, but the answer is used for a window. A grant revoked one millisecond after evaluation still permits every request already in flight.

```text
FORMULA [SR-07]: N_inflight = q * tau
WHERE: N_inflight is the number of requests that may be answered under a stale
       grant [requests]; q is the arrival rate [requests/s]; tau is the
       revocation propagation window [s].
CONSTRAINT: this is an upper bound on requests started inside the window. The
       MEAN is q*tau/2 under uniform arrival, which is the appropriate planning
       figure; the maximum is the correct figure for a worst-case statement.
       Neither counts requests that actually touch the revoked document.
NUMERICAL EXAMPLE: q = 25 requests/s, tau = 30 s.
       N_inflight(max)  = 25 * 30 = 750 requests.
       N_inflight(mean) = 25 * 30 / 2 = 375 requests.
       With h = 0.004 (the constructed fraction of requests that touch a
       document in the revoked set), expected affected requests are
       375 * 0.004 = 1.5 (mean) and 750 * 0.004 = 3.0 (worst).
INTERPRETATION: reducing tau from 30 s to 5 s cuts the expected affected
       requests from 1.5 to 0.25, at the cost of the PDP handling more
       eligibility resolutions. Since the PDP budget is 10 ms and the eligibility
       work is cached per (principal, ACL version), the shorter window is
       affordable; the binding cost is cache churn, not compute.
```

The instructive part is the multiplication by `h`. A revocation window stated in seconds sounds alarming in isolation and sounds trivial when multiplied by the fraction of traffic that touches the affected documents. Both framings are wrong on their own: the seconds framing ignores that most requests never touch the document, and the `h` framing ignores that `h` is not uniform — a revoked contractor's documents are exactly the ones the contractor's recent queries were asking about, so the relevant `h` for that principal is far above the corpus average. A review that uses the average `h` will understate the exposure for the population at risk.

### 6.8 Step 8 — cost per query and per day

Cost enters the decision in one specific place: it decides whether physical partitioning at 1,000 workspaces is affordable, and it sets the price of the security design relative to the baseline.

```text
FORMULA [SR-08]: c_q = (T_in * p_in + T_out * p_out) + c_infra
WHERE: c_q is cost per query [currency/query]; T_in is input tokens including
       system prompt, question and context [tokens]; p_in is the input price
       [currency/token]; T_out is output tokens [tokens]; p_out is the output
       price [currency/token]; c_infra is the amortised retrieval and index
       cost [currency/query].
CONSTRAINT: uses list prices with a stated date; excludes retries, failed
       requests, evaluation traffic, human review and egress. Prices are
       constructed teaching values, not a quotation.
NUMERICAL EXAMPLE: system prompt 850 tokens, question 60, k = 8 chunks at
       420 tokens = 3,360; T_in = 850 + 60 + 3,360 = 4,270 tokens.
       T_out = 300 tokens. p_in = 0.60 per million tokens,
       p_out = 2.40 per million tokens.
       c_q = 4,270 * 0.60e-6 + 300 * 2.40e-6
           = 0.002562 + 0.000720 = 0.003282 currency/query.
       At 120,000 queries/day: 393.84 currency/day, about 11,815 in 30 days.
INTERPRETATION: the context is 79 percent of input tokens, so k is the cost
       lever, not the system prompt. Moving k from 8 to 12 raises input tokens
       to 5,950 and per-query cost to 0.00429, a 30.7 percent increase, while
       raising the residual-leak term at fixed eps by 44 percent (from
       1-(1-eps)^8 to 1-(1-eps)^12).
```

The direction of that trade is what matters for the review. Increasing `k` improves answer quality and simultaneously worsens both cost and residual leak, so a proposal to "improve answers by adding context" is a proposal to weaken the security gate. That is not an argument against larger `k`; it is an argument that `k` must be chosen once, with both effects on the table, rather than drifted upward by quality experiments.

### 6.9 Sensitivity and boundary analysis

The conclusion must survive plausible changes to the constructed inputs. The table varies one input at a time and records whether the *decision* changes, distinguishing a change in magnitude from a change in recommendation.

| Stress test | Original | Alternative | Effect on the quantities | Decision consequence |
|---|---|---|---|---|
| Traffic mix shifts to narrow roles (20/60/20 to 50/40/10) | H_bar = 34.46 | H_bar = 40*(0.5*0.9875+0.4*0.94+0.1*0.5) = 40*0.91975 = 36.79 | Exposure rises 6.6 percent; narrow-band latency requirement becomes the binding constraint for half of traffic | Strengthens the pre-filter decision; partitioning must ship with the first release, not later |
| Narrow-band selectivity improves to f = 0.05 | m* = 1,049 | m* = 260 | Post-retrieval fetch becomes 260; rerank 130 ms | Still exceeds the 210 ms pre-generation budget with ANN and policy, so the decision holds but the margin widens |
| Rerank throughput doubles to 4,000 pairs/s | 524.5 ms at m = 1,049 | 262.25 ms | Post-retrieval filtering becomes latency-viable at the narrow band | **Decision flips on latency grounds only.** Exposure (SR-01) is unchanged, so the security argument still rejects the baseline; the review would need to reject it on exposure alone, which it can |
| Label error is 10x worse than assumed (eps_label = 5e-3) | UCR = 5.2e-3 | UCR = 4.05e-2 | The pre-filter design also fails the 0.1 percent gate | Gate becomes unreachable until the labelling pipeline is fixed; recommendation becomes "fix labels first, then index" |
| k reduced from 8 to 4 | UCR multiplier 1-(1-eps)^8 | 1-(1-eps)^4 | Required eps for 0.1 percent relaxes from 1.25e-4 to 2.50e-4 | Security gate becomes twice as easy; answer quality presumably falls. A legitimate trade if quality is measured |
| Cache key omits workspace | eps_cache = 0 | eps_cache = 0.001 or unbounded | UCR jumps to at least 8.0e-3 and potentially to 1.0 at the cache hit rate | **Decision changes to "ship nothing until the cache is partitioned."** No index design compensates |
| Corpus grows 10x at fixed query volume | 4.8M chunks | 48M chunks | Index storage 147.5 GB of vectors; nightly build 3.3 h at 4,000 chunks/s | Partitioning cost grows with tenant count, not corpus size; decision unchanged, build window becomes the constraint |

Two rows deserve emphasis because they show what kind of conclusion this is.

The rerank-throughput row is the honest one. If reranking became four times faster, post-retrieval filtering would fit the latency budget, and a review that had rejected it *only* on latency would have to reverse. The reason the recommendation stands is that the rejection has an independent leg: 34.46 ineligible chunks per request is a property of the fetch geometry, not of the clock. A good design review keeps at least two independent reasons for a rejection, because any single reason is vulnerable to a hardware improvement.

The cache row is the one that changes the *order of work*. A cache key that omits the workspace produces cross-workspace answer reuse, and no retrieval-side control can undo it, because the context was assembled for someone else. This is why Section 8 sequences the cache change before the index change: shipping a secure index in front of an insecure cache produces a system that leaks while looking correct.

### 6.10 What we tried first, and what it establishes

No alternative was implemented and measured. The baseline is the shipped design as described in the review packet, and the three alternatives are analysed on paper. The case therefore records a **reasoning result**, not an experimental one, and it says so in every section where a number could be mistaken for a measurement.

Two things were explicitly not tested and should not be read as rejected on evidence. First, filtered ANN recall at low selectivity was not measured; the A-04 probe is specified in Section 9 and its outcome would change the index choice, not the boundary choice. Second, the label-error rate `eps_label` was not measured on the real corpus; the 5e-4 value is constructed, and the sensitivity row above shows that a tenfold error in it inverts the implementation order.

The branch where a different reasonable choice would change the answer is the choice of the **generation context size**. Everything about the residual-leak budget scales with `k`. A team that chose `k = 4` for cost reasons would face a gate twice as easy, and a team that chose `k = 16` would need `eps <= 6.3e-5`, which is below what a nightly batch labelling pipeline typically achieves. `k` is therefore not a free quality knob; it is a security parameter that happens to also affect quality.
