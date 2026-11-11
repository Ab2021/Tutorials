---
case_id: "CASE-04-02-06"
case_title: "Authorized Policy Retrieval"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "illustrative teaching construction; no company corpus, vendor, or measured result"
as_of_date: "2026-11-10"
owner_role: "customer-care policy owner"
source_index: "INDEX.md"
review_trigger: "Policy corpus change, authority-matrix change, or escalation capacity change"
---

# Authorized Policy Retrieval

Every count, rate, dollar figure, and latency in this case is **ILLUSTRATIVE**: a constructed teaching input whose arithmetic is checked but whose values describe no real retailer, corpus, or deployment. Nothing here is a measured vendor benchmark or a reported company result. Where a quantity would in practice come from an audit, the text says so and labels the substitute as constructed. The case is `proposed` and `draft`; the design described was not executed anywhere.

## 1. The One-Line Framing

A customer-care assistant may answer a policy question only when it can cite the clause that governed the decision at the decision timestamp, in the customer's jurisdiction and channel, and only when the action the answer implies sits inside the answering agent's delegated authority.
The retrieval problem is therefore not "find the most similar passage" but "find the passage that was in force, for this population, and that authorises this specific commitment".
The binding constraint is authorization, not fluency: an answer that cannot be traced to an in-force clause, reversed by the same clause, and audited after the fact is not eligible to be sent, however good it reads.
The reusable pattern is evaluation-mismatch: similarity score, answer quality score, and citation correctness are three different quantities, and only the last one governs the decision.
Takeaway: build the gate around the authorization predicate, then measure whether the gate's cost trade-off beats simply escalating everything.

## 2. Context & Background

An omnichannel retailer answers policy questions from three different consumers, and they do not share the same tolerance for error.
A human contact-centre agent uses the assistant as a lookup tool while the customer is on the phone; a wrong clause costs a recontact within minutes.
A chat surface shows the answer directly to the customer; a wrong clause becomes a written commitment the retailer may have to honour.
A store colleague uses it on a handheld during a return at the counter; there is a queue in front of them, so a slow correct answer and a fast wrong answer are both failures, in different currencies.
One corpus feeds all three, so the answer contract has to be decided once, at the corpus level, and then narrowed per surface.

The corpus itself is the first source of difficulty, because retail policy is not one document.
It is a family of documents per market: returns, price matching, warranty, delivery guarantees, loyalty tier benefits, gift-card rules, marketplace-seller exceptions, and staff-delegated-authority matrices.
Each document carries a jurisdiction, an owning function, an approval record, and a validity interval.
Each document is amended on its own rhythm: returns policy twice a year, promotional price-match rules every campaign, warranty terms only when a supplier contract changes.

The unit that must be retrieved is the clause, not the document.
A document-level answer is useless operationally, because the question "can this customer return an opened cosmetic bought in a store 45 days ago" is decided by one exception clause nested inside a general rule, and the general rule alone gives the wrong answer.
Clause identity also has to be stable across amendments, otherwise a citation that was correct last month cannot be audited this month.
The practical rule is that every clause has an identifier that survives re-publication, and every amendment is recorded as a new version of that identifier rather than as a new identifier.

Clause identity has a second property that matters as much as stability: a clause must be independently citable without its container.
A clause row therefore carries its own heading path, its own validity interval, and its own authority class, rather than inheriting any of the three from the document at query time.
Inheritance looks economical during ingestion and becomes a defect at the first document split, because a document split renumbers the container and silently reparents every clause beneath it.
The audit consequence is that a citation recorded last quarter resolves to a different container this quarter, and the dispute reviewer cannot reconstruct what the customer was told.

Clause text also has to survive normalisation, which is a smaller version of the same problem.
Quotation verification compares a span against the clause text, and whitespace, typographic quotes, hyphenation, and non-breaking spaces all differ between the published document and the extracted text.
A verifier that ignores normalisation rejects correct quotations; one that normalises aggressively accepts quotation of a different clause that happens to normalise to the same string.
The resolution is to normalise both sides identically with a recorded normalisation version, and to store the raw span alongside the normalised one so a reviewer can see what the model actually quoted.

Authorization is a property of a tuple, not of a text span.
Four predicates decide whether a retrieved clause may be used for a specific question.
The clause must be in force at the decision timestamp: `valid_from <= t < valid_to`, with the open interval on the right so that an amendment effective at midnight does not produce two controlling clauses for the same second.
The clause must apply to the customer's jurisdiction and sales channel, because a marketplace order and a first-party store order are governed by different exception sets even inside one country.
The clause must authorise the specific commitment being made, which is a narrower test than "is relevant to the topic".
The commitment must sit inside the authority of the actor who will be bound by it: a chat assistant acting for a tier-one agent cannot approve a goodwill credit that the matrix reserves for a supervisor.

Supersession is where most retrievers fail, and it fails silently.
A policy amendment publishes a new document version; the old version stays in the content system because archived versions are required for dispute handling; the vector index is rebuilt on a schedule that does not know about effective dates.
The result is an index that contains both the old and the new wording, with embeddings that are nearly identical, and a ranker that has no reason to prefer one.
The failure is invisible in offline answer-quality scoring, because the old clause answers the old question perfectly.
It becomes visible only when a customer is told the pre-amendment rule and the case is later reviewed against the post-amendment one.

The baseline is a deterministic retrieval-by-keyword tool plus a curated FAQ, maintained by the policy team and used by agents who then apply judgement.
That baseline is not a straw man: it is cheap, auditable, and its failure mode is slow lookup rather than confident error.
The case only proposes to replace it for the subset of questions where a clause-level citation is both available and within authority, and it keeps the baseline as the fallback route for everything else.

The secondary pattern is human-in-the-loop, but with an unusual shape.
The human is not reviewing the model's answer; the human is receiving the escalated case instead of the answer.
That makes review capacity a hard constraint on the design rather than a soft quality control, because every abstention converts into queue minutes.
A design that improves citation precision by abstaining more is not automatically better; it is a trade at a rate set by the cost of a queue minute against the cost of a wrong commitment.

Distribution shift also has a specific retail shape here.
The population that asks policy questions changes with the promotional calendar: the week a seasonal sale ends, returns questions triple and the mix moves toward the exception clauses that the sale introduced.
A retrieval threshold calibrated on a quiet month is therefore calibrated on the wrong mixture, and the shift is predictable rather than random, which means it can be handled by scheduled recalibration instead of drift detection.

Takeaway: the corpus, the clause identity scheme, and the authority matrix are the design objects; the language model is the least interesting component.

## 3. The Business Question

**Question:** for each incoming policy question at time `t`, should the assistant answer with a cited clause, abstain and route to a human, or escalate immediately as out-of-scope?
**Owner:** the customer-care policy owner, who owns the corpus and the authority matrix, not the model.
**Trigger:** every inbound policy question, plus a scheduled recalibration of the answer gate each promotional period.
**Actions available:** answer with citation; answer with citation and a required agent skim; abstain to a human queue; escalate as out-of-policy; return the baseline keyword tool result.
**Objective:** maximise correctly authorised commitments per agent-hour, subject to a hard ceiling on unauthorised commitments and a hard ceiling on escalation queue time.
**Hard constraints:** no clause may be used outside its validity interval; no commitment may exceed the acting tier's authority; escalation load may not exceed the queue capacity at the agreed service level; every delivered answer must carry a resolvable clause identifier and corpus version.
**Scope:** one market family, one quarter of questions, clause-level retrieval over an illustrative corpus of 1,240 documents and 38,400 clauses.
**Out of scope:** legal interpretation, vendor selection, multilingual expansion, and any claim of realised savings.

| Decision element | Definition used in this case | Why it is stated this way |
|---|---|---|
| Decision unit | one inbound policy question, keyed by `question_id`, at timestamp `t` | mixing questions and contacts hides multi-question calls |
| Decision timestamp | the moment the answer would be sent, not the moment the customer wrote | a clause amended mid-conversation must not be applied backwards |
| Action set | answer / answer-with-skim / abstain / escalate-out-of-policy / baseline-tool | a binary answer-or-not frame loses the cheap middle route |
| Objective | authorised commitments resolved per agent-hour | agent-hours are the scarce resource, not API calls |
| Hard constraint 1 | zero use of superseded clauses | a single superseded citation can invalidate a whole campaign's commitments |
| Hard constraint 2 | commitment within acting-tier authority | policy compliance does not imply delegated authority |
| Hard constraint 3 | escalation queue at agreed service level | the abstain route is only real if a human picks it up in time |
| Comparator | deterministic keyword tool plus curated FAQ, human-applied | the incumbent process, not a null model |

The question is deliberately not "how accurate is the retriever".
Accuracy of retrieval is an input to the decision, and a high-accuracy retriever that ignores effective dates still produces unauthorised commitments.
The question is also not "should we deploy an assistant", because the answer depends entirely on the gate's cost ratio and on the escalation capacity that exists on the day.

Takeaway: the decision has five actions, not two, and the cheap middle route (answer with a required skim) is where most of the value sits.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Verification | Fallback if false |
|---|---|---|---|
| A-01 | Every clause carries a machine-readable validity interval | sample 200 clauses, compare against the published document | exclude undated clauses from answering, keep them for browsing |
| A-02 | Clause identifiers survive amendment | diff two consecutive corpus versions, count identifier churn | freeze answering for the affected document family |
| A-03 | Jurisdiction and channel are recorded per document, not per clause | inspect 50 documents spanning three markets | infer from document path, flag as low confidence |
| A-04 | The authority matrix is current within one quarter | compare matrix version to HR delegation record | cap all answers at the lowest tier until reconciled |
| A-05 | Adjudicators agree on what "controlling clause" means | double-adjudicate a 300-question sample, measure kappa | collapse to document-level citation and re-scope |
| A-06 | Escalation arrivals can be forecast per hour | compare forecast to actual over four weeks, measure error | staff to the peak hour of the observed week |
| A-07 | Agent skim time on an answered question is materially below full handling | time study on 120 answered questions | withdraw the answer-with-skim action entirely |
| C-01 | No clause may be used outside its validity interval | continuous check on every delivered citation | block delivery, alert corpus owner |
| C-02 | No commitment may exceed the acting tier | pre-send authority check against the matrix | downgrade to escalate |
| C-03 | Escalation queue must hold a service level at peak | hourly simulation against staffing plan | raise the answer threshold to cut escalations |
| C-04 | Answers must be reproducible from the logged corpus version | replay 50 answers against the archived index | suspend answering until replay is clean |
| C-05 | Customer-supplied text is untrusted input | injection screen on every inbound message | strip quoted instructions, log the event |

The authority matrix is the constraint that most often surprises modellers, because it has nothing to do with the corpus.
It is a table of what an actor may commit, expressed in the same currency as the customer's request: money for goodwill, days for an extension, units for a replacement.

| Action class | Tier 1 (front line) | Tier 2 (senior agent) | Tier 3 (team lead) | Supervisor |
|---|---|---|---|---|
| Goodwill credit | up to $45 | up to $180 | up to $600 | above $600 |
| Return-window extension | up to 14 days | up to 30 days | up to 60 days | any |
| Price-match difference | up to $25 | up to $90 | up to $250 | above $250 |
| Replacement without return | not permitted | single unit, once per customer | up to three units | any |
| Delivery-fee waiver | permitted | permitted | permitted | permitted |

An answer that reads correctly but implies an action above the acting tier is an authority breach, and it is the most expensive failure class in this case because it creates an enforceable commitment.
The matrix therefore runs as a separate post-check on the drafted answer: extract the implied commitment, compare to the tier, and downgrade to escalate when the comparison fails.

Two assumptions deserve emphasis because the whole cost model rests on them.
A-05 says adjudicators can agree on the controlling clause; if they cannot, the ground truth is not a ground truth and no threshold is meaningful, so this is tested before any modelling.
A-07 says a skimmed answer is cheaper than full handling; if agents re-derive the answer from scratch to trust it, the answer-with-skim action collapses into full handling and the design loses its only cheap route.

The constraints also bound what the analysis may claim.
Because C-01 forbids any superseded citation, a design cannot buy recall by relaxing the effective-date filter, and any proposal that improves answer rate that way is out of bounds rather than merely risky.
Because C-03 ties abstention to staffing, a threshold cannot be chosen from precision alone; it has to be chosen jointly with the escalation forecast.

Takeaway: authority is a table lookup that runs after retrieval, and it is the constraint most likely to be omitted from a retrieval-only design.

## 5. Data, Tools & Method

### 5a Data Specification

Grain: one row per inbound policy question, keyed by `question_id`, with one row per candidate clause retrieved for that question.
Population: all policy questions arriving through chat, voice-transcript, and store-handheld surfaces in one market family during one quarter, excluding greetings and order-status lookups that carry no policy content.
Cut-off: the decision timestamp `t` recorded when the answer would have been sent; every feature, every clause candidate, and every authority lookup is evaluated as of `t`.
Clause table keys: `clause_id` (stable), `clause_version` (monotone per amendment), `document_id`, `jurisdiction`, `channel_scope`, `valid_from`, `valid_to`, `authority_action_class`, `owner_function`.
Question table keys: `question_id`, `t`, `surface`, `customer_id`, `order_id`, `acting_tier`, `raw_text`, `language`.
Label table keys: `question_id`, `controlling_clause_id`, `controlling_clause_version`, `adjudicator_id`, `adjudication_time`, `verdict`.
Units: money in USD, durations in days, time in UTC seconds, scores dimensionless on `[0,1]`, rates per question unless stated.

What is **not** available at decision time, and must therefore never appear as a feature:
the adjudicated controlling clause, which is produced after the fact by a human reading the corpus;
the customer's later recontact or chargeback, which is the outcome the design is trying to avoid;
the amendment that will take effect next month, even though the policy team already knows about it;
the corrected order record after a fulfilment fix, because the correction is written after the question is answered.
A feature built from any of these will improve offline scores and will not exist in production; that is the leakage class this case is most exposed to.

### 5b Tools & Stack

Components, described by role rather than by product, because no vendor is asserted:
a versioned corpus export that preserves document identity and validity intervals;
a clause splitter that respects clause boundaries and refuses to emit a fragment without its parent heading;
a hybrid retriever combining lexical matching for clause numbers and product names with dense matching for paraphrase;
an authorization filter applied as a predicate over retrieval results, not as a reranking feature;
a citation verifier that checks the quoted span exists verbatim in the cited clause version;
an authority checker that maps the drafted commitment to a matrix row and compares tiers;
an answer gate that compares a calibrated confidence against a cost-derived threshold;
an append-only decision log storing the corpus version, candidate set, filter decisions, gate score, and final action.

The stack choice that matters is the ordering of the last four components.
Verification must run before the gate, because the gate's confidence is only meaningful for answers whose citation has already been checked.
The authority check must run after drafting, because the commitment is a property of the answer text, not of the retrieved clause.
The gate must run before the queue, because the queue is the scarce resource and the gate is what protects it.

### 5c Method Summary

Step one: constrain the candidate set by the authorization predicate — jurisdiction, channel, validity at `t` — before any similarity ranking is scored, so that a superseded clause cannot win on similarity.
Step two: rank the surviving candidates and take the top five, because recall at five is the binding retrieval quantity and deeper lists mostly add near-duplicates.
Step three: verify that the span the model intends to quote appears in the cited clause version, and reject answers that quote a paraphrase, a neighbouring clause, or an invented identifier.
Step four: draft the answer, extract the implied commitment, and compare it to the acting tier.
Step five: compare the calibrated confidence against the cost-derived threshold and route to answer, answer-with-skim, or abstain.
Step six: log every decision with enough state to replay it, then measure the four decision metrics against adjudicated labels.

The method deliberately does not fine-tune a generator on policy text.
Fine-tuning would make the model fluent in the corpus's phrasing, which is exactly the property that makes a wrong citation sound authoritative; the verifier is a cheaper and more auditable control.
The method also does not use the model to decide applicability, because applicability is a predicate over structured fields and a deterministic filter is both cheaper and testable.

### 5d Provenance, Quality, Exclusions, Leakage

Provenance: every clause row carries the document version it came from, the approval record identifier, the publication timestamp, and the extraction timestamp.
Four timestamps must be kept distinct — authored, approved, published, effective — because the gap between published and effective is where a stale index hides: a clause can be published on day 1, effective on day 15, and indexed on day 4, so an index built on day 20 is correct while an index built on day 10 applying the new text is wrong.
Quality checks on the corpus, each producing a count rather than a pass mark: clauses with missing validity interval; clauses whose text is shorter than 25 words; clause pairs above 0.93 cosine similarity; documents whose jurisdiction field disagrees with their path; amendments whose identifier changed instead of versioning.
Exclusions, with the excluded share measured rather than assumed: questions with no order reference and no product reference; questions in languages outside the supported set; questions whose text is under 12 tokens; questions arriving during the two-day window when a corpus export failed.
Leakage controls: labels are adjudicated without seeing model output; the adjudication queue is sampled across surfaces rather than from the answered set; the evaluation uses the corpus version that was live at `t`, not the current one; near-duplicate clause pairs are grouped so that a clause appearing in both the development and the evaluation period is counted once.
The most dangerous leak is the last one: because policy text barely changes, random splits put near-identical clauses on both sides, and a retriever that has memorised the wording scores well without having learned to distinguish versions.

### 5e Reproducibility & Method-Selection Ledger

| Choice | Alternative considered | Why this one | Cost of being wrong |
|---|---|---|---|
| Clause-level chunks with stable identifiers | fixed 180-token windows | citation must name one clause; windows span several | unverifiable citations, roughly two thirds of returned spans truncated |
| Predicate filter before ranking | rerank with an "is current" feature | a superseded clause must be impossible, not unlikely | superseded citations survive at a low but nonzero rate |
| Verbatim span verification | model self-assessment of groundedness | self-assessment is the failure being measured | fabricated clause identifiers pass |
| Cost-derived gate threshold | fixed 0.8 confidence | the threshold is a property of the cost ratio, not of the model | systematically wrong answer volume |
| Answer-with-skim as a third action | binary answer or escalate | a skimmed answer is materially cheaper than full handling | losing the only cheap route, and overloading the queue |
| Adjudicated controlling clause as label | answer-level thumbs up or down | the decision is about the citation, not the tone | measuring style instead of authorization |

Reproducibility requirements: the corpus version, the splitter version, the retriever configuration, the verifier ruleset, the authority matrix version, the gate threshold, and the queue staffing plan are all recorded with each evaluation run.
A replay of any single answer must reconstruct the candidate set from the archived index and produce the same action; if it cannot, the decision is not auditable and should not have been taken.
The case is illustrative, so no such replay was executed; the requirement is stated because it defines what would have to be true for the numbers to mean anything.

Takeaway: reproducibility here means reproducing the decision from the decision-time corpus, not reproducing a model artefact.
