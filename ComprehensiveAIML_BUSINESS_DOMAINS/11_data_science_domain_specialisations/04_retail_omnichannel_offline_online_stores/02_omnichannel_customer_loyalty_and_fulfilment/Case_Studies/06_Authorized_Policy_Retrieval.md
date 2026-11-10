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

## 6. Analysis Walkthrough

The walkthrough answers one question in four steps: what is the cheapest defensible answer threshold, does that threshold actually beat the incumbent process, which stage of the pipeline limits coverage, and what does the effective-date rule buy when it is priced rather than asserted. Every count below is constructed; every arithmetic step is shown so the reader can substitute their own inputs.

### 6.1 Deriving the answer threshold from costs, not from taste

**FORMULA**

```text
tau* = (C_e - C_a) / C_w
```

**WHERE**

| Symbol | Meaning | Units | Valid range |
|---|---|---|---|
| `tau*` | break-even probability that a delivered answer is unauthorised | dimensionless | 0 to 1 |
| `C_e` | incremental cost of escalating one question to a human | USD per question | > 0 |
| `C_a` | cost of delivering one answered question, verification included | USD per question | > 0 |
| `C_w` | expected cost of one unauthorised commitment reaching the customer | USD per event | > 0 |

Cost components, each built from a time study or an audit rather than asserted:

```text
C_a = skim_min * loaded_rate + compute_per_question
    = 1.5 * 0.60 + 0.20 = 1.10 USD

C_e = handle_min * loaded_rate + downstream_per_escalation
    = 13.0 * 0.60 + 1.40 = 9.20 USD

C_w = honoured_commitment + remediation + review
    = 96 + 28 + 16 = 140 USD
```

**CONSTRAINT**

The derivation holds only when four conditions are met: costs are linear in question count, so a thousand escalations cost a thousand times one; `C_e > C_a`, otherwise no question should ever be escalated; `C_w` is estimated from breaches that were actually detected, which makes it a lower bound; and the model's confidence is calibrated, so that a stated probability of 0.04 means four in a hundred across the population rather than four in a hundred within a convenient slice.

**NUMERICAL EXAMPLE**

```text
tau* = (9.20 - 1.10) / 140
     = 8.10 / 140
     = 0.05786
```

Check the break-even directly: an answer with unauthorised probability 0.05786 costs `0.05786 * 140 + 1.10 = 8.10 + 1.10 = 9.20`, which equals the escalation cost exactly. Below that probability answering is cheaper; above it, escalating is cheaper.

**INTERPRETATION**

The threshold is a property of the operating economics, not of the model. If the loaded agent rate rises from $0.60 to $0.72 per minute, `C_e` becomes `13.0 * 0.72 + 1.40 = 10.76` and `C_a` becomes `1.5 * 0.72 + 0.20 = 1.28`, so `tau* = 9.48 / 140 = 0.0677`. The threshold moves by a full percentage point because labour got 20 percent dearer, with no change to the model or the corpus. A design that fixes the threshold at 0.05 or 0.10 and never revisits it is therefore tracking the wrong object.

### 6.2 Does the threshold beat the incumbent process?

**FORMULA**

```text
T(theta) = N_a(theta) * C_a + N_e(theta) * C_e + N_w(theta) * C_w
```

**WHERE**

| Term | Meaning | Units |
|---|---|---|
| `T(theta)` | total cost of the answer policy at threshold `theta` | USD per quarter |
| `N_a(theta)` | questions answered under the policy | questions |
| `N_e(theta)` | questions escalated under the policy | questions |
| `N_w(theta)` | answered questions whose answer is unauthorised | questions |
| `N` | total questions, with `N_a + N_e = N` | questions |

The marginal form is the one a practitioner actually applies to a score band `b` with observed wrong-answer rate `p_b`:

```text
accept band b  <=>  p_b * C_w + C_a < C_e
               <=>  p_b < 0.05786
```

**CONSTRAINT**

`N_w` is the sum of per-band wrong rates weighted by band volume, so the total cost is a piecewise-linear, generally non-convex function of the threshold. It is non-convex whenever bands are not sorted by wrong rate, which happens when the confidence score is miscalibrated across surfaces: the store-handheld slice can have a high score and a high wrong rate at the same time. Non-convexity means the marginal rule can accept a band and reject a cheaper one, so bands must be sorted by observed wrong rate, not by score.

**NUMERICAL EXAMPLE**

Band structure over the questions that survive retrieval, verification, and authority checks — 36,734 of 48,000:

| Band | Questions | Observed unauthorised rate | Accept at tau*? | Contribution to `N_w` |
|---|---|---|---|---|
| 0.90 to 1.00 | 8,400 | 0.006 | yes | 50 |
| 0.80 to 0.90 | 10,600 | 0.039 | yes | 413 |
| 0.70 to 0.80 | 9,300 | 0.124 | no | 1,153 |
| below 0.70 | 8,434 | 0.270 | no | 2,277 |

Three policies compared against the baseline of full human handling for all 48,000 questions:

```text
Baseline:     48,000 * 9.20                                    = 441,600
tau* = 0.0579: 19,000 * 1.10 + 29,000 * 9.20 + 463 * 140
               = 20,900 + 266,800 + 64,820                    = 352,520
tau  = 0.0100:  8,400 * 1.10 + 39,600 * 9.20 +  50 * 140
               =  9,240 + 364,320 +  7,000                    = 380,560
tau  = 0.1300: 28,300 * 1.10 + 19,700 * 9.20 + 1,616 * 140
               = 31,130 + 181,240 + 226,240                   = 438,610
```

Marginal check on the middle band, which is the band the threshold decision actually turns on:

```text
accept band 2: 10,600 * 1.10 + 413 * 140 - 10,600 * 9.20
             = 11,660 + 57,820 - 97,520 = -28,040   (saves money)
accept band 3:  9,300 * 1.10 + 1,153 * 140 -  9,300 * 9.20
             = 10,230 + 161,420 - 85,560 = +86,090  (loses money)
```

**INTERPRETATION**

The saving over baseline at `tau*` is `441,600 - 352,520 = 89,080`, or 20.2 percent of the baseline cost. The saving is not proportional to the answer rate: answering 39.6 percent of questions produces a 20.2 percent saving, because the answered questions still carry a 2.4 percent wrong rate and because the escalated 60.4 percent is where most of the money stays. A pitch that quotes the answer rate as if it were the saving is overstating the case by roughly a factor of two, in either direction depending on which rate and which cost component is chosen.

The asymmetry between the two marginal tests is the transferable result. Accepting the 10,600-question band saves $28,040 per quarter; accepting the 9,300-question band below it loses $86,090. Two adjacent bands that look alike in score differ by more than $114,000 in consequence, which is why the threshold has to be located on measured band economics rather than interpolated.

### 6.3 Which stage actually limits coverage?

**FORMULA**

```text
A/N = (1 - z) * r5 * v * a * g
```

**WHERE**

| Factor | Meaning | Units | Illustrative value |
|---|---|---|---|
| `z` | share of questions whose candidate set is empty after the authorization predicate | fraction | 0.065 |
| `r5` | share of remaining questions whose controlling clause is in the top five | fraction | 0.912 |
| `v` | share whose intended quotation appears verbatim in the cited clause version | fraction | 0.964 |
| `a` | share whose implied commitment sits inside the acting tier | fraction | 0.931 |
| `g` | share of survivors whose band is accepted at `tau*` | fraction | 19,000 / 36,734 = 0.5172 |

**CONSTRAINT**

The product is valid only if the stages are conditionally independent given the question. They are not: a question whose controlling clause is a long nested exception has both low retrieval recall and low verbatim-quote pass rate, because the exception text is unusual and the model paraphrases it. Positive dependence makes the product an upper bound on true coverage, and the bound loosens as dependence grows. The rule of thumb is that the product is usable for prioritisation and not usable as a forecast when any pairwise correlation between stage residuals exceeds roughly 0.3.

**NUMERICAL EXAMPLE**

```text
A/N = 0.935 * 0.912 * 0.964 * 0.931 * 0.5173
    = 0.935 * 0.912        = 0.85272
    0.85272 * 0.964        = 0.82202
    0.82202 * 0.931        = 0.76530
    0.76530 * 0.5173       = 0.39583

questions answered = 0.39583 * 48,000 = 19,000
```

Stage counts, checked by successive multiplication:

```text
48,000 * (1 - 0.065) = 44,880   have a candidate set
44,880 * 0.912       = 40,931   controlling clause retrieved
40,931 * 0.964       = 39,457   quotation verified
39,457 * 0.931       = 36,734   commitment inside authority
36,734 * 0.5173      = 19,000   band accepted
```

Elasticity, derived by differentiating the logarithm of the product: because `ln(A/N) = ln(1-z) + ln(r5) + ln(v) + ln(a) + ln(g)`, a relative gain of 1 percent in any stage yields a relative gain of 1 percent in coverage. Converting to the absolute movement needed for each additional percentage point of coverage:

```text
needed relative gain = 0.01 / 0.39583 = 0.02526
recall:    0.02526 * 0.912  = 0.0230   -> +2.30 points
verifier:  0.02526 * 0.964  = 0.0244   -> +2.44 points
authority: 0.02526 * 0.931  = 0.0235   -> +2.35 points
gate:      0.02526 * 0.5173 = 0.0131   -> +1.31 points
```

**INTERPRETATION**

The gate is the cheapest place to buy coverage, because it is the only stage sitting near one half while the others sit above nine tenths. That conclusion is uncomfortable, because raising the gate acceptance means accepting a higher wrong-answer rate, which is exactly what the cost derivation forbids at `tau*`. The resolution is that gate coverage can only be improved by better calibration, not by a looser threshold: if the confidence score separated bands more sharply, the acceptable mass would grow at constant `tau*`. Retrieval improvements, by contrast, raise coverage without touching the wrong-answer rate, but they cost 1.8 times as much per point because recall starts high.

### 6.4 What the effective-date predicate is worth, priced rather than asserted

**FORMULA**

```text
E = N * s * (w / L) * C_w
```

**WHERE**

| Symbol | Meaning | Units | Illustrative value |
|---|---|---|---|
| `E` | expected quarterly cost of answering from a clause that is published but not yet effective | USD per quarter | computed |
| `N` | questions per quarter | questions | 48,000 |
| `s` | share of questions whose controlling clause family is amended during the quarter | fraction | 0.042 |
| `w` | gap between publication and effective date | days | 9 |
| `L` | amendment cycle length for that family | days | 182.5 |
| `C_w` | cost of one unauthorised commitment | USD per event | 140 |

**CONSTRAINT**

Valid when amendments are published before they take effect and the interval `w/L` is small relative to 1, so that at most one pre-effective window is open per clause at a time. It also assumes the index refresh does not matter, which is true only if the predicate is evaluated against the clause's own validity fields at query time. If instead the predicate is baked into a nightly index rebuild, the relevant interval is the rebuild lag `delta`, not `w`, and the formula must be re-derived with `delta` in place of `w`.

**NUMERICAL EXAMPLE**

```text
E = 48,000 * 0.042 * (9 / 182.5) * 140
  = 2,016 * 0.049315 * 140
  = 99.4 * 140
  = 13,919 USD per quarter
```

The same formula with a seven-day index rebuild lag and no query-time predicate, taking `w = 0` because amendments are effective on publication:

```text
E' = 48,000 * 0.042 * (7 / 182.5) * 140
   = 2,016 * 0.038356 * 140
   = 77.33 * 140
   = 10,826 USD per quarter
```

**INTERPRETATION**

Two things follow, and the second is the one that changes architecture. First, the dollar value of the predicate is small — about $13,900 per quarter, which is under a sixth of the gate's saving — so pricing it as an economic optimisation would lose the argument against a cheaper alternative. It is justified as a compliance invariant under constraint C-01, not as a return on investment, and mixing the two framings lets a cost-benefit review retire a control that exists to make a class of error impossible.

Second, the comparison of `E` and `E'` is misleading on its own, because the two quantities are different kinds of failure. A stale index that still contains the prior clause version, with that version's own interval, will have the prior clause excluded by a query-time predicate: correctness is preserved and the loss is recall, which shows up as a smaller candidate set. A stale index whose text has been updated while the interval has not is the dangerous case, because the predicate then admits the wrong wording while the interval test still passes. The invariant is therefore not "filter by date" but "version the text and the interval together, and never refresh one without the other".

### 6.5 Where the formulas break

Each derivation above has a stated constraint; three concrete counterexamples show how the break is observed rather than argued.

**The heavy-tail break in 6.1.** The break-even threshold assumes `C_w` is an average. If one answer in twenty thousand is screenshot and forwarded to a regulator, the cost distribution is heavy-tailed, the mean is unstable, and the expected-cost comparison understates the loss. The diagnostic is the ratio of the 99th-percentile commitment cost to the mean: under 5, the linear model holds; above 20, no threshold computed from averages is defensible and the gate must be replaced by a hard rule that prohibits the class of clause involved.

**The dependence break in 6.3.** Recompute recall separately for questions whose controlling clause contains a nested exception list. In this constructed case their share is 0.22 of questions and their recall is 0.771 against 0.951 for the rest, giving a weighted recall of `0.22 * 0.771 + 0.78 * 0.951 = 0.1696 + 0.7418 = 0.9114`, which matches the aggregate 0.912. The aggregate is right and the product is still an upper bound, because verifier pass on the same subset is 0.884 rather than 0.964. Multiplying the subset's own four factors gives `0.935 * 0.771 * 0.884 * 0.931 * 0.5172 = 0.307`, against 0.396 for the population, so exception questions get answered at about 78 percent of the population rate. The aggregate funnel never reveals this; a sliced funnel does.

**The interval break in 6.4.** If `valid_to` is stored as an empty string for open-ended clauses, a comparison written as `valid_to > t` silently drops every open-ended clause, which in this corpus is 61 percent of the clause count. The predicate still runs, still returns rows, and still looks healthy: the only symptom is that `z` jumps from 0.065 to 0.41. A corpus with an unexpectedly high zero-candidate share is therefore an interval-encoding bug until proven otherwise, and the check is to count clauses by whether `valid_to` parses as a date.

Takeaway: every number here is a decision input with a stated validity condition, and the conditions fail in ways that produce plausible-looking output.

## 7. The Finding

The finding is stated as four claims, each with the quantity that supports it and the condition under which it would stop being true.

**Claim one: coverage is limited by the answer gate, but the gate cannot be loosened.** At the cost-derived threshold of 0.0579 the gate accepts 51.7 percent of the questions that reach it, while every upstream stage sits above 93 percent. Coverage of 39.6 percent of inbound questions is therefore arithmetically dominated by the cheapest-looking lever, and that lever is the one the cost derivation forbids pulling. The condition that would reverse this is a change in the cost ratio: if `C_w` fell from $140 to $40, `tau*` becomes `8.10 / 40 = 0.2025`, the third band at 0.124 becomes acceptable, and coverage rises to `(19,000 + 9,300) / 36,734 = 0.7704` without any model change at all.

**Claim two: the middle action carries a third of the value for a seventh of the risk.** Of the 19,000 answered questions, 10,600 carry a required agent skim because their band sits between `tau*` and 0.12, where the observed wrong-answer rate is 0.039. Those answers are worth `10,600 * 8.10 - 413 * 140 = 85,860 - 57,820 = 28,040`, while the 8,400 fully trusted answers are worth `8,400 * 8.10 - 50 * 140 = 68,040 - 7,000 = 61,040`; the two sum to the 89,080 net saving. The skimmed band thus contributes 31.5 percent of the value while generating 89.3 percent of the wrong-answer events, which is precisely the trade the third action exists to price rather than to hide. Remove it — accepting only zero-skim answers — and coverage falls from 39.6 to `8,400 / 48,000 = 17.5` percent and the saving falls by 31.5 percent.

**Claim three: the effective-date predicate is a compliance invariant masquerading as a small optimisation.** Its expected cost avoided is about $13,900 per quarter against a net system saving of $89,080, so at a 6:1 ratio it loses any head-to-head cost comparison with, say, a better reranker. It is nevertheless non-negotiable under C-01 because the failure it prevents is a class of error — a customer told a rule that never applied to them — that cannot be remedied by later accuracy. The practical consequence is that the predicate belongs in the query path as an invariant, not in a scoring model as a feature, because a feature can be outweighed and an invariant cannot.

**Claim four: the corpus, not the model, is the binding asset.** Three of the four upstream failure modes in the constructed funnel trace to document engineering rather than to model quality: empty candidate sets from missing validity intervals, unverifiable quotations from clause spans that cross headings, and authority breaches from a matrix that is not joined to the corpus. Raising retrieval recall from 0.912 to 0.94 through better chunking adds `0.028 * 0.935 * 0.964 * 0.931 * 0.5172 * 48,000 = 583` answered questions per quarter, which at the marginal band cost of `9.20 - 1.10 = 8.10` and a wrong rate near 0.02 is worth `583 * (8.10 - 2.80) = 583 * 5.30 = 3,090` per quarter. The number is small in absolute terms and large relative to its cost, because chunking is a one-off engineering task rather than a quarterly model refresh.

| Finding | Supporting quantity | Reversal condition |
|---|---|---|
| Gate binds coverage | 0.5172 accepted at gate vs 0.912/0.964/0.931 upstream | `C_w` falls below about $45, or confidence calibration doubles the acceptable mass |
| Middle action carries value | 11,100 of 19,000 answered questions are skimmed | skim time study shows agents re-derive answers, collapsing the action |
| Date predicate is compliance, not ROI | $13,919 avoided per quarter vs $89,080 net saving | never, while C-01 stands |
| Corpus engineering beats model tuning | +583 answers per quarter from chunking, ~$3,090 value at one-off cost | chunking already near ceiling, or clause text not the limiting factor |

The finding that did **not** survive scrutiny is the one the project started with: that answer quality was the problem. Answer quality, measured as adjudicator-rated helpfulness on answered questions, sat at 0.88 in the constructed sample, well above the 0.72 the team had feared, and raising it would not have moved coverage at all because the gate rejected those questions before quality was observable. This is the evaluation-mismatch pattern in its purest retail form: the team was optimising a metric computed only on the subset that had already passed the decision gate.

Takeaway: the lever with the most headroom is the one the constraint forbids moving, so the real work is calibrating the gate rather than widening it.

## 8. System/Solution Design

### 8.1 Request path

The path has seven stages, and their order is the design.

```text
inbound question q at time t
  -> [1] surface and language gate        (deterministic)
  -> [2] authorization predicate          (deterministic, over clause validity + jurisdiction + channel)
  -> [3] hybrid retrieval, top five       (lexical + dense)
  -> [4] verbatim span verifier           (deterministic string check against cited clause_version)
  -> [5] answer draft + commitment extract (model, structured output)
  -> [6] authority checker                (deterministic table lookup against matrix version)
  -> [7] calibrated gate                  (threshold tau*, recomputed monthly)
  -> action: answer | answer_with_skim | abstain | escalate_out_of_policy | baseline_tool
```

Stage 2 before stage 3 is the load-bearing choice. Running the predicate after ranking means computing similarity for clauses that will be discarded, which wastes effort, and more importantly means that the top of the ranking can be occupied by ineligible clauses, which changes the depth needed to find five eligible ones. In the constructed corpus, filtering first reduces the candidate pool from 38,400 clauses to a median of 4,180 per question and leaves recall at five essentially unchanged at 0.912, whereas filtering after ranking requires a depth of 14 to reach the same recall.

Stage 4 before stage 5 is the second load-bearing choice, and it is the one that a fluent-model design usually gets wrong. Verifying a quotation after the answer is drafted turns every unverifiable answer into a rejected answer, which is expensive because the draft already consumed the inference budget and because rejection rate then depends on the model's paraphrasing habits rather than on corpus quality. Verifying candidate spans before drafting lets the retriever prefer candidates with extractable spans, which converts the verifier from a filter into a selection signal.

### 8.2 Corpus engineering

The corpus is the primary engineering artefact, and it is built by a pipeline rather than by an ingestion script.

Clause splitting: split on document structure — numbered headings, lettered exceptions — never on token count. A clause carries its heading path as metadata, so a fragment retains the context that makes it interpretable. Splitting on structure is what makes stable identifiers possible, because a heading path plus a document identifier is stable across amendments while a token offset is not.

Versioning: each clause has `clause_id` (stable), `clause_version` (increments on any text change), `valid_from`, `valid_to`, and `supersedes_version`. An amendment writes a new version row rather than mutating the existing row, so a query at any historical timestamp can reconstruct the corpus as it stood. This is the only structure that supports both dispute handling and leakage-free evaluation, and it is also the structure that most content systems lack, which is why the retrofit is usually the largest line item.

Validity encoding: `valid_to` is either a date or explicitly `open`, never an empty string, and the predicate treats `open` as `+infinity` rather than as a missing value. This single encoding decision is worth the 61 percent of clauses it would otherwise drop.

Jurisdiction and channel: recorded per document, with a per-clause override for the exception clauses that narrow a document's scope. The override share is measured, because a high override rate means the document-level field is wrong rather than that the clauses are exceptional.

Authority binding: each clause declares the action class it authorises, drawn from the matrix vocabulary, so the authority check at stage 6 has a structured field to compare rather than having to infer intent from prose. Clauses whose action class is unknown are excluded from answering, which is an honest restriction rather than a silent default.

### 8.3 The answer gate

The gate takes two inputs: a calibrated confidence and a threshold recomputed monthly from the current cost table.
Calibration is fitted per surface, because the store-handheld slice has a systematically different score distribution: its questions are shorter, more often refer to a physical item in hand, and its wrong-answer rate at a fixed score is about 1.6 times the chat slice's rate. A single global calibration curve therefore over-answers on the handheld surface, and the symptom is not a drop in aggregate accuracy but an excess of unauthorised commitments on one surface.

The threshold recomputation is a small scheduled job with a human approval step, because it changes customer-visible behaviour and because a cost table entry can be wrong. The job emits the four cost inputs, the band table with observed wrong rates, the implied `tau*`, and the coverage and cost at the new threshold; a human approves or rejects. Automating the approval is tempting and wrong for the same reason that the authority limit exists: the threshold encodes a policy choice about acceptable error, and that choice has an owner.

### 8.4 Escalation route and capacity

Abstentions land in the existing human queue with a structured handoff: the question, the candidate clauses with their validity intervals, the verifier's reason for rejection, and the authority gap if there is one. The structured handoff is what makes the escalated case cheaper than the queued case in the baseline process, where the agent starts from the question alone.

Capacity is a constraint, not an output, and in the constructed case it binds in the direction that is easy to get backwards. The plan holds 21,400 escalations per quarter with peak-hour headroom, while `tau*` produces 29,000: the 11,266 questions that fail upstream stages are already escalations under any gate, so the gate only controls the remaining 36,734. Fitting the queue therefore requires *accepting more*, not fewer — 26,600 answers, or 72.4 percent of survivors, instead of 19,000. That means pulling 7,600 questions out of the 0.124 band into the answer route:

```text
queue-constrained: 26,600 * 1.10 + 21,400 * 9.20 + 1,405 * 140
                 = 29,260 + 196,880 + 196,700 = 422,840
saving vs baseline 441,600 - 422,840 = 18,760  (4.2 percent, not 20.2)
```

The capacity constraint thus destroys 78.9 percent of the system's value, and it does so by forcing acceptance of a band whose marginal economics are negative. The correct response is not to loosen the gate and absorb the risk; it is to treat the 7,600-question gap as the business case for additional handlers: `7,600 * 13.0 = 98,800` agent-minutes, or 1,647 agent-hours per quarter, which at 13 weeks of 40 hours and a 75 percent occupancy ceiling is `98,800 / (13 * 40 * 60 * 0.75) = 4.2` full-time equivalents. The design states this rather than silently choosing either horn, because both horns are owner decisions about staffing and risk appetite.

| Component | Failure it prevents | Its own failure mode |
|---|---|---|
| Structural clause splitter | unverifiable spans crossing headings | over-splitting a nested exception into fragments |
| Versioned clause rows | applying amended text backwards | version churn without identifier stability |
| Explicit `open` validity | silent drop of open-ended clauses | mislabelling a genuinely dated clause as open |
| Predicate before ranking | ineligible clauses consuming the top-k | empty candidate sets when jurisdiction is misrecorded |
| Verbatim verifier | fabricated or paraphrased citations | rejecting legitimately normalised whitespace |
| Authority checker | commitment above acting tier | matrix staleness after a delegation change |
| Per-surface calibration | over-answering on one channel | thin slices with unstable calibration |
| Structured handoff | agent re-deriving the case from scratch | handoff fields that the agent does not read |

### 8.5 What the design deliberately omits

No fine-tuned generator. Fluency in policy prose makes a wrong citation sound authoritative, and the verifier already provides the control that fine-tuning would be intended to provide.

No agentic multi-hop loop over the corpus. Multi-hop retrieval helps when the answer requires combining two clauses, which in the constructed population is 4.1 percent of questions; the loop's cost is paid on all questions and its benefit accrues to the 4.1 percent, so it belongs behind a classifier rather than on the main path.

No learned authority inference. Authority is a table, and inferring it from prose is how a system ends up promising a credit it cannot grant.

No real-time learning from agent edits. Corrections arrive attributed to agents whose interpretations differ, and a model that learns from them reproduces the loudest agent's reading of the corpus rather than the corpus.

Takeaway: the design is mostly deterministic plumbing around one model call, and that ratio is the point.

## 9. Evaluation & Validation

The evaluation is organised as four questions, each with an explicit numerator, denominator, population, window, comparator, guardrail, and uncertainty, and each marked with whether it was executed. None was executed: this case is illustrative and no corpus, model, or deployment exists behind it. The status column states that plainly so that no reader mistakes a specified test for a run one.

### 9.1 Primary metric — authorised resolution rate

| Field | Definition |
|---|---|
| Numerator | answered questions whose cited clause equals the adjudicated controlling clause version, whose quotation verifies, and whose implied commitment is inside the acting tier |
| Denominator | all eligible policy questions in the window |
| Population | chat, voice-transcript, and store-handheld surfaces in one market family, excluding greetings, order-status lookups, and questions in unsupported languages |
| Window | one quarter, evaluated weekly with a quarterly roll-up |
| Comparator | incumbent keyword tool plus curated FAQ, with human-applied judgement, scored by the same adjudication protocol |
| Guardrail | unauthorised commitment rate at or below 0.0579 overall and at or below 0.0900 on the handheld surface |
| Uncertainty | Wilson interval at 95 percent on the weekly rate, reported as a band rather than a point |
| Status | **not-executed** — the specified test, with no run behind it |

The metric deliberately requires all three conjuncts. A citation that matches but whose quotation does not verify is a fabrication with a lucky identifier; a verified quotation whose commitment exceeds the tier is an authority breach that reads correctly. Scoring citation match alone would pass both.

### 9.2 Secondary metrics

| Metric | Numerator / denominator | Guardrail it protects | Status |
|---|---|---|---|
| Retrieval recall at five | questions whose adjudicated clause appears in the top five / questions with any candidate | coverage | not-executed |
| Verifier pass rate | answers whose quoted span matches verbatim / answers emitted | fabrication | not-executed |
| Authority breach rate | answers whose implied commitment exceeds acting tier / answered questions | hard constraint C-02 | not-executed |
| Superseded-citation count | answers citing a clause version outside its interval / answered questions | hard constraint C-01, must be exactly zero | not-executed |
| Escalation queue time | median and 90th percentile wait for abstained questions | hard constraint C-03 | not-executed |
| Agent skim time | median minutes to approve a skimmed answer | the middle action's viability | not-executed |
| Recontact rate | questions followed by a related contact within 72 hours / answered questions | silent wrongness | not-executed |
| Replay fidelity | answers reproduced from the archived index / sampled answers | auditability | not-executed |

Two of these deserve their own treatment because they are usually missing.

**Superseded-citation count is a binary guardrail.** It is not a rate to be trended; it is a count that must be zero. An organisation that reports it as a rate has already accepted a non-zero volume, and the first enforcement action against a customer commitment will discover the count from the other direction. The evaluation therefore reports the count with its 95 percent upper bound, and a count of zero on 19,000 answers gives an upper bound of `3.0 / 19,000 = 0.000158` by the rule of three, which is the honest statement: not "never happens" but "below 1.6 in 10,000".

**Replay fidelity is the auditability test.** Fifty answers are sampled, the archived index and corpus version are restored, and the pipeline is re-run. Fidelity counts answers whose action and citation are identical. A fidelity below 1.0 means the decision log is missing a dependency — an unversioned prompt, an unrecorded threshold, a live feature — and the correct response is to suspend answering for that surface until the missing dependency is logged, because an unexplainable decision cannot be defended.

### 9.3 Adjudication protocol

The label is the controlling clause version, produced by two adjudicators reading the corpus as of the decision timestamp, with a third resolving disagreements.

Sampling is stratified across surfaces and across the band table, not uniform, because the bands near the threshold carry the decision and are the ones a uniform sample under-covers. A constructed allocation of 300 adjudicated questions:

| Stratum | Questions sampled | Agreement (kappa) | Note |
|---|---|---|---|
| Handheld, band 0.80 to 0.90 | 90 | 0.81 | short questions, item in hand, frequent exception clauses |
| Chat, band 0.70 to 0.80 | 90 | 0.88 | threshold band, highest decision value |
| Voice transcript, band 0.90 to 1.00 | 60 | 0.74 | transcription errors change the controlling clause |
| Mixed, below 0.70 | 60 | 0.69 | ambiguous questions, high genuine disagreement |

Kappa of 0.69 in the low band is the number that matters, because it says the label is unreliable exactly where the gate rejects. That is tolerable: a rejected question costs an escalation that would have happened anyway. The corresponding number in the threshold band, 0.88, is what makes the threshold defensible, and the 0.74 on voice transcripts is a warning that voice should not share a threshold with chat until transcription quality improves.

The protocol requires adjudicators to record the clause version they used and the corpus export they read, because an adjudicator reading today's corpus will apply today's rules to a question asked last month, which inflates the measured error rate of a correct system.

### 9.4 Validation design and what it can and cannot show

The comparison is a retrospective replay on the same quarter for both the incumbent process and the proposed gate, with the same adjudicated labels. That design controls for question mix, which a live A/B would not, and it cannot control for the human adaptation that a live rollout produces: agents working with the assistant change what they escalate, and customers receiving cited answers ask follow-ups differently.

The honest summary of a replay is a directional comparison with a stated bias direction, not an effect size. Here the bias runs against the incumbent, because the replay scores the incumbent's keyword output without the human judgement that the incumbent process actually applies. The 20.2 percent saving computed in §6.2 is therefore an upper bound on the true saving, and the queue-constrained 4.2 percent in §8.4 is the more defensible figure.

A live test would need a cluster-randomised design over agents or stores, because questions are not independent — one agent's handling of a hard case contaminates their next three — and because the queue is shared, so a treatment effect on escalations spills into the control arm. The minimum detectable effect for a store-level randomisation with 40 stores, a weekly question volume of 3,700 per store, and a between-store standard deviation of 4.2 percentage points on the answer rate:

```text
MDE = 2.8 * sigma * sqrt(2 / k)
    = 2.8 * 4.2 * sqrt(2 / 40)
    = 11.76 * 0.2236
    = 2.63 percentage points
```

A 2.63-point MDE on a 39.6 percent answer rate is detectable within a quarter, so the design is feasible; but the same calculation on unauthorised commitment rate, whose between-store standard deviation is 1.1 points, gives `2.8 * 1.1 * 0.2236 = 0.69` points, which is above the entire observed rate of 0.24 percent — the guardrail is unmeasurable at that sample size, which is why it must be monitored as a count with a rule-of-three bound rather than tested for a difference.

Takeaway: the primary metric is measurable in a quarter and the guardrail is not, so the guardrail is enforced by construction and audited by count.

## 10. Action Taken

The actions below are what the case owner would do, sequenced by what unblocks the most work, each with an owner role and a stopping condition. They are described as decisions taken in the illustrative narrative; no organisation executed them.

**Action 1 — Freeze the corpus export and measure its defects.** Owner: policy owner with a data engineer. Deliverable: counts of clauses with missing validity intervals, clauses below 25 words, near-duplicate pairs above 0.93 cosine, jurisdiction fields disagreeing with paths, and identifiers that changed instead of versioning. Stopping condition: the five counts are published and the two largest are assigned. This action runs first because every later number depends on clause identity, and because the counts are cheap.

**Action 2 — Rebuild the splitter on document structure.** Owner: retrieval engineer. Deliverable: a clause table with stable identifiers, heading paths, and per-clause validity, plus a diff against the previous split showing identifier churn. Stopping condition: identifier churn on a re-export of unchanged documents is under 0.5 percent. Churn above that means the splitter depends on content that changes, and the audit trail will break at the next amendment.

**Action 3 — Move the authorization predicate into the query path.** Owner: retrieval engineer with the compliance owner. Deliverable: a predicate evaluated at query time against clause validity, jurisdiction, and channel, with `open` encoded explicitly, plus a test suite covering boundary instants: `t` exactly equal to `valid_from`, exactly equal to `valid_to`, one second either side, and the open-ended case. Stopping condition: all four boundary tests pass and the zero-candidate share `z` is measured. A `z` above 0.15 on a healthy corpus indicates an interval-encoding defect, not a retrieval defect.

**Action 4 — Join the authority matrix to the corpus.** Owner: policy owner with the delegation-record owner. Deliverable: an action-class field on every answerable clause, a versioned matrix, and a checker that maps a drafted commitment to a matrix row. Stopping condition: every answerable clause has an action class, and clauses without one are excluded from answering rather than defaulted. Exclusion is the honest choice; a default quietly grants authority nobody granted.

**Action 5 — Fit per-surface calibration and recompute the threshold.** Owner: decision scientist. Deliverable: calibration curves for chat, voice, and handheld; the cost table with its sources; the band table with observed wrong rates; the implied `tau*` and its sensitivity to each cost input. Stopping condition: a human approves the threshold, and the approval record names the cost table version.

**Action 6 — Reconcile the gate against escalation capacity.** Owner: customer-care operations lead. Deliverable: the hourly escalation forecast against the staffing plan, the implied gate constraint, and either a staffing request or an explicit risk acceptance. Stopping condition: the plan and the gate are consistent for the peak week of the promotional calendar, not for the average week.

**Action 7 — Run the replay evaluation and publish it with its bias direction.** Owner: evaluation owner. Deliverable: the four decision metrics, the guardrail counts with rule-of-three bounds, the adjudication kappa by stratum, and the replay fidelity result. Stopping condition: the report states which comparisons are directional and which are measured, and no metric appears without its denominator.

**Action 8 — Hold the rollout at answer-with-skim.** Owner: policy owner. Deliverable: a decision on when, if ever, the skim is removed. Stopping condition: the skim-removal criterion is written before the data arrives, so that it cannot be rationalised afterwards. The criterion in this case is a 12-week window in which the recontact rate for zero-skim answers is not worse than for skimmed ones by more than 0.3 percentage points.

| Action | Unblocks | Owner | Stopping condition |
|---|---|---|---|
| 1. Corpus defect census | everything | policy owner + data engineer | five counts published, two largest assigned |
| 2. Structural splitter | citation stability | retrieval engineer | identifier churn under 0.5 percent |
| 3. Query-path predicate | coverage measurement | retrieval engineer + compliance | four boundary tests pass, `z` measured |
| 4. Authority join | hard constraint C-02 | policy owner + delegation owner | no answerable clause without an action class |
| 5. Calibration and threshold | the gate itself | decision scientist | human approval naming the cost table version |
| 6. Capacity reconciliation | C-03 at peak | operations lead | peak week consistent, not average week |
| 7. Replay evaluation | the claim | evaluation owner | bias direction stated, denominators shown |
| 8. Skim-removal criterion | the middle action's exit | policy owner | criterion written before data |

Two sequencing notes carry the transferable content. Action 1 precedes Action 2 because a splitter built on a corpus with unstable identifiers will encode the instability into the new scheme. Action 6 precedes Action 7 because an evaluation run against a gate that the queue cannot support measures a policy that will not be operated.

Takeaway: eight actions, of which five are corpus and authority work and only one is modelling.

## 11. Outcome & Results

The outcome is a design that clears its compliance invariants, clears its cost test in the unconstrained case, and fails its capacity test in the constrained case. That mixed result is the honest report.

| Quantity | Value | Basis | Status |
|---|---|---|---|
| Eligible questions per quarter | 48,000 | constructed | ILLUSTRATIVE |
| Questions with a candidate set after predicate | 44,880 (93.5 percent) | constructed counts | ILLUSTRATIVE |
| Retrieval recall at five | 0.912 | constructed | ILLUSTRATIVE |
| Verifier pass rate | 0.964 | constructed | ILLUSTRATIVE |
| Authority pass rate | 0.931 | constructed | ILLUSTRATIVE |
| Answered share at cost threshold | 39.6 percent (19,000) | derived in §6.3 | ILLUSTRATIVE |
| Answer-route cost | $20,900 | derived in §6.2 | ILLUSTRATIVE |
| Escalation cost | $266,800 | derived in §6.2 | ILLUSTRATIVE |
| Wrong-answer cost | $64,820 | derived in §6.2 | ILLUSTRATIVE |
| Total cost at threshold | $352,520 | derived in §6.2 | ILLUSTRATIVE |
| Baseline total cost | $441,600 | derived in §6.2 | ILLUSTRATIVE |
| Unconstrained saving | $89,080 (20.2 percent) | derived in §6.2 | ILLUSTRATIVE |
| Saving under capacity constraint | $18,760 (4.2 percent) | derived in §8.4 | ILLUSTRATIVE |
| Superseded citations | 0 required; observed upper bound 1.6e-4 by rule of three | specified | not-executed |
| Deployed impact | unknown | no deployment | not-measured |

The gap between the 20.2 percent and the 4.2 percent is the single most important number in the case, because the first is what a business case would quote and the second is what the operation would deliver. The mechanism is not subtle: the queue is shared with the questions the pipeline cannot answer at all, so the 11,266 upstream failures are non-negotiable escalations that consume over half the capacity before the gate makes a single decision.

Recovering the gap requires one of three owner decisions, and each has a different risk profile:

More staffing, at 4.2 full-time equivalents, restores the 20.2 percent saving minus the 4.2 FTE cost. At a loaded rate of `0.60 * 60 = 36` USD per agent-hour, 1,647 agent-hours cost $59,300 per quarter, so the net becomes `89,080 - 59,300 = 29,780`, or 6.7 percent. The staffing route is real but consumes two thirds of the saving it unlocks.

A higher wrong-answer tolerance, if `C_w` is genuinely $45 rather than $140 because most breaches are caught before honouring, gives `tau* = 8.10 / 45 = 0.18`, which accepts the third band and cuts escalations by 9,300 to 19,700 — inside capacity, with a saving computed as `28,300 * 1.10 + 19,700 * 9.20 + 1,616 * 140 = 31,130 + 181,240 + 226,240 = 438,610`, or 0.7 percent. Loosening tolerance is nearly worthless because the wrong-answer cost rises faster than the escalation cost falls.

Corpus and splitter work that lifts `z` from 0.065 toward 0.02 adds 2,160 questions to the candidate pool and, at a 0.5172 acceptance rate, about 1,100 answered questions, moving the saving by `1,100 * 5.30 = 5,830` without changing capacity much. This is the best value per dollar of the three, which is consistent with claim four in §7.

**What would have to be measured to convert this from design to result.** Four things, in order of difficulty: a real corpus with real validity intervals, so that `z` is a measurement rather than a construction; adjudicated labels at the kappa levels assumed here, so that the threshold band is defensible; a time study that supports the 1.5-minute skim and the 13-minute handling assumptions, since both enter `tau*` linearly; and a live replay with the bias direction confirmed rather than assumed. Without the third, the entire cost derivation is an assumption dressed as a calculation, and a 20 percent change in either time estimate moves the threshold by roughly the same proportion.

Takeaway: the design's value is contingent on staffing and on two time-study numbers, and neither is a modelling question.

## 12. Failure Modes & What Went Wrong

The illustrative build did not go to plan in five ways that are worth separating, because they have different symptoms and different fixes. Each is stated with the mechanism, what it looks like from the outside, the check that discriminates it from its nearest lookalike, and the design change that closes it.

### 12.1 Clauses published before they are effective

**Mechanism.** The corpus export includes every published version, and an amendment is published on day 1 with an effective date 9 days later. A nightly index rebuild picks up the new text on day 2, while the clause's `valid_from` still says day 10. If the predicate reads validity from the index's own metadata but the index carries the new text under the old version identifier, the predicate admits text that is not yet law.

**Symptom.** Answers that are correct against the published corpus and wrong against the in-force corpus, concentrated in the 9-day windows around amendments, which is 4.2 percent of questions.

**Competing causes.** A genuine policy change in the market (not a build problem); a training-data or prompt update that shifted phrasing; an adjudication error where the adjudicator read the newer corpus.

**Discriminating check.** Join answers to the amendment calendar. If the error concentration sits inside the open pre-effective windows rather than uniformly across the quarter, it is the versioning defect. If errors are spread uniformly, it is not.

**Correction and prevention.** Version text and interval together, in one row, written by one transaction; never refresh one field without the other. Add a build-time assertion that no clause row has a text hash differing from its version's recorded hash. Verification: replay 50 answers from a pre-effective window and confirm they cite the prior version.

### 12.2 Valid but non-controlling clause wins on similarity

**Mechanism.** An exception clause and its parent rule are both in force. The question mentions a specific product and a specific channel, which lexically matches the exception's vocabulary more strongly than the general rule's, so the exception outranks the rule. The exception narrows the general rule rather than replacing it, so citing the exception alone states a condition without the rule it modifies.

**Symptom.** Citations that are individually accurate and jointly incomplete. Adjudicators mark these wrong, but the answer reads well, so human reviewers without the corpus in front of them approve it.

**Competing causes.** A retrieval recall problem, where the right clause was never a candidate; a chunking problem, where the rule and the exception were merged into one span; a drafting problem, where the model cited the right clause and wrote the wrong answer.

**Discriminating check.** Test recall of the controlling clause with the exception excluded from the corpus. If recall jumps, the clause is retrieval-competitive rather than absent and the answer is a selection problem. Check span lengths against the splitter's output to rule out merging. Re-read the draft against the cited span to rule out drafting.

**Correction and prevention.** Index the heading path in the retrievable text, so an exception carries its parent rule's terms; retrieve parent-plus-child as a unit when the splitter marks a dependency; require the answer to cite the controlling rule whenever it cites an exception. Verification: measure the share of answers citing an exception without its parent, which should be zero by construction.

### 12.3 Fabricated clause identifiers that pass a plausibility check

**Mechanism.** The model is asked to cite and produces an identifier in the corpus's format that does not exist, because it interpolated between two real neighbours. A format check passes, a human reviewer who does not open the link accepts it, and the answer ships.

**Symptom.** Citation resolvability below 1.0 — typically 0.982 in the constructed run, which sounds excellent and means 342 unresolvable citations per quarter across 19,000 answers.

**Competing causes.** A stale index (identifier exists but is not yet indexed); a rendering bug in the link template; an identifier that was legitimately retired.

**Discriminating check.** Three-way classification of every unresolvable citation: not in the current corpus at all; in the corpus but not in the archived version used at decision time; in the corpus and in the archived version but not in the index. A fabricated identifier lands in the first bucket and also fails a neighbour-similarity test against real identifiers.

**Correction and prevention.** Constrain generation to a closed set: the model selects an identifier from the retrieved candidate list rather than emitting one, and the verifier requires the quoted span to be a substring of that candidate's text after whitespace normalisation. Verification: count citations not present in the candidate set, which must be zero, and count verbatim failures separately so the two failure types are never conflated.

### 12.4 Authority breach that reads as compliance

**Mechanism.** The citation is correct, the quotation verifies, and the drafted answer offers a goodwill credit of $120 to a tier-one agent whose limit is $45. Nothing in the retrieval path detects this, because authority is not a property of the clause — the clause authorises the credit class; the tier bounds the amount.

**Symptom.** Unauthorised commitments at a rate uncorrelated with citation quality. In the constructed run, 6.9 percent of otherwise correct answers breached the matrix, and the breaches clustered in the goodwill and price-match classes.

**Competing causes.** A stale matrix (the agent's limit was raised last month); an extraction failure (the commitment was in the answer but the extractor missed it); a genuine policy gap (the clause authorises an action no tier may take).

**Discriminating check.** Compare the matrix version against the delegation record for the decision date. If current, run the commitment extractor over the answer with a human-labelled set to measure extraction recall; if extraction recall is high, the breach is real. A policy gap is identified by searching for authorising clauses whose action class has no tier that permits it.

**Correction and prevention.** Run the authority check on the drafted answer, not on the retrieved clause; make the check a hard gate rather than a score; require the extractor to emit an action class and a magnitude and to fail closed when either is missing. Verification: the breach rate on a held-out adjudicated set, reported per action class rather than in aggregate, because the aggregate hides the goodwill concentration.

### 12.5 Calibration that drifts by surface

**Mechanism.** Calibration is fitted on pooled data. The handheld surface answers shorter questions with a different score distribution, and its wrong-answer rate at a score of 0.85 is 1.6 times the pooled rate. The pooled curve therefore over-answers on handheld.

**Symptom.** Aggregate accuracy stable while one surface's unauthorised rate exceeds its guardrail. This is the failure that a global dashboard cannot see.

**Competing causes.** A genuine shift in the handheld population (a new product category); a prompt or retriever change that affects short questions more; a queue effect where handheld abstentions are handled by an agent who re-answers in chat, so the handheld answer's error is attributed to chat.

**Discriminating check.** Compute per-surface reliability diagrams. If the handheld curve is above the diagonal at the high-score end while the pooled curve is on it, the pooled curve is the defect. Cross-check by re-attributing recontacts to their originating surface; if the surface gap closes, the cause is attribution, not calibration.

**Correction and prevention.** Fit and store calibration per surface; require a minimum slice size before a surface gets its own threshold, and route below-threshold surfaces to the strictest existing calibration rather than to the pooled one. Verification: per-surface guardrail rates reported alongside the aggregate on every evaluation run.

### 12.6 What went wrong in the process, not in the system

The build also made three process errors worth naming, because they recur.

The team measured answer quality before measuring citation correctness, and spent six weeks improving helpfulness on a set that the gate would have rejected. The lesson is that the metric must be computed on the same population the decision acts on.

The team treated the escalation queue as a downstream consequence rather than an input, which produced a design that cleared its cost test and could not be operated. The lesson is that capacity constraints enter the objective, not the implementation.

The team quoted the unconstrained saving of 20.2 percent in its first review, and the number was arithmetically correct and operationally meaningless. The lesson is that a saving figure is only interpretable alongside the constraint set under which it was computed.

Takeaway: five technical failure mechanisms, three of which are corpus-engineering defects that no amount of model tuning would have fixed.

## 13. Transferable Lesson

The transferable content is a five-part discipline for any system that answers questions from an authoritative document set, and each part is stated with the failure it prevents.

**Lesson one: model the authorization predicate before you model relevance.** In every document-governed domain — policy, contract, tariff, clinical protocol, regulatory filing — the question "is this passage applicable" has a structured answer and the question "is this passage similar" does not. The structured question must be answered first and deterministically. When it is answered second, as a reranking feature, ineligible clauses occupy the top of the list, and the system's best behaviour becomes citing something that was true last quarter. The diagnostic is simple: if you cannot state the applicability predicate as a boolean expression over named fields, you do not yet have a retrieval problem, you have a document-model problem.

**Lesson two: the metric must be computed on the population the decision acts on.** Answer quality measured on answered questions cannot improve coverage, because coverage is decided before quality is observable. This is the general form of the evaluation-mismatch pattern, and it is not confined to retrieval: a fraud model scored on flagged transactions, a ranking model scored on shown items, a credit model scored on approved applicants all share the defect. The corrective question is always "which decisions produced the rows in this metric", and if the answer is "the decisions I am trying to improve", the metric is circular.

**Lesson three: price the threshold, do not choose it.** A confidence threshold is a statement about the relative cost of two errors and one correct action. It follows that the threshold has an owner, a derivation, and a revision schedule, and that none of the three is the modeller's taste. When the threshold is chosen as a round number, two things happen: it drifts out of alignment as labour costs change, and it cannot be defended in a review, because there is no argument behind it other than observed performance on a labelled set whose error costs differ from the production ones.

**Lesson four: cheap middle actions dominate binary choices.** Answer-or-escalate is a two-point approximation to a decision with at least three useful outcomes, and the middle one — answer but require a cheap human confirmation — usually carries a disproportionate share of the value at a small share of the risk. In this case it carried 31.5 percent of the value and 89.3 percent of the wrong-answer events, which is exactly what a middle action is for: it absorbs the cases that are nearly good enough. Designs that skip it are not more conservative; they are more expensive per unit of risk avoided.

**Lesson five: version the text and the interval as one object.** Every domain with effective dating eventually produces a system where the wording and the validity metadata can be refreshed independently, and every such system eventually applies a rule before or after it was in force. The defect is structural, not behavioural, so the fix is structural: one row, one transaction, one hash assertion. The check that catches it in production is to count answers delivered inside a published-but-not-yet-effective window, which should be zero and is cheap to compute.

| Lesson | Signal that it is missing | Cheapest corrective test |
|---|---|---|
| Predicate before relevance | ineligible clauses appear in top-k review samples | express applicability as a boolean; count empty candidate sets |
| Metric on the decision population | improvement work that never moves the primary metric | ask which decisions generated the metric's rows |
| Threshold priced | a round-number threshold with no owner | recompute `tau*` from the cost table |
| Middle action present | escalation volume forced up by risk avoidance | time-study the skim; price the band |
| Coupled versioning | citations citing text that was not yet in force | assert text hash equals version hash at build time |

The one-sentence version, if a reader remembers nothing else: the hard part of authorized retrieval is proving that the passage you cited was in force, for this customer, at this moment, and inside the authority of whoever is about to act on it — and none of those four proofs is a similarity computation.

## 14. Connections & Cross-References

| Related artefact | What it contributes here | What this case adds |
|---|---|---|
| `Cheat_Sheets/01_Omnichannel_Event_Taxonomy.md` | the event and entity grains that make `question_id` and `decision timestamp` well defined | a clause-level grain as a second dimension alongside event grain |
| `Cheat_Sheets/02_Customer_Identity_Controls.md` | identity resolution across store, chat, and order records | the authority tier as an identity attribute that changes what may be promised |
| `Cheat_Sheets/03_Fulfilment_Promise_Metrics.md` | promise-versus-actual measurement and its censoring | citation correctness as a promise made to the customer and audited after the fact |
| `Cheat_Sheets/04_25_Common_Issues.md` | the failure catalogue for the unit | two entries sharpened: version coupling, and authority read from prose |
| `Case_Studies/01_BOPIS_Promise_Accuracy.md` | a promise that is measured against a later actual | the same as-of join discipline applied to a validity interval instead of a delivery event |
| `Case_Studies/02_Loyalty_Next_Best_Action.md` | ranking under a capacity constraint | a constraint that is a compliance invariant rather than a budget |
| `Case_Studies/03_Returns_Risk_Intervention.md` | the return-window policy as a business rule | the return rule as a retrievable clause with a version and an authority class |
| `Case_Studies/04_Cross_Channel_Attribution.md` | channel identity and its effect on measurement | channel as an applicability field that changes which clause governs |
| `Case_Studies/05_Omnichannel_Fulfilment_Allocation.md` | allocation under stock and capacity | authority as a second allocation constraint on what may be committed |

The nearest contrast inside the unit is Case 05, allocation. Both cases have a hard constraint that is not the model's objective, both derive a threshold from costs, and both find that the binding constraint is operational capacity rather than accuracy. They differ in what the constraint attaches to: Case 05 allocates a physical unit to a node, so the constraint is stock and capacity in the network; this case allocates a commitment to a customer, so the constraint is a delegated authority in an organisational matrix. A reader who understands the contrast has the general shape: the model proposes, a table disposes, and the table's version is part of the audit record.

The nearest contrast outside the unit, in pattern terms, is any case where the primary pattern is data leakage. Here the leakage is not a feature built from the future but a clause retrieved from the future, which behaves identically in evaluation — it inflates the measured correctness rate — and is harder to see because the leak lives in the corpus rather than in the feature pipeline. The discriminating test is the same: reconstruct the state at `t` and check whether the winning input existed then.

What this case deliberately does not connect to is model-quality literature on grounded generation. That literature is about whether an answer is supported by retrieved text; this case is about whether the retrieved text was the right text. A perfect groundedness score is compatible with a completely wrong citation, and conflating the two is how a team ends up with a well-grounded system that misquotes last quarter's policy.

### Interview pack

**Q1. What does a similarity score fail to tell you about a candidate clause?**

- **Answer:** It tells you the clause is about the same subject, and nothing about whether it may be used. Four properties must hold separately: the clause is in force at the decision timestamp, it applies to this jurisdiction and channel, it authorises the action class the answer implies, and the acting tier may commit that action. A clause can score 0.97 and fail every one of the four.
- **Reasoning:** embeddings compress topical relatedness and discard metadata; validity intervals, jurisdiction codes, and authority classes are metadata. Two versions of the same clause differ by a handful of tokens, so their cosine distance is tiny even when their legal effect is opposite.
- **Follow-up:** which of the four properties can be checked without reading the answer text, and which requires the drafted answer?
- **Red flag:** "the score is calibrated, so a high score means it is right" — calibration is about the probability of relevance, not of applicability.
- **Scoring:** 3 points for naming applicability as separate from relevance; 5 for listing at least three of the four predicates; 7 for noting that the authority test needs the drafted answer.

**Q2. With escalation at $10.76, answering at $1.28, and a breach at $140, what is the break-even wrong-answer probability?**

- **Answer:** `(10.76 - 1.28) / 140 = 9.48 / 140 = 0.0677`. Any band whose observed unauthorised rate is below 6.77 percent should be answered; above it, escalated.
- **Reasoning:** the comparison is between the certain cost of escalating and the expected cost of answering, so the expected cost is linear in the wrong rate and the break-even is the ratio of the cost difference to the breach cost. Check: at 0.0677, `0.0677 * 140 + 1.28 = 9.48 + 1.28 = 10.76`, exactly the escalation cost.
- **Follow-up:** the loaded rate rises 20 percent. Recompute and explain why the threshold moves more than the cost difference does.
- **Red flag:** dividing by the answering cost, or subtracting the breach cost from the escalation cost.
- **Scoring:** 4 for the correct ratio and value; 6 for showing the check that both routes cost the same; 7 for correctly recomputing at 0.0677 after the rate change.

**Q3. A band holds 9,300 questions with a 12.4 percent wrong rate. Accept or escalate, and by how much?**

- **Answer:** Escalate. Accepting costs `9,300 * 1.10 + 1,153 * 140 = 10,230 + 161,420 = 171,650` against escalating at `9,300 * 9.20 = 85,560`, a loss of 86,090.
- **Reasoning:** the band's wrong rate is more than twice the break-even of 5.79 percent, so the marginal test fails decisively rather than marginally. The dominant term is the 1,153 breaches at $140, which is 94 percent of the accepted-route cost.
- **Follow-up:** what wrong rate would make this band exactly break even, and how far is that from the observed one?
- **Red flag:** comparing the wrong rate to 50 percent, or accepting because 87.6 percent are correct.
- **Scoring:** 3 for escalating; 5 for the correct 86,090 figure; 7 for identifying the breach term as dominant and computing the break-even rate of 5.79 percent.

**Q4. Coverage is 39.6 percent. How much does recall have to improve to add one point of coverage, and what does that imply about where to invest?**

- **Answer:** One point of coverage on a 39.6 percent base is a 2.53 percent relative gain, which at recall 0.912 requires 2.30 points of recall improvement. The same point costs 1.31 points of gate acceptance.
- **Reasoning:** coverage is a product of five factors, so a relative gain in any single factor moves coverage proportionally; the absolute movement needed differs by each factor's size. Recall near 0.91 needs a 2.3-point absolute gain; the gate near 0.52 needs only 1.31 points.
- **Follow-up:** why is buying coverage at the gate usually the wrong choice even though it is the cheapest in points?
- **Red flag:** claiming recall has more headroom because 0.912 is below 1.0, without computing the relative conversion.
- **Scoring:** 2 for the elasticity insight; 5 for both conversions; 7 for explaining that gate gains trade against the wrong-answer guardrail while recall gains do not.

**Q5. The zero-candidate share `z` jumps from 6.5 to 41 percent overnight. What is the first hypothesis?**

- **Answer:** An interval-encoding defect, most likely `valid_to` stored as an empty string so that a `valid_to > t` comparison drops every open-ended clause — 61 percent of this corpus.
- **Reasoning:** the predicate runs, returns rows, and logs no error; the only symptom is the empty-candidate count. A genuine corpus change large enough to empty 41 percent of candidate sets is implausible overnight, whereas a schema or serialisation change is exactly the kind of change that lands overnight.
- **Follow-up:** name a second hypothesis and the check that separates it from the first.
- **Red flag:** "the retriever is broken, retrain it" — the retriever is not in the failing path.
- **Scoring:** 3 for naming the encoding defect; 5 for the open-ended mechanism; 7 for a discriminating check such as counting clauses by whether `valid_to` parses as a date.

**Q6. Choose between fixed 180-token windows and structural clause splitting. Justify with a measurable consequence.**

- **Answer:** Structural splitting, because citation requires naming one clause and windows span several. The measurable consequence is the verifier pass rate: with windows, most returned spans are truncated mid-clause, so the verifier rejects or the citation names a clause whose text is not in the span.
- **Reasoning:** the decision unit is a clause, and a chunking scheme that does not align with the decision unit makes every downstream check approximate. The cost of structural splitting is engineering effort on the splitter and a dependency on document formatting quality.
- **Follow-up:** what does identifier churn measure, and what threshold would you set before trusting the splitter?
- **Red flag:** choosing windows because they are simpler, without connecting the choice to the verifier.
- **Scoring:** 3 for choosing structural; 5 for the verifier-based justification; 7 for naming churn on re-export as the acceptance test.

**Q7. Zero superseded citations are observed in 19,000 answers. How confident may you be, and how should it be reported?**

- **Answer:** By the rule of three, the 95 percent upper bound is `3 / 19,000 = 0.000158`, so the honest statement is "below 1.6 in 10,000", not "never". Report it as a count with its upper bound, and treat it as a binary guardrail whose target is zero rather than a rate to be trended.
- **Reasoning:** a zero count carries no information about the rate beyond an upper bound, and the bound scales inversely with the number of opportunities. Reporting a percentage of zero invites the reader to treat the absence as a property rather than an observation.
- **Follow-up:** how many answers would you need to observe to claim a bound of 1 in 100,000?
- **Red flag:** reporting "0 percent" as a measured rate, or claiming the guardrail is verified.
- **Scoring:** 3 for the rule of three; 5 for the correct bound; 7 for the reporting recommendation and the sample-size implication, which is 30,000 answers.

**Q8. Aggregate accuracy is stable but the handheld surface breaches its guardrail. Diagnose.**

- **Answer:** Pooled calibration is the leading cause: the handheld slice has a different score distribution, so its wrong-answer rate at a given score is about 1.6 times the pooled rate, and a pooled curve over-answers there. The competing cause is attribution — handheld abstentions resolved in chat attribute their errors to chat.
- **Reasoning:** the pooled curve is on the diagonal while the handheld curve sits above it at the high-score end. A stable aggregate is exactly what a slice-level defect looks like when the slice is a minority of volume.
- **Follow-up:** what minimum slice size would you require before giving a surface its own threshold?
- **Red flag:** concluding the model degraded, when the aggregate says it did not.
- **Scoring:** 3 for naming calibration; 5 for the mechanism; 7 for the attribution cross-check that separates the two causes.

**Q9. When may the agent skim be removed from the answer-with-skim action?**

- **Answer:** Only against a criterion written before the data arrives. Here: a 12-week window in which the recontact rate for zero-skim answers is not worse than for skimmed answers by more than 0.3 percentage points, with the escalation queue unchanged.
- **Reasoning:** the skim exists to absorb the band whose wrong rate is between the threshold and roughly 0.12, so removing it converts a cheap confirmation into either a full escalation or an unconfirmed answer. The criterion has to be fixed in advance because the natural post-hoc criterion — "accuracy looks fine" — is satisfied by both a good and a lucky sample.
- **Follow-up:** which quantity would you check first if the skim time study came back at 6 minutes rather than 1.5?
- **Red flag:** "remove it once accuracy exceeds 95 percent" — accuracy on answered questions says nothing about the skim's marginal contribution.
- **Scoring:** 3 for requiring a pre-registered criterion; 5 for the recontact comparison; 7 for recognising that a 6-minute skim collapses the middle action into full handling.

**Q10. Escalation capacity is 21,400 but the threshold produces 29,000. What happens to the projected saving?**

- **Answer:** The queue can only be satisfied by answering more, not fewer: upstream failures already consume 11,266 of the capacity, so the gate must accept 26,600 of 36,734 survivors. Cost becomes `29,260 + 196,880 + 196,700 = 422,840`, so the saving falls from 89,080 to 18,760, or 4.2 percent.
- **Reasoning:** the capacity constraint is on total escalations, and the gate only controls the share of questions that survive upstream. Forcing fewer gate escalations requires accepting the 7,600-question band at a 12.4 percent wrong rate, which costs about $86,000 in breach exposure.
- **Follow-up:** what staffing would restore the unconstrained saving, and how much of it would staffing consume?
- **Red flag:** assuming the queue constraint means escalating more, which is true only when upstream failures are small.
- **Scoring:** 3 for the direction; 5 for the 422,840 figure; 7 for the 4.2 percent saving and the observation that the constraint destroys 79 percent of the value.

**Q11. Why must the authority check run on the drafted answer rather than on the retrieved clause?**

- **Answer:** Because authority bounds a magnitude, and the magnitude appears only in the answer. A clause authorises the *class* of action — goodwill credit — while the matrix bounds the *amount*, and the amount is composed during drafting.
- **Reasoning:** binding authority to the clause would require every clause to enumerate amounts for every tier, which duplicates the matrix in the corpus and guarantees the two drift apart. Binding it to the answer keeps one copy of the matrix and makes the check a lookup against the extracted commitment.
- **Follow-up:** what happens to the check when the extractor cannot find a commitment in the answer text?
- **Red flag:** "the clause is authoritative, so anything it says is allowed" — that is the exact confusion that produces a $120 credit from a tier-one agent.
- **Scoring:** 3 for the class-versus-magnitude distinction; 5 for the single-copy argument; 7 for failing closed when extraction is empty.

**Q12. Fine-tune the generator on policy prose, or spend the effort on a verbatim verifier?**

- **Answer:** The verifier. Fine-tuning makes the model fluent in the corpus's register, which makes a wrong citation sound right; the verifier makes a wrong citation impossible to emit, regardless of fluency.
- **Reasoning:** the two interventions target different quantities. Fine-tuning can raise the base rate of correct citation styling; the verifier bounds the worst case. In a domain where one breach costs 127 answered questions' worth of budget, bounding the worst case dominates raising the mean.
- **Follow-up:** what would fine-tuning buy that the verifier cannot, and under what condition would that be worth it?
- **Red flag:** choosing fine-tuning because it addresses the metric that is easiest to plot.
- **Scoring:** 3 for choosing the verifier; 5 for the fluency-is-harm argument; 7 for the mean-versus-worst-case framing with a numeric comparison.

**Q13. How many additional handlers does a 7,600-question escalation gap require?**

- **Answer:** `7,600 * 13.0 = 98,800` agent-minutes, or 1,647 agent-hours per quarter, which is `98,800 / (13 * 40 * 60 * 0.75) = 4.2` full-time equivalents at a 75 percent occupancy ceiling.
- **Reasoning:** the occupancy ceiling is not decoration: at 100 percent occupancy a queue has no slack and wait times diverge, so staffing must be computed against a realistic ceiling. Thirteen weeks of 40 hours is the quarterly supply per head.
- **Follow-up:** at a loaded rate of $36 per agent-hour, what share of the 89,080 saving does this consume?
- **Red flag:** dividing by 2,080 hours without the occupancy ceiling, which understates the requirement by a quarter.
- **Scoring:** 3 for the agent-minute conversion; 5 for the correct 4.2 FTE; 7 for the occupancy reasoning and the 59,300 cost that consumes two thirds of the saving.

**Q14. Citation resolvability is 0.982. Is that good, and what does it hide?**

- **Answer:** It is 342 unresolvable citations per quarter, and it hides three different failures: identifiers absent from the corpus entirely (fabrication), present but absent from the archived version at `t` (versioning), and present in both but missing from the index (index lag).
- **Reasoning:** a single resolvability rate aggregates causes with different fixes. Fabrication needs constrained decoding; versioning needs coupled text-and-interval writes; index lag needs a rebuild guarantee. Treating them as one number means the fix addresses whichever cause dominates.
- **Follow-up:** which of the three would you expect to dominate immediately after a corpus migration, and why?
- **Red flag:** calling 98.2 percent excellent without converting to a count or splitting the causes.
- **Scoring:** 3 for the count; 5 for three distinct causes; 7 for matching each cause to its own fix.

**Q15. The breach cost has a heavy tail: one event in twenty thousand costs a hundred times the mean. What changes?**

- **Answer:** The linear break-even stops being defensible. When the 99th-percentile cost exceeds about twenty times the mean, no threshold computed from an average protects against the tail, and the control has to become a prohibition on the class of clause involved rather than a probability threshold.
- **Reasoning:** a threshold prices the mean event, so it is optimal for the body of the distribution and silent about the tail. If the tail carries the regulatory or reputational consequence, the objective is no longer expected cost but a constraint on the tail, which is a different decision problem with a different answer.
- **Follow-up:** what diagnostic ratio would you compute, and what would you do if it came back at 6?
- **Red flag:** raising the breach cost estimate and leaving the threshold-based control in place.
- **Scoring:** 3 for recognising the linear model's limit; 5 for the tail-versus-mean distinction; 7 for proposing a prohibitive control and naming the ratio check.

**Q16. When is a multi-hop retrieval loop worth its cost here?**

- **Answer:** Only behind a classifier. Multi-hop helps when the answer needs two clauses combined, which is 4.1 percent of questions; paying the loop on all questions and accruing benefit on 4.1 percent is a losing trade unless the loop's marginal cost is under about 4 percent of the answering cost.
- **Reasoning:** the arithmetic is a coverage-weighted average of two regimes. If the loop costs `c` extra per question and adds value `v` on the 4.1 percent, the break-even is `c < 0.041 * v`. A classifier that routes 6 percent of questions to the loop captures most of `v` at 6 percent of `c`.
- **Follow-up:** what is the classifier's own error cost, and does routing still pay at 80 percent routing precision?
- **Red flag:** adding the loop because it improves the hardest examples, without weighting by their share.
- **Scoring:** 3 for the routing answer; 5 for the break-even inequality; 7 for the classifier-precision analysis.

Takeaway: the interview set is weighted toward thresholds, capacity, and diagnosis, because those are where this pattern is actually decided.

## 15. Quick Reference Card

| Item | Value or rule |
|---|---|
| Decision | answer / answer-with-skim / abstain / escalate-out-of-policy / baseline, per question at `t` |
| Four predicates | in force at `t`; right jurisdiction and channel; authorises the action class; inside the acting tier |
| Break-even threshold | `tau* = (C_e - C_a) / C_w = 8.10 / 140 = 0.0579` |
| Coverage funnel | `(1 - z) * r5 * v * a * g = 0.935 * 0.912 * 0.964 * 0.931 * 0.5172 = 0.396` |
| Elasticity | a relative gain in any single factor moves coverage proportionally; one point of coverage costs 2.30 points of recall or 1.31 points of gate acceptance |
| Guardrail reporting | superseded citations are a count, not a rate; zero on 19,000 answers gives an upper bound of `3 / 19,000 = 1.6e-4` |
| Capacity rule | upstream failures consume capacity first; check whether the queue binding forces more answers or fewer |
| Cost comparison | unconstrained saving 89,080 (20.2 percent); capacity-constrained 18,760 (4.2 percent); the gap is 4.2 FTE of staffing |
| Corpus invariants | text and validity written in one transaction; `open` encoded explicitly; identifier stable across amendments |
| Order of operations | predicate → retrieve → verify → draft → authority → gate → queue |

Three diagnostics worth memorising, because each catches a defect that is otherwise invisible: a jump in the empty-candidate share means an interval-encoding bug, not a retrieval bug; a stable aggregate with one surface breaching means calibration is pooled across slices; and a falling verifier pass rate with rising answer rate means the gate is admitting drafts the retriever could not support.

The single boundary that decides whether this design is appropriate: it requires that the governing rule be written down, versioned, and attributable to an authority. Where policy lives only in experienced agents' judgement, the predicate cannot be constructed, the citation cannot be verified, and the correct design is a decision-support tool that shows candidate clauses without answering — which is the baseline, and honestly labelled as such.
