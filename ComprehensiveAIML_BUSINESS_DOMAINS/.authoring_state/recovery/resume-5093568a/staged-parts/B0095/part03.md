
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
