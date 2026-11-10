---
case_id: "CASE-MLSD-001"
case_title: "Choosing an enforcement contract for a multilingual moderation classifier"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Platform trust and safety / content policy enforcement"
primary_pattern: "decision contract"
secondary_patterns: ["calibrated threshold selection", "bounded human review capacity"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Trust and safety policy owner with ML platform lead"
source_index: "INDEX.md"
review_trigger: "Changed policy definition, prevalence, review capacity, appeal rate, or model calibration evidence"
---

# Choosing an enforcement contract for a multilingual moderation classifier

## 1. The One-Line Framing

Given a stream of user-generated posts with a calibrated policy-violation score per post, decide which posts are auto-restricted, which are queued for human review, and which pass untouched, for a trust-and-safety operations team, to reduce confirmed policy harm per day without exceeding the human review budget or wrongly removing legitimate speech, compared with the current rule-plus-review baseline.

The task category is classification coupled to an allocation decision, and the decision repeats continuously at post-ingestion time.
The decision unit is one post at one policy version; it is not a user, a thread, or an account strike.
The primary success condition is the daily count of confirmed violating posts that reach no enforcement action, expressed per million posts ingested, subject to a bound on wrongful restrictions per day and a bound on reviewer minutes consumed.
The primary reusable pattern is the decision contract: owner, input grain, available actions, objective, guardrails, and evidence boundary are all stated before any model is selected.
The two secondary patterns are calibrated threshold selection and bounded human review capacity.
The baseline is the existing keyword and hash-match ruleset with a first-in-first-out review queue and no score ordering.

**Source declaration:** This case is an illustrative teaching construct.
The platform, its prevalence figures, its score distributions, its reviewer productivity, its cost coefficients, its appeal rate, and every timestamp are invented for instruction.
Nothing in this document is a measurement of any named organization, a reconstruction of any published moderation study, or evidence of a realized outcome.
The arithmetic is checked and reproducible; the inputs are not evidence.

The decisive trade-off is that a fixed review capacity converts a ranking problem into an enforcement problem.
At the invented prevalence of 0.8% and a 2% false-positive rate, a naive "flag everything above 0.5" policy produces roughly 793,600 wrongful candidate restrictions per day against 272,000 true ones, so precision near 25.5% forces either a much higher auto-action threshold, a ranked review queue, or both.
The obvious simpler framing, "train a better classifier and deploy it", misses that the binding constraints are reviewer minutes, the cost asymmetry between a missed violation and a wrongful restriction, and the fact that a threshold is a policy instrument rather than a model hyperparameter.

## 2. Context & Background

An invented social posting platform receives user submissions across 14 declared interface languages on one ingestion API.
Each accepted post is written to a durable store and simultaneously emitted to an asynchronous policy pipeline.
The pipeline currently applies deterministic rules: a blocklist of exact hashes for known illegal material, a curated multilingual keyword list, and an account-level rate rule that holds rapid repeat-posters for review.
Items not matched by rules are published immediately.
Items matched are placed in a single review queue ordered by arrival time.
Reviewers see the post, the matched rule, the author's recent enforcement history, and a policy excerpt, and choose restrict, approve, or escalate to a specialist team.

The relevant stakeholders are the policy owner who defines what counts as a violation, the operations lead who staffs reviewers and owns turnaround, the ML platform lead who owns model training and serving, the appeals team who reverse decisions, the legal and privacy functions who constrain retention and access, and the product teams whose surfaces render restricted content as unavailable.
The policy owner may change the definition of a violation; when that definition changes, every historical label and every threshold justification must be re-examined, because a score threshold is only meaningful relative to a policy version and a language population.

Scale is invented but internally consistent. The platform ingests 40,000,000 posts per day, which is about 462.96 posts per second on average, with a stated diurnal peak factor of 2.4, giving a peak of about 1,111.1 posts per second.
Of those, 0.8% are assumed to be violations of the current policy, or 320,000 posts per day.
The assumed review capacity is 24 reviewers working 6 productive hours each at 45 seconds per item, which yields 11,520 reviews per day.
Review turnaround is targeted at 24 hours from ingestion for queued items, and auto-restriction takes effect within 30 seconds of ingestion.

Prior attempts are part of the invented history and are deliberately not flattering.
An earlier single-language classifier was trained on English data and applied to all languages; its measured behaviour in this scenario is unknown, but the design review recorded that the label distribution differed sharply by language and that the score distribution was not comparable across languages.
A previous proposal to lower the threshold to 0.3 was rejected because reviewers were already saturated and the queue delay exceeded 36 hours.
No production experiment, A/B test, or measured precision figure exists in this teaching case.

Timing matters in two clocks.
The ingestion clock determines when a post becomes visible to other users, so an auto-restriction decision that arrives after publication has already allowed the harm window to open.
The label clock determines when a ground-truth adjudication exists: a violation is confirmed only when a trained reviewer or an appeal decision records a final policy determination, which in this scenario lags ingestion by up to 72 hours for queued items.
Any evaluation that joins scores to labels must respect that lag, otherwise it silently excludes recent posts and over-represents easy, fast-adjudicated cases.

Jurisdiction is deliberately not asserted.
The platform operates in multiple regions with different speech rules, and this case does not determine which law applies to which post.
Where a legal obligation would bind the design, a qualified owner must supply the authoritative source and date; the teaching arithmetic here only demonstrates how a decision contract would consume such a constraint.

## 3. The Business Question

**Illustrative stakeholder question:** "Can we stop more of the content our policy forbids, without removing more legitimate posts than we do today, and without hiring reviewers we do not have?"

| Decision contract element | Content for this case |
|---|---|
| Question | Which enforcement tiers and score thresholds should route a scored post to auto-restriction, human review, or publication? |
| Decision owner | Trust and safety policy owner, with the ML platform lead accountable for calibration evidence |
| Trigger | A multilingual classifier has finished offline training and its scores are available in shadow mode |
| Deadline | A decision is required before the next policy review cycle; enforcement may not change without a recorded policy version |
| Available actions | Keep rules only; deploy a single-threshold auto-action policy; deploy a three-tier ranked policy; defer enforcement and collect labels |
| Value and downside | Reducing confirmed violating posts that remain visible versus wrongful restrictions, appeal load, reviewer burnout, and reputational exposure |
| Objective | Minimize confirmed violating posts that receive no action per million posts, at fixed reviewer minutes |
| Hard constraints | No auto-restriction without calibration evidence for the language; no restriction of a post whose policy classification is unknown; no reviewer queue without a stated maximum delay |
| Scope | One platform, 14 interface languages, the current policy version, a 30-day operating window |
| Out of scope | Account-level sanctions, law-enforcement referral, legal determinations, ranking or recommendation changes |

The objective and its proxy deserve separation.
The business objective is harm reduction, which no dashboard observes directly.
The measurable proxy is the count of posts that a later adjudication confirms as violations and that received no enforcement action within the harm window.
That proxy is imperfect in two directions: an adjudicator may confirm a violation that caused no real-world harm, and a harmful post may be adjudicated as compliant because the policy text is ambiguous.
The guardrails that keep the proxy honest are the count of wrongful restrictions per day, the appeal overturn rate, the review turnaround at the 95th percentile, and reviewer minutes per confirmed action.

A false decision has an asymmetric shape here.
A missed violation leaves harmful content visible until a user report or a later sweep finds it, and the cost is an expected downstream exposure multiplied by the probability that the content is seen before removal.
A wrongful restriction removes legitimate speech, generates an appeal, consumes appeal-reviewer minutes, and erodes trust in the enforcement system; the cost per event may be lower or higher than a missed violation depending on the platform's stated values, and that comparison is a policy judgement, not a modelling result.
The case therefore treats the cost ratio as an explicit input with a sensitivity band rather than a discovered parameter.

The question is answerable by an action: select a tier structure and thresholds, or record a justified decision to keep the rules-only baseline while labels mature.
It is not answerable by reporting an offline AUC, because AUC does not encode prevalence, capacity, cost asymmetry, or the harm window.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Scores are calibrated within each language, not only globally | model | Reliability diagrams and expected calibration error per language on a time-respecting holdout | unverified | Threshold probability no longer equals expected precision, so the cost ledger is invalid | Recalibrate per language with isotonic or Platt scaling fitted inside folds | Any retrain or language addition / ML lead |
| A-02 | Prevalence is 0.8% of ingested posts for the current policy version | data | Stratified adjudication sample of 5,000 posts per week | unverified | Precision at fixed TPR/FPR moves roughly with prevalence | Recompute the ledger and re-rank the queue | Policy change or prevalence shift / policy owner |
| A-03 | Reviewers sustain 45 s per item across 6 productive hours | operations | Time study on a sampled queue with measurement of context switching | unverified | Capacity in reviews per day changes proportionally | Re-derive capacity and re-run threshold selection | Staffing or tooling change / operations lead |
| A-04 | Adjudicated labels arrive within 72 hours for queued items | data | Timestamp audit joining ingestion, review, and appeal decisions | unverified | Evaluation silently excludes recent posts and biases toward easy cases | Restrict evaluation to mature-label cohorts and report coverage | Backlog growth / data owner |
| A-05 | An auto-restriction can be reversed with the original post restored | infrastructure | Rehearsal of restriction and restoration on a staging corpus | unverified | Irreversible actions convert a false positive into permanent loss | Escalate to review instead of auto-action for irreversible policy classes | Any change to storage or audit / platform lead |
| C-01 | Reviewer capacity is fixed at 11,520 items per day | capacity | Staffing plan and roster | confirmed as scenario assumption | Any threshold producing more queued items creates unbounded delay | Rank the queue and cap queue admission | Hiring or tooling change / operations lead |
| C-02 | No enforcement decision may be made without a policy version identifier | governance | Schema review of the enforcement audit record | confirmed as scenario assumption | Decisions cannot be explained or appealed coherently | Block enforcement until the field is populated | Schema change / policy owner |
| C-03 | Retention of restricted content is limited to an invented 90-day audit window | privacy | Data-retention review with the privacy function | review needed | Replay and appeal analysis may lack material | Reduce retention or store derived features only | Regulatory or policy change / privacy owner |

The three assumptions most likely to reverse the design are A-01, A-02, and A-03.
If calibration fails within a language, the entire cost ledger collapses because a score of 0.995 no longer implies a 0.5% false-discovery rate.
If prevalence is materially higher than 0.8%, precision rises at the same operating point and a lower threshold becomes defensible.
If reviewer throughput is 30 seconds rather than 45, capacity rises by 50% to 17,280 items per day, which changes the optimal split between auto-action and review.

Two constraints are structural rather than local choices.
The first is that enforcement must be explainable to an appellant: a decision needs the policy clause, the evidence shown, the model version, and the reviewer identity where applicable.
The second is that the model may not be the final authority on a contested policy boundary; contested categories route to humans regardless of score, because a confidence number cannot substitute for an accountable policy judgement.

The invented budget is deliberately left unpriced.
No per-reviewer cost, no vendor inference price, and no legal exposure estimate is asserted here, because none is documented in the working tree.
Where an economic conclusion would depend on prices, the case states the required unit cost and the sensitivity of the decision to it instead of inventing a currency total.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification for this case |
|---|---|
| Unit of analysis | One post at one policy version, keyed by post identifier; a post edited after publication creates a new decision record |
| Volume and window | 40,000,000 posts per day ingested; the analysis window is a 30-day operating month; peak factor 2.4 |
| Inputs | Text and permitted media descriptors, declared language, author account age bucket, prior enforcement count, ingestion timestamp, surface metadata |
| Outcome / label | Policy violation yes/no from adjudication, with the policy clause, adjudicator role, adjudication timestamp, and final or appealed state |
| Population and exclusions | All ingested posts; excluded from training are deleted posts, posts under active legal hold, and posts whose language is outside the 14 declared languages |
| Join semantics | Score records join to adjudications on post identifier and policy version; multiple adjudications per post collapse to the latest final state, and earlier states are retained for appeal analysis |
| Splits | Entity-disjoint by author and time-respecting by ingestion date; the final test window is the most recent 14 days with mature labels only |

The label is not a property of the text alone; it is a property of the text, the policy version, and the adjudication process.
That is why the label record carries the policy clause and the adjudicator role: a change from "harassment" to "hate" changes the training target even when the post is unchanged.
Label maturity matters because a post restricted today may be appealed tomorrow; the evaluated label should be the final state after the appeal window closes, and any evaluation run before that closure must report what fraction of labels were still provisional.

Class balance in this invented scenario is 0.8% positive overall, but the case deliberately assumes heterogeneity: the invented per-language prevalence ranges from 0.4% to 1.6%.
That range is not a measured fact; it exists so that the analysis must confront the fact that a single global threshold applied across languages with different prevalence produces different precision in each language even when the model's discrimination is identical.
The per-language positive counts at 40,000,000 posts split evenly across 14 languages, with 2,857,142.86 posts per language, would be 11,428.57 at 0.4% and 45,714.29 at 1.6%; the equal-split assumption is itself a teaching simplification and is flagged as such.

Sampling for evaluation uses two distinct designs.
The first is a stratified random sample of 5,000 posts per week drawn from the ingested stream without regard to score, which estimates prevalence and the false-positive rate on the clean population.
The second is a score-stratified sample that deliberately over-samples the decision boundary, which estimates local precision near the threshold efficiently but cannot estimate prevalence.
Confusing the two is a classic error: a boundary-enriched sample will report a precision that has no relationship to the precision experienced in the production queue.

For a non-trivial share of posts the adjudication is genuinely uncertain, and the invented scenario assumes 3% of sampled posts are recorded as "policy unclear" and routed to a policy review board.
Those records are neither positive nor negative; treating them as negatives understates harm, and treating them as positives overstates model error.
The analysis therefore reports a three-way confusion structure and separates "unclear" from "compliant" in every table.

### 5b. Tools & Stack

The proposed stack in this teaching scenario is a text and metadata feature pipeline, a fine-tuned multilingual encoder classifier, a calibration layer per language, a policy engine that maps score bands to actions, and a durable enforcement audit store.
The serialization and serving details are intentionally generic because no deployed version exists in the working tree.
Each component has an owner: the feature pipeline belongs to the data platform, the classifier and calibration layer to the ML platform, the policy engine to the trust-and-safety engineering team, and the audit store to the platform team under privacy review.

The policy engine is the component that must be boring.
It evaluates a small number of declared rules in a fixed order: hard-block classes that require deterministic matching, contested classes that always route to humans, score bands that map to auto-restriction or review, and default publication.
Its inputs and outputs are logged with the policy version, model version, calibration version, and threshold version, so an appeal can reconstruct exactly why a post was restricted.
Version identifiers are not decoration; the invented failure analysis later shows that a threshold change without a matching calibration version produces a silent precision collapse.

No specific framework, tokenizer, accelerator, or hosting price is asserted.
Where a deployment decision would need a measured latency or cost, the case states the measurement required: per-language p50 and p95 end-to-end scoring latency at peak ingest, the fraction of posts exceeding a 250 ms scoring budget, and the marginal cost per 1,000 scored posts at the observed peak.

### 5c. Method Summary

The method is a sequence of five steps from input to decision.
First, validate the post and the declared language and attach the policy version.
Second, compute features that are available at ingestion time only; author history features are read from a store that is updated before scoring, and the read is timestamped.
Third, produce a violation score and map it to a calibrated probability per language.
Fourth, apply the policy engine's ordered rules to choose auto-restriction, human review, or publication, with a reason code.
Fifth, write the decision record, and for queued items rank by calibrated probability with a tie-break on ingestion time.

Ranking is a separate decision from thresholding and must be stated explicitly.
If the review queue admits at most 11,520 items per day, the queue policy is "admit the highest calibrated probability first, subject to per-language and per-category minimum coverage".
The per-language minimum exists because a global ranking lets a high-prevalence language dominate the queue, starving a low-prevalence language of any review at all; the invented minimum is 4% of daily capacity per language, or 460.8 items, which is rounded operationally to 461.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Rules only (baseline) | Invented rule catalogue and operations notes | Explainable, no calibration dependency, cheap | Cannot generalise to paraphrases; keyword lists drift with language | Baseline; not measured | Any improvement claim must beat it on the same population |
| Single global threshold on an uncalibrated score | Training AUC only | Simple to operate | Score is not a probability; precision varies with prevalence and language | Rejected | Never as an enforcement basis |
| Per-language calibrated score with three tiers | Calibration and capacity analysis | Separates auto-action risk from review risk; respects capacity | Requires per-language calibration data and version discipline | Proposed | Calibration drift or language addition |
| Score-ranked review without auto-action | Ranking analysis | No irreversible wrongful removal | Leaves obvious violations visible for hours | Considered as a fallback | If auto-action precision cannot be validated |
| Defer enforcement and collect labels | Label maturity analysis | Removes policy risk during uncertainty | Harm continues; no decision value | Considered | If prevalence or policy meaning is contested |

The simplest credible baseline is the ruleset, and every proposed gain must be expressed as a difference against it on the same population, window, and adjudication protocol.
Model-only metrics are insufficient: a 0.92 AUC can coexist with worse precision at the operating point than a keyword list, because AUC integrates over the whole score range while operations consume one narrow band.

### 5d. Data provenance, quality, exclusions and leakage controls

All data in this case is constructed.
No private platform data, no user content, and no adjudication record was accessed, and the numbers are chosen for arithmetic clarity rather than realism about any specific corpus.
If implemented, access to post text would be limited to service identities and reviewers with a recorded purpose, and diagnostic work would use redacted identifiers and aggregate counts.

Five quality signals must be monitored separately because they fail in different ways.
Missing language declarations cause the calibration layer to fall back to a global curve, which silently changes precision; the diagnostic is a per-language count of fallback calibrations.
Duplicate post identifiers from client retries inflate volume and can double-count harm; the diagnostic is a de-duplication ratio on ingestion and a comparison of raw accepted posts against distinct identifiers.
Clock skew between ingestion and adjudication breaks label-maturity joins; the diagnostic is the distribution of negative latency values.
Schema drift in adjudication records breaks the policy-clause mapping; the diagnostic is the fraction of decisions whose clause is unknown.
Score distribution drift between training and serving changes the effective threshold; the diagnostic is a population-stability index per feature and per score band, reviewed weekly.

Leakage controls are specific to moderation.
Author history features must be computed from the state before the post was scored, not from a table that is backfilled after enforcement; a backfilled enforcement count encodes the model's own past decisions and produces a training signal that cannot be reproduced at serving time.
Labels adjudicated by reviewers who saw the model's score are not independent; the evaluation protocol must therefore use blind adjudication for at least the sampled quality measurements, or record that the adjudication was score-informed and treat the resulting precision as optimistic.
Finally, training data must exclude posts that entered the corpus because a model restricted them, otherwise the model reinforces its own prior errors.

Known exclusions in this teaching case are deleted posts, legal-hold posts, non-declared languages, and posts whose media could not be lawfully inspected.
Unknown exclusions are a named limitation: the case cannot state how much harm occurs inside excluded categories, and a real deployment would need a separate measurement plan for them.

### 5e. Reproducibility and method-selection ledger

| ID | Location and date | Supports | Limits |
|---|---|---|---|
| SRC-01 | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), method reference accessed for this draft, version unspecified | Leakage and inconsistent-preprocessing reasoning; fitting transforms inside folds | Does not validate this case's model, prevalence, or threshold |
| SRC-02 | [Google SRE Workbook: implementing SLOs](https://sre.google/workbook/implementing-slos/), method reference | User-centred service indicators, windows, and error budgets for the review queue | Does not establish this case's turnaround target |
| SRC-03 | [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), public framework, version and date to be confirmed before reliance | Vocabulary for documenting risk, testing, and governance ownership | A framework is not a legal determination or an efficacy result |
| CALC-01 | Section 6 of this document | Closed-form prevalence, precision, capacity, and expected-value arithmetic | Constant rates, illustrative cost coefficients, no measured distribution |
| CALC-02 | Section 6 sensitivity table | Boundary behaviour when prevalence, throughput, or cost ratio changes | One-factor-at-a-time; interactions are not enumerated |
| RUN-01 | No run exists | Nothing | No training job, evaluation suite, shadow deployment, or measured outcome |

The arithmetic in this case is reproducible with decimal arithmetic and stated rounding.
Where a percentage is reported, the base is stated in the same sentence, because the whole point of the exercise is that a precision number without its base population is not a decision input.
The case does not claim that any published moderation system behaves as modelled; the closed forms are planning instruments.

<!-- PART2 -->
