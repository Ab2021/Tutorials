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
## 6. Analysis Walkthrough

The walkthrough treats the action as a constrained allocation problem rather than a classifier leaderboard.

All inputs in this section are ILLUSTRATIVE planning values, not observed platform data.

The model emits a language-calibrated probability p that the post violates policy version P-2026-11.

A score has no enforcement meaning until a policy version and calibration version are attached.

Let N denote posts ingested per day, measured in posts/day.

Let pi denote violation prevalence in the eligible population, measured as a probability in [0,1].

Let TPR(t) denote the true-positive rate when threshold t is applied.

Let FPR(t) denote the false-positive rate when threshold t is applied.

Let R denote available review slots per day, measured in items/day.

Let q(t) denote the fraction of all posts that would enter a score band at threshold t.

The first calculation establishes why raw volume makes a seemingly small false-positive rate operationally large.

```text
FORMULA
Expected true positives at threshold t = N × pi × TPR(t).
Expected false positives at threshold t = N × (1 - pi) × FPR(t).

WHERE
N is eligible posts/day; pi is final-label violation prevalence; TPR and FPR are rates on a mature-label evaluation population.

CONSTRAINT
The rates must use the same policy version, language population, and action definition; random sampling is required for the FPR estimate.

NUMERICAL EXAMPLE
ILLUSTRATIVE: N = 40,000,000 posts/day, pi = 0.008, TPR(0.995) = 0.42, and FPR(0.995) = 0.00004.
True positives = 40,000,000 × 0.008 × 0.42 = 134,400 posts/day.
False positives = 40,000,000 × 0.992 × 0.00004 = 1,587.2 posts/day.
Rounded operationally, the automatic band creates about 134,400 correct restrictions and 1,587 wrongful restrictions per day.

INTERPRETATION
The false-positive rate looks tiny in percentage form, but it still produces more than one thousand legitimate posts requiring reversal capacity.
```

The dimensional check is straightforward: posts/day multiplied by a probability and a conditional rate remains posts/day.

The result is not a claim that the classifier actually achieves 0.42 TPR or 0.00004 FPR.

It is a planning calculation conditional on a measured, future validation report supporting those rates.

The formula breaks if prevalence is calculated on a score-enriched sample while FPR is calculated on random traffic.

That mismatch mixes populations and cannot yield an expected daily count.

The next calculation converts operating rates into precision, which is the probability a restricted item is truly violating.

```text
FORMULA
Precision(t) = [pi × TPR(t)] / [pi × TPR(t) + (1 - pi) × FPR(t)].

WHERE
Precision(t) is dimensionless positive predictive value for an action band; every term is estimated on the same final-label cohort.

CONSTRAINT
The denominator must be positive, and calibration or prevalence shift makes the historical estimate unsuitable for a new language or policy version.

NUMERICAL EXAMPLE
ILLUSTRATIVE: numerator = 0.008 × 0.42 = 0.00336.
False-positive contribution = 0.992 × 0.00004 = 0.00003968.
Denominator = 0.00336 + 0.00003968 = 0.00339968.
Precision = 0.00336 / 0.00339968 = 0.988328..., or 98.83%.
The corresponding illustrative false-discovery rate is 1 - 0.988328... = 1.17%.

INTERPRETATION
An automatic restriction at this threshold is an appeal-risk decision, not merely a high score, so its lower confidence bound matters more than its point estimate.
```

Precision changes materially when prevalence changes even if TPR and FPR stay constant.

At pi = 0.004 with the same rates, the numerator becomes 0.00168.

The negative contribution becomes 0.996 × 0.00004 = 0.00003984.

The resulting precision is 0.00168 / 0.00171984 = 97.68%.

That 1.15 percentage-point reduction is important when millions of items are scored.

A single threshold should therefore not be justified with a global precision number alone.

The workflow reports precision by language, policy clause, content modality, and score band.

The third calculation asks whether the review tier is physically feasible.

```text
FORMULA
Review capacity R = reviewers × productive_hours/reviewer/day × 3,600 seconds/hour ÷ seconds/item.
Queue admission rate q_max = R ÷ N.

WHERE
R is completed reviews/day; productive hours exclude meetings, breaks, calibration, and system downtime; N is eligible posts/day.

CONSTRAINT
Items must have comparable handling times or capacity must be computed by queue class; arrivals above R require a delay or a rejection rule.

NUMERICAL EXAMPLE
ILLUSTRATIVE: reviewers = 24, productive_hours = 6, and seconds/item = 45.
R = 24 × 6 × 3,600 ÷ 45 = 11,520 reviews/day.
q_max = 11,520 ÷ 40,000,000 = 0.000288.
Expressed as a percentage, q_max = 0.0288% of eligible posts/day.

INTERPRETATION
A review band that admits 0.1% of traffic would send 40,000 items/day, exceed capacity by 28,480 items/day, and cannot meet a 24-hour target.
```

The review calculation assumes every reviewer is interchangeable and every item takes 45 seconds.

That is a deliberate simplification, not an operational promise.

Specialist queues must have their own capacity ledger because rare classes may take several minutes per item.

The boundary is sharp: if arrivals remain above service for multiple days, queue length grows even when each individual reviewer meets the 45-second average.

A daily capacity average can hide hourly overload.

With a peak arrival rate of 1,111.1 posts/second, a 250 millisecond scoring budget requires concurrency beyond a single serialized model worker.

That performance requirement must be measured with production-shaped payloads before an auto-action policy is enabled.

The fourth calculation compares actions using an explicit, policy-owned loss ratio.

Let L_FN be the loss unit for a missed violation within the defined harm window.

Let L_FP be the loss unit for a wrongful restriction that is later restored.

Let c_r be reviewer minutes per item, and let lambda be the stated loss units assigned to one reviewer minute.

For an item with calibrated probability p, the expected loss of publication is p times L_FN.

For an item with calibrated probability p, the expected loss of restriction is (1 - p) times L_FP plus any reversible-operation cost.

An item should only be auto-restricted when the restriction loss is lower and the action is reversible for that policy class.

```text
FORMULA
Choose restrict when (1 - p) × L_FP + c_action < p × L_FN.
Rearranging gives p > (L_FP + c_action) / (L_FN + L_FP), when action cost is expressed in the same loss units.

WHERE
p is a calibrated probability; L_FN and L_FP are policy-approved loss units; c_action is incremental restriction cost in those loss units.

CONSTRAINT
Loss units encode values rather than measured currency, so the threshold is a governance choice; never treat it as a discovered fact.

NUMERICAL EXAMPLE
ILLUSTRATIVE: L_FN = 20 units, L_FP = 1 unit, c_action = 0.05 units.
Threshold = (1 + 0.05) / (20 + 1) = 1.05 / 21 = 0.05.
A score p = 0.06 has publication loss 0.06 × 20 = 1.2 units.
Its restriction loss is 0.94 × 1 + 0.05 = 0.99 units.
Because 0.99 < 1.2, the simple expected-loss rule selects restriction.

INTERPRETATION
Capacity, calibration uncertainty, reversibility, and fairness guardrails can still veto that mathematically preferred action.
```

The loss calculation is intentionally not used as the final production threshold in this case.

A threshold near 0.05 would create an unmanageable volume when prevalence is low and calibration error is concentrated near the decision boundary.

The proper role of the loss calculation is to expose the values hidden inside a claim that “we should be stricter.”

A policy owner can change L_FN or L_FP, but the owner must then approve the resulting action distribution explicitly.

The fifth calculation determines a conservative queue cutoff from an illustrative score-band forecast.

Suppose a 24-hour shadow sample estimates 4,000 items in the 0.90–0.95 band, 9,000 items in the 0.85–0.90 band, and 31,000 items in the 0.80–0.85 band.

The queue has 11,520 daily slots before language coverage reservations.

After reserving 14 × 461 = 6,454 slots for minimum language coverage, 5,066 slots remain for global highest-risk ranking.

All 4,000 items in the 0.90–0.95 band enter the ranked allocation.

Only 1,066 of the next 9,000 items can enter absent a more granular score ranking.

The remaining 7,934 items from that second band receive default publication or a lower-risk deferred sample, depending on the approved experiment design.

This is why bands are operational descriptions, while exact priority must still use the continuous calibrated score.

The language reservation is not a claim that equal allocation is fair.

It is a coverage floor designed to surface quality failures that a pure global ranking would hide.

A review plan must separately measure whether the floor changes confirmed-harm capture relative to a purely risk-ranked queue.

A sixth calculation checks the arithmetic of the automatic band against the appeal team workload.

If 1,587 automatic decisions are false positives in the illustrative scenario and 40% of affected authors appeal, then expected appeals are 634.8 per day.

If 55% of those appeals are filed with sufficient information for review, the intake queue is 349.14 reviewable appeals/day.

If an appeal takes 8 minutes, appeal work is 2,793.12 minutes/day, or 46.552 hours/day.

This calculation excludes valid appeals from correct restrictions, which may also consume capacity.

It therefore cannot be used to staff a real appeals team.

Its purpose is to show that automatic enforcement costs do not end at the first decision.

The model decision record needs a stable reason code before the appeal begins.

The reason code should distinguish hard hash match, policy rule, model confidence band, and human adjudication.

A generic “our systems detected a violation” notice is not enough for diagnosis or meaningful appeal.

The analysis also distinguishes action time from label time.

An item can be auto-restricted at 10:00 UTC and receive a blind quality label at 14:00 UTC.

The later label must not be made available to the model feature store at 10:00 UTC in a retrospective training snapshot.

That would leak the future outcome into the decision representation.

For threshold selection, construct a time-respecting table with one row per decision.

Include ingestion timestamp, score timestamp, model version, calibration version, policy version, action, queue rank, final label, and appeal final state.

A post whose label is unsettled is marked immature rather than silently discarded.

The denominator for a mature-label precision report is completed decisions with a final label, not all decisions made that day.

Report label coverage beside every metric because delayed or selective adjudication changes what the metric represents.

Use a fixed evaluation cut-off, such as labels finalized by day 21 after ingestion.

The 21-day lag is ILLUSTRATIVE and must be replaced only by a documented policy and operations window.

A score threshold can be stable while the policy taxonomy changes underneath it.

For that reason, a new policy clause starts in rules-only or review-only mode until enough blind adjudications support calibration evidence.

Historical labels mapped from an old clause require a documented equivalence decision.

If no equivalence exists, the data are not a valid training continuation.

The analysis uses language as an operational partition, not a proxy for protected status.

Per-language monitoring can discover localized degradation, but it cannot prove fairness across people.

Where lawful and appropriate, a qualified governance process must define additional audit strata and protections.

No demographic inference should be created merely to fill a dashboard.

The actionable conclusion from the calculations is conditional.

Enable a high-confidence automatic tier only after lower confidence bounds, reversibility, and appeal capacity satisfy the written contract.

Use the middle tier as a capped priority queue, not as an unlimited “send to human” escape hatch.

Keep low-confidence content published unless deterministic rules or policy exceptions specify otherwise.

## 7. The Finding

The proposed enforcement contract is a three-tier policy engine with deterministic overrides.

Tier zero is a hard rule for confirmed prohibited material where policy and legal owners authorize immediate restriction.

Tier one is an auto-restriction tier restricted to policy classes with reversible enforcement and language-specific calibration evidence.

Tier two is a capped human-review tier ranked by calibrated risk and constrained by explicit service capacity.

Tier three is publication with sampled quality adjudication, not an assertion that the content is compliant.

The finding does not authorize deployment because no measured calibration, latency, or appeal evidence exists in this illustrative case.

It specifies what evidence would make deployment defensible and what conditions would prohibit it.

The auto tier requires a lower confidence bound on precision chosen by the policy owner.

For illustration, a governance review might require a one-sided 95% lower bound of at least 98% for each enabled language-policy pair.

That illustrative criterion means the deployment gate is not met by a 98.5% point estimate with a wide interval.

The review tier requires its forecasted admission volume to stay under capacity at the p95 traffic forecast, not merely on a daily average.

The publication tier requires a random audit sample sufficient to measure harm escape and false-negative discovery over time.

The selected architecture treats rules and model scores as separate evidence sources.

A deterministic hash match does not need to be pushed through the probabilistic threshold ledger.

A contested policy category does not become safe for automation merely because a model is confident.

A language without calibration evidence never enters an automatic model action tier.

The policy engine must resolve precedence deterministically.

First, reject malformed or duplicate decision events and emit an ingestion reason code.

Second, apply legally mandated or confirmed hard-block rules that have their own audit authority.

Third, route protected contested classes to specialist review independent of model score.

Fourth, apply the language-specific automatic threshold if a valid calibration artifact exists.

Fifth, consider the review queue only if admission capacity and service-level risk permit it.

Sixth, publish and, when sampling selects it, create a blind adjudication task.

Every action receives a decision identifier that is immutable even if a later appeal reverses the action.

The reversal is a new event linked to the original identifier rather than an overwrite.

That event model preserves the answer to “what did the system know at the time?”

The finding rejects a single global threshold because it combines different prevalences, calibration quality, and policy ambiguity into one opaque number.

It also rejects unlimited human routing because that replaces a model error with an unbounded queue delay.

It rejects using reviewer agreement alone as validation because reviewers may share a policy ambiguity or see score cues.

The contract uses a no-action outcome when required evidence is missing.

For example, a newly added language can be published under existing non-model rules while a blind-label program accumulates calibration data.

That is a conscious safety choice, not a silent fallback.

The resulting governance artifact has five version references on every decision.

The first reference is policy version, defining prohibited behavior.

The second is model version, defining score generation.

The third is calibration version, defining score-to-probability mapping.

The fourth is threshold version, defining the action boundary.

The fifth is queue-policy version, defining admission and ordering under capacity.

A sixth optional reference is feature-schema version when the score depends on evolving metadata.

A production record that lacks any mandatory reference is invalid for automated enforcement.

The observed action mix should be compared with the forecasted action mix by language and policy clause.

A large deviation can mean prevalence shift, score serving drift, a malformed routing rule, or a missing language label.

The first response is to hold the affected automation scope, not to tune all thresholds globally.

The conclusion is intentionally narrow: adopt the contract and collect its evidence, rather than claiming an improvement already occurred.
## 8. System/Solution Design

The design separates content transport, evidence generation, policy choice, and human adjudication.

That separation prevents a model deployment from silently redefining policy.

### 8.1 Decision lifecycle

At ingestion, the gateway assigns an immutable post identifier and records the event-time timestamp.

The gateway deduplicates client retries before any score or queue capacity is consumed.

The content service stores the payload under the platform's approved retention controls.

A moderation event contains a pointer or authorized retrieval token rather than copying content into every downstream log.

The language resolver returns a declared or detected language plus a confidence and resolver version.

Low-confidence language resolution does not silently select a language-specific threshold.

Instead, it routes through a documented unknown-language policy path.

The feature service creates an as-of snapshot keyed by decision time.

It may use account age bucket and prior final enforcement count only if those values were available before the current decision.

It must not read future reports, future appeals, or enforcement outcomes caused by the current request.

The scoring service accepts the immutable content reference and feature snapshot identifier.

It returns a raw score, model version, feature-schema version, and inference status.

The calibration service maps the raw score to a probability using the selected language and calibration artifact.

It returns either a valid calibrated probability or an explicit unavailable status.

The policy engine owns action selection.

It does not re-implement feature extraction or model scoring.

Its request includes policy version, policy class candidates, calibrated probability, language, and system health state.

Its response contains one action, reason codes, tier, deadline if queued, and all version identifiers.

The enforcement service performs the reversible restriction or records publication.

It writes an idempotent action event so retrying a failed message cannot create duplicate restrictions.

The queue service accepts only review-eligible items and stores priority, reservation bucket, and expiration policy.

The reviewer workstation retrieves the minimum necessary context and displays policy text without model probability by default for blinded quality samples.

The adjudication service records the reviewer decision, policy clause, evidence category, confidence, timestamp, and escalation reason.

The appeal service creates a linked case and exposes the original action reason at an appropriate detail level.

The audit ledger receives append-only events from every transition.

### 8.2 Contract between services

The input contract is strict because moderation systems often fail at ambiguous boundaries.

`decision_id` is a UUID identifying one attempted action for one post and policy version.

`post_id` identifies the content object but does not replace `decision_id` after an edit.

`event_time_utc` is the source ingestion time in UTC, while `processing_time_utc` is when a service handled the event.

`policy_version` names a reviewed taxonomy and must be present before model action.

`language_code` carries an IETF-style language tag or an explicit `undetermined` value.

`language_confidence` is a numeric diagnostic and cannot be treated as probability of policy violation.

`raw_score` is a model output with range and meaning documented by its model artifact.

`calibrated_probability` is nullable; null means no calibrated policy decision is permitted.

`model_version`, `calibration_version`, and `threshold_version` are immutable strings chosen at decision time.

`action` is one of publish, queue_review, restrict, escalate, hold, or reject_event.

`reason_codes` are a nonempty ordered list because more than one rule may influence a decision.

`reversibility_class` records whether restoration is supported and the maximum restoration delay.

`audit_retention_class` determines the permitted retention treatment, not a blanket content-retention promise.

The output contract forbids an action of `restrict` with an absent policy clause.

The output contract forbids a model-based `restrict` with absent calibration and threshold versions.

The output contract forbids `queue_review` without a queue class and expected deadline.

The output contract forbids an event from changing prior version identifiers.

Schema validation occurs before action selection.

An invalid schema becomes `reject_event` or `hold` according to the failure safety policy.

A malformed event must never fall through to unrestricted publication merely because parsing failed.

### 8.3 Policy configuration

Policy configuration is declarative, reviewable, and separately deployed from model binaries.

A configuration entry has a policy class, eligible languages, action tier, threshold, review priority function, and expiry behavior.

Each entry also states an owner, approval date, evidence link, and rollback target.

A change review compares the proposed action distribution against the current action distribution.

A threshold change without a linked calibration version is rejected at configuration validation.

A language addition starts disabled for automatic action until its validation gate is passed.

A policy class marked specialist-only ignores generic model thresholds.

A hard rule has a source authority and test corpus reference rather than a numeric model threshold.

The configuration compiler emits a signed artifact with a content hash.

The running policy engine accepts only a signed artifact from the approved configuration publisher.

This design protects against accidental edits and gives an auditor a stable decision reference.

The smallest safe rollback unit is one policy-class by language pair.

Global rollback remains available for platform-wide serving or audit failures.

### 8.4 Queue allocation

The review queue is a capacity allocator, not a storage bucket for uncertain content.

The allocator begins each day with a capacity plan derived from active reviewer shifts and class-specific handling-time estimates.

It reserves slots for policy classes and languages only when the reservation has a stated monitoring purpose.

Unused reserved capacity returns to the global risk-ranked pool at a published cut-off time.

Priority is a function of calibrated risk, expected exposure while waiting, reversibility, and policy severity.

The priority function must not include an opaque user attribute with no policy justification.

Ties use oldest event time after the defined priority values.

The queue records both assigned rank and realized service start time.

This permits diagnosis of whether delay arose from ranking, staffing, or workstation availability.

A queue item expires only through a documented policy transition.

Expiry cannot silently delete the item from measurement.

For a low-severity class, expiry may produce publication plus sampled audit.

For a high-severity class, expiry may escalate to a specialist or temporary hold if the policy permits it.

The action after expiry must be recorded as a new event.

The system should forecast next-day queue volume from recent traffic but must retain a conservative hard admission cap.

Forecasts guide staffing; they do not make excess queue volume disappear.

### 8.5 Human decision design

Reviewers require policy context, accessible content rendering, and a way to state uncertainty.

The user interface shows the active policy clause and version before the final action controls.

For operational review, the interface may show the routing reason when it helps triage.

For independent quality evaluation, reviewers must not see model score, rank, or prior reviewer outcome.

The interface includes `compliant`, `violation`, `policy unclear`, and `escalate` outcomes.

Forcing uncertainty into compliant or violation corrupts labels and makes model error appear larger or smaller than it is.

The interface logs policy excerpt version and tool latency so disagreement can be linked to interface or policy changes.

A reviewer can flag missing context, corrupted rendering, or suspected adversarial manipulation.

Those flags feed an incident queue separate from content enforcement.

Specialist escalation has an owner and service objective rather than an unspecified mailbox.

Quality calibration sessions use double-blind samples and record agreement by policy clause.

Agreement is evidence about label consistency, not evidence that the policy itself is socially correct.

### 8.6 Appeals and restoration path

An appeal begins with the original immutable decision identifier.

The appellant receives the applicable policy clause, action date, and a channel to provide relevant context where policy allows.

The appeal path must be usable without revealing model weights, abuse-detection rules, or another user's private data.

A fresh reviewer or designated appeal role reviews the case under the policy version that governed the original action.

If policy changed after enforcement, the system records both the historical rule and any current remedy rule.

An overturned restriction triggers restoration if restoration is technically and policy-wise possible.

The restoration event includes elapsed time from restriction to restoration.

An upheld decision is not evidence that the initial model was correct if a human independently supplied the basis.

Appeal outcomes feed monitoring only after accounting for selection bias.

People who appeal are not a random sample of restricted authors.

The system therefore combines appeal outcomes with blinded random audits.

### 8.7 Security and privacy controls

Content payload access is least-privilege and purpose-bound.

Audit logs store identifiers and reason metadata where possible instead of duplicating raw content.

Reviewer access is logged and periodically reconciled against active role assignment.

Service-to-service calls use authenticated identities and authorization scoped to each data purpose.

The model-serving service cannot read the appeal narrative unless that feature was explicitly approved and available at decision time.

Training exports must exclude direct identifiers unless a documented, approved process requires them.

Feature stores retain time-travel history only for the minimum period needed for reproducibility and appeal obligations.

Deletion or retention requests require a policy-aware process so that legal preservation and appeal fairness are not accidentally broken.

Secrets, signed configuration keys, and audit-store credentials rotate under the platform's existing secrets process.

An attacker who can alter threshold configuration can cause harm without changing model weights, so configuration access is production-critical.

### 8.8 Availability and degraded modes

If the model server is unavailable, deterministic hard rules continue when their dependencies are healthy.

For policy classes that require model evidence, the policy engine selects a documented degraded mode.

A low-severity, reversible class may publish with increased random audit sampling.

A severe class may queue or hold only if the policy grants that authority and capacity can support it.

A degraded mode must never be invented ad hoc during an incident.

If calibration lookup fails, automatic model restriction is disabled for the affected language-policy pair.

If the audit ledger is unavailable, model action is held because explainability evidence cannot be reconstructed later.

If the queue database is unavailable, the system does not repeatedly enqueue the same item.

It writes an idempotent pending state and applies the class-specific fallback.

If a downstream enforcement call times out, the coordinator queries idempotency status before retrying.

This avoids a race in which a post is restored or restricted twice.

### 8.9 Operational telemetry

Every service emits counts by policy version, language, action, and result code.

The score service emits latency distributions and error rates by model version.

The calibration service emits unavailable-artifact counts and score-range diagnostics.

The policy engine emits action-mix counts and configuration artifact hash.

The queue emits arrivals, admissions, abandonments, age percentiles, and completed-review counts.

The reviewer tool emits render failures, handling-time percentiles, and policy-unclear selections.

The appeal service emits filings, resolution time, outcome, and restoration completion time.

The audit pipeline emits event lag and schema-validation failures.

Monitoring dimensions must have a minimum-count rule to avoid drawing conclusions from unstable slices.

A dashboard showing a rate without its denominator and maturity coverage is rejected from decision review.

### 8.10 Rollout plan

Phase one is offline replay using mature historical labels and a frozen policy mapping.

Phase two is shadow scoring with no action, comparing score distributions and schema completeness.

Phase three is review-only routing with a fixed capacity cap and blind quality sample.

Phase four enables a narrowly scoped automatic tier for one reversible policy class only after gates pass.

Phase five expands only by language-policy pair, with each expansion generating a new evidence record.

A rollout has no pre-declared success verdict because results are not available in this case.

Before every phase, operators rehearse rollback, restoration, and audit lookup on non-production fixtures.

After every phase, the review group checks the evidence window rather than reacting to a single day's spike.

A material safety trigger pauses only the affected scope first, preserving unrelated deterministic protections.

## 9. Evaluation & Validation

Evaluation starts by naming the decision population.

The primary population is eligible posts ingested under a fixed policy version during a mature-label evaluation window.

The numerator for auto-tier precision is final-label violations among automatically restricted posts.

The denominator for auto-tier precision is automatically restricted posts with a final quality label.

The numerator for harm escape is final-label violations that remained visible beyond the stated harm window.

The denominator for harm escape is all eligible posts with a final quality label, reported per million posts.

The numerator for wrongful restriction is final-label compliant posts that were restricted by the system.

The denominator for wrongful restriction is all final-label compliant eligible posts, with a separate count per day.

The review-service numerator is queued items completed within the stated deadline.

The review-service denominator is all queued items due within the reporting window, including those still open at the cut-off.

The comparator is the current rules-plus-FIFO-review baseline evaluated on the same population and label protocol.

A pre/post comparison without overlapping traffic allocation is descriptive only because content mix can change.

The preferred early experiment is randomized assignment among candidates eligible for the review tier, if policy and safety owners approve.

Automatic restriction experiments may use a staggered, reversible rollout rather than randomized exposure when a no-action control is unsafe.

Every metric report states executed, not executed, or planned.

In this manuscript all outcome metrics are planned and no experiment has been executed.

### 9.1 Offline validity

Split data by time so the final test cohort occurs after training and calibration fitting.

Split by author or near-duplicate cluster where feasible to reduce memorization leakage.

Fit preprocessing and calibration within each training fold only.

Keep the final test window untouched until the threshold candidates and slice plan are fixed.

Evaluate with final labels after the appeal maturation window, plus an explicitly marked preliminary view for faster monitoring.

Report coverage: final labels divided by decisions in the eligible cohort.

If coverage differs by action tier, a naïve precision comparison can be biased.

Use a random audit sample of published content to estimate false negatives.

Use a random audit sample of restricted content to estimate false positives independently of appeals.

Use score-stratified labels only for curve estimation and weight them back to the target population when estimating population metrics.

### 9.2 Statistical uncertainty

Point estimates are insufficient for rare but consequential error rates.

For each precision and false-positive estimate, report a confidence interval or credible interval with method and assumptions.

For a simple binomial interval, state the number of labeled decisions and the count of events.

Do not present zero observed failures as proof of zero failure probability.

A slice with six reviewed items can be useful for investigation but cannot clear an automatic enforcement gate.

Define minimum labeled counts by policy class and language before looking at model performance.

Use sequential monitoring carefully because repeatedly peeking at a fixed-significance interval increases false alarms.

For operations, pair statistical evidence with practical thresholds such as maximum daily wrongful restrictions.

### 9.3 Calibration checks

Bucket calibrated probabilities by language and action band.

For each bucket, compare mean predicted probability with observed final-label frequency on blind-labeled, mature cases.

Report expected calibration error with bin definitions and sample counts, but do not let a single aggregate number hide a dangerous high-score bucket.

Inspect reliability plots at the automatic threshold because global calibration can coexist with local miscalibration.

Recompute calibration after a policy-labeling guideline change even if model weights remain unchanged.

Calibration evidence expires when a new language, content modality, or material traffic shift changes the target population.

A recalibration artifact must be evaluated on data not used to fit it.

### 9.4 Fairness and safety validation

Safety review compares action rates, quality-label outcomes, appeal outcomes, and time-to-restoration across justified audit strata.

Language is one operational stratum because it affects calibration and review availability.

Differences are hypotheses requiring diagnosis, not automatic proof of unfairness or compliance.

Potential causes include prevalence variation, translation quality, annotation guidance, sample size, score calibration, and policy ambiguity.

A detected gap triggers a review of error counts and content examples under approved access controls.

The remedy may be data collection, a threshold change, specialist routing, policy clarification, or suspension of automation.

Never “correct” a disparity by changing labels after the fact.

A safety test includes adversarial examples such as obfuscation, code-switching, quoted harmful language, satire, and counter-speech.

It also includes accessibility formats and media-description missingness when those inputs affect routing.

The validation dataset records whether an example is synthetic, red-team generated, or drawn from approved historical data.

### 9.5 Guardrail table

| Guardrail | Numerator | Denominator | Window | Trigger | Status |
|---|---|---|---|---|---|
| Auto-tier precision | Final violations among auto restrictions | Auto restrictions with mature blind labels | Weekly mature cohort | Lower bound below policy gate | Planned |
| Wrongful restriction count | Final compliant auto restrictions | Calendar day | Daily | Above approved daily ceiling | Planned |
| Review timeliness | Reviews completed by deadline | Reviews due in cohort | Rolling seven days | Below service objective | Planned |
| Appeal overturn rate | Overturned appeal decisions | Resolved appeals | Monthly | Investigate, not sole quality proof | Planned |
| Calibration deviation | Absolute observed minus predicted rate | Labeled score bucket | Weekly mature cohort | Above approved band | Planned |
| Language coverage | Languages with adequate audit labels | Enabled language-policy pairs | Monthly | Missing evidence blocks expansion | Planned |
| Audit-log completeness | Decisions with all mandatory version keys | All decisions | Daily | Any missing mandatory field | Planned |

The guardrail table does not claim these triggers were crossed or met.

It defines the evidence that an operating review would need.

### 9.6 Acceptance gates

A new automatic tier passes only when all gates are satisfied for its scoped language-policy pair.

Gate one is a signed policy configuration with a reversible action path.

Gate two is a complete audit record in a staging rehearsal.

Gate three is mature, blind-label calibration evidence sufficient for the predeclared precision bound.

Gate four is a queue and appeal capacity forecast under the conservative traffic scenario.

Gate five is a safety review of intended audit strata and adversarial cases.

Gate six is a successful rollback and restoration rehearsal.

Gate seven is approval by the owners named in the decision contract.

A failed gate produces a recorded no-launch decision or narrowed scope.

It must not be bypassed by relabeling the rollout as a “test” if it can enforce against users.

### 9.7 Evaluation limitations

Final labels can be imperfect even with blinded review because policy language may be ambiguous.

Appeal outcomes are selected and may reflect who has time or knowledge to appeal.

Random audit samples can be expensive and slow for rare severe classes.

Synthetic adversarial tests have unknown correspondence to live abuse behavior.

A model may be calibrated overall but become wrong after a coordinated behavior shift.

The evaluation plan therefore supports bounded decisions; it does not certify that moderation is error-free.
## 10. Action Taken

No production action has been taken in this illustrative case.

The proposed action is to approve an evidence-gated design decision, not to activate a classifier.

The first deliverable is a signed moderation decision contract.

It names the policy owner, ML owner, operations owner, appeal owner, privacy reviewer, and final launch approver.

The contract binds each enabled language-policy pair to a policy version, model version, calibration version, and threshold version.

The second deliverable is a schema-validated audit event and an appeal lookup path.

The third deliverable is a shadow-mode measurement plan with no user-facing enforcement change.

The fourth deliverable is a capacity-limited review queue that can run independently of automatic action.

The fifth deliverable is a reversible, narrowly scoped automatic restriction path only after acceptance gates pass.

### 10.1 Work sequence

First, policy authors define the prohibited behavior, edge cases, and appeal remedy for each candidate class.

Second, data owners create a versioned final-label dataset with maturity flags and documented exclusions.

Third, ML owners train or validate the classifier without future-state leakage and fit calibration separately by language.

Fourth, engineering implements the immutable event schema, signed configuration artifact, and idempotent enforcement call.

Fifth, operations supplies shift-specific review capacity, handling-time distributions, and escalation ownership.

Sixth, the quality team runs blind adjudication on random and score-stratified samples.

Seventh, the decision group reviews gates for a single language-policy pair.

Eighth, the platform runs shadow scoring to verify serving latency, configuration integrity, and action forecasts.

Ninth, the platform runs review-only routing with a hard cap and a restoration rehearsal.

Tenth, if and only if gates hold, the group considers a reversible automatic tier with a documented stop rule.

Each step has a durable artifact because oral agreement is insufficient for an enforcement decision.

### 10.2 Stop and rollback conditions

An affected automatic tier is paused when its audit record is incomplete.

It is paused when the calibration artifact is unavailable or mismatched to the configured language.

It is paused when quality evidence falls below the approved lower confidence bound.

It is paused when a policy mapping defect assigns the wrong clause to decisions.

It is paused when restoration fails for a wrongly restricted staging or live case.

It is paused when a severe incident indicates active model or configuration compromise.

A pause changes future actions; it does not delete historical audit evidence.

Rollback selects the prior signed configuration and records the operator, reason, scope, and time.

A rollback review checks whether queued items need re-ranking or whether prior restrictions need targeted restoration review.

The decision group distinguishes a model pause from a hard-rule pause because their evidence and remedies differ.

### 10.3 Ownership matrix

| Artifact or decision | Accountable owner | Required consultation | Evidence before approval |
|---|---|---|---|
| Policy taxonomy and clause mapping | Policy owner | Legal and specialist reviewers | Versioned policy text and edge-case decisions |
| Label protocol and maturity definition | Quality lead | Policy owner and data owner | Blind-adjudication guide and sample audit |
| Model and calibration artifact | ML platform lead | Data science and policy owner | Time-split evaluation and calibration slices |
| Queue capacity and service target | Operations lead | Workforce planning and policy owner | Shift plan, handling-time data, overflow plan |
| Enforcement schema and idempotency | Platform lead | Security and privacy | Staging trace and restoration rehearsal |
| Threshold configuration | Policy owner | ML and operations leads | Signed gate record and rollback target |
| Appeal experience and restoration | Appeals lead | Policy and platform leads | Case trace from decision to remedy |
| Launch or pause decision | Named decision group | All accountable owners | Current evidence packet and recorded dissent |

### 10.4 Evidence packet

The launch packet includes the exact policy text and effective date.

It includes the decision population, exclusions, label maturity window, and sampling scheme.

It includes calibration tables with sample sizes and uncertainty for every enabled scope.

It includes threshold rationale, expected action mix, and capacity forecast under a stated traffic scenario.

It includes fairness and safety review notes with unresolved questions marked as unresolved.

It includes the audit-schema test trace and a restoration test trace.

It includes the rollback configuration identifier and pager ownership.

It includes a statement that values labeled ILLUSTRATIVE in this case are not production evidence.

The evidence packet is versioned and stored with the configuration artifact reference.

## 11. Outcome & Results

No deployment, A/B test, shadow run, or measured company outcome exists for this teaching case.

Accordingly, this section reports expected observables and decision states, not realized results.

The intended near-term outcome is an auditable choice between review-only, narrow automation, or no model action.

The intended operational result is that no queue receives more admitted items than its stated capacity plan.

The intended safety result is that every restriction can be traced to a policy clause and a versioned decision path.

The intended learning result is that an offline metric cannot be mistaken for an enforcement authorization.

### 11.1 Illustrative forecast ledger

| Quantity | ILLUSTRATIVE calculation | Unit | Valid only if |
|---|---|---|---|
| Eligible ingestion | 40,000,000 posts/day | posts/day | The stated traffic assumption holds |
| Policy prevalence | 0.008 × 40,000,000 = 320,000 | violations/day | Final-label prevalence is 0.8% |
| Review capacity | 24 × 6 × 3,600 ÷ 45 = 11,520 | reviews/day | Staffing and handling time match assumptions |
| Auto true positives | 40,000,000 × 0.008 × 0.42 = 134,400 | correct restrictions/day | TPR is 42% in the action population |
| Auto false positives | 40,000,000 × 0.992 × 0.00004 = 1,587.2 | wrongful restrictions/day | FPR is 0.004% in the action population |
| Auto precision | 134,400 ÷ (134,400 + 1,587.2) = 98.83% | probability | Both counts are comparable and mature |
| Minimum language reserve | 14 × 461 = 6,454 | review slots/day | Every language needs the same floor |
| Remaining global slots | 11,520 - 6,454 = 5,066 | review slots/day | Capacity and reservation policy apply |

The table is a forecast ledger rather than a promise.

The most decision-relevant future observation is the lower confidence bound on auto-tier precision, not the illustrative point estimate.

The next most important observation is whether review delay remains within the policy-owned target during peak traffic.

A result that improves harm capture while breaking restoration or appeal access would fail the overall contract.

A result that meets precision overall while failing a materially affected enabled language-policy pair would block expansion for that scope.

### 11.2 Result interpretation rules

If shadow action mix exceeds forecast, inspect schema drift, language fallback, score distribution, and policy configuration before changing a threshold.

If review volume is lower than forecast, do not immediately lower the review threshold; first verify label coverage and traffic mix.

If automatic precision declines, hold the affected scope and test calibration, label change, and serving-model identity.

If appeal overturn rate rises, inspect selection bias and reason-code distribution before declaring a model regression.

If restoration time rises, diagnose enforcement idempotency, content storage, and appeal handoff separately from classifier quality.

If a language has inadequate labels, report unknown rather than borrowing certainty from a global aggregate.

### 11.3 What cannot be concluded

This case cannot conclude that a multilingual model is more accurate than the rules baseline.

It cannot conclude that a 98.83% precision estimate will occur in production.

It cannot conclude that a particular fairness outcome is acceptable.

It cannot conclude that the illustrative review capacity is sufficient for any real organization.

It cannot conclude that policy obligations in any jurisdiction permit this enforcement design.

It can conclude that these questions must be answered in a versioned, capacity-aware contract before automation is enabled.

## 12. Failure Modes & What Went Wrong

The following mechanisms are deliberately distinct so that an operator can discriminate causes instead of merely “monitoring the model.”

### FM-01: Calibration artifact mismatch

Mechanism: the scoring service uses model version M-7 while the policy engine loads a calibration curve fitted for M-6.

Symptom: action volume may look normal while high-score precision degrades in one or more languages.

Discriminating check: join each decision's model version to calibration version and reject pairs absent from the approved compatibility manifest.

Immediate response: disable automatic model action for the mismatched scope and restore the signed compatible configuration.

Prevention: publish model and calibration artifacts atomically with a machine-validated compatibility field.

### FM-02: Policy definition drift masquerades as model drift

Mechanism: reviewers receive revised guidance but the training label taxonomy remains mapped to the prior clause.

Symptom: reviewer disagreement and apparent precision change rise around the guidance effective date.

Discriminating check: stratify labels by adjudication-guide version and compare matched examples before and after the change.

Immediate response: stop comparing those labels as one target and route contested cases to policy review.

Prevention: require a policy-version migration decision before using labels across a policy change.

### FM-03: Queue overload from arrival burst

Mechanism: hourly arrivals exceed available reviewer service even though the daily mean forecast looked feasible.

Symptom: p95 queue age rises while reviewers maintain ordinary handling times.

Discriminating check: compare arrivals per queue class by hour with staffed service rate and admitted-volume cap.

Immediate response: enforce the cap, apply class-specific expiry rules, and add on-call staffing only through the approved plan.

Prevention: capacity-plan using peak scenarios and expose backlog age, not just daily completions.

### FM-04: Selective-label bias

Mechanism: only posts selected by prior rules or high scores receive human labels.

Symptom: offline precision appears excellent but random audit finds unknown or lower-quality behavior outside the selected cohort.

Discriminating check: compare label probability by score band and estimate prevalence from a random ingestion sample.

Immediate response: suspend claims about population false-negative rates and fund blind random sampling.

Prevention: maintain separate random and score-stratified adjudication designs with their distinct estimands.

### FM-05: Feature time-travel leakage

Mechanism: training uses enforcement-history features recomputed after the current post's outcome.

Symptom: offline validation is unusually strong but shadow scoring cannot reproduce feature values.

Discriminating check: replay feature snapshots using only records with timestamps earlier than each decision event.

Immediate response: invalidate the leaked evaluation and rebuild data with as-of joins.

Prevention: enforce point-in-time feature-store queries and test temporal invariants in the pipeline.

### FM-06: Language fallback silently broadens automation

Mechanism: missing language tags are routed through a global calibration curve that was approved only for a subset of languages.

Symptom: automatic actions grow in `undetermined` or low-confidence language categories.

Discriminating check: chart action counts and calibration-availability status by language resolver outcome.

Immediate response: disable automatic action for fallback language states and investigate resolver regression.

Prevention: make unknown language a first-class configuration state with review-only or publish behavior.

### FM-07: Duplicate event creates conflicting enforcement

Mechanism: a retry after a timeout produces a second decision event without idempotency protection.

Symptom: the same post has multiple restrictions, restoration attempts, or reviewer tasks.

Discriminating check: count distinct decision identifiers and actions per post-policy-version pair over a short time interval.

Immediate response: deduplicate pending work, reconcile final state, and preserve duplicate audit events for incident analysis.

Prevention: use idempotency keys at the coordinator and downstream enforcement boundary.

### FM-08: Appeal channel becomes the only quality signal

Mechanism: leaders treat appeal overturn rate as the false-positive rate.

Symptom: a low appeal rate is celebrated even when affected people face access barriers or do not know how to appeal.

Discriminating check: compare blind audit results with appeal outcomes and measure appeal eligibility and completion funnel.

Immediate response: remove appeal rate as a standalone launch gate and retain it as a diagnostic slice.

Prevention: require independent random audits for both restricted and published content.

### FM-09: Adversarial text changes score semantics

Mechanism: users add spacing, transliteration, images, or quoted counter-speech to evade a text-trained classifier.

Symptom: a stable aggregate score distribution conceals rising misses in a specific abuse pattern.

Discriminating check: run reviewed challenge sets and inspect shifts in error taxonomy, not just aggregate AUC.

Immediate response: route affected pattern to specialist review or deterministic rules if policy authorizes them.

Prevention: maintain a controlled red-team corpus and include new confirmed evasion patterns in a governed data process.

### FM-10: Audit log loss prevents explanation

Mechanism: asynchronous logging fails after enforcement succeeds, leaving an action with no recoverable policy or threshold reference.

Symptom: appeals cannot reconstruct why a decision occurred despite an existing restriction.

Discriminating check: reconcile enforcement events against audit-ledger events by decision identifier with a zero-tolerance missing-field alert.

Immediate response: halt model-based enforcement until logging durability is restored and investigate affected decisions.

Prevention: make audit acknowledgement part of the action transaction or use a durable outbox with replay verification.

### FM-11: Aggregated fairness metric hides a localized failure

Mechanism: strong performance in high-volume languages dominates a global metric while a low-volume language has unsafe false positives.

Symptom: global precision clears a gate, but slice confidence intervals or qualitative review reveal a severe exception.

Discriminating check: review per-scope counts, uncertainty, action rates, and error examples before aggregation.

Immediate response: pause the affected language-policy pair without changing unrelated validated scopes.

Prevention: make expansion gates scope-specific and define a minimum evidence threshold for every enabled scope.

### FM-12: Restoration path is theoretically reversible but operationally blocked

Mechanism: content metadata or media references expire before an overturned appeal can restore the post.

Symptom: appeal is granted but the user sees no restored content or a broken reference.

Discriminating check: run end-to-end restoration drills with records near each retention boundary.

Immediate response: manually remediate affected cases and suspend auto-action for classes without verified restoration.

Prevention: align retention policy, storage lifecycle, and appeal service objective in a joint design review.

The common recovery principle is scope containment.

Pause the smallest affected automation scope, preserve evidence, and diagnose the violated contract before broad threshold changes.

## 13. Transferable Lesson

A classifier score is evidence, not a decision.

The decision requires a policy definition, a reversible action path, a capacity allocation, an audit record, and a quality measurement plan.

That principle transfers to fraud holds, medical triage queues, marketplace integrity, credit operations, and safety alerting.

In each domain, the model's ranking may be useful while the action threshold belongs to accountable owners.

Capacity is a first-class model input whenever humans resolve the ambiguous middle band.

If a queue is finite, its admission rule is part of the product and safety design.

Calibration is not an optional chart when scores authorize differentiated treatment.

A globally good model can be locally unsafe where prevalence, language, process, or policy differs.

Versioning is a behavioral requirement rather than paperwork.

Without policy, model, calibration, and threshold references, neither appeal nor learning can reconstruct the decision.

Random audits protect against selective-label bias because user reports and appeals reveal only selected failures.

Reversibility changes the acceptable automation boundary but does not eliminate the need for error control.

A well-designed no-launch decision is an outcome of responsible system design, not an absence of progress.

## 14. Connections & Cross-References

This case connects to calibration theory because expected-loss decisions require probabilities with defensible local meaning.

It connects to queueing theory because review delay depends on arrival bursts and service capacity, not merely average daily volume.

It connects to causal evaluation because action-based labels and appeal outcomes are selected rather than random observations.

It connects to data engineering because point-in-time feature correctness determines whether offline evidence can reproduce serving behavior.

It connects to privacy engineering because moderation evidence, reviewer access, and retention choices constrain what can be measured.

It connects to incident management because configuration, audit, and restoration failures need scoped rollback playbooks.

It connects to human-computer interaction because reviewer interfaces shape label quality and appeal access shapes observed outcomes.

It connects to governance because policy owners, not model metrics, authorize which error trade-offs are acceptable.

### Interview pack

#### Question 1: Why is AUC insufficient for choosing an automatic restriction threshold?

**Answer:** AUC averages ranking quality across thresholds and omits prevalence, capacity, reversibility, and the cost of false restriction.

**Reasoning:** Enforcement consumes one narrow operating region, where precision and false-positive counts matter at the actual traffic volume.

**Follow-up:** Ask the candidate to name the action population and the relevant guardrail denominator.

**Red flag:** Treating a higher AUC as a deployment authorization.

**Scoring:** Strong answers distinguish ranking discrimination from a policy-owned action decision.

#### Question 2: Compute daily review capacity for 18 reviewers, 5.5 productive hours each, and 55 seconds per item.

**Answer:** ILLUSTRATIVE capacity is 18 × 5.5 × 3,600 ÷ 55 = 6,480 reviews/day.

**Reasoning:** Productive hours convert to 356,400 reviewer-seconds; dividing by 55 seconds/item gives 6,480 items.

**Follow-up:** Ask how the answer changes if specialist items take 110 seconds.

**Red flag:** Dividing by minutes without converting units or treating scheduled hours as productive hours.

**Scoring:** Full credit requires units and an explicit assumption about homogeneous handling time.

#### Question 3: A model has TPR 0.50, FPR 0.001, and prevalence 0.01. What is precision?

**Answer:** ILLUSTRATIVE precision is 0.01×0.50 ÷ [0.01×0.50 + 0.99×0.001] = 0.005 ÷ 0.00599 = 83.47%.

**Reasoning:** The numerator is the positive joint probability; the second denominator term is the false-positive joint probability.

**Follow-up:** Ask what happens to precision if prevalence halves while rates remain fixed.

**Red flag:** Calling TPR itself precision or forgetting the negative-population term.

**Scoring:** Full credit requires the Bayes calculation and recognition that prevalence matters.

#### Question 4: A high-score tier has 99% point precision from 100 blind labels. Is it ready for automation?

**Answer:** Not necessarily; one error in 100 has a wide uncertainty interval and may not meet a predeclared lower-bound gate.

**Reasoning:** The evidence must be scoped by language and policy class, account for reversibility, and meet sample-size requirements.

**Follow-up:** Ask which additional evidence should be collected without leaking model score to reviewers.

**Red flag:** Launching from a point estimate without counting labels or defining the action scope.

**Scoring:** Strong answers ask for uncertainty, label independence, and a written threshold criterion.

#### Question 5: Review backlog grows while average handling time is unchanged. Diagnose the likely subsystem.

**Answer:** First inspect hourly arrival rate, admission cap, reviewer availability, and queue-class routing rather than model quality alone.

**Reasoning:** Stable service time with rising age suggests arrivals or capacity allocation exceed service during part of the window.

**Follow-up:** Ask for a metric that distinguishes burst arrival from absent reviewers.

**Red flag:** Recommending a lower classifier threshold without checking whether it would worsen inflow.

**Scoring:** Full credit identifies queue dynamics and proposes discriminating measurements.

#### Question 6: A new language lacks calibration data but has high raw model scores. What action is permitted?

**Answer:** The contract permits no automatic model restriction for that language; use documented rules, review-only routing, or publication with audit according to policy.

**Reasoning:** Raw scores are not comparable probabilities without approved calibration evidence.

**Follow-up:** Ask how to collect labels that estimate population error rather than only high-score precision.

**Red flag:** Reusing a global threshold because the encoder is multilingual.

**Scoring:** Strong answers separate language support from calibrated enforcement authorization.

#### Question 7: Derive the expected-loss threshold with missed-harm loss 12, false-restriction loss 3, and action cost 0.

**Answer:** ILLUSTRATIVE threshold is 3 ÷ (12 + 3) = 0.20; restrict only when calibrated p exceeds 0.20 under this simple ledger.

**Reasoning:** Set p×12 greater than (1-p)×3 and solve 12p > 3 - 3p, so 15p > 3.

**Follow-up:** Ask why capacity and calibration uncertainty can override the resulting threshold.

**Red flag:** Subtracting losses or claiming the value is learned from the classifier.

**Scoring:** Full credit shows the inequality and labels the values as policy choices.

#### Question 8: Why can appeals not replace random quality audits?

**Answer:** Appeals are self-selected, so they reveal some dissatisfied decisions but not the error rate among all affected people or missed violations.

**Reasoning:** Access, awareness, and expected benefit vary, creating selection bias in both filing and completion.

**Follow-up:** Ask for one random sample from restricted content and one from published content.

**Red flag:** Equating a low overturn rate with low false-positive rate.

**Scoring:** Strong answers name selection bias and a complementary sampling design.

#### Question 9: A policy clause changes wording. Which artifacts need review before retaining the old threshold?

**Answer:** Review label mapping, adjudication guide, training target, calibration evidence, threshold rationale, and appeal explanations.

**Reasoning:** A threshold is defined relative to the meaning of the positive label, which may have changed even if model weights did not.

**Follow-up:** Ask when historical labels may be reused.

**Red flag:** Updating only a displayed policy string.

**Scoring:** Full credit recognizes policy-version drift as a data and governance change.

#### Question 10: How do you test whether history features leak future information?

**Answer:** Replay feature construction at each decision timestamp and verify every source event precedes that timestamp.

**Reasoning:** A feature backfilled after enforcement can encode the label or earlier model action, inflating offline quality.

**Follow-up:** Ask what author-level splitting protects against in addition to time splitting.

**Red flag:** Checking only that train and test files have different row identifiers.

**Scoring:** Strong answers use point-in-time joins and discuss correlated author behavior.

#### Question 11: A queue reserves 461 slots for each of 14 languages from 11,520 total slots. How many global slots remain?

**Answer:** ILLUSTRATIVE remaining capacity is 11,520 - (14 × 461) = 11,520 - 6,454 = 5,066 slots.

**Reasoning:** Reservations consume capacity before global ranking; unused reservations need an explicit release rule.

**Follow-up:** Ask whether equal reservation is always fair or efficient.

**Red flag:** Forgetting that reserved capacity can crowd out higher-risk items.

**Scoring:** Full credit computes correctly and discusses the coverage-versus-capture trade-off.

#### Question 12: What must be in an appeal-ready decision record?

**Answer:** At minimum: decision identifier, policy clause and version, action time, action, model and calibration versions if used, threshold version, reason code, and linked evidence under authorized access.

**Reasoning:** The record must reconstruct the decision context without exposing security-sensitive details unnecessarily.

**Follow-up:** Ask how a later policy change should be represented for an old appeal.

**Red flag:** Keeping only the final action label.

**Scoring:** Strong answers include immutable version references and a restoration linkage.

#### Question 13: An auto-tier precision estimate drops in one language. What is the first safe response?

**Answer:** Pause automatic model action for that language-policy pair, preserve evidence, and test calibration, language resolution, label changes, and serving versions.

**Reasoning:** A local scope pause contains harm while avoiding unsupported changes to unrelated validated scopes.

**Follow-up:** Ask why a global threshold reduction may be the wrong first action.

**Red flag:** Retuning all languages from a single aggregate metric.

**Scoring:** Full credit proposes containment plus discriminating checks.

#### Question 14: Which timestamps are required to evaluate a 24-hour review commitment correctly?

**Answer:** Record ingestion event time, queue admission time, review start time, review completion time, and any appeal or restoration time separately.

**Reasoning:** Processing time and final-label time answer different questions; collapsing them hides backlog and label maturity.

**Follow-up:** Ask how negative duration values can arise.

**Red flag:** Measuring only the date on which a dashboard query ran.

**Scoring:** Strong answers distinguish event time, processing time, and adjudication time.

## 15. Quick Reference Card

| Decision element | Required rule | Evidence or calculation | Stop condition |
|---|---|---|---|
| Unit | One post at one policy version | Immutable post and decision identifiers | Edited content reuses old decision |
| Policy | Action maps to a clause and version | Signed policy configuration | Missing or ambiguous policy clause |
| Score | Raw score is not an action | Language-specific calibration artifact | Calibration unavailable or mismatched |
| Automatic tier | Restrict only within approved scope | Lower confidence bound and restoration rehearsal | Gate failure or audit incompleteness |
| Review tier | Admit only within capacity | Shift capacity and queue-age monitoring | Arrival exceeds capped service plan |
| Publication tier | Publish does not mean compliant | Random blind audit sample | Audit sampling stops without replacement plan |
| Fairness and safety | Inspect justified slices and edge cases | Counts, uncertainty, and qualitative review | Local safety signal lacks diagnosis |
| Appeals | Preserve original action context | Immutable decision record and remedy trace | Restoration cannot be completed |
| Versions | Persist policy, model, calibration, threshold, queue versions | Audit lookup test | Any mandatory version missing |
| Rollback | Pause smallest affected scope first | Signed prior configuration | Broad change made without diagnosis |
| Outcomes | Report only observed results as results | Mature labels and executed status | ILLUSTRATIVE values presented as measured |
