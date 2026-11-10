---
case_id: "CASE-CAPSTONE-002"
plan_row: "PLAN-CAPSTONE-002"
case_title: "Choose from six cross-domain capstones"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Capstone portfolio selection"
primary_pattern: "screen-then-rank selection under effort uncertainty"
secondary_patterns: ["evidence-credit scoring", "minimax versus expected value"]
evidence_level: "illustrative"
source_basis: "illustrative selection exercise over constructed candidate parameters"
as_of_date: "2026-11-10"
owner_role: "capstone programme lead"
source_index: "INDEX.md"
review_trigger: "Any change to the term calendar, the effort estimate, a reuse claim, a tier probability, or the minimum-evidence floor."
---

# Choose From Six Cross-Domain Capstones

**Source and evidence declaration.** This is an ILLUSTRATIVE selection exercise. The six candidates, their nominal effort, their reuse fractions, and their tier probabilities are constructed inputs, not measurements of any real project. Every score, probability, and ratio below is recomputed by hand from those inputs and is labelled ILLUSTRATIVE. No employer, vendor, course cohort, or student outcome is described. The only external anchors are the sibling release-gate case and the unit's evidence rubric, both of which supply vocabulary rather than data.

## 1. The One-Line Framing

> Given a 22-week term with 8 discretionary hours per week, six candidate capstones whose cost and evidential yield are estimated but not measured, and one portfolio slot, the capstone programme lead selects exactly one candidate to carry to a release decision — for an external reviewer, under three explicit screens, against the alternative of deferring the slot for a term — and publishes the resulting ranking with its weakest assumption attached.

The decision is not "which capstone is most interesting". It is "which capstone converts a fixed and uncertain pool of hours into the largest expected quantity of checkable claims without becoming infeasible in a plausible bad week". Those are different questions with different answers, and the gap between them is the substance of this case.

| Element | Statement |
|---|---|
| Selected candidate | C6, policy Q&A with a mandatory abstention rule |
| Decision grain | One candidate locked for one 22-week term, one portfolio slot |
| Decision owner | Capstone programme lead, accountable for the published ranking |
| Challenge owner | Independent reviewer who may reject a screen definition |
| Baseline | Defer the slot: reuse the existing artifact, publish no new claim |
| Runner-up | C5, credit scorecard with prevalence stress, 3.47% below on the ranking metric |
| Rejected on base score | C4, highest raw score, removed by the reuse-tolerance screen |
| Rejected on feasibility | C1, C3 nominally; C2 once its contingency is included |

The selected candidate is not the one with the highest unadjusted score. C4 scores 0.0170794 against C6's 0.0164238, a 3.99% advantage that rests on a reuse credit with one hour of margin. C2 scores 0.0207170 — 26% above C6 — but only while its 70-hour contingency stays unrealised, and that contingency carries an estimated 40% probability. The case is therefore an argument about which numbers to trust, expressed as a ranking.

**What the selection does not claim.** It does not claim that C6 is the best capstone in any absolute sense, that its tier probabilities are accurate to a percentage point, or that a term actually has 22 usable weeks. It claims that under the stated screens and the stated inputs, C6 is the candidate whose ranking is least sensitive to the assumptions most likely to be wrong, and it names the assumption that would reverse the choice.

**Reusable pattern.** Screen-then-rank: apply feasibility and robustness screens that do not depend on the scoring model, then rank only the survivors. The anti-pattern it replaces is rank-then-rationalise, in which a single composite score is computed first and every objection is folded into that score as an adjustment weight, which hides the objection inside a number.

## 2. Context & Background

A learner finishing a structured AI/ML programme typically accumulates several plausible project ideas and has one term in which to produce one portfolio artifact. The constraint is not idea generation; it is that effort, reuse, and evidential yield are all uncertain at the moment of commitment, and the commitment is expensive to reverse because a half-built artifact has little portfolio value.

| Context field | Bounded statement | Evidence status |
|---|---|---|
| Stakeholders | programme lead selects; reviewer challenges the screens; the learner executes | illustrative roles, no named people |
| Planning horizon | 22 weeks, of which 8 hours per week are discretionary | constructed calendar |
| Capacity | 176 discretionary hours for the term | arithmetic from the calendar |
| Baseline | carry the existing artifact forward with new framing only | considered baseline, not executed |
| Alternative | defer the slot one term and re-plan with better estimates | available, explicitly valued at zero |
| Portfolio aim | a claim set that a reviewer can check, across more than one domain family | derived from the rubric |
| Constraint | one slot, because two half-projects yield no checkable claim | design constraint |
| Out of scope | employment outcomes, course grading, publication, team scaling | scope boundary |

The six candidates were not chosen to be equal. They differ deliberately along four axes that interact: domain family (signal, forecasting, agentic, streaming, tabular risk, retrieval), nominal effort (120 to 260 hours), reuse available from prior work (5% to 30%), and the kind of evidence the domain can produce. That last axis matters more than it first appears, because a domain whose evidence is a held-out score has a different tier ladder from a domain whose evidence is a recorded trace.

| Candidate | Domain family | Nominal hours | Reuse share | Evidence shape |
|---|---|---|---|---|
| C1 | Device signal to explanation | 200 | 0.00 | recorded trace plus hardware tier |
| C2 | Retail demand forecasting | 120 | 0.10 | backtest with a time split |
| C3 | Support triage with tool calls | 260 | 0.20 | trajectory replay plus authorisation log |
| C4 | Payment anomaly stream | 180 | 0.25 | alert-budget evaluation on a labelled window |
| C5 | Credit risk scorecard | 150 | 0.30 | discrimination and calibration under prevalence shift |
| C6 | Policy Q&A with abstention | 140 | 0.05 | answer-plus-citation audit with refusal cases |

The six are not equally well matched to a 176-hour term, and two of them are not matched at all. C1 at 200 nominal hours and C3 at 260 exceed the capacity before any reuse is counted, and C3 exceeds it even after its 20% reuse credit. Those two are included in the exercise precisely because a plausible candidate list contains infeasible entries, and a selection process that cannot say "this one does not fit" will silently compress the estimate instead.

**Why six rather than two.** A two-candidate comparison invites a false binary and hides the screening step, because with two options the decision reduces to a single number comparison. Six candidates make the screens observable: three candidates are removed by rules that never look at their scores, and only then does the ranking metric run. That separation is the transferable content, and it is invisible in a pairwise choice.

**Where the hours come from.** The 8-hour week is a planning convention, not an observation. It represents time that survives teaching, work, and ordinary life, and the 0.88 delivery probability attached to each week later in this case is a constructed allowance for the weeks that do not survive. Both numbers are ILLUSTRATIVE and both are the sort of input a real plan would calibrate after two terms of history.

## 3. The Business Question

| Decision contract | Definition |
|---|---|
| Question | Which single candidate should consume the term's 176 discretionary hours? |
| Decision owner | Capstone programme lead, accountable for the published ranking and its caveats |
| Challenge owner | Reviewer, who may reject a screen definition or a floor before ranking begins |
| Grain | One candidate, one term, one portfolio slot, one published ranking |
| Cut-off timestamp | The instant the term plan is frozen; estimates and reuse claims after it belong to the next cycle |
| Trigger | A slot is open and at least two candidates could plausibly fill it |
| Actions | Select one candidate; defer the slot; reject all six and re-scope the list |
| Baseline | Defer: republish the existing artifact, adding no new claim |
| Objective | Maximise expected checkable claims per committed hour, subject to feasibility in a bad term |
| Hard constraint | No candidate whose expected hours exceed capacity may be selected on the strength of a reuse claim alone |
| Soft constraint | Prefer a domain that broadens the portfolio's evidence shapes over one that duplicates an existing shape |
| Out of scope | Grades, hiring outcomes, publication, multi-person coordination |

**Why the baseline needs a number.** "Do nothing" is not free, and the discipline of valuing it prevents selection by default momentum. Here the deferral value is set to zero expected credit: republishing an existing artifact adds no new tier, because the artifact already carries whatever tier it earned. That zero is not a judgement that the existing artifact is worthless; it is a statement that the marginal evidential yield of re-publication is nil. Setting it to zero also makes the selection rule honest, because every candidate must beat zero rather than beat each other.

| Proxy metric | Numerator | Denominator | Why only a proxy |
|---|---|---|---|
| Evidence credit per hour | expected earned tiers | effective hours | assumes tiers are equally valuable, which they are not across domains |
| Release probability | probability the hours suffice | one | treats a late artifact as worthless, which overstates the loss |
| Reuse tolerance | hours of reuse the plan can lose before a week-count step | planned reuse credit | discretises a smooth risk into one step |
| Portfolio breadth | distinct evidence shapes covered | slots used | ignores depth within a shape |

The first row is the metric actually used for ranking, and its weakness is stated rather than hidden: a hardware tier and a backtest tier are counted equally even though they cost different amounts and persuade different readers. Weighting them differently would be more accurate and less defensible, because the weights would be constructed too, and a second layer of constructed inputs would make the ranking look precise without making it better. Equal weights with a stated limitation is the more honest choice.

**The decision this case is not making.** It is not choosing the topic the learner finds most interesting, and it is not choosing the candidate with the largest nominal scope. A reader who finishes this case and concludes "pick the most interesting one anyway" has understood the arithmetic and rejected the frame, which is a legitimate position, but the rejection should be explicit rather than accomplished by quietly ignoring the screens.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Check | Status | If wrong | Fallback / owner |
|---|---|---|---|---|---|---|
| A-01 | A week delivers 8 discretionary hours with probability 0.88, independently across weeks | schedule | compare against two prior terms | OPEN | release probabilities are all wrong in the same direction | recalibrate from history / lead |
| A-02 | Reuse fractions are achievable as stated, with no rewrite of the reused part | effort | inspect the prior artifact for interface drift | OPEN | effective hours rise and the plan may not fit | re-estimate before ranking / lead |
| A-03 | Tier probabilities are elicited from reference classes, not from enthusiasm | evidence | name the reference class for each probability | OPEN | scores are optimistic in a correlated way | require a written basis per number / reviewer |
| A-04 | Tiers are equally valuable inside the ranking metric | scoring | confirm against the portfolio's existing shapes | OPEN | the ranking favours cheap tiers | apply a stated weight vector and re-rank / lead |
| A-05 | The slot cannot be split profitably between two candidates | portfolio | check whether either half-artifact would earn a tier | OPEN | splitting could dominate | re-open as a three-way comparison / lead |
| A-06 | Contingency hours, where declared, are additive to nominal effort | effort | compare with similar past tasks | OPEN | risk-adjusted scores are too pessimistic | recompute with a partial-credit contingency / lead |
| C-01 | No candidate may be selected whose nominal hours exceed capacity by more than its verified reuse | feasibility, binding | arithmetic on the stated inputs | binding | a selected plan is infeasible by construction | reject the candidate / lead |
| C-02 | The minimum-evidence floor is E ≥ 2.0 earned tiers in expectation | evidence, binding | apply the floor before ranking | binding | de-scoping games the metric | re-scope the artifact class / reviewer |
| C-03 | Ranking uses expected value, not maximin, with screens carrying the robustness burden | method, binding | the screens must be checkable without the score | binding | the ranking becomes a pessimism contest | restate the rule before ranking / reviewer |
| C-04 | Every published number carries its unit and its formula identifier | honesty, binding | audit the final table | binding | a constructed number reads as a measurement | retract the sentence / lead |

A-01 and A-02 are the two assumptions that actually move the answer, and they move it in opposite directions. Optimism about weeks raises every release probability, which compresses the differences between candidates because most of them saturate near one. Optimism about reuse raises effective hours, which lowers the probabilities and *widens* the differences, because the candidates then straddle the week-count steps. The net effect is not a shift but a change in which candidate wins, which is why both are screened rather than averaged.

C-03 deserves a note because it is a methodological constraint that a reasonable person could reject. Maximin ranking — maximise the worst scenario — has a real claim on this problem, since a failed term is costly. It was rejected because the worst scenario here was constructed to be extreme (a 50% loss of reuse credit) and assigning it the same standing as the base case turns the exercise into a pessimism contest. The compromise adopted is to treat the extreme scenario as a *screen* that removes candidates which fall off a cliff, and to keep expected value for the ranking of survivors. That way the extreme case has power without dominating.

| Constraint deliberately absent | Why | What its absence costs |
|---|---|---|
| A minimum number of candidates | the list is given, not generated | the case cannot comment on idea sourcing |
| A required domain | breadth is preferred, not mandated | a portfolio of one shape remains permissible |
| A maximum score threshold | selection is comparative, not absolute | a weak field could still yield a selection |
| A learner-hours quality factor | no measurement supports one | a slower learner is modelled as a wider week distribution only |

The fourth absence is worth defending. Adjusting capacity for individual pace would be the single most useful input and the least defensible, because no observation supports a multiplier. Modelling it instead as a lower delivery probability per week keeps the arithmetic honest: the same formula absorbs the variation, and the assumption is visible as a probability rather than buried as a coefficient.

## 5. Data, Tools & Method

### 5a. Data Specification

The selection table is the entire dataset. Every row is a candidate; every column is an estimate with a named basis. The critical discipline is that the basis is recorded beside the number, because a bare estimate cannot be audited and cannot be revised coherently.

| Field | Type / unit | Admission rule | Recorded evidence |
|---|---|---|---|
| `candidate_id` | string, C1–C6 | unique within the cycle | exact value |
| `domain_family` | enumerated string | one of the six declared families | exact value |
| `nominal_hours` | decimal hours | positive; the estimate before reuse | value and estimation basis |
| `reuse_share` | decimal in [0, 0.6] | claimed only with a named prior artifact | value plus artifact identifier |
| `effort_hours` | decimal hours | equals `nominal_hours × (1 − reuse_share)` | recomputed, never copied |
| `contingency_hours` | decimal hours, may be zero | declared per candidate, additive | value plus the risk it covers |
| `contingency_prob` | decimal in [0, 1] | required when `contingency_hours > 0` | value plus basis |
| `p_design` | decimal in [0, 1] | probability of earning the design tier | reference class named |
| `p_trace` | decimal in [0, 1] | probability of the trace tier given the design tier | reference class named |
| `p_hardware` | decimal in [0, 1] | probability of the domain's top tier given both lower tiers | reference class named |
| `evidence_credit` | decimal in [0, 3] | computed, never estimated directly | formula identifier |
| `release_prob` | decimal in [0, 1] | computed from the week model | formula identifier |
| `density` | decimal, credit per hour | computed; zero when infeasible | formula identifier |

**Grain.** One row per candidate per selection cycle. A candidate re-entered in a later cycle is a new row, because its reuse share and tier probabilities have changed and the old row must remain readable as evidence of what was believed then.

**Cut-off.** The freeze instant is when the plan is committed. Any re-estimate after that instant is recorded as a change event with a timestamp rather than as an edit, because an estimate revised after the outcome is known is not an estimate.

**Population.** Six constructed candidates. There is no sampling frame and no claim that six is a representative number of ideas; the six exist to make the screens observable, not to estimate how often a screen fires in general.

**Keys.** `candidate_id` identifies a row within a cycle; the cycle identifier plus `candidate_id` identifies it across cycles. The published ranking must be keyed the same way, or a later reader cannot tell whether "C4 ranked first" refers to this cycle or the next.

**Units.** Hours for effort throughout; percentages appear only in text and never inside a formula, so that a reader cannot accidentally multiply hours by 25 instead of 0.25. Credit is measured in tiers, an expected count in [0, 3], and density is tiers per hour with units written as tier/h.

**Not available at decision time.** Actual hours consumed by any comparable prior project; the true reuse achievable on each prior artifact; any observation of a learner completing a term at this intensity; and the reviewer's eventual reading of the published artifact. Each absence is recorded here so that no later section can quietly treat an estimate as a measurement.

### 5b. Tools & Stack

The exercise needs no software beyond a spreadsheet, and saying so is part of the point: the failure mode this case addresses is not computational, it is choosing which quantity to maximise.

| Element | Role | Deliberately not claimed | Owner |
|---|---|---|---|
| selection table | the only dataset; one row per candidate | any link to a project-management tool | lead |
| arithmetic check | independent recomputation of every derived column | that a spreadsheet is reproducible across versions | reviewer |
| screen checklist | feasibility and tolerance rules applied without scores | that the screens are complete | reviewer |
| sensitivity runner | one-at-a-time variation of weeks, reuse, and tier probabilities | that interactions are exhausted | lead |
| ranking record | the ordered survivors plus the rejected candidates and reasons | that the ranking is stable under a different weight vector | lead |
| publication note | the selected candidate, the runner-up, and the weakest assumption | any outcome claim about the artifact | lead |

The reviewer's arithmetic check is listed as a separate element because it is the cheapest available control against a whole class of error: a derived column that disagrees with its own formula. In practice the disagreements are not subtle, they are transcription errors between the estimate and the formula, and they survive review only when the reviewer re-solves rather than reads.

### 5c. Method Summary

`estimate nominal hours -> subtract verified reuse -> add declared contingency at its probability -> compute evidence credit -> compute release probability from the week model -> compute density -> apply feasibility screen -> apply tolerance screen -> rank survivors by expected density -> publish with the weakest assumption`

| Candidate approach | What it optimises | Strength | Limit | Verdict |
|---|---|---|---|---|
| Pick by interest | motivation and stamina | real, and non-trivial over 22 weeks | no arithmetic, no audit trail | rejected as the primary rule |
| Pick the largest scope that fits | ambition | produces an impressive artifact if it lands | ignores the probability of landing | rejected |
| Composite weighted score only | one comparable number | simple to communicate | hides screening inside weights | rejected |
| Screen then rank | feasibility and robustness first | objections stay visible and checkable | more steps to explain | adopted |
| Maximin ranking | worst case | protects a fragile term | extreme scenarios dominate | rejected, then re-used as a screen |

The row that matters is the fourth, and its cost is real: three rules instead of one number. The compensation is that each rule can be argued separately and each rejection has a sentence attached, so a reader who disagrees with the ranking can disagree with a specific screen rather than with an opaque composite.

**The screens, stated without reference to any score.**

| Screen | Rule | What it removes | Why it must not use the score |
|---|---|---|---|
| S1 feasibility | effective hours, including contingency, must not exceed 176 | candidates that cannot fit even in a good term | a high score would tempt a waiver |
| S2 tolerance | reuse credit must have at least 25% headroom before a week-count step | candidates whose plan rests on one hour of slack | the score rises with reuse, so it is biased toward the fragile plan |
| S3 evidence floor | expected credit must reach 2.0 tiers | de-scoped artifacts whose density is high only because they are small | density mechanically rewards shrinkage |

S2's threshold of 25% is a design choice with a stated rationale: a reuse overclaim of one quarter is common in first estimates, so a plan that cannot survive a 25% miss is a plan that cannot survive contact with an honest re-estimate. S3's floor of 2.0 tiers is likewise a choice, and its function is precisely to stop the ranking metric from being gamed in the way Section 6's boundary case demonstrates.

### 5d. Provenance, Quality, Exclusions, and Leakage

Every input in the selection table is constructed and every one must be labelled as such. The provenance record therefore looks less like a data lineage and more like an audit of authorship.

| Input class | Author | Basis | Revision rule |
|---|---|---|---|
| nominal hours | programme lead | analogy to two prior artifacts of similar shape | revise only with a written reason and a new timestamp |
| reuse share | lead, challenged by reviewer | inspection of the prior artifact's interfaces | revise if the interface is found to differ |
| tier probabilities | lead, from a named reference class per tier | frequency in the prior five similar artifacts, or an explicit prior when none exist | revise with a new reference class, not by nudging |
| contingency | lead | the specific risk named in the row | revise if the risk is retired or materialises |
| week model | planning convention | 8 hours per week, 0.88 delivery | recalibrate from term history |

The weakest column is tier probability, and pretending otherwise would be the main quality defect available here. There are no five prior artifacts behind these numbers in a constructed exercise, so the reference class is declared rather than counted, and the honest label is "elicited prior". The compensation is sensitivity analysis: instead of defending the probabilities, the case shows how far each would have to move to change the selection, which converts an unfalsifiable input into a bounded one.

**Exclusions.** Excluded from the ranking: candidates whose reuse claim cannot be tied to a named prior artifact; candidates whose top tier would require an unavailable external dependency such as a labelled dataset with unresolved permission; and any candidate re-scored after the freeze without a recorded change event.

**Leakage in a selection exercise.** The analogue of data leakage here is *preference leakage*: the lead forms a favourite and then chooses estimates that make the favourite rank first. The symptom is a table where one candidate's inputs are suspiciously round and its competitors' inputs are suspiciously pessimistic. The control is ordering: all six rows are estimated before any score is computed, and the estimation is signed off by the reviewer, so the ranking cannot be produced until the inputs are frozen. It is a procedural control, which is the only kind available when the "data" is a person's judgement.

### 5e. Reproducibility and Method-Selection Ledger

| ID | Location or identity | Version | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | this manuscript, Section 5a | constructed | input schema | local | inputs are elicited, not observed |
| SRC-02 | `01_local_signal_demo_release_gate.md` | sibling case | tier vocabulary and gate mechanics | local | supplies no effort data |
| SRC-03 | `../Cheat_Sheets/01_Capstone_Evidence_Rubric.md` | rubric | tier definitions and evidence boundary | local | rubric, not measurement |
| CALC-01 | formulas B-01 to B-07 | constructed | every derived column | in this manuscript | arithmetic only, no run |
| SCR-01 | screens S1–S3 | constructed | candidate removal | in this manuscript | thresholds are choices |
| CYC-01 | this selection cycle | 2026-11-10 | the ranking being decided | in this manuscript | no follow-up cycle yet |

A later reader must be able to recompute the ranking from the table alone. That requirement drives three design choices: every derived column names its formula; every formula appears in full rather than as a spreadsheet expression; and the week model is written out as a distribution rather than quoted as a probability. A ranking that cannot be recomputed is not a ranking, it is an assertion with a table attached.

**Method selection against alternatives.** A full Monte Carlo over all inputs jointly would produce a distribution of rankings rather than a point ranking, which is strictly more information. It was not used because the input distributions are themselves constructed, so the output distribution would inherit their arbitrariness while presenting itself as a quantified risk. One-at-a-time sensitivity plus a three-scenario expectation is the honest middle: it shows which inputs matter without implying that their distributions are known.

## 6. Analysis Walkthrough

**Decisive result:** C6 is selected, C5 is the runner-up at 3.47% below, C4 is removed by the tolerance screen despite the highest raw score, and C2 is removed by the feasibility screen once its contingency is counted. Every figure below is ILLUSTRATIVE arithmetic over constructed inputs.

### Capacity and effective effort

```text
FORMULA [B-01]: H_eff = H_nom * (1 - r) ;  slack = C - H_eff ;  N = ceil(H_eff / 8)
WHERE: H_nom [h] nominal effort; r [dimensionless, 0..0.6] reuse share;
       H_eff [h] effective effort; C = 176 [h] term capacity (22 weeks x 8 h);
       N [weeks] the number of week-slots the plan consumes; 8 [h/week].
CONSTRAINT: reuse is counted only against a named prior artifact whose interfaces
            match; H_eff is never rounded down before the ceiling is taken.
NUMERICAL EXAMPLE: C6: H_nom = 140 h; r = 0.05; credit = 140 * 0.05 = 7.0 h;
                   H_eff = 140 - 7.0 = 133.0 h;
                   slack = 176 - 133.0 = 43.0 h;
                   N = ceil(133.0 / 8) = ceil(16.625) = 17 weeks.
INTERPRETATION: C6 consumes 17 of 22 week-slots, leaving 43 hours of calendar
                slack and 3 hours of unused capacity inside its final week.
```

The ceiling is where most of the sensitivity lives. C6's 16.625 weeks rounds to 17; at 16.0 it would round to 16 and gain a whole step of release probability. That discontinuity is a real property of a week-structured calendar, not an artefact: a plan that needs 16.2 weeks still occupies 17.

| Candidate | H_nom [h] | r | credit [h] | H_eff [h] | N [weeks] | slack [h] | slack share of 176 |
|---|---|---|---|---|---|---|---|
| C1 | 200 | 0.00 | 0.0 | 200.0 | 25 | −24.0 | −13.64% |
| C2 | 120 | 0.10 | 12.0 | 108.0 | 14 | +68.0 | +38.64% |
| C3 | 260 | 0.20 | 52.0 | 208.0 | 26 | −32.0 | −18.18% |
| C4 | 180 | 0.25 | 45.0 | 135.0 | 17 | +41.0 | +23.30% |
| C5 | 150 | 0.30 | 45.0 | 105.0 | 14 | +71.0 | +40.34% |
| C6 | 140 | 0.05 | 7.0 | 133.0 | 17 | +43.0 | +24.43% |

C1 and C3 are infeasible on the arithmetic alone and need no probability to remove them. C1's nominal 200 hours exceed capacity by 24 hours with zero reuse available, and C3's 208 effective hours exceed capacity by 32 even after a 20% credit that is already the second-largest in the table. A selection process that reported either of them as a live option would be reporting an aspiration.

### Evidence credit

```text
FORMULA [B-02]: E = p1 * (1 + p2 + p2 * p3)
WHERE: E [tiers] expected earned tiers, range 0..3; p1 [probability] of the design
       tier; p2 [probability] of the trace tier given the design tier;
       p3 [probability] of the domain's top tier given both.
CONSTRAINT: the tiers are nested, so each probability is conditional on the one
            before it; a candidate whose p1 is zero earns nothing regardless of p2.
NUMERICAL EXAMPLE: C6: p1 = 0.90; p2 = 0.85; p3 = 0.80;
                   p1*p2 = 0.90 * 0.85 = 0.765;
                   p1*p2*p3 = 0.765 * 0.80 = 0.612;
                   E = 0.90 + 0.765 + 0.612 = 2.277 tiers.
INTERPRETATION: a candidate with a 90% chance of the first tier and 80% of the
                third still earns only 2.277 of a possible 3 tiers.
```

The nested form is what makes a low first-tier probability so expensive. C5 has p2 = 0.90 and p3 = 0.85, both better than C6's, yet its credit is far lower because its p1 is 0.55: `0.55 + 0.495 + 0.42075 = 1.46575`. The multiplication by p1 reaches into both higher tiers, so weakness at the base of the ladder is transmitted upward rather than localised.

| Candidate | p1 | p2 | p3 | p1·p2 | p1·p2·p3 | E [tiers] |
|---|---|---|---|---|---|---|
| C1 | 1.00 | 0.85 | 0.80 | 0.8500 | 0.68000 | 2.53000 |
| C2 | 0.90 | 0.85 | 0.75 | 0.7650 | 0.57375 | 2.23875 |
| C3 | 0.90 | 0.80 | 0.70 | 0.7200 | 0.50400 | 2.12400 |
| C4 | 0.95 | 0.85 | 0.80 | 0.8075 | 0.64600 | 2.40350 |
| C5 | 0.55 | 0.90 | 0.85 | 0.4950 | 0.42075 | 1.46575 |
| C6 | 0.90 | 0.85 | 0.80 | 0.7650 | 0.61200 | 2.27700 |

C5's 0.55 is the one input in the table that carries a specific, nameable risk: the scorecard needs a labelled dataset with a genuine time split, and the dataset's permission is unresolved. That is why the number is low while the other two are high — it is not a general pessimism about the candidate, it is one blocked dependency at the base of the ladder.

### Release probability

```text
FORMULA [B-03]: W ~ Binomial(22, 0.88) ;  P_rel = P(W >= N)
WHERE: W [weeks] the number of weeks in which any discretionary hours materialise;
       22 [weeks] the term length; 0.88 [probability] the per-week delivery chance,
       assumed independent across weeks; N [weeks] from B-01.
CONSTRAINT: independence is assumed, so a correlated failure such as illness across
            three consecutive weeks is under-represented in the left tail.
NUMERICAL EXAMPLE: C6 needs N = 17. Term k = 17:
                   C(22,17) = 22!/(17!5!) = 26334;
                   0.88^17 = 0.113817; 0.12^5 = 0.00002488;
                   term = 26334 * 0.113817 * 0.00002488 = 0.074581.
                   P(X=18) = 0.151924; P(X=19) = 0.234550; P(X=20) = 0.258005;
                   P(X=21) = 0.180194; P(X=22) = 0.060065;
                   P_rel = 0.074581 + 0.151924 + 0.234550 + 0.258005
                         + 0.180194 + 0.060065 = 0.959319.
INTERPRETATION: a 17-week plan releases with probability 0.959, and its mean
                available time is 22 * 0.88 = 19.36 weeks = 154.9 hours.
```

The distribution has mean 19.36 and standard deviation `sqrt(22 × 0.88 × 0.12) = sqrt(2.3232) = 1.5242` weeks. The steps in the tail are large, which is the mechanism that punishes a fragile plan: moving from N = 17 to N = 18 costs `0.959319 − 0.884738 = 0.074581` of probability, and moving to N = 19 costs a further `0.151924`. Those two step sizes are larger than most of the score differences in this case, which is the single most important fact in the analysis.

| N [weeks] | 14 | 17 | 18 | 19 | 20 | 21 | 22 | 25 |
|---|---|---|---|---|---|---|---|---|
| P(W ≥ N) | 0.999412 | 0.959319 | 0.884738 | 0.732813 | 0.498263 | 0.240259 | 0.060065 | 0.000000 |

### Density score

```text
FORMULA [B-04]: d = (E * P_rel) / H_eff , with d = 0 when H_eff > 176
WHERE: d [tiers/h] expected earned tiers per effective hour; E [tiers] from B-02;
       P_rel [probability] from B-03; H_eff [h] from B-01.
CONSTRAINT: H_eff must be positive, and a candidate above capacity scores zero
            rather than negative, because an unbuilt artifact earns no tier at all.
NUMERICAL EXAMPLE: C6: E = 2.27700; P_rel = 0.959319; H_eff = 133.0 h;
                   E * P_rel = 2.184369;
                   d = 2.184369 / 133.0 = 0.01642382 tiers/h.
INTERPRETATION: C6 yields about 0.0164 expected tiers per committed hour, which is
                3.99% below C4 and 3.47% above C5.
```

| Candidate | E [tiers] | P_rel | E·P_rel | H_eff [h] | d [tiers/h] | Status |
|---|---|---|---|---|---|---|
| C1 | 2.53000 | 0.000000 | 0.000000 | 200.0 | 0.0000000 | infeasible |
| C2 | 2.23875 | 0.999412 | 2.237430 | 108.0 | 0.0207170 | feasible |
| C3 | 2.12400 | 0.000000 | 0.000000 | 208.0 | 0.0000000 | infeasible |
| C4 | 2.40350 | 0.959319 | 2.305722 | 135.0 | 0.0170794 | feasible |
| C5 | 1.46575 | 0.999412 | 1.464888 | 105.0 | 0.0139513 | feasible |
| C6 | 2.27700 | 0.959319 | 2.184369 | 133.0 | 0.0164238 | feasible |

Reading the table in order of density gives C2, C4, C6, C5, then the two zeros. That is not the ranking the case publishes, and the difference is entirely the screens. A reader who stops at this table has computed the same numbers and reached a different decision, which is the correct response to a case that hides its screening step — and precisely why the screens are stated before the ranking rather than after it.

### Reuse tolerance

```text
FORMULA [B-05]: r* = 1 - H_step / H_nom ;  headroom = credit - (H_nom - H_step)
WHERE: r* [dimensionless] the reuse share below which N increases by one;
       H_step [h] the largest effective effort that still fits the planned N,
       equal to 8 * N_planned; the rest as in B-01.
CONSTRAINT: the step boundary falls on a multiple of 8 h, so headroom computed in
            hours is only meaningful relative to the planned credit, not absolutely.
NUMERICAL EXAMPLE: C4: N_planned = 17, so H_step = 8 * 17 = 136 h;
                   r* = 1 - 136/180 = 1 - 0.755556 = 0.244444;
                   planned credit = 180 * 0.25 = 45.0 h;
                   hours the plan can lose = 180 - 136 = 44.0 h;
                   headroom = 45.0 - 44.0 = 1.0 h = 2.222% of the credit.
INTERPRETATION: C4 tolerates a 1-hour shortfall in a 45-hour reuse claim before it
                drops a week-slot, so its plan is inside the noise of the estimate.
```

| Candidate | N | H_step [h] | r* | planned credit [h] | required credit [h] | headroom [h] | headroom share |
|---|---|---|---|---|---|---|---|
| C2 | 14 | 112 | 0.066667 | 12.0 | 8.0 | 4.0 | 33.33% |
| C4 | 17 | 136 | 0.244444 | 45.0 | 44.0 | 1.0 | 2.22% |
| C5 | 14 | 112 | 0.093333 | 45.0 | 14.0 | 31.0 | 68.89% |
| C6 | 17 | 136 | 0.028571 | 7.0 | 4.0 | 3.0 | 42.86% |

C4's row is the one that removes it. A 45-hour reuse claim with one hour of margin means that a 2.2% estimation error changes the week count from 17 to 18, which drops `P_rel` from 0.959319 to 0.884738 and the density from 0.0170794 to 0.0147670 — below C6. The screen's 25% threshold is calibrated exactly to this: it rejects any plan whose ranking depends on a margin smaller than a plausible estimation error.

C5's 68.89% looks like strength and is not quite that, because C5's exposure is at the base of the ladder rather than in the schedule. A candidate can pass every schedule screen and still be fragile in the dimension the screens do not cover, which is why the case reports both the screen result and the underlying p1 risk rather than a single verdict.

### Scenario expectation

```text
FORMULA [B-06]: d_bar = (1/3) * sum over s in {base, reuse/2, effort x1.15} of d_s
WHERE: d_s [tiers/h] the density recomputed under scenario s; the three scenarios
       are treated as equiprobable by assumption, not by measurement;
       reuse/2 means the reuse share is halved; effort x1.15 means H_nom rises 15%.
CONSTRAINT: contingency is handled separately as a probability weight, not as a
            fourth scenario, so that a declared risk is not double counted.
NUMERICAL EXAMPLE: C6: base 0.0164238; reuse/2 (H_eff = 136.5, N = 18,
                   P = 0.884738) gives 2.277*0.884738/136.5 = 0.0147586;
                   effort x1.15 (H_eff = 152.95, N = 20, P = 0.498263) gives
                   2.277*0.498263/152.95 = 0.0074178;
                   d_bar = (0.0164238 + 0.0147586 + 0.0074178)/3 = 0.0128667.
INTERPRETATION: C6 retains about 78% of its base density in the average scenario,
                and the effort-overrun scenario, not the reuse scenario, is what
                costs it the most.
```

| Candidate | base d | reuse/2 d | effort ×1.15 d | d_bar | min scenario |
|---|---|---|---|---|---|
| C2 | 0.0207170 | 0.0195815 | 0.0178115 | 0.0193700 | 0.0178115 |
| C4 | 0.0170794 | 0.0076037 | 0.0077139 | 0.0107990 | 0.0076037 |
| C5 | 0.0139513 | 0.0113597 | 0.0119947 | 0.0124352 | 0.0113597 |
| C6 | 0.0164238 | 0.0147586 | 0.0074178 | 0.0128667 | 0.0074178 |

C2 also carries a 70-hour contingency with an estimated 0.40 probability, which under B-06's constraint is applied as a weight rather than a scenario: `0.60 × 0.0193700 = 0.0116220`. That adjustment is what removes C2 from contention even before the feasibility screen, and the screen then confirms the removal on a simpler basis — with the contingency realised, `H_eff = 108 + 70 = 178 > 176`, so the candidate cannot fit at all.

### Ranking and the boundary case

The published ranking of survivors, after S1, S2, and S3, is C6 at 0.0128667, C5 at 0.0124352, C4 at 0.0107990, and C2 at 0.0116220 after its contingency weight. Ordering those four numbers gives C6 first, then C5, then C2, then C4.

```text
FORMULA [B-07]: gap_in_credit = (d_1 - d_2) * H_eff(1) / m ;  m = 1 + p2 + p2*p3
WHERE: the gap is expressed in units of p1 rather than density so that it can be
       compared with the plausible error in a probability estimate; m is the
       multiplier that converts p1 into E in B-02.
CONSTRAINT: the conversion assumes p2 and p3 are held fixed, so it bounds one
            input at a time and says nothing about joint movement.
NUMERICAL EXAMPLE: C6 vs C5: 0.0128667 - 0.0124352 = 0.0004315 tiers/h;
                   in credit terms 0.0004315 * 133 = 0.057390 tiers;
                   m = 1 + 0.85 + 0.68 = 2.53;
                   required p1 change = 0.057390 / 2.53 = 0.022684.
INTERPRETATION: C6's first-tier probability would have to fall by 2.27 percentage
                points, from 0.90 to 0.877, for C5 to overtake it on d_bar.
```

The same conversion applied to C4 against C6 explains why the tolerance screen is doing more work than the score. C4's advantage is `0.0006556 tiers/h`; in credit units at 133 hours that is `0.087195`, or `0.034465` of p1 — a 3.45 percentage point move. A three-and-a-half point difference between two elicited probabilities is not a finding; it is the resolution of the inputs. But C4's one-hour reuse margin converts to a 0.074581 probability step, which is twice as large as the score gap it was defending. The screen is therefore not overriding a real advantage; it is refusing to treat a sub-resolution advantage as real.

**Boundary counterexample.** The density metric can be gamed by shrinking scope, and the arithmetic shows how badly. Take an artifact of 4 hours with expected credit 0.5 tiers: `d = 0.5 × 1 / 4 = 0.125 tiers/h`, which is `0.125 / 0.0207170 = 6.03` times the best real candidate's density. Nothing about that artifact is good — it earns half a tier and probably earns none — but per unit of effort it dominates the table. The metric's blind spot is the denominator: it rewards smallness as readily as it rewards yield.

The correction is the S3 floor, and its arithmetic is simple. With `E ≥ 2.0` required, the smallest artifact that can compete at C4's density needs `2.0 / 0.0170794 = 117.1` effective hours, which is 15 week-slots. That floor rules out the four-hour artifact on the evidence axis and rules out de-scoping as a ranking strategy in general. A second boundary sits at `H_eff = 0`, where the ratio is undefined rather than infinite; the floor removes that case from consideration without needing a special rule.

## 7. The Finding

C6 is selected because it is the only candidate that passes every screen and also ranks first among survivors on the probability-weighted metric. It does not rank first on the unadjusted score, and it does not have the best worst case, so the selection rests on three separate results rather than one.

| Finding | Value | Derivation | Claim strength |
|---|---|---|---|
| Selected candidate's base density | 0.0164238 tiers/h | B-02, B-03, B-04 | illustrative, arithmetic checked |
| Gap to the highest raw score (C4) | 3.99% | 0.0170794 / 0.0164238 − 1 | illustrative |
| C4's reuse headroom | 1.0 h of 45.0 h (2.22%) | B-05 | illustrative |
| Screen threshold C4 fails | 25% headroom required | rule S2 | design choice, stated |
| C6's headroom | 3.0 h of 7.0 h (42.86%) | B-05 | illustrative |
| C2's effective hours with contingency | 178 h against 176 h capacity | B-01 | illustrative |
| C6 scenario mean | 0.0128667 tiers/h | B-06 | illustrative |
| C5 runner-up margin | 3.47% below C6 | 0.0128667 / 0.0124352 − 1 | illustrative |
| p1 movement that would flip C6 and C5 | 2.27 percentage points | B-07 | illustrative |
| p1 movement that would flip C6 and C4 | 3.45 percentage points | B-07 | illustrative |
| Probability step at the 17→18 week boundary | 0.074581 | B-03 | illustrative |

Two of these numbers deserve to travel with any summary of the decision. The first is the 0.074581 step: it is the cost of missing a single week-slot, and it exceeds every score gap in the table except the C2 gap. A reader who remembers only that number will understand why the tolerance screen exists. The second is the 2.27-point flip threshold, because it is the honest statement of how much confidence the ranking deserves: two candidates separated by less than a rounding error in an elicited probability are, for practical purposes, tied.

**The weakest link, named.** C6's scenario mean is the ranking basis, and its worst scenario is the lowest among survivors at 0.0074178. That is not a detail to be filed away; it is the assumption the case is least protected against. If effort overruns by 15% — which for a first estimate is an ordinary outcome rather than a tail event — C6 falls below both C4 and C5 in that scenario and finishes 42% below its base. The ranking is therefore a statement about the central case with a weak tail, and the mitigation in Section 8 exists precisely because of it.

| Sentence permitted by this finding | Basis |
|---|---|
| "Under the stated screens, C6 ranks first at 0.0129 expected tiers per hour across three scenarios." | B-06, arithmetic checked |
| "C4's higher raw score depends on a reuse credit with one hour of margin." | B-05 |
| "C2 is infeasible once its 70-hour contingency is counted." | B-01 with contingency |
| "The C6–C5 gap is smaller than the plausible error in a first-tier probability." | B-07 |

| Sentence prohibited by this finding | Why |
|---|---|
| "C6 is the best capstone." | the case compares six constructed options under constructed inputs |
| "C4 would have failed." | C4's screen failure is about estimate resolution, not execution |
| "The term has 176 hours." | capacity is a planning convention |
| "C6 will earn 2.28 tiers." | credit is an expectation, not a forecast of one outcome |

## 8. System/Solution Design

The artifact of this case is a selection procedure, so its "system design" is the procedure's structure: who estimates, who screens, who ranks, and what happens when the plan starts to slip.

```text
[idea pool] -> [estimate all six rows] -> [reviewer signs the estimates] -> [screen S1]
                                                                              |
                     [publish + weakest-assumption note] <- [rank survivors] <- [screen S3]
                                     ^                    |
                                     |                    v
                          [screen S2: tolerance]   [week-11 checkpoint]
                                                          |
                                            [continue | de-scope | switch]
```

The order matters. Estimates are signed before scoring, screens run before ranking, and the checkpoint is defined before the term starts. Each of those is a control against a specific failure: preference leakage, fragile plans disguised by high scores, and sunk-cost escalation.

| Stage | Owner | Input | Output | Failure it prevents |
|---|---|---|---|---|
| estimate | lead | candidate descriptions and prior artifact inventory | six signed rows | a favourite scored on tuned inputs |
| sign-off | reviewer | the six rows | acceptance or a named objection | silent revision of a weak input |
| screen S1 | reviewer | effective hours with contingency | feasible set | an infeasible plan carried on optimism |
| screen S2 | reviewer | reuse headroom | non-fragile set | a ranking resting on one hour |
| screen S3 | reviewer | expected credit | floor-compliant set | scope shrinkage gaming the metric |
| rank | lead | survivors and three scenarios | ordered list with d_bar | a single-scenario decision |
| checkpoint | lead | actual hours at week 11 | continue, de-scope, or switch | sunk-cost escalation |

**Checkpoint design.** Week 11 is chosen because it is the midpoint of the 22-week term and, for C6, it falls after 88 planned hours of a 133-hour plan — 66.2% of the work. The trigger is arithmetic rather than felt: if actual hours to date exceed plan by more than the remaining slack ratio, the plan is off track by a stated amount.

| Checkpoint reading at week 11 | Hours used | Projection | Action |
|---|---|---|---|
| On plan | ≤ 88 | ≤ 133 | continue unchanged |
| Mild overrun | 89 to 96 | ≤ 145 | drop the third-tier commitment only |
| Severe overrun | > 96 | > 145 | stop the artifact at two tiers and rewrite the claim |
| Under plan | < 78 | < 118 | consider restoring one de-scoped item, not adding scope |

The third row is the important one, and its arithmetic is why the case forbids de-scoping as a ranking strategy. Dropping the third tier reduces C6's credit from 2.277 to `0.90 + 0.765 = 1.665` tiers and its hours by roughly 35, giving a density of `1.665 × 0.999412 / 98 = 0.016982` — higher than the selected plan's 0.0164238. De-scoping always raises density, because the metric divides by effort. The only thing that stops a plan from shrinking to a maximally dense nothing is the S3 floor: 1.665 is below 2.0, so the de-scoped artifact leaves the comparable class entirely rather than winning it.

That is a genuinely uncomfortable design property and it should be stated plainly. The floor is not a refinement of the metric; it is a boundary that says the metric's answers are only meaningful above a certain size, and below that size the honest claim is "this is a smaller artifact with a different standard", not "this is a more efficient artifact".

**Switch rule.** The case does not permit an open-ended switch, because a switch after week 8 wastes the sunk hours without producing the alternative. It permits exactly one switch, at week 11, and only to C5, whose remaining work at that point is 105 − 88 = 17 hours if the reused components overlap, or a full 105 hours if they do not. The second number is the realistic one, and it exceeds the 11 remaining weeks at 8 hours each only in the favourable case. The switch is therefore available in principle and unlikely in practice, which is the honest description of most switch options discovered halfway through.

## 9. Evaluation & Validation

| Metric | Numerator / denominator | Window / comparator | Guardrail | Status |
|---|---|---|---|---|
| Screen agreement | cases where two reviewers remove the same candidate / all candidates | one cycle, both reviewers independently | disagreement is recorded, not averaged | not executed |
| Arithmetic fidelity | derived cells matching an independent recomputation / derived cells | all twelve derived columns | no cell copied from another cell | not executed |
| Ranking stability | orderings preserved under one-at-a-time input variation / orderings tested | the three scenarios plus four p1 variations | the top two are reported as tied if within 1 step | not executed |
| Tolerance check accuracy | candidates correctly classified at the 25% threshold / all candidates | the reuse column | headroom computed in hours and in share | not executed |
| Floor compliance | survivors above E = 2.0 / survivors | after screening | no candidate admitted by rounding | not executed |
| Preference-leakage control | whether all six rows predate the score computation / one cycle | timestamp comparison | a violated order voids the ranking | not executed |
| Realised effort | actual hours consumed / planned effective hours | end of term | overrun reason recorded before the checkpoint | not measured |
| Realised tier | tiers actually earned / tiers expected | end of term | the expectation is not revised after the outcome | not measured |

**Uncertainty.** Three sources dominate, and their magnitudes are comparable to the effects being measured, which is the central honesty problem of this case. First, the week model's 0.88 is a constructed probability, and at 0.80 the probability that a 17-week plan releases falls from 0.959319 to 0.732638, which would reorder the survivors. Second, the tier probabilities are elicited priors whose plausible error is a few percentage points per tier, and the C6–C5 gap corresponds to 2.27 points. Third, the nominal-hour estimates are analogies to prior artifacts, and a 15% overrun — the mild scenario — moves C6 from first to third.

| Input varied | Base → varied | Effect on C6's rank | Why |
|---|---|---|---|
| Delivery probability 0.88 → 0.80 | P_rel 0.959319 → 0.732638 | C6 falls below C5 | C6 sits on the 17-week step, C5 on the 14-week step |
| Delivery probability 0.88 → 0.92 | P_rel 0.959319 → 0.988134 | C6 rises and C4 becomes viable | both 17-week plans gain the same step |
| C6 p1 0.90 → 0.85 | E 2.277 → 2.1505 | C6 falls below C5 | the multiplier 2.53 amplifies the p1 move |
| C6 reuse 0.05 → 0.30 | H_eff 133 → 98 | C6 rises to 0.0210 | N falls to 13, so P_rel reaches 0.999899 |
| Term length 22 → 18 weeks | N unchanged, capacity falls to 144 | C4 becomes infeasible | 135 fits, 133 fits, but slack collapses |
| Capacity 176 → 160 hours | N unchanged | C4 and C6 both infeasible | a 10% calendar loss removes both finalists |

The fifth row is the one a reader should test first, because it shows the fragility of the whole exercise. An 18-week term caps capacity at 144 hours: C6's 133 still fits, but its 43-hour slack falls to 11, which is 8.3% of capacity, and the S2-style reasoning that protected it no longer applies. A selection that survives a 22-week calendar and collapses at 18 weeks is not a robust selection; it is a selection calibrated to one calendar.

**Population and comparator.** The population is the six candidates in one cycle. The comparator for the ranking is the deferral baseline at zero credit and, within the field, the runner-up. No historical selection is used as a comparator because none was recorded with these inputs.

**Executed versus not executed.** Nothing here was executed in the sense of a measurement. No term was run, no hours were logged, no artifact was built, and no reviewer independently screened the six candidates. Every status entry reads "not executed" or "not measured", and the ranking is a decision made from constructed inputs rather than a result of an experiment.

**Why a pre-registered ranking matters.** The evaluation would be worthless if the ranking were revised after the term. The selection record must therefore fix three things before week 1: the ordered list, the expected credit per candidate, and the sentence that names the weakest assumption. A later reader can then compare the expectation with the outcome and learn something about the estimates, which is the only way a constructed selection process improves.

## 10. Action Taken

The action is the selection itself plus the record that makes it reviewable. No artifact was built. The actions below are the proposed steps that follow from the decision, each with a stop condition so that a failure does not propagate silently into the term.

| Action | Owner | Required input | Stop condition | What it unlocks |
|---|---|---|---|---|
| Freeze the six-row table with sign-off | reviewer | all twelve derived columns recomputed | any disagreement between reviewers | a ranking that may be published |
| Record the ranking and the weakest assumption | lead | the frozen table plus B-06 | the assumption sentence is missing | an auditable decision |
| Write the C6 scope to the evidence floor | lead | E ≥ 2.0 with the three tiers named | the scope yields fewer than 2 tiers | a comparable artifact |
| Define the week-11 checkpoint arithmetic | lead | plan hours and the three readings | the thresholds are not numeric | a de-scope rule that can actually fire |
| Confirm the C5 fallback's data permission | lead | a decision on the labelled dataset | permission unresolved | a genuine second option |
| Publish the rejected candidates with reasons | lead | screen results per candidate | a rejection has no stated rule | a ranking a reader can contest |

The third action is the one that most often fails in practice, because writing a scope to a floor feels like artificial precision. Its purpose is different: naming the three tiers before work starts is what makes the mid-term de-scope decision decidable. A plan that says "build the capstone" cannot tell at week 11 whether dropping a component leaves a coherent artifact; a plan that says "tier one is the design record, tier two is the trace, tier three is the hardware exercise" can drop tier three and know exactly what it has lost.

**What is deliberately not actioned.** No second candidate is started in parallel. The temptation is real — 43 hours of slack look like room for a second thread — and the arithmetic rejects it: two candidates would each need their own 133 and 105 effective hours, totalling 238 against a capacity of 176, so at least one would be abandoned mid-term and the abandonment would consume the slack that made the first plan survivable.

**What happens if the selection is wrong.** If C6 stalls at week 11 with more than 96 hours consumed, the recorded fallback is to finish at two tiers and publish a two-tier artifact with the failure of the third tier as part of its content. That outcome still clears the floor at `0.90 + 0.765 = 1.665`? It does not, and the record must say so: 1.665 is below 2.0, so a two-tier finish is a different artifact class and its claim set must be rewritten rather than republished. Deciding this in advance is what stops the week-11 conversation from becoming an argument about whether the shortfall is acceptable.

## 11. Outcome & Results

**Outcome status: not measured.** No term was run. The outputs are a ranking, a set of screens with thresholds, and a checkpoint rule. The numbers below are the arithmetic of the decision, not its consequences.

| Output | Value | Type |
|---|---|---|
| Selected candidate | C6 | decision |
| Ranking metric at selection | 0.0128667 tiers/h scenario mean | derived, illustrative |
| Base density of the selection | 0.0164238 tiers/h | derived, illustrative |
| Runner-up and margin | C5 at 3.47% below | derived, illustrative |
| Candidates removed by S1 | C1, C3, and C2 with contingency | rule application |
| Candidates removed by S2 | C4 | rule application |
| Candidates surviving S3 | C5, C6 | rule application |
| Flip threshold versus C5 | 2.27 percentage points of p1 | derived, illustrative |
| Checkpoint week | 11, at 66.2% of planned hours | design choice |

The distinction between "decision" and "derived" in the third column is doing real work. The selection is a commitment that can be wrong; the arithmetic is a computation that can be checked. Mixing them is how a constructed score becomes a remembered fact, and the type column exists to keep them apart.

| Claim ID | Bounded claim | Evidence type | Supports | Does not establish |
|---|---|---|---|---|
| CLM-01 | Under the stated screens and inputs, C6 ranks first at 0.0128667 tiers/h | illustrative, arithmetic checked | the selection and its record | that C6 will earn its expected credit |
| CLM-02 | C4's 3.99% raw advantage rests on 1.0 h of reuse headroom | illustrative, arithmetic checked | the tolerance screen's purpose | that C4's plan is wrong |
| CLM-03 | C2 becomes infeasible at 178 effective hours with contingency | illustrative, arithmetic checked | the feasibility screen | that the 70-hour contingency is correctly sized |
| CLM-04 | A 4-hour artifact with 0.5 expected credit scores 6.03× the best candidate | illustrative boundary | the need for an evidence floor | that any real artifact has such parameters |
| CLM-05 | Missing one week-slot costs 0.074581 of release probability | illustrative, arithmetic checked | the emphasis on tolerance over score | the true week distribution |
| CLM-06 | Two candidates 2.27 points apart in p1 are practically tied | derived from B-07 | the honest precision of the ranking | anything about which one performs better |

CLM-06 is the claim a reviewer should press hardest, because it undercuts the headline. If the C6–C5 gap is within the resolution of the inputs, then the selection between them is not determined by evidence and the case should say so rather than dress up a 3.47% difference. The defence is not that the gap is large; it is that C5 carries a separately identified base-of-ladder risk (an unresolved data permission) that the ranking metric captures only through its p1, and that a reader who finds the p1 values unreliable is left with a genuine tie and should treat it as one.

## 12. Failure Modes & What Went Wrong

| Failure mechanism | Competing causes | Discriminating check | Containment | Prevention |
|---|---|---|---|---|
| Reuse overclaim | interface drift; forgotten dependencies; optimistic memory | compute headroom in hours, not as a share | re-estimate before the freeze | S2 with a 25% threshold |
| Preference leakage | lead decided first and estimated second | timestamps of rows versus timestamps of scores | void the ranking, re-estimate | estimate and sign before scoring |
| Score-driven scope inflation | high score from a large candidate | check H_eff against capacity before reading the score | remove on feasibility alone | S1 applied first, never waived |
| Metric gaming by shrinkage | small artifacts score high per hour | compare credit in tiers, not density | reclassify as a smaller artifact | S3 floor at 2.0 tiers |
| Correlated bad weeks | illness, deadline clusters, travel | left tail of the week distribution versus the model | extend the term or de-scope | do not claim independence; record the assumption |
| Step-boundary blindness | rounding 16.6 weeks to 16 | recompute N from unrounded H_eff | correct the week count | ceiling applied to the raw value |
| Contingency double counting | counting hours and probability twice | check whether the row appears as both a scenario and a weight | remove one treatment | state contingency policy in B-06 |
| Sunk-cost escalation | 88 hours already spent at week 11 | compare remaining work with remaining capacity | switch once, or finish at two tiers | pre-register the checkpoint readings |
| Domain mismatch with the portfolio | six similar evidence shapes | count distinct shapes across the portfolio | prefer a different family next cycle | include shape in the candidate table |
| Tier ladder assumed identical across domains | treating the scorecard's tiers as the trace's tiers | list the three tiers per domain in words | re-elicit p3 per domain | define tiers before eliciting |
| Capacity treated as measured | 176 taken as a contract | compare with two prior terms | treat capacity as a distribution | recalibrate A-01 each cycle |
| Ranking republished after the outcome | temptation to record the winner as the forecast | freeze timestamp versus revision timestamp | keep both records | publish the expectation at week 0 |

The first four rows are the ones that damage the decision rather than the term, and they share a signature: each produces a ranking that looks arithmetically sound. A reuse overclaim produces correct arithmetic on a wrong input; preference leakage produces a correct ranking of tuned inputs; scope inflation produces a high score for an infeasible plan. The discriminating checks are therefore all about inputs and order, never about the output, which is why they must be run by someone who did not produce the numbers.

The fifth row deserves expansion because it is the assumption the whole week model leans on. Independence across weeks is false in the way that matters: a single illness can remove three consecutive weeks, which is a left-tail event the binomial assigns almost no probability to. The model gives `P(W ≤ 14) = 1 − 0.999412 = 0.000588` for a 3-week loss; three weeks lost to one event is `P(W ≤ 19)` in the independent model only if the loss were three *separate* single weeks, which is a different event with a different probability. The model therefore understates correlated loss by construction, and the containment is a longer term rather than a better formula.

The ninth and tenth rows are domain-specific and easy to miss. A portfolio of six candidates that all produce "a backtest plus a dataset description" is a portfolio of one evidence shape, and the ranking cannot see that because shape is not a column in the score. Adding it as a qualitative column — never as a weight — is the cheap fix, and re-eliciting p3 per domain matters because a hardware tier and a calibration tier are not the same achievement even when both are called the third tier.

This table is a diagnostic of the decision process, not a record of an incident. No row describes a defect discovered in a term, because no term was run. The containment column is what the process would do, and the prevention column is what the next cycle would change.

## 13. Transferable Lesson

The transferable content is the ordering: screens that do not depend on the score, then a score that does not depend on the screens. The pattern applies wherever a fixed budget must be allocated across options whose yield is estimated rather than measured, and the failure it prevents is the same in every domain — a fragile option winning because its fragility is invisible to the metric that ranks it.

| Domain | Options | Effort unit | Screen that must precede ranking | Fragility it catches |
|---|---|---|---|---|
| Model vendor selection | three vendors | licence plus integration weeks | data-residency and exit-clause checks | a contract that cannot be exited |
| Hiring | four candidates | interviewer hours | reference and scope checks | a candidate whose evidence is one project |
| Research portfolio | five studies | instrument time | ethics and sample-access checks | a study that cannot recruit |
| Infrastructure migration | four plans | engineer-weeks | reversibility check | a migration with no rollback |
| Curriculum design | six modules | teaching hours | prerequisite check | a module with an unmet prerequisite |

In each row the screen answers a yes/no question that no amount of scoring can substitute for. A vendor with the best integration estimate and no exit clause is not a good vendor; the clause is a precondition, not a feature. That is exactly the status of C4 in this case: its score is genuinely higher, and its plan is genuinely one hour deep.

**Why "screen then rank" beats "weight everything".** Folding an objection into a composite score requires choosing a weight, and the weight is unmeasurable. A 25% reuse-tolerance requirement expressed as a screen is verifiable — the headroom is 1.0 hour, the threshold is 25%, the candidate fails — whereas the same requirement expressed as a penalty of, say, 0.002 tiers per hour would be a number pulled from nowhere and impossible to argue with. Screens convert judgement into a checkable comparison at the cost of refusing to trade off, which is the right trade when the objection is about feasibility rather than preference.

**Where the lesson stops.** Screens can be over-used. A process with fifteen screens selects nothing, and each screen can be tuned to remove an inconvenient option. The discipline that prevents this is that every screen must state what it removes and why the removal does not depend on the score, and the number of screens should be small enough that a reviewer can hold them. Three screens here, each with a one-line rationale and a numeric threshold, is close to the practical maximum.

**A second limit, about the metric itself.** Density per hour is the right metric only when the artifact class is fixed. The moment candidates can vary in size, the metric rewards shrinkage, and the floor becomes load-bearing. Any metric of the form "value per unit cost" has this property, and the general lesson is that such metrics need an admissibility rule on the value side — a minimum credit, a minimum scope, a minimum sample — or they will be optimised by shrinking the denominator.

**What this case would do differently with real history.** Two terms of logged hours would replace the 0.88 with a measured distribution; one completed artifact per domain would replace the elicited p2 and p3 with frequencies; and the reuse estimates could be checked against what was actually reused. With those three, the ranking would still be a decision, but it would be a decision with inputs rather than priors. Until then the honest summary is: the process is sound, the numbers are illustrative, and the weakest assumption is named in the record.

## 14. Connections & Cross-References

Related cases and methods: the [release-gate case](./01_local_signal_demo_release_gate.md) supplies the tier vocabulary this selection scores against and demonstrates the evidence discipline that C1 and C6 would have to satisfy; the [capstone evidence rubric](../Cheat_Sheets/01_Capstone_Evidence_Rubric.md) defines the tiers, the evidence boundary, and the rule that a passing test is not hardware evidence; the [unit index](../INDEX.md) records both cases as proposed. The rubric supplies definitions, not effort data, and neither source contains a measured hour of any project.

Relation to the sibling case is one-directional and worth stating precisely. That case decides whether one artifact's evidence supports a claim; this case decides which artifact to build. A reader can apply the gate to a candidate this case rejected, and the gate would say DESIGN-only for all six today — which is the correct result, since none of them exists.

### Interview pack

**1. Question — What is the decision, at what grain, and who owns it?**
- **Checked answer:** Selecting one candidate for one 22-week term and one portfolio slot, owned by the capstone programme lead and challenged by a reviewer who may reject a screen.
- **Reasoning:** The grain fixes what "one decision" means, so a mid-term question about whether to add a second candidate is out of scope rather than a judgement call.
- **Follow-up:** What would make this two decisions rather than one?
- **Red flags:** "Pick the best project", which names no grain, no owner, and no commitment.
- **Scoring:** Full credit requires grain, owner, and the challenger role.

**2. Question — Compute C6's effective hours, slack, and week count.**
- **Checked answer:** `140 × 0.05 = 7.0` hours of credit; `140 − 7 = 133.0` effective hours; slack `176 − 133 = 43` hours; `ceil(133/8) = ceil(16.625) = 17` weeks.
- **Reasoning:** Reuse is subtracted before the ceiling is taken, because the ceiling is a calendar step rather than a linear cost.
- **Follow-up:** What happens to the week count if the 16.625 is rounded to 16 before the ceiling?
- **Red flags:** Reporting 16 weeks, or applying reuse after computing the week count.
- **Scoring:** Require the order of operations to be stated.

**3. Question — Why does a 0.05 reuse share still matter for C6?**
- **Checked answer:** It saves 7 hours, and only 4 of those hours are needed to keep the plan inside 17 weeks, so the credit's function is schedule protection rather than headline effort reduction.
- **Reasoning:** The step boundary is at 136 hours, so credit beyond that point does not change the week count and therefore does not change the release probability.
- **Follow-up:** What is the largest effective effort that still fits 17 weeks?
- **Red flags:** Treating all reuse as equally valuable regardless of where it falls relative to a step.
- **Scoring:** Credit only when the 136-hour boundary is identified.

**4. Question — Compute the evidence credit for C5 and explain why it is low despite strong higher tiers.**
- **Checked answer:** `0.55 + 0.55×0.90 + 0.55×0.90×0.85 = 0.55 + 0.495 + 0.42075 = 1.46575` tiers; the low p1 of 0.55 multiplies through both higher tiers.
- **Reasoning:** The tiers are nested, so a weak base caps everything above it rather than costing a fixed amount.
- **Follow-up:** What single change would raise C5's credit most, and by how much?
- **Red flags:** Averaging the three probabilities instead of multiplying along the nesting.
- **Scoring:** Require the multiplicative structure and the identification of p1 as the lever.
- *(Lever check: raising p1 to 0.90 gives `0.90 + 0.81 + 0.6885 = 2.3985` tiers.)*

**5. Question — Compute P(W ≥ 17) for W ~ Binomial(22, 0.88) and show one term.**
- **Checked answer:** `0.959319`, from terms `0.074581 + 0.151924 + 0.234550 + 0.258005 + 0.180194 + 0.060065`; the k = 17 term is `C(22,17) × 0.88^17 × 0.12^5 = 26334 × 0.113817 × 0.00002488 = 0.074581`.
- **Reasoning:** The plan releases if at least 17 weeks deliver, so the probability is a right tail rather than a point mass.
- **Follow-up:** What is the mean available time in hours, and how far above the requirement is it?
- **Red flags:** Computing `P(W = 17)` and calling it the release probability.
- **Scoring:** Require the tail, one explicit term, and the mean.
- *(Mean: `22 × 0.88 = 19.36` weeks = 154.88 hours.)*

**6. Question — How much probability does missing one week-slot cost at the 17-week boundary?**
- **Checked answer:** `0.959319 − 0.884738 = 0.074581`, about 7.5 percentage points for a single additional week.
- **Reasoning:** The distribution's standard deviation is only 1.5242 weeks, so a one-week shift is two-thirds of a standard deviation in the relevant region.
- **Follow-up:** Compare that 0.074581 with the density gap between the two finalists and interpret the comparison.
- **Red flags:** Treating a week as a small cost because the mean slack is 2.36 weeks.
- **Scoring:** Require the step size and the comparison with the score gap.

**7. Question — Compute the density score for C4 and for C6, and state the margin.**
- **Checked answer:** C4 `2.4035 × 0.959319 / 135 = 0.0170794`; C6 `2.277 × 0.959319 / 133 = 0.0164238`; C4 leads by `0.0006556` tiers/h, or 3.99%.
- **Reasoning:** Both share the same release probability because both need 17 weeks, so the difference comes only from credit and effective hours.
- **Follow-up:** What is that gap in units of p1, and is it larger than a plausible estimation error?
- **Red flags:** Comparing credit without dividing by hours, which would favour C1.
- **Scoring:** Require both scores and the percentage.
- *(In p1 units: `0.0006556 × 133 / 2.53 = 0.034465`, or 3.45 points.)*

**8. Question — Why is C4 removed despite the highest raw score among feasible candidates?**
- **Checked answer:** It needs 44 of its 45 credited reuse hours to stay inside 17 weeks, so a 1-hour shortfall (2.22%) pushes it to 18 weeks, dropping its density to 0.0147670 and its scenario mean to 0.0107990.
- **Reasoning:** The screen exists because a margin smaller than a plausible estimation error cannot support a ranking claim, whatever the score says.
- **Follow-up:** At what reuse share would C4 have 25% headroom?
- **Red flags:** Keeping C4 because its score is highest, which is exactly the failure the screen prevents.
- **Scoring:** Require the headroom arithmetic, not just the word "fragile".
- *(25% headroom needs credit 55 hours, so `r ≥ 55/180 = 0.306`.)*

**9. Question — Why is C2 removed, given it has the highest base density and the most slack?**
- **Checked answer:** Its 70-hour contingency takes effective effort to 178 hours against a 176-hour capacity, so the plan is infeasible whenever the contingency is realised; weighted at 0.40 that is a score of `0.60 × 0.0193700 = 0.0116220`.
- **Reasoning:** Slack measured without declared contingencies is not slack, it is unallocated risk.
- **Follow-up:** At what contingency size would C2 still keep a release probability above 0.90?
- **Red flags:** Comparing C2's base score with others' base scores while ignoring its declared contingency.
- **Scoring:** Require the contingency to enter the arithmetic explicitly.
- *(Above 0.90 needs `H_eff ≤ 152` hours, so contingency ≤ 44 hours.)*

**10. Question — Diagnose a ranking in which the candidate with the most slack finishes last.**
- **Checked answer:** Slack is not a ranking input; it only matters through the release probability and through the tolerance screen, so a candidate can have large slack and low credit at the same time.
- **Reasoning:** C5 has 71 hours of slack and finishes second because its credit is 1.46575 tiers, driven by a 0.55 first-tier probability.
- **Follow-up:** Which two columns would you inspect first to confirm that diagnosis?
- **Red flags:** Assuming the ranking must correlate with slack, or "fixing" the metric to include it.
- **Scoring:** Require the separation of schedule robustness from evidential yield.

**11. Question — A 4-hour artifact with 0.5 expected credit scores 0.125 tiers/h. What does that prove?**
- **Checked answer:** That the density metric rewards smallness, since 0.125 is 6.03 times the best real candidate's 0.0207170; the metric needs an admissibility floor on credit.
- **Reasoning:** Any value-per-cost ratio can be inflated by shrinking the denominator, so the value side must carry a minimum.
- **Follow-up:** With an `E ≥ 2.0` floor, how many hours does a competitive artifact need?
- **Red flags:** Concluding the 4-hour artifact is efficient, or discarding the metric entirely.
- **Scoring:** Require the floor arithmetic and the general principle.
- *(Floor arithmetic: `2.0 / 0.0170794 = 117.1` effective hours.)*

**12. Question — Why is contingency handled as a probability weight rather than as a fourth scenario?**
- **Checked answer:** Because listing it as both a scenario and a weight would count the same risk twice, inflating its influence on the expected score.
- **Reasoning:** The three scenarios describe estimation error, which applies to all candidates; the contingency describes a candidate-specific risk with its own probability.
- **Follow-up:** What would change if the contingency probability were 0.10 rather than 0.40?
- **Red flags:** Adding contingency hours into the base case while also weighting them.
- **Scoring:** Require the double-counting argument.
- *(At 0.10: `0.90 × 0.0193700 = 0.0174330`, which would rank C2 first.)*

**13. Question — Why does the case use expected value for ranking and maximin only for screening?**
- **Checked answer:** Because the extreme scenario (halved reuse) was constructed to be severe, and giving it equal standing turns the ranking into a pessimism contest; screens capture the cliff without letting it dominate.
- **Reasoning:** Maximin would select C5 at 0.0113597 minimum, whose base is 15% below C6's, purely because its schedule is insensitive.
- **Follow-up:** Name one situation where maximin should replace expected value entirely.
- **Red flags:** Claiming expected value is always correct, or that maximin is always safer.
- **Scoring:** Require both the rejection and its limit.
- *(Maximin case: a single irreversible commitment with no recovery path.)*

**14. Question — C6's worst scenario is the lowest among survivors. How do you respond?**
- **Checked answer:** By naming it as the weakest assumption, defining a week-11 checkpoint with numeric triggers, and accepting that a 15% overrun moves C6 to third.
- **Reasoning:** A ranking that does not disclose its own weakest scenario is a ranking that cannot be falsified; disclosure plus a checkpoint is the available mitigation.
- **Follow-up:** What does the checkpoint do at 96 hours consumed, and why that number?
- **Red flags:** Recomputing the scenario weights until C6's worst case looks acceptable.
- **Scoring:** Require disclosure plus a numeric checkpoint.
- *(At 96 hours: 72.2% of plan consumed at the midpoint, so the third tier is dropped.)*

**15. Question — What does de-scoping do to density, and why is that a problem?**
- **Checked answer:** It raises density, because credit falls less than hours: C6 at two tiers is `1.665 × 0.999412 / 98 = 0.016982`, above its three-tier 0.0164238.
- **Reasoning:** The metric divides by effort, so removing work is always rewarded; only the credit floor distinguishes a smaller artifact from a better one.
- **Follow-up:** Where does the two-tier C6 sit relative to the floor, and what follows?
- **Red flags:** Presenting the de-scoped density as an efficiency gain.
- **Scoring:** Require the computation and the floor comparison.
- *(Below the floor: 1.665 < 2.0, so it leaves the comparable class.)*

**16. Question — Which single input change would reverse the selection, and why that one?**
- **Checked answer:** A delivery probability of 0.80 instead of 0.88: `P(W ≥ 17)` falls to 0.732638 while `P(W ≥ 14)` stays at 0.979858, so C6 falls below C5.
- **Reasoning:** C6 sits on the 17-week step and C5 on the 14-week step, so a lower per-week probability penalises the candidate closer to the capacity edge.
- **Follow-up:** What would you measure first to decide between 0.88 and 0.80?
- **Red flags:** Naming a change that shifts every candidate equally, which cannot reverse anything.
- **Scoring:** Require a change with a differential effect and the arithmetic to show it.

### Reviewer drill-down

The cards below are the review a published ranking must survive. None reports a completed review.

**RD-01 — Estimation order.** Do all six rows carry timestamps earlier than the first computed score? A later row is not automatically wrong, but it must carry a change event naming the reason. Fail signal: a row revised in the same minute the ranking was produced. Owner: reviewer.

**RD-02 — Reuse evidence.** Does each reuse share name a prior artifact, and was the interface inspected rather than remembered? Fail signal: a share with no artifact identifier, which makes the credit unfalsifiable. Owner: lead, checked by reviewer.

**RD-03 — Ceiling discipline.** Is the week count computed from unrounded effective hours? Fail signal: 16.6 weeks reported as 16, which silently grants a week the plan does not have. Owner: reviewer.

**RD-04 — Contingency treatment.** Does each declared contingency appear exactly once, either as hours in a scenario or as a probability weight? Fail signal: the same 70 hours discounted twice. Owner: reviewer.

**RD-05 — Tolerance arithmetic.** Is headroom reported in hours and as a share of the credit? Fail signal: a 2.22% margin described as adequate because the hours look small. Owner: reviewer.

**RD-06 — Floor application.** Was S3 applied before the ranking, and does every survivor clear 2.0 tiers? Fail signal: a candidate admitted at 1.98 by rounding. Owner: reviewer.

**RD-07 — Weight disclosure.** Does the record state that tiers are equally weighted and name the consequence? Fail signal: a ranking whose weights are unstated but implied by its ordering. Owner: lead.

**RD-08 — Scenario honesty.** Are the three scenarios applied identically to all candidates, with the same multiplier? Fail signal: a candidate evaluated only in its favourable scenario. Owner: lead.

**RD-09 — Tail disclosure.** Does the record name correlated week loss as an unmodelled mechanism? Fail signal: the word "independent" absent from the assumption list. Owner: lead.

**RD-10 — Baseline valuation.** Is the deferral baseline assigned a number, with a reason? Fail signal: a baseline of "nothing" that silently equals the highest candidate. Owner: reviewer.

## 15. Quick Reference Card

| Field | Answer |
|---|---|
| Decision | Which of six candidates takes the term's 176 hours. |
| Owner and challenger | Programme lead selects; reviewer may reject a screen. |
| Grain and cut-off | One candidate, one term, one slot; frozen at plan commitment. |
| Baseline | Defer the slot at zero marginal credit. |
| Capacity | 22 weeks × 8 hours = 176 hours; effective effort `H(1 − r)`. |
| Screens | Feasibility with contingency, 25% reuse headroom, `E ≥ 2.0` tiers. |
| Ranking metric | `d = E·P_rel/H_eff`, averaged over base, reuse/2, effort ×1.15. |
| Selected | C6 at 0.0128667 tiers/h; C5 runner-up at 0.0124352. |
| Removed | C1 and C3 infeasible; C2 at 178 hours with contingency; C4 on 1.0-hour headroom. |
| Key arithmetic | 0.959319 release at 17 weeks; 0.074581 per lost week; 3.47% C6–C5 gap. |
| Weakest assumption | C6's 15%-overrun case is the lowest among survivors at 0.0074178. |
| Control | Week-11 checkpoint at 88 planned hours with three numeric readings. |

**Final status:** Draft selection case. The six candidates are constructed, no term was run, and the ranking is a decision from illustrative inputs rather than a measured result. The `PLAN-CAPSTONE-002` commitment is satisfied by this file; the remaining plan row stays proposed.
