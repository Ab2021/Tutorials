---
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "02"
topic: "Funnel, Cohort, and Retention Analysis"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Funnel stage definition, cohort entry rule, retention event definition, data cut-off, or censoring policy changes"
notation_style: "plain-text"
prerequisites: ["SQL aggregation and window functions", "ratio arithmetic", "set nesting"]
pattern_tags: ["business-domain-analytics"]
source_status: "OPEN"
owner_role: "analytics lead"
source_index: "INDEX.md"
---

# Funnel, Cohort, and Retention Analysis

## 1. File Header

> **Purpose:** Define, compute, and defend three linked measurements — ordered funnel conversion, cohort retention, and the hazard of leaving — so that a stakeholder can decide where to intervene and know what evidence would overturn the decision.
> **Prerequisites:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md) for outcome/driver/guardrail discipline; SQL `GROUP BY`, `COUNT(DISTINCT ...)`, window functions, and hash joins; comfort with products of ratios and with reading a triangular table.
> **Key outcome:** Given an event table, a cohort entry rule, and a data cut-off, you can produce a nested funnel whose stage counts are monotone, a cohort retention matrix with an explicitly marked maturity diagonal, a censoring-aware retention curve, a projected value for immature cells with its assumption stated, and a decision recommendation that survives a mix-shift and noise audit.
> **This file does NOT cover:** causal attribution of a conversion change to a shipped change (Module 08 experimentation), forecasting revenue beyond a retention-based lifetime estimate, and warehouse/dashboard implementation details (03 Dashboard Design and Operational Alerts).
> **Level:** 1 Foundations, with production-grade diagnostics in Sections 8–10.
> **Version / reviewed:** 2.0 / not independently reviewed; `status: draft`, review OPEN.
> **Evidence:** All mechanisms are DERIVED; every arithmetic result below is VERIFIED by recomputation; all population inputs are ILLUSTRATIVE and describe no real company, product, or campaign. No benchmark, incident, or regulatory finding is claimed. Statistical results are standard large-sample approximations, stated with their conditions.
> **Retrieval cue:** If someone says "conversion dropped" or "retention is up", start at Section 8 (mix shift and maturity are the two most common non-behavioural causes), then Section 4.4 for the invalidity conditions.

This sheet is about three artifacts that are usually drawn as pictures and therefore rarely defined precisely. The funnel is a nested-set count over an ordered event sequence. The cohort matrix is a two-dimensional tabulation indexed by entry time and elapsed time. The retention curve is a survival function estimated from that tabulation. Once each is stated as a set definition, the arithmetic becomes mechanical and the failure modes become checkable rather than a matter of taste.

A funnel answers "of the entities that started, how many completed each ordered step, and where is the largest recoverable loss". It is a within-attempt measurement: every entity is counted once, and the steps must be strictly ordered in time for each entity. A cohort answers "for the group that started together, what fraction is still present after h periods". It is a between-time measurement: it lets you separate "the product got worse" from "we acquired a different mix of users". Hazard answers "given that an entity has survived to h, what is the chance it leaves before h+1" — the derivative-like view that makes retention curves comparable when they start from different baselines.

The three artifacts share a single contract, and most analysis failures are violations of that contract rather than arithmetic mistakes. The contract has six clauses: the entity identity used for counting, the entry event and its timestamp, the ordering and window rules for stages, the definition of "still active" at horizon h, the cut-off timestamp that determines which cells are observable, and the decision the number is supposed to change. If any clause is unstated, two analysts can compute two defensible numbers from the same table and neither can be shown wrong.

**Takeaway:** a funnel is a nested-set count, a cohort matrix is an entry-time × elapsed-time tabulation, and a retention curve is a survival estimate from that tabulation — state all three as definitions before drawing any of them.

## 2. Concept Map / Mental Model

```text
                         [ decision: where to intervene, and is it working? ]
                                            |
              +-----------------------------+-----------------------------+
              |                             |                             |
              v                             v                             v
   [ FUNNEL: one attempt ]        [ COHORT: one entry group ]   [ HAZARD: conditional exit ]
   nested sets U_0 > U_1 > ...    rows = entry period c         lambda_h = 1 - S(h+1)/S(h)
   counts N_j, steps s_j          cols = elapsed period h       constant lambda -> geometric
              |                             |                             |
              v                             v                             v
   where is the loss?            is the group itself worse?      how fast do survivors leave?
              |                             |                             |
              +--------------+--------------+--------------+--------------+
                             |                             |
                             v                             v
                 [ maturity check: is cell observable? ]  [ mix check: did weights change? ]
                             |                             |
                    observable? no -> project / withhold   weights changed? yes -> standardize
                             |                             |
                             v                             v
                 [ uncertainty: interval, not point ]  [ decision + guardrail + reversal test ]
```

Read the diagram left to right as three questions of increasing time horizon, and bottom-up as two audits that must pass before any of the three answers is used in a decision. The funnel is instantaneous in the sense that it describes one attempt path; the cohort requires waiting; the hazard requires enough elapsed time that the ratio of adjacent survival values is estimable. The two audits — maturity and mix — are the ones that most often turn a headline number into a non-finding.

The mental model that makes the three fit together is a **state machine over entities**. An entity enters at stage 0, may transition forward through stages, and after a terminal stage either stays active or leaves. The funnel measures the forward transition probabilities within a fixed window. The cohort matrix measures the occupancy of the "active" state at each elapsed period. The hazard is the exit intensity out of that state. If you keep the state machine in mind, you will not confuse a funnel step rate (a conditional transition given the previous stage) with a cumulative conversion (a product of transitions), nor a retention rate (occupancy) with a churn rate (exit flow).

```text
per entity e:  enter(t0) -> s1 -> s2 -> s3 -> active
                                            |  \
                                            |   \--> dormant --(return)--> active
                                            +------> churned (absorbing, by definition of the metric)
```

Two modelling decisions are embedded in that diagram and both must be recorded. First, is "churned" absorbing, or can a dormant entity return and be counted as retained at a later horizon? Most retention definitions count *presence in period h*, which allows return; a strict churn definition makes exit absorbing and requires a different estimator. Second, is the entity key stable across devices and accounts? If the key changes, the same human appears as a new entrant and as a churn simultaneously, which inflates both the funnel's entry count and the churn count.

A third decision is the **observation window per entity**. If the funnel window is 30 days and a payment can arrive on day 45, the funnel undercounts late conversions and the undercount is systematic, not random: slower-converting segments look worse. The corresponding cohort decision is the maturity diagonal: cells on the diagonal are partially observed and cannot be compared to fully observed cells in the same column.

The concept map also shows where the three artifacts disagree productively. A funnel can look healthy while cohort retention falls, because the funnel only measures the first attempt and says nothing about whether the entity returns. A retention curve can look flat while the funnel degrades, because retention conditions on entities that already reached the value moment. When the two disagree, the disagreement is the finding: it localizes the problem to acquisition (funnel entry quality), activation (first-value step), or habit (post-value return).

**Takeaway:** hold all three artifacts as one state machine — forward transitions (funnel), state occupancy (cohort), and exit intensity (hazard) — and audit maturity and mix before believing any of them.

## 3. Foundational Definitions

Each definition below is written so that it can be turned directly into a `WHERE` clause. Where a term has a common informal meaning that differs from the operational one, the difference is called out explicitly, because most metric disputes are disputes about these clauses rather than about mathematics.

1. **Entity (unit of analysis).** The thing counted once per stage: a user, account, device, household, or order. Formal condition: a stable key exists that does not change across the measurement window, and every event in the table can be mapped to exactly one key. Nonexample: counting "sessions" as the entity for a 90-day retention question, because a returning user can create several sessions and the denominator then measures visits, not people.
2. **Event.** A timestamped record asserting that an entity performed an action, with a type and a server-side receipt time. Formal condition: the event has both an occurrence time (client or business time) and a receipt time, and the analysis names which one it uses. Nonexample: a row generated by a batch backfill that stamps all historical actions with the export date.
3. **Stage.** An ordered predicate on entity behaviour, `stage_j(e) = true` when entity `e` has performed the required action at least once inside the measurement window. Nonexample: a stage defined as "has ever paid", which makes a 30-day funnel silently cumulative over all history.
4. **Funnel.** A sequence of nested sets `U_0 ⊇ U_1 ⊇ ... ⊇ U_K` where `U_0` is the entrant set and `U_j` contains the entities that completed stages 1..j in order, within the window, with each completion timestamp strictly greater than the previous one. Nonexample: a chart of four unrelated counts labelled as a funnel, where an entity can appear at stage 3 without appearing at stage 2.
5. **Stage count `N_j`.** The cardinality `|U_j|` of the nested set at stage j. Formal condition: computed with distinct entity keys, not with event rows. Nonexample: `COUNT(*)` over stage events, which counts a double-submit as two conversions.
6. **Step conversion `s_j`.** The conditional ratio `N_j / N_{j-1}`: among entities that reached stage `j-1`, the fraction that reached stage j. Nonexample: `N_j / N_0`, which is the cumulative rate and answers a different question.
7. **Cumulative conversion `C_j`.** The product `s_1 · s_2 · ... · s_j`, equal to `N_j / N_0` when the sets are genuinely nested. Nonexample: the arithmetic mean of the step rates, which has no interpretation for a path.
8. **Drop-off at step j.** The count `N_{j-1} − N_j` and the ratio `1 − s_j`. Formal condition: reported with the stage pair named, because "drop-off of 50%" is ambiguous between a step and a cumulative rate. Nonexample: attributing all non-completion at step 1 to the step-1 user interface when part of that population had no intent to proceed.
9. **Cohort.** A set of entities grouped by a shared entry event and entry period, usually the calendar month or week of first qualifying action. Formal condition: membership is fixed at entry and never revised by later behaviour; an entity belongs to exactly one acquisition cohort by construction. Nonexample: a "cohort" defined as "users active in March", which is a cross-section and will move when behaviour changes.
10. **Cohort size `N(c,0)`.** The number of entities whose entry event falls in cohort period `c`, counted after de-duplication and after applying the eligibility filter. Formal condition: the denominator is frozen at entry; it does not grow when late-arriving events are reprocessed. Nonexample: recomputing cohort size each refresh, which makes historical retention rates move without any behaviour change.
11. **Elapsed period `h`.** The number of whole periods between cohort entry and the observation, indexed from 0 at entry. Formal condition: period boundaries are computed in the entity's entry-relative clock, so a cohort's period 1 means "the first 30 days after entry", not "the first calendar month". Nonexample: mixing calendar-month columns with entry-relative columns in one matrix.
12. **Retention `R(c,h)`.** The proportion `N(c,h) / N(c,0)` where `N(c,h)` counts cohort-`c` entities with at least one qualifying active event during elapsed period h. Nonexample: counting an entity as retained at h because it was retained at h−1, without evidence of activity in h.
13. **Active event.** The behaviour that defines "still present": an open, a session, a transaction, a login, a subscription renewal. Formal condition: chosen so that it cannot be triggered by the company itself on the entity's behalf. Nonexample: counting a marketing email open as user activity when the open is a proxy prefetch.
14. **Retention curve `S(h)`.** The retention proportion as a function of elapsed period, for a fixed cohort or a pooled set. Nonexample: a single number called "retention" without a horizon, since 30-day and 90-day retention differ in both magnitude and shape.
15. **Hazard (churn intensity) `λ_h`.** The conditional probability of leaving between h and h+1 given presence at h, `1 − S(h+1)/S(h)`. Nonexample: the unconditional churn rate `1 − S(h)`, which conflates early and late exit.
16. **Censoring.** A cell whose outcome is not yet observable at the data cut-off, because the entity's elapsed time has not reached h. Formal condition: censoring is recorded per entity, not inferred from a missing row. Nonexample: treating a not-yet-observable entity as churned, which makes recent cohorts look catastrophic.
17. **Maturity diagonal.** The set of cells `(c,h)` where `c + h` reaches the cut-off, so the cell is partially or fully observed. Formal condition: cells below the diagonal are complete; cells on it are partial; cells above it are empty and must be projected or suppressed.
18. **Chain-ladder factor `f_h`.** The pooled development ratio `Σ_c N(c,h+1) / Σ_c N(c,h)` computed over cohorts where both cells are mature. Nonexample: averaging per-cohort ratios with equal weight, which over-weights small cohorts.
19. **Mix shift.** A change in the composition of the measured population across segments or cohorts with different underlying rates. Formal condition: a shift is only a mix shift if segment-level rates are unchanged; the aggregate move is then an arithmetic consequence of weights. Nonexample: calling any unexplained aggregate movement "mix".
20. **Standardization.** Recomputing an aggregate rate using a fixed reference set of weights so that composition changes cannot move it. Nonexample: comparing a standardized rate to an unstandardized one and reporting the difference as an effect.
21. **Measurement window.** The closed time interval `[t_start, t_end)` over which stage and activity events are collected, expressed in a declared time zone. Nonexample: a window specified only by date, where the boundary hour silently differs between the event store and the reporting tool.
22. **Cut-off timestamp.** The instant at which the underlying tables were frozen for the analysis. Formal condition: every count is "as of" this instant and the instant travels with the number. Nonexample: a dashboard refreshed continuously, where two panels read different snapshots and cannot be reconciled.
23. **Survivorship artefact.** A spurious improvement created by conditioning on entities that survived an earlier filter. Nonexample: computing step conversion only among entities that completed the full funnel, which is by construction 100%.
24. **Guardrail.** A secondary measure that must not degrade for the primary improvement to count as a win, such as refund rate, support contacts, or revenue per retained entity.

Two definitional distinctions do most of the practical work and are worth stating separately. The first is **conditional versus cumulative**: `s_j` is conditional on arrival at `j-1` while `C_j` is cumulative from entry; reporting one while discussing the other is the single most common funnel error, and it changes the implied opportunity size by an order of magnitude. The second is **stock versus flow**: `R(c,h)` is a stock (presence during a period) while `1 − R(c,h)` accumulates flows; a curve that is flat in the stock can still hide a large flow because returns offset exits.

**Takeaway:** every definition here is a clause you can paste into SQL — entity key, entry event, active event, ordered stages, frozen denominator, recorded censoring — and a dispute you cannot resolve by pointing at one of those clauses is a dispute about the decision, not the metric.

## 4. Core Formulas with Full Derivation

The derivation strategy is uniform. First define the sets. Then express the count as a cardinality. Then use the nesting property to convert a cardinality ratio into a product of conditional ratios. Every formula below is a consequence of set nesting or of the product rule for conditional probabilities; nothing is assumed beyond what the `CONSTRAINT` line states.

### 4.1 Base measures

The base layer has two primitives: the step conversion (a conditional ratio within a funnel) and the period retention (a ratio within a cohort). Both are proportions with an explicit denominator, and both inherit their validity from the nesting and freezing conditions.

Validity before use: the funnel sets must be nested and computed on distinct entity keys inside one window with one time zone; the retention denominator must be frozen at cohort entry and the numerator must require an active event inside the elapsed period, not merely the absence of a churn event.

Derivation of the step conversion: let `U_{j-1}` be the set of entities that reached stage `j-1`, and `U_j ⊆ U_{j-1}` those that also reached stage j in order. Counting gives `N_{j-1} = |U_{j-1}|` and `N_j = |U_j|`. Because `U_j` is a subset, the conditional frequency of stage-j completion among arrivals is exactly the cardinality ratio. No independence assumption is used; the ratio is a definition, not an estimate, for a fixed window and population.

```text
[1] FORMULA [FC-01]: s_j = N_j / N_{j-1},   j = 1..K
[2] WHERE: N_j [entities] = |U_j|, distinct entity keys reaching stage j in order
           within the declared window; s_j is dimensionless, 0 <= s_j <= 1.
[3] CONSTRAINT: U_j is a subset of U_{j-1}; stage timestamps strictly increasing;
           one time zone; window identical for numerator and denominator;
           N_{j-1} > 0.
[4] NUMERICAL EXAMPLE: N_2 = 4,320 verified accounts, N_3 = 2,160 first-value
           accounts; s_3 = 2160/4320 = 0.5000.
[5] INTERPRETATION: half of the entities that verified went on to reach first value;
           the other half is the recoverable population for a step-3 intervention.
```

Boundary/counterexample: if `N_j > N_{j-1}` the sets are not nested — usually because stage order was not enforced in the query, or because stage j was computed on a wider window than stage `j-1`. The result is not a small error: a "conversion" above 1 is arithmetically possible and visibly wrong, while a milder violation (say 0.98 at a step where the truth is 0.72) is invisible without an explicit nesting check.

Validity before use for cumulative conversion: valid when the per-entity stage sequence is ordered and complete; if an entity can skip stage j and still reach j+1, the telescoping product still equals `N_K/N_0` only when the sets are nested, so either enforce nesting by defining `U_j` as "reached all stages up to j in order" or report the ratio of counts and label it a non-nested path rate.

Derivation: `C_K = N_K/N_0`. Insert the intermediate counts, each positive, to obtain a telescoping product. Each factor is a step conversion by [FC-01]. The product therefore equals the cumulative rate exactly, with no approximation, provided the sets are nested and the counts are of the same entity key.

```text
[1] FORMULA [FC-02]: C_K = N_K / N_0 = prod_{j=1..K} s_j
[2] WHERE: C_K dimensionless, the fraction of entrants reaching the terminal stage.
[3] CONSTRAINT: nested sets; identical window and entity key at every stage;
           all N_{j-1} > 0; K fixed before looking at the data.
[4] NUMERICAL EXAMPLE: s = (0.4500, 0.8000, 0.5000, 0.3000);
           0.45 * 0.80 = 0.3600; 0.36 * 0.50 = 0.1800; 0.18 * 0.30 = 0.0540;
           check 648 / 12000 = 0.0540. Agrees.
[5] INTERPRETATION: 5.40% of entrants pay, and that single number is fully
           reconstructible from the four step rates and the entrant count.
```

Boundary/counterexample: the mean of the four step rates is 0.5125, which is neither the cumulative rate nor any interpretable quantity. Reporting "average step conversion 51%" next to "end-to-end conversion 5.4%" invites the reader to conclude that the funnel is mostly healthy, when the product says the opposite.

Validity before use for retention: valid when cohort membership is frozen at entry, the numerator counts an active event inside the elapsed period, and the denominator counts all cohort members who have reached that elapsed period. If some members have not yet reached h, the ratio is a partial-cohort rate and must be labelled as such.

Derivation: fix cohort `c` with frozen size `N(c,0)`. Let `A(c,h)` be the subset of that cohort with at least one active event in elapsed period h. Then `N(c,h) = |A(c,h)|` and the empirical proportion of the cohort present in h is the cardinality ratio. This is a direct frequency estimate for the cohort; it is not an estimate of a population parameter unless the cohort is treated as a sample.

```text
[1] FORMULA [FC-03]: R(c,h) = N(c,h) / N(c,0)
[2] WHERE: N(c,0) [entities] frozen at entry; N(c,h) [entities] active in period h;
           R dimensionless in [0,1]; h = 0,1,2,... elapsed periods.
[3] CONSTRAINT: entry-defined cohort; active event defined independently of company
           action; period h observable for all members (else partial-cohort rate);
           cut-off recorded.
[4] NUMERICAL EXAMPLE: January cohort N(c,0) = 1,000; N(c,3) = 410;
           R(Jan,3) = 410/1000 = 0.410.
[5] INTERPRETATION: 41.0% of the January entrants were active in their fourth month;
           the complement includes both churned entities and temporary absentees.
```

Boundary/counterexample: if cohort size is recomputed after a backfill adds 40 late-recorded entrants, the same cohort's `R(c,3)` becomes `410/1040 = 0.3943`, a 1.6 percentage-point drop caused entirely by denominator revision. Freezing is not a stylistic preference; it is what makes the historical row comparable to itself.

Validity before use for the pooled retention: valid when pooling is intended to describe the combined population and the weights are the cohort entry sizes. The pooled rate is a weighted mean and is not equal to the unweighted mean of cohort rates whenever cohort sizes differ and rates differ.

Derivation: let cohorts `c = 1..C` have sizes `n_c` and period-h active counts `a_c`. The combined population's period-h rate is `Σ a_c / Σ n_c`. Writing `a_c = R_c n_c` gives `Σ R_c n_c / Σ n_c`, a size-weighted mean of cohort rates. The algebra is exact; the interpretive content is that the weights are part of the metric.

```text
[1] FORMULA [FC-04]: R_pooled(h) = sum_c w_c R(c,h),  w_c = N(c,0) / sum_k N(k,0)
[2] WHERE: w_c dimensionless weights summing to 1; R(c,h) cohort rates; pooling
           restricted to cohorts with cell h mature.
[3] CONSTRAINT: only mature cells enter; cohort sizes frozen; pooling across
           cohorts is a deliberate descriptive choice, not a substitute for trend.
[4] NUMERICAL EXAMPLE: two mature cohorts, (n=1,000, R=0.32) and (n=1,500, R=0.40):
           R_pooled = (1000*0.32 + 1500*0.40)/2500 = (320 + 600)/2500 = 0.368.
           Unweighted mean = (0.32 + 0.40)/2 = 0.360.
[5] INTERPRETATION: the 0.8 pp gap between pooled and unweighted is pure weighting;
           reporting either without the weight is a reporting defect, not a rounding
           difference.
```

Boundary/counterexample: pooling a mature cohort with an immature one mixes a complete estimate with a partial one. If the immature cohort contributes `R = 0.20` because only half of its members have reached h, the pooled value falls even though no entity behaved differently. Maturity must be checked before pooling, not after the number looks wrong.

**Takeaway:** all four base measures are cardinality ratios — the step is conditional on arrival, the cumulative is their exact telescoping product, retention is frozen-denominator presence, and pooling is a size-weighted mean whose weights are part of the meaning.

### 4.2 General comparison

The general layer converts the base ratios into a comparable object across cohorts and across time. Four formulas carry most of the weight: the cohort matrix as a table of counts, the hazard derived from adjacent survival values, the censoring-adjusted survival estimate, and the projection of immature cells.

Validity before use for the hazard: valid when the retention definition counts presence in a period (a discrete-time stock), so that the discrete hazard is the complement of the ratio of adjacent survival values. If churn is absorbing and the data are recorded as event times, a continuous-time estimator is required instead, and the discrete ratio is biased by the period length.

Derivation: by the chain rule of conditional probability, presence at horizon `h+1` requires presence at h and non-exit during h. Let `S(h) = P(T >= h)` with `T` the number of complete periods present. Then `P(T >= h+1 | T >= h) = S(h+1)/S(h)`, so the conditional exit probability is one minus that ratio. Substituting the empirical retention values gives the plug-in estimator. When `S(h)` is estimated from a cohort, the ratio inherits the cohort's sampling variability, which is why adjacent-value ratios are noisy in small cells.

```text
[1] FORMULA [FC-05]: lambda_h = 1 - S(h+1)/S(h),   S(h) = R(c,h) for a fixed cohort
[2] WHERE: lambda_h dimensionless in [0,1]; h = 0,1,2,...; S(h) > 0 for the ratio
           to exist; S(0) = 1 by construction when period 0 is entry.
[3] CONSTRAINT: discrete-time presence definition; exit absorbing within the period;
           equal period lengths across the curve being compared; no re-entry
           between h and h+1 uncounted (else the stock understates the flow).
[4] NUMERICAL EXAMPLE: S(3) = 0.4100, S(4) = 0.3280;
           lambda_3 = 1 - 0.3280/0.4100 = 1 - 0.8000 = 0.2000.
[5] INTERPRETATION: among entities still present in month 3, 20.0% left during
           month 4; the unconditional 4-month churn of 67.2% is a different number.
```

Boundary/counterexample: if `S(4) = 0.3280` comes from a cohort with 40% of members censored before month 4, the ratio silently divides by an over-complete `S(3)` and understates the hazard, because censored entities are treated as present. That is the censoring failure formalized in [FC-07].

Validity before use for a cohort matrix: valid when each row is a frozen cohort and each column is elapsed period, with cell values as counts (not rates) so that pooling and projection remain possible. Rates cannot be re-aggregated without their denominators, so a matrix stored as percentages is a dead end for every downstream computation.

Derivation: the matrix is a two-way tabulation of the same entity-level indicator. For each entity `e` in cohort `c(e)`, define `y_{e,h} = 1` if active in elapsed period h. Then `N(c,h) = Σ_{e: c(e)=c} y_{e,h}` and `N(c,0) = Σ_{e: c(e)=c} 1`. Column sums give period-aligned active counts, row sums give cohort lifetime activity, and the diagonal identifies maturity. No extra statistics are needed; every derived quantity in this sheet is a function of this matrix.

```text
[1] FORMULA [FC-06]: N(c,h) = sum_{e in c} y_{e,h},  R(c,h) = N(c,h)/N(c,0)
[2] WHERE: y_{e,h} in {0,1} per entity-period; c indexes entry periods; h indexes
           elapsed periods; N in entities; R dimensionless.
[3] CONSTRAINT: one row per entity; y computed from an explicitly defined active
           event; period boundaries in entry-relative time; cut-off applied before
           any cell is read.
[4] NUMERICAL EXAMPLE: cohort of 1,000 with month-4 actives 328 -> N(c,4)=328,
           R = 0.3280; month-3 actives 410 -> R = 0.4100; row is internally
           consistent because 328 <= 410 <= 1000 is not required (returns allowed)
           but 328 and 410 are each <= 1000.
[5] INTERPRETATION: storing counts preserves every recombination; storing only
           percentages destroys the ability to pool, project, or test.
```

Boundary/counterexample: a matrix whose rows are "users active in month m" rather than "users who entered in month m" is a cross-section. Its diagonal will look stable even when every cohort is deteriorating, because each row is a different mixture of cohorts at different tenures. The tell is that the first column is not approximately the acquisition series.

Validity before use for censoring-adjusted survival: valid when censoring is independent of the exit process conditional on time — the standard non-informative censoring assumption. If entities are censored precisely because they are about to churn (for example, an account is closed and its rows stop), the estimator is biased upward.

Derivation sketch (Kaplan–Meier, discrete case): at each period h, risk set `n_h` counts entities present at the start of h and not previously censored; `d_h` counts exits during h; censored entities `u_h` leave the risk set at the period boundary. The conditional survival factor is `1 − d_h/n_h`, and the survival estimate is the product of factors over periods. In the no-censoring case with a full cohort, this collapses to `S(h) = N(c,h)/N(c,0)`.

```text
[1] FORMULA [FC-07]: S_KM(h) = prod_{k=0..h-1} (1 - d_k / n_k),  S_KM(0) = 1
[2] WHERE: n_k [entities] at risk at start of period k; d_k exits in period k;
           u_k censored at end of k, with n_{k+1} = n_k - d_k - u_k;
           S_KM dimensionless.
[3] CONSTRAINT: non-informative censoring; exits observable within the period;
           consistent risk-set construction; single exit per entity.
[4] NUMERICAL EXAMPLE: n_0=1000, d_0=600, u_0=0 -> S(1)=0.4000;
           n_1=400, d_1=80 -> S(2)=0.4000*(320/400)=0.3200;
           n_2=320, d_2=64 -> S(3)=0.3200*(256/320)=0.2560;
           n_3=256, d_3=51 -> S(4)=0.2560*(205/256)=0.2050.
           Cross-check: 205/1000 = 0.2050 exactly, since u=0 throughout.
[5] INTERPRETATION: with roughly a 20% conditional exit each period, 20.5% remain
           after four periods; the curve is geometric, not linear, and a linear
           extrapolation would have predicted 20.0% at h=4 and 0% at h=5, both wrong.
```

Boundary/counterexample: suppose in the same cohort 300 entities were censored at the end of period 0 (they were acquired through a channel that stops reporting). Then `n_1 = 100` and `S_KM(1) = 0.40` still, but `d_1` is measured against 100 rather than 400, so `S_KM(2) = 0.40 · (1 − 80/100) = 0.08` instead of 0.32 if exits did not scale down. The naive cohort ratio would have reported `320/1000 = 0.32`, overstating survival by a factor of four. The lesson is that censoring must be modelled, not dropped.

Validity before use for chain-ladder projection: valid when the development pattern is stable across cohorts — each `f_h` is the same for old and new cohorts — so that the pooled factor estimated on mature cohorts can be applied to the immature tail. When the pattern is changing (a recent product change altered the curve shape), pooling old cohorts to project new ones propagates the old pattern forward and the projection becomes a statement about history rather than about the new cohort.

Derivation: assume a multiplicative development model `E[N(c,h+1)] = f_h E[N(c,h)]` with `f_h` independent of `c`. Summing over cohorts where both cells are mature gives `Σ N(c,h+1) = f_h Σ N(c,h)`, so the method-of-moments estimator is the ratio of column sums. Cumulative survival to `h+m` from `h` is the product of the intervening factors; applying it to an immature cell gives the projected value with its assumption exposed.

```text
[1] FORMULA [FC-08]: f_h = sum_{c: (c,h+1) mature} N(c,h+1) / sum_{c: (c,h+1) mature} N(c,h)
                     S_hat(c,h+m) = N(c,h) * prod_{k=h..h+m-1} f_k
[2] WHERE: f_h dimensionless development factor; S_hat projected active count;
           sums restricted to cohorts mature at h+1.
[3] CONSTRAINT: stable development pattern across cohorts; maturity determined by
           the cut-off; projection labelled as projection and never mixed into the
           observed matrix without a marker.
[4] NUMERICAL EXAMPLE: mature pairs at h=4 -> column sums 328 and 262 ->
           f_4 = 262/328 = 0.7988. Immature cohort at h=4 has N = 900;
           projected N(c,5) = 900 * 0.7988 = 718.9 -> 719 entities (round at the
           end, not inside the product).
[5] INTERPRETATION: the mature cohorts lost 20.1% between periods 4 and 5; if that
           pattern holds, the new cohort retains 719 of 2,000 entrants at h=5,
           i.e. 35.95%, but that is an assumption-dependent projection.
```

Boundary/counterexample: if the product change that shipped last quarter flattened the tail, the pooled `f_4` mixes pre-change and post-change cohorts and is a weighted average of two patterns, matching neither. The check is to compute `f_h` separately for the two regimes; if the two differ by more than the interval width, pooling is invalid and the projection must be restricted to post-change cohorts or withheld entirely.

**Takeaway:** the hazard exposes the conditional exit rate, the count matrix preserves everything downstream, Kaplan–Meier repairs censoring, and chain-ladder makes the projection's stability assumption explicit — none of the four is optional once a retention number is used to size an intervention.

### 4.3 Variants

Variants differ in how the window and the ordering are defined, and each changes the number materially. The four that matter most in practice are the unbounded-lookback funnel, the strict-window funnel, N-day versus unbounded retention, and bracket retention.

**Unbounded-lookback funnel.** Each stage is "has ever done the action", with ordering enforced by first-occurrence timestamps and no per-entity window. Validity: appropriate for lifetime path analysis, where the question is "how many eventually reach value", not "how many reach it quickly". Derivation: identical to [FC-01] but with `U_j` defined over all history up to the cut-off; ordering comes from comparing first-occurrence times rather than from a fixed window.

```text
[1] FORMULA [FC-09]: N_j^unb = |{ e : min t_j(e) > min t_{j-1}(e) > ... > t_0(e) }|
[2] WHERE: t_j(e) the first occurrence timestamp of stage j for entity e; min over
           all occurrences within history; counts are entities.
[3] CONSTRAINT: history complete back to each entity's true entry; no truncated
           retention of old events by the event store (a real constraint — some
           systems expire raw events after 12–24 months).
[4] NUMERICAL EXAMPLE: 12,000 entrants over all history; 7,200 ever verified
           (0.6000); 6,480 reached first value after verifying at any time
           (s_3 = 0.9000); 3,888 reached the value event (s_4 = 0.6000); 1,361 paid
           (s_5 = 0.3500). C_5 = 1361/12000 = 0.1134.
[5] INTERPRETATION: unbounded funnels measure eventual attainment; here the
           eventual pay rate (11.34%) is more than double the 30-day pay rate
           (5.40%), so the two cannot appear in one comparison table unlabelled.
```

Boundary/counterexample: with event-store expiry at 18 months, entities older than expiry have their `t_0` recomputed from the earliest surviving event, which can make them appear to enter later and reorder stages. The check is to compare `N_0^unb` against the acquisition table; a shortfall indicates truncation.

**Strict-window funnel.** The window resets at each attempt or at each entity's entry, and a stage only counts if it falls within `W` of the previous stage. Validity: appropriate for operational diagnosis of a single session or trial. Derivation: replace the global ordering with a per-entity constraint `t_j − t_{j-1} <= W`; the set is then nested by construction because completion of stage j implies arrival at j−1.

```text
[1] FORMULA [FC-10]: U_j^W = { e : exists t_0 < t_1 < ... < t_j, with t_k - t_{k-1} <= W }
[2] WHERE: t_k occurrence timestamps for entity e; W the per-step window in
           declared time units; U_j^W nested in U_{j-1}^W by construction.
[3] CONSTRAINT: a single consistent W per step (a different W per step is allowed
           but must be published per step); server-side timestamps preferred to
           client clocks, which drift and lie.
[4] NUMERICAL EXAMPLE: 12,000 entrants; W = 7 days. N_1 = 8,400 -> s_1 = 0.7000;
           N_2 = 5,880 -> s_2 = 0.7000; N_3 = 4,116 -> s_3 = 0.7000;
           N_4 = 2,470 -> s_4 = 0.6001; C_4 = 0.2058.
[5] INTERPRETATION: with a constant 70% step rate, cumulative conversion decays
           geometrically; a fourth step at 60% is the visibly worse step and the
           one whose interface deserves the experiment.
```

Boundary/counterexample: widening W from 7 to 30 days can only increase every `N_j` and hence every `s_j` (superset). A "conversion improvement" produced by widening W is a definition change; the reversal test is to recompute the prior period under the new W and compare like with like.

**N-day (bounded) versus unbounded retention.** N-day retention asks "active on the specific day or in the specific period N"; unbounded retention asks "active at any time at or after N". Validity: N-day measures habit at a specific cadence; unbounded measures ever-returned and is monotone non-increasing in N by construction in the opposite direction — it can only be at least as large as N-day.

```text
[1] FORMULA [FC-11]: R_Nday(c,N) = |{e in c : active in exactly period N}| / N(c,0)
                     R_unb(c,N)  = |{e in c : active in some period >= N}| / N(c,0)
[2] WHERE: both dimensionless; N-day is not monotone in N; unbounded is
           non-increasing in N by set inclusion.
[3] CONSTRAINT: period length fixed; "exactly" means at least one active event in
           the period, not activity confined to a single day unless the period is a
           day; comparisons must use the same family.
[4] NUMERICAL EXAMPLE: 1,000 entrants. Active in period 3 only: 120; in 4 or
           later but not 3: 150. R_Nday(3) = 0.120; R_unb(3) = (120 + 150)/1000 =
           0.270. R_Nday(4) = 0.080; R_unb(4) = 0.190.
[5] INTERPRETATION: the N-day series can rise (users who skipped period 3 return in
           4) while the unbounded series cannot; a report that shows N-day rising is
           not evidence that more users are retained overall.
```

Boundary/counterexample: comparing a competitor's published "90-day retention" against your N-day-30 figure is a family mismatch, and it is not correctable by scaling. State the family before comparing, or do not compare.

**Bracket retention.** Retention measured as presence in any of a range of periods, used when the active cadence is irregular (a monthly invoice paid on variable dates). Validity: appropriate when the natural period is unclear; it smooths cadence noise at the cost of resolution. Derivation: replace the indicator with an OR over the bracket `[h, h+m)`, then divide by the frozen cohort size and by nothing else — the bracket width must be fixed across compared cells.

```text
[1] FORMULA [FC-12]: R_bracket(c, h..h+m-1) = |{e in c : active in any period in the bracket}| / N(c,0)
[2] WHERE: bracket width m fixed for all compared cells; h the bracket start.
[3] CONSTRAINT: m constant across the comparison; brackets non-overlapping if
           summed; cohort size frozen.
[4] NUMERICAL EXAMPLE: 1,000 entrants; active in period 4 or 5 or 6: 330;
           R_bracket = 0.330, against N-day-5 of 0.190 and unbounded-at-5 of 0.250.
[5] INTERPRETATION: bracket retention sits between the two families and is the
           honest choice for irregular cadence, provided every cell uses m = 3.
```

Boundary/counterexample: widening the bracket raises retention mechanically. Two teams using brackets of different widths will disagree by construction; the width belongs in the metric name, for example `R_3mo`.

**Cohort-weighted versus period-weighted aggregation.** The pooled rate [FC-04] weights by cohort size. A period-weighted view weights each calendar period equally, which answers "what did a typical period look like" rather than "what did the combined population do". Validity: choose one and state it; the two differ whenever cohort sizes trend, which is the normal case for a growing product.

```text
[1] FORMULA [FC-13]: R_period(h) = (1/C) * sum_c R(c,h)   over mature cohorts
[2] WHERE: C number of mature cohorts contributing; each R(c,h) equally weighted.
[3] CONSTRAINT: only mature cells; C large enough that the mean is stable; used
           for describing a typical cohort, never for sizing revenue.
[4] NUMERICAL EXAMPLE: three mature cohorts at h=3 with R = 0.24, 0.30, 0.42 and
           sizes 5,000, 1,500, 800.
           R_period = (0.24+0.30+0.42)/3 = 0.3200.
           R_pooled = (5000*0.24+1500*0.30+800*0.42)/7300
                    = (1200+450+336)/7300 = 1986/7300 = 0.2721.
[5] INTERPRETATION: the period-weighted figure (32.0%) is 4.8 pp higher than the
           pooled figure (27.2%) because the largest cohort is the worst one;
           quoting the period-weighted number to management overstates the base.
```

Boundary/counterexample: if cohort sizes are growing, period-weighting over-represents small early cohorts and understates the current population's retention; if shrinking, the bias reverses. The directional argument is checkable from the acquisition series alone.

**Takeaway:** window and family are part of the metric's name — an unbounded funnel, a 7-day strict funnel, N-day retention and 3-month bracket retention are four different measurements and none of them can be converted into another by a fixed multiplier.

### 4.4 Limits and counterexamples

The limits below are the boundary conditions under which the formulas stop being valid. Each is stated as a condition, a counterexample with numbers, and the repair. The goal is that a reader who knows only these four paragraphs can reject most bad funnel and retention analyses on sight.

**Limit 1: non-nesting.** The step formula requires `U_j ⊆ U_{j-1}`. A counterexample: an analyst computes stage 2 as "opened the app in the last 30 days" and stage 3 as "completed a purchase in the last 30 days" without requiring the purchase to follow the open. An entity that bought on day 1 and opened on day 28 is counted at stage 3 but has no ordered stage-2 event before it, so `N_3` and `N_2` overlap without containment. The repair is to construct each `U_j` as a cumulative predicate over the ordered sequence (`exists t_1 < ... < t_j`), which guarantees nesting by definition, and to assert `N_j <= N_{j-1}` as a query-level test that fails loudly.

**Limit 2: censoring treated as churn.** The retention formula requires that every member of the denominator has had the opportunity to be observed in period h. A counterexample: a 90-day-old cohort evaluated at h = 3 (months 3 to 4). Members acquired late in the cohort's calendar month may have only 60 days of elapsed time, so they cannot possibly be active in period 3; counting them in `N(c,0)` depresses `R(c,3)` by the fraction not yet observable. With 1,000 entrants and 250 not yet observable, an observed 410 actives among the 750 observable gives a true rate of `410/750 = 0.5467` but a naive `410/1000 = 0.4100` — a 13.7 pp error that would be read as a collapse. The repair is to restrict the denominator to members who have reached the period, report that denominator alongside the rate, and mark the cell as partial.

**Limit 3: mix shift.** The comparison of two aggregates is valid only if composition is held fixed or the difference is decomposed. A counterexample: the 30-day retention rate is 42.0% in month 1 and 44.0% in month 2, apparently an improvement. Segment rates: paid users 60.0% in both months, free users 30.0% in both months. Weights: month 1 was 40% paid, month 2 was 46.67% paid. Then `0.40·0.60 + 0.60·0.30 = 0.42` and `0.4667·0.60 + 0.5333·0.30 = 0.44`; every segment rate is identical. The repair is standardization: recompute both months at a fixed weight vector (say the month-1 mix) — `0.40·0.60 + 0.60·0.30 = 0.42` for both months, so the entire 2 pp "gain" is composition. Report the standardized series and the mix decomposition together.

**Limit 4: small cells and noise.** A retention cell is a proportion estimated from a finite denominator; at small denominators the interval is wide and the ranking of segments is unstable. A counterexample: segment A has 40 entrants and 10 actives at h=1 (`R = 0.25`), segment B has 2,000 entrants and 460 actives (`R = 0.23`). The standard error of A is `sqrt(0.25·0.75/40) = 0.0685`, so its 95% interval is roughly `[0.116, 0.384]`, which contains B's point estimate comfortably. Declaring A "better" is unsupported. The repair is to publish intervals, suppress or pool cells below a stated denominator threshold (with the threshold justified by the decision's tolerance), and never rank segments on point estimates alone.

A fifth limit deserves separate mention because it is a validity rather than a precision problem: **the active event must not be confounded with the treatment**. If the intervention sends a push notification and the active event is "opened the app", the treated group's retention rises mechanically while the underlying behaviour is unchanged. The repair is to define an activity event that the intervention cannot itself generate, or to measure retention on a lagged window that excludes the notification's effect window.

```text
Invalidity summary
  non-nesting          -> s_j can exceed 1, or be biased upward invisibly
  censoring as churn   -> recent cohorts and immature cells understated
  mix shift            -> aggregate moves with unchanged segment behaviour
  small cells          -> segment ranking unstable; interval omitted
  treatment-generated  -> retention measures the prompt, not the behaviour
```

**Takeaway:** each limit has a single assertion that catches it — `N_j <= N_{j-1}`, denominator equals observable population, standardized weights reported, interval width published — and those four assertions are cheaper to run than a review meeting.

## 5. Worked Numerical Example

This section uses **one internally consistent dataset** so that every figure can be traced from the raw counts through to the decision. The per-formula examples in Section 4 were small, self-contained illustrations; the numbers below are a single connected case. All inputs are ILLUSTRATIVE — a fictional subscription product with no real company behind it — while every arithmetic step is VERIFIED by recomputation and cross-checked two ways.

**The setting.** A subscription product with a five-step signup path. The analysis answers one question: *should the team spend the next quarter on the activation step or on the referral step?* The cut-off is 2026-09-30 end of day, all timestamps in UTC, entity key is `account_id` (verified to be stable for the window), stages ordered by server receipt time, and the funnel window is 30 days per step edge.

### Step 0: Declare the contract before touching the data

| Clause | Value for this example | Consequence if left unstated |
|---|---|---|
| Entity key | `account_id` | Counting `session_id` would multiply the entry count and inflate conversion |
| Entry event | `signup_completed` | Using "first visit" would include bounces with no intent |
| Active event | `core_action_completed` | Using "app open" would be generated by our own push notifications |
| Stage window | 30 days per step edge, strict order | Unbounded lookback would raise every rate |
| Cohort calendar | monthly entry cohorts, entry-relative elapsed periods | Calendar-month columns would break the diagonal |
| Cut-off | 2026-09-30 23:59:59 UTC | Without it the maturity diagonal is unknowable |

The active event is chosen so the company cannot generate it. `app_open` is rejected explicitly because the retention intervention in this product is a push notification; measuring retention by app opens would score the notification, not the behaviour.

### Step 1: Build the nested funnel and check nesting

Month-1 entrance cohort, 12,000 accounts, 30-day window per edge.

| Stage | Predicate | Count `N_j` | Step rate `s_j` | Cumulative `C_j` |
|---|---|---:|---:|---:|
| 1 `N_0` | signed up | 12,000 | — | 1.0000 |
| 2 `N_1` | verified email | 8,400 | 0.7000 | 0.7000 |
| 3 `N_2` | completed profile | 6,720 | 0.8000 | 0.5600 |
| 4 `N_3` | reached first value | 4,032 | 0.6000 | 0.3360 |
| 5 `N_4` | invited a teammate | 2,016 | 0.5000 | 0.1680 |
| 6 `N_5` | converted to paid | 1,512 | 0.7500 | 0.1260 |

Arithmetic, shown in full because the telescoping is the point:

```text
s_1 = 8400/12000 = 0.7000
s_2 = 6720/8400  = 0.8000
s_3 = 4032/6720  = 0.6000
s_4 = 2016/4032  = 0.5000
s_5 = 1512/2016  = 0.7500

C_2 = 0.7000 * 0.8000 = 0.5600          check 6720/12000 = 0.5600  ok
C_3 = 0.5600 * 0.6000 = 0.3360          check 4032/12000 = 0.3360  ok
C_4 = 0.3360 * 0.5000 = 0.1680          check 2016/12000 = 0.1680  ok
C_5 = 0.1680 * 0.7500 = 0.1260          check 1512/12000 = 0.1260  ok
```

Nesting assertions, run as a query-level test rather than by eye:

```text
N_0 >= N_1 >= N_2 >= N_3 >= N_4 >= N_5  ->  12000 >= 8400 >= 6720 >= 4032 >= 2016 >= 1512  PASS
each N_j computed with COUNT(DISTINCT account_id), not COUNT(*)                                  PASS
each stage event timestamp strictly greater than the previous stage's for the same account        PASS
```

The last assertion is the one that silently fails in production. If the query takes "has a profile" from a denormalized `profiles` table without comparing timestamps, an account that created its profile during onboarding and verified a week later is counted at stage 3 without an ordered stage 2, and `N_3` can exceed `N_2` in a subpopulation even while the aggregate looks plausible.

### Step 2: Do not average step rates; use the product

The mean of the five step rates is `(0.70 + 0.80 + 0.60 + 0.50 + 0.75)/5 = 3.35/5 = 0.6700`. Reporting "67% average step conversion" alongside "12.6% end-to-end" invites a reader to think the path is two-thirds healthy. The two numbers describe different objects: the mean is an average of conditional transitions, the product is the joint probability of completing all of them. Only the product sizes the paid population.

A useful diagnostic is the **step-loss share**: which stage contributes most of the total loss?

```text
loss at step j (absolute)  = N_{j-1} - N_j
loss share at step j       = (N_{j-1} - N_j) / (N_0 - N_5)

step 1: 12000 - 8400 = 3600    share 3600/10488 = 0.3432
step 2:  8400 - 6720 = 1680    share 1680/10488 = 0.1602
step 3:  6720 - 4032 = 2688    share 2688/10488 = 0.2563
step 4:  4032 - 2016 = 2016    share 2016/10488 = 0.1922
step 5:  2016 - 1512 =  504    share  504/10488 = 0.0481
total: 12000 - 1512 = 10488     shares sum to 1.0000
```

Check the shares sum: `0.3432 + 0.1602 + 0.2563 + 0.1922 + 0.0481 = 1.0000`. ✓

Absolute loss says step 1 is the biggest hole (3,600 accounts). But absolute loss is the wrong ranking for intervention, because recovering accounts at step 1 still leaves them four conditional gates away from revenue. The right ranking weights each recovery by the downstream conversion remaining.

### Step 3: Compute leverage, not raw loss

Define leverage for a step as the expected paid accounts gained per one-percentage-point absolute improvement in that step's rate:

```text
leverage_j = 0.01 * N_{j-1} * prod_{k>j} s_k

step 1: 0.01 * 12000 * (0.80*0.60*0.50*0.75) = 120.0 * 0.1800 = 21.60
step 2: 0.01 *  8400 * (0.60*0.50*0.75)      =  84.0 * 0.2250 = 18.90
step 3: 0.01 *  6720 * (0.50*0.75)           =  67.2 * 0.3750 = 25.20
step 4: 0.01 *  4032 * (0.75)                =  40.32* 0.7500 = 30.24
step 5: 0.01 *  2016 * 1                     =  20.16* 1.0000 = 20.16
```

Verification of step 3 by the direct route: raise `s_3` from 0.6000 to 0.6100. New `N_3 = 6720 * 0.61 = 4099.2`; delta `= 67.2`; new `N_4 = 4099.2 * 0.5 = 2049.6`; new `N_5 = 2049.6 * 0.75 = 1537.2`; delta payers `= 1537.2 - 1512 = 25.2`. Matches the leverage figure of 25.20. ✓

Verification of step 4: raise `s_4` from 0.5000 to 0.5100. New `N_4 = 4032 * 0.51 = 2056.32`; delta `40.32`; new `N_5 = 2056.32 * 0.75 = 1542.24`; delta `30.24`. ✓

The ranking is `step 4 (30.24) > step 3 (25.20) > step 1 (21.60) > step 5 (20.16) > step 2 (18.90)`. The stage with the largest absolute loss (step 1) ranks third, and the stage with the smallest absolute loss (step 5) ranks fourth — close to the top. Raw loss ranking and leverage ranking are genuinely different instruments and answering the wrong question wastes a quarter.

Now convert leverage to a decision quantity with an explicitly illustrative price of $29 per paid account per month and 12 expected paid months:

```text
value per 1 pp at step 4 = 30.24 * 29 * 12 = 30.24 * 348 = $10,523.52
value per 1 pp at step 3 = 25.20 * 29 * 12 = 25.20 * 348 = $ 8,769.60
value per 1 pp at step 1 = 21.60 * 29 * 12 = 21.60 * 348 = $ 7,516.80
```

If the referral redesign is expected to move step 4 by 5 pp and the activation redesign is expected to move step 3 by 8 pp, the comparison is `5 * 30.24 = 151.2` versus `8 * 25.20 = 201.6` paid accounts, so activation wins on this arithmetic despite lower leverage per point. This is the shape of the real decision: leverage per point is not the same as opportunity size, and both must be estimated before ranking projects.

### Step 4: Build the cohort matrix in counts

Six monthly cohorts, cut-off 2026-09-30. Cells hold **counts**, not percentages, so that pooling, projection, and interval construction remain possible.

| Cohort | Size `N(c,0)` | h=0 | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan | 10,000 | 10,000 | 5,200 | 4,000 | 3,400 | 3,050 | 2,830 | 2,680 |
| Feb | 11,000 | 11,000 | 5,720 | 4,400 | 3,740 | 3,355 | 3,113 | 2,948 |
| Mar | 12,000 | 12,000 | 6,720 | 5,340 | 4,704 | 4,320 | 4,080 | 3,936 |
| Apr | 12,000 | 12,000 | 6,720 | 5,340 | 4,704 | 4,320 | 4,080 | — |
| May | 13,000 | 13,000 | 7,280 | 5,785 | 5,096 | 4,680 | — | — |
| Jun | 14,000 | 14,000 | 7,840 | 6,230 | 5,488 | — | — | — |

The em-dashes above the diagonal are **not zeros**. They are unobservable cells: the June cohort cannot have a month-4 measurement because only three elapsed months exist at the cut-off. Writing a zero there and averaging the column would report a collapse that did not happen.

`R(c,0) = 1.0000` by declared convention: the entry event (`signup_completed`) is itself a qualifying active event in period 0. State this convention whenever it is used, because a reader who assumes `h=0` means "still active in the entry month" will misread every row.

### Step 5: Convert the matrix to rates and read the diagonal

| Cohort | h=0 | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jan | 1.0000 | 0.5200 | 0.4000 | 0.3400 | 0.3050 | 0.2830 | 0.2680 |
| Feb | 1.0000 | 0.5200 | 0.4000 | 0.3400 | 0.3050 | 0.2830 | 0.2680 |
| Mar | 1.0000 | 0.5600 | 0.4450 | 0.3920 | 0.3600 | 0.3400 | 0.3280 |
| Apr | 1.0000 | 0.5600 | 0.4450 | 0.3920 | 0.3600 | 0.3400 | — |
| May | 1.0000 | 0.5600 | 0.4450 | 0.3920 | 0.3600 | — | — |
| Jun | 1.0000 | 0.5600 | 0.4450 | 0.3920 | — | — | — |

Spot-check two cells from the counts: `R(Mar,3) = 4704/12000 = 0.3920` ✓; `R(May,2) = 5785/13000 = 0.4450` ✓. Two regimes are visible: Jan–Feb at 0.3400 at h=3, and Mar–Jun at 0.3920. The step from 0.3400 to 0.3920 is a 5.2 pp improvement that persists across four consecutive cohorts, which is much stronger evidence than a single cohort moving.

This is also the correct way to look for a product change: a single cohort improving could be a campaign or a season; four consecutive cohorts improving at the same elapsed horizon, with the same measurement definition, is consistent with a durable change. It is still not causal evidence — Section 9 and the Module 08 handoff cover that.

### Step 6: Compute the hazard and check the curve shape

For the January cohort (`S(h) = R(Jan,h)`):

```text
lambda_0 = 1 - 0.5200/1.0000 = 0.4800
lambda_1 = 1 - 0.4000/0.5200 = 1 - 0.769231 = 0.230769
lambda_2 = 1 - 0.3400/0.4000 = 1 - 0.850000 = 0.150000
lambda_3 = 1 - 0.3050/0.3400 = 1 - 0.897059 = 0.102941
lambda_4 = 1 - 0.2830/0.3050 = 1 - 0.927869 = 0.072131
lambda_5 = 1 - 0.2680/0.2830 = 1 - 0.947000 = 0.053004
```

For the March cohort:

```text
lambda_0 = 1 - 0.5600 = 0.4400
lambda_1 = 1 - 0.4450/0.5600 = 1 - 0.794643 = 0.205357
lambda_2 = 1 - 0.3920/0.4450 = 1 - 0.880899 = 0.119101
lambda_3 = 1 - 0.3600/0.3920 = 1 - 0.918367 = 0.081633
lambda_4 = 1 - 0.3400/0.3600 = 1 - 0.944444 = 0.055556
lambda_5 = 1 - 0.3280/0.3400 = 1 - 0.964706 = 0.035294
```

Both hazards fall monotonically, which is the signature of a **mixture of stayers and leavers**: the leavers exit early and the residual population is increasingly composed of durable accounts. This shape is why a constant-hazard (geometric) model is wrong here, and why a linear extrapolation of the retention curve is badly wrong: January's curve loses 48% in the first period and only 2.2 pp of the original cohort between h=4 and h=5.

A shape check that costs one line: compare the hazard at h=0 with the hazard at the last mature period. `0.4800 / 0.053004 = 9.06` for January. A ratio near 1 means a constant-hazard product (extrapolate geometrically); a ratio of 5–20 means a mixture (do not extrapolate the early slope); a rising hazard means a cohort-quality or contract-expiry problem and should be investigated rather than modelled.

### Step 7: Repair an immature cell by hand

Take the July cohort, which entered after the six above and is partially mature at the cut-off.

```text
July entrants:                          20,000
  entered July 1-15, fully exposed:     15,000
  entered July 16-31, partially exposed: 5,000
  of the partially exposed, h=2 observable:      3,000
  of the partially exposed, h=2 not observable:  2,000

observed actives in h=2 (all entrants):   8,010

naive denominator:                     20,000
naive R(Jul,2) = 8010/20000          = 0.4005
observable denominator: 15,000 + 3,000 = 18,000
corrected R(Jul,2) = 8010/18000      = 0.4450
```

The naive figure understates the rate by `0.4450 - 0.4005 = 0.0445`, i.e. 4.45 pp — comparable to the entire genuine improvement across the whole matrix. A reader comparing naive July (0.4005) to March (0.4450) would conclude retention had collapsed; the corrected comparison (0.4450 vs 0.4450) shows July is exactly on the March pattern.

Rule for reporting: every cell computed on a restricted denominator must print **both** the rate and the denominator it used, plus a `partial` flag. A rate without its denominator cannot be distinguished from a rate computed on a different population.

### Step 8: Project the immature tail with chain-ladder, and show the invalid version first

Invalid version, pooling all cohorts with mature `h=3 -> h=4` pairs (Jan–May):

```text
sum N(c,4) over Jan..May = 3050 + 3355 + 4320 + 4320 + 4680 = 19,725
sum N(c,3) over Jan..May = 3400 + 3740 + 4704 + 4704 + 5096 = 21,644
f_3(pooled) = 19725/21644 = 0.911338
projected N(Jun,4) = 5488 * 0.911338 = 5,001.4  ->  5,001
```

Valid version, restricting to cohorts in the same regime (Mar–May):

```text
sum N(c,4) over Mar..May = 4320 + 4320 + 4680 = 13,320
sum N(c,3) over Mar..May = 4704 + 4704 + 5096 = 14,504
f_3(regime B) = 13320/14504 = 0.918367
projected N(Jun,4) = 5488 * 0.918367 = 5,040.0  ->  5,040
```

Continue for the next two horizons using the same regime restriction:

```text
f_4(regime B) from Mar,Apr: (4080+4080)/(4320+4320) = 8160/8640 = 0.944444
projected N(Jun,5) = 5040 * 0.944444 = 4,760.0  ->  4,760
f_5(regime B) from Mar only: 3936/4080 = 0.964706
projected N(Jun,6) = 4760 * 0.964706 = 4,592.0  ->  4,592
```

Resulting projected rates: `R(Jun,4) = 5040/14000 = 0.3600`; `R(Jun,5) = 4760/14000 = 0.3400`; `R(Jun,6) = 4592/14000 = 0.3280`. Because the generating curve is the same regime-B curve, the projection reproduces `S_B(4) = 0.3600`, `S_B(5) = 0.3400`, `S_B(6) = 0.3280` exactly. That is a **consistency check on the arithmetic and the maturity bookkeeping — not evidence about the future.** A real projection whose inputs come from a stable regime will still miss if the regime changes, and the honest label on these cells is "projection, assumption: regime-B development persists".

The gap between the two versions is `5040 - 5001 = 39` accounts at h=4, about `39/14000 = 0.28 pp` of the June cohort. Small here, because regime A and B differ modestly in tail development. When a product change affects exactly the tail — a contract-length change, a paywall move — the pooled factor can be wrong by many points, and the regime split is not optional.

### Step 9: Mix decomposition — is the conversion gain real?

Aggregate 30-day conversion rose from 12.60% to 17.00% between month 1 and month 3. Channel detail:

| Month | Paid entrants | Paid conv. | Organic entrants | Organic conv. | Aggregate |
|---|---:|---:|---:|---:|---:|
| M1 | 4,800 (40%) | 0.2400 | 7,200 (60%) | 0.0500 | 0.1260 |
| M3 | 7,500 (50%) | 0.2800 | 7,500 (50%) | 0.0600 | 0.1700 |

Check the aggregates from the channel cells:

```text
M1 paid conversions    = 4800 * 0.2400 = 1152
M1 organic conversions = 7200 * 0.0500 =  360
M1 total               = 1152 + 360 = 1512;  1512/12000 = 0.1260   matches the funnel N_5
M3 paid conversions    = 7500 * 0.2800 = 2100
M3 organic conversions = 7500 * 0.0600 =  450
M3 total               = 2100 + 450 = 2550;  2550/15000 = 0.1700
```

The funnel in Step 1 is exactly the M1 aggregate, which is why the two sections reconcile: `N_5 = 1,512` is the sum of the two channel conversion counts.

Standardization at fixed M1 weights (40% paid / 60% organic):

```text
rate-only counterfactual: 0.40*0.2800 + 0.60*0.0600 = 0.1120 + 0.0360 = 0.1480
total change:             0.1700 - 0.1260 = 0.0440
genuine (rate) component: 0.1480 - 0.1260 = 0.0220   (50.00% of the change)
mix component:            0.1700 - 0.1480 = 0.0220   (50.00% of the change)
```

Two-way decomposition, both orderings, because the split depends on which weights you fix and the reader is entitled to know:

```text
Form A (fix base weights for the rate effect):
  rate effect = 0.40*(0.28-0.24) + 0.60*(0.06-0.05) = 0.0160 + 0.0060 = 0.0220
  mix  effect = 0.28*(0.50-0.40) + 0.06*(0.50-0.60) = 0.0280 - 0.0060 = 0.0220
  sum = 0.0440   ok

Form B (fix new weights for the rate effect):
  rate effect = 0.50*(0.28-0.24) + 0.50*(0.06-0.05) = 0.0200 + 0.0050 = 0.0250
  mix  effect = 0.24*(0.50-0.40) + 0.05*(0.50-0.60) = 0.0240 - 0.0050 = 0.0190
  sum = 0.0440   ok
```

Both orderings reproduce the total exactly (the identity holds because each adds and subtracts the same interaction term), but they allocate 22/22 versus 25/19. **The decomposition is not unique.** Publish the form used. If the two forms materially disagree — as they do at this scale, a 3 pp difference in the mix attribution — report the range rather than a single attribution, or use a symmetric allocation (Shapley-style average) and say so.

### Step 10: Put an interval on the comparison

Proportions from finite cohorts carry sampling error. For the retention comparison January versus March at h=3:

```text
R(Jan,3) = 0.3400, n = 10,000
  SE = sqrt(0.3400*0.6600/10000) = sqrt(2.2440e-5) = 0.004737
R(Mar,3) = 0.3920, n = 12,000
  SE = sqrt(0.3920*0.6080/12000) = sqrt(1.9861e-5) = 0.004457
difference = 0.0520
SE(diff) = sqrt(2.2440e-5 + 1.9861e-5) = sqrt(4.2301e-5) = 0.006504
z = 0.0520/0.006504 = 7.995
95% CI on the difference = 0.0520 +/- 1.96*0.006504 = [0.0393, 0.0647]
```

Under independent cohorts — reasonable here because they are disjoint sets of accounts — the improvement is far outside sampling noise. For the conversion comparison (two different populations, not nested):

```text
p1 = 0.1260, n1 = 12,000;  p3 = 0.1700, n3 = 15,000
SE(p1) = sqrt(0.1260*0.8740/12000) = sqrt(9.1770e-6) = 0.003029
SE(p3) = sqrt(0.1700*0.8300/15000) = sqrt(9.4067e-6) = 0.003067
SE(diff) = sqrt(9.1770e-6 + 9.4067e-6) = 0.004311
difference = 0.0440;  z = 0.0440/0.004311 = 10.21
95% CI = [0.0356, 0.0524]
```

The total change is unambiguous. The standardized change (0.0220) inherits additional variance from the estimated weights, so its interval is wider than the naive scaling suggests; the honest statement is "at least half of the observed gain is composition and the remainder is a within-channel improvement whose interval is not computed here." That sentence is less satisfying than a single number and it is the correct one.

What an interval does **not** fix: it does not repair a definition change, a nesting violation, or censoring. An interval is a statement about sampling from a fixed measurement process; a broken process produces a tight interval around the wrong number.

### Step 11: Convert all of it into the decision

```text
Decision: allocate next quarter's analytics-supported effort to one of
          (a) activation step 3 redesign, (b) referral step 4 redesign.
Inputs (all ILLUSTRATIVE): leverage per 1 pp = 25.20 and 30.24 paid accounts;
          expected effects 8 pp and 5 pp; $29/month; 12 expected paid months.
(a) 8 * 25.20 = 201.6 paid accounts -> 201.6 * 348 = $70,156.80
(b) 5 * 30.24 = 151.2 paid accounts -> 151.2 * 348 = $52,617.60
Ranking flips if (b) reaches 6.67 pp: 6.67 * 30.24 = 201.7, matching (a).
  Breakeven on (b): 201.6/30.24 = 6.6667 pp.
Guardrail: refund rate and support contacts per 1,000 paid accounts must not rise;
  if either rises by more than its stated tolerance, the intervention is reverted
  regardless of the paid-account gain.
Reversal test: if the post-launch step-3 rate moves less than 2 pp (the lower edge
  of a plausible detection threshold for a 30-day window), the intervention is
  declared not established and the next quarter's ranking is recomputed.
```

The breakeven calculation is the part that survives scrutiny. Because the two options differ in leverage by a known factor, the decision only turns on the effect-size estimate: `(a)` wins unless `(b)`'s effect exceeds `8 * 25.20/30.24 = 6.6667` pp. Stating the condition rather than the conclusion lets the reader re-run the decision when the effect estimate changes.

**Takeaway:** the worked example shows the whole chain — nesting check, step product, leverage ranking, count matrix, hazard shape, censoring repair, projection, mix decomposition, interval, decision — and every stage changed either the ranking or the confidence in it, so no stage is decorative.

## 6. Visual / Geometric Intuition

Four pictures do almost all the intuition work: the nested-band funnel, the cohort triangle with its maturity diagonal, the curve shapes that distinguish hazard regimes, and the leverage rectangle. Each is drawn below with the invariant it encodes, because a diagram that does not encode an invariant is decoration.

### 6.1 The funnel as narrowing bands, not as a pipe

```text
entrants          12000 ################################################
verified           8400 #################################
profile            6720 ##########################
first value        4032 ###############
invited            2016 #######
paid               1512 #####

band widths above are proportional to N_j; the visual narrowing is the STEP rate
between adjacent bands, not the cumulative rate from the top.
```

The geometric content is that each band is **contained** in the one above it. Any drawing where a lower band is wider than the one above it is showing a non-nested count set, and the picture is lying about the arithmetic. When you see a funnel chart with a bulge, the correct response is not "interesting" but "the stages were computed on incomparable populations".

Two readings of the same picture answer different questions, and confusing them is the most common funnel error:

```text
cumulative reading (from the top):     paid/entrants = 1512/12000 = 12.60%
step reading (from the band above):    paid/invited  = 1512/2016 = 75.00%
```

The step reading is the useful one for a product team, because it isolates the gate they can change. The cumulative reading is the useful one for planning, because it sizes the funnel's yield per entrant. A slide that shows "12.6% conversion" next to a discussion of the invitation screen is mixing the two: the invitation screen controls the 75%, and it cannot move conversion by more than `(1 − 0.75) · 0.168 = 4.2 pp` in absolute terms even if every non-inviter converts.

### 6.2 The cohort triangle and the maturity diagonal

```text
elapsed period h ->      0      1      2      3      4      5      6
cohort c
  Jan  (oldest)       [#####][#### ][###  ][##   ][#    ][#    ][#    ]
  Feb                 [#####][#### ][###  ][##   ][#    ][#    ]  ...
  Mar                 [#####][#### ][###  ][##   ][#    ]  ...    ...
  Apr                 [#####][#### ][###  ][##   ]  ...    ...    ...
  May                 [#####][#### ][###  ]  ...    ...    ...    ...
  Jun  (newest)       [#####][#### ][###  ]  ...    ...    ...    ...
                              \
                               \___ the maturity frontier: everything to the
                                    right is UNOBSERVABLE, not zero
```

The triangle is a **Lexis diagram** in disguise: cells are `(entry time, elapsed time)`, and the cut-off is a straight anti-diagonal across it. Along that anti-diagonal, `entry time + elapsed time = cut-off`, so all cells on one anti-diagonal were observed at the same instant. Cells to the right of the frontier have no data at all.

Consequences that follow from the geometry alone, without any modelling:

- Any column average must use only the rows that reach that column. Averaging the visible cells of column 6 (Jan and Feb only) gives a different population from column 3 (all six rows).
- Any row comparison at a fixed `h` compares cohorts of different ages. That is the point of the matrix: `R(Mar,3)` and `R(Jan,3)` are both 3-month measurements taken 2 months apart in calendar time.
- Reading **down** a column mixes cohort quality; reading **across** a row mixes tenure. A "retention is falling" conclusion from a down-column read is often a statement about which cohorts are currently mature, not about behaviour.
- The frontier moves one cell per period, so a matrix refreshed monthly adds exactly one anti-diagonal. A refresh that adds two means the cut-off jumped; a refresh that adds none means a backfill.

A second geometric fact: because the frontier is a single line, the number of mature cells is exactly `C · (H+1) − C(C−1)/2` for `C` cohorts and `H+1` columns. With `C=6, H=6`: `6·7 − 15 = 27` mature cells out of 42 positions. Everything else must be projected or suppressed, and knowing the count in advance prevents a table from being quietly padded with zeros.

### 6.3 Retention curve shapes and what they identify

```text
S(h)
1.00 |*
     | *
0.80 |  *
     |   *
0.60 |    *
     |      *
0.40 |        *  *  *
     |                 *  *  *        <- mixture: fast early loss, then a floor
0.20 |
     |
0.00 +----------------------------------> h
       0  1  2  3  4  5  6  7  8  9

shape           diagnostic meaning                     what to do
--------------- -------------------------------------- ---------------------------
pure geometric  constant hazard, single population     extrapolate from one point
mixture         falling hazard, durable core           fit two components, do not extrapolate the early slope
linear          implausible for a stock; usually a     suspect a definition change or a censored denominator
                censored denominator artefact
floor then drop cliff at a contract/notice boundary    align analysis to contract terms, not calendar months
rising          re-engagement or a metric that counts   check the active event and the denominator before modelling
                returns
```

The most useful transform is the **log-survival plot**. A constant hazard makes `log S(h)` linear in h with slope `log(1−λ)`. A curved log-survival plot falsifies the constant-hazard model more clearly than the raw curve, because small hazards compress near 1 on the raw scale.

```text
log S(h) for the January cohort (natural log):
  h=0: 0.0000
  h=1: ln 0.5200 = -0.6539
  h=2: ln 0.4000 = -0.9163
  h=3: ln 0.3400 = -1.0788
  h=4: ln 0.3050 = -1.1874
  h=5: ln 0.2830 = -1.2621
  h=6: ln 0.2680 = -1.3167

successive slopes: -0.6539, -0.2624, -0.1625, -0.1086, -0.0747, -0.0546
```

The slopes shrink by roughly a factor of 0.4–0.7 each period, which is a curved line on the log scale and therefore a decisively non-constant hazard. Fitting a line to the first two points and extrapolating would predict `ln S(6) = -0.6539 - 5·0.2624 = -1.9659`, i.e. `S(6) ≈ 0.140`, whereas the observed value is 0.268 — nearly a factor of two too pessimistic. **Extrapolating a mixture curve with an early-period slope is the single most common retention forecasting error.**

### 6.4 The leverage rectangle

```text
paid accounts gained per 1 pp at step j
   |                    +--------+
30 |                    |  30.24 |  step 4  (4032 downstream x 0.75)
   |            +-------+        |
25 |            | 25.20 |        |  step 3  (6720 downstream x 0.375)
   |    +-------+       |        |
20 |    | 21.60 |       |        |
   |    |       |       |        |
15 |    |       | 18.90 |        |
   |    |       |       |        |
10 |    |       |       |        |
   |    |       |       |        |
 5 |    |       |       |        |
   |    |       |       |        |
 0 +----+-------+-------+--------+------> step
          1       2       3     4     5
                              (20.16)
```

The rectangle's height is `0.01 · N_{j-1} · Π_{k>j} s_k`, which is a product of an upstream count and a downstream survival. Geometrically, it is the area of the "reachable" region: how many entities are still in play at the step, times how many of them will make it through everything after. A step late in the funnel has a high downstream survival but a small upstream count; an early step has the reverse. The product peaks somewhere in the middle, and the peak location is where leverage-based prioritisation and absolute-loss-based prioritisation disagree most.

A useful corollary from the shape: **the maximum possible gain from any single step** is bounded by `N_{j-1} · Π_{k>j} s_k`, achieved only if `s_j` goes to 1. For step 1 that ceiling is `12000 · 0.18 = 2,160` paid accounts; for step 4 it is `4032 · 0.75 = 3,024`. Step 4 has a higher ceiling than step 1 despite having far fewer entrants, because step 1's downstream survival is only 18%. This is arithmetic, not a hypothesis, and it is derivable from the matrix alone.

### 6.5 Why the two audits look like geometry

Mix shift and censoring both appear as geometric distortions rather than numeric anomalies, which is why plots catch them faster than summary tables.

```text
MIX SHIFT: two segments, one aggregate line that moves without either segment moving

rate
0.30 |                     organic --------- (flat at 0.06)
0.25 |        paid   ------------- (flat at 0.24->0.28)
0.20 |
0.15 |          aggregate ...... (0.126 -> 0.170)  <- moves because weights moved
0.10 |
     +------------------------------------------------
            M1                    M3

CENSORING: the newest rows appear to fall off a cliff

S(h)
0.60 |  ** ** ** **
0.40 |             ** ** **
0.30 |                      **
0.20 |                        \   <- artificial slope: denominator includes
0.10 |                         \     entrants with no exposed time
     +--------------------------------
        Jan Feb Mar Apr May Jun
```

Both distortions have the same repair shape: **fix the population and re-plot**. For mix, hold weights constant and the aggregate line flattens if segment lines are flat. For censoring, restrict the denominator to exposed entities and the artificial cliff disappears. Neither repair requires new data; both require knowing what the picture is supposed to be invariant to.

**Takeaway:** each diagram encodes one invariant — containment for the funnel, the anti-diagonal frontier for the matrix, curvature of log-survival for the hazard regime, and the downstream product for leverage — and a chart that cannot display its invariant cannot be audited.

## 7. Comparison / Selection Table

Selection questions in this domain are usually framed as tool questions ("should we use a funnel or a cohort?") when they are really population questions ("what population and horizon does the decision require?"). The tables below are ordered from the most common selection to the least, and each row names the condition under which the option is invalid, because a selection table without disqualifiers invites cargo-cult use.

### 7.1 Which funnel variant

| Variant | Use when | Avoid when | Disqualifier | Fallback |
|---|---|---|---|---|
| Strict-window funnel (`W` per step) | Diagnosing a single attempt or trial; conversion is expected within days | The natural path is long and variable (enterprise sales, mortgage) | `W` chosen to make a number look familiar | Unbounded funnel, labelled as lifetime attainment |
| Unbounded-lookback funnel | Sizing eventual attainment; lifetime path analysis | Comparing periods or teams; the event store truncates history | History incomplete (expired raw events) | Restrict to the truncated window and say so |
| Session-scoped funnel | Measuring within one visit; UI-level diagnosis | Cross-session journeys (research → purchase weeks later) | Session definition changes across platforms | Entity-scoped strict funnel |
| Reverse funnel | Diagnosing a terminal stage (why do payers not renew) | Early-stage acquisition questions | Terminal event is not well defined | Forward funnel on the complement population |
| Non-nested stage rate | Never, as a funnel; only as a population share | Any question about a path | A "funnel" whose counts are not monotone | Rebuild with ordered predicates |

The disqualifier column is the operative part. A strict funnel with `W` chosen after seeing the numbers is not a measurement; it is a choice of answer. The defensible procedure is to fix `W` from the domain cadence (how long the action genuinely takes, measured on a pilot), publish it in the metric name, and never widen it silently.

### 7.2 Which retention family

| Family | Definition | Use when | Avoid when | Reading hazard |
|---|---|---|---|---|
| N-day / N-period | Active in exactly period `N` | The cadence is regular and the question is "are they still in the habit" | Cadence is irregular; users legitimately skip periods | Can rise; not a survival curve |
| Unbounded (ever after N) | Active in any period `>= N` | The question is "did they ever come back" | You need to distinguish habit from a single return | Monotone non-increasing in N |
| Bracket (any of `h..h+m-1`) | Active anywhere in a window | Irregular cadence; smoothing sampling noise | Comparing across different bracket widths | Depends on `m`; declare it |
| Rolling / DAU-style | Active in the last `k` days, measured daily | Operational monitoring; near-real-time alerts | Cohort-level inference; overlapping windows | Overlapping windows bias the level |
| Classic (day-N exact) | Active on day `N` precisely | Daily-cadence consumer products | Weekly or monthly cadence products | Very noisy at small cohorts |
| Return-rate | Re-activated after a gap | Measuring resurrection campaigns | Mixing with continuation retention | Not a survival function at all |

The critical selection rule is that **families cannot be mixed within one series**. A dashboard that shows "retention" as N-day for the first three months and unbounded thereafter produces a curve that rises at the switch point, which readers will interpret as recovery. The correct construction is to pick one family, publish its name in the metric title, and keep alternate families in a separate diagnostic view.

### 7.3 Cohort versus cross-section versus period-aligned

| View | Rows defined by | Answers | Fails when |
|---|---|---|---|
| Entry cohort | Entry period (first qualifying action) | "Is a group of users, followed from the same starting point, doing better?" | Entry definition changes; cohort sizes tiny |
| Behavioural cohort | A behaviour (used feature X in week 1) | "Does behaviour X predict retention?" | The behaviour is a consequence of retention, not a cause |
| Cross-section | Active in a period | "How many users did we have?" | Any question about change over time; composition drives it |
| Period-aligned | Calendar period, all users | "What was the aggregate rate last month?" | Mix changes; it is not a retention estimate |
| Tenure-aligned | All users at the same tenure | "What does a typical month-3 user look like?" | Cohort quality differs; hides cohort effects |

The behavioural cohort row is the one most often misused. Defining cohorts by "used feature X in week 1" and then finding they retain better is close to tautological when using feature X requires being active, and is confounded when the feature is adopted by already-engaged users. The design that makes it informative is a pre-period definition: behaviour measured strictly before the retention window starts, with the feature adoption timestamp required to precede the first retained period.

### 7.4 Aggregation choice

| Aggregation | Weights | Use | Failure |
|---|---|---|---|
| Pooled (size-weighted) | `N(c,0)` | Sizing the population; revenue impact | Dominated by the largest cohort; hides cohort trend |
| Period-weighted (equal cohorts) | `1/C` | Describing a typical cohort | Over-weights small cohorts; not a population rate |
| Standardized | Fixed reference mix | Comparing across periods with mix change | The reference itself must be published |
| Decomposed | Explicit rate + mix terms | Explaining a move | Non-unique; must name the decomposition form |
| Segment-first | Segment weights held, then summed | Deciding which segment to act on | Segment definitions drift |

The rule of thumb that follows: **size decisions on pooled rates, trend judgements on per-cohort series, and comparisons between periods on standardized rates.** Using one number for all three purposes guarantees that at least two of the three will be wrong.

### 7.5 Active-event selection

| Candidate active event | Measures | Good when | Bad when |
|---|---|---|---|
| Login / app open | Access | Access is the value | Push notifications or prefetch generate it |
| Core action completed | Value delivery | The action is what the user came for | Very rare events produce sparse cells |
| Session with ≥ 2 core actions | Habit | Depth matters more than presence | Punishes legitimate light users |
| Paid transaction | Commercial value | Revenue is the objective | Misses engaged non-payers; sparse for low-frequency businesses |
| Any of several actions | Broad presence | The product has multiple value paths | Dilutes the signal; hides path-specific decline |
| Composite score above threshold | Engagement intensity | You need a single ordering | The threshold becomes a tuning knob |

A defensible default is the **core action**, because it is the behaviour the product promises and it is usually not generatable by the company. The moment the active event can be triggered by a marketing send, retention becomes a measure of the marketing calendar.

### 7.6 Diagnostic sequencing from a symptom

| Symptom | First check | Second check | Third check | Typical true cause |
|---|---|---|---|---|
| Conversion fell at one step | Definition/dimension change | Stage nesting and window | Segment mix | Instrumentation or eligibility change |
| Conversion fell at all steps | Entry population change | Time zone / cut-off | Genuine product regression | Acquisition mix shifted |
| Retention fell for the newest cohort only | Maturity/censoring | Cohort size and interval | Genuine behaviour | Immature cells counted as churn |
| Retention fell for all cohorts | Active-event definition | Instrumentation | Genuine behaviour | Event logging change |
| Retention rose without a launch | Denominator shrink | Active event loosened | Data loss in the event pipeline | Fewer entities observable counted as retained |
| Cohort matrix rows disagree wildly | Cohort size 20 vs 2,000 | Statistical interval | Selection rule | Small-cohort noise |
| Retention curve has a cliff | Contract/notice boundary | Billing cycle alignment | Genuine step change | Calendar-month alignment against contract terms |

Note the ordering principle: **definitional causes come before behavioural causes in every row.** This is not cynicism, it is base rates. In most organisations, instrumentation and definition changes are more frequent than genuine large behavioural shifts, and they are far cheaper to check.

**Takeaway:** select by population and horizon, not by chart type; publish the variant in the metric's name; and sequence diagnosis definitional-first, because instrumentation changes outnumber real behaviour changes.

### 7.7 Where each artifact belongs in a decision

| Decision type | Primary artifact | Supporting artifact | Never sufficient alone |
|---|---|---|---|
| Where to intervene in a path | Leverage-ranked funnel | Segment breakdown of the target step | A single conversion percentage |
| Whether a cohort is degrading | Hazard by cohort | Interval on the difference | Pooled retention |
| Whether to keep a channel | Cohort retention by channel, pooled to contribution | Payback by channel | First-purchase conversion |
| Whether an intervention worked | Pre/post same-cohort comparison with a holdout | Guardrail metrics | Post-only retention |
| How long to keep investing | Retention curve tail with a fitted regime | Hazard ratio to a benchmark cohort | Early-period slope extrapolation |

**Takeaway:** the artifact must match the decision's horizon — funnels for path choices, hazards for cohort quality, tails for investment duration — and the wrong artifact produces a confident answer to a question nobody asked.

## 8. Failure Modes & Edge Cases

Each entry below is written so it can be used as a diagnostic: the symptom is what a reader observes, the competing causes are the ones that plausibly produce that symptom, the discriminating check is a single query or comparison whose result separates them, the fix is bounded (it does not require rebuilding the pipeline), and the verification is how you know the fix worked. Provenance is HYPOTHETICAL teaching material unless a label says otherwise; none of these describes an observed incident at a real company.

### 8.1 Non-nested stage counts

**Symptom.** A step rate is close to 1 where the domain expects 0.6–0.7, or a stage count exceeds the one before it.

**Competing causes.** (a) Stage predicates were computed independently over a fixed lookback window without ordering, so an entity that did action B before A is still counted at B. (b) Different windows per stage, with a wider window downstream. (c) Different entity keys per stage (`account_id` upstream, `user_id` downstream) so the sets are not comparable at all. (d) A join fan-out inflated the downstream count through duplicate dimension rows.

**Discriminating check.** Run the assertion `N_j <= N_{j-1}` for every j and, separately, count entities in `U_j` with no ordered predecessor:

```text
SELECT COUNT(DISTINCT e.account_id)
FROM   stage_j e
WHERE  NOT EXISTS (SELECT 1 FROM stage_{j-1} p
                   WHERE p.account_id = e.account_id
                     AND p.ts < e.ts);
```

A nonzero result isolates cause (a). A zero result with unequal windows points to (b); differing `COUNT(DISTINCT)` values across stages on the same day points to (c); comparing `COUNT(*)` to `COUNT(DISTINCT key)` isolates (d).

**Bounded fix.** Rebuild each `U_j` as a cumulative ordered predicate (`EXISTS t_1 < ... < t_j`) and materialize the stage timestamps into one per-entity table with explicit `stage1_ts ... stageK_ts` columns. The fan-out is repaired by de-duplicating the entity table before joining.

**Verification.** Nesting assertion passes for three consecutive refreshes; a hand check of 20 randomly sampled entities shows the ordered timestamps are consistent; the step rates land in the domain-plausible band and the change is explained.

### 8.2 Censoring counted as churn

**Symptom.** The newest one or two cohort rows fall off sharply, while mature cohorts are stable. The cliff's steepness correlates with how recent the cohort is.

**Competing causes.** (a) Genuine decline in a recent acquisition cohort. (b) Denominator includes entities with no exposed time in the period. (c) A change in the entry event that admits a different population this month. (d) Partial-period recording (the current period is only 60% elapsed).

**Discriminating check.** Recompute the last row with an exposure-restricted denominator and compare:

```text
full-denominator rate   = actives_in_h / N(c,0)
exposed-denominator rate = actives_in_h / N_exposed(c,h)
```

If the second is materially higher and close to the mature-cohort rate, cause (b) or (d) dominates. Cause (a) survives both corrections and would also show a lower rate among exposed entities. Cause (c) shows up as a step in the entry-day distribution: plot entrants by day-of-month within the cohort; a spike near the cohort boundary means the definition changed.

**Bounded fix.** Compute `N_exposed(c,h)` from entry timestamps and the cut-off, restrict the denominator, and mark the cell `partial`. For the current period, either wait or pro-rate with the exposure fraction and label it.

**Verification.** The corrected recent row lands within its interval of the mature rows; the `partial` flag appears in the published table; a reader can reproduce the exposure denominator from the published entry-timestamp rule.

### 8.3 Mix shift presented as behaviour change

**Symptom.** An aggregate rate moves while every segment rate is flat, or the aggregate moves in the opposite direction from most segments (Simpson's reversal).

**Competing causes.** (a) Weights changed (acquisition mix, channel spend shift). (b) A segment definition changed, so entities moved between buckets. (c) Genuine differential improvement concentrated in a large segment. (d) A new segment appeared with no history, absorbing part of the population.

**Discriminating check.** Standardize to a fixed weight vector and recompute both periods. If the standardized difference is near zero while the raw difference is not, cause (a) dominates. Then recompute segment rates under the old and new segment definition; a rate that moves when only the definition moved isolates (b). For (d), check whether a bucket labelled "unknown" or "other" changed size.

**Bounded fix.** Publish the standardized series alongside the raw one, with the reference weights named. Add a mix-decomposition line to the report rather than an explanation in prose. Freeze segment definitions in a versioned dimension table.

**Verification.** Standardized and raw series are both present; the decomposition sums to the total change under the published form; the reference weight vector is dated.

### 8.4 Active-event contamination by the intervention

**Symptom.** Retention improves after a notification or email campaign, then decays back over the following weeks, with no change in revenue.

**Competing causes.** (a) Genuine re-engagement caused by the campaign. (b) The active event is generated by the campaign itself (open, click, prefetch). (c) Selection: the campaign targeted users who were already returning. (d) Seasonal return independent of the campaign.

**Discriminating check.** Recompute retention using an active event the campaign cannot generate (a core action, a transaction, a session containing the core action) and compare. If the improvement disappears, cause (b) dominates. For (c), compare against a matched non-targeted segment over the same weeks; a pre-existing upward trend in the control isolates (c). For (d), check the same weeks in the prior year.

**Bounded fix.** Replace the active event with a non-generatable one and re-state the retention definition with a version bump. If the campaign's effect on the core action is the real question, measure that action with a holdout.

**Verification.** The redefined series is stable across the campaign window; revenue per retained entity does not diverge from the retained count; the metric version and its effective date are recorded.

### 8.5 Denominator revision (backfill and late records)

**Symptom.** Historical retention values change on refresh, with no announcement; a row that read 0.410 earlier reads 0.394.

**Competing causes.** (a) Late-arriving entry events added to the cohort. (b) Late-arriving activity events added to a numerator. (c) A re-run of an upstream model changed entity identity. (d) A time-zone correction moved boundary events between periods.

**Discriminating check.** Persist cohort sizes daily and compare. A changed `N(c,0)` isolates (a) or (c). An unchanged denominator with changed numerators isolates (b) or (d). For (d), recompute with the prior time-zone rule; if historical rows move, the rule changed.

**Bounded fix.** Freeze cohort membership as a materialized snapshot at a stated maturity (for example, `T+7` days after the cohort period closes), then never recompute it. Record late entries in a separate "late additions" column so the effect is visible rather than silent.

**Verification.** Historical rows are byte-identical across two refreshes; the late-additions column accounts for the difference between the live entry count and the frozen size.

### 8.6 Immature-cell projection treated as observation

**Symptom.** A projected cell appears in the same table as observed cells, with no marker, and is quoted in a decision.

**Competing causes.** (a) Deliberate projection without labelling. (b) A BI tool's default fill that substitutes the previous value or zero. (c) A join that produces a row for every cohort-column pair, with NULLs coerced to zero by an aggregation. (d) A hand-built spreadsheet where the projection was pasted over the empty cells.

**Discriminating check.** Compare the count of non-empty cells per column against the maturity frontier formula. Any column with more populated cells than mature cohorts has fabricated values. Also check for exact zeros: a genuine zero-activity cell is rare in retention data and usually indicates missing data.

**Bounded fix.** Store projection in a separate column (`projected_N`) rather than in the observed column, and render it with a distinct style. Block zero-fill at the query layer with an explicit `NULLIF`.

**Verification.** The populated-cell count per column equals the maturity count; no decision document cites a `projected_` value without the assumption sentence attached.

### 8.7 Small-cohort ranking

**Symptom.** A segment or cohort with a handful of entities tops a leaderboard, and the ranking changes completely on the next refresh.

**Competing causes.** (a) Small denominator noise. (b) A genuinely distinctive niche segment. (c) Survivorship: only the best entities of that segment were retained by a filter. (d) A different active-event rule applied to that segment's data source.

**Discriminating check.** Attach a Wilson or normal-approximation interval and see whether the leader's interval overlaps the runner-up's point estimate. Also check the segment's entity count against the minimum viable denominator for the decision's tolerance:

```text
n_min such that 1.96*sqrt(0.25/n) <= tolerance
tolerance 0.05 -> n >= 384
tolerance 0.02 -> n >= 2,401
```

**Bounded fix.** Suppress cells below the stated threshold, or pool them into "other" with the pooling rule published. If the segment matters, extend the window rather than shrinking the threshold.

**Verification.** No published ranking contains a cell below the threshold without an interval; the threshold is derived from the decision tolerance, not chosen for aesthetics.

### 8.8 Definition drift across a dashboard's history

**Symptom.** A long time series has a step change at a date that matches no launch, and the step is present in every segment.

**Competing causes.** (a) A real product change on that date. (b) An instrumentation change (SDK version, event rename, tracking-plan edit). (c) A pipeline change (deduplication added or removed). (d) A time-zone or DST boundary.

**Discriminating check.** Compare the raw event volume for the active event across the boundary; an instrumentation change usually shows a level shift in raw counts that is larger than any plausible behaviour change. Then check the tracking-plan version history and the deployment log for that date. A step present in *every* segment, including segments untouched by the product change, points to (b) or (c).

**Bounded fix.** Split the series at the known definition change and publish both halves with a labelled break. Do not "adjust" the historical values to remove the step unless the adjustment is documented and reversible.

**Verification.** The step is annotated with its cause and date; a reader can see the break; downstream comparisons use only post-break data or both halves with the break visible.

### 8.9 Time-zone and DST boundary errors

**Symptom.** One day per year has anomalous counts; a cohort's period-0 cell is slightly off; a funnel step shows a small dip on a specific calendar date.

**Competing causes.** (a) Naive timestamps interpreted in local time by one query and UTC by another. (b) A DST transition adding or removing an hour from a "day". (c) `DATE_TRUNC` applied to a local timestamp while cohort assignment used UTC. (d) Business-day definitions that skip weekends, applied inconsistently.

**Discriminating check.** Count events in the 24 hours around each DST transition and compare against adjacent days; a ±4% deviation concentrated in one hour isolates (b). Recompute one month with everything forced to UTC and compare row totals; a difference isolates (a) or (c).

**Bounded fix.** Store and compute in UTC end to end; apply local time only for presentation. Define the cohort period boundary explicitly as a UTC instant and include the rule in the metric definition.

**Verification.** The two anomalous days normalise; the metric definition document names the time zone and the boundary rule; a re-run under a different session time zone produces identical results.

### 8.10 Entity identity instability

**Symptom.** Both the entrant count and the churn count are higher than the business believes; some users appear to churn and re-enter repeatedly.

**Competing causes.** (a) Duplicate accounts per human (multiple devices or sign-in methods). (b) Key rotation (an anonymous ID replaced by an authenticated ID mid-session). (c) Household or shared-account usage counted as one entity. (d) Deliberate multi-accounting.

**Discriminating check.** Measure the distribution of entities per human proxy (email domain + hashed identifier, payment instrument, device graph). A material mass above one isolates (a) and (d). Check the share of entities whose first event is authenticated versus anonymous; a high anonymous share isolates (b).

**Bounded fix.** Define a canonical entity key with a documented merge rule and a stability window; use the canonical key for both entry and activity. Do not merge retrospectively across a cohort boundary without re-running the cohort.

**Verification.** Re-entry rate falls to a plausible level; cohort sizes match the acquisition report within a stated tolerance; the merge rule is versioned and dated.

### 8.11 Guardrail blindness

**Symptom.** Retention or conversion improves and a revenue or satisfaction metric quietly degrades in the same period; nobody notices because the guardrail was not on the same view.

**Competing causes.** (a) Genuine trade-off (the intervention bought retention with incentives). (b) A shift in the mix toward a lower-value population. (c) A pricing or packaging change in the same window. (d) A guardrail metric that is itself badly defined.

**Discriminating check.** Compute guardrails per retained entity rather than in aggregate: revenue per retained entity, support contacts per 1,000 retained, refund rate among the retained. If the per-entity guardrail is flat while the aggregate moves, the movement is mix (b). If the per-entity guardrail degrades, the trade-off is real (a).

**Bounded fix.** Put the guardrail on the same view as the primary metric with an explicit tolerance and a reversal rule. Per-retained-entity denominators, not totals.

**Verification.** The guardrail is visible on the primary view; the tolerance and reversal rule are written; the trade-off, if real, is stated in the decision memo rather than discovered later.

### 8.12 Survivorship in a "retained users" analysis

**Symptom.** A feature-usage analysis shows that users who use feature X retain at 80%, and the recommendation is to promote X.

**Competing causes.** (a) X genuinely causes retention. (b) X requires being active, so using X is a consequence of retention. (c) X is adopted by users who were already highly engaged before X existed for them. (d) The time window for X usage overlaps the retention window.

**Discriminating check.** Enforce strict temporal precedence: X adoption strictly before the retention window starts, and measure engagement before adoption as a covariate. If the X-adopters' pre-adoption retention was already higher, the effect is selection (c). If X can only be used while active, the analysis is circular (b).

**Bounded fix.** Restrict to a pre-period cohort definition, adjust for pre-period engagement (stratify or match), and label the result as associational. A causal claim requires an experiment or a design with a credible counterfactual.

**Verification.** Pre-adoption retention is balanced or adjusted; the write-up states associational language; the recommendation's expected effect is discounted for selection.

### 8.13 Aggregating percentages

**Symptom.** A regional roll-up of retention does not equal the weighted average of its parts; a total conversion rate computed from step rates disagrees with the direct count.

**Competing causes.** (a) Averaging rates instead of summing counts. (b) Different denominators across regions. (c) One region's rate computed on a partial denominator. (d) A join that duplicated rows in one region only.

**Discriminating check.** Recompute the roll-up from counts and compare with both the averaged-rates figure and the direct-count figure. The direct count is authoritative; if the averaged figure differs, (a) is present. Per-region denominator checks isolate (b) and (c).

**Bounded fix.** Store counts; derive rates at the presentation layer only. Add a reconciliation assertion that the sum of region counts equals the total count.

**Verification.** `Σ region counts = total count` for every stage and period; the published roll-up equals the direct computation to the displayed precision.

**Takeaway:** the diagnostic order is definition → population → maturity → mix → behaviour, because each earlier stage is cheaper to check and more frequently the actual cause.

## 9. Complexity & Cost Analysis

Costs in this domain are dominated by scans of event tables and by the number of cells the analysis materialises, not by arithmetic. The sizing below is expressed in units that a warehouse team can translate directly: rows scanned, bytes read, shuffle volume, and wall-clock at a stated cluster size. Every figure is an ILLUSTRATIVE model with the arithmetic shown; no vendor benchmark is claimed.

### 9.1 Cost of the funnel

The funnel requires, per stage, a scan of the event table filtered by event type and time window, then a distinct-entity aggregation, then an ordered join across stages.

```text
Let  R      = total event rows in the window            [rows]
     f_j    = fraction of rows matching stage j's event  [dimensionless]
     E      = bytes per row                              [bytes/row]
     K      = number of stages
     S      = shuffle bytes per distinct aggregation pass

naive per-stage scans:  bytes_read = K * R * E
single-pass with CASE:  bytes_read = R * E          (one scan, pivot by event type)
sort/hash for ordering: work ~ R log R  comparisons (ordering by (entity, ts))
```

Worked sizing with illustrative inputs:

```text
R = 400,000,000 rows;  E = 220 bytes;  K = 5 stages;  f = (0.03, 0.021, 0.0168,
0.0101, 0.0038) summing to 0.0817

naive:      bytes_read = 5 * 400e6 * 220 = 4.40e11 bytes = 440.0 GB
single pass: bytes_read = 400e6 * 220     = 8.80e10 bytes =  88.0 GB
saving: 440.0 - 88.0 = 352.0 GB, i.e. 80.0% of scan volume

distinct-entity output: <= 400e6 * 0.0817 = 32.68e6 rows before de-duplication;
after de-duplication bounded by entrants = 12.0e6 entities, so the shuffle for the
final join is at most 12.0e6 rows * (8-byte key + 5 * 8-byte timestamps) = 576 MB.
```

The single-pass version wins by construction because the event table is scanned once and the stage predicates are folded into conditional aggregations. The ordering requirement is the real cost: `ORDER BY entity_id, ts` on 400M rows is a large sort, and on a distributed engine it is a range shuffle. The cheaper alternative is to compute per-stage *first occurrence* timestamps with `MIN(ts) FILTER (WHERE event = 'stage_j')` grouped by entity — an aggregation, not a sort — and then compare the resulting columns. That reduces the ordering work from `R log R` to a hash aggregation of `R` and a five-column comparison per entity.

```text
work comparison, illustrative
  sort path:        400e6 * log2(400e6) ~ 400e6 * 28.6 = 1.14e10 comparison units
  hash-agg path:    400e6 hash inserts + 12e6 * 5 comparisons = 4.00e8 + 6.0e7
                    ~ 4.6e8 units
  ratio:            ~24.8x less comparison work, at the cost of memory for the
                    hash table (12e6 keys * ~120 bytes = 1.44 GB, shardable)
```

### 9.2 Cost of the cohort matrix

The cohort matrix is a two-dimensional group-by over the same event scan. Its cost is dominated by the number of entity-period cells materialised.

```text
Let  N      = entities in all cohorts                 [entities]
     H+1    = number of elapsed columns               [cells/entity upper bound]
     b_cell = bytes per aggregated cell                [bytes]

cells = N * (H+1)                      worst case, before activity filtering
      = 12e6 * 7 = 84e6 cells
bytes = 84e6 * 24 = 2.016e9 bytes = 2.016 GB aggregated output
```

Two implementation routes:

```text
Route A: entity x period expansion then group-by
  expand each entity to H+1 rows, join to activity, aggregate
  intermediate rows = N * (H+1) = 84e6; shuffle of 84e6 * ~40 bytes = 3.36 GB

Route B: aggregate activity to (entity, period) first, then pivot
  activity pairs = distinct (entity, active-period) combinations, bounded by
  N * (H+1) but in practice much smaller: with mean 2.6 active periods per entity,
  12e6 * 2.6 = 31.2e6 pairs -> 31.2e6 * 40 bytes = 1.248 GB shuffle
```

Route B is preferable whenever activity is sparse, which it almost always is. The trade-off is an extra aggregation stage and the need to left-join back to the cohort frame to produce zero cells explicitly — and that left join is where the zero-fill failure of Section 8.6 originates, so it must emit `NULL` and not `0`.

### 9.3 Incremental versus full recomputation

A monthly refresh does not need to recompute matured history. Because cohort denominators are frozen and matured rows are immutable, only two things change: the newest anti-diagonal of cells, and any restatement from late data.

```text
full rebuild:  scan R rows, rebuild N*(H+1) cells        cost = C_full
incremental:   scan only events in the last period + late window
               rows = R / P  where P = periods of history
               cost ~ C_full / P + restatement overhead

with P = 24 periods: incremental scan is 1/24 = 4.17% of the full scan
saving on scan: 95.83%; the join/aggregate for the new diagonal is proportional
to the new cohort size, e.g. 20,000 entities * 4 cells = 80,000 cells.
```

The restatement overhead is the honest complication: late events arrive for prior periods, so the incremental window must extend backwards by the measured lateness distribution (for example, `p99` lateness). If `p99` lateness is 6 days, the incremental job reprocesses 6 extra days plus the period boundary.

```text
lateness-driven reprocessing: 6 days / 30-day period = 20.0% extra scan on top
of the period's own volume, so effective incremental scan = 1.20 / 24 = 5.0% of full
```

### 9.4 Query cost of the diagnostic checks

The checks in Section 8 are cheap relative to the main computation, which is why they should be automated rather than argued about.

| Check | Additional work | Illustrative cost as share of one full scan |
|---|---|---|
| Nesting assertion `N_j <= N_{j-1}` | five integer comparisons | < 0.1% |
| Ordered-predecessor count | one anti-join per stage | 10–20% per stage |
| Exposure denominator | one join on entry timestamps | 2–5% |
| Standardized rate | one group-by on segment | 5–10% |
| Interval computation | arithmetic on aggregated cells | < 0.1% |
| Mix decomposition | arithmetic on two segment tables | < 0.1% |
| Entity-per-human distribution | one group-by on a proxy key | 10–30% |

The anti-join for ordered predecessors is the expensive one and is the check that catches the most damaging error, which is a reasonable trade. It can be run on a sample (1% of entities) for routine refreshes and on the full set for a release gate.

### 9.5 Human and decision cost

The expensive resource is not compute. A typical mis-specified retention review consumes several analyst-days and one or more decision cycles, and the failure is usually detected late because the number looked plausible.

```text
cost model, illustrative
  analyst time to build first version               3 person-days
  rework after definition dispute                   2 person-days
  decision deferral                                 1 cycle (2 weeks)
  cost of a wrong prioritisation at 200 paid accounts
    * $348 lifetime value                        $69,600
```

Compare this with the cost of the checks: automating the five assertions in Section 8.4 is roughly one engineer-day once, plus negligible run cost. The asymmetry justifies building the checks before building the dashboard.

### 9.6 Storage and retention of the matrix

The aggregated matrix is small (2 GB at the illustrative scale), so it should be stored physically with a schema that enforces its semantics: `cohort_id, elapsed_period, active_count, cohort_size, exposed_count, is_partial, projected, cutoff_ts, metric_version`. The last three columns are what make the table auditable; a table without them cannot distinguish observation from projection.

```text
storage growth: 42 cells (27 observed + 15 projected) * 24 bytes = 1,008 bytes
per metric version; 12 monthly refreshes = 12,096 bytes/year for one product.
Even at 10,000 metric-version x product combinations: 120.96 MB/year.
```

The cost argument for versioning is therefore trivial: keeping every version of every cell costs megabytes, while losing the version history costs decision quality.

**Takeaway:** one scan should produce both the funnel and the matrix; ordering by timestamp is the expensive operation and is avoidable with filtered `MIN` aggregates; incremental refresh saves roughly 95% of scan volume but must extend backwards for lateness; and the diagnostic assertions cost a fraction of the rework they prevent.

## 10. Key Theorems / Results Reference

The results below are the formal backbone. Each is stated with its hypotheses, because the hypotheses are exactly the conditions under which the corresponding analysis is valid. Where a result is standard, the statement is given as a sketch with its name rather than as a full proof; none of these is presented as novel.

### 10.1 Nesting implies telescoping

**Statement.** Let `U_0 ⊇ U_1 ⊇ ... ⊇ U_K` be finite sets with `N_j = |U_j| > 0` for `j < K`. Then `N_K/N_0 = Π_{j=1..K} (N_j/N_{j-1})`.

**Hypotheses.** Nesting; positivity of intermediate denominators; same entity key throughout. Nothing else — no independence, no distributional assumption.

**Why it matters.** Step rates and the end-to-end rate are the same information in two forms. Any report where they disagree has an arithmetic or definitional bug, not a modelling subtlety.

**Proof sketch.** Expand the product; each `N_{j-1}` appears once in a numerator and once in a denominator and cancels. ∎

**Consequence for practice.** Because the identity is exact, a mismatch is a *test*. Compute both and assert equality to within display rounding; the assertion is free.

### 10.2 Step rates are not averageable

**Statement.** For `s_j ∈ (0,1)` not all equal, the arithmetic mean `(1/K) Σ s_j` is strictly greater than the geometric mean `(Π s_j)^{1/K}`, and the cumulative rate `Π s_j` is strictly less than `((1/K) Σ s_j)^K`.

**Hypotheses.** `K ≥ 2`; at least two distinct `s_j`; all `s_j > 0`.

**Name.** AM–GM inequality, applied to the step rates.

**Why it matters.** It quantifies the direction of the bias: averaging step rates always *overstates* the end-to-end path rate, and the overstatement grows with dispersion. In the Section 5 example, mean 0.6700 versus end-to-end 0.1260 — a factor of 5.3.

**Numerical illustration.** `K=5`, rates `(0.70, 0.80, 0.60, 0.50, 0.75)`: AM `= 0.6700`; GM `= 0.1260^{0.2} = exp(ln 0.1260 / 5) = exp(-2.0712 / 5) = exp(-0.41424) = 0.6608`. The GM of the step rates is the per-step equivalent rate, which is a legitimate summary; the AM is not.

### 10.3 Constant hazard implies geometric survival

**Statement.** If `λ_h = λ` for all `h`, then `S(h) = (1−λ)^h`, and `log S(h) = h·log(1−λ)` is linear with slope `log(1−λ)`.

**Hypotheses.** Discrete periods of equal length; constant conditional exit; single homogeneous population (no unobserved frailty); no re-entry counted as continuation.

**Why it matters.** It gives a one-parameter model that is exactly right for a homogeneous constant-risk population, and it gives the diagnostic for when that model is wrong: curvature in `log S(h)`.

**Corollary (the mixture result).** If the population is a mixture of groups with different constant hazards `λ_g` and fixed weights `w_g`, then `S(h) = Σ_g w_g (1−λ_g)^h`, which is a decreasing convex combination of exponentials; its hazard falls monotonically toward `min_g λ_g` as `h` grows, and `log S(h)` is convex (bowed upward) — never linear.

**Why it matters in practice.** Falling hazard in a real retention curve is *evidence of heterogeneity*, not evidence that "the product improves with tenure". Both are possible; the mixture explanation is the default because it requires no product change at all. Distinguishing them requires segmenting.

**Numerical check.** A 50/50 mixture of `λ_A = 0.60` and `λ_B = 0.10`: `S(1) = 0.5·0.40 + 0.5·0.90 = 0.65`; `S(2) = 0.5·0.16 + 0.5·0.81 = 0.485`; `S(3) = 0.5·0.064 + 0.5·0.729 = 0.3965`. Hazards: `λ_0 = 0.35`, `λ_1 = 1 − 0.485/0.65 = 0.2538`, `λ_2 = 1 − 0.3965/0.485 = 0.1825`. Falling, as predicted, with no group's hazard changing.

### 10.4 Kaplan–Meier consistency under non-informative censoring

**Statement.** With independent censoring times and independent (or conditionally independent) event times, the product-limit estimator `Ŝ(h) = Π (1 − d_k/n_k)` is consistent for `S(h)`, with variance estimable by Greenwood's formula.

**Hypotheses.** Censoring independent of the event process (or independent conditional on covariates included in a stratified analysis); at-risk sets correctly constructed; a single event type.

**Why it matters.** It is the licence to use partial data. Without it, restricting to fully observed entities is the only defensible option, and that restriction reintroduces survivorship bias by selecting on tenure.

**Greenwood variance (discrete form).** `Var(Ŝ(h)) = Ŝ(h)^2 Σ_{k<h} d_k / (n_k (n_k − d_k))`. For the Section 4.2 example at `h=4`: `Ŝ = 0.2050`; terms `600/(1000·400) = 0.0015`, `80/(400·320) = 0.000625`, `64/(320·256) = 0.00078125`, `51/(256·205) = 0.0009718`; sum `= 0.00387805`; `Var = 0.2050^2 · 0.00387805 = 0.042025 · 0.00387805 = 1.62975e-4`; `SE = 0.012766`. A 95% interval is roughly `[0.180, 0.230]` on the linear scale.

**Caveat.** The interval is conditional on the independence assumption, which is untestable from the data alone. A censoring mechanism correlated with churn (accounts deleted on exit) violates it, and the interval will be misleadingly narrow.

### 10.5 Chain-ladder is exact under a multiplicative development model

**Statement.** If `E[N(c,h+1) | N(c,h)] = f_h N(c,h)` with `f_h` independent of `c`, then the column-sum ratio is an unbiased moment estimator of `f_h`, and the projected cumulative factor is the product of the estimated factors.

**Hypotheses.** Multiplicative development; development factors independent of cohort (the "stable pattern" assumption); mature cells only in the sums; no cohort-specific shocks.

**Why it matters.** It makes the projection assumption *falsifiable*: fit `f_h` by cohort and test for heterogeneity. If the per-cohort factors differ by more than their intervals, the pooled factor is not an estimate of anything the new cohort will experience.

**Test.** Compute `f_h^{(c)}` for each cohort separately; a chi-square-style comparison of the ratios, or simply the range across cohorts relative to the interval width, suffices for a screening decision.

### 10.6 Variance of a difference of proportions

**Statement.** For independent samples with proportions `p_1, p_2` and sizes `n_1, n_2`, `Var(p_1 − p_2) = p_1(1−p_1)/n_1 + p_2(1−p_2)/n_2`, estimated by plug-in; the normal approximation is adequate when `n_i p_i ≥ 10` and `n_i (1−p_i) ≥ 10`.

**Hypotheses.** Independent samples; the same measurement definition; proportions from a single Bernoulli trial per entity.

**Why it matters.** It converts "retention rose 5.2 pp" into "the rise is 8 standard errors", which is the difference between a finding and a fluctuation. It also shows why small cells cannot be ranked.

**Caveat.** In retention data, the per-entity trials at different horizons are *dependent* (an entity present at h+1 must have been present at h in the absorbing-exit model), so the formula applies to a comparison at a single fixed horizon, not to a comparison of whole curves. Comparing curves requires a joint covariance or a bootstrap over entities.

### 10.7 Simpson's reversal condition

**Statement.** An aggregate rate can move in the opposite direction to every stratum rate when the strata weights change and the stratum rates differ. Formally, with two strata and two periods, the aggregate difference has the sign of `Δ(weights) · (rate difference)` when rates move little.

**Hypotheses.** At least two strata; differing base rates; changing weights; a consistent stratum definition across periods.

**Why it matters.** It is the reason Section 8.3 exists and the reason `standardize before comparing` is a rule rather than a preference.

**Numerical illustration.** Strata A and B with rates `(0.90, 0.20)`. Period 1 weights `(0.10, 0.90)`: aggregate `= 0.09 + 0.18 = 0.27`. Period 2 weights `(0.90, 0.10)`: aggregate `= 0.81 + 0.02 = 0.83`. Both stratum rates are unchanged and the aggregate tripled. Any conclusion about improvement from the aggregate is false.

### 10.8 The leverage identity

**Statement.** For a nested funnel, the change in terminal count from a change `Δs_j` in step `j` is `ΔN_K = N_{j-1} · Δs_j · Π_{k>j} s_k`, exactly, holding other steps fixed.

**Hypotheses.** Nested sets; independent step changes (no interaction); fixed downstream rates.

**Why it matters.** It ranks interventions by downstream-weighted opportunity and shows that absolute loss is the wrong ranking whenever downstream survival differs across steps.

**Corollary.** The maximum gain from step `j` is `N_{j-1} · (1 − s_j) · Π_{k>j} s_k`, and the total of these maxima across steps exceeds `N_0 − N_K`, so the per-step maxima cannot be summed — they are alternatives, not parts. This is the arithmetic reason a "fix everything" plan does not compose additively.

### 10.9 Bias from conditioning on survival

**Statement.** For any downstream metric `M` and any interim event `A`, `E[M | A]` need not equal `E[M]`, and the difference is generally nonzero whenever `M` and `A` are associated. Analysing only entities that passed `A` therefore estimates a conditional quantity, not the population quantity.

**Hypotheses.** Association between `A` and `M`; a defined conditioning set.

**Why it matters.** "Retention among users who completed onboarding" is a conditional statistic, and the onboarding step is precisely the filter that removes the entities least likely to be retained. Reporting it as "our retention" overstates the product's performance for the population actually acquired.

**Repair.** Report both the conditional and the unconditional figure, and state the conditioning event. If the decision concerns all entrants, use the unconditional figure for sizing and the conditional one only for diagnosis.

### 10.10 Conservation identity for the cohort frame

**Statement.** For each cohort and horizon, `N(c,0) = active(c,h) + inactive(c,h)` where the two sets partition the cohort. Consequently `R(c,h) + (1 − R(c,h)) = 1` exactly, and any table where the complement does not reconcile has a population error.

**Hypotheses.** Frozen cohort membership; a defined active indicator; no double counting.

**Why it matters.** It is the cheapest reconciliation available: `active + inactive` must equal the frozen size for every cell. A cell that fails this check has a join fan-out, a duplicate, or a membership change.

**Illustration.** March cohort, 12,000: `h=3` active 4,704, inactive `12,000 − 4,704 = 7,296`. If a table shows inactive 7,310, the 14-entity discrepancy is a duplicate somewhere, not rounding.

**Takeaway:** the exact identities — telescoping, AM–GM, the leverage formula, the conservation check — are free tests, while the statistical results — Kaplan–Meier, proportion differences, chain-ladder — are conditional on assumptions that must be stated and, where possible, tested.

## 11. Interview / Exam Questions

Fifteen questions, ordered from definitional to diagnostic. Each has a checked answer with the reasoning exposed, plus the red flag that a weak answer raises. All arithmetic in the answers is verified; inputs are ILLUSTRATIVE teaching data.

### Q1. What is the difference between a step conversion rate and an end-to-end conversion rate, and when does each matter?

**Answer.** A step rate conditions on arrival at the previous stage, `s_j = N_j/N_{j-1}`; an end-to-end rate conditions only on entry, `C_K = N_K/N_0`. With `s = (0.70, 0.80, 0.60, 0.50, 0.75)` the mean step rate is `3.35/5 = 0.6700` while the end-to-end rate is `0.70·0.80·0.60·0.50·0.75 = 0.1260`. Step rates locate the gate to change; the end-to-end rate sizes yield per entrant and feeds volume planning.

**Follow-up — why is the mean step rate not a summary of the path?** By AM–GM the arithmetic mean of positive unequal numbers exceeds their geometric mean, so the mean step rate always overstates the per-step equivalent and, raised to the `K`-th power, overshoots the joint probability. The geometric mean `0.1260^{0.2} = exp(-2.0712/5) = exp(-0.41424) = 0.6608` is the legitimate single-number per-step summary, and even it can only be multiplied back to recover the total because the steps are conditionally independent by construction of the path.

**Red flag:** saying the two are "the same thing measured differently" without noting that the product identity requires nesting, or quoting an average of step rates as the conversion rate.

### Q2. Your funnel has a step with a 95% conversion. What do you check first, and why?

**Answer.** Check nesting and eligibility, not the product. A 95% step is either a genuinely trivial gate or a predicate almost everyone satisfies — a page view, a default-on setting, an event fired at install. Run the ordered-predecessor anti-join (`stage_j` entities with no earlier `stage_{j-1}` event); a nonzero count means the query did not enforce order. Then check whether the predicate can be satisfied automatically. Only after both pass is a 95% rate a fact about behaviour.

**Reasoning:** base rates favour definitional explanations, and the check costs one join. Note also the arithmetic bound: a step at 95% has at most `1 − 0.95 = 5.00%` of its arrivals available for improvement, so even a perfect fix moves end-to-end conversion by `0.05 × N_{j-1}/N_0` — for step 2 of the worked funnel that is `0.05 × 0.70 = 3.50 pp` of entrants, i.e. `0.035 × 12,000 = 420` additional accounts at that stage and `420 × 0.225 = 94.5` additional payers.

**Red flag:** projecting downstream value from an unvalidated stage, or celebrating a high rate that is really a query artefact.

### Q3. Conversion rose from 12.6% to 17.0%, but every segment improved only modestly. What is happening and how do you prove it?

**Answer.** Likely mix shift; prove it by standardizing. Paid/organic rates moved `0.24 → 0.28` and `0.05 → 0.06` while weights moved `(0.40, 0.60) → (0.50, 0.50)`. Raw change `= 0.1700 − 0.1260 = 0.0440`. Standardized at base weights: `0.40·0.28 + 0.60·0.06 = 0.1120 + 0.0360 = 0.1480`, so the rate effect is `0.0220` and the mix effect is `0.0220` — exactly half the movement is composition. Under new weights the split is `0.0250 / 0.0190`; both orderings sum to `0.0440`, so publish the form used.

**Follow-up — what if the two forms disagree by more than the decision tolerance?** Report the range, or use a symmetric (Shapley-style) allocation that averages the two forms: `(0.0220 + 0.0250)/2 = 0.0235` for the rate effect and `(0.0220 + 0.0190)/2 = 0.0205` for mix. A single attribution beyond that precision is a rhetorical choice presented as arithmetic.

**Red flag:** attributing the full 4.4 pp to the product, or saying "mix" with no computed split.

### Q4. Retention improved for a new cohort with no behaviour change. Give three measurement mechanisms that would produce that.

**Answer.** (i) **Denominator shrink** — entities that become unobservable drop out of both numerator and denominator, so removing low-retention entities raises the computed rate; with 1,000 entrants, 100 of them churned-and-deleted, and 340 retained, the reported rate is `340/900 = 0.3778` instead of `340/1000 = 0.3400`, a 3.78 pp artefact. (ii) **Active-event loosening** — adding an auto-generated event (prefetch, server-side session) counts entities that did not act. (iii) **Censoring removed from the denominator** — restricting to exposed entities raises immature cells (Section 5, Step 7: `0.4005 → 0.4450`). Others: cohort boundary shift admitting higher-intent traffic; a duplicate-merge rule collapsing multi-account users.

**Reasoning:** each mechanism changes population or indicator, never behaviour, and each is detectable by comparing raw event volume and entity counts across the boundary.

**Red flag:** naming only "better product" or "seasonality" — the explanations that require evidence while the measurement explanations are the null.

### Q5. Compute 90-day retention where 20% of the cohort has only 60 days of exposure.

**Answer.** Restrict the denominator to exposed entities: 1,000 entrants, 200 unexposed, 410 active in the 90-day window → naive `410/1000 = 0.4100`, corrected `410/800 = 0.5125`. Publish the corrected rate with a `partial` flag, the denominator used, and the unexposed count. The interval is computed on the restricted denominator: `SE = sqrt(0.5125·0.4875/800) = sqrt(3.1230e-4) = 0.01767`, giving a 95% interval of roughly `[0.4779, 0.5471]`.

**Follow-up — is 80% exposure good enough?** It depends on the decision tolerance. The bias from including unexposed members is bounded by the unexposed fraction times the maximum possible rate difference, here `0.20 × (1 − 0.5125) = 0.0975`, i.e. up to 9.75 pp of downward bias in the naive figure. If the decision turns on a 2 pp difference, 80% exposure is not good enough and the analysis must wait or pro-rate with the exposure fraction explicitly labelled as an estimate.

**Red flag:** publishing 41% as a decline without mentioning exposure, or "waiting" while the number stays on the dashboard.

### Q6. When is the hazard function more useful than the retention curve itself?

**Answer.** When comparing populations that start at different levels, and when identifying the shape of the exit process. `λ_h = 1 − S(h+1)/S(h)` strips the accumulated level, so two cohorts with the same endpoint but different shapes become distinguishable. It also exposes the mixture signature: a monotonically falling hazard with no product change is evidence of heterogeneity, not of improving tenure effects.

**Follow-up — why does the hazard typically fall with tenure, and what would a rising hazard tell you?** Falling hazard is the default signature of unobserved heterogeneity: fragile entities exit early, the surviving pool is increasingly durable, so the conditional exit probability declines although no individual's risk changed. A 50/50 mixture with `λ_A = 0.60, λ_B = 0.10` produces hazards `0.3500, 0.2538, 0.1825` across the first three periods — falling, with both component hazards constant. A *rising* hazard is not explained by that mechanism: look for a contract or trial ending, a fixed renewal date, a pricing change taking effect, or a cohort whose quality declined.

**Red flag:** computing `λ` from unconditional churn `1 − S(h)`, which is not a hazard, or attributing a falling hazard to "stickier product" without segmenting.

### Q7. Your curve is flat after month 3, but the team wants to extrapolate the first-90-day slope for lifetime value. What do you say?

**Answer.** The early slope estimates the mixture's fast component and cannot be extrapolated. In the January cohort the hazard falls from `0.4800` at `h=0` to `0.0530` at `h=5`, a ratio of `9.06`. A log-linear fit through the first two points gives slope `ln 0.5200 = −0.6539` per period, predicting `ln S(6) = −0.6539 − 5(0.2624) = −1.9659`, i.e. `S(6) ≈ 0.140`, against an observed `0.2680` — nearly a factor of two too pessimistic. Use the tail hazard instead: the average of the last three mature ratios, `(0.1029 + 0.0721 + 0.0530)/3 = 0.0760`, and discount the extrapolation's dependence on regime stability.

**Follow-up — how much history do you need?** Enough that the tail hazard is stable and the decision horizon has mature cells: at least three consecutive mature periods whose hazard ratios sit inside the decision tolerance, plus one full cycle of any seasonality in the active event. For a monthly cadence, 6–12 mature columns; a quarterly business needs 8 or more quarters. Compute the maturity frontier before promising the analysis, and never pick the window to produce a desired answer.

**Red flag:** a log-linear fit forced through the origin, which imposes the constant hazard the data already reject.

### Q8. A product manager says "March cohort retains 39%, January 34%, so the product got better". What is missing?

**Answer.** An interval on the difference, a definition/cut-off equivalence check, and the associational caveat. At these sizes the difference survives: `R(Jan,3) = 0.3400, n = 10,000` gives `SE = sqrt(0.34·0.66/10000) = 0.004737`; `R(Mar,3) = 0.3920, n = 12,000` gives `SE = 0.004457`; `SE(diff) = sqrt(2.2440e-5 + 1.9861e-5) = 0.006504`; `z = 0.0520/0.006504 = 7.995`, `95% CI = [0.0393, 0.0647]`. With 400-entity cohorts instead, `SE(diff) = sqrt(2·0.35·0.65/400) = 0.0337` and the interval `[−0.014, 0.118]` includes zero.

**Follow-up — what else must match?** The elapsed horizon (both h=3 here, good), the active event, the entry definition, the time zone, and the cut-off. A cohort difference at matched horizons is still correlational: a campaign, a season, or an acquisition-channel shift can produce it without any product change.

**Red flag:** ranking cohorts on point estimates with no interval and no definition check.

### Q9. Explain N-day versus unbounded retention, and give a case where they move in opposite directions.

**Answer.** N-day counts activity in exactly period N; unbounded counts activity in any period `>= N`. Unbounded is monotone non-increasing in N by set inclusion; N-day is not, because entities can skip and return. Disagreement: N-day rises `0.120 → 0.130` from N=3 to N=4 while unbounded falls `0.270 → 0.190` — the N-day rise is a cadence shift (some users moved their activity), while the unbounded fall shows fewer entities ever return.

**Follow-up — which one should a dashboard show?** One family only, named in the title, with the other kept as a separate diagnostic view. Mixing families mid-series produces a curve that rises at the switch point, which readers interpret as recovery. For a decision about habit, N-day at a cadence-matched horizon; for a decision about churn, unbounded or a hazard estimate.

**Red flag:** treating N-day as a survival curve, or comparing your N-day-30 against a published "90-day retention" from a source using unbounded.

### Q10. Given a count matrix, how do you tell whether a recent cohort is genuinely worse or merely immature?

**Answer.** Restrict to exposed entities, recompute, and compare at matched horizons with intervals. If the exposure-restricted rate still falls outside the mature comparison's interval, and raw event volumes show no instrumentation step, the difference is behavioural. A second discriminator: compare hazards at matched periods — a level difference with the same shape suggests cohort quality; a difference that converges suggests measurement.

**Follow-up — what would make you withhold the cell entirely?** If the exposed fraction is below the level at which the bias bound exceeds the decision tolerance (Section 8.2 and Q5), publish `not yet estimable` with the maturity date rather than a number with a wide caveat. A withheld cell is a stronger professional signal than a hedged one, because it cannot be quoted out of context.

**Red flag:** comparing a recent cohort's h=1 cell to an old cohort's h=6 cell, or accepting a "collapse" without an exposure check.

### Q11. What is chain-ladder projection and what is its main assumption failure?

**Answer.** Estimate development factors from mature cohort pairs, `f_h = Σ N(c,h+1) / Σ N(c,h)`, then apply the product to immature cells. Main assumption: `f_h` is cohort-invariant. Failure: a regime change makes the pooled factor a weighted average of two patterns matching neither. In the worked example the pooled factor was `f_3 = 0.911338` projecting 5,001, while the post-change regime gave `0.918367` projecting 5,040 — a 39-account difference here, but the same mechanism can be worth many points when the change affects the tail directly.

**Follow-up — how do you test the assumption?** Compute `f_h` per cohort and compare the range to the interval width. If the range exceeds the interval, pooling is invalid for projection. Restrict to the coherent regime, disclose the restriction, and label the output as a projection that inherits the regime's persistence assumption.

**Red flag:** presenting a projected cell as observed, or pooling across a known product change because "more data is better".

### Q12. How would you size the cost of this analysis, and where does the time go?

**Answer.** Cost is one scan plus ordering. With `R = 400,000,000` rows at 220 bytes, per-stage scans cost `5 × 88.0 GB = 440.0 GB`; a single pass with conditional aggregates costs `88.0 GB`, an 80% saving. Ordering by `(entity, ts)` costs about `R log2 R ≈ 1.14e10` comparison units, avoidable with `MIN(ts) FILTER (WHERE event = ...)` per stage and a per-entity comparison — roughly `4.0e8 + 6.0e7 = 4.6e8` units, about 24.8× less, at the cost of a hash table of `12e6 × 120 bytes = 1.44 GB` shardable across workers. Incremental refresh scans `1/24 = 4.17%` of history, plus `6/30 = 20%` for a 6-day `p99` lateness window, i.e. about `5.0%` effective.

**Follow-up — what dominates at very large scale?** The shuffle, not the arithmetic: the entity-level output is bounded by entrants (12e6 rows × 48 bytes of key and timestamps = 576 MB), while the scan is 88 GB. So the design goal is one scan, and the second-order goal is avoiding a wide row shuffle.

**Red flag:** quoting wall-clock without row count, byte size, and cluster shape — a number nobody can reproduce.

### Q13. A stakeholder wants one number that summarises retention. What do you give them?

**Answer.** One number plus its contract, or better, a two-number pair. The single number must carry the family, horizon, cohort size, interval, and cut-off: "month-3 retention 39.2% (N-period, monthly cohorts, entry = signup, n = 12,000, 95% CI 38.3–40.1%, as of 2026-09-30)". The pair adds the tail hazard — level for sizing, hazard for durability — because they answer different questions and can move in opposite directions.

**Follow-up — what if they insist on one number for a scorecard?** Use the retention-based lifetime estimate with its assumption printed: `L ≈ Σ_{h=0..H} S(h) · (1/1)` in period units, truncating at `H`. For the January cohort that is `1.0000 + 0.5200 + 0.4000 + 0.3400 + 0.3050 + 0.2830 + 0.2680 = 3.1160` periods observed, and extending with the tail hazard `0.0760` adds a geometric tail of `S(6)/λ_tail = 0.2680/0.0760 = 3.526` periods, giving `3.1160 + 3.526 = 6.642` periods — but that number is an assumption-dependent extrapolation, not a measurement, and must be labelled as such.

**Red flag:** a "retention score" averaging different horizons or families, or refusing to give any number at all.

### Q14. How do you handle late-arriving events so history does not change on refresh?

**Answer.** Freeze cohort membership at a stated maturity — for example `T+7` days after the cohort period closes — store it as an immutable snapshot, and report late additions in a separate column. Historical `N(c,0)` then never changes; only the newest anti-diagonal and the late column move. The freeze delay comes from the measured lateness distribution: if `p99` lateness is 6 days, a 7-day freeze captures 99% of the eventual population and the residual appears in the late column rather than silently rewriting history.

**Follow-up — what does a mutable denominator cost?** In the Section 4.1 counterexample, adding 40 late entrants to a 1,000-entity cohort moved `R(c,3)` from `0.4100` to `0.3943`, a 1.57 pp change with no behavioural cause. Any comparison against a published figure becomes unreproducible, and a trend computed across refreshes contains denominator noise as a signal.

**Red flag:** recomputing cohorts on every refresh and calling the drift "a small correction".

### Q15. What guardrail would you attach to a retention-improving intervention, and why per-retained-entity?

**Answer.** Revenue per retained entity, support contacts per 1,000 retained, and refund rate among the retained. Per-retained denominators matter because aggregate guardrails move with population size: an intervention that retains many low-value entities can raise aggregate revenue while lowering revenue per entity, so it would look good on both the primary metric and the aggregate guardrail.

**Follow-up — what makes a guardrail operational?** A tolerance, a reversal rule, and visibility on the same view as the primary metric. "Revenue per retained entity must not fall more than 3% below the pre-launch four-week mean; if it does, revert and re-plan" is operational. "Watch revenue" is not, because nobody can act on it.

**Red flag:** a guardrail checked only when the primary metric improves, or one with no tolerance and no owner.

**Takeaway:** strong answers name the population, the conditional structure, and the check that would falsify the claim; weak answers quote a level without a denominator or a trend without an interval.

## 12. Quick Reference Card

| | Rule | Condition | Failure if violated |
|---|---|---|---|
| **1. Definition** | Entity key, entry event, active event, ordered stages, window, cut-off, owner — all seven written before any number | Always; every refresh inherits them | Two analysts, two defensible numbers, no way to adjudicate |
| **2. Funnel** | `s_j = N_j / N_{j-1}`, `C_K = Π s_j` | Sets nested, distinct entity keys, one window, `N_{j-1} > 0` | Averaged step rates overstate the path; in the worked case 0.6700 vs 0.1260 |
| **3. Retention** | `R(c,h) = N(c,h) / N(c,0)` with `N(c,0)` frozen at entry | Numerator counts activity in period h; denominator counts exposed members only | Backfill rewrites history; recent cohorts read as collapses |
| **4. Hazard** | `λ_h = 1 − S(h+1)/S(h)` | Discrete equal-length periods, presence definition | Unconditional churn `1 − S(h)` misread as a hazard; early slope extrapolated to 0.140 vs actual 0.268 |
| **5. Maturity** | Below the anti-diagonal: observed. On it: partial. Above it: projected or withheld | Every read of the cohort matrix | Zero-filled cells read as churn; projections quoted as facts |
| **6. Comparison** | Standardize to fixed weights; report rate effect and mix effect separately, naming the decomposition form | Any period-over-period or segment-over-segment claim | Mix shift sold as improvement; Simpson's reversal; in the worked case half the 4.4 pp was composition |
| **7. Priority** | Rank by `0.01 · N_{j-1} · Π_{k>j} s_k`, not by absolute loss | Nested funnel with stable downstream rates | The largest absolute loss (step 1, 3,600) ranked third on leverage (21.60 vs 30.24) |
| **8. Evidence** | Publish `n`, interval, family, horizon, and cut-off with every rate | Any rate entering a decision | Small cells ranked on noise; 5.2 pp reported with no interval |
| **9. Guardrail** | Revenue, support contacts, refunds per retained entity, with tolerance and reversal rule | Any retention or conversion win | Trade-off discovered after rollout; aggregate masks per-entity decline |

Five assertions that cost almost nothing and catch most errors:

```text
1. N_0 >= N_1 >= ... >= N_K                     nesting
2. every stage counted with COUNT(DISTINCT key) fan-out / duplicates
3. active + inactive = frozen cohort size       conservation per cell
4. populated cells per column = mature cohorts  no fabricated or zero-filled cells
5. sum of segment counts = total count          aggregation integrity
```

Four numbers worth memorising from the worked example, because they are the ones most often misquoted: end-to-end conversion `12.60%`; step-5 (payment) conversion `75.00%`; month-3 retention `34.00%` (January) rising to `39.20%` (March); and the hazard ratio from h=0 to h=5 of `9.06`, which is what forbids extrapolating the early slope.

One sentence to carry into the room: *a funnel is a nested-set count, a cohort matrix is an entry-by-tenure tabulation with a visible maturity frontier, and a retention curve is a survival estimate whose shape tells you whether to extrapolate at all* — and any number that cannot name its denominator, its horizon, and its cut-off is not yet a measurement.
