---
case_id: "CASE-2026-ANA001"
case_title: "Marketplace Conversion Diagnosis: Separating a Real Funnel Decline from a Session-Definition Change"
case_version: "2.1"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "marketplace analytics / conversion and channel mix"
primary_pattern: "decision-ready-metric-contract"
secondary_patterns: ["component-reconciliation", "guardrail-governance"]
evidence_level: "illustrative"
source_basis: "Illustrative teaching scenario; no company measurements or deployment claims."
as_of_date: "2026-11-10"
owner_role: "analytics lead (marketplace growth)"
source_index: "INDEX.md"
review_trigger: "Session, payment, attribution, or release-rule change"
---

# Marketplace Conversion Diagnosis

This case is an **ILLUSTRATIVE** teaching scenario. CraftBay, every count, every dollar amount, and every release detail are constructed inputs checked for internal arithmetic; they do not report a company outcome, architecture, or experiment.

## 1. The One-Line Framing

> Decide whether a fictional marketplace should revert, narrow, or retain a single-page checkout after its reported paid-order conversion rate fell from 3.420% to 3.110%, while a simultaneous sessionization change may have enlarged the denominator.

The decision is not “did a dashboard move?” but “which operational action has the best expected consequence under a frozen and comparable metric definition?”
The decision owner is the marketplace growth lead, who needs a release recommendation before a weekly train closes.
The analytical unit is an eligible buyer session, not an event, device, visitor, seller, or cart line.
The outcome is one deduplicated captured payment attached to that eligible session.
The evidence boundary is strict: this case supplies a reproducible illustration, not causal proof of a real checkout effect.
The primary risk is a false rollback: engineering could spend USD 18,000 on a change whose apparent harm is partly denominator fragmentation.
The converse risk is delay: a real checkout defect can lose paid orders while the team debates instrumentation.
The recommended sequence is to repair comparability, localize the residual decline, then choose an action with an explicit stop rule.

### Decision snapshot

| Contract field | Defined value |
|---|---|
| Owner | Fictional marketplace growth lead |
| Decision date | 2026-11-05 release review |
| Baseline window | 2026-09-08 through 2026-10-05 UTC, 28 complete days |
| Current window | 2026-10-06 through 2026-11-02 UTC, 28 complete days |
| Baseline paid orders / raw sessions | 40,356 / 1,180,000 |
| Current paid orders / raw sessions | 38,580 / 1,240,500 |
| Candidate actions | retain, channel-narrow, revert, or pause changes while measuring |
| Primary metric | paid orders per eligible comparable session |
| Guardrails | captured-payment integrity, average order value, support contacts, and checkout latency |

## 2. Context & Background

CraftBay is a fictional consumer marketplace in which independent makers sell physical goods through one buyer checkout.
A buyer can arrive from organic search, paid social, email, or direct traffic, browse listings, create a cart, and submit payment.
The marketplace records browser events, server checkout events, payment callbacks, campaign attribution, and support contacts.
The business reports sales in USD; tax and shipping are excluded from the order-value metric in this scenario.
The growth lead owns the go/no-go recommendation, while checkout engineering owns the code path and payments operations owns callback reconciliation.
Finance is a stakeholder because order conversion and revenue reporting can diverge if the numerator is defined at the wrong payment state.

### Timeline and change surface

| Date | Constructed event | Analytical implication |
|---|---|---|
| 2026-09-08 | Baseline observation begins | establishes the old 30-minute inactivity rule |
| 2026-10-05 | Baseline observation ends | provides 28 complete comparable calendar days |
| 2026-10-06 | Single-page checkout reaches all traffic | no contemporaneous untreated traffic remains |
| 2026-10-06 | Client library heartbeat changes from 30 to 5 minutes | sessions can split without a shopper changing intent |
| 2026-10-20 | Support labels begin capturing address-form complaints | creates a qualitative diagnostic, not a conversion numerator |
| 2026-11-02 | Current observation ends | leaves callback maturation as a cutoff concern |
| 2026-11-03 | Frozen analysis snapshot is created | protects the decision memo from late table rewrites |
| 2026-11-05 | Release train decision is due | makes irreversible delay costly |

The single-page layout combines shipping address, delivery preference, and payment fields on one view.
The former layout divided those inputs across three steps and emitted a server event at each completed step.
The redesign could plausibly reduce friction by removing navigation, or increase friction by showing too much form content at once.
A layout effect should not be inferred merely because the release and rate change share a date.
The heartbeat change is a different mechanism: it can increase observed sessions even when the same shopper and same order behavior remain unchanged.
Those mechanisms have opposite remedies, which is why the metric contract must precede a rollback decision.

### Actors, incentives, and disagreements

- **Growth lead:** Needs an action recommendation that survives the release review and does not hide uncertainty.
- **Checkout engineer:** Needs a segment-specific symptom before accepting a costly full revert.
- **Payments operations analyst:** Needs captured-payment counts to reconcile with gateway callbacks before accepting an order numerator.
- **Finance partner:** Needs revenue translation that distinguishes lost orders from changes in basket value.
- **Lifecycle marketer:** Needs channel definitions held fixed before changing audience or messaging.
- **Customer support manager:** Needs an addressable failure signature rather than a generic “conversion down” explanation.
- **Data engineer:** Needs versioned event semantics, keys, and cutoff rules to reproduce the dashboard.
- **Privacy reviewer:** Needs hashed identities and a minimal aggregation rather than unrestricted behavioral exports.

The operating disagreement is constructive: each stakeholder sees a different possible failure, and no single dashboard ratio settles them.
The case therefore treats the dashboard as a signal generator and the decision record as the accountable artifact.

## 3. The Business Question

The question is whether the single-page layout should remain exposed to all eligible traffic, be constrained to a low-risk segment, be reverted, or be left unchanged while instrumentation is repaired.
The baseline comparison alone cannot answer causality because both the layout and client session rule changed on the same date.
The decision can still be made under uncertainty if the alternatives, loss horizon, and disconfirming observations are stated in advance.

### Decision contract

| Field | Contract |
|---|---|
| Decision | Choose retain, narrow, revert, or measure-first for the next release train |
| Objective | Maximize captured paid orders per comparable eligible session |
| Comparator | 28-day baseline reconstructed under the same 30-minute session rule |
| Population | Non-bot buyer sessions with a non-empty cart in five supported countries |
| Numerator | Distinct `order_id` with a captured callback before the snapshot cutoff |
| Denominator | Eligible sessions reconstructed with a 30-minute inactivity rule |
| Window | Complete UTC calendar days; no partial release day included |
| Minimum evidence | Payment reconciliation plus channel-level corrected rate table |
| Hard guardrail | Do not improve order count by admitting duplicated payments or failed captures |
| Soft guardrail | Investigate an average order value decline exceeding 2% before broadening exposure |
| Decision horizon | Five weeks, matching the planned release and observation cadence |
| Reversal rule | Halt expansion if corrected conversion or payment integrity fails its threshold |

A rate is useful only after its numerator and denominator refer to the same population and time boundary.
A session that starts before the cutoff but pays after the cutoff creates a timing choice; this case assigns the order to session start but requires the capture callback before snapshot extraction.
An order retry creates another choice; this case counts one unique captured `order_id`, not payment-attempt events.
A shopper with two carts creates two potential eligible sessions, so unique buyers are not a valid denominator substitute.
A bot with a cart-shaped payload is excluded because the business objective concerns human buyer opportunity, not traffic volume.

### Competing hypotheses

- **H1: denominator fragmentation:** The five-minute heartbeat split inactive browsing into more raw sessions; paid-order behavior did not fall by the full raw-rate gap.
- **H2: paid-social form friction:** Mobile paid-social shoppers encounter address validation or field overload more often in the single-page view.
- **H3: channel-mix movement:** More paid-social traffic and less direct traffic changed the aggregate rate despite stable within-channel behavior.
- **H4: payment-event loss:** The redesigned confirmation event misses some captured orders, making the numerator too small.
- **H5: payment completion harm:** A real checkout defect causes buyers to abandon after cart creation or to fail authorization.
- **H6: callback latency:** The final current days have incomplete capture callbacks at snapshot time, depressing the current numerator.
- **H7: campaign-quality drift:** A campaign attracted lower-intent visitors independent of the checkout layout.
- **H8: attribution reclassification:** A first-touch rule or cookie behavior shifted sessions across channels, corrupting the mix decomposition.

The team must avoid treating H1 and H5 as mutually exclusive; a measurement artefact can coexist with a real usability problem.
The team must also avoid treating H3 as a causal explanation; composition explains an aggregate movement but does not identify why traffic composition changed.

## 4. Assumptions & Constraints

Each assumption is a decision dependency, not a sentence placed in a methodology section for decoration.
A violated assumption either changes the metric, weakens attribution, or changes the financial threshold for action.

### A-01
**Assumption:** Both windows contain 28 complete UTC days.
**Discriminating check:** Compare hourly volumes at each edge with normal daily shape.
**Fallback:** If false, trim symmetrically before comparing.
**Owner and trigger:** Analytics lead rechecks this when the relevant definition or release changes.

### A-02
**Assumption:** `order_id` is unique for a captured order.
**Discriminating check:** Count duplicates and reconcile to gateway settlement records.
**Fallback:** If false, deduplicate before conversion arithmetic.
**Owner and trigger:** Payments operations rechecks this when the relevant definition or release changes.

### A-03
**Assumption:** The 30-minute reconstructed session rule can link events through a stable hashed visitor key.
**Discriminating check:** Measure missing or rotating identity coverage by client version.
**Fallback:** If false, publish an uncertainty range rather than a point correction.
**Owner and trigger:** Data engineering rechecks this when the relevant definition or release changes.

### A-04
**Assumption:** Channel assignment uses the same first-touch rule in both windows.
**Discriminating check:** Compare rule version and unknown-channel share.
**Fallback:** If false, suppress channel mix claims.
**Owner and trigger:** Lifecycle marketing rechecks this when the relevant definition or release changes.

### A-05
**Assumption:** No other material checkout release reached all traffic in the observation period.
**Discriminating check:** Review release calendar and feature flags.
**Fallback:** If false, split the interval or classify causality as unresolved.
**Owner and trigger:** Release manager rechecks this when the relevant definition or release changes.

### A-06
**Assumption:** Captured callbacks mature within the adopted cutoff allowance.
**Discriminating check:** Plot callback lag distribution by payment method and day.
**Fallback:** If false, delay the final days or estimate only mature cohorts.
**Owner and trigger:** Payments operations rechecks this when the relevant definition or release changes.

### A-07
**Assumption:** The rollback effort estimate is USD 18,000.
**Discriminating check:** Obtain engineering estimate with included work stated.
**Fallback:** If false, recompute the breakeven loss threshold.
**Owner and trigger:** Engineering manager rechecks this when the relevant definition or release changes.

### A-08
**Assumption:** Average order value excludes tax and shipping consistently.
**Discriminating check:** Inspect field definitions and currency conversion version.
**Fallback:** If false, use gross merchandise value only as a separately labeled sensitivity.
**Owner and trigger:** Finance partner rechecks this when the relevant definition or release changes.

### A-09
**Assumption:** Support contacts can be linked to checkout version without exposing personal details.
**Discriminating check:** Use aggregate tagged counts and version exposure date.
**Fallback:** If false, retain support evidence as qualitative only.
**Owner and trigger:** Support manager rechecks this when the relevant definition or release changes.

### A-10
**Assumption:** Bot filtering is comparable across windows.
**Discriminating check:** Compare bot-rule version and excluded-session proportion.
**Fallback:** If false, reconstruct both periods under one rule.
**Owner and trigger:** Trust analyst rechecks this when the relevant definition or release changes.

The most material known violation is A-03 in its raw form: the heartbeat change altered session construction.
The analytic response is not to erase the raw series; it is to label it non-comparable and reconstruct a comparable denominator.
The material unknown is A-05, because a full-traffic release removes a clean concurrent counterfactual.

## 5. Data, Tools & Method

### 5a Data Specification

The grain is one eligible reconstructed buyer session, represented by `session_id_reconstructed` and joined to zero or one deduplicated captured order.
Event time is the UTC timestamp emitted by the source system; processing time is warehouse ingestion time, and the two must not be substituted.
The cutoff timestamp is 2026-11-03T02:00:00Z; callbacks received later are unavailable at decision time.
The population is sessions with a cart, a supported country, non-staff identity, and non-bot classification under the frozen rule.
The primary key for orders is `order_id`; the primary key for raw events is `event_id`; neither key is a buyer identifier.
The stable linkage key is a salted, privacy-preserving visitor hash available only in the restricted snapshot.
Units are sessions, captured orders, USD excluding tax and shipping, milliseconds, and percentage points where stated.
Information not available at decision time includes late callbacks, later refunds, future support resolution, and post-decision campaign changes.

| Field | Type / units | Role in decision |
|---|---|---|
| `session_id_raw` | string | Client-generated raw session identifier; changes when the heartbeat rule closes a session. |
| `visitor_hash` | string | Salted pseudonymous link key used only to reconstruct inactivity gaps. |
| `event_time_utc` | timestamp | Event-time ordering field; ties are resolved by ingestion sequence only for auditing. |
| `client_lib_version` | categorical | Version that identifies five-minute versus thirty-minute heartbeat semantics. |
| `channel_first_touch` | categorical | Organic search, paid social, email, direct, or explicit unknown. |
| `cart_created` | boolean | Eligibility signal; it precedes payment and therefore cannot leak the captured outcome. |
| `order_id` | string | Unique business order identifier after payment retry collapse. |
| `capture_status` | categorical | Captured, failed, pending, or reversed; only captured is numerator-eligible. |
| `capture_time_utc` | timestamp | Callback time used to enforce the snapshot cutoff. |
| `order_value_usd` | decimal USD | Item value excluding tax and shipping; kept separate from conversion count. |
| `country_code` | categorical | Used to limit the population to supported storefront countries. |
| `bot_rule_version` | categorical | Allows a comparability check for automated-traffic exclusion. |

The snapshot deliberately contains no post-purchase satisfaction score because that signal would arrive after the action deadline.
The snapshot deliberately contains no seller-side inventory state because the bounded decision is checkout exposure, not supply management.

### 5b Tools & Stack

- **Versioned warehouse snapshot:** Holds the frozen events, order callbacks, and rule-version fields used for reproducible aggregation.
- **SQL common-table expressions:** Makes eligibility, deduplication, session reconstruction, and aggregation reviewable as separable transformations.
- **Payment reconciliation extract:** Checks that captured-order counts agree with a source independent of the confirmation event.
- **Version-controlled metric contract:** Records numerator, denominator, population, cutoff, owner, and change history.
- **Notebook or spreadsheet check:** Recomputes totals and formulas outside the grouped SQL query to expose join mistakes.
- **Feature-flag service:** Would allow a future controlled rollout; it does not create a counterfactual retroactively.
- **Dashboard annotation log:** Warns operators that raw session conversion changed definition on 2026-10-06.
- **Release calendar:** Tests the claim that the suspected layout was the only concurrent production change.

### 5c Method Summary

Step 1 filters events to the decision population and complete calendar windows.
Step 2 deduplicates payment attempts into one captured order per `order_id`.
Step 3 reconstructs comparable sessions by applying a 30-minute inactivity gap to both windows.
Step 4 groups reconstructed sessions and captured orders by window and first-touch channel.
Step 5 compares channel rates with a fixed baseline-weight decomposition and a separate mix term.
Step 6 translates the corrected rate gap into orders and USD only under explicitly labeled illustrative assumptions.
Step 7 evaluates candidate actions against primary and guardrail metrics, then writes a reversal condition.

### 5d Provenance, Quality, Exclusions, Leakage

The source basis is an **ILLUSTRATIVE** frozen snapshot designed for teaching arithmetic and diagnostic discipline.
No query result is presented as observed company evidence, and no system implementation is implied by the listed tools.
Provenance is still specified because a trustworthy fictional case must say what a real reviewer would need to inspect.

| ID | Artifact | Version / cutoff | Supports | Access basis | Limitation |
|---|---|---|---|---|---|
| SRC-SNAP-01 | Frozen event and payment snapshot | 2026-11-03T02:00:00Z | Counts, timestamps, channel labels, client versions | Restricted teaching construct | Late callbacks are unavailable. |
| SRC-PAY-02 | Captured-payment reconciliation extract | Same cutoff | Independent captured-order comparison | Restricted teaching construct | Settlement timing can differ from callback timing. |
| SRC-REL-03 | Release calendar record | 2026-10-06 | Layout and client-library release date | Illustrative record | Does not prove no hidden configuration change. |
| SRC-MET-04 | Metric contract version | v2.1 | Definitions and owners | Illustrative governance artifact | Cannot reconstruct missing raw fields. |
| SRC-SUP-05 | Aggregated support-tag summary | 2026-10-20 onward | Potential address-form symptom | Illustrative qualitative input | Tags are not a randomized outcome. |
| RUN-06 | Hand arithmetic worksheet | This manuscript | Independent count, rate, and revenue checks | Public in case | Depends on supplied constructed inputs. |

### Quality gate sequence

#### Completeness
**Check:** Compare event volume by UTC hour at each window boundary.
**Why it matters:** A missing final current day can look like a conversion collapse.
**If it fails:** Keep only complete matched days.

#### Uniqueness
**Check:** Count `order_id` and payment-attempt duplicates separately.
**Why it matters:** Retries can inflate the numerator without increasing customer orders.
**If it fails:** Deduplicate by business order before grouping.

#### Referential integrity
**Check:** Measure orders lacking an eligible reconstructed session.
**Why it matters:** An orphaned order cannot be assigned a valid denominator unit.
**If it fails:** Report orphan rate and exclude only with an explicit reconciliation.

#### Temporal validity
**Check:** Require capture time no later than the snapshot cutoff.
**Why it matters:** Future knowledge would leak a late success into an earlier decision.
**If it fails:** Freeze the cutoff and rerun both windows identically.

#### Definition parity
**Check:** Compare client library and session-rule versions by window.
**Why it matters:** A denominator rule change makes raw rates non-comparable.
**If it fails:** Reconstruct both windows using a common rule.

#### Attribution parity
**Check:** Compare first-touch rule version and unknown-channel fraction.
**Why it matters:** A channel shift can be classification drift rather than demand.
**If it fails:** Collapse to total or hold attribution fixed.

#### Population parity
**Check:** Compare bot, employee, and country filters by rule version.
**Why it matters:** Different exclusions alter both traffic mix and rate.
**If it fails:** Reapply one rule to both periods.

#### Currency integrity
**Check:** Verify USD decimal scale and excluded charges.
**Why it matters:** Mixed currencies or tax inclusion distort order-value guardrails.
**If it fails:** Normalize monetary field before summary.

#### Join cardinality
**Check:** Profile session-to-order joins before aggregation.
**Why it matters:** An inner join removes non-converters; a fan-out duplicates converters.
**If it fails:** Use a left join after one-row-per-order collapse.

#### Lag maturity
**Check:** Plot callback lag by payment method and date.
**Why it matters:** Recent cards can remain pending at the cutoff.
**If it fails:** Use mature cohorts or state an uncertainty bound.

The analysis excludes internal staff, bots, unsupported countries, and raw events that cannot be placed in either complete UTC window.
The analysis does not exclude a failed payment attempt from the denominator; failure is behavior relevant to an eligible checkout opportunity.
The analysis does not use post-cutoff refunds to redefine the captured-order numerator because refunds are not known at the decision instant.
The analysis does not condition the denominator on seeing a confirmation event because that condition would use a downstream outcome.

### 5e Reproducibility & Method-Selection Ledger

| Candidate method | Decision use | Status | Boundary | Revisit condition |
|---|---|---|---|---|
| Aggregate raw conversion | Fast dashboard comparison | Rejected as decision basis | Raw session semantics changed at release. | Only as an alert that triggers diagnosis. |
| Comparable reconstructed conversion | Same session rule in both windows | Chosen primary descriptive metric | Depends on visitor-link coverage and inactivity assumption. | As the release-decision baseline. |
| Channel decomposition | Separates within-channel rate and mix effects | Chosen diagnostic method | Depends on stable attribution taxonomy. | When channel assignment passes parity checks. |
| Payment reconciliation | Tests numerator against independent callback source | Chosen validation gate | Cannot diagnose user friction by itself. | Before publishing an order metric. |
| Pre/post causal claim | Associates release date and rate movement | Rejected for causal proof | Full rollout and concurrent instrumentation change confound it. | Only after a controlled future rollout. |
| Difference in differences | Would compare treated and untreated exposure | Unavailable | No clean untreated channel or cohort exists. | When a feature flag retains a control arm. |
| Time-trend regression | Could adjust long-run drift | Not selected | Two 28-day windows are insufficient for robust seasonality modeling. | With a longer stable history. |
| Synthetic control | Could use another storefront | Not applicable | This bounded scenario has one storefront. | If comparable markets exist later. |

Reproduction order matters: rebuilding a rate before deduplicating payments can produce a consistent but wrong answer.
A reviewer should first verify row grain, then totals, then channel rates, then the decomposition, and only then the financial translation.
A reviewer should be able to substitute any corrected session total into the formulas below without changing their definitions.

## 6. Analysis Walkthrough

All numerical inputs in this walkthrough are **ILLUSTRATIVE** and have been checked by direct multiplication or division.
Percentages are reported in percent; one basis point (bp) equals 0.01 percentage points; USD is a nominal teaching currency.
The baseline and current calendar windows are each 28 complete days, but raw sessions are not comparable until sessionization is aligned.

### 6.1 Reconcile raw totals before interpreting a rate

| Window | Raw eligible sessions | Captured orders | Raw conversion | Revenue excluding tax/shipping |
|---|---:|---:|---:|---:|
| Baseline | 1,180,000 | 40,356 | 3.4200% | USD 2,315,986 |
| Current | 1,240,500 | 38,580 | 3.1100% | USD 2,144,110 |

The raw denominator rises by 60,500 sessions while the captured numerator falls by 1,776 orders.
The raw rate change is 3.1100% minus 3.4200%, or -0.3100 percentage points, which is -31.00 bp.
The raw relative change is -0.3100 / 3.4200 = -0.09064, or -9.064%.
Neither percentage presentation is a decision-ready effect estimate because the session rule changed at the current-window boundary.

```text
FORMULA [C-01]: r = O / S
WHERE: r is captured-order conversion [orders/session]; O is distinct captured orders [orders];
       S is eligible comparable sessions [sessions].
CONSTRAINT: O must be deduplicated, each order must have a valid session assignment, and S must be defined identically in both windows.
NUMERICAL EXAMPLE: baseline r_b = 40,356 / 1,180,000 = 0.034200 = 3.4200%;
                   current raw r_raw = 38,580 / 1,240,500 = 0.03110036 = 3.1100% after rounding.
INTERPRETATION: the raw dashboard reports a 31.00 bp decline, but it mixes behavioral movement with a changed denominator.
```

Dimensional check: orders divided by sessions has units orders per session, conventionally displayed as a rate.
Arithmetic check: 1,180,000 multiplied by 0.0342 equals 40,356 exactly.
Arithmetic check: 1,240,500 multiplied by 0.03110036 is approximately 38,580, with the small difference due to rounding.
Boundary: if `S` were counted after an inner join to orders, every remaining session would convert and the formula would yield 100%, which is invalid.

### 6.2 Build the comparable denominator

The current client library closes a raw session after five minutes of inactivity, while the baseline closes after thirty minutes.
For illustration, replaying current events through a thirty-minute inactivity rule produces 1,200,000 comparable current sessions.
The correction merges 40,500 raw current sessions into longer eligible reconstructed sessions.
This means 40,500 / 60,500 = 66.94% of the observed raw-session growth relative to baseline is compatible with definition fragmentation.
The remaining 20,000-session growth is not attributed here; it can reflect real traffic growth, eligibility movement, or another unmeasured mechanism.

```text
FORMULA [C-02]: S_c = S_raw - F, and r_c = O_c / S_c
WHERE: S_c is comparable sessions [sessions]; S_raw is raw current sessions [sessions];
       F is sessions created by rule fragmentation [sessions]; O_c is captured current orders [orders].
CONSTRAINT: F may be subtracted only when raw sessions can be linked through a stable identity and the common inactivity rule is specified before inspecting the outcome.
NUMERICAL EXAMPLE: S_c = 1,240,500 - 40,500 = 1,200,000 sessions;
                   r_c = 38,580 / 1,200,000 = 0.03215 = 3.2150%.
INTERPRETATION: common-rule conversion declines 20.50 bp from baseline, not the raw 31.00 bp.
```

Independent check: 1,200,000 multiplied by 0.03215 equals 38,580 captured orders exactly.
Sensitivity check: if the reconstruction merges only 30,000 sessions, rate becomes 38,580 / 1,210,500 = 3.1863%.
Sensitivity check: if it merges 50,000 sessions, rate becomes 38,580 / 1,190,500 = 3.2407%.
The corrected decline therefore lies between 17.93 bp and 23.37 bp in this stated ±10,000-session reconstruction sensitivity.
The decision should not describe the corrected rate as exact beyond the quality of identity linkage and inactivity assumptions.
Boundary: if visitor hashes rotate frequently after the client release, `F` cannot be inferred reliably and the point correction must be replaced with a range or withheld.

### 6.3 Localize the residual movement by channel

| Channel | Baseline sessions | Baseline orders | Baseline rate | Current comparable sessions | Current orders | Current comparable rate |
|---|---:|---:|---:|---:|---:|---:|
| Organic search | 500,000 | 18,000 | 3.6000% | 516,000 | 18,060 | 3.5000% |
| Paid social | 300,000 | 10,200 | 3.4000% | 318,000 | 9,540 | 3.0000% |
| Email | 200,000 | 6,576 | 3.2880% | 188,000 | 6,150 | 3.2713% |
| Direct | 180,000 | 5,580 | 3.1000% | 178,000 | 4,830 | 2.7135% |
| Total | 1,180,000 | 40,356 | 3.4200% | 1,200,000 | 38,580 | 3.2150% |

Organic loses 10.00 bp, a modest movement relative to the aggregate decline.
Paid social loses 40.00 bp and supplies the largest weighted within-channel contribution because it combines a material share with a large rate loss.
Email loses about 1.67 bp, which is too small to justify targeting before checking data quality.
Direct loses about 38.65 bp on a smaller base, making it a secondary diagnostic priority rather than a conclusive cause.
The observed paid-social rate is not proof that layout friction is mobile-specific; device, campaign, and payment-method mix remain competing explanations.

### 6.4 Decompose rate change into behavior and mix

Let baseline channel weight `w_b,g` equal baseline channel sessions divided by total baseline sessions.
Use the baseline weights to ask a counterfactual descriptive question: what would current conversion be if only channel-specific rates changed while baseline traffic composition remained fixed?
This is a decomposition, not a causal attribution model.

```text
FORMULA [C-03]: Delta_r = sum_g w_b,g * (r_c,g - r_b,g) + sum_g (w_c,g - w_b,g) * r_c,g
WHERE: Delta_r is total rate change [orders/session]; g indexes channel; w_b,g and w_c,g are session shares [dimensionless];
       r_b,g and r_c,g are baseline and current comparable channel rates [orders/session].
CONSTRAINT: channel categories must be mutually exclusive, exhaustive after explicit unknown handling, and governed by a stable assignment rule.
NUMERICAL EXAMPLE: weighted current-at-baseline-mix = (500000/1180000)*0.035 + (300000/1180000)*0.030
                   + (200000/1180000)*0.0327128 + (180000/1180000)*0.0271348 = 0.0321401 = 3.21401%.
                   within change = 3.21401% - 3.42000% = -0.20599 percentage points = -20.599 bp;
                   mix change = 3.21500% - 3.21401% = +0.099 bp after rounding.
INTERPRETATION: nearly all corrected aggregate movement is within-channel, not the listed-channel traffic mix.
```

Arithmetic check: the four baseline weights sum to 1.000000 within rounding.
Arithmetic check: current comparable sessions sum to 516,000 + 318,000 + 188,000 + 178,000 = 1,200,000.
Arithmetic check: current captured orders sum to 18,060 + 9,540 + 6,150 + 4,830 = 38,580.
Interpretation boundary: if an unreported unknown channel has changed materially, forcing it into direct would invalidate the decomposition even though the totals still reconcile.

### 6.5 Translate the corrected rate gap into an illustrative decision scale

Holding current comparable traffic at 1,200,000 sessions, the baseline conversion rate would imply 1,200,000 times 0.0342, or 41,040 captured orders.
The current illustration has 38,580 captured orders, creating a rate-equivalent gap of 2,460 orders over 28 days.
Baseline illustrative average order value is USD 2,315,986 divided by 40,356, or USD 57.3898 per captured order.
The current illustrative average order value is USD 2,144,110 divided by 38,580, or USD 55.5752 per captured order.
The AOV difference is -USD 1.8146, or -3.16% relative to the baseline value; it crosses the stated 2% investigation trigger.

```text
FORMULA [C-04]: G_O = S_c * (r_b - r_c), and G_R = G_O * AOV_b
WHERE: G_O is rate-equivalent order gap [orders]; S_c is current comparable sessions [sessions];
       r_b and r_c are comparable conversion rates [orders/session]; G_R is illustrative revenue-equivalent gap [USD]; AOV_b is baseline average order value [USD/order].
CONSTRAINT: this is a scenario translation, not a causal forecast; it holds traffic, order mix, and baseline AOV constant and excludes refunds, margin, and elasticity.
NUMERICAL EXAMPLE: G_O = 1,200,000 * (0.03420 - 0.03215) = 1,200,000 * 0.00205 = 2,460 orders;
                   AOV_b = 2,315,986 / 40,356 = USD 57.3898/order;
                   G_R = 2,460 * USD 57.3898 = USD 141,178.91 over 28 days after rounding.
INTERPRETATION: the illustrative scale exceeds a USD 18,000 revert cost, but only if a reversible action plausibly recovers a meaningful portion of the gap.
```

A strict financial decision would use contribution margin and causal recovery probability rather than revenue-equivalent value.
For example, with a 30% contribution margin and a 25% chance that a revert recovers the 2,460-order gap, expected contribution is 2,460 × 57.3898 × 0.30 × 0.25 = USD 10,588.42.
That illustrative expected contribution is below USD 18,000, so a full revert is not automatically justified on this conservative assumption.
With a 60% recovery probability under the same margin, expected contribution becomes USD 25,412.22 and exceeds the rework cost.
This sensitivity is why the preferred next action is a measurable constrained exposure rather than a blind all-traffic rollback.


### 6.6 Check uncertainty without confusing it for causality

Sampling uncertainty answers whether the observed counts could arise from binomial noise under a stable rate; it does not repair confounding from a release and an instrumentation change.
Using the raw current rate for this rough screen is conservative only in a descriptive sense; the decision rate remains the reconstructed rate.
For two independent large session samples, a standard error for a difference in proportions is the square root of the sum of each binomial variance term.
The independence assumption is imperfect because sessions from a returning buyer can correlate; the calculation is a scale check, not a final inferential certificate.

```text
FORMULA [C-05]: SE(r_b - r_c) = sqrt(r_b*(1-r_b)/S_b + r_c*(1-r_c)/S_c)
WHERE: SE is the approximate standard error [orders/session]; r_b and r_c are rates [orders/session];
       S_b and S_c are eligible session counts [sessions].
CONSTRAINT: uses a large-sample independent Bernoulli approximation; clustered visitors, time dependence, and changed definitions can make the interval misleading.
NUMERICAL EXAMPLE: sqrt(0.0342*0.9658/1,180,000 + 0.03215*0.96785/1,200,000)
                   = sqrt(0.00000002799 + 0.00000002594) = sqrt(0.00000005393)
                   = 0.0002322 = 2.322 bp; approximate 95% interval for a 20.50 bp difference is 20.50 ± 4.55 bp.
INTERPRETATION: the corrected descriptive decline is much larger than ordinary random counting noise, but that does not identify its cause.
```

The approximate interval lies roughly from 15.95 bp to 25.05 bp under the stated model.
If visitor-level clustering doubles the variance, the standard error rises by the square root of two to about 3.28 bp, still below the point movement.
If session reconstruction is materially wrong, its definition uncertainty can dominate the binomial standard error; that uncertainty is operational rather than sampling noise.
A confidence interval is therefore reported only beside its validity conditions, never as evidence that the layout caused the movement.

### 6.7 Capacity and recovery arithmetic

The proposed controlled exposure requires observability capacity as well as checkout capacity.
Assume an **ILLUSTRATIVE** controlled exposure sends 20% of 1,200,000 comparable sessions over 28 days to the single-page variant.
That exposure is 240,000 sessions over 28 days, or 8,571.43 sessions per day on average.
Assume the telemetry pipeline receives 14 checkout events per exposed session and retains raw event payloads for 45 days.

```text
FORMULA [C-06]: E = q * S_c * e, and V = E * b * d
WHERE: E is telemetry events per 28-day exposure [events]; q is treatment share [dimensionless];
       S_c is comparable sessions per 28 days [sessions]; e is events per session [events/session];
       V is retention payload [bytes]; b is average payload [bytes/event]; d is retained exposure windows [dimensionless].
CONSTRAINT: assumes a stable event rate, excludes indexes and replication, and uses the average rather than peak payload size.
NUMERICAL EXAMPLE: E = 0.20 * 1,200,000 * 14 = 3,360,000 events per 28 days;
                   with b=900 bytes and d=45/28=1.60714, V=3,360,000*900*1.60714 = 4,860,000,000 bytes = 4.526 GiB.
INTERPRETATION: payload capacity is modest in this illustration, but callback reconciliation and identity-link health remain the limiting controls.
```

Dimensional check: sessions times events per session yields events; multiplying events by bytes per event yields bytes.
A peak check is also needed: a daily average cannot size a burst-prone mobile campaign.
If peak hour traffic is four times the daily average rate, the pipeline should test approximately 47,? events per hour only after specifying the hourly distribution; no peak claim is made from this average.
The runbook should queue noncritical diagnostic events before allowing payment callbacks to contend for the same critical processing path.
A telemetry outage must not block payment capture; observability is a measurement dependency, not a business transaction dependency.

### 6.8 Analysis limits and disconfirming observations

A comparable rate decline would be weakened if captured payment reconciliation shows many current orders are missing from the event snapshot but present in the gateway extract.
A paid-social usability hypothesis would be weakened if the decline persists equally on desktop organic traffic with no address-field interaction increase.
A sessionization hypothesis would be weakened if current sessions reconstructed under 30 minutes remain close to raw sessions because identity linkage finds few mergeable gaps.
A channel-mix explanation would be weakened by the fixed-weight result: its approximately +0.10 bp mix term cannot explain a -20.50 bp corrected decline.
A rollback benefit hypothesis would be weakened if a constrained legacy-form cohort has no higher captured conversion after the same payment and session checks.
A current AOV concern would be weakened if order-value calculation under the same item-only definition no longer crosses the 2% guardrail.

The analysis establishes a prioritized measurement and action design, not a retrospective causal verdict.
The minimum useful conclusion is that the raw -31.00 bp dashboard movement overstates the comparable decline, while the remaining decline is too large and too localized to dismiss as mix alone.

## 7. The Finding

The constructed evidence supports a narrow descriptive finding: the raw dashboard decline is partly caused by a changed session denominator, but a comparable-session decline of 20.50 bp remains after reconstruction.
The corrected rate is 3.2150% versus a baseline 3.4200%, representing 2,460 fewer captured orders over 28 days if current comparable traffic is held constant.
The four-channel fixed-weight decomposition assigns approximately -20.60 bp to within-channel rate movement and +0.10 bp to channel mix.
Paid social has the largest absolute rate drop among high-volume channels, while direct traffic has a substantial secondary drop.
The current item-only illustrative average order value is 3.16% below baseline, so a primary-metric-only rollout would violate the stated investigation trigger.

The evidence does **not** establish that the single-page form caused the within-channel decline.
Full exposure, a contemporaneous client-library change, potential campaign drift, and limited history block a causal pre/post claim.
The evidence does **not** establish that reverting all traffic will restore 2,460 orders.
The evidence does establish that leaving raw session conversion unannotated would overstate the apparent decline by 10.50 bp.

### Finding-to-action matrix

| Observation | What it rules in | What it cannot rule out | Immediate implication |
|---|---|---|---|
| Raw rate is down 31.00 bp | A visible business signal exists | Comparable behavior change magnitude | Do not decide from dashboard alone |
| Common-rule rate is down 20.50 bp | A residual descriptive decline exists | Layout causality | Test a reversible exposure |
| Mix term is about +0.10 bp | Listed channel mix is not the main arithmetic driver | Campaign quality within channel | Prioritize within-channel diagnostics |
| Paid social loses 40.00 bp | A high-value segment is abnormal | Device, payment, and creative mechanisms | Stratify the next test by device and payment method |
| AOV is down 3.16% | Value guardrail merits investigation | Whether reduced value is caused by checkout | Include value as a stop metric |
| Heartbeat changed at release | Raw denominator is definition-sensitive | Exact fragmentation amount without linkage audit | Version and reconstruct sessions |
| Gateway reconciliation passes | Captured-order numerator is plausible | User experience mechanism | Proceed to experience diagnosis |

A sound recommendation separates certainty levels.
High confidence: raw and reconstructed definitions differ, so raw conversion is not comparable across the boundary.
Moderate confidence: after correction, a material rate decline persists in the constructed snapshot.
Low confidence: the layout itself is the sole or principal cause of the residual decline.

### Recommendation

Choose **measure-first with a constrained, randomized legacy-form holdout**, rather than an immediate full revert or unrestricted retention.
Repair the session metric contract before the holdout begins, so the experiment is not judged by the contaminated raw denominator.
Start with paid-social and direct traffic because their rate movements create the strongest prioritization signal, but preserve a randomized concurrent comparison within each stratum.
Keep the existing layout for the remaining traffic only while payment integrity, latency, and AOV guardrails remain inside predefined bounds.
Escalate to a full revert if a mature, reconciled holdout shows a practically material legacy-form advantage and the expected contribution exceeds the stated rework cost.

## 8. System/Solution Design

The solution is a decision-support pipeline, not a claim that an advanced predictive model is required.
Its job is to produce a versioned, comparable conversion metric and an auditable recommendation from events that can arrive late or duplicate.
The design separates critical commerce processing from analytical observability so a measurement incident cannot prevent a buyer from paying.

### Input-to-output architecture

```text
Browser events + server checkout events + payment callbacks + release metadata
        -> schema validation and event-time normalization
        -> privacy-preserving visitor linkage and common-rule session reconstruction
        -> eligibility filter and one-row-per-order payment deduplication
        -> channel/device/payment-method aggregates with definition version
        -> reconciliation and quality gates
        -> metric store, annotated dashboard, decision memo, and experiment monitor
```

The browser source emits session activity and form milestones, but it is not trusted as the payment-of-record source.
The checkout server emits validation and submit events, which help diagnose form friction but can be retried.
The payment provider callback supplies captured status and callback time; it remains the primary numerator source after `order_id` deduplication.
Release metadata binds each aggregate to layout version, client-library version, and session-rule version.
The transformation layer writes both raw and reconstructed session counts so historical dashboards remain explainable without silently rewriting the past.

### Component contracts

- **Event validator:** Rejects malformed timestamps and unknown schema versions into a quarantine count; it must not silently coerce a payment status.
- **Identity linker:** Maps permitted event records to a salted visitor hash and records linkage coverage by client library; it never exports raw personal identifiers.
- **Session reconstructor:** Applies a named inactivity threshold and emits `session_rule_version`; it can replay the same threshold for both windows.
- **Eligibility builder:** Requires cart creation and permitted country before an observation enters the denominator; it does not require downstream checkout success.
- **Order deduplicator:** Chooses one captured record per `order_id` under an explicit conflict policy and retains duplicate-attempt counts for audit.
- **Attribution resolver:** Assigns first-touch channel from a versioned rule and emits explicit `unknown` rather than fabricating direct traffic.
- **Aggregate writer:** Stores window, population version, segment, numerator, denominator, and cutoff together to prevent orphaned dashboard numbers.
- **Reconciliation gate:** Compares aggregate captured orders with the callback source and blocks publication beyond its tolerance.
- **Experiment assigner:** Assigns a durable variant before checkout rendering and logs assignment exposure even if a user abandons.
- **Decision memo generator:** Shows action, uncertainty, guardrails, and reversibility; it does not auto-execute a product change.

### State transitions and invariant

Precondition: an event has a valid source type, event-time timestamp, and permitted identity linkage state.
Transition 1: activity events are grouped into a reconstructed session when consecutive events for the same visitor are no more than 30 minutes apart.
Transition 2: a reconstructed session becomes eligible only after a cart-created event and applicable exclusion checks.
Transition 3: payment attempts collapse to one business order, and only a captured callback by cutoff may populate the numerator.
Transition 4: aggregates are emitted only with a complete metric-definition version and a quality-gate result.
Invariant: every published conversion ratio has a numerator and denominator drawn from the same population version, cutoff, and window.
Justification: storing these fields as aggregate dimensions makes a mismatched ratio detectable rather than dependent on dashboard-memory.
Recovery behavior: if a gate fails, the system publishes a stale-data annotation and a diagnostic count rather than a fresh-looking but invalid rate.
Failure boundary: stable visitor linkage is required for a reconstruction; without it, the system may preserve raw rates but must label them non-comparable.

### Design alternatives and choices

| Alternative | Benefit | Rejection or selection reason |
|---|---|---|
| Replace historical raw sessions with reconstructed sessions silently | Simplifies one chart | Rejected because it destroys auditability and hides the definition change |
| Use a unique-buyer denominator | Reduces session fragmentation | Rejected because the business question is per eligible checkout opportunity, not per person |
| Count client confirmation events as orders | Fast and rich in UI context | Rejected because confirmations can duplicate or fail independently of capture |
| Count authorization success as order | Earlier signal | Rejected because authorization is not captured revenue |
| Aggregate only total conversion | Simple executive view | Rejected because channel and device diagnostics would be impossible |
| Run a full rollback | Potentially fast harm removal | Deferred because causality and expected recovery remain uncertain |
| Run permanent 50/50 test | Strong comparison | Rejected initially because an unbounded exposure lacks a safety stop design |
| Run capped randomized holdout | Provides concurrent evidence with reversibility | Chosen subject to traffic and guardrail gates |

### Experiment and decision design

The next release creates a 20% legacy-form holdout within eligible paid-social and direct traffic, stratified by channel and mobile versus desktop.
Assignment occurs before the checkout form renders and is sticky for the session, avoiding treatment changes after a form error.
The same 30-minute reconstructed-session definition is applied to treatment and control, with a single capture cutoff and identical bot filters.
The primary estimand is the difference in captured orders per eligible comparable session between legacy and single-page assignment within the mature observation window.
The primary comparison is concurrent, not a comparison with the earlier baseline, because concurrent assignment reduces calendar, campaign, and measurement drift.
The experiment does not use AOV as a selection filter; AOV is a guardrail computed after eligible sessions are counted.

### Operational readiness gates

#### G-01 payment reconciliation
**Gate:** Distinct captured `order_id` count differs from gateway extract by no more than 0.20% for two consecutive daily runs.
**Decision effect:** Blocks outcome publication if failed.

#### G-02 assignment integrity
**Gate:** Variant assignment is present for at least 99.5% of eligible sessions and split is within ±1.0 percentage point of planned allocation.
**Decision effect:** Blocks causal comparison if failed.

#### G-03 session-rule parity
**Gate:** Every aggregate carries the same 30-minute rule version, and link coverage is reported by variant.
**Decision effect:** Blocks conversion-rate comparison if failed.

#### G-04 latency safety
**Gate:** P95 checkout submit latency for either variant stays below the predeclared operational limit.
**Decision effect:** Pauses the affected variant if failed.

#### G-05 value guardrail
**Gate:** Mature AOV difference does not breach the 2% review threshold without an owner decision.
**Decision effect:** Escalates to finance and growth review if failed.

#### G-06 support safety
**Gate:** Address-form support contacts per 1,000 exposed sessions do not breach its predeclared alert threshold.
**Decision effect:** Triggers manual issue review if failed.

#### G-07 exposure cap
**Gate:** Legacy-form exposure cannot exceed 20% without a signed review of mature evidence.
**Decision effect:** Prevents uncontrolled expansion.

#### G-08 rollback readiness
**Gate:** The flag can route traffic to the legacy form without a database migration or data deletion.
**Decision effect:** Blocks exposure if failed.

The proposed design has no learning model, feature store, or automated actor because none improves the bounded diagnosis.
Adding a classifier to predict abandonment would not solve the counterfactual problem and could distract from definition integrity.

## 9. Evaluation & Validation

Evaluation is divided into descriptive validity, experimental validity, operational safety, and decision validity.
A metric can pass arithmetic validation while failing causal validity; both states must be reported separately.

### Metric scorecard

| Metric | Numerator | Denominator | Population / window | Comparator | Status |
|---|---|---|---|---|---|
| Corrected conversion | Captured unique orders by cutoff | Eligible 30-minute reconstructed sessions | All eligible sessions; 28 days | Baseline reconstructed rate | Descriptive, computed |
| Holdout conversion | Captured unique orders assigned to a variant | Eligible sessions assigned to same variant | Mature randomized strata | Legacy versus single-page | Proposed, not executed |
| Payment integrity | Reconciled captured orders | Gateway captured orders | Daily mature cohorts | Difference from zero | Proposed gate |
| AOV guardrail | Item-only USD on captured orders | Captured orders | Variant and total, mature window | Legacy versus single-page | Proposed, not executed |
| Latency guardrail | Submit latency observations | Completed submit requests | Variant and device class | Pre-release SLO | Proposed, not executed |
| Support contact rate | Tagged contacts | Exposed eligible sessions | Variant and channel | Concurrent variant rate | Proposed, not executed |

The descriptive result is executed only in the illustrative scenario; no live experiment or company validation is claimed.
The holdout evaluation is a design recommendation, so its thresholds are proposed controls rather than achieved results.

- **Numerator reconciliation:** Compare deduplicated captured `order_id` totals with gateway extract by date and payment method. A mismatch indicates event loss, delayed callbacks, duplicate handling error, or settlement-definition difference.
- **Denominator reconstruction:** Inspect distribution of inter-event gaps and report share of current raw sessions merged under 30 minutes. An implausible merge pattern indicates identity rotation or changed event delivery.
- **Population parity:** Compare exclusion rates for bots, staff, country, and unknown channel across periods and variants. A material difference indicates the denominator is changing composition before conversion is measured.
- **Randomization balance:** Compare assigned sessions by channel, device, country, and hour before examining outcomes. Imbalance indicates assignment or logging failure, not product effect.
- **Maturity:** Exclude payment cohorts whose observed callback lag has not reached the adopted percentile. Recent pending captures can bias a variant with different payment-method mix.
- **Guardrail review:** Calculate AOV, P95 latency, payment integrity, and support rate beside conversion. A conversion gain with a breached guardrail is not automatically a winning treatment.
- **Sensitivity:** Repeat descriptive reconstruction at 20, 30, and 45 minute thresholds. A conclusion that reverses with ordinary threshold variation is not decision-stable.
- **Decision audit:** Record which metric version, cutoff, and owner signed the action. A later dispute can then distinguish new evidence from a changed definition.

### Uncertainty and decision thresholds

The test plan should report an effect estimate and uncertainty interval for each stratum and pooled result, not only a significance label.
A practical threshold must be named before outcome review; for illustration, the team could require a legacy-form advantage of at least 10 bp and positive expected contribution after margin and rework cost.
The 10 bp threshold is a policy choice in this example, not a universal marketplace standard.
The analysis should use a predeclared minimum mature exposure and avoid stopping on the first favorable daily point.
Daily monitoring remains appropriate for safety guards, but efficacy inference needs the planned observation rule.
A sequential design could use alpha spending or a Bayesian decision rule, but its choice must be specified before seeing variant outcomes.
If the team lacks statistical design expertise, it should keep the holdout small and use a conservative release decision rather than treating a dashboard chart as a randomized result.

### Illustrative decision table

| Mature holdout result | Payment / latency / AOV guards | Recommended response | Rationale |
|---|---|---|---|
| Legacy advantage below 10 bp with interval spanning zero | All pass | Retain single-page, continue monitoring | Evidence does not clear practical threshold |
| Legacy advantage at least 10 bp and contribution threshold met | All pass | Plan a targeted revert or retain legacy for affected strata | Benefit is decision-material and measured concurrently |
| Single-page advantage at least 10 bp | All pass | Expand only after cap review | Positive result still needs gradual operational scaling |
| Any conversion result | Payment integrity fails | Stop interpretation and repair numerator | Outcome metric is not trustworthy |
| Any conversion result | AOV declines beyond review threshold | Hold expansion and investigate value mechanism | Order count alone is insufficient |
| Any conversion result | P95 latency breaches limit | Pause affected treatment | Technical harm can overwhelm rate benefit |
| Effects differ strongly by channel/device | All pass | Preserve stratified decision; do not average away heterogeneity | Aggregate can conceal a harmful segment |

### Validation scenarios

#### Scenario A: successful legacy holdout
**Evidence pattern:** Legacy has a 15 bp higher captured conversion in paid-social mobile sessions, gateway reconciliation passes, and AOV is neutral.
**Decision:** A targeted legacy fallback for paid-social mobile is evidence-consistent; a full-site revert still needs its own value case.

#### Scenario B: raw metric falls again
**Evidence pattern:** Raw rate drops while comparable conversion and gateway orders remain stable.
**Decision:** Annotate the raw dashboard and inspect session-rule or identity changes before declaring demand loss.

#### Scenario C: numerator disagreement
**Evidence pattern:** Gateway captures exceed deduplicated event orders by 1.2%.
**Decision:** Block conversion publication; an event-loss issue can mimic abandonment.

#### Scenario D: traffic-quality shift
**Evidence pattern:** Paid-social conversion declines in both variants after a creative campaign change.
**Decision:** Treat campaign quality as a shared shock; variant contrast remains usable only if assignment is balanced.

#### Scenario E: AOV tradeoff
**Evidence pattern:** Single-page produces more orders but 4% lower item-only AOV.
**Decision:** Estimate contribution, product mix, refund risk, and seller economics before selecting a winner.

#### Scenario F: capacity incident
**Evidence pattern:** Callback processing lags for one payment method during a campaign burst.
**Decision:** Extend maturity window and do not count pending orders as failures until reconciliation resolves them.

#### Scenario G: identity-link degradation
**Evidence pattern:** New client version lowers visitor-hash continuity.
**Decision:** Report raw and reconstructed series separately; do not force a fabricated correction.

#### Scenario H: treatment leakage
**Evidence pattern:** A cached page shows the legacy form to some assigned single-page sessions.
**Decision:** Analyze by assignment for intent-to-treat, log actual exposure, and repair rendering before expansion.

The evaluation record should distinguish an observed fact, a derived quantity, a proposed test, and an unmeasured outcome.
That distinction prevents a future reviewer from reading a proposed guardrail as if it had already passed.

## 10. Action Taken

Because this is an illustrative case, no real deployment, rollback, or business outcome occurred.
The action record below is the recommended decision protocol for the fictional owner, with ownership and evidence gates made explicit.

### Recommended action sequence

#### 1. Freeze the decision snapshot
**Owner action:** Analytics lead stores the source IDs, cutoff, rule versions, and the 28-day tables used in the review.
**Purpose:** Prevents a moving dashboard from changing the claim after the meeting.

#### 2. Annotate raw conversion
**Owner action:** BI owner labels the 2026-10-06 session-definition change and stops using raw pre/post rate as the release score.
**Purpose:** Prevents a known comparability failure from reaching finance unchanged.

#### 3. Reconcile captured orders
**Owner action:** Payments operations compares daily unique captures against the independent callback extract by payment method.
**Purpose:** Establishes whether the numerator is credible enough to test user behavior.

#### 4. Audit reconstruction coverage
**Owner action:** Data engineering reports visitor-link continuity and merged-session share by client library and channel.
**Purpose:** Bounds confidence in the corrected denominator.

#### 5. Review concurrent releases
**Owner action:** Release manager identifies any checkout, payment, campaign, or attribution changes overlapping the windows.
**Purpose:** Prevents an oversimplified single-cause narrative.

#### 6. Create an exposure flag
**Owner action:** Checkout engineering verifies a reversible, durable assignment before form rendering.
**Purpose:** Makes a concurrent holdout operationally feasible.

#### 7. Define guardrail thresholds
**Owner action:** Growth, finance, support, and payments owners write thresholds and response owners before outcome data are reviewed.
**Purpose:** Avoids post-hoc threshold shopping.

#### 8. Start capped stratified holdout
**Owner action:** Eligible paid-social and direct traffic receives a 20% legacy-form arm, stratified by mobile and desktop.
**Purpose:** Creates evidence where risk and diagnostic value are highest.

#### 9. Monitor safety daily
**Owner action:** Operations monitors payment integrity, latency, error codes, and support signal; it does not declare efficacy from daily noise.
**Purpose:** Separates safety response from statistical decision timing.

#### 10. Review mature result
**Owner action:** At the planned maturity point, the group compares corrected conversion, AOV, and expected contribution.
**Purpose:** Converts analysis into a documented action rather than a dashboard discussion.

#### 11. Decide segment scope
**Owner action:** Growth selects retain, targeted fallback, or broader revert according to the decision table.
**Purpose:** Avoids assuming a global aggregate requires a global change.

#### 12. Archive the decision record
**Owner action:** Analytics records action, rationale, dissent, unresolved assumptions, and recheck trigger.
**Purpose:** Makes later learning possible without reconstructing meeting memory.

### Escalation and recovery path

If payment reconciliation fails, stop conversion interpretation before changing the checkout layout; a revert cannot correct an untrusted numerator.
If the feature flag cannot guarantee stable assignment, do not start the holdout; repair control-plane behavior before exposing buyers.
If latency or payment failures breach a safety threshold, immediately route the affected stratum to the known safer form and preserve logs for diagnosis.
If all integrity checks pass but the result remains ambiguous, retain the cap and extend the mature window rather than converting ambiguity into a universal rollout.
If a targeted fallback wins, document why the remaining segments retain the single-page form and schedule a re-evaluation after the next material campaign or session-rule change.
If the full revert threshold is crossed, calculate a current cost estimate before execution; the original USD 18,000 planning estimate is not a perpetual fact.

### Decision communication

The growth lead’s memo should begin with the corrected descriptive rate, then explicitly say why it is not causal proof.
Finance should receive separate lines for order-equivalent gap, AOV movement, revenue-equivalent scenario, and any contribution-margin assumption.
Engineering should receive the affected segments, gate conditions, rollback capability requirement, and event-contract changes rather than a vague request to “fix conversion.”
Support should receive observable form symptoms and a feedback tagging rule, not an instruction to use contacts as a conversion numerator.
Executives should receive the action selected, what could reverse it, and the date of the next evidence review.

## 11. Outcome & Results

No company action was executed and no company result was observed; the outcomes in this section are therefore limited to computed **ILLUSTRATIVE** diagnostic results and the proposed decision state.
The raw dashboard calculation yields 3.1100% current conversion against 3.4200% baseline conversion.
The common-rule reconstruction yields 3.2150% current conversion, reducing the apparent decline from 31.00 bp to 20.50 bp.
The constructed correction accounts for 40,500 of 60,500 raw-session growth, or 66.94% of that growth, without claiming that all remaining movement is behavioral.
The baseline-weight decomposition produces a -20.599 bp within-channel term and an approximately +0.099 bp mix term.
The illustrative rate-equivalent loss at current comparable volume is 2,460 orders over 28 days.
The illustrative baseline AOV is USD 57.3898 and current AOV is USD 55.5752, a decline of USD 1.8146 or 3.16%.
The illustrative revenue-equivalent gap is USD 141,178.91 when baseline AOV is held constant; this is not realized revenue, margin, or recovered value.

### Result classification

| Result | Evidence class | Executed status | Decision use | Limitation |
|---|---|---|---|---|
| Raw 31.00 bp decline | Derived from constructed counts | Computed illustration | Alert only | Denominator definition changed |
| Corrected 20.50 bp decline | Derived from constructed reconstruction | Computed illustration | Descriptive primary signal | Depends on identity linkage and gap rule |
| Paid-social 40.00 bp decline | Derived segment result | Computed illustration | Prioritization signal | Not causal by itself |
| +0.10 bp mix term | Derived decomposition result | Computed illustration | Rejects mix as main arithmetic explanation | Requires stable channel taxonomy |
| 3.16% AOV decline | Derived from constructed USD fields | Computed illustration | Guardrail investigation trigger | Does not equal margin change |
| Capped holdout | Proposed experiment | Not executed | Recommended next evidence | No outcome exists yet |
| Full revert | Candidate action | Not executed | Reserved for threshold crossing | Recovery probability unknown |

### What changed in the decision record

Before diagnosis, the draft narrative would have stated that conversion fell roughly 9% and implied that the new checkout caused it.
After diagnosis, the record states that the raw metric is not comparable, the common-rule descriptive movement is smaller but still material, and causal attribution requires concurrent evidence.
Before diagnosis, the financial statement would multiply a raw-rate decline by traffic without noting denominator fragmentation.
After diagnosis, the financial statement uses a comparable-rate scenario and labels the result as revenue-equivalent, excluding margin, refunds, and recovery probability.
Before diagnosis, paid social might receive a generic campaign adjustment based on an aggregate.
After diagnosis, paid social and direct are prioritization strata for a reversible holdout, while organic and email remain monitored.

### Outcome review cadence

| Cadence | Inputs | Accountable role | Output |
|---|---|---|---|
| Daily operational review | Payment reconciliation delta, callback maturity, assignment integrity, latency, and error rates. | Operational analyst and payments owner. | Pause or repair safety failures; do not claim efficacy. |
| Twice-weekly diagnostic review | Channel-device rate patterns, identity-link coverage, support themes, and campaign changes. | Analytics lead and product partners. | Decide whether the measurement assumptions remain valid. |
| Mature evidence review | Planned holdout effect, uncertainty, AOV, contribution sensitivity, and rework cost. | Growth lead with finance and engineering. | Choose retain, targeted fallback, broader rollback, or extended observation. |
| Post-decision audit | Versioned decision memo, actual action, dissent, and trigger for revisiting. | Analytics governance owner. | Preserve accountability and prevent retrospective metric substitution. |

## 12. Failure Modes & What Went Wrong

The original failure was a governance failure before it was an arithmetic failure: a changed session definition reached an executive dashboard without a comparability annotation.
The cases below distinguish mechanisms because “monitor conversion” cannot tell an operator which corrective action is safe.

### F-01 Session fragmentation
**Mechanism:** A shorter heartbeat closes a raw session and creates multiple denominators for the same browsing episode.
**Discriminating check:** Raw sessions rise abruptly at client-version adoption while reconstructed sessions rise much less; inter-event gaps cluster between five and thirty minutes.
**Correction:** Replay both windows using a versioned common inactivity rule and show raw versus reconstructed series together.
**Prevention:** Do not silently overwrite historical raw counts; retain a definition-change annotation and test identity-link coverage.

### F-02 Inner-join denominator loss
**Mechanism:** An analyst joins sessions to orders with an inner join, dropping every non-converting session.
**Discriminating check:** The calculated conversion approaches 100%, and eligible-session count equals order-linked-session count.
**Correction:** Use a left join from eligible sessions to one-row-per-order data and assert that zero-order sessions remain.
**Prevention:** Add a pipeline test that expects a large count of null order IDs in the denominator table.

### F-03 Payment retry duplication
**Mechanism:** Multiple callbacks or event retries create several records for one business order.
**Discriminating check:** Raw confirmation events exceed distinct captured `order_id` counts, often concentrated around timeout or retry codes.
**Correction:** Deduplicate on stable `order_id` with a documented captured-state precedence and reconcile to gateway extract.
**Prevention:** Retain retry counts as a quality metric rather than deleting evidence of payment instability.

### F-04 Callback maturity bias
**Mechanism:** Recent current-window orders are still pending or have not arrived at the snapshot cutoff.
**Discriminating check:** The apparent decline is largest in the last hours or days and narrows when only mature cohorts are compared.
**Correction:** Set a maturity lag from historical callback distribution and exclude or flag immature cohorts.
**Prevention:** Publish cutoff time and late-arrival policy with every daily conversion view.

### F-05 Attribution drift
**Mechanism:** Cookie behavior, UTM parser logic, or first-touch rules move sessions among channels without buyer behavior changing.
**Discriminating check:** Unknown-channel share or channel proportions jump on the same day as an attribution release while total comparable conversion is stable.
**Correction:** Version the resolver, preserve unknown as a category, and rerun a common attribution rule for decomposition.
**Prevention:** Do not treat a channel table as causal evidence until taxonomy parity passes.

### F-06 Campaign-quality confounding
**Mechanism:** A new creative or bidding target changes visitor intent within paid social.
**Discriminating check:** Both randomized variants decline similarly inside the campaign cohort, while non-campaign paid social is stable.
**Correction:** Add campaign and creative strata, compare concurrent variants, and coordinate with marketing before product attribution.
**Prevention:** Record campaign launches beside product releases in the diagnostic calendar.

### F-07 Device-specific form defect
**Mechanism:** Address autocomplete, keyboard behavior, or validation differs on mobile browsers.
**Discriminating check:** Form-error events, latency, or support contacts rise in mobile treatment but not desktop treatment under the same campaign.
**Correction:** Inspect field-level error codes, browser versions, and replay flows; route only the affected device class to legacy if safety requires.
**Prevention:** Keep device version in exposure logs and exercise mobile browser regression tests before release.

### F-08 Payment-method interaction
**Mechanism:** The layout changes how wallets, cards, or redirects return to checkout.
**Discriminating check:** Conversion gap concentrates in one payment method and gateway captures or redirect-return events differ by variant.
**Correction:** Stratify by payment method, inspect callback and redirect state, and avoid a global layout conclusion.
**Prevention:** Treat payment method as a guardrail slice, not an afterthought to aggregate conversion.

### F-09 Bot-filter rule change
**Mechanism:** A new fraud or bot classifier excludes a different fraction of visits after the release.
**Discriminating check:** Excluded-session rate shifts sharply with classifier version, especially in one channel, while raw events remain stable.
**Correction:** Apply a common frozen exclusion rule to both windows and quantify sensitivity to the rule.
**Prevention:** Version bot rules with metrics and alert on denominator exclusion changes.

### F-10 Time-zone boundary error
**Mechanism:** A local-day dashboard includes different UTC hours across daylight or reporting rules.
**Discriminating check:** The first or last day has partial volume, and re-aggregation by UTC changes the result materially.
**Correction:** Use complete UTC days with explicit inclusive and exclusive timestamps.
**Prevention:** Store event time, processing time, and reporting-zone choice in the metric contract.

### F-11 Order-to-session orphaning
**Mechanism:** Captured orders cannot be linked to an eligible session because identity or cart linkage is missing.
**Discriminating check:** Orphan order rate rises by client version or payment method and reconciliation passes only at total-order level.
**Correction:** Report an orphan bucket, investigate linking loss, and avoid silently assigning it to direct traffic.
**Prevention:** Track linkage success as an independent quality gate.

### F-12 Simpson aggregation
**Mechanism:** Aggregate conversion improves while a high-value segment deteriorates, or the reverse, due to changing segment weights.
**Discriminating check:** Segment rates and weights tell a different story from the total; standardized rates change sign relative to raw total.
**Correction:** Use fixed-weight summaries and segment guardrails before selecting a global action.
**Prevention:** Define required slices at metric-design time rather than after an executive question.

### F-13 Threshold gaming
**Mechanism:** Owners alter exposure, window, or inclusion rules after seeing an unfavorable result.
**Discriminating check:** Decision thresholds appear only after outcome review or change repeatedly between meetings.
**Correction:** Timestamp the analysis plan, approval, and any justified amendment; separate safety pauses from efficacy stopping.
**Prevention:** Require an owner and reason for every metric-contract revision.

### F-14 Rollback non-reversibility
**Mechanism:** The legacy form cannot safely coexist because a migration, cache, or payment state is one-way.
**Discriminating check:** Flag test shows inconsistent cart state or a revert requires destructive data repair.
**Correction:** Prove rollback in a safe environment before live exposure; make the experiment cap contingent on it.
**Prevention:** Treat reversibility as a launch gate, not a recovery promise.

These failures are intentionally non-overlapping in their primary mechanism, though several can co-occur in a real incident.
For example, an attribution change can hide a device defect, and callback lag can amplify a session-definition error near a window boundary.
The response order is therefore to establish numerator integrity, denominator integrity, and population parity before diagnosing product behavior.

## 13. Transferable Lesson

A conversion diagnosis is a measurement-contract problem before it becomes a product-optimization problem.
The formula `orders / sessions` is simple, but every word in that fraction encodes eligibility, identity, time, deduplication, and payment-state choices.
A large raw movement should create urgency, not permission to skip comparability checks.
A corrected descriptive movement should create a focused experiment, not a retrospective causal story.
A segment with the biggest rate drop is a prioritization signal; it is not automatically the root cause.
A channel mix decomposition can rule out one arithmetic explanation without proving why within-channel rates moved.
A revenue translation is most useful when it names what it holds constant and what it excludes.
A rollback is an investment decision: its expected recovered contribution must be weighed against cost, delay, reversibility, and evidence quality.
A dashboard annotation is part of the product system; silent metric-definition changes create operational debt.
A guardrail is not a secondary chart: it is a condition that can reverse the action selected by the primary metric.

### Reusable diagnostic sequence

- **Name the decision:** Specify owner, deadline, alternatives, irreversible cost, and what counts as a reversible action.
- **Freeze the metric:** Write numerator, denominator, population, keys, time boundary, and rule versions before calculating change.
- **Reconcile the numerator:** Use a payment-of-record source and deduplicate business orders before trusting an event count.
- **Rebuild denominator parity:** Detect definition changes and reconstruct or explicitly bound their effect.
- **Stratify purposefully:** Choose slices that represent plausible mechanisms, such as channel, device, payment method, and cohort.
- **Decompose arithmetic:** Separate within-segment movement from composition before blaming a source of traffic.
- **Translate carefully:** Convert a rate gap to orders and economics only with labeled assumptions and sensitivity.
- **Design concurrent evidence:** Use a durable randomized holdout or acknowledge that causal attribution remains unresolved.
- **Operate with gates:** Treat integrity, latency, value, and support as explicit stop conditions.
- **Record reversibility:** Preserve the action, evidence, uncertainty, and recheck triggers for the next release.

## 14. Connections & Cross-References

This case connects metric-tree design to an operational decision: a north-star rate is useful only when its leaf metrics share definitions and time boundaries.
It connects funnel analysis to data modeling: funnel stages cannot be interpreted when session, order, and payment grains are joined incorrectly.
It connects segmentation to causal inference: a segment table localizes risk, whereas randomized exposure is needed to attribute an intervention.
It connects dashboard design to governance: definition versions and annotations are features that preserve semantic continuity.
It connects stakeholder narratives to economics: finance needs an assumption-labeled order and contribution translation, not a decorative revenue number.
It connects experimentation design to release engineering: a reversible flag and durable assignment are prerequisites for reliable learning.
Suggested related study topics are funnel/cohort analysis, segmentation for decisions, dashboard operational alerts, causal inference, and metric trees.
Open question: how should the organization set a practical effect threshold when contribution margin varies materially by seller and category?
Open question: how should a privacy-preserving identity design quantify reconstruction uncertainty without retaining excessive behavioral detail?
Open question: can a pre-release synthetic journey test detect session-rule regression before it reaches an executive dashboard?

### Interview pack

#### Question 1. Why is the raw conversion comparison invalid?
**Answer:** The raw denominator uses different inactivity rules: 30 minutes at baseline and 5 minutes currently.
**Reasoning:** A shorter rule can split one browsing episode into several sessions, lowering orders per raw session without equivalent behavioral change.
**Follow-up:** What fields would you inspect to reconstruct sessions?
**Red flag:** Saying “more sessions means more demand” without checking rule version.
**Scoring:** Full credit names numerator/denominator parity and stable linkage.
**Acceptable alternatives:** State that the raw series is an alert but not a pre/post decision metric.
**Answer support:** Sections 5a, 6.1, and 6.2.

#### Question 2. Compute the baseline conversion rate from 40,356 orders and 1,180,000 sessions.
**Answer:** 40,356 / 1,180,000 = 0.0342 = 3.4200%.
**Reasoning:** Orders divided by eligible comparable sessions yields orders per session.
**Follow-up:** What unit does the result have before percentage formatting?
**Red flag:** Dividing sessions by orders or calling 3.42 a count.
**Scoring:** Full credit shows division, decimal, percent, and unit.
**Acceptable alternatives:** 0.03420 orders/session is equivalent before multiplying by 100.
**Answer support:** Formula C-01.

#### Question 3. Recompute current comparable conversion after merging 40,500 of 1,240,500 raw sessions.
**Answer:** Comparable sessions are 1,200,000, and 38,580 / 1,200,000 = 3.2150%.
**Reasoning:** The merged amount changes the denominator only; captured orders stay fixed pending numerator reconciliation.
**Follow-up:** What would make this correction invalid?
**Red flag:** Subtracting sessions without a stable visitor linkage or common rule.
**Scoring:** Full credit derives both denominator and rate.
**Acceptable alternatives:** Provide a range if linkage coverage is incomplete.
**Answer support:** Formula C-02.

#### Question 4. What does a +0.10 bp mix term mean here?
**Answer:** It means listed channel-share changes slightly offset, rather than explain, the corrected aggregate decline.
**Reasoning:** The fixed-weight rate is about 3.21401% and actual current comparable rate is 3.21500%.
**Follow-up:** Which condition must hold before using this decomposition?
**Red flag:** Claiming that mix caused the -20.50 bp decline.
**Scoring:** Full credit distinguishes arithmetic decomposition from causal explanation.
**Acceptable alternatives:** Call it effectively zero after stating its rounded magnitude.
**Answer support:** Formula C-03.

#### Question 5. Which segment should be prioritized for diagnosis and why?
**Answer:** Paid social is first because its 40.00 bp rate decline combines a material volume share with a large movement.
**Reasoning:** Direct is also worth inspection because its decline is large, but it has fewer sessions.
**Follow-up:** What additional slice would you add?
**Red flag:** Declaring paid social the cause without device, campaign, or payment evidence.
**Scoring:** Full credit uses both rate magnitude and exposure volume.
**Acceptable alternatives:** Prioritize direct first if evidence shows an urgent payment-method concentration.
**Answer support:** Section 6.3.

#### Question 6. Calculate the illustrative rate-equivalent order gap.
**Answer:** 1,200,000 × (0.03420 - 0.03215) = 1,200,000 × 0.00205 = 2,460 orders.
**Reasoning:** It holds traffic at current comparable sessions and compares baseline versus current conversion.
**Follow-up:** Why is this not a causal recovery forecast?
**Red flag:** Calling all 2,460 orders guaranteed rollback recovery.
**Scoring:** Full credit shows the rate difference and multiplication.
**Acceptable alternatives:** State that contribution margin and recovery probability are needed for action economics.
**Answer support:** Formula C-04.

#### Question 7. What is the most dangerous join error in this case?
**Answer:** An inner join from sessions to orders removes non-converting sessions and destroys the denominator.
**Reasoning:** The resulting rate can approach 100% because only order-linked sessions remain.
**Follow-up:** How would you test for it?
**Red flag:** Using an inner join because it has fewer nulls.
**Scoring:** Full credit proposes a left join and a null-order assertion.
**Acceptable alternatives:** Compare pre-join eligible sessions to post-join distinct sessions.
**Answer support:** Failure F-02.

#### Question 8. A gateway extract has 1.2% more captured orders than the dashboard. What should happen?
**Answer:** Stop interpreting conversion and reconcile the numerator before changing checkout design.
**Reasoning:** The difference could be event loss, callback lag, duplicate collapse, or a definition mismatch.
**Follow-up:** Which breakdown narrows the cause?
**Red flag:** Rolling back immediately because lower dashboard counts prove friction.
**Scoring:** Full credit names at least two competing causes and a segmented reconciliation.
**Acceptable alternatives:** Freeze the dashboard with a data-quality annotation while repair proceeds.
**Answer support:** Sections 5d and 12 F-03/F-04.

#### Question 9. Design a valid next experiment under a 20% legacy-form cap.
**Answer:** Randomize durable legacy versus single-page assignment within eligible paid-social and direct sessions, stratified by channel and device.
**Reasoning:** Concurrent assignment controls shared calendar and campaign shocks better than the earlier baseline.
**Follow-up:** What are the primary and guardrail metrics?
**Red flag:** Comparing next week’s total conversion with last month’s raw rate.
**Scoring:** Full credit states assignment timing, denominator parity, mature cutoff, and stop gates.
**Acceptable alternatives:** Include all channels if capacity and risk policy permit.
**Answer support:** Section 8 experiment design and Section 9 scorecard.

#### Question 10. How should AOV influence the decision if conversion rises?
**Answer:** Treat a 3.16% AOV decline as a guardrail investigation, then estimate contribution and mix before calling the variant better.
**Reasoning:** More orders can still reduce economic value if order composition, discounting, or refund behavior worsens.
**Follow-up:** What monetary definition must stay fixed?
**Red flag:** Ignoring AOV because conversion is the north-star metric.
**Scoring:** Full credit distinguishes item-only AOV, revenue, and contribution margin.
**Acceptable alternatives:** Use gross margin per eligible session as the combined objective if data are mature.
**Answer support:** Sections 6.5 and 9.

#### Question 11. What failure could make paid-social look worse in both variants?
**Answer:** A campaign-quality shift can lower visitor intent independently of the checkout layout.
**Reasoning:** If both randomized arms decline similarly within the campaign cohort, the shared shock is not a treatment effect.
**Follow-up:** What check would you run?
**Red flag:** Assuming randomization makes campaign metadata unnecessary.
**Scoring:** Full credit proposes campaign or creative strata and concurrent comparison.
**Acceptable alternatives:** Pause a newly launched campaign as an operational mitigation.
**Answer support:** Failure F-06.

#### Question 12. When is a full revert justified in this decision framework?
**Answer:** After metric integrity passes, a mature holdout shows a practical legacy advantage, guards pass or favor safety, and expected recovered contribution exceeds current revert cost.
**Reasoning:** A revert is an investment under uncertainty, not a ceremonial response to a red dashboard.
**Follow-up:** What could change this decision tomorrow?
**Red flag:** Using raw percentage decline alone as a sufficient trigger.
**Scoring:** Full credit states evidence, economics, and reversibility conditions.
**Acceptable alternatives:** An emergency safety failure can justify immediate rollback before efficacy evidence.
**Answer support:** Sections 7, 9, and 10.

## 15. Quick Reference Card

| Field | Decision-ready card |
|---|---|
| Decision | Use a capped randomized legacy-form holdout after metric repair; do not use raw pre/post conversion as causal proof. |
| Primary metric | Distinct captured orders / eligible 30-minute reconstructed sessions, with shared cutoff and population. |
| Key result | ILLUSTRATIVE corrected conversion: 3.2150% current versus 3.4200% baseline, a -20.50 bp descriptive gap. |
| Key decomposition | ILLUSTRATIVE within-channel movement: about -20.60 bp; listed-channel mix: about +0.10 bp. |
| Guardrails | Payment reconciliation, assignment integrity, P95 submit latency, item-only AOV, and tagged support contacts. |
| Economic lens | ILLUSTRATIVE 2,460-order rate-equivalent gap; USD 141,178.91 revenue-equivalent at baseline AOV; not a causal recovery forecast. |
| Stop / revert rule | Pause on integrity or safety breach; consider targeted or full revert only after mature concurrent evidence and current cost review. |
| Evidence / stage | Illustrative constructed arithmetic; proposed experiment and action design; no deployment or measured company outcome. |
| Review trigger | Reassess when session, payment, attribution, release, campaign, or cost assumptions change. |
