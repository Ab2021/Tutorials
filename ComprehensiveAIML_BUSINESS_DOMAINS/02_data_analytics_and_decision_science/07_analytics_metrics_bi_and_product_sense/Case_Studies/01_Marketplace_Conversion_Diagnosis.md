---
case_id: "CASE-2026-ANA001"
case_title: "Marketplace Conversion Diagnosis: Separating a Real Funnel Decline from a Session-Definition Change"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "marketplace analytics / conversion and channel mix"
primary_pattern: "decision-ready-metric-contract"
secondary_patterns: ["component-reconciliation", "guardrail-governance"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "analytics lead (marketplace growth)"
source_index: "INDEX.md"
review_trigger: "Session or order definition, channel attribution rule, comparison window, or revert policy changes"
---

# Marketplace Conversion Diagnosis

## 1. The One-Line Framing

> Given **28 days of identity-level session and order records from a fictional two-sided marketplace, split into a baseline window and a current window**, decide **whether to revert, keep, or narrow a checkout-layout change** for **the marketplace growth lead** before the next release train, subject to **a one-release rollback cost of USD 18,000 and a guardrail on order value**, compared with **the do-nothing baseline of leaving the change live and accepting the current funnel rate**.

This is an illustrative decision exercise. The company, windows, traffic, and money are constructed; the arithmetic is checked and reproducible by hand. No production deployment, company result, or measured efficacy is claimed anywhere in this case.

The task category is **diagnosis under a decision deadline**, repeated weekly while the release train runs. One primary success condition: after the artefact correction described in Section 6, the owner can state a conversion rate, a revenue consequence, and a revert decision whose supporting numbers reconcile to the same frozen snapshot.

The primary reusable pattern is `decision-ready-metric-contract`; the two secondary patterns are `component-reconciliation` and `guardrail-governance`. Both secondary patterns are load-bearing here: the aggregate rate alone is not diagnosable, and the guardrail is what turns an ambiguous rate movement into an action.

The decisive tradeoff, established in Sections 6 and 7, is that **a −31.00 basis-point aggregate rate movement decomposes into −32.95 bp of within-channel rate change and +1.95 bp of channel-mix change, and roughly half of the observed denominator growth is a measurement artefact rather than demand.** The obvious simpler framing — "conversion fell 9%, revert the checkout change" — misses that the artefact inflates the denominator and that one channel carries most of the genuine decline. Reverting on the raw number spends USD 18,000 to fix a problem that is partly a reporting defect.

## 2. Context & Background

CraftBay is a fictional marketplace where independent makers list physical goods and buyers purchase through a single checkout. Sellers are not analysed in this case; the unit of analysis is the buyer session.

| Context field | This case |
|---|---|
| Domain and product | Consumer marketplace; checkout funnel from session start to paid order |
| Workflow before the decision | Buyer lands from a channel, browses, adds to cart, enters checkout, pays; the order is confirmed by a payment callback |
| Workflow after the decision | Growth lead either reverts the layout, keeps it, or narrows it to one channel; analytics publishes the corrected number and the instrumentation ticket |
| Stakeholders | Growth lead (decision owner), analytics lead (author), checkout engineering, payments operations, finance (revenue reporting) |
| Baseline process | Weekly aggregate funnel dashboard built from a single events table; no channel decomposition, no frozen snapshot |
| Scale | 1.18M eligible sessions and 40,356 orders in the baseline window; 1.24M sessions and 38,580 orders in the current window |
| Timing | Baseline window 2026-09-08 to 2026-10-05 inclusive; current window 2026-10-06 to 2026-11-02 inclusive; both 28 days |
| Geography and boundaries | One storefront, five buyer countries, single currency USD; tax and shipping excluded from order value |
| Prior attempts | A cart-recovery email test ran in August and was inconclusive at the available sample; it is outside both windows |
| Applicable requirements | Internal metric-contract policy: each decision metric names owner, grain, eligibility, cutoff, numerator and denominator |

The change under suspicion is a checkout layout released to 100% of traffic on 2026-10-06: the three-step address form was replaced by a single page, and the order-confirmation event was re-instrumented in the same release. Two things changed at once, which is the structural reason this case is a diagnosis problem rather than a measurement problem.

The aggregate dashboard that the team actually used reports sessions from a client-side heartbeat with a 30-minute inactivity window. The same release shipped a client library that shortened the heartbeat to 5 minutes. Nobody registered that as a metric change, because the events table schema did not change.

## 3. The Business Question

The decision owner's question, paraphrased from the operating review agenda (not a quotation from a real person): "Conversion is down nine percent since the checkout release. Do we roll the layout back before Thursday, and what do we tell finance about the revenue gap?"

| Decision contract | Fill in |
|---|---|
| Question | Should the single-page checkout stay for all traffic, be narrowed to selected channels, or be reverted? |
| Decision owner | Growth lead, with checkout engineering as the executing team |
| Trigger and deadline | Two consecutive weekly reviews show a declining aggregate rate; the release train closes Thursday 2026-11-05 |
| Available actions | (a) keep at 100%, (b) revert to the three-step form, (c) keep but exclude one channel, (d) defer and repair instrumentation first |
| Value and downside | Revert costs USD 18,000 in engineering and a two-week delay of a conversion experiment; keeping a harmful layout costs order volume every day |
| Objective | Maximise paid orders per eligible session at a fixed traffic level, without degrading order value |
| Hard constraints | No change may be evaluated on a population whose denominator is defined differently between windows |
| Scope | Buyer sessions and paid orders, 28-day windows, all channels, one storefront |
| Out of scope | Seller-side supply, refunds after 30 days, paid-media bidding strategy, causal attribution of the layout itself |

The business objective is paid orders per eligible session. The proxy in the dashboard is sessions per the *client* definition, which is exactly where the two windows diverge. Naming that conflict is the first analytical act of this case, and the guardrail (order value per paid order) exists because a layout that raises order count while collapsing basket value would satisfy the primary metric and harm the business.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Both windows are 28 complete days with no partial-day truncation | data | Row counts by day; last-day completeness check against payment callback lag | verified in the snapshot used here | A partial final day biases the current rate downward | Trim the final day from both windows | Window boundary change / analytics lead |
| A-02 | `order_id` is unique per paid order and is not reused across retries | data | Duplicate count on `order_id`; compare to payment callback count | verified (zero duplicates in snapshot) | Double-counted orders inflate the numerator | Deduplicate on `order_id` before aggregation | Payment schema change / payments ops |
| A-03 | Channel is assigned from the first-touch record and does not move between windows | data | Compare channel-assignment rule version in both windows | unverified | Mix effect is an attribution artefact, not behaviour | Recompute with last-touch as a sensitivity | Attribution rule change / growth |
| A-04 | Session identity is stable within a window | data | Compare session-id generation code version | **violated** — heartbeat changed 30 min to 5 min on 2026-10-06 | Denominator inflates; rate falls mechanically | Re-derive corrected sessions from the 30-minute rule | Client library release / checkout engineering |
| A-05 | The checkout release is the only material storefront change in the window | business | Release calendar review | unverified | Another change confounds the comparison | Split the window at the second change | Any release / release manager |
| C-01 | Revert consumes one release train and USD 18,000 of engineering | cost | Engineering estimate, single point | confirmed as a planning estimate | The revert decision is over- or under-priced | Re-quote before committing | Cost model change / engineering manager |
| C-02 | The decision must be made by 2026-11-05 | schedule | Release calendar | confirmed | A correct but late analysis has no effect | Interim keep plus instrumentation repair | Calendar change / growth lead |
| C-03 | Finance reports revenue from the same order snapshot | governance | Report-to-snapshot reconciliation | unverified | Two revenue numbers circulate | Publish the reconciliation delta with the memo | Finance close / finance |

The three assumptions most likely to reverse the recommendation are A-04 (already violated, and quantified in Section 6), A-05 (a second change would break the single-cause story), and A-03 (attribution movement would make the mix term meaningless). C-01 matters because the decision is a cost trade, not a purity test: at USD 18,000, reverting is justified only if the ongoing loss exceeds roughly USD 3,600 per week over a five-week horizon, which Section 6 compares against.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Detail |
|---|---|
| Unit of analysis | One eligible buyer session; orders are attached to the session that produced them |
| Volume and window | Baseline: 1,180,000 sessions, 40,356 orders, USD 2,315,986 revenue. Current: 1,240,500 sessions, 38,580 orders, USD 2,144,110 revenue |
| Inputs | `session_id`, `channel`, `session_start_utc`, `order_id`, `order_value_usd`, `country`, `client_lib_version`, `cart_created` flag |
| Outcome definition | A paid order: payment callback with status `captured`, deduplicated on `order_id`, attributed to the session that created the cart |
| Population and exclusions | Sessions with a non-empty cart; excludes internal staff sessions (flagged), bot-classified sessions, and sessions in the five non-supported countries |
| Join semantics | Session-to-order is one-to-many at the raw event level but zero-or-one at the business level; the snapshot enforces one order per session after payment deduplication |
| Splits | Two calendar windows of 28 days each; no randomisation, no holdout |
| Frozen snapshot | `snap-2026-11-03T02:00Z`, immutable; all numbers in this case come from it |

The channel dimension has four members: organic search, paid social, email, direct. Direct means "no attributable campaign", which is why it is the most likely home of attribution drift and of returning-customer behaviour.

### 5b. Tools & Stack

The analysis is specified as warehouse SQL plus a small amount of checked arithmetic, because the deliverable must be reproducible by a reviewer in a different organisation.

| Component | Role | Why it is here |
|---|---|---|
| Columnar warehouse | Stores the frozen session/order snapshot; executes grouped aggregations | The decomposition needs four-way grouping, not row-level iteration |
| SQL with window functions and CTEs | Computes per-channel counts, weights, and the corrected denominator | Keeps the metric logic auditable and reviewable |
| Snapshot/version control | Freezes both windows under one identifier | A moving "latest" table cannot reproduce a published decision |
| Notebook or spreadsheet arithmetic | Independent recomputation of the decomposition | Catches SQL join errors that produce plausible totals |
| Dashboard with versioned metric definitions | Publishes the corrected metric with its contract | Prevents re-derivation from the wrong definition |

No experimentation platform was used, because no experiment existed. That absence is a limitation of the case, not a design choice to celebrate: it caps every causal claim at "consistent with", and Module 08 is the correct handoff.

### 5c. Method Summary

Input: the frozen snapshot. Transformation: eligibility filter, payment deduplication, per-window per-channel aggregation. Estimation: exact arithmetic decomposition of the rate change into a within-channel term and a mix term, plus a corrected-denominator scenario. Post-processing: translation into orders and revenue. Decision: apply the pre-declared revert rule.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Aggregate rate comparison (the dashboard) | Both windows, single ratio | Fast, matches the number the business already watches | Cannot separate composition from behaviour; blind to definition change | Rejected as the decision basis | Never as a sole basis |
| Per-channel rate comparison and weighted decomposition | Same snapshot, four groups | Separates behaviour from mix; exposes the offending segment | Needs stable attribution; small cells are noisy | **Chosen** | Attribution rule changes |
| Difference-in-differences against an untreated channel | Requires a channel that did not receive the release | Would support a causal reading | Release went to 100% of traffic; no clean control | Not available | A future partial rollout |
| Pre/post regression with a time trend | Longer history than two windows | Handles gradual drift | Four weeks is too short to fit a stable trend; overfits | Considered, not used | Twelve or more weeks of history |
| Synthetic control from other storefronts | Multiple storefronts | Plausible counterfactual | Only one storefront exists in this scenario | Not applicable | Multi-storefront expansion |

The chosen method is deliberately arithmetic rather than statistical. The rate difference is far larger than sampling noise would explain (Section 6.6 quantifies this), so the analytical work is decomposition, not significance testing; the significance check exists to justify skipping a heavier model, not to carry the argument.

### 5d. Data provenance, quality, exclusions and leakage controls

| Item | Detail |
|---|---|
| Source ID | `SRC-SNAP-01`: frozen warehouse snapshot `snap-2026-11-03T02:00Z`, owner analytics platform |
| Access basis | Read-only analyst role; no row-level personal data in the extract (session IDs are salted hashes) |
| Extraction date | 2026-11-03, 02:00 UTC; payment callbacks up to 2026-11-02 23:59 UTC included |
| Freshness risk | Payment callbacks for the final two hours may mature after extraction; A-01 checks the final day against the expected callback lag |
| Missingness | 0.4% of sessions lack a channel value; they are reported as an explicit `unknown` cell in the audit, not silently dropped |
| Duplicates | Zero duplicate `order_id`; 6,210 duplicate raw `order_confirmed` events collapsed to 4,880 orders by `order_id` in the current window |
| Outliers | Single orders above USD 5,000 excluded from the order-value guardrail (0.01% of orders) but retained in revenue totals |
| Permissions and retention | Session hashes retained 90 days; order records 7 years for finance |
| Leakage control | The channel and session definitions are frozen per window from code-version metadata; the analysis does not read post-cutoff events |

The most dangerous join in this case is session-to-order. It looks like one-to-many and is treated as such by the raw events table, but the business question needs zero-or-one. An inner join silently deletes non-converting sessions — the entire denominator — and produces a 100% conversion rate. That failure is one of the diagnostics in Section 12.

### 5e. Reproducibility and method-selection ledger

| Artifact / source ID | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-SNAP-01` | Warehouse snapshot | 2026-11-03 | All counts, revenue, decomposition inputs | Restricted to course author | Cannot be re-extracted by a reader |
| `SRC-POL-01` | Internal metric-contract policy | v3 | Field list for a decision metric | Internal | Paraphrased, not quoted |
| `SRC-REL-01` | Release calendar entry | 2026-10-06 | Establishes the single change under suspicion | Internal | Release notes not reviewed here |
| `RUN-01` | Hand arithmetic in Section 6 | This case | Decomposition, counterfactuals, significance | Public in the case | Arithmetic is reproducible; inputs are not |
| `RUN-02` | Boundary checks in Section 6.7 | This case | Corrected-denominator and threshold behaviour | Public in the case | Scenario arithmetic only |

Every number in this case can be recomputed from the four channel rows and their totals, which are printed in Section 6.1. A reader who does not trust a total can rebuild it in four additions. That is the reproducibility standard this case claims, and it is the only one it claims.
