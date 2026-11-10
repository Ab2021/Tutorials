---
case_id: "CASE-06-01-01"
case_title: "Substitution recommendation at pick time in a dark store"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "quick commerce / inventory and assortment / exception handling"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "correctness-contract"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Inventory and fulfillment analytics lead, with store operations and CX as decision approvers"
source_index: "Case_Studies/INDEX.md"
review_trigger: "Change to substitution consent scope, allergen attribute coverage, offer latency budget, or refund/support cost basis"
---

# Substitution recommendation at pick time in a dark store

**Evidence declaration.** Every number in this case is ILLUSTRATIVE teaching arithmetic for a constructed single-store scenario. No company's substitution policy, acceptance rate, support cost, or margin is reported here. Where a claim about a mechanism could be tested, the case says what test would be required and marks the test as proposed rather than executed. Arithmetic that has been recomputed by hand is marked VERIFIED ARITHMETIC; it is verified as arithmetic, not as a fact about any market.

**Reading contract.** Sections 1–5 define the decision, the constraint set, and the data that is legitimately available at the moment a picker discovers an empty shelf. Section 6 performs the walkthrough with a frozen unit, currency, and window. Sections 7–9 convert the walkthrough into a policy, a serving design, and a validation plan. Sections 10–11 separate what the operator would do from what has actually been measured. Sections 12–14 cover failure, transfer, and the interview defence. Section 15 is the card, and nothing follows it.

## 1. The One-Line Framing

> Given **a single unavailable order line at the moment a picker scans an empty shelf location**, decide **whether to offer a substitute, which one to offer first, and what price adjustment to attach**, for **the customer who is waiting inside a live picking window**, to improve **order-level contribution and completion without violating safety or consent**, subject to **a 350 ms p99 offer-service latency budget, a hard allergen/dietary exclusion set, and consent scope captured before ranking**, compared with **the store's current first-in-stock same-category rule**.

Four things in that sentence carry the whole case. First, the unit of analysis is the order line, not the order, not the customer, and not the SKU-day: the decision happens once per empty shelf encounter and its outcome is observable within seconds. Second, the decision is composite — an offer/no-offer gate, a candidate choice, and a price adjustment — and those three sub-decisions have different cost structures. Third, the binding constraints are not "accuracy" but safety, consent, and latency, all three of which are contract properties rather than model properties. Fourth, the comparator is a named incumbent rule, so any claimed gain is a gain over an explicit process rather than over an imagined zero-substitution world.

**Task category:** ranking plus a bounded decision gate, executed at human-in-the-loop speed. **Decision frequency:** on the order of hundreds per store-day in the illustrative workload, so the decision is frequent enough for statistical evaluation within weeks and infrequent enough that a bad single decision is visible to one customer rather than to an aggregate.

**Primary success condition, plainly stated:** a customer who chose to accept a substitute receives a safe, legally sellable, reasonably equivalent item at a price that does not surprise them, and the store converts a line that would otherwise have been refunded, without increasing contacts, cancellations, or safety incidents. The full metric definition lives in Section 9 because the definition is where most substitution programmes quietly fail.

**Patterns.** Primary `evaluation-mismatch`, because the offline target that is easy to label (did the customer tap accept?) is not the outcome the business cares about (was the basket safe, completed, and worth repeating?). Secondary `human-in-the-loop`, because the picker's scan event is the decision trigger and the customer retains veto. Secondary `correctness-contract`, because allergen and consent rules are hard filters whose violation is not a model error to be traded off against margin.

**The decisive tradeoff, stated before the arithmetic.** A more aggressive offer policy raises conversion of failed lines and therefore contribution per failure, but moves risk from a refund (bounded, cheap, emotionally neutral) to a substitution (unbounded if unsafe, expensive if disliked, trust-destroying if repeated). The case therefore ranks candidates by a utility that is only meaningful *after* hard exclusions, and it refuses to let the ranking model see margin in a way that could outvote an exclusion.

## 2. Context & Background

### 2.1 The operating setting

The constructed setting is one dark store serving a dense urban catchment, operating 07:00–23:00 with a promised delivery window of 8–14 minutes depending on distance band. The store carries about 2,400 active SKUs across fresh produce, dairy, bakery, frozen, beverages, packaged staples, personal care, and a small pharmacy-adjacent health section. Orders arrive through a mobile app; the store receives a wave of orders, pickers walk a routed pick list, and any line whose shelf location is empty triggers an exception.

### 2.2 Why this decision exists at all

A dark store is not a supermarket. There is no customer physically present to choose an alternative, no shelf-talker, and no clerk who can point at the "next best" jar. The picker is a warehouse operator on a timed route. The customer is a phone screen. The only mechanism that can convert an empty shelf into a completed line is a software-mediated offer, and that offer must be constructed from catalog attributes the store already holds, inventory state that is already partially wrong (that is why the shelf is empty), and a consent scope that the customer set possibly weeks earlier.

### 2.3 The current process, described honestly

The incumbent rule is deliberately simple. When the picker marks a line short, the app looks at the same leaf category, finds SKUs with a positive sellable balance, sorts by unit price ascending, and offers the cheapest in-stock candidate, with the customer's pre-set substitution preference applied only as a coarse flag (`allow_similar`, `no_substitutes`, `call_me`). The picker may override and offer a second candidate if the customer declines, and may skip the offer entirely if the picker judges the customer unlikely to accept. There is no allergen check in the offer path beyond what the picker remembers, no quantity-equivalence check, and no price-adjustment rule: the candidate is charged at its own shelf price.

### 2.4 Why this became a decision worth a case

Three observations, all ILLUSTRATIVE, motivate the work. First, the failure rate at pick time is not negligible: 6.8% of order lines encounter an empty shelf in the baseline window. Second, acceptance under the incumbent rule is 46%, meaning more than half of the offers are rejected and the line is then refunded with an added support contact. Third, the cheapest-in-stock rule systematically offers the lowest-margin, least-equivalent candidate, which produces a predictable pattern of complaints about "smaller pack, same price" and "different brand, different taste". None of these three observations is a measured company result; they are the constructed motivation for a bounded redesign.

### 2.5 Stakeholders and what each one actually controls

| Stakeholder | Controls | Cannot control | What they need from this case |
|---|---|---|---|
| Store operations lead | Picker workflow, exception codes, whether pickers may skip offers | Model quality, catalog attribute coverage | A policy that pickers can execute inside the existing timed route |
| CX lead | Consent capture wording, support scripts, goodwill credits | Store stock, candidate quality | A hard guarantee that offers respect consent scope |
| Category managers | Assortment, pack sizes, attribute completeness | Real-time offer ranking | Visibility into which substitutions erode category trust |
| Analytics | Candidate features, calibration, evaluation design | Refund policy, support staffing | A defensible comparator and guardrail set |
| Legal/compliance (advisory) | Whether a category may be substituted at all | Operational design | A fail-closed rule list rather than a model judgement |

The table matters because a substitution programme usually fails organizationally before it fails statistically: analytics optimises acceptance, operations is measured on pick speed, CX is measured on contacts, and nobody owns the joint objective. This case assigns the joint objective to a single named owner and gives each other stakeholder a veto on exactly one contract.

### 2.6 Scope boundary

In scope: candidate construction, hard-exclusion enforcement, ranking, price adjustment, offer presentation, and the measurement design for one store. Out of scope: assortment decisions that change which SKUs are stocked (a separate case), replenishment (separate), rider dispatch after the bag is packed (separate), and any claim about a real company's substitution policy.

## 3. The Business Question

### 3.1 The decision as a question with a deadline

The question is: *for this failed line, in this store, at this second, should we offer a substitute, and if so which one at what price, given that the answer must reach the picker's device before the picker walks past the location?*

The deadline is not decorative. A picker who waits more than roughly one second at a shelf creates a queue behind them, and a queue in a 300-square-metre store with a routed pick list translates directly into missed delivery windows. The 350 ms p99 figure used here is an ILLUSTRATIVE engineering budget chosen so that the round trip plus rendering fits inside the picker's natural pause; it is not a measured requirement from any system.

### 3.2 Sub-questions the decision decomposes into

1. **Gate:** is there any candidate that passes hard exclusions and consent? If not, no offer; refund immediately rather than showing a dead end.
2. **Safety:** does the candidate violate allergen, dietary, age-restriction, or regulated-goods constraints for this customer and this line?
3. **Equivalence:** is the candidate substitutable in quantity, unit basis, and function (for example 2×250 g for 1×500 g, or a different cut of the same protein)?
4. **Value:** what price adjustment, if any, keeps the offer inside a fairness envelope the customer will accept without a support contact?
5. **Ranking:** among surviving candidates, which maximises expected contribution net of the expected cost of a rejection?
6. **Presentation:** how many candidates to show, in what order, with what copy, and with what timeout?

### 3.3 What would count as a good answer

A good answer is not "improve acceptance". A good answer is a policy that (a) never violates an exclusion, (b) never offers without consent, (c) converts more failed lines at equal or better customer treatment, (d) is cheap enough to run inside the picking loop, and (e) can be turned off in one switch if a guardrail breaks. The case evaluates candidate policies against exactly those five properties, and it treats a policy that improves (c) while damaging (a) as invalid rather than as a tradeoff.

### 3.4 The baseline and the alternatives that must be considered

| Option | Mechanism | Where it wins | Where it loses |
|---|---|---|---|
| B0: refund only | No offers at all | Zero safety risk; simplest support model | Lowest completion; highest refund volume |
| B1: incumbent cheapest-in-stock | Same leaf category, cheapest positive balance | Simple, explainable, cheap | Low equivalence, low margin, no allergen check |
| B2: rule-based equivalence ladder | Ordered attribute rules: same brand → same variant size → same category | Auditable, no model risk | Brittle when attributes are missing |
| B3: learned ranking over hard-filtered candidates | Constraint filter then calibrated utility model | Best expected contribution at scale | Needs labels, calibration, monitoring |
| B4: ask the customer to choose from a shortlist | Show 3 candidates, no ranking | Maximum agency | Adds latency to a live pick; abandonment |

The case selects B3 with B2 as the fallback and B4 as an escalation for high-value or high-risk lines. B0 and B1 remain the comparators, and B1 is the one that must be beaten on the joint objective rather than on acceptance alone.

### 3.5 Why not simply ask the customer every time

Because the customer is inside a promised delivery window and the picker is standing at a shelf. A shortlist interaction adds a human round trip whose latency is measured in tens of seconds and whose abandonment cost is a cancelled line plus a support contact. The case therefore treats "ask" as a bounded escalation for lines where the utility model's uncertainty is high or the value at stake is large, not as the default.

## 4. Assumptions & Constraints

### 4.1 Structural assumptions

- **A1 — One decision per failed line.** Each empty-shelf event produces at most one offer episode, with up to three candidate presentations inside it. Re-offers after a decline are counted inside the same episode.
- **A2 — Consent precedes ranking.** The customer's substitution preference is read before candidate generation; ranking cannot widen the consent scope.
- **A3 — Catalog attributes are the safety source of truth.** If an allergen field is missing for a candidate, the candidate is ineligible. Missing data fails closed, not open.
- **A4 — Inventory truth is approximate by construction.** The trigger for the decision is evidence that the recorded balance was wrong, so the system must not assume that a positive balance implies physical presence.
- **A5 — Price is a policy variable with a floor.** The store may absorb part of the difference; it may not charge more than the original line price without explicit customer acceptance.
- **A6 — The picker retains a documented override.** Overrides are logged with a reason code and are themselves evaluated, because an override channel is where policy silently dies.

### 4.2 Quantitative assumptions used in the walkthrough

| Assumption | Value | Unit | Basis |
|---|---|---|---|
| Store orders per day | 1,150 | orders/day | ILLUSTRATIVE constructed workload |
| Lines per order | 9.4 | lines/order | ILLUSTRATIVE |
| Empty-shelf rate at pick | 6.8% | fraction of lines | ILLUSTRATIVE |
| Baseline offer rate given a failure | 71% | fraction | ILLUSTRATIVE (picker skips) |
| Baseline acceptance given an offer | 46% | fraction | ILLUSTRATIVE |
| Average failed line value | 180 | INR | ILLUSTRATIVE |
| Average candidate price | 165 | INR | ILLUSTRATIVE |
| Average candidate COGS | 120 | INR | ILLUSTRATIVE |
| Extra support cost when declined | 12 | INR | ILLUSTRATIVE fully loaded per contact |
| Offer service p99 budget | 350 | ms | ILLUSTRATIVE engineering budget |
| Offer presentation timeout | 90 | s | ILLUSTRATIVE |

### 4.3 Constraints that cannot be traded

1. **Safety exclusion.** Allergen, dietary, age-restricted, and regulated categories are hard-excluded. No margin, acceptance, or latency consideration may override an exclusion.
2. **Consent.** A `no_substitutes` customer receives no offer; a `call_me` customer receives a contact flow, not an automated offer.
3. **No price surprise.** The final charged amount must equal the amount shown at offer time, including tax and any fee re-computation.
4. **Bounded latency.** p99 of the offer decision service must stay inside the picking-loop budget, measured server-side and end-to-end.
5. **Reversibility.** A single configuration flag must disable learned ranking and revert to B2 without a deploy.
6. **Auditability.** Every offer must be reconstructable after the fact: candidates considered, exclusions applied, scores, policy version, model version, and timestamp.

### 4.4 Constraints that are genuinely negotiable

Support staffing during peak, the size of the price-adjustment envelope, the number of candidates shown, and whether high-value lines escalate to a human. These are policy dials with cost consequences, and Section 6 prices them rather than asserting values.

### 4.5 What is explicitly unknown at decision time

Whether the customer is at home, whether they have guests, whether the missing item was the reason for the order, and whether the customer's preference flag is still accurate. The policy must not pretend to know these; it must be robust to their being wrong, which is why the decline path is designed to be cheap and non-punitive.

## 5. Data, Tools & Method

### 5a. Data Specification

**Unit of analysis:** the order line, identified by `line_id`, at the empty-shelf event identified by `exception_id`. All joins are as-of the event timestamp, not as-of the analysis date.

| Dataset | Grain | Key fields | As-of rule | Why it is needed |
|---|---|---|---|---|
| `order_line` | one row per requested line | `line_id`, `order_id`, `store_id`, `sku_id`, `qty`, `unit_price`, `promised_at`, `pick_started_at` | Immutable after order creation except status fields | Defines the original request and its price |
| `pick_exception` | one row per scanner exception | `exception_id`, `line_id`, `exception_code`, `scanned_at`, `picker_id` | Event time `scanned_at` | The decision trigger and its reason code |
| `inventory_balance` | one row per store-SKU-version | `store_id`, `sku_id`, `on_hand`, `reserved`, `version`, `as_of` | Latest version with `as_of <= scanned_at` | Candidate availability, known to be imperfect |
| `catalog_attribute` | one row per SKU-attribute | `sku_id`, `attr_name`, `attr_value`, `source`, `updated_at` | `updated_at <= scanned_at` | Allergen, dietary, size, brand, category path |
| `consent` | one row per customer-policy version | `customer_id`, `substitution_scope`, `effective_from` | Latest `effective_from <= scanned_at` | Hard gate; must not be inferred from behaviour |
| `substitution_offer` | one row per presented candidate | `offer_id`, `exception_id`, `candidate_sku`, `rank`, `score`, `policy_version`, `shown_at` | Written at presentation | Logging, replay, and audit |
| `offer_response` | one row per customer action | `offer_id`, `action`, `responded_at`, `decline_reason` | Event time | Label source and diagnosis |
| `fulfillment` | one row per terminal line state | `line_id`, `terminal_state`, `charged_amount`, `closed_at` | After maturity window | Outcome and reconciliation |
| `refund_ledger` | one row per refund | `line_id`, `amount`, `reason`, `posted_at` | After posting | Cost side of the ledger |
| `support_contact` | one row per contact | `contact_id`, `order_id`, `reason`, `handled_at`, `cost_basis` | After handling | Extra cost when an offer is declined |

**Time contract.** Three clocks exist and must never be mixed: event time (when the shelf was scanned), decision time (when the offer was computed), and processing time (when the warehouse wrote the row). The case uses event time for features, decision time for policy version selection, and processing time only for pipeline health. A feature whose event time exceeds decision time is excluded, not zero-filled.

**Quantity contract.** `qty` is in selling units, and the selling unit is not always the consumption unit. A 500 g pack and two 250 g packs are equal in mass but differ in unit count, price, and pack overhead. The case therefore carries a normalized `content_amount` and `content_unit` pair alongside `qty`, and equivalence is evaluated on the normalized pair with an explicit tolerance.

### 5b. Tools & Stack

The constructed stack is deliberately boring. Candidate generation uses an inverted index keyed by leaf category plus attribute facets, not a vector database, because the candidate universe per exception is small (tens, not millions) and the constraint filter is exact. Feature assembly is a keyed lookup over the as-of tables above. Scoring is a gradient-boosted or logistic model behind a versioned endpoint. The offer state machine is a small durable workflow with explicit states, an idempotency key, and a timeout. Evaluation uses an offline replay harness and a randomised exploration slice; both are described in Section 9. Storage is a columnar warehouse for history and a low-latency key-value store for balances. None of these choices is the point of the case; they are named so the failure analysis has something concrete to break.

### 5c. Method Summary

The method has five ordered stages, and the order is itself a design decision.

```text
stage 1  gate        consent scope and regulated-category blocklist
stage 2  generate    leaf-category candidates with positive sellable balance as-of event time
stage 3  exclude     allergen, dietary, age, brand-exclusion, pack-form, quantity-tolerance filters
stage 4  score       calibrated acceptance probability times contribution, minus rejection cost
stage 5  present     top-k with price adjustment, deterministic tie-break, 90 s timeout
```

Exclusions before scoring is not a stylistic preference. If scoring runs first and filtering second, the model's training distribution silently includes candidates that will never be shown, and more importantly a future refactor that "optimises" by trimming the filter can reintroduce unsafe candidates without changing any model metric. Filtering first makes the unsafe set structurally unreachable.

### 5d. Provenance, Quality, Exclusions and Leakage

**Provenance.** Catalog attributes come from supplier feeds and manual entry; the case assumes an attribute coverage audit is required before launch and does not assume it has been done. Inventory balances come from the store's own ledger, whose known failure mode is the very discrepancy that triggers the decision. Consent comes from the customer profile, whose known failure mode is staleness.

**Quality rules that block promotion.** A SKU with a missing allergen field is ineligible for any food-category offer. A SKU whose `content_amount` is null is ineligible wherever quantity equivalence is required. An exception without a reason code is still processed, but its outcome is excluded from model training until the code is repaired, because an unknown reason contaminates the label.

**Leakage controls.** The label is `offer_response.action == accept` recorded at `responded_at`, and every feature is restricted to `event_time <= shown_at`. The following fields are banned from the feature set because they are only known after the decision: whether the order was ultimately completed, the refund amount, the support contact, the delivery outcome, the customer's later rating, and any field derived from them. A second, stricter rule bans `decline_reason` from training entirely, since the reason is recorded after the response and would leak the label's cause into the features.

**Selection effects.** Logged offers exist only for candidates the incumbent policy chose to show. Training on that log teaches the model to imitate the incumbent's candidate distribution, not to rank a wider set. The case therefore requires an exploration slice with randomised candidate presentation before any off-policy claim is made, and it reports the effective sample size of that slice rather than the raw row count.

**Exclusions from analysis.** Lines cancelled by the customer before the pick began are outside the decision population. Lines in regulated categories (age-restricted, pharmacy-adjacent) are analysed separately and are excluded from the contribution claim because their substitution is a compliance decision, not a ranking decision.

### 5e. Reproducibility and Method-Selection Ledger

| Dimension | Choice made | Alternative rejected | Why the alternative was rejected | What would reopen the choice |
|---|---|---|---|---|
| Candidate source | Leaf-category inverted index | Embedding nearest-neighbour over full catalog | Cross-category neighbours are frequently unsafe or absurd; exactness of the exclusion filter matters more than recall | If recall audit shows >5% of acceptable candidates missing |
| Filter order | Filter then score | Score then filter | Score-first hides unsafe candidates in the training distribution | Only if a proof of equivalence is produced |
| Label | Immediate accept/decline | Eventual completion | Completion mixes delivery and other failures into a substitution label | If immediate response is too sparse for high-value lines |
| Ranking objective | Expected contribution net of rejection cost | Acceptance probability alone | Acceptance alone rewards cheap, low-equivalence items | If contribution accounting is disputed by finance |
| Price adjustment | Absorb difference up to a cap | Charge candidate shelf price | Shelf price produces "smaller pack, same price" complaints | If margin per line falls below the floor in two consecutive reviews |
| Fallback | Rule ladder B2 | No fallback | A model outage must not stop picking | If the ladder itself is shown to violate exclusions |
| Evaluation | Replay plus randomised slice | Replay only | Replay cannot rank candidates never shown | If the exploration slice is judged too costly |

**Reproducibility record.** A rerun requires: the policy version, the model artifact hash, the as-of data snapshot identifiers for the four lookup tables, the tie-break rule (`score` descending, then `price` ascending, then `sku_id` ascending), the random seed used for exploration assignment, and the exclusion-list version. Two runs with those six inputs fixed must produce byte-identical offer sequences; anything less means the pipeline has hidden state.

**Method-selection takeaway.** The chosen design is the one whose failure is cheapest to detect: a missing exclusion shows up as a hard-filter test failure, whereas a mis-ranked candidate shows up only as a slow drift in acceptance.

## 6. Analysis Walkthrough

### 6.1 Freezing the denominator before looking at anything

The population is all order lines in the baseline window 2026-10-05 through 2026-11-01 at the constructed store, restricted to lines whose pick began and which were not cancelled by the customer beforehand. The illustrative counts are:

| Quantity | Value | Unit | Derivation |
|---|---|---|---|
| Orders per day | 1,150 | orders/day | ILLUSTRATIVE store workload |
| Lines per order | 9.4 | lines/order | ILLUSTRATIVE basket |
| Lines per day | 10,810 | lines/day | 1,150 × 9.4 |
| Window length | 28 | days | fixed before analysis |
| Lines in window | 302,680 | lines | 10,810 × 28 |
| Empty-shelf rate | 6.8% | fraction of lines | ILLUSTRATIVE exception log |
| Failed lines per day | 735.08 | lines/day | 10,810 × 0.068 |
| Failed lines in window | 20,582.24 | lines | 735.08 × 28 |

The fractional counts are honest consequences of rates applied to integer bases, not measurement precision. A readout would report the integer floor and state the rate; the case keeps the fractional form because every subsequent step multiplies a rate by this base and rounding at each step would accumulate error. VERIFIED ARITHMETIC: 10,810 × 0.068 = 735.08, and 735.08 × 28 = 20,582.24.

### 6.2 The baseline ledger, step by step

The baseline has three components: offers shown, offers accepted, and cost of the decline path.

**Step 1 — offers shown.** The picker does not always offer. The illustrative rate is 71%, because pickers skip offers they judge futile, skip offers when the queue is long, and sometimes simply mark the line short without opening the substitution panel.

```text
FORMULA [SUB-01]: O = F * r_offer
WHERE: O offers/day [offers]; F failed lines/day [lines]; r_offer [fraction].
CONSTRAINT: one offer episode per failed line; re-offers inside an episode are not counted separately.
NUMERICAL EXAMPLE: F = 735.08, r_offer = 0.71 -> O = 735.08 * 0.71 = 521.9068 offers/day.
INTERPRETATION: 213.17 failed lines per day receive no offer at all under the incumbent process.
```

**Step 2 — accepted offers.** Acceptance is conditional on being offered.

```text
FORMULA [SUB-02]: A = O * r_accept
WHERE: A accepted offers/day [offers]; r_accept [fraction of shown offers].
CONSTRAINT: acceptance is an immediate customer action, not eventual order completion.
NUMERICAL EXAMPLE: O = 521.9068, r_accept = 0.46 -> A = 240.0771 accepted/day.
INTERPRETATION: 240 lines per day are converted from would-be refunds.
```

**Step 3 — declines.** Every shown-and-not-accepted offer creates a refund plus a likelihood of contact.

```text
FORMULA [SUB-03]: D = O - A = O * (1 - r_accept)
NUMERICAL EXAMPLE: D = 521.9068 - 240.0771 = 281.8297 declines/day.
CHECK: 521.9068 * 0.54 = 281.8297. VERIFIED ARITHMETIC.
```

**Step 4 — contribution of an accepted substitution.** The candidate is charged at its own price in the incumbent rule. Contribution per accepted line is candidate price minus candidate cost of goods sold.

```text
FORMULA [SUB-04]: m = p_c - c_c
WHERE: m contribution per accepted line [INR/line]; p_c candidate price [INR/line]; c_c candidate COGS [INR/line].
CONSTRAINT: excludes picking labour, packaging, delivery, and payment fees, which are unchanged by the substitution decision.
NUMERICAL EXAMPLE: p_c = 165, c_c = 120 -> m = 45 INR per accepted line.
INTERPRETATION: this is the incremental contribution of converting a failed line, relative to refunding it.
```

**Step 5 — cost of a declined offer.** A decline is not free: the line refunds (zero contribution, already the counterfactual), and a fraction of declines generate a support contact.

```text
FORMULA [SUB-05]: k = r_contact * c_contact
WHERE: k expected contact cost per decline [INR/decline]; r_contact [fraction]; c_contact [INR/contact].
NUMERICAL EXAMPLE: r_contact = 0.30, c_contact = 12 -> k = 3.6 INR/decline.
INTERPRETATION: the decline path is cheap per event, which is why aggressive offering is usually rational — until trust effects appear.
```

**Step 6 — the baseline daily net.**

```text
FORMULA [SUB-06]: N = A * m - D * k
NUMERICAL EXAMPLE: N = 240.0771 * 45 - 281.8297 * 3.6 = 10803.4695 - 1014.5871 = 9788.8824 INR/day.
```

A small rounding difference appears if intermediate values are rounded; carrying four decimals throughout gives 9,788.8839. The case reports **9,788.884 INR/day** as the baseline, and states that the readout tolerance is ±0.5 INR/day arising from rounding the offer count to four decimals. That tolerance is far below the size of any decision the case will make, so it does not affect conclusions.

### 6.3 Step A — closing the skip channel

The first intervention does not change the model at all. It removes the picker's discretion to skip the offer panel, replacing the skip with an explicit reason code that is audited. The illustrative effect is to raise the offer rate from 71% to 92%; the residual 8% is the genuinely blocked set (regulated category, consent `no_substitutes`, or no sellable candidate after exclusions).

```text
FORMULA [SUB-07]: N_A = F * r_offer_A * (r_accept * m - (1 - r_accept) * k)
NUMERICAL EXAMPLE: O_A = 735.08 * 0.92 = 676.2736
                 A_A = 676.2736 * 0.46 = 311.0859
                 D_A = 676.2736 - 311.0859 = 365.1877
                 N_A = 311.0859 * 45 - 365.1877 * 3.6 = 13998.8635 - 1314.6759 = 12684.1876 INR/day
DELTA: 12684.1876 - 9788.8839 = +2895.3037 INR/day.
```

Note what did **not** change: acceptance stayed at 46%, per-offer margin stayed at 45 INR, and per-decline cost stayed at 3.6 INR. The entire gain comes from showing 154.37 more offers per day at the same conditional acceptance. This is the cheapest intervention in the case and it is a process change, not a model.

**Why the skip channel was worth 2,895 INR/day.** The skipped lines carry the same conditional acceptance as the offered ones under this assumption. Their expected daily value is:

```text
SKIPPED = 735.08 * 0.29 * (0.46 * 45 - 0.54 * 3.6) = 735.08 * 0.29 * 18.756 = 3998.28 INR/day
```

That is *larger* than the observed gain, because closing the channel costs something the formula ignores: picker time. Each additional offer adds roughly 12 seconds of picker interaction, and 154 extra offers per day across a shift is about 31 picker-minutes. At a fully loaded picker cost of 0.55 INR/minute that is 17 INR/day — negligible. The gap between 3,998 and 2,895 is therefore not explained by labour; it is explained by the fact that the 8% residual block is not randomly distributed. The blocked set is enriched in exactly the lines where an offer would have been accepted (regulated and consent-blocked lines are shown as offers in the 71% baseline count only when a picker overrode). The case does not need to resolve this precisely; it needs to note that "closing the skip channel" gains less than a naive multiplication suggests, and the readout must therefore measure the closed-channel acceptance directly rather than assuming it.

### 6.4 Step B — replacing cheapest-in-stock with an equivalence-ranked candidate

The second intervention changes which candidate is presented first. Under the incumbent rule the model's implicit objective is "cheapest positive balance", which selects low-margin, low-equivalence items. Under the proposed rule the first candidate is chosen to maximise a calibrated utility over hard-filtered candidates.

The illustrative effect is conditional acceptance rising from 46% to 55%, at an unchanged candidate price and cost. That is a strong claim and the case is explicit that it is an assumption to be tested, not a result. It is plausible for a mechanical reason: the incumbent systematically offers the cheapest SKU in the leaf category, which is frequently a smaller pack or a lower-tier brand, and customers reject those at a higher rate than they reject a same-brand same-size candidate. The mechanism is testable with an A/B on the candidate-selection rule alone, holding the offer rate fixed.

```text
FORMULA [SUB-08]: N_B = O_A * (r_accept_B * m - (1 - r_accept_B) * k)
NUMERICAL EXAMPLE: A_B = 676.2736 * 0.55 = 371.9505
                 D_B = 676.2736 - 371.9505 = 304.3231
                 N_B = 371.9505 * 45 - 304.3231 * 3.6 = 16737.7716 - 1095.5632 = 15642.2084 INR/day
DELTA vs A: 15642.2084 - 12684.1876 = +2958.0208 INR/day.
```

**The candidate-mix arithmetic behind the acceptance claim.** To make the claim concrete rather than mystical, consider the illustrative candidate distribution under each rule for a 400 ml vs 500 ml beverage line.

| Rule | First candidate | Content | Price | Unit price | Margin | Illustrative accept rate |
|---|---|---|---|---|---|---|
| Incumbent cheapest-in-stock | 400 ml competitor brand | 400 ml | 165 INR | 41.25 INR/100 ml | 45 INR | 0.46 |
| Equivalence-ranked | 500 ml same brand variant | 500 ml | 180 INR | 36.00 INR/100 ml | 45 INR | 0.55 |

The unit-price ratio under the incumbent is 41.25/36.00 = 1.1458, meaning the customer pays 14.58% more per 100 ml for a smaller pack. Under the equivalence rule the unit price is identical to the original at 36.00 INR/100 ml. The acceptance difference is therefore not a black-box model output; it is a mechanically explicable consequence of the incumbent rule selecting on absolute price while the customer evaluates on unit price and brand continuity. VERIFIED ARITHMETIC: 165/400 = 0.4125 INR/ml = 41.25 INR/100 ml; 180/500 = 0.36 INR/ml = 36.00 INR/100 ml; 41.25/36.00 = 1.14583.

This is the central `evaluation-mismatch` of the case: the incumbent rule optimises a proxy (lowest absolute price) that is negatively correlated with the outcome the customer cares about (unit-price parity and brand continuity). Any offline evaluation that scores candidates by price-rank will *prefer* the incumbent's choice and will report the new rule as worse. That is why Section 9 rejects price-rank as an evaluation proxy.

### 6.5 Step C — the fairness cap and why it costs money

The equivalence rule creates a new problem. The same-brand 500 ml candidate costs 180 INR against an original line price of 180 INR — parity. But many equivalence-ranked candidates cost more than the original. The case adopts a cap: the customer is never charged more than 105% of the original line price, and the store absorbs the difference.

The illustrative exposure: 22% of accepted lines end up with a capped candidate whose effective contribution falls from 45 INR to 31.20 INR. The blended margin over accepted lines becomes 0.78 × 45 + 0.22 × 31.20 = 35.10 + 6.864 = 41.964 INR.

```text
FORMULA [SUB-09]: m_mix = (1 - f_cap) * m + f_cap * (p_cap - c_c)
WHERE: f_cap share of accepted lines hitting the cap [fraction];
       p_cap capped charge [INR]; c_c candidate COGS [INR].
NUMERICAL EXAMPLE: 400 ml equivalent charged at cap = 400 * 1.05 * 0.36 = 151.20 INR
                 margin at cap = 151.20 - 120 = 31.20 INR
                 m_mix = 0.78*45 + 0.22*31.20 = 41.964 INR.
```

```text
FORMULA [SUB-10]: N_C = O_A * (r_accept_B * m_mix - (1 - r_accept_B) * k)
NUMERICAL EXAMPLE: N_C = 676.2736 * (0.55 * 41.964 - 0.45 * 3.6)
                     = 676.2736 * (23.0802 - 1.62)
                     = 676.2736 * 21.4602 = 14512.9667 INR/day
DELTA vs B: 14512.9667 - 15642.2084 = -1129.2417 INR/day.
TOTAL vs baseline: 14512.9667 - 9788.8839 = +4724.0828 INR/day.
```

The cap **costs** 1,129.24 INR/day and the case keeps it anyway. The reason is not captured by this ledger: a capped candidate that preserves unit-price parity is a different customer experience from a candidate charged at its own higher shelf price, and the model's acceptance estimate of 55% was measured (illustratively) under a presentation where price parity was visible. An uncapped policy would present a 14.58% unit-price premium on the very candidate the ranking model prefers, which is precisely the pattern that generates "smaller pack, same price" complaints. The case treats the cap as a constraint whose cost is disclosed, not as a decision variable to be optimised away. This is the second `evaluation-mismatch`: the ledger prefers the uncapped policy, and the ledger is not the whole objective.

### 6.6 Sizing the offer service

The offer service is small. At the illustrative failure rate, offers per hour are not uniform; the case uses a peak-hour share of 9% of daily failures.

```text
FORMULA [SUB-11]: q_peak = F * s_peak / 3600
WHERE: q_peak peak request rate [requests/s]; s_peak share of daily failures in peak hour [fraction].
NUMERICAL EXAMPLE: q_peak = 735.08 * 0.09 / 3600 = 66.16 / 3600 = 0.0184 requests/s.
```

Eighteen milliseconds between requests on average in the peak hour is trivially small for any service, and it is the reason the latency budget is not the hard part of this case. The hard part is the *p99 under fan-out*: each offer may consult four as-of tables, an attribute index, a model endpoint, and a price service. At 0.0184 requests/s the concurrency is effectively one, so p99 is dominated by the slowest dependency, not by queueing. Using the fluid-queue bound from the course standards, backlog at a 350 ms budget is `(λ - μ) * t` with λ = 0.0184 and μ = 1/0.350 = 2.857 requests/s, which is negative, so the queue never builds under this model. The model's real value here is to show that the service is not capacity-bound and that any p99 breach will be a dependency-tail event rather than a saturation event.

### 6.7 Sensitivity: what actually moves the decision

The decision to proceed depends on acceptance, offer rate, and cap share. The grid below holds everything else at the Section 6.4–6.5 values and reports total delta INR/day against the 9,788.8839 baseline.

| Acceptance | Net INR/day | Delta INR/day | Reading |
|---|---|---|---|
| 0.40 | 9,890.91 | +102.02 | Barely better than baseline; cap and decline costs consume the gain |
| 0.46 | 11,739.73 | +1,950.85 | Equivalent to incumbent acceptance with better candidate distribution |
| 0.50 | 12,972.28 | +3,183.40 | Modest ranking benefit is already worth the build |
| 0.55 | 14,512.97 | +4,724.08 | Case's design assumption |
| 0.60 | 16,053.65 | +6,264.77 | Unlikely without a packaging change |
| 0.65 | 17,594.34 | +7,805.46 | Outside plausible range for same-category substitution |

The break-even acceptance, holding the cap and decline costs, solves `A* * m_mix - (O - A*) * k = N_base` for `A*`:

```text
FORMULA [SUB-12]: A* = (N_base + O * k) / (O * (m_mix + k))
NUMERICAL EXAMPLE: A* = (9788.8839 + 676.2736*3.6) / (676.2736*(41.964+3.6))
                     = (9788.8839 + 2434.5850) / (676.2736 * 45.564)
                     = 12223.4689 / 30813.77 = 0.39669
```

So acceptance must exceed **39.7%** for the project to beat the baseline on this ledger. The incumbent achieves 46% conditional acceptance while offering only 71% of failures. That comparison is the real insight: the incumbent's acceptance is not bad, its *reach* is bad. Step A captures the reach; Step B adds the ranking benefit; Step C discloses its cost.

**Offer-rate sensitivity at acceptance 0.55:**

| Offer rate | Delta INR/day | Reading |
|---|---|---|
| 0.60 | −323.91 | Below the reach needed to recover fixed monitoring cost |
| 0.71 | +1,411.34 | Incumbent reach, new ranking only |
| 0.80 | +2,831.09 | Partial channel closure |
| 0.92 | +4,724.08 | Case's design point |
| 1.00 | +5,986.08 | Only reachable if consent-blocked lines are also offered, which is illegal under the consent contract |

The 1.00 row is included deliberately as a counterexample: the arithmetic says it is best and the constraint says it is unreachable. A sensitivity table that does not show the unreachable optimum invites someone to pursue it.

**Cap-share sensitivity at acceptance 0.55:**

| Cap share of accepted lines | Blended margin | Delta INR/day |
|---|---|---|
| 0.00 | 45.000 | +5,853.32 |
| 0.11 | 43.482 | +5,288.70 |
| 0.22 | 41.964 | +4,724.08 |
| 0.40 | 39.480 | +3,800.16 |
| 0.60 | 36.720 | +2,773.57 |

Even at 60% of accepted lines hitting the cap, the policy beats baseline by 2,773.57 INR/day. The cap is therefore affordable across a wide range, which is the quantitative justification for treating it as a constraint rather than an optimisation target. The boundary is a cap share where the blended margin falls below the break-even implied by the decline cost; solving `m_mix * 0.55 = 0.45 * 3.6` gives `m_mix = 2.945`, far below any plausible cap-affected margin, so the cap would have to be near-total and the COGS near the charge for the policy to fail. The real risk is not arithmetic; it is that a near-total cap share signals an assortment problem (the store does not stock equivalents), which is the separate assortment case.

### 6.8 Independent re-derivation of the headline

To guard against a spreadsheet error in the chain, the headline is re-derived from per-failure unit economics rather than from window aggregates:

- Per failed line, baseline: 0.71 × (0.46 × 45 − 0.54 × 3.6) = 0.71 × 18.756 = 13.31676 INR.
- Per failed line, proposed: 0.92 × (0.55 × 41.964 − 0.45 × 3.6) = 0.92 × 21.4602 = 19.74338 INR.
- Delta per failed line: 6.42662 INR.
- Multiply by 735.08 failures/day: 6.42662 × 735.08 = 4724.08 INR/day.
- Multiply by 365: 1,724,290 INR/year (rounded to the nearest rupee).

Both routes agree to 0.01 INR/day. The annual figure is presented with the caveat that a store does not run at a constant rate for 365 days; the annual number is a scaling device, not a forecast, and the case does not use it for payback claims.

## 7. The Finding

### 7.1 The finding in one paragraph

Under the stated assumptions the dominant lever is **reach, not ranking**: closing the picker's skip channel is worth 2,895 INR/day and improving candidate ranking is worth a further 2,958 INR/day, but the second gain depends on an unproven acceptance lift and the first does not. The fairness cap then removes 1,129 INR/day of the gain and is retained because it aligns the presentation with the unit-price parity the customer actually evaluates. The policy beats the incumbent under a wide band of assumptions, with a break-even acceptance of 39.7%, well below the incumbent's conditional 46%.

### 7.2 What the finding does and does not establish

Establishes: the arithmetic of the ledger, the ordering of levers by size, the break-even acceptance, the insensitivity of the conclusion to the cap share, and the fact that the latency budget is not binding at this store's volume.

Does not establish: that acceptance would rise to 55%; that the cap reduces complaints; that the store's attribute coverage is sufficient for allergen exclusion; that any customer's trust changes; that the policy is safe in regulated categories. Each of those is a separate test, listed in Section 9 with its acceptance criterion.

### 7.3 The finding's most fragile number

The 55% acceptance is the load-bearing assumption and the weakest. It is worth stating why explicitly: the baseline 46% is a conditional acceptance on a *specific candidate distribution*, and the proposed 55% is a counterfactual on a different distribution that has never been served. Comparing them as if they were two measurements of the same quantity is a category error. The honest formulation is "if presenting equivalence-ranked candidates raises conditional acceptance from 46% to at least 39.7%, the policy pays for itself on this ledger" — and 39.7% is below the incumbent rate, so the risk is genuinely bounded. The case can therefore proceed to a pilot even if the 55% figure is wrong, provided the pilot has a pre-registered stopping rule.

### 7.4 Finding: the constraint that binds is consent, not latency

The offer service at 0.0184 requests/s is not a systems problem. The binding constraint is that 8% of failures cannot be offered at all, and a further share of *offerable* lines are blocked by missing allergen attributes. The attribute-coverage audit is therefore a prerequisite deliverable, and the case records it as OPEN rather than assuming it passes. This is the kind of finding that changes a roadmap: the engineering team expects to be asked for a low-latency service and is instead asked for a catalog data-quality programme.

## 8. System/Solution Design

### 8.1 Input-to-output architecture

```text
                 +-----------------------------+
 picker scan --->| exception intake            |
 (line_id,       | - idempotency key           |
  exception)     | - event time stamped        |
                 +-------------+---------------+
                               |
                    +----------v-----------+
                    | consent gate         |<--- consent as-of event time
                    | scope + blocklist    |
                    +----------+-----------+
                               | allow
                    +----------v-----------+
                    | candidate generator  |<--- inventory balance as-of event time
                    | leaf category facets |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    | hard exclusions      |<--- attribute store (fail closed)
                    | allergen/diet/age    |
                    | pack form / quantity |
                    +----------+-----------+
                               | survivors (possibly empty)
                    +----------v-----------+
                    | utility scorer       |<--- model artifact (versioned)
                    | P(accept) x margin   |
                    | - rejection cost     |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    | price adjuster       |<--- cap policy, tax recompute
                    | cap at 105%          |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    | offer presenter      |---> picker device (p99 < 350 ms)
                    | top-k, 90 s timeout  |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    | durable episode log  |---> warehouse (audit + labels)
                    +----------------------+
```

Every stage has one owner and one failure behaviour. The consent gate and the exclusion filter fail closed: an error there produces "no offer" and an immediate refund, never a permissive offer. The generator fails open to an empty set, which the presenter renders as the no-candidate path. The scorer fails open to the rule ladder B2 with a flag, so a model outage degrades ranking but not safety. The price adjuster fails closed to the original price, never to an unbounded charge.

### 8.2 The episode state machine

```text
   CREATED --(gate block)--> BLOCKED_NO_OFFER --> REFUNDED
      |
      +--(no survivors)--> NO_CANDIDATE --> REFUNDED
      |
      +--(survivors)--> OFFERED --(accept)--> ACCEPTED --> FULFILLED
                          |  |
                          |  +--(decline)--> DECLINED --> REFUNDED
                          |
                          +--(timeout 90 s)--> EXPIRED --> REFUNDED
                          |
                          +--(picker override w/ reason)--> OVERRIDDEN --> FULFILLED|REFUNDED
```

Required invariants: exactly one terminal state per episode; the terminal state is written with the same idempotency key as creation; `ACCEPTED` implies a price that equals the price shown; `EXPIRED` never converts to `ACCEPTED` on a late customer tap. The late-tap rule is a real design decision: a customer who taps accept at 95 seconds must be handled by a support flow, not by fulfilling a line the picker already abandoned, because the bag may already be sealed.

### 8.3 Latency design inside a 350 ms budget

| Stage | Budget | Why that number | Failure behaviour |
|---|---|---|---|
| Intake and idempotency | 10 ms | in-process plus one write | Retry with same key |
| Consent lookup | 25 ms | single key-value read | Fail closed to `no_substitutes` |
| Candidate generation | 60 ms | inverted-index lookup over tens of entries | Empty set |
| Hard exclusions | 55 ms | batch attribute fetch, one round trip | Fail closed to empty set |
| Scoring | 120 ms | model endpoint, batch of ≤20 | Fall back to rule ladder |
| Price adjustment | 30 ms | local cap arithmetic plus tax table | Charge original price |
| Present and respond | 50 ms | network plus render margin | Show retry affordance |

The sum is 350 ms exactly, which is intentional and slightly dishonest: budgets that sum exactly to the target leave no headroom. In practice the target would be set at 350 ms with a stage sum of 280 ms, leaving 70 ms of slack, and the case records that the original sum-to-target allocation is a design smell to correct in implementation. This detail is included because "the budget adds up" is a common and false comfort.

### 8.4 Offline and online paths

**Offline path.** Nightly computation of candidate-distribution features, attribute-coverage reports, exclusion-list diffs, and the training set with strict as-of joins. The offline path may be late; it may not be wrong about event times.

**Online path.** Synchronous offer decision inside the picking loop, with a 90-second customer response window and a durable episode log. The online path may be degraded; it may not be permissive.

**Read path for support.** An agent must be able to answer "why did the customer see this candidate?" from the episode log alone, without replaying the model. That requires the log to record the top-20 candidate list with scores and the exclusions that removed anything that would have ranked above the shown set. Without the removed candidates, a support agent cannot distinguish "the model chose badly" from "the exclusion list blocked the obvious match".

### 8.5 Operational readiness

Before launch the case requires: an attribute-coverage audit with a named owner; a hard-filter test suite covering allergen, dietary, age, pack-form, and quantity tolerance; a shadow mode that computes offers without presenting them, to measure candidate availability and latency; a one-flag rollback to B2; a support script for the late-tap and forced-substitution paths; and a pre-registered stopping rule for the pilot. Each is a deliverable with a checkbox, not a sentiment.

### 8.6 Why this architecture rather than a retrieval-first design

A retrieval-first design (embed the order line, nearest-neighbour over candidates, then filter) is attractive because it handles messy catalog text. It is rejected here for one reason: nearest-neighbour recall is not auditable in the way an exclusion filter must be. With filter-first, the question "could this policy ever show an allergen-bearing candidate to an allergic customer?" is answered by reading one function. With retrieval-first, it is answered by an argument about embedding geometry, which is exactly the kind of argument that fails in review. The case accepts lower recall and worse handling of free-text catalog noise in exchange for a structurally checkable safety property.

## 9. Evaluation & Validation

### 9.1 Metric definitions with numerator, denominator, population, window, comparator

Each metric below is written as a complete definition because a bare name is not a metric.

| Metric | Numerator | Denominator | Population | Window | Comparator | Guardrail? |
|---|---|---|---|---|---|---|
| Conditional acceptance | Offers accepted | Offers shown | Lines with a shown offer in the store | 14-day pilot | Incumbent rule in matched time-of-day cells | No |
| Offer reach | Offers shown | Failed lines | Failed lines after exclusions | 14-day pilot | 0.71 baseline | No |
| Line completion | Lines fulfilled by original or accepted substitute | Eligible failed lines | Failed lines in window | 14-day | Baseline completion | No |
| Safety-exclusion violations | Offers of an excluded candidate | Offers shown | All offers | Continuous | Zero tolerance | Yes |
| Consent violations | Offers to `no_substitutes` customers | Offers shown | All offers | Continuous | Zero tolerance | Yes |
| Contact rate | Support contacts attributed to substitution | Accepted + declined offers | Failed lines with an offer | 7-day lagged | Baseline contact rate | Yes |
| Contribution per failed line | Accepted margin − decline cost | Failed lines | Failed lines in window | 14-day | 13.31676 INR | No |
| p99 offer latency | 99th percentile server latency | Offer requests | All requests | Continuous | 350 ms | Yes |
| Price-consistency defects | Offers charged ≠ price shown | Accepted offers | Accepted offers | Continuous | Zero tolerance | Yes |

**Why contact rate is lagged seven days.** Contacts arrive after delivery, sometimes after the next order. A same-day readout systematically under-counts substitution-related contacts and would let a bad policy look clean during the pilot. The case uses a seven-day lagged window and states that the pilot cannot be read on day 3.

**Why there are four zero-tolerance guardrails.** Because a policy that reduces contribution is a bad policy, but a policy that serves an allergen-bearing substitute to an allergic customer is a different category of event that no contribution gain justifies. Guardrails are evaluated first, and a guardrail breach stops the pilot regardless of the primary metric.

### 9.2 The validation ladder

| Rung | What it checks | How | Pass criterion | If it fails |
|---|---|---|---|---|
| 1. Unit fixtures | Filter logic, price cap, quantity tolerance | Hand-built cases including 2×250 g vs 1×500 g | All fixtures exact | Fix before anything else |
| 2. Replay on logged offers | Ranking quality on the incumbent's candidate distribution | Score logged candidates; compare against observed responses | Better log-loss than the incumbent rule's implicit ordering | Do not proceed to exposure |
| 3. Shadow mode | Candidate availability, latency, exclusion counts | Compute offers without presenting for 7 days | p99 inside budget; exclusion counts match a manual audit | Repair data or latency |
| 4. Randomised candidate slice | Unbiased ranking estimate | Randomise first-candidate among surviving set for 10% of episodes | Off-policy estimate with reported effective sample size | Extend slice |
| 5. Store pilot | End-to-end effect | Interleaved store-week design or matched control | Primary metric up; guardrails clean; contact rate not worse | Roll back |

Rung 2 is where most substitution programmes stop, and it is exactly where the case says the evidence is weakest. Replay on logged offers can only re-rank candidates the incumbent already showed; it cannot discover that a better candidate exists but was never presented. The case therefore requires rung 4 before any claim that the ranking model "improves acceptance", and it reports the fraction of episodes in which the randomised candidate differed from the model's top choice — if that fraction is small, the slice carries little information.

### 9.3 Estimation and its limits

The pilot estimate is a difference in means between pilot and control cells. With 735.08 failures/day and a 14-day window, the failure population is about 10,291 lines per arm under a 50/50 split. The primary metric's standard deviation is not known at design time; the case uses a planning assumption of a per-failure contribution standard deviation of 55 INR, derived from the spread between an accepted line (≈42 INR contribution) and a declined line (≈−3.6 INR), which spans roughly 45 INR. Using that:

```text
FORMULA [SUB-13]: n_per_arm = 2 * (z_{1-alpha/2} + z_{1-beta})^2 * sigma^2 / delta^2
WHERE: n [lines/arm]; sigma [INR/line]; delta minimum detectable effect [INR/line].
NUMERICAL EXAMPLE: alpha=0.05 -> z=1.96; power 0.80 -> z=0.8416; sigma=55; delta=6.4266
  n = 2 * (2.8016)^2 * 3025 / 41.3016
    = 2 * 7.8490 * 3025 / 41.3016
    = 47486.5 / 41.3016 = 1149.7 lines per arm.
```

At 5,145.5 failures per arm in a 14-day 50/50 split, the design is roughly 4.5× over-powered for the case's headline effect. That is comfortable, and it means the pilot can afford a 90/10 treatment allocation instead of 50/50, or a shorter window, or a smaller store. It also means the *statistical* question is easy and the *operational* question — does the policy survive a real peak, a real attribute gap, a real angry customer — is the hard one.

**What this calculation does not establish.** It assumes independent observations, which failed lines are not: lines within an order share a customer and a picker, and orders within a time cell share store congestion. The true variance is larger than the line-level estimate by an intra-class factor the case does not know. The case therefore treats the computed sample size as a lower bound and recommends clustering the standard error by store-day at analysis time. Stating this is more useful than reporting a precise n and implying precision the design does not have.

### 9.4 Identification: why this is not a clean experiment

A store-level pilot cannot randomise the *store*, because the treatment changes picker workflow and the control pickers would see it. Randomising at the line level within a store is possible but creates contamination: a picker who has learned to trust equivalence-ranked offers behaves differently on control lines. The case therefore proposes an interleaved store-week design (alternate weeks) with a matched control store, and it names the confounders that this design cannot remove: weather, local promotion calendars, and competitor activity. A difference-in-differences estimate with store and week fixed effects is the strongest available design under the operational constraint, and it is weaker than a clean randomised trial. The case says so rather than calling the pilot "causal".

### 9.5 Validation of the exclusion filter itself

The filter is validated by construction, not by accuracy. The test suite enumerates the attribute space for each restricted category and asserts that no candidate lacking a required attribute can be returned. A useful negative test: inject a SKU with a null allergen field into the candidate set and assert the offer service returns an empty set rather than a permissive offer. A second negative test injects a candidate whose `content_unit` differs from the original (grams vs millilitres) and asserts the quantity tolerance rejects it rather than treating 500 as 500. These two tests catch the majority of real-world exclusion defects, and neither requires a model.

### 9.6 Executed versus proposed

| Check | Status |
|---|---|
| Ledger arithmetic (Sections 6.2–6.8) | EXECUTED in this document; recomputed by two independent routes |
| Sensitivity grids | EXECUTED arithmetic over stated assumptions |
| Sample-size calculation | EXECUTED arithmetic; variance assumption unvalidated |
| Unit fixtures for exclusions | PROPOSED — not executed |
| Shadow mode | PROPOSED — not executed |
| Randomised candidate slice | PROPOSED — not executed |
| Store pilot | PROPOSED — not executed |
| Attribute-coverage audit | OPEN — data not available to this document |

Nothing in the EXECUTED rows is evidence about a real store. They are evidence that the stated assumptions imply the stated conclusion, which is what a teaching case can honestly provide.

## 10. Action Taken

### 10.1 The action, stated as a decision

**Decision:** proceed to a two-store pilot of the filter-then-score policy with the 105% price cap and a mandatory reason code replacing the picker skip, with a hard rollback to the rule ladder on any guardrail breach.

**Owner:** inventory and fulfillment analytics lead, with store operations executing the workflow change and CX approving the offer copy.

**Stage:** proposed. No part of this action has been executed in any real store, and the case does not claim otherwise.

### 10.2 Why this action and not the alternatives

| Alternative action | Why rejected as the *first* action |
|---|---|
| Build the full learned ranker with embeddings first | The reach gain is larger, cheaper, and independent of model quality; sequencing the model first postpones the certain gain behind the uncertain one |
| Launch with no price cap | The cap's cost is affordable across a wide band, and uncapped presentation reproduces the exact complaint pattern that motivated the case |
| Pilot the customer-shortlist interaction instead | Adds live-picking latency and abandonment; reserved as an escalation for high-value or high-uncertainty lines |
| Wait for complete attribute coverage before any change | The reach intervention does not depend on attributes for non-food categories; waiting forfeits a certain gain for an uncertain prerequisite |

### 10.3 The sequenced plan

1. **Weeks 1–2 — coverage audit and fixtures.** Enumerate allergen, dietary, and age attributes per restricted category; build the two negative tests; publish a coverage percentage by category. Gate: no restricted category may enter the pilot below full coverage.
2. **Weeks 3–4 — shadow mode.** Compute offers without presenting for seven days; compare exclusion counts to a manual audit of 50 sampled episodes; measure latency distribution.
3. **Weeks 5–6 — reason-code rollout.** Replace the skip with a reason code in both pilot and control stores so the control measurement is not contaminated by workflow differences.
4. **Weeks 7–8 — randomised slice.** Present a uniformly random surviving candidate as the first option in 10% of episodes, recording which episodes differed from the model's top choice.
5. **Weeks 9–10 — pilot.** Full policy in two stores, matched control, interleaved weeks, seven-day lagged contact readout.
6. **Week 11 — decision review.** Continue, adjust, or roll back based on the pre-registered rule.

### 10.4 The pre-registered stopping rule

The pilot stops and rolls back if any of the following occurs: any safety-exclusion violation; any consent violation; any price-consistency defect; p99 latency above 350 ms for two consecutive hours; or a lagged contact rate more than 2 percentage points worse than control with a lower confidence bound above zero. The pilot continues but does not expand if the primary metric improves while the contact rate worsens by 1–2 points, because that pattern suggests the contribution gain is being financed by customer friction.

### 10.5 What would change the decision

- If the coverage audit finds a restricted category below 90% attribute completeness, the case excludes that category from the pilot rather than delaying everything.
- If the randomised slice shows the model's top choice almost never differs from the incumbent's, the ranking investment is deferred and only the reach intervention proceeds.
- If the cap share exceeds 40% of accepted lines, the case treats it as an assortment signal and routes the finding to the local-assortment case rather than widening the cap.

## 11. Outcome & Results

### 11.1 What has actually been established

Only the arithmetic. The documented results of this case are:

| Claim | Evidence type | Value | Scope |
|---|---|---|---|
| Ledger delta of the full policy over the incumbent | illustrative, estimated | +4,724.08 INR/day | Constructed store at constructed rates |
| Delta attributable to reach alone | illustrative, estimated | +2,895.30 INR/day | Same |
| Delta attributable to ranking alone | illustrative, estimated | +2,958.02 INR/day | Same, conditional on the 55% assumption |
| Cost of the price cap | illustrative, estimated | −1,129.24 INR/day | Same |
| Break-even conditional acceptance | derived | 0.39669 | Same |
| Pilot sample size per arm | derived | 1,150 lines | Under stated variance assumption |
| Realised store effect | not-measured | none | No pilot has run |

### 11.2 The honest reading of the numbers

The composition is the result: reach contributes 61.3% of the gross gain (2,895.30 / 4,724.08 net of the cap's drag is not the right denominator; against the pre-cap gross of 5,853.32, reach is 49.5%). Presenting the 4,724.08 figure alone would let a reader believe a model produced it. The case states the decomposition explicitly because the decomposition is what transfers: in almost every substitution programme the reach gap is larger than the ranking gap, and it is usually fixable without machine learning.

### 11.3 Counterfactual honesty

The baseline is a real process with a real acceptance rate. The proposed policy is a counterfactual. Comparing a counterfactual's modelled contribution against a baseline's observed contribution is legitimate for a teaching ledger, provided the reader understands that the counterfactual's acceptance (55%) and its cap share (22%) are both assumptions. The case's sensitivity tables exist precisely so that a reader who distrusts 55% can see the decision under 46% (still +1,950.85) and under 40% (+102.02, essentially nothing).

### 11.4 What a reader should carry forward from the outcome section

That a substitution programme's economics are dominated by two numbers that are not model metrics at all: the share of failures that receive an offer, and the share of offers whose price the customer finds fair. The model sits between them and can at best move the middle term.

## 12. Failure Modes & What Went Wrong

The failures below are the ones this specific design invites. Each names the mechanism, the discriminating check, the bounded fix, and how the fix is verified. None of them is a claim that the failure occurred; they are design-time foresight with executable detection.

### 12.1 F-01 — allergen attribute present but stale

**Mechanism.** A supplier reformulates a product and the feed updates the ingredient list but not the structured allergen field. The attribute exists, so the fail-closed rule does not trigger, and an excluded candidate becomes eligible.

**Discriminating check.** Compare the structured allergen field's `updated_at` against the free-text ingredient list's `updated_at` for the same SKU; a gap beyond the feed's expected latency indicates a partially applied update. Second check: sample 30 SKUs per category monthly and compare structured fields to package images.

**Bounded fix.** Treat the structured field as untrusted when the free-text field is newer; route affected SKUs to manual review rather than auto-eligibility.

**Verification.** Inject a synthetic SKU with a newer free-text field and assert it is excluded; then confirm the manual-review queue received it.

### 12.2 F-02 — consent read at the wrong time

**Mechanism.** The consent service is read at presentation time rather than at event time, so a customer who revoked substitution consent during the pick (after the shelf was scanned) still receives an offer.

**Discriminating check.** Compare the consent version used in the offer log against the consent version effective at the scan timestamp. Any offer whose logged version differs from the event-time version is a defect.

**Bounded fix.** Pin the consent version at intake and carry it through the episode; re-check before fulfilment and treat a mid-episode revocation as a decline.

**Verification.** Fixture: create an episode, revoke consent, advance; assert the episode terminates as declined and no fulfilment occurs.

### 12.3 F-03 — quantity equivalence across incompatible units

**Mechanism.** The normalizer converts pack sizes to a common numeric field but drops the unit, so 500 g and 500 ml compare equal. A 500 ml liquid becomes an "equivalent" for a 500 g solid.

**Discriminating check.** Scan for candidates whose `content_unit` differs from the original line while the numeric amounts match; these are the exact rows the defect produces.

**Bounded fix.** Make the unit part of the comparison key and require an explicit conversion table for legitimate cross-unit substitutions; default to ineligible when no table entry exists.

**Verification.** Fixture with a grams-vs-millilitres pair asserts ineligibility; a legitimate conversion (1 l = 1000 ml) asserts eligibility.

### 12.4 F-04 — price cap applied before tax

**Mechanism.** The cap is computed on the pre-tax amount and tax is added afterwards, so the customer is charged more than the displayed cap and the "no price surprise" constraint is violated even though the cap logic looks correct in isolation.

**Discriminating check.** Recompute every accepted offer's charged amount from the displayed amount plus the tax table; flag any mismatch.

**Bounded fix.** Compute the cap on the tax-inclusive amount and derive the pre-tax figure for accounting; store both in the offer log.

**Verification.** Fixture with two tax rates asserts the charged amount equals the displayed amount exactly for both.

### 12.5 F-05 — idempotency key reuse across episodes

**Mechanism.** The picker's scanner retries on a flaky connection and the retry is deduplicated by `line_id` rather than by `exception_id`, so two genuine empty-shelf encounters on the same line (a second unit also missing) collapse into one episode and the second unit is silently unfulfilled.

**Discriminating check.** Compare the count of raw scanner events to the count of episodes per line; a line with multiple units should be able to produce multiple episodes.

**Bounded fix.** Key idempotency on `(line_id, unit_index, scan_sequence)` with an explicit duplicate window of two seconds.

**Verification.** Fixture sends two scans 0.5 s apart and two scans 5 s apart; asserts one episode and two episodes respectively.

### 12.6 F-06 — model scoring on candidates the filter would remove

**Mechanism.** Feature computation runs the scorer on the full candidate set before exclusions, and the training pipeline logs those scores. The model learns from candidates that are never shown, and its calibration degrades in a way no serving metric reveals.

**Discriminating check.** Compare the candidate set entering the scorer with the set exiting the filter; if the scorer sees a superset, the training distribution is contaminated.

**Bounded fix.** Enforce the pipeline order in code with a type that only accepts filtered candidates, so the scorer cannot be handed an unfiltered list.

**Verification.** A compile-time or unit-level assertion that the scorer's input type excludes unfiltered candidates; a test that attempts to pass one and expects a failure.

### 12.7 F-07 — late customer tap after the picker moved on

**Mechanism.** The customer accepts at 100 seconds; the episode already expired at 90; a naive implementation honours the late tap and marks the line fulfilled although the bag is sealed and the rider has left.

**Discriminating check.** Count episodes where `responded_at > shown_at + 90 s` and the terminal state is `ACCEPTED`. Any positive count is the defect.

**Bounded fix.** Make expiry terminal; route late taps to a support flow that can issue a credit or a next-order voucher, never a fulfilment.

**Verification.** Fixture asserts a tap at 91 s produces a support-flow record and a `REFUNDED` terminal state.

### 12.8 F-08 — picker override used to bypass exclusions

**Mechanism.** The override channel is intended for "the shelf actually has stock, the scan was wrong", but it is also usable to present an excluded candidate. Overrides are logged but not reviewed, so the bypass is invisible.

**Discriminating check.** Report override reason codes by category; an elevated share of overrides in restricted categories is the signal.

**Bounded fix.** Restrict the override vocabulary so that no reason code permits presenting an excluded candidate; require a supervisor code for stock-correction overrides.

**Verification.** Fixture attempts an override on a restricted category and asserts rejection; attempts a stock-correction override and asserts it succeeds only with the supervisor code.

### 12.9 F-09 — contribution accounting that double-counts picking cost

**Mechanism.** The ledger uses candidate price minus COGS as contribution, and a separate team's store P&L allocates picking labour per line. Adding the two produces a total that exceeds the actual gain and makes the programme look better than it is.

**Discriminating check.** Reconcile the substitution ledger against the store P&L for one week; any double-counted cost line appears as a discrepancy.

**Bounded fix.** Publish the ledger with an explicit "excluded costs" column listing picking labour, packaging, delivery, and payment fees, and require the P&L owner to sign it.

**Verification.** Weekly reconciliation for four weeks with zero unexplained residual.

### 12.10 F-10 — the "no candidate" path becomes the default

**Mechanism.** As the exclusion filter tightens, more episodes return an empty candidate set. The presenter shows a spinner, the picker waits, then marks the line short. The policy silently becomes refund-only, and the primary metric looks flat while a guardrail is not breached.

**Discriminating check.** Track the empty-set rate by category and week; a rising trend with a flat primary metric is the signal.

**Bounded fix.** Surface an explicit "no safe substitute available" message with a one-tap refund, and monitor the empty-set rate as a first-class metric with an owner.

**Verification.** Dashboard assertion that the empty-set rate is reported; alert threshold set at a level agreed with the category owner.

### 12.11 What went wrong in the design process itself

The most instructive failure in this case is not any of the ten above. It is that the first version of the ledger omitted the price cap entirely and reported a gross gain of 5,853.32 INR/day. That number was arithmetically correct and strategically wrong, because it priced a policy that would have reproduced the complaint pattern the project existed to fix. The design process failure was optimising the model that was easy to write and forgetting the constraint that was hard to encode. The case keeps the cap's cost visible in the final ledger for exactly that reason.

## 13. Transferable Lesson

### 13.1 The lesson in one sentence

In a constrained recommendation problem, the constraint set and the reach of the decision are usually worth more than the ranking model, and a ledger that omits the cost of honouring the constraint will recommend the wrong policy.

### 13.2 The mechanism, stated so it generalises

Three conditions made reach dominant here. First, a human intermediary had discretion to skip the decision (the picker). Second, the skip was unmeasured in the primary metric because skipped lines simply became refunds. Third, the conditional quality of the decision was already acceptable. Whenever those three hold, the first intervention is to close the discretion and measure the closed path, and the ranking model comes second. The mechanism is not specific to quick commerce: it applies to fraud review queues where analysts skip cases, to collections where agents ignore prioritisation, to content moderation where reviewers triage manually.

### 13.3 Prerequisites for transfer

A transferable application needs: a measurable trigger event; a decision that is currently made by a human with discretion; a way to observe the outcome of the closed path; and a constraint set that can be encoded as a filter rather than a penalty. Without the fourth, the constraint becomes a soft penalty and the optimiser will trade it away.

### 13.4 Where the lesson does not transfer

- **When the constraint cannot be encoded as a filter.** If fairness must be enforced as a statistical property across a population rather than per-decision, the filter-first architecture does not apply and the tradeoff is genuinely continuous.
- **When the trigger is rare.** At 0.0184 requests/s the latency design is irrelevant; at a genuinely high request rate the systems constraints become binding and the reach intervention may be infeasible.
- **When acceptance is the binding problem, not reach.** If the incumbent already offers to nearly every failure and acceptance is 20%, closing discretion gains nothing and the ranking (or the assortment) is the lever.

### 13.5 The counterexample that sharpens the lesson

Suppose the store's pickers skipped offers because customers had repeatedly complained. Closing the skip channel would then *increase* harm while increasing the ledger. The case's guardrails (contact rate, safety, consent) exist to catch exactly this, and the pre-registered stopping rule names the contact-rate deterioration as a stop condition. The lesson is therefore not "always close the skip channel"; it is "measure the closed path before assuming the skip was wasteful".

### 13.6 What a reader should be able to do after this case

Given a similar decision, the reader should be able to: name the decision grain and trigger; write the constraint set as a filter order; build the reach/quality/cost decomposition; compute the break-even conditional quality; state the cap's cost explicitly; and design a validation ladder that does not stop at replay.

## 14. Connections & Cross-References

### 14.1 Related methods and artifacts

| Artifact | Relation to this case | What to read there instead of duplicating here |
|---|---|---|
| [Availability KPI tree](../Cheat_Sheets/01_Quick_Commerce_Availability_KPI_Tree.md) | Defines the line-level funnel this case assumes | The gate sequence, terminal-state conservation, and promise-error definition |
| Local assortment optimisation case (planned, `CASE-06-01-02`) | Owns the decision that changes which SKUs exist | Pack-size and equivalence coverage are assortment outputs, not substitution outputs |
| Perishable waste forecast (planned, `CASE-06-01-03`) | Supplies expiry risk that changes candidate eligibility | Expiry-aware candidate ranking is a separate contract |
| Exact inventory range totals (planned, `CASE-06-01-06`) | Supplies the total mechanism behind availability counts | Range-total correctness and snapshot visibility |
| Offer experiment checklist (planned, `Cheat_Sheets/02`) | General experiment discipline for consent and pricing tests | Pre-registration, interference, and guardrail design |

### 14.2 Open questions this case does not answer

1. Does a fairness cap reduce contacts, or does it merely reduce contribution while contacts stay flat? The case assumes the former; it has not tested it.
2. Is the 55% acceptance reachable with same-category candidates, or does it require cross-category equivalence? Cross-category raises both the opportunity and the safety risk.
3. What is the customer's actual evaluation function — unit price, brand continuity, absolute price, or pack size? The case assumes unit price plus brand continuity from the complaint pattern, which is a hypothesis.
4. Does a substitution that is accepted still reduce repeat purchase probability for that category? This is the trust question the ledger cannot see.
5. How should a partial-quantity substitution (2×250 g for 1×500 g) be priced when the pack overhead differs?

### 14.3 Interview pack

Each question below has a checked answer, the reasoning behind it, a follow-up, acceptable alternatives, red flags, a scoring guide, and the evidence that supports the answer. The questions progress from recall through calculation to judgement and diagnosis.

---

#### Q1. Why is "raise acceptance rate" an inadequate objective for this decision?

**Checked answer.** Because acceptance is conditional on what is offered, and the offer set is itself a policy choice. Raising acceptance by offering only the safest, most expensive-adjacent candidate can increase the rate while reducing the number of failures converted and the contribution per failure. The objective must combine reach (offers per failure), conditional acceptance, and contribution net of decline cost, with hard guardrails outside the objective.

**Reasoning.** The ledger has three multiplicative terms; optimising the middle one alone permits the outer two to collapse. The case's own numbers demonstrate the point: raising reach from 0.71 to 0.92 at unchanged acceptance adds more (2,895.30) than raising acceptance from 0.46 to 0.55 at unchanged reach (2,958.02 conditional on an unproven assumption) — and reach is the certain one.

**Follow-up.** If you could only measure one of the three for six weeks, which would you choose and why?

**Acceptable alternatives.** A stated composite objective (contribution per failed line) with guardrails; an explicit constraint formulation where acceptance is maximised subject to reach and safety floors.

**Red flags.** Answering "acceptance is what the customer controls" without noticing that the offer set is not customer-controlled; proposing AUC as the objective.

**Scoring.** 3 points: names conditioning on offer set. +1: names reach as a separate lever. +1: places guardrails outside the objective. 5 max.

**Answer support.** Section 6.2–6.5 decomposition; Section 7.1.

---

#### Q2. Compute the break-even conditional acceptance for the full policy and explain what it means operationally.

**Checked answer.** With `N_base = 9,788.8839`, `O = 676.2736`, `m_mix = 41.964`, `k = 3.6`:

```text
A* = (N_base + O*k) / (O*(m_mix + k))
   = (9788.8839 + 2435.0) / (676.2736 * 45.564)
   = 12,223.88 / 30,813.8 = 0.3967
```

So conditional acceptance must exceed 39.67%. Operationally this means the policy can tolerate a substantial degradation from the 55% design assumption — down to roughly 40% — before it stops beating the incumbent. Since the incumbent achieves 46% on a worse candidate distribution, the risk of the ranking change being *harmful* is bounded, which is what justifies piloting rather than proving first.

**Reasoning.** Break-even is found by setting the proposed net equal to the baseline net. The numerator adds back the decline cost because at zero acceptance the policy still pays decline costs for every offer it shows.

**Follow-up.** What happens to the break-even if the cap share rises to 40%?

**Acceptable alternatives.** Solving numerically by bisection and reporting 0.397; expressing it as "the policy needs 40% of a 92% reach". Accept 0.397 ± 0.005.

**Red flags.** Computing break-even against zero rather than against the baseline; forgetting the decline cost term; reporting 55% as if it were a threshold.

**Scoring.** 2 points: correct formula. +1: correct value within tolerance. +1: operational interpretation referencing the incumbent's 46%. +1: notes the decline-cost term's role. 5 max.

**Answer support.** Formula SUB-12; Section 6.7. Arithmetic re-verified: 676.2736 × 45.564 = 30,813.8; 12,223.88 / 30,813.8 = 0.39670.

---

#### Q3. The candidate is a 400 ml pack at 165 INR and the original is 500 ml at 180 INR. Is that a fair substitution, and how would you decide?

**Checked answer.** On unit price it is not fair: 165/400 = 41.25 INR per 100 ml versus 180/500 = 36.00 INR per 100 ml, a 14.58% premium per unit of content. On absolute price it looks cheaper. The honest decision is to compute unit-price parity and apply a tolerance band, then either choose a different candidate with parity or charge a capped amount so that the effective unit premium stays inside the band. The case's cap charges 400 × 1.05 × 0.36 = 151.20 INR, which holds the premium at 5% and costs the store 13.80 INR against the candidate's shelf price.

**Reasoning.** The customer's evaluation is on content-normalised value, not on absolute price, which is why the incumbent's cheapest-in-stock rule produces systematically disliked offers. The tolerance band converts a judgement into an arithmetic gate.

**Follow-up.** What if the original is 500 g of a solid and the candidate is 500 ml of a liquid?

**Acceptable alternatives.** A stated percentage band (5–10%); a per-category band set by the category manager; refusing substitutes with a unit premium above the band.

**Red flags.** Comparing 165 < 180 and declaring it cheaper; ignoring units entirely; applying a band without stating it.

**Scoring.** 2 points: computes both unit prices. +1: names the 14.58% premium. +1: proposes a band or cap. +1: quantifies the cap's cost. 5 max.

**Answer support.** Section 6.4 table; Formula SUB-09.

---

#### Q4. Your model reports 0.62 calibration and the pilot shows acceptance at 0.48. Diagnose before concluding the model is wrong.

**Checked answer.** Three candidates must be separated. First, **population shift**: calibration was measured on logged offers from the incumbent's candidate distribution; the pilot serves a different distribution, so the calibration does not transfer by construction. Second, **presentation effects**: the pilot shows the candidate with a cap and new copy, which changes the response to the same candidate. Third, **a genuine calibration defect**: the model's predicted probabilities were computed with a feature that is unavailable or different at serving time, for example an attribute that is null in production but imputed offline.

Discriminate by replaying the pilot's *actual* served candidates through the model and comparing predicted to observed; if calibration holds on the served set, the model is fine and the discrepancy is distribution or presentation. If it fails on the served set, inspect the feature-availability diff between offline and online.

**Reasoning.** Calibration is a property of a distribution and a feature contract, not of a model artifact in isolation. The most common real cause is an offline/online feature skew that the offline evaluation cannot see because the offline pipeline computes the feature from a complete table.

**Follow-up.** Which check would you run first if you had one hour?

**Acceptable alternatives.** Prioritising the offline/online skew check first, since it is cheap and decisive; using a reliability diagram on the served set.

**Red flags.** Immediately retraining; concluding the model is broken without checking the served distribution; blaming the pickers.

**Scoring.** 2 points: names distribution shift. +1: names presentation. +1: names feature skew. +1: proposes the decisive replay check. 5 max.

**Answer support.** Section 5d leakage/selection effects; Section 9.2 rung 4.

---

#### Q5. Why is filtering before scoring, rather than scoring then filtering, a correctness requirement and not a performance choice?

**Checked answer.** Because scoring-then-filtering makes the unsafe set reachable by any future refactor that relaxes the filter, and it contaminates the training distribution with candidates that are never shown. Filter-first makes the unsafe set structurally unreachable: the scorer's input type does not admit an unfiltered candidate, so a mistake is a type error rather than a policy regression. The performance argument is secondary and in this case irrelevant at 0.0184 requests/s.

**Reasoning.** Safety properties should be enforced by construction where possible, because review attention is finite and filter code is the least glamorous code in the system. A soft penalty on an unsafe candidate is tradeable by an optimiser; a filter is not.

**Follow-up.** How would you enforce the order in a codebase where the scorer is a shared service used by other teams?

**Acceptable alternatives.** A validated wrapper type; a contract test that fails if the scorer receives a candidate lacking a required exclusion tag; an assertion on the presence of an `exclusions_applied` marker.

**Red flags.** "We'll filter after scoring for recall"; treating the constraint as a penalty term; relying on review rather than construction.

**Scoring.** 2 points: unreachability argument. +1: training-distribution argument. +1: proposes an enforcement mechanism. +1: ranks safety over performance explicitly. 5 max.

**Answer support.** Section 5c; Section 8.6; F-06.

---

#### Q6. A picker says the shelf is empty but the system shows 4 units on hand. What do you do with that information?

**Checked answer.** Treat it as evidence that the balance is wrong, not as evidence that the picker is wrong, and not as evidence that the item is present. The decision path must fail toward "no offer from this SKU" and toward a physical-check exception. Then diagnose with a small decision tree: (a) is the reservation recorded? (b) is the location mapping correct for this SKU? (c) was there a recent receiving event that was entered but not put away? (d) is the unit a multi-pack counted as one? Each branch has a different owner.

**Reasoning.** The trigger for the whole case is a discrepancy between recorded and physical state. Using the record to override the human observation inverts the evidence hierarchy: the human is at the shelf, the record is a model of the shelf.

**Follow-up.** How would you use this event to improve the balance without creating a write-amplification problem?

**Acceptable alternatives.** Recording a discrepancy event and letting a reconciliation job adjust the balance after a threshold; requiring a second scan to confirm.

**Red flags.** Adjusting the balance to zero on a single report; ignoring the event; treating the picker as the error source by default.

**Scoring.** 2 points: evidence hierarchy. +1: fail-toward-safe direction. +1: named diagnostic branches with owners. +1: proposes a bounded reconciliation. 5 max.

**Answer support.** Section 4.1 A4; Section 5a inventory contract.

---

#### Q7. Design the test that would catch a unit-mismatch defect before launch.

**Checked answer.** A fixture pair. Positive case: original 1 l of milk, candidate 1000 ml of the same product — must be eligible, because the conversion is exact and the unit table contains the entry. Negative case: original 500 g of flour, candidate 500 ml of oil — must be ineligible even though the numeric amounts match, because no conversion entry exists and the dimensions are incompatible. A third case: original 500 g, candidate 2×250 g — must be eligible with a quantity-equivalence note. The test asserts on the *eligibility decision*, not on a score, so it survives model changes.

**Reasoning.** Unit defects are the highest-frequency category of silent substitution error because they pass every numeric check that ignores units. A fixture that only tests the happy path cannot distinguish a correct normalizer from one that drops the unit field.

**Follow-up.** Where does the conversion table live, and who owns it?

**Acceptable alternatives.** Property-based testing over generated unit pairs asserting that dimensionally incompatible pairs are always ineligible.

**Red flags.** Testing only same-unit cases; asserting on score thresholds; omitting the cross-dimension negative case.

**Scoring.** 2 points: positive and negative cases. +1: cross-dimension case. +1: asserts on eligibility not score. +1: names table ownership. 5 max.

**Answer support.** Section 5a quantity contract; F-03; Section 9.5.

---

#### Q8. The pilot improves contribution by 8% but the lagged contact rate rises 2.5 points. What is your recommendation?

**Checked answer.** Stop expansion and investigate, per the pre-registered rule, because a 2.5-point contact increase with a lower confidence bound above zero is a named stop condition. The pattern means the contribution gain is being financed by customer friction: more offers are converting, but more customers are also contacting support, which is both a cost and a trust signal. The right next step is to segment the contacts by reason — price surprise, wrong item, missing item, late — because only some reasons are attributable to the substitution policy.

**Reasoning.** Contact rate is a guardrail, not a secondary metric, because it captures harm the contribution ledger cannot see. A policy that improves the ledger while worsening contacts may still be net-positive economically and net-negative strategically, and the decision authority for that trade belongs to CX, not analytics.

**Follow-up.** If contacts rose only in the "wrong item" reason, what would you change?

**Acceptable alternatives.** Continue at current scale but freeze the ranking change and isolate the reach change; run a contact-reason decomposition before any decision.

**Red flags.** Continuing because the primary metric improved; ignoring the lag structure; attributing all contacts to the policy without decomposition.

**Scoring.** 2 points: invokes the stopping rule. +1: names financing-by-friction. +1: proposes reason decomposition. +1: names CX as decision authority. 5 max.

**Answer support.** Section 9.1 guardrail row; Section 10.4.

---

#### Q9. Why does the case keep a policy that costs 1,129 INR/day?

**Checked answer.** Because the ledger is not the objective. The cap keeps the effective unit-price premium inside a 5% band, and the acceptance estimate was formed under a presentation where parity was visible. Removing the cap would present a 14.58% unit premium on the model's preferred candidate, reproducing the complaint pattern that motivated the project. The cost is disclosed and bounded: even at a 60% cap share the policy still beats baseline by 2,773.57 INR/day.

**Reasoning.** A constraint that is dropped whenever it is inconvenient is not a constraint. The discipline is to price it, publish the price, and keep it unless the price crosses a stated threshold.

**Follow-up.** What evidence would justify widening the cap to 10%?

**Acceptable alternatives.** A trial of two cap levels with contact rate as the primary readout; a per-category cap set by the category manager with a stated rationale.

**Red flags.** Optimising the cap away because the model prefers it; keeping the cap without pricing it; claiming the cap improves contribution.

**Scoring.** 2 points: identifies constraint-not-objective. +1: quantifies the cost. +1: connects to the complaint mechanism. +1: states the boundary at which the cap would be revisited. 5 max.

**Answer support.** Section 6.5; Formula SUB-09.

---

#### Q10. An engineer proposes an embedding-based retrieval over the full catalog to improve candidate recall. Evaluate.

**Checked answer.** It solves a real problem (messy catalog text and cross-category equivalence) and creates an unauditable safety property. The question to ask is what recall gap the current leaf-category index has, measured against a manual audit of acceptable candidates. If that gap is small, the retrieval change buys little and costs the filter's auditability. If the gap is large, the right design is retrieval *inside* the filter: retrieve from an expanded candidate pool, then apply the same exclusion filter before scoring, and accept the latency cost. What is not acceptable is allowing retrieval to reach the scorer unfiltered.

**Reasoning.** The architecture's value is that one function answers the safety question. Any change that moves candidates around that function must preserve the function's position, not remove it.

**Follow-up.** How would you measure the recall gap without a labelled ground truth?

**Acceptable alternatives.** A human audit of 100 episodes per category, judging whether an acceptable candidate existed outside the returned leaf category; precision-at-k against auditor judgement.

**Red flags.** Accepting retrieval-first because it is modern; rejecting it without measuring the gap; allowing unfiltered candidates to reach the scorer.

**Scoring.** 2 points: separates recall benefit from safety cost. +1: proposes measuring the gap. +1: proposes retrieval-inside-filter. +1: states the non-negotiable position of the filter. 5 max.

**Answer support.** Section 8.6; Section 5e candidate-source row.

---

#### Q11. What is the minimum set of fields that makes an offer reconstructable after the fact?

**Checked answer.** Episode identity and idempotency key; event time of the scan; order line and SKU with quantity and content unit; the consent version used and its effective time; the candidate list with scores *including candidates removed by exclusions and the exclusion that removed each*; the policy version, model artifact hash, exclusion-list version, and as-of snapshot identifiers; the presented price and the charged price; the response with its timestamp; the terminal state. Without the removed candidates, a support agent cannot distinguish a bad model from a correct exclusion.

**Reasoning.** Reconstruction is the ability to answer "why did the customer see this?" from the log alone. Any field whose absence forces a replay of the model or a query against current state breaks reconstruction, because current state is not decision-time state.

**Follow-up.** Which of those fields would you drop first under storage pressure, and what capability do you lose?

**Acceptable alternatives.** Storing the full candidate set with scores and the exclusion trace as a single serialized blob; storing hashes of the policy inputs plus a version pointer, provided the versions are immutable.

**Red flags.** Logging only the shown candidate; logging the model score but not the exclusion trace; relying on current catalog state to explain a past offer.

**Scoring.** 3 points: candidate list plus exclusion trace. +1: versions and as-of identifiers. +1: price shown versus charged. 5 max.

**Answer support.** Section 4.3 auditability; Section 8.4 support read path.

---

#### Q12. The store's failure rate doubles during rain. How does that change the design?

**Checked answer.** It changes three things and not others. It changes **candidate pressure**: more simultaneous failures mean more competition for the same surviving candidates, so the second-order effect of the policy is that popular substitutes run out and the offered set degrades — the case's per-failure economics assume an independent candidate pool, which fails during a correlated event. It changes **latency**: request rate scales with failures, though at 0.0368 requests/s even a doubling is trivial. It changes **the decline path cost**: riders are delayed, contacts rise for delivery reasons, and the attribution of a contact to substitution becomes ambiguous. It does not change the exclusion logic or the consent gate.

**Reasoning.** Correlated failures break the independence assumption in the ledger, not the safety contract. The correct response is to monitor candidate-pool depth per category during weather events and to widen the presentation to second-choice candidates rather than to relax exclusions.

**Follow-up.** How would you detect candidate-pool exhaustion from logs alone?

**Acceptable alternatives.** Tracking the ratio of offers to distinct candidate SKUs per category per hour; alerting when the distinct-candidate count falls below a threshold while failure rate rises.

**Red flags.** Relaxing exclusions under pressure; assuming the ledger scales linearly; ignoring candidate competition.

**Scoring.** 2 points: candidate competition. +1: independence assumption broken. +1: latency still trivial. +1: contact attribution ambiguity. 5 max.

**Answer support.** Section 4.2; Section 6.6; Section 9.1 contact metric.

---

#### Q13. Your ledger says the programme is worth 1.72M INR/year. A finance partner says the number is inflated. Who is right?

**Checked answer.** Both, in different senses. The arithmetic is right for the stated assumptions, and the assumptions are explicitly illustrative: constant failure rate for 365 days, 55% acceptance, 22% cap share, 0.30 contact rate at 12 INR. The annualisation inflates because a store does not run at a constant rate, seasonal assortment changes candidate availability, and the policy's effect will decay as the easy conversions are captured. The honest statement is the daily delta with its sensitivity band, and the annual figure only as a scaling device. The finance partner is right that the annual number should not enter a payback calculation without a ramp and decay model.

**Reasoning.** Annualising a steady-state daily delta is the most common way an illustrative case becomes a false forecast. The fix is not to refuse to annualise but to label the annual figure as a scaling device and to state what would make it wrong.

**Follow-up.** What ramp and decay shape would you assume, and what would you measure to fit it?

**Acceptable alternatives.** Reporting a 90-day run-rate range; reporting only the daily delta and the break-even; adding an explicit "steady-state assumption" caveat in the claim ledger.

**Red flags.** Defending the annual figure as a forecast; refusing to give any annualised figure; not naming which assumption drives the inflation.

**Scoring.** 2 points: separates arithmetic from forecast. +1: names at least three inflating assumptions. +1: proposes a bounded reporting alternative. +1: identifies decay. 5 max.

**Answer support.** Section 6.8; Section 11.1 evidence table.

---

#### Q14. A colleague proposes adding a "customer likely to accept anything" feature derived from past acceptance history. Evaluate the feature.

**Checked answer.** It is a legitimate behavioural feature with three problems. First, **feedback loop**: past acceptance was observed only on candidates the incumbent chose to show, so the feature encodes the incumbent's policy rather than the customer's preference. Second, **fairness and consent**: using inferred acquiescence to target customers with lower-equivalence candidates is a consent-scope violation in spirit even if it is legal, and it concentrates harm on the least assertive customers. Third, **leakage risk**: if the feature is computed from any post-decision field it becomes a label proxy. It can be used only if it is computed strictly from pre-decision history, audited for differential candidate quality across the feature's strata, and excluded from the first candidate's ranking (used only for presentation depth).

**Reasoning.** Features that describe how a customer responded to a *policy* are policy artefacts. Using them to justify applying more of the same policy is circular, and the circularity is invisible in offline metrics that were computed on the same policy's log.

**Follow-up.** What audit would you run to detect differential candidate quality by feature stratum?

**Acceptable alternatives.** Using the feature only to decide how many candidates to show, not which; requiring an explicit consent field rather than inferred acquiescence.

**Red flags.** Accepting the feature without the loop analysis; using it to rank the first candidate; ignoring differential quality.

**Scoring.** 2 points: feedback loop. +1: consent/fairness. +1: leakage. +1: bounded permitted use. 5 max.

**Answer support.** Section 5d selection effects; Section 4.3 consent constraint.

---

#### Q15. In one paragraph, what would you tell a product director who wants to skip the pilot and go straight to full rollout?

**Checked answer.** That the certain part of the gain is a workflow change with no model risk and can go first, while the uncertain part depends on an acceptance lift that has never been served and on an attribute-coverage prerequisite that has never been audited. A full rollout bundles a certain 2,895 INR/day with an unproven 2,958 INR/day and a safety dependency, so a failure in the unproven part would force a rollback of the certain part too. Sequence them: close the skip channel in all stores with reason codes, run the coverage audit, then pilot the ranking in two stores. If the director insists on speed, the compromise is to close the skip channel everywhere and pilot the ranking in all stores at low traffic share, which preserves the certain gain and bounds the model's exposure.

**Reasoning.** The pilot exists to bound the downside of the unproven component, not to satisfy process. Bundling a proven and an unproven change destroys the ability to attribute a rollback, which is the real cost.

**Follow-up.** What single metric would you watch daily during the low-share rollout?

**Acceptable alternatives.** Any answer that separates the two changes and assigns different evidence requirements to each.

**Red flags.** Agreeing to full rollout without separating components; refusing any rollout without a pilot (over-conservative); proposing a pilot that also changes the workflow, destroying attribution.

**Scoring.** 2 points: separates certain from uncertain. +1: names the safety dependency. +1: proposes a sequencing. +1: names the attribution cost of bundling. 5 max.

**Answer support.** Section 10.2; Section 7.1.

---

#### Q16. What is the single most likely way this policy causes harm that no metric in Section 9.1 would catch?

**Checked answer.** Gradual assortment narrowing. If the policy consistently converts failures by offering a small set of popular equivalents, the store learns (through its own replenishment and category reviews) that those equivalents are the ones that sell, and the local assortment drifts toward them. Customers who wanted the original item slowly lose access to it, while every metric in Section 9.1 improves: acceptance rises, contacts fall, contribution rises. The harm is a slow reduction in the variety available to the customers whose preferences are least represented, and it is invisible in a per-line substitution metric because there is no line to fail when the item is not stocked at all.

**Reasoning.** Metrics that measure the substitution decision cannot observe the assortment decision, and the two are coupled through demand signals. This is the same class of failure as a recommender that narrows catalog diversity while improving engagement.

**Follow-up.** What would you monitor to detect it, given that out-of-assortment demand is unobserved?

**Acceptable alternatives.** Tracking distinct-SKU coverage per leaf category against a fixed reference list; monitoring search-with-no-result and request-for-item signals; a periodic customer panel.

**Red flags.** Naming a metric already in Section 9.1; identifying harm that is only operational noise; claiming no such harm exists.

**Scoring.** 2 points: names an assortment/diversity coupling. +1: explains metric invisibility. +1: proposes a detection signal outside the substitution metrics. +1: names the population bearing the harm. 5 max.

**Answer support.** Section 14.2 open question 4; Section 13.4 non-transfer condition on statistical constraints.

---

#### Q17. Why does the case use a 90-second response window rather than waiting for the customer indefinitely?

**Checked answer.** Because the picker is standing at a shelf inside a timed route and the bag will be sealed. Waiting indefinitely means either the picker blocks (queue behind them, missed delivery windows) or the bag is sealed while the offer is open (a late acceptance cannot be honoured). Ninety seconds is an operational budget derived from the picking loop, not a customer-patience estimate. The design consequence is that expiry must be terminal, with late taps routed to a support flow that can issue a credit rather than a fulfilment.

**Reasoning.** The system has two clocks — the picker's and the customer's — and the decision must close on the earlier one. Treating the customer's clock as authoritative creates a class of "accepted but unfulfillable" states that no downstream process can resolve cleanly.

**Follow-up.** What would you change if the store moved to a batch-picking model where the bag is sealed minutes later?

**Acceptable alternatives.** A longer window with an explicit "picker waiting" indicator; a two-phase offer where the picker can hold the bag.

**Red flags.** Treating the window as a UX preference; allowing late acceptance to fulfil; no terminal expiry state.

**Scoring.** 2 points: picker clock binds. +1: names the unfulfillable state. +1: terminal expiry plus support routing. +1: identifies how batching would change it. 5 max.

**Answer support.** Section 8.2 state machine; F-07.

---

## 15. Quick Reference Card

| Field | Content |
|---|---|
| **Decision** | At an empty-shelf scan, offer a substitute (which one, at what price) or refund — for one order line, inside the picking loop, versus the incumbent cheapest-in-stock rule |
| **Constraint order** | Consent gate → candidate generation → hard exclusions (allergen/diet/age/pack-form/quantity) → scoring → price cap → presentation. Exclusions fail closed; ranking never widens consent |
| **Baseline ledger** | 735.08 failures/day; 0.71 reach × 0.46 acceptance; 45 INR margin; 3.6 INR per decline → 9,788.88 INR/day |
| **Proposed ledger** | 0.92 reach × 0.55 acceptance; capped blended margin 41.964 INR → 14,512.97 INR/day; net +4,724.08 INR/day |
| **Lever decomposition** | Reach +2,895.30 (certain, no model); ranking +2,958.02 (unproven 55%); cap −1,129.24 (disclosed constraint cost) |
| **Break-even** | Conditional acceptance 39.67% beats baseline; incumbent already achieves 46% on a worse candidate distribution |
| **Fairness arithmetic** | 165/400 = 41.25 vs 180/500 = 36.00 INR/100 ml → 14.58% unit premium; cap charges 400×1.05×0.36 = 151.20 INR, costing 13.80 INR |
| **Sizing** | 0.0184 offer requests/s at peak; queue never builds; p99 dominated by dependency tail, not capacity |
| **Hard stop conditions** | Any safety-exclusion or consent violation; any price-shown ≠ price-charged; p99 > 350 ms for two hours; lagged contact rate +2 pts with lower bound above zero |
| **Evidence** | ILLUSTRATIVE ledger, VERIFIED arithmetic, OPEN attribute coverage, no pilot executed, no company claim |

**Supplemental record.** Evidence level: illustrative. Implementation stage: proposed. Editorial status: draft; `last_reviewed` null. Source basis: constructed teaching case with the availability KPI tree as its only internal dependency. Review trigger: any change to the consent scope, allergen attribute coverage, offer latency budget, or refund/support cost basis. Independent review, arithmetic re-execution, and any pilot remain OPEN.
