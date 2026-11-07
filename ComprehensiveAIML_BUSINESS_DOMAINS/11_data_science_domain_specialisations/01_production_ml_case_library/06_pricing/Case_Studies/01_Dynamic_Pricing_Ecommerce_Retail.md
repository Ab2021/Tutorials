---
case_id: CASE-01-06-001
title: Dynamic pricing for an ecommerce retail assortment
case_type: hybrid
domain: ecommerce pricing
primary_pattern: causal-confounding
secondary_patterns: [evaluation-mismatch, model-monitoring]
evidence_level: not-measured
source_basis: illustrative
as_of_date: 2026-11-07
last_reviewed: null
owner_role: pricing science lead
source_index: INDEX.md
review_trigger: Change in eligible products, pricing rules, or experiment design
editorial_status: draft-unreviewed
---
# Dynamic pricing for an ecommerce retail assortment

## 1. The One-Line Framing
Given a product-market-day, recommend one public shelf price for the next day to a retailer's pricing owner, maximizing expected contribution under inventory, price movement, and customer-protection constraints, against a fixed-price rule.
**Evidence:** This is a constructed teaching case, not an account of a particular retailer.
Every number, operating target, role, policy, and scenario below is illustrative; no system, experiment, or business outcome is claimed to exist.
The primary pattern is **causal-confounding**; secondary patterns are **evaluation-mismatch** and **model-monitoring**.
This is an allocation and constrained optimization decision repeated daily, not a personalized offer.
The important tradeoff is that a price with higher revenue can have lower contribution once costs are included.
Demand prediction alone does not reveal what demand would have been at a different price.
The proposed next action is to obtain experimental evidence, not to assert an optimized price works in production.

## 2. Context & Background
The fictional retailer sells replenishable, non-essential household goods through a single ecommerce channel.
The pilot covers one market, one currency, and a defined cohort of eligible products.
A product-market-day is the decision unit; all customers in that market see the same shelf price for that product during the assigned interval.
The baseline is a manually approved fixed shelf price, held unless the merchant intervenes.
The baseline is a constructed comparator, not a reported practice or measured deficiency.
The merchant decides the price; the pricing scientist proposes the policy; operations and legal can stop the pilot.
Catalog operations maintain product eligibility and vendor restrictions.
Inventory operations own availability and replenishment inputs.
Finance defines contribution and matures return-related adjustments.
Customer support handles price display disputes and records complaints.
The proposed decision happens before the local trading day starts.
The price system publishes an approved version to listing, product, cart and checkout surfaces.
An already accepted order retains its contracted price; legal and product teams must approve any cart-price policy before exposure.
Illustrative planning scale is 10,000 eligible SKUs and one batch decision per SKU per day.
This is a workload assumption, not a retailer's actual volume.
Peak shopping traffic affects read capacity separately from batch recommendation volume.
Cross-product substitution and common inventory introduce interference across product decisions.
No actual geography or applicable legal rule is established here.
Legal approval and jurisdiction-specific requirements are prerequisites, not assumed facts.

## 3. The Business Question
**Illustrative owner question:** Should the retailer replace the current rule with a bounded, inventory-aware daily price policy for eligible products?
The decision owner is the pricing director, who can approve, reject, or defer the policy.
The trigger is a hypothetical quarterly assortment review; the deadline is before a proposed pilot.
The actions are retain baseline, test a constrained price policy, or defer until identification and operations improve.
The primary objective is incremental realized contribution per eligible product-day, compared with the current rule, after a prespecified maturity period.
Contribution includes revenue minus merchandise cost and specified variable transaction, fulfillment, and return costs.
Lost sales, customer harm, legal breaches, and operational outages are not acceptable ways to improve that objective.
Price constraints are hard eligibility gates, not penalties that can be traded off against profit.
The scope excludes personalized pricing, sensitive products, private seller offers, advertising auctions, and supplier contract renegotiation.
A price change is expensive when it erodes customer trust or exhausts inventory that would have served later customers.
Conversion and revenue are diagnostic proxies, not the decision objective.
A final decision can be to collect evidence rather than launch.

## 4. Assumptions & Constraints
| ID | Condition and type | Status / verification | Consequence and fallback | Owner / trigger |
|---|---|---|---|---|
| A1 | Causal demand response transports to eligible pilot items | Unverified; randomized price contrasts | Wrong price ranking; retain baseline | Science; assortment or season changes |
| A2 | Product inventory is available and timely | Unverified; reconcile snapshots with fulfillment | Overselling or false low demand; freeze SKU | Inventory; stale or discrepant feed |
| A3 | Costs and return liabilities can be attributed | Unverified; finance ledger reconciliation | Wrong contribution; use conservative cost or exclude | Finance; vendor or fee change |
| A4 | Public price is consistent across surfaces | Unverified; synthetic journey and receipt comparison | Broken promises; stop publication | Commerce engineering; version drift |
| A5 | Sufficient eligible clusters and stable exposure | Unverified; pre-pilot design and power study | Inconclusive treatment effect; expand horizon or defer | Experimentation; recruitment shortfall |
| C1 | Price lies between approved floor and ceiling | Scenario rule; rules team must approve | Reject proposed price, retain approved price | Merchant; rule update |
| C2 | Price moves at most 10% from last approved daily price | Scenario rule, not a law | Restrict feasible grid; retain valid baseline if empty | Merchant; override |
| C3 | No protected-attribute or individual-specific price | Scenario policy; legal review outstanding | Reject request and audit | Legal; new use case |
| C4 | Existing order terms must be honored | Scenario contractual safeguard; legal confirms | Hold order price and investigate | Commerce; inconsistent checkout |
| C5 | Inventory must exceed safety reserve before an experimental markdown | Scenario safety rule | Disallow markdown or exclude product | Inventory; reserve update |
The three assumptions most likely to reverse the recommendation are A1, A2 and A3.
A scenario constraint is not proof that any real regulation or contract imposes it.
An infeasible rule set must be escalated; simply falling back to a now-invalid price is unsafe.
A manually approved emergency price is valid only after the same applicable rules are checked.

## 5. Data, Tools & Method
### 5a. Data Specification
The panel grain is product × market × local day, with a separate event grain for exposure and orders.
An illustrative historical window is 52 weeks, but its existence and sufficiency are not established.
Product, market, day, decision version, and channel identify a policy decision.
Exposure events additionally require session and event identifiers; orders require unique order-line identifiers.
The decision cutoff precedes the day being priced; point-in-time joins select only records available by that cutoff.
Price is the public shelf price in currency per unit, before any separately recorded discount.
Record displayed, cart, and paid prices independently to detect inconsistent treatment.
Cost is landed unit cost in currency per unit, effective and known at decision time.
A separate mature ledger records refunds, variable fulfillment, and other prespecified contribution adjustments.
Inventory is sellable on-hand units minus allocations, with event time and ingestion time.
Reservations and replenishment expectations are distinct fields, not interchangeable with physical stock.
Sales quantity is fulfilled units, not latent demand; stockouts censor sales.
Record stockout times, page impressions, search placement, delivery promise and known marketing campaigns.
For causal estimation, log assignment probabilities, eligibility, cluster, time block and actual displayed treatment.
Zero sales on an available day are valid outcomes; an unlisted product is not a zero-demand observation.
Exclude products with invalid cost, inconsistent product identity, contract restriction, or unresolved stock data before randomization.
Record exclusions and compare eligible with excluded products to delimit the estimand.
Use time-ordered training, validation and final test periods, with product-group boundaries appropriate to the target population.
Do not fit encoders, imputation or scaling on the final test period.
Delayed return labels mature on a fixed window; reconcile late adjustments rather than silently relabeling past outcomes.
Link order lines by stable keys with documented one-to-many relationships and deduplicate event IDs before aggregation.
Count orphan events and late arrivals; fail the relevant batch when reconciliation exceeds a pre-agreed tolerance.

### 5b. Tools & Stack
A proposed warehouse stores immutable event snapshots and effective-dated dimensions.
A proposed Python batch job validates inputs, estimates demand and enumerates approved price candidates.
A versioned artifact registry stores model, feature specification, pricing rules and cost assumptions together.
An approval service publishes versioned prices to the commerce catalog.
A separate monitoring pipeline joins exposures, sales, inventory and finance outcomes after maturity.
These are roles in a design, not assertions that a named product, version or implementation was used.
Infrastructure owners would select versions and deployment configuration before any implementation.

### 5c. Method Summary
The baseline holds the approved price for each eligible product, except documented merchant actions.
The proposed method predicts demand under each feasible candidate and selects the candidate with maximum expected constrained contribution.
A simple rule-based markdown is an alternative; it is transparent but can react to stock without identifying demand response.
An observational regression is another alternative; it predicts within its historical distribution but can learn biased price coefficients.
A randomized price-grid experiment is proposed to identify local response within approved limits.
A bandit is not selected for the initial pilot because adaptive assignment complicates inference and exposure governance.
None of these alternatives has been tested in an actual business in this case.
The optimizer enumerates the discrete permitted price grid; ties keep the current price to avoid unnecessary churn.
The stopping condition is evaluation of all feasible candidates; there is no iterative numerical solver in this proposal.
A candidate is eligible only after floor, ceiling, movement, cost, inventory and merchant-rule checks.
An uncertain estimate outside experimentally supported price ranges is not a reason to extrapolate or lift the bounds.
If no valid candidate remains, do not publish; escalate to the merchant and use an approved safe state.

### 5d. Data provenance, quality, exclusions and leakage controls
The case contains no actual data extract and no actual business event logs.
A real implementation would document source owner, permissions, snapshot ID, extraction time, and retention for every table.
The catalog owns product identity; commerce owns exposure and order events; inventory owns stock records; finance owns cost and returns.
A price change caused by a promotion can coincide with advertising or seasonal demand, creating confounding.
Post-decision sales, rankings, returns and inventory movements must not become same-day predictive inputs.
Prior-day versions are usable only when available before cutoff, not merely timestamped earlier by an upstream source.
Use both event and ingestion timestamps to detect late-arriving data and avoid hindsight joins.
Do not use current realized units to infer the counterfactual inventory at a different price.
Audit missingness and exclusions by category and treatment; missing outcomes can change the target population.
Use access-controlled, minimized customer data, retaining only what the approved attribution and audit purposes require.
Anonymization does not by itself guarantee privacy; retention and access rules require local approval.

### 5e. Reproducibility and method-selection ledger
| ID | Artifact / reference | Status and scope | Reproduction boundary |
|---|---|---|---|
| SRC-01 | Source-tree filename in directory.txt | Topic provenance only | Source content and business results not supplied |
| SRC-02 | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Method reference for leakage | Not evidence of a pricing result |
| SRC-03 | [Google SRE SLO workbook](https://sre.google/workbook/implementing-slos/) | Method reference for user-centered reliability | Not a target or benchmark for this proposal |
| ART-01 | Cost, demand and candidate grid in Section 6 | Constructed numerical example | Arithmetic only; no fitted model |
| RUN-01 | Experiment and deployment | Not executed | No code, data snapshot, seed or run ID exists |
For a real run, record code revision, package versions, parameters, split boundaries, randomization seed, data snapshots and model checksum.
Keep the assignment ledger and rule version with each decision so a reviewer can reconstruct eligibility and exposure.
A seed does not guarantee bitwise reproducibility across changed runtime, libraries, or nondeterministic compute.

## 6. Analysis Walkthrough
The decisive teaching tradeoff is that lower modeled revenue need not imply lower modeled contribution.
All inputs in this section are illustrative, not estimates from a retailer.
Suppose an item has a current approved price of 100 currency units, a unit merchandise cost of 60, and variable cost of 5 per fulfilled unit.
Assume no fixed cost changes between candidate prices, no taxes included, and all relevant returns already represented in the variable cost assumption.
Assume at least 100 sellable units remain above the safety reserve for the day, so inventory does not cap any candidate in this first example.
The allowed prices are 90, 100, and 110, respecting the illustrative 10% daily movement rule and approved bounds of 90 to 110.
Assume for this calculation the potential demand under each candidate is known: 100, 80, and 60 units, respectively.
In practice those counterfactual quantities must be estimated; their values here are not empirical evidence.
Validity: the following deterministic calculation applies only if candidate-specific fulfilled units and variable costs are as stated, with no cross-product effects.
Derivation: each sold unit earns price minus merchandise and variable costs, multiplied by fulfilled units.
```text
FORMULA: contribution(p) = (p - c - v) × q(p)
WHERE: p, c, v are currency/unit; q(p) is fulfilled units/day.
CONSTRAINT: feasible p and known q(p); costs include all variable components relevant to the comparison.
NUMERICAL EXAMPLE: at p=90, (90 - 60 - 5) × 100 = 2,500 currency/day.
INTERPRETATION: this is modeled daily contribution, not realized profit.
```
| Price | Demand, units/day | Revenue, currency/day | Unit contribution | Total contribution, currency/day |
|---:|---:|---:|---:|---:|
| 90 | 100 | 9,000 | 25 | 2,500 |
| 100 | 80 | 8,000 | 35 | 2,800 |
| 110 | 60 | 6,600 | 45 | 2,700 |
Under these assumptions, retain 100 because 2,800 exceeds 2,700 and 2,500.
The 90 price has the highest revenue, but 300 less contribution than the baseline price.
An independent arithmetic check is 80 × 35 = 8 × 35 × 10 = 2,800.
The 110 candidate loses 100 contribution relative to baseline despite its higher unit margin.
A price-dependent demand estimate may be systematically wrong if merchant promotions coincide with high expected demand.
The optimizer cannot repair a biased or unsupported demand curve.
**Sensitivity — changed unit cost:** if merchandise cost rises to 70 while other assumptions stay fixed, contributions are 1,500, 2,000, and 2,100 respectively.
At that cost the highest-price candidate becomes best by 100; the recommendation is cost-sensitive.
**Sensitivity — scarce inventory:** if only 50 units can be sold above the reserve and all demand is certain, each candidate sells 50 units.
Their contributions are 1,250, 1,750, and 2,250; the 110 candidate wins in this one-day example.
This naive cap is not a general expected-demand formula: expected minimum of random demand and stock is not minimum of their expectations.
It also ignores the option value of stock in later days and substitution into other products.
For stochastic demand, estimate the expectation of margin times fulfilled units and subtract any modeled, defensible inventory opportunity cost.
If replenishment is delayed, a multi-day inventory model may reverse the one-day ranking; defer if its assumptions cannot be supported.
**Not tried:** no historical regression, randomized price test, or deployment has been executed.
**Highest leverage:** identify price response in the eligible region before treating a ranking as actionable.

## 7. The Finding
For the constructed one-day example, retain the 100 price: modeled contribution is 2,800 versus 2,500 at 90 and 2,700 at 110.
These values are a calculation from assumed demand, not a measured or causal business outcome.
The preferred price changes under the cost and inventory sensitivity examples, and the demand curve has not been identified.
The owner should therefore retain the existing approved rule while designing an ethically and operationally bounded randomized test.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
```text
Versioned catalog + cost + inventory + exposures + mature orders
  -> cutoff / schema / lineage checks -> quarantine invalid product-days
  -> point-in-time feature panels -> temporal splits -> demand model or experiment analysis
  -> candidate-grid optimizer -> constraint gate -> merchant approval
  -> versioned price manifest -> catalog publish -> reconciliation and audit
```
The batch produces recommendations, not authoritative customer prices, until the merchant approves the manifest.
Train only on observations available before the cutoff and log the exact snapshot and model version.
Calculate uncertainty by candidate; unsupported or highly uncertain decisions fall back to a previously approved valid price.
The gate checks effective-dated rules before publication and again if rules change during approval.
Training, evaluation, configuration and serving artifact versions are bound in the manifest.
Failures quarantine affected SKUs rather than silently publishing incomplete estimates.

### 8b. Online / Serving / Real-Time Path
```text
Shopper -> catalog price read -> listing / product / cart -> checkout
               |                        |                    |
         approved version       exposure event         committed order price
               |                        |                    |
       last valid safe price      audit / metrics       order / finance ledger
               ^                        |
       rule-validated fallback <- incident control <- mismatch alert
```
The shopper reads a precomputed public price; model inference is not required on the checkout path.
Catalog reads authenticate internal clients, use scoped service identity, and return price version and effective time.
Price display caches must invalidate or expire within a defined propagation window before publication is considered complete.
Checkout verifies the applicable price version and honors binding order terms, with exceptions defined by approved policy.
Exposure, assignment, decision version and order events are written with unique identifiers for later reconciliation.
If the catalog dependency fails, a previously approved price can be served only while still effective and compliant.
If that cannot be verified, the safe action is to stop offering the affected SKU, not invent a price.
A kill switch prevents new dynamic publications; it cannot rewrite orders already committed.

### 8c. Key Design Decisions
| Choice | Alternative | Reason / boundary | Revisit trigger |
|---|---|---|---|
| Daily SKU-market price | Individual price | Auditability and consistent exposure; market effects can interfere | New legal or product policy |
| Enumerate valid grid | Continuous optimizer | Clear hard-rule checks; coarse grid can miss an optimum | Grid granularity review |
| Cluster randomization | Session-level randomization | Avoid simultaneous public prices and within-cluster contamination | New independent market boundary |
| Price manifest | Online model call | Checkout survives model outage; manifest may go stale | Freshness or scale requirements |
| Hold current approved rule | Automatic unsupported extrapolation | Safer under weak causal identification; misses possible gains | Experiment establishes local response |

### 8d. Architecture Artifacts and Operational Readiness
The pricing director owns rule approval and the on-call commerce engineer owns publication recovery.
The proposed contract contains SKU, market, currency, price, effective interval, rule version, model version, decision ID and approval ID.
The catalog rejects malformed, unauthorized, stale, out-of-bound or overlapping active manifests.
Use a unique decision ID and idempotent writes so retried publishes do not create multiple price changes.
A rule update during publication requires revalidation rather than replaying a stale manifest.
The ledger is append-only for decisions and approvals; price state has a single authoritative owner.
Backups must cover manifests, rules, approvals and decision-event offsets, with access and retention approved by governance.
Proposed user-facing SLI: fraction of eligible product views where a valid current price is shown and matches checkout policy, measured over a rolling 30-day window.
An illustrative target of 99.9% is a planning objective, not a measured service level or universal requirement.
The denominator is eligible views with a resolvable price contract; rejected requests remain separately monitored.
Synthetic end-to-end journeys and sampled real journey reconciliation provide complementary evidence.
Error-budget exhaustion pauses expansion and triggers investigation by the commerce on-call and pricing owner.
Timeouts and retry budgets must be configured per dependency; retries use backoff and cannot bypass idempotency or approval.
Alert on invalid price, stale inventory, display-checkout mismatch, assignment imbalance and unaccounted orders.
Shadow mode compares recommendations without changing displayed prices; a canary then uses approved, randomized scope.
Rollback freezes new publications, restores an approved compatible manifest, invalidates caches, and verifies all surfaces.
If schema or rules changed, the old manifest must pass current checks; otherwise withdraw affected offers and escalate.
The mechanism does not claim a rollback time without a tested recovery exercise.
**Illustrative capacity calculation:** 10,000 product decisions/day over a two-hour batch window is 10,000 / 7,200 ≈ 1.39 decisions/second on average.
If each evaluates 11 candidate prices, that is 110,000 candidate evaluations per day, approximately 15.28 per second averaged over the two hours.
At 2 KB per decision record, 10,000 records require about 20 MB/day before indexes, replicas, compression or telemetry.
At 30 days of unreplicated records, the arithmetic estimate is about 600 MB; actual storage requires measurement and retention approval.
These averages do not establish peak capacity, runtime, compute sizing or end-to-end latency.
Read traffic, catalog fanout, inventory joins and bursty publish concurrency may dominate the batch calculations.
Capacity validation would model peak reads, maximum concurrent publication, cache invalidations, retries, network and recovery replay.
No component percentile is added to claim an end-to-end p99.

## 9. Evaluation & Validation
The intended primary estimand is the treatment-minus-control mean mature contribution per eligible randomized product-cluster-day over the prespecified pilot horizon.
The numerator sums finance-reconciled contribution from all eligible cluster-days, including zero-sale days.
The denominator is all eligible randomized cluster-days, not only purchasers or exposed checkout sessions.
The comparison is intention-to-treat: assigned policy versus baseline policy, regardless of compliance.
Assign mutually substitutable products within a market together where feasible; randomize independent market-category clusters in time blocks.
The exact cluster definition and minimum sample size require a pre-pilot design based on availability, correlation, interference and variance.
Use blocked assignment to balance prespecified product and market characteristics, and log the randomization probability.
A finite number of clusters, shared stock and customers crossing markets can limit independence and effective sample size.
Estimate uncertainty at the randomization-unit level, using an analysis suited to the actual blocking and number of clusters.
If few independent clusters are available, use an appropriate randomization-based analysis or redesign rather than treating product-days as independent.
Predefine the minimum economically meaningful effect, significance or decision interval, maximum duration and interim stopping rule before assignment.
Check sample-ratio mismatch against the assigned allocation and distinguish eligibility errors from post-assignment attrition.
Track price compliance, stockouts, complaints, margin floor violations, service reliability and customer cohort disparities as guardrails.
A customer fairness audit requires an approved definition, adequate subgroup data, privacy review and a plan for small or missing groups.
Do not interpret lack of statistical significance in a small subgroup as proof of fairness.
Returns and cancellations mature on a prespecified horizon; report preliminary and mature windows separately.
Analyze stockout and substitution effects as part of policy impact, not by excluding post-treatment low-stock observations.
A separate local randomized price-grid design can estimate response at permitted prices under sufficient overlap and stable exposure.
Counterfactual claims need consistency, randomized or defensibly adjusted assignment, overlap and an explicit interference boundary.
Observational adjustment alone cannot guarantee removal of unmeasured merchant information.
Check pre-treatment balance, treatment delivery, missingness and alternative exposure definitions without replacing the primary estimand.
Status: no experiment, analysis, sample-size calculation or statistical result has been executed.
Decision: not measured; preregister and obtain approvals before any pilot.

## 10. Action Taken
No action taken as of 2026-11-07; this is a design exercise with no identified organization or deployment.
The proposed recommendation is to maintain the approved baseline while preparing a cluster experiment and operational controls.
Pricing, finance, legal, inventory, experimentation and commerce would each sign off their respective responsibilities before exposure.
The pilot would begin with shadow validation, followed by a small randomized release and gated expansion.
Any invalid-price breach, checkout inconsistency, severe customer impact or critical data failure would stop publication.
The incident commander and pricing director would authorize rollback under a preapproved runbook.
No effective market, implementation date, or model version exists in this exercise.

## 11. Outcome & Results
**Source declaration:** Figures elsewhere are constructed teaching values, not measurements of any named organization or realized impact.
| Metric | Baseline | Treatment | Change / uncertainty | Evidence / limitation |
|---|---|---|---|---|
| Mature contribution per eligible cluster-day | Not measured | Not measured | Not available | No pilot or finance data |
| Invalid-price rate | Not measured | Not measured | Not available | No live deployment |
| Complaint rate | Not measured | Not measured | Not available | No customer outcomes |
| Price consistency SLI | Not measured | Not measured | Not available | Proposed target only |
The numerical ranking in Section 6 is not a before/after outcome and is not placed in this table as one.
The experimentation owner would report both assigned groups after the agreed pilot and label-maturity window.
The decision owner would compare the interval and guardrails with preregistered criteria; otherwise the result remains inconclusive.

## 12. Failure Modes & What Went Wrong
No incident occurred in this constructed case; all diagnostic scenarios below are **hypothetical**.
Each scenario is a distinguishable failure hypothesis, not a claim that the system was built or that a mitigation succeeded.
### F-01. Promotions confound price
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Sales rise when price falls during a campaign.
- **First evidence:** Check campaign timing and randomized price contrasts.
- **Candidate mechanism:** Promotion exposure rather than price may drive lift.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare within randomized campaign strata.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Pause causal use of observational coefficients.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-02. Merchant anticipates demand
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** High prices correlate with strong sales.
- **First evidence:** Compare merchant forecasts available before the change.
- **Candidate mechanism:** Expected demand may cause the price decision.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Audit forecast logs against assignment.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Keep manual baseline until randomized contrasts exist.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Pricing science coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-03. Stockout censoring
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A low-price day shows unexpectedly few units.
- **First evidence:** Inspect in-stock minutes and replenishment events.
- **Candidate mechanism:** Actual demand may exceed observed fulfilled units.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare availability-aware exposures across arms.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Do not interpret low sales as low demand.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Inventory science coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-04. Missing zero-sales days
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Estimated demand exceeds reality.
- **First evidence:** Reconcile panel days against eligible catalog calendar.
- **Candidate mechanism:** An inner join may remove valid zero outcomes.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Count no-order eligible days before and after join.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Rebuild the panel with explicit zero units.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Data engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-05. Future cost leakage
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Backtest contribution is implausibly accurate.
- **First evidence:** Compare cost ingestion timestamps to decision cutoff.
- **Candidate mechanism:** Later vendor invoices may overwrite historical cost.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Reconstruct as-known cost snapshots.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Invalidate leaked backtest and retrain.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Finance data coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-06. Late exposure events
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Some assigned products appear never shown.
- **First evidence:** Inspect event-time versus ingestion-time lag.
- **Candidate mechanism:** Telemetry delay may mimic treatment noncompliance.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Reconcile catalog access logs with exposure stream.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Hold analysis until agreed lateness window.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Observability lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-07. Duplicate orders
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Contribution appears to jump after retry storm.
- **First evidence:** Count order-line IDs and delivery attempts.
- **Candidate mechanism:** At-least-once delivery may duplicate ledger records.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare immutable order IDs with event IDs.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Deduplicate by order-line lifecycle semantics.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Commerce data coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-08. Partial refunds omitted
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Revenue improves while realized profit falls.
- **First evidence:** Reconcile return ledger by original order line.
- **Candidate mechanism:** Return liability may be missing or delayed.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare mature cohorts with and without refund adjustments.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Use mature contribution for launch decision.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Finance coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-09. Incorrect currency conversion
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A market appears unprofitable across all prices.
- **First evidence:** Inspect currency code and effective exchange-rate timestamps.
- **Candidate mechanism:** Cross-currency joins may mix local and reporting units.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Recompute a sampled receipt in local currency.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Quarantine affected market until reconciled.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Finance data coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-10. Tax treatment mismatch
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Checkout margin differs from modeled margin.
- **First evidence:** Trace tax-inclusive and tax-exclusive price fields.
- **Candidate mechanism:** Tax may be counted as revenue or subtracted twice.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Reconcile a receipt with finance definitions.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Freeze affected product calculations.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Tax and finance coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-11. Price rounding breach
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A candidate looks valid before publication but not after rounding.
- **First evidence:** Compare pre-round and published prices against effective bounds.
- **Candidate mechanism:** Currency rounding can cross a floor or ceiling.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Run constraint checks after quantization.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Reject invalid candidate rather than silently clamp.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Pricing engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-12. Movement measured from wrong baseline
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Daily price jumps farther than permitted.
- **First evidence:** Inspect prior approved versus proposed price.
- **Candidate mechanism:** An unpublished recommendation may be treated as the previous price.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Replay with authoritative publication history.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Restore an approved valid manifest.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Catalog engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-13. Contradictory rules
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** All candidates fail the constraint gate.
- **First evidence:** Inspect effective floor ceiling and movement limits.
- **Candidate mechanism:** A rule update may make the grid infeasible.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Evaluate each rule against the same timestamp.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Withdraw or escalate instead of publishing invalid fallback.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Merchant operations coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-14. Stale inventory snapshot
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Markdown occurs just before stock exhaustion.
- **First evidence:** Compare feed timestamps and physical reservation state.
- **Candidate mechanism:** Delayed inventory may overstate sellable stock.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Reconcile inventory service and warehouse snapshots.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Freeze markdowns for affected products.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Inventory operations coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-15. Double-counted reservations
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Optimizer acts as if inventory is scarce.
- **First evidence:** Trace allocated units through inventory fields.
- **Candidate mechanism:** Reservations may already be excluded from sellable stock.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Recalculate availability from the authoritative contract.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Use baseline until the definition is corrected.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Inventory engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-16. Replenishment promise fails
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Projected stock remains high but availability collapses.
- **First evidence:** Compare expected receipts with received units.
- **Candidate mechanism:** Supplier delays can invalidate the availability forecast.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Inspect supplier confirmations and actual receipts.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Constrain markdowns and reassess horizon.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Supply chain coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-17. Shared-stock interference
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Control products sell out during treatment.
- **First evidence:** Map shared inventory pools and substitution paths.
- **Candidate mechanism:** Treatment demand can consume stock available to control.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Inspect stock movements by randomized cluster.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Redesign clusters around inventory pools.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-18. Cross-product cannibalization
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A promoted SKU gains while category contribution drops.
- **First evidence:** Compare category-level units and contribution by assignment.
- **Candidate mechanism:** Demand may move from a substitute rather than increase.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Measure prespecified category-level effect.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Evaluate portfolio outcome before expansion.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Category owner coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-19. Unsupported price extrapolation
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Model proposes the most extreme allowed price.
- **First evidence:** Inspect historical and randomized support by price.
- **Candidate mechanism:** A flexible model can extrapolate beyond observed variation.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Plot response and uncertainty inside the local support.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Restrict candidates to supported intervals.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Pricing science coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-20. Flat response from bad join
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Predicted demand is identical for all candidate prices.
- **First evidence:** Trace candidate price into the model feature vector.
- **Candidate mechanism:** Feature overwrite or stale cache can remove variation.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Replay one SKU at distinct valid candidate values.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Disable optimizer until candidate features differ.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** ML engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-21. Model monotonicity violation
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Higher prices predict implausibly larger demand.
- **First evidence:** Inspect comparable randomized observations and price features.
- **Candidate mechanism:** Confounding or data errors can produce a positive association.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare response across similar supported contexts.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Require review; do not blindly enforce a false universal law.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Pricing science coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-22. Outlier-driven margin
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** One product dominates the aggregate uplift.
- **First evidence:** Inspect product-level influence and data validity.
- **Candidate mechanism:** A legitimate high-volume item or a data error can dominate.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Recompute with and without the item as sensitivity.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Investigate before applying any exclusion.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-23. Wrong price displayed
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Product page and checkout show different prices.
- **First evidence:** Compare manifest version and cache headers on both surfaces.
- **Candidate mechanism:** Cache invalidation or propagation may be incomplete.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Trace a single SKU across all surfaces.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Stop changes and honor applicable commitments.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Commerce on-call coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-24. Stale safe-price fallback
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Catalog outage triggers an expired previous price.
- **First evidence:** Compare fallback effective interval and current rules.
- **Candidate mechanism:** A previously valid price may now be prohibited.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Exercise outage replay after an effective-date change.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Withdraw affected offer if no valid fallback exists.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Commerce on-call coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-25. Duplicate publish
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** An operator retries after a timeout and sees repeated updates.
- **First evidence:** Inspect decision IDs and catalog write audit.
- **Candidate mechanism:** The first request may have succeeded despite timeout.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Replay identical idempotency key in staging.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Stop retries until write status is reconciled.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Platform engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-26. Split-brain price owner
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Two services publish competing price manifests.
- **First evidence:** Inspect authorization and overlapping effective intervals.
- **Candidate mechanism:** Multiple writers can race without a shared authority.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare version histories and writer identities.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Revoke unauthorized writer and reconcile state.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Security and platform coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-27. Unauthorized override
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A price changes without approved merchant action.
- **First evidence:** Trace principal approval ID and signature.
- **Candidate mechanism:** An overprivileged service or compromised credential may write.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare policy logs to the approved change record.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Revoke access and restore a compliant price.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Security on-call coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-28. Invalid approval after rule change
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A formerly approved manifest violates new bounds.
- **First evidence:** Compare approval time and rule effective time.
- **Candidate mechanism:** Approval may refer to an obsolete rule set.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Re-evaluate manifest against current rules.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Block publish and obtain new approval.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Pricing governance coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-29. Treatment assignment imbalance
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** One arm has far fewer eligible clusters.
- **First evidence:** Check assignment ledger before looking at outcomes.
- **Candidate mechanism:** Hashing, eligibility or instrumentation can create imbalance.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare assigned cluster counts to planned allocation.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Pause enrollment and diagnose without reallocating ad hoc.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-30. Post-treatment eligibility filter
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Treatment looks profitable after dropping unavailable products.
- **First evidence:** Reconstruct eligibility at randomization time.
- **Candidate mechanism:** Treatment itself may cause stockouts and exclusions.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare intention-to-treat with filtered analysis.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Restore prespecified denominator.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-31. Underpowered cluster test
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Point estimate is large but confidence interval is wide.
- **First evidence:** Count independent clusters rather than product-days.
- **Candidate mechanism:** Within-cluster correlation reduces effective information.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Inspect cluster-level outcome variation.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Continue only under the prespecified design or conclude inconclusive.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Statistician coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-32. Network contamination
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Customers observe more than one experimental market.
- **First evidence:** Inspect delivery geography and cross-market customer journeys.
- **Candidate mechanism:** A shared audience can dilute the treatment contrast.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Measure cross-market exposure with privacy safeguards.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Reassign broader clusters or narrow the target.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-33. Sequential peeking
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A result crosses a threshold after repeated checks.
- **First evidence:** Review the preregistered monitoring schedule.
- **Candidate mechanism:** Unplanned repeated testing inflates false positive risk.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare actual looks with the planned stopping rule.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Do not declare a win; analyze under a valid design.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Statistician coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-34. Model drift
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Candidate ranking changes sharply in a new season.
- **First evidence:** Compare covariate distributions and calibration by time.
- **Candidate mechanism:** Seasonality or a pipeline error can change predicted demand.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Check data freshness and holdout calibration separately.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Revert to the approved rule pending review.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** ML operations coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-35. Feedback loop
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Frequent markdowns train the model on selected inventory.
- **First evidence:** Inspect historical policy and selection into each price.
- **Candidate mechanism:** The policy determines which counterfactuals are observed.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare randomized exploration with policy-selected days.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Do not treat policy data as unconfounded.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Pricing science coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-36. Model version skew
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Offline approved recommendation differs from served value.
- **First evidence:** Compare manifest model checksum with approval artifact.
- **Candidate mechanism:** Publisher may load a different artifact or transform.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Recompute the same decision with recorded inputs.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Block mismatched manifests and restore compatible version.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** ML platform coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-37. Late finance corrections
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A pilot win reverses after refunds arrive.
- **First evidence:** Plot cumulative contribution by maturity age.
- **Candidate mechanism:** Returns and chargebacks are delayed labels.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare equally mature cohorts across arms.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Delay final decision until the agreed window.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Finance coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-38. Order attribution ambiguity
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Reported treatment revenue exceeds ledger revenue.
- **First evidence:** Trace order-line allocation to product and market.
- **Candidate mechanism:** A multi-item order may be joined more than once.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Reconcile sum of attributed lines to ledger totals.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Hold outcome report until attribution balances.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Analytics engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-39. Margin floor violation
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A transaction falls below required contribution margin.
- **First evidence:** Compare actual fees and discounts with policy inputs.
- **Candidate mechanism:** An unmodeled promotion can stack with a valid shelf price.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Replay checkout with all active discounts.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Disable the conflicting promotion or withdraw SKU.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Merchant operations coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-40. Promotion stacking
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Displayed shelf price is compliant but paid price is not.
- **First evidence:** Trace coupon and loyalty deductions at checkout.
- **Candidate mechanism:** Several independently valid offers can combine unsafely.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Test complete offer composition against rules.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Block conflicting offer combination.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Commerce and legal coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-41. Legal policy uncertainty
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A pilot includes a newly restricted product.
- **First evidence:** Compare eligibility list with approved legal scope.
- **Candidate mechanism:** Catalog classification or requirements may have changed.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Obtain a jurisdiction-specific legal determination.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Exclude affected products pending approval.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Legal coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-42. Sensitive segment disparity
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Complaint rates rise for a customer group.
- **First evidence:** Compare exposure-adjusted complaints and uncertainty.
- **Candidate mechanism:** Price, accessibility or support-channel differences can all contribute.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Audit eligible exposure and treatment across approved groups.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Pause affected rollout and investigate customer impact.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Fairness reviewer coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-43. Small subgroup inference
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** One segment appears unaffected despite few observations.
- **First evidence:** Inspect group counts and interval width.
- **Candidate mechanism:** Low power can conceal meaningful harm.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Report uncertainty and missingness alongside rates.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Avoid declaring the guardrail passed from a null test.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Statistician coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-44. Cold-start product
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A new SKU has no prior price variation.
- **First evidence:** Inspect similar-product evidence and support overlap.
- **Candidate mechanism:** No local elasticity estimate may exist.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare baseline demand features without extrapolating price effect.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Use approved manual price until evidence exists.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Merchant coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-45. Price war feedback
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Competitor repricing follows repeated changes.
- **First evidence:** Inspect timestamps and external market context.
- **Candidate mechanism:** Competitors and the retailer can respond to common shocks.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Avoid interpreting correlation as a competitor reaction.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Apply approved movement limits and merchant review.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Category strategy coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-46. Demand spike outside forecast
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Event-driven traffic exceeds predicted units.
- **First evidence:** Compare campaign calendar, sessions and available stock.
- **Candidate mechanism:** A real demand shock or broken feature feed can look similar.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Cross-check independent traffic and campaign records.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Freeze changes if input integrity is uncertain.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Operations coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-47. Batch overrun
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Price manifest misses the daily cutoff.
- **First evidence:** Inspect queue time, input completeness and per-stage duration.
- **Candidate mechanism:** Data delay and insufficient compute have different remedies.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Trace a single batch through its stage timestamps.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Retain an approved still-valid price and investigate.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Data platform coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-48. Cache stampede
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A bulk price update causes read latency spikes.
- **First evidence:** Inspect invalidation rate and cache miss concurrency.
- **Candidate mechanism:** Simultaneous expiry can overload catalog backends.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare misses and backend saturation during publication.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Throttle publication and serve compliant cached values.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Commerce on-call coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-49. Rollback schema mismatch
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** An old manifest cannot load after deployment.
- **First evidence:** Inspect schema compatibility and rule version.
- **Candidate mechanism:** Restoring an artifact alone may not restore valid state.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Test rollback using current validation rules.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Withdraw affected offers if no compatible approved manifest exists.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Release engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-50. Audit log gap
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A customer dispute cannot be reconstructed.
- **First evidence:** Trace decision ID from display through order.
- **Candidate mechanism:** Missing joins or retention expiry can break evidence.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Replay a sample journey from immutable records.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Preserve available records and escalate rather than infer.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Governance coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-51. Incorrect sample window
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** An apparent treatment change is a holiday effect.
- **First evidence:** Compare matched calendar blocks and randomization timestamps.
- **Candidate mechanism:** Seasonality and staggered rollout can confound time comparisons.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Analyze concurrent randomized controls within blocks.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Do not use simple before-after attribution.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Experimentation lead coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-52. Unexpected negative margin
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** Higher volume coincides with lower contribution.
- **First evidence:** Decompose per-unit contribution and quantity by arm.
- **Candidate mechanism:** Fees, returns or product mix may overwhelm revenue.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Reconcile treatment and control cost components.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Stop expansion until finance validates the decomposition.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Finance coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-53. Unstable rules across regions
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** A price is valid in one market but invalid in another.
- **First evidence:** Inspect market-specific rule and currency identifiers.
- **Candidate mechanism:** Rules may be cached without the market key.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Replay identical SKU across two market contexts.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Quarantine incorrect scope and invalidate caches.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Catalog engineering coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.
### F-54. Inventory opportunity cost ignored
- **Provenance:** HYPOTHETICAL; this is a teaching diagnostic, not an observed incident.
- **Trigger / symptom:** One-day profit rises while later lost sales increase.
- **First evidence:** Inspect sell-through and replenishment over the full horizon.
- **Candidate mechanism:** A one-day optimizer can consume scarce stock too early.
- **Competing explanation:** A genuine demand or assortment change can produce a superficially similar aggregate signal.
- **Discriminating test:** Compare multi-day contribution and stockout timing.
- **Interpretation:** A test supporting the candidate mechanism narrows the investigation; it does not alone establish all downstream impact.
- **Containment:** Add an approved inventory-horizon constraint or defer.
- **Blast radius:** Limit investigation to affected product, market, rule version and time interval before considering wider rollback.
- **Responsible role:** Inventory science coordinates the investigation and records the decision.
- **Prevention:** Encode the relevant input, assignment or publication invariant as an owned automated check where feasible.
- **Verification:** Replay the affected product-day against the corrected contract, then reconcile the affected ledger and customer-facing state.
- **Limitation:** A replay checks the specified mechanism; it does not prove absence of every other cause.

## 13. Transferable Lesson
| Field | Lesson |
|---|---|
| Canonical pattern | causal-confounding: the policy chooses price using information that also predicts demand |
| Principle | Optimize a business outcome over feasible actions only after learning the action response with credible identification |
| Preconditions | Stable unit of assignment, observable mature outcomes, feasible safe price variation and sufficient support |
| Mechanism | Randomization breaks measured and unmeasured pre-assignment confounding in expectation, subject to design and implementation |
| Evidence | Constructed arithmetic illustrates a decision reversal; it does not demonstrate real-world causal impact |
| Avoid when | Price experiments are prohibited, exposure cannot be controlled, or interference invalidates the chosen assignment unit |
| Portability test | Can the organization assign a bounded action without contaminating control, and measure all relevant downstream costs? |
| Adaptation | In travel, perishable capacity and time-to-departure replace replenishable-stock assumptions; the decision horizon must change |
The principle is a proposed design hypothesis, not a universal finding established by a single constructed example.
Hard safety and legal constraints remain structural; the illustrative 10% movement limit is a local scenario choice.

## 14. Connections & Cross-References
### 14a. Related cases, theory, and cheat sheets
| Relationship | Target | Shared mechanism / boundary |
|---|---|---|
| Collection | [Pricing case index](INDEX.md) | Other pricing topics are planned, not validated comparisons |
| Methods | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Training-only preprocessing and leakage; does not identify causal effects |
| Operations | [Google SRE SLO workbook](https://sre.google/workbook/implementing-slos/) | User-centered service indicators; target here remains illustrative |
| Planned contrast | Airline and hotel revenue management | Perishable inventory and booking horizon change the optimization; no authored case is claimed |
No completed comparable case in this local catalog is asserted.

### 14b. What this case leaves open
| Question | Evidence gap | Proposed resolution | Priority / owner |
|---|---|---|---|
| What is causal demand response in the pilot range? | No randomized data or retailer observations | Approved price-grid experiment with support checks | Before optimization / pricing science |
| How much substitution crosses clusters? | No actual basket or market journey data | Inventory-pool mapping and interference study | Before randomization / experimentation |
| What is the appropriate inventory horizon? | No replenishment or opportunity-cost evidence | Multi-day simulation validated against inventory records | Before scarce-stock rollout / inventory |
| Are legal and customer safeguards adequate? | No jurisdiction or approved policy defined | Legal determination and end-to-end customer journey review | Before exposure / legal and product |

### 14c. Intense interview pack
Scoring for each question: 0 incorrect; 1 partial; 2 correct with explicit assumptions; 3 correct and defensible under the follow-up.
Questions are prompts for a design discussion, not claims of actual interview or project experience.
#### Q01 — Framing
- **Candidate prompt:** What decision and baseline are being compared?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** The action is one approved SKU-market-day public price; compare constrained policy with the fixed approved rule.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A manual review can remain the decision if evidence is weak.
- **Follow-up:** What if no candidate satisfies current rules?
- **Follow-up answer:** Publish no invalid price; escalate and use only a rule-validated safe state.
- **Red flag:** Calling the task pure demand prediction omits the action and constraints.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 1–4.
#### Q02 — Objective
- **Candidate prompt:** Why not maximize conversion?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Conversion ignores unit margin, refunds, fulfillment and inventory opportunity cost, so it can choose a lower-contribution price.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** Revenue may be a secondary metric if finance approves contribution as primary.
- **Follow-up:** What if fixed costs differ by price?
- **Follow-up answer:** Include price-dependent costs in the objective before comparing.
- **Red flag:** Equating more orders with more profit is wrong.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 3 and 6.
#### Q03 — Calculation
- **Candidate prompt:** Calculate contribution at price 100 with 80 units, cost 60 and variable cost 5.
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Unit contribution is 100−60−5 = 35 currency per unit; 80 × 35 = 2,800 currency per day.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A receipt-level sum is equivalent under the stated constant costs.
- **Follow-up:** What if cost becomes 70?
- **Follow-up answer:** The same quantity gives 80 × 25 = 2,000.
- **Red flag:** Subtracting costs once per day rather than per unit is dimensionally incorrect.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 6.
#### Q04 — Sensitivity
- **Candidate prompt:** Which candidate wins when unit cost becomes 70, holding the given demand fixed?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Contributions are 100×15 = 1,500, 80×25 = 2,000, and 60×35 = 2,100; choose 110.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** Choose no change if those assumed demands lack support.
- **Follow-up:** What if only 50 units can be sold with original costs?
- **Follow-up answer:** Assuming deterministic demand, contributions become 1,250, 1,750 and 2,250.
- **Red flag:** Treating that toy cap as expected stochastic sales is invalid.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 6.
#### Q05 — Causal inference
- **Candidate prompt:** Why is regressing sales on price alone unreliable?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Merchants may lower price during campaigns or raise it when demand is already expected to be high; these factors also affect sales.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A well-justified observational design can supplement, but does not automatically replace, randomization.
- **Follow-up:** What if randomized treatment cannot be assigned?
- **Follow-up answer:** Retain baseline or use a carefully justified quasi-experiment with explicit untestable assumptions.
- **Red flag:** A high predictive fit does not establish the causal price effect.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 5 and 9.
#### Q06 — Data grain
- **Candidate prompt:** How do you prevent stockout days from masquerading as zero demand?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Use product-market-day panels, stock availability timestamps and exposure data; fulfilled sales are censored when stock runs out.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A censored-demand model is possible if its assumptions are defended.
- **Follow-up:** What if the product was not listed?
- **Follow-up answer:** Exclude that period from the eligible opportunity denominator according to preassignment rules.
- **Red flag:** Dropping all low-stock days after treatment can create selection bias.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 5a and 9.
#### Q07 — Leakage
- **Candidate prompt:** Which timestamp determines whether a cost feature is usable?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** The feature must have been available to the decision system by the cutoff, not merely have an earlier business effective date.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** An immutable as-known snapshot or bitemporal join can implement that rule.
- **Follow-up:** What if late invoices revise past costs?
- **Follow-up answer:** Use as-known costs for prediction and mature finance data for outcome reconciliation.
- **Red flag:** Using the latest overwritten cost creates hindsight leakage.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 5d.
#### Q08 — Experiments
- **Candidate prompt:** What would you randomize for a public-price experiment?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Randomize sufficiently independent market-category or inventory-pool clusters in blocks, with public prices consistent within a cluster.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** Another unit is valid if it avoids cross-arm price exposure and shared-stock interference.
- **Follow-up:** What if customers shop across markets?
- **Follow-up answer:** Measure contamination and enlarge clusters or narrow scope if interference is material.
- **Red flag:** Independent session prices can break public consistency and contaminate controls.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 9.
#### Q09 — Inference
- **Candidate prompt:** Why not use each product-day as an independent observation?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Repeated days and products within a randomized cluster are correlated; the independent assignment unit governs uncertainty.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A cluster-aware model or suitable randomization inference can be used when assumptions hold.
- **Follow-up:** What if there are few clusters?
- **Follow-up answer:** Use a method suited to the finite assignment design or redesign; report limited precision.
- **Red flag:** A tiny standard error from pseudoreplication is misleading.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 9.
#### Q10 — Sizing
- **Candidate prompt:** How much batch work is implied by 10,000 SKUs and 11 prices?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** There are 110,000 candidate evaluations per day, or about 15.28 per second averaged over two hours.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** Actual sizing requires measured per-evaluation cost, dependency traffic and peak behavior.
- **Follow-up:** What if all work arrives in the last ten minutes?
- **Follow-up answer:** The average becomes about 183.33 evaluations per second in that interval, before concurrency or overhead.
- **Red flag:** Calling the daily average a measured peak capacity is wrong.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 8d.
#### Q11 — Reliability
- **Candidate prompt:** A price publish times out; should the client immediately publish a new decision ID?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** First reconcile whether the original write succeeded; retry with the same idempotency key under the bounded retry policy.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A read-after-write status endpoint can support reconciliation.
- **Follow-up:** What if the catalog is unavailable and the prior price expired?
- **Follow-up answer:** Withdraw the affected offer and escalate rather than serve an unvalidated price.
- **Red flag:** A new ID on every retry can duplicate effects.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 8.
#### Q12 — Debugging
- **Candidate prompt:** Sales fall after a markdown; what would you inspect first?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Compare in-stock time, exposure, actual displayed price and campaign context; stockout, telemetry and true demand changes are competing explanations.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A randomized contrast helps when assignment and exposure are intact.
- **Follow-up:** What if stockout times are missing?
- **Follow-up answer:** Treat demand as unidentified for that interval and investigate inventory telemetry.
- **Red flag:** Declaring negative elasticity from fulfilled sales alone ignores censoring.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 5 and 12.
#### Q13 — Governance
- **Candidate prompt:** How does rollback handle committed orders and changed rules?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Stop new publications, validate an approved compatible manifest against current rules, restore and reconcile surfaces, and honor applicable committed order terms.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** Withdraw affected offers if no valid manifest exists.
- **Follow-up:** What if the old schema is no longer accepted?
- **Follow-up answer:** Use a compatible approved artifact or withdraw; never force an invalid old manifest.
- **Red flag:** Rewriting past order prices or bypassing current constraints is unsafe.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 8 and 10.
#### Q14 — Evaluation
- **Candidate prompt:** What denominator should the primary experiment metric use?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** All eligible randomized cluster-days, including zero-sale days, for both assigned arms; report mature contribution divided by that denominator.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A total contribution estimand can be useful if prespecified and cluster exposure is appropriately handled.
- **Follow-up:** What if treatment creates stockouts?
- **Follow-up answer:** Keep those cluster-days in the intention-to-treat outcome and study stockouts as a mechanism and guardrail.
- **Red flag:** Restricting to purchasers or available days after assignment induces selection.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 9.
#### Q15 — Transfer
- **Candidate prompt:** How would the design change for perishable hotel rooms?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** Model remaining room-night inventory and booking horizon; unused rooms expire, and capacity constraints are time-specific.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** A fixed-rule comparator and bounded randomized policy remain useful if legally and operationally feasible.
- **Follow-up:** What if different booking channels share capacity?
- **Follow-up answer:** Randomize or account for shared-capacity interference at a suitable unit.
- **Red flag:** Copying replenishable inventory assumptions unmodified is not justified.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Section 13.
#### Q16 — Uncertainty
- **Candidate prompt:** Does the toy contribution ranking prove that price 100 is optimal in production?
- **Given facts:** Use the illustrative case assumptions; no real company results are supplied.
- **Clarify:** State the decision unit, relevant constraints and evidence status before answering.
- **Strong answer:** No; it is conditional arithmetic using invented demand and costs, with no real data, identification, uncertainty or implementation evidence.
- **Reasoning check:** Tie the result to the stated population, time window and units where applicable.
- **Acceptable alternative:** It is still useful as an auditable example of objective and sensitivity.
- **Follow-up:** What would change the decision?
- **Follow-up answer:** Credible causal estimates, uncertainty, mature costs, inventory dynamics and cleared constraints could change the ranking.
- **Red flag:** Presenting the illustrative 2,800 as actual observed profit fabricates evidence.
- **Scoring:** Apply the 0–3 rubric above; full credit requires defending the boundary.
- **Answer support:** Sections 6, 7 and 11.

## 15. Quick Reference Card
| Field | Value |
|---|---|
| Domain | Illustrative ecommerce retail, one product-market-day |
| Problem type | Constrained public price selection under demand and inventory uncertainty |
| Method | Enumerate valid candidates; compare expected contribution with a fixed approved-price baseline |
| Primary result | Not measured; example alone favors 100 under stated assumptions |
| Key lesson | A causal demand response and mature business objective are needed before trusting optimized prices |
| Tags | causal-confounding; evaluation-mismatch; model-monitoring |
| Case ID / version / as-of | CASE-01-06-001 / draft 1 / 2026-11-07 |
| Evidence level | not-measured / illustrative; no business dataset or run |
| Implementation stage | proposed / draft-unreviewed; last reviewed null |
| Baseline / primary metric | Approved fixed rule; measure profit and customer guardrails in a governed randomized test |
| Key limitation | Price bounds, inventory, customer treatment and legal approval constrain the proposed test; no individual pricing or observed outcome |
| Review trigger / owner | Changes to eligibility, constraints, experiment, or finance definitions / pricing science lead |
