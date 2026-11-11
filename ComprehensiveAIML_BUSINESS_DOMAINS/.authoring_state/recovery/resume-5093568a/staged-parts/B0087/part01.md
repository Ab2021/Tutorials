---
title: "Shelf Availability Metric Card"
topic: "Shelf Availability Metric Card"
collection: "retail-01"
file_id: "02"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
status: "draft"
version: "2.0"
reviewed: null
review_trigger: "Audit instrument, assortment, replenishment cadence, or weighting policy changes"
notation_style: "plain-text"
prerequisites: ["basic rates, timestamps, and cohort definitions"]
pattern_tags: ["retail-operations", "point-in-time"]
source_status: "OPEN"
owner_role: "retail analytics practitioner"
source_index: "INDEX.md"
---

# Shelf Availability Metric Card

> **Purpose:** Specify, compute, and defend one number for on-shelf availability (OSA) so that a replenishment or store-execution decision can be taken on it.
> **Prerequisites:** Ratio literacy, event-time versus processing-time discipline, and willingness to audit a sample by hand.
> **Key outcome:** Choose the right weighting for the decision at hand, compute OSA with a defensible denominator, attach an uncertainty interval, and refuse the action when the interval straddles the decision threshold.
> **This file does NOT cover:** Retailer-specific systems, planogram authoring rules owned by merchandising, or any measured deployment result.
> **Level:** core — a practitioner can run this end to end after one supervised cycle.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED arithmetic on ILLUSTRATIVE inputs; every constructed quantity is labelled at first use.
> **Retrieval cue:** If a previously stable OSA number moves after an assortment change, a backfill, or a scanner upgrade, start at Section 8.

## 1. File Header

This card exists because "shelf availability" is one of the few retail metrics that four different teams believe they already agree on, while in practice they compute four different numbers and then argue about the gap at a steering meeting. The card pins one definition per decision, states the denominator explicitly, and gives the audit procedure that keeps the definition honest.

| Field | Value for this card |
|---|---|
| Metric family | Availability and in-stock; a dimensionless coverage fraction, not a count |
| Metric name | OSA — on-shelf availability, also called shelf in-stock rate |
| Decision supported | Whether to fund a replenishment or shelf-execution change in a given assortment class |
| Decision owner | Replenishment lead, with store operations as the executing party |
| Reporting owner | Retail analytics; this card is the definition of record |
| Decision grain | SKU-store-observation, rolled to SKU-store-day then to class and chain |
| Time grain of the estimate | Audit round (typically weekly) and rolling four rounds |
| Cadence | Weekly measurement, monthly review, quarterly definition review |
| Unit of the numerator | An observation whose shelf slot held at least one sellable unit |
| Unit of the denominator | An observation that was eligible to be counted under the stated rule |
| Valid range | 0 to 1 inclusive, in practice 0.55 to 0.99 for grocery dry goods |
| Headline aggregation | Sales-weighted across velocity classes; unweighted only as a diagnostic |
| Precision target | Half-width of 2 percentage points chain-wide at 95% confidence |
| Action threshold | A class-level drop of 2.5 points or more, sustained two rounds |
| Known exclusions | Discontinued items, recalled items, items with no planogram slot, seasonal delists |
| Known contamination | Phantom inventory, capacity censoring, mid-round restocks, post-round record edits |
| Source status | OPEN — no vendor or retailer system behaviour is asserted here |
| Evidence class | DERIVED: the formulas are standard; the numbers are constructed to be checkable |

The header deliberately separates the decision owner from the reporting owner. When one person owns both, the temptation to tune the weighting until the number supports the already-chosen action is structural rather than personal, and no amount of review discipline fixes it.

Three properties make this metric fragile and are worth stating before any formula appears.

First, the denominator is a policy choice, not a fact. Whether a facing with four units behind one unit is "available" depends on whether you define availability at the slot level or the unit level, and the two answers diverge most sharply exactly where the business cares: high-velocity items that sell through between visits.

Second, the measurement instrument is a human with a handheld device, and that instrument has a measurable false-positive and false-negative rate. Any OSA series built from scans inherits those rates, and the bias is not symmetric.

Third, availability is a rate over a population that changes as the assortment changes. A chain that delists its slowest 12 per cent of SKUs will see OSA rise without any operational improvement, because the hard cases left the denominator.

**What "good" looks like for this card.** A reader should be able to take the four formulas in Section 4, apply them to their own audit extract, and get a number they can defend line by line to a finance partner. They should also be able to say precisely which decision that number does and does not support.

**What this card refuses to do.** It does not convert OSA into a currency figure without an explicit elasticity assumption, because the conversion dominates the answer. Section 5 shows the same availability gap valued at three different recovery fractions, and the annual range spans a factor of about 2.6.

**Versioning rule.** Any change to the denominator rule, the weight vector, the audit instrument, or the rounding convention is a new version of this card, not a silent edit. Comparisons across versions are invalid unless the earlier series is recomputed.

## 2. Concept Map / Mental Model

The mental model is a five-stage pipeline with two side channels that exist purely to catch bad input. Nothing downstream can repair a stage that was wrong upstream, which is why the quarantine branch is drawn as a terminal state rather than a retry loop.

```text
  [scan / count event]
          |
          v
  (1) identity and time validation ---- fail ----> [quarantine: reason-coded]
          |
          v
  (2) eligibility filter  (assortment, planogram slot, not delisted, not recalled)
          |
          v
  (3) denominator construction  (slot-minute | slot-visit | SKU-day)
          |
          v
  (4) weighting  (unit | time | SKU | sales)  <---- weight vector from finance
          |
          v
  (5) estimate + interval  ---- interval straddles threshold ----> [abstain, no action]
          |
          v
     [action: replenishment rule change | shelf-capacity change | audit redesign]
```

Read the diagram left to right as a chain of custody. Stage 1 decides whether a row in the extract is a fact at all. Stage 2 decides whether the fact is in scope for this decision. Stage 3 is where most disputes actually live, because two teams can agree on stages 1, 2, 4, and 5 and still disagree by six points.

A useful way to hold the whole thing in mind is to ask three questions in order, each of which maps to a stage.

- *What was counted?* — an observation, at a stated grain, with a stated eligibility rule.
- *What was it divided by?* — a denominator whose construction rule is written down before the data arrives.
- *What was it weighted by?* — a vector that encodes whose loss matters, and which must be supplied by the decision owner rather than the analyst.

The concept map also encodes a design principle: the quarantine branch is not an error handler, it is a measurement. A rising quarantine rate is the earliest signal that a scanner firmware change, a planogram reset, or an assortment refresh has broken comparability. In the illustrative audit in Section 5 the quarantine branch absorbs 4.2 per cent of raw rows, and that rate is reported alongside OSA on the same dashboard.

**Two-grain discipline.** Availability is estimated at the observation grain and consumed at the class grain. Mixing them in one table is the single most common source of the "your number does not match mine" conversation. The rule on this card is that every published table states its grain in the column header, and no table contains two grains.

**Baseline discipline.** OSA has no natural zero, so a level is meaningless without a comparator. The comparator on this card is the same class, same weight vector, same audit instrument, trailing four rounds, with the current round excluded. Frozen baselines are stored as immutable extracts, not recomputed from current data.

Takeaway: the pipeline is a chain of custody; the weight vector is a policy input; the quarantine rate is a first-class metric.

## 3. Foundational Definitions

Definitions here are operational, meaning each one names the field or rule that implements it. A definition that cannot be traced to a column is a slogan.

| Term | Definition used on this card | Units / domain |
|---|---|---|
| Observation | One audit record: one SKU, one store, one shelf slot, one visit, one timestamp | categorical key plus event time |
| Eligible observation | Observation whose SKU was on the planogram for that store on that date and was not delisted, recalled, or marked do-not-order | boolean |
| Present | Eligible observation where the shelf slot held at least one sellable unit at the moment of the count | boolean |
| OSA (slot-visit) | Present observations divided by eligible observations | fraction, 0 to 1 |
| OOS | The complement: eligible observations that were not present | fraction, 0 to 1 |
| Exposure | Eligible shelf-minutes, the time integral of an empty slot over the observation window | slot-minutes |
| OSA (time-weighted) | One minus exposure divided by total eligible slot-minutes | fraction, 0 to 1 |
| Velocity class | Assigned band from trailing 13-week unit sales, fixed for the audit year | ordered category A to D |
| Weight vector | Nonnegative weights, one per velocity class, summing to one | dimensionless, sums to 1 |
| Audit round | One complete pass over the sampled stores and SKUs | categorical, weekly |
| Design effect | Variance inflation from measuring clusters rather than independent draws | dimensionless, at least 1 |
| Phantom inventory | System on-hand above zero while the shelf slot is empty | boolean per observation |
| Capacity censoring | Slot capacity below demand over the replenishment cycle, so stock-out occurs despite correct ordering | boolean per SKU-store |
| Days of supply | On-hand or capacity units divided by mean daily demand | days |
| Fill rate | Fraction of demand met from shelf without backorder or substitution | fraction, 0 to 1 |
| Service level | Probability of not stocking out during a replenishment cycle | fraction, 0 to 1 |
| Recovery fraction | Share of unavailable demand that converts to a later sale rather than being lost | fraction, 0 to 1 |
| Effective time | The timestamp at which a fact is permitted to influence a decision | timestamp |
| Maturity | Elapsed time after which an outcome is treated as final | duration |

Three of these definitions carry most of the argumentative weight, so they get a paragraph each.

**Eligible observation.** The eligibility rule is where assortment changes enter the metric. If a delisted SKU stays in the denominator, OSA falls through no operational fault. If a newly listed SKU enters the denominator with zero stock on its first day, OSA falls for a reason that is arguably a launch-execution failure but not a replenishment failure. The rule on this card is that eligibility is evaluated at the observation's effective date against the planogram version in force on that date, and both the planogram version and the eligibility decision are stored on the row.

**Exposure.** Exposure is the integral of emptiness over time. A slot empty for two hours and a slot empty for two days are both one OOS observation under slot-visit counting, but they represent a 24-fold difference in lost selling opportunity. Time-weighted OSA is the metric that respects this, and it is the one to use when the decision is about labour allocation rather than about range compliance.

**Recovery fraction.** The recovery fraction converts availability into money, and it is the assumption most often smuggled in without a label. In the illustrative case in Section 5, the same weekly availability gap is worth 5,981 in one scenario and 15,551 in another, purely from this parameter. Any currency figure printed without a stated recovery fraction is unusable.

**Grain words to avoid.** "SKU-level OSA" and "store-level OSA" are ambiguous because they do not say whether the SKU is weighted equally across stores or by store volume. This card uses only the four named grains — slot-visit, exposure-weighted, SKU-day, and class-rolled — and any other phrasing is a defect in the request.

**Notation.** Symbols are defined once and reused: `N` for the count of eligible observations, `P` for present observations, `a` for the availability fraction, `u = 1 - a` for unavailability, `w_c` for the weight of velocity class `c`, `n_c` for eligible observations in class `c`. Where a symbol is used before its formal definition, the meaning is given inline.

Takeaway: eligibility and exposure are the two definitions that decide the number; recovery fraction decides the money.
