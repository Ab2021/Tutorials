# Master-Data Drift, Worked

> Module 02 — The Supply Chain Data Landscape
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01, 02.02, 02.03

---

## 1. Why this topic exists

A forecasting model does not consume a company. It consumes a table with a key column, a time
column, and a measure column. If the key is stable and the measure is expressed in one unit, the
model can learn a pattern. Master-data drift breaks one of those two conditions, usually without
changing the schema, and often without changing the key. The table still loads, the row counts
still reconcile, and the pattern the model learns is a pattern in the record rather than in the
business.

Every forecast is a comparison across time, and the comparison is only valid if the thing being
counted is the same at both ends. Three things can stop it being the same:

1. **The thing changed.** A pot of yoghurt is reduced from 150 g to 120 g under the same code. Unit
   demand is flat; tonnage demand falls by 20%. A forecast in units is correct, and a capacity plan
   in tonnes built from it is 20% too high.
2. **The name changed.** A case of 12 is replaced by a case of 10 and a new Global Trade Item Number
   (GTIN) is allocated. The old series stops and the new one starts at zero months of history. The
   model now has two products, one of which is new, and neither of which has the history that
   actually exists.
3. **The meaning changed.** A lifecycle status value `A` meant "sellable" until a quality programme
   redefined it as "orderable, possibly on hold". No schema changed, no key changed, and every
   report that treats `A` as available stock has overstated availability from that date.

The three cases look different and fail the same way. In each, the column is still populated, the
data type is still correct, and every individual row is valid. What has changed is the relationship
between a row written last year and a row written this year.

This is why the topic sits here, after the data-quality dimensions of Topic 08. Completeness,
validity, consistency, timeliness, uniqueness, and accuracy are properties of a record or snapshot.
Drift is a property of the *sequence* of snapshots, and a dataset can score perfectly on all six and
still be unusable as history: a pack change produces a file that is complete, valid, unique, freshly
loaded, and compares two different physical objects.

The commercial consequence is asymmetric. A duplicate item inflates counts and is found when two
planners order the same thing. Drift does not inflate counts, does not create a visible duplicate,
and does not break a referential check; it produces a series that is plausible at every point and
wrong in its transitions, and the error surfaces months later as a forecast that was "accurate" and
a supply plan that was not.

> **Note:** drift is not the same as bad data quality, and the remedies are different. Cleaning a
> record fixes a wrong value. Restating a history fixes a change of meaning. A team that responds to
> a drift symptom with a data-quality cleanse will improve the record and leave the series broken.

The topic is the module's worked case. The systems of record from 02.01 determine where the change
was recorded; the master-data model from 02.02 determines what the item is; the transaction data
from 02.03 determines what the key points at; the quality dimensions from 02.08 explain why no check
fired. What is new here is time: two clocks instead of one, and a reconstruction that has to be
deliberate.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a described master-data change into one of the five drift types — attribute,
   structural, identity, supersession, semantic — and **state** which field and which unit the
   change propagates through.
2. **Distinguish** valid time from transaction time, and **explain** why a history rebuilt from
   current master data is not the history the business saw.
3. **Detect** a level shift caused by a pack-size change in a monthly series, and **separate** the
   unit artifact from a genuine change in demand.
4. **Convert** a history across a pack change using an effective-dated factor, and **prove** the
   conversion with a control total in an invariant unit.
5. **Reconstruct** a comparable series across a one-to-many supersession by allocating observed
   share and carrying the residual as a separate, owned assumption.
6. **Quantify** the demand-plan error produced by a drifted series at three layers — unit mix,
   model specification, and downstream conversion — and **state** which layers are physical.
7. **Design** a change-notification, effective-dating, and pre-run gate control that stops a model
   run before it consumes an unrestated series, and **write** the three bitemporal queries — as-was,
   as-is, as-should-have-been — for a named planning decision.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is a single item code running across three years, with five different
kinds of change landing on it. The second is a plane whose two axes are the two clocks that govern
every master record.

```text
   ONE ITEM CODE, THIRTY-SIX MONTHS
   M1 -------------------------------------------------------> M36

   [1] ATTRIBUTE      weight 150 g -> 120 g            key unchanged
   [2] STRUCTURAL     case 12 -> 10                    key may survive
   [3] IDENTITY       new GTIN for the 10-pack         key changes
   [4] SUPERSESSION   old item retired, successors     key changes + map
   [5] SEMANTIC       "active" now means orderable     no key change

   WHAT THE SERIES SHOWS IN EACH CASE
   [1] level unchanged, cost per unit wrong
   [2] step of +20% in cases, no step in bottles
   [3] one series ends, one begins at zero
   [4] one series ends, N begin, no shared scale
   [5] level unchanged, every filter silently wrong
```

The five are not a taxonomy for its own sake. Each one breaks a different assumption that a time
series model makes, and each needs a different repair. Treating them as one phenomenon called
"data change" produces the wrong fix: a new code where a conversion was needed, or a conversion
where a mapping was needed.

### 3.1 The five drifts and what the model assumes

| Drift type | What changes | What the key does | What the model assumes | What actually arrives |
|---|---|---|---|---|
| Attribute | Weight, dimensions, shelf life, temperature class | Usually unchanged | The unit is the same physical quantity | Same count, different mass or volume |
| Structural | Case pack, inner pack, pallet configuration, TI/HI | May be retained, may not | The case is a constant quantity | Same word, different count |
| Identity | A new GTIN is allocated for a changed pack | New key, old key retired | Each key is one product for its life | History splits into two short series |
| Supersession | Old item replaced by a successor, possibly a different item | New key plus a mapping | The successor's own history is sufficient | No history, or history at the wrong scale |
| Semantic | A field's meaning is redefined without a schema change | Unchanged | A code value means the same thing tomorrow | Same code, different population |

**Structural drift hides inside a correct-looking number.** A case of 12 and a case of 10 are both
"one case", and the count is valid. Nothing in the warehouse management system (WMS) flags it,
because the case is the handling unit and the handling unit did not change. The error exists only
in the ratio between cases and the physical quantity the rest of the business uses — bottles,
litres, kilograms, pallet positions, excise duty.

**Semantic drift has no data event at all.** No new GTIN, no new key, no effective date, often no
change record. A definition is agreed in a meeting, a validation rule is relaxed, a status value
starts serving a second purpose, and from that date the field means something else. It is found by
reading a data dictionary's version history, or by noticing that a metric moved on a date when
nothing physical happened.

### 3.2 The two clocks

Every master record has two times, and conflating them is the root of most reconstruction errors.

- **Valid time** is when the fact was true in the world. The 10-pack was the sellable case from
  1 January.
- **Transaction time** is when the database recorded that fact. The change was keyed on
  14 January, after the first production run.

```text
                        VALID TIME  (when it was true in the world)
                        past ------------------------------> now
                 +-----------------------------+---------------------+
   TRANSACTION   |  (A) AS-WAS                 |  (C) AS-IS          |
   TIME          |  what the plan saw on       |  what we believe    |
   (when it was  |  31 March, including the    |  now, for today     |
    recorded)    |  error and the late change  |                     |
                 +-----------------------------+---------------------+
                 |  (B) AS-SHOULD-HAVE-BEEN    |  (D) CURRENT MASTER |
                 |  restated history: valid    |  the master table   |
                 |  then, recorded later       |  as it stands today |
                 +-----------------------------+---------------------+
```

Four cells, four questions. Cell A reproduces the forecast the business actually ran. Cell B is what
you need to backtest honestly. Cell C is a current-state report. Cell D is one row per item and
cannot answer a question about the past. A history rebuilt from current master data is a **cell D
join**: every transaction attributed to the item as defined today, internally consistent, and not
the history the business saw. A planner told on 10 January that the item was a 12-pack decided on
that basis; restating the past to say it was always a 10-pack erases the decision.

### 3.3 The silent failure loop

```text
   pack change, key retained or history concatenated
                        |
                        v
         history mixes two physical quantities
              /                       \
             v                         v
   model fits the step as trend   case KPI compares
   (level or slope absorbs it)    cases with cases
              \                       /
               v                     v
          plan passes the accuracy gate
                        |
                        v
   execution converts cases at the old factor
                        |
                        v
   supply 20-30% above requirement, months later
```

**Property 1 — The model is not the weak point; the unit boundary is.** A flexible model will absorb
a one-time step into its level or trend and report an acceptable fit. The error does not disappear.
It moves into the parameter that is then extrapolated forward. The more flexible the model, the
more completely the drift is hidden, and the more confidently the resulting plan is wrong.

**Property 2 — Every check that could catch it is computed inside the drifted unit.** Forecast
accuracy compares a forecast in cases with an actual in cases. Service level counts order lines.
Inventory value uses standard cost per case, which finance may or may not have restated. As long as
numerator and denominator share the same unit, the 20% cancels, and the metric reports on the model
rather than on the physical world. The one check that cannot cancel is a control total in an
invariant unit: bottles, kilograms, litres, or base eaches.

### 3.4 The reconstruction decision tree

```text
   key changed? --no--> unit changed? --no--> attribute or semantic: no series surgery
      |                              +--yes--> structural: convert, keep key
      +--yes--> dated successor map? --no--> split; like-model the new series
                                   +--yes--> 1:1 with a known factor?
                                              +--yes--> convert and join
                                              +--no --> allocate by observed
                                                        share; carry residual
```

The order matters. Analysts start at the bottom branch, because supersession is the visible event,
and skip the two above it, which are quieter and more common.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Master-data drift | Change in what a master record means over time without a change in its schema, usually without a change in its key | Used to mean data decay or staleness | None |
| Attribute change | A change to a non-identifying property such as weight, dimensions, shelf life, or temperature class | Treated as a labelling change | Grams, millimetres, days, degrees Celsius |
| Structural change | A change to the packaging structure: case pack, inner pack, pallet configuration, TI/HI | Confused with an attribute change | Count per level |
| Identity change | A change of identifier for a changed product, typically a new GTIN | Assumed to be automatic and complete | GTIN-8/12/13/14 |
| Supersession | The replacement of an item by one or more successors, with a dated mapping and a quantity relationship | Used loosely to mean "discontinued" | Item-to-item map plus factor |
| Semantic change | A redefinition of what a field or code value means, with no schema or key change | Invisible in most data catalogues | None |
| Item master | The record holding identity, attributes, units of measure, and lifecycle for an item | Assumed to be one system and one row per product | Record |
| Stock keeping unit (SKU) | The internal identifier for an item as managed in a channel or location | Used as a synonym for product | Internal alphanumeric |
| GTIN | Global Trade Item Number: a globally unique identifier for a trade item at a specified packaging level | Used without a packaging-level qualifier | 8, 12, 13, or 14 digits |
| GTIN reuse | Allocating an existing GTIN to a materially different product | Confused with a legitimate re-listing | Forbidden practice |
| Case pack | The number of consumer units in a case, also called the case quantity | Used interchangeably with "pack size" | Consumer units per case |
| Unit of measure (UOM) | The unit in which a quantity is expressed | Stored implicitly rather than with the fact | Code list |
| Conversion factor | The multiplier between two units, for example 12 bottles per case | Keyed once and never re-verified | Ratio |
| Effective-dated factor | A conversion factor with a valid-from and valid-to date, so a historical quantity is converted at the factor in force at the time | Overwritten in place | Ratio plus date range |
| Valid time | The time interval during which a fact was true in the world | Confused with the date of entry | Timestamp interval |
| Transaction time | The time at which a fact was recorded in the database | Assumed to equal valid time | Timestamp |
| Bitemporal | A record design carrying both valid time and transaction time | Used to mean "has a date field" | Two timestamp intervals |
| As-of query | A query that reconstructs the database as it was known at a past transaction time | Confused with a query about the past | Transaction-time predicate |
| As-at query | A query that reconstructs what was true at a past valid time, using current knowledge | Confused with an as-of query | Valid-time predicate |
| Effectivity date | The date from which a master record version applies | Backdated without a change record | Date |
| Supersession chain | A sequence of items replacing one another, A to B to C, possibly with branches | Assumed to be linear | Directed graph |
| One-to-one replacement | One predecessor replaced by exactly one successor, with a known factor | Assumed to be the normal case | Cardinality 1:1 |
| One-to-many replacement | One predecessor replaced by several successors | Split by guess rather than observed share | Cardinality 1:n |
| Many-to-one replacement | Several predecessors consolidated into one successor | History summed without checking overlap | Cardinality n:1 |
| Data realignment | Writing predecessor demand into the successor's history without altering the raw records | Confused with deletion or overwrite | Derived series |
| Restatement | Recomputing a historical series in a consistent unit or definition, with the rule recorded | Done silently, without a version | Derived series plus rule |
| Comparable series | A series in which every observation is expressed in the same physical unit and definition | Assumed to be any series with one key | Invariant unit |
| Equivalent unit (EQ) | A normalised unit that removes pack-size differences, typically volume or a standard pack | Used without stating the standard | Litres, kilograms, or standard cases |
| Level shift | A one-time change in the mean of a series | Read as trend by an unfitted model | Measure per period |
| Structural break | A change in the data-generating process at a point in time | Detected, then ignored | Date plus model parameter |
| Intervention | An explicit model term for a known event, such as a step or pulse at a date | Omitted because "the model should learn it" | Regressor |
| Phase-in / phase-out | The ramp period during which a new item gains volume and an old item loses it | Treated as a step change on one date | Proportion per period |
| Like modelling | Forecasting a new item from the history of a similar or predecessor item | Applied with a default weight of 100% | Reference item plus weight |
| Residual (replacement) | The part of post-launch volume that the allocated history does not explain | Forced into a successor series | Physical units |
| Cross-reference table | The dated mapping between retired and current identifiers | Assumed to be complete | Table |
| Change notification | A message to downstream consumers that a master record will change, with the effective date | Sent after the change is live | Event plus date |
| Change log | An auditable record of who changed which field, when, and with what approval | Kept for 90 days and then purged | Table with transaction time |
| Cannibalisation | Volume taken by a new item from an existing item rather than from the market | Confused with replacement | Units per period |

Three of these carry most of the analytical weight, so they get a paragraph each.

**Valid time versus transaction time.** Valid time answers "when was this true"; transaction time
answers "when did we know". The pack changed on 1 January and was keyed on 14 January. An as-of
query at 10 January must return the 12-pack, because that is what the business knew. An as-at query
for 10 January, using today's knowledge, may return the 10-pack if the change was backdated. Both
are correct answers to different questions, and a report that does not say which it used cannot be
reconciled with one that used the other.

**Restatement versus realignment.** Restatement recomputes history in a consistent unit and records
the rule. Realignment writes predecessor demand into a successor's series and leaves the raw rows
untouched. Realignment is the vendor term for a derived view; restatement is the analytical act.
If the raw rows survive, the restatement can be reversed and re-derived, which is the property an
audit needs.

**Identity change versus supersession.** An identity change is the same product under a new number.
A supersession is a replacement relationship, and the successor may be a different product with a
different formulation, size, or use. The first needs a mapping; the second needs a mapping, a
factor, and often an allocation rule. Treating the second as the first is how a reformulated product
inherits the old demand curve.

## 5. The mechanics, step by step

This section follows one change from the moment it is proposed to the moment a model is allowed to
consume it. The sequence is written as ten steps because the failure is almost always a skipped
step rather than a wrong calculation.

### Phase A — Detect

**Step 1. Classify the change against the five types.**

- Artifact: a change record with a class, a reason code, and the fields affected.
- Key branch: **one change or two?** A pack reduction usually carries a structural change (12 to
  10), an identity change (new GTIN), and often a semantic change (the planning item now means "the
  10-pack family"). Classify each separately; the repairs differ.
- Failure if skipped: recorded as "pack size updated", no GTIN requested, and the retailer still
  holds a 12-pack description against a 10-pack number.

**Step 2. Date it on both clocks.**

- Artifact: a valid-from date, a valid-to date for the prior version, and the transaction timestamp.
- Key branch: **was it backdated?** If production changed on 1 January and the master was keyed on
  14 January, every transaction in between used the new physical pack against the old record. That
  window is a known, bounded defect and must be recorded rather than discovered.
- Failure if skipped: the effective date is set to the entry date, and the two-week window disappears
  from every reconstruction.

**Step 3. Find every key the change touches.**

- Artifact: affected identifiers — internal SKU, GTIN at each packaging level, retailer item
  numbers, pallet and case codes, tariff code, and any planning-item grouping.
- Key branch: **does the planning key differ from the selling key?** A pack change can leave a
  consolidated planning item intact while every selling key beneath it changes. That is the exact
  configuration that produces a silent merge.
- Failure if skipped: the change is fixed where it was noticed and left broken in the three other
  systems carrying a copy of the identifier.

**Step 4. Test whether the series crosses it.**

- Artifact: the measure around the effective date, in the recorded unit and in an invariant unit.
- Key branch: **does the step appear in the invariant unit?** A step in cases with no step in bottles
  is a unit artifact. A step in both is a demand event. A step in neither, with a change in cost per
  unit, is an attribute change.
- Failure if skipped: the analyst treats a unit artifact as demand and builds an elasticity or a
  promotion response around it.

### Phase B — Reconstruct

**Step 5. Build the dated mapping, then convert at the factor in force.**

- Artifact: one cross-reference row per predecessor–successor pair with direction, valid-from date,
  factor, and reason code; then a derived series in an invariant unit, with the factor and its
  validity window stamped on each row.
- Key branch: **graph or chain, and forward or backward?** A chain A to B to C can collapse to A to
  C, but a branching graph cannot, and a cycle will make a naive recursive join loop, so constrain
  by date and detect cycles. Convert forward to the invariant unit, then backward into the current
  pack; direct old-case to new-case conversion works only when both factors are stable.
- Failure if skipped: conversion is applied to a successor that already includes the predecessor's
  volume, or the current factor is applied to five years of history, restating quantities at a pack
  size that did not exist.

**Step 6. Allocate the split, then carry the residual separately.**

- Artifact: an allocation rule with an observed share, the window it was measured over, an owner,
  and a residual line with an assumption and a review date.
- Key branch: **observed or assumed, and growth or loss?** Use physical volume over the first six to
  twelve post-launch months; the first week's share is a launch artifact. Above the allocated
  history the residual is category growth or share gain, below it transition loss, and neither
  belongs inside a successor's history.
- Failure if skipped: a 12-pack replaced by a 10-pack and a 6-pack is split 50/50 because that is
  fair, and a one-time transition loss is smeared into both successors as a permanent property.

### Phase C — Consume

**Step 7. Choose restatement, intervention, or truncation.**

- Artifact: a documented modelling decision per series.
- Key branch: **can the break be modelled?** Restate when the physical quantity is known.
  Intervene when the date is known but the conversion is not. Truncate when neither is known and
  the pre-break data cannot be trusted.
- Failure if skipped: the model is fitted across the break with no term for it, and the level shift
  is absorbed by whichever parameter is most flexible.

**Step 8. Re-baseline the model and the metric together.**

- Artifact: a backtest window starting after the break, a restated error baseline, and a model-card
  note.
- Key branch: **does the accuracy metric use the same unit as the supply decision?** If accuracy is
  in cases and supply is in bottles, the metric cannot see the error that matters. Add a control
  total in the invariant unit.
- Failure if skipped: the model is judged against a pre-break baseline in a different unit, and the
  apparent change is an artifact.

### Phase D — Govern

**Step 9. Notify before the effective date, not after.**

- Artifact: a change notification to planning, procurement, logistics, finance, and every external
  partner holding the identifier, with lead time stated in days.
- Key branch: **how much lead time?** At least one full forecast cycle plus the frozen horizon. A
  notification after the monthly freeze is a notification about a problem, not a control.
- Failure if skipped: the change is found in the exception report three weeks later, and the only
  option left is a manual override.

**Step 10. Gate the model run on a drift check.**

- Artifact: a pre-run validation comparing the master version used by the model with the version at
  the last run, blocking on an unclassified difference in a planning-critical field.
- Key branch: **block or warn?** A warning is dismissed. A block with a documented override and a
  named approver is the only version that changes behaviour near quarter end.
- Failure if skipped: the model consumes the new master silently, and the first evidence is a plan
  nobody can explain.

### 5.1 The bitemporal test

Ask five questions of any historical dataset before using it as model input.

- **Transaction timestamp?** One date column is not bitemporal, whatever the table is called.
- **Master version archived?** If not, cell A cannot be reconstructed for that process.
- **Join on valid time or current state?** A join with no date predicate is a cell D join.
- **Can a restatement be reversed?** If the raw rows were overwritten, no.
- **Which cell does the report claim?** If it cannot answer, it shows the current master as history.

A dataset that passes all five is rare; one that passes the first and the fourth is usable, because
the rest can be rebuilt. The two things to insist on in a data request are the transaction timestamp
and the immutability of the raw layer.

### 5.2 Restate, intervene, or truncate

| Situation | Method | Consequence | When it is wrong |
|---|---|---|---|
| Physical factor known and stable | Restate to the invariant unit | Comparable series, clean backtest | The factor itself was wrong at the time |
| Date known, factor uncertain | Intervention term at the date | Preserves raw history, models the break | The break is not a pure step |
| Neither known, pre-break suspect | Truncate at the break | Honest, short series | The series is too short to estimate seasonality |
| Successor is a different product | Like modelling with a stated weight | Usable prior | The weight is left at 100% |
| One-to-many with observed share | Allocate plus residual | Transparent split | The share is measured over one week |

Truncation is underused. A series with 14 clean months after a break is more useful than 36 months
of mixed units, provided the seasonality can be borrowed from a related item. The temptation is
always to keep the length, because models are judged on observation counts, and length is not the
same as information.

### 5.3 The control that catches it

Of the three controls usually proposed, only one works. An annual audit finds attribute errors and
misses everything between audits. A volume exception report fires on promotions and stock-outs as
often as on drift. A pre-run master diff works because it is dated, automatic, and blocks rather
than warns: hash the planning-critical fields per item, store the hash with the run, and stop when
the hash changes without a change record.

## 6. Worked example

Ardleigh Beverages makes Kestrel sparkling water in 500 ml glass bottles. For two years it sells
one case configuration: 12 bottles per case, GTIN 5019876543202, item 10048213. At the start of the
third year it moves to a case of 10, GTIN 5019876543209, item 10077310, and discontinues the
12-pack. The liquid, the bottle, the label, and the retail price per bottle do not change. All
figures are illustrative and internally consistent; every arithmetic step is shown so that a reader
can reproduce it.

### 6.1 The change

| Field | Before | After | Changed? |
|---|---|---|---|
| Internal item | 10048213 | 10077310 | Yes |
| GTIN | 5019876543202 | 5019876543209 | Yes |
| Bottles per case | 12 | 10 | Yes |
| Bottle size | 500 ml | 500 ml | No |
| Case dimensions | 400 × 300 × 240 mm | 340 × 300 × 240 mm | Yes |
| Gross weight per case | 7.4 kg | 6.2 kg | Yes |
| Cases per pallet | 60 | 72 | Yes |
| Bottles per pallet | 720 | 720 | No |
| Standard cost per case | EUR 6.24 | EUR 5.20 | Yes, pro rata |
| Retail price per bottle | EUR 0.99 | EUR 0.99 | No |

Two things in this table matter more than the rest. First, bottles per pallet did not change,
because the pallet was re-engineered from 60 cases of 12 to 72 cases of 10. A pallet-based capacity
plan is therefore correct while a case-based one is not. Second, cost per case fell by 16.7%, which
means any margin report that compares cost per case across the boundary understates the old
product's cost by that amount.

The GTIN changed because the pack quantity changed. Under the GS1 GTIN Management Standard, rule
2.8, a change to the number of trade items in a case always requires a new GTIN at the affected
higher-level packaging. This is not a company choice, and it is the one part of the change that the
standard makes visible.

### 6.2 Assumptions stated before any arithmetic

| # | Assumption | Consequence if false |
|---:|---|---|
| 1 | Demand is withdrawals from the plant to customers, in cases, in the pack in force on the shipment date | Retail scans would show consumer take-away, not plant shipments |
| 2 | One bottle is 500 ml in every pack; the physical product is invariant | A bottle-size change would be a second drift inside the case change |
| 3 | No promotions, no price change per bottle, no stock-outs; demand is uncensored | A promotion would look like a level shift |
| 4 | 12-pack shipments stop at the end of month 24; 10-pack shipments begin in month 25; residual 12-pack stock ships and is counted in month 25 | A mixed window would need shipment-level reconstruction |
| 5 | A month is a calendar month; there are 36 consecutive months | Unequal periods would bias the trend |
| 6 | The successor mapping is complete, dated, and known before the analysis | Section 6.8 relaxes this for the one-to-many case |
| 7 | Bottle demand is the invariant quantity, and every check is against it | Without a control total, the 20% step is invisible |
| 8 | The series is deliberately clean, with no noise | The arithmetic is visible; estimation error is not |

Assumption 4 is generous. In practice a transition window of four to eight weeks contains both
packs, and the split within the window must be reconstructed from shipment records. Section 6.8
shows the messier version.

### 6.3 Thirty-six months in three columns

The underlying physical demand is stated in bottles; the transaction system recorded cases in
whatever pack was in force; the third column is the same demand in 12-bottle equivalent cases.

| Month | Pack | Bottles (btl) | Recorded (cs) | Equivalent (cs12) |
|---:|---:|---:|---:|---:|
| 1 | 12 | 54,000 | 4,500 | 4,500 |
| 2 | 12 | 51,600 | 4,300 | 4,300 |
| 3 | 12 | 57,000 | 4,750 | 4,750 |
| 4 | 12 | 59,400 | 4,950 | 4,950 |
| 5 | 12 | 63,000 | 5,250 | 5,250 |
| 6 | 12 | 66,000 | 5,500 | 5,500 |
| 7 | 12 | 67,200 | 5,600 | 5,600 |
| 8 | 12 | 64,800 | 5,400 | 5,400 |
| 9 | 12 | 58,800 | 4,900 | 4,900 |
| 10 | 12 | 57,600 | 4,800 | 4,800 |
| 11 | 12 | 60,600 | 5,050 | 5,050 |
| 12 | 12 | 64,800 | 5,400 | 5,400 |
| 13 | 12 | 55,200 | 4,600 | 4,600 |
| 14 | 12 | 52,800 | 4,400 | 4,400 |
| 15 | 12 | 58,200 | 4,850 | 4,850 |
| 16 | 12 | 60,600 | 5,050 | 5,050 |
| 17 | 12 | 64,200 | 5,350 | 5,350 |
| 18 | 12 | 67,200 | 5,600 | 5,600 |
| 19 | 12 | 68,400 | 5,700 | 5,700 |
| 20 | 12 | 66,000 | 5,500 | 5,500 |
| 21 | 12 | 60,000 | 5,000 | 5,000 |
| 22 | 12 | 58,800 | 4,900 | 4,900 |
| 23 | 12 | 61,800 | 5,150 | 5,150 |
| 24 | 12 | 66,000 | 5,500 | 5,500 |
| 25 | 10 | 56,400 | 5,640 | 4,700 |
| 26 | 10 | 54,000 | 5,400 | 4,500 |
| 27 | 10 | 59,400 | 5,940 | 4,950 |
| 28 | 10 | 61,800 | 6,180 | 5,150 |
| 29 | 10 | 65,400 | 6,540 | 5,450 |
| 30 | 10 | 68,400 | 6,840 | 5,700 |
| 31 | 10 | 69,600 | 6,960 | 5,800 |
| 32 | 10 | 67,200 | 6,720 | 5,600 |
| 33 | 10 | 61,200 | 6,120 | 5,100 |
| 34 | 10 | 60,000 | 6,000 | 5,000 |
| 35 | 10 | 63,000 | 6,300 | 5,250 |
| 36 | 10 | 67,200 | 6,720 | 5,600 |

The table is the whole case in one view. The recorded column is a valid transaction record at every
row. The equivalent column is the same business. Between months 24 and 25 the recorded column rises
while the physical quantity falls.

### 6.4 The boundary, in four numbers

| Measure | Month 24 | Month 25 | Change |
|---|---:|---:|---:|
| Bottles (btl) | 66,000 | 56,400 | −14.5% |
| Recorded cases (cs) | 5,500 cs12 | 5,640 cs10 | +2.5% |
| Equivalent cases (cs12) | 5,500 | 4,700 | −14.5% |
| Bottles per case | 12 | 10 | −16.7% |

```text
recorded change      = (5,640 - 5,500) / 5,500 = +2.55%
physical change      = (56,400 - 66,000) / 66,000 = -14.55%
sign reversal        = +2.55 - (-14.55) = 17.10 percentage points
pack factor          = 10 / 12 = 0.8333
phantom cases in M25 = 5,640 x (1 - 10/12) = 5,640 x 0.1667 = 940 cs
```

The recorded series rises by 2.5% in a month when physical demand fell by 14.5%. The 17.1-point gap
is not noise and it is not a demand event. It is the difference between counting cases and counting
bottles, and it appears at exactly the moment the case changed meaning.

The 940 phantom cases are worth naming precisely. They are not lost sales and they are not
inventory; they are the arithmetic difference between 5,640 cases of 10 and 5,640 cases of 12,
expressed in the old unit. Nobody can hold, ship, or sell them. They exist only in the series.

### 6.5 The false growth rate

| Year | Bottles (btl) | Recorded (cs) | Equivalent (cs12) | Recorded ÷ equivalent |
|---|---:|---:|---:|---:|
| 1 | 724,800 | 60,400 | 60,400 | 100.0% |
| 2 | 739,200 | 61,600 | 61,600 | 100.0% |
| 3 | 753,600 | 75,360 | 62,800 | 120.0% |
| **Total** | **2,217,600** | **197,360** | **184,800** | **106.8%** |

```text
physical growth, year 2 to year 3 = 753,600 / 739,200 - 1 = +1.95%
recorded growth, year 2 to year 3 = 75,360 / 61,600 - 1 = +22.34%
difference                        = 20.39 percentage points
total phantom cases, year 3       = 75,360 - 62,800 = 12,560 cs
phantom share of year 3           = 12,560 / 62,800 = 20.0%
phantom share of 36 months        = 12,560 / 184,800 = 6.80%
```

This is the number that enters a business plan. A company that reads its own history in cases
concludes that the brand grew 22.3% in the year of the pack change. It grew 2.0%. If the following
year's plan is set at "last year plus the same growth", the plan is 20.1% above the physical
requirement before a single judgement about the market is made:

```text
plan carried forward = 1.2234 / 1.0195 = 1.2000 -> +20.0% physical
```

The error is not in the arithmetic. Every one of those percentages is correct in its own unit. The
error is that the two ends of the comparison are different objects, and the percentage has no
meaning until somebody states which one it is measuring.

### 6.6 Is the case series wrong?

No: the case series is a correct record of cases shipped. If the business plans in cases, produces to a
case plan, and ships cases, the 20% step is not an error at all: there really were more, smaller
cases. It is wrong only where a case is converted into something else, and this business has four
such interfaces: manufacturing schedules bottles, procurement buys glass in tonnes, logistics plans
pallet positions, and finance reports cost per case, which fell 16.7% for the same bottle cost.

So the case series is not a unit. Every conversion from it needs the factor in force on the date of
the row, and a table keyed to the current master supplies the wrong factor for 24 of 36 months.

```text
bottles required in month 25 = 5,640 cs10 x 10 = 56,400 btl   correct
bottles implied by old factor = 5,640 cs10 x 12 = 67,680 btl   +20.0%
```

That 11,280-bottle gap is the entire physical consequence of the unit change, and it appears only
if the factor is stale. It is worth separating from the modelling error in section 6.7, because the
two have different owners: the factor belongs to master data, the model belongs to planning.

### 6.7 The demand-plan error, in four cells

Fit two models of the same class to the 36 months: a linear trend plus eleven month dummies, by
ordinary least squares. The first is fitted to the recorded case series. The second is fitted to
the equivalent 12-bottle series. Both are then extrapolated to month 37, the next January. Month 37
is known in this example so that the error can be measured rather than asserted: 57,600 bottles,
which is 5,760 cases of 10 or 4,800 equivalent cases.

| Model fitted on | Month 37 forecast | In the forecast unit | Physical (btl) | Error vs 57,600 |
|---|---:|---|---:|---:|
| Recorded cases | 6,160 | cs10 | 61,600 | +6.9% |
| Equivalent cases | 4,800 | cs12 | 57,600 | 0.0% |

Now cross the model with the conversion factor that the plant uses. The plant's routing was set
when a case was 12 bottles, and the factor is a field in the manufacturing system that nobody
changed.

The equivalent forecast of 4,800 cs12 is 57,600 bottles, which is 5,760 cases of the current 10-pack;
that is the quantity the plant releases.

| Model | Cases released (cs10) | Factor applied | Supply (btl) | Error vs 57,600 |
|---|---:|---:|---:|---:|
| Equivalent, correct | 5,760 | 10 | 57,600 | 0.0% |
| Equivalent, stale | 5,760 | 12 | 69,120 | +20.0% |
| Recorded, correct | 6,160 | 10 | 61,600 | +6.9% |
| Recorded, stale | 6,160 | 12 | 73,920 | +28.3% |

```text
correct model, stale factor   = 5,760 x 12 = 69,120 btl   = +20.0%
drifted model, correct factor = 6,160 x 10 = 61,600 btl   = +6.9%
drifted model, stale factor   = 6,160 x 12 = 73,920 btl   = +28.3%
multiplicative check          = 1.2000 x 1.0694 = 1.2833  = +28.3%
```

The two error layers multiply, and they have different owners. The factor layer is a master-data
failure with a one-line fix; the model layer is a series-keying failure that survives the fix. A
programme that corrects only the factor halves the problem and reports it as solved.

**Why the model layer is silent.** Backtested over the 24-month fit window, the recorded-case model
scores 5.6% mean absolute percentage error (MAPE) and the equivalent-case model 6.1%. The drifted
model is marginally *better* on the only test model selection normally runs, and the half-point
difference points the wrong way. A backtest cannot see a change in what a case means, because it
compares forecasts with actuals in the same drifted unit. The accuracy key performance indicator
(KPI) cancels the same way: 6,160 cs10 forecast against 5,760 cs10 actual is a 6.9% case error, which
passes a 10% threshold, while the physical error with the stale factor is 28.3%. The KPI answers
"how well did we forecast cases"; the warehouse asks "how many bottles did we make".

### 6.8 A successor that is not one-to-one

The clean version of this case is a luxury. The realistic version has the 12-pack discontinued and
two successors introduced: a 10-pack and a 6-pack, aimed at different channels. There is no single
conversion factor, and there is no historical split, because the split did not exist before launch.

Take the first six months after launch, months 25 to 30, and compare them with the same calendar
months one year earlier.

| Quantity | Bottles (btl) | Cases |
|---|---:|---|
| Pre-change, M13–M18 | 358,200 | 29,850 cs12 |
| Post-launch, M25–M30 | 365,400 | — |
| 10-pack, 60% of history | 214,920 | 21,492 cs10 |
| 6-pack, 40% of history | 143,280 | 23,880 cs6 |
| Residual (post-launch less history) | +7,200 | +2.0% |

```text
pre-change total          = 55,200 + 52,800 + 58,200 + 60,600 + 64,200 + 67,200
                          = 358,200 btl
post-launch total         = 56,400 + 54,000 + 59,400 + 61,800 + 65,400 + 68,400
                          = 365,400 btl
residual                  = 365,400 - 358,200 = +7,200 btl = +2.01%
10-pack allocation @ 60%  = 358,200 x 0.60 = 214,920 btl = 21,492 cs10
6-pack allocation @ 40%   = 358,200 x 0.40 = 143,280 btl = 23,880 cs6
observed minus allocated  = +4,320 btl (10-pack), +2,880 btl (6-pack)
```

The residual is the part with no counterpart. It could be category growth, it could be the 6-pack
recruiting new buyers, or it could be a measurement artifact of the first six months. It is not a
property of the old 12-pack, and it must not be pushed into either successor's history. Allocate
the predecessor's history across the successors at the observed share, then carry the residual as a
separate line with an owner, an assumption, and a review date.

The sensitivity to a guessed split is larger than most teams expect. If the analyst splits 50/50
because no share data is available at the time of the change:

| Series | 50/50 (btl) | 50/50 (cs) | 60/40 (cs) | Error |
|---|---:|---:|---:|---:|
| 10-pack | 179,100 | 17,910 cs10 | 21,492 cs10 | −16.7% |
| 6-pack | 179,100 | 29,850 cs6 | 23,880 cs6 | +25.0% |

A 10-point error in the assumed share produces a 17% error in one series and a 25% error in the
other, in opposite directions. The portfolio total is roughly preserved, which is exactly why the
error survives aggregate review: the two mistakes cancel at the level where managers look.

Six rules make the split defensible.

| # | Rule | Why |
|---:|---|---|
| 1 | Allocate in bottles, never in cases | Case-based allocation cannot be checked against a control total |
| 2 | Measure the share on physical volume | A 6-pack otherwise looks like a larger share of the business |
| 3 | Use a full seasonal cycle or like calendar months | A short window confuses seasonality with the split |
| 4 | Hold the residual out; do not normalise shares to 100% of history | Normalising absorbs transition loss into both successors |
| 5 | Re-estimate the share monthly for six to twelve months | The launch share is not the steady-state share |
| 6 | Never leave the predecessor weight at 100% on both successors | Two full inheritances double the portfolio, invisibly |

Rule 6 is broken by default rather than by decision. Planning tools that carry a reference-item
weight frequently initialise it at 100%, and a 100% weight on two successors is a doubling that
nobody typed.

### 6.9 What it cost, and what to ask

At EUR 5.20 per 10-pack case, the excess supply is:

```text
correct model, stale factor = 11,520 btl / 10 = 1,152 cs x EUR 5.20 = EUR 5,990
drifted model, stale factor = 16,320 btl / 10 = 1,632 cs x EUR 5.20 = EUR 8,486
```

Per planning cycle. Three cycles before the inventory signal forces a correction makes EUR 18,000
to EUR 25,000 for one item, before the production slots and glass committed to product nobody
needed. Three questions should have been asked before sign-off: which clock produced the 22.3%
growth figure; whether standard cost per case was restated with the pack; and who owns the +2.0%
residual in the one-to-many split.

## 7. Code, query, or calculation

The reconstruction is one join and one multiplication, provided the two supporting tables exist.
The query below computes the comparable series in bottles and in 12-bottle equivalent cases. It
assumes a monthly demand fact, a dated successor map, and an effective-dated conversion table.

```sql
-- fact_demand_monthly(item_id, month_id, qty_cases)
-- item_successor(predecessor_item_id, successor_item_id, effective_from)
-- item_uom_history(item_id, valid_from, valid_to, bottles_per_case)
select d.month_id,
       sum(d.qty_cases * u.bottles_per_case)        as bottles,
       sum(d.qty_cases * u.bottles_per_case) / 12.0 as equivalent_cases_12
from fact_demand_monthly d
join item_uom_history u
  on u.item_id = d.item_id
 and d.month_id >= u.valid_from
 and d.month_id <  coalesce(u.valid_to, date '9999-12-31')
left join item_successor s
  on s.predecessor_item_id = d.item_id
 and d.month_id >= s.effective_from
group by d.month_id
order by d.month_id;
```

Three properties matter more than the syntax.

- **The factor comes from the item that shipped, not from its successor.** A row's cases are only
  convertible at the factor in force for that item on that date. Look the factor up through the
  successor and 24 months of 12-packs are silently restated at 10.
- **The factor join is a range join.** Joining on `item_id` alone returns whichever version is
  current, which is the same restatement by another route.
- **The output is in an invariant unit.** `bottles` is the control total; `equivalent_cases_12` is
  the comparable series. Neither is the number the business reports, which is the point.

The query still does not allocate a one-to-many successor; that needs an observed share and a
residual, which is a judgement rather than a join.

## 8. Data, standards, and regulatory context

Drift is lightly regulated and heavily standardised. The standards govern identity and packaging;
almost nothing governs the retention of the history that makes a change reconstructable. That gap
is the practical problem.

### 8.1 Item-change rules

| Rule or standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN Management Standard 1.1 (2023), rule 2.3 | Declared net content | Any increase or decrease requires a new GTIN; no tolerance | B |
| GS1 GTIN Management Standard 1.1, rule 2.4 | Dimensional or gross weight change | A change over 20% on any axis, or in gross weight, requires a new GTIN | B |
| GS1 GTIN Management Standard 1.1, rule 2.8 | Pack or case quantity | Any change to trade items per case or cases per pallet always requires a new GTIN at the affected level | B |
| GS1 General Specifications | GTIN allocation | A GTIN shall not be reallocated to a different product | B |
| GS1 Global Data Model (GDM) | Attribute definitions | Packaging hierarchy, TI/HI, and pack attributes that partners exchange | B |
| GDSN | Item data synchronisation | Publication of the changed item to trading partners through a certified data pool | B |
| EPCIS 2.0 | Event data | Object-level events that let a change be reconciled across parties | B |

Two consequences for the analyst. First, the change that caused the drift was *supposed* to produce
a new identifier, so a retained key is a control failure rather than a physical inevitability.
Second, the 20% threshold in rule 2.4 applies to dimensions and gross weight only; pack quantity has
no threshold. A common error is to argue that a pack change of less than 20% does not need a new
GTIN, which is true of weight and false of pack count.

### 8.2 Time and history

| Standard or pattern | Governs | What it gives you | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation | Unambiguous dates, including time zone offsets | B |
| SQL:2011 system-versioned tables | Transaction-time history in a relational database | Automatic row versioning on update and delete | B |
| SCD type 2 | Warehouse dimension history | Valid-time versions with a current flag | C, convention |
| Event sourcing | Application state as an append-only log | Replay to any transaction time | C, convention |
| Bitemporal modelling | Both clocks in one record | As-of and as-at queries over the same data | A as theory, no dominant product standard |

The important absence: there is no widely adopted standard for bitemporal *master data* exchange
between companies. GDSN publishes current item data with an effective date and does not publish the
prior version as a first-class object, so a partner that needs last March's version relies on its own
copy. That is why the retailer's catalogue is a separate source of drift and why the transition
window is the hardest part of the reconstruction.

### 8.3 Traceability obligations that drift can break

| Requirement | Jurisdiction | Why a supersession break matters | Grade |
|---|---|---|---|
| DSCSA | United States, pharmaceuticals | Transaction information must trace a product through ownership changes; a reused or unmapped identifier breaks the chain | B |
| EU Falsified Medicines Directive | European Union | Verification at dispensing depends on a stable product identifier | B |
| FSMA 204 | United States, food | Key data elements including lot and location must be retrievable across a product's life | B |
| EU general food law traceability | European Union | One step back, one step forward, across reformulations | B |

None of these requires a comparable demand series. They require that a physical unit can be traced,
which is a weaker condition. But the same master-data discipline serves both, and an organisation
that cannot map a predecessor to a successor for forecasting usually cannot map it for a recall
either. The forecast failure is the visible symptom of a traceability exposure.

What is convention rather than requirement: "one GTIN per SKU" is an internal rule, not a GS1 one; a
24-month history window is a planning habit, too short for two full cycles plus a transition; and
"the warehouse keeps history" is an assumption, since most warehouses keep transactions and
overwrite the dimension.

## 9. Industry practice and case evidence

### 9.1 What the planning tools claim to do

| Vendor and document | Mechanism described | Grade | What to take from it |
|---|---|---|---|
| SAP, supersession documentation for SCM and service parts planning | 1:1, 1:n, n:1, 1:0 and multi-level chains; quantity factors and partial-substitution percentages; data realignment writes predecessor demand into successor history without changing raw data | D | The cardinalities are the right vocabulary; the "raw data unchanged" property is the one to insist on |
| Oracle Fusion Cloud Demand Management, supersession forecasting | Aggregates prior revisions' history and applies it to the current revision; covers reciprocal relationships; 1:n uses the highest-ranked child, n:1 aggregates across the chain | D | Tool support is conditional on relationship type and plan scope, which is where implementations fail |
| Infor Supply Chain Planning, supersession | Controlled replacement with rules for transferring history and controlling forecast periods during the transition; no circular chains | D | Transition periods are a first-class object, not a single date |
| AWS Supply Chain, product lineage (November 2023) | Links a product to predecessors with an alternate quantity weight and an effective period, producing a surrogate history | D | "Surrogate history" is the vendor term for a like-modelled series; the weight is an assumption |

These are vendor documents describing capability, not evidence of outcomes. They are useful because
they show that the mechanism is well understood by tool builders and still routinely misconfigured
by implementers. The recurring configuration failures are the ones in section 10: a default weight
left at 100%, a relationship type the tool does not support, and a transition window collapsed to a
single date.

### 9.2 Peer-reviewed evidence on breaks and transitions

Huang, Fildes, and Soopramanien (2019) study retailer product sales under structural change across
1,831 IRI stock keeping units in 28 categories; conventional models assume constant marketing-mix
effects and are biased when the process changes. They combine a sequential Chow test for break
detection with intercept correction and estimation-window combining, and report a MASE improvement
of roughly 11% over a base-lift benchmark. The setting is promotional rather than packaging change,
but the mechanism is identical: a model fitted across a break misattributes the break to its
parameters. Grade A.

Box and Tiao (1975) supply the canonical form of intervention analysis: an outcome series plus an
explicit step or pulse input at a known date, with a dynamic response. A pack change is a step
intervention with a known date, the easiest case in the framework and the one most often left
unfitted (Grade A for the method; the exposition consulted is a teaching chapter, not evidence).
Altansukh and Osborn (2021) add break-date uncertainty, averaging forecasts over a confidence set,
which matters when the effective date is disputed. Huang, Hardie, and Fader (2004) model new-product
sales with an evolving changepoint, the closest published treatment of phase-in. Both Grade A.

What the literature does *not* provide is a method for a unit change with no physical change. That
is not a forecasting problem but a measurement problem, solved by conversion rather than by a better
model.

### 9.3 Practitioner evidence on unit conversion

| Source | Claim or observation | Grade | Use |
|---|---|---|---|
| DemandLabs, unit-of-measure conversion article, undated, accessed 2026 | A wrong conversion produces a clean multiple, not noise, so it passes outlier checks and percentage-error review; correcting it creates a step change that biases history unless restated, truncated, or flagged | D | The clean-multiple observation is the best short statement of why the error is silent |
| Scout, case-pack glossary | A pack-size change from 12 to 24 silently breaks historical comparisons and makes velocity look like a cliff when demand did not change | D | Directional only; the mechanism is correct |
| Circana, equivalized sales definition | Pack-size normalisation, units multiplied by pack size and divided by a standard equivalent size, is the standard commercial method for removing pack bias | C | The industry has a name for the invariant unit; use it |
| Nestlé Brazil case study, SAS conference paper, 2019 | Sales history may need to be realigned between different SKUs because of resizing, and can be transformed into different units of measure | C | A named company describing the problem, without quantified outcome |
| demandplanning.net, product-lifecycle forecasting note | Two common configuration traps: predecessor weighting left at the default 100%, and the self-mapping pair omitted so the new item never learns from its own actuals | D | The 100% trap is the single most consequential configuration error |

The convergence is worth stating plainly. The conversion error is multiplicative, it survives
percentage-based metrics, the correction itself creates a break, and the tooling's defaults can
double a portfolio. None of that is controversial, which is why the problem persists.

### 9.4 Bitemporal practice

A 2026 systematic literature review of bitemporal databases (54 papers, 1995–2026) finds the
conceptual model settled and the product support weak: limited database support, little cloud-native
optimisation, and little integration with machine learning. It names supply chain and inventory
control among the application areas. Grade A.

Practitioner treatments use the same distinction in different words: Fowler's "actual time" and
"record time", and the XTDB split between as-of queries over transaction time and as-at over valid
time. Both are correct and both are Grade C or D. The vocabulary here follows the database
literature.

The conclusion is uncomfortable: the theory has existed for three decades, storage is trivial, and
most planning landscapes still keep one row per item. Nobody owns the history as a product.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Pack change under a retained code | Series steps once and never returns | Key stable, unit changed | Every conversion downstream is wrong by the pack ratio | New identifier plus effective-dated factor |
| New GTIN with no successor link | New item starts at zero history | Identity change without a map | Like-model guess, usually too high or too low | Dated cross-reference at creation |
| Conversion factor frozen at the old pack | Case counts reconcile; bottle counts do not | Factor keyed once | 20% over- or under-supply, invisible in cases | Range join on an effective-dated table |
| History rebuilt from current master | Every report agrees and every backtest is optimistic | Current-state join | The past is restated to a pack that did not exist | Keep raw rows immutable; use as-of queries |
| Effective date backdated without a change record | Two weeks of transactions use the new pack against the old master | Valid time set to entry date | A bounded window of unexplained quantities | Record both clocks; flag the window |
| One-to-many split by assumption | Both successor series look plausible; portfolio total is stable | Share guessed, residual smeared | One series 17% low, the other 25% high | Observed share plus residual line |
| Predecessor weight left at 100% | Two new items each carry the full old volume | Tool default | Portfolio forecast doubles | Explicit weight, sum checked to 100% |
| Semantic change with no schema change | A metric moves on a date when nothing physical happened | Definition renegotiated | Every filter using the field is wrong | Versioned definition register |
| Supersession cycle | Recursive query never returns | A to B to A map | Pipeline failure, or silent truncation | Cycle detection and a date constraint |
| Transition window treated as a single date | Both packs ship in the same month; one is unattributed | Two physical items, one effective date | Double count or loss in the crossover month | Model the window, reconcile to shipments |
| Case-based accuracy KPI | KPI stable, service worsens | Numerator and denominator share the drifted unit | The metric cannot see the physical error | Control total in an invariant unit |
| Change notification after the planning freeze | Planner discovers the change in the exception report | Lead time shorter than the cycle | Manual override, then distrust of the system | Notify one full cycle before the effective date |

Two of these deserve expansion.

**Pack change under a retained code** is the parent failure, creating the conditions for four of the
others, and the one a competent ERP configuration prevents. **Case-based accuracy KPI** is why the
problem survives detection: the measurement system is denominated in the thing that drifted, so it
reports on the model rather than the physical world.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| One identifier denotes one physical configuration for its life | A series needs a stable object | Retained keys, reused GTINs | Compare pack attributes per key across dates |
| The conversion factor is stable between changes | Aggregation assumes a fixed ratio | Supplier pack changes, catch weight, re-engineering | Recompute totals from base units |
| The successor map is complete and dated | Reconstruction depends on it | Manual maps, partial coverage, undated rows | Count active items with no predecessor link |
| The transition is a step, not a ramp | The worked example uses a clean break | Both packs ship for weeks | Count shipment lines by pack in the window |
| Bottles are the invariant unit | Control totals need one | Reformulation changes bottle size | Check that the physical unit is unchanged |
| Demand is uncensored | The series is demand, not sales | Stock-outs, allocation, rationing | Compare shipments with orders and stock |
| The effective date is known | The break can be located | Backdating, disputed dates | Compare the change log with the first shipment |
| Standard cost per case was restated | Margin comparisons need it | Finance left the old cost | Divide cost per case by bottles per case |
| The historical master version exists | Cell A of the plane is reconstructable | Current-state-only dimensions | Try to reproduce last year's plan input |

Two limits are worth stating.

**The worked example is deliberately clean.** A real transition has a ramp, returns of the old pack,
retailer delistings, and a launch promotion. Those make the arithmetic messier without changing the
mechanism. A reader who can reproduce section 6 can handle the mess; one who cannot will be lost.

**The reconstruction is descriptive, not causal.** It tells you what the comparable series was, not
whether the pack change caused the volume change. If the 10-pack launched with a price promotion,
the +2.0% residual in section 6.8 mixes the two effects, and no amount of conversion separates them;
that requires a design, which Module 05 treats.

## 12. Exercises

### Exercise 1 — Classify the drift (recall and application, 40 minutes)

For each change, name the drift type or types, the field that carries the change, and the unit in
which the series stops being comparable.

1. A shampoo bottle moves from 400 ml to 375 ml at the same price, same SKU.
2. A case of 24 is replaced by a case of 20, new GTIN, old item retired.
3. A supplier moves from a 500 g bag to a 450 g bag but keeps the buyer's item code.
4. "Status A" is redefined from "sellable" to "orderable, quality release pending".
5. A 12-pack is replaced by a 10-pack and a 6-pack; the 10-pack inherits the old planning item.

**Deliverable:** a five-row table with type, field, and unit.
**Acceptance criteria:** at least two rows carry more than one type; every unit is named (ml, g,
bottles, eaches); no row says only "data change".

### Exercise 2 — Build the comparable series (calculation, 45 minutes)

Eight quarters of demand. Pack 24 for Q1–Q4, pack 18 from Q5. Express every quarter in bottles and
in 24-bottle equivalent cases, and compute the recorded-versus-equivalent overstatement.

| Quarter | Cases | Pack |
|---|---:|---:|
| Q1 | 12,000 | 24 |
| Q2 | 11,400 | 24 |
| Q3 | 12,600 | 24 |
| Q4 | 13,200 | 24 |
| Q5 | 16,800 | 18 |
| Q6 | 15,600 | 18 |
| Q7 | 17,400 | 18 |
| Q8 | 18,000 | 18 |

**Deliverable:** an eight-row table with bottles, equivalent cases, and recorded cases; the total
overstatement in cases and as a percentage; the Q4-to-Q5 change in both units.
**Acceptance criteria:** bottles reconcile exactly; the overstatement is attributed to the post-
change quarters only; the sign of the Q4-to-Q5 change is stated for each unit.

### Exercise 3 — The four-cell error table (calculation, 60 minutes)

A 20-pack becomes a 16-pack. The model, fitted across the break, forecasts 8,400 cases of 16 for the
next period. The true requirement is 134,400 bottles. The plant still converts at 20 bottles per
case.

**Deliverable:** the four-cell table (correct or drifted model × correct or stale factor) with
supply in bottles and percentage error against 134,400; the multiplicative check.
**Acceptance criteria:** all four cells computed; the two error layers separated; the multiplicative
identity shown; one sentence on which layer belongs to master data and which to planning.

### Exercise 4 — Split a one-to-many successor (design and calculation, 60 minutes)

A 10-pack is discontinued and replaced by an 8-pack and a 4-pack. The comparable six months before
the change total 240,000 bottles. The first six months after launch total 252,000 bottles, split
two-thirds to the 8-pack and one-third to the 4-pack.

**Deliverable:** the allocation at the observed share in bottles and cases, the residual, and a
sensitivity table for a 50/50 split.
**Acceptance criteria:** the residual is carried separately, not allocated; the share is applied to
physical volume; the sensitivity states the percentage error in each successor; a named owner and
review date are proposed for the residual.

### Exercise 5 — Three queries, three questions (design, 50 minutes)

For a forecast run dated 31 March, specify three queries: as-was at that date, as-is today, and
as-should-have-been for the same valid date.

**Deliverable:** one paragraph per query naming the transaction-time and valid-time predicates, and
a fourth paragraph stating which one reproduces the plan the business actually ran.
**Acceptance criteria:** the two clocks are distinguished in every query; the as-is query is
identified as a current-master join; the answer to the fourth paragraph is justified, not asserted.

### Exercise 6 — Drift register (audit, 75 minutes)

Take any public product catalogue or an internal item master you can access. Build a register of
items that changed pack, size, or identifier in the last 24 months.

**Deliverable:** a table with item, change date, change type, old and new pack, mapping status, and
whether the history was restated.
**Acceptance criteria:** at least 15 items; at least one item of each of the five drift types or a
stated reason why a type is absent; every row states the evidence used to date the change.

### Exercise 7 — Change notification design (design, 60 minutes)

Design the notification for a pack change at a company with a monthly planning cycle, a four-week
frozen horizon, and three external retail customers.

**Deliverable:** a one-page process with the trigger, the audience, the lead time in days, the
content, and the escalation path when the notification is late.
**Acceptance criteria:** lead time is at least one cycle plus the frozen horizon; the external
customers are named as recipients with a separate deadline; a late notification has a defined
consequence, not just an apology.

### Exercise 8 — The model-run gate (design, 45 minutes)

Specify the pre-run check that compares the master version used at the last run with the current
one.

**Deliverable:** the field list, the hash or comparison method, the block-and-override rule, and
the log record.
**Acceptance criteria:** the field list is limited to planning-critical attributes and justified;
the check is dated, not sampled; the override requires a named approver and a reason code; the log
is queryable by item and run date.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
Q1 12,000 cs x 24 = 288,000 btl = 12,000 cs24
Q2 11,400 cs x 24 = 273,600 btl = 11,400 cs24
Q3 12,600 cs x 24 = 302,400 btl = 12,600 cs24
Q4 13,200 cs x 24 = 316,800 btl = 13,200 cs24
Q5 16,800 cs x 18 = 302,400 btl = 12,600 cs24
Q6 15,600 cs x 18 = 280,800 btl = 11,700 cs24
Q7 17,400 cs x 18 = 313,200 btl = 13,050 cs24
Q8 18,000 cs x 18 = 324,000 btl = 13,500 cs24

bottles total        = 2,401,200 btl
equivalent total     = 100,050 cs24
recorded total       = 117,000 cs
overstatement        = 117,000 - 100,050 = 16,950 cs
as share of equivalent = 16,950 / 100,050 = 16.94%
post-change recorded = 67,800 cs; post-change equivalent = 50,850 cs
post-change ratio    = 67,800 / 50,850 = 1.3333 = +33.3%

Q4 to Q5 recorded    = (16,800 - 13,200) / 13,200 = +27.27%
Q4 to Q5 physical    = (302,400 - 316,800) / 316,800 = -4.55%
```

The post-change ratio is 33.3%, exactly `24/18 − 1`. Reporting 16.94% as "the error" averages the
whole window and misses the number that applies from Q5 onward; the average is diluted by
pre-change quarters that are correct by construction.

**Exercise 3 arithmetic.**

```text
true requirement            = 134,400 btl = 8,400 cs16
correct model, factor 16    = 134,400 btl  ->  0.0%
correct model, factor 20    = 168,000 btl  -> +25.0%
drifted model, factor 16    = 144,000 btl  -> +7.14%
drifted model, factor 20    = 180,000 btl  -> +33.93%

multiplicative check        = 1.0714 x 1.2500 = 1.3393
```

The factor layer is master data: one field, one owner, one fix. The model layer is planning: a
series concatenated across two pack sizes and fitted with a trend. Correcting the factor alone
leaves a 7.1% over-supply, inside most tolerance bands and therefore never investigated.

**Exercise 4 arithmetic.**

```text
pre-change total            = 240,000 btl
post-launch total           = 252,000 btl
residual                    = +12,000 btl = +5.0%

8-pack observed, 2/3        = 168,000 btl = 21,000 cs8
4-pack observed, 1/3        =  84,000 btl = 21,000 cs4

allocated at 2/3 and 1/3 of 240,000:
8-pack                      = 160,000 btl = 20,000 cs8
4-pack                      =  80,000 btl = 20,000 cs4
residual by successor       = +8,000 btl (8-pack), +4,000 btl (4-pack)

50/50 sensitivity on 252,000:
8-pack                      = 126,000 btl = 15,750 cs8  -> -25.0% vs 21,000
4-pack                      = 126,000 btl = 31,500 cs4  -> +50.0% vs 21,000
```

A strong answer allocates from the pre-change total at the observed share, holds the 12,000 bottles
out as an owned residual, and shows that 50/50 preserves the portfolio total while being 25% low on
one series and 50% high on the other. A weak answer normalises the shares to the post-launch total,
absorbing the residual and making the growth look structural.

**Exercises 5, 7, and 8 rubrics.**

A strong query answer writes predicates explicitly: as-was uses `transaction_time <= '2026-03-31'`
and a `valid_time` interval containing the date; as-is uses the maximum transaction time and no
valid-time restriction beyond today; as-should-have-been uses the maximum transaction time with the
valid-time predicate. It states that as-was reproduces the plan actually run, because the plan
consumed the master as known on that date, including its errors.

A strong notification design triggers on the approved change record, not the first shipment, and
notifies planning, procurement, manufacturing, finance, logistics, and the three retail customers.
Lead time is at least eight weeks for a monthly cycle with a four-week frozen horizon; external
customers get a separate, longer deadline because their catalogue updates are slower; and a late
notification escalates to a named role with authority to delay the effective date.

A strong gate hashes a short planning-critical field set per item — pack count, base unit, conversion
factor, weight, shelf life, lifecycle status, GTIN — and compares it with the hash stored at the last
successful run. A difference with no approved change record blocks the run; the override needs a
named approver and a reason code, and the log stores item, old and new hash, run date, approver, and
reason. A weak answer in any of the three warns rather than blocks, samples 5% of items, or checks
only the item count.

## 14. Interview questions

**1. "What is master-data drift, and why does a data-quality score not catch it?"**
A strong answer defines drift as a change in what a field or key means over time without a schema
change, and separates it from staleness and from wrong values. It notes that completeness,
validity, and uniqueness are evaluated per record or per snapshot, while drift is a property of the
sequence, so a file can pass every dimension and still compare two different things. A weak answer
calls it "bad data" and proposes a cleanse.

**2. "A pack changes from 12 to 10 under the same item code. What do you check first?"**
A strong answer goes in order: whether the GTIN was reallocated, whether the conversion factor is
effective-dated, and whether the history was restated. It immediately asks for the invariant unit
and computes a control total in bottles. A weak answer says "update the item master" and stops.

**3. "Explain valid time and transaction time."**
A strong answer uses a concrete dated example — the pack changed on 1 January but was keyed on
14 January — and says which query returns which answer for 10 January. It notes that an as-was
query reproduces the decision the business made, including its errors. A weak answer says both are
"dates" and cannot say why the distinction matters.

**4. "Why is a history rebuilt from current master data not the history the business saw?"**
A strong answer describes the current-state join: every historical transaction attributed to the
item as defined today, internally consistent and decision-invalid. It notes that the plan consumed
the master as known at the time, so a backtest on restated history is optimistic. A weak answer
says the current master is "more accurate" and therefore better.

**5. "A new GTIN is created. What else must exist before a model can use it?"**
A strong answer lists four things: a dated successor mapping, an effective-dated conversion factor,
a lifecycle status change on the predecessor, and a decision on how much predecessor history to
use. A weak answer says the GTIN is enough because it is the global standard.

**6. "A 12-pack is replaced by a 10-pack and a 6-pack. How do you split the history?"**
A strong answer converts to bottles, measures the observed share over a seasonal window, allocates
the predecessor total at that share, and carries the residual separately with an owner. It warns
against the 100% default on two successors. A weak answer splits 50/50 or sums the whole history
into both.

**7. "Your forecast accuracy is 95% and the plant is over-supplied by 25%. How?"**
A strong answer locates the unit boundary: accuracy in cases cancels the pack change, while supply
is released in bottles at a stale factor. It separates the model layer from the conversion layer
and says which owner fixes which. A weak answer blames the forecast or the plant.

**8. "How do you detect a semantic change?"**
A strong answer says you diff the definition, not the data: a versioned register of planning-
critical fields, their allowed values, and their owners, compared across dates. It gives an example
such as a status code whose population changed with no schema change. A weak answer says you look
for outliers.

**9. "Restate, intervene, or truncate — how do you choose?"**
A strong answer gives the conditions: restate when the physical factor is known, intervene when the
date is known but the conversion is not, truncate when neither is known and the pre-break data is
suspect. It notes that truncation is underused and that a shorter clean series beats a longer mixed
one. A weak answer always restates.

**10. "Design a control that stops a model consuming a drifted series."**
A strong answer proposes a pre-run master diff over a short planning-critical field set, hashed per
item, compared with the last run, blocking on an unexplained change with a named override. It says
why an annual audit and a volume exception report fail. A weak answer proposes another dashboard.

## 15. Key terms

- **Master-data drift** — change in what a record means over time without a schema change.
- **Attribute change** — weight, dimensions, shelf life, or temperature class changes; key usually
  survives.
- **Structural change** — case pack, inner pack, or pallet configuration changes; the case stops
  being a constant quantity.
- **Identity change** — a new identifier, typically a GTIN, for a changed product.
- **Supersession** — a dated replacement relationship, with a factor and often a cardinality.
- **Semantic change** — a field or code value is redefined; no key, no schema, no data event.
- **Valid time** — when a fact was true in the world.
- **Transaction time** — when the database recorded the fact.
- **Bitemporal** — carrying both clocks in one record.
- **As-of / as-at query** — reconstruct the database as known at a past transaction time / what was
  true at a past valid time.
- **GTIN** — Global Trade Item Number, unique per trade item per packaging level.
- **GTIN reuse** — reallocating an identifier to a different product; forbidden.
- **Case pack** — consumer units per case.
- **UOM** — unit of measure; the **base unit** is the lowest counted unit, usually the each.
- **Conversion factor** — the ratio between two units.
- **Effective-dated factor** — a factor with a validity window, applied according to the row date.
- **Supersession chain** — a directed graph of replacements, possibly branching.
- **One-to-one / one-to-many / many-to-one** — replacement cardinalities; each needs a different
  reconstruction.
- **Restatement** — recomputing a history in a consistent unit, with the rule recorded.
- **Comparable series** — every observation in the same physical unit and definition.
- **Equivalent unit (EQ)** — a normalised unit that removes pack-size differences.
- **Level shift** — a one-time change in the mean of a series; a **structural break** is the change
  in process behind it.
- **Intervention** — an explicit model term for a known event.
- **Like modelling** — forecasting a new item from a reference item's history; phase-in and
  phase-out are the ramps at either end of a transition.
- **Residual** — post-launch volume not explained by the allocated history.
- **Cross-reference table** — the dated map between retired and current identifiers.
- **Change notification** — a dated message to downstream consumers before a change takes effect.

## 16. Sources

**Peer-reviewed and academic**

- Huang, T., Fildes, R., and Soopramanien, D. (2019). "Forecasting retailer product sales in the
  presence of structural change." *European Journal of Operational Research*. 1,831 IRI SKUs across
  28 categories; sequential Chow test, intercept correction, estimation-window combining; about 11%
  MASE improvement over a base-lift benchmark. (Grade A)
- Box, G. E. P., and Tiao, G. C. (1975). Intervention analysis with applications to economic and
  environmental problems. *Journal of the American Statistical Association*. The step-and-pulse
  model form used in section 5.2. (Grade A for method)
- Altansukh, G., and Osborn, D. R. (2021). "Using structural break inference for forecasting time
  series." *Empirical Economics*. Forecast averaging over a break-date confidence set. (Grade A)
- Huang, T., Hardie, B. G. S., and Fader, P. S. (2004). "A dynamic changepoint model for new product
  sales forecasting." *Marketing Science*. Evolving changepoint for trial and repeat. (Grade A)
- Neelamegam et al. (2026). "Comprehensive insights into bitemporal databases: a PRISMA-guided
  systematic literature review." *Journal of Data, Information and Management*. 54 papers,
  1995–2026; valid and transaction time; supply chain applications; gaps in database and machine
  learning support. (Grade A)
- Papadakis, G. et al. (2025). "Heterogeneity in entity matching." arXiv:2508.08076. Cited only for
  item-catalogue matching being a well-benchmarked problem. (Grade A)

**Standards and official**

- GS1, *GTIN Management Standard*, Release 1.1, ratified September 2023. Rules 2.3 (declared net
  content), 2.4 (dimensional or gross weight change over 20%), 2.8 (pack/case quantity).
  `https://ref.gs1.org/standards/gtin-management/1.1.0/` (Grade B)
- GS1, *General Specifications*, current edition, GTIN allocation and non-reuse. (Grade B)
- GS1, *Global Data Model* and *GDM Attribute Implementation Guideline*, current releases. (Grade B)
- GS1, *GDSN* and *EPCIS 2.0* service and standard documentation. (Grade B)
- ISO 8601, date and time representation. (Grade B)
- ISO/IEC 9075-2:2011 (SQL:2011), system-versioned tables. (Grade B)
- U.S. Food and Drug Administration, DSCSA; U.S. Food Traceability Rule (FSMA 204); EU Falsified
  Medicines Directive; EU general food law traceability. Cited as obligations that a broken
  identifier chain can breach. (Grade B)

**Industry, analyst, and vendor**

- SAP, supersession and data-realignment documentation for SCM and service parts planning, `https://help.sap.com/`. (Grade D; the cardinality vocabulary and the raw-data-unchanged property are the transferable content)
- Oracle, *Overview of Forecasting Using Supersession Relationships*, Fusion Cloud Demand Management 26d. (Grade D)
- Infor, supersession documentation, Supply Chain Planning 2026.10. (Grade D)
- AWS Supply Chain, "Improving demand planning for products with varying lifecycles," November 2023; product lineage, `alternate_product_qty`, surrogate history. (Grade D)
- SAP Community, "Navigating Product Lifecycle Management in SAP IBP for Demand," July 2024; SAP Learning, "Planning Product Lifecycles." 120-day change log; reference-product weighting. (Grade D)
- DemandLabs, "The forecast is in eaches and the order goes out in cases," undated, accessed 2026; clean-multiple conversion errors, effective-dated conversions. (Grade D)
- Scout, "Case pack and selling units in CPG," glossary entry, accessed 2026. (Grade D)
- Circana, "Equivalized (EQ) Sales," CPG dictionary, accessed 2026. (Grade C, industry metric definition)
- Nestlé Brazil, demand planning with SAS Forecast Server, conference paper, 2019. (Grade C, company case with no quantified outcome)
- demandplanning.net, "How to Solve the PLM Trap in Demand Forecasting," undated, accessed 2026; the 100% default predecessor weight and the missing self-mapping pair. (Grade D)
- Martin Fowler, "Bitemporal History," 2021, `https://martinfowler.com/articles/bitemporal-history.html`. (Grade C, practitioner explanation)
- XTDB documentation, "Bitemporality," v1 docs; as-of versus as-at semantics. (Grade D)

**Note on citations.** Vendor documents are cited for what they describe, not for outcomes, and are
graded D throughout. The 2019 Huang et al. figure is reported as the paper states it; the manuscript
consulted is the accepted version. No source found in this research quantifies the forecast-error
cost of a pack change in a named company, so the worked example is constructed and labelled as
illustrative rather than sourced. Where a claim about industry practice is directional only, the
text says so.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_master-data.md`](02_master-data.md)
- Related: [`08_data-quality-dimensions.md`](08_data-quality-dimensions.md)
- Next topic: [`10_source-selection-trust.md`](10_source-selection-trust.md)
- Related: [`../01_supply_chain_foundations/03_network-entities.md`](../01_supply_chain_foundations/03_network-entities.md)
- Related: [`../01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md`](../01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
