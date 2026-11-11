---
case_id: "CASE-04-01-04"
case_title: "Planogram Compliance"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "merchandising lead"
source_index: "INDEX.md"
review_trigger: "Planogram version, detector sensitivity or specificity, labour rate, or space elasticity changes"
---

# Planogram Compliance

Every number in this case is **ILLUSTRATIVE**. The store count, bay count, audit minutes, labour rate,
detector sensitivity and specificity, facing allocation, space elasticity, and dollar values are
teaching inputs constructed for this manuscript. They are not measurements of any retailer, camera
vendor, category, software product, or deployment, and no financial or sales result is claimed.

Where a quantity corresponds to established retail practice, the practice is described generically:
a planogram is a versioned layout document, a facing is one unit-width of a SKU on a shelf, a reset
is a scheduled re-merchandising event, and share of shelf is facings weighted by commercial role.
The parameter values attached to those practices here are constructed.

The arithmetic in section 6 is recomputed line by line and reproduces with a calculator. Everything
in sections 7 to 13 depends on that arithmetic and inherits its status: conditional conclusions
about a designed decision, not observed outcomes. Section 11 states `not-measured` and names the
evidence that would be required to replace a conditional conclusion with a measured one.

## 1. The One-Line Framing

**A compliance score becomes a business quantity only after it is multiplied by a labour minute and
a dollar, and only if the detector that produced it still has its validated error rates in the store
where the fix will happen.**

- The decision is small and weekly.
- A store has a fixed number of merchandising minutes this wave.
- It has a list of shelf bays, each bound to a planogram version.
- Some bays do not match that version.
- A crew can bring a bay back into agreement, at a cost in minutes.
- Those minutes cannot be spent on a second bay.

The question is therefore not "which bays are non-compliant" in the abstract.
It is "which bays, among those the crew can actually reach before the wave closes, return the most
value per minute spent, given that the evidence about each bay is itself fallible."

Three framings compete for the merchandising lead's attention. Only the third survives the labour
budget.

| Framing | Question it answers | Why it fails alone |
|---|---|---|
| Quality | What percentage of bays match the planogram? | A percentage does not tell a crew where to start on Monday |
| Model | How accurate is the shelf-image classifier? | Accuracy can improve while cost per corrected bay worsens |
| Decision | Corrected bays per labour hour at an acceptable false-fix rate | Requires prevalence, capacity, and dollars in one expression |

The quality framing is the number printed on the operations dashboard, and stores learn to manage
it. It answers a question nobody has to act on.
The model framing is measurable and defensible, and it is nearly unrelated to the decision, because
a classifier can raise its F1 by moving a threshold while the queue it emits becomes less executable.
The decision framing forces two quantities into one expression that the other two keep apart: the
prevalence of non-compliance in *this* store *this* week, and the labour minutes available to fix
anything at all.

The reusable pattern is **evaluation-mismatch**: the metric convenient to compute offline is not the
metric that determines whether the operating decision creates value.
Planogram compliance is an unusually clean instance, because three candidate quality metrics —
facing match, share-of-shelf deviation, and position error — are all computable from the same
observation and rank the same bays differently.

The takeaway is deliberately narrow. A compliance figure is a measurement. The fix list is a
decision. The conversion between them depends on labour capacity, detector error rates, and the
dollar value of a facing. Change any one of the three and the correct fix list changes while the
compliance percentage stays exactly where it was.

## 2. Context & Background

### 2.1 What a planogram actually specifies

A planogram is a versioned specification of what a shelf bay should look like. For each bay it
states six things, and a compliance programme must handle all six or it will measure something else.

| Element | What it fixes | Typical failure when ignored |
|---|---|---|
| Assortment | Which SKUs belong in the bay | A delisted SKU is scored against a current planogram and looks like a breach |
| Facings | How many unit-widths each SKU receives | Counts drift silently; nobody notices until a reset |
| Order | Left-to-right sequence of SKU blocks | Adjacent block swaps look compliant to a count-only check |
| Level | Which shelf each block occupies | Eye-level placement drives pickup; a level error is invisible in totals |
| Depth | Units behind each facing | Depth errors cause phantom out-of-stocks, not facing errors |
| Effective date | When the version becomes binding | Two versions are legitimately live during a reset window |

The version identifier matters more than people expect. The same physical bay can be compliant with
version 4 and non-compliant with version 5. During a reset window both versions are correctly live
in different stores at the same time, because waves are staggered to fit labour and label supply.

A programme that compares every bay to "the current planogram" without binding the version will
manufacture exceptions in exactly the stores at the leading edge of the reset wave. Those stores
will then be penalised for being ahead of schedule.

### 2.2 Three quantities that get conflated

**Facings** are a count of unit-widths. A SKU with four facings occupies four positions side by side
and presents four units to the shopper. Facings are integers, and they are bounded by the physical
width of the fixture: the sum of facing widths over all SKUs in a bay cannot exceed usable shelf
width. A record claiming otherwise is an observation error, not a compliance finding.

**Share of shelf** is facings weighted by commercial importance, usually proxied by the SKU's share
of category sales or by its assigned category role (destination, routine, convenience, seasonal).
Two bays can have identical facing counts and very different shares of shelf if the SKUs differ.

**Position** is where a block sits: which shelf level, and how far along the shelf from the left-hand
fixture edge. Position governs sight-lines and pickup convenience, which is why planograms specify
order and level and not only counts.

Conflating them produces three different mistakes. Treating facings as share of shelf makes a
high-velocity SKU with four facings look equivalent to a slow SKU with four facings. Treating share
of shelf as facings makes a cosmetic count error on a tail SKU look like a commercial emergency.
Treating position as facings makes a block swap on the middle shelf invisible.

### 2.3 Resets, waves, and drift

A **reset** (also called a relay or re-merchandising event) is the scheduled work of making a bay
match a new version: pulling product forward, moving blocks, changing facings, replacing shelf-edge
labels. Resets are batched into **waves**, typically every four weeks, because labour is cheaper
when a crew does many adjacent bays in one visit and because printed labels and point-of-sale
material arrive per wave.

Between waves, bays drift. The mechanisms are ordinary and each has a distinct signature.

| Drift mechanism | What an observer sees | Why it looks like a compliance failure |
|---|---|---|
| Sell-down | Gaps deepen on the fastest seller | A gap reads as a missing facing |
| Gap fill | Staff put any available SKU in the hole | Assortment breach with plausible-looking facings |
| Promotion | A featured item gets an extra facing | Facings exceed plan on one SKU, short on another |
| Delist | A discontinued line leaves a permanent hole | Count matches a superseded planogram version |
| Depth error | Facings correct, stock behind them short | Compliant by facing count, out of stock by shopper experience |
| Picking role | A bay also serves online order picking | Deliberate deviations for pick-face access |

Drift is why a compliance programme exists at all. If bays only changed at resets, one inspection
per wave would suffice and the problem would reduce to scheduling. Because bays drift continuously,
compliance is a **stock**, not an event: high immediately after a reset, decaying toward whatever
equilibrium the store's replenishment habits support.

This has a direct consequence for measurement. A single compliance figure averaged across a wave
mixes bays audited on day one with bays audited on day twenty-seven, so it reports the audit
schedule as much as the shelf. Section 6 derives a decay term that converts a mid-wave observation
into an average-over-wave quantity, and section 9 requires the audit day to be recorded per bay
because without it the number cannot be interpreted.

### 2.4 Three inspection technologies

The choice among these is the actual decision, and each has a different cost, error structure, and
blind spot.

**Walk audit.** A person stands in front of the bay with a handheld device and counts facings against
the printed planogram. Accurate when done carefully, expensive per bay, and subject to drift within
the auditor: the same person counts the same bay differently at 09:00 and at 17:00, and two auditors
disagree systematically on partially occluded blocks.

**Shelf-image capture.** A fixed or handheld camera photographs the bay; either a human reviews the
photograph or a vision model counts facings and identifies SKUs. Cheap per bay and censused rather
than sampled, but limited by occlusion, lighting, fixture depth, and visual similarity of packaging
within a category. The error rates are not uniform across categories: breakfast cereal and pet food
are not equally hard to read.

**Inferred compliance.** No direct observation: infer a probable layout breach from sales anomalies,
out-of-stock scans, or replenishment exceptions. Cheapest, weakest, and prone to confusing a
planogram design error with an execution error, because both show up as a SKU selling below
expectation.

A defensible programme usually uses two of the three. The photo census establishes coverage and
generates candidate exceptions; the walk audit establishes a prevalence estimate on a probability
sample and validates the photo path's error rates. Neither alone is sufficient: the census cannot
estimate its own error rate, and the sample cannot cover 380 bays at a tolerable labour cost.

### 2.5 The economic logic of a bay

A bay has finite width. The planogram's facing allocation is a bet about how much space each SKU
deserves. When the bet is wrong, the bay loses sales for a reason no compliance work can fix.

This is the single most important distinction in the case, and section 6 makes it quantitative. A
bay can be fully compliant and underperform because its planogram assigns too few facings to its
fastest seller. A bay can be badly non-compliant and cost nothing because every deviation is on a
slow-moving tail SKU whose facings are commercially irrelevant.

Any programme that counts deviations without weighting them will spend its budget in the wrong
place. The weighting used here is a constant-elasticity space response, which is the simplest
assumption that produces a declining marginal value per facing. Section 6 states its limits: the
elasticity is borrowed, not measured, and the case therefore reports value in a form that can be
re-scaled if the reader's own space test returns a different elasticity.

### 2.6 Organisational context

Three parties own different parts of the problem, and each is measured on something different.

| Party | Owns | Measured on | Failure it will resist |
|---|---|---|---|
| Category team | Planogram content and facing allocation | Category sales, margin | Escalations that admit the layout is wrong |
| Store team | Execution and labour | Compliance percentage, labour cost | Queues longer than the crew can clear |
| Audit function | Measurement | Coverage, timeliness | Slower, better-instrumented audits |

A programme that raises the compliance number without raising sales will be welcomed by two parties
and resisted by the third. A programme that raises measured compliance by changing the measurement
rather than the shelf will be undetectable for at least one wave. Section 8 therefore freezes three
things before the first wave — the metric definition, the version binding, and the audit instrument —
and records all three in the decision log so a later comparison cannot quietly change its own baseline.

### 2.7 The omnichannel edge

The same SKU appears in an e-commerce assortment with a different presentation rule. A bay that
looks non-compliant may be deliberately serving a click-and-collect pick path, where depth and
pick-face accessibility trade off against shopper-facing facings.

A compliance programme that does not know which bays carry a picking role will generate false
exceptions in exactly those bays. Those false exceptions concentrate in the stores that do the most
online fulfilment, so the programme will appear to find that high-fulfilment stores are worse
executors — a conclusion that is an artefact of the measurement, not a finding about the stores.

Section 12 treats this as its own failure mechanism with its own discriminating check, because no
improvement to the detector can fix it: the detector is reporting accurately that the shelf differs
from the shopper planogram. The error is in the comparison, not the observation.

## 3. The Business Question

**Question.** Given, for each bay in a store, a planogram version effective on the audit date and one
or more observations of the bay (photograph, walk audit, or both), which bays should the store's
merchandising crew reset in the current wave, and which observations must be rejected or deferred
because they cannot support a decision?

- **Owner:** the merchandising lead for the cluster, accountable for both the compliance number and
  the reset labour spent to move it.
- **Trigger:** a scheduled wave opens (four-weekly), or a store-level exception exceeds a threshold,
  or a planogram version change makes a subset of bays newly non-compliant by definition.
- **Unit of decision:** one bay in one store for one wave. Not a store, not a SKU, not a photograph.
- **Available actions:** fix now; defer with a reason code; reject the observation and re-observe;
  escalate to the category team as a design error; take no action.
- **Objective:** maximise expected value of corrected facings per labour minute, subject to a fix
  capacity and a stated tolerance for wasted fixes.
- **Hard constraints:** no observation timestamped after the decision may influence the ranking; the
  queue may not exceed measured capacity; every fix must cite a planogram version and a named
  operator; a rejected observation may never be counted as compliant.
- **Measured in:** bays fixed per labour hour, true non-compliant bays corrected as a fraction of
  true non-compliant bays, false-fix rate, and dollars of estimated sales-at-risk addressed.
- **Scope here:** one cluster, 28 stores, 380 bays per store, 6 SKUs per bay, one wave, with a photo
  census plus a walk-audit probability sample. ILLUSTRATIVE.
- **Out of scope:** vendor selection, camera procurement, supplier contract findings, and any claim
  of realised sales uplift. Uplift needs a controlled design this case does not execute.

The question contains a trap that must be named before analysis begins. The phrase "which bays are
non-compliant" invites a threshold on a continuous score, and the threshold that maximises agreement
with a human auditor is not the threshold that maximises corrected bays per minute.

So the case answers the selection question twice: once under a classification objective, once under
the capacity-constrained value objective. The two orderings disagree on a material number of bays,
and where they disagree the capacity-constrained ordering governs, because it is the only one the
crew can execute.

A second trap is the word "corrected". A bay is corrected when a later observation, taken after the
fix and before the next drift interval, matches the planogram within the declared tolerance. If the
programme never takes that second observation, "corrected" is an assumption about the crew's
workmanship and the entire value calculation inherits it. The design keeps a verification observation
at a sampled rate precisely because the value arithmetic is otherwise unfalsifiable.

A third trap is denominator choice, and it silently decides the answer. Four candidate denominators
exist, and they give four different numbers from one dataset.

| Denominator | Definition | What it answers | What it hides |
|---|---|---|---|
| All bays | Every bay in the store | Coverage-weighted quality | Nothing about severity |
| Observed bays only | Bays with a usable observation | Quality of what was seen | Rejections, which are not random |
| Flagged bays | Bays the detector called non-compliant | Precision of the queue | Recall, and the missed bays |
| Facings | Facings matching, over facings planned | Granular quality | Bays, which are the unit of work |

The case uses **bays with a usable observation and a version binding** as the primary denominator,
reports the all-bays figure as a coverage-adjusted companion, and reports facings as a second
measure only. The reason is operational: the crew fixes bays, one visit each, so a bay is the unit
whose count has to be reconciled against capacity. A facings-weighted number can improve while the
number of fixable bays is unchanged, and that improvement is not deployable.

## 4. Assumptions & Constraints

Each assumption carries an owner, a falsification test, and a fallback that keeps the decision safe
when the test fails. The table is the contract the rest of the manuscript relies on.

| ID | Assumption or constraint | Why needed | Falsification test | Fallback |
|---|---|---|---|---|
| A-01 | Each observation binds to exactly one effective planogram version | Facings are meaningless without the version compared against | List all versions effective on the audit date for a sample of bays; count bays with two or more | Mark ambiguous, exclude from primary numerator, count separately |
| A-02 | Observation timestamp precedes decision timestamp for every record used | A later observation can encode the fix it is meant to justify | Re-rank using only records with observed_at < decided_at; compare orderings | Drop records, re-rank, log the change |
| A-03 | Facing counts are integers consistent with fixture width | An impossible layout signals observation error, not non-compliance | Check summed facing widths against usable shelf width per bay | Quarantine the observation, route to re-capture |
| A-04 | Detector sensitivity and specificity transport to the ranked stores | Queue length and precision both depend on them | Stratify the labelled set by cluster, fixture type, lighting; test heterogeneity | Use per-cluster error rates, or gate the store off the photo path |
| A-05 | Non-compliance prevalence clears the level at which the queue is informative | Below that level a specific detector emits mostly false positives | Estimate prevalence from the walk-audit sample before enabling triage | Disable triage for that store; walk audits or inferred path only |
| A-06 | Wave capacity is known within about ten percent | Capacity is the binding constraint on queue length | Compare planned against completed fix minutes for two waves | Cap the queue at the lower bound of measured capacity |
| A-07 | A fix persists for at least the drift half-life | Value is booked per corrected bay, not per corrected moment | Re-observe a sample of fixed bays at the next wave | Discount value by the observed reversion rate |
| A-08 | Dollar value of a corrected facing is bounded by a stated elasticity | Without a value, ranking collapses to counting deviations | Compare the assumed elasticity against a category space test if one exists | Rank by facings restored, report no dollars, and say so |
| A-09 | Rejected observations are not treated as compliant | Otherwise rejecting evidence improves the number | Trace every rejection to a disposition code and to the deferral list | Force rejection to route into deferrals |
| C-01 | Queue length may not exceed the wave's fix capacity | A longer queue produces silent, undocumented deferral | Sum queued fix minutes; compare with available minutes | Truncate at capacity; log the deferred tail with rank positions |
| C-02 | No action may be taken on a bay whose version binding is ambiguous | A reset against the wrong version creates a real breach | Sample audit of ambiguous bays after the wave | Defer to next wave, escalate to category team |
| C-03 | Fix authority stays with a named operator | An unattributed change cannot be reversed or explained | Attempt to trace a random fix to an operator identifier | Block the action until attribution exists |

The binding assumptions are A-04, A-05, and A-08.

A-04 and A-05 are the standard evaluation-mismatch pair. A detector validated on a curated photo set
is being used to rank stores whose packaging mix, fixture depth, and lighting differ from that set.
The error rates that made the queue work in validation are exactly the quantities most likely to
have moved, and they move in a direction that hurts: specificity falls first when categories are
visually similar, and falling specificity floods the queue with compliant bays.

A-08 is the assumption most often left unstated and most consequential. Without a dollar value per
facing, ranking collapses to "most facings wrong first". That rule systematically prioritises wide
bays with trivial cosmetic deviations over narrow bays with one commercially important SKU at zero
facings. It is not a neutral default; it is an implicit and usually wrong value judgement.

C-01 interacts with A-06 in a way that punishes iteration. If capacity is overestimated by ten
percent, the queue is ten percent too long; the crew fixes what it can in score order and the tail is
deferred without a reason code. The observed compliance improvement then understates the queue's true
value, and the natural response — lengthen the queue further — deepens the understatement. Section 8
breaks this loop by making queue length a function of a conservative capacity estimate and by logging
every deferred bay with its rank position so the tail is auditable rather than invisible.

The most decision-reversing risk in the case is temporal. An observation or a planogram version that
only became available after the decision time can make a bay look non-compliant, or compliant, in a
way nobody could have known when the crew was scheduled. The fallback never varies: drop the record,
re-rank, record that the ranking changed. A ranking computed with a later record and then relabelled
as if it had been computed earlier is not a conservative approximation. It is a different decision
wearing the same name.

Two constraints are organisational rather than technical, and both are load-bearing.

First, the store team must be able to see and contest the ranking. An operator who believes a bay is
compliant and is told to reset it needs an appeal path that records the disagreement, because
unresolved disagreements are the main source of the false-fix cost that section 6 prices at $3.71
per attempt.

Second, the category team must be able to receive design-error escalations. A programme that converts
design errors into execution failures will consume the entire fix budget on bays that cannot be fixed
by fixing them, and it will report a stable compliance percentage while doing so.
