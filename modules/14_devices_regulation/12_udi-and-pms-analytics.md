# Unique Device Identification and Post-Market Surveillance Analytics

> **Module 14 — Medical Devices, SaMD, and Regulation**
> Topic 12 of 12
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 05, 08, and 10
> Version 2026.3

## 1. Why this topic exists

A device is cleared, launched, and implanted. Then something goes wrong. A hospital reports that a
catheter hub cracked under pressure, or that a valve frame migrated, or that a pump delivered ten
times the programmed dose. Someone has to answer three questions quickly: is this happening more
often than it should, which devices are affected, and who has them inside their patients.

Most organisations cannot answer any of the three from their own systems. They can count
complaints. They cannot divide by anything meaningful, because the denominator lives in a
different system with a different definition. They cannot identify the affected units, because the
hospital recorded "catheter" and the lot number is on a sticker in a paper file. And they cannot
find the patients, because the implant record names a model that was renamed two years ago.

That is the problem this topic is about. It is not a regulatory paperwork problem, although it
produces paperwork. It is a data linkage problem, and the regulatory obligations are what make the
linkage worth building.

Two systems exist to solve it. The **Unique Device Identification (UDI)** system gives every device
a globally unique identifier and, for most devices, a production identifier that names the lot or
serial number. The **post-market surveillance (PMS)** system collects events, complaints, and
recalls. Neither is useful alone. UDI without event data is a catalogue; event data without UDI is
a pile of anecdotes. The analytics is the join.

Three decisions depend on that join, and each is blocked in a different way.

The first is a field safety corrective action. A supplier changes a polymer in a connector. Six
months later, complaints about connector fracture rise. The manufacturer must decide whether the
problem is confined to one lot, one production period, or the whole design. Getting that wrong in
one direction triggers an unnecessary recall of a hundred thousand units; getting it wrong in the
other leaves a defective lot in use.

The second is a periodic safety report. Under the European Union Medical Device Regulation, a
manufacturer of a Class IIb or Class III device must produce a periodic safety update report with
rates, not counts, and must estimate the size of the population using the device. A report that
says "68 complaints were received" without a denominator is not a safety report; it is a log.

The third is a regulator comparing devices. A reviewer looking at two competing devices needs rates
per unit implanted or per patient-year, standardised for case mix. Raw counts favour the device
that sold less.

The reason this is a data-science topic rather than a compliance topic is that all three decisions
fail in the same way: the numerator and the denominator are collected by different people, for
different purposes, with different definitions, and joined on a key that is often missing. Every
statistical method in the world will give a confident answer to the wrong question if the join is
wrong. Most of the work is in the join.

> **Note:** Regulatory material in this file is educational and is not legal advice. Reporting
> obligations are jurisdiction-specific and time-bound, and a missed deadline is a separate
> violation from the underlying device problem. Work with a regulatory professional on any actual
> decision.

## 2. Learning outcomes

By the end of this topic you will be able to:

1. **Parse** a UDI into its device identifier and production identifier components for the three
   accredited issuing agencies, and **validate** a GS1 check digit.
2. **Choose** an exposure denominator for a given device and question, and **justify** the choice
   against at least two alternatives.
3. **Compute** a complaint rate with a confidence interval and a rate ratio, and **explain** why the
   raw count comparison misleads.
4. **Correct** an observed rate for capture change, reporting lag, channel inventory, and case mix,
   and **state** the direction each correction pushes.
5. **Detect** a lot-level signal with a rate ratio and a CUSUM chart, and **explain** why the
   aggregate can hide it.
6. **Trace** a recall or field safety corrective action to affected units through UDI and GUDID.
7. **Distinguish** the US and EU reporting triggers and timelines, and **state** what starts the
   clock.
8. **Design** a PMS analytics specification that a quality system can audit.

## 3. The core mental model

Post-market analytics is the intersection of three graphs that are maintained by three different
parts of the world.

```text
   DEVICE GRAPH                 EVENT GRAPH                 EXPOSURE GRAPH
   (manufacturer,               (complaints, MDRs,          (sales, shipping,
    GUDID, EUDAMED)              recalls, registries)        implants, procedures)

   DI  ------> model            event ----> DI?             shipment ----> DI
    |            |                |           |                |            |
    |            |                |           |                |            |
   lot ----> units              lot? ------> PI?             units ----> period
    |            |                |           |                |            |
   unit ----> patient?          patient?                   patient-years
        \           |                |           /                |
         \          v                v          /                 v
          +------>  JOIN on DI, and on PI where it exists  <-------+
                             |
                             v
                   RATE = events / exposure
```

Each graph has its own key, its own definition of a unit, and its own error modes. The device graph
is keyed on the device identifier (DI) and, below it, the lot or serial. The event graph is keyed on
whatever the reporter wrote down, which is often a brand name. The exposure graph is keyed on the
DI and the shipping period, and counts units sold rather than units used.

The join is the whole problem, and it fails in four distinct ways.

**Numerator failure.** The events are not all captured, and capture changes over time. A new
complaint portal, a new distributor, a new training module, or a new vigilance officer changes the
count without changing the device. A rise in the numerator may be a rise in reporting.

**Denominator failure.** The exposure is not what you think it is. Units sold include units in
distributor inventory, units in hospital stock, and units expired and discarded. For a reusable
device, units sold is the wrong unit entirely: the exposure is procedures. For an implant, the
exposure that matters for a late failure is patient-years, not units.

**Time failure.** Events and exposures are dated differently. A complaint is dated when it is
received; the event happened weeks earlier, and the device was implanted months before that. A
recall is dated when it is classified; the correction started earlier. Comparing a numerator dated
by receipt with a denominator dated by shipment creates a step change whenever either process
changes.

**Linkage failure.** The event cannot be attributed to a device, or is attributed to the wrong one.
The lot was not recorded. The brand name changed. The same device appears under two DIs because it
was repackaged. Two reports of one event are counted twice.

```text
   what the dashboard shows        what actually happened
   ------------------------        -----------------------
   complaints up 66%               units up 30%, so rate up 28%
   rate up 28%                     capture up 20%, so rate up 6%
   rate up 6%                     reporting lag on recent months, so rate up ~15%
   flat                            one lot at 8x the baseline
```

The last line is the point. A well-built aggregate can be flat, correctly, while a serious
lot-specific problem is present. Aggregation is not a substitute for stratification, and the
stratification that matters is almost always by lot, by model, by site, and by period.

### Why rates and not counts

A count answers "how many". A rate answers "how often per unit of use", which is the question that
supports a decision. The two diverge whenever the denominator changes, and the denominator almost
always changes: the product launches, the sales territory expands, a competitor exits, a pandemic
changes procedure volumes.

The rate is not the end of the analysis either. A rate computed on a changing denominator is only
comparable if the denominator is defined the same way in both periods. Most of the analytical work
in this topic is making that true.

## 4. Definitions and vocabulary

| Term | Working definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Unique Device Identification (UDI) | A globally unique identifier for a device model plus a production identifier for the unit | Used to mean the barcode | GS1, HIBCC, or ICCBBA syntax |
| Device identifier (DI) | The static part of the UDI, identifying the model and version | Confused with the catalogue number | GTIN-14, HIBCC DI, or ICCBBA product code |
| Production identifier (PI) | The dynamic part: lot, serial, expiry, manufacturing date | Omitted because "the DI is enough" | Alphanumeric |
| Basic UDI-DI | The EU identifier for a device family, used in EUDAMED and certificates | Confused with the US DI | EU MDR Article 27 |
| GUDID | The US Global Unique Device Identification Database | Called an API; it is a database with an API in front of it | FDA |
| EUDAMED | The European database on medical devices | Assumed to be fully live | European Commission |
| Global Trade Item Number (GTIN) | The GS1 identifier used as a DI, 14 digits | Treated as a 12-digit UPC | GS1 |
| Health Industry Business Communications Council (HIBCC) | A UDI issuing agency using a plus-prefixed syntax | Assumed to be US-only | HIBCC standard |
| ICCBBA | The issuing agency for blood and tissue products, using ISBT 128 | Overlooked for human-derived devices | ISBT 128 |
| Medical Device Reporting (MDR) | The US regulation at 21 CFR Part 803 requiring adverse event reports | Confused with the EU Medical Device Regulation, also abbreviated MDR | 21 CFR 803 |
| MAUDE | The FDA database of device adverse event reports | Treated as a complete census | FDA |
| Recall | Removal or correction of a device that violates the law | Confused with a market withdrawal | 21 CFR 7 and 806 |
| Correction | Repair, modification, relabelling, or inspection of a distributed device | Used interchangeably with recall | 21 CFR 806 |
| Field safety corrective action (FSCA) | An EU action to reduce or prevent a serious incident | Treated as optional communication | EU MDR Article 2(68) |
| Field safety notice (FSN) | The communication sent to users about an FSCA | Treated as the action itself | EU MDR |
| Complaint | Any written or oral communication alleging a deficiency | Counted per contact rather than per event | ISO 13485 clause 8.2.2 |
| Exposure denominator | The unit of use against which events are counted | Substituted by units sold without comment | Units, procedures, or patient-years |
| Patient-year | One patient exposed for one year | Confused with a calendar year | Years |
| Reporting lag | The delay between an event and its appearance in the data | Ignored, biasing recent periods low | Days |
| Right truncation | The incompleteness of the most recent period | Mistaken for a decline in events | Statistical property |
| Capture rate | The proportion of true events that enter the system | Assumed constant | Proportion |
| Signal | A new or changed association that warrants investigation | Treated as proof of causation | Not a unit |
| Signal detection | Systematic screening for signals | Confused with hypothesis testing | Method |
| CUSUM | Cumulative sum control chart for detecting a shift | Tuned to the data it monitors | Standard deviations |
| Standardised rate | A rate adjusted to a reference case mix | Confused with a crude rate | Per 1,000 or per 1,000 patient-years |
| Class I recall | The most serious recall classification, for reasonable probability of serious harm | Confused with a Class I device | FDA classification |
| PSUR | Periodic safety update report under EU MDR Article 86 | Confused with the US periodic report | EU MDR |
| PMCF | Post-market clinical follow-up | Confused with PMS generally | EU MDR Annex III |
| GMDN | Global Medical Device Nomenclature, used in GUDID | Confused with EMDN | GMDN |
| EMDN | European Medical Device Nomenclature, used in EUDAMED | Confused with GMDN | EMDN |

**The two MDRs.** The abbreviation MDR means "Medical Device Reporting" in the United States and
"Medical Device Regulation" in the European Union. Both appear in the same documents and both are
load-bearing. Spell them out on first use in every document you write.

**Class I recall is not Class I device.** A Class I device is low risk; a Class I recall is the most
serious. A Class I device can have a Class I recall if the failure could cause serious harm. The
collision causes real errors in dashboards.
## 5. The mechanics, step by step

The procedure builds a PMS analytics capability that can answer the three questions from section 1.
It is written for an implantable or single-use device with a UDI; the adaptations for reusable
devices and for software are noted where they change the method. Each step states its input, its
output, the reason it exists, the procedure, its branch points, and its edge cases. Three decision
tables cover the steps where more than two outcomes are possible: the choice of denominator, the
response to a signal, and the reportability decision.

### Step 1 — Define the analytic question and the unit of analysis

**Input.** The safety question, the device description, and the decision that will be taken.

**Output.** A question statement naming the device, the population, the period, the outcome, and
the unit of analysis.

**Rationale.** "Are complaints increasing" has no answer until the unit is fixed. The same data
support a count of reports, a count of events, a count of affected patients, and a count of affected
units, and they give different answers. The unit must match the decision: a recall decision is
about units, a safety report is about patients, a reliability analysis is about device-years.

**Procedure.** (1) Write the decision. (2) Name the outcome precisely: failure mode, severity, and
whether it is reported or observed. (3) Name the population and the period. (4) Name the unit of
analysis. (5) Name the comparison: prior period, other lots, class benchmark, or pre-specified
threshold. (6) Record the date the question was fixed, because the analysis will later be tempted
to drift.

**Branch points.** A question about a trend needs a time unit and a stable definition across
periods. A question about a lot needs a lot-level denominator, which often does not exist. A
question about a class needs standardisation, which needs covariates that may not be collected.

**Edge cases.** A question written after the data were seen has already been shaped by them. A
question that mixes a safety outcome and a business outcome will be answered by whichever is easier
to measure. A question that names a threshold without naming who set it invites the threshold to
move once the answer is known.

### Step 2 — Build the device identity spine

**Input.** The device master list, the label artwork, the GUDID and EUDAMED records, and the
manufacturing bill of materials.

**Output.** A table with one row per DI, carrying the model, the size, the basic UDI-DI, the product
code, the GMDN and EMDN terms, the launch and retirement dates, and the predecessor and successor
DIs.

**Rationale.** Every join in the rest of the procedure goes through this table. If it is wrong, the
analysis is wrong in a way that no statistical method will reveal. Devices are renamed, repackaged,
and re-registered; a model that changed its brand name in 2023 has two DIs and one clinical
identity, and the event history must be continuous across both.

**Procedure.** (1) Extract the DI and attributes from GUDID. (2) Map each DI to a commercial model
and a size. (3) Record the launch and retirement dates. (4) Record the predecessor and successor
DIs for any rename or repackage. (5) Map to GMDN and EMDN. (6) Record the basic UDI-DI for EU
purposes. (7) Validate: every DI in the complaint system must appear, and every DI in the table
must have a launch date.

**Branch points.** A device with many sizes has many DIs and one clinical identity; the analysis
must be able to aggregate to a model without losing the size, because size is a confounder. A device
with a software component has a hardware DI and, in some jurisdictions, a separate identity for the
software version, so a version-specific safety question needs a version-specific spine. A device
that is OEM-manufactured for several brands has one clinical identity and several DIs belonging to
different legal manufacturers, and the event history is split across them.

**Edge cases.** A retired DI that was never removed from the catalogue will attract events from
replacement devices if the model string matches loosely. A
device with a software component has DIs for the hardware and, in some jurisdictions, a separate
identity for the software version.

### Step 3 — Choose and construct the exposure denominator

**Input.** The unit of analysis from step 1, the device identity spine, and the shipping, sales, and
implant data available.

**Output.** An exposure table with a value per DI per period, plus a documented statement of what
the denominator does and does not include.

**Rationale.** The denominator is where most PMS analyses fail, and the failure is usually silent.
Units sold is available and almost always wrong in a specific, quantifiable way. The choice must be
justified against the alternatives and the bias direction stated.

**Decision table — choosing a denominator**

| Denominator | Correct when | Biased when | Direction of bias |
|---|---|---|---|
| Units sold | Single-use device, short channel, stable inventory | Distributor stocking or destocking | Overstates exposure when inventory grows, understating the rate |
| Units shipped to customers | Direct sales, no distributor layer | Hospital stock held on consignment | Overstates exposure by the consignment stock |
| Units implanted or used | Implant registries or procedure coding exist | Registry coverage is partial | Understates exposure if coverage is incomplete, overstating the rate |
| Procedures | Reusable or reprocessed devices | Procedure coding does not identify the device | Unknown, because the device may be one of several used |
| Patient-years | Late failures, survivorship questions | Follow-up is short relative to the failure time | Understates late hazard if follow-up is truncated |
| Active devices in the field | Software or capital equipment | Decommissioned units are not removed from the count | Overstates exposure, understating the rate |

**Procedure.** (1) Select the denominator from the table and write the justification. (2) Reconcile
sales against shipping against implant records for one period, and quantify the gap. (3) Convert to
the chosen unit, for example by subtracting channel inventory. (4) Record the conversion
assumption. (5) State the direction of the residual bias.

The reconciliation in step 2 is the part that is usually skipped and usually decisive. Take one
quarter and build a bridge from units manufactured to units implanted: manufactured, minus
scrapped, minus finished goods in the plant, equals units shipped; shipped, minus distributor
inventory change, minus hospital consignment change, equals units sold to the provider; sold to the
provider, minus stock on the shelf, minus expired and returned, equals units used. Every term in the
bridge is a place where the denominator can be wrong, and the size of the gap tells you whether the
choice of denominator matters at the precision of your decision.

**Edge cases.** A device sold in a tender-driven market has lumpy distributor stocking, so the
inventory correction dominates. A device with a consignment model has exposure that lives on
hospital shelves. A device with a short shelf life has expiry losses that must be subtracted or the
exposure is overstated.

**Branch points.** A device with no implant registry needs a proxy, and the proxy must be validated
once against a sample of hospital records. A device sold in several countries needs per-country
denominators, because Channel and reimbursement structures differ.

### Step 4 — Link events to devices

**Input.** The complaint system, the MDR and vigilance feeds, the recall database, and the device
identity spine.

**Output.** An event table with a DI, a lot or serial where available, and a linkage confidence
flag.

**Rationale.** An event without a device is a narrative. The linkage is usually partial: the UDI is
recorded in a minority of MDR reports, and the lot in fewer. The analysis must distinguish events
that are linked with certainty from those linked by inference, because the two have different
evidential weight in an investigation.

**Procedure.** (1) Extract the DI or UDI from structured fields first. (2) Fall back to brand name
and model string matching, with a controlled vocabulary. (3) Fall back to the reporter's free text
only with human review. (4) Flag each link as exact, probable, or unlinked. (5) Quantify the linkage
rate by period and by source, because it changes. (6) Never impute a lot number.

**Branch points.** A device with a high exact-linkage rate supports lot-level analysis. A device
with a low rate supports only model-level analysis, and the report must say so. A device whose
competitors share a product code can be compared at class level but not at lot level.

**Edge cases.** A brand name shared across sizes links to the model but not the size, so size-adjusted
analysis is impossible. A DI that was retired and replaced links to a history that must be merged
deliberately, not by string matching. A report that names only the manufacturer and the product
category cannot be linked at all and should be counted as unlinked rather than dropped.

### Step 5 — Clean, deduplicate, and assign time

**Input.** The linked event table and the exposure table.

**Output.** A clean event table with one row per event, an event date and a report date, and a
deduplication key.

**Rationale.** One event can generate several reports: the hospital, the distributor, the
manufacturer, and a clinician may each file. Counting reports rather than events inflates the
numerator, and the inflation is not constant, because reporting behaviour changes. Time assignment
matters for the same reason: a period defined by receipt date will show a step whenever the intake
process changes.

**Procedure.** (1) Build a deduplication key from device, date, site, and failure mode. (2) Review
near-duplicates with human judgement. (3) Record both the event date and the report date. (4) Assign
each event to a period by event date for rate calculations and by report date for regulatory
timeliness. (5) Quantify the reporting lag distribution. (6) Flag the most recent period as
incomplete.

**Branch points.** A device with a single reporter per event needs no deduplication. A device
reported through distributors needs aggressive deduplication, because the same event arrives twice
by design.

### Step 6 — Compute rates with intervals

**Input.** The cleaned event counts and the exposure table.

**Output.** A rate per unit of exposure per period, with a confidence interval, and the same for
each stratum.

**Rationale.** A rate without an interval invites over-reading. With 41 events, the 95 percent
interval on the rate spans roughly a factor of 1.8, so a "27 percent increase" may be entirely
compatible with no change. The interval is also the honest way to present a rate to a committee that
wants a number.

**Procedure.** (1) Compute the rate as events divided by exposure, in a stated unit. (2) Compute a
Wilson or exact Poisson interval. (3) For a comparison, compute the rate ratio with a log-scale
interval. (4) Present counts, denominators, rates, and intervals together, never a rate alone. (5)
Round consistently and state the unit.

**Branch points.** Small counts need exact methods. Rates near zero need a different presentation
because the interval is asymmetric. A rate computed on a denominator that changed definition
mid-period cannot be compared without restatement.

**Edge cases.** A rate expressed per unit and a rate expressed per procedure differ by the units per
procedure, which varies by site and by year. A rate with a denominator below a few hundred has an
interval so wide that it should be presented as a count. A rate on a device with a long implant life
accumulates late events in a period whose denominator was built years earlier.

### Step 7 — Stratify, standardise, and compare

**Input.** Rates by stratum, and a reference population or case mix.

**Output.** Standardised rates, stratum-specific rates, and a statement of which differences are
case-mix and which are device.

**Rationale.** A device used in sicker patients has more adverse events without being worse. A
device launched into a new territory has a different case mix. Standardisation removes the mix
effect so that the remaining difference can be attributed, tentatively, to the device. Stratification
does the opposite and is often more useful: it shows where the problem is concentrated.

**Procedure.** (1) Choose strata that plausibly affect the outcome: age, sex, anatomy, operator
volume, facility type, device size. (2) Compute stratum-specific rates. (3) Directly standardise to
a reference distribution. (4) Compare standardised rates across periods or devices. (5) Report both
the crude and the standardised rate, and say which is which.

**Branch points.** If the strata are unavailable for the historical period, standardisation cannot
be applied retrospectively and the comparison must be restricted to the period where the data
exist.

**Edge cases.** Standardising to a reference distribution that no longer resembles the market
produces a number that is comparable but not meaningful. Stratifying on a variable that is itself
caused by the device, such as procedure duration, removes part of the effect you are trying to
measure. A stratum with three events and no denominator cannot be reported as a rate.

### Step 8 — Detect signals

**Input.** The rate series by stratum and period.

**Output.** A signal log with date, stratum, magnitude, method, and initial assessment.

**Rationale.** Signal detection is a screening activity, not a proof. A pre-specified method
protects against the temptation to notice a pattern after the fact. Different methods suit
different questions: a CUSUM detects a persistent shift, a funnel plot compares many units against
a common benchmark, and a simple threshold catches a single catastrophic event that no statistical
method would flag.

**Decision table — responding to a signal**

| Signal type | Evidence | Response | Timeframe |
|---|---|---|---|
| Single death or serious injury with a plausible device cause | Case report | Immediate assessment, reportability decision | Hours |
| Cluster of similar failures in one lot | Rate ratio with interval excluding 1 | Containment, lot trace, root cause | Days |
| Gradual upward shift across all lots | CUSUM or trend test | Process investigation, supplier audit | Weeks |
| One site or one operator above the funnel limit | Stratum-specific rate | Site review, training check, data quality check | Weeks |
| No signal but a known failure mode from the risk file | Risk file review | Targeted query, not a statistical search | Per PMS plan |

**Procedure.** (1) Apply the pre-specified rules. (2) Log every trigger, including false alarms.
(3) Assign an owner and a deadline. (4) Do not adjust the rule after seeing the data; if the rule
was wrong, change it for the next period and say so.

**Edge cases.** A CUSUM tuned on the data it monitors will alarm on noise and miss the shift it was
meant to catch. A funnel plot with many small units will produce false outliers at the edges, where
the limits are wide but the eye is drawn. A threshold rule applied to a rate whose denominator
changed will alarm on the denominator, not the numerator.

### Step 9 — Triage for reportability and corrective action

**Input.** The signal, the event details, the jurisdiction, and the device history.

**Output.** A reportability decision per jurisdiction with rationale, and a corrective action
decision.

**Rationale.** Reportability is a legal determination with a clock. In the United States, a
manufacturer must report a death, serious injury, or reportable malfunction within 30 calendar days
of becoming aware, and certain events within five working days. In the European Union, a serious
incident must be reported within 15 days, a death or unanticipated serious deterioration in health
within 10 days, and a serious public health threat within 2 days. The clock starts at awareness,
not at the end of the investigation.

**Decision table — reportability and action**

| Event | US (21 CFR 803) | EU (MDR Article 87) | Corrective action |
|---|---|---|---|
| Death or serious injury possibly caused by the device | 30-day report; 5-day if remedial action is needed to prevent imminent harm | 10 days if death or unanticipated serious deterioration; 15 days otherwise | FSCA assessment, risk file update |
| Malfunction that could cause death or serious injury if it recurred | 30-day report | Reportable if it led or could have led to a serious incident | FSCA if the risk is not acceptable |
| Use error with device contribution | Reportable if it meets the serious injury criteria | Reportable on the same basis | Usability and labelling review |
| Event with no device contribution | Not reportable; recorded | Not reportable; recorded | None beyond complaint handling |
| Trend without an individual serious event | Not reportable | Not reportable | PMS review, possible CAPA |

**Procedure.** (1) Determine whether the event meets the definition. (2) Determine causation or
contribution. (3) Determine each jurisdiction. (4) Start the clock at awareness and record the
date. (5) File the initial report, then supplement as the investigation develops.

**Edge cases.** Awareness is dated when any employee of the manufacturer could reasonably have
concluded that a reportable event occurred, not when the committee met. A report filed in one
jurisdiction does not satisfy another. A trend that is not itself reportable can still require an
FSCA if the risk is no longer acceptable.

### Step 10 — Act, communicate, and close the loop

**Input.** The corrective action decision, the affected population, and the communication plan.

**Output.** Executed action, field safety notice where required, updated risk file, and an updated
PMS report.

**Rationale.** An analysis that does not change anything is a cost. The loop closes when the action
is implemented, the effectiveness is checked, and the finding is fed back into design, labelling,
and the risk file. The check is the part most often skipped: a recall is closed when the units are
accounted for, not when the notice is sent.

**Procedure.** (1) Choose the action: correction, removal, labelling change, or no action with
rationale. (2) Identify the affected units through UDI and lot. (3) Notify users with a field
safety notice where required. (4) Track effectiveness. (5) Update the risk file and the PMS report.
(6) Record the closure date and the evidence.

**Branch points.** A correction that restores the device in the field is usually preferable to a
removal, and is classified differently. A removal without a legal violation is a market withdrawal,
not a recall.

**Edge cases.** A notice sent to distributors but not to users does not reach the operating room. A
correction that depends on a field engineer visiting every site takes months, during which the
exposure continues to accumulate. A recall closed on the basis of units returned, rather than units
accounted for, overstates its own effectiveness.
## 6. Worked example

The device, company, and numbers in this case are constructed for teaching. The arithmetic is real
and is reproduced by the script in section 7. This is not a real recall, a real MDR, or a real
regulatory action.

### 6.1 The product and the question

VasoSeal is a single-use vascular closure device, Class II in the United States under a 510(k) and
Class IIb in the European Union. It is sold in boxes of five, distributed through three national
distributors and used in catheterisation laboratories. Each unit carries a GS1 UDI with a lot
number.

In the second half of the year the quality team notices that complaints have risen. The head of
quality asks the question that starts this analysis: "Are complaints increasing, and if so, is it
the device?"

That question contains three separate questions. Is the count rising. Is the rate rising. Is the
rate rising for a reason attributable to the device. The first is trivial, the second is the
analysis, and the third is the investigation.

### 6.2 The raw counts

```text
first half  (H1): 41 complaints
second half (H2): 68 complaints
```

The count rose by 27, which is 65.9 percent. A dashboard showing only this would flag red. It is
also the least informative number in the case, because the denominator changed at the same time:
the sales territory expanded in April and the second half includes two regions that were not
selling in the first.

### 6.3 The denominator

Units sold were 24,000 in H1 and 31,200 in H2, an increase of 30 percent.

```text
H1: 41 / 24,000 = 0.001708 per unit = 1.708 per 1,000 units
    95% interval 1.260 to 2.317
H2: 68 / 31,200 = 0.002179 per unit = 2.179 per 1,000 units
    95% interval 1.720 to 2.762
```

The rate ratio is 2.179 divided by 1.708, which is 1.276, with a 95 percent interval from 0.866 to
1.880. The point estimate says 28 percent higher; the interval includes no change. The 65.9 percent
count increase has become a 27.6 percent rate increase with an interval that cannot exclude 1.

That is the first lesson, and it is not the interesting one. The interesting one is that the
denominator may still be wrong.

### 6.4 Units sold are not units used

In H2 the distributors increased their stock by 2,400 units, anticipating a tender. Those units
were sold by the manufacturer and not used on a patient. The exposure that matters is units
implanted:

```text
H2 units implanted = 31,200 - 2,400 = 28,800
H2 rate            = 68 / 28,800 = 2.361 per 1,000
rate ratio         = 2.361 / 1.708 = 1.382
95% interval       = 0.938 to 2.036
```

The direction is predictable: when the channel absorbs stock, units sold overstates exposure and
the rate is understated. When the channel drains stock, the opposite happens and the rate is
overstated. Neither error is visible in the sales figure, and both are large enough to matter at
the precision the decision needs.

Note what did not happen. The interval still includes 1, so the corrected analysis still does not
establish an increase. The correction moved the point estimate and left the conclusion alone. That
is common, and it is why the interval matters more than the point estimate.

### 6.5 Three process changes that move the numerator

The count is also not stable. Three changes in the second half affect how many complaints are
captured for a fixed number of true events.

**Capture.** A new complaint portal went live in July, with a mobile form and a QR code on the
pack. The vigilance team estimates that the portal raised capture by about 20 percent for the same
underlying events, based on the increase in reports from unchanged accounts. Dividing the count by
1.20 gives 56.7 events and a rate of 1.816 per 1,000, a ratio of 1.063 against H1.

**Reporting lag.** Complaints for the most recent weeks are still arriving. About 32 percent of H2
complaints fall in the last two months, and those months are estimated to be 62 percent complete.
Adding back the missing reports gives 68 + 21.8 x (1/0.62 - 1) = 81.3 events and a rate of 2.607,
a ratio of 1.526.

**Case mix.** H2 contains a higher proportion of complex interventions, which are more likely to
complicate regardless of device. Weighting H2 to the H1 case mix gives 65.3 events and a rate of
2.092, a ratio of 1.225.

The three corrections push in different directions and none is small. Capture lowers the estimate;
lag raises it; mix lowers it slightly. An analysis that applies one and not the others is not
neutral, it is a choice, and the choice determines the answer.

### 6.6 The lot the aggregate hid

While the aggregate was being argued about, the lot-level table was showing something else.

| Lot | Complaints | Units | Rate per 1,000 | Rate ratio vs other lots | 95% interval |
|---|---:|---:|---:|---:|---|
| B7719A | 9 | 640 | 14.062 | 8.19 | 4.00 to 16.75 |
| B7720B | 4 | 1,180 | 3.390 | 1.97 | 0.71 to 5.49 |
| C1103A | 5 | 1,420 | 3.521 | 2.05 | 0.81 to 5.17 |
| C1104A | 3 | 990 | 3.030 | 1.76 | 0.55 to 5.68 |
| D2201C | 2 | 760 | 2.632 | 1.53 | 0.37 to 6.32 |
| All other lots | 45 | 26,210 | 1.717 | 1.00 | reference |

Lot B7719A has nine complaints in 640 units, a rate of 14.1 per 1,000, which is 8.2 times the rate
in the remaining 26,210 units, with a 95 percent interval from 4.0 to 16.8. That interval excludes
1 decisively, unlike every aggregate comparison in this case.

The monthly counts for the lot are 0, 1, 0, 2, 4, 2. A CUSUM with a target of 0.2 complaints per
month, a reference value of 0.5 and a decision interval of 3.0 accumulates to 4.60 in month five
and alarms. The lot was in production for six months and has since been discontinued.

This is the finding that matters, and it is invisible in every aggregate statistic computed so far.
The aggregate rate was flat after correction; one lot was running at eight times the background
rate. Aggregation is not a neutral simplification, because the units are not exchangeable.

### 6.7 Two diagnostics that reveal a broken analysis

**Diagnostic 1: the period boundary is defined by receipt, not by event.** The dashboard assigns
each complaint to the period in which it was received. Because the portal launched in July, six
complaints about June events were entered in July. Reassigning them by event date moves six reports
from H2 to H1:

```text
H1: 47 / 24,000 = 1.958 per 1,000
H2: 62 / 31,200 = 1.987 per 1,000
rate ratio      = 1.015
```

The 27.6 percent increase becomes 1.5 percent. Nothing about the device changed; the definition of
the period did. A dashboard that dates by receipt will show a step at every intake-process change,
and intake processes change more often than devices do.

**Diagnostic 2: the denominator is a moving target.** The exposure table is built from monthly
sales, and the monthly sales figures are restated by finance. In H2 the restatement added 2,400
units of distributor stock to the reported sales after the initial report. The dashboard was built
before the restatement and still used the original figure. The same period therefore had two
denominators in circulation, differing by 8 percent, and the choice between them moved the rate
ratio from 1.276 to 1.382.

Either diagnostic produces a number that looks precise and is wrong. Both are data-definition
problems rather than statistical ones, and neither would be caught by a better confidence interval.
The check is to reconcile the numerator's date field and the denominator's source against the
process calendar once per period, and to record which version was used.

### 6.8 Vary one assumption

Every correction in section 6.5 depends on an estimate. The most consequential is the capture gain
from the portal, because it is the one the team is least able to measure and most likely to assume.

| Assumed capture gain | Adjusted H2 rate per 1,000 | Ratio versus H1 |
|---:|---:|---:|
| 0 percent | 2.179 | 1.276 |
| 10 percent | 1.981 | 1.160 |
| 20 percent | 1.816 | 1.063 |
| 30 percent | 1.677 | 0.982 |

At a 30 percent capture gain the ratio falls below 1, meaning the device is performing better than
in the first half once the reporting change is removed. The conclusion flips from "a possible
increase" to "no increase" across a range of assumptions that all sound plausible in a meeting.

The right response is not to pick the most convenient value. It is to measure the capture gain, for
example by resubmitting a sample of historical complaints through the new portal and comparing
capture, and to report the result as a sensitivity rather than a point. The lot finding does not
depend on this assumption at all, which is one reason it is the more robust result.

### 6.9 What the case does not show

The case does not establish causation for lot B7719A. A rate ratio of 8.2 with an interval excluding
1 establishes an association between the lot and the complaints, not a mechanism. The mechanism
requires the investigation: a review of the lot's production records, a comparison of incoming
material, and bench testing of retained samples.

The case also does not show that the aggregate analysis was unnecessary. It shows the opposite: the
aggregate analysis was necessary to establish that there was no system-wide problem, which is what
allowed the investigation to focus on one lot rather than on the design.
## 7. Code, query, or calculation

The script below parses UDI strings for the three issuing agencies, validates a GS1 check digit,
computes complaint rates with intervals and rate ratios, corrects for capture and reporting lag,
and runs a CUSUM on a lot-level series. It uses only the standard library and is deterministic: there
are no random draws, so the output is exact.

Save it as `udi_pms.py` and run `python udi_pms.py`.

```python
"""UDI parsing, complaint rates with exposure denominators, and lot-level signal detection.

Standard library only.  Run: python udi_pms.py
"""

import math

# ---------------------------------------------------------------- UDI parsing
GS1_FIXED = {"00": 18, "01": 14, "02": 14, "11": 6, "13": 6, "15": 6, "17": 6, "20": 2}
GS1_VAR = {"10", "21", "22", "30", "37", "240", "241", "250", "400", "410", "21"}


def gtin_check_ok(gtin):
    """GS1 mod-10 check digit over a 14-digit GTIN."""
    if len(gtin) != 14 or not gtin.isdigit():
        return False
    body, check = gtin[:13], int(gtin[13])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(body))
    return (10 - total % 10) % 10 == check


def parse_gs1(raw):
    """Parse an element string.  Variable-length fields end at GS (0x1D) or end of string."""
    out, i = {}, 0
    while i < len(raw):
        ai = raw[i:i + 2]
        if ai in GS1_FIXED:
            n = GS1_FIXED[ai]
            out[ai] = raw[i + 2:i + 2 + n]
            i += 2 + n
        elif ai in GS1_VAR:
            j = raw.find("\x1d", i + 2)
            j = len(raw) if j == -1 else j
            out[ai] = raw[i + 2:j]
            i = j + 1 if j < len(raw) else j
        else:
            break
    return out


def classify(raw):
    if raw.startswith("(01)") or raw[:2].isdigit():
        return "GS1"
    if raw.startswith("+"):
        return "HIBCC"
    if raw.startswith("="):
        return "ICCBBA"
    return "unknown"


def udi_report(samples):
    for raw in samples:
        kind = classify(raw)
        print(f"  raw       : {raw!r}")
        print(f"  issuing   : {kind}")
        if kind == "GS1":
            f = parse_gs1(raw)
            gtin = f.get("01", "")
            print(f"  DI (01)   : {gtin}   check digit {'valid' if gtin_check_ok(gtin) else 'INVALID'}")
            print(f"  PI        : lot={f.get('10','-')}  expiry={f.get('17','-')}  serial={f.get('21','-')}")
            print(f"  is_di     : {gtin}")
        elif kind == "HIBCC":
            body = raw[1:].split("+")
            print(f"  LIC       : {body[0]}")
            print(f"  PCN       : {body[1] if len(body) > 1 else '-'}")
            print(f"  is_di     : +{body[0]}{body[1] if len(body) > 1 else ''}")
        elif kind == "ICCBBA":
            body = raw[1:]
            print(f"  donation  : {body.split('<')[0][:13]}")
            print(f"  is_di     : ={body[:5]}")
        print()


# ------------------------------------------------------- rates and intervals
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 1000, (c + h) * 1000


def rate_ratio(a, na, b, nb):
    rr = (a / na) / (b / nb)
    se = math.sqrt(1 / a + 1 / b)
    return rr, rr * math.exp(-1.959964 * se), rr * math.exp(1.959964 * se)


def report_rate(label, k, n):
    lo, hi = wilson(k, n)
    print(f"  {label:<28} {k:>5} / {n:>7,} = {k / n * 1000:5.3f} per 1,000  "
          f"(95% CI {lo:.3f} to {hi:.3f})")


# --------------------------------------------------------------- CUSUM
def cusum(counts, target, k=0.5, h=5.0):
    s, out = 0.0, []
    for i, c in enumerate(counts, 1):
        s = max(0.0, s + (c - target) - k)
        out.append((i, c, round(s, 2), s > h))
        if s > h:
            s = 0.0
    return out


# ------------------------------------------------------------------ main
print("== 1. UDI structure ==")
udi_report([
    "01008472530012371728063010B7719A" + "\x1d" + "21SN-998822",
    "+M1234567890123+PCN778812",
    "=W0001234567890<A1234",
])

print("== 2. Complaint rate by period, three denominators ==")
h1_c, h2_c = 41, 68
h1_units, h2_units = 24000, 31200
h2_inventory_growth = 2400
h2_implanted = h2_units - h2_inventory_growth
report_rate("H1 units sold", h1_c, h1_units)
report_rate("H2 units sold", h2_c, h2_units)
report_rate("H1 units implanted", h1_c, h1_units)
report_rate("H2 units implanted", h2_c, h2_implanted)
rr, lo, hi = rate_ratio(h2_c, h2_units, h1_c, h1_units)
print(f"  rate ratio, units sold     {rr:.3f} (95% CI {lo:.3f} to {hi:.3f})")
rr2, lo2, hi2 = rate_ratio(h2_c, h2_implanted, h1_c, h1_units)
print(f"  rate ratio, units implanted {rr2:.3f} (95% CI {lo2:.3f} to {hi2:.3f})")

print("\n== 3. Corrections, applied one at a time ==")
capture_gain, trunc_complete, late_share, mix_factor = 0.20, 0.62, 0.32, 0.96
adj_capture = h2_c / (1 + capture_gain)
adj_trunc = h2_c / (1 - late_share * (1 - trunc_complete) / 1.0) if False else h2_c
late = h2_c * late_share
adj_trunc = h2_c + late * (1 / trunc_complete - 1)
adj_mix = h2_c * mix_factor
for label, k in (("raw H2", h2_c), ("capture-adjusted", adj_capture),
                 ("truncation-adjusted", adj_trunc), ("mix-adjusted", adj_mix)):
    r = k / h2_units * 1000
    print(f"  {label:<22} count {k:6.2f}  rate {r:5.3f}  vs H1 1.708  ratio {r/1.708:5.3f}")

print("\n== 4. Lot-level rates ==")
lots = [("B7719A", 9, 640), ("B7720B", 4, 1180), ("C1103A", 5, 1420),
        ("C1104A", 3, 990), ("D2201C", 2, 760)]
base_k = h2_c - sum(k for _, k, _ in lots)
base_n = h2_units - sum(n for _, _, n in lots)
report_rate("all other lots", base_k, base_n)
for name, k, n in lots:
    r, l, h = rate_ratio(k, n, base_k, base_n)
    flag = "  <-- CI excludes 1" if l > 1 else ""
    print(f"  lot {name}  {k:>2} / {n:>5,} = {k/n*1000:6.3f} per 1,000  "
          f"RR {r:5.2f} (95% CI {l:.2f} to {h:.2f}){flag}")

print("\n== 5. CUSUM on monthly complaints for lot B7719A ==")
monthly = [0, 1, 0, 2, 4, 2]
for m, c, s, alarm in cusum(monthly, target=0.2, k=0.5, h=3.0):
    print(f"  month {m}  complaints {c}  CUSUM {s:5.2f}  {'ALARM' if alarm else ''}")

print("\n== 6. Effect of the capture assumption ==")
for g in (0.0, 0.10, 0.20, 0.30):
    adj = h2_c / (1 + g)
    r = adj / h2_units * 1000
    print(f"  capture gain {g:4.2f}  adjusted rate {r:5.3f}  ratio vs H1 {r/1.708:5.3f}")
```

Output on a machine running Python 3.11.15:

```text
== 1. UDI structure ==
  raw       : '01008472530012371728063010B7719A\x1d21SN-998822'
  issuing   : GS1
  DI (01)   : 00847253001237   check digit valid
  PI        : lot=B7719A  expiry=280630  serial=SN-998822
  is_di     : 00847253001237

  raw       : '+M1234567890123+PCN778812'
  issuing   : HIBCC
  LIC       : M1234567890123
  PCN       : PCN778812
  is_di     : +M1234567890123PCN778812

  raw       : '=W0001234567890<A1234'
  issuing   : ICCBBA
  donation  : W000123456789
  is_di     : =W0001

== 2. Complaint rate by period, three denominators ==
  H1 units sold                   41 /  24,000 = 1.708 per 1,000  (95% CI 1.260 to 2.317)
  H2 units sold                   68 /  31,200 = 2.179 per 1,000  (95% CI 1.720 to 2.762)
  H1 units implanted              41 /  24,000 = 1.708 per 1,000  (95% CI 1.260 to 2.317)
  H2 units implanted              68 /  28,800 = 2.361 per 1,000  (95% CI 1.863 to 2.992)
  rate ratio, units sold     1.276 (95% CI 0.866 to 1.880)
  rate ratio, units implanted 1.382 (95% CI 0.938 to 2.036)

== 3. Corrections, applied one at a time ==
  raw H2                 count  68.00  rate 2.179  vs H1 1.708  ratio 1.276
  capture-adjusted       count  56.67  rate 1.816  vs H1 1.708  ratio 1.063
  truncation-adjusted    count  81.34  rate 2.607  vs H1 1.708  ratio 1.526
  mix-adjusted           count  65.28  rate 2.092  vs H1 1.708  ratio 1.225

== 4. Lot-level rates ==
  all other lots                  45 /  26,210 = 1.717 per 1,000  (95% CI 1.283 to 2.296)
  lot B7719A   9 /   640 = 14.062 per 1,000  RR  8.19 (95% CI 4.00 to 16.75)  <-- CI excludes 1
  lot B7720B   4 / 1,180 =  3.390 per 1,000  RR  1.97 (95% CI 0.71 to 5.49)
  lot C1103A   5 / 1,420 =  3.521 per 1,000  RR  2.05 (95% CI 0.81 to 5.17)
  lot C1104A   3 /   990 =  3.030 per 1,000  RR  1.76 (95% CI 0.55 to 5.68)
  lot D2201C   2 /   760 =  2.632 per 1,000  RR  1.53 (95% CI 0.37 to 6.32)

== 5. CUSUM on monthly complaints for lot B7719A ==
  month 1  complaints 0  CUSUM  0.00  
  month 2  complaints 1  CUSUM  0.30  
  month 3  complaints 0  CUSUM  0.00  
  month 4  complaints 2  CUSUM  1.30  
  month 5  complaints 4  CUSUM  4.60  ALARM
  month 6  complaints 2  CUSUM  1.30  

== 6. Effect of the capture assumption ==
  capture gain 0.00  adjusted rate 2.179  ratio vs H1 1.276
  capture gain 0.10  adjusted rate 1.981  ratio vs H1 1.160
  capture gain 0.20  adjusted rate 1.816  ratio vs H1 1.063
  capture gain 0.30  adjusted rate 1.677  ratio vs H1 0.982
```

Four things in the output are worth reading carefully.

The check digit is computed, not assumed. The first sample string carries the GTIN 00847253001237
and the routine confirms it; a single transposed digit would fail. This is the cheapest integrity
check in the whole pipeline and it catches a large share of manual-entry errors.

The rate ratio interval in block 2 includes 1 for both denominators. That is the honest headline
for the aggregate: a 27.6 percent increase in the rate, with an interval from 0.866 to 1.880, is
compatible with no change at all. The count increase of 65.9 percent is not.

The three corrections in block 3 are applied one at a time and each is computed on the raw count,
not on the previous correction. Applying them in sequence would compound them, and the order would
change the answer, which is a reason to report them separately.

The lot table in block 4 is where the signal is. Only lot B7719A has an interval excluding 1, and
it excludes it by a wide margin. The CUSUM in block 5 alarms in month five, which is later than a
human would have noticed the pattern by eye, but earlier than the aggregate would have prompted
anyone to look.

The script is a method demonstration, not a PMS system. A production implementation adds
deduplication, jurisdiction-specific reportability rules, audit trails, and a validated data
pipeline, and its outputs are subject to computerised system validation under the quality system.
## 8. Data, standards, and regulatory context

The regulatory instruments below create the data. They are worth reading not as compliance
obligations but as specifications: each one tells you what a field means, who owns it, and what it
may be joined to.

| Instrument | What it requires | What it does not do |
|---|---|---|
| 21 CFR Part 830 | UDI on the label and package; data submission to GUDID; the DI as the key | Require hospitals to record the UDI in the record |
| 21 CFR Part 803 | Report deaths, serious injuries, and reportable malfunctions; 30-day and 5-day timelines | Provide a denominator or a rate |
| 21 CFR Part 806 | Report corrections and removals; maintain distribution records | Define a rate or a benchmark |
| 21 CFR Part 7 | Recall policy, classification, and public notification | Link a recall to a specific patient |
| EU MDR Articles 27 and 29 | UDI assignment and registration in EUDAMED | Populate the US GUDID, or vice versa |
| EU MDR Article 86 | PSUR for Class IIa and above, with sales volume and population estimates | Specify the statistical method |
| EU MDR Article 87 | Serious incident reporting: 15 days, 10 days for death or unanticipated serious deterioration, 2 days for a serious public health threat | Define device causation |
| EU MDR Article 88 | Trend reporting of incidents that are not serious but increasing | Specify a detection method |
| ISO 13485:2016 clauses 8.2.1 to 8.2.3 | Feedback, complaint handling, and notification to authorities | Require analytics |
| IMDRF UDI guidance N48 | Harmonised UDI principles and data elements | Create a global database |

**The US UDI system.** The DI is a fixed identifier for the model and version; the PI carries lot,
serial, expiry, and manufacturing date where applicable. Implantable and life-sustaining devices
require a serial number or lot. The phase-in followed device class, with Class III first, then
implantable and life-sustaining devices, then Class II, then Class I, and a later deadline for direct
marking. The compliance dates are in 21 CFR 830.50 and should be read there rather than from a
secondary source, because they have been amended.

The FDA recognises three issuing agencies: GS1, whose DI is a GTIN-14; the Health Industry Business
Communications Council (HIBCC), whose syntax is plus-prefixed; and ICCBBA, whose ISBT 128 syntax is
used for human-derived products. A parser must handle all three, and a system that assumes GS1 will
silently fail on tissue-based devices.

**GUDID and EUDAMED are not the same database.** GUDID holds US device records keyed on the DI, with
attributes including the brand name, model, product code, GMDN term, and whether the device is
implantable or life-sustaining. EUDAMED holds EU records keyed on the basic UDI-DI at the device
family level, with the EMDN classification and the certificate links. A device sold in both markets
has two records with different keys and different attributes, and reconciling them is a manual
mapping exercise, not a join.

The EUDAMED timeline matters for anyone building this in 2026. Commission Decision (EU) 2025/2371 of
27 November 2025 declared the first four modules functional and started a transition period.
Mandatory use of actor registration, UDI and device registration, notified bodies and certificates,
and market surveillance begins on 28 May 2026. Devices already on the market must be registered by
28 November 2026. The vigilance and post-market surveillance module is expected to become mandatory
around the second quarter of 2027, which means that for the remainder of 2026 the EU incident data
still flow through national competent authorities rather than a central database.

> **Note:** The EUDAMED dates above were checked on 27 October 2026. Transition dates in this area
> have moved before, and a plan that depends on a future module should be checked against the
> Commission's page before it is committed to.

**Reportability timelines.** In the United States, the manufacturer reports within 30 calendar days
of becoming aware of a reportable event, and within five working days when the event necessitates
immediate remedial action to prevent a public health hazard, or when the FDA requests a five-day
report. In the European Union, Article 87 sets 15 days for a serious incident, 10 days for death or
unanticipated serious deterioration in health, and 2 days for a serious public health threat. The
clock starts at awareness, which is earlier than most teams assume: it runs from when any employee
could reasonably have concluded that a reportable event occurred.

**Coding systems.** GUDID uses GMDN terms and FDA product codes; EUDAMED uses EMDN. IMDRF adverse
event terminology provides codes for the device problem, the patient problem, and the health effect,
and using them makes cross-manufacturer comparison possible. A complaint system that stores free
text and no codes can be searched but not aggregated.

**What is binding, what is convention.**

```text
  BINDING                              CONVENTION                  GUIDANCE
  -------                              ----------                  --------
  21 CFR 803, 806, 830:                IMDRF AET codes:            FDA guidance on UDI
  report, record, identify             common vocabulary           and on MDR

  EU MDR Articles 27, 86, 87:          GS1 General                  MDCG guidance on
  register, report, update             Specifications              vigilance and FSCA

  ISO 13485 via the QMSR:              Rate per 1,000 units        Notified body
  complaint handling and               or per 1,000 patient-years  expectations, case
  notification records                                             by case
```

The distinction has a practical consequence. A rate expressed per 1,000 units is a convention, not a
requirement, and a regulator may prefer per patient-year. Nothing in the regulation says which; the
PSUR guidance asks for rates and population estimates without prescribing the denominator. That
freedom is why the denominator choice must be documented and defended rather than defaulted.

## 9. Industry practice and case evidence

**Documented.** Three recalls are routinely cited as the reason UDI exists, and all three predate
it. Medtronic recalled its Sprint Fidelis defibrillation leads in 2007 after fractures were linked
to patient deaths, and identifying affected patients depended on hospital records keyed by model
and serial number rather than a standard identifier. DePuy recalled the ASR hip resurfacing system
in 2010, and the recall's patient identification problem was the same one. Philips Respironics
recalled several positive airway pressure devices in 2021 over sound-abatement foam degradation, a
recall that affected millions of units and depended on device registration to reach users.

**Inference.** What these cases show is not that UDI would have prevented the failures. It is that
the response was slower and less complete than it needed to be because the device identity was not
a join key in the systems that held the patients. A recall is a data problem before it is a
logistics problem: you cannot notify a patient you cannot find.

**Documented.** The FDA's openFDA service exposes device endpoints for UDI records derived from
GUDID, adverse events from MAUDE, recalls and enforcement reports from the Recall Enterprise System,
and registration and listing data. Each endpoint returns at most 1,000 records per request, so a
complete extraction requires paging, and the API documentation warns that MAUDE reports are
voluntary, unvalidated, and not evidence of causation.

**Inference.** The 1,000-record limit is not a detail. It means that any analysis of a device family
with more than a thousand events requires pagination with a stable sort, and that a naive query will
silently return a biased subset if the sort order correlates with the outcome, which it does when
records are returned in submission order. Teams that pull one page and treat it as the population
produce a sample that over-represents whichever period the API happens to return first.

**What manufacturers actually do.** Three patterns are common, and only the first is good practice.

The first is a validated pipeline: complaint intake codes the event, the DI is scanned or selected,
the lot is captured at intake, sales and shipping feed a denominator table, and the whole thing is
version-controlled and audited. This is what a mature quality system looks like and it is uncommon.

The second is a spreadsheet reconciliation. The complaint log is exported monthly, the sales figures
come from finance, and an analyst joins them by hand. It works until the analyst leaves, and it
cannot be audited.

The third is no denominator at all. Complaints are counted, trends are eyeballed, and the PSUR is
written from the count. This is more common than the industry would like to admit, and it is what
the regulatory emphasis on rates is trying to change.

**The hospital side.** UDI adoption at the point of care is the weak link. Labels carry the UDI
because they must; hospitals scan it at receiving in many systems; but documentation of the implant
in the patient record is often manual or absent. A manufacturer can therefore know which lots it
shipped and still be unable to tell a hospital which of its patients received them. Registries fill
part of the gap for high-risk implants, and their coverage is partial and varies by country.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Denominator is units sold with changing channel stock | Sales and shipping diverge from implant records | Distributor stocking absorbed or released | Rate trend in the wrong direction | Reconcile quarterly; use units implanted |
| Capture changed when a process changed | A new portal, form, or distributor coincides with a step | More reports for the same events | False signal | Measure capture by resubmission or by unchanged-account comparison |
| Period assigned by receipt date | Step change at a process launch | Events dated later than they occurred | Trend artefact | Assign by event date for rates, receipt date for timeliness |
| Right truncation ignored | The latest period looks better than the last | Reports still arriving | Understated recent rate | Model the lag; flag incomplete periods |
| Reports counted instead of events | Duplicate narratives from site and distributor | One event, several reports | Inflated numerator | Deduplicate on device, date, site, failure mode |
| Lot not captured at intake | Linkage rate below 50 percent | Free-text only | Cannot contain a lot | Make lot a required field; scan UDI |
| Brand name used as the key | Model renamed in a period | Two identities for one device | History split | Maintain a DI crosswalk with predecessors |
| Class I recall confused with Class I device | Severity field derived from device class | Two different classifications | Wrong escalation | Store recall class and device class separately |
| Rates compared across jurisdictions | Different reporting regimes | Mandatory versus voluntary reporting | Class comparison invalid | Compare within jurisdiction or standardise |
| CUSUM tuned on the monitored data | Alarm rate far from design | Reference value fitted post hoc | Missed shift or alarm fatigue | Fix parameters in the PMS plan |
| Unlinked events dropped silently | Denominator of the rate is unclear | Convenience | Selection bias | Report the linkage rate alongside the result |
| Recall status assumed current | OpenFDA notes that recall status is not updated | Snapshot data treated as live | Wrong containment decision | Track status in your own system |
| Denominator ignores expiry and returns | Expired stock counted as exposure | Units never used | Understated rate | Subtract expiry and returns in the bridge |
| Class comparison uses raw counts | Large manufacturer looks worse | Volume, not risk | Wrong benchmark | Standardise by volume and case mix |
## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Every DI maps to exactly one commercial model | The spine is the join key for everything | Renames, repackaging, OEM variants | Reconcile the DI list against the ERP item master quarterly |
| Units sold approximate units used after an inventory correction | Implant records rarely exist | Channel stocking, consignment, expiry | Build the manufactured-to-implanted bridge for one quarter |
| The capture rate is estimable and stable within a period | The numerator is otherwise uninterpretable | Process changes, new distributors, new staff | Resubmit a sample of old complaints through the new process |
| Event dates are recorded and accurate | Period assignment depends on it | Reporters recall dates loosely; some systems store only receipt | Compare event and receipt date distributions; flag long lags |
| Duplicate reports can be identified | Counting reports inflates the numerator | Free-text narratives, missing device identifiers | Sample and manually review; measure the duplicate rate |
| The lot is captured for a useful share of events | Lot-level containment depends on it | Free-text intake, distributor reporting | Report the exact-linkage rate by period and source |
| Rates are comparable across periods | The trend is the question | Definition changes, coding changes, system migrations | Restate the historical period under the current definition |
| Stratification variables exist in both periods | Case-mix adjustment requires them | New fields introduced mid-period | Restrict the comparison to the period where the fields exist |
| A benchmark population is comparable | Class comparisons need a reference | Different jurisdiction, different reporting regime | Compare within jurisdiction; state the benchmark's limits |
| The PMS plan's thresholds were set before the data | Pre-specification protects against drift | Thresholds tuned to last year's results | Check the plan's version date against the analysis date |

Two limits apply to everything in this topic.

The data are generated for regulatory purposes, not for measurement. A complaint is a report of a
perceived deficiency, filtered through a reporter's judgement, a distributor's process, and a
manufacturer's intake form. It is not an incident rate, and the gap between the two is not
observable from the data alone.

Regulatory databases are snapshots with their own lags. GUDID, EUDAMED, MAUDE, and the recall
databases update on different schedules, and a comparison across them inherits the difference. Any
figure taken from a public database should carry the date it was retrieved.

## 12. Exercises

**Exercise 1 — Parse a UDI (application).**
Given the GS1 element string `01008472530012371728063010B7719A<GS>21SN-998822`, extract the DI, lot,
expiry, and serial, validate the check digit, and state which fields are static and which vary per
unit.
*Deliverable:* a table of the four fields with their AI codes and one sentence on DI versus PI.
*Acceptance:* DI identified as 00847253001237 with a valid check digit; lot B7719A; expiry 28 June
2030; serial SN-998822; static and dynamic fields correctly separated.
*Time:* 20 minutes.

**Exercise 2 — Choose a denominator (critique).**
A reusable surgical stapler is reprocessed and reused up to 20 times per unit. The quality team
proposes complaints per 1,000 units sold.
*Deliverable:* a 250-word critique naming the bias and proposing an alternative, with the data
needed to construct it.
*Acceptance:* identifies that units sold understates exposure by up to a factor of 20; proposes
procedures or uses as the denominator; names the reprocessing records as the data source; states the
direction of the bias if the reuse count is unknown.
*Time:* 30 minutes.

**Exercise 3 — Compute a rate and a ratio (calculation).**
Period 1: 41 complaints, 24,000 units. Period 2: 68 complaints, 31,200 units. Compute both rates per
1,000, the rate ratio, and its 95 percent interval, and state whether the increase is established.
*Deliverable:* the four numbers and a one-sentence conclusion.
*Acceptance:* rates 1.708 and 2.179; ratio 1.276; interval 0.866 to 1.880; conclusion that the
increase is not established at the 5 percent level despite the 65.9 percent count increase.
*Time:* 25 minutes.

**Exercise 4 — Correct for capture (application).**
A new intake portal is estimated to have raised capture by 20 percent. Recompute the period 2 rate
and ratio, then repeat for capture gains of 0, 10, and 30 percent.
*Deliverable:* a four-row sensitivity table and a recommendation.
*Acceptance:* values within 0.01 of 2.179, 1.981, 1.816 and 1.677; recommendation states that the
conclusion depends on an unmeasured quantity and proposes how to measure it.
*Time:* 30 minutes.

**Exercise 5 — Find the lot (calculation).**
From the lot table in section 6, compute the rate ratio and interval for lot B7719A against all other
lots, and state whether the lot should be contained.
*Deliverable:* the ratio, the interval, and a containment recommendation with a rationale.
*Acceptance:* ratio near 8.2; interval near 4.0 to 16.8; recommendation to contain, because the
interval excludes 1 and the exposure window is closed.
*Time:* 35 minutes.

**Exercise 6 — Design a PMS analytics specification (design).**
Write the analytics section of a PMS plan for a Class IIb implant with a UDI, sold through
distributors in three countries.
*Deliverable:* a four-page specification covering metrics, denominators, data sources, detection
rules, thresholds, review cadence, and audit requirements.
*Acceptance:* every metric has a numerator, a denominator, and a unit; each detection rule has a
pre-specified parameter and a response; the reconciliation from manufactured to implanted is
described; the specification states who reviews and how often.
*Time:* 90 minutes.

**Exercise 7 — Trace a recall (application).**
A lot of 5,000 units is found to have an inadequate sterilization validation. Describe, step by
step, how UDI and the device graph let you identify the affected units and the patients.
*Deliverable:* a process diagram and a 300-word note on where the chain breaks.
*Acceptance:* distinguishes DI from PI; names the distributor and hospital records needed; states
that patient identification depends on the implant record and is often the weakest link; proposes a
registry check as a fallback.
*Time:* 45 minutes.

**Exercise 8 — Critique a PSUR section (critique).**
You are given a PSUR paragraph: "In the reporting period 68 complaints were received, compared with
41 in the previous period, an increase of 66 percent. The benefit-risk profile remains favourable."
*Deliverable:* a 400-word critique with the missing elements and a rewritten paragraph.
*Acceptance:* identifies the missing denominator, interval, capture history, and stratification;
the rewrite contains a rate, an interval, a denominator definition, and a statement of what is not
known.
*Time:* 40 minutes.

## 13. Self-check answers

**Exercise 1.** The AI 01 field is the GTIN-14 and is static; 17 is the expiry and varies by lot; 10
is the lot and varies by batch; 21 is the serial and varies by unit. The DI is 00847253001237 and
the check digit validates. The PI is the combination of lot, expiry, and serial.

**Exercise 2.** Complaints per 1,000 units sold understates exposure by the average number of uses
per unit, so the rate is overstated by roughly the reuse factor. If a unit is used 12 times on
average, the true rate per use is about one twelfth of the reported rate. The right denominator is
procedures or uses, built from reprocessing records or from operating-room logs; if neither exists,
the analysis should report complaints per unit with an explicit statement that the unit is not the
exposure.

**Exercise 3.** 41/24,000 x 1,000 = 1.708. 68/31,200 x 1,000 = 2.179. The ratio is 1.276. The log-scale
standard error is the square root of (1/68 + 1/41), which is 0.1975, so the interval is
exp(ln 1.276 ± 1.96 x 0.1975), giving 0.866 to 1.880. The interval includes 1, so the increase is not
established, even though the count rose by 65.9 percent.

**Exercise 4.** Dividing 68 by 1.20 gives 56.67, a rate of 1.816 and a ratio of 1.063. At 0 percent
capture gain the ratio is 1.276; at 10 percent, 1.160; at 30 percent, 0.982. The conclusion moves from
"a possible increase" to "no increase" across a range of assumptions that all sound plausible, which
is why the capture gain has to be measured rather than assumed.

**Exercise 5.** Lot rate 9/640 x 1,000 = 14.062. Other lots 45/26,210 x 1,000 = 1.717. The ratio is
8.19, with a log-scale interval from 4.00 to 16.75. Containment is justified because the interval
excludes 1 by a wide margin, the lot is closed to new production, and the failure mode is serious
enough that a false positive is cheaper than a delay.

**Exercises 6 to 8** are assessed against a rubric.

| Criterion | Weight | What earns full credit |
|---|---:|---|
| Every metric has a numerator, denominator, and unit | 25% | No bare counts anywhere |
| Denominator choice is justified against alternatives | 20% | Bias direction stated for the rejected options |
| Detection rules are pre-specified | 20% | Parameters and thresholds dated before the data |
| Reconciliation and provenance are described | 20% | The manufactured-to-implanted bridge is explicit |
| Limits are stated | 15% | Names what the data cannot show |

## 14. Interview questions

**1. What is a UDI and what is it for?** Strong: a DI identifying the model plus a PI identifying the
unit, used as the join key between device, event, and exposure data. Weak: describes it as a barcode.

**2. Why is a complaint count not a safety metric?** Strong: it has no denominator and its numerator
depends on capture, which changes; a rate with an interval is the metric. Weak: says counts are fine
if you compare the same period each year.

**3. Which denominator would you use for an implant?** Strong: units implanted or patient-years
depending on the failure mode, with the choice justified and the bias direction stated. Weak: units
sold.

**4. What is right truncation and why does it matter?** Strong: recent events have not all been
reported yet, so the latest period is incomplete and looks better than it is. Weak: confuses it with
missing data generally.

**5. How would you detect a lot-specific problem?** Strong: lot-level rates against the remaining
population, with intervals, plus a CUSUM within the lot. Weak: says to look at the complaint list.

**6. What is the difference between a correction and a removal?** Strong: a correction fixes or
relabels devices in the field; a removal takes them out of use; both can be recalls and both can be
market withdrawals depending on whether a violation is involved. Weak: uses recall for everything.

**7. When does the MDR reporting clock start?** Strong: at awareness, when any employee could
reasonably have concluded a reportable event occurred. Weak: when the investigation closes.

**8. What is wrong with comparing your rate to a competitor's using MAUDE counts?** Strong: counts
reflect volume and reporting behaviour, not risk; compare rates within a jurisdiction with a
standardised denominator. Weak: says MAUDE is public so it is fair game.

**9. How do you keep a signal-detection rule honest?** Strong: pre-specify it in the PMS plan, log
every trigger including false alarms, and change it only between periods with a documented reason.
Weak: says to tune it until the alarms look right.

**10. What would make you distrust a flat trend line?** Strong: a falling denominator, rising capture,
a widening linkage gap, or an aggregate hiding a stratum. Weak: accepts the line at face value.

## 15. Key terms

- **21 CFR 803** — US Medical Device Reporting regulation.
- **21 CFR 830** — US Unique Device Identification regulation.
- **Basic UDI-DI** — the EU device-family identifier used in EUDAMED.
- **Capture rate** — the share of true events that enter the reporting system.
- **Correction** — repair, relabel, or inspect a distributed device.
- **CUSUM** — cumulative sum chart for detecting a persistent shift.
- **Device identifier (DI)** — the static part of a UDI.
- **EUDAMED** — the European database on medical devices.
- **Exposure denominator** — the unit of use against which events are counted.
- **FSCA** — field safety corrective action under the EU MDR.
- **GMDN** — Global Medical Device Nomenclature, used in GUDID.
- **EMDN** — European Medical Device Nomenclature, used in EUDAMED.
- **GS1** — a UDI issuing agency whose DI is a GTIN-14.
- **GUDID** — the US Global Unique Device Identification Database.
- **HIBCC** — a UDI issuing agency using a plus-prefixed syntax.
- **ICCBBA** — the issuing agency for ISBT 128, used for human-derived products.
- **MAUDE** — the FDA database of device adverse event reports.
- **MDR (US)** — Medical Device Reporting, 21 CFR Part 803.
- **MDR (EU)** — Medical Device Regulation (EU) 2017/745.
- **Patient-year** — one patient exposed for one year.
- **Production identifier (PI)** — the dynamic part of a UDI: lot, serial, expiry.
- **PSUR** — periodic safety update report under EU MDR Article 86.
- **Recall** — removal or correction of a device that violates the law.
- **Reporting lag** — the delay between an event and its appearance in the data.
- **Right truncation** — incompleteness of the most recent period.
- **Signal** — a new or changed association that warrants investigation.
- **Standardised rate** — a rate adjusted to a reference case mix.

## 16. Sources

1. US Food and Drug Administration. *Unique Device Identification System*, 21 CFR Part 830.
2. US Food and Drug Administration. *Medical Device Reporting*, 21 CFR Part 803.
3. US Food and Drug Administration. *Medical Devices; Reports of Corrections and Removals*, 21 CFR
   Part 806.
4. US Food and Drug Administration. *Enforcement Policy; Recall Policy*, 21 CFR Part 7.
5. European Parliament and Council. *Regulation (EU) 2017/745 on medical devices*, Official Journal
   of the European Union, L 117, 5 May 2017, Articles 27, 29, 86, 87 and 88.
6. European Commission. *Commission Decision (EU) 2025/2371* of 27 November 2025 on the functionality
   of EUDAMED modules, and the accompanying transition timetable. Checked 27 October 2026.
7. International Medical Device Regulators Forum. *UDI Application Guide*. IMDRF/UDI WG/N48, 2020.
8. GS1. *General Specifications*, current edition, section on the GTIN and Application Identifiers.
9. Health Industry Business Communications Council. *HIBCC UDI Standard*.
10. ICCBBA. *ISBT 128 Standard Technical Specification*.
11. International Organization for Standardization. *ISO 13485:2016, Medical devices — Quality
    management systems — Requirements for regulatory purposes*, clauses 8.2.1 to 8.2.3.
12. US Food and Drug Administration. *Quality Management System Regulation*, final rule of 2 February
    2024, effective 2 February 2026, incorporating ISO 13485:2016 by reference.
13. US Food and Drug Administration. openFDA device endpoints: `udi`, `event`, `recall`,
    `enforcement`, `registrationlisting`. Documentation at open.fda.gov, checked 27 October 2026.
14. Montgomery DC. *Introduction to Statistical Quality Control*, 8th edition. Wiley, 2019. Source for
    CUSUM design and average run length.
15. Institute for Safe Medication Practices and device recall literature on UDI adoption at the point
    of care. Cited generically; confirm the specific source before quoting a figure.

Sources 1 to 7, 11, 12 and 13 were checked against the issuing body's publication on 27 October 2026.
Sources 8 to 10 and 14 are standards and textbooks cited by title; the reader should confirm the
edition in force.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`11_in-silico-trials-and-simulation.md`](11_in-silico-trials-and-simulation.md)
- Related: [`10_post-market-surveillance.md`](10_post-market-surveillance.md) — the surveillance
  process this topic supplies data for.
- Related: [`08_eu-mdr-and-ivdr.md`](08_eu-mdr-and-ivdr.md) — the EU obligations referenced in
  section 8.
- Root survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs and datasets: [`../../labs/`](../../labs/)
