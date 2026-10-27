# Reference Data and External Signals

> Module 02 — The Supply Chain Data Landscape
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01, 02.02, 02.03
> Version 2026.5

---

## 1. Why this topic exists

A category manager notices the same pattern two years running: sales in the final week of March
exceed sales in the first week of April by roughly 18%, and then recover. Nothing in the order
history explains it. The product did not change, the price did not change, the promotion calendar
did not change. Easter changed. It fell on 31 March in 2024 and on 20 April in 2025, and the
buying window moved with it. The history contained the signal. The model did not, because the
model only knew the date.

That is the case for external data in one paragraph. Your internal systems record what your
company did under conditions they never recorded. Two order histories can look identical while
the world around them differs in temperature, income, tariff, and holiday timing. When those
differences drive demand, a model fitted on internal data alone will attribute to seasonality,
trend, or noise what actually belongs to an outside variable.

Two kinds of outside data are used here, and the distinction matters throughout this topic:

- **Reference data** is the slow-changing scaffolding that lets other data join and be
  interpreted: calendars, country and currency codes, tariff schedules, unit conversions,
  holiday definitions, administrative geographies. It is mostly used as keys and flags.
- **External signals** are time-varying measurements produced outside the firm: weather,
  macroeconomic aggregates, port activity, traffic, footfall, commodity prices. They are mostly
  used as covariates.

Both are external. Reference data tells you *when* and *where* a row belongs; signals tell you
*under what conditions* it occurred. Most failures attributed to "external data" are failures of
the first kind wearing the costume of the second.

Three claims organise the topic.

**Claim 1 — External signals can explain variance that history cannot.** The evidence is real but
modest and context-specific. A study of 1.3 billion transactions across 2,043 United Kingdom
stores found that daily weather added about 3.5 percentage points of explained variance (R²)
overall, and up to 6.4 points for health foods (Rose and Dolega, 2021, Grade A). A study of a
large Canadian retailer reported weather explaining up to 47% additional variance for individual
products and 56% for categories beyond a baseline model (Chan and Wahab, 2024, Grade A, single
retailer). A German grocery study found turnover in peak holiday weeks roughly 50% above normal,
with forecast error ranging from 5% to 65% across retailers (Obermair, Holzapfel and Kühn, 2022,
Grade A, exploratory sample). These are not uniform gains. They are gains in categories where a
mechanism exists.

**Claim 2 — The value is in the join, not the source.** Weather data is free and abundant. Knowing
which grid cell represents a store whose customers drive from four directions, on which day the
purchase occurs, and whether the relevant exposure is the day of sale or the three days before
it, is not free. The same is true of every family in this topic: the public source is the easy
part, and the alignment is the work.

**Claim 3 — External data is the most common silent source of leakage in supply chain models.**
Macroeconomic series are revised after first publication. Weather forecasts are not observations.
Holiday definitions change. Tariff schedules are reissued dozens of times a year. A backtest that
uses today's revised series to predict last year's demand is not a backtest; it is a model that
has seen the future and will fail in production without warning.

| Family | What it mainly explains | Native grain | Main hazard |
|---|---|---|---|
| Calendar | Timing of demand, period comparability | Day, week, fiscal period | Misalignment across fiscal systems |
| Weather | Short-run demand and disruption | Grid cell or station, hour or day | Wrong location, wrong exposure window |
| Macro | Level and direction of demand | Country or region, month or quarter | Revision and publication lag |
| Trade and policy | Cost, availability, eligibility | HS line, country pair, date | Classification drift, effective-date lag |
| Geospatial | Activity, access, congestion | Point, polygon, or raster cell | Inference chain from pixels to behaviour |

A model that omits Easter does not fail loudly. It produces a seasonal profile that is three
weeks wide where the truth is one week wide, and it will keep doing so every year, because the
error is correlated with a calendar that repeats. A model that includes revised GDP as a feature
also does not fail loudly. It reports a backtest accuracy it cannot reproduce on live data. The
first error is visible only if you look at the residual by week. The second is visible only if
you reconstruct the information set that existed at the decision time. This topic teaches both
habits.

**Where this sits in the module.** Topic 01 established which system of record holds which
record. Topic 02 established the entities and identifiers. Topic 03 established the grain of
transaction data. This topic adds the layer that surrounds transactions: the calendar that
buckets them, the conditions under which they occurred, and the policy regime that priced them.
Topic 05 moves to sensor and image data generated by operations themselves. Topic 10 returns to
the question this topic raises but cannot settle: how much to trust a source for a specific
decision.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** reference data from external signals, and **classify** a named source as one,
   the other, or both, with a stated reason.
2. **Convert** between Gregorian, ISO week, and 4-5-4 retail calendars, and **demonstrate** how a
   fixed holiday lands in different fiscal weeks in consecutive years.
3. **Explain** the 53-week rule, the moving-holiday problem, and the trading-day effect, and
   **identify** which of the three is distorting a given series.
4. **Separate** weather as a demand driver from weather as a disruption, and **state** the
   exposure window each requires.
5. **Classify** a macroeconomic series as leading, coincident, or lagging, and **explain** why
   the classification is a claim about mechanism rather than about timing alone.
6. **Reconstruct** an as-of dataset using vintages, and **audit** a backtest for look-ahead
   leakage from revised macro data.
7. **Map** a product to a tariff line and **state** the publication lag and revision cadence of
   the schedule used.
8. **Trace** the inference chain from a geospatial measurement (pixel, vessel position, footfall
   count) to a supply chain quantity, naming each assumption in the chain.
9. **Design** a validation protocol for an external signal that includes a placebo test, a
   no-lookahead split, and a retirement criterion.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two clocks in your head. The first is **event time**: when something happened in the world.
The second is **knowledge time**: when you could have known it, in the form you can now see it.
Every external dataset has both, and most analytical errors come from treating them as one.

```text
   EVENT TIME -- what happened in the world ------------------------------>
        |                |                |                 |
     weather          holiday          port call         tariff
        |                |                |                 |
        v                v                v                 v
   +-------------------------------------------------------------------+
   |  GATE 1  RELEVANCE   is there a mechanism linking it to demand?   |
   |  GATE 2  GRAIN       does it join at item x place x time?         |
   |  GATE 3  TIMING      was it knowable at the decision timestamp?   |
   |  GATE 4  STABILITY   does its definition survive revision?        |
   +-------------------------------------------------------------------+
        |
        v
   KNOWLEDGE TIME -- what you could have known, and when --------------->
        |
        v
   feature ---> model ---> decision ---> outcome ---> feedback
```

Four properties of this picture explain most of what follows.

**Property 1 — A signal is a proxy, never the thing itself.** Rainfall at a weather station is not
rainfall in a store catchment. A port call count is not tonnage. A satellite-derived container
index is not a customs declaration. Each step from measurement to construct is an assumption, and
the assumptions compound. When a signal works, it is because the proxy correlates strongly enough
with the construct over the range that matters. When it fails, it is usually because the
correlation broke at the tails, which is exactly where supply chain decisions live.

**Property 2 — The join is on time and place, and both are lossy.** Transaction data joins on
identifiers: item, location, party. External data joins on a date and a geography. The date is
usually resolved to a day, but the geography is almost always resolved to something that does not
match your network: a postal code, a census tract, a grid cell, a port boundary, a country. Every
such join is an approximation with a measurable error, and the error is not random with respect
to demand.

**Property 3 — Publication is a process, not an instant.** A macro series is first published,
then revised, then benchmark-revised. A tariff schedule is reissued. A holiday date is known
years ahead but its *effect window* is estimated after the fact. The dataset you download today
is a snapshot of a process that will continue without you. If your model is retrained on the
snapshot, it learns a version of history that no decision-maker ever saw.

**Property 4 — Relevance is a mechanism claim, not a correlation claim.** "Temperature affects
ice cream demand" is a mechanism: heat increases consumption of cold products. "The Baltic Dry
Index correlates with our widget sales" is a correlation until you can state the causal path.
Correlations discovered by searching many candidate signals are expected under the null. This is
not a reason to avoid external data. It is a reason to pre-register the mechanism, limit the
number of signals tested, and correct the significance threshold for the number of trials.

### 3.1 Driver versus disruptor

The same physical variable plays two roles, and they require different treatments.

- **Driver.** Weather shifts the level of demand in a predictable direction over a range:
  temperature and cold beverages, snowfall and snow-clearing equipment, heat and air conditioning.
  A driver is modelled as a covariate with a smooth response function, and its value comes from
  the forecast for the selling period.
- **Disruptor.** Weather interrupts supply or access: a storm closes a port, a flood closes a
  road, a freeze damages a crop. A disruptor is modelled as an event with a probability, a
  duration, and a recovery path. Its value comes from the forecast *and* from the uncertainty
  around it.

Confusing the two produces two classic errors. Modelling a hurricane as a demand covariate yields
a model that predicts higher sales in the week of the storm, when the actual mechanism is store
closure and lost sales. Modelling seasonal temperature as an event yields a model that treats
every summer as an anomaly.

### 3.2 The resolution ladder

External data arrives at a resolution that is almost never the resolution of the decision.

```text
   measurement        aggregation         join              decision
   ------------       -----------         ----              --------
   station (1)   ->   grid cell     ->    catchment   ->    store
   hourly        ->   daily         ->    week        ->    replenishment cycle
   vessel ping   ->   port-day      ->    lane        ->    shipment plan
   pixel (10 m)  ->   port polygon  ->    country     ->    import forecast
```

Each arrow is a lossy transformation. Upsampling does not recover information; it invents it.
A store assigned the nearest station's temperature inherits that station's microclimate, which
may differ by several degrees in a coastal city. A country-level tariff rate applied to a
plant-level cost ignores the fact that the plant imports a component classified differently.
State the rung of the ladder you are on, and the error it introduces, before you model.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Reference data | Slow-changing data used to interpret or join other data: calendars, codes, schedules, geographies | Confused with master data | None |
| Master data | The core entities of the business: item, location, party, bill of materials | Used as a synonym for any lookup table | Entity identifier |
| External signal | A time-varying measurement produced outside the firm and used as a covariate | Treated as a fact rather than a proxy | Varies by source |
| Calendar dimension | A table mapping every date to its week, period, quarter, and year under one or more calendars | Built once and never maintained | Date key |
| Fiscal year | The accounting year used for reporting, which need not start on 1 January | Assumed to equal the calendar year | Date range |
| 4-5-4 calendar | Retail calendar with periods of 4, 4, and 5 weeks in each 13-week quarter | Called "the NRF calendar" without qualification | Period and week number |
| 4-4-5 calendar | Retail calendar with periods of 4, 4, and 5 weeks, typically starting the Sunday nearest 1 February | Confused with 4-5-4 | Period and week number |
| 5-4-4 calendar | Retail calendar with a five-week first period in each quarter | Rarely named explicitly in data dictionaries | Period and week number |
| 53-week year | A retail year with an extra week, added roughly every five to six years | Treated as a data error | Week count |
| Retail week | A seven-day reporting week defined by the retailer, usually Sunday to Saturday | Assumed to match ISO week | Week number plus year |
| ISO 8601 week | International standard week numbering, Monday start, week 1 containing the first Thursday | Assumed to match retail week | `YYYY-Www` |
| Moving holiday | A holiday whose date changes each year, such as Easter or Ramadan | Treated as a fixed date | Date |
| Lunar holiday | A holiday set by a lunar calendar, such as Chinese New Year or Diwali | Assumed to fall in the same Gregorian month each year | Date |
| Easter effect | The demand distortion caused by Easter's shifting date and pre-holiday buildup | Modelled as a single holiday dummy | Effect window in days |
| Trading-day effect | Distortion from the number and mix of weekdays in a period | Confused with seasonality | Day counts per period |
| Degree day | A measure of heating or cooling demand relative to a base temperature | Used without stating the base | °C-day or °F-day |
| Base temperature | The outdoor temperature above or below which heating or cooling is assumed to begin | Assumed universal | °C or °F |
| Climate normal | A 30-year average used as a reference for a location and day | Treated as a forecast | °C or °F, with period |
| Forecast horizon | The lead time between forecast issue and the target period | Confused with the planning horizon | Days |
| Ensemble forecast | A set of forecasts from perturbed initial conditions, used to express uncertainty | Reduced to its mean | Member count |
| Reanalysis | A retrospective, consistent reconstruction of past weather fields | Treated as an observation | Gridded dataset |
| Grid cell | A rectangular area of a gridded dataset | Treated as a point | Degrees of latitude and longitude |
| Catchment area | The geography from which a store draws its customers | Assumed to be a fixed radius | Polygon or drive-time band |
| Leading indicator | A series whose movement tends to precede a target variable | Treated as causal | Index or growth rate |
| Coincident indicator | A series that moves with the target variable | Used as a forecast | Index |
| Lagging indicator | A series that moves after the target variable | Used as a predictor | Index |
| Vintage | The version of a dataset as published at a specific date | Ignored entirely | Release date |
| Real-time dataset | A dataset that preserves the values known at each historical date | Confused with a live feed | Release date plus series |
| Revision | A change to a previously published value | Treated as noise | Magnitude and date |
| Benchmark revision | A periodic, large revision incorporating new source data or methods | Underestimated in planning | Effective date |
| Seasonal adjustment | Removal of recurring calendar and seasonal effects from a series | Confused with deseasonalising your own demand | Method and vintage |
| Nowcast | An estimate of the present or very recent past before official data exists | Called a forecast | Target period |
| HS code | Harmonised System tariff nomenclature, maintained by the World Customs Organization | Assumed stable across revisions | 6-digit code |
| HTS | Harmonized Tariff Schedule of the United States, national extension of HS | Used for non-US trade | 8- or 10-digit code |
| TARIC | Integrated Tariff of the European Union, national extension of HS | Confused with the Combined Nomenclature alone | 10-digit code |
| MFN rate | Most-favoured-nation tariff rate applied to WTO members absent a preferential agreement | Confused with the applied rate | Percent ad valorem |
| Bound versus applied tariff | The WTO ceiling versus the rate actually charged | Used interchangeably | Percent |
| Quota | A quantity limit on imports, often with in-quota and out-of-quota rates | Modelled as a tariff only | Quantity plus period |
| Rules of origin | Criteria determining whether goods qualify for a preferential rate | Assumed from the supplier's country | Legal text |
| Sanctions list | A published list of persons, entities, or vessels subject to restrictions | Treated as a static reference | Identifier plus programme |
| AIS | Automatic Identification System, vessel position broadcasts received by satellite and shore stations | Treated as complete coverage | Position, timestamp, vessel ID |
| Port call | An arrival at and departure from a port by a vessel, as inferred from AIS | Equated with cargo volume | Count per day |
| Spatial join | Assignment of a point or area to a containing or nearest geographic unit | Assumed error-free | Method plus distance |
| Alternative data | Commercially or publicly sourced data not produced for the analytical purpose at hand | Treated as inherently superior | Varies |

Three definitions deserve expansion, because they drive most of the errors in this topic.

**4-5-4 versus 4-4-5 versus 5-4-4.** The National Retail Federation publishes a 4-5-4
merchandising and sales-reporting calendar: each quarter is 13 weeks, split into months of four,
five, and four weeks. Many retailers instead run a 4-4-5 calendar beginning the Sunday nearest
1 February, and some run 5-4-4 to place the five-week period at the start of each quarter. All
three produce a 364-day year and all three insert a 53rd week occasionally. They are not
interchangeable, and a data dictionary that says "NRF calendar" without naming the variant is
incomplete. Confirm the variant, the year-start rule, and the week-start day against the
retailer's published calendar before you join anything.

**Degree day.** A heating degree day (HDD) is the number of degrees by which the mean daily
temperature falls below a base, summed over days; a cooling degree day (CDD) is the equivalent
above the base. The base is a convention, commonly 65 °F (18.3 °C) in United States energy and
weather practice, and the choice of base changes the series materially. Two datasets labelled
"HDD" with different bases are not comparable, and a model trained on one will misprice the
other. Always carry the base and the unit with the value.

**Vintage.** A vintage is the entire dataset as it stood at one release. It is not a single value.
The United States advance estimate of GDP for a quarter is revised in subsequent estimates and
again in annual and comprehensive revisions. The Federal Reserve Bank of St. Louis documents
eight vintages of 1991 real GDP and notes that the 2010:Q2 first estimate of 2.4% was later
revised to 1.6% (Grade B). If your model uses the current vintage for a 2019 feature, it is using
numbers that were not available in 2019.

## 5. The mechanics, step by step

This section is a procedure. It assumes you have a decision, a demand series, and access to at
least one external source. The order matters: each step constrains the next, and doing step 4
before step 3 produces a join you cannot audit.

### Step 1 — Name the decision and its as-of timestamp

Write down the decision, the owner, the cadence, and the moment at which the decision is made.
The timestamp is the single most important input to everything that follows, because it defines
which data was knowable.

- Example: a weekly replenishment order placed Thursday 10:00 for delivery the following
  Wednesday.
- The as-of timestamp is Thursday 10:00. Any value published after that moment is unavailable,
  regardless of the period it describes.
- **Key branch:** is the decision made on a schedule or on an event? Scheduled decisions can use
  release calendars. Event-driven decisions need a rule for what was known at the event, which
  is usually harder to reconstruct.

A common mistake is to define the as-of time as the start of the forecast period. That is wrong.
The forecast for next week is made this week, using data published up to this week. Using data
published during next week is leakage even if the period matches.

### Step 2 — Inventory candidate signals by mechanism

For each candidate, write one sentence of the form: "X affects Y because Z, over a range of A to
B, with a lag of L." If you cannot write the sentence, the candidate is a correlation, and it
belongs in an exploratory analysis, not in the model.

- Classify each candidate as driver or disruptor.
- Classify each as reference (a key or flag) or signal (a covariate).
- Record the native grain and the native frequency.
- **Key branch:** is the signal available at the decision grain, or only at a coarser one? A
  monthly macro series cannot inform a weekly order without an assumption about within-month
  timing.

### Step 3 — Build and maintain the calendar dimension

The calendar dimension is the most-used and least-maintained table in supply chain analytics. It
must contain, for every date: Gregorian date, ISO week, ISO year, fiscal year, fiscal quarter,
fiscal period, fiscal week, week-start day, holiday flags for each relevant country, days to and
from each moving holiday, payday flags, and school-term flags.

- Generate it from rules, not from a spreadsheet maintained by hand.
- Version it. When a retailer changes its fiscal calendar, the old mapping must remain available
  for historical restatement.
- Test it. For every year in the data, confirm that the number of weeks in the fiscal year is 52
  or 53 and that periods sum correctly.
- **Key branch:** does the business report on a 4-5-4, 4-4-5, or 5-4-4 calendar, and does the
  fiscal year start on the Sunday nearest a fixed date or on a fixed date? The two rules diverge
  in some years.

### Step 4 — Resolve the geographic join

Decide how each transaction location maps to each external geography. Options, in increasing
order of effort: nearest station, containing administrative unit, containing grid cell,
population-weighted overlay of grid cells, drive-time catchment.

- Record the method and the match quality for every location.
- Measure the sensitivity: recompute the feature with a second method and correlate the two. If
  the correlation is below roughly 0.9, the join is a modelling choice, not a lookup.
- **Key branch:** is the relevant geography the store, the delivery address, the port, or the
  lane? A weather feature for a distribution centre is irrelevant to a store 300 km away, and a
  port congestion feature is irrelevant to a shipment that moved by rail.

### Step 5 — Retrieve with vintages

For every series that is revised, retrieve the value as published at or before the as-of
timestamp.

- For FRED, use `realtime_start` and `realtime_end`; the default returns the latest revised
  values. The ALFRED archive preserves the releases.
- For statistical agencies, use the release calendar and the archived release files rather than
  the current download.
- For tariff schedules, retrieve the revision in force on the transaction date, not the current
  revision.
- **Key branch:** if the source offers no vintage archive, you cannot use it in a backtest
  without an explicit assumption. Write the assumption down and test its sensitivity.

### Step 6 — Validate without look-ahead

Build the validation split so that every training fold ends before the as-of timestamp of every
test fold, and every feature in the test fold was published before that timestamp.

- Use expanding-window backtests, not random cross-validation.
- Add a placebo test: replace the signal with a version shifted forward in time by one period,
  or with a randomly permuted series of the same autocorrelation. If the model performs as well,
  the signal is not doing what you think.
- Check coefficient stability across folds. A signal whose sign flips is a signal whose mechanism
  is not established.
- **Key branch:** is the target itself revised? If so, the target must also be taken at the
  vintage known at scoring time, or the evaluation measures something no one could have achieved.

### Step 7 — Monitor, document, and retire

External sources change: APIs deprecate, definitions revise, coverage expands, vendors exit.

- Store the source version, retrieval date, and checksum with every extract.
- Monitor the distribution of each feature against its training distribution and alert on shift.
- Set a retirement criterion in advance: for example, retire the signal if its incremental
  out-of-sample error reduction is below a stated threshold for two consecutive quarters.
- **Key branch:** who owns the feed? An unowned external feed is a future outage.

## 6. Worked example

A manufacturer of bagged garden products sells through one national retailer. The product is
strongly seasonal, peaking in the three weeks before Easter. All figures below are illustrative
and internally consistent; the arithmetic is the point, not the magnitudes.

### 6.1 The setup

- Baseline demand outside the season: 5,000 units per week chain-wide, described as index 100.
- Demand rises for three weeks before Easter, peaks in Easter week, and decays over the
  following three weeks.
- The retailer reports point-of-sale (POS) data by fiscal week on a 4-5-4 calendar.
- The manufacturer reports shipments by ISO week.
- The manufacturer's planner builds a weekly forecast keyed on "Easter is in fiscal week 9",
  estimated from the most recent complete year.

### 6.2 The demand profile in event time

Express each week as an offset from Easter Sunday. The profile is stable by construction; only
the mapping to calendar labels changes.

| Offset | Week relative to Easter | Demand index | Units (100 index = 5,000) |
|---:|---|---:|---:|
| −6 | six weeks before | 100 | 5,000 |
| −5 | five weeks before | 110 | 5,500 |
| −4 | four weeks before | 130 | 6,500 |
| −3 | three weeks before | 170 | 8,500 |
| −2 | two weeks before | 220 | 11,000 |
| −1 | one week before | 260 | 13,000 |
| 0 | Easter week | 240 | 12,000 |
| +1 | one week after | 160 | 8,000 |
| +2 | two weeks after | 120 | 6,000 |
| +3 | three weeks after | 105 | 5,250 |
| +4 | four weeks after | 100 | 5,000 |

### 6.3 Fiscal 2024: Easter in fiscal week 9

The retailer's fiscal 2024 began on Sunday 4 February 2024 and ran 52 weeks to 1 February 2025.
The first two periods are four weeks each, and the third is five weeks:

- Period 1: weeks 1–4, 4 February to 2 March.
- Period 2: weeks 5–8, 3 March to 30 March.
- Period 3: weeks 9–13, 31 March to 4 May.

Easter Sunday 2024 fell on 31 March, which is the first day of fiscal week 9 and therefore the
first day of period 3. The three-week ramp (offsets −3 to −1) sits in weeks 6, 7, and 8, all
inside period 2. Easter week itself sits in week 9, inside period 3.

```text
   FY2024   W5   W6   W7   W8  |  W9   W10  W11  W12  W13
   offset   -4   -3   -2   -1  |   0    +1   +2   +3   +4
   index   130  170  220  260  | 240   160  120  105  100
            \_____ period 2 ___/ \______ period 3 ______/
```

Period totals:

- Period 2: 130 + 170 + 220 + 260 = 780 index units over 4 weeks, average 195.
- Period 3: 240 + 160 + 120 + 105 + 100 = 725 over 5 weeks, average 145.

The period-level series therefore shows a peak in period 2 (index 195) and a decline in period 3
(index 145). A model fitted at period grain learns that the season peaks in period 2.

### 6.4 Fiscal 2025: Easter in fiscal week 12

The retailer's fiscal 2025 began on Sunday 2 February 2025, which is consistent with a 52-week
2024 ending on 1 February. Under the same 4-5-4 layout:

- Period 1: weeks 1–4, 2 February to 1 March.
- Period 2: weeks 5–8, 2 March to 29 March.
- Period 3: weeks 9–13, 30 March to 3 May.

Easter Sunday 2025 fell on 20 April, which is fiscal week 12. The ramp now sits in weeks 9, 10,
and 11, and Easter week in week 12. All four peak weeks are inside period 3.

Period totals for the same underlying demand process:

- Period 2: weeks 5–8 at offsets −7 to −4, index 100 + 100 + 110 + 130 = 440 over 4 weeks,
  average 110.
- Period 3: weeks 9–13 at offsets −3 to +1, index 170 + 220 + 260 + 240 + 160 = 1,050 over
  5 weeks, average 210.

### 6.5 What the misalignment costs

Compare the two years at period grain, with no change whatsoever in the demand process:

| Period | FY2024 average index | FY2025 average index | Change |
|---|---:|---:|---:|
| Period 2 | 195 | 110 | −85 (−44%) |
| Period 3 | 145 | 210 | +65 (+45%) |

In units, where 100 index points equal 5,000 units (50 units per index point):

- Period 2 is over-forecast by 85 index points, or 4,250 units per week, over four weeks:
  17,000 excess units.
- Period 3 is under-forecast by 65 index points, or 3,250 units per week, over five weeks:
  16,250 missing units.

The commercial consequences are not symmetric. The excess units are inventory: they occupy space,
consume working capital, and in a seasonal category may become markdown. The missing units are
lost sales and a service failure at the moment of peak demand. If period 3 demand is 10,500 units
per week and supply is planned at 7,250, the fill rate is `7,250 / 10,500 = 69.0%`, before any
recovery through expediting.

Now consider the planner who carries forward the rule "Easter is fiscal week 9". In fiscal 2025:

- Week 9 is offset −3, with true demand index 170, but the rule plans for 240. Over-forecast by
  70 index points, or 3,500 units.
- Week 12 is offset 0, with true demand index 240, but the rule plans 100. Under-forecast by
  140 index points, or 7,000 units.
- The absolute error across these two weeks alone is 210 index points, or 10,500 units.

### 6.6 What actually fixes it

Three changes, in order of importance:

1. **Index time to the event, not only to the calendar.** Add "weeks until Easter" as a feature
   alongside fiscal week. The offset-indexed profile is stable across years by construction,
   which is exactly the property the fiscal-week profile lacks.
2. **Keep the fiscal week as a control.** Retailers change promotional weeks, paydays, and
   assortment resets on fiscal weeks. Removing fiscal week entirely would trade one
   misalignment for another.
3. **Re-estimate the mapping every year.** The offset of Easter within the fiscal year changes
   annually. Any constant mapping will be wrong in most years; it happened to be right in 2024.

### 6.7 What this example does not show

The arithmetic assumes a known demand profile and isolates the calendar effect. It does not
estimate an elasticity, does not account for price or promotion, and does not prove that Easter
is the only driver. A real model must validate the offset feature out of sample, across at least
three Easters, and must check that the residual seasonality is not simply being absorbed by the
offset term because the offset term happens to correlate with spring weather.

> **Note:** Easter dates used here are 31 March 2024 and 20 April 2025. The fiscal-year start
> dates are consistent with a 52-week 4-5-4 year but should be confirmed against a specific
> retailer's published calendar before being used in production.

## 7. Code, query, or calculation

The single most useful artefact for this topic is a calendar-alignment table. Build it once per
retailer and keep it versioned. The table below shows the alignment for the two years in the
worked example; the "offset" column is the feature that survives the misalignment.

| Fiscal week | FY2024 dates | Days to Easter 2024 | FY2025 dates | Days to Easter 2025 |
|---:|---|---:|---|---:|
| 5 | 3–9 Mar 2024 | 28 | 2–8 Mar 2025 | 49 |
| 6 | 10–16 Mar 2024 | 21 | 9–15 Mar 2025 | 42 |
| 7 | 17–23 Mar 2024 | 14 | 16–22 Mar 2025 | 35 |
| 8 | 24–30 Mar 2024 | 7 | 23–29 Mar 2025 | 28 |
| 9 | 31 Mar–6 Apr 2024 | 0 | 30 Mar–5 Apr 2025 | 21 |
| 10 | 7–13 Apr 2024 | −7 | 6–12 Apr 2025 | 14 |
| 11 | 14–20 Apr 2024 | −14 | 13–19 Apr 2025 | 7 |
| 12 | 21–27 Apr 2024 | −21 | 20–26 Apr 2025 | 0 |
| 13 | 28 Apr–4 May 2024 | −28 | 27 Apr–3 May 2025 | −7 |

Read the table as a warning. Fiscal week 9 is Easter week in 2024 and three weeks before Easter
in 2025. A join on fiscal week number is a join on a label that means different things in
different years. The offset column means the same thing every year, which is why it generalises.

Two formulas make the table reproducible from a date dimension:

```text
days_to_easter(d) = easter_date(year(d)) - d          (in days, signed)
fiscal_week(d)    = floor((d - fiscal_year_start(year(d))) / 7) + 1
```

The Easter date is computed from the Gregorian computus, not from a lookup table, if you need it
for years outside your history. The fiscal week formula is only valid for a 4-5-4 calendar in
which every year starts on the same weekday; for a 53-week year, the week count must be carried
explicitly.

## 8. Data, standards, and regulatory context

### 8.1 Calendar standards

| Standard or source | What it defines | Access | Grade |
|---|---|---|---|
| NRF 4-5-4 calendar | Retail merchandising and sales-reporting calendar, including the 53-week rule | Published PDFs by year range | B |
| ISO 8601 | Week numbering, date and time representation | Standard text; implemented in most languages | B |
| Retailer fiscal calendars | The calendar a specific retailer actually uses, including period boundaries | Vendor portals or the retailer's finance function | B |
| Public holiday tables | National and subnational holiday dates and observances | Government labour, interior, or statistics agencies | B |

The National Retail Federation's rule for the 53rd week is explicit: after laying out 52 weeks,
if four or more days remain in January, a 53rd week is added. Documented 53-week years include
1995, 2000, 2006, 2012, 2017, and 2023, with fiscal 2028 also scheduled as a 53-week year in the
NRF 2026–2028 calendar. In a 53-week year, week numbers after the inserted week shift, and
year-over-year comparisons must either restate the prior year or exclude the extra week.

> **Note:** The NRF publishes a 4-5-4 calendar. Many retailers and most trade press refer to
> "the NRF 4-4-5 calendar", which conflates two variants. Verify the variant before using any
> published week numbering.

### 8.2 Weather and climate sources

| Source | What it provides | Native resolution | Access and licence | Grade |
|---|---|---|---|---|
| NOAA NCEI Climate Data Online | Station observations, normals, degree days | Station, daily or hourly | Free API with token; 5 requests/second, 10,000/day | B |
| NCEI 1991–2020 normals | Daily and monthly climate normals, including HDD and CDD | Station, daily | Free bulk files and API | B |
| ECMWF open data | IFS and AIFS forecasts | 0.25° grid, four cycles daily | CC-BY-4.0; rolling 12-run archive | B |
| Copernicus Climate Data Store | Reanalysis and derived climate products | Gridded, various | Free registration; licence per dataset | B |
| National meteorological services | Local forecasts and warnings | Station or grid | Varies by country | B |
| Commercial weather vendors | Forecasts, historical series, industry-specific indices | Grid or postcode | Paid; terms vary | D |

ECMWF opened its entire real-time catalogue on 1 October 2025 under CC-BY-4.0, with the free
subset at approximately 0.25° (about 25 km) resolution; a 0.1° (about 9 km) IFS subset was added
in October 2026. The practical consequence for a supply chain analyst is that a medium-range
forecast at 0.25° is now free, which is adequate for regional demand drivers and inadequate for a
single store's microclimate.
The forecast horizon matters more than the resolution for most planning uses: operational skill
at day 10 is at record levels, but skill decays with lead time, and a 10-day forecast for a
specific storm location carries substantial uncertainty even when its broad pattern is right.

### 8.3 Macroeconomic sources

| Source | Content | Frequency | Revision behaviour | Grade |
|---|---|---|---|---|
| FRED | Aggregated economic time series | Daily to annual | Serves latest revised values by default | B |
| ALFRED | Vintage archive of FRED series | As released | Preserves every release | B |
| Eurostat | European official statistics | Monthly, quarterly | First release then revisions; release calendar published | B |
| OECD | Composite leading indicators and member statistics | Monthly | Monthly revisions from filtering and component updates | B |
| National statistics offices | GDP, retail sales, industrial production | Monthly or quarterly | Advance, preliminary, final, benchmark | B |
| IMF and World Bank | Cross-country aggregates and forecasts | Monthly to annual | Forecast vintages and data revisions | B |

The OECD documents that its composite leading indicators are revised monthly because component
series pass through seasonal adjustment, outlier detection, de-trending, and smoothing, and that
the smoothing filters cause the largest revisions. This is not a defect. It is what a composite
indicator is. The analytical consequence is that a CLI value quoted today for a past month is
not the value a planner saw at the time.

### 8.4 Trade and policy sources

| Source | Content | Granularity | Update cadence | Grade |
|---|---|---|---|---|
| WTO Tariff and Trade Data | Applied and bound tariffs, imports | HS 6-digit, by reporter and year | Analytical Database updated monthly; 1996 onward | B |
| UN Comtrade | Bilateral trade flows | HS 6-digit, by reporter, partner, year | Continuous; recent years incomplete | B |
| USITC HTS | United States tariff schedule with rates and notes | 8- and 10-digit lines | Basic edition plus frequent revisions | B |
| EU TARIC | EU tariff measures, including quotas and preferences | 10-digit codes | Daily updates; full database dumps | B |
| OFAC Sanctions List Service | US sanctions lists, including SDN | Entity and vessel identifiers | Continuous; delta files by year | B |
| EU consolidated sanctions list | EU financial sanctions | Entity identifiers | Updated approximately daily | B |

Two features of this group matter more than the content. First, **the schedule is a document with
effective dates, not a table with current values.** The 2025 United States HTS ran to at least 32
revisions, with Revision 32 dated 5 December 2025, and the 2026 edition already had 21 revisions
by early October 2026. A tariff rate is only meaningful with the revision and the effective date.
Second, **coverage is uneven.** UN Comtrade annual data for the most recent year is populated by
the fastest reporters only; a practitioner rule of thumb is to treat the year two years before
the current one as the newest safely complete year, and to check availability per reporter. That
heuristic is Grade E; the underlying API behaviour is Grade B.

### 8.5 Geospatial sources

| Source | Content | Resolution | Access and licence | Grade |
|---|---|---|---|---|
| OpenStreetMap | Roads, buildings, land use, points of interest | Vector, variable completeness | Open; attribution and share-alike terms for derived databases | B for licence, A for quality studies |
| Sentinel-2 | Multispectral imagery | 10 m, 20 m, 60 m bands | Free and open under the Copernicus programme | B |
| IMF PortWatch | Daily port calls and chokepoint transits from AIS | Port-day | Free; weekly update on Tuesdays | B |
| National road authorities | Traffic counts and incident feeds | Point or segment | Varies | B |
| Commercial AIS and mobility vendors | Vessel, footfall, card panels | Fine-grained | Paid; contractual restrictions | D |

PortWatch reports daily port calls and trade estimates for 1,666 ports and 24 chokepoints, built
from satellite AIS for roughly 120,000 ships, with series from 1 January 2019 and a weekly update
on Tuesdays at 09:00 Eastern Time. Its 2025 working paper documents a seven-working-day lag and
correlations of about 0.95 in value and 0.80 in volume with official trade data. Those are
nowcast properties, not measurement properties: the series estimates activity before customs
data exists, and it is revised as AIS coverage and port boundaries improve.

### 8.6 Publication lag as a first-class attribute

| Series | Typical first release | Typical revision pattern | Usable as-of |
|---|---|---|---|
| Official retail sales | Weeks after month end; varies by country | Revised for one or more months | Release date |
| GDP advance estimate | About a month after quarter end; varies by country | Second, third, annual, benchmark | Release date |
| OECD CLI | Monthly, with component lags | Every month, back to series start | Release date |
| Port calls (PortWatch) | Weekly, ~7 working days | Methodology and coverage revisions | Release date |
| Tariff schedule | On effective date, with notice | Many revisions per year | Revision in force |
| Sanctions list | On designation | Deltas plus full refresh | Publication timestamp |

Any feature built from these series must carry the release date and the revision identifier. A
table that stores only the period and the value cannot be audited.

## 9. Industry practice and case evidence

### 9.1 What the weather evidence actually shows

The peer-reviewed record is consistent in direction and modest in magnitude. Three studies
illustrate the range.

- Rose and Dolega (2021) used 1.3 billion transactions from 2,043 United Kingdom stores and a
  random forest model. Weather added about 3.5 percentage points of R² overall and up to 6.4
  points for health foods. Wind was the most influential weather variable, followed by
  temperature, and effects were strongest in spring and summer and in London and high-street
  stores (Grade A).
- Badorf and Hoberg (2020) studied daily weather effects in brick-and-mortar stores and reported
  roughly 4% additional explained variance (Grade A, single-market).
- Chan and Wahab (2024) reported weather explaining up to 47% additional variance for individual
  products and 56% for categories beyond a baseline, in a large Canadian retailer (Grade A,
  single-retailer case; the upper figures apply to weather-sensitive products, not to the
  assortment average).

The honest reading is that weather is worth including where a mechanism exists, that the gain is
typically a few percentage points of variance at the category level, and that headline figures
above 40% describe product-level fits in weather-sensitive categories. A business case that
promises 40% improvement across a portfolio is not supported by this literature.

### 9.2 What the calendar evidence shows

Calendar effects are well established and harder to estimate than they look. The United States
Census Bureau's work on moving holidays documents that Easter is the most common moving-holiday
effect in United States economic series, that retail sales are elevated in the days before
Easter, and that the effect window (for example `easter[8]`, eight days before) must be estimated
rather than assumed. Monsell (2007) compared Easter regressors across 16 to 17 Census retail
series and found that only about 6 of 16 showed a genuine out-of-sample forecast improvement over
a model with no Easter term, even though the Easter term was often selected by information
criteria (Grade A). Zhang, McLaren, and Leung (2001) found that a linear pre-Easter buildup fit
Australian retail turnover better than a constant-level window (Grade A).

The lesson is not that Easter does not matter. It is that the effect is series-specific, that
the window is an estimated parameter, and that in-sample selection criteria overstate the benefit.
A retailer with a strong seasonal category will see it. A portfolio model applied across
thousands of items will mostly see noise.

Payday effects are similarly real and similarly uneven. Hastings and Washington (2010), using
Nevada scanner data, found grocery spending falling about 20% from the first to the second week
of the benefit month and about 34% by the fourth week, driven by quantity rather than quality,
with store prices falling about 3% by week four (Grade A). Pagel and Vardardottir, using Icelandic
payment records, report that more than half of the population raises spending by more than 25% on
paydays, and that only about 3% hold less than one day's spending in liquid savings, which argues
against a pure liquidity explanation (Grade A, as summarised by Columbia Business School).
Payday is therefore a legitimate feature for consumer-facing demand at daily grain, and a weak
one at monthly grain where the effect averages out.

School terms are the weakest of the calendar signals in the evidence base. Saputra and Kumar
(2024) modelled school holidays separately in an intervention analysis and found a positive but
statistically insignificant coefficient (165.2, p = 0.92), and a German grocery study identified
the year-to-year overlap of school and public holidays as a source of forecast error rather than
estimating a school effect (Grade A for the null result, Grade A for the qualitative finding).
Use school terms as a control where the mechanism is local, not as a headline driver.

### 9.3 Port, satellite, and geospatial evidence

The strongest recent evidence links physical activity measured from space to economic quantities.
Spelta, Pagnottoni, and Pecora (2025), in the Journal of the Royal Statistical Society Series A,
built a Port Saturation Index from Sentinel-2 imagery at 10 m resolution for 16 major European
container ports, using a U-Net segmentation model with about 88% global accuracy. Feeding the
index into dynamic factor models reduced monthly GDP nowcast error by roughly 8% in sample, with
area-under-curve values of about 84.7 for upturns and 77.7 for downturns (Grade A). Import and
export gains were smaller and less consistent.

AIS-based congestion measures are an established nowcasting input, with 2025 work moving toward
richer models: inverse reinforcement learning on berth scheduling, state-space models across
twenty United States East Coast terminals, and attention-based networks on multi-year daily
observations (Grade A for the peer-reviewed work; the reported accuracies of 74% to 95% are
task-specific and not comparable across studies). The inference chain is long in every case:
vessel position to port call to berth occupancy to container throughput to trade value. Each
link is an assumption, and the reported accuracy is for a specific link, not for the whole chain.

OpenStreetMap quality is well studied and geographically uneven. Li et al. (2025) compared OSM
road data with United States Census TIGER data at county level and found systematically better
completeness and positional accuracy in metropolitan counties than in rural ones (Grade A). For
a logistics model, this means routing features are more reliable in dense markets and less
reliable exactly where alternative routes matter most.

### 9.4 Macro data: the revision is the story

Croushore and Stark (2003) established the real-time dataset for macroeconomists and showed that
the choice of vintage can change published empirical results (Grade A). The St. Louis Fed's own
documentation gives the canonical example: the first estimate of 2010:Q2 real GDP was 2.4%, later
revised to 1.6% (Grade B). The OECD documents that its leading indicators are revised monthly
because of filtering, with smoothing as the dominant driver of revisions (Grade B).

For supply chain forecasting the practical implication is narrower than the macro literature
suggests. Most demand models use macro variables as controls, not as the primary signal. A
one-percentage-point revision in GDP growth rarely changes a weekly order. What does change is
the *evaluation*: a backtest built on revised GDP will report accuracy that cannot be reproduced,
and the error is largest around turning points, which is when the forecast matters most.

### 9.5 What is not evidenced here

Commercial alternative-data vendors sell footfall panels, card-transaction panels, and
satellite-derived activity indices. Their published accuracy claims are marketing material
(Grade D) and are not cited as evidence in this topic. The published research that does exist
tends to evaluate a specific dataset in a specific market, and the results do not transfer
automatically. Treat a vendor claim as a hypothesis to be tested on your own data, with your own
as-of timestamps, before it enters a planning process.

The general caution is supported. Compton et al. (2023) found that adding external datasets
degraded worst-group accuracy in 43% of tested settings by inducing spurious correlations
(Grade A, medical imaging). Bailey and López de Prado (2021) show that as few as three
independent trials can turn a null strategy into an apparently significant one, and that 65% to
82% of 452 published anomalies fail replication (Grade A). The supply chain analogue is a
feature store with hundreds of candidate external signals and no correction for multiple testing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Fiscal-calendar drift | Seasonal peak moves between periods with no demand change | Fiscal year start shifts relative to Gregorian dates | Wrong period-level forecast; inventory in the wrong month | Offset-indexed features; versioned calendar dimension |
| Easter week reuse | "Easter is week 9" persists after year one | Easter offset changes annually | Peak forecast three weeks early | Recompute offset every year; store days-to-Easter |
| 53-week year unhandled | Period totals jump 2% with no volume change | Extra week inserted; week numbers shift | False trend; distorted year-over-year | Flag 53-week years; restate or exclude |
| ISO and retail weeks mixed | Weekly totals differ by system | Monday versus Sunday week start | Joins drop or duplicate a day | One canonical week definition; conversion table |
| Macro vintage leakage | Backtest accuracy far above live performance | Revised values used for historical features | Model cannot be reproduced in production | `realtime_start`/`realtime_end`; ALFRED vintages |
| Publication lag ignored | Feature contains data released after the order | Period alignment mistaken for availability | Silent look-ahead | As-of retrieval; release calendar audit |
| Weather at the wrong location | Signal significant in aggregate, useless by region | Nearest station or grid cell does not represent catchment | Misallocated inventory by store | Catchment weighting; sensitivity to join method |
| Weather as driver where it disrupts | Model predicts high sales in storm weeks | Disruption modelled as demand shift | Stock builds into a closed store | Separate driver and disruptor models |
| Degree-day base mismatch | Two HDD series diverge by 20% | Different base temperature or unit | Coefficient not transferable | Carry base and unit with every value |
| Spurious correlation from search | Many signals tested; one looks strong | Multiple comparisons | Overfit feature set; unstable coefficients | Pre-register mechanism; deflated significance |
| Geographic coverage bias | Model performs well in cities, poorly rurally | Open data completeness varies by area | Service failure in underserved regions | Coverage diagnostics by region |
| Tariff revision mismatch | Duty cost differs from invoice | Current schedule applied to old shipment | Cost and landed-cost error | Effective-dated rate table |
| Sanctions false positives | Screening blocks legitimate partners | Name matching without identifiers | Delayed shipments; compliance friction | Use identifiers, not names alone |
| Feed discontinued | Feature silently null after vendor change | No ownership or monitoring | Model degrades without alert | Contract, checksum, freshness alert |

Two deserve emphasis.

**Macro vintage leakage** is the failure that most often survives review, because the backtest
looks excellent and the feature is legitimate. The only reliable defence is to retrieve every
revised series with an explicit as-of parameter and to store the release date alongside the
value. If the source cannot provide either, the feature belongs in a scenario analysis, not in a
backtested model.

**Spurious correlation from signal search** is the failure that most often gets published
internally as a success. The mechanism is simple: test 50 external signals against one demand
series, and roughly two will appear significant at the 5% level even if none is real. The
mitigation is procedural, not statistical: write the mechanism sentence first, limit the number
of candidates, and pre-register the evaluation.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision timestamp is known and stable | Defines the information set | Decisions made asynchronously across regions | Audit decision logs; sample cases |
| The external source measures what it claims | Every feature is a proxy | Instrument change, coverage change | Compare with an independent source |
| The join geography is a good proxy for exposure | Transactions and signals use different geographies | Catchment shifts, store relocations | Sensitivity across two join methods |
| The signal's relationship is stable over the horizon | Coefficients are estimated on history | Regime change, new competitor, policy shift | Rolling coefficient plots |
| Publication lag is known and constant | As-of retrieval depends on it | Agency delays, outages | Compare release calendar with actual |
| Revisions are small relative to the signal | Otherwise vintage choice dominates | Benchmark revisions, methodology changes | Recompute features across vintages |
| The calendar variant is documented | Period comparisons depend on it | Retailer changes calendar; M&A | Reconcile totals to retailer reports |
| Coverage is adequate for the geography used | Sparse data produces unstable features | Rural areas, small ports, emerging markets | Coverage diagnostics by segment |
| The signal is not a mediator of the treatment | Causal interpretation requires it | Signal sits between price and demand | Draw the causal graph before modelling |
| More signals improve the model | Intuition from feature-rich domains | Overfitting, collinearity, spurious correlation | Ablation and placebo tests |

Two limits of the whole approach are worth stating plainly.

**External signals explain conditional variation, not total demand.** They shift a baseline. If
the baseline is wrong because the product, price, distribution, or competitive position changed,
no amount of weather or macro data will fix the forecast. The base model must be adequate before
external signals are added; otherwise the signals absorb the base model's error and the
coefficients become uninterpretable.

**The value of a signal is decision-specific.** The same weather series can be valuable for
short-cycle replenishment and worthless for a twelve-month capacity plan, because the forecast
horizon exceeds the skill horizon. Relevance is not a property of the dataset. It is a property
of the dataset, the decision, and the horizon together, which is why source selection (Topic 10)
cannot be settled by a data-quality score alone.

## 12. Exercises

### Exercise 1 — Build a date dimension (application, 60 minutes)

Build a date table covering five years for one retailer, containing Gregorian date, ISO week,
fiscal year, fiscal quarter, fiscal period, fiscal week, days to Easter, and flags for three
national holidays.

**Deliverable:** a table with one row per date and a written definition of each column.
**Acceptance criteria:** the fiscal year contains 52 or 53 weeks in every year; period week
counts follow the stated 4-5-4 or 4-4-5 pattern; days-to-Easter is correct for at least three
known years; every column has a stated unit or coding system.

### Exercise 2 — Calendar alignment audit (analysis, 75 minutes)

Using the worked example in section 6, recompute the period averages for a retailer whose fiscal
year starts one week later, and state how much the apparent seasonality changes.

**Deliverable:** a table of period averages under both start dates, and a short paragraph on the
implication for a period-level forecast.
**Acceptance criteria:** arithmetic shown for at least two periods; the direction of the error
stated; one sentence on what would happen if the analyst reused the previous year's mapping.

### Exercise 3 — Weather signal diagnostic (analysis, 90 minutes)

Choose a product with a plausible weather mechanism. State the mechanism sentence, the exposure
window, and the geographic join you would use. Then compute the correlation between the signal
and demand at three different lags.

**Deliverable:** the mechanism sentence, a lag-correlation table, and a recommendation.
**Acceptance criteria:** the exposure window is justified, not assumed; at least one alternative
join method is tested; the recommendation states the expected gain and the condition under which
it would not appear.

### Exercise 4 — Vintage reconstruction (analysis, 75 minutes)

Take one revised macro series. Retrieve its value as published at two different dates for the
same reference period and quantify the difference.

**Deliverable:** a table of values by vintage, the revision magnitude in levels and percent, and
a statement of the leakage risk.
**Acceptance criteria:** at least three vintages; the revision direction stated; an explicit
statement of which value a planner would have seen at the decision date.

### Exercise 5 — Tariff line mapping (application, 60 minutes)

Map three imported components to their tariff lines, and record the rate, the schedule revision,
and the effective date.

**Deliverable:** a mapping table with component, HS heading, national line, rate, revision, and
source.
**Acceptance criteria:** each line traceable to a specific revision; at least one component
identified where classification is ambiguous; the ambiguity explained.

### Exercise 6 — Geospatial inference chain (critique, 60 minutes)

Take one geospatial source (AIS port calls, satellite imagery, or footfall) and write the chain
from measurement to the supply chain quantity you would use.

**Deliverable:** a chain diagram with at least five links, each labelled with its assumption.
**Acceptance criteria:** every link identifies what could break; at least one link is quantified
with a number from a cited source; the final quantity is defined with units.

### Exercise 7 — Placebo test (analysis, 75 minutes)

Take one external feature from a model you know, or construct one. Refit the model with the
feature shifted forward by one period and with a permuted version preserving autocorrelation.

**Deliverable:** a comparison table of out-of-sample error across the real, shifted, and permuted
features.
**Acceptance criteria:** the split is time-ordered; the shifted feature is constructed correctly;
the conclusion distinguishes "no signal" from "signal not captured by this specification."

### Exercise 8 — Signal retirement memo (design, 45 minutes)

Write a one-page memo proposing a retirement criterion for an external feed used in planning.

**Deliverable:** the criterion, the measurement, the review cadence, and the owner.
**Acceptance criteria:** the criterion is quantitative; the review cadence is realistic; the
owner is a role, not a department; the memo states what happens to historical features on
retirement.

### Exercise 9 — Source dossier with grades (research, 90 minutes)

Assemble a dossier for one external source you would actually use, covering access, licence,
native grain, publication lag, revision behaviour, coverage, and known defects.

**Deliverable:** a two-page dossier with an evidence grade for each factual claim.
**Acceptance criteria:** at least six attributes documented; every grade justified; at least one
limitation that would change a modelling decision.

## 13. Self-check answers

**Exercise 2 arithmetic.**

Take the fiscal year starting one week later than in section 6, so that period 2 runs from
10 March to 6 April and period 3 from 7 April to 11 May. Easter 2024 remains 31 March.

- Period 2 now contains offsets −3, −2, −1, and 0: 170 + 220 + 260 + 240 = 890 over four weeks,
  average 222.5.
- Period 3 now contains offsets +1 to +5, roughly 160 + 120 + 105 + 100 + 100 = 585 over five
  weeks, average 117.

Compare with the original alignment: period 2 average 195 and period 3 average 145. A one-week
change in the fiscal year start moves the period 2 average by +27.5 index points (+14%) and the
period 3 average by −28 (−19%). The apparent seasonality is not a property of demand. It is a
property of where the period boundary falls relative to the holiday.

If the analyst reuses the previous year's mapping, the error compounds: the mapping encodes both
the fiscal-start rule and the Easter offset, and both change independently.

**Exercise 4 expected result.**

Revisions are usually small in normal times and larger around turning points. A typical pattern
for a quarterly GDP series is a first estimate that is later revised by a few tenths of a
percentage point, with occasional revisions above one point during recessions or methodological
changes. The direction is not systematic, which is exactly why the leakage is dangerous: it does
not bias the model in a predictable direction, it inflates the apparent precision of the backtest.
A planner at the decision date saw the first estimate. Any other value is hindsight.

The correct statement is not "the revised value is wrong." It is "the revised value answers a
different question: what we now believe happened, rather than what was believed at the time."

**Exercise 7 rubric.**

A strong submission uses a time-ordered split, constructs the shifted feature by moving values
forward without recomputing them from future data, and reports that the permuted feature performs
at or near the real feature's level when the model is overfitting. It distinguishes two outcomes:
the real feature beats both placebos (evidence of signal), or it does not (no evidence, regardless
of in-sample fit). A weak submission uses random cross-validation, which leaks across time, or
reports only the real-feature result.

**Exercise 1 rubric.**

Full marks require: 52 or 53 weeks per fiscal year, verified by summing period weeks; correct
handling of at least one 53-week year; a documented week-start day; days-to-Easter computed from
the computus or a verified table; and a written definition for every column. A common failure is
to generate the calendar from a single year's pattern and repeat it, which silently breaks in
53-week years and at fiscal-start boundaries.

**Exercise 6 rubric.**

A strong chain for AIS port calls reads: vessel broadcast → terrestrial or satellite reception →
position cleaning and gap filling → port boundary assignment → arrival and departure inference →
daily call count → throughput proxy → import volume estimate. Each link carries an assumption:
reception coverage, identity resolution, boundary definition, dwell threshold, and the
call-to-cargo ratio. A weak answer jumps from "AIS data" to "port congestion" without naming any
intermediate step, which makes the error untraceable.

**Exercise 9 rubric.**

A strong dossier separates what the publisher states from what independent studies find, grades
each separately, and names at least one defect that would change a decision, such as a two-year
lag in annual trade data or a coverage gap in rural geographies. A weak dossier reproduces the
publisher's documentation without an independent check and assigns a single overall grade.

**Design rubric for the whole topic.** Any design that uses external signals should answer five
questions in writing: what is the mechanism, what is the as-of timestamp, what is the join, how
was it validated without look-ahead, and when will it be retired. A design that answers fewer
than four is not ready for production, regardless of its measured accuracy.

## 14. Interview questions

**1. "What is the difference between reference data and an external signal?"**

A strong answer separates the join layer (calendars, codes, schedules) from the covariate layer
(weather, macro, traffic), and notes that reference data failures present as join failures while
signal failures present as model failures. A weak answer treats both as "external data" and gives
no operational distinction.

**2. "Why is a 4-5-4 calendar a problem for a forecast?"**

A strong answer explains that the fiscal year is 364 days, that period boundaries move relative
to Gregorian dates, that a 53rd week is inserted periodically, and that a fixed event such as
Easter lands in a different fiscal week each year. A weak answer says "it is not the normal
calendar" and stops.

**3. "You are told Easter is in week 9. What do you check?"**

A strong answer checks which year, which calendar variant, and whether week 9 is Easter week or
three weeks before it; then converts to a days-to-Easter offset. A weak answer accepts the
statement and builds a dummy variable for week 9.

**4. "How would you use weather in a demand model?"**

A strong answer separates driver from disruptor, states the exposure window and the geographic
join, and cites the modest evidence base (a few percentage points of variance at category level).
A weak answer says "add temperature as a feature" without a mechanism or a join.

**5. "What is a degree day and why does the base matter?"**

A strong answer defines HDD and CDD relative to a base, notes that 65 °F is a convention, and
explains that two series with different bases are not comparable. A weak answer gives the formula
without the base.

**6. "Why is revised macro data a leakage risk?"**

A strong answer explains that the current vintage contains information not available at the
decision date, that revisions are largest around turning points, and that the fix is as-of
retrieval with `realtime_start`/`realtime_end` or an equivalent archive. A weak answer says
"revisions are small, so it does not matter."

**7. "Leading, coincident, lagging — how do you tell?"**

A strong answer says the classification is a claim about mechanism and timing, that it must be
tested rather than assumed, and that a leading indicator can become coincident after a structural
change. A weak answer defines the three by correlation at different lags and treats that as
sufficient.

**8. "How would you validate an external signal?"**

A strong answer describes a time-ordered backtest, as-of feature construction, a placebo or
permutation test, coefficient stability across folds, and a pre-registered mechanism. A weak
answer reports in-sample correlation or a single holdout.

**9. "A vendor claims 30% forecast improvement from their footfall data. What do you ask?"**

A strong answer asks for the baseline, the evaluation split, the as-of construction, the
geography and coverage, the licence terms for derived models, and whether the result replicates
on your data. A weak answer asks for the price.

**10. "What is the inference chain from a satellite image to a supply chain quantity?"**

A strong answer names pixels, segmentation, physical area, occupancy, throughput, and value, and
states that accuracy at one link does not imply accuracy at the end. A weak answer says
"satellites measure activity" without intermediate steps.

**11. "When would you deliberately not use an external signal?"**

A strong answer cites horizon beyond skill (weather beyond about ten days), unstable definitions,
no vintage archive for a backtest, a mechanism that is really a mediator, or a decision where the
signal cannot change the action. A weak answer cannot imagine a case.

**12. "How do you stop external-signal search from overfitting?"**

A strong answer combines a mechanism-first candidate list, a limit on the number of tests, a
deflated significance threshold or holdout, and a monitoring/retirement rule. A weak answer says
"use cross-validation", which does not address multiple comparisons across signals.

## 15. Key terms

- **Reference data** — slow-changing lookup data used to join and interpret other data.
- **External signal** — a time-varying measurement from outside the firm used as a covariate.
- **Calendar dimension** — the date table mapping each day to every calendar in use.
- **4-5-4 calendar** — 13-week quarters split into 4-, 5-, and 4-week periods.
- **4-4-5 calendar** — the variant starting the Sunday nearest 1 February.
- **53-week year** — a retail year with an inserted week, roughly every five to six years.
- **ISO 8601 week** — Monday-start week numbering; not the same as a retail week.
- **Moving holiday** — a holiday whose date changes annually, such as Easter.
- **Easter effect** — the demand distortion from Easter's date and pre-holiday buildup.
- **Trading-day effect** — distortion from the weekday composition of a period.
- **Degree day** — heating or cooling demand relative to a stated base temperature.
- **Climate normal** — a 30-year average used as a reference, not a forecast.
- **Reanalysis** — a retrospective, consistent reconstruction of past weather.
- **Grid cell** — a rectangular unit of a gridded dataset.
- **Catchment** — the geography from which a store draws customers.
- **Leading indicator** — a series that tends to move before the target.
- **Coincident indicator** — a series that moves with the target.
- **Lagging indicator** — a series that moves after the target.
- **Vintage** — the state of a dataset at one release date.
- **Revision** — a change to a previously published value.
- **Benchmark revision** — a large periodic revision from new sources or methods.
- **Nowcast** — an estimate of the present before official data exists.
- **HS code** — the international 6-digit tariff nomenclature.
- **HTS / TARIC** — the United States and European national tariff extensions.
- **MFN rate** — the tariff applied absent a preferential agreement.
- **Rules of origin** — the criteria for preferential tariff eligibility.
- **Sanctions list** — a published list of restricted parties, with identifiers and programmes.
- **AIS** — vessel position broadcasts used to infer port activity.
- **Spatial join** — assignment of a point to a containing or nearest geographic unit.
- **Alternative data** — externally sourced data not produced for the analytical purpose.

## 16. Sources

**Peer-reviewed and academic**

- Rose and Dolega (2021). "It's the Weather: Quantifying the Impact of Weather on Retail
  Sales." *Applied Spatial Analysis and Policy*. DOI: 10.1007/s12061-021-09397-0. Grade A.
- Badorf and Hoberg (2020). "The impact of daily weather on retail sales: An empirical
  study in brick-and-mortar stores." *Journal of Retailing and Consumer Services*. DOI:
  10.1016/j.jretconser.2019.101921. Grade A.
- Chan and Wahab (2024). "A machine learning framework for predicting weather impact on
  retail sales." *Supply Chain Analytics*. DOI: 10.1016/j.sca.2024.100058. Grade A, single
  retailer.
- Zhang, McLaren and Leung (2001). "An Easter Proximity Effect: Modelling and
  Adjustment." *Australian & New Zealand Journal of Statistics*, 43(3), 269–280. DOI:
  10.1111/1467-842X.00173. Grade A.
- Monsell (2007). "Issues in Modeling and Adjusting Calendar Effects in Economic Time
  Series." *Proceedings of the 2007 ICES* (American Statistical Association). Grade A/B.
- Findley, Wills and Monsell (2005). "Issues in Estimating Easter Regressors Using
  RegARIMA Models with X-12-ARIMA." U.S. Census Bureau working paper. Grade B.
- Obermair, Holzapfel and Kühn (2022). "Operational Planning for Public Holidays in
  Grocery Retailing." *Operations Management Research*. DOI: 10.1007/s12063-022-00342-z. Grade A.
- Saputra and Kumar (2024). "Modeling the Impact of Holidays and Events on Retail Demand
  Forecasting in Online Marketing Campaigns using Intervention Analysis." *Journal of Digital
  Market and Digital Currency*, 1(2). DOI: 10.47738/jdmdc.v1i2.9. Grade A for peer review;
  single small study, treat as indicative.
- Hastings and Washington (2010). "The First of the Month Effect: Consumer Behavior and
  Store Responses." *American Economic Journal: Economic Policy*, 2(2), 142–162. DOI:
  10.1257/pol.2.2.142. Grade A.
- Agarwal, Marwell and McGranahan (2017). "Consumption Responses to Temporary Tax
  Incentives: Evidence from State Sales Tax Holidays." *American Economic Journal: Economic
  Policy*. DOI: 10.1257/pol.20130429. Grade A.
- Croushore and Stark (2003). "A Real-Time Data Set for Macroeconomists: Does the Data
  Vintage Matter?" *Review of Economics and Statistics*, 85(3), 605–617. Grade A.
- Spelta, Pagnottoni and Pecora (2025). "Big brothering the economy: nowcasting and
  forecasting with port satellite images." *Journal of the Royal Statistical Society Series A*,
  189(1), 463. DOI: 10.1093/jrsssa/qnae157. Grade A.
- Zhang et al. (2024). "Prediction of container port congestion status and its impact on
  ship's time in port based on AIS data." *Maritime Policy & Management*, 51(5), 669–697.
  Grade A.
- "Temporal-IRL: Modeling Port Congestion and Berth Scheduling with Inverse Reinforcement
  Learning." arXiv:2506.19843, 2025. Grade A preprint.
- "Predicting Ports Congestion by Utilizing State Space Models." IEEE High Performance Extreme
  Computing Conference, 2025. DOI: 10.1109/hpec67600.2025.11196391. Grade A.
- "Doing shipping well with predictions: machine learning-based port congestion analysis."
  *Maritime Policy & Management*, 53(2), 2025. DOI: 10.1080/03088839.2025.2483993. Grade A.
- Li et al. (2025). "Mapping the Completeness and Positional Accuracy of OpenStreetMap Road
  Data at the County Level in the Contiguous United States." *Transactions in GIS*. DOI:
  10.1111/tgis.70077. Grade A.
- Compton et al. (2023). "When More is Less: Incorporating Additional Datasets Can Hurt
  Performance By Introducing Spurious Correlations." arXiv:2308.04431. Grade A preprint.
- Bailey and López de Prado (2021). "How 'Backtest Overfitting' in Finance Leads to
  False Discoveries." *Significance*, 18(6), 22. Grade A.
- Liu, K. (2024). "The measurement errors of Google Trends data." *Discover Data*. DOI:
  10.1007/s44248-024-00013-3. Grade A.
- Medeiros and Pires. "The Proper Use of Google Trends in Forecasting Models."
  arXiv:2104.03065. Grade A preprint.
- Fleder and Shah (2019). "Forecasting with Alternative Data." *Proceedings of the ACM on
  Measurement and Analysis of Computing Systems*. DOI: 10.1145/3366694. Grade A.
- Hansen and Borch (2022). "Alternative data and sentiment analysis: Prospecting
  non-standard data in machine learning-driven finance." *Big Data & Society*. DOI:
  10.1177/20539517211070701. Grade A.

**Standards and official statistics**

- National Retail Federation, "4-5-4 Calendar," nrf.com/resources/4-5-4-calendar, and the
  2026–2028 Retail Sales Reporting and 4-5-4 Merchandising Calendar. Grade B.
- ISO 8601, date and time representation standard. Grade B.
- NOAA National Centers for Environmental Information, "Climate Data Online Web Services API v2"
  and "Daily Normals 1991–2020" documentation. Grade B.
- ECMWF, "Open data" and "ECMWF makes its entire Real-time Catalogue open to all," 1 October
  2025. Grade B.
- ECMWF, "Evaluation of ECMWF forecasts," Technical Memorandum 931, September 2025. Grade B.
- U.S. International Trade Commission, "2025 HTS Revision 32," 5 December 2025, and the HTS
  archive. Grade B.
- European Commission, "TARIC Consultation," and TARIC full and daily database downloads.
  Grade B.
- WTO, Tariff and Trade Data portal, and the WTO Analytical Database (first published March
  2025). Grade B.
- UN Comtrade, "UN Comtrade API" documentation and data availability endpoints. Grade B.
- U.S. Department of the Treasury, OFAC "Sanctions List Service" and FAQ 1641. Grade B.
- European Union, "Consolidated list of persons, groups and entities subject to EU financial
  sanctions," data.europa.eu. Grade B.
- IMF PortWatch, "Daily Port Activity Data and Trade Estimates," and IMF Working Paper WP/25/93,
  "Nowcasting Global Trade from Space," May 2025. Grade B.
- OECD, "Composite Leading Indicators Frequently Asked Questions," 2024, and STD/DOC(2007)1 on
  CLI revisions. Grade B.
- Federal Reserve Bank of St. Louis, "Data Revisions with FRED," *Page One Economics*, August
  2022, and the ALFRED documentation. Grade B.
- Eurostat, "Rapid estimates" glossary, Statistics Explained. Grade B.

**Industry and practitioner**

- Swiggy, "The OSM Distance Service Part 1: Evaluation Metrics and Routing Configurations,"
  *Swiggy Bytes*, January 2025. Grade C (company engineering case).
- Practitioners' heuristic that annual trade data is safely complete only at year minus two:
  reported in an open-source project's test suite, not an official statement. Grade E.

**Note on citations.** Figures attributed to studies are reported as the sources state them and
are not independently re-estimated. The supply chain application of several studies (for example,
the weather-variance percentages) is an interpretation of retail evidence, not a result
established for planning models. Where a number is uncertain or context-specific, the text says
so rather than quoting precision it cannot support.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_transaction-data.md`](03_transaction-data.md)
- Next topic: [`05_telemetry-iot-vision.md`](05_telemetry-iot-vision.md)
- Related: [`10_source-selection-trust.md`](10_source-selection-trust.md)
- Planning context: [`../01_supply_chain_foundations/04_planning-processes.md`](../01_supply_chain_foundations/04_planning-processes.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
