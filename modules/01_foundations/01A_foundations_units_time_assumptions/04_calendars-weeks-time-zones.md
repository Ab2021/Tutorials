# Calendars, Weeks, Time Zones, and Buckets

> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 04 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01

---

## 1. Why this topic exists

Topic 03 discretised quantity. This topic discretises time. Both are conversions from a
continuous thing into something a system can store, and both lose information that the
downstream model cannot recover. The difference is that quantity errors are usually visible —
somebody notices 4.8 cases — while time errors are invisible, because every timestamp is a
valid timestamp and every week number is an integer.

A supply chain runs on three clocks at once, and they do not agree.

- **Calendar time** is the Gregorian date: 2026-03-12. It is what the database column says and
  what the auditor asks for. It has no business meaning on its own.
- **Business time** is the period the company reports and plans in: fiscal week 11, fiscal
  period 3, ISO week 2026-W11, retail week 6 of the 4-5-4 year. It is a label attached to a set
  of dates by a rule that somebody chose.
- **Event time** is when the physical thing actually happened, expressed as an instant — a
  point on a globally unambiguous timeline. It is what the goods did.

A fourth time sits underneath all three and is usually missing from the data model: **record
time**, the moment the row was written. An event that happened at 23:50 on Friday may have been
recorded at 09:12 on Monday. Every report is a statement about the database at a moment, not
about the world.

The gaps between these four are where most time-series defects live, and they are defects of
representation rather than of modelling. Four examples set the shape of the problem.

**One.** A retailer with a 53-week fiscal year grows 1.50% on a like-for-like basis and reports
2.65% total growth. Both numbers are computed correctly. The 1.15-point gap is the 53rd week.
Nobody in the room knows which number the incentive plan uses.

**Two.** A German supplier ships on ISO weeks, Monday to Sunday. Its American customer plans on
the National Retail Federation (NRF) 4-5-4 calendar, Sunday to Saturday. Both call their buckets
"week 12". A join on the week number is wrong for every row, is right on the annual total by
coincidence, and cannot be detected by any reconciliation that only checks totals.

**Three.** A container is despatched from Hamburg at 05:50 on Sunday 28 December 2025. In ISO
terms it is in week 2025-W52. In the destination's local time it is 23:50 on Saturday 27
December, the last day of the retailer's fiscal year. In the origin's local time it is already
the first day of the next fiscal year. One physical event, three defensible period labels, and
a year-end number that depends on which one the reporter used.

**Four.** A monthly history is used to build a forecast. The weekly pattern — the Friday peak,
the Sunday trough, the Monday order spike — was destroyed at aggregation. No amount of modelling
recovers it, because the information is not in the data. The forecast is monthly-accurate and
weekly-useless, and it will be disaggregated to weekly by dividing by 4.33, which is a model in
disguise pretending to be arithmetic.

Each of these is a units problem. The unit is not a case or a kilogram; it is a week, a period,
or an instant. Topic 01 established that a quantity without a unit is not a number. The same
holds for a date without a calendar, a timestamp without an offset, and a bucket without a
boundary rule.

**Why this belongs in a foundations module.** Every later topic in this program consumes a time
series. Forecast error metrics in Topic 09 compare actual against predicted over a bucket.
Lead-time distributions in Topic 05 are measured in elapsed days between two events. Demand and
lead-time distributions in Topic 11 assume the observations are exchangeable, which they are
not if the bucket length changed mid-history. If the calendar is wrong, every one of those
measurements is wrong in a way that looks like model error. Analysts spend weeks tuning models
against a defect that a calendar join would have fixed in an afternoon.

**Where it leads.** Topic 05 takes the first of the two time quantities a planner actually needs
— the lead time — and shows that a nominal lead time, an actual lead time, and a promised lead
time are three different distributions. This topic supplies the clock those distributions are
measured against.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** calendar time, business time, event time, and record time for a single
   record, and **state** which one a given report is measuring.
2. **Compute** ISO 8601 week dates for any Gregorian date using the first-Thursday rule, and
   **identify** the three cases in which a January date belongs to the previous ISO year.
3. **Explain** why the NRF retail calendar is 4-5-4 rather than 4-4-5 or 5-4-4, and **state**
   what each variant does differently.
4. **Calculate** the effect of a 53-week fiscal year on total growth, weekly-average growth,
   and like-for-like growth, and **say** which of the three answers a given question requires.
5. **Reconcile** a weekly shipment series from an ISO calendar to a 4-5-4 calendar using a date
   spine, and **bound** the error introduced by a naive week-number join.
6. **Diagnose** a bucket-boundary defect from a timestamp, an offset, and a cut-off rule, and
   **specify** the correction.
7. **Explain** the ambiguous and nonexistent local times produced by daylight saving
   transitions, and **show** why storing local time without an offset is a data defect rather
   than a formatting choice.
8. **Choose** a planning bucket from the decision cadence, the data granularity, and the lead
   time, and **justify** why a finer bucket would be false precision.
9. **Quantify** what a monthly aggregation destroys relative to a weekly one, and **state** the
   condition under which a weekly forecast can be aggregated to monthly without loss.
10. **Write** a calendar specification for a dataset: bucket length, boundary rule, cut-off
    instant, time zone, and late-arrival policy, each with a test.

Outcomes 2 to 5 and 9 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold three pictures. The first is four clocks attached to one record. The second is a fiscal year
laid out as a grid of weeks rather than a row of months. The third is a filter: a bucket is a
low-pass filter applied to a signal, and the filter has a cutoff frequency whether or not
anybody designed it.

### 3.1 Four clocks on one record

```text
   ONE SHIPMENT, FOUR TIMES
   ------------------------------------------------------------------------------

   EVENT TIME          when the goods moved
   2025-12-28T04:50Z   a single instant, no ambiguity, no calendar
        |
        |  + offset (+01:00)  ->  2025-12-28 05:50  Hamburg local
        |  - offset (-05:00)  ->  2025-12-27 23:50  New York local
        v
   BUSINESS TIME       which period the company files it in
        |                    NRF FY2026 W01   (origin local date)
        |                    NRF FY2025 W52   (destination local date)
        |                    ISO 2025-W52     (ISO week date)
        v
   CALENDAR TIME       which Gregorian date the row carries
        |                    2025-12-28  or  2025-12-27
        v
   RECORD TIME         when the row was written
                        2025-12-29T14:12Z   (ASN processed next working day)
```

Read the diagram downward and each step adds a decision. The event instant is a fact. The
offset is a fact for that place at that moment. The business label is a *policy*. The record
time is an *operational* fact about a system, and it is the one that determines what a report
run at 06:00 on Saturday could possibly have known.

The single most useful habit this diagram installs: when somebody gives you a date, ask which
of the four it is.

### 3.2 A 4-5-4 year is a grid of weeks, not a row of months

```text
   NRF 4-5-4 FISCAL YEAR — 52 weeks, four 13-week quarters, 364 days
   (fiscal year normally begins on the Sunday nearest 31 January)

   Q1  |---- 13 weeks ----|   Q2  |---- 13 weeks ----|
       | 4  | 5  | 4      |       | 4  | 5  | 4      |
       |Feb |Mar |Apr     |       |May |Jun |Jul     |
       | 28 | 35 | 28     |       | 28 | 35 | 28     |
         ^         ^                ^         ^
         every fiscal month starts on the same weekday
         every fiscal quarter is exactly 13 weeks

   Q3  |---- 13 weeks ----|   Q4  |---- 13 weeks ----|
       | 4  | 5  | 4      |       | 4  | 5  | 4      |
       |Aug |Sep |Oct     |       |Nov |Dec |Jan     |
       | 28 | 35 | 28     |       | 28 | 35 | 28     |

   53rd week (roughly every 5-6 years): appended at the end of January
   when four or more days of January remain after the 52nd week.

   Contrast a calendar month: 28-31 days, so it holds 4 or 5 of each weekday.
   One extra Saturday in a month moves monthly sales by about 4.6%
   (a Saturday carrying 20% of weekly volume over a 4.35-week month).
```

That last line is the whole reason the retail calendar exists. A calendar month is not a fixed
number of weeks, so month-over-month comparisons mix a real change with a change in weekday
composition. A 4-5-4 month is a whole number of weeks, so every month contains exactly the same
weekday pattern and like days compare with like days. The cost is that the fiscal month is not
the calendar month, so payroll, rent, statutory filing, and the accounts payable calendar all
drift against it. Companies that adopt 4-5-4 keep a second, Gregorian calendar for everything
that is not merchandising.

Note the label. The NRF calendar is **4-5-4**. Variants exist — 4-4-5 and 5-4-4 are used by some
retailers, and trade writing frequently calls all three "4-4-5". The three differ only in where
the five-week month sits inside each 13-week quarter: 4-5-4 puts it second, 4-4-5 puts it third,
5-4-4 puts it first. All three total 13 weeks. A data dictionary that says "4-4-5" while the
source system implements 4-5-4 will be wrong by a week in two months out of three, and the
error will look like a demand anomaly.

### 3.3 A bucket is a filter

Aggregation is not a neutral sum. A bucket of length `L` averages the signal over `L` days and
therefore attenuates any cycle whose period is short relative to `L`.

| Bucket | What survives | What is destroyed |
|---|---|---|
| Daily | Day-of-week, holiday, promotion start, payday | Intra-day timing |
| Weekly (7 d) | Week-of-year seasonality, month-end effects | Day-of-week pattern |
| 4-week period | Month-of-year seasonality, weekday pattern preserved | Week-to-week variation |
| Calendar month (28–31 d) | Month-of-year seasonality | Week-to-week and day-of-week, plus a weekday-composition artefact |
| Quarter (13 wk) | Annual seasonality at coarse resolution | Everything shorter |

A 28-day bucket is exactly four weekly cycles, so it cancels the weekly pattern cleanly. A
30-day bucket does not: it cancels most of it and leaves a beat, which is why calendar-month
data has a residual weekday artefact that moves from month to month. This is arithmetic, not
opinion, and it is the reason "monthly" is a different measurement instrument from "4-week
period", not merely a coarser one.

The practical consequence: **you cannot compare a variance computed on one bucket with a
variance computed on another.** A coefficient of variation on weekly data and a coefficient of
variation on monthly data are measurements of different signals. Topic 09 will make the same
point about error metrics.

## 4. Definitions and vocabulary

The fourth column names the unit or coding system, because a definition without one is the defect
this module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Calendar time | The Gregorian date and clock reading, with no business meaning attached | Assumed to be the only kind of time | ISO 8601 calendar date, `YYYY-MM-DD` |
| Business time | The period label a company reports and plans in | Assumed to coincide with calendar time | Fiscal or retail calendar label |
| Event time | The instant at which a physical event occurred | Confused with the date on the document | UTC instant, ISO 8601 with `Z` |
| Record time | The instant at which the database row was written or last changed | Assumed to equal event time | UTC instant, plus a load identifier |
| Valid time | The interval during which a fact is true in the world | Omitted from the model | Half-open interval `[start, end)` |
| System time | The interval during which a fact is current in the database | Omitted, so history cannot be reconstructed | Half-open interval `[start, end)` |
| Bitemporal | A model carrying both valid time and system time | Used to mean "has a timestamp" | Two interval pairs |
| Time bucket | A contiguous interval to which records are assigned for aggregation | Called a period, a bin, or a cycle interchangeably | Interval of fixed or variable length |
| Bucket boundary | The rule that decides which bucket an instant on the edge belongs to | Left implicit | Half-open interval convention |
| Cut-off | The instant after which a record is not included in a report run | Confused with the period end | UTC instant, plus the reporting zone |
| As-of date | The system time at which a report's contents are defined | Omitted, so two runs of the same report disagree | UTC instant |
| Fiscal year (FY) | The accounting year a company reports in | Assumed to start in January | Named year plus start and end dates |
| Fiscal period | A division of the fiscal year, usually a month or four weeks | Assumed to equal a calendar month | Period number plus start and end dates |
| 4-5-4 calendar | A retail fiscal calendar with months of 4, 5, and 4 weeks in each 13-week quarter | Called 4-4-5 generically | Week pattern per quarter |
| 4-4-5 calendar | A variant with months of 4, 4, and 5 weeks in each quarter | Used as the name for the NRF standard, which is 4-5-4 | Week pattern per quarter |
| 5-4-4 calendar | A variant with months of 5, 4, and 4 weeks in each quarter | Treated as identical to 4-5-4 | Week pattern per quarter |
| 52/53-week year | A fiscal year of 364 or 371 days, always ending on the same weekday | Reported without saying which | Weeks per year |
| 53rd week | The extra week inserted to keep the fiscal calendar aligned to the seasons | Counted as growth | One week, 7 days |
| Restatement | Re-labelling the following year's weeks so that a 53-week year compares with a 52-week year | Assumed to be lossless | Mapping table between years |
| Like-for-like (LFL) | A comparison over periods of equal length with equal calendar composition | Confused with total growth | Percentage over a defined base |
| Comparable sales | Retail sales growth over stores open in both periods, usually on an equal-week basis | Reported without the exclusion list | Proportion, with exclusions stated |
| ISO 8601 | The international standard for date and time representation | Treated as a calendar rather than a notation standard | Standard identifier |
| ISO week date | A date expressed as week-year, week number, and weekday | Confused with a Gregorian date | `YYYY-Www-D` |
| ISO week-year | The year that owns the week, which may differ from the Gregorian year | Assumed to equal the Gregorian year | Four-digit year |
| Week 01 | The ISO week containing the year's first Thursday | Assumed to be the week containing 1 January | Week number |
| Ordinal date | A date expressed as year plus day-of-year | Rarely used, occasionally essential | `YYYY-DDD` |
| Time zone | A region's legally defined offset history, including transition rules | Treated as a fixed offset | IANA `Area/Location` identifier |
| UTC | Coordinated Universal Time, the reference timeline | Called GMT, which is a time zone with a different legal status | Instant, suffixed `Z` |
| Offset | The difference between local time and UTC at a stated instant | Treated as a property of the zone rather than of the instant | `±HH:MM` |
| IANA tz database | The maintained registry of time zone identifiers and their transition rules | Assumed to be static | Release identifier such as `2026b` |
| Daylight saving time (DST) | A seasonal offset change mandated by a jurisdiction | Assumed to be universal and simultaneous | Offset change at a stated instant |
| Spring-forward gap | A local time range that does not exist because clocks moved forward | Written into data as if it existed | Nonexistent local interval |
| Fall-back fold | A local time range that occurs twice because clocks moved back | Silently resolved to the first occurrence | Ambiguous local interval |
| Epoch timestamp | A count of seconds or milliseconds since a fixed reference instant | Confused with a formatted date string | Integer, usually seconds since 1970-01-01T00:00Z |
| RFC 3339 | The internet timestamp format: date, time, and mandatory offset | Called "ISO format" without the offset requirement | `YYYY-MM-DDThh:mm:ss±hh:mm` |
| Naive datetime | A date and time with no offset and no zone | Stored as if it were an instant | Local wall-clock reading |
| Granularity | The finest distinction a dataset can represent | Confused with precision | Stated unit, e.g. one day |
| Aggregation level | The bucket at which a series is reported | Changed mid-history without a note | Bucket length in days |
| Temporal hierarchy | The set of nested aggregation levels a series can be viewed at | Assumed to be consistent across systems | Level list from day to year |
| Planning bucket | The interval a plan is expressed in | Chosen to match the report rather than the decision | Days or weeks |
| Frozen horizon | The portion of the plan that cannot be changed | Ignored when choosing bucket length | Days |
| Week-ending date | The last day of a week, used as a bucket key in retail | Joined across calendars without checking the weekday | Date |
| Trading day | A day on which the business actually operates and books volume | Counted as a calendar day | Count per period |
| Trading-day adjustment | A correction for the number of trading days in a period | Applied without stating the day-of-week weights | Multiplicative factor |
| Holiday shift | The movement of a public holiday between periods across years | Read as a demand change | Date plus jurisdiction |
| Leap day | 29 February, present in leap years | Ignored in daily averages | One day per four years, with exceptions |
| Year-over-year (YoY) | Comparison of a period with the same period one year earlier | Computed across calendars of different length | Percentage |
| Trailing twelve months (TTM) | A rolling 12-month window ending at the current period | Compared across 53-week years without adjustment | Currency or units |
| Seasonality index | A multiplicative or additive factor by position in the year | Estimated on data with inconsistent bucket lengths | Dimensionless |
| Date spine | A table with one row per date and every calendar label as a column | Replaced by joining on period numbers | One row per date |
| Late-arriving data | Records whose event time precedes the as-of date but whose record time follows it | Ignored, so history silently changes | Count and value by period |

Six rows carry most of the damage.

**Event time versus record time.** These are the two that get collapsed into a single
`created_at` column. When they are collapsed, a weekly report cannot be reproduced: re-running
it a month later gives a different number, because records that arrived late have been
backfilled. The report was not wrong when it ran. It was a correct statement about a different
database state. Without a system-time column you cannot tell the two apart, and you cannot
answer the question "what did we know on the day we decided?"

**4-5-4 versus 4-4-5.** These are not synonyms and the confusion is endemic. The NRF calendar is
4-5-4. A vendor deck that says "4-4-5 retail calendar" may mean the NRF calendar loosely, or may
mean a genuinely different convention. The only way to know is to check which fiscal month
contains five weeks in the source system's own calendar table.

**52-week versus 53-week.** A 53-week year is 371 days against 364, which is 1.923% longer. Any
total compared across the boundary without adjustment carries that as a bias. The bias has a
sign: it is always upward in the 53-week year.

**Offset versus time zone.** An offset is a property of an instant at a place: `−05:00` on a
January date in New York. A time zone is the rule set that produces offsets over time, including
future legislative changes. Storing `−05:00` tells you what the clock said. Storing
`America/New_York` tells you what the clock will say next year, which is what you need for a
future delivery appointment.

**Naive datetime versus instant.** A naive datetime is a wall-clock reading with nothing to
anchor it. Two naive datetimes from two systems, both reading `2025-12-28 05:50`, may refer to
instants six hours apart. Sorting a column of naive datetimes produces a sequence that is wrong
across every transition.

**Granularity versus precision.** A daily series built by dividing a monthly forecast by the
number of days has daily granularity and monthly precision. It will produce a confidence
interval that is far too narrow and a plan that appears responsive and is not.

## 5. The mechanics, step by step

This section turns one record into one correctly labelled row, then turns a set of rows into a
series that can be compared with another series from another calendar. It is twelve steps in four
phases. Each step names the rule, what the data should carry, and the failure that occurs when
the step is skipped.

### Phase A — Fix the instant

**Step 1. Record the event instant in UTC, with full precision.**

- Rule: every event gets a timestamp that is unambiguous on a global timeline, stored at a
  precision finer than the finest bucket you will ever report. One second is enough; one day is
  not.
- Data: `event_ts_utc`, an ISO 8601 instant with `Z`, plus the source system identifier.
- If skipped: the instant is inferred from a date and a location later, and the inference is
  wrong exactly at boundaries, which is where the interesting cases are.

**Step 2. Record the offset and the zone that were in force at that instant.**

- Rule: store the offset as data (`+01:00`) and, where future scheduling matters, the IANA zone
  identifier (`Europe/Berlin`). The offset answers "what did the clock say". The zone answers
  "what will the clock say".
- Data: `event_offset`, `event_tz`.
- If skipped: the local reading cannot be reconstructed, and any recalculation after a
  legislative change silently rewrites history.

**Step 3. Record the record time.**

- Rule: `recorded_at_utc` is set by the writing system, not by the business user, and is never
  updated. A correction produces a new row with a new record time, not an overwrite.
- Data: `recorded_at_utc`, plus `load_id` and `source_system`.
- If skipped: the dataset can only answer "what is true now", never "what was known then", and
  no report can be reproduced.

**Step 4. Distinguish the business events that share one physical movement.**

- Rule: despatch, departure, arrival at port, customs clearance, arrival at destination,
  receipt, and put-away are seven events, not one. Each has its own instant and each may fall in
  a different bucket.
- Data: an event table with an event type, not a wide row with seven date columns.
- If skipped: the company chooses one date — usually the despatch date — and calls it "the
  shipment date", which means the series measures when goods left, not when they arrived.

### Phase B — Choose the bucket

**Step 5. Choose the bucket length from the decision, not from the report.**

- Rule: the bucket should match the cadence at which somebody can act. A weekly replenishment
  decision needs weekly data. A monthly financial close needs months. A daily dashboard fed by a
  weekly plan is decoration.
- Data: a documented bucket length in days, with the decision it serves named.
- If skipped: the bucket is inherited from whichever report the analyst was shown first, and the
  decision it serves is never identified.

**Step 6. Fix the boundary rule and write it down.**

- Rule: state four things — the week start day, the fiscal year start rule, the cut-off clock
  time, and the cut-off time zone. Then state the interval convention: half-open `[start, end)`
  so that no instant is counted twice.
- Data: a calendar table with `period_start`, `period_end_exclusive`, and the rule stated in its
  documentation.
- If skipped: two teams produce two different weekly numbers from the same warehouse and both
  are defensible, which is worse than one being wrong.

**Step 7. Choose the cut-off instant, not the cut-off date.**

- Rule: a cut-off is an instant — Friday 23:59:59 in the reporting zone, converted to UTC. A
  cut-off expressed as a date has no time and therefore no meaning for anything that happens
  during the day.
- Data: `cutoff_utc`, plus the reporting zone it was derived from.
- If skipped: the cut-off silently defaults to midnight server time, which is a third zone
  nobody chose.

**Step 8. Decide the late-arrival policy and the restatement policy.**

- Rule: choose one of three. Freeze the period at cut-off and never restate. Restate for a
  stated number of days and publish the revision. Restate indefinitely with as-of reporting.
  Each is legitimate. Having no policy is not.
- Data: a revision table with `period_id`, `as_of_utc`, `value`, and `revision_reason`.
- If skipped: the same report run on two days gives two answers, and the difference is
  attributed to the business rather than to the pipeline.

### Phase C — Map between calendars

**Step 9. Build a date spine before you build any series.**

- Rule: one row per date, with every label the organisation uses as a column — Gregorian date,
  ISO week-year, ISO week, ISO weekday, fiscal year, fiscal period, fiscal week, week start,
  week end, trading-day flag, holiday flag. One row per date is the only structure that makes
  the mapping many-to-many in the right way.
- Data: a calendar dimension table with a documented generation rule and a version.
- If skipped: each system carries its own label logic, the definitions drift, and the drift is
  invisible until a boundary year.

**Step 10. Label at the finest grain and aggregate at the coarsest.**

- Rule: join facts to the spine on date, then aggregate. Never join an aggregate to an
  aggregate on a period label.
- Data: fact rows at their native grain, with a date, joined to the spine.
- If skipped: the join is on a label that two calendars define differently, and the error is
  proportional to the volume on the boundary day.

**Step 11. Reconcile the totals explicitly.**

- Rule: after mapping, check that the sum over the new calendar equals the sum over the old one,
  and that the difference is explained by the partial weeks at the boundaries. A reconciliation
  that only checks the annual total will pass a series that is wrong in every week.
- Data: a reconciliation table with `calendar_a_total`, `calendar_b_total`, `difference`,
  `explained_by_boundary_weeks`.
- If skipped: the annual total agrees and the weekly series is unusable, which is the worst
  combination because it defeats both the sanity check and the use.

### Phase D — Compare

**Step 12. Normalise before comparing, and state the normalisation.**

- Rule: compare periods of equal length and equal weekday composition. For a 53-week year,
  choose one of three normalisations and say which: exclude the 53rd week, use the weekly
  average, or restate the following year. They give different answers and all three are
  defensible for different questions.
- Data: the comparison table carries the normalisation as a column, not as a footnote.
- If skipped: growth is reported at 2.65% when the business grew 1.50%, and the incentive plan
  pays on the difference.

The twelve steps are ordered so that each one is cheap and each one makes the next one possible.
The expensive failures come from doing step 10 before step 9, which is the default in every
organisation that has more than one reporting system.

## 6. Worked example

Halden Home is a fictional American homewares retailer. It reports on the NRF 4-5-4 calendar,
with the fiscal year beginning on the Sunday nearest 31 January. Its largest supplier, Brückner
Werke, is a German manufacturer that despatches on ISO weeks and reports despatch volume by ISO
week number. All figures are constructed to be internally consistent.

### 6.1 The 53-week comparison

Halden's fiscal 2025 had 53 weeks. Fiscal 2024 had 52. The reported totals were 4,804,200 units
and 4,680,000 units. The 53rd week of fiscal 2025 — the light week at the end of January —
contained 54,000 units.

Three computations follow, all from the same two numbers.

| Measure | FY2025 | FY2024 | Growth |
|---|---:|---:|---:|
| Total units as reported | 4,804,200 | 4,680,000 | +2.65% |
| Units per week over all weeks | 90,645 | 90,000 | +0.72% |
| Units over 52 comparable weeks | 4,750,200 | 4,680,000 | +1.50% |

The arithmetic, shown so it can be checked:

```text
   naive total growth    = 4,804,200 / 4,680,000 - 1 = 0.026538 = +2.65%
   weekly average FY2025 = 4,804,200 / 53        = 90,645.3
   weekly average FY2024 = 4,680,000 / 52        = 90,000.0
   weekly average growth = 90,645.3 / 90,000 - 1 = 0.00717  = +0.72%
   52-week FY2025        = 4,804,200 - 54,000    = 4,750,200
   like-for-like growth  = 4,750,200 / 4,680,000 = 1.01500  = +1.50%
```

The true underlying growth is 1.50%. The naive total overstates it by 1.15 percentage points.
The weekly average understates it by 0.78 points. The two wrong answers bracket the right one,
which is a useful property: if your total and your weekly average disagree, and the like-for-like
number does not sit between them, you have an arithmetic error rather than a calendar effect.

Why the weekly average is biased downward here: the extra week is not an average week. It falls
in late January, after the Christmas peak, and carries 54,000 units against a 90,645 average. So
dividing by 53 dilutes the mean. Had the 53rd week been a peak week, the weekly average would
have overstated growth instead. **A weekly average is only like-for-like when the extra week is
representative, which it almost never is.**

The general identity is worth memorising:

```text
   total growth  ~=  like-for-like growth x (weeks this year / weeks last year) x (mix effect)
```

The weeks ratio here is 53/52 = 1.01923, which is the 1.923% that the naive comparison carries
before any real growth. Call it 1.9%, as the literature does, but use 1.01923 when you multiply.

### 6.2 One despatch, three period labels

Brückner despatches a container from Hamburg at 05:50 local time on Sunday 28 December 2025.
Halden's fiscal 2025 ends on Saturday 27 December 2025.

| Frame | Instant or reading | Period label |
|---|---|---|
| Event time | 2025-12-28T04:50Z | none — an instant has no period |
| Origin local | 2025-12-28 05:50 `+01:00` | NRF FY2026, week 1 (day 1) |
| Destination local | 2025-12-27 23:50 `−05:00` | NRF FY2025, week 52 (last day) |
| ISO week date | Monday 22 – Sunday 28 December | ISO 2025-W52, Sunday |
| ASN record time | 2025-12-29T14:12Z | whatever the Monday run assigns |

Four defensible answers to "which week was this shipment in". The supplier's report puts it in
week 1 of the new year. The retailer's warehouse report puts it in week 52 of the old year. The
ISO series puts it in the last week of 2025. The ASN, processed on the Monday, puts it wherever
Monday's run puts it.

The financial consequence is not hypothetical. A single container at 1,850 cases against a
weekly volume of 41,000 cases is 4.5% of that week. If it is the last container of the fiscal
year, it moves the year's total by 1,850 units and the year's growth rate by 0.04 percentage
points — and it moves the *following* year's opening week by the same amount in the opposite
direction, so a bad boundary decision is a permanent level shift in two series.

### 6.3 The ISO-to-4-5-4 mismatch

Now take the annual reconciliation. Brückner despatches 20,000 cases in each ISO week of 2026.
Because 1 January 2026 is a Thursday, week 01 of 2026 begins on Monday 29 December 2025, and ISO
2026 runs to week 53, ending on Sunday 3 January 2027. The derivation:

```text
   2026-01-01 is a Thursday  -> it lies in week 01
   week 01 begins Monday 2025-12-29
   53 weeks x 7 days = 371 days
   2025-12-29 + 370 days = 2027-01-03, a Sunday
   2027-01-01 is a Friday, so week 01 of 2027 begins Monday 2027-01-04
   therefore 2027-01-03 belongs to 2026-W53
```

So the ISO year has 53 weeks and the NRF fiscal year has 52. Three specific failures follow.

**Failure A — the missing week 53.** A join of ISO week number to NRF week number finds no
partner for week 53. An inner join drops it: the annual total becomes 1,040,000 cases against a
true 1,060,000, an understatement of 20,000 cases, or 1.89%. A left join with a clamp to week 52
puts 40,000 cases in that bucket instead of 20,000, a 100% spike in one week. Either way the
series is wrong and the annual control total may still look acceptable, because the analyst
reconciles to the ISO total and the join was on the ISO side.

**Failure B — four January weeks in the wrong fiscal year.** Halden's fiscal 2026 begins on
Sunday 1 February 2026, the Sunday nearest 31 January. ISO weeks 01 to 04 of 2026 lie between 29
December 2025 and 25 January 2026, all of which falls inside fiscal 2025. An analyst who treats
the ISO year as the fiscal year moves roughly four weeks of volume — about 80,000 cases, or 7.7%
of a 52-week year — from fiscal 2025 into fiscal 2026. The error is not visible at the annual
total of the ISO series. It is visible only when the two fiscal years are compared.

**Failure C — the one-day offset at every boundary.** ISO weeks run Monday to Sunday; NRF weeks
run Sunday to Saturday. Every ISO week contains six days of one NRF week and one day of the next.
A join on week number therefore assigns six-sevenths of each week correctly and one-seventh to a
bucket that is one day out. If Sunday volume equals the daily average the errors cancel. If
Sunday is a non-working day, they do not, and the boundary week is misstated by up to one day's
volume.

| Join strategy | What it does | Annual total | Weekly series |
|---|---|---|---|
| On ISO week-year and week number | Matches labels that mean different date ranges | Drops week 53: −1.89% | Wrong at every boundary; spikes or gaps at year end |
| On week-ending date | Matches ISO Sunday against NRF Saturday | Matches nothing | Empty result, or a silent zero-row fill |
| On date spine | Expands each week to seven dates, then relabels | Reconciles | Correct, subject to a daily allocation assumption |

### 6.4 The correct reconciliation and its residual

The procedure is the one in section 5: expand, label, aggregate.

```text
   fiscal 2026 ends Sunday 2027-01-31, a Sunday
   the ISO week containing that date is 2027-W04 = Mon 2027-01-25 ... Sun 2027-01-31
   daily volume under a uniform assumption = 20,000 / 7 = 2,857.14 cases
   days falling in fiscal 2026 (25-30 Jan) = 6 x 2,857.14 = 17,142.9 cases
   days falling in fiscal 2027 (31 Jan)    = 1 x 2,857.14 =  2,857.1 cases
   check: 17,142.9 + 2,857.1 = 20,000.0 cases
```

The same arithmetic applied to 2026-W53 shows that the 53rd week falls wholly inside fiscal
2026 — it straddles the calendar year, not the fiscal one — which is why a fiscal reconciliation
and a calendar-year reconciliation need different boundary weeks.

The result is a 4-5-4 series with no spike and a total that reconciles to the fiscal year. The
residual error is the daily allocation assumption. If Brückner does not despatch on Sundays,
the true daily volume on the six working days is 20,000/6 = 3,333.3 cases and Sunday is zero. A
boundary that cuts on a Sunday then receives 0 rather than 2,857 cases, an error of 2,857 cases
— 14.3% of a week — at each of the year's two boundaries.

Bounding it honestly: each fiscal boundary cuts one ISO week, and the misallocated share is at
most six-sevenths of that week. With two boundaries the worst case is twelve-sevenths of a week,
which is 1.71 weeks, or 3.3% of a 52-week year. In practice the two errors usually fall in
opposite directions and partly cancel in the annual total, which is exactly why annual-total
reconciliation is not a sufficient test.

**The rule this example teaches.** You cannot reconcile a weekly series to a different weekly
calendar without a daily allocation assumption. The best available outcome is an explicit
assumption, a stated bound, and a reconciliation that checks the boundary weeks individually
rather than the year in total.

## 7. Code, query, or calculation

One block, because the arithmetic in section 6 is the substance and the query is only the
mechanical form of step 9. It builds the spine and shows the mapping that a join on week number
would have hidden.

```sql
-- One row per date; every calendar label attached before any aggregation.
with spine as (
    select d::date as cal_date
    from generate_series(date '2025-12-01', date '2027-02-28', interval '1 day') as d
),
labeled as (
    select
        s.cal_date,
        to_char(s.cal_date, 'IYYY-"W"IW') as iso_week,   -- ISO week-year and week
        extract(isodow from s.cal_date)   as iso_dow,    -- 1 = Monday, 7 = Sunday
        n.fiscal_year,
        n.fiscal_week,
        n.week_start,
        n.week_end_exclusive
    from spine s
    join ref.nrf_calendar n
      on s.cal_date >= n.week_start
     and s.cal_date <  n.week_end_exclusive                -- half-open interval
)
select iso_week, fiscal_year, fiscal_week, count(*) as days
from labeled
group by iso_week, fiscal_year, fiscal_week
order by min(cal_date);
```

Three things to check in the output. An ISO week that returns `days` summing to more than 7 has
been double-counted, which means the calendar table has overlapping weeks. An ISO week that
appears against two fiscal years is a boundary week and must be split, not assigned. An ISO week
with no row is a week the fiscal calendar does not cover, which is the week-53 failure in its
silent form.

The last line matters more than the query. `to_char(..., 'IYYY-"W"IW')` returns the ISO week-year,
not the Gregorian year. Using `'YYYY-"W"WW'` instead returns a Gregorian year with a non-ISO week
number, and the two disagree for a few days every January. That single substitution is the most
common ISO bug in production SQL, and it produces a date that is a valid string and a wrong fact.

## 8. Data, standards, and regulatory context

Every calendar decision sits between a standard that fixes the representation and a regulation
that fixes the reporting period. Neither fixes the bucket you should plan in. Knowing which is
which prevents the two common errors: treating a convention as a law, and treating a law as a
formatting detail.

### 8.1 Representation standards

| Standard | Governs | What it fixes | Grade |
|---|---|---|---|
| ISO 8601-1:2019 | Date and time representation | Calendar dates, times of day, UTC offsets, and week dates. Monday is the first day of the week; week 01 is the week containing the first Thursday | B |
| ISO 8601-2:2019 | Extensions to ISO 8601 | Reduced precision, uncertain and approximate dates, intervals and durations | B |
| ISO/R 2015:1971 | Week numbering, historical | The original four-day rule that ISO 8601 later absorbed | B |
| IANA time zone database | Civil time offsets and transitions | Zone identifiers, historical offsets, and future transition rules, released periodically | B |
| RFC 3339 | Internet timestamps | Requires an explicit offset on every timestamp; declares unqualified local time unacceptable | B |
| W3C Working with Time and Timezones | Implementation guidance | Recommends UTC for storage and serialisation; restricts floating time to genuinely local concepts | B |

Three consequences worth spelling out.

**ISO 8601 is a notation standard, not a calendar system.** It tells you how to write
`2026-W01-1` and what that string means. It does not tell you which weeks your company reports
in, when its fiscal year starts, or what its cut-off is. A company can be fully ISO 8601
compliant and still have a fiscal calendar that no other company shares.

**The week-01 rule has a precise consequence.** Because week 01 is the week containing the first
Thursday, the ISO year begins on the Monday between 29 December and 4 January, and ends on the
Sunday between 28 December and 3 January. An ISO year is therefore 364 or 371 days. A date in
early January can belong to week 52 or 53 of the previous ISO year. Specifically, if 1 January
falls on a Friday, Saturday, or Sunday, it belongs to the previous ISO year. This is not an edge
case to be handled later; it happens in three years out of seven.

**RFC 3339 makes the offset mandatory for a reason.** The standard states that unqualified local
time is unacceptable for the internet because it cannot be interpreted unambiguously in most of
the world. A timestamp column without a zone is not a lenient version of a timestamp. It is a
different, weaker type that cannot answer the question it appears to answer.

### 8.2 Fiscal and tax conventions

| Instrument | Jurisdiction | What it establishes | Grade |
|---|---|---|---|
| NRF 4-5-4 calendar | Retail, international by adoption | A 52/53-week merchandising calendar with 4-5-4 months and 13-week quarters | C |
| 26 CFR §1.441-2 | United States | Election of a tax year of 52-53 weeks, ending on a fixed weekday | B |
| IRS Publication 538 | United States | The 52-53-week year: same weekday each year, last or nearest occurrence to month end | B |
| IAS 34 | IFRS reporters | Interim reporting, including comparability of interim periods | B |

The tax rules are worth reading because they show how much freedom a company has and where the
constraints are. Under the United States rules, a 52-53-week year must end on the same weekday
every year, on either the last occurrence of that weekday in a chosen month or the occurrence
nearest the month end. The year may end up to six days before month end, or within about three
days of it. Short periods of 359 days or more are treated as full years, and periods of six days
or fewer are attached to the following year. Those are the degrees of freedom a retail calendar
is built inside.

What the rules do **not** do is tell you which variant to use. 4-5-4, 4-4-5, and 5-4-4 all fit
within a 52-53-week election. The choice is a merchandising decision, made because the buying
and planning cycle is weekly and the comparison of like weeks matters more than alignment with
the calendar month.

> **Note:** The NRF calendar is 4-5-4. It is common to see it described as "4-4-5" in vendor
> documentation and trade articles, and some retailers genuinely operate a 4-4-5 or 5-4-4
> variant. The three are not interchangeable: they place the five-week month at different
> positions within each quarter, so the same fiscal week number covers different dates. If a
> data dictionary and a source system disagree on the variant, the source system's own calendar
> table is the authority.

### 8.3 Time zone data as governed reference data

The IANA time zone database is not a static lookup. It is a maintained dataset with periodic
releases, because jurisdictions change their offset rules, their DST dates, and occasionally
their entire zone. A pipeline that pins a tz database version and never updates will compute
correct historical offsets and incorrect future appointments.

Two implications for a supply chain dataset:

- **Historical events need the offset that was in force, not the offset in force now.** If the
  offset is recomputed at query time from a current tz release, a legislative change can rewrite
  the bucket assignment of historical shipments. Storing the offset at write time makes history
  immutable.
- **Future appointments need the zone identifier, not the offset.** A delivery slot booked for
  next March must be stored as a local time plus a zone, so that the correct offset is applied
  when the date arrives and the rules for that year are known.

The database itself does not resolve ambiguity. The IANA documentation is explicit that local
times which do not exist, and local times which occur twice, are properties of every zone with
seasonal offset changes, and that resolving them is a decision for the consumer of the data —
the application, the database, or the analyst. Two conventions are common and they give
different answers:

| Convention | Spring-forward gap | Fall-back fold | Consequence |
|---|---|---|---|
| Earlier offset | Shifts forward by the gap | First occurrence | Ambiguous records sort before their duplicates' successors |
| Later offset | Shifts backward by the gap | Second occurrence | Duration calculations across the transition are off by the offset change |

Neither is wrong. Choosing neither, and letting the library default, means the choice is made
by whoever wrote the parsing code, usually without knowing they made it.

## 9. Industry practice and case evidence

### 9.1 The retail calendar in practice

The 4-5-4 calendar was developed in the 1930s and widely adopted by American retailers in the
1940s. Its stated purpose is comparability: with every fiscal month containing a whole number of
weeks and every fiscal month beginning on the same weekday, like days compare to like days. The
NRF publishes the calendar for a rolling set of years, together with the sales release dates
that member companies use.

The 53rd week is inserted when, after laying out 52 weeks, four or more days of January remain.
NRF's published list of recent 53-week years is 1995, 2000, 2006, 2012, 2017, and 2023 — roughly
every five to six years, with the interval set by the interaction between the 364-day year and
the leap-year cycle.

There are two documented ways to handle the comparison, and they are not equivalent.

- **Exclude the 53rd week.** Compare the first 52 weeks of the long year with the 52 weeks of
  the short year. Simple, transparent, and it drops a real week of trading.
- **Restate the following year.** Shift the following year's weeks by one so that holiday weeks
  align, which costs one week at the start of that year. This is the approach NRF describes, and
  it has the advantage of keeping every reported week as a real trading week.

Either is defensible. What is not defensible is doing neither, which is the default in most
companies that report on a 4-5-4 calendar and have never written the rule down.

Comparable sales figures published by retailers usually exclude stores not open in both periods
and are usually computed on an equal-week basis, but the specific convention is disclosed in the
filing rather than standardised. Two retailers' "comparable sales growth" figures are therefore
not guaranteed to be the same measurement. This is a Grade C observation about practice, not a
regulatory requirement.

### 9.2 What enterprise systems actually implement

Commercial planning and retail systems implement calendar logic in incompatible ways, and the
differences are visible in their documentation.

- Retail-oriented database functions expose the retail calendar directly, including a parameter
  that selects restated or non-restated weeks for a 53-week year. The existence of that parameter
  is the clearest evidence that the restatement decision is real and has to be made explicitly.
- Cloud forecasting services expose "time alignment boundaries" — the rule that decides which
  bucket a record belongs to — with defaults such as weeks beginning on Monday and months
  beginning on day 1. The defaults are the ones that quietly misalign with a retail calendar.
- Enterprise resource planning systems support several fiscal year variants, including
  calendar-year, 4-4-5, 4-5-4, and 5-4-4, and the variant is a configuration setting rather than a
  data attribute. When the setting changes, historical data retains the old labels.

The pattern across all three: the calendar is configuration, configuration is invisible in the
data, and the data therefore cannot tell you which calendar it is in. That is why section 5's
step 9 puts the labels in a table rather than in a system setting.

### 9.3 Evidence on bucket size

The evidence on how bucket size affects forecast accuracy is thinner and more conditional than
practitioner writing suggests.

An academic working paper on aggregation and decision frequency, using simulation and three
real datasets, reports that modelling at a finer granularity and forecasting up to the decision
frequency generally performs better than modelling aggregated data — provided the
higher-frequency model is correctly specified. The condition matters: an incorrectly specified
fine-grained model can lose to a well-specified coarse one.

A vendor-published test on a single client demand dataset reports the opposite direction, with
quarterly buckets beating monthly by roughly 2.5 mean absolute percentage error (MAPE) points
and weekly by roughly 4 points, largely through better model selection at the coarser level.
This is one dataset, published by a consultancy, with no independent replication. It should be
read as an existence proof that coarse buckets can win, not as a base rate.

> **Note:** These two results are not in conflict so much as measuring different things. One
> asks whether information is lost by aggregating before modelling; the other asks which bucket
> a model selection procedure happens to do well on. For a planning decision, the first question
> governs. For a reporting comparison, the second may.

What the evidence does not support is a general rule that finer is better. It supports a narrower
claim: aggregation destroys information, and whether that loss costs you depends on whether the
destroyed pattern was actionable.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| 53-week year compared naively | Growth jumps about 1.9% in a year with no operational change | Total compared without normalising for the extra week | Overstated growth, bonuses paid on a calendar artefact | Publish the week count with every total; state the normalisation |
| Week-number join across calendars | Annual total reconciles, weekly series has a spike or a gap at year end | Two calendars number weeks differently and one has a week 53 | Series unusable; model learns a spurious year-end spike | Join on date spine; never on week number |
| Local time stored without offset | Two systems' timestamps differ by a constant that nobody can name | Naive datetime has no anchor | Mis-sorted events, wrong buckets, unreproducible reports | Store UTC instant plus offset plus zone |
| DST gap written into data | Timestamps that never existed, such as 02:30 on a spring-forward date | Naive construction or arithmetic on local times | Records silently shifted an hour or dropped | Validate against tz rules; reject nonexistent local times |
| Cut-off time zone undocumented | Two weekly reports disagree on a Friday shipment | Cut-off expressed as a date or in server-local time | Boundary disputes that cannot be adjudicated | State cut-off instant and zone in the metric definition |
| Month bucket used for weekly decisions | Weekly plan is flat within each month | Aggregation removed the weekly pattern | Replenishment timing wrong even when totals are right | Match bucket to decision cadence |
| Partial period included in a total | First and last buckets of an extract are about half normal | Extract window cut mid-bucket | Trend endpoints exaggerated | Exclude or flag incomplete buckets |
| Fiscal year label applied to ISO year | Four January weeks appear in the wrong fiscal year | Two different year definitions share a name | Fiscal comparisons wrong by about 7.7% of annual volume | Carry both year labels explicitly |
| `YYYY-WW` used instead of ISO week formatting | A handful of early-January dates carry the wrong year | Gregorian year combined with an ISO-derived week number | Valid-looking strings that are wrong facts | Use ISO week-year formatting; test the first week of January |
| Late-arriving records silently backfilled | Re-running last month's report gives a different number | Event time used as if it were record time | History changes; audit trail lost | As-of reporting; never overwrite |
| Leap day treated as an ordinary day | 29 February shows a demand spike or a productivity dip | Daily averages computed over 365 days | Annual comparisons biased in leap years | Count days per period explicitly |
| Daily plan built by dividing a monthly forecast | Daily values are constant within the month | False granularity: monthly information at daily resolution | Narrow confidence intervals, poor execution | Disaggregate only with an explicit day-of-week profile |
| Server time zone used as the business zone | Warehouse in one zone, planning in another, no documented rule | Default inherited from infrastructure | Boundary records assigned by accident | One named reporting zone per metric |

Two of these deserve expansion.

**The week-number join is the highest-consequence failure on the list**, because it passes the
check that most organisations actually run. An annual reconciliation compares totals, and a
total is exactly what a boundary error preserves. The error lives in the distribution across
buckets, which nobody reconciles. The test that catches it is to plot the series and look for a
spike or a gap at the fiscal year end, and to check that every ISO week appears exactly once.

**Leap day is the smallest and most persistent.** A daily average computed over a leap year is
divided by 366 rather than 365, which biases every daily figure downward by 0.27%. That is small
enough to be invisible and large enough to matter when compounded into an annual cost, a
capacity plan, or a rate card. The fix is trivial — count the days in the period rather than
assuming 365 — and the reason it persists is that the assumption is never written down.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The reporting calendar is stable across the analysis window | Trends and seasonality need comparable buckets | A change of fiscal variant, a 53-week year, a calendar migration | Compare period lengths by year; flag any year whose week count differs |
| Event time and record time are distinguishable | Reproducibility and as-of reporting | A single `created_at` column, or overwrites in place | Re-run a historical report and diff the result |
| All source systems agree on the fiscal calendar | A join on period label is only valid if the label means the same thing | Multiple ERP instances, acquisitions, regional variants | Join the calendar tables of each system and count mismatched dates |
| The tz database version is known and current | Future offsets and transitions depend on it | Pinned version, no update process | Log the tz release with every load; test a known future transition |
| Offsets stored with historical events are the ones in force at the time | History must not change when rules change | Offsets recomputed at query time | Compare stored offset against recomputed for a sample of historical dates |
| Weekly volume is uniformly distributed across days within the boundary week | Required to split a week across two fiscal periods | No Sunday despatch, holiday shutdowns, end-of-month pushes | Compare daily and weekly series over an overlapping window |
| Planning buckets are no finer than the underlying data | Finer buckets imply information that does not exist | Allocation models disguised as data | Trace a daily value back to its source record |
| The planning bucket is no finer than the lead time | Buckets inside the frozen horizon cannot be acted on | Lead time shorter than the bucket, or a genuinely responsive process | Compare bucket length with the frozen horizon and the replenishment cycle |
| The cut-off is the same for every metric in a report | Otherwise the report mixes two database states | Different teams run extracts at different times | List the cut-off instant per metric in the report metadata |
| Public holidays are jurisdictional, not global | Holiday shifts explain demand movement | A single global holiday calendar applied to all sites | Map holiday calendars to site country and region |

Two limits of the whole framing are worth stating.

**Calendar correctness does not make a comparison meaningful.** Two periods can be perfectly
aligned and still incomparable because the product mix changed, a promotion moved, or a
distribution centre opened. Correct calendars are a necessary condition, not a sufficient one.
The remaining work is the subject of Topics 06 and 07, and of the segmentation material later in
the program.

**The date spine is only as good as its generation rule.** A calendar table is a piece of
software with a version, and it can be wrong. The common defects are duplicate rows for a date,
overlapping week ranges, and a fiscal year with the wrong number of weeks. Every one of them is
detectable with three assertions — one row per date, no overlapping intervals, and a week count
of 52 or 53 — and almost none of them are tested in production.

## 12. Exercises

### Exercise 1 — ISO week labelling (recall and application, 25 minutes)

Assign the ISO week date to each of these Gregorian dates, showing the rule you applied:
2025-12-28, 2025-12-29, 2025-12-31, 2026-01-01, 2026-01-04, 2026-01-05, 2027-01-01.

**Deliverable:** a seven-row table with the date, the ISO week date, and the Gregorian year each
belongs to.
**Acceptance criteria:** the first-Thursday rule is stated; the two dates whose ISO year differs
from the Gregorian year are identified; no date is assigned a week number without the week-year.
**Time:** 25 minutes.

### Exercise 2 — The 53-week arithmetic (calculation, 30 minutes)

A retailer reports fiscal 2024 at 2,340,000 units over 52 weeks, and fiscal 2025 at 2,410,200
units over 53 weeks, of which the 53rd week contributed 38,000 units.

**Deliverable:** total growth, weekly-average growth, and like-for-like growth, each to two
decimal places, with the formula shown.
**Acceptance criteria:** all three numbers differ; the direction of each bias is stated; a
one-sentence recommendation says which figure should appear in the annual report and which in
the incentive plan.
**Time:** 30 minutes.

### Exercise 3 — The bucket that moves by an hour (calculation, 30 minutes)

A despatch leaves Hamburg at 05:10 Central European Time. Convert it to New York local time on
two dates: 5 March 2026 and 12 March 2026. Both are Thursdays. State the date and the offset
used in each case.

**Deliverable:** a two-row table with the UTC instant, the New York local reading, the offset,
and the resulting calendar date.
**Acceptance criteria:** the correct offset is chosen for each date and the reason is stated;
the difference in calendar date between the two Thursdays is identified as greater than seven
days; the consequence for a daily report is stated in one sentence.
**Time:** 30 minutes.

### Exercise 4 — Build and test a date spine (application, 35 minutes)

Construct a date spine for 2026-01-01 to 2027-01-03 with columns for date, ISO week-year, ISO
week, ISO weekday, and week start. Then run the three assertions from section 11.

**Deliverable:** the spine (any tool), the three assertion results, and a count of rows per ISO
week.
**Acceptance criteria:** exactly 368 rows; every ISO week has exactly 7 rows; the total number of
distinct ISO weeks is 53; the assertion that fails, if any, is explained rather than patched.
**Time:** 35 minutes.

### Exercise 5 — Write the cut-off policy (design, 45 minutes)

For a company with warehouses in the Netherlands, the United States, and Singapore, all
reporting into a single weekly operations review, write the cut-off policy.

**Deliverable:** one page stating the reporting zone, the cut-off instant in UTC, the handling of
late records, the restatement window, and the boundary case of a despatch at 00:30 local in each
of the three sites.
**Acceptance criteria:** the cut-off is an instant, not a date; the reporting zone is chosen and
justified rather than inherited; the three boundary cases produce stated, different outcomes;
the policy states what happens when a period is restated.
**Time:** 45 minutes.

### Exercise 6 — Critique the chart (critique, 30 minutes)

A weekly demand chart shows a flat series of about 20,000 units per week, with one week at
roughly 40,000 units in the last week of December, repeated every year. The analyst concludes
that year-end demand doubles.

**Deliverable:** a written critique of no more than 300 words, and a corrected version of the
claim.
**Acceptance criteria:** at least two alternative explanations are given, one of which is a
calendar artefact; the arithmetic that distinguishes them is shown; the critique does not
require any additional data to reach a provisional conclusion.
**Time:** 30 minutes.

### Exercise 7 — Reconcile the two calendars (application, 50 minutes)

A supplier reports 20,000 cases in every ISO week of 2026. The customer reports on a 52-week
fiscal year beginning on the Sunday nearest 31 January. Reconcile the two series.

**Deliverable:** a table showing, for ISO 2026-W53, the seven dates, the fiscal year each falls
in, and the volume allocated under a uniform daily assumption; plus the annual totals under
three join strategies.
**Acceptance criteria:** the totals under the week-number inner join, the clamp join, and the
date-spine join are all stated; the difference between them is attributed to a named mechanism;
the residual assumption is stated with its bound.
**Time:** 50 minutes.

### Exercise 8 — What the month destroyed (application, 40 minutes)

Take any daily series of at least two years that has a weekly pattern. Aggregate it to calendar
months and to 4-week periods. Compare the variance, the first-order autocorrelation, and the
apparent seasonal shape.

**Deliverable:** a table of the three statistics at three granularities, plus one paragraph on
which patterns are recoverable and which are not.
**Acceptance criteria:** the weekly pattern's amplitude is quantified at daily granularity and
shown to be absent at monthly; the 4-week aggregation is distinguished from the calendar-month
aggregation; no claim is made that disaggregation can recover the lost pattern without an
assumed profile.
**Time:** 40 minutes.

### Exercise 9 — Audit a timestamp column (audit, 40 minutes)

Take a timestamp column from any system you can inspect. Determine whether it is an instant, a
naive local time, or a formatted string; whether an offset is stored; and whether the zone is
recoverable.

**Deliverable:** a finding of no more than one page, with the evidence for the classification
and a list of the queries that would silently produce wrong answers.
**Acceptance criteria:** the classification is supported by a specific test rather than by the
column name; at least one query is shown to be wrong under the finding; the recommended fix
distinguishes historical repair from going-forward policy.
**Time:** 40 minutes.

### Exercise 10 — Assumptions ledger entry (design, 30 minutes)

Write the assumptions ledger entry for the calendar of a dataset you know, in the format of
section 11.

**Deliverable:** three rows — the assumption, why it is needed, what breaks it, how to test it —
plus the bucket definition in one sentence.
**Acceptance criteria:** each test is executable against the data rather than a request for
documentation; the bucket definition states length, boundary rule, cut-off instant, and zone;
at least one assumption is one the organisation has never written down.
**Time:** 30 minutes.

## 13. Self-check answers

**Exercise 1 — ISO week dates.**

| Gregorian date | Weekday | ISO week date | ISO year |
|---|---|---|---|
| 2025-12-28 | Sunday | 2025-W52-7 | 2025 |
| 2025-12-29 | Monday | 2026-W01-1 | 2026 |
| 2025-12-31 | Wednesday | 2026-W01-3 | 2026 |
| 2026-01-01 | Thursday | 2026-W01-4 | 2026 |
| 2026-01-04 | Sunday | 2026-W01-7 | 2026 |
| 2026-01-05 | Monday | 2026-W02-1 | 2026 |
| 2027-01-01 | Friday | 2026-W53-5 | 2026 |

The rule: 1 January 2026 is a Thursday, so it lies in week 01, and week 01 runs from Monday 29
December 2025 to Sunday 4 January 2026. Three of the seven dates have an ISO year different from
their Gregorian year: 2025-12-28 belongs to ISO 2025 while sitting at the end of the Gregorian
year, and 2027-01-01 (and 2025-12-29 to 2026-01-04) sit in ISO 2026. The two dates in the
question whose ISO year differs from the Gregorian year are 2025-12-28 and 2027-01-01.

A common wrong answer is to assign 2027-01-01 to 2027-W01. That is wrong because 2027 begins on
a Friday, and a 1 January on a Friday always belongs to the previous ISO year's last week.

**Exercise 2 — the 53-week arithmetic.**

```text
   total growth      = 2,410,200 / 2,340,000 - 1 = 0.030000 = +3.00%
   weekly average    = 2,410,200 / 53 = 45,475.47
   prior average     = 2,340,000 / 52 = 45,000.00
   average growth    = 45,475.47 / 45,000 - 1 = 0.010566 = +1.06%
   52-week base      = 2,410,200 - 38,000 = 2,372,200
   like-for-like     = 2,372,200 / 2,340,000 - 1 = 0.013761 = +1.38%
```

The naive total overstates by 1.62 points. The weekly average understates by 0.32 points,
because the 53rd week at 38,000 units is 83.6% of the average week. The like-for-like figure of
+1.38% is the one that describes what the business did.

The recommendation: report the like-for-like figure in the annual report and disclose the
53-week year explicitly. If the incentive plan is set on total growth, and the plan year is a
53-week year, the plan is 1.9% easier than the prior year's and nobody has told the participants.
The fix is either to set the target on a like-for-like basis or to state the week count in the
target itself.

**Exercise 3 — the bucket that moves by an hour.**

```text
   2026-03-05 05:10 +01:00 (CET)  = 2026-03-05T04:10Z
   New York on 5 March is on EST (UTC-05:00) because US DST begins 8 March 2026
   2026-03-05T04:10Z - 5 h         = 2026-03-04 23:10 EST   -> date 4 March

   2026-03-12 05:10 +01:00 (CET)  = 2026-03-12T04:10Z
   New York on 12 March is on EDT (UTC-04:00)
   2026-03-12T04:10Z - 4 h         = 2026-03-12 00:10 EDT   -> date 12 March
```

Both origin dates are Thursdays, seven days apart. The destination dates are eight days apart,
because the first falls before the local midnight and the second after it. The daily report at
the destination assigns a despatch from the same hour of the same weekday to two different
calendar days for three weeks a year — the window between the American transition on 8 March
and the European transition on 29 March, during which the transatlantic offset difference is
five hours rather than six.

The general lesson: a one-hour offset change is enough to move an event across a date, week, and
month boundary simultaneously when the event sits near midnight. Events do not cluster away from
midnight. End-of-shift confirmations and batch interfaces cluster exactly there.

**Exercise 4 — the date spine.**

The window 2026-01-01 to 2027-01-03 inclusive contains 368 days: 365 days of 2026 plus three days
of 2027. It spans 53 distinct ISO weeks, because ISO 2026 runs from Monday 29 December 2025 to
Sunday 3 January 2027.

The assertion results:

| Assertion | Result | Explanation |
|---|---|---|
| Exactly one row per date | Pass, 368 rows | Window is contiguous and inclusive at both ends |
| No overlapping week ranges | Pass | ISO weeks are disjoint by construction |
| Every week has exactly 7 rows | Fail | Week 2026-W01 contributes only 1–4 January, four rows |
| Week count is 52 or 53 | Pass, 53 | ISO 2026 is a 53-week year |

The third assertion fails because the window starts mid-week. That is not a defect in the spine;
it is a defect in the assertion, which should be stated as "every week has 7 rows unless it is a
boundary week of the window". The exercise is designed to show that a test which fails for a
benign reason is as dangerous as one that passes for a false reason. Patching the window to start
on 2025-12-29 fixes the test and silently changes the analysis period.

**Exercise 7 — reconciling the two calendars.**

| Strategy | Annual total (cases) | Error vs 1,060,000 | Weekly series |
|---|---:|---:|---|
| Inner join on week number | 1,040,000 | −20,000 (−1.89%) | Week 53 missing |
| Left join, week clamped to 52 | 1,060,000 | 0 | Week 52 = 40,000 (+100%) |
| Date spine, uniform daily split | 1,060,000 | 0 | No spike; boundary weeks split |

The clamp join is the instructive one. It gets the annual total exactly right and produces the
worst weekly series of the three. Any control that compares totals will approve it. A model
trained on it will learn that the last week of every year is twice as large as every other week,
and will forecast a spike that does not exist.

The date-spine allocation for the fiscal boundary week:

```text
   2027-W04, 20,000 cases, uniform daily = 20,000 / 7 = 2,857.14
   fiscal 2026 share (6 days) = 17,142.9 cases
   fiscal 2027 share (1 day)  =  2,857.1 cases
```

The residual assumption is uniform daily despatch. If no despatch occurs on Sunday, the true
share of the Sunday in fiscal 2027 is zero and the error is 2,857 cases, or 14.3% of a week. The
bound is six-sevenths of a week per boundary, twelve-sevenths across the two boundaries of a
fiscal year, or about 3.3% of annual volume in the worst case.

**Exercise 8 — what the month destroyed.**

A model answer reports three things.

| Statistic | Daily | 4-week period | Calendar month |
|---|---|---|---|
| Weekly-cycle amplitude | Full | Absent | Absent, with a moving weekday artefact |
| Variance | Highest | Lower, by roughly the within-week variance | Lowest |
| First-order autocorrelation | Low at daily lag | Higher, induced by aggregation | Higher still |
| Seasonal shape | Week-of-year visible | Month-of-year visible | Month-of-year visible, distorted |

The mechanism to state: aggregation is a moving average, and a moving average is a low-pass
filter. Averaging over 28 days removes a 7-day cycle exactly, because 28 is a multiple of 7. It
removes it because the four Mondays, four Tuesdays, and so on each contribute equally. Averaging
over 30 days removes most of it and leaves a beat whose phase advances through the calendar,
which is the weekday artefact that makes calendar-month retail data hard to compare.

The claim that must not be made: that a weekly pattern can be recovered from monthly data by
disaggregation. It can be *imposed* from an assumed profile, which is a model, and the model will
be right only if the profile is stable and known. In a promotion-heavy business it is neither.

**Exercise 5 rubric.** A strong answer names one reporting zone and justifies it by where the
decisions are made, not by where the servers are. It states the cut-off as a UTC instant derived
from that zone, and it states the DST rule that applies when the zone's offset changes. It handles
the three boundary cases differently and says why: a 00:30 despatch in the Netherlands is the
previous day in New York, a 00:30 despatch in Singapore is the previous day in New York and the
previous afternoon in the Netherlands. It states a restatement window in days and says what
readers of a restated report are told. A weak answer picks UTC as the reporting zone "to avoid
time zones", which removes the local-date problem by removing the business meaning, and then
reports a weekly total that no operations manager recognises.

**Exercise 6 rubric.** A strong critique names at least two mechanisms: the ISO or fiscal week
53 duplicate produced by a join, and a genuine seasonal peak that happens to fall in the same
week every year. It distinguishes them with arithmetic: a week-53 artefact appears once every
five or six years, whereas a seasonal peak appears every year, and the chart is described as
showing the spike every year, which rules the join artefact out and points to seasonality or to
a year-end promotional calendar. It then asks the question that decides the claim: is the spike
present in the daily data, and is it present in the same week of the fiscal calendar or the same
week of the Gregorian calendar? A weak critique says "correlation is not causation" and stops.

**Exercise 9 rubric.** A strong audit classifies the column by test rather than by name. The
decisive tests are: does the column contain values that differ by exactly one hour across a
transition date; does it contain any value with an offset suffix; and does a sort of the column
place an event that happened after another event before it. It then shows a query that is wrong
under the finding, for example a `date_trunc('week', ts)` that uses the server's zone. The repair
recommendation distinguishes historical reconstruction, which needs the offset recovered from
the system that wrote the row, from going-forward policy, which needs the offset stored at write
time. A weak audit reports the column type from the schema and stops.

**Exercise 10 rubric.** A strong ledger entry has tests that can be run tonight. "Check that the
fiscal calendar has no duplicate dates" is executable. "Confirm with finance that the calendar is
correct" is not. The bucket definition should read like a specification: seven-day buckets
beginning Sunday 00:00 in `America/New_York`, cut off at Saturday 23:59:59 local, half-open
intervals, restated for five business days. A weak entry states the bucket length and nothing else.

## 14. Interview questions

**1. "What is the difference between a date and a timestamp?"**

A strong answer says a date is a label with no instant and a timestamp is an instant with no
inherent label, and notes that most time-series bugs come from treating one as the other. It
mentions that a timestamp without an offset is a wall-clock reading rather than an instant. A
weak answer says a timestamp has a time component.

**2. "Why does ISO 8601 sometimes put 1 January in the previous year?"**

A strong answer gives the first-Thursday rule, derives the consequence — week 01 is the week
containing 4 January, so an ISO year begins between 29 December and 4 January — and notes that
this happens in three years out of seven. A weak answer says "because of leap years", which is
not the mechanism.

**3. "What is a 4-5-4 calendar and why would a retailer use one?"**

A strong answer describes the 4-5-4 week pattern within each 13-week quarter, states that the
NRF calendar is 4-5-4, distinguishes it from the 4-4-5 and 5-4-4 variants, and gives the reason:
every fiscal month contains a whole number of weeks and begins on the same weekday, so like days
compare with like days. It mentions the cost: the fiscal month does not align with the calendar
month, so a second calendar is needed. A weak answer calls it "4-4-5" and says it makes months
equal, which is false — the months are 28 or 35 days.

**4. "A company reports 3% growth in a 53-week year. What do you ask?"**

A strong answer asks for the week count in both years, computes the 1.9% mechanical effect, and
asks which normalisation the company uses. It may compute the like-for-like figure on the spot.
A weak answer asks whether the growth was organic.

**5. "How would you join a supplier's weekly shipment data to a retailer's weekly sales data?"**

A strong answer refuses to join on week number, explains the ISO Monday–Sunday versus NRF
Sunday–Saturday offset and the missing week 53, and proposes a date spine with facts aggregated
after labelling. It states the daily allocation assumption and its bound. A weak answer joins on
week-ending date, which matches nothing, or on week number, which matches the wrong rows.

**6. "What happens to a timestamp at 02:30 on the day clocks go forward?"**

A strong answer says that local time does not exist, that the IANA database records the
transition but does not resolve the ambiguity for you, and that the consuming application must
choose a convention — typically shifting forward by the gap — and document it. It notes the
mirror case in autumn, where the time occurs twice. A weak answer says the timestamp is invalid
and would be rejected, which is true of a good system and false of most.

**7. "Why not store local time? It is what the user sees."**

A strong answer separates the display concern from the storage concern: store the instant plus
the offset plus the zone, render locally. It gives the failure — two records with the same local
reading and different instants, sorted wrongly, bucketed wrongly, and unreproducible after a
legislative change. A weak answer says "UTC is best practice" without the mechanism.

**8. "How do you choose the bucket for a demand forecast?"**

A strong answer starts from the decision cadence and the lead time, then checks the data
granularity, then states that a bucket finer than the data or the lead time is false precision.
It mentions that finer buckets give more observations but noisier ones, and that the choice
interacts with seasonality estimation. A weak answer says "weekly, because weekly is standard".

**9. "What is lost when you aggregate daily demand to monthly?"**

A strong answer names the day-of-week pattern, promotion timing within the month, payday and
month-end effects, and the weekday composition artefact of calendar months. It states that the
loss is irreversible without an assumed profile. A weak answer says "detail".

**10. "How would you detect that a weekly series has a calendar defect?"**

A strong answer proposes three checks: plot the series and look for a spike or gap at the fiscal
year end; verify that every week in the source year appears exactly once in the target; and
compare the number of weeks per year against 52 or 53. It notes that an annual total
reconciliation will not catch it. A weak answer says "check for outliers".

**11. "A dashboard and a finance report disagree on last month's volume by 3%. Where do you
look first?"**

A strong answer looks at the cut-off instant and the reporting zone before looking at the data,
then at whether one is restated and the other frozen, then at whether one includes a partial
period. It notes that a 3% gap is the right order of magnitude for a boundary effect and too
small for a data loss. A weak answer starts by reconciling row counts.

**12. "What would you put in a calendar specification for a dataset?"**

A strong answer lists bucket length, boundary rule, cut-off instant with zone, interval
convention, late-arrival policy, restatement policy, and the time zone identifier, each with a
test. It mentions that the specification is versioned because the calendar itself changes. A
weak answer lists the date format.

## 15. Key terms

- **Calendar time** — the Gregorian date and clock reading, with no business meaning attached.
- **Business time** — the fiscal or retail period label a company reports in.
- **Event time** — the instant a physical event occurred, unambiguous on a global timeline.
- **Record time** — the instant a database row was written; the basis of as-of reporting.
- **Valid time / system time** — when a fact is true in the world, versus when it is current in
  the database.
- **Time bucket** — a contiguous interval used for aggregation.
- **Bucket boundary** — the rule assigning an instant on the edge to one bucket.
- **Cut-off** — the instant after which a record is excluded from a report run.
- **As-of date** — the system time that defines a report's contents.
- **Fiscal year** — the accounting year, not necessarily the calendar year.
- **4-5-4** — the NRF retail calendar: months of 4, 5, and 4 weeks within each 13-week quarter.
- **4-4-5 / 5-4-4** — variants placing the five-week month third or first in each quarter.
- **52/53-week year** — a fiscal year of 364 or 371 days, ending on a fixed weekday.
- **53rd week** — the extra week inserted roughly every five to six years.
- **Restatement** — re-labelling weeks so a 53-week year compares with a 52-week year.
- **Like-for-like (LFL)** — a comparison over equal-length periods with equal calendar
  composition.
- **Comparable sales** — retail growth over stores open in both periods, usually equal-week.
- **ISO 8601** — the notation standard for dates, times, and week dates.
- **ISO week date** — `YYYY-Www-D`, where the year is the ISO week-year.
- **Week 01** — the ISO week containing the year's first Thursday, equivalently 4 January.
- **ISO week-year** — the year owning the week, which may differ from the Gregorian year.
- **Ordinal date** — year plus day-of-year, `YYYY-DDD`.
- **UTC** — Coordinated Universal Time, the reference timeline.
- **Offset** — the difference between local time and UTC at a specific instant.
- **Time zone** — a region's legislated offset history and transition rules.
- **IANA tz database** — the maintained registry of zone identifiers and transitions.
- **DST** — daylight saving time, a seasonal offset change.
- **Spring-forward gap** — a local time range that does not exist.
- **Fall-back fold** — a local time range that occurs twice.
- **Epoch timestamp** — an integer count of seconds or milliseconds since a fixed instant.
- **RFC 3339** — the timestamp format requiring an explicit offset.
- **Naive datetime** — a wall-clock reading with no offset and no zone.
- **Granularity** — the finest distinction a dataset can represent.
- **Temporal hierarchy** — the nested set of aggregation levels for a series.
- **Planning bucket** — the interval a plan is expressed in, chosen from the decision cadence.
- **Frozen horizon** — the part of the plan that cannot change; a constraint on bucket choice.
- **Week-ending date** — the last day of a week, used as a retail bucket key.
- **Trading day** — a day on which the business books volume.
- **Trading-day adjustment** — a correction for the number of trading days in a period.
- **Holiday shift** — a public holiday moving between periods across years.
- **Leap day** — 29 February; biases daily averages if not counted explicitly.
- **Year-over-year (YoY)** — comparison with the same period one year earlier.
- **Trailing twelve months (TTM)** — a rolling twelve-month window.
- **Date spine** — one row per date, carrying every calendar label as a column.
- **Late-arriving data** — records whose event time precedes the as-of date but whose record time
  follows it.
- **Bitemporal** — a model that carries both valid time and system time.

## 16. Sources

**Peer-reviewed and academic**

- Dershowitz, D., and Reingold, E. M., *Calendrical Calculations*. Cambridge University Press.
  The arithmetic of week-based calendars; the ISO year as a sequence of 52 or 53 whole weeks, 364
  or 371 days, beginning on the Monday between 29 December and 4 January. Grade A.
- "The ISO Calendar." Chapter in a Cambridge University Press volume on calendrical algorithms.
  DOI 10.1017/cbo9781107051119.009. The week-numbering rule and its historical development.
  Grade A.
- Erasmus Research Institute of Management, "To Aggregate or Not to Aggregate: Should decisions
  and models have the same frequency?" Working paper ERS-2010-046-MKT. Simulation plus three
  real datasets; modelling at a finer granularity and forecasting up to the decision frequency
  generally outperforms modelling aggregated data, conditional on correct specification.
  Grade B — academic working paper; peer-review status not confirmed, so its findings are
  reported as the authors state them.

**Standards and official**

- ISO 8601-1:2019, *Date and time — Representations for information interchange — Part 1: Basic
  rules*. Week dates, Monday as first day, week 01 as the week containing the first Thursday.
  Grade B.
- ISO 8601-2:2019, *Part 2: Extensions*. Reduced precision, uncertain dates, intervals. Grade B.
- ISO/R 2015:1971, *Numbering of weeks*. The original four-day rule. Grade B.
- IANA, "Theory and pragmatics of the tz code and data," <https://www.iana.org/time-zones/theory>.
  The tz model of civil time and its limits. Grade B.
- IANA, "How to Read the tz Database," tzdb 2026b release documentation. Rule and Zone records,
  the SAVE column, and the handling of transition instants. Grade B.
- RFC 3339, *Date and Time on the Internet: Timestamps*, section 4.4. Unqualified local time is
  unacceptable for the internet; the `-00:00` convention. Grade B.
- W3C, "Working with Time and Timezones." Storage and serialisation guidance; floating time
  restricted to genuinely local concepts. Grade B.
- 26 CFR §1.441-2, *Election of taxable year consisting of 52-53 weeks*. Grade B.
- IRS Publication 538 (Rev. January 2022), *Accounting Periods and Methods*. The 52-53-week tax
  year: fixed weekday, last or nearest occurrence to month end. Grade B.
- IAS 34, *Interim Financial Reporting*. Comparability of interim periods. Grade B.

**Industry conventions and company practice**

- National Retail Federation, "4-5-4 Calendar," <https://nrf.com/resources/4-5-4-calendar>.
  Current calendars, the four-day rule for inserting a 53rd week, and the list of 53-week years
  including 1995, 2000, 2006, 2012, 2017, and 2023. Grade C — a voluntary industry convention,
  not a regulation; the variant named here is 4-5-4.
- National Retail Federation, *2025-2027 Retail Sales Reporting and 4-5-4 Merchandising
  Calendar*. Full week grid with release dates and holidays. Grade C.
- DataWorks, "National Retail Federation on Calendars," product help. Discusses 4-4-5, 4-5-4,
  and 5-4-4, and the 52-53-week year under US GAAP and the Internal Revenue Code. Grade D —
  vendor documentation; it describes 4-4-5 as the most common variant, which conflicts with the
  NRF's own naming and is one instance of the conflation this topic warns about.
- Oracle Database SQL Language Reference 26, `RETAIL` functions (`RETAIL_YEAR`,
  `RETAIL_QUARTER`, `RETAIL_MONTH`, `RETAIL_WEEK`, `RETAIL_DAY`), including the restated and
  non-restated parameter for 53-week years. Grade D.
- Oracle Support Document 745534.1, "What is the Impact of a 53-Week Retail Calendar in RDF?"
  Grade D.

**Vendor and analyst**

- AWS Forecast Developer Guide, "Data aggregation for different forecast frequencies." Time
  alignment boundaries and their defaults: weeks beginning Monday, months beginning on day 1.
  Grade D.
- OneStream, "Frequency Aggregation for Forecast Accuracy Comparison." Weeks do not divide evenly
  into months; straddling weeks are double-counted or truncated; partial periods at the ends of
  a window distort comparisons. Grade D.
- Brightwork Research and Analysis, "Test Results: Quarterly Versus Monthly Forecasting Buckets."
  A single client dataset; quarterly buckets beat monthly by roughly 2.5 MAPE points and weekly
  by roughly 4, attributed largely to model selection. Grade D — one dataset, vendor-published,
  no replication.

**Note on citations and grades.** Two sources do not fit the five-level scheme cleanly and are
labelled explicitly above: the Erasmus working paper, which is academic but not confirmed
peer-reviewed, and the NRF calendar, which is an industry convention with wide adoption rather
than a standard or a regulation. The DataWorks FAQ is included because it is widely cited and
because its 4-4-5 naming is itself evidence of the conflation problem. Vendor documentation is
reported as the vendor states it; parameter names and defaults change between releases. No
figure in this topic is presented as a base rate.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_case-pack-moq-rounding.md`](03_case-pack-moq-rounding.md)
- Next topic: [`05_lead-time-definitions.md`](05_lead-time-definitions.md)
- Related: [`../01_supply_chain_foundations/04_planning-processes.md`](../01_supply_chain_foundations/04_planning-processes.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)
