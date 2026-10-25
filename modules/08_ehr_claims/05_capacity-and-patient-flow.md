# Capacity and Patient Flow

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 05 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Topics 01 and 04
> Version 2026.3

## 1. Why this topic exists

Capacity is the constraint that turns a clinical decision into an operational one. A hospital can have enough clinicians, enough equipment, and enough demand, and still fail to admit a patient because no staffed bed is free. The patient waits in the emergency department (ED), the ED fills, ambulances divert, and patients leave without being seen. None of these failures is visible in an average occupancy number, because the average hides the peak.

The problem has worsened. Janke et al. analysed 46.2 million hospitalizations in *Health Affairs* in 2025 and found that ED boarding increased from 2017 to 2024; at the January 2022 peak, 40.1 percent of patients boarded more than four hours and 6.3 percent boarded more than 24 hours. Sangal et al., using 148.6 million ED encounters from 2017 to June 2024, reported that access failures rose from 18.5 to 28.7 percent, boarding over four hours rose from 2.6 to 6.2 percent, and ED length of stay over eight hours rose from 7.8 to 13.9 percent. Per-capita ED use and admissions were flat, which implicates declining hospital capacity and prolonged inpatient stays rather than rising demand. Berlyand et al. found that reduced functional bed capacity from inpatient boarding was associated with a 1.59 percentage point absolute increase in left-without-being-seen (LWBS) at three EDs.

Three failure patterns recur in capacity analytics.

**Averages hide the constraint.** A hospital at 85 percent average occupancy can be full every Monday and empty every Saturday. The average is a poor guide to the probability that an arriving patient will find a bed. Queueing theory says waiting time rises nonlinearly as utilisation approaches one, so the difference between 85 and 92 percent is not 7 percentage points of pressure; it is a multiple. Capacity decisions based on annual averages systematically understate the need.

**Throughput is confused with capacity.** Adding beds without adding staff, imaging capacity, or post-acute placement does not increase throughput; it moves the queue. The bottleneck determines the flow, and the bottleneck is often not the bed. It may be the CT scanner, the discharge pharmacist, the transport team, the SNF placement, or the cleaning crew. A capacity model that does not identify the bottleneck will recommend the wrong expansion.

**An intervention is evaluated without a flow model.** A discharge-before-noon programme can increase the share of morning discharges without reducing length of stay or boarding. The evidence is mixed: Artenstein et al. reported a 0.30-day reduction in length of stay and a 2.1-hour reduction in boarding, while Kausar et al. found no independent association with length of stay and a 41-minute increase in median boarding, and Hodges et al. reported longer order-to-discharge times and increased boarding. The difference is usually in the implementation: a programme that moves the discharge order earlier without moving the actual departure, or that creates a morning bottleneck in transport and pharmacy, does not free beds when they are needed.

Capacity analytics is therefore not a forecasting exercise alone. It is a model of a queueing system with variability, a bottleneck, and a feedback loop between the ED and the inpatient units. The output is a scenario comparison: which intervention, at what cost, reduces waiting or boarding by how much, and for whom.

A capacity decision also has a distributional consequence. The patients who board longest are often older, poorer, and less likely to speak the dominant language. A programme that improves the average without measuring the distribution can leave the gap unchanged or widen it. The equity analysis is part of the capacity analysis, not a separate exercise.

The operating question is not "how many beds do we have" but "how many patients can we admit today, at the hour they arrive, without delaying the next patient". That question requires a time-varying capacity model, not an annual bed count. It also requires a definition of the delay: boarding in the ED, waiting for a transfer, or waiting for a post-acute placement. The three are different queues with different owners.

> **Note:** Capacity is measured in staffed units, not licensed units. A licensed bed without a nurse is not available. A model that uses licensed beds will overstate capacity and understate boarding, which is the error that makes a hospital feel full while the dashboard says it has slack.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** an ED-to-inpatient flow as an input-throughput-output system and **identify** the bottleneck.
2. **Compute** utilisation, average concurrent patients, and waiting time using Little's law.
3. **Explain** why waiting time rises nonlinearly with utilisation, and **estimate** the effect of a capacity change on waiting.
4. **Distinguish** boarding time measured from ED arrival, from admission decision, and from bed request, and **choose** the definition that matches the decision.
5. **Build** a scenario model for a discharge-timing, bed-expansion, or staffing intervention, with cost per avoided boarding hour.
6. **Validate** a capacity model against observed wait times and census, and **diagnose** a model that reproduces the average but not the peak.
7. **Evaluate** a flow intervention with a design that handles seasonality, secular trends, and concurrent changes.
8. **Recognise** the equity consequences of capacity decisions, including which patients board longest and who leaves without being seen.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 2, 3, and 5 are the quantitative core.

## 3. The core mental model

A hospital is a network of queues. Patients arrive, wait for a resource, receive service, and move to the next queue. The flow is constrained by the slowest resource in the network, and the waiting is created by variability in arrivals and service times.

```text
INPUT         THROUGHPUT          OUTPUT
ED arrivals  -> triage -> workup -> decision -> bed request -> transfer
  |         |     |     |       |      |
  v         v     v     v       v      v
 arrival rate   provider  imaging  admission   bed avail-  discharge
 variability   capacity  capacity  decision   ability   destination
                             |
                             v
                          housekeeping,
                          transport, SNF
```

The system has three properties that determine its behaviour.

**Little's law.** For a stable system, the average number of patients in the system equals the arrival rate multiplied by the average time in the system: L = lambda x W. The law is exact under stationarity and conservation of flow, and it is robust to the distribution of arrivals and service times. It links the census, the admission rate, and the length of stay. If a hospital has 264 occupied beds and 58 admissions per day, the average length of stay is 264 / 58 = 4.55 days. If the length of stay rises to 5.0 days with the same admissions, the census rises to 290, and the hospital has 10 fewer beds of slack.

**Utilisation and waiting.** Utilisation is the arrival rate divided by the service capacity, or the occupied resource divided by the available resource. For a single-server queue, the average wait rises roughly as utilisation divided by one minus utilisation. At 50 percent utilisation, the wait is about one service time; at 80 percent, about four; at 90 percent, about nine; at 95 percent, about nineteen. The exact formula depends on the number of servers and the variability, but the shape is universal. A hospital that operates at 90 percent average occupancy has long waits even when the average looks manageable.

**Variability and the peak.** Arrivals and service times vary. The peak is not the average plus a small margin; it is the tail of a distribution. A hospital with 58 admissions per day may have 75 on a Monday and 40 on a Saturday. A bed model that uses the average will be wrong on the days that matter. The capacity needed to meet a service level is a quantile of the demand distribution, not the mean.

A fourth property is the feedback loop between the ED and the inpatient units. When inpatient beds are unavailable, admitted patients board in the ED, which consumes ED stretchers and nursing time, which slows the ED's ability to treat new arrivals, which increases LWBS and ambulance diversion. The ED is not the cause of the problem; it is the visible symptom of an inpatient capacity constraint. A model that treats the ED as an independent queue will miss the feedback.

### Little's law in practice

| Occupied beds | Staffed beds | Occupancy | Admissions per day | Implied ALOS | Slack beds |
|---:|---:|---:|---:|---:|---:|
| 240 | 300 | 80.0% | 58 | 4.14 | 60 |
| 264 | 300 | 88.0% | 58 | 4.55 | 36 |
| 276 | 300 | 92.0% | 58 | 4.76 | 24 |
| 285 | 300 | 95.0% | 58 | 4.91 | 15 |

The table is read in both directions. If the admission rate is fixed, a rise in ALOS raises the census and reduces slack. If the census is fixed, a rise in ALOS must be offset by a fall in admissions or an increase in beds. The arithmetic is exact, and it is the first check on any capacity claim.

### Waiting and utilisation

| Utilisation | Approximate wait multiplier | Interpretation |
|---:|---:|---|
| 50% | 1x service time | Comfortable, some slack |
| 70% | 2.3x | Noticeable waits at peak |
| 80% | 4x | Frequent waits, boarding likely |
| 90% | 9x | Long waits, access failures |
| 95% | 19x | System effectively saturated |

The multiplier is for a simple single-server queue and is illustrative. The direction is what matters: the last 5 points of occupancy cost more waiting than the first 50. A hospital that plans to 85 percent average occupancy is planning for significant peak waits, because the peak is above the average.


### Flow equations

```text
Utilisation (u) = arrival rate / service rate
Average number in system (L) = arrival rate x time in system (W)
Average time in system (W) = L / arrival rate
Queue ratio = u / (1 - u)
Approximate wait = service time x queue ratio
Service level = P(wait <= target)
```

The equations are linked. A change in arrival rate changes utilisation, which changes the queue ratio, which changes the wait. A change in service time changes both the wait directly and the utilisation. A capacity model that changes one variable without the others is incomplete.

The feedback loop is the part most often omitted. Boarding consumes ED capacity, which reduces the ED's service rate, which raises ED utilisation, which increases waits for new arrivals. The loop means that a bed shortage in the inpatient unit appears as an ED problem, and an ED improvement without an inpatient change may not reduce boarding.

### A queueing example with numbers

| Arrival rate | Service rate | Utilisation | Queue ratio | Wait if service is 30 min |
|---:|---:|---:|---:|---:|
| 4 per hour | 8 per hour | 50% | 1.0 | 30 min |
| 6 per hour | 8 per hour | 75% | 3.0 | 90 min |
| 7 per hour | 8 per hour | 87.5% | 7.0 | 210 min |
| 7.5 per hour | 8 per hour | 93.8% | 15.0 | 450 min |

The table is the case for not planning to the average. A 7 percent increase in arrivals from 7 to 7.5 per hour more than doubles the wait, because the system is already close to saturation.

> **Note:** The queueing formulas in this topic are approximations for teaching. A production model should use the observed distribution, a discrete-event simulation, or a validated queueing model with the right number of servers and priority classes. The approximations are useful for direction and magnitude, not for a capital decision without validation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Capacity | The maximum service rate of a resource in a period | Licensed beds instead of staffed | Patients per day or beds |
| Demand | The arrival rate of patients requiring service | Annual average instead of peak | Patients per hour or day |
| Utilisation | Demand divided by capacity, or occupied divided by available | Computed on licensed beds | Percent |
| Queue | Patients waiting for a resource | Confused with census | Count and wait time |
| Wait time | Time from arrival to service start | Confused with length of stay | Hours |
| Length of stay | Time from admission to discharge | Confused with ED LOS | Days |
| ED LOS | Time from ED arrival to ED departure | Confused with boarding | Hours |
| Boarding | Time an admitted patient waits in the ED for an inpatient bed | Measured from arrival instead of decision | Hours |
| Bed request | The time an inpatient bed is requested | Confused with admission decision | Timestamp |
| Admission decision | The time a clinician decides to admit | Confused with bed request | Timestamp |
| LWBS | Left without being seen | Confused with left against advice | Percent of arrivals |
| Ambulance diversion | Temporary closure of the ED to ambulance arrivals | Confused with ED crowding | Hours or episodes |
| Occupancy | Occupied bed-days divided by staffed bed-days | Computed with licensed beds | Percent |
| Staffed bed | A bed that can be occupied with available staff | Confused with licensed bed | Count |
| Licensed bed | A bed permitted by licensure | Confused with available capacity | Count |
| Bed-day | One bed occupied for one day | Confused with a patient day | Count |
| Census | Patients in beds at a point in time | Confused with admissions | Count |
| Turnover | Time to clean and prepare a bed | Ignored in bed models | Minutes |
| Discharge before noon | Proportion of discharges before 12:00 | Treated as a flow intervention | Percent |
| Discharge order time | Time the discharge order is written | Confused with departure time | Timestamp |
| Departure time | Time the patient leaves the unit | Confused with order time | Timestamp |
| Discharge barrier | A non-clinical reason for delayed departure | Ignored in LOS models | Category |
| Post-acute placement | SNF, rehab, or home health arrangement | Treated as a discharge | Days of delay |
| SNF | Skilled nursing facility | Treated as available capacity | Facility and bed |
| Observation stay | A short stay under observation status | Counted as inpatient | Hours |
| Transfer | Movement between units or facilities | Counted as a new admission | From and to |
| Throughput | Rate of patient flow through a process | Confused with volume | Patients per time |
| Bottleneck | The resource that limits system throughput | Assumed to be beds | Process step |
| Cycle time | Time to complete one patient's flow | Confused with wait time | Hours |
| Takt time | The pace required to meet demand | Confused with cycle time | Minutes per patient |
| Queue discipline | The rule for selecting the next patient | Assumed to be first-in, first-out | Rule |
| Priority queue | A queue ordered by clinical urgency | Confused with fairness | Rule |
| Service level | Probability that wait is below a threshold | Confused with utilisation | Percent |
| Arrival rate | Patients arriving per unit time | Confused with census | Patients per hour |
| Service rate | Patients served per unit time | Confused with capacity | Patients per hour |
| Server | A resource that serves patients | Confused with a bed only | Count |
| M/M/c | A queue with Poisson arrivals, exponential service, c servers | Treated as realistic | Model |
| Discrete-event simulation | A simulation of events and resources over time | Confused with forecasting | Scenario output |
| Forecast | A prediction of future demand | Confused with capacity | Count per period |
| Seasonality | Regular variation by time of year | Ignored in annual averages | Pattern |
| Peak | The highest demand in a period | Confused with average | Count per hour |
| Surge | A temporary increase in demand | Confused with capacity | Scenario |
| Census management | Moving patients to balance units | Confused with discharge planning | Operational practice |
| Bed huddle | A daily meeting to allocate beds | Confused with a model | Workflow |
| Discharge lounge | A waiting area for discharged patients | Treated as a bed | Physical space |
| Hospital at home | Acute care delivered at home | Treated as no capacity | Programme |
| Transitional care | Care between settings | Confused with post-acute | Programme |
| Access block | The inability to admit a patient because no bed is available | Confused with ED crowding | Hours |
| Functional bed capacity | Staffed beds minus beds occupied by boarders | Confused with licensed capacity | Beds |
| NEDOCS | A numerical ED overcrowding score | Treated as a diagnosis | Score |
| Input-throughput-output | A model of ED flow | Treated as a linear process | Framework |
| Throughput time | Time from arrival to departure for one patient | Confused with wait time | Hours |
| Arrival pattern | The distribution of arrivals by hour | Assumed to be uniform | Count per hour |
| Discharge pattern | The distribution of departures by hour | Assumed to match arrivals | Count per hour |
| Bed blocking | A bed held by a patient who cannot leave | Confused with occupancy | Bed-hours |
| Surge bed | A bed opened temporarily for demand | Counted as staffed capacity | Count |
| Staffing ratio | Patients per nurse | Assumed constant | Ratio |
| Float pool | Staff reassigned across units | Treated as free capacity | FTE |
| Capacity calendar | The record of capacity changes over time | Ignored in trends | Date and beds |


## 5. The mechanics, step by step

The procedure below converts a capacity complaint into a flow model with a bottleneck, a scenario comparison, and a cost per avoided boarding hour. It assumes access to ED timestamps, bed assignment data, discharge timestamps, staffing, and census.

### Step 1. Define the flow problem and the decision

**Input:** a complaint such as "the ED is always full" or "we need more beds".

**Output:** one sentence naming the flow, the population, the time window, the decision, and the owner.

**Rationale:** "the ED is full" is a symptom, not a problem. The problem may be that admitted patients wait for beds, that discharged patients leave late, that imaging turnaround is slow, or that post-acute placement is delayed. Each has a different owner and a different intervention. The decision may be to add beds, change discharge timing, add a discharge lounge, or reallocate staff. The sentence should name the decision and the person who makes it.

Write the sentence as: "For [patient flow] during [period], [owner] decides [action] to reduce [wait metric] without increasing [harm]." A sentence that cannot name the wait metric is not ready for modelling.

**Branch point:** if the decision is capital (build or convert beds), the analysis needs a longer horizon and a full cost basis. If the decision is operational (discharge timing, staffing), a shorter horizon and a marginal cost basis are appropriate.

### Step 2. Map the process

**Input:** the flow from arrival to departure.

**Output:** a process map with resources, queues, and decision points.

**Rationale:** the map identifies where patients wait and which resource they wait for. A typical ED-to-inpatient flow has these queues: ED arrival to triage, triage to provider, provider to decision, decision to bed request, bed request to bed assignment, bed assignment to transfer, transfer to unit, and unit to discharge. Each queue has a resource and a service time. The longest queue is not always the bottleneck; a short queue with a high arrival rate can constrain the system more.

The map should include the feedback loops: a full inpatient unit blocks ED beds, which slows ED throughput, which increases LWBS. A linear map misses this loop.

**Branch point:** if the process differs by service (medical, surgical, cardiac, paediatric), map each separately. Aggregating them hides the bottleneck in the smallest service.

### Step 3. Choose the unit and time granularity

**Input:** the decision and the data availability.

**Output:** the unit (patient, bed, bed-hour, day) and the time step (hour, shift, day, week).

**Rationale:** capacity problems are peak problems, so the time step must be fine enough to see the peak. A daily average hides the 10 a.m. bed crunch. An hourly census is better for ED boarding; a shift-level census is adequate for staffing. The unit must match the decision: a bed expansion is a bed count, a discharge-timing intervention is a bed-hour, and a staffing change is a nurse-hour.

State the time zone and the timestamp source. ED timestamps may come from the ED information system, the EHR, or the bed management system, and they may differ by minutes to hours. The definition must specify which source is authoritative.

**Branch point:** if the data are only daily, the analysis can still identify the bottleneck but cannot quantify peak waits. State the limitation and use the finest available granularity.

### Step 4. Measure demand

**Input:** arrival timestamps by setting and service.

**Output:** arrival rate by hour, day, and season, with the peak and the distribution.

**Rationale:** demand is not the annual average. It has a day-of-week pattern, a time-of-day pattern, and a seasonal pattern. ED arrivals peak in the afternoon and evening; admissions peak in the late afternoon; discharges peak in the late morning. The mismatch between admission and discharge timing is a primary cause of boarding.

Report the mean, the 90th percentile, and the maximum by hour. A capacity model that uses the mean will under-provide for the peak. A model that uses the maximum will over-provide for most days. The service level determines which quantile to use.

**Branch point:** if the demand series has a trend or a structural break (a new service, a closure, a pandemic), model the periods separately. A single average across a break is not a demand estimate.

### Step 5. Measure capacity

**Input:** staffed beds by unit, staffing rosters, diagnostic capacity, and post-acute availability.

**Output:** capacity by resource, by hour and day, with the staffed constraint stated.

**Rationale:** capacity is a vector, not a scalar. A unit may have 30 licensed beds, 27 staffed beds, and 24 available beds because three are closed for isolation and three are occupied by boarders. The effective capacity for a new admission is the staffed beds minus the beds occupied by patients who cannot leave. A model that uses licensed beds overstates capacity by 10 to 20 percent in a typical hospital.

Measure capacity at the same time granularity as demand. A unit that closes beds at night has a different capacity by hour. A diagnostic resource (CT, MRI, echo) may be the binding constraint even when beds are free.

**Branch point:** if the capacity data are not available by hour, use the staffing roster to reconstruct the staffed beds by shift. A bed count without a staff count is a licensed-bed count in disguise.

### Step 6. Measure variability

**Input:** the arrival and service time distributions.

**Output:** the coefficient of variation, the peak-to-mean ratio, and the day-of-week pattern.

**Rationale:** variability is the reason a system with spare average capacity still has waits. Two hospitals with the same average occupancy can have very different wait times if one has smooth demand and the other has a Monday surge. The coefficient of variation (standard deviation divided by the mean) summarizes the variability; the peak-to-mean ratio summarizes the planning challenge.

Measure variability in arrivals and in service times. A discharge process with a long tail (a few patients waiting days for SNF placement) creates a bed shortage even when the average length of stay is stable. The tail, not the mean, determines the capacity needed for the service level.

**Branch point:** if variability is driven by a few outliers (complex discharges, social barriers), model the outliers separately and consider a discharge lounge or a transitional unit rather than more beds.

### Step 7. Compute utilisation and Little's law

**Input:** demand, capacity, census, admissions, and length of stay.

**Output:** utilisation by resource, implied length of stay, and average concurrent patients.

**Rationale:** Little's law links the three core metrics and exposes inconsistencies. If the census is 264, admissions are 58 per day, and the stated ALOS is 4.0 days, one of the numbers is wrong: 58 x 4.0 = 232, not 264. The inconsistency is usually a definition (observation, transfers, outpatients in beds) or a data error. Finding it before modelling saves a wrong conclusion.

Utilisation should be computed for each resource, not only for beds. The CT scanner may be at 95 percent utilisation while the beds are at 88 percent; the scanner is the bottleneck. The resource with the highest utilisation, or the longest queue, is the candidate bottleneck.

**Branch point:** if utilisation exceeds 100 percent, the denominator is wrong or the period includes a capacity change. Reconcile before proceeding.

### Step 8. Locate the bottleneck

**Input:** the queue lengths and utilisation by resource.

**Output:** a ranked list of candidate bottlenecks with evidence.

**Rationale:** the bottleneck is the resource where queues accumulate and where a small increase in capacity produces the largest reduction in total wait. Queue accumulation is visible as a rising backlog by hour; utilisation is visible as a ratio. The two should agree. If the ED queue grows while the inpatient unit is at 85 percent, the bottleneck may be bed assignment or housekeeping, not the bed itself.

Use a simple test: add one unit of capacity to each candidate resource in the model and see which reduces the target wait the most. The resource with the largest marginal effect is the bottleneck. This is more reliable than intuition, which usually points to the most visible queue.

**Branch point:** if the bottleneck is a process (bed assignment, transport, cleaning) rather than a resource, the intervention is a workflow change, not a capacity expansion.

### Step 9. Build the baseline model

**Input:** the process map, demand, capacity, and variability.

**Output:** a model that reproduces the observed census, wait time, and boarding hours.

**Rationale:** the model can be a spreadsheet with Little's law and a queue approximation, a discrete-event simulation, or a system dynamics model. The choice depends on the question. A spreadsheet is adequate for a steady-state capacity comparison; a simulation is needed for a time-varying system with priority queues and blocking. The model should be as simple as the question allows, and no simpler.

A discrete-event simulation represents patients as entities, resources as servers, and time as a sequence of events. It can model priority, blocking, and shift changes. It requires calibration and validation, and it can be overbuilt. A spreadsheet with Little's law and an M/M/c approximation is often enough to rank interventions.

**Branch point:** if the model cannot reproduce the observed boarding time within 20 percent, do not use it for a decision. Calibrate the service times and the capacity first.

### Step 10. Calibrate and validate

**Input:** the model output and the observed data.

**Output:** a calibration table comparing model and observed census, wait, and boarding by hour and day.

**Rationale:** a model that matches the annual average but not the peak is not validated. Compare by hour and day, and compare the distribution, not only the mean. A model that understates the peak will understate the benefit of a peak-focused intervention.

Validate on a holdout period, not the period used for calibration. If the model was calibrated on 2024, validate on 2025. If the process changed in 2025, validate on the period after the change and state the difference.

**Branch point:** if the model fails validation, the process map or the capacity assumption is wrong. Return to step 2 or step 5 rather than adjusting parameters to fit.

### Step 11. Scenario analysis and cost per avoided boarding hour

**Input:** the calibrated model and the candidate interventions.

**Output:** a scenario table with the effect, cost, net benefit, and cost per avoided boarding hour.

**Rationale:** the scenario table is the decision product. Each scenario changes one or more parameters: discharge timing, bed count, staffing, imaging capacity, or post-acute placement. The effect is the change in boarding hours, wait time, or LWBS. The cost is the incremental operating or capital cost. The cost per avoided boarding hour is the cost divided by the avoided hours.

The value of an avoided boarding hour is the staff and space cost of the ED care that is freed, plus the downstream value of reduced LWBS and diversion. The first is measurable; the second is an estimate. State both and do not double-count.

**Branch point:** if the cost per avoided boarding hour exceeds the value, the intervention is not viable on financial grounds. It may still be justified on clinical or access grounds; state that separately.

### Step 12. Implement, monitor, and iterate

**Input:** the chosen intervention and the monitoring plan.

**Output:** a dashboard with census, boarding, LWBS, discharge timing, and capacity by hour, plus a review cadence.

**Rationale:** capacity interventions fail when the workflow does not change. A discharge-before-noon programme that moves the order time but not the departure time does not free beds. Monitor the actual departure time, the bed availability at the peak, and the boarding hours. Review the dashboard daily during the pilot and monthly afterward.

Monitor for unintended consequences: earlier discharges may increase readmissions, and a discharge lounge may shift the burden to the patient. Include readmission, patient experience, and equity metrics in the monitoring plan.

**Branch point:** if the intervention does not move the target metric within the pre-specified period, stop or redesign. Do not continue a programme because it is popular.

### Decision table: boarding time definitions

| Definition | Start | End | What it measures | Use |
|---|---|---|---|---|
| Arrival-to-bed | ED arrival | Bed assignment | Total ED experience including workup | Patient experience, ED LOS |
| Decision-to-bed | Admission decision | Bed assignment | Inpatient capacity delay | Capacity and flow |
| Request-to-bed | Bed request | Bed assignment | Bed management and housekeeping | Bed management |
| Assignment-to-transfer | Bed assignment | Physical transfer | Transport and unit readiness | Transfer process |

The choice matters. In the worked example, arrival-to-bed is 8.1 hours and decision-to-bed is 6.0 hours; the 2.1-hour difference is ED evaluation, not boarding. A programme that targets boarding should use decision-to-bed.

### Decision table: capacity levers

| Lever | Mechanism | Time to effect | Cost basis | Risk |
|---|---|---|---|---|
| Add staffed beds | Increases capacity | Months to years | Capital plus operating | Fixed cost if demand falls |
| Improve discharge timing | Frees beds earlier | Weeks | Staff and transport | Readmission if too early |
| Discharge lounge | Frees bed before departure | Weeks | Space and staff | Patient experience |
| Post-acute partnership | Reduces placement delay | Months | Contract and liaison | Dependency on partner |
| Imaging capacity | Reduces workup time | Months | Equipment and staff | Underuse if demand shifts |
| Staffing ratio | Increases effective beds | Weeks | Nurse wages | Recruitment |
| Transfer center | Balances load across sites | Weeks | Staff and IT | Patient choice |
| Hospital at home | Diverts admissions | Months | Programme and technology | Eligibility and safety |

### Edge cases that invalidate a capacity model

**Missing bed data.** Bed assignment timestamps may be missing or entered retrospectively. A bed assigned at 14:00 but documented at 16:00 understates boarding by two hours. Detect by comparing the bed management system to the EHR and by auditing a sample of records.

**Boarding definition drift.** A change in the definition (arrival versus decision) can create a step change in the metric without any change in flow. Detect by maintaining a definition version and segmenting the trend at the change date.

**Observation stays.** Observation patients occupy beds but are not counted as inpatients in some systems. A model that excludes them understates occupancy and overstates capacity. Detect by reconciling the bed census to the observation census.

**Discharge timing versus departure.** A discharge order at 10:00 with a departure at 15:00 does not free a bed at 10:00. Measure the departure, not the order. Detect by comparing order and departure timestamps.

**Transfer delays.** A patient assigned to a bed but not transferred still occupies an ED stretcher. The bed is available but the patient is not in it. Detect by measuring assignment-to-transfer time separately.

**Mid-year capacity changes.** A unit closure, a bed conversion, or a staffing reduction changes capacity mid-period. A model that uses an annual average will misstate the peak. Detect by maintaining a capacity calendar.

**Demand shifts.** A new service line, a closure of a nearby hospital, or an infectious disease surge changes demand. A model calibrated on the prior year may not represent the current year. Detect by monitoring arrivals by week and by comparing to the forecast.

### Flow metrics and their denominators

| Metric | Numerator | Denominator | Time window | Common error |
|---|---|---|---|---|
| ED arrival rate | ED arrivals | Hours in the period | Hour | Daily average |
| Admission rate | ED admissions | ED arrivals | Day | Excluding transfers |
| Boarding hours per patient | Sum of decision-to-bed hours | Admitted patients | Day | Starting at arrival |
| Boarding bed-hours | Sum of boarding hours | Not applicable | Day | Counting patients, not hours |
| Occupancy | Occupied bed-days | Staffed bed-days | Day | Licensed beds |
| Discharge before noon | Discharges before 12:00 | Discharges | Day | Order time, not departure |
| LWBS rate | LWBS patients | ED arrivals | Day | Excluding left against advice |
| ED LOS | Departure minus arrival | ED visits | Day | Excluding boarders |
| Turnover time | Bed ready minus bed empty | Discharges | Day | Scheduled, not actual |
| Service level | Waits below threshold | All waits | Week | Mean wait |

The denominator column is the one that gets lost. A boarding metric computed per admitted patient and a boarding metric computed per ED visit answer different questions, and a table that mixes them cannot be compared.

### Queue model comparison

| Model | Assumptions | Output | Use | Limitation |
|---|---|---|---|---|
| Little's law | Stationary, conserved flow | Average number in system | Census and LOS checks | No wait distribution |
| M/M/c | Poisson arrivals, exponential service, c servers | Average wait and queue length | Steady-state capacity | Understates variability |
| M/G/c approximation | General service distribution | Approximate wait | Service-time variability | Requires calibration |
| Priority queue | Classes with different urgency | Wait by class | ED triage | Complex to calibrate |
| Discrete-event simulation | Explicit events and resources | Time-varying waits and queues | Scenario analysis | Build and validation cost |
| System dynamics | Stocks and flows | Aggregate behaviour | Strategic planning | Little individual detail |

The choice should follow the question. A board asking "how many beds" needs a scenario model, not a priority queue. An ED director asking "why are ESI-3 patients waiting longest" needs a priority queue or a simulation.

### Bottleneck test

| Candidate resource | Utilisation | Queue growth | Marginal effect of +1 unit | Rank |
|---|---:|---|---:|---:|
| Medical-surgical beds | 92% | +6 patients by 14:00 | -1.8 h boarding | 1 |
| CT scanner | 88% | +4 by 15:00 | -0.4 h boarding | 2 |
| Housekeeping | 76% | +2 by 13:00 | -0.9 h boarding | 3 |
| Transport | 64% | +1 by 16:00 | -0.2 h boarding | 4 |

The marginal effect column is the result of adding one unit of capacity in the model, not a correlation. Housekeeping ranks above CT despite lower utilisation because a cleaning delay blocks a bed that is already empty. The ranking is the output of the bottleneck test, and it can contradict the most visible queue.

### Little's law consistency checks

```text
Census = Admissions per day x ALOS
ALOS = Census / Admissions per day
Admissions per day = Census / ALOS
Occupied bed-days = Census summed over days
Occupancy = Occupied bed-days / Staffed bed-days
```

A useful audit is to compute the implied ALOS from the census and admissions, then compare it to the reported ALOS. A difference above 0.2 days usually means observation patients, transfers, or a discharge definition difference. The audit is cheap and catches the most common data error before it enters a model.

### Peak demand table

| Hour | ED arrivals | Admissions | Discharges | Net bed demand | Cumulative gap |
|---|---:|---:|---:|---:|---:|
| 08:00 | 8 | 3 | 2 | +1 | 1 |
| 10:00 | 12 | 4 | 8 | -4 | -3 |
| 12:00 | 14 | 5 | 6 | -1 | -4 |
| 14:00 | 16 | 6 | 4 | +2 | -2 |
| 16:00 | 18 | 6 | 8 | -2 | -4 |
| 18:00 | 15 | 5 | 4 | +1 | -3 |
| 20:00 | 12 | 4 | 2 | +2 | -1 |

The cumulative gap is the running difference between admissions and discharges. A negative gap means beds are being freed; a positive gap means demand is outpacing supply. The peak gap, not the daily total, is what determines boarding.

### Costing a flow intervention

```text
Avoided boarding hours = Patients affected x Hours earlier per patient
Value per avoided boarding hour = ED nursing cost per bed-hour
Net = Avoided hours x Value per hour - Programme cost
Cost per avoided boarding hour = Programme cost / Avoided hours
Breakeven avoided hours = Programme cost / Value per hour
```

The value per boarding hour is not the full ED cost per visit; it is the marginal staff and space cost of holding one additional patient for one hour. Using the full cost overstates the benefit. Using zero understates it. A defensible estimate is the nursing cost per bed-hour plus the marginal cost of ED space, stated explicitly.

### What to measure before modelling

| Data element | Source | Granularity | Why it matters |
|---|---|---|---|
| ED arrival | ED information system | Minute | Demand and wait |
| Admission decision | EHR order | Minute | Boarding start |
| Bed request | Bed management | Minute | Bed management delay |
| Bed assignment | Bed management | Minute | Capacity delay |
| Physical transfer | EHR or bed management | Minute | Transport delay |
| Discharge order | EHR | Minute | Discharge process |
| Physical departure | EHR or housekeeping | Minute | Bed availability |
| Staffed beds | Staffing roster | Shift | Capacity |
| Occupied beds | Bed census | Hour | Utilisation |
| Diversion status | ED log | Hour | Access failure |

### Validation targets

| Output | Acceptable error | Consequence if exceeded |
|---|---|---|
| Mean census | Within 3% | Capacity is misstated |
| Peak census | Within 5% | Peak intervention is mis-sized |
| Mean boarding | Within 20% | Benefit is unreliable |
| 90th percentile boarding | Within 25% | Tail is mis-modelled |
| LWBS rate | Within 1 percentage point | Access impact is unreliable |

A model that hits the mean but misses the peak is a model that will recommend the wrong intervention. The peak columns are the ones that matter for capacity.

The validation targets are not arbitrary. A 20 percent error in mean boarding is tolerable for ranking interventions but not for a business case; a 5 percent error in peak census can change whether a unit is described as full. Match the tolerance to the decision, and state it before the model is built.

A model that cannot meet the tolerance for its decision should not be used for that decision, even if it is the best model available.

That is a judgement about evidence, not about modelling skill.

The next section applies the procedure to a 300-bed hospital.
## 6. Worked example

A 300-bed community hospital is considering two interventions to reduce ED boarding: a discharge-timing programme and a 12-bed expansion. The medical-surgical unit has 300 staffed beds and 264 occupied beds, for 88 percent occupancy. There are 58 discharges and 58 admissions per day, of which 54 are ED admissions and 4 are direct admissions. The average length of stay implied by Little's law is 264 / 58 = 4.55 days. The ED sees 180 visits per day and admits 30 percent.

### Part A: demand, capacity, and boarding

The hospital's service area has 250,000 people. Annual ED visits are 180 x 365 = 65,700, or 262.8 visits per 1,000 population per year. That denominator is useful for comparing with regional benchmarks; it is not used in the bed calculation.

```text
Staffed beds = 300
Occupied beds = 264
Occupancy = 264 / 300 = 88.0%
Slack beds = 300 - 264 = 36
ED admissions per day = 180 x 0.30 = 54
Direct admissions per day = 4
Total admissions per day = 58
Implied ALOS = 264 / 58 = 4.55 days
```

Boarding is measured from the admission decision to bed assignment. The average is 6.0 hours for the 54 ED admissions.

```text
Boarding bed-hours per day = 54 x 6.0 = 324
Average concurrent boarders = 324 / 24 = 13.5
ED nursing cost per bed-hour = $68 / 4 patients per nurse = $17.00
Boarding nursing cost per day = 324 x $17.00 = $5,508
Annual boarding nursing cost = $5,508 x 365 = $2,010,420
```

The 13.5 concurrent boarders occupy ED stretchers and nursing time. They are not in inpatient beds, so the 36 slack beds overstate the effective capacity available to new admissions. The functional bed capacity is lower than 300 by the number of beds that cannot be used because the unit is staffed for 264 inpatients and the boarders are waiting.

### Part B: discharge-timing programme

The programme adds two discharge coordinators and a transport team to move 15 of the 58 daily discharges from the afternoon to before noon. Each moved discharge frees a bed about three hours earlier.

```text
Avoided boarding bed-hours per day = 15 x 3 = 45
New boarding bed-hours = 324 - 45 = 279
New average boarding time = 279 / 54 = 5.17 hours
Annual avoided bed-hours = 45 x 365 = 16,425
Value of avoided bed-hours = 16,425 x $17.00 = $279,225
Programme cost = 2.0 FTE coordinators x $95,000 + $30,000 transport and IT = $220,000
Net result = $279,225 - $220,000 = $59,225
Cost per avoided boarding hour = $220,000 / 16,425 = $13.39
Breakeven avoided hours = $220,000 / $17.00 = 12,941 per year = 35.5 per day
```

The programme avoids 45 bed-hours per day, above the 35.5 breakeven, and returns $59,225 per year on a nursing-cost basis. The margin is thin: a 22 percent shortfall in the number of moved discharges, or a $80,000 increase in programme cost, would eliminate the gain.

### Part C: 12-bed expansion

The alternative is to open 12 additional staffed beds at an annual operating cost of $1,150,000. The beds raise staffed capacity from 300 to 312, and occupancy falls from 88.0 to 84.6 percent.

```text
New occupancy = 264 / 312 = 84.6%
Queue ratio at 88.0% = 0.88 / 0.12 = 7.33
Queue ratio at 84.6% = 0.846 / 0.154 = 5.49
Wait reduction = 1 - (5.50 / 7.33) = 25.0%
New boarding time = 6.0 x (1 - 0.250) = 4.50 hours
New boarding bed-hours = 54 x 4.50 = 243.0
Avoided bed-hours = 324 - 243.0 = 81.0 per day
Annual value = 81.0 x $17.00 x 365 = $502,605
Net result = $502,605 - $1,150,000 = -$647,395
Cost per avoided boarding hour = $1,150,000 / (81.0 x 365) = $38.90
```

The bed expansion avoids more boarding hours than the discharge programme (81.0 versus 45 per day) but costs $38.90 per avoided hour against a value of $17.00. On a nursing-cost basis, it does not pay back. It may still be justified by access, safety, or revenue from additional admissions, but those benefits must be stated separately and not folded into the boarding calculation.

### Diagnostic 1: occupancy is computed on licensed beds

The unit has 320 licensed beds and 300 staffed beds. If the analyst uses the licensed denominator, occupancy is 264 / 320 = 82.5 percent, not 88.0 percent. The hospital appears to have 56 slack beds rather than 36, and the pressure looks manageable. The error is not arithmetic; it is a capacity definition. The staffed bed is the constraint, and the licensed bed is a regulatory permission.

The same error appears in reverse when a unit closes beds for staffing. If 18 beds are closed, staffed capacity is 282 and occupancy is 264 / 282 = 93.6 percent. The hospital is much closer to saturation than the licensed-bed dashboard suggests. The diagnostic is to reconcile the bed count to the staffing roster every month and to report licensed, staffed, and available beds separately.

### Diagnostic 2: boarding is measured from ED arrival

The ED information system records arrival-to-bed time as 8.1 hours. The admission decision occurs at 6.0 hours before bed assignment, and the remaining 2.1 hours is ED evaluation before the decision. Using 8.1 hours as boarding overstates boarding by 35 percent.

```text
Arrival-to-bed boarding bed-hours = 54 x 8.1 = 437.4
Decision-to-bed boarding bed-hours = 54 x 6.0 = 324
Overstatement = 437.4 - 324 = 113.4 bed-hours per day
Annual overstatement at $17 per hour = 113.4 x 365 x $17 = $703,647
```

A programme sized on the arrival-to-bed metric would target the wrong delay and would claim credit for reducing ED workup time, which is not boarding. The diagnostic is to plot the distribution of arrival-to-decision and decision-to-bed times separately, and to use decision-to-bed for capacity decisions.

### Varying one assumption

The discharge programme's net result is sensitive to its cost. If the two coordinators and transport team cost $300,000 rather than $220,000, the net becomes $279,225 - $300,000 = -$20,775, and the breakeven avoided hours rise to $300,000 / $17 = 17,647 per year, or 48.3 per day, above the 45 achieved. The decision reverses from fund to reject.

| Scenario | Avoided bed-hours/day | Annual value | Cost | Net | Cost per avoided hour | Decision |
|---|---:|---:|---:|---:|---:|---|
| Discharge timing, base | 45.0 | $279,225 | $220,000 | $59,225 | $13.39 | Fund |
| Discharge timing, higher cost | 45.0 | $279,225 | $300,000 | -$20,775 | $18.26 | Reject |
| 12-bed expansion | 81.0 | $502,605 | $1,150,000 | -$647,395 | $38.90 | Reject on cost |
| Discharge timing, 25 moves | 75.0 | $465,375 | $220,000 | $245,375 | $8.04 | Fund |

The table makes the decision rule explicit: the discharge programme is viable if it moves at least 48 discharges per day at the higher cost, or 35.5 at the base cost. The bed expansion is not viable on boarding cost alone at the assumed value per bed-hour.

### What the model does not capture

The model does not capture the revenue from additional admissions, the clinical harm of boarding, or the patient experience of a long ED stay. A bed expansion that loses money on boarding cost may still be justified if it enables additional surgical volume or reduces mortality. Those benefits require a different analysis, with a different outcome and a different valuation. Mixing them into the boarding calculation would make the arithmetic look better and the decision less clear.

### Part D: validation against observed boarders

The model's average of 13.5 concurrent boarders should be compared with the observed count by hour, not only with the daily mean.

| Hour | Observed boarders | Model boarders | Difference |
|---|---:|---:|---:|
| 08:00 | 8 | 9 | -1 |
| 11:00 | 12 | 14 | -2 |
| 14:00 | 17 | 17 | 0 |
| 17:00 | 18 | 17 | +1 |
| 20:00 | 12 | 11 | +1 |
| 23:00 | 7 | 6 | +1 |

The mean difference is small, but the morning is under-predicted by two boarders. The model uses a constant admission rate, while the observed morning has a higher arrival rate. A model calibrated only on the daily mean would miss the morning peak, which is exactly when the discharge programme has its effect. The validation target for peak boarding is within 25 percent, and the 11:00 error is 14 percent, so the model is acceptable for ranking but not for a precise capacity guarantee.

### Part E: sensitivity of the wait reduction

The 25.1 percent wait reduction from 12 beds depends on the queue approximation. If the system is more variable than the simple model assumes, the same capacity increase produces a smaller reduction; if it is less variable, a larger one.

| Assumption | Base wait | Wait after 12 beds | Reduction |
|---|---:|---:|---:|
| Simple queue, as modelled | 6.00 h | 4.50 h | 25.0% |
| Lower variability | 6.00 h | 4.05 h | 32.5% |
| Higher variability | 6.00 h | 5.10 h | 15.0% |
| Bottleneck is housekeeping, not beds | 6.00 h | 5.85 h | 2.5% |

The last row is the important one. If the binding constraint is housekeeping or transport, adding beds does not reduce boarding much, and the cost per avoided hour rises by an order of magnitude. The model must test the bottleneck assumption, not only the capacity parameter.

### Part F: who boards longest

The average boarding time hides a distribution. A sample of 540 boarded patients in a month shows:

| Group | Patients | Average boarding | Share of boarding hours |
|---|---:|---:|---:|
| Age 65 and older | 210 | 7.1 h | 44.3% |
| Limited English proficiency | 48 | 7.8 h | 11.1% |
| Medicaid | 160 | 6.9 h | 32.7% |
| All boarded patients | 540 | 6.0 h | 100.0% |

The groups overlap, so the shares do not sum. Older patients and patients with limited English proficiency board longer, which means a programme that reduces the average may not reduce the disparity. The equity analysis should report boarding by subgroup and ask whether the intervention reaches the patients who wait longest. A discharge-timing programme that frees beds in the morning may help the patients admitted in the afternoon; a transport programme may help the patients waiting for a bed after assignment.

### Part G: the decision

The discharge-timing programme is the better buy on boarding cost, at $13.39 per avoided hour against $38.66 for the bed expansion. It is also reversible and can be piloted in one unit. The bed expansion is a capital decision with a multi-year commitment and a negative boarding-cost return; it should be evaluated on access, revenue, and clinical grounds separately.

The recommendation is to pilot the discharge programme for 90 days in the medical unit, with a pre-specified target of 45 avoided bed-hours per day, a comparison unit, and a stop rule if boarding does not fall by at least 10 percent. The pilot should measure departure time, not order time, and should report boarding by subgroup.

A note on the value per avoided boarding hour: $17.00 is the ED nursing cost of holding one patient for one hour at a 1:4 ratio. It excludes physician time, space, equipment, and the downstream cost of LWBS. If the hospital values an avoided boarding hour at $40 because of those additional costs, the discharge programme's net rises to $437,000 and the bed expansion's net becomes +$32,600, because 81.0 x $40 x 365 = $1,182,600. The valuation is the most consequential assumption in the model, and it should be stated and tested.

That sensitivity is why the valuation should be agreed with finance before the scenario table is presented, not after the recommendation is challenged.

| Value per bed-hour | Discharge net | Bed expansion net |
|---|---:|---:|
| $17 | $59,225 | -$644,292 |
| $40 | $437,000 | +$32,600 |
| $60 | $765,500 | +$623,900 |

## 7. Code, query, or calculation

The Python script below reproduces the capacity arithmetic and the two scenarios from Section 6. It uses only the standard library and states every assumption in a variable.

Assumptions:

- 300 staffed beds, 264 occupied.
- 58 admissions per day, 54 of them from the ED.
- Boarding is measured from the admission decision to bed assignment, 6.0 hours on average.
- ED nursing cost is $68 per hour at a 1:4 nurse-to-patient ratio.
- The discharge programme moves 15 discharges three hours earlier for $220,000, with two sensitivity variants.
- The bed expansion adds 12 staffed beds for $1,150,000 per year, and the wait reduction uses a single-server queue ratio.

```python
beds = 300
occupied = 264
admits = 58
ed_admits = 54
boarding_h = 6.0
nurse_cost = 68.0
ratio = 4.0
days = 365

util = occupied / beds
alos = occupied / admits
bed_hours = ed_admits * boarding_h
concurrent = bed_hours / 24
cost_per_bed_hour = nurse_cost / ratio
annual_cost = bed_hours * cost_per_bed_hour * days

print(f"occupancy={util:.3f} alos={alos:.2f} "
   f"boarding_bed_hours={bed_hours:.0f} concurrent={concurrent:.1f} "
   f"cost_per_bed_hour={cost_per_bed_hour:.2f} "
   f"annual_boarding_cost={annual_cost:,.0f}")

def discharge_scenario(moved, hours_earlier, program_cost):
  avoided = moved * hours_earlier
  new_hours = bed_hours - avoided
  annual_value = avoided * cost_per_bed_hour * days
  net = annual_value - program_cost
  cost_per_avoided = program_cost / (avoided * days)
  breakeven = program_cost / cost_per_bed_hour / days
  return (avoided, new_hours / ed_admits, annual_value, net,
      cost_per_avoided, breakeven)

for moved, cost in ((15, 220_000), (15, 300_000), (25, 220_000)):
  avoided, avg_b, value, net, cpa, be = discharge_scenario(moved, 3, cost)
  print(f"discharge moved={moved} cost={cost:,} avoided/day={avoided:.0f} "
     f"avg_boarding={avg_b:.2f} value={value:,.0f} net={net:,.0f} "
     f"cost_per_avoided={cpa:.2f} breakeven_per_day={be:.1f}")

new_beds = 312
new_util = occupied / new_beds
base_ratio = util / (1 - util)
new_ratio = new_util / (1 - new_util)
wait_reduction = 1 - new_ratio / base_ratio
new_boarding = boarding_h * (1 - wait_reduction)
avoided = ed_admits * (boarding_h - new_boarding)
value = avoided * cost_per_bed_hour * days
capital_cost = 1_150_000
print(f"beds new_util={new_util:.3f} wait_reduction={wait_reduction:.3f} "
   f"new_boarding={new_boarding:.2f} avoided/day={avoided:.1f} "
   f"value={value:,.0f} net={value - capital_cost:,.0f} "
   f"cost_per_avoided={capital_cost / (avoided * days):.2f}")
```

Expected output:

```text
occupancy=0.880 alos=4.55 boarding_bed_hours=324 concurrent=13.5 cost_per_bed_hour=17.00 annual_boarding_cost=2,010,420
discharge moved=15 cost=220,000 avoided/day=45 avg_boarding=5.17 value=279,225 net=59,225 cost_per_avoided=13.39 breakeven_per_day=35.5
discharge moved=15 cost=300,000 avoided/day=45 avg_boarding=5.17 value=279,225 net=-20,775 cost_per_avoided=18.26 breakeven_per_day=48.3
discharge moved=25 cost=220,000 avoided/day=75 avg_boarding=4.61 value=465,375 net=245,375 cost_per_avoided=8.04 breakeven_per_day=35.5
beds new_util=0.846 wait_reduction=0.250 new_boarding=4.50 avoided/day=81.0 value=502,605 net=-647,395 cost_per_avoided=38.90
```

Read the output in three passes. The first line establishes the baseline: 88.0 percent occupancy, 4.55-day implied ALOS, 324 boarding bed-hours per day, 13.5 concurrent boarders, and $2.01 million of annual ED nursing cost. The discharge lines show the base case returning $59,225, the higher-cost variant losing $20,775, and the 25-move variant returning $245,375. The bed line shows a 25.0 percent wait reduction, 81 avoided bed-hours per day, and a net loss of $647,395.

> **Note:** The bed-expansion wait reduction uses the single-server queue ratio `u/(1-u)`. A real inpatient unit has many servers, and the reduction would be smaller. The calculation is a direction and magnitude check, not a capital-planning result. A capital decision needs a discrete-event simulation calibrated to the unit's arrival and discharge distributions.

### Deterministic versus stochastic capacity models

The script is deterministic: it uses averages and a queue approximation. A stochastic model would draw arrivals and service times from distributions and simulate many days. The difference matters at high utilisation, where the average wait is stable but the tail is not.

A simple stochastic check is to add variability to the daily admissions and recompute the boarding hours. If admissions have a standard deviation of 8 per day and discharges have a standard deviation of 7, the peak day has about 74 admissions and 51 discharges, and the bed gap is larger than the average model suggests. The deterministic model understates the peak by construction.

| Model | Input | Output | Use |
|---|---|---|---|
| Deterministic | Averages | Mean boarding | Screening |
| Queue approximation | Utilisation and service time | Mean wait | Capacity comparison |
| Discrete-event simulation | Distributions and resources | Wait distribution | Capital decision |
| System dynamics | Stocks and flows | Aggregate behaviour | Strategic planning |

The model should be as complex as the decision requires. A unit-level discharge pilot can use the deterministic model; a $50 million bed tower cannot.

### PostgreSQL dialect: hourly census

The following block is **PostgreSQL** syntax. It computes the hourly census and the number of boarders from an admission, discharge, and bed-assignment table.

```sql
-- PostgreSQL 15+
WITH hours AS (
 SELECT generate_series(
      TIMESTAMP '2025-01-01 00:00',
      TIMESTAMP '2025-12-31 23:00',
      INTERVAL '1 hour') AS hour_start
), events AS (
 SELECT bed_id, admitted_at, discharged_at FROM inpatient_stays
 UNION ALL
 SELECT bed_id, boarding_start, boarding_end FROM boarding_stays
)
SELECT h.hour_start,
    COUNT(*) FILTER (WHERE e.admitted_at <= h.hour_start
             AND e.discharged_at > h.hour_start) AS occupied,
    COUNT(*) FILTER (WHERE e.boarding_start <= h.hour_start
             AND e.boarding_end > h.hour_start) AS boarders
FROM hours h
LEFT JOIN events e
 ON e.admitted_at < h.hour_start + INTERVAL '1 hour'
 AND e.discharged_at > h.hour_start
GROUP BY h.hour_start
ORDER BY h.hour_start;
```

The query counts a stay in every hour it overlaps. The `LEFT JOIN` preserves hours with no events, which is necessary for a complete series. In SQLite, the same logic uses a recursive date series and a join on the interval; the arithmetic is identical.

The hourly census is the most useful single table for a capacity review. It shows the morning dip, the afternoon admission surge, and the evening boarding peak. A daily average collapses all three into one number and hides the period when the constraint binds.

## 8. Data, standards, and regulatory context

Capacity and flow are governed by a mix of legal obligations, reporting requirements, and professional conventions. The legal obligations are binding; the benchmarks are not.

### Regulatory and reporting standards

| Standard or requirement | What it governs | Binding status | What it does not constrain |
|---|---|---|---|
| EMTALA | Medical screening and stabilization for ED patients | Binding federal law | Boarding time or bed capacity |
| CMS Conditions of Participation | Hospital licensure and patient rights | Binding for Medicare participation | Specific staffing ratios |
| CMS weekly bed reporting | Staffed bed capacity and occupancy during respiratory season | Binding from 1 November 2024 | Real-time bed availability |
| NHSN Hospital Capacity | Bed and occupancy data collection | Programme requirement | Local flow metrics |
| State licensure | Licensed beds and unit designation | Binding state law | Staffed beds |
| Joint Commission standards | Patient flow and crowding requirements | Accreditation condition | A specific queueing method |
| CMS ED measures | Timely and effective care in the ED | Reporting programme | Boarding definition |
| NEDOCS | ED overcrowding score | A validated score, not a regulation | Local capacity |

EMTALA requires a medical screening examination and stabilization for anyone who presents to the ED, regardless of ability to pay. It does not require a bed within a specific time, but a patient who is admitted and boarded remains the hospital's responsibility. A capacity model that assumes patients can be diverted does not apply to the EMTALA population.

The CMS Conditions of Participation require a governing body, a qualified medical staff, nursing services, and patient rights. They do not specify nurse-to-patient ratios, although some states do. A capacity model that assumes a fixed ratio must state the ratio and its source.

The FY2025 IPPS final rule added a Condition of Participation requiring hospitals to report acute respiratory illness data, including staffed bed capacity and occupancy, on a weekly basis beginning 1 November 2024. The data flow through the NHSN Hospital Capacity module. The requirement is a reporting obligation, not a real-time capacity measure; the weekly snapshot cannot be used to manage a daily bed huddle.

### Benchmarks and measurement conventions

MedPAC reports a national hospital occupancy rate of about 69 percent in 2023, with about 674,000 inpatient beds and wide variation: the 5th percentile is about 12 percent and the 95th percentile about 89 percent. The national average is not a target. A hospital at 69 percent average occupancy can still have a peak problem, and a hospital at 89 percent average occupancy is in the top 5 percent of the distribution and likely has chronic boarding.

Boarding definitions vary. The most common are arrival-to-bed, decision-to-bed, and request-to-bed. The CMS ED measures use different windows for different measures. A local dashboard should state its definition and keep it stable; a change in definition creates a step change that looks like a flow improvement.

Bed counts also vary. Licensed beds are the regulatory maximum; staffed beds are the beds with available staff; available beds are staffed beds not occupied or blocked. A dashboard that reports one number without the other two is ambiguous. The minimum reporting set is licensed, staffed, occupied, and available, by unit and by hour.


### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| EMTALA screening | Yes | Federal law | None |
| Weekly bed reporting | Yes, during the reporting period | CMS CoP | Data submission process |
| Licensed bed count | Yes | State licensure | None |
| Staffed bed count | No | Local staffing | Roster and ratio |
| Boarding definition | No | Local | Start and end timestamps |
| Occupancy target | No | Local | Service level and case mix |
| Equity stratification | Programme-specific | Accreditation and payer | Subgroups |

### What the standards do not settle

1. **The boarding definition.** No federal standard defines boarding for operational management. The local definition governs.
2. **The capacity target.** No standard says what occupancy is safe. The service level and the case mix determine it.
3. **The bottleneck.** No standard identifies the constraint. The flow model does.
4. **The intervention.** No standard says whether to add beds, change discharge timing, or add a discharge lounge.
5. **The equity metric.** No standard requires boarding to be reported by subgroup, although accreditation and equity programmes encourage it.

> **Note:** Regulatory citations are dated to 25 October 2026. The CMS weekly reporting requirement and the ED measures change with annual rules. Verify the current requirement before building a compliance report.

## 9. Industry practice and case evidence

### Documented trends

Janke et al., using national data on 46.2 million hospitalizations from 2017 to 2024, reported that ED boarding became more common over the period. At the January 2022 peak, 40.1 percent of patients boarded more than four hours and 6.3 percent boarded more than 24 hours. The study is a description of the trend, not a causal analysis; it establishes that boarding is a national operational problem, not a local anomaly.

Sangal et al. analysed 148.6 million ED encounters from 2017 to June 2024 and reported that access failures rose from 18.5 to 28.7 percent, boarding over four hours rose from 2.6 to 6.2 percent, and ED length of stay over eight hours rose from 7.8 to 13.9 percent. Per-capita ED use and admissions were flat, which points to capacity and inpatient length of stay rather than demand growth. The finding is an inference from the trends, not a decomposition of causes.

Berlyand et al. studied 373,388 visits across three EDs from October 2022 to June 2024 and introduced a "functional bed capacity" metric that subtracts beds occupied by boarders from staffed capacity. Low functional capacity was associated with a 1.59 percentage point absolute increase in left-without-being-seen, a 26.5 percent relative increase. The metric is a useful operational definition: it measures the capacity actually available to new patients, not the licensed or staffed count.

Howlett et al., in a UK preprint, reported that each additional four hours of boarding was associated with 13 additional hours of inpatient length of stay and 6 percent higher odds of 30-day mortality, and that medical boarders accounted for 82 percent of total boarding time. The study is a preprint and should be labelled as such, but it illustrates the clinical consequence of the operational metric.

### Documented intervention evidence

Wiler et al. modelled an urban ED with about 87,700 annual visits using a queueing model and found that a 10 percent increase in hourly arrivals raised LWBS from about 4 to 10.7 percent, while a 10 percent reduction in boarding reduced LWBS by about 0.8 percentage points and a 50 percent reduction reduced it to about 1.5 percent. The model supports the claim that boarding is a major lever on ED access, but it is a model, not a trial.

The discharge-before-noon literature is mixed. Artenstein et al. reported that a sustained programme with more than 50 percent of discharge orders before noon reduced inpatient length of stay by 0.30 days, reduced ED boarding hours per patient by about 2.1 hours, and reduced walkout rates by about 32 percent. Wertheimer et al. reported that discharge-before-noon rose from 11 to 38 percent, average discharge time moved 1 hour 31 minutes earlier, and observed-to-expected length of stay improved from 1.06 to 0.96. Kausar et al. found that discharge-before-noon improved from 5.0 to 11.4 percent and discharge times moved earlier, but the change was not independently associated with length of stay or readmissions, and median ED boarding increased by 41 minutes. Hodges et al. reported longer order-to-discharge times, increased boarding, and a 21.9 percent higher length-of-stay ratio after a "discharge by 11 a.m." programme.

The mixed results are the lesson. A discharge-timing programme changes when the order is written, not necessarily when the bed is free. If transport, pharmacy, or the discharge lounge is the bottleneck, moving the order earlier can create a morning queue without freeing a bed. The evaluation must measure departure time and bed availability, not only order time.


### Evidence table

| Study | Setting | Design | Reported finding | What it supports |
|---|---|---|---|---|
| Janke et al., 2025 | 46.2M hospitalizations | Repeated cross-section | 40.1% boarded >4h at Jan 2022 peak | Boarding is a national trend |
| Sangal et al., 2025 | 148.6M ED encounters | Repeated cross-section | Access failures 18.5% to 28.7% | Capacity, not demand, is implicated |
| Berlyand et al., 2025 | 373,388 visits, 3 EDs | Observational | Low functional capacity +1.59 pp LWBS | Functional capacity predicts access failure |
| Artenstein et al., 2017 | Single system | Pre-post | LOS -0.30 d; boarding -2.1 h | Discharge timing can work with fidelity |
| Kausar et al., 2023 | Single hospital | Pre-post | Boarding +41 min | Order time is not departure time |

### Inference

> **Inference:** The national boarding trend and the single-site intervention results do not establish that a local discharge programme will work. The effect depends on the local bottleneck, the fidelity of the programme, and the baseline boarding time. A local pilot with a comparison unit is the minimum evidence for a rollout.

### What mature organisations do

They measure functional capacity, not licensed capacity. The daily bed huddle uses staffed beds minus blocked and boarding beds, and it reports the number by unit and by hour. The metric is uncomfortable because it shows less capacity than the licensed count, which is the point.

They separate the discharge order from the departure. The operational target is the time the bed is clean and available, not the time the order is written. A programme that reports only order time will look successful while the bed remains occupied.

They evaluate flow interventions with a comparison unit and a pre-specified metric. The comparison unit is usually another medical unit or a similar hospital in the system. The metric is boarding hours per admitted patient, not the share of discharges before noon. The evaluation runs for at least 90 days to cover seasonality.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Occupancy on licensed beds | The hospital looks empty while the ED boards | Licensed beds exceed staffed beds | Capacity is overstated | Report staffed and functional capacity |
| Annual average hides the peak | Monthly occupancy is stable but Monday is full | Demand and discharge timing vary | Peak is under-provided | Model by hour and day |
| Boarding measured from arrival | Boarding time falls when ED workup improves | Arrival includes evaluation before the decision | Wrong target, overstated benefit | Use decision-to-bed |
| Discharge order treated as departure | Order time moves earlier, bed stays occupied | Transport, pharmacy, or cleaning is the bottleneck | No bed is freed | Measure bed-ready time |
| Bottleneck assumed to be beds | Adding beds does not reduce boarding | Housekeeping or imaging limits flow | Capital is wasted | Run the marginal-effect test |
| Ignoring variability | Model matches the mean, misses the tail | Averages smooth the peak | Service level is not met | Use distributions or simulation |
| Observation excluded from census | Occupancy is understated | Observation patients occupy beds | Capacity is overstated | Reconcile observation census |
| Transfer delay ignored | A bed is "available" but the patient is in the ED | Assignment-to-transfer time is long | Boarding persists | Measure transfer separately |
| Capacity change mid-period | A step change in occupancy | Unit closure or bed conversion | Trend is misread | Maintain a capacity calendar |
| Demand shift ignored | Arrivals change after a closure or new service | The model uses last year's demand | Forecast is biased | Monitor weekly arrivals |
| Intervention evaluated pre-post | The improvement coincides with a regional trend | Secular trend and regression to the mean | Effect is overstated | Comparison unit or ITS |
| Equity ignored | Average boarding falls, disparity persists | The intervention reaches the wrong patients | Gap widens | Report boarding by subgroup |
| Discharge lounge becomes a bottleneck | Patients wait in the lounge for transport | The constraint moves downstream | No net flow gain | Measure lounge-to-departure time |
| Staffing ratio changed without updating capacity | Staffed beds are misstated | Roster and bed count diverge | Model and reality diverge | Monthly reconciliation |
| Dashboard without an owner | Nobody acts on the bed numbers | The product is a report | No change | Assign a daily huddle owner |

Three failures deserve emphasis.

**The licensed-bed illusion.** A hospital with 320 licensed and 300 staffed beds reports 82.5 percent occupancy when 264 are occupied, but the operational occupancy is 88 percent. The 20-bed difference is not a rounding error; it is the difference between "we have room" and "we are boarding". The mitigation is to make staffed and functional capacity the primary metric and to report licensed beds only as context.

**The discharge-order proxy.** A programme that increases the share of orders written before noon can report success while the beds are still occupied at 3 p.m. The order is a necessary but not sufficient condition for a free bed. The operational metric is the time the bed is clean and ready, and the programme should be evaluated on that.

**The bottleneck that moves.** A discharge lounge can reduce boarding in the ED by moving patients out of the stretcher, but if transport is the constraint, the patient waits in the lounge instead. The total time to departure may not change, and the ED may look better while the patient experience is the same. The evaluation must measure the end-to-end flow, not the intermediate queue.


### Pre-mortem for a capacity decision

1. Is occupancy computed on staffed, functional, or licensed beds?
2. Is boarding measured from arrival, decision, or bed request?
3. What is the bottleneck, and was it tested by adding one unit?
4. Does the model reproduce the peak, not only the mean?
5. What is the value per avoided boarding hour, and who agreed it?
6. Is there a comparison unit and a stop rule?

A capacity proposal that cannot answer all six is not ready for capital or operating review.
> **Note:** A falling boarding time is not proof that access improved. If the hospital diverts ambulances or if patients leave without being seen, the boarding time can fall because the patients who would have boarded are no longer counted. Pair the boarding metric with LWBS, diversion, and total ED arrivals.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Staffed beds are the capacity constraint | Occupancy depends on it | Bed closures, staffing changes | Reconcile to roster monthly |
| Boarding is decision-to-bed | The metric must match the delay | Definition drift | Audit timestamps |
| Demand is stationary within the model period | Averages and queue formulas require it | Seasonality, new service, closure | Plot arrivals by week |
| The queue approximation is adequate | It estimates the effect of capacity | High variability, priority queues | Compare with simulation |
| The value per bed-hour is the nursing cost | The business case depends on it | Space and downstream costs | Sensitivity analysis |
| The intervention moves departure time | The benefit depends on it | Order time moves but departure does not | Pilot measurement |
| The comparison unit is similar | The effect estimate depends on it | Different case mix or staffing | Compare baseline trends |
| No concurrent intervention | The effect is attributable | Another flow programme launches | Maintain an intervention calendar |
| The period is long enough | Seasonality is covered | A 30-day pilot in flu season | Use 90 days or more |
| Subgroup effects are measured | Equity depends on it | Small samples | Pool periods or sites |


### Assumption owners

| Assumption | Owner | Review cadence |
|---|---|---|
| Staffed bed count | Nursing operations | Monthly |
| Boarding definition | ED and analytics | Quarterly |
| Demand forecast | Analytics | Monthly |
| Bottleneck assessment | Operations engineering | Each major change |
| Value per bed-hour | Finance | Annual |
| Equity metrics | Health equity office | Quarterly |

The limits follow from the assumptions. A capacity model is a simplification of a queueing network, and its output is a scenario comparison, not a forecast. It can tell you which intervention is likely to reduce boarding more per dollar, and it can tell you the magnitude within a stated tolerance. It cannot tell you whether the patients who board are the ones who suffer most, unless the model includes equity and clinical outcomes.

A second limit is that the model is calibrated on a period with a particular case mix and discharge process. A change in the post-acute network, the surgical schedule, or the staffing model can shift the bottleneck. The model must be revalidated after a material change, even if the bed count is unchanged.

A third limit is that the value of avoided boarding is contested. The nursing cost per bed-hour is measurable; the value of reduced LWBS, improved patient experience, and avoided harm is not. A business case that uses only the nursing cost will understate the benefit; one that includes a large clinical value without evidence will overstate it. State the valuation and test it.

## 12. Exercises

### Exercise 1. Map the flow (recall and application)

Draw an input-throughput-output map for an admitted ED patient from arrival to inpatient bed. Label at least six queues, the resource for each, and two feedback loops. Identify the queue you would investigate first and why.

**Deliverable:** a one-page process map with queues, resources, and feedback loops.

**Acceptance criteria:** at least six queues; each has a resource; the feedback loops are explicit; the first investigation is justified by a hypothesis, not by visibility.

**Time:** 45 minutes.

### Exercise 2. Compute occupancy and boarding (application)

A unit has 250 staffed beds and 220 occupied beds. There are 50 admissions per day and 46 ED admissions. Average decision-to-bed boarding is 5.0 hours. Nursing cost is $72 per hour at a 1:4 ratio. Compute occupancy, implied ALOS, boarding bed-hours, average concurrent boarders, cost per bed-hour, and annual boarding nursing cost.

**Deliverable:** a calculation table with units.

**Acceptance criteria:** occupancy uses staffed beds; ALOS uses admissions; bed-hours use ED admissions only; the annual cost uses 365 days.

**Time:** 45 minutes.

### Exercise 3. Size a discharge-timing intervention (application)

Using the unit in Exercise 2, a programme moves 12 discharges three hours earlier. Compute avoided bed-hours per day, the new average boarding time, the annual value at $18 per bed-hour, the net result for a $180,000 programme, the cost per avoided bed-hour, and the breakeven avoided hours per day.

**Deliverable:** a calculation table and a recommendation.

**Acceptance criteria:** the new boarding time uses the same denominator; the breakeven is expressed per day; the recommendation states the condition.

**Time:** 45 minutes.

### Exercise 4. Critique a boarding metric (critique)

A dashboard reports that boarding fell from 7.5 to 6.2 hours after a programme, using arrival-to-bed time. The ED workup time also fell by 1.1 hours over the same period. Identify the measurement problem, the likely true change in decision-to-bed boarding, and the additional data you would request.

**Deliverable:** a short critique with the corrected estimate.

**Acceptance criteria:** the definition difference is identified; the corrected estimate is arithmetic; the data request is specific.

**Time:** 30 minutes.

### Exercise 5. Build a bottleneck test (application)

A model can add one unit of capacity to four resources. The marginal reductions in boarding are: beds 1.6 hours, CT 0.5 hours, housekeeping 1.1 hours, transport 0.3 hours. The costs are: beds $900,000, CT $400,000, housekeeping $120,000, transport $80,000. Compute the cost per avoided boarding hour for each and rank the options.

**Deliverable:** a table with the marginal effect, annual avoided hours, cost, and cost per avoided hour.

**Acceptance criteria:** the annual hours use 54 ED admissions and 365 days; the ranking is by cost per avoided hour; the recommendation names the bottleneck.

**Time:** 45 minutes.

### Exercise 6. Design an evaluation (design)

A hospital plans to open a discharge lounge. Design an evaluation with a comparison unit, a primary metric, a pre-period, a post-period, and a stop rule. State the confounders and how you would address them.

**Deliverable:** a two-page evaluation plan.

**Acceptance criteria:** the comparison unit is justified; the primary metric is bed-ready time or boarding, not lounge use; the pre-period covers seasonality; the stop rule is explicit.

**Time:** 60 minutes.

### Exercise 7. Build a business case (application)

A programme moves 10 discharges per day three hours earlier at a cost of $150,000 per year. The value of an avoided boarding hour is $20. Compute avoided bed-hours per day, annual value, net result, cost per avoided bed-hour, and breakeven avoided hours per day. State whether the programme should be funded.

**Deliverable:** a business case with arithmetic and a recommendation.

**Acceptance criteria:** the value and cost bases are stated; the breakeven is per day; the recommendation is conditional on the value assumption.

**Time:** 45 minutes.

### Exercise 8. Design an equity analysis (design)

A hospital's average boarding time is 6.0 hours, but patients over 65 board 7.1 hours and patients with limited English proficiency board 7.8 hours. Design an equity analysis: the subgroups, the metrics, the comparison, the actions, and the review body.

**Deliverable:** a two-page plan.

**Acceptance criteria:** at least four subgroups; metrics include boarding, LWBS, and reach of the intervention; actions distinguish flow changes from language services; the review body is named.

**Time:** 60 minutes.

### Assessment notes

Exercises 1 to 4 can be completed individually and checked against Section 13. Exercises 5 to 8 are assessed with the rubric. Exercise 6 benefits from a real bed-management extract; if none is available, use the timestamps in Section 5. Add 20 minutes to Exercise 5 for learners who need to build the cost table from scratch.

## 13. Self-check answers

### Exercise 1

A strong map has these queues: arrival to triage, triage to provider, provider to workup complete, workup complete to admission decision, decision to bed request, bed request to bed assignment, assignment to physical transfer, and unit arrival to discharge. The resources are triage nurses, ED physicians and advanced practice providers, imaging and laboratory, bed management, housekeeping, transport, and inpatient nursing. The two feedback loops are: a full inpatient unit blocks ED stretchers, which slows ED throughput; and a long ED workup delays the admission decision, which delays the bed request. The first queue to investigate is the one with the largest accumulation relative to its service rate, not the most visible one.

### Exercise 2

```text
Occupancy = 220 / 250 = 88.0%
Implied ALOS = 220 / 50 = 4.40 days
Boarding bed-hours = 46 x 5.0 = 230 per day
Average concurrent boarders = 230 / 24 = 9.58
Cost per bed-hour = $72 / 4 = $18.00
Annual boarding nursing cost = 230 x $18.00 x 365 = $1,511,100
```

The denominator for boarding is ED admissions (46), not all admissions (50). The four direct admissions do not board in the ED. Using all 50 would overstate boarding bed-hours by 20.

### Exercise 3

```text
Avoided bed-hours = 12 x 3 = 36 per day
New boarding bed-hours = 230 - 36 = 194
New average boarding = 194 / 46 = 4.22 hours
Annual value = 36 x $18.00 x 365 = $236,520
Net = $236,520 - $180,000 = $56,520
Cost per avoided bed-hour = $180,000 / (36 x 365) = $13.70
Breakeven avoided hours per day = $180,000 / $18.00 / 365 = 27.4
```

The programme is viable at 36 avoided bed-hours per day against a breakeven of 27.4. The margin is 24 percent, so a shortfall of about 9 bed-hours per day would eliminate it.

### Exercise 4

The dashboard uses arrival-to-bed, which includes ED evaluation. If ED workup time fell by 1.1 hours, the arrival-to-bed improvement of 1.3 hours (7.5 to 6.2) overstates the boarding improvement by 1.1 hours. The corrected change in decision-to-bed boarding is about 0.2 hours, not 1.3. The additional data needed are the admission decision timestamp, the bed request timestamp, and the bed assignment timestamp, so that the intervals can be separated.

### Exercise 5

```text
Beds:     1.6 x 54 x 365 = 31,536 hours; $900,000 / 31,536 = $28.54
CT:      0.5 x 54 x 365 = 9,855 hours; $400,000 / 9,855 = $40.59
Housekeeping: 1.1 x 54 x 365 = 21,681 hours; $120,000 / 21,681 = $5.53
Transport:   0.3 x 54 x 365 = 5,913 hours; $80,000 / 5,913 = $13.53
```

Ranking by cost per avoided boarding hour: housekeeping ($5.53), transport ($13.53), beds ($28.54), CT ($40.59). Housekeeping is the best buy even though it is not the most visible queue. The ranking assumes the marginal effects are independent; in reality, adding housekeeping may shift the bottleneck to transport.

### Exercises 6 to 8 rubric

Exercise 6 is assessed on the comparison unit, the primary metric, the pre-period, the post-period, the stop rule, and the confounders. A strong plan uses bed-ready time or boarding hours per admitted patient, covers at least 12 months before and 3 months after, and names seasonality, concurrent interventions, and case-mix change as confounders.

Exercise 7 is assessed on the arithmetic and the conditional recommendation. A strong answer reports $219,000 of annual value, a $69,000 net, $13.70 per avoided bed-hour, and a breakeven of 20.5 avoided bed-hours per day, and states that the programme is viable only if the $20 value per bed-hour is accepted.

Exercise 8 is assessed on the subgroups, metrics, comparison, actions, and review body. A strong plan includes age, language, race and ethnicity, payer, and disability; measures boarding, LWBS, and intervention reach; and distinguishes flow changes (discharge timing, transport) from language services (interpreters, translated instructions).


A note on precision: the answers round to two decimals. In a production model, carry full precision through the bed-hour and cost calculations and round only in the final table. At 54 admissions per day, a 0.01-hour rounding in boarding changes the annual cost by about $335.

## 14. Interview questions

**1. How do you know a hospital needs more beds?**
A strong answer checks staffed and functional capacity, peak occupancy, boarding, LWBS, and the bottleneck test before concluding that beds are the constraint. A weak answer points to high average occupancy.

**2. What is Little's law and how do you use it?**
A strong answer states L = lambda x W, gives the census-admissions-ALOS example, and uses it to check consistency. A weak answer recites the formula without application.

**3. Why does waiting rise so fast near full occupancy?**
A strong answer explains the queueing relationship and the role of variability, and notes that the last few points of occupancy cost the most wait. A weak answer says "because it is busy".

**4. What is the difference between arrival-to-bed and decision-to-bed?**
A strong answer explains that arrival includes ED evaluation and that decision-to-bed isolates the inpatient capacity delay; the choice depends on the decision. A weak answer treats them as interchangeable.

**5. How would you evaluate a discharge-before-noon programme?**
A strong answer measures departure and bed-ready time, uses a comparison unit, covers seasonality, and checks readmissions and equity. A weak answer compares the share of orders before noon.

**6. What is functional bed capacity?**
A strong answer defines it as staffed beds minus beds occupied by boarders and blocked beds, and explains why it predicts LWBS better than licensed occupancy. A weak answer confuses it with available beds.

**7. How do you find the bottleneck?**
A strong answer uses queue accumulation, utilisation, and the marginal effect of adding one unit of capacity, and notes that the bottleneck can be a process rather than a resource. A weak answer assumes it is the ED.

**8. What are the harms of adding beds?**
A strong answer names fixed cost, stranded capacity if demand falls, staffing dilution, and the risk of inducing demand. A weak answer says beds are always beneficial.

**9. How do you handle seasonality in a capacity model?**
A strong answer models by month or week, uses at least a year of data, and validates on a different season. A weak answer uses an annual average.

**10. How would you measure equity in boarding?**
A strong answer reports boarding and LWBS by age, language, race and ethnicity, payer, and disability, and checks whether the intervention reaches the groups that wait longest. A weak answer reports the overall average.

**11. What would you do if the model fits the average but not the peak?**
A strong answer investigates the arrival and discharge distributions, the capacity calendar, and the bottleneck, and refuses to use the model for peak decisions until it is recalibrated. A weak answer adjusts the mean.

**12. How do you value an avoided boarding hour?**
A strong answer uses the marginal nursing and space cost, states the basis, and tests a range; it separates the clinical and access value from the financial value. A weak answer uses the full ED cost per visit.


**13. What is the difference between capacity and throughput?**
A strong answer says capacity is the maximum service rate and throughput is the actual flow, and that throughput is limited by the bottleneck, not by the sum of resources. A weak answer uses the terms interchangeably.

**16. How do you know the model is good enough for a capital decision?**
A strong answer requires peak validation, a bottleneck test, a scenario table, and a stated tolerance; it refuses a capital decision from a mean-only model. A weak answer relies on the model's average fit.

**15. How would you explain queueing to a hospital board?**
A strong answer uses the occupancy-to-wait table, states that the last few points of occupancy cost the most wait, and connects it to the local boarding data. A weak answer uses equations without an operational translation.

**14. Why can adding beds fail to reduce boarding?**
A strong answer explains that the bottleneck may be housekeeping, transport, imaging, or post-acute placement, and that the added beds do not relieve the binding constraint. A weak answer assumes beds are always the constraint.

## 15. Key terms

- **Capacity:** the maximum service rate of a resource in a period.
- **Demand:** the arrival rate of patients requiring service.
- **Utilisation:** demand divided by capacity, or occupied divided by available.
- **Queue:** patients waiting for a resource.
- **Wait time:** time from arrival to service start.
- **Length of stay:** time from admission to discharge.
- **ED LOS:** time from ED arrival to ED departure.
- **Boarding:** time an admitted patient waits in the ED for an inpatient bed.
- **Bed request:** the time an inpatient bed is requested.
- **Admission decision:** the time a clinician decides to admit.
- **LWBS:** left without being seen.
- **Diversion:** temporary closure of the ED to ambulance arrivals.
- **Occupancy:** occupied bed-days divided by staffed bed-days.
- **Staffed bed:** a bed that can be occupied with available staff.
- **Licensed bed:** a bed permitted by licensure.
- **Functional bed capacity:** staffed beds minus beds occupied by boarders and blocked beds.
- **Census:** patients in beds at a point in time.
- **Turnover:** time to clean and prepare a bed.
- **Discharge before noon:** proportion of discharges before 12:00.
- **Discharge order time:** time the discharge order is written.
- **Departure time:** time the patient leaves the unit.
- **Discharge barrier:** a non-clinical reason for a delayed departure.
- **Post-acute placement:** SNF, rehab, or home health arrangement.
- **Observation stay:** a short stay under observation status.
- **Throughput:** rate of patient flow through a process.
- **Bottleneck:** the resource that limits system throughput.
- **Cycle time:** time to complete one patient's flow.
- **Service level:** probability that wait is below a threshold.
- **Arrival rate:** patients arriving per unit time.
- **Service rate:** patients served per unit time.
- **M/M/c:** a queue with Poisson arrivals, exponential service, and c servers.
- **Discrete-event simulation:** a simulation of events and resources over time.
- **Forecast:** a prediction of future demand.
- **Seasonality:** regular variation by time of year.
- **Surge:** a temporary increase in demand.
- **Bed huddle:** a daily meeting to allocate beds.
- **Discharge lounge:** a waiting area for discharged patients.
- **Hospital at home:** acute care delivered at home.
- **Access block:** inability to admit because no bed is available.
- **NEDOCS:** a numerical ED overcrowding score.
- **Bed-ready time:** time a bed is clean and available.
- **Discharge order:** the clinical order authorising departure.
- **Bed assignment:** the allocation of a specific bed.
- **Transfer time:** time from assignment to unit arrival.
- **Bed management:** the function that assigns beds.
- **Housekeeping turnaround:** time to clean a room.
- **Transport:** the function that moves patients.
- **Post-acute liaison:** the role that arranges SNF or rehab.



- **Queue ratio:** utilisation divided by one minus utilisation.
- **Bed blocking:** a bed held by a patient who cannot leave.
- **Surge bed:** a temporarily opened bed.
- **Staffing ratio:** patients per nurse.
- **Capacity calendar:** the record of capacity changes.
- **Discharge pattern:** the distribution of departures by hour.
- **Arrival pattern:** the distribution of arrivals by hour.
- **Throughput time:** time from arrival to departure for one patient.
- **Bed-ready time:** the time a bed is clean and available for the next patient.
- **Discharge order:** the clinical order authorizing departure.
- **Bed assignment:** the allocation of a specific bed to a patient.
- **Transfer time:** time from bed assignment to physical arrival on the unit.

## 16. Sources

All URLs were accessed on 25 October 2026.

### Boarding and access

1. Janke, A. T., Burke, R. C., Haimovich, A. et al. Hospital "boarding" of patients in the emergency department increasingly common, 2017-24. *Health Affairs*, 2025. https://doi.org/10.1377/hlthaff.2024.01513
2. Sangal, R. B., Rothenberg, C., Taylor, R. A., Venkatesh, A. K. Emergency care access based on a proposed CMS national quality measure. *JAMA Health Forum*, 2025. https://doi.org/10.1001/jamahealthforum.2025.0417
3. Berlyand, Y. et al. Reduced functional bed capacity due to inpatient boarding is associated with increased rates of left without being seen in the emergency department. *Western Journal of Emergency Medicine*, 2025. https://pubmed.ncbi.nlm.nih.gov/41380054/
4. Howlett, M., Cameron, P., Wood, S. Delay-related harm: direct and indirect impacts of boarding medical patients in the emergency department. *medRxiv* preprint, 2025. https://doi.org/10.1101/2025.02.21.25322666
5. Yousufuddin, M. et al. Inpatient boarding in the emergency departments and clinical outcomes: a propensity-matched study. *American Journal of Emergency Medicine*, 2025. https://doi.org/10.1016/j.ajem.2025.09.043

### Queueing and flow

6. Wiler, J. L. et al. An emergency department patient flow model based on queueing theory principles. *Academic Emergency Medicine*, 2013. https://onlinelibrary.wiley.com/doi/10.1111/acem.12215
7. Little, J. D. C., Graves, S. C. Chapter 5: Little's Law. https://ic.unicamp.br/~nfonseca/MO648/doc/Little%27s%20Law-Published.pdf
8. Joseph, J. W. Queuing theory and modeling emergency department resource utilization. *Emergency Medicine Clinics of North America*, 2020. https://doi.org/10.1016/j.emc.2020.04.006
9. Smalley, C. M. et al. The impact of hospital boarding on the emergency department waiting room. *JACEP Open*, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7593429/

### Discharge timing

10. Artenstein, A. W. et al. Decreasing emergency department walkout rate and boarding hours by improving inpatient length of stay. *Western Journal of Emergency Medicine*, 2017. https://pubmed.ncbi.nlm.nih.gov/29085527/
11. Wertheimer, B. et al. Discharge before noon: an achievable hospital goal. *Journal of Hospital Medicine*, 2014. https://cdn.mdedge.com/files/s3fs-public/pdfs/journals/jhm2154.pdf
12. Kausar, S. et al. Implementing and evaluating a discharge before noon initiative in a large tertiary care urban hospital. *Joint Commission Journal on Quality and Patient Safety*, 2023. https://www.jointcommissionjournal.com/article/S1553-7250(23)00212-X/abstract
13. Hodges, J. et al. Driving in the wrong direction: exploring the unintended consequences of an early discharge program. *Quality Management in Health Care*, 2024. https://doi.org/10.1097/qmh.0000000000000466

### Regulation and benchmarks

14. CMS. Hospital Inpatient Quality Reporting Program Measures. https://qualitynet.cms.gov/inpatient/iqr/measures
15. CMS. FY2025 IPPS Final Rule Fact Sheet. https://www.cms.gov/newsroom/fact-sheets/fy-2025-hospital-inpatient-prospective-payment-system-ipps-long-term-care-hospital-prospective-0
16. CDC. NHSN Healthcare Capacity Group User Dashboard Quick Reference Guide. https://www.cdc.gov/nhsn/bed-capacity/Bed-Capacity-Group-Dashboard-QRG.pdf
17. MedPAC. March 2025 Report to the Congress: Hospital Inpatient and Outpatient Services. https://www.medpac.gov/wp-content/uploads/2025/03/Mar25_Ch3_MedPAC_Report_To_Congress_SEC.pdf
18. CMS. Hospital Readmissions Reduction Program. https://qualitynet.cms.gov/inpatient/hrrp

### Methods

19. Lopez Bernal, J., Cummins, S., Gasparrini, A. Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5407170/
20. Li, L. et al. Three statistical approaches for assessment of intervention effects: a primer for practitioners. https://pmc.ncbi.nlm.nih.gov/articles/PMC7910529/

> **Note:** The Howlett et al. study is a preprint and has not completed peer review. It is cited for the direction of the association, not as a confirmed effect size.


23. Joint Commission. Patient flow and crowding standards. https://www.jointcommission.org/
24. CMS. EMTALA. https://www.cms.gov/medicare/health-safety-standards/certification-compliance/emergency-medical-treatment-labor-act
25. CMS. Conditions of Participation. https://www.cms.gov/medicare/health-safety-standards/certification-compliance

### A note on the evidence

The national studies describe trends; they do not identify the cause of a local bottleneck. The single-site studies describe programmes; they do not establish that the same programme will work elsewhere. The correct use of the evidence is to set an expectation and to justify a local pilot, not to substitute a published effect for a local measurement.


26. Institute for Healthcare Improvement. Flow and capacity resources. https://www.ihi.org/
27. NHS England. Emergency department capacity and flow guidance. https://www.england.nhs.uk/
28. American College of Emergency Physicians. Boarding and crowding resources. https://www.acep.org/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_operational-models.md`](04_operational-models.md) — no-show, length of stay, and readmission models.
- Next topic: [`06_quality-measures.md`](06_quality-measures.md) — HEDIS and CMS measure logic.
- Related topic: [`08_evaluating-operational-interventions.md`](08_evaluating-operational-interventions.md) — the designs used to evaluate flow interventions.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Diagnose ED boarding | Sections 5 and 6 | Topic 04 for the readmission model |
| Size a bed expansion | Sections 5, 6, and 9 | Topic 08 for the evaluation |
| Evaluate a discharge programme | Sections 5 and 9 | Topic 08 |
| Report capacity to the board | Sections 6 and 11 | Topic 01 for the operating model |
| Link flow to quality | Sections 8 and 9 | Topic 06 |

Capacity is the constraint that connects the clinical, financial, and operational views in this module. A bed that is occupied by a boarder is not available for a new admission, a discharge that is ordered but not completed does not free capacity, and an occupancy average that hides the peak misleads the board. The flow model is the tool that makes those connections visible.

A capacity model that cannot be explained in the daily bed huddle will not change the huddle. The final test of the topic is whether the reader can stand in front of the operations team, point to the bottleneck, and show the arithmetic for the next intervention.

A final caution: the queueing numbers are a lens, not a licence. They show why the last few beds matter more than the first few, and why a discharge programme can beat a bed tower. They cannot tell you which patient will be harmed while waiting. That judgement belongs to the clinical team, and the model's job is to give that team more time and more options.

The distinction between the three queues is the first question to ask in any capacity review.

That standard is higher than the standard for an operational pilot, and the difference should be stated in the model card.

| Decision | Required evidence |
|---|---|
| Pilot | Direction and operational feasibility |
| Capital | Peak validation and scenario cost |
