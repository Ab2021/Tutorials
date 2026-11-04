# OEE and the Loss Tree

> Module 15 — Manufacturing and Production Planning  
> Topic 09 of 11  
> Estimated study time: 4 hours  
> Prerequisites: [manufacturing master data](02_manufacturing-master-data.md), [capacity requirements](05_capacity-requirements.md), and [finite scheduling](06_finite-scheduling.md)  
> Version 2026.5

## 1. Why this topic exists

A plant can appear to have enough scheduled machine hours and still miss its production plan.
A work center can be busy for an entire shift and still deliver fewer saleable units than promised.
The difference between occupied clock time and productive output is a management problem, not just a reporting problem.

Overall equipment effectiveness (OEE) expresses fully productive time as a fraction of planned production time.
Its three familiar components are availability, performance, and quality.
The decomposition helps a cross-functional team ask whether a shortfall comes from stopped time, running below an ideal rate, or producing units that are not good first time.
The ratio is a starting point for investigation; it is not an explanation or a customer-service measure.

Consider a line whose weekly schedule assumes 900 good units per day.
If the line has historically delivered 700 good units under comparable conditions, the scheduler needs a capacity decision before accepting additional orders.
Maintenance needs the actual stop causes and repair opportunities.
Operations needs to know whether small interruptions, changeovers, or slow running dominate.
Quality needs to distinguish first-pass rejects from defects found only after shipment.
Finance needs a credible estimate of incremental contribution and project cost.
None of these decisions follows automatically from one percentage.

The denominator also carries a business choice.
A company may exclude an unpaid meal break from planned production time while another includes it.
Two identical shifts can then report different OEE even if their physical output is the same.
A scorecard comparison is useful only when the time boundary, cycle standard, counting point, and quality rule are aligned.

The fictional pump-component case in this lesson uses one product, one machine, and one shift.
It uses minutes, units, and United States dollars (USD).
Its values are illustrative, not a public company result.
A forecast is an estimate of future demand.
A production plan is an intended schedule.
A target is a desired level of performance.
A customer commitment is an authorized promise, not a consequence of a favorable OEE calculation.
An improvement estimate is neither a plan nor a commitment until its assumptions and owners have been approved.

## 2. Learning outcomes

After studying this topic, you can:

1. **Define** the time and count boundaries for an OEE calculation; exercise 1 checks your boundary choices.
2. **Calculate** availability, performance, quality, and OEE from a shift ledger; exercise 2 checks the arithmetic.
3. **Build** an additive loss tree in ideal minutes without double counting; exercise 3 tests the reconciliation.
4. **Diagnose** event and counter integrity defects before releasing a score; exercise 4 requires a remediation decision.
5. **Compare** feasible interventions using contribution, constraints, and incremental costs; exercise 5 tests the economic choice.
6. **Critique** comparisons across shifts and product mixes; exercise 6 addresses denominator and standard changes.
7. **Design** a measurement and governance process for a pilot; exercise 7 uses a practical rubric.
8. **Explain** why an OEE increase need not improve flow, customer service, or profit; exercise 8 tests the limitation.

A successful calculation is reproducible from the source records.
A successful recommendation also identifies the decision owner and the evidence required to change the operating plan.

## 3. The core mental model

Start with the time during which the resource was expected to produce.
Remove mutually exclusive losses until the remainder represents the ideal time needed to produce good first-pass output.
The same physical story can be expressed as a product of three ratios or as an additive tree in minutes.

```text
                         Planned production time
                                  |
                 +----------------+----------------+
                 |                                 |
          Availability loss                     Run time
          stops and setups                         |
                                  +----------------+----------------+
                                  |                                 |
                           Performance loss                   Net run time
                           microstops and speed                    |
                                                   +---------------+---------------+
                                                   |                               |
                                             Quality loss                 Fully productive time
                                             rejects at ideal time        good units at ideal time
```

The chart is an accounting identity once the definitions are fixed.
It does not prove that one branch caused another.
For example, a rushed setup could increase later defects, but the tree allocates the setup minutes to availability and the reject ideal minutes to quality.
Root-cause investigation follows the tree; it is not replaced by it.

For a single product with a fixed ideal cycle time:

```text
planned time = availability loss + performance loss + quality loss
               + fully productive time
availability = run time / planned time
performance  = (ideal cycle time × total units) / run time
quality      = good first-pass units / total units
OEE          = availability × performance × quality
             = ideal cycle time × good first-pass units / planned time
```

All time quantities in the additive tree are in minutes.
Ideal cycle time is in minutes per unit.
Counts are in units at a specified measurement point.
The ratios are dimensionless and may be shown as percentages.
The cancellation in the OEE equation is useful for reconciliation, but it can also conceal offsetting errors in intermediate components.
A correct final percentage does not establish that stop coding or quality classification is correct.

The ideal-time basis is important.
A rejected unit consumes actual resources, but quality loss in the standard tree is its ideal production time.
The additional time consumed while the line ran below its ideal rate is already included in performance loss.
Assigning actual time per reject to quality as well as retaining the full performance loss would double count.

The tree describes one resource and one time window.
A system may have parallel resources, buffers, assembly joins, and a moving bottleneck.
The arithmetic can be correct at one machine while the factory remains constrained elsewhere.
Always connect the measurement to the decision: which order, constraint, customer, cost, and capacity window are affected?

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Asset | Physical machine or defined production line being measured; asset ID | Combining machines with different boundaries without disclosure. |
| Reporting window | Start and end timestamps for a shift or other bucket | Assigning overnight events wholly to the day on which they started. |
| Calendar time | All elapsed time in a window, in minutes | Treating an idle weekend as a machine failure. |
| Planned production time | Calendar minutes selected for production under the stated policy | Quietly removing difficult periods to improve the score. |
| Exclusion | Time outside planned production by approved policy | Reclassifying an unplanned shortage as an exclusion after it occurs. |
| Stop | A period classified as unavailable under the measurement policy | Counting overlapping alarms as separate elapsed time. |
| Changeover | Time required to switch a resource between production conditions | Automatically excluding a scheduled setup from OEE without stating the policy. |
| Availability loss | Planned minutes assigned to stopped or setup states | Adding quality reject time to stops as well. |
| Run time | Planned production time minus availability loss, in minutes | Assuming a run-state signal proves units were made. |
| Ideal cycle time | Best demonstrated sustainable ideal time per unit under the defined product and configuration, in minutes per unit | Using a budget rate merely to make performance approach 100%. |
| Total count | Units produced at the specified counting point, including rejects | Counting pulses or cavities as units without conversion. |
| Good first-pass count | Total units meeting the specified quality criterion without rework, in units | Counting repaired units as first-pass good. |
| Reject | A unit failing the first-pass quality criterion, whether or not later recovered | Assuming every defect is permanent scrap. |
| Availability | Run time divided by planned production time | Comparing plants with different exclusion rules. |
| Net run time | Ideal cycle time multiplied by total units, in ideal minutes | Calling it elapsed running time. |
| Performance | Net run time divided by run time | Treating a value above 100% as an automatic success. |
| Quality | Good first-pass units divided by total units for a single product | Using shipped quantity when the denominator counts at the machine. |
| Fully productive time | Ideal cycle time multiplied by good first-pass units, in ideal minutes | Treating it as payroll time saved. |
| OEE | Fully productive time divided by planned production time | Using a single OEE value as proof of profitability. |
| Minor stop | Short interruption classified under performance by the stated threshold | Recording it both as a stop and as an inferred performance loss. |
| Speed loss | Difference between ideal output time and actual running time not allocated to microstops, in minutes | Assuming every unclassified minute is a known speed cause. |
| Loss tree | Reconciliation of planned time to fully productive time by mutually exclusive categories | Treating correlation or an accounting category as a verified cause. |
| Bottleneck | Resource or constraint limiting system throughput in the relevant period | Assuming the lowest-OEE machine is necessarily the bottleneck. |
| Contribution | Relevant incremental revenue less relevant incremental variable costs, in USD per additional sold unit | Multiplying extra output by selling price and calling it profit. |
| First-pass yield | Fraction meeting a specified quality criterion without rework | Equating a local machine's first-pass yield to shipped-product quality. |
| Counter reset | Change in counter value caused by restart, rollover, or correction rather than production | Treating a negative counter delta as negative production. |
| Reason code | Controlled category assigned to a state interval or loss event | Allowing a code to define its own duration by overlapping another interval. |
| Threshold | Policy boundary for classifying an interruption as an availability stop or a performance microstop | Changing the threshold between comparison periods without restatement. |

## 5. The mechanics, step by step

### Step 1 — State the decision and the unit of analysis

**Input:** the operating question, asset list, and planning horizon.

1. Name the decision: for example, whether to repair a recurring breakdown before next month's capacity plan is approved.
2. Identify the physical asset and its relevant boundary.
3. Identify the product, process stage, count point, and shift.
4. Choose the reporting period needed for the decision.
5. Specify the responsible operations, engineering, quality, planning, and finance owners.
6. Record whether the output is intended for local diagnosis, capacity planning, or an external comparison.

**Output:** a measurement charter with a decision, boundary, and accountable owner.

A machine-level OEE is not automatically a line-level OEE.
If a line has intermediate buffers, each machine can be running while a different machine prevents shipment.
If the analysis covers an entire line, define the controlling output counter and line state explicitly.
Do not average constituent machine percentages and label the result line OEE.

A single shift can be useful for learning the calculation.
A capital request should usually use a longer, representative baseline and show uncertainty.
Choose the baseline before inspecting which period produces the most favorable estimate.

### Step 2 — Establish the planned-time calendar

**Input:** shift calendar, approved breaks, planned production assignments, and downtime policy.

1. Express the shift as a half-open interval: include the start timestamp and exclude the end timestamp.
2. Subtract calendar periods that the measurement charter excludes from planned production.
3. Specify how holidays, meal breaks, no-demand idle periods, maintenance windows, and engineering trials are treated.
4. Document whether a scheduled setup remains inside planned production time.
5. Apply the same policy to the comparison periods.
6. Retain excluded intervals in the raw calendar so another denominator can be calculated transparently.

**Output:** planned production minutes and a versioned exclusion ledger.

If an approved 30-minute meal break is outside planned production in a 480-minute shift, planned time is 450 minutes.
This does not mean the break is free to the business.
It means the selected OEE measure does not assess it.
A separate calendar utilization measure can address time outside the OEE denominator.

A shortage of material is not automatically an exclusion simply because the machine cannot run.
The organization must choose and publish the policy before reviewing the results.
The analysis should retain the underlying shortage reason regardless of the scorecard treatment.

### Step 3 — Build a non-overlapping state timeline

**Input:** state events, stop reasons, shift boundaries, and planned intervals.

1. Convert timestamps to a common time basis while preserving the plant's local shift label.
2. Split events at shift and planned-time boundaries.
3. Resolve duplicate events and simultaneous alarm records.
4. Assign each instant to one authoritative production state.
5. Apply a documented priority rule when state signals conflict.
6. Separate physical state from reason attribution.
7. Calculate the duration of each resulting interval.
8. Reconcile the intervals to planned production minutes.

**Output:** disjoint planned-time state intervals with one state per instant.

A 10-minute interval with three active alarms is 10 minutes of elapsed time, not 30 minutes.
Multiple causes may be recorded for investigation, but their durations cannot each be summed into the additive tree.
Where no defensible primary cause exists, assign the interval to an unresolved reason and investigate it.
Do not create a false precision by dividing the interval arbitrarily among alarms.

An overnight shift can cross midnight or a daylight-saving transition.
Store event timestamps with timezone information or in coordinated universal time (UTC).
Calculate elapsed time using real instants, not a subtraction of ambiguous local clock labels.
If source clocks disagree, record the uncertainty and decide whether the score is fit for release.

### Step 4 — Measure availability loss

**Input:** planned-time intervals, stop classifications, and the microstop threshold.

1. Mark intervals assigned to breakdown, setup, planned maintenance inside the denominator, and other stops.
2. Apply the same rule for distinguishing a stop from a short interruption in every compared period.
3. Sum the disjoint stop durations within planned time.
4. Subtract the sum from planned production time to obtain run time.
5. Calculate availability as run time divided by planned production time.
6. Compare stop totals with operator logs and maintenance records.

**Output:** availability loss by reason, run time, and availability ratio.

A 25-minute setup remains an availability loss in the worked example even if it appears on the schedule.
The schedule explains why the time occurred; it does not make the minutes productive.
An improvement project can reduce setup duration without assuming that the entire duration is removable.
Safety checks, cleaning, and quality release steps may be mandatory operating controls.

When run time is zero, availability is zero if planned time is positive.
Performance cannot be calculated because its denominator is zero.
Report the overall productive output and reason for the zero-run period; do not invent a performance value.

### Step 5 — Establish ideal cycle time and count definitions

**Input:** approved product routing, equipment configuration, engineering study, and production counters.

1. Identify the item, revision, tooling, number of cavities, and approved operating mode.
2. Express the standard in minutes per counted unit.
3. Ensure the count point corresponds to the cycle-time standard.
4. Define whether a counter records strokes, packs, cavities, or individual units.
5. Convert to units only when the conversion is supported by the configuration record.
6. Define good first-pass and rejected counts at the same output boundary.
7. Date and version the standard and its approval.

**Output:** an effective-dated ideal cycle time and compatible counts.

An ideal cycle is a measurement reference, not a promise that the machine can run safely at that rate indefinitely under every product condition.
It should reflect an appropriate best demonstrated condition for the defined configuration.
A standard that is too slow inflates performance; one that is physically unattainable makes ordinary running look artificially poor.
Never increase the ideal cycle time simply to remove a performance value over 100% without investigating the count, configuration, and standard.

If 10 strokes produce four units each, the total count is 40 units only when all four cavities actually produced units.
A blocked cavity or a changed mold configuration breaks that conversion.
Record the evidence for the effective cavity count.

### Step 6 — Compute performance and its losses

**Input:** run minutes, ideal minutes per unit, and total units.

1. Multiply ideal cycle time by total count to obtain net run time in ideal minutes.
2. Divide net run time by run time to obtain performance.
3. Subtract net run time from run time to obtain the performance-loss residual.
4. If microstop durations are reliably measured inside run time, attribute them separately.
5. Treat the remaining residual as speed or unclassified loss only to the extent supported by evidence.
6. Check whether the ideal-time output exceeds the available running time.

**Output:** performance ratio and a reconciled performance-loss branch.

A line may be in a run state while briefly starved or blocked.
Those interruptions can remain in the performance branch if the stated classification threshold treats them as minor stops.
They must not also be counted as availability stops.
A measured microstop estimate greater than the performance residual indicates inconsistent definitions or data.
Investigate rather than forcing the residual to zero.

If performance exceeds 100%, check the ideal cycle standard, product-to-standard mapping, counts, cavities, and event timing.
Do not cap the value at 100% in raw reporting; the excess is diagnostic evidence.
The standard OEE loss tree no longer has a nonnegative performance branch until the inconsistency is resolved.

### Step 7 — Compute quality and its losses

**Input:** total output count, first-pass good count, and the ideal cycle time.

1. Count total produced units at the declared measurement point.
2. Count units meeting the declared quality criterion without rework.
3. Calculate rejected first-pass units as total minus good.
4. Calculate quality as good divided by total.
5. Multiply rejected units by ideal cycle time to obtain quality-loss minutes.
6. Multiply good units by ideal cycle time to obtain fully productive minutes.
7. Reconcile the counts with inspection, rework, scrap, and disposition records.

**Output:** quality ratio, quality loss, and fully productive time.

A part can be reworked into a saleable part and still be a first-pass reject.
Its final recovery affects economic value, but it does not retroactively make the original pass good.
A defect detected downstream may require a later correction to the original shift if traceability supports the assignment.
Preserve the originally published value and the revision history.

When total output is zero, the quality fraction is undefined.
Do not substitute 100% because no rejects were recorded.
A zero-output shift can be economically serious even though a conventional quality ratio is unavailable.

### Step 8 — Reconcile ratios and the additive loss tree

**Input:** planned time, run time, total units, good units, and ideal cycle time.

1. Multiply availability, performance, and quality for the single-product period.
2. Independently divide ideal cycle time times good units by planned time.
3. Reconcile the two expressions, allowing only disclosed rounding differences.
4. Sum availability loss, performance loss, quality loss, and fully productive time.
5. Compare that sum with planned time.
6. Investigate negative branches, missing time, or discrepancies.

**Output:** a reconciled OEE score and loss tree.

The two OEE formulas share inputs, so agreement is an arithmetic check rather than independent validation.
Validate counters and event records against independent evidence where possible.
For example, compare sampled machine events with operator observations and compare count totals with inspected lot records.

A Pareto chart can rank reason categories in minutes after reconciliation.
An unattributed residual belongs in the chart as unknown, not in a convenient named cause.
Sorting identifies where to investigate; it does not prove that the largest category is the most profitable improvement.

### Step 9 — Aggregate without averaging away the exposure

**Input:** comparable asset-period records and effective-dated product standards.

1. Confirm that each record has the same planned-time policy and compatible boundaries.
2. Sum planned minutes and fully productive ideal minutes across the chosen periods.
3. Divide total fully productive minutes by total planned minutes.
4. Sum the underlying loss minutes before ranking categories.
5. Show each shift's values as well as the aggregate.
6. Explain product-mix and calendar changes.

**Output:** a time-weighted aggregate with its component distribution.

An unweighted average gives a 30-minute period the same influence as a 450-minute shift.
For example, 90% over 30 minutes and 60% over 450 minutes aggregate to 61.875%, not 75%.
The first period contributes 27 fully productive minutes; the second contributes 270 minutes.
Together they contribute 297 fully productive minutes out of 480 planned minutes.

A mix of products with different ideal cycle times needs ideal-time weighting.
Let product index `i` have ideal cycle `c_i`, total count `T_i`, and good count `G_i`.
Then net run time is `sum(c_i × T_i)` and fully productive time is `sum(c_i × G_i)`.
Use weighted quality `sum(c_i × G_i) / sum(c_i × T_i)` if the factors must multiply back to OEE.
The ordinary unit-weighted first-pass yield `sum(G_i) / sum(T_i)` can still be reported, but it is a different measure.
Do not multiply it by mixed-product availability and performance and assume the result is the ideal-time OEE.

### Step 10 — Trace causes and select an intervention

**Input:** reconciled loss tree, reason evidence, process observations, and proposed remedies.

1. Select a loss branch and inspect the underlying events, not just the percentage.
2. Observe the process and interview operators and relevant specialists.
3. Separate symptom codes from verified causal mechanisms.
4. Estimate the technically achievable reduction, not the entire historical loss.
5. Check whether the intervention changes another branch or creates a new constraint.
6. Identify the downstream demand and capacity needed to use any recovered output.
7. Estimate incremental contribution and all relevant incremental costs in a common time horizon.
8. Show a conservative scenario and the conditions under which the decision changes.

**Output:** a ranked, conditional business case and an owner-approved experiment proposal.

If recovered time is on a non-bottleneck machine and no customer demand exists, it may not create additional sales.
It may still reduce overtime, inventory exposure, or maintenance risk, but each benefit requires separate evidence.
Avoid adding the full value of extra output and the full value of labor saved when the same hours cannot support both benefits.

A quality intervention can also reduce rework or warranty costs.
Count those benefits only when they are incremental and not already included in the unit contribution assumption.
A safer process or a regulatory requirement may justify action even if the estimated short-run contribution is negative.
Such decisions should be identified as risk or compliance decisions, not disguised as OEE economics.

### Step 11 — Feed the capacity and scheduling process

**Input:** approved intervention, measured sustainable results, planning calendar, and routing data.

1. Distinguish demonstrated performance from a project target.
2. Test whether the measured improvement persists over representative product and staffing conditions.
3. Translate the improvement into good units or usable resource minutes in the relevant planning bucket.
4. Check material, labor, tooling, quality-release, and downstream constraints.
5. Update the capacity assumption through the master-data approval process.
6. Communicate any schedule or customer-commitment change to the authorized owner.
7. Retain the original baseline and an effective date for the revised assumption.

**Output:** a controlled planning update or a documented decision to retain the previous capacity assumption.

Do not multiply a planning capacity figure by OEE if that figure already incorporates the same downtime, speed, or yield losses.
The result would double derate capacity.
Document each conversion from calendar time to effective good output and include each loss once.

### Step 12 — Govern revisions and sustain the measure

**Input:** approved definitions, access controls, raw events, and review cadence.

1. Assign ownership for the asset calendar, state mapping, ideal cycles, and quality dispositions.
2. Keep raw source events immutable or auditable.
3. Record all overrides with a user, timestamp, reason, and approver.
4. Version the threshold and state-priority rules.
5. Establish a cutoff and revision policy for late quality results.
6. Publish the percentage together with absolute minutes, counts, and coverage indicators.
7. Review recurring unknown causes and data-quality exceptions.
8. Recalculate historical comparisons if a definition changes, or clearly mark a break in series.

**Output:** a defensible scorecard and an action log, not merely a dashboard.

A daily meeting should separate a data defect from a process defect.
An unexplained counter jump is not evidence that operators ran faster.
A missing stop reason is not evidence that no stop occurred.
Escalate unresolved measurement issues before attaching financial targets or incentives.

## 6. Worked example

### Case and decision

A fictional plant machines a single pump sleeve on a CNC turning cell.
CNC means computer numerical control.
The cell is the temporary constraint for a confirmed production campaign, and an additional good unit can be completed downstream and sold during the same campaign.
The planning team must choose between two small interventions for the next comparable shift.

The shift calendar is 480 minutes.
A 30-minute unpaid meal break is outside planned production by the published policy.
The remaining planned production time is 450 minutes.
All other stop time, including a scheduled setup, remains inside the denominator.

The approved ideal cycle time is 0.50 minute per unit for the product and configuration in this shift.
It is illustrative and must be validated in a real plant.
The counter records individual completed sleeves at the output of the cell.
The quality count is first-pass acceptance at that same boundary.
The shift's late inspection results are included before the score is finalized.

| Shift input | Value | Interpretation |
|---|---:|---|
| Calendar time | 480 minutes | One shift. |
| Excluded meal | 30 minutes | Outside the chosen planned-time denominator. |
| Planned production | 450 minutes | 480 minus 30 minutes. |
| Breakdown | 35 minutes | Disjoint stop interval inside planned time. |
| Changeover | 25 minutes | Disjoint stop interval inside planned time. |
| Run time | 390 minutes | 450 minus 35 minus 25 minutes. |
| Ideal cycle | 0.50 minute/unit | Effective standard for this item and cell. |
| Total count | 660 units | Includes first-pass rejects. |
| First-pass good count | 627 units | Total minus 33 rejected units. |
| First-pass rejected count | 33 units | Includes any units later reworked. |

The plan assumed 700 good sleeves in the shift.
That number is a production plan, not a verified capacity or a customer commitment.
Actual first-pass good output was 627 units, a shortfall of 73 units against that plan.
The OEE tree explains the shift's ideal-time losses; it does not by itself prove why the plan was set at 700 units.

### Calculate the three ratios

Availability is `390 / 450 = 0.8666667`, or **86.67%**.
The numerator is the 390 minutes classified as run time.
The 60 minutes of availability loss comprises 35 minutes of breakdown plus 25 minutes of changeover.

Net run time is `660 units × 0.50 minute/unit = 330 minutes`.
Performance is `330 / 390 = 0.8461538`, or **84.62%**.
Performance loss is `390 − 330 = 60 minutes`.

Quality is `627 / 660 = 0.95`, or **95.00%**.
Quality loss is `33 units × 0.50 minute/unit = 16.5 ideal minutes`.
Fully productive time is `627 units × 0.50 minute/unit = 313.5 ideal minutes`.

OEE is `0.8666667 × 0.8461538 × 0.95 = 0.6966667`, or **69.67%**.
The direct check is `313.5 / 450 = 0.6966667`, also **69.67%**.
The result is not a claim that 69.67% of employees' paid time was productive.
It is a ratio of ideal output time to selected planned machine time.

### Build the loss tree

For this illustration, independently captured microstop intervals total 25 minutes and are included in the 390-minute run state.
The remaining 35 minutes of performance loss are an unattributed speed-and-other residual.
They are not established as a single root cause.
The availability and microstop intervals do not overlap.

| Branch | Minutes | Basis |
|---|---:|---|
| Breakdown | 35.0 | Observed availability stop. |
| Changeover | 25.0 | Observed availability stop. |
| Microstops | 25.0 | Observed inside run time. |
| Speed and other residual | 35.0 | 60.0 performance loss minus 25.0 microstops. |
| First-pass quality loss | 16.5 | 33 rejected units at 0.50 minute/unit. |
| Fully productive time | 313.5 | 627 good units at 0.50 minute/unit. |
| Total planned production time | 450.0 | Sum of all rows above. |

The reconciliation is `35 + 25 + 25 + 35 + 16.5 + 313.5 = 450 minutes`.
Performance loss is a residual derived from the cycle standard and count.
Attributing 25 minutes to microstops is defensible here only because those intervals were separately captured and were not counted as availability stops.
If that evidence were absent, the entire 60-minute performance loss would remain unattributed until investigated.

### Compare two conditional actions

Assume the plant can sell any extra first-pass good sleeves during the campaign and that downstream capacity and material are available.
Assume relevant incremental contribution is USD 25 per additional sold good sleeve, after the variable costs of making and selling it.
The following calculations are expected-value approximations; actual output is integer and outcomes are uncertain.
Both proposals concern one comparable shift and have one-shift incremental costs.
They are alternatives for a scarce engineering slot, not automatically additive projects.

**Action A — repair a recurring fault.**
Engineering estimates that the repair would recover 20 of the 35 breakdown minutes in the next comparable shift.
The repair's incremental one-shift cost is USD 400.
Assume the recovered 20 minutes run at the baseline performance and first-pass quality rates.
At an ideal cycle of 0.50 minute/unit, 20 minutes at ideal speed corresponds to 40 total units before performance and quality adjustments.
The expected incremental good output is `20 / 0.50 × (330 / 390) × 0.95 = 32.1538 units`.
Expected incremental contribution is `32.1538 × USD 25 = USD 803.85`.
After USD 400 incremental cost, the expected net benefit is **USD 403.85 per comparable shift**.
The break-even recovery under these assumptions is `400 / (25 × (1 / 0.50) × (330 / 390) × 0.95) = 9.91 minutes`.

**Action B — prevent a first-pass defect.**
Quality estimates that a fixture adjustment would convert 11 of the 33 first-pass rejects into first-pass good units without changing total count or other losses.
The incremental one-shift cost is USD 90.
Expected incremental good output is 11 units.
Expected incremental contribution is `11 × USD 25 = USD 275`.
After USD 90 incremental cost, the expected net benefit is **USD 185 per comparable shift**.
The break-even defect reduction is `90 / 25 = 3.6 units` in expectation, or at least 4 units if the outcome is a whole number.
This calculation assumes that the baseline rejects would not otherwise have been recovered and sold at comparable contribution.
If baseline rework produces saleable units, use the incremental recovery and rework savings instead of treating every converted reject as a new sale.

Under these assumptions, choose Action A for the scarce slot because USD 403.85 exceeds USD 185.
This is a conditional recommendation, not a commitment to a delivery date.
Action A's 20-minute recovery must be verified, and the cell must remain the binding constraint during the extra production.
If only 8 minutes are recovered, Action A yields approximately `8 / 0.50 × (330 / 390) × 0.95 × 25 − 400 = −USD 78.46`.
Under that sensitivity, Action B is preferable if its assumptions remain valid.

The expected additional OEE from Action A is `32.1538 × 0.50 / 450 = 3.57 percentage points`.
The expected additional OEE from Action B is `11 × 0.50 / 450 = 1.22 percentage points`.
These are modeled differences, not measured results.
The choice rests on expected net benefit and feasibility, not on percentage points alone.

### Decision controls

The maintenance owner confirms the repair scope, expected recovery, and safe work method.
The quality owner checks that the repair does not alter product conformity or require a new process approval.
The planner confirms raw material and downstream availability in the same shift.
Finance validates the contribution assumption, incremental cost, and treatment of recoverable rejects.
The sales owner confirms that additional output can be sold in the relevant period.
The scheduler changes the production plan only after the intervention and capacity assumption are authorized.
The customer team changes a customer commitment only through its normal approval process.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reproduces the single-product worked example and its two conditional interventions.
Run it as a script, for example with `python oee_example.py` after saving the block locally.
It uses floating-point arithmetic and rounds only for display.
The script treats counts and times as numeric values, not as evidence that the source records are correct.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Shift:
    calendar_min: float
    excluded_min: float
    breakdown_min: float
    changeover_min: float
    ideal_min_per_unit: float
    total_units: int
    good_units: int
    microstop_min: float


def calculate(shift: Shift) -> dict[str, float]:
    planned = shift.calendar_min - shift.excluded_min
    stops = shift.breakdown_min + shift.changeover_min
    run = planned - stops
    if planned <= 0 or run <= 0:
        raise ValueError("This example requires positive planned and run time")
    if shift.ideal_min_per_unit <= 0 or shift.total_units <= 0:
        raise ValueError("This example requires a positive cycle and count")
    if not 0 <= shift.good_units <= shift.total_units:
        raise ValueError("Good units must be between zero and total units")
    if min(shift.excluded_min, shift.breakdown_min,
           shift.changeover_min, shift.microstop_min) < 0:
        raise ValueError("Durations must be nonnegative")

    net_run = shift.ideal_min_per_unit * shift.total_units
    productive = shift.ideal_min_per_unit * shift.good_units
    performance_loss = run - net_run
    if performance_loss < 0 or shift.microstop_min > performance_loss:
        raise ValueError("Investigate cycle, counts, and stop classification")

    availability = run / planned
    performance = net_run / run
    quality = shift.good_units / shift.total_units
    quality_loss = shift.ideal_min_per_unit * (
        shift.total_units - shift.good_units
    )
    speed_residual = performance_loss - shift.microstop_min
    tree_total = stops + shift.microstop_min + speed_residual + quality_loss + productive
    if abs(tree_total - planned) > 1e-9:
        raise ValueError("Loss tree does not reconcile")

    return {
        "planned": planned,
        "run": run,
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "oee": availability * performance * quality,
        "direct_oee": productive / planned,
        "availability_loss": stops,
        "performance_loss": performance_loss,
        "quality_loss": quality_loss,
        "productive": productive,
        "speed_residual": speed_residual,
    }


shift = Shift(
    calendar_min=480,
    excluded_min=30,
    breakdown_min=35,
    changeover_min=25,
    ideal_min_per_unit=0.50,
    total_units=660,
    good_units=627,
    microstop_min=25,
)
result = calculate(shift)

print(f"Planned: {result['planned']:.1f} min; run: {result['run']:.1f} min")
for key in ("availability", "performance", "quality", "oee", "direct_oee"):
    print(f"{key}: {result[key] * 100:.2f}%")
print(
    "Losses: "
    f"availability={result['availability_loss']:.1f} min, "
    f"performance={result['performance_loss']:.1f} min, "
    f"quality={result['quality_loss']:.1f} min, "
    f"productive={result['productive']:.1f} min"
)
print(f"Performance residual after microstops: {result['speed_residual']:.1f} min")

contribution_usd = 25.0
recovered_min = 20.0
repair_cost_usd = 400.0
repair_good = (
    recovered_min / shift.ideal_min_per_unit
    * result["performance"] * result["quality"]
)
repair_net = repair_good * contribution_usd - repair_cost_usd
quality_good = 11
quality_cost_usd = 90.0
quality_net = quality_good * contribution_usd - quality_cost_usd
print(f"Repair: {repair_good:.2f} expected good units; net USD {repair_net:.2f}")
print(f"Quality: {quality_good} additional good units; net USD {quality_net:.2f}")
```

Expected output:

```text
Planned: 450.0 min; run: 390.0 min
availability: 86.67%
performance: 84.62%
quality: 95.00%
oee: 69.67%
direct_oee: 69.67%
Losses: availability=60.0 min, performance=60.0 min, quality=16.5 min, productive=313.5 min
Performance residual after microstops: 35.0 min
Repair: 32.15 expected good units; net USD 403.85
Quality: 11 additional good units; net USD 185.00
```

This is deliberately a small instructional calculation, not a production data pipeline.
It rejects zero run or total count because the factor ratios would be undefined in that case.
A production scorecard should preserve those periods and display undefined factors explicitly.
It should also check interval overlap, timestamp boundaries, count provenance, standards, late inspections, and configuration changes.
Those checks require source data not represented in this in-memory example.

The repair calculation uses an expected number of good units rather than a promise of a fractional physical sleeve.
The approach assumes that the recovered time experiences the baseline performance and quality rates.
If a repair also changes those rates, estimate the combined effect using a validated scenario rather than holding them constant by assumption.

## 8. Data, standards, and regulatory context

### Data required for a defensible score

A minimal asset calendar identifies the asset, shift, start and end instants, and approved excluded intervals.
The raw event table needs event identifiers, source timestamps, states, reasons, and the producing system.
The production record needs a counter identifier, product and configuration, counter values or transactions, and the measurement boundary.
The quality record needs the lot or unit linkage, first-pass disposition, inspection time, and later changes.
The ideal-cycle record needs the item, equipment configuration, effective dates, units, and approval history.

These tables must be joinable without multiplying records.
A single production count joined to several overlapping alarms can be accidentally repeated once for each alarm.
Aggregate or resolve event intervals before joining to counts.
Track the number of source records, matched records, and unmatched records for every published period.

At minimum, retain:

- Asset and plant identifiers, with ownership and valid dates.
- Shift start and end instants, plus the local timezone.
- Scheduled breaks and other approved exclusions.
- State intervals with source and resolved classification.
- Reason code version, reason assignment, and override history.
- Product, revision, operation, and equipment configuration.
- Total and first-pass good counts with the count point.
- Counter reset, rollover, and correction indicators.
- Inspection status and the quality data cutoff.
- Ideal cycle and effective date, in minutes per counted unit.
- Published score version and the calculation rule version.

A reason code's vocabulary should be stable enough for a trend and detailed enough to support action.
A broad `other` category may be necessary initially.
If it dominates the loss tree, the team cannot reliably select a root-cause intervention.
Assign an owner and a target date to investigate the missing attribution rather than inventing detail.

### Time integrity

State events commonly arrive late, duplicate, or out of order.
Use event time to construct the physical timeline and ingestion time to monitor latency.
Keep both timestamps.
A late event can change a previously published shift; the revision policy must say when and how the score is restated.

Clip each event to the planned intervals that it overlaps.
A stop from 13:55 to 14:10 crossing a shift boundary at 14:00 contributes 5 minutes to the first shift and 10 minutes to the second, subject to each shift's planned calendar.
Do not assign all 15 minutes to both shifts.
If the stop crosses an excluded break, the overlapping excluded minutes do not belong in planned-time availability loss.

State resolution must also address gaps.
A 7-minute gap between messages is not necessarily 7 minutes of run time.
Whether a prior state can be carried forward depends on the source system's semantics and heartbeat policy.
Mark unresolved gaps as unknown if the evidence is insufficient.
A dashboard should display coverage and unknown duration beside OEE.

### Count and quality integrity

A counter delta can be wrong if a device resets, rolls over, or changes product during the interval.
Use explicit reset and configuration records where available.
Do not infer a production count from a negative delta without a documented rollover limit and supporting evidence.
Compare totals with independent lot, packaging, or inspection records when their boundaries are compatible.

Quality results often arrive after the production shift.
Distinguish pending units from good first-pass units; absence of a recorded reject is not necessarily an inspection pass.
Publish a provisional score if policy permits and label the inspection coverage.
When a later disposition changes a count, retain both the previous published score and the restatement.

Some products have multiple quality checks or multiple opportunities for defects.
The chosen first-pass criterion must define whether a unit fails on any relevant characteristic.
Do not count defects as defective units if one unit can have several defects.
For multi-stage production, identify whether a rejection belongs to the local process, an upstream process, or an unknown source before assigning a local cause.

### Standards and legal context

[ISO 22400-2:2014](https://www.iso.org/standard/54497.html) describes key performance indicators for manufacturing operations management, including definitions and associated elements.
It provides a reference vocabulary and framework; an implementation still needs a local, explicit boundary and data mapping.
Consult the licensed standard and applicable later editions or amendments before claiming conformance.
This lesson's example is an educational convention and does not assert certification or conformance.

[IEC 62264-1](https://webstore.iec.ch/en/publication/6675) addresses enterprise-control system integration models and terminology.
It is relevant when connecting equipment, manufacturing operations, and enterprise planning records.
It does not by itself guarantee that an individual counter, ideal cycle, or quality disposition is correct.
A mapping project must still validate identifiers, timestamps, and business semantics.

[ISO 9001](https://www.iso.org/standard/62085.html) describes requirements for a quality management system when an organization elects or is required by contract to use that standard.
It does not make a particular OEE threshold a universal legal requirement.
Industry-specific regulations, customer specifications, safety rules, and record-retention obligations depend on product and jurisdiction.
Consult the applicable requirements and qualified professionals; this lesson is not legal or regulatory advice.

The cited standards are voluntary standards unless made applicable through law, contract, or another binding obligation.
The existence of a standard is not evidence that a plant follows it.
An OEE target is an internal management objective unless a particular agreement states otherwise.
Do not use a generic percentage as a substitute for risk assessment or validated capacity.

### Access and auditability

Production, quality, and maintenance records can reveal commercially sensitive information.
Grant access according to operational need and retain evidence of authorized changes.
A correction should preserve the original value, corrected value, reason, timestamp, and responsible person.
A supervisor override can be valid, but it should not silently erase the source event.

The data team owns implementation quality; process owners remain accountable for the business meaning of the states and standards.
Quality owns the acceptance rule and disposition process.
Engineering owns the technical basis for the cycle standard.
Planning owns use of the resulting capacity assumption.
Finance owns the economic assumptions used in an investment case.

## 9. Industry practice and case evidence

### Use the metric as a question, not a verdict

Equipment effectiveness is widely used in total productive maintenance (TPM) practice.
TPM is an operating approach that involves production and maintenance in sustaining equipment performance.
The OEE decomposition is useful because it connects a high-level result to time and count records that can be investigated.
It does not establish that every loss can be economically removed.

Muchiri and Pintelon review performance measurement using OEE in manufacturing and discuss the interpretation and improvement of the measure.
Their peer-reviewed paper is a useful methodological source, not evidence that this lesson's fictional plant achieves a particular return.
The paper supports treating OEE as a structured measurement tool whose definitions and context matter.
See [Muchiri and Pintelon (2008)](https://doi.org/10.1080/00207540601142645).

The practical implication is to show both the aggregate and the loss ledger.
A manager who sees only 69.67% cannot tell whether a repair, setup improvement, quality intervention, or data correction is appropriate.
A team that can inspect the underlying shift records can test those possibilities.
The local engineering review determines which intervention is safe and feasible.

### Public case: semiconductor manufacturing

Scott and Pisa describe equipment efficiency and factory-level performance in semiconductor manufacturing in their 1998 paper, cited in the sources below.
Their discussion distinguishes individual equipment effectiveness from the effects of equipment interactions and factory constraints.
The case is evidence that equipment-level measures need a factory context; it is not evidence about the fictional pump cell.
Semiconductor equipment, product mix, reentrant flows, and capital intensity differ from a simple single-machine example.
Do not transfer a numerical target or expected savings without local analysis.

[INFERENCE] A plant reviewing a low-OEE tool in a reentrant flow should check whether extra tool capacity increases completed wafers after downstream resource and demand constraints.
The inference follows from the factory-level distinction, but the actual answer requires that plant's routing, buffers, schedule, and demand data.
An OEE increase can be valuable for reliability even when immediate shipment volume is unchanged.

### Public case: batch and pharmaceutical production

Chikwendu, Chima, and Edith studied a tablet compression machine at Gauze Pharmaceutical and Laboratory Limited in Nigeria.
They report OEE factors before and after a total productive maintenance intervention, using a case-study design rather than a randomized comparison.
Their sector-specific study is not a regulatory ruling and does not establish that every pharmaceutical plant should use the same loss boundaries.
Its reported before-and-after change does not isolate the effect of maintenance from other possible changes over time.
In batch operations, product changeovers, validated cleaning, release testing, and product-specific cycles can materially affect the meaning of reported utilization.
A site must document how those activities enter its own denominator and comply with its applicable quality system.

[INFERENCE] A batch plant comparing two periods with different product mixes should calculate ideal-time-weighted output and also show the mix and validated changeover requirements.
A change in the percentage may reflect the mix, the measurement standard, or operational performance.
Only plant-specific records can separate these possibilities.

### Interpreting a public or vendor benchmark

A public percentage may use a different planned-time boundary, product mix, first-pass definition, or ideal cycle.
Some published figures are vendor marketing, not independently audited performance data.
Before using an external number, ask whether the asset type, period, operating context, exclusions, and definitions match.
If those details are not available, the percentage is not a valid comparative benchmark.

Internal comparisons have the same problem.
A shift with a difficult product or a mandatory cleaning interval should not be penalized against another shift without adjusting or explaining the context.
A fair review can report unadjusted loss minutes and a stratified view by product and configuration.
Avoid creating incentives to suppress stop codes, move rejects to another shift, or relax the cycle standard.

### Typical operating cadence

A shift team reviews yesterday's stop events and immediate containment needs.
Maintenance reviews recurring failure modes and repair opportunities over a longer period.
Quality investigates first-pass defects using traceable inspection and disposition evidence.
Planning reviews capacity assumptions at the horizon where a schedule or production plan can still change.
Finance validates benefits when a proposal competes for money or scarce engineering time.

The daily review should record the observation, suspected cause, action owner, due date, and closure evidence.
It should distinguish an unverified hypothesis from a confirmed cause.
A later comparison should use the same definitions and a representative product mix.
Where an intervention coincides with a mix change, a simple before-and-after OEE comparison may overstate or understate the causal effect.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Denominator manipulation | Exclusions rise after targets are missed | Difficult time is removed from planned production | Score improves without more output | Approve exclusions prospectively and publish calendar utilization. |
| Overlapping alarms | Summed stop minutes exceed elapsed minutes | Several alarms describe the same interval | Availability is understated | Resolve disjoint states and retain multi-cause annotations separately. |
| Hidden setup | Setup appears nowhere in the ledger | Scheduled changeover is removed without explanation | False comparison and overstated capacity | State the policy and reconcile setup to the schedule. |
| Incorrect cycle standard | Performance exceeds 100% repeatedly | Standard, unit, or configuration is wrong | Misleading speed and loss estimates | Investigate mapping and approve an effective-dated standard. |
| Counter reset | Negative or implausibly large count delta | Restart or rollover mistaken for output | Invalid net run time | Detect reset, inspect source, and document correction. |
| Cavity mismatch | Count changes after a tooling swap | Pulses are converted with the wrong cavity count | Inflated or understated units | Version the configuration and validate output at the count point. |
| Pending inspection treated as pass | Perfect quality before inspection closes | Missing disposition is equated to good | Overstated OEE and capacity | Publish pending counts and revise after disposition. |
| Rework counted as first-pass good | Quality improves while rework rises | Final recovery replaces original disposition | Quality loss is hidden | Preserve first-pass and final disposition separately. |
| Defects counted as units | Reject count exceeds affected lots or units | Multiple defects on one part are summed as parts | Quality ratio is wrong | Deduplicate by unit or use defensible lot accounting. |
| Misclassified microstops | Performance and availability jump after a threshold change | Same events move between branches | False improvement narrative | Version the threshold and restate comparisons. |
| Gaps assumed to be run | Long periods lack heartbeat messages | Missing data is silently filled | Unsupported availability | Measure coverage and use a documented state-carry rule. |
| Midnight double count | Shift totals exceed daily total | Intervals are not clipped at boundaries | Inflated losses | Use half-open intervals and split events. |
| Simple percentage averaging | Short and long shifts receive equal weight | Ratios are averaged instead of numerators and denominators | Biased aggregate | Sum productive and planned minutes first. |
| Mixed-product quality error | Factor product does not equal direct OEE | Unit-weighted yield is applied to unequal ideal cycles | Incorrect OEE decomposition | Use ideal-time-weighted quality for the OEE factors. |
| Double derating | Capacity falls unexpectedly after an OEE update | Historical losses are already in effective capacity | Artificial overload and poor commitments | Trace each loss from calendar to good output once. |
| Non-bottleneck optimization | Local output rises but shipments do not | Another constraint limits flow | Extra work in process and no sale | Check system constraint, demand, and downstream capacity. |
| Counting unsold output as profit | Inventory rises after improvement | Additional production is assumed to be additional sales | Inflated benefit case | Use feasible incremental sales or separately evidenced cost avoidance. |
| Ignoring rework economics | Reject reduction valued at full new-sale contribution | Baseline rejects were recoverable | Benefit overstated | Model incremental recovery, cost, and timing. |
| Incentive gaming | Unknown stops or late corrections increase near target dates | People optimize the score instead of flow | Unreliable data and unsafe behavior | Audit definitions and reward evidence-based improvement. |
| Late standard changes | Past scores move without a process change | New cycle is applied retrospectively without disclosure | Broken trend and credibility | Version standards and publish restated versus original series. |
| Double-counted benefit | Project claims both full throughput gain and full saved labor for the same minutes | One recovered resource is assigned twice | Business case exceeds feasible value | Map time use and approve mutually consistent scenarios. |
| Safety trade-off | Operators bypass checks to improve run time | Metric becomes more important than the operating control | Injury or nonconforming product | Preserve safety and quality controls; escalate unsafe targets. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test it |
|---|---|---|---|
| Planned-time policy is stable | Comparable denominators | Exclusions or break rules change | Compare versioned calendars and recalculate a common basis. |
| State intervals are disjoint | Additive time tree | Overlapping alarms or duplicate events | Reconcile resolved intervals to the shift timeline. |
| Ideal cycle matches configuration | Correct ideal-time numerator | New tool, cavity count, speed, or unit definition | Verify effective-dated engineering record against the shift. |
| Counts share an output boundary | Valid total and good ratio | Rejects are recorded upstream or after a yield-changing step | Trace sample lots through count and inspection points. |
| First-pass disposition is complete | Quality factor is meaningful | Pending inspection or late defect discovery | Report inspection coverage and compare restatements. |
| Microstop estimate lies within run time | Performance attribution is mutually exclusive | Same interval is also an availability stop | Inspect timestamp overlap and the classification threshold. |
| Recovered time runs at baseline rates | Expected action A output | Repair changes speed, quality, staffing, or product mix | Measure the post-repair distribution under comparable conditions. |
| Additional output can be sold | Contribution is realized | Demand, shipment capacity, or another bottleneck limits sales | Check firm demand, downstream plan, and available material. |
| Unit contribution is incremental | Benefit represents relevant economics | Fixed costs or avoided costs are included twice | Finance reviews marginal revenue and variable cost components. |
| Quality action converts rejects into new sales | Action B's full unit benefit applies | Rejected units would have been reworked and sold | Compare final disposition and incremental rework cost. |
| Baseline is representative | A shift supports the decision | Atypical product, crew, or fault | Compare multiple matched shifts and uncertainty. |
| No harmful side effect occurs | Benefit estimate is complete | Faster production increases defects or risk | Quality and safety approval plus monitored pilot. |
| Ideal-time weighting is acceptable | Mixed-product OEE reconciles | Standard differs in quality or is not comparable | Validate per-product standards and disclose mix. |
| One resource boundary is meaningful | Local loss can guide action | Buffers and interactions dominate line output | Map the flow and check constraint status in each period. |

A single percentage is a descriptive measure of a selected resource and window.
It is not a causal estimate, a profit margin, a service level, or a safety measure.
A loss tree is an accounting decomposition, not an assurance that all losses are independently recoverable.
A change in OEE can reflect a change in denominator, cycle standards, quality timing, or product mix as well as a real process change.

A constraint can move after an improvement.
Before promising a throughput gain, model the next likely constraint and the demand available to absorb the output.
Where uncertainty is material, report ranges and decision thresholds rather than a precise but unsupported point estimate.

## 12. Exercises

Use the definitions and assumptions in this lesson unless an exercise explicitly changes them.
Show units and disclose rounding.

### Exercise 1 — Define the boundary

**Time:** 15 minutes.

A shift lasts 480 minutes and has a 30-minute meal break excluded by policy.
A scheduled setup takes 25 minutes, and an unexpected material shortage takes 20 minutes.
The published policy includes both setup and shortage in planned production time.

**Deliverable:** a three-line ledger for calendar time, planned production time, and availability loss.

**Acceptance criteria:** the ledger applies the stated policy, distinguishes an exclusion from a stop, and states the resulting run time assuming no other stop.

### Exercise 2 — Calculate the ratios

**Time:** 20 minutes.

Use the worked example: planned time 450 minutes, run time 390 minutes, ideal cycle 0.50 minute per unit, 660 total units, and 627 first-pass good units.

**Deliverable:** availability, performance, quality, and OEE as percentages, plus the direct OEE check.

**Acceptance criteria:** each formula has compatible units and the factor product agrees with the direct calculation within rounding.

### Exercise 3 — Reconcile an additive tree

**Time:** 20 minutes.

A different single-product shift has 400 planned minutes and 340 run minutes.
The ideal cycle is 0.40 minute per unit; total output is 700 units and first-pass good output is 665 units.
Of the performance loss, 20 minutes are verified microstops; the remainder is unattributed.

**Deliverable:** a loss tree in minutes, with availability, microstops, other performance loss, quality loss, and fully productive time.

**Acceptance criteria:** the branches are nonnegative, sum to 400 minutes, and reproduce OEE by the direct formula.

### Exercise 4 — Critique data integrity

**Time:** 20 minutes.

An asset reports a 20-minute stop across a shift boundary, two overlapping alarms for the same interval, and a counter reset at the boundary.
The next shift's quality results are 40% pending.
A dashboard has assigned the entire stop to both shifts, summed both alarm durations, and counted pending units as good.

**Deliverable:** a prioritized defect list and a release decision for the dashboard.

**Acceptance criteria:** address interval clipping, overlapping alarms, reset handling, pending quality, and provisional versus final publication.

### Exercise 5 — Compare interventions

**Time:** 25 minutes.

Use the worked example's USD 25 contribution per incremental sold good unit and baseline performance and quality rates.
A repair can recover 12 breakdown minutes at an incremental cost of USD 300 for the shift.
A quality change can prevent 8 first-pass rejects at an incremental cost of USD 80 for the shift.
Assume recovered output is saleable, downstream capacity is available, and rejects would otherwise not be recovered.
The changes are alternatives for the same engineering slot.

**Deliverable:** expected good units and net benefit for each option, and your recommendation.

**Acceptance criteria:** apply both performance and quality to recovered run time, do not apply them again to the 8 already-good units, and state the conditions on which the recommendation depends.

### Exercise 6 — Aggregate and compare

**Time:** 20 minutes.

Shift A has 30 planned minutes and 27 fully productive minutes.
Shift B has 450 planned minutes and 270 fully productive minutes.
A manager claims the combined OEE is the simple mean of 90% and 60%.

**Deliverable:** the aggregate OEE and a short critique of the manager's calculation.

**Acceptance criteria:** sum the compatible numerators and denominators, show the arithmetic, and identify the weighting error.

### Exercise 7 — Design a pilot

**Time:** 30 minutes.

A plant wants to investigate frequent short interruptions on a mixed-product packaging line.
The existing dashboard has no versioned stop threshold and uses one ideal rate for every package size.
There are separate counters for packs and individual units.

**Deliverable:** a one-page pilot design covering measurement, ownership, quality, intervention, and comparison.

**Acceptance criteria:** define the line boundary, state and count mapping, product-specific ideal cycles, a non-overlapping microstop rule, inspection timing, responsible owners, and an economic decision gate.
Include a method for detecting whether the improvement shifts the bottleneck.

### Exercise 8 — Explain the system limit

**Time:** 15 minutes.

A non-bottleneck machine increases OEE from 70% to 80%, but completed shipments do not change and work in process rises.

**Deliverable:** a short explanation and two follow-up checks.

**Acceptance criteria:** distinguish local ideal-time output from system throughput, identify a plausible constraint or demand explanation, and propose checks that can falsify it.

## 13. Self-check answers

### Exercise 1

Calendar time is 480 minutes.
Planned production time is `480 − 30 = 450 minutes`.
Availability loss is `25 + 20 = 45 minutes`, because both stop types remain inside the denominator under the stated policy.
Run time is `450 − 45 = 405 minutes`.
The meal break is excluded from planned production, not counted again as a stop.
A different policy could produce a different score, but it must be explicit and consistently applied.

### Exercise 2

Availability is `390 / 450 = 86.67%`.
Net run time is `660 × 0.50 = 330 minutes`.
Performance is `330 / 390 = 84.62%`.
Quality is `627 / 660 = 95.00%`.
OEE is `(390 / 450) × (330 / 390) × (627 / 660) = 69.67%`.
The direct check is `(627 × 0.50) / 450 = 69.67%`.
Rounding the displayed factors before multiplying can create a small difference.

### Exercise 3

Availability loss is `400 − 340 = 60 minutes`.
Net run time is `700 × 0.40 = 280 minutes`.
Performance loss is `340 − 280 = 60 minutes`, comprising 20 verified microstop minutes and 40 unattributed minutes.
Rejected units are `700 − 665 = 35 units`.
Quality loss is `35 × 0.40 = 14 minutes`.
Fully productive time is `665 × 0.40 = 266 minutes`.
The sum is `60 + 20 + 40 + 14 + 266 = 400 minutes`.
OEE is `266 / 400 = 66.50%`.
The unattributed 40 minutes should not be labeled a verified speed cause without further evidence.

### Exercise 4

Do not release the displayed values as final.
First split the stop at the shift boundary and clip each piece to planned time.
Then resolve the simultaneous alarms as one elapsed interval while retaining their separate causal annotations.
Investigate the counter reset using source records and a documented reset or rollover rule.
Treat the 40% pending inspection as pending, not first-pass good.
Publish a provisional score only if the reporting policy permits it and visibly shows the uncertainty and coverage.
Restate the affected shifts when the counts and dispositions are established, preserving the original publication.

### Exercise 5

The repair produces an expected `12 / 0.50 × (330 / 390) × 0.95 = 19.2923` additional good units.
Its expected net benefit is `19.2923 × USD 25 − USD 300 = USD 182.31`.
The quality option yields 8 additional good units and `8 × USD 25 − USD 80 = USD 120` net benefit.
Under the stated assumptions, choose the repair for the scarce slot.
The expected advantage is USD 62.31, so the recommendation is sensitive to the recovery estimate and demand assumptions.
If the additional units cannot be sold, or the rejects would otherwise be reworked into saleable units, recalculate the relevant economics.

### Exercise 6

The aggregate numerator is `27 + 270 = 297 fully productive minutes`.
The aggregate denominator is `30 + 450 = 480 planned minutes`.
Aggregate OEE is `297 / 480 = 61.875%`, or 61.88% when rounded to two decimal places.
The simple mean is 75%, but it gives the 30-minute period the same weight as the 450-minute period.
Summing numerator and denominator weights each period by its planned minutes.

### Exercise 7 rubric

A strong design identifies the physical line and its controlling output count.
It distinguishes packs from units and uses configuration records to map the counts.
It establishes per-product effective-dated ideal cycles and ideal-time-weighted aggregation.
It defines the stop threshold before the pilot and keeps short interruptions out of the availability branch.
It measures unknown state time and pending inspection rather than silently filling either.
It names operations, maintenance, quality, data, planning, and finance owners.
It compares matched product and staffing conditions and retains both loss minutes and first-pass quality.
It tests whether downstream capacity and demand can absorb extra output before claiming contribution.
It sets a safe intervention, review date, data-quality gate, and criterion for revising the plan.
A weak design supplies only a new OEE target or a vendor benchmark.

### Exercise 8 rubric

A strong explanation states that OEE measures local ideal-time output relative to planned time, not shipments.
A downstream constraint, limited demand, or a release delay could prevent additional shipments.
The observed rise in work in process is consistent with a downstream constraint but does not prove it.
Check downstream utilization, queues, downtime, and actual release capacity over the same period.
Also check available orders and shipment schedules to test whether demand limited throughput.
A weak explanation assumes all extra local output is sold or declares the improvement worthless without checking other benefits.

## 14. Interview questions

### 1. What does OEE measure?

A strong answer defines fully productive ideal time divided by a stated planned production time and names availability, performance, and first-pass quality.
It explains the asset and time boundary.
A weak answer calls OEE a profit margin or a measure of employee effort.

### 2. How do you treat a planned changeover?

A strong answer asks whether it lies within the published planned-time policy and applies that policy consistently.
It preserves setup information for scheduling and improvement even if another reporting denominator is also used.
A weak answer silently excludes setup solely to improve the percentage.

### 3. Why can performance exceed 100%?

A strong answer checks cycle standard, counts, units, tooling, cavities, and state timing.
It treats the excess as a data or definition investigation rather than automatically capping it.
A weak answer declares that the machine has exceeded a physical ideal without checking its inputs.

### 4. How do you avoid double-counting losses?

A strong answer constructs disjoint state intervals and separates availability stops from microstops inside run time.
It assigns reject units their ideal time in the quality branch and reconciles all branches to planned minutes.
A weak answer sums overlapping alarms and full actual reject time on top of performance loss.

### 5. How should mixed products be aggregated?

A strong answer sums product-specific ideal time for total and good units and divides by compatible planned minutes.
It distinguishes ideal-time-weighted quality from ordinary unit-weighted first-pass yield.
A weak answer averages percentages without regard to time, cycle, or volume.

### 6. What do you do with a zero-output shift?

A strong answer reports the planned time and zero good output while labeling performance or quality undefined when its denominator is zero.
It investigates the state and records the relevant stop cause.
A weak answer declares quality to be 100% because no defects were recorded.

### 7. How would you validate event data?

A strong answer checks clock alignment, timezone, interval overlap, gaps, state precedence, and boundary clipping.
It compares sampled events with independent operational evidence and reports unknown coverage.
A weak answer assumes that a machine-generated timestamp is always correct.

### 8. When does recovered time create financial value?

A strong answer requires a feasible route from recovered time to incremental sold output or evidenced cost avoidance.
It checks demand, the constraint, material, quality, downstream capacity, and incremental cost.
A weak answer multiplies all recovered minutes by selling price without considering whether units can be made or sold.

### 9. Why might OEE rise while shipments stay flat?

A strong answer identifies an alternative constraint, demand limit, inventory accumulation, or changing denominator as possible explanations.
It proposes data that distinguish them.
A weak answer assumes the reporting system must be wrong or that the company necessarily earned more.

### 10. How do you handle late quality results?

A strong answer separates pending, first-pass good, reject, and final disposition; it publishes coverage and follows a restatement policy.
A weak answer counts pending units as good or erases the prior published result.

### 11. What makes an OEE comparison fair?

A strong answer aligns asset boundary, calendar policy, ideal cycle, product mix, stop threshold, and quality timing.
It presents absolute loss minutes and uncertainty alongside the ratios.
A weak answer compares any two percentages as if they had the same denominator and operating conditions.

### 12. How does OEE feed a production plan?

A strong answer translates demonstrated results into effective good capacity for the relevant calendar and checks for losses already embedded in the planning standard.
It distinguishes a measured baseline from a target and a customer commitment.
A weak answer applies the OEE percentage to every capacity figure without tracing existing derating.

## 15. Key terms

- **Asset boundary:** the physical resource whose states and output are measured; see [the mental model](#3-the-core-mental-model).
- **Availability:** run time divided by planned production time; see [step 4](#step-4--measure-availability-loss).
- **Availability loss:** planned time allocated to stops; see [step 4](#step-4--measure-availability-loss).
- **Calendar time:** total elapsed time in the reporting window; see [step 2](#step-2--establish-the-planned-time-calendar).
- **Changeover:** activity switching the resource between production conditions; see [the case](#case-and-decision).
- **Contribution:** incremental revenue less relevant incremental variable costs; see [the action comparison](#compare-two-conditional-actions).
- **Counter reset:** a counter change caused by restart rather than output; see [data integrity](#count-and-quality-integrity).
- **Cycle standard:** the approved ideal time per counted unit; see [step 5](#step-5--establish-ideal-cycle-time-and-count-definitions).
- **Exclusion:** calendar time outside the selected planned-time denominator; see [step 2](#step-2--establish-the-planned-time-calendar).
- **First-pass good:** units passing the declared criterion without rework; see [step 7](#step-7--compute-quality-and-its-losses).
- **Fully productive time:** good units multiplied by ideal cycle time; see [the mental model](#3-the-core-mental-model).
- **Ideal cycle time:** reference ideal minutes per unit for a defined configuration; see [step 5](#step-5--establish-ideal-cycle-time-and-count-definitions).
- **Ideal-time weighting:** aggregation using each product's ideal time; see [step 9](#step-9--aggregate-without-averaging-away-the-exposure).
- **Loss tree:** additive reconciliation of planned time to fully productive time; see [step 8](#step-8--reconcile-ratios-and-the-additive-loss-tree).
- **Microstop:** short interruption assigned to performance under a stated rule; see [step 6](#step-6--compute-performance-and-its-losses).
- **Net run time:** total units multiplied by ideal cycle time; see [step 6](#step-6--compute-performance-and-its-losses).
- **OEE:** overall equipment effectiveness, a ratio of fully productive time to planned time; see [the mental model](#3-the-core-mental-model).
- **Performance:** net run time divided by run time; see [step 6](#step-6--compute-performance-and-its-losses).
- **Planned production time:** selected calendar time during which the resource is expected to produce; see [step 2](#step-2--establish-the-planned-time-calendar).
- **Quality:** first-pass good divided by total output for a single product; see [step 7](#step-7--compute-quality-and-its-losses).
- **Reason code:** controlled attribution for an event or interval; see [data requirements](#data-required-for-a-defensible-score).
- **Rework:** additional processing of a unit that failed its initial pass; see [step 7](#step-7--compute-quality-and-its-losses).
- **Run time:** planned production time minus availability loss; see [step 4](#step-4--measure-availability-loss).
- **Speed residual:** performance loss not explained by separately verified microstops; see [the loss tree](#build-the-loss-tree).
- **State interval:** elapsed period assigned one authoritative operating state; see [step 3](#step-3--build-a-non-overlapping-state-timeline).
- **Stop threshold:** rule that distinguishes an availability stop from a performance microstop; see [step 4](#step-4--measure-availability-loss).
- **System constraint:** resource or condition limiting throughput in the relevant period; see [assumptions and limits](#11-assumptions-and-limits).
- **Total count:** produced units at the defined count point, including rejects; see [step 5](#step-5--establish-ideal-cycle-time-and-count-definitions).

## 16. Sources

1. ISO. [ISO 22400-2:2014, *Automation systems and integration — Key performance indicators (KPIs) for manufacturing operations management — Part 2: Definitions and descriptions*](https://www.iso.org/standard/54497.html). Standard reference for manufacturing operations indicators; consult the current edition and licensed text for conformance.
2. IEC. [IEC 62264-1, *Enterprise-control system integration — Part 1: Models and terminology*](https://webstore.iec.ch/en/publication/6675). Reference for integration terminology and models; it does not validate local event data.
3. ISO. [ISO 9001:2015, *Quality management systems — Requirements*](https://www.iso.org/standard/62085.html). Quality-management context; applicability depends on the organization and its obligations.
4. Muchiri, Peter, and Liliane Pintelon. “Performance measurement using overall equipment effectiveness (OEE): literature review and practical application discussion.” *International Journal of Production Research* 46, no. 13 (2008): 3517–3535. [https://doi.org/10.1080/00207540601142645](https://doi.org/10.1080/00207540601142645). Peer-reviewed review and practical discussion.
5. Scott, D., and R. Pisa. “Can overall factory effectiveness prolong Moore's law?” *Solid State Technology* 41, no. 3 (1998): 75–82. Factory-effectiveness discussion in semiconductor manufacturing.
6. Braglia, Marcello, Marco Frosolini, and Francesco Zammori. “Overall equipment effectiveness of a manufacturing line (OEEML): an integrated approach to assess systems performance.” *Journal of Manufacturing Technology Management* 20, no. 1 (2009): 8–29. [https://doi.org/10.1108/17410380910925389](https://doi.org/10.1108/17410380910925389). Peer-reviewed line-level measurement discussion.
7. Chikwendu, Okpala Charles, Anozie Stephen Chima, and Mgbemena Chika Edith. “The optimization of overall equipment effectiveness factors in a pharmaceutical company.” *Heliyon* 6, no. 4 (2020): e03796. [https://doi.org/10.1016/j.heliyon.2020.e03796](https://doi.org/10.1016/j.heliyon.2020.e03796). Case study of a tablet compression machine.

The worked plant, costs, counts, and intervention results in this lesson are fictional teaching assumptions.
The cited publications support methods and context, not the fictional results or a universal OEE benchmark.
Publications and standards may be updated; verify the applicable edition and requirements at the time of use.

## 17. Links

- [Module 15 index](_index.md) — topic sequence and scope.
- [Manufacturing and production planning survey](../../../course/15_manufacturing_production_planning.md) — module-level narrative and survey examples.
- [Manufacturing master data](02_manufacturing-master-data.md) — asset, routing, and effective-dated master data.
- [Capacity requirements](05_capacity-requirements.md) — translate resource time into dated planning capacity without double derating.
- [Finite scheduling](06_finite-scheduling.md) — sequence and resource feasibility after an intervention.
- [Production planning labs survey](../../../course/30_simulation_optimization_production_labs.md) — simulation and optimization exercises for production systems.
