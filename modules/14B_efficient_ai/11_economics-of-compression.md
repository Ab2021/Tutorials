# Economics of Compression

> Module 14B — Efficient AI and On-Device Inference
> Topic 11 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08, 09
> Version 2026.3

## 1. Why this topic exists

Compression is usually justified as a technical necessity: the model does not fit, the latency is too high, the battery is too small. It is also an economic decision, and the economic case is often the one that determines whether the product ships.

The economics are not obvious. A compressed model can reduce the bill of materials (BOM) by allowing a smaller part, or increase it by requiring a part with an accelerator. It can remove a cloud bill, or add a validation bill. It can extend battery life, or force a larger battery. It can shorten the time to market, or delay it with a requalification.

Three industry problems follow.

**The BOM-only analysis.** A team compares the cost of two chips and picks the cheaper one. The cheaper chip needs a larger battery, a bigger enclosure, and a longer validation because the model does not fit. The total cost is higher, and the analysis missed it because it stopped at the BOM.

**The cloud-is-free assumption.** A team assumes that cloud inference has no marginal cost because the servers are already running. At 144 inferences per day per device and 50,000 devices, the marginal cost is real, and it recurs for the life of the product.

**The requalification surprise.** A model change that costs 20,000 USD in engineering triggers a 120,000 USD validation and regulatory package. The change is economically negative, and nobody calculated it before starting.

This topic teaches the total-cost model: variable cost per unit, fixed cost, break-even volume, battery-life arithmetic, requalification cost, and the sensitivity analysis that shows which assumption the decision depends on.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the cost boundary for an on-device versus cloud decision.
2. **Compute** variable cost per unit, including cloud inference and connectivity over the product life.
3. **Compute** fixed cost, including non-recurring engineering (NRE) and requalification.
4. **Derive** the break-even volume and interpret it against the sales forecast.
5. **Model** battery life from a duty-cycle current budget and a derated capacity.
6. **Include** the cost of a model change in the lifecycle cost.
7. **Run** a sensitivity analysis on volume, cloud price, and requalification cost.
8. **Write** an economic decision memo with assumptions and residual risk.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A cost boundary table |
| 2 | Exercise 2 | A variable-cost calculation |
| 3 | Exercise 3 | A fixed-cost table |
| 4 | Exercise 4 and Section 6 | A break-even derivation |
| 5 | Exercise 5 | A battery-life calculation |
| 6 | Exercise 6 | A lifecycle cost with requalification |
| 7 | Exercise 7 | A sensitivity table |
| 8 | Exercise 8 | A memo with assumptions |

## 3. The core mental model

The decision is between two cost structures. One has low fixed cost and high variable cost; the other has high fixed cost and low variable cost.

```text
   cost per unit
        ▲
        │  cloud: low fixed, high variable
        │  ●
        │   ●
        │    ●
        │     ●───────────────  device: high fixed, low variable
        │      ●
        │       ●
        │        ●
        └────────┴──────────────────────────► volume
                 ▲
            break-even
```

The break-even volume is where the two lines cross. Below it, the low-fixed option is cheaper per unit; above it, the high-fixed option is.

### 3.1 The cost boundary

A cost model is only as good as its boundary. The boundary decides which costs are inside the comparison and which are outside.

| Cost | Include when comparing device vs cloud | Notes |
|---|---|---|
| Device BOM | Yes | Silicon, memory, battery, enclosure |
| Cloud inference | Yes | Per request, over the product life |
| Connectivity | Yes | Cellular or Wi-Fi subscription, per device |
| NRE | Yes | Firmware, model, toolchain, validation |
| Requalification | Yes | Per model change, amortized |
| Manufacturing test | Yes | Per unit |
| Field service | Yes | Per failure, expected value |
| Recall risk | As expected value | Probability times cost |
| Engineering time | Yes | Opportunity cost |
| Energy | Small for a coin cell | Large for an edge box |
| Latency cost | Clinical or commercial | Hard to monetize; state it |

A comparison that includes the BOM but not the cloud bill, or the NRE but not the requalification, is not a comparison; it is a preference with numbers.

### 3.2 Variable and fixed cost

**Variable cost** scales with units: BOM, manufacturing test, cloud inference per device, connectivity, packaging.

**Fixed cost** does not scale with units: NRE, tooling, certification, the validation that is performed once per model version.

The per-unit cost at volume N is:

```text
cost_per_unit(N) = variable_cost + fixed_cost / N
```

The break-even between two options A and B is where their per-unit costs are equal:

```text
variable_A + fixed_A / N = variable_B + fixed_B / N
N = (fixed_B - fixed_A) / (variable_A - variable_B)
```

The formula requires that the option with higher fixed cost also has lower variable cost. If it does not, one option dominates and there is no break-even.

### 3.3 Battery life is a cost driver

Battery life is not only a user experience; it is a cost. A device that lasts 7 days needs a 235 mAh cell; a device that lasts 2 days needs a 700 mAh cell, which is larger, heavier, and more expensive. A device that lasts 30 days may use a smaller cell or a cheaper chemistry.

The life calculation:

```text
average_current = Σ (current_i × duty_i)
life_hours = derated_capacity_mAh / average_current_mA
```

The derating accounts for temperature, pulse drain, and end-of-life voltage. A CR2032 rated 235 mAh at 21 °C delivers about 66% of that at 0 °C, and less under high pulse currents.

### 3.4 Requalification is part of the model

A model change is not free after launch. It requires:

- Revalidation on the target.
- A regulatory assessment, possibly a submission.
- Quality review and document updates.
- Manufacturing and service updates.

The cost per change can be comparable to the original validation. It must be amortized over the units affected and included in the lifecycle cost.

| Change type | Typical cost drivers |
|---|---|
| Quantization within a PCCP | Validation, document update |
| Distillation | Validation, regulatory review, possibly a submission |
| Architecture change | Full validation, new submission |
| Threshold change | Validation, labeling update |

The numbers are organization-specific; the categories are not.

### 3.5 The decision is not only cost

Cost is one dimension of the decision. Others are:

- **Latency:** cloud inference adds network latency, which may violate the clinical deadline.
- **Privacy:** on-device inference keeps raw data local, which may be a regulatory or commercial advantage.
- **Reliability:** cloud inference depends on connectivity, which may be absent.
- **Flexibility:** cloud models can be updated centrally; device models need an update path.
- **Risk:** a device model that fails in the field is harder to fix than a cloud model.

A cost model that ignores these dimensions will recommend the cheaper option even when it is not deployable. The memo should state the non-cost constraints and their effect on the decision.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| BOM | Bill of materials; per-unit component cost | Confused with retail price | USD/unit |
| NRE | Non-recurring engineering; one-time design cost | Amortized incorrectly | USD |
| Variable cost | Cost that scales with units | Confused with fixed cost | USD/unit |
| Fixed cost | Cost that does not scale with units | Assumed to be zero at volume | USD |
| Break-even | Volume at which two options cost the same per unit | Confused with profitability | Units |
| Gross margin | Revenue minus cost of goods | Confused with net margin | USD/unit |
| TCO | Total cost of ownership | Confused with purchase price | USD |
| Requalification | Revalidation after a change | Confused with recertification | USD |
| Amortization | Spreading a fixed cost over units | Confused with depreciation | USD/unit |
| Duty cycle | Fraction of time a function is active | Confused with utilization | Proportion |
| Average current | Time-weighted current | Confused with peak current | mA |
| Capacity | Charge a battery can deliver | Confused with energy | mAh |
| Derating | Reduction from nameplate capacity | Ignored | Proportion |
| Self-discharge | Capacity lost while idle | Ignored in long-life products | %/year |
| Pulse drain | Capacity loss under high-current pulses | Ignored | Proportion |
| Cloud inference cost | Cost per request | Assumed zero | USD/request |
| Connectivity cost | Subscription per device | Ignored | USD/month |
| Latency cost | Commercial or clinical cost of delay | Monetized arbitrarily | USD |
| Field failure cost | Expected cost of a failure | Ignored | USD/unit |
| Recall cost | Cost of a field action | Ignored in risk analysis | USD |
| Opportunity cost | Value of the next best use of resources | Ignored | USD |
| Volume | Units sold over the product life | Confused with annual units | Units |
| Product life | Years in market | Confused with warranty | Years |
| Sensitivity analysis | Effect of changing one assumption | Confused with scenario analysis | Analysis |
| Scenario analysis | Effect of changing several assumptions together | Confused with sensitivity | Analysis |
| Payback period | Time to recover an investment | Confused with break-even | Months |
| Net present value | Discounted future cash flows | Ignored in simple models | USD |
| Discount rate | Rate used to discount future cash | Assumed zero | Proportion |
| Cost of goods sold | Direct cost of producing a unit | Confused with operating cost | USD/unit |
| Manufacturing test | Per-unit test cost | Ignored | USD/unit |
| Yield | Fraction of units that pass test | Ignored | Proportion |
| Warranty | Cost of replacement under warranty | Ignored | USD/unit |
| Regulatory cost | Cost of submissions and maintenance | Ignored | USD |

Two definitions matter most.

**Break-even is not profitability.** It is the volume at which two options have the same cost per unit. A product can break even on the manufacturing choice and still lose money on price, margin, or volume.

**Amortization is not a cash flow.** Spreading a fixed cost over units is an accounting convenience. The cash is spent before the units are sold, and a product that sells fewer units than planned does not recover it.

## 5. The mechanics, step by step

The procedure produces an economic comparison and a decision memo. It assumes a defined product, a sales forecast, and two or more technical options.

### Step 1. Define the cost boundary and the product life

**Input:** The product plan, the clinical use, and the options.

**Output:** A boundary statement: which costs are included, over what life, at what volume.

**Rationale:** The boundary determines the answer. A comparison over one year favors the low-fixed option; a comparison over five years may favor the high-fixed one.

The statement includes:

- Product life in years.
- Expected units per year and total.
- Geographic markets and their connectivity costs.
- Warranty period.
- Model update cadence.

**Branch point:** If the product life or volume is unknown, the comparison is a scenario, not a decision. Present it as a range.

**Edge case:** A product with a 10-year life and a 2-year model cadence has five update cycles. The requalification cost is five times the per-change cost, not once.

### Step 2. Estimate variable cost per unit

**Input:** The BOM, the manufacturing test, the cloud price, and the connectivity price.

**Output:** A variable cost for each option, per unit, over the product life.

**Rationale:** Variable cost is the number that scales. It is also the number most often underestimated, because cloud and connectivity costs recur.

For a cloud option:

```text
variable_cloud = BOM + manufacturing_test
               + inferences_per_unit × price_per_inference
               + connectivity_per_month × months
```

For a device option:

```text
variable_device = BOM + manufacturing_test
```

**Branch point:** If the cloud price is tiered or volume-discounted, use the tier the product will actually reach. A price at 1 million requests is not the price at 1,000.

**Edge case:** A device that sends only events, not raw data, has a much lower connectivity cost than one that streams. The data architecture is an economic decision.

### Step 3. Estimate fixed cost

**Input:** The engineering plan, the validation plan, and the regulatory plan.

**Output:** A fixed cost for each option.

**Rationale:** Fixed cost is where the options differ most. A cloud integration is usually cheaper to build than a device firmware and validation programme, but it may have higher recurring costs.

The fixed costs:

- Firmware and runtime engineering.
- Model development and compression.
- Toolchain and licence costs.
- Validation on target.
- Regulatory submission or PCCP preparation.
- Manufacturing tooling and test development.
- Certification.

**Branch point:** If a fixed cost is shared with another product, allocate it explicitly. An unallocated shared cost makes both options look cheaper than they are.

**Edge case:** A PCCP reduces the per-change fixed cost but adds a fixed cost to prepare the plan. The plan is worth it if the product will change more than once.

### Step 4. Model battery life and its cost consequence

**Input:** The duty-cycle current budget and the battery capacity.

**Output:** Battery life in days, and the battery cost or size consequence.

**Rationale:** Battery life determines the cell, the enclosure, and the user experience. A shorter life can force a larger cell, which changes the BOM.

The calculation:

```text
average_current = Σ (current_i × time_i) / cycle_time
life_days = derated_capacity_mAh / average_current_mA / 24
```

**Branch point:** If the calculated life is below the product requirement, the options are a larger cell, a lower-power design, or a change in the clinical duty cycle. Each has a cost.

**Edge case:** A device that transmits continuously may be limited by the radio, not the model. Compressing the model does not fix a radio-dominated current budget.

### Step 5. Derive the break-even volume

**Input:** The variable and fixed costs of two options.

**Output:** The volume at which their per-unit costs are equal.

**Rationale:** The break-even converts the cost structure into a decision rule: below it, choose the low-fixed option; above it, choose the low-variable option.

```text
N_break_even = (fixed_high - fixed_low) / (variable_low - variable_high)
```

where "high" and "low" refer to fixed cost.

**Branch point:** If the denominator is negative, the option with higher fixed cost also has higher variable cost. It is dominated; there is no break-even.

**Edge case:** A break-even below the minimum viable volume is not a decision; both options are viable and other factors decide. A break-even above the forecast is a warning that the high-fixed option will not pay back.

### Step 6. Include requalification and lifecycle cost

**Input:** The model update cadence and the per-change cost.

**Output:** A lifecycle fixed cost that includes the updates.

**Rationale:** A model that changes every year has a recurring fixed cost. Ignoring it makes the device option look cheaper than it is.

```text
fixed_device_total = NRE + changes × cost_per_change
```

**Branch point:** If the changes are covered by a PCCP, the per-change cost is lower. If they are not, each change may require a submission.

**Edge case:** A change that is reverted still consumes the validation cost. The cadence should count attempts, not only releases.

### Step 7. Run the sensitivity analysis

**Input:** The cost model and the uncertain assumptions.

**Output:** A table showing the break-even under different assumptions.

**Rationale:** The decision depends on a few numbers, and those numbers are estimates. The sensitivity analysis shows which estimate matters.

Vary:

- Volume (forecast, half, double).
- Cloud price per inference.
- Requalification cost.
- Battery capacity and derating.
- Product life.

**Branch point:** If the decision flips under a plausible change in one assumption, the decision is not robust. Either gather better data or choose the option that is robust across the range.

**Edge case:** A sensitivity analysis that varies one assumption at a time can miss interactions. A scenario analysis that varies several together is a useful supplement.

### Step 8. Compare the total cost of ownership

**Input:** The per-unit cost at the forecast volume, including fixed and lifecycle costs.

**Output:** A TCO table for each option.

**Rationale:** The TCO is the number that compares with the price. A product with a lower BOM but a higher TCO may have a lower margin.

```text
TCO_per_unit = variable + fixed_total / N
```

**Branch point:** If the TCO exceeds the target cost, the product is not viable at that volume. The options are to raise the price, reduce the cost, or increase the volume.

**Edge case:** A TCO that excludes the cost of a field failure or a recall is optimistic. Include an expected value for both.

### Step 9. Decide with the non-cost constraints

**Input:** The TCO, the latency, privacy, reliability, and flexibility requirements.

**Output:** A decision with a rationale.

**Rationale:** The cheapest option is not always deployable. A cloud option that misses the latency deadline is not an option at any price.

The decision matrix:

| Criterion | Weight | Cloud | Device | Notes |
|---|---|---:|---:|---|
| TCO at forecast | 40% | | | From Step 8 |
| Latency | 20% | | | Clinical deadline |
| Privacy | 15% | | | Raw data locality |
| Reliability | 10% | | | Connectivity dependence |
| Flexibility | 10% | | | Update cadence |
| Risk | 5% | | | Field failure cost |

**Branch point:** If a hard constraint eliminates an option, the weights are irrelevant. Apply the hard constraints first.

**Edge case:** A hybrid option (on-device triage, cloud confirmation) can dominate both pure options, at the cost of complexity. Evaluate it as a third option.

### Step 10. Record the assumptions and monitor the actuals

**Input:** The model, the decision, and the actual costs after launch.

**Output:** A cost record and a review cadence.

**Rationale:** The model is a set of assumptions. Recording them makes it possible to learn whether they were right.

The record includes:

- The boundary and the product life.
- The variable and fixed costs, with sources.
- The volume forecast.
- The break-even and the TCO.
- The sensitivity table.
- The non-cost constraints.
- The decision and the owner.

**Branch point:** If the actual cloud price or volume differs from the assumption by more than 20%, revisit the decision. The model is a living document, not a one-time calculation.

**Edge case:** A cost model that is not updated after launch becomes a historical artifact. The review cadence is what keeps it useful.

### 5.1 Decision table for the build-versus-buy decision

| Situation | Likely choice | Reason |
|---|---|---|
| Volume below break-even | Cloud | Low fixed cost |
| Volume above break-even | Device | Low variable cost |
| Latency-critical | Device | No network round trip |
| Privacy-critical | Device | Raw data stays local |
| Connectivity unreliable | Device | No network dependency |
| Frequent model updates | Cloud, or device with OTA | Update cost |
| Regulated with narrow PCCP | Device with a planned update path | Change control |
| Very low volume, research use | Cloud | No tooling cost |

### 5.2 Edge cases in the economics

**Outlier channels.** A model with outlier channels may need per-channel quantization, which adds metadata but avoids a larger model. The economic effect is negligible at the model scale, but the accuracy effect can avoid a requalification.

**Per-tensor versus per-channel scales.** Per-channel scales can recover accuracy that would otherwise force a larger model or a new submission. The cost of the metadata is trivial compared with the cost of a failed validation.

**Activation clipping.** Clipping can reduce the model size and the memory, but it can also change the calibration and trigger a validation. The economic benefit must exceed the validation cost.

**Operator fallback.** A fallback can force a larger part or a longer validation. The cost of replacing the operator is often lower than the cost of the fallback in production.

**Unsupported operators.** A model that requires a new runtime or a new toolchain has a fixed cost that is easy to underestimate. The toolchain licence and the engineering time are part of the NRE.

**Zero-point drift.** A drift that requires a recalibration is a model change with a validation cost. The monitoring that detects it is cheaper than the recall that follows it.

**Calibration-set bias.** A calibration set that under-represents a subgroup can cause a post-market finding, which is a cost. The stratified calibration is cheaper.

**Thermal throttling.** A device that throttles may need a larger battery or a better enclosure, both of which are BOM costs. The thermal design is part of the economic model.

## 6. Worked example

This is a constructed case. The costs are illustrative and chosen so that the arithmetic closes.

### 6.1 The two options

**Option A: cloud inference.** The patch streams a 4-second window every 2 seconds to a phone, which forwards it to a cloud service. The device BOM is 12 USD. The cloud service charges 0.00042 USD per inference. Connectivity costs 0.50 USD per device per year. Integration NRE is 80,000 USD.

**Option B: on-device inference.** The patch runs the INT8 model locally and sends only alerts. The device BOM is 18 USD, which includes the extra flash and SRAM. There is no cloud inference cost and no connectivity subscription. Firmware and validation NRE is 250,000 USD. One model change is expected in three years, at 120,000 USD.

The product life is three years, the volume forecast is 50,000 units, and the device runs 144 inferences per day.

### 6.2 Scale and zero point by hand

The on-device model's first convolution has 40 weights with a minimum of -0.061 and a maximum of 0.412.

```text
Affine: S = 0.473 / 255 = 0.00185490; Z = 33
q(0.412) = 255; q(-0.061) = 0

Symmetric: S = 0.412 / 127 = 0.00324409; Z = 0
```

The quantization is what makes the 41,816-byte model fit the 18 USD BOM. Without it, the FP32 model is 164,864 bytes, which requires a larger flash part and adds roughly 4 USD to the BOM in this constructed case. The compression is worth about 4 USD per unit, or 200,000 USD at 50,000 units, which is comparable to the NRE.

That comparison is the economic case for compression in one line: the memory saved pays for the engineering that produced it, at volume.

### 6.3 Memory and energy

| Quantity | FP32 | INT8 | Change |
|---|---:|---:|---:|
| Weight bytes | 164,864 | 41,816 | -74.6% |
| KiB | 161.0 | 40.8 | -120.2 |
| Energy per inference (mJ) | 3.395 | 1.124 | -66.9% |
| p99 latency (ms) | 138.0 | 45.7 | -66.9% |

### 6.4 Battery life

The on-device cycle energy is 5.039 mJ per 2 seconds, which is an average current of 0.840 mA, plus 3 µA of sleep current:

```text
average_current = 0.843 mA
ideal_life = 235 mAh / 0.843 mA / 24 = 11.6 days
derated_life = 11.6 × 0.66 = 7.7 days
```

The cloud option streams continuously. The radio draws 12 mA for 0.16 seconds per 2-second cycle, which is 0.96 mA average, and the sensor and microcontroller add 3.24 mA, for 4.2 mA total:

```text
ideal_life = 235 / 4.2 / 24 = 2.33 days
derated_life = 2.33 × 0.66 = 1.54 days
```

The on-device option lasts 5.0 times longer. In product terms, that is the difference between a 7-day patch and a 1.5-day patch, which may be the difference between a viable product and one that requires a larger battery.

### 6.5 Cloud cost per unit

```text
inferences = 144/day × 365 days × 3 years = 157,680
cloud_inference = 157,680 × 0.00042 = 66.23 USD
connectivity = 0.50 × 3 = 1.50 USD
variable_cloud = 12 (BOM) + 66.23 + 1.50 = 79.73 USD
variable_device = 18 USD
```

The cloud variable cost is 4.4 times the device variable cost over the product life. The BOM difference of 6 USD is small compared with the recurring cloud and connectivity cost.

### 6.6 Break-even derivation

```text
fixed_cloud = 80,000
fixed_device = 250,000 + 120,000 = 370,000
variable_cloud = 79.73
variable_device = 18

N = (370,000 - 80,000) / (79.73 - 18)
  = 290,000 / 61.73
  = 4,698 units
```

At 4,698 units, the two options have the same TCO per unit. Above that volume, the device option is cheaper; below it, the cloud option is.

The calculation can be checked by evaluating both at the break-even:

```text
cloud: 80,000/4,698 + 79.73 = 17.03 + 79.73 = 96.76 USD
device: 370,000/4,698 + 18 = 78.76 + 18 = 96.76 USD
```

Both equal 96.76 USD, which confirms the derivation.

### 6.7 TCO at different volumes

| Volume | Cloud TCO/unit | Device TCO/unit | Cheaper |
|---:|---:|---:|---|
| 2,000 | 119.73 | 203.00 | Cloud |
| 5,000 | 95.73 | 92.00 | Device |
| 10,000 | 87.73 | 55.00 | Device |
| 50,000 | 81.33 | 25.40 | Device |

At 50,000 units, the device option is 55.9 USD cheaper per unit, or 2.79 million USD over the product life. The fixed cost is recovered many times over.

At 2,000 units, the device option is 83.27 USD more expensive per unit, or 166,540 USD over the life. The fixed cost is not recovered.

### 6.8 Accuracy and calibration

| Metric | Cloud FP32 | Device INT8 | Difference |
|---|---:|---:|---:|
| Sensitivity | 0.942 | 0.938 | -0.4 pp |
| ECE | 0.021 | 0.029 | +0.008 |
| Worst subgroup | 0.931 | 0.926 | -0.5 pp |
| p99 latency | 480 ms | 45.7 ms | -434.3 ms |

The device option is slightly less accurate and much faster. The cloud option's 480 ms includes the network round trip, which is the dominant term and is variable.

The accuracy difference is within the change margin from Topic 09, so it does not trigger a new submission in this constructed case. If it did, the requalification cost would rise, and the break-even would move.

### 6.9 Diagnostic one: outlier-driven scale collapse

The fully connected layer's weights have a median absolute value of 0.0610 and a maximum of 1.2912. The symmetric scale is 0.01016693, or 16.7% of the median.

```text
step / median = 0.1667
weights rounded to zero = 1,411 (4.3%)
distinct levels used = 102 of 255
SQNR = 30.00 dB
```

The economic consequence is indirect: the accuracy loss from scale collapse is what determines whether the compressed model needs a larger architecture or a new validation. Per-channel scales, at 2,560 bytes of metadata, are cheaper than either.

### 6.10 Diagnostic two: per-channel skew

The 512 channels have ranges from 0.2756 to 2.3202, a ratio of 8.4. Per-channel scales raise SQNR from 30.00 to 44.67 dB.

The metadata cost is 2,560 bytes, which is 6.1% of the 41,816-byte model. In BOM terms, 2.5 KiB is within the rounding of the flash part, so the accuracy gain is essentially free. This is a case where the cheapest technical choice is also the best economic one.

### 6.11 Varying one assumption

**Volume: 50,000 to 2,000.** The device TCO rises from 25.40 to 203.00 USD per unit, and the cloud option becomes cheaper. The decision flips at 4,698 units. A product with a 2,000-unit forecast should not build the on-device path, even though it is technically superior.

**Cloud price: 0.00042 to 0.0001 per inference.** The cloud variable cost falls from 79.73 to 29.27 USD, and the break-even rises to 25,737 units. A cheaper cloud makes the device option harder to justify.

**Requalification: 120,000 to 200,000.** The device fixed cost rises to 450,000 USD, and the break-even rises to 5,994 units. A product with a frequent model cadence should count each change.

**Battery: 235 mAh to 150 mAh.** The on-device life falls from 7.7 to 4.9 days, and the cloud life falls from 1.54 to 0.98 days. The ratio is unchanged, but the absolute life may fall below the product requirement, forcing a larger cell and a higher BOM.

**Product life: 3 to 5 years.** The cloud inference cost rises to 110.38 USD over five years, and the device break-even falls. A longer life favors the device option, because the recurring cost compounds while the fixed cost does not.

### 6.12 The decision

At the 50,000-unit forecast, the device option has a TCO of 25.40 USD per unit against 81.33 USD for the cloud option. The device option also meets the latency and privacy requirements that the cloud option cannot. The decision is on-device inference with INT8 quantization.

The decision is sensitive to volume: below 4,698 units, the cloud option is cheaper. It is also sensitive to the cloud price: at 0.0001 USD per inference, the break-even rises to 25,737 units. The forecast of 50,000 units is well above both thresholds, so the decision is robust to plausible changes in those assumptions.

The decision is not sensitive to the compression method's metadata or to the battery derating, which change the TCO by less than 5%.

## 7. Code, query, or calculation

The script computes variable cost, fixed cost, break-even, TCO at several volumes, and battery life. It reproduces Section 6.

```python
def cloud_variable(bom, inferences_per_day, years, price, connectivity_per_year):
    inferences = inferences_per_day * 365 * years
    return {
        "inferences": inferences,
        "cloud": inferences * price,
        "connectivity": connectivity_per_year * years,
        "variable": bom + inferences * price + connectivity_per_year * years,
    }


def break_even(fixed_low, fixed_high, variable_low, variable_high):
    """Option with higher fixed cost has lower variable cost."""
    return (fixed_high - fixed_low) / (variable_low - variable_high)


def tco(variable, fixed, volume):
    return variable + fixed / volume


def battery_life(capacity_mah, average_ma, derate=0.66):
    return capacity_mah / average_ma / 24 * derate
```

Compute the case:

```python
cloud = cloud_variable(12, 144, 3, 0.00042, 0.50)
print("inferences =", cloud["inferences"])
print("cloud cost = %.2f" % cloud["cloud"])
print("connectivity = %.2f" % cloud["connectivity"])
print("variable cloud = %.2f" % cloud["variable"])

fixed_cloud = 80_000
fixed_device = 250_000 + 120_000
variable_device = 18
n = break_even(fixed_cloud, fixed_device, cloud["variable"], variable_device)
print("break-even = %.0f units" % n)

for volume in (2_000, 5_000, 10_000, 50_000):
    print("volume %6d cloud %6.2f device %6.2f"
          % (volume, tco(cloud["variable"], fixed_cloud, volume),
             tco(variable_device, fixed_device, volume)))
```

Expected output:

```text
inferences = 157680
cloud cost = 66.23
connectivity = 1.50
variable cloud = 79.73
break-even = 4698 units
volume   2000 cloud 119.73 device 203.00
volume   5000 cloud  95.73 device  92.00
volume  10000 cloud  87.73 device  55.00
volume  50000 cloud  81.33 device  25.40
```

Battery life:

```python
print("on-device = %.2f days" % battery_life(235, 0.843))
print("cloud = %.2f days" % battery_life(235, 4.2))
```

Expected output:

```text
on-device = 7.67 days
cloud = 1.54 days
```

Sensitivity:

```python
for price in (0.00042, 0.0002, 0.0001):
    v = cloud_variable(12, 144, 3, price, 0.50)
    print("price %.5f variable %.2f break-even %.0f"
          % (price, v["variable"],
             break_even(fixed_cloud, fixed_device, v["variable"], variable_device)))

for requal in (60_000, 120_000, 200_000):
    fd = 250_000 + requal
    print("requal %6d break-even %.0f"
          % (requal, break_even(fixed_cloud, fd, cloud["variable"], variable_device)))
```

Expected output:

```text
price 0.00042 variable 79.73 break-even 4698
price 0.00020 variable 45.04 break-even 10726
price 0.00010 variable 29.27 break-even 25737
requal  60000 break-even 3726
requal 120000 break-even 4698
requal 200000 break-even 5994
```

Quantization arithmetic:

```python
import math


def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    scale = (hi - lo) / (2 ** bits - 1)
    zero_point = min(2 ** bits - 1, max(0, round_half_away(-lo / scale)))
    return scale, zero_point


weights = [-0.061, -0.034, 0.012, 0.021, 0.045, 0.089, 0.412]
s, z = affine_params(weights, 8)
print("affine S=%.8f Z=%d" % (s, z))
print("symmetric S=%.8f" % (max(abs(v) for v in weights) / 127))
print("model fp32 =", 41216 * 4, "int8 =", 41216 + 120 * 5)
```

Expected output:

```text
affine S=0.00185490 Z=33
symmetric S=0.00324409
model fp32 = 164864 int8 = 41816
```

Assumptions: the BOM values are illustrative; the cloud price is constant over the life; the connectivity price is per year; the inference rate is constant; the battery derates to 66% at 0 °C; and the model change occurs once in three years.

## 8. Data, standards, and regulatory context

The economics of compression are shaped by standards and regulations in three ways: they define what must be validated, they define what a change costs, and they constrain the claims that can be made about cost savings.

### 8.1 IEC 62304 and the cost of validation

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79, requires verification and configuration management. Each compression change that alters a verified requirement triggers re-verification, and that is a cost.

The standard does not set a price, but it defines the work: requirements, tests, records, and reviews. A cost model that omits this work underestimates the fixed cost of the device option.

### 8.2 ISO 14971 and the cost of risk

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be controlled. A compressed model that degrades a subgroup creates a risk that must be controlled, monitored, or accepted. Each control has a cost, and the residual risk has an expected cost.

A cost model that treats the risk as zero is not conservative; it is wrong. The expected cost of a field action can exceed the entire NRE.

### 8.3 FDA PCCP and the cost of change

The FDA's final guidance on Predetermined Change Control Plans was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022.

A PCCP changes the economics of updates. Without one, each model change may require a submission; with one, the change can be implemented under the protocol. The plan itself has a cost, which is worth paying if the product will change more than once.

The economic model should therefore include the PCCP preparation as a fixed cost and the per-change cost under the plan, not the cost of a full submission for every change.

### 8.4 EU framework and market access

In the European Union, Regulation (EU) 2017/745 governs the device, and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk systems. The Annex I applicability date was originally 2 August 2027; the AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves it to 2 August 2028. Verify the current date before relying on it.

Market access costs, including conformity assessment and clinical evaluation, are fixed costs that differ by market. A product launched in two markets has roughly twice the fixed cost, which raises the break-even.

### 8.5 Cybersecurity and lifecycle cost

The FDA's cybersecurity guidance and the IEC 62443 series require secure updates, vulnerability monitoring, and an SBOM. These are recurring costs: a vulnerability response is not a one-time activity.

A device with a field-update path has a lower cost of fixing a problem and a higher cost of building the path. The economic model should include both.

### 8.6 What is binding and what is convention

| Item | Status | Economic consequence |
|---|---|---|
| IEC 62304 | Binding where required | Re-verification cost per change |
| ISO 14971 | Binding where required | Risk controls and residual risk |
| FDA PCCP guidance | Non-binding; statute binding | Lower per-change cost if covered |
| EU MDR / AI Act | Binding regulation | Market-access fixed cost |
| IEC 62443 / FDA cybersecurity | Binding or expected | Recurring security cost |
| Cloud price | Commercial | Variable cost |
| BOM quote | Commercial | Variable cost |
| Break-even formula | Convention | Decision aid, not a rule |

### 8.7 The cost record

| Item | Source | Assumption |
|---|---|---|
| BOM | Supplier quote at volume | Volume tier |
| Cloud price | Vendor contract | Constant price |
| Connectivity | Carrier plan | Per device per year |
| NRE | Engineering plan | Hours times rate |
| Requalification | Quality and regulatory plan | Per change |
| Battery capacity | Datasheet | Derating at 0 °C |
| Product life | Product plan | Years |
| Volume | Sales forecast | Units |

## 9. Industry practice and case evidence

### 9.1 What is public

Public data on the economics of on-device inference is sparse. Companies rarely publish BOM costs, cloud contracts, or validation costs. What is public falls into three categories.

- **Component prices** from distributors and vendor announcements, which give the BOM order of magnitude.
- **Cloud pricing** from public price lists, which give the per-request or per-instance cost.
- **Energy and battery data** from datasheets, which give the capacity and derating.

These are inputs, not conclusions. The conclusion depends on the product's volume, life, and regulatory path, which are not public.

### 9.2 Battery evidence

The CR2032 datasheet gives 235 mAh at 21 °C under a light load, with about 66% of that available at 0 °C. Pulse drain can reduce the usable capacity further: a cell that delivers 240 mAh at 0.5 mA continuous may deliver only 175–185 mAh with 30 mA peaks, and about 70 mAh with 80 mA peaks.

The inference current in the case is 8.2 mA, which is below the 30 mA pulse range, so the continuous derating is the relevant one. A device with a radio that draws 80 mA peaks would need the pulse derating, and the battery life would be much shorter.

### 9.3 Cloud pricing evidence

Public cloud price lists give per-request or per-instance prices. The per-request price depends on the instance type, the utilization, and the model size. A price of 0.00042 USD per inference is plausible for a small model on a shared CPU instance, but it is an assumption, not a published number for this workload.

The economic model should use a price from the vendor's list or contract, not a guess. The sensitivity analysis shows how much the conclusion depends on it.

### 9.4 What the evidence supports

| Claim | Status |
|---|---|
| CR2032 delivers about 235 mAh at 21 °C | Datasheet |
| Capacity falls to about 66% at 0 °C | Datasheet |
| Pulse drain reduces usable capacity | Datasheet and application notes |
| Cloud inference has a marginal cost | Commercial fact |
| A PCCP reduces per-change regulatory cost | Inference from the guidance |
| On-device inference is cheaper at high volume | Arithmetic, given the assumptions |
| On-device inference is always cheaper | Not supported; depends on volume and price |
| The case's BOM values are realistic | Constructed; not a quote |

### 9.5 Why medical economics differ

**The validation cost is larger.** A consumer model can be updated with an app release; a medical model needs evidence. The fixed cost of the device option is therefore higher, and the break-even moves up.

**The risk cost is larger.** A field failure can trigger a recall, which is a cost that dominates the BOM. The economic model should include an expected value, even if it is uncertain.

**The privacy value is real.** On-device inference keeps raw data local, which can be a market advantage and a regulatory simplification. That value is hard to quantify but should be stated.

**The latency value is clinical.** A 480 ms cloud round trip may be acceptable for a notification and unacceptable for an alarm. The latency constraint can eliminate an option before the cost model matters.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| BOM-only analysis | Comparison stops at the chip | Recurring costs omitted | Wrong choice | Include TCO |
| Cloud assumed free | "Servers already run" | Marginal cost ignored | Margin erosion | Price per request |
| Volume overestimated | Forecast from a pilot | Fixed cost not recovered | Loss | Sensitivity on volume |
| Requalification omitted | "One-time validation" | Change cadence ignored | Understated fixed cost | Count changes |
| Battery undersized | Life below requirement | Current budget wrong | Larger cell, higher BOM | Duty-cycle model |
| Connectivity cost ignored | "Wi-Fi is free" | Subscription per device | Recurring cost | Include per year |
| Discount rate ignored | Future costs undiscounted | Overstated cloud cost | Bias to device | State the assumption |
| Field failure ignored | Expected value zero | Risk not priced | Understated TCO | Include expected value |
| Shared NRE unallocated | Both options look cheap | Cost hidden | Wrong decision | Allocate explicitly |
| Price tier wrong | Using the 1M-request price | Volume mismatch | Understated cloud cost | Use the actual tier |
| Model change cadence wrong | Annual updates assumed rare | Five changes in five years | Understated fixed cost | Use the plan |
| Non-cost constraints ignored | Cost-only decision | Latency or privacy violation | Unshippable product | Decision matrix |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Volume forecast is accurate | Break-even comparison | Market shift | Sensitivity analysis |
| Cloud price is constant | Variable cost | Price change, contract renewal | Scenario analysis |
| Inference rate is constant | Cloud cost | Clinical use varies | Range of rates |
| Product life is three years | Total cost | Longer or shorter life | Sensitivity |
| BOM quote is at volume | Variable cost | Volume shortfall | Quote at three volumes |
| Requalification cost is known | Fixed cost | First-time change | Estimate range |
| Battery derating is 66% | Life calculation | Colder or pulse-heavy use | Datasheet curves |
| Model change occurs once | Fixed cost | Frequent updates | Cadence plan |
| Connectivity is per year | Variable cost | Bundled or free | Contract terms |
| Discount rate is zero | Simple comparison | Long life, high rate | State and test |
| Field failure cost is zero | TCO | Recall or service | Expected value |
| The options are technically feasible | Comparison | Latency or privacy violation | Engineering review |

The central limit is that the cost model is only as good as its inputs, and most inputs are estimates. The sensitivity analysis is what makes the model useful: it shows which estimate the decision depends on, and therefore which estimate deserves more effort.

A second limit is that the model compares two options at a point in time. A product that starts on the cloud and migrates to the device later has a hybrid cost structure, with the cloud NRE paid first and the device NRE paid later. The hybrid can be the best option, and it needs its own model.

## 12. Exercises

**Exercise 1 — Cost boundary (design, 30 minutes).**
Define the cost boundary for a wearable that can run a model on-device or in the cloud. List the included and excluded costs, with a reason for each exclusion.
*Deliverable:* a two-column table.
*Acceptance criteria:* recurring costs included; exclusions justified; product life stated.
*Time:* 30 minutes.

**Exercise 2 — Variable cost (calculation, 35 minutes).**
A device runs 288 inferences per day for four years. The cloud price is 0.0003 USD per inference and connectivity is 0.40 USD per year. The BOM is 15 USD. Compute the cloud variable cost.
*Deliverable:* a calculation.
*Acceptance criteria:* 420,480 inferences; cloud 126.14 USD; connectivity 1.60 USD; variable 142.74 USD.
*Time:* 35 minutes.

**Exercise 3 — Fixed cost (analysis, 30 minutes).**
List five fixed costs for an on-device option and estimate their relative size. State which is most uncertain.
*Deliverable:* a table with ranges.
*Acceptance criteria:* NRE, validation, regulatory, tooling, certification; the most uncertain is identified with a reason.
*Time:* 30 minutes.

**Exercise 4 — Break-even (calculation, 40 minutes).**
Fixed costs are 60,000 USD for cloud and 300,000 USD for device. Variable costs are 70 USD and 20 USD. Compute the break-even and the TCO at 3,000 and 20,000 units.
*Deliverable:* a derivation and a table.
*Acceptance criteria:* break-even 4,800 units; TCO correct at both volumes; interpretation stated.
*Time:* 40 minutes.

**Exercise 5 — Battery life (calculation, 30 minutes).**
A device draws 1.2 mA average from a 150 mAh cell, derated to 70%. Compute the life in days. State whether a 7-day requirement is met.
*Deliverable:* a calculation.
*Acceptance criteria:* 3.65 days; requirement not met; options stated.
*Time:* 30 minutes.

**Exercise 6 — Lifecycle cost (analysis, 35 minutes).**
A device option has 250,000 USD NRE and a model change every two years at 100,000 USD. Over six years at 30,000 units, compute the fixed cost per unit.
*Deliverable:* a calculation.
*Acceptance criteria:* three changes; fixed total 550,000 USD; 18.33 USD per unit.
*Time:* 35 minutes.

**Exercise 7 — Sensitivity (analysis, 40 minutes).**
Using the Section 6 case, compute the break-even at cloud prices of 0.0005, 0.0003, and 0.0001 USD. State the decision at 10,000 units for each.
*Deliverable:* a table and a conclusion.
*Acceptance criteria:* break-evens correct; decision consistent with volume; the conclusion names the price at which the decision flips.
*Time:* 40 minutes.

**Exercise 8 — Memo (design, 50 minutes).**
Write a one-page economic decision memo for the Section 6 case, including the boundary, the break-even, the sensitivity, and the non-cost constraints.
*Deliverable:* a one-page memo.
*Acceptance criteria:* assumptions listed; break-even derived; non-cost constraints applied; decision with residual risk.
*Time:* 50 minutes.

## 13. Self-check answers

**A1. Variable cost for Exercise 2.**
Inferences = 288 × 365 × 4 = 420,480. Cloud = 420,480 × 0.0003 = 126.144 USD. Connectivity = 0.40 × 4 = 1.60 USD. Variable = 15 + 126.144 + 1.60 = 142.744 USD.

**A2. Break-even for Exercise 4.**
N = (300,000 - 60,000) / (70 - 20) = 240,000 / 50 = 4,800 units. At 3,000: cloud = 60,000/3,000 + 70 = 90 USD; device = 300,000/3,000 + 20 = 120 USD. At 20,000: cloud = 3 + 70 = 73 USD; device = 15 + 20 = 35 USD.

**A3. Battery life for Exercise 5.**
150 mAh × 0.70 = 105 mAh usable. 105 / 1.2 = 87.5 h = 3.65 days. The 7-day requirement is not met; options are a larger cell, a lower average current, or a shorter duty cycle.

**A4. Lifecycle fixed cost for Exercise 6.**
Six years with a change every two years is three changes. Fixed total = 250,000 + 3 × 100,000 = 550,000 USD. Per unit at 30,000 = 18.33 USD.

**A5. Why the break-even formula fails when the denominator is negative.**
If the option with higher fixed cost also has higher variable cost, it is more expensive at every volume. There is no volume at which it becomes cheaper, so there is no break-even.

**A6. Why the cloud option can be cheaper at low volume.**
Because its fixed cost is lower. At low volume, the fixed cost dominates the per-unit cost, and the option with the smaller fixed cost wins even if its variable cost is higher.

**A7. Why requalification belongs in the model.**
Because a model change after launch requires validation and regulatory work, and that work is a fixed cost per change. A device option with frequent changes has a recurring fixed cost that the cloud option may not have, because the cloud model can be updated centrally.

**A8. Why battery life is an economic variable.**
Because it determines the cell size, the enclosure, and the user experience. A shorter life may require a larger cell, which changes the BOM and the industrial design.

**A9. Rubric for Exercise 1 (cost boundary).**
A strong boundary includes BOM, cloud, connectivity, NRE, requalification, manufacturing test, and field failure, and justifies each exclusion. A weak boundary includes only the BOM and the cloud price.

**A10. Rubric for Exercise 8 (memo).**
A strong memo states the boundary and the life, derives the break-even, shows the sensitivity, applies the non-cost constraints, and states the residual risk. A weak memo gives a cost comparison and a recommendation without assumptions.

## 14. Interview questions

**Q1. How do you decide between on-device and cloud inference?**
A strong answer compares TCO at the forecast volume, applies latency and privacy constraints, and derives the break-even. A weak answer compares BOM only.

**Q2. What is break-even volume?**
A strong answer defines it as the volume where two options have equal per-unit cost and gives the formula. A weak answer calls it the profitability point.

**Q3. Why is the cloud not free at the margin?**
A strong answer explains per-request pricing and recurring connectivity, and multiplies by the device fleet and life. A weak answer assumes idle servers are free.

**Q4. How do you calculate battery life?**
A strong answer uses a duty-cycle current budget, derated capacity, and states the temperature assumption. A weak answer divides capacity by peak current.

**Q5. Why does requalification matter economically?**
A strong answer says each model change has a validation and regulatory cost that recurs, and that the device option may have more changes than the cloud. A weak answer treats validation as one-time.

**Q6. What makes a cost model robust?**
A strong answer says a sensitivity analysis on volume, price, and requalification, and a decision that holds across the plausible range. A weak answer says a precise estimate.

**Q7. When is the cheaper option not deployable?**
A strong answer names latency, privacy, reliability, or regulatory constraints that eliminate it. A weak answer assumes cost is the only criterion.

**Q8. How does compression change the economics?**
A strong answer says it can reduce the BOM by fitting a smaller part, reduce energy and battery size, and avoid a larger model's validation, but it can also add a requalification cost. A weak answer says it always saves money.

**Q9. What is the TCO?**
A strong answer defines it as variable plus fixed per unit, including lifecycle and risk costs. A weak answer confuses it with the purchase price.

**Q10. How do you handle uncertainty in the volume forecast?**
A strong answer uses a range, computes the break-even, and states which side of it the forecast must be on. A weak answer uses a single number.

**Q11. Why is a PCCP an economic decision?**
A strong answer says it has a preparation cost and reduces the per-change cost, so it pays back if the product changes more than once. A weak answer treats it as a regulatory formality.

**Q12. What would you reject in an economic analysis?**
A strong answer rejects a BOM-only comparison, an unstated cloud price, an omitted requalification cost, a battery life without derating, and a single-point forecast without sensitivity. A weak answer rejects only an arithmetic error.

## 15. Key terms

- **Amortization** — spreading a fixed cost over units.
- **Average current** — time-weighted current.
- **Battery capacity** — charge a cell can deliver, in mAh.
- **BOM** — bill of materials.
- **Break-even** — volume at which two options cost the same per unit.
- **Capacity derating** — reduction from nameplate capacity.
- **Cloud inference cost** — cost per request.
- **Connectivity cost** — subscription per device.
- **Cost of goods sold** — direct cost of producing a unit.
- **Duty cycle** — fraction of time a function is active.
- **Expected value** — probability times cost.
- **Field failure cost** — expected cost of a failure.
- **Fixed cost** — cost that does not scale with units.
- **Gross margin** — revenue minus cost of goods.
- **Manufacturing test** — per-unit test cost.
- **NRE** — non-recurring engineering.
- **Opportunity cost** — value of the next best use of resources.
- **Payback period** — time to recover an investment.
- **Product life** — years in market.
- **Pulse drain** — capacity loss under high-current pulses.
- **Recall cost** — cost of a field action.
- **Requalification** — revalidation after a change.
- **Sensitivity analysis** — effect of changing one assumption.
- **Self-discharge** — capacity lost while idle.
- **TCO** — total cost of ownership.
- **Variable cost** — cost that scales with units.
- **Volume** — units sold over the product life.
- **Warranty** — cost of replacement under warranty.
- **Yield** — fraction of units that pass test.

## 16. Sources

All URLs accessed 25 October 2026.

**Battery and energy**
- Energizer, CR2032 lithium coin datasheet and Lithium Coin Handbook. https://data.energizer.com/pdfs/cr2032.pdf
- Texas Instruments, "Coin Cells and Peak Current Draw", Application Note SWRA349. https://www.ti.com/lit/wp/swra349/swra349.pdf
- Analog Devices, "Power Supply Design Considerations for Primary Cell Powered Remote Patient Monitors". https://www.analog.com/en/resources/technical-articles/primary-cell-powered-remote-patient-monitors.html

**Hardware and performance**
- NVIDIA, "NVIDIA H100 GPU Datasheet". https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306
- Arm, "Ethos-U85" product page. https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85
- MLCommons, "MLPerf Tiny v1.2 Results", April 2024. https://mlcommons.org/2024/04/mlperf-tiny-v1-2-results/

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- FDA, "Cybersecurity in Medical Devices". https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

**Quantization**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877

> **Note:** The Section 6 costs are constructed for teaching and are not quotes. Battery figures are datasheet values under stated conditions; cloud prices are assumptions.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`10_validation-on-target-hardware.md`](10_validation-on-target-hardware.md)
- Related topic: [`01_the-deployment-pareto-frontier.md`](01_the-deployment-pareto-frontier.md)
- Related topic: [`08_on-device-architecture.md`](08_on-device-architecture.md)
- Related topic: [`09_medical-and-regulatory-constraints.md`](09_medical-and-regulatory-constraints.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
