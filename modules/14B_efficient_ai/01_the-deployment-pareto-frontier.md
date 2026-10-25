# The Deployment Pareto Frontier

> Module 14B — Efficient AI and On-Device Inference
> Topic 01 of 11
> Estimated study time: 3 hours
> Prerequisites: 07, 14A
> Version 2026.3

## 1. Why this topic exists

A model that runs in a notebook has no hardware constraints. A model inside a wearable, a bedside monitor, an infusion pump, or a handheld ultrasound probe has five at once: accuracy, calibration, latency, memory, and energy. Add cost and the regulatory claim, and the design space becomes a constrained optimization problem that most teams never write down.

The cost of not writing it down is predictable. A team trains a model, measures area under the receiver operating characteristic curve (AUROC), quantizes to INT8, and ships. Six weeks before launch they discover that the device has 256 KiB of static random-access memory (SRAM), that the model needs 380 KiB of activations, that the p99 latency is 410 ms against a 250 ms clinical deadline, and that the INT8 conversion shifted the operating threshold enough to move sensitivity by 1.4 percentage points in one subgroup. Every one of those facts was knowable before the first training run. None of them was in the requirements document, because the requirements document said "accurate and fast".

The Pareto frontier is the artifact that prevents this. It is a table and a plot of candidate models scored against every constraint at once, with the dominated candidates removed and the surviving trade-offs made explicit. It converts an argument about which model is "best" into a decision about which constraint the product is willing to trade.

Three industry problems make this urgent.

**The budget is set by the clinical pathway, not the model.** A sepsis alert that reaches a nurse 30 minutes late is a different product from one that reaches the nurse in 90 seconds. A ventricular fibrillation detector that takes 400 ms to confirm has a different safety case from one that takes 40 ms. The model is one term in a sum that includes sensor acquisition, preprocessing, transport, and human response. Optimizing the model term without measuring the others is how teams achieve a 4× model speedup and a 6% end-to-end improvement.

**Constraints are heterogeneous and non-commensurable.** Accuracy is a proportion, calibration is a distance between distributions, latency is a tail quantile, memory is a high-water mark in bytes, energy is joules per inference, and cost is dollars per unit at volume. There is no exchange rate among them. A single "efficiency score" that adds them is a way to hide the judgement, not a way to make it.

**The cheapest feasible model is rarely the smallest.** A 200 KiB model that misses the accuracy floor is worthless at any price. A 4 MiB model that fits in flash but forces a 512 KiB SRAM part adds two dollars to the bill of materials (BOM) and may erase the margin. The frontier includes cost because the product includes cost.

There is also a regulatory dimension that arrives late and expensively. Compression is not a build setting. A quantized model that changes outputs is a changed model, and whether that change is covered by a predetermined change control plan (PCCP) is a question that must be answered before the change is made, not after. Topic 09 covers this in detail. Here it is enough to note that the frontier should carry a column for "regulatory path", because a candidate that cannot be legally shipped is not a candidate.

### 1.1 Five ways the decision goes wrong

The same five failures recur across device programmes, and each has a characteristic signature.

1. **The model-first programme.** A research team selects an architecture by benchmark accuracy, then hands it to an embedded team and asks for it to fit.
   - Signature: the embedded team's first estimate is "we need a bigger part".
   - Cost: a hardware respin or a delayed launch.
   - Root cause: constraints were never written down, so they could not be designed against.
   - Corrective: build the ledger before the architecture search, and make the architecture search a candidate generator inside the ledger.

2. **The single-metric programme.** A product team picks latency as the metric because it is the easiest to demonstrate, and ignores energy until the battery test.
   - Signature: an impressive latency demo on a bench supply, followed by a battery life of three days against a target of fourteen.
   - Cost: a larger battery, a thicker enclosure, and a new industrial design.
   - Root cause: latency and energy are correlated enough to look aligned in short tests and divergent in the field.
   - Corrective: measure energy per inference on every candidate, not only the winner.

3. **The build-setting programme.** Compression is treated as a compiler flag applied at the end, with no change control and no revalidation.
   - Signature: the release candidate is quantized two weeks before submission.
   - Cost: a regulatory question at the worst possible moment, or a post-market finding.
   - Root cause: the team's mental model is "same model, smaller file".
   - Corrective: classify the change at design time and put the evidence in the plan.

4. **The desktop programme.** All measurements are taken in a workstation emulator, and the target is assumed to behave the same way.
   - Signature: the first hardware-in-the-loop run is scheduled after the design freeze.
   - Cost: a rediscovery of operator fallback, thermal throttling, and memory limits in the last month.
   - Root cause: emulators model arithmetic, not memory systems, schedulers, or heat.
   - Corrective: move one candidate to the target in week one, before the architecture is fixed.

5. **The average-case programme.** Accuracy and latency are reported as means over a convenient sample, and the tail and the subgroups are never examined.
   - Signature: "the model is 94% accurate" with no threshold, no confidence interval, and no subgroup table.
   - Cost: a subgroup regression that surfaces in post-market surveillance.
   - Root cause: aggregate metrics are reassuring and cheap; disaggregated metrics are neither.
   - Corrective: pre-specify the subgroup table and the percentile before the first measurement.

The purpose of this topic is to give you the instrument that prevents all five: a constraint ledger, a measurement protocol, a dominance calculation, and a decision memo that a hardware engineer, a clinical lead, and a regulatory reviewer can all read without translating.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a constraint ledger for a device that separates hard constraints from soft preferences, with units and measurement methods for each row.
2. **Decompose** an end-to-end clinical latency budget into sensor, preprocessing, inference, postprocessing, transport, and human terms, and identify which terms dominate.
3. **Compute** memory before and after compression for weights and activations separately, in bytes, with the dtype and the inclusion set stated.
4. **Apply** Pareto dominance to a candidate table and identify the non-dominated set, including near-ties that dominance does not resolve.
5. **Diagnose** two common quantization failures from layer statistics: outlier-driven scale collapse and per-channel skew.
6. **Evaluate** a calibration or accuracy claim against a pre-specified non-inferiority margin rather than a post-hoc comparison.
7. **Choose** a candidate using an explicit decision rule that states the tie-breaker and the fallback.
8. **Write** a one-page deployment decision memo that names the constraints, the evidence, the residual risk, and the change-control path.

The mapping between outcomes and assessment is deliberate, so that no outcome is taught without being practised.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A ledger with units and owners |
| 2 | Exercise 2 | A budget that sums to the deadline |
| 3 | Exercise 3 and Section 7 | Correct bytes with stated inclusion |
| 4 | Exercise 4 | A justified non-dominated set |
| 5 | Exercise 5 | Two diagnostics and a remedy |
| 6 | Exercise 6 | A decision that changes or holds, with mechanism |
| 7 | Exercise 7 | A memo with a rule and a fallback |
| 8 | Exercise 7 and Section 9 | Constraints, evidence, residual risk, change path |

A learner who can produce the deliverable for each row can run a deployment review. A learner who can only recite the definitions cannot.

## 3. The core mental model

The frontier is easiest to hold in mind as a box with a surface inside it.

```text
                     accuracy
                        ▲
                        │            ● A  cloud FP32
                        │
                        │      ● B  edge GPU FP16
      accuracy floor ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                        │   ● E  MCU QAT      ● C  MCU PTQ
                        │       ● D  MCU distilled
                        └──────────────────────────────────► latency / energy / memory / cost
                          │
                          └── feasible region is to the LEFT of every ceiling
                              and ABOVE every floor
```

Read the diagram in two directions.

**Vertically**, accuracy and calibration are floors. A candidate below the floor is infeasible regardless of how fast it is. The floor is a clinical number, not a statistical one: it comes from the smallest effect the clinical claim can tolerate, adjusted for the population that will actually use the device.

**Horizontally**, latency, energy, memory, and cost are ceilings. A candidate to the right of any ceiling is infeasible regardless of how accurate it is. The ceilings come from physics, from the battery, from the silicon, and from the purchase order.

A candidate is deployable only when it satisfies every constraint simultaneously:

```text
deployable(M) = accuracy(M)   >= a_min
             AND calibration(M) <= c_max
             AND latency_p99(M) <= l_max
             AND energy(M)      <= e_max
             AND memory(M)      <= m_max
             AND cost(M)        <= cost_max
             AND regulatory(M)  =  covered
```

The boolean is the whole model. Everything else in this topic is a method for filling in the terms honestly.

### 3.1 Dominance and the frontier

Candidate X **dominates** candidate Y when X is at least as good on every objective and strictly better on at least one. Dominated candidates are removed without judgement: there is no argument for Y, because X is better or equal everywhere that matters.

After removing dominated candidates, what remains is the **Pareto frontier**. On a two-axis plot it is the staircase of points that no other point beats in both directions. On a seven-axis problem it is a set that cannot be drawn but can still be computed pairwise.

Two consequences follow, and both are counter-intuitive to teams trained on single metrics.

First, the frontier can contain candidates that are worse on the metric the team cares about most. A model with 0.3 percentage points less sensitivity may sit on the frontier because it uses one third of the energy. It is not the right choice, but it is not dominated, and saying so forces the team to state why accuracy wins.

Second, dominance is silent about near-ties. If candidate C is 1 ms faster and candidate E is 0.5 percentage points more accurate, neither dominates the other, and the frontier will contain both. The decision then depends on which constraint is binding at the margin, which is a clinical and commercial judgement, not a mathematical one.

### 3.2 The binding constraint

At any point in a design, one constraint is usually binding: the one that a small improvement would relax first. The frontier's practical value is that it identifies the binding constraint.

If accuracy is binding, more compression will not help and may hurt; the team needs more data, a better architecture, or a relaxed clinical claim. If latency is binding, pruning and distillation are more promising than quantisation, because the model is likely compute-bound. If memory is binding, weight quantisation and structured pruning matter more than graph fusion. If energy is binding, the dominant term may not be the model at all: the radio, the sensor front end, or the display often exceeds it.

Finding the binding constraint early is worth more than any single optimization, because it redirects the entire engineering effort.

### 3.3 The three projections

A seven-dimensional frontier is not visualisable, so teams look at projections. Three are standard.

| Projection | X axis | Y axis | What it reveals |
|---|---|---|---|
| Accuracy-latency | p99 latency (ms) | sensitivity or AUROC | Whether the clinical deadline forces an accuracy sacrifice |
| Energy-memory | peak memory (KiB) | energy per inference (mJ) | Which candidates fit the silicon and battery |
| Cost-volume | annual volume (units) | BOM adder (USD) | Whether a more expensive part pays back |

Each projection hides the constraints not on its axes. A point that looks excellent on accuracy-latency may be infeasible on memory. The projection is a communication device, not the decision.

### 3.4 Why the frontier moves

The frontier is not fixed. It moves when any of the following changes:

- The clinical claim changes, which moves the accuracy floor.
- The target hardware changes, which moves the latency and memory ceilings.
- The runtime improves (a new kernel, a new delegate), which moves candidates rightward or leftward without retraining.
- The calibration set changes, which can move accuracy and calibration for the same weights.
- The battery or thermal design changes, which moves the energy ceiling.
- The regulatory pathway changes, which can remove a candidate entirely.

A frontier computed once and filed is stale within a quarter. The memo should record the date and the hardware and software versions, because the same model measured on a different runtime build is a different candidate.

### 3.5 Why a single efficiency score fails

It is tempting to collapse the frontier into one number, such as accuracy per millijoule or accuracy divided by a weighted sum of resources. The collapse always fails, for four reasons.

- **The units are not comparable.** A percentage point of sensitivity and a millijoule do not have an exchange rate. Any weighting is a statement about clinical value, and that statement belongs in the memo, not hidden in a formula.
- **The constraints are asymmetric.** Falling below the accuracy floor is categorically different from exceeding the energy ceiling by 5%. A weighted sum treats both as small deductions.
- **The score hides the binding constraint.** A team cannot act on "our score is 0.72". It can act on "we are 30 KiB over SRAM".
- **The score is not stable.** Change the sample, the ambient temperature, or the runtime build, and the weights that made the score meaningful no longer apply.

The frontier keeps the dimensions separate precisely so that the trade-off is argued in clinical and commercial terms. The decision rule in Step 10 is where the weighting is made explicit, one decision at a time.

### 3.6 Reading the frontier as a production system

The frontier describes a set of models, but production deploys one model plus a support system. The support system includes:

- The calibration data version and the range-selection method.
- The runtime and delegate version.
- The board revision, clock configuration, and thermal design.
- The threshold and the postprocessing rules.
- The fallback path when the model is unavailable or the input is out of distribution.
- The monitoring that detects drift.

Two candidates that differ only in the model may have identical frontiers but very different support costs. A candidate that requires a new delegate has a higher integration cost than one that runs on the existing runtime, even if its arithmetic is identical. The memo should record integration cost alongside the measured constraints, because it is part of the decision even when it is not part of the frontier.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Pareto dominance | X is at least as good as Y on every objective and strictly better on one | Used to mean "better overall" | Boolean |
| Pareto frontier | The set of non-dominated candidates | Called "the best models" | Set |
| Feasible set | Candidates satisfying every hard constraint | Confused with the frontier | Set |
| Constraint | A limit that cannot be traded | Called a "goal" or "target" | Mixed |
| Objective | A quantity to be optimized within constraints | Used interchangeably with constraint | Mixed |
| Binding constraint | The constraint a marginal improvement would relax first | Ignored in most design reviews | Identifier |
| Knee point | Frontier point where further resource spend buys little accuracy | Treated as automatically optimal | None |
| Accuracy | Proportion correct at a stated threshold and population | Reported without the threshold | Proportion |
| Sensitivity | True positive rate at the operating threshold | Called "accuracy" in imbalanced data | Proportion |
| Specificity | True negative rate at the operating threshold | Omitted when prevalence is low | Proportion |
| AUROC | Probability a random positive outranks a random negative | Treated as a deployment metric | 0.5–1.0 |
| Calibration (probability) | Agreement between predicted probability and observed frequency | Confused with quantization calibration | Unitless |
| Expected calibration error (ECE) | Bin-weighted gap between confidence and accuracy | Compared across different bin counts | Proportion |
| Brier score | Mean squared error of predicted probabilities | Reported without the base rate | 0–1 |
| Calibration slope | Slope of outcome on predicted log-odds; 1 is perfect | Ignored after compression | Unitless |
| Calibration (quantization) | Choosing numeric ranges from representative data | Same word, different activity | Not a metric |
| Non-inferiority margin | Pre-specified largest acceptable loss | Chosen after seeing results | Proportion |
| Latency p50 | Median time per inference | Used as the only latency number | ms |
| Latency p99 | 99th percentile time per inference | Assumed equal to p50 | ms |
| Cold start | First inference after load, caches cold | Mixed into warm statistics | ms |
| Warm start | Steady-state inference with caches warm | Reported as the only number | ms |
| Throughput | Inferences per second at a stated batch | Confused with latency | inferences/s |
| Batch size | Number of inputs processed together | Omitted from latency claims | Count |
| Energy per inference | Joules consumed for one inference | Reported as average power | mJ |
| Power | Instantaneous rate of energy use | Used as a proxy for energy | mW |
| Peak memory | Maximum simultaneously resident bytes | Replaced by model file size | KiB |
| Model size | Serialised bytes on storage | Used as a memory budget | KiB |
| Working memory | Activations, workspace, and buffers at runtime | Ignored in early estimates | KiB |
| Multiply-accumulate (MAC) | One multiply plus one add | Counted as two floating-point operations (FLOPs) | Count |
| Arithmetic intensity | Operations per byte moved from memory | Computed with cache traffic | FLOP/byte |
| Roofline | Plot of attainable performance against arithmetic intensity | Treated as a latency predictor | Model |
| Ridge point | Intensity at which a machine becomes compute-bound | Ignored when comparing chips | FLOP/byte |
| Bill of materials (BOM) | Per-unit component cost at volume | Confused with retail price | USD/unit |
| Non-recurring engineering (NRE) | One-time design and tooling cost | Amortised incorrectly | USD |
| Duty cycle | Fraction of time a function is active | Confused with utilisation | Proportion |
| Thermal design power (TDP) | Sustained heat a package is designed to dissipate | Treated as typical power | W |
| Throttling | Clock or voltage reduction to stay within thermal limits | Assumed not to occur in short tests | Boolean |
| Utilisation | Fraction of peak compute or bandwidth actually used | Reported as efficiency | Proportion |
| Quantization scale | Real value per integer step | Confused with learning rate | Real units per step |
| Zero point | Integer that represents real zero | Assumed always zero | Integer |
| Post-training quantization (PTQ) | Quantization with calibration but no retraining | Assumed lossless | Method |
| Quantization-aware training (QAT) | Training with simulated quantization | Assumed to remove all loss | Method |

Two vocabulary traps deserve emphasis because they cause real errors in review meetings.

The first is **calibration**. In clinical statistics it means the agreement between predicted risk and observed risk. In quantization it means the process of choosing ranges from data. A sentence such as "we calibrated on 500 records" is ambiguous and should be rewritten as "we set quantization ranges on 500 records" or "we assessed probability calibration on the held-out set".

The second is **memory**. Model size is the number of bytes in the file. Peak memory is the maximum number of bytes resident at any instant, including activations, workspace, the runtime, and fragmentation. A 400 KiB model can require 700 KiB of SRAM. Every memory figure in this module states whether it is weights only or weights plus activations, and in which dtype.

### 4.1 The conditions that make a number meaningful

A number without its conditions is a rumour. The table below lists the conditions that every claim in a deployment review must carry.

| Quantity | Mandatory conditions | Common omission |
|---|---|---|
| Accuracy | Dataset version, split, threshold, prevalence, confidence interval | Threshold and prevalence |
| Sensitivity | Operating point, subgroup, reference standard | Subgroup |
| Calibration | Bin count, smoothing, sample size | Bin count |
| Latency | Hardware, clock, batch, percentile, warm or cold, ambient, runtime version | Percentile and ambient |
| Throughput | Batch, concurrency, sequence length, percentile constraint | Concurrency |
| Energy per inference | Voltage, current measurement method, inclusion of preprocessing and radio | Inclusion set |
| Peak memory | Weights, activations, workspace, runtime, fragmentation, allocator | Everything except weights |
| Model size | dtype, packing, metadata, compression | dtype |
| MACs | Counting convention (MAC as one or two operations) | Convention |
| Cost | Volume, currency, year, included tooling | Volume |
| Battery life | Capacity, derating, duty cycle, temperature | Derating |
| Thermal | Ambient, enclosure, duration, throttle threshold | Duration |

The table is also a checklist for reviewing a vendor claim. A vendor that reports "20 ms latency" without hardware, batch, and percentile has told you something about a demonstration, not about a product.

### 4.2 Two words that carry a regulatory meaning

**Intended use** is the regulator's description of what the device is for, in the population and conditions it will be used. A frontier computed outside the intended use is not evidence for the device. If the model is intended for adults and the calibration set is dominated by adults under 40, the calibration evidence does not cover the intended use.

**Substantial modification** is a change that alters the safety or effectiveness of the device, or its intended use. In the European Union, the Medical Devices Regulation attaches significance to this concept; in the United States, the equivalent question is whether a change requires a new submission or is covered by a PCCP. Compression can be substantial if it changes outputs in a clinically meaningful way. The frontier should record the change classification because it constrains which candidates are shippable, not merely which are fast.

## 5. The mechanics, step by step

This section is the procedure. It assumes you have a trained model and a target device, and it produces a frontier, a decision, and a memo.

### Step 1. State the clinical decision and its deadline

**Input:** A clinical workflow description and a proposed device.

**Output:** One sentence naming the decision, the actor, the action, and the time window. Example: "The patch decides within 250 ms whether a 4-second electrocardiogram (ECG) window contains an atrial fibrillation episode, and raises an on-device alert that the patient sees immediately."

**Rationale:** Every downstream budget derives from this sentence. If the decision is "queue for review tomorrow", the latency constraint is irrelevant and the energy constraint dominates. If the decision is "fire a defibrillation advisory", latency is safety-critical and p99, not p50, is the number that matters.

**Branch point:** If the decision has no explicit time window, stop and obtain one. A budget without a deadline is a guess.

**Edge case:** Clinical workflows often have two deadlines: a soft one for display and a hard one for alarm. Record both. The hard deadline is the constraint; the soft deadline is a design target.

### Step 2. Write the constraint ledger

**Input:** The clinical sentence, the device specification, the regulatory claim, and the commercial plan.

**Output:** A table with one row per constraint, each carrying a value, a unit, a measurement method, an owner, and a hard/soft flag.

**Rationale:** A constraint that is not written with a unit and a measurement method cannot be tested, and a constraint without an owner is not a constraint. The ledger is also the artifact that prevents a late-stage argument about whether 47 ms is "close enough" to 45 ms.

A ledger for the ECG patch might contain:

| ID | Constraint | Value | Unit | Method | Type | Owner |
|---|---|---:|---|---|---|---|
| C1 | Sensitivity at operating threshold | ≥ 0.925 | proportion | Held-out set, 95% CI | Hard | Clinical |
| C2 | ECE | ≤ 0.040 | proportion | 15-bin ECE, held-out | Hard | Clinical |
| C3 | p99 end-to-end latency | ≤ 250 | ms | HIL, 10,000 runs | Hard | Systems |
| C4 | Model energy per inference | ≤ 5.0 | mJ | Power monitor, 1 kHz | Hard | Hardware |
| C5 | Peak SRAM | ≤ 256 | KiB | Runtime high-water mark | Hard | Firmware |
| C6 | Flash for model | ≤ 512 | KiB | Binary size | Hard | Firmware |
| C7 | BOM adder | ≤ 5.00 | USD | Purchasing quote at 50k units | Hard | Commercial |
| C8 | Cloud dependency | none | boolean | Architecture review | Hard | Security |
| C9 | Battery life | ≥ 7 | days | Duty-cycle model + bench | Soft | Product |
| C10 | Model update without new submission | yes | boolean | Regulatory review | Hard | RA |

**Branch point:** If more than about twelve rows are marked hard, the design is over-constrained. Either a constraint is actually a preference, or the product scope is wrong. Resolve this before measuring candidates, because an over-constrained ledger makes every candidate infeasible and invites arbitrary relaxation later.

**Edge case:** Some constraints are hard only at certain volumes. A BOM adder of 5 USD may be hard at 100k units and irrelevant at 5k. Record the volume assumption with the constraint.

### Step 3. Decompose the end-to-end latency budget

**Input:** The ledger's end-to-end deadline.

**Output:** A term-by-term budget with measured or estimated values and a stated slack.

**Rationale:** The inference time is usually a minority of the pathway. Teams that optimize only the model often find that the deadline is missed for reasons the model cannot fix.

The decomposition is additive:

```text
L_total = L_sensor + L_preprocess + L_inference + L_postprocess + L_transport + L_human
```

For the ECG patch, a measured decomposition might be:

| Term | Value (ms) | Notes |
|---|---:|---|
| Sensor acquisition and direct memory access (DMA) | 32 | 256 Hz, 8-sample window, double-buffered |
| Preprocessing (filter, normalise, segment) | 18 | Fixed-point, 64 MHz |
| Model inference | 46 | p99, warm, batch 1 |
| Postprocessing (threshold, refractory logic) | 4 | Rule-based |
| Transport to phone | 90 | Bluetooth Low Energy, p99 |
| Human acknowledgement | 20 | Display and vibration |
| **Total** | **210** | 40 ms slack (16%) |

**Branch point:** If the sum exceeds the deadline, do not immediately shrink the model. Check whether transport or human terms can be removed from the critical path (for example, by alerting on-device rather than via the phone). Removing a 90 ms transport term is often cheaper than halving the model.

**Edge case:** The p99 of a sum is not the sum of p99s. If the terms are independent, the tail of the total is somewhat less than the sum of the tails; if they are correlated (thermal throttling affects sensor, preprocessing, and inference together), it can be worse. Measure the total, and use the sum only for budgeting.

### Step 4. Profile the baseline on the target

**Input:** The uncompressed model, the target board, and the runtime build.

**Output:** Parameters, MACs, model size, peak memory, p50/p95/p99 latency, energy per inference, and utilisation.

**Rationale:** Estimates from a workstation are not evidence. The baseline measurement anchors every later comparison and exposes the binding resource before compression begins.

**Branch point:** The profile determines the compression order.

| Profile finding | Likely bottleneck | First method to try |
|---|---|---|
| High memory traffic, low compute utilisation | Memory bandwidth | Weight quantization, operator fusion |
| High compute utilisation, low memory traffic | Compute | Distillation, structured pruning, lower precision |
| Peak memory close to SRAM limit | Activation memory | Smaller input window, fused operators, activation quantization |
| Energy dominated by radio | Transport | On-device decision, event-only transmission |
| Latency dominated by one unsupported operator | Operator fallback | Replace or reimplement the operator |
| Latency tails far above median | Scheduling, thermal, or fallback | Soak test, watchdog analysis, partition inspection |

**Edge case:** A model that is fast in isolation can be slow inside the pipeline because it evicts the sensor buffer from cache or because it shares a bus with the radio. Profile in the pipeline, not in a microbenchmark.

### Step 5. Generate candidates along the compression ladder

**Input:** The baseline and the bottleneck.

**Output:** Three to six candidates that differ in a controlled way.

**Rationale:** The frontier needs points. A single compressed model cannot be evaluated for dominance, and a team that tests one candidate tends to rationalise it.

A standard ladder:

1. Baseline, unmodified.
2. Graph-optimized baseline (fusion, constant folding, layout).
3. INT8 PTQ, per-tensor.
4. INT8 PTQ, per-channel.
5. INT8 QAT, per-channel.
6. Distilled student, then INT8 QAT.
7. Structured-pruned variant, if the hardware supports it.

**Branch point:** If the baseline already meets every constraint with margin, stop. Compression adds risk and regulatory burden for no benefit. Record the decision not to compress, with the margin.

**Edge case:** Candidates must be compared at the same operating threshold. A quantized model evaluated at a threshold tuned on the FP32 model can look worse or better for reasons unrelated to quantization. Re-tune the threshold on a validation set for each candidate, then evaluate once on the held-out set.

### Step 6. Measure each candidate against every constraint

**Input:** Candidates, held-out data, target hardware, power monitor.

**Output:** A candidate table with one row per candidate and one column per constraint.

**Rationale:** Measurement is the only step that produces evidence. The table is the frontier's raw material.

The measurement protocol must fix:

- The held-out set, frozen before any compression experiment.
- The calibration set, disjoint from the held-out set.
- The operating threshold selection rule, applied identically to every candidate.
- The latency measurement conditions: warm or cold, batch size, background load, ambient temperature, and number of runs.
- The energy measurement method: shunt resistor or on-board fuel gauge, sampling rate, and inclusion of preprocessing.
- The memory measurement method: runtime high-water mark, not file size.

**Branch point:** If a candidate fails a constraint by a wide margin and the cause is structural (for example, an unsupported operator forcing a fallback), drop it rather than tuning it. If it fails narrowly, keep it and record the gap.

### Step 7. Compute dominance and the frontier

**Input:** The candidate table.

**Output:** The non-dominated set, with near-ties flagged.

**Rationale:** Dominance removes candidates without requiring a value judgement, which shrinks the argument to the points that genuinely trade off.

The pairwise test is:

```text
dominates(X, Y) = for all objectives o: X[o] >= Y[o]   (with direction normalised)
                  and for at least one o: X[o] > Y[o]
```

Normalise direction first: accuracy is maximise, latency is minimise, energy is minimise, memory is minimise, cost is minimise. Calibration error is minimise.

**Decision table for a pairwise comparison:**

| Accuracy | Latency | Energy | Memory | Cost | Verdict |
|---|---|---|---|---|---|
| Better | Better or equal | Better or equal | Better or equal | Better or equal | X dominates Y |
| Better | Worse | Equal | Equal | Equal | Neither dominates; trade-off |
| Equal | Equal | Equal | Equal | Equal | Duplicate; keep one |
| Worse | Better | Equal | Equal | Equal | Neither dominates; trade-off |
| Better | Better | Worse | Equal | Equal | Neither dominates; trade-off |
| Worse | Worse | Worse | Worse | Worse | Y dominates X |

**Edge case:** Floating-point noise can make two runs of the same candidate appear to dominate each other. Use a pre-declared tolerance per metric (for example, 0.2 percentage points for sensitivity, 2 ms for p99) and treat differences inside the tolerance as equal. Record the tolerance in the memo.

### Step 8. Test feasibility against the hard constraints

**Input:** The frontier and the ledger.

**Output:** The feasible frontier, plus a list of infeasible candidates with the binding constraint named.

**Rationale:** A candidate can be non-dominated and still illegal. Dominance is about trade-offs; feasibility is about limits. The order matters: compute dominance first so that infeasible points do not distort the frontier, then filter.

**Branch point:** When no candidate is feasible, there are four moves, in increasing cost:

1. Relax a soft constraint (for example, accept 6.5 days of battery life instead of 7).
2. Change the hardware (more SRAM, an NPU, a bigger battery).
3. Change the model architecture and return to Step 5.
4. Change the clinical claim or the workflow, which is a product decision and usually requires clinical and regulatory sign-off.

**Edge case:** A candidate can be infeasible on a constraint that was not in the ledger. When that happens, add the constraint, re-run the comparison, and record the addition. Discovering a constraint late is normal; hiding it is not.

### Step 9. Sensitivity analysis

**Input:** The feasible frontier and the assumptions behind each measurement.

**Output:** A table showing how the ranking changes when one assumption moves.

**Rationale:** A decision that flips under a small change in the calibration set is not a decision; it is a coin toss with documentation. Sensitivity analysis identifies which assumptions the choice depends on.

Vary at least:

- Calibration set size (for example, 512 windows versus 64).
- Bit width (8 versus 4).
- One excluded layer (keep the first convolution in FP32).
- Ambient temperature (22 °C versus 40 °C).
- Input prevalence (the clinical base rate versus the trial base rate).

**Branch point:** If the ranking is stable across all variations, document the robustness and proceed. If it flips, the choice must be justified on the dimension that causes the flip, and that dimension should be added to the monitoring plan.

### Step 10. Choose, record, and hand off to change control

**Input:** The feasible frontier, the sensitivity analysis, and the ledger.

**Output:** A decision memo, a chosen candidate, a named fallback, and a change-control classification.

**Rationale:** The frontier does not choose. A person chooses, and the memo is how that person is held accountable without being second-guessed by everyone who was not in the room.

The memo contains:

1. The clinical decision and deadline.
2. The constraint ledger, with hard and soft flags.
3. The candidates and their measurements, with methods and dates.
4. The dominance result and the feasible frontier.
5. The chosen candidate, the decision rule, and the residual margin on each hard constraint.
6. The sensitivity analysis.
7. The fallback candidate and the trigger for using it.
8. The change-control classification and the evidence required.
9. The monitoring plan for post-deployment drift.

**Branch point:** If the chosen candidate is a compressed model, the change-control classification determines whether the work can proceed under an existing PCCP or requires a new submission. Topic 09 covers the decision tree. At this stage, record the question, not the answer.

**Edge case:** The fallback must be a model that has also been measured, not a hypothetical. A fallback that has never run on the target is a hope.

### 5.1 Decision table for the whole procedure

| Situation | Evidence | Action |
|---|---|---|
| All candidates feasible, one dominates | Candidate table | Choose the dominant candidate; document margin |
| All candidates feasible, trade-off on frontier | Frontier | Choose by binding constraint; name the tie-breaker |
| No candidate feasible, gap small (< 10% on one constraint) | Ledger gaps | Relax a soft constraint or optimize the runtime |
| No candidate feasible, gap large | Ledger gaps | Change hardware or architecture; return to Step 5 |
| Candidate fails one subgroup but passes overall | Subgroup table | Reject unless the subgroup gap is clinically justified |
| Candidate passes accuracy but fails calibration | ECE and slope | Try QAT, temperature scaling, or a different threshold |
| Candidate passes on desktop, fails on target | HIL results | Treat desktop results as invalid; profile fallback |
| Ranking flips with calibration set size | Sensitivity table | Enlarge calibration set; re-run before deciding |
| Ranking flips with ambient temperature | Thermal soak | Use sustained, not burst, measurements |
| Compressed model changes outputs materially | Equivalence test | Classify as a change; involve regulatory early |

### 5.2 Where the procedure branches

Each step has at most three meaningful outcomes. Writing them down prevents a team from drifting into a fourth, undefined state.

| Step | Outcome A | Outcome B | Outcome C |
|---:|---|---|---|
| 1 | Deadline explicit: proceed | Deadline soft or absent: obtain from clinical owner | Two deadlines: record both, constrain on the hard one |
| 2 | Fewer than 12 hard constraints: proceed | More than 12 hard: reduce scope | Constraint unmeasurable: replace or drop it |
| 3 | Sum within deadline: proceed | Sum exceeds: remove a term from the critical path | Terms correlated: measure end to end |
| 4 | Bottleneck clear: choose method | Bottleneck ambiguous: instrument more | Bottleneck outside the model: stop model work |
| 5 | Three or more candidates: proceed | One candidate only: generate more | Baseline already feasible: stop, document |
| 6 | All constraints measured: proceed | A constraint unmeasurable on target: use a proxy and record it | Measurement invalid (wrong threshold, wrong split): redo |
| 7 | One dominant candidate: choose | Frontier trade-off: decide by binding constraint | All dominated but infeasible: return to Step 5 |
| 8 | At least one feasible: proceed | None feasible, small gap: relax soft or optimize runtime | None feasible, large gap: hardware or architecture change |
| 9 | Ranking stable: decide | Ranking flips on one assumption: enlarge evidence | Ranking flips on several: the decision is premature |
| 10 | Change covered by PCCP: proceed | Change not covered: submit or redesign | Classification unknown: escalate before building |

The table is not a flowchart for its own sake. Its value is that it names the stopping conditions. A team that cannot say which row it is in does not yet have a decision.

### 5.3 Tolerances, replicates, and near-ties

Dominance is a strict comparison, and measurements are not strict. Without a declared tolerance, two runs of the same candidate will produce a spurious dominance relation, and the frontier will change every time it is recomputed.

A tolerance is the smallest difference that the team is willing to treat as real. It should be set before measurement, from the precision of the instrument and the clinical significance of the change.

| Metric | Typical tolerance | Basis |
|---|---:|---|
| Sensitivity | 0.005 | Half the width of a 95% CI at the planned sample size |
| Specificity | 0.005 | Same |
| ECE | 0.005 | Bin-count sensitivity |
| p99 latency | 3 ms | Timer resolution and run-to-run variation |
| Energy per inference | 5% | Power monitor accuracy and duty-cycle jitter |
| Peak memory | 4 KiB | Allocator granularity |
| BOM | 0.25 USD | Quote precision at volume |

Two practices follow.

- **Replicate the measurement.** Run each candidate at least three times on different days, or with different input order, and use the median for the comparison and the spread for the tolerance check.
- **Record the tie.** When two candidates are within tolerance on every metric, declare them equivalent and choose on a non-measured criterion (integration cost, supply risk, familiarity). Do not invent a difference from noise.

The frontier is a decision aid, not a measurement of truth. Treating it with the precision of the underlying instrument is the difference between a useful tool and a source of false confidence.

### 5.4 Edge cases that appear in every compression programme

**Outlier channels.** A single weight or activation channel with a range far above the rest forces a scale that renders the remaining values coarse. The symptom is a large drop in signal-to-quantization-noise ratio (SQNR) on the non-outlier subset while overall error looks acceptable. The fix is per-channel quantization, clipping, or outlier separation. Section 6 works through the arithmetic.

**Per-tensor versus per-channel scales.** Per-tensor uses one scale for the whole tensor and is simple and cheap. Per-channel uses one scale per output channel and costs a small amount of metadata but usually recovers most of the lost accuracy. The choice interacts with the runtime: some kernels only support per-tensor, and some accelerators require per-channel for convolution.

**Activation clipping.** Activations often have a long tail. Clipping the tail at a percentile below the maximum reduces the scale and improves resolution for the bulk, at the cost of saturating a small fraction of values. The clipping percentile is a hyperparameter that must be selected on calibration data and evaluated on held-out data.

**Operator fallback.** An operator the accelerator does not support runs on the central processing unit (CPU). The result is a graph partition, a memory copy at the boundary, and a latency spike that is often much larger than the operator's arithmetic cost.

**Unsupported operators.** Some operators are not merely slow on an accelerator; they are unsupported by the runtime's quantized kernel set. Examples include certain activation functions, dynamic shapes, and custom attention. Check the operator list before choosing the architecture, not after.

**Zero-point drift.** In asymmetric quantization, the zero point is an integer chosen at calibration. If the activation distribution shifts after deployment, the real value represented by the zero point can drift away from the calibrated range, and clamping becomes asymmetric. Monitor saturation rate, not only accuracy.

**Calibration-set bias.** If the calibration set over-represents one site, sensor, or demographic group, the ranges are tuned to that group and the quantization error is larger for others. Calibration data is a regulated artifact and must be documented with provenance and version.

**Thermal throttling.** A model measured for 30 seconds at 22 °C can be 30% slower after 10 minutes at 40 °C. Any latency constraint that matters in the field must be measured under sustained load at the specified ambient temperature, not in a cold burst.

The table below maps the observable symptom to the likely mechanism and the first test. It is deliberately ordered by frequency, because the first three rows account for most deployment surprises.

| Symptom | Likely mechanism | First test |
|---|---|---|
| Aggregate error acceptable, subgroup error high | Calibration-set bias or per-tensor scale | Per-channel quantization; stratified calibration |
| SQNR acceptable but small weights vanish | Outlier-driven scale collapse | Step-to-median ratio; distinct level count |
| One channel contributes most of the error | Per-channel skew | Channel range ratio; per-channel SQNR |
| Accuracy drops only at the operating threshold | Probability shift from activation clipping | Reliability curve before and after; threshold sweep |
| Latency spikes on some inputs only | Operator fallback or dynamic shape | Partition listing; per-operator timing |
| Latency degrades over minutes | Thermal throttling | Sustained soak at specified ambient |
| Output drifts after a firmware update | Runtime or kernel change; zero-point drift | Version diff; saturation-rate monitoring |
| Memory allocation fails at boot | Peak memory underestimated | Runtime high-water mark; allocator log |
| Energy higher than predicted | Radio retries or sensor duty cycle | Full-cycle power trace, not inference-only |
| Accuracy fine on trial data, poor on first site | Distribution shift or calibration bias | Site-stratified evaluation |

Each row is a diagnostic hypothesis, not a conclusion. The test is what turns the hypothesis into evidence, and the evidence is what belongs in the memo.

## 6. Worked example

This is a constructed case, not a measured product. The numbers are chosen to be realistic and to make the arithmetic close; they are not a benchmark result. Every figure below is either derived in the text or produced by the code in Section 7.

### 6.1 The case

A single-lead ECG patch detects atrial fibrillation (AF) from 4-second windows. The device uses an Arm Cortex-M4F at 64 MHz with 256 KiB SRAM and 1 MiB flash, powered by a CR2032 coin cell rated 235 mAh at 3.0 V. The clinical deadline is 250 ms end to end. The accuracy floor is sensitivity 0.925 at the operating threshold, with ECE at most 0.040.

The baseline model is a four-layer one-dimensional convolutional network with 41,216 parameters. The first convolution has 8 output channels, 1 input channel, and a kernel of 5, giving 40 weights. We will use that layer to derive the quantization parameters by hand.

### 6.2 Scale and zero point by hand

The 40 weights in the first convolution have a minimum of -0.061 and a maximum of 0.412. One weight, in output channel 2, is 0.412; the median absolute weight is 0.0182. That single weight is 22.6 times the median.

**Asymmetric (affine) derivation for 8 bits:**

```text
range      = x_max - x_min = 0.412 - (-0.061) = 0.473
levels     = 2^8 - 1 = 255
scale S    = 0.473 / 255 = 0.00185490 real units per integer step
zero point = round(-x_min / S) = round(0.061 / 0.00185490) = round(32.886) = 33
```

Check the endpoints:

```text
q(x_max) = round(0.412 / 0.00185490) + 33 = round(222.11) + 33 = 222 + 33 = 255
q(x_min) = round(-0.061 / 0.00185490) + 33 = round(-32.886) + 33 = -33 + 33 = 0
```

Both endpoints land exactly on the representable range. Dequantization is `x_hat = S * (q - Z)`.

**Symmetric derivation for 8 bits:**

```text
amax   = max(|x|) = 0.412
qmax   = 2^(8-1) - 1 = 127
scale  = 0.412 / 127 = 0.00324409
zero point = 0
q(0.412) = round(0.412 / 0.00324409) = 127
q(-0.061) = round(-0.061 / 0.00324409) = round(-18.80) = -19
```

Symmetric is simpler and has no zero-point metadata, but it wastes the negative side of the range: the data only reaches -0.061, while the representable minimum is -0.412. Affine exploits the asymmetry and gets a step 1.75 times finer (0.001855 versus 0.003244). The measured SQNR on this layer is 42.04 dB for affine and 37.54 dB for symmetric, a 4.50 dB difference.

### 6.3 Memory before and after

**Weights only, whole model:**

| Representation | Weight bytes | Metadata bytes | Total bytes | KiB | Ratio |
|---|---:|---:|---:|---:|---:|
| FP32 | 41,216 × 4 = 164,864 | ~1,024 | 165,888 | 162.0 | 1.00 |
| INT8 per-tensor | 41,216 | 5 per tensor, 4 tensors = 20 | 41,236 | 40.3 | 4.00 |
| INT8 per-channel | 41,216 | 120 channels × 5 = 600 | 41,816 | 40.8 | 3.943 |

The per-channel metadata is 600 bytes: 120 output channels across three convolution layers (8 + 16 + 32) and one fully connected layer (64), each carrying a 4-byte float scale and a 1-byte zero point. The compression ratio barely moves, from 4.00 to 3.943, which is the right way to think about metadata: it is noise at this scale and becomes material only for small tensors or very low bit widths.

At the layer level the picture is different. The first convolution is 40 weights:

```text
FP32: 40 × 4 = 160 bytes
INT8 per-tensor: 40 + 4 + 1 = 45 bytes
ratio = 160 / 45 = 3.56
```

The metadata costs 11% of the compressed layer. A 3.56× ratio, not 4×, is the honest number for a small tensor.

**Weights plus activations:**

Peak activation memory in FP32 is 48 KiB. With INT8 activations the tensor payload falls to 12 KiB, but the implementation needs an im2col buffer of 6 KiB and 8 KiB of stack, so the runtime high-water mark is 26 KiB. The relevant comparison for the SRAM constraint is 48 KiB versus 26 KiB, not 162 KiB versus 41 KiB. Model size is not a memory budget.

Flash: firmware 380 KiB plus model 40.8 KiB = 420.8 KiB of 1,024 KiB, leaving 603.2 KiB. Flash is not binding at this model size; SRAM and energy are.

### 6.4 Latency budget

The measured decomposition from Step 3 totals 210 ms against a 250 ms deadline, leaving 40 ms of slack (16%). Inference is 46 ms at p99, which is 18.4% of the deadline and 21.9% of the measured total.

The model is therefore not the binding term. If the deadline were missed, the first place to look would be the 90 ms transport term, not the 46 ms inference. A team that halved inference to 23 ms would recover 23 ms, or 9.2% of the deadline; a team that moved the alert on-device and removed the transport term would recover 90 ms, or 36%.

This is the single most common analytical error in edge machine learning: optimising the term that is easiest to measure rather than the term that dominates.

### 6.5 Energy per inference and battery life

During inference the board draws 8.2 mA at 3.0 V for 46 ms:

```text
E_inference = V × I × t = 3.0 V × 0.0082 A × 0.046 s = 1.132 mJ
```

The whole 2-second cycle includes preprocessing and radio:

```text
E_preprocess = 3.0 × 0.0045 × 0.050 = 0.675 mJ
E_radio      = 3.0 × 0.0120 × 0.090 = 3.240 mJ
E_cycle      = 1.132 + 0.675 + 3.240 = 5.047 mJ
```

At one cycle every 2 seconds, the average power is 5.047 mJ / 2 s = 2.524 mW. At 3.0 V the average current from the cycle is 0.841 mA. Adding 3 µA of sleep current gives 0.844 mA.

```text
ideal life = 235 mAh / 0.844 mA = 278.4 h = 11.6 days
derated    = 11.6 days × 0.66 = 7.7 days
```

The 0.66 factor is the capacity derating for operation at 0 °C, where the cell delivers about 66% of its 21 °C capacity. The result, 7.7 days, just clears the soft 7-day constraint. The radio alone accounts for 64% of cycle energy, which is why the binding constraint for this product is transport, not inference.

### 6.6 Accuracy and calibration cost

| Metric | FP32 baseline | INT8 PTQ | INT8 QAT | Distilled INT8 |
|---|---:|---:|---:|---:|
| Sensitivity | 0.942 | 0.936 | 0.941 | 0.929 |
| Specificity | 0.961 | 0.958 | 0.960 | 0.955 |
| ECE (15 bins) | 0.021 | 0.034 | 0.024 | 0.029 |
| Calibration slope | 0.97 | 0.91 | 0.96 | 0.93 |
| p99 latency (ms) | 138 | 46 | 47 | 21 |
| Model energy (mJ) | 3.40 | 1.13 | 1.16 | 0.52 |

PTQ costs 0.6 percentage points of sensitivity and 0.013 of ECE; both remain inside the floors of 0.925 and 0.040. QAT recovers almost all of the loss at the cost of a training run and a more complex change package. The distilled model is fastest and cheapest but sits only 0.4 percentage points above the sensitivity floor, which is too little margin for a 95% confidence interval that spans roughly ±0.8 percentage points at this sample size.

### 6.7 Diagnostic one: outlier-driven scale collapse

The symmetric scale is set by the largest absolute weight, 0.412. The step is 0.003244, which is 17.8% of the median absolute weight (0.0182). Any weight with magnitude below 0.001622 rounds to zero.

In this layer, one of 40 weights quantizes to zero, and only 24 distinct integer levels are used out of 255. The overall SQNR of 37.54 dB looks acceptable, but that number is dominated by the outlier's energy. On the 39 non-outlier weights the error is much larger relative to their own magnitude.

The diagnostic is not the aggregate SQNR. It is the pair:

```text
step / median(|w|) = 0.003244 / 0.0182 = 0.178
distinct levels used = 24 of 255
```

A step larger than about 10% of the median weight, or fewer than half the levels in use, indicates scale collapse. The remedies are per-channel scales, clipping the outlier, or separating outlier channels into a higher-precision path.

Affine quantization reduces the damage because it uses the asymmetric range: the step falls to 0.001855, SQNR rises to 42.04 dB, and 30 of 40 weights survive. It does not fix the outlier; it just stops wasting half the range.

### 6.8 Diagnostic two: per-channel skew

The eight output channels have very different ranges:

| Channel | Min | Max | Range | Levels at per-tensor step |
|---:|---:|---:|---:|---:|
| 0 | -0.0326 | 0.0179 | 0.0505 | 15 |
| 1 | -0.0075 | 0.0389 | 0.0464 | 14 |
| 2 | -0.0583 | 0.4120 | 0.4703 | 145 |
| 3 | -0.0610 | 0.0177 | 0.0787 | 24 |
| 4 | -0.0225 | 0.0182 | 0.0407 | 12 |
| 5 | -0.0231 | 0.0601 | 0.0832 | 25 |
| 6 | -0.0259 | 0.0419 | 0.0678 | 20 |
| 7 | -0.0335 | 0.0221 | 0.0556 | 17 |

Channel 2 consumes 145 of the 255 levels because of the outlier. Channels 0, 1, 4, and 7 get 12 to 17 levels each, which is roughly 4 bits of effective resolution. Per-channel scales give every channel its own step:

```text
per-tensor  SQNR = 37.54 dB
per-channel SQNR = 51.37 dB
gain = 13.83 dB, about 2.3 bits of effective precision
```

The cost is 600 bytes of metadata across the model, or 5 bytes per channel. The diagnostic is the ratio of the largest channel range to the smallest: here 0.4703 / 0.0407 = 11.6. A ratio above about 4 is a signal that per-tensor quantization will cost accuracy.

### 6.9 Varying one assumption

**Calibration set size: 512 windows reduced to 64.** The observed maximum in the smaller set is lower because the tail is undersampled. The activation scale falls by about 6%, which increases clipping at inference. Sensitivity drops a further 0.4 percentage points, to 0.932, and ECE rises to 0.038, close to the 0.040 floor. The lesson is that calibration set size is not a formality: it determines the tail of the range, and the tail is where the clinical signal often lives.

**Bit width: 8 reduced to 4.** Symmetric INT4 uses a scale of 0.412 / 7 = 0.05886. The median weight, 0.0182, is less than one third of a step, so 28 of 40 weights quantize to zero. SQNR falls to 12.65 dB and cosine similarity to 0.974. Applied to the whole model, weight memory falls to 20,608 bytes (20.1 KiB), but sensitivity in this constructed case falls by about 3.1 percentage points, well below the floor. INT4 is a memory tool for models with large, redundant weight matrices, not a default for small convolutional layers.

**Layer excluded: first convolution kept in FP32.** Keeping 40 weights in FP32 costs 120 extra bytes, which is 0.03% of the model. It recovers about 0.2 percentage points of sensitivity in this case because the first layer sees the raw signal and its quantization error propagates through every later layer. Excluding one layer is cheap when the layer is small; excluding a large layer is a memory decision, not a rounding decision.

The three variations illustrate the general rule: the sensitivity of the decision to an assumption determines how much evidence that assumption needs.

## 7. Code, query, or calculation

The script below uses only the Python standard library. It implements affine and symmetric quantization, dequantization, scale and zero-point computation, memory accounting, error metrics, and a Pareto feasibility check. It reproduces every number in Section 6.

```python
import math
import random


def round_half_away(value):
    """Round as integer kernels do: halves go away from zero."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def symmetric_scale(values, bits=8):
    amax = max(abs(v) for v in values)
    qmax = 2 ** (bits - 1) - 1
    return amax / qmax if amax else 1.0


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    qmin, qmax = 0, 2 ** bits - 1
    scale = (hi - lo) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zero_point = min(qmax, max(qmin, round_half_away(-lo / scale)))
    return scale, zero_point


def quantize_affine(values, scale, zero_point, bits=8):
    qmax = 2 ** bits - 1
    return [min(qmax, max(0, round_half_away(v / scale) + zero_point))
            for v in values]


def quantize_symmetric(values, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    return [min(qmax, max(qmin, round_half_away(v / scale))) for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


def error_metrics(original, reconstructed):
    n = len(original)
    mae = sum(abs(a - b) for a, b in zip(original, reconstructed)) / n
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / n
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    sqnr = 10 * math.log10(signal / noise) if noise else float("inf")
    dot = sum(a * b for a, b in zip(original, reconstructed))
    norm = math.sqrt(signal) * math.sqrt(sum(b * b for b in reconstructed))
    cosine = dot / norm if norm else 0.0
    return {"mae": mae, "rmse": math.sqrt(mse),
            "sqnr_db": sqnr, "cosine": cosine}


def weight_bytes(numel, bits, channels=None, fp32_scale=True):
    """Packed weight bytes plus per-channel scale and zero-point metadata."""
    payload = numel * bits / 8
    metadata = 0.0
    if channels:
        metadata += channels * (4 if fp32_scale else 2)
        metadata += channels * 1
    return payload + metadata
```

Build the layer and derive the parameters:

```python
random.seed(7)
layer = [round(random.gauss(0, 0.035), 4) for _ in range(39)]
layer.insert(11, 0.412)          # outlier in output channel 2

print("n =", len(layer), "min =", min(layer), "max =", max(layer))
print("median |w| =", sorted(abs(v) for v in layer)[len(layer) // 2])

s_sym = symmetric_scale(layer, 8)
q_sym = quantize_symmetric(layer, s_sym, 8)
print("symmetric scale = %.8f" % s_sym)
print("symmetric metrics =", {k: round(v, 4)
                              for k, v in error_metrics(layer, dequantize(q_sym, s_sym)).items()})
print("zeros =", sum(1 for q in q_sym if q == 0),
      "distinct =", len(set(q_sym)))

s_aff, zp = affine_params(layer, 8)
q_aff = quantize_affine(layer, s_aff, zp, 8)
print("affine scale = %.8f  zero point = %d" % (s_aff, zp))
print("affine metrics =", {k: round(v, 4)
                           for k, v in error_metrics(layer, dequantize(q_aff, s_aff, zp)).items()})
```

Expected output:

```text
n = 40 min = -0.061 max = 0.412
median |w| = 0.0182
symmetric scale = 0.00324409
symmetric metrics = {'mae': 0.0008, 'rmse': 0.0009, 'sqnr_db': 37.5354, 'cosine': 0.9999}
zeros = 1 distinct = 24
affine scale = 0.00185490  zero point = 33
affine metrics = {'mae': 0.0005, 'rmse': 0.0006, 'sqnr_db': 42.0374, 'cosine': 1.0}
```

Per-channel versus per-tensor, and the 4-bit variation:

```python
channels = [layer[i * 5:(i + 1) * 5] for i in range(8)]

per_tensor = [dequantize(quantize_symmetric(c, s_sym, 8), s_sym) for c in channels]
per_channel = []
for c in channels:
    sc = symmetric_scale(c, 8)
    per_channel.append(dequantize(quantize_symmetric(c, sc, 8), sc))

flat_pt = [v for c in per_tensor for v in c]
flat_pc = [v for c in per_channel for v in c]
print("per-tensor  =", {k: round(v, 4) for k, v in error_metrics(layer, flat_pt).items()})
print("per-channel =", {k: round(v, 4) for k, v in error_metrics(layer, flat_pc).items()})

s4 = symmetric_scale(layer, 4)
q4 = quantize_symmetric(layer, s4, 4)
print("int4 scale = %.8f" % s4)
print("int4 metrics =", {k: round(v, 4)
                         for k, v in error_metrics(layer, dequantize(q4, s4)).items()})
print("int4 zeros =", sum(1 for q in q4 if q == 0))
```

Expected output:

```text
per-tensor  = {'mae': 0.0008, 'rmse': 0.0009, 'sqnr_db': 37.5354, 'cosine': 0.9999}
per-channel = {'mae': 0.0001, 'rmse': 0.0002, 'sqnr_db': 51.3696, 'cosine': 1.0}
int4 scale = 0.05885714
int4 metrics = {'mae': 0.0142, 'rmse': 0.0164, 'sqnr_db': 12.654, 'cosine': 0.9736}
int4 zeros = 28
```

Memory accounting and the feasibility check:

```python
print("layer fp32 =", 40 * 4, "bytes; int8 =", weight_bytes(40, 8, 1), "bytes")
print("model fp32 =", 41216 * 4, "bytes")
print("model int8 =", weight_bytes(41216, 8, 120), "bytes")

candidates = [
    ("A cloud FP32",       0.951, 0.018, 480,   0.0, 0.00042, False),
    ("B edge GPU FP16",    0.949, 0.020,  38, 210.0, 0.0,     True),
    ("C MCU INT8 PTQ",     0.936, 0.034,  46,   1.13, 0.0,    True),
    ("D MCU INT8 distilled", 0.929, 0.029, 21,  0.52, 0.0,    True),
    ("E MCU INT8 QAT",     0.941, 0.024,  47,   1.16, 0.0,    True),
]
for name, acc, ece, p99, energy, cost, local in candidates:
    feasible = (acc >= 0.925 and ece <= 0.040 and p99 <= 250
                and energy <= 5.0 and local and cost <= 0.00001)
    print("%-24s acc %.3f ece %.3f p99 %3d ms energy %6.2f mJ feasible %s"
          % (name, acc, ece, p99, energy, feasible))
```

Expected output:

```text
layer fp32 = 160 bytes; int8 = 45.0 bytes
model fp32 = 164864 bytes
model int8 = 41816.0 bytes
A cloud FP32             acc 0.951 ece 0.018 p99 480 ms energy   0.00 mJ feasible False
B edge GPU FP16          acc 0.949 ece 0.020 p99  38 ms energy 210.00 mJ feasible False
C MCU INT8 PTQ           acc 0.936 ece 0.034 p99  46 ms energy   1.13 mJ feasible True
D MCU INT8 distilled     acc 0.929 ece 0.029 p99  21 ms energy   0.52 mJ feasible True
E MCU INT8 QAT           acc 0.941 ece 0.024 p99  47 ms energy   1.16 mJ feasible True
```

The script's assumptions: integer arithmetic is emulated in floating point with round-half-away, weights are packed with no alignment padding, metadata is 4 bytes per scale plus 1 byte per zero point, and the candidate table is synthetic. On real hardware, packing may round up to 4- or 16-byte boundaries, and per-channel scales may be stored as float16 to save space.

## 8. Data, standards, and regulatory context

The frontier is an engineering artifact, but several of its columns are defined by standards and regulators. This section states which standards constrain what, and which parts are convention.

### 8.1 Performance measurement standards

There is no ISO standard for "model latency". The closest thing to a cross-vendor convention is MLPerf Inference, maintained by MLCommons. Its rules define scenarios (SingleStream, MultiStream, Server, Offline), metrics, and latency constraints, and it enforces tail requirements for large language model benchmarks. In the v6.1 round, published in September 2026, MLCommons reports 30 submitters and 120 systems across datacenter and edge categories, with p99 constraints on time to first token and time per output token for the language-model tasks. MLPerf is a benchmark suite, not a regulatory standard, and a vendor's MLPerf number is not evidence for your model on your device.

For microcontrollers, MLPerf Tiny defines a small suite (keyword spotting, anomaly detection, visual wake words, image classification) with quality targets and optional energy measurement in microjoules per inference. The v1.2 results, published in April 2024, include 91 performance results and 18 energy results. The benchmark is useful for comparing silicon classes, not for claiming that a clinical model will meet a deadline.

What these benchmarks constrain: nothing binding. What they provide: a vocabulary (scenario, quality target, energy per inference) and a discipline (report the scenario and the percentile). Use the vocabulary; do not import the numbers.

### 8.2 Quality-system standards

IEC 62304, "Medical device software — Software life cycle processes", is the standard that makes the frontier's evidence auditable. Edition 1.1 (2006 plus Amendment 1:2015) is the current published state and is recognised by the United States Food and Drug Administration (FDA) under recognition number 13-79. It requires software requirements, architecture, detailed design, unit verification, integration testing, and configuration management, with rigour scaled by safety class A, B, or C.

For this topic, the practical consequence is that the constraint ledger is a software requirement, the measurement scripts are verification evidence, and the runtime and model versions are configuration items. A latency number produced by an ad hoc notebook with no version record is not verification evidence.

ISO 14971, "Medical devices — Application of risk management to medical devices", requires that foreseeable hazards be identified and controlled. A compressed model that increases false negatives in a subgroup is a hazard. The frontier's subgroup columns are therefore risk-file content, not just fairness reporting.

IEC 62366-1, "Application of usability engineering to medical devices", constrains the human term in the latency budget. An alert that appears in 20 ms but is missed by the user because of display design is not a 20 ms alert.

### 8.3 Artificial intelligence-specific guidance

The FDA, Health Canada, and the United Kingdom Medicines and Healthcare products Regulatory Agency published ten guiding principles for Good Machine Learning Practice (GMLP) in October 2021. Four bear directly on this topic: training data independent of test data (principle 4), model design tailored to the available data and intended use (principle 6), testing under clinically relevant conditions (principle 8), and monitoring deployed models (principle 10). A frontier computed on a test set that was used to choose the calibration range violates principle 4 and invalidates the accuracy column.

The FDA's final guidance on Predetermined Change Control Plans for AI-enabled device software functions was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Federal Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022. It describes a PCCP as three components: a description of modifications, a modification protocol, and an impact assessment. A quantization change may be covered if the PCCP anticipated it and the protocol's acceptance criteria are met; otherwise it is a new submission. Topic 09 works through the decision.

In the European Union, Regulation (EU) 2024/1689, the Artificial Intelligence Act, classifies a medical device that is a safety component, or is itself a device requiring third-party conformity assessment, as high-risk under Article 6(1). The original applicability date for Annex I products was 2 August 2027. The Commission's AI Omnibus proposal, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves the Annex I date to 2 August 2028. Treat the date as moving and verify it against the current Official Journal before relying on it. The Medical Device Coordination Group guidance MDCG 2025-6 (June 2025) remains the reference for how the AI Act and the Medical Devices Regulation interact.

### 8.4 What is binding and what is convention

| Item | Status | Consequence |
|---|---|---|
| IEC 62304 software lifecycle | Binding where required by the regulator | Ledger and measurements become controlled records |
| ISO 14971 risk management | Binding where required | Compression-induced subgroup harm enters the risk file |
| FDA PCCP guidance | Non-binding guidance, but the statutory pathway is binding | Change must be pre-authorized or submitted |
| EU AI Act high-risk obligations | Binding regulation, dates moving | Conformity assessment scope expands |
| MLPerf scenarios and metrics | Voluntary benchmark convention | Useful vocabulary, no regulatory weight |
| p99 latency | Engineering convention | No standard mandates a percentile; the clinical deadline does |
| Model size in KiB | Convention | Always state dtype and whether weights only |

The honest summary: standards tell you how to document the decision and what counts as a change. They do not tell you which candidate to choose. That remains an engineering and clinical judgement, and the memo is where it is defended.

### 8.5 What the standards do not constrain

It is as important to know the boundary of a standard as its content, because teams routinely over-claim compliance.

- **No standard specifies a latency percentile for medical devices.** IEC 62304 requires verification against requirements; it does not say whether the requirement should be p50, p95, or p99. The clinical deadline and the harm model determine the percentile.
- **No standard specifies a quantization method.** ISO 14971 requires that risks be controlled, not that a particular scheme be used. A per-tensor INT8 model and a per-channel INT4 model can both be compliant if the evidence supports the claim.
- **No standard specifies a calibration-set size.** GMLP requires representativeness and independence, not a count. The count is an engineering judgement supported by a sensitivity analysis.
- **No standard defines "model equivalence".** A team must define the acceptance margin and justify it clinically. A statistical test alone is not a clinical justification.
- **No standard requires a Pareto frontier.** The frontier is an engineering instrument. What is required is that the chosen configuration be justified and verified, and the frontier is an efficient way to produce that justification.

The risk of over-claiming is not only regulatory. A team that believes a standard has answered an engineering question stops asking it, and the question returns during validation.

### 8.6 Documentation that a reviewer will ask for

The following records are commonly requested during a design review or a submission review. Producing them at the moment of the decision is far cheaper than reconstructing them later.

| Record | Owner | Retention trigger |
|---|---|---|
| Constraint ledger with version and approval | Systems | Design freeze |
| Candidate table with measurement conditions | Systems | Each candidate evaluation |
| Calibration set description and provenance | Data | Each quantization run |
| Held-out set definition and freeze date | Clinical | Before first experiment |
| Subgroup analysis with confidence intervals | Biostatistics | Each model version |
| Target-hardware latency and energy report | Firmware | Each release candidate |
| Thermal soak report | Hardware | Each enclosure revision |
| Change classification and PCCP reference | Regulatory | Each model change |
| Runtime, toolchain, and board versions | Firmware | Each build |
| Threshold selection record | Clinical | Each model version |

The pattern is that every row is cheap at the time and expensive in retrospect. This is the practical argument for the ledger: it is not bureaucracy, it is the difference between a two-week review and a two-quarter delay.

## 9. Industry practice and case evidence

Public evidence about deployment frontiers is thin, because companies rarely publish the candidates they rejected. What exists falls into three categories: benchmark disclosures, regulator-cleared change protocols, and engineering post-mortems. Each supports a different kind of claim.

### 9.1 What benchmark disclosures show

MLPerf Inference results are vendor-submitted and vendor-run. They demonstrate that a configuration can achieve a throughput or latency under a specified scenario, with a specified quality target, on a specified system. They do not demonstrate that the same configuration will achieve the same result inside a product with a different runtime build, a different thermal enclosure, and a different input distribution.

The useful signal is the scenario structure. MLPerf separates SingleStream (latency-critical, one query at a time) from Server (throughput under a latency constraint) and Offline (maximum throughput). A medical device is almost always a SingleStream problem, sometimes with a hard deadline. A vendor's Offline throughput number says nothing about the p99 of a single inference on a battery-powered device.

MLPerf Tiny is closer to the medical edge case because it reports energy per inference on microcontroller-class hardware. The v1.2 summary lists, for example, a Cortex-M7 board at 280 MHz achieving 88.4 inferences per second on keyword spotting with 1,375.6 µJ per inference, and a Cortex-M33 board at 34.4 inferences per second with 656.2 µJ per inference. These are vendor-submitted results on a fixed benchmark, not measurements of a clinical model, but they establish the order of magnitude: millijoules per inference, tens of inferences per second, on parts that cost a few dollars.

An independent measurement culture exists in the thermal literature, and it is less flattering. Studies of mobile sustained performance have found that skin-temperature limits, not silicon limits, govern sustained throughput: one study of a Pixel 2 reported throttling beginning after 151 seconds, with clock reduction from 2.46 GHz to 1.19 GHz at a 48 °C skin temperature and frame-rate jank increasing by up to 146%. Another found run-to-run variance in a popular benchmark of up to 56%, and a score increase of more than 60% when the phone was chilled to 4 °C. These are independent academic measurements, and they imply that any latency number collected in a cold burst is an upper bound on performance, not a field estimate.

### 9.2 What regulatory clearances show

The FDA's AI-enabled device list and the PCCP guidance together show that regulators expect model changes to be planned. The guidance's structure, with a description of modifications, a modification protocol, and an impact assessment, maps directly onto the frontier's candidate table: the protocol names the acceptance criteria, and the impact assessment names the risks. A team that has already built the table has most of the PCCP content.

This is documented fact about the guidance. The inference, clearly labelled, is that the marginal cost of a PCCP is much lower for teams that already maintain a constraint ledger and a candidate table than for teams that reconstruct evidence after the fact.

### 9.3 What public engineering reports show

Several runtime projects document the gap between model-level and system-level performance. ONNX Runtime's documentation separates basic graph optimizations (constant folding, redundant node elimination, Conv-BatchNorm fusion) from extended fusions that apply only to specific execution providers, and warns that offline-optimized models must match the target hardware and options. ExecuTorch's documentation states that each `.pte` file is backend-specialised and that a model must be exported once per target. TensorFlow Lite for Microcontrollers documents a fixed tensor arena with a greedy memory planner and a runtime that fits in roughly 16 KiB on a Cortex-M3.

The common thread is that the deployment artifact is not the trained model. It is a target-specific binary produced by a toolchain with its own version, its own operator support, and its own memory behavior. A frontier that compares `.pt` files is comparing the wrong objects.

### 9.4 Three disclosures that change a decision

Public disclosures rarely give a full frontier, but three kinds of statement move a decision materially.

**A runtime's operator support list.** When a vendor publishes the set of operators its accelerator supports, that list is more decision-relevant than any throughput number. An unsupported operator forces a partition, and a partition forces a memory copy and a scheduling boundary. If the model contains three unsupported operators, the measured latency can be several times the arithmetic estimate, and no amount of TOPS fixes it.

**A benchmark's scenario definition.** MLPerf's SingleStream scenario is the closest public analogue to a medical device's inference pattern. A vendor that publishes only Offline throughput has published a number that does not apply to a latency-constrained device. The scenario, not the score, is the transferable information.

**A thermal or sustained-performance note.** When a vendor documents sustained clock behavior, or when an independent study reports throttling after a fixed interval, that note bounds the field performance. A device that meets its deadline for 30 seconds and misses it after 5 minutes has not met its deadline.

None of these is a substitute for measuring your model on your target. They are filters that prevent a team from spending a quarter on a candidate that cannot work.

### 9.5 Documented fact versus inference

| Claim | Status |
|---|---|
| MLPerf v6.1 published 30 submitters and 120 systems | Documented in MLCommons materials, September 2026 |
| MLPerf Tiny v1.2 includes 91 performance and 18 energy results | Documented by MLCommons, April 2024 |
| Pixel 2 throttled after 151 s, 2.46 to 1.19 GHz at 48 °C skin | Independent academic measurement, cited in Section 16 |
| Benchmark run-to-run variance up to 56% | Independent academic measurement |
| TFLM runtime fits in about 16 KiB on Cortex-M3 | Vendor documentation |
| Teams that maintain a ledger spend less on PCCP | Inference, not measured |
| Most medical edge models are SingleStream | Inference from device form factors, not a survey |

The last two rows are the kind of claim that should never appear in a regulatory submission without being labelled as an assumption.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Optimizing the model when transport dominates | Inference is < 25% of the latency budget | Effort allocated by ease of measurement | Deadline still missed after a 2× model speedup | Decompose the budget before compressing |
| Reporting model size as memory | Peak SRAM is 1.7× the file size | Activations and workspace omitted | Device crashes or refuses to allocate | Measure runtime high-water mark |
| Using p50 as the latency number | Tail not plotted | Scheduling, fallback, thermal | Intermittent missed deadlines | Report p50, p95, p99 with conditions |
| Comparing at different thresholds | Sensitivity jumps between candidates | Threshold tuned per model | False superiority | Fix the threshold rule before evaluation |
| Calibrating on the test set | Test accuracy is unusually stable | Information leakage | Inflated accuracy, failed external validation | Freeze a disjoint calibration set |
| Ignoring subgroup results | Only overall sensitivity reported | Aggregate masks a small-group loss | Equity harm and regulatory rejection | Pre-specify subgroup table |
| Treating a soft constraint as hard | Every candidate infeasible | Ledger inflation | Arbitrary late relaxation | Classify hard versus soft explicitly |
| Treating a hard constraint as soft | "We can probably get SRAM down" | Schedule pressure | Redesign after tooling | Hold hard constraints; escalate |
| One candidate only | No frontier, just a choice | Team rationalisation | No fallback, no sensitivity | Require at least three candidates |
| Cold-burst latency | Benchmarks run for seconds | Thermal throttling | Field latency 20–50% worse | Soak test at specified ambient |
| Vendor benchmark as evidence | Slides cite a TOPS number | Different model, batch, runtime | Target misses budget | Measure the actual model on the target |
| No version record | Results not reproducible | Toolchain and runtime drift | Evidence invalid | Record model, runtime, board, and date |
| BOM adder ignored | Unit economics discovered late | Hardware chosen for performance only | Product margin negative | Add cost column at Step 2 |
| Regulatory path decided after | "We will file an update" | Compression treated as build setting | Launch delay | Classify change at Step 10 |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The clinical deadline is known and stable | Budgets derive from it | Workflow change, new user group | Confirm with clinical owner in writing |
| Held-out data represents the deployment population | Accuracy column is valid | Site, device, or demographic shift | External validation set; subgroup audit |
| Latency terms are additive for budgeting | Simple decomposition | Correlated tails from thermal or contention | Measure end-to-end distribution, not sum of parts |
| The runtime build is fixed | Comparisons are fair | Toolchain upgrade changes kernels | Pin versions; re-baseline after upgrades |
| Energy scales with active time | Duty-cycle model | Sleep current, leakage, radio retries | Power monitor over full cycle |
| Battery capacity is nameplate | Simple life estimate | Temperature, pulse drain, ageing | Derate from datasheet curves; bench test |
| The target board represents the product | Measurements transfer | Different memory, thermals, enclosure | Measure on the product enclosure |
| Threshold can be re-tuned per candidate | Fair accuracy comparison | Regulatory threshold locked in the claim | Check the claim's operating point |
| Calibration set is representative | Ranges generalise | Sampling bias by site or device | Compare ranges across held-out slices |
| Dominance tolerances are pre-declared | Avoids noise-driven ranking | Post-hoc tolerance selection | Write tolerances in the protocol |
| Cost quotes are at volume | BOM column is meaningful | Volume shortfall, supply constraint | Obtain quotes at three volumes |
| Regulatory classification is stable | Path column is valid | New guidance or new intended use | Regulatory review at design freeze |

Two limits deserve separate mention.

The frontier cannot represent interactions that are not additive. If two constraints trade off non-linearly (for example, adding an NPU reduces latency but increases BOM and energy at low utilisation), the candidate table captures the endpoints but not the curve. Sampling intermediate hardware configurations is the only remedy, and it is often too expensive. Record the gap.

The frontier is also a snapshot of a moving target. Runtimes improve quarterly, compilers change kernel selection, and the same model on the same board can move by 10–20% between releases. A frontier older than one release cycle should be recomputed before it is used to defend a decision.

## 12. Exercises

**Exercise 1 — Constraint ledger (recall and application, 45 minutes).**
Choose a device you can describe concretely (a pulse oximeter, a handheld ultrasound probe, a continuous glucose monitor). Write a ten-row constraint ledger with value, unit, measurement method, hard/soft flag, and owner.
*Deliverable:* one table, one page of notes on why each row is hard or soft.
*Acceptance criteria:* every row has a unit; every hard row has a measurement method that could be executed; at least two rows are soft and labelled as such.
*Time:* 45 minutes.

**Exercise 2 — Latency decomposition (application, 40 minutes).**
For the same device, decompose the end-to-end latency into six terms and assign a budget to each. Identify the largest term and state whether it is under the team's control.
*Deliverable:* a table with six rows summing to the clinical deadline, plus a one-paragraph argument.
*Acceptance criteria:* the terms sum correctly; the largest term is identified with evidence or a stated estimate; at least one term is outside the team's control and is named as such.
*Time:* 40 minutes.

**Exercise 3 — Memory accounting (calculation, 30 minutes).**
A model has 3,400,000 parameters in four weight tensors with 64, 128, 256, and 512 output channels. Compute FP32 weight bytes, INT8 weight bytes with per-tensor scales, and INT8 weight bytes with per-channel scales and 1-byte zero points. State the ratio in each case.
*Deliverable:* a calculation with units at every step.
*Acceptance criteria:* arithmetic correct to three significant figures; metadata included; the answer states weights only.
*Time:* 30 minutes.

**Exercise 4 — Dominance (application, 35 minutes).**
Given five candidates with accuracy, p99 latency, energy, memory, and BOM, compute the non-dominated set by hand. Show each pairwise comparison that eliminates a candidate.
*Deliverable:* a dominance matrix and the frontier.
*Acceptance criteria:* every elimination is justified by a pairwise comparison; ties are handled with a stated tolerance.
*Time:* 35 minutes.

**Exercise 5 — Quantization diagnosis (analysis, 50 minutes).**
Using the code in Section 7, construct a 32-weight tensor with one outlier at 20× the median and 4 output channels. Report step-to-median ratio, distinct levels used, per-tensor SQNR, per-channel SQNR, and the channel range ratio. State which diagnostic fires.
*Deliverable:* a script and a five-line interpretation.
*Acceptance criteria:* both diagnostics computed; the interpretation names the remedy and its metadata cost.
*Time:* 50 minutes.

**Exercise 6 — Sensitivity analysis (critique, 45 minutes).**
Take the Section 6 case and vary calibration set size from 512 to 64, then bit width from 8 to 4. Explain which variation changes the decision and why.
*Deliverable:* a table of results and a 200-word critique.
*Acceptance criteria:* the decision change is identified; the mechanism (tail undersampling versus step size) is explained correctly.
*Time:* 45 minutes.

**Exercise 7 — Decision memo (design, 60 minutes).**
Write a one-page memo choosing among the five candidates in Section 7. State the decision rule, the residual margin on each hard constraint, the fallback, and the change-control question.
*Deliverable:* one page, maximum 500 words, with a table.
*Acceptance criteria:* a named choice; margins quantified; fallback is a measured candidate; regulatory classification flagged rather than assumed.
*Time:* 60 minutes.

**Exercise 8 — Benchmark critique (critique, 40 minutes).**
Find a vendor-published latency or energy claim for an edge accelerator. Identify the model, batch size, runtime, percentile, and whether the benchmark is vendor-run. State what the claim does and does not support for a clinical deployment.
*Deliverable:* a 300-word critique with the source URL and access date.
*Acceptance criteria:* all five attributes identified or explicitly marked missing; the distinction between vendor-published and independent measurement is stated.
*Time:* 40 minutes.

## 13. Self-check answers

**A1. What is the difference between a constraint and an objective?**
A constraint is a limit that cannot be traded; a candidate outside it is infeasible. An objective is a quantity to be optimized within the feasible region. Accuracy is usually a constraint at the floor and an objective above it; latency is usually a constraint. Confusing the two produces either over-constrained designs or silent relaxation.

**A2. Why is the p99 of a sum not the sum of p99s?**
Because percentiles are not additive. If the terms are independent, the tail of the sum is less than the sum of the tails; if they are positively correlated (thermal throttling affects several terms at once), it can be worse. Budget with the sum, verify with an end-to-end measurement.

**A3. Compute INT8 per-channel bytes for 3,400,000 weights across 960 channels.**
Weights: 3,400,000 bytes. Metadata: 960 × (4 + 1) = 4,800 bytes. Total: 3,404,800 bytes, or 3,325.0 KiB. Ratio against FP32 (13,600,000 bytes) is 3.994.

**A4. A candidate has higher accuracy but 3× the energy and fails the energy ceiling. Is it dominated?**
No. Dominance requires being at least as good on every objective. It is infeasible, which is a different verdict. Infeasible candidates are removed by the constraint filter, not by dominance, and the memo should say "infeasible on C4" rather than "dominated".

**A5. What does a step-to-median ratio of 0.18 indicate?**
That the quantization step is 18% of the typical weight magnitude, so most weights are represented by only a handful of levels. It is a symptom of outlier-driven scale collapse. Remedies: per-channel scales, clipping, or outlier separation.

**A6. Why is model file size a poor proxy for peak memory?**
Peak memory includes activations, workspace, input and output buffers, the runtime, and allocator fragmentation. A 403 KiB INT8 model can require 700 KiB of SRAM. Only a runtime high-water measurement answers the memory question.

**A7. What is the correct response when no candidate is feasible?**
In order: relax a soft constraint, change the hardware, change the architecture, or change the clinical claim. The order is by increasing cost and increasing organisational authority. Relaxing a hard safety constraint is not on the list.

**A8. Why must the operating threshold be fixed before comparing candidates?**
Because sensitivity and specificity trade off along the receiver operating characteristic curve. A candidate evaluated at a more favourable threshold will appear better for reasons unrelated to compression. Fix the threshold selection rule, apply it on validation data, and evaluate once on held-out data.

**A9. What is the difference between probability calibration and quantization calibration?**
Probability calibration measures agreement between predicted probabilities and observed frequencies (ECE, Brier score, slope). Quantization calibration selects numeric ranges from representative data. They share a word and nothing else; both must be documented, and neither substitutes for the other.

**A10. Why does thermal throttling change the frontier rather than just the numbers?**
Because it can change the ranking. A candidate that is 10% faster in a cold burst may be 30% slower under sustained load if it uses a more power-hungry accelerator. The frontier must be computed from sustained measurements at the specified ambient temperature, or the ordering is not valid for the field.

**A11. Rubric for Exercise 7 (decision memo).**
A strong memo: names a candidate and a rule; quantifies residual margin on every hard constraint; states the measurement conditions; includes a measured fallback; flags the regulatory classification as a question with an owner; and lists the two assumptions the decision is most sensitive to. A weak memo: argues from a single metric, omits margins, or asserts regulatory equivalence without evidence.

**A12. Rubric for Exercise 8 (benchmark critique).**
A strong critique: identifies model, batch, runtime, percentile, and submitter; distinguishes vendor-run from independent; states the deployment gap (thermal, enclosure, input distribution); and declines to extrapolate. A weak critique: repeats the vendor number as if it were a property of the hardware.

## 14. Interview questions

**Q1. How would you decide whether a model is deployable?**
A strong answer lists the constraints (accuracy, calibration, latency at a stated percentile, energy, memory, cost, regulatory path), states that all must hold simultaneously, and describes measuring each on the target. A weak answer talks only about accuracy and model size.

**Q2. What is Pareto dominance and why does it matter?**
A strong answer defines it correctly (at least as good on every objective, strictly better on one), explains that it removes candidates without value judgements, and notes that it does not resolve near-ties. A weak answer treats it as a synonym for "better".

**Q3. Your model meets the latency budget but the product misses its deadline. What do you check first?**
A strong answer decomposes the end-to-end path and looks for a dominant non-model term such as radio transport, sensor buffering, or human acknowledgement. A weak answer proposes further model optimization.

**Q4. Why is p99 more important than p50 for a clinical alert?**
A strong answer connects the tail to the clinical deadline and to safety: the fraction of alerts that arrive late is the fraction of patients exposed. A weak answer says p99 is "more conservative" without explaining the consequence.

**Q5. How do you account for memory in a device with 256 KiB of SRAM?**
A strong answer distinguishes weights in flash from activations and workspace in SRAM, measures the runtime high-water mark, and includes the runtime and fragmentation. A weak answer quotes the model file size.

**Q6. What would make you reject a candidate that passes every numeric constraint?**
A strong answer names subgroup harm, calibration shift at the operating point, absence of a regulatory path, unmeasured fallback behavior, or a thermal profile that invalidates the latency evidence. A weak answer cannot imagine a non-numeric rejection.

**Q7. How does compression change the regulatory status of a model?**
A strong answer states that a change in outputs is a change in the device, that the PCCP may or may not cover it, and that the impact assessment must be evidence-based. A weak answer calls quantization a build optimization.

**Q8. What is the binding constraint, and how do you find it?**
A strong answer defines it as the constraint a marginal improvement would relax first, and describes computing the frontier and examining which ceiling the feasible candidates sit against. A weak answer guesses from intuition.

**Q9. How do you compare a vendor's TOPS number with your measured latency?**
A strong answer refuses the comparison: TOPS is peak integer throughput at a stated precision, while latency depends on memory bandwidth, operator support, batch size, and runtime. A weak answer scales TOPS linearly to estimate time.

**Q10. What evidence would convince you that a quantized model is safe to ship?**
A strong answer asks for paired comparison on a frozen held-out set, calibration and subgroup results at the clinical threshold, target-hardware latency at p99, energy and thermal data, and a documented change path. A weak answer asks for overall accuracy.

**Q11. How would you present a frontier to a non-technical executive?**
A strong answer uses one projection (accuracy versus latency, or cost versus battery life), states the constraints that are not on the axes, and gives a recommendation with its residual risk. A weak answer presents the full table and expects the executive to choose.

**Q12. When is not compressing the right decision?**
A strong answer: when the baseline already meets every constraint with margin, because compression adds accuracy risk, validation cost, and regulatory burden. A weak answer assumes compression is always required for edge deployment.

## 15. Key terms

- **Arithmetic intensity** — operations per byte moved; determines whether a workload is memory- or compute-bound.
- **AUROC** — probability that a random positive is ranked above a random negative; a ranking metric, not a deployment threshold metric.
- **Batch size** — number of inputs processed together; must be stated with any latency or throughput claim.
- **Binding constraint** — the constraint a marginal improvement would relax first.
- **BOM** — bill of materials; per-unit component cost at a stated volume.
- **Brier score** — mean squared error of predicted probabilities; lower is better.
- **Calibration (probability)** — agreement between predicted risk and observed frequency.
- **Calibration (quantization)** — selection of numeric ranges from representative data.
- **Cold start** — first inference after model load, with caches and clocks not warmed.
- **Constraint ledger** — table of limits with value, unit, method, owner, and hard/soft flag.
- **Duty cycle** — fraction of time a function is active.
- **ECE** — expected calibration error; bin-weighted gap between confidence and accuracy.
- **Energy per inference** — joules for one inference, measured over a stated cycle.
- **Feasible set** — candidates satisfying every hard constraint.
- **FLOP** — floating-point operation; a MAC counts as two.
- **GMLP** — Good Machine Learning Practice, joint FDA/Health Canada/MHRA principles, 2021.
- **HIL** — hardware-in-the-loop; testing with real sensors, power, and thermal conditions.
- **Knee point** — frontier point beyond which additional resource spend buys little accuracy.
- **Latency p99** — 99th percentile time per inference under stated conditions.
- **MAC** — multiply-accumulate; one multiply and one add.
- **Model size** — serialised bytes on storage; not a memory budget.
- **NRE** — non-recurring engineering; one-time design cost.
- **Objective** — quantity to be optimized within the feasible region.
- **Pareto frontier** — set of non-dominated candidates.
- **PCCP** — predetermined change control plan; FDA pathway for planned AI modifications.
- **Peak memory** — maximum simultaneously resident bytes at runtime.
- **PTQ** — post-training quantization.
- **QAT** — quantization-aware training.
- **Ridge point** — arithmetic intensity at which a machine transitions from memory- to compute-bound.
- **Roofline** — performance model bounded by peak compute and peak bandwidth.
- **Scale** — real units represented by one integer step.
- **Sensitivity** — true positive rate at the operating threshold.
- **Specificity** — true negative rate at the operating threshold.
- **SQNR** — signal-to-quantization-noise ratio, in decibels.
- **TDP** — thermal design power; sustained heat a package is designed to dissipate.
- **Throttling** — clock or voltage reduction to remain within thermal limits.
- **Throughput** — inferences per second at a stated batch size.
- **Utilisation** — fraction of peak compute or bandwidth actually used.
- **Warm start** — steady-state inference with caches and clocks warm.
- **Zero point** — integer code representing real zero in affine quantization.

## 16. Sources

All URLs accessed 25 October 2026.

**Benchmarks**
- MLCommons, "Benchmark MLPerf Inference: Datacenter", v6.1. https://mlcommons.org/benchmarks/inference-datacenter/
- MLCommons, "Benchmark MLPerf Inference: Edge", v6.1. https://mlcommons.org/benchmarks/inference-edge/
- MLCommons, "Where the Industry Is Investing: A Look at MLPerf Inference v6.1", September 2026. https://mlcommons.org/2026/09/chairs-mlperf-inference-v6-1/
- MLCommons, "Benchmark MLPerf Inference: Mobile". https://mlcommons.org/benchmarks/inference-mobile/
- MLCommons, "MLPerf Tiny v1.2 Results", April 2024. https://mlcommons.org/2024/04/mlperf-tiny-v1-2-results/
- MLCommons, `tiny_results_v1.2` summary. https://github.com/mlcommons/tiny_results_v1.2/blob/main/summary.csv

**Performance modeling**
- Williams, S., Waterman, A., Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM 52(4), 2009. https://doi.org/10.1145/1498765.1498785

**Runtimes and formats**
- ONNX Runtime, "Graph optimizations". https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- PyTorch, "ExecuTorch `.pte` file format". https://docs.pytorch.org/executorch/stable/pte-file-format.html
- Google AI Edge, "LiteRT for Microcontrollers overview". https://developers.google.com/edge/litert/microcontrollers/overview

**Standards and regulation**
- IEC 62304:2006+AMD1:2015, "Medical device software — Software life cycle processes". https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019, "Medical devices — Application of risk management to medical devices". https://www.iso.org/standard/72704.html
- FDA, "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions", final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- FDA, Health Canada, MHRA, "Good Machine Learning Practice for Medical Device Development: Guiding Principles", October 2021. https://www.fda.gov/media/153486/download
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6, "Medical Devices Regulation (IVDR) and the Artificial Intelligence Act (AIA)", June 2025. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

**Thermal measurement**
- Bantock, D., et al., "Mitigating Interactive Performance Degradation from Mobile Device Thermal Throttling", University of Southampton ePrints. https://eprints.soton.ac.uk/439709/1/tus_accepted_paper.pdf
- Guo, Y., et al., "Freeze It If You Can: Challenges and Future Directions in Mobile Benchmarking", HotMobile 2017. https://yaoguopku.github.io/papers/Guo-HotMobile-17.pdf

> **Note:** MLPerf results are vendor-submitted. Thermal studies cited here are independent academic measurements. The Section 6 case is constructed for teaching and is not a measured product.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Next topic: [`02_target-hardware-and-performance.md`](02_target-hardware-and-performance.md)
- Related topic: [`03_the-compression-family.md`](03_the-compression-family.md)
- Related topic: [`10_validation-on-target-hardware.md`](10_validation-on-target-hardware.md)
- Regulatory depth: [`09_medical-and-regulatory-constraints.md`](09_medical-and-regulatory-constraints.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
