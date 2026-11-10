---
case_id: "CASE-M35-02"
case_title: "Battery-powered industrial vibration monitor: choose the sampling, feature and transmission budget"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "streaming-systems"
domain: "Industrial edge monitoring — rotating machinery condition monitoring on a microcontroller node"
primary_pattern: "bounded acquisition"
secondary_patterns: ["anomaly review", "loss accounting"]
evidence_level: "illustrative"
source_basis: "illustrative"
source_status: "OPEN — the edge-arm-firmware source inventory names ADC/DMA, RTOS, CMSIS-DSP and power topics; no asset recordings, accelerometer datasheet, bearing geometry, maintenance log or target trace was supplied"
as_of_date: "2026-11-10"
owner_role: "reliability engineering manager (decision owner); embedded lead (implementation owner); maintenance planner (action owner)"
source_index: "INDEX.md"
planned_from: "PLAN-M35-02"
review_trigger: "Change to the sampled band, sensor mounting, bearing geometry, reporting cadence, radio technology, battery chemistry, threshold policy or maintenance cost basis"
---

# Case study: Battery-powered industrial vibration monitor

## 1. The One-Line Framing

Given **a 2-second tri-axial accelerometer burst captured once every 15 minutes on a battery-powered node bolted to a pump bearing housing**, decide **which envelope and band-energy features to compute on the node, which scalar summary to transmit, and when to raise an alert**, for the **reliability engineering manager who owns the maintenance budget**, to catch degrading bearings before they become unplanned downtime, subject to **the node surviving 24 months on one non-replaceable cell and never reporting a feature whose sample chain is provably broken**, compared with **a four-weekly manual route-inspection baseline and a raw-waveform-upload alternative**.

The task is a periodic feature-extraction and threshold decision, executed 96 times per day per asset, with a human maintenance planner consuming the resulting alerts on a shift cadence. It is a monitoring decision, not a protection function: nothing on this node trips a machine, and no output is permitted to be treated as a safety interlock.

**Evidence declaration.** The node, the plant, the 48 pumps, the bearing geometry, the current draw, the radio duty cycle, the cost figures, the failure rate and every numerical output below are constructed teaching values. They are selected so that each arithmetic step is auditable; they are not measurements of any machine, product or site. Values that follow from the stated inputs are labeled DERIVED and were recomputed independently. Values that are inputs are labeled ILLUSTRATIVE. Where a standard or a named method appears, it identifies a technique whose applicability must be verified, not a compliance conclusion.

The decisive tradeoff is specific and slightly counter-intuitive: the cheapest-looking architecture — sample the full band continuously and transmit raw windows — is not merely expensive, it is **560 times** more energy-expensive per day than the burst architecture, which converts the problem from "which radio" into "which duty cycle". The second tradeoff is that the alert threshold, not the sampling design, dominates the annual cost of ownership: moving a Gaussian-tail threshold from 4σ to 4.5σ cuts illustrative false-alert labour from about €6,232 to about €669 per year, which is larger than the entire manual-inspection baseline it replaces.

Three naive framings fail here. "Sample fast enough for the highest frequency of interest" is incomplete, because the decimation that makes the buffer fit also folds any energy above the new Nyquist frequency back into the band where the bearing fault lines live. "Compute an FFT on the node" is incomplete, because the transform is 4.8 ms while the filter bank that feeds it is 38–55 ms, so the timing budget is set by filtering, not by the transform. And "send the spectrum" is incomplete, because the report itself is 212 bytes and the radio is the second-largest energy consumer after the accelerometer, so the report cadence — not the model — sets battery life.

**Interview cue.** A reader should be able to state in thirty seconds: unit (one asset-burst), output (feature vector plus optional alert), beneficiary (reliability manager), baseline (four-weekly route inspection), binding constraint (24-month single-cell life with a broken-chain prohibition), evidence basis (illustrative).

**Acceptance check.** Input, output, beneficiary, baseline, constraint and evidence basis are all readable from this section, and the output is something a named role would actually act on.

## 2. Context & Background

### 2.1 The physical setting

The illustrative plant runs 48 centrifugal pumps on a continuous process. Each pump is driven by a 30 Hz induction motor through a coupled shaft; the monitored bearing is the pump drive-end rolling-element bearing. Productive operation is 24 hours per day, so every rate in this case uses a 24-hour denominator — unlike a shift-limited operation, where using 24 hours would understate hourly hazard and labour rates.

The vibration node is a coin-sized device clamped to the bearing housing with a stud mount. The mount is the single most consequential mechanical decision in the case: a stud mount transmits the housing's high-frequency content faithfully, a magnetic disc attenuates it, and a hand-held probe held at an angle attenuates and low-passes it in a way that varies with the technician. The node's firmware cannot detect a bad mount, which is why mounting is an entry in the assumption register rather than a software feature.

Each node holds one lithium thionyl chloride cell of nominal 9,000 J usable capacity at the operating temperature range declared for the site. The cell is potted into the housing; replacing it means replacing the node. That single sentence drives the entire energy analysis: the decision is not "minimise power", it is "fit the required evidence into a fixed 24-month energy budget with margin for the coldest month".

### 2.2 The workflow the node interrupts

The current workflow is a route inspection. A technician walks a fixed route every four weeks with a handheld analyser, takes a spot reading at each of the 48 assets, compares the reading against a route card of limits, and writes a verdict. Findings feed a maintenance planner who schedules a repair window.

Three properties of that baseline matter later. First, the sampling interval is 28 days, so a fault that develops in ten days is caught at best 18 days late, and on average about 14 days late under a uniform onset assumption. Second, the reading is a single spot measurement taken while the machine runs at whatever load the process happens to demand, so readings across visits are not strictly comparable. Third, the route is labour, not capital: stopping it recovers a recurring cost immediately, which makes the business case sensitive to avoided failures rather than to hardware price.

The node changes the workflow in one place only: it replaces the spot reading with a scheduled burst and a transmitted feature vector. The planner still decides, the technician still repairs, and the route card still defines what "actionable" means. A design that also tried to schedule repairs would be a different case with a different owner.

### 2.3 Stakeholders and authority

| Role | Authority in this decision | What they cannot do |
|---|---|---|
| Reliability engineering manager | Approve or reject the node programme; own the annual maintenance budget | Override the plant's isolation and permit-to-work rules |
| Embedded lead | Accept or reject a firmware design as infeasible on the target | Widen the alert threshold to hit a schedule |
| Maintenance planner | Act on alerts; schedule repair windows | Declare a bearing healthy without a physical check |
| Condition-monitoring analyst | Tune band limits and thresholds; adjudicate ambiguous spectra | Approve capital expenditure |
| Site electrical authority | Approve installation and enclosure rating | Accept an unverified radio certification |

The split matters because the alert threshold is a joint parameter. The embedded lead owns whether the node can compute a feature; the analyst owns whether that feature's value means anything; the manager owns the labour cost of acting on it. A single "the team decided" statement would hide the fact that tightening the threshold is cheap for the embedded lead and expensive for the planner.

### 2.4 The baseline and its documented weaknesses

| Baseline property | Value used in this case | Documented weakness |
|---|---|---|
| Route interval | 28 days | Detection delay up to 28 days; mean about 14 days |
| Measurement | One spot reading per asset | Load-dependent; not comparable visit to visit |
| Instrument | Handheld analyser, technician-held probe | Probe angle and pressure vary; high-frequency attenuation uncontrolled |
| Verdict rule | Route-card limits per asset class | Limits are static and were set from fleet history, not per asset |
| Cost | 48 assets × 4 visits × €145 = €27,840/yr | Labour scales with fleet size, not with failures |
| Coverage | All 48 assets on the same route | No continuous record between visits |

The baseline persists because it is cheap per visit, familiar to the technicians, and produces a document that satisfies the plant's maintenance-management process. It is not irrational: a node programme that costs more than €27,840/yr in labour and hardware while producing alerts nobody trusts would be strictly worse.

### 2.5 Prior attempts and organisational history

An earlier pilot attached a commercial vibration logger to six assets and uploaded raw waveforms hourly over Wi-Fi. It produced excellent data and was abandoned for two reasons the case takes seriously: the loggers needed cell replacement every few weeks, and the resulting volumes were never analysed, so the plant accumulated data without accumulating decisions. Both failures are architectural, not incidental — they are the direct motivation for computing features on the node and for a 15-minute cadence.

A second attempt added a simple RMS threshold to the plant historian. It raised so many alerts during load changes that the planner stopped reading them. That history is the reason the alert policy in Section 8 uses band ratios and a persistence rule rather than a single broadband level.

### 2.6 Scale, timing and boundaries

| Quantity | Value | Consequence |
|---|---|---|
| Assets in scope | 48 pumps | Fleet-level alert economics, not single-asset |
| Bursts per asset per day | 96 (15-minute cadence) | 4,608 bursts/day for the fleet |
| Sampling window | 2.0 s per burst | Frequency resolution 0.5 Hz |
| Sample rate | 25,600 samples/s/channel | Nyquist 12.8 kHz before decimation |
| Deployment horizon | 24 months on one cell | Energy budget 9,000 J / 730 d ≈ 12.33 J/day |
| Decision cadence | Planner reads alerts once per shift | Sub-shift alerting is unnecessary |
| Out of scope | Machine protection, trip logic, ISO certification, bearing-life prediction, spare-parts optimisation | Handed to the plant protection system and the planner |

The timing boundary is worth stating precisely: the node's job is to make a measurement every 15 minutes and to transmit a small summary; the planner's job is to decide within a shift. A latency budget measured in seconds would be solving a problem the plant does not have, and a latency budget measured in weeks would reproduce the baseline's weakness.

## 3. The Business Question

The manager's question, paraphrased from an illustrative planning conversation rather than quoted: "Can a sealed battery node on 48 pumps produce trustworthy bearing-degradation evidence for two years without a cell change, and will the alerts be specific enough that my planner still reads them in month eighteen?"

| Decision contract | Bounded definition |
|---|---|
| Question | Choose the on-node sampling/decimation chain, the transmitted feature set, and the alert rule — or keep route inspection. |
| Decision owner | Reliability engineering manager; embedded lead holds a technical veto on feasibility. |
| Trigger and deadline | Annual maintenance-budget planning cycle; node hardware must be frozen before the enclosure order. |
| Available actions | Keep route inspection; add RMS-only historian alerts; deploy burst-feature nodes; deploy continuous raw-waveform nodes; defer and instrument six assets first. |
| Value | Earlier detection of degrading bearings, and route labour released for other work. |
| Downside | False alerts consuming planner time; missed degradation; dead nodes mid-horizon; unusable spectra from a bad mount. |
| Objective | Minimise annual total cost of monitoring plus failure consequences, subject to a stated alert-precision floor. |
| Hard constraints | 24-month single-cell life; no transmitted feature without an intact sample chain; no protection function; alert precision floor of 0.75 at the chosen threshold. |
| Scope | 48 pumps, one drive-end bearing each, 24 months, one site, one radio technology. |
| Out of scope | Repair scheduling optimisation, spare-parts policy, warranty adjudication, safety-instrumented-function claims. |

The objective is deliberately written as total cost rather than as detection accuracy, because the two diverge here. A threshold low enough to catch every incipient fault also produces false alerts whose labour cost exceeds the inspection baseline it replaced. Section 6 shows the divergence numerically and Section 9 makes it a guardrail: an architecture that improves detection while breaching the precision floor is rejected, not merely noted.

The proxy metric is alert precision at a fixed recall target. Precision is measurable in the pilot because a planner's physical check adjudicates every alert. Bearing life extension is not measurable in the pilot because it requires years and a repair record; any claim about it must stay a modeled projection.

### 3.1 What an expensive wrong decision looks like

Two errors have different shapes. A **false negative** — missing a degrading bearing — costs an unplanned failure: the illustrative basis is €4,200 per event including production loss and emergency labour. A **false positive** — sending the planner to a healthy machine — costs an inspection: €58.5 of planner and technician time including travel within the plant.

The asymmetry is about 72:1 per event, but the *rates* invert the intuition. In a healthy fleet, most alerts are false simply because most bearings are healthy at any moment, so a threshold tuned only against miss cost will drown the planner. That is exactly why the earlier RMS pilot failed, and why the persistence rule in Section 8b exists.

### 3.2 Why a model score does not answer the question

A classifier that outputs "fault probability 0.62" does not tell the manager whether to spend €4,560 on nodes. The decision needs three quantities the score does not contain: the energy cost of producing that score for two years, the labour cost of acting on its positive outputs at the fleet's base rate, and the evidence that the score's inputs were physically valid at capture. The case therefore treats the score as one component inside an evidence ledger, not as the answer.

## 4. Assumptions & Constraints

### 4.1 Assumption and constraint register

| ID | Assumption or constraint | Type | Evidence / check | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Accelerometer bandwidth extends to at least 10 kHz with ±3 dB flatness | sensor | datasheet plus a shaker check | unverified | bearing-resonance band is attenuated before sampling | choose a wider sensor | embedded lead |
| A-02 | Stud mounting transmits 1–6 kHz content without uncontrolled loss | mechanical | comparative mount test | unverified | fault lines are buried in mount loss | mandate stud mount, reject magnetic | reliability manager |
| A-03 | Shaft speed is 30 Hz ±1% in steady state | process | tachometer log | illustrative | order-tracked band edges drift | widen bands or add a tacho | analyst |
| A-04 | Bearing geometry is 9 balls, ball diameter 7.94 mm, pitch diameter 39.04 mm | mechanical | bearing catalogue | illustrative | fault frequencies shift | recompute frequencies from the real part | analyst |
| A-05 | Usable cell capacity is 9,000 J over the temperature range | energy | discharge test at both temperature extremes | unverified | 24-month life claim fails in cold months | reduce cadence or enlarge cell | embedded lead |
| A-06 | Accelerometer plus front-end draws 16.5 mW while enabled | energy | current trace | illustrative | energy ledger is wrong by the dominant term | re-measure | embedded lead |
| A-07 | MCU draws 29.7 mW active at the chosen clock | energy | current trace during a burst | illustrative | compute energy is misestimated | re-measure | embedded lead |
| A-08 | Radio draws 138.6 mW for 90 ms per report | energy | current trace per transaction | illustrative | report energy misestimated | re-measure | embedded lead |
| A-09 | Sleep current is 26.4 µW | energy | long-interval coulomb count | illustrative | standby drain dominates and is missed | re-measure over 24 h | embedded lead |
| A-10 | Alert base rate is about 6 degrading bearings per fleet-year | maintenance | two years of repair records | illustrative | precision arithmetic changes entirely | recompute from the site's own history | reliability manager |
| A-11 | False alert costs €58.5 and unplanned failure €4,200 | economics | planner time study plus downtime record | illustrative | economic ranking may reverse | rebuild the ledger with site costs | reliability manager |
| A-12 | Decimation to 3,200 Hz preserves the envelope band of interest | signal | sweep test with a known fault | unverified | aliasing buries or fabricates fault lines | raise the decimated rate | analyst |
| C-01 | A burst must complete within its 15-minute slot with ≥3× timing margin | timing | worst-case execution trace | required | bursts collide with the next slot | reduce work or lengthen slot | embedded lead |
| C-02 | No feature is transmitted from a burst with a detected gap, overrun or saturation | integrity | injected fault tests | required | corrupted feature drives a wrong repair | drop and count the burst | embedded lead |
| C-03 | The node cannot be treated as a protection device | scope | design review sign-off | confirmed | unsafe reliance on a monitor | keep the protection system independent | site electrical authority |
| C-04 | Threshold and band limits are frozen before pilot data are examined | evaluation | versioned configuration record | required | threshold fitted to the pilot inflates precision | re-run on a held-out period | analyst |
| C-05 | Route inspection continues unchanged during the pilot | evaluation | route records | required | no comparable baseline exists | extend the pilot | reliability manager |
| C-06 | Any transmitted field is limited to features, node identity, time and quality flags | privacy/security | payload schema review | required | raw plant acoustics leave the site | block the field | security owner |

### 4.2 The three assumptions most likely to reverse the design

The first is **A-05, the usable cell capacity**. Every life figure in Section 6 is a linear function of it. A cell that delivers 6,000 J rather than 9,000 J at the site's winter temperature converts a 1,263-day design into an 842-day design, which fails the 730-day requirement with no other change. The mitigation is not a software optimisation; it is either a larger cell or a longer cadence, and the cadence lever is bounded below by detection latency.

The second is **A-12, the decimation boundary**. The node cannot hold a full 300 KiB three-axis buffer in 96 KiB of SRAM, so the chain must decimate. Decimation is the one operation in the chain that can *create* a fault signature that is not present, by folding an unrelated high-frequency source — a gear mesh, a VFD switching harmonic, a neighbouring machine — down onto the ball-pass frequency. A design that gets every other step right and this one wrong produces confident, reproducible, wrong alerts.

The third is **A-10, the alert base rate**. Precision is a function of prevalence. At six degrading bearings per fleet-year against thousands of evaluations, a threshold that looks reasonable in isolation produces a false-alert flood; the same threshold at a site with thirty degrading bearings would look excellent. Any precision number quoted without its base rate is uninterpretable, which is why Section 9 fixes the denominator.

### 4.3 Constraints that are choices, and choices that look like constraints

Three items are commonly mislabeled. The 15-minute cadence is a *choice*, constrained only by the 24-month energy budget and by the requirement that detection latency stay below the planner's scheduling horizon; it is not a physical limit. The 2-second window is a *choice* driven by the 0.5 Hz resolution needed to separate the ball-pass frequency from its shaft-rate sidebands; a 1-second window would give 1 Hz and merge them. The 3,200 Hz decimated rate is a *choice* bounded by the highest envelope harmonic required to characterise the fault, not by the sensor.

Conversely, the prohibition on transmitting raw waveforms is close to a hard constraint at this site: the plant's network egress review treats continuous acoustic data as sensitive process information, and the 24-month energy budget excludes it by three orders of magnitude anyway.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is **one burst from one asset**: 2.0 seconds of tri-axial acceleration at 25,600 samples/s/channel, captured on a defined trigger, reduced on the node to a feature vector, transmitted with a header and quality flags, and stored as one row keyed by `(asset_id, burst_start_utc, node_fw_version, config_version)`.

| Field group | Fields | Units / encoding | Availability at decision time |
|---|---|---|---|
| Identity | `asset_id`, `node_serial`, `mount_type`, `bearing_part_number` | string / enum | at capture |
| Time | `burst_start_utc`, `burst_end_utc`, `node_monotonic_ticks`, `clock_offset_estimate` | UTC seconds; ticks at 32,768 Hz | at capture, offset known only after sync |
| Capture quality | `sample_count`, `gap_count`, `overrun_count`, `saturated_sample_count`, `adc_range_g` | counts; g | at capture |
| Envelope features | `env_band_k1..k4_energy`, `env_total_energy`, `bpfo_1x..3x_amplitude`, `bpfi_1x_amplitude` | g² (band energy); g (amplitude) | at capture |
| Time-domain features | `rms_g`, `peak_g`, `crest_factor`, `kurtosis`, `zero_crossings` | g, g, dimensionless, dimensionless, count | at capture |
| Context | `load_fraction`, `shaft_speed_hz`, `ambient_c`, `battery_mv` | fraction, Hz, °C, mV | at capture, load from the process historian |
| Outcome | `planner_verdict`, `repair_finding`, `verdict_timestamp` | enum, free text, UTC | weeks later — never a feature |

The **label** is not a model target in the usual sense. It is the planner's adjudicated verdict after a physical check, recorded as one of `no_fault_found`, `lubrication`, `misalignment`, `bearing_degradation`, `other_mechanical`, `sensor_fault`, with a confidence and the technician's free-text finding. Label maturity is deliberately slow: a verdict exists only after the check, so any evaluation must respect the fact that a burst captured on day 1 receives its verdict on day 5 or day 40.

The **population** is 48 assets over 24 months, subject to exclusions: bursts during scheduled shutdowns, bursts during a repair window, bursts with a flagged capture-quality violation, and bursts whose asset was reconfigured mid-horizon. Exclusions are counted, never silently dropped; a node that loses 20% of its bursts to overruns has a different evidence base than one that loses 0.2%.

**Splits.** Because the unit is a burst and bursts from one asset are highly correlated, a random burst split would leak. The design uses three partitions: 12 assets for threshold development, 12 for threshold confirmation, and 24 held out for the final evaluation. Within an asset, all bursts stay in that asset's partition. Time is *not* used as the primary split boundary, because seasonality would confound it; time is used as a secondary check to detect drift within a partition.

**Join semantics.** Burst rows join to the process historian on `(asset_id, minute)` to obtain load and shaft speed, with a one-to-many risk if the historian emits multiple rows per minute. The rule is explicit: take the historian row with the latest timestamp at or before `burst_start_utc`, and record which row was used. Joining to any row inside the burst window would admit information created after capture.

### 5b. Tools & Stack

| Component | Proposed role | Required evidence |
|---|---|---|
| MEMS accelerometer, 3-axis, ≥10 kHz bandwidth | Transduce housing vibration | Datasheet bandwidth curve plus shaker verification |
| Analog anti-alias filter, 4th-order, fc ≈ 9 kHz | Attenuate content above 12.8 kHz before the ADC | Measured response, not a simulation only |
| 12-bit SAR ADC at 25,600 samples/s | Convert with ≥70 dB dynamic range in the band | Noise and linearity measurement |
| DMA controller, double-buffered, 1,024 frames per block | Move samples without CPU involvement | Ownership trace with timestamps |
| Cortex-M4F at 84 MHz with FPU | Filter, rectify, decimate, transform, classify | Worst-case execution trace per stage |
| CMSIS-DSP filter and transform routines | Fixed and floating filter primitives, real FFT | Version-pinned library, call-site audit |
| FreeRTOS with two tasks (acquisition, processing) | Rate-monotonic split with explicit priorities | Priority and blocking analysis |
| Sub-GHz LPWAN radio, 212-byte reports | Transmit features and alerts | Duty-cycle and certification record |
| Encrypted flash journal, 256 KiB sector | Buffer reports across radio outages | Wear and recovery test |
| Plant historian (read-only) | Load and speed context | Schema and latency record |
| Gateway service and time-series store | Ingest, deduplicate, serve the planner UI | Idempotency key definition |

The stack is deliberately conservative. Floating point is available on this core, so the filter design uses single precision for coefficient storage and double precision nowhere; the fixed-point alternative is analyzed in Section 9 as a variant rather than assumed. The transform is a real-input FFT of 6,400 points, which is not a power of two — the implementation plan therefore uses a mixed-radix routine or zero-pads to 8,192, and Section 8c records that as an explicit decision rather than an implementation detail.

### 5c. Method Summary

The chain is: sample → anti-alias → block → band-pass → rectify → low-pass → decimate → transform → features → threshold → report.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Four-weekly route inspection (baseline) | Route records and cost basis | Cheap, familiar, human judgement at the machine | 28-day detection delay; load-dependent spot readings | Retained as comparator | Fleet grows past about 70 assets |
| Broadband RMS threshold on the historian | Prior pilot history | Trivial to compute and explain | Fires on load changes; no frequency discrimination | Rejected — measured false-alert flood in the earlier pilot | Never, unless load is held constant |
| Continuous raw-waveform upload | Energy arithmetic in §6.6 | Maximum analytical freedom | 3,992 J/day versus a 12.3 J/day budget | Rejected on energy, not on merit | If mains power becomes available |
| Burst + envelope features (chosen) | This design | Fits the energy budget; produces frequency-selective evidence | Requires a valid mount and a correct decimation boundary | Chosen, illustrative | If the mount or band changes |
| Burst + full on-node spectrum | Transform cost arithmetic | Richer evidence than four band energies | 148 bytes cannot carry 6,400 bins; report grows | Rejected at this cadence | If the radio budget grows tenfold |
| Burst + on-node classifier | Same feature basis | Could suppress nuisance alerts | Needs labelled fleet data that does not yet exist | Deferred, not tested | After 12 months of adjudicated alerts |

The chosen method computes a bearing-envelope spectrum because a localised raceway defect excites a structural resonance that is amplitude-modulated at the ball-pass rate. Envelope analysis recovers the modulation; a plain spectrum of the raw signal would show the resonance and hide the fault rate. The four band energies are non-overlapping slices of the envelope spectrum placed around the first harmonics of the ball-pass frequencies, which makes them interpretable individually rather than only as a composite score.

Two alternatives were considered and are recorded as untested: an order-tracked variant that resamples against shaft speed, and a cepstral variant that would suppress the transfer path. Both are defensible; neither was implemented, and the case does not claim they were evaluated.

### 5d. Data provenance, quality, exclusions and leakage controls

| Source ID | Content | Access | Limitation |
|---|---|---|---|
| SRC-01 | Node firmware configuration and feature schema | Internal, versioned | Describes the design, not its performance |
| SRC-02 | Two years of route-inspection records | Internal, restricted | Verdicts are technician judgement, not tear-down reports |
| SRC-03 | Repair records with parts used | Internal, restricted | Free text; part numbers incomplete for older jobs |
| SRC-04 | Process historian load and speed | Internal, read-only | Minute-grain; missing during historian restarts |
| SRC-05 | Bearing catalogue geometry | Vendor catalogue | Geometry only; no defect-initiation data |
| SRC-06 | Accelerometer and ADC datasheets | Vendor public documents | Typical curves, not unit-specific |

Quality controls are explicit. Duplicate reports are identified by `(asset_id, burst_start_utc, node_serial)` and resolved by keeping the earliest accepted copy; this matters because the LPWAN stack may retry. Out-of-order arrival is normal and is handled by ordering on capture time, never on receipt time. Saturation is detected from `saturated_sample_count` and disqualifies the burst's amplitude features while leaving its band-ratio features usable only if the saturated samples fall outside the analysis band — a conditional that must be checked, not assumed.

**Leakage controls, stated as prohibitions.** No feature may be computed from data captured after the burst window. No verdict recorded after `burst_start_utc + 28 days` may be used to label that burst in threshold development, because the route inspection would otherwise be informed by the node's own alert. Threshold and band limits are frozen in a versioned configuration before the confirmation partition is examined. The process historian join uses only rows at or before burst start.

The largest unavoidable bias is **verification bias**: an alert is only adjudicated if it was raised, so the pilot observes outcomes for alerted bursts and not for silent ones. Precision is therefore estimable and recall is not, from planner verdicts alone. Section 9 makes this the primary validity threat and prescribes a sampled audit of silent assets to bound it.

### 5e. Reproducibility and method-selection ledger

| Artifact | Location | Version | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `FW-01` node firmware | Internal repo | tag `vib-1.0.0` | Sampling, features, alert rule | Restricted | Requires the target board to execute |
| `CFG-01` band and threshold configuration | Internal config store | `cfg-2026-11-a` | Frozen thresholds and band edges | Restricted | Values are illustrative |
| `CALC-01` energy and memory ledger | This document, §6 | 2026-11-10 | Derive life and sizing | Public in course | Arithmetic only; inputs illustrative |
| `RUN-01` synthetic chain test | Offline script, not retained | — | Check alias and decimation arithmetic | Not retained | Documentation smoke, not a device test |
| `DAT-01` pilot dataset | Not available | — | Would support precision estimation | Unavailable | No recordings were supplied |

Reproduction of the arithmetic requires only the inputs stated in each formula block; reproduction of the *design's* behaviour requires the target board, the exact accelerometer and a shaker. The distinction is stated because a reader can verify every number in Section 6 and still know nothing about how this node behaves on a real pump.

**Method-selection ledger.** The envelope method was selected over raw-spectrum thresholding because the fault evidence is modulation, not absolute energy. It was selected over a learned classifier because the fleet has no adjudicated label history at the start of the horizon, and a classifier trained on the six events per year the fleet produces would have a label base rate too low to fit anything stable. It was selected over order tracking because the process holds speed within ±1% in steady state, which Section 6.10 shows costs about 1.2 bins of smearing — tolerable at 0.5 Hz resolution, not tolerable at 1 Hz.

<!-- SECTION-MARKER-2 -->
