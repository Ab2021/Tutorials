# Safety and Standards

> Module 24 — Hardware and Robotics Labs
> Topic 1 of 9
> Estimated study time: 4 hours
> Prerequisites: 14A
> Version 2026.3

## 1. Why this topic exists

A hardware lab is the first place in this program where a mistake can hurt someone. Every other module fails into a stack trace, a bad metric, or an invalid cohort. A bench mistake fails into a burn, a shock, a fire, a destroyed board, or a person connected to a circuit that was never designed to touch a person.

The industry problem is not that engineers are careless. It is that the boundary between a safe teaching bench and a regulated medical device is invisible unless someone draws it. A 5 V development board and a 5 V patient-connected front end look almost identical on a desk. They are governed by different standards, different isolation requirements, different evidence, and different supervision. Teams that cannot name the boundary cross it by accident, then report a bench measurement as if it established something about a product.

Three decisions depend on this topic:

1. **What can hurt someone in this experiment.** The answer is a hazard inventory, not a feeling. It covers electrical energy, stored energy, thermal energy, mechanical energy, chemical exposure, optical exposure, and battery failure.
2. **Which standard governs the activity.** A lab instrument, a medical device, a robot, and a piece of health software each sit under a different family. The standard determines the limits, the test methods, and what counts as evidence.
3. **What the result does and does not establish.** A bench measurement with a multimeter is not a type test. A pre-compliance scan is not a compliance certificate. A simulated emergency stop is not a safety-rated function.

The commercial cost of getting this wrong is concrete. A late electrical-safety finding can force an isolation redesign, a new transformer, a new enclosure, and a new accredited test booking. An EMC finding found after tooling is a quarter of engineering time. A robot without a rated safety function can be blocked from a site. A lab injury ends a program and can end a career.

This topic establishes the safety grammar that the other eight topics assume. It does not certify anyone to work on mains equipment, and it does not turn a learner into a qualified electrical worker. It teaches a learner to recognise when an experiment has crossed from a supervised student bench into work that requires a qualified person, an accredited laboratory, or a clinical environment.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. Jewellery is removed and long hair is tied back near machinery. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work.

## 2. Learning outcomes

By the end of this topic you can:

- Build a hazard inventory for a bench experiment across electrical, thermal, mechanical, chemical, optical, and stored-energy categories.
- Classify a circuit as extra-low voltage, mains-referenced, or patient-connected, and state the supervision each classification requires.
- Distinguish a bench measurement, a design verification, a type test, and a compliance certificate, and state which one a given experiment produced.
- Read a leakage-current limit table correctly, including applied-part type, normal condition, and single-fault condition.
- Explain the purpose of basic, supplementary, double, and reinforced insulation, and of means of operator protection (MOOP) and means of patient protection (MOPP).
- Apply a hazard-control hierarchy and show why PPE is the last control, not the first.
- Write a pre-power checklist and a stop-work criterion that another person could audit.
- Map a lab activity to the standards that constrain it, and name the standards the lab cannot satisfy.
- Reconstruct a leakage-current budget from measured voltage, a model impedance, and instrument uncertainty.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

Safety is a chain, not a property of a component. A hazard becomes harm only when a sequence of events creates exposure. The chain has four links: source, path, exposure, and harm. Every control acts on one of the four.

```text
   SOURCE            PATH              EXPOSURE            HARM
   mains voltage --> insulation -----> hand on chassis --> shock
   stored charge --> probe ----------> finger -----------> burn
   servo torque  --> unguarded arc ---> hand in workspace -> crush
   lithium cell  --> thermal runaway -> bench surface ----> fire
   laser beam    --> reflection -----> eye --------------> retinal injury
```

The safety question is always: which link am I breaking, and what evidence shows that the link is broken?

### 3.1 The three electrical zones

The single most useful diagram in a hardware lab separates three zones.

```text
+---------------------------------------------------------------+
| ZONE 3: PATIENT-CONNECTED                                     |
| applied part, isolation, MOPP, leakage limits, clinical use    |
| requires: qualified supervision, isolation transformer,        |
|           defined measurement network, risk file               |
+---------------------------------------------------------------+
| ZONE 2: MAINS-REFERENCED                                      |
| exposed conductor at line voltage, protective earth, fuses    |
| requires: qualified electrical worker, lockout/tagout, PPE     |
+---------------------------------------------------------------+
| ZONE 1: EXTRA-LOW-VOLTAGE BENCH                               |
| current-limited supply, fused USB, no exposed mains           |
| the only zone this course authorises for learners              |
+---------------------------------------------------------------+
```

The zones are not about voltage alone. A 5 V circuit that is galvanically connected to mains is Zone 2. A battery circuit that can deliver a short-circuit current of hundreds of amperes is a stored-energy hazard even though its voltage is low. A circuit that touches a person is Zone 3 regardless of its supply, because the failure mode now includes the person as a conductor.

### 3.2 The control hierarchy

Controls are ranked by how much they depend on human behaviour.

| Rank | Control type | Example | Dependence on the person |
|---:|---|---|---|
| 1 | Elimination | Do not use mains; use a battery or a simulator | None |
| 2 | Substitution | Replace a 24 V actuator with a 12 V torque-limited one | None |
| 3 | Engineering control | Interlock, guard, current limit, fuse, isolation barrier | None if designed correctly |
| 4 | Administrative control | Procedure, training, signage, supervision | High |
| 5 | PPE | Glasses, gloves, lab coat | Highest; fails silently |

A lab that begins with PPE and ends with procedure has the hierarchy inverted. The correct order is to remove the energy, then guard it, then procedure it, and only then dress for the residual.

### 3.3 The evidence ladder inside a lab

The survey file's evidence ladder runs from concept to post-market. Inside a single lab, a shorter ladder applies.

```text
calculation -> simulation -> bench measurement -> design verification -> type test
```

Each rung answers a narrower question.

| Rung | Question | Typical artefact | Not proved |
|---|---|---|---|
| Calculation | What does the model predict? | Spreadsheet, notebook | That the model matches reality |
| Simulation | Does the circuit behave as modelled? | ngspice or LTspice plot | Parasitics, layout, thermal reality |
| Bench measurement | What did this unit do today? | Scope capture, DMM log | That other units behave the same |
| Design verification | Does the output meet the input requirement? | Test report with acceptance criteria | Safety in use, clinical benefit |
| Type test | Does the design type meet the standard? | Accredited test report | Production conformity of every unit |

A type test is performed on a representative sample under the standard's conditions by a competent body. It is not a bench experiment with a better instrument. The distinction matters because a learner who says "we tested leakage current" when they measured a voltage across a model resistor has made a claim the data cannot support.

### 3.4 Risk is a product, not a feeling

Risk is the combination of the probability of occurrence of harm and the severity of that harm. The ISO 14971 model separates the hazard from the hazardous situation and the harm.

```text
hazard --> sequence of events --> hazardous situation --> harm
```

A hot surface is a hazard. A hand reaching into the guard is a sequence of events. Contact with the hot surface is the hazardous situation. A burn is the harm. A control can act on any arrow: remove the hot surface, interlock the guard, train the operator, or require a glove. The control's position determines how much it depends on the operator.

This decomposition is also the reason a risk file cannot be replaced by a severity rating. Two hazards with the same severity can have very different probabilities, and two controls with the same effect can have very different reliability.

### 3.5 The compliance boundary

A lab can demonstrate a mechanism. It cannot demonstrate conformity. Conformity requires a quality management system (QMS), design controls, accredited testing where applicable, clinical evidence where applicable, and a conformity-assessment body.

The boundary is stated once and repeated in every report:

- A lab result is evidence about a prototype under lab conditions.
- It is not a design verification unless it was run against a written requirement with a pre-defined acceptance criterion and controlled configuration.
- It is not a type test unless it followed the standard's method with calibrated equipment and the required sample.
- It is never a compliance certificate.

> **Note:** The phrase "IEC 60601 compliant" is a claim about a product and a conformity-assessment route. A student lab cannot make it. The acceptable phrase is "designed with reference to IEC 60601-1; not tested for conformity."

## 4. Definitions and vocabulary

Every term below appears later in this file or in a sibling topic. The "common misuse" column is the one to read twice.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Hazard | Potential source of harm | Used as a synonym for risk | None |
| Hazardous situation | Circumstance in which a person is exposed to a hazard | Confused with the hazard itself | None |
| Harm | Injury or damage to health, property, or environment | Narrowed to "injury" only | None |
| Risk | Combination of probability of harm and severity of harm | Treated as severity alone | Dimensionless or scored |
| Residual risk | Risk remaining after controls | Assumed to be zero after a control | Same scale as risk |
| Risk control | Action that reduces risk | Called a "fix" without a verification link | None |
| Hazard inventory | Structured list of hazards and energy sources | Replaced by a generic checklist | None |
| Extra-low voltage (ELV) | Voltage below the applicable limit in the governing standard | Assumed to be a universal 30 V/60 V figure | V |
| Safety extra-low voltage (SELV) | ELV system that cannot exceed the limit in normal or single-fault conditions | Confused with a floating supply | V |
| Protective extra-low voltage (PELV) | ELV system that is earthed | Treated as identical to SELV | V |
| Mains-referenced | Circuit conductively connected to the mains supply | Assumed safe because the voltage is "only" 5 V | None |
| Applied part | Part of a medical device that contacts the patient | Used for any enclosure surface | None |
| Type B applied part | Applied part not suitable for direct cardiac contact | Assumed to be "body-safe" in all uses | Classification |
| Type BF applied part | Floating type B applied part | Confused with type CF | Classification |
| Type CF applied part | Floating applied part suitable for direct cardiac contact | Assumed to be required for every ECG | Classification |
| Patient leakage current | Current flowing from an applied part through the patient to earth | Measured with a multimeter and called compliance | µA |
| Touch current | Current flowing from an accessible part to earth through a person | Confused with earth leakage | µA |
| Earth leakage current | Current flowing from the mains protective earth conductor | Confused with patient leakage | µA |
| Patient auxiliary current | Current flowing between patient connections | Ignored in single-lead designs | µA |
| Basic insulation | Insulation providing basic protection against shock | Assumed to be any thin coating | mm or V |
| Supplementary insulation | Independent insulation in addition to basic | Counted twice when it is the same layer | mm or V |
| Double insulation | Basic plus supplementary | Used loosely for "a lot of insulation" | mm or V |
| Reinforced insulation | Single insulation with protection equivalent to double | Treated as a marketing term | mm or V |
| MOOP | Means of operator protection | Confused with MOPP | Classification |
| MOPP | Means of patient protection | Assumed to be satisfied by a MOOP barrier | Classification |
| Creepage distance | Shortest path along an insulating surface | Confused with clearance | mm |
| Clearance | Shortest path through air | Confused with creepage | mm |
| Dielectric strength | Voltage an insulation withstands without breakdown | Called "hipot" without stating the waveform | V AC or V DC |
| Type test | Conformity test on a representative sample to a standard method | Called any bench test with a good instrument | None |
| Routine test | Test performed on every production unit | Confused with type test | None |
| Design verification | Evidence that design output meets design input | Confused with validation | None |
| Design validation | Evidence that the device meets user needs and intended use | Confused with verification | None |
| Pre-compliance | Early in-house EMC testing for margin | Presented as compliance | None |
| Bench measurement | Measurement of one unit under lab conditions | Presented as a type test | None |
| Measurement uncertainty | Quantified doubt in a measured value | Omitted from the report | Same unit as measurand |
| Traceability | Unbroken chain of calibrations to a reference | Claimed because an instrument has a sticker | None |
| Calibration | Comparison against a reference with stated uncertainty | Confused with adjustment | Certificate |
| PPE | Equipment worn to reduce exposure | Used as the primary control | Category or standard |
| ESD | Electrostatic discharge | Treated only as a board-damage issue | V, J |
| ESD wrist strap | Conductive strap that equalises worker and bench potential | Worn loose or over clothing | Ω (typically 1 MΩ series) |
| Lockout/tagout (LOTO) | Procedure that isolates and secures energy sources | Substituted by "I switched it off" | Procedure |
| Arc flash boundary | Distance at which incident energy equals the defined threshold | Confused with the shock approach boundary | mm or cal/cm² |
| Limited approach boundary | Shock boundary outside which unqualified persons need escort | Ignored in lab planning | mm |
| Restricted approach boundary | Shock boundary inside which only qualified workers with PPE may go | Treated as advisory | mm |
| DUT | Device under test | Called "the board" in a report | None |
| EUT | Equipment under test, the EMC term for the DUT | Used interchangeably without definition | None |
| LISN | Line impedance stabilisation network | Called a "filter" | Ω, dB |
| CDN | Coupling/decoupling network | Confused with a LISN | Ω, dB |
| Safety-rated function | Function with a declared performance level or safety integrity level | Claimed for any software check | PL or SIL |
| Performance level (PL) | Discrete level of safety-function reliability in ISO 13849 | Confused with probability of harm | a–e |
| Safety integrity level (SIL) | Discrete level in IEC 61508 and IEC 62061 | Quoted without the standard | 1–4 |
| Emergency stop | Manual function that stops hazardous motion | Assumed to remove all energy | None |
| Guard | Physical barrier that prevents access | Assumed to be an interlock | None |
| Interlock | Device that prevents operation when a guard is open | Assumed to be safety-rated | None |
| Safety envelope | Set of limits within which motion is permitted | Confused with the mechanical workspace | Units per axis |
| Qualified person | Person trained and authorised for the electrical work | Assumed to mean "has a degree" | None |
| NRTL | Nationally Recognised Testing Laboratory | Confused with any test house | Registration |
| Accreditation | Formal recognition of a laboratory's competence | Confused with certification of a product | ISO/IEC 17025 scope |
| QMS | Quality management system | Treated as a document set rather than a process | ISO 13485 |
| DHF | Design history file | Reconstructed after the design is finished | Record set |

## 5. The mechanics, step by step

This section is the lab procedure for qualifying a bench before any powered experiment. It is written so that a partner can audit it. Each step states its input, output, and rationale. The procedure assumes a learner bench with a current-limited supply and no mains work. Steps that would cross into Zone 2 or Zone 3 are marked as stop points.

### Step 1. Write the claim and the claim boundary

**Input:** the experiment question and the decision it informs.

**Output:** one sentence stating what will be measured, on what unit, under what conditions, and what the result will not establish.

**Rationale:** a claim that is not written cannot be bounded. "Measure the noise floor of the ECG front end" is a topic. "Measure the input-referred noise of the AD8232 breakout at gain 100, 0.5–40 Hz, on one board at 23 °C, and compare it with the datasheet typical value" is a claim with a boundary.

The claim must name the unit, the instrument, and the conditions. If the claim cannot name an instrument, the experiment is not yet designed. If the claim cannot name a condition, the result will not be reproducible.

**Stop point:** if the claim requires a patient, a mains connection, or a clinical population, stop. Those claims need supervision, approvals, and a different protocol.

### Step 2. Build the hazard inventory

**Input:** the circuit, the mechanical setup, the materials, and the environment.

**Output:** a table with one row per hazard, its energy source, its path, its exposure, and its controls.

**Rationale:** a generic checklist misses the specific hazard. The inventory must be generated from the actual setup. Six categories are mandatory.

| Category | Questions | Typical source in this module | Control |
|---|---|---|---|
| Electrical | What voltage, current, and frequency? Is any conductor mains-referenced? | Bench supply, USB, signal generator | SELV, fuse, current limit, insulated probe |
| Stored energy | What capacitance, inductance, battery, or spring stores energy? | Bulk capacitors, servo, lithium cell | Bleeder, discharge, guarded enclosure |
| Thermal | What surface exceeds 40 °C? What can ignite? | Soldering iron, hot water, regulator, cell | Hold time, tongs, thermal cut-out |
| Mechanical | What moves, spins, or stores torque? | Servo arm, fan, syringe under pressure | Guard, torque limit, pressure relief |
| Chemical | What is corrosive, flammable, or sensitising? | Solder flux, isopropyl alcohol, adhesives | Ventilation, gloves, eye protection |
| Optical | What emits ultraviolet, visible, or infrared energy? | Curing lamp, IR emitter, laser | Enclosure, eyewear, beam stop |
| Ergonomic and procedural | What requires sustained force or awkward posture? | Long soldering, cable pulls | Breaks, bench height, task rotation |

The inventory is reviewed by a second person before power is applied. A one-person inventory is a single point of failure.

### Step 3. Classify the electrical zone

**Input:** the schematic, the supply, and the connection to the outside world.

**Output:** Zone 1, Zone 2, or Zone 3, with the reason.

**Rationale:** the zone determines the supervision, the instrumentation, and the evidence. The classification is not a voltage measurement. It is a topology question.

Ask four questions in order:

1. Is any conductor connected to a mains source, directly or through a component that can fail short? If yes, Zone 2 or Zone 3.
2. Does any conductor contact a person, or an applied part, in normal use? If yes, Zone 3.
3. Can a single fault put mains voltage on an accessible part? If yes, Zone 2 or Zone 3.
4. Is the supply a SELV source, such as a battery, a certified SELV adapter, or a current-limited bench supply with a floating output? If yes, Zone 1.

A "floating" bench supply is not automatically SELV. A floating output can still be referenced to earth through the oscilloscope ground, the USB cable, or the building's equipotential bonding. The classification must include the test equipment, not only the DUT.

**Stop point:** if the answer to question 1 or 2 is yes, the experiment is outside this course's authorisation. Hand it to a qualified supervisor.

### Step 4. Verify instrument ratings and calibration

**Input:** the instrument list, including probes, leads, and shunts.

**Output:** a table with model, serial number, rating, calibration status, and the parameter it measures.

**Rationale:** an instrument used outside its rating is a hazard, and an uncalibrated instrument is a source of false confidence. Both must be visible before the measurement.

| Instrument | Rating to check | Calibration evidence | Common trap |
|---|---|---|---|
| Bench supply | Output voltage, current limit, isolation | Internal check or lab calibration | Current limit set above the DUT's rating |
| DMM | CAT rating and voltage rating | Certificate or lab check | CAT II probe used on a CAT III circuit |
| Oscilloscope | Input voltage, probe attenuation, bandwidth | Calibration sticker with date | Ground clip connected to a live node |
| Probe | Voltage rating, derating with frequency | Probe datasheet | 10× probe used at a voltage above its derating curve |
| Logic analyser | Input voltage, threshold | Self-test | Threshold set for 5 V logic on a 1.8 V board |
| Power monitor | Shunt range, burden voltage | Datasheet and shunt tolerance | Burden voltage changes the DUT behaviour |
| Signal generator | Output amplitude, offset, source impedance | Calibration or loopback check | Offset not zero, output clipped |
| Reference thermometer | Range, accuracy, immersion depth | Certificate | Ice bath not stirred, no equilibration |
| Load cell | Range, non-linearity, creep | Certificate or known mass check | Off-axis loading, temperature drift |
| ESD strap | Resistance, continuity | Daily check | Worn over a sleeve, no skin contact |

The calibration question is not "is there a sticker?" It is "does the certificate cover the range and the parameter I am using, and is it still within its interval?" A certificate for a 10 V range does not calibrate a 1000 V range.

**Rule:** record the instrument ID in the report. A result without an instrument ID cannot be reproduced.

### Step 5. Establish the physical setup

**Input:** the bench layout, the guard, the fire boundary, and the access route.

**Output:** a bench that can be left in a safe state and an annotated photograph of the setup.

**Rationale:** most lab injuries are layout problems. A probe tip rolls, a cable pulls a board, a hot iron is placed where a hand reaches, a battery is charged next to solvent.

The layout rules are:

- The DUT is at least 30 cm from the bench edge.
- The supply is switched off at the wall when not in use, and the plug is accessible.
- The soldering iron sits in a stand, never on the bench surface.
- Flammable material is at least 1 m from any heat source.
- The emergency stop, if present, is reachable without reaching across the motion.
- The fire extinguisher and the first-aid kit are visible and unobstructed.
- Cables are routed so that a person cannot trip on them or pull the DUT.
- The work area is lit to at least the level needed to read component markings.

### Step 6. Apply ESD control and PPE

**Input:** the handling task and the board's sensitivity.

**Output:** a bench that is at the same potential as the operator, and PPE appropriate to the task.

**Rationale:** ESD is both a device-damage mechanism and, at high energy, an ignition source. PPE is the last control, but it is still a control.

ESD rules:

- Wear a wrist strap with a 1 MΩ series resistor, in contact with bare skin.
- Connect the strap to a common point on a grounded ESD mat.
- Check the strap daily with an ESD tester or a continuity check.
- Handle boards by the edges; never touch pins or connectors.
- Store boards in conductive bags or on the mat, not on a plastic tray.
- Do not wear the strap when working on a live circuit above the SELV boundary; the strap is a conductive path.

PPE rules:

| Task | Minimum PPE | Additional control |
|---|---|---|
| Soldering | Safety glasses, lab coat | Fume extraction, iron stand |
| Cutting or stripping wire | Safety glasses | Sharp tool, cut away from body |
| Hot water above 40 °C | Safety glasses, tongs | Temperature limit below 60 °C |
| Pressure above 50 kPa | Safety glasses, face shield | Pressure relief, rated tubing |
| Servo arm motion | Safety glasses | Guard, torque limit, e-stop |
| Lithium cell charging | Safety glasses | Fire-resistant bag, attended charging |
| Handling solvents | Safety glasses, nitrile gloves | Ventilation, no ignition source |

### Step 7. Inspect the electrical assembly

**Input:** the assembled circuit, the leads, and the connectors.

**Output:** a pass or a defect list, recorded before power.

**Rationale:** most bench faults are visible before power. Finding them with a meter is cheaper and safer than finding them with a current.

Inspect in this order:

1. Polarity of every supply connection, including the barrel jack and the header.
2. Continuity from the supply positive to the intended rail, and no continuity to ground.
3. Resistance from the supply rail to ground. A value below the expected load resistance indicates a short.
4. Solder bridges on fine-pitch parts, using magnification.
5. Component orientation: diodes, electrolytic capacitors, integrated circuits, connectors.
6. Lead dress: no strand of wire outside the terminal, no exposed conductor where a hand can reach.
7. Insulation: no nick in a probe lead, no cracked strain relief, no tape in place of a proper joint.
8. Fuse: correct value and type, in the correct holder, not bypassed.
9. Earth bond, if the enclosure is earthed: measure the resistance from the exposed metal to the earth pin.

A short check on a fresh board is not optional. A 3.3 V supply can deliver enough current into a short to destroy a board and start a fire on a contaminated surface.

### Step 8. Set the current limit and the first-power sequence

**Input:** the expected current, the DUT rating, and the supply.

**Output:** a current limit that protects the DUT and the bench, and a first-power log.

**Rationale:** a current limit is a substitute for a fuse only during bring-up, and only if it is set correctly. It does not replace a fuse for a fault that persists.

The first-power sequence:

1. Set the supply to the minimum voltage and the current limit to the expected current plus 20%, or to the DUT's absolute maximum, whichever is lower.
2. Connect the DUT with the supply off.
3. Switch on, and observe the current with the meter, not with the DUT's display.
4. If the current is at the limit, switch off immediately. Do not "see if it settles".
5. Raise the voltage in steps of no more than 10% of the target, checking current and temperature at each step.
6. Record the quiescent current, the supply voltage at the DUT, and the temperature after one minute.
7. Only after the quiescent state is verified, apply the stimulus.

Acceptance criteria for first power: quiescent current within the expected range, no component above 40 °C unless designed to run hot, no smell, no sound, no smoke, and a stable supply rail.

### Step 9. Establish the measurement reference

**Input:** the measurement point, the instrument ground, and the circuit ground.

**Output:** a documented ground reference and probe connection.

**Rationale:** an oscilloscope ground clip is connected to protective earth. If it is clipped to a node that is not at earth potential, the probe creates a short through the earth conductor. This is one of the most common ways to destroy a board and to create a shock hazard.

Rules:

- Connect the probe ground to circuit ground only, and only after checking that circuit ground is not above the SELV boundary.
- For a floating measurement, use a differential probe or two probes with a subtract operation, not a ground clip on a live node.
- Keep the ground lead short. A long ground lead adds inductance and picks up interference.
- Check the probe attenuation and the scope's input impedance before interpreting amplitude.
- Do not exceed the probe's voltage derating at the frequency of the signal.
- Use a current probe or a shunt for current, not a probe across a component.

### Step 10. Rehearse the fault response

**Input:** the list of foreseeable faults for the experiment.

**Output:** a demonstrated safe state for each fault, with the response time recorded.

**Rationale:** a safe state that has never been entered is a hypothesis. The rehearsal is the evidence.

Rehearse at least:

- Loss of supply.
- Loss of the control signal.
- Stuck output.
- Sensor disconnect.
- Over-temperature.
- Mechanical obstruction.
- Software hang, if software controls an output.

For each fault, record the detection mechanism, the time to detect, the time to reach the safe state, and the residual energy. For a servo, the residual motion after the e-stop is not zero; it is the distance travelled during deceleration.

### Step 11. Verify the guard and the emergency stop

**Input:** the mechanical setup and the actuator.

**Output:** a guard that prevents access, an e-stop that removes actuator power, and a test record.

**Rationale:** a software stop is not an emergency stop. An emergency stop must be hardware, must be reachable, and must remove energy rather than request that software remove it.

Checks:

1. The guard cannot be opened without a tool, or opening it triggers an interlock.
2. The e-stop is within reach of the operator's normal position, without reaching across the motion.
3. Pressing the e-stop removes actuator power, verified by measuring the actuator supply, not by observing the arm.
4. The e-stop latches; releasing it does not restart motion.
5. Restart requires a deliberate action and a check that the workspace is clear.
6. The guard and the e-stop are labelled.

### Step 12. Control thermal and battery hazards

**Input:** the heat sources, the cells, and the charging arrangement.

**Output:** a thermal budget and a charging procedure.

**Rationale:** thermal faults are slow and easy to miss. A regulator that runs at 90 °C is a reliability problem; a lithium cell that vents is a fire.

Thermal rules:

- Keep hot water below 60 °C for temperature calibration, and handle it with tongs.
- Keep the soldering iron in a stand, with the cord away from the work.
- Measure the case temperature of every regulator after five minutes at full load.
- Provide ventilation for any enclosure that dissipates more than 2 W.
- Do not cover a powered board with paper, cloth, or a notebook.

Battery rules:

- Use a protected cell with an integrated protection circuit.
- Use a charger designed for the cell chemistry.
- Charge on a non-flammable surface, in a fire-resistant bag, while attended.
- Do not charge a cell that is swollen, dented, or has a damaged wrapper.
- Do not puncture, crush, or short a cell.
- Keep a Class D extinguisher or a fire blanket appropriate to the chemistry.

### Step 13. Shut down and discharge

**Input:** the powered setup.

**Output:** a bench with no stored energy and a labelled DUT.

**Rationale:** a significant fraction of bench shocks happen after the experiment, when the supply is off but a capacitor is still charged.

Shutdown sequence:

1. Remove the stimulus.
2. Stop the firmware, if it controls an output.
3. Switch off the supply, then unplug it.
4. Wait for the bleeder resistor to discharge the bulk capacitance. If the time constant is unknown, wait five time constants.
5. Measure the voltage across every bulk capacitor. Do not assume the bleeder works.
6. Discharge with a rated resistor, not with a screwdriver or a wire.
7. Remove the DUT and place it on the ESD mat or in a conductive bag.
8. Label the DUT with the date, the configuration, and the last test.
9. Record the teardown in the log.

### Step 14. Write the record and report the near miss

**Input:** the log, the calibration records, the raw data, and the observations.

**Output:** a lab record that another person can follow, and a near-miss report if anything unexpected happened.

**Rationale:** the record is the only durable output of the lab. A near miss that is not reported is a control that will fail again.

The record contains:

- The claim and the claim boundary.
- The hazard inventory and the zone classification.
- The instrument IDs, ranges, resolutions, and calibration status.
- The setup photograph and the wiring diagram.
- The firmware commit, the model hash, and the toolchain versions.
- The raw data location and a SHA-256 hash.
- The environmental conditions that matter.
- The method, the uncertainty, and the acceptance criterion.
- The verdict and what it does not prove.
- The near misses and the actions taken.

### 5.15 Acceptance criteria for the bench

The bench is released for the experiment only when all of the following are true.

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Claim and boundary written | One-paragraph charter | Rewrite |
| 2 | Hazard inventory reviewed by a second person | Signed table | Review |
| 3 | Zone classification documented | Schematic annotation | Reclassify |
| 4 | Every instrument within rating and calibration | Instrument table | Replace or recalibrate |
| 5 | ESD strap checked | Tester reading | Replace strap |
| 6 | No exposed mains conductor | Visual inspection | Stop |
| 7 | Continuity and short check passed | Meter log | Repair |
| 8 | Current limit set below the DUT rating | Photo of supply display | Reset |
| 9 | First-power current within range | First-power log | Stop |
| 10 | Probe ground connected to circuit ground only | Photo | Reconnect |
| 11 | Fault response rehearsed | Fault table with times | Rehearse |
| 12 | Guard and e-stop verified | Test record | Do not run |
| 13 | Thermal check after five minutes | Temperature log | Improve cooling |
| 14 | Shutdown and discharge procedure demonstrated | Checklist | Redo |
| 15 | Record complete | Lab record | Complete |

### 5.16 The stop-work rule

Any learner can stop any lab, at any time, without justifying the decision. The stop is not a complaint; it is a control. The person who stops work must not be penalised, and the reason must be recorded.

Stop-work triggers include:

- Smoke, smell, discolouration, or an unusual sound.
- A temperature that is rising without a bound.
- A conductor that is accessible and might be live.
- A guard that is missing, loose, or defeated.
- An e-stop that does not latch or does not remove power.
- A person who is unsure whether the setup is safe.
- A measurement that requires touching a live conductor.

### 5.17 What the lab cannot establish

The procedure above produces a safe experiment and a defensible record. It does not produce:

- A leakage-current compliance result.
- A dielectric-strength type test.
- An EMC compliance result.
- A safety-rated function.
- A clinical validation.
- A QMS audit.

Those require equipment, methods, and organisations that a teaching lab does not have. The value of the lab is that it teaches the mechanism, the measurement discipline, and the boundary. The boundary is the lesson.

## 6. Worked example

### 6.1 The question

A team is preparing an ECG front-end lab. The prototype is powered from a 5 V USB adapter. The adapter is a Class II product with no protective earth. The team wants to know whether the arrangement could be connected to a patient, and what the measured leakage path implies.

The team does not connect the circuit to a person. It models the patient path with a 10 kΩ resistor from the circuit ground to earth, measures the voltage across the resistor with a digital multimeter (DMM), and computes the current.

The instrument record is part of the result. Without it, the number cannot be reproduced or audited.

| Item | Value | Why it matters | Recorded |
|---|---|---|---|
| DMM | Handheld, 3.5-digit, 20 V DC range | Resolution and accuracy set the uncertainty | Model and serial |
| DMM resolution | 10 mV | 5.77 mV standard uncertainty | Datasheet |
| DMM accuracy | ±(0.5% + 3 counts) | 32.1 mV bound at 0.42 V | Datasheet |
| DMM input impedance | About 10 MΩ | Loads the model resistor by 0.12% | Datasheet |
| Model resistor | 9.98 kΩ, 1%, metal film | Sets the current scale | Measured value |
| Source | 5 V USB adapter, Class II, no earth | Determines the leakage path | Model and certificate |
| Environment | 23 °C, 45% relative humidity | Affects surface leakage | Logged |
| Readings | 5, one minute apart | Shows short-term stability | Raw log |

### 6.2 The measurement

The DMM is a handheld unit on the 20 V DC range, with a resolution of 10 mV and an accuracy specification of ±(0.5% of reading + 3 counts). The resistor is a 10 kΩ metal-film resistor with a 1% tolerance and a measured value of 9.98 kΩ on the same DMM.

Five readings are taken, one minute apart, with the adapter connected to the same socket and the DUT idle.

| Reading | Voltage across 9.98 kΩ | Current | Note |
|---:|---:|---:|---|
| 1 | 0.42 V | 42.1 µA | Idle, display on |
| 2 | 0.43 V | 43.1 µA | Idle |
| 3 | 0.41 V | 41.1 µA | Idle |
| 4 | 0.42 V | 42.1 µA | Idle |
| 5 | 0.42 V | 42.1 µA | Idle |

The mean voltage is 0.42 V. The mean current is:

\[
I = \frac{V}{R} = \frac{0.42\ \text{V}}{9.98 \times 10^3\ \Omega} = 4.21 \times 10^{-5}\ \text{A} = 42.1\ \mu\text{A}
\]

### 6.3 The uncertainty

The voltage uncertainty has a resolution component and an accuracy component.

Resolution: 10 mV, rectangular distribution, standard uncertainty \(10/\sqrt{3} = 5.77\) mV.

Accuracy at 0.42 V: 0.5% of reading plus 3 counts = 2.1 mV + 30 mV = 32.1 mV. Treated as a rectangular bound, the standard uncertainty is \(32.1/\sqrt{3} = 18.5\) mV.

The combined voltage uncertainty is:

\[
u_V = \sqrt{5.77^2 + 18.5^2} = 19.4\ \text{mV}
\]

The resistance uncertainty is 1% of 9.98 kΩ = 99.8 Ω, rectangular, standard uncertainty 57.6 Ω.

The current uncertainty, for a quotient, is:

\[
\frac{u_I}{I} = \sqrt{\left(\frac{u_V}{V}\right)^2 + \left(\frac{u_R}{R}\right)^2}
= \sqrt{\left(\frac{0.0194}{0.42}\right)^2 + \left(\frac{0.0576}{9.98}\right)^2}
= \sqrt{0.00213 + 0.0000333} = 0.0465
\]

So \(u_I = 0.0465 \times 42.1 = 1.96\ \mu\text{A}\). The expanded uncertainty at \(k=2\) is 3.9 µA. The result is 42.1 µA ± 3.9 µA (k = 2).

The dominant term is the DMM accuracy, not the resistor. A better meter would reduce the uncertainty; a better resistor would not.

### 6.3a The uncertainty budget as a table

The budget is easier to audit when every component is listed with its distribution and its divisor.

| Source | Value | Distribution | Divisor | Standard uncertainty | Share of variance |
|---|---|---:|---|---:|---:|---:|
| DMM resolution | 10 mV | Rectangular | √3 | 5.77 mV | 8.9% |
| DMM accuracy at 0.42 V | 32.1 mV | Rectangular | √3 | 18.5 mV | 91.0% |
| Resistor tolerance | 99.8 Ω | Rectangular | √3 | 57.6 Ω | 0.1% |
| Repeatability of the mean | 0.71 µA | Normal | √5 | 0.32 µA | <0.1% |
| **Combined** | | | | **1.99 µA** | **100%** |

The accuracy term contributes about 91% of the variance. The repeatability term, which is the only Type A component, contributes less than 0.1%. This is a common pattern in a screening measurement: the instrument specification dominates, and repeating the reading does not improve the result. The correct response is a better instrument or a narrower range, not more readings.

The resistor tolerance is negligible because the 1% tolerance is small compared with the 4.6% relative voltage uncertainty. If the resistor were 0.1%, the budget would barely change. If the meter were 0.05%, the resistor term would become visible, and the choice of resistor would start to matter.

### 6.3b What the budget does not include

The budget covers the electrical measurement. It does not include:

- The temperature coefficient of the resistor, which is small for a metal-film part but not zero.
- The self-heating of the resistor, which is negligible at 42 µA.
- The adapter's drift over time, which is not characterised by five readings in one minute.
- The frequency response of the model, which is the largest omission and the reason the result is a screening estimate.
- The common-mode voltage of the adapter output relative to earth, which the single-ended measurement does not capture.

A complete budget for a compliance measurement would include these, and the standard's measuring network would replace the resistor. The lab budget is honest about what it covers and explicit about what it does not.

### 6.4 The comparison with the limits

IEC 60601-1 Table 3, as summarised in published guidance, gives patient leakage current limits for a type CF applied part of 10 µA in normal condition and 50 µA under single-fault condition. For type B and BF applied parts, the corresponding values are 100 µA and 500 µA. The measured 42.1 µA is above the type CF normal-condition limit and below the type BF normal-condition limit.

| Applied part | Condition | Limit | Measured | Margin | Verdict against this model |
|---|---|---:|---:|---:|---|
| Type CF | Normal | 10 µA | 42.1 µA | −32.1 µA | Fail |
| Type CF | Single fault | 50 µA | 42.1 µA | +7.9 µA | Pass, but only just |
| Type BF | Normal | 100 µA | 42.1 µA | +57.9 µA | Pass |
| Type BF | Single fault | 500 µA | 42.1 µA | +457.9 µA | Pass |

This comparison is not a compliance result. It uses a 10 kΩ resistor in place of the measuring device defined in the standard, it does not use the standard's test conditions, and it does not test the adapter's single-fault behaviour. It is a screening calculation that tells the team the arrangement is not obviously suitable for a type CF applied part.

### 6.4a Why a DC reading can understate an AC leakage path

The measurement was made on the DC range. If the adapter's leakage is dominated by a capacitor to the mains, the current is alternating at the line frequency, and a DC reading of its average may be near zero. The 0.42 V reading is therefore evidence of a DC path, not proof that the AC path is small.

The correct screening measurement for an AC path is a true-RMS meter with a bandwidth that covers the line frequency, or better, the standard's measuring network with a defined frequency response. A DC meter can miss a 50 Hz current entirely. This is one reason a bench screening result cannot be promoted to a compliance result: the instrument's frequency response is part of the method, and the method is part of the claim.

The same caution applies to the model resistor. At 50 Hz, a 10 kΩ resistor and a 10 kΩ resistor in parallel with 0.015 µF have different impedances. The capacitor's impedance at 50 Hz is:

\[
Z_C = \frac{1}{2\pi \times 50 \times 0.015 \times 10^{-6}} = 212\ \text{k}\Omega
\]

In parallel with 10 kΩ, the magnitude is about 9.98 kΩ, so the difference at 50 Hz is small. At 10 kHz, the capacitor's impedance is 1.06 kΩ, and the parallel combination is about 0.96 kΩ, a 4% difference. At 100 kHz, the capacitor dominates and the network impedance is about 106 Ω. A switching adapter's leakage can have content at tens to hundreds of kilohertz, where the resistor-only model is wrong by orders of magnitude. The model is adequate at the line frequency and inadequate above it. The report must say so.

### 6.5 Two checks that would invalidate the result

**Check 1: the model impedance is not the standard's measuring network.** IEC 60601-1 specifies a measuring device with a defined frequency response, commonly described as a 10 kΩ resistor in parallel with a 0.015 µF capacitor, plus a 1 kΩ resistor in some configurations, to model the patient's impedance over frequency. A pure 10 kΩ resistor has a different impedance at 50/60 Hz and at the switching frequencies of the adapter. If the leakage has a significant AC component, the measured voltage across a resistor will not represent the current the standard's network would measure. The result would be invalid as a leakage estimate, though still useful as a gross screening number.

**Check 2: the DMM's own input impedance loads the circuit.** A handheld DMM on the 20 V range has an input impedance of about 10 MΩ. In parallel with the 9.98 kΩ resistor, the effective resistance is:

\[
R_{\text{eff}} = \frac{9.98 \times 10^3 \times 10 \times 10^6}{9.98 \times 10^3 + 10 \times 10^6} = 9.968\ \text{k}\Omega
\]

The loading error is 0.12%, which is negligible here. But if the model resistor were 1 MΩ, the loading error would be about 9%, and the result would be invalid without a correction. The check is to compare the model impedance with the meter's input impedance before trusting the reading.

### 6.6 A varied condition: the adapter is replaced by a battery

The team repeats the measurement with a 9 V battery and a linear regulator. The battery is a floating SELV source, and there is no mains connection.

| Condition | Source | Voltage across 9.98 kΩ | Current | Type CF normal limit | Verdict |
|---|---|---:|---:|---:|---|
| Original | 5 V USB adapter, Class II | 0.42 V | 42.1 µA | 10 µA | Fail |
| Varied | 9 V battery, linear regulator | 0.004 V | 0.4 µA | 10 µA | Pass with margin |

The change is a factor of about 100. The lesson is that the leakage path is a property of the power architecture, not of the signal chain. A battery-powered prototype can look excellent on a leakage test and a mains-powered version of the same circuit can fail, because the mains adapter's isolation capacitance provides the path.

The team should not conclude that the battery version is suitable for a patient. It has removed one path, but it has not demonstrated the isolation, the defibrillation protection, the single-fault behaviour, or the applied-part classification. It has only shown that the dominant path in the previous setup is gone.

### 6.6a What the next experiment would be

The screening result points to a specific next experiment: an isolated, medically rated supply with a documented isolation barrier, measured with the standard's network by a qualified person under supervision. The table below separates what the team can do on its own from what it must hand off.

| Activity | Who can do it | Evidence produced | Still not proved |
|---|---|---|---|
| Repeat the screening on a battery supply | Learner | Bench note with uncertainty | Isolation for mains use |
| Characterise the adapter's AC leakage | Learner with a true-RMS meter and supervision | AC current at line frequency | Compliance with Table 3 |
| Measure with the standard's network | Qualified person, supervised | Screening closer to the method | Type test |
| Dielectric-strength test | Accredited laboratory | Type-test report | Production conformity |
| Single-fault leakage test | Accredited laboratory | Type-test report | Clinical safety |

The hand-off is the point. A lab that stops at the right boundary is more useful than a lab that overruns it, because the boundary is what the next stage needs to know.

### 6.7 What the example demonstrates

The arithmetic closes: 0.42 V across 9.98 kΩ is 42.1 µA, with an expanded uncertainty of 3.9 µA dominated by the DMM's accuracy specification. The result is four times the type CF normal-condition limit and less than half the type BF limit. The battery variant reduces the current by a factor of about 100.

The example also demonstrates the boundary. A measured voltage, a model resistor, and a limit table do not add up to a compliance result. They add up to a screening estimate with a stated model, a stated uncertainty, and a stated limitation. That is what a lab can produce, and it is enough to make a design decision: do not connect this adapter to a patient; use an isolated, medically rated supply and re-test under supervision.

The same structure applies to every worked example in this module. Measure what you can, model what you cannot, state the uncertainty, and name the boundary. A reader who can reproduce the arithmetic and see the boundary can decide how much weight to put on the result. A reader who sees only a number cannot.

## 7. Code, query, or calculation

The program below processes a leakage-screening log. It uses only the Python standard library. It computes the mean, the standard deviation, the standard error, the Type A and Type B uncertainty components, the expanded uncertainty, and the margin against a limit. It also computes the prospective current if the same model impedance were exposed to a 230 V mains source, as a calculation only, not as a test.

The input is embedded as a CSV string so that the program runs without a file. In a real lab the same program would read the CSV exported by the instrument.

```python
import csv
import io
import math

LOG = """reading,voltage_v,resistance_ohm,note
1,0.42,9980,idle
2,0.43,9980,idle
3,0.41,9980,idle
4,0.42,9980,idle
5,0.42,9980,idle
"""

def read_log(text):
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        rows.append({
            "reading": int(row["reading"]),
            "v": float(row["voltage_v"]),
            "r": float(row["resistance_ohm"]),
            "note": row["note"],
        })
    return rows

def mean(xs):
    return sum(xs) / len(xs)

def stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

rows = read_log(LOG)
voltages = [r["v"] for r in rows]
resistances = [r["r"] for r in rows]
r_mean = mean(resistances)

currents_ua = [v / r * 1e6 for v, r in zip(voltages, resistances)]
i_mean = mean(currents_ua)
i_sd = stdev(currents_ua)
i_sem = i_sd / math.sqrt(len(currents_ua))

# Type B components for one reading, then for the mean.
resolution_v = 0.010
accuracy_v = 0.005 * i_mean * r_mean / 1e6 + 3 * resolution_v
u_res = resolution_v / math.sqrt(3)
u_acc = accuracy_v / math.sqrt(3)
u_v = math.hypot(u_res, u_acc)
u_r = 0.01 * r_mean / math.sqrt(3)

v_mean = mean(voltages)
rel = math.hypot(u_v / v_mean, u_r / r_mean)
u_b_current = rel * i_mean
u_c = math.hypot(i_sem, u_b_current)

print(f"n = {len(rows)}")
print(f"mean voltage       = {v_mean:.4f} V")
print(f"mean resistance    = {r_mean:.1f} ohm")
print(f"mean current       = {i_mean:.2f} uA")
print(f"repeatability sd   = {i_sd:.3f} uA")
print(f"standard error     = {i_sem:.3f} uA")
print(f"Type B (V and R)   = {u_b_current:.2f} uA")
print(f"combined u_c       = {u_c:.2f} uA")
print(f"expanded U (k=2)   = {2*u_c:.2f} uA")

for name, limit in (("CF normal", 10.0), ("CF single fault", 50.0),
                     ("BF normal", 100.0), ("BF single fault", 500.0)):
    margin = limit - i_mean
    verdict = "PASS" if i_mean <= limit else "FAIL"
    print(f"{name:18s} limit {limit:6.1f} uA  margin {margin:+8.1f} uA  {verdict}")

# Calculation only: prospective current if the model path saw 230 V.
prospective_ma = 230.0 / r_mean * 1e3
print(f"prospective current at 230 V = {prospective_ma:.1f} mA (calculation, not a test)")
```

Expected output:

```text
n = 5
mean voltage       = 0.4200 V
mean resistance    = 9980.0 ohm
mean current       = 42.08 uA
repeatability sd   = 0.709 uA
standard error     = 0.317 uA
Type B (V and R)   = 1.96 uA
combined u_c       = 1.99 uA
expanded U (k=2)   = 3.97 uA
CF normal          limit   10.0 uA  margin    -32.1 uA  FAIL
CF single fault    limit   50.0 uA  margin     +7.9 uA  PASS
BF normal          limit  100.0 uA  margin    +57.9 uA  PASS
BF single fault    limit  500.0 uA  margin   +457.9 uA  PASS
prospective current at 230 V = 23.0 mA (calculation, not a test)
```

The program separates repeatability from the instrument specification. The repeatability standard deviation is 0.75 µA, which is small compared with the Type B component of 1.96 µA. The dominant term is the DMM accuracy, which is why the report should name the meter and its range. A different meter with a 0.05% accuracy would change the conclusion's uncertainty, not the conclusion.

The 230 V calculation is included to make the hazard concrete: 23 mA through a 10 kΩ path is above the let-go threshold and in the range where sustained contact can be dangerous. It is not a test, and the lab never applies 230 V to the model. The number belongs in the hazard inventory, not in the measurement table.

## 8. Data, standards, and regulatory context

### 8.1 The IEC 60601-1 family

IEC 60601-1 is the general standard for basic safety and essential performance of medical electrical equipment. The current consolidated edition is Edition 3.2 (2005 + Amendment 1:2012 + Amendment 2:2020). The standard is published by the International Electrotechnical Commission (IEC) and adopted nationally, for example as ANSI/AAMI ES60601-1 in the United States.

The clauses that matter for a hardware lab are:

- Clause 8.7, leakage currents and patient auxiliary currents, with Table 3 for the limits and Figures 13–20 for the measurement circuits.
- Clause 8.5, protection against electrical hazards, including insulation and separation.
- Clause 8.6, protective earthing.
- Clause 11, excessive temperatures and other hazards.
- Clause 13, hazardous situations and fault conditions.
- Clause 14, programmable electrical medical systems.
- Clause 17, electromagnetic compatibility.

The published limit values commonly quoted for patient leakage current are:

| Applied part | Normal condition (AC) | Single-fault condition (AC) | Normal condition (DC) | Single-fault condition (DC) |
|---|---:|---:|---:|---:|
| Type B | 100 µA | 500 µA | 10 µA | 50 µA |
| Type BF | 100 µA | 500 µA | 10 µA | 50 µA |
| Type CF | 10 µA | 50 µA | 10 µA | 50 µA |

Earth leakage current is commonly quoted as 5 mA in normal condition and 10 mA under single fault. Touch current is 100 µA normal and 500 µA single fault. These figures are widely reproduced in test-house guides, but the exact value depends on the applied-part type, the condition, the edition, and the clause. A report must cite the edition and the table, not a secondary summary.

> **Note:** This file uses the values above as teaching numbers. They were checked against published guidance summaries, not against the purchased text of the standard. Before using any value in a regulatory document, read the current edition and confirm the clause, the table, and the condition.

### 8.2 The collateral and particular standards

IEC 60601-1-2 is the collateral standard for electromagnetic compatibility. Edition 4.1 (2014 + Amendment 1:2020) specifies immunity test levels by intended environment. For the professional healthcare facility environment, the radiated RF immunity level is 3 V/m from 80 MHz to 2.7 GHz with 80% amplitude modulation at 1 kHz. For the home healthcare environment, the level is 10 V/m. Electrostatic discharge is ±8 kV contact and ±15 kV air in both environments. Rated power-frequency magnetic field immunity is 30 A/m. Amendment 1 added proximity magnetic field tests at 134.2 kHz (65 A/m) and 13.56 MHz (7.5 A/m), with applicability conditions.

IEC 60601-1-11 covers medical electrical equipment used in the home healthcare environment. IEC 60601-1-12 covers emergency medical services environments.

The particular standards narrow the general standard for a device type. IEC 60601-2-25 covers electrocardiographs, IEC 60601-2-27 covers electrocardiographic monitoring equipment, and IEC 60601-2-49 covers multifunction patient monitoring equipment. For robotics, IEC 80601-2-77 covers robotically assisted surgical equipment and IEC 80601-2-78 covers medical robots for rehabilitation, assessment, compensation, or alleviation. Both were published in 2019 and have since been amended; both are under revision as of 2024.

### 8.3 IEC 61010-1 and the measurement-category system

IEC 61010-1 is the safety standard for electrical equipment for measurement, control, and laboratory use. It is the standard that governs the bench instruments themselves, not the medical device. Edition 3.0 was published in 2010, with Amendment 1 in 2016.

The measurement-category (CAT) system rates an instrument's ability to withstand transient overvoltages on a circuit. The categories are defined in IEC 61010-2-030, which moved the testing-and-measuring-circuit requirements out of Part 1.

| Category | Where the circuit sits | Example | Typical transient withstand |
|---|---|---|---|
| CAT II | Utilization point, downstream of a socket | Appliance, portable tool | 2.5 kV for 300 V, 4 kV for 600 V |
| CAT III | Distribution level, fixed installation | Distribution board, busbar | 4 kV for 300 V, 6 kV for 600 V |
| CAT IV | Origin of installation, service entrance | Meter, overhead line | 6 kV for 300 V, 8 kV for 600 V |

A CAT rating is meaningless without a voltage. "CAT III 600 V" is a rating. "CAT III" alone is not. The rating determines the creepage, the clearance, and the transient withstand, and it must be matched by the test leads.

This course operates in CAT II or below, and most experiments are not connected to a mains circuit at all. A CAT II 300 V meter is adequate for the bench. It is not adequate for a CAT III installation, and no learner should be working on one.

### 8.4 ISO 14971 and the risk file

ISO 14971:2019 is the risk-management standard for medical devices. It defines the hazard, the hazardous situation, the harm, and the risk, and it requires the manufacturer to identify reasonably foreseeable sequences of events that lead from a hazard to a hazardous situation.

The standard does not prescribe a risk matrix. It requires a process: risk analysis, risk evaluation, risk control, evaluation of residual risk, and monitoring in production and post-production. The risk file is the record of that process.

For a lab, ISO 14971 supplies the vocabulary. A lab hazard inventory is not a risk file, because it does not cover the device lifecycle, the clinical use, or the production variation. It is a subset that makes the lab safer and teaches the structure.

### 8.5 IEC 62304 and software

IEC 62304:2006 + Amendment 1:2015 defines the software life-cycle processes for medical device software. Amendment 1 replaced the severity-only classification with a risk-based scheme:

- Class A: software cannot contribute to a hazardous situation, or the risk is acceptable after external controls.
- Class B: software can contribute to a hazardous situation with non-serious injury.
- Class C: software can contribute to a hazardous situation with death or serious injury.

The classification drives the required processes, the documentation, and the verification. A lab that writes firmware for a prototype is not implementing IEC 62304, but it can use the classification to decide how much rigour the firmware needs.

### 8.6 Robot standards

ISO 10218-1:2025 is the third edition of the industrial robot safety standard. It was published in February 2025, supersedes ISO 10218-1:2011, and incorporates the collaborative-application requirements that were formerly in ISO/TS 15066:2016. It introduces a Class I / Class II robot classification with different performance-level requirements, expands the safety-function list, and adds cybersecurity requirements.

ISO/TS 15066:2016 remains the reference for collaborative robot force and pressure limits. Annex A gives body-region-specific quasi-static and transient limits, derived from a pain-onset study. The values are not universal: they depend on the body region, the contact area, and the contact dynamics. A robot lab must not quote a single force limit for all contacts.

IEC 80601-2-77:2019 and IEC 80601-2-78:2019 are the medical particular standards for surgical and rehabilitation robots. They sit under IEC 60601-1 and add requirements for the clinical function, the interaction with the patient, and the essential performance. They are not replacements for ISO 10218-1; a medical robot may need both the medical particular standard and the industrial safety standard, depending on its use and jurisdiction.

### 8.7 Occupational safety regulation

In the United States, occupational electrical safety is regulated by the Occupational Safety and Health Administration (OSHA) under 29 CFR 1910 Subpart S. The relevant provisions include 1910.303 (electrical installations), 1910.333 (work practices), 1910.334 (portable equipment), and 1910.147 (lockout/tagout). OSHA's Laboratory Safety Guidance (OSHA 3404) applies these requirements to laboratories and states that only qualified persons may work on electrical circuits.

NFPA 70E is the consensus standard for electrical safety in the workplace. Article 350 covers research and development laboratories. NFPA 70E defines the limited, restricted, and arc flash boundaries and the PPE categories. OSHA treats NFPA 70E as consensus guidance rather than an incorporated standard, but many institutions adopt it contractually.

In the European Union, the applicable framework includes the Low Voltage Directive for equipment in its scope and the Machinery Regulation for machinery, alongside the Medical Device Regulation for medical devices. The institution's own safety organisation is the authority for lab work.

### 8.8 What is binding and what is convention

| Document | Status | What it binds | What it does not bind |
|---|---|---|---|
| IEC 60601-1 | International standard; national adoption varies | Medical electrical equipment safety in the adopting jurisdiction | A student bench |
| IEC 60601-1-2 | Collateral standard | EMC of medical electrical equipment | A pre-compliance scan |
| IEC 61010-1 | International standard | Lab and measurement equipment | A custom DUT |
| ISO 14971 | International standard | Risk management process for medical devices | A lab hazard list |
| IEC 62304 | International standard | Medical device software lifecycle | A prototype firmware |
| ISO 10218-1 | International standard | Industrial robot safety | A toy servo arm |
| ISO/TS 15066 | Technical specification | Collaborative robot limits, referenced by ISO 10218-1:2025 | A medical robot |
| NFPA 70E | Consensus standard, adopted contractually | Electrical work practices where adopted | A jurisdiction that has not adopted it |
| OSHA 1910 Subpart S | Regulation | Employers in the US | A learner outside the US |
| Institutional safety policy | Contractual or employment condition | Everyone on site | Nothing else |

The last row is the one that governs the lab. A standard is not a substitute for the institution's safety office, and the institution's rules are not optional because a standard is silent.

### 8.9 The regulatory pathway in one paragraph

A medical device reaches the market through a conformity-assessment route. In the United States, the route is typically a 510(k) premarket notification with a predicate, a De Novo request, or a premarket approval application (PMA), supported by a QMS under 21 CFR 820. In the European Union, the route is a conformity assessment under Regulation (EU) 2017/745, with technical documentation under Annex II and a notified body for higher-risk classes. In both, the evidence includes design controls, risk management, verification and validation, software lifecycle, and clinical evidence where applicable. A lab contributes to the verification and validation evidence. It does not replace the route.

## 9. Industry practice and case evidence

### 9.1 What test houses actually do

An accredited test laboratory works from a test plan derived from the standard and the device's intended use. For IEC 60601-1, the laboratory:

- Reviews the construction and the risk file.
- Identifies the applied parts and classifies them.
- Measures leakage current with the standard's measuring device, at the specified voltages, polarities, and fault conditions.
- Performs dielectric-strength tests at the specified voltage and duration.
- Checks creepage, clearance, and insulation.
- Tests temperature rise under the specified load.
- Tests abnormal operation and single-fault conditions.
- Issues a report with the results, the conditions, and the deviations.

The laboratory's accreditation defines the scope. A laboratory accredited for IEC 60601-1 may not be accredited for IEC 80601-2-77. The accreditation certificate lists the standards and the clauses.

### 9.2 The type test versus the routine test

A type test is performed once on a representative sample to show that the design meets the standard. A routine test is performed on every production unit to show that the manufacturing process has not introduced a defect. Dielectric strength and earth-bond continuity are common routine tests. Leakage current is sometimes a routine test and sometimes a type test, depending on the device and the standard.

The distinction matters commercially. A type test costs a booking, a sample, and a report. A routine test costs a fixture and a few seconds per unit. A production line that performs a type test on every unit is wasting money; a production line that performs no routine test is shipping unverified units.

### 9.3 Hospital clinical engineering

Hospital clinical engineering departments perform acceptance testing, preventive maintenance, and safety testing on medical equipment. Their testers are calibrated, and their protocols are derived from the standards and the manufacturer's instructions. They test leakage current, earth continuity, and insulation, and they compare the results with the manufacturer's limits and the standard's limits.

A hospital engineer who receives a prototype with a bench report will ask three questions: what is the applied-part classification, what is the leakage current under normal and single-fault conditions, and what is the evidence that the measurement was made with a calibrated instrument and the correct network. A bench report that answers none of these is not useful.

### 9.4 Public recall narratives

The survey file lists several public recall narratives that map to the lab mechanisms. They are not lab results, and the labs do not reproduce the devices. They are included because they show what happens when an integrated test is missing.

| Event | Reported mechanism | Relevant lab habit | What the lab still does not prove |
|---|---|---|---|
| CMR Surgical Versius console recall, 2025 (identifier to be verified) | Secure Boot reportedly not enabled on a console | Firmware configuration verification, Topic 5 | That the same configuration or attack path applies |
| Virtual Incision MIRA recall, 2025 (identifier to be verified) | Software could interrupt or lag motion between controllers and arms | Latency and command-age testing, Topic 7 | Clinical consequence in a real procedure |
| Quantum Surgical EPIONE recall, 2025 (identifier to be verified) | Software bug could misalign the arm relative to the planned trajectory | Registration and frame testing, Topic 8 | Registration error on a patient |
| Accuray CyberKnife recall Z-1023-2025 | Snap-ring detachment could allow uncontrolled roll-axis rotation | Mechanical fault injection and guard design | Structural failure under clinical loads |
| Medtronic MiniMed 600 series recall, 2019 | Incorrect reservoir retention ring could cause over- or under-delivery | Calibration and mechanical verification, Topic 4 | Dosing error in a patient |

The pattern is consistent: a configuration or mechanical detail that was never exercised by an integrated test. The lab habit that catches these is fault injection tied to the risk file, not more nominal testing.

### 9.5 What institutions do

Research universities operate an electrical-safety programme that classifies work by hazard. Stanford University's Electrical Safety Management Plan for Research, for example, defines routine, hazardous, high-hazard, and restricted categories with voltage and energy thresholds, requires a qualified person designated by the principal investigator, and establishes an electrical safety committee. The plan incorporates NFPA 70E Article 350 and the relevant OSHA requirements.

The practical lesson for a learner is that the institution already has a process. The lab procedure in this topic is a teaching version of that process, not a replacement. Before any experiment outside the SELV boundary, the learner must find the institution's process and follow it.

### 9.6 What industry does with the compliance boundary

Device manufacturers maintain a distinction between:

- **Design verification**, which is internal and can use bench data.
- **Design validation**, which uses production-equivalent units and simulated or actual use.
- **Type testing**, which is external and accredited.
- **Production testing**, which is routine and documented.

A startup that presents bench data as a type test is not lying about the measurement; it is mislabelling the evidence. The mislabelling becomes a problem at a due-diligence review, a regulatory submission, or a recall investigation, when the reviewer asks for the report and finds a notebook.

### 9.7 Inference

It is an inference, not a documented finding, that teams which write the claim boundary before the experiment produce more defensible evidence. The reasoning is that a pre-written boundary forces the team to name the instrument and the acceptance criterion, and those names are what a reviewer checks. The public recall narratives are consistent with this inference, but they do not test it.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Bench treated as a type test | Report says "tested to IEC 60601" with no accredited lab | Confusion of evidence rungs | Rejected submission; redesign late | Label every result with its rung |
| 2 | Leakage measured with a DMM only | No measuring network, no condition table | Wrong impedance and frequency response | False pass or false fail | Use the standard network; state the model |
| 3 | Mains on the bench | Exposed line conductor, no qualified supervisor | Direct contact or arc | Fatal shock, fire | SELV boundary; stop-work |
| 4 | Scope ground on a live node | Probe ground clip on a non-earth node | Earth loop through the probe | Board destruction, shock | Differential probe; check reference |
| 5 | Current limit set too high | Supply current at limit on first power | Fault energy not bounded | Fire, destroyed DUT | Set limit before connection |
| 6 | Capacitor not discharged | Voltage present after shutdown | Stored energy | Burn, shock | Bleeder, measure, discharge |
| 7 | ESD strap not in contact with skin | Strap over a sleeve | No equalisation | Latent board damage | Daily check; bare skin |
| 8 | Guard removed "for a moment" | Interlock bypassed | Access to motion | Crush, cut | Interlock, stop-work |
| 9 | Software stop called an e-stop | No hardware removal of power | Software hang | Uncontrolled motion | Hardware e-stop, verified |
| 10 | Battery charged unattended | Cell warm, swollen | Thermal runaway | Fire | Attended charging, fire bag |
| 11 | Hot water above 60 °C | Steam, no thermometer | Scald | Burn | Temperature limit, tongs |
| 12 | Instrument out of calibration | Sticker expired or missing | Unknown error | Wrong decision | Calibration register |
| 13 | No near-miss reporting | "It was fine" culture | Control failure repeats | Injury later | Blameless reporting |
| 14 | Claim boundary omitted | Report has no limitations section | Overclaim | Loss of credibility | Template with mandatory section |
| 15 | PPE used as the only control | "We wear gloves" | Single barrier | Injury when PPE fails | Hierarchy of controls |
| 16 | Zone classification ignores test equipment | USB and scope not considered | Hidden mains path | Shock | Classify the whole setup |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The supply is SELV and remains SELV under single fault | The lab procedure only authorises Zone 1 | A faulty adapter, a shared earth, a scope ground | Inspect the supply certificate; measure output to earth |
| 2 | The 10 kΩ model represents the patient path | It gives a screening number | Frequency-dependent impedance, contact area, skin | Compare with the standard's network; state the limitation |
| 3 | The DMM's input impedance is much higher than the model resistance | The loading correction is negligible | A high-impedance model, a low-impedance meter | Calculate the parallel resistance |
| 4 | The instrument accuracy specification applies at the used range | The uncertainty budget depends on it | Wrong range, wrong temperature, drift | Read the certificate; check the range |
| 5 | The DUT is the only energy source | The hazard inventory is complete | A shared bench supply, a charged neighbour board | Inspect the bench; isolate |
| 6 | The guard prevents access | The mechanical control is credited | A gap, a defeated interlock, a tool | Try to reach the hazard with a probe |
| 7 | The e-stop removes actuator power | The safe state depends on it | A software-only stop, a back-drivable actuator | Measure the actuator supply during the stop |
| 8 | The operator is trained and rested | Administrative controls are credited | Fatigue, unfamiliar equipment, time pressure | Supervision; task rotation |
| 9 | The standard values quoted are current | The comparison is meaningful | A new edition, a national deviation | Check the edition and the national adoption |
| 10 | The lab result is not a compliance result | The claim boundary holds | A report that omits the boundary | Review the report against the template |
| 11 | The environment is a normal lab | Temperature, humidity, and contamination are bounded | A wet bench, a hot room, conductive debris | Measure and record the environment |
| 12 | The firmware is not safety-rated | No safety function is credited to software | A claim that software prevents a hazard | Review the safety-function claims |

## 12. Exercises

### Exercise 1. Hazard inventory (recall and application)

**Deliverable:** a one-page hazard inventory for the ECG front-end lab, covering all six categories in Section 5, Step 2.

**Acceptance criteria:** at least 12 rows; each row names the source, the path, the exposure, and the control; at least one control is an elimination or substitution, not a procedure; the inventory is reviewed by a second person and signed.

**Time:** 45 minutes.

### Exercise 2. Zone classification (application)

**Deliverable:** a labelled schematic of a USB-powered sensor board connected to an oscilloscope and a laptop, with each conductor classified as Zone 1, 2, or 3.

**Acceptance criteria:** the classification accounts for the laptop charger, the USB cable, the scope ground, and the adapter's isolation; the answer states which single fault would change the classification; the reasoning is written in four sentences.

**Time:** 30 minutes.

### Exercise 3. Evidence-rung sorting (critique)

**Deliverable:** a table that sorts ten statements from lab reports into calculation, simulation, bench measurement, design verification, and type test, with a one-line justification for each.

**Acceptance criteria:** all ten are classified; at least three are deliberately ambiguous and the answer explains why; no statement is classified as a type test without a named accredited laboratory.

**Time:** 30 minutes.

### Exercise 4. Leakage screening (calculation)

**Deliverable:** a calculation of the leakage current from a measured voltage across a model resistor, with a full uncertainty budget at k = 2, compared with the type CF and type BF limits.

**Acceptance criteria:** the budget separates Type A and Type B components; the dominant component is named; the report states that the result is a screening estimate, not a compliance result; the arithmetic is reproducible from the submitted data.

**Time:** 45 minutes.

### Exercise 5. Pre-power checklist (design)

**Deliverable:** a checklist for a 12 V, 2 A servo-arm bench, adapted from Section 5.15, with the acceptance criteria and the fail action for each item.

**Acceptance criteria:** at least 15 items; each item has an objective test, not an opinion; the checklist includes the guard, the e-stop, the current limit, the thermal check, and the discharge step; a partner can execute it without asking a question.

**Time:** 60 minutes.

### Exercise 6. Fault rehearsal (application)

**Deliverable:** a fault table for a firmware-controlled output, with detection mechanism, detection time, safe state, and residual energy for five faults.

**Acceptance criteria:** at least one fault is a software hang; at least one is a sensor disconnect; the detection times are measured, not estimated; the safe state is demonstrated with a capture; the residual motion of the actuator is measured.

**Time:** 60 minutes.

### Exercise 7. Standards map (critique)

**Deliverable:** a two-page memo mapping a wearable ECG prototype to the standards that would apply at each stage: bench, design verification, type test, and clinical use.

**Acceptance criteria:** the memo names the general standard, the collateral standard, the particular standard, the risk standard, the software standard, and the cybersecurity standard; it distinguishes binding requirements from conventions; it names at least three things the prototype has not done.

**Time:** 60 minutes.

### Exercise 8. Compliance-boundary review (design)

**Deliverable:** a review of a provided lab report that contains three overclaims, with a corrected sentence for each and a note on what evidence would be needed to support the original claim.

**Acceptance criteria:** all three overclaims are identified; the corrections preserve the true finding; the evidence note names the instrument, the method, and the organisation that could produce it; the review does not replace the finding with a disclaimer.

**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 What is the difference between a hazard and a hazardous situation?

A hazard is a potential source of harm. A hazardous situation is a circumstance in which a person is exposed to the hazard. A hot surface is a hazard; a hand reaching into an unguarded opening is the sequence of events; contact with the surface is the hazardous situation; a burn is the harm. The distinction matters because a control can act on the hazard (remove the heat), the exposure (guard the opening), or the harm (reduce the contact time).

### 13.2 Why is a DMM reading of "no voltage" not proof of isolation?

The DMM has a finite input impedance and a limited frequency response. It can read near zero on a circuit that is isolated at DC but coupled through a capacitor at mains frequency, or on a circuit whose source is floating at a high common-mode voltage, and it can read a plausible DC value that says nothing about the AC path that matters for leakage. Isolation is a property of the insulation system and the creepage and clearance, verified by inspection and by a dielectric test, not by a single voltage reading. A meter can also be on the wrong range or have a blown fuse.

### 13.3 A 5 V circuit is connected to a laptop charger. Which zone is it in?

It depends on the charger and the connections. If the charger is a certified SELV adapter with reinforced insulation and the laptop's USB ground is not connected to earth through another path, the circuit can be Zone 1. If the charger is a two-wire adapter with a Y-capacitor to the output, the output floats at about half the mains voltage through a high impedance, and the circuit is not a clean SELV source for a patient connection. If the laptop is also connected to an earthed monitor, the USB ground can be earthed. The classification must include every connection, not only the supply.

### 13.4 Compute the current through a 10 kΩ resistor with 0.42 V across it.

\(I = 0.42 / 10{,}000 = 42\ \mu\text{A}\). With the measured 9.98 kΩ, the current is 42.1 µA.

### 13.5 If the current is 42.1 µA and the type CF normal-condition limit is 10 µA, what is the margin?

The margin is \(10 - 42.1 = -32.1\ \mu\text{A}\). The result fails the limit by a factor of 4.21. The margin is negative, so the result is a failure, not a pass with a small margin.

### 13.6 Why does the type CF limit differ from the type BF limit?

Type CF applied parts are intended for direct cardiac contact, where a small current can cause ventricular fibrillation. Type BF applied parts are floating but not intended for direct cardiac contact. The CF limit is therefore an order of magnitude lower in normal condition. The exact values and conditions are in the standard's Table 3; the teaching numbers in this file are consistent with published summaries but must be confirmed against the current edition.

### 13.7 What is the difference between a type test and a routine test?

A type test is performed on a representative sample to demonstrate that the design meets the standard. A routine test is performed on every production unit to demonstrate that the manufacturing process has not introduced a defect. A type test is a design activity; a routine test is a production activity. A bench measurement is neither.

### 13.8 Why is PPE the last control?

Because PPE depends on the person wearing it correctly, every time, and it fails silently. An engineering control such as a guard or an interlock works without the person's attention. The hierarchy places elimination and engineering controls first because they are more reliable, and PPE last because it protects only the person who uses it and only while it is intact.

### 13.9 What does an emergency stop have to do?

It must be manually actuated, reachable, latched, and capable of removing energy from the actuators. It must not depend on software. It must not restart motion when released. It must be verified by measuring the actuator supply, not by watching the motion stop.

### 13.10 Rubric for Exercise 5

A strong pre-power checklist has objective tests, named instruments, pass/fail thresholds, and a fail action for every item. It covers the supply, the wiring, the current limit, the thermal check, the guard, the e-stop, the discharge, and the record. A weak checklist has items such as "check the board looks fine" and no fail action. A checklist that cannot be executed by a second person is not a checklist.

### 13.11 Rubric for Exercise 7

A strong standards map names the general, collateral, and particular standards, distinguishes binding from convention, and states the evidence each stage requires. A weak map lists standards without saying what they constrain, or claims that a bench test satisfies a type test. The strongest maps also name the standards that do not apply and explain why.

### 13.12 Rubric for Exercise 8

A strong review identifies the overclaim, preserves the true finding, and names the evidence that would support the stronger claim. A weak review replaces the finding with "results are preliminary" and does not say what was actually measured. The correction should be specific: not "this is not compliant" but "this bench measurement of 42 µA across a 10 kΩ model is not a type CF leakage test."

## 14. Interview questions

### 14.1 What is the difference between a bench measurement and a type test?

A strong answer names the evidence rung, the standard's method, the accredited laboratory, the sample, and the report. It says a bench measurement is one unit under lab conditions, while a type test is a conformity activity against a published method. A weak answer treats a good instrument as sufficient, or says a type test is "more accurate."

### 14.2 How would you classify an applied part?

A strong answer asks whether the part contacts the patient, whether it is floating, and whether it is intended for direct cardiac contact. It names type B, BF, and CF, and it says the classification drives the leakage limits and the isolation. A weak answer says "it touches the body, so it is type B" without the floating and cardiac distinctions.

### 14.3 Why is patient leakage current limited more tightly than touch current?

A strong answer explains the current path through the heart, the lower threshold for ventricular fibrillation, and the direct cardiac contact of a type CF part. It distinguishes macroshock from microshock and notes that the standard's limits are condition-specific. A weak answer says "patients are more sensitive" without the mechanism.

### 14.4 What does "SELV" mean, and what is the common mistake?

A strong answer defines a safety extra-low-voltage system as one that cannot exceed the applicable voltage limit in normal and single-fault conditions, and notes that there is no single universal number; the product standard governs. The common mistake is to treat any floating supply as SELV, or to treat 30 V/60 V as a universal definition. A weak answer quotes 30 V without a source or a condition.

### 14.5 What is the purpose of a bleeder resistor?

A strong answer says it discharges stored energy after power is removed, and that its time constant must be shorter than the access time. It notes that the resistor can fail open, so the voltage must still be measured before touching the circuit. A weak answer says "it makes the capacitor safe" without the failure mode.

### 14.6 Why can an oscilloscope create a hazard?

A strong answer explains that the probe ground is connected to protective earth, so clipping it to a non-earth node creates a current path through the earth conductor. It names the differential probe or the isolation transformer as the correct solutions, and it notes that a battery-powered scope is not automatically safe. A weak answer says "the scope is earthed, so it is safe."

### 14.7 What is the difference between a guard and an interlock?

A strong answer says a guard is a physical barrier and an interlock is a device that prevents operation when the guard is open. A guard without an interlock protects only against incidental contact; an interlock without a guard can be defeated. A safety function may need both. A weak answer uses the terms interchangeably.

### 14.8 How do you know an instrument is calibrated?

A strong answer asks for the certificate, the range, the parameter, the uncertainty, the date, and the traceability statement. It distinguishes calibration from adjustment and notes that a sticker is not evidence. A weak answer says "it has a calibration sticker" or "it is new."

### 14.9 What is the compliance boundary, and why does it matter commercially?

A strong answer says a lab demonstrates mechanisms and produces evidence about a prototype, while compliance requires a QMS, design controls, accredited testing, and a conformity-assessment route. It matters because a mislabelled claim fails at due diligence, at a regulatory submission, or in a recall investigation. A weak answer says "we are not allowed to say compliant" without explaining what would be needed.

### 14.10 How would you respond to a colleague who wants to connect a prototype to themselves to "see if it works"?

A strong answer refuses, explains the hazard, names the correct surrogate (a resistor network, a signal generator, a commercial simulator, or a recorded waveform), and escalates if the colleague persists. It does not negotiate about the voltage or the duration. A weak answer says "as long as it is under 5 V" or "only for a moment."

### 14.11 What is the first thing you do when a bench smells hot?

A strong answer stops work, removes power at the source without touching the DUT, warns others, and does not investigate until the circuit is de-energised and cool. It records the event and reports it. A weak answer says "I would check which component is hot" while the circuit is still powered.

### 14.12 What evidence would you need before claiming a device is safe?

A strong answer says safety is a claim about a defined use, a defined population, and a defined environment, supported by a risk file, verification and validation, clinical evidence where applicable, and post-market monitoring. It notes that no single lab result establishes safety, and that "safe" is not an absolute property. A weak answer says "we tested it and nothing bad happened."

## 15. Key terms

- **Applied part** — part of a medical device that contacts the patient in normal use.
- **Arc flash boundary** — distance at which incident energy equals the defined threshold; PPE is required inside it.
- **Basic insulation** — insulation providing basic protection against electric shock.
- **Bench measurement** — measurement of one unit under lab conditions; not a type test.
- **Calibration** — comparison against a reference with a stated uncertainty.
- **CAT rating** — measurement-category rating of an instrument, meaningful only with a voltage.
- **CDN** — coupling/decoupling network used in conducted-immunity testing.
- **Clearance** — shortest path through air between two conductive parts.
- **Compliance boundary** — the line between lab evidence and regulatory conformity.
- **Creepage** — shortest path along an insulating surface between two conductive parts.
- **Design validation** — evidence that the device meets user needs and intended use.
- **Design verification** — evidence that design output meets design input.
- **DHF** — design history file; the record of the design process.
- **Dielectric strength** — voltage an insulation withstands without breakdown.
- **Double insulation** — basic plus supplementary insulation.
- **DUT** — device under test.
- **Earth leakage current** — current flowing in the protective earth conductor.
- **ELV** — extra-low voltage; the limit is defined by the governing standard, not universally.
- **Emergency stop** — manual, latching function that removes hazardous energy.
- **ESD** — electrostatic discharge.
- **EUT** — equipment under test; the EMC term for the DUT.
- **Guard** — physical barrier that prevents access to a hazard.
- **Harm** — injury or damage to health, property, or environment.
- **Hazard** — potential source of harm.
- **Hazardous situation** — circumstance in which a person is exposed to a hazard.
- **Interlock** — device that prevents operation when a guard is open.
- **LISN** — line impedance stabilisation network.
- **LOTO** — lockout/tagout; isolation and securing of energy sources.
- **MOOP** — means of operator protection.
- **MOPP** — means of patient protection.
- **NRTL** — Nationally Recognised Testing Laboratory.
- **Patient auxiliary current** — current flowing between patient connections.
- **Patient leakage current** — current flowing from an applied part through the patient to earth.
- **PELV** — protective extra-low voltage; an earthed ELV system.
- **Performance level (PL)** — discrete reliability level of a safety function in ISO 13849.
- **PPE** — personal protective equipment.
- **Pre-compliance** — early in-house EMC testing for margin; not compliance.
- **QMS** — quality management system.
- **Qualified person** — person trained and authorised for the electrical work.
- **Reinforced insulation** — single insulation with protection equivalent to double.
- **Residual risk** — risk remaining after controls.
- **Risk** — combination of probability of harm and severity of harm.
- **Routine test** — test performed on every production unit.
- **Safety envelope** — set of limits within which motion is permitted.
- **Safety integrity level (SIL)** — discrete reliability level in IEC 61508 and IEC 62061.
- **SELV** — safety extra-low voltage; cannot exceed the limit in normal or single-fault conditions.
- **Supplementary insulation** — independent insulation in addition to basic.
- **Touch current** — current flowing from an accessible part to earth through a person.
- **Traceability** — unbroken chain of calibrations to a reference.
- **Type B, BF, CF** — applied-part classifications with different leakage limits.
- **Type test** — conformity test on a representative sample to a standard method.
- **Uncertainty** — quantified doubt in a measured value.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and regulatory

- IEC 60601-1:2005+AMD1:2012+AMD2:2020, Medical electrical equipment — Part 1: General requirements for basic safety and essential performance. https://webstore.iec.ch/
- IEC 60601-1-2:2014+AMD1:2020, Electromagnetic disturbances — Requirements and tests. https://webstore.iec.ch/en/publication/67554
- IEC 60601-1-11:2015+AMD1:2020, Medical electrical equipment used in the home healthcare environment. https://webstore.iec.ch/
- IEC 61010-1:2010+AMD1:2016, Safety requirements for electrical equipment for measurement, control, and laboratory use. https://webstore.iec.ch/en/publication/4279
- IEC 61010-2-030, Particular requirements for testing and measuring circuits. https://webstore.iec.ch/
- IEC 60479-1:2018, Effects of current on human beings and livestock — Part 1: General aspects. https://webstore.iec.ch/
- IEC 61140:2016, Protection against electric shock — Common aspects for installation and equipment. https://webstore.iec.ch/
- ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
- IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes. https://www.iso.org/standard/64686.html
- IEC 80601-2-77:2019+AMD1:2023, Robotically assisted surgical equipment. https://www.iso.org/standard/68473.html
- IEC 80601-2-78:2019+AMD1:2024, Medical robots for rehabilitation, assessment, compensation or alleviation. https://www.iso.org/standard/68474.html
- ISO 10218-1:2025, Robotics — Safety requirements — Part 1: Industrial robots. https://www.iso.org/standard/73933.html
- ISO/TS 15066:2016, Robots and robotic devices — Collaborative robots. https://www.iso.org/standard/62996.html
- Regulation (EU) 2017/745 on medical devices, Annex II. https://eur-lex.europa.eu/eli/reg/2017/745/oj

### Occupational safety

- OSHA 29 CFR 1910 Subpart S, Electrical. https://www.osha.gov/laws-regs/regulations/standardnumber/1910
- OSHA 3404, Laboratory Safety Guidance. https://www.osha.gov/sites/default/files/publications/OSHA3404.pdf
- OSHA 4472, Protecting Employees from Electric-Arc Flash Hazards. https://www.osha.gov/sites/default/files/publications/OSHA4472.pdf
- OSHA 4474, Establishing Boundaries Around Arc Flash Hazards. https://www.osha.gov/sites/default/files/publications/OSHA4474.pdf
- NFPA 70E, Standard for Electrical Safety in the Workplace. https://www.nfpa.org/
- Stanford University, Electrical Safety Management Plan for Research. https://ehs-library.stanford.edu/wp-content/uploads/Electrical-Safety-Management-Plan-for-Research.pdf

### Guidance and datasheets

- Rigel Medical, A Practical Guide to IEC 60601. https://www.rigelmedical.com/
- Fluke Biomedical, Introduction to Electrical Safety Testing. https://www.flukebiomedical.com/
- Megger, The Measurement Category (CAT) rating. https://www.megger.com/en/safety/measurement-category
- FDA, Recognized Consensus Standards database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfStandards/search.cfm
- FDA, Design Controls. https://www.fda.gov/media/142013/download

> **Note:** The standard texts are paywalled. The secondary guides above were used to check the teaching values, not to substitute for the standards. Confirm every clause and table against the current edition before citing a number in a regulatory document.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: none; this is the first topic.
- Next topic: [`02_bench-and-simulation-stack.md`](02_bench-and-simulation-stack.md)
- Sibling: [`03_lab-conventions.md`](03_lab-conventions.md)
- Sibling: [`09_evidence-dossier.md`](09_evidence-dossier.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
