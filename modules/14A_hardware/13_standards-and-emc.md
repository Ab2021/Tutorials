# Standards and EMC

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 13 of 14
> Estimated study time: 4 hours
> Prerequisites: 01, 08
> Version 2026.3

## 1. Why this topic exists

A medical device must work in an electromagnetic environment full of radios, switching supplies, electrosurgery, and electrostatic discharge. It must also not emit interference that disrupts other devices. The standards that govern this are the IEC 60601-1 family and CISPR 11, and they are the most common cause of a late redesign.

The industry problem is that EMC is treated as a test at the end of the project rather than a design property. A device that passes functional tests on the bench fails radiated emissions because of a cable, or fails immunity because of a filter that was omitted, or passes at 3 V/m and fails at 10 V/m in the home. Each failure costs a board revision, a re-test, and a delay.

Three decisions depend on standards and EMC literacy:

1. **Which standard applies.** The general standard, the collateral standards, the particular standards, and the emissions standard each constrain a different aspect. A device can meet one and fail another.
2. **What the test levels mean.** A 3 V/m immunity level is not a guarantee of performance; it is a test condition. The device must meet its performance criterion at that level, and the criterion must be defined.
3. **What a clause requires versus what a test practice does.** A clause is a requirement; a test practice is a way to demonstrate it. The distinction determines what evidence is acceptable and what is not.

This topic maps the IEC 60601-1 family, explains the EMC requirements and test levels, distinguishes pre-compliance from accredited testing, and works an EMC test-plan budget and a failure case. It assumes the signal-chain, front-end, and firmware material from earlier topics.

A useful framing: EMC is a property of the whole device, not of a filter. The enclosure, the cables, the PCB, the power supply, and the firmware all contribute. A filter that fixes one frequency can create a resonance at another.

## 2. Learning outcomes

By the end of this topic you can:

- Map a device to the applicable IEC 60601-1 general, collateral, and particular standards.
- Explain the difference between emissions and immunity, and between CISPR 11 Group 1/2 and Class A/B.
- State the IEC 60601-1-2 test levels for ESD, radiated RF, conducted RF, and magnetic fields, and the environment that applies.
- Distinguish a clause requirement from a test practice.
- Design a pre-compliance programme and an accredited test plan with a budget.
- Diagnose a conducted-emissions failure and propose a filter and layout fix.
- Explain why a device that passes at 3 V/m may fail at 10 V/m.
- Write a verification matrix that links each requirement to a test.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A device is an electromagnetic system with sources, paths, and victims. The source is a switching supply, a clock, a radio, or an electrosurgical unit. The path is a cable, a PCB trace, an enclosure seam, or the air. The victim is the device itself or another device.

```text
   SOURCE ---> PATH ---> VICTIM
      |          |          |
   switching   cable      sensor
   clock       trace      processor
   radio       seam       radio
   ESD         air        patient
      |          |          |
      +----------+----------+
                 |
          EMISSIONS (source)  and  IMMUNITY (victim)
```

The model has four quantities:

| Quantity | Emissions | Immunity |
|---|---|---|
| Source | Device | External |
| Path | Cable, enclosure | Cable, enclosure |
| Victim | Other device | Device |
| Standard | CISPR 11 | IEC 60601-1-2 |

An emissions problem is the device acting as a source. An immunity problem is the device acting as a victim. The same cable can be both: it can radiate the device's noise and pick up external noise.

### 3.1 The standards map

| Layer | Standard | Scope |
|---|---|---|
| General | IEC 60601-1 | Basic safety and essential performance |
| Collateral | IEC 60601-1-2 | Electromagnetic disturbances |
| Collateral | IEC 60601-1-8 | Alarm systems |
| Collateral | IEC 60601-1-11 | Home healthcare |
| Particular | IEC 60601-2-25 | Electrocardiographs |
| Particular | IEC 80601-2-77 | Robotically assisted surgical equipment |
| Particular | IEC 80601-2-78 | Rehabilitation robots |
| Emissions | CISPR 11 | Industrial, scientific, and medical equipment |

The general standard applies to all devices. The collateral standards apply to the aspects they cover. The particular standards apply to the device type and take precedence over the general and collateral standards where they conflict.

### 3.2 Emissions: CISPR 11

CISPR 11 classifies equipment by group and class.

| Group | Definition | Example |
|---|---|---|
| Group 1 | No intentional RF energy | Most monitors, processors |
| Group 2 | Intentional RF energy | Diathermy, induction heaters |

| Class | Environment | Limit |
|---|---|---|
| Class A | Industrial, professional | Less stringent |
| Class B | Residential, home | More stringent |

A home-use device must meet Class B. A professional device may meet Class A, but many hospitals require Class B. The group and class determine the limits and the test methods.

### 3.3 Immunity: IEC 60601-1-2

IEC 60601-1-2:2014+AMD1:2020 sets the immunity test levels by environment. The levels below are the ones a device must withstand, with a performance criterion defined by the manufacturer.

| Test | Professional healthcare | Home healthcare |
|---|---|---|
| ESD contact | ±8 kV | ±8 kV |
| ESD air | ±15 kV | ±15 kV |
| Radiated RF 80 MHz–2.7 GHz | 3 V/m | 10 V/m |
| Conducted RF 150 kHz–80 MHz | 3 V RMS | 3 V RMS (6 V in ISM) |
| Power-frequency magnetic field | 30 A/m | 30 A/m |
| Proximity magnetic field | 134.2 kHz: 65 A/m; 13.56 MHz: 7.5 A/m | Same, plus 30 kHz: 8 A/m |

The performance criterion must be stated for each test. A device that "continues to operate" is not a criterion; "the heart-rate error remains within 5 bpm" is.

### 3.4 Requirement versus test practice

A clause is a requirement. A test practice is a method to demonstrate it. The distinction matters because a test practice can be changed without changing the requirement, and a requirement can be met by a different test.

| Clause | Requirement | Test practice |
|---|---|---|
| 60601-1-2, 8.9 | Immunity to radiated RF | IEC 61000-4-3, 3 V/m, 80% AM |
| 60601-1-2, 8.10 | Immunity to conducted RF | IEC 61000-4-6, 3 V RMS |
| 60601-1-2, 8.11 | Proximity magnetic fields | IEC 61000-4-39 |
| CISPR 11 | Emissions limits | Test at 10 m or 3 m, depending on the standard |

A test laboratory applies the test practice. The manufacturer is responsible for the requirement. A device that passes the test practice but fails the requirement in use is still non-compliant in spirit, even if the test report says otherwise.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| EMC | Electromagnetic compatibility | Confused with EMI | None |
| EMI | Electromagnetic interference | Confused with EMC | None |
| Emissions | Energy leaving the device | Confused with immunity | dBµV/m |
| Immunity | Ability to withstand interference | Confused with emissions | V/m, kV |
| CISPR 11 | Emissions standard | Confused with 60601-1-2 | Standard |
| Group 1 | No intentional RF | Confused with Class 1 | Group |
| Group 2 | Intentional RF | Confused with Class 2 | Group |
| Class A | Industrial limits | Confused with Class B | Class |
| Class B | Residential limits | Confused with Class A | Class |
| Conducted emissions | Noise on cables | Confused with radiated | dBµV |
| Radiated emissions | Noise in the air | Confused with conducted | dBµV/m |
| ESD | Electrostatic discharge | Confused with surge | kV |
| Surge | Transient overvoltage | Confused with ESD | kV |
| EFT | Electrical fast transient | Confused with surge | kV |
| Burst | Repetitive fast transient | Confused with surge | kV |
| Dip | Short supply reduction | Confused with brown-out | % |
| Interruption | Supply loss | Confused with dip | s |
| Immunity level | Test level | Confused with the performance criterion | V/m |
| Performance criterion | Required behaviour during the test | Confused with the test level | None |
| Pre-compliance | In-house test | Confused with accredited | None |
| Accredited test | Test by an accredited lab | Confused with pre-compliance | None |
| Chamber | Shielded room | Confused with an open-area test site | None |
| LISN | Line impedance stabilization network | Confused with a filter | Ω |
| Filter | Component that attenuates noise | Confused with a shield | dB |
| Common-mode choke | Inductor that blocks common-mode current | Confused with a differential filter | H |
| Shield | Conductive barrier | Confused with a filter | dB |
| Grounding | Reference connection | Confused with shielding | Ω |
| Bonding | Low-impedance connection | Confused with grounding | Ω |
| Resonance | Frequency where impedance peaks | Confused with a filter corner | Hz |
| Spread spectrum | Modulation of the clock | Confused with a filter | % |
| Harmonics | Multiples of a frequency | Confused with emissions | Hz |
| Flicker | Supply voltage fluctuation | Confused with harmonics | % |

## 5. The mechanics, step by step

The procedure below designs and verifies EMC compliance. Each step states its input, output, and rationale. The required edge cases are mapped to the steps and summarised in Section 5.13.

### Step 1. Determine the applicable standards

**Input:** the device's function, environment, and technology.

**Output:** a list of general, collateral, particular, and emissions standards.

**Rationale:** the standards determine the tests, the levels, and the evidence. A home-use device needs IEC 60601-1-11 and CISPR 11 Class B; a surgical robot needs IEC 80601-2-77. A device that is both needs both.

The list must be reviewed by regulatory and EMC. A standard that is omitted is a test that is not performed.

### Step 2. Define the intended environment

**Input:** the use location, the nearby equipment, and the user.

**Output:** a professional or home healthcare classification.

**Rationale:** the environment determines the immunity levels. A home device must meet 10 V/m radiated RF, which is 10 dB harsher than the professional 3 V/m. A device that is used in both must meet the harsher level.

The environment is part of the intended use, not a marketing choice. A device labelled for home use must meet the home levels.

### Step 3. Identify the sources and victims

**Input:** the schematic, the clock, the power supply, and the radios.

**Output:** a list of noise sources and susceptible circuits.

**Rationale:** the sources are the switching supply, the clocks, the digital buses, the radios, and the electrosurgical environment. The victims are the analogue front end, the sensors, the radios, and the patient-connected circuits.

A source that is not identified cannot be filtered. A victim that is not identified cannot be protected.

### Step 4. Design the PCB and enclosure

**Input:** the sources, the victims, and the mechanical design.

**Output:** a stack-up, a layout, and an enclosure with seams and cables.

**Rationale:** the PCB is the first defence. A solid ground plane, short return paths, and separated analogue and digital sections reduce the emissions and the susceptibility. The enclosure's seams and cable entries are the second defence.

A layout that is fixed after the EMC test is a layout that will be redesigned. The EMC review must happen at the schematic and layout stages.

### Step 5. Design the filters

**Input:** the source spectrum and the limits.

**Output:** a filter topology, component values, and an attenuation estimate.

**Rationale:** a filter must attenuate the source at the frequencies where it exceeds the limit. A single-stage LC filter has a 40 dB/decade roll-off; a two-stage filter has 80 dB/decade. The filter's corner must be below the lowest frequency of concern, and its components must handle the current.

A filter that is designed for differential noise may not attenuate common-mode noise. The topology must match the noise mode.

### Step 6. Design the shielding and grounding

**Input:** the enclosure, the cables, and the safety requirements.

**Output:** a shielding strategy and a bonding scheme.

**Rationale:** a shield without a low-impedance bond is an antenna. The cable shield must be bonded at the connector, and the enclosure must be bonded to the ground plane. The safety ground and the EMC ground must be compatible; a safety requirement can prohibit a bond.

The grounding scheme must be designed with the leakage current requirement. A bond that reduces emissions can increase the leakage current.

### Step 7. Design the firmware for EMC

**Input:** the processor, the peripherals, and the communication.

**Output:** a firmware strategy: spread spectrum, watchdog, error detection, recovery.

**Rationale:** firmware can reduce emissions by spreading the clock and can improve immunity by detecting and recovering from errors. A watchdog that resets the device after an upset is an immunity measure.

A firmware that ignores a communication error is a firmware that will fail the immunity test. The error handling is part of the EMC design.

### Step 8. Run pre-compliance tests

**Input:** a prototype and an in-house test setup.

**Output:** a spectrum, a list of failures, and a margin estimate.

**Rationale:** pre-compliance finds the big problems before the accredited test. A spectrum analyzer, a LISN, a near-field probe, and a TEM cell can measure the emissions and the susceptibility at a fraction of the cost.

A pre-compliance test is not a substitute for the accredited test. It is a screening tool; the accredited test is the evidence.

### Step 9. Fix the failures and re-test

**Input:** the failures and the design changes.

**Output:** a revised design and a new pre-compliance result.

**Rationale:** the fix must address the source, the path, or the victim. A filter at the connector fixes the path; a spread-spectrum clock fixes the source; a shielded cable fixes the victim. A fix that only moves the resonance is not a fix.

The re-test must use the same setup, or the comparison is invalid.

### Step 10. Plan the accredited test

**Input:** the standards, the test laboratory, and the schedule.

**Output:** a test plan, a budget, and a booking.

**Rationale:** the accredited test is a scheduled event with a chamber, a laboratory, and a report. The plan must include the test list, the operating modes, the cables, the worst-case configuration, and the performance criteria.

The laboratory's accreditation must cover the tests and the standards. A laboratory that is accredited for CISPR 11 may not be accredited for IEC 60601-1-2.

### Step 11. Execute the accredited test

**Input:** the device, the test plan, and the laboratory.

**Output:** a test report with the results and the margins.

**Rationale:** the test must be performed in the worst-case configuration. A device that passes in a favourable configuration may fail in the intended use. The operating mode, the cable routing, and the patient simulator must be documented.

The report must include the performance criterion and the observed behaviour, not only the pass/fail.

### Step 12. Document and maintain

**Input:** the test report, the design, and the production process.

**Output:** a compliance record and a change-control procedure.

**Rationale:** a change to the PCB, the enclosure, the cable, or the firmware can invalidate the EMC result. The change control must assess the EMC impact and re-test if necessary.

A device that is modified after the test without a re-assessment is a device with an unknown EMC status.

### 5.12a Design review checklist

The review is held twice: once at schematic capture and once at layout release. Each item has an owner and an evidence artefact. An item that is not reviewed is an item that is discovered in the chamber.

| Item | Question | Evidence | Owner |
|---|---|---|---|
| Standards list | Are all applicable standards identified? | Standards matrix | Regulatory |
| Environment | Professional or home? | Intended-use statement | Regulatory |
| Emissions class | Group and class stated? | Classification memo | EMC |
| Supply topology | Switching frequency and harmonics known? | Supply datasheet | Hardware |
| Clock plan | Frequencies, edges, spread spectrum? | Clock tree | Hardware |
| Cable list | Every cable, length, shield, connector? | Cable schedule | Hardware |
| Enclosure | Material, seams, apertures, bonding? | Mechanical drawing | Mechanical |
| Grounding | Safety and EMC ground compatible? | Grounding diagram | Hardware |
| Filter | Topology matches noise mode? | Filter calculation | Hardware |
| Filter placement | At the connector, short return? | Layout plot | Layout |
| ESD paths | Discharge path to ground defined? | ESD map | Hardware |
| Reset lines | Filtered and pulled up? | Schematic | Hardware |
| Firmware recovery | Watchdog and error detection? | Firmware spec | Firmware |
| Patient circuits | Isolation and leakage verified? | Isolation report | Hardware |
| Labelling | EMC warning and cable list? | Label proof | Regulatory |
| Change control | EMC impact assessed? | Change procedure | Quality |

### 5.12b Test setup record

The test report is only reproducible if the setup is recorded. The table is the minimum record for each test.

| Parameter | Example | Why it matters |
|---|---|---|
| Standard and edition | IEC 60601-1-2:2014+AMD1:2020 | Levels differ by edition |
| Test level | 10 V/m | Determines the stress |
| Frequency range | 80 MHz–2.7 GHz | Determines the sweep |
| Modulation | 80% AM, 1 kHz | Determines the envelope |
| Antenna distance | 3 m | Field calibration |
| Antenna height | 1.5 m | Uniformity |
| Device orientation | 3 axes | Worst case |
| Cable routing | As intended use | Coupling path |
| Patient simulator | Specified load | Realistic impedance |
| Operating mode | Worst-case mode | Emission and immunity |
| Performance criterion | Defined per test | Pass/fail |
| Ambient | Temperature, humidity | ESD and leakage |

### 5.12c Operating modes for the test

A device that is tested in one mode may pass while failing in another. The plan must list the modes and select the worst case with a rationale.

| Mode | Load | Radio | Display | Why tested |
|---|---|---|---|---|
| Standby | None | Off | Dim | Lowest emission |
| Idle | None | On | Bright | Clock and radio |
| Nominal | 50% | On | Bright | Typical use |
| Maximum | 100% | On | Bright | Supply stress |
| Charging | Battery | On | Bright | Charger emission |
| Alarm | 100% | On | Flashing | Audible and visible |
| Update | None | On | Progress | Flash write |
| Fault | Simulated | On | Alarm | Recovery |

### 5.12d Applicability decision table

The decision has more than two outcomes. The table maps the device and the environment to the standards and the tests.

| Device | Environment | Emissions | Immunity | Particular |
|---|---|---|---|---|
| Monitor, no RF | Professional | CISPR 11 Group 1 Class A | 60601-1-2 professional | 60601-2-25 if ECG |
| Monitor, no RF | Home | CISPR 11 Group 1 Class B | 60601-1-2 home | 60601-1-11 |
| Monitor, Bluetooth | Home | CISPR 11 Group 1 Class B | 60601-1-2 home | 60601-1-11 |
| Infusion pump | Home | CISPR 11 Group 1 Class B | 60601-1-2 home | 60601-2-24 |
| Surgical robot | Professional | CISPR 11 Group 1 Class A | 60601-1-2 professional | 80601-2-77 |
| Rehab robot | Professional | CISPR 11 Group 1 Class A | 60601-1-2 professional | 80601-2-78 |
| Diathermy | Professional | CISPR 11 Group 2 Class A | 60601-1-2 professional | 60601-2-2 |
| Home ventilator | Home | CISPR 11 Group 1 Class B | 60601-1-2 home | 60601-2-12 |

The table is not a substitute for the standards. It is a starting point for the standards matrix, which the regulatory function must confirm against the current editions.

### 5.12e Clause requirement and evidence

Each requirement needs an evidence artefact. The table links the clause family to the evidence.

| Clause family | Requirement | Evidence | Typical test |
|---|---|---|---|
| 60601-1, 11 | Leakage current | Type test report | Leakage meter |
| 60601-1, 8.5 | Isolation | Dielectric test | Hi-pot |
| 60601-1-2, 8.9 | Radiated RF immunity | EMC report | 61000-4-3 |
| 60601-1-2, 8.10 | Conducted RF immunity | EMC report | 61000-4-6 |
| 60601-1-2, 8.11 | Proximity field immunity | EMC report | 61000-4-39 |
| 60601-1-2, 9 | Emissions | EMC report | CISPR 11 |
| 60601-1-8 | Alarm performance | Alarm test | Audible and visual |
| 60601-1-11 | Home environment | Usability and EMC | Both |
| 62304 | Software lifecycle | Software file | Audit |
| 14971 | Risk management | Risk file | Review |

The evidence must be current for the device configuration. A report for a previous revision is not evidence for the current revision unless the change was assessed as having no EMC impact.

### 5.12f Pre-compliance equipment

A pre-compliance bench does not need a full chamber. It needs enough sensitivity to find the failures that would cost a re-test.

| Instrument | Use | Typical capability | Limitation |
|---|---|---|---|
| Spectrum analyser | Emissions scan | 9 kHz–3 GHz | No chamber |
| LISN | Conducted emissions | 150 kHz–30 MHz | Mains only |
| Near-field probe | Locate source | 10 MHz–1 GHz | No absolute level |
| TEM cell | Radiated immunity | Up to 1 GHz | Small device |
| ESD simulator | ESD immunity | ±8 kV contact | Repeatability |
| Signal generator | Conducted immunity | 150 kHz–80 MHz | Needs amplifier |
| Current probe | Cable current | 10 kHz–100 MHz | Calibration |
| Oscilloscope | Transient capture | 1 GHz, 5 GSa/s | Probe loading |

The bench's absolute accuracy is limited. The bench's value is relative: it shows whether a change reduced or increased the emission, and whether a transient appears on a line.

### 5.12g Failure triage

When a test fails, the triage determines whether the fix is a filter, a layout, a shield, or firmware. The decision has more than two outcomes.

| Observation | Likely cause | First check | Fix |
|---|---|---|---|
| Emission peaks at switching frequency | Supply | Supply spectrum | Input filter |
| Emission peaks at clock harmonic | Clock | Clock tree | Spread spectrum, filter |
| Emission rises when cable attached | Cable | Current probe | Choke, shield bond |
| Emission rises when enclosure closed | Seam, aperture | Near-field probe | Bond, gasket |
| Immunity fails at one frequency | Resonance | Sweep resolution | Damping, layout |
| ESD resets processor | Reset line | Scope on reset | Filter, firmware |
| EFT resets processor | Supply transient | Supply rail | TVS, filter |
| Conducted RF corrupts data | Cable coupling | Error log | Choke, shielding |
| Magnetic field shifts sensor | Loop area | Probe position | Orientation, shield |
| Dip causes reset | Hold-up time | Rail monitor | Capacitor, threshold |

The triage is iterative. A fix that changes the symptom without changing the cause will reappear at another frequency.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 5, 8 | Filter or amplifier saturates | Spectrum; functional test | Loss of signal |
| Common-mode rejection failure | 4, 6 | Cable or ground | Immunity test | Interference |
| Electrode offset | 4 | Front-end saturation | Functional test | Loss of signal |
| Clock jitter | 7 | Spread-spectrum modulation | Spectrum | Timing error |
| Brown-out | 7, 11 | Supply dip | Immunity test | Reset |
| Watchdog reset loop | 7, 11 | Firmware upset | Reset log | Device unavailable |
| Thermal throttling | 5 | Filter heating | Temperature | Performance loss |
| Firmware update failure | 12 | Update during EMC | Version check | Bricked device |
| Calibration drift | 12 | Component ageing | Calibration check | Bias |

The EMC edge cases are dominated by saturation, ground loops, and firmware upset. The processor edge cases enter through the immunity test.

### 5.14 Emissions versus immunity tests

| Test | Standard | What it measures | Typical failure |
|---|---|---|---|
| Conducted emissions | CISPR 11 | Noise on the mains | Switching supply |
| Radiated emissions | CISPR 11 | Noise in the air | Clock, cable |
| ESD | IEC 61000-4-2 | Discharge immunity | Reset, latch-up |
| Radiated RF | IEC 61000-4-3 | Field immunity | Sensor error |
| Conducted RF | IEC 61000-4-6 | Cable immunity | Communication error |
| EFT | IEC 61000-4-4 | Fast transient immunity | Reset |
| Surge | IEC 61000-4-5 | Slow transient immunity | Damage |
| Magnetic field | IEC 61000-4-8 | Power-frequency immunity | Sensor error |
| Dips and interruptions | IEC 61000-4-11 | Supply immunity | Reset |

The table is the test map. A device must pass the emissions tests and the immunity tests; a failure in either is a non-compliance.

### 5.15 Pre-compliance versus accredited

| Aspect | Pre-compliance | Accredited |
|---|---|---|
| Location | In-house | Accredited lab |
| Cost | Low | High |
| Evidence | Internal | Report |
| Accuracy | Moderate | High |
| Use | Design iteration | Submission |

The pre-compliance test is a design tool. The accredited test is the evidence. A team that skips pre-compliance pays for the learning in the accredited chamber, at a higher hourly rate.

### 5.16 Test-plan budget template

| Item | Hours | Rate | Cost |
|---|---:|---:|---:|
| Pre-compliance emissions | 40 | $150 | $6,000 |
| Pre-compliance immunity | 24 | $150 | $3,600 |
| Accredited emissions | 24 | $250 | $6,000 |
| Accredited immunity | 32 | $250 | $8,000 |
| Setup and documentation | 16 | $200 | $3,200 |
| Re-test allowance (50%) | 28 | $250 | $7,000 |
| **Total** | **164** | | **$33,800** |

The budget is illustrative. The point is that the re-test allowance is a line item, not a contingency. A design that fails once and needs a board revision can double the cost.

## 6. Worked example

### 6.1 The device

A home-use infusion pump has a 24 V, 2 A switching supply at 100 kHz, a 32-bit microcontroller at 100 MHz, a Bluetooth radio, and a patient-connected pressure sensor. The device must meet CISPR 11 Class B emissions and IEC 60601-1-2 home healthcare immunity.

### 6.1a Device parameters

| Parameter | Value | Source | Consequence |
|---|---|---|---|
| Supply | 24 V, 2 A, 100 kHz | Supply datasheet | Fundamental at 100 kHz |
| Supply topology | Flyback | Schematic | Rich harmonic content |
| MCU clock | 100 MHz | Clock tree | Harmonics to 1 GHz |
| Radio | Bluetooth, 2.4 GHz | Module datasheet | Intentional radiator |
| Sensor | Piezoresistive pressure | Sensor datasheet | Low-level analogue |
| Enclosure | ABS, uncoated | Mechanical drawing | No shielding |
| Mains cable | 2 m, unshielded | Cable schedule | Common-mode path |
| Patient cable | 1.5 m, shielded | Cable schedule | Shield bond critical |
| Emissions class | CISPR 11 Group 1 Class B | Classification | Home limits |
| Immunity | 60601-1-2 home | Intended use | 10 V/m |
| Performance criterion | Rate error ≤5% | Risk file | Pass/fail |
| Test laboratory | Accredited, ISO 17025 | Procurement | Report accepted |

### 6.2 The emissions failure

The pre-compliance scan shows a conducted emission at 150 kHz of 72 dBµV, against the CISPR 11 Group 1 Class B quasi-peak limit of 66 dBµV. The margin is −6 dB, a failure. The component at 300 kHz is 64 dBµV against a limit of 60.2 dBµV, a −3.8 dB margin. The limit falls linearly with the logarithm of frequency between 150 kHz and 500 kHz, from 66 dBµV to 56 dBµV.

The source is the switching supply's fundamental. The 150 kHz component is the 1.5th harmonic of the 100 kHz switching frequency, which is not a pure harmonic; it is a component of the supply's jitter and the rectifier's nonlinearity.

### 6.2a Measured emissions

The pre-compliance scan used a LISN on the mains input and a spectrum analyser with a 9 kHz resolution bandwidth. The quasi-peak detector was used for the comparison with the Class B limit.

| Frequency | Measured | Class B limit | Margin | Verdict |
|---|---:|---:|---:|---|
| 150 kHz | 72 dBµV | 66.0 dBµV | −6.0 dB | Fail |
| 300 kHz | 64 dBµV | 60.2 dBµV | −3.8 dB | Fail |
| 500 kHz | 52 dBµV | 56.0 dBµV | +4.0 dB | Pass |
| 1 MHz | 44 dBµV | 56.0 dBµV | +12.0 dB | Pass |
| 5 MHz | 40 dBµV | 56.0 dBµV | +16.0 dB | Pass |
| 10 MHz | 42 dBµV | 60.0 dBµV | +18.0 dB | Pass |
| 30 MHz | 36 dBµV | 60.0 dBµV | +24.0 dB | Pass |

The failure is at the low end, where the supply's harmonics are strongest and the limit is highest in absolute terms but the emission rises faster. The 300 kHz result fails by 3.8 dB, which is outside the measurement uncertainty of the pre-compliance bench, so it must be fixed rather than retested.

### 6.3 The filter calculation

A two-stage LC filter with \(L = 100\) µH and \(C = 10\) µF has a corner at:

\[
f_c = \frac{1}{2\pi\sqrt{LC}} = \frac{1}{2\pi\sqrt{100 \times 10^{-6} \times 10 \times 10^{-6}}} = 5.03\ \text{kHz}
\]

The attenuation at 150 kHz for a second-order filter is:

\[
A = 40 \log_{10}\left(\frac{150}{5.03}\right) = 59.0\ \text{dB}
\]

The filter is more than enough at 150 kHz. The problem is that the filter's actual attenuation is limited by the layout, the component parasitics, and the common-mode path. A measured attenuation of 12 dB is more realistic for a first-pass layout.

With 12 dB of measured attenuation at 150 kHz, the 72 dBµV emission becomes 60 dBµV, which is 6 dB below the 66 dBµV limit. The margin is 6 dB, which is adequate but not generous. The same filter gives about 16 dB at 300 kHz on paper; the measured value must be used, not the ideal 71 dB.

### 6.3a Filter components

| Component | Value | Function | Constraint |
|---|---|---|---|
| L1 | 100 µH | Series impedance | 2 A saturation |
| C1 | 10 µF | Shunt impedance | X7R, 50 V |
| L2 | 100 µH | Second stage | 2 A saturation |
| C2 | 10 µF | Second stage shunt | X7R, 50 V |
| Common-mode choke | 4.7 mH | Common-mode path | 2 A, 4430 Ω at 150 kHz |
| Y-capacitor | 2.2 nF | Common-mode shunt | Leakage limit |
| TVS | 33 V | Surge clamp | Standoff above 24 V |
| Bleeder | 10 kΩ | Discharge | Power loss |

The differential filter handles the supply's ripple. The common-mode choke and the Y-capacitor handle the cable current. The Y-capacitor's value is limited by the leakage current requirement, which is why the common-mode attenuation is often the harder problem.

The common-mode choke's impedance at 150 kHz is \(2\pi f L = 2\pi 	imes 150\,000 	imes 4.7 	imes 10^{-3} = 4430\ \Omega\). The impedance falls with frequency above the self-resonance, so the choke's datasheet must be checked at the frequencies of interest.

### 6.4 The re-test

The re-test with the filter shows 60 dBµV at 150 kHz and 48 dBµV at 300 kHz. The margins are 6 dB and 12.2 dB. The device passes the pre-compliance scan.

The accredited test confirms the result, with a 5 dB margin at 150 kHz. The difference between the pre-compliance and the accredited result is 1 dB, which is within the measurement uncertainty.

### 6.4a Re-test results

| Frequency | Before | After | Change | Limit | Margin |
|---|---:|---:|---:|---:|---:|
| 150 kHz | 72 dBµV | 60 dBµV | −12 dB | 66.0 dBµV | +6.0 dB |
| 300 kHz | 64 dBµV | 48 dBµV | −16 dB | 60.2 dBµV | +12.2 dB |
| 500 kHz | 52 dBµV | 32 dBµV | −20 dB | 56.0 dBµV | +24.0 dB |
| 1 MHz | 44 dBµV | 22 dBµV | −22 dB | 56.0 dBµV | +34.0 dB |

The measured attenuation rises with frequency, but far more slowly than the ideal 40 dB/decade because the capacitors' parasitic inductance and the layout's inductance limit the shunt path. The result is adequate, but the margin at 150 kHz is only 6 dB, which is the minimum normally accepted for a design that will be produced in volume.

The accredited test measured 61 dBµV at 150 kHz, a 5 dB margin. The 1 dB difference from the pre-compliance result is within the combined uncertainty of the two setups.

### 6.5 Diagnostic 1: the filter works on the bench but not in the enclosure

The filter is tested on an open board and shows 20 dB of attenuation. In the enclosure, the attenuation falls to 8 dB. The cable from the filter to the connector is picking up the switching noise and bypassing the filter.

The diagnostic is to measure the spectrum with the enclosure open and closed. If the emission rises when the enclosure is closed, the cable or the seam is the path. A near-field probe can locate the leak.

The design flaw is the layout, not the filter. The fix is to place the filter at the connector, shorten the cable, and bond the enclosure.

### 6.5a Diagnostic matrix

| Symptom | Measurement | Observation | Conclusion |
|---|---|---|---|
| Enclosure open vs closed | Spectrum at 150 kHz | +12 dB when closed | Cable or seam couples |
| Near-field probe on seam | Field map | Hot spot at seam | Bond the seam |
| Current probe on cable | Cable current | 8 dB higher after filter | Filter bypassed |
| Filter input vs output | Voltage | 20 dB bench, 8 dB installed | Layout inductance |
| Reset line during ESD | Scope | 2 V transient | Reset line susceptible |
| Supply rail during EFT | Scope | 1.5 V dip | Hold-up insufficient |
| Error log | Firmware | CRC errors during RF | Cable coupling |

The matrix separates the path from the victim. A measurement that does not distinguish the two will lead to a fix that treats the symptom.

### 6.6 Diagnostic 2: the immunity test resets the processor

The device passes the emissions test but resets during the 8 kV ESD test. The reset is caused by a discharge to the enclosure that couples into the reset line.

The diagnostic is to monitor the reset line during the ESD test. A transient on the reset line confirms the coupling. A second diagnostic is to add a capacitor to the reset line and repeat the test.

The design flaw is the reset line's susceptibility, not the ESD generator. The fix is a filter and a pull-up on the reset line, and a firmware check that logs the reset cause.

### 6.7a Environment comparison

| Parameter | Professional | Home | Difference |
|---|---|---|---|
| Radiated RF immunity | 3 V/m | 10 V/m | +10.46 dB |
| Conducted RF immunity | 3 V RMS | 3 V RMS, 6 V ISM | +6 dB in ISM |
| ESD contact | ±8 kV | ±8 kV | 0 dB |
| ESD air | ±15 kV | ±15 kV | 0 dB |
| Magnetic field | 30 A/m | 30 A/m | 0 dB |
| Emissions class | Class A allowed | Class B | Stricter |
| Proximity field | 65 A/m at 134 kHz | 65 A/m plus 8 A/m at 30 kHz | Additional test |

The radiated immunity change is the largest. A device with 6 dB of margin at 3 V/m has a −4.46 dB margin at 10 V/m, which is a failure. The fix must add at least 5 dB, and preferably 10 dB for production variation.

### 6.7 Vary one assumption: the device moves to the home environment

The professional version meets 3 V/m radiated RF. The home version must meet 10 V/m, which is 10.5 dB harsher. The device's analogue front end, which had a 6 dB margin at 3 V/m, now has a −4.5 dB margin and fails.

The fix is a shield, a filter, or a lower-impedance sensor. The change adds cost and size. The lesson is that the environment is a design input, not a label.

### 6.7b What the home transition costs

| Fix option | Added attenuation | Cost | Size | Verdict |
|---|---:|---|---|---|
| Shielded enclosure | 15–20 dB | High | None | Effective, heavy |
| Filter on sensor cable | 10–15 dB | Low | Small | Preferred |
| Lower-impedance sensor | 6–10 dB | Medium | Small | Requires redesign |
| Firmware averaging | 3–6 dB | Very low | None | Insufficient alone |
| Orientation restriction | 0 dB | None | None | Not acceptable |

The preferred fix is the filter on the sensor cable, because it addresses the coupling path without changing the sensor or the enclosure. The firmware averaging is a useful supplement but not a substitute: it reduces random noise, not a coherent interference.

### 6.8 What the example demonstrates

The arithmetic closes: 5.03 kHz corner, 59 dB ideal attenuation, 12 dB measured, 60 dBµV after the filter, a 6 dB margin against 66 dBµV, and a 10.46 dB increase when the environment changes from professional to home. The two diagnostics reveal different flaws: one is a layout path, the other is a reset line.

The general method is to identify the source, the path, and the victim, design the filter and the layout, pre-test, fix, and verify. A filter that is correct on the bench and wrong in the enclosure is a layout problem.

## 7. Code, query, or calculation

The program below reproduces the worked example. It uses only the standard library, and it can be run as written. The three parts are the CISPR 11 limit interpolation, the filter and margin calculation, and the test-plan budget.

```python
import math

def limit_qp(f_hz):
    # CISPR 11 group 1 class B, a.c. mains port, quasi-peak limit.
    f_mhz = f_hz / 1e6
    if 0.15 <= f_mhz < 0.5:
        frac = math.log10(f_mhz / 0.15) / math.log10(0.5 / 0.15)
        return 66.0 - 10.0 * frac
    if 0.5 <= f_mhz < 5.0:
        return 56.0
    if 5.0 <= f_mhz <= 30.0:
        return 60.0
    raise ValueError("outside 150 kHz to 30 MHz")

L, C = 100e-6, 10e-6
fc = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
print(f"filter corner: {fc:.2f} Hz")
print(f"ideal attenuation at 150 kHz: {40*math.log10(150e3/fc):.1f} dB")
print(f"ideal attenuation at 300 kHz: {40*math.log10(300e3/fc):.1f} dB")

print("\nfreq      limit   before   after   margin")
rows = [(150e3, 72.0, 12.0), (300e3, 64.0, 16.0),
        (500e3, 52.0, 20.0), (1e6, 44.0, 22.0)]
for f, emitted, att in rows:
    lim = limit_qp(f)
    after = emitted - att
    print(f"{f/1e3:6.0f} kHz {lim:6.1f} {emitted:7.1f} {after:7.1f} {lim-after:+7.1f}")

def dbuv_to_uv(db):
    return 10.0 ** (db / 20.0)

print(f"\n66 dBuV = {dbuv_to_uv(66):.1f} uV")
print(f"60 dBuV = {dbuv_to_uv(60):.1f} uV")
print(f"72 dBuV = {dbuv_to_uv(72):.1f} uV")

print(f"\nradiated 3 to 10 V/m: {20*math.log10(10/3):.2f} dB")
print(f"choke 4.7 mH at 150 kHz: {2*math.pi*150e3*4.7e-3:.0f} ohm")
print(f"ESD 8 kV into 330 ohm: {8000/330:.1f} A")

budget = [("pre-compliance emissions", 40, 150),
          ("pre-compliance immunity", 24, 150),
          ("accredited emissions", 24, 250),
          ("accredited immunity", 32, 250),
          ("setup and documentation", 16, 200),
          ("re-test allowance", 28, 250)]
total_h = sum(h for _, h, _ in budget)
total_c = sum(h * r for _, h, r in budget)
print(f"\ntest plan: {total_h} h, ${total_c}")
for name, h, r in budget:
    print(f"  {name:30s} {h:3d} h x ${r:3d} = ${h*r:6d}")
```

The output is:

```text
filter corner: 5032.92 Hz
ideal attenuation at 150 kHz: 59.0 dB
ideal attenuation at 300 kHz: 71.0 dB

freq      limit   before   after   margin
   150 kHz   66.0    72.0    60.0    +6.0
   300 kHz   60.2    64.0    48.0   +12.2
   500 kHz   56.0    52.0    32.0   +24.0
  1000 kHz   56.0    44.0    22.0   +34.0

66 dBuV = 1995.3 uV
60 dBuV = 1000.0 uV
72 dBuV = 3981.1 uV

radiated 3 to 10 V/m: 10.46 dB
choke 4.7 mH at 150 kHz: 4430 ohm
ESD 8 kV into 330 ohm: 24.2 A

test plan: 164 h, $33800
  pre-compliance emissions        40 h x $150 = $  6000
  pre-compliance immunity         24 h x $150 = $  3600
  accredited emissions            24 h x $250 = $  6000
  accredited immunity             32 h x $250 = $  8000
  setup and documentation         16 h x $200 = $  3200
  re-test allowance               28 h x $250 = $  7000
```

Three checks on the output:

1. The limit interpolation at 300 kHz gives 60.2 dBµV. The linear-in-log-frequency segment from 66 dBµV at 150 kHz to 56 dBµV at 500 kHz has a slope of \(10/\log_{10}(500/150) = 19.1\) dB per decade. At 300 kHz, one octave above 150 kHz, the limit is 66 − 5.76 = 60.24 dBµV, which the program prints as 60.2.
2. The 6 dB margin at 150 kHz is the difference between 66 dBµV and 60 dBµV, not a ratio of the measured values. A 6 dB margin means the emission is half the limit in voltage terms, or one quarter in power terms.
3. The budget total is 164 hours and $33,800. The re-test allowance is 28 hours, not a contingency percentage; it is a scheduled activity with its own chamber time.

The program does not model the filter's parasitics, the layout's inductance, or the chamber's uncertainty. Those are measurement facts, and the program uses their measured result (12 dB at 150 kHz) as an input. A model that predicts the filter's attenuation without the layout is a model that will be wrong at the first prototype.

The limit function is deliberately explicit about its range. A call outside 150 kHz to 30 MHz raises an error rather than extrapolating, because the standard's limits change at the band edges and an extrapolated value would be a fabricated number.


## 8. Data, standards, and regulatory context

### 8.1 The IEC 60601-1 architecture

| Layer | Standard | Edition and date | What it constrains |
|---|---|---|---|
| General | IEC 60601-1 | Ed. 3.2 (2005 + A1:2012 + A2:2020) | Basic safety and essential performance |
| Collateral | IEC 60601-1-2 | Ed. 4.1 (2014 + A1:2020) | Electromagnetic disturbances |
| Collateral | IEC 60601-1-8 | Ed. 2.2 | Alarm systems |
| Collateral | IEC 60601-1-11 | Ed. 2.1 | Home healthcare environment |
| Collateral | IEC 60601-1-6 | Ed. 3.2 | Usability engineering |
| Particular | IEC 60601-2-25 | Ed. 3.0 | Electrocardiographs |
| Particular | IEC 60601-2-27 | Ed. 3.0 | ECG monitoring equipment |
| Particular | IEC 80601-2-77 | Ed. 1.0 | Robotically assisted surgical equipment |
| Particular | IEC 80601-2-78 | Ed. 1.0 | Rehabilitation robots |
| Emissions | CISPR 11 | Ed. 6.2 (2015 + A1:2016 + A2:2019) | Industrial, scientific, and medical RF |
| Immunity methods | IEC 61000-4 series | Various | Test methods |

The edition matters. A test report against IEC 60601-1-2:2007 (Ed. 3) is not evidence against Ed. 4.1: the immunity levels, the environment categories, and the proximity-field tests changed. A regulatory submission must cite the edition that the recognised consensus standard list currently accepts.

### 8.2 What IEC 60601-1-2 Ed. 4.1 changed

The fourth edition reorganised the requirements around ports and environments. The changes that matter to a hardware team are:

1. The environment is explicitly professional healthcare or home healthcare, and the home levels are harsher for radiated RF.
2. The proximity-field tests for RF wireless equipment were added in Table 9, with band-specific levels up to 28 V/m.
3. Amendment 1 added proximity magnetic field tests in Table 11, applying to home healthcare equipment, at 8 A/m at 30 kHz, 65 A/m at 134.2 kHz, and 7.5 A/m at 13.56 MHz.
4. The cable exemption for SIP/SOP ports was tightened from 3 m to 1 m, which brings more cables into the test.
5. The risk-management annex was rewritten to align with ISO 14971:2019.
6. Conducted emissions and voltage dips are tested at minimum and maximum rated voltage where the range exceeds 25%.

The proximity magnetic field test is a common surprise. A device with a magnetometer, a Hall sensor, or a magnetic actuator can fail at 134.2 kHz even though it passed the 50/60 Hz test at 30 A/m.

### 8.3 CISPR 11 limits

CISPR 11 Group 1 Class B, a.c. mains port, quasi-peak and average limits:

| Frequency band | Quasi-peak | Average |
|---|---:|---:|
| 150 kHz to 500 kHz | 66 dBµV decreasing to 56 dBµV | 56 dBµV decreasing to 46 dBµV |
| 500 kHz to 5 MHz | 56 dBµV | 46 dBµV |
| 5 MHz to 30 MHz | 60 dBµV | 50 dBµV |

CISPR 11 Group 1 Class A, a.c. mains port, for comparison:

| Frequency band | Quasi-peak | Average |
|---|---:|---:|
| 150 kHz to 500 kHz | 79 dBµV | 66 dBµV |
| 500 kHz to 30 MHz | 73 dBµV | 60 dBµV |

The Class A limits are 13 to 16 dB higher than Class B. A device that is designed to Class A and then marketed for home use will fail by roughly that margin. The class is a property of the intended environment, and it cannot be changed by labelling alone.

Radiated emissions limits are specified at 10 m for Class B and at 30 m for Class A in the standard's test-site tables, although measurements at 3 m are permitted with a conversion. The conversion is not a free-space inversion; it depends on the site and the frequency, and the test report must state the method.

### 8.4 The immunity test levels

IEC 60601-1-2 Ed. 4.1 Table 4 and Table 5, summarised:

| Port | Test | Professional | Home |
|---|---|---|---|
| Enclosure | ESD contact | ±8 kV | ±8 kV |
| Enclosure | ESD air | ±15 kV | ±15 kV |
| Enclosure | Radiated RF, 80 MHz–2.7 GHz | 3 V/m | 10 V/m |
| Enclosure | Power-frequency magnetic field | 30 A/m | 30 A/m |
| Enclosure | Proximity magnetic field | Not required | 8/65/7.5 A/m |
| a.c. mains | Conducted RF, 150 kHz–80 MHz | 3 V RMS | 3 V RMS, 6 V ISM |
| a.c. mains | EFT | ±2 kV | ±2 kV |
| a.c. mains | Surge | ±0.5/±1 kV line-line, ±0.5/±1/±2 kV line-earth | Same |
| a.c. mains | Dips and interruptions | 0%/70% for 25 cycles, etc. | Same |
| SIP/SOP | Conducted RF | 3 V RMS | 3 V RMS |

The table is a summary, not a substitute for the standard. The modulation (80% AM at 1 kHz), the sweep rate, the dwell time, and the calibration procedure are part of the test practice and affect the result.

### 8.4a The IEC 61000-4 methods

Each immunity requirement is demonstrated by a method in the IEC 61000-4 series. The method defines the generator, the coupling, the calibration, and the performance check.

| Method | Disturbance | Coupling | Key parameter |
|---|---|---|---|
| IEC 61000-4-2 | ESD | Contact or air | ±8 kV contact, ±15 kV air |
| IEC 61000-4-3 | Radiated RF | Antenna, field | 3 or 10 V/m |
| IEC 61000-4-4 | EFT | Capacitive clamp or CDN | ±2 kV, 5/50 ns |
| IEC 61000-4-5 | Surge | CDN, 1.2/50 µs | ±1 kV, ±2 kV |
| IEC 61000-4-6 | Conducted RF | CDN or clamp | 3 V RMS, 150 kHz–80 MHz |
| IEC 61000-4-8 | Power-frequency magnetic field | Induction coil | 30 A/m, 50/60 Hz |
| IEC 61000-4-11 | Dips and interruptions | Programmable source | 0/40/70% residual |
| IEC 61000-4-39 | Proximity magnetic field | Loop antenna | 8/65/7.5 A/m |

The method and the requirement are separate. IEC 60601-1-2 sets the level and the performance criterion; IEC 61000-4-x sets the generator and the coupling. A laboratory that uses the wrong coupling has not demonstrated the requirement, even if the report cites the right standard.

### 8.5 Clause requirement versus test practice, in detail

A clause requirement is a statement of behaviour or property. A test practice is a procedure that produces evidence about that property. They are not interchangeable.

| Example | Clause requirement | Test practice | Gap |
|---|---|---|---|
| Radiated immunity | Device meets its performance criterion at 3 V/m | 61000-4-3 sweep, 1% steps, 3 s dwell | Dwell time may miss a transient |
| ESD | Device meets its criterion at ±8 kV | 61000-4-2, 10 discharges per polarity | Discharge points may miss a seam |
| Conducted emissions | Disturbance below the limit | CISPR 11 LISN, quasi-peak | Cable routing may differ from use |
| Leakage current | Current below the limit | 60601-1 Figure 13 network | Measurement bandwidth matters |

A device can pass the test practice and fail the requirement in a real environment. The reverse is also possible: a device can fail a poorly configured test practice while meeting the requirement. The manufacturer's obligation is the requirement; the test report is evidence, not the obligation itself.

### 8.6 FDA and EU context

The FDA's final guidance "Electromagnetic Compatibility (EMC) of Medical Devices" was issued on 6 June 2022 and superseded the 2016 guidance. It lists the content of an EMC section in a premarket submission: device characteristics, intended environment, risk assessment, consensus standards, essential performance, quantitative pass/fail criteria, configuration tested, results, deviations, and labelling. It also names common emitters that the risk assessment should consider: RFID, electronic article surveillance, near-field communication, wireless power transfer, 5G, electrocautery, MRI, and diathermy.

The guidance does not create new test levels. It specifies what evidence the submission must contain and how the manufacturer should justify the choice of standards. A submission that says "tested to IEC 60601-1-2" without the performance criteria and the configuration is incomplete under the guidance.

In the European Union, the MDR requires compliance with the general safety and performance requirements, and the harmonised standards give presumption of conformity. The EMC standard is harmonised, but the manufacturer must still justify the edition and the environment. The MDR's clinical evaluation and post-market surveillance obligations also apply to EMC-related incidents, because an interference-induced malfunction is a safety event.

### 8.6a Recognised consensus standards

A regulatory submission cites consensus standards that the regulator recognises. The recognition is edition-specific and can lag the publication of a new edition. A manufacturer that tests to a new edition before it is recognised may need to re-test, and one that tests to an old edition after recognition is withdrawn may have the evidence rejected.

The practical rule is to confirm the recognised edition at the start of the project and again before the test. The list is published by the regulator and changes on a known cycle. A standards matrix without a date is a matrix that may be out of date.

### 8.7 Risk management and EMC

ISO 14971:2019 requires the manufacturer to identify hazards, estimate and evaluate risk, control risk, and monitor the effectiveness of the controls. For EMC, the hazard is a malfunction caused by an electromagnetic disturbance, and the harm is the clinical consequence.

A risk file that treats EMC as a pass/fail test without a hazard analysis is incomplete. The analysis must ask what happens if the device resets, if the sensor saturates, if the display freezes, or if the alarm is delayed. The performance criterion in the EMC test is derived from that analysis, not chosen for convenience.

The link between the risk file and the test plan is the performance criterion. A criterion of "no visible malfunction" is not derived from a hazard analysis; a criterion of "SpO2 error remains within ±4% and the alarm remains functional" is.

### 8.8 Data quality of an EMC report

An EMC report is evidence only if it is reproducible. The report should contain:

| Field | Why it matters |
|---|---|
| Standard and edition | Levels change between editions |
| Laboratory accreditation | ISO 17025 scope |
| Test date | Configuration control |
| Device serial number | Traceability |
| Software version | Firmware affects emissions |
| Cable list and routing | Coupling path |
| Operating mode | Worst case |
| Performance criterion | Pass/fail definition |
| Measurement uncertainty | Margin interpretation |
| Modifications during test | Validity of result |

A report that omits the software version cannot be tied to the device that was submitted. A report that omits the cable routing cannot be reproduced by a second laboratory.

### 8.9 Report acceptance checklist

| Check | Accept | Reject |
|---|---|---|
| Standard edition | Current recognised edition | Superseded edition |
| Laboratory scope | Covers the test | Scope gap |
| Device identity | Serial and software version | Unidentified unit |
| Configuration | Cables and modes listed | Not stated |
| Performance criterion | Quantitative | "No malfunction" |
| Uncertainty | Stated and below margin | Not stated |
| Modifications | Recorded | Undisclosed |

The checklist is applied before the report is filed, not after the submission is rejected. A report that fails one row is not necessarily invalid, but the gap must be closed with a justification or a re-test.

## 9. Industry practice and case evidence

### 9.1 The design-gate model

Mature medical hardware organisations place EMC gates at four points:

1. **Architecture gate.** Standards matrix, environment, emissions class, and the supply topology.
2. **Schematic gate.** Filters, protection, grounding, isolation, and the cable list.
3. **Layout gate.** Stack-up, return paths, filter placement, and enclosure bonding.
4. **Pre-compliance gate.** A full scan against the limits with a margin policy.

A programme that skips the first three gates discovers the same issues at the fourth, when the cost of change is highest. The gate model is a cost-control mechanism, not a paperwork exercise.

### 9.2 The margin policy

A common industrial policy is a 6 dB margin on emissions and 6 dB on immunity, with 10 dB for a first-of-kind design. The margin covers production variation, measurement uncertainty, and the difference between the pre-compliance and accredited setups.

The policy must be written down. A team that accepts a 1 dB margin because the test passed has a device that may fail on a different unit, at a different laboratory, or after a component change. The margin is a design parameter, not a test outcome.

### 9.3 Common findings

The recurring findings in medical device EMC programmes are consistent across product types:

- A switching supply's low-frequency harmonics exceed the conducted limit, and the input filter is either absent or placed too far from the connector.
- A cable shield is bonded at one end only, or bonded through a long pigtail, which turns the shield into an antenna.
- The enclosure has an uncoated seam or an aperture that is a fraction of a wavelength at the highest test frequency.
- A reset or interrupt line is routed near the enclosure edge without a filter.
- The firmware has no watchdog, so an upset requires a power cycle.
- The device is tested in a favourable orientation and fails in the intended use.
- The home-environment levels are applied to the professional version without re-testing.
- The Y-capacitor value is chosen for EMC and then reduced for leakage, without re-checking the emissions.

Each finding is a design issue, not a test issue. The test reveals it; the design causes it.

### 9.3a Review findings and their cost

| Finding | Stage found | Typical fix | Schedule effect |
|---|---|---|---|
| Missing standards | Architecture | Add standard and test | Days |
| Wrong environment | Architecture | Re-test at home levels | Weeks |
| Filter placement | Layout | Move filter, revise board | Weeks |
| Shield bond | Layout | Add gasket, revise enclosure | Weeks |
| Reset line | Pre-compliance | Add RC, revise board | Days to weeks |
| Firmware watchdog | Pre-compliance | Firmware change | Days |
| Cable routing | Accredited | Revise cable, re-test | Weeks |
| Edition outdated | Submission | Gap assessment, re-test | Months |

The cost rises by roughly an order of magnitude at each stage. The table is the argument for the design gates: the same finding is cheap at architecture and expensive at submission.

### 9.4 Case evidence: electrosurgery and monitoring

Electrosurgical units (ESUs) are among the most demanding emitters in a hospital. They deliver RF energy, typically in the 300 kHz to 5 MHz range, with harmonics extending into the tens of MHz. A monitor or a sensor cable near an active ESU can pick up enough energy to saturate an analogue front end.

The industry response has three parts: shielding and filtering of the patient cable, a front-end design with enough common-mode rejection and headroom, and a firmware detector that flags a saturated or implausible signal rather than displaying it as valid. The third part is often missing. A device that displays a saturated value as a measurement is more dangerous than one that alarms.

IEC 60601-1-2's immunity tests do not reproduce a full ESU environment. The standard's radiated and conducted tests are a baseline; the risk analysis must consider the actual clinical environment, and the manufacturer may need additional testing or a use restriction.

### 9.5 Case evidence: wireless emitters and the home

The home environment contains emitters that the professional environment does not: mobile phones, Wi-Fi routers, smart meters, wireless power chargers, and RFID readers. IEC 60601-1-2 Ed. 4.1 raised the home radiated immunity level to 10 V/m and added the proximity-field tables for exactly this reason.

A device designed to the professional 3 V/m level has roughly 10.5 dB less margin than the home requirement. The failure mode is often not a reset but a gradual sensor error: an analogue front end rectifies the RF envelope, the baseline shifts, and the displayed value drifts. The device passes a functional check before and after the test, but the value during the test is wrong.

This is why the performance criterion must be quantitative and monitored during the test. A criterion of "no reset" misses the drift; a criterion of "heart rate within 5 bpm throughout" catches it.

### 9.6 Case evidence: the accredited test as a schedule risk

An accredited EMC test is typically booked weeks in advance and costs a fixed daily rate. A failure found on the first day consumes the booked time and requires a new booking after the fix. The schedule consequence is often larger than the test cost.

The industrial mitigation is a pre-compliance programme with the same limits and a similar setup, plus a design review that looks for the recurring findings before the chamber. The pre-compliance bench does not need to be accredited; it needs to be sensitive enough to show a 6 dB margin.

### 9.6a Questions to ask a test laboratory

| Question | Why it matters |
|---|---|
| Is your scope accredited for this standard and edition? | Report acceptance |
| What is your measurement uncertainty? | Margin interpretation |
| Do you test at 3 m or 10 m? | Limit comparison |
| How do you document cable routing? | Reproducibility |
| Do you record the software version? | Configuration control |
| What is the chamber time per test? | Budget and schedule |
| What is the re-test policy? | Cost of failure |
| Can we observe the test? | Learning and diagnosis |

The answers determine whether the report will be usable and whether the team can learn from the test. A laboratory that does not allow observation costs more in the long run, because the team cannot see the failure mode.

### 9.7 What the evidence supports and what it does not

The evidence supports three claims: EMC failures are usually design issues found late; a pre-compliance programme reduces the number and cost of accredited re-tests; and the home environment is materially harsher than the professional one. The evidence does not support a specific failure rate for a device type, because the published data are not collected with a common method. Any statement of the form "X% of devices fail EMC" should be treated as an industry estimate unless the source and the sample are stated.

## 10. Failure modes and red flags

| # | Failure mode | Mechanism | Symptom | Detection | Mitigation |
|---:|---|---|---|---|---|
| 1 | Conducted emission exceeds limit | Switching supply harmonics | Low-frequency peak | LISN scan | Input filter at connector |
| 2 | Radiated emission exceeds limit | Clock or cable | Broadband peak | Chamber or near-field | Spread spectrum, shield |
| 3 | ESD reset | Discharge into reset line | Reset during test | Scope on reset | Filter, firmware log |
| 4 | ESD latch-up | Discharge into I/O | Device hangs | Current monitor | TVS, series resistor |
| 5 | Radiated immunity drift | RF rectified in front end | Sensor bias shift | Monitor during sweep | Shield, filter, headroom |
| 6 | Conducted immunity corruption | Cable coupling | CRC errors | Error log | Choke, shielded cable |
| 7 | EFT reset | Fast transient on supply | Reset | Rail monitor | TVS, filter |
| 8 | Surge damage | Slow transient | Component failure | Post-test check | MOV, TVS, spacing |
| 9 | Dip reset | Supply below threshold | Reset | Rail monitor | Hold-up capacitor |
| 10 | Magnetic field shift | Loop pickup | Sensor offset | Probe test | Orient, shield, twist |
| 11 | Proximity field failure | 134 kHz coupling | Magnetometer error | 61000-4-39 | Filter, distance, risk |
| 12 | Filter resonance | LC with parasitics | New peak | Wide scan | Damping, ferrite |
| 13 | Shield pigtail | High-impedance bond | Emission rise | Current probe | 360-degree bond |
| 14 | Y-capacitor leakage | Excess capacitance | Leakage failure | Leakage test | Reduce, re-check EMC |
| 15 | Firmware update during EMC | Flash write upset | Corrupt image | Version check | A/B image, CRC |
| 16 | Calibration drift after ESD | EEPROM upset | Bias error | Post-test calibration | CRC, redundant copy |
| 17 | Auxiliary equipment changes emission | Extra monitor or pump | New peak | Full configuration test | Test the system |
| 18 | Test at one voltage | Wide-range supply | Different harmonic | Min and max voltage | Test both |
| 19 | Battery charging path | Charger switching | Conducted peak | Charge-mode scan | Filter charger |
| 20 | Ultrasound or motor drive | Burst emission | Broadband | Near-field probe | Shield, filter |


The first six rows account for most findings. Rows 13 to 20 are the ones more likely to pass a single-configuration test and fail in the field, because they depend on the configuration, the firmware, or the environment.

### 10.1 Red flags in a design review

- The filter is drawn on the schematic but its placement is not fixed at layout.
- The cable shield is bonded through a wire longer than 10 mm.
- The enclosure has a painted seam with no masking.
- The reset line has no series resistor and no capacitor.
- The firmware has no watchdog and no error counters.
- The EMC plan lists a single operating mode.
- The test report has no software version.
- The margin is under 3 dB and the team calls it a pass.
- The home label is planned but the home tests are not budgeted.
- The Y-capacitor was chosen after the leakage test.

Each red flag is observable before the chamber. A review that checks them costs hours; a failure that misses them costs weeks.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | Effect if wrong |
|---:|---|---|---|---|
| 1 | The device is used in the stated environment | Selects the test levels | Home use of a professional device | Under-tested by 10.5 dB |
| 2 | The supply's switching frequency is fixed | Predicts harmonics | Spread spectrum or burst mode | Peaks move, filter mistuned |
| 3 | The filter's measured attenuation applies in production | Predicts margin | Component tolerance, layout | Margin shrinks |
| 4 | The cable routing in the test matches use | Represents coupling | Different cable, longer run | Emission or immunity changes |
| 5 | The enclosure is the final one | Seams and apertures matter | Coating, label, screw | New resonance |
| 6 | The firmware version is the released one | Emissions depend on clocks | Debug build, update | Test not representative |
| 7 | The performance criterion is clinically derived | Defines pass/fail | Criterion chosen for convenience | Test passes, risk remains |
| 8 | The laboratory's accreditation covers the test | Report is accepted | Scope gap | Report rejected |
| 9 | The test levels are the current edition | Evidence is current | Edition superseded | Submission gap |
| 10 | The measurement uncertainty is smaller than the margin | Margin is real | Margin under 3 dB | False pass |
| 11 | The device is tested in the worst-case mode | Covers use | Missing mode | Field failure |
| 12 | The production unit matches the tested unit | Change control | Silent substitution | Unknown EMC status |

The assumptions are not equally strong. The first, sixth, and eleventh are the ones most often wrong in practice, because the intended use, the firmware, and the operating modes change during development while the EMC plan is written once.

### 11.1 How the assumptions interact

The assumptions are not independent. If the environment is wrong (1), the test levels are wrong, and the margin calculation (3) is meaningless. If the firmware version is wrong (6), the emissions result may not represent the released product, and the performance criterion (7) may not have been exercised. A verification plan should state which assumptions are checked and how, rather than listing them as caveats.

The strongest check is a configuration record that ties the tested unit to the released unit: serial number, software version, cable list, and enclosure revision. Without that record, the assumptions are beliefs rather than facts.

## 12. Exercises

### Exercise 1. Standards matrix

**Task.** Take a home-use device with a Bluetooth radio and a patient-connected pressure sensor. Build a table with the general, collateral, particular, and emissions standards that apply. For each, state the edition and the clause family that drives a test.

**Deliverable.** A one-page matrix with columns: standard, edition, clause family, requirement, evidence.

**Criteria.** All four layers present; edition stated; the particular standard is correct for the device type; the emissions class is justified by the intended environment.

**Time.** 45 minutes.

### Exercise 2. Limit interpolation

**Task.** Compute the CISPR 11 Group 1 Class B quasi-peak limit at 200 kHz, 400 kHz, and 700 kHz. Show the interpolation formula and the arithmetic.

**Deliverable.** Three values with the calculation.

**Criteria.** The 150 kHz to 500 kHz segment is interpolated logarithmically; the 500 kHz to 5 MHz segment is constant; the units are dBµV.

**Time.** 20 minutes.

### Exercise 3. Filter design

**Task.** A 24 V supply switches at 80 kHz and produces 78 dBµV at 160 kHz. The limit is 66 dBµV. Design a two-stage LC filter that gives at least 12 dB of measured margin, and state the corner frequency and the ideal attenuation. Explain why the measured attenuation will be lower than the ideal.

**Deliverable.** Component values, corner frequency, ideal attenuation, measured estimate, and a paragraph on parasitics.

**Criteria.** The corner is below the lowest frequency of concern; the attenuation exceeds the requirement with margin; the explanation names layout inductance and capacitor ESR.

**Time.** 45 minutes.

### Exercise 4. Test-plan budget

**Task.** Build a budget for a device that needs conducted and radiated emissions, ESD, radiated RF, conducted RF, EFT, surge, and dips. State the hours, the rate, and the total. Include a re-test allowance and justify it.

**Deliverable.** A table with line items and a total.

**Criteria.** Every test is costed; the re-test allowance is a line item; the total is arithmetically correct.

**Time.** 40 minutes.

### Exercise 5. Failure triage

**Task.** A device passes emissions with the enclosure open and fails with it closed. List four possible causes, the measurement that distinguishes them, and the fix for each.

**Deliverable.** A four-row table: cause, measurement, expected observation, fix.

**Criteria.** The causes cover seam, aperture, cable, and internal resonance; the measurements are specific; the fixes address the path rather than the symptom.

**Time.** 30 minutes.

### Exercise 6. Requirement versus test practice

**Task.** Choose one clause from IEC 60601-1-2 (for example, radiated RF immunity). Write the requirement, the test practice, and two ways a device could pass the practice but fail the requirement.

**Deliverable.** A short memo with the three parts.

**Criteria.** The requirement is stated as behaviour; the practice is stated as a procedure; the two gaps are plausible and specific.

**Time.** 30 minutes.

### Exercise 7. Home-environment transition

**Task.** A professional monitor has a 4 dB margin at 3 V/m. Compute the margin at 10 V/m and state whether it passes. Propose three fixes with an estimated attenuation and a cost, and choose one.

**Deliverable.** The arithmetic, a three-row fix table, and a recommendation.

**Criteria.** The dB difference is 10.46 dB; the margin is negative; the fixes are technically distinct; the recommendation is justified.

**Time.** 30 minutes.

### Exercise 8. Risk-link audit

**Task.** Take the performance criteria in a sample EMC test plan and trace each one to a hazard in a risk file. Identify the criteria that have no hazard behind them.

**Deliverable.** A traceability table and a list of orphan criteria.

**Criteria.** Every criterion has a hazard or is flagged; the orphan list is specific; the recommendation is actionable.

**Time.** 45 minutes.

## 13. Self-check answers

**1. What is the difference between a general, collateral, and particular standard?**

The general standard applies to all medical electrical equipment and covers basic safety and essential performance. A collateral standard applies to one aspect across all device types, such as EMC or alarms. A particular standard applies to one device type and takes precedence where it conflicts with the general or collateral standard.

**2. Why must a home-use device meet a higher radiated immunity level?**

The home environment contains consumer emitters at close range, including phones, routers, and wireless chargers. IEC 60601-1-2 Ed. 4.1 sets 10 V/m for home healthcare against 3 V/m for professional healthcare, a difference of 10.46 dB.

**3. What is the CISPR 11 Group 1 Class B quasi-peak limit at 150 kHz?**

66 dBµV. It decreases linearly with the logarithm of frequency to 56 dBµV at 500 kHz, then stays at 56 dBµV to 5 MHz, and rises to 60 dBµV from 5 MHz to 30 MHz.

**4. Why is a 3 dB margin a weak pass?**

The combined measurement uncertainty of the laboratory and the pre-compliance bench can exceed 3 dB at some frequencies, and production variation adds more. A 3 dB margin can disappear on a different unit or a different day. A 6 dB margin is the usual minimum policy.

**5. What is the difference between a clause requirement and a test practice?**

A clause states a required property or behaviour. A test practice is a procedure that produces evidence about it. Passing the practice does not prove the requirement in every environment, and failing a poorly configured practice does not prove non-compliance.

**6. What does a LISN do?**

A line impedance stabilization network provides a defined mains impedance to the device under test and a filtered port to the measuring receiver. It makes the conducted-emissions measurement reproducible across laboratories.

**7. Why can a filter work on the bench and fail in the enclosure?**

The cable between the filter and the connector, the enclosure seam, and the layout's parasitic inductance provide paths that bypass the filter. The filter attenuates the conducted path, but the radiated or common-mode path remains.

**8. What is the difference between pre-compliance and accredited testing?**

Pre-compliance is an in-house screening test used for design iteration. Accredited testing is performed by a laboratory whose scope covers the standard and produces the report used for regulatory evidence.

**9. Why is the performance criterion part of the test, not an afterthought?**

The test applies a stress and observes behaviour. Without a quantitative criterion, the observation is a judgement, and a gradual drift or a delayed alarm can be missed. The criterion is derived from the risk analysis and defines pass/fail.

**10. What changed in IEC 60601-1-2 Amendment 1?**

It added proximity magnetic field tests for home healthcare, tightened the SIP/SOP cable exemption to 1 m, aligned the risk-management annex with ISO 14971:2019, and required conducted emissions and dips at minimum and maximum rated voltage for wide-range supplies.

**11. Why is the Y-capacitor a design conflict?**

A larger Y-capacitor improves common-mode attenuation but increases the leakage current, which is limited by IEC 60601-1. The value must satisfy both requirements, and a change for one must be re-checked against the other.

**12. What makes an EMC report non-reproducible?**

A missing software version, an undocumented cable routing, an unstated operating mode, or an unrecorded modification. Any of these breaks the link between the report and the device configuration.

**13. Why does an ESD event reset a processor through the reset line?**

The discharge creates a fast transient field that couples into the reset trace. If the trace is long, unfiltered, and near the enclosure edge, the coupled voltage can cross the logic threshold. A series resistor, a capacitor, and a pull-up reduce the coupling.

**14. What is the risk of treating EMC as a final test?**

The design is fixed when the test is run, so a failure requires a board or enclosure change at the highest cost point. The schedule impact is usually larger than the test cost, because the chamber must be rebooked.

**15. Why are the proximity-field tests a common surprise?**

They were added in Amendment 1 and apply to home healthcare. A device with a magnetometer or a magnetic actuator can fail at 134.2 kHz or 13.56 MHz even though it passed the 50/60 Hz test. The test is easy to omit from a plan written against the previous edition.

## 14. Interview questions

**1. A device passes radiated emissions at 3 m but fails at 10 m. What do you check first?**

The measurement distance and the site correction. The limits are distance-dependent, and a 3 m result converted to 10 m is not a free-space inversion. Then check the cable routing and the antenna height, because the two setups may couple differently.

**2. How would you decide between a filter and a shield?**

Identify the path. If the emission leaves on a cable, a filter or a choke at the connector is usually cheaper. If it leaves through a seam or an aperture, a shield or a bond is needed. A filter cannot fix a radiated path through the enclosure.

**3. What is the first thing you ask a test laboratory?**

Whether its accreditation scope covers the standard and edition, and whether the report will state the software version, the cable list, and the performance criteria. A report that omits these is weak evidence.

**4. Why does the home environment matter for a device that is only used in clinics?**

Because the intended use may include the patient's home, and the label determines the environment. If the device is used in a home, even occasionally, the home levels apply. The classification is a clinical and regulatory decision, not a marketing one.

**5. How do you set a performance criterion?**

Start from the hazard analysis. Ask what clinical function must be preserved and what error is acceptable. State it quantitatively and monitor it during the test. A criterion that only says "no reset" misses drift and delayed alarms.

**6. What is the difference between a 6 dB margin and a 6 dB limit?**

A margin is the distance between the measured value and the limit. A limit is the absolute value in the standard. A 6 dB margin means the measurement is half the limit in voltage, not that the limit is 6 dB.

**7. Why is the switching supply the usual suspect in conducted emissions?**

It is the largest periodic source on the mains side, it has a rich harmonic spectrum, and its input filter is often an afterthought. The low-frequency harmonics are close to the 150 kHz limit, where the filter's attenuation is lowest.

**8. What would you do if a fix reduced the 150 kHz emission but created a peak at 400 kHz?**

The fix changed the resonance rather than removing the source. Check the filter's damping and the layout's parasitics. A ferrite or an RC damper can suppress the new peak, but the source and the path must still be addressed.

**9. How does firmware contribute to EMC?**

It can spread the clock to reduce emissions, detect communication errors, log resets, recover from upsets, and avoid unsafe behaviour during a disturbance. It cannot compensate for a hardware path that couples interference into the front end.

**10. What is the role of ISO 14971 in an EMC programme?**

It defines the hazard, the harm, and the risk controls that the performance criteria must reflect. EMC testing without a risk link produces criteria that are convenient rather than clinically meaningful.

**11. When would you re-test after a change?**

When the change affects the PCB, the enclosure, the cables, the firmware clocks, or the supply. A change to a label or a manual does not normally require re-testing, but a change to the cable list does, because the cable is part of the coupling path.

**12. What is the most common mistake in an EMC plan?**

Testing one operating mode in one orientation with a favourable cable routing. The plan must cover the worst case, and the worst case must be justified by the emission and immunity mechanisms, not by convenience.

## 15. Key terms

| Term | Meaning |
|---|---|
| EMC | Electromagnetic compatibility: the ability to function without causing or suffering interference |
| EMI | Electromagnetic interference: the disturbance itself |
| CISPR 11 | The emissions standard for industrial, scientific, and medical equipment |
| Group 1 | Equipment without intentional RF energy |
| Group 2 | Equipment with intentional RF energy |
| Class A | Limits for industrial and professional environments |
| Class B | Limits for residential and home environments |
| LISN | Line impedance stabilization network |
| Quasi-peak | A detector weighting repetitive emissions by their repetition rate |
| Immunity | The ability to withstand a disturbance while meeting a performance criterion |
| Performance criterion | The defined behaviour that must be preserved during an immunity test |
| ESD | Electrostatic discharge |
| EFT | Electrical fast transient |
| Surge | A slow, high-energy transient |
| Dip | A short reduction in supply voltage |
| Common-mode | A disturbance that appears equally on both conductors |
| Differential-mode | A disturbance that appears between two conductors |
| Y-capacitor | A capacitor from a conductor to earth, limited by leakage current |
| Transfer impedance | The voltage produced on a shield's interior per unit current on its exterior |
| Pre-compliance | In-house testing used for design iteration |
| Accredited test | Testing by a laboratory whose scope covers the standard |
| Essential performance | Performance whose loss or degradation creates unacceptable risk |

## 16. Sources

1. International Electrotechnical Commission. *IEC 60601-1-2:2014+AMD1:2020 CSV, Medical electrical equipment — Part 1-2: General requirements for basic safety and essential performance — Collateral standard: Electromagnetic disturbances — Requirements and tests.* IEC Webstore. https://webstore.iec.ch/en/publication/67554. Accessed 25 October 2026.
2. International Electrotechnical Commission. *IEC 60601-1-2:2014/AMD1:2020, Amendment 1.* Sample PDF. https://cdn.standards.iteh.ai/samples/100132/e7d4039bdb1f4559aa3c012ff789f5b6/IEC-60601-1-2-2014-AMD1-2020.pdf. Accessed 25 October 2026.
3. International Electrotechnical Commission. *CISPR 11:2015+AMD1:2016+AMD2:2019 CSV, Industrial, scientific and medical equipment — Radio-frequency disturbance characteristics — Limits and methods of measurement.* Sample PDF. https://cdn.standards.iteh.ai/samples/18863/afaa25d0ed1f4ad68be0e09e7bd9c8f9/CISPR-11-2015.pdf. Accessed 25 October 2026.
4. U.S. Food and Drug Administration. *Electromagnetic Compatibility (EMC) of Medical Devices.* Guidance for Industry and Food and Drug Administration Staff, final, 6 June 2022. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/electromagnetic-compatibility-emc-medical-devices. Accessed 25 October 2026.
5. U.S. Food and Drug Administration. *Electromagnetic Compatibility (EMC) of Medical Devices* (PDF). https://www.fda.gov/media/94758/download. Accessed 25 October 2026.
6. Federal Register. *Electromagnetic Compatibility of Medical Devices; Guidance for Industry and Food and Drug Administration Staff; Availability.* 87 FR 34275, 6 June 2022. https://www.federalregister.gov/documents/2022/06/06/2022-12099/electromagnetic-compatibility-of-medical-devices-guidance-for-industry-and-food-and-drug. Accessed 25 October 2026.
7. Philip McGaw. *EMC Charts and Graphs.* https://philipmcgaw.com/emc-charts-and-graphs/. Accessed 25 October 2026.
8. Cadence System Analysis. *What to Know About Conducted Emissions Limits.* https://resources.system-analysis.cadence.com/blog/msa2022-what-to-know-about-conducted-emissions-limits. Accessed 25 October 2026.
9. International Organization for Standardization. *ISO 14971:2019, Medical devices — Application of risk management to medical devices.* https://www.iso.org/standard/72704.html. Accessed 25 October 2026.
10. International Electrotechnical Commission. *IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes.* https://webstore.iec.ch/en/publication/22794. Accessed 25 October 2026.

The CISPR 11 limit values in Section 8 are taken from the standard's Group 1 Class B table as reported by sources 3, 7, and 8. The immunity levels in Section 8 are from source 1 and 2. The FDA submission content is from sources 4 to 6. The test-plan rates in Sections 5 and 6 are illustrative planning figures, not quotations from a laboratory; a real budget must use the laboratory's current rate card.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`12_surgical-and-rehab-robotics.md`](12_surgical-and-rehab-robotics.md)
- Next topic: [`14_reliability-and-sterilization.md`](14_reliability-and-sterilization.md)
- Survey chapter: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab directory: [`../../labs/`](../../labs/)

The previous topic covers registration, teleoperation, and rehabilitation control, which is where the safety envelope and the latency budget are introduced. The next topic covers reliability, accelerated life testing, biocompatibility, and sterilization, which is where the lifetime and the materials of the same device are analysed.

The survey chapter places EMC in the wider device lifecycle and lists the standards that a submission typically cites. The lab directory contains the measurement exercises that accompany this module; the EMC exercise uses a pre-compliance bench rather than an accredited chamber, because the chamber is not available in a teaching laboratory.
