# Reliability and Sterilization

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 14 of 14
> Estimated study time: 4 hours
> Prerequisites: 09, 13
> Version 2026.3

## 1. Why this topic exists

A medical device must remain safe and effective for its stated lifetime, and it must be sterile and biocompatible before it is used. These are not separate concerns. A sterilization cycle can degrade a polymer that was chosen for its mechanical properties, and a thermal cycle in the field can crack a solder joint that passed the accelerated life test. Reliability, materials, and sterilization are one design problem.

The industry problem is that reliability is often reduced to a single number, the mean time between failures (MTBF), which is then quoted without its assumptions. MTBF is meaningful only when the failure rate is constant, the failure mechanisms are known, and the population is defined. A quoted MTBF of 200,000 hours does not mean the device lasts 22.8 years; it means that, under the exponential model, about 4.3% of units fail in the first year.

The second problem is that accelerated life testing (ALT) can accelerate the wrong mechanism. A test at 85 °C and 85% relative humidity may produce a failure that never occurs in the field, while the field failure mode is a connector that the test never stressed. The acceleration model is a hypothesis, and the test is an experiment to test it, not a proof of it.

The third problem is that biocompatibility and sterilization are treated as paperwork. ISO 10993-1 requires a biological evaluation, not a fixed test list. The evaluation depends on the contact category and the duration, and the extractables and leachables work must be designed around the materials and the clinical exposure.

This topic covers the arithmetic of failure rates and MTBF, the Arrhenius and Coffin-Manson acceleration models and their assumptions, the ISO 10993-1 evaluation logic, and the three common sterilization methods. It works an ALT case that converts accelerated hours to field-equivalent years, and a sterilization dose case.

The unifying idea is that every reliability claim is conditional. The conditions are the use profile, the failure mechanism, the acceleration model, and the statistical confidence. A claim without conditions is a number without a meaning.

## 2. Learning outcomes

By the end of this topic you can:

- Compute failure rate, MTBF, FIT, and reliability for a series system.
- Explain the constant-failure-rate assumption and when it fails.
- Apply the Arrhenius model and compute an acceleration factor with stated activation energy.
- Apply the Coffin-Manson model and compute a thermal-cycle acceleration factor.
- Design an ALT with stress levels, sample size, duration, and failure criteria.
- Convert accelerated hours to field-equivalent years and state the confidence.
- Classify a device under ISO 10993-1 by contact nature and duration.
- Explain the extractables and leachables logic and the analytical evaluation threshold.
- Compare ethylene oxide, gamma, and moist heat sterilization by dose, cycle, and material impact.
- Compute a sterilization dose from bioburden, D-value, and the required sterility assurance level.
- Identify the failure modes that invalidate a reliability or sterilization claim.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A reliability claim has four parts: the population, the stress profile, the failure definition, and the time. A claim that omits any of them cannot be compared with another claim.

```text
   POPULATION ---> STRESS PROFILE ---> FAILURE DEFINITION ---> TIME
        |                  |                    |                 |
     units, lot      temperature,        what counts as      hours, cycles,
     screening       humidity, cycles    a failure           shelf life
        |                  |                    |                 |
        +------------------+--------------------+-----------------+
                                   |
                            ACCELERATION MODEL
                                   |
                            FIELD-EQUIVALENT TIME
```

The model converts accelerated time to field time. The conversion is valid only if the failure mechanism is the same at both stresses.

### 3.1 The bathtub curve

| Phase | Failure rate | Cause | Design response |
|---|---|---|---|
| Infant mortality | Decreasing | Manufacturing defects | Screening, burn-in |
| Useful life | Constant | Random overstress | Derating, redundancy |
| Wear-out | Increasing | Degradation | Lifetime test, replacement |

The exponential model applies to the useful-life phase. A device that is still in infant mortality or already in wear-out is not described by a constant failure rate, and an MTBF computed from such data is misleading.

### 3.2 Failure rate arithmetic

For a constant failure rate \(\lambda\):

\[
R(t) = e^{-\lambda t}, \qquad F(t) = 1 - e^{-\lambda t}, \qquad \text{MTBF} = \frac{1}{\lambda}
\]

The failure rate is often expressed in FIT (failures in time), where 1 FIT = \(10^{-9}\) failures per hour:

\[
\text{FIT} = \lambda \times 10^{9}
\]

A system of independent components in series has:

\[
\lambda_{\text{system}} = \sum_i \lambda_i
\]

The series model assumes that any component failure causes a system failure. A parallel or redundant path changes the arithmetic.

### 3.3 Acceleration models

**Arrhenius** for temperature-driven chemical or diffusion mechanisms:

\[
AF = \exp\left[\frac{E_a}{k}\left(\frac{1}{T_{\text{use}}} - \frac{1}{T_{\text{stress}}}\right)\right]
\]

where \(E_a\) is the activation energy in electron-volts, \(k = 8.617 \times 10^{-5}\) eV/K, and the temperatures are in kelvin.

**Coffin-Manson** for thermal-cycle fatigue:

\[
AF = \left(\frac{\Delta T_{\text{stress}}}{\Delta T_{\text{use}}}\right)^m
\]

where \(m\) is the Coffin-Manson exponent, typically 1.9 to 2.5 for solder joints, and \(\Delta T\) is the cycle excursion.

**Norris-Landzberg** adds a frequency and a maximum-temperature term for solder:

\[
AF = \left(\frac{\Delta T_{\text{stress}}}{\Delta T_{\text{use}}}\right)^m \left(\frac{f_{\text{use}}}{f_{\text{stress}}}\right)^n \exp\left[\frac{E_a}{k}\left(\frac{1}{T_{\max,\text{use}}} - \frac{1}{T_{\max,\text{stress}}}\right)\right]
\]

The extra terms matter when the stress test uses a different cycle frequency or a different peak temperature. A Coffin-Manson calculation that ignores them can over- or under-accelerate by a large factor.

### 3.4 The sterilization logic

A sterilization claim has four parts: the method, the dose or cycle, the bioburden, and the sterility assurance level (SAL). The SAL is the probability that a single item is non-sterile after the process; for medical devices it is usually \(10^{-6}\).

\[
\text{SAL} = N_0 \times 10^{-D/D_{\text{value}}}
\]

where \(N_0\) is the bioburden in colony-forming units, \(D\) is the delivered dose, and \(D_{\text{value}}\) is the dose that reduces the population by one logarithm. The same equation is written in log form:

\[
\log_{10} \text{SAL} = \log_{10} N_0 - \frac{D}{D_{\text{value}}}
\]

For a required SAL, the minimum dose is:

\[
D_{\min} = D_{\text{value}} \left(\log_{10} N_0 - \log_{10} \text{SAL}\right)
\]

The equation is simple; the difficulty is in the bioburden estimate, the D-value for the actual bioburden, and the validation that the process delivers the dose to the most difficult location.

### 3.5 Biocompatibility logic

ISO 10993-1 evaluates a device by two axes:

| Contact nature | Examples |
|---|---|
| Surface device | Skin, mucosa, intact skin |
| External communicating device | Blood path, tissue, dentin |
| Implant device | Bone, tissue, blood |

| Contact duration | Definition |
|---|---|
| Limited | Less than 24 hours |
| Prolonged | 24 hours to 30 days |
| Long-term | More than 30 days |

The category determines the biological endpoints. A surface device with limited contact needs cytotoxicity, sensitization, and irritation; an implant with long-term contact needs a much larger set, including chronic toxicity, carcinogenicity, and degradation.

The evaluation is risk-based. Testing is one source of evidence; chemical characterization, material history, and clinical experience can reduce the test burden, but the justification must be documented.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Reliability | Probability of performing a required function under stated conditions for a stated time | Confused with quality | Dimensionless |
| MTBF | Mean time between failures, the reciprocal of the failure rate under the exponential model | Read as lifetime | Hours |
| MTTF | Mean time to failure, for non-repairable items | Confused with MTBF | Hours |
| Failure rate | Failures per unit time | Confused with failure probability | 1/h |
| FIT | Failures in time, \(10^{-9}\) per hour | Confused with a count | FIT |
| Bathtub curve | Failure rate over life: infant, useful, wear-out | Assumed for all devices | None |
| Infant mortality | Early failures from defects | Confused with wear-out | None |
| Wear-out | Increasing failure rate from degradation | Confused with random failure | None |
| Derating | Operating a component below its rating | Confused with screening | % |
| Acceleration factor | Ratio of field life to test life | Confused with a test duration | Dimensionless |
| Activation energy | Energy barrier in the Arrhenius model | Treated as a universal constant | eV |
| Coffin-Manson exponent | Slope of the thermal-cycle fatigue curve | Treated as universal | Dimensionless |
| ALT | Accelerated life test | Confused with HALT | None |
| HALT | Highly accelerated life test, qualitative | Confused with ALT | None |
| Bioburden | Microbial population before sterilization | Confused with sterility | CFU |
| D-value | Dose or time for one log reduction | Confused with SAL | kGy or min |
| SAL | Sterility assurance level | Confused with a test result | Probability |
| EO | Ethylene oxide | Confused with ETO | None |
| Gamma | Ionizing radiation from cobalt-60 | Confused with electron beam | None |
| Autoclave | Moist heat sterilization | Confused with dry heat | None |
| F0 | Equivalent time at 121 °C | Confused with cycle time | Minutes |
| Extractables | Substances released by a material under exaggerated conditions | Confused with leachables | µg |
| Leachables | Substances released under clinical conditions | Confused with extractables | µg |
| AET | Analytical evaluation threshold | Confused with a limit | µg |
| ISO 10993-1 | Biological evaluation standard | Treated as a test list | None |
| ISO 11135 | EO sterilization standard | Confused with ISO 11137 | None |
| ISO 11137 | Radiation sterilization standard | Confused with ISO 11135 | None |
| ISO 17665 | Moist heat sterilization standard | Confused with dry heat | None |

## 5. The mechanics, step by step

The procedure below covers the reliability and sterilization programme. Each step states its input, output, and rationale. The edge cases required by the module are mapped at Step 8 and summarised in Section 5.14.

### Step 1. Define the use profile

**Input:** the intended use, the clinical workflow, and the lifetime claim.

**Output:** a use profile in hours per day, cycles per day, temperature and humidity range, and shelf life.

**Rationale:** every reliability calculation is per unit time and per stress cycle. A device used 24 hours a day has a different field exposure from one used for 10 minutes. A device that is stored for two years before use has a shelf-life requirement that is separate from its operating life.

The profile must be stated in the claim. "Five-year life" is not a profile; "two hours per day, 20 cycles per day, 15 to 30 °C, 5 years" is.

### Step 2. Build the reliability block diagram

**Input:** the system architecture and the failure definitions.

**Output:** a block diagram with series and redundant paths.

**Rationale:** the diagram determines how component failure rates combine. A series path adds the rates; a redundant path multiplies the unreliabilities. A diagram that ignores a redundant path overestimates the system failure rate; one that assumes redundancy where the switch is a single point of failure underestimates it.

The diagram must state what counts as a failure. A display failure may be cosmetic in one mode and safety-critical in another.

### Step 3. Allocate the failure-rate budget

**Input:** the system target and the block diagram.

**Output:** an allocated failure rate for each block.

**Rationale:** the allocation turns a system target into design inputs. A target of 5,000 FIT for the system becomes 1,000 FIT for the analogue board, 2,000 FIT for the processor, and 2,000 FIT for the power supply, with a reserve.

The allocation should be proportional to complexity and to the ability to improve. A block with a mature design can accept a lower allocation than a new block.

### Step 4. Select components and derating

**Input:** the allocation and the environment.

**Output:** a component list with derating rules and a predicted failure rate.

**Rationale:** derating reduces the stress on a component and therefore its failure rate. A capacitor operated at 50% of its rated voltage has a lower failure rate than one at 90%. The derating rule must match the mechanism: voltage for dielectrics, current for connectors, temperature for semiconductors.

The prediction is only as good as the data. A handbook prediction and a vendor prediction can differ by a factor of two or more.

### Step 5. Plan the accelerated life test

**Input:** the mechanism, the use profile, and the allocation.

**Output:** a test plan with stress levels, sample size, duration, and failure criteria.

**Rationale:** the test must accelerate the mechanism that is expected in the field, and it must produce enough failures to estimate the rate. A test with zero failures gives a lower bound, not a rate. A test with one failure gives a wide confidence interval.

The plan must include the readout intervals, the measurement method, and the criterion for a failure. A failure that is detected only at the end of the test is a failure at an unknown time.

### Step 6. Choose the acceleration model

**Input:** the mechanism and the stress.

**Output:** a model, its parameters, and a justification.

**Rationale:** temperature-driven mechanisms use Arrhenius; thermal-cycle fatigue uses Coffin-Manson; humidity-driven mechanisms use a Peck or a similar model. The model's parameters must come from the literature, a prior test, or a materials study, not from a fit to the same data that will be used to validate it.

The model must be checked for mechanism consistency. If the failure mode changes between stress levels, the model does not apply across the range.

### Step 7. Run the test and record the failure modes

**Input:** the plan, the units, and the chamber.

**Output:** time-to-failure data and a failure analysis for each unit.

**Rationale:** the failure mode is the evidence that the mechanism is the intended one. A test that records only the time and not the mode cannot distinguish a valid acceleration from an invalid one.

The failure analysis should include the location, the appearance, and the material. A solder crack at the joint, a wire bond lift, and a connector corrosion are different mechanisms with different models.

### Step 8. Analyse and convert to field time

**Input:** the time-to-failure data and the acceleration factor.

**Output:** a field-equivalent life, a failure rate, and a confidence interval.

**Rationale:** the conversion multiplies the test time by the acceleration factor. The result is a point estimate; the confidence interval depends on the number of failures and the test type.

The analysis must state the distribution. The exponential model is appropriate for random failures; a Weibull model with a shape parameter greater than one is appropriate for wear-out. Using the exponential model for wear-out underestimates the risk at the end of life.

### Step 9. Verify against field data

**Input:** the field returns, the complaint data, and the installed base.

**Output:** an observed failure rate and a comparison with the prediction.

**Rationale:** the field is the only test that uses the real stress profile. A prediction that is not compared with field data is a hypothesis. The comparison should be by failure mode, not only by rate, because a rate that matches can hide two compensating errors.

The field data must be normalised by exposure. A rate per unit shipped is not a rate per unit hour unless the exposure is known.

### Step 10. Classify the device for biological evaluation

**Input:** the contact nature, the duration, and the materials.

**Output:** an ISO 10993-1 category and a list of biological endpoints.

**Rationale:** the category determines the endpoints. A surface device with limited contact does not need the same tests as an implant with long-term contact. The classification must be justified by the clinical use, not by the desire to reduce testing.

The evaluation must consider the manufacturing process, the sterilization method, and the packaging, because each can introduce residues.

### Step 11. Plan the extractables and leachables work

**Input:** the materials, the contact conditions, and the category.

**Output:** an extraction protocol, an analytical method, and an AET.

**Rationale:** extractables are what the material releases under exaggerated conditions; leachables are what it releases under clinical conditions. The extraction solvents and temperatures must represent the clinical exposure, and the analytical method must be sensitive enough to detect the AET.

The AET is derived from the toxicological risk assessment. A method that cannot detect the AET cannot support the conclusion.

### Step 12. Select and validate the sterilization method

**Input:** the materials, the packaging, the bioburden, and the category.

**Output:** a method, a dose or cycle, and a validation plan.

**Rationale:** the method must be compatible with the materials and the packaging, and it must deliver the required SAL to the most difficult location. Gamma and EO are common for single-use devices; moist heat is common for reusable instruments.

The validation must include the bioburden, the dose audit, the package integrity, and the residuals. A sterilization cycle that is validated on a surrogate but not on the actual device is not validated.

### Step 13. Establish the shelf life and the package integrity

**Input:** the packaging, the sterilization method, and the storage conditions.

**Output:** a shelf-life claim and an accelerated ageing study.

**Rationale:** the shelf life depends on the package's barrier properties and the material's ageing. Accelerated ageing uses the Arrhenius model for the package and the adhesive, with the same caution about mechanism changes.

The package integrity test must be performed after the ageing and the sterilization, because both can degrade the seal.

### Step 14. Document and maintain

**Input:** the test reports, the process records, and the change control.

**Output:** a reliability and sterilization file with a change procedure.

**Rationale:** a change to a component, a material, a process, or a supplier can invalidate the reliability or sterilization claim. The change control must assess the impact and re-test if necessary.

A device that is modified after validation without an assessment has an unknown reliability and sterilization status.

### 5.13a Derating rules by component class

| Component | Parameter | Typical derating | Mechanism |
|---|---|---|---|
| Ceramic capacitor | Voltage | 50% of rated | Dielectric breakdown |
| Electrolytic capacitor | Voltage, temperature | 80%, 10 K below rated | Wear-out |
| Resistor | Power | 50% of rated | Drift |
| Semiconductor | Junction temperature | 25 K below maximum | Arrhenius |
| Connector | Current | 50% of rated | Contact fatigue |
| Relay | Load | 50% of rated | Contact erosion |
| Battery | Charge, temperature | 80%, 10 K below maximum | Capacity fade |
| PCB | Thermal cycle | 50% of strain | Solder fatigue |

The derating rule is a design input, not a verification result. A component that is operated at its rating will have a shorter life than the handbook prediction assumes, and the prediction will be wrong in the optimistic direction.

### 5.13b Selecting stress levels

| Stress | Use level | Test level | Acceleration | Constraint |
|---|---|---|---|---|
| Temperature | 37 °C | 85 °C | Arrhenius | Below glass transition |
| Temperature | 37 °C | 105 °C | Arrhenius | Mechanism change risk |
| Thermal cycle | 15 K | 80 K | Coffin-Manson | Within material limits |
| Humidity | 50% RH | 85% RH | Peck | Condensation avoided |
| Voltage | 5 V | 6.5 V | Inverse power law | Below breakdown |
| Cycle rate | 2/day | 10/day | Frequency term | Self-heating |

The test level must be high enough to accelerate, and low enough to avoid a new mechanism. The window is often narrower than it appears, especially for polymers and solder.

### 5.13c Choosing the distribution

| Distribution | Shape | Use | Consequence of misuse |
|---|---|---|---|
| Exponential | Constant rate | Random failures | Underestimates wear-out |
| Weibull | Shape β | Wear-out, infant mortality | Needs enough failures |
| Lognormal | Log time | Fatigue, repair time | Sensitive to outliers |
| Normal | Symmetric | Measurement error | Wrong for time-to-failure |

A Weibull fit with three or more failures gives a shape parameter that distinguishes infant mortality (β < 1), random failure (β ≈ 1), and wear-out (β > 1). A fit with one or two failures is not informative about the shape.

### 5.13d ISO 10993-1 endpoint matrix

| Contact | Duration | Core endpoints |
|---|---|---|
| Surface, skin | Limited | Cytotoxicity, sensitization, irritation |
| Surface, mucosa | Prolonged | Core plus systemic toxicity |
| External, blood path | Limited | Core plus hemocompatibility |
| External, tissue | Prolonged | Core plus implantation |
| Implant, bone | Long-term | Core plus chronic toxicity, carcinogenicity |
| Implant, blood | Long-term | Core plus hemocompatibility, degradation |

The matrix is a summary. The standard's tables are more detailed, and the evaluation must justify any endpoint that is omitted. A device that contacts blood for less than 24 hours is not exempt from hemocompatibility.

### 5.13e Sterilization method comparison

| Method | Standard | Dose or cycle | Material impact | Residual |
|---|---|---|---|---|
| Ethylene oxide | ISO 11135 | 4 half cycles plus full | Polymer absorption, aeration | EO, ECH |
| Gamma | ISO 11137 | 25 kGy typical | Chain scission, discoloration | None |
| Electron beam | ISO 11137 | 25 kGy typical | Similar to gamma, faster | None |
| Moist heat | ISO 17665 | 121 °C, 15 min or 134 °C, 4 min | Hydrolysis, warping | None |
| Dry heat | ISO 20857 | 160 °C, 2 h | Oxidation | None |
| Hydrogen peroxide | ISO 22441 | Cycle-specific | Compatibility varies | H2O2 |

The method must be compatible with the materials, the packaging, and the residuals limit. A method that is effective but leaves a toxic residual is not acceptable.

### 5.13f Sterilization validation evidence

| Evidence | What it shows | Method |
|---|---|---|
| Bioburden | Pre-sterilization load | ISO 11737-1 |
| D-value | Resistance of the bioburden | ISO 11138 series |
| Dose map | Minimum dose location | Dosimetry |
| SAL | Process effectiveness | ISO 11137-2 |
| Residuals | Patient exposure | ISO 10993-7 |
| Package integrity | Barrier after process | Seal strength, dye penetration |
| Shelf life | Ageing | Accelerated and real-time |

The evidence set is interlocking. A dose map without a bioburden is not a validation, and a SAL calculation without a D-value for the actual bioburden is an assumption.

### 5.13g Reliability test types

| Test | Purpose | Sample | Output |
|---|---|---|---|
| ALT | Quantify life | 10–30 per cell | Time to failure |
| HALT | Find limits | 5–10 | Operating and destruct limits |
| Burn-in | Screen infant mortality | 100% | Defect removal |
| HASS | Screen production | 100% | Stress screen |
| Field trial | Confirm in use | 20–100 | Complaints, returns |
| Shelf life | Confirm storage | 3–10 per timepoint | Ageing curve |

ALT and HALT are often confused. ALT quantifies a life under a known mechanism; HALT finds the limits and does not produce a life estimate. A HALT result cannot be converted to field years.

### 5.13h Statistical confidence

A point estimate without a confidence interval is a number without a range. The interval depends on the number of failures, the test time, and the assumed distribution.

| Failures | Test type | Point estimate | 60% lower bound | 90% lower bound |
|---:|---|---:|---:|---:|
| 0 | time-terminated | undefined | 1.09 T | 0.43 T |
| 1 | time-terminated | T | 0.49 T | 0.26 T |
| 2 | time-terminated | T/2 | 0.39 T | 0.22 T |
| 3 | time-terminated | T/3 | 0.32 T | 0.19 T |

The table uses the chi-square relationship for a time-terminated test, where T is the total device-hours. A test with zero failures gives a lower bound only, and the bound is weaker than a test with failures at the same total time. This is why a zero-failure test is often extended rather than accepted.

The confidence statement must name the bound. "MTBF is 200,000 hours" and "the 60% lower confidence bound on MTBF is 157,000 hours" are different claims, and only the second is defensible from a test with two failures.

### 5.13i Failure analysis methods

| Method | What it finds | Destructive | Typical use |
|---|---|---|---|
| Visual inspection | Location, discoloration | No | First step |
| X-ray | Solder voids, cracks | No | BGA, connector |
| Cross-section | Crack path, intermetallics | Yes | Confirmation |
| Scanning acoustic microscopy | Delamination, voids | No | Package, die attach |
| Energy-dispersive X-ray | Composition | No | Corrosion, residue |
| Fourier-transform infrared | Polymer degradation | No | Coating, housing |
| Ion chromatography | Ionic residue | Yes | Cleaning, corrosion |
| Thermal imaging | Hot spots | No | Power, thermal |

The method must match the question. A visual inspection cannot find a cracked solder joint under a ball grid array, and an X-ray cannot identify a polymer's chemical change. A failure analysis that stops at the first plausible finding often misattributes the mechanism.

The tables above are the working vocabulary of a reliability programme. They are not independent: the derating rule sets the stress, the stress sets the acceleration, the acceleration sets the equivalent life, and the confidence sets the claim. A programme that uses one table without the others produces a number that cannot be defended.

The same logic applies to sterilization. The bioburden sets the log reduction, the D-value sets the dose, the dose map sets the minimum delivered dose, and the residuals limit sets the aeration. A validation that reports a SAL without the bioburden and the dose map is a calculation, not evidence.

### 5.14 Edge cases and where they enter

| Edge case | Step | Mechanism in this topic | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 4, 7 | Component operated at its rating | Derating review | Early failure |
| Common-mode rejection failure | 7 | Stress test couples noise | Functional check | False failure |
| Electrode offset | 7 | Sensor drift under stress | Calibration check | Measurement error |
| Clock jitter | 7 | Timing margin at temperature | Functional test | Intermittent failure |
| Brown-out | 7, 12 | Supply stress | Rail monitor | Reset, corruption |
| Watchdog reset loop | 7, 12 | Firmware recovery | Reset log | Unavailable device |
| Thermal throttling | 5, 7 | Processor power at temperature | Temperature log | Performance loss |
| Firmware update failure | 12, 14 | Update during validation | Version check | Invalid test article |
| Calibration drift | 8, 9 | Sensor ageing | Field comparison | False reliability estimate |

The edge cases enter through the stress test and the change control. A reliability test that does not monitor them can attribute a functional failure to the wrong mechanism.

### 5.15 Reliability and sterilization interface

| Interface | Reliability concern | Sterilization concern | Shared evidence |
|---|---|---|---|
| Polymer housing | Cracking, creep | Gamma discoloration, EO absorption | Material certificate |
| Adhesive | Creep, debonding | Radiation degradation | Peel test |
| Electronics | Thermal fatigue | Radiation dose to components | Dose map |
| Battery | Capacity fade | EO compatibility | Cycle test |
| Packaging | Seal strength | Barrier after sterilization | Seal test |
| Sensor | Drift | Residual EO | Calibration |

The interface is where most surprises occur. A material that is reliable in the field can fail after sterilization, and a sterilization cycle that is validated on a coupon can degrade a component that the reliability test never stressed.

## 6. Worked example

### 6.1 The device and the claim

A reusable patient monitor has a 5-year operating life, is used 8 hours per day, and is sterilized between patients by ethylene oxide. The electronics are on a 6-layer FR-4 board with a 0.8 mm ball-grid-array processor, a 1206 ceramic capacitor bank, and a board-to-board connector.

The reliability claim is 5,000 FIT for the electronics, which corresponds to an MTBF of 200,000 hours. The sterilization claim is a SAL of \(10^{-6}\) with a bioburden of 100 colony-forming units and a D-value of 1.5 kGy for the routine bioburden.

### 6.2 The MTBF arithmetic

A test of 200 units for 2,000 hours produced 2 failures. The device-hours are:

\[
T = 200 \times 2{,}000 = 400{,}000\ \text{device-hours}
\]

The point estimate of MTBF is:

\[
\text{MTBF} = \frac{T}{r} = \frac{400{,}000}{2} = 200{,}000\ \text{hours}
\]

The failure rate is:

\[
\lambda = \frac{1}{200{,}000} = 5 \times 10^{-6}\ \text{per hour} = 5{,}000\ \text{FIT}
\]

The one-year reliability, assuming a constant failure rate and 8,760 hours per year, is:

\[
R(1\ \text{year}) = e^{-5 \times 10^{-6} \times 8{,}760} = 0.9571
\]

The one-year failure probability is 4.29%. The five-year reliability is:

\[
R(5\ \text{years}) = e^{-5 \times 10^{-6} \times 43{,}800} = 0.8033
\]

The five-year failure probability is 19.7%. The MTBF of 200,000 hours is not a lifetime; it is the reciprocal of a rate that produces a 19.7% failure probability over the claimed life.

### 6.3 The accelerated life test

The dominant wear-out mechanism is a solder-joint fatigue driven by the on/off thermal cycle. The field cycle is 15 K; the accelerated cycle is 80 K. The Coffin-Manson exponent for the solder is 2.5.

\[
AF = \left(\frac{80}{15}\right)^{2.5} = 65.7
\]

A test of 500 cycles is equivalent to:

\[
500 \times 65.7 = 32{,}845\ \text{field cycles}
\]

At two cycles per day, that is:

\[
\frac{32{,}845}{2 \times 365} = 45.0\ \text{years}
\]

The test therefore supports the 5-year claim with a large margin, provided the failure mechanism is the same.

### 6.3a Field exposure arithmetic

| Quantity | Value | Derivation |
|---|---:|---|
| Use hours per day | 8 | Use profile |
| Use days per year | 365 | Calendar |
| Use hours per year | 2,920 | 8 × 365 |
| Five-year use hours | 14,600 | 2,920 × 5 |
| Cycles per day | 2 | On/off |
| Five-year cycles | 3,650 | 2 × 365 × 5 |
| Coffin-Manson equivalent | 32,845 | 500 × 65.7 |
| Margin on cycles | 9.0× | 32,845 / 3,650 |

The solder-cycle test supports the claim with a 9.0× margin in cycles. The margin is not a safety factor on time; it is a ratio of equivalent cycles, and it assumes that the field cycle is 15 K and the test cycle is 80 K.

### 6.4 The temperature acceleration

A second mechanism is a chemical degradation of the conformal coating. The Arrhenius model with an activation energy of 0.7 eV, a use temperature of 310.15 K (37 °C), and a stress temperature of 358.15 K (85 °C) gives:

\[
AF = \exp\left[\frac{0.7}{8.617 \times 10^{-5}}\left(\frac{1}{310.15} - \frac{1}{358.15}\right)\right] = 33.45
\]

A test of 2,000 hours at 85 °C is equivalent to:

\[
2{,}000 \times 33.45 = 66{,}909\ \text{field hours} = 7.64\ \text{years}
\]

The two mechanisms give different equivalent lives: 45.0 years for the solder cycle and 7.64 years for the coating. The lower value governs the claim, and the test plan must report both.

### 6.5 Diagnostic 1: the activation energy is inconsistent

A second ALT at 105 °C gives a much shorter life than the 85 °C test predicts. Fitting the two points gives an activation energy of 0.52 eV, not 0.7 eV. The inconsistency means that either the mechanism changed or the model is wrong.

The failure analysis shows a different mode at 105 °C: the coating cracks and the failure is at the coating, not at the substrate. The 105 °C test has accelerated a new mechanism, so it cannot be combined with the 85 °C data.

The design flaw is the coating's thermal expansion mismatch, not the model. The fix is a compliant coating or a lower stress temperature, and the 105 °C data must be reported separately.

### 6.6 Diagnostic 2: the field returns do not match the test

The field returns show connector corrosion, not solder fatigue. The ALT never stressed the connector because the test board used a soldered connection. The test was valid for the solder joint and irrelevant to the dominant field failure.

The diagnostic is to compare the failure mode distribution in the field with the distribution in the test. A test that produces one mode while the field produces another has not tested the field mechanism.

The design fix is a sealed connector or a corrosion-resistant plating, and the test plan must include a damp-heat test at the connector.

### 6.7 Vary one assumption: the activation energy

If the activation energy is 0.5 eV instead of 0.7 eV, the acceleration factor falls from 33.45 to 12.3, and the 2,000-hour test is equivalent to 2.80 years, not 7.64 years. The claim of a 5-year life would not be supported by that test alone.

| Activation energy | Acceleration factor | Equivalent years from 2,000 h |
|---:|---:|---:|
| 0.5 eV | 12.3 | 2.80 |
| 0.6 eV | 20.3 | 4.63 |
| 0.7 eV | 33.5 | 7.64 |
| 0.8 eV | 55.2 | 12.61 |

The lesson is that the activation energy is a measured or literature-derived parameter with a real uncertainty. A reliability claim that depends on a single assumed value is a claim with an unstated sensitivity.

### 6.8 The sterilization dose

The bioburden is 100 CFU, the D-value is 1.5 kGy, and the required SAL is \(10^{-6}\). The minimum dose is:

\[
D_{\min} = 1.5 \left(\log_{10} 100 - \log_{10} 10^{-6}\right) = 1.5 \times (2 + 6) = 12.0\ \text{kGy}
\]

A 25 kGy gamma dose delivers:

\[
\frac{25}{1.5} = 16.7\ \text{log reductions}
\]

which gives:

\[
\text{SAL} = 100 \times 10^{-16.7} = 2.2 \times 10^{-15}
\]

The 25 kGy dose is more than double the minimum. The margin covers the bioburden uncertainty and the dose distribution in the package.

If the D-value is 2.5 kGy instead of 1.5 kGy, the 25 kGy dose delivers 10.0 log reductions and the SAL is \(1.0 \times 10^{-8}\), which is still below \(10^{-6}\) but with less margin. The D-value is therefore a critical parameter, and it must be measured for the actual bioburden, not assumed.

### 6.8a Dose sensitivity

| D-value | Log reduction at 25 kGy | SAL from 100 CFU | Meets 10⁻⁶ |
|---:|---:|---:|---|
| 1.5 kGy | 16.7 | 2.2 × 10⁻¹⁵ | Yes |
| 2.0 kGy | 12.5 | 3.2 × 10⁻¹¹ | Yes |
| 2.5 kGy | 10.0 | 1.0 × 10⁻⁸ | Yes |
| 3.0 kGy | 8.3 | 4.6 × 10⁻⁷ | Yes |
| 3.5 kGy | 7.1 | 7.2 × 10⁻⁶ | No |

The table shows why the D-value must be measured, not assumed. At 3.5 kGy, the same 25 kGy dose fails the SAL requirement. A D-value that is 40% higher than assumed can turn a passing validation into a failing one.

### 6.9 What the example demonstrates

The arithmetic closes: 400,000 device-hours and 2 failures give 200,000 hours MTBF and 5,000 FIT; 5,000 FIT gives 4.29% one-year and 19.7% five-year failure probability; the Coffin-Manson factor of 65.7 converts 500 cycles to 45.0 field years; the Arrhenius factor of 33.45 converts 2,000 hours to 7.64 years; and the sterilization dose of 12.0 kGy is the minimum for the stated bioburden and SAL.

The two diagnostics reveal different flaws: one is a mechanism change at high stress, the other is a mismatch between the test and the field. The sensitivity to activation energy shows how much the claim depends on a parameter that is often assumed.

## 7. Code, query, or calculation

The program below reproduces the worked example with the standard library only. It covers the MTBF and reliability arithmetic, the two acceleration models and the activation-energy sensitivity, and the sterilization dose and SAL.

```python
import math

K_B = 8.617333e-5

def arrhenius(ea, t_use_c, t_stress_c):
    t_use = t_use_c + 273.15
    t_stress = t_stress_c + 273.15
    return math.exp((ea / K_B) * (1.0 / t_use - 1.0 / t_stress))

units, hours, failures = 200, 2000, 2
device_hours = units * hours
mtbf = device_hours / failures
rate = 1.0 / mtbf
print(f"device-hours {device_hours}")
print(f"MTBF {mtbf:.0f} h = {mtbf/8760:.2f} years")
print(f"failure rate {rate:.2e} /h = {rate*1e9:.0f} FIT")
for years in (1, 5):
    r = math.exp(-rate * years * 8760)
    print(f"R({years}y) {r:.4f}  F {1-r:.4f}")
print(f"availability {mtbf/(mtbf+8):.6f}")

af = arrhenius(0.7, 37, 85)
print(f"\nArrhenius AF {af:.2f}")
print(f"2000 h at 85 C -> {2000*af:.0f} field h = {2000*af/8760:.2f} years")
for ea in (0.5, 0.6, 0.7, 0.8):
    a = arrhenius(ea, 37, 85)
    print(f"Ea {ea:.1f} eV  AF {a:6.1f}  years {2000*a/8760:5.2f}")

cm = (80.0 / 15.0) ** 2.5
print(f"\nCoffin-Manson AF {cm:.1f}")
print(f"500 cycles -> {500*cm:.0f} field cycles = {500*cm/730:.1f} years at 2/day")
print(f"5-year cycle demand {2*365*5}; margin {(500*cm)/(2*365*5):.1f}x")

n0, d_value, sal_target = 100.0, 1.5, 1e-6
dose_min = d_value * (math.log10(n0) - math.log10(sal_target))
print(f"\nminimum dose {dose_min:.1f} kGy")
for d in (1.5, 2.0, 2.5, 3.0, 3.5):
    logs = 25.0 / d
    sal = n0 * 10.0 ** (-logs)
    print(f"D {d:.1f} kGy  logs {logs:5.1f}  SAL {sal:.1e}  pass {sal <= sal_target}")
```

The output is:

```text
device-hours 400000
MTBF 200000 h = 22.83 years
failure rate 5.00e-06 /h = 5000 FIT
R(1y) 0.9571  F 0.0429
R(5y) 0.8033  F 0.1967
availability 0.999960

Arrhenius AF 33.45
2000 h at 85 C -> 66909 field h = 7.64 years
Ea 0.5 eV  AF   12.3  years  2.80
Ea 0.6 eV  AF   20.3  years  4.63
Ea 0.7 eV  AF   33.5  years  7.64
Ea 0.8 eV  AF   55.2  years 12.61

Coffin-Manson AF 65.7
500 cycles -> 32845 field cycles = 45.0 years at 2/day
5-year cycle demand 3650; margin 9.0x

minimum dose 12.0 kGy
D 1.5 kGy  logs  16.7  SAL 2.2e-15  pass True
D 2.0 kGy  logs  12.5  SAL 3.2e-11  pass True
D 2.5 kGy  logs  10.0  SAL 1.0e-08  pass True
D 3.0 kGy  logs   8.3  SAL 4.6e-07  pass True
D 3.5 kGy  logs   7.1  SAL 7.2e-06  pass False
```

Four checks on the output:

1. The MTBF of 200,000 hours is 22.83 years only if the failure rate is constant for that long. The five-year failure probability of 19.67% is the more useful statement, because it is the probability over the claimed life, not the reciprocal of a rate.
2. The Arrhenius factor is 33.45 at 0.7 eV. The 0.1 eV change from 0.6 to 0.7 eV changes the factor from 20.3 to 33.5, a 65% increase. The activation energy is therefore one of the most sensitive inputs in the calculation.
3. The Coffin-Manson factor of 65.7 is a ratio of thermal excursions, not of times. It converts cycles, not hours, and the conversion to years depends on the assumed two cycles per day.
4. The dose table shows that the same 25 kGy process passes at a D-value of 3.0 kGy and fails at 3.5 kGy. The D-value is the parameter that determines whether the process is adequate, and it must be measured for the actual bioburden.

The program deliberately does not compute a confidence interval. A confidence interval requires a distributional assumption and a chi-square or likelihood calculation, and a hand-rolled approximation would be less trustworthy than a stated bound from a validated tool. Section 5.13h gives the bounds for the common cases.

The model also does not include the uncertainty in the acceleration factor. A production programme should treat the factor as a distribution, not a point, and propagate the uncertainty to the equivalent life. The sensitivity table is a simple version of that propagation.

## 8. Data, standards, and regulatory context

### 8.1 The standards map

| Area | Standard | Scope |
|---|---|---|
| Biological evaluation | ISO 10993-1:2018 | Evaluation and testing within a risk process |
| Cytotoxicity | ISO 10993-5 | In vitro cytotoxicity |
| Sensitization and irritation | ISO 10993-10 | Skin sensitization and irritation |
| Systemic toxicity | ISO 10993-11 | Systemic toxicity |
| Hemocompatibility | ISO 10993-4 | Blood interaction |
| Genotoxicity, carcinogenicity | ISO 10993-3 | Genotoxicity and carcinogenicity |
| Implantation | ISO 10993-6 | Local effects after implantation |
| EO residuals | ISO 10993-7:2026 | EO and ECH limits |
| Degradation | ISO 10993-9, -13, -14, -15 | Degradation and identification |
| Bioburden | ISO 11737-1 | Microbial population |
| Sterility test | ISO 11737-2 | Sterility testing |
| EO sterilization | ISO 11135:2014 | EO process development and validation |
| Radiation sterilization | ISO 11137-1, -2, -3 | Radiation process, dose setting, dosimetry |
| Moist heat | ISO 17665-1 | Moist heat process |
| Dry heat | ISO 20857 | Dry heat process |
| Packaging | ISO 11607-1, -2 | Packaging and validation |
| Reliability | IEC 62506 | Accelerated testing methods |
| Reliability prediction | IEC 62380, MIL-HDBK-217F, SR-332 | Failure-rate prediction |
| Risk management | ISO 14971:2019 | Risk management |
| Quality system | ISO 13485:2016 | Quality management |

The table is a map, not a reading list. A device programme uses the parts that its category and process require, and it justifies the parts it omits.

### 8.2 ISO 10993-1:2018 logic

The 2018 edition changed the matrix from "tests" to "endpoints to be considered." The distinction matters: an endpoint can be addressed by existing data, by chemical characterization, or by a test, and the evaluation must state which.

| Nature of contact | Duration | Endpoints to consider |
|---|---|---|
| Surface, intact skin | Limited | Cytotoxicity, sensitization, irritation |
| Surface, mucosal membrane | Prolonged | Cytotoxicity, sensitization, irritation, systemic toxicity |
| External, indirect blood path | Limited | Hemocompatibility, cytotoxicity, sensitization, irritation |
| External, tissue or bone | Prolonged | Implantation, cytotoxicity, sensitization, irritation, systemic toxicity |
| Implant, tissue or bone | Long-term | Chronic toxicity, carcinogenicity, implantation, genotoxicity, degradation |
| Implant, blood | Long-term | Hemocompatibility, chronic toxicity, carcinogenicity, degradation |

The duration categories are: limited (≤24 hours), prolonged (>24 hours to 30 days), and long-term (>30 days). The category is determined by the clinical use, and a device that may remain in place longer than the labelled duration is evaluated at the longer duration.

The FDA guidance of September 2023 reproduces a modified matrix that adds endpoints for some categories, including acute systemic toxicity and material-mediated pyrogenicity for breached surfaces, and genotoxicity for extracorporeal circuits regardless of duration. A manufacturer submitting to the FDA must use the FDA-modified matrix, not only the ISO table.

### 8.3 Extractables and leachables

The evaluation has two layers:

| Layer | Conditions | Purpose | Output |
|---|---|---|---|
| Extractables | Exaggerated solvent, temperature, time | Identify what can be released | Compound list, semi-quantitative |
| Leachables | Clinical or simulated use | Quantify what is released in use | Concentration per device |
| Toxicological risk | Exposure model | Compare with tolerable intake | Safety conclusion |

The analytical evaluation threshold (AET) is the concentration below which a compound does not require identification because its exposure is below the tolerable intake. The AET is derived from the dose, the device contact area, and the toxicological reference value. A method whose detection limit is above the AET cannot support the conclusion that the device is safe.

The common error is to run an extraction with a solvent that does not represent the clinical exposure, find nothing, and conclude that the device is clean. A non-detection is only meaningful if the method could have detected the compound at the AET.

### 8.4 Radiation sterilization and dose setting

ISO 11137-1 sets the requirements for the radiation process; ISO 11137-2 sets the methods for establishing the dose; ISO 11137-3 covers dosimetry.

| Method | Basis | Output |
|---|---|---|
| Method 1 | Bioburden and D-value | Minimum dose for SAL |
| Method 2 | Fractional or sublethal cycles | Dose for SAL |
| Method 3 | Bioburden and survival curve | Dose for SAL |
| VDmax25 | Verification dose experiment | Substantiates 25 kGy |
| VDmax15 | Verification dose experiment | Substantiates 15 kGy |

VDmax25 is limited to products with an average bioburden of 1,000 CFU or less. The verification dose for a bioburden of 100 CFU is about 7 kGy, and the test is 10 items irradiated at that dose with a sterility test. All 10 negative accepts the substantiation; one positive requires a repeat with 10 more items; two or more positives fail and require another method.

The 25 kGy dose is not a universal requirement. It is a commonly used dose that VDmax25 can substantiate for low-bioburden products. A product with a higher bioburden or a more resistant bioburden needs a higher dose or a different method.

### 8.5 Ethylene oxide

ISO 11135 covers the development, validation, and routine control of EO processes. The key parameters are EO concentration, temperature, humidity, dwell time, and aeration.

| Parameter | Why it matters | Typical control |
|---|---|---|
| EO concentration | Lethality | Specified range |
| Temperature | Reaction rate | 37–55 °C |
| Relative humidity | Microbial resistance | 40–80% |
| Dwell time | Exposure | Cycle-specific |
| Aeration | Residual removal | Temperature and time |
| Packaging | EO penetration | Porous path |

The half-cycle method compares the lethality of a half cycle with the full cycle to demonstrate that the full cycle has a margin. The overkill method uses a known resistant biological indicator and a cycle that provides a large margin. Both must demonstrate a SAL of \(10^{-6}\).

The residuals are limited by ISO 10993-7. EO and ethylene chlorohydrin (ECH) are the main analytes, and the limits depend on the contact duration and the patient population. A cycle that achieves the SAL but leaves residuals above the limit is not acceptable.

### 8.6 Moist heat

ISO 17665 covers moist heat sterilization. The lethality is described by the \(F_0\) value, the equivalent time at 121 °C with a \(z\)-value of 10 °C:

\[
F_0 = \Delta t \times 10^{(T - 121)/10}
\]

A cycle of 15 minutes at 121 °C has \(F_0 = 15\) minutes. A cycle of 4 minutes at 134 °C has:

\[
F_0 = 4 \times 10^{(134-121)/10} = 4 \times 10^{1.3} = 79.8\ \text{minutes}
\]

The 134 °C cycle is much more lethal per minute, which is why it is shorter. The \(F_0\) value is a process equivalence, not a SAL; the SAL is demonstrated by the validation with biological indicators.

Moist heat is common for reusable instruments and is generally incompatible with single-use electronics, batteries, and many polymers.

### 8.7 Reliability prediction standards

| Standard | Basis | Use | Limitation |
|---|---|---|---|
| MIL-HDBK-217F | Component models | Military, legacy | Pessimistic, dated |
| Telcordia SR-332 | Field data and models | Telecom, electronics | Needs field data |
| IEC 62380 | Mission profile | Automotive, industrial | Complex input |
| FIDES | Physics of failure and field | Aerospace, telecom | Requires data |
| Vendor data | Manufacturer test | Component selection | Not comparable across vendors |

A prediction is not a measurement. Two standards can differ by a factor of two or more for the same component, because they use different models and different field data. A programme that quotes a prediction without the standard and the mission profile is quoting a number that cannot be compared.

### 8.8 Regulatory context

In the United States, the FDA's 2023 guidance on ISO 10993-1 describes the content of a biocompatibility evaluation in a premarket submission. The evaluation must include the contact category, the materials, the manufacturing and sterilization processes, the chemical characterization, the toxicological risk assessment, and the test results where testing is needed.

In the European Union, the MDR requires a biological evaluation as part of the technical documentation, and the harmonised ISO 10993 standards give presumption of conformity. The MDR also requires a sterilization validation and a packaging validation, and it treats the sterilization process as a special process that must be validated.

The quality system standard, ISO 13485, requires the organisation to validate processes whose output cannot be verified by subsequent inspection. Sterilization is the canonical example: the sterility of every unit cannot be tested without destroying it, so the process must be validated and monitored.

### 8.9 Fields of a validation report

| Field | Reliability report | Sterilization report |
|---|---|---|
| Article identity | Part number, revision | Part number, packaging |
| Population | Units, lot, screening | Bioburden batches |
| Stress | Temperature, cycles, voltage | Dose, concentration, temperature |
| Duration | Hours or cycles | Dose or cycle time |
| Failure definition | Functional, parametric | Sterility test result |
| Failure mode | Location, mechanism | Organism, recovery |
| Model | Arrhenius, Coffin-Manson | Method 1, VDmax25 |
| Parameters | Activation energy, exponent | D-value, bioburden |
| Result | Equivalent life, rate | SAL, minimum dose |
| Confidence | Interval or bound | Dose audit frequency |
| Deviations | Recorded | Recorded |

The two reports share a structure because they answer the same question: what was tested, under what conditions, and what can be claimed. A report that omits the article identity or the failure definition cannot support a claim, regardless of the result.

## 9. Industry practice and case evidence

### 9.1 Reliability growth

A reliability programme does not start with a number; it starts with a mechanism. The practice is:

1. Identify the dominant mechanisms from the design, the field history, and the physics of failure.
2. Design them out where possible, and test the remainder.
3. Measure the failure mode, not only the time.
4. Feed the result back into the design and the derating rules.

A programme that tests a generic "life" without a mechanism produces a number that cannot be improved. A programme that tests the mechanism produces a design change.

### 9.2 HALT before ALT

Highly accelerated life testing (HALT) is a qualitative test that finds the operating and destruct limits by stepping stress until failure. It is performed early, on a small number of units, and it is not used to compute a life.

ALT is a quantitative test that operates at defined stress levels and measures time to failure. It is performed later, with a larger sample, and it is used to compute a life.

The common practice is HALT first, to find the weak points, then design changes, then ALT on the improved design. A programme that runs ALT without HALT often tests a design with an unknown weak link.

### 9.3 Field data and the installed base

A reliability prediction is verified against field returns. The verification must normalise by exposure:

\[
\lambda_{\text{field}} = \frac{\text{returns}}{\text{units} \times \text{average exposure hours}}
\]

A return rate per unit shipped is not a failure rate unless the exposure is known. A device that ships 10,000 units and receives 50 returns in the first year has a 0.5% return rate, but the failure rate per hour depends on whether the units were used for 100 hours or 8,000 hours.

The field data must also be classified by failure mode. A rate that matches the prediction can hide a mechanism that is different from the one that was tested.

### 9.4 Supplier data and derating

Component vendors publish failure-rate data, often at a reference temperature and a defined confidence. The data are useful for comparison within a vendor's family, but they are not directly comparable across vendors unless the test conditions and the models are the same.

The industrial practice is to derate and to qualify. A component that is operated at 50% of its rated voltage and 25 K below its maximum junction temperature has a much lower failure rate than one at the rating, and the qualification test confirms that the assembly survives the mission profile.

### 9.5 Sterilization dose audits

A radiation sterilization process is validated once and audited periodically. The dose audit tests a sample at the original verification dose and confirms that the bioburden and its resistance have not changed. A failed audit triggers a dose augmentation or a revalidation.

The audit is the mechanism that keeps the original validation valid. A process that is validated and never audited has no evidence that the bioburden is still within the assumed range.

### 9.6 Biocompatibility across a material change

A material change is the most common trigger for a new biological evaluation. A supplier changes a resin, a colourant, or a release agent, and the extractables profile changes. The evaluation must consider the change, and the change control must require the supplier to notify the manufacturer.

The industrial practice is a material declaration and a change-notification agreement. Without it, the manufacturer learns about the change from a field complaint or a regulatory inspection.

### 9.7 Case evidence: the sterilization that changed the polymer

A common failure is a polymer that passes the mechanical and biocompatibility tests before sterilization and fails after it. Gamma radiation can cause chain scission and discoloration; EO can be absorbed and released slowly; moist heat can hydrolyse a polyester.

The evidence is the post-sterilization test, not the pre-sterilization test. A biocompatibility evaluation that uses unsterilized samples does not represent the final device, and a mechanical test that uses unsterilized samples does not represent the shelf life.

### 9.8 Case evidence: the ALT that tested the wrong mechanism

A device passes a 1,000-hour damp-heat test but fails in the field at the connector. The damp-heat test used a soldered coupon, and the field failure is a crimped connector with a different plating. The test was valid for the coupon and irrelevant to the product.

The lesson is that the test article must represent the production article, including the connector, the plating, the harness, and the housing. A test on a simplified coupon is a test of the coupon.

### 9.9 What the evidence supports and what it does not

The evidence supports the claims that mechanism-based testing finds more relevant failures than generic life testing, that post-sterilization testing is necessary, and that field-data normalisation is required for a valid failure rate. The evidence does not support a universal MTBF for a device class, because the use profiles and the definitions differ. A quoted MTBF without the standard, the mission profile, and the confidence is not comparable.

### 9.10 Field data sources

| Source | Content | Bias | Use |
|---|---|---|---|
| Complaints | Customer-reported failures | Under-reports silent failures | Signal detection |
| Service records | Repairs and replacements | Depends on service access | Rate estimation |
| Returned units | Failure analysis | Only returns that come back | Mechanism identification |
| Remote telemetry | Usage and errors | Only connected devices | Exposure and error rates |
| Warranty claims | Failed units | Commercial definition of failure | Cost model |
| Post-market studies | Structured follow-up | Selected sites | Clinical performance |

The sources have different biases, and a rate computed from one is not comparable with a rate from another. The strongest programme triangulates at least two sources and reconciles the definitions.

## 10. Failure modes and red flags

| # | Failure mode | Mechanism | Symptom | Detection | Mitigation |
|---:|---|---|---|---|---|
| 1 | MTBF quoted as lifetime | Exponential model misread | Unmet life claim | Reliability review | State R(t), not MTBF |
| 2 | Constant-rate assumption on wear-out | Bathtub misuse | Late-life failures | Weibull fit | Wear-out test |
| 3 | Zero-failure test accepted | No rate estimate | False confidence | Confidence bound | Extend test |
| 4 | Wrong acceleration model | Mechanism mismatch | Field failures | Failure analysis | Mechanism study |
| 5 | Activation energy assumed | Literature value misapplied | Life error | Sensitivity | Measure Ea |
| 6 | Mechanism change at high stress | New failure mode | Test-only failure | Multi-level test | Reduce stress |
| 7 | Test article not production | Coupon vs product | Field failure | Build comparison | Production article |
| 8 | Field data not normalised | Exposure unknown | Wrong rate | Exposure audit | Hours-based rate |
| 9 | Sterilization before biocompat test | Material change | Post-sterile failure | Test sequence | Test final device |
| 10 | Bioburden underestimated | Sampling error | SAL shortfall | Audit | Valid sampling |
| 11 | D-value assumed | Resistance unknown | Dose shortfall | Dose audit | Measure D-value |
| 12 | Dose map missing | Minimum dose unknown | Local SAL failure | Dosimetry | Dose mapping |
| 13 | EO residuals above limit | Insufficient aeration | Toxicity | Residual test | Aeration validation |
| 14 | Package seal degraded | Sterilization effect | Loss of sterility | Seal test | Post-process test |
| 15 | Material change not assessed | Supplier change | New extractable | Supplier notice | Change control |
| 16 | Shelf life without real-time data | Accelerated only | Late failure | Real-time study | Both |

The first eight rows are reliability errors; the last eight are sterilization and biocompatibility errors. Both groups share a common cause: a claim that is not tied to the conditions that produced it.

### 10.1 Red flags in a reliability review

- The MTBF is quoted without the standard and the confidence.
- The acceleration factor is quoted without the activation energy.
- The test has zero failures and is called a pass.
- The failure mode is not recorded.
- The test article is a coupon, not the product.
- The field return rate is per unit shipped, not per unit hour.
- The sterilization validation uses unsterilized samples for biocompatibility.
- The bioburden is a single batch.
- The D-value is from a different product.
- The dose map is from a different packaging configuration.

Each red flag is observable before the test is designed. A review that checks them costs hours; a failed validation costs months.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | Effect if wrong |
|---:|---|---|---|---|
| 1 | Failure rate is constant in useful life | Exponential model | Wear-out, infant mortality | Wrong R(t) |
| 2 | Failures are independent | Series rate addition | Common-cause failure | Underestimated rate |
| 3 | The test mechanism is the field mechanism | Acceleration validity | Mechanism change | Invalid life |
| 4 | The acceleration model applies across the range | Extrapolation | New mechanism at stress | Over- or under-acceleration |
| 5 | The activation energy is known | Arrhenius | Literature value mismatch | Life error of 2–3× |
| 6 | The Coffin-Manson exponent is known | Cycle model | Material or geometry change | Cycle error |
| 7 | The test article represents production | Generalisation | Coupon, prototype | Field failure |
| 8 | The use profile is representative | Field conversion | Different clinical use | Wrong exposure |
| 9 | The bioburden is representative | Dose calculation | Seasonal or supplier variation | SAL shortfall |
| 10 | The D-value is for the actual bioburden | Dose calculation | Resistant species | Dose shortfall |
| 11 | The dose map covers the worst location | SAL assurance | Packaging change | Local failure |
| 12 | The material is unchanged | Biocompatibility | Supplier reformulation | New extractable |
| 13 | The sterilization effect is included | Final device state | Pre-sterile testing | False pass |
| 14 | The package maintains the barrier | Shelf life | Seal degradation | Loss of sterility |

The assumptions interact. If the use profile (8) is wrong, the acceleration factor (4) converts to the wrong field time, and the MTBF (1) is wrong even if the test is valid. A verification plan should state which assumptions are tested and which are accepted with a justification.

The strongest evidence is a chain: the mechanism is identified, the model is justified, the test uses the production article, the failure mode is confirmed, and the field data agree. A claim that skips a link is a claim with an unknown error.

## 12. Exercises

### Exercise 1. MTBF and reliability

**Task.** A test of 150 units for 1,500 hours produces 3 failures. Compute the device-hours, the MTBF, the failure rate in FIT, and the one-year and five-year reliability. State one assumption behind each result.

**Deliverable.** The arithmetic and a short assumption list.

**Criteria.** Device-hours are 225,000; MTBF is 75,000 hours; rate is 13,333 FIT; R(1 year) is 0.890; R(5 years) is 0.558. The assumptions include a constant failure rate and a representative use profile.

**Time.** 30 minutes.

### Exercise 2. Arrhenius sensitivity

**Task.** Compute the acceleration factor for an activation energy of 0.6, 0.7, and 0.8 eV, with a use temperature of 30 °C and a stress temperature of 90 °C. Plot or tabulate the equivalent years from a 1,500-hour test.

**Deliverable.** A table with the three factors and the equivalent years.

**Criteria.** The temperatures are converted to kelvin; the factor increases with activation energy; the equivalent years are computed with 8,760 hours per year.

**Time.** 30 minutes.

### Exercise 3. Coffin-Manson and field cycles

**Task.** A solder joint has a field excursion of 12 K and a test excursion of 75 K. The exponent is 2.3. Compute the acceleration factor, the field cycles from 400 test cycles, and the years at three cycles per day.

**Deliverable.** The factor, the field cycles, and the years.

**Criteria.** The factor is 67.7; the field cycles are 27,076; the years are 24.7. The units and the cycle assumption are stated.

**Time.** 25 minutes.

### Exercise 4. ALT design

**Task.** Design an ALT for a device with a 5-year life and a solder-fatigue mechanism. State the stress levels, the sample size, the duration, the failure criteria, and the readout intervals. Justify each choice.

**Deliverable.** A one-page test plan.

**Criteria.** The stress level avoids a mechanism change; the sample is large enough for the intended confidence; the failure criteria are objective; the readouts allow time-to-failure analysis.

**Time.** 45 minutes.

### Exercise 5. ISO 10993-1 classification

**Task.** Classify a urinary catheter that is used for 14 days and a blood-contacting sensor that is used for 4 hours. List the endpoints to consider for each.

**Deliverable.** A two-row table with category, duration, and endpoints.

**Criteria.** The catheter is external communicating, prolonged; the sensor is external communicating, limited; the endpoints include hemocompatibility for the blood path and implantation for the prolonged tissue contact where applicable.

**Time.** 30 minutes.

### Exercise 6. Sterilization dose

**Task.** A device has a bioburden of 250 CFU and a D-value of 2.0 kGy. Compute the minimum dose for a SAL of \(10^{-6}\), and the SAL delivered by 25 kGy. State whether VDmax25 is applicable.

**Deliverable.** The dose, the SAL, and the applicability statement.

**Criteria.** The minimum dose is 16.8 kGy; the 25 kGy SAL is \(7.9 \times 10^{-11}\); VDmax25 is applicable because the bioburden is below 1,000 CFU, but the verification dose must be looked up for 250 CFU.

**Time.** 30 minutes.

### Exercise 7. Extractables and leachables

**Task.** Design an extraction protocol for a polymer housing that contacts skin for 7 days. State the solvents, the temperature, the time, the analytical method, and how the AET is derived.

**Deliverable.** A protocol outline with the AET rationale.

**Criteria.** The solvents represent the clinical exposure; the temperature and time are exaggerated but not degradative; the method's detection limit is below the AET; the toxicological reference is stated.

**Time.** 40 minutes.

### Exercise 8. Failure-mode audit

**Task.** A field return shows connector corrosion, but the ALT report shows solder fatigue. List the questions you would ask and the tests you would add. Explain why the existing ALT cannot support the field claim.

**Deliverable.** A question list and a revised test plan.

**Criteria.** The questions cover the test article, the stress, the failure definition, and the field exposure; the revised plan adds a damp-heat or mixed-flowing-gas test at the connector; the explanation distinguishes mechanism validity from test duration.

**Time.** 40 minutes.

## 13. Self-check answers

**1. What does MTBF mean?**

The mean time between failures is the reciprocal of the failure rate under the exponential model. It is not the lifetime of a unit. A device with an MTBF of 200,000 hours has a 4.29% probability of failing in the first year and a 19.67% probability over five years.

**2. When is the constant-failure-rate assumption valid?**

During the useful-life phase of the bathtub curve, when failures are caused by random overstress rather than manufacturing defects or wear-out. It is not valid during infant mortality or wear-out.

**3. What is the difference between MTBF and MTTF?**

MTBF applies to repairable systems and includes the repair time between failures. MTTF applies to non-repairable items and is the mean time to the first failure. For a non-repairable device, MTTF is the appropriate term.

**4. What is a FIT?**

A FIT is one failure per \(10^9\) device-hours. A failure rate of \(5 \times 10^{-6}\) per hour is 5,000 FIT.

**5. What does the Arrhenius model assume?**

That a single temperature-driven mechanism has a rate proportional to \(\exp(-E_a/kT)\), that the activation energy is constant over the stress range, and that no new mechanism appears at the higher temperature.

**6. What does the Coffin-Manson model assume?**

That thermal-cycle fatigue is driven by the plastic strain range, which scales with the temperature excursion, and that the exponent is constant over the range. It ignores frequency and peak temperature unless the Norris-Landzberg form is used.

**7. Why can a high-stress ALT be invalid?**

Because the dominant failure mechanism can change at the higher stress. A test that produces a coating crack at 105 °C when the field failure is solder fatigue at 60 °C has accelerated a different mechanism and cannot be converted to field life.

**8. What is the difference between ALT and HALT?**

ALT is quantitative and estimates life at defined stress levels. HALT is qualitative and finds the operating and destruct limits. A HALT result cannot be converted to field years.

**9. What are the ISO 10993-1 duration categories?**

Limited (≤24 hours), prolonged (>24 hours to 30 days), and long-term (>30 days). The category is determined by the clinical exposure, including the possibility of longer use than labelled.

**10. What is the difference between extractables and leachables?**

Extractables are released under exaggerated laboratory conditions and identify what can come out of the material. Leachables are released under clinical conditions and quantify the patient exposure. Extractables support the method; leachables support the safety conclusion.

**11. What is the AET?**

The analytical evaluation threshold is the concentration below which a compound does not need identification because its exposure is below the toxicological limit. A method that cannot detect at the AET cannot support a safety conclusion.

**12. What is the SAL and why is it \(10^{-6}\)?**

The sterility assurance level is the probability that a single item is non-sterile after the process. \(10^{-6}\) means one in a million, the level conventionally required for medical devices that contact sterile tissue or the bloodstream.

**13. How is the minimum sterilization dose computed?**

\(D_{\min} = D_{\text{value}} \times (\log_{10} N_0 - \log_{10} \text{SAL})\). For 100 CFU, a D-value of 1.5 kGy, and a SAL of \(10^{-6}\), the minimum is 12.0 kGy.

**14. Why is 25 kGy not a universal dose?**

Because the required dose depends on the bioburden and its resistance. VDmax25 substantiates 25 kGy only for products with a bioburden at or below 1,000 CFU and a successful verification dose test. A higher bioburden or a more resistant organism needs a higher dose or a different method.

**15. What is the difference between EO and gamma sterilization?**

EO is a chemical process that requires humidity, dwell, and aeration to remove residuals. Gamma is a radiation process that delivers a dose and leaves no chemical residual, but it can degrade polymers. The choice depends on the materials and the packaging.

**16. What is \(F_0\)?**

The equivalent time at 121 °C with a \(z\)-value of 10 °C. A cycle of 4 minutes at 134 °C has \(F_0 = 79.8\) minutes, which is why the higher temperature allows a shorter cycle.

**17. Why must biocompatibility testing use the final sterilized device?**

Because sterilization can change the material, create degradation products, or leave residuals. A test on an unsterilized sample does not represent the patient exposure.

**18. What is the weakest link in a reliability claim?**

The assumption that the test mechanism is the field mechanism. If that is wrong, the acceleration factor and the equivalent life are both wrong, regardless of the arithmetic.

## 14. Interview questions

**1. A supplier quotes an MTBF of one million hours. What do you ask?**

Which standard, which mission profile, which confidence, and whether the number is a prediction or a test result. A prediction from a handbook and a test result are not comparable, and a point estimate without a confidence bound is not a claim.

**2. How would you design an ALT for a new device?**

Start with the dominant mechanism from the physics of failure and the field history. Choose a stress that accelerates that mechanism without changing it, use enough units for the intended confidence, record the failure mode, and plan the analysis before the test starts.

**3. Why is a zero-failure test weak?**

Because it gives a lower bound, not a rate. The bound depends on the total device-hours and the confidence, and it is weaker than a test with failures at the same total time. A zero-failure result is often extended rather than accepted.

**4. How do you know the acceleration model is valid?**

By testing at more than one stress level and confirming that the failure mode and the fitted parameters are consistent. A single stress level cannot distinguish a valid model from a coincidental fit.

**5. What is the most common sterilization error?**

Validating the process on a surrogate or a coupon rather than the production device and packaging. The dose map, the bioburden, and the residuals must all be measured on the final configuration.

**6. How do you choose between EO and gamma?**

By the material compatibility, the packaging, the residuals, and the supply chain. Gamma is simpler but can degrade polymers; EO is gentler for some materials but requires aeration and residual control. The choice is a materials and process decision, not a preference.

**7. What is the role of the bioburden in a sterilization validation?**

It sets the log reduction needed and therefore the dose or cycle. A bioburden that is higher or more resistant than assumed can make a validated process inadequate, which is why dose audits are required.

**8. How do you handle a material change after validation?**

Assess the extractables profile, the mechanical properties, and the sterilization compatibility. Re-run the affected tests and update the technical file. The supplier change notification is the control that makes this possible.

**9. What is the difference between a reliability prediction and a demonstration?**

A prediction uses a model and component data; a demonstration uses a test or field data. A prediction is useful for allocation and comparison; a demonstration is the evidence for the claim.

**10. Why is field data normalisation important?**

Because a return rate per unit shipped does not account for exposure. A failure rate per device-hour requires the installed base and the average use time, and without it two products with different use profiles cannot be compared.

**11. What would you do if the field failure mode differs from the ALT failure mode?**

Treat the ALT as invalid for that mechanism, investigate the field mechanism, and add a test that reproduces it. The existing test may still be valid for the mechanism it did exercise, but it cannot support the field claim.

**12. How do you set a shelf life?**

With real-time data on the final package after sterilization, supported by accelerated ageing. The accelerated study is a predictor; the real-time study is the evidence. A shelf life based only on accelerated ageing is provisional.

## 15. Key terms

| Term | Meaning |
|---|---|
| Reliability | Probability of performing a required function under stated conditions for a stated time |
| MTBF | Mean time between failures, the reciprocal of the failure rate |
| MTTF | Mean time to failure for a non-repairable item |
| Failure rate | Failures per unit time |
| FIT | One failure per \(10^9\) device-hours |
| Bathtub curve | Infant mortality, useful life, and wear-out phases |
| Derating | Operating a component below its rating |
| Acceleration factor | Ratio of field life to test life |
| Activation energy | Energy barrier in the Arrhenius model |
| Coffin-Manson exponent | Slope of the thermal-cycle fatigue relationship |
| ALT | Accelerated life test, quantitative |
| HALT | Highly accelerated life test, qualitative |
| Weibull distribution | A lifetime distribution with a shape parameter |
| Bioburden | Microbial population on a device before sterilization |
| D-value | Dose or time for a one-log reduction |
| SAL | Sterility assurance level |
| EO | Ethylene oxide |
| ECH | Ethylene chlorohydrin |
| Gamma sterilization | Ionizing radiation sterilization |
| \(F_0\) | Equivalent time at 121 °C |
| Extractables | Substances released under exaggerated conditions |
| Leachables | Substances released under clinical conditions |
| AET | Analytical evaluation threshold |
| ISO 10993-1 | Biological evaluation standard |
| ISO 11135 | EO sterilization standard |
| ISO 11137 | Radiation sterilization standard |
| ISO 17665 | Moist heat sterilization standard |

## 16. Sources

1. U.S. Food and Drug Administration. *Use of International Standard ISO 10993-1, "Biological evaluation of medical devices — Part 1: Evaluation and testing within a risk management process."* Guidance for Industry and Food and Drug Administration Staff, final, 8 September 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-international-standard-iso-10993-1-biological-evaluation-medical-devices-part-1-evaluation-and. Accessed 25 October 2026.
2. U.S. Food and Drug Administration. *Biocompatibility Evaluation Endpoints by Contact Duration Periods.* https://www.fda.gov/medical-devices/biocompatibility-assessment-resource-center/biocompatibility-evaluation-endpoints-contact-duration-periods. Accessed 25 October 2026.
3. U.S. Food and Drug Administration. *Use of International Standard ISO 10993-1* (PDF). https://www.fda.gov/media/142959/download. Accessed 25 October 2026.
4. International Organization for Standardization. *ISO 10993-1:2018, Biological evaluation of medical devices — Part 1: Evaluation and testing within a risk management process.* Preview. https://webstore.ansi.org/preview-pages/ISO/preview_ISO+10993-1-2018.pdf. Accessed 25 October 2026.
5. International Organization for Standardization. *ISO/DIS 11137-2, Sterilization of health care products — Radiation — Part 2: Establishing the sterilization dose.* https://www.iso.org/standard/90093.html. Accessed 25 October 2026.
6. International Organization for Standardization. *ISO 11137-2:2013, Sterilization of health care products — Radiation — Part 2: Establishing the sterilization dose.* Preview. https://cdn.standards.iteh.ai/samples/62442/e5305d45164940eabfe2a57bb740d671/ISO-11137-2-2013.pdf. Accessed 25 October 2026.
7. Coakley, D., et al. "Power of Parametric: Methods to Validate Ethylene Oxide Sterilization Parametric Release." *Biomedical Instrumentation and Technology* 57, no. 4 (2023): 129–135. https://pmc.ncbi.nlm.nih.gov/articles/PMC10764058/. Accessed 25 October 2026.
8. McEvoy, B., et al. "Principles of Parametric Release: Emphasis on Data Collection and Interpretation." *Biomedical Instrumentation and Technology* 57, no. 4 (2023): 163. https://doi.org/10.2345/0899-8205-57.4.163. Accessed 25 October 2026.
9. Sadowski, P., et al. "The Ethylene Oxide Product Test of Sterility: Limitations and Interpretation of Results." *Biomedical Instrumentation and Technology* 55, no. S3 (2021): 45–56. https://pmc.ncbi.nlm.nih.gov/articles/PMC8655700/. Accessed 25 October 2026.
10. International Organization for Standardization. *ISO 14971:2019, Medical devices — Application of risk management to medical devices.* https://www.iso.org/standard/72704.html. Accessed 25 October 2026.
11. International Organization for Standardization. *ISO 13485:2016, Medical devices — Quality management systems — Requirements for regulatory purposes.* https://www.iso.org/standard/59752.html. Accessed 25 October 2026.
12. International Organization for Standardization. *ISO 10993-7:2026, Biological evaluation of medical devices — Part 7: Ethylene oxide sterilization residuals.* https://www.iso.org/obp/ui. Accessed 25 October 2026.

The ISO 10993-1 categories and endpoints in Section 8 are from sources 1 to 4. The VDmax25 bioburden limit and verification-dose logic are from sources 5 and 6. The EO validation logic and residual linkage are from sources 7 to 9. The standards list is from the standards themselves; the edition numbers should be confirmed against the current catalogue at the time of a submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`13_standards-and-emc.md`](13_standards-and-emc.md)
- Topic 12: [`12_surgical-and-rehab-robotics.md`](12_surgical-and-rehab-robotics.md)
- Survey chapter: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab directory: [`../../labs/`](../../labs/)

The previous topic covers standards and EMC, which is where the test levels and the requirement-versus-practice distinction are introduced. The reliability methods in this topic use the same evidence discipline: a claim without its conditions and its confidence is not evidence.

Topic 12 covers surgical and rehabilitation robotics, where the safety envelope and the lifetime of a reusable instrument are part of the same design problem. The survey chapter places reliability and sterilization in the device lifecycle, and the lab directory contains the measurement and analysis exercises that accompany this module.
