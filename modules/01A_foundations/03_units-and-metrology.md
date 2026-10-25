# Units and Metrology

> Module 01A — Foundations, Assumptions, and Units
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

A laboratory result is a value, a unit, a method, and an uncertainty. Systems that store only the
value are storing an incomplete measurement, and the missing parts cause failures that are both
common and severe.

- A glucose result of 126 is stored without a unit. One interface assumes mg/dL and another assumes
  mmol/L. The same row is read as 126 mg/dL by the diabetes registry and as 126 mmol/L by the
  pharmacy system. The second value is physiologically impossible by a factor of 18 and would be
  caught by a plausibility check; the first is plausible and therefore passes silently.
- A creatinine of 1.4 mg/dL is converted with 88.4 but the receiving column is labelled µmol/L while
  the sending system already converted. The value becomes 10,976 µmol/L, an impossible number that
  triggers an eGFR calculation of near zero.
- A device reports weight in pounds with a default of kilograms. The field is numeric with no unit.
  A 198 lb patient is recorded as 198 kg, and a weight-based dose is calculated at 90 kg too high.
- A pulse oximeter reports 92% in a patient with darker skin pigment. The arterial saturation is 85%.
  The device is accurate on average and biased in the population that needs it most.
- A calibration drifts by 3% over six months. Every result in the second quarter is biased in the
  same direction, the quality control chart is within limits because the limits were set from the
  drifted baseline, and the bias is discovered only when a new lot is introduced.

These are not software bugs in the ordinary sense. They are metrological failures: the measurement
was separated from the information needed to interpret it. The purpose of this topic is to make that
separation impossible to overlook.

Metrology is the science of measurement. It supplies the vocabulary (measurand, traceability,
uncertainty, calibration), the standards (the International System of Units, or SI), and the
procedures (uncertainty budgets, conformity assessment). Clinical units are a special case: they are
not all SI, they are not all convertible by a single factor, and some of them (international units,
enzyme units, arbitrary units) are defined by biological assay rather than by physics.

The industry problem is concrete. Every regulated laboratory, device manufacturer, and clinical
software vendor must demonstrate that its numbers are traceable and that its uncertainty is stated.
In the European Union, the In Vitro Diagnostic Regulation (IVDR, Regulation (EU) 2017/746) requires
metrological traceability of calibrators and control materials. In the United States, the Clinical
Laboratory Improvement Amendments (CLIA) require calibration and verification procedures. A data
scientist who joins a clinical team without this vocabulary will build pipelines that cannot pass an
audit, because the audit asks questions the pipeline was never designed to answer.

> **Note:** This topic covers the measurement side of units. The algebraic checking of composite units
> is Topic 04. The device signal chain that produces the raw value is Topic 08.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the seven SI base units, the defining constants since the 2019 redefinition, and the
   difference between a base unit and a derived unit. Assessed in Exercise 1.
2. **Convert** clinical values between conventional and SI units using exact factors, and justify the
   factor from the molar mass where one exists. Assessed in Exercises 2 and 3.
3. **Distinguish** accuracy from precision, trueness from repeatability, and calibration from
   verification, using the ISO vocabulary. Assessed in Exercise 4 and self-check item 2.
4. **Build** an uncertainty budget for a measured quantity, combining Type A and Type B components and
   reporting an expanded uncertainty with a coverage factor. Assessed in Exercise 5.
5. **Round** a result to the precision its uncertainty supports, and justify the number of significant
   figures. Assessed in Exercise 6.
6. **Design** a unit-handling policy for a data pipeline: canonical unit, original unit retention,
   conversion versioning, and plausibility rules. Assessed in Exercise 7.
7. **Audit** a clinical dataset for unit defects, including missing units, mixed units in one column,
   and unit-carrying codes that disagree with the value. Assessed in Exercise 8.
8. **Explain** why some quantities (pH, INR, eGFR, international units) cannot be converted by a
   simple factor, and state what their units mean. Assessed in Exercise 6 and interview question 9.

## 3. The core mental model

A measurement is a chain from a definition to a decision. Each link adds information and each link can
break.

```text
  DEFINITION            REALISATION           CALIBRATION
  what is the           how the unit is       how the instrument
  measurand?            physically realised   is tied to the realisation
  (quantity, unit,      (primary standard,    (calibrator, chain of
   matrix, condition)    e.g. caesium clock)   comparisons, certificate)
        |                      |                      |
        +----------------------+----------------------+
                               |
                               v
                         MEASUREMENT
                         instrument + procedure + operator + environment
                               |
                               v
                         RESULT
                         value + unit + uncertainty + method + time
                               |
                               v
                         INTERPRETATION
                         reference interval + decision limit + context
                               |
                               v
                         DECISION
```

Two consequences follow.

**The result is a set, not a scalar.** A defensible result is a tuple: numeric value, unit, uncertainty,
method, and time. Drop any element and the result becomes an assertion. This is why a data model that
stores only a float is structurally incapable of representing a clinical measurement.

**Traceability is a property of the chain, not of the number.** A value is traceable if it can be
related to a reference through an unbroken chain of calibrations, each with a stated uncertainty. A
number copied from a device without its calibration history is not traceable, even if it is correct.

### 3.1 The unit triple

Every quantity in a biomedical dataset should be represented as three linked fields:

```text
value      126
unit       mg/dL          (UCUM code)
system     http://unitsofmeasure.org
```

Plus, for regulated use, the method and the uncertainty. When a pipeline collapses these into a single
string such as "126 mg/dL", parsing becomes the new source of error: the string must be split, the
case normalised, and the unit mapped to a code. When the pipeline stores only the float, the unit is
lost entirely. Both are worse than three explicit fields.

### 3.2 The conversion decision

Converting a unit is a lossy operation in one direction and a precision-creating operation in the
other. Converting 126 mg/dL to 6.99 mmol/L creates a third significant figure that the original did
not have. Converting 6.99 mmol/L back to mg/dL gives 125.9, not 126. Round-tripping is therefore not
an identity, and a pipeline that converts on write and again on read will drift.

The rule that prevents drift: **convert once, at ingestion, store both the original value with its
unit and the canonical value with its unit, and never reconvert the canonical value.**

### 3.3 The two uncertainty questions

Every measurement has two distinct uncertainty questions:

1. **How far is this value from the true value?** This is accuracy, or trueness, and it is assessed
   against a reference method. It cannot be estimated from repeated measurements alone, because a
   biased instrument is precisely repeatable.
2. **How much would this value move if I measured again?** This is precision, and it is estimated
   from replicates. It says nothing about bias.

A measurement system can be precise and biased (a well-calibrated pipette that is consistently 5% low)
or accurate on average and imprecise (a noisy assay with no systematic offset). The two properties
require different experiments and different statistics, which is why CLSI separates EP05 (precision)
from method comparison and bias studies.

### 3.4 The SI in one paragraph

Since 20 May 2019 the SI is defined by fixing the numerical values of seven defining constants: the
caesium hyperfine frequency ΔνCs, the speed of light c, the Planck constant h, the elementary charge
e, the Boltzmann constant k, the Avogadro constant N_A, and the luminous efficacy K_cd. The seven base
units (second, metre, kilogram, ampere, kelvin, mole, candela) are deduced from these constants. The
distinction between base and derived units is retained for convenience, not because it is fundamental.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Metrology | The science of measurement and its application | Confusing with meteorology | none |
| Measurand | The quantity intended to be measured | Calling the instrument reading the measurand | quantity unit |
| True value | The value consistent with the definition of the measurand | Treating a reference method result as truth | quantity unit |
| Accuracy | Closeness of agreement between a measured value and the true value | Using it to mean precision | quantity unit |
| Trueness | Closeness of the mean of many measurements to the true value | Reporting it from a single replicate | quantity unit |
| Precision | Closeness of agreement among repeated measurements | Using it to mean accuracy | quantity unit |
| Repeatability | Precision under identical conditions, short interval, same operator | Calling it reproducibility | quantity unit |
| Intermediate precision | Precision under varied conditions within one laboratory | Confusing with reproducibility across labs | quantity unit |
| Reproducibility | Precision across laboratories, operators, and instruments | Using it for within-run replicates | quantity unit |
| Bias | Difference between the expected measurement and the true value | Confusing with random error | quantity unit |
| Calibration | Establishing the relationship between instrument response and reference values | Calling any adjustment a calibration | none |
| Verification | Confirming that a calibrated system meets a specification | Confusing with calibration | none |
| Adjustment | Changing the instrument so its output matches a reference | Calling it calibration | none |
| Traceability | An unbroken chain of calibrations to a reference, each with stated uncertainty | Claiming it because a certificate exists | none |
| Standard uncertainty (u) | Uncertainty of a component expressed as a standard deviation | Reporting it as an expanded interval | quantity unit |
| Type A uncertainty | Evaluated from repeated observations by statistical methods | Assuming it captures all error | quantity unit |
| Type B uncertainty | Evaluated from certificates, specifications, or judgement | Treating it as less legitimate | quantity unit |
| Combined uncertainty (u_c) | Root-sum-square of independent standard components | Adding components linearly | quantity unit |
| Expanded uncertainty (U) | u_c multiplied by a coverage factor k | Omitting k | quantity unit |
| Coverage factor (k) | Multiplier for the expanded uncertainty; k = 2 gives about 95% for a normal distribution | Assuming exactly 95% always | dimensionless |
| Resolution | The smallest change in a quantity that the instrument can distinguish | Confusing with accuracy | quantity unit |
| Sensitivity | Change in instrument response per unit change in the measurand | Confusing with diagnostic sensitivity | response per quantity |
| Limit of detection (LOD) | The lowest quantity reliably distinguished from blank | Confusing with the lowest reportable value | quantity unit |
| Limit of quantification (LOQ) | The lowest quantity measurable with stated precision and trueness | Using LOD as LOQ | quantity unit |
| Linearity | Proportionality of response across the measuring interval | Assuming it without testing | none |
| Drift | Change in response over time at a fixed measurand | Ignoring it in longitudinal data | quantity unit per time |
| Hysteresis | Direction-dependent response, differing on increasing and decreasing input | Ignoring it in pressure sensors | quantity unit |
| Tolerance | The permitted difference between a measured value and a specification | Confusing with uncertainty | quantity unit |
| Conformity assessment | Deciding whether a measurement lies within a specification | Ignoring the decision rule | none |
| Guard band | A margin subtracted from the tolerance to account for uncertainty | Applying it without documentation | quantity unit |
| SI | The International System of Units | Calling all clinical units SI | none |
| Base unit | One of the seven units from which others are derived | Assuming base units are more accurate | none |
| Derived unit | A unit formed from base units by multiplication and division | Assuming it is less fundamental | none |
| Coherent derived unit | A derived unit with no numerical factor other than 1 | Confusing with a non-coherent unit such as mmHg | none |
| Defining constant | A physical constant with a fixed numerical value under the 2019 SI | Treating it as measured | exact |
| Conventional unit | A unit still used clinically but not SI, such as mg/dL or mmHg | Treating it as SI | unit |
| SI unit | The unit recommended by the SI for a quantity | Assuming only one unit is permitted | unit |
| UCUM | Unified Code for Units of Measure, a machine-readable unit syntax | Using free-text units in interfaces | code |
| LOINC | A terminology for laboratory and clinical observations | Using it as a unit system | code |
| Molar mass | Mass of one mole of a substance | Confusing with molecular weight as a pure number | g/mol |
| International unit (IU) | A unit defined by biological activity, set by WHO reference preparations | Converting it to mass without a specific preparation | IU |
| Enzyme unit (U) | 1 µmol of substrate converted per minute | Converting it to mass | µmol/min |
| Equivalent (Eq) | One mole of charge | Using it for uncharged solutes | mol of charge |
| Osmole | One mole of osmotically active particles | Equating to mole for dissociating salts | mol of particles |
| pH | Negative decadic logarithm of hydrogen ion activity | Treating it as concentration | dimensionless |
| INR | International Normalised Ratio, a ratio to a reference thromboplastin | Converting it to seconds | dimensionless |
| eGFR | Estimated glomerular filtration rate | Treating it as a measured GFR | mL/min/1.73 m² |
| Corrected calcium | Calcium adjusted for albumin | Treating the correction as measured | mg/dL |
| Anion gap | Difference between measured cations and anions | Using it without the lab's reference range | mmol/L |
| Osmolal gap | Measured minus calculated osmolality | Ignoring the units of the calculated term | mOsm/kg |

Two clusters of definitions cause most of the confusion in practice.

**Accuracy, trueness, precision.** ISO 5725 separates trueness (bias) from precision (spread). The
word "accuracy" is used loosely in industry to mean either, which is why a specification such as
"accurate to ±5%" is ambiguous: it could mean a bias bound, a repeatability bound, or a combined
uncertainty. A defensible specification states which.

**Calibration, verification, adjustment.** Calibration establishes the relationship between response
and reference. Verification confirms a specification is met. Adjustment changes the instrument.
Regulations require calibration and verification; adjustment is an intervention that must itself be
documented and followed by verification. Calling an adjustment a calibration is a documentation
defect that auditors treat as a finding.

## 5. The mechanics, step by step

The procedure converts a raw measurement into a defensible, auditable result. It is written for a data
pipeline but applies equally to a bench scientist recording a result in a notebook.

### Step 1 — Define the measurand precisely

**Input.** A request such as "record the glucose".

**Output.** A measurand statement: analyte, specimen type, measurement property, and conditions.

**Reasoning.** "Glucose" is ambiguous. Plasma glucose concentration measured by the hexokinase method
in a fasting inpatient is a different measurand from whole-blood glucose measured by a point-of-care
meter in a capillary sample. They differ in matrix, in reference interval, and in uncertainty. The
measurand must name all four.

**Branch point.** If the specimen type or method is unknown, the value cannot be compared with a
reference interval. Treat it as uninterpretable rather than assuming plasma.

### Step 2 — Identify the unit and its code

**Input.** The unit as printed or transmitted.

**Output.** A UCUM code.

**Reasoning.** UCUM is the machine-readable syntax adopted by HL7 FHIR and recommended by LOINC. It is
case-sensitive: `mg/dL` is correct, `mg/dl` is not a valid code. Special units use brackets:
`mm[Hg]` for millimetres of mercury, `[in_i]` for international inches. Annotations in braces carry no
semantic value, so `mg{creat}` and `mg` are equivalent for conversion.

**Branch point.** If the unit is free text ("mg per dl", "mg%"), map it explicitly and log the mapping.
Do not rely on a fuzzy match in production.

### Step 3 — Confirm the result is traceable

**Input.** The instrument and its calibration record.

**Output.** A traceability statement or a flag.

**Reasoning.** Traceability means an unbroken chain to a reference. For a clinical assay, the chain
typically runs from the manufacturer's master calibrator to a reference method or certified reference
material. If the chain is broken, the uncertainty cannot be stated and the result cannot be compared
across laboratories.

**Branch point.** If traceability cannot be established, record the result with a flag rather than
silently treating it as comparable.

### Step 4 — Record the conditions that affect the value

**Input.** Patient state and pre-analytical conditions.

**Output.** Metadata fields attached to the result.

**Reasoning.** Glucose falls in an unpreserved tube at roughly 5–7% per hour at room temperature
because of glycolysis, so a delayed sample reads low. Potassium rises with hemolysis because red cells
contain far more potassium than plasma. Calcium falls with an under-filled EDTA tube. These are not
measurement errors in the instrument; they are errors in the measurand created before the instrument
saw the sample.

**Branch point.** If a pre-analytical flag is present (hemolysis index above the lab's threshold),
suppress or annotate the result according to the laboratory's policy. Do not average it with clean
results.

### Step 5 — Store the raw value and the raw unit

**Input.** The instrument output.

**Output.** Two immutable fields.

**Reasoning.** The raw value is the evidence. If a conversion is later found to be wrong, the raw value
allows recomputation. Pipelines that overwrite the raw value with the converted value destroy the
ability to audit the conversion.

**Branch point.** If the source system provides only a converted value, record that fact as a
provenance note.

### Step 6 — Normalise the unit string, not the value

**Input.** The raw unit.

**Output.** A canonical unit code plus a display string.

**Reasoning.** Normalisation should change representation, not quantity. `MG/DL`, `mg/dl`, and
`mg/dL` all map to the code `mg/dL` with no numeric change. Applying a numeric conversion during
normalisation conflates two operations and makes the log unreadable.

**Branch point.** If the unit maps to more than one code (for example `%` for both fraction and
saturation), resolve using the LOINC code or the measurand, not the unit alone.

### Step 7 — Select the canonical unit

**Input.** The analyte and the organisational policy.

**Output.** One canonical unit per analyte.

**Reasoning.** A canonical unit makes joins and comparisons possible. The choice is a policy decision,
not a scientific one: a US laboratory may canonicalise to mg/dL for glucose while an international
registry canonicalises to mmol/L. What matters is that the policy is documented, applied uniformly,
and versioned.

**Branch point.** If two systems must exchange data and disagree on the canonical unit, convert at the
boundary and state the direction in the interface contract.

### Step 8 — Apply the conversion factor exactly

**Input.** A value, a source unit, a target unit.

**Output.** A converted value with its unit.

**Reasoning.** Use the exact factor, not a rounded one, and state which convention the factor follows.
For glucose, the clinical convention used in this program is 18.0182 mg/dL per mmol/L, so mmol/L =
mg/dL / 18.0182. The often-quoted 18 is a rounded rule of thumb used in patient-facing material. The
IUPAC molar mass of anhydrous D-glucose, 180.156 g/mol, implies 18.0156; that value is used by some
diabetes software and differs from 18.0182 by about 0.015%.

**Branch point.** If the analyte has no single conversion factor (IU, enzyme units, osmolality), stop.
These require assay-specific or formula-specific handling, covered in Steps 16 to 19.

### Step 9 — Propagate the uncertainty through the conversion

**Input.** Value with standard uncertainty.

**Output.** Converted value with converted uncertainty.

**Reasoning.** A multiplicative conversion scales the uncertainty by the same factor. If u = 2.5 mg/dL
and the factor is 1/18.0182, then u = 0.1388 mmol/L. An additive conversion, such as the offset in
°F to °C, scales the uncertainty by 5/9 and adds nothing, because the offset is exact.

**Branch point.** If the conversion factor itself has uncertainty (for example, a molar mass known to
limited precision), combine it as an additional relative component.

### Step 10 — Round to the precision the uncertainty supports

**Input.** Converted value and uncertainty.

**Output.** A reported value with a justified number of digits.

**Reasoning.** The convention used here: report the uncertainty to one or two significant figures, and
report the value to the same decimal place as the uncertainty. A glucose of 6.99293 mmol/L with
u = 0.139 mmol/L is reported as 6.99 ± 0.14 mmol/L, or as 7.0 mmol/L when the uncertainty is rounded
to one figure (0.1). The original 126 mg/dL had three significant figures; reporting five is false
precision.

**Branch point.** If the value is used downstream, retain full precision in storage and round only at
display. Never round twice.

### Step 11 — Compare with a reference interval appropriate to the patient

**Input.** Result and patient attributes.

**Output.** An interpretation flag.

**Reasoning.** Reference intervals are partition-dependent. Creatinine differs by sex and age;
haemoglobin differs by sex and pregnancy; alkaline phosphatase differs by age; many analytes differ by
assay. CLSI EP28-A3c requires at least 120 reference individuals per partition for a nonparametric
interval, and permits verification with as few as 20 when transferring an interval.

**Branch point.** If the patient belongs to a population excluded from the reference study (pregnancy,
paediatrics, dialysis), the interval does not apply. Flag rather than compare.

### Step 12 — Apply the decision limit with a stated rule

**Input.** Result, uncertainty, decision limit.

**Output.** A conformity decision.

**Reasoning.** A threshold of 126 mg/dL for impaired fasting glucose, applied to a result of 126 with
an uncertainty of ±5 mg/dL, cannot decide conformity without a rule. The simple rule accepts values
below the limit; a guard-banded rule accepts only values below limit minus U; a probabilistic rule
reports the probability of exceeding the limit. The rule must be stated before the result is seen.

**Branch point.** If the result lies within one expanded uncertainty of the limit, report it as
indeterminate under the chosen rule rather than forcing a binary call.

### Step 13 — Store both values with provenance

**Input.** Raw and canonical values.

**Output.** A row with raw value, raw unit, canonical value, canonical unit, conversion identifier, and
method.

**Reasoning.** Provenance makes the row reproducible. The conversion identifier points to a versioned
table, so a later correction can be applied retroactively and the affected rows identified.

**Branch point.** If the pipeline cannot store provenance, it cannot support a regulatory audit. Treat
this as a design defect, not a display issue.

### Step 14 — Validate plausibility with units in mind

**Input.** The canonical value.

**Output.** Pass, warn, or reject.

**Reasoning.** Plausibility bounds must be unit-specific. A glucose of 700 is plausible in mg/dL and
impossible in mmol/L. A creatinine of 1.4 is plausible in mg/dL and implausibly low in µmol/L. A
weight of 198 is plausible in lb and obese in kg. The bounds must therefore be attached to the
canonical unit, and the check must run after conversion, not before.

**Branch point.** If a value fails the bound, quarantine it and route it for review. Do not silently
null it, because a null loses the evidence of the defect.

### Step 15 — Version the conversion table

**Input.** The factor set.

**Output.** A versioned artefact with an effective date.

**Reasoning.** Conversion factors change when a convention changes, and the change must be auditable.
The glucose factor is a live example: 18.0182, 18.0156, and 18 all appear in production systems.
Recording which version produced a value allows a registry to explain why two sites report slightly
different mmol/L values for the same mg/dL input.

**Branch point.** If a factor changes, recompute only forward unless a retrospective correction is
explicitly authorised and documented.

### Step 16 — Handle logarithmic and ratio quantities correctly

**Input.** pH, INR, eGFR, or a log-transformed analyte.

**Output.** A value with an explicit dimensionless status and a method.

**Reasoning.** pH is the negative decadic logarithm of hydrogen ion activity, not concentration, and
is dimensionless. A change from 7.40 to 7.10 is a factor of 1.995 in activity (10^0.30), not a 0.30
change in a linear quantity. INR is a ratio corrected for the thromboplastin sensitivity index and is
dimensionless; it cannot be converted to seconds without the local ISI and the mean normal
prothrombin time.

**Branch point.** If a system requests a unit for pH or INR, the correct response is "1" (dimensionless)
rather than a fabricated unit.

### Step 17 — Handle composite and rate units

**Input.** mg/kg/min, mL/kg/h, µg/kg/min, mEq/L.

**Output.** A decomposed unit and a conversion path.

**Reasoning.** Composite units convert component by component. Converting mg/kg/min to µg/kg/h
multiplies by 1,000 for mass and by 60 for time, giving a factor of 60,000. Errors here are common in
infusion calculations because the time basis changes (per minute to per hour) while the mass basis also
changes.

**Branch point.** If a rate is expressed per kilogram, the weight used must be stated: actual,
ideal, or adjusted body weight. The unit alone does not say.

### Step 18 — Handle biological units

**Input.** IU, U/L, mEq, mOsm/kg.

**Output.** A statement of the reference preparation or the definition.

**Reasoning.** An international unit is defined by the biological activity of a WHO reference
preparation, so the mass equivalent differs by analyte and can change when the preparation changes.
Heparin, insulin, and vitamin D are familiar examples. Enzyme units are defined kinetically
(1 U = 1 µmol/min) and depend on assay conditions. These units cannot be converted to mass with a
single factor.

**Branch point.** If a request asks for a mass conversion of an IU, refuse and cite the preparation.
Any number produced would be fabricated.

### Step 19 — Handle temperature and other offset scales

**Input.** °F, °C, K.

**Output.** The converted value with the correct offset handling.

**Reasoning.** Temperature conversion is affine, not multiplicative: °C = (°F − 32) × 5/9, and
K = °C + 273.15. A multiplicative-only converter will produce wrong values, and a ratio of temperatures
in °C is meaningless because the zero is arbitrary. Physiological temperatures should be stored in °C
with the site recorded (oral, rectal, tympanic, core), because site differences exceed the conversion
uncertainty.

**Branch point.** If a temperature is stored without a site, it cannot be compared with a
site-specific threshold. Flag it.

### Step 20 — Handle pressure and other non-coherent units

**Input.** mmHg, cmH₂O, kPa.

**Output.** A converted value in the canonical unit.

**Reasoning.** 1 mmHg = 0.133322 kPa exactly, from the definition of standard atmospheric pressure
(101.325 kPa = 760 mmHg). 1 cmH₂O = 0.0980665 kPa. These are non-coherent with the SI because they
carry a numerical factor, which is why they are accepted for use but not part of the coherent set.

**Branch point.** If a device reports in arbitrary units, do not convert. Record the arbitrary unit
and the device model.

### Step 21 — Audit the dataset for unit defects

**Input.** A table of results.

**Output.** A defect report with counts by type.

**Reasoning.** The audit looks for: missing units; more than one unit in a column; unit strings that
disagree with the value range; duplicate rows differing only by unit; and values that are exact powers
of the conversion factor away from the column median. The last check catches double conversion, which
produces a cluster at 18× or 88× the expected value.

**Branch point.** If more than a small fraction of a column is defective, stop the analysis. Unit
defects are not missing at random, and imputing them will bias every downstream estimate.

### Step 22 — Document the method, the unit policy, and the uncertainty

**Input.** All of the above.

**Output.** A methods paragraph and a data dictionary entry.

**Reasoning.** A reader must be able to answer: what was measured, on what specimen, by what method,
in what unit, converted with what factor, with what uncertainty, against what reference interval. If
any answer is missing, the result is not reproducible.

**Branch point.** If the documentation cannot be completed, the analysis is not ready for a regulated
submission or a clinical decision.

### 5.1 The unit-handling branch map

```text
Raw result
  |
  +-- unit missing? --------------> quarantine; do not infer from range alone
  |
  +-- multiple units in column? --> split by unit; convert each with provenance
  |
  +-- biological unit (IU, U)? ---> no mass conversion; record preparation
  |
  +-- logarithmic (pH, log VL)? --> dimensionless; report base and method
  |
  +-- ratio (INR, eGFR)? --------> dimensionless or indexed; do not convert to seconds
  |
  +-- offset scale (F, C, K)? ---> affine conversion, not multiplicative
  |
  +-- composite (mg/kg/min)? ----> convert each component; state weight basis
  |
  +-- value implausible? --------> check for double conversion before rejecting
  |
  +-- conversion applied? -------> store raw + canonical + factor version
```

### 5.2 The four checks that catch most unit errors

1. **Does every numeric column have a unit field?** If not, the column cannot be interpreted.
2. **Is the value plausible in the stated unit?** A glucose of 6.99 is plausible in mmol/L, not mg/dL.
3. **Is the value an exact multiple of a common factor from the median?** 18×, 88×, 10×, and 1,000×
   are the usual suspects.
4. **Does the round trip return the original?** Convert and convert back; a discrepancy beyond
   rounding indicates a wrong factor or a double conversion.

## 6. Worked example

A 58-year-old man attends a clinic. The laboratory reports a panel in conventional units, the device
reports vital signs, and the record includes body size. The task is to convert the panel to SI units
for an international registry, attach an uncertainty to the glucose result, compute the estimated
glomerular filtration rate, and detect a unit defect in a creatinine column that contains one
already-converted value.

### 6.1 Convert the laboratory panel

The conversions use the factors stated in the table, each with its basis. Molar masses are IUPAC
values; the glucose factor uses the clinical convention 18.0182 rather than the molar-mass value.

| Analyte | Result | Factor | Basis of factor | Converted | Reported |
|---|---:|---|---|---:|---:|
| Glucose | 126 mg/dL | ÷ 18.0182 | clinical convention | 6.99293 mmol/L | 6.99 mmol/L |
| Creatinine | 1.4 mg/dL | × 88.4 | MW 113.12 g/mol | 123.760 µmol/L | 124 µmol/L |
| Total cholesterol | 212 mg/dL | × 0.02586 | MW 386.65 g/mol | 5.4823 mmol/L | 5.48 mmol/L |
| Triglycerides | 180 mg/dL | × 0.01129 | triolein basis, MW 885.4 | 2.0322 mmol/L | 2.03 mmol/L |
| Total bilirubin | 2.1 mg/dL | × 17.1 | MW 584.66 g/mol | 35.91 µmol/L | 35.9 µmol/L |
| Calcium | 9.2 mg/dL | × 0.2495 | MW 40.08 g/mol | 2.2954 mmol/L | 2.30 mmol/L |
| Haemoglobin | 13.4 g/dL | × 10 | dL to L | 134.0 g/L | 134 g/L |

Each conversion is shown with its arithmetic.

```text
Glucose
  126 mg/dL / 18.0182 (mg/dL per mmol/L) = 6.99293 mmol/L
  Rounded to two decimals: 6.99 mmol/L

  Why 18.0182 and not 18:
    1 mmol/L of glucose = 180.156 mg/L (IUPAC molar mass) = 18.0156 mg/dL
    The clinical convention uses 18.0182, a 0.015% difference from the molar-mass value.
    The rounded 18 is a rule of thumb for patient-facing material and is not used here.
    At 126 mg/dL the three factors give 7.000, 6.99293, and 6.99379 mmol/L.
    The spread is 0.007 mmol/L, far below the measurement uncertainty of 0.146 mmol/L.

Creatinine
  1 mg/dL = 10 mg/L
  10 mg/L / 113.12 g/mol = 0.0884 mmol/L = 88.4 umol/L
  1.4 mg/dL x 88.4 = 123.76 umol/L

Cholesterol
  1 mmol/L = 386.65 mg/L = 38.665 mg/dL
  1/38.665 = 0.025863 -> 212 x 0.02586 = 5.4823 mmol/L

Triglycerides
  1 mmol/L of triolein = 885.4 mg/L = 88.54 mg/dL
  1/88.54 = 0.011294 -> 180 x 0.01129 = 2.0322 mmol/L
  Note: the factor is an approximation because triglyceride mixtures vary in fatty acid
  composition; the clinical convention uses 88.57 or 0.01129.

Bilirubin
  1 umol/L = 584.66 ug/L = 0.058466 mg/dL
  1/0.058466 = 17.10 -> 2.1 x 17.1 = 35.91 umol/L

Calcium
  1 mmol/L = 40.08 mg/L = 4.008 mg/dL
  1/4.008 = 0.24950 -> 9.2 x 0.2495 = 2.2954 mmol/L
```

The reported values carry the number of digits the uncertainty supports, not the number the
calculator produced. Glucose at 6.99 mmol/L has three significant figures, consistent with the
three-significant-figure input of 126 mg/dL. Reporting 6.992930 mmol/L would imply a precision the
assay does not have.

### 6.2 Convert temperature and body size

```text
Temperature
  C = (F - 32) x 5/9
    = (98.6 - 32) x 5/9
    = 66.6 x 5/9
    = 37.00 C
  K = 37.00 + 273.15 = 310.15 K

  The conversion is affine: the offset 32 is exact, and the scale factor 5/9 is exact.
  The uncertainty in the Fahrenheit reading (about 0.2 F for an oral thermometer) becomes
  0.11 C. Site matters more than conversion: an oral 37.0 C corresponds to a core
  temperature roughly 0.3 to 0.5 C higher.

Weight
  198 lb x 0.45359237 kg/lb = 89.8113 kg
  Reported: 89.8 kg

Height
  5 ft 11 in = (5 x 12) + 11 = 71 in
  71 in x 2.54 cm/in = 180.34 cm

Body mass index
  BMI = 89.8113 kg / (1.8034 m)^2
      = 89.8113 / 3.25225
      = 27.6151
  Reported: 27.6 kg/m2
```

The BMI is reported to one decimal place because the height and weight inputs are not precise to
more. A BMI of 27.615 implies a precision of 10 g in weight and 0.1 mm in height, neither of which a
clinical scale and stadiometer provide.

### 6.3 Convert blood pressure and compute the mean arterial pressure

```text
Systolic  148 mmHg x 0.133322 kPa/mmHg = 19.732 kPa
Diastolic  92 mmHg x 0.133322 kPa/mmHg = 12.266 kPa

MAP = DBP + (SBP - DBP)/3
    = 92 + (148 - 92)/3
    = 92 + 18.667
    = 110.67 mmHg
    = 110.67 x 0.133322 = 14.75 kPa
```

The MAP formula is an approximation that assumes a standard waveform and a resting heart rate. It
underestimates MAP when heart rate is high, because the diastolic decay occupies a larger fraction of
the cycle. The formula is adequate for a dashboard and inadequate for a physiological study.

### 6.4 Build the uncertainty budget for glucose

Three components contribute to the uncertainty of the glucose result.

| Component | Type | Value (mg/dL) | Source |
|---|---|---:|---|
| Calibrator certificate | B | 1.2 | Manufacturer certificate, k = 2 value converted to standard uncertainty |
| Within-run repeatability | A | 1.8 | Replicate measurements, standard deviation |
| Bias correction | B | 1.5 | Method comparison study, residual bias |

```text
Combined standard uncertainty
  u_c = sqrt(1.2^2 + 1.8^2 + 1.5^2)
      = sqrt(1.44 + 3.24 + 2.25)
      = sqrt(6.93)
      = 2.6325 mg/dL

Expanded uncertainty at k = 2
  U = 2 x 2.6325 = 5.2650 mg/dL

Interval
  126 - 5.27 = 120.74 mg/dL
  126 + 5.27 = 131.26 mg/dL

Converted to mmol/L
  u_c = 2.6325 / 18.0182 = 0.14610 mmol/L
  U   = 0.29220 mmol/L
  glucose = 6.993 mmol/L, interval 6.701 to 7.285 mmol/L
```

The reported result is 6.99 mmol/L with an expanded uncertainty of 0.29 mmol/L (k = 2), or
equivalently 6.99 ± 0.29 mmol/L. Rounded for clinical display: 7.0 mmol/L, interval 6.7 to 7.3.

The arithmetic of combining in quadrature assumes the three components are independent. The
calibrator certificate and the bias correction may not be: if the bias estimate was derived using the
same calibrator, part of the uncertainty is shared. A rigorous budget would carry the covariance. The
independent assumption is stated, not hidden.

### 6.5 Compute the eGFR

The 2021 CKD-EPI equation for adults uses creatinine, age, and sex, with no race term.

```text
eGFR = 142 x min(Scr/kappa, 1)^alpha x max(Scr/kappa, 1)^(-1.200) x 0.9938^age
       x 1.012 if female

For a male: kappa = 0.9 mg/dL, alpha = -0.302
Scr = 1.4 mg/dL
Scr/kappa = 1.4/0.9 = 1.5556

min(1.5556, 1) = 1
max(1.5556, 1) = 1.5556
1.5556^(-1.200) = 0.58848
0.9938^58 = 0.69722

eGFR = 142 x 1 x 0.58848 x 0.69722
     = 58.26
Reported: 58 mL/min/1.73 m2
```

Two metrological points follow. First, the equation expects creatinine in mg/dL; feeding it
123.76 µmol/L without conversion would produce a value near zero. Second, the result is an estimate
with its own uncertainty, typically wider than the creatinine assay uncertainty alone because the
equation was fitted to a population and does not capture individual variation in creatinine
generation.

### 6.6 Detect the unit defect

A creatinine column contains: 1.4, 1.5, 1.3, 1.6, 1.4, 123.8, 1.5, 1.4.

```text
Sorted: 1.3, 1.4, 1.4, 1.4, 1.5, 1.5, 1.6, 123.8
Median = (1.4 + 1.5)/2 = 1.45  (the script uses the upper median, 1.5)

123.8 / 1.5 = 82.5x the median
The factor 88.4 is the creatinine conversion. A value 82.5x the median is a converted
value in a column of unconverted values.
```

The correct handling is to quarantine the row, confirm with the source system that the value is in
µmol/L, and convert the rest of the column or the single row as appropriate. Deleting the row loses
the evidence; converting the whole column would corrupt the seven correct values by a factor of 88.4.

### 6.7 Round-trip and drift

```text
126 mg/dL -> 6.99293 mmol/L -> 6.99293 x 18.0182 = 126.0000 mg/dL
Drift = 0.0000 mg/dL
```

The round trip is exact when the full-precision value is carried. If the pipeline stores the rounded
6.99 and converts back, it gets 125.947 mg/dL, a drift of 0.053 mg/dL per cycle. Repeated
conversion on read and write accumulates this drift and eventually moves a value across a decision
threshold. The rule is to convert once and store both.

### 6.8 What the example demonstrates

| Task | Operation | Failure prevented |
|---|---|---|
| Panel conversion | Exact factor with stated basis | 18×, 88×, and 39× errors |
| Significant figures | Round at the end | False precision in a registry |
| Temperature | Affine conversion | Multiplicative-only converter error |
| Body size | Exact lb and in factors | kg/lb and cm/in confusion |
| Pressure | 0.133322 kPa/mmHg | Unitless pressure comparisons |
| Uncertainty | Quadrature combination, k = 2 | A point value with no interval |
| eGFR | Unit-correct equation input | Near-zero eGFR from µmol/L input |
| Defect detection | Ratio to median | A single converted value corrupting a column |
| Round trip | Full-precision storage | Drift across repeated conversions |

## 7. Code, query, or calculation

The script converts the panel, builds the uncertainty budget, computes the eGFR, and runs the defect
detection. It uses only the standard library.

```python
"""Topic 03 verification: clinical unit conversions, uncertainty, eGFR."""
from math import sqrt

F = {
    "glucose_mgdl_per_mmol": 18.0182,     # clinical convention
    "creatinine_umol_per_mgdl": 88.4,     # MW 113.12
    "cholesterol_mmol_per_mgdl": 0.02586, # MW 386.65
    "triglyceride_mmol_per_mgdl": 0.01129,# triolein basis
    "bilirubin_umol_per_mgdl": 17.1,      # MW 584.66
    "calcium_mmol_per_mgdl": 0.2495,      # MW 40.08
    "mmhg_to_kpa": 0.133322,
    "lb_to_kg": 0.45359237,
    "in_to_cm": 2.54,
}

print("6.1 Panel conversions")
glu = 126 / F["glucose_mgdl_per_mmol"]
print(f"  glucose    126 mg/dL   -> {glu:.5f} mmol/L -> {glu:.2f}")
cr = 1.4 * F["creatinine_umol_per_mgdl"]
print(f"  creatinine 1.4 mg/dL   -> {cr:.3f} umol/L -> {cr:.0f}")
chol = 212 * F["cholesterol_mmol_per_mgdl"]
print(f"  cholesterol 212 mg/dL  -> {chol:.4f} mmol/L -> {chol:.2f}")
tg = 180 * F["triglyceride_mmol_per_mgdl"]
print(f"  triglycerides 180      -> {tg:.4f} mmol/L -> {tg:.2f}")
bili = 2.1 * F["bilirubin_umol_per_mgdl"]
print(f"  bilirubin  2.1 mg/dL   -> {bili:.2f} umol/L -> {bili:.1f}")
ca = 9.2 * F["calcium_mmol_per_mgdl"]
print(f"  calcium    9.2 mg/dL   -> {ca:.4f} mmol/L -> {ca:.2f}")
hb = 13.4 * 10
print(f"  hemoglobin 13.4 g/dL   -> {hb:.1f} g/L")

print("\n6.2 Temperature and body size")
f = 98.6
c = (f - 32) * 5 / 9
k = c + 273.15
print(f"  {f} F -> {c:.2f} C -> {k:.2f} K")
lb = 198
kg = lb * F["lb_to_kg"]
print(f"  {lb} lb -> {kg:.4f} kg -> {kg:.1f}")
inches = 5 * 12 + 11
cm = inches * F["in_to_cm"]
print(f"  {inches} in -> {cm:.2f} cm")
bmi = kg / (cm / 100) ** 2
print(f"  BMI = {bmi:.4f} -> {bmi:.1f} kg/m2")

print("\n6.3 Blood pressure and MAP")
sbp, dbp = 148, 92
print(f"  SBP {sbp} mmHg -> {sbp*F['mmhg_to_kpa']:.3f} kPa")
print(f"  DBP {dbp} mmHg -> {dbp*F['mmhg_to_kpa']:.3f} kPa")
map_mmhg = dbp + (sbp - dbp) / 3
print(f"  MAP = {map_mmhg:.2f} mmHg -> {map_mmhg*F['mmhg_to_kpa']:.2f} kPa")

print("\n6.4 Uncertainty budget for glucose")
u_components = {
    "calibrator certificate": 1.2,   # mg/dL, Type B
    "within-run repeatability": 1.8, # mg/dL, Type A
    "bias correction": 1.5,          # mg/dL, Type B
}
uc = sqrt(sum(u * u for u in u_components.values()))
for name, u in u_components.items():
    print(f"  {name:26s} u = {u:.1f} mg/dL")
print(f"  combined u_c = {uc:.4f} mg/dL")
print(f"  expanded U (k=2) = {2*uc:.4f} mg/dL")
lo, hi = 126 - 2 * uc, 126 + 2 * uc
print(f"  interval = {lo:.2f} to {hi:.2f} mg/dL")
uc_mmol = uc / F["glucose_mgdl_per_mmol"]
print(f"  in mmol/L: u_c = {uc_mmol:.5f}, U = {2*uc_mmol:.5f}")
glu_lo = lo / F["glucose_mgdl_per_mmol"]
glu_hi = hi / F["glucose_mgdl_per_mmol"]
print(f"  glucose interval in mmol/L = {glu_lo:.3f} to {glu_hi:.3f}")

print("\n6.5 CKD-EPI 2021 eGFR, male age 58, creatinine 1.4 mg/dL")
def ckd_epi_2021(scr, age, female=False):
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = scr / kappa
    egfr = (142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200
            * 0.9938 ** age)
    if female:
        egfr *= 1.012
    return egfr
egfr = ckd_epi_2021(1.4, 58)
print(f"  ratio = {1.4/0.9:.4f}")
print(f"  eGFR = {egfr:.4f} -> {egfr:.0f} mL/min/1.73 m2")

print("\n6.6 Round trip")
back = glu * F["glucose_mgdl_per_mmol"]
print(f"  126 -> {glu:.5f} -> {back:.4f} mg/dL")
print(f"  drift = {back - 126:.6f} mg/dL")

print("\n6.7 Double-conversion detection")
col = [1.4, 1.5, 1.3, 1.6, 1.4, 123.8, 1.5, 1.4]
med = sorted(col)[len(col)//2]
print(f"  median = {med}")
for v in col:
    if v / med > 20:
        print(f"  suspicious: {v} is {v/med:.1f}x the median")
```

**Expected output.**

```text
6.1 Panel conversions
  glucose    126 mg/dL   -> 6.99293 mmol/L -> 6.99
  creatinine 1.4 mg/dL   -> 123.760 umol/L -> 124
  cholesterol 212 mg/dL  -> 5.4823 mmol/L -> 5.48
  triglycerides 180      -> 2.0322 mmol/L -> 2.03
  bilirubin  2.1 mg/dL   -> 35.91 umol/L -> 35.9
  calcium    9.2 mg/dL   -> 2.2954 mmol/L -> 2.30
  hemoglobin 13.4 g/dL   -> 134.0 g/L

6.2 Temperature and body size
  98.6 F -> 37.00 C -> 310.15 K
  198 lb -> 89.8113 kg -> 89.8
  71 in -> 180.34 cm
  BMI = 27.6151 -> 27.6 kg/m2

6.3 Blood pressure and MAP
  SBP 148 mmHg -> 19.732 kPa
  DBP 92 mmHg -> 12.266 kPa
  MAP = 110.67 mmHg -> 14.75 kPa

6.4 Uncertainty budget for glucose
  calibrator certificate     u = 1.2 mg/dL
  within-run repeatability   u = 1.8 mg/dL
  bias correction            u = 1.5 mg/dL
  combined u_c = 2.6325 mg/dL
  expanded U (k=2) = 5.2650 mg/dL
  interval = 120.74 to 131.26 mg/dL
  in mmol/L: u_c = 0.14610, U = 0.29220
  glucose interval in mmol/L = 6.701 to 7.285

6.5 CKD-EPI 2021 eGFR, male age 58, creatinine 1.4 mg/dL
  ratio = 1.5556
  eGFR = 58.2596 -> 58 mL/min/1.73 m2

6.6 Round trip
  126 -> 6.99293 -> 126.0000 mg/dL
  drift = 0.000000 mg/dL

6.7 Double-conversion detection
  median = 1.5
  suspicious: 123.8 is 82.5x the median
```

**Assumptions stated by the script.**

1. The glucose factor is the 18.0182 clinical convention. The script does not mix it with 18 or
   18.0156.
2. Components in the uncertainty budget are independent and are combined in quadrature. Shared
   calibrator uncertainty would require a covariance term.
3. The eGFR equation is the 2021 CKD-EPI creatinine equation for adults, without a race term. It is
   not valid for children, pregnancy, or dialysis patients.
4. The triglyceride factor is an approximation because the analyte is a mixture.
5. The defect detector assumes the majority of the column is in one unit. A column that is mostly
   converted would be flagged in the opposite direction, which is why the median must be established
   from source metadata before acting.

## 8. Data, standards, and regulatory context

### 8.1 The SI and its 2019 redefinition

The SI Brochure, 9th edition (BIPM, 2019, revised through version 4.01 in 2026), defines the system
by fixing seven constants: ΔνCs = 9,192,631,770 Hz, c = 299,792,458 m/s, h = 6.62607015 × 10⁻³⁴ J s,
e = 1.602176634 × 10⁻¹⁹ C, k = 1.380649 × 10⁻²³ J/K, N_A = 6.02214076 × 10²³ mol⁻¹, and
K_cd = 683 lm/W. The kilogram, ampere, kelvin, and mole are now realised from these constants rather
than from artefacts. For clinical data work the practical consequence is small: the units did not
change size. The documentation consequence is real, because a laboratory's traceability statement
must reference the current realisation.

NIST Special Publication 330, *The International System of Units (SI)*, and NIST Special Publication
811, *Guide for the Use of the International System of Units*, are the US implementations. SP 811 is
the practical reference for unit notation, prefix rules, and conversion conventions.

### 8.2 Machine-readable units

**UCUM** (Unified Code for Units of Measure, Regenstrief Institute) is the syntax used by HL7 FHIR
and recommended by LOINC. It is case-sensitive. `mg/dL` and `mmol/L` are valid; `mm[Hg]` uses
brackets because the unit is not metric; `10*3/uL` is the UCUM form for thousands per microlitre.
Annotations in braces, such as `mg{creat}`, carry no semantic value.

**LOINC** provides the observation code, not the unit. The same analyte has separate LOINC codes for
mass concentration and substance concentration: glucose in mg/dL is LOINC 2345-7 and in mmol/L is
14749-6; creatinine in mg/dL is 2160-0 and in µmol/L is 14682-9. A data model that stores a LOINC code
and a numeric value without a UCUM unit can still be ambiguous if the LOINC code is mapped
incorrectly, which is why both fields are required.

The NLM publishes a *Table of Example UCUM Codes for Electronic Messaging* that maps common
laboratory units to UCUM codes. It is the practical reference for interface work.

### 8.3 Laboratory and device standards

| Standard | Scope | Binding status |
|---|---|---|
| ISO 15197:2013 | Accuracy of self-monitoring blood glucose systems | Voluntary; required by many regulators and payers |
| ISO 80601-2-61 | Particular requirements for pulse oximeters | Voluntary; referenced in device approvals |
| ISO 5725 | Trueness and precision of measurement methods | Voluntary; used in accreditation |
| CLSI EP05-A3 | Precision evaluation for manufacturers | Voluntary; expected in submissions |
| CLSI EP15-A3 | User verification of precision | Voluntary; CLIA-accredited labs use it |
| CLSI EP28-A3c | Reference intervals | Voluntary; FDA-recognised consensus standard |
| JCGM 100:2008 (GUM) | Expression of uncertainty | Voluntary; referenced by accreditation bodies |
| IVDR (EU) 2017/746 | In vitro diagnostic regulation | Binding in the EU |
| CLIA (42 CFR 493) | US laboratory requirements | Binding in the US |

**ISO 15197:2013** requires that at least 95% of results fall within ±15 mg/dL of the reference at
concentrations below 100 mg/dL, and within ±15% at concentrations at or above 100 mg/dL. At least 99%
of results must fall within zones A and B of the consensus error grid. Each strip lot must meet the
criteria, and the evaluation requires three lots and at least 100 samples per lot across specified
glucose intervals. This is a metrological specification with a statistical acceptance rule, which is
why Topic 02 and this topic are read together.

**IVDR** requires metrological traceability of calibrators and control materials to a reference
measurement procedure or certified reference material, and requires the manufacturer to state the
measuring interval and the uncertainty. It is binding for products placed on the EU market.

**CLIA** requires laboratories to perform calibration and calibration verification, to establish
performance specifications for accuracy, precision, and reportable range, and to verify the
manufacturer's reference intervals or establish their own.

### 8.4 What the standards do not constrain

- They do not choose the canonical unit for a registry. That is a policy decision.
- They do not fix the number of significant figures for clinical display. The GUM gives principles.
- They do not resolve the glucose factor discrepancy between 18.0182 and 18.0156. Both are within the
  measurement uncertainty of every clinical assay.
- They do not make a non-traceable value comparable. A standard can require traceability; it cannot
  supply it retroactively.
- They do not cover every clinical unit. Many, including mmHg, IU, and mOsm/kg, are accepted for use
  but not defined by the SI.

> **Note:** The regulatory content here is educational and not legal advice. Device and laboratory
> compliance requires review by a qualified regulatory professional.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Clinical laboratories** maintain a metrological traceability chain, a daily quality control
programme, and periodic calibration verification. They report results with the method and the
reference interval, and they participate in external quality assessment, which compares their results
with peer laboratories on the same material. The external assessment is the operational test of
trueness; internal replicates test precision.

**Device manufacturers** establish accuracy specifications under standards such as ISO 15197 and
report them with the study population and the reference method. A specification quoted without the
population is incomplete, because accuracy can differ by haematocrit, altitude, temperature, and skin
pigmentation.

**Health systems and registries** maintain a unit dictionary that maps source units to canonical
units, with a version and an owner. Mature organisations treat the dictionary as a controlled
artefact, because a change to it changes every downstream report.

**Software vendors** that handle clinical data are increasingly required to demonstrate unit handling
in their validation documentation. A vendor that stores a naked float and infers the unit from the
range will fail a metrology review, because inference is not traceability.

### 9.2 Case evidence

**The 2019 SI redefinition.** On 20 May 2019 the kilogram, ampere, kelvin, and mole were redefined in
terms of fixed constants. The change was coordinated by the BIPM and implemented through the CGPM
resolution. Documented consequence for laboratories: the practical realisation changed, and
traceability statements were updated, even though the unit sizes were unchanged within the existing
uncertainties.

**Pulse oximetry and skin pigmentation.** Sjoding et al. (*New England Journal of Medicine* 383,
2477–2478, 2020) reported that occult hypoxaemia, defined as arterial oxygen saturation below 88%
despite a pulse oximeter reading of 92% to 96%, was roughly three times more frequent in Black patients
than in White patients in a large clinical dataset. The FDA issued a safety communication in 2021
noting that pulse oximeter accuracy can be affected by skin pigmentation and other factors and
advising clinicians to be aware of the limitation. This is a documented case of a measurement being
accurate on average and biased in a subgroup, which is exactly the distinction between precision and
trueness in section 3.3.

**Blood glucose meter accuracy.** ISO 15197:2013 tightened the acceptance criterion from the 2003
edition: ±15 mg/dL below 100 mg/dL and ±15% at or above 100 mg/dL for 95% of results, plus 99% within
consensus error grid zones A and B. The change was driven by evidence that earlier meters could be
off by 20% at high glucose, which matters for insulin dosing. The standard is a documented example of
a metrological specification written as a statistical acceptance rule.

**Glucose conversion in diabetes software.** The long-standing clinical factor is 18.0182 mg/dL per
mmol/L. The IUPAC molar mass of anhydrous D-glucose, 180.156 g/mol, implies 18.0156. Open-source
diabetes projects moved to 18.01559: Nightscout in 2025 and xDrip in a 2026 nightly release, as
documented in a public issue thread. The discrepancy is about 0.015% and clinically irrelevant at
typical glucose values, but it produces visible mismatches between apps displaying the same underlying
mg/dL value. The lesson is not that one factor is right; it is that the factor version must be stored
with the value.

**The Mars Climate Orbiter.** In 1999 the orbiter was lost after a unit mismatch between pound-force
seconds and newton seconds in navigation software. The failure is not biomedical, and it is included
only as a documented analogue: the interface contract did not carry units, and the error survived
review because each side's number was plausible in its own unit. The biomedical equivalent is the
naked numeric field with an assumed unit.

> **Note:** The pulse oximetry and glucose conversion cases are cited to specific publications and
> public issue threads. The interpretation that these are metrological failures, rather than isolated
> software defects, is inference from the cited material.

### 9.3 The practical pattern

Organisations that avoid unit failures do four things:

1. They store value, unit, and provenance as separate fields.
2. They own a versioned unit dictionary with a named maintainer.
3. They run plausibility checks in the canonical unit, after conversion.
4. They never convert twice.

The cost of these practices is small and front-loaded. The cost of the alternative is a class of error
that is invisible until it affects a patient or an audit.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Unit missing from the data model | Column named `glucose` with float values | Unit treated as documentation rather than data | Silent 18× or 88× misread | Store value, UCUM unit, and system as separate fields |
| 2 | Unit inferred from the value range | Code branches on "if value > 30 then mg/dL" | Range overlap between units | Misclassification at the boundary | Never infer; require the unit or quarantine |
| 3 | Mixed units in one column | Two clusters separated by a factor of 18 or 88 | Interface merge of two source systems | Aggregate statistics are meaningless | Split by unit, convert with provenance |
| 4 | Double conversion | Values are exactly one factor away from expected | Conversion applied on write and again on read | Values off by 18× or 88× | Convert once; store raw and canonical |
| 5 | Rounded factor used as exact | 18 instead of 18.0182; 0.0555 instead of 1/18.0182 | Memorised rule of thumb | Small bias that accumulates in totals | Versioned factor table with basis documented |
| 6 | Multiplicative conversion of an offset scale | 98.6 °F becomes 54.8 °C | Missing the 32 offset | Physiologically impossible temperature | Affine converters; unit tests for temperature |
| 7 | Mass conversion of an international unit | "10,000 IU = 10 mg" | IU defined by biological activity | Dose error, potentially fatal | Refuse mass conversion; cite the preparation |
| 8 | eGFR computed from µmol/L creatinine | eGFR near zero | Equation expects mg/dL | CKD misclassification | Convert before the equation; assert the input unit |
| 9 | Reference interval applied to the wrong partition | Creatinine flagged normal in a child | Interval from adults | Missed disease | Partition by age, sex, pregnancy; verify transferred intervals |
| 10 | Decision limit applied without a rule | Result exactly at the threshold | Uncertainty ignored at the boundary | Non-reproducible conformity decisions | State the decision rule and guard band before analysis |
| 11 | Calibration drift undetected | QC within limits for months, then a shift | Limits derived from the drifted baseline | A quarter of biased results | Independent reference material; periodic calibration verification |
| 12 | Hemolysed sample reported as valid | Potassium high with a normal creatinine | Red cell lysis releases intracellular analytes | False hyperkalaemia | Check the hemolysis index; suppress per policy |
| 13 | Delayed glucose sample | Glucose lower than the clinical picture suggests | Glycolysis in an unpreserved tube | False hypoglycaemia | Citrate or fluoride tube; document draw time |
| 14 | Unit string parsed by fuzzy match | `mg/dl` and `MG/DL` both accepted | Case-insensitive matching hides typos | Wrong mapping for a genuinely different unit | Explicit mapping table; reject unmapped codes |
| 15 | Device accuracy assumed constant across subgroups | Specification quoted without population | Bias varies with skin tone, haematocrit, altitude | Inequitable measurement error | Stratified validation; report subgroup performance |
| 16 | Uncertainty omitted from the result | A point value with no interval | Uncertainty treated as academic | Overconfident threshold decisions | Report U with k; store the budget |

Two rows warrant a longer note.

**Row 3, mixed units.** The statistical signature is bimodal. A histogram of a creatinine column with
values around 1.2 and around 106 has two modes separated by roughly 88. A histogram of glucose with
modes at 110 and 6.1 has modes separated by roughly 18. These are easy to detect with a plot and
almost impossible to detect with a mean.

**Row 5, rounded factors.** The error from using 18 instead of 18.0182 is 0.1% per conversion. That is
negligible for a single result and not negligible for a population total or a longitudinal trend: a
0.1% bias applied to a 5 mmol/L threshold shifts the classified fraction by a small but systematic
amount. More importantly, mixing 18 and 18.0182 across systems produces a discrepancy that a
reconciliation exercise will spend days explaining.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The measurand is the one the unit describes | A unit is meaningless without its quantity | Glucose in whole blood versus plasma | Read the method and specimen fields |
| 2 | The calibration is current and traceable | Comparability across time and labs | Expired calibrator, broken chain | Calibration record audit; certificate review |
| 3 | The sample is representative of the patient at the time | The result describes a physiological state | Delay, hemolysis, wrong tube, IV contamination | Pre-analytical checks; draw-site policy |
| 4 | The conversion factor is correct for the analyte and convention | Conversion changes the number | Using 18 for a registry that expects 18.0182 | Versioned factor table; round-trip test |
| 5 | The uncertainty components are independent | Quadrature combination | Shared calibrator between components | Covariance analysis; sensitivity check |
| 6 | The uncertainty is approximately normal at k = 2 | Expanded uncertainty coverage | Strongly skewed or bounded distributions | Compare with a Monte Carlo interval |
| 7 | The reference interval fits the patient | Interpretation of normal versus abnormal | Age, sex, pregnancy, assay differences | Age- and sex-specific intervals; local verification |
| 8 | The device performs in the deployment population as in the validation population | Transportability of accuracy claims | Skin pigmentation, haematocrit, altitude, motion | Stratified validation in the target population |
| 9 | The unit code is semantically correct, not merely syntactically valid | UCUM validity does not imply the right unit | `mg/dL` used for a substance concentration | Cross-check against the LOINC property |
| 10 | The result is not a composite of several measurements | A single value describes one measurand | Averaged blood pressures, calculated eGFR | Store the components and the formula |
| 11 | Rounding is applied once | Repeated rounding drifts | Display rounding in intermediate tables | Carry full precision; round at the end |
| 12 | The conversion table version is recorded | Retrospective audit and correction | Factor changed without versioning | Immutable conversion log |
| 13 | Biological units have a stable reference preparation | IU comparability across products | A change in the WHO preparation | Cite the preparation and the assay |
| 14 | The instrument's measuring interval covers the value | Extrapolation beyond the calibrated range | A result above the analytical measuring range | Check the reportable range; dilution protocols |

The limits of this topic:

- It does not cover the physics of any specific sensor. That is Topic 08.
- It does not cover the statistics of method comparison (regression, Bland-Altman) in depth. Those are
  referenced in Topic 02 and applied in the labs.
- It does not cover regulatory submission strategy. It covers the metrological content a submission
  must contain.
- It does not resolve which canonical unit an organisation should choose. That is a governance
  decision.
- It does not cover every clinical unit. It covers the classes of unit problem that recur.

## 12. Exercises

### Exercise 1 — SI structure (recall)

List the seven SI base units with their symbols and quantities. Name five derived units used in
medicine or physiology, state their base-unit expansion, and identify which are coherent and which
carry a numerical factor.

**Deliverable.** Two tables.

**Acceptance criteria.** All seven base units correct; five derived units with correct expansions (for
example, the pascal is kg·m⁻¹·s⁻²); mmHg and cmH₂O identified as non-coherent; the 2019 defining
constants named at least in outline.

**Time.** 25 minutes.

### Exercise 2 — Convert a panel (calculation)

Convert the following to the stated SI unit, showing the factor and the arithmetic: glucose 89 mg/dL to
mmol/L; creatinine 0.9 mg/dL to µmol/L; total cholesterol 180 mg/dL to mmol/L; total bilirubin
1.2 mg/dL to µmol/L; calcium 10.1 mg/dL to mmol/L; haemoglobin 11.8 g/dL to g/L; temperature 100.4 °F
to °C; weight 165 lb to kg; height 64 in to cm; systolic pressure 128 mmHg to kPa.

**Deliverable.** A table with input, factor, arithmetic, output, and reported value.

**Acceptance criteria.** Glucose 4.94 mmol/L; creatinine 79.6 µmol/L; cholesterol 4.65 mmol/L;
bilirubin 20.5 µmol/L; calcium 2.52 mmol/L; haemoglobin 118 g/L; temperature 38.0 °C; weight 74.8 kg;
height 162.6 cm; systolic 17.07 kPa. Each output rounded consistently with the input precision.

**Time.** 40 minutes.

### Exercise 3 — Derive the factor (critique and calculation)

The molar mass of anhydrous D-glucose is 180.156 g/mol. Derive the mg/dL to mmol/L conversion factor
from first principles, compare it with 18.0182 and with 18, and state at what glucose concentration
the difference between 18 and 18.0182 would exceed a 1 mg/dL rounding difference.

**Deliverable.** A derivation of no more than 250 words with the arithmetic.

**Acceptance criteria.** Derivation shows 1 mmol/L = 180.156 mg/L = 18.0156 mg/dL; the comparison
quantifies the 0.015% difference; the threshold calculation is correct; the conclusion states which
factor the program uses and why.

**Time.** 25 minutes.

### Exercise 4 — Accuracy and precision (critique)

Two glucose meters are evaluated against a reference method. Meter A reads 102, 101, 103, 102, 102
mg/dL when the reference is 110 mg/dL. Meter B reads 108, 113, 109, 112, 108 mg/dL for the same
reference.

Compute the mean, the bias, and the standard deviation for each. State which meter is more precise,
which is more accurate, and which you would prefer for a threshold decision at 100 mg/dL.

**Deliverable.** A table with mean, bias, SD, and a recommendation.

**Acceptance criteria.** Meter A mean 102, bias −8, SD 0.71; Meter B mean 110, bias 0, SD 2.24; the
recommendation states that Meter B is more accurate and Meter A more precise, and justifies the choice
for a threshold decision.

**Time.** 30 minutes.

### Exercise 5 — Uncertainty budget (calculation)

A creatinine assay has three uncertainty components: calibrator certificate 0.02 mg/dL (Type B),
within-run repeatability 0.03 mg/dL (Type A), and bias correction 0.015 mg/dL (Type B). The result is
1.10 mg/dL.

Compute the combined standard uncertainty, the expanded uncertainty at k = 2, and the interval in both
mg/dL and µmol/L.

**Deliverable.** A budget table and the converted interval.

**Acceptance criteria.** u_c = 0.0391 mg/dL; U = 0.0781 mg/dL; interval 1.02 to 1.18 mg/dL; converted
interval 90.3 to 104.3 µmol/L; the budget states the independence assumption.

**Time.** 30 minutes.

### Exercise 6 — Significant figures and dimensionless quantities (critique)

A report states: "pH 7.38, a 0.12 unit increase from 7.26, representing a 32% increase in acidity.
Serum creatinine 1.10 mg/dL, eGFR 58.2596 mL/min/1.73 m², INR 2.345."

Identify every reporting defect and rewrite the sentence.

**Deliverable.** A marked-up sentence plus a corrected version.

**Acceptance criteria.** pH change identified as a factor of 1.32 in hydrogen ion activity, not a 32%
increase in "acidity" expressed as a percentage of pH units; eGFR rounded to a whole number; INR to
one decimal; pH to two decimals; the corrected sentence uses dimensionless labels correctly.

**Time.** 30 minutes.

### Exercise 7 — Design a unit policy (design)

Design a unit-handling policy for a registry that ingests laboratory data from 12 hospitals in three
countries. Specify: the canonical unit per analyte for at least eight analytes; the storage schema;
the conversion versioning scheme; the plausibility rules; and the handling of unmapped units.

**Deliverable.** A two-page policy with a schema diagram and a rules table.

**Acceptance criteria.** Schema stores raw value, raw unit, canonical value, canonical unit, factor
version, and method; at least eight analytes with justified canonical units; plausibility bounds are
unit-specific and numeric; unmapped units are quarantined, not guessed; the policy names an owner and
a review cadence.

**Time.** 60 minutes.

### Exercise 8 — Audit a dataset (application)

Using a public laboratory dataset or a synthetic extract, produce a unit-defect report: count of
missing units, count of columns with more than one unit, list of values that are outliers by a factor
of 10 or more from the column median, and a recommendation for each defect class.

**Deliverable.** A defect table with counts and a one-page recommendation.

**Acceptance criteria.** Each defect class has a count and an example row; the factor-outlier check is
implemented, not eyeballed; recommendations distinguish quarantine, conversion, and rejection; no row
is silently deleted.

**Time.** 45 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Quantity | Unit | Symbol |
|---|---|---|
| Length | metre | m |
| Mass | kilogram | kg |
| Time | second | s |
| Electric current | ampere | A |
| Thermodynamic temperature | kelvin | K |
| Amount of substance | mole | mol |
| Luminous intensity | candela | cd |

| Derived unit | Quantity | Base expansion | Coherent |
|---|---|---|---|
| pascal (Pa) | pressure | kg·m⁻¹·s⁻² | Yes |
| joule (J) | energy | kg·m²·s⁻² | Yes |
| watt (W) | power | kg·m²·s⁻³ | Yes |
| becquerel (Bq) | activity | s⁻¹ | Yes |
| katal (kat) | catalytic activity | mol·s⁻¹ | Yes |
| mmHg | pressure | 133.322 Pa | No |
| cmH₂O | pressure | 98.0665 Pa | No |

The coherent derived units have no numerical factor other than 1. mmHg and cmH₂O carry a numerical
factor and are accepted for use with the SI but are not coherent.

### Answer to Exercise 2

```text
Glucose        89 / 18.0182        = 4.9395  -> 4.94 mmol/L
Creatinine     0.9 x 88.4          = 79.56   -> 79.6 umol/L
Cholesterol    180 x 0.02586       = 4.6548  -> 4.65 mmol/L
Bilirubin      1.2 x 17.1          = 20.52   -> 20.5 umol/L
Calcium        10.1 x 0.2495       = 2.51995 -> 2.52 mmol/L
Haemoglobin    11.8 x 10           = 118     -> 118 g/L
Temperature    (100.4 - 32) x 5/9  = 38.00   -> 38.0 C
Weight         165 x 0.45359237    = 74.8427 -> 74.8 kg
Height         64 x 2.54           = 162.56  -> 162.6 cm
Systolic       128 x 0.133322      = 17.065  -> 17.07 kPa
```

Each output is rounded to the precision of the input. The glucose input has two significant figures,
so 4.94 is already generous; 4.9 would also be defensible. The temperature output is reported to one
decimal because the Fahrenheit input has one decimal and the conversion preserves that resolution.

### Answer to Exercise 3

```text
1 mol of glucose = 180.156 g
1 mmol            = 180.156 mg

1 mmol/L = 180.156 mg/L
         = 18.0156 mg/dL

Therefore mg/dL -> mmol/L divides by 18.0156.

The clinical convention uses 18.0182, which corresponds to a molar mass of 180.182 g/mol.
Difference from the molar-mass value: (18.0182 - 18.0156)/18.0156 = 0.000144 = 0.014%.

Difference between 18 and 18.0182:
  at 100 mg/dL: 100/18 = 5.5556; 100/18.0182 = 5.5499; difference 0.0057 mmol/L
  at 500 mg/dL: 500/18 = 27.7778; 500/18.0182 = 27.7497; difference 0.0281 mmol/L

A 1 mg/dL rounding difference in the mg/dL value corresponds to about 0.055 mmol/L.
The factor difference exceeds that at roughly 1,000 mg/dL, which is outside the clinical range.
```

The program uses 18.0182 because it is the published clinical convention and because registries and
device vendors expect it. The molar-mass value is more physically defensible and is used by some
software; the difference is far below assay uncertainty.

### Answer to Exercise 4

```text
Meter A: mean = (102+101+103+102+102)/5 = 510/5 = 102.0 mg/dL
         bias = 102.0 - 110 = -8.0 mg/dL
         SD   = sqrt(((0)^2+(1)^2+(1)^2+(0)^2+(0)^2)/4) = sqrt(2/4) = 0.707 mg/dL

Meter B: mean = (108+113+109+112+108)/5 = 550/5 = 110.0 mg/dL
         bias = 110.0 - 110 = 0.0 mg/dL
         SD   = sqrt((4+9+1+4+4)/4) = sqrt(22/4) = 2.345 mg/dL
```

Meter A is more precise (SD 0.71 versus 2.35) and less accurate (bias −8 versus 0). Meter B is more
accurate and less precise.

For a threshold decision at 100 mg/dL, Meter B is preferable because a bias of −8 mg/dL would classify
a true 105 mg/dL as 97 and call it normal, while Meter B's random error of about 2.3 mg/dL is less
likely to cross the threshold systematically. Precision without accuracy is consistently wrong;
accuracy without precision is unreliably right, and for a threshold the systematic error is the more
dangerous of the two.

### Answer to Exercise 5

```text
u_c = sqrt(0.02^2 + 0.03^2 + 0.015^2)
    = sqrt(0.0004 + 0.0009 + 0.000225)
    = sqrt(0.001525)
    = 0.03905 mg/dL

U = 2 x 0.03905 = 0.07810 mg/dL

Interval in mg/dL: 1.10 - 0.078 = 1.022 to 1.178 -> 1.02 to 1.18 mg/dL

Converted: 1.022 x 88.4 = 90.34 umol/L
           1.178 x 88.4 = 104.14 umol/L
Interval in umol/L: 90.3 to 104.1 umol/L
```

The combination assumes the three components are independent. If the bias correction was derived
using the same calibrator as the certificate, the two are correlated and the combined uncertainty is
larger than the quadrature result.

### Answer to Exercise 6

Defects:

1. "0.12 unit increase" is correct as an arithmetic difference in pH, but "32% increase in acidity"
   is wrong. A pH fall of 0.12 corresponds to an increase in hydrogen ion activity by a factor of
   10^0.12 = 1.318, or 31.8%. The sentence describes a pH rise from 7.26 to 7.38, which is a
   decrease in hydrogen ion activity, not an increase.
2. eGFR 58.2596 has four decimal places; the equation's uncertainty does not support more than a
   whole number.
3. INR 2.345 has three decimal places; clinical INR is reported to one decimal place, occasionally
   two.
4. pH is dimensionless but is correctly reported to two decimals.

Corrected: "pH rose from 7.26 to 7.38, a 0.12 unit increase corresponding to a 24% decrease in
hydrogen ion activity (a factor of 0.758). eGFR 58 mL/min/1.73 m². INR 2.3."

Check the factor: 10^(7.26−7.38) = 10^−0.12 = 0.759, so activity fell by 24.1%. The original claim of a
32% increase in the wrong direction is a compound error: wrong direction and wrong arithmetic.

### Answer to Exercise 7 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Canonical units | 20% | At least eight analytes, each with a justified canonical unit and a UCUM code |
| Schema | 25% | Raw value, raw unit, canonical value, canonical unit, factor version, method, timestamp |
| Versioning | 15% | Immutable factor table with effective dates and a change procedure |
| Plausibility rules | 20% | Unit-specific numeric bounds and a quarantine route |
| Unmapped units | 10% | Explicit quarantine, no inference from range |
| Governance | 10% | Named owner, review cadence, and audit trail |

A weak policy lists conversion factors without a schema or a versioning scheme, and handles unmapped
units by guessing from the value range.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Completeness | 25% | All four defect classes counted, with examples |
| Method | 25% | Factor-outlier check implemented as code, not visual inspection |
| Interpretation | 20% | Distinguishes double conversion from a genuine high value |
| Recommendations | 20% | Quarantine, convert, or reject per class, with a rationale |
| Preservation | 10% | No silent deletion; evidence retained |

A weak audit deletes outliers and reports a clean dataset, which destroys the evidence of the defect
and biases every downstream estimate.

## 14. Interview questions

### Question 1 — "What is the difference between accuracy and precision?"

**Strong answer.** "Accuracy is closeness to the true value; precision is closeness among repeats.
ISO 5725 separates trueness, which is bias, from precision, which is spread. A meter can be precise
and biased, which is consistently wrong, or accurate on average and imprecise, which is unreliably
right. They require different experiments: trueness needs a reference method, precision needs
replicates."

**Weak answer.** "Accuracy is how close you are, precision is how many decimal places." The second
half confuses precision with resolution or significant figures.

### Question 2 — "Why does a laboratory result need a unit and an uncertainty?"

**Strong answer.** "Because a value alone is not a measurement. The unit fixes the scale and makes the
value comparable; the uncertainty fixes the resolution and makes a threshold decision reproducible.
Without the unit, 126 could be mg/dL or mmol/L; without the uncertainty, a result at the threshold
cannot be classified consistently."

**Weak answer.** "Because the report looks more professional." The requirement is interpretive, not
cosmetic.

### Question 3 — "A creatinine column has values around 1.3 and one value of 115. What do you do?"

**Strong answer.** "Check the unit metadata first. A value about 88 times the median is a value in
µmol/L sitting in a mg/dL column. I would quarantine the row, confirm with the source, and convert
with provenance. I would not delete it, because deletion destroys the evidence, and I would not convert
the whole column, because that would corrupt the correct values."

**Weak answer.** "Remove the outlier." That destroys the evidence and hides a systematic interface
defect that probably affects more rows.

### Question 4 — "Why can you not convert international units to milligrams?"

**Strong answer.** "Because an IU is defined by the biological activity of a specific WHO reference
preparation, not by mass. The mass equivalent differs by analyte and can change when the preparation
changes. Converting without naming the preparation fabricates a number."

**Weak answer.** "You can, with the right factor." There is no universal factor; there is a
preparation-specific one, and it must be cited.

### Question 5 — "How do you combine uncertainty components?"

**Strong answer.** "Convert each to a standard uncertainty, then combine independent components in
quadrature: u_c = √(Σu_i²). Multiply by the coverage factor for the expanded uncertainty, usually
k = 2 for about 95%. If components are correlated, I need the covariance, and if the function is
strongly nonlinear I would use Monte Carlo."

**Weak answer.** "Add them up." Linear addition overstates the combined uncertainty when the
components are independent.

### Question 6 — "What is the difference between calibration and verification?"

**Strong answer.** "Calibration establishes the relationship between instrument response and reference
values. Verification confirms that the calibrated instrument meets a specification. Adjustment changes
the instrument to match a reference and must be followed by verification. Auditors distinguish these
because calling an adjustment a calibration hides an intervention."

**Weak answer.** "They are the same thing." The distinction determines what is documented and what is
audited.

### Question 7 — "Why is 18.0182 used instead of 18 for glucose?"

**Strong answer.** "18.0182 is the published clinical convention for mg/dL per mmol/L. The molar mass
of glucose, 180.156 g/mol, implies 18.0156, and 18 is a rounded rule of thumb. The differences are far
below assay uncertainty, but the convention matters for interoperability, so I would store the factor
version with the value."

**Weak answer.** "18 is close enough." Close enough for a single display, not for a registry that
reconciles two systems.

### Question 8 — "How would you design a unit policy for a multi-site registry?"

**Strong answer.** "One canonical unit per analyte, stored alongside the raw value and unit, with a
versioned factor table and a named owner. Plausibility bounds in the canonical unit, run after
conversion. Unmapped units are quarantined rather than inferred. Every row records the factor version
so a correction can be applied retrospectively."

**Weak answer.** "Convert everything to SI on ingest." That is the first step, not the policy: it
omits provenance, versioning, and quarantine.

### Question 9 — "Why is eGFR reported in mL/min/1.73 m²?"

**Strong answer.** "Because it is indexed to a standard body surface area so that values are
comparable across people of different sizes. The unit is not a plain flow rate; it is a flow rate
normalised to 1.73 m². Reporting it as mL/min without the index changes the meaning."

**Weak answer.** "It is the same as creatinine clearance." Creatinine clearance is a measured quantity
with a urine collection; eGFR is an estimate from a fitted equation.

### Question 10 — "A device specification says accurate to ±15%. What is missing?"

**Strong answer.** "The population, the reference method, the measurand, the concentration range, and
whether ±15% is a bias bound, a repeatability bound, or an expanded uncertainty. ISO 15197, for
example, specifies ±15 mg/dL below 100 mg/dL and ±15% at or above 100 mg/dL for 95% of results. A bare
±15% does not tell me which of these it is."

**Weak answer.** "Nothing; ±15% is a standard specification." The number is uninterpretable without its
scope and its statistical basis.

### Question 11 — "How do you handle a temperature recorded as 98.6 with no unit?"

**Strong answer.** "I do not infer from the value alone, although 98.6 is almost certainly Fahrenheit.
I check the source field or the device configuration. If I cannot confirm, I quarantine the value,
because a multiplicative conversion of a Fahrenheit value to Celsius would produce 54.8 °C, which is
lethal and obviously wrong, but a mislabelled 37.0 could pass silently."

**Weak answer.** "It is Fahrenheit because 98.6 is the normal human temperature." That is a plausible
inference, and inference is not traceability.

### Question 12 — "What is the metrological problem with pulse oximetry?"

**Strong answer.** "The device estimates arterial oxygen saturation from optical measurements, and the
calibration curve was derived in a validation population. Bias varies with skin pigmentation,
haematocrit, perfusion, and motion. A device can meet its specification on average and be biased in a
subgroup, which is a trueness problem, not a precision problem, and it requires stratified validation
to detect."

**Weak answer.** "Pulse oximeters are inaccurate." They are accurate enough for many uses and biased
for some populations; the distinction determines the mitigation.

## 15. Key terms

- **Accuracy** — closeness of a measured value to the true value.
- **Adjustment** — changing an instrument so its output matches a reference.
- **Bias** — systematic difference between the expected measurement and the true value.
- **Calibration** — establishing the relationship between instrument response and reference values.
- **CLIA** — Clinical Laboratory Improvement Amendments; US laboratory regulation.
- **Coherent derived unit** — a derived unit with no numerical factor other than 1.
- **Combined standard uncertainty (u_c)** — root-sum-square of independent standard components.
- **Conformity assessment** — deciding whether a result lies within a specification.
- **Conventional unit** — a non-SI unit still used clinically, such as mg/dL.
- **Coverage factor (k)** — multiplier applied to u_c to obtain the expanded uncertainty.
- **Decision rule** — the rule that maps a result and its uncertainty to accept, reject, or
  indeterminate.
- **Defining constant** — a physical constant with a fixed numerical value under the 2019 SI.
- **Derived unit** — a unit formed from base units by multiplication or division.
- **Drift** — change in instrument response over time at a fixed measurand.
- **eGFR** — estimated glomerular filtration rate, indexed to 1.73 m².
- **Enzyme unit (U)** — 1 µmol of substrate converted per minute.
- **Expanded uncertainty (U)** — u_c multiplied by the coverage factor.
- **Guard band** — a margin subtracted from a tolerance to account for uncertainty.
- **Hysteresis** — direction-dependent response of an instrument.
- **Intermediate precision** — precision under varied conditions within one laboratory.
- **International unit (IU)** — a unit defined by the biological activity of a WHO reference
  preparation.
- **INR** — international normalised ratio; a dimensionless prothrombin time ratio.
- **IVDR** — In Vitro Diagnostic Regulation (EU) 2017/746.
- **Limit of detection (LOD)** — the lowest quantity reliably distinguished from blank.
- **Limit of quantification (LOQ)** — the lowest quantity measurable with stated precision and
  trueness.
- **Linearity** — proportionality of response across the measuring interval.
- **LOINC** — a terminology for laboratory and clinical observations.
- **Measurand** — the quantity intended to be measured.
- **Metrology** — the science of measurement.
- **Molar mass** — mass of one mole of a substance, in g/mol.
- **Osmolality** — concentration of osmotically active particles per kilogram of solvent.
- **pH** — negative decadic logarithm of hydrogen ion activity.
- **Precision** — closeness of agreement among repeated measurements.
- **Repeatability** — precision under identical conditions.
- **Reproducibility** — precision across laboratories, operators, and instruments.
- **Resolution** — the smallest change an instrument can distinguish.
- **SI** — the International System of Units.
- **Standard uncertainty (u)** — uncertainty of a component expressed as a standard deviation.
- **Tolerance** — the permitted difference between a measured value and a specification.
- **Traceability** — an unbroken chain of calibrations to a reference, with stated uncertainty.
- **Trueness** — closeness of the mean of many measurements to the true value.
- **Type A uncertainty** — evaluated statistically from repeated observations.
- **Type B uncertainty** — evaluated from certificates, specifications, or judgement.
- **UCUM** — Unified Code for Units of Measure.
- **Verification** — confirming that a calibrated system meets a specification.

## 16. Sources

All URLs were accessed on 24 October 2026.

### SI, metrology, and uncertainty

- BIPM, *The International System of Units (SI Brochure)*, 9th edition, 2019, revised version 4.01,
  2026. https://www.bipm.org/en/publications/si-brochure — DOI 10.59161/AUEZ1291.
- BIPM, *SI Brochure 9th edition, concise summary*.
  https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-concise-EN.pdf
- NIST, *Guide for the Use of the International System of Units (SI)*, NIST SP 811.
  https://www.nist.gov/pml/special-publication-811
- NIST, *The International System of Units (SI)*, NIST SP 330.
  https://www.nist.gov/pml/special-publication-330
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in
  measurement*. https://www.bipm.org/en/committees/jc/jcgm/publications
- ISO 5725-1:1994, *Accuracy (trueness and precision) of measurement methods and results*.
  https://www.iso.org/standards.html

### Units in clinical data

- Regenstrief Institute, *Unified Code for Units of Measure*. https://ucum.org/
- UCUM, *Common UCUM units*. https://ucum.org/docs/common-units
- National Library of Medicine, *Table of Example UCUM Codes for Electronic Messaging*, version 1.4,
  2016. https://ucum.nlm.nih.gov/example-UCUM-Codes-v1.4.pdf
- LOINC, *Logical Observation Identifiers Names and Codes*. https://loinc.org/
- HL7, *FHIR ValueSet ucum-common*. https://hl7.org/FHIR/R5/valueset-ucum-common.html
- McDonald, C. J. et al., "LOINC, a universal standard for identifying laboratory observations",
  *Clinical Chemistry* 49(4), 624–633, 2003. https://doi.org/10.1373/49.4.624

### Laboratory and device standards

- CLSI EP28-A3c, *Defining, Establishing, and Verifying Reference Intervals in the Clinical
  Laboratory; Approved Guideline—Third Edition*, 2010.
  https://clsi.org/standards/products/method-evaluation/documents/ep28
- CLSI EP05-A3, *Evaluation of Precision of Quantitative Measurement Procedures*, 2014.
  https://clsi.org/standards/products/method-evaluation/documents/ep05/
- ISO 15197:2013, *In vitro diagnostic test systems — Requirements for blood-glucose monitoring
  systems for self-testing in managing diabetes mellitus*.
  https://www.iso.org/standard/54976.html
- ISO 80601-2-61:2017, *Medical electrical equipment — Particular requirements for the basic safety
  and essential performance of pulse oximeter equipment*. https://www.iso.org/standards.html
- Regulation (EU) 2017/746 on in vitro diagnostic medical devices (IVDR).
  https://eur-lex.europa.eu/eli/reg/2017/746/oj
- Centers for Medicare & Medicaid Services, *Clinical Laboratory Improvement Amendments (CLIA)*,
  42 CFR Part 493. https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments

### Clinical evidence and equations

- Inker, L. A. et al., "New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race",
  *New England Journal of Medicine* 385, 1737–1749, 2021. https://doi.org/10.1056/NEJMoa2102953
- Sjoding, M. W. et al., "Racial Bias in Pulse Oximetry Measurement", *New England Journal of
  Medicine* 383, 2477–2478, 2020. https://doi.org/10.1056/NEJMc2029240
- US Food and Drug Administration, *Safety Communication: Pulse Oximeter Accuracy and Limitations*,
  2021 (updated). https://www.fda.gov/medical-devices/safety-communications/pulse-oximeter-accuracy-and-limitations-fda-safety-communication
- Nightscout Foundation, issue #4928, "Alignment of Glucose Conversion Factor".
  https://github.com/nightscout/AndroidAPS/issues/4928
- NASA, *Mars Climate Orbiter Mishap Investigation Board Phase I Report*, 1999.
  https://solarsystem.nasa.gov/resources/776/mars-climate-orbiter-mishap-investigation-board-phase-i-report/

> **Note:** CLSI and ISO documents are sold standards; the URLs are product pages. Summaries here are
> from publicly available front matter and preview material and should be checked against the
> purchased document before use in a regulated process.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`02_probability-and-statistics-basics.md`](02_probability-and-statistics-basics.md)
- Next topic: [`04_dimensional-analysis.md`](04_dimensional-analysis.md)
- Mathematical basis of ratios and logarithms: [`01_mathematical-basics.md`](01-mathematical-basics.md)
- Device measurement chain: [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md)
- Assumption testing: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `01A_unit_conversion.ipynb` is the conversion lab; `14B_quantization.ipynb`
  covers resolution and quantisation in sampled data.
