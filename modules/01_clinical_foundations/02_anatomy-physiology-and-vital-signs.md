# Anatomy, Physiology, and Vital Signs

> Module 01 — Clinical Foundations for Data Scientists
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: 01
> Version 2026.3

---

## 1. Why this topic exists

Vital signs are the most frequently recorded physiological data in the health system and the most
frequently misused. They are cheap, repeated, and apparently simple: a number with a unit. That
appearance is deceptive. Each vital sign is a measurement produced by a specific method, in a
specific position, at a specific moment, under specific physiological conditions. When those
conditions change, the number changes for reasons that have nothing to do with the disease you
are modelling.

Three industry problems follow.

**Problem 1: the feature is not the physiology.** A systolic blood pressure of 88 mmHg may be
shock, or it may be a cuff on the wrong arm in a patient who has just stood up. A respiratory
rate of 16 breaths/min may be a measured value, a default entered by a nurse at the end of a
shift, or an imputed value carried forward from triage. If your model treats all three as the
same quantity, it learns the charting process.

**Problem 2: compensation hides severity.** Physiology defends itself. A young patient can lose
30% of blood volume and maintain a normal blood pressure until the compensation fails abruptly.
A model trained on blood pressure alone will call that patient stable. The variables that reveal
compensation — heart rate, shock index, lactate, urine output, capillary refill — are either
measured less often or not recorded at all.

**Problem 3: normal ranges are population-specific.** Adult ranges do not apply to children,
pregnant patients, or people with chronic disease. A creatinine of 1.2 mg/dL is normal in a
muscular man and abnormal in a frail 85-year-old woman. A heart rate of 110 beats/min is
tachycardia in a resting adult and normal in a 2-year-old. Hard-coding one range produces
systematic error that is correlated with age, sex, and pregnancy status, which means it produces
bias.

This topic gives you the physiology needed to look at a vital sign and ask: what is this
measuring, what would make it wrong, and what would the body do next?

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** the major organ systems and the vital sign or laboratory variable most directly
   associated with each.
2. **State** adult, pediatric, and pregnancy-specific ranges for heart rate, blood pressure,
   respiratory rate, temperature, and oxygen saturation, with the population and source named.
3. **Compute** mean arterial pressure (MAP), shock index, cardiac output, cardiac index, arterial
   oxygen content, oxygen delivery, oxygen consumption, and oxygen extraction ratio.
4. **Explain** the compensatory response to hemorrhage, sepsis, and heart failure, and **predict**
   which variables change first.
5. **Critique** a vital-sign feature for measurement method, position, device, and missingness.
6. **Distinguish** hypotension from shock, and **explain** why a normal blood pressure does not
   exclude inadequate perfusion.
7. **Design** a plausibility rule set for vital signs that catches unit errors, default values,
   and impossible trajectories.

## 3. The core mental model

Physiology is a control system. The controlled variables are pressure, volume, oxygenation,
temperature, and pH. The controllers are neural, hormonal, and renal. Disease perturbs the
controlled variable; compensation pushes back; measurement observes the result at one point in
time.

```text
                        OXYGEN DELIVERY CHAIN

   Lungs ---- ventilation ----> PaO2 ----> SaO2
     |                                        |
     |                                        v
   Heart ---- cardiac output = HR x SV ----> CaO2 = 1.34 x Hb x SaO2 + 0.003 x PaO2
     |                                        |
     |                                        v
     +-------------------------------> DO2 = CO x CaO2 x 10
                                              |
                                              v
                                     tissue uptake (VO2)
                                              |
                                              v
                                     extraction ratio = VO2 / DO2
                                              |
                        +---------------------+---------------------+
                        |                                           |
                   adequate                                    inadequate
                        |                                           |
                   aerobic metabolism                     anaerobic metabolism
                        |                                           |
                    lactate stable                           lactate rises
```

The chain matters because failure at any link produces the same final sign: inadequate tissue
oxygenation. A patient can fail at the lung (low SaO2), the hemoglobin (anemia), the pump (low
CO), or the distribution (sepsis). A model that uses only one link will misclassify failures at
the others.

### 3.1 The compensation loop

```text
   stimulus: loss of volume, pressure, or oxygen
                    |
                    v
   baroreceptors / chemoreceptors / renal sensors
                    |
      +-------------+-------------+
      |             |             |
      v             v             v
   sympathetic    RAAS          ADH
      |             |             |
   HR up         Na/water      water
   contractility retention    retention
   vasoconstriction
      |             |             |
      +-------------+-------------+
                    |
                    v
   MAP maintained, often at the cost of
   perfusion to skin, gut, and kidney
```

The clinical consequence is that **compensation buys time and hides severity**. The data
consequence is that the first variables to move are not always the ones recorded most densely.
Heart rate rises before blood pressure falls. Lactate rises before urine output falls. A model
that uses only the last variable in the chain will be late.

### 3.2 The measurement layer

Every vital sign passes through four layers before it reaches your table:

```text
   biology -> sensor -> charting -> interface -> warehouse
```

- **Biology**: the true physiological state, which is not directly observable.
- **Sensor**: cuff, arterial line, pulse oximeter, thermistor, impedance belt.
- **Charting**: manual entry, automatic feed, default value, or template.
- **Interface**: HL7 message, device integration, manual transcription.

Error can enter at any layer. The same biological state can produce different numbers depending
on the sensor and the charting path. That is why a data dictionary entry for "systolic blood
pressure" is incomplete without method and source.

### 3.3 Pressure, flow, and resistance

Three relationships organize most cardiovascular reasoning:

- **MAP ≈ DBP + (SBP - DBP) / 3.** The approximation assumes a typical waveform; it is less
  accurate at very high or very low heart rates.
- **MAP ≈ CO x SVR + CVP**, where SVR is systemic vascular resistance and CVP is central venous
  pressure. In practice CVP is small relative to the product, so MAP is approximately CO x SVR.
- **CO = HR x SV.** Stroke volume depends on preload, afterload, and contractility.

These equations explain why the same MAP can represent opposite physiological states. In
hemorrhage, CO falls and SVR rises, so MAP may be maintained. In sepsis, CO rises and SVR falls,
so MAP may be maintained or fall. The number is the same; the patient is not.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Heart rate (HR) | Ventricular contractions per minute | Equated with pulse rate in arrhythmia | beats/min |
| Stroke volume (SV) | Blood ejected per beat | Treated as constant | mL/beat |
| Cardiac output (CO) | Volume ejected per minute | Confused with cardiac index | L/min |
| Cardiac index (CI) | CO divided by body surface area | Reported without BSA method | L/min/m² |
| Preload | Ventricular filling at end-diastole | Equated with CVP | mmHg or volume |
| Afterload | Resistance to ejection | Equated with blood pressure | mmHg or dyn·s/cm⁵ |
| Contractility | Intrinsic force of contraction | Inferred from ejection fraction alone | Not directly measured |
| Mean arterial pressure (MAP) | Time-weighted average arterial pressure | Computed as (SBP+DBP)/2 | mmHg |
| Systolic blood pressure (SBP) | Peak arterial pressure | Treated as perfusion pressure | mmHg |
| Diastolic blood pressure (DBP) | Minimum arterial pressure | Ignored in shock assessment | mmHg |
| Shock index (SI) | HR / SBP | Treated as a diagnosis | Dimensionless |
| Systemic vascular resistance (SVR) | Resistance to systemic flow | Confused with pulmonary resistance | dyn·s/cm⁵ |
| Central venous pressure (CVP) | Pressure in the vena cava or right atrium | Used alone to guide fluids | mmHg |
| Arterial oxygen content (CaO2) | Oxygen carried per deciliter of arterial blood | Confused with SaO2 | mL O2/dL |
| Oxygen delivery (DO2) | Oxygen delivered per minute | Confused with oxygen content | mL/min |
| Oxygen consumption (VO2) | Oxygen taken up by tissue per minute | Assumed constant in shock | mL/min |
| Extraction ratio | VO2 / DO2 | Ignored in favour of saturation | Proportion |
| Oxygen saturation (SaO2) | Fraction of hemoglobin bound to oxygen, arterial | Confused with SpO2 | % |
| Peripheral oxygen saturation (SpO2) | Oximeter estimate of SaO2 | Treated as identical to SaO2 | % |
| Partial pressure of oxygen (PaO2) | Dissolved oxygen tension, arterial | Confused with saturation | mmHg |
| Respiratory rate (RR) | Breaths per minute | Frequently defaulted | breaths/min |
| Tidal volume (Vt) | Volume per breath | Confused with minute ventilation | mL |
| Minute ventilation (VE) | Vt x RR | Ignored in respiratory failure | L/min |
| Dead space | Ventilated but not perfused lung | Confused with shunt | mL or fraction |
| Shunt | Perfused but not ventilated lung | Confused with dead space | Fraction |
| Temperature | Core or site-specific thermal state | Pooled across sites | °C |
| Febrile | Temperature above the normal range for the site | Defined differently across studies | °C |
| Hypothermia | Core temperature below 35 °C | Missed with oral measurement | °C |
| Tachycardia | HR above the age-specific normal range | Defined as >100 in all ages | beats/min |
| Bradycardia | HR below the age-specific normal range | Defined as <60 in all ages | beats/min |
| Hypotension | Blood pressure below the age-specific threshold | Equated with shock | mmHg |
| Shock | Inadequate tissue perfusion | Equated with low blood pressure | Clinical syndrome |
| Compensated shock | Shock with maintained blood pressure | Missed because BP is normal | Clinical state |
| Orthostatic hypotension | BP fall on standing | Measured inconsistently | mmHg |
| Pulse pressure | SBP - DBP | Ignored as a perfusion marker | mmHg |
| Capillary refill time | Time for colour to return after blanching | Poorly standardized | seconds |
| Urine output | Volume of urine per unit time and weight | Charted in aggregates | mL/kg/h |
| Body surface area (BSA) | Estimated skin surface area | Method often unstated | m² |
| Body mass index (BMI) | Weight divided by height squared | Used as a sole risk measure | kg/m² |
| Frailty | Multisystem vulnerability to stressors | Equated with age | Index score |

Three definitions need expansion.

**Shock is not hypotension.** Shock is a state of inadequate oxygen delivery relative to demand.
Blood pressure is a late and imperfect indicator. A patient with a normal blood pressure and a
lactate of 5 mmol/L may be in compensated shock. A patient with a low blood pressure and a
normal lactate may have chronic low pressure and adequate perfusion. The distinction matters
because a model trained on hypotension will detect shock late.

**SpO2 is not SaO2.** Pulse oximetry estimates arterial saturation from the ratio of light
absorption at two wavelengths. It is affected by motion, poor perfusion, venous pulsation, nail
polish, and skin pigmentation. The last is not a small effect: studies have documented clinically
important overestimation of SaO2 in patients with darker skin at low saturation ranges.

**Normal is a range, not a threshold.** The reference range for a vital sign is typically the
central 95% of a healthy population, which means 5% of healthy people fall outside it. A value
outside the range is a signal to investigate, not a diagnosis.

## 5. The mechanics, step by step

This section walks through measurement and interpretation. Steps 1 to 4 establish general
principles; steps 5 to 11 cover each vital sign; steps 12 to 14 cover derived indices; steps 15
to 17 cover compensation and shock; steps 18 to 20 cover data representation.

### General principles

**Step 1. Identify the population before the range.**

- Input: patient age, sex, pregnancy status, chronic conditions.
- Operation: select the appropriate reference population.
- Output: a range with a named population and source.
- Reasoning: adult, pediatric, neonatal, and pregnant ranges differ materially. A single adult
  range applied to a child will flag normal children as abnormal and miss abnormal ones.
- Branch point: if age is missing, do not default to adult. Treat the range as unknown and flag
  the record.
- Analytic artifact: `age_years`, `age_group`, `pregnant_flag`, `reference_set_id`.

**Step 2. Identify the measurement method.**

- Input: device, site, position, and observer.
- Operation: record the method alongside the value.
- Output: a method code.
- Reasoning: an arterial line pressure and a cuff pressure are not interchangeable. A manual
  respiratory count and an impedance-derived rate are not interchangeable. A temporal artery
  temperature and a rectal temperature are not interchangeable.
- Branch point: if method is absent, downgrade confidence and record the assumption.
- Analytic artifact: `method`, `device_id`, `site`, `position`.

**Step 3. Identify the time and context.**

- Input: timestamp, activity state, recent treatment.
- Operation: record whether the patient was resting, ambulatory, asleep, agitated, or receiving
  vasoactive medication.
- Output: a context flag.
- Reasoning: a heart rate of 110 after climbing stairs is normal; at rest it is not. A blood
  pressure after dialysis is not comparable to one before. A temperature after antipyretic is
  not the peak.
- Branch point: context is almost always missing. State the assumption in the feature
  dictionary.
- Analytic artifact: `activity_state`, `position`, `preceding_event`.

**Step 4. Check the value against the measurement's own limits.**

- Input: value and unit.
- Operation: apply a hard physiological range and a device range.
- Output: accept, flag, or reject.
- Reasoning: some values are impossible (temperature 45 °C in a living patient, HR 400 in an
  adult at rest, SpO2 100% with PaO2 30 mmHg). Others are possible but implausible (SBP 300
  mmHg). Hard limits catch unit errors and device artifacts; soft limits trigger review.
- Branch point: reject versus correct. Prefer flagging to deleting, because deletion destroys
  the evidence of a data quality problem.
- Analytic artifact: `plausibility_flag`, `plausibility_rule_id`.

### Heart rate

**Step 5. Measure and interpret heart rate.**

- Input: ECG, plethysmograph, arterial waveform, or manual pulse.
- Operation: count beats over a fixed interval; average over 10 seconds if irregular.
- Output: beats per minute.
- Reasoning: in atrial fibrillation, beat-to-beat intervals vary, so a short count is unstable.
  A monitor may average or may display the instantaneous rate. Exercise, pain, fever, anxiety,
  anemia, hypovolemia, and drugs all raise HR. Beta-blockers and calcium-channel blockers lower
  it and can mask compensatory tachycardia.
- Normal adult resting: 60-100 beats/min. Athletes may be 40-60 beats/min at rest.
- Pediatric ranges (awake): neonate 100-205, infant 100-180, toddler 98-140, preschool 80-120,
  school age 75-118, adolescent 60-100 beats/min. These are approximate PALS reference ranges;
  sources differ slightly at the youngest ages.
- Branch point: if the patient is on a rate-controlling drug, a normal HR may represent
  inadequate compensation rather than stability.
- Analytic artifact: `heart_rate`, `rhythm`, `source`.

### Blood pressure

**Step 6. Measure and interpret blood pressure.**

- Input: cuff, arterial line, or patient report.
- Operation: measure with correct cuff size, arm supported at heart level, after five minutes
  of rest; average at least two readings.
- Output: systolic and diastolic pressure in mmHg.
- Reasoning: cuff size is the most common source of error. An undersized cuff overestimates; an
  oversized cuff underestimates. Arm position below heart level raises the reading. White-coat
  effect raises office readings; masked hypertension lowers them. Arterial lines measure a
  different quantity and are more accurate at extremes.
- Adult categories (2025 AHA/ACC guideline, unchanged from 2017): normal <120 and <80; elevated
  120-129 and <80; stage 1 hypertension 130-139 or 80-89; stage 2 ≥140 or ≥90 mmHg. If systolic
  and diastolic fall in different categories, assign the higher. Classification uses the average
  of at least two readings on at least two occasions.
- Treatment goal in the 2025 guideline: <130/80 mmHg for most adults, with individualization for
  frailty, limited life expectancy, and pregnancy.
- Pregnancy: hypertension is defined as ≥140/90 mmHg; severe range is ≥160/110 mmHg. Normal
  blood pressure is lower in the first and second trimesters than in non-pregnant adults.
- Children: thresholds are age-, sex-, and height-specific percentile-based; do not apply adult
  cutoffs.
- Branch point: a single reading is not a diagnosis. The 2025 guideline makes out-of-office
  confirmation a Class 1 recommendation and does not recommend cuffless devices for diagnosis.
- Analytic artifact: `sbp`, `dbp`, `cuff_size`, `arm`, `position`, `reading_count`.

### Respiratory rate

**Step 7. Measure and interpret respiratory rate.**

- Input: manual count, impedance pneumography, capnography, or ventilator log.
- Operation: count for 30 or 60 seconds; record the method.
- Output: breaths per minute.
- Reasoning: respiratory rate is the most predictive routine vital sign for deterioration and
  the most poorly recorded. It is often entered as a default (16 or 20), copied from a previous
  entry, or derived from a monitor that misdetects motion. In a busy ward, it may be estimated
  rather than counted.
- Normal adult resting: 12-20 breaths/min.
- Pediatric: neonate 30-53, infant 30-53, toddler 22-37, preschool 20-28, school age 18-25,
  adolescent 12-20 breaths/min (approximate PALS ranges).
- Branch point: a respiratory rate of exactly 16 or 20 repeated across many encounters is a
  default-value signature. Treat it as missing, not as data.
- Analytic artifact: `respiratory_rate`, `method`, `default_flag`.

### Temperature

**Step 8. Measure and interpret temperature.**

- Input: oral, rectal, tympanic, temporal, axillary, or core probe.
- Operation: measure at a consistent site; record the site.
- Output: degrees Celsius.
- Reasoning: sites differ. Rectal is close to core; oral is about 0.3-0.5 °C lower; axillary is
  lower and less reliable; tympanic depends on technique; temporal artery is affected by
  ambient conditions and perfusion. Antipyretics change the value without changing the disease.
- Normal adult oral: approximately 36.5-37.5 °C. Rectal is approximately 0.3-0.5 °C higher.
  Hypothermia is core temperature below 35 °C.
- Pediatric: normal range is similar but thermoregulation is less stable in neonates; axillary
  measurement is common and less accurate.
- Pregnancy: baseline temperature is slightly higher, and the definition of fever does not
  change.
- Branch point: a temperature recorded as 98.6 in a metric field is a unit error, not a normal
  value. Check the unit before the value.
- Analytic artifact: `temperature`, `unit`, `site`, `method`.

### Oxygen saturation

**Step 9. Measure and interpret oxygen saturation.**

- Input: pulse oximeter, arterial blood gas (ABG), or co-oximetry.
- Operation: record SpO2 with the sensor site and oxygen delivery.
- Output: percent saturation.
- Reasoning: pulse oximetry is a ratio of light absorption and depends on a stable pulsatile
  signal. Motion, low perfusion, venous congestion, nail polish, and ambient light cause error.
  Skin pigmentation causes systematic overestimation at low saturation, documented in clinical
  studies and addressed in FDA safety communications. SpO2 cannot distinguish normal hemoglobin
  from carboxyhemoglobin or methemoglobin.
- Normal adult at sea level on room air: 95-100%. Values below 90% are generally considered
  hypoxemia and require assessment.
- Pregnancy: normal saturation is unchanged, but oxygen consumption and minute ventilation are
  increased, so desaturation occurs faster with apnea.
- Branch point: if the patient is on supplemental oxygen, record the flow rate or fraction of
  inspired oxygen (FiO2). SpO2 without oxygen context is incomplete.
- Analytic artifact: `spo2`, `o2_flow_lpm`, `fio2`, `sensor_site`.

### Other bedside variables

**Step 10. Measure consciousness and mental status.**

- Input: observation, Glasgow Coma Scale (GCS), or a delirium screen.
- Operation: record the scale, the component scores, and the time.
- Output: score plus narrative.
- Reasoning: GCS is often recorded as a total, which loses information. A patient with eyes 1,
  verbal 1, motor 6 and a patient with eyes 4, verbal 2, motor 2 can share a total of 8 but have
  different prognoses. Sedation and intubation confound the score.
- Analytic artifact: `gcs_total`, `gcs_components`, `sedated_flag`, `intubated_flag`.

**Step 11. Measure urine output and fluid balance.**

- Input: catheter, voiding record, dialysis record.
- Operation: sum over a defined interval; divide by weight and hours.
- Output: mL/kg/h.
- Reasoning: urine output is a downstream marker of renal perfusion and is affected by
  diuretics, contrast, chronic kidney disease, and obstruction. Oliguria is defined as less than
  0.5 mL/kg/h in adults; children generally require 1-2 mL/kg/h, and infants 2-3 mL/kg/h.
- Branch point: charted output may be aggregated over a shift, which hides the timing of the
  change.
- Analytic artifact: `urine_ml`, `interval_hours`, `weight_kg`, `urine_rate`.

### Derived indices

**Step 12. Compute mean arterial pressure.**

- Input: SBP and DBP.
- Operation: MAP = DBP + (SBP - DBP) / 3.
- Output: mmHg.
- Reasoning: MAP approximates the time-weighted average pressure that perfuses organs. The
  formula assumes a normal waveform and heart rate; it overestimates at high heart rates and
  underestimates with certain waveforms. A MAP below 65 mmHg is a common resuscitation target in
  sepsis, though the 2026 Surviving Sepsis Campaign guideline retains 65 mmHg as a strong
  recommendation and suggests 60-65 mmHg for patients aged 65 and older.
- Branch point: if an arterial line is available, use the measured mean rather than the formula.
- Analytic artifact: `map_calculated`, `map_source`.

**Step 13. Compute shock index and related ratios.**

- Input: HR and SBP.
- Operation: SI = HR / SBP.
- Output: dimensionless.
- Reasoning: a normal SI is approximately 0.5-0.7 in adults. A value above 0.9 suggests
  compensated shock and is associated with worse outcomes in trauma and hemorrhage. The index is
  simple, available at triage, and less sensitive to age-related baseline blood pressure than
  SBP alone. It is not validated as a standalone diagnostic.
- Branch point: beta-blockade lowers HR and therefore lowers SI, potentially masking shock.
- Analytic artifact: `shock_index`.

**Step 14. Compute oxygen delivery and consumption.**

- Input: CO, hemoglobin (Hb), SaO2, PaO2, and VO2 if available.
- Operation:
  - CaO2 (mL O2/dL) = 1.34 x Hb (g/dL) x SaO2 + 0.003 x PaO2 (mmHg).
  - DO2 (mL/min) = CO (L/min) x CaO2 (mL/dL) x 10.
  - DO2 index = DO2 / BSA.
  - Extraction ratio = VO2 / DO2.
- Output: mL/min, mL/min/m², and a proportion.
- Reasoning: the 1.34 coefficient is the oxygen-binding capacity of hemoglobin in mL O2 per gram.
  The 0.003 term is dissolved oxygen and is small at normal PaO2. DO2 index in a healthy resting
  adult is approximately 500-600 mL/min/m². VO2 index is approximately 110-160 mL/min/m².
  Extraction ratio at rest is approximately 25-30%; in shock it rises as tissues extract more
  from each unit of delivered oxygen.
- Branch point: if VO2 is not measured, assume a value and label it as an assumption. Do not
  present a derived extraction ratio as a measurement.
- Analytic artifact: `cao2`, `do2`, `do2_index`, `vo2_index`, `extraction_ratio`.

### Compensation and shock

**Step 15. Classify the shock state.**

- Input: vitals, lactate, mentation, urine output, capillary refill, and context.
- Operation: assign a mechanism — hypovolemic, cardiogenic, distributive, obstructive, or mixed.
- Output: a working classification with supporting evidence.
- Reasoning: the classification determines treatment and determines which variables matter.
  Hypovolemic shock: low preload, low CO, high SVR. Cardiogenic: high preload, low CO, high SVR.
  Distributive: low SVR, often high CO early. Obstructive: mechanical impediment to filling or
  ejection.
- Branch point: mixed states are common. Sepsis with cardiomyopathy is distributive and
  cardiogenic. Do not force a single label.
- Analytic artifact: `shock_type` if documented; otherwise not derivable.

**Step 16. Identify compensated shock.**

- Input: trend of HR, SBP, SI, lactate, urine output, and mentation.
- Operation: look for the pattern of compensation before hypotension appears.
- Output: a risk flag.
- Reasoning: the body maintains MAP by raising HR and SVR. A rising HR with a stable SBP is an
  early signal. A narrowing pulse pressure is another. Lactate rises when extraction is
  insufficient. A normal blood pressure is not reassurance.
- Branch point: young, fit patients compensate longer; older patients and those on
  beta-blockers may decompensate without tachycardia.
- Analytic artifact: `compensation_flag` derived from trend features, with the rule stated.

**Step 17. Reassess after intervention.**

- Input: repeat measurements after fluids, vasopressors, or transfusion.
- Operation: compare before and after with the same method and site.
- Output: a delta with a time stamp.
- Reasoning: a single post-intervention value is uninterpretable without the pre-intervention
  value. A rise in blood pressure after fluids may reflect volume or may reflect the natural
  course. A fall in lactate is reassuring but lagged.
- Branch point: if the method changed between measurements, the delta is confounded.
- Analytic artifact: `delta_value`, `delta_hours`, `method_changed_flag`.

### Data representation

**Step 18. Store value, unit, method, and time separately.**

- Input: source data.
- Operation: normalize units but preserve the original.
- Output: a tidy row per observation.
- Reasoning: unit conversion errors are common and silent. Preserve `value_original` and
  `unit_original` alongside the normalized value so the conversion can be audited.
- Analytic artifact: a long-format observations table with one row per measurement.

**Step 19. Model the missingness explicitly.**

- Input: the observations table.
- Operation: add indicators for measured, defaulted, imputed, and not assessed.
- Output: a missingness flag per feature.
- Reasoning: missingness is informative. A respiratory rate that is missing because the patient
  was on a monitor is different from one missing because the ward was busy.
- Analytic artifact: `is_measured`, `is_default`, `is_imputed`, `missing_reason`.

**Step 20. Validate the distribution against physiology.**

- Input: the feature distribution.
- Operation: plot by age, sex, setting, and site; compare to published ranges.
- Output: a distribution report.
- Reasoning: a systolic blood pressure distribution with a spike at 120 mmHg suggests default
  entry. A temperature distribution with a spike at 37.0 °C suggests templating. A respiratory
  rate distribution with a spike at 16 suggests the same.
- Analytic artifact: a data quality report with the spike locations and counts.

## 6. Worked example

A 72-year-old woman presents to the emergency department with two days of black stools and one
episode of vomiting blood. She takes aspirin 81 mg daily and apixaban 5 mg twice daily for atrial
fibrillation. She is alert but anxious. Her skin is cool and clammy. A nasogastric tube returns
coffee-ground material.

### 6.1 The measurements

| Item | Value | Unit | Method or context |
|---|---:|---|---|
| Heart rate | 112 | beats/min | Monitor, sinus tachycardia |
| Blood pressure | 96/64 | mmHg | Cuff, right arm, semi-recumbent |
| Respiratory rate | 24 | breaths/min | Manual count |
| Oxygen saturation | 96 | % | Pulse oximetry, room air |
| Temperature | 36.4 | °C | Oral |
| Weight | 68 | kg | Historical, from chart |
| Height | 160 | cm | Historical, from chart |
| Hemoglobin | 7.8 | g/dL | Venous blood, complete blood count |
| Baseline hemoglobin | 12.4 | g/dL | Two months earlier |
| Lactate | 4.2 | mmol/L | Venous blood gas |
| Urine output | 15 | mL in 1 hour | Foley catheter |

### 6.2 Step 1: the derived hemodynamics

Mean arterial pressure:

```text
MAP = DBP + (SBP - DBP) / 3
    = 64 + (96 - 64) / 3
    = 64 + 10.67
    = 74.7 mmHg
```

A MAP of 74.7 mmHg is above the common resuscitation target of 65 mmHg. It is not reassuring on
its own: her baseline may have been 95 mmHg, and her compensation is what is holding the mean
up.

Shock index:

```text
SI = HR / SBP = 112 / 96 = 1.17
```

A normal adult shock index is approximately 0.5-0.7. A value of 1.17 is substantially elevated
and is consistent with compensated hypovolemic shock.

Body surface area (Mosteller):

```text
BSA = sqrt((height_cm x weight_kg) / 3600)
    = sqrt((160 x 68) / 3600)
    = sqrt(10880 / 3600)
    = sqrt(3.0222)
    = 1.74 m²
```

### 6.3 Step 2: cardiac output

Stroke volume is not measured in this case. For the calculation, assume 45 mL/beat, which is
reduced from a typical adult value of 60-100 mL/beat because of low preload. Label this as an
assumption, not a measurement.

```text
CO = HR x SV = 112 x 45 = 5,040 mL/min = 5.04 L/min
CI = CO / BSA = 5.04 / 1.74 = 2.90 L/min/m²
```

A cardiac index of 2.90 L/min/m² is within the normal range of approximately 2.5-4.0 L/min/m².
This is the trap: a normal cardiac index does not exclude shock, because the index is normalized
to body size and because a falling stroke volume can be offset by a rising heart rate.

### 6.4 Step 3: oxygen content and delivery

Assume a PaO2 of 90 mmHg, consistent with an SpO2 of 96% on room air at sea level. The PaO2 is
not measured in this case; it is an assumption used to estimate the dissolved oxygen term.

```text
CaO2 = 1.34 x Hb x SaO2 + 0.003 x PaO2
     = 1.34 x 7.8 x 0.96 + 0.003 x 90
     = 10.03 + 0.27
     = 10.30 mL O2/dL
```

```text
DO2 = CO x CaO2 x 10
    = 5.04 x 10.30 x 10
    = 519 mL O2/min
```

```text
DO2 index = 519 / 1.74 = 298 mL O2/min/m²
```

A normal DO2 index at rest is approximately 500-600 mL/min/m². Her value is roughly half of
normal. The two contributors are anemia (Hb 7.8 g/dL) and a reduced stroke volume; the
tachycardia partly compensates for the latter.

Assume a resting oxygen consumption of 250 mL/min, or 144 mL/min/m². This is an assumption; VO2
is not measured.

```text
Extraction ratio = VO2 / DO2 = 250 / 519 = 0.48
```

The normal extraction ratio is approximately 0.25-0.30. An extraction ratio of 0.48 means the
tissues are removing nearly half of the delivered oxygen, which is a compensatory response and a
sign that delivery is marginal. If extraction were maximal and still inadequate, lactate would
rise further.

### 6.5 Step 4: the acid-base and lactate picture

A lactate of 4.2 mmol/L is above the normal range of approximately 0.5-2.0 mmol/L. Lactate
rises when anaerobic metabolism contributes to energy production, but also under aerobic
conditions with accelerated glycolysis (for example, beta-adrenergic stimulation). It is a
marker of illness severity, not a direct measure of oxygen debt. In this case, the combination of
hypovolemia, anemia, tachycardia, and oliguria makes hypoperfusion the leading explanation.

Urine output:

```text
15 mL / 1 h / 68 kg = 0.22 mL/kg/h
```

Oliguria in adults is conventionally defined as less than 0.5 mL/kg/h. Her output is less than
half of that threshold.

### 6.6 Step 5: estimating blood loss

Estimated blood volume in an adult woman is approximately 65 mL/kg:

```text
Blood volume = 65 x 68 = 4,420 mL
```

The hemoglobin fell from 12.4 to 7.8 g/dL, a difference of 4.6 g/dL. In a stabilized adult, a
fall of 1 g/dL roughly corresponds to one unit of packed red blood cells, or about 450 mL of
whole blood. That gives an approximate loss of 4.6 units, or about 2,070 mL, or 47% of her
estimated blood volume.

> **Note:** This estimate is only valid after hemodilution has occurred. In acute hemorrhage,
> hemoglobin may remain normal for hours while volume is lost. A low hemoglobin at presentation
> indicates either substantial loss, pre-existing anemia, or dilution from fluids. The estimate
> is a teaching approximation, not a measurement.

A 47% loss would place her in class IV hemorrhage by the Advanced Trauma Life Support (ATLS)
classification, which is inconsistent with her alert mental status and a systolic pressure of 96
mmHg. The discrepancy is the point: the hemoglobin-derived estimate and the clinical
classification disagree, and the clinician must reconcile them. Possible explanations include
pre-existing iron deficiency, a longer chronic bleed, or hemodilution from oral intake. The ATLS
classes are a trauma framework and do not map cleanly onto a slow gastrointestinal bleed.

### 6.7 Step 6: the physiology of her compensation

Her blood pressure is maintained by three mechanisms:

1. **Sympathetic activation.** Heart rate rose to 112 beats/min, and systemic vascular resistance
   rose, maintaining the mean pressure despite a falling stroke volume.
2. **Renin-angiotensin-aldosterone system (RAAS).** Sodium and water retention expand plasma
   volume, but this takes hours to days and is not an acute rescue.
3. **Antidiuretic hormone (ADH).** Water retention and vasoconstriction contribute, again with a
   slower time course.

The cost of compensation is perfusion to the skin, gut, and kidney. Her cool skin, oliguria, and
rising lactate are the evidence. A model that uses only blood pressure will see a patient who is
"almost normal." A model that includes shock index, lactate, and urine output will see the
compensation.

### 6.8 Step 7: the post-resuscitation state

After two units of packed red blood cells and 1 L of crystalloid over four hours, repeat
measurements are:

| Item | Before | After 4 h | Unit |
|---|---:|---:|---|
| Heart rate | 112 | 98 | beats/min |
| Blood pressure | 96/64 | 108/68 | mmHg |
| Hemoglobin | 7.8 | 9.1 | g/dL |
| Lactate | 4.2 | 2.8 | mmol/L |
| Urine output | 15 | 35 | mL/h |

Recompute MAP and shock index:

```text
MAP after = 68 + (108 - 68) / 3 = 68 + 13.3 = 81.3 mmHg
SI after  = 98 / 108 = 0.91
```

Recompute oxygen delivery, assuming stroke volume rose to 55 mL/beat and SaO2 is 0.97:

```text
CO after   = 98 x 55 = 5,390 mL/min = 5.39 L/min
CaO2 after = 1.34 x 9.1 x 0.97 + 0.003 x 90
           = 11.83 + 0.27
           = 12.10 mL O2/dL
DO2 after  = 5.39 x 12.10 x 10 = 652 mL O2/min
DO2 index  = 652 / 1.74 = 375 mL O2/min/m²
```

Oxygen delivery rose by 26% (from 519 to 652 mL/min). Two factors contributed: a higher
hemoglobin and a higher cardiac output. If only the hemoglobin had risen and the heart rate had
stayed at 112 with the same stroke volume, the increase would have been smaller. This is why
transfusion alone is not a treatment for shock: it treats the oxygen-carrying capacity, not the
volume deficit or the pump.

The lactate is falling, which is reassuring, but it lags. A falling lactate at four hours does
not prove that perfusion is adequate; it indicates that the trend is in the right direction.

### 6.9 Step 8: the data-quality review

A reviewer examining this record would check:

| Check | Rule | Finding | Action |
|---|---|---|---|
| Unit plausibility | Temperature between 30 and 43 °C | 36.4 | Pass |
| Unit plausibility | Hemoglobin between 3 and 20 g/dL | 7.8 | Pass |
| Cross-variable | SpO2 96% with PaO2 90 mmHg | Consistent | Pass |
| Cross-variable | Lactate 4.2 with pH and bicarbonate | Not available | Request ABG |
| Temporal | Hemoglobin fall over 2 months | 4.6 g/dL | Flag for review |
| Default value | RR exactly 16 or 20 | 24 | Pass |
| Method | BP cuff size recorded | Not recorded | Flag |
| Derived | MAP formula versus arterial line | No line | Accept formula |

The missing cuff size is a real limitation. An undersized cuff can raise systolic pressure by
5-10 mmHg, which would make her true pressure lower than recorded. In a shock assessment, that
matters.

## 7. Code, query, or calculation

The following block reproduces sections 6.2 to 6.8 and adds a blood pressure classifier using
the 2025 AHA/ACC categories.

```python
"""Hemodynamic and oxygen-delivery calculations for a GI bleed case.

All inputs are explicit. Assumptions are marked in comments and printed with the result.
"""

from math import sqrt


def map_from_bp(sbp: float, dbp: float) -> float:
    """Mean arterial pressure approximation. Valid for a typical waveform."""
    return dbp + (sbp - dbp) / 3.0


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    return sqrt((height_cm * weight_kg) / 3600.0)


def oxygen_delivery(co_lpm: float, hb_gdl: float, sao2: float,
                    pao2_mmhg: float, bsa_m2: float) -> dict:
    """CaO2 in mL/dL, DO2 in mL/min, DO2 index in mL/min/m2."""
    cao2 = 1.34 * hb_gdl * sao2 + 0.003 * pao2_mmhg
    do2 = co_lpm * cao2 * 10.0
    return {"cao2": cao2, "do2": do2, "do2_index": do2 / bsa_m2}


def bp_category(sbp: float, dbp: float) -> str:
    """2025 AHA/ACC adult category. Not for children or pregnancy."""
    if sbp >= 140 or dbp >= 90:
        return "stage 2 hypertension"
    if sbp >= 130 or dbp >= 80:
        return "stage 1 hypertension"
    if 120 <= sbp <= 129 and dbp < 80:
        return "elevated"
    if sbp < 120 and dbp < 80:
        return "normal"
    return "unclassified"


height_cm, weight_kg = 160.0, 68.0
bsa = bsa_mosteller(height_cm, weight_kg)
print(f"BSA: {bsa:.3f} m2")

# Before resuscitation
hr, sbp, dbp = 112.0, 96.0, 64.0
sv_ml = 45.0                       # assumed, not measured
hb, sao2, pao2 = 7.8, 0.96, 90.0   # PaO2 assumed
vo2_ml_min = 250.0                 # assumed

co = hr * sv_ml / 1000.0
map_before = map_from_bp(sbp, dbp)
si_before = hr / sbp
od = oxygen_delivery(co, hb, sao2, pao2, bsa)
extraction = vo2_ml_min / od["do2"]

print(f"MAP before: {map_before:.1f} mmHg")
print(f"Shock index before: {si_before:.3f}")
print(f"CO before: {co:.2f} L/min; CI: {co / bsa:.2f} L/min/m2")
print(f"CaO2 before: {od['cao2']:.2f} mL/dL")
print(f"DO2 before: {od['do2']:.1f} mL/min")
print(f"DO2 index before: {od['do2_index']:.1f} mL/min/m2")
print(f"Extraction ratio before (assumed VO2 250 mL/min): {extraction:.3f}")

# After resuscitation
hr2, sbp2, dbp2 = 98.0, 108.0, 68.0
sv2_ml = 55.0
hb2, sao2_2 = 9.1, 0.97
co2 = hr2 * sv2_ml / 1000.0
od2 = oxygen_delivery(co2, hb2, sao2_2, pao2, bsa)
print()
print(f"MAP after: {map_from_bp(sbp2, dbp2):.1f} mmHg")
print(f"Shock index after: {hr2 / sbp2:.3f}")
print(f"CO after: {co2:.2f} L/min")
print(f"DO2 after: {od2['do2']:.1f} mL/min")
print(f"DO2 index after: {od2['do2_index']:.1f} mL/min/m2")
print(f"DO2 change: {(od2['do2'] / od['do2'] - 1) * 100:.1f}%")
print()
print(f"BP category at presentation: {bp_category(sbp, dbp)}")
print(f"BP category after: {bp_category(sbp2, dbp2)}")

# Pediatric reference lookup, included to show that one range does not fit all.
PEDS_HR = {
    ("neonate", "awake"): (100, 205),
    ("infant", "awake"): (100, 180),
    ("toddler", "awake"): (98, 140),
    ("preschool", "awake"): (80, 120),
    ("school_age", "awake"): (75, 118),
    ("adolescent", "awake"): (60, 100),
}
for key in ("neonate", "school_age", "adolescent"):
    low, high = PEDS_HR[(key, "awake")]
    print(f"{key}: {low}-{high} beats/min (awake, approximate PALS range)")
```

Expected output:

```text
BSA: 1.738 m2
MAP before: 74.7 mmHg
Shock index before: 1.167
CO before: 5.04 L/min; CI: 2.90 L/min/m2
CaO2 before: 10.30 mL/dL
DO2 before: 519.3 mL/min
DO2 index before: 298.7 mL/min/m2
Extraction ratio before (assumed VO2 250 mL/min): 0.481

MAP after: 81.3 mmHg
Shock index after: 0.907
CO after: 5.39 L/min
DO2 after: 652.1 mL/min
DO2 index after: 375.1 mL/min/m2
DO2 change: 25.6%

BP category at presentation: normal
BP category after: normal
```

The classifier returns "normal" for 96/64 mmHg because the 2025 categories describe
hypertension, not hypotension. That is a defect in using a hypertension classifier for a shock
case, and it is deliberate in the example: a category system built for one decision should not be
reused for another. A shock workflow needs a hypotension threshold, and in an adult that is
commonly SBP below 90 mmHg or a MAP below 65 mmHg, with adjustment for the patient's baseline.

A SQL check for default respiratory rates:

```sql
-- Share of respiratory-rate observations at exactly 16 or 20 per site.
SELECT site_id,
       COUNT(*) AS n_obs,
       SUM(CASE WHEN respiratory_rate IN (16, 20) THEN 1 ELSE 0 END) AS n_default_like,
       ROUND(100.0 * SUM(CASE WHEN respiratory_rate IN (16, 20) THEN 1 ELSE 0 END)
             / COUNT(*), 2) AS pct_default_like
FROM vitals
WHERE respiratory_rate IS NOT NULL
GROUP BY site_id
ORDER BY pct_default_like DESC;
```

A site with more than about 40% of values at exactly 16 or 20 is likely defaulting the field.
The threshold is a heuristic, not a validated cut-off; the purpose is to trigger a workflow
review.

## 8. Data, standards, and regulatory context

**FHIR vital signs.** The FHIR R4 vital signs profile defines the core observations and their
required units. The commonly used LOINC codes are:

| Vital sign | LOINC | UCUM unit |
|---|---|---|
| Heart rate | 8867-4 | /min |
| Systolic blood pressure | 8480-6 | mm[Hg] |
| Diastolic blood pressure | 8462-4 | mm[Hg] |
| Respiratory rate | 9279-1 | /min |
| Body temperature | 8310-5 | Cel |
| Oxygen saturation | 2708-6 | % |
| Body weight | 29463-7 | kg |
| Body height | 8302-2 | cm |
| Body mass index | 39156-5 | kg/m2 |

The Unified Code for Units of Measure (UCUM) is the required unit syntax in FHIR. A pressure
stored as "mmHg" in a local table must be expressed as `mm[Hg]` in FHIR. This is a binding
format requirement for conformant resources; it is not a clinical recommendation.

**Early warning scores.** The National Early Warning Score 2 (NEWS2), published by the Royal
College of Physicians in 2017, combines respiratory rate, oxygen saturation, supplemental
oxygen, systolic blood pressure, pulse, consciousness, and temperature. It is a screening tool
for deterioration, not a diagnosis. The Quick Sequential Organ Failure Assessment (qSOFA),
introduced by the Sepsis-3 consensus in 2016, uses respiratory rate ≥22 breaths/min, systolic
blood pressure ≤100 mmHg, and altered mentation. Both scores are population-dependent and
threshold-dependent. A model that reproduces NEWS2 is not thereby validated; it inherits the
score's limitations.

**Hypertension guidance.** The 2025 AHA/ACC guideline retains the 2017 categories and sets a
treatment goal of below 130/80 mmHg for most adults. It makes out-of-office measurement
(ambulatory or home blood pressure monitoring) a Class 1 recommendation for confirming
diagnosis, and it does not recommend cuffless or smartwatch devices for diagnosis or management.
For industry, this constrains consumer-device blood pressure claims: a wrist-worn device that
has not been validated against a cuff protocol cannot be presented as diagnostic.

**Pregnancy.** Hypertensive disorders of pregnancy use different thresholds: ≥140/90 mmHg for
hypertension and ≥160/110 mmHg for severe range, per the American College of Obstetricians and
Gynecologists (ACOG). Normal blood pressure in pregnancy is lower than in non-pregnant adults,
particularly in the second trimester. Applying non-pregnant ranges to pregnant patients will
misclassify normal pregnancies as hypotensive and miss pregnancy-specific thresholds.

**Pediatrics.** Pediatric vital signs are age-dependent and, for blood pressure, percentile-based
by age, sex, and height. The PALS reference ranges used in this topic are approximate teaching
values; individual sources differ at the youngest ages. A pediatric early warning score must use
age-specific thresholds.

**Devices.** Pulse oximeters are regulated medical devices. In the United States, the FDA
requires premarket review for oximeters intended for clinical use, and the relevant performance
standard is ISO 80601-2-61. The FDA has issued safety communications about accuracy in patients
with darker skin pigmentation. A data pipeline that treats SpO2 as an unbiased measurement is
ignoring a regulated device limitation.

> **Note:** This section is educational. It is not medical advice, and it does not establish
> that a device, score, or product is safe or compliant for a particular use.

## 9. Industry practice and case evidence

**Pulse oximetry bias.** Sjoding and colleagues published a research letter in the *New England
Journal of Medicine* in 2020 comparing pulse oximetry with arterial co-oximetry in patients
whose race was recorded. They found that occult hypoxemia — an SpO2 in the normal range with a
low arterial saturation — occurred at substantially higher rates in patients recorded as Black
than in patients recorded as White. The study is a measurement study; it does not estimate the
clinical consequences in every setting. The documented fact is a device-level measurement bias.
The inference for data science is that any model using SpO2 inherits a race-correlated
measurement error, and subgroup performance should be reported at clinically relevant saturation
thresholds.

**Impossible values in public critical care data.** The MIMIC-IV documentation states that
implausible values may be present as an artifact of the archival process and that the database
is not cleaned by its custodians. Public issue reports have documented a height of 825,070
inches in the chart data, thousands of weights below 25 kg or above 300 kg, and timestamp offsets
of about one year in a small number of rows. These are not exotic failures; they are the normal
residue of clinical data entry. A pipeline that does not apply hard physiological limits will
carry them into a model.

**Default charting.** Health systems routinely audit vital-sign fields for default values. A
respiratory rate of exactly 16 or 20 repeated across a ward, a temperature of exactly 37.0 °C,
or a pain score of 0 recorded at every assessment are signatures of templated or default entry.
The published literature on documentation burden and copy-forward supports the general
mechanism; the specific prevalence at a given site must be measured locally.

**Wearables and cuffless blood pressure.** Consumer devices increasingly report blood pressure
without a cuff. The 2025 AHA/ACC guideline's position is that cuffless devices are not
recommended for diagnosis or management. For a data team, this means wearable blood pressure
values should be treated as a different measurement type, not pooled with cuff readings. Their
error structure, calibration, and population validity differ.

**Early warning implementation.** Hospitals that deploy NEWS2 or a similar score typically pair
it with a response protocol: a trigger threshold, a responsible clinician, and an escalation
path. The score alone does not change outcomes. Documented implementations vary in threshold and
response, which is why a published validation cannot be transferred without local evaluation.
This is inference from implementation science, not a finding of a single trial.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Adult ranges applied to children | Pediatric records flagged as bradycardic | Age-specific thresholds ignored | Alarm fatigue and missed deterioration | Age-partitioned reference sets |
| Pregnancy ranges ignored | Normal pregnant BP treated as hypotension | Plasma volume and vasodilation lower BP | Unnecessary intervention | Pregnancy flag and ACOG thresholds |
| Unit confusion | Temperature 98.6 in a Celsius field | Fahrenheit entered as Celsius | Missed fever or false fever | Unit column; hard range check |
| Default respiratory rate | Spike at exactly 16 or 20 | Templated or estimated entry | Model learns charting, not breathing | Default flag; treat as missing |
| Cuff size unrecorded | SBP systematically higher at one site | Undersized cuff overestimates | Misclassified hypertension | Record cuff size; training |
| Single reading used for diagnosis | BP category assigned from one value | Normal variability | Overdiagnosis | Average multiple readings; out-of-office confirmation |
| SpO2 treated as unbiased | Subgroup saturation gaps | Skin pigmentation, perfusion | Differential misclassification | Subgroup audit; ABG confirmation |
| Shock defined by hypotension | Normal BP with high lactate | Compensation maintains pressure | Late detection | Include SI, lactate, urine output |
| Cardiac index assumed normal | CI normal but SV low | HR compensates for SV | Missed cardiogenic component | Report CO and SV separately when available |
| Post-intervention value only | Delta uninterpretable | No baseline | Wrong conclusion about response | Store paired values |
| Weight estimated from chart | Dosing and BSA errors | Fluid shifts, amputation, pregnancy | Wrong dose or index | Measure weight; record method |
| Urine output aggregated by shift | Timing of oliguria lost | Charting convention | Delayed recognition | Store interval and timestamp |
| Arterial and cuff pressures pooled | Distribution has two modes | Different measurement methods | Confounded features | Method flag; separate analyses |
| Device change mid-stay | Morphology and values shift | Sensor replacement or calibration | Artificial trend | Device ID and calibration events |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The recorded vital sign represents the patient at that time | Time-series modelling depends on it | Back-entry, defaults, delayed charting | Compare chart time with device time on a sample |
| The measurement method is constant within a feature | Pooling requires comparability | Cuff versus line, oral versus rectal | Method distribution by site and year |
| The reference range matches the population | Flagging depends on it | Age, pregnancy, chronic disease, altitude | Stratify values by subgroup |
| Stroke volume can be assumed for derived calculations | SV is rarely measured | Cardiomyopathy, sepsis, valve disease | Compare with echocardiography where available |
| VO2 is approximately 250 mL/min at rest | Extraction ratio requires it | Fever, shivering, sepsis, sedation | Label as assumption; sensitivity analysis |
| MAP formula approximates true mean pressure | Arterial mean is not always available | Extreme heart rates, abnormal waveforms | Compare with arterial line means |
| Oxygen saturation is accurate across skin tones | SpO2 is used as a proxy for SaO2 | Device bias, poor perfusion, motion | Paired SpO2 and ABG by subgroup |
| Urine output reflects renal perfusion | Oliguria is used as a shock marker | Diuretics, CKD, obstruction | Review medication and history |
| The patient is at sea level | SpO2 thresholds assume it | Altitude, supplemental oxygen | Record altitude and FiO2 |
| The first value is representative | Summary features often use it | White-coat effect, pain, anxiety | Use averages and trends |

## 12. Exercises

**Exercise 1 — Range construction (application, 45 minutes).**
Build a reference table for heart rate, systolic pressure, respiratory rate, temperature, and
SpO2 for three populations: adults 18-64, adults 65 and older, and children 1-5 years. Cite the
source and year for each range. Deliverable: a table with population, range, source, and caveat.
Acceptance criteria: every range names its population; pediatric values are age-specific;
pregnancy is flagged as a separate population. Time: 45 minutes.

**Exercise 2 — Derived hemodynamics (calculation, 40 minutes).**
A 55-year-old man has HR 128, BP 84/52, Hb 9.2 g/dL, SpO2 94%, PaO2 70 mmHg, weight 80 kg,
height 175 cm. Assume SV 40 mL and VO2 260 mL/min. Compute MAP, shock index, BSA, CO, CI, CaO2,
DO2, DO2 index, and extraction ratio. Deliverable: a calculation sheet with units at every step.
Acceptance criteria: arithmetic correct; assumptions labelled; interpretation states whether
delivery is adequate. Time: 40 minutes.

**Exercise 3 — Compensation reasoning (critique, 50 minutes).**
Explain why a 24-year-old with a 25% blood loss may have a normal blood pressure while a
78-year-old on a beta-blocker with the same loss may be hypotensive. Deliverable: a 400-word
explanation with a variable-by-variable comparison. Acceptance criteria: at least three
mechanisms; the role of age and medication is explicit; no claim that one patient is "healthier."
Time: 50 minutes.

**Exercise 4 — Default-value audit (application, 60 minutes).**
Using a public vital-sign dataset, compute the proportion of respiratory-rate, temperature, and
pain values at common default values, stratified by care unit. Deliverable: a table and three
plots. Acceptance criteria: the audit distinguishes exact repeats from near repeats; the
interpretation does not claim intent. Time: 60 minutes.

**Exercise 5 — Unit-error hunt (application, 45 minutes).**
Plant five unit errors in a small vital-sign table (for example, a Fahrenheit temperature in a
Celsius column, a weight in pounds in a kilogram column). Write rules that detect each.
Deliverable: code and a detection report. Acceptance criteria: all five detected; no more than
one false positive; each rule has a physiological justification. Time: 45 minutes.

**Exercise 6 — Shock case review (critique, 60 minutes).**
Take the GI bleed case and argue for or against the statement "her blood pressure was normal, so
she was not in shock." Deliverable: a 500-word argument citing at least five variables.
Acceptance criteria: distinguishes hypotension from shock; uses the oxygen delivery chain;
acknowledges the assumptions in the calculation. Time: 60 minutes.

**Exercise 7 — Pediatric safety review (design, 75 minutes).**
Design a pediatric vital-sign alert for a hospital ward. Specify age bands, thresholds, alert
tiers, and escalation. Deliverable: a one-page specification plus a test plan. Acceptance
criteria: thresholds are age-specific; the alert has a stated false-positive tolerance; the
escalation path names a role. Time: 75 minutes.

**Exercise 8 — Measurement provenance (design, 60 minutes).**
Design a schema for vital signs that records value, unit, method, site, device, position, and
context. Show how it would represent a cuff reading, an arterial line reading, and a
patient-reported reading of the same quantity. Deliverable: a schema and three example rows.
Acceptance criteria: the three rows are distinguishable without free text; units are explicit;
the schema maps to FHIR Observation. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** BSA = sqrt((175 x 80) / 3600) = sqrt(3.8889) = 1.972 m². MAP = 52 +
(84 - 52)/3 = 62.7 mmHg. Shock index = 128/84 = 1.524. CO = 128 x 40 / 1000 = 5.12 L/min. CI =
5.12/1.972 = 2.60 L/min/m². CaO2 = 1.34 x 9.2 x 0.94 + 0.003 x 70 = 11.59 + 0.21 = 11.80 mL/dL.
DO2 = 5.12 x 11.80 x 10 = 604 mL/min. DO2 index = 604/1.972 = 306 mL/min/m². Extraction = 260/604
= 0.43. Delivery is below the normal index of 500-600 mL/min/m², and the extraction ratio is
well above the normal 0.25-0.30, so delivery is marginal despite a normal cardiac index.

**Answer to self-check on pulse pressure.** Pulse pressure is SBP minus DBP. A narrowing pulse
pressure in hemorrhage reflects reduced stroke volume and increased vasoconstriction. In the
case, 96 - 64 = 32 mmHg, which is narrow relative to the systolic pressure. It is an early
signal, not a diagnosis.

**Answer to self-check on MAP.** For 96/64, MAP = 64 + 32/3 = 74.7 mmHg. For 108/68, MAP = 68 +
40/3 = 81.3 mmHg. The approximation assumes a typical waveform; at very high heart rates the
diastolic period shortens and the formula becomes less accurate.

**Answer to self-check on normal ranges.** Adult resting heart rate 60-100 beats/min; adult
normal blood pressure below 120/80 mmHg by the 2025 AHA/ACC categories; adult respiratory rate
12-20 breaths/min; adult oral temperature approximately 36.5-37.5 °C; adult SpO2 95-100% at sea
level on room air. These do not apply to children or pregnant patients without modification.

**Answer to self-check on shock.** Shock is inadequate tissue perfusion relative to demand. It is
not defined by a blood pressure threshold. Compensated shock can present with a normal blood
pressure and an elevated lactate, shock index, or narrowed pulse pressure.

**Answer to self-check on oxygen delivery.** DO2 depends on cardiac output, hemoglobin, and
saturation. A patient with a normal cardiac output and severe anemia can have low DO2. A patient
with a normal hemoglobin and cardiogenic shock can also have low DO2. The chain, not any single
variable, determines delivery.

**Rubric for Exercise 7.** Full credit: age bands align with a named source; thresholds are
tiered; false-positive tolerance is stated as a number; escalation names a role and a time; the
test plan includes a retrospective silent-mode evaluation. Half credit: thresholds are present
but unsourced or not tiered. No credit: one set of thresholds for all ages.

**Rubric for Exercise 8.** Full credit: the schema distinguishes method, site, device, position,
and context; three example rows are unambiguous; the FHIR mapping is correct for at least the
core fields. Half credit: provenance is present but incomplete. No credit: free-text method.

## 14. Interview questions

**1. What is the difference between shock and hypotension?**
A strong answer defines shock as inadequate perfusion and notes that blood pressure can be
normal in compensated shock. A weak answer equates the two.

**2. Why is respiratory rate both important and unreliable?**
A strong answer explains its predictive value for deterioration and its frequent defaulting or
imputation. A weak answer says it is "often missing" without a mechanism.

**3. How would you compute oxygen delivery, and what does it tell you?**
A strong answer gives CO x CaO2 x 10, defines CaO2, and explains extraction ratio. A weak answer
confuses delivery with saturation.

**4. Why can a patient with a normal cardiac index still be in shock?**
A strong answer explains normalization by body size, compensation by heart rate, and the
difference between flow and perfusion. A weak answer says the index is "not sensitive" without
mechanism.

**5. What are the adult blood pressure categories in the 2025 guideline?**
A strong answer gives normal, elevated, stage 1, and stage 2 with thresholds and notes
out-of-office confirmation. A weak answer gives only 140/90.

**6. How do pediatric vital signs differ from adult values?**
A strong answer gives age-specific ranges and notes that blood pressure is percentile-based. A
weak answer says children have faster heart rates without numbers.

**7. Why is pulse oximetry a fairness issue?**
A strong answer explains device bias by skin pigmentation and differential misclassification. A
weak answer says "biased data" without a mechanism.

**8. What would make you distrust a blood pressure value?**
A strong answer lists cuff size, position, method, single reading, and white-coat effect. A weak
answer says "measurement error."

**9. How do you handle a vital sign that is missing 80% of the time?**
A strong answer distinguishes mechanisms, adds a missingness indicator, and considers whether the
feature is usable at all. A weak answer imputes the mean.

**10. When is a derived variable better than a raw one?**
A strong answer explains that derived variables encode physiology (shock index, MAP) and can be
more robust, but inherit assumptions. A weak answer says derived variables are always better.

**11. How would you validate a vital-sign feature across sites?**
A strong answer compares distributions, methods, and missingness by site and period, and tests
transportability. A weak answer reports overall AUC.

**12. What is the risk of using a hypertension classifier in a shock workflow?**
A strong answer notes that the categories describe high blood pressure and return "normal" for
hypotension, so the tool is silent exactly when it is needed. A weak answer says the classifier
is "not validated."

## 15. Key terms

- **Afterload** — resistance against which the ventricle ejects.
- **Cardiac index** — cardiac output per square meter of body surface area.
- **Cardiac output** — volume ejected by the heart per minute.
- **CaO2** — arterial oxygen content, mL O2 per dL of blood.
- **Compensated shock** — inadequate perfusion with maintained blood pressure.
- **CVP** — central venous pressure.
- **DO2** — oxygen delivery per minute.
- **Extraction ratio** — VO2 divided by DO2.
- **FiO2** — fraction of inspired oxygen.
- **GCS** — Glasgow Coma Scale.
- **Hemorrhagic shock** — shock from blood loss.
- **Hypotension** — blood pressure below the relevant threshold.
- **MAP** — mean arterial pressure.
- **NEWS2** — National Early Warning Score 2.
- **Oliguria** — urine output below the age- and weight-specific threshold.
- **Orthostatic hypotension** — blood pressure fall on standing.
- **PaO2** — arterial partial pressure of oxygen.
- **PALS** — Pediatric Advanced Life Support.
- **Preload** — ventricular filling at end-diastole.
- **Pulse pressure** — systolic minus diastolic pressure.
- **qSOFA** — Quick Sequential Organ Failure Assessment.
- **RAAS** — renin-angiotensin-aldosterone system.
- **SaO2** — arterial oxygen saturation.
- **Shock** — inadequate tissue perfusion.
- **Shock index** — heart rate divided by systolic blood pressure.
- **SpO2** — peripheral oxygen saturation by pulse oximetry.
- **Stroke volume** — volume ejected per beat.
- **SVR** — systemic vascular resistance.
- **Tachycardia** — heart rate above the age-specific range.
- **UCUM** — Unified Code for Units of Measure.
- **VO2** — oxygen consumption per minute.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Guidelines and standards**

- American Heart Association and American College of Cardiology. 2025
  AHA/ACC/AANP/AAPA/ABC/ACCP/ACPM/AGS/AMA/ASPC/NMA/PCNA/SGIM Guideline for the Prevention,
  Detection, Evaluation and Management of High Blood Pressure in Adults. *Circulation*, 2025.
  https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000001356
- American Heart Association. Part 4: Pediatric Basic and Advanced Life Support: 2020 American
  Heart Association Guidelines for Cardiopulmonary Resuscitation and Emergency Cardiovascular
  Care. *Circulation*, 2020. https://www.ahajournals.org/doi/10.1161/CIR.0000000000000901
- Royal College of Physicians. National Early Warning Score (NEWS) 2, 2017.
  https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/
- Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for
  Sepsis and Septic Shock (Sepsis-3). *JAMA*, 2016.
  https://doi.org/10.1001/jama.2016.0287
- Surviving Sepsis Campaign. International Guidelines for Management of Sepsis and Septic Shock
  2026. *Critical Care Medicine*, 2026.
  https://doi.org/10.1097/CCM.0000000000007089
- Health Level Seven International. FHIR R4 vital signs profile.
  https://hl7.org/fhir/R4/observation-vitalsigns.html
- Unified Code for Units of Measure. https://ucum.org/
- International Organization for Standardization. ISO 80601-2-61: Medical electrical equipment —
  Particular requirements for basic safety and essential performance of pulse oximeter equipment.

**Evidence**

- Sjoding MW, Dickson RP, Iwashyna TJ, Gay SE, Valley TS. Racial bias in pulse oximetry
  measurement. *New England Journal of Medicine*, 2020.
  https://doi.org/10.1056/NEJMc2029240
- US Food and Drug Administration. Pulse oximeter safety communications.
  https://www.fda.gov/medical-devices/safety-communications
- Johnson A, Bulgarelli L, Pollard T, et al. MIMIC-IV documentation, version 3.1, PhysioNet,
  2024. https://physionet.org/content/mimiciv/3.1/

**Background reading**

- West JB, Luks AM. *West's Respiratory Physiology*, 11th edition.
- Lilly LS, et al. *Pathophysiology of Heart Disease*, 7th edition.
- Bickley LS. *Bates' Guide to Physical Examination and History Taking*, 13th edition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_clinical-reasoning-and-workflow.md`](01_clinical-reasoning-and-workflow.md)
- Next topic: [`03_laboratory-medicine.md`](03_laboratory-medicine.md)
- Related topic: [`07_cardiology-and-ecg-basics.md`](07_cardiology-and-ecg-basics.md)
- Related topic: [`09_special-populations.md`](09_special-populations.md)
- Survey file: [`../../01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- ECG quality lab: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)

