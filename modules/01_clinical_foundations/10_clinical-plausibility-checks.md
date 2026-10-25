# Clinical Plausibility Checks

> Module 01 — Clinical Foundations for Data Scientists
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04, 05, 06, 07, 08, 09
> Version 2026.3

---

## 1. Why this topic exists

Every clinical dataset contains values that are impossible, inconsistent, or implausible. Some
are unit errors. Some are device artifacts. Some are documentation defaults. Some are genuine
physiology that looks impossible at first glance. The job of a plausibility check is not to make
the data clean; it is to make the data's defects visible, bounded, and documented.

The industry cost of skipping this step is predictable.

**Cost 1: a model learns an artifact.** A systolic blood pressure of 0 mmHg is not a patient in
extremis; it is a missing value encoded as zero. A model trained without a check will learn that
zero blood pressure predicts a particular outcome, and it will look accurate for the wrong
reason.

**Cost 2: a cohort is silently wrong.** An age of 0 in an adult ICU indicates a missing birth
date, not a neonate. A discharge date before the admission date is a data entry error, not a
negative length of stay. A cohort selected on these fields will include and exclude the wrong
patients.

**Cost 3: a result is not reproducible.** A cleaning rule applied once and not documented cannot
be reapplied. A colleague who reruns the pipeline gets a different cohort and cannot tell why.

**Cost 4: a safety signal is missed.** A potassium of 7.2 mmol/L in a non-hemolyzed sample is a
critical value. A pipeline that filters it as an outlier removes the very observation that
should trigger action.

This topic is the capstone of the module. It turns the clinical vocabulary of topics 1 through 9
into a checklist that can be run on any dataset.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** plausibility in terms of range, unit, temporal, cross-variable, cohort, duplicate,
   and drift checks.
2. **Write** a rule set that flags impossible values without deleting legitimate extremes.
3. **Apply** physiological consistency checks for acid-base, electrolytes, oxygenation, and renal
   function.
4. **Detect** impossible trajectories (discharge before admission, result before order, negative
   intervals).
5. **Audit** a cohort for implausible demographics, prevalence, and missingness by site.
6. **Quantify** the effect of cleaning decisions on cohort size and outcome estimates.
7. **Produce** a plausibility memo that a clinical reviewer can act on.

## 3. The core mental model

Plausibility is a layered set of constraints. Each layer catches a different class of defect, and
no single layer is sufficient.

```text
   LAYER 1  Single value        range, unit, type, missing code
   LAYER 2  Within record       temporal order, duplicate, derived consistency
   LAYER 3  Cross-variable      physiology (acid-base, electrolytes, oxygenation)
   LAYER 4  Across cohort       distributions, prevalence, site, era
   LAYER 5  Across time         drift, version change, migration
   LAYER 6  Against workflow    order-to-result, documentation, device
```

### 3.1 The impossibility hierarchy

```text
   Impossible in a living patient
        |
        +-- e.g., temperature 60 C, age 200 years, SpO2 100% with PaO2 20 mmHg
        |
   Possible but implausible
        |
        +-- e.g., SBP 320 mmHg, HR 260, weight 400 kg
        |
   Possible and unusual
        |
        +-- e.g., SBP 70 mmHg in a young athlete, HR 45 in a beta-blocked patient
        |
   Possible and normal
```

The hierarchy matters because the action differs. Impossible values should be rejected or
corrected. Implausible values should be flagged and reviewed. Unusual values should be kept and
explained. A rule that treats all three the same will either delete real patients or keep
artifacts.

### 3.2 The physiology consistency triangle

```text
              pH
             /  \
            /    \
        PaCO2 -- HCO3
```

The three acid-base variables are linked by the Henderson-Hasselbalch equation. If two are known,
the third is determined. A recorded triple that does not satisfy the equation is internally
inconsistent, which means at least one value is wrong, or the values were not drawn at the same
time.

```text
   pH = 6.1 + log10(HCO3 / (0.03 x PaCO2))
```

This is the single most useful cross-variable check in critical care data.

### 3.3 The cohort as a distribution

A cohort has a shape. Age, sex, length of stay, mortality, and disease prevalence have expected
ranges for a given setting. A cohort with a mean age of 12 in an adult ICU, a mortality of 60% in
a low-acuity clinic, or a sepsis prevalence of 40% in an outpatient dataset is implausible. The
check is not a proof of error; it is a prompt to investigate.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Plausibility | Whether a value is consistent with the patient and the measurement | Confused with normality | Judgment |
| Validity | Whether a value measures what it claims | Confused with plausibility | Assessment |
| Completeness | Proportion of expected values present | Confused with accuracy | Proportion |
| Accuracy | Agreement with the true value | Confused with precision | Error |
| Consistency | Agreement across related values | Confused with completeness | Rule-based |
| Timeliness | Whether data are available when needed | Confused with accuracy | Time |
| Range check | Whether a value falls within hard limits | Confused with a reference interval | Rule |
| Unit check | Whether the unit matches the expected unit | Ignored when units are implicit | Rule |
| Temporal check | Whether times are ordered and plausible | Confused with a range check | Rule |
| Cross-variable check | Whether related values are jointly consistent | Confused with a range check | Rule |
| Cohort check | Whether distributions match the setting | Confused with a model evaluation | Rule |
| Duplicate check | Whether records represent the same event | Confused with repeated measurement | Rule |
| Drift check | Whether distributions change over time | Confused with a real trend | Rule |
| Sentinel value | A code used to represent missing or special values | Treated as a real value | Convention |
| Missing code | A value such as -1, 999, or "N/A" | Parsed as a number | Convention |
| Physiologic impossibility | A value incompatible with life | Confused with an extreme value | Rule |
| Artifact | A value produced by the measurement process | Confused with biology | Rule |
| Default value | A value entered because the field required one | Confused with a measurement | Rule |
| Imputation | Filling a missing value | Confused with observation | Method |
| Delta check | Comparison with a prior value | Confused with a range check | Rule |
| Outlier | A value far from the rest | Confused with an error | Statistics |
| Winsorization | Capping extreme values | Confused with correction | Method |
| Truncation | Removing values beyond a limit | Confused with cleaning | Method |
| Henderson-Hasselbalch | Relationship between pH, PaCO2, and HCO3 | Ignored in acid-base data | Equation |
| Winter's formula | Expected PaCO2 in metabolic acidosis | Used outside metabolic acidosis | Equation |
| Anion gap | Na - (Cl + HCO3) | Computed without albumin correction | mmol/L |
| Delta-delta | Change in anion gap divided by change in bicarbonate | Misinterpreted without the direction | Ratio |
| Osmolal gap | Measured minus calculated osmolality | Computed with the wrong formula | mOsm/kg |
| Corrected sodium | Sodium adjusted for hyperglycemia | Applied outside the valid glucose range | mmol/L |
| Corrected calcium | Calcium adjusted for albumin | Applied when albumin is unreliable | mg/dL |
| A-a gradient | Alveolar-arterial oxygen difference | Computed without the right FiO2 | mmHg |
| P/F ratio | PaO2 divided by FiO2 | Confused with the A-a gradient | mmHg |
| Shock index | HR divided by SBP | Used as a diagnosis | Dimensionless |
| eGFR | Estimated glomerular filtration rate | Used for drug dosing | mL/min/1.73 m² |
| Cockcroft-Gault | Creatinine clearance equation | Confused with eGFR | mL/min |
| ALCOA+ | Data integrity principles | Treated as a software feature | Principles |
| Data integrity | Completeness, consistency, and accuracy of records | Confused with data security | Concept |
| Audit trail | Record of changes to data | Confused with a log file | Metadata |

## 5. The mechanics, step by step

### Layer 1: single values

**Step 1. Enforce type and parse rules.**

- Input: raw field.
- Operation: parse as numeric or categorical; reject unparseable strings; record parse failures.
- Output: typed values with a parse status.
- Reasoning: a numeric field containing "N/A" or "see note" will be silently coerced to missing
  or to a wrong number. The parse failure rate is itself a quality metric.
- Branch point: a high parse failure rate in one field suggests a unit or format change.
- Analytic artifact: `parse_status`, `parse_failure_count`.

**Step 2. Detect sentinel values.**

- Input: typed values.
- Operation: identify values used as missing codes (-1, -999, 999, 9999, 0 in some fields).
- Output: a sentinel flag.
- Reasoning: a systolic blood pressure of 0 is usually a missing value, not a measurement. A
  weight of 999 kg is a placeholder. A temperature of -1 °C is a missing code.
- Branch point: some zeros are real (for example, a pain score of 0). The field context matters.
- Analytic artifact: `sentinel_flag`, `sentinel_rule`.

**Step 3. Apply hard physiological limits.**

- Input: value and unit.
- Operation: compare with a range outside which survival is implausible.
- Output: an impossible flag.
- Reasoning: a temperature above 45 °C or below 25 °C in a living patient is not credible. A
  heart rate above 300 or below 10 in an adult is not credible. A systolic pressure above 300
  or below 30 is not credible without device support.
- Branch point: the limits differ by age and by device. A neonate can have a heart rate of 200;
  an adult cannot sustain it at rest.
- Analytic artifact: `impossible_flag`, `limit_source`.

**Step 4. Apply unit checks.**

- Input: value, unit, and the distribution of values.
- Operation: check whether the unit is the expected one; detect bimodal distributions consistent
  with mixed units.
- Output: a unit flag.
- Reasoning: a glucose distribution with modes at 5.5 and 100 is a mixture of mmol/L and mg/dL.
  A temperature distribution with modes at 37 and 98.6 is a mixture of Celsius and Fahrenheit.
  A weight distribution with modes at 70 and 154 is a mixture of kilograms and pounds.
- Branch point: a unit may be correct but unusual (for example, hemoglobin in g/L rather than
  g/dL).
- Analytic artifact: `unit_flag`, `mode_check`.

**Step 5. Check against age- and sex-specific ranges.**

- Input: value, age, sex, pregnancy status.
- Operation: compare with the appropriate reference interval.
- Output: an implausible flag, not an abnormal flag.
- Reasoning: an adult range applied to a child produces false impossibilities. A non-pregnant
  range applied to a pregnant patient produces false anemia.
- Branch point: reference intervals differ by assay and laboratory.
- Analytic artifact: `range_flag`, `reference_set_id`.

### Layer 2: within the record

**Step 6. Check temporal order.**

- Input: timestamps.
- Operation: verify admission before discharge, order before result, birth before death.
- Output: an order violation flag.
- Reasoning: a discharge before admission is impossible. A result before the order is usually a
  timestamp error, occasionally a back-entered result.
- Branch point: time zones and daylight saving can create apparent violations of one hour.
- Analytic artifact: `temporal_flag`, `interval_h`.

**Step 7. Check duplicate records.**

- Input: identifiers and event details.
- Operation: identify exact and near-duplicate rows.
- Output: a duplicate flag.
- Reasoning: duplicate laboratory rows inflate counts and can bias means. Duplicate encounters
  can double-count patients.
- Branch point: two measurements at the same time can be legitimate (for example, repeat
  potassium). The check should distinguish exact duplicates from repeated measurements.
- Analytic artifact: `duplicate_flag`, `duplicate_group_id`.

**Step 8. Check derived-value consistency.**

- Input: component values and derived values.
- Operation: recompute the derived value and compare.
- Output: a consistency flag.
- Reasoning: BMI, MAP, eGFR, anion gap, and LDL can be recomputed. A mismatch indicates a
  transcription error, a different formula, or a unit error.
- Branch point: the source may have used a different formula (for example, Bazett versus
  Fridericia for QTc).
- Analytic artifact: `derived_flag`, `recompute_difference`.

**Step 9. Check internal arithmetic.**

- Input: related values.
- Operation: verify sums, differences, and ratios.
- Output: a consistency flag.
- Reasoning: a hematocrit that is not roughly three times the hemoglobin suggests a unit or
  transcription error. A bicarbonate that does not fit the Henderson-Hasselbalch equation
  suggests a timing or measurement error.
- Branch point: the relationship is approximate, not exact; use a tolerance.
- Analytic artifact: `arithmetic_flag`.

### Layer 3: cross-variable physiology

**Step 10. Check acid-base consistency.**

- Input: pH, PaCO2, HCO3.
- Operation: compute the predicted pH from Henderson-Hasselbalch and compare.
- Output: a consistency flag.
- Reasoning: the three values must satisfy the equation. A discrepancy of more than about 0.05
  pH units suggests a timing mismatch or a transcription error.
- Branch point: mixed disorders can produce values that look inconsistent but are real; the
  equation still holds if the values are simultaneous.
- Analytic artifact: `abg_consistency_flag`, `predicted_ph`.

**Step 11. Check compensation.**

- Input: pH, PaCO2, HCO3.
- Operation: apply Winter's formula for metabolic acidosis and the expected compensation for
  other disorders.
- Output: a compensation assessment.
- Reasoning: Winter's formula: expected PaCO2 = 1.5 x HCO3 + 8, plus or minus 2. A measured
  PaCO2 outside the range suggests a second disorder.
- Branch point: chronic disorders have different compensation; the formulas are approximations.
- Analytic artifact: `expected_pco2`, `compensation_flag`.

**Step 12. Check the anion gap and delta-delta.**

- Input: Na, Cl, HCO3, albumin.
- Operation: AG = Na - (Cl + HCO3); correct for albumin; delta-delta = (AG - 12) / (24 - HCO3).
- Output: an anion gap and a delta ratio.
- Reasoning: a high anion gap metabolic acidosis with a delta ratio of 1 to 2 is a pure high-gap
  acidosis. A ratio below 1 suggests a mixed non-gap acidosis. A ratio above 2 suggests a mixed
  metabolic alkalosis.
- Branch point: the normal anion gap varies by laboratory and by analyzer.
- Analytic artifact: `anion_gap`, `corrected_ag`, `delta_ratio`.

**Step 13. Check osmolality.**

- Input: measured osmolality, Na, glucose, BUN.
- Operation: calculated osmolality = 2 x Na + glucose/18 + BUN/2.8; gap = measured - calculated.
- Output: an osmolal gap.
- Reasoning: a gap above about 10 mOsm/kg suggests an unmeasured osmole (toxic alcohol, mannitol,
  glycol). A negative gap suggests a calculation or unit error.
- Branch point: the formula varies by institution; use the local one.
- Analytic artifact: `osmolal_gap`.

**Step 14. Check electrolyte corrections.**

- Input: sodium, glucose, calcium, albumin.
- Operation: corrected Na = Na + 1.6 x (glucose - 100)/100 for glucose 100-400; corrected Ca =
  Ca + 0.8 x (4.0 - albumin).
- Output: corrected values.
- Reasoning: hyperglycemia dilutes sodium; hypoalbuminemia lowers total calcium without changing
  ionized calcium. Using uncorrected values can misclassify severity.
- Branch point: the sodium correction factor is debated and varies at very high glucose; the
  calcium correction is unreliable in critical illness.
- Analytic artifact: `corrected_sodium`, `corrected_calcium`.

**Step 15. Check oxygenation.**

- Input: PaO2, FiO2, PaCO2, SpO2.
- Operation: compute the P/F ratio and the A-a gradient; compare SpO2 with PaO2.
- Output: an oxygenation flag.
- Reasoning: an SpO2 of 100% with a PaO2 of 40 mmHg is not physiologically coherent. A P/F ratio
  below 300 suggests impaired oxygenation. The A-a gradient increases with age and with
  shunt.
- Branch point: SpO2 is affected by device bias, so a mismatch may be measurement error rather
  than data error.
- Analytic artifact: `pf_ratio`, `aa_gradient`, `spo2_pao2_flag`.

**Step 16. Check renal and dosing consistency.**

- Input: creatinine, age, sex, weight, eGFR, CrCl.
- Operation: recompute both equations; compare the recorded values.
- Output: a consistency flag.
- Reasoning: an eGFR that does not match the equation, or a CrCl that omits the sex factor,
  suggests a calculation error. A dose that exceeds the labeled maximum for the renal function
  suggests a prescribing or data error.
- Branch point: the equation version matters; 2009 and 2021 CKD-EPI give different values.
- Analytic artifact: `egfr_recomputed`, `crcl_recomputed`, `dose_plausibility`.

### Layer 4: cohort

**Step 17. Check demographics.**

- Input: age, sex, race, ethnicity.
- Operation: compare distributions with the setting and the source population.
- Output: a demographic report.
- Reasoning: an adult ICU with a mean age of 20 is implausible. A pediatric dataset with 40% of
  patients over 65 is implausible. A sex distribution of 95% male in an obstetric cohort is
  impossible.
- Branch point: a specialty clinic can have a skewed distribution that is real.
- Analytic artifact: `demographic_report`.

**Step 18. Check prevalence and outcomes.**

- Input: cohort labels and outcomes.
- Operation: compare with published ranges for the setting, or with a registry.
- Output: a prevalence report.
- Reasoning: a sepsis prevalence of 40% in an outpatient cohort suggests a label error. A
  mortality of 0.1% in an ICU suggests a linkage error.
- Branch point: a highly selected cohort can have a prevalence outside the usual range; the
  selection criteria explain it.
- Analytic artifact: `prevalence_report`.

**Step 19. Check missingness by site and group.**

- Input: missingness indicators.
- Operation: compute by site, year, age group, sex, and payer.
- Output: a missingness heatmap.
- Reasoning: missingness that varies by site or group is a signal of workflow bias, not random
  absence. A model trained on such data can learn the missingness as a proxy for the group.
- Branch point: small sites have unstable estimates; report counts with the proportions.
- Analytic artifact: `missingness_table`.

**Step 20. Check duplicate patients and encounters.**

- Input: identifiers.
- Operation: count records per patient; identify overlapping encounters.
- Output: a duplication report.
- Reasoning: a patient with two overlapping inpatient encounters is either a transfer or a data
  error. A patient with 50 outpatient visits in a year may be a high utilizer or a duplicate
  record.
- Branch point: transfers create legitimate overlaps; the check should flag, not delete.
- Analytic artifact: `encounter_overlap_flag`.

### Layer 5: time

**Step 21. Check drift by year.**

- Input: feature distributions by year.
- Operation: compare means, missingness, and code frequencies across years.
- Output: a drift report.
- Reasoning: a step change at a fiscal year boundary suggests a code or assay change. A gradual
  change suggests a population or practice change.
- Branch point: a real change (new treatment, pandemic) can look like drift; the check prompts
  investigation, not correction.
- Analytic artifact: `drift_report`, `change_point`.

**Step 22. Check version changes.**

- Input: code system, assay, equation, device versions.
- Operation: record the version and test for a step change at the transition.
- Output: a versioned analysis.
- Reasoning: ICD-10-CM changes annually; eGFR equations changed in 2021; troponin assays change;
  EHR migrations change documentation. Each can produce an artifact.
- Branch point: if the version is unknown, the analysis cannot distinguish artifact from change.
- Analytic artifact: `version_field`, `transition_test`.

### Layer 6: workflow

**Step 23. Check order-to-result timing.**

- Input: order time, collect time, result time.
- Operation: compute intervals; flag negative or implausibly long intervals.
- Output: a timing report.
- Reasoning: a result before the order is a timestamp error. A turnaround time of 10 minutes for
  a send-out test is impossible. A turnaround time of 48 hours may be real but should be
  explained.
- Branch point: point-of-care tests have different timing than central laboratory tests.
- Analytic artifact: `tat_minutes`, `timing_flag`.

**Step 24. Check documentation timing.**

- Input: note signed time, encounter time.
- Operation: flag notes signed before the encounter or long after discharge.
- Output: a documentation timing report.
- Reasoning: a discharge summary signed before discharge is a draft. A note signed weeks later
  may be a late entry. Both affect availability at time zero.
- Branch point: some notes are signed in batches; the batch time is not the event time.
- Analytic artifact: `note_timing_flag`.

**Step 25. Check device plausibility.**

- Input: device type, measurements.
- Operation: apply device-specific rules.
- Output: a device flag.
- Reasoning: an arterial line waveform with a flat segment suggests damping. A non-invasive
  blood pressure in a patient with an LVAD may be unreliable. A pulse oximeter with a poor
  plethysmographic waveform may be inaccurate.
- Branch point: device data may not be linked to the measurement.
- Analytic artifact: `device_flag`, `signal_quality`.

**Step 26. Document every rule and its action.**

- Input: the rule set.
- Operation: record the rule, the threshold, the action, and the rationale.
- Output: a plausibility rulebook.
- Reasoning: a rule without a rationale cannot be reviewed or updated. A rule that deletes data
  without recording the deletion is not reproducible.
- Branch point: rules should be versioned with the dataset.
- Analytic artifact: `rule_id`, `rule_version`, `action`, `rationale`.

**Step 27. Quantify the effect of cleaning.**

- Input: before and after cohorts.
- Operation: compare size, demographics, prevalence, and outcomes.
- Output: a sensitivity analysis.
- Reasoning: cleaning changes the cohort. The change should be reported, not hidden. If the
  conclusion changes after cleaning, the conclusion was fragile.
- Branch point: if cleaning removes a large fraction, the data source may be unsuitable.
- Analytic artifact: `cleaning_impact_report`.

**Step 28. Escalate critical values.**

- Input: flagged values.
- Operation: distinguish a data error from a clinical critical value; route the latter to a
  clinician.
- Output: an escalation record.
- Reasoning: a potassium of 7.2 mmol/L may be a real critical value or a hemolyzed sample. The
  pipeline should not silently delete it. It should flag it and, where the use case warrants,
  route it for review.
- Branch point: a research pipeline is not a clinical alerting system; the escalation path must
  be designed, not assumed.
- Analytic artifact: `critical_flag`, `escalation_path`.

## 6. Worked example

A research team receives an extract of 5,000 intensive care unit (ICU) stays from a single
hospital for a sepsis mortality model. The extract has been through an institutional review board
process and a de-identification step. The team is asked to decide whether the data are fit for
the intended model.

### 6.1 The audit

The team runs a rule set across six layers. The results:

| Check | Rule | Records flagged | Action | Stays excluded |
|---|---|---:|---|---:|
| Temperature range | 25-45 °C | 12 | Exclude | 12 |
| Heart rate range | 20-250 beats/min | 38 | Exclude 20, correct 18 | 20 |
| Systolic pressure range | 40-300 mmHg | 21 | Exclude 10, correct 11 | 10 |
| Weight range | 20-300 kg | 45 | Exclude 15, correct 30 | 15 |
| Temporal order | Discharge after admission | 15 | Exclude | 15 |
| Age range | 0-105 years | 3 | Exclude | 3 |
| Duplicate lab rows | Exact duplicate within 60 s | 63 rows | Deduplicate | 0 |
| ABG consistency | Predicted pH within 0.05 | 8 | Exclude | 8 |
| SpO2/PaO2 mismatch | SpO2 above 98% with PaO2 below 60 mmHg | 120 | Flag only | 0 |
| Missing sex | Null or unknown | 4 | Exclude | 4 |
| **Total** |  |  |  | **87** |

The exclusions total 87 stays, leaving 4,913. The duplicate rows are removed without excluding a
stay. The SpO2/PaO2 mismatch is flagged but retained, because the mismatch may reflect device
bias rather than a data error.

### 6.2 The cohort impact

Before cleaning:

```text
Mortality = 410 / 5,000 = 8.20%
```

After cleaning, 31 of the 410 deaths are among the 87 excluded stays:

```text
Deaths remaining = 410 - 31 = 379
Cohort remaining = 5,000 - 87 = 4,913
Mortality = 379 / 4,913 = 7.71%
```

The cleaning reduces the mortality estimate by 0.49 percentage points, or about 6% relative. That
is not trivial. The excluded patients were sicker than average, which is why the mortality fell.
A paper that reports only the cleaned estimate without the before-and-after comparison hides a
selection effect.

### 6.3 The acid-base case

One stay has the following arterial blood gas and chemistry:

| Variable | Value | Unit |
|---|---:|---|
| pH | 7.30 | — |
| PaCO2 | 40 | mmHg |
| HCO3 | 19 | mmol/L |
| Sodium | 138 | mmol/L |
| Chloride | 104 | mmol/L |
| Albumin | 3.0 | g/dL |
| Glucose | 180 | mg/dL |
| BUN | 30 | mg/dL |
| Measured osmolality | 300 | mOsm/kg |

Step 1: Henderson-Hasselbalch.

```text
Predicted pH = 6.1 + log10(HCO3 / (0.03 x PaCO2))
             = 6.1 + log10(19 / (0.03 x 40))
             = 6.1 + log10(19 / 1.2)
             = 6.1 + log10(15.833)
             = 6.1 + 1.1996
             = 7.2996
```

The recorded pH is 7.30, a difference of 0.0004. The triple is internally consistent.

Step 2: anion gap.

```text
AG = Na - (Cl + HCO3) = 138 - (104 + 19) = 15 mmol/L
```

A normal anion gap is approximately 8-12 mmol/L without potassium. Correct for albumin:

```text
Corrected AG = AG + 2.5 x (4.0 - albumin)
             = 15 + 2.5 x (4.0 - 3.0)
             = 15 + 2.5
             = 17.5 mmol/L
```

Step 3: delta-delta.

```text
Delta ratio = (Corrected AG - 12) / (24 - HCO3)
            = (17.5 - 12) / (24 - 19)
            = 5.5 / 5
            = 1.1
```

A delta ratio of 1.1 is in the 1-2 range, consistent with a pure high-anion-gap metabolic
acidosis. There is no evidence of a mixed non-gap acidosis or a mixed metabolic alkalosis.

Step 4: compensation.

```text
Expected PaCO2 = 1.5 x HCO3 + 8 = 1.5 x 19 + 8 = 36.5 mmHg
Expected range = 34.5-38.5 mmHg
```

The measured PaCO2 is 40 mmHg, slightly above the expected range. That suggests a mild additional
respiratory acidosis, or a measurement timing difference. It is not a dramatic inconsistency, but
it is worth flagging.

Step 5: osmolal gap.

```text
Calculated osmolality = 2 x Na + glucose/18 + BUN/2.8
                      = 2 x 138 + 180/18 + 30/2.8
                      = 276 + 10 + 10.7
                      = 296.7 mOsm/kg

Osmolal gap = 300 - 296.7 = 3.3 mOsm/kg
```

A gap below 10 mOsm/kg is normal. There is no evidence of an unmeasured osmole.

### 6.4 A second acid-base case

Another stay has pH 7.52, PaCO2 28 mmHg, and HCO3 22 mmol/L.

```text
Predicted pH = 6.1 + log10(22 / (0.03 x 28))
             = 6.1 + log10(22 / 0.84)
             = 6.1 + log10(26.19)
             = 6.1 + 1.418
             = 7.518
```

The recorded pH is 7.52, consistent with the triple. The primary disorder is respiratory
alkalosis: the PaCO2 is low and the bicarbonate is near normal. Winter's formula does not apply
to a primary respiratory alkalosis; it is used for metabolic acidosis. A naive application would
compute an expected PaCO2 of 41 mmHg and wrongly conclude that the patient has a second disorder.

This is the most common misuse of Winter's formula: applying it outside its indication. The
formula estimates the compensatory response to a metabolic acidosis, not the expected value in
every acid-base disorder.

### 6.5 The SpO2/PaO2 mismatch

The 120 flagged records have an SpO2 above 98% and a PaO2 below 60 mmHg. The combination is
physiologically unusual at sea level on room air. Possible explanations:

| Explanation | Mechanism | Action |
|---|---|---|
| Device bias | Pulse oximeter overestimates in dark skin or poor perfusion | Compare with co-oximetry |
| Timing mismatch | SpO2 and ABG drawn at different times | Align timestamps |
| Supplemental oxygen | SpO2 high on oxygen, PaO2 low before oxygen | Record FiO2 |
| Data error | One value transcribed incorrectly | Review source |

The correct action is to flag and investigate, not to delete. Deleting all 120 would remove a
signal that may be a real fairness issue.

### 6.6 The data quality report

A reviewer would expect a report with:

1. **Scope:** the extract, the date range, the source systems, the row counts.
2. **Rules:** each rule, its threshold, its rationale, and its action.
3. **Counts:** flagged, excluded, corrected, and retained.
4. **Impact:** the change in cohort size, demographics, prevalence, and outcome.
5. **Limitations:** what the rules cannot detect, and what remains unknown.
6. **Recommendation:** proceed, proceed with conditions, or stop.

The recommendation in this case is "proceed with conditions": retain the 120 SpO2/PaO2 flags as a
sensitivity analysis, report the before-and-after mortality, and validate the sepsis label on a
chart-reviewed sample.

### 6.7 What the audit cannot catch

The audit cannot detect:

- A patient who was never in the ICU but appears in the extract.
- A sepsis label that is clinically wrong but internally consistent.
- A treatment that was ordered but not given.
- A social factor that is missing because it was never asked.
- A model that is accurate and useless because no one acts on it.

Those require chart review, workflow analysis, and clinical judgment. The plausibility audit is a
filter, not a validation.

## 7. Code, query, or calculation

The block implements the core checks and reproduces the acid-base arithmetic.

```python
"""Clinical plausibility checks: ranges, acid-base, and cleaning impact."""

import math
from typing import Optional


def hh_ph(hco3: float, paco2: float) -> float:
    """Henderson-Hasselbalch prediction."""
    return 6.1 + math.log10(hco3 / (0.03 * paco2))


def winters_expected_paco2(hco3: float) -> float:
    """Expected PaCO2 in metabolic acidosis, +/- 2 mmHg."""
    return 1.5 * hco3 + 8


def anion_gap(na: float, cl: float, hco3: float) -> float:
    return na - (cl + hco3)


def corrected_anion_gap(ag: float, albumin: float) -> float:
    return ag + 2.5 * (4.0 - albumin)


def delta_ratio(ag_corrected: float, hco3: float) -> float:
    return (ag_corrected - 12.0) / (24.0 - hco3)


def osmolal_gap(measured: float, na: float, glucose: float, bun: float) -> float:
    calculated = 2 * na + glucose / 18.0 + bun / 2.8
    return measured - calculated


def corrected_sodium(na: float, glucose: float) -> float:
    """Valid for glucose roughly 100-400 mg/dL; use 2.4 above 400."""
    return na + 1.6 * (glucose - 100.0) / 100.0


def flag_record(rec: dict) -> list[str]:
    flags = []
    if not (25 <= rec.get("temp", 37) <= 45):
        flags.append("temperature_range")
    if not (20 <= rec.get("hr", 80) <= 250):
        flags.append("heart_rate_range")
    if not (40 <= rec.get("sbp", 120) <= 300):
        flags.append("systolic_range")
    if rec.get("los_days", 0) < 0:
        flags.append("negative_los")
    if not (0 < rec.get("age", 40) <= 105):
        flags.append("age_range")
    if rec.get("sex") not in ("F", "M"):
        flags.append("missing_sex")
    return flags


records = [
    {"id": 1, "temp": 36.8, "hr": 88, "sbp": 120, "los_days": 3, "age": 64, "sex": "F"},
    {"id": 2, "temp": 45.5, "hr": 92, "sbp": 118, "los_days": 2, "age": 71, "sex": "M"},
    {"id": 3, "temp": 37.1, "hr": 300, "sbp": 110, "los_days": 5, "age": 55, "sex": "M"},
    {"id": 4, "temp": 36.9, "hr": 80, "sbp": 0, "los_days": 4, "age": 60, "sex": "F"},
    {"id": 5, "temp": 37.0, "hr": 76, "sbp": 130, "los_days": -1, "age": 48, "sex": "M"},
    {"id": 6, "temp": 36.7, "hr": 90, "sbp": 125, "los_days": 6, "age": 0, "sex": "F"},
    {"id": 7, "temp": 38.2, "hr": 110, "sbp": 100, "los_days": 8, "age": 82, "sex": None},
]

counts: dict[str, int] = {}
for rec in records:
    for flag in flag_record(rec):
        counts[flag] = counts.get(flag, 0) + 1

print("flag counts:")
for flag in sorted(counts):
    print(f"  {flag}: {counts[flag]}")

ph = hh_ph(19, 40)
print()
print(f"predicted pH: {ph:.4f}")
print(f"Winter expected PaCO2: {winters_expected_paco2(19):.1f} mmHg")
ag = anion_gap(138, 104, 19)
agc = corrected_anion_gap(ag, 3.0)
print(f"anion gap: {ag:.1f} mmol/L")
print(f"corrected anion gap: {agc:.1f} mmol/L")
print(f"delta ratio: {delta_ratio(agc, 19):.2f}")
print(f"osmolal gap: {osmolal_gap(300, 138, 180, 30):.1f} mOsm/kg")
print(f"corrected sodium at glucose 600: {corrected_sodium(132, 600):.1f} mmol/L")

# Cleaning impact
before_n, before_deaths = 5000, 410
after_n = before_n - 87
after_deaths = before_deaths - 31
print()
print(f"mortality before: {before_deaths / before_n * 100:.2f}%")
print(f"mortality after: {after_deaths / after_n * 100:.2f}%")
print(f"absolute change: {(after_deaths / after_n - before_deaths / before_n) * 100:.2f} points")

assert round(ph, 4) == 7.2996
assert round(agc, 1) == 17.5
assert round(delta_ratio(agc, 19), 2) == 1.1
assert round(osmolal_gap(300, 138, 180, 30), 1) == 3.3
assert round(corrected_sodium(132, 600), 1) == 140.0
```

Expected output:

```text
flag counts:
  age_range: 1
  heart_rate_range: 1
  missing_sex: 1
  negative_los: 1
  systolic_range: 1
  temperature_range: 1

predicted pH: 7.2996
Winter expected PaCO2: 36.5 mmHg
anion gap: 15.0 mmol/L
corrected anion gap: 17.5 mmol/L
delta ratio: 1.10
osmolal gap: 3.3 mOsm/kg
corrected sodium at glucose 600: 140.0 mmol/L

mortality before: 8.20%
mortality after: 7.71%
absolute change: -0.49 points
```

The small table exercises each rule once. In production, the same functions run over millions of
rows and produce counts rather than flags.

A SQL query for temporal violations:

```sql
-- Stays with discharge before admission, or a negative length of stay.
SELECT stay_id,
       patient_id,
       admit_ts,
       discharge_ts,
       EXTRACT(EPOCH FROM (discharge_ts - admit_ts)) / 3600.0 AS los_hours
FROM icu_stays
WHERE discharge_ts < admit_ts
   OR EXTRACT(EPOCH FROM (discharge_ts - admit_ts)) / 3600.0 < 0;
```

A second query for unit-mixing detection:

```sql
-- Bimodality check for glucose values, a signature of mg/dL and mmol/L mixing.
SELECT width_bucket(value, 0, 600, 60) AS bucket,
       COUNT(*) AS n
FROM lab_results
WHERE loinc_code = '2345-7'
GROUP BY bucket
ORDER BY bucket;
```

A distribution with one cluster near 5 and another near 100 is a unit mix, not a physiological
bimodality. The check should be run before any conversion.

## 8. Data, standards, and regulatory context

**Data integrity principles.** In regulated clinical research, data integrity is described by the
ALCOA+ principles: attributable, legible, contemporaneous, original, accurate, complete,
consistent, enduring, and available. These principles are part of Good Clinical Practice and are
enforced through audit trails, source documentation, and inspection. They apply to trial data
submitted to regulators; they are a useful standard for any clinical dataset, even when not
legally required.

**Electronic records.** In the United States, 21 CFR Part 11 defines requirements for electronic
records and electronic signatures, including audit trails and access controls. The European
Union has equivalent requirements in Annex 11 of Good Manufacturing Practice. These are binding
for regulated submissions, not for every research dataset.

**Clinical trial data standards.** The Clinical Data Interchange Standards Consortium (CDISC)
defines the Study Data Tabulation Model (SDTM) and the Analysis Data Model (ADaM). Define-XML
describes the metadata. Regulators expect validation against these standards, and tools such as
Pinnacle 21 check conformance. Conformance is not the same as clinical plausibility; a dataset can
be perfectly conformant and physiologically impossible.

**Observational data standards.** The Observational Medical Outcomes Partnership (OMOP) Common
Data Model, maintained by the Observational Health Data Sciences and Informatics (OHDSI)
community, provides a common structure for observational data. OHDSI's DataQualityDashboard runs
a large set of checks across plausibility, conformance, and completeness. Achilles profiles the
data. These tools are open source and widely used.

**FHIR and USCDI.** FHIR resources carry timestamps, units, and status codes that support
plausibility checks. USCDI defines the minimum data elements for interoperability. Neither
guarantees that a value is correct.

**Public data caveats.** The MIMIC-IV documentation states that implausible values may be present
as an artifact of the archival process and that the database is not cleaned by its custodians.
This is an explicit warning, not a defect to be ignored. A team using MIMIC must bring its own
plausibility layer.

**Privacy.** De-identification can introduce artifacts: date shifting, age capping at 89, removal
of rare identifiers, and geographic aggregation. These transformations can break temporal checks
or create impossible ages. The de-identification method should be documented alongside the
plausibility rules.

> **Note:** This section is educational and is not medical advice or legal advice. Data integrity
> requirements depend on the regulatory context, the data source, and the intended use.

## 9. Industry practice and case evidence

**Public critical care data contain documented implausible values.** The MIMIC-IV documentation
warns that implausible values may be present. Public issue reports have documented a height of
825,070 inches in the chart data, thousands of weights below 25 kg or above 300 kg, and timestamp
offsets of about one year in a small number of rows. These are not rare edge cases; they are the
normal residue of clinical data entry and archival. A pipeline without hard limits will carry
them into a model.

**Data quality frameworks are available and underused.** OHDSI's DataQualityDashboard implements
hundreds of checks across plausibility, conformance, and completeness for OMOP data. The checks
include impossible values, impossible temporal relationships, and implausible distributions. The
existence of the tool does not mean it is run; many analyses skip this step because the data
"look fine." The documented fact is the tool; the inference is that the step is often omitted.

**Phenotype validation changes prevalence estimates.** Studies that compare code-based or
note-based phenotypes with chart review consistently find imperfect sensitivity and positive
predictive value. The illustrative sepsis cohort in topic 05 showed a code sensitivity of 43.6%
against a clinical-criteria reference. The precise value is site-specific, but the direction is
consistent: administrative labels are not diagnoses, and the gap is large enough to change
conclusions.

**Unit errors are a recurring safety problem.** Pediatric medication errors from pound-kilogram
confusion and decimal errors are well documented in patient safety literature. The same errors
appear in datasets as implausible weights or doses. A hard range check catches the extreme cases;
a unit column and a distribution check catch the subtler ones.

**Cleaning changes estimates.** The worked example shows a 0.49 percentage point change in
mortality after excluding 87 of 5,000 stays. The direction and magnitude depend on the data, but
the principle is general: cleaning is an analytic decision, and its effect should be reported.
A paper that reports only the cleaned cohort without the before-and-after comparison is hiding a
selection step.

**What is documented versus inferred.** The MIMIC caveat, the OHDSI tools, and the phenotype
validation findings are documented. The claim that a specific cleaning rule improves a specific
model is not; it requires local evaluation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Zero treated as a measurement | BP or weight distribution spikes at 0 | Missing code | False physiology | Sentinel rules |
| Unit mixing | Bimodal distribution | mg/dL and mmol/L | Wrong thresholds | Unit column; mode check |
| Impossible values retained | Temperature 60 °C | No hard limits | Model learns artifact | Hard range checks |
| Legitimate extremes deleted | Potassium 7.2 removed | Outlier filtering | Safety signal lost | Flag, do not delete |
| Temporal violations ignored | Discharge before admission | Timestamp error | Negative LOS | Temporal rules |
| Duplicate rows counted | Event counts exceed orders | Duplicate extraction | Inflated prevalence | Deduplicate by key |
| ABG triple inconsistent | Predicted pH differs by 0.2 | Timing or transcription | Wrong acid-base label | HH check |
| Winter applied outside acidosis | Expected PaCO2 computed for alkalosis | Formula misuse | False mixed disorder | Indication check |
| Anion gap uncorrected | Normal AG with low albumin | Hypoalbuminemia | Missed high-gap acidosis | Albumin correction |
| Osmolal gap wrong formula | Negative gap | Unit or formula error | Missed toxic alcohol | Local formula; range check |
| Sodium uncorrected | Severe hyponatremia in hyperglycemia | Dilutional effect | Overstated severity | Corrected sodium |
| Cohort prevalence implausible | Sepsis 40% in outpatients | Label error | Wrong model target | Prevalence check |
| Missingness by site ignored | One site missing 80% of labs | Workflow difference | Site confounding | Missingness heatmap |
| Drift mistaken for trend | Step change at fiscal year | Code or assay change | Wrong temporal conclusion | Version field; change-point test |
| Cleaning effect hidden | Cohort size drops 10% | Exclusion | Selection bias | Before-and-after report |
| Critical value deleted | Potassium 7.2 filtered | Outlier rule | Harm | Escalation path |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The hard limits are valid for the population | Range checks depend on them | Age, device, altitude | Stratify limits by group |
| The unit is known | Conversion depends on it | Unit missing or mixed | Unit column; distribution check |
| Timestamps are in one time zone | Temporal checks depend on it | Daylight saving, multi-site | Timezone field; interval audit |
| The reference interval matches the assay | Range flags depend on it | Assay change | Assay identifier |
| The cohort is representative | Prevalence checks depend on it | Selection, referral | Compare with registry |
| Missingness is random | Imputation depends on it | Workflow bias | Missingness by group |
| The cleaning rule is stable | Reproducibility | Rule change | Versioned rulebook |
| The label is valid | Outcome checks depend on it | Code or note error | Chart review |
| The data source is complete | Cohort completeness | Out-of-network, archive gaps | Linkage coverage |
| The clinical context is known | Interpretation depends on it | No indication, no FiO2 | Context fields |

## 12. Exercises

**Exercise 1 — Rule design (application, 45 minutes).**
Write five hard-limit rules for adult vital signs, with the limit, the rationale, and the action.
Include one rule that differs for children. Deliverable: a five-row rule table. Acceptance
criteria: limits justified; actions distinguish reject, correct, and flag; pediatric difference
included. Time: 45 minutes.

**Exercise 2 — Unit detection (calculation, 40 minutes).**
Given a glucose distribution with modes at 5.5 and 110, explain the likely cause and propose a
detection rule. Compute the conversion factor and show what 110 mg/dL is in mmol/L. Deliverable:
a 300-word analysis and a rule. Acceptance criteria: conversion correct; rule is distribution-
based; no assumption that all high values are errors. Time: 40 minutes.

**Exercise 3 — Acid-base consistency (calculation, 50 minutes).**
For pH 7.28, PaCO2 30 mmHg, HCO3 14 mmol/L, compute the predicted pH, the anion gap (Na 140, Cl
100), the albumin-corrected gap (albumin 2.8), the delta ratio, and the Winter expected PaCO2.
State the acid-base diagnosis. Deliverable: a calculation sheet. Acceptance criteria: all values
correct; diagnosis consistent with the numbers. Time: 50 minutes.

**Exercise 4 — Temporal audit (application, 45 minutes).**
Using a synthetic table, find all records with a result before the order, a discharge before
admission, or a negative interval. Quantify each. Deliverable: a table and a 200-word summary.
Acceptance criteria: all three checks implemented; counts correct; timezone caveat stated. Time:
45 minutes.

**Exercise 5 — Cleaning impact (calculation, 50 minutes).**
A cohort of 8,000 has a mortality of 12%. Cleaning excludes 240 patients, 60 of whom died.
Compute the before-and-after mortality and the absolute change. Discuss whether the excluded
patients were sicker. Deliverable: a calculation and a 300-word discussion. Acceptance criteria:
arithmetic correct; selection effect discussed; no causal claim. Time: 50 minutes.

**Exercise 6 — Cohort plausibility (critique, 50 minutes).**
A pediatric sepsis dataset has a mean age of 42 years and a mortality of 35%. Critique the
cohort and propose three checks. Deliverable: a 400-word critique. Acceptance criteria: at least
three implausibilities; checks are specific; next steps proposed. Time: 50 minutes.

**Exercise 7 — Rulebook (design, 75 minutes).**
Write a plausibility rulebook for an ICU dataset. Include at least 15 rules across the six
layers, each with a threshold, action, and rationale. Deliverable: a rulebook table. Acceptance
criteria: all six layers represented; actions differentiated; version field. Time: 75 minutes.

**Exercise 8 — Memo (design and communication, 60 minutes).**
Write a one-page memo to a clinical reviewer summarizing the audit in section 6, the cleaning
impact, and a recommendation. Deliverable: one page, maximum 600 words. Acceptance criteria:
counts and impact included; limitations stated; recommendation is conditional and actionable.
Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** The modes at 5.5 and 110 are consistent with a mixture of mmol/L and
mg/dL. The conversion is mg/dL / 18 = mmol/L, so 110 mg/dL = 6.11 mmol/L. A detection rule could
flag values below 30 as likely mmol/L in a field expected to be mg/dL, or fit a two-component
mixture and flag the smaller mode. The rule should flag, not automatically convert, because some
patients genuinely have values in the overlap range.

**Answer to Exercise 3.** Predicted pH = 6.1 + log10(14/(0.03 x 30)) = 6.1 + log10(15.556) = 6.1
+ 1.192 = 7.292. The recorded pH of 7.28 differs by 0.012, consistent. AG = 140 - (100 + 14) =
26 mmol/L. Corrected AG = 26 + 2.5 x (4.0 - 2.8) = 26 + 3.0 = 29.0. Delta ratio = (29 - 12)/(24 -
14) = 17/10 = 1.7, consistent with a pure high-gap metabolic acidosis. Winter expected PaCO2 =
1.5 x 14 + 8 = 29 mmHg, and the measured PaCO2 is 30, within the expected range. The diagnosis is
a high-anion-gap metabolic acidosis with appropriate respiratory compensation.

**Answer to Exercise 5.** Before: 0.12 x 8,000 = 960 deaths. After: (960 - 60) / (8,000 - 240) =
900 / 7,760 = 0.1160, or 11.60%. The absolute change is -0.40 percentage points. The excluded
patients had a mortality of 60/240 = 25%, higher than the cohort average, so the excluded group
was sicker. The cleaned estimate is lower because the sickest patients were removed.

**Answer to self-check on Henderson-Hasselbalch.** The equation links pH, bicarbonate, and PaCO2.
If two values are known, the third is determined. A recorded triple that does not satisfy the
equation suggests a timing mismatch or a transcription error, not a new disease.

**Answer to self-check on Winter's formula.** Winter's formula estimates the expected PaCO2 in
metabolic acidosis: 1.5 x HCO3 + 8, plus or minus 2. It does not apply to respiratory alkalosis
or metabolic alkalosis. Using it outside its indication produces false conclusions about mixed
disorders.

**Answer to self-check on anion gap.** The anion gap is Na - (Cl + HCO3). It should be corrected
for albumin because hypoalbuminemia lowers the gap. A "normal" uncorrected gap in a patient with
low albumin can hide a high-gap acidosis.

**Rubric for Exercise 7.** Full credit: 15 rules across six layers; thresholds and actions;
rationale for each; version field; at least one rule that flags rather than deletes. Half credit:
rules present but uneven across layers. No credit: only range checks.

**Rubric for Exercise 8.** Full credit: counts, impact, limitations, conditional recommendation,
reviewer actions. Half credit: summary without impact. No credit: recommendation without
evidence.

## 14. Interview questions

**1. What is a plausibility check, and how is it different from a range check?**
A strong answer places range checks as one layer among several, including temporal,
cross-variable, cohort, and drift checks. A weak answer equates the two.

**2. How would you detect a unit error?**
A strong answer uses a unit column, distribution modes, and cross-variable consistency, and
flags rather than silently converts. A weak answer says "check the range."

**3. Why is a zero blood pressure usually not a measurement?**
A strong answer explains sentinel values and missing codes. A weak answer says zero is
impossible.

**4. What is the Henderson-Hasselbalch equation used for in data quality?**
A strong answer explains internal consistency of the acid-base triple and timing mismatches. A
weak answer says it calculates pH.

**5. When does Winter's formula not apply?**
A strong answer says it applies to metabolic acidosis, not respiratory or metabolic alkalosis. A
weak answer says it always applies.

**6. Why correct the anion gap for albumin?**
A strong answer explains that hypoalbuminemia lowers the gap and can mask a high-gap acidosis. A
weak answer says it is more accurate.

**7. How would you audit a cohort for implausible prevalence?**
A strong answer compares with a registry or published range for the setting and examines the
label definition. A weak answer says "check the rate."

**8. What is the effect of cleaning on an outcome estimate?**
A strong answer explains selection and shows a before-and-after comparison. A weak answer says
cleaning improves accuracy.

**9. How do you handle an extreme but real value?**
A strong answer flags and retains it, with a sensitivity analysis. A weak answer deletes it as an
outlier.

**10. Why is missingness by site a plausibility issue?**
A strong answer explains workflow bias and the risk of learning site as a proxy. A weak answer
says missing data are a problem.

**11. What is drift, and how do you detect it?**
A strong answer compares distributions by year and looks for step changes at known version
boundaries. A weak answer says "the data change."

**12. What can a plausibility audit not catch?**
A strong answer names label validity, workflow, social factors, and clinical utility. A weak
answer says it catches all data errors.

## 15. Key terms

- **ALCOA+** — data integrity principles.
- **Anion gap** — Na minus (Cl + HCO3).
- **A-a gradient** — alveolar-arterial oxygen difference.
- **Audit trail** — record of changes to data.
- **CDISC** — Clinical Data Interchange Standards Consortium.
- **Cohort check** — distribution check across the cohort.
- **Completeness** — proportion of expected values present.
- **Consistency** — agreement across related values.
- **Corrected calcium** — calcium adjusted for albumin.
- **Corrected sodium** — sodium adjusted for hyperglycemia.
- **Cross-variable check** — joint consistency of related values.
- **Data integrity** — completeness, consistency, and accuracy.
- **Delta check** — comparison with a prior value.
- **Delta-delta** — change in anion gap over change in bicarbonate.
- **Drift** — change in distribution over time.
- **Duplicate check** — detection of repeated records.
- **Henderson-Hasselbalch** — pH, bicarbonate, and PaCO2 relationship.
- **Impossibility** — a value incompatible with life.
- **Imputation** — filling a missing value.
- **Missing code** — a value used to represent missingness.
- **OHDSI** — Observational Health Data Sciences and Informatics.
- **OMOP** — Observational Medical Outcomes Partnership common data model.
- **Osmolal gap** — measured minus calculated osmolality.
- **Outlier** — a value far from the rest.
- **P/F ratio** — PaO2 divided by FiO2.
- **Plausibility** — consistency with the patient and measurement.
- **Range check** — comparison with hard limits.
- **Sentinel value** — a code for missing or special values.
- **Shock index** — HR divided by SBP.
- **Temporal check** — verification of time order.
- **Truncation** — removing values beyond a limit.
- **Unit check** — verification of units.
- **Validity** — whether a value measures what it claims.
- **Winsorization** — capping extreme values.
- **Winter's formula** — expected PaCO2 in metabolic acidosis.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Standards and guidance**

- US Food and Drug Administration. 21 CFR Part 11: Electronic Records; Electronic Signatures.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
- International Council for Harmonisation. ICH E6(R3): Good Clinical Practice.
  https://database.ich.org/
- Clinical Data Interchange Standards Consortium. SDTM and ADaM.
  https://www.cdisc.org/standards
- Observational Health Data Sciences and Informatics. DataQualityDashboard.
  https://ohdsi.github.io/DataQualityDashboard/
- Observational Health Data Sciences and Informatics. Achilles.
  https://ohdsi.github.io/Achilles/
- Health Level Seven International. FHIR R4.
  https://hl7.org/fhir/R4/

**Evidence**

- Johnson A, Bulgarelli L, Pollard T, et al. MIMIC-IV documentation, version 3.1, PhysioNet,
  2024. https://physionet.org/content/mimiciv/3.1/
- MIT-LCP. Heights and weights in MIMIC-IV v2.2, issue 1781, 2024.
  https://github.com/MIT-LCP/mimic-code/issues/1781
- MIT-LCP. charttime offsets in MIMIC-IV, issue 1694, 2024.
  https://github.com/MIT-LCP/mimic-code/issues/1694
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*, 2021.
  https://doi.org/10.1001/jamainternmed.2021.2626

**Background reading**

- Olson JE, et al. *Data Quality: The Accuracy Dimension*.
- Bickley LS. *Bates' Guide to Physical Examination and History Taking*, 13th edition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_special-populations.md`](09_special-populations.md)
- First topic: [`01_clinical-reasoning-and-workflow.md`](01_clinical-reasoning-and-workflow.md)
- Related topic: [`03_laboratory-medicine.md`](03_laboratory-medicine.md)
- Related topic: [`08_documentation-and-clinical-narratives.md`](08_documentation-and-clinical-narratives.md)
- Survey file: [`../../01_clinical_foundations.md`](../../01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

