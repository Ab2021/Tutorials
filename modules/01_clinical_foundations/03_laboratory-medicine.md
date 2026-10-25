# Laboratory Medicine

> Module 01 — Clinical Foundations for Data Scientists
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

---

## 1. Why this topic exists

Laboratory results are the backbone of clinical data science. They are numeric, timestamped,
repeated, and apparently objective. They are also the output of a multi-stage measurement process
in which the specimen, the tube, the transport time, the assay, the calibration, the reference
interval, and the reporting rule all shape the number that reaches your table.

The industry failures are predictable.

**Failure 1: pooling incompatible assays.** Two hospitals measure troponin with different assays
and different 99th-percentile upper reference limits. A feature that pools them creates a
site-specific threshold effect that a model can learn as a site effect. The model may look good
in aggregate and fail when one site changes its assay.

**Failure 2: treating a result as a diagnosis.** A ferritin of 8 ng/mL strongly suggests iron
deficiency. A ferritin of 45 ng/mL does not exclude it, because ferritin is an acute-phase
reactant and rises with inflammation. A model that uses a single cutoff will misclassify patients
with inflammatory disease.

**Failure 3: ignoring pre-analytical error.** Hemolysis, delayed processing, wrong tube, and
insufficient volume change results before any assay runs. A potassium of 5.8 mmol/L in a
hemolyzed sample may be 4.1 mmol/L when repeated. If your pipeline does not carry a hemolysis
index, it will treat a specimen-handling artifact as hyperkalemia.

**Failure 4: hard-coding adult reference ranges.** Reference intervals are derived from a
reference population, usually the central 95% of apparently healthy individuals, and they differ
by age, sex, pregnancy status, assay, and laboratory. A creatinine of 1.1 mg/dL is within the
adult female range at one laboratory and above it at another.

This topic teaches you to read a laboratory result as a measurement with provenance, not as a
fact. It covers specimen handling, assay performance, reference intervals, test characteristics,
and the specific failure modes that appear in laboratory data.

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a laboratory result from order to report and **identify** at least six points where
   error can enter.
2. **Select** the correct specimen tube and handling requirement for common tests.
3. **Explain** how a reference interval is established, what the central 95% means, and why 5% of
   healthy people fall outside it.
4. **Compute** sensitivity, specificity, predictive values, and likelihood ratios from a 2x2
   table, and **interpret** them at two different prevalences.
5. **Convert** between common conventional and SI units for glucose, creatinine, bilirubin,
   cholesterol, triglycerides, calcium, and hemoglobin.
6. **Detect** pre-analytical artifacts (hemolysis, dilution, wrong tube) and **describe** their
   effect on common analytes.
7. **Critique** a laboratory-based phenotype for assay dependence, reference-interval mismatch,
   and delta-check failures.

## 3. The core mental model

A laboratory result is a chain, and every link can change the number.

```text
   PATIENT
      |
      v
   ORDER ---------> indication, priority, time
      |
      v
   COLLECTION -----> tourniquet time, tube, order of draw, volume
      |
      v
   TRANSPORT ------> time, temperature, agitation
      |
      v
   PROCESSING -----> centrifugation, aliquoting, storage
      |
      v
   ASSAY ----------> method, calibration, interference, precision
      |
      v
   VALIDATION -----> delta check, critical value, autoverification
      |
      v
   REPORT ---------> units, reference interval, flags, comments
      |
      v
   EHR ------------> interface, rounding, display
```

The first four links are pre-analytical. They account for the majority of laboratory errors. The
fifth is analytical. The last three are post-analytical.

### 3.1 The reference interval is a population statement

A reference interval is usually the central 95% of values from a reference population, estimated
as the 2.5th and 97.5th percentiles. It is not a threshold for disease. By construction, 5% of
reference individuals fall outside it. With 20 tests on a panel, the probability that at least
one result is outside its interval in a healthy person is high:

```text
P(at least one abnormal) = 1 - 0.95^20 = 1 - 0.3585 = 0.6415
```

About 64% of healthy people will have at least one flagged result on a 20-test panel. This is
not a reason to ignore flags. It is a reason to interpret them in context.

### 3.2 The test characteristics are not properties of the test alone

Sensitivity and specificity are measured in a specific population, at a specific threshold, with
a specific assay, against a specific reference standard. Change any of those and the numbers
change. Predictive values depend additionally on prevalence. A test with 90% sensitivity and 95%
specificity has a PPV of 15.3% at 1% prevalence and 78.3% at 20% prevalence. The test did not
change; the population did.

### 3.3 The delta check is a temporal model

A delta check compares a result with the patient's previous result and flags a change larger than
expected. It is a simple, effective post-analytical tool for catching specimen mix-ups, unit
errors, and sudden physiological change. A sodium falling from 138 to 128 mmol/L in 24 hours is
not impossible, but it is large enough to warrant review.

Delta checks are also a model of expected variation. The threshold must be analyte-specific and
time-aware. A hemoglobin that falls 2 g/dL in an hour after trauma is expected; the same fall
over a year may be a slow bleed or a laboratory error.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Analyte | The substance being measured | Used loosely for the test | None |
| Specimen | The material analyzed | Confused with the order | Blood, urine, CSF, tissue |
| Serum | Plasma without clotting factors | Used interchangeably with plasma | None |
| Plasma | Liquid blood component after anticoagulation | Used interchangeably with serum | None |
| Whole blood | Blood with cells and plasma | Used for tests requiring cells | None |
| Pre-analytical | Everything before the assay | Ignored in data quality | Phase label |
| Analytical | The measurement itself | Assumed error-free | Phase label |
| Post-analytical | Validation and reporting | Assumed instantaneous | Phase label |
| Reference interval | Central 95% of a reference population | Treated as a disease threshold | Analyte-specific units |
| Reference population | The group used to derive the interval | Assumed to match the patient | Defined by inclusion criteria |
| Sensitivity | TP / (TP + FN) | Confused with PPV | Proportion |
| Specificity | TN / (TN + FP) | Confused with NPV | Proportion |
| PPV | TP / (TP + FP) | Reported as test-intrinsic | Proportion |
| NPV | TN / (TN + FN) | Reported as test-intrinsic | Proportion |
| Likelihood ratio | Factor by which a result changes odds | Confused with relative risk | Dimensionless |
| Prevalence | Proportion with disease in the population | Ignored in test interpretation | Proportion |
| Accuracy | Agreement with the true value | Confused with precision | Analyte units |
| Precision | Reproducibility of repeated measurements | Confused with accuracy | CV, % |
| Bias | Systematic difference from the true value | Confused with imprecision | Analyte units or % |
| Coefficient of variation (CV) | Standard deviation divided by the mean | Reported without the mean | % |
| Limit of detection (LoD) | Lowest concentration reliably distinguished from zero | Confused with limit of quantification | Analyte units |
| Limit of quantification (LoQ) | Lowest concentration with acceptable precision and bias | Assumed equal to LoD | Analyte units |
| Linearity | Range over which the assay response is proportional | Assumed to cover all values | Analyte units |
| Interference | A substance that changes the measured value | Confused with imprecision | Analyte units or index |
| Hemolysis index | Spectrophotometric estimate of red cell breakdown | Ignored when present | 0-4+ or mg/dL |
| Lipemia index | Estimate of lipid turbidity | Ignored when present | 0-4+ |
| Icterus index | Estimate of bilirubin interference | Ignored when present | 0-4+ |
| Delta check | Comparison with a prior result | Applied with one threshold for all analytes | Analyte-specific |
| Critical value | A result requiring immediate notification | Confused with an abnormal value | Analyte-specific |
| Turnaround time (TAT) | Time from order or collection to result | Measured from the wrong start | Minutes |
| Autoverification | Automatic release of results meeting rules | Assumed to be error-free | Rule set |
| Standardization | Alignment of assays to a common reference | Confused with calibration | Program-specific |
| Calibration | Adjustment of an assay against standards | Confused with standardization | Assay-specific |
| Point-of-care testing (POCT) | Testing near the patient | Assumed equivalent to central lab | Device-specific |
| Chain of custody | Documented specimen handling | Omitted in research extracts | Metadata |
| Aliquot | A portion of a specimen | Confused with the primary tube | None |
| Additive | Substance in a tube (EDTA, citrate, heparin) | Ignored in result interpretation | Tube type |
| Order of draw | Sequence of tubes to avoid contamination | Violated in difficult draws | CLSI GP41 |
| Carryover | Contamination from a previous sample | Confused with interference | Analyte units |
| Matrix effect | Specimen-specific interference | Confused with analytical error | Assay-specific |
| Standard deviation interval | Spread used in reference interval estimation | Confused with reference range | Analyte units |

## 5. The mechanics, step by step

### Pre-analytical phase

**Step 1. Confirm the order and the indication.**

- Input: clinician order, test code, priority.
- Operation: verify the test matches the clinical question; check for duplicate orders.
- Output: an order with an indication.
- Reasoning: the indication determines the pretest probability and therefore the interpretation.
  A D-dimer ordered to "rule out" pulmonary embolism in a low-probability patient has a different
  meaning from one ordered in a patient with active cancer.
- Branch point: duplicate orders within a short window produce repeated results that are not
  independent observations.
- Analytic artifact: `order_id`, `test_code`, `indication`, `priority`, `ordered_ts`.

**Step 2. Prepare the patient.**

- Input: patient state.
- Operation: fasting status, timing relative to medications, posture, tourniquet time.
- Output: a documented collection context.
- Reasoning: glucose and triglycerides depend on fasting. Potassium and protein depend on
  posture and tourniquet time. Cortisol depends on time of day. Drug levels depend on dosing
  time.
- Branch point: if fasting status is unknown, the result is interpretable only with a caveat.
- Analytic artifact: `fasting_flag`, `collection_context`.

**Step 3. Select the tube and the order of draw.**

- Input: the test panel.
- Operation: choose tubes by additive and follow the order of draw.
- Output: a tube set with labels.
- Reasoning: EDTA binds calcium and is used for hematology; citrate binds calcium and is used for
  coagulation; heparin inhibits thrombin and is used for chemistry; serum tubes contain clot
  activators. Wrong tube produces wrong results: potassium EDTA contamination raises potassium
  and lowers calcium; citrate dilution lowers all coagulation factors.
- Analytic artifact: `container_type`, `additive`, `draw_order`.
- Branch point: difficult draws may require a smaller tube, which changes the additive-to-blood
  ratio and invalidates coagulation results.

**Step 4. Collect the specimen.**

- Input: vein, artery, or capillary.
- Operation: venipuncture, arterial puncture, or fingerstick.
- Output: a specimen with a collection time.
- Reasoning: hemolysis increases with small needles, vigorous aspiration, prolonged tourniquet
  time, and shaking. Tissue fluid contamination in a difficult draw dilutes the sample. Arterial
  and venous samples differ substantially for blood gases.
- Analytic artifact: `collected_ts`, `site`, `device`, `difficulty_flag`.

**Step 5. Label at the bedside.**

- Input: specimen and patient identity.
- Operation: label in the presence of the patient; verify two identifiers.
- Output: a labeled specimen.
- Reasoning: mislabeled specimens are a serious safety event and a source of wrong-patient
  results. Many laboratories reject unlabeled or mislabeled specimens outright.
- Branch point: a rejected specimen creates missing data that is not random; it is more common
  in emergency and high-workload settings.
- Analytic artifact: `label_status`, `rejection_reason`.

**Step 6. Transport and store.**

- Input: labeled specimen.
- Operation: transport at the required temperature within the required time.
- Output: a specimen received by the laboratory.
- Reasoning: delays raise potassium from cell leakage, lower glucose through glycolysis, and
  alter coagulation. Ammonia and lactate require immediate processing. Light exposure degrades
  bilirubin.
- Analytic artifact: `received_ts`, `transport_condition`, `processing_delay_min`.

**Step 7. Accession and assess specimen quality.**

- Input: received specimen.
- Operation: verify volume, check for clots, measure hemolysis, lipemia, and icterus indices.
- Output: an accepted or rejected specimen with quality indices.
- Reasoning: the hemolysis index is the laboratory's own estimate of red cell breakdown. It is
  measured spectrophotometrically and reported on a scale that varies by manufacturer. A result
  released despite a high index carries a comment, or is suppressed.
- Branch point: the same index value may trigger rejection for potassium and release with a
  comment for sodium.
- Analytic artifact: `hemolysis_index`, `lipemia_index`, `icterus_index`, `specimen_status`.

### Analytical phase

**Step 8. Prepare the analyzer and calibrate.**

- Input: reagents, calibrators, quality control material.
- Operation: calibrate, run quality control, verify acceptance criteria.
- Output: an analyzer ready for patient samples.
- Reasoning: calibration adjusts the assay response to known standards. Quality control verifies
  that the assay is within its expected performance on that day. A failed control stops the run.
- Analytic artifact: `calibration_id`, `qc_lot`, `qc_result`, `qc_pass`.

**Step 9. Measure the analyte.**

- Input: specimen and assay.
- Operation: immunoassay, spectrophotometry, chromatography, mass spectrometry, or
  electrochemistry.
- Output: a signal converted to a concentration.
- Reasoning: each method has its own interference profile. Immunoassays can be affected by
  heterophile antibodies, biotin, and high-dose hook effects. Spectrophotometry is affected by
  hemolysis, lipemia, and icterus. Mass spectrometry is less affected by immunoassay
  interference but has its own matrix effects.
- Branch point: a result above the analytical measuring range must be diluted and re-measured;
  the dilution factor must be applied and recorded.
- Analytic artifact: `method`, `analyzer_id`, `dilution_factor`, `raw_signal`.

**Step 10. Assess precision and bias.**

- Input: repeated measurements and reference materials.
- Operation: compute CV, bias, total error; compare to the allowable total error.
- Output: a performance statement.
- Reasoning: precision without bias control can be consistently wrong. A CV of 2% is excellent
  if the mean is correct and useless if the assay is biased by 20%.
- Analytic artifact: `cv_percent`, `bias_percent`, `total_error_percent`.

**Step 11. Handle interference and special cases.**

- Input: flagged specimen or implausible result.
- Operation: repeat, dilute, use an alternative method, or reject.
- Output: a verified result or a comment.
- Reasoning: hemolysis raises potassium, lactate dehydrogenase (LDH), and aspartate
  aminotransferase (AST), and lowers sodium and chloride through dilution. Lipemia raises
  turbidity and can falsely lower or raise analytes depending on the method. Icterus interferes
  with creatinine and bilirubin assays.
- Analytic artifact: `interference_flag`, `repeat_result`, `comment`.

### Post-analytical phase

**Step 12. Apply delta checks.**

- Input: current result and prior result.
- Operation: compare the change to an analyte-specific threshold over a defined interval.
- Output: pass, flag, or hold.
- Reasoning: delta checks catch specimen mix-ups and sudden changes. They also generate false
  positives in patients with genuine rapid change, such as dialysis or acute bleeding.
- Branch point: the threshold must account for the time interval. A 2 g/dL hemoglobin change in
  one hour is different from the same change in one year.
- Analytic artifact: `delta_flag`, `prior_value`, `delta_interval_h`.

**Step 13. Apply critical value rules.**

- Input: result.
- Operation: compare to the critical value list; notify the responsible clinician.
- Output: a documented notification.
- Reasoning: critical values are not the same as abnormal values. A potassium of 6.5 mmol/L
  requires a call; a potassium of 5.3 mmol/L usually does not. The Joint Commission requires
  accredited organizations to have a critical value reporting policy.
- Analytic artifact: `critical_flag`, `notification_ts`, `notified_role`.

**Step 14. Release and report.**

- Input: verified result.
- Operation: attach units, reference interval, flags, and comments; transmit to the EHR.
- Output: a reportable result.
- Reasoning: the reference interval attached to the result may be the laboratory's own or a
  vendor's. If your analytic table takes the value but not the interval, you lose the ability to
  judge abnormality consistently.
- Analytic artifact: `result_ts`, `value`, `unit`, `ref_low`, `ref_high`, `flag`, `comment`.

**Step 15. Audit and monitor.**

- Input: result stream.
- Operation: monitor turnaround time, rejection rate, hemolysis rate, critical value
  notification compliance, and QC failures.
- Output: quality metrics.
- Reasoning: laboratory quality is a process, not a single measurement. The metrics reveal where
  the process fails and whether a change improved it.
- Analytic artifact: a quality dashboard with definitions and denominators.

### Interpretation

**Step 16. Convert units before comparing.**

- Input: value and unit.
- Operation: convert to a consistent unit using a defined factor.
- Output: a normalized value with the original preserved.
- Reasoning: conversion factors are analyte-specific and not interchangeable. A glucose of 100
  mg/dL is 5.55 mmol/L; a creatinine of 1.0 mg/dL is 88.4 µmol/L; a bilirubin of 1.0 mg/dL is
  17.1 µmol/L.
- Analytic artifact: `value_si`, `unit_si`, `conversion_factor`.

**Step 17. Compare with the right reference interval.**

- Input: normalized value and patient population.
- Operation: select the interval for age, sex, pregnancy, and assay.
- Output: a flag with the interval source.
- Reasoning: adult intervals do not apply to children. Pregnancy alters many analytes:
  hemoglobin dilutes, creatinine falls, alkaline phosphatase rises, and thyroid-binding globulin
  rises. The laboratory's interval may be partitioned; if it is not, the partition is your
  responsibility.
- Analytic artifact: `reference_set_id`, `flag`, `interval_source`.

**Step 18. Interpret in clinical context.**

- Input: result, history, other results.
- Operation: combine with pretest probability and other findings.
- Output: an interpretation, not a diagnosis.
- Reasoning: a single abnormal value is a signal. A pattern is more informative: low ferritin
  with low MCV and high RDW suggests iron deficiency; low ferritin with high CRP suggests
  inflammation. The pattern, not the single value, drives the decision.
- Analytic artifact: a derived feature set (for example, transferrin saturation, Mentzer index).

**Step 19. Check internal consistency.**

- Input: a panel of related results.
- Operation: verify relationships that must hold (for example, anion gap, osmolar gap, Winter's
  formula, transferrin saturation).
- Output: a consistency flag.
- Reasoning: impossible combinations indicate unit errors, transcription errors, or specimen
  mix-ups. A pH of 7.40 with a PaCO2 of 20 mmHg and a bicarbonate of 30 mmol/L is not
  physiologically coherent; one of the values is wrong.
- Analytic artifact: `consistency_flag`, `rule_id`.

**Step 20. Record the interpretation and its limits.**

- Input: the result and the context.
- Operation: write a structured interpretation with assumptions.
- Output: a note or a feature dictionary entry.
- Reasoning: the interpretation is reusable only if its assumptions are explicit. "Ferritin low"
  is less useful than "ferritin 8 ng/mL, below the laboratory's 30 ng/mL threshold, consistent
  with iron deficiency in the absence of inflammation."
- Analytic artifact: a feature dictionary with definition, unit, source, and caveat.

### 5.1 Interference effects by analyte

| Analyte | Hemolysis | Lipemia | Icterus | Mechanism |
|---|---|---|---|---|
| Potassium | Marked increase | Minimal | Minimal | Intracellular potassium release |
| Lactate dehydrogenase | Marked increase | Minimal | Minimal | Intracellular enzyme release |
| Aspartate aminotransferase | Increase | Minimal | Minimal | Intracellular enzyme release |
| Sodium | Slight decrease | Minimal | Minimal | Dilution by red cell contents |
| Chloride | Slight decrease | Minimal | Minimal | Dilution by red cell contents |
| Glucose | Decrease if delayed | Minimal | Minimal | Glycolysis by cells |
| Bilirubin | Increase or decrease | Increase | Not applicable | Spectrophotometric interference |
| Creatinine | Minimal | Method-dependent | Increase | Non-specific chromogens |
| Troponin | Minimal | Minimal | Minimal | Immunoassay interference uncommon |
| Prostate-specific antigen | Minimal | Minimal | Minimal | Immunoassay interference uncommon |

The table is a starting point, not a universal rule. The magnitude depends on the assay, the
analyzer, and the degree of interference. The laboratory's own validation is the authority.

### 5.2 Specimen rejection criteria

| Criterion | Typical action | Rationale |
|---|---|---|
| Unlabeled or mislabeled | Reject | Patient safety |
| Insufficient volume | Reject or microtainer | Additive-to-blood ratio |
| Clotted specimen | Reject | Invalid result |
| Gross hemolysis | Reject for affected analytes | Interference |
| Wrong tube | Reject or recollect | Additive interference |
| Delayed transport beyond stability | Reject or comment | Analyte degradation |
| Improper temperature | Reject or comment | Analyte stability |
| Contamination from infusion line | Recollect | Dilution and drug contamination |

Rejection is a data-generating event, not a neutral absence. A rejected specimen creates a
missing value that is more likely in difficult draws, emergency settings, and specific units.

## 6. Worked example

A 34-year-old woman presents to her primary care clinic with six months of fatigue, exertional
dyspnea, and heavy menstrual bleeding. She is not pregnant. She takes no medications other than
an occasional ibuprofen. She looks pale. Her spleen is not palpable.

### 6.1 The results

| Test | Result | Unit | Reference interval (adult female, local laboratory) |
|---|---:|---|---|
| Hemoglobin | 9.8 | g/dL | 12.0-15.5 |
| Hematocrit | 31 | % | 36-46 |
| Mean corpuscular volume (MCV) | 74 | fL | 80-100 |
| Mean corpuscular hemoglobin (MCH) | 24 | pg | 27-33 |
| Red cell distribution width (RDW) | 16.5 | % | 11.5-14.5 |
| Red blood cell count (RBC) | 5.2 | x10^6/µL | 4.0-5.2 |
| White blood cell count (WBC) | 6.2 | x10^3/µL | 4.0-11.0 |
| Platelets | 450 | x10^3/µL | 150-400 |
| Serum iron | 30 | µg/dL | 60-170 |
| Total iron-binding capacity (TIBC) | 480 | µg/dL | 250-450 |
| Ferritin | 8 | ng/mL | 11-307 |
| Sodium | 138 | mmol/L | 135-145 |
| Potassium (initial) | 5.8 | mmol/L | 3.5-5.0 |
| Potassium (repeat) | 4.1 | mmol/L | 3.5-5.0 |
| Creatinine | 0.8 | mg/dL | 0.6-1.1 |
| Glucose | 92 | mg/dL | 70-99 fasting |

The initial potassium specimen was described by the laboratory as hemolyzed with a hemolysis
index of 3+ on the manufacturer's scale.

### 6.2 Step 1: classify the anemia

The hemoglobin is 9.8 g/dL. The World Health Organization (WHO) defines anemia in non-pregnant
women as hemoglobin below 12.0 g/dL. Severity bands are mild (11.0-11.9 g/dL), moderate
(8.0-10.9 g/dL), and severe (below 8.0 g/dL). She is in the moderate band.

The MCV of 74 fL is below the normal range of 80-100 fL, so the anemia is microcytic. The RDW
of 16.5% is above the normal range, indicating increased variation in red cell size. A
microcytic anemia with a high RDW suggests a nutritional deficiency or a mixed picture more
than a hereditary trait, which typically has a normal RDW.

### 6.3 Step 2: compute the iron indices

Transferrin saturation (TSAT):

```text
TSAT = serum iron / TIBC x 100
     = 30 / 480 x 100
     = 6.25%
```

A normal TSAT is approximately 20-50%. A value of 6.25% is markedly low and is consistent with
iron deficiency. In anemia of inflammation, TSAT is also low, but ferritin is usually normal or
high because ferritin is an acute-phase reactant. Her ferritin of 8 ng/mL is below the
laboratory's lower limit of 11 ng/mL and well below the commonly used diagnostic threshold of
30 ng/mL.

The Mentzer index:

```text
Mentzer index = MCV / RBC = 74 / 5.2 = 14.23
```

The Mentzer index was proposed as a discriminator between iron deficiency anemia and
thalassemia trait. A value below 13 suggests thalassemia trait; a value above 13 suggests iron
deficiency. The index is a heuristic with moderate accuracy, not a diagnostic test. In this
case, it points toward iron deficiency, consistent with the ferritin and the TSAT.

### 6.4 Step 3: the ferritin threshold in context

Ferritin is the most useful single test for iron deficiency in most outpatients, but its
interpretation depends on inflammation. Assume, for illustration, that a ferritin below 30 ng/mL
has 75% sensitivity and 95% specificity for iron deficiency in this population.

At a prevalence of 20% in a fatigue clinic:

- Diseased: 200 per 1,000.
- True positives: 200 x 0.75 = 150.
- False negatives: 50.
- Not diseased: 800.
- True negatives: 800 x 0.95 = 760.
- False positives: 40.
- PPV = 150 / 190 = 78.9%.
- NPV = 760 / 810 = 93.8%.

At a prevalence of 5% in a general screening population:

- Diseased: 50.
- True positives: 37.5.
- False negatives: 12.5.
- Not diseased: 950.
- True negatives: 902.5.
- False positives: 47.5.
- PPV = 37.5 / 85 = 44.1%.
- NPV = 902.5 / 915 = 98.6%.

The same ferritin result means different things in the two settings. This is why a laboratory
threshold is not a diagnosis.

### 6.5 Step 4: the hemolyzed potassium

The initial potassium of 5.8 mmol/L is above the reference interval. The specimen was hemolyzed
at 3+. Red cells contain potassium at a concentration far above plasma, so hemolysis releases
potassium into the serum or plasma and raises the measured value. The repeat, drawn without
hemolysis, is 4.1 mmol/L.

The difference is 1.7 mmol/L. That difference is large enough to change management: 5.8 mmol/L
may trigger an electrocardiogram, a repeat draw, and a review of medications; 4.1 mmol/L
requires none of that.

> **Note:** The exact increment from hemolysis depends on the degree of hemolysis, the analyte,
> and the assay. There is no universal correction factor. The correct response is to reject or
> repeat the specimen, not to subtract a fixed number.

If the analytic table stores only the maximum potassium per encounter, this patient's feature
value is 5.8 mmol/L, and any hyperkalemia rule will fire. If the table stores the result without
the hemolysis index, the reviewer cannot tell why.

### 6.6 Step 5: the iron deficit and the treatment plan

The Ganzoni formula estimates the total iron deficit for intravenous iron replacement:

```text
Total deficit (mg) = weight (kg) x (target Hb - actual Hb) (g/dL) x 2.4 + iron stores (mg)
```

Using a body weight of 60 kg, a target hemoglobin of 12.0 g/dL, an actual hemoglobin of 9.8
g/dL, and an iron store estimate of 500 mg:

```text
Deficit = 60 x (12.0 - 9.8) x 2.4 + 500
        = 60 x 2.2 x 2.4 + 500
        = 316.8 + 500
        = 816.8 mg
```

The result is approximately 817 mg of iron. The formula is an estimate used to plan replacement;
it assumes a linear relationship between hemoglobin and iron stores and does not account for
ongoing blood loss. In a patient with menorrhagia, ongoing loss must be treated or the deficit
will recur.

Oral iron is often first-line for stable outpatients. The calculation is more relevant to
intravenous iron planning. Either way, the arithmetic makes the reasoning auditable.

### 6.7 Step 6: the oxygen-carrying consequence

Hemoglobin is the dominant term in arterial oxygen content. At an SaO2 of 98% and a PaO2 of 90
mmHg:

```text
CaO2 at Hb 9.8 = 1.34 x 9.8 x 0.98 + 0.003 x 90
               = 12.87 + 0.27
               = 13.14 mL O2/dL

CaO2 at Hb 12.0 = 1.34 x 12.0 x 0.98 + 0.003 x 90
                = 15.76 + 0.27
                = 16.03 mL O2/dL
```

The difference is 2.89 mL O2/dL, or about 18% of the oxygen-carrying capacity at the target
hemoglobin. A young patient may tolerate this through increased cardiac output and increased
extraction. An older patient with coronary disease may not, because the heart's oxygen supply is
limited at the level of the coronary circulation.

### 6.8 Step 7: what the record should contain

| Element | Value in this case | Why it matters |
|---|---|---|
| Specimen type | Venous blood | Distinguishes from capillary or arterial |
| Container | Serum separator or EDTA depending on test | Additive affects analytes |
| Collection time | Recorded | Needed for delta checks and TAT |
| Hemolysis index | 3+ | Explains potassium |
| Assay | Local laboratory method | Reference interval depends on it |
| Reference interval | Attached to each result | Enables consistent flagging |
| Units | Conventional and SI | Prevents conversion errors |
| Prior result | None for potassium | Delta check unavailable |
| Clinical context | Heavy menses, fatigue | Pretest probability |

A pipeline that captures only the value and the timestamp has lost the information needed to
interpret the case. That is the central lesson of this topic.

## 7. Code, query, or calculation

The block below computes a reference interval from a synthetic sample, performs the unit
conversions, and reproduces the case arithmetic. The sample is synthetic and the interval is a
demonstration, not a CLSI-compliant establishment.

```python
"""Laboratory calculations: reference interval, unit conversion, iron indices.

The reference interval is computed from 120 synthetic creatinine values.
CLSI EP28-A3c recommends a minimum of 120 reference individuals per partition
for a nonparametric central 95% interval. This sample satisfies the count but
is not a real reference population.
"""

import random


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    k = (len(sorted_values) - 1) * q / 100.0
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = k - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


random.seed(7)
creatinine = sorted(random.gauss(mu=0.95, sigma=0.18) for _ in range(120))
ri_low = percentile(creatinine, 2.5)
ri_high = percentile(creatinine, 97.5)
print(f"n = {len(creatinine)}")
print(f"central 95% interval: {ri_low:.2f} to {ri_high:.2f} mg/dL")

# Unit conversions. Factors are analyte-specific.
conversions = {
    "glucose mg/dL -> mmol/L": (100.0, 1 / 18.0, "mmol/L"),
    "creatinine mg/dL -> umol/L": (1.0, 88.4, "umol/L"),
    "bilirubin mg/dL -> umol/L": (1.0, 17.1, "umol/L"),
    "cholesterol mg/dL -> mmol/L": (200.0, 0.02586, "mmol/L"),
    "triglycerides mg/dL -> mmol/L": (150.0, 0.01129, "mmol/L"),
    "calcium mg/dL -> mmol/L": (9.5, 0.2495, "mmol/L"),
}
for label, (value, factor, unit) in conversions.items():
    print(f"{label}: {value:g} -> {value * factor:.2f} {unit}")

# Iron indices
serum_iron = 30.0      # ug/dL
tibc = 480.0           # ug/dL
mcv = 74.0             # fL
rbc = 5.2              # x10^6/uL

tsat = serum_iron / tibc * 100
mentzer = mcv / rbc
print(f"transferrin saturation: {tsat:.2f}%")
print(f"Mentzer index: {mentzer:.2f}")

# Ganzoni iron deficit
weight_kg = 60.0
target_hb = 12.0
actual_hb = 9.8
deficit = weight_kg * (target_hb - actual_hb) * 2.4 + 500
print(f"estimated iron deficit: {deficit:.0f} mg")

# Ferritin PPV and NPV at two prevalences
def ferritin_metrics(prevalence: float, sensitivity: float = 0.75,
                     specificity: float = 0.95, n: int = 1000) -> tuple[float, float]:
    diseased = prevalence * n
    not_diseased = n - diseased
    tp = sensitivity * diseased
    fn = diseased - tp
    tn = specificity * not_diseased
    fp = not_diseased - tn
    return tp / (tp + fp), tn / (tn + fn)


for prev in (0.20, 0.05):
    ppv, npv = ferritin_metrics(prev)
    print(f"prevalence {prev:.2f}: PPV {ppv * 100:.1f}%, NPV {npv * 100:.1f}%")

# Oxygen content at two hemoglobins
def cao2(hb: float, sao2: float = 0.98, pao2: float = 90.0) -> float:
    return 1.34 * hb * sao2 + 0.003 * pao2


print(f"CaO2 at Hb 9.8: {cao2(9.8):.2f} mL/dL")
print(f"CaO2 at Hb 12.0: {cao2(12.0):.2f} mL/dL")
print(f"relative difference: {(cao2(12.0) - cao2(9.8)) / cao2(12.0) * 100:.1f}%")

assert 0.55 < ri_low < 0.65
assert 1.15 < ri_high < 1.30
assert abs(tsat - 6.25) < 0.01
assert abs(mentzer - 14.23) < 0.01
assert round(deficit) == 817
```

Expected output:

```text
n = 120
central 95% interval: 0.60 to 1.21 mg/dL
glucose mg/dL -> mmol/L: 100 -> 5.56 mmol/L
creatinine mg/dL -> umol/L: 1 -> 88.40 umol/L
bilirubin mg/dL -> umol/L: 1 -> 17.10 umol/L
cholesterol mg/dL -> mmol/L: 200 -> 5.17 mmol/L
triglycerides mg/dL -> mmol/L: 150 -> 1.69 mmol/L
calcium mg/dL -> mmol/L: 9.5 -> 2.37 mmol/L
transferrin saturation: 6.25%
Mentzer index: 14.23
estimated iron deficit: 817 mg
prevalence 0.20: PPV 78.9%, NPV 93.8%
prevalence 0.05: PPV 44.1%, NPV 98.6%
CaO2 at Hb 9.8: 13.14 mL/dL
CaO2 at Hb 12.0: 16.03 mL/dL
relative difference: 18.0%
```

The reference interval in the output, 0.60 to 1.21 mg/dL, is narrower than the local laboratory
interval of 0.6-1.1 mg/dL used in the case. That is expected: the synthetic sample has a
different distribution, and reference intervals are population-specific. The discrepancy is a
teaching point, not a bug.

A SQL delta-check query:

```sql
-- Flag potassium changes greater than 1.0 mmol/L within 24 hours.
WITH k AS (
    SELECT patient_id,
           result_ts,
           value AS potassium,
           LAG(value) OVER (PARTITION BY patient_id ORDER BY result_ts) AS prior_k,
           LAG(result_ts) OVER (PARTITION BY patient_id ORDER BY result_ts) AS prior_ts
    FROM results
    WHERE loinc_code = '2823-3'      -- potassium, serum or plasma
      AND value IS NOT NULL
)
SELECT patient_id,
       prior_ts,
       result_ts,
       prior_k,
       potassium,
       potassium - prior_k AS delta,
       EXTRACT(EPOCH FROM (result_ts - prior_ts)) / 3600.0 AS hours_between
FROM k
WHERE prior_k IS NOT NULL
  AND ABS(potassium - prior_k) > 1.0
  AND result_ts - prior_ts <= INTERVAL '24 hours'
ORDER BY ABS(potassium - prior_k) DESC;
```

The 1.0 mmol/L threshold is illustrative. A laboratory's delta-check policy is analyte-specific
and locally validated; it should not be copied from a textbook example.

## 8. Data, standards, and regulatory context

**Laboratory regulation in the United States.** The Clinical Laboratory Improvement Amendments of
1988 (CLIA) establish quality standards for laboratory testing and require certification.
CLIA is administered by the Centers for Medicare and Medicaid Services (CMS), with the Centers
for Disease Control and Prevention (CDC) providing scientific and technical support, and the
Food and Drug Administration (FDA) regulating test systems. Laboratories are surveyed by CMS or
by an approved accrediting organization such as the College of American Pathologists (CAP).

**Accreditation and quality standards.** CAP accreditation is widely used in the United States.
ISO 15189 is the international standard for medical laboratories, covering quality management
and competence. Both require documented procedures for specimen handling, validation,
quality control, and critical value reporting.

**CLSI documents.** The Clinical and Laboratory Standards Institute (CLSI) publishes consensus
documents used by laboratories and regulators. The ones most relevant to data work are:

| Document | Subject | What it constrains |
|---|---|---|
| EP28-A3c | Reference intervals | Minimum 120 reference individuals per partition; central 95% |
| EP05 | Precision | Replicate design and CV estimation |
| EP06 | Linearity | Verification of the measuring interval |
| EP07 | Interference | Hemolysis, lipemia, icterus testing |
| EP17 | Detection capability | Limit of detection and limit of quantification |
| GP41 | Specimen collection | Order of draw and tube selection |

These are consensus standards, not statutes. Their authority comes from accreditation and from
the laboratory's own quality system.

**Reference intervals.** CLSI EP28-A3c recommends a minimum of 120 reference individuals per
partition for a nonparametric central 95% interval. Laboratories may establish their own
intervals, transfer an interval from a validated source, or verify a manufacturer's interval with
a small sample. A transferred interval is only valid if the population and method are comparable.
This is why the same analyte can have different intervals at two hospitals.

**IVD regulation.** In vitro diagnostic (IVD) devices in the United States generally require
FDA clearance or approval before marketing, unless exempt. Tests intended for near-patient use
may require a CLIA waiver, which depends on the test's complexity and error risk. A laboratory
developed test has a separate regulatory pathway. These requirements shape which assays are
available and how their performance is documented, but they do not guarantee clinical utility.

**Terminology and units.** LOINC identifies the test; UCUM specifies the unit; SNOMED CT may be
used for specimen and finding concepts. The 2025 LOINC release (version 2.80) contains 104,672
terms. A result without a LOINC code and a UCUM unit is not safely poolable across sites.

**Critical values.** The Joint Commission's National Patient Safety Goal NPSG.02.03.01 requires
accredited organizations to have a process for reporting critical results. The specific list of
critical values is set by the laboratory and approved by the medical staff; it is not a national
list.

**Standardization programs.** Glycated hemoglobin (HbA1c) is standardized through the National
Glycohemoglobin Standardization Program (NGSP) and the International Federation of Clinical
Chemistry (IFCC). Creatinine is standardized against isotope-dilution mass spectrometry (ID-MS).
Lipids are standardized through the CDC Lipid Standardization Program. Standardization reduces
bias across laboratories but does not eliminate assay-specific differences in interference and
precision.

> **Note:** This section is educational and is not medical advice. Regulatory statements
> describe published requirements and standards; they do not establish that a particular
> laboratory, assay, or product is compliant.

## 9. Industry practice and case evidence

**Pre-analytical error is the dominant error category.** A 2024 study in *Clinical Chemistry*
analyzed approximately 11 million specimens and 40 million results at a large US hospital. Among
46,270 non-minimal-hemolysis errors, 97.0% were pre-analytical, 1.0% analytical, and 2.0%
post-analytical; errors occurred in about 0.8% of specimens. Hemolyzed but reportable specimens
were 47.2% of reported errors. These are single-center data, but the direction is consistent with
decades of laboratory quality literature. The inference for data teams is that specimen quality
flags carry real information and should not be discarded.

**Hemolysis in the emergency department.** A 2024 review reported in vitro hemolysis as the most
common pre-analytical problem, accounting for 40-70% of unsuitable blood samples, with emergency
department hemolysis rates reported up to 18.1% for arterial blood gas specimens and about 8.8%
for venous specimens. The variation across studies reflects differences in definition, device,
and population. The practical point is that hemolysis is common enough to require a systematic
flag, not an occasional manual review.

**Troponin assay heterogeneity.** High-sensitivity cardiac troponin assays differ in their 99th
percentile upper reference limits, their limits of detection, and their sex-specific thresholds.
The 2023 European Society of Cardiology acute coronary syndrome guideline and the 2025
ACC/AHA/ACEP/NAEMSP/SCAI guideline both emphasize assay-specific thresholds and serial sampling.
A dataset that pools troponin values across assays without a mapping is not measuring one
variable.

**Estimated glomerular filtration rate and race.** The National Kidney Foundation and American
Society of Nephrology task force recommended in 2021 that laboratories adopt the CKD-EPI 2021
creatinine equation without a race coefficient. The equation is:

```text
eGFR = 142 x min(SCr/kappa, 1)^alpha x max(SCr/kappa, 1)^-1.200 x 0.9938^age x 1.012 (if female)
```

where kappa is 0.7 for women and 0.9 for men, and alpha is -0.241 for women and -0.302 for men.
The change is a documented policy shift with a measurable effect on cohort composition. Any
longitudinal analysis that spans 2021 must account for the equation change; a step change in
eGFR values at the transition is an artifact, not a clinical improvement.

**HbA1c interference.** HbA1c results can be affected by hemoglobin variants, anemia, recent
transfusion, and conditions that alter red cell survival. The NGSP certifies methods, but
certification does not eliminate all interference. A dataset that uses HbA1c as a glycemic
feature should carry the method and a flag for conditions known to interfere.

**Ferritin and inflammation.** Ferritin rises in inflammation, liver disease, and malignancy. A
ferritin within the reference interval does not exclude iron deficiency in a patient with a high
C-reactive protein (CRP). The threshold of 30 ng/mL is commonly used in outpatients; higher
thresholds are used in inflammatory states. This is clinical guidance, not a universal rule, and
it varies by guideline.

**What is documented versus inferred.** The studies above document error rates, assay
differences, and policy changes. They do not establish that any specific pipeline is unsafe. The
inference is that these mechanisms are common enough to be designed against.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Hemolysis ignored | Potassium or LDH spikes without clinical correlate | Red cell breakdown releases intracellular analytes | False hyperkalemia, false liver injury | Carry hemolysis index; reject or repeat |
| Wrong tube | Potassium high with calcium low | EDTA contamination | Pseudohyperkalemia and pseudohypocalcemia | Tube type field; order-of-draw audit |
| Delayed processing | Glucose low, potassium high | Glycolysis and cell leakage | Misclassified diabetes and hyperkalemia | Collection and receipt timestamps |
| Assay pooling | Troponin distribution has two modes | Different assays and URLs | Site-specific thresholds and bias | Assay identifier; assay-specific rules |
| Reference interval mismatch | Abnormality rate differs by site | Different intervals or populations | Differential flagging | Store interval with result |
| Adult ranges for children | High flag rate in pediatrics | Age-specific ranges ignored | Alarm fatigue, missed disease | Age-partitioned intervals |
| Delta check absent | Specimen mix-ups undetected | No temporal comparison | Wrong-patient results | Analyte-specific delta rules |
| Critical value not communicated | Delayed treatment | Reporting process failure | Harm | Notification tracking |
| Unit conversion error | Values off by a constant factor | mg/dL versus mmol/L | Wrong thresholds and dosing | Unit column; conversion audit |
| Ferritin interpreted without inflammation | Normal ferritin in a sick patient | Acute-phase reactant | Missed iron deficiency | Add CRP; use higher threshold |
| HbA1c used in anemia | Discordance with glucose | Altered red cell survival | Misclassified glycemia | Flag interfering conditions |
| eGFR equation change ignored | Step change in 2021 | Race coefficient removed | Cohort discontinuity | Equation version field |
| Point-of-care and central lab pooled | Two distributions | Different devices and calibration | Confounded values | Device and method flags |
| Result used as diagnosis | Single value drives cohort | Measurement is not a diagnosis | Label noise | Require confirmatory evidence |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The specimen is representative of the patient | All inference depends on it | Hemolysis, dilution, wrong tube | Quality indices; repeat values |
| The assay is stable over the study period | Pooling across time requires it | Reagent lot change, recalibration, method change | Change-point analysis by lot and date |
| The reference interval matches the patient | Flagging depends on it | Age, sex, pregnancy, assay | Stratify flags by subgroup |
| The result timestamp is the availability time | Time-zero features depend on it | Interface delays, batch release | Compare result time with order and collect time |
| Sensitivity and specificity are stable | Test interpretation depends on them | Spectrum bias, assay change | Re-estimate in subgroups |
| Ferritin thresholds are population-appropriate | Iron deficiency classification | Inflammation, liver disease | Add CRP; sensitivity analysis |
| The delta-check threshold is valid | Error detection depends on it | Genuine rapid change | Review flagged cases |
| The laboratory's interval is partitioned | Pediatric and pregnancy comparisons | Unpartitioned intervals | Request partitioned intervals |
| LOINC mapping is correct | Cross-site pooling | Local codes mapped incorrectly | Manual mapping audit |
| Units are correctly converted | All thresholds | Factor errors | Round-trip conversion test |

## 12. Exercises

**Exercise 1 — Specimen journey (recall and application, 45 minutes).**
Map the journey of a serum potassium from order to result. Identify the tube, the handling
requirements, and four points where error can enter. Deliverable: a one-page flow diagram with
annotations. Acceptance criteria: tube type correct; at least four error points; each point has a
detection method. Time: 45 minutes.

**Exercise 2 — Reference interval reasoning (calculation, 40 minutes).**
A laboratory reports a reference interval of 0.6-1.1 mg/dL for creatinine in adult women. A
clinician asks why 5% of healthy women are outside the interval. Explain the construction of the
interval and compute the probability that a healthy woman has at least one abnormal result on a
12-test panel. Deliverable: a 300-word explanation with the calculation. Acceptance criteria:
correct probability; the explanation distinguishes reference interval from decision threshold.
Time: 40 minutes.

**Exercise 3 — Unit conversion drill (calculation, 30 minutes).**
Convert: glucose 126 mg/dL to mmol/L; creatinine 1.4 mg/dL to µmol/L; bilirubin 2.1 mg/dL to
µmol/L; cholesterol 240 mg/dL to mmol/L; triglycerides 400 mg/dL to mmol/L; calcium 8.8 mg/dL to
mmol/L. Deliverable: a table with the factor and the result. Acceptance criteria: six of six
correct to two decimal places; factors stated. Time: 30 minutes.

**Exercise 4 — Test characteristics (calculation and critique, 60 minutes).**
A ferritin threshold of 30 ng/mL has 75% sensitivity and 95% specificity in a fatigue clinic.
Compute PPV and NPV at prevalence 5%, 20%, and 40%. Explain why the same test is more useful in
the clinic than in a population screening program. Deliverable: a table and a 300-word
explanation. Acceptance criteria: arithmetic correct; the explanation names prevalence and
spectrum. Time: 60 minutes.

**Exercise 5 — Hemolysis audit (application, 60 minutes).**
Using a laboratory dataset with hemolysis indices, compute the rejection rate by collection site
and by hour of day. Identify the highest-risk site and propose one process change. Deliverable:
a table, a plot, and a one-paragraph recommendation. Acceptance criteria: denominators explicit;
the recommendation is testable; no causal claim beyond the data. Time: 60 minutes.

**Exercise 6 — Assay pooling critique (critique, 45 minutes).**
A dataset combines troponin values from three hospitals using two assays. Write a 400-word
critique of the practice and propose a fix. Deliverable: a critique and a revised data
specification. Acceptance criteria: identifies the threshold problem; proposes an assay
identifier and assay-specific rules; acknowledges the loss of sample size. Time: 45 minutes.

**Exercise 7 — Delta-check design (design, 60 minutes).**
Design delta-check rules for sodium, potassium, hemoglobin, and creatinine. Specify the
threshold, the time window, and the action. Deliverable: a rule table with four rows. Acceptance
criteria: thresholds are analyte-specific; time windows are explicit; each action is either
release, hold, or repeat. Time: 60 minutes.

**Exercise 8 — Feature dictionary (design, 75 minutes).**
Write a feature dictionary for ten laboratory features used in a sepsis model. Each entry must
include the LOINC code, unit, assay caveat, reference interval source, and missingness
mechanism. Deliverable: a table with ten rows. Acceptance criteria: all fields present; at least
three entries flag an interference or population caveat. Time: 75 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** A reference interval is typically the central 95% of values from a
reference population, estimated as the 2.5th and 97.5th percentiles. By construction, 2.5% of
healthy people fall below and 2.5% above. For 12 independent tests, the probability that all are
within their intervals is 0.95^12 = 0.5404, so the probability of at least one abnormal result is
1 - 0.5404 = 0.4596, or 46.0%. The tests are not truly independent, so the actual figure differs,
but the direction is correct.

**Answer to Exercise 3.** Glucose 126 / 18 = 7.00 mmol/L. Creatinine 1.4 x 88.4 = 123.76 µmol/L.
Bilirubin 2.1 x 17.1 = 35.91 µmol/L. Cholesterol 240 x 0.02586 = 6.21 mmol/L. Triglycerides 400
x 0.01129 = 4.52 mmol/L. Calcium 8.8 x 0.2495 = 2.20 mmol/L. Note that the Friedewald
LDL calculation is not valid at triglycerides above 400 mg/dL; the conversion is still
arithmetically correct, but the derived LDL would not be.

**Answer to Exercise 4.** At 5%: TP 37.5, FP 47.5, PPV 44.1%, NPV 98.6%. At 20%: TP 150, FP 40,
PPV 78.9%, NPV 93.8%. At 40%: TP 300, FP 30, PPV 90.9%, NPV 84.5%. The test is more useful in the
clinic because the pretest probability is higher, so a positive result is more likely to be a
true positive. In screening, most positives are false positives even with high specificity.

**Answer to self-check on hemolysis.** Hemolysis raises potassium, LDH, and AST, and can lower
sodium and chloride through dilution. It also interferes with spectrophotometric assays. The
correct response is to reject or repeat, not to apply a universal correction factor.

**Answer to self-check on reference intervals.** A reference interval is a population statement,
not a treatment threshold. It describes where 95% of a reference population falls. Clinical
decision thresholds may be different and are chosen based on harm and benefit.

**Answer to self-check on ferritin.** Ferritin is an acute-phase reactant. A normal ferritin does
not exclude iron deficiency in a patient with inflammation. A low ferritin is more specific. The
threshold should be interpreted with CRP and the clinical picture.

**Rubric for Exercise 7.** Full credit: four analyte-specific thresholds; explicit time windows;
actions that distinguish release, hold, and repeat; a rationale for each threshold. Half credit:
thresholds present but not time-specific. No credit: one threshold for all analytes.

**Rubric for Exercise 8.** Full credit: ten rows with LOINC, unit, assay caveat, reference
interval source, and missingness mechanism; at least three interference or population caveats.
Half credit: LOINC and unit present but caveats generic. No credit: values without provenance.

## 14. Interview questions

**1. Why do most laboratory errors occur before the assay?**
A strong answer describes specimen collection, handling, transport, and accessioning, and gives
examples. A weak answer says "human error" without a stage.

**2. What is a reference interval, and what does the central 95% mean?**
A strong answer explains the reference population and the 2.5th/97.5th percentiles, and notes
that 5% of healthy people fall outside. A weak answer calls it the normal range.

**3. How does hemolysis affect common lab results?**
A strong answer names potassium, LDH, AST, and dilution effects, and says to reject or repeat. A
weak answer says it "changes results" without analytes.

**4. Why is a troponin value not comparable across hospitals?**
A strong answer explains assay-specific 99th-percentile URLs and serial sampling. A weak answer
says "different machines."

**5. What is a delta check, and what is it for?**
A strong answer explains comparison with a prior result, analyte-specific thresholds, and uses
for specimen mix-ups and rapid change. A weak answer says it checks for abnormal values.

**6. What is the difference between analytical sensitivity and clinical sensitivity?**
A strong answer distinguishes the limit of detection from the proportion of diseased patients
detected. A weak answer treats them as the same.

**7. How would you handle unit differences in a multi-site dataset?**
A strong answer proposes a unit column, UCUM mapping, conversion factors, and round-trip tests.
A weak answer says "convert everything to SI" without a validation step.

**8. Why did eGFR equations change in 2021, and what does it mean for a longitudinal model?**
A strong answer explains the removal of the race coefficient and the step change in values. A
weak answer says "to be more equitable" without the data consequence.

**9. What is the difference between a critical value and an abnormal value?**
A strong answer explains that critical values require immediate notification and are set
locally. A weak answer says critical values are "very abnormal."

**10. How would you validate a lab-based phenotype?**
A strong answer proposes chart adjudication, assay mapping, reference interval review, and
subgroup analysis. A weak answer reports AUC only.

**11. When is a point-of-care result not equivalent to a central laboratory result?**
A strong answer names device differences, calibration, operator dependence, and unit/reporting
differences. A weak answer says point-of-care is "less accurate" without mechanism.

**12. What would make you distrust a ferritin value?**
A strong answer names inflammation, liver disease, and assay differences, and proposes CRP. A
weak answer says "it is variable."

## 15. Key terms

- **Analyte** — the substance being measured.
- **Analytical phase** — the measurement itself.
- **Autoverification** — automatic release of results meeting rules.
- **Bias** — systematic difference from the true value.
- **Calibration** — adjustment against known standards.
- **CLIA** — Clinical Laboratory Improvement Amendments.
- **CLSI** — Clinical and Laboratory Standards Institute.
- **Coefficient of variation** — standard deviation divided by the mean.
- **Critical value** — a result requiring immediate notification.
- **Delta check** — comparison with a prior result.
- **EDTA** — anticoagulant that binds calcium.
- **Ferritin** — iron storage protein and acute-phase reactant.
- **Hemolysis** — red cell breakdown in the specimen.
- **Hemolysis index** — spectrophotometric estimate of hemolysis.
- **Interference** — a substance that changes the measured value.
- **Icterus** — bilirubin-related interference.
- **IVD** — in vitro diagnostic device.
- **Limit of detection** — lowest concentration distinguishable from zero.
- **Limit of quantification** — lowest concentration with acceptable performance.
- **Linearity** — range over which response is proportional.
- **Lipemia** — lipid-related turbidity.
- **LOINC** — Logical Observation Identifiers Names and Codes.
- **Mentzer index** — MCV divided by RBC count.
- **Microcytic** — small red cells.
- **Point-of-care testing** — testing near the patient.
- **Post-analytical** — validation and reporting.
- **PPV** — positive predictive value.
- **Pre-analytical** — everything before the assay.
- **Precision** — reproducibility.
- **Reference interval** — central 95% of a reference population.
- **Sensitivity** — TP / (TP + FN).
- **Specificity** — TN / (TN + FP).
- **Transferrin saturation** — serum iron divided by TIBC.
- **Turnaround time** — elapsed time to result.
- **UCUM** — Unified Code for Units of Measure.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Standards and regulation**

- Clinical and Laboratory Standards Institute. EP28-A3c: Defining, Establishing, and Verifying
  Reference Intervals in the Clinical Laboratory; Approved Guideline, third edition, 2010.
  https://clsi.org/standards/products/method-evaluation/documents/ep28
- Clinical and Laboratory Standards Institute. GP41: Collection of Diagnostic Venous Blood
  Specimens. https://clsi.org/
- Centers for Medicare and Medicaid Services. Clinical Laboratory Improvement Amendments (CLIA).
  https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments
- College of American Pathologists. Laboratory Accreditation Program.
  https://www.cap.org/laboratory-improvement/accreditation/
- International Organization for Standardization. ISO 15189: Medical laboratories — Requirements
  for quality and competence.
- Logical Observation Identifiers Names and Codes. Version 2.80 release notes, February 2025.
  https://loinc.org/kb/release-notes-archive/2.80
- The Joint Commission. National Patient Safety Goal NPSG.02.03.01: Reporting critical results.
  https://www.jointcommission.org/

**Evidence**

- Lin Y, Spies N, Zaydman M, Farnsworth C. Preanalytical phase errors constitute the vast
  majority of errors in clinical laboratory testing. *Clinical Chemistry*, 2024.
  https://doi.org/10.1093/clinchem/hvae106.207
- Skala K. Best practices in preanalytical sample quality: managing and preventing in vitro
  hemolysis in the emergency department. *MLO Online*, June 2024.
  https://www.mlo-online.com/diagnostics/hematology/article/55056397/
- Delgado C, Baweja M, Crews DC, et al. A unifying approach for GFR estimation: recommendations
  of the NKF-ASN Task Force on Reassessing the Inclusion of Race in Diagnosing Kidney Disease.
  *Journal of the American Society of Nephrology*, 2021. https://doi.org/10.1681/ASN.2021070988
- Inker LA, Eneanya ND, Coresh J, et al. New creatinine- and cystatin C-based equations to
  estimate GFR without race. *New England Journal of Medicine*, 2021.
  https://doi.org/10.1056/NEJMoa2102953
- National Kidney Foundation. Recommendations for implementing the CKD-EPI 2021 race-free eGFR
  calculation: guidelines for clinical laboratories.
  https://www.kidney.org/recommendations-implementing-ckd-epi-2021-race-free-egfr-calculation-guidelines-clinical
- Byrne RA, Rossello X, Coughlan JJ, et al. 2023 ESC Guidelines for the management of acute
  coronary syndromes. *European Heart Journal*, 2023. https://doi.org/10.1093/eurheartj/ehad191

**Background reading**

- Burtis CA, Bruns DE, et al. *Tietz Fundamentals of Clinical Chemistry and Molecular
  Diagnostics*.
- Bickley LS. *Bates' Guide to Physical Examination and History Taking*, 13th edition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_anatomy-physiology-and-vital-signs.md`](02_anatomy-physiology-and-vital-signs.md)
- Next topic: [`04_pharmacology-basics.md`](04_pharmacology-basics.md)
- Related topic: [`09_special-populations.md`](09_special-populations.md)
- Related topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Survey file: [`../../01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Unit conversion lab: [`../../labs/01A_unit_conversion.ipynb`](../../labs/01A_unit_conversion.ipynb)

