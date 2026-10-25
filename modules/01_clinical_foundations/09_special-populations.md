# Special Populations

> Module 01 — Clinical Foundations for Data Scientists
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: 02, 03, 04
> Version 2026.3

---

## 1. Why this topic exists

Most clinical evidence is generated in adults aged 18 to 65 who do not have severe comorbidity.
Most patients are not that. Children, pregnant people, older adults, patients with kidney or
liver failure, and patients dependent on devices are systematically underrepresented in trials
and systematically overrepresented in care. A model trained on the average patient and deployed
on the full population will fail at the edges, and the edges are where the harm is largest.

Four problems recur.

**Problem 1: reference ranges do not transfer.** A creatinine of 1.2 mg/dL is normal in a
muscular young man and abnormal in a frail 85-year-old woman. A heart rate of 110 beats/min is
tachycardia in a resting adult and normal in a 2-year-old. A hemoglobin of 11 g/dL is anemia in a
non-pregnant woman and normal in pregnancy. Applying adult, non-pregnant ranges produces
systematic error correlated with age, sex, and pregnancy status.

**Problem 2: dosing does not transfer.** Drug clearance depends on renal and hepatic function,
body composition, and enzyme activity. A standard adult dose can be toxic in a 4 kg neonate or a
95-year-old with a creatinine clearance of 25 mL/min. A dose that is therapeutic in a
non-pregnant adult may be subtherapeutic in pregnancy because clearance increases.

**Problem 3: presentation differs.** Older adults often present with delirium, falls, or
functional decline rather than the classic symptoms of a disease. Children compensate for
hypovolemia until they decompensate abruptly. Pregnant patients have physiological changes that
mimic disease (lower blood pressure, higher heart rate, dilutional anemia) and disease that is
unique to pregnancy (preeclampsia, peripartum cardiomyopathy).

**Problem 4: the data are sparse and the labels are noisy.** Special populations have fewer
records, more missing fields, and more heterogeneous documentation. A model evaluated on the
overall population may have no meaningful performance estimate for a subgroup of 200 patients.

This topic covers the physiology, ranges, dosing, and data failure modes for each special
population, with explicit attention to what a model can and cannot learn from small subgroups.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the physiological changes of pregnancy that alter vital signs, laboratory values,
   and drug clearance.
2. **Apply** age-specific pediatric vital sign ranges and weight-based dosing rules.
3. **Explain** how aging changes renal function, body composition, and drug response.
4. **Compute** creatinine clearance and classify chronic kidney disease stage.
5. **Adjust** drug dosing for renal and hepatic impairment using labeled criteria.
6. **Identify** device-generated data artifacts and their effect on features.
7. **Critique** a model's subgroup performance and **propose** a safer evaluation plan.

## 3. The core mental model

A special population is defined by a physiological difference that changes the relationship
between a measurement and the underlying state, or between a dose and the concentration at the
target.

```text
   STANDARD ADULT
   measurement <-> physiology <-> dose <-> concentration <-> effect

   SPECIAL POPULATION
   measurement <-> ALTERED physiology <-> ALTERED dose <-> ALTERED concentration <-> effect
```

The alterations are not random noise. They are systematic and often predictable:

| Population | Main alteration | Consequence |
|---|---|---|
| Neonate | Immature renal and hepatic clearance; high total body water | Higher Vd for hydrophilic drugs; longer half-life |
| Child | Higher metabolic rate; changing body composition | Weight-based dosing; age-specific vitals |
| Pregnant | Increased plasma volume, cardiac output, GFR; decreased albumin | Dilutional anemia; lower drug concentrations; altered binding |
| Older adult | Reduced GFR, muscle mass, and hepatic flow; polypharmacy | Higher drug concentrations; interactions; atypical presentation |
| Renal impairment | Reduced clearance of renally eliminated drugs | Accumulation and toxicity |
| Hepatic impairment | Reduced metabolism and protein synthesis | Accumulation; coagulopathy; altered binding |
| Device-dependent | Mechanical or electrical support | Artifacts; altered physiology; device-specific data |

### 3.1 The pregnancy physiology map

```text
   Plasma volume      +40-50%
   Cardiac output     +30-50%
   Heart rate         +10-20 beats/min
   Blood pressure     -5 to -10 mmHg (nadir second trimester)
   GFR                +50%
   Creatinine         0.4-0.8 mg/dL
   Hemoglobin         dilutional, about 11-12 g/dL
   Tidal volume       +30-40%
   PaCO2              28-32 mmHg
   Bicarbonate        18-22 mmol/L
   White blood cells  up to 16,000/uL in labor
```

The acid-base picture is a compensated respiratory alkalosis. A PaCO2 of 30 mmHg with a
bicarbonate of 20 mmol/L and a pH of 7.44 is normal in pregnancy, not a primary disorder. A
clinician who applies non-pregnant ranges will call it abnormal.

### 3.2 The pediatric physiology map

```text
   Total body water   higher (75-80% in neonates vs about 60% in adults)
   GFR                about 30-40% of adult at birth, matures by 6-12 months
   Hepatic enzymes    immature at birth
   Protein binding    lower, so more free drug
   Metabolic rate     higher per kg
   Compensatory reserve lower in infants
```

The clinical consequence is that infants dehydrate faster, overdose faster, and compensate less
reserve. The data consequence is that adult ranges and adult doses are wrong in both directions.

### 3.3 The older adult physiology map

```text
   GFR                declines with age, even when creatinine is normal
   Muscle mass        lower, so creatinine is a poor marker of GFR
   Total body water   lower, so hydrophilic drugs have a smaller Vd
   Body fat           higher, so lipophilic drugs have a larger Vd
   Hepatic flow       lower
   Polypharmacy       common
   Homeostasis        slower
```

A "normal" creatinine in an 85-year-old woman may represent a creatinine clearance of 40 mL/min.
That is why Cockcroft-Gault uses age, weight, and sex, and why eGFR equations include age and
sex.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Neonate | Infant in the first 28 days of life | Confused with infant | Age band |
| Infant | Child from 1 to 12 months | Confused with neonate | Age band |
| Child | Age 1 to 12 years in many guidelines | Defined differently by guideline | Age band |
| Adolescent | Age 12 to 18 years | Confused with adult | Age band |
| Gestational age | Age from the last menstrual period | Confused with postnatal age | Weeks |
| Postmenstrual age | Gestational age plus postnatal age | Confused with gestational age | Weeks |
| Pediatric dose | Weight- or BSA-based dose | Confused with adult dose | mg/kg or mg/m² |
| Allometric scaling | Dose scaling by metabolic size | Confused with linear mg/kg | Exponent-based |
| Total body water | Fraction of body mass that is water | Assumed constant across ages | Proportion |
| Glomerular filtration rate (GFR) | Volume filtered per time | Confused with creatinine clearance | mL/min or mL/min/1.73 m² |
| Creatinine clearance (CrCl) | Estimated clearance of creatinine | Confused with eGFR | mL/min |
| CKD | Chronic kidney disease | Confused with acute kidney injury | KDIGO stage |
| AKI | Acute kidney injury | Confused with CKD | KDIGO stage |
| CKD stage G1 | eGFR 90 or higher | Treated as normal without albuminuria | mL/min/1.73 m² |
| CKD stage G2 | eGFR 60-89 | Treated as normal | mL/min/1.73 m² |
| CKD stage G3a | eGFR 45-59 | Confused with G3b | mL/min/1.73 m² |
| CKD stage G3b | eGFR 30-44 | Confused with G3a | mL/min/1.73 m² |
| CKD stage G4 | eGFR 15-29 | Confused with G5 | mL/min/1.73 m² |
| CKD stage G5 | eGFR below 15 | Confused with dialysis dependence | mL/min/1.73 m² |
| Child-Pugh | Cirrhosis severity score | Confused with MELD | Class A/B/C |
| MELD | Model for End-Stage Liver Disease | Confused with Child-Pugh | Score |
| Frailty | Multisystem vulnerability | Equated with age | Phenotype or index |
| Polypharmacy | Multiple concurrent medications | Defined differently (often 5 or more) | Count |
| Beers Criteria | Potentially inappropriate medications in older adults | Treated as an absolute prohibition | AGS list |
| Deprescribing | Planned reduction or discontinuation | Confused with non-adherence | Process |
| Preeclampsia | Pregnancy-specific hypertensive disorder | Confused with chronic hypertension | Clinical diagnosis |
| Teratogen | Agent that causes fetal malformation | Confused with any fetal risk | Drug property |
| PLLR | Pregnancy and Lactation Labeling Rule | Confused with the old letter categories | FDA rule |
| PEWS | Pediatric Early Warning Score | Confused with NEWS2 | Score |
| Apnea of prematurity | Pauses in breathing in preterm infants | Confused with periodic breathing | Clinical diagnosis |
| Ductal-dependent lesion | Congenital heart lesion dependent on the ductus arteriosus | Missed after ductal closure | Clinical category |
| LVAD | Left ventricular assist device | Confused with a pacemaker | Device |
| ICD | Implantable cardioverter-defibrillator | Confused with a pacemaker | Device |
| Pacemaker | Device for bradycardia pacing | Confused with ICD | Device |
| Driveline | Cable from an implanted device to an external controller | Ignored in infection data | Device component |
| eGFR | Estimated glomerular filtration rate | Used for drug dosing without adjustment | mL/min/1.73 m² |
| Adjusted body weight | Weight adjusted for obesity in dosing | Confused with ideal body weight | kg |
| Ideal body weight | Weight based on height and sex | Confused with actual weight | kg |
| Lean body weight | Weight excluding fat mass | Confused with ideal body weight | kg |
| BSA | Body surface area | Method often unstated | m² |

## 5. The mechanics, step by step

**Step 1. Identify the population and its defining physiology.**

- Input: age, sex, pregnancy status, renal function, hepatic function, devices.
- Operation: assign the patient to one or more special populations.
- Output: a population flag set.
- Reasoning: a patient can belong to several populations at once — a 78-year-old with CKD and a
  pacemaker is older, renally impaired, and device-dependent. Each alters a different part of the
  measurement-to-effect chain.
- Branch point: age alone is not a sufficient definition. A fit 80-year-old and a frail
  80-year-old have different physiology.
- Analytic artifact: `population_flags`, `age_years`, `pregnancy_status`, `ckd_stage`.

**Step 2. Select the correct reference range.**

- Input: population flags and laboratory values.
- Operation: choose age-, sex-, pregnancy-, and assay-specific intervals.
- Output: a range with a source.
- Reasoning: adult ranges applied to children will flag normal children and miss abnormal ones.
  Non-pregnant ranges applied to pregnant patients will misclassify dilutional anemia and normal
  lower blood pressure.
- Branch point: if a partitioned range is unavailable, state the limitation rather than silently
  using the adult range.
- Analytic artifact: `reference_set_id`, `range_source`.

**Step 3. Adjust for body size.**

- Input: weight, height, age, sex.
- Operation: compute BMI, BSA, ideal body weight, adjusted body weight, and lean body weight as
  appropriate.
- Output: size metrics with the method.
- Reasoning: mg/kg dosing assumes that weight tracks the relevant volume. That assumption fails
  in obesity, edema, amputation, and pregnancy. For some drugs, adjusted body weight is more
  appropriate; for others, actual body weight is.
- Branch point: the choice of weight can change the dose by 30% or more.
- Analytic artifact: `weight_kg`, `weight_type`, `bmi`, `bsa_m2`.

**Step 4. Assess renal function with the right equation.**

- Input: age, sex, weight, serum creatinine.
- Operation: compute Cockcroft-Gault CrCl for drug dosing; compute eGFR for CKD staging.
- Output: both values, with the equation named.
- Reasoning: Cockcroft-Gault is in most drug labels and is not normalized to BSA. eGFR is
  normalized to 1.73 m² and is used for staging. They can differ by 10-20% or more, especially at
  extremes of body size.
- Branch point: in acute kidney injury, the creatinine is not at steady state, so both equations
  are unreliable.
- Analytic artifact: `crcl_ml_min`, `egfr_ml_min_1_73`, `equation_version`.

**Step 5. Classify chronic kidney disease.**

- Input: eGFR and albuminuria.
- Operation: assign a KDIGO category.
- Output: a stage with an albuminuria category.
- Reasoning: CKD is defined by either decreased GFR or evidence of kidney damage for at least
  three months. A single low eGFR is not CKD; it may be acute.
- Branch point: CKD staging requires chronicity, which administrative data often cannot
  establish.
- Analytic artifact: `ckd_stage`, `albuminuria_category`, `chronicity_flag`.

**Step 6. Assess hepatic function.**

- Input: bilirubin, albumin, INR, ascites, encephalopathy.
- Operation: compute Child-Pugh class or MELD.
- Output: a severity category.
- Reasoning: hepatic impairment affects metabolism, protein binding, and coagulation. Child-Pugh
  and MELD predict prognosis and are used for some dose adjustments, but they do not measure
  specific enzyme activity.
- Branch point: a patient with normal transaminases can still have impaired clearance.
- Analytic artifact: `child_pugh_class`, `meld_score`.

**Step 7. Adjust the dose.**

- Input: drug label, renal and hepatic function, weight, age.
- Operation: apply the labeled adjustment; if none exists, consult a pharmacist.
- Output: a dose with a rationale.
- Reasoning: dose adjustment is drug-specific. Some drugs require a reduced dose; some require a
  longer interval; some are contraindicated. A single "renal dose" rule is wrong.
- Branch point: dialysis removes some drugs and not others; the timing relative to dialysis
  matters.
- Analytic artifact: `dose_adjusted`, `adjustment_reason`, `label_source`.

**Step 8. Apply pediatric dosing rules.**

- Input: age, weight, indication.
- Operation: use mg/kg or BSA; apply age-specific maximums; check for neonatal immaturity.
- Output: a dose with a frequency.
- Reasoning: neonates and infants have immature clearance and different body composition. A dose
  that is appropriate for a 10-year-old may be toxic for a 1-month-old at the same mg/kg.
- Branch point: some drugs are dosed by postmenstrual age, not chronological age.
- Analytic artifact: `dose_mg_per_kg`, `age_days`, `postmenstrual_age_weeks`.

**Step 9. Compute pediatric maintenance fluids.**

- Input: weight.
- Operation: apply the 4-2-1 rule: 4 mL/kg/h for the first 10 kg, 2 mL/kg/h for the next 10 kg,
  1 mL/kg/h for each kilogram above 20.
- Output: mL/h.
- Reasoning: the rule approximates hourly maintenance needs. It does not account for deficits,
  ongoing losses, or cardiac or renal disease.
- Branch point: in children with heart failure or kidney failure, maintenance may need to be
  reduced.
- Analytic artifact: `maintenance_ml_h`, `weight_kg`.

**Step 10. Compute pediatric fluid deficit.**

- Input: weight and estimated dehydration percentage.
- Operation: deficit = dehydration fraction x weight x 1,000.
- Output: mL.
- Reasoning: a 5% deficit is 50 mL/kg; a 10% deficit is 100 mL/kg. The estimate is clinical and
  imprecise. Overly rapid correction can cause cerebral edema.
- Branch point: the deficit is replaced over hours to days depending on severity and ongoing
  losses.
- Analytic artifact: `deficit_ml`, `dehydration_percent`.

**Step 11. Apply pregnancy-specific thresholds.**

- Input: pregnancy status and gestational age.
- Operation: use pregnancy ranges for blood pressure, hemoglobin, creatinine, and acid-base.
- Output: a pregnancy-adjusted interpretation.
- Reasoning: normal pregnancy includes lower blood pressure, dilutional anemia, lower creatinine,
  and compensated respiratory alkalosis. Applying non-pregnant thresholds produces false
  positives.
- Branch point: preeclampsia is defined by new hypertension after 20 weeks with proteinuria or
  organ dysfunction; the thresholds are 140/90 and 160/110 mmHg for severe range.
- Analytic artifact: `pregnancy_flag`, `gestational_age_weeks`, `pregnancy_range_source`.

**Step 12. Assess teratogenic risk.**

- Input: medication list and pregnancy status.
- Operation: identify known teratogens and counsel on alternatives.
- Output: a risk flag with a recommendation.
- Reasoning: some drugs are contraindicated in pregnancy (for example, warfarin, ACE inhibitors,
  angiotensin receptor blockers, isotretinoin, methotrexate, valproate). The risk depends on
  timing and dose.
- Branch point: the PLLR replaced the old letter categories in 2015; a legacy category is not a
  current risk assessment.
- Analytic artifact: `teratogen_flag`, `label_source`, `counseling_flag`.

**Step 13. Adjust for older adults.**

- Input: age, medications, frailty, renal function.
- Operation: apply Beers Criteria, assess anticholinergic burden, consider deprescribing.
- Output: a medication risk flag.
- Reasoning: older adults have reduced renal clearance, altered body composition, and more
  interactions. Beers is a consensus list of potentially inappropriate medications, not an
  absolute prohibition.
- Branch point: a Beers-listed drug may be appropriate for a specific patient after shared
  decision-making.
- Analytic artifact: `beers_flag`, `anticholinergic_burden`, `frailty_index`.

**Step 14. Handle device-dependent patients.**

- Input: device type and status.
- Operation: identify device-specific physiology and data artifacts.
- Output: a device flag with caveats.
- Reasoning: a patient with a left ventricular assist device may have no palpable pulse and a
  non-invasive blood pressure that is unreliable. A patient with a pacemaker may have an ECG
  that shows pacing spikes. A patient on dialysis has cyclical fluid and electrolyte shifts.
- Branch point: device interrogation data may be stored outside the EHR.
- Analytic artifact: `device_type`, `device_status`, `device_data_source`.

**Step 15. Adjust for obesity.**

- Input: weight, height, drug.
- Operation: choose actual, ideal, adjusted, or lean body weight according to the drug's
  pharmacology.
- Output: a dosing weight with a rationale.
- Reasoning: obesity increases adipose tissue and lean mass. Lipophilic drugs have a larger Vd;
  hydrophilic drugs have a smaller increase. Renal clearance may be higher because of increased
  lean mass.
- Branch point: BSA-based chemotherapy dosing may need a cap.
- Analytic artifact: `dosing_weight`, `weight_type`.

**Step 16. Evaluate subgroup performance.**

- Input: model predictions and population flags.
- Operation: compute performance by subgroup, with confidence intervals.
- Output: a stratified evaluation.
- Reasoning: an overall AUC can hide poor performance in a small subgroup. A subgroup with 50
  events cannot support a precise estimate.
- Branch point: if the subgroup is too small, report the uncertainty and consider pooling across
  sites.
- Analytic artifact: a subgroup performance table.

**Step 17. Assess representation.**

- Input: cohort composition.
- Operation: compare with the target population.
- Output: a representation report.
- Reasoning: a model trained on a cohort that excludes pregnant patients cannot be assumed to
  work in pregnancy. Exclusion is not neutrality.
- Branch point: the target population may be defined by the deployment setting, not the trial.
- Analytic artifact: `representation_table`, `exclusion_reasons`.

**Step 18. Document assumptions and limits.**

- Input: all adjustments.
- Operation: record the equations, thresholds, and sources.
- Output: a provenance record.
- Reasoning: special-population adjustments are assumptions. Without documentation, they cannot
  be audited.
- Analytic artifact: a methods appendix.

### 5.1 Population-specific ranges at a glance

| Variable | Adult (non-pregnant) | Child 1-5 years | Neonate | Pregnancy |
|---|---|---|---|---|
| Heart rate | 60-100 beats/min | 80-120 beats/min | 100-205 beats/min (awake) | 60-100 plus 10-20 |
| Respiratory rate | 12-20 breaths/min | 20-28 breaths/min | 30-53 breaths/min | 12-20, often slightly higher |
| Systolic blood pressure | Below 120 mmHg (normal) | 89-112 mmHg | 67-84 mmHg | 5-10 mmHg lower |
| Hemoglobin | 12.0-15.5 g/dL (women) | About 11-14 g/dL | About 14-24 g/dL at birth, falls | About 11-12 g/dL |
| Creatinine | 0.6-1.1 mg/dL (women) | Age-specific | Age-specific | 0.4-0.8 mg/dL |
| PaCO2 | 35-45 mmHg | 35-45 mmHg | 35-45 mmHg | 28-32 mmHg |
| Bicarbonate | 22-26 mmol/L | 22-26 mmol/L | 22-26 mmol/L | 18-22 mmol/L |

The pediatric and neonatal values are approximate teaching ranges, not decision thresholds. The
pregnancy values describe normal physiology, not disease. Every range must be paired with the
population and the source before it is used in a rule.

### 5.2 Dosing principles by population

| Population | Starting principle | Common error | Verification |
|---|---|---|---|
| Neonate | Dose by postmenstrual age and weight; longer intervals | Linear mg/kg from an older child | Neonatal formulary; pharmacist review |
| Infant | mg/kg with age-specific maximum | Adult dose scaled by weight | Weight-based reference |
| Child | mg/kg or BSA; cap at adult dose | Exceeding adult maximum | BSA and maximum check |
| Adolescent | Adult dose often appropriate; check weight | Assuming adult physiology | Age and pubertal status |
| Pregnancy | Often higher dose because clearance rises | Assuming non-pregnant clearance | Pregnancy-specific guidance |
| Older adult | Start low, titrate; adjust for renal function | Using creatinine as a normal marker | CrCl and frailty |
| Obesity | Choose actual, adjusted, or ideal weight by drug | Using actual weight for all drugs | Drug-specific guidance |
| Renal impairment | Reduce dose or extend interval | Reducing dose but not interval | Label and levels |
| Hepatic impairment | Child-Pugh-based adjustment | Assuming normal clearance | Child-Pugh and albumin |
| Dialysis | Supplemental dose after session | Ignoring dialysis clearance | Dialysis-specific guidance |

These principles are not a dosing manual. They are a checklist for the questions a data team
should ask before treating a dose field as comparable across patients.


### 5.3 Monitoring by population

| Population | Monitoring focus | Frequency | Data source |
|---|---|---|---|
| Neonate | Weight, bilirubin, glucose, temperature | Frequent in the first days | Bedside and laboratory |
| Infant | Hydration, feeding, weight | Each visit | Chart and scale |
| Child | Hydration, electrolytes, growth | Per admission | Laboratory and chart |
| Pregnant | Blood pressure, proteinuria, fetal growth | Each prenatal visit | Chart and imaging |
| Older adult | Renal function, falls, cognition, medications | Each admission | Chart and pharmacy |
| CKD | eGFR, albuminuria, potassium | Every 3-12 months | Laboratory |
| Dialysis | Pre- and post-dialysis weight, electrolytes | Each session | Dialysis record |
| Device-dependent | Device interrogation, battery, leads | Per device schedule | Device clinic |

Monitoring is a data-generating process. A population that is monitored less often will have
more missing data, and the missingness will be correlated with the population label.

## 6. Worked example

A 78-year-old woman with atrial fibrillation, hypertension, and chronic kidney disease presents
for a medication review. She weighs 54 kg and is 160 cm tall. Her serum creatinine is 1.4 mg/dL,
her hemoglobin is 11.2 g/dL, and her albumin is 3.8 g/dL. Her blood pressure is 132/78 mmHg and
her heart rate is 76 beats/min in sinus rhythm. She takes metoprolol, amlodipine, and
furosemide. She is not on an anticoagulant.

### 6.1 Step 1: stroke risk

CHA2DS2-VASc:

| Component | Points | This patient |
|---|---:|---:|
| Congestive heart failure | 1 | 0 |
| Hypertension | 1 | 1 |
| Age 75 or older | 2 | 2 |
| Diabetes mellitus | 1 | 0 |
| Stroke or transient ischemic attack | 2 | 0 |
| Vascular disease | 1 | 0 |
| Age 65-74 | 1 | 0 (already counted in the 75+ category) |
| Sex category female | 1 | 1 |
| **Total** |  | **4** |

A score of 4 is high risk. The 2023 ACC/AHA/ACCP/HRS atrial fibrillation guideline recommends
anticoagulation when the annual thromboembolic risk is at least 2% per year, which corresponds
to a score of 2 or more in men and 3 or more in women. Anticoagulation is recommended.

### 6.2 Step 2: renal function

Cockcroft-Gault, including the 0.85 factor for women:

```text
CrCl = ((140 - 78) x 54) / (72 x 1.4) x 0.85
     = (62 x 54) / 100.8 x 0.85
     = 3348 / 100.8 x 0.85
     = 33.21 x 0.85
     = 28.2 mL/min
```

CKD-EPI 2021 eGFR:

```text
eGFR = 142 x min(1.4/0.7, 1)^-0.241 x max(1.4/0.7, 1)^-1.200 x 0.9938^78 x 1.012
     = 142 x (2.0)^-1.200 x 0.6156 x 1.012
     = 38.5 mL/min/1.73 m²
```

The two values differ: 28.2 mL/min versus 38.5 mL/min/1.73 m². They are not interchangeable.
CrCl is used for drug dosing; eGFR is used for CKD staging. By eGFR, she has CKD stage G3b
(30-44 mL/min/1.73 m²). By CrCl, she is in the range often described as severe renal impairment.

> **Note:** The 0.85 factor matters. Omitting it gives 33.2 mL/min instead of 28.2 mL/min, a 17%
> difference. At a threshold of 30 mL/min, that error changes the category.

### 6.3 Step 3: the apixaban decision

The US prescribing information for apixaban reduces the dose to 2.5 mg twice daily when at least
two of the following three criteria are present:

| Criterion | Threshold | This patient | Met? |
|---|---|---:|---|
| Age | 80 years or older | 78 | No |
| Weight | 60 kg or less | 54 kg | Yes |
| Serum creatinine | 1.5 mg/dL or greater | 1.4 mg/dL | No |

Only one criterion is met, so the US label specifies the full dose of 5 mg twice daily, even
though the CrCl is 28.2 mL/min.

This is counterintuitive and worth pausing on. A clinician might reduce the dose because of the
low CrCl. The label's criteria are not a renal function rule; they are a composite developed in
the trials. The EU summary of product characteristics takes a different approach and recommends
2.5 mg twice daily for a CrCl of 15-29 mL/min. The same patient could receive different doses in
different jurisdictions.

The lesson for data science is not which dose is right. It is that dose is a function of the
label in the jurisdiction, the patient's criteria, and the clinical context. A feature that
derives "appropriate dose" from CrCl alone will misclassify patients whose label uses different
criteria.

### 6.4 Step 4: the creatinine rises

Six months later, her serum creatinine is 1.6 mg/dL. Recompute:

```text
CrCl = (62 x 54) / (72 x 1.6) x 0.85
     = 3348 / 115.2 x 0.85
     = 29.06 x 0.85
     = 24.7 mL/min

eGFR = 142 x (1.6/0.7)^-1.200 x 0.9938^78 x 1.012
     = 32.8 mL/min/1.73 m²
```

Now two criteria are met: weight 54 kg or less and serum creatinine 1.5 mg/dL or greater. The US
label reduces the dose to 2.5 mg twice daily. The EU label would also reduce it, based on the
CrCl of 24.7 mL/min.

The example shows why renal function must be a time-varying feature. A model that uses the
admission creatinine for a two-year period will miss the transition.

### 6.5 Step 5: what else changes with age

Her age affects more than the dose:

| Domain | Change | Consequence |
|---|---|---|
| Renal function | Lower GFR for the same creatinine | Drug accumulation |
| Body composition | Lower muscle mass, higher fat fraction | Smaller Vd for hydrophilic drugs |
| Hepatic flow | Reduced | Slower metabolism of high-extraction drugs |
| Homeostasis | Slower | Greater risk of orthostasis and falls |
| Polypharmacy | Multiple medications | Interactions and cumulative risk |
| Cognition | Higher risk of delirium | Atypical presentation of infection |

Her furosemide, metoprolol, and amlodipine all have fall-related risks. A model that predicts
falls from age alone will miss the medication contribution; a model that predicts from
medications alone will miss the age-related physiology. The interaction is the signal.

### 6.6 Step 6: a pediatric comparison

A 4-year-old boy weighing 16 kg presents with gastroenteritis and an estimated 5% dehydration.
He is alert, has dry mucous membranes, and has had no urine output for eight hours.

Maintenance fluids by the 4-2-1 rule:

```text
First 10 kg:  4 mL/kg/h x 10 kg = 40 mL/h
Next 6 kg:    2 mL/kg/h x 6 kg  = 12 mL/h
Total:        52 mL/h
```

Daily maintenance:

```text
52 mL/h x 24 h = 1,248 mL/day
```

Fluid deficit:

```text
Deficit = 5% x 16 kg x 1,000 mL/kg = 800 mL
```

Total fluid for the first 24 hours, ignoring ongoing losses:

```text
1,248 + 800 = 2,048 mL/day
```

An initial bolus of 20 mL/kg:

```text
20 x 16 = 320 mL
```

If the dehydration were 10%, the deficit would be 1,600 mL and the total would be 2,848 mL/day.
The arithmetic is simple; the physiology is not. Children have a higher surface-area-to-mass
ratio and a higher metabolic rate, so they dehydrate faster and tolerate deficits less well than
adults. The bolus is given as an isotonic crystalloid; hypotonic fluid is not used for
resuscitation.

### 6.7 Step 7: the documentation

| Item | Adult case | Pediatric case |
|---|---|---|
| Weight used | 54 kg actual | 16 kg actual |
| Renal function | CrCl 28.2 mL/min | Not applicable |
| Dose basis | Label criteria | mg/kg and 4-2-1 |
| Key risk | Bleeding, renal decline | Dehydration, hypoglycemia |
| Data caveat | CrCl versus eGFR | Weight-based dosing error |
| Monitoring | Hemoglobin, creatinine | Urine output, weight, electrolytes |

The two cases share a structure: a measurement, a population-specific interpretation, a
dose or fluid decision, and a monitoring plan. The content differs; the reasoning does not.

## 7. Code, query, or calculation

The block computes the renal values, the apixaban criteria, and the pediatric fluids.

```python
"""Special-population calculations: renal function, label criteria, pediatric fluids."""


def cockcroft_gault(age: int, weight_kg: float, scr_mg_dl: float,
                    female: bool = False) -> float:
    crcl = ((140 - age) * weight_kg) / (72 * scr_mg_dl)
    if female:
        crcl *= 0.85
    return crcl


def ckd_epi_2021(age: int, scr_mg_dl: float, female: bool = False) -> float:
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = scr_mg_dl / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** age
    return egfr * 1.012 if female else egfr


def apixaban_criteria(age: int, weight_kg: float, scr_mg_dl: float) -> tuple[int, str]:
    criteria = [age >= 80, weight_kg <= 60, scr_mg_dl >= 1.5]
    count = sum(criteria)
    dose = "2.5 mg twice daily" if count >= 2 else "5 mg twice daily"
    return count, dose


def pediatric_maintenance(weight_kg: float) -> float:
    """4-2-1 rule, mL/h."""
    if weight_kg <= 10:
        return 4 * weight_kg
    if weight_kg <= 20:
        return 40 + 2 * (weight_kg - 10)
    return 40 + 20 + 1 * (weight_kg - 20)


age, weight, height = 78, 54.0, 160.0
for scr in (1.4, 1.6):
    crcl = cockcroft_gault(age, weight, scr, female=True)
    egfr = ckd_epi_2021(age, scr, female=True)
    count, dose = apixaban_criteria(age, weight, scr)
    print(f"SCr {scr:.1f}: CrCl {crcl:.1f} mL/min; eGFR {egfr:.1f}; "
          f"criteria {count}; {dose}")

print()
peds_weight = 16.0
maint = pediatric_maintenance(peds_weight)
deficit_5 = 0.05 * peds_weight * 1000
deficit_10 = 0.10 * peds_weight * 1000
print(f"pediatric maintenance: {maint:.0f} mL/h ({maint * 24:.0f} mL/day)")
print(f"5% deficit: {deficit_5:.0f} mL; total day 1: {maint * 24 + deficit_5:.0f} mL")
print(f"10% deficit: {deficit_10:.0f} mL; total day 1: {maint * 24 + deficit_10:.0f} mL")
print(f"20 mL/kg bolus: {20 * peds_weight:.0f} mL")

assert round(cockcroft_gault(age, weight, 1.4, True), 1) == 28.2
assert round(ckd_epi_2021(age, 1.4, True), 1) == 38.5
assert apixaban_criteria(age, weight, 1.4)[0] == 1
assert apixaban_criteria(age, weight, 1.6)[0] == 2
assert pediatric_maintenance(16) == 52
```

Expected output:

```text
SCr 1.4: CrCl 28.2 mL/min; eGFR 38.5; criteria 1; 5 mg twice daily
SCr 1.6: CrCl 24.7 mL/min; eGFR 32.8; criteria 2; 2.5 mg twice daily

pediatric maintenance: 52 mL/h (1248 mL/day)
5% deficit: 800 mL; total day 1: 2048 mL
10% deficit: 1600 mL; total day 1: 2848 mL
20 mL/kg bolus: 320 mL
```

A SQL query for CKD staging with the equation version recorded:

```sql
-- Latest eGFR per patient in the past 12 months, with equation version.
SELECT patient_id,
       MAX(result_ts) AS latest_ts,
       MAX(CASE WHEN result_ts = latest THEN egfr END) AS latest_egfr,
       MAX(CASE WHEN result_ts = latest THEN equation_version END) AS equation
FROM (
    SELECT patient_id,
           result_ts,
           egfr,
           equation_version,
           MAX(result_ts) OVER (PARTITION BY patient_id) AS latest
    FROM egfr_results
    WHERE result_ts >= CURRENT_DATE - INTERVAL '12 months'
) t
GROUP BY patient_id;
```

The `equation_version` column matters because the CKD-EPI 2021 equation produces different values
from the 2009 equation. A cohort that spans the transition will have a step change that is an
artifact of the equation, not a change in kidney function.

## 8. Data, standards, and regulatory context

**Kidney disease.** The Kidney Disease: Improving Global Outcomes (KDIGO) 2024 clinical practice
guideline for the evaluation and management of chronic kidney disease defines CKD by decreased
GFR or markers of kidney damage for at least three months. Staging uses eGFR categories G1 to G5
and albuminuria categories A1 to A3. The guideline endorses the CKD-EPI 2021 race-free equation
for eGFR. Drug dosing still commonly uses Cockcroft-Gault, which is not the same equation.

**Pregnancy.** The American College of Obstetricians and Gynecologists (ACOG) defines
hypertension in pregnancy as 140/90 mmHg or higher and severe-range hypertension as 160/110 mmHg
or higher. Preeclampsia is a pregnancy-specific disorder defined by new hypertension after 20
weeks with proteinuria or organ dysfunction. These thresholds differ from non-pregnant adult
categories.

**Pregnancy labeling.** The FDA Pregnancy and Lactation Labeling Rule took effect on 30 June
2015. It removed the A, B, C, D, and X categories and replaced them with narrative subsections on
pregnancy, lactation, and females and males of reproductive potential. A dataset that still uses
the letter categories is using a retired scheme.

**Pediatrics.** The Pediatric Advanced Life Support (PALS) guidelines published by the American
Heart Association in 2020 provide age-specific vital sign ranges and resuscitation guidance.
Pediatric drug labeling is shaped by the Pediatric Research Equity Act of 2003 and the Best
Pharmaceuticals for Children Act, which require and incentivize pediatric studies for many
drugs. Many drugs remain off-label in children, and the evidence base is thinner than in adults.

**Older adults.** The American Geriatrics Society Beers Criteria, updated in 2023, lists
potentially inappropriate medications for adults aged 65 and older. The 2023 update moved
rivaroxaban to "avoid" for long-term treatment of atrial fibrillation and venous
thromboembolism, added warfarin to "avoid" as initial therapy unless a direct oral anticoagulant
is contraindicated, and moved aspirin to "avoid" for initiation for primary prevention. These
are consensus recommendations, not regulations.

**Devices.** Pacemakers, implantable cardioverter-defibrillators (ICDs), left ventricular assist
devices (LVADs), dialysis machines, insulin pumps, and ventilators generate data and alter
physiology. Device data may be stored in the EHR, in a vendor cloud, or only on the device.
Regulatory requirements for device data and cybersecurity are separate from clinical data
governance.

**Representation.** The FDA and other regulators have emphasized the inclusion of older adults,
pregnant people, and children in clinical trials where appropriate. Representation in trials
does not guarantee representation in a local dataset, and a local dataset may not represent the
deployment population.

> **Note:** This section is educational and is not medical advice. Dosing, thresholds, and
> monitoring must be verified against current labeling and institutional protocols.

## 9. Industry practice and case evidence

**Race-free eGFR changed cohort composition.** The NKF-ASN task force recommended in 2021 that
laboratories adopt the CKD-EPI 2021 creatinine equation without a race coefficient. The change
was implemented widely in the United States. A longitudinal analysis that spans the transition
will show a step change in eGFR for some patients, not because their kidney function changed but
because the equation did. The documented fact is the recommendation and the adoption. The
inference is that equation version must be a feature in any multi-year analysis.

**Beers Criteria are updated, not static.** The 2023 update changed several recommendations,
including anticoagulant and aspirin guidance. A model or quality metric built on the 2019 list
will be out of date. The criteria apply to adults aged 65 and older in ambulatory, acute, and
institutional settings, not to hospice or end-of-life care. Applying them without those
exclusions will generate false positives.

**Drug labels differ across jurisdictions.** The apixaban example is not unique. Renal dosing
thresholds, contraindications, and pregnancy warnings can differ between the US prescribing
information and the EU summary of product characteristics. A global dataset that applies one
label to all sites will misclassify some patients. The documented fact is the label difference;
the inference is that the label source and jurisdiction must be recorded.

**Special populations are underrepresented in trials.** Older adults, pregnant people, and
children have historically been excluded from many clinical trials, and the evidence base for
these groups is thinner. The COVID-19 pandemic illustrated the problem: pregnant people were
initially excluded from vaccine trials, and recommendations were made from observational data.
The documented fact is the exclusion; the inference is that models trained on trial-like
populations may not transport to the groups most likely to be excluded.

**Pediatric dosing errors are a known safety problem.** Weight-based dosing, unit conversion, and
decimal errors are recurring causes of pediatric medication errors. A data pipeline that parses
"mg/kg" without a current weight, or that converts pounds to kilograms incorrectly, can
reproduce those errors at scale. The documented problem is clinical; the data lesson is that
weight, unit, and age must be first-class fields.

**Device data are fragmented.** Device interrogation data for pacemakers, ICDs, and LVADs are
often stored in vendor systems outside the EHR. A patient may have a device that is not recorded
in the problem list, and device settings may not be available to the model. The documented fact
is the fragmentation; the inference is that device-dependent patients are systematically
under-described in EHR-based features.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Adult ranges for children | High flag rate in pediatrics | Age-specific thresholds ignored | Alarm fatigue, missed disease | Age-partitioned ranges |
| Non-pregnant ranges for pregnancy | Dilutional anemia flagged | Plasma volume expansion | Unnecessary treatment | Pregnancy-specific ranges |
| Cockcroft-Gault without sex factor | CrCl 17% too high in women | 0.85 factor omitted | Overdosing | Verify equation implementation |
| CrCl and eGFR conflated | Dose based on eGFR | Different quantities | Wrong dose | Record equation and use |
| Fixed dose in renal impairment | Drug accumulation | Clearance reduced | Toxicity | Label-based adjustment |
| Weight missing for mg/kg | Default weight used | Documentation gap | Wrong dose | Require measured weight |
| Pounds entered as kilograms | Dose 2.2x too high | Unit error | Overdose | Unit field; range check |
| Neonatal dose scaled linearly | Toxicity at mg/kg | Immature clearance | Harm | Postmenstrual age dosing |
| Beers list outdated | Recommendations conflict | Annual updates | Wrong risk flag | Version the criteria |
| Frailty equated with age | All older adults flagged | Heterogeneity ignored | False positives | Frailty index |
| Device not in problem list | Missing device flag | Fragmented data | Wrong physiology assumptions | Device registry linkage |
| LVAD blood pressure interpreted normally | Low BP with adequate perfusion | Continuous flow | Wrong treatment | Device-specific targets |
| Dialysis timing ignored | Drug level misinterpreted | Clearance varies by session | Wrong dose | Record dialysis schedule |
| Subgroup too small for evaluation | Wide confidence interval | Sparse data | Unreliable estimate | Report uncertainty; pool |
| Exclusion mistaken for safety | No pregnant patients in cohort | Trial exclusion | Unknown risk | State the limit |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Age is accurately recorded | All age-specific logic | Approximation, data entry | Compare with birth date |
| Weight is measured, not estimated | Dosing and BSA | Bed scale, historical weight | Weight source field |
| Creatinine is at steady state | CrCl and eGFR | AKI, dialysis | Trend creatinine |
| Pregnancy status is complete | Pregnancy ranges | Undocumented pregnancy | Lab and diagnosis linkage |
| The equation matches the label | Dose adjustment | Different equations | Label review |
| The Beers version is current | Medication risk | Annual updates | Version field |
| Device status is known | Physiology and artifacts | Fragmented data | Device registry |
| Subgroup performance is estimable | Fairness evaluation | Small n | Confidence intervals |
| The cohort represents the deployment population | Transportability | Exclusion of special groups | Representation table |
| The label applies in the jurisdiction | Dosing | Regional differences | Jurisdiction field |

## 12. Exercises

**Exercise 1 — Pregnancy physiology (recall and application, 45 minutes).**
List five physiological changes in pregnancy that alter common laboratory values, and state the
direction and approximate magnitude of each. Deliverable: a five-row table. Acceptance criteria:
each change has a direction and a range; the acid-base picture is included. Time: 45 minutes.

**Exercise 2 — Pediatric fluids (calculation, 40 minutes).**
A 2-year-old weighs 12 kg and is 5% dehydrated. Compute maintenance fluids by the 4-2-1 rule, the
deficit, the total for day 1, and a 20 mL/kg bolus. Deliverable: a calculation sheet. Acceptance
criteria: units correct; the bolus is distinguished from maintenance; assumptions stated. Time:
40 minutes.

**Exercise 3 — Renal dosing (calculation, 50 minutes).**
A 70-year-old woman, 65 kg, SCr 1.3 mg/dL, needs a drug that requires a 50% dose reduction when
CrCl is below 30 mL/min. Compute CrCl and decide whether to reduce. Repeat with SCr 1.5.
Deliverable: calculations and a decision. Acceptance criteria: female factor applied; threshold
applied correctly; the decision is explicit. Time: 50 minutes.

**Exercise 4 — Apixaban criteria (application, 40 minutes).**
For three patients — a 82-year-old man, 75 kg, SCr 1.2; a 79-year-old woman, 58 kg, SCr 1.6; a
65-year-old man, 95 kg, SCr 1.0 — count the dose-reduction criteria and state the US label dose.
Deliverable: a table. Acceptance criteria: criteria counted correctly; the dose matches the
count; the limitation of the criteria is noted. Time: 40 minutes.

**Exercise 5 — Beers critique (critique, 45 minutes).**
A quality metric flags every older adult on a Beers-listed drug as inappropriate. Critique the
metric and propose a better one. Deliverable: a 400-word critique. Acceptance criteria:
distinguishes potentially inappropriate from absolutely contraindicated; addresses exclusions;
proposes a patient-specific review. Time: 45 minutes.

**Exercise 6 — Subgroup evaluation (application, 60 minutes).**
A model has an overall AUC of 0.84 but an AUC of 0.68 in patients aged 80 and older (n = 120, 18
events). Interpret the result and propose next steps. Deliverable: a 300-word analysis with a
confidence interval. Acceptance criteria: uncertainty quantified; at least two explanations;
next steps are concrete. Time: 60 minutes.

**Exercise 7 — Device data design (design, 60 minutes).**
Design a device registry linkage for patients with pacemakers, ICDs, and LVADs. Specify the
fields, the source, and the update frequency. Deliverable: a schema and a linkage plan.
Acceptance criteria: device type and status included; data source named; update frequency
defined. Time: 60 minutes.

**Exercise 8 — Representation audit (design, 75 minutes).**
Audit a model's training cohort against the deployment population by age, pregnancy status, and
CKD stage. Propose a mitigation for the largest gap. Deliverable: an audit table and a one-page
plan. Acceptance criteria: at least three dimensions; gap quantified; mitigation is feasible.
Time: 75 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** Maintenance: 4 x 10 + 2 x 2 = 44 mL/h; daily 1,056 mL. Deficit: 0.05 x
12 x 1,000 = 600 mL. Total day 1: 1,656 mL. Bolus: 20 x 12 = 240 mL. The bolus is given first
for resuscitation and is not subtracted from the maintenance calculation in this simplified
example; in practice, the bolus volume is part of the total fluid balance.

**Answer to Exercise 3.** At SCr 1.3: CrCl = ((140 - 70) x 65) / (72 x 1.3) x 0.85 = (70 x 65) /
93.6 x 0.85 = 4,550 / 93.6 x 0.85 = 48.6 x 0.85 = 41.3 mL/min. No reduction. At SCr 1.5: CrCl =
4,550 / 108 x 0.85 = 42.1 x 0.85 = 35.8 mL/min. Still above 30, so no reduction. A higher
creatinine or lower weight would cross the threshold.

**Answer to Exercise 4.** Patient 1: age 82 (yes), weight 75 (no), SCr 1.2 (no) = 1 criterion,
5 mg twice daily. Patient 2: age 79 (no), weight 58 (yes), SCr 1.6 (yes) = 2 criteria, 2.5 mg
twice daily. Patient 3: age 65 (no), weight 95 (no), SCr 1.0 (no) = 0 criteria, 5 mg twice
daily. The criteria are a composite, not a renal function rule.

**Answer to self-check on pregnancy.** Plasma volume rises 40-50%, cardiac output rises 30-50%,
heart rate rises 10-20 beats/min, blood pressure falls 5-10 mmHg, GFR rises about 50%, creatinine
falls to 0.4-0.8 mg/dL, hemoglobin dilutes to about 11-12 g/dL, tidal volume rises 30-40%, PaCO2
falls to 28-32 mmHg, and bicarbonate falls to 18-22 mmol/L.

**Answer to self-check on pediatric dosing.** Pediatric doses are weight- or BSA-based because
children have different body composition, metabolic rate, and clearance. A linear mg/kg
conversion from an adult dose is not sufficient; neonates and infants may need dosing by
postmenstrual age.

**Answer to self-check on Beers.** Beers lists potentially inappropriate medications, not absolute
contraindications. A drug may be appropriate for a specific patient after considering indications,
alternatives, and goals of care. The list applies to adults 65 and older, excluding hospice and
end-of-life care.

**Rubric for Exercise 6.** Full credit: confidence interval reported; at least two explanations
(sample size, case mix, true effect modification); concrete next steps (pool sites, collect more
data, recalibrate, restrict use). Half credit: interprets the AUC without uncertainty. No credit:
claims the model is unbiased because overall AUC is high.

**Rubric for Exercise 8.** Full credit: three dimensions; gap quantified; mitigation feasible;
monitoring plan. Half credit: audit without mitigation. No credit: assumes representation.

## 14. Interview questions

**1. Why are pregnancy reference ranges different?**
A strong answer names plasma volume expansion, vasodilation, increased GFR, and dilutional
anemia. A weak answer says "hormones."

**2. How do you dose a drug in renal impairment?**
A strong answer uses the label, the correct equation, and the indication, and notes that dose
and interval can both change. A weak answer says "reduce the dose."

**3. Why is Cockcroft-Gault still used for drug dosing?**
A strong answer explains that most labels were developed with it and that it is not normalized to
BSA. A weak answer says it is more accurate.

**4. What is the difference between CrCl and eGFR?**
A strong answer explains the units, the normalization, and the use cases. A weak answer treats
them as the same.

**5. How does aging change drug handling?**
A strong answer covers renal function, body composition, hepatic flow, and polypharmacy. A weak
answer says "metabolism slows."

**6. What is the Beers Criteria, and what are its limits?**
A strong answer describes the list, the population, and the exclusions. A weak answer says it
lists drugs older people should not take.

**7. Why are children underrepresented in drug labels?**
A strong answer explains the history, the Pediatric Research Equity Act, and the remaining
off-label use. A weak answer says trials are hard.

**8. How would you evaluate a model in a small subgroup?**
A strong answer reports confidence intervals, considers pooling, and avoids overinterpretation. A
weak answer reports a point estimate.

**9. What data problems occur with device-dependent patients?**
A strong answer names fragmented device data, artifacts, and altered physiology. A weak answer
says devices are noisy.

**10. How does pregnancy change drug clearance?**
A strong answer explains increased renal clearance, increased volume, and altered protein
binding, and notes that some drugs need higher doses. A weak answer says "the dose changes."

**11. What is the risk of applying adult ranges to children?**
A strong answer explains false positives and false negatives, and the age dependence. A weak
answer says children are different.

**12. How would you audit representation in a training cohort?**
A strong answer compares the cohort with the deployment population on age, pregnancy, renal
function, and comorbidity. A weak answer says "check demographics."

## 15. Key terms

- **Adjusted body weight** — weight adjusted for obesity.
- **AKI** — acute kidney injury.
- **Allometric scaling** — dose scaling by metabolic size.
- **Beers Criteria** — AGS list of potentially inappropriate medications.
- **BSA** — body surface area.
- **Child-Pugh** — cirrhosis severity score.
- **CKD** — chronic kidney disease.
- **Cockcroft-Gault** — creatinine clearance equation.
- **CrCl** — creatinine clearance.
- **Deprescribing** — planned reduction or discontinuation.
- **Ductal-dependent lesion** — congenital heart lesion dependent on the ductus arteriosus.
- **eGFR** — estimated glomerular filtration rate.
- **Frailty** — multisystem vulnerability.
- **Gestational age** — age from the last menstrual period.
- **GFR** — glomerular filtration rate.
- **ICD** — implantable cardioverter-defibrillator.
- **Ideal body weight** — weight based on height and sex.
- **KDIGO** — Kidney Disease: Improving Global Outcomes.
- **Lean body weight** — weight excluding fat mass.
- **LVAD** — left ventricular assist device.
- **MELD** — Model for End-Stage Liver Disease.
- **Neonate** — infant in the first 28 days.
- **Pacemaker** — device for bradycardia pacing.
- **PALS** — Pediatric Advanced Life Support.
- **PEWS** — Pediatric Early Warning Score.
- **PLLR** — Pregnancy and Lactation Labeling Rule.
- **Polypharmacy** — multiple concurrent medications.
- **Postmenstrual age** — gestational plus postnatal age.
- **Preeclampsia** — pregnancy-specific hypertensive disorder.
- **Teratogen** — agent that causes fetal malformation.
- **Total body water** — fraction of body mass that is water.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Guidelines and regulation**

- Kidney Disease: Improving Global Outcomes. KDIGO 2024 Clinical Practice Guideline for the
  Evaluation and Management of Chronic Kidney Disease. *Kidney International*, 2024.
  https://kdigo.org/guidelines/ckd-evaluation-and-management/
- American College of Obstetricians and Gynecologists. Gestational Hypertension and
  Preeclampsia: ACOG Practice Bulletin No. 222, 2020.
  https://www.acog.org/clinical/clinical-guidance/practice-bulletin
- US Food and Drug Administration. Pregnancy and Lactation Labeling Final Rule, effective 30 June
  2015. https://www.fda.gov/vaccines-blood-biologics/biologics-rules/pregnancy-and-lactation-labeling-final-rule
- American Geriatrics Society. 2023 updated AGS Beers Criteria for potentially inappropriate
  medication use in older adults. *Journal of the American Geriatrics Society*, 2023.
  https://doi.org/10.1111/jgs.18372
- American Heart Association. Part 4: Pediatric Basic and Advanced Life Support: 2020 American
  Heart Association Guidelines. *Circulation*, 2020.
  https://www.ahajournals.org/doi/10.1161/CIR.0000000000000901
- Delgado C, Baweja M, Crews DC, et al. A unifying approach for GFR estimation: recommendations
  of the NKF-ASN Task Force. *Journal of the American Society of Nephrology*, 2021.
  https://doi.org/10.1681/ASN.2021070988

**Labeling**

- Apixaban prescribing information, United States.
  https://www.accessdata.fda.gov/scripts/cder/daf/
- Apixaban summary of product characteristics, European Medicines Agency.
  https://www.ema.europa.eu/en/medicines

**Background reading**

- Bickley LS. *Bates' Guide to Physical Examination and History Taking*, 13th edition.
- Brunton LL, Knollmann BC, et al. *Goodman and Gilman's The Pharmacological Basis of
  Therapeutics*.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_documentation-and-clinical-narratives.md`](08_documentation-and-clinical-narratives.md)
- Next topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Related topic: [`02_anatomy-physiology-and-vital-signs.md`](02_anatomy-physiology-and-vital-signs.md)
- Related topic: [`03_laboratory-medicine.md`](03_laboratory-medicine.md)
- Related topic: [`04_pharmacology-basics.md`](04_pharmacology-basics.md)
- Survey file: [`../../01_clinical_foundations.md`](../../01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

