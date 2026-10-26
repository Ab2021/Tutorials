# Pediatrics and Obstetrics

> Module 23 — Specialty Electives
> Topic 05 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a dosing
> authority, and not a substitute for the current WHO growth standards, CDC growth
> charts, ACOG practice guidance, or institutional formulary. Every threshold below
> names its source and version. Growth references and drug labels change; confirm the
> version in force at the date of your analysis.

## 1. Why this topic exists

Pediatrics and obstetrics are the two specialties where the unit of analysis is least
stable. A pediatric record changes meaning every month because the patient grows; an
obstetric record contains two patients whose outcomes are linked but not identical. A
data team that imports adult assumptions into either domain produces numbers that look
clinical and are not.

The ordinary path of a newborn prescription shows why. An infant born at 34 weeks and 2
days is now 10 days old and weighs 2.04 kg. The order is written in milligrams per
kilogram per day. The pharmacy system holds a weight field with no unit, a dose field
with no cap, and an age field that defaults to chronological age. The prescriber
intends 5 mg/kg/day, which is 10.2 mg. If the weight is keyed as 20.4 kg, the same
order becomes 102 mg, a tenfold overdose, and nothing in the system objects because 102
mg is a number. The error is caught only by a second, independent check: a weight that
implies a growth z-score of +10.8 at 35 weeks postmenstrual age is not a patient.

The signature failures of pediatric and obstetric data science are measurement
failures:

- **Age-clock confusion.** Gestational age, postmenstrual age, chronological age, and
  corrected age are four different variables. Using the wrong one changes the growth
  reference, the dosing interval, the screening eligibility, and the outcome
  definition.
- **Reference drift.** The World Health Organization (WHO) standards cover ages 0 to 5,
  the WHO 2007 reference covers 5 to 19, and the Centers for Disease Control and
  Prevention (CDC) 2000 charts cover 2 to 20. The CDC recommends WHO charts under age
  2. A pipeline that uses one chart for all ages has created a discontinuity that looks
  like a growth disorder.
- **Weight-unit errors.** Kilograms, grams, and pounds coexist in the same record.
  A factor of 1,000 or 2.2 is invisible to a type check.
- **Tenfold decimal errors.** A misplaced decimal in a weight-based dose is the classic
  pediatric harm. The safeguard is a hard cap plus an independent plausibility check,
  not a warning that a busy clinician can override.
- **Maternal-fetal confounding.** A treatment given to the mother affects the fetus,
  and the indication for the treatment is also a risk factor for the fetal outcome.
  Comparing exposed and unexposed pregnancies without the indication is a comparison of
  who was sick.
- **Denominator substitution.** Pregnancies, deliveries, and live births are three
  different denominators. A maternal mortality rate per live birth is not the same as a
  rate per pregnancy, and the difference is not a rounding error.

None of these failures raises an exception. The pipeline runs, the growth chart plots,
the dose is dispensed, and the cohort is quietly wrong.

### 1.1 Who is blocked when the measurement is wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Child | A dose that is tenfold too high, or a growth flag that is false | Cannot see the weight unit or the reference chart |
| Parent | Reassurance that is based on the wrong chart | Cannot audit the growth reference |
| Pediatrician | Trust in an alert that fires on a normal child | Cannot reconstruct the age clock |
| Pharmacist | A dose check that passes because the cap is missing | The rule engine is configured elsewhere |
| Obstetrician | A risk score that confounds indication with exposure | Cannot see the model's covariate set |
| Hospital | Avoidable harm, readmission, and unreimbursed software | Buys a product, not a measurement specification |
| Medicaid or insurer | A quality measure that reflects documentation, not care | Sees claims, not gestational age |
| Regulator | A pediatric claim that cannot be tied to a validated endpoint | Receives a summary, not the cohort |
| Data scientist | A model that validates internally and fails at another site | Inherited definitions were never documented |

The last row is the professional risk. A pediatric model that cannot state its age
clock, its growth reference, its weight unit, and its dose cap is not a weak model; it
is an unfalsifiable one. A pharmacist or a pediatrician will establish that in the
first meeting, usually by asking what the z-score is for a 2 kg infant at 35 weeks.

### 1.2 What this topic covers and what it does not

This topic covers the measurement layer. It explains how a pediatric or obstetric
episode becomes structured data, how the age clocks are defined and converted, how a
growth reference is chosen and applied, how weight-based and body-surface-area doses
are computed and capped, how maternal and fetal records are linked and confounded, and
how denominators are chosen. It does not teach pediatric practice, obstetric
management, or pharmacokinetics. It assumes familiarity with regression and does not
re-derive the LMS method.

The emphasis is deliberate. Most pediatric artificial-intelligence courses spend their
time on image models and almost none on the fact that the same child can have three
different ages in three different tables. The second problem destroys more deployments.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Convert between gestational age, chronological age, postmenstrual age, and corrected age, and state which clock a decision uses | Exercise 1; self-check 1 |
| 2 | Choose a growth reference for a stated age, gestation, and condition, and defend the choice | Exercise 2; interview question 2 |
| 3 | Compute an LMS z-score and percentile from a reference row, and flag implausible values | Exercise 3; self-check 3 |
| 4 | Check a weight-based or body-surface-area dose against a maximum cap and a tenfold-error guard | Exercise 4; self-check 4 |
| 5 | Link a maternal record to a neonatal record and name the confounders that the linkage creates | Exercise 5; interview question 5 |
| 6 | Choose the correct denominator among pregnancies, deliveries, and live births for a stated measure | Exercise 6; self-check 6 |
| 7 | Evaluate a rare-event pediatric or obstetric model with AUPRC, calibration, and subgroup counts | Exercise 7; interview question 7 |
| 8 | Map a pediatric or obstetric product to its regulatory and consent pathway | Exercise 8; interview question 9 |

Outcomes 1 and 2 are the measurement core. They are separated because a learner can
often compute an age and still be unable to say which reference the age selects. The
second skill is the one that survives a chart update.

Outcomes 3 and 4 are the quantitative core and the only outcomes that require
arithmetic. A learner who can compute a z-score but cannot state the reference row has
learned a formula, not a measurement system.

Outcomes 5 and 6 are about linkage and denominators. They are the obstetric half of
the topic, and they are assessed by constructing a comparison, not by naming a bias.

Outcomes 7 and 8 are about context. They connect the measurement layer to the
regulatory and equity questions that decide whether the work can be used.

### 2.1 How the outcomes map to the module

Outcomes 1 to 3 are prerequisites for the cross-specialty comparison in topic 07,
where age and time zero reappear under different names. Outcome 4 connects to module
14 (device regulation), because a dose calculator is often regulated as software.
Outcome 6 is shared with public health, where denominators decide every rate. Outcome
8 connects to module 01 (clinical foundations) and to the research-ethics material on
vulnerable populations.

If you retain only one outcome, retain outcome 1. Every other error is visible in a
table of values; an age-clock error is invisible because it produces a plausible
z-score and a plausible dose.

## 3. The core mental model

Pediatrics and obstetrics both run on clocks, and the clocks do not start at the same place. A pediatric record is
anchored at birth; an obstetric record is anchored at the last menstrual period or a dating ultrasound. The two
records meet at delivery, which is where most linkage errors occur.

```text
  MOTHER                                      FETUS / NEWBORN
    |                                               |
    | last menstrual period (LMP)                   |
    |        |                                      |
    |        +--> dating ultrasound --> estimated    |
    |             due date (EDD)                    |
    |        |                                      |
    |        v                                      |
    |  gestational age (GA) = time since LMP        |
    |        |                                      |
    |        |  pregnancy record --------------------+
    |        |                                      |
    |        v                                      v
    |   maternal outcome                      birth at GA
    |   (preeclampsia,                        |
    |    delivery mode,                       +--> chronological age (since birth)
    |    morbidity)                           |
    |                                         +--> postmenstrual age
    |                                              = GA at birth + chronological age
    |                                         |
    |                                         +--> corrected age
    |                                              = chronological age - weeks preterm
    v                                               v
  DENOMINATORS                              GROWTH REFERENCE
  pregnancies                               depends on age clock,
  deliveries                                gestation, sex, and condition
  live births
```

Read the diagram as two linked records with three clocks, not as one patient. The maternal record answers questions
about the pregnancy; the neonatal record answers questions about the child. A join on the delivery identifier creates
a dyad, the correct unit when the exposure is maternal and the outcome is neonatal.

### 3.1 The age clocks

| Clock | Origin | Used for | Typical error |
|---|---|---|---|
| Gestational age | First day of the last menstrual period, corrected by ultrasound or assisted reproduction | Pregnancy dating, preterm definition, screening eligibility | Using GA at birth as if it were the current age |
| Chronological age | Birth | Growth after the newborn period, dosing by age band, developmental milestones | Using it for a preterm infant without correction |
| Postmenstrual age | Conception, expressed as gestational age plus time since birth | Neonatal dosing, retinopathy of prematurity screening, apnea of prematurity | Using GA at birth and ignoring the days since birth |
| Corrected age | Chronological age minus the number of weeks born before 40 weeks | Growth and development for preterm infants, commonly to age 2 | Correcting beyond the period the guidance supports |

The arithmetic is simple and the consequences are not. An infant born at 34 weeks and 2 days who is now 10 days old
has a postmenstrual age of 35 weeks and 5 days, because 34 weeks 2 days is 240 days and 10 more days is 250 days,
which is 35 weeks and 5 days. The corrected age is 10 days minus 40 days, negative 30 days: the infant is still 30
days short of term. A pipeline that plots this infant at chronological age 10 days on a term chart has moved him
forward by 30 days and changed his growth reference.

### 3.2 The measurement model: reference, z-score, percentile

Growth is not a raw measurement. It is a position relative to a reference population, and the reference is a function
of age, sex, and sometimes condition. The LMS method, developed by Tim Cole, describes the distribution with three
parameters at each age: L for the power transformation, M for the median, and S for the coefficient of variation.

```text
  raw value (kg, cm, kg/m2)
        |
        +--> select reference row: sex, age clock, gestation, condition
        |
        +--> LMS parameters: L, M, S
        |
        v
  z = ((value / M)^L - 1) / (L * S)        [L != 0]
  z = ln(value / M) / S                    [L = 0]
        |
        v
  percentile = normal CDF(z)
```

The reference choice is the measurement. A weight of 2.04 kg is at the 17th percentile against one row and below the
0.1st percentile against another. Neither is wrong; they answer different questions, and the defect is reporting a
percentile without naming the reference, the age clock, and the sex.

### 3.3 The maternal-fetal dyad and confounding

Obstetric data has two outcomes and one exposure. A treatment given during pregnancy can affect the mother, the
fetus, or both, and the indication for the treatment is usually also a risk factor for the outcome. This is
confounding by indication, and it is structural rather than accidental.

```text
  indication (e.g., preterm labour)
        |                    \
        v                     v
  treatment given          outcome risk
        \                     /
         \___________________/
          observed association
```

A naive comparison of treated and untreated pregnancies estimates the combined effect of the drug, the indication, and
the decision to treat. The defensible approaches are a target trial framework with a defined eligibility window, a
new-user active comparator design, and negative-control outcomes. None removes unmeasured severity, and all require
the gestational-age clock to be correct.

### 3.4 The denominator ladder

Obstetric rates are exquisitely sensitive to the denominator because the numerator is small and the choices differ by 1 to 2 percent.

| Denominator | Counts | Used for | Consequence of substitution |
|---|---|---|---|
| Pregnancies | All pregnancies, including multiples and losses | Incidence of pregnancy-related conditions | Higher than live-birth denominators |
| Deliveries | Birth events, with multiples counted once | Delivery-related procedures | Lower than pregnancies, higher than births |
| Live births | Infants born alive | Infant and maternal mortality rates | Excludes stillbirths and losses |
| Women | Unique women | Prevalence and recurrence | Lower than any event count |

A maternal death after a stillbirth is a pregnancy-related death but not a live-birth death. A rate per 100,000 live
births therefore excludes exactly the events that are most severe, and the denominator is part of the definition.

## 4. Definitions and vocabulary

Every term below is used later. The common-misuse column is the part worth memorising,
because the misuse is what appears in real analytic tables.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Gestational age | Time since the first day of the last menstrual period, corrected by ultrasound or assisted reproduction | Treated as age since conception | Completed weeks and days |
| Chronological age | Time since birth | Used for preterm infants without correction | Days, months, years |
| Postmenstrual age | Gestational age at birth plus chronological age | Confused with gestational age at birth | Weeks and days |
| Corrected age | Chronological age minus weeks born before 40 | Applied beyond the supported period | Weeks or months |
| Estimated due date | Expected date of delivery, 280 days from the last menstrual period | Treated as a fixed date rather than an estimate | Date |
| Preterm birth | Birth before 37 completed weeks of gestation | Defined by birth weight instead of gestation | Completed weeks |
| Late preterm | Birth at 34 weeks 0 days to 36 weeks 6 days | Pooled with term infants | Completed weeks |
| Small for gestational age | Birth weight below the 10th percentile for gestational age and sex | Used as a synonym for growth restriction | Percentile |
| Intrauterine growth restriction | A fetal growth pattern below genetic potential | Used as a synonym for small for gestational age | Clinical diagnosis |
| Large for gestational age | Birth weight above the 90th percentile | Defined without a reference | Percentile |
| LMS | Cole's method using L, M, and S to describe a reference distribution | Applied with parameters from a different age or sex | Reference-specific |
| Z-score | Standard deviations from the reference median | Reported without the reference row | Standard deviation |
| Percentile | Position in the reference distribution | Treated as a clinical diagnosis | 0–100 |
| Growth standard | A prescriptive description of how children should grow | Confused with a reference | WHO 2006 |
| Growth reference | A descriptive description of how children did grow | Used as a standard | CDC 2000 |
| WHO 2006 | WHO Child Growth Standards, birth to 5 years | Used beyond 5 years | 0–60 months |
| WHO 2007 | WHO Growth Reference, 5 to 19 years | Used under 5 years | 61–228 months |
| CDC 2000 | CDC growth charts, 2 to 20 years | Used under 2 years in US clinical care | 24–240 months |
| Fenton 2013 | Preterm growth chart, 22 to 50 weeks postmenstrual age | Used for term infants | Weeks |
| INTERGROWTH-21st | International fetal and newborn growth standards | Treated as interchangeable with Fenton | Weeks |
| Body surface area | Estimated surface area used for some drug doses | Computed with the wrong unit or formula | Square metres |
| Mosteller formula | BSA = square root of height in cm times weight in kg divided by 3,600 | Applied with height in metres | Square metres |
| Maximum dose | The adult or protocol ceiling for a drug | Omitted in pediatric order sets | mg or mg/kg |
| Tenfold error | A decimal slip that multiplies or divides a dose by 10 | Treated as a prescriber problem rather than a system problem | Factor |
| Preeclampsia | New hypertension after 20 weeks with proteinuria or organ dysfunction | Diagnosed before 20 weeks | Clinical definition |
| Gestational hypertension | New hypertension after 20 weeks without proteinuria or severe features | Confused with preeclampsia | Clinical definition |
| Eclampsia | Seizure in a patient with preeclampsia | Used for any seizure in pregnancy | Clinical diagnosis |
| Severe maternal morbidity | Transfusion, hysterectomy, ICU admission, organ failure | Coded inconsistently across hospitals | ICD-10-CM codes |
| Maternal mortality | Death during pregnancy or within 42 days of delivery | Denominated inconsistently | Per 100,000 live births |
| Umbilical artery pH | Arterial cord blood pH at delivery | Measured on venous blood | pH units |
| Base deficit | Buffer deficit in cord blood | Reported without the extracellular correction | mmol/L |
| Cardiotocography | Continuous fetal heart rate and uterine contraction recording | Interpreted without inter-observer agreement data | Trace |
| Retinopathy of prematurity | Proliferative retinal vascular disease in preterm infants | Screened by birth weight alone | Stage and plus disease |
| Phoenix criteria | 2024 pediatric sepsis definition using a four-organ score | Applied to preterm infants below 37 weeks | Score 0–13 |
| Assent | A child's agreement to research participation | Treated as consent | Regulatory concept |

### 4.1 Ages that look interchangeable

The single most consequential distinction in the file is between gestational age and
postmenstrual age. Gestational age is fixed at birth; postmenstrual age keeps
increasing. A neonatal intensive care unit (NICU) order set that stores "age" without a
definition will apply a 34-week dosing interval to an infant who is now 40 weeks
postmenstrual, or the reverse.

Corrected age is a third variable with a limited window. The American Academy of
Pediatrics recommends correcting for prematurity in growth and developmental
assessment, commonly to age 2, but the exact period depends on the degree of
prematurity and the instrument. Applying a correction at age 5 to a child born at 34
weeks overcorrects by six weeks and can mask a real developmental concern.

### 4.2 Denominators that look interchangeable

Three obstetric rates are routinely confused.

First, the maternal mortality ratio is deaths per 100,000 live births. It excludes
stillbirths and pregnancy losses from the denominator but includes deaths up to 42 days
after delivery. A pregnancy-related death after a stillbirth is in the numerator and
not in the denominator, which raises the ratio.

Second, the stillbirth rate is deaths per 1,000 total births, where total births
include live births and stillbirths. Substituting live births alone changes the rate by
roughly the stillbirth fraction.

Third, the preterm birth rate is preterm live births divided by all live births. A
hospital that transfers preterm mothers before delivery will have a lower rate than the
population it serves, because the births happen elsewhere.

### 4.3 What counts as an event

| Endpoint | Event | Denominator | Ascertainment risk |
|---|---|---|---|
| Preterm birth | Live birth before 37 weeks | Live births | Gestational-age dating method |
| Preeclampsia | New hypertension after 20 weeks plus proteinuria or organ dysfunction | Deliveries or pregnancies | Definition version and late onset |
| Severe maternal morbidity | Transfusion, hysterectomy, ICU, organ failure | Deliveries | Coding and hospital variation |
| Maternal mortality | Death during pregnancy or within 42 days | Live births | Underreporting and review-committee attribution |
| Neonatal acidemia | Umbilical artery pH below 7.00 or base deficit at least 12 mmol/L | Deliveries with cord gas | Sampling site and missing gases |
| Growth failure | Weight-for-age z below −2 | Children with a measured weight | Reference choice and measurement error |
| Adverse drug event | Harm from a medication | Doses administered | Underreporting and causality attribution |
| ROP requiring treatment | Stage 3 plus disease or worse | Infants screened | Examiner variability and image quality |

> **Note:** "Per 1,000 live births" and "per 1,000 deliveries" are not the same rate.
> The difference is small in a term population and large in a population with
> stillbirths or multiples.
