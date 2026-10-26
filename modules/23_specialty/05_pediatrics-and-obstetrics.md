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
## 5. The mechanics, step by step

This section is the procedure for turning pediatric and obstetric records into an analysable cohort. It is written as
ten steps. Each step states its input, its output, the reason it exists, its branch points, and its edge cases. A team
that can execute these ten steps in order, and document the choices, can defend the resulting dataset to a
pediatrician, an obstetrician, a pharmacist, and a replication attempt.

The procedure assumes the question is prognostic, comparative, or safety-related in a defined pediatric or obstetric
population. Health-services questions add requirements (capacity, access, insurance) that are noted but not developed.

### Step 1 — Fix the decision and the unit of analysis

**Input.** A written question naming a decision, a decision-maker, an action, and a population.

**Output.** An eligibility specification and an explicit unit of analysis: child, pregnancy, delivery, dyad, or dose.

**Rationale.** The unit of analysis determines the denominator, the independence assumption, and the linkage. A
question about a child's growth uses the child; a question about a drug's effect on the fetus uses the pregnancy or
the dyad. Mixing them creates duplicated rows and non-independent observations.

**Procedure.**

1. Write the decision as a sentence with an actor and a verb: "the pediatrician will start caffeine for apnea using
   this risk score."
2. State whether the unit is the child, the pregnancy, the delivery, or the mother-infant dyad, and record why.
3. If the unit is the dyad, define the linkage key and the direction of the exposure.
4. Define the age range by the clock the decision uses, not by a convenient calendar field.
5. Specify the data-availability rule and count every exclusion.

**Branch point.** If the exposure is maternal and the outcome is neonatal, the unit is the dyad and the mother is the
index patient. If the exposure is neonatal and the outcome is developmental, the unit is the child and the mother is a
covariate.

**Edge case.** Multiple gestation. Twins share a pregnancy but not an outcome. A pregnancy-level analysis counts the
pregnancy once; a child-level analysis counts each twin, and the observations are correlated. Use a mixed model or
cluster-robust standard errors, and say which unit the estimate describes.

### Step 2 — Establish the gestational-age clock

**Input.** Last menstrual period, ultrasound reports with crown-rump length or biparietal diameter, assisted
reproduction dates, and the delivery record.

**Output.** One gestational age at birth, in completed weeks and days, with the method that established it and the
date of the estimate.

**Rationale.** Gestational age sets preterm status, screening eligibility, growth reference, dosing band, and outcome
definition. A one-week error moves an infant across a clinical threshold, and the error is not random: it is larger
when dating is based on the last menstrual period alone.

**Procedure.**

1. Prefer assisted reproduction dates when available; they establish the clock most precisely.
2. Prefer a first-trimester ultrasound at or before 13 weeks 6 days using crown-rump length. The American College of
   Obstetricians and Gynecologists (ACOG) Committee Opinion 700 (2017) treats this as the most accurate method.
3. Apply the ACOG redating thresholds: revise the estimate if the ultrasound differs by more than 5 days at or before
   8 weeks 6 days, more than 7 days at 9 weeks 0 days to 15 weeks 6 days, more than 10 days at 16 to 21 weeks 6 days,
   more than 14 days at 22 to 27 weeks 6 days, and more than 21 days at 28 weeks or later.
4. If no ultrasound before 22 weeks 0 days exists, record the pregnancy as suboptimally dated and run a sensitivity
   analysis.
5. Record the method and the version of the rule used. A gestational age without a method is not interpretable.

**Branch point.** If the last menstrual period and the ultrasound disagree below the redating threshold, keep the
last menstrual period estimate; above the threshold, adopt the ultrasound estimate. Record both values.

**Edge case.** A patient who presents late with an uncertain last menstrual period. The gestational age may be
estimated from a later ultrasound with a wider error, and the preterm classification becomes uncertain. Report the
uncertainty rather than assigning a point value.

### Step 3 — Convert between age clocks

**Input.** Gestational age at birth, date of birth, current date, and the intended use of the age.

**Output.** Chronological age, postmenstrual age, and corrected age, each with its definition and the period over
which it is valid.

**Rationale.** The three ages answer different questions. Chronological age is time since birth. Postmenstrual age is
gestational age at birth plus chronological age, and it is the variable neonatal dosing and retinopathy screening use.
Corrected age subtracts the degree of prematurity and is used for growth and development in early childhood.

**Procedure.**

1. Compute chronological age in days from the birth date.
2. Compute postmenstrual age as gestational age at birth in days plus chronological age in days, then convert to weeks
   and days.
3. Compute corrected age as chronological age minus the number of days between birth and 40 weeks of gestation.
4. Record the clock used for each derived variable. A growth z-score and a dose interval may use different clocks.
5. Apply correction only within the period the guidance supports; the American Academy of Pediatrics (AAP) commonly
   supports correction to age 2, and the exact period depends on the degree of prematurity.
6. Do not compute a corrected age for a term infant; the correction is zero and adds no information.

**Branch point.** If the infant is still in the newborn period, postmenstrual age is the primary clock. If the infant
has reached term-equivalent age, chronological or corrected age becomes primary depending on the question.

**Edge case.** A leap year or a time-zone change at birth. Day counts computed by subtracting dates can be off by one.
Use a date library with explicit day arithmetic and test the boundary.

### Step 4 — Choose the growth reference

**Input.** Age, sex, gestational age, current age clock, and any condition that affects growth.

**Output.** One named reference, one named age variable, and the LMS row used.

**Rationale.** A percentile is a position in a named distribution. The distribution differs by age, sex, gestation,
and condition, and the choice changes the answer. The CDC recommends the WHO Child Growth Standards from birth to 2
years and the CDC 2000 charts from 2 to 20 years for United States clinical care, a recommendation published in the
Morbidity and Mortality Weekly Report in 2010 (59(RR-9)).

**Decision table for reference selection.** More than two outputs exist, so the branch must be explicit.

| Population and age | Reference | Age variable | Consequence if mishandled |
|---|---|---|---|
| Birth to under 24 months, term or corrected preterm | WHO 2006 Child Growth Standards | Chronological or corrected age | CDC charts under 2 misclassify normal breastfed growth |
| 24 to 60 months | WHO 2006, or CDC 2000 if the program is CDC-based | Chronological age | Mixing charts creates a step at the transition |
| 5 to 19 years | WHO 2007 Growth Reference, or CDC 2000 for US clinical care | Chronological age | Adult body mass index thresholds are wrong before 18 |
| Preterm, before term-equivalent age | Fenton 2013 or INTERGROWTH-21st | Postmenstrual age | Term charts label normal preterm infants as severely small |
| Down syndrome or Turner syndrome | Condition-specific chart | Chronological age | General charts over-diagnose short stature |
| Unknown age or gestation | None | None | A z-score cannot be computed |
| Pregnancy (maternal) | Not applicable | Gestational age | Adult growth charts do not apply to pregnancy |

**Branch point.** If a program changes reference at age 2, the percentile can jump at the boundary because the
references differ. Report the discontinuity and avoid interpreting a single boundary crossing as a growth event.

**Edge case.** A child adopted from another country with an unknown birth date. The age is uncertain, and the growth
position is uncertain with it. Record the uncertainty and do not impute a precise age.

### Step 5 — Compute LMS z-scores and percentiles

**Input.** A measurement with its unit, a reference row with L, M, and S, and the sex and age used to select the row.

**Output.** A z-score, a percentile, the reference name and version, and a plausibility flag.

**Rationale.** The LMS method converts a measurement into a standard deviation score that is comparable across ages.
The formula is not optional arithmetic; the L parameter exists because growth measurements are skewed, and treating
them as normal without the transformation biases the tails.

**Procedure.**

1. Confirm the measurement unit and convert to the reference unit. Weight in kilograms, length or height in
   centimetres, head circumference in centimetres.
2. Select the row by sex and by the age clock the reference requires. For preterm references, the clock is
   postmenstrual age.
3. Compute the z-score. For L not equal to zero: z equals the quantity (value divided by M) raised to L, minus 1,
   divided by the product of L and S. For L equal to zero: z equals the natural logarithm of value divided by M,
   divided by S.
4. Convert the z-score to a percentile with the standard normal cumulative distribution function.
5. Apply the plausibility flags. The WHO tables flag weight-for-age z below −6 or at least 5, height-for-age z below
   −6 or at least 6, and weight-for-length, body mass index, or head circumference z below −5 or above 5.
6. Store the reference name, the version, the age, the sex, and the raw value with the z-score.
7. For a series of measurements, compute the change in z rather than the change in raw value. Crossing two major
   percentile lines is clinically meaningful even when each value is within the normal range.

**Branch point.** If the measurement is a length before age 2 and a height after, the two are not the same variable.
Recumbent length is about 0.8 cm longer than standing height, so a chart transition can produce a spurious drop.

**Edge case.** A value exactly at a reference boundary, such as a weight that rounds to the median. Report the
unrounded z-score; do not round the input before computing.

### Step 6 — Compute weight-based and body-surface-area doses with caps

**Input.** Weight in kilograms with its unit and date, height in centimetres, the prescribed dose and its basis,
the maximum dose, and the patient's age and postmenstrual age.

**Output.** A calculated dose, the basis, the cap applied, and a pass or reject decision from the guardrails.

**Rationale.** Pediatric doses are usually milligrams per kilogram or milligrams per square metre, with a maximum
adult or protocol dose. A dose without a cap is not a pediatric dose. The classic harm is a tenfold decimal error, and
the only reliable defence is a hard stop plus an independent plausibility check.

**Procedure.**

1. Confirm the weight unit. Convert pounds to kilograms with 2.20462, and grams to kilograms with 1,000. Reject a
   weight with no unit.
2. Check the weight against an age- and gestation-specific plausible range. A 10-day-old infant at 35 weeks
   postmenstrual age cannot weigh 20 kg.
3. Compute the dose on the stated basis. For milligrams per kilogram, multiply. For milligrams per square metre,
   compute body surface area first.
4. Compute body surface area with the Mosteller formula: the square root of height in centimetres times weight in
   kilograms divided by 3,600.
5. Apply the maximum dose. If the calculated dose exceeds the cap, either cap it and document the cap, or reject the
   order. Never silently exceed it.
6. Apply a tenfold guard. Compare the calculated dose with a reference dose for the indication and reject a factor of
   10 or more in either direction.
7. Check the daily total across all orders, not only the single dose. Duplicate orders are a common source of
   overdose.
8. Record every check and its result so that the decision can be audited.

**Decision table for the dose guard.**

| Situation | Action | Output |
|---|---|---|
| Weight unit missing | Reject | No dose; pharmacy query |
| Weight outside the plausible range for age and gestation | Reject | No dose; safety event |
| Dose exceeds the maximum per dose | Cap or reject per protocol | Capped dose with reason |
| Daily total exceeds the maximum per day | Reject | No dose; prescriber query |
| Calculated dose is 10 times the reference dose | Reject | No dose; tenfold-error flag |
| Height is in metres where centimetres are expected | Reject | No dose; unit flag |
| All checks pass | Accept | Dose with basis and checks |

**Branch point.** If the drug has no maximum dose because it is titrated to effect, the cap is replaced by a
protocol-defined starting range and a maximum rate. Record the replacement.

**Edge case.** Obesity. A weight-based dose computed on actual weight can exceed the adult dose. Many protocols use
adjusted body weight or cap at the adult dose. The choice is drug-specific and must be stated.

### Step 7 — Link the dyad and handle maternal-fetal confounding

**Input.** Maternal records, delivery records, and neonatal records with identifiers and times.

**Output.** A linked dyad table with one row per mother-infant pair, plus a linkage quality report.

**Rationale.** The maternal and neonatal records are separate systems with separate identifiers. A join on surname,
date, and hospital is probabilistic and can fail, and a failed join is not random: it is more common for mothers who
transfer, deliver outside the system, or have a different surname from the infant.

**Procedure.**

1. Link on a stable identifier where one exists, such as a shared encounter or a maternal-newborn linkage number.
2. Where only probabilistic linkage is possible, report the match rate, the false-positive rate from a reviewed
   sample, and the characteristics of unmatched records.
3. Define the exposure window relative to the gestational-age clock, not to the calendar.
4. List the confounders that the indication creates. For a treatment for preterm labour, the indication is also a
   risk factor for preterm birth and for neonatal morbidity.
5. Choose a design that addresses the indication: a target trial framework, an active comparator, or a
   negative-control outcome.
6. Report the analysis at the dyad level and state whether the estimate describes the mother, the child, or the pair.

**Branch point.** If the linkage rate differs between exposed and unexposed pregnancies, the comparison is biased by
linkage. Report the rate by exposure group.

**Edge case.** A mother with two pregnancies in the window and two infants. Each pregnancy is a separate dyad, but the
mother appears twice. Use a cluster-robust variance or a mixed model.

### Step 8 — Choose and audit the denominator

**Input.** The event definition, the population at risk, and the available counts of pregnancies, deliveries, live
births, and women.

**Output.** A stated denominator with its source, the rate, and a sensitivity analysis under an alternative
denominator.

**Rationale.** The denominator is part of the definition. A maternal mortality ratio per live birth and a
pregnancy-related mortality rate per pregnancy answer different questions, and the difference is largest exactly where
the events are most severe.

**Decision table for denominator selection.**

| Measure | Correct denominator | Alternative that changes the answer | Why it matters |
|---|---|---|---|
| Maternal mortality ratio | Live births | Pregnancies | Excludes stillbirths from the denominator |
| Pregnancy-related mortality | Pregnancies or women | Live births | Includes losses that the ratio excludes |
| Preterm birth rate | Live births | Deliveries | Multiples counted once in deliveries |
| Stillbirth rate | Total births, live plus still | Live births | Excludes the numerator from the denominator |
| Severe maternal morbidity | Deliveries | Live births | Includes intrapartum events |
| Neonatal mortality | Live births | Total births | Excludes stillbirths |
| ROP screening coverage | Infants eligible by gestation and weight | All NICU admissions | Eligibility is not the same as admission |

**Procedure.**

1. Write the numerator and denominator as a sentence before computing the rate.
2. Match the denominator to the numerator: a birth outcome uses births, a pregnancy outcome uses pregnancies.
3. Report the count in the numerator, not only the rate. A rate from 3 events is not a stable estimate.
4. Report the source of the denominator and the year.
5. Compute the rate under at least one alternative denominator and report the difference.

**Branch point.** If the facility transfers high-risk mothers before delivery, a facility-level rate undercounts the
events. Use a population-based denominator or a risk-adjusted comparison.

**Edge case.** A death after 42 days. The standard maternal mortality definition stops at 42 days, but a
pregnancy-related death can occur later. State the window and do not mix definitions.

### Step 9 — Audit the assembled dataset

**Input.** The analytic table and the code that built it.

**Output.** An audit report with counts, cross-tabs, recomputation checks, and a list of resolved and unresolved
discrepancies.

**Rationale.** Every step above can fail silently. The audit is the only defence, and it is cheap relative to the cost
of a retracted model or a harmed child.

**Procedure.**

1. Recompute the gestational age from the source documents and compare with the stored value.
2. Cross-tabulate the age clock used against the variable it feeds. Any variable fed by two clocks is a risk.
3. Recompute a sample of z-scores by hand and compare with the pipeline.
4. Recompute a sample of doses by hand, including the cap and the tenfold check.
5. Count missing weights, heights, and gestational ages by subgroup, and distinguish not measured from measured and
   missing.
6. Reconcile the dyad count against the delivery count and the infant count.
7. Recompute the rate under an alternative denominator.
8. Produce a cohort flow diagram with every exclusion and its count.

**Branch point.** If the recomputation disagreement exceeds a pre-set threshold, stop and fix the derivation before
any modelling.

**Edge case.** Duplicate rows from a transfer between hospitals. Two delivery records for one birth can double the
cohort. Deduplicate on the person identifier and the date, and record which row won.

### Step 10 — Freeze, version, and document

**Input.** The audited dataset and the code that built it.

**Output.** A versioned extract with a data dictionary, a changelog, and a reproducibility statement.

**Rationale.** Growth references, dosing labels, and obstetric definitions change. A dataset without a version cannot
be compared with the next extract, and a result without a version cannot be reproduced after the next chart release.

**Procedure.**

1. Pin the reference name and version, the age-clock definitions, the dosing source, the denominator, and the code
   commit.
2. Write a data dictionary with every derived field, its rule, and its source fields.
3. Record every exclusion with a count and a reason, and store the audit outputs with the dataset.
4. State the intended use and the known limits in a model or dataset card.

**Edge case.** A reference update arrives mid-project. Freeze the analysis on the version in force at the extract date
and run a sensitivity analysis under the new version. Do not silently re-plot the cohort.

## 6. Worked example

An infant is born at 34 weeks and 2 days by vaginal delivery after spontaneous preterm labour. Birth weight is 1,980 g,
birth length 44 cm, and head circumference 31 cm. He is now 10 days old and weighs 2,040 g. He has apnea of
prematurity and is prescribed caffeine citrate at a loading dose of 20 mg/kg and a maintenance dose of 5 mg/kg/day,
with a unit protocol cap of 10 mg/kg/day for maintenance.

### 6.1 The clocks

| Clock | Calculation | Value |
|---|---|---|
| Gestational age at birth | 34 weeks 2 days = 240 days | 34+2 |
| Chronological age | Day 10 of life | 10 days |
| Postmenstrual age | 240 + 10 = 250 days | 35+5 |
| Corrected age | 10 − (280 − 240) = 10 − 40 | −30 days |

The infant is 30 days short of term. A pipeline that plots him at chronological age 10 days has moved him forward by
30 days and selected a term reference. A pipeline that doses by gestational age at birth has frozen him at 34+2 and
ignored ten days of maturation. Both errors are invisible in a table that stores only "age: 10 days".

The dating itself is a measurement with a rule. The last menstrual period was uncertain, so a first-trimester
ultrasound at 12 weeks 3 days was used to establish the estimated due date, and the recorded gestational age at birth
is 34+2. Under ACOG Committee Opinion 700, a first-trimester crown-rump measurement is the most accurate dating method,
and the redating threshold in the 9 to 15 week window is 7 days. If a later ultrasound had disagreed by 9 days, the
estimate would be revised; at 5 days it would not. The gestational age in the record is therefore the output of a
rule, not a direct observation, and the rule version belongs in the data dictionary.

### 6.2 Growth z-scores

The teaching reference rows are supplied with this case. The preterm row is the value the institution's extract uses
at 35+5 weeks for boys; the term row is the value a term chart would use at 10 days. Both are stated so the arithmetic
can be checked, and both must be replaced with the official table row in production.

| Row | L | M (kg) | S | Weight (kg) | z | Percentile |
|---|---:|---:|---:|---:|---:|---:|
| Preterm, 35+5 weeks | −0.15 | 2.40 | 0.17 | 2.04 | −0.97 | 16.7 |
| Term, 10 days | 0.20 | 3.20 | 0.13 | 2.04 | −3.31 | 0.05 |

The preterm z-score is computed as:

```text
z = ((2.04 / 2.40)^(-0.15) - 1) / (-0.15 * 0.17)
  = (0.85^(-0.15) - 1) / (-0.0255)
  = (1.02468 - 1) / (-0.0255)
  = -0.97
```

The term z-score is:

```text
z = ((2.04 / 3.20)^(0.20) - 1) / (0.20 * 0.13)
  = (0.6375^(0.20) - 1) / 0.026
  = (0.91391 - 1) / 0.026
  = -3.31
```

The difference is 2.34 standard deviations. On the preterm row the infant is at the 16.7th percentile, which is
normal. On the term row he is at the 0.05th percentile, which triggers a growth-failure pathway. The measurement did
not change; the reference did.

### 6.3 The dose and the tenfold error

The maintenance dose is 5 mg/kg/day. At the correct weight of 2.04 kg:

```text
5 mg/kg/day * 2.04 kg = 10.2 mg/day
```

The loading dose is 20 mg/kg:

```text
20 mg/kg * 2.04 kg = 40.8 mg
```

The unit protocol caps maintenance at 10 mg/kg/day, which is 20.4 mg/day for this infant. The correct 10.2 mg is
below the cap.

Now suppose the weight is keyed as 20.4 kg instead of 2.04 kg, a single decimal shift. The maintenance dose becomes
5 × 20.4 = 102 mg/day, which is exactly ten times the intended dose. The excess is 91.8 mg/day. The cap is 20.4 mg,
so the order exceeds it by a factor of five, and a cap check alone would catch the error only if the cap is stored in
milligrams and not recomputed from the same wrong weight.

### 6.4 A second dose on a body-surface-area basis

Some neonatal drugs are dosed per square metre, which adds a second arithmetic path. With height 44 cm and weight
2.04 kg, the Mosteller body surface area is:

```text
BSA = sqrt(44 * 2.04 / 3600) = sqrt(0.024933) = 0.158 m2
```

At 100 mg per square metre, the dose is 100 × 0.158 = 15.8 mg. With the erroneous 20.4 kg weight and the same height:

```text
BSA = sqrt(44 * 20.4 / 3600) = sqrt(0.249333) = 0.499 m2
```

The dose becomes 100 × 0.499 = 49.9 mg, which is 3.16 times the correct dose. If the protocol caps this drug at 50 mg
per dose, the cap does not fire, because 49.9 mg is below 50 mg. The cap protects against an adult-sized dose, not
against a threefold error in a small patient. This is why a cap and a plausibility check are different controls and
both are required.

### 6.5 Two diagnostics that reveal a broken analysis

The first diagnostic is the growth plausibility check. At 35+5 weeks postmenstrual age, a weight of 20.4 kg produces
a z-score of:

```text
z = ((20.4 / 2.40)^(-0.15) - 1) / (-0.0255)
  = (8.5^(-0.15) - 1) / (-0.0255)
  = (0.72538 - 1) / (-0.0255)
  = +10.77
```

A z-score of +10.8 is not a large baby; it is an impossible one. The same weight also changes body surface area. With
length 44 cm, the Mosteller estimate at 2.04 kg is the square root of (44 × 2.04 / 3,600), which is 0.158 square
metres. At 20.4 kg it is the square root of (44 × 20.4 / 3,600), which is 0.499 square metres, 3.16 times larger. A
drug dosed per square metre would be tripled as well as multiplied by ten if the weight also entered the BSA.

The second diagnostic is the age clock. Suppose the pipeline selects the term reference because the infant is "10 days
old". The z-score moves from −0.97 to −3.31, a shift of 2.34 standard deviations, and the percentile moves from 16.7
to 0.05. The clinical consequence is a false growth-failure flag, a feeding intervention, and possibly a longer stay.
The error is not in the weight; it is in the clock that selects the row.

One caveat about thresholds: a guard set at exactly 10 times catches this error because the ratio is exactly 10.0, but
a 9 times error passes. Set the guard on a plausible range and a maximum dose, and treat the tenfold check as a backstop
rather than the primary control. Had the weight been transmitted in grams and stored as kilograms, the ratio would be
1,000 and the dose 10,200 mg; the same ratio check catches both, which is why the ratio matters more than the absolute
value.

Both diagnostics are cheap: compute the z-score under both references and count the discordance; check the weight
against an age-specific range. Neither requires new data. Neither is a modelling question.

### 6.6 Vary one assumption

The birth weight is 1,980 g. Suppose it had been 1,480 g, which is within the range seen in this population. At birth,
using a 34+2 row with L = −0.15, M = 2.10 kg, and S = 0.17:

```text
At 1.98 kg: z = ((1.98 / 2.10)^(-0.15) - 1) / (-0.0255) = -0.35
At 1.48 kg: z = ((1.48 / 2.10)^(-0.15) - 1) / (-0.0255) = -2.11
```

The conclusion changes from appropriate for gestational age to small for gestational age, because the second value
crosses the −2 threshold. The clinical pathway changes with it: different feeding, different glucose monitoring, and a
different growth trajectory. A 500 g difference in birth weight, which is about 25 percent, moves the classification
across the diagnostic boundary. The boundary is a convention, and the convention is doing the work.

### 6.7 What the case does not establish

The case is a measurement worked example, not a treatment claim. It does not show that caffeine reduces apnea, that
the infant is thriving, or that the unit's cap is correct. It does not generalise to a different reference row, a
different gestation, or a different drug.

What it does establish is narrower and more useful: every number in the case depends on a choice that is not in the
data. The table separates the choices.

| Choice | Value used here | Alternative that changes the answer | How to disclose it |
|---|---|---|---|
| Gestational age | 34+2 by ultrasound | Last menstrual period dating | Method and date fields |
| Age clock | Postmenstrual 35+5 | Chronological 10 days | Clock field per variable |
| Growth reference | Preterm row at 35+5 | Term row at 10 days | Reference name and version |
| Weight unit | Kilograms | Grams or pounds | Unit field and range check |
| Dose cap | 10 mg/kg/day | No cap, or adult cap | Cap source and value |
| Denomination | Per day | Per dose | Basis field |
| Birth weight | 1,980 g | 1,480 g | Raw value with unit |

An analysis that discloses these seven choices can be reproduced and criticised. One that discloses none produces a
number that cannot be checked.
## 7. Code, query, or calculation

The block below implements the LMS z-score, the normal percentile, a weight-based dose
with a maximum cap, a tenfold guard, and the Mosteller body surface area. It is
standard-library Python and runs as written.

### 7.1 Growth and dose calculator

```python
"""Growth z-scores and weight-based dosing checks. Standard library only."""

import math


def z_lms(x, L, M, S):
    if abs(L) < 1e-9:
        return math.log(x / M) / S
    return ((x / M) ** L - 1.0) / (L * S)


def percentile(z):
    return 100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def dose_mg(weight_kg, mg_per_kg, max_mg=None):
    raw = weight_kg * mg_per_kg
    capped = min(raw, max_mg) if max_mg is not None else raw
    return raw, capped, (max_mg is not None and raw > max_mg)


def tenfold_guard(entered_kg, prior_kg, ratio=10.0):
    return entered_kg / prior_kg >= ratio or prior_kg / entered_kg >= ratio


def bsa_mosteller(height_cm, weight_kg):
    return math.sqrt(height_cm * weight_kg / 3600.0)


preterm = z_lms(2.04, -0.15, 2.40, 0.17)
term = z_lms(2.04, 0.20, 3.20, 0.13)
print(f"preterm z={preterm:+.2f} percentile={percentile(preterm):.1f}")
print(f"term    z={term:+.2f} percentile={percentile(term):.2f}")

raw, capped, hit = dose_mg(2.04, 5.0, max_mg=20.4)
print(f"maintenance raw={raw:.1f} mg capped={capped:.1f} cap_hit={hit}")
raw_bad, capped_bad, hit_bad = dose_mg(20.4, 5.0, max_mg=20.4)
print(f"bad weight raw={raw_bad:.1f} mg capped={capped_bad:.1f} cap_hit={hit_bad}")
print("tenfold guard:", tenfold_guard(20.4, 2.04))

print(f"bsa correct={bsa_mosteller(44, 2.04):.3f} m2")
print(f"bsa wrong  ={bsa_mosteller(44, 20.4):.3f} m2")
print(f"z at 20.4 kg, 35+5 row = {z_lms(20.4, -0.15, 2.40, 0.17):+.2f}")
```

Expected output:

```text
preterm z=-0.97 percentile=16.7
term    z=-3.31 percentile=0.05
maintenance raw=10.2 mg capped=10.2 cap_hit=False
bad weight raw=102.0 mg capped=20.4 cap_hit=True
tenfold guard: True
bsa correct=0.158 m2
bsa wrong  =0.499 m2
z at 20.4 kg, 35+5 row = +10.77
```

The cap fires on the erroneous weight, but only because the cap is expressed in
milligrams. If the cap were stored as 10 mg/kg/day and recomputed from the same wrong
weight, it would produce 204 mg and pass. The tenfold guard is independent of the cap and
catches the error regardless of how the cap is expressed.

### 7.2 What the code does not do

The reference rows are illustrative. The L, M, and S values in the block are teaching
values, not the official WHO or CDC table rows, and a production system must read the
official row for the exact age in days, sex, and measure. The interpolation between table
rows is a source of error, and WHO publishes the tables at daily intervals for the first
period precisely to avoid it.

The code does not handle gestational-age correction beyond selecting a row, does not
compute BMI-for-age or length-for-age, and does not implement the CDC or WHO rules for
implausible values. The dose functions do not check the indication, the interval, the
route, the concentration of the formulation, or the renal and hepatic adjustment. They
are arithmetic aids, not a prescribing system.

## 8. Data, standards, and regulatory context

### 8.1 Growth references

The WHO Child Growth Standards describe growth for children from birth to 5 years and
were derived from a multi-country longitudinal sample of breastfed infants. The WHO
Growth Reference covers 5 to 19 years. Both are expressed as L, M, and S parameters by
age and sex, which is why a z-score can be computed for any measurement without a lookup
table of percentiles.

The CDC growth charts, revised in 2000, are the US clinical reference for children aged 2
years and older. CDC and the American Academy of Pediatrics recommend the WHO standards
for children under 2 years in the United States, because the CDC charts for that age were
derived from a predominantly formula-fed US sample and describe a different growth
pattern in infancy.

What this constrains: the reference name, the version, and the age range are part of the
measurement. A z-score without a reference name is not interpretable. What it does not
constrain: whether a given child is thriving, which is a clinical judgement that combines
growth, feeding, development, and examination.

### 8.2 Age definitions

Gestational age is the time since the first day of the last menstrual period, expressed
in weeks and days. Postmenstrual age is gestational age plus chronological age. Corrected
age is chronological age minus the number of weeks born before 40 weeks. ACOG Committee
Opinion 700 (2017) sets the dating hierarchy and the redating thresholds: a first-trimester
crown-rump length is the most accurate method, with a 5-day threshold before 9 weeks and a
7-day threshold between 9 and 15 weeks. The thresholds matter because a redating decision
changes every downstream age, and therefore every growth reference row.

### 8.3 Pediatric drug development and dosing standards

ICH E11(R1) (2017) sets the framework for pediatric drug development, including
extrapolation and modelling. The US Pediatric Research Equity Act and Best
Pharmaceuticals for Children Act create requirements and incentives for pediatric studies;
the EU Paediatric Regulation provides a parallel pathway. Off-label use remains common,
and many doses in neonates are derived from pharmacokinetic modelling rather than from
labelled trials.

Weight-based dosing uses milligrams per kilogram, with caps at adult doses. Body-surface
area dosing uses square metres, commonly estimated by the Mosteller formula. Neonatal
clearance matures with postmenstrual age and organ function, so a neonate is not a scaled
child. A rule engine must therefore carry the age clock, the weight with its unit, the
cap with its basis, and a plausibility range, and it must fail closed rather than
defaulting.

### 8.4 Research protections

In the United States, research with pregnant women, fetuses, and neonates is governed by
45 CFR 46 Subpart B, and research with children by 45 CFR 46 Subpart D and 21 CFR 50
Subpart D. These rules require parental permission, child assent where appropriate, and
additional protections for fetuses and neonates. They are not administrative details;
they determine which data can exist, which in turn determines what can be studied.

### 8.5 Obstetric standards and endpoints

Intrapartum fetal monitoring terminology follows FIGO guidance, which defines baseline
rate, variability, accelerations, and decelerations, and the categories that combine
them. Interpretation agreement between clinicians is modest, which makes the reference
standard for any automated reading contested.

Preeclampsia is defined by hypertension after 20 weeks plus proteinuria or organ
dysfunction, with definitions updated over time. Preterm birth is birth before 37 weeks,
and the denominator for reporting is live births unless stated otherwise. Maternal
mortality is reported per 100,000 live births in US surveillance, and pregnancy-related
deaths extend to one year under some definitions, which changes the numerator and
therefore the rate.

### 8.6 Device regulation for pediatric and obstetric tools

A growth calculator that reports a z-score for clinical use is generally device software
unless it qualifies for an exemption. A fetal monitoring algorithm that claims to
interpret a tracing carries a higher evidentiary burden because the action it informs,
operative delivery, is irreversible. The regulatory class depends on the intended use,
the user, and the claim, not on whether the model is "AI".

## 9. Industry practice and case evidence

### 9.1 Pediatric sepsis criteria

The Phoenix criteria (Schlapbach et al., JAMA, 2024) replaced earlier pediatric sepsis
definitions with a consensus based on suspected infection plus organ dysfunction, derived
and validated across multiple datasets. Documented fact: the criteria were developed
through an international consensus process and validated in several cohorts. Inference:
adult Sepsis-3 criteria do not transfer to children, because the physiology, the
measurement ranges, and the outcomes differ. What did not transfer: the adult thresholds,
and the assumption that a definition validated in adults is a definition for children.

### 9.2 Retinopathy of prematurity screening

Brown et al. (JAMA Ophthalmology, 2018) reported a deep convolutional network that
diagnosed plus disease from retinal images with performance comparable to expert graders
in a research setting. Documented fact: the model performed at expert level on the study
images. Inference: image-based screening could support a severe shortage of ROP experts.
What did not transfer: image quality, camera type, infant characteristics, and the fact
that the model was not a substitute for the full screening examination.

### 9.3 Computer-aided fetal monitoring

The INFANT trial (Brocklehurst et al., Lancet, 2017) randomised a large UK cohort to
computer-aided cardiotocography interpretation or standard interpretation and found no
improvement in poor neonatal outcomes. Documented fact: the decision aid did not improve
the primary outcome. Inference: a technically plausible aid can fail when the clinical
pathway and the reference standard are weak. What did not transfer: the assumption that
better detection of abnormal patterns improves outcomes, when the response pathway is the
binding constraint.

### 9.4 Aspirin for preterm preeclampsia prevention

ASPRE (Rolnik et al., New England Journal of Medicine, 2017) randomised women identified as
high risk by a first-trimester screening algorithm to aspirin or placebo and reported a
reduction in preterm preeclampsia. Documented fact: aspirin reduced preterm preeclampsia
in the screened high-risk group. Inference: a multivariable risk model plus a cheap
intervention can change outcomes when the intervention is effective and the population is
well selected. What did not transfer: the result is specific to the screening algorithm,
the population, adherence, and the preterm endpoint.

### 9.5 Placental growth factor testing

NICE diagnostic guidance DG49 recommended placental growth factor-based testing to help
rule out preeclampsia in women presenting with suspected disease. Documented fact: the
guidance recommends the test as an aid to rule out, within a defined pathway. Inference: a
biomarker can shorten a diagnostic pathway without being a standalone diagnosis. What did
not transfer: thresholds and assay platforms differ, and the test is not a substitute for
clinical assessment.

### 9.6 The common pattern

In each case, the measurement problem was upstream of the model. The pediatric sepsis case
was a definition transfer problem. The ROP case was an image-acquisition problem. The
INFANT case was a workflow problem. The ASPRE case was a screening-algorithm and adherence
problem. The PlGF case was a threshold and assay problem. None of them was solved by a
better architecture.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Chronological age used for a preterm infant | Weight percentile far below the plotted curve | Term reference applied before term age | False growth-failure diagnosis | Use postmenstrual or corrected age |
| Reference not named | z-scores differ between two reports for one child | WHO, CDC, and local charts differ | Inconsistent classification | Store reference name and version |
| Gestational age from unreliable dating | Redating changes the age by more than a week | Last menstrual period versus ultrasound | Every downstream age is wrong | Record the dating method |
| Weight unit error | Dose differs by a factor of 1,000 | Grams stored as kilograms | Tenfold or thousandfold overdose | Unit field, range check, guard |
| Cap recomputed from wrong weight | Cap does not fire on an implausible dose | Cap and dose share the same erroneous input | Overdose passes validation | Independent plausibility range |
| BSA from wrong weight | Dose scales with the square root of the error | Height and weight both enter the formula | Threefold error in a neonate | Cross-check BSA against age range |
| Adult reference range applied to a child | Lab flag rates are implausible | Reference intervals are age-specific | Missed or false diagnosis | Age-specific intervals |
| Rare event evaluated by AUROC | Confidence interval spans 0.5 to 0.9 | Too few events | Unstable conclusion | AUPRC, sensitivity, intervals |
| Denominator confusion | Maternal mortality rate differs by a factor of two | Pregnancies versus live births | Invalid comparison | State the denominator |
| Pregnancy physiology ignored | Lab values flagged as abnormal in normal pregnancy | Plasma volume expansion and dilution | Unnecessary investigation | Pregnancy-specific ranges |
| Fetal monitoring model without a pathway | Alert rate high, action rate low | No defined response to a tracing | No outcome effect | Design the response pathway |
| Adolescent confidentiality breach | Proxy access exposes sensitive data | Portal permissions not age-aware | Legal and trust harm | Age-aware access rules |
| Growth curve crossing ignored | One percentile is normal, two crossings are not | Single-point assessment | Missed failure to thrive | Serial measurements |
| Off-label dose treated as validated | Dose source is a handbook, not a label | Extrapolation from adults | Under- or overdose | State the evidence source |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Gestational age is accurately dated | All age clocks depend on it | Uncertain last menstrual period | Compare dating methods; record redating |
| The growth reference matches the population | z-scores are relative to a reference | Different ancestry, feeding, or altitude | Compare with local charts |
| Weight is measured, not estimated | Dosing and growth depend on it | Bed scale, conversion, transcription | Range check and repeat measurement |
| The weight unit is known | Dose arithmetic requires it | Grams versus kilograms | Unit field and plausibility guard |
| The cap is independent of the dose input | Caps must catch input errors | Cap recomputed from the same weight | Independent range check |
| Age-specific lab intervals are used | Pediatric values differ from adult | Adult reference applied | Compare flag rates by age |
| Rare events need interval estimates | Small counts are unstable | AUROC reported alone | Report AUPRC and intervals |
| Pregnancy physiology is accounted for | Reference ranges shift | Adult thresholds applied | Use pregnancy-specific ranges |
| The denominator matches the question | Rates require the right base | Pregnancies versus live births | State and reconcile |
| The response pathway exists | A prediction needs an action | No protocol for the alert | Workflow mapping |

The largest limit is that pediatric and obstetric data are thin. Rare outcomes, small
samples, and protections on research with pregnant women and children mean that many
models are developed on a few hundred events and transported without a target evaluation.
The correct response is not to abandon the models but to report intervals honestly, to
prefer sensitivity and decision curves over AUROC, and to treat every deployment as a
pilot with a pre-specified review.

A second limit is that the growth reference is a description of a population, not a norm
for an individual. A child at the 5th percentile may be entirely healthy, and a child at
the 50th percentile may be failing. The percentile is a screening input, not a diagnosis,
and the clinical decision combines it with the trajectory, the examination, and the
feeding history.

## 12. Exercises

### Exercise 1 — Convert the clocks (application, 25 minutes)

**Deliverable.** For a supplied infant, compute gestational age at birth, chronological
age, postmenstrual age, and corrected age at three follow-up visits.

**Acceptance criteria.** Each age is labelled with its definition and units; the
arithmetic is shown; the visit at which correction should stop is stated with a reason.

### Exercise 2 — Compute a z-score (application, 35 minutes)

**Deliverable.** LMS z-scores and percentiles for weight at two ages, using two supplied
reference rows, with a comparison.

**Acceptance criteria.** The LMS formula is applied correctly; the percentile is computed
from the normal distribution; the write-up explains why the two rows disagree.

### Exercise 3 — Dose check (application, 30 minutes)

**Deliverable.** A dose table for three drugs (mg/kg, mg/kg/day, mg/m²) with caps, for a
supplied weight and height, plus one deliberately corrupted weight.

**Acceptance criteria.** Caps are stated with their basis; the corrupted weight is caught
by at least two independent checks; the write-up explains why a cap alone is insufficient.

### Exercise 4 — Denominator audit (critique, 30 minutes)

**Deliverable.** A table of five perinatal indicators with the correct denominator, the
common wrong denominator, and the direction of bias.

**Acceptance criteria.** At least one indicator uses live births, one uses pregnancies,
one uses women delivered, and one uses neonatal admissions; each bias direction is
justified.

### Exercise 5 — Rare-event evaluation plan (design, 45 minutes)

**Deliverable.** An evaluation plan for a model predicting a 2 percent outcome in a
neonatal cohort.

**Acceptance criteria.** The plan reports AUPRC, sensitivity at fixed alert volume,
calibration, and confidence intervals; it states the events per variable; it includes a
rule-based baseline.

### Exercise 6 — Maternal-fetal confounding (critique, 40 minutes)

**Deliverable.** A causal diagram and a short memo on a question that links a maternal
exposure to a fetal outcome.

**Acceptance criteria.** The diagram distinguishes maternal, fetal, and delivery factors;
the memo names two sources of confounding and one collider; it states what the data
cannot identify.

### Exercise 7 — Growth chart critique (critique, 30 minutes)

**Deliverable.** A review of a supplied growth chart with a flagged percentile, identifying
the reference, the age clock, and whether the flag is justified.

**Acceptance criteria.** The review names the reference and version; it checks the age
clock; it distinguishes a single low point from a crossing trajectory.

### Exercise 8 — Consent and data design (design, 35 minutes)

**Deliverable.** A data-governance note for a pediatric or obstetric registry, covering
permission, assent, and re-use.

**Acceptance criteria.** The note cites the applicable subpart; it distinguishes parental
permission from assent; it states the re-use conditions and the adolescent
confidentiality rule.

## 13. Self-check answers

**1. Postmenstrual age.** For an infant born at 34 weeks 2 days and now 10 days old:
gestational age at birth is 240 days, chronological age is 10 days, postmenstrual age is
250 days, or 35 weeks 5 days. Corrected age is 10 minus 40, which is −30 days, meaning the
infant is 30 days short of term.

**2. LMS z-score.** With L = −0.15, M = 2.40, S = 0.17, and x = 2.04:
z = ((2.04/2.40)^−0.15 − 1) / (−0.15 × 0.17) = (1.02468 − 1) / (−0.0255) = −0.97,
which is the 16.7th percentile. The same weight against a term row with M = 3.20 gives
z = −3.31, the 0.05th percentile.

**3. Dose with a cap.** At 2.04 kg and 5 mg/kg/day, the raw dose is 10.2 mg/day. The cap of
10 mg/kg/day is 20.4 mg/day, so the cap does not fire. With the weight keyed as 20.4 kg,
the raw dose is 102 mg/day and the cap fires at 20.4 mg, but only because the cap is
stored in milligrams.

**4. Denominators.** Maternal mortality is reported per 100,000 live births. Preterm birth
is reported as a proportion of live births. Stillbirth is reported per 1,000 total births.
Neonatal mortality is reported per 1,000 live births. Using pregnancies as the denominator
for maternal mortality raises the rate, because pregnancies exceed live births.

**5. Rare-event evaluation.** AUROC is unstable with few events and can look acceptable
while sensitivity at a usable alert volume is poor. Report AUPRC, sensitivity at the
planned alert volume, calibration, and confidence intervals, and compare with a simple
rule.

**6. Maternal-fetal confounding.** Maternal age, socioeconomic position, and access to care
affect both the exposure and the outcome. Conditioning on a delivery complication that is
caused by the exposure can introduce collider bias. The data cannot identify the
counterfactual fetal outcome under a different maternal exposure.

**7. Growth chart review.** Check the reference name and version, the age clock (corrected
or chronological), the measurement position and equipment, and the trajectory. A single
point below the 5th percentile is not failure to thrive; a crossing of two major
percentile lines is concerning.

**8. Consent.** Research with pregnant women, fetuses, and neonates falls under 45 CFR 46
Subpart B; research with children falls under Subpart D. Parental permission is required,
and assent is required where the child can provide it. Re-use must be within the scope of
the original permission.

**Rubric for Exercises 5, 6, and 8.** A strong answer states the assumption, shows the
arithmetic or the citation, and names what the analysis cannot establish. A weak answer
asserts a conclusion without the mechanism or the denominator.

## 14. Interview questions

**1. Why is weight-based dosing a data problem, not only a clinical one?**
Strong answer: units, decimal errors, weight entry, caps, and independent plausibility
checks, with a fail-closed design. Weak answer: "the pharmacist checks it".

**2. What is the difference between gestational, postmenstrual, and chronological age?**
Strong answer: age since conception, age since birth plus gestational age, and age since
birth, each with a different reference. Weak answer: treats them as interchangeable.

**3. Why do adult sepsis criteria fail in children?**
Strong answer: different physiology, different definitions, and different outcome rates;
pediatric criteria were developed separately. Weak answer: "children are smaller".

**4. How do you evaluate a model for a 2 percent outcome?**
Strong answer: AUPRC, sensitivity at fixed alert volume, calibration, and intervals, with
a rule-based baseline. Weak answer: "AUROC above 0.8".

**5. What changes in pregnancy physiology that affects lab interpretation?**
Strong answer: plasma volume expansion, dilutional anemia, increased glomerular
filtration, and lower blood pressure, with pregnancy-specific ranges. Weak answer: "labs
are different".

**6. Why did the INFANT trial fail to show benefit?**
Strong answer: the decision aid did not improve the response pathway; interpretation and
action, not detection alone, determine outcomes. Weak answer: "the algorithm was bad".

**7. How should maternal mortality be denominated?**
Strong answer: per 100,000 live births for standard reporting, with the surveillance
system stated and the pregnancy-related window defined. Weak answer: per population.

**8. What is the risk of using adult reference ranges for children?**
Strong answer: misclassification of normal variation and missed or false diagnoses,
because reference intervals are age-specific. Weak answer: "the values are similar".

**9. How do you handle a pregnant patient in a trial?**
Strong answer: Subpart B protections, risk-benefit assessment, and inclusion rather than
blanket exclusion where appropriate. Weak answer: "exclude pregnant women".

**10. Which equity issues are specific to obstetrics?**
Strong answer: access to maternity care, Medicaid coverage, language, racism, and rural
closures, which are confounders rather than biology. Weak answer: "none, the model is
clinical".

## 15. Key terms

- **Gestational age** — time since the first day of the last menstrual period.
- **Chronological age** — time since birth.
- **Postmenstrual age** — gestational age plus chronological age.
- **Corrected age** — chronological age minus weeks born before 40 weeks.
- **LMS method** — z-score method using skewness, median, and coefficient of variation.
- **z-score** — standard deviations from the reference median.
- **Percentile** — proportion of the reference population below a value.
- **WHO Child Growth Standards** — reference for ages 0 to 5 years.
- **WHO Growth Reference** — reference for ages 5 to 19 years.
- **CDC growth charts** — US clinical reference for ages 2 years and older.
- **Small for gestational age** — birth weight below the 10th percentile for age.
- **Large for gestational age** — birth weight above the 90th percentile for age.
- **Mosteller formula** — body surface area from height and weight.
- **Body surface area** — dosing basis in square metres.
- **Maximum dose cap** — upper limit on a dose, usually an adult dose.
- **Tenfold error** — decimal shift producing a tenfold dose error.
- **Neonate** — infant in the first 28 days of life.
- **Preterm birth** — birth before 37 completed weeks.
- **Postterm birth** — birth at or after 42 weeks.
- **Preeclampsia** — hypertension after 20 weeks with proteinuria or organ dysfunction.
- **CTG** — cardiotocography, fetal heart rate and uterine contractions.
- **FIGO classification** — standard intrapartum fetal heart rate terminology.
- **Umbilical artery pH** — cord gas measure of fetal acid-base status.
- **Phoenix criteria** — pediatric sepsis consensus criteria, 2024.
- **Subpart B** — US research protections for pregnant women, fetuses, and neonates.
- **Subpart D** — US research protections for children.
- **Assent** — child's agreement to research participation.

## 16. Sources

- World Health Organization. Child growth standards. https://www.who.int/tools/child-growth-standards (accessed 26 October 2026).
- Centers for Disease Control and Prevention. Growth charts. https://www.cdc.gov/growthcharts/ (accessed 26 October 2026).
- American College of Obstetricians and Gynecologists. Committee Opinion 700: Methods for estimating the due date (2017). https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2017/05/methods-for-estimating-the-due-date (accessed 26 October 2026).
- US Department of Health and Human Services. 45 CFR 46 Subpart B. https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/common-rule-subpart-b/index.html (accessed 26 October 2026).
- US Code of Federal Regulations. 21 CFR 50 Subpart D. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-50/subpart-D (accessed 26 October 2026).
- US Food and Drug Administration. Pediatric Research Equity Act. https://www.fda.gov/drugs/development-resources/pediatric-research-equity-act-prea (accessed 26 October 2026).
- International Council for Harmonisation. ICH E11(R1) addendum on pediatric drug development. https://www.ich.org/page/efficacy-guidelines (accessed 26 October 2026).
- Schlapbach LJ, et al. International consensus criteria for pediatric sepsis and septic shock. JAMA. 2024. https://doi.org/10.1001/jama.2024.0179 (accessed 26 October 2026).
- Brocklehurst P, et al. Computerised interpretation of fetal heart rate during labour (INFANT): a randomised controlled trial. Lancet. 2017. https://doi.org/10.1016/S0140-6736(17)30568-8 (accessed 26 October 2026).
- Rolnik DL, et al. Aspirin versus placebo in pregnancies at high risk for preterm preeclampsia. N Engl J Med. 2017. https://doi.org/10.1056/NEJMoa1704559 (accessed 26 October 2026).
- Brown JM, et al. Automated diagnosis of plus disease in retinopathy of prematurity using deep convolutional neural networks. JAMA Ophthalmol. 2018. https://doi.org/10.1001/jamaophthalmol.2018.1934 (accessed 26 October 2026).
- National Institute for Health and Care Excellence. DG49: PlGF-based testing to help diagnose suspected preterm pre-eclampsia. https://www.nice.org.uk/guidance/dg49 (accessed 26 October 2026).
- PhysioNet. CTU-UHB CTG database. https://physionet.org/content/ctu-uhb-ctgdb/ (accessed 26 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Previous topic: [`04_critical-care.md`](04_critical-care.md)
- Next topic: [`06_public-health.md`](06_public-health.md)
- Measurement comparison: [`07_cross-specialty-comparison.md`](07_cross-specialty-comparison.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Foundations: [`../01A_foundations/`](../01A_foundations/)
