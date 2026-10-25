# Pharmacology Basics

> Module 01 — Clinical Foundations for Data Scientists
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03
> Version 2026.3

---

## 1. Why this topic exists

Medication data are the most abundant treatment data in the health system and the easiest to
misread. An order is not a dose given. A dose given is not a dose absorbed. A dose absorbed is
not a concentration at the target organ. A concentration is not an effect. Each step is a
different variable, and each has a different error structure.

Data scientists encounter four recurring problems.

**Problem 1: exposure is measured at the wrong layer.** A medication order records intent. A
medication administration record (MAR) records what a nurse documented as given. A pharmacy claim
records what was dispensed. A patient's actual ingestion is unobserved. A model that uses orders
as exposure will misclassify patients who never received the drug, and a model that uses claims
will misclassify patients who filled the prescription but did not take it.

**Problem 2: dose is not a number without a body size.** A dose of 1 mg/kg means different
things in a 4 kg neonate and a 140 kg adult. A dose of 500 mg in a patient with a creatinine
clearance of 20 mL/min may be an overdose. Dose normalization is a clinical judgment, not a
division.

**Problem 3: drug effects are time-dependent.** A drug started on day 1 may not reach steady
state until day 5. A drug stopped on day 3 may remain active for days. A feature that records
"on drug" as a binary ignores the concentration curve, the indication, and the duration.

**Problem 4: interactions and genetics are invisible in claims.** Two drugs can interact without
either order being wrong. A patient can have a genetic variant that makes a standard dose
ineffective or toxic. Neither fact appears in a claims table unless someone coded it.

This topic gives you the pharmacokinetic and pharmacodynamic vocabulary to treat medication data
as an exposure with a time course, a dose, a route, and a patient-specific clearance.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** order, dispensing, administration, and adherence as different exposure
   variables.
2. **Explain** absorption, distribution, metabolism, and excretion (ADME) and the parameters
   that describe each.
3. **Compute** volume of distribution, clearance, half-life, loading dose, maintenance dose, and
   steady-state concentration for a one-compartment drug.
4. **Estimate** creatinine clearance with the Cockcroft-Gault equation and **explain** why it
   differs from estimated glomerular filtration rate (eGFR).
5. **Identify** common cytochrome P450 (CYP) interactions and the difference between
   pharmacokinetic and pharmacodynamic interactions.
6. **Critique** a medication-based feature for indication bias, immortal time, and
   dose-normalization error.
7. **Design** a medication exposure table with drug, dose, route, time, and provenance.

## 3. The core mental model

Pharmacology has two halves. Pharmacokinetics is what the body does to the drug. Pharmacodynamics
is what the drug does to the body. Data science usually observes the dose and the outcome and
infers the middle.

```text
   DOSE GIVEN
       |
       v
   ABSORPTION ---> bioavailability (F), first-pass metabolism
       |
       v
   DISTRIBUTION -> volume of distribution (Vd), protein binding
       |
       v
   CONCENTRATION AT EFFECT SITE
       |
       +-----------------------+
       |                       |
       v                       v
   METABOLISM              RECEPTOR / TARGET
   (CYP, phase II)              |
       |                       v
       v                   EFFECT (benefit, toxicity)
   EXCRETION                   |
   (renal, biliary)            v
       |                   ADAPTATION / TOLERANCE
       v
   ELIMINATION
```

The central equation for a one-compartment model with first-order elimination is:

```text
   C(t) = C0 x e^(-k x t)
   k = 0.693 / t1/2
   t1/2 = 0.693 x Vd / CL
```

At steady state, the average concentration is:

```text
   Css_avg = (F x dose rate) / CL
```

And the loading dose is:

```text
   Loading dose = (Vd x target concentration) / F
```

These four equations explain most of clinical dosing. The difficulty is not the algebra; it is
estimating Vd, CL, and F in a specific patient.

### 3.1 The exposure ladder

```text
   prescribed -> dispensed -> administered -> absorbed -> at target -> effect
       |            |             |             |            |          |
     order        claim          MAR          PK model     PD model   outcome
```

Each rung is a different variable. Most analytic errors come from using a higher rung as a proxy
for a lower one without stating the assumption.

### 3.2 The therapeutic window

```text
   effect
     ^
     |            ___________
     |           /           \
     |          /             \
     |         /               \
     |        /                 \
     +-------+---------+---------+------> concentration
          subtherapeutic  therapeutic  toxic
```

The width of the plateau is the therapeutic index. Drugs with a narrow index — warfarin,
digoxin, lithium, phenytoin, tacrolimus, gentamicin, levothyroxine — require monitoring and are
the most sensitive to dosing error. Drugs with a wide index are more forgiving but not immune to
interactions.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Pharmacokinetics (PK) | What the body does to the drug | Confused with pharmacodynamics | — |
| Pharmacodynamics (PD) | What the drug does to the body | Confused with efficacy | — |
| Absorption | Movement from the site of administration into blood | Assumed complete for oral drugs | Rate and extent |
| Bioavailability (F) | Fraction of the dose reaching systemic circulation | Assumed 100% for all routes | Proportion 0-1 |
| First-pass metabolism | Hepatic or intestinal metabolism before systemic circulation | Ignored for oral drugs | Proportion |
| Distribution | Movement from blood to tissues | Confused with absorption | — |
| Volume of distribution (Vd) | Apparent volume relating dose to concentration | Interpreted as a real anatomical volume | L or L/kg |
| Protein binding | Fraction bound to albumin or other proteins | Assumed constant across disease states | Proportion |
| Clearance (CL) | Volume of plasma cleared of drug per time | Confused with elimination rate | L/h or mL/min |
| Half-life (t1/2) | Time for concentration to halve | Assumed to determine onset | Hours |
| Steady state | State in which input equals output | Assumed immediate | Time |
| Cmax | Maximum concentration after a dose | Confused with average concentration | mg/L |
| Cmin / trough | Minimum concentration before the next dose | Confused with Cmax | mg/L |
| AUC | Area under the concentration-time curve | Confused with a single level | mg·h/L |
| MIC | Minimum inhibitory concentration | Confused with a dose | mg/L |
| Loading dose | Dose to reach target concentration quickly | Omitted when time matters | mg |
| Maintenance dose | Dose to maintain target at steady state | Confused with loading dose | mg per interval |
| Therapeutic index | Ratio of toxic to effective exposure | Confused with therapeutic range | Dimensionless |
| Therapeutic drug monitoring (TDM) | Measurement of drug concentration to guide dosing | Used for all drugs | mg/L |
| Agonist | Drug that activates a receptor | Confused with antagonist | — |
| Antagonist | Drug that blocks a receptor | Confused with inverse agonist | — |
| Emax | Maximum effect | Confused with efficacy at a given dose | Effect units |
| EC50 | Concentration producing 50% of Emax | Confused with potency in vivo | mg/L |
| Potency | Amount of drug needed for a given effect | Confused with efficacy | mg |
| Efficacy | Maximum effect achievable | Confused with potency | Effect units |
| Tolerance | Reduced effect with repeated exposure | Confused with resistance | — |
| Adverse drug reaction (ADR) | Harm caused by a drug at a therapeutic dose | Confused with medication error | — |
| Medication error | Preventable error in prescribing or administration | Confused with ADR | — |
| Drug-drug interaction (DDI) | Effect of one drug on another | Confused with ADR | — |
| CYP450 | Cytochrome P450 enzyme family | Treated as a single enzyme | Enzyme family |
| Induction | Increased enzyme expression | Confused with inhibition | — |
| Inhibition | Reduced enzyme activity | Confused with induction | — |
| Prodrug | Inactive drug requiring metabolism for activity | Confused with active metabolite | — |
| Active metabolite | Metabolite with pharmacological activity | Ignored in exposure models | — |
| Renal clearance | Clearance by glomerular filtration and secretion | Confused with total clearance | mL/min |
| Hepatic clearance | Clearance by hepatic metabolism and biliary excretion | Confused with renal clearance | L/h |
| Cockcroft-Gault | Creatinine clearance estimation equation | Confused with eGFR | mL/min |
| eGFR | Estimated glomerular filtration rate | Used for drug dosing without adjustment | mL/min/1.73 m² |
| Body surface area (BSA) | Estimated skin area | Used for all drugs | m² |
| Allometric scaling | Dose scaling by metabolic size | Confused with linear mg/kg | — |
| Adherence | Extent to which a patient takes the prescribed regimen | Confused with persistence | Proportion |
| Persistence | Duration of continued treatment | Confused with adherence | Time |
| Primary non-adherence | Prescription never filled | Invisible in claims | Proportion |
| Proportion of days covered (PDC) | Days covered divided by days in period | Confused with adherence | Proportion |
| Medication possession ratio (MPR) | Days supply divided by refill interval | Can exceed 1 | Proportion |
| RxNorm | Normalized drug nomenclature | Confused with NDC | Coding system |
| NDC | National Drug Code | Confused with RxNorm | Coding system |
| ATC | Anatomical Therapeutic Chemical classification | Confused with a clinical class | Coding system |

Three definitions are worth expanding.

**Vd is apparent, not anatomical.** It is the volume that would be required to produce the
observed concentration if the drug were evenly distributed. A drug with a Vd of 200 L in a 70 kg
patient is not in a 200 L compartment; it is concentrated in tissue, and the plasma
concentration is low. A large Vd means that dialysis removes little drug and that loading doses
are large.

**Clearance is the parameter that determines maintenance dose.** Half-life is convenient for
describing time to steady state, but it is a derived quantity. Two patients can have the same
half-life with very different clearances if their volumes differ. Maintenance dosing is driven by
clearance and bioavailability.

**Adherence is not a single behavior.** A patient can be adherent in the sense of filling
prescriptions and non-adherent in the sense of taking them. Claims measure filling. Pill counts
measure possession. Electronic monitors measure opening. None measures ingestion.

## 5. The mechanics, step by step

**Step 1. Define the exposure variable.**

- Input: a clinical question.
- Operation: decide whether the exposure is prescribing, dispensing, administration, or
  exposure at the target.
- Output: an explicit exposure definition.
- Reasoning: each layer answers a different question. "Was the drug prescribed?" is a
  prescribing question. "Was it given?" is an administration question. "Was the patient
  exposed?" requires an assumption about absorption and adherence.
- Branch point: if the study is observational, the exposure layer determines the confounding
  structure. Prescribing is confounded by indication; administration is confounded by nursing
  workflow and patient refusal.
- Analytic artifact: `exposure_layer`, `exposure_start`, `exposure_end`.

**Step 2. Normalize the drug identity.**

- Input: order text, NDC, local formulary code.
- Operation: map to RxNorm ingredient, strength, and dose form; keep the original.
- Output: a normalized drug identifier.
- Reasoning: the same drug appears under brand names, generic names, combination products, and
  different salts. "Metoprolol tartrate" and "metoprolol succinate" are not interchangeable for
  dosing frequency. RxNorm provides ingredient-level and clinical-drug-level normalization.
- Branch point: combination products require splitting into components or treating as a single
  exposure, and the choice changes the analysis.
- Analytic artifact: `rxnorm_cui`, `ingredient`, `strength`, `dose_form`, `brand_flag`.

**Step 3. Capture dose, route, and frequency.**

- Input: order or MAR.
- Operation: parse dose, unit, route, frequency, and PRN status.
- Output: structured dose fields.
- Reasoning: "1 tab" is not a dose. "BID" is not a rate. "PRN" means the drug was available but
  not necessarily given. A dose of "10 mg/kg" requires a weight.
- Branch point: sliding-scale insulin and titrated infusions have no fixed dose; they require a
  time-series representation.
- Analytic artifact: `dose_value`, `dose_unit`, `route`, `frequency`, `prn_flag`, `weight_kg`.

**Step 4. Normalize the dose to body size.**

- Input: dose and patient size.
- Operation: divide by weight, BSA, or allometric weight.
- Output: a normalized dose.
- Reasoning: mg/kg is standard for children and many adults; BSA is standard for chemotherapy;
  fixed doses are standard for many oral drugs. The choice should follow the drug's label and
  pharmacology, not convenience.
- Branch point: obesity. Adjusted body weight or ideal body weight may be more appropriate for
  some drugs, and the choice changes the dose by a large factor.
- Analytic artifact: `dose_mg_per_kg`, `weight_type`, `bsa_m2`.

**Step 5. Assess renal function.**

- Input: age, sex, weight, serum creatinine.
- Operation: compute Cockcroft-Gault creatinine clearance for drug dosing; compute eGFR for
  kidney staging.
- Output: CrCl in mL/min and eGFR in mL/min/1.73 m².
- Reasoning: Cockcroft-Gault estimates creatinine clearance and is the equation used in most
  drug labels. eGFR is normalized to body surface area and is used for chronic kidney disease
  staging. They are not interchangeable for dosing.
- Branch point: in patients with unstable creatinine, low muscle mass, or amputation,
  Cockcroft-Gault is unreliable; measured clearance or alternative markers may be needed.
- Analytic artifact: `crcl_ml_min`, `egfr_ml_min_1_73`, `equation_version`.

**Step 6. Assess hepatic function.**

- Input: liver enzymes, bilirubin, albumin, INR, ascites, encephalopathy.
- Operation: classify with Child-Pugh or Model for End-Stage Liver Disease (MELD).
- Output: a hepatic impairment category.
- Reasoning: hepatic clearance depends on liver blood flow, enzyme activity, and protein
  binding. Child-Pugh and MELD predict prognosis and are used for some dose adjustments, but they
  do not measure the activity of a specific enzyme.
- Branch point: a patient with normal transaminases can still have impaired clearance if blood
  flow is low or if a CYP enzyme is inhibited.
- Analytic artifact: `child_pugh_class`, `meld_score`.

**Step 7. Choose the pharmacokinetic model.**

- Input: drug, route, available concentrations.
- Operation: one-compartment versus two-compartment; linear versus nonlinear; population model.
- Output: a model with parameters and assumptions.
- Reasoning: most drugs are approximately linear over the therapeutic range. Phenytoin, ethanol,
  and high-dose salicylates are nonlinear because their metabolizing enzymes saturate. A
  one-compartment model is adequate for many clinical decisions but not for immediate
  post-infusion concentrations of a rapidly distributed drug.
- Branch point: if concentrations are available, fit a model; if not, use population estimates
  and label them.
- Analytic artifact: `model_type`, `parameter_source`.

**Step 8. Estimate volume of distribution.**

- Input: drug, weight, disease state.
- Operation: use a population value (L/kg) or a patient-specific estimate.
- Output: Vd in L.
- Reasoning: Vd depends on lipophilicity, protein binding, and tissue binding. It increases in
  obesity for lipophilic drugs and in edema for hydrophilic drugs. In critical illness, capillary
  leak can increase Vd for many drugs.
- Branch point: a large Vd means a large loading dose and poor dialysis clearance.
- Analytic artifact: `vd_l`, `vd_l_per_kg`, `vd_source`.

**Step 9. Estimate clearance.**

- Input: renal function, hepatic function, drug properties.
- Operation: use a population equation or a measured clearance.
- Output: CL in L/h.
- Reasoning: for renally cleared drugs, clearance scales with creatinine clearance. For
  hepatically cleared drugs, clearance is harder to predict and depends on enzyme activity and
  blood flow.
- Branch point: if the patient is on renal replacement therapy, clearance depends on the
  modality, filter, and flow rate.
- Analytic artifact: `cl_l_h`, `cl_source`.

**Step 10. Compute half-life.**

- Input: Vd and CL.
- Operation: t1/2 = 0.693 x Vd / CL.
- Output: hours.
- Reasoning: half-life determines time to steady state (about 4-5 half-lives) and the duration of
  effect after stopping. It is a derived parameter, not an independent property.
- Branch point: in renal failure, half-life lengthens; in enzyme induction, it shortens.
- Analytic artifact: `half_life_h`.

**Step 11. Compute the loading dose.**

- Input: Vd, target concentration, bioavailability.
- Operation: loading dose = (Vd x target concentration) / F.
- Output: mg.
- Reasoning: the loading dose fills the volume of distribution. It does not depend on clearance.
  For intravenous drugs, F is 1. For oral drugs, F is less than 1 and the loading dose must be
  increased accordingly.
- Branch point: a loading dose can cause toxicity if the target is based on a population Vd that
  is too small for the patient.
- Analytic artifact: `loading_dose_mg`, `target_concentration`.

**Step 12. Compute the maintenance dose.**

- Input: CL, target average concentration, bioavailability, dosing interval.
- Operation: dose rate = (CL x target concentration) / F; dose = dose rate x interval.
- Output: mg per interval.
- Reasoning: maintenance dosing replaces what is eliminated. It depends on clearance, not on Vd.
  At steady state, average concentration is dose rate divided by clearance.
- Branch point: for drugs with a narrow therapeutic index, the interval matters as much as the
  total daily dose, because peak and trough concentrations differ.
- Analytic artifact: `maintenance_dose_mg`, `interval_h`.

**Step 13. Estimate time to steady state.**

- Input: half-life.
- Operation: steady state is approximately 4-5 half-lives.
- Output: hours or days.
- Reasoning: after 4 half-lives, approximately 94% of steady state is reached; after 5,
  approximately 97%. A drug with a 24-hour half-life takes 4-5 days.
- Branch point: if the clinical situation requires immediate effect, give a loading dose.
- Analytic artifact: `time_to_steady_state_h`.

**Step 14. Interpret a drug concentration.**

- Input: concentration, time relative to dose, dose history, assay.
- Operation: compare with the therapeutic range; assess timing and steady state.
- Output: an interpretation with a recommendation.
- Reasoning: a concentration without a time is uninterpretable. A trough drawn too early is
  falsely high; a peak drawn too late is falsely low. The assay matters: immunoassays can cross-
  react with metabolites.
- Branch point: for vancomycin, the guideline target is an AUC/MIC ratio of 400-600 mg·h/L, not
  a trough; trough-only monitoring is no longer recommended for serious infections.
- Analytic artifact: `concentration`, `sample_ts`, `dose_ts`, `assay`, `interpretation`.

**Step 15. Assess pharmacokinetic interactions.**

- Input: concomitant drugs.
- Operation: identify effects on absorption, distribution, metabolism, or excretion.
- Output: a mechanism and an expected direction.
- Reasoning: CYP inhibition raises concentrations of substrates; induction lowers them. The
  effect is not instantaneous: inhibition is rapid, induction takes days to weeks. Transporters
  (P-glycoprotein, OATP1B1) also matter.
- Branch point: a drug can be both a substrate and an inhibitor, creating complex interactions.
- Analytic artifact: `interaction_mechanism`, `victim_drug`, `perpetrator_drug`, `expected_effect`.

**Step 16. Assess pharmacodynamic interactions.**

- Input: concomitant drugs with similar or opposing effects.
- Operation: identify additive toxicity or efficacy.
- Output: a mechanism and an expected effect.
- Reasoning: two drugs that prolong the QT interval can produce torsades de pointes even when
  neither is above its own threshold. Two sedatives can produce respiratory depression. The
  interaction is not mediated by concentration.
- Branch point: PD interactions are invisible in PK data and require clinical knowledge.
- Analytic artifact: `pd_interaction`, `effect`.

**Step 17. Assess pharmacogenomics.**

- Input: genotype if available.
- Operation: map to phenotype (poor, intermediate, normal, rapid metabolizer) and to guidance.
- Output: a dose or drug recommendation.
- Reasoning: CYP2C19 poor metabolizers have reduced activation of clopidogrel; CYP2D6 poor
  metabolizers have reduced conversion of codeine to morphine; HLA-B*57:01 carriers are at risk
  of abacavir hypersensitivity. Guidelines exist but are not universally implemented.
- Branch point: genotype is often missing; absence is not evidence of normal metabolism.
- Analytic artifact: `gene`, `genotype`, `phenotype`, `guideline_source`.

**Step 18. Assess adverse drug reactions.**

- Input: drug exposure and clinical event.
- Operation: assess temporality, plausibility, alternative causes, and dechallenge.
- Output: a causality assessment.
- Reasoning: an ADR is not the same as a medication error. An ADR can occur at a correct dose. A
  medication error is preventable. Both matter, but they require different interventions.
- Branch point: documentation of ADRs is incomplete, so absence in the allergy list is not
  evidence of absence.
- Analytic artifact: `adr_flag`, `causality`, `severity`.

**Step 19. Handle special populations.**

- Input: age, pregnancy, renal or hepatic impairment, obesity.
- Operation: adjust dose, choose alternative, or avoid.
- Output: a plan with monitoring.
- Reasoning: neonates have immature clearance; older adults have reduced renal function and
  altered body composition; pregnancy changes volume, clearance, and protein binding. Standard
  doses may be ineffective or toxic.
- Branch point: pregnancy labeling changed in 2015; the old letter categories are no longer in
  current labels.
- Analytic artifact: `population_flag`, `dose_adjustment`, `monitoring_plan`.

**Step 20. Measure adherence.**

- Input: pharmacy claims or fills.
- Operation: compute PDC or MPR over a defined window.
- Output: a proportion with a denominator.
- Reasoning: PDC is the proportion of days covered in a period; MPR is days supply divided by the
  refill interval and can exceed 1. Neither measures ingestion. Primary non-adherence — a
  prescription never filled — is invisible in a claims-based adherence measure unless the
  denominator includes the prescription.
- Branch point: switching drugs or changing dose can create artificial gaps.
- Analytic artifact: `pdc`, `mpr`, `window_days`, `switch_flag`.

**Step 21. Represent exposure as time series.**

- Input: orders, MAR, claims.
- Operation: build intervals of exposure with start, end, dose, and route.
- Output: one row per exposure interval.
- Reasoning: a binary "ever exposed" feature loses timing, dose, and duration. An interval
  representation supports time-varying covariates and avoids immortal time bias.
- Branch point: overlapping intervals from different sources must be reconciled.
- Analytic artifact: `exposure_id`, `start_ts`, `end_ts`, `dose`, `route`, `source`.

**Step 22. Validate against an independent source.**

- Input: exposure table.
- Operation: compare with chart review, MAR, or pharmacy records.
- Output: agreement statistics and error examples.
- Reasoning: no single source is complete. Orders omit administration; MAR omits take-home use;
  claims omit inpatient administration. Triangulation is the only way to estimate error.
- Branch point: if disagreement is systematic by site or drug, the exposure definition is not
  transportable.
- Analytic artifact: a validation report with agreement and error categories.

## 6. Worked example

A 68-year-old man is admitted with fever, chills, and a tunneled dialysis catheter exit-site
infection. Blood cultures grow methicillin-resistant *Staphylococcus aureus* (MRSA) in two of two
bottles. He weighs 70 kg, is 175 cm tall, and has a baseline serum creatinine of 1.1 mg/dL. On
admission his creatinine is 1.6 mg/dL. He takes lisinopril 20 mg daily, atorvastatin 40 mg
nightly, omeprazole 20 mg daily, and metoprolol succinate 50 mg daily. He has no documented drug
allergies.

The infectious diseases team recommends vancomycin for MRSA bacteremia. The pharmacy is asked to
design the initial regimen.

### 6.1 Step 1: renal function

Cockcroft-Gault creatinine clearance:

```text
CrCl = ((140 - age) x weight) / (72 x SCr) x 0.85 if female
     = ((140 - 68) x 70) / (72 x 1.6)
     = (72 x 70) / 115.2
     = 5040 / 115.2
     = 43.8 mL/min
```

The chronic kidney disease epidemiology collaboration (CKD-EPI) 2021 eGFR for the same patient:

```text
eGFR = 142 x min(SCr/0.9, 1)^-0.302 x max(SCr/0.9, 1)^-1.200 x 0.9938^68
     = 142 x 1 x (1.7778)^-1.200 x 0.655
     = 46.6 mL/min/1.73 m²
```

The two numbers are close in this case (43.8 versus 46.6), but they are not the same quantity.
Cockcroft-Gault estimates creatinine clearance in mL/min and is the equation used in most drug
labels. eGFR is normalized to 1.73 m² and is used for kidney staging. For dosing, use the
equation the label specifies and state which one you used.

> **Note:** The patient's baseline creatinine was 1.1 mg/dL. His admission creatinine of 1.6
> mg/dL may represent acute kidney injury on chronic kidney disease, or dehydration. A single
> value is not a baseline. If the creatinine is rising, the clearance estimate is already stale.

### 6.2 Step 2: volume of distribution and clearance

Vancomycin is a hydrophilic glycopeptide, eliminated largely unchanged by glomerular filtration.
A typical volume of distribution is approximately 0.7 L/kg in adults, though it ranges from about
0.4 to 1.0 L/kg depending on fluid status and body composition.

```text
Vd = 0.7 L/kg x 70 kg = 49 L
```

Clearance is estimated from creatinine clearance. A commonly used relationship is:

```text
CL (mL/min) = 0.689 x CrCl + 3.66
            = 0.689 x 43.8 + 3.66
            = 30.1 + 3.66
            = 33.8 mL/min
```

Convert to liters per hour:

```text
CL = 33.8 mL/min x 60 min/h / 1000 mL/L = 2.03 L/h
```

This is a population relationship, not a measurement. It assumes the patient's creatinine
clearance reflects vancomycin clearance, which is reasonable for vancomycin but not for drugs
with substantial hepatic metabolism.

### 6.3 Step 3: half-life and time to steady state

```text
t1/2 = 0.693 x Vd / CL
     = 0.693 x 49 / 2.03
     = 33.96 / 2.03
     = 16.7 h
```

Time to steady state is approximately 4-5 half-lives:

```text
4 x 16.7 = 66.8 h
5 x 16.7 = 83.5 h
```

Steady state is therefore about 3 to 3.5 days away. Without a loading dose, the patient would be
underdosed for the first two days of a bacteremia, which is precisely when bacterial killing
matters most.

### 6.4 Step 4: loading dose

The 2020 ASHP/IDSA/PIDS/SIDP vancomycin guideline recommends a loading dose of 25-30 mg/kg
actual body weight for serious MRSA infections in adults, with a maximum of 3,000 mg. A common
choice is 25 mg/kg:

```text
Loading dose = 25 mg/kg x 70 kg = 1,750 mg
```

The loading dose is capped at 3,000 mg, so 1,750 mg is acceptable. It is infused over at least
one hour to reduce infusion-related reactions.

### 6.5 Step 5: maintenance dose

The guideline target for serious MRSA infection is an area under the concentration-time curve
over 24 hours (AUC24) to minimum inhibitory concentration (MIC) ratio of 400-600 mg·h/L,
assuming a broth microdilution MIC of 1 mg/L. For a MIC of 1 mg/L, the AUC24 target is 400-600
mg·h/L.

At steady state, average concentration is dose rate divided by clearance:

```text
AUC24 = daily dose / CL
```

Solve for the daily dose that gives an AUC24 of 500 mg·h/L:

```text
Daily dose = 500 x 2.03 = 1,015 mg/day
```

A practical regimen is 500 mg every 12 hours, which is 1,000 mg/day:

```text
Predicted AUC24 = 1,000 / 2.03 = 493 mg·h/L
```

That is within the 400-600 mg·h/L target. The regimen is 500 mg intravenously every 12 hours,
with a 1,750 mg loading dose.

### 6.6 Step 6: what the concentration would look like

The average concentration at steady state is:

```text
Css_avg = AUC24 / 24 = 493 / 24 = 20.5 mg/L
```

The trough will be lower than the average and the peak higher. The exact values depend on the
infusion duration and the two-compartment behaviour of vancomycin, which is why the guideline
prefers AUC-guided dosing over trough-only monitoring. A trough of 15-20 mg/L was the old
surrogate target; it is no longer recommended as the primary target for serious infection
because it does not reliably reflect AUC and because higher troughs are associated with
nephrotoxicity.

### 6.7 Step 7: the renal function changes

On day 4, the patient's creatinine rises to 2.4 mg/dL and urine output falls to 20 mL/h. Recompute
clearance:

```text
CrCl = ((140 - 68) x 70) / (72 x 2.4)
     = 5040 / 172.8
     = 29.2 mL/min

CL = 0.689 x 29.2 + 3.66
   = 20.1 + 3.66
   = 23.8 mL/min
   = 1.43 L/h

t1/2 = 0.693 x 49 / 1.43 = 23.8 h
```

If the dose remains 500 mg every 12 hours:

```text
AUC24 = 1,000 / 1.43 = 701 mg·h/L
```

That is above the target range and into the toxicity-associated range. A revised regimen of 750
mg every 24 hours gives:

```text
AUC24 = 750 / 1.43 = 526 mg·h/L
```

That is within target. The example shows why renal function must be treated as a time-varying
covariate, not a baseline characteristic. A model that uses admission creatinine for the entire
admission will mis-dose the patient from day 4 onward.

### 6.8 Step 8: the exposure data trail

| Event | Timestamp | Source | What it records | What it does not record |
|---|---|---|---|---|
| Order | Day 1, 14:05 | Computerized provider order entry | Intent to give 1,750 mg | Whether it was given |
| Dispense | Day 1, 14:40 | Pharmacy system | Product prepared | Whether it reached the patient |
| Administration | Day 1, 15:10 | MAR | Nurse documented the infusion | Actual delivered dose |
| Level drawn | Day 2, 05:30 | Laboratory | Serum concentration | Whether steady state was reached |
| Result | Day 2, 08:15 | Laboratory | Value with units | Interpretation |
| Dose change | Day 4, 09:00 | Order | New regimen | Adherence after discharge |

A dataset built from orders would record one exposure starting at 14:05. A dataset built from the
MAR would record an administration at 15:10. A dataset built from claims would not see the
inpatient administration at all. Each is a different variable.

### 6.9 A second exposure: warfarin and fluconazole

The same patient later develops oral candidiasis and is started on fluconazole 200 mg daily. He
is also on warfarin 5 mg daily for atrial fibrillation, with a stable INR of 2.4.

Fluconazole inhibits CYP2C9, the enzyme that metabolizes the more potent S-enantiomer of
warfarin. The result is reduced warfarin clearance and a rise in INR. The interaction is
pharmacokinetic and begins within days.

| Day | Warfarin dose | Fluconazole | INR | Action |
|---|---|---:|---:|---|
| 0 | 5 mg daily | none | 2.4 | Baseline |
| 3 | 5 mg daily | 200 mg daily | 3.1 | Monitor |
| 5 | 5 mg daily | 200 mg daily | 4.8 | Hold one dose; reduce warfarin |
| 8 | 3 mg daily | 200 mg daily | 3.2 | Continue monitoring |
| 12 | 3 mg daily | 200 mg daily | 2.6 | Stable |

The numbers are illustrative of a typical trajectory, not a prediction for every patient. The
mechanism is documented; the magnitude varies with dose, genotype, diet, and adherence. The data
lesson is that an interaction is a time-varying exposure effect, not a binary flag.

## 7. Code, query, or calculation

The block below reproduces the case calculations and adds a PDC calculation.

```python
"""Pharmacokinetic calculations for vancomycin and a PDC adherence measure.

The clearance relationship CL = 0.689 x CrCl + 3.66 is a population estimate.
It is not a substitute for measured concentrations.
"""


def cockcroft_gault(age: int, weight_kg: float, scr_mg_dl: float,
                    female: bool = False) -> float:
    """Creatinine clearance in mL/min. SCr in mg/dL."""
    crcl = ((140 - age) * weight_kg) / (72 * scr_mg_dl)
    if female:
        crcl *= 0.85
    return crcl


def ckd_epi_2021(age: int, scr_mg_dl: float, female: bool = False) -> float:
    """eGFR in mL/min/1.73 m2, race-free 2021 equation."""
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = scr_mg_dl / kappa
    egfr = (142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200
            * 0.9938 ** age)
    if female:
        egfr *= 1.012
    return egfr


def vancomycin_pk(crcl_ml_min: float, weight_kg: float,
                  vd_l_per_kg: float = 0.7) -> dict:
    vd = vd_l_per_kg * weight_kg
    cl_ml_min = 0.689 * crcl_ml_min + 3.66
    cl_l_h = cl_ml_min * 0.06
    half_life = 0.693 * vd / cl_l_h
    return {"vd_l": vd, "cl_l_h": cl_l_h, "half_life_h": half_life}


age, weight, height = 68, 70.0, 175.0
scr = 1.6
crcl = cockcroft_gault(age, weight, scr)
egfr = ckd_epi_2021(age, scr)
pk = vancomycin_pk(crcl, weight)

print(f"CrCl: {crcl:.1f} mL/min")
print(f"eGFR: {egfr:.1f} mL/min/1.73 m2")
print(f"Vd: {pk['vd_l']:.1f} L")
print(f"CL: {pk['cl_l_h']:.2f} L/h")
print(f"t1/2: {pk['half_life_h']:.1f} h")

loading = 25 * weight
daily = 1000.0
auc = daily / pk["cl_l_h"]
print(f"Loading dose: {loading:.0f} mg")
print(f"Maintenance: 500 mg q12h = {daily:.0f} mg/day")
print(f"Predicted AUC24: {auc:.0f} mg*h/L")

# Renal decline
scr2 = 2.4
crcl2 = cockcroft_gault(age, weight, scr2)
pk2 = vancomycin_pk(crcl2, weight)
auc2 = daily / pk2["cl_l_h"]
auc2_reduced = 750.0 / pk2["cl_l_h"]
print()
print(f"CrCl at SCr 2.4: {crcl2:.1f} mL/min")
print(f"CL at SCr 2.4: {pk2['cl_l_h']:.2f} L/h")
print(f"t1/2 at SCr 2.4: {pk2['half_life_h']:.1f} h")
print(f"AUC24 on 1,000 mg/day: {auc2:.0f} mg*h/L")
print(f"AUC24 on 750 mg/day: {auc2_reduced:.0f} mg*h/L")

# Proportion of days covered
days_covered = 60
window_days = 90
pdc = days_covered / window_days
print()
print(f"PDC over {window_days} days: {pdc:.3f}")

assert 43.7 < crcl < 43.8
assert 46.5 < egfr < 46.7
assert round(pk["vd_l"]) == 49
assert round(auc) == 493
assert round(auc2_reduced) == 526
```

Expected output:

```text
CrCl: 43.8 mL/min
eGFR: 46.6 mL/min/1.73 m2
Vd: 49.0 L
CL: 2.03 L/h
t1/2: 16.7 h
Loading dose: 1750 mg
Maintenance: 500 mg q12h = 1000 mg/day
Predicted AUC24: 493 mg*h/L

CrCl at SCr 2.4: 29.2 mL/min
CL at SCr 2.4: 1.43 L/h
t1/2 at SCr 2.4: 23.8 h
AUC24 on 1,000 mg/day: 702 mg*h/L
AUC24 on 750 mg/day: 526 mg*h/L

PDC over 90 days: 0.667
```

A SQL query for exposure intervals from an MAR:

```sql
-- Build vancomycin exposure intervals from documented administrations.
WITH vanc AS (
    SELECT patient_id,
           administered_ts,
           dose_mg,
           route,
           LEAD(administered_ts) OVER (
               PARTITION BY patient_id ORDER BY administered_ts
           ) AS next_ts
    FROM medication_administrations
    WHERE LOWER(drug_name) LIKE '%vancomycin%'
      AND status = 'given'
)
SELECT patient_id,
       administered_ts AS start_ts,
       COALESCE(next_ts, administered_ts + INTERVAL '12 hours') AS end_ts,
       dose_mg,
       route
FROM vanc
ORDER BY patient_id, start_ts;
```

The `COALESCE` is an assumption: the last dose is assumed to cover 12 hours. If the regimen is
every 24 hours, the interval is wrong. Exposure intervals should be built from the order's
frequency, not from the gap between administrations, whenever the order is available.

## 8. Data, standards, and regulatory context

**Drug identification.** The National Drug Code (NDC) identifies a specific product, package,
and manufacturer. It changes when a manufacturer or package changes, which makes it unstable for
longitudinal analysis. RxNorm provides normalized names and identifiers at the ingredient,
strength, dose form, and clinical drug levels; it is the preferred nomenclature for analytics in
the United States. The Anatomical Therapeutic Chemical (ATC) classification is used
internationally for drug classes and utilization studies.

**Labeling.** Prescription drug labels in the United States follow a structured format. The
Pregnancy and Lactation Labeling Rule (PLLR), effective 30 June 2015, removed the pregnancy
letter categories A, B, C, D, and X and replaced them with narrative subsections on pregnancy,
lactation, and females and males of reproductive potential. A dataset that still carries a
pregnancy category is using a retired scheme.

**Drug interaction guidance.** The International Council for Harmonisation (ICH) M12 guideline
*Drug Interaction Studies* was adopted at Step 4 on 21 May 2024 and finalized by the FDA in
August 2024. It harmonizes in vitro and clinical drug interaction assessment, including reaction
phenotyping for CYP1A2, 2B6, 2C8, 2C9, 2C19, 2D6, and 3A4/5, transporter evaluation, and the use
of physiologically based pharmacokinetic (PBPK) modelling. The FDA's 2020 in vitro drug
interaction guidance remains relevant background. These are regulatory expectations for drug
developers, not clinical prescribing rules.

**Bioequivalence.** For generic approval in the United States, the FDA generally requires the
90% confidence interval of the ratio of generic to reference product exposure to fall within
80-125% for the relevant pharmacokinetic parameters. This is a regulatory standard for
interchangeability, not a guarantee of identical clinical effect in every patient.

**Therapeutic drug monitoring.** The 2020 ASHP/IDSA/PIDS/SIDP vancomycin guideline recommends
AUC-guided dosing with a target AUC24/MIC of 400-600 mg·h/L for serious MRSA infection. It
recommends Bayesian software with a validated population model as the preferred approach, using
one or two concentrations. Trough-only monitoring targeting 15-20 mg/L is no longer recommended.
The guideline is a consensus document, not a regulatory requirement.

**Older adults.** The American Geriatrics Society Beers Criteria, updated in 2023, lists
potentially inappropriate medications for adults aged 65 and older. The 2023 update moved
rivaroxaban to "avoid" for long-term treatment of atrial fibrillation and venous
thromboembolism, added warfarin to "avoid" as initial therapy unless a direct oral anticoagulant
is contraindicated, and moved aspirin to "avoid" for initiation for primary prevention. These are
clinical recommendations with exceptions; they are not billing rules.

**Pharmacogenomics.** The Clinical Pharmacogenetics Implementation Consortium (CPIC) publishes
peer-reviewed guidelines for gene-drug pairs. The FDA has included pharmacogenomic information in
labels for many drugs, including a boxed warning for clopidogrel concerning reduced effectiveness
in CYP2C19 poor metabolizers. Genotype data are often missing from routine records, and absence
of a genotype is not evidence of a normal phenotype.

**Adherence metrics.** PDC and MPR are standard in pharmacy claims analysis. Neither is a
validated measure of ingestion, and both are sensitive to the denominator, the refill gap, and
drug switching. A PDC of 0.80 is a common threshold in quality measures; it is a convention, not
a biological threshold.

> **Note:** This section is educational and is not medical advice. Drug doses, targets, and
> interactions must be verified against current labeling and institutional protocols before any
> clinical use.

## 9. Industry practice and case evidence

**Vancomycin dosing has shifted from troughs to AUC.** The 2020 consensus guideline replaced
trough-only monitoring with AUC-guided dosing, based on evidence that AUC correlates better with
efficacy and nephrotoxicity. A 2021 Canadian commentary argued that the randomized evidence for
AUC-guided dosing is limited, that Bayesian software is not regulated as a device in all
jurisdictions, and that 31 different population pharmacokinetic models had been identified in one
review. The disagreement is documented. The inference for data teams is that "the guideline
says" is not enough: the model, the assay, and the implementation vary.

**Beers Criteria and real-world prescribing.** The 2023 Beers update is a consensus list, not an
observed practice standard. Studies of potentially inappropriate prescribing consistently find
that a substantial fraction of older adults receive at least one Beers-listed medication. The
criteria are useful as a feature or a quality metric, but they are not a complete measure of
prescribing appropriateness, and they do not account for individualized risk-benefit.

**Pregnancy labeling changed, but old data persist.** The PLLR removed the letter categories in
2015, yet many clinical datasets and reference tools still carry them. A feature derived from a
pregnancy category is therefore both outdated and oversimplified: the categories conflated
animal and human data and were widely misinterpreted as a graded risk scale. A current analysis
should use the narrative label sections and the date of the label.

**Drug interaction evidence is mechanistic and often not outcome-based.** The ICH M12 framework
standardizes how interactions are studied, but most labeled interactions are characterized by
changes in exposure, not by clinical events. A 30% increase in AUC may or may not matter
depending on the therapeutic index. Data scientists should distinguish a documented
pharmacokinetic change from a documented clinical harm.

**Adherence measurement is a measurement problem.** Claims-based PDC is widely used in quality
and outcomes research. It measures filling, not taking. A patient who fills a 90-day supply and
discards it has a PDC of 1.0. A patient who takes half the prescribed dose has a PDC of 1.0 if
they refill on time. The metric is useful for population-level inference and weak for
individual-level exposure.

**What is documented versus inferred.** The guideline targets, label changes, and interaction
mechanisms above are documented. The claim that a specific model or product improves outcomes is
not established by these sources. That requires a prospective evaluation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Order used as exposure | Exposure rate exceeds pharmacy dispensing | Intent recorded, administration not | Misclassified exposure | Use MAR or claims with explicit layer |
| Dose without weight | mg/kg orders with no weight | Weight missing or estimated | Wrong dose | Weight field with source and time |
| Renal function frozen at admission | Creatinine rises, dose unchanged | Time-varying clearance ignored | Toxicity | Recompute clearance over time |
| Cockcroft-Gault and eGFR conflated | Dosing uses eGFR | Different quantities and units | Systematic under- or overdosing | Label which equation was used |
| Trough interpreted without timing | Level drawn at the wrong time | Wrong sampling relative to dose | False reassurance or false alarm | Record dose and sample times |
| Interaction treated as binary | Flag present but no timing | Inhibition versus induction differ in onset | Missed delayed effect | Model interaction start and mechanism |
| Prodrug assumed active | Clopidogrel exposure counted as effect | Requires CYP2C19 activation | Misclassified efficacy | Include genotype or effect marker |
| Active metabolite ignored | Parent drug level normal, toxicity present | Metabolite accumulates | Toxicity missed | Use metabolite where available |
| Adherence assumed from fills | PDC 1.0 with no refill gaps | Filling is not ingestion | Exposure misclassification | Triangulate; state the proxy |
| Primary non-adherence invisible | New prescription never filled | Claims start at fill | Denominator excludes non-fillers | Link prescription to fill |
| Drug name not normalized | Brand and generic counted separately | Free-text names | Duplicate exposures | RxNorm mapping |
| Combination product split incorrectly | Components counted twice | Ingredient-level mapping | Dose inflation | Map at clinical drug level |
| Pediatric dose scaled linearly | Neonatal dose from adult mg/kg | Immature clearance | Toxicity or failure | Age-specific dosing; allometry |
| Pregnancy category used | Retired A/B/C/D/X field | Legacy data | Misleading risk assessment | Use PLLR narrative and date |
| Immortal time bias | Exposure starts after cohort entry | Time-varying exposure ignored | Inflated benefit | Landmark or time-varying analysis |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The order reflects the administered dose | Orders are widely available | Cancellations, omissions, refusals | Compare with MAR |
| The MAR reflects ingestion | Administration is closer to exposure | Documentation errors, held doses | Chart review sample |
| Creatinine clearance approximates drug clearance | Dosing equations depend on it | Non-renal elimination, unstable creatinine | Measure concentrations |
| Vd is constant for the patient | Loading dose depends on it | Obesity, edema, critical illness | Use weight-adjusted estimates; measure levels |
| Clearance is linear | Maintenance dose depends on it | Saturable metabolism (phenytoin, ethanol) | Check dose-concentration linearity |
| Bioavailability is known | Oral dosing depends on it | Food, formulation, first-pass variability | Use IV where possible; measure levels |
| The therapeutic range applies to the patient | Interpretation depends on it | Age, indication, genotype | Use indication-specific targets |
| Interaction magnitude is known | Dose adjustment depends on it | Genotype, dose, comedications | Monitor concentrations or effects |
| Adherence is constant | Exposure models assume it | Intermittent use, cost barriers | PDC over time; triangulate |
| The drug name maps to one ingredient | Pooling depends on it | Combination products, salts | RxNorm audit |

## 12. Exercises

**Exercise 1 — Exposure ladder (recall and application, 40 minutes).**
For a statin prescribed at discharge, list the order, dispensing, administration, and adherence
variables available in a typical claims-plus-EHR dataset. Identify which layer each variable
measures. Deliverable: a four-row table with source and limitation. Acceptance criteria: each
layer distinguished; at least one limitation per layer. Time: 40 minutes.

**Exercise 2 — PK calculation (calculation, 50 minutes).**
A 50-year-old woman, 60 kg, SCr 1.0 mg/dL, receives a drug with Vd 0.5 L/kg and clearance equal
to 0.8 times Cockcroft-Gault CrCl. Compute CrCl, Vd, CL, t1/2, and the maintenance dose for a
target average concentration of 2 mg/L with F = 1. Deliverable: a calculation sheet. Acceptance
criteria: units at every step; assumptions labelled; arithmetic correct. Time: 50 minutes.

**Exercise 3 — Renal dosing (application, 45 minutes).**
Using the vancomycin case, recompute the regimen if the patient's weight is 110 kg and the
creatinine is 1.6 mg/dL. Explain how obesity changes Vd and whether actual, ideal, or adjusted
body weight should be used. Deliverable: a revised regimen with a rationale. Acceptance criteria:
weight choice justified; AUC within target; assumptions stated. Time: 45 minutes.

**Exercise 4 — Interaction critique (critique, 50 minutes).**
A dataset flags "drug interaction" as a binary when warfarin and fluconazole overlap. Critique
the feature and propose a time-varying alternative. Deliverable: a 400-word critique and a schema.
Acceptance criteria: distinguishes PK from PD; accounts for onset; proposes monitoring. Time: 50
minutes.

**Exercise 5 — Adherence metric (calculation and critique, 40 minutes).**
A patient receives three 30-day fills over 90 days, with the third fill on day 75. Compute PDC
and MPR. Explain why the two differ and what neither measures. Deliverable: calculations and a
300-word explanation. Acceptance criteria: both metrics correct; the limitation is explicit.
Time: 40 minutes.

**Exercise 6 — Pregnancy labeling (critique, 35 minutes).**
A model uses a legacy pregnancy category A/B/C/D/X field as a feature. Explain why the field is
obsolete and propose a replacement. Deliverable: a 300-word critique and a data specification.
Acceptance criteria: cites the PLLR effective date; proposes narrative label sections and date;
notes that absence of data is not safety. Time: 35 minutes.

**Exercise 7 — Exposure table design (design, 75 minutes).**
Design a medication exposure table for a sepsis cohort. Include drug, dose, route, start, end,
source layer, and provenance. Show how you would represent a titrated vasopressor. Deliverable: a
schema and three example rows. Acceptance criteria: time-varying; source explicit; titrated drug
represented as a rate over time. Time: 75 minutes.

**Exercise 8 — Pharmacogenomics feature (design, 60 minutes).**
Design a feature set for CYP2C19 phenotype in a clopidogrel cohort. Address missing genotypes,
phenoconversion from concomitant CYP2C19 inhibitors, and the difference between genotype and
phenotype. Deliverable: a feature specification with three features. Acceptance criteria:
missingness addressed; phenoconversion defined; no claim that genotype equals effect. Time: 60
minutes.

## 13. Self-check answers

**Answer to Exercise 2.** CrCl = ((140 - 50) x 60) / (72 x 1.0) = 5400 / 72 = 75.0 mL/min. Vd =
0.5 x 60 = 30 L. CL = 0.8 x 75 = 60 mL/min = 3.6 L/h. t1/2 = 0.693 x 30 / 3.6 = 5.78 h.
Maintenance dose rate = CL x target = 3.6 L/h x 2 mg/L = 7.2 mg/h = 172.8 mg/day. With F = 1,
the daily dose is approximately 173 mg.

**Answer to Exercise 3.** At 110 kg, Vd at 0.7 L/kg = 77 L. CrCl by Cockcroft-Gault using actual
body weight = ((140 - 68) x 110) / (72 x 1.6) = 7920 / 115.2 = 68.8 mL/min. CL = 0.689 x 68.8 +
3.66 = 51.1 mL/min = 3.07 L/h. A loading dose of 25 mg/kg actual body weight would be 2,750 mg,
which is below the 3,000 mg cap but high; many protocols cap the loading dose and use actual body
weight for the first dose. Maintenance for AUC 500 would be 500 x 3.07 = 1,535 mg/day, but
obesity changes Vd and clearance non-linearly, and AUC monitoring is essential. The key point is
that the choice of weight is a clinical decision with a large effect, not a default.

**Answer to Exercise 5.** PDC = days covered / 90. Three 30-day fills cover 90 days if there are
no gaps, so PDC = 1.0. MPR = total days supply / days in the refill interval. If the observation
window is 90 days and total supply is 90 days, MPR = 1.0 as well. The two differ when the
denominator or the gap definition differs. Neither measures ingestion; a patient can fill and
discard.

**Answer to self-check on order versus MAR.** An order records intent; a MAR records a nurse's
documentation of administration. The MAR is closer to exposure but still not a measurement of
ingestion. Claims record dispensing. None is ground truth.

**Answer to self-check on half-life.** Half-life determines time to steady state (about 4-5
half-lives) and the duration of effect after stopping. It does not determine the maintenance dose
directly; clearance does.

**Answer to self-check on CYP inhibition versus induction.** Inhibition reduces enzyme activity
and raises substrate concentrations, often within hours to days. Induction increases enzyme
expression and lowers substrate concentrations, typically over days to weeks as new enzyme is
synthesized.

**Rubric for Exercise 7.** Full credit: time-varying intervals; source layer explicit; titrated
drug represented as rate over time; provenance fields. Half credit: table is static or lacks
source. No credit: one row per patient with "ever exposed."

**Rubric for Exercise 8.** Full credit: genotype and phenotype distinguished; missingness
handled; phenoconversion defined; no causal overclaim. Half credit: genotype feature only. No
credit: assumes genotype is always available.

## 14. Interview questions

**1. What is the difference between an order and an administration?**
A strong answer distinguishes intent, documentation of delivery, and actual ingestion, and says
which is closer to exposure. A weak answer treats them as the same.

**2. Why is creatinine clearance used for drug dosing instead of eGFR?**
A strong answer explains that Cockcroft-Gault is in most labels and is not normalized to body
surface area. A weak answer says they are equivalent.

**3. How do you compute a loading dose, and what does it depend on?**
A strong answer gives Vd x target / F and notes that it does not depend on clearance. A weak
answer confuses loading and maintenance.

**4. What determines the maintenance dose?**
A strong answer gives clearance x target / F. A weak answer says half-life.

**5. What is the difference between a pharmacokinetic and a pharmacodynamic interaction?**
A strong answer distinguishes concentration change from effect change and gives examples. A weak
answer treats all interactions as concentration changes.

**6. Why did vancomycin monitoring move from troughs to AUC?**
A strong answer explains the correlation with efficacy and nephrotoxicity and the limitations of
troughs. A weak answer says "the guideline changed."

**7. What is a prodrug, and why does it matter for exposure models?**
A strong answer explains that the parent drug is inactive and requires metabolism, so exposure to
the parent is not exposure to the active moiety. A weak answer says it is "inactive."

**8. How would you measure adherence from claims, and what are the limits?**
A strong answer gives PDC or MPR, defines the denominator, and states that filling is not
ingestion. A weak answer says "count refills."

**9. What is immortal time bias in a medication study?**
A strong answer explains that exposure cannot start before cohort entry without guaranteeing
survival to treatment, and proposes a landmark or time-varying analysis. A weak answer says it is
"a bias."

**10. Why is pregnancy labeling a data problem?**
A strong answer explains the 2015 PLLR, the retirement of letter categories, and the persistence
of legacy fields. A weak answer says categories are "outdated" without the date or replacement.

**11. How would you represent a titrated infusion in a dataset?**
A strong answer proposes rate over time with start and end timestamps, not a daily dose. A weak
answer sums the total dose.

**12. What makes a drug interaction clinically important?**
A strong answer combines the magnitude of exposure change, the therapeutic index, and the
availability of monitoring. A weak answer says any interaction is important.

## 15. Key terms

- **Absorption** — movement of drug from the administration site into blood.
- **Active metabolite** — a metabolite with pharmacological activity.
- **Adherence** — extent to which a patient takes the prescribed regimen.
- **ADR** — adverse drug reaction.
- **AUC** — area under the concentration-time curve.
- **Bioavailability (F)** — fraction of the dose reaching systemic circulation.
- **Clearance (CL)** — volume of plasma cleared per unit time.
- **Cmax** — maximum concentration after a dose.
- **Cockcroft-Gault** — creatinine clearance estimation equation.
- **CYP450** — cytochrome P450 enzyme family.
- **DDI** — drug-drug interaction.
- **eGFR** — estimated glomerular filtration rate.
- **First-pass metabolism** — metabolism before systemic circulation.
- **Half-life** — time for concentration to halve.
- **Induction** — increased enzyme expression.
- **Inhibition** — reduced enzyme activity.
- **Loading dose** — dose to reach target concentration quickly.
- **Maintenance dose** — dose to maintain target at steady state.
- **MAR** — medication administration record.
- **MIC** — minimum inhibitory concentration.
- **MPR** — medication possession ratio.
- **NDC** — National Drug Code.
- **PD** — pharmacodynamics.
- **PDC** — proportion of days covered.
- **Persistence** — duration of continued treatment.
- **PK** — pharmacokinetics.
- **Prodrug** — inactive drug requiring metabolism for activity.
- **RxNorm** — normalized drug nomenclature.
- **Steady state** — input equals output.
- **TDM** — therapeutic drug monitoring.
- **Therapeutic index** — ratio of toxic to effective exposure.
- **Vd** — apparent volume of distribution.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Guidelines and regulation**

- Rybak MJ, Le J, Lodise TP, et al. Therapeutic monitoring of vancomycin for serious
  methicillin-resistant *Staphylococcus aureus* infections: a revised consensus guideline and
  review by ASHP, IDSA, PIDS, and SIDP. *American Journal of Health-System Pharmacy*,
  2020;77(11):835-864. https://doi.org/10.1093/ajhp/zxaa036
- International Council for Harmonisation. M12 Drug Interaction Studies, Step 4, 21 May 2024.
  https://database.ich.org/sites/default/files/ICH_M12_Step4_Guideline_2024_0521_0.pdf
- US Food and Drug Administration. M12 Drug Interaction Studies, August 2024.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m12-drug-interaction-studies
- US Food and Drug Administration. Pregnancy and Lactation Labeling Final Rule, effective 30 June
  2015. https://www.fda.gov/vaccines-blood-biologics/biologics-rules/pregnancy-and-lactation-labeling-final-rule
- American Geriatrics Society. 2023 updated AGS Beers Criteria for potentially inappropriate
  medication use in older adults. *Journal of the American Geriatrics Society*, 2023.
  https://doi.org/10.1111/jgs.18372
- Clinical Pharmacogenetics Implementation Consortium. Guidelines.
  https://cpicpgx.org/guidelines/

**Evidence and background**

- Stewart JJ, et al. A Canadian perspective on the revised 2020 ASHP-IDSA-PIDS-SIDP guidelines
  for vancomycin AUC-based therapeutic drug monitoring. *Journal of the Association of Medical
  Microbiology and Infectious Disease Canada*, 2021.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9612435/
- Delgado C, Baweja M, Crews DC, et al. A unifying approach for GFR estimation: recommendations
  of the NKF-ASN Task Force. *Journal of the American Society of Nephrology*, 2021.
  https://doi.org/10.1681/ASN.2021070988
- US Food and Drug Administration. Bioequivalence studies with pharmacokinetic endpoints for
  drugs submitted under an abbreviated new drug application.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents

**Background reading**

- Rowland M, Tozer TN. *Clinical Pharmacokinetics and Pharmacodynamics: Concepts and
  Applications*.
- Brunton LL, Knollmann BC, et al. *Goodman and Gilman's The Pharmacological Basis of
  Therapeutics*.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_laboratory-medicine.md`](03_laboratory-medicine.md)
- Next topic: [`05_disease-taxonomy-and-coding.md`](05_disease-taxonomy-and-coding.md)
- Related topic: [`09_special-populations.md`](09_special-populations.md)
- Related topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Survey file: [`../../01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

