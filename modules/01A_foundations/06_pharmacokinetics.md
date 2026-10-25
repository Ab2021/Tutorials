# Pharmacokinetics

> Module 01A — Foundations, Assumptions, and Units
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 03, 04, 05
> Version 2026.3

## 1. Why this topic exists

Pharmacokinetics (PK) is the quantitative description of what the body does to a drug: absorption,
distribution, metabolism, and excretion. It is the bridge between a dose and an exposure, and exposure
is what determines both effect and toxicity. Every dose adjustment in a label, every renal dosing
recommendation, and every drug interaction warning is a PK statement.

The failures are consequential and common.

- A loading dose is reduced in a patient with renal impairment. The loading dose depends on volume of
  distribution, not clearance, so reducing it delays the achievement of therapeutic concentrations
  without protecting the patient from anything.
- A maintenance dose is calculated from ideal body weight for a patient with severe obesity, and the
  volume of distribution is underestimated by a factor of two. The patient is under-dosed for days
  before therapeutic drug monitoring corrects it.
- An eGFR indexed to 1.73 m² is used directly for dosing in a patient with a body surface area of
  2.4 m². The indexed value underestimates the absolute clearance by about 39%, and the drug
  accumulates.
- A strong CYP3A4 inhibitor is added to a drug with a narrow therapeutic index. The AUC rises fivefold,
  and the dose is not adjusted because the interaction alert was dismissed as "not clinically
  significant."
- A trough concentration is drawn 30 minutes after a dose instead of 30 minutes before the next one.
  The measured value is interpreted as a trough, the dose is reduced, and the patient is
  under-treated.
- A once-daily aminoglycoside is monitored with a peak and trough when the relevant target is the
  area under the curve over 24 hours. The monitored values look acceptable while the exposure is
  subtherapeutic.

These errors share a structure: a PK parameter is applied outside the population, unit, or model in
which it was estimated. The arithmetic is usually right. The model is wrong.

The topic matters to data scientists for three reasons. First, PK data are among the most
quantitatively structured in medicine, and they are a natural place to practise the modelling habits
that later modules assume. Second, dose adjustment logic is increasingly embedded in software, and a
data scientist who cannot read a compartment model cannot validate the software. Third, PK/PD is the
prototype for the exposure-response thinking used in device calibration, biomarker programmes, and
health economic models.

> **Note:** This topic teaches the one- and two-compartment models, the parameters that define them,
> and the calculations that follow. It does not teach clinical prescribing. The numerical examples are
> illustrative and are not dosing recommendations.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** absorption, distribution, metabolism, and excretion, and place each in a compartment
   model with its parameter. Assessed in Exercise 1.
2. **Compute** volume of distribution, clearance, elimination rate constant, half-life, and AUC from a
   dose and a concentration-time profile, with units. Assessed in Exercises 2 and 3.
3. **Distinguish** a loading dose from a maintenance dose and explain why renal impairment changes one
   and not the other. Assessed in Exercise 4.
4. **Predict** accumulation and time to steady state from the half-life and the dosing interval.
   Assessed in Exercise 5.
5. **Adjust** a dose for renal function using an absolute clearance estimate, and explain why an
   indexed eGFR must be de-indexed before use. Assessed in Exercise 6.
6. **Interpret** a therapeutic drug monitoring result with the sampling time, the dosing history, and
   the assay. Assessed in Exercise 7.
7. **Explain** the mechanisms of metabolic drug interactions and the AUC-ratio classification of
   inhibitors and inducers. Assessed in Exercise 8.
8. **Critique** a published PK claim for model misspecification, population mismatch, and unit error.
   Assessed in all exercises through the reporting rubric.

## 3. The core mental model

A drug enters the body, distributes into a volume, and is removed by clearance. The simplest model
treats the body as one well-stirred compartment with a volume and a clearance.

```text
                       DOSE (F x dose reaches systemic circulation)
                                     |
                                     v
                 +---------------------------------------+
                 |        CENTRAL COMPARTMENT            |
                 |        volume Vd, concentration C     |
                 +---------------------------------------+
                          |                     |
                          | metabolism          | excretion
                          v                     v
                    CLEARANCE CL  =  k x Vd      (liver, kidney, other)

   Mass balance:  dA/dt = input - CL x C
   After a bolus: C(t) = C0 x exp(-k t),  C0 = dose / Vd
   Exposure:      AUC = dose / CL
   Half-life:     t1/2 = ln(2) / k
```

Four relationships organise everything else.

**Concentration is amount divided by volume.** A dose of 449 mg into a volume of 22.45 L gives an
initial concentration of 20.0 mg/L. The volume is apparent: it is the volume that would produce the
observed concentration if the drug were uniformly distributed, which it is not.

**Exposure is dose divided by clearance.** AUC = dose/CL. Clearance, not half-life, determines the
average concentration at steady state. This is why renal impairment changes the maintenance dose: it
changes clearance.

**Half-life is a derived quantity.** t½ = ln(2)/k = ln(2)·Vd/CL. It changes when either volume or
clearance changes, which is why a half-life quoted without its population is incomplete.

**Steady state is approached, not reached.** After n half-lives, the fraction of the final steady-state
concentration achieved is 1 − 0.5^n. At four half-lives it is 93.75%; at five, 96.875%.

### 3.1 The two questions every PK calculation answers

```text
   How high?   ->  loading dose  =  Vd x target concentration / F
   How often?  ->  maintenance dose = CL x target AUC / F  (per dosing interval)
```

The loading dose fills the volume. The maintenance dose replaces what clearance removes. Confusing the
two is the mechanism behind the renal-impairment loading dose error in section 1: the loading dose does
not depend on renal function, because it depends on volume, and volume is largely unchanged in kidney
disease.

### 3.2 Linear versus non-linear kinetics

```text
   Linear (first-order):     CL constant, AUC proportional to dose
   Non-linear (saturable):   CL falls as concentration rises, AUC rises more than proportionally
```

Most drugs are approximately linear in the therapeutic range. Phenytoin, ethanol, and high-dose
salicylates are the classic exceptions. In non-linear kinetics, a small dose increase can produce a
large concentration increase, and half-life is concentration-dependent. A model that assumes linearity
will under-predict accumulation near saturation.

### 3.3 The two-compartment refinement

After an intravenous bolus, many drugs show a rapid early decline (distribution) followed by a slower
terminal decline (elimination).

```text
   C(t) = A exp(-alpha t) + B exp(-beta t)

   alpha phase:  distribution into tissues, half-life of minutes
   beta phase:   terminal elimination, half-life of hours to days
```

The terminal half-life is the one usually quoted, and it is the one that determines time to steady
state on repeated dosing. The distribution phase matters for peak concentration and for toxicity
immediately after a bolus.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Pharmacokinetics | What the body does to a drug | Confusing with pharmacodynamics | none |
| Pharmacodynamics | What the drug does to the body | Confusing with pharmacokinetics | none |
| Absorption | Movement of drug from the administration site into circulation | Confusing with bioavailability | mass/time |
| Bioavailability (F) | Fraction of an administered dose reaching systemic circulation | Assuming 1 for oral drugs | fraction |
| Distribution | Movement of drug between plasma and tissues | Confusing with elimination | none |
| Metabolism | Chemical transformation of the drug | Assuming it always inactivates | none |
| Excretion | Removal of drug or metabolites from the body | Confusing with clearance | mass/time |
| First-pass effect | Metabolism during absorption before systemic circulation | Ignoring it for oral drugs | fraction |
| Volume of distribution (Vd) | Apparent volume relating amount to concentration | Treating it as a physiological volume | L |
| Clearance (CL) | Volume of plasma cleared of drug per time | Confusing with elimination rate | L/h |
| Elimination rate constant (k) | Fractional removal per time | Reporting it without a time unit | 1/h |
| Half-life (t½) | Time for concentration to halve | Assuming it is constant in non-linear kinetics | h |
| AUC | Area under the concentration-time curve | Reporting without the time interval | mg·h/L |
| Cmax | Maximum concentration after a dose | Confusing with the steady-state peak | mg/L |
| Cmin / trough | Minimum concentration before the next dose | Sampling at the wrong time | mg/L |
| Css | Average concentration at steady state | Confusing with the peak | mg/L |
| Loading dose | Dose to reach a target concentration quickly | Reducing it in renal impairment | mg |
| Maintenance dose | Dose to replace cleared drug | Confusing it with the loading dose | mg per interval |
| Dosing interval (τ) | Time between doses | Choosing it without reference to half-life | h |
| Accumulation factor | Ratio of steady-state to single-dose exposure | Ignoring it for short intervals | dimensionless |
| Steady state | Condition in which input equals elimination | Claiming it before 4–5 half-lives | none |
| First-order kinetics | Elimination proportional to concentration | Assuming it for saturable drugs | none |
| Zero-order kinetics | Elimination at a constant rate | Confusing with first-order | none |
| Michaelis-Menten | Saturable elimination described by Vmax and Km | Treating Km as a therapeutic threshold | concentration |
| Compartment | A Kinetics-defined pool of drug | Confusing with an anatomical compartment | none |
| One-compartment model | Body treated as a single well-stirred volume | Assuming it fits all drugs | none |
| Two-compartment model | Central and peripheral compartments | Ignoring the distribution phase | none |
| Distribution half-life | Half-life of the early decline | Confusing with the terminal half-life | h |
| Terminal half-life | Half-life of the elimination phase | Using it to time a peak | h |
| Protein binding | Fraction of drug bound to plasma proteins | Assuming binding is constant across concentrations | fraction |
| Free fraction (fu) | Unbound fraction, pharmacologically active | Ignoring it in highly bound drugs | fraction |
| Therapeutic index | Ratio of toxic to effective exposure | Treating it as a safety guarantee | dimensionless |
| Narrow therapeutic index | Small difference between effective and toxic exposure | Monitoring less than for other drugs | none |
| Therapeutic drug monitoring (TDM) | Measurement of drug concentration to guide dosing | Sampling without regard to timing | mg/L |
| Pharmacogenomics | Genetic influence on drug response and disposition | Treating genotype as deterministic | variant |
| Drug interaction | A change in exposure caused by another agent | Assuming all interactions are contraindications | none |
| CYP450 | The cytochrome P450 family of oxidative enzymes | Treating all isoforms as interchangeable | isoform |
| Inhibitor | An agent that reduces metabolic clearance | Assuming the effect is immediate | AUC ratio |
| Inducer | An agent that increases metabolic clearance | Assuming the effect is immediate | AUC ratio |
| Transporters | Membrane proteins moving drugs across barriers | Ignoring them for oral absorption | none |
| Bioequivalence | Similar exposure from two formulations | Assuming it means identical | ratio |
| Dialysis clearance | Drug removal by an extracorporeal circuit | Assuming it equals renal clearance | L/h |
| CRRT | Continuous renal replacement therapy | Treating it as intermittent dialysis | L/h |
| Obesity-adjusted weight | Weight basis for dosing in obesity | Using total body weight blindly | kg |
| Ideal body weight (IBW) | Weight predicted from height and sex | Using it for volume in obesity | kg |
| Adjusted body weight (ABW) | IBW plus a fraction of excess weight | Using it without stating the fraction | kg |
| AUC/MIC | Exposure relative to minimum inhibitory concentration | Reporting AUC without the MIC | h |
| Time above MIC | Fraction of the dosing interval above the MIC | Ignoring it for time-dependent drugs | percent |
| Peak/MIC | Peak relative to MIC | Ignoring it for concentration-dependent drugs | dimensionless |
| MIC | Minimum inhibitory concentration | Treating it as a fixed organism property | mg/L |

Three definitions are used inconsistently in practice.

**Clearance versus half-life.** Clearance is the primary determinant of steady-state exposure;
half-life is a derived quantity that also depends on volume. Two drugs with the same half-life can have
very different clearances if their volumes differ. A dose adjustment based on half-life alone is
therefore incomplete.

**Apparent volume.** Vd is not a physiological space. A drug with a Vd of 200 L in a 70 kg patient is
not distributed into 200 L of fluid; it is extensively bound to tissue, so the plasma concentration is
low relative to the amount in the body. The volume is the proportionality constant between amount and
plasma concentration, nothing more.

**Trough.** A trough is the concentration immediately before the next dose, at steady state. A sample
drawn at any other time is not a trough, and interpreting it as one produces a dose error. The
sampling time must be recorded with the result.

## 5. The mechanics, step by step

The procedure takes a dosing question from the clinical target to a defensible dose and monitoring plan.

### Step 1 — State the question

**Input.** A clinical request.

**Output.** One of three question types: efficacy (is exposure adequate), toxicity (is exposure too
high), or interaction (has exposure changed).

**Reasoning.** The question determines which parameter matters. Efficacy usually depends on AUC or time
above MIC; toxicity often depends on peak or trough; an interaction changes clearance and therefore
AUC. Calculating all three when only one is relevant wastes data and confuses the interpretation.

**Branch point.** If the question is about an immediate effect after a bolus, the distribution phase
matters and a one-compartment model may be inadequate. Go to Step 2.

### Step 2 — Choose the structural model

**Input.** The drug, route, and available data.

**Output.** A one-compartment, two-compartment, or physiologically based model.

**Reasoning.** A one-compartment model is adequate when the distribution phase is fast relative to the
dosing interval and the sampling times. A two-compartment model is needed when early concentrations
matter, as with an intravenous bolus of a lipophilic drug. A physiologically based model is needed when
tissue-specific concentrations matter or when extrapolating across species.

**Branch point.** If the data show a biexponential decline, a one-compartment fit will misestimate the
terminal half-life. Use two compartments or restrict the analysis to the terminal phase.

### Step 3 — Assemble the parameters from a stated source

**Input.** The drug and the population.

**Output.** Vd, CL, F, and their variability.

**Reasoning.** Parameters come from the label, a population PK study, or a local dataset. Each source
has a population: healthy volunteers, adults with a specific disease, children, or obese patients.
Using a parameter outside its population is an extrapolation. The source must be recorded because it
determines the uncertainty.

**Branch point.** If no parameter is available for the population, use a population model with
covariate effects or state that the calculation is an extrapolation.

### Step 4 — Establish the route and bioavailability

**Input.** The route of administration.

**Output.** F, and the time to peak for extravascular routes.

**Reasoning.** An intravenous dose has F = 1 by definition. An oral dose has F below 1 because of
incomplete absorption and first-pass metabolism. Using an intravenous Vd with an oral dose without
accounting for F overestimates the concentration by 1/F.

**Branch point.** If F is unknown for the formulation, the calculation is for the intravenous route
only, or it must be expressed as a dose-normalised exposure.

### Step 5 — Compute the loading dose

**Input.** Vd, target concentration, F.

**Output.** Loading dose in mg.

**Reasoning.** LD = Vd × target concentration / F. The loading dose fills the volume and is independent
of clearance. This is why renal impairment does not change it: kidney disease changes clearance, not
the apparent volume, at least to a first approximation.

**Branch point.** If the target is a trough rather than a peak, the loading dose must be computed for
the concentration at the time of the target, not for C0.

### Step 6 — Compute clearance for the patient

**Input.** Population CL and patient covariates.

**Output.** An individual CL estimate.

**Reasoning.** Clearance scales with renal function for renally cleared drugs, with hepatic function
for hepatically cleared drugs, and with body size for both. The scaling relationship is
drug-specific: a linear relationship with creatinine clearance for aminoglycosides, an allometric
relationship with weight for many drugs, and a complex relationship with age for paediatric patients.

**Branch point.** If the drug has both renal and hepatic clearance, the contributions must be combined.
Suppressing one pathway does not eliminate the other.

### Step 7 — Compute the elimination rate constant and half-life

**Input.** CL and Vd.

**Output.** k in 1/h and t½ in h.

**Reasoning.** k = CL/Vd and t½ = ln(2)/k. Both are derived, not independent. A change in volume
changes the half-life without changing clearance, and a change in clearance changes both.

**Branch point.** If k is estimated from a concentration-time slope, the slope must be fitted to the
terminal phase and the units of the slope are 1/time.

### Step 8 — Compute the maintenance dose for the target exposure

**Input.** CL, target AUC over the interval, F.

**Output.** Dose per interval in mg.

**Reasoning.** At steady state, the average rate in equals the average rate out: dose·F/τ = CL·Css, so
dose = CL × AUC_target × ... over the interval, dose = CL × AUC(0-τ). The maintenance dose is
proportional to clearance, which is why renal impairment reduces it.

**Branch point.** If the target is a trough or a peak rather than an AUC, the dose and interval must be
solved together, because both affect the peak-trough difference.

### Step 9 — Choose the dosing interval

**Input.** Half-life and therapeutic window.

**Output.** τ in hours.

**Reasoning.** The interval determines the fluctuation between peak and trough. A short interval
relative to the half-life produces little fluctuation and accumulation; a long interval produces large
fluctuation and little accumulation. For concentration-dependent antibiotics, a long interval with a
high peak is deliberate; for time-dependent antibiotics, frequent dosing or a prolonged infusion is
preferred.

**Branch point.** If the required interval is shorter than the practical dosing schedule, a different
formulation or route is needed.

### Step 10 — Predict the peak concentration

**Input.** Dose, Vd, F, and the infusion duration.

**Output.** Cmax in mg/L.

**Reasoning.** For a bolus, C0 = dose·F/Vd. For a short infusion, the peak is lower and occurs at the
end of the infusion: Cmax = (dose·F/(Vd·k·T_inf))·(1 − e^(−k·T_inf)) plus any residual from previous
doses. Ignoring the infusion duration overestimates the peak.

**Branch point.** If the infusion is long relative to the half-life, the peak approaches the
steady-state value and the bolus formula is invalid.

### Step 11 — Predict the trough concentration

**Input.** Cmax and the interval.

**Output.** Cmin in mg/L.

**Reasoning.** Cmin = Cmax·e^(−k·(τ − T_inf)) for a short infusion, or C0·e^(−k·τ) for a bolus. At
steady state, the trough includes accumulation from previous doses. The trough is the concentration
that determines whether the next dose is safe for drugs with concentration-dependent toxicity.

**Branch point.** If the trough is below the assay's limit of quantification, the assay cannot support
the monitoring plan. Use a more sensitive assay or a different target.

### Step 12 — Compute AUC

**Input.** Dose, CL, or a concentration-time profile.

**Output.** AUC in mg·h/L.

**Reasoning.** For linear kinetics, AUC(0-inf) = dose·F/CL. From data, AUC is computed by the
trapezoidal rule plus the terminal tail C_last/k. The two should agree within sampling error; a large
discrepancy indicates a model misspecification or a dosing history error.

**Branch point.** If the sampling is sparse around the peak, the trapezoidal AUC underestimates the
true exposure. State the sampling schedule.

### Step 13 — Compute accumulation at steady state

**Input.** k and τ.

**Output.** Accumulation factor.

**Reasoning.** For repeated bolus dosing, the accumulation factor is 1/(1 − e^(−k·τ)). It approaches 1
for τ much longer than the half-life and τ/(t½·ln2) for τ much shorter. The factor applies to both peak
and trough, so a short interval can produce substantial accumulation even for a drug with a moderate
half-life.

**Branch point.** If the drug has non-linear kinetics, the accumulation factor is not constant and
must be computed by simulation.

### Step 14 — Determine time to steady state

**Input.** Half-life.

**Output.** Time in hours or days.

**Reasoning.** Steady state is approached with the same exponential as elimination: after n half-lives,
the fraction of steady state achieved is 1 − 0.5^n. Four to five half-lives is the usual approximation.
A loading dose bypasses the wait by filling the volume immediately.

**Branch point.** If the half-life is long and the clinical situation is urgent, a loading dose is
required. If the half-life is short, steady state is reached quickly and a loading dose adds little.

### Step 15 — Check linearity

**Input.** Dose range and observed concentrations.

**Output.** A judgement about linear versus non-linear kinetics.

**Reasoning.** Plot dose-normalised AUC against dose. A horizontal line indicates linear kinetics; a
rising line indicates saturation. For a drug with Michaelis-Menten elimination, the apparent clearance
falls as concentration rises, and a dose increase produces a disproportionate exposure increase.

**Branch point.** If non-linearity is present, all the linear formulas in Steps 5 to 14 are local
approximations and must be replaced by simulation.

### Step 16 — Adjust for renal function

**Input.** Creatinine clearance or eGFR and the fraction of drug cleared renally.

**Output.** An adjusted maintenance dose.

**Reasoning.** For a drug cleared entirely by the kidney, CL is proportional to creatinine clearance.
For a drug cleared partly by the kidney, CL = CL_nonrenal + CL_renal × (CrCl/100). The adjustment
applies to the maintenance dose, not the loading dose.

**Branch point.** If the eGFR is indexed to 1.73 m², de-index it before use: absolute eGFR =
indexed eGFR × BSA/1.73. The FDA's 2024 renal impairment guidance recommends using eGFR in mL/min, not
mL/min/1.73 m², for dosing recommendations.

### Step 17 — Adjust for hepatic function

**Input.** Liver function and the fraction metabolised.

**Output.** A qualitative or quantitative adjustment.

**Reasoning.** Hepatic clearance depends on liver blood flow and intrinsic enzyme activity. Cirrhosis
reduces both, and it also reduces albumin and increases the free fraction of highly bound drugs. There
is no single equivalent of creatinine clearance for the liver, so adjustments are drug-specific and
often qualitative.

**Branch point.** If the drug is highly extracted and the route of administration is oral, cirrhosis
can increase bioavailability substantially by reducing first-pass metabolism, even as it reduces
systemic clearance.

### Step 18 — Adjust for body size

**Input.** Weight, height, sex, and the drug's distribution characteristics.

**Output.** A dosing weight.

**Reasoning.** Volume scales with weight, but not linearly in obesity. For many drugs, Vd is better
predicted by ideal body weight plus a fraction of excess weight (adjusted body weight) than by total
body weight. For highly lipophilic drugs, total body weight may be appropriate. The choice is
drug-specific and must be stated.

**Branch point.** If the drug's labelling specifies a weight basis, use it. If not, use a population
model with a body-size covariate rather than a rule of thumb.

### Step 19 — Adjust for age and development

**Input.** Age and developmental stage.

**Output.** A paediatric or geriatric adjustment.

**Reasoning.** Neonates have immature clearance pathways and higher body water; children have higher
weight-normalised clearance for many drugs; older adults have reduced renal clearance and altered body
composition. Adult doses scaled by weight are not paediatric doses.

**Branch point.** If the patient is outside the studied age range, the calculation is an extrapolation.
State it and monitor.

### Step 20 — Assess drug interactions

**Input.** Concomitant medications.

**Output.** A predicted change in exposure and an adjustment.

**Reasoning.** Metabolic interactions are classified by the AUC ratio: a strong inhibitor increases AUC
at least fivefold, a moderate inhibitor two- to fivefold, and a weak inhibitor 1.25- to twofold. A
strong inducer decreases AUC by at least 80%, a moderate inducer by 50% to 80%, and a weak inducer by
20% to 50%. These classes are defined in ICH M12 (2024).

**Branch point.** If the interacting drug affects a pathway that contributes only part of the total
clearance, the exposure change is smaller than the class suggests. The fraction metabolised by the
affected pathway determines the magnitude.

### Step 21 — Assess protein binding

**Input.** Free fraction and the binding protein concentration.

**Output.** An interpretation of total versus free concentration.

**Reasoning.** Only unbound drug is pharmacologically active and available for clearance. For highly
bound drugs, a change in albumin changes the free fraction and therefore both effect and clearance.
Total concentration can be misleading in liver disease, malnutrition, and pregnancy.

**Branch point.** If the drug is highly bound and the binding protein is abnormal, measure the free
concentration or interpret the total with caution.

### Step 22 — Assess extracorporeal clearance

**Input.** Dialysis or CRRT settings.

**Output.** An additional clearance term.

**Reasoning.** Intermittent haemodialysis removes drug during sessions, producing a sawtooth
concentration profile. CRRT removes drug continuously and contributes a steady clearance that can be a
substantial fraction of total clearance. The dialyser, blood flow, dialysate flow, and filter life all
affect the removal.

**Branch point.** If the patient is on CRRT, the renal adjustment for native kidney function is not
sufficient; the extracorporeal clearance must be added.

### Step 23 — Design the monitoring schedule

**Input.** The target, the assay, and the dosing interval.

**Output.** Sampling times.

**Reasoning.** A trough is drawn immediately before the next dose; a peak is drawn at a specified time
after the end of the infusion, and the time is part of the result. For AUC-guided dosing, sampling can
be sparse with a Bayesian estimator, or multiple samples can be used with the trapezoidal rule.

**Branch point.** If the sample time is unknown, the concentration cannot be interpreted as a peak or
trough. Treat it as a random concentration and use a population model to interpret it.

### Step 24 — Interpret the monitoring result

**Input.** Concentration, sampling time, dosing history, and assay.

**Output.** A dose recommendation.

**Reasoning.** The interpretation requires the time since the last dose, the dose and interval, and
the assay's specificity. A concentration interpreted without these is a number without a context. The
result should be compared with the target for the indication, not with a generic reference range.

**Branch point.** If the concentration is inconsistent with the dosing history, suspect a recording
error, a sampling error, or a non-linear process before adjusting the dose.

### Step 25 — Individualise with a Bayesian approach when data allow

**Input.** Prior population parameters and one or more concentrations.

**Output.** A posterior estimate of the individual's parameters.

**Reasoning.** A Bayesian estimator combines the population prior with the observed concentrations to
estimate individual clearance and volume. It allows sparse sampling and explicit uncertainty. The
prior must be appropriate to the population, and the model must be checked against the observed data.

**Branch point.** If the model fits poorly, do not individualise on it. Re-examine the dosing history
and the sample times first.

### Step 26 — Link exposure to effect

**Input.** AUC, MIC, or concentration, and the PD target.

**Output.** A probability of target attainment.

**Reasoning.** For concentration-dependent antibiotics, the target is often AUC/MIC or peak/MIC; for
time-dependent antibiotics, it is the fraction of the interval above MIC. The target is
drug-, organism-, and indication-specific. A dose that achieves the target in the average patient may
fail in the upper half of the clearance distribution.

**Branch point.** If the target attainment is below the accepted threshold, increase the dose or
change the interval, and re-check the toxicity constraint.

### Step 27 — Record the calculation and re-evaluate when the state changes

**Input.** All parameters and their sources, and any change in renal function, weight, co-medication,
or clinical status.

**Output.** A reproducible dosing note and a revised dose when the state changes.

**Reasoning.** The note must state the model, the parameters and their sources, the weight basis, the
renal function estimate and its equation, the assumed bioavailability, and the target. Clearance and
volume then change with acute kidney injury, fluid resuscitation, amputation, pregnancy, and recovery,
so a dose calculated on admission may be wrong within days.

**Branch point.** If a parameter is a guess, say so. If the change is large and the drug has a narrow
therapeutic index, re-dose and re-monitor rather than waiting for the next scheduled review.

### 5.1 The branch map

```text
Dosing question
  |
  +-- immediate post-bolus effect? ---> two-compartment
  |
  +-- oral route? --------------------> apply F and first-pass
  |
  +-- renal impairment? --------------> adjust maintenance, not loading
  |
  +-- indexed eGFR? ------------------> de-index before dosing
  |
  +-- obesity? -----------------------> choose weight basis explicitly
  |
  +-- interacting drug? --------------> AUC ratio class, fraction metabolised
  |
  +-- dialysis or CRRT? --------------> add extracorporeal clearance
  |
  +-- concentration available? -------> check sample time before interpreting
  |
  +-- non-linear kinetics? -----------> simulate, do not use linear formulas
```

### 5.2 The four checks that catch most PK errors

Check the units of k, the loading-maintenance distinction, the weight basis, and the sample time. Each
is a one-line check that catches a named failure mode from section 10.

## 6. Worked example

A 58-year-old man, 180.34 cm and 89.8113 kg, has a creatinine clearance of 73.06 mL/min estimated by
Cockcroft-Gault. He is prescribed gentamicin 5 mg/kg intravenously once daily for a serious
gram-negative infection. The population volume of distribution is 0.25 L/kg, and the elimination rate
constant is estimated from creatinine clearance by k = 0.00293·CrCl + 0.014.

The case computes the loading dose, the maintenance exposure, the concentration-time profile, the
accumulation under an every-8-hour regimen, the effect of renal impairment, the AUC by trapezoid, the
effect of a strong inhibitor, and the two-compartment refinement.

### 6.1 Volume, dose, and primary parameters

```text
Volume of distribution
  Vd = 0.25 L/kg x 89.8113 kg
     = 22.4528 L

Dose
  5 mg/kg x 89.8113 kg = 449.06 mg
  Rounded to the nearest milligram: 449 mg

Elimination rate constant
  k = 0.00293 /h per mL/min x 73.06 mL/min + 0.014 /h
    = 0.21407 + 0.014
    = 0.22807 /h

Half-life
  t1/2 = ln(2)/k = 0.693147/0.228066 = 3.0392 h

Clearance
  CL = k x Vd = 0.228066 x 22.4528 = 5.1207 L/h

Peak concentration after a bolus
  Cmax = dose/Vd = 449/22.4528 = 19.9975 mg/L

Exposure
  AUC = dose/CL = 449/5.1207 = 87.683 mg*h/L
```

The units are worth checking: mg divided by L gives mg/L; L/h multiplied by mg/L gives mg/h, which is
a mass rate; mg divided by L/h gives mg·h/L, the unit of AUC. The chain is dimensionally consistent,
which is the Topic 04 check applied to PK.

The AUC/MIC ratio at an MIC of 1 mg/L is 87.7 h. For once-daily aminoglycosides, an AUC over 24 hours
of 70 to 100 mg·h/L is a commonly cited efficacy target for serious gram-negative infection, so this
dose is at the lower end of the range.

### 6.2 Concentrations after a single dose

```text
C(8 h)  = 19.9975 x e^(-0.228066 x 8)
        = 19.9975 x e^(-1.82453)
        = 19.9975 x 0.16127
        = 3.2255 mg/L

C(12 h) = 19.9975 x e^(-2.73679)
        = 19.9975 x 0.06478
        = 1.2954 mg/L

C(24 h) = 19.9975 x e^(-5.47358)
        = 19.9975 x 0.00419
        = 0.0839 mg/L
```

After 24 hours, 99.6% of the dose has been eliminated. The concentration is below the typical assay's
lower limit of quantification, which is why a 24-hour sample is not useful for this drug at this
clearance.

### 6.3 Accumulation under an every-8-hour regimen

```text
Accumulation factor
  = 1/(1 - e^(-k x tau))
  = 1/(1 - e^(-0.228066 x 8))
  = 1/(1 - 0.16127)
  = 1/0.83873
  = 1.1923

Steady-state peak
  = 19.9975 x 1.1923 = 23.8433 mg/L

Steady-state trough
  = 23.8433 x e^(-1.82453)
  = 23.8433 x 0.16127
  = 3.8458 mg/L
```

An every-8-hour regimen produces a trough of 3.85 mg/L, which is above the usual target of below
1 mg/L for gentamicin and carries a risk of nephrotoxicity and ototoxicity. The once-daily regimen
avoids this accumulation while achieving a higher peak, which is the rationale for extended-interval
dosing of concentration-dependent antibiotics.

### 6.4 Renal impairment

If the same patient's creatinine clearance falls to 30 mL/min:

```text
k = 0.00293 x 30 + 0.014
  = 0.0879 + 0.014
  = 0.1019 /h

t1/2 = 0.693147/0.1019 = 6.802 h

C(8 h)  = 19.9975 x e^(-0.8152) = 19.9975 x 0.4426 = 8.8499 mg/L
C(24 h) = 19.9975 x e^(-2.4456) = 19.9975 x 0.0866 = 1.7333 mg/L

Accumulation at q8h
  = 1/(1 - e^(-0.1019 x 8))
  = 1/(1 - 0.4426)
  = 1.7939
```

The half-life doubles, the 24-hour concentration rises by a factor of about 20, and the accumulation
factor rises from 1.19 to 1.79. The maintenance dose must fall, but the loading dose does not: the
volume of distribution is unchanged, so 449 mg is still required to reach a peak near 20 mg/L.

This is the calculation behind the rule "adjust the maintenance dose, not the loading dose." It is
also the calculation that is skipped when a loading dose is reduced in renal impairment, leaving the
patient subtherapeutic for days.

### 6.5 Loading dose independence

```text
LD = Vd x target Cmax
   = 22.4528 L x 20.0 mg/L
   = 449.06 mg
```

The loading dose depends only on volume and target. Renal function does not appear. This is the
cleanest demonstration in the topic that the two doses answer different questions.

### 6.6 AUC by trapezoid and the sampling artefact

```text
Samples at 0, 0.25, 0.5, 1, 2, 4, 8, 12, 24 h
Trapezoid AUC(0-24) = 92.7243 mg*h/L
Analytic  AUC(0-24) = 87.3150 mg*h/L
Ratio = 1.0620
Analytic  AUC(0-inf) = 87.6830 mg*h/L
```

The trapezoid overestimates by 6.2% because the mono-exponential decline after a bolus is convex, and
the chord between samples lies above the curve. In real data the opposite error is more common: a
distribution phase makes the early concentrations fall faster than a single exponential, and sparse
early sampling then underestimates the true AUC. The direction of the error is a property of the curve
and the sampling schedule, not a constant.

The practical rule is to report the sampling schedule with the AUC, because the AUC is a property of
the data and the method, not only of the patient.

### 6.7 Drug interaction

A strong CYP3A4 inhibitor is added, and the drug is assumed to be cleared entirely by CYP3A4.

```text
AUC ratio = 5.0 (strong inhibitor, per ICH M12)
New AUC = 87.683 x 5.0 = 438.41 mg*h/L
Dose reduction to match exposure = 1/5 = 0.20 of the original
```

If the drug were cleared only 50% by CYP3A4, the expected AUC ratio would be
1/(0.5 + 0.5/5) = 1/(0.6) = 1.67, not 5.0. The fraction metabolised by the affected pathway determines
the magnitude of the interaction, which is why the class label (strong, moderate, weak) is a property
of the inhibitor, while the exposure change is a property of the pair.

### 6.8 The two-compartment refinement

For a drug with a distribution phase, the concentration-time curve is biexponential.

```text
C(t) = 12.0 x e^(-3.0 t) + 8.0 x e^(-0.08 t)

C(0.25 h) = 12.0 x 0.4724 + 8.0 x 0.9802 = 5.668 + 7.842 = 13.510 mg/L
C(1.00 h) = 12.0 x 0.0498 + 8.0 x 0.9231 = 0.598 + 7.385 = 7.982 mg/L
C(8.00 h) = 12.0 x 3.35e-11 + 8.0 x 0.5273 = 0.000 + 4.218 = 4.218 mg/L
C(24.0 h) = 12.0 x 1.0e-31 + 8.0 x 0.1466 = 0.000 + 1.173 = 1.173 mg/L

alpha half-life = ln(2)/3.0 = 0.231 h
beta half-life  = ln(2)/0.08 = 8.664 h
```

The concentration falls by 41% in the first hour and then declines with an 8.7-hour terminal half-life.
A one-compartment fit to the whole curve would overestimate the terminal half-life or misestimate the
volume, depending on the sampling. This is why the sampling schedule must cover the phase of interest:
early samples for the peak and distribution, late samples for the terminal half-life.

### 6.9 De-indexing an eGFR before dosing

The patient's indexed eGFR is 58 mL/min/1.73 m² and his body surface area is 2.1211 m².

```text
Absolute eGFR = indexed eGFR x BSA/1.73
              = 58 x 2.1211/1.73
              = 58 x 1.2261
              = 71.11 mL/min
```

The absolute value is 71 mL/min, not 58. Using the indexed value directly would underestimate clearance
by about 18% and produce a dose that is too low. The FDA's 2024 renal impairment guidance recommends
using eGFR in mL/min, not mL/min/1.73 m², when making dosing recommendations, and multiplying by
BSA/1.73 to de-index.

For this patient the de-indexed value is close to the Cockcroft-Gault value of 73.06 mL/min, which is a
useful cross-check: the two estimates of renal function should agree within about 20% to 30% in a
patient of average body size, and a larger discrepancy suggests an error in one of the inputs.

### 6.10 What the example demonstrates

| Calculation | Result | Concept |
|---|---|---|
| Vd from weight | 22.4528 L | Volume is a proportionality constant |
| k from CrCl | 0.228066 /h | Clearance depends on renal function |
| Half-life | 3.039 h | Derived from k |
| Cmax | 19.9975 mg/L | Dose divided by volume |
| AUC | 87.683 mg·h/L | Dose divided by clearance |
| Accumulation q8h | 1.1923 | Trough 3.85 mg/L, above target |
| Renal impairment | t½ 6.802 h | Maintenance falls, loading does not |
| Trapezoid AUC | +6.2% | Sampling and curve shape determine the error |
| Interaction | AUC × 5 | Fraction metabolised determines the magnitude |
| Two compartments | beta t½ 8.664 h | Phase matters for sampling |
| De-indexed eGFR | 71.11 mL/min | Indexed values are not doses |

## 7. Code, query, or calculation

The script reproduces every number in section 6 with the standard library.

```python
"""Topic 06 verification: one-compartment gentamicin kinetics, renal adjustment, DDI."""
from math import exp, log

print("6.1 One-compartment parameters")
kg = 89.8113
vd_per_kg = 0.25
vd = vd_per_kg * kg
dose = round(5 * kg)
crcl = 73.06
print(f"  Vd  = {vd_per_kg} L/kg x {kg:.4f} kg = {vd:.4f} L")
print(f"  dose = 5 mg/kg x {kg:.4f} kg = {5*kg:.2f} -> {dose} mg")

k = 0.00293 * crcl + 0.014
t_half = log(2) / k
cl = k * vd
cmax = dose / vd
auc = dose / cl
print(f"  k   = 0.00293 x {crcl:.2f} + 0.014 = {k:.6f} /h")
print(f"  t1/2 = {t_half:.4f} h")
print(f"  CL  = k x Vd = {cl:.4f} L/h")
print(f"  Cmax = dose/Vd = {cmax:.4f} mg/L")
print(f"  AUC  = dose/CL = {auc:.4f} mg*h/L")

print("\n6.2 Concentrations after a single dose")
for t in (8, 12, 24):
    c = cmax * exp(-k * t)
    print(f"  C({t:2d} h) = {c:.4f} mg/L")

print("\n6.3 Accumulation with q8h dosing")
tau = 8.0
acc = 1 / (1 - exp(-k * tau))
print(f"  accumulation factor = 1/(1 - e^(-k*tau)) = {acc:.4f}")
print(f"  steady-state peak = {cmax*acc:.4f} mg/L")
print(f"  steady-state trough = {cmax*acc*exp(-k*tau):.4f} mg/L")

print("\n6.4 Renal impairment, CrCl 30 mL/min")
crcl2 = 30.0
k2 = 0.00293 * crcl2 + 0.014
t2 = log(2) / k2
print(f"  k = {k2:.6f} /h, t1/2 = {t2:.3f} h")
print(f"  C(8 h)  = {cmax*exp(-k2*8):.4f} mg/L")
print(f"  C(24 h) = {cmax*exp(-k2*24):.4f} mg/L")
print(f"  accumulation q8h = {1/(1-exp(-k2*8)):.4f}")

print("\n6.5 Loading dose is independent of renal function")
target_peak = 20.0
ld = vd * target_peak
print(f"  LD = Vd x target Cmax = {vd:.4f} x {target_peak} = {ld:.2f} mg")

print("\n6.6 AUC by trapezoid from simulated samples")
times = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
conc = [cmax * exp(-k * t) for t in times]
auc_trap = 0.0
for i in range(1, len(times)):
    auc_trap += (conc[i-1] + conc[i]) / 2 * (times[i] - times[i-1])
auc_24 = (cmax / k) * (1 - exp(-k * 24))
print(f"  trapezoid AUC(0-24) = {auc_trap:.4f} mg*h/L")
print(f"  analytic AUC(0-24)  = {auc_24:.4f} mg*h/L")
print(f"  ratio               = {auc_trap/auc_24:.4f}")
print(f"  analytic AUC(0-inf) = {auc:.4f} mg*h/L")

print("\n6.7 Drug interaction")
auc_ratio = 5.0
print(f"  strong CYP3A inhibitor, AUC ratio {auc_ratio}")
print(f"  new AUC = {auc*auc_ratio:.2f} mg*h/L")
print(f"  dose reduction to match exposure = {1/auc_ratio:.2f} of the original")

print("\n6.8 Two-compartment illustration")
alpha, beta = 3.0, 0.08
A, B = 12.0, 8.0
for t in (0.25, 1.0, 8.0, 24.0):
    c = A * exp(-alpha * t) + B * exp(-beta * t)
    print(f"  C({t:5.2f} h) = {c:.4f} mg/L")
print(f"  alpha half-life = {log(2)/alpha:.3f} h")
print(f"  beta half-life  = {log(2)/beta:.3f} h")
```

**Expected output.**

```text
6.1 One-compartment parameters
  Vd  = 0.25 L/kg x 89.8113 kg = 22.4528 L
  dose = 5 mg/kg x 89.8113 kg = 449.06 -> 449 mg
  k   = 0.00293 x 73.06 + 0.014 = 0.228066 /h
  t1/2 = 3.0392 h
  CL  = k x Vd = 5.1207 L/h
  Cmax = dose/Vd = 19.9975 mg/L
  AUC  = dose/CL = 87.6830 mg*h/L

6.2 Concentrations after a single dose
  C( 8 h) = 3.2255 mg/L
  C(12 h) = 1.2954 mg/L
  C(24 h) = 0.0839 mg/L

6.3 Accumulation with q8h dosing
  accumulation factor = 1/(1 - e^(-k*tau)) = 1.1923
  steady-state peak = 23.8433 mg/L
  steady-state trough = 3.8458 mg/L

6.4 Renal impairment, CrCl 30 mL/min
  k = 0.101900 /h, t1/2 = 6.802 h
  C(8 h)  = 8.8499 mg/L
  C(24 h) = 1.7333 mg/L
  accumulation q8h = 1.7939

6.5 Loading dose is independent of renal function
  LD = Vd x target Cmax = 22.4528 x 20.0 = 449.06 mg

6.6 AUC by trapezoid from simulated samples
  trapezoid AUC(0-24) = 92.7243 mg*h/L
  analytic AUC(0-24)  = 87.3150 mg*h/L
  ratio               = 1.0620
  analytic AUC(0-inf) = 87.6830 mg*h/L

6.7 Drug interaction
  strong CYP3A inhibitor, AUC ratio 5.0
  new AUC = 438.41 mg*h/L
  dose reduction to match exposure = 0.20 of the original

6.8 Two-compartment illustration
  C( 0.25 h) = 13.5100 mg/L
  C( 1.00 h) = 7.9824 mg/L
  C( 8.00 h) = 4.2183 mg/L
  C(24.00 h) = 1.1729 mg/L
  alpha half-life = 0.231 h
  beta half-life  = 8.664 h
```

**Assumptions stated by the script.**

1. The model is one compartment with first-order elimination. A two-compartment drug would require the
   biexponential form.
2. The volume of distribution is 0.25 L/kg of total body weight. Obesity may require adjusted body
   weight.
3. The elimination rate constant is predicted from Cockcroft-Gault creatinine clearance by a published
   relationship. It is a population estimate, not an individual measurement.
4. The dose is rounded to the nearest milligram; the rounding changes Cmax by 0.004 mg/L.
5. The interaction is assumed to affect the entire clearance. If only part of clearance is affected,
   the AUC ratio is smaller.
6. The two-compartment parameters in 6.8 are illustrative and are not derived from the one-compartment
   parameters.

## 8. Data, standards, and regulatory context

### 8.1 FDA guidance on renal impairment

The FDA's final guidance *Pharmacokinetics in Patients with Impaired Renal Function — Study Design,
Data Analysis, and Impact on Dosing* was issued in March 2024, finalising a 2020 draft. Its relevant
recommendations for data work are:

- A dedicated renal impairment study is warranted when at least 30% of the drug or an active
  metabolite is excreted unchanged in urine, and for certain therapeutic proteins, antibody-drug
  conjugates, and drugs likely to be used with dialysis.
- Renal function should be assessed with eGFR using a contemporary equation such as CKD-EPI.
- **When recommending doses, use eGFR in mL/min, not mL/min/1.73 m²**, multiplying by BSA/1.73 to
  de-index. Cockcroft-Gault creatinine clearance remains acceptable, using ideal or adjusted body
  weight in obese patients.
- Model renal function as a continuous variable, with a categorical analysis as a supplement.
- Base dose recommendations on exposure matching, for example within the 5th to 95th percentile of the
  reference group, or on an established no-effect boundary.
- For dialysis, include on- and off-dialysis periods and record the dialyser and flow settings; for
  CRRT, capture pre-filter, post-filter, and effluent samples.

The de-indexing requirement is the regulatory expression of the Topic 03 point that an indexed
quantity is not a plain quantity. An eGFR of 58 mL/min/1.73 m² is not 58 mL/min, and the difference is
18% for a patient with a BSA of 2.12 m².

### 8.2 ICH M12 drug interaction studies

**ICH M12**, *Drug Interaction Studies*, reached Step 4 on 21 May 2024 and replaced earlier regional
guidances. It defines the in vitro and clinical approach to metabolic and transporter-mediated
interactions.

The in vitro decision rules include:

- Reversible CYP inhibition can be excluded when the unbound Ki exceeds 50 times the unbound Cmax
  (Cmax,u/Ki,u < 0.02).
- For orally administered CYP3A inhibitors, intestinal inhibition can be excluded when the unbound
  IC50 exceeds 0.1 times the dose divided by 250 mL.
- Time-dependent inhibition is assessed with KI and kinact, and risk can be excluded when
  kobs/kdeg < 1.25.
- Induction is assessed in hepatocytes from at least three donors, with CYP3A4, 2B6, and 1A2 as
  mandatory markers.

The clinical classification of inhibitors is defined by the AUC ratio: strong at least 5, moderate 2 to
less than 5, weak 1.25 to less than 2. Inducers are classified by the percentage decrease in AUC:
strong at least 80%, moderate 50% to less than 80%, weak 20% to less than 50%. These are the classes
used in section 6.7.

### 8.3 Other applicable standards

| Standard | Scope | Binding status |
|---|---|---|
| ICH E11(R1) | Clinical investigation of medicinal products in the paediatric population | Binding for ICH-region paediatric submissions |
| ICH E9(R1) | Estimands and sensitivity analysis | Binding for ICH-region submissions |
| FDA bioequivalence guidance | Exposure similarity of formulations | Binding for generic approvals |
| EMA guideline on PK in renal impairment | European equivalent of the FDA guidance | Binding in the EU |
| CLSI EP05 and EP15 | Assay precision for the TDM assay | Voluntary; expected in accreditation |
| ISO 15197 | Accuracy of blood glucose monitors | Voluntary; relevant for glucose-lowering drugs |

### 8.4 What the standards do not constrain

- They do not provide the population parameters. Vd and CL come from the label or the literature, and
  their applicability to a specific patient is a clinical judgement.
- They do not validate a dosing nomogram. A nomogram is a model, and it requires local validation.
- They do not resolve the choice between Cockcroft-Gault and CKD-EPI. The FDA accepts both; the choice
  must be stated and applied consistently.
- They do not cover every interaction. M12 defines the study design and the classification, not the
  full list of pairwise effects.
- They do not replace TDM. A model predicts; a measurement confirms.

> **Note:** The regulatory content here is educational. Dosing decisions require clinical
> pharmacological review and, where applicable, regulatory agreement.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Hospital antimicrobial stewardship programmes** use AUC-guided dosing for vancomycin, with Bayesian
estimation from one or two concentrations, and extended-interval dosing for aminoglycosides. The 2020
ASHP/IDSA/PIDS/SIDP vancomycin guideline recommends AUC-guided dosing over trough-only monitoring for
serious MRSA infection, with a target AUC over 24 hours of 400 to 600 mg·h/L. The change from trough to
AUC monitoring is a documented example of a PK target replacing a surrogate.

**Pharmaceutical sponsors** conduct dedicated renal and hepatic impairment studies, drug interaction
studies, and paediatric studies under the ICH framework, and they build population PK models to
support labelling. The models are submitted and reviewed, and their covariate effects become the
dosing instructions.

**Clinical pharmacology software vendors** implement Bayesian dose individualisation and interaction
checking. The quality of the software depends on the priors, the covariate models, and the handling of
sample times, which is why validation against a published dataset is part of procurement.

**Payers and health technology assessors** use exposure-response models to support dose selection and
to compare regimens. A model that ignores renal function will overstate the value of a drug in a
population with a high prevalence of kidney disease.

### 9.2 Case evidence

**Vancomycin AUC-guided dosing.** The 2020 consensus guideline (Rybak et al., *American Journal of
Health-System Pharmacy* 77(11):835–864) recommends AUC-guided dosing for serious MRSA infections,
citing evidence that AUC monitoring reduces nephrotoxicity while maintaining efficacy compared with
trough-only monitoring. The guideline is a documented case of a PK metric replacing a simpler surrogate
because the surrogate did not predict toxicity well.

**Aminoglycoside extended-interval dosing.** Once-daily dosing of aminoglycosides exploits the
concentration-dependent killing and the post-antibiotic effect, achieving a high peak and a low trough.
The practice is supported by meta-analyses showing comparable efficacy and reduced nephrotoxicity
compared with multiple daily dosing. The PK reasoning is the accumulation calculation in section 6.3:
a short interval produces a trough of 3.85 mg/L, while a long interval allows the concentration to fall
below the toxic threshold.

**Phenytoin non-linearity.** Phenytoin follows Michaelis-Menten kinetics, so a dose increase from
300 mg to 400 mg per day can produce a concentration increase far greater than 33%, sometimes crossing
from therapeutic to toxic. The case is a standard example in clinical pharmacology and a reminder that
linear formulas fail near saturation.

**CYP3A4 interactions.** The class of strong CYP3A4 inhibitors, including clarithromycin, itraconazole,
and ritonavir, produces AUC increases of fivefold or more for sensitive substrates. The ICH M12
classification in section 8.2 is the codification of decades of interaction studies, and the
classification is used by interaction-checking software.

> **Note:** The guideline recommendations and the phenytoin example are cited to published sources. The
> interpretation of each as a data-science or software-validation lesson is inference, not a claim
> about a specific institution.

### 9.3 The practical pattern

Teams that avoid PK errors do four things:

1. They record the parameter source and the population for every calculation.
2. They distinguish loading from maintenance in the code, in the variable names, and in the review.
3. They de-index renal function before dosing and state the equation used.
4. They record the sample time with every concentration and refuse to interpret a level without it.

Each practice is a control on a specific failure mode, and each is checkable without a clinical
degree.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Loading dose reduced in renal impairment | A renal adjustment is applied to every dose | Loading depends on volume, not clearance | Delayed therapeutic concentration | Separate loading and maintenance in the code and the note |
| 2 | Indexed eGFR used as absolute clearance | eGFR reported in mL/min/1.73 m² and entered directly | Indexation is ignored | Under-dose in large patients, over-dose in small ones | De-index with BSA/1.73; state the equation |
| 3 | Weight basis unspecified | A mg/kg dose with no statement of total, ideal, or adjusted weight | Different bases give different doses | Under- or over-dose in obesity | Name the weight basis in the parameter |
| 4 | Half-life quoted without its population | A single half-life used for all patients | Half-life depends on volume and clearance | Wrong interval and time to steady state | Use a population model with covariates |
| 5 | Steady state assumed too early | A level drawn after two half-lives is interpreted as a trough | Steady state takes 4–5 half-lives | Falsely low level, dose increased | Wait 4–5 half-lives or use a loading dose |
| 6 | Sample time not recorded | A concentration is labelled "trough" without a time | A level at any other time is not a trough | Dose changed on a misinterpreted value | Record time since dose; reject unlabelled levels |
| 7 | Peak sampled at the wrong time | A level drawn immediately after a short infusion | Distribution is not complete | Falsely high peak, dose reduced | Standardise the sampling time after the infusion |
| 8 | Linear model applied to a saturable drug | Dose-normalised AUC rises with dose | Michaelis-Menten kinetics | Disproportionate concentration increase | Plot dose-normalised AUC; simulate |
| 9 | Interaction class applied without the fraction metabolised | A "strong inhibitor" is assumed to multiply AUC by 5 for every drug | Only the affected pathway changes | Over- or under-adjustment | Compute the expected ratio from the fraction |
| 10 | Protein binding ignored in liver disease | Total concentration is normal but the patient is toxic | Free fraction rises as albumin falls | Toxicity at a "normal" total level | Measure free drug or interpret with albumin |
| 11 | Dialysis clearance ignored | A patient on CRRT is dosed as if anuric | Extracorporeal clearance is substantial | Subtherapeutic concentrations | Add extracorporeal clearance to CL |
| 12 | First-pass metabolism ignored for oral dosing | An oral dose is calculated from IV parameters without F | Bioavailability below 1 | Overdose by 1/F | Apply F; use the correct route's parameters |
| 13 | Paediatric dose scaled linearly from adult | A mg/kg dose from adults is applied to a neonate | Immature clearance pathways | Toxicity or failure | Use age-specific models; monitor |
| 14 | TDM result interpreted without the assay | A level is compared with a target from a different assay | Assay specificity differs | Systematic bias in dose adjustment | Use assay-specific targets |
| 15 | Non-linear accumulation assumed linear | Trough rises more than predicted | Saturation | Unexpected toxicity | Simulate at the current dose |
| 16 | Units of k omitted | k reported as 0.23 without a time basis | 1/h versus 1/day | Factor-of-24 error | Name the unit in the variable and the output |

Two rows deserve emphasis.

**Row 2, indexation.** This is the most common PK error in software. An eGFR field labelled
"mL/min/1.73 m²" is often consumed by a function that expects mL/min, and the difference is 15% to 40%
depending on body size. The FDA's 2024 guidance addresses it explicitly, which is a signal that the
error is common enough to regulate.

**Row 9, interaction magnitude.** The class label describes the inhibitor, not the pair. A strong
inhibitor of CYP3A4 will multiply the AUC of a drug that is 100% cleared by CYP3A4 by about 5, but a
drug that is 20% cleared by CYP3A4 will rise by only about 1.2-fold. Applying the class label without
the fraction metabolised over-adjusts and can leave the patient subtherapeutic.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The body behaves as one (or two) well-stirred compartments | Simplicity and identifiability | Tissue-specific distribution, active transport | Compare model predictions with measured concentrations |
| 2 | Elimination is first-order | Half-life, accumulation, and AUC formulas | Saturable metabolism, zero-order drugs | Dose-normalised AUC plot |
| 3 | Parameters are drawn from a comparable population | Transportability of Vd and CL | Different age, weight, disease, or ethnicity | Compare covariates with the source population |
| 4 | Bioavailability is known for the route | Converting dose to exposure | Formulation differences, food effects | Use route-specific data or measure |
| 5 | Protein binding is stable | Total concentration proxies free concentration | Liver disease, malnutrition, pregnancy, displacement | Measure free fraction when binding is abnormal |
| 6 | Renal function is accurately estimated | Dosing in renal impairment | Non-steady creatinine, extreme body size, drugs blocking secretion | Use a measured clearance or a cystatin C equation |
| 7 | The weight basis matches the label | Dose per kg | Obesity, oedema, amputation | Use the labelled basis; document deviations |
| 8 | The sampling time is accurate | Interpretation of peak and trough | Delayed or undocumented sampling | Record time since dose; use a Bayesian model |
| 9 | The assay measures the parent drug specifically | TDM target validity | Metabolite cross-reactivity | Use an assay-specific target |
| 10 | The interaction affects only the assumed pathway | Interaction magnitude | Multiple pathways, transporters, protein binding | Use a PBPK model or a clinical study |
| 11 | Steady state has been reached | Trough interpretation | Long half-life, recent dose change | Confirm with two levels or a model |
| 12 | Extracorporeal clearance is either negligible or quantified | Dosing on dialysis | CRRT, high-flux dialysers | Measure pre- and post-filter concentrations |
| 13 | The PD target is valid for the indication | Exposure-response link | Different organism, site, or severity | Use indication-specific targets |
| 14 | Adherence matches the recorded dosing history | Interpreting levels | Missed doses, delayed administration | Reconcile the administration record |

The limits of this topic:

- It does not cover physiologically based pharmacokinetic modelling in depth. PBPK is a specialty with
  its own validation standards.
- It does not cover pharmacodynamics beyond the exposure-response targets used in dosing.
- It does not cover pharmacogenomics in detail. Genotype-guided dosing is mentioned as a covariate
  source, not developed.
- It does not provide dosing recommendations. The numbers are illustrative.
- It does not cover every route or formulation. It covers the concepts that generalise.

## 12. Exercises

### Exercise 1 — ADME and parameters (recall)

For each process, name the parameter that quantifies it, the organ or system responsible, and one
clinical condition that alters it: absorption; distribution; metabolism; excretion.

**Deliverable.** A four-row table.

**Acceptance criteria.** Absorption: bioavailability F, gut and liver first pass, co-administration with
food or antacids; distribution: Vd, tissue binding and body composition, obesity or pregnancy;
metabolism: CL_met, liver CYP enzymes, cirrhosis or enzyme induction; excretion: CL_renal, kidney,
acute kidney injury. Each row names a measurable parameter.

**Time.** 25 minutes.

### Exercise 2 — Compute the parameters (calculation)

A 500 mg intravenous bolus produces a concentration of 25 mg/L at time zero and 12.5 mg/L at 8 hours.
Compute Vd, k, half-life, clearance, and AUC(0-inf).

**Deliverable.** Five values with units and the formulas.

**Acceptance criteria.** Vd = 20 L; k = 0.08664 /h; t½ = 8.00 h; CL = 1.733 L/h; AUC = 288.6 mg·h/L;
each value derived from the previous ones where appropriate.

**Time.** 25 minutes.

### Exercise 3 — Trapezoid versus analytic AUC (calculation)

Using the same data, compute AUC(0-8) by the trapezoidal rule and compare it with the analytic
AUC(0-8). Explain the direction of the difference.

**Deliverable.** Two values, a ratio, and a two-sentence explanation.

**Acceptance criteria.** Trapezoid 150 mg·h/L; analytic 144.3 mg·h/L; ratio 1.039; the explanation states
that for a single exponential the trapezoid is exact only when the sampling matches the exponential, and
that in general the chord lies above a convex curve.

**Time.** 20 minutes.

### Exercise 4 — Loading and maintenance (calculation)

A drug has Vd 30 L, CL 4 L/h, and F 1.0. The target peak is 15 mg/L and the target AUC over 24 hours is
100 mg·h/L. Compute the loading dose and the total daily maintenance dose.

**Deliverable.** Two doses and the reasoning.

**Acceptance criteria.** LD = 450 mg; maintenance = 400 mg per 24 h; the answer states that the loading
dose is independent of clearance and the maintenance dose is proportional to it.

**Time.** 25 minutes.

### Exercise 5 — Accumulation and steady state (calculation)

A drug has a half-life of 12 hours. Compute the accumulation factor and the time to 90% of steady state
for dosing every 12 hours and every 24 hours.

**Deliverable.** Four values.

**Acceptance criteria.** q12h factor 2.00; q24h factor 1.333; time to 90% is 3.32 half-lives = 39.9 h
for both, independent of the interval; the answer explains why.

**Time.** 25 minutes.

### Exercise 6 — De-index an eGFR (calculation)

A patient has an indexed eGFR of 45 mL/min/1.73 m² and a body surface area of 1.90 m². Compute the
absolute eGFR. Then compute it for a BSA of 2.40 m² and state which patient is at greater risk of
under-dosing if the indexed value is used directly.

**Deliverable.** Two values and a one-sentence conclusion.

**Acceptance criteria.** 49.4 mL/min at BSA 1.90; 62.4 mL/min at BSA 2.40; the larger patient is at
greater risk because the indexed value underestimates absolute clearance by 28%.

**Time.** 20 minutes.

### Exercise 7 — Interpret a TDM result (application)

A trough sample labelled "12 hours post-dose" is reported as 18 mg/L. The drug has a half-life of
6 hours, and the target trough is below 10 mg/L. Confirm whether the sample is a true trough, and
compute the expected concentration at 12 hours if the level was actually drawn 4 hours after the dose.

**Deliverable.** A calculation and a recommendation.

**Acceptance criteria.** k = 0.1155 /h; expected 12-hour concentration from a 4-hour value of 18 is
18 × e^(−0.1155×8) = 7.15 mg/L; the sample is not a trough and the dose should not be reduced on it;
the recommendation is to redraw at the correct time.

**Time.** 30 minutes.

### Exercise 8 — Critique an interaction adjustment (critique)

A patient on drug X, which is 40% cleared by CYP3A4 and 60% by renal excretion, is started on a strong
CYP3A4 inhibitor. A colleague reduces the dose of X by 80%. Compute the expected AUC ratio and state
whether the reduction is appropriate.

**Deliverable.** An AUC ratio and a recommendation.

**Acceptance criteria.** Expected ratio = 1/(0.6 + 0.4/5) = 1/0.68 = 1.47; the 80% reduction is far too
large; the appropriate adjustment is a reduction of about 32% to maintain exposure, with monitoring.

**Time.** 30 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Process | Parameter | Organ or system | Condition that alters it |
|---|---|---|---|
| Absorption | Bioavailability F | Gut wall and liver first pass | Food, antacids, malabsorption |
| Distribution | Vd | Tissue binding, body composition | Obesity, pregnancy, ascites |
| Metabolism | CL_met | Liver, CYP enzymes | Cirrhosis, enzyme induction |
| Excretion | CL_renal | Kidney | Acute kidney injury, dialysis |

The parameters are not independent: a change in protein binding alters Vd and CL together, which is why
liver disease is difficult to dose without measurement.

### Answer to Exercise 2

```text
Vd = dose/C0 = 500 mg / 25 mg/L = 20 L

k = ln(C0/C8)/8 = ln(25/12.5)/8 = ln(2)/8 = 0.086643 /h

t1/2 = ln(2)/k = 0.693147/0.086643 = 8.00 h

CL = k x Vd = 0.086643 x 20 = 1.7329 L/h

AUC(0-inf) = dose/CL = 500/1.7329 = 288.6 mg*h/L
```

The half-life is exactly 8 hours because the concentration halved in 8 hours, which is the definition
of the half-life. The AUC can also be computed as C0/k = 25/0.086643 = 288.6 mg·h/L, which agrees.

### Answer to Exercise 3

```text
Trapezoid AUC(0-8) = (25 + 12.5)/2 x 8 = 18.75 x 8 = 150 mg*h/L

Analytic AUC(0-8) = (C0/k) x (1 - e^(-k x 8))
                  = (25/0.086643) x (1 - 0.5)
                  = 288.6 x 0.5
                  = 144.3 mg*h/L
```

The ratio is 150/144.3 = 1.039, not 1.000. The trapezoid overestimates by 3.9% because the chord
between two points lies above the convex exponential curve. With only two samples the error is largest;
with denser sampling it falls. The earlier expectation of an exact match was wrong, and the calculation
corrects it.

### Answer to Exercise 4

```text
Loading dose
  LD = Vd x target peak / F
     = 30 L x 15 mg/L / 1.0
     = 450 mg

Maintenance dose
  dose per 24 h = CL x target AUC(0-24) / F
                = 4 L/h x 100 mg*h/L / 1.0
                = 400 mg
```

The loading dose is 450 mg and the maintenance dose is 400 mg per day. The two are similar by
coincidence in this example; in general they are unrelated, because they depend on different
parameters. A patient with renal impairment would keep the 450 mg loading dose and reduce the 400 mg
maintenance dose.

### Answer to Exercise 5

```text
q12h, t1/2 = 12 h:
  accumulation = 1/(1 - e^(-k x 12)) = 1/(1 - 0.5) = 2.00

q24h, t1/2 = 12 h:
  e^(-k x 24) = e^(-0.05776 x 24) = e^(-1.386) = 0.25
  accumulation = 1/(1 - 0.25) = 1.333
```

The q24h factor is 1.333, not 1.414. A factor of 1.414 would correspond to a 24-hour interval with a
24-hour half-life, which is a different drug.

```text
Time to 90% of steady state:
  1 - 0.5^n = 0.90
  0.5^n = 0.10
  n = ln(0.10)/ln(0.5) = 3.322 half-lives
  = 3.322 x 12 = 39.9 h
```

The time to steady state depends only on the half-life, not on the dosing interval, because it is the
approach of the accumulation process, which has the same exponential as elimination.

### Answer to Exercise 6

```text
BSA 1.90 m2:
  absolute eGFR = 45 x 1.90/1.73 = 45 x 1.0983 = 49.42 mL/min

BSA 2.40 m2:
  absolute eGFR = 45 x 2.40/1.73 = 45 x 1.3873 = 62.43 mL/min
```

The larger patient is at greater risk of under-dosing. The indexed value of 45 understates the absolute
clearance by 28% at a BSA of 2.40 m², so a drug dosed on the indexed value would be under-dosed. The
de-indexing step is not optional for dosing.

### Answer to Exercise 7

```text
k = ln(2)/6 = 0.115525 /h

If the sample was drawn 4 h after the dose, the concentration at 12 h is
  C(12) = 18 x e^(-0.115525 x 8)
        = 18 x e^(-0.9242)
        = 18 x 0.3968
        = 7.14 mg/L
```

The sample is not a trough. A true trough at 12 hours would be 7.14 mg/L, which is below the 10 mg/L
target. The reported 18 mg/L is a mid-interval concentration and cannot be compared with the trough
target. The correct action is to redraw immediately before the next dose and not to reduce the dose on
the basis of this result.

### Answer to Exercise 8

```text
Fraction cleared by CYP3A4 = 0.40
Fraction cleared renally   = 0.60

With complete inhibition of CYP3A4:
  new CL = 0.60 x CL + 0.40 x CL/5
         = CL x (0.60 + 0.08)
         = 0.68 x CL

AUC ratio = CL_old/CL_new = 1/0.68 = 1.47
```

The expected AUC increase is 47%, not 400%. Reducing the dose by 80% would leave the patient at about
one fifth of the original exposure, which is likely subtherapeutic. The appropriate adjustment is a
reduction of about 32% (1 − 1/1.47 = 0.32) if exposure matching is the goal, with monitoring to
confirm. The example shows why the interaction class alone is insufficient: it describes the inhibitor,
not the pair.

## 14. Interview questions

### Question 1 — "What determines the average concentration at steady state?"

**Strong answer.** "Clearance and the dose rate. At steady state, input equals output, so average
concentration is dose rate divided by clearance, or AUC over the interval is dose divided by clearance.
Half-life affects the time to reach steady state and the fluctuation, not the average."

**Weak answer.** "The half-life." Half-life determines the approach and the peak-trough difference; it
is not the determinant of average exposure.

### Question 2 — "Why does renal impairment change the maintenance dose but not the loading dose?"

**Strong answer.** "The loading dose depends on volume of distribution, which is largely unchanged in
kidney disease. The maintenance dose depends on clearance, which falls. Reducing a loading dose delays
the achievement of therapeutic concentrations without protecting the patient."

**Weak answer.** "Because the kidneys clear the drug." That explains the maintenance dose, not the
distinction, which is the point of the question.

### Question 3 — "Why must an indexed eGFR be de-indexed before dosing?"

**Strong answer.** "Because the index is a normalisation to a standard body surface area of 1.73 m².
Dosing depends on absolute clearance, so the indexed value must be multiplied by BSA/1.73. For a patient
with a BSA of 2.4 m², the indexed value underestimates clearance by 28%."

**Weak answer.** "Because the units are different." They are different quantities, not just different
units.

### Question 4 — "What is the difference between clearance and half-life?"

**Strong answer.** "Clearance is the volume of plasma cleared per unit time and determines steady-state
exposure. Half-life is ln(2)·Vd/CL and determines the time course and the fluctuation. Two drugs can
share a half-life and differ in clearance if their volumes differ."

**Weak answer.** "They are inversely related." They are related through volume, and the relationship
matters for dosing.

### Question 5 — "How do you compute accumulation?"

**Strong answer.** "For first-order kinetics and a fixed interval, the accumulation factor is
1/(1 − e^(−k·τ)). It applies to peak and trough. For a 12-hour half-life and a 12-hour interval it is
2.0; for a 24-hour interval it is 1.33."

**Weak answer.** "Multiply by the number of doses." Accumulation is bounded by the exponential, not
linear in the number of doses.

### Question 6 — "A level is reported as a trough. What do you check?"

**Strong answer.** "The time since the last dose, the dosing history, whether steady state was reached,
and the assay. A trough is drawn immediately before the next dose at steady state. Without the time, the
value cannot be interpreted as a trough."

**Weak answer.** "Whether it is below the target." The comparison is meaningless if the sample is not a
trough.

### Question 7 — "How does a strong CYP3A4 inhibitor change exposure?"

**Strong answer.** "It depends on the fraction of the drug cleared by CYP3A4. If the drug is entirely
cleared by CYP3A4, AUC rises about fivefold. If 40% is cleared by CYP3A4 and 60% renally, the AUC rises
by about 1.47-fold. The class describes the inhibitor; the magnitude depends on the pair."

**Weak answer.** "It increases exposure fivefold." That is the maximum for a sensitive substrate, not a
general rule.

### Question 8 — "What is the difference between first-order and zero-order elimination?"

**Strong answer.** "In first-order elimination, a constant fraction is removed per unit time, so the
half-life is constant and AUC is proportional to dose. In zero-order elimination, a constant amount is
removed per unit time, so the half-life changes with concentration and small dose increases can produce
large concentration increases."

**Weak answer.** "Zero-order is when the drug is saturated." That is the mechanism; the consequence for
dosing is the point.

### Question 9 — "What does AUC/MIC mean and when is it the target?"

**Strong answer.** "It is the area under the 24-hour concentration curve divided by the minimum
inhibitory concentration, and it is the target for concentration-dependent antibiotics such as
vancomycin and aminoglycosides. For time-dependent antibiotics, the target is the fraction of the
interval above the MIC instead."

**Weak answer.** "It is a measure of potency." It is an exposure-response index, and its target is
drug- and organism-specific.

### Question 10 — "How would you validate a dosing calculator?"

**Strong answer.** "Against published worked examples and a reference dataset, with unit tests for the
loading-maintenance distinction, the de-indexing step, the weight basis, and the accumulation formula. I
would also test the failure modes: an indexed eGFR entered directly, a missing sample time, and a
non-linear drug."

**Weak answer.** "Check that it gives the same answer as the label." The label gives a dose for a
population, not a test case for an individual.

### Question 11 — "What is the difference between a one- and two-compartment model?"

**Strong answer.** "A one-compartment model treats the body as a single well-stirred volume and predicts
a mono-exponential decline. A two-compartment model adds a peripheral compartment and predicts a
biexponential decline: a fast distribution phase and a slower terminal phase. The terminal half-life
determines time to steady state; the distribution phase determines the early peak."

**Weak answer.** "Two compartments are more accurate." They are more accurate when the data show a
distribution phase, and unnecessary otherwise.

### Question 12 — "Why is therapeutic drug monitoring not a substitute for a model?"

**Strong answer.** "A measurement is a point observation with error, and it needs a model to interpret
it and to predict future concentrations. The model needs the measurement to individualise. The two are
complementary: the model provides the prior and the sampling design, and the measurement updates it."

**Weak answer.** "Monitoring is more accurate than a model." A single level without a model cannot
predict a trough or an AUC.

## 15. Key terms

- **Absorption** — movement of drug from the administration site into circulation.
- **Accumulation factor** — 1/(1 − e^(−k·τ)) for repeated dosing.
- **Adjusted body weight** — ideal body weight plus a fraction of excess weight.
- **AUC** — area under the concentration-time curve, mg·h/L.
- **AUC/MIC** — exposure relative to the minimum inhibitory concentration.
- **Bioavailability (F)** — fraction of a dose reaching systemic circulation.
- **Bioequivalence** — similar exposure from two formulations.
- **Clearance (CL)** — volume of plasma cleared of drug per time.
- **Cmax** — maximum concentration after a dose.
- **Cmin** — minimum concentration, usually the trough.
- **Compartment** — a kinetic pool of drug.
- **CRRT** — continuous renal replacement therapy.
- **CYP450** — the cytochrome P450 family of oxidative enzymes.
- **Distribution** — movement of drug between plasma and tissues.
- **Distribution half-life** — half-life of the early decline.
- **Dosing interval (τ)** — time between doses.
- **Elimination rate constant (k)** — fractional removal per time.
- **Excretion** — removal of drug or metabolites from the body.
- **First-pass effect** — metabolism during absorption before systemic circulation.
- **First-order kinetics** — elimination proportional to concentration.
- **Free fraction (fu)** — unbound fraction of drug.
- **Half-life (t½)** — time for concentration to halve.
- **Ideal body weight** — weight predicted from height and sex.
- **Inducer** — an agent that increases metabolic clearance.
- **Inhibitor** — an agent that reduces metabolic clearance.
- **Loading dose** — dose to reach a target concentration quickly.
- **Maintenance dose** — dose to replace cleared drug.
- **Michaelis-Menten** — saturable elimination described by Vmax and Km.
- **MIC** — minimum inhibitory concentration.
- **Narrow therapeutic index** — small difference between effective and toxic exposure.
- **One-compartment model** — body treated as a single well-stirred volume.
- **Peak/MIC** — peak concentration relative to MIC.
- **Pharmacodynamics** — what the drug does to the body.
- **Pharmacogenomics** — genetic influence on drug response and disposition.
- **Pharmacokinetics** — what the body does to a drug.
- **Protein binding** — fraction of drug bound to plasma proteins.
- **Steady state** — input equals elimination.
- **Terminal half-life** — half-life of the elimination phase.
- **Therapeutic drug monitoring** — measurement of concentration to guide dosing.
- **Therapeutic index** — ratio of toxic to effective exposure.
- **Time above MIC** — fraction of the interval above the MIC.
- **Two-compartment model** — central and peripheral compartments.
- **Vd** — apparent volume of distribution.
- **Zero-order kinetics** — elimination at a constant rate.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Regulatory guidance

- US Food and Drug Administration, *Pharmacokinetics in Patients with Impaired Renal Function — Study
  Design, Data Analysis, and Impact on Dosing*, final guidance, March 2024.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pharmacokinetics-patients-impaired-renal-function-study-design-data-analysis-and-impact-dosing
- ICH, *M12 Drug Interaction Studies*, Step 4, 21 May 2024.
  https://database.ich.org/sites/default/files/ICH_M12_Step4_Guideline_2024_0521.pdf
- ICH, *M12 Drug Interaction Studies: Questions and Answers*, Step 4, 2024.
  https://database.ich.org/sites/default/files/ICH_M12_Step4_Q%26As_2023_0521.pdf
- ICH, *E11(R1) Addendum: Clinical Investigation of Medicinal Products in the Pediatric Population*,
  2017. https://www.ich.org/page/efficacy-guidelines
- European Medicines Agency, *Guideline on the evaluation of the pharmacokinetics of medicinal
  products in patients with decreased renal function*, 2015. https://www.ema.europa.eu/

### Clinical pharmacology references

- Rybak, M. J. et al., "Therapeutic monitoring of vancomycin for serious methicillin-resistant
  Staphylococcus aureus infections: A revised consensus guideline", *American Journal of
  Health-System Pharmacy* 77(11), 835–864, 2020. https://doi.org/10.1093/ajhp/zxaa036
- Rowland, M., Tozer, T. N., *Clinical Pharmacokinetics and Pharmacodynamics: Concepts and
  Applications*, 4th edition, Lippincott Williams & Wilkins, 2011.
- Holford, N., Sheiner, L. B., "Understanding the dose-effect relationship: clinical application of
  pharmacokinetic-pharmacodynamic models", *Clinical Pharmacokinetics* 6(6), 429–453, 1981.
  https://doi.org/10.2165/00003088-198106060-00002
- Inker, L. A. et al., "New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race",
  *New England Journal of Medicine* 385, 1737–1749, 2021. https://doi.org/10.1056/NEJMoa2102953

### Assay and monitoring standards

- CLSI EP05-A3, *Evaluation of Precision of Quantitative Measurement Procedures*, 2014.
  https://clsi.org/standards/products/method-evaluation/documents/ep05/
- CLSI EP15-A3, *User Verification of Precision and Estimation of Bias*, 2014.
  https://clsi.org/standards/products/method-evaluation/documents/ep15/

> **Note:** The FDA and ICH documents are freely available. The textbook is a standard reference and is
> cited for the compartment model and parameter definitions, not for a specific numerical claim.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`05_biological-basics.md`](05_biological-basics.md)
- Next topic: [`07_epidemiology-basics.md`](07_epidemiology-basics.md)
- Units and conversions used in dosing: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Dimensional checks of clearance and AUC: [`04_dimensional-analysis.md`](04_dimensional-analysis.md)
- Exponential and half-life mathematics: [`01_mathematical-basics.md`](01_mathematical-basics.md)
- Assumption recording: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — the survey file's Lab 01A.5 exercises the one-compartment fit
  and loading-dose calculation; no dedicated notebook is present in `labs/` for this topic.
