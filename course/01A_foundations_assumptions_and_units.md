# Module 01A — Foundations, Units, Measurement, and Assumptions

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 20 hours (8 lecture, 4 clinic, 8 lab)
> **Track fit:** Common core; take before Modules 02–06
> **Prerequisites:** High-school algebra; curiosity; no clinical or engineering background
> **Tools:** Python, NumPy, pandas, SciPy, Pint, a calculator, a assumptions-ledger template
> **Industry question:** "What must be true for this number, model, or device reading to mean what we think it means?"

---

## 1. Why this module exists

Biomedical data science fails in ordinary ways:

- A glucose value is entered in mmol/L but interpreted as mg/dL.
- A model assumes independent observations when each patient contributes 40 visits.
- A sepsis label is treated as a diagnosis rather than a treatment trigger.
- A pulse oximeter reading is treated as arterial oxygen saturation.
- A surgical robot's kinematics are assumed perfect while its registration is 8 mm off.
- A vendor reports AUROC on a random split and calls it clinical validation.
- A budget impact model assumes 80% adoption and zero clinician review time.

None of these is exotic. They are **assumption failures**.

This module supplies the minimum mathematical, biological, clinical, engineering, and data foundations needed to notice those failures. It is deliberately a primer, not a substitute for a degree. Its most important artifact is the **assumptions ledger**: a living document that states what must be true, how it will be checked, and what happens if it is false.

## 2. Learning outcomes

By the end of this module, you can:

- Use units, dimensions, and clinical conversions correctly.
- Explain accuracy, precision, bias, calibration, resolution, uncertainty, and signal-to-noise.
- Describe the biological and physiological basis of common clinical variables.
- Distinguish incidence, prevalence, risk, odds, hazard, and utility.
- Read and write basic clinical and epidemiological formulas.
- Explain the signal chain from sensor to decision.
- State the core assumptions behind common statistical, causal, ML, clinical, device, and economic claims.
- Detect assumption violations in a paper, dataset, or product claim.
- Build an assumptions ledger for a project.
- Communicate uncertainty without false precision.

## 3. Mental model: the assumption stack

```
Biology → Measurement → Data → Model → Decision → Outcome
   │          │           │       │         │         │
Assumptions about each layer, and about the links between them
```

A claim is only as strong as the weakest assumption in the stack.

### 3.1 The five questions

1. **What is the thing?** (biology, disease, device, workflow)
2. **How is it measured?** (sensor, lab, code, note, survey)
3. **What is the number?** (units, scale, reference range, uncertainty)
4. **What must be true?** (assumptions)
5. **What decision does it change?** (action, harm, value)

## 4. Mathematical basics

### 4.1 Numbers, scales, and dimensions

- **Nominal:** categories with no order (blood type, sex, race).
- **Ordinal:** ordered categories (pain 0–10, NYHA class).
- **Interval:** equal intervals, arbitrary zero (°C).
- **Ratio:** equal intervals, true zero (weight, creatinine).
- **Count:** non-negative integers (admissions).
- **Compositional:** parts of a whole (microbiome, cell-type proportions).

**Dimension:** the physical quantity (length, mass, time). **Unit:** the scale (cm, kg, s). A number without a unit is not a measurement.

### 4.2 Algebra and functions

- Linear: \(y = mx + b\).
- Exponential: \(y = ae^{bx}\) (growth, decay, pharmacokinetics).
- Logarithmic: \(y = a\log(x)\) (pH, decibels, viral load).
- Logistic: \(p = 1/(1+e^{-z})\) (risk, dose-response).
- Power law: \(y = ax^b\) (scaling, allometry).
- Sigmoid: threshold response.

**Clinical example:** pH is \(-\log_{10}[H^+]\). A pH change from 7.4 to 7.1 is a 2-fold increase in hydrogen ion concentration, not a 0.3-unit change in “acidity.”

### 4.3 Linear algebra

- Vectors and matrices.
- Matrix multiplication.
- Inverse and pseudoinverse.
- Eigenvalues and eigenvectors.
- Projections.
- Rank and conditioning.

**Uses:** regression, PCA, image transforms, graph embeddings, robotics kinematics.

### 4.4 Calculus and optimization

- Derivative: rate of change.
- Integral: accumulation (AUC, dose exposure).
- Partial derivative: sensitivity to one input.
- Gradient: direction of steepest change.
- Convexity: whether a minimum is global.
- Constrained optimization: capacity, cost, safety limits.

**Clinical example:** drug concentration over time is an integral of exposure; clearance relates dose to exposure.

### 4.5 Probability

- Random variable.
- PMF, PDF, CDF.
- Expectation and variance.
- Conditional probability.
- Bayes' theorem.
- Independence and conditional independence.
- Distributions: Bernoulli, binomial, Poisson, normal, log-normal, exponential, Weibull.

**Bayes:**

\[
P(D|T) = \frac{P(T|D)P(D)}{P(T)}
\]

A positive test is not a diagnosis; it depends on pretest probability.

### 4.6 Statistics

- Sampling distributions.
- Standard error.
- Confidence intervals.
- Hypothesis tests.
- Effect sizes.
- Correlation and regression.
- Maximum likelihood.
- Bootstrap.
- Bayesian inference.

**Rule:** a p-value is not the probability that the null hypothesis is true, nor the probability of replication.

### 4.7 Graphs and networks

- Nodes, edges, paths, components.
- Degree, centrality, clustering.
- Directed and undirected graphs.
- DAGs for causal assumptions.
- Graphs for care pathways, protein interactions, and hospital operations.

### 4.8 Numerical methods

- Floating-point precision.
- Numerical stability.
- Root finding.
- Optimization.
- Interpolation and extrapolation.
- Simulation.
- Reproducibility.

**Clinical example:** subtracting two large, nearly equal lab values can lose precision; report with appropriate significant figures.

## 5. Units and metrology

### 5.1 SI base units

| Quantity | SI unit | Symbol |
|---|---|---|
| Length | metre | m |
| Mass | kilogram | kg |
| Time | second | s |
| Electric current | ampere | A |
| Thermodynamic temperature | kelvin | K |
| Amount of substance | mole | mol |
| Luminous intensity | candela | cd |

Derived units: Hz, N, Pa, J, W, C, V, F, Ω, S, Wb, T, H, lm, lx, Bq, Gy, Sv, katal.

### 5.2 Common clinical conversions

| Quantity | From | To | Factor |
|---|---|---|---|
| Glucose | mg/dL | mmol/L | × 0.0555 |
| Glucose | mmol/L | mg/dL | × 18.0 |
| Creatinine | mg/dL | µmol/L | × 88.4 |
| BUN | mg/dL | mmol/L urea | × 0.357 |
| Total cholesterol | mg/dL | mmol/L | × 0.02586 |
| Triglycerides | mg/dL | mmol/L | × 0.01129 |
| Bilirubin | mg/dL | µmol/L | × 17.1 |
| Calcium | mg/dL | mmol/L | × 0.25 |
| Hemoglobin | g/dL | g/L | × 10 |
| Albumin | g/dL | g/L | × 10 |
| Pressure | mmHg | kPa | × 0.133322 |
| Length | in | cm | × 2.54 |
| Weight | lb | kg | × 0.453592 |
| Energy | kcal | kJ | × 4.184 |
| Radiation absorbed dose | rad | Gy | × 0.01 |
| Equivalent dose | rem | Sv | × 0.01 |

**Rule:** convert to a canonical unit at ingestion, store the original unit, and test the conversion.

### 5.3 Common formulas

| Formula | Expression | Notes |
|---|---|---|
| BMI | weight kg / height m² | Not a direct measure of body fat |
| BSA (Mosteller) | √(height cm × weight kg / 3600) | Used for dosing |
| MAP | DBP + (SBP − DBP)/3 | Approximate |
| Cardiac output | HR × stroke volume | L/min |
| Oxygen content | 1.34 × Hb × SaO₂ + 0.003 × PaO₂ | mL O₂/dL |
| Anion gap | Na − (Cl + HCO₃) | Normal range lab-specific |
| A-a gradient | PAO₂ − PaO₂ | PAO₂ = FiO₂ × (Patm − 47) − PaCO₂/0.8 |
| Cockcroft-Gault | ((140 − age) × weight)/(72 × SCr) × 0.85 if female | Creatinine clearance estimate |
| Sensitivity | TP/(TP + FN) | Detects disease |
| Specificity | TN/(TN + FP) | Rules out disease |
| PPV | TP/(TP + FP) | Prevalence-dependent |
| NPV | TN/(TN + FN) | Prevalence-dependent |
| ARR | control event rate − treatment event rate | Absolute risk reduction |
| NNT | 1/ARR | Number needed to treat |
| NNH | 1/ARI | Number needed to harm |
| Incidence rate | events/person-time | Person-years |
| Prevalence | cases/population at a time | Point or period |
| Odds ratio | (a/b)/(c/d) | Case-control |
| Relative risk | (a/(a+b))/(c/(c+d)) | Cohort |
| Half-life | ln 2 / k | First-order elimination |
| Steady state | ~4–5 half-lives | Approximation |
| Clearance | dose/AUC | Volume per time |
| Volume of distribution | dose/C₀ | Apparent volume |
| ICER | ΔCost/ΔQALY | Cost-effectiveness |
| DALY | YLL + YLD | Burden of disease |

**Rule:** formulas are models with assumptions. Cockcroft-Gault assumes stable creatinine and a particular body-size relationship; it is not valid in every population.

### 5.4 Measurement quality

| Term | Meaning | Example |
|---|---|---|
| Accuracy | Closeness to true value | BP reads 2 mmHg high on average |
| Precision | Repeatability | BP varies ±5 mmHg on repeats |
| Bias | Systematic error | Cuff too small reads high |
| Calibration | Mapping raw signal to reference | Scale zeroing |
| Resolution | Smallest detectable change | 1 mmHg display |
| Limit of detection | Lowest reliable measured value | Troponin assay |
| Uncertainty | Range consistent with measurement | ±0.2 °C |
| Signal-to-noise ratio | Signal relative to noise | ECG vs muscle artifact |
| Drift | Change over time | CGM sensor drift |
| Linearity | Proportional response | PPG at high heart rates |
| Hysteresis | Direction-dependent response | Pressure sensors |

**Rule:** precision without accuracy is consistently wrong. Accuracy without precision is unreliably right.

### 5.5 Significant figures and reporting

- Do not report more digits than the measurement supports.
- A risk of 0.247 is false precision if the model's calibration uncertainty is ±0.08.
- Report units, reference range, and method.
- Use consistent decimal separators.
- State whether a value is measured, calculated, or imputed.

## 6. Biological basics

### 6.1 The cell

- Membrane, cytoplasm, nucleus, organelles.
- DNA → RNA → protein (central dogma).
- Metabolism: glycolysis, oxidative phosphorylation.
- Cell signaling and receptors.
- Cell cycle, apoptosis, senescence.
- Differentiation and stem cells.

### 6.2 Tissue and organ systems

| System | Core function | Common data |
|---|---|---|
| Cardiovascular | Perfusion, oxygen delivery | BP, HR, ECG, echo, troponin, BNP |
| Respiratory | Gas exchange | RR, SpO₂, ABG, spirometry, imaging |
| Renal | Filtration, electrolytes, acid-base | Creatinine, eGFR, urine output, electrolytes |
| Neurologic | Sensing, control, cognition | Exam, NIHSS, EEG, imaging |
| Endocrine | Hormonal regulation | Glucose, HbA1c, thyroid, cortisol |
| Immune | Defense and inflammation | WBC, CRP, cultures, cytokines |
| Hematologic | Oxygen transport, clotting | Hb, platelets, INR, coagulation |
| Gastrointestinal | Digestion, absorption | Symptoms, endoscopy, liver tests |
| Musculoskeletal | Structure and movement | Exam, imaging, strength, gait |
| Integumentary | Barrier, temperature | Skin exam, wounds, perfusion |

### 6.3 Homeostasis

Physiology maintains variables within ranges via feedback:

- Negative feedback: BP, glucose, temperature, pH.
- Positive feedback: labor, clotting, action potentials.
- Set points vary by age, pregnancy, altitude, disease, and medication.

**Data implication:** a “normal” lab value may be abnormal for a given patient. Reference ranges are population summaries, not individual targets.

### 6.4 Pharmacokinetics

| Process | Meaning | Data |
|---|---|---|
| Absorption | Drug enters circulation | Route, bioavailability |
| Distribution | Drug moves to tissues | Volume of distribution |
| Metabolism | Drug is transformed | Liver enzymes, interactions |
| Excretion | Drug leaves body | Renal/hepatic clearance |

- **Bioavailability (F):** fraction reaching systemic circulation.
- **Half-life (t½):** time for concentration to halve.
- **Clearance (CL):** volume cleared per time.
- **Volume of distribution (Vd):** apparent distribution volume.
- **Steady state:** accumulation balance; roughly 4–5 half-lives.
- **Therapeutic index:** toxic dose / effective dose.

### 6.5 Pharmacodynamics

- Dose-response.
- Receptor agonism/antagonism.
- Potency and efficacy.
- Tolerance and tachyphylaxis.
- Adverse effects.
- Drug interactions.

### 6.6 Genetics and genomics

- DNA, genes, alleles, chromosomes.
- Mendelian and complex inheritance.
- Variants: SNV, indel, CNV, structural.
- Penetrance and expressivity.
- Pharmacogenomics.
- Somatic vs germline.
- Ancestry and population genetics.

### 6.7 Immunology and microbiology

- Innate and adaptive immunity.
- Antibodies, T cells, cytokines.
- Inflammation: local and systemic.
- Sepsis: dysregulated host response to infection.
- Colonization vs infection.
- Antimicrobial resistance.
- Vaccines and herd immunity.

### 6.8 Special populations and reference ranges

| Population | Why the usual assumptions fail | Data implication |
|---|---|---|
| Children | Weight-based dosing, developing organs, age-specific reference ranges | Do not apply adult ranges; use pediatric-specific models |
| Pregnant patients | Plasma volume, renal function, immune changes; fetal safety | Many drugs and devices lack adequate evidence; flag uncertainty |
| Older adults | Multimorbidity, polypharmacy, frailty, atypical presentation | Competing risks; functional outcomes; deprescribing |
| Disabled people | Access barriers, communication needs, equipment fit | Accessibility and participatory design |
| Neurodivergent people | Sensory, cognitive, and communication differences | Multiple formats; avoid time pressure; avoid pathologizing |
| Low-resource settings | Lab/device availability, connectivity, staffing | Validate on the deployment context, not only tertiary centers |
| Psychiatric conditions | Overlapping symptoms, stigma, variable insight, social context | Use validated scales; avoid pathologizing language; protect privacy |
| Palliative care | Goals of care, symptom burden, prognosis | Patient-reported outcomes; avoid aggressive default targets |

**Rule:** a reference range is a population summary, not an individual target. “Normal” depends on age, pregnancy, altitude, comorbidities, and the assay.

### 6.9 Pathology and laboratory medicine basics

**Pathology.** Tissue is fixed, sectioned, stained (commonly H&E), and examined. Immunohistochemistry (IHC) detects proteins; special stains detect organisms or molecules. Diagnosis includes tissue type, grade, stage, margins, and molecular markers. Whole-slide images are gigapixel; stain, scanner, and section thickness affect appearance.

**Laboratory medicine.** The total testing process has three phases:

1. **Pre-analytical:** patient preparation, collection, tube type, transport, hemolysis, timing.
2. **Analytical:** method, calibration, quality control, interference, limit of detection.
3. **Post-analytical:** reference range, delta check, critical-value notification, reporting.

**Assumptions:** the sample is representative, the assay is calibrated, the result is comparable across labs, and the reference range fits the patient. **Failure modes:** wrong tube, delayed transport, hemolysis, dilution, assay drift, and look-alike patient errors.

**Regulatory context:** CLIA/CAP in the US; IVDR in the EU. Laboratory-developed tests have a separate regulatory history.

## 7. Clinical and epidemiological basics

### 7.1 Clinical vocabulary

| Term | Meaning |
|---|---|
| Sign | Observable finding |
| Symptom | Patient-reported experience |
| Syndrome | Cluster of signs/symptoms without a single cause |
| Diagnosis | Best explanation after evaluation |
| Differential diagnosis | Ranked list of possibilities |
| Acute | Sudden/short duration |
| Chronic | Long duration |
| Benign | Not cancerous |
| Malignant | Cancerous, invasive |
| Morbidity | Illness burden |
| Mortality | Death |
| Comorbidity | Additional condition |
| Iatrogenic | Caused by care |
| Idiopathic | Unknown cause |
| Contraindication | Reason not to use a treatment |
| Off-label | Use outside approved indication |

### 7.2 Risk measures

- **Absolute risk:** probability of an event in a group.
- **Relative risk:** ratio of risks.
- **Odds ratio:** ratio of odds; common in case-control studies.
- **Hazard ratio:** relative instantaneous hazard; time-to-event.
- **Risk difference:** absolute difference.
- **NNT/NNH:** number needed to treat/harm.
- **QALY:** quality-adjusted life year.
- **DALY:** disability-adjusted life year.

**Rule:** relative effects sound larger than absolute effects. Always ask “compared with what baseline?”

### 7.3 Study designs

- Case report/series.
- Cross-sectional.
- Case-control.
- Cohort.
- Randomized controlled trial.
- Pragmatic trial.
- Interrupted time series.
- Difference-in-differences.
- Target trial emulation.

### 7.4 Bias

- Selection.
- Confounding.
- Information/measurement.
- Attrition.
- Immortal time.
- Lead-time.
- Length-time.
- Recall.
- Publication.
- Reporting.
- Automation.

## 8. Engineering and device basics

### 8.1 Signal chain

```
Sensor → analog front end → ADC → processor → algorithm → output
```

At each stage:

- Sensor: physics, contact, sensitivity, drift.
- AFE: amplification, filtering, impedance, noise.
- ADC: sampling rate, resolution, aliasing.
- Processor: latency, power, memory, numerical precision.
- Algorithm: assumptions, calibration, failure modes.
- Output: display, alarm, actuator, API.

### 8.2 Sampling and aliasing

Nyquist: sample at more than twice the highest frequency of interest.

- ECG: 250–1,000 Hz.
- PPG: 25–1,000 Hz.
- EEG: 200–1,000 Hz.
- Respiratory: 10–100 Hz.
- Accelerometry: 25–100 Hz.

**Aliasing** folds high-frequency content into lower frequencies. It cannot be removed by post-hoc filtering.

### 8.3 Noise and filtering

- Thermal noise.
- Shot noise.
- Flicker noise.
- Motion artifact.
- Powerline interference (50/60 Hz).
- Baseline wander.
- Quantization noise.

Filters: low-pass, high-pass, band-pass, notch, median. Filter cutoffs must be justified and applied consistently.

### 8.4 Electrical safety

- Macroshock and microshock.
- Leakage current.
- Grounding.
- Isolation.
- Defibrillation protection.
- IEC 60601-1.
- EMC: emissions and immunity.

**Data implication:** a device that is safe in the lab can fail in the presence of electrosurgery, MRI, or a poor grounding environment.

### 8.5 Materials and sterilization

- Biocompatibility: ISO 10993.
- Sterilization: autoclave, ethylene oxide, radiation, hydrogen peroxide.
- Hermeticity.
- Corrosion.
- Wear and fatigue.
- Battery safety.

### 8.6 Software and control

- Requirements.
- Architecture.
- State machines.
- Real-time constraints.
- Watchdogs.
- Fail-safe defaults.
- Version control.
- Verification and validation.

### 8.7 Human factors

- Use error.
- Cognitive load.
- Alarm design.
- Accessibility.
- Training.
- Workflow fit.
- Automation bias.

## 9. Data and computing basics

| Topic | Why it matters |
|---|---|
| Data types | Wrong type breaks joins and models |
| Relational model | Most clinical data are relational |
| SQL | Cohort construction and validation |
| Python/R | Analysis and modeling |
| Git | Reproducibility and collaboration |
| APIs | FHIR, trial registries, model services |
| Cloud | Scale, governance, cost |
| Security | PHI protection |
| Versioning | Data, code, environment, model |
| Testing | Prevent silent errors |
| Documentation | Handoff and audit |

## 10. The assumptions ledger

### 10.1 Why a ledger

Most assumptions are implicit. A ledger makes them explicit, assigns evidence, and defines what would change the decision.

### 10.2 Template

| ID | Assumption | Type | Why needed | Evidence | Test | If false | Owner | Status |
|---|---|---|---|---|---|---|---|---|
| A1 | Patients are independent | Statistical | Standard errors and splits | Patient IDs | Cluster check | Use clustered SE/group split | Data lead | Open |
| A2 | No unmeasured confounding | Causal | Treatment effect | DAG, negative control | E-value | Bounds, sensitivity | Clinical lead | Open |
| A3 | Labels reflect disease | Clinical | Target validity | Chart review | PPV sample | Re-label/limit claim | Clinical lead | Open |
| A4 | Sensor accurate across skin tones | Device | Fair measurement | Bench study | Stratified error | Recalibrate/restrict | Hardware lead | Open |
| A5 | Clinicians act on output | Workflow | Impact | Workflow study | Shadow mode | Redesign | Product lead | Open |

### 10.3 Statistical assumptions

| Assumption | Why needed | How to check | If violated |
|---|---|---|---|
| IID observations | Standard errors, random splits | Patient/cluster IDs; autocorrelation | Cluster/group methods |
| Random sampling | Generalization | Compare to target population | Weighting, external validation |
| Normality | Small-sample tests | QQ plot; robust tests | Nonparametric/robust methods |
| Homoscedasticity | Linear model inference | Residual plots | Robust/weighted models |
| Linearity | Linear model | Residuals, splines | Nonlinear terms |
| Proportional hazards | Cox model | Schoenfeld residuals | Time-varying/stratified |
| No measurement error | Unbiased estimates | Validation substudy | Error models, sensitivity |
| Missing at random | Imputation | Missingness analysis | MNAR sensitivity |

### 10.4 Causal assumptions

| Assumption | Meaning | Check | If violated |
|---|---|---|---|
| Consistency | Observed treatment corresponds to the defined strategy | Treatment definition | Redefine exposure |
| Positivity | Every patient has nonzero chance of each treatment | Propensity overlap | Restrict population |
| Exchangeability | No unmeasured confounding | DAG, negative controls | Sensitivity/bounds |
| SUTVA/no interference | One patient's treatment does not affect another | Cluster/contagion check | Cluster-randomized design |
| Correct time zero | Follow-up starts at the same point | Timeline audit | Rebuild cohort |
| No immortal time | Exposure not guaranteed before time zero | Landmark/time-varying | Redesign |

### 10.5 Machine-learning assumptions

| Assumption | Why needed | Check | If violated |
|---|---|---|---|
| Train/test exchangeability | Generalization | Temporal/site validation | Domain adaptation |
| Representative data | Fairness and accuracy | Subgroup comparison | Reweight/collect |
| Valid labels | Supervised learning | Chart review, agreement | Re-label |
| No leakage | Honest performance | Timing audit, ablation | Rebuild features |
| Stable concept | Deployment performance | Monitoring | Revalidate |
| Adequate sample size | Stable estimates | Learning curves | Simplify/collect |
| Calibration | Probability decisions | Reliability plot | Recalibrate |
| No feedback loop | Valid future data | Deployment analysis | Randomize/audit |

### 10.6 Clinical and workflow assumptions

- The label means the disease.
- The measurement is comparable across sites and groups.
- The patient population is similar to deployment.
- Clinicians will use the output as intended.
- The alert fits the workflow.
- The intervention exists and is effective.
- Patients can access the intervention.
- Documentation reflects care.

### 10.7 Device assumptions

- Sensor accuracy and precision.
- Contact and perfusion.
- Power and connectivity.
- Environmental tolerance.
- Electromagnetic compatibility.
- Sterilization and biocompatibility.
- Calibration and maintenance.
- Cybersecurity.
- Usability.
- Clinician and patient adherence.

### 10.8 Economic assumptions

- Willingness to pay.
- Reimbursement.
- Adoption.
- Workflow cost.
- Time horizon.
- Perspective.
- Discount rate.
- Budget impact.

### 10.9 Ethical assumptions

- Consent is valid.
- Benefits and harms are fairly distributed.
- Accessibility is adequate.
- Communities are involved.
- Data are used for the stated purpose.
- Recourse exists for harm.

## 11. Unit and assumption error gallery

| Error | Why it happens | Consequence | Prevention |
|---|---|---|---|
| Glucose unit mix-up | mg/dL vs mmol/L | 18× dosing error | Canonical units, validation |
| Weight in lb vs kg | Device default | Drug overdose | Unit field, plausibility range |
| Creatinine unit | mg/dL vs µmol/L | eGFR error | LOINC/UCUM, lab-specific |
| Temperature site | Oral vs core | Sepsis screen error | Site metadata |
| BP cuff size | Obesity | False hypertension | Cuff-size protocol |
| SpO₂ skin tone | Optical bias | Missed hypoxemia | Stratified validation |
| Patient leakage | Multiple visits | Inflated AUC | Group split |
| Random split | Temporal drift | Deployment failure | Temporal split |
| Label leakage | Future note | False performance | Time-zero audit |
| Immortal time | Treatment delay | False benefit | Landmark analysis |
| Missing not at random | Sick patients untested | Bias | Sensitivity analysis |
| Wrong denominator | Claims vs EHR | Wrong incidence | Define population |
| Survivorship bias | Only survivors sampled | False association | Cohort design |
| Healthy-user bias | Wearable owners | Unrepresentative | Weighting/validation |
| Automation bias | Clinician over-trust | Harm | Human factors, audit |

## 12. Labs

### Lab 01A.1 — Dimensional analysis and clinical conversions

**Objective:** Convert a panel of clinical values and catch unit errors.

**Steps:**

1. Convert glucose, creatinine, cholesterol, bilirubin, calcium, pressure, weight, and temperature.
2. Check each with dimensional analysis.
3. Build a Python function with a units library.
4. Add range validation.
5. Write a test suite with planted errors.

**Deliverable:** conversion module and tests.

**Acceptance criteria:** correct factors; original unit stored; implausible values rejected.

### Lab 01A.2 — Measurement error simulation

**Objective:** Explore accuracy, precision, bias, and calibration.

**Steps:**

1. Simulate a true signal.
2. Add random noise.
3. Add systematic bias.
4. Calibrate the noisy measurements.
5. Compare error before and after.
6. Plot Bland-Altman.

**Deliverable:** simulation and memo.

**Acceptance criteria:** distinguishes bias from variance; calibration improves accuracy.

### Lab 01A.3 — Assumption audit

**Objective:** Audit a published clinical prediction model.

**Steps:**

1. Identify the claim.
2. List statistical, causal, ML, clinical, and workflow assumptions.
3. Find evidence for each.
4. Flag unsupported assumptions.
5. Write a 1-page decision memo.

**Deliverable:** assumptions ledger.

**Acceptance criteria:** specific; evidence-linked; decision-relevant.

### Lab 01A.4 — Sampling and aliasing

**Objective:** Demonstrate aliasing in a physiological signal.

**Steps:**

1. Generate a 10 Hz signal.
2. Sample at 12 Hz and 100 Hz.
3. Reconstruct both.
4. Show the aliased frequency.
5. Explain why post-hoc filtering cannot fix it.

**Deliverable:** plots and explanation.

**Acceptance criteria:** correct Nyquist reasoning; clear clinical example.

### Lab 01A.5 — Pharmacokinetics and half-life

**Objective:** Compute half-life, steady state, and loading dose.

**Steps:**

1. Fit a one-compartment model to concentration-time data.
2. Estimate elimination rate and half-life.
3. Predict steady-state concentration.
4. Calculate a loading dose.
5. Discuss assumptions.

**Deliverable:** analysis and memo.

**Acceptance criteria:** units correct; assumptions stated.

### Lab 01A.6 — NNT and cost-effectiveness

**Objective:** Convert trial results into absolute effects and value.

**Steps:**

1. Compute ARR, NNT, and NNH.
2. Build a simple cost-effectiveness model.
3. Calculate ICER.
4. Run one-way sensitivity analysis.
5. Explain to a payer and a clinician.

**Deliverable:** model and two-audience memo.

**Acceptance criteria:** absolute vs relative effects clear; perspective stated.

## 13. Assignment: assumptions ledger

Choose a project or published study and build a 2,000-word assumptions ledger:

1. Claim and decision.
2. Biology and measurement assumptions.
3. Statistical assumptions.
4. Causal assumptions.
5. ML assumptions.
6. Clinical/workflow assumptions.
7. Device assumptions, if relevant.
8. Economic and ethical assumptions.
9. Evidence and tests.
10. What would change the decision.

Use the table template in section 10.2.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Units and measurement | 20% | Converts by memory | Correct units and uncertainty | Catches subtle unit/metrology errors |
| Biology and clinical | 20% | Uses terms loosely | Correct vocabulary and mechanisms | Connects to decisions and harms |
| Assumptions | 30% | Lists generic caveats | Specific, evidence-linked ledger | Prioritizes and tests load-bearing assumptions |
| Quantitative reasoning | 20% | Formula plugging | Correct formulas and interpretation | Sensitivity and decision analysis |
| Communication | 10% | Jargon | Clear to clinicians | Clear to clinical and executive audiences |

## 15. Regulatory, ethical, and commercial lens

- **Units and labels:** device labeling and lab reporting are regulated; unit errors can be safety events.
- **Measurement bias:** pulse oximetry and PPG bias are regulatory and equity issues.
- **Assumptions in submissions:** regulators expect intended use, validation populations, and limitations.
- **Economic assumptions:** budget impact and adoption assumptions must be transparent.
- **Ethics:** assumptions about “normal,” adherence, and access can encode stigma.

## 16. Failure modes and red flags

- No unit field.
- Hard-coded reference ranges.
- Treating a measurement as ground truth.
- Ignoring clustering.
- Assuming missingness is random.
- Confusing relative and absolute risk.
- Using a formula outside its validation population.
- Assuming a device is accurate across skin tones or body sizes.
- Assuming clinicians will use an alert.
- Reporting false precision.

## 17. Interview and discussion questions

1. **Why is a number without a unit not a measurement?**
   Hint: dimension and scale.
2. **What is the difference between accuracy and precision?**
   Hint: bias vs variance.
3. **Why can a test with high sensitivity have low PPV?**
   Hint: low prevalence.
4. **What assumptions does a random train/test split make?**
   Hint: exchangeability, independence, no temporal drift.
5. **What is positivity in causal inference?**
   Hint: overlap; every patient has a chance of each treatment.
6. **Why is a p-value not the probability the null is true?**
   Hint: frequentist definition.
7. **What is the difference between incidence and prevalence?**
   Hint: new cases vs existing cases.
8. **Why is Cockcroft-Gault not valid for everyone?**
   Hint: stable creatinine, body composition, population.
9. **What does Nyquist say?**
   Hint: sample above twice the highest frequency.
10. **What assumptions underlie a wearable heart-rate measurement?**
    Hint: contact, perfusion, motion, skin tone, device algorithm.
11. **What is an assumptions ledger?**
    Hint: explicit, evidence-linked, testable assumptions.
12. **When would you stop a project because an assumption is untestable?**
    Hint: when the decision depends on it and no evidence can be obtained.

## 18. Key terms

| Term | Definition |
|---|---|
| Dimension | Physical quantity such as length or mass |
| Unit | Scale used to express a dimension |
| Accuracy | Closeness to truth |
| Precision | Repeatability |
| Bias | Systematic error |
| Calibration | Mapping measurement to reference |
| Resolution | Smallest distinguishable change |
| Limit of detection | Lowest reliably measurable value |
| Uncertainty | Range consistent with a measurement |
| Signal-to-noise ratio | Signal relative to noise |
| Nyquist limit | Half the sampling rate |
| Aliasing | High-frequency folding |
| Homeostasis | Maintenance of internal stability |
| Pharmacokinetics | What the body does to a drug |
| Pharmacodynamics | What the drug does to the body |
| Half-life | Time for concentration to halve |
| Clearance | Volume cleared per time |
| Volume of distribution | Apparent distribution volume |
| Incidence | New cases per population-time |
| Prevalence | Existing cases at a time |
| Absolute risk | Probability of an event |
| Relative risk | Ratio of risks |
| Odds ratio | Ratio of odds |
| Hazard ratio | Relative instantaneous hazard |
| NNT | Number needed to treat |
| QALY | Quality-adjusted life year |
| DALY | Disability-adjusted life year |
| Assumption | Condition required for a claim to hold |
| Exchangeability | No unmeasured confounding |
| Positivity | Overlap in treatment probabilities |
| Consistency | Observed treatment matches defined strategy |
| SUTVA | No interference between units |

## 19. Sources and further reading

### Units and metrology
- BIPM, *SI Brochure*. https://www.bipm.org/en/publications/si-brochure
- UCUM: https://ucum.org/
- LOINC: https://loinc.org/
- NIST, *Measurement Uncertainty*. https://www.nist.gov/

### Mathematics and statistics
- Wasserman, L., *All of Statistics*.
- Casella, G., Berger, R. L., *Statistical Inference*.
- Strang, G., *Linear Algebra and Its Applications*.
- Boyd, S., Vandenberghe, L., *Convex Optimization*.
- Hernán, M. A., Robins, J. M., *Causal Inference: What If*. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

### Biology and clinical
- Alberts, B. et al., *Molecular Biology of the Cell*.
- Guyton, A. C., Hall, J. E., *Textbook of Medical Physiology*.
- Bickley, L. S., *Bates' Guide to Physical Examination and History Taking*.
- West, J. B. et al., *West's Respiratory Physiology*.
- Lilly, L. S. et al., *Pathophysiology of Heart Disease*.
- Goldberger, A. L. et al., *Goldberger's Clinical Electrocardiography*.

### Engineering and devices
- Webster, J. G., *Medical Instrumentation: Application and Design*.
- IEC 60601-1: https://www.iec.ch/
- IEC 62304: https://www.iec.ch/
- ISO 14971: https://www.iso.org/standard/72704.html
- ISO 10993: https://www.iso.org/

### Epidemiology
- Rothman, K. J., Greenland, S., Lash, T. L., *Modern Epidemiology*.
- Gordis, L., *Epidemiology*.

> **Next:** [Module 02](02_biomedical_data_landscape.md) applies these foundations to real biomedical data sources and standards.
