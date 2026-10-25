# Biological Basics

> Module 01A — Foundations, Assumptions, and Units
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 03
> Version 2026.3

## 1. Why this topic exists

A dataset column named `lactate` looks like a number. It is actually the output of a physiological
process with a production rate, a clearance rate, a compartment, a measurement method, and a
timescale. An analyst who treats it as a number will build features and labels that are statistically
valid and biologically meaningless, and the model will fail in deployment for a reason that no amount
of cross-validation can reveal.

The failures are specific.

- A deterioration model uses the last recorded observation before an event. For a patient whose
  observation frequency increases as they deteriorate, the "last value" is measured closer to the
  event in sicker patients. The model learns the sampling pattern, not the physiology, and it
  degrades when a hospital changes its monitoring protocol.
- A readmission model treats a diagnosis code as a disease. The code records a billing decision made
  after discharge, often by a coder who never saw the patient. The label correlates with
  documentation practice and with hospital coding intensity, and the model transports poorly across
  sites.
- A glucose prediction model is trained on hourly capillary measurements and deployed on continuous
  glucose monitor data. The measurement error structure, the sampling interval, and the physiological
  lag all differ. The model's performance drops and nobody can say why.
- A sepsis alert is built on the systemic inflammatory response criteria. Those criteria were
  explicitly rejected by the Sepsis-3 consensus because they have poor discriminant validity. The
  alert fires on a large fraction of inpatients and the team concludes that "sepsis is unpredictable"
  rather than that the label is wrong.
- A genomics pipeline reports a variant allele fraction of 0.18 as "heterozygous" without accounting
  for tumour purity. The expected fraction at 60% purity is 0.30, so 0.18 suggests a subclonal or
  copy-number-altered variant, which changes the interpretation.

Each of these is a biological assumption failure. The statistics can be correct and the biology wrong.

The topic exists because biomedical data science differs from generic data science in one specific
way: the generating process is a physical, regulated, evolved system with compartments, feedback loops,
and timescales. Ignoring that structure is not neutral. It produces features that are proxies for
care delivery rather than for disease, and labels that measure documentation rather than health.

There is also a commercial reason. Biomarker programmes fail in late-stage trials because the
biomarker was assumed to measure a process it only correlated with. Device programmes fail because the
validation population's physiology differed from the deployment population's. An analyst who can ask
"what process does this variable measure, in which compartment, at what timescale, and with what
feedback" prevents a class of expensive failure that statistics alone cannot detect.

> **Note:** This topic is a primer, not a physiology course. It covers the concepts that recur in data
> work: the levels of organisation, the timescales, homeostasis, the central dogma, and the major
> organ systems as sources of variables. It does not cover pathology in depth. Clinical content is
> educational and is not medical advice.

## 2. Learning outcomes

By the end of this topic you can:

1. **Place** a biomedical variable at the correct level of organisation (molecule, cell, tissue, organ,
   system, person, population) and state what that level implies for aggregation. Assessed in
   Exercise 1.
2. **Assign** a characteristic timescale to a physiological process and choose a sampling interval
   consistent with it. Assessed in Exercise 2.
3. **Explain** negative and positive feedback and identify the set point, the sensor, and the effector
   for a given homeostatic variable. Assessed in Exercise 3.
4. **Distinguish** a state variable from a rate and from a trajectory, and choose the feature type that
   matches the clinical question. Assessed in Exercise 4.
5. **Describe** the central dogma and classify a genomic variant as germline or somatic, and as
   coding or non-coding, with the appropriate nomenclature. Assessed in Exercise 5.
6. **Compute** basic physiological quantities (oxygen delivery, oxygen extraction ratio, urine output
   per kilogram, lactate clearance, SOFA score) from clinical data. Assessed in Exercise 6.
7. **Critique** a label definition for biological validity, including whether it measures disease,
   treatment, or documentation. Assessed in Exercise 7.
8. **Design** a feature set for a stated clinical prediction problem, justifying each feature by the
   biological process it proxies and its timescale. Assessed in Exercise 8.

## 3. The core mental model

Biology is organised as a hierarchy of levels, and every level has its own characteristic timescale.
A data variable is a measurement at one level and one timescale, and it cannot be interpreted without
both.

```text
   LEVEL                 EXAMPLE QUANTITY            TYPICAL TIMESCALE
   population            incidence rate              years
   person                weight, diagnosis           months to years
   organ system          cardiac output              seconds to years
   organ                 glomerular filtration       minutes to years
   tissue                perfusion, inflammation     minutes to days
   cell                  membrane potential          milliseconds to days
   molecule              enzyme activity, mRNA       microseconds to hours
```

A variable measured at one level does not substitute for a variable at another. A serum creatinine is
a molecular concentration that proxies organ-level filtration; it is not a measurement of filtration,
and its relationship to filtration depends on muscle mass, diet, and the compartment it is sampled
from.

### 3.1 Three axes of any biological variable

Every clinical variable has three coordinates:

1. **Level.** Where in the hierarchy it is measured.
2. **Compartment.** Which fluid or tissue it is sampled from: intracellular, interstitial, plasma,
   urine, cerebrospinal fluid, tissue.
3. **Time.** Whether it reflects an instantaneous state, an average over a window, or a cumulative
   exposure.

A plasma sodium is a level-molecule, compartment-plasma, instantaneous state. An HbA1c is a
level-molecule, compartment-red-cell, cumulative exposure over roughly 120 days. A creatinine is a
level-molecule, compartment-plasma, state that lags filtration by hours to days because of the
muscle-buffer. Treating all three as "instantaneous lab values" destroys the information that
distinguishes them.

### 3.2 Homeostasis and why set points move

Physiology maintains variables near a set point through negative feedback: a sensor detects deviation,
an effector counteracts it, and the loop restores the variable. The set point is not fixed. It varies
with age, pregnancy, altitude, time of day, posture, medication, and disease.

```text
        set point
            |
            v
   sensor --> controller --> effector --> variable
      ^                                      |
      |                                      |
      +------------- feedback ---------------+
```

Three data consequences follow.

**A value within the population reference range can be abnormal for the individual.** A creatinine of
1.1 mg/dL is normal in a muscular young man and elevated in an 80-year-old woman whose baseline is 0.6.
The reference range is a population summary, not an individual target.

**Compensation hides dysfunction until reserve is exhausted.** A patient can maintain a normal blood
pressure with a falling cardiac output until the compensatory vasoconstriction fails, at which point
the pressure falls abruptly. A model trained on pressure alone sees a stable variable and then a
sudden event, and interprets the event as unpredictable when it was preceded by a slow decline in a
variable that was not measured.

**Positive feedback produces rapid, non-linear change.** Clotting, labour, and action potentials are
positive-feedback processes. They are not well described by a linear trend, and a model that assumes
gradual change will be surprised by them.

### 3.3 The central dogma and its exceptions

```text
   DNA --transcription--> RNA --translation--> protein --> phenotype
    |                      |
    +-- replication        +-- regulation, splicing, editing
```

The central dogma describes the usual flow of information. It has documented exceptions and
refinements that matter for data work:

- **Retroviruses** reverse-transcribe RNA into DNA.
- **Alternative splicing** produces multiple proteins from one gene, so a gene count is not a protein
  count.
- **Non-coding RNAs** regulate expression without being translated.
- **Epigenetic marks** modify expression without changing the sequence.
- **Post-translational modification** changes protein activity without changing abundance.

The practical consequence is that a genomic variant is not a phenotype. A variant's effect depends on
where it is, whether it changes the protein, whether it is present in one or both alleles, and what
else is expressed in the tissue. This is why genomic risk scores are probabilistic and why variant
interpretation requires functional evidence, not only sequence.

### 3.4 The physiological reserve model

Most organ systems have a large reserve. Dysfunction becomes visible only when demand exceeds reserve
or when reserve falls below a threshold.

```text
   function
      |
      |--------  normal reserve
      |        \
      |         \  compensation
      |          \
      |           \________  decompensation
      |
      +----------------------> demand or time
```

The model explains why early disease is silent, why a stressor reveals it, and why a single
measurement at rest can be normal in a patient who is severely ill. It also explains why a model built
on resting measurements may need a provocation (a stress test, a fluid challenge, a glucose load) to
detect disease.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Cell | The basic structural and functional unit of life | Treating a cell line as a patient | count |
| Organelle | A membrane-bound structure with a specialised function | Confusing with an organ | count |
| Tissue | A group of cells with a common function | Using it for an organ | none |
| Organ | A structure of several tissues performing a function | Confusing with an organ system | none |
| Organ system | A group of organs with a coordinated function | Using it for a single organ | none |
| Genome | The complete DNA sequence of an organism | Using it for one gene | base pairs |
| Gene | A DNA segment that encodes a functional product | Equating a gene with a protein | HGNC symbol |
| Allele | One of the variant forms of a gene at a locus | Using it for a genotype | HGVS |
| Genotype | The combination of alleles at a locus | Using it for a phenotype | HGVS |
| Phenotype | An observable characteristic | Equating it with a diagnosis | none |
| Locus | A position on a chromosome | Confusing with an allele | chromosomal coordinate |
| Chromosome | A DNA-protein structure carrying genes | Counting chromatids as chromosomes | 1–22, X, Y |
| Transcription | Synthesis of RNA from a DNA template | Confusing with translation | none |
| Translation | Synthesis of protein from an mRNA template | Confusing with transcription | none |
| Protein | A chain of amino acids with a function | Equating abundance with activity | g/L or mol/L |
| Enzyme | A protein that catalyses a reaction | Assuming activity equals concentration | U/L |
| Receptor | A molecule that binds a ligand and transduces a signal | Confusing with a transporter | count per cell |
| Ligand | A molecule that binds a receptor | Using it for any drug | concentration |
| Signal transduction | The intracellular cascade following receptor activation | Reducing it to one molecule | none |
| Metabolism | The chemical reactions sustaining life | Using it for glucose only | none |
| Catabolism | Breakdown of molecules with energy release | Confusing with excretion | none |
| Anabolism | Synthesis of molecules with energy consumption | Confusing with absorption | none |
| ATP | The principal energy currency of the cell | Treating it as a nutrient | mol/L |
| Homeostasis | Maintenance of internal variables near set points | Assuming the set point is fixed | none |
| Set point | The value a feedback loop defends | Treating it as the population mean | variable unit |
| Negative feedback | A loop that counteracts deviation | Confusing with a downward trend | none |
| Positive feedback | A loop that amplifies deviation | Assuming all feedback is stabilizing | none |
| Allostasis | Adaptation of set points to sustained demand | Confusing with homeostasis | none |
| Circadian rhythm | An approximately 24-hour endogenous cycle | Treating time of day as noise | hours |
| Hormone | A signalling molecule transported in blood | Confusing with a neurotransmitter | concentration |
| Cytokine | A signalling molecule of the immune system | Using it as a synonym for inflammation | concentration |
| Inflammation | A coordinated response to injury or infection | Equating it with infection | none |
| Sepsis | Life-threatening organ dysfunction from a dysregulated response to infection | Equating it with bacteraemia | SOFA change |
| Bacteraemia | Bacteria in the blood | Equating it with sepsis | CFU/mL |
| Colonisation | Presence of organisms without disease | Treating it as infection | none |
| Antigen | A molecule recognised by the immune system | Confusing with an antibody | none |
| Antibody | An immunoglobulin that binds an antigen | Confusing with an antigen | titre |
| Cytopenia | A reduced count of a blood cell lineage | Using it without specifying the lineage | cells/µL |
| Anaemia | Reduced haemoglobin or red cell mass | Defining it by a single threshold | g/L |
| Leucocytosis | An increased white cell count | Treating it as diagnostic of infection | cells/µL |
| Thrombocytopenia | A reduced platelet count | Confusing with platelet dysfunction | cells/µL |
| Homeostatic reserve | The capacity to maintain function under stress | Assuming it from a normal resting value | none |
| Compartment | A distinct fluid or tissue space | Treating plasma as the whole body | volume |
| Turnover | The rate of replacement of a cell or molecule | Confusing with half-life | mass/time |
| Biological half-life | Time for half a substance to be cleared | Confusing with drug half-life | time |
| Biomarker | A measurable indicator of a biological process | Treating it as a diagnosis | variable unit |
| Surrogate endpoint | A biomarker used in place of a clinical outcome | Assuming it is validated | variable unit |
| Reference interval | The central 95% of a reference population | Treating it as a target range | variable unit |
| Penetrance | The proportion of carriers of a variant who express the phenotype | Treating it as certain | proportion |
| Expressivity | The degree of phenotype in carriers | Confusing with penetrance | none |
| Germline variant | A variant present in all cells, inherited or de novo | Confusing with somatic | HGVS |
| Somatic variant | A variant acquired in a cell lineage | Confusing with germline | HGVS |
| SNV | A single nucleotide variant | Calling all variants mutations | HGVS |
| Indel | An insertion or deletion | Reporting it without the reference | HGVS |
| CNV | A copy number variant | Confusing with an SNV | copy number |
| VAF | Variant allele fraction | Reporting it without purity | fraction |
| Tumour purity | The fraction of cells in a sample that are tumour | Ignoring it in VAF interpretation | fraction |
| Microbiome | The community of microorganisms in a body site | Treating it as a single organism | relative abundance |
| Dysbiosis | A disrupted microbial community | Using it as a diagnosis | none |
| Iatrogenic | Caused by medical care | Treating it as a natural history | none |
| Comorbidity | An additional condition | Treating it as a confounder without checking | ICD code |

Three clusters of definitions are used loosely in industry and cause specific analytic errors.

**Sepsis, bacteraemia, and infection.** Sepsis is organ dysfunction caused by a dysregulated host
response to infection. Bacteraemia is the presence of bacteria in blood. Infection is the invasion of
tissue by an organism. A patient can have bacteraemia without sepsis, sepsis without bacteraemia
(cultures are negative in a large fraction of cases), and neither with a positive culture from a
colonised site. A model trained on blood culture results learns to predict sampling, not sepsis.

**Biomarker and surrogate endpoint.** A biomarker is any measurable indicator of a biological process.
A surrogate endpoint is a biomarker substituted for a clinical outcome in a trial, and it requires a
specific validation argument. Most biomarkers are not surrogates. Treating a correlate as a surrogate
is the mechanism behind several late-stage trial failures.

**Penetrance and expressivity.** Penetrance is the probability of any phenotype given a genotype;
expressivity is the severity given that a phenotype occurs. A variant with 40% penetrance is not
"the cause" of a disease in an individual; it is a risk factor whose effect depends on other variants
and environment.

## 5. The mechanics, step by step

The procedure converts a biological question into a set of variables, features, and labels that a
model can use without violating the biology.

### Step 1 — State the biological process, not the variable

**Input.** A request such as "predict deterioration".

**Output.** A named process: "the transition from compensated to decompensated circulatory failure in
a patient with infection."

**Reasoning.** The variable list follows from the process. If the process is circulatory failure, the
relevant variables are perfusion, pressure, cardiac output, and lactate; if it is respiratory failure,
they are oxygenation, work of breathing, and ventilation. Starting from a variable list produces
models that measure whatever was convenient.

**Branch point.** If the process cannot be named, the prediction problem is not defined. A model
cannot be validated against an unnamed target.

### Step 2 — Locate the level of organisation

**Input.** The process.

**Output.** The level at which the outcome is defined and the level at which the predictors are
measured.

**Reasoning.** An outcome at the person level (death, readmission) is predicted by variables measured
at the organ, tissue, cell, or molecule level. The mismatch is normal, but it determines the
aggregation: multiple molecular measurements per person must be summarised before they can predict a
person-level outcome, and the summary is a modelling choice with biological consequences.

**Branch point.** If the predictors and outcome are at the same level and the same unit, check for
leakage. A diagnosis code predicting the same diagnosis code is not a model.

### Step 3 — Identify the compartment

**Input.** Each candidate variable.

**Output.** The compartment: plasma, serum, whole blood, interstitial fluid, urine, tissue, CSF.

**Reasoning.** The same molecule can have different concentrations in different compartments, and the
relationship between them depends on permeability, protein binding, and transport. A plasma
concentration is not a tissue concentration. A urine concentration is affected by water reabsorption
and by collection completeness.

**Branch point.** If the compartment is unknown, the value cannot be compared with a reference
interval derived in a different compartment. Treat it as a different quantity.

### Step 4 — Assign the timescale

**Input.** The process and the variable.

**Output.** A characteristic time constant.

**Reasoning.** Each process has a time constant that sets the sampling interval and the feature window.
An action potential lasts milliseconds; a cardiac cycle about 0.8 s; a circadian cycle 24 h; gut
epithelium renews in about 4 days; red cells live about 120 days; HbA1c integrates about 120 days.
A feature computed over a window much shorter than the time constant is noise; a window much longer
averages away the signal.

**Branch point.** If the sampling interval exceeds the time constant, the variable is undersampled and
its dynamics cannot be recovered. Do not model the dynamics; model the level.

### Step 5 — Classify the variable as state, rate, or cumulative exposure

**Input.** The variable.

**Output.** A type.

**Reasoning.** A state is an instantaneous value (blood pressure, temperature). A rate is a change per
unit time (urine output, lactate clearance). A cumulative exposure integrates over time (HbA1c,
pack-years, area under the concentration curve). The three require different features and different
missing-data handling: a missing state can be interpolated, a missing rate cannot be recovered without
the underlying states, and a cumulative exposure cannot be imputed from a single point.

**Branch point.** If a variable is treated as a state but is actually cumulative, the model will
respond to the wrong timescale. Check the analyte's biology before assuming.

### Step 6 — Identify the feedback loop

**Input.** The variable and its regulators.

**Output.** A loop description: sensor, controller, effector, and the direction of feedback.

**Reasoning.** Feedback changes the interpretation of a correlation. If two variables are linked by a
negative feedback loop, a change in one produces a compensatory change in the other, and the observed
correlation may be weak or reversed relative to the causal effect. Blood pressure and heart rate
during haemorrhage are the standard example: pressure is maintained while heart rate rises.

**Branch point.** If a variable is an effector of a loop, it is a response to the process, not a
measure of it. Include the driver if possible.

### Step 7 — Establish the set point and its modifiers

**Input.** The variable.

**Output.** A set point, or a model of how it varies.

**Reasoning.** Set points vary with age, pregnancy, altitude, time of day, posture, and medication. A
model that uses a single threshold across these groups will misclassify at the boundaries. The
modifiers should be either included as features or used to stratify.

**Branch point.** If the set point cannot be established for a subgroup, do not apply a population
threshold to it. Report the uncertainty.

### Step 8 — Choose a measurement that proxies the process

**Input.** The process and the available measurements.

**Output.** A proxy with a stated lag and error structure.

**Reasoning.** Most clinical variables are proxies. Creatinine proxies filtration with a lag of hours
to days; CRP proxies inflammation with a lag of about a day; lactate proxies the balance between
anaerobic production and clearance with a lag of minutes to hours. The proxy's lag determines how
early a model can detect the process, and its error structure determines how much signal is available.

**Branch point.** If the proxy's lag exceeds the decision window, the model cannot be early enough
regardless of its accuracy. Choose a different proxy or a different decision window.

### Step 9 — Define normal and abnormal with the right population

**Input.** The proxy.

**Output.** A reference interval and a decision threshold, each with its population.

**Reasoning.** Reference intervals are population summaries, partitioned by age, sex, pregnancy, and
assay. A threshold derived in adults does not apply to children; a threshold derived in a healthy
volunteer cohort does not apply to hospitalised patients. The threshold's purpose (screening,
diagnosis, monitoring) also determines its position.

**Branch point.** If the patient is outside the reference population, the interval does not apply.
Use a clinical decision limit with a stated basis instead.

### Step 10 — Account for circadian and postural variation

**Input.** The variable and the sampling time.

**Output.** A adjustment or a stratification.

**Reasoning.** Cortisol peaks in the early morning; blood pressure falls at night; body temperature
has a circadian nadir; plasma volume changes with posture and shifts sodium and protein
concentrations. A model that ignores sampling time learns time-of-day artefacts as if they were
biology, and it fails when the sampling schedule changes.

**Branch point.** If sampling time is unavailable, the model cannot separate circadian effects from
disease. Record the time or accept the limitation explicitly.

### Step 11 — Account for turnover and lag

**Input.** The variable's biological half-life.

**Output.** A lag structure for the feature.

**Reasoning.** A variable with a long half-life responds slowly and integrates over its lifetime; a
variable with a short half-life responds quickly and is noisy. HbA1c integrates about 120 days of
glucose; a capillary glucose integrates minutes. Using a long-half-life variable to detect an acute
event is a category error, and using a short-half-life variable to assess chronic control is equally
wrong.

**Branch point.** If the decision requires acute detection, choose a short-half-life variable. If it
requires chronic assessment, choose a long-half-life variable. Do not substitute.

### Step 12 — Distinguish compensated from decompensated states

**Input.** The variables and the clinical context.

**Output.** A judgement about reserve.

**Reasoning.** A normal value can coexist with severe disease when compensation is intact. The
relevant feature may be the trend, the variance, or the response to a challenge rather than the level.
A falling blood pressure within the normal range can be more informative than a single low value.

**Branch point.** If the model uses only levels, it cannot detect compensated deterioration. Add trend
features or a provocation if the data allow.

### Step 13 — Separate acute from chronic adaptation

**Input.** The time series.

**Output.** A classification of the change.

**Reasoning.** An acute rise in creatinine is injury; a chronic stable elevation may be reduced muscle
mass or chronic kidney disease. The same value has different meanings depending on the baseline and
the time course. A model without a baseline cannot distinguish them.

**Branch point.** If no baseline is available, use a population reference and state the assumption, or
restrict the claim to the acute setting.

### Step 14 — Separate driver from marker

**Input.** A candidate predictor.

**Output.** A classification: driver, marker, or consequence.

**Reasoning.** A driver causes the outcome; a marker correlates with it; a consequence is caused by it.
Treating a marker as a driver produces models that are predictive but not actionable, and treating a
consequence as a driver produces models that are predictive only because they detect the outcome
early. The distinction determines whether an intervention on the variable could change the outcome.

**Branch point.** If the variable is a consequence of the outcome, it is a leakage risk in a
prospective model. Check the timing.

### Step 15 — Consider subtypes and heterogeneity

**Input.** The population.

**Output.** A statement of known subtypes and their implications.

**Reasoning.** Many diseases are syndromes with distinct mechanisms. Sepsis, heart failure, and
depression all have subtypes with different responses to treatment. A model trained on a mixture can
perform well on average and poorly in every subgroup, and a treatment that works in one subtype can
appear ineffective overall.

**Branch point.** If subtypes are known but unlabelled, consider unsupervised stratification or
interaction terms, and validate subgroup performance.

### Step 16 — Classify genetic variation

**Input.** A variant.

**Output.** Germline or somatic; coding or non-coding; the appropriate nomenclature.

**Reasoning.** Germline variants are present in all cells and can be inherited; somatic variants are
acquired in a lineage and may be present at a low allele fraction. Coding variants change the protein
sequence; non-coding variants may affect regulation. The classification determines the interpretation,
the privacy requirements, and the assay used to detect it.

**Branch point.** If the variant's origin is unknown, do not report it as hereditary. State the
tissue and the assay.

### Step 17 — Interpret the variant allele fraction with purity and copy number

**Input.** A VAF, a tumour purity, and a local copy number.

**Output.** An expected VAF and an interpretation.

**Reasoning.** In a tumour sample of purity p, a heterozygous somatic variant in a diploid region has
an expected VAF of p × 0.5. A value below that suggests a subclonal variant, a copy-number loss, or a
dilution by normal tissue. Interpreting the raw VAF without purity misclassifies clonality, which
changes prognosis and treatment selection.

**Branch point.** If purity is unknown, the VAF cannot be interpreted as clonality. Report the raw
value with the limitation.

### Step 18 — Treat the microbiome as a community, not a pathogen

**Input.** Sequencing or culture data from a body site.

**Output.** A compositional summary with a stated reference.

**Reasoning.** Microbiome data are compositional: the abundances sum to a fixed total, so an increase
in one taxon mechanically decreases the reported abundance of others. Correlations between taxa are
therefore not independent, and standard correlation methods produce spurious negatives. The ratio of
bacteria to human cells is about 1.3 to 1 by current estimates, not the often-quoted 10 to 1.

**Branch point.** If the analysis treats relative abundances as absolute, the results are
compositionally biased. Use compositional methods or absolute quantification.

### Step 19 — Account for iatrogenic effects

**Input.** The treatment variables.

**Output.** A list of variables that are consequences of care.

**Reasoning.** Many recorded values reflect treatment rather than disease: a low blood pressure may be
the result of an antihypertensive, a low glucose the result of insulin, a low haemoglobin the result
of phlebotomy. A model that predicts deterioration from these variables may be predicting the
treatment that precedes deterioration, which is a form of leakage.

**Branch point.** If a variable is a treatment effect, include the treatment as a feature or exclude
the variable. Do not let it stand in for the disease.

### Step 20 — Account for age and development

**Input.** The patient's age.

**Output.** A statement of developmental stage and its implications.

**Reasoning.** Children are not small adults. Organ function, body composition, reference intervals,
drug clearance, and the immune response all change with age. A model trained on adults and applied to
children is an extrapolation with no biological basis, and paediatric reference intervals must be
age-specific.

**Branch point.** If the model is applied outside its age range, treat it as unvalidated. Do not
adjust a threshold by intuition.

### Step 21 — Choose the sampling schedule before the feature

**Input.** The process timescale and the decision window.

**Output.** A sampling interval and a feature window.

**Reasoning.** The schedule determines what can be estimated. Hourly sampling can detect a process
with a time constant of hours; daily sampling cannot. Informative sampling, in which the schedule
depends on the patient's state, biases every summary computed from the data and must be modelled or
avoided.

**Branch point.** If the sampling is informative, a feature computed from it is a proxy for the
sampling decision. Use observation-process features explicitly or restrict to scheduled measurements.

### Step 22 — Define the label with clinical input

**Input.** The outcome concept.

**Output.** An operational label with a source, a time window, and a validation.

**Reasoning.** Labels are definitions, not discoveries. A sepsis label defined by a blood culture and
an antibiotic reflects the clinical suspicion of sepsis, not sepsis itself. A readmission label
defined by a claims code reflects billing. The label's construct validity must be assessed against a
chart review or a clinical adjudication before it is used as a target.

**Branch point.** If the label cannot be validated, the model's performance is a statement about the
label, not about the disease. Report it that way.

### Step 23 — Write the biological assumption into the ledger

**Input.** The variable list and the label.

**Output.** Ledger rows with evidence and tests.

**Reasoning.** Each biological assumption becomes a testable statement: "the last observed lactate
reflects the current perfusion state within 2 hours", "the diagnosis code reflects the disease
documented in the chart", "tumour purity is above 0.4 in all samples". The ledger makes the assumption
visible and assigns a test.

**Branch point.** If an assumption cannot be tested, mark it open and state what would change the
decision if it were false.

### Step 24 — Validate against an independent biological measurement

**Input.** The model or the variable.

**Output.** A validation result.

**Reasoning.** A statistical validation shows that the variable predicts the outcome in the data. A
biological validation shows that it measures the process it claims to measure, using an independent
method: a dilution technique for cardiac output, a measured GFR for a filtration marker, a chart
review for a diagnosis label. Without it, the model may be predicting care rather than disease.

**Branch point.** If no independent measurement exists, state that the variable is a proxy with an
unvalidated mechanism and limit the claim to prediction, not mechanism.

### 5.1 The branch map

```text
Clinical question
  |
  +-- process unnamed? -------------> stop; define the target
  |
  +-- level mismatch? --------------> aggregate explicitly
  |
  +-- compartment unknown? ---------> different quantity; do not compare
  |
  +-- sampling slower than process? -> model level, not dynamics
  |
  +-- cumulative variable? ---------> do not treat as instantaneous
  |
  +-- feedback loop? ---------------> include the driver, not only the effector
  |
  +-- compensated state? -----------> use trend or challenge
  |
  +-- label from billing? ----------> validate against chart review
  |
  +-- variant VAF? -----------------> adjust for purity and copy number
  |
  +-- microbiome relative abundance? -> compositional method
```

### 5.2 The four checks that catch most biological errors

1. **Name the process, the compartment, and the timescale for every variable.** If any is unknown, the
   variable is not yet usable.
2. **Ask whether the variable is a driver, a marker, or a consequence.** The answer determines whether
   the model is actionable.
3. **Ask what the label measures: disease, treatment, or documentation.** Most label failures are
   documentation failures.
4. **Look for the compensation before trusting a normal value.** A normal value in a stressed patient
   is a statement about reserve, not about health.

## 6. Worked example

A 58-year-old man is admitted with pneumonia. He is 89.8113 kg and 180.34 cm. His observations are:
cardiac output estimated at 5.0 L/min, haemoglobin 13.4 g/dL, arterial oxygen saturation 0.97,
arterial oxygen partial pressure 92 mmHg, platelets 88 × 10⁹/L, bilirubin 2.1 mg/dL, mean arterial
pressure 62 mmHg on norepinephrine 0.08 µg/kg/min, Glasgow Coma Scale 13, creatinine 1.4 mg/dL, and
urine output 0.3 mL/kg/h over six hours. Lactate is 4.2 mmol/L and falls to 2.1 mmol/L after six
hours.

The case computes oxygen delivery and extraction, urine output, lactate clearance, the SOFA score, and
a variant allele fraction, and then derives the feature windows that the timescales imply.

### 6.1 Oxygen delivery and extraction

```text
Arterial oxygen content (from Topic 04)
  CaO2 = 1.34 x 13.4 x 0.97 + 0.003 x 92
       = 17.4173 + 0.2760
       = 17.6933 mL O2/dL

Oxygen delivery
  DO2 = cardiac output x CaO2 x 10
      = 5.0 L/min x 17.6933 mL/dL x 10
      = 884.66 mL O2/min

The factor 10 converts dL to L: 1 L = 10 dL.

Oxygen consumption
  VO2 = 250 mL O2/min (a typical resting adult value)

Oxygen extraction ratio
  O2ER = VO2 / DO2
       = 250 / 884.66
       = 0.2826 = 28.3%
```

A normal extraction ratio is about 25% at rest. A ratio above about 30% to 35% suggests that delivery
is marginal relative to demand, and a ratio above 50% suggests that the compensation is failing. The
value of 28.3% is at the upper edge of normal, not because delivery is low in absolute terms but
because demand is elevated by fever and infection.

The biological point is that a normal blood pressure does not imply adequate delivery. MAP 62 mmHg is
low, cardiac output is preserved, and the extraction ratio is near the limit. A model using MAP alone
sees one abnormal number; a model using DO2 and O2ER sees the compensation.

### 6.2 Urine output and the oliguria threshold

```text
Hourly rate
  0.3 mL/kg/h x 89.8113 kg = 26.94 mL/h

Six-hour total
  26.94 mL/h x 6 h = 161.7 mL

Oliguria threshold
  0.5 mL/kg/h x 89.8113 kg = 44.91 mL/h

Daily equivalent at the observed rate
  26.94 x 24 = 646.6 mL/day
```

The patient is below the 0.5 mL/kg/h threshold and would meet the SOFA renal criterion for urine output
below 500 mL/day if the rate continued for 24 hours. The six-hour total alone does not establish the
daily criterion, which is why the SOFA renal score uses creatinine here and the urine criterion is
noted as a trend rather than applied to a six-hour window.

The unit matters: 0.3 mL/kg/h is not 0.3 mL/h. A pipeline that stores the rate without the body-weight
basis cannot recompute the threshold for a different patient.

### 6.3 Lactate kinetics

```text
Observed fall: 4.2 mmol/L to 2.1 mmol/L over 6.0 h
Ratio: 4.2 / 2.1 = 2.000
Half-life: 6.0 h (because the ratio is exactly 2)

Rate constant
  k = ln(4.2/2.1) / 6.0
    = ln(2) / 6.0
    = 0.693147 / 6.0
    = 0.115525 /h

Half-life from k
  t_half = ln(2) / k = 6.000 h

Percent cleared
  (1 - 2.1/4.2) x 100 = 50.0%
```

Lactate clearance of 50% in six hours is a favourable trend. The calculation assumes first-order
kinetics, which is an approximation: lactate production and clearance are both concentration-dependent
and flow-dependent, and the apparent half-life changes with the clinical state.

The biological point is that lactate is a flow-and-metabolism marker, not a diagnosis. It rises with
hypoperfusion, with impaired clearance in liver failure, with seizures, with beta-agonists, and with
certain toxins. A model that treats a high lactate as evidence of sepsis will fire on all of these.

### 6.4 The SOFA score

The Sequential Organ Failure Assessment (SOFA) score sums six organ subscores. Each is computed from
the worst value in a 24-hour window.

```text
Respiration
  PaO2/FiO2 = 92 / 0.21 = 438.1
  Score: 0 (threshold is 400)

Coagulation
  Platelets 88 x 10^9/L
  Score: 2 (below 100)

Liver
  Bilirubin 2.1 mg/dL
  Score: 2 (2.0 to 5.9 mg/dL)

Cardiovascular
  MAP 62 mmHg on norepinephrine 0.08 ug/kg/min
  Score: 3 (any dose of norepinephrine at or below 0.1)

Central nervous system
  GCS 13
  Score: 1 (13 to 14)

Renal
  Creatinine 1.4 mg/dL
  Score: 1 (1.2 to 1.9 mg/dL)
  Urine output 647 mL/day equivalent does not meet the below-500 criterion.

Total SOFA = 0 + 2 + 2 + 3 + 1 + 1 = 9
```

A SOFA of 9 represents substantial organ dysfunction. The Sepsis-3 consensus defines sepsis as a
SOFA increase of at least 2 points consequent to infection, and reports that a SOFA of 2 or more
corresponds to an in-hospital mortality above 10%. A score of 9 is well above that threshold.

Two details matter for data work.

First, the score is defined on the **worst** value in a 24-hour window, not the latest. A pipeline that
takes the last observation will systematically underestimate the score.

Second, **lactate is not a SOFA component**. The Sepsis-3 task force considered adding it and did not
find that it improved predictive validity enough to justify the added complexity. Lactate is part of
the septic shock definition, which requires vasopressors to maintain a MAP of at least 65 mmHg and a
lactate above 2 mmol/L despite adequate fluid resuscitation. A model that uses lactate as a SOFA
component is not implementing SOFA.

### 6.5 Variant allele fraction in a tumour sample

A tumour sequencing report describes a sample with 60% tumour purity and a somatic single nucleotide
variant with an observed variant allele fraction of 0.18.

```text
Expected VAF for a heterozygous somatic variant in a diploid region
  VAF_expected = purity x 0.5
               = 0.60 x 0.5
               = 0.30

Observed / expected
  0.18 / 0.30 = 0.60
```

The observed fraction is 60% of the expected value. The biological interpretations include a
subclonal variant present in about 60% of tumour cells, a copy-number loss at the locus, or a
heterozygous germline variant with loss of the wild-type allele in part of the tumour. The raw 0.18
cannot distinguish them; the purity-adjusted comparison narrows the possibilities and directs the
next assay.

A report that calls 0.18 "heterozygous" is using a germline convention for a somatic measurement. The
label is wrong even though the number is right.

### 6.6 From timescales to feature windows

| Variable | Biological time constant | Feature window | Reason |
|---|---|---|---|
| Heart rate | seconds to minutes | 5–15 min rolling mean | Captures variability without averaging the signal |
| Blood pressure | seconds to minutes | 15-min rolling minimum | Hypotension episodes matter more than the mean |
| Lactate | minutes to hours | 2–6 h change | Half-life of about 6 h in this case |
| Creatinine | hours to days | 24–48 h change from baseline | Muscle buffer and renal lag |
| CRP | about 24 h | 24 h change | Synthesis lag |
| HbA1c | about 120 days | Single value | Integrates red cell lifespan |
| SOFA | 24 h | Daily worst value | Definition uses the 24-hour worst |

The table is the operational output of the biological reasoning. Each window is justified by a time
constant, not chosen by convention. A model that uses a one-hour window for creatinine is measuring
noise; a model that uses a 24-hour window for heart rate is measuring the trend and missing the
arrhythmia.

### 6.7 What the example demonstrates

| Calculation | Quantity | Biological point |
|---|---|---|
| CaO2 and DO2 | 17.69 mL/dL, 884.66 mL/min | Delivery depends on content and flow, not pressure alone |
| O2ER | 28.3% | Compensation is visible before failure |
| Urine output | 26.94 mL/h, 161.7 mL/6 h | The per-kilogram basis is part of the definition |
| Lactate clearance | 50% in 6 h, k = 0.1155/h | Lactate is a flow-and-metabolism marker |
| SOFA | 9 | Worst value in 24 h, and lactate is not a component |
| VAF | 0.18 observed vs 0.30 expected | Purity changes the interpretation |
| Feature windows | 5 min to 120 days | Windows follow time constants |

## 7. Code, query, or calculation

The script computes the physiological quantities and the SOFA total. It uses only the standard
library.

```python
"""Topic 05 verification: physiological calculations and a SOFA total."""
from math import log

print("5.1 Oxygen delivery and extraction")
co_l_min = 5.0
cao2_ml_dl = 17.6933
do2 = co_l_min * cao2_ml_dl * 10
vo2 = 250.0
print(f"  DO2 = {co_l_min} L/min x {cao2_ml_dl} mL/dL x 10 = {do2:.2f} mL/min")
print(f"  VO2 = {vo2:.0f} mL/min")
print(f"  O2ER = {vo2/do2:.4f} = {vo2/do2*100:.2f}%")

print("\n5.2 Urine output")
kg = 89.8113
rate = 0.3
print(f"  hourly = {rate} mL/kg/h x {kg:.4f} kg = {rate*kg:.2f} mL/h")
print(f"  six-hour total = {rate*kg*6:.1f} mL")
print(f"  oliguria threshold 0.5 mL/kg/h = {0.5*kg:.2f} mL/h")
print(f"  daily equivalent of the observed rate = {rate*kg*24:.0f} mL/day")

print("\n5.3 Lactate clearance")
lac0, lac6, hours = 4.2, 2.1, 6.0
k = log(lac0 / lac6) / hours
print(f"  ratio = {lac0/lac6:.3f}")
print(f"  k = ln({lac0}/{lac6})/{hours:g} = {k:.6f} /h")
print(f"  half-life = ln(2)/k = {log(2)/k:.3f} h")
print(f"  percent cleared = {(1 - lac6/lac0)*100:.1f}%")

print("\n5.4 SOFA components")
def sofa_resp(pf):
    if pf >= 400: return 0
    if pf >= 300: return 1
    if pf >= 200: return 2
    if pf >= 100: return 3
    return 4

def sofa_coag(plt):
    if plt >= 150: return 0
    if plt >= 100: return 1
    if plt >= 50: return 2
    if plt >= 20: return 3
    return 4

def sofa_liver(bili_mg_dl):
    if bili_mg_dl < 1.2: return 0
    if bili_mg_dl < 2.0: return 1
    if bili_mg_dl < 6.0: return 2
    if bili_mg_dl < 12.0: return 3
    return 4

def sofa_cns(gcs):
    if gcs >= 15: return 0
    if gcs >= 13: return 1
    if gcs >= 10: return 2
    if gcs >= 6: return 3
    return 4

def sofa_renal(creat_mg_dl, urine_ml_day):
    if creat_mg_dl >= 5.0 or urine_ml_day < 200: return 4
    if creat_mg_dl >= 3.5 or urine_ml_day < 500: return 3
    if creat_mg_dl >= 2.0: return 2
    if creat_mg_dl >= 1.2: return 1
    return 0

pf = 92 / 0.21
components = {
    "respiration": sofa_resp(pf),
    "coagulation": sofa_coag(88),
    "liver": sofa_liver(2.1),
    "cardiovascular": 3,
    "cns": sofa_cns(13),
    "renal": sofa_renal(1.4, 647),
}
print(f"  PaO2/FiO2 = 92/0.21 = {pf:.1f} -> {components['respiration']}")
for name, score in components.items():
    print(f"  {name:16s} = {score}")
print(f"  total SOFA = {sum(components.values())}")

print("\n5.5 Variant allele fraction")
purity = 0.60
expected_vaf = purity * 0.5
observed_vaf = 0.18
print(f"  expected heterozygous somatic VAF = {purity} x 0.5 = {expected_vaf:.2f}")
print(f"  observed = {observed_vaf:.2f}")
print(f"  observed/expected = {observed_vaf/expected_vaf:.2f}")

print("\n5.6 Timescales")
scales = [
    ("action potential", 0.001, "s"),
    ("cardiac cycle", 0.8, "s"),
    ("circadian cycle", 24.0, "h"),
    ("gut epithelial turnover", 4.0, "days"),
    ("red cell lifespan", 120.0, "days"),
    ("HbA1c window", 120.0, "days"),
]
for name, value, unit in scales:
    print(f"  {name:26s} {value:>8g} {unit}")
```

**Expected output.**

```text
5.1 Oxygen delivery and extraction
  DO2 = 5.0 L/min x 17.6933 mL/dL x 10 = 884.66 mL/min
  VO2 = 250 mL/min
  O2ER = 0.2826 = 28.26%

5.2 Urine output
  hourly = 0.3 mL/kg/h x 89.8113 kg = 26.94 mL/h
  six-hour total = 161.7 mL
  oliguria threshold 0.5 mL/kg/h = 44.91 mL/h
  daily equivalent of the observed rate = 647 mL/day

5.3 Lactate clearance
  ratio = 2.000
  k = ln(4.2/2.1)/6 = 0.115525 /h
  half-life = ln(2)/k = 6.000 h
  percent cleared = 50.0%

5.4 SOFA components
  PaO2/FiO2 = 92/0.21 = 438.1 -> 0
  respiration      = 0
  coagulation      = 2
  liver            = 2
  cardiovascular   = 3
  cns              = 1
  renal            = 1
  total SOFA = 9

5.5 Variant allele fraction
  expected heterozygous somatic VAF = 0.6 x 0.5 = 0.30
  observed = 0.18
  observed/expected = 0.60

5.6 Timescales
  action potential              0.001 s
  cardiac cycle                   0.8 s
  circadian cycle                  24 h
  gut epithelial turnover           4 days
  red cell lifespan               120 days
  HbA1c window                    120 days
```

**Assumptions stated by the script.**

1. CaO2 is taken from the Topic 04 calculation with haemoglobin in g/dL. It is not recomputed here.
2. VO2 of 250 mL/min is a typical resting adult value, not a measurement. A measured VO2 would change
   the extraction ratio.
3. Lactate clearance is treated as first-order. The apparent half-life depends on hepatic and renal
   clearance and on continued production.
4. The SOFA subscores use the worst value in a 24-hour window by convention, but this case supplies a
   single set of values, so the window is assumed to contain no worse value.
5. The daily urine equivalent is an extrapolation of a six-hour rate and is not a measured 24-hour
   volume.

## 8. Data, standards, and regulatory context

### 8.1 Clinical terminology and data standards

| Standard or terminology | What it covers | Binding status |
|---|---|---|
| LOINC | Observation codes, including laboratory and vital signs | Required by many interoperability profiles |
| SNOMED CT | Clinical concepts: findings, procedures, organisms | Required in several national programmes |
| ICD-11 | Disease classification and coded diagnoses | Binding for mortality and morbidity coding in adopting countries |
| HGNC | Human gene nomenclature | Community standard; required by major journals and databases |
| HGVS | Variant nomenclature | Required for clinical variant reporting |
| ClinVar | Variant-disease assertions with review status | Reference database; not a regulatory determination |
| FHIR Observation | Structure for a clinical observation | Binding where a jurisdiction adopts the profile |
| CDISC SDTM | Trial data tabulation, including biomarker domains | Required for regulatory submissions |
| AJCC TNM | Cancer stage classification | Required for cancer registration and staging |
| GA4GH | Genomic data sharing standards | Voluntary; widely adopted |

**LOINC** provides the code that says what was measured: the analyte, the property, the timing, the
specimen, the scale, and the method where relevant. A glucose measurement in serum or plasma has
different codes from one in whole blood, which is the terminology encoding of the compartment
distinction in Step 3.

**SNOMED CT** provides the concepts for clinical findings and procedures. A diagnosis of sepsis is a
SNOMED concept; a blood culture result is a LOINC observation; the organism is a SNOMED organism
concept. A data model that conflates these loses the distinction between a finding, a test, and a
result.

**HGVS** nomenclature is the standard for describing a variant at the DNA, RNA, or protein level.
`NM_000546.6:c.215C>G` identifies a transcript, a coding position, and a substitution. A variant
reported without the transcript and the reference sequence is ambiguous, because the same genomic
position can map to different coding positions in different transcripts.

**FHIR Observation** is the structure in which a measurement is exchanged: a code, a value, a unit, a
subject, an effective time, an issued time, a performer, and an interpretation. The `effectiveDateTime`
and `issued` fields are distinct, and the difference between them is often the difference between a
valid and a leaked feature.

### 8.2 Consensus definitions

**Sepsis-3** (Singer et al., *JAMA* 2016;315(8):801–810) defines sepsis as life-threatening organ
dysfunction caused by a dysregulated host response to infection, operationalised as an acute increase
in SOFA of at least 2 points. It explicitly rejected the systemic inflammatory response syndrome
(SIRS) criteria as having poor discriminant validity. It defines septic shock as vasopressors needed to
maintain a MAP of at least 65 mmHg plus a lactate above 2 mmol/L despite adequate fluid resuscitation.

The definition is a consensus, not a regulation, but it is the reference against which sepsis labels
are judged. A model that uses SIRS criteria and calls the output "sepsis" is using a superseded
definition, and reviewers will say so.

**TNM staging** (AJCC) classifies tumours by primary tumour extent (T), regional nodes (N), and
metastasis (M). The stage is a summary that changes with edition. A dataset that records "stage III"
without the edition cannot be compared across eras, because the definitions changed.

### 8.3 Regulatory context for biological data

**Genomic data** are sensitive personal data. The EU General Data Protection Regulation treats genetic
data as a special category, and the US Genetic Information Nondiscrimination Act restricts their use in
employment and insurance. A pipeline that stores germline variants must treat them differently from
somatic variants, which are usually less identifying but still sensitive.

**Biomarker qualification** is a regulatory process. The FDA's biomarker qualification programme and
the European Medicines Agency's qualification procedure allow a biomarker to be used in a defined
context of use for drug development. A biomarker that has not been qualified can still be used
exploratorily, but a claim that it is a surrogate endpoint requires the qualification evidence.

**Laboratory developed tests** and in vitro diagnostics are regulated under CLIA and IVDR
respectively. A genomic test that reports a variant classification is a diagnostic product, and its
claims are constrained by the regulation even when the underlying science is published.

### 8.4 What the standards do not constrain

- They do not tell you which variables are biologically valid for a given prediction problem.
- They do not validate a label. LOINC and SNOMED encode what was recorded, not whether it is true.
- They do not make a biomarker a surrogate.
- They do not resolve the difference between a reference interval and an individual target.
- They do not cover the physiology itself. A perfectly coded observation can be physiologically
  meaningless if it was measured in the wrong compartment or at the wrong time.

> **Note:** Clinical and regulatory content in this section is educational and is not medical or legal
> advice. Sepsis definitions and staging systems are cited as published consensus documents, not as
> instructions for care.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Health systems** build phenotype definitions for sepsis, heart failure, and other conditions from a
combination of diagnosis codes, laboratory values, medication orders, and vital signs. Mature
organisations validate the phenotype against chart review on a sample before using it as a model
label, and they report the positive predictive value of the phenotype. Immature organisations use the
code alone and discover the error when the model fails at another site.

**Pharmaceutical and diagnostics companies** distinguish exploratory biomarkers from qualified ones.
A programme that plans to use a biomarker as a surrogate endpoint builds the qualification evidence in
parallel with the clinical programme, because the qualification is a regulatory submission in its own
right.

**Genomic laboratories** report variants with HGVS nomenclature, a transcript reference, a
classification with an evidence code, and the assay's limit of detection. Somatic reports include the
tumour purity and, where relevant, the VAF, because the interpretation depends on both.

**Wearable and remote monitoring companies** face the timescale problem directly. A photoplethysmographic
heart rate is an average over a short window and is corrupted by motion; a nightly oxygen saturation
is an average over hours. A model trained on one sampling regime does not transfer to the other, and
the failure is often attributed to population shift rather than to a change in the measurement
process.

### 9.2 Case evidence

**The cell count revision.** Sender, Fuchs, and Milo (2016, *PLoS Biology* 14(8):e1002533) estimated
about 30 trillion human cells and about 38 trillion bacteria in a 70 kg reference man, replacing the
widely quoted 10:1 bacterial-to-human ratio with about 1.3:1. The revision is a documented example of
a biological constant that circulated for decades without a primary source. The lesson for data work
is that a number repeated in textbooks and slide decks may not have a traceable derivation.

**Sepsis-3 and the SIRS criteria.** The Sepsis-3 consensus replaced the SIRS-based definition after
finding that SIRS criteria had poor discriminant validity; SOFA had an area under the curve of about
0.74 in ICU patients, compared with 0.64 for SIRS in the same analysis. The documented consequence is
that sepsis alerts built on SIRS have high false-positive rates, which is a label-validity problem
rather than an algorithm problem.

**Pulse oximetry bias.** Sjoding et al. (*New England Journal of Medicine* 383, 2477–2478, 2020)
reported that occult hypoxaemia was roughly three times more frequent in Black patients than in White
patients for the same pulse oximeter readings. The device measures an optical signal and estimates
saturation through a calibration derived in a validation population. The biological measurement is
biased in a subgroup, which is a trueness problem with clinical consequences.

**Tumour purity and variant interpretation.** Clinical genomics laboratories routinely adjust variant
allele fractions for tumour purity and local copy number before classifying clonality. The practice is
documented in professional guidance for somatic variant interpretation. A pipeline that ignores purity
will systematically misclassify subclonal variants as heterozygous germline variants, with consequences
for therapy selection.

> **Note:** The cell-count revision, Sepsis-3 performance figures, and pulse oximetry finding are
> cited to the published sources. The interpretation of each as a data-science failure mode is
> inference from those sources, not a claim about a specific organisation's internal results.

### 9.3 The practical pattern

Teams that avoid biological errors do four things:

1. They name the process, compartment, and timescale for every variable in the data dictionary.
2. They validate every label against a chart review or a clinical adjudication sample.
3. They set feature windows from time constants, not from what the data happen to contain.
4. They distinguish biomarkers from surrogate endpoints in every claim they make.

The practices are cheap relative to the cost of a late-stage failure, and each one is checkable by a
reviewer who is not a biologist.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Variable used without its compartment | The same analyte has different reference intervals across specimen types | Plasma, serum, and whole blood differ | Values compared across incomparable measurements | Store the specimen type; use the compartment-specific code |
| 2 | Sampling interval shorter than the process time constant | Hourly creatinine values that fluctuate randomly | Measurement noise dominates the signal | Model learns noise | Set the window from the time constant |
| 3 | Sampling interval longer than the time constant | Daily lactate used to detect a 2-hour change | Undersampling | Missed deterioration | Choose a variable or schedule that matches the decision window |
| 4 | Cumulative variable treated as instantaneous | HbA1c used to detect today's glucose control | The analyte integrates about 120 days | Wrong response to an intervention | Match the variable to the timescale of the question |
| 5 | Informative sampling treated as random | Sicker patients have more observations | The observation process depends on the state | Features proxy the care process | Model the observation process or restrict to scheduled tests |
| 6 | Label defined by billing code | The code is assigned after discharge by a coder | Documentation, not disease | Model predicts coding intensity | Validate against chart review; report the PPV |
| 7 | Sepsis label built on SIRS | Alert fires on a large fraction of inpatients | SIRS has poor discriminant validity | Alert fatigue; dismissed signal | Use Sepsis-3 criteria; validate the label |
| 8 | Lactate treated as diagnostic of sepsis | A high lactate in a patient on a beta-agonist is called sepsis | Lactate reflects flow and metabolism, not infection | False positives | Use lactate as a severity marker, not a diagnostic |
| 9 | Tumour VAF interpreted without purity | A VAF of 0.18 is called heterozygous | Purity scales the expected fraction | Misclassified clonality | Adjust for purity and copy number |
| 10 | Variant reported without transcript | The same genomic position maps to different coding positions | Transcript ambiguity | Wrong HGVS description | Report the transcript and reference sequence |
| 11 | Microbiome relative abundance treated as absolute | An increase in one taxon forces others down | Compositional data | Spurious correlations | Compositional methods or absolute quantification |
| 12 | Normal value assumed to mean normal function | A normal blood pressure in a bleeding patient | Compensation maintains the value | Missed deterioration | Use trends, provocation, and perfusion markers |
| 13 | Adult reference interval applied to a child | A paediatric creatinine flagged normal | Age-dependent physiology | Missed disease | Age-specific intervals |
| 14 | Treatment effect treated as disease severity | A low glucose in a patient on insulin | Iatrogenic | Model learns the treatment | Include the treatment or exclude the variable |
| 15 | Bacteraemia equated with sepsis | A positive culture in a colonised patient | Colonisation versus infection | Overtreatment and a biased label | Use the clinical definition, not the culture alone |
| 16 | Biomarker treated as a surrogate endpoint | A correlate is used to claim clinical benefit | Surrogacy requires validation | Failed trial or rejected claim | State the context of use; do not claim surrogacy without evidence |

Three rows deserve emphasis.

**Row 5, informative sampling.** This is the most under-appreciated failure in clinical machine
learning. The number of observations per patient is itself a predictor of deterioration, because
clinicians order more tests when they are worried. A model that uses "number of tests in the last six
hours" as a feature will perform well in retrospective data and fail when a protocol changes the
testing schedule. The feature is not wrong; it is a measure of clinician concern, and it must be
labelled as such.

**Row 6, billing labels.** Diagnosis codes are assigned for reimbursement, often by professional
coders working from the chart after discharge. Coding intensity varies by hospital and by payer mix.
A model trained on codes learns the coding process, and its performance degrades across sites with
different coding practices. Chart review of a sample is the only way to estimate the label's positive
predictive value.

**Row 11, compositional data.** Relative abundances sum to one by construction. A change in one
component mechanically changes the others, so a standard correlation matrix will show negative
correlations that are artefacts of the constraint. The correct methods are compositional: log-ratio
transformations, or absolute quantification.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The variable measures the process it is named for | Interpretation and feature design | A proxy with a different mechanism | Independent biological validation |
| 2 | The compartment is known and consistent | Reference intervals and comparisons | Mixed specimen types | Audit the specimen field |
| 3 | The timescale of the variable matches the decision | Early detection or chronic assessment | A long-half-life marker used acutely | Compare the time constant with the decision window |
| 4 | The sampling process is independent of the patient's state | Features reflect biology | Informative sampling | Model the observation rate; test schedule changes |
| 5 | The label measures disease rather than documentation | Target validity | Billing codes, coder behaviour | Chart review; PPV estimate |
| 6 | The reference population matches the patient | Normal versus abnormal | Age, pregnancy, altitude, comorbidity | Partitioned intervals; local verification |
| 7 | Compensation has not masked the dysfunction | A normal value is informative | Reserve exhaustion | Trend and provocation features |
| 8 | The patient is within the model's developmental age range | Transportability | Paediatric or geriatric extremes | Stratified validation |
| 9 | Treatment variables are distinguished from disease variables | Causal interpretation | Iatrogenic values | Include treatment as a feature or exclude the variable |
| 10 | Genetic variants are classified with the right origin and nomenclature | Correct interpretation and privacy | Germline versus somatic confusion | Report tissue, assay, transcript, and origin |
| 11 | Tumour purity is adequate for VAF interpretation | Clonality assessment | Low purity or copy-number alteration | Report purity; adjust the expectation |
| 12 | Microbial abundances are analysed compositionally | Valid correlations | Relative abundance treated as absolute | Compositional methods |
| 13 | The biomarker is not assumed to be a surrogate | Claim validity | Surrogacy asserted without evidence | State the context of use |
| 14 | The physiological model behind the feature is correct | Feature validity | A plausible but wrong mechanism | Independent measurement; mechanistic study |

The limits of this topic:

- It does not cover pathology, pharmacology, or clinical medicine in depth. Those are separate
  disciplines, and this topic is a primer.
- It does not cover the statistical methods for compositional data, survival analysis, or longitudinal
  modelling.
- It does not validate any biomarker or label. It provides the questions that a validation must
  answer.
- It does not cover genomics in depth. Variant interpretation is a specialty with its own standards.
- It does not make clinical recommendations. The numerical examples are illustrative.

## 12. Exercises

### Exercise 1 — Levels and timescales (application)

For each variable, state the level of organisation, the compartment, the type (state, rate, or
cumulative), and the characteristic timescale: serum sodium; HbA1c; urine output; heart rate;
tumour VAF; faecal calprotectin; blood pressure; eGFR.

**Deliverable.** A four-column table.

**Acceptance criteria.** Sodium: molecule/plasma/state/minutes to hours; HbA1c: molecule/red
cell/cumulative/about 120 days; urine output: organ/urine/rate/minutes to hours; heart rate:
organ/plasma-independent/state/seconds; VAF: molecule/tumour tissue/state/not a time series; BP:
system/arterial/state/seconds to minutes; eGFR: organ/plasma-derived estimate/state/days; calprotectin:
molecule/stool/state/days.

**Time.** 25 minutes.

### Exercise 2 — Choose a sampling interval (design)

A patient is monitored for lactic acidosis. Lactate has an effective half-life of about 6 hours in this
patient, and the clinical decision window is 12 hours. Recommend a sampling interval, a feature window,
and a minimum number of observations needed to estimate a trend. Justify each with the time constant.

**Deliverable.** A short memo with the three recommendations and the reasoning.

**Acceptance criteria.** Sampling every 2–4 hours; feature window of 6–12 hours; at least three
observations to estimate a slope; the justification references the half-life and the decision window;
the memo states that first-order kinetics is an approximation.

**Time.** 25 minutes.

### Exercise 3 — Feedback loops (recall and application)

For each pair, identify the sensor, the controller, the effector, and whether the loop is negative or
positive: blood pressure and heart rate during haemorrhage; blood glucose and insulin; plasma calcium
and parathyroid hormone; clotting and thrombin.

**Deliverable.** A table with four rows and four columns.

**Acceptance criteria.** Haemorrhage: baroreceptor, autonomic control, heart rate and vasoconstriction,
negative; glucose: pancreatic beta cell, insulin, glucose uptake, negative; calcium: parathyroid
calcium-sensing receptor, PTH, bone and kidney, negative; clotting: thrombin generation, amplification,
fibrin, positive.

**Time.** 30 minutes.

### Exercise 4 — State, rate, or cumulative (application)

Classify each as state, rate, or cumulative, and state one modelling consequence: serum creatinine;
creatinine clearance; HbA1c; urine output; pack-years; lactate clearance; blood pressure; cumulative
antibiotic days.

**Deliverable.** A table with the classification and the consequence.

**Acceptance criteria.** Creatinine state; clearance rate; HbA1c cumulative; urine output rate;
pack-years cumulative; lactate clearance rate; blood pressure state; antibiotic days cumulative. Each
consequence names a missing-data or feature-window implication.

**Time.** 25 minutes.

### Exercise 5 — Variant classification (application)

A tumour sample has 70% purity. A somatic variant in a diploid region has an observed VAF of 0.31.
Compute the expected VAF for a heterozygous variant and the observed-to-expected ratio. State two
biological explanations for a ratio below 1.

**Deliverable.** Two computed values and two explanations.

**Acceptance criteria.** Expected 0.35; ratio 0.886; explanations include subclonality and copy-number
loss; the answer states that a germline variant would be expected at about 0.5 in a pure sample and at
0.35 at this purity, so origin cannot be settled by VAF alone.

**Time.** 25 minutes.

### Exercise 6 — Physiological calculations (calculation)

A patient has cardiac output 4.2 L/min, haemoglobin 9.8 g/dL, arterial oxygen saturation 0.91, and
arterial oxygen partial pressure 70 mmHg. Compute CaO₂, DO₂, and the oxygen extraction ratio assuming
VO₂ = 250 mL/min. State whether the extraction ratio suggests marginal delivery.

**Deliverable.** Three values with units and a one-sentence interpretation.

**Acceptance criteria.** CaO₂ = 12.16 mL/dL; DO₂ = 510.7 mL/min; O2ER = 48.9%; the interpretation
states that the ratio is high and suggests marginal delivery relative to demand.

**Time.** 30 minutes.

### Exercise 7 — Critique a label (critique)

A sepsis model is trained on a label defined as "ICD code for sepsis plus a blood culture order within
24 hours of admission." Critique the label on biological and operational grounds, and propose a
better definition with its data requirements.

**Deliverable.** A one-page critique with three defects and a proposed definition.

**Acceptance criteria.** Defects include coding dependence, culture ordering as a proxy for clinical
suspicion, and the mismatch with Sepsis-3; the proposed definition uses SOFA change plus suspected
infection with a stated time window; the data requirements are named.

**Time.** 40 minutes.

### Exercise 8 — Design a feature set (design)

For a model predicting deterioration in hospitalised patients with infection, design a feature set of
8 to 12 features. For each, state the biological process, the compartment, the timescale, the window,
and whether it is a driver, marker, or consequence.

**Deliverable.** A table of 8 to 12 rows plus a one-paragraph justification of the windows.

**Acceptance criteria.** At least one feature from each of respiratory, cardiovascular, renal, and
inflammatory domains; every feature has a stated window justified by a time constant; at least one
feature is explicitly labelled a consequence and justified for inclusion or exclusion; no feature is
included "because it is usually available."

**Time.** 50 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Variable | Level | Compartment | Type | Timescale |
|---|---|---|---|---|
| Serum sodium | Molecule | Plasma | State | Minutes to hours |
| HbA1c | Molecule | Red cell | Cumulative | About 120 days |
| Urine output | Organ | Urine | Rate | Minutes to hours |
| Heart rate | Organ | — | State | Seconds |
| Tumour VAF | Molecule | Tumour tissue | State | Not a time series |
| Faecal calprotectin | Molecule | Stool | State | Days |
| Blood pressure | Organ system | Arterial | State | Seconds to minutes |
| eGFR | Organ (estimated) | Plasma-derived | State | Days |

The eGFR row is the one that requires care: it is an estimate of an organ-level function derived from a
molecular plasma measurement, so its level is organ but its compartment of origin is plasma. The
distinction matters when the estimate is compared with a measured GFR.

### Answer to Exercise 2

With a half-life of 6 hours, the lactate concentration changes appreciably over 2 to 4 hours. Sampling
every 2 to 4 hours gives at least three observations within the 12-hour decision window and allows a
slope to be estimated.

A feature window of 6 to 12 hours captures at least one half-life, so a change of clinical interest is
visible. A window shorter than 3 hours would be dominated by measurement noise and by production
fluctuations; a window longer than 24 hours would average across the decision point.

Three observations are the minimum for a slope with a check on curvature; four or more allow a
non-linear trend to be detected. The first-order assumption is an approximation: lactate production
and clearance are flow-dependent, so the apparent half-life changes as the patient's state changes.

### Answer to Exercise 3

| Process | Sensor | Controller | Effector | Loop |
|---|---|---|---|---|
| Haemorrhage | Baroreceptor | Autonomic nervous system | Heart rate, vasoconstriction | Negative |
| Glucose | Pancreatic beta cell | Insulin secretion | Glucose uptake and storage | Negative |
| Calcium | Parathyroid calcium-sensing receptor | Parathyroid hormone | Bone resorption, renal reabsorption | Negative |
| Clotting | Thrombin generation | Coagulation cascade amplification | Fibrin formation | Positive |

The clotting loop is positive: thrombin generates more thrombin, which is why clotting is rapid and
threshold-like rather than gradual. A model that assumes a linear response will misrepresent it.

### Answer to Exercise 4

| Variable | Type | Consequence |
|---|---|---|
| Serum creatinine | State | Can be interpolated over hours; baseline needed for acute interpretation |
| Creatinine clearance | Rate | Cannot be recovered from a single value; needs dose and AUC or a timed collection |
| HbA1c | Cumulative | Cannot be imputed from one glucose; represents about 120 days |
| Urine output | Rate | Needs a collection interval; a single volume is not a rate |
| Pack-years | Cumulative | Cannot be reconstructed without the smoking history |
| Lactate clearance | Rate | Needs two values and a time interval |
| Blood pressure | State | Interpolatable over minutes; a single value hides lability |
| Cumulative antibiotic days | Cumulative | Needs the full administration record |

The distinction determines missing-data handling: a missing state can be interpolated, a missing rate
cannot, and a missing cumulative exposure cannot be reconstructed from a single observation.

### Answer to Exercise 5

```text
Expected VAF = purity x 0.5 = 0.70 x 0.5 = 0.35
Observed / expected = 0.31 / 0.35 = 0.886
```

Two explanations for a ratio below 1:

1. **Subclonality.** The variant is present in only a fraction of tumour cells, so the allele fraction
   is diluted by tumour cells that do not carry it.
2. **Copy-number loss.** If the wild-type allele is lost and the variant is present on the retained
   copy, the expected fraction can exceed 0.5; conversely, a copy-number gain of the wild-type allele
   dilutes the variant fraction. A copy-number-neutral loss of heterozygosity changes the expectation
   in the other direction.

A germline heterozygous variant at 70% purity would also be expected near 0.35 if the tumour is
diploid and the variant is not copy-number altered, so VAF alone cannot distinguish germline from
somatic. The assay, the tissue, and a matched normal sample are required.

### Answer to Exercise 6

```text
CaO2 = 1.34 x 9.8 x 0.91 + 0.003 x 70
     = 11.950 + 0.210
     = 12.160 mL/dL

DO2 = 4.2 L/min x 12.160 mL/dL x 10
    = 510.72 mL/min

O2ER = 250 / 510.72
     = 0.4895 = 49.0%
```

An extraction ratio of about 49% is well above the normal 25% and indicates that delivery is marginal
relative to demand. At this level, a further fall in delivery or a rise in demand would exceed the
maximum extraction and anaerobic metabolism would increase, raising lactate. The calculation is a
reason to look for the cause of the low delivery, not a diagnosis in itself.

### Answer to Exercise 7 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Coding critique | 25% | Codes reflect documentation and reimbursement, not disease; coding intensity varies |
| Culture critique | 25% | A culture order reflects clinical suspicion and local practice; cultures are often negative in sepsis |
| Definition mismatch | 25% | The label does not implement Sepsis-3; SOFA change and suspected infection are missing |
| Proposed definition | 25% | SOFA increase of at least 2 with suspected infection, a stated time window, and named data requirements |

A weak critique says the label is "not accurate" without naming the mechanism, or proposes a better
label without the data needed to compute it.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Domain coverage | 25% | Respiratory, cardiovascular, renal, and inflammatory features present |
| Window justification | 30% | Each window is tied to a time constant, not to data availability |
| Driver/marker/consequence | 25% | Each feature labelled, with a justification for inclusion or exclusion |
| Parsimony | 10% | 8–12 features, not 50 |
| Leakage awareness | 10% | At least one consequence feature identified and handled |

A weak design lists every available variable and justifies the set by availability. A strong design
starts from the process, chooses one or two proxies per domain, and states what each proxy can and
cannot detect.

## 14. Interview questions

### Question 1 — "What does a serum creatinine measure?"

**Strong answer.** "It is a plasma concentration of a muscle breakdown product that is cleared mainly by
glomerular filtration. It proxies filtration but lags it by hours to days, depends on muscle mass and
diet, and is affected by drugs that block tubular secretion. It is not a measurement of GFR, and an
eGFR is an estimate with its own uncertainty."

**Weak answer.** "It measures kidney function." That is the shorthand, and it hides the lag, the
muscle-mass dependence, and the difference between concentration and clearance.

### Question 2 — "Why does the timescale of a variable matter for feature engineering?"

**Strong answer.** "Because a feature window should match the time constant of the process. A creatinine
has a time constant of hours to days, so a one-hour window measures noise. A heart rate has a time
constant of seconds, so a 24-hour mean hides arrhythmia. The window is a modelling choice with a
biological justification, not a default."

**Weak answer.** "Because different variables are measured at different frequencies." Frequency is the
sampling rate; the time constant is the biology, and the two must be matched deliberately.

### Question 3 — "How would you validate a sepsis label?"

**Strong answer.** "Against a chart review or a clinical adjudication sample, using the Sepsis-3
definition: suspected infection plus an acute SOFA increase of at least 2. I would report the positive
predictive value and the sensitivity of the label, and I would check whether the label's performance
differs by site, because coding and culture practices vary."

**Weak answer.** "Check the ICD code against the problem list." Both are documentation artefacts, so
the comparison does not establish validity.

### Question 4 — "What is the difference between a biomarker and a surrogate endpoint?"

**Strong answer.** "A biomarker is any measurable indicator of a biological process. A surrogate
endpoint is a biomarker used in place of a clinical outcome, and it requires evidence that a treatment
effect on the biomarker predicts the effect on the outcome. Most biomarkers are not surrogates, and
claiming surrogacy without that evidence is a common cause of late-stage failure."

**Weak answer.** "A surrogate is a biomarker that is accepted by regulators." Acceptance is the
procedural consequence, not the definition.

### Question 5 — "Why can a normal blood pressure be a sign of severe illness?"

**Strong answer.** "Because compensation maintains the pressure until reserve is exhausted.
Vasoconstriction and tachycardia can hold the pressure normal while cardiac output falls. The
informative features are the trend, the extraction ratio, lactate, and urine output, not the pressure
alone."

**Weak answer.** "Because blood pressure is unreliable." It is reliable for what it measures; the
problem is that it measures pressure, not perfusion.

### Question 6 — "What is the central dogma and where does it break?"

**Strong answer.** "DNA is transcribed to RNA and translated to protein. The exceptions that matter for
data work are reverse transcription in retroviruses, alternative splicing that produces multiple
proteins per gene, non-coding RNAs, epigenetic regulation, and post-translational modification. The
practical consequence is that a variant is not a phenotype and a transcript abundance is not a protein
abundance."

**Weak answer.** "DNA makes RNA makes protein, always." The exceptions are not trivia; they are why
genomic prediction is probabilistic.

### Question 7 — "A tumour report says VAF 0.2. What else do you need?"

**Strong answer.** "The tumour purity, the local copy number, and whether the variant is germline or
somatic. At 60% purity a heterozygous somatic variant should be about 0.30, so 0.20 suggests
subclonality or a copy-number effect. Without purity the number cannot be interpreted."

**Weak answer.** "Nothing; 0.2 means 20% of cells have it." That interpretation assumes pure tumour,
diploidy, and heterozygosity, none of which is established.

### Question 8 — "What is informative sampling and why is it a problem?"

**Strong answer.** "It is sampling whose frequency or timing depends on the patient's state. Sicker
patients get more tests, so the number of observations and the recency of the last observation become
predictors of the outcome. A model can learn the clinician's behaviour rather than the disease, and it
fails when the monitoring protocol changes."

**Weak answer.** "It means missing data." It is a specific mechanism of missingness and observation,
not a generic data-quality issue.

### Question 9 — "Why are microbiome abundances analysed with compositional methods?"

**Strong answer.** "Because relative abundances sum to a fixed total, so an increase in one taxon
mechanically decreases the others. Standard correlations therefore produce spurious negative
associations. Log-ratio transformations or absolute quantification are required."

**Weak answer.** "Because the data are non-normal." Non-normality is a distributional issue; the
constraint is structural.

### Question 10 — "What is the difference between germline and somatic variants, and why does it matter for privacy?"

**Strong answer.** "Germline variants are present in all cells and can be inherited; somatic variants
are acquired in a lineage and may be present at low allele fraction. Germline data are identifying and
can reveal risks to relatives, which is why they are treated as a special category under GDPR and
restricted in employment and insurance under GINA in the US."

**Weak answer.** "Germline is inherited, somatic is acquired." True but incomplete: the privacy and
interpretation consequences are the reason the distinction matters in data work.

### Question 11 — "What does the SOFA score measure, and what is it not?"

**Strong answer.** "It summarises acute organ dysfunction across six systems using the worst value in a
24-hour window. It is a severity score, not a diagnosis. It does not include lactate, and a SOFA of 2
or more in the context of infection is the Sepsis-3 operational criterion."

**Weak answer.** "It measures how sick the patient is." It measures organ dysfunction, which is one
component of sickness, and it is defined by a specific set of variables.

### Question 12 — "How would you decide whether a variable is a driver or a marker?"

**Strong answer.** "By timing and by mechanism. A driver precedes the outcome and lies on a causal path;
a marker correlates without being on the path; a consequence follows the outcome. I would check the
timing in the data, look for an intervention that changes the variable, and use a causal diagram to
state the assumed path."

**Weak answer.** "By whether it is statistically significant." Significance does not distinguish a
driver from a marker.

## 15. Key terms

- **Allostasis** — adaptation of set points to sustained demand.
- **Anaemia** — reduced haemoglobin or red cell mass.
- **Antibody** — an immunoglobulin that binds an antigen.
- **Antigen** — a molecule recognised by the immune system.
- **ATP** — the principal energy currency of the cell.
- **Bacteraemia** — bacteria in the blood.
- **Biomarker** — a measurable indicator of a biological process.
- **Cell** — the basic structural and functional unit of life.
- **Central dogma** — DNA to RNA to protein, with documented exceptions.
- **Circadian rhythm** — an approximately 24-hour endogenous cycle.
- **CNV** — copy number variant.
- **Colonisation** — presence of organisms without disease.
- **Compartment** — a distinct fluid or tissue space.
- **Comorbidity** — an additional condition.
- **Cytokine** — a signalling molecule of the immune system.
- **Dysbiosis** — a disrupted microbial community.
- **Enzyme** — a protein that catalyses a reaction.
- **Expressivity** — the degree of phenotype in carriers of a variant.
- **Gene** — a DNA segment encoding a functional product.
- **Genome** — the complete DNA sequence of an organism.
- **Genotype** — the combination of alleles at a locus.
- **Germline variant** — a variant present in all cells.
- **HGNC** — HGNC gene nomenclature.
- **HGVS** — variant nomenclature standard.
- **Homeostasis** — maintenance of internal variables near set points.
- **Iatrogenic** — caused by medical care.
- **Indel** — an insertion or deletion.
- **Inflammation** — a coordinated response to injury or infection.
- **Ligand** — a molecule that binds a receptor.
- **Metabolism** — the chemical reactions sustaining life.
- **Microbiome** — the community of microorganisms in a body site.
- **Negative feedback** — a loop that counteracts deviation.
- **Organ** — a structure of several tissues performing a function.
- **Organelle** — a membrane-bound structure with a specialised function.
- **Penetrance** — the proportion of carriers who express the phenotype.
- **Phenotype** — an observable characteristic.
- **Positive feedback** — a loop that amplifies deviation.
- **Protein** — a chain of amino acids with a function.
- **Receptor** — a molecule that binds a ligand and transduces a signal.
- **Reference interval** — the central 95% of a reference population.
- **Sepsis** — life-threatening organ dysfunction from a dysregulated response to infection.
- **Set point** — the value a feedback loop defends.
- **SNV** — single nucleotide variant.
- **SOFA** — Sequential Organ Failure Assessment score.
- **Somatic variant** — a variant acquired in a cell lineage.
- **Surrogate endpoint** — a biomarker used in place of a clinical outcome.
- **Tissue** — a group of cells with a common function.
- **Tumour purity** — the fraction of cells in a sample that are tumour.
- **Turnover** — the rate of replacement of a cell or molecule.
- **VAF** — variant allele fraction.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Physiology and cell biology

- Alberts, B. et al., *Molecular Biology of the Cell*, 7th edition, W. W. Norton, 2022.
  https://wwnorton.com/books/9780393884821
- Guyton, A. C., Hall, J. E., *Guyton and Hall Textbook of Medical Physiology*, 14th edition,
  Elsevier, 2020. https://www.elsevier.com/books/guyton-and-hall-textbook-of-medical-physiology/hall/978-0-323-59712-8
- Hanahan, D., Weinberg, R. A., "Hallmarks of Cancer: The Next Generation", *Cell* 144(5), 646–674,
  2011. https://doi.org/10.1016/j.cell.2011.02.013
- Sender, R., Fuchs, S., Milo, R., "Revised Estimates for the Number of Human and Bacteria Cells in
  the Body", *PLoS Biology* 14(8), e1002533, 2016. https://doi.org/10.1371/journal.pbio.1002533

### Clinical definitions and evidence

- Singer, M. et al., "The Third International Consensus Definitions for Sepsis and Septic Shock
  (Sepsis-3)", *JAMA* 315(8), 801–810, 2016. https://doi.org/10.1001/jama.2016.0287
- Seymour, C. W. et al., "Assessment of Clinical Criteria for Sepsis", *JAMA* 315(8), 762–774, 2016.
  https://doi.org/10.1001/jama.2016.0288
- Shankar-Hari, M. et al., "Developing a New Definition and Assessing New Clinical Criteria for
  Septic Shock", *JAMA* 315(8), 775–787, 2016. https://doi.org/10.1001/jama.2016.0289
- Sjoding, M. W. et al., "Racial Bias in Pulse Oximetry Measurement", *New England Journal of
  Medicine* 383, 2477–2478, 2020. https://doi.org/10.1056/NEJMc2029240

### Terminologies and genomic standards

- LOINC, *Logical Observation Identifiers Names and Codes*. https://loinc.org/
- SNOMED International, *SNOMED CT*. https://www.snomed.org/
- World Health Organization, *ICD-11*. https://icd.who.int/
- HGNC, *HUGO Gene Nomenclature Committee*. https://www.genenames.org/
- HGVS, *Human Genome Variation Society nomenclature*. https://varnomen.hgvs.org/
- NCBI, *ClinVar*. https://www.ncbi.nlm.nih.gov/clinvar/
- HL7, *FHIR R5 Observation resource*. https://hl7.org/fhir/R5/observation.html
- CDISC, *Study Data Tabulation Model (SDTM)*. https://www.cdisc.org/standards/foundational/sdtm
- AJCC, *Cancer Staging Manual*, 8th edition. https://www.springer.com/gp/book/9783319406176
- Global Alliance for Genomics and Health, *GA4GH standards*. https://www.ga4gh.org/

### Data protection

- Regulation (EU) 2016/679, *General Data Protection Regulation*.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- US Genetic Information Nondiscrimination Act of 2008 (GINA). https://www.eeoc.gov/laws/statutes/gina.cfm

> **Note:** Textbook URLs are publisher pages. The physiology content in this topic is standard
> material available in any current edition; the citations identify a specific edition rather than a
> unique claim.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`04_dimensional-analysis.md`](04_dimensional-analysis.md)
- Next topic: [`06_pharmacokinetics.md`](06_pharmacokinetics.md)
- Measurement and units: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Epidemiological measures built on these variables: [`07_epidemiology-basics.md`](07_epidemiology-basics.md)
- Device measurement of physiological signals: [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md)
- Assumption recording: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `03_cohort_pipeline.ipynb` builds a cohort from clinical data and
  is the natural place to apply the label-validity material in section 9.
