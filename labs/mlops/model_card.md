# Model Card — Reference Risk Service

## Intended use
Teaching demonstration of a production service. **Not a clinical device. Not for patient care.**

## Model
Logistic regression on age, systolic blood pressure, creatinine, and prior admissions. Coefficients are synthetic/illustrative.

## Output
Probability of the synthetic outcome, threshold, routine/review decision, and model version.

## Evaluation
The sample model is not clinically validated. Tests cover API behavior, input validation, and metrics only.

## Limitations
No external validation, calibration study, subgroup analysis, or workflow evaluation. Do not use with real patients.

## Monitoring
Request rate, error rate, latency, prediction volume, input ranges, and model version. Drift script provided for numeric features.

## Change control
Any model change requires revalidation, a new version, and a documented rollback path. In a regulated product, assess PCCP coverage.
