# Starter Labs

These notebooks are **student starters**, not solution keys. Each contains a setup cell, TODO cells, and acceptance checks. Instructor sample solutions live in `solutions/` and should not be distributed before the due date.

| Notebook | Module | Purpose |
|---|---|---|
| `01A_unit_conversion.ipynb` | 01A | Units, conversions, dimensional analysis, plausibility |
| `03_cohort_pipeline.ipynb` | 03 | Cohort definition, temporal features, tests |
| `06_calibration_decision_curve.ipynb` | 06 | Calibration, Brier score, decision curves |
| `11_ecg_quality.ipynb` | 11 | ECG quality, R-peak detection, heart rate |
| `14B_quantization.ipynb` | 14B | Affine quantization, calibration, error analysis |

## Reference stacks

| Path | Module | Contents |
|---|---|---|
| `cloud/` | 26 | Terraform modules, Kubernetes manifests, Redpanda/Kafka producer/consumer |
| `mlops/` | 15 | FastAPI service, Dockerfile, tests, CI, K8s, Prometheus rules, drift, runbook |

These are teaching scaffolds. They are not HIPAA-compliant or clinically validated.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

`requirements.txt` is intentionally minimal. Add dataset-specific packages only when needed.

## Data

Use synthetic or public teaching data. Do not place PHI in the repository. If a notebook needs credentialed data, use a clearly marked demo substitute and document the difference.

## Submission

Submit the notebook with outputs, a short methods note, and the acceptance checks completed. For labs with figures, export a PDF or HTML report.
