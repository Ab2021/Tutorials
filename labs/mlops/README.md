# MLOps Reference Stack

A minimal production-shaped service for Module 15.

## Run locally

```bash
cd labs/mlops
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Test

```bash
python -m pytest -q
docker build -t biomed-risk:0.1.0 .
```

## Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /predict`
- `GET /metrics`

## What is intentionally missing

- Real model validation and clinical evidence.
- A real database or feature store.
- TLS, ingress, and identity provider.
- A production model registry and artifact signing.
- HIPAA/BAA review.

Use the K8s manifests as a starting point, not a compliant deployment.

## Verification status

- `python -m pytest -q`: 5 tests passed in the authoring environment.
- Python modules compile.
- YAML manifests parse.
- Docker image and Kubernetes deployment were not run because Docker/kubectl were unavailable in the authoring environment.
