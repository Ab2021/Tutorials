# Model service and container

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 08 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: Python basics, HTTP and JSON basics, a terminal, and Modules 20, 26, and 29  
> Version 2026.5

## 1. Why this topic exists

A planning team has a demand model that works in a notebook.
A replenishment application needs a forecast for an item at a specific location.
An analyst could copy values between the notebook and the application every morning.
That handoff is slow, difficult to audit, and vulnerable to using the wrong model or input.
A service makes the calculation callable through a stable interface.
The interface does not make the underlying forecast correct.

The business decision is how much stock to review or order.
The service produces a forecast of units per day, not an order.
A downstream policy may turn the forecast into a proposed plan.
A target is a desired service or inventory level.
A commitment is an authorized obligation to buy or deliver.
None of those is created by the example endpoint.

A service owner needs a reproducible artifact, a defined request, and a clear failure response.
An operations owner needs to distinguish a live process from a process ready to serve.
A security owner needs to know what crosses the boundary and what is logged.
A model owner needs to identify the calculation and its version.
The lab joins these needs in a deliberately small implementation.

You will package a transparent deterministic forecast as a Python HTTP application.
Its input is synthetic recent demand and a known horizon.
Its output is an arithmetic baseline and a model version.
It has request validation, liveness and readiness endpoints, and structured request logs.
You will run it locally, exercise automated tests, build a Docker image, and verify the container.
The tests are instructions for you to execute; no results are claimed here.

This is a sandbox, not a production deployment.
It has no authentication, transport encryption, request rate limiting, registry signing, or orchestration.
It must not be exposed to the public internet or given real customer information.
Installing packages and building the image require access to trusted registries.
Docker additionally requires an approved installation and a running daemon.
The commands assume a POSIX-compatible shell; Windows users can use an approved shell such as WSL.
A local virtual environment requires Python 3.11 and network access for installation.

## 2. Learning outcomes

After this topic you can:

1. Trace a forecast request from an HTTP client to a model and back, identifying the decision boundary.
2. Specify and validate request and response units, bounds, and version identity.
3. Calculate the sample forecast by hand and compare it with the service result.
4. Build and run the supplied API locally with pinned direct dependencies.
5. Explain liveness, readiness, and their different failure consequences.
6. Test valid and invalid requests without leaking payloads into logs.
7. Build a non-root Docker image and verify it through the host loopback interface.
8. Assess reproducibility, security, monitoring, and release limitations before production use.

Exercises 1–8 in section 12 map to these outcomes.
The worked example and tests use only fictional demand in units per day.
Passing them checks a small interface contract, not business fitness or predictive accuracy.

## 3. The core mental model

```text
planner / caller
    |
    | JSON: item, location, recent daily demand, horizon
    v
HTTP boundary -- reject malformed or invalid input --> 4xx response
    |
    v
validated request -- deterministic baseline --> forecast + version
    |                                      |
    +---------------- request metadata ----+--> structured log
    |
    +--> GET /health/live  : process can answer HTTP
    +--> GET /health/ready : model is available for inference

source + pinned direct packages + model code + base image
    -> container image -> locally bound container -> verified responses
```

The boundary accepts untrusted bytes and returns an explicit result or error.
Validation occurs before inference.
The model receives numeric daily observations rather than arbitrary JSON.
A response states the horizon and units so the caller does not mistake total units for units per day.
The request also carries a model identity so the caller can correlate behavior with a release.

The model here is a seven-observation-or-shorter simple mean.
It predicts the same daily quantity for each future day.
Multiplying that mean by the horizon gives a horizon total.
It does not train, call an external model registry, or persist state.
Calling this a model service describes the interface around a baseline calculation, not proof of model quality.

Liveness asks whether the running application can respond.
Readiness asks whether this instance should receive prediction traffic.
Both are HTTP requests, but they answer different operational questions.
If a future model fails to load, readiness should fail while the process may still be live.
Our in-memory model is created on application startup; the endpoint checks its presence.
There is no external dependency to probe in this exercise.

Logs describe the request path, method, status, elapsed time, and a generated request ID.
They deliberately omit request and response bodies.
A log line is evidence of an observed application event, not an audit of a business decision.
Container image identity, dependency resolution, host configuration, and inputs also matter for reproducibility.
A mutable image tag or a pinned top-level package list alone is not a complete release record.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| API | Application programming interface; here an HTTP contract | Assuming every HTTP response is a successful prediction |
| HTTP | Hypertext Transfer Protocol for request and response exchange | Treating transport alone as authorization |
| JSON | JavaScript Object Notation used for request and response bodies | Assuming a parsed JSON value satisfies domain rules |
| Endpoint | Address and method for one service operation | Calling a container port an endpoint contract |
| Request | Client-sent method, path, headers, and optional body | Treating client data as trusted model input |
| Response | Status, headers, and optional body returned by the service | Reading an error body as a prediction |
| Status code | HTTP integer indicating response class | Returning 200 for a rejected request |
| Schema | Structural and type constraints on a message | Assuming schema validation proves data truth |
| Domain validation | Constraints derived from the meaning of data | Accepting negative demand because it is numeric |
| Forecast | Estimate of future demand, here units per day | Treating the estimate as a purchase commitment |
| Horizon | Number of future days covered by a prediction | Mixing horizon totals and daily rates |
| Observation | One historical daily demand value in units | Treating a missing day as zero without justification |
| Baseline | Simple reference method for comparison | Assuming the baseline is an approved production model |
| Model version | Identifier for a specified model calculation | Using the version as a guarantee of quality |
| Inference | Applying a model to an input | Confusing inference with model training |
| Liveness | Whether the process can respond to a basic probe | Making it depend on every optional downstream service |
| Readiness | Whether this instance can serve the intended operation | Equating process existence with a usable model |
| HTTP 422 | Unprocessable request under this framework's validation | Treating every 422 as a server outage |
| HTTP 503 | Service unavailable, used here for an unavailable model | Retrying indefinitely without a limit |
| Structured log | Machine-readable event with named fields | Dumping entire payloads into a log |
| Request ID | Identifier for correlating one request and its log event | Treating an ID as authentication |
| Latency | Elapsed request-processing time, here milliseconds | Calling a single sample a latency percentile |
| Fixture | Fixed synthetic test input | Presenting the fixture as a real customer record |
| Unit test | Focused check of an isolated behavior | Assuming it verifies a deployed image |
| Integration test | Check across collaborating components | Assuming an in-process HTTP test exercises the network |
| Container | Isolated process using an image and the host kernel | Treating it as a complete security boundary |
| Image | Layered artifact used to start a container | Assuming a tag is immutable |
| Image digest | Content-addressed identifier for an image manifest | Confusing it with a software bill of materials |
| Build context | Files sent to the Docker builder | Including credentials or virtual environments in it |
| Dependency pin | Exact selected package version | Assuming direct pins freeze every transitive artifact |
| Lockfile | Complete resolved dependency set, ideally with hashes | Calling a short direct requirements list a full lock |
| Health check | Observation used to assess a process or instance | Assuming Docker health automatically restarts an unhealthy container |
| Loopback | Local-only network interface, such as 127.0.0.1 | Publishing a sandbox API on all interfaces by accident |
| ASGI | Asynchronous Server Gateway Interface between Python app and server | Confusing the web framework with the server process |
| Pydantic | Python library for validating structured data | Assuming field validation alone proves model suitability |
| Non-root user | Container process identity without root user ID | Assuming it makes all mounted host files safe |
| OpenAPI | Machine-readable description of HTTP operations | Treating generated documentation as an access control |
| Backtest | Evaluation on earlier historical periods | Treating one synthetic input as a backtest |
| Drift | Change in input or outcome distribution over time | Treating every change as evidence that the model is wrong |

## 5. The mechanics, step by step

### 5.1 Establish the decision and scope

1. Name the decision owner and downstream consumer.
   The example consumer is a replenishment planner's application.
   Its decision remains outside this service.
2. Specify the output unit before choosing the endpoint shape.
   This service returns both average units per day and total forecast units over the requested horizon.
3. Specify the data available at prediction time.
   The caller supplies one through seven consecutive daily demand observations.
   The service does not fetch historical data or verify their dates.
4. Separate model behavior from operational packaging.
   The arithmetic mean is the baseline; HTTP and Docker make the baseline callable and portable.
5. Define acceptance before coding.
   A valid fixture returns the independently calculated values and model version.
   Invalid values receive a validation error and do not invoke the forecast.

The business owner should approve units, horizon, and the meaning of missing days.
A positive test of the API does not substitute for that approval.
A real release also needs measured forecast error on representative historical data.

### 5.2 Define the request contract

Use `POST /predict` and `application/json`.
The request contains four fields:

- `item_id`: a nonempty identifier containing letters, digits, underscore, or hyphen, at most 40 characters.
- `location_id`: the same identifier policy.
- `daily_demand`: one to seven finite nonnegative numeric values, each at most 1000000 units.
- `horizon_days`: an integer from 1 through 28 days.

The numeric upper limit is a sandbox guardrail, not a universal supply-chain limit.
The identifier restrictions reduce accidental whitespace and log-unfriendly characters.
They do not authorize access to a particular item or location.
The model does not use the identifiers in its calculation; they provide response context.

Reject unknown fields rather than silently ignoring a misspelled input.
Reject booleans as numeric values even though Python considers `bool` a subclass of `int`.
Reject numeric strings rather than silently converting them.
Reject NaN and infinity as nonfinite inputs.
Reject an empty list because its arithmetic mean is undefined.
Return the framework's 422 validation response for these failures.
The response can describe the invalid field, so avoid placing secrets in a request.

### 5.3 Define the response contract

A successful prediction includes:

- The supplied item and location identifiers.
- The number of days in the requested horizon.
- The average daily forecast, in units per day, rounded to three decimal places.
- The horizon forecast, in units, rounded to three decimal places.
- A fixed model version identifying this baseline.

Compute the horizon total from the unrounded mean, then round the result.
Rounding the daily value first can change the total.
This is a numerical reporting rule, not a promise about binary floating-point exactness.
Do not interpret a fractional forecast unit as a physically divisible shipment.
The downstream planning policy owns rounding to order quantities or case packs.

### 5.4 Separate the model from transport

The model class receives a validated list and a horizon.
It computes `sum(daily_demand) / len(daily_demand)`.
It returns the two numeric quantities and its version.
Keeping the calculation separate lets a test call it directly if desired.
The API route handles request validation, model availability, and response serialization.
A later model could be loaded from a signed artifact, but that is not implemented in this lab.

The direct model is deterministic for a given ordered input in the stated runtime.
The mean itself does not depend on order mathematically, though floating-point summation can vary slightly with order.
No random seed is needed for this calculation.
The same forecast does not imply the same input was appropriate or current.

### 5.5 Design health checks

`GET /health/live` returns `{"status":"alive"}` when the web application can answer.
It does not claim that the model is ready or that a forecast would be accurate.
`GET /health/ready` returns `{"status":"ready","model_version":"mean-v1"}` only when the model object is present.
If it is absent, the route returns HTTP 503.
The prediction endpoint also returns HTTP 503 if the model is absent.
This keeps the inference outcome consistent with readiness.

The local model is installed during application lifespan startup.
If startup fails, the server should not accept traffic normally.
In a real service, model artifact loading, compatibility checks, and dependency policy must be specified explicitly.
Do not make a liveness probe depend indiscriminately on an external database: an outage could trigger repeated restarts without fixing the database.
Do not use readiness as a statistical model-quality monitor: an available model can be wrong.

### 5.6 Decide what to log

The middleware generates a random request ID and records method, path, status, and elapsed milliseconds.
It returns the same ID in `X-Request-ID` for ordinary completed responses.
No item ID, location ID, demand list, or response body is written by this application logger.
Use stdout so the local container runtime can collect the event stream.
The sample disables Uvicorn's access log to avoid a second, differently formatted request log.

The elapsed time uses a monotonic performance clock.
It measures this application's handling interval, not end-to-end client network latency.
A single duration cannot establish a percentile or service-level objective.
Unexpected exceptions are logged with a 500 status and then re-raised for the server's error handling.
The server may separately emit diagnostic traces; operational log configuration and retention still require review.
Do not put secrets into URLs, exception messages, or test inputs.

### 5.7 Pin dependencies and bound reproducibility claims

The lab pins FastAPI, Uvicorn, HTTPX, and pytest directly.
The first two run the service; HTTPX and pytest support in-process tests.
The exact versions in the lab are an example set, not a claim that they are currently vulnerability-free.
Review advisories and use an approved update process.

Direct pins do not freeze all transitive dependencies, package artifacts, or operating-system packages.
For stronger reproducibility, resolve a complete dependency lock with artifact hashes and retain the Python version and platform.
For a release image, pin the base image by an approved digest rather than relying only on the example tag.
Record the built image digest, source revision, model version, build inputs, and test evidence.
A digest identifies bytes; it does not independently establish trust in their producer.

### 5.8 Build the service locally

Create the files exactly as shown in section 7 under an empty sandbox directory.
Create a Python 3.11 virtual environment and activate it in your shell.
Install the pinned requirements.
Start Uvicorn bound to `127.0.0.1` on port 8000.
Use the provided `curl` commands from a second terminal.
Stop the server with Ctrl+C after observing the responses.

The local server is intentionally single-process for clarity.
Multiple workers create separate model objects and emit independent request events.
The demonstration does not establish thread safety or capacity for another model implementation.
Generated OpenAPI documentation is available locally at `/docs`; disable or restrict it according to your real deployment policy.

### 5.9 Exercise the tests

The test client enters the application lifespan so startup installs the model.
The tests cover a valid fixture, malformed inputs, the two health endpoints, and an unavailable model.
They also inspect one structured event for a request ID without requiring a real network socket.
The tests do not start Docker and do not establish production throughput or resilience.
The container verification commands separately exercise a real local HTTP connection.

An invalid request must not be corrected silently.
For example, `"horizon_days":"3"` is not equivalent to the integer `3` under this contract.
If a client needs conversion, it must make that conversion deliberately before sending.
If an item contains a space, the request fails until caller and service agree on an identifier convention.
The model's output is not stored; callers must manage their own decision records if required.

### 5.10 Build and run a container

The Dockerfile copies only dependency and application files into a slim Python image.
A `.dockerignore` reduces accidental inclusion of local environments, test caches, and version-control metadata.
The container runs as a non-root user and listens on port 8000 inside the container.
The host mapping binds only `127.0.0.1:8000`, not all host interfaces.
The health check calls the readiness endpoint using Python's standard library.
The health check indicates state to Docker; it does not itself guarantee restart or traffic routing.

The example image tag is local and mutable.
A release system should retain the immutable image digest after building and use a trusted registry policy.
The base tag can resolve to different image bytes over time.
A container is an OS-level packaging and isolation mechanism, not a substitute for host hardening.
Never mount a Docker socket or cloud credentials into this lab container.

### 5.11 Verify behavior and handle a failure

First request liveness and readiness.
Then submit the fixture and compare its daily and horizon outputs to section 6.
Submit a negative-demand fixture and confirm HTTP 422.
Inspect the container logs for structured request events without input values.
Inspect the Docker health status after its initial checks have run.
Remove the container after the exercise.

A failure may be a port conflict, an unavailable daemon, a failed package download, or invalid JSON.
Read the error before changing the application contract.
If port 8000 is occupied, select a different host port and change the client URL accordingly.
If a health check fails, distinguish application startup, model readiness, and network publishing before restarting anything.
Do not interpret an unauthenticated loopback smoke test as evidence of an internet-ready service.

### 5.12 Define a release gate beyond this sandbox

A production candidate needs a dataset and feature contract, including missingness and freshness rules.
It needs model performance evidence against a baseline and a business acceptance threshold.
It also needs authenticated clients, encrypted transport, authorized item and location access, and a threat model.
Operational requirements include resource limits, timeout and retry policy, observability, rollback, incident ownership, and dependency updates.
Where a service has an external model registry, readiness and startup must handle registry and artifact failures deliberately.
These requirements are not implemented by the exercise and must not be inferred from a passing test suite.

## 6. Worked example

A fictional planning application asks for an eight-day forecast for item `SKU-17` at location `DC-04`.
The caller supplies seven consecutive daily sales quantities in units:

| Historical day | Observed demand (units) |
|---:|---:|
| 1 | 18 |
| 2 | 21 |
| 3 | 16 |
| 4 | 25 |
| 5 | 19 |
| 6 | 23 |
| 7 | 20 |

The observations sum to 142 units.
The number of observations is 7 days.
The unrounded mean is 142 / 7 = 20.2857142857 units per day.
The requested horizon is 8 days.
The unrounded horizon forecast is `(142 / 7) * 8 = 162.2857142857` units.
The service reports `20.286` units per day and `162.286` units over the horizon.
The model version is `mean-v1`.

A downstream policy might round a planned order to full cases, subtract on-hand inventory, and account for lead time.
None of those adjustments occurs here.
A forecast of 162.286 units is neither a plan to order 162 units nor a commitment to deliver them.
It is also not a confidence interval.
A demand spike, promotion, stockout, or seasonal pattern can make this simple baseline inappropriate.

Now compare a different input with three observations: 10, 12, and 14 units.
Their mean is 12 units per day.
For a horizon of 3 days the forecast is 36 units.
If the client instead sends an empty list, there is no defined mean and the expected response is HTTP 422.
If it sends `-1` as an observation, the value violates this lab's nonnegative-demand contract and is rejected.
If the service has no model object, readiness and prediction return HTTP 503 instead of a fabricated zero forecast.

There are two distinct kinds of evidence in this example.
The arithmetic above is a hand-derived oracle for a deterministic function.
An HTTP test can verify that the service returns the same reported values for the same input.
Neither establishes that the future eight days will actually have 162.286 units of demand.
That claim would require an appropriate backtest and subsequent operational monitoring.

## 7. Code, query, or calculation

The following files make a self-contained project.
Create a fresh directory, for example `model-service-lab`, and copy each block to the named file.
The input fixture is embedded in the tests and in the `curl` request, so no external dataset is required.
Use only synthetic values.

### 7.1 `requirements.txt`

```text
fastapi==0.115.6
uvicorn==0.34.0
httpx==0.28.1
pytest==8.3.4
```

These direct pins are intentionally explicit but are not a fully hashed lockfile.
Transitive dependencies can change unless separately resolved and locked.
The test dependencies remain in this simple image for transparency; a hardened release should separate build and runtime dependencies.

### 7.2 `app.py`

```python
import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator


logger = logging.getLogger("model_service")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

Identifier = Annotated[str, Field(min_length=1, max_length=40,
                                  pattern=r"^[A-Za-z0-9_-]+$")]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: Identifier
    location_id: Identifier
    daily_demand: list[float] = Field(min_length=1, max_length=7)
    horizon_days: int = Field(strict=True, ge=1, le=28)

    @field_validator("daily_demand", mode="before")
    @classmethod
    def validate_demand(cls, values):
        if not isinstance(values, list):
            raise ValueError("daily_demand must be an array")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("demand values must be JSON numbers")
            if not 0 <= value <= 1_000_000 or not math.isfinite(value):
                raise ValueError("demand values must be finite and in range")
        return values


class PredictionResponse(BaseModel):
    item_id: str
    location_id: str
    horizon_days: int
    daily_forecast_units: float
    horizon_forecast_units: float
    model_version: str


class MeanModel:
    version = "mean-v1"

    def predict(self, values: list[float], horizon: int) -> tuple[float, float]:
        mean = sum(values) / len(values)
        return round(mean, 3), round(mean * horizon, 3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = MeanModel()
    yield
    app.state.model = None


app = FastAPI(title="Synthetic demand baseline", lifespan=lifespan)


@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        event = {
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
        }
        logger.info(json.dumps(event, separators=(",", ":")))


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    return {"status": "ready", "model_version": model.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    daily, total = model.predict(payload.daily_demand, payload.horizon_days)
    return PredictionResponse(
        item_id=payload.item_id,
        location_id=payload.location_id,
        horizon_days=payload.horizon_days,
        daily_forecast_units=daily,
        horizon_forecast_units=total,
        model_version=model.version,
    )
```

The input model applies range checks before calling the calculation.
The `finally` block records an application request event even when an exception propagates.
A server-generated 500 response after an uncaught exception may not include the request-ID header.
The supplied logger does not itself suppress server exception logs.
No body is intentionally logged, but do not send sensitive data to this sandbox.

### 7.3 `test_app.py`

```python
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import MeanModel, app, logger


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample():
    return {
        "item_id": "SKU-17",
        "location_id": "DC-04",
        "daily_demand": [18, 21, 16, 25, 19, 23, 20],
        "horizon_days": 8,
    }


def test_prediction(client, sample):
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json() == {
        "item_id": "SKU-17",
        "location_id": "DC-04",
        "horizon_days": 8,
        "daily_forecast_units": 20.286,
        "horizon_forecast_units": 162.286,
        "model_version": "mean-v1",
    }
    assert len(response.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    "field,value",
    [
        ("daily_demand", []),
        ("daily_demand", [-1]),
        ("daily_demand", [True]),
        ("daily_demand", ["20"]),
        ("daily_demand", [1000001]),
        ("daily_demand", [1] * 8),
        ("horizon_days", 0),
        ("horizon_days", 29),
        ("horizon_days", "8"),
        ("item_id", "bad id"),
    ],
)
def test_invalid_values(client, sample, field, value):
    sample[field] = value
    assert client.post("/predict", json=sample).status_code == 422


def test_unknown_field(client, sample):
    sample["horizn_days"] = 8
    assert client.post("/predict", json=sample).status_code == 422


def test_health_and_missing_model(client, sample):
    assert client.get("/health/live").json() == {"status": "alive"}
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": "mean-v1"}
    app.state.model = None
    try:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json=sample).status_code == 503
    finally:
        app.state.model = MeanModel()


def test_log_has_metadata_not_payload(client, sample):
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(json.loads(record.getMessage()))

    handler = Capture()
    logger.addHandler(handler)
    try:
        response = client.post("/predict", json=sample)
    finally:
        logger.removeHandler(handler)
    event = records[-1]
    assert event["event"] == "http_request"
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["status"] == 200
    assert event["path"] == "/predict"
    assert "daily_demand" not in json.dumps(event)
    assert sample["item_id"] not in json.dumps(event)
```

The parameterized cases contribute ten individual tests; together with the four other test functions the suite has 14 cases.
A test client sends requests in process and enters startup and shutdown through its context manager.
The health test restores the model after simulating absence so that the test cannot contaminate subsequent checks.
The logging test installs a temporary handler and removes it even if the request fails.

### 7.4 `.dockerignore`

```dockerignore
.venv
__pycache__
.pytest_cache
.git
*.pyc
.env
.env.*
```

The ignore file reduces accidental context inclusion; it does not discover every possible secret.
Keep the sandbox directory free of sensitive files even if they appear to be ignored.

### 7.5 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser
COPY app.py ./
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).read()" || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
```

The container listens on all interfaces inside its own network namespace.
The host publish command below restricts the host-facing binding to loopback.
The image tag in `FROM` is deliberately readable but mutable; pin a trusted digest for a controlled release.
The health command exits nonzero if readiness cannot be fetched successfully.
Docker's health state and container process state are separate.

### 7.6 Install and test locally

From the project directory in a POSIX-compatible shell:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Expected result: all 14 cases pass; elapsed time and pytest formatting depend on the environment.
If Python 3.11 is not named `python3.11` on the machine, use an approved Python 3.11 interpreter path.
A package download failure is an environment or registry issue, not a passing test.
Run commands from the directory containing the five files above.

Start the local service:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal, request health and a prediction:

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"item_id":"SKU-17","location_id":"DC-04","daily_demand":[18,21,16,25,19,23,20],"horizon_days":8}'
```

The expected JSON values are:

```json
{"status":"alive"}
```

```json
{"status":"ready","model_version":"mean-v1"}
```

```json
{"item_id":"SKU-17","location_id":"DC-04","horizon_days":8,"daily_forecast_units":20.286,"horizon_forecast_units":162.286,"model_version":"mean-v1"}
```

Whitespace and key ordering are not part of the contract.
The server console should include one JSON request event per call with a distinct request ID and a nonnegative duration.
The exact IDs and timings are nondeterministic.
Stop the local server before starting the container on the same host port.

Check a deliberately invalid value:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"item_id":"SKU-17","location_id":"DC-04","daily_demand":[-1],"horizon_days":8}'
```

Expected status: `422`.
A 422 is a contract rejection, not a forecast of zero.

### 7.7 Build and verify the container

Run these commands only when Docker is installed, approved, and running:

```bash
docker build -t model-service-lab:local .
docker run -d --name model-service-lab -p 127.0.0.1:8000:8000 model-service-lab:local
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"item_id":"SKU-17","location_id":"DC-04","daily_demand":[18,21,16,25,19,23,20],"horizon_days":8}'
docker inspect --format '{{.State.Health.Status}}' model-service-lab
docker logs model-service-lab
docker rm -f model-service-lab
```

The response values should match the local example after startup.
The health state may initially be `starting`; after successful periodic probes it should be `healthy`.
The health probes themselves generate request log events.
The exact image build output, timing, and generated IDs vary.
The final command removes this named lab container, not the image or other containers.
If a previous lab container has the same name, inspect it before deciding whether to remove it.

## 8. Data, standards, and regulatory context

### 8.1 Input semantics

A number in a JSON array is not automatically a valid demand observation.
The producer and consumer must agree whether it means sales, orders, shipments, consumption, or unconstrained demand.
A stockout can censor sales, so sales may understate demand.
The example assumes observations are comparable, complete daily demand values in units.
It cannot check those assumptions because dates, stock availability, and provenance are not in the request.

The array is ordered from oldest to newest by convention, although the mean ignores the order mathematically.
If a day is missing, the caller must resolve that issue upstream rather than silently insert zero.
A missing day and a known day with zero demand carry different information.
A real interface should define timezone, daily cutoff, item unit, location scope, and data freshness.
When demand is measured in cases rather than units, a conversion factor and versioned unit definition are necessary.

The identifiers here are arbitrary sandbox strings, not GS1 Global Trade Item Numbers or location numbers.
GS1 identifiers can provide standardized identification where an organization adopts them.
Merely using a string called `item_id` does not make it a GS1 identifier.
The lab's character check is not a GS1 check-digit validator.
It also does not establish item ownership, existence, or entitlement.

### 8.2 HTTP and machine-readable contracts

HTTP status codes communicate the outcome of a request, not the quality of its business input.
This lab follows FastAPI's 422 response for request-model validation.
HTTP 503 indicates an unavailable model instance, while 200 indicates a completed prediction.
A client should handle these classes differently and must not treat a 422 as a successful forecast.
Retrying an invalid request unchanged does not correct its input.
An operational retry policy for 503 needs explicit limits, backoff, and ownership; none is implemented here.

FastAPI generates an OpenAPI description from the annotated routes and models.
The description supports discoverability and client generation but does not replace a semantic data contract.
Version the interface when making incompatible changes and communicate the migration to clients.
Do not place credentials or access tokens in the generated example payloads.

### 8.3 Security and privacy context

The National Institute of Standards and Technology (NIST) Secure Software Development Framework describes practices for reducing software vulnerabilities.
Its recommendations are not automatic legal obligations for every reader.
An organization may adopt them as policy or be subject to a contract or jurisdiction-specific rule.
The framework does not certify this sample image.

Even identifiers can be confidential when they reveal product, location, customer, or shipment information.
The log omits request bodies, but URL paths and server exceptions still need a data-classification review.
Real applications should define retention, access controls, deletion, and incident response for logs.
The sample creates request IDs locally and does not accept caller-controlled IDs.
It provides no cryptographic user identity and no authorization.

The NIST AI Risk Management Framework provides a voluntary structure for governing AI risks.
A deterministic mean can still cause operational harm if its assumptions or use are inappropriate.
Model governance should connect the intended use, evaluation, owner, change process, and monitoring.
The framework is not proof of compliance or model fitness.
A jurisdiction's privacy, consumer, employment, or sector-specific rules depend on actual data and use; obtain qualified review where applicable.

### 8.4 Supply-chain traceability and artifacts

A traceable service release should link a source revision, dependency lock, image digest, model artifact, and configuration.
If model inputs come from a dataset, record the dataset version and data contract as well.
Software bill of materials and artifact provenance are useful inventory and evidence tools.
They do not guarantee that every component is safe or that an artifact is authorized.
A scan result is time-dependent because new vulnerabilities are disclosed after a build.

This lab uses a mutable base tag and direct pins for ease of reproduction.
Its image is therefore not a complete immutable supply-chain record.
An approved deployment process should resolve and verify image digests, dependency hashes, and registry provenance.
Do not infer compliance from running a container as a non-root user.

## 9. Industry practice and case evidence

### 9.1 Published patterns, not claimed lab outcomes

Google's Site Reliability Engineering book distinguishes monitoring symptoms from causes and emphasizes actionable alerting.
That public source supports the idea that a health signal should answer a specific operational question.
It does not report results for this lab.
Here liveness and readiness are deliberately narrow signals; neither measures forecast accuracy.
A production team must decide what action each alert triggers and who owns it.

Kubernetes documentation separately defines liveness, readiness, and startup probes.
A failed readiness probe can remove a pod from service endpoints, while liveness is used for restart decisions.
That documented platform behavior is not exercised merely by running Docker locally.
Docker's `HEALTHCHECK` records container health but does not itself implement Kubernetes routing or restart behavior.
The example deliberately avoids claiming that a healthy Docker status proves end-to-end service availability.

Netflix's published work on the Metaflow framework describes tooling for data-science workflows and reproducible execution.
It illustrates that organizations can invest in workflow and deployment infrastructure around models, not only modeling code.
It is not evidence that Netflix uses this example's mean forecast or achieved a particular inventory improvement.
[INFERENCE] The separation of model identity from service identity helps a team explain changes when either layer is updated.
The exact release controls and benefits must be established in the organization's own environment.

### 9.2 Model serving choices

A batch forecast may be cheaper and easier to audit when the decision cadence is daily and inputs do not change intraday.
An online API can be appropriate when a caller needs a prediction on demand with a bounded response time.
Online serving introduces availability, latency, authentication, and capacity responsibilities.
The lab teaches the API pattern, not a conclusion that online inference is always superior.
A hybrid design can publish approved batch forecasts behind a read API without running a model on every request.

Teams frequently separate a model artifact from the server container so they can release them independently.
That separation requires compatibility checks and an artifact loading policy.
Other teams bundle the model with the image to simplify deployment and rollback.
This sample bundles the baseline as source code and uses `mean-v1` as its identity.
For a learned model, its parameters, preprocessing, library version, and feature ordering also require version control.

### 9.3 What operational evidence would be useful

A release record might contain the source revision, image digest, resolved dependency list, test report, and approval.
A model record might contain training data scope, backtest periods, metrics, limitations, and model owner.
A service record might contain deployment configuration, health policy, dashboards, runbooks, and rollback instructions.
These records answer different questions and should reference each other.
None is generated automatically by this educational example.

For forecast quality, measure error on decisions and time periods that match intended use.
Compare with a baseline and analyze performance by item and location where sample sizes support it.
For service reliability, measure successful requests and latency across meaningful traffic windows.
A healthy endpoint with poor forecasts can still cause operational harm.
A good backtest with a failing service can still block planners.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong unit | Values differ by a case-pack factor | Cases interpreted as eaches | Materially wrong forecast | Define units and conversion version in the contract |
| Censored demand | Recorded sales fall during stockouts | Unavailable stock hides demand | Baseline underestimates demand | Track availability and evaluate uncensoring |
| Missing dates | Fewer observations without explanation | Caller silently drops missing days | Biased mean | Validate completeness upstream and expose time coverage |
| Stale data | Forecast inputs lag decision time | Pipeline stopped or cached old values | Obsolete plan | Set freshness limits and alert on age |
| Invalid input | Rising 422 rate | Client contract changed or data degraded | Requests rejected | Monitor by client and fix source without logging bodies |
| Model missing | Readiness returns 503 | Startup or loading failed | No forecasts served | Fail closed, inspect model load and restore approved artifact |
| Alive but unusable | Liveness passes and readiness fails | Process answers while model unavailable | Traffic sent to a broken instance if probes are confused | Route on readiness and investigate separately |
| Overbroad liveness | Restarts coincide with external outage | Probe depends on a downstream service | Restart storm | Keep liveness focused on recoverable process failures |
| Mutable image tag | Same tag yields different image digests | Registry tag moved | Unexplained release change | Deploy approved digest and record provenance |
| Incomplete dependency pin | Rebuild resolves different transitives | Only direct versions were pinned | Non-identical runtime or regression | Lock all dependencies with hashes and test rebuilds |
| Sensitive logs | Demand or identifiers appear in logs | Body or exception logging enabled | Confidentiality incident | Minimize fields, review server logs, restrict access |
| Open host binding | Unexpected external connections | Port published on all interfaces | Unauthorized predictions or data exposure | Bind loopback for sandbox; use network and identity controls for deployment |
| Floating-point confusion | Total differs from daily value times days at displayed precision | Intermediate rounding changes arithmetic | Reconciliation dispute | Specify rounding order and tolerances |
| Model/version mismatch | Identical version label produces different behavior | Code changed without version discipline | Untraceable forecast change | Bind version to immutable artifact and release evidence |
| Probe floods logs | Health calls dominate event volume | Each probe is logged | Cost and poor signal | Tune collection and retention without hiding failures |
| Unbounded retries | Request volume spikes during outage | Clients retry 503 without limits | Cascading overload | Agree on bounded backoff and capacity policies |
| Wrong readiness semantics | 200 returned with missing model | Probe checks process only | Requests fail after routing | Test model-present and model-absent states |
| Public example mistaken for approval | Sandbox container reused in production | No release gate | Security and reliability gaps | Document limitations and require formal deployment review |

A status code is a symptom that needs context.
For example, an increase in 422 responses could indicate a caller defect or a valid contract change not yet coordinated.
It is not proof that the model is inaccurate.
A 503 in this implementation means no model object is present; other implementations may use 503 for additional reasons.
Always interpret signals against the actual contract in the deployed version.

The table's mitigations are controls to design and evaluate, not features already implemented by this lab.
Production alert thresholds require observed traffic and a response owner.
Never publish arbitrary numerical targets as measured service-level performance.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or challenge it |
|---|---|---|---|
| Inputs represent daily demand | Mean has a consistent interpretation | Sales are censored or represent shipments | Compare source definitions and availability records |
| Days are consecutive | A seven-value window represents recent history | Missing calendar days are omitted | Reconcile upstream dates to a calendar |
| Values use a common unit | Summation is meaningful | Cases and individual units are mixed | Validate product unit metadata and conversion |
| Baseline is useful | Simple mean provides a benchmark | Strong seasonality or promotions dominate | Backtest by horizon and compare alternative baselines |
| Horizon is 1–28 days | Contract bounds the operation | Business needs hourly or longer-range output | Review use cases and introduce an explicit new contract |
| Caller owns identifier validity | Service has no item master | Unknown or unauthorized ID is sent | Integrate an authorized catalog in a separately designed service |
| Model can be held in memory | Readiness is a presence check | Large or remote artifact cannot load | Test realistic loading, resources, and failure states |
| Logs contain no sensitive body | Middleware records metadata only | Server, proxy, or exception logging includes input | Inspect the complete logging pipeline with synthetic probes |
| Local binding is sufficient for the lab | Only the local learner needs access | Host mapping or firewall is changed | Inspect host networking and access from an approved client |
| Direct dependency pins are enough to learn | Keeps instructions short | Transitively resolved packages change | Resolve a full lock and compare installed artifacts |
| A single worker is adequate | Sample has negligible compute | Real workload has concurrent requests | Load-test an approved deployment and set capacity controls |
| Test arithmetic is a suitable oracle | Formula is independently computed | Contract changes rounding or formula | Recalculate with explicit precision and update versioned tests |
| Request validation limits are appropriate | Bounds reduce accidental extreme values | Legitimate demand exceeds sandbox limit | Analyze real approved data before changing limits |
| Model object presence means locally ready | Baseline has no external dependencies | Future model requires files or a remote service | Define and test a model-specific readiness policy |

These assumptions are part of the exercise, not universal truths.
The model does not learn, backtest, estimate uncertainty, or adjust for causal drivers.
It does not validate the truth, timeliness, or authorization of supplied observations.
The deployment does not include TLS, credentials, rate limits, resource quotas, or a persistent audit trail.
Do not connect real operational systems to it without a separate engineering and governance process.

## 12. Exercises

### Exercise 1 — Trace the decision boundary

**Time:** 10 minutes.

**Deliverable:** Draw the path from client input through validation and model calculation to response and downstream planning.
Mark the forecast, plan, target, and commitment as distinct concepts.

**Acceptance:** Your diagram includes both a 422 branch and a 503 branch.
It does not claim the endpoint places an order.
It names the party responsible for the downstream decision.

### Exercise 2 — Contract review

**Time:** 15 minutes.

**Deliverable:** Classify these requests as accepted or rejected and give the reason for each: `daily_demand` of `[0, 2]`, `[]`, `[true]`, and `["2"]`; horizons `1`, `28`, and `29`.
Assume all other fields are valid.

**Acceptance:** Identify each boundary and distinguish a JSON number from a boolean or string.
State the expected HTTP code for rejected input.
Explain why nonnegative values do not prove data completeness.

### Exercise 3 — Compute the result

**Time:** 10 minutes.

**Deliverable:** Calculate the daily and five-day horizon forecasts for observations `[11, 14, 19]` units.
Show the unrounded mean, multiplication, and three-decimal reported values.

**Acceptance:** Use the unrounded mean for the total.
Give units for both outputs.
Explain why neither number is an authorized order.

### Exercise 4 — Run the local artifact

**Time:** 35 minutes.

**Deliverable:** Create the section 7 files in an empty sandbox, install dependencies, run the tests, start the service, and submit the fixture.
Capture the test summary, response JSON, and one redacted structured request event.

**Acceptance:** The fixture matches the independent calculation in section 6.
The event has method, path, status, request ID, and duration but not the demand list.
A failure is reported with its actual output rather than replaced by an invented passing result.

### Exercise 5 — Probe and failure design

**Time:** 15 minutes.

**Deliverable:** Write a table for live/ready/predict behavior with a present model and with a missing model.
Describe one external dependency that should not automatically be a liveness dependency.

**Acceptance:** Include expected 200 and 503 outcomes for both states.
Explain why a process restart may not repair an external dependency.
Do not claim readiness proves model accuracy.

### Exercise 6 — Test the rejection boundary

**Time:** 20 minutes.

**Deliverable:** Add two independent pytest cases: one for a missing required identifier and one for a fractional horizon.
Run them in your sandbox and capture the actual test outcome.

**Acceptance:** Both requests are rejected with 422.
Tests do not depend on a real network connection or external service.
They use synthetic data and do not weaken the existing assertions.

### Exercise 7 — Container verification

**Time:** 25 minutes.

**Deliverable:** Build and run the image using section 7, call readiness and prediction, inspect health and logs, then remove the named container.
Record the actual image identifier and the observed response.

**Acceptance:** The host port is bound to loopback.
The prediction matches the local fixture and the health status is interpreted only after probes have had time to run.
Explain the difference between an image tag and an immutable digest.
If Docker is unavailable, state the missing prerequisite rather than inventing the result.

### Exercise 8 — Release critique

**Time:** 25 minutes.

**Deliverable:** Write a one-page release decision for a request to expose this API to a supplier.
List at least five missing controls and identify the accountable owner for each.

**Acceptance:** Include authentication and authorization, transport encryption, model evaluation, dependency provenance, and operational response.
Distinguish service availability from forecast quality.
Reject or condition approval based on missing evidence, not on a fabricated performance statistic.

## 13. Self-check answers

### Answer 1

The client sends a JSON request to `POST /predict`.
The framework validates its structure and domain constraints.
An invalid request produces 422 before the forecast is computed.
A valid request reaches the route, which checks the model.
If the model is missing, it returns 503.
If present, the model computes the mean and the route returns the forecast.
A planning application or authorized person decides what action to take.
A target is a desired business outcome; a commitment requires an authorized decision.
Neither is created by this API.

### Answer 2

`[0, 2]` is accepted by the demand constraints because both values are numeric and nonnegative.
An empty list is rejected because at least one observation is required.
`[true]` is rejected because booleans are explicitly excluded.
`["2"]` is rejected because a numeric string is not a JSON number.
Horizons 1 and 28 days are accepted; 29 days is rejected.
Invalid cases return 422 when all other inputs are valid.
A valid zero might mean no demand, but it might also hide missing or censored data.
The service cannot distinguish these cases with the provided fields.

### Answer 3

The sum is `11 + 14 + 19 = 44` units.
The mean is `44 / 3 = 14.666666...` units per day.
The reported daily value is `14.667` units per day.
The five-day total is `(44 / 3) * 5 = 73.333333...` units.
The reported total is `73.333` units.
Multiplying the displayed daily value by five would give `73.335`, which differs because of intermediate rounding.
An inventory decision still needs on-hand stock, lead time, policy, and authorization.

### Answer 4

A satisfactory run has a real pytest summary and the values in section 6.
Each event should show an `http_request` event with a generated request ID.
The request payload must not appear in the event emitted by the application logger.
Exact duration, request ID, and test duration vary.
If installation or execution fails, diagnose and record the failure; no expected result is evidence of execution by itself.

### Answer 5

| Model state | Liveness | Readiness | Prediction with valid input |
|---|---:|---:|---:|
| Present | 200 | 200 | 200 |
| Missing | 200 | 503 | 503 |

An external registry outage is not necessarily repaired by restarting a healthy process.
Whether a registry affects readiness depends on whether the loaded model can safely serve without it.
The example has no registry, so it cannot demonstrate this dependency decision.

### Answer 6

A missing `item_id` fails the required-field check.
A horizon of `2.5` fails strict integer validation.
Each should receive 422 when the rest of the fixture is valid.
A strong test changes one input at a time so its failure has an interpretable cause.
The test client should be used as a context manager to run application startup.

### Answer 7

A strong result includes a locally built image, loopback mapping, matching response, and observed health state.
A Docker image tag is a mutable name; a digest identifies particular content.
The Docker health check periodically asks the readiness endpoint, so immediate inspection may show `starting`.
Docker health alone does not demonstrate an orchestrator has removed traffic or restarted anything.
Container logs include probe requests as well as explicit client requests.
Do not claim the image was tested if the daemon or registry was unavailable.

### Answer 8

A strong answer withholds external release until identity, authorization, and encrypted transport are designed and tested.
It also assigns owners for model quality evaluation, release provenance, operational alerts, rollback, and incident response.
A sound plan defines who can request which item and location, not just who can reach the API.
It distinguishes a documented requirement from a local policy decision.
A weak answer points to a successful liveness probe as proof that the forecast is accurate or the endpoint is secure.

## 14. Interview questions

1. **What is the contract of a model endpoint?**
   A strong answer covers inputs, units, output semantics, errors, versioning, and compatibility.
   A weak answer says only that it accepts JSON.
2. **How do liveness and readiness differ?**
   A strong answer explains process recovery versus traffic eligibility and their distinct failure actions.
   A weak answer treats them as interchangeable HTTP 200 checks.
3. **Why reject a numeric string?**
   A strong answer explains explicit schema behavior, client coordination, and avoiding silent coercion.
   A weak answer claims strings cannot represent numbers at all.
4. **Why is a mean baseline useful and dangerous?**
   A strong answer identifies its interpretability and susceptibility to missing data, promotions, and seasonality.
   A weak answer claims it is production-ready because it is deterministic.
5. **How do you avoid a logging leak?**
   A strong answer minimizes fields and reviews application, proxy, server, retention, and access controls.
   A weak answer assumes structured JSON automatically makes logs safe.
6. **What does a dependency pin prove?**
   A strong answer distinguishes direct pins, transitive resolution, artifact hashes, and platform dependencies.
   A weak answer says one exact version makes the whole environment immutable.
7. **Why use an image digest?**
   A strong answer explains immutable content identity and the separate need to verify provenance.
   A weak answer assumes a digest guarantees the image is safe.
8. **How would you verify the arithmetic?**
   A strong answer derives an independent expected value and checks units and rounding order.
   A weak answer copies the model's output into the assertion without calculation.
9. **What happens when the model is unavailable?**
   A strong answer describes readiness 503, prediction 503, monitoring, and an explicit recovery owner.
   A weak answer returns zero as if it were a forecast.
10. **When would you choose batch instead of online serving?**
    A strong answer compares decision cadence, data freshness, cost, latency, and operational burden.
    A weak answer assumes an API is always the better architecture.

## 15. Key terms

- **API:** callable interface with documented input and output; see section 4.
- **ASGI:** Python asynchronous server interface; see section 4.
- **Backtest:** historical evaluation of predictive behavior; see section 9.
- **Baseline:** simple reference forecast; see section 3.
- **Build context:** files available to a Docker build; see section 5.
- **Container:** isolated process started from an image; see section 5.
- **Dependency pin:** exact direct package version; see section 7.
- **Digest:** content-derived image identifier; see section 8.
- **Domain validation:** checks on meaning and bounds; see section 5.
- **Fixture:** fixed synthetic test input; see section 7.
- **Forecast:** estimate of future units, not an order; see section 6.
- **Health check:** observation of process or service state; see section 5.
- **Horizon:** number of forecast days; see section 4.
- **HTTP:** request-response protocol; see section 8.
- **Image:** container artifact composed of layers; see section 5.
- **Inference:** applying a model to input; see section 3.
- **JSON:** structured request and response representation; see section 7.
- **Latency:** elapsed processing duration; see section 5.
- **Liveness:** whether the process responds; see section 5.
- **Lockfile:** resolved dependencies, ideally including hashes; see section 5.
- **Model version:** identifier for model behavior; see section 7.
- **OpenAPI:** machine-readable API description; see section 8.
- **Readiness:** whether an instance can receive prediction traffic; see section 5.
- **Request ID:** correlation identifier for an event; see section 7.
- **Schema:** structural constraints on data; see section 4.
- **Structured log:** event with named fields; see section 5.
- **Test oracle:** independent expectation for a test; see section 6.
- **Unit:** measurement basis for an observation or forecast; see section 8.

## 16. Sources

The sources below are public documentation or primary institutional material.
They support interface, deployment, and governance concepts; none is evidence that this lab has been run or validated in production.

1. FastAPI, [Request Body](https://fastapi.tiangolo.com/tutorial/body/) and [Testing](https://fastapi.tiangolo.com/tutorial/testing/).
   Official framework documentation for request models and the test client.
2. Pydantic, [Validators](https://docs.pydantic.dev/latest/concepts/validators/) and [Fields](https://docs.pydantic.dev/latest/concepts/fields/).
   Official documentation for field constraints and validation.
3. Uvicorn, [Settings](https://www.uvicorn.org/settings/).
   Official server documentation for network binding and logging options.
4. Docker, [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) and [Container networking: published ports](https://docs.docker.com/engine/network/port-publishing/).
   Official descriptions of build instructions, health checks, and port publishing.
5. Kubernetes, [Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).
   Official documentation on separate probe roles; Kubernetes behavior is not exercised in this Docker-only lab.
6. Google, [Site Reliability Engineering: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/).
   Public engineering guidance on symptoms, causes, and actionable monitoring.
7. NIST, [Secure Software Development Framework, SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final), 2022.
   Institutional guidance for secure software development practices.
8. NIST, [AI Risk Management Framework, AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework), 2023.
   Voluntary risk-management framework, not a certification of this example.
9. GS1, [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin).
   Official description of GTIN identification; the sandbox identifiers are not asserted to be GTINs.
10. Netflix Technology Blog, [Open Sourcing Metaflow, a Human-Centric Framework for Data Science](https://netflixtechblog.com/open-sourcing-metaflow-a-human-centric-framework-for-data-science-1b6b28d5f65d), 2019.
    Named public case about data-science workflow tooling, not evidence of this service's performance.
11. IETF, [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110), 2022.
    Primary reference for HTTP semantics and status codes.
12. OpenAPI Initiative, [OpenAPI Specification](https://spec.openapis.org/oas/latest.html).
    Specification for describing HTTP API operations and schemas.

## 17. Links

- [Module 30 topic index](_index.md).
- [Module 30 survey](../../../course/30_simulation_optimization_production_labs.md).
- [Environment and repository layout](01_environment-repo.md): reproducible environments and fixtures.
- [Demand forecast pipeline](02_demand-forecast-pipeline.md): evaluate a forecast before serving it.
- [Inventory policy simulation](03_inventory-policy-simulation.md): distinguish a forecast from a policy decision.
- [Network MILP](04_network-milp.md): optimization as a separate decision layer.
- [VRP and dispatch](05_vrp-dispatch.md): another downstream operational decision.
- The executable API, fixtures, tests, and Docker lab are self-contained in section 7 above.
