# Environment and repository layout

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 01 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: Python basics, a terminal, basic SQL, and Modules 20, 26, and 29  
> Version 2026.5

## 1. Why this topic exists

A planner asks whether a proposed inventory policy improves service without increasing stock.
An analyst runs a simulation and sends a spreadsheet with a promising result.
A reviewer reruns the same code the next morning and gets a different ranking.
The difference may come from data, random draws, dependency versions, or a changed policy.
Until those causes are separated, the ranking is not a decision-ready result.

The first production-lab deliverable is therefore not a sophisticated optimizer.
It is a small, inspectable contract for data, code, environment, randomness, and evidence.
The contract makes an experiment repeatable enough to challenge rather than merely trust.
The business owner can then ask whether the model addresses the decision, not whether the analyst's laptop was special.

This lab creates a synthetic demand fixture, queries it with DuckDB, and runs a seeded inventory calculation.
Tests are written before the implementation so that the expected behavior is visible independently of the implementation.
The same project can run in a local Python virtual environment or an optional Docker container.
A deliberately small example makes every output auditable by hand.
It is not a forecasting benchmark, a production deployment, or evidence of actual service improvement.

A forecast estimates uncertain future demand.
A plan chooses operational actions using information available at a particular time.
A target is a desired performance level; a commitment is an authorized obligation.
The sample demand is historical test input, and the simulated service is an illustration, not any of these four business artifacts.

## 2. Learning outcomes

After working through this topic, you can:

1. Create an isolated Python environment and explain what dependency pins do and do not guarantee.
2. Organize source code, tests, synthetic fixtures, generated outputs, and configuration without mixing their responsibilities.
3. Write a failing test before implementing a DuckDB query and make the test pass for the stated behavior.
4. Compute and verify a stockout and fill-rate example from a fixed demand fixture.
5. Run the same seeded simulation twice and explain the limits of a seed as a reproducibility control.
6. Package the lab in a container and distinguish an image tag from an immutable image digest.
7. Construct a run manifest and identify inputs, environment, code, and parameters required for a meaningful comparison.
8. Critique a reproducibility claim against security, data governance, and operational acceptance criteria.

Exercises 1–8 and their self-checks map to these outcomes in order.
The lab's acceptance tests check the educational artifact, not business suitability.

## 3. The core mental model

```text
  approved inputs       versioned code          run configuration
  fixture + schema      query + model           seed + policy
         \                  |                     /
          \                 |                    /
           +---- controlled Python environment --+
                              |
                       DuckDB query
                              |
                   seeded simulation
                              |
              metrics + test results + manifest
                              |
                 review, compare, decide
```

A run is a function of more than source code.
It also depends on input bytes, explicit parameters, software versions, operating environment, and any hidden external state.
A seed controls a pseudorandom sequence only in the context of a particular generator and implementation.
A reproducible result is not necessarily correct, unbiased, authorized, or useful.
Tests establish small claims about behavior; they do not validate all model assumptions.

Separate four kinds of evidence.
The fixture says what data the test used.
The assertion says what behavior is expected.
The run manifest says what was actually run.
The review record says whether the output is fit for a particular decision.
When two scenarios differ, hold inputs and random draws constant if the intent is to isolate a policy change.
For final uncertainty assessment, use multiple independent seeds and report variation rather than selecting a favorable seed.

### The boundary of this sandbox

The example reads only local synthetic data and needs no accounts, network service, cloud project, or database server after installation.
Installing dependencies and building the image normally requires a network connection and access to trusted package registries.
Docker is optional; use a machine on which your administrator permits it and the daemon is running.
A container shares the host kernel and is not a complete security boundary for untrusted code.
Do not place credentials, customer data, or regulated records in the fixture or Docker build context.

## 4. Definitions and vocabulary

| Term | Meaning and unit where applicable | Frequent misuse |
|---|---|---|
| Fixture | Fixed synthetic input, here five order lines measured in units | Treating representative-looking rows as a representative population |
| Repository | Version-controlled files and their history | Treating an uncommitted working directory as a published release |
| Working tree | Files currently checked out, including local modifications | Assuming its contents match the last commit |
| Virtual environment | Python interpreter environment with isolated installed packages | Treating it as a complete operating-system image |
| Dependency pin | Exact declared package version | Assuming it fixes all transitive artifacts and platforms |
| Lockfile | Resolved dependency versions, ideally with hashes | Confusing an unverified requirements list with a complete lock |
| Seed | Initial state input to a pseudorandom generator | Claiming the seed fixes changing input order or generator algorithms |
| Pseudorandom generator | Deterministic state machine producing random-looking values | Treating it as cryptographically secure |
| Test oracle | Independently specified expected behavior | Copying the implementation's result into an assertion |
| Unit test | Focused test of a small behavior | Assuming a passing unit test proves business validity |
| Integration test | Test across components, such as Python and DuckDB | Assuming it covers deployment or real production data |
| DuckDB | Embedded analytical SQL database | Assuming it is a networked multi-writer service |
| SQL | Structured Query Language | Assuming rows have an order without `ORDER BY` |
| Manifest | Recorded identity and parameters for a run | Treating a note with only the seed as sufficient evidence |
| Digest | Content-derived identifier, such as SHA-256 | Treating a mutable image tag as a content digest |
| Image | Packaged container filesystem and configuration | Treating a tag as a permanent immutable reference |
| Container | Running or stopped instance of an image | Treating a container as a separate physical computer |
| Build context | Files sent to a container build | Sending secrets and local outputs unintentionally |
| Fill rate | Filled units divided by requested units, dimensionless | Confusing it with the percentage of orders filled completely |
| Stockout | Demand exceeding available stock in this simplified example | Assuming a stockout implies lost sales in every business |
| Lost sales | Demand not filled and not backordered | Applying the assumption without business approval |
| Run | One execution with identified inputs and parameters | Comparing outputs from materially different data as if only policy changed |
| Common random numbers | Reusing comparable random draws across alternatives | Assuming equal seeds ensure alignment after control-flow changes |
| Reproducibility | Ability to regenerate a result under stated conditions | Equating bitwise equality with external validity |
| Repeatability | Agreement under the same stated setup | Assuming it demonstrates portability to a different system |
| CI | Continuous integration: automated build and test on changes | Treating a green build as approval to deploy |
| Synthetic data | Artificial data with no intended real-person records | Assuming every synthetic data-generation method is private |
| Sandbox | Restricted environment for learning and experimentation | Assuming a sandbox is automatically safe for sensitive data |
| Granularity | Meaning of one record, here an order line | Mixing order-level and line-level metrics without explanation |
| UTC | Coordinated Universal Time | Using local timestamps without a timezone in a run record |
| Supply | Stock available before demand in this example, in units | Treating stock as a replenishment schedule |
| Contract | Explicit input, output, error, and acceptance conditions | Treating implementation details as a substitute for expected outcomes |

## 5. The mechanics, step by step

### 5.1 Establish the decision and boundaries

1. Name the decision: can a reviewer reproduce a small lost-sales calculation from synthetic order lines?
2. Identify the unit of analysis: one CSV record is one demand order line for one SKU on one date.
3. State the input window: the fixture contains two dates, 2025-01-06 and 2025-01-07.
4. State the output: total demand by SKU and a seeded one-period stockout calculation.
5. Separate the exercise from deployment: the CSV and inventory are synthetic; there is no customer commitment.
6. Ask the business owner which loss, backlog, replenishment, and service definitions would apply before using real data.
7. Record the decisions not modeled: lead time, supplier capacity, returns, substitutions, and intraday order priority.
8. Reject any suggestion that a passing test establishes an optimal policy or production-ready service level.

### 5.2 Check prerequisites and choose a workspace

Install Python 3.11, Git if version control is available, and optionally Docker Desktop or Docker Engine.
On Windows, use PowerShell; on macOS or Linux, use a terminal.
The commands below create a new directory named `supply-lab` under your current directory.
Choose an empty location; do not overwrite an existing project.
Check that Python resolves to the intended version:

```bash
python --version
```

On a Windows installation where `python` is not configured, use `py -3.11` instead in Python commands.
Where the commands call `.venv` directly, substitute the Windows paths shown below.
If Docker is unavailable, complete every local step; the container branch is optional.
If installation fails because a network or corporate proxy is blocked, use an approved internal package mirror or ask the administrator.
Do not disable certificate checks or paste credentials into files to work around a network problem.

### 5.3 Create a minimal repository

Use the shell to create and enter the project:

```bash
mkdir supply-lab
cd supply-lab
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ('src', 'tests', 'fixtures')]"
```

You will create these files in the numbered steps that follow:

```text
supply-lab/
  .dockerignore
  .gitignore
  Dockerfile
  requirements.txt
  fixtures/
    demand.csv
  src/
    __init__.py
    lab.py
  tests/
    test_lab.py
```

`src` contains importable implementation; `tests` contains independent behavioral checks.
`fixtures` contains only the tiny, synthetic inputs needed to reproduce those checks.
A generated output, a downloaded production extract, and a virtual environment do not belong in the committed source tree.
This sample prints its outputs instead of persisting an untracked results directory.
For a larger project, write results outside the repository or to a documented ignored directory, and retain an immutable manifest separately.

### 5.4 Pin direct dependencies and isolate Python

Create `requirements.txt` with exactly these lines:

```text
duckdb==1.1.3
pytest==8.3.4
```

Create and populate a virtual environment from the project root:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze
```

On Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip freeze
```

Use the virtual-environment interpreter for every subsequent local Python command.
The examples below write `python`; substitute `.venv/bin/python` or `.venv\Scripts\python.exe` as appropriate.
`pip freeze` is evidence of what was installed, not a hash-verified cross-platform lockfile.
These direct pins alone do not constrain all dependencies, downloaded package hashes, system libraries, or interpreter builds.
For a controlled release, resolve and review a platform-specific lock with hashes, pin the interpreter build, and retain the exact distribution artifacts from an approved source.
Do not commit `.venv`; recreate it from the environment specification.

### 5.5 Add ignore rules and synthetic input

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
.env
results/
```

Create `.dockerignore`:

```dockerignore
.venv
.git
__pycache__
.pytest_cache
.env
results
```

Ignore rules are a convenience, not secret protection.
Never create credentials in the project, and review the Docker build context before sending it to a daemon.
Create `fixtures/demand.csv`:

```csv
date,sku,units
2025-01-06,A,7
2025-01-06,B,4
2025-01-07,A,11
2025-01-07,B,6
2025-01-07,A,5
```

The fixture has five rows and no missing cells.
SKU A has three lines and SKU B has two lines.
The second date has two separate A lines; neither is a duplicate because this example does not define an order identifier.
A real contract would need identifiers and a defensible duplicate rule.
The explicit integer quantities and ISO-formatted dates make the data easy to inspect.
Their small size does not establish representativeness.

### 5.6 Write the tests before the implementation

Create the empty file `src/__init__.py` so the source directory is importable as a package.
Create `tests/test_lab.py` with the exact test block in Section 7.1.
Run from the repository root:

```bash
python -m pytest -q
```

The initial run should fail during collection because `src.lab` does not exist yet.
This is the intended red state, not evidence that the project is complete.
If it instead fails because `pytest` or `duckdb` cannot be imported, fix the interpreter or installation first.
Once the implementation exists, repeat the same command and expect all tests to pass.
The tests compare against independently calculated totals, not a snapshot captured from the code.

### 5.7 Implement the narrow contract

Create `src/lab.py` using Section 7.2.
The query reads a caller-supplied CSV path as data through a bound SQL parameter.
It sums units by SKU and orders the result by SKU explicitly.
The simulation uses its own `random.Random` instance, avoiding mutation of Python's global generator state.
The fixture demand and simulated demand are separate demonstrations: the random demand is not fitted to the fixture.
The function validates the stock and draw count before doing work.
The executable entry point prints the fixture totals and verifies repeatability for a fixed seed.

### 5.8 Observe the green state

Run the tests and the module from the project root:

```bash
python -m pytest -q
python -m src.lab
```

The four tests should pass.
The program should print totals of 23 units for A and 10 units for B, followed by `seeded runs equal: True`.
The exact simulated list is deliberately not used as a portable golden value.
Record the Python and DuckDB versions and fixture digest along with the seed if the output is used in a comparison.
If the actual output differs, inspect the failing assertion and environment before changing the expected value.

### 5.9 Optional container branch

Create `Dockerfile` with the block in Section 7.3.
From the project root, with a running and approved Docker installation:

```bash
docker build -t supply-lab:local .
docker run --rm supply-lab:local
docker run --rm supply-lab:local python -m src.lab
```

The first run executes the same tests; the second prints the same deterministic totals and equality check.
The build uses the tag `python:3.11.11-slim`, not a digest-pinned base.
The image tag is therefore a convenience for the lab, not an immutable supply-chain guarantee.
For a controlled release, resolve an approved digest for the target platform, pin it, lock and verify package artifacts, scan the resulting image, and rebuild deliberately when patches are approved.
No port is published, no Docker volume is needed, and no cloud service is used.
Use an approved environment; the example does not claim a hardened production image.

### 5.10 Preserve the run record

For a local run, capture the following values without putting secrets into the record:

```bash
python --version
python -m pip freeze
python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('fixtures/demand.csv').read_bytes()).hexdigest())"
```

If using Git, also capture `git rev-parse HEAD` and whether the working tree is clean using `git status --short`.
A commit hash is not sufficient when uncommitted changes exist.
If using Docker, record the image ID and the resolved base digest as well as the Dockerfile and platform.
Record the UTC run time, operating system and architecture, input digest, seed, policy parameters, test result, and any deviations.
A SHA-256 digest identifies the bytes you hashed; it does not prove that the input is authorized or correct.
Keep the corresponding artifacts in a controlled location with an appropriate retention policy.

### 5.11 Compare deliberately

Hold the fixture, code revision, versioned dependencies, and random stream constant when comparing two policy settings.
Change one declared policy parameter at a time when teaching causal isolation.
Real policies may consume different numbers of random draws, so the same seed alone may not align comparable events.
For more complex simulation, pre-generate a common demand stream and feed the identical stream to every policy.
Repeat across independent seeds to estimate variation; do not treat a single seeded run as an uncertainty interval.
After a comparison, ask a domain reviewer whether the modeled behavior and data window support the intended decision.
Only an authorized owner can turn a reviewed plan into a commitment.

### 5.12 A release-ready evidence checklist

The following checks are a review procedure for extending the educational lab.
They are not claims that the lab implements a production release process.
Each check specifies a question, an artifact, and an escalation condition.

1. **Decision ownership.**
   Identify who requested the analysis and who is authorized to act on it.
   Record the decision, deadline, and alternatives in a short decision note.
   Distinguish a recommendation from an approved operational commitment.
   Escalate if the owner or decision boundary is unclear.
   Do not infer approval from the existence of a successful script.

2. **Input authorization.**
   Identify the data owner and the allowed analytical purpose.
   Record the source, classification, retention period, and authorized storage.
   Use the fabricated fixture until an approved extract is available.
   Escalate if the source contains personal data or sensitive commercial information.
   Do not send an unreviewed extract to an external image builder or service.

3. **Semantic agreement.**
   Define whether demand means orders placed, sales shipped, or another quantity.
   State the unit, time bucket, SKU identity, and treatment of cancellations.
   Record a reconciliation from source totals to the model input.
   Escalate if owners disagree about the denominator of the service metric.
   A technically valid CSV can still express the wrong business event.

4. **Data quality.**
   Specify allowable nulls, ranges, identifiers, and duplicate behavior.
   Create representative negative tests for the production ingestion boundary.
   Reconcile row counts and unit totals before and after transformation.
   Escalate unexpected missingness or volume shifts instead of silently filling values.
   The toy query only casts types and is not that ingestion boundary.

5. **Experiment identity.**
   Assign a run identifier and record the input digest and code revision.
   Record all policy parameters separately from random-generator parameters.
   Preserve whether local modifications were present at execution time.
   Escalate if the output cannot be tied to the exact inputs and command.
   A filename such as `final.csv` is not a versioned experiment record.

6. **Environment identity.**
   Record Python, dependency versions, operating system, and architecture.
   If containerized, record the image identity and resolved base digest.
   Store a reviewed lock and package hashes for a controlled release.
   Escalate if dependencies came from an unapproved or unknown source.
   A version string alone does not authenticate downloaded bytes.

7. **Randomness control.**
   Record generator family, seed, and the number of simulation periods.
   Use a pre-generated demand sequence for aligned policy comparisons.
   Retain or regenerate the sequence under the declared environment.
   Escalate if scenario control flow consumes different random draws unexpectedly.
   Multiple seeds are needed to examine sensitivity to sampled demand.

8. **Metric definition.**
   Specify numerator, denominator, aggregation level, and zero-denominator rule.
   Name the time window and whether partial orders count as fulfilled.
   Reconcile aggregated results to a small manually checked example.
   Escalate if the metric differs from the business owner's established definition.
   Unit fill rate and order fill rate are not interchangeable labels.

9. **Model validation.**
   Compare simulated behaviors to historical or controlled observations when appropriate.
   Check assumptions about lead time, stock state, and lost versus backordered demand.
   Record known limitations and plausible failure conditions.
   Escalate material gaps between the model and the intended decision context.
   Identical reruns do not establish that a simulation is realistic.

10. **Security review.**
    Keep credentials out of source, fixtures, logs, and build contexts.
    Use least privilege and approved mechanisms for any necessary runtime secrets.
    Review dependency provenance and known vulnerabilities under organizational policy.
    Escalate suspicious packages, unexpected network access, or sensitive output.
    A container image is not a substitute for host and access controls.

11. **Independent review.**
    Give the reviewer the input contract, method, tests, manifest, and limitations.
    Ask them to reconstruct a small calculation without relying on the implementation.
    Record disagreements, resolutions, and remaining uncertainty.
    Escalate if an assumption changes the sign or practical meaning of the comparison.
    A reviewer should distinguish observed facts from inference.

12. **Publication and retention.**
    Label every figure or table as synthetic, simulated, estimated, or observed as appropriate.
    Preserve the run record with the approved output and its review decision.
    Apply retention and deletion rules to inputs and generated files.
    Escalate if a proposed external disclosure exceeds the approved purpose.
    Do not represent a plan or target as a contractual customer commitment.

## 6. Worked example

A distributor wants to check whether a repeatable calculation can distinguish demand from available inventory.
The fixture describes five synthetic order lines for two SKUs over two dates.
No actual customer, warehouse, price, service-level target, or commitment is represented.
For this worked example only, assume each SKU starts with 20 units and there is no replenishment.
Demand not filled in this single period is lost rather than backordered.
This is a stated modeling assumption, not a measured property of a real business.

| Date | SKU | Requested units | Running SKU demand, units |
|---|---|---:|---:|
| 2025-01-06 | A | 7 | 7 |
| 2025-01-06 | B | 4 | 4 |
| 2025-01-07 | A | 11 | 18 |
| 2025-01-07 | B | 6 | 10 |
| 2025-01-07 | A | 5 | 23 |

For SKU A, total requested demand is 7 + 11 + 5 = 23 units.
Available stock is 20 units, so filled demand is min(23, 20) = 20 units.
The lost demand is max(23 - 20, 0) = 3 units.
The unit fill rate is 20 / 23 = approximately 0.869565, or 86.96%.
For SKU B, requested demand is 4 + 6 = 10 units.
With 20 units available, filled demand is 10 units and lost demand is 0 units.
The unit fill rate is 10 / 10 = 1.0, or 100%.
Total requested demand is 23 + 10 = 33 units.
Total filled demand is 20 + 10 = 30 units.
The aggregate unit fill rate is 30 / 33 = approximately 90.91%.
This aggregate weights each unit equally; the unweighted mean of SKU rates answers a different question.
Remaining stock is 0 units for A and 10 units for B.
The calculation does not move B's unused inventory to A, because different SKUs are not interchangeable by default.

The code in Section 7 verifies both SKU totals and checks the lost-sales arithmetic as a standalone example.
Its seeded simulation separately draws integers between 0 and 10 units, inclusive, for each period.
It calculates per-period lost units against a fixed capacity; it does not deplete an inventory balance across periods.
Do not mix that simulation's capacity with the cumulative inventory calculation above.
Its role is to demonstrate repeatable random inputs and controlled comparisons.

A forecast would estimate a future demand distribution from information available before that future period.
The five observed rows are a synthetic fixture, not such a forecast.
A policy choice based on them would be a plan only after its assumptions and scope were defined.
A 95% fill rate could be a target if a business owner chose it; this example sets no target.
A customer delivery commitment would require authorization and operational feasibility outside this sandbox.

## 7. Code, query, or calculation

All file paths below are relative to the `supply-lab` project root from Section 5.
Use the pinned Python packages and create the fixture before running the tests.
The implementation is intentionally small enough for the test oracle to be calculated by hand.

### 7.1 Tests: `tests/test_lab.py`

```python
from pathlib import Path

import pytest

from src.lab import demand_totals, lost_units, simulate


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "demand.csv"


def test_fixture_totals_are_ordered_and_correct():
    assert demand_totals(FIXTURE) == [("A", 23), ("B", 10)]


def test_lost_sales_math():
    assert lost_units(demand=23, stock=20) == 3
    assert lost_units(demand=10, stock=20) == 0


def test_seeded_simulation_repeats_and_respects_bounds():
    first = simulate(seed=37, periods=12, capacity=6)
    assert first == simulate(seed=37, periods=12, capacity=6)
    assert len(first) == 12
    assert all(0 <= demand <= 10 for demand, _ in first)
    assert all(lost == max(demand - 6, 0) for demand, lost in first)


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError):
        lost_units(demand=-1, stock=20)
    with pytest.raises(ValueError):
        lost_units(demand=1, stock=-1)
    with pytest.raises(ValueError):
        simulate(seed=37, periods=-1, capacity=6)
    with pytest.raises(ValueError):
        simulate(seed=37, periods=1, capacity=-1)
```

The seed test checks equality in the same environment, bounds, and the business arithmetic.
It does not claim that all Python versions, future generator implementations, or platforms yield identical random sequences.
The negative input checks make the function's basic numeric domain explicit.
They do not implement a full production schema-validation layer.

### 7.2 Implementation: `src/lab.py`

```python
from pathlib import Path
import random

import duckdb


def demand_totals(csv_path: Path) -> list[tuple[str, int]]:
    """Return demand by SKU from the documented synthetic CSV fixture."""
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute(
            """
            SELECT sku, CAST(SUM(units) AS BIGINT) AS total_units
            FROM read_csv(?, columns={
                'date': 'DATE', 'sku': 'VARCHAR', 'units': 'INTEGER'
            }, header=true)
            GROUP BY sku
            ORDER BY sku
            """,
            [str(csv_path)],
        ).fetchall()
    finally:
        connection.close()


def lost_units(demand: int, stock: int) -> int:
    """Calculate lost units assuming unmet demand is not backordered."""
    if demand < 0 or stock < 0:
        raise ValueError("demand and stock must be nonnegative")
    return max(demand - stock, 0)


def simulate(seed: int, periods: int, capacity: int) -> list[tuple[int, int]]:
    """Draw demand and calculate loss against reset capacity each period."""
    if periods < 0 or capacity < 0:
        raise ValueError("periods and capacity must be nonnegative")
    rng = random.Random(seed)
    result = []
    for _ in range(periods):
        demand = rng.randint(0, 10)
        result.append((demand, lost_units(demand, capacity)))
    return result


def main() -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "demand.csv"
    print("fixture totals:", demand_totals(fixture))
    first = simulate(seed=37, periods=12, capacity=6)
    second = simulate(seed=37, periods=12, capacity=6)
    print("seeded runs equal:", first == second)


if __name__ == "__main__":
    main()
```

Expected program output:

```text
fixture totals: [('A', 23), ('B', 10)]
seeded runs equal: True
```

The DuckDB connection is in-memory and explicitly closed after the query.
No implicit row order is assumed: `ORDER BY sku` is part of the contract.
The path is supplied as a bound parameter instead of interpolating it into SQL text.
This is a local trusted-fixture example, not an authorization mechanism or a safe parser for arbitrary hostile files.

### 7.3 Container: `Dockerfile`

```dockerfile
FROM python:3.11.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY tests/ tests/
COPY fixtures/ fixtures/
CMD ["python", "-m", "pytest", "-q"]
```

The default container command runs four tests and exits.
A successful exit is a lab acceptance signal, not evidence of a hardened runtime.
The image installs test dependencies and runs with the base image's default user; it is intentionally not a deployment template.
For production, implement your organization's non-root, provenance, image-signing, vulnerability, secret, and lifecycle controls.

## 8. Data, standards, and regulatory context

### Data contract for the exercise

The fixture contract is deliberately narrow.
It is not a comprehensive schema or a claim about any real supply-chain data source.
The file encoding is UTF-8, the delimiter is a comma, and the first row contains the column names.
The implementation declares the date, SKU, and unit types rather than relying on automatic inference.

| Field | Expected form | Meaning in this sandbox | Production question |
|---|---|---|---|
| `date` | `YYYY-MM-DD` | Date of synthetic demand | Which business timezone and event timestamp determine the date? |
| `sku` | Nonempty text | Synthetic stock-keeping unit | Is the identifier stable across systems and effective dates? |
| `units` | Nonnegative integer | Requested pieces on one order line | Are units ordered, shipped, returned, or netted? |

A date with no timezone is sufficient for this two-day synthetic example only because the business calendar is stipulated.
It cannot represent an event's instant or resolve a daylight-saving transition.
The code casts values into DuckDB types, but it does not explicitly reject all nulls, malformed business identifiers, duplicate events, or negative CSV quantities.
The function's test fixture meets those assumptions; a production ingestion contract would need explicit checks and quarantine behavior.
An absent input file or a malformed typed value should surface as an error rather than silently produce plausible totals.
The schema also does not define whether returns are negative units or separate records.
Do not extend it to returns without changing the contract and tests.

### Standards and legal boundaries

ISO 8601 describes representations of dates and times; an ISO-style date alone does not establish a timezone or an event definition.
For real trading-partner identifiers, GS1 standards may define identifier structures and usage, but this example's A and B are not asserted to be GS1 identifiers.
The Python Packaging User Guide describes dependency and environment practices; it is a technical guide, not a legal requirement.
Docker image references and Open Container Initiative image specifications address artifact representation and distribution, not business model validity.
NIST's Secure Software Development Framework describes recommended secure development practices; whether a specific control is binding depends on the organization's applicable obligations.

Where personal data is processed, applicable law depends on jurisdiction, purpose, role, and the facts.
The EU General Data Protection Regulation (GDPR), for example, specifies principles and obligations for processing personal data within its scope; this synthetic dataset contains no intended personal data.
A date and SKU can still become identifying when joined to other information, so a column list alone does not prove anonymity.
Synthetic data can leak training records if generated from confidential data without appropriate safeguards.
Consult the organization's privacy and security teams before using actual order or event extracts.
This material is educational, not legal advice; check current authoritative sources and contract terms before relying on a rule.

### Input lineage checklist

For every data extract, identify the source system and the authorized data owner.
Record the extraction query or export process and the time window.
Record whether the timestamps represent order placement, shipment, receipt, or an update.
Retain a checksum for the exact exported bytes and the schema version.
Document any filtering, joins, deduplication, and business exclusions.
A hash helps detect byte changes; it does not validate completeness, authenticity, or semantic correctness.

## 9. Industry practice and case evidence

### What public sources support

DuckDB's documentation describes an embedded database that can query CSV files directly.
The implementation in this lab uses an in-memory connection, so no server process is provisioned.
That documented capability supports the choice of a compact lab environment; it does not establish that DuckDB is suitable for every operational workload.
The Docker documentation distinguishes images from containers and explains how image references work.
A tag may be moved, so controlled releases generally need a content digest and a deliberate update process.
These are documented tool properties, not claims about a named company's performance.

NIST's Secure Software Development Framework recommends integrating security practices throughout software development.
The lab applies only a few introductory ideas: isolate dependencies, avoid secrets in the build context, and capture software provenance.
It does not claim conformance with the full framework.
The SLSA framework describes levels and requirements for software supply-chain provenance.
The simple manifest in this lesson is a teaching record, not a SLSA attestation or a verified provenance statement.

### Named public case: SolarWinds

The 2020 SolarWinds compromise is a documented example of adversaries affecting a software supply chain.
The U.S. Cybersecurity and Infrastructure Security Agency (CISA) issued emergency guidance on the incident and the affected Orion products.
The case motivates scrutiny of build provenance, dependency trust, and access controls.
It does not imply that a pinned version, a container, or this lab's manifest would by itself prevent the attack.
[Inference] An analytics team should treat the build and dependency path as part of the trust boundary, rather than assuming that internal analysis code is exempt from supply-chain risk.
No operational or financial result for the fictional distributor is inferred from that incident.

### Named public case: reproducible research

The Reproducible Builds project publicly documents practices for making build outputs verifiable from source.
Its subject is build artifact reproducibility, which differs from simulation reproducibility and statistical validity.
[Inference] The same discipline of controlling inputs and recording toolchains can help reviewers investigate differences between analytical runs.
A reproducible executable can still contain a flawed model or use invalid demand data.
Neither a green build nor an identical result is a substitute for decision ownership.

### Applying the evidence

Choose DuckDB here because the fixture is local and small, not because a vendor claims universal superiority.
Choose a container here to teach packaging, not to imply that all users must run Docker.
Use the test-first sequence to expose a narrow expected behavior before implementing it.
When scaling the exercise, measure performance on an approved representative workload and report uncertainty and costs.
Keep documented external facts, local measurements, and analyst inference visibly separate.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong interpreter | Package missing despite installation | System Python runs instead of virtual environment | Tests cannot be reproduced | Invoke the virtual-environment interpreter explicitly |
| Mutable dependency | New install differs after weeks | A transitive package resolves differently | Changed behavior | Use reviewed lockfiles and package hashes |
| Mutable image | Same tag produces a different image ID | Registry tag moves | Different system libraries | Pin and record a platform-specific digest |
| Seed omitted | Repeated simulations differ | Generator state starts differently | Unexplained comparison | Pass and record seed explicitly |
| Seed overtrusted | Same seed but different result after code change | Random draws consumed in different order | Misleading policy comparison | Pre-generate shared demand streams |
| SQL order assumed | Lists appear in different order | Relational result has no guaranteed ordering | Flaky test or misaligned output | Explicitly order by a stable key |
| Fixture altered | Golden totals suddenly fail | Input bytes changed | Invalid comparison | Review changes and record a fixture digest |
| Wrong unit | Units unexpectedly much larger | Cases and individual pieces combined | Incorrect stockout estimate | Record and validate unit conversions |
| Hidden duplicates | Totals exceed source expectations | Replayed events treated as new lines | Inflated demand | Define event IDs and idempotency |
| Missing rows | Total falls on extract day | Partial export or late events | Optimistic service estimate | Reconcile counts and time windows |
| Null quantity | Totals seem too low | SQL aggregation ignores nulls | Understated demand | Reject or quarantine missing values |
| Negative CSV demand | Total decreases unexpectedly | Return or bad record treated as demand | Invalid model input | Define returns separately and validate ranges |
| Secrets in context | Build uploads unexpected files | Broad Docker context includes credentials | Credential exposure | Exclude sensitive files and never place secrets in context |
| False anonymity | Synthetic fixture contains copied customer details | Real records were used for convenience | Privacy incident | Use fabricated data and review provenance |
| Shared mutable database | Parallel writers encounter conflicts | Local file used as a service | Missing or failed runs | Design appropriate concurrency and storage |
| Fixture as benchmark | All policies look perfect on five lines | Small input is unrepresentative | False confidence | Evaluate on approved representative samples |
| Reused output | Old result has a recent filename | Output overwritten or manually copied | Wrong decision evidence | Store immutable run IDs and input hashes |
| Local clock confusion | Run records cannot be sequenced | Naive timestamps mix timezones | Broken audit trail | Record UTC and source event timezone |
| Production assumption | Review calls sample fill rate actual | Synthetic result lacks labels | Misleading business report | Label outputs simulated and state limits |
| Test oracle copied | Assertion always matches implementation | Expected value taken from actual output | Defect passes | Calculate an independent expected value |
| No denominator | Fill rate shows a number for zero demand | Division by zero or arbitrary default | Misinterpreted metric | Define zero-denominator behavior explicitly |
| Docker unavailable | Build cannot contact daemon | Service not installed or authorized | Container branch blocked | Complete local branch and request approved environment |
| Dependency compromise | Correct version contains malicious code | Version pinning fixes identity, not trust | Data or host compromise | Use approved sources, review provenance, scan and patch |
| Uncommitted changes | Results cannot be reproduced from commit | Working tree differs from revision | Missing implementation evidence | Record dirty state and preserve reviewed patch |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or challenge it |
|---|---|---|---|
| CSV is synthetic | Safe, portable teaching example | Copied real records | Review fixture provenance and contents |
| Row means one order line | Aggregation has a defined grain | One row already aggregates orders | Reconcile to a source specification |
| Quantities are nonnegative pieces | Lost-sales arithmetic is meaningful | Returns or case quantities mixed in | Validate range and unit metadata |
| SKU is stable | Grouping maps like with like | Reused or changed product identifiers | Review master-data effective dates |
| Dates use one agreed calendar | Two-day window has meaning | Records come from multiple timezones | Compare source timestamps and business calendar |
| Stock is available at start | Hand calculation has a clear denominator | Replenishment arrives during period | Replay timestamped inventory movements |
| Shortfall is lost | `lost_units` represents the outcome | Backorders or substitutions occur | Ask operations and reconcile orders later filled |
| SKUs are not substitutable | Per-SKU stocks remain separate | Product substitution is authorized | Review substitution rules with commercial owner |
| All demand is observed | Requested units are the denominator | Customers do not order during shortages | Compare availability with demand proxies |
| No order-priority effects | Aggregate fill is sufficient | Allocation differs by customer or channel | Replay order-level rules with timestamps |
| Each simulated period resets capacity | Independent loss calculation is interpretable | Inventory carries across periods | Add and test an explicit state transition |
| Random integers 0–10 are illustrative | Seed example is small and transparent | Real demand has seasonality or tails | Fit and backtest a justified demand model |
| Same environment repeats draws | Equality check has meaning | Interpreter or generator changes | Capture versions and test on target environment |
| Fixture schema is valid | Query is intentionally narrow | Null, malformed, or negative rows appear | Add ingestion validation and bad-row tests |
| Packages can be obtained safely | Local setup is possible | Network blocked or registry untrusted | Use an approved offline mirror with verified artifacts |
| Tests run from the project root | Import and commands resolve predictably | Different working directory or packaging | Use a proper package configuration and CI invocation |
| Image tag is adequate for illustration | Docker example remains readable | Tag changes or platform differs | Resolve and pin digest for an audited release |
| Four tests cover stated contract | Small demonstration is understandable | Other business cases or operational failures matter | Expand tests using approved acceptance criteria |

None of these assumptions is proven by running the four tests.
Document any change in assumptions before comparing results.
A passing toy fixture is not evidence of accuracy on future demand.
The sample is not a capacity plan, service forecast, production deployment, or legal compliance assessment.

## 12. Exercises

Each exercise has a deliverable, acceptance criteria, and an estimated time.
Do not use real customer records or production credentials.
Record any additional assumption rather than hiding it in code.

### Exercise 1 — Environment identity (15 minutes)

**Task:** Build the local virtual environment and identify the interpreter and installed packages.
Use the commands in Section 5.4 and record the operating system and architecture.
Explain why the version pins in `requirements.txt` are useful but incomplete.
Describe what to do if your environment is offline or the package registry is not approved.

**Deliverable:** A short environment record and a two-sentence explanation of its limitations.
**Acceptance:** The record identifies Python, DuckDB, pytest, platform, and the dependency source.
It distinguishes the direct requirements from the full installed dependency set.
It does not include credentials, tokens, or a claim of hash verification when none occurred.
**Maps to:** Learning outcome 1.

### Exercise 2 — Repository boundaries (15 minutes)

**Task:** Draw the project tree and classify each file as source, test, fixture, environment specification, or build configuration.
Choose a location for generated outputs and state whether it belongs in version control.
Explain what the ignore rules do not prevent.
Consider whether the Docker context needs `.git` or the virtual environment.

**Deliverable:** Annotated tree plus a brief rationale for three boundaries.
**Acceptance:** Source, tests, fixture, requirements, Dockerfile, and ignore rules are placed correctly.
Generated outputs and local environments are not presented as canonical source artifacts.
The explanation states that ignoring a file is not a substitute for keeping secrets out.
**Maps to:** Learning outcome 2.

### Exercise 3 — Red, green, review (25 minutes)

**Task:** Create the test file before `src/lab.py` in an empty practice workspace.
Run the test command and note the expected missing-module collection error.
Add the implementation and rerun the same tests.
Inspect the hand-calculated fixture totals without deriving them from the SQL output.

**Deliverable:** A short red-to-green record naming each command and observed result.
**Acceptance:** The initial failure is attributable to the missing implementation, not a missing dependency.
The completed implementation passes all four tests in the specified environment.
The record includes the independent expected totals of 23 and 10 units.
**Maps to:** Learning outcome 3.

### Exercise 4 — Compute a policy outcome (15 minutes)

**Task:** Use the original five fixture rows and change the illustrative starting stock to 18 units of A and 12 units of B.
Assume no replenishment, no substitutions, and lost sales for all unmet demand.
Calculate requested, filled, lost, and remaining units for each SKU.
Calculate the aggregate unit fill rate and explain its weighting.

**Deliverable:** A two-row SKU table and aggregate arithmetic.
**Acceptance:** Each SKU's requested quantity is traceable to fixture lines.
Filled units never exceed either stock or requested units; all quantities are nonnegative.
The aggregate rate uses total filled units divided by total requested units.
State that the result is a synthetic calculation, not a measured service level.
**Maps to:** Learning outcome 4.

### Exercise 5 — Seeds and fairness (20 minutes)

**Task:** Compare capacity 4 and capacity 6 using a common set of 12 generated demand values.
Explain how you know the two lists refer to the same demand values.
Identify why using the same seed can fail when two implementations make different numbers of random calls.
Suggest an experiment across multiple seeds without asserting a result you have not measured.

**Deliverable:** A comparison method, pseudocode, and a one-paragraph limitations note.
**Acceptance:** One generated demand stream feeds both capacities.
The metric is lost units per period and the input stream is not regenerated conditionally by policy.
The answer distinguishes repeatability from representativeness and names more than one independent seed.
**Maps to:** Learning outcome 5.

### Exercise 6 — Image identity (15 minutes)

**Task:** Explain the difference between the Dockerfile, an image, a tag, an image digest, and a container.
If Docker is permitted and available, build and run the image using Section 5.9.
Otherwise, review the Dockerfile and describe the expected commands without claiming to have executed them.
Explain why pinned Python package versions do not freeze the base operating-system layers.

**Deliverable:** A five-term distinction and a documented build result or an explicitly unexecuted plan.
**Acceptance:** The response identifies `python:3.11.11-slim` as a tag, not a content digest.
It describes the default test command and the alternative module command.
It does not claim the container is hardened or a complete security boundary.
**Maps to:** Learning outcome 6.

### Exercise 7 — Manifest and provenance (20 minutes)

**Task:** Draft a run manifest for the synthetic lab.
Include code identity, local modifications, Python and package versions, input identity, seed, policy, and outcome.
If using a container, include image identity and platform as well.
Add a reviewer field and an explicit data classification appropriate to synthetic inputs.

**Deliverable:** A manifest template with example values or clearly labeled placeholders.
**Acceptance:** The fixture has a SHA-256 field; seed and policy parameters are separate fields.
The record distinguishes a clean commit from a dirty working tree.
Unknown values are marked unknown instead of fabricated.
It records whether tests were actually run and their observed status.
**Maps to:** Learning outcome 7.

### Exercise 8 — Release critique (25 minutes)

**Task:** A stakeholder proposes publishing the 90.91% synthetic fill rate as the site's measured performance.
They also propose copying a live order export into the fixture and using a shared administrator token in the Docker build.
Critique the proposal and specify a safer review path.
Separate technical, statistical, privacy, and decision-authorization concerns.

**Deliverable:** A review memo with at least four distinct objections and corresponding mitigations.
**Acceptance:** The memo says the rate is synthetic, not a measured production result.
It rejects embedding credentials and demands data-owner approval for real data.
It asks for representative validation and an authorized business owner before decisions.
It does not promise that containerization alone fixes the risks.
**Maps to:** Learning outcome 8.

## 13. Self-check answers

### Answer 1 — Environment identity

The requirements file pins the direct dependencies to DuckDB 1.1.3 and pytest 8.3.4.
Python 3.11 is the intended interpreter family; the exact installed patch build should be recorded.
A direct version pin does not specify all transitive package versions, hashes, operating-system libraries, or package origin.
`pip freeze` records the installed environment but is not itself a verified artifact lock.
For an offline setup, obtain approved packages through an authorized mirror or offline distribution process.
A strong answer reports the observed environment rather than inventing installed versions.

### Answer 2 — Repository boundaries

`src/lab.py` is implementation; `tests/test_lab.py` is a behavioral test.
`fixtures/demand.csv` is a small synthetic fixture.
`requirements.txt` declares direct Python dependencies; `Dockerfile` specifies the example image build.
`.gitignore` and `.dockerignore` control common inclusion behavior for Git and Docker respectively.
A local virtual environment and generated outputs should not be the authoritative versioned inputs.
Ignore rules do not remove a secret that was already committed, copied elsewhere, or explicitly included.

### Answer 3 — Red, green, review

Before `src/lab.py` exists, importing `src.lab` during test collection should fail.
The exact exception text depends on the test runner and Python version.
After implementation, the four tests cover fixture totals, lost-sales arithmetic, seeded repeatability, and negative inputs.
The independent totals are 7 + 11 + 5 = 23 units for A and 4 + 6 = 10 units for B.
A missing pytest or DuckDB installation is an environment error, not the intended red test.
A strong report records what actually happened and does not invent a successful run.

### Answer 4 — Compute a policy outcome

| SKU | Requested, units | Starting stock, units | Filled, units | Lost, units | Remaining, units |
|---|---:|---:|---:|---:|---:|
| A | 23 | 18 | 18 | 5 | 0 |
| B | 10 | 12 | 10 | 0 | 2 |

Total requested is 23 + 10 = 33 units.
Total filled is 18 + 10 = 28 units.
Total lost is 5 units and total remaining is 2 units, distributed by SKU as above.
The aggregate unit fill rate is 28 / 33, approximately 84.85%.
The calculation weights each requested unit equally; it is not the unweighted mean of two SKU percentages.

### Answer 5 — Seeds and fairness

Generate one sequence of 12 demand values with one seeded generator.
For every value `d`, calculate `max(d - 4, 0)` and `max(d - 6, 0)` against the same `d`.
The higher capacity cannot lose more units on the same demand stream under this particular formula.
If one policy makes an extra random draw, two identically seeded generators may no longer align event by event.
Pre-generating the inputs avoids that problem for this toy model.
Repeat with independently selected seeds and report the distribution of differences; a seed is not a representative dataset.

### Answer 6 — Image identity

A Dockerfile describes build steps.
An image is the resulting packaged filesystem and configuration.
A tag is a human-readable reference that may be repointed.
A digest identifies content; record the applicable platform because multi-platform image references can resolve differently.
A container is a running or stopped instance created from an image.
The lab's default command runs pytest; the explicit alternative runs `python -m src.lab`.
A correct answer does not claim execution if Docker was not available.

### Answer 7 — Manifest rubric

A complete manifest names the code revision and records whether local changes were present.
It names the input path and SHA-256 digest, seed 37, 12 periods, and capacity 6 units per period.
It records interpreter, package versions, operating system, architecture, command, UTC timestamp, and actual test outcome.
If containerized, it records the image identity, resolved base, and platform.
The record classifies inputs as synthetic and names the reviewer and retention location when known.
Unknown values remain explicitly unknown; a placeholder is not evidence.

### Answer 8 — Release critique rubric

Reject the claim that the fixture's 90.91% is measured site performance.
The rows and stock assumptions are synthetic and do not establish representative demand or observed availability.
Reject the administrator token in the build; use an approved secret mechanism only when a real system requires one.
Request data-owner approval, privacy review, least-privilege access, and an appropriate retention policy for any live extract.
Ask for validation against representative data, clear metric definitions, and independent domain review.
Obtain an authorized decision owner and document whether the result is a forecast, a plan, a target, or a commitment.
Containerization, a passing unit test, or a fixed seed does not remove those requirements.

### Diagnostic scenario A — Module import failure

1. Observe whether the error is `No module named pytest`, `duckdb`, or `src.lab`.
2. If pytest is missing, identify the interpreter used by the command.
3. Compare its path with the virtual-environment interpreter specified in Section 5.4.
4. Install the declared requirements into the correct virtual environment if needed.
5. If `src.lab` is missing during the deliberate red step, the failure is expected.
6. If it is missing after implementation, confirm the file is named `src/lab.py`.
7. Confirm `src/__init__.py` exists and the command is run from the project root.
8. Do not mask the problem by installing an unrelated package named `src`.
9. Record the corrected command and actual result in the lab evidence.
10. A missing package failure is not evidence that the lost-sales test passed or failed.

### Diagnostic scenario B — Fixture total mismatch

1. Recalculate A's three lines and B's two lines from the fixture.
2. Check that the header is `date,sku,units` and that the intended file was opened.
3. Compare the file's SHA-256 digest with the digest recorded for the reference run.
4. A changed digest tells you the bytes differ; it does not tell you which version is correct.
5. Inspect the proposed change and its business meaning before updating an expected total.
6. Check whether a unit is an individual item or a case.
7. Inspect whether returns, cancellations, or duplicate events were mixed into demand.
8. Confirm the query groups on the expected identifier and orders the output explicitly.
9. Keep the original fixture if the aim is to reproduce the published example.
10. Treat an intentional fixture change as a new input version requiring review.

### Diagnostic scenario C — Seeded outputs differ

1. Compare the generator type, interpreter version, and seed.
2. Confirm both runs use `random.Random(seed)` rather than shared global state.
3. Compare the number and ordering of random calls made by each run.
4. Confirm the same period count and demand distribution boundaries.
5. Check that both runs receive the same policy inputs.
6. Check whether a code change introduced conditional draws.
7. Capture an explicit demand stream if the comparison needs event-by-event alignment.
8. Keep that stream fixed while computing each alternative policy's outcomes.
9. Repeat across distinct seeds to assess sensitivity to the sampled stream.
10. Do not present a single favorable seed as the distribution of possible outcomes.

### Diagnostic scenario D — Local passes, container differs

1. Compare Python patch versions and package versions inside and outside the container.
2. Compare architecture and operating-system libraries where relevant.
3. Confirm the Docker build copied the intended source, tests, and fixture.
4. Check whether an ignore rule excluded a required input.
5. Confirm the build occurred from the project root and used the intended Dockerfile.
6. Record the image ID and the actual command used to start the container.
7. Check the fixture hash in each environment if a file discrepancy is suspected.
8. Do not assume the same image tag identifies identical image bytes over time.
9. Investigate any difference before changing an assertion or publishing results.
10. A local success does not imply that the container branch was exercised.

### Diagnostic scenario E — Apparent improvement

1. State whether the reported quantity is requested, filled, lost, or remaining units.
2. Identify the population, time window, and unit of measurement.
3. Determine whether the result is from synthetic data or an approved operational extract.
4. Verify that both scenarios use the same demand stream and comparable assumptions.
5. Check whether the metric is unit fill rate, line fill rate, or order fill rate.
6. Verify how zero demand and negative or missing quantities were handled.
7. Inspect the differences across multiple seeds or time windows where appropriate.
8. Check whether the change adds costs, service constraints, or feasibility problems.
9. Ask a domain owner to review the interpretation and intended decision.
10. Label unmeasured operational impact as unknown rather than extrapolating from this fixture.

## 14. Interview questions

### 1. What does a fixed seed guarantee?

A strong answer says that a specific pseudorandom generator starts from a specified state.
It also names code path, call ordering, generator implementation, and environment as relevant conditions.
It distinguishes repeatability from a valid estimate of uncertainty.
A weak answer says the seed makes every future version of the program produce identical results.

### 2. Why write a test before a query?

A strong answer defines the expected output independently and observes an intended failure before implementation.
It explains how an explicit example catches a grouping or arithmetic defect.
It notes that a single passing fixture does not cover all data quality conditions.
A weak answer copies the query output into an assertion and calls it independent validation.

### 3. Why does `ORDER BY` matter?

A strong answer explains that SQL relations have no guaranteed order without an explicit ordering clause.
It notes that deterministic comparison also needs a complete tie-breaking key when duplicate sort keys are possible.
It distinguishes stable ordering from proving the totals are correct.
A weak answer assumes insertion order is always preserved by an analytical database.

### 4. Is a pinned `requirements.txt` a complete lock?

A strong answer distinguishes direct pins, resolved transitive versions, artifact hashes, and interpreter or platform details.
It describes trusted package sources and controlled updates.
It recognizes that an exact version can still contain a vulnerability or be obtained from an untrusted source.
A weak answer claims two direct version pins eliminate every dependency risk.

### 5. What should be included in a run manifest?

A strong answer identifies source revision and dirty state, input identity and hash, software environment, seed, and parameters.
It includes the executed command, time, platform, observed status, and output location.
It treats the manifest as an evidence record rather than proof of model quality.
A weak answer records only the filename and seed.

### 6. How would you compare two simulated policies fairly?

A strong answer supplies both policies with the same demand stream and defines the same metric and horizon.
It calls out control-flow-induced random stream misalignment and evaluates multiple seeds.
It keeps business constraints and costs visible rather than comparing service alone.
A weak answer uses different demand samples and attributes the entire difference to policy.

### 7. What does a Docker image digest add?

A strong answer says that a digest identifies image content more precisely than a mutable tag.
It also records platform and the resolved image, and describes deliberate patch updates.
It explains that a digest does not attest that the image is secure or approved.
A weak answer claims a pinned digest is equivalent to a vulnerability scan and signature verification.

### 8. How can a five-row fixture mislead?

A strong answer says it is useful as an auditable oracle but cannot establish production representativeness.
It asks about missing demand, seasonality, data grain, and units.
It distinguishes the fixture's arithmetic from operational performance.
A weak answer treats the sample fill rate as the warehouse's measured service level.

### 9. How do unit fill rate and order fill rate differ?

A strong answer gives unit fill rate as filled units divided by requested units.
It defines order fill rate using a specified rule for fully filled orders and a count of orders.
It notes that this fixture lacks an order identifier and thus cannot establish order fill rate.
A weak answer uses the two terms interchangeably.

### 10. What changes before using real data?

A strong answer requests data-owner approval, access controls, classification, retention, and data-quality checks.
It adds representative validation, documented semantics, monitoring, and review by a business owner.
It separates legal advice and production authorization from this educational exercise.
A weak answer simply replaces the CSV and republishes the toy output.

### 11. What if local and container results differ?

A strong answer compares exact inputs, package and interpreter versions, platform, code, image identity, and commands.
It inspects ordering, random draw alignment, and local uncommitted changes.
It preserves evidence before changing assertions.
A weak answer selects whichever output is more favorable.

### 12. What does a green test suite prove?

A strong answer says that the exercised assertions passed in the observed environment.
It identifies untested data cases, assumptions, security controls, and business decisions.
It treats external validation and operational approval as separate evidence.
A weak answer claims that four unit tests establish production readiness.

## 15. Key terms

These terms are defined in detail in [Section 4](#4-definitions-and-vocabulary) and applied in the linked sections.

- **Acceptance criterion:** Observable condition for completing a stated exercise; see [Exercises](#12-exercises).
- **Artifact:** Retained output or input with an identity; see [run record](#510-preserve-the-run-record).
- **Backorder:** Unfilled demand retained for later fulfillment; excluded by the [worked example](#6-worked-example).
- **Build context:** Files sent to a container builder; see [ignore rules](#55-add-ignore-rules-and-synthetic-input).
- **Capacity:** Maximum units available to fill each independent simulated period; see [implementation](#72-implementation-srclabpy).
- **Commitment:** Authorized obligation, distinct from a forecast; see [decision boundaries](#51-establish-the-decision-and-boundaries).
- **Container:** Instance of an image; see [container branch](#59-optional-container-branch).
- **Continuous integration:** Automated checks on changes; see [failure modes](#10-failure-modes-and-red-flags).
- **Data contract:** Agreed data fields, semantics, and constraints; see [data context](#8-data-standards-and-regulatory-context).
- **Dependency pin:** Declared exact package version; see [environment](#54-pin-direct-dependencies-and-isolate-python).
- **Digest:** Content-derived identifier; see [run record](#510-preserve-the-run-record).
- **DuckDB:** Embedded analytical database; see [implementation](#72-implementation-srclabpy).
- **Fill rate:** Filled units divided by requested units; see [worked example](#6-worked-example).
- **Fixture:** Fixed synthetic test input; see [fixture](#55-add-ignore-rules-and-synthetic-input).
- **Forecast:** Estimate of uncertain future demand; see [worked example](#6-worked-example).
- **Granularity:** Meaning of one record; see [data contract](#data-contract-for-the-exercise).
- **Image:** Packaged filesystem and configuration; see [container](#73-container-dockerfile).
- **Lost sales:** Unfilled demand assumed not to be backordered; see [assumptions](#11-assumptions-and-limits).
- **Manifest:** Evidence record for an execution; see [run record](#510-preserve-the-run-record).
- **Oracle:** Independent expected result; see [tests](#71-tests-teststest_labpy).
- **Plan:** Chosen action, distinct from a forecast and commitment; see [worked example](#6-worked-example).
- **Provenance:** Evidence describing where an artifact came from; see [industry practice](#9-industry-practice-and-case-evidence).
- **Pseudorandom generator:** Deterministic random-looking sequence generator; see [seeds](#answer-5--seeds-and-fairness).
- **Reproducibility:** Ability to regenerate results under stated conditions; see [mental model](#3-the-core-mental-model).
- **Sandbox:** Restricted learning environment, not automatically secure; see [sandbox boundary](#the-boundary-of-this-sandbox).
- **Seed:** Generator initialization input; see [simulation](#72-implementation-srclabpy).
- **SKU:** Stock-keeping unit identifier; see [data contract](#data-contract-for-the-exercise).
- **Stockout:** Demand exceeding available stock under the defined scope; see [worked example](#6-worked-example).
- **Synthetic data:** Artificial records used instead of real customer records; see [data context](#8-data-standards-and-regulatory-context).
- **Target:** Desired performance level rather than a measured result; see [worked example](#6-worked-example).
- **Test-first:** Specify and observe an expected failing behavior before implementing it; see [tests](#56-write-the-tests-before-the-implementation).
- **Virtual environment:** Isolated Python package installation context; see [setup](#54-pin-direct-dependencies-and-isolate-python).

## 16. Sources

These sources support tool behavior and general practice; none demonstrates a production outcome for the synthetic example.
Consult the current version of a standard or regulation before applying it.

1. Python Software Foundation, [venv — Creation of virtual environments](https://docs.python.org/3.11/library/venv.html).
   The official interpreter documentation explains environment creation and isolation.
2. Python Software Foundation, [random — Generate pseudo-random numbers](https://docs.python.org/3.11/library/random.html).
   The documentation describes generators, seeds, and reproducibility limitations.
3. Python Packaging Authority, [Installing packages](https://packaging.python.org/en/latest/tutorials/installing-packages/).
   This guide covers package installation and environment practices.
4. pip maintainers, [Repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/).
   This guide distinguishes pinning and hash-checking approaches.
5. DuckDB, [CSV import](https://duckdb.org/docs/stable/data/csv/overview.html).
   This documentation describes reading CSV data with DuckDB.
6. DuckDB, [Python DB API](https://duckdb.org/docs/stable/clients/python/dbapi.html).
   This documentation describes connections and parameterized execution.
7. pytest developers, [Getting started](https://docs.pytest.org/en/stable/getting-started.html).
   This documentation describes collecting and running tests.
8. Docker, [Dockerfile reference](https://docs.docker.com/reference/dockerfile/).
   This reference defines the instructions used in the lab image.
9. Docker, [Image digests](https://docs.docker.com/dhi/core-concepts/digests/).
   This documentation explains content-addressable image references.
10. Open Container Initiative, [Image Specification](https://github.com/opencontainers/image-spec).
    This specification describes image format and content-addressable components.
11. NIST, [Secure Software Development Framework, SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final).
    This framework describes secure development practices rather than certifying this exercise.
12. SLSA, [Supply-chain Levels for Software Artifacts](https://slsa.dev/spec/).
    This specification describes software supply-chain provenance and requirements.
13. CISA, [Emergency Directive 21-01: Mitigate SolarWinds Orion Code Compromise](https://www.cisa.gov/news-events/directives/ed-21-01-mitigate-solarwinds-orion-code-compromise).
    This public incident directive provides context for the named supply-chain case.
14. Reproducible Builds, [Reproducible Builds documentation](https://reproducible-builds.org/docs/).
    The project documents reproducible build techniques and their scope.
15. ISO, [ISO 8601 date and time format](https://www.iso.org/iso-8601-date-and-time-format.html).
    This overview describes the date and time representation standard.
16. GS1, [GS1 standards](https://www.gs1.org/standards).
    The standards catalog provides context for supply-chain identifiers and data exchange.
17. European Union, [Regulation (EU) 2016/679, GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
    The official legal text applies only within its defined scope and is not interpreted here as legal advice.

## 17. Links

- [Module 30 topic index](_index.md).
- [Module 30 survey and lab map](../../../course/30_simulation_optimization_production_labs.md).
- [Demand forecast pipeline](02_demand-forecast-pipeline.md), when the planned sibling is published.
- [Inventory policy simulation](03_inventory-policy-simulation.md), when the planned sibling is published.
- [Model service and container](08_model-service-container.md), when the planned sibling is published.
- [This topic's executable fixture and tests](#5-the-mechanics-step-by-step).
- [Module expansion specification](../../../course/MODULE_EXPANSION_SPEC.md).

