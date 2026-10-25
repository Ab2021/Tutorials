# Python Engineering

> Module 03 — Data Engineering and Interoperability
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01; Python fundamentals
> Version 2026.3

## 1. Why this topic exists

A clinical pipeline is a Python program that someone else must run in eighteen months. The science may
be settled, the SQL may be correct, and the model may be validated, but if the program cannot be
installed, configured, and tested from a clean machine, none of that matters. The failure is not
analytical. It is engineering.

The same incidents recur across biomedical teams.

- A pipeline works on the analyst's laptop because it imports a package installed two years ago at a
  version that is no longer on the package index. A new hire cannot reproduce the result, and the
  published analysis cannot be re-run.
- Two dependencies require incompatible versions of a numerical library. The resolver picks one, the
  other silently degrades, and a conversion factor changes in the fourth decimal place. The cohort count
  moves by 0.4%, which is enough to change a secondary analysis.
- A configuration file contains a site-specific database host and an API token. The file is committed to
  a repository. The token is rotated after an incident, but the copy remains in the commit history.
- A function that computes age uses the current date. The pipeline is re-run for a historical cohort and
  produces different eligibility, because "current" moved by two years.
- A module writes an intermediate file to a hard-coded path. Two runs on the same day overwrite each
  other, and the second run reads a partial file from the first.
- A test suite exists but tests only the happy path. The boundary case, a patient aged exactly 18, is
  wrong by one day, and every cohort that includes 18-year-olds is off by a small, systematic amount.

None of these is exotic. They are the ordinary failure modes of code that was written to answer a
question once and then had to answer it repeatedly, under audit, by other people.

This topic teaches the engineering discipline that makes biomedical code reproducible: a declared
environment, a typed boundary, a configuration layer that fails before a job runs, tests that encode
clinical invariants, structured logs that never contain protected health information (PHI), and a release
process that produces an artifact someone else can install.

> **Note:** The code examples use only the Python standard library so that every block runs in a bare
> interpreter. Production projects will use third-party libraries, but the principles are the same:
> declare dependencies, isolate the environment, type the boundaries, test the logic, and log the run.

## 2. Learning outcomes

By the end of this topic you can:

1. **Structure** a Python project with a source layout, declared metadata, and separated runtime and
   development dependencies. Assessed in Exercise 1.
2. **Build** a reproducible environment from a lockfile and explain why a version range is not a lock.
   Assessed in Exercise 2 and self-check item 3.
3. **Implement** a configuration layer with documented precedence that validates before execution.
   Assessed in Exercise 3.
4. **Write** type-annotated functions at the boundaries and pure functions for clinical logic.
   Assessed in Exercise 4.
5. **Test** boundary conditions, invariants, and failure paths, and distinguish unit from integration
   tests. Assessed in Exercise 5.
6. **Log** run context and row counts without logging PHI. Assessed in Exercise 6.
7. **Package** and version a release so another engineer can install and run it. Assessed in Exercise 7.
8. **Diagnose** a "works on my machine" failure by tracing the dependency and configuration closure.
   Assessed in Exercise 8.

## 3. The core mental model

The unit of reproducibility is not the script. It is the closure of everything the script depends on.

```text
                    +--------------------------------------+
                    |  ARTIFACT                            |
                    |  wheel or container image            |
                    |  immutable, hash-addressed           |
                    +------------------^-------------------+
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
+-------+--------+            +--------+--------+            +--------+--------+
| CODE           |            | ENVIRONMENT     |            | CONFIGURATION   |
| git commit     |            | interpreter     |            | defaults        |
| package version|            | lockfile        |            | site file       |
|                |            | OS libraries    |            | env vars        |
+-------+--------+            +--------+--------+            +--------+--------+
        |                              |                              |
        +------------------------------+------------------------------+
                                       |
                          +------------+------------+
                          | INPUT DATA SNAPSHOT     |
                          | path + hash + schema    |
                          +-------------------------+
```

Every arrow is a dependency that can change the output. If any one of the four boxes is unpinned, the
result is not reproducible, even if the code is identical. A git commit without a lockfile reproduces
the source, not the behavior. A lockfile without a data hash reproduces the environment, not the result.

### 3.1 The pure core and the I/O shell

Clinical logic should be pure: given the same inputs, it returns the same output and has no side effects.
Reading files, calling APIs, writing tables, and reading the clock belong in a thin shell around the core.

```text
+-------------------------------------------------------------+
|  SHELL: I/O, time, network, database, filesystem             |
|  +-------------------------------------------------------+  |
|  |  CORE: eligibility, conversion, dedup rules, scoring  |  |
|  |  pure functions, explicit inputs, typed outputs       |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

Why this matters clinically: a pure eligibility function can be tested against a table of boundary cases
without a database, a clock, or a network. A function that reads `datetime.now()` internally cannot be
tested for a historical cohort without monkey-patching, and monkey-patching is a sign that the clock
should have been a parameter.

### 3.2 Configuration has one direction

Configuration flows from general to specific, and the most specific value wins:

```text
defaults in code  ->  base config file  ->  site config file  ->  environment variables  ->  CLI flags
lowest precedence                                                              highest precedence
```

The order must be documented and tested. A common bug is that environment variables override a site file
in one module and are overridden by it in another, because two libraries implement configuration
differently. Pick one mechanism at the application boundary and pass resolved values inward.

Secrets are not configuration. A database password in a YAML file is a configuration file that must be
encrypted, access-controlled, and rotated. A secret manager or an injected environment variable is
better, and the code should fail loudly when a required secret is absent rather than falling back to a
default.

### 3.3 The test pyramid for data code

```text
                 /\
                /  \        end-to-end: 2-5 tests
               /----\       run the pipeline on a fixture, compare output hash
              /      \
             /--------\     integration: 10-30 tests
            /          \    real database, real files, contract checks
           /------------\
          /              \  unit: 100+ tests
         /                \ pure functions, boundaries, conversions, errors
        /------------------\
```

Most tests should be unit tests because they are fast and precise about what failed. Integration tests
prove that the pieces connect. End-to-end tests prove that the pipeline produces the expected artifact,
and they are expensive, so there should be few.

The common inversion is a team with fifty end-to-end tests and no unit tests. Failures take an hour to
diagnose, coverage of boundary cases is accidental, and the suite is skipped before releases because it
is too slow.

### 3.4 Types are documentation that can fail

Type hints do not change runtime behavior in CPython. Their value is that a checker such as mypy or pyright
can prove that a function is called with the wrong shape before the pipeline runs, and that a reader can
see the contract without reading the body.

Annotate boundaries: function parameters, return values, configuration objects, and data-frame schemas.
Do not annotate every local variable; the value is at the interfaces, not in the middle of a loop.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Module | A single `.py` file | Confused with a package | — |
| Package | A directory of modules with `__init__.py`, or a distributable project | Confused with a distribution | — |
| Distribution | An installable artifact published to an index | Called a package | Wheel or sdist |
| Wheel | A built binary distribution with a `.whl` extension | Assumed to be source | PEP 427 |
| Sdist | A source distribution archive | Assumed to build everywhere | `.tar.gz` |
| pyproject.toml | Standard project metadata and build configuration file | Used only for tool settings | PEP 518, PEP 621 |
| Build backend | The tool that builds a wheel or sdist | Confused with the build frontend | PEP 517 |
| Lockfile | A resolved, pinned set of exact versions and hashes | Confused with a requirements range | `uv.lock`, `pylock.toml` |
| PEP 751 | Standard for `pylock.toml`, finalized March 2025 | Assumed supported by every installer | — |
| Virtual environment | An isolated interpreter with its own site-packages | Assumed to isolate system libraries | `venv` |
| SemVer | Semantic versioning: major, minor, patch | Used loosely for internal versions | `MAJOR.MINOR.PATCH` |
| Pin | An exact version constraint | Confused with a lower bound | — |
| Range | A constraint allowing multiple versions | Assumed reproducible | — |
| Transitive dependency | A dependency of a dependency | Ignored in reviews | — |
| Dependency confusion | An attack using a public package with an internal name | Considered only for large companies | — |
| SBOM | Software bill of materials listing components | Treated as a security guarantee | SPDX, CycloneDX |
| Reproducible build | A build that yields identical bytes from identical inputs | Confused with a successful build | — |
| `src` layout | Placing importable code under `src/` | Considered stylistic | — |
| Entry point | A console script declared in metadata | Hard-coded in a shell script instead | — |
| Type hint | Annotation describing expected types | Assumed to be enforced at runtime | PEP 484 |
| Protocol | Structural typing interface | Confused with an abstract base class | `typing.Protocol` |
| Dataclass | Class with generated init, repr, and equality | Used for objects with invariants that need validation | — |
| Frozen dataclass | Immutable dataclass | Assumed deeply immutable | — |
| Validation | Rejecting invalid input at the boundary | Confused with testing | — |
| Configuration precedence | Order in which config sources override each other | Undocumented | — |
| Secret | Credential or key that must not be in source control | Stored in YAML | — |
| Structured logging | Log records as key-value fields | Printing prose | JSON lines |
| Correlation ID | Identifier linking log lines to one run | Absent | UUID |
| Idempotent | Re-running produces the same state | Confused with retry-safe | — |
| Deterministic | Same inputs produce same outputs | Confused with fast | — |
| Test fixture | Reusable test input or setup | Confused with production data | — |
| Property-based test | Test asserting an invariant over generated inputs | Confused with example tests | Hypothesis |
| Coverage | Fraction of lines executed by tests | Treated as a quality score | Percent |
| CI | Continuous integration: automated build and test on change | Confused with deployment | — |
| Artifact | Immutable build output | Rebuilt at deploy time | Wheel, image digest |
| Image digest | Content hash of a container image | Confused with a tag | `sha256:...` |

Two distinctions carry most of the practical weight.

**Pin versus range.** A range says "any version in this interval is acceptable"; a lock says "this exact
set was resolved and tested." Libraries should declare ranges so that consumers can resolve; applications
should ship locks so that they reproduce. Confusing the two produces either an uninstallable library or
an unreproducible application.

**Validation versus test.** Validation rejects bad input at runtime and protects the pipeline. A test
proves that the code behaves as intended during development. A pipeline needs both: validation catches a
bad site configuration in production, and a test catches a logic error before release.

## 5. The mechanics, step by step

The procedure builds a small, installable, tested Python package for a clinical transformation. It
assumes Python 3.11 or later; the code in Section 7 was executed on CPython 3.11.15.

### Step 1 — Name the package and its boundary

Input: a description of the transformation.
Output: a package name, a one-sentence responsibility, and an explicit list of what is out of scope.

Example: `cohort_core` converts raw laboratory values to canonical units, validates ranges, and computes
eligibility flags. It does not read files, call databases, or write outputs.

Reason: a package with one responsibility can be tested and reused. A package named `utils` becomes a
dependency of everything and owned by no one.

### Step 2 — Choose the interpreter and support matrix

Input: the deployment environment and library constraints.
Output: a minimum supported Python version and a tested maximum.

Record the exact versions tested, for example 3.11, 3.12, and 3.13. Python 3.14 was released on
7 October 2025 and is supported by major libraries, but a pipeline should adopt it only after its
dependencies publish wheels and its numerical stack is validated.

Reason: the interpreter version changes language semantics and C-extension behavior. "Python 3" is not a
support matrix.

### Step 3 — Create the source layout

Input: the package name.
Output: a directory tree with `src/<package>/`, `tests/`, `pyproject.toml`, and a README.

```text
project/
  pyproject.toml
  README.md
  src/cohort_core/
    __init__.py
    config.py
    units.py
    eligibility.py
  tests/
    test_units.py
    test_config.py
    test_eligibility.py
  config/
    base.yaml
    sites/
  dags/
  data/
```

Reason: the `src` layout prevents accidental imports from the working directory. Without it, a test can
import the local folder instead of the installed package and pass while the built artifact is broken.

### Step 4 — Declare metadata in `pyproject.toml`

Input: package name, version, description, dependencies, entry points.
Output: a PEP 621 `[project]` table and a build-system table.

Declare `requires-python`, runtime dependencies with ranges, and a build backend. Do not put
configuration in `setup.py` for a new project; PEP 621 is the standard and every current build frontend
reads it.

Reason: metadata is the contract with the installer. A missing `requires-python` lets a user install the
package on an unsupported interpreter and fail at runtime instead of at resolution.

### Step 5 — Separate runtime and development dependencies

Input: the dependency list.
Output: runtime dependencies in `[project.dependencies]`, development tools in a dependency group.

Runtime dependencies are what the pipeline needs to run: a database driver, a parsing library, a numerical
library. Development dependencies are pytest, mypy, ruff, and similar tools. Installing development
tools in production widens the attack surface and slows builds.

Reason: the production image should contain only what production executes. The separation also makes the
dependency review smaller and more meaningful.

### Step 6 — Constrain versions deliberately

Input: each dependency and its release history.
Output: a lower bound that includes the feature you need and an upper bound that excludes the next major
version, or a documented reason for no upper bound.

Example: `pandas>=2.2,<3` says the code targets the 2.x API and has not been validated against 3.x.
`pandas>=2.2` says any future version is acceptable, which is a claim the team has not tested.

Reason: upper bounds prevent surprise breakage; lower bounds document the feature requirement. Both are
statements about testing, not about fashion.

### Step 7 — Lock the application environment

Input: the direct dependencies and the resolved transitive graph.
Output: a lockfile with exact versions and hashes for every platform in the support matrix.

Use a tool that resolves the full graph: `uv lock` produces `uv.lock`; PEP 751 defines `pylock.toml`,
which uv can export and install from. A requirements file generated by `pip freeze` records what happened
to be installed, not what the resolver chose from declared constraints.

Reason: transitive dependencies are where breakage hides. Pinning only direct dependencies leaves the
graph free to change.

### Step 8 — Build the artifact

Input: the source tree and the lockfile.
Output: a wheel, an sdist, and their hashes.

Build in a clean environment, not the development virtual environment. Record the artifact hash in the
release notes. If the build is not byte-reproducible, record the inputs and the build environment so that
the difference can be explained.

Reason: the artifact is what gets deployed. Testing the source tree and deploying a wheel tests two
different things.

### Step 9 — Install into an isolated environment

Input: the wheel and the lockfile.
Output: a virtual environment or container image with only the resolved dependencies.

Create the environment from scratch in CI on every run. Do not reuse a cached environment for the release
build; caching is for speed in development, not for proof of reproducibility.

Reason: a clean install is the only test of the install path. Most "works on my machine" failures are
missing transitive dependencies that were installed manually months earlier.

### Step 10 — Implement configuration as a typed object

Input: the configuration sources.
Output: a frozen dataclass or validated model with explicit fields and defaults.

Every field has a type, a default where appropriate, and a validation rule. Unknown fields should be
rejected, not ignored; a typo in a site configuration should fail at startup, not silently use a default.

Reason: configuration errors are the most common cause of a pipeline producing a plausible wrong result.
A misspelled `lookback_day` becomes a default 365 and nobody notices.

### Step 11 — Document and test precedence

Input: the configuration sources.
Output: a precedence table and a test for each transition.

Example: defaults, then `config/base.yaml`, then `config/sites/<site>.yaml`, then environment variables
prefixed `COHORT_`, then CLI flags. Test that an environment variable overrides the site file and that a
CLI flag overrides the environment.

Reason: precedence that is implemented but untested drifts. A library upgrade can change the order
without any code change in the project.

### Step 12 — Keep secrets out of configuration files

Input: credentials, tokens, keys.
Output: references resolved from a secret manager or injected environment variables at runtime.

Fail at startup when a required secret is missing. Never log the value. Never include it in an error
message that is written to a shared log.

Reason: a secret in a config file is a secret in a backup, a repository, and a support bundle. Rotation
then requires finding every copy.

### Step 13 — Type the boundaries

Input: the public functions and data structures.
Output: annotations on parameters, returns, and configuration objects.

Use `Protocol` for dependencies that have multiple implementations, such as a data reader that can be a
CSV file in tests and a database in production. Use concrete types inside the core.

Reason: annotations at boundaries let a checker catch interface mismatches before runtime and let a
reader understand the contract without reading the implementation.

### Step 14 — Keep clinical logic pure

Input: the transformation functions.
Output: functions that take all inputs as parameters and return values, with no I/O and no clock access.

Pass `as_of: date` explicitly instead of calling `date.today()`. Pass a reader interface instead of
opening a file. Return a result object rather than printing.

Reason: purity makes boundary tests trivial and makes historical re-runs correct. A function that reads
the clock cannot reproduce a 2024 cohort in 2026.

### Step 15 — Use `Decimal` for unit conversion and thresholds

Input: numeric clinical values and conversion factors.
Output: decimal arithmetic with an explicit rounding rule.

Binary floating point cannot represent 0.1 exactly. For a conversion factor such as 88.4, the error is
small, but for accumulation over millions of rows and for threshold comparisons at exact boundaries, it
can change a classification. Use `Decimal` for conversion and boundary decisions, and convert to float
only at the analytics boundary if the downstream system requires it.

Reason: clinical thresholds are exact by regulation or protocol. A value of 1.30 that becomes
1.2999999999 can fall on the wrong side of a cutoff.

### Step 16 — Write unit tests for boundary cases

Input: each pure function.
Output: tests for the lower bound, the upper bound, just inside, just outside, null, and invalid input.

For an age threshold of 18, test 17 years 364 days, 18 years exactly, and 18 years 1 day. For a unit
conversion, test zero, a normal value, and a value at the plausibility limit.

Reason: boundary errors are systematic and survive happy-path tests. They are also the errors most likely
to affect eligibility.

### Step 17 — Add property-based tests for invariants

Input: a transformation with an algebraic property.
Output: generated-input tests asserting the property.

Examples: converting from unit A to B and back returns the original value within rounding; sorting a list
and reversing it twice returns the original order; deduplication never increases the row count.

Reason: example tests check the cases the author thought of. Property tests check the cases the author
did not.

### Step 18 — Add integration tests with fixtures

Input: a small, synthetic dataset with known expected output.
Output: a test that runs the real reader and writer against the fixture.

Keep fixtures small enough to review, and store them in version control with a documented provenance.
Never use production PHI in a fixture. If the fixture is derived from real data, de-identify it and record
the method.

Reason: integration tests catch wiring errors that unit tests cannot: a column renamed in the reader, a
timestamp parsed in the wrong timezone, a writer that drops nulls.

### Step 19 — Log structured run context

Input: the pipeline execution.
Output: JSON log lines with run ID, code version, config fingerprint, input hash, stage, row counts, and
duration.

Every log line should be machine-parseable. Use a correlation ID so that lines from one run can be joined.
Log rejected rows with a reason code, not with the row contents.

Reason: logs are the first diagnostic tool during an incident. Prose logs cannot be aggregated; logs with
PHI cannot be shared.

### Step 20 — Run the test matrix in CI

Input: the test suite and the support matrix.
Output: a CI configuration that runs lint, type check, unit tests, integration tests, and a build on every
change.

Fail the build on type errors and on test failures. Do not allow a "known failure" list to grow without
an owner and an expiry date.

Reason: a test that is not run is documentation. CI is the mechanism that makes the tests a gate rather
than a suggestion.

### Step 21 — Version and release

Input: a validated build.
Output: a semantic version, a changelog entry, a tagged commit, and a published artifact.

Use SemVer for the package: a breaking change to a public function increments the major version; a new
feature increments the minor; a fix increments the patch. Record the supported Python versions and the
lockfile hash in the release notes.

Reason: downstream teams need to know whether an upgrade is safe. A version number that does not reflect
compatibility forces every consumer to read the diff.

### Step 22 — Build the container image reproducibly

Input: the wheel and a base image.
Output: an image tagged with the version and referenced by digest in deployment.

Pin the base image by digest, install from the lockfile, run as a non-root user, and copy only the
artifact and its runtime dependencies. Record the image digest in the run metadata.

Reason: a tag such as `latest` can point to different bytes tomorrow. The digest is the identity.

### Step 23 — Handle configuration and dependency edge cases

Input: the validation and release procedure.
Output: a decision table that states the behavior for each edge case.

| Edge case | Behavior | Test |
|---|---|---|
| Missing required config value | Fail at startup with the field name | Instantiate config without the field |
| Unknown config key | Reject, do not ignore | Add a typo key; expect error |
| Site identifier collision | Reject if two sites resolve to the same fingerprint | Register two sites with the same ID |
| Null optional value | Use the documented default, never a silent zero | Pass `None`; assert default |
| Timezone-naive timestamp | Reject at the boundary, require an offset | Feed a naive value; expect error |
| Duplicate dependency with conflicting ranges | Fail resolution, do not pick silently | Add an incompatible pin |
| Partial test failure | Fail the build; do not publish | Inject one failing test |
| Lockfile missing a hash | Reject the install in CI | Remove a hash; expect failure |

Reason: each edge case is a decision that otherwise gets made accidentally by a library default. The
table makes the behavior reviewable and testable.

### Branch point A — A notebook is the deliverable

If the deliverable is an exploratory analysis, a notebook is acceptable, provided it is executed
top-to-bottom in a fresh kernel, its environment is recorded, and its inputs are hashed. A notebook that
depends on hidden state from earlier cells is not reproducible even if it runs.

### Branch point B — A library, not an application

A library should declare ranges rather than pins and should not ship a lockfile. Its CI matrix must test
the lowest and highest supported versions of its dependencies. An application should pin and ship a lock.
The distinction is the consumer, not the code size.

### Branch point C — Regulated validation

If the software is part of a regulated submission, the development process itself becomes evidence.
Version control, code review, test records, and change control must be documented. A tool such as a
validated notebook environment may be required. The engineering practices here are necessary but not
sufficient; the validation plan is a separate artifact.

## 6. Worked example

A research team maintains a laboratory conversion package used by three cohorts. The package converts
creatinine from milligrams per deciliter (mg/dL) to micromoles per liter (µmol/L) using a factor of 88.4,
validates the result against a plausibility range, and classifies values against a threshold of
106 µmol/L.

### 6.1 The incident

A new analyst installs the package on a fresh machine. The install succeeds. The cohort count differs from
the published count by 312 patients out of 7,412, a difference of 4.2%. The investigation finds:

1. The published run used a version of the conversion package that rounded with `ROUND_HALF_UP` at two
   decimal places.
2. The fresh install resolved a newer transitive dependency that changed the default decimal context
   precision from 28 to 15 significant digits in one code path.
3. Values exactly at the threshold 106.00 were classified as below the threshold in the new environment
   because the computed value was 105.99999999999999.

The logic was unchanged. The environment was not.

### 6.2 The arithmetic

The cohort has 7,412 patients. Of them, 1,940 have a creatinine value within 0.01 µmol/L of the threshold.
In the new environment, 312 of those cross from at-or-above to below. The classification change is
312 / 7,412 = 4.21% of the cohort, concentrated entirely at the boundary.

The conversion for a value of 1.20 mg/dL:

- Exact decimal: 1.20 × 88.4 = 106.08 µmol/L.
- Binary float: 1.2 × 88.4 = 106.08000000000001 (on CPython 3.11), which rounds to 106.08 at two places.
- Truncation at one decimal: 106.0, which is at the threshold and would classify as at-or-above if the
  rule is `>= 106`.
- Rounding half-up at zero places: 106, same classification.

The failure appears when the comparison is made before rounding, with a float that lands just below an
integer threshold. The fix is to compare `Decimal` values quantized to the reporting precision, not raw
binary floats.

### 6.3 The remediation

The team makes four changes.

1. The conversion function uses `Decimal` and an explicit `quantize(Decimal("0.01"), ROUND_HALF_UP)`.
2. The threshold comparison is performed on the quantized value.
3. The package declares `requires-python = ">=3.11"` and pins the decimal behavior in a test.
4. The release notes record the lockfile hash and the supported interpreter versions.

The re-run reproduces the published count exactly at 7,412.

### 6.4 Test table

| Input mg/dL | Exact µmol/L | Quantized | Classification at 106 |
|---:|---:|---:|---|
| 1.19 | 105.196 | 105.20 | below |
| 1.20 | 106.08 | 106.08 | at or above |
| 1.199 | 105.9916 | 105.99 | below |
| 1.2001 | 106.08884 | 106.09 | at or above |
| 0.00 | 0.00 | 0.00 | below |
| 50.00 | 4420.00 | 4420.00 | above, implausible |

The last row is a plausibility failure, not a classification success. The validator should reject values
above 4,000 µmol/L as likely unit errors, because a creatinine of 50 mg/dL is not physiologically
compatible with life at that precision. Validation and classification are separate concerns.

### 6.5 What the example demonstrates

The defect was not in the formula. It was in the interaction between floating-point representation, an
implicit rounding rule, and an environment that resolved differently. A lockfile would have prevented the
incident; a boundary test would have caught it before release; a `Decimal` comparison would have made it
impossible. Good engineering uses all three, because each covers a different failure.

### 6.6 Resolved configuration and call graph

The run that produced the published cohort used this resolved configuration:

| Parameter | Value | Source |
|---|---|---|
| `site_id` | `SITE-A` | Site file |
| `min_age` | 18 | Default |
| `lookback_days` | 730 | Site file |
| `creatinine_threshold_umol_l` | 106.00 | Default |
| Configuration fingerprint | `e8ce12697a20` | Computed |
| Interpreter | CPython 3.11.15 | Image |
| Package version | 1.4.2 | Wheel |

The call path is `load_site_config()` -> `SiteConfig.__post_init__()` -> `creatinine_umol_l()` ->
`classify_creatinine()` -> `is_plausible()`. Each function is pure except the first, which reads the site
file. The fingerprint is computed from the resolved object, not from the file, so an environment
variable that changes `lookback_days` changes the fingerprint.

### 6.7 Cohort-level reconciliation of the fix

The unit fix touches one column, but the effect propagates to eligibility and to the outcome flag.

| Stage | Before fix | After fix | Difference |
|---|---:|---:|---:|
| Patients in cohort | 7,412 | 7,412 | 0 |
| Creatinine values converted | 7,412 | 7,412 | 0 |
| Values within ±0.01 of threshold | 1,940 | 1,940 | 0 |
| Classification changes at threshold | — | 312 | 312 |
| Readmission flags changed | — | 38 | 38 |
| Reported rate | 15.084% | 15.597% | +0.513 pp |

The cohort count is unchanged because eligibility does not depend on creatinine. The rate changes because
38 patients crossed the readmission definition after the AKI flag changed. The other 274 patients whose
classification changed did not change the readmission outcome; they changed an AKI feature used in a
secondary analysis.

### 6.8 Second run and fingerprint check

| Check | Run 1 | Run 2 | Expected |
|---|---|---|---|
| Tests passed | 4/4 | 4/4 | Equal |
| Configuration fingerprint | `e8ce12697a20` | `e8ce12697a20` | Equal |
| Cohort rows | 7,412 | 7,412 | Equal |
| Rate | 15.597% | 15.597% | Equal |

Changing `lookback_days` from 730 to 365 changes the fingerprint but does not change this cohort, because
the cohort has no exclusions in the 365-730 day window that affect it. The fingerprint change is still
recorded: it signals that a different configuration produced the run, even when the output happens to
match.

### 6.9 Two diagnostic checks

**Check A: threshold sweep.** Convert 11 values from 1.195 to 1.205 mg/dL in 0.001 steps. Expected: 5
below the threshold (1.195-1.199) and 6 at or above (1.200-1.205), with the transition between 1.199 and
1.200. A sweep that shows a different transition point means the rounding mode or the factor changed.

**Check B: dependency closure.** The project declares 3 runtime dependencies. The lockfile resolves 41
packages including transitive dependencies. Expected: 41, with a hash for each. A count of 40 means a
dependency was dropped; a count of 42 means a new transitive package appeared without review. Either is a
release blocker until explained.

A third check that is easy to omit: run the test suite with the system clock set to a historical date. If
any test fails, the code reads the clock internally, and a historical cohort cannot be reproduced.

### 6.10 Decision table for an environment mismatch

| Symptom | First diagnostic | Likely cause | Action |
|---|---|---|---|
| Cohort differs by <1% | Compare config fingerprints | Configuration drift | Pin the config; re-run |
| Cohort differs by 4% | Compare lockfile hashes | Dependency resolution | Rebuild from lock; re-test |
| Boundary cases flip | Run threshold sweep | Rounding or float comparison | Use `Decimal`; quantize |
| Tests pass locally, fail in CI | Compare interpreter versions | Unsupported Python | Pin `requires-python`; matrix |
| Install fails on a new machine | Check lockfile hashes | Missing platform wheel | Resolve for the platform |
| Same input, different output | Run twice; compare hashes | Clock, randomness, ordering | Remove nondeterminism |

### 6.11 What would falsify the fix

The fix is falsified if any of the following occurs:

1. A value exactly at 106.00 µmol/L classifies as below on a second platform. That would mean the
   comparison still depends on binary representation.
2. The threshold sweep transitions at a different input. That would mean the factor or the rounding
   changed.
3. A historical re-run produces a different cohort with the same configuration fingerprint. That would
   mean the fingerprint does not capture a relevant input.
4. The dependency closure changes without a lockfile change. That would mean the install is not
   reproducible.

Each is a test, not an opinion. A fix without a falsification condition is a hope.

## 7. Code, query, or calculation

### 7.1 Configuration, conversion, and tests (standard library, executable)

The script below is a complete, runnable miniature of the package described in Section 6. It validates
configuration, converts with `Decimal`, and runs four tests. It was executed on CPython 3.11.15.

```python
"""Miniature cohort_core package. Python 3.11+, standard library only."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable


@dataclass(frozen=True)
class SiteConfig:
    site_id: str
    min_age: int = 18
    lookback_days: int = 365
    creatinine_threshold_umol_l: Decimal = Decimal("106.00")

    def __post_init__(self) -> None:
        if not self.site_id or not self.site_id.replace("-", "").isalnum():
            raise ValueError(f"invalid site_id: {self.site_id!r}")
        if not 0 <= self.min_age <= 120:
            raise ValueError(f"min_age out of range: {self.min_age}")
        if self.lookback_days < 0:
            raise ValueError(f"lookback_days must be >= 0: {self.lookback_days}")
        if self.creatinine_threshold_umol_l <= 0:
            raise ValueError("threshold must be positive")

    def fingerprint(self) -> str:
        payload = {
            "site_id": self.site_id,
            "min_age": self.min_age,
            "lookback_days": self.lookback_days,
            "creatinine_threshold_umol_l": str(self.creatinine_threshold_umol_l),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


CREATININE_FACTOR = Decimal("88.4")
REPORTING_PRECISION = Decimal("0.01")


def creatinine_umol_l(mg_dl: Decimal) -> Decimal:
    """Convert creatinine mg/dL to umol/L, quantized for reporting."""
    if mg_dl < 0:
        raise ValueError("creatinine cannot be negative")
    return (mg_dl * CREATININE_FACTOR).quantize(
        REPORTING_PRECISION, rounding=ROUND_HALF_UP
    )


def classify_creatinine(value_umol_l: Decimal, threshold: Decimal) -> str:
    """Classify a quantized value; boundary is inclusive at the threshold."""
    return "at_or_above" if value_umol_l >= threshold else "below"


def is_plausible(value_umol_l: Decimal) -> bool:
    return Decimal("1.00") <= value_umol_l <= Decimal("4000.00")


# --- a tiny deterministic test harness (pytest is not available here) ---

def test_conversion_boundary() -> None:
    assert creatinine_umol_l(Decimal("1.20")) == Decimal("106.08")
    assert classify_creatinine(
        creatinine_umol_l(Decimal("1.20")), Decimal("106.00")
    ) == "at_or_above"


def test_conversion_just_below() -> None:
    assert creatinine_umol_l(Decimal("1.199")) == Decimal("105.99")
    assert classify_creatinine(
        creatinine_umol_l(Decimal("1.199")), Decimal("106.00")
    ) == "below"


def test_invalid_config_rejected() -> None:
    try:
        SiteConfig(site_id="site one", min_age=17)
    except ValueError as exc:
        assert "site_id" in str(exc)
    else:
        raise AssertionError("invalid site_id was accepted")


def test_fingerprint_stable() -> None:
    a = SiteConfig(site_id="SITE-A")
    b = SiteConfig(site_id="SITE-A")
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 12


TESTS: dict[str, Callable[[], None]] = {
    "test_conversion_boundary": test_conversion_boundary,
    "test_conversion_just_below": test_conversion_just_below,
    "test_invalid_config_rejected": test_invalid_config_rejected,
    "test_fingerprint_stable": test_fingerprint_stable,
}

if __name__ == "__main__":
    failures: list[str] = []
    for name in sorted(TESTS):
        try:
            TESTS[name]()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"FAIL {name}: {exc}")
    print(f"{len(TESTS) - len(failures)}/{len(TESTS)} passed")

    cfg = SiteConfig(site_id="SITE-A", lookback_days=730)
    print("config_fingerprint", cfg.fingerprint())
    print("creatinine_1.20", creatinine_umol_l(Decimal("1.20")))
    print("plausible_50mgdl", is_plausible(creatinine_umol_l(Decimal("50.00"))))
```

Expected output:

```text
PASS test_conversion_boundary
PASS test_conversion_just_below
PASS test_fingerprint_stable
PASS test_invalid_config_rejected
4/4 passed
config_fingerprint e8ce12697a20
creatinine_1.20 106.08
plausible_50mgdl False
```

The fingerprint depends on the exact canonical JSON and is stable across runs and machines. If the
configuration changes, the fingerprint changes, which is the point: it can be recorded in every output and
compared during an incident.

### 7.2 Dependency audit with `tomllib` (standard library, executable)

`tomllib` was added in Python 3.11. This block parses a `pyproject.toml` fragment and reports the declared
runtime dependencies in sorted order.

```python
import tomllib

PYPROJECT = """
[project]
name = "cohort-core"
version = "1.4.2"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.2,<3",
  "pyarrow>=15,<18",
  "duckdb>=1.0,<2",
]

[dependency-groups]
dev = ["pytest>=8", "mypy>=1.10", "ruff>=0.5"]
"""

data = tomllib.loads(PYPROJECT)
project = data["project"]
print("name", project["name"])
print("version", project["version"])
print("requires-python", project["requires-python"])
print("runtime_dependencies")
for dep in sorted(project["dependencies"]):
    print(" ", dep)
print("dev_dependency_count", len(data["dependency-groups"]["dev"]))
```

Expected output:

```text
name cohort-core
version 1.4.2
requires-python >=3.11
runtime_dependencies
  duckdb>=1.0,<2
  pandas>=2.2,<3
  pyarrow>=15,<18
dev_dependency_count 3
```

Note what the audit does not do: it does not prove that the constraints resolve, that the packages exist,
or that their licenses are acceptable. It reports what the project declares. Resolution and license
scanning are separate CI steps.

### 7.3 A minimal `pyproject.toml`

The following is copy-ready for a project using a PEP 517 backend. It is configuration, not executable
code.

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "cohort-core"
version = "1.4.2"
description = "Unit conversion and eligibility primitives for clinical cohorts"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.2,<3",
  "pyarrow>=15,<18",
]

[project.optional-dependencies]
postgres = ["psycopg[binary]>=3.2,<4"]

[dependency-groups]
dev = ["pytest>=8", "mypy>=1.10", "ruff>=0.5"]

[project.scripts]
cohort-validate = "cohort_core.cli:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
```

`optional-dependencies` are installed by the consumer only when requested; `dependency-groups` are for
development and are not installed with the package. The distinction matters for the production image,
which should install neither the development group nor the optional database driver unless it uses it.

## 8. Data, standards, and regulatory context

### 8.1 Packaging standards

| Standard | What it defines | Status | Consequence |
|---|---|---|---|
| PEP 517 | Build-system interface | Final | Build frontends can call any backend |
| PEP 518 | `pyproject.toml` build requirements | Final | Build dependencies are declared, not implicit |
| PEP 621 | `[project]` metadata table | Final | One metadata format across backends |
| PEP 427 | Wheel format | Final | Binary distributions are installable without building |
| PEP 440 | Version identification and specifiers | Final | Version ordering and range semantics |
| PEP 751 | `pylock.toml` lockfile | Final, March 2025 | Tool-agnostic lock export |
| PEP 484 / 561 | Type hints and `py.typed` | Final | Checkers and consumers can rely on annotations |
| SemVer 2.0.0 | Version semantics | De facto | Major/minor/patch meaning |

PEP 751 is recent enough that support varies. uv can export `pylock.toml` and install from it, while
continuing to use its own `uv.lock` for the project interface. A team that standardizes on `pylock.toml`
gains portability across installers; a team that uses `uv.lock` gains features that the standard format
cannot express. Either choice must be recorded, because the lockfile is part of the reproducibility claim.

### 8.2 Supply-chain standards

An SBOM in SPDX or CycloneDX format lists the components in an artifact. It supports vulnerability
scanning and license review; it does not prove that the components are safe. SLSA (Supply-chain Levels
for Software Artifacts) defines levels of build provenance, from a documented build to a hardened,
isolated build with signed provenance. For a hospital pipeline, the practical minimum is a lockfile with
hashes, a pinned base image, and a dependency scanner in CI. Claiming SLSA compliance without the
provenance attestations is a marketing statement.

### 8.3 Clinical and regulatory constraints

- **PHI in logs.** A Python log line that includes a patient identifier is a disclosure. The HIPAA Security
  Rule requires audit controls, but audit logs should record access metadata, not record contents.
- **Reproducibility for audit.** Under 21 CFR Part 11, electronic records in regulated submissions need
  audit trails and retained versions. The code version, environment lock, and input snapshot are part of
  that record.
- **Model cards and intended use.** A package that computes a clinical score may be a medical device
  function depending on its use. The software engineering practices here do not establish regulatory
  status; the intended-use statement does.
- **Licensing.** Clinical deployments often prohibit certain open-source licenses. The dependency audit
  should include license identification, and the result should be reviewed by counsel, not by the
  engineering team alone.

### 8.4 What the standards do not cover

No packaging standard specifies:

- Which dependencies are clinically validated.
- Whether a numerical result is correct.
- How to test a cohort definition.
- How to log without PHI.
- When a change requires re-validation.

Those are engineering and regulatory decisions. A project can be fully PEP-compliant and still compute
the wrong dose.

## 9. Industry practice and case evidence

### 9.1 The shift to locked, tool-agnostic environments

The Python packaging ecosystem has moved toward declared metadata (PEP 621), explicit build backends
(PEP 517), and standardized lockfiles (PEP 751). uv's documentation describes `uv.lock` as the project
lock and `pylock.toml` as an export target, which reflects a practical split: application teams want a
fast resolver and rich features, while auditors want a standard, readable file. The direction of travel
is toward recording the full resolved graph rather than a list of top-level packages.

### 9.2 Python release cadence

CPython releases a new minor version annually in October. Python 3.14 was released on 7 October 2025,
with free-threaded builds officially supported (PEP 779) and a single-threaded penalty of roughly 5-10%
per the release documentation. A biomedical pipeline does not need the newest interpreter; it needs a
version with wheels for its numerical stack and a support window that matches the product's maintenance
plan. Adopting a new minor version in month one is a choice to absorb ecosystem breakage.

### 9.3 Reproducibility in published research

The reproducibility crisis in biomedical research is partly an environment problem. Papers that depend on
a specific version of a statistical package, a specific compiler, or an unstated random seed cannot be
reproduced from the text. Journals and funders increasingly require code and data availability statements,
but availability is not reproducibility: a repository without a lockfile and an input hash may still fail
to reproduce. The practices in this topic are the minimum for a claim of computational reproducibility.

### 9.4 What is inference

- That lockfiles reduce incidents. They eliminate a class of resolution differences; they do not prevent
  logic errors or data changes.
- That type checking reduces production failures. It catches interface errors, not semantic ones.
- That container digests improve auditability. They identify the bytes; they do not prove the bytes are
  correct.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unpinned transitive dependency | Build works today, fails next month | Resolver selects a new version | Silent behavior change | Lockfile with hashes |
| Works on one machine | New hire cannot install | Manual global installs | Irreproducible result | Clean venv in CI |
| Config typo ignored | A parameter has no effect | Unknown fields silently dropped | Wrong cohort with no error | Strict validation |
| Secret in repository | Token appears in a config file | Convenience | Credential exposure | Secret manager; history scan |
| Clock in core logic | Historical re-run changes | `date.today()` inside function | Non-reproducible eligibility | Pass `as_of` explicitly |
| Float threshold comparison | Boundary cases flip | Binary representation | Systematic misclassification | `Decimal` and quantize |
| Test only happy path | Coverage high, bugs at boundaries | Example-based tests | Boundary defects ship | Boundary and property tests |
| Logging PHI | Patient IDs in log lines | Debug convenience | Reportable disclosure | Structured logging with allowlist |
| Unstructured logs | Cannot filter by run | `print` statements | Slow incident response | JSON logs with correlation ID |
| Rebuilding at deploy | Image differs from tested artifact | Build on deploy | Untested bytes in production | Build once, promote digest |
| Version number lies | Patch release breaks consumers | No SemVer discipline | Upgrade incidents | Compatibility policy |
| Dev tools in production | Large image, scanner noise | Same environment everywhere | Attack surface | Separate dependency groups |

Three red flags predict a reproducibility failure before it happens.

**No lockfile in the repository.** If the project declares ranges and nothing else, the environment is a
moment in time, not a specification.

**A test that sleeps or calls the network.** Tests that depend on external services are flaky; flaky
tests get disabled; disabled tests stop protecting anything. Use fixtures and fakes at the boundary.

**A configuration file with a real hostname.** It means the code knows about one site. A second site will
fork the repository or add an `if` statement, and the fork will drift.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The lockfile covers all target platforms | Reproducibility across OS and architecture | A platform-specific wheel is missing | Resolve on each platform in CI |
| The package index is trustworthy | Installation without vendoring | A compromised or typosquatted package | Hashes, private index, review |
| Clinical logic is deterministic | Re-runs produce the same result | Random seeds, clock, concurrency | Run twice and compare output hash |
| Fixtures represent production shape | Integration tests are meaningful | Fixture drift after a schema change | Refresh fixtures from a de-identified sample |
| Type hints are accurate | Checkers catch interface errors | `Any` and `# type: ignore` accumulate | Strict mode; review suppressions |
| Secrets are available at runtime | No credentials in code | Secret manager outage | Startup check with a clear error |
| The artifact is immutable | Audit can identify bytes | Rebuild on deploy | Digest pinning; registry immutability |
| Tests run fast enough to be used | Developers run them locally | Slow integration suite | Keep unit suite under 30 seconds |
| The team can maintain the package | Long-term reproducibility | One maintainer leaves | Documentation, ownership, bus factor review |

The largest limit is organizational. A perfect lockfile does not help if the team cannot rebuild the
artifact because the build service was decommissioned, or if the only person who understands the
conversion factor has left. Reproducibility is a property of an institution, not only of a repository.

A second limit is that reproducibility is not validity. A pipeline can reproduce the same wrong answer
indefinitely. Tests encode the team's understanding; they do not establish that the understanding is
clinically correct.

## 12. Exercises

### Exercise 1 — Project skeleton (application, 30 minutes)

Deliverable: a `src`-layout project with `pyproject.toml`, one module, one test, and a README that states
the supported Python versions.

Acceptance criteria: `python -m build` produces a wheel and sdist; installing the wheel in a clean venv
makes the module importable; the README names the build command.

### Exercise 2 — Lock and reproduce (application, 45 minutes)

Deliverable: a lockfile for the project and a script that creates a fresh environment, installs from the
lock, and runs the tests.

Acceptance criteria: the script succeeds twice from a clean directory; the lockfile records hashes; the
output includes the resolved versions.

### Exercise 3 — Configuration precedence (design, 40 minutes)

Deliverable: a configuration module with defaults, a base file, a site file, environment variables, and
CLI flags, plus a precedence table and three tests.

Acceptance criteria: an unknown key raises an error; a missing required secret raises at startup; each
precedence transition has a test.

### Exercise 4 — Pure eligibility function (application, 45 minutes)

Deliverable: `is_eligible(birth_date, index_date, min_age, observation_start)` as a pure function, with
tests at the boundary.

Acceptance criteria: no clock access; correct on the day before, the day of, and the day after the
eighteenth birthday; leap-day birth date handled explicitly.

### Exercise 5 — Test pyramid (design, 60 minutes)

Deliverable: a test plan for a pipeline with at least 20 unit tests, 5 integration tests, and 2 end-to-end
tests, with a rationale for the distribution.

Acceptance criteria: each level names what it catches that the others do not; fixtures are synthetic or
de-identified; the unit suite runs in under 30 seconds.

### Exercise 6 — PHI-safe logging (application, 30 minutes)

Deliverable: a logging configuration that emits JSON lines with run ID, stage, row count, and duration,
and a review checklist that prevents identifiers from being logged.

Acceptance criteria: no free-text patient fields; correlation ID present on every line; a test asserts
that a sample log record contains none of a list of forbidden keys.

### Exercise 7 — Release (application, 40 minutes)

Deliverable: a version bump, a changelog entry, a tag, and a built artifact with a recorded hash.

Acceptance criteria: the version follows SemVer; the changelog states compatibility; the artifact hash
matches the one in the release notes.

### Exercise 8 — Diagnose the incident (critique, 45 minutes)

A pipeline produces a cohort 4.2% smaller on a new machine. Deliverable: a 400-word diagnosis with a
ranked list of causes and the commands or queries used to test each.

Acceptance criteria: distinguishes environment, configuration, and data causes; names a lockfile or hash
comparison for each; proposes a prevention measure that is enforceable in CI.

## 13. Self-check answers

**1. Why is the script not the unit of reproducibility?**

Because the script's behavior depends on the interpreter, the resolved dependencies, the configuration,
and the input data. Reproducing the source without those produces a different result. The unit is the
closure of all four.

**2. What is the difference between a wheel and an sdist?**

A wheel is a built distribution that installs without running a build. An sdist is source that must be
built, which requires a compiler and build dependencies. Wheels are faster and more deterministic;
sdists are more portable across platforms but less reproducible.

**3. Why is a version range not a lock?**

A range permits multiple versions. The resolver may choose a different one tomorrow, and transitive
dependencies are not constrained by the range at all. A lock records the exact resolved graph and hashes.

**4. What belongs in the pure core?**

Clinical logic: eligibility, conversion, classification, deduplication rules, and scoring. It takes all
inputs as parameters and returns values. I/O, clocks, and network calls belong in the shell.

**5. Why use `Decimal` for thresholds?**

Because binary floating point cannot represent most decimal fractions exactly. A value intended to be
106.00 may be stored as 105.99999999999999 and compare as below the threshold. `Decimal` with an explicit
rounding rule makes the comparison match the stated precision.

**6. What is the difference between validation and testing?**

Validation rejects invalid input at runtime. Testing proves behavior during development. A pipeline needs
both; validation protects against bad configuration, testing protects against bad logic.

**7. Why record a configuration fingerprint?**

Because it identifies the exact parameter set used for a run. When two runs differ, the fingerprint shows
whether the configuration changed. It is a cheap, deterministic identifier that can be logged and
compared.

**8. What makes a test an integration test?**

It exercises a real boundary: a database, a file system, an API, or a serialization format. Unit tests use
fakes or pure functions. Integration tests are slower and fewer because their value is in catching wiring
errors.

**9. Why pin the base image by digest?**

Because a tag can be reassigned. The digest identifies the exact bytes. Two builds from the same digest
and the same lockfile are more likely to be identical.

**10. What does an SBOM prove?**

It lists the components in an artifact. It supports vulnerability and license review. It does not prove
that the components are free of vulnerabilities or that the artifact is safe.

## 14. Interview questions

**1. "How do you make a Python pipeline reproducible?"**

Strong: lockfile with hashes, pinned interpreter, immutable artifact, configuration fingerprint, input
snapshot hash, and a clean-environment test in CI. Weak: "we use a virtualenv," which isolates but does not
pin.

**2. "A pipeline works locally but not in production. What do you check?"**

Strong: interpreter version, lockfile, OS libraries, environment variables, working directory, file
permissions, and network access, in that order. Weak: "it is probably a dependency issue," with no method.

**3. "Why separate runtime and development dependencies?"**

Strong: smaller production image, smaller attack surface, faster installs, and a clearer dependency
review. Weak: "it is convention."

**4. "When would you not pin a dependency?"**

Strong: in a library, where ranges allow consumers to resolve, combined with a CI matrix at the lowest
and highest supported versions. Weak: "never pin, it causes conflicts."

**5. "How do you test clinical logic?"**

Strong: pure functions, boundary tables, property tests for invariants, and integration tests on
de-identified fixtures. Weak: "we compare the output to the old spreadsheet."

**6. "How do you prevent PHI in logs?"**

Strong: structured logging with an allowlist of fields, no record contents, tests that scan log records,
and a review checklist. Weak: "we do not log patient data," with no enforcement.

**7. "What is the difference between reproducibility and repeatability?"**

Strong: reproducibility is another person getting the same result from the same inputs and artifacts;
repeatability is the same team getting the same result. Both require determinism, but reproducibility also
requires portable artifacts and documentation.

**8. "How do you handle a dependency with a critical vulnerability?"**

Strong: identify exposure, check whether the vulnerable path is reachable, patch or pin, test, and record
the decision. Weak: "upgrade everything," without a compatibility assessment.

**9. "What belongs in a code review for a clinical pipeline?"**

Strong: logic and boundary cases, configuration and secrets, logging, tests, error handling, and
reproducibility metadata. Weak: style only.

**10. "How do you version a change that alters cohort counts?"**

Strong: a major version if the public behavior changes, a documented rationale, a comparison of old and
new counts with attrition, and a migration note. Weak: a patch release because "it is a small change."

## 15. Key terms

- **Artifact** — immutable build output.
- **Build backend** — tool that produces a wheel or sdist.
- **CI** — continuous integration.
- **Configuration precedence** — order of config source override.
- **Correlation ID** — identifier linking log lines to a run.
- **Coverage** — fraction of lines executed by tests.
- **Dataclass** — class with generated methods.
- **Dependency group** — development-only dependency set.
- **Deterministic** — same inputs, same outputs.
- **Distribution** — installable package artifact.
- **Entry point** — declared console script.
- **Fixture** — reusable test input or setup.
- **Frozen dataclass** — immutable dataclass.
- **Idempotent** — re-run produces the same state.
- **Image digest** — content hash of a container image.
- **Lockfile** — exact resolved dependency set with hashes.
- **PEP 517 / 518 / 621 / 751** — build, metadata, and lock standards.
- **Pin** — exact version constraint.
- **Protocol** — structural typing interface.
- **Pure function** — no side effects, deterministic.
- **pyproject.toml** — project metadata and tool configuration.
- **Range** — constraint permitting multiple versions.
- **Reproducible build** — identical bytes from identical inputs.
- **SBOM** — software bill of materials.
- **SemVer** — major.minor.patch versioning.
- **Sdist** — source distribution.
- **Secret** — credential that must not be in source control.
- **src layout** — importable code under `src/`.
- **Structured logging** — key-value log records.
- **Transitive dependency** — dependency of a dependency.
- **Validation** — runtime rejection of invalid input.
- **Virtual environment** — isolated interpreter environment.
- **Wheel** — built binary distribution.

## 16. Sources

All URLs accessed 25 October 2026.

1. Python Software Foundation. "Python 3.14.0." https://www.python.org/downloads/release/python-3140/
2. Python Software Foundation. "What's New In Python 3.14." https://docs.python.org/3/whatsnew/3.14.html
3. PEP 517. "A build-system independent format for source trees." https://peps.python.org/pep-0517/
4. PEP 518. "Specifying minimum build system requirements." https://peps.python.org/pep-0518/
5. PEP 621. "Storing project metadata in pyproject.toml." https://peps.python.org/pep-0621/
6. PEP 751. "A file format to record Python dependencies for installation reproducibility."
   https://peps.python.org/pep-0751/
7. PEP 440. "Version Identification and Dependency Specification." https://peps.python.org/pep-0440/
8. PEP 484. "Type Hints." https://peps.python.org/pep-0484/
9. uv. "Structure and files." https://docs.astral.sh/uv/concepts/projects/layout/
10. uv. "Exporting a lockfile." https://docs.astral.sh/uv/concepts/projects/export/
11. Semantic Versioning 2.0.0. https://semver.org/
12. SPDX. https://spdx.dev/
13. CycloneDX. https://cyclonedx.org/
14. SLSA. https://slsa.dev/
15. pytest documentation. https://docs.pytest.org/
16. mypy documentation. https://mypy.readthedocs.io/
17. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
18. HHS. "HIPAA Security Rule." https://www.hhs.gov/hipaa/for-professionals/security/index.html

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_sql-for-clinical-data.md`](02_sql-for-clinical-data.md)
- Next topic: [`04_orchestration.md`](04_orchestration.md)
- Related sibling: [`10_testing-and-observability.md`](10_testing-and-observability.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../course/03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.1 requires a script that runs from a clean checkout.
