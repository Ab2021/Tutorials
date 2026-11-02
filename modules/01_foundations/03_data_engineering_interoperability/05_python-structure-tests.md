> Module 03 — Data Engineering and Interoperability
> Topic 05 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01, 01A.01

# Python Project Structure, Types, and Tests

---

## 1. Why this topic exists

A supply chain analytics repository has two audiences. The first is the planning run that must
happen on Thursday. The second is the analyst who inherits the repository eighteen months later,
after the author has left, and is asked why the safety stock for one item tripled in March.

Most teaching about analytics code optimises for the first audience: a script that loads a file,
computes a number, and prints it. That script is a demonstration, not an asset. A demonstration
carries its meaning in the author's head; an asset carries it in the repository.

Four artifacts carry that meaning, and this topic is about all four:

| Artifact | The question it answers for the stranger | What its absence produces |
|---|---|---|
| Layout | Where does the program start, and what may import what? | A script that runs only from one directory |
| Environment | Which interpreter and which exact package versions produced this number? | A result nobody can reproduce |
| Types | What shape of value does this function accept, and what does it return? | A unit or null-handling bug that the tests never see |
| Tests | What must remain true after I change this rule? | A change that silently breaks a downstream plan |

The four are not independent. Layout determines what can be imported without side effects, which
determines what can be tested without a database. The environment determines whether a passing
test means anything on another machine. Types determine which tests are worth writing, because a
checker removes a whole class of test that would only restate the signature.

**Why this belongs in a data engineering module.** Topic 03.01 argued that a data platform needs
layers, contracts, and prohibitions at each boundary. That argument is about the data. This topic
is the same argument about the code that produces the data. A conformed layer built by an
untested script is a well-organised route to a wrong number. The boundary contracts in 03.01 are
enforced, in the end, by functions that either behave as promised or do not.

The connection to 01A.01 is narrower and sharper. Units and rounding are where analytics code
most often fails silently: a conversion that is applied twice, a case-pack constant that no
longer matches the item master, a ceiling applied at the wrong level of a cascade. These defects
produce plausible numbers. They do not raise exceptions. A test strategy that does not target
them is a test strategy for a different kind of program.

Five symptoms recur in repositories that were written for the first audience only:

1. **The import that only works from one folder.** The package sits at the repository root, the
   current working directory is on the import path, and the tests import the source tree rather
   than the installed package. Packaging defects — a missing data file, a module excluded from the
   wheel — stay invisible until a user installs the distribution.
2. **The environment that cannot be rebuilt.** Dependencies are listed as `pandas` and `numpy`,
   with no versions. Six months later a minor release changes a default, and the plan moves by
   two percent with no code change.
3. **The credential in the script.** A database password sits in a string literal, the author
   leaves, the password is rotated, and the pipeline stops. The repository also now contains a
   secret in its history, which no later commit removes.
4. **The core that cannot be called without a connection.** The rounding rule is entangled with
   a query, so testing it requires a warehouse. Nobody tests it, because the test would take
   forty seconds and a credential.
5. **The suite that exists and is not run.** Four hundred tests, twelve minutes, two flaky. The
   team runs them before releases, then before releases only when there is time, then not at all.

Each symptom is cheap to prevent and expensive to discover. The rest of this topic is the
prevention, ordered by the cost of getting it wrong.

**Scope.** This is not a general software engineering course. It selects the practices that pay
for themselves in a team of two to fifteen people building planning and analytics pipelines. It
does not cover publishing a library to a public index, container orchestration, or web
frameworks. Topic 06 applies the same discipline to SQL models through dbt; Topic 11 applies it to
data quality, quarantine, and lineage.

## 2. Learning outcomes

By the end of this topic you can:

1. **Lay out** a Python analytics project using a `src` layout, a single metadata file, and an
   explicit separation between pure logic and input/output, and **explain** what each choice makes
   testable that a flat script does not.
2. **Separate** configuration from secrets, **implement** a typed settings object that fails at
   startup with a message naming the missing variable, and **state** the two failure modes this
   prevents.
3. **Annotate** a module boundary with type hints, **choose** a strictness level for new and
   legacy code, and **state** three things the type checker cannot prove.
4. **Produce** a locked, hash-verified environment from abstract dependencies, and **distinguish**
   what the lock guarantees from what it does not.
5. **Select** the correct test level — unit, property, contract, or golden file — for a named
   supply chain defect, and **justify** the choice on cost and localisation grounds.
6. **Write** a property-based test for a rounding or conversion invariant, with a stated input
   domain, a stated invariant, and a shrinking story.
7. **Calculate** the annual cost of a test suite and use the result to decide what to delete,
   parallelise, or move out of the commit path.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a pyramid whose levels are not defined by "size" but by **what kind of lie the
test can catch**. At the bottom, a test can only catch a mistake in a single expression. At the
top, a test can catch a mistake in the order in which five correct components were assembled.

```text
                          /\
                         /  \        GOLDEN FILE        a handful
                        /    \       frozen week -> full plan
                       /------\      catches ordering, joins, integration
                      /        \
                     / CONTRACT \    SCHEMA + SEMANTICS      tens
                    /            \   keys unique, units tagged, nulls bounded
                   /--------------\
                  /                \
                 /    PROPERTY      \  INVARIANTS             tens
                /                    \ round-trip, residue bound, conservation
               /----------------------\
              /          UNIT          \  PURE FUNCTIONS      hundreds
             /                          \ ceil_div, bucketing, allocation order
            /----------------------------\
           /           STATIC             \  TYPES + LINT       whole repo
          /                                \ annotations, imports, formatting
         /----------------------------------\
```

Read upward: each level costs more to run and catches a broader class of defect. Read downward:
each level is cheaper, runs more often, and localises a failure to fewer lines.

The supply chain examples at each level are the point of the diagram. A golden-file test at the
top might compare a full weekly plan for three distribution centres against a frozen expected
plan. A contract test asserts that the item master has one row per item and that every quantity
carries a unit code. A property test asserts that rounding up to a case never supplies less than
the demand and never wastes a whole case. A unit test asserts that the week bucket for a
timestamp at 23:30 in a named time zone falls on the correct local day. The static level asserts
that a function annotated to return `Decimal` does not return `None`.

### 3.1 Pure core, imperative shell

The layout that makes the pyramid possible is a separation of two kinds of code:

```text
   +-------------------------------------------------------------+
   |  SHELL: reads config, opens connections, reads files,       |
   |         writes outputs, logs, retries, exits                |
   |                                                             |
   |     parse_args() -> load_config() -> run(cfg) -> write()    |
   |                                                             |
   |     +-------------------------------------------------+     |
   |     |  CORE: pure functions                           |     |
   |     |  no network, no clock, no random, no globals    |     |
   |     |                                                 |     |
   |     |  convert(qty, from_uom, to_uom) -> Decimal      |     |
   |     |  round_up_to_case(eaches, pack) -> int          |     |
   |     |  allocate(demand, supply, rule) -> dict         |     |
   |     +-------------------------------------------------+     |
   +-------------------------------------------------------------+
```

A **pure function** returns the same output for the same input and changes nothing outside
itself. That property is what makes the bottom four levels of the pyramid cheap: no fixture
needed, no database, no clock, no cleanup, and failures reproduce exactly.

The shell is allowed to be ugly, because the shell is not where the supply chain logic lives.
The shell reads the configuration, opens the connection, calls the core, and writes the result.
If the shell is hard to test, that is acceptable. If the core is hard to test, the design is
wrong.

### 3.2 Five properties worth memorising

**Property 1 — A test suite is a claim about change, not a proof of correctness.** A passing
suite says that the behaviours someone thought to encode still hold. It says nothing about the
behaviour nobody encoded. This is why coverage is a weak target and why the question "what would
this suite fail to catch?" is more useful than "what is the coverage?"

**Property 2 — Put the test at the seam where the defect becomes invisible.** A unit error is
visible in a unit test. A schema drift is invisible until a downstream join silently drops rows,
so it needs a contract test at the boundary. An ordering error between two correct steps is
invisible until the assembled plan is compared with a known-good plan. The level follows the
defect, not the fashion.

**Property 3 — Cost per run determines whether the test runs.** A test that takes four minutes
will be skipped on the commit that most needs it. The binding constraint is not the total suite
time but the time between "I think I am done" and "I know whether I broke something."

**Property 4 — A failing test that is usually wrong is worse than no test.** A flaky test teaches
the team that red means "rerun". Once that lesson is learned, every genuine failure is also
rerun. One unreliable test can devalue the entire suite, because the team's response is a
property of the suite, not of the test.

**Property 5 — Layout decides what can be tested at all.** If the rounding rule can only be
reached through a database query, no amount of testing discipline will produce a fast unit test
for it. Structure is upstream of culture.

> **Note:** The pyramid is a budget, not a law. Its proportions describe where effort tends to
> pay off, not a required ratio. Data code legitimately has more schema tests than a web
> application and fewer unit tests than a compiler. Section 9 gives the critiques, including the
> argument that machine-learning systems have emergent behaviour with no deterministic unit to
> test.

The rest of this topic takes each artifact in turn, then shows how they combine in one repository
and one incident.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Repository layout | The arrangement of source, tests, configuration, and metadata in a version-controlled tree | Treated as personal taste | Directory path |
| Package | An importable Python namespace containing modules and subpackages | Confused with the project or the distribution | Dotted import path |
| Distribution | A versioned archive that a package index can install, built from a project | Used interchangeably with package | Name plus version |
| `src` layout | A layout in which importable code lives under `src/<package>/`, outside the repository root's import path | Treated as a preference rather than a testability decision | Directory path |
| Flat layout | A layout in which the package directory sits at the repository root | Confused with "no layout" | Directory path |
| Editable install | An installation that points the interpreter at the working tree so source edits take effect without reinstalling | Confused with running from the repository root | Install mode plus path |
| Wheel | A built binary distribution installable without a build step | Used to mean any packaged artifact | `.whl` file plus platform tag |
| `pyproject.toml` | The project metadata and tool configuration file read by modern build backends and tools | Treated as a replacement for a lockfile | File path plus schema version |
| Lockfile | A resolved, exact record of every package version installed, including transitive dependencies | Confused with a requirements file that lists only direct dependencies | File plus resolution hash |
| Dependency specifier | A declared constraint on an acceptable version range | Written as an exact version and called "pinned" | Package name plus version constraint |
| Pin | An exact version equality constraint | Assumed to cover transitive dependencies | `==` version |
| Hash pinning | Recording a cryptographic digest per artifact and refusing to install a mismatch | Confused with version pinning | SHA-256 digest |
| Virtual environment | An isolated interpreter and package directory attached to a project | Assumed to make the build reproducible by itself | Directory path plus interpreter version |
| Interpreter | The Python executable and its version, including patch level | Left implicit in documentation | Major.minor.patch |
| Configuration | Values that vary between deployments but are not secret | Mixed with secrets or hard-coded | Environment variable or settings field |
| Secret | A credential whose disclosure grants access | Stored in the repository, a notebook, or a committed `.env` file | Secret manager reference |
| Environment variable | A process-level key–value pair used to inject configuration | Used as a general-purpose global store | Uppercase key plus string value |
| Type hint | A machine-readable annotation of a value's expected type | Treated as runtime enforcement | Annotation syntax |
| Stub file | A `.pyi` file declaring types for code implemented elsewhere | Confused with documentation | File path |
| `py.typed` | A marker file declaring that a package ships its own type information | Assumed to make a package fully typed | Marker file |
| Gradual typing | Adding annotations to part of a codebase while untyped code remains unchecked | Treated as all-or-nothing | Proportion of annotated functions |
| `Any` | A type that disables checking for the value it annotates | Used to silence an error without a reason | Type expression |
| `Unknown` | A checker-specific type for a value whose type was never declared | Confused with `Any` | Type expression |
| `Protocol` | A structural interface: any object with the required methods satisfies it | Confused with a base class | Class definition |
| Strict mode | A checker configuration that rejects untyped definitions and implicit `Any` | Assumed identical across checkers | Checker plus rule set |
| Pure function | A function whose output depends only on its arguments and which has no side effects | Used to mean "simple function" | Function signature |
| Imperative shell | The thin outer layer that performs input/output and calls pure functions | Treated as the place for business rules | Module boundary |
| Fixture | A setup function whose return value is injected into a test | Used to mean any test data file | Function plus scope |
| `conftest.py` | A file whose fixtures are shared across a directory tree without imports | Used as a dumping ground for helpers | File path |
| Parametrisation | Running one test body against several input cases, each reported separately | Confused with a loop inside a test | Decorator plus case identifiers |
| Test pyramid | A heuristic for distributing tests across levels of granularity | Treated as a required ratio | Level plus count |
| Unit test | A test of one function or small unit, with no input/output | Used to mean "any fast test" | Test identifier |
| Contract test | A test of the agreement at a boundary: schema, keys, units, nullability, grain | Confused with a full schema validation of production data | Contract version |
| Property-based test | A test that asserts an invariant over a generated input space and shrinks failures | Confused with random testing without assertions | Generator domain plus invariant |
| Golden-file test | A test that compares a full output against a reviewed, versioned expected file | Auto-updated whenever it fails | Golden file plus review record |
| Flaky test | A test that passes and fails on unchanged code | Tolerated as normal | Failure rate per 1,000 runs |
| Coverage | The proportion of statements or branches executed by the suite | Used as a quality target | Percentage, with tool and scope stated |
| Determinism | The property that identical inputs and code produce identical outputs | Assumed when the code contains no obvious randomness | Seed, clock, and ordering policy |

Four of these distinctions do the most work.

**Package versus distribution.** The package is what you import (`planning.units`). The
distribution is what you install (`ellsworth-planning==2.4.0`). One distribution can contain
several packages, and one package can be split across several distributions. Confusing them
produces two errors: believing that installing the project makes a module importable under a name
that was never declared, and believing that a version bump on the distribution guarantees a
version bump on every package inside it.

**`src` layout versus flat layout.** In a flat layout, the package directory is at the repository
root, and Python places the current working directory first on the import path. Tests therefore
import the working tree, not the installed distribution. A file missing from the build, a module
excluded by package discovery, or a dependency that only exists in the development environment
will all pass locally and fail for the user. In a `src` layout the package is not on the import
path by default, so tests must run against an installed or editable-installed copy. The cost is
one extra setup step; the benefit is that the test result describes the artifact rather than the
folder.

**Configuration versus secret.** Configuration is a value that differs between deployments and
is safe to read aloud: a warehouse host name, a batch size, a cut-off time. A secret is a value
whose disclosure grants access. Both belong outside the repository, but only the secret requires
redaction in logs, rotation, and access control. Treating them as one category produces the
common failure in which a settings object is printed at startup and leaks a password into a log
aggregator. Treating them as entirely separate produces the opposite failure, in which
configuration drifts between environments with no record.

**Type hint versus docstring.** A docstring states an intention in prose and is checked by a
reader. A type hint states an intention in a form a machine checks on every run of the checker.
Neither replaces the other: the hint cannot express "positive", "in base units", or "effective
from the first day of the fiscal week". The correct division is that types carry shape and
nullability, and prose carries domain constraints that no checker can see.

## 5. The mechanics, step by step

The procedure has fifteen steps in six phases. The order is not arbitrary: layout determines
importability, importability determines testability, and the lock determines whether a test
result means anything on a different machine.

### Phase A — Decide what the importable unit is

**Step 1. Name the distribution and the package separately.**

- Decision: a distribution name for installation (`ellsworth-planning`) and one importable
  package name (`planning`). Write both down before creating directories.
- Why: the two names serve different consumers. The distribution name appears in a lockfile and
  an incident report; the package name appears in every import.
- If skipped: the project is installed under one name and imported under another, and the
  mismatch is discovered by a new team member rather than by the build.

**Step 2. Adopt a `src` layout.**

- Decision: `src/<package>/` for importable code, `tests/` beside `src/`, both outside the
  repository root's import path.
- Why: it forces tests to run against the installed artifact, which is the only artifact that
  exists in production.
- If skipped: packaging defects and missing dependencies remain invisible until deployment.
- Evidence it worked: a clean checkout plus an editable install plus the test command succeeds
  with no `PYTHONPATH` manipulation.

**Step 3. Separate the pure core from the imperative shell.**

- Decision: modules that compute (`units.py`, `demand.py`, `allocate.py`) import no input/output
  library and accept no connection object. Modules that read and write (`io/reader.py`,
  `io/writer.py`) contain no business rules.
- Why: purity is what makes the cheap levels of the pyramid possible.
- If skipped: every test needs a fixture, a credential, or a mock, and the suite becomes slow and
  brittle at the same time.

**Step 4. Name modules by role, not by layer of convenience.**

- Decision: forbid `utils.py`, `helpers.py`, and `common.py`. A function that has no better home
  is usually a sign that a domain concept has not been named.
- Why: a `utils.py` becomes a gravity well. It acquires imports from every part of the system,
  then creates import cycles, then gets excluded from tests because importing it opens a
  connection.
- If skipped: the dependency graph becomes unreadable and the core stops being importable in
  isolation.

### Phase B — Make the environment reproducible

**Step 5. Pin the interpreter, not only the packages.**

- Decision: declare the minimum and maximum supported Python version in project metadata, and
  record the exact development version in a version file read by the environment manager.
- Why: package resolution depends on the interpreter version, and syntax and standard-library
  behaviour change across minor releases.
- If skipped: the lockfile is valid only for whichever interpreter happened to resolve it.

**Step 6. Separate abstract dependencies from the resolved lock.**

- Decision: declare direct dependencies with meaningful version ranges in project metadata.
  Generate a lockfile that records every direct and transitive package at an exact version.
- Why: ranges express compatibility intent for humans and libraries; the lock expresses the exact
  set that was tested. An application needs both. A library should not ship a lock, because its
  consumers must resolve against their own environment.
- If skipped: either the project is over-constrained and cannot accept security patches, or it is
  under-constrained and the build changes without a commit.

**Step 7. Install from the lock, with hashes, in every automated environment.**

- Decision: continuous integration installs with the lock frozen and hash verification enabled.
  A lockfile update is a reviewed change, not a side effect of a routine build.
- Why: a version pin says which version; a hash says which bytes. Index substitution and
  compromised artifacts are both defeated by the hash, and neither is defeated by the pin.
- If skipped: two machines with the same lockfile can install different code.
- Evidence it worked: a build on a clean runner, with no cache, produces the same test results as
  the developer machine.

### Phase C — Separate configuration from secrets

**Step 8. Put configuration in the environment and secrets in a secret store.**

- Decision: deployment-varying values arrive as environment variables; credentials arrive from
  the platform's secret manager or an injected file outside the repository. Neither is a literal
  in code, and neither is a committed `.env` file.
- Why: the test for configuration is whether the repository could be made public without leaking
  anything. A committed `.env` fails that test even when it is listed in `.gitignore`, because
  the history retains it.
- If skipped: rotation breaks the pipeline, and the secret is in the history permanently.

**Step 9. Parse the environment once, into a typed settings object, at startup.**

- Decision: one module constructs a settings object, validates required fields, applies defaults,
  and raises an error that names the missing variable. Nothing else in the code reads the
  environment directly.
- Why: it converts a class of runtime failure into a startup failure with an actionable message,
  and it gives tests a single object to construct.
- If skipped: a missing variable surfaces four hours into a run, or worse, is silently replaced by
  a default that points at the wrong warehouse.
- Logging rule: log the resolved settings with secret fields redacted, once, at startup.

### Phase D — Make types carry the boundary

**Step 10. Annotate boundaries before internals.**

- Decision: annotate every public function, every dataclass field, and every input/output port.
  Internal helpers can wait.
- Why: boundary annotations are the ones a stranger reads and the ones that catch the errors that
  cross modules. Annotating internals first spends effort where the checker already has context.
- If skipped: the checker has almost nothing to work with, because many checkers skip unannotated
  functions, and their return values are treated as untyped.

**Step 11. Ratchet strictness instead of demanding it everywhere.**

- Decision: enable strict settings for new modules and for the core; record existing violations
  in a per-module exemption list; count suppressions and treat the count as a budget that may not
  grow.
- Why: a codebase-wide strict flag on legacy code produces thousands of errors, which produces a
  disabled flag. A ratchet produces a monotonic improvement that survives staff turnover.
- If skipped: the team oscillates between "types are mandatory" and "types are off".
- Suppression rule: every suppression names an error code and a reason, on the same line.

### Phase E — Build the suite at the right levels

**Step 12. Write unit tests for pure functions, starting with the ones that encode money.**

- Decision: unit conversion, rounding, allocation order, week bucketing, null handling. These are
  the functions where a wrong answer is plausible and cheap to detect.
- Why: they run in milliseconds, localise failure to a line, and need no infrastructure.
- If skipped: the defects that produce plausible wrong numbers remain undetected, because no
  higher-level test can tell a correct plan from a plausible one without a reference.

**Step 13. Add property tests where the input space is large and the invariant is simple.**

- Decision: round-trip identity, residue bounds, conservation of quantity, monotonicity of
  rounding. State the generator domain explicitly, including what is excluded and why.
- Why: example tests check the cases the author thought of. Property tests check the boundaries
  the author did not think of, and shrink a failure to a minimal example.
- If skipped: the suite passes on n = 100 and fails on n = 1, which is the case a planner will
  eventually hit.

**Step 14. Add contract tests at every boundary and golden files only where assembly matters.**

- Decision: one contract test per published input and output table, asserting keys, grain, unit
  tags, nullability, and value domains. One golden-file test per decision output, with a frozen
  clock and seed.
- Why: contract tests catch schema drift at the boundary where it is cheap to diagnose. Golden
  files catch errors of ordering and integration that no component test can see.
- If skipped: a renamed column silently becomes null in a join, and the plan is wrong without
  anything failing.

### Phase F — Make the suite unavoidable

**Step 15. Put the fast gates in the commit path and the slow gate in continuous integration.**

- Decision: formatting, linting, type checking, unit tests, and property tests run before every
  commit and must finish in seconds. Contract and golden tests run in continuous integration on
  every push. A nightly job runs the full suite against a fresh environment.
- Why: the commit path is the only place where feedback arrives before the author has moved on.
  The nightly job is where environment drift and data-dependent failures appear.
- If skipped: the suite becomes a release ritual, and release rituals are skipped under deadline.

## 6. Worked example

Ellsworth Components distributes electrical connectors and terminals to industrial customers
from three distribution centres. The planning team runs a weekly replenishment plan across about
9,000 stock-keeping units (SKUs). All figures are illustrative and internally consistent.

### 6.1 The repository as inherited

The plan was produced by one file, `plan.py`, of about 1,400 lines. It had been written over two
years by one analyst and improved in place. It worked, in the sense that it produced a file every
Thursday.

Its first forty lines contained:

- a hard-coded input directory on a network share;
- a database password in a string literal;
- `EACHES_PER_CASE = 24`, a constant that had been correct when written;
- three module-level data frames loaded at import time;
- a commented-out block labelled "old logic, do not delete".

There were no tests. There was no lockfile. The README said "run the cells in order".

### 6.2 Three defects, one script

**Defect 1 — the pack constant.** A supplier changed the case pack for connector family C-14
from 24 eaches to 20. The item master was updated. `plan.py` was not, because it did not read the
item master; it used the constant. For a weekly requirement of 240,000 eaches, the plan ordered
10,000 cases. At 20 eaches per case, that supplied 200,000 eaches: a shortfall of 40,000 eaches,
or 16.7% of demand.

The shortfall did not raise an error. The plan was internally consistent: cases times the
constant equalled the demand. The constant was the only thing that was wrong.

**Defect 2 — the time zone.** A new distribution centre in a different time zone sent inventory
snapshots at 23:30 local time. The script parsed timestamps without a time zone and bucketed by
date. Snapshots after 23:00 local were attributed to the following day, so one day of stock was
counted twice at the week boundary. The effect was small in most weeks and large in the week that
contained a public holiday, when the plan was already tight.

**Defect 3 — the credential.** The database password was rotated as part of an unrelated security
review. The script stopped. Nobody could run it for four days, because the password existed only
in the file and in the memory of the analyst who had left.

### 6.3 The rebuilt layout

```text
ellsworth-planning/
|-- pyproject.toml
|-- uv.lock
|-- .python-version
|-- .pre-commit-config.yaml
|-- src/
|   `-- planning/
|       |-- __init__.py
|       |-- config.py          # typed settings, fails fast, redacts secrets
|       |-- units.py           # pure: conversion and rounding
|       |-- demand.py          # pure: bucketing and cleansing rules
|       |-- allocate.py        # pure: allocation rules
|       |-- io/
|       |   |-- reader.py      # Protocol plus warehouse implementation
|       |   `-- writer.py
|       |-- pipeline.py        # imperative shell
|       `-- cli.py             # argument parsing, settings, exit codes
`-- tests/
    |-- conftest.py            # fixtures: item_master, tiny_history, frozen_clock
    |-- unit/test_units.py
    |-- property/test_rounding.py
    |-- contract/test_item_master.py
    `-- golden/test_weekly_plan.py
```

Three details matter more than the tree itself.

First, `units.py` imports nothing from `io/`. It can be imported, called, and tested in a fresh
interpreter with no configuration. That is the property that makes the unit and property levels
cheap.

Second, `config.py` is the only module that reads environment variables. It exposes a settings
object with fields for the warehouse host, the schema, the cut-off time, and the credential. The
credential field is excluded from the object's representation, so a log line cannot leak it.

Third, `pyproject.toml` is the only place where dependencies and tool settings are declared.
There is no `setup.py`, no `requirements.txt`, and no tool configuration in a separate file that
nobody updates.

### 6.4 The tests that would have caught each defect

| Defect | Test level | What the test asserts | Cost to write |
|---|---|---|---|
| Pack constant | Contract | The item master contains a pack quantity per item; the planner reads it; no pack constant exists in `units.py` | 1.5 hours |
| Pack constant | Property | For any demand and any pack from 1 to 200, rounding supplies at least the demand and wastes less than one case | 45 minutes |
| Time zone | Unit | A timestamp at 23:30 in a named zone buckets to the local day, and the same instant in UTC buckets to a different day | 30 minutes |
| Credential | Configuration | Constructing settings with a missing variable raises an error naming that variable | 20 minutes |
| Whole script | Golden file | A frozen week of inputs produces a plan that matches a reviewed expected plan | 2 hours |

The first row is the one that would have prevented the incident, and it is not a test of the
arithmetic. It is a test of the *dependency*: the planner must obtain the pack from the item
master rather than from a constant. No amount of testing `ceil_div` would have found the defect,
because `ceil_div` was correct.

This is the general lesson of the worked example. **The most valuable test is often the one that
asserts where a value came from, not what the value is.**

### 6.5 What it cost and what it bought

The rebuild took eleven working days. The test suite described above took about four and a half
hours to write and runs in under forty seconds.

The incident it was designed around cost:

| Item | Quantity | Unit cost | Cost |
|---|---:|---:|---:|
| Emergency buy of 40,000 eaches | 40,000 EA | EUR 0.46 premium per EA | EUR 18,400 |
| Planner overtime | 30 h | EUR 72 per h | EUR 2,160 |
| Quantified total | | | **EUR 20,560** |

The service consequence at the customer was not quantified: two expedites and a line-down risk
on a harness assembly line. The number in the table is therefore a floor, not a total.

Against that, the four and a half hours of test writing cost about EUR 324 at the same loaded
rate, and the forty-second suite costs nothing measurable per run. One avoided incident pays for
the suite roughly sixty times over. The arithmetic is developed in section 13.

The honest caveat is that the tests were written after the incident, which is what told the team
which tests to write. That is the normal order in small teams; the three defects are a checklist,
not a unique story.

## 7. Code, query, or calculation

One test file, because the point is the shape of the claim rather than the size of the program.
It encodes the rounding rule from 01A.01 and three properties that a rounding function must
satisfy for any demand.

```python
from hypothesis import given, strategies as st

EA_PER_CASE, CASES_PER_PALLET = 24, 48

def ceil_div(n: int, d: int) -> int:
    return -(-n // d)

@given(st.integers(min_value=0, max_value=10**6))
def test_case_rounding_is_tight(n: int) -> None:
    cases = ceil_div(n, EA_PER_CASE)
    assert cases * EA_PER_CASE >= n
    assert n == 0 or (cases - 1) * EA_PER_CASE < n

@given(st.integers(min_value=1, max_value=10**6))
def test_cascade_never_orders_less_than_direct(n: int) -> None:
    cascaded = ceil_div(ceil_div(n, EA_PER_CASE), CASES_PER_PALLET)
    assert cascaded >= ceil_div(n, EA_PER_CASE * CASES_PER_PALLET)

@given(st.integers(min_value=0, max_value=10**5))
def test_pallet_residue_is_bounded(n: int) -> None:
    unit = EA_PER_CASE * CASES_PER_PALLET
    residue = ceil_div(n, unit) * unit - n
    assert 0 <= residue < unit
```

**What the first test claims.** For every non-negative demand up to one million eaches, rounding
up to whole cases supplies at least the demand, and removing one case would supply less. The
second clause is what makes the function *tight*: without it, a function that always returned
one million cases would pass the first clause.

**What the second test claims.** Rounding in two steps never orders fewer cases than rounding
once. This is the conservation property behind the cascade in 01A.01: rounding up at each level
can only add, never remove. It does not claim that the two-step result equals the one-step
result. It usually does not, and that difference is the residue.

**What the third test claims.** The pallet residue is always in the half-open interval from zero
to one pallet. It cannot be negative, because rounding up never removes material, and it cannot
reach a full pallet, because that would mean one fewer pallet was sufficient.

**Why integers rather than floating point.** Case counts are integers and pack sizes are
integers, so the domain is exact. The moment the same logic is expressed in binary floating
point, `0.1 + 0.2 != 0.3`, and a test that compares floats for equality becomes a test of the
platform's rounding mode. Where a conversion genuinely involves a fraction — kilograms per case,
for instance — use `Decimal` with an explicit quantisation rule, and compare with a stated
tolerance rather than equality.

**Why property-based rather than example-based.** An example test would assert
`ceil_div(100, 24) == 5`. It would pass against an implementation that returned `n // 24` for
every value except 100. A property test generates values across the domain, and when it finds a
failure, the Hypothesis library shrinks the input to a minimal counterexample — often `n = 1` or
`n = 23`, the two cases an author is least likely to write by hand.

**What none of these tests claims.**

| Not tested | Why not | Where it belongs |
|---|---|---|
| That 24 is the correct pack size | The test hard-codes 24 as a domain assumption | Contract test against the item master |
| That the pack is in eaches and not cases | Both are integers; the type checker cannot distinguish them | Tagged type or unit-carrying value object |
| That the rounding direction is the policy the business wants | Ceiling is a choice, not a truth | A documented decision, reviewed with planning |
| That the generated inputs resemble real demand | The generator is uniform over a range | A separate fixture or production-monitoring check |

> **Note:** The generator domain is part of the test's meaning. `st.integers(min_value=0,
> max_value=10**6)` excludes negative demand, which is correct for this function but would hide a
> defect if the function were also used for adjustments. State exclusions explicitly; do not let
> them be an accident of the generator.

**The one-line version of the lesson.** A property test is a claim about all inputs in a stated
domain. The domain, the invariant, and the excluded cases are the test. The code is only the
notation.

## 8. Data, standards, and regulatory context

Most of this topic is convention. Python has an official specification for packaging metadata
and an official specification for type information, but neither specifies how a supply chain
team should lay out its repository. The distinction matters because a convention can be
challenged and a requirement cannot.

### 8.1 Packaging and typing specifications

| Specification | What it defines | Relevance here | Grade |
|---|---|---|---|
| PEP 484, Type Hints | Syntax and semantics for function and variable annotations, and the `typing` module | The language-level basis for every annotation in the code | B |
| PEP 561, Distributing and Packaging Type Information | The `py.typed` marker, bundled and separate stub packages, and resolution order | Determines whether a downstream checker can see your types at all | B |
| Typing specification, distributing type information | The current canonical statement of stub distribution and resolution | Supersedes parts of PEP 561; consult before publishing a typed package | B |
| Python Packaging User Guide, `src` layout versus flat layout | The documented rationale for placing importable code under `src/` | The evidence behind step 2 | C |
| pytest, Good Integration Practices | Recommended test layout and the `tests/__init__.py` import-path caveat | Explains why an installed-package test can silently become a source-tree test | C |
| setuptools, Package Discovery | How flat-layout and `src`-layout discovery select packages | Explains which files end up in the artifact | C |
| `pyproject.toml` project metadata convention | Where dependencies, build backend, and tool configuration are declared | The single-file convention used throughout this topic | C |

### 8.2 Configuration, secrets, and the twelve-factor convention

The Twelve-Factor App's third factor states that configuration should be stored in the
environment. Its stated litmus test is whether the codebase could be made public at any moment
without compromising credentials. That test is more useful than the prescription, because it
applies to repositories that do not follow the rest of the twelve factors. (Grade C: a widely
adopted convention, not a standard and not an empirical study.)

Two consequences for an analytics repository:

- **A committed `.env` file fails the test even when it is ignored.** The file is in the working
  tree, and the history retains it if it was ever committed. Ignoring a file prevents a future
  commit, not a past one.
- **A notebook is a code path with no environment boundary.** Credentials typed into a cell are
  stored in the notebook's output, in its checkpoint, and often in an exported HTML file. The
  twelve-factor test applies to notebooks too, and they fail it more often than scripts do.

The regulatory angle is indirect but real. Where software is validated inside a quality
management system — medical devices, pharmaceutical manufacturing, some financial controls —
the build and test evidence becomes part of the record, and the question "which code and which
dependencies produced this result" has a documentary answer. This topic does not give legal
advice, and the exact obligation depends on the regime; the practical point is that a lockfile,
a test report, and a tagged release are the artifacts such a regime will ask for. Grade C for the
general claim; consult the quality function for the specific one.

### 8.3 What the standards do not specify

None of the standards above will tell you:

- whether a quantity is in eaches, cases, or pallets;
- whether rounding should be up, down, or half-even;
- whether a null quantity means zero, unknown, or not applicable;
- whether two item records with the same key are a duplicate or a revision.

Those are domain decisions, and they are exactly the decisions that a contract test or a tagged
type has to encode. A repository can be fully compliant with every specification in the table and
still add 16.7% to an order because a pack constant was stale.

## 9. Industry practice and case evidence

### 9.1 The `src` layout: a documented rationale, not a style preference

The Python Packaging User Guide gives three reasons for the `src` layout, and all three are
about test fidelity rather than aesthetics. First, with a flat layout, Python places the current
working directory first on the import path, so tests import the in-development source tree rather
than the installed package. Second, packaging defects such as a file missing from the
distribution are therefore masked. Third, path-configuration-based editable installs can expose
project files such as `README.md` and `noxfile.py` to imports under a flat layout, so an import
that works editably can fail on a regular install.

The pytest documentation goes further and states that a `src` layout is strongly suggested,
particularly with the default import mode. It also documents a trap that survives the layout
change: adding an `__init__.py` to the `tests/` directory causes pytest to prepend the repository
root to the import path, which makes the package importable from the source tree and defeats
installed-package testing. The fix is either to omit that file or to set the import mode to
`importlib` and add the source directory explicitly.

setuptools documents the complementary failure from the packaging side: flat-layout automatic
discovery can sweep in directories such as `docs/` and `tests`, and it refuses projects with
multiple top-level packages. The `src` layout makes discovery unambiguous because there is only
one candidate.

Taken together, these three sources support a specific claim and no more: **the layout changes
what the test command actually tests.** They do not support a claim that `src` layout improves
code quality, team velocity, or defect rates. Those would require evidence that none of the
sources provides. (Grade C: authoritative project documentation, not empirical study.)

### 9.2 The test pyramid and its critics

The pyramid is usually attributed to Mike Cohn's *Succeeding with Agile* (2009). Ham Vocke's
widely cited practitioner article, "The Practical Test Pyramid" (2018), is both a defence and a
critique: it calls the original pyramid "overly simplistic and can therefore be misleading",
argues that the layer names are poor, and reduces the idea to two rules — write tests at
different granularities, and have fewer tests as you go higher.

The critiques that matter for data code are different from the general software critique.

- **Radziwill and Freeman**, in "Reframing the Test Pyramid for Digitally Transformed
  Organizations" (arXiv, 2020), argue that machine-learning systems have emergent properties with
  no deterministic test cases, which makes unit decomposition impractical, and propose a pyramid
  organised by *where* tests run (data and pipelines, models, services, interfaces) rather than by
  test type. The paper is a preprint and reports informal validation through an insurance-company
  failure-mode exercise; treat it as a framing contribution, not as measured evidence. (Grade C.)
- **Eugene Yan**, in "Writing Robust Tests for Data and ML Pipelines" (2022), makes a practical
  observation from pipeline work: new data or logic has an additive effect on row-level unit and
  schema tests, but a retroactive effect on column-level and table-level integration tests, which
  makes the latter brittle. His recommendation is many row-level and schema tests and a handful
  of the rest, plus property-based testing. (Grade C: practitioner account, no measurement.)
The synthesis used in this topic is therefore deliberately loose. The levels are named by *what
they can catch*, not by a required count. The proportions are a starting budget, adjusted by the
cost of running each level and the cost of the defect it catches.

### 9.3 What the evidence does not establish

Two claims are commonly made about this material and cannot be supported from the sources
consulted:

1. **"Typed Python has fewer production defects."** No controlled study of Python type
   annotations in supply chain analytics was found. The mechanism is plausible — a checker
   removes a class of error before runtime — but the size of the effect in this domain is unknown.
2. **"Most analytics pipelines are untested."** No reliable public measurement was found. Survey
   figures exist for software testing generally, but they use inconsistent definitions of "test"
   and rarely isolate analytics or supply chain teams.

None of these gaps weakens the practical argument, because the practical argument is local: in a
named repository, with a named incident, a test either catches the defect or it does not. That is
measurable without a population study.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Flat layout masks packaging defects | Tests pass locally, install from wheel fails | Working directory precedes the installed package on the import path | Missing files discovered in production | `src` layout plus a clean-runner build |
| Unpinned dependencies | "It worked last month" with no code change | Transitive resolution drifts on every install | Numbers change without a commit | Lockfile plus frozen installs |
| Lockfile not used in continuous integration | CI installs latest compatible versions | The tested set differs from the released set | A green build that cannot be reproduced | Install with the lock and hash verification |
| Secret committed to the repository | A `.env` file in the history, or a password in a notebook cell | Convenience at authoring time | Credential exposure and a rotation outage | Secret store; secret scanning in the commit hook |
| Tests import the source tree | Coverage is high but the built artifact is never exercised | `tests/__init__.py` or `PYTHONPATH` manipulation | Packaging regressions reach users | Remove the file; set the import mode explicitly |
| `Any` used to silence the checker | Suppression count rises every sprint | The quickest way to make an error disappear | Type checking becomes decorative | Count suppressions as a budget; require an error code and a reason |
| Suite too slow to run | Developers commit with the hook bypassed | Feedback arrives after the author has moved on | Defects found in review or production | Move slow levels to CI; keep the commit path under seconds |
| Flaky tests tolerated | "Just rerun it" in the team chat | Non-determinism from time, ordering, or shared state | Every genuine failure is also rerun | Quarantine, fix, or delete; never normalise the rerun |
| Golden files auto-updated in CI | The expected file changes on every green build | The update command runs in the pipeline | The suite records output and asserts nothing | Update only by human review; fail on drift |
| Over-mocking | Tests assert that a mock was called, not what was computed | The test double becomes the subject | Refactoring breaks tests without changing behaviour | Mock at the boundary, assert on outputs |
| Coverage used as a target | A sprint goal of "90% coverage" | Tests written to execute lines, not to assert | Assertion-free tests and false confidence | Track coverage as a diagnostic, never a target |
| Testing the framework, not the rule | Tests fail on a pandas upgrade | Assertions about library behaviour | Maintenance cost with no defect detection | Test your use of the library at the boundary |
| Shared mutable fixtures | Tests pass alone and fail in a suite | One test mutates a fixture another test reads | Order-dependent failures | Factory fixtures; function scope by default |
| Wall-clock and time-zone dependencies | Tests fail near midnight or after a DST change | The clock is an undeclared input | Intermittent failures that look random | Inject a clock; freeze it in tests |
| The notebook is the only artifact | "Run cells 1 to 14, but not 11" | Program state lives in the kernel, not the file | The pipeline cannot be rerun by anyone else | Extract to modules; keep notebooks for exploration |

Two of these deserve expansion.

**Coverage as a target.** Coverage measures lines executed, not behaviours asserted. A test that
calls a function and asserts nothing raises coverage as much as one that checks the result. Once
coverage is a target, the cheapest way to raise it is to execute code without checking it. The
useful question is which invariant has no test that would fail if it were broken.

**The flaky test.** Flakiness is a failure of the suite's meaning, not an annoyance. A suite is
useful only if red implies a defect. One test with a ten percent false-failure rate changes the
team's interpretation of every red result, and the habit persists after the test is fixed. Treat
it as an incident: quarantine it the same day, then fix or delete it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A package index is reachable from every build environment | Installation and locking require it | Air-gapped environments, proxy restrictions | Build from a clean runner with no cache |
| A continuous integration runner exists | The slow gates need a machine that is not a laptop | Very small teams, regulated environments with no pipeline | Time a clean checkout by hand; if it exceeds an hour, fix the process before adding tests |
| Tests can run without production credentials | Hermetic tests are the cheap ones | Tests that read the live warehouse | Run the suite with no network and no environment file |
| Synthetic fixtures are representative enough | Small data keeps tests fast | Pathologies that only appear at scale, such as skew or late arrival | Compare fixture assumptions with production profiling |
| The type checker's model matches runtime | Annotations are not enforced at runtime | Dynamic imports, monkey-patching, `eval` | Runtime validation at boundaries, not only static checking |
| Golden outputs are deterministic given a seed and a frozen clock | A golden file is only meaningful if it is stable | Floating-point aggregation, dictionary ordering, parallel writes | Run the same input twice and diff byte for byte |
| Dependencies provide wheels for the target platform | Source builds can fail or produce different binaries | Scientific packages on unusual architectures | Build the environment on the deployment image |
| The suite fits the team's patience budget | A suite that is not run has no value | Suite growth without pruning | Measure the time from commit to result, not the test count |
| One deployable per lockfile | The lock describes one environment | Monorepos with several services sharing a lock | Split locks per deployable |
| Test maintenance is funded | Tests are code and decay | Ownership assigned to nobody | Track time spent fixing tests as a line item |

Three limits are worth stating plainly.

**Types cannot express the domain.** A type checker will accept a function that adds kilograms to
eaches if both are annotated `float`. It will accept a safety-stock calculation that uses the
wrong service-level target, because the target is a valid number. Types remove a class of error
concerned with shape and nullability. They do not remove errors of meaning, and the tests that
catch those errors have to encode the meaning explicitly.

**Tests cannot establish that the requirement was right.** A suite can prove that the code does
what the team decided. It cannot prove that the decision was correct. A perfectly tested
rounding rule that rounds in the wrong direction is a perfectly tested defect, and the test will
defend it against correction. This is why section 7's table includes "that the rounding direction
is the policy the business wants" as a non-test, and why the decision belongs in a review rather
than an assertion.

**Small synthetic data cannot reproduce production pathologies.** A fixture with twelve rows
cannot exhibit the skew, the duplicate keys, or the late-arriving records that a million-row
table contains. That is not a reason to abandon fixtures; it is a reason to pair them with
production data monitoring, which is the subject of Topic 11. Tests and monitors answer different
questions: tests ask whether the code is correct, monitors ask whether the data is.

## 12. Exercises

### Exercise 1 — Audit an inherited script (analysis, 60 minutes)

Take a Python script you have written or inherited that produces a recurring report or plan.
Audit it against the five symptoms in section 1 and the three defects in section 6.2.

**Deliverable:** a one-page audit with one row per finding, the symptom it matches, and the
failure it could cause.
**Acceptance criteria:** at least five findings; each finding names a concrete line or construct,
not a general worry; at least one finding is about where a value came from rather than what it
computes; the audit distinguishes "would fail today" from "would fail after a plausible change".
**Time:** 60 minutes.

### Exercise 2 — Rebuild the layout (application, 90 minutes)

Convert the audited script into a `src` layout with a pure core, an input/output shell, a typed
settings module, and a test directory. Do not change the business logic.

**Deliverable:** a repository tree, the metadata file, and a test that imports the package and
calls one pure function.
**Acceptance criteria:** the test runs from a clean checkout after an editable install with no
`PYTHONPATH` change; no module in the core imports a database, file, or clock library; the
settings module is the only place that reads the environment; the tree contains no `utils.py`.
**Time:** 90 minutes.

### Exercise 3 — Property test and rounding cascade (calculation and code, 60 minutes)

A product has 12 eaches per inner pack, 4 inner packs per case, 6 cases per layer, 5 layers per
pallet, and 24 pallets per container. Weekly demand is 250,000 eaches.

(a) Compute the rounding cascade, rounding up at each level, and state the residue in eaches at
each level and as a percentage of demand.
(b) Write a property test asserting that the cascade never orders fewer units than a single-step
rounding to containers.
(c) State one invariant your test does not check and where it belongs instead.

**Deliverable:** the cascade table, the test, and the non-test statement.
**Acceptance criteria:** arithmetic correct to the unit; every level named with its pack factor;
the property test states its generator domain; the non-test is a real invariant, not a
formality.
**Time:** 60 minutes.

### Exercise 4 — Contract test for an item master (design, 60 minutes)

Design a contract test for an item master table that feeds a planner. Include the columns you
would require, the key and grain, the unit representation, and the null policy.

**Deliverable:** a table of assertions with the failure message each would produce.
**Acceptance criteria:** at least eight assertions; at least two assert a *relationship* between
columns rather than a property of one column; one assertion would have caught the pack-constant
defect in section 6.2; the failure messages name the offending column and value.
**Time:** 60 minutes.

### Exercise 5 — Golden file with determinism controls (application, 75 minutes)

Create a golden-file test for a planning output. It must be stable across runs and must fail when
the ordering of two rows changes.

**Deliverable:** the test, the golden file, a script or command that regenerates it under human
review, and a note listing every source of non-determinism you had to control.
**Acceptance criteria:** two consecutive runs produce byte-identical output; the test fails when a
row is added, removed, or reordered; regeneration is a separate, explicit command; the note names
at least clock, seed, row ordering, and floating-point formatting.
**Time:** 75 minutes.

### Exercise 6 — Configuration and secret separation (design, 45 minutes)

Refactor a script that reads three environment variables and one password. Produce a settings
object that validates at startup, redacts the secret in its representation, and fails with a
message naming any missing variable.

**Deliverable:** the settings module and three tests: valid configuration, missing variable,
redacted representation.
**Acceptance criteria:** no other module reads the environment; the error message names the
variable and does not print its value; the redaction test asserts that the secret string does not
appear in `repr(settings)`; defaults are documented and orthogonal.
**Time:** 45 minutes.

### Exercise 7 — Reproduce the environment (experiment, 50 minutes)

Take a project with unpinned dependencies. Produce a lockfile, install from it with hash
verification on a clean machine or container, and record what changed.

**Deliverable:** the lockfile, the install command, and a table of version changes for direct and
transitive dependencies.
**Acceptance criteria:** the clean install succeeds without network access to anything except the
package index; the table separates direct from transitive changes; at least one transitive change
is identified that would have been invisible in the original requirements file; the repository
records how to update the lock deliberately.
**Time:** 50 minutes.

### Exercise 8 — Test economics (calculation, 40 minutes)

A team of eight engineers each pushes six times a day, twenty days a month. The suite takes 6.5
minutes. Loaded engineering cost is EUR 72 per hour. A change that reduces the suite to 1.5
minutes takes three working days.

(a) Compute the waiting cost per month before and after, the annual saving, and the payback
period.
(b) A defect of the kind described in section 6.5 costs EUR 20,560 and occurs about once every
two years. A proposed contract test costs EUR 324 to write, EUR 144 per year to maintain, and
would catch it with probability 0.6. Compute the expected annual benefit and state whether the
test pays.
(c) State one assumption in (b) that is doing most of the work.

**Deliverable:** the arithmetic, the decision, and the assumption.
**Acceptance criteria:** units stated at every step; the payback period is expressed in months and
days; the expected-value calculation separates probability of occurrence from probability of
detection; the assumption is one that could plausibly be wrong.
**Time:** 40 minutes.

### Exercise 9 — Design the gate (design, 60 minutes)

Design the commit and continuous integration gates for a repository with unit, property,
contract, and golden tests. Allocate each check to a stage and give a time budget.

**Deliverable:** a table of stage, checks, time budget, and failure action.
**Acceptance criteria:** the commit stage completes in under ten seconds on a normal laptop; no
check exists in both stages without a stated reason; the golden test is in the stage where a
human will see the diff; the bypass procedure is documented and logged.
**Time:** 60 minutes.

### Exercise 10 — Red-team a suite (critique, 45 minutes)

Take a test suite you know. For each test, write the defect it would fail to catch.

**Deliverable:** a table of test, what it asserts, and one defect it cannot catch.
**Acceptance criteria:** at least ten tests; at least three findings are about tests that assert
implementation rather than behaviour; at least two findings identify an invariant with no test
at all; one recommendation proposes deletion, not addition.
**Time:** 45 minutes.

## 13. Self-check answers

**Exercise 2 — layout rubric.** A strong submission installs cleanly, tests the installed
package, and keeps input/output out of the core; a weak one leaves the database call inside the
rounding function.

**Exercise 3(a) — the cascade, in full.**

Pack factors: 12 eaches per inner, 4 inner per case (48 eaches), 6 cases per layer (288 eaches),
5 layers per pallet (1,440 eaches), 24 pallets per container (34,560 eaches).

```text
level       exact          rounded up     eaches represented     residue    % of demand
inner       250,000/12     20,834         20,834 x 12 = 250,008        +8        0.003%
case        20,834/4        5,209          5,209 x 48 = 250,032       +32        0.013%
layer        5,209/6           869            869 x 288 = 250,272    +272        0.109%
pallet         869/5           174          174 x 1,440 = 250,560    +560        0.224%
container    174/24              8        8 x 34,560 = 276,480    +26,480       10.592%
```

The incremental additions are 8, 24, 240, 288, and 25,920 eaches. The last step accounts for
25,920 of the 26,480 total residue, or 97.9% of it. A planner looking only at case-level rounding
would conclude that rounding costs 32 eaches, which is 0.013% of demand and invisible. The
container step costs 10.6%, which is a decision.

Rounding down instead gives 20,833 inner, 5,208 cases, 868 layers, 173 pallets, and 7 containers:
241,920 eaches, an 8,080-each shortfall (3.2%). The direction is a policy decision, and the two
directions differ by 13.8 percentage points of supply at the container level.

At a unit cost of EUR 1.20, the 26,480-each residue represents EUR 31,776 of material pulled
forward. Held for an assumed 45 days at an 18% annual holding rate, the carrying cost is
`31,776 x 0.18 x 45 / 365 = EUR 705`. That figure is small relative to the material value, which
is exactly why the residue is easy to ignore and why it is worth stating explicitly.

**Exercise 3(b) — the property test.**

The test in section 7 is the required shape. The acceptance point is that the generator domain
is stated: non-negative integers, because a negative demand is not a demand. The invariant is
`cascade(n) >= direct(n)` for all n in the domain. A submission that tests only `n = 250,000`
has written an example test and should be marked accordingly, regardless of whether it passes.

**Exercise 3(c) — what the test does not check.**

The strongest answers name one of: that the pack factors match the item master (contract test);
that the rounding direction is the policy the business wants (reviewed decision); that the
quantities are in eaches rather than cases (tagged type or validated value object); or that the
demand figure itself is correct (a data quality check upstream). A weak answer says "the test
does not check everything", which is true of every test and therefore says nothing.

**Exercise 4 — rubric for the contract test.**

A strong submission asserts, at minimum: one row per item key; the key is unique and non-null;
the grain is item by location by effective date; every quantity column is numeric; every quantity
column has a companion unit code from a closed list; the pack quantity is a positive integer; the
pack quantity is consistent with the inner-pack and case fields; the effective-from date is not
later than the effective-to date; and no item has two open versions at the same location.

The relationship assertions are the discriminating ones. "Pack quantity is positive" is a column
property that a schema validator can generate. "Pack quantity equals inner per case times eaches
per inner" is a domain relationship that only a person who understands the pack hierarchy will
write, and it is the assertion that catches the class of defect in section 6.2. A submission with
ten column checks and no relationship check has missed the point.

**Exercise 5 — rubric for the golden file.**

A strong submission produces byte-identical output on two consecutive runs, sorts rows by a
declared key before comparison, formats floats with a fixed number of decimal places, injects a
frozen clock, and seeds any random process. It regenerates the golden file only through an
explicit command that a reviewer runs, and it fails on any diff, including a pure reordering.

**Exercise 6 — configuration rubric.** A strong submission validates once, names the missing
variable, and redacts the secret; a weak one reads the environment in five modules.

**Exercise 8(a) — the waiting cost, in full.**

```text
pushes per month        = 8 engineers x 6 pushes/day x 20 days = 960
before: 960 x 6.5 min   = 6,240 min = 104.0 h
         104.0 h x EUR 72/h = EUR 7,488 per month
after:  960 x 1.5 min   = 1,440 min = 24.0 h
         24.0 h x EUR 72/h = EUR 1,728 per month
saving  = 7,488 - 1,728 = EUR 5,760 per month
annual  = 5,760 x 12 = EUR 69,120 per year
cost    = 3 days x 8 h x EUR 72/h = EUR 1,728
payback = 1,728 / 5,760 = 0.30 month = 9 days
```

The payback is nine days, assuming waiting time is a real cost; a team that context-switches
recovers some of it, so treat the figure as an upper bound. At half the value the payback is under
three weeks.

**Exercise 8(b) — expected value of the contract test.**

```text
events per year                 = 0.5
probability test catches one    = 0.6
expected catches per year       = 0.5 x 0.6 = 0.3
benefit per catch               = EUR 20,560
expected annual benefit         = 0.3 x 20,560 = EUR 6,168
annual cost                     = EUR 144 maintenance
first-year cost                 = EUR 324 to write + EUR 144 = EUR 468
first-year net                  = 6,168 - 468 = EUR 5,700
```

The test pays, and it pays for a reason that is worth naming: the cost of writing it is about
1.6% of the loss it is designed to prevent. The result is sensitive to the 0.6 detection
probability: at 0.5 the benefit is EUR 5,140, and at an event rate of once per ten years it is
EUR 1,234. The conclusion holds in both cases, with a thinner margin in the second.

**Exercise 8(c) — the assumption doing the most work.**

The assumption that matters most is that the test would actually be run and would actually fail
on the defect. A contract test that is not in the commit or continuous integration gate, or one
whose failure is routinely ignored, has an expected benefit close to zero regardless of its
detection probability in principle. This is the economics of a test that nobody runs, stated as
an assumption rather than a slogan.

**Exercise 9 — rubric for the gate design.**

A strong design puts formatting, linting, type checking, and unit and property tests in the
commit stage with a budget of about five to ten seconds, using incremental checking where the
tool supports it. It puts contract and golden tests in continuous integration on every push,
with a budget of a few minutes, and it puts a clean-environment install and full suite in a
nightly job. It documents the bypass and logs its use, so that a rising bypass rate is visible
before it becomes a habit.

**Exercise 10 — rubric for the red-team.**

The strongest submissions identify tests that assert a mock was called, tests that assert an
internal helper was invoked, and tests whose only assertion is that no exception was raised. All
three execute code without constraining behaviour. The best also propose deletion: a test that
cannot fail for a reason anyone cares about is maintenance cost with no benefit.

## 14. Interview questions

**1. "Why would you put the package under `src/` rather than at the repository root?"**

A strong answer explains the import-path mechanism: with a flat layout the working directory is
first on the path, so tests import the source tree rather than the installed distribution, and
packaging defects stay invisible. It mentions that a `src` layout forces an install step and that
the cost is small. A weak answer says it is "cleaner" or "the standard", which does not explain
what changes about the test result.

**2. "What does a type hint give you that a docstring does not?"**

A strong answer says a hint is machine-checked on every run of the checker, so it cannot drift
silently, while a docstring is checked by a reader and drifts as soon as the code changes. It
adds that the hint carries shape and nullability, and that domain constraints such as "positive"
or "in base units" still need prose or a validated value object. A weak answer treats the hint as
documentation with different syntax and cannot say what it catches.

**3. "Where does configuration end and a secret begin?"**

A strong answer gives the operational test: configuration may be read aloud and logged;
a secret grants access and must be redacted, rotated, and stored outside the repository. It notes
that both belong outside the code, that a committed `.env` file fails even when ignored because
history retains it, and that notebooks leak secrets into outputs. A weak answer says "use
environment variables", which is a mechanism rather than a distinction.

**4. "What does a lockfile guarantee, and what does it not guarantee?"**

A strong answer says it fixes the exact version of every direct and transitive dependency, and
with hashes it fixes the bytes. It does not guarantee the same operating system, the same
interpreter patch level, the same native libraries, or the same data. It also notes that a lock
is only meaningful if every environment installs from it. A weak answer says "it makes the build
reproducible" and stops, which overstates the guarantee.

**5. "How would you test a unit conversion?"**

A strong answer separates the conversion factor (a contract or configuration concern) from the
conversion function (a unit concern), tests the function with exact integer or `Decimal`
arithmetic, and adds a property test for round-trip identity within a stated tolerance. It
mentions that a float equality assertion is testing the platform, not the code. A weak answer
writes one example with two numbers and calls it done.

**6. "What is a contract test, and how is it different from a schema check?"**

A strong answer says a schema check asks whether columns and types exist; a contract test asks
whether the producer's promise holds, including key uniqueness, grain, unit codes, null policy,
and value domains. It notes that a contract test runs at the boundary and should fail with a
message that names the offending column and value. A weak answer uses the terms
interchangeably.

**7. "When is a golden-file test the right choice?"**

A strong answer says when the defect is in the assembly of correct components — ordering, joins,
aggregation — and no component test can see it. It adds the conditions: deterministic output,
a frozen clock and seed, a reviewed regeneration step, and a diff a human will actually read. A
weak answer says "for regression testing", which is true of every test.

**8. "What would you not test?"**

A strong answer names third-party library behaviour, trivial accessors, exact log strings,
implementation details that are not the contract, and production data itself, which is
monitoring rather than testing. It gives a reason for each: maintenance cost without defect
detection. A weak answer says "I test everything important", which cannot be operationalised.

**9. "Your suite takes twelve minutes. What do you do?"**

A strong answer measures where the time goes before changing anything, then moves slow levels
out of the commit path, parallelises what is independent, and deletes tests that cannot fail for
a reason anyone cares about. It computes the waiting cost and compares it with the cost of the
fix. A weak answer says "add more CI runners", which may not address the slowest test.

**10. "How do you handle a flaky test?"**

A strong answer quarantines it the same day, identifies the non-determinism — clock, ordering,
shared state, network — and either fixes or deletes it. It explains that the damage is to the
team's interpretation of red, not only to that test. A weak answer says "rerun it" or "increase
the timeout", both of which hide the cause.

**11. "How would you introduce type checking into a 50,000-line legacy codebase?"**

A strong answer ratchets: strict settings for new and core modules, an exemption list for
existing violations, suppressions counted as a budget, and error codes with reasons. It
annotates boundaries first, because those are what a stranger reads. A weak answer enables strict
mode everywhere, which produces thousands of errors and then a disabled checker.

**12. "What is the main weakness of the test pyramid for data code?"**

A strong answer says the pyramid assumes a deterministic unit with a clear boundary, and data
code often has neither: an aggregation is correct only relative to a grain, a schema is a
contract rather than a unit, and machine-learning behaviour can be emergent. It notes that teams
therefore end up with many schema checks and few classic unit tests, and that the levels should
be defined by what they catch rather than by a required ratio. A weak answer recites the
proportions.

## 15. Key terms

- **Repository layout** — the arrangement of source, tests, and metadata in a version-controlled tree.
- **Package** — an importable namespace; **distribution** — the versioned artifact that installs it.
- **`src` layout** — importable code under `src/`, outside the repository root's import path.
- **Flat layout** — the package at the repository root, importable from the working directory.
- **Editable install** — an install that points the interpreter at the working tree.
- **Wheel** — a built distribution installable without a build step.
- **Lockfile** — the exact resolved set of direct and transitive dependencies.
- **Hash pinning** — recording a digest per artifact and refusing a mismatch.
- **Configuration** — deployment-varying, non-secret values; **secret** — a value whose disclosure grants access.
- **Typed settings** — a validated object constructed once at startup from the environment.
- **Type hint** — a machine-checked annotation of a value's expected type.
- **Gradual typing** — annotating part of a codebase while untyped code remains unchecked.
- **`Any`** — a type that disables checking; **`Unknown`** — a checker-specific undeclared type.
- **`Protocol`** — a structural interface satisfied by any object with the required methods.
- **Strict mode** — a checker configuration rejecting untyped definitions and implicit `Any`.
- **Pure function** — output depends only on arguments; no side effects.
- **Imperative shell** — the outer layer that performs input and output and calls the core.
- **Fixture** — setup injected into a test; a factory fixture returns a callable so each test builds its own data.
- **Parametrisation** — one test body run against several input cases, reported separately.
- **Test pyramid** — a heuristic for distributing tests by granularity, not a required ratio.
- **Unit test** — a test of one function or small unit with no input or output.
- **Contract test** — a test of keys, grain, units, nullability, and domains at a boundary.
- **Property-based test** — an invariant asserted over a generated domain, with shrinking.
- **Golden-file test** — a full output compared with a reviewed, versioned expected file.
- **Hermetic test** — a test that depends on nothing outside its declared inputs.
- **Flaky test** — a test that passes and fails on unchanged code.
- **Coverage** — the proportion of statements executed, not of behaviour asserted.
- **Determinism** — identical inputs and code produce identical outputs, given stated seed and clock.
- **Pre-commit hook** — a check before a commit; **continuous integration gate** — the required check on a push.

## 16. Sources

Grades follow the module convention: A peer-reviewed or audited, B regulatory or official
specification, C company case study, practitioner account, or authoritative project
documentation, D vendor or project marketing, E press summary. A grade describes the strength of
the claim the source can support, not the quality of the document. All URLs were consulted in
November 2026.

**Official specifications and standards (Grade B)**

- PEP 484, "Type Hints". Python Enhancement Proposal. https://peps.python.org/pep-0484/
- PEP 561, "Distributing and Packaging Type Information". Python Enhancement Proposal.
  https://peps.python.org/pep-0561/
- "Distributing type information", Python Typing Specification.
  https://typing.python.org/en/latest/spec/distributing.html

**Authoritative project documentation (Grade C)**

- Python Packaging User Guide, "`src` layout vs flat layout".
  https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- pytest documentation, "Good Integration Practices".
  https://docs.pytest.org/en/latest/goodpractices.html
- pytest documentation, "Fixtures reference".
  https://docs.pytest.org/en/stable/reference/fixtures.html
- pytest documentation, "How to parametrize fixtures and test functions".
  https://docs.pytest.org/en/stable/how-to/parametrize.html
- setuptools documentation, "Package Discovery and Namespace Packages".
  https://setuptools.pypa.io/en/stable/userguide/package_discovery.html
- The Twelve-Factor App, Factor III, "Config". https://12factor.net/config
- pre-commit documentation. https://pre-commit.com/
- Hypothesis documentation, "Anatomy of a Hypothesis Based Test".
  https://hypothesis.works/articles/anatomy-of-a-test/

**Technical reports and practitioner evidence (Grade C)**

- Vocke, H. (2018). "The Practical Test Pyramid". martinfowler.com.
  https://martinfowler.com/articles/practical-test-pyramid.html
- Radziwill, N. and Freeman, G. (2020). "Reframing the Test Pyramid for Digitally Transformed
  Organizations". arXiv:2011.00655. Preprint; informal validation. https://arxiv.org/pdf/2011.00655
- Yan, E. (2022). "Writing Robust Tests for Data and ML Pipelines".
  https://eugeneyan.com/writing/testing-pipelines/

**Vendor and project documentation (Grade D)**

- Microsoft, pyright configuration documentation, including `typeCheckingMode` levels.
  https://github.com/microsoft/pyright/blob/main/docs/configuration.md
- Astral, uv documentation, "Locking environments" and "Locking and syncing".
  https://docs.astral.sh/uv/pip/compile/ and https://docs.astral.sh/uv/concepts/projects/sync/
- Astral, Ruff integration documentation. https://docs.astral.sh/ruff/integrations/
- Pandera documentation, "DataFrame Models". https://pandera.readthedocs.io/en/stable/dataframe_models.html

**Note on evidence.** No source consulted provides a controlled measurement of type checking or
test coverage in supply chain analytics teams. Claims in section 9.3 that would require such
evidence were stated as unsupported rather than estimated. Figures in sections 6 and 13 are
illustrative and constructed for teaching; they are internally consistent but are not drawn from
a named company.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous: [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md)
- Foundation: [`01_layered-architecture.md`](01_layered-architecture.md)
- Foundation: [`02_batch-microbatch-streaming.md`](02_batch-microbatch-streaming.md)
- Foundation: [`03_ingestion-patterns.md`](03_ingestion-patterns.md)
- Next: `06_dbt-models-tests.md` — planned; not yet written.
- Later: `11_quality-quarantine-lineage.md` — planned; not yet written.
- Units prerequisite: [`../01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md`](../01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Return to the part map: [`../_index.md`](../_index.md)
