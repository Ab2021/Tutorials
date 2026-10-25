# How to Use This Package

> Module 00 — Orientation and Industry Map
> Topic 09 of 10
> Estimated study time: 2.5 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

This package is large. It contains thirty modules, each with a survey file and a folder of topic
files, plus labs, assessments, and a capstone. A learner who opens it without a plan will read
the first module, skip to the most interesting topic, run one lab, and stop. That is the normal
failure mode of self-directed technical education, and it is not a motivation problem. It is a
navigation problem.

The package has three layers, and each layer has a different purpose. The survey file gives the
arc of a module in one pass. The topic files give the depth. The labs give the practice. Reading
them in the wrong order, or using one layer as a substitute for another, produces the feeling of
having learned without the ability to do anything.

This topic gives you the map and the schedule. It explains how the layers relate, how to choose a
track, how to allocate hours, how to use the exercises and self-checks, and how to build the
portfolio that the program is designed to produce. It is the instruction manual for the rest of
the package.

### 1.1 Who this is for

Four readers use the package differently.

| Reader | Goal | Recommended entry |
|---|---|---|
| Self-directed learner | Build competence and a portfolio | Topic 01, then the track path |
| University student | Pass a course and build depth | Module survey files, then topics |
| Instructor | Teach a unit | Survey file, topic files, labs, rubrics |
| Working professional | Apply to a live project | Topic 01, 05, 07, then the relevant domain module |

The package is designed to be usable in all four modes. The difference is the sequence and the
depth, not the content.

### 1.2 What the package is not

It is not accredited, clinically validated, or regulatory approved. It does not confer a
credential recognized by a licensing body. It is a structured curriculum and a reference. The
certificate, if one is issued, attests to completion, not competence. Competence is demonstrated
by the portfolio and by performance on the assessments.

> **Note:** The program's own validation status is stated in `VALIDATION_STATUS.md` at the root.
> Read it before citing the package as evidence of anything beyond curriculum design.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Explain the three layers of the package and the purpose of each | Exercise 1; self-check 1 |
| 2 | Choose a track and a starting sequence based on prior experience and goal | Exercise 2; self-check 3 |
| 3 | Build a personal study plan with hours, weeks, and milestones | Exercise 3; self-check 5 |
| 4 | Use retrieval practice and spacing rather than rereading | Exercise 4 |
| 5 | Map each module to a portfolio artifact | Exercise 5 |
| 6 | Run a lab reproducibly from a clean environment | Exercise 6 |
| 7 | Use the self-checks and rubrics to diagnose gaps | Exercise 7 |
| 8 | Adapt the plan to a 12-week, 24-week, or corporate cohort | Exercise 8 |

## 3. The core mental model

The package is a graph, not a list. Modules depend on other modules, topics depend on sibling
topics, and labs depend on the topics that precede them. The graph is the map; the schedule is
your route through it.

### 3.1 Three layers

```text
   LAYER 1: SURVEY FILE (one per module)
   ~300-1,100 lines. The arc, vocabulary, industry context.
   Read first, in one sitting. Do not try to master it.
        |
        v
   LAYER 2: TOPIC FILES (10 or so per module)
   1,000-1,800 lines each. Mechanics, worked example, code,
   failure modes, exercises.
   Read in order within a module. Do the exercises.
        |
        v
   LAYER 3: LABS (notebooks and datasets)
   Reproducible implementation on realistic data.
   Run after the relevant topic, not before.
        |
        v
   OUTPUT: PORTFOLIO ARTIFACTS
   Charter, validation report, model card, triage memo,
   value case, capstone dossier.
```

The layers are not interchangeable. A survey file cannot teach mechanics. A topic file cannot
replace practice. A lab cannot teach judgement about whether the analysis is appropriate.

### 3.2 The dependency graph

The module dependency map in the survey file is reproduced here in simplified form.

```text
00 Orientation
 ├── 01 Clinical foundations
 ├── 01A Foundations: units, measurement, assumptions
 ├── 02 Data landscape
 │    └── 03 Data engineering and interoperability
 ├── 04 Statistics and causal inference
 │    └── 05 Study design and evidence
 ├── 06 ML foundations
 │    ├── 07 Deep learning, multimodal, LLMs
 │    ├── 09 Clinical NLP
 │    ├── 10 Imaging
 │    └── 11 Signals
 ├── 12 Omics and precision medicine
 ├── 14 Devices, SaMD, regulation
 │    └── 14A Hardware, sensors, chips, robotics
 │         └── 14B Efficient AI
 ├── 16 Privacy, security, ethics
 ├── 17 Product, economics, leadership
 └── 19 Capstone and hackathons
```

Modules are not strictly sequential. The recommended order is 00, 01, 01A, 02, 03, 04, 05, 06,
07, 08, 09, 10, 11, 12, 13, 14, 14A, 14B, 15, 16, 17, 18, 19, 20, 21, 22, then 23 to 26 as
electives. Track electives can run in parallel after the core.

### 3.3 The spiral

The program is designed as a spiral, not a line. You will encounter the same concepts at
increasing depth.

| Concept | First encounter | Second encounter | Third encounter |
|---|---|---|---|
| Validation | Module 00, topic 01 | Module 05, study design | Module 19, capstone |
| Calibration | Module 00, topic 06 | Module 06, ML foundations | Module 15, monitoring |
| Regulation | Module 00, topic 07 | Module 14, devices | Module 18, quality |
| Economics | Module 00, topic 02 | Module 17, product | Module 19, value case |
| Ethics | Module 00, topic 08 | Module 16, privacy | Capstone review |

The spiral means you do not need to master a concept on first contact. You need to recognize it,
use it at the current level, and expect to revisit it.

### 3.4 The time budget

The program is designed for 17 to 19 hours per week over 40 weeks, which is roughly 680 to 760
hours. The core modules total about 690 to 730 hours. Module 14B adds 28 hours. Advanced electives
23 to 26 add up to 98 hours when labs, case clinics, project studio, and capstone are included.

```text
   Weekly rhythm (full-time cohort)
   lecture / seminar        3 h
   lab                      3 h
   case clinic              2 h
   project studio           4 h
   independent reading      2-4 h
   assessment / portfolio   2-3 h
   ------------------------------
   total                   16-19 h per week
```

A part-time learner at 12 hours per week needs more calendar weeks, not less content. Section 6
works the arithmetic.

### 3.5 The artifact chain

Each module is designed to produce an artifact that feeds the next.

```text
00 charter  ->  03 data quality report  ->  06 risk model with calibration
   ->  05 causal protocol  ->  09 NLP evaluation  ->  10 imaging validation
   ->  14 regulatory memo  ->  15 model card and monitoring plan
   ->  17 value case  ->  19 capstone dossier
```

The chain is the portfolio. A learner who completes the modules but skips the artifacts has
learned content and has nothing to show. The artifacts are the evidence of competence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Survey file | Root-level file giving a module's arc | Treated as the whole module | Document |
| Topic file | Detailed teaching unit within a module folder | Treated as optional | Document |
| Lab | Notebook or exercise with data and expected output | Treated as a demo | Practical work |
| Module | Thematic unit with a survey file and topic folder | Confused with a lecture | Curriculum unit |
| Topic | Self-contained teaching unit | Confused with a section | Curriculum unit |
| Track | Sequence of modules and electives for a career direction | Confused with a specialty | Program path |
| Prerequisite | Module or topic required before another | Assumed to be advisory | Dependency |
| Core | Modules required for all learners | Confused with easy | Requirement |
| Elective | Optional module for depth | Confused with advanced | Requirement |
| Capstone | Final integrative project | Confused with a thesis | Assessment |
| Formative assessment | Assessment for learning, low stakes | Confused with practice | Assessment type |
| Summative assessment | Assessment of learning, graded | Confused with formative | Assessment type |
| Rubric | Criteria and levels for judging work | Confused with a checklist | Assessment tool |
| Retrieval practice | Recalling information without looking it up | Confused with rereading | Study method |
| Spacing | Distributing practice over time | Confused with cramming | Study method |
| Interleaving | Mixing related topics in practice | Confused with multitasking | Study method |
| Deliberate practice | Focused practice on weaknesses with feedback | Confused with repetition | Study method |
| Portfolio | Collection of artifacts demonstrating competence | Confused with a CV | Artifact set |
| Artifact | Tangible output of work | Confused with a certificate | Deliverable |
| Cohort variant | Schedule adaptation for a specific group | Confused with a different curriculum | Delivery mode |
| Case clinic | Session critiquing a real product or decision | Confused with a lecture | Session type |
| Project studio | Team work session with critique | Confused with a lab | Session type |
| Competency framework | Structured description of skills and levels | Confused with a syllabus | Framework |
| CPD | Continuing professional development | Confused with initial training | Professional requirement |
| Microcredential | Short, assessed credential | Confused with a degree | Credential |
| Accreditation | Formal recognition of a program | Confused with certification | Status |
| Learning objective | Observable outcome of a unit | Confused with a topic title | Statement |
| Worked example | Fully solved problem with reasoning | Confused with an exercise | Teaching device |
| Self-check | Questions with answers for self-assessment | Confused with a graded quiz | Learning device |
| Spiral curriculum | Revisiting concepts at increasing depth | Confused with repetition | Design principle |
| Cognitive load | Mental effort required to process information | Confused with difficulty | Psychological construct |
| Transfer | Applying learning in a new context | Confused with recall | Learning outcome |
| Fidelity | Degree to which a lab resembles real work | Confused with accuracy | Design property |

Two definitions matter for planning.

**Formative versus summative.** The exercises and self-checks are formative. They are designed to
show you what you do not yet know. The midterm dossier, track playbook, and capstone are
summative.

**Transfer.** The goal is not to recall the worked example. It is to apply the method to a new
problem. A learner who can reproduce the sepsis example but cannot structure a new charter has
not achieved transfer.

## 5. The mechanics, step by step

This section is a twelve-step procedure for turning the package into a personal plan. It assumes
you have read topics 01 and 02 and have a rough direction.

### Step 1: Audit your available time honestly

**Input:** your calendar.
**Output:** hours per week available for study, in three scenarios: optimistic, realistic, minimum.

**Reasoning.** Plans fail on time, not on motivation. An honest audit uses the last four weeks of
actual behavior, not the schedule you wish you had.

**Branch point.** If the realistic figure is below 8 hours per week, choose a single track and a
reduced module set. Do not attempt the full core.

### Step 2: State the goal in artifact terms

**Input:** career direction.
**Output:** one sentence describing the artifact you want to be able to produce.

**Reasoning.** "Learn machine learning" is not a goal. "Produce a validated risk model with
calibration and a monitoring plan on a public cohort" is a goal that determines which modules and
labs matter.

**Branch point.** If the goal is a promotion or a role change, identify the artifact that the
target role would recognize. That is the capstone direction.

### Step 3: Choose a track

**Input:** goal and current role.
**Output:** a track and a primary elective set.

**Reasoning.** The three tracks (health systems analytics, biopharma and translational, medtech,
sensors and imaging) share the core and diverge in modules 08 to 14. Choosing early prevents
wasted electives.

**Branch point.** If you are unsure, choose the track closest to your current access to data. Data
access is the constraint that most often determines what you can actually build.

### Step 4: Map prerequisites

**Input:** track and module list.
**Output:** a dependency-ordered list with any gaps marked.

**Reasoning.** Skipping a prerequisite does not save time; it moves the cost into the next module.
The dependency graph in section 3.2 is the reference.

**Branch point.** If a prerequisite is missing, either take it or plan a bridge week using the
survey file and one lab.

### Step 5: Allocate hours by activity

**Input:** total hours and module list.
**Output:** hours per module and per activity.

**Reasoning.** Reading alone does not produce competence. A balanced allocation is roughly 30%
reading, 35% labs, 25% project work, and 10% assessment. The exact split matters less than
including all four.

**Branch point.** If you have less than 10 hours per week, reduce reading first and keep labs.
Reading without practice produces familiarity, not skill.

### Step 6: Build the weekly cadence

**Input:** weekly hours.
**Output:** a repeating weekly schedule with fixed blocks.

**Reasoning.** Fixed blocks survive busy weeks. A plan that depends on finding time fails in the
first month. The cadence should include one block for retrieval practice, not only new material.

**Branch point.** If your week is fragmented, use two shorter blocks rather than one long one.
Spacing is more effective than massing.

### Step 7: Read the survey file first, once, quickly

**Input:** module.
**Output:** a one-page map of the module in your own words.

**Reasoning.** The survey file gives the arc and the vocabulary. Reading it quickly, without
trying to memorize, creates a scaffold that makes the topic files easier to place.

**Branch point.** If the survey file feels sufficient, resist the temptation to skip the topics.
The survey is the map, not the territory.

### Step 8: Work the topic files in order, with exercises

**Input:** topic files.
**Output:** completed exercises and self-check notes.

**Reasoning.** The exercises are where transfer happens. Reading a worked example produces
recognition; solving an exercise produces retrieval. The self-checks diagnose gaps.

**Branch point.** If an exercise takes more than twice its stated time, stop and reread the
mechanics section. The exercise is testing a prerequisite you have not consolidated.

### Step 9: Run the lab after the relevant topic

**Input:** lab notebook.
**Output:** a reproducible run with your own notes and one extension.

**Reasoning.** The lab converts concepts into operational skill. Running it once is not enough;
change one parameter, break it, and fix it. That is where understanding consolidates.

**Branch point.** If the lab fails in a clean environment, fix the environment before proceeding.
Reproducibility is a core competency, not an IT issue.

### Step 10: Schedule spaced review

**Input:** completed topics.
**Output:** a review calendar at 1 week, 4 weeks, and 12 weeks.

**Reasoning.** Forgetting is predictable. Spaced retrieval, not rereading, is the efficient
response. A 15-minute retrieval session at one week is worth more than a 60-minute reread.

**Branch point.** If a review reveals a gap, return to the topic's exercises rather than
rereading the whole file.

### Step 11: Build the artifact at the end of each module

**Input:** module content and exercises.
**Output:** one portfolio artifact per module.

**Reasoning.** The artifact is the transfer test. It also becomes the evidence for the capstone
and for job applications.

**Branch point.** If the artifact cannot be made public because of data restrictions, produce a
public-safe version with synthetic or open data and describe the restricted work separately.

### Step 12: Review the plan monthly and adjust

**Input:** actual progress.
**Output:** a revised plan with slippage absorbed explicitly.

**Reasoning.** Plans drift. A monthly review catches drift before it becomes abandonment. The
adjustment should be explicit: which module is delayed, which elective is dropped, what the new
completion date is.

**Branch point.** If you are more than three weeks behind, cut scope rather than hours of sleep.
Scope cuts are sustainable; sleep cuts are not.

### 5.1 Reading a topic file efficiently

A topic file is long by design. It is not meant to be read once, linearly, at full attention. A
three-pass method works better.

**Pass 1, 15 minutes.** Read section 1, section 3, and section 10. You now know why the topic
exists, the mental model, and the failure modes. Decide whether to continue now or schedule it.

**Pass 2, 60 to 90 minutes.** Read sections 2, 4, 5, and 6. Work the worked example on paper
before reading the solution.

**Pass 3, 45 minutes.** Do section 7 in a notebook, then the exercises in section 12. Check
against section 13. Read section 14 aloud or write answers to three interview questions.

This method uses retrieval and spacing, and it keeps the reading load manageable.

### 5.2 Using the exercises

| Exercise type | Purpose | How to know you are done |
|---|---|---|
| Recall | Consolidate vocabulary and definitions | You can answer without looking |
| Application | Transfer a method to a new case | The deliverable exists and meets criteria |
| Critique | Develop judgement | You can name what is wrong and why |
| Design | Integrate and decide | A reviewer can act on your artifact |

The acceptance criteria are the contract. If you meet them, you are done, even if the answer
differs from the self-check. If you do not meet them, the exercise is not finished.

### 5.3 Using the labs

| Lab | Module | What it trains |
|---|---|---|
| `01A_unit_conversion.ipynb` | 01A | Units, dimensional analysis, measurement error |
| `03_cohort_pipeline.ipynb` | 03 | Cohort construction, data quality, provenance |
| `06_calibration_decision_curve.ipynb` | 06 | Calibration, decision curves, threshold choice |
| `11_ecg_quality.ipynb` | 11 | Signal quality, artefact detection |
| `14B_quantization.ipynb` | 14B | Model compression, latency, accuracy trade-off |

Run each lab in a clean environment before modifying it. Record the environment, the data version,
and the output. A lab that only works on your machine is not reproducible.

### 5.4 Cohort variants

| Variant | Duration | Trade-off | Best for |
|---|---|---|---|
| Full-time, 40 weeks | 40 weeks at 17-19 h/week | Broadest coverage | Career changers with runway |
| 12-week intensive | 12 weeks at 35-45 h/week | Less modality breadth | Teams with a deadline |
| 24-week part-time | 24 weeks at 20-25 h/week | Less deep omics and imaging | Working professionals |
| Corporate cohort | 8-12 weeks, live workshops | Tailored to one use case | Organizations with a real project |

The 12-week variant compresses modules 00 to 04 into four weeks, 05 to 08 plus one modality into
four weeks, and 14 to 18 plus a scoped capstone into four weeks. The compression is real; some
depth is lost.

### 5.5 Assessment linkage

Each assessment draws on specific modules. Knowing the linkage prevents studying the wrong
material before a deadline.

| Assessment | Weight | Modules that feed it | Artifact |
|---|---:|---|---|
| Weekly labs | 20% | 01A, 03, 06, 11, 14B and module labs | Lab notebooks |
| Case critiques | 15% | 00, 05, 06, 07 | Written critiques |
| Midterm evidence dossier | 20% | 04, 05, 06, 07, 14 | Dossier |
| Track playbook | 15% | 08-14 track modules, 17 | Playbook |
| Capstone | 30% | All core, plus track electives | Dossier and defense |

Passing requires at least 50% on the individual capstone defense, regardless of the team score.
That rule is stated in the survey file and is worth designing for from the first module: keep an
individual record of your contributions, because the defense is individual.

### 5.6 Ten questions learners ask

**"Do I need to read every topic file?"** For the core modules, yes, if you want the competence
the module claims. For electives, read the survey and the topics that bear on your artifact.

**"Can I skip the maths?"** No. The arithmetic in the worked examples is the minimum. If you
cannot reproduce it by hand, you cannot audit a vendor's claim or defend your own.

**"How much programming do I need?"** Enough Python to read and modify a notebook, enough SQL to
join and filter tables, and enough Git to version your work. Module 01A and the labs build the
rest.

**"Is the certificate worth anything?"** It evidences completion, not competence. The portfolio is
what employers assess.

**"How long should a topic take?"** The stated study time is 2.5 to 3.5 hours for a first pass.
Exercises add 30 to 45 minutes each. A deep pass with the lab takes twice as long.

**"What if I disagree with the material?"** Write the disagreement down and test it. The topics
state assumptions explicitly so that they can be challenged. A reasoned disagreement is a good
sign of engagement.

**"Can I use an LLM to help?"** Yes, for explanation and critique, under the policy in the survey
file. You remain accountable for every claim and every line of code, and you must not paste
identifiable data into a public model.

**"What if my data are restricted?"** Build the public-safe version with open or synthetic data and
describe the restricted work in an appendix without sharing the data.

**"How do I know I am ready for the capstone?"** You have a charter, a validated model with
calibration, a monitoring plan, and a regulatory triage memo. If one is missing, the capstone will
expose it.

**"What if I fall behind?"** Cut scope explicitly, protect the labs, and revise the completion
date. Do not cut sleep or the artifact work. A revised plan with a later date is still a plan; an
abandoned plan is not. That is a schedule decision, not a failure.

A revised plan is still a plan. The failure mode is not the delay; it is the silent abandonment
that follows an unrevised schedule.

## 6. Worked example

A working professional has 12 hours per week and wants to complete the core program. This example
computes the schedule, identifies the shortfall, and adjusts.

### 6.1 Inputs

| Input | Value | Unit |
|---|---:|---|
| Available study time | 12 | hours per week |
| Core program hours (midpoint) | 710 | hours |
| Elective hours (modules 23-26) | 98 | hours |
| Buffer for illness and work travel | 10 | % of study weeks |
| Desired completion | as soon as feasible | date |

### 6.2 Straight calculation

```text
Weeks for core = 710 / 12 = 59.2 weeks
With 10% buffer = 59.2 x 1.10 = 65.1 weeks
Completion from 2 November 2026:
  65.1 weeks = 456 days
  2 November 2026 + 456 days = 1 February 2028
```

A 40-week target at 12 hours per week would provide 480 hours, which is 230 hours short of the
710-hour core. The shortfall is 32% of the program. The plan is not feasible as stated.

### 6.3 Options

| Option | Hours available | Weeks | Trade-off |
|---|---:|---:|---|
| Increase to 14 h/week | 14 x 40 = 560 | 710 / 14 = 50.7, with buffer 55.8 | Requires 2 more hours per week |
| Increase to 17 h/week | 680 over 40 weeks | 710 / 17 = 41.8, with buffer 45.9 | Close to full-time load |
| Drop electives | 710 core only | 65.1 with buffer | No elective depth |
| Take core plus one elective track | 710 + 25 = 735 | 735 / 12 = 61.2, with buffer 67.4 | Longer but broader |
| Reduce to a focused track | 480 hours of selected modules | 480 / 12 = 40, with buffer 44 | Narrower credential |

The realistic choice for a working professional is 14 hours per week with a focused elective set,
completing in about 56 weeks, or 12 hours per week over 65 weeks with the full core.

### 6.4 Weekly allocation at 12 hours

| Activity | Share | Hours per week |
|---|---:|---:|
| Reading topic files | 30% | 3.6 |
| Labs | 35% | 4.2 |
| Project and artifact work | 25% | 3.0 |
| Assessment and review | 10% | 1.2 |
| **Total** | **100%** | **12.0** |

### 6.5 A four-week sample schedule

| Week | Reading | Lab | Artifact | Review |
|---|---|---|---|---|
| 1 | Module 00, topics 01-03 | None | Charter draft | 30 min retrieval |
| 2 | Module 00, topics 04-06 | None | Charter v1 | 30 min retrieval |
| 3 | Module 01, survey and topics 01-02 | Unit conversion | Vocabulary map | 45 min retrieval |
| 4 | Module 01, topics 03-05 | Cohort pipeline | Data quality note | 45 min retrieval |

The schedule alternates reading-heavy and lab-heavy weeks. That alternation is intentional: it
spaces practice and prevents the fatigue that comes from a month of reading.

### 6.6 Completion arithmetic for the adjusted plan

```text
Adjusted hours per week = 14
Core hours = 710
Weeks = 710 / 14 = 50.7
Buffer = 10% -> 55.8 weeks
Start 2 November 2026 + 55.8 weeks (391 days) = 28 November 2027
```

The adjusted plan finishes about nine weeks earlier than the 12-hour plan and requires two
additional hours per week. Whether that trade is worth it depends on the deadline, which is a
personal constraint, not a curriculum one.

## 7. Code, query, or calculation

The block computes weeks to completion, activity allocation, and scenario comparisons. Standard
library only.

```python
"""Study plan arithmetic for the program."""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Plan:
    name: str
    hours_per_week: float
    program_hours: float
    buffer: float = 0.10
    start: date = date(2026, 11, 2)


def schedule(p: Plan) -> dict:
    raw_weeks = p.program_hours / p.hours_per_week
    weeks = raw_weeks * (1 + p.buffer)
    days = round(weeks * 7)
    end = p.start + timedelta(days=days)
    return {
        "raw_weeks": raw_weeks,
        "weeks": weeks,
        "end": end,
        "total_hours": p.hours_per_week * weeks,
    }


def allocation(hours_per_week: float) -> dict:
    return {
        "reading": hours_per_week * 0.30,
        "labs": hours_per_week * 0.35,
        "project": hours_per_week * 0.25,
        "assessment": hours_per_week * 0.10,
    }


CORE = 710.0
plans = [
    Plan("12 h/week, core", 12, CORE),
    Plan("14 h/week, core", 14, CORE),
    Plan("17 h/week, core", 17, CORE),
    Plan("12 h/week, core + 1 elective", 12, CORE + 25),
]

for p in plans:
    s = schedule(p)
    print(f"{p.name:32s} raw {s['raw_weeks']:5.1f} wk, "
          f"buffered {s['weeks']:5.1f} wk, end {s['end'].isoformat()}, "
          f"total {s['total_hours']:6.0f} h")

print("\nAllocation at 12 h/week")
for k, v in allocation(12).items():
    print(f"  {k:12s} {v:4.1f} h")

print("\nShortfall against a 40-week target at 12 h/week")
available = 12 * 40
print(f"  available {available:.0f} h; required {CORE:.0f} h; "
      f"shortfall {CORE - available:.0f} h ({(CORE - available) / CORE:.1%})")

print("\nWeeks required at various weekly hours (no buffer)")
print(f"{'hours/week':>11s} {'weeks':>7s}")
for h in (8, 10, 12, 14, 17, 20):
    print(f"{h:11d} {CORE / h:7.1f}")
```

Expected output:

```text
12 h/week, core                  raw  59.2 wk, buffered  65.1 wk, end 2028-02-01, total    781 h
14 h/week, core                  raw  50.7 wk, buffered  55.8 wk, end 2027-11-28, total    781 h
17 h/week, core                  raw  41.8 wk, buffered  45.9 wk, end 2027-09-20, total    781 h
12 h/week, core + 1 elective     raw  61.2 wk, buffered  67.4 wk, end 2028-02-17, total    808 h

Allocation at 12 h/week
  reading       3.6 h
  labs          4.2 h
  project       3.0 h
  assessment    1.2 h

Shortfall against a 40-week target at 12 h/week
  available 480 h; required 710 h; shortfall 230 h (32.4%)

Weeks required at various weekly hours (no buffer)
 hours/week   weeks
          8    88.8
         10    71.0
         12    59.2
         14    50.7
         17    41.8
         20    35.5
```

Check the buffer arithmetic. 59.2 weeks times 1.10 equals 65.1 weeks. 65.1 weeks times 7 equals
456 days. 2 November 2026 plus 456 days is 1 February 2028. The total hours column is 12 times
65.1, which equals 781 hours, more than the 710 required because the buffer is included.

> **Note:** The buffer is applied to weeks, not to content. The extra hours are contingency, not
> additional curriculum. If no contingency is needed, the plan finishes earlier.

## 8. Data, standards, and regulatory context

The package is a curriculum, not a regulated product. The relevant standards are educational and
professional, not device or privacy law. Three frameworks are useful for mapping the curriculum
to external expectations.

| Framework | Issuer | What it provides | Relevance |
|---|---|---|---|
| National Competency Framework for Data Professionals in Health and Care | NHS England | Competencies for data analysis, engineering, and science across five levels | Maps module content to role expectations |
| Australian Health Informatics Competency Framework, 2nd edition (2025) | Australasian Institute of Digital Health | Six domains, competencies levelled by Bloom's taxonomy | Used for CHIA certification |
| DECODE framework (JAMA Network Open, 2025) | International Delphi consensus | Four domains, 19 competencies for digital health in medical education | Health data science domain aligns with this program |
| ICMJE recommendations | ICMJE | Authorship and reporting | Applies to portfolio publications |
| ACCME / CPD requirements | Accrediting bodies | Continuing education credit | Relevant if the package is used for CPD |

### 8.1 The DECODE framework

The Digital Health Competencies in Medical Education (DECODE) framework was published in JAMA
Network Open in January 2025, based on a Delphi consensus with 211 experts from 79 countries. It
defines four domains, 19 competencies, and 33 mandatory plus 145 discretionary learning outcomes.
Domain 4 is health data science, covering public health informatics, AI in healthcare,
computational thinking, and precision medicine.

The framework is aimed at undergraduate medical education, not at data science training. It is
useful as a checklist of clinical context that a data scientist should be able to recognize.

### 8.2 The NHS competency framework

The National Competency Framework for Data Professionals in Health and Care describes
competencies across five practitioner levels, from associate to leading practitioner, with
optional specialisms such as machine learning and advanced statistics. It is used for appraisal,
recruitment, and career pathways in the UK health system.

For a learner, the framework is useful for translating module content into the language of a job
description. "I completed Module 06" is less persuasive than "I can build, calibrate, and monitor
a prediction model, which maps to the data science practitioner competencies at level 3 and 4".

### 8.3 Accreditation and credit

This package is not accredited by a recognized educational body and does not confer academic
credit. If you need credit, check whether a partner institution offers the package for credit and
what its requirements are. Do not represent completion as accreditation.

> **Note:** The absence of accreditation is stated in the root README and in the validation
> status file. It is a feature of an open curriculum, not a defect, but it matters for how you
> describe the credential.

### 8.4 CPD and professional registration

Some professional bodies accept structured learning for continuing professional development. The
requirements differ by body and jurisdiction. Keep a log of hours, topics, and artifacts if you
intend to claim CPD. The portfolio is the natural evidence.

## 9. Industry practice and case evidence

### 9.1 What the learning science says

**Documented.** A 2025 meta-analysis by Mawson and Kang in Behavioral Sciences, covering 22
classroom-based studies with more than 3,000 participants, reported a moderate effect favoring
distributed over massed practice, d = 0.54 (95% CI 0.31 to 0.77), with high heterogeneity
(I-squared approximately 92%).

**Documented.** A 2025 meta-analysis by Murray, Horner, and Gobel in Educational Psychology
Review, covering 27 studies and 53 effect sizes, reported a small-to-medium effect of spacing on
mathematics learning, g = 0.28. The effect for isolated materials was g = 0.43 and for
course-embedded materials g = 0.24. The testing effect was not robust in mathematics, with a
confidence interval crossing zero.

**Inference.** Spacing works, but the effect size depends on the domain and the implementation.
For a technical curriculum, the practical implication is to distribute practice across weeks
rather than concentrate it before assessments, and to use retrieval rather than rereading even
where the evidence is mixed. The evidence does not support a specific schedule; it supports not
massing.

### 9.2 What competency frameworks add

**Documented.** The DECODE framework (JAMA Network Open, January 2025) and the NHS National
Competency Framework both organize skills by domain and level rather than by course. The NHS
framework includes a points-based progression model and maps to the UK Government Digital, Data
and Technology (DDaT) framework.

**Inference.** Employers increasingly use competency language, not course lists. A portfolio that
maps artifacts to competencies travels better than a transcript. This is the practical reason the
package is artifact-driven.

### 9.3 Completion versus competence

**Inference.** There is no public dataset comparing completion of biomedical data science
programs to job performance. The available proxies, such as portfolio review in hiring, suggest
that artifacts matter more than certificates. A learner should optimize for artifacts.

### 9.4 How employers read a portfolio

**Inference, from public hiring practice.** A portfolio is read in three passes. First, does the
candidate have a project with a real decision and real data? Second, is the validation credible?
Third, can the candidate explain the limitations? A portfolio that answers all three in ten
minutes is stronger than a longer portfolio that requires an hour.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reading without practice | Highlights but no artifacts | Passive learning | Familiarity without skill | Labs and exercises first |
| Skipping prerequisites | Confusion in module 06 | Gap in foundations | Repeated failure | Bridge week |
| Over-optimistic time budget | Plan assumes 20 h/week | Planning fallacy | Abandonment | Use actual hours |
| Massing before assessment | All study in final week | Cramming | Poor retention | Spaced schedule |
| Course collecting | Many certificates, no project | Credential chasing | Weak portfolio | One deep artifact |
| Tool chasing | New library every month | Novelty seeking | Shallow skill | Depth in two tools |
| No review cycle | No retrieval after week 1 | Forgetting curve | Wasted study | 1/4/12 week reviews |
| Ignoring rubrics | Work submitted without criteria | Misreading expectations | Low scores | Read rubric first |
| Unreproducible labs | Works only on one machine | Environment drift | Can't demonstrate | Clean environment, pinned versions |
| Private-data portfolio | Nothing can be shown | Confidentiality | No evidence | Public-safe version |
| No cohort or feedback | Studying alone | No correction | Blind spots | Case clinic or peer review |
| Plan never revised | Same schedule after slippage | Rigidity | Guilt and abandonment | Monthly review |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The hour estimates are accurate | Scheduling | Different reading speed | Track actual hours for two weeks |
| Modules are independent enough to reorder | Flexibility | Hard prerequisites | Check dependency graph |
| Labs run in a standard environment | Reproducibility | Version conflicts | Clean install test |
| Artifacts can be made public-safe | Portfolio | Restricted data | Synthetic version |
| Learners have baseline programming | Pace | No Python or SQL | Take a readiness check |
| Feedback is available | Skill correction | Studying alone | Join a cohort |
| The track maps to the target role | Career relevance | Role ambiguity | Interview three practitioners |
| Assessment rubrics are applied consistently | Fairness | Grader variability | Calibration session |

### 11.1 Limits

The hour estimates are planning figures, not measurements. Individual pace varies by background
and by the depth at which a topic is studied.

The package cannot substitute for clinical experience, regulatory practice, or supervised
research. It provides the vocabulary and the methods to collaborate, not the license to practice.

The learning-science evidence cited in section 9 is about general learning, not about this
specific curriculum. It supports the design choices; it does not validate the program.

## 12. Exercises

### Exercise 1: Map the layers (recall, 20 minutes)

**Deliverable.** A one-page diagram of the three layers and the artifact chain, in your own words.

**Acceptance criteria.** All three layers present. At least six artifacts in the chain. No copied
diagram.

**Time.** 20 minutes.

### Exercise 2: Choose a track (application, 30 minutes)

**Deliverable.** A one-page justification of a track and three electives, with the role it targets.

**Acceptance criteria.** Names the target role and the artifact that role would recognize. States
the data access you have. Identifies one risk to the plan.

**Time.** 30 minutes.

### Exercise 3: Personal study plan (design, 45 minutes)

**Deliverable.** A 12-week plan with weekly hours, modules, labs, and artifacts.

**Acceptance criteria.** Uses actual available hours. Includes buffer. Every week has a
deliverable. The plan is feasible against the hour estimates.

**Time.** 45 minutes.

### Exercise 4: Retrieval schedule (application, 20 minutes)

**Deliverable.** A review calendar for four topics at 1, 4, and 12 weeks, with the questions you
will use.

**Acceptance criteria.** Dates specific. Questions require recall, not recognition. At least one
question per topic is a calculation.

**Time.** 20 minutes.

### Exercise 5: Artifact map (application, 30 minutes)

**Deliverable.** A table mapping eight modules to the artifact each will produce and the audience
for each artifact.

**Acceptance criteria.** Each artifact has an audience and a public-safe alternative. No artifact
is "notes".

**Time.** 30 minutes.

### Exercise 6: Reproducible lab run (application, 45 minutes)

**Deliverable.** A clean-environment run of one lab, with a record of the environment, data
version, and output, plus one modification.

**Acceptance criteria.** Environment documented. Output reproducible. Modification explained and
its effect stated.

**Time.** 45 minutes.

### Exercise 7: Diagnose with self-checks (application, 25 minutes)

**Deliverable.** A list of five self-check questions you answered incorrectly, with the concept
each reveals and the topic section to revisit.

**Acceptance criteria.** Specific questions. Concepts named. Sections identified.

**Time.** 25 minutes.

### Exercise 8: Adapt to a cohort variant (design, 30 minutes)

**Deliverable.** A 12-week intensive plan that preserves the core and drops the least essential
content, with the trade-offs stated.

**Acceptance criteria.** Core preserved. At least three trade-offs named. Weekly hours realistic
for the variant.

**Time.** 30 minutes.

## 13. Self-check answers

**1. What are the three layers?**
Survey files, topic files, and labs. The survey gives the arc, the topics give depth, the labs
give practice.

**2. Why read the survey file first?**
It provides the scaffold and vocabulary that make the topic files easier to place. It is a map,
not a substitute.

**3. How do you choose a track?**
By the artifact the target role recognizes and by the data you can actually access.

**4. Why is reading without practice insufficient?**
Because recognition is not retrieval. Practice, especially retrieval and application, is what
produces transferable skill.

**5. If you have 12 hours per week and the core is 710 hours, how many weeks?**
710 / 12 = 59.2 weeks before buffer; 65.1 weeks with a 10% buffer.

**6. What is the shortfall against a 40-week plan at 12 hours per week?**
480 hours available versus 710 required, a shortfall of 230 hours, or 32.4% of the core.

**7. Why schedule review at 1, 4, and 12 weeks?**
Because forgetting is predictable and spaced retrieval is more efficient than rereading. The
intervals are a practical default, not a precise optimum.

**8. What makes a lab reproducible?**
A documented environment, pinned versions, a data version, and an output that can be regenerated
from a clean install.

**9. What is the difference between completion and competence?**
Completion is finishing the content. Competence is producing an artifact that meets external
criteria. The portfolio is the evidence of the latter.

**10. Why is the buffer applied to weeks rather than content?**
Because the buffer is contingency for disruption, not additional material. Applying it to weeks
keeps the curriculum fixed and makes the schedule honest.

### 13.1 Rubric for exercises 3 and 6

Exercise 3 is scored on realism of hours (0 to 3), presence of deliverables (0 to 3), and buffer
and review (0 to 2). A plan with no buffer scores at most 6.

Exercise 6 is scored on reproducibility (0 to 4), documentation (0 to 3), and quality of the
modification (0 to 3).

## 14. Interview questions

**1. "How did you learn this material?"**
Strong: describes a structured plan, artifacts, and feedback. Weak: "I took some courses".

**2. "How do you keep up with the field?"**
Strong: specific sources, a review cadence, and a filter for relevance. Weak: "I read Twitter".

**3. "How would you get up to speed on a new domain in three months?"**
Strong: survey the field, identify the decision, build one artifact, find a practitioner.
Weak: "read everything".

**4. "What is in your portfolio?"**
Strong: names artifacts, decisions, and limitations. Weak: lists courses.

**5. "How do you know when you understand something?"**
Strong: can teach it, apply it to a new case, and predict where it fails. Weak: "when I can
reproduce the example".

**6. "How do you prioritize learning when time is short?"**
Strong: by the artifact needed and the binding constraint. Weak: "start at the beginning".

**7. "How do you use feedback?"**
Strong: seeks specific critique, revises, records the change. Weak: "I take it on board".

**8. "What would you do if you were behind schedule?"**
Strong: cut scope explicitly, protect practice, revise the plan. Weak: "work more hours".

**9. "How do you evaluate a course or a credential?"**
Strong: looks at artifacts produced, assessment rigor, and relevance to the role. Weak: looks at
the brand.

**10. "How do you avoid tool chasing?"**
Strong: depth in two tools, learn concepts that transfer. Weak: "I try everything".

**11. "What makes a portfolio credible?"**
Strong: real decisions, credible validation, honest limitations, public-safe data. Weak: high
accuracy numbers.

**12. "How would you teach this to a colleague?"**
Strong: starts with the decision, uses a worked example, gives an exercise. Weak: starts with
theory.

## 15. Key terms

- **Accreditation** — formal recognition of a program.
- **Artifact** — tangible output of work.
- **Capstone** — final integrative project.
- **Case clinic** — session critiquing a real product or decision.
- **Cognitive load** — mental effort required to process information.
- **Cohort variant** — schedule adaptation for a group.
- **Competency framework** — structured description of skills and levels.
- **Core** — required modules.
- **CPD** — continuing professional development.
- **Deliberate practice** — focused practice with feedback.
- **Elective** — optional module.
- **Formative assessment** — assessment for learning.
- **Interleaving** — mixing related topics in practice.
- **Lab** — practical work with data.
- **Learning objective** — observable outcome.
- **Microcredential** — short assessed credential.
- **Module** — thematic curriculum unit.
- **Portfolio** — collection of artifacts.
- **Prerequisite** — required prior module.
- **Project studio** — team work session with critique.
- **Retrieval practice** — recall without looking things up.
- **Rubric** — criteria and levels for judging work.
- **Self-check** — questions with answers for self-assessment.
- **Spacing** — distributing practice over time.
- **Spiral curriculum** — revisiting concepts at increasing depth.
- **Summative assessment** — assessment of learning.
- **Survey file** — module arc document.
- **Topic** — self-contained teaching unit.
- **Track** — sequence for a career direction.
- **Transfer** — applying learning in a new context.
- **Worked example** — fully solved problem with reasoning.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- Mawson, L., Kang, S. H. K., "The Distributed Practice Effect on Classroom Learning: A
  Meta-Analytic Review of Applied Research," *Behavioral Sciences*, 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12189222/
- Murray, S., Horner, S., Gobel, S. M., "A Meta-analytic Review of the Effectiveness of Spacing
  and Retrieval Practice for Mathematics Learning," *Educational Psychology Review*, 2025.
  https://doi.org/10.1007/s10648-025-10035-1
- Latimier, A., Peyre, H., Ramus, F., "A Meta-Analytic Review of the Benefit of Spacing out
  Retrieval Practice Episodes on Retention," *Educational Psychology Review*, 2021.
  https://doi.org/10.1007/s10648-020-09572-8
- NHS England, National Competency Framework for Data Professionals in Health and Care.
  https://transform.england.nhs.uk/media/documents/NCF_Framework_Booklet.pdf
- Australasian Institute of Digital Health, Australian Health Informatics Competency Framework,
  2nd edition, March 2025.
  https://digitalhealth.org.au/chia/wp-content/uploads/sites/37/2025/03/Second-Edition_AHICF_March-2025.pdf
- Digital Health Competencies in Medical Education (DECODE) framework, *JAMA Network Open*,
  January 2025 (Delphi consensus, 211 experts, 79 countries).
- NHS England, NHS Data and Analytics Academy.
  https://transform.england.nhs.uk/key-tools-and-info/nhsx-analytics-unit/data-and-analytics-partnership-gateway/nhs-data-and-analytics-academy/
- Public Health Foundation, Data Science and Informatics Training.
  https://phf.org/tools-resources/data-science-and-informatics-training/
- International Committee of Medical Journal Editors, Recommendations.
  https://www.icmje.org/recommendations/

> **Note:** The learning-science studies cited here concern general classroom learning, not this
> curriculum. They support design choices; they do not validate the program.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_ethics-and-professional-conduct.md`](08_ethics-and-professional-conduct.md)
- Next topic: [`10_success-metrics-and-portfolio.md`](10_success-metrics-and-portfolio.md)
- Related topic: [`04_career-tracks-and-role-definitions.md`](04_career-tracks-and-role-definitions.md)
- Related topic: [`05_the-project-charter.md`](05_the-project-charter.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../course/00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Unit conversion lab: [`../../labs/01A_unit_conversion.ipynb`](../../labs/01A_unit_conversion.ipynb)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
