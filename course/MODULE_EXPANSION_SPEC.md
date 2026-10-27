# Module Expansion Specification

> **Purpose:** Define how each of the 48 modules is expanded from a survey file into a folder of detailed, topic-by-topic Markdown files.
>
> **Version:** 2026.5
> **Applies to:** every folder under `modules/`
> **Status:** Production spec for the expanded edition.

---

## 1. Why expand at all

The single-file modules are **survey depth**: roughly 450–700 lines covering a whole domain. That is enough to plan a course and enough for an instructor to build a lecture, but not enough for a learner to work through a topic without a second source, and not enough to answer "but how does that actually work?"

The expanded edition keeps the survey file as the **module map** and adds a folder of **topic files**. Each topic file is a self-contained teaching unit:

- It can be assigned as required reading for one 2–4 hour block.
- It contains enough worked detail that a competent learner can reproduce the reasoning.
- It names its assumptions, its failure modes, and its evidence.
- It links back to the survey file and sideways to sibling topics.

The survey file is not deleted. It becomes the index and the place where the module's narrative arc lives. Topic files are the depth.

## 2. Folder layout

```
modules/
  02_methods/                      one of seven part folders
    _index.md                      part map: purpose, modules, sequence
    07_time_series_forecasting/
      _index.md                    module map, topic table, status
      01_forecastability.md
      02_components-and-decomposition.md
      03_benchmarks.md
      04_exponential-smoothing.md
      05_arima.md
      06_regression-with-exogenous-variables.md
      07_backtesting.md
      08_error-metrics.md
      ...
      _assets/                     optional: figures, diagrams, CSV
```

Rules:

Modules sit inside one of seven part folders — `01_foundations`, `02_methods`, `03_commerce`,
`04_value_governance`, `05_labs_electives`, `06_domain_electives`, `07_engineering_electives`.
See `../modules/README.md` for the mapping. Within a part:

1. One folder per module, named `<id>_<short_slug>` using the same numeric id as the root file (`00`, `01A`, `14A`, `29`).
2. `_index.md` is mandatory. It lists every topic file, its scope, its target length, and its status.
3. Topic files are named `NN_slug.md` with a two-digit order prefix.
4. No topic file may be a stub. A file either exists at target length or does not exist and is listed as `planned`.
5. Images, if any, live in `_assets/` and are referenced relatively. Prefer text diagrams; the package must survive without binary assets.
6. The root-level survey file stays and gains a pointer to its folder.

## 3. Length target

| Item | Target | Floor |
|---|---:|---:|
| Topic file | 1,200–1,800 lines | **1,000 lines** |
| Module folder total | 10,000–18,000 lines | 8,000 lines |
| `_index.md` | 100–200 lines | 80 lines |

A line is a Markdown line, including blank lines and table rows. Do not pad with whitespace. The floor exists so that each file carries real explanation, worked examples, and exercises rather than a summary.

If a topic genuinely cannot reach 1,000 honest lines, **split it** into two topics rather than padding. If a topic runs past 2,000 lines, split it as well.

## 4. Required section template

Every topic file follows this skeleton. Headings are `##`, numbered. Sections may be expanded with `###` subsections but not removed.

```
# <Topic title>

> Module NN — <module title>
> Topic NN of MM
> Estimated study time: <n> hours
> Prerequisites: <topic ids or "none">
> Version 2026.1

## 1. Why this topic exists
## 2. Learning outcomes
## 3. The core mental model
## 4. Definitions and vocabulary
## 5. The mechanics, step by step
## 6. Worked example
## 7. Code, query, or calculation
## 8. Data, standards, and regulatory context
## 9. Industry practice and case evidence
## 10. Failure modes and red flags
## 11. Assumptions and limits
## 12. Exercises
## 13. Self-check answers
## 14. Interview questions
## 15. Key terms
## 16. Sources
## 17. Links
```

### 4.1 What belongs in each section

**1. Why this topic exists.** The industry problem, not the academic one. Who is blocked, what decision is delayed, what error is expensive. 60–120 lines.

**2. Learning outcomes.** 5–8 outcomes phrased as observable verbs ("compute", "critique", "choose", "write"). Each outcome maps to an exercise or self-check item. 40–80 lines.

**3. The core mental model.** One diagram (ASCII or Mermaid) plus prose that explains the causal or information flow. The reader should be able to redraw it from memory. 80–150 lines.

**4. Definitions and vocabulary.** A table of every term used later, with the definition, the common misuse, and the unit or coding system where relevant. 100–200 lines.

**5. The mechanics, step by step.** The longest section. Numbered procedure with the reasoning for each step, the input and output of each step, and the branch points. This is where the topic earns its length. 250–400 lines.

**6. Worked example.** One realistic case carried end to end with actual numbers, including the arithmetic. Not a toy with round numbers unless rounding is the point. 150–250 lines.

**7. Code, query, or calculation.** Runnable or copy-ready artifact: Python, SQL, shell, Terraform, or a hand calculation. Must state its assumptions and expected output. 100–220 lines.

**8. Data, standards, and regulatory context.** Which standard applies (GS1, EDI, SCOR, ISA-95, ISO, EU, U.S.), what it constrains, and what it does not. Distinguish binding requirement from convention. 120–220 lines.

**9. Industry practice and case evidence.** What companies and supply chains actually do, with named public cases where they exist. Separate documented fact from inference. 120–220 lines.

**10. Failure modes and red flags.** A table: failure, early warning sign, mechanism, consequence, mitigation. At least 10 rows. 100–180 lines.

**11. Assumptions and limits.** Explicit ledger: assumption, why it is needed, what breaks it, how to test it. At least 8 rows. 80–150 lines.

**12. Exercises.** 6–10 exercises across recall, application, critique, and design. Each states deliverable, acceptance criteria, and time. 120–200 lines.

**13. Self-check answers.** Answers to the recall and calculation items, with the reasoning. Design exercises get a rubric, not an answer. 100–180 lines.

**14. Interview questions.** 8–12 questions with what a strong answer contains and what a weak answer reveals. 100–160 lines.

**15. Key terms.** 20–40 entries, one line each, cross-linked. 40–80 lines.

**16. Sources.** Real, checkable sources: standards bodies, regulators, peer-reviewed papers with DOI, official documentation. No invented citations. If unsure of a DOI, cite the title and venue without one. 30–80 lines.

**17. Links.** Relative links to the module `_index.md`, sibling topics, the root survey file, and relevant labs. 10–30 lines.

## 5. Quality bar

A topic file passes when all of the following are true.

| # | Check | Test |
|---:|---|---|
| 1 | Length | `wc -l` ≥ 1,000 |
| 2 | Template | All 17 sections present, in order |
| 3 | Numbers | The worked example's arithmetic is correct and reproducible |
| 4 | Code | Code blocks are syntactically valid and state expected output |
| 5 | Claims | Regulatory claims are attributed and dated, not asserted |
| 6 | Case evidence | Named cases are real; inference is labelled as inference |
| 7 | Exercises | Every exercise has a deliverable and acceptance criteria |
| 8 | Links | Relative links resolve |
| 9 | Voice | Industry-facing, direct, no marketing tone |
| 10 | Honesty | Assumptions and limits are stated, not buried |

## 6. Voice and formatting

- Second person for instructions, third person for evidence.
- Short paragraphs. Tables for comparisons. Numbered steps for procedures.
- No emoji. No exclamation marks. No "revolutionary", "cutting-edge", "game-changing".
- Define an acronym at first use in every file; files are read out of order.
- Units always. A number without a unit is a defect. State the currency and the time bucket.
- Use `> **Note:**` for caveats that change interpretation.
- Use fenced code blocks with a language tag.
- Prefer an ASCII table or Mermaid diagram over a described diagram.
- Distinguish forecast, plan, target, and commitment in every example.

## 7. Cross-linking rules

Every topic file must contain:

- A link up to `_index.md`.
- Links to at least two sibling topics.
- A link to the root survey file.
- A link to at least one lab or dataset where applicable.

Every `_index.md` must contain:

- A table of topics with status.
- The module's position in the program.
- Prerequisites and what the module unlocks.
- Assessment linkage.

## 8. Status vocabulary

| Status | Meaning |
|---|---|
| `planned` | Listed in the index, no file yet |
| `draft` | File exists, below 1,000 lines or missing sections |
| `complete` | Meets the quality bar |
| `reviewed` | Complete and read by a second person |
| `revised` | Reviewed and corrected |

Only `complete` and above count toward the module's line total.

## 9. Parallel production rules

- At most **two** topic-writing agents run at once.
- Each agent owns exactly one module folder. No two agents write the same folder.
- An agent must not edit the root survey file, `README.md`, `COURSE_OUTLINE.md`, or any governance file. It writes only inside its assigned `modules/<part>/<folder>/`.
- An agent must not invent citations, reviewers, regulatory approvals, or case outcomes.
- An agent that cannot reach 1,000 honest lines on a topic must report the shortfall rather than pad.
- After an agent finishes, the index status is updated by the orchestrator, not the agent.

## 10. Verification commands

```bash
# Per-module line count
find modules/02_methods/07_time_series_forecasting -name '*.md' | xargs wc -l | tail -1

# Every topic file meets the floor
find modules -name '[0-9][0-9]_*.md' -exec sh -c 'n=$(wc -l < "$1"); [ "$n" -lt 1000 ] && echo "$1 $n"' _ {} \;

# Template section presence
for f in modules/*/*/[0-9][0-9]_*.md; do
  c=$(grep -c '^## ' "$f")
  [ "$c" -lt 17 ] && echo "$f has $c sections"
done

# Broken relative links (requires markdown-link-check or a small script)
```

## 11. What expansion does not change

- The course is still **not accredited, not vendor-certified, and not pilot-tested**.
- The regulatory and commercial content remains educational. It is not legal, trade-compliance, or procurement advice.
- Source claims remain desk-checked. See `VALIDATION_STATUS.md`.

## 12. Progress tracking

`modules/PROGRESS.md` (to be generated at the modules root) is the single source of truth for expansion status. It is regenerated from the folder contents, not hand-maintained.
