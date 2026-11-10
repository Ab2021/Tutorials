# Agent authoring contract (resumable, per-subtree)

**Purpose:** single source of truth for every authoring worker in this course. Read fully before writing. This file is integration-owned: workers MUST NOT edit it, `INDEX.md`, `README.md`, `COVERAGE_LEDGER.*`, or `00_*` files at the course root.

**Status vocabulary used below:** MUST / NEVER are absolute; SHOULD is the default path; MAY is optional.

## 0. Hard anti-padding rule (the defect this contract exists to stop)

Previous passes produced files of 1,015 physical lines by repeating one sentence up to 900 times (`Operational note 879: ...`). That is **rejected**. Acceptance is by *distinct* content, not physical length.

- Distinct-content floor: **at least 700 distinct non-blank prose lines** and **≥1,000 physical lines** per case and per cheat sheet.
- A long line (≥70 characters) MUST NOT appear more than **twice** in a file.
- Duplicate long-line ratio MUST stay **below 0.15** (`1 - distinct_long_lines / total_long_lines`).
- NEVER emit numbered filler ("Operational note N", "Check N", "Step N") as a way to grow length. If you cannot think of substantive content, write fewer sections' worth of *real* reasoning — then continue the topic in the next file, not filler.
- Every section MUST add content that a practitioner could act on: a decision, a number, a mechanism, a diagnostic, or a counterexample.

Verify with the lint tool after **every** file:

```bash
python "D:/Biomedical/Train/ComprehensiveAIML_BUSINESS_DOMAINS/Tools/manuscript_lint.py" "<file.md>"
```

Exit code 0 = pass. Fix and re-run before moving on. Do not report a file as done while lint fails.

## 1. What to author, in what order

Work strictly inside your assigned subtree. Order:

1. **Cheat sheets first** (they carry the theory the cases cite), foundation → core → advanced, then the final Common Issues file.
2. **Case studies** in catalog order.
3. **Collection indexes last** (`Case_Studies/INDEX.md`, `Cheat_Sheets/INDEX.md`, and the unit `INDEX.md` status table) so links and statuses are true.

Each teaching unit is `<leaf>/Case_Studies/` + `<leaf>/Cheat_Sheets/`. A unit is complete only when:

- every case row promised in `Case_Studies/INDEX.md` either has a real linked `.md` file, or is honestly re-scoped with a stated reason in the index (no silent deletion);
- `Cheat_Sheets/` contains at least **3** topic sheets plus **`NN_25_Common_Issues.md`**;
- both collection indexes carry their full required section sets (7 and 11) with real links;
- the unit `INDEX.md` topic table statuses reflect reality (`authored` / `planned`), not aspiration.

## 2. Case study structure (exactly these H2 headings, in order)

```
---
case_id: "CASE-XX-YY-NN"
case_title: "..."
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical|predictive-ml|llm-agentic|streaming-systems|incident|hybrid"
domain: "..."
primary_pattern: "..."
secondary_patterns: ["...", "..."]
evidence_level: "illustrative|published|estimated|simulated|not-measured"
source_basis: "..."
as_of_date: "2026-11-10"
owner_role: "..."
source_index: "INDEX.md"
review_trigger: "..."
---

# <Title>

<one-paragraph evidence disclaimer: what is illustrative, what is sourced>

## 1. The One-Line Framing
## 2. Context & Background
## 3. The Business Question
## 4. Assumptions & Constraints
## 5. Data, Tools & Method
### 5a Data Specification
### 5b Tools & Stack
### 5c Method Summary
### 5d Provenance, Quality, Exclusions, Leakage
### 5e Reproducibility & Method-Selection Ledger
## 6. Analysis Walkthrough
## 7. The Finding
## 8. System/Solution Design
## 9. Evaluation & Validation
## 10. Action Taken
## 11. Outcome & Results
## 12. Failure Modes & What Went Wrong
## 13. Transferable Lesson
## 14. Connections & Cross-References
### Interview pack (12–18 questions, each with Answer / Reasoning / Follow-up / Red flag / Scoring)
## 15. Quick Reference Card
```

Section 15 is always last. Retain a conditional section with a specific `N/A — <reason>`; never use `N/A` for unknown evidence (write `unknown` / `not-measured`).

Required depth per case:

- **§5a** states grain, cut-off timestamp, population, keys, units, and what is *not* available at decision time.
- **§6** contains at least **3 checked numeric derivations** in fenced `text` blocks using the anatomy `FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION`, plus one boundary/counterexample where the formula breaks. Recompute arithmetic yourself; show intermediate values.
- **§9** defines numerator, denominator, population, window, comparator, guardrails, uncertainty, and executed/not-executed status.
- **§12** has ≥4 *distinct* failure mechanisms with discriminating checks (not "monitor the model").
- **§14** interview pack: 12–18 distinct questions mixing recall/calculation/diagnosis/judgment; ≥half require computation, selection, or diagnosis; every question has its own answer text (no shared boilerplate).
- Evidence honesty: no invented company architecture, no invented measured results. Illustrative arithmetic is labelled illustrative.

## 3. Cheat sheet structure (exactly these H2 headings, in order)

```
---
title: "..."
topic: "..."
collection: "..."
level: "foundation|core|advanced"
evidence: "DERIVED|VERIFIED|SOURCED|ILLUSTRATIVE|HEURISTIC|OPEN"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# <Title>

## 1. File Header
## 2. Concept Map / Mental Model
## 3. Foundational Definitions
## 4. Core Formulas with Full Derivation
### 4.1 Base case
### 4.2 General case
### 4.3 Variants
### 4.4 Limits and counterexamples
## 5. Worked Numerical Example
## 6. Visual / Geometric Intuition
## 7. Comparison / Selection Table
## 8. Failure Modes & Edge Cases
## 9. Complexity & Cost Analysis
## 10. Key Theorems / Results Reference
## 11. Interview / Exam Questions
## 12. Quick Reference Card
```

Required depth: §4 uses the five-part anatomy for each substantive result and ends with an explicit validity condition; §5 has an independent arithmetic re-check and a sensitivity note; §8 has ≥4 failure entries with *competing causes* and *discriminating checks*; §11 has 10–15 answered questions (≥half computational/diagnostic); §12 is a 5–10 row card and nothing follows it.

For non-mathematical topics keep §4's heading and use a **mechanism/invariant block**: preconditions → state transition → invariant and justification → worked trace → failure boundary.

## 4. Common Issues file (`NN_25_Common_Issues.md`)

Same 12 headings as a sheet (it is a sheet variant), but §8/§11 carry the issue catalog: **≥25 distinct complete issues**, each with:

`ISSUE-ID | category (VIOLATED_ASSUMPTION | NUMERICAL_EDGE | SCALE_FAILURE | CONCEPT_CONFUSION) | symptom | competing causes | discriminating check | correction | prevention | verification`

Empty rows never count. IDs are stable and unique within the file.

## 5. Collection index requirements

`Case_Studies/INDEX.md` → 7 sections: 1 Collection metadata, 2 Case catalog (11 columns: Case ID, linked title, domain, case type, decision, primary pattern, up to 2 secondary patterns, evidence, stage, last reviewed, result/limitation), 3 Canonical reusable patterns, 4 Similarity/contrast map, 5 Questions/future-case backlog, 6 Review/retirement log, 7 Pre-publish checks.

`Cheat_Sheets/INDEX.md` → 11 sections: 1 Overview, 2 Topic ladder, 3 Quick lookup, 4 Beginner/Practitioner/Advanced paths, 5 Prerequisite skills, 6 Exclusions/handoffs, 7 Master symbol glossary, 8 Card directory, 9 Common Issues locator, 10 Sources/versions/maintenance, 11 Acceptance checklist.

Index catalogs MUST link only files that exist. A planned row stays `planned` until its file exists.

## 6. Numeric and notation conventions

- Display math: fenced ```text blocks. One convention per file.
- Define each symbol at first use: meaning, units, shape/type, valid range.
- Check dimensions, rounding, signs, and one material limit case.
- Currency, dates, and units explicit. Distinguish event time from processing time.
- Constructed inputs are labelled `ILLUSTRATIVE` even when the arithmetic is verified.

## 7. Progress accounting (mandatory, makes the run resumable)

Maintain `<subtree>/AUTHORING_PROGRESS.md` (create if absent; append, never rewrite history):

```markdown
| Timestamp (UTC) | Artifact | Status | Lint | Note |
|---|---|---|---|---|
| 2026-11-10T12:00Z | Cheat_Sheets/01_x.md | authored | pass | derived 4 formulas, 3 checks |
```

Rules: one row per completed artifact; status ∈ `authored | revised | blocked | planned`; note states why if `blocked`. This table is the resume checkpoint — a later worker continues from the first `planned` or `blocked` row.

## 8. Never do

- NEVER write a file you have not linted.
- NEVER claim a measurement, deployment, or company result you did not source.
- NEVER touch another subtree, the course root indexes, or `Templates/`.
- NEVER delete an existing catalog commitment without recording the reason in the index backlog.
- NEVER rename existing published files (incoming links break); add new numbered files.
- NEVER leave `TODO`, `<placeholder>`, `TBD`, or "content omitted" markers in a delivered file.
