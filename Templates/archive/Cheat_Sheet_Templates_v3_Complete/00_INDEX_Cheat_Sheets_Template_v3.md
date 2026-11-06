# [Collection Name] — Cheat Sheet INDEX (`00_INDEX.md`)

> **Purpose:** [One-paragraph overview of what this collection covers, its target audience and what it assumes.]
> **Audience:** [who will use the collection]
> **Prerequisites:** [what to know before opening File 01]
> **Version:** 1.0 | **Last reviewed:** YYYY-MM-DD | **Owner:** [role]
> **Math display style:** [LaTeX OR plain-text fenced formulas; one canonical choice]

**Workflow:** Plan the topic ladder/INDEX stub **before** authoring, populate the lookup/glossary as you go, then **finalize this INDEX last** once all actual filenames and links exist (A6, Part E).

## 1. One-paragraph overview

[Scope, use cases, assumed foundations and what “practitioner-ready” means in this domain.]

## 2. Topic ladder (required)

| Level | File number | Linked topic | Reader outcome | Prerequisites | Key formula / skill |
|---|---|---|---|---|---|
| 1 — Foundations | 01 | `[01_Foundations.md]` | [testable outcome] | None | [...] |
| 1 — Foundations | 02 | `[02_Basic_Concepts.md]` | [...] | 01 | [...] |
| 2 — Core Methods | 03 | `[03_Core_Method.md]` | [...] | 01, 02 | [...] |
| 3 — Advanced / Production | 04 | `[04_Advanced_Topic.md]` | [...] | 03 | [...] |
| Final — Common Issues | NN | `[NN_<N>_Common_Issues.md]` | Diagnose ≥25 issues | All relevant | [...] |

*Replace all backtick placeholders with working relative Markdown links before publishing. Do not create forward dependencies in the chosen learning order.*

## 3. Quick topic lookup (required)

| Keyword / question | File | Section | Best use case | Failure warning |
|---|---|---|---|---|
| [topic] | `[NN_Topic.md]` | [section] | [condition] | [avoid if] |

## 4. Reading paths — three options (required)

- **Beginner (3–4 files):** 01 → [...] → [...]. Aim: definitions and calculation basics.
- **Practitioner (6–7 files):** 01 → [...] → [...]. Aim: decisions, scaling and diagnosis.
- **Advanced (all files):** 01 → 02 → ... → NN. Aim: derivation, edge cases, production readiness.

## 5. Prerequisite skills (required)

- **Math / theory:** [linear algebra, probability, etc.]
- **Domain:** [subject-specific knowledge]
- **Systems / tools:** [platform, software, prerequisites]

## 6. What this collection does NOT cover (required)

- [Related topic explicitly out of scope, and where to learn it.]
- [Another excluded topic.]

## 7. Master symbol glossary (original Tip 5)

| Symbol | Meaning | Units / shape | Valid range / context | File(s) |
|---|---|---|---|---|
| `φ` | AR coefficient | scalar | [stationarity condition for model] | 02, 03 |
| `σ²` | Innovation variance | squared units | nonnegative | 02, 03, 05 |
| `d_model` | Embedding dimension | integer | positive | 07, 08 |
| `[symbol]` | [define] | [...] | [...] | [...] |

*The first three rows illustrate the original photograph; use only symbols relevant to this collection. No conflicting symbol meaning across sheets without qualification.*

## 8. Quick Reference Card directory (original Tip 10)

| File | Core equation | One-line use rule | Most critical assumption | Avoid when |
|---|---|---|---|---|
| `[01_Topic.md]` | [...] | [...] | [...] | [...] |

## 9. Common Issues locator (original A1 / Tip 8)

**Final troubleshooting file:** `NN_<N>_Common_Issues.md`. Fill **at least 25** distinct issues (or 50 for the extended collection). Placeholder rows are **not** completed issues.

| Issue ID | Category | Observable symptom | Diagnostic check | Location / file |
|---|---|---|---|---|
| ISSUE-001 | Assumption / Numerical / Scale / Conceptual | [...] | [...] | [...] |

## 10. Empirical sources, versions and collection maintenance — editorial extension

| Source / paper / manual | Supported claim | Year / version | Workload / conditions | Verified on | Review trigger |
|---|---|---|---|---|---|
| [primary source] | [statement] | [year] | [relevant setup] | YYYY-MM-DD | [event] |

| File | Last reviewed | Evidence completeness | Open issues | Publication status |
|---|---|---|---|---|
| `[NN_Topic.md]` | YYYY-MM-DD | [verified/sourced/illustrative] | [...] | draft |

## 11. Collection acceptance checklist

- [ ] Topic ladder includes Level 1, Level 2 and Level 3, with ordered file numbers.
- [ ] All file links exist, navigation works and no prerequisite points forward to an unread file.
- [ ] Quick Topic Lookup and all three reading paths are complete.
- [ ] Master symbol glossary is consistent with each file's definitions.
- [ ] Every cheat sheet has the 7-field Quick Reference Card as its final section.
- [ ] Source or measured numbers are labeled and claims are checked where practical.
- [ ] The last file contains ≥25 distinct, actionable Common Issues.
- [ ] Scope boundaries, prerequisites, version/maintenance details are recorded.
- [ ] INDEX links and status were **finalized last**.
