# Part V — Labs and Industry Breadth

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Folder:** `modules/05_labs_electives/`
> **Modules:** 29, 30
> **Version:** 2026.5

---

## 1. Purpose of this part

Two modules that sit between the core and the electives:

- **Module 29 — Industry electives.** Short sector primers (CPG, pharma and cold chain, automotive,
  semiconductor, humanitarian, fashion) that teach vocabulary, constraints, and data standards
  quickly. Their purpose is to let a learner enter a new sector conversation without claiming
  full-depth domain expertise. That depth lives in Part VI.
- **Module 30 — Simulation and production labs.** The hands-on bench: discrete-event simulation,
  MILP network design, VRP, inventory simulation, streaming events, containerised planning services,
  infrastructure as code, observability, and disaster recovery.

Module 30 is the natural bridge to Part VII, where the same engineering questions are treated as a
discipline rather than a lab.

## 2. Modules

| # | Module | Core question | Topics |
|---:|---|---|---:|
| 29 | [`29_industry_electives/`](29_industry_electives/_index.md) | What does this sector care about, in its own words? | 8 |
| 30 | [`30_simulation_optimization_production_labs/`](30_simulation_optimization_production_labs/_index.md) | Can you build and run it? | 10 |

## 3. Sequence

```text
Part II methods ──> 29 Industry electives (breadth)
Part II methods ──> 30 Simulation and production labs (depth)
                        │
                        └─> Part VII engineering electives
```

Module 29 can be taken any time after Part II. Module 30 assumes Part II and benefits from Module 20.

## 4. What this part does not cover

- Full-depth sector treatment: see Part VI.
- Production MLOps, security engineering, and facility design: see Part VII.
- Assessment design and grading calibration: see `../../GRADING_CALIBRATION.md`.

## 5. Status

| Status | Count |
|---|---:|
| planned | 2 |
| draft | 0 |
| complete | 0 |

The two survey files in `../../course/` are complete at survey depth. Topic files are indexed but
not yet written.

## 6. Sources

Each module's `_index.md` and survey file lists its own sources. Module 29 draws on the same
evidence grading as Module 28; sector claims from vendors are graded C or D.
