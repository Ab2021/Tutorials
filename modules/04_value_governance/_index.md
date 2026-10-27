# Part IV — Value, Evidence, and Governance

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Folder:** `modules/04_value_governance/`
> **Modules:** 25, 26, 27, 28
> **Version:** 2026.5

---

## 1. Purpose of this part

A model that cannot be valued, validated, resourced, or explained does not get adopted. This part
covers the four things that turn analysis into a governed decision:

| Module | Function |
|---|---|
| 25 | Build the value case: TCO, ROI, adoption, vendor choice, leadership |
| 26 | Prove and control the model: validation, model cards, audit, monitoring |
| 27 | Know what data and tools exist, and what their limits are |
| 28 | Learn from cases, and test whether the lesson transfers |

This is also where evidence grading is defined (Module 28), which every other module uses.

## 2. Modules

| # | Module | Core question | Topics |
|---:|---|---|---:|
| 25 | [`25_product_economics_leadership/`](25_product_economics_leadership/_index.md) | Is this worth doing, and who will adopt it? | 10 |
| 26 | [`26_quality_evaluation_model_risk/`](26_quality_evaluation_model_risk/_index.md) | Can we trust this model in operation? | 10 |
| 27 | [`27_datasets_tools_benchmarks/`](27_datasets_tools_benchmarks/_index.md) | What can we build on, and what are the benchmark traps? | 10 |
| 28 | [`28_domain_case_library/`](28_domain_case_library/_index.md) | What has actually happened, and does it transfer? | 10 |

## 3. Sequence

```text
18 Deep learning / 20 Simulation ─┐
21–24 Commercial models ──────────┼─> 25 Value and leadership
                                  │        │
                                  ├─> 26 Model risk and validation
                                  ├─> 27 Datasets, tools, benchmarks
                                  └─> 28 Case library and evidence grading
```

Modules 25–27 assume the methods parts. Module 28 can be read at any point and is the evidence
reference for the whole program.

## 4. What this part does not cover

- The analytical methods being valued and validated: see Part II.
- Financial treasury and credit modelling: see Module 42.
- Security assurance and third-party risk: see Module 43.
- Production MLOps implementation: see Module 44.

## 5. Status

| Status | Count |
|---|---:|
| planned | 4 |
| draft | 0 |
| complete | 0 |

The four survey files in `../../course/` are complete at survey depth. Topic files are indexed but
not yet written.

## 6. Sources

Each module's `_index.md` and survey file lists its own sources. Evidence grading is defined in
Module 28: A peer-reviewed or audited, B regulatory or official statistic, C company case study or
investor claim, D vendor marketing, E press summary or anecdote.
