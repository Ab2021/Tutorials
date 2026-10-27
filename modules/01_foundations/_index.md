# Part I — Foundations

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Folder:** `modules/01_foundations/`
> **Modules:** 00, 01, 01A, 02, 03, 04, 05
> **Version:** 2026.5

---

## 1. Purpose of this part

Foundations answer three questions before any model is built:

1. **What is the decision?** Module 00 builds a decision inventory and a charter.
2. **What is the chain?** Module 01 teaches the entities, flows, and processes; Module 01A fixes the units, calendars, and time assumptions that every later calculation depends on.
3. **What can the data support?** Modules 02–05 cover the data landscape, engineering and interoperability, statistical inference, and the design of credible experiments and pilots.

Everything later in the program assumes this part. A forecasting model built on the wrong unit of measure, or evaluated against a decision nobody makes, fails here — not in Module 07.

## 2. Modules

| # | Module | Core question | Topics |
|---:|---|---|---:|
| 00 | [`00_orientation_and_industry_map/`](00_orientation_and_industry_map/_index.md) | Which decisions matter, and what does this program cover? | 8 |
| 01 | [`01_supply_chain_foundations/`](01_supply_chain_foundations/_index.md) | How does a supply chain actually work? | 10 |
| 01A | [`01A_foundations_units_time_assumptions/`](01A_foundations_units_time_assumptions/_index.md) | What do the numbers mean, and over what time? | 11 |
| 02 | [`02_supply_chain_data_landscape/`](02_supply_chain_data_landscape/_index.md) | What data exists, where, and how good is it? | 10 |
| 03 | [`03_data_engineering_interoperability/`](03_data_engineering_interoperability/_index.md) | How does data get from source to model, reliably? | 11 |
| 04 | [`04_statistics_causal_inference/`](04_statistics_causal_inference/_index.md) | What can be inferred, and what is only correlation? | 10 |
| 05 | [`05_experimentation_pilots_evidence/`](05_experimentation_pilots_evidence/_index.md) | How do we know an intervention worked? | 10 |

## 3. Sequence

```text
00 Orientation and industry map
  └─> 01 Supply chain foundations
        └─> 01A Units, time, and assumptions
              ├─> 02 Data landscape
              │     └─> 03 Data engineering and interoperability
              ├─> 04 Statistics and causal inference
              └─> 05 Experimentation and evidence
```

Modules 02 and 04 can be taken in parallel after 01A. Module 03 assumes 02. Module 05 assumes 04.

## 4. What this part does not cover

- Forecasting, inventory, and optimisation methods: see Part II.
- Commercial and consumer-facing models: see Part III.
- Value cases, model governance, and case method: see Part IV.
- Any industry-specific constraint: see Parts VI and VII.

## 5. Status

| Status | Count |
|---|---:|
| planned | 7 |
| draft | 0 |
| complete | 0 |

The seven survey files in `../../course/` are complete at survey depth. Topic files are indexed
but not yet written.

## 6. Sources

Each module's `_index.md` and survey file lists its own sources. Evidence grading follows
Module 28: A peer-reviewed or audited, B regulatory or official statistic, C company case study
or investor claim, D vendor marketing, E press summary or anecdote.
