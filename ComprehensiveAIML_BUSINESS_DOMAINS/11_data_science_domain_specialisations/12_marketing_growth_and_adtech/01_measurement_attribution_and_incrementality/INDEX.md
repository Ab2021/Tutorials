# Measurement, Attribution and Incrementality

> **Family:** [Marketing, Growth and AdTech](../INDEX.md)  
> **Purpose:** Marketing decision systems that distinguish correlation from causal impact under fragmented journeys and privacy constraints.  
> **Company / context examples:** Brands, retailers, media, B2B

## Detailed topic map

The source crosswalk has no entry assigning a reconstructed source body to this
module as of 2026-11-09. The commitments below therefore come from the original
local catalog. They are teaching commitments, not evidence that a particular
vendor, platform, or source lesson was available. A completed teaching artifact
must preserve that distinction and label constructed numerical workloads
`ILLUSTRATIVE`.

| # | Topic | Testable outcome | Required predecessor | Teaching evidence state |
|---:|---|---|---|---|
| 1 | measurement strategy | Define the decision, estimand, denominator, guardrail, and decision-time data contract. | Basic probability and business metric arithmetic. | planned |
| 2 | MMM | State an aggregate causal model, its priors/identifiability limits, and a calibration plan. | 1; regression and time series basics. | planned |
| 3 | MTA limitations | Diagnose when path credit is descriptive rather than incremental. | 1; event-time joins. | planned |
| 4 | experiments | Design an experiment with randomization unit, exposure, outcome, power, and guardrails. | 1; sampling and uncertainty. | planned |
| 5 | geo lift | Select matched markets, estimate lift, and reject contamination or interference. | 4; cluster designs. | planned |
| 6 | causal inference | Name the estimand and identify the assumptions behind observational alternatives. | 1; potential outcomes. | planned |
| 7 | privacy | Minimize measurement data while preserving the stated estimand and audit boundary. | 1; data-governance basics. | planned |
| 8 | identity | Specify linkage confidence, false-link cost, and aggregate fallback when identity is absent. | 7; join semantics. | planned |
| 9 | budget allocation | Allocate a constrained budget using marginal incremental value and uncertainty. | 2, 4–6; optimization basics. | planned |
| 10 | reporting | Build a decision report that separates observed correlation, causal evidence, and open uncertainty. | 1–9. | planned |
| 11 | governance | Assign metric ownership, change control, and escalation for material measurement changes. | 7, 10. | planned |
| 12 | stakeholders | Resolve conflicts among finance, channel, analytics, and privacy owners with a decision record. | 10–11. | planned |

**Completion boundary:** no row above is an authored sheet merely because it is
listed here. The local applied collections remain the teaching-artifact homes;
their catalogs must link a real file before any status becomes draft.

## Applied collections

- [Case Studies](./Case_Studies/INDEX.md): decision-centred cases using the local v2 catalog and integrated outline standard.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): concise references for recurring design, analysis, and operating decisions.

## Scope boundary

Use public company material only where cited. Treat company-specific systems as unknown unless a primary or clearly labelled published source supports the claim. Learning cases may be simulated or illustrative, but cannot claim measured production impact.