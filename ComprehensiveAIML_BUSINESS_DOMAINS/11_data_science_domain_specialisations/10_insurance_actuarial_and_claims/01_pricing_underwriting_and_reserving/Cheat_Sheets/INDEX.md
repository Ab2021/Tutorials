# Pricing, Underwriting and Reserving — Cheat Sheets

## 1. Overview

This collection supports actuarial analysts and reviewers making exposure-normalized pricing, underwriting and reserving decisions. It assumes basic insurance terminology, arithmetic and probability. A draft is teaching material, not an approved rate or publication-ready actuarial opinion. The [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked master template govern all sheets.

## 2. Topic ladder

| Level | Topic | Outcome and prerequisite | Status |
|---|---|---|---|
| Core | [03 Non-Life Pricing: Pure Premium, Frequency–Severity and Exposure](03_NonLife_Pricing_Ratemaking.md) | Reconcile earned exposure and claim counts; calculate and diagnose expected loss cost. Requires basic ratios, expected values and coverage terminology. | Draft; not reviewed |
| Core | [05 Claims Reserving Methods](05_Claims_Reserving_Methods.md) | Develop a paid triangle, reconcile paid outstanding, distinguish incurred-based IBNR and investigate temporal, validation and uncertainty risks; requires ratios and claims terminology. | Draft; not independently reviewed |
| Advanced | Actuarial model review | Assess assumptions, inputs and validation. | Planned |
| Advanced | Rating-factor governance | Review factor decisions, permissible use and monitoring. | Planned |
| Advanced | Reserving validation | Compare claim development assumptions and validation. | Planned |
| Final | Common Issues | Consolidated issue locator and complete troubleshooting artifact. | Planned; no issue file exists |

## 3. Quick topic lookup

| Question | Location | Caution |
|---|---|---|
| How is pure premium derived? | [Formulas, Section 4](03_NonLife_Pricing_Ratemaking.md#4-core-formulas-with-full-derivation) | Match loss, count and exposure populations. |
| How do I translate loss cost into a rate? | [Worked example, Section 5](03_NonLife_Pricing_Ratemaking.md#5-worked-numerical-example) | Premium-based expense fractions must use the stated base. |
| Why does frequency times severity fail to reconcile? | [Failure modes, Section 8](03_NonLife_Pricing_Ratemaking.md#8-failure-modes--edge-cases) | Different count rules, maturity or coverage can invalidate cancellation. |
| How does a log exposure offset work? | [Formulas, Section 4](03_NonLife_Pricing_Ratemaking.md#4-core-formulas-with-full-derivation) | Positive exposure and proportional count intensity are required. |
| How is paid-based outstanding calculated? | [Worked example, Section 5](05_Claims_Reserving_Methods.md#5-worked-numerical-example) | Matched factors, tail and financial basis must be appropriate. |
| How do case reserves relate to IBNR? | [Formulas, Section 4](05_Claims_Reserving_Methods.md#4-core-formulas-with-full-derivation) | Specify broad versus strictly unreported IBNR and avoid double-counting case reserves. |
| What can distort development? | [Failure modes, Section 8](05_Claims_Reserving_Methods.md#8-failure-modes--edge-cases) | Calendar effects, claims handling and sparse observations require investigation. |

## 4. Reading paths

- **Beginner:** Review arithmetic, probability and contract terminology, then read Sections 1–5 of the pricing and reserving sheets.
- **Practitioner:** Read both sheets, reproduce their examples, and apply their diagnostics to reconciled exposure and claims extracts.
- **Advanced:** Examine method assumptions and operational boundaries in both sheets; extend to reserving validation and governance when those planned sheets are authored.

## 5. Prerequisite skills

Calculate ratios and weighted averages, distinguish conditional means from individual outcomes, and interpret contract periods, limits and deductibles. The [CAS Basic Ratemaking reference](https://www.casact.org/abstract/basic-ratemaking) is a relevant external foundation; use jurisdiction-specific professional guidance when making actual decisions.

## 6. Exclusions and handoffs

The authored sheets do not provide a complete reserving validation program, full generalized linear model inference, reinsurance structuring, or legal advice. Reserving validation and rating-factor governance are planned rather than available handoffs. A qualified actuary and relevant legal and governance owners must approve production use.

## 7. Master symbol glossary

| Symbol / term | Meaning and units | Type and domain | First appearance / warning |
|---|---|---|---|
| E | Earned exposure in chosen risk-years or other exposure units | Nonnegative scalar | [Definitions](03_NonLife_Pricing_Ratemaking.md#3-foundational-definitions); ratio denominator must be positive. |
| N | Count of included claims | Nonnegative scalar | [Definitions](03_NonLife_Pricing_Ratemaking.md#3-foundational-definitions); count convention must match severity. |
| L | Included covered loss in currency units | Scalar on stated loss basis | [Definitions](03_NonLife_Pricing_Ratemaking.md#3-foundational-definitions); distinguish gross, net, paid and ultimate. |
| f | Frequency, claims per exposure | Scalar rate | [Definitions](03_NonLife_Pricing_Ratemaking.md#3-foundational-definitions); not necessarily a probability. |
| s | Severity, currency units per included claim | Scalar mean | [Definitions](03_NonLife_Pricing_Ratemaking.md#3-foundational-definitions); undefined as a sample ratio with zero claims. |
| p | Pure premium, currency units per exposure | Scalar mean or estimate | [Definitions](03_NonLife_Pricing_Ratemaking.md#3-foundational-definitions); distinct from charged premium. |
| F | Selected loss development factor | Positive dimensionless scalar | [Formulas](03_NonLife_Pricing_Ratemaking.md#4-core-formulas-with-full-derivation); depends on valuation age and loss basis. |
| R | Indicated premium per exposure | Scalar currency rate | [Formulas](03_NonLife_Pricing_Ratemaking.md#4-core-formulas-with-full-derivation); requires stated expense and profit bases. |
| C(i,j) | Cumulative paid for accident cohort i through development age j, currency units | Scalar cell on stated financial basis | [Reserving definitions](05_Claims_Reserving_Methods.md#3-foundational-definitions); not incremental paid. |
| f(j) | Matched volume-weighted age-to-age factor, dimensionless | Scalar; matched denominator nonzero | [Reserving formulas](05_Claims_Reserving_Methods.md#4-core-formulas-with-full-derivation); distinct from pricing frequency f. |
| T | Tail development factor, dimensionless | Selected scalar | [Reserving formulas](05_Claims_Reserving_Methods.md#4-core-formulas-with-full-derivation); one is an assumption, not proof of maturity. |
| U(i) | Estimated cohort ultimate paid, currency units | Scalar on stated basis | [Reserving formulas](05_Claims_Reserving_Methods.md#4-core-formulas-with-full-derivation); not outstanding alone. |
| K | Case reserve for known claims, currency units | Scalar at valuation | [Reserving definitions](05_Claims_Reserving_Methods.md#3-foundational-definitions); included in paid-based outstanding. |

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Avoid |
|---|---|---|---|
| [Non-life pricing card](03_NonLife_Pricing_Ratemaking.md#12-quick-reference-card) | Matched loss divided by earned exposure; reconcile frequency × severity | Positive exposure and consistent coverage, time and loss basis | Applying mismatched claim counts or undeveloped losses as ultimate |
| [Claims reserving card](05_Claims_Reserving_Methods.md#12-quick-reference-card) | Ultimate less paid equals paid outstanding | Matched development and consistent paid basis | Treating paid outstanding as pure IBNR or a confidence interval |

## 9. Common Issues locator

No separate Common Issues file has been authored; the planned collection-level target is at least 25 complete distinct issues. The draft [pricing](03_NonLife_Pricing_Ratemaking.md#8-failure-modes--edge-cases) and [reserving](05_Claims_Reserving_Methods.md#8-failure-modes--edge-cases) sheets contain local hypothetical diagnostic entries; these are not a completed collection-level issue artifact.

## 10. Sources, versions and maintenance

| Artifact | Evidence and source | Editorial state | Maintenance trigger |
|---|---|---|---|
| [Non-life pricing sheet](03_NonLife_Pricing_Ratemaking.md) | Constructed examples and derived identities; source inventory names the topic, not its inaccessible contents. CAS *Basic Ratemaking*, fifth edition, 2016, is identified as an educational reference; detailed review remains open. | Version 1.0, draft; not independently reviewed | Exposure definition, claim development, contract terms, applicable guidance or source changes |
| [Claims reserving sheet](05_Claims_Reserving_Methods.md) | Constructed examples and derived identities; [Friedland, *Estimating Unpaid Claims Using Basic Techniques*, CAS, 2010](https://www.casact.org/sites/default/files/database/studynotes_friedland_estimating.pdf), Mack's 1993 and 1999 papers, and ASOP 43 are linked for their stated scope. The original source inventory lists the filename; its contents were not inspected. | Version 1.2, draft; numerical, source and editorial independent review open | Claims handling, valuation basis, tail evidence, applicable guidance or source changes |

The source crosswalk assigns the original `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/03_NonLife_Pricing_Ratemaking.md` topic to this module; this entry does not assert that the original file was inspected or reproduced.
The source crosswalk also names `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/05_Claims_Reserving_Methods.md`; its original contents were unavailable and were not inspected or reproduced.

## 11. Collection acceptance checklist

- [x] Two actual drafts have linked topics, cards and statuses.
- [x] Draft evidence is distinguished from real insurer experience.
- [ ] Independent numerical, source, link and editorial review completed.
- [ ] Planned neighboring topics and final Common Issues artifact authored and reviewed.
- [ ] Collection-wide publication and governance approval completed.
