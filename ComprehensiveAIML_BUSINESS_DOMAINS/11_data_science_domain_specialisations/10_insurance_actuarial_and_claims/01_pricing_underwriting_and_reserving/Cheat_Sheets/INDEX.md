# Pricing, Underwriting and Reserving — Cheat Sheets

## 1. Overview

This collection supports actuarial analysts and reviewers making exposure-normalized pricing, underwriting and reserving decisions. It assumes basic insurance terminology, arithmetic and probability. A draft is teaching material, not an approved rate or publication-ready actuarial opinion. The [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked master template govern all sheets.

## 2. Topic ladder

| Level | Topic | Outcome and prerequisite | Status |
|---|---|---|---|
| Core | [03 Non-Life Pricing: Pure Premium, Frequency–Severity and Exposure](03_NonLife_Pricing_Ratemaking.md) | Reconcile earned exposure and claim counts; calculate and diagnose expected loss cost. Requires basic ratios, expected values and coverage terminology. | Draft; not reviewed |
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

## 4. Reading paths

- **Beginner:** Review arithmetic, basic probability and contract terminology, then read Sections 1–5 of the pricing sheet; the collection has only one authored sheet.
- **Practitioner:** Read the full pricing sheet, work its example independently, and use its diagnostics on a matched exposure and claims extract.
- **Advanced:** Read the full sheet, test the component assumptions and operational boundaries, and extend to reserving and governance when their planned sheets are authored.

## 5. Prerequisite skills

Calculate ratios and weighted averages, distinguish conditional means from individual outcomes, and interpret contract periods, limits and deductibles. The [CAS Basic Ratemaking reference](https://www.casact.org/abstract/basic-ratemaking) is a relevant external foundation; use jurisdiction-specific professional guidance when making actual decisions.

## 6. Exclusions and handoffs

The authored sheet does not provide a complete reserving model, a full generalized linear model inference treatment, reinsurance structuring, or legal advice. Reserving validation and rating-factor governance are planned rather than available handoffs. A qualified actuary and relevant legal and governance owners must approve production use.

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

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Avoid |
|---|---|---|---|
| [Non-life pricing card](03_NonLife_Pricing_Ratemaking.md#12-quick-reference-card) | Matched loss divided by earned exposure; reconcile frequency × severity | Positive exposure and consistent coverage, time and loss basis | Applying mismatched claim counts or undeveloped losses as ultimate |

## 9. Common Issues locator

No separate Common Issues file has been authored; the planned collection-level target is at least 25 complete distinct issues. The draft pricing sheet contains its own [hypothetical diagnostic entries](03_NonLife_Pricing_Ratemaking.md#8-failure-modes--edge-cases); these are not a completed collection-level issue artifact.

## 10. Sources, versions and maintenance

| Artifact | Evidence and source | Editorial state | Maintenance trigger |
|---|---|---|---|
| [Non-life pricing sheet](03_NonLife_Pricing_Ratemaking.md) | Constructed examples and derived identities; source inventory names the topic, not its inaccessible contents. CAS *Basic Ratemaking*, fifth edition, 2016, is identified as an educational reference; detailed review remains open. | Version 1.0, draft; not independently reviewed | Exposure definition, claim development, contract terms, applicable guidance or source changes |

The source crosswalk assigns the original `Insurance_Analytical_Modelling/Insurance_Cheat_Sheets/03_NonLife_Pricing_Ratemaking.md` topic to this module; this entry does not assert that the original file was inspected or reproduced.

## 11. Collection acceptance checklist

- [x] One actual draft has a linked topic, card and status.
- [x] Draft evidence is distinguished from real insurer experience.
- [ ] Independent numerical, source, link and editorial review completed.
- [ ] Planned neighboring topics and final Common Issues artifact authored and reviewed.
- [ ] Collection-wide publication and governance approval completed.
