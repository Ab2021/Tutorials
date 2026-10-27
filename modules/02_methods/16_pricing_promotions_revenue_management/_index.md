# Module 16 — Pricing, Promotions, and Revenue Management

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
> **Folder:** `modules/02_methods/16_pricing_promotions_revenue_management/`
> **Version:** 2026.5

---

## 1. Module purpose

This module connects elasticity, markdown, promotion, and revenue management to margin, with an explicit legal and competitive boundary.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 04, 05, 07 |
| Unlocks | 17, 24 |
| Assessment | Elasticity and markdown study (Week 19) |
| Survey file | `16_pricing_promotions_revenue_management.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_pricing-decision-stack.md`](01_pricing-decision-stack.md) | The Pricing Decision Stack | List price, trade terms, markdown, personalized offer, and who owns each. | 1,200–1,800 | planned |
| 2 | [`02_value-wtp.md`](02_value-wtp.md) | Price, Value, and Willingness to Pay | Value-based pricing and the evidence needed to support it. | 1,200–1,800 | planned |
| 3 | [`03_elasticity-identification.md`](03_elasticity-identification.md) | Own-Price and Cross-Price Elasticity | Estimating response without confusing it with correlation. | 1,200–1,800 | planned |
| 4 | [`04_lerner-rule.md`](04_lerner-rule.md) | From Elasticity to Price: The Lerner Rule | Margin-maximizing price and the assumptions behind it. | 1,200–1,800 | planned |
| 5 | [`05_markdown-optimization.md`](05_markdown-optimization.md) | Markdown Optimization with Salvage | Timing, depth, and the cost of waiting. | 1,200–1,800 | planned |
| 6 | [`06_promotion-trade-spend.md`](06_promotion-trade-spend.md) | Promotion Optimization and Trade Spend | Incremental margin after cannibalization and pull-forward. | 1,200–1,800 | planned |
| 7 | [`07_revenue-management.md`](07_revenue-management.md) | Revenue Management and Littlewood's Rule | Capacity, segmentation, and protection levels. | 1,200–1,800 | planned |
| 8 | [`08_dynamic-personalized.md`](08_dynamic-personalized.md) | Dynamic and Personalized Pricing | Data requirements, fairness, and legal exposure. | 1,200–1,800 | planned |
| 9 | [`09_competitive-response.md`](09_competitive-response.md) | Competitive Response and Price Wars | Reaction functions, signaling, and antitrust red lines. | 1,200–1,800 | planned |
| 10 | [`10_price-testing.md`](10_price-testing.md) | Price Testing, Reference Prices, and Psychological Pricing | Measuring response and interpreting the number a customer sees. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Pricing Decision Stack ->
02 Price, Value, and Willingness to Pay ->
03 Own-Price and Cross-Price Elasticity ->
04 From Elasticity to Price: The Lerner Rule ->
05 Markdown Optimization with Salvage ->
06 Promotion Optimization and Trade Spend ->
07 Revenue Management and Littlewood's Rule ->
08 Dynamic and Personalized Pricing ->
09 Competitive Response and Price Wars ->
10 Price Testing, Reference Prices, and Psychological Pricing
```

## 5. What this module does not cover

- It does not set corporate pricing strategy or provide legal advice on competition law.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`16_pricing_promotions_revenue_management.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
