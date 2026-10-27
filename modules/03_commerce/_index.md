# Part III — Commerce and Consumer Models

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Folder:** `modules/03_commerce/`
> **Modules:** 21, 22, 23, 24
> **Version:** 2026.5

---

## 1. Purpose of this part

Part II teaches methods that work for any chain. This part teaches the four commercial models where
the method and the economics are entangled:

- **E-commerce** changes the unit economics: many small orders, parcel cost, delivery promise, returns.
- **Quick commerce** changes the time constraint: dark stores, sub-hour slots, courier dispatch, waste.
- **Omnichannel** changes the inventory question: one pool, many promises, and channel conflict.
- **Recommendation** changes the demand signal itself: the model influences what is bought next.

A planner who applies a distribution-centre model to a dark store, or ignores the feedback loop
between recommendations and demand, produces a plan that is arithmetically correct and operationally wrong.

## 2. Modules

| # | Module | Core question | Topics |
|---:|---|---|---:|
| 21 | [`21_ecommerce_supply_chain_analytics/`](21_ecommerce_supply_chain_analytics/_index.md) | How does an online order become a profitable delivery? | 10 |
| 22 | [`22_quick_commerce_on_demand_delivery/`](22_quick_commerce_on_demand_delivery/_index.md) | How do you promise minutes and still make money? | 10 |
| 23 | [`23_omnichannel_fulfillment_returns/`](23_omnichannel_fulfillment_returns/_index.md) | Which node should serve which promise? | 10 |
| 24 | [`24_recommendation_systems_personalization/`](24_recommendation_systems_personalization/_index.md) | What does the model put in front of the customer, and what does that do to demand? | 11 |

## 3. Sequence

```text
10 Inventory optimisation ──┐
12 Transportation ──────────┼─> 21 E-commerce
13 Warehousing ─────────────┘        │
                                     ├─> 22 Quick commerce
                                     └─> 23 Omnichannel
18 Deep learning ──────────────────> 24 Recommendation
```

Modules 21–23 assume the Part II inventory, transport, and fulfilment methods. Module 24 assumes
Module 18.

## 4. What this part does not cover

- Core forecasting and inventory methods: see Part II.
- Pricing and revenue management: see Module 16.
- Assortment and space: see Module 17.
- Returns as a reverse network beyond the omnichannel case: see Module 37.

## 5. Status

| Status | Count |
|---|---:|
| planned | 4 |
| draft | 0 |
| complete | 0 |

The four survey files in `../../course/` are complete at survey depth. Topic files are indexed but
not yet written.

## 6. Sources

Each module's `_index.md` and survey file lists its own sources. Evidence grading follows
Module 28. Commercial claims from platforms and vendors are graded C or D unless independently audited.
