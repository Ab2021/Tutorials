# Part II — Analytical Methods

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Folder:** `modules/02_methods/`
> **Modules:** 06, 07, 08, 09, 10, 11, 12, 13, 14, 14A, 15, 16, 17, 18, 19, 20
> **Version:** 2026.5

---

## 1. Purpose of this part

This is the analytical core of the program: the methods that convert data into a decision. It
runs in four blocks.

| Block | Modules | What it decides |
|---|---|---|
| Predict | 06, 07, 08, 09 | How much will be demanded, with what uncertainty, at what level of the hierarchy |
| Position and move | 10, 11, 12, 13 | How much stock, where, and how it travels |
| Source and make | 14, 14A, 15 | Who supplies it, what happens when they fail, and how it is produced |
| Price, model, and see | 16, 17, 18, 19, 20 | What it is worth, what the customer sees, and how the system is simulated |

Modules are deliberately ordered so that later ones can assume the vocabulary of earlier ones.
Each still stands alone at survey depth.

## 2. Modules

| # | Module | Core question | Topics |
|---:|---|---|---:|
| 06 | [`06_machine_learning_foundations/`](06_machine_learning_foundations/_index.md) | What makes a prediction model valid? | 10 |
| 07 | [`07_time_series_forecasting/`](07_time_series_forecasting/_index.md) | How do we forecast a series honestly? | 10 |
| 08 | [`08_hierarchical_intermittent_probabilistic_forecasting/`](08_hierarchical_intermittent_probabilistic_forecasting/_index.md) | How do forecasts reconcile, and how is uncertainty expressed? | 10 |
| 09 | [`09_demand_sensing_promotions_npi/`](09_demand_sensing_promotions_npi/_index.md) | How do promotions and new products change demand? | 10 |
| 10 | [`10_inventory_optimization_multi_echelon/`](10_inventory_optimization_multi_echelon/_index.md) | How much stock, where, and against what service target? | 10 |
| 11 | [`11_network_design_optimization/`](11_network_design_optimization/_index.md) | Where should capacity and inventory sit? | 10 |
| 12 | [`12_transportation_routing_last_mile/`](12_transportation_routing_last_mile/_index.md) | How does freight move, and what does it cost? | 10 |
| 13 | [`13_warehousing_fulfillment_labor/`](13_warehousing_fulfillment_labor/_index.md) | How does a facility meet throughput with labour? | 10 |
| 14 | [`14_procurement_supplier_risk_spend/`](14_procurement_supplier_risk_spend/_index.md) | Who should supply, and how is spend and risk managed? | 10 |
| 14A | [`14A_resilience_disruption_analytics/`](14A_resilience_disruption_analytics/_index.md) | What breaks, and how fast can it recover? | 10 |
| 15 | [`15_manufacturing_production_planning/`](15_manufacturing_production_planning/_index.md) | How is the plan turned into a schedule? | 11 |
| 16 | [`16_pricing_promotions_revenue_management/`](16_pricing_promotions_revenue_management/_index.md) | What should it cost, and how is revenue managed? | 10 |
| 17 | [`17_assortment_category_retail_analytics/`](17_assortment_category_retail_analytics/_index.md) | What should be on the shelf? | 10 |
| 18 | [`18_deep_learning_foundation_models_llm_agents/`](18_deep_learning_foundation_models_llm_agents/_index.md) | When do deep and foundation models help? | 10 |
| 19 | [`19_control_towers_visibility_events/`](19_control_towers_visibility_events/_index.md) | How does the organisation see and act on exceptions? | 10 |
| 20 | [`20_digital_twins_simulation_optimization_uncertainty/`](20_digital_twins_simulation_optimization_uncertainty/_index.md) | How do we test a decision before taking it? | 10 |

## 3. Sequence

```text
06 Machine learning foundations
  └─> 07 Time series forecasting
        └─> 08 Hierarchical, intermittent, probabilistic
              └─> 09 Demand sensing, promotions, NPI
                    └─> 10 Inventory optimisation
                          └─> 11 Network design
                                ├─> 12 Transportation and routing
                                │     └─> 13 Warehousing and labour
                                ├─> 14 Procurement and supplier risk
                                │     └─> 14A Resilience and disruption
                                └─> 15 Manufacturing and production planning

Parallel depth after 10:
16 Pricing and revenue management
17 Assortment and category
18 Deep learning and foundation models
19 Control towers and visibility
20 Digital twins and simulation
```

## 4. What this part does not cover

- Data engineering, inference, and experimental design: see Part I.
- Commercial models for e-commerce, quick commerce, omnichannel, and recommendation: see Part III.
- Value cases, model risk, and case method: see Part IV.
- Sector constraints: see Parts VI and VII.

## 5. Status

| Status | Count |
|---|---:|
| planned | 16 |
| draft | 0 |
| complete | 0 |

The sixteen survey files in `../../course/` are complete at survey depth. Topic files are indexed
but not yet written.

## 6. Sources

Each module's `_index.md` and survey file lists its own sources. Evidence grading follows
Module 28.
