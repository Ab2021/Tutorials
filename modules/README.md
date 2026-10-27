# Module Folders

This directory holds the topic-level expansion of the course. The 48 modules are grouped into
**seven parts** so that the tree can be navigated by purpose rather than by number alone.

```text
modules/
  01_foundations/             00, 01, 01A, 02–05     (7 modules)
  02_methods/                 06–20, 14A             (16 modules)
  03_commerce/                21–24                  (4 modules)
  04_value_governance/        25–28                  (4 modules)
  05_labs_electives/          29–30                  (2 modules)
  06_domain_electives/        31–40                  (10 modules)
  07_engineering_electives/   41–45                  (5 modules)
```

Each part has its own `_index.md` that explains the part's purpose, lists its modules, and shows
how they depend on each other. Each module folder has an `_index.md` topic map that breaks the
survey file into 2-4 hour teaching units.

The survey files live one level up in `course/`. A module folder is the **topic-level expansion**:
the survey file is the narrative arc; the folder is the sequence of files that let a learner work
through a topic without a second source.

## How to read a module folder

1. Open the part `_index.md` to see where the module sits.
2. Open the module's `_index.md`.
3. Read sections 1 and 2 for purpose, prerequisites, and assessment.
4. Use the topic table in section 3 to choose a topic, or read them in order.
5. Section 4 shows the dependency chain and the minimum path.
6. Section 5 states what the module deliberately excludes.

## Status key

| Status | Meaning |
|---|---|
| planned | Indexed with scope, length target, and position; file not yet written |
| draft | File exists but is below target depth or unreviewed |
| complete | File exists at target length with sources and exercises |

All 48 modules are currently indexed at `planned`. The survey files are complete at survey depth
and can be taught on their own.

## Part map

### 01_foundations — what the chain is and what the data can support

| # | Folder | Survey file |
|---:|---|---|
| 00 | [`00_orientation_and_industry_map/`](01_foundations/00_orientation_and_industry_map/_index.md) | `00_orientation_and_industry_map.md` |
| 01 | [`01_supply_chain_foundations/`](01_foundations/01_supply_chain_foundations/_index.md) | `01_supply_chain_foundations.md` |
| 01A | [`01A_foundations_units_time_assumptions/`](01_foundations/01A_foundations_units_time_assumptions/_index.md) | `01A_foundations_units_time_assumptions.md` |
| 02 | [`02_supply_chain_data_landscape/`](01_foundations/02_supply_chain_data_landscape/_index.md) | `02_supply_chain_data_landscape.md` |
| 03 | [`03_data_engineering_interoperability/`](01_foundations/03_data_engineering_interoperability/_index.md) | `03_data_engineering_interoperability.md` |
| 04 | [`04_statistics_causal_inference/`](01_foundations/04_statistics_causal_inference/_index.md) | `04_statistics_causal_inference.md` |
| 05 | [`05_experimentation_pilots_evidence/`](01_foundations/05_experimentation_pilots_evidence/_index.md) | `05_experimentation_pilots_evidence.md` |

### 02_methods — the analytical core

| # | Folder | Survey file |
|---:|---|---|
| 06 | [`06_machine_learning_foundations/`](02_methods/06_machine_learning_foundations/_index.md) | `06_machine_learning_foundations.md` |
| 07 | [`07_time_series_forecasting/`](02_methods/07_time_series_forecasting/_index.md) | `07_time_series_forecasting.md` |
| 08 | [`08_hierarchical_intermittent_probabilistic_forecasting/`](02_methods/08_hierarchical_intermittent_probabilistic_forecasting/_index.md) | `08_hierarchical_intermittent_probabilistic_forecasting.md` |
| 09 | [`09_demand_sensing_promotions_npi/`](02_methods/09_demand_sensing_promotions_npi/_index.md) | `09_demand_sensing_promotions_npi.md` |
| 10 | [`10_inventory_optimization_multi_echelon/`](02_methods/10_inventory_optimization_multi_echelon/_index.md) | `10_inventory_optimization_multi_echelon.md` |
| 11 | [`11_network_design_optimization/`](02_methods/11_network_design_optimization/_index.md) | `11_network_design_optimization.md` |
| 12 | [`12_transportation_routing_last_mile/`](02_methods/12_transportation_routing_last_mile/_index.md) | `12_transportation_routing_last_mile.md` |
| 13 | [`13_warehousing_fulfillment_labor/`](02_methods/13_warehousing_fulfillment_labor/_index.md) | `13_warehousing_fulfillment_labor.md` |
| 14 | [`14_procurement_supplier_risk_spend/`](02_methods/14_procurement_supplier_risk_spend/_index.md) | `14_procurement_supplier_risk_spend.md` |
| 14A | [`14A_resilience_disruption_analytics/`](02_methods/14A_resilience_disruption_analytics/_index.md) | `14A_resilience_disruption_analytics.md` |
| 15 | [`15_manufacturing_production_planning/`](02_methods/15_manufacturing_production_planning/_index.md) | `15_manufacturing_production_planning.md` |
| 16 | [`16_pricing_promotions_revenue_management/`](02_methods/16_pricing_promotions_revenue_management/_index.md) | `16_pricing_promotions_revenue_management.md` |
| 17 | [`17_assortment_category_retail_analytics/`](02_methods/17_assortment_category_retail_analytics/_index.md) | `17_assortment_category_retail_analytics.md` |
| 18 | [`18_deep_learning_foundation_models_llm_agents/`](02_methods/18_deep_learning_foundation_models_llm_agents/_index.md) | `18_deep_learning_foundation_models_llm_agents.md` |
| 19 | [`19_control_towers_visibility_events/`](02_methods/19_control_towers_visibility_events/_index.md) | `19_control_towers_visibility_events.md` |
| 20 | [`20_digital_twins_simulation_optimization_uncertainty/`](02_methods/20_digital_twins_simulation_optimization_uncertainty/_index.md) | `20_digital_twins_simulation_optimization_uncertainty.md` |

### 03_commerce — consumer-facing and on-demand models

| # | Folder | Survey file |
|---:|---|---|
| 21 | [`21_ecommerce_supply_chain_analytics/`](03_commerce/21_ecommerce_supply_chain_analytics/_index.md) | `21_ecommerce_supply_chain_analytics.md` |
| 22 | [`22_quick_commerce_on_demand_delivery/`](03_commerce/22_quick_commerce_on_demand_delivery/_index.md) | `22_quick_commerce_on_demand_delivery.md` |
| 23 | [`23_omnichannel_fulfillment_returns/`](03_commerce/23_omnichannel_fulfillment_returns/_index.md) | `23_omnichannel_fulfillment_returns.md` |
| 24 | [`24_recommendation_systems_personalization/`](03_commerce/24_recommendation_systems_personalization/_index.md) | `24_recommendation_systems_personalization.md` |

### 04_value_governance — turning analysis into a governed decision

| # | Folder | Survey file |
|---:|---|---|
| 25 | [`25_product_economics_leadership/`](04_value_governance/25_product_economics_leadership/_index.md) | `25_product_economics_leadership.md` |
| 26 | [`26_quality_evaluation_model_risk/`](04_value_governance/26_quality_evaluation_model_risk/_index.md) | `26_quality_evaluation_model_risk.md` |
| 27 | [`27_datasets_tools_benchmarks/`](04_value_governance/27_datasets_tools_benchmarks/_index.md) | `27_datasets_tools_benchmarks.md` |
| 28 | [`28_domain_case_library/`](04_value_governance/28_domain_case_library/_index.md) | `28_domain_case_library.md` |

### 05_labs_electives — practice and breadth

| # | Folder | Survey file |
|---:|---|---|
| 29 | [`29_industry_electives/`](05_labs_electives/29_industry_electives/_index.md) | `29_industry_electives.md` |
| 30 | [`30_simulation_optimization_production_labs/`](05_labs_electives/30_simulation_optimization_production_labs/_index.md) | `30_simulation_optimization_production_labs.md` |

### 06_domain_electives — industry-specific supply chains

| # | Folder | Survey file |
|---:|---|---|
| 31 | [`31_aftermarket_spare_parts_service/`](06_domain_electives/31_aftermarket_spare_parts_service/_index.md) | `31_aftermarket_spare_parts_service.md` |
| 32 | [`32_food_agriculture_perishables/`](06_domain_electives/32_food_agriculture_perishables/_index.md) | `32_food_agriculture_perishables.md` |
| 33 | [`33_healthcare_provider_life_sciences/`](06_domain_electives/33_healthcare_provider_life_sciences/_index.md) | `33_healthcare_provider_life_sciences.md` |
| 34 | [`34_energy_utilities_capital_projects/`](06_domain_electives/34_energy_utilities_capital_projects/_index.md) | `34_energy_utilities_capital_projects.md` |
| 35 | [`35_public_sector_defence_humanitarian/`](06_domain_electives/35_public_sector_defence_humanitarian/_index.md) | `35_public_sector_defence_humanitarian.md` |
| 36 | [`36_global_freight_ports_trade_policy/`](06_domain_electives/36_global_freight_ports_trade_policy/_index.md) | `36_global_freight_ports_trade_policy.md` |
| 37 | [`37_circular_economy_reverse_networks/`](06_domain_electives/37_circular_economy_reverse_networks/_index.md) | `37_circular_economy_reverse_networks.md` |
| 38 | [`38_process_industries_commodities/`](06_domain_electives/38_process_industries_commodities/_index.md) | `38_process_industries_commodities.md` |
| 39 | [`39_mining_metals_forestry/`](06_domain_electives/39_mining_metals_forestry/_index.md) | `39_mining_metals_forestry.md` |
| 40 | [`40_telecom_technology_hardware/`](06_domain_electives/40_telecom_technology_hardware/_index.md) | `40_telecom_technology_hardware.md` |

### 07_engineering_electives — cross-cutting engineering disciplines

| # | Folder | Survey file |
|---:|---|---|
| 41 | [`41_maritime_rail_air_operations/`](07_engineering_electives/41_maritime_rail_air_operations/_index.md) | `41_maritime_rail_air_operations.md` |
| 42 | [`42_treasury_credit_financial_supply_chain/`](07_engineering_electives/42_treasury_credit_financial_supply_chain/_index.md) | `42_treasury_credit_financial_supply_chain.md` |
| 43 | [`43_security_engineering_supply_chain/`](07_engineering_electives/43_security_engineering_supply_chain/_index.md) | `43_security_engineering_supply_chain.md` |
| 44 | [`44_mlops_production_analytics/`](07_engineering_electives/44_mlops_production_analytics/_index.md) | `44_mlops_production_analytics.md` |
| 45 | [`45_plant_layout_detailed_engineering/`](07_engineering_electives/45_plant_layout_detailed_engineering/_index.md) | `45_plant_layout_detailed_engineering.md` |

Modules 31–40 are **domain electives**; 41–45 are **engineering electives**. Both sit after the
40-week core and do not renumber Modules 00–30. Each assumes the core methods and adds the
vocabulary, constraints, data, and cases of a specific industry or discipline.

## Authoring a topic file

Follow [`../course/MODULE_EXPANSION_SPEC.md`](../course/MODULE_EXPANSION_SPEC.md). A topic file is not a summary of the survey section. It
replaces that section with a self-contained teaching unit: industry problem, mechanics, worked
example, code or calculation, assumptions, failure modes, exercises, and sources.

When a topic file is written, change its status in the module `_index.md` from `planned` to `draft`
or `complete`, and update the status counts in section 6.
