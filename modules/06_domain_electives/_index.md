# Part VI — Domain Electives

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Folder:** `modules/06_domain_electives/`
> **Modules:** 31–40
> **Version:** 2026.5

---

## 1. Purpose of this part

Ten full-depth industry modules. Each is 26 hours with a worked example, five labs, an assignment,
a rubric, failure modes, and a sector-specific regulatory lens. They sit after the core and assume
its methods; none renumbers Modules 00–30.

The organising idea is that **every industry has a constraint the generic method does not contain**:

| Module | The constraint that changes the plan |
|---|---|
| 31 | Installed base and failure demand, not end-customer demand |
| 32 | Shelf life, harvest yield, and cold chain |
| 33 | Clinical consequence, traceability, and item-level identity |
| 34 | Asset criticality and a project schedule, not a steady-state flow |
| 35 | Public accountability, donor funding, and access |
| 36 | Borders, documents, and trade policy |
| 37 | Reverse flow and disposition, not forward fulfilment |
| 38 | Continuous process, yield, and specification |
| 39 | Geology and grade, not a production rate you can choose |
| 40 | Configuration, allocation, and long qualification cycles |

## 2. Modules

| # | Module | Sector | Topics |
|---:|---|---|---:|
| 31 | [`31_aftermarket_spare_parts_service/`](31_aftermarket_spare_parts_service/_index.md) | Aftermarket and service | 9 |
| 32 | [`32_food_agriculture_perishables/`](32_food_agriculture_perishables/_index.md) | Food and agriculture | 9 |
| 33 | [`33_healthcare_provider_life_sciences/`](33_healthcare_provider_life_sciences/_index.md) | Healthcare and life sciences | 9 |
| 34 | [`34_energy_utilities_capital_projects/`](34_energy_utilities_capital_projects/_index.md) | Energy, utilities, capital projects | 9 |
| 35 | [`35_public_sector_defence_humanitarian/`](35_public_sector_defence_humanitarian/_index.md) | Public sector, defence, humanitarian | 10 |
| 36 | [`36_global_freight_ports_trade_policy/`](36_global_freight_ports_trade_policy/_index.md) | Global freight and trade | 10 |
| 37 | [`37_circular_economy_reverse_networks/`](37_circular_economy_reverse_networks/_index.md) | Circular economy and reverse | 10 |
| 38 | [`38_process_industries_commodities/`](38_process_industries_commodities/_index.md) | Process industries and commodities | 10 |
| 39 | [`39_mining_metals_forestry/`](39_mining_metals_forestry/_index.md) | Mining, metals, forestry | 10 |
| 40 | [`40_telecom_technology_hardware/`](40_telecom_technology_hardware/_index.md) | Telecom, technology, aerospace | 10 |

## 3. Sequence and pairing

```text
Part II methods
  ├─> 31 Aftermarket ──────────┬─> 34 Energy and capital projects
  │                            └─> 40 Telecom and technology
  ├─> 32 Food and agriculture ─┬─> 37 Circular and reverse
  │                            └─> 38 Process industries ──> 39 Mining and forestry
  ├─> 33 Healthcare ───────────┴─> 31 Aftermarket
  ├─> 35 Public sector ──────────> 36 Global freight
  ├─> 36 Global freight ─────────> 41 Maritime, rail, air (Part VII)
  └─> 38 Process industries ─────> 39 Mining and forestry
```

Teach at most two in a 40-week cohort. Pairings that work well: 31 with 34, 32 with 37 and 38,
33 with 31, 35 with 36, 39 with 38 and 34, 40 with 31 and 37.

## 4. What this part does not cover

- The core methods each module assumes: see Part II.
- Operations engineering of ports, rail, and air cargo: see Module 41.
- Financial, security, platform, and facility-design disciplines: see Part VII.
- Classified defence operations, clinical medicine, and legal or tax advice: explicit boundaries,
  listed in `../../TOPIC_COVERAGE_MAP.md` §13.

## 5. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

The ten survey files in `../../course/` are complete at survey depth. Topic files are indexed but
not yet written.

## 6. Sources

Each module's `_index.md` and survey file lists its own sources. Sector claims are graded per
Module 28; company disclosures are C, vendor material is D.
