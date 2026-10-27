# Changelog

All substantive changes to the curriculum are recorded here.

## 2026.7.0 — 27 October 2026

### Added — Module 02 written in full
- Module 02 (`01_foundations/02_supply_chain_data_landscape/`) is now written in full: ten topic
  files, 12,900 lines, each with all 17 required sections.
- `01_systems-of-record.md` — what ERP, WMS, TMS, OMS, POS, PLM, and CRM each record and, more
  importantly, what each cannot answer.
- `02_master-data.md` — item, location, supplier, customer, and BOM masters; the identifier
  stack that joins them; the arithmetic of why duplicate item masters do not double statistical
  safety stock.
- `03_transaction-data.md` — orders, shipments, receipts, invoices, and inventory snapshots at
  their correct grain; the header–line–allocation hierarchy.
- `04_reference-and-external.md` — calendar, weather, macroeconomic, trade, and geospatial
  sources, with their known defects.
- `05_telemetry-iot-vision.md` — sensor, telematics, RFID, and image data as evidence, including
  the difference between a sensor reading and a decision-grade measurement.
- `06_data-movement.md` — EDI, API, flat file, and event transport, and how each constrains
  latency, completeness, and schema stability.
- `07_access-and-licensing.md` — access pathways, commercial licences, privacy, and contractual
  limits on use.
- `08_data-quality-dimensions.md` — completeness, validity, consistency, timeliness, uniqueness,
  and accuracy against ISO/IEC 25012 and ISO 8000 framing.
- `09_master-data-drift.md` — the module's worked case: five drift types, supersession chains,
  bitemporal capture, and a 36-month pack-change reconstruction with the phantom-case
  arithmetic shown.
- `10_source-selection-trust.md` — the hierarchy of trust, per-field source of record, and the
  source-selection record that closes the module.

### Changed
- Module 02 status table reports 10 `complete` and the module total.
- `modules/01_foundations/_index.md` and `VALIDATION_STATUS.md` updated to 20 of 476 topic units.

### Notes
- Several candidate statistics were removed rather than repeated after the underlying study
  could not be retrieved: the "$3.1 trillion cost of bad data" figure, the "15–25% of revenue"
  claim, the "80% of analyst time spent cleaning" claim, and the "3% of data meets quality
  standards" vendor survey are all present only as graded E or D entries marked "do not use".
- The NRF retail calendar is recorded as 4-5-4 throughout; the widespread "4-4-5" usage is
  described as a variant, not as the standard.

## 2026.6.0 — 27 October 2026

### Added — expanded topic files, first module
- Module 01 (`01_foundations/01_supply_chain_foundations/`) is now written in full: ten topic
  files, 13,169 lines, each with all 17 required sections.
- `01_three-flows-one-network.md` — physical, information, and financial flows; push–pull
  boundaries; bullwhip mechanics; cash conversion cycle arithmetic; GS1/EPCIS/Incoterms/IFRS 15
  interfaces.
- `02_decision-stack.md` — strategic, tactical, operational, and execution layers; time fences;
  decision rights; decision latency; agentic planning and decision governance.
- `03_network-entities.md` — party, location, item, and logistics-unit identity; unit-of-measure
  conversion; BOM structures; network topology; multi-tier visibility.
- `04_planning-processes.md` — S&OP, IBP, S&OE; the planning calendar; consensus forecasting and
  forecast value added; maturity models and their misuse.
- `05_inventory-and-service.md` — the inventory taxonomy; Little's law; CSL versus fill rate
  divergence; intermittent-demand distributions; obsolete and at-risk stock.
- `06_cost-and-working-capital.md` — the full cost taxonomy; holding-cost composition; DIO/DSO/DPO
  with correct denominators; supply chain finance and term stretching; total landed cost.
- `07_manufacturing-basics.md` — manufacturing archetypes; MPS/MRP/capacity; BOM accuracy and
  error propagation; routing and queue time; OEE and its misuse; ISA-95 data ownership.
- `08_transport-and-incoterms.md` — mode economics; the eleven Incoterms 2020 rules; landed-cost
  composition; duty and CBAM; carrier liability regimes.
- `09_warehousing-and-sourcing.md` — warehouse process flow and labour arithmetic; automation
  payback; the Kraljic portfolio and its critiques; category strategy.
- `10_archetypes-and-metric-conflicts.md` — functional/innovative and efficient/responsive
  alignment; Goodhart-style definition gaming; paired metrics; cost-of-compromise arithmetic.

### Changed
- The module index status table now reports 10 `complete`, 0 `planned`, and the module total.
- `VALIDATION_STATUS.md` records topic-level production as in progress: 10 of 476 units written.

### Notes
- Style follows the clinical-foundations exemplar: theory-dense, one short code or calculation
  block per file, evidence graded A–E. Several subagent drafts corrected claims that could not be
  sourced at first hand (Walmart OTIF targets, Incoterms revision status, OEE benchmarks,
  forecast-averaging claims) rather than repeating them.

## 2026.5.0 — 27 October 2026

### Changed — structural reorganization
- The 48 module folders under `modules/` are now grouped into **seven part folders**:
  `01_foundations` (7), `02_methods` (16), `03_commerce` (4), `04_value_governance` (4),
  `05_labs_electives` (2), `06_domain_electives` (10), `07_engineering_electives` (5).
- Module numbers are unchanged. The parts are a navigation layer, not a renumbering: every
  reference to Modules 00–45, every prerequisite, and every week mapping still resolves.
- All 48 module `_index.md` files updated: the survey-file link moved from `../../course/` to
  `../../../course/`, and the `Folder:` metadata now records the full part path.
- All 48 module `_index.md` version stamps moved to 2026.5.

### Added
- Seven part-level `_index.md` files. Each states the part's purpose, lists its modules with topic
  counts, draws the intra-part dependency sequence, and records what the part deliberately excludes.
- `README.md` at the package root: a single entry point with the directory tree, a "start here"
  table, the seven parts, the core term sequence, package counts, and the evidence conventions.

### Fixed
- `modules/README.md` linked to `../MODULE_EXPANSION_SPEC.md`, which does not exist at that path;
  it now points to `../course/MODULE_EXPANSION_SPEC.md`.
- `MODULE_EXPANSION_SPEC.md` folder-layout tree, agent write-scope rule, and verification commands
  updated for the part-folder depth (`modules/*/*/[0-9][0-9]_*.md`).
- `README.md` and `VALIDATION_STATUS.md` no longer describe 48 flat folders.

## 2026.4.0 — 27 October 2026

### Added
- Module 41 — Maritime, rail, and air cargo operations engineering: berth and quay crane capacity, yard dwell and rehandles, gate and hinterland evacuation, rail paths and shunting, ULD build-up and cutoffs, disruption recovery, terminal simulation, dangerous goods and ISPS, automation payback.
- Module 42 — Treasury, credit, and financial supply chain modelling: cash conversion cycle, driver-based cash forecasting, PD/LGD/EAD expected loss, trade finance instruments, reverse factoring and dynamic discounting, FX and interest-rate hedging, NPV and cost of capital, credit policy, KYC/AML/sanctions, supply chain finance disclosure.
- Module 43 — Security engineering for supply chains: STRIDE threat modelling, attack trees and ALE, OT and the Purdue model, zero trust and segmentation, third-party risk, SBOM and software provenance, cargo crime and C-TPAT/AEO, payment fraud, incident response and continuity, standards mapping.
- Module 44 — MLOps and production analytics engineering: lifecycle framing, versioning and reproducibility, experiment tracking, point-in-time features and leakage, orchestration, serving patterns, CI/CD/CT, drift and delayed-label monitoring, retraining and champion–challenger, model governance and platform economics.
- Module 45 — Plant layout, facility design, and detailed engineering interfaces: facility types and flow, systematic layout planning, storage and dock capacity, handling equipment and automation, P&ID and isometrics, MTO and revision control, safety and fire, layout simulation, design for supply chain.
- Five matching `modules/41_*`–`modules/45_*` index folders with 50 topic units.

### Changed
- Program package is now 48 module files and 48 module index folders. Modules 00–30 keep their numbering; 31–40 are post-core domain electives and 41–45 are cross-cutting engineering electives.
- `README.md`, `COURSE_OUTLINE.md`, `COMPETENCY_MAP.md`, `VALIDATION_STATUS.md`, `_course_spec.md`, `MODULE_EXPANSION_SPEC.md`, `INSTRUCTOR_KIT.md`, `GRADING_CALIBRATION.md`, `modules/README.md`, and `TOPIC_COVERAGE_MAP.md` updated for the new module set.
- `TOPIC_COVERAGE_MAP.md` now maps 93 topics; the former engineering gaps (maritime/rail/air operations, treasury and credit, security engineering, standalone MLOps, plant layout and detailed engineering) are closed by dedicated modules.

### Fixed
- Module 40's closing footer now links forward to Module 41 instead of claiming to end the elective set.

## 2026.3.0 — 27 October 2026

### Added
- Module 31 — Aftermarket, spare parts, and service supply chains: installed base, failure demand, criticality, METRIC sparing, repair loop, field service, availability contracts.
- Module 32 — Food, agriculture, and perishables: harvest and yield, perishable forecasting, FEFO, newsvendor with salvage, cold chain, HACCP/FSMA 204, food waste.
- Module 33 — Healthcare provider and life sciences: item master and UDI, par levels, OR case carts, consignment, shortages and recalls, clinical trial supply.
- Module 34 — Energy, utilities, and capital projects: critical spares, outage kitting, long-lead equipment, bulk logistics, earned value, commodity exposure, storm response.
- Module 35 — Public sector, defence, and humanitarian logistics: public procurement, whole-life cost, PBL, stockpiles, pipeline, last mile, cash versus in-kind, donor reporting.
- Module 36 — Global freight, ports, and trade policy: Incoterms, landed cost, classification and origin, port congestion, demurrage, freight procurement, chokepoints, CBAM and sanctions.
- Module 37 — Circular economy and reverse networks: disposition hierarchy, grading, reverse network design, refurbishment, fraud, packaging reuse, battery second life, circular metrics.
- Module 38 — Process industries and commodities: continuous and batch planning, yield and mass balance, blending LP, campaigns, bulk storage, hedging, laytime, debottlenecking, process safety.
- Module 39 — Mining, metals, and forestry: reserves and grade, cut-off, blending, extraction, pit-to-port, offtake, harvest scheduling, social licence and closure.
- Module 40 — Telecom, technology hardware, and aerospace: configuration and BOM, allocation, EMS, NPI, network rollout, channel sell-through, AOG pools, obsolescence, export control.
- Ten matching `modules/31_*`–`modules/40_*` index folders with 96 topic units.

### Changed
- Program package is now 43 module files and 43 module index folders. Modules 00–30 keep their numbering; 31–40 are post-core domain electives.
- `README.md`, `COURSE_OUTLINE.md`, `COMPETENCY_MAP.md`, `VALIDATION_STATUS.md`, `_course_spec.md`, `MODULE_EXPANSION_SPEC.md`, `modules/README.md`, and `TOPIC_COVERAGE_MAP.md` updated for the new module set.
- `TOPIC_COVERAGE_MAP.md` now maps 88 topics; the former business-domain gaps (energy, healthcare, agriculture, public sector, freight, reverse flows, service, process industries, mining, technology) are closed by dedicated modules. Remaining gaps are explicitly out-of-scope boundaries.

### Fixed
- The coverage audit no longer claims that business-domain depth is complete while offering only a "build a primer" method for half the economy.

## 2026.2.1 — 27 October 2026

### Added
- `TOPIC_COVERAGE_MAP.md`: 78 canonical topics mapped to modules and depth, with 10 known gaps stated explicitly.
- Module 15 §4.16: S&OP, IBP, and S&OE cycle, a worked demand–supply reconciliation, maturity model, and S&OP-specific failure modes.
- Module 10 §4.15: distribution requirements planning, deployment, and shortage-allocation rules with a worked example.
- Module 25 §4.5: supply chain finance instruments, including reverse factoring, dynamic discounting, and inventory finance.
- Module 27 §5.4–5.5, §6.6–6.8: licence and provenance audit, benchmark reproduction checklist, tool selection by constraint, build versus buy, and TCO.
- Module 28 §4.5–4.6, §7.7: evidence audit, counter-case discipline, and a pharmaceutical cold-chain deep dive.
- Module 29 §10A–10D: build-a-primer method, readiness diagnostic, thirty-minute sector interview, and primer quality checklist.
- Module 30 §3B–3D, §13B–13C, §14.2–14.4: data contracts, cost guardrails, local-only path, architecture review, demo script, lab map, and portfolio checklist.

### Changed
- Module 29 expanded to seven applied labs and 18 interview questions; module 30 expanded to 550 lines with the integration and review material.
- Module 25 and 26 lab numbering corrected from inherited 23.x and 24.x to 25.x and 26.x.
- `README.md`, `COURSE_OUTLINE.md`, and `VALIDATION_STATUS.md` now reference the coverage map and state the module-index status.

### Fixed
- Coverage gaps identified by audit: S&OP had only passing mentions; DRP and supply chain finance were not named topics; responsible-AI language was implicit rather than explicit.

## 2026.2.0 — 27 October 2026

### Added
- Module 21 — E-commerce supply chain analytics.
- Module 22 — Quick commerce and on-demand delivery.
- Module 23 — Omnichannel fulfillment and returns.
- Module 24 — Recommendation systems and personalization.
- Module 28 — Domain case library and sector playbooks.
- Domain-project assessment replacing the capstone: 13-artifact dossier and individual defense.

### Changed
- Module numbering after the removals was made contiguous: product/economics moved to 25, quality/model risk to 26, datasets to 27, domain cases to 28, electives to 29, labs to 30.
- `COURSE_OUTLINE.md`, `README.md`, `COMPETENCY_MAP.md`, `INSTRUCTOR_KIT.md`, `VALIDATION_STATUS.md`, and `GRADING_CALIBRATION.md` were rewritten around the new domain map.
- Term 3 is now "AI, Channels, and Domain Analytics"; assessment is labs 25%, case critiques 20%, midterm forecast dossier 20%, track playbook 20%, domain project 15%.
- Program outcomes now include e-commerce/quick-commerce/omnichannel fulfillment, recommendation systems, domain cases, and sector playbooks.

### Removed
- Module 21 — Deployment and MLOps (monitoring, drift, and incident content retained in Modules 19, 26, and 30).
- Module 22 — Privacy, security, ethics, and responsible AI (retained in Modules 26 and 29).
- Module 25 — Capstone and hackathons (replaced by the domain project).
- Module 26 — Career, portfolio, and interviews (interview questions retained in every module).
- Module 28 — Sources and further reading (source ladders retained in Modules 27 and 28).

### Known issues
- Topic-level files under `modules/` are indexed but not yet written.
- External review and learner pilot are pending.

## 2026.1.0 — 27 October 2026

### Added
- Full program architecture: 40 weeks, three terms, three tracks, 33 module files.
- Module 00 orientation and 2026 industry map.
- Modules 01–28 covering supply chain foundations, data, statistics, machine learning, forecasting, inventory, network design, transportation, warehousing, procurement, resilience, manufacturing, pricing, assortment, AI, control towers, simulation, e-commerce, quick commerce, omnichannel, recommendations, product economics, model risk, datasets, and domain cases.
- Advanced electives 29–30 for industry primers and production labs.
- `_research/industry_brief.md` with dated 2025–2026 sources.
- `COMPETENCY_MAP.md`, `VALIDATION_STATUS.md`, `MODULE_EXPANSION_SPEC.md`, and `_course_spec.md`.

### Design decisions
- Forecasting, inventory, and optimization are treated as one decision system rather than separate specialties.
- Every technical module requires a baseline comparison and a cost or service interpretation.
- Regulation and standards are integrated into domain modules, not isolated in an appendix.
- Vendor and analyst claims are labelled and dated.
- The capstone requires a decision, a baseline, temporal validation, an impact estimate, and a defense.
- The 2026.2.0 domain project applies the same standard to an e-commerce, quick-commerce, omnichannel, or sector case.

### Known issues
- External review and learner pilot are pending.
- Some regulatory dates and market figures are dated snapshots and require re-verification.
- Module-level line counts vary; the expanded topic-level edition is planned but not produced.
- Labs reference public datasets whose licenses and availability may change.

## Planned

- 2026.1.1: link and consistency pass; lab starter notebooks.
- 2026.2.0: expanded topic files under `modules/` per `MODULE_EXPANSION_SPEC.md`.
- 2026.2.1: instructor kit with anchor solutions and grading calibration.
- 2027.1.0: regulatory refresh and cohort feedback.
