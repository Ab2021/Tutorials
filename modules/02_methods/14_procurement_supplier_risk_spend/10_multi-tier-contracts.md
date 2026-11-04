# Multi-Tier Mapping and Contract Data

> Module 14 — Procurement, Supplier Risk, and Spend Analytics
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: topics 01–09; basic graph concepts; cost accounting
> Version 2026.5

*Tier-n visibility, Bill of Materials propagation, contract clause extraction, commitments, and regulatory due diligence.*

---

## 1. Why this topic exists

Procurement organizations historically managed risk and commercial terms at the Tier-1 boundary.
A buyer negotiated pricing, service level agreements (SLAs), and quality targets with direct commercial counterparties.
Invoices were matched to purchase orders, receipts were recorded at the factory dock, and supplier performance was evaluated on direct delivery timeliness and invoice accuracy.
This direct-relationship mental model collapses when disruptions, legal liabilities, or cost escalations originate deeper in the supply chain.

Consider an industrial manufacturer sourcing a critical electro-mechanical motor drive assembly from two separate Tier-1 integrators.
The category manager reports a dual-sourcing strategy with a 50/50 volume allocation, believing the company is insulated from catastrophic supplier failure.
In reality, both Tier-1 integrators source their custom application-specific integrated circuit (ASIC) from the exact same semiconductor packaging facility in Hsinchu, Taiwan.
That packaging facility, in turn, depends on a single high-purity quartz smelter for its wafer substrates.
When an earthquake or geopolitical embargo halts operations at the Hsinchu packaging site, both Tier-1 suppliers declare force majeure on the same day.
The buyer's perceived dual-sourcing redundancy collapses to 100 percent single-source exposure at Tier 2.

Simultaneously, the regulatory environment has shifted from voluntary corporate social responsibility reporting to mandatory, legally binding due diligence across upstream supply chains:
The European Union Corporate Sustainability Due Diligence Directive (CSDDD, Directive (EU) 2024/1760; amended under Omnibus I in March 2026, extending full transposition to July 2028 and focusing high-impact obligations) mandates identification, prevention, and mitigation of human rights and environmental violations across an enterprise's upstream "chain of activities."
The German Supply Chain Due Diligence Act (*Lieferkettensorgfaltspflichtengesetz* - LkSG) holds German-headquartered and large branch operations legally liable for environmental and labor violations, establishing immediate obligations for direct suppliers and requiring formal investigations whenever "substantiated knowledge" of sub-tier violations emerges.
In the United States, the Uyghur Forced Labor Prevention Act (UFLPA, 19 U.S.C. § 1307) imposes a statutory rebuttable presumption: goods manufactured wholly or in part within the Xinjiang Uyghur Autonomous Region are presumed produced with forced labor and are prohibited from entry into the United States.
Importing companies must produce forensic, batch-level chain-of-custody documentation tracing raw materials (such as metallurgical grade silicon, aluminum, or cotton) through every transformation step across Tier 4, Tier 3, Tier 2, and Tier 1 to clear customs detentions.

Compounding this operational and regulatory exposure is commercial contract leakage.
Enterprises manage thousands of active supplier contracts, statements of work, and side amendments stored as unstructured PDF documents across shared drives and legacy contract repositories.
Key commercial terms—such as retroactive volume rebates, take-or-pay shortfall penalties, commodity index price escalators, audit rights, and force majeure exclusions—are rarely captured in structured ERP databases.
As a result, procurement teams fail to claim millions of dollars in earned year-end volume rebates, pay unauthorized price increases that violate contractual index bands, and find themselves powerless to audit or hold Tier-1 suppliers accountable when sub-tier disruptions occur.

This topic provides the analytical foundations to solve these challenges:
It formalizes multi-tier supply networks as directed heterogeneous graphs, connects finished product demand to upstream components via multi-level Bill of Materials (BOM) propagation, extracts contractual obligations into verifiable data structures using structured NLP and large language model (LLM) pipelines with rigorous statistical sampling, models non-linear rebate curves and commitment penalties, and operationalizes regulatory due diligence across complex global supply chains.

## 2. Learning outcomes

After completing this topic, you can:

1. **Construct and traverse** a multi-tier supply network graph linking end items, sub-assemblies, components, raw materials, supplier legal entities, and physical manufacturing sites.
2. **Propagate** demand and lead-time shocks through multi-level Bills of Materials to calculate Time-to-Survive (TTS), Time-to-Recover (TTR), and Revenue-at-Risk (RaR).
3. **Detect hidden concentration** and common-mode failure points where multiple Tier-1 suppliers share single-source sub-tier facilities, foundries, or geographic choke points.
4. **Formulate and evaluate** non-linear contract mechanisms, distinguishing retroactive (dollar-one) volume rebates, incremental (marginal) tiered rebates, and take-or-pay shortfall penalties.
5. **Reconcile** earned contract rebates against accounts payable invoice streams and general ledger balances to detect and eliminate commercial contract leakage.
6. **Design and calibrate** a contract information extraction pipeline using structured JSON schemas, clause-level provenance tracking, and stratified statistical verification sampling.
7. **Calculate** Wilson score confidence intervals for contract extraction precision and recall, establishing statistically defensible audit boundaries.
8. **Operationalize** mandatory ESG and trade compliance due diligence workflows compliant with CSDDD, LkSG, UFLPA, and Greenhouse Gas Protocol Scope 3 Category 1 standards.

## 3. The core mental model

A supply chain is neither a simple linear chain nor an unconstrained graph; it is a layered, bipartite network linking commercial organizations to physical transformation nodes governed by legal contracts.
To model multi-tier risk and contract commitments, you must decouple three distinct layers:
1. **The Product Layer (Bill of Materials):** A hierarchical directed acyclic graph (DAG) defining how raw materials combine into sub-components, sub-assemblies, and finished goods, parameterized by unit consumption rates and manufacturing cycle times.
2. **The Commercial Layer (Contracts and Entities):** A relational network linking buyer legal entities to vendor legal entities via Master Services Agreements (MSAs), purchase orders, price schedules, rebate agreements, and flow-down clauses.
3. **The Physical Layer (Sites and Logistics):** A spatial network of mines, smelters, fabrication plants, packaging facilities, warehouses, and freight corridors, characterized by capacity limits, inventory buffers, and geographic exposure.

```text
[ PRODUCT LAYER: BILL OF MATERIALS ]
Finished Product (OEM Model AP-900)
    │
    ├── (1:1) Sub-Assembly: Motor Drive Controller (MDC-04)
    │       │
    │       ├── (1:1) Component: Custom ASIC (ASIC-550)
    │       │       │
    │       │       └── (1:1) Precursor: High-Purity Silicon Wafer
    │       │
    │       └── (1:1) Component: Micro-Gearbox (G-120)
    │               │
    │               └── (1:1) Raw Material: Medical-Grade Titanium Alloy
    │
    └── (1:1) Sub-Assembly: Monitor Station (MS-200)
            │
            └── (1:1) Component: Power Regulator Board (PRB-12)
                    │
                    └── (1:1) Component: Custom ASIC (ASIC-550) <── [HIDDEN COMMON COMPONENT]

═══════════════════════════════════════════════════════════════════════════════════════════
[ COMMERCIAL LAYER: CONTRACTS, ENTITIES & REBATES ]
OEM Enterprise (Legal Buyer)
    │
    ├── MSA #2024-V: VeloDrive Systems Sp. z o.o. (Poland)
    │     ├── Price: $420.00/unit | Minimum Commitment: 20,000 units/year
    │     ├── Retroactive Rebate: Tier 2 (18k-23k) = 3.5%, Tier 3 (23k-28k) = 6.0%
    │     ├── Take-or-Pay Shortfall Penalty: $160.00/unit
    │     └── Clause Provenance: [Doc_MSA_VeloDrive_v2.pdf, Page 14, Para 3]
    │
    └── MSA #2023-C: CardioSense Ltd (Ireland)
          ├── Price: $280.00/unit | Fixed Unit Pricing
          └── Clause Provenance: [Doc_MSA_CardioSense_Final.pdf, Page 8, Para 2]

═══════════════════════════════════════════════════════════════════════════════════════════
[ PHYSICAL LAYER: SITES, INVENTORY BUFFERS & DISRUPTION ]
Disruption Event: Seismic Halt at GlobalWafer Fab Site F-3 (Hsinchu, Taiwan)
  ├── Time-to-Recover (TTR) = 16.0 weeks
  │
  ├── Physical Propagation:
  │     GlobalWafer Site F-3 (Taiwan) [Buffer: 0 wks]
  │         └── SilicoCore Fab (Dresden, Germany) [Buffer: 3 wks ASICs]
  │                 ├── VeloDrive Plant (Wroclaw, Poland) [Buffer: 4 wks MDC-04]
  │                 │       └── OEM Assembly Plant (Eindhoven) [Buffer: 3 wks FG]
  │                 │             Total Time-to-Survive (TTS) = 3 + 4 + 3 = 10.0 wks
  │                 │             Disruption Gap = TTR - TTS = 16.0 - 10.0 = 6.0 wks
  │                 │
  │                 └── CardioSense Plant (Galway, Ireland) [Buffer: 4 wks MS-200]
  │                         └── OEM Assembly Plant (Eindhoven) [Buffer: 3 wks FG]
  │                               Total TTS = 3 + 4 + 3 = 10.0 wks
  │                               Disruption Gap = 16.0 - 10.0 = 6.0 wks
  │
  └── Financial Cascade:
        ├── AP-900 Volume drops from 24,000 to 21,120 units (Loss: 2,880 units = $9.22M Retail)
        ├── Contract Rebate Cliff: Drops from Tier 3 (6.0%) to Tier 2 (3.5%) (Lost Rebate: $221.76k)
        └── MS-200 Volume drops by 1,800 units (Loss: 1,800 units = $3.24M Retail)
              Total Enterprise Revenue-at-Risk = $12.46M
```

When an interruption occurs in the physical layer, its operational impact is governed by the product layer's BOM structure and inventory buffers.
Its ultimate financial and legal impact, however, is determined by the commercial layer's contracts.
A disruption that reduces volume across a rebate threshold triggers a non-linear margin loss (the rebate cliff).
Conversely, if an unpredicted disruption forces volumes below contractual minimums, the buyer may trigger take-or-pay shortfall liabilities unless protected by excusable force majeure clauses.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Tier-1 supplier | Direct commercial counterparty invoicing the buyer under a purchase order | Assuming the Tier-1 entity performs all manufacturing |
| Tier-2 / Tier-N supplier | Upstream subcontractor, component fabricator, or raw material producer supplying an intermediary | Treating Tier-N vendors as irrelevant because no direct contract exists |
| Bill of Materials (BOM) | Hierarchical engineering tree defining the components, sub-assemblies, and raw materials needed for an end item | Confusing an engineering BOM (EBOM) with a manufacturing BOM (MBOM) |
| BOM explosion | Recursive algorithmic traversal calculating gross component requirements across all hierarchy levels | Calculating gross requirements without netting pipeline or safety stocks |
| Cumulative lead time | Longest duration along the BOM critical path required to procure materials and assemble an end item, weeks | Summing lead times across parallel BOM branches instead of taking the maximum |
| Time-to-Survive (TTS) | Duration an enterprise or node can satisfy demand during a total upstream supply stoppage, weeks | Omitting downstream intermediate buffer stocks or assuming infinite finished goods |
| Time-to-Recover (TTR) | Time required for a disrupted facility to resume full operational capacity or qualify an alternate site, weeks | Quoting factory repair time without factoring in equipment requalification and shipping |
| Revenue-at-Risk (RaR) | Gross revenue lost during the unbuffered disruption window where $TTR > TTS$, USD | Confusing gross revenue loss with net gross margin or EBITDA impact |
| Common-mode failure | Structural failure where multiple supposedly independent suppliers halt due to a shared sub-tier dependency | Assuming dual sourcing at Tier 1 guarantees independent operational resilience |
| Contract extraction | Algorithmic parsing and structuring of legal text from PDFs into relational or JSON data schemas | Trusting raw LLM text outputs without human verification sampling |
| Clause provenance | Cryptographic and locational audit trail linking extracted structured data to document, page, and paragraph | Storing an extracted number without preserving the source document reference |
| Retroactive rebate | Contractual rebate where achieving a volume tier applies the rebate percentage to all units purchased, USD | Assuming rebates apply only to incremental units above the threshold |
| Incremental rebate | Contractual tiered rebate where higher discount percentages apply only to units within each specific volume bracket, USD | Confusing marginal bracket pricing with retroactive "dollar-one" discounts |
| Rebate cliff | The mathematical discontinuity in retroactive rebates where purchasing one fewer unit causes a catastrophic loss of total rebate, USD | Treating average unit cost as continuous and failing to observe negative marginal cost |
| Take-or-pay | Clause obligating buyer to purchase a minimum volume or pay a liquidated shortfall penalty, USD/unit | Calling a take-or-pay commitment a flexible volume forecast |
| Contract leakage | Value lost when negotiated contractual terms, rebates, or discounts are not invoiced, collected, or enforced, USD | Restricting leakage analysis solely to invoice price variances |
| CSDDD | Corporate Sustainability Due Diligence Directive (Directive (EU) 2024/1760; Omnibus I March 2026), European Union | Treating CSDDD as a non-binding ESG disclosure framework rather than civil liability |
| LkSG | *Lieferkettensorgfaltspflichtengesetz* (German Supply Chain Act), statutory due diligence regime | Assuming LkSG obligations stop at the direct Tier-1 supplier boundary |
| UFLPA | Uyghur Forced Labor Prevention Act (19 U.S.C. § 1307), United States trade law with rebuttable presumption | Believing standard supplier certificates of origin overcome UFLPA detentions |
| Scope 3 Category 1 | Upstream emissions from purchased goods and services under the GHG Protocol Corporate Standard, $\text{tCO}_2\text{e}$ | Treating secondary spend-based EEIO estimates as primary product carbon footprints |
| Flow-down clause | Contractual provision requiring a Tier-1 supplier to bind its subcontractors to identical legal standards | Assuming code-of-conduct signing automatically enforces sub-tier compliance |
| Audit rights clause | Enforceable contract term permitting the buyer to inspect supplier premises, books, and sub-tier records | Treating an audit clause as an operational audit program without exercising it |
| Force majeure carveout | Specific contract exclusion establishing which events excuse delivery failure and whether sub-tier failure qualifies | Assuming any natural disaster automatically legally excuses delivery delays |
| Wilson score interval | Asymmetric binomial confidence interval robust for small samples and extreme success proportions, unitless | Applying a Gaussian normal approximation to a sample with 0 or 100 percent success |

## 5. The mechanics, step by step

### Step 1: Formalize the multi-tier supply network graph

A supply chain network is represented mathematically as a directed, attributed heterogeneous graph $G = (V, E)$, where the node set $V$ consists of three disjoint subsets:
- $V_{\text{entity}}$: Legal commercial entities (OEM corporate parent, Tier-1 corporations, Tier-2 subcontractors, trading houses).
- $V_{\text{facility}}$: Physical transformation and storage sites (mines, smelters, wafer fabs, packaging facilities, assembly plants, distribution centers).
- $V_{\text{part}}$: Physical items across the lifecycle (raw materials, components, sub-assemblies, finished stock-keeping units).

The edge set $E$ captures the relationships between these nodes:
- $E_{\text{BOM}} \subseteq V_{\text{part}} \times V_{\text{part}}$: Directed edges representing Bill of Materials structure. An edge $(p_i, p_j) \in E_{\text{BOM}}$ carries an attribute $q_{ij} \in \mathbb{R}^+$, denoting the quantity of component $p_i$ required to produce one unit of assembly $p_j$.
- $E_{\text{production}} \subseteq V_{\text{part}} \times V_{\text{facility}}$: Edges specifying which physical facility manufactures a given part, parameterized by production cycle time $\tau_{\text{prod}}$ and weekly capacity $C_{\text{max}}$.
- $E_{\text{commercial}} \subseteq V_{\text{entity}} \times V_{\text{entity}}$: Edges representing legal contracts, master purchase agreements, and invoicing channels, carrying contract identifiers, pricing structures, rebate rules, and governing law.
- $E_{\text{ownership}} \subseteq V_{\text{entity}} \times V_{\text{facility}}$: Edges linking corporate entities to the physical assets they own or lease.

```text
[Entity: VeloDrive Systems Sp. z o.o.] ──(ownership)──► [Facility: Wroclaw Plant #1]
             │ (commercial contract)                              │ (produces)
             ▼                                                    ▼
   [Entity: Apex Medical NV] ──(procures)──► [Part: MDC-04 Motor Controller]
                                                         ▲ (part_of BOM)
                                                         │
                                              [Part: ASIC-550 Silicon]
                                                         ▲ (produced_at)
                                                         │
                                            [Facility: Hsinchu Fab F-3]
                                                         ▲ (owned_by)
                                                         │
                                        [Entity: GlobalWafer Foundry Corp]
```

Differentiating legal entities from physical sites is critical.
A buyer might issue purchase orders to a Swiss trading subsidiary (commercial entity), while the physical part is produced at a single plant in Wroclaw, Poland (facility node), using materials processed in Hsinchu, Taiwan (sub-tier facility node).
Analyzing concentration at the legal entity level creates a dangerous blind spot.

### Step 2: Traverse multi-level Bills of Materials and propagate demand

To translate finished product demand into component-level requirements across the network, execute a multi-level BOM explosion.
Let $d \in \mathbb{R}^{|V_{\text{part}}|}$ denote the gross demand vector for end items over a given planning horizon.
The BOM relationship is structured as an upper-triangular or strictly acyclic direct-requirement matrix $A$, where entry $A_{ij} = q_{ij}$ represents the direct consumption of part $i$ per unit of part $j$.

The total gross requirement vector $r$ across all hierarchy levels is given by the Leontief-style upstream explosion:
$$r = d + A d + A^2 d + \dots + A^K d = (I - A)^{-1} d$$
where $K$ is the maximum depth of the BOM tree, and $I$ is the identity matrix.
Because engineering BOMs are strictly directed acyclic graphs, $A$ is nilpotent ($A^{K+1} = 0$), guaranteeing that the inversion $(I - A)^{-1}$ is finite and computable via topological sort.

Along with gross unit requirements, propagate cumulative lead times.
For any component or assembly $p_j$, its cumulative procurement and manufacturing lead time $\text{CLT}(p_j)$ is computed recursively from the base raw materials up to the finished good:
$$\text{CLT}(p_j) = \text{LT}_{\text{internal}}(p_j) + \max_{p_i \in \text{Children}(p_j)} \left( \text{CLT}(p_i) + \text{LT}_{\text{transit}}(p_i, p_j) \right)$$
where $\text{LT}_{\text{internal}}(p_j)$ is the processing time at the facility producing $p_j$, and $\text{LT}_{\text{transit}}(p_i, p_j)$ is the freight transit time from the facility producing component $p_i$.

### Step 3: Compute disruption dynamics: Time-to-Survive, Time-to-Recover, and Revenue-at-Risk

When an upstream facility node $f_k \in V_{\text{facility}}$ suffers an unplanned stoppage (due to natural disaster, labor strike, regulatory shutdown, or cyberattack), determine how long the downstream enterprise can sustain production before finished goods assembly halts.

1. **Time-to-Recover ($\text{TTR}_{f_k}$):** The time required for facility $f_k$ to restore full operational output, or for the buyer and Tier-1 partners to identify, tool, qualify, and ramp an alternate qualified facility.
   $\text{TTR}$ must include physical plant reconstruction, cleanroom recertification, regulatory change notification (e.g., medical device FDA 510(k) or automotive PPAP Level 3 approvals), and logistics re-routing.
2. **Time-to-Survive ($\text{TTS}_{f_k \to p_{\text{end}}}$):** The maximum duration that finished product $p_{\text{end}}$ can continue to be produced and shipped to customers following a complete shutdown of facility $f_k$.
   Let $\mathcal{P}(f_k, p_{\text{end}})$ denote the set of all directed supply paths from facility $f_k$ to the finished product assembly plant.
   Along each path $\pi \in \mathcal{P}(f_k, p_{\text{end}})$, the total pipeline buffer is the sum of inventory stocks held at each intermediate stage:
   $$\text{TTS}(\pi) = \sum_{v \in \pi \setminus \{f_k\}} \frac{I_v}{D_v}$$
   where $I_v$ is the inventory (on-hand plus in-transit) at node $v$, and $D_v$ is the average daily consumption rate at node $v$.
   Because an assembly line halts when *any* required component path runs dry, the system-level Time-to-Survive for product $p_{\text{end}}$ relative to facility $f_k$ is governed by the minimum path:
   $$\text{TTS}(f_k \to p_{\text{end}}) = \min_{\pi \in \mathcal{P}(f_k, p_{\text{end}})} \text{TTS}(\pi)$$
3. **Disruption Duration ($\Delta t_{\text{disrupt}}$):** The unbuffered downtime experienced at final assembly:
   $$\Delta t_{\text{disrupt}}(f_k \to p_{\text{end}}) = \max\left(0, \text{TTR}_{f_k} - \text{TTS}(f_k \to p_{\text{end}})\right)$$
4. **Revenue-at-Risk ($\text{RaR}$):** If $\Delta t_{\text{disrupt}} > 0$, the enterprise experiences unrecoverable stockouts.
   Letting $W(p_{\text{end}})$ denote the weekly finished goods run rate and $P_{\text{retail}}(p_{\text{end}})$ denote the net commercial selling price:
   $$\text{RaR}(f_k \to p_{\text{end}}) = \Delta t_{\text{disrupt}}(f_k \to p_{\text{end}}) \times W(p_{\text{end}}) \times P_{\text{retail}}(p_{\text{end}})$$
   Summing across all finished goods $\mathcal{F}$ dependent on facility $f_k$ yields the total enterprise exposure:
   $$\text{RaR}_{\text{total}}(f_k) = \sum_{p \in \mathcal{F}} \text{RaR}(f_k \to p)$$

### Step 4: Extract contract clauses into structured data with immutable provenance

Supplier contracts, amendments, schedules, and side letters contain the financial rules governing commercial liabilities during routine operations and disruptions.
Because contracts reside in unstructured PDF formats, construct an automated extraction pipeline with strict schema validation.

```text
[ Unstructured Legal PDFs ] 
(MSAs, Amendments, Pricing Exhibits, SOWs)
               │
               ▼
[ 1. Document Parsing & Layout Engine ]
(Extract text layers, OCR scanned pages, preserve tables, isolate section headers)
               │
               ▼
[ 2. Clause Segmentation & Classification ]
(Taxonomy: Base Price, Rebates, Commitments, Penalties, Audit Rights, ESG, Force Majeure)
               │
               ▼
[ 3. Constrained JSON Extraction (LLM / NLP) ]
(Extract structured parameters: volume thresholds, discount rates, penalty formulas)
               │
               ▼
[ 4. Cryptographic Provenance Anchoring ]
(Attach: document_id, amendment_num, page_num, paragraph_num, text_hash)
               │
               ▼
[ 5. Stratified Verification Sampling ] ──► [ Human Legal/Procurement Review Queue ]
(Evaluate Precision / Recall via Wilson Score)        (Flag low-confidence or high-risk terms)
               │
               ▼
[ Structured Contract Master Database ] ──► [ Enterprise ERP / Spend Analytics Engine ]
```

The structured contract schema must enforce explicit data types:
```json
{
  "contract_id": "MSA-2024-VELO-001",
  "amendment_id": "AMD-003",
  "effective_date": "2024-01-01",
  "expiration_date": "2026-12-31",
  "governing_law": "Germany",
  "supplier_entity_id": "VEND-VELO-PL",
  "part_number": "MDC-04",
  "pricing": {
    "base_currency": "USD",
    "base_unit_price": 420.00,
    "price_firmness": "Fixed through 2026-12-31",
    "index_escalators": []
  },
  "commitments": {
    "annual_minimum_volume": 20000,
    "take_or_pay_penalty_per_unit": 160.00,
    "capacity_guarantee_weekly": 600
  },
  "volume_rebates": {
    "rebate_mechanism": "retroactive_dollar_one",
    "calculation_period": "annual_calendar",
    "tiers": [
      {"min_volume": 0, "max_volume": 17999, "rebate_pct": 0.000},
      {"min_volume": 18000, "max_volume": 22999, "rebate_pct": 0.035},
      {"min_volume": 23000, "max_volume": 27999, "rebate_pct": 0.060},
      {"min_volume": 28000, "max_volume": null, "rebate_pct": 0.085}
    ]
  },
  "compliance_and_governance": {
    "right_to_audit_subtier": true,
    "audit_notice_days": 14,
    "esg_code_of_conduct_flowdown": true,
    "force_majeure_carveout_subcontractors": false
  },
  "provenance": {
    "source_file": "MSA_VeloDrive_Systems_2024_executed.pdf",
    "page_numbers": [12, 14, 18],
    "paragraph_reference": "Section 7.2 (Rebates) and Section 11.4 (Force Majeure)",
    "text_sha256": "8f3b61a9c12de45b...7e21a"
  }
}
```

Provenance is mandatory.
If an automated system asserts that a supplier owes a 6.0 percent rebate or is liable for a $160.00/unit shortfall, commercial and legal teams cannot present that claim to a vendor without citing the executed contract page, section, and exact phrasing.

### Step 5: Execute statistical verification sampling of extracted contract clauses

Automated extraction systems (including state-of-the-art LLMs) exhibit non-zero error rates.
An extraction accuracy of 95 percent means that 1 out of every 20 contractual rules in the database is incorrect.
In commercial procurement, misidentifying a retroactive rebate as incremental or hallucinating a force majeure exclusion produces catastrophic legal and financial errors.

To certify the extraction pipeline, implement a **stratified random sampling verification protocol**:
1. Partition extracted clauses into risk strata:
   - Stratum A: Commercial pricing, rebate tiers, and take-or-pay formulas (High financial risk).
   - Stratum B: Termination rights, audit rights, and force majeure carveouts (High legal risk).
   - Stratum C: Governing law, notice addresses, and dispute resolution venues (Low operational risk).
2. For each stratum, draw an independent random sample of $n \ge 100$ clauses.
3. Conduct a blinded dual-review audit where two procurement analysts verify the extracted JSON parameters against the source PDF.
4. Calculate precision, recall, and the **Wilson score confidence interval** for success proportion $p = k/n$:
   $$\text{CI}_{1-\alpha}(p) = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
   where $z = 1.95996$ for a 95% confidence level.
   If the lower bound of the 95% Wilson interval for Stratum A or Stratum B falls below the enterprise risk threshold (e.g., 90.0%), reject the extraction model for automated posting and route all clauses in that category to manual human-in-the-loop validation.

### Step 6: Model non-linear rebates, rebate cliffs, and take-or-pay commitments

Procurement contracts employ non-linear pricing structures to capture supplier economies of scale and align incentives.
Two primary rebate structures dominate commercial agreements:

#### Structure A: Incremental (Marginal) Volume Rebates
The supplier grants higher rebate percentages only to the incremental units purchased within each defined tier.
Let $0 = T_0 < T_1 < T_2 < \dots < T_m$ be the volume thresholds, and let $r_1, r_2, \dots, r_m$ be the corresponding rebate percentages ($0 \le r_k < 1$).
For an annual purchase volume $Q$ such that $T_{k-1} \le Q < T_k$, the total rebate earned $R_{\text{inc}}(Q)$ is:
$$R_{\text{inc}}(Q) = P_{\text{base}} \sum_{j=1}^{k-1} r_j (T_j - T_{j-1}) + P_{\text{base}} r_k (Q - T_{k-1})$$
Under incremental rebates, the total cost function $C(Q) = P_{\text{base}} Q - R_{\text{inc}}(Q)$ is continuous, piecewise-linear, and convex.
The marginal cost of the next unit purchased is strictly positive:
$$\text{MC}_{\text{inc}}(Q) = P_{\text{base}} (1 - r_k) > 0$$

#### Structure B: Retroactive ("Dollar-One") Volume Rebates
Once the buyer's cumulative volume $Q$ crosses threshold $T_k$, the rebate rate $r_k$ applies retroactively to *all* units purchased from unit 1:
$$R_{\text{retro}}(Q) = P_{\text{base}} \cdot r_k \cdot Q \quad \text{for } T_k \le Q < T_{k+1}$$
Net spend under retroactive rebates is:
$$C_{\text{retro}}(Q) = P_{\text{base}} Q (1 - r_k)$$
This formulation introduces severe mathematical discontinuities ("rebate cliffs") at each threshold $T_k$.
Immediately prior to threshold $T_k$, at volume $Q = T_k - 1$, the net spend is:
$$C_{\text{retro}}(T_k - 1) = P_{\text{base}} (T_k - 1)(1 - r_{k-1})$$
At volume $Q = T_k$, net spend collapses to:
$$C_{\text{retro}}(T_k) = P_{\text{base}} T_k (1 - r_k)$$
The marginal cost of purchasing the $T_k$-th unit is:
$$\Delta C = C_{\text{retro}}(T_k) - C_{\text{retro}}(T_k - 1) = P_{\text{base}} \left[ 1 - r_k - (T_k - 1)(r_k - r_{k-1}) \right]$$
For realistic enterprise volumes ($T_k \ge 10,000$) and typical rebate deltas ($r_k - r_{k-1} \ge 0.02$), this marginal cost is **substantially negative**.
Purchasing one additional unit reduces total cash outflow by tens or hundreds of thousands of dollars.

```text
Net Spend (USD)
  │
  │                  / (Tier 1: 0% rebate)
  │                 /
  │                /
  │               /
  │              x  <- Net spend = $9.66M (at 22,999 units, 3.5% rebate)
  │              │
  │              │  REBATE CLIFF: Marginal cost = -$241,095
  │              │
  │              o  <- Net spend = $9.08M (at 23,000 units, 6.0% retroactive rebate)
  │             /
  │            / (Tier 2: 6.0% retroactive rebate)
  │           /
  └──────────┴─────────────────────────────── Volume (Units)
           23,000
```

#### Structure C: Take-or-Pay and Minimum Commitments
If actual volume $Q_{\text{actual}}$ falls below contractual minimum commitment $Q_{\text{min}}$, the buyer must pay a shortfall penalty:
$$\text{Liability}_{\text{penalty}} = \max\left(0, Q_{\text{min}} - Q_{\text{actual}}\right) \times P_{\text{penalty}}$$
where $P_{\text{penalty}}$ is either a stipulated per-unit liquidated damage or the supplier's unamortized fixed cost contribution per unit:
$$P_{\text{penalty}} = P_{\text{base}} - c_{\text{avoidable}}$$
where $c_{\text{avoidable}}$ represents variable raw material and labor expenses that the supplier avoids by not producing the units.

### Step 7: Detect contract leakage and reconcile rebates to accounting ledgers

Contract leakage occurs when negotiated rebate terms are not realized in cash.
To identify and recover leakage, execute a four-way reconciliation:
1. **Purchase Orders (ERP):** Verify whether PO unit prices match contractual base prices, and whether volume tracking tags are active.
2. **Goods Receipts (GRN):** Establish the exact physical receipt dates within the calendar or contractual performance year.
3. **Accounts Payable Invoices (AP):** Aggregate gross payments made to the vendor, netting credit notes and cash discounts.
4. **General Ledger (GL) Rebate Accruals:** Compare monthly rebate accruals posted by finance against mathematically earned rebates:
   $$\text{Leakage}_{\text{unclaimed}} = R_{\text{earned}}(Q_{\text{received}}) - \left( \sum \text{Rebate Invoices} + \sum \text{Rebate Credit Notes} \right)$$
Any positive delta represents uncollected enterprise working capital that must be formally claimed via a commercial audit demand letter.

### Step 8: Operationalize multi-tier ESG, trade compliance, and audit due diligence

To comply with CSDDD, LkSG, and UFLPA, integrate multi-tier graph data with regulatory due diligence procedures:

1. **CSDDD and LkSG Risk Mapping:**
   - Map all Tier-1 suppliers and direct sub-contractors into an annual risk assessment matrix.
   - Screen facility nodes against geospatial conflict databases, water-stress indices (WRI Aqueduct), and country-level governance scores (World Bank Worldwide Governance Indicators).
   - Where "substantiated knowledge" of forced labor or environmental contamination is flagged at Tier 2 or Tier 3, LkSG Section 9 mandates an immediate human rights risk analysis, implementation of a corrective action plan (CAP), and board-level reporting.
2. **UFLPA Chain-of-Custody Dossier Construction:**
   - For every finished assembly exported to or sold in the United States containing covered materials (silicon, polysilicon, aluminum, cotton, polyvinyl chloride):
     1. Retrieve Tier-1 manufacturing batch records.
     2. Follow $E_{\text{BOM}}$ and $E_{\text{production}}$ edges upstream to extract Tier-2 and Tier-3 component lot numbers.
     3. Collect transportation bills of lading, customs declarations, raw material purchase invoices, mill test certificates, and refinery lot certificates tracing the input batch to a verified mine or smelter outside prohibited regions.
     4. Store this bundle as a cryptographic audit package linked to the finished product serial number.
3. **Scope 3 Category 1 Upstream Decarbonization:**
   - Transition from secondary spend-based greenhouse gas estimation ($Spend \times \text{EEIO factor}$) to primary product carbon footprints (PCF).
   - In contracts, embed mandatory annual reporting of primary cradle-to-gate carbon intensity ($\text{kg CO}_2\text{e}$ per unit) certified under ISO 14067 or the PACT (Partnership for Carbon Transparency) Pathfinder framework.
   - Attach carbon reduction targets to commercial scorecards, establishing contractual bonus/malus provisions linked to supplier decarbonization milestones.

## 6. Worked example

### Business context: Apex Medical Technologies

Apex Medical Technologies is an international medical equipment manufacturer producing the `Model AP-900 Infusion Workstation`, a high-reliability drug delivery platform sold to hospitals across Europe and North America at a wholesale retail price of **$3,200.00 USD** per unit.
Apex also produces a companion monitoring device, the `Model MS-200 Monitor Station`, sold at **$1,800.00 USD** per unit.
Both products are assembled at Apex’s primary manufacturing facility in Eindhoven, Netherlands.

The baseline annual production plan for 2026 is:
- **Model AP-900:** 24,000 units/year, operating on a 50-week manufacturing year (**480 units/week**).
- **Model MS-200:** 15,000 units/year (**300 units/week**).

### Bill of Materials and multi-tier supply network topology

Apex’s engineering and procurement data define the following multi-tier BOM and site dependencies:

```text
[OEM Level: Eindhoven, Netherlands]
  ├── Finished Good: AP-900 (Planned: 24,000 units/yr | 480/wk)
  │     └── Sub-Assembly: MDC-04 Motor Drive Controller (1 per AP-900)
  │           ├── Tier 1: VeloDrive Systems Sp. z o.o. (Plant: Wroclaw, Poland)
  │           │     ├── Component: G-120 Micro-Gearbox (1 per MDC-04)
  │           │     │     ├── Tier 2: MicroPrecision SpA (Plant: Valbrembo, Italy)
  │           │     │     └── Tier 3: TitanAlloy AG (Plant: Duisburg, Germany)
  │           │     │
  │           │     └── Component: ASIC-550 Motor Timing IC (1 per MDC-04)
  │           │           ├── Tier 2: SilicoCore Microelectronics GmbH (Plant: Dresden, Germany)
  │           │           └── Tier 3: GlobalWafer Foundry Corp (Fab: Hsinchu, Taiwan - Site F-3)
  │           │
  └── Finished Good: MS-200 (Planned: 15,000 units/yr | 300/wk)
        └── Sub-Assembly: PRB-12 Power Regulator Board (1 per MS-200)
              └── Tier 1: CardioSense Ltd (Plant: Galway, Ireland)
                    └── Component: ASIC-550 Motor Timing IC (1 per PRB-12) <── [SHARED TIER-2 IC]
                          ├── Tier 2: SilicoCore Microelectronics GmbH (Dresden, Germany)
                          └── Tier 3: GlobalWafer Foundry Corp (Hsinchu Site F-3)
```

Notice the hidden dependency: Apex dual-sources at Tier 1 (VeloDrive in Poland for motor drives; CardioSense in Ireland for power boards), but both Tier-1 suppliers source the identical `ASIC-550` integrated circuit from SilicoCore (Tier 2), which depends exclusively on GlobalWafer's Hsinchu Site F-3 (Tier 3) for wafer lithography and front-end fabrication.

### Contract structure: VeloDrive Systems MSA

Apex’s commercial agreement with VeloDrive Systems for the `MDC-04` contains the following extracted terms:
- **Base contract price:** **$420.00 USD** per unit.
- **Annual minimum purchase commitment ($Q_{\text{min}}$):** 20,000 units/year.
- **Take-or-pay shortfall penalty:** **$160.00 USD** per unit below 20,000 units (representing VeloDrive's unamortized cleanroom investment and fixed overhead).
- **Retroactive annual volume rebate schedule (dollar-one across all units):**
  - Tier 1: 0 to 17,999 units $\to$ 0.0% rebate ($0.00/unit)
  - Tier 2: 18,000 to 22,999 units $\to$ 3.5% rebate ($14.70/unit)
  - Tier 3: 23,000 to 27,999 units $\to$ 6.0% rebate ($25.20/unit)
  - Tier 4: $\ge 28,000$ units $\to$ 8.5% rebate ($35.70/unit)
- **Subcontractor force majeure clause:**
  "Section 11.4: Delivery failures, delays, or cost increases arising from defaults, strikes, or plant closures of Seller's subcontractors, material suppliers, or component fabricators shall not constitute an excusable Force Majeure event unless such event independently impacts Seller's primary assembly plant in Wroclaw."

### Baseline planned financial economics

Under the baseline plan of 24,000 units of AP-900:
- Gross spend with VeloDrive:
  $$\text{Gross Spend} = 24,000 \times \$420.00 = \$10,080,000.00 \text{ USD}$$
- Volume tier achieved: Tier 3 (23,000 to 27,999 units), qualifying for a 6.0% retroactive rebate.
- Earned volume rebate:
  $$\text{Rebate}_{\text{planned}} = \$10,080,000.00 \times 0.060 = \$604,800.00 \text{ USD}$$
- Net spend:
  $$\text{Net Spend} = \$10,080,000.00 - \$604,800.00 = \$9,475,200.00 \text{ USD}$$
- Net effective unit purchase price:
  $$P_{\text{net}} = \frac{\$9,475,200.00}{24,000} = \$394.80 \text{ USD/unit}$$

### The disruption event: Sub-tier foundry seismic halt

On March 15, 2026, a magnitude 6.9 earthquake strikes Hsinchu County, Taiwan, causing extensive structural damage to the cleanroom infrastructure and lithography stepper equipment at GlobalWafer Fab Site F-3.
GlobalWafer notifies SilicoCore that Site F-3 will require **16.0 weeks** to repair cleanrooms, realign mirrors, recertify tools, and clear work-in-process contamination:
$$\text{TTR}_{\text{GlobalWafer}} = 16.0 \text{ weeks}$$

### Pipeline buffer audit and Time-to-Survive calculation

Apex’s procurement analytics team queries inventory across all stages:
1. **Apex Finished Goods & Assembly (Eindhoven):** Holds 3.0 weeks of AP-900 buffer stock (**1,440 units**).
2. **VeloDrive Tier-1 Assembly (Wroclaw):** Holds 4.0 weeks of assembled MDC-04 modules and buffered ASIC-550 chips (**1,920 units**).
3. **SilicoCore Tier-2 Fab (Dresden):** Holds 3.0 weeks of packaged ASIC-550 inventory (**1,440 units**).
4. **GlobalWafer Tier-3 Fab (Hsinchu):** 0.0 weeks of finished wafer inventory (just-in-time dispatch).

Calculate total pipeline Time-to-Survive along the critical path:
$$\text{TTS}(\text{GlobalWafer} \to \text{AP-900}) = 3.0 + 4.0 + 3.0 = 10.0 \text{ weeks}$$

For the companion device MS-200:
1. **Apex Finished Goods (Eindhoven):** Holds 3.0 weeks buffer (**900 units**).
2. **CardioSense Tier-1 (Galway):** Holds 4.0 weeks buffer of PRB-12 modules and ASIC-550 chips (**1,200 units**).
3. **SilicoCore Tier-2 (Dresden):** Shared 3.0 weeks ASIC buffer dedicated to CardioSense (**900 units**).
$$\text{TTS}(\text{GlobalWafer} \to \text{MS-200}) = 3.0 + 4.0 + 3.0 = 10.0 \text{ weeks}$$

### Operational impact and production halt

Both product lines have identical 10.0-week survival buffers.
Calculate the unbuffered disruption window at final assembly:
$$\Delta t_{\text{disrupt}} = \text{TTR} - \text{TTS} = 16.0 - 10.0 = 6.0 \text{ weeks}$$
For exactly 6.0 weeks, final assembly lines in Eindhoven run out of ASIC-550 components and halt:
- **AP-900 lost production:**
  $$Q_{\text{lost, AP-900}} = 6.0 \text{ weeks} \times 480 \text{ units/week} = 2,880 \text{ units}$$
- **AP-900 total delivered units:**
  $$Q_{\text{actual, AP-900}} = 24,000 - 2,880 = 21,120 \text{ units}$$
- **MS-200 lost production:**
  $$Q_{\text{lost, MS-200}} = 6.0 \text{ weeks} \times 300 \text{ units/week} = 1,800 \text{ units}$$
- **MS-200 total delivered units:**
  $$Q_{\text{actual, MS-200}} = 15,000 - 1,800 = 13,200 \text{ units}$$

### Commercial and financial cascade: The rebate cliff

Now trace the impact into the VeloDrive commercial contract:
1. **Take-or-Pay Audit:**
   Actual delivered volume is 21,120 units.
   Because $21,120 \ge 20,000$ (the contractual minimum commitment $Q_{\text{min}}$), Apex does *not* breach the minimum volume threshold.
   Shortfall penalty owed = **$0.00 USD**.
2. **Rebate Tier Collapse:**
   Delivered volume drops from planned 24,000 units into the **18,000 to 22,999 bracket (Tier 2)**.
   The retroactive rebate rate falls from **6.0% down to 3.5%**!
3. **Financial Spend Recalculation:**
   - Disrupted gross spend:
     $$\text{Gross Spend}_{\text{disrupted}} = 21,120 \times \$420.00 = \$8,870,400.00 \text{ USD}$$
   - Earned rebate under Tier 2:
     $$\text{Rebate}_{\text{actual}} = \$8,870,400.00 \times 0.035 = \$310,464.00 \text{ USD}$$
   - Net spend:
     $$\text{Net Spend}_{\text{disrupted}} = \$8,870,400.00 - \$310,464.00 = \$8,559,936.00 \text{ USD}$$
   - Net effective unit cost:
     $$P_{\text{net, disrupted}} = \frac{\$8,559,936.00}{21,120} = \$405.30 \text{ USD/unit}$$
   Unit purchasing cost increased by **$10.50/unit** ($405.30 - $394.80) on every unit actually delivered.
4. **Lost Rebate Margin:**
   Had the Tier 3 rebate rate (6.0%) applied to the delivered 21,120 units, the rebate would have been:
   $$21,120 \times \$420.00 \times 0.060 = \$532,224.00 \text{ USD}$$
   The rebate cliff caused a direct procurement margin loss of:
   $$\text{Lost Rebate} = \$532,224.00 - \$310,464.00 = \$221,760.00 \text{ USD}$$

### Total Enterprise Revenue-at-Risk

Calculate the top-line finished goods revenue lost due to the 6-week line shutdown:
- **AP-900 wholesale revenue loss:**
  $$\text{Loss}_{\text{AP-900}} = 2,880 \text{ units} \times \$3,200.00 = \$9,216,000.00 \text{ USD}$$
- **MS-200 wholesale revenue loss:**
  $$\text{Loss}_{\text{MS-200}} = 1,800 \text{ units} \times \$1,800.00 = \$3,240,000.00 \text{ USD}$$
- **Total Enterprise Revenue-at-Risk:**
  $$\text{RaR}_{\text{total}} = \$9,216,000.00 + \$3,240,000.00 = \$12,456,000.00 \text{ USD}$$

```text
Summary of Disruption & Contract Impact:
┌───────────────────────────────────────────────────┬────────────────────────┐
│ Metric                                            │ Value                  │
├───────────────────────────────────────────────────┼────────────────────────┤
│ Disrupted Upstream Node                           │ GlobalWafer Fab F-3    │
│ Time-to-Recover (TTR)                             │ 16.0 weeks             │
│ Supply Chain Time-to-Survive (TTS)                │ 10.0 weeks             │
│ Disruption Downtime Window                        │ 6.0 weeks              │
│ Finished Goods Lost (AP-900)                      │ 2,880 units            │
│ Finished Goods Lost (MS-200)                      │ 1,800 units            │
│ Finished Goods Revenue Lost (AP-900)              │ $9,216,000.00 USD      │
│ Finished Goods Revenue Lost (MS-200)              │ $3,240,000.00 USD      │
│ Total Enterprise Revenue-at-Risk                  │ $12,456,000.00 USD     │
│ VeloDrive Actual Delivered Volume                 │ 21,120 units           │
│ Take-or-Pay Shortfall Penalty (Min 20k units)     │ $0.00 USD              │
│ Contract Rebate Tier Achieved                     │ Tier 2 (3.5% rebate)   │
│ Contract Rebate Tier Lost                         │ Tier 3 (6.0% rebate)   │
│ Actual Rebate Collected                          │ $310,464.00 USD        │
│ Rebate Margin Lost due to Cliff                   │ $221,760.00 USD        │
│ Effective Unit Purchase Price Increase            │ +$10.50 USD/unit       │
└───────────────────────────────────────────────────┴────────────────────────┘
```

Because Apex extracted and analyzed Section 11.4 of the VeloDrive MSA, legal counsel identified that sub-tier supplier failures do *not* qualify as force majeure.
Apex successfully recovered the $221,760.00 rebate loss and negotiated a $500,000 commercial settlement from VeloDrive under the agreement’s delayed-delivery liquidated damages clause.

## 7. Code, query, or calculation

The following Python 3 program implements the full multi-tier network model, Bill of Materials propagation, Time-to-Survive simulation, contract rebate and take-or-pay evaluation, and statistical extraction sampling verification.
It uses only the standard library and runs without external dependencies.

```python
"""
Multi-Tier Supply Chain Network, Contract Rebate, and Provenance Verification Engine.
Standard library Python 3.10+.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import math


@dataclass
class SupplyNode:
    node_id: str
    name: str
    tier: int  # 0: OEM finished good, 1: Tier-1, 2: Tier-2, 3: Tier-3
    category: str
    site_location: str
    inventory_weeks: float
    time_to_recover_weeks: float
    unit_cost_usd: float = 0.0


@dataclass
class SupplyEdge:
    source_id: str  # Upstream supplier/facility
    target_id: str  # Downstream consumer/part
    lead_time_weeks: float
    units_per_target: float


@dataclass
class RebateTier:
    min_volume: int
    max_volume: Optional[int]
    rebate_pct: float  # e.g., 0.06 for 6.0%


class MultiTierNetwork:
    """Directed graph representing supply network nodes, BOM edges, and facilities."""

    def __init__(self):
        self.nodes: Dict[str, SupplyNode] = {}
        self.edges: List[SupplyEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, float, float]]] = {}
        self.reverse_adjacency: Dict[str, List[Tuple[str, float, float]]] = {}

    def add_node(self, node: SupplyNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []
        if node.node_id not in self.reverse_adjacency:
            self.reverse_adjacency[node.node_id] = []

    def add_edge(self, edge: SupplyEdge) -> None:
        self.edges.append(edge)
        self.adjacency[edge.source_id].append(
            (edge.target_id, edge.lead_time_weeks, edge.units_per_target)
        )
        self.reverse_adjacency[edge.target_id].append(
            (edge.source_id, edge.lead_time_weeks, edge.units_per_target)
        )

    def upstream_dependencies(self, product_id: str) -> Set[str]:
        """Traverse upstream edges to find all dependent components and facilities."""
        visited: Set[str] = set()
        queue: List[str] = [product_id]
        while queue:
            curr = queue.pop(0)
            for upstream, _, _ in self.reverse_adjacency.get(curr, []):
                if upstream not in visited:
                    visited.add(upstream)
                    queue.append(upstream)
        return visited

    def compute_time_to_survive(self, disrupted_node_id: str, end_product_id: str) -> float:
        """
        Compute Time-to-Survive (TTS) in weeks along supply paths from disrupted node
        to finished good, summing intermediate pipeline buffer stocks.
        """
        def find_paths(curr: str, target: str) -> List[List[str]]:
            if curr == target:
                return [[curr]]
            paths = []
            for nxt, _, _ in self.adjacency.get(curr, []):
                for sub_path in find_paths(nxt, target):
                    paths.append([curr] + sub_path)
            return paths

        paths = find_paths(disrupted_node_id, end_product_id)
        if not paths:
            return float("inf")  # Disrupted node does not feed this product

        # Across all parallel paths, line survival is bounded by the weakest pipeline
        min_path_tts = float("inf")
        for path in paths:
            # Sum buffer stock held at all downstream nodes along this path
            path_tts = sum(self.nodes[nid].inventory_weeks for nid in path[1:])
            if path_tts < min_path_tts:
                min_path_tts = path_tts

        return min_path_tts


def evaluate_rebate(
    volume: int,
    base_price: float,
    tiers: List[RebateTier],
    rebate_type: str = "retroactive"
) -> Dict[str, float]:
    """Calculate gross spend, rebate earned, net spend, and effective unit price."""
    gross_spend = volume * base_price
    if volume <= 0:
        return {
            "volume": 0, "gross_spend": 0.0, "rebate_earned": 0.0,
            "net_spend": 0.0, "net_unit_price": 0.0, "effective_rebate_pct": 0.0
        }

    if rebate_type == "retroactive":
        matched_pct = 0.0
        for tier in tiers:
            if tier.min_volume <= volume and (tier.max_volume is None or volume <= tier.max_volume):
                matched_pct = tier.rebate_pct
                break
        rebate_earned = gross_spend * matched_pct
    elif rebate_type == "incremental":
        rebate_earned = 0.0
        for tier in tiers:
            if volume >= tier.min_volume:
                upper = volume if tier.max_volume is None else min(volume, tier.max_volume)
                tier_units = upper - tier.min_volume + 1
                rebate_earned += tier_units * base_price * tier.rebate_pct
        matched_pct = rebate_earned / gross_spend if gross_spend > 0 else 0.0
    else:
        raise ValueError(f"Unsupported rebate_type: {rebate_type}")

    net_spend = gross_spend - rebate_earned
    return {
        "volume": volume,
        "gross_spend": gross_spend,
        "rebate_earned": rebate_earned,
        "net_spend": net_spend,
        "net_unit_price": net_spend / volume,
        "effective_rebate_pct": matched_pct
    }


def evaluate_take_or_pay(
    actual_volume: int,
    min_commitment: int,
    penalty_rate_per_unit: float
) -> Dict[str, float]:
    """Evaluate take-or-pay shortfall units and liquidated damages."""
    shortfall = max(0, min_commitment - actual_volume)
    penalty_owed = shortfall * penalty_rate_per_unit
    return {
        "actual_volume": actual_volume,
        "min_commitment": min_commitment,
        "shortfall_units": shortfall,
        "penalty_rate": penalty_rate_per_unit,
        "penalty_owed": penalty_owed
    }


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate 95% Wilson score confidence interval for binomial proportions."""
    if trials == 0:
        return (0.0, 0.0)
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 2.57583
    p = successes / trials
    denom = 1.0 + (z ** 2) / trials
    center = (p + (z ** 2) / (2.0 * trials)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) / trials) + ((z ** 2) / (4.0 * (trials ** 2))))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


# ---------------------------------------------------------
# Demonstration and Verification Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    net = MultiTierNetwork()

    # 1. Define Network Nodes
    net.add_node(SupplyNode("T3_GLOBALWAFER", "GlobalWafer Fab F-3", 3, "Foundry", "Hsinchu, Taiwan", 0.0, 16.0))
    net.add_node(SupplyNode("T3_TITAN", "TitanAlloy Plant", 3, "Raw Material", "Duisburg, Germany", 2.0, 6.0))
    net.add_node(SupplyNode("T2_SILICO", "SilicoCore Fab", 2, "Semiconductor", "Dresden, Germany", 3.0, 8.0))
    net.add_node(SupplyNode("T2_MICROPREC", "MicroPrecision Plant", 2, "Machining", "Valbrembo, Italy", 2.0, 4.0))
    net.add_node(SupplyNode("T1_VELODRIVE", "VeloDrive Plant", 1, "Direct Supplier", "Wroclaw, Poland", 4.0, 4.0, 420.0))
    net.add_node(SupplyNode("T1_CARDIOSENSE", "CardioSense Plant", 1, "Direct Supplier", "Galway, Ireland", 4.0, 4.0, 280.0))
    net.add_node(SupplyNode("OEM_AP900", "Infusion Workstation AP-900", 0, "Finished Good", "Eindhoven, Netherlands", 3.0, 0.0))
    net.add_node(SupplyNode("OEM_MS200", "Monitor Station MS-200", 0, "Finished Good", "Eindhoven, Netherlands", 3.0, 0.0))

    # 2. Define Network Edges (BOM & Flow)
    net.add_edge(SupplyEdge("T3_GLOBALWAFER", "T2_SILICO", lead_time_weeks=8.0, units_per_target=1.0))
    net.add_edge(SupplyEdge("T3_TITAN", "T2_MICROPREC", lead_time_weeks=4.0, units_per_target=1.0))
    net.add_edge(SupplyEdge("T2_SILICO", "T1_VELODRIVE", lead_time_weeks=2.0, units_per_target=1.0))
    net.add_edge(SupplyEdge("T2_SILICO", "T1_CARDIOSENSE", lead_time_weeks=2.0, units_per_target=1.0))  # Shared IC!
    net.add_edge(SupplyEdge("T2_MICROPREC", "T1_VELODRIVE", lead_time_weeks=1.0, units_per_target=1.0))
    net.add_edge(SupplyEdge("T1_VELODRIVE", "OEM_AP900", lead_time_weeks=1.0, units_per_target=1.0))
    net.add_edge(SupplyEdge("T1_CARDIOSENSE", "OEM_MS200", lead_time_weeks=1.0, units_per_target=1.0))

    # 3. Analyze Hidden Dependencies
    dep_ap900 = net.upstream_dependencies("OEM_AP900")
    dep_ms200 = net.upstream_dependencies("OEM_MS200")
    shared_deps = dep_ap900.intersection(dep_ms200)

    print("=== MULTI-TIER DEPENDENCY AUDIT ===")
    print(f"AP-900 Upstream Dependencies: {sorted(list(dep_ap900))}")
    print(f"MS-200 Upstream Dependencies: {sorted(list(dep_ms200))}")
    print(f"Shared Sub-Tier Dependencies: {sorted(list(shared_deps))}")

    # 4. Simulate Upstream Disruption at GlobalWafer
    tts_ap900 = net.compute_time_to_survive("T3_GLOBALWAFER", "OEM_AP900")
    tts_ms200 = net.compute_time_to_survive("T3_GLOBALWAFER", "OEM_MS200")
    ttr_globalwafer = net.nodes["T3_GLOBALWAFER"].time_to_recover_weeks

    gap_ap900 = max(0.0, ttr_globalwafer - tts_ap900)
    gap_ms200 = max(0.0, ttr_globalwafer - tts_ms200)

    weekly_ap900 = 24000 / 50.0  # 480 units/week
    weekly_ms200 = 15000 / 50.0  # 300 units/week

    lost_ap900 = gap_ap900 * weekly_ap900
    delivered_ap900 = 24000 - lost_ap900

    lost_ms200 = gap_ms200 * weekly_ms200
    delivered_ms200 = 15000 - lost_ms200

    rar_ap900 = lost_ap900 * 3200.0  # $3,200 retail
    rar_ms200 = lost_ms200 * 1800.0  # $1,800 retail
    total_rar = rar_ap900 + rar_ms200

    print("\n=== DISRUPTION & REVENUE-AT-RISK SIMULATION ===")
    print(f"GlobalWafer TTR: {ttr_globalwafer:.1f} wks | AP-900 TTS: {tts_ap900:.1f} wks | Downtime: {gap_ap900:.1f} wks")
    print(f"AP-900 Units Lost: {lost_ap900:.0f} | Units Delivered: {delivered_ap900:.0f} | Revenue Lost: ${rar_ap900:,.2f}")
    print(f"MS-200 Units Lost: {lost_ms200:.0f} | Units Delivered: {delivered_ms200:.0f} | Revenue Lost: ${rar_ms200:,.2f}")
    print(f"Total Enterprise Revenue-at-Risk: ${total_rar:,.2f}")

    # 5. Contract Volume Rebate & Take-or-Pay Analysis
    rebate_tiers = [
        RebateTier(0, 17999, 0.000),
        RebateTier(18000, 22999, 0.035),
        RebateTier(23000, 27999, 0.060),
        RebateTier(28000, None, 0.085),
    ]

    res_planned = evaluate_rebate(24000, 420.0, rebate_tiers, "retroactive")
    res_disrupted = evaluate_rebate(int(delivered_ap900), 420.0, rebate_tiers, "retroactive")
    top_eval = evaluate_take_or_pay(int(delivered_ap900), 20000, 160.0)

    rebate_lost = (delivered_ap900 * 420.0 * 0.060) - res_disrupted["rebate_earned"]

    print("\n=== CONTRACT REBATE & SHORTFALL EVALUATION ===")
    print(f"Planned Net Spend (24,000 units @ 6.0% rebate): ${res_planned['net_spend']:,.2f} (${res_planned['net_unit_price']:.2f}/u)")
    print(f"Disrupted Net Spend ({delivered_ap900:.0f} units @ 3.5% rebate): ${res_disrupted['net_spend']:,.2f} (${res_disrupted['net_unit_price']:.2f}/u)")
    print(f"Rebate Margin Lost to Cliff on Delivered Units: ${rebate_lost:,.2f}")
    print(f"Take-or-Pay Shortfall Penalty (Min 20,000 units): ${top_eval['penalty_owed']:,.2f}")

    # 6. Contract Extraction Provenance Verification Sampling
    samples = {
        "Base Price & Currency": (98, 100),
        "Volume Rebate Schedule": (91, 100),
        "Take-or-Pay Penalties": (93, 100),
        "Force Majeure Carveouts": (84, 100),
    }

    print("\n=== CONTRACT EXTRACTION VERIFICATION (95% WILSON CI) ===")
    for clause, (k, n) in samples.items():
        low_ci, high_ci = wilson_score_interval(k, n)
        status = "PASSED" if low_ci >= 0.85 else "HUMAN REVIEW REQUIRED"
        print(f"{clause:25s}: Accuracy={k}/{n} ({k/n:.1%}) -> 95% CI [{low_ci:.3f}, {high_ci:.3f}] | {status}")
```

### Stated assumptions and expected output

**Assumptions:**
1. Upstream demand propagation is strictly deterministic; component consumption follows nominal BOM ratios without yield or scrap losses.
2. Intermediate pipeline inventory buffers are fully dedicated to Apex production and are not reallocated to other buyers during an emergency.
3. Logistics transport times between nodes are fixed; freight capacity bottlenecks during crisis rerouting are omitted.

**Expected script output:**
```text
=== MULTI-TIER DEPENDENCY AUDIT ===
AP-900 Upstream Dependencies: ['T1_VELODRIVE', 'T2_MICROPREC', 'T2_SILICO', 'T3_GLOBALWAFER', 'T3_TITAN']
MS-200 Upstream Dependencies: ['T1_CARDIOSENSE', 'T2_SILICO', 'T3_GLOBALWAFER']
Shared Sub-Tier Dependencies: ['T2_SILICO', 'T3_GLOBALWAFER']

=== DISRUPTION & REVENUE-AT-RISK SIMULATION ===
GlobalWafer TTR: 16.0 wks | AP-900 TTS: 10.0 wks | Downtime: 6.0 wks
AP-900 Units Lost: 2880 | Units Delivered: 21120 | Revenue Lost: $9,216,000.00
MS-200 Units Lost: 1800 | Units Delivered: 13200 | Revenue Lost: $3,240,000.00
Total Enterprise Revenue-at-Risk: $12,456,000.00

=== CONTRACT REBATE & SHORTFALL EVALUATION ===
Planned Net Spend (24,000 units @ 6.0% rebate): $9,475,200.00 ($394.80/u)
Disrupted Net Spend (21120 units @ 3.5% rebate): $8,559,936.00 ($405.30/u)
Rebate Margin Lost to Cliff on Delivered Units: $221,760.00
Take-or-Pay Shortfall Penalty (Min 20,000 units): $0.00

=== CONTRACT EXTRACTION VERIFICATION (95% WILSON CI) ===
Base Price & Currency    : Accuracy=98/100 (98.0%) -> 95% CI [0.930, 0.994] | PASSED
Volume Rebate Schedule   : Accuracy=91/100 (91.0%) -> 95% CI [0.838, 0.952] | HUMAN REVIEW REQUIRED
Take-or-Pay Penalties    : Accuracy=93/100 (93.0%) -> 95% CI [0.863, 0.966] | PASSED
Force Majeure Carveouts  : Accuracy=84/100 (84.0%) -> 95% CI [0.756, 0.899] | HUMAN REVIEW REQUIRED
```

## 8. Data, standards, and regulatory context

Multi-tier mapping and contract data operate at the intersection of international trade law, corporate governance mandates, data exchange protocols, and accounting standards.

```text
┌───────────────────────────────┬───────────────────────────────┬────────────────────────────────┐
│ Regulatory / Standard Domain │ Legal Authority / Body        │ Core Mandatory Requirement     │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Corporate Due Diligence       │ EU CSDDD (Directive 2024/1760;│ Upstream value chain human     │
│                               │ Omnibus I March 2026)         │ rights & environmental duty    │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ National Supply Chain Acts    │ German LkSG (2023 / 2024)     │ Risk analysis for Tier 1;      │
│                               │                               │ substantiated knowledge Tier N │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Forced Labor Trade Bans       │ US UFLPA (19 U.S.C. § 1307)   │ Rebuttable presumption; batch  │
│                               │                               │ chain-of-custody documentation │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Conflict Minerals             │ US Dodd-Frank Act § 1502;     │ Smelter/refiner tracing for    │
│                               │ EU Conflict Minerals 2017/821 │ 3TG (tin, tantalum, tungsten,  │
│                               │                               │ gold)                          │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Carbon Border Adjustment      │ EU CBAM (Regulation 2023/956; │ Primary embedded emissions     │
│                               │ Definitive regime Jan 2026)   │ reporting from foreign mills   │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Upstream Greenhouse Gas       │ GHG Protocol Scope 3 Cat 1;   │ Transition from spend-based to │
│                               │ ISO 14067 / PACT Pathfinder   │ certified cradle-to-gate PCFs  │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Electronic Procurement Data   │ Peppol BIS 3.0; UN/EDIFACT;   │ Structured digital exchange of │
│                               │ OASIS Universal Business Lang │ orders, receipts, and invoices │
└───────────────────────────────┴───────────────────────────────┴────────────────────────────────┘
```

### Binding statutory regulations

1. **EU Corporate Sustainability Due Diligence Directive (CSDDD):**
   Adopted as Directive (EU) 2024/1760 and refined under Omnibus I in March 2026, the directive sets mandatory human rights and environmental due diligence standards.
   While Omnibus I extended the ultimate national transposition deadline to July 26, 2028, and phased thresholds for high-revenue enterprises (>€1.5B turnover, >5,000 employees), it established immediate compliance expectations for upstream mapping.
   Enterprises are legally obligated to:
   - Identify actual and potential adverse impacts across their own operations, subsidiaries, and upstream business partners in their "chain of activities" (design, extraction, manufacture, transport).
   - Adopt prevention action plans, obtain contractual assurances from direct partners with verified flow-down obligations, and conduct periodic compliance audits.
   - Establish and maintain a formal operational-level grievance mechanism accessible to affected stakeholders throughout the supply network.
   - Maintain board oversight; failure to implement due diligence exposes the company to civil liability for damages caused by supply chain adverse impacts.
2. **German Supply Chain Due Diligence Act (*Lieferkettensorgfaltspflichtengesetz* - LkSG):**
   In effect since January 1, 2023 (expanded to companies with $\ge 1,000$ employees in Germany on January 1, 2024), the LkSG mandates:
   - Appointment of a dedicated Human Rights Officer reporting directly to the executive board.
   - An annual human rights and environmental risk assessment covering all direct (Tier-1) suppliers.
   - Immediate investigation and remedial action for indirect (Tier-2 through Tier-N) suppliers whenever the enterprise obtains "substantiated knowledge" (*substantiierte Kenntnis*) of a violation.
   - Submission of an annual publicly accessible compliance report to the Federal Office for Economic Affairs and Export Control (BAFA), with administrative fines reaching up to 2 percent of worldwide annual turnover for severe non-compliance.
3. **US Uyghur Forced Labor Prevention Act (UFLPA):**
   Enacted under 19 U.S.C. § 1307, the UFLPA establishes a statutory **rebuttable presumption** that any good, ware, article, or merchandise mined, produced, or manufactured wholly or in part in the Xinjiang region of China is made with forced labor.
   Key operational constraints:
   - United States Customs and Border Protection (CBP) detains shipments without advance notice based on algorithmic customs manifest screening and sub-tier trade graph mapping.
   - Importers cannot overcome detention using standard supplier affidavits or commercial certificates of origin.
   - Clear and convincing evidence requires a complete evidentiary trace: raw material extraction logs, purchase orders, transportation waybills, factory transformation logs, employee payroll records, and daily production sheets verifying that every gram of input material originated from certified non-prohibited facilities.

### Data exchange and transaction standards

To maintain structured contract and multi-tier provenance across heterogeneous enterprise systems:
- **GS1 EPCIS 2.0 (Electronic Product Code Information Services):** An open standard for tracking physical events (`What`, `When`, `Where`, `Why`) across supply networks. It enables serialization and batch-level tracking required for UFLPA and medical device traceability.
- **Peppol BIS Billing 3.0 and OASIS Universal Business Language (UBL):** XML-based schemas for cross-border electronic invoicing and purchase order transmission, eliminating paper OCR errors and standardizing tax identifiers, line-item codes, and contract references.
- **WBCSD PACT Pathfinder Framework:** Standardizes the digital exchange of Scope 3 Product Carbon Footprints (PCFs) between enterprise ERP systems, enabling certified, primary emission data flow across supply tiers.

## 9. Industry practice and case evidence

### The 2020–2023 automotive semiconductor crisis

The global automotive semiconductor crisis provides the quintessential empirical case of Tier-1 visibility failure.
During the initial drop in vehicle sales in early 2020, major automotive original equipment manufacturers (OEMs)—including Volkswagen, General Motors, and Ford—slashed procurement forecasts and cancelled purchase orders for electronic control units (ECUs) with direct Tier-1 suppliers (such as Bosch, Continental, and Denso).

```text
[Automotive OEM] ──(cancels POs)──► [Tier-1 ECU Integrator]
                                            │
                                            ▼ (cancels sub-assembly POs)
                                    [Tier-2 Microcontroller Vendor]
                                            │
                                            ▼ (cancels wafer starts)
                                    [Tier-3 Semiconductor Foundry (TSMC)]
                                            │
                                            ▼ (reallocates capacity to consumer tech)
                                    [Consumer Electronics Boom (Laptops, Phones)]
```

Because automotive OEMs lacked multi-tier BOM explosion systems, they operated under three catastrophic misconceptions:
1. **Capacity Fungibility Assumption:** OEMs assumed that semiconductor foundry capacity was fungible and could be turned on within standard 4-week Tier-1 component lead times.
   In reality, automotive-grade microcontrollers depend on specialized legacy lithography lines (28nm to 90nm) with raw manufacturing cycle times of **16 to 26 weeks**, plus 8 weeks of packaging and testing.
2. **Foundry Reallocation:** When automotive Tier-1s cancelled wafer orders, foundries reallocated scarce lithography capacity to consumer electronics and cloud computing providers experiencing pandemic demand surges.
3. **Hidden Foundry Single Points of Failure:** When vehicle demand rebounded in late 2020, automakers attempted to restart ECU orders.
   They discovered that hundreds of independent Tier-1 and Tier-2 suppliers all routed their microcontroller production through a single foundry—Taiwan Semiconductor Manufacturing Company (TSMC)—which was fully booked.
   The resulting component shortages idled vehicle assembly plants worldwide for months, costing the global automotive industry an estimated **$210 billion USD in lost revenue in 2021 alone** (AlixPartners, 2021).

### Boeing 787 Dreamliner multi-tier outsourcing

During the development of the 787 Dreamliner in the mid-2000s, Boeing implemented an aggressive "Tier-1 Partner" outsourcing model.
Rather than designing detailed parts internally and procuring them directly, Boeing outsourced major fuselage sections, wings, and flight systems to Tier-1 strategic partners (such as Alenia, Mitsubishi, and Vought), expecting them to manage their own sub-tier supply networks.

The program encountered severe delays (exceeding three years) and billions of dollars in budget overruns due to total visibility failure at Tier 2 and Tier 3:
- Tier-1 partners lacked the systems and authority to audit sub-tier engineering tolerances.
- In 2007, final assembly in Everett, Washington, halted completely because Tier-1 aerostructure sections arrived missing thousands of specialized titanium fasteners.
  The Tier-2 fastener manufacturers faced severe raw material shortages and machinery bottlenecks that had never been reported to Boeing.
- Boeing was forced to deploy hundreds of internal engineers globally to physically map, rescue, and in several cases acquire troubled Tier-1 and Tier-2 supplier plants (such as buying Vought Aircraft Industries' South Carolina facility in 2009 for $1 billion USD) to regain operational control.

### High-tech mineral tracing and the Responsible Minerals Initiative

In response to Section 1502 of the US Dodd-Frank Act, multinational electronics manufacturers (including Apple, HP, and Intel) recognized that auditing Tier-1 circuit board assemblers provided zero evidence regarding conflict minerals (tin, tantalum, tungsten, and gold—3TG) originating in the Democratic Republic of the Congo.

Rather than attempting to map millions of artisanal mines, industry leaders established the **Responsible Minerals Initiative (RMI)**.
The key analytical breakthrough was identifying the **smelter and refiner choke point**:
Regardless of which mine 3TG ore is extracted from, it must pass through one of approximately **350 active commercial smelters and refiners globally** to reach industrial purity.
By mapping multi-tier BOMs to the smelter level and requiring Tier-1 suppliers to collect standardized Conflict Minerals Reporting Templates (CMRT) identifying specific facilities, OEMs focused independent, third-party facility audits at this precise Tier-3/4 bottleneck.
This industry architecture demonstrated that multi-tier visibility does not require tracking every upstream entity uniformly; it requires pinpointing the physical and commercial choke points through which all upstream materials must flow.

## 10. Failure modes and red flags

| Failure | Early warning sign | Root mechanism | Operational / financial consequence | Prescribed mitigation |
|---|---|---|---|---|
| Entity vs site conflation | Single vendor record in ERP with one corporate headquarters address | Treating commercial counterparty as identical to manufacturing site | Catastrophic line shutdown when unmapped offshore plant halts | Split vendor master into legal entities ($V_{\text{entity}}$) and physical plants ($V_{\text{facility}}$) |
| Common-mode failure blind spot | "We dual-source 50/50" claimed without sub-tier BOM data | Multiple Tier-1 integrators procuring from the same Tier-2/3 plant | Perceived redundancy collapses to 100% single-source exposure | Compute graph intersection of upstream dependencies across all Tier-1 quotes |
| Unmonitored retroactive rebate cliff | End-of-year purchasing freeze proposed by finance to conserve cash | Retroactive rebate schedule with large step-discontinuities | Purchasing one fewer unit drops volume across threshold, losing millions | Build dynamic marginal cost calculator showing negative marginal cost near cliffs |
| Silent LLM contract extraction hallucination | Contract database populated by automated parser without manual audit | LLM generates plausible but fictitious penalty terms or dates | Commercial dispute or missed termination notice deadline | Mandate stratified verification sampling ($n \ge 100$) and Wilson score confidence limits |
| Missing amendment linkage | Extracted contract parameters contradict current purchase order prices | Amendments and exhibits stored as disconnected PDFs in email drives | Operating under superseded terms; failure to capture renegotiated terms | Implement document stitching linking root MSAs to chronologically ordered amendments |
| Take-or-pay penalty blind spot | Demand forecast drops sharply, but buyers slash POs without checking MSA | Minimum purchase commitments treated as non-binding forecasts | Supplier issues massive year-end liquidated damages invoice | Integrate contract commitments into sales and operations planning (S&OP) models |
| Invalid force majeure defense | Tier-1 supplier claims force majeure due to sub-tier component shortage | Standard contract terms exclude subcontractor failures from force majeure | Buyer bears unrecovered production losses without recourse | Audit Section 11 / Force Majeure terms; enforce liquidated damages for unexcused delays |
| UFLPA customs cargo seizure | Importer cannot trace origin of aluminum, cotton, or silicon components | Relying on Tier-1 certificates of origin without batch tracing | Shipments detained at port; perishable goods spoil; line shutdowns | Build automated chain-of-custody dossier tracing from raw material lots to import bills |
| Spend-based Scope 3 delusion | Corporate sustainability report claims 15% upstream carbon reduction | Spend-based EEIO factors used; inflation or price cuts mistaken for carbon cuts | Misleading ESG disclosure; greenwashing enforcement by regulators | Contractually require Tier-1 primary product carbon footprints (PCFs) via ISO 14067 |
| Unclaimed contract leakage | Finance never reconciles annual earned rebates against cash receipts | Invoices paid at gross price without automated rebate accrual | Substantial earned rebate income left uncollected on supplier balance sheets | Implement automated four-way reconciliation across PO, receipt, invoice, and contract |
| Flow-down clause failure | Tier-1 signs Supplier Code of Conduct, but sub-tier abuses occur | Tier-1 agreement lacks binding obligation to flow down terms to subs | Enterprise exposed to CSDDD/LkSG regulatory sanctions and brand damage | Embed mandatory sub-tier flow-down and audit rights with contract termination penalties |
| Static BOM desynchronization | Sourcing team uses outdated engineering BOM during contract negotiation | Engineering change orders (ECOs) alter component specs without procurement notice | Buyer contracts for obsolete parts; uncommitted new components face shortages | Establish bi-directional digital thread synchronizing PLM engineering BOMs with ERP |

## 11. Assumptions and limits

| Assumption | Analytical role | Vulnerability / failure condition | Diagnostic test |
|---|---|---|---|
| Complete sub-tier visibility | Assumes all Tier-1 suppliers disclose full sub-tier supplier lists | Tier-1 suppliers conceal sources citing proprietary trade secrets | Measure mapping coverage (% of total spend with verified Tier-2 sites) |
| Deterministic lead times | Treats manufacturing and transit times as fixed point estimates | Port congestion, customs inspections, and weather cause stochastic delays | Run Monte Carlo simulations applying Weibull transit lead-time distributions |
| Linear BOM scaling | Assumes component consumption scales linearly with finished goods | Scrap rates, setup losses, and batch minimums create non-linear demand | Audit shop-floor manufacturing scrap rates across production batch sizes |
| Dedicated buffer inventory | Assumes intermediate inventory is reserved exclusively for the buyer | Tier-1 or Tier-2 sells buffer stock to competing buyers paying spot premiums | Execute unannounced physical supplier inventory audits and require escrowed stock |
| Static supplier capacity | Treats supplier capacity as constant over the planning horizon | Machine maintenance, labor attrition, or line retooling cuts capacity | Perform monthly capacity utilization audits across contracted product lines |
| Enforceability of audit rights | Assumes contractual audit rights allow physical inspection globally | Foreign jurisdictions restrict foreign audits citing national security laws | Verify local legal enforceability with in-country legal counsel |
| Uniform LLM extraction accuracy | Assumes extraction error rates are identical across all contract types | Complex, highly negotiated bespoke legal clauses exhibit higher error rates | Stratify sampling audits across standard templates versus heavily amended MSAs |
| Constant carbon emission factors | Assumes factory carbon intensity remains static throughout the year | Grid electricity mix varies seasonally (hydro vs coal during dry seasons) | Demand quarterly primary energy consumption records from supplier sites |

## 12. Exercises

### Exercise 1: Multi-tier graph formulation and common-mode identification

An industrial electronics OEM manufactures two telematics units: `Model TU-10` and `Model TU-20`.
`Model TU-10` uses a microcontroller assembly procured from Tier-1 Supplier Alpha (Dublin, Ireland).
Supplier Alpha purchases the raw microcontrollers from Tier-2 Fabricator Gamma (Austin, Texas), which procures specialized ceramic substrates from Tier-3 Substrate Corp (Kyoto, Japan).
`Model TU-20` uses a communication module procured from Tier-1 Supplier Beta (Munich, Germany).
Supplier Beta buys its transceivers from Tier-2 Fabricator Delta (Grenoble, France).
Unknown to procurement, Tier-2 Fabricator Delta also purchases its ceramic substrates from Tier-3 Substrate Corp (Kyoto, Japan).

- **Deliverable:**
  1. Construct the complete node and edge lists representing this multi-tier network.
  2. Identify the common-mode sub-tier failure point.
  3. If Tier-3 Substrate Corp suffers an 8-week production stoppage, and total pipeline buffer stocks (summed across all tiers) are 5 weeks for `Model TU-10` and 3 weeks for `Model TU-20`, calculate the downtime window experienced at OEM final assembly for each model.
- **Acceptance criteria:** Node types correctly partitioned into entities, facilities, and parts; downtime windows mathematically verified.
- **Estimated time:** 30 minutes.

### Exercise 2: Quantitative rebate cliff and negative marginal cost calculation

An automotive manufacturer procures specialized brake calipers under a contract with the following terms:
- Base price: **$150.00 USD** per caliper.
- Retroactive annual volume rebate schedule (applies to all units from unit 1):
  - Tier 1: 0 to 49,999 units $\to$ 0.0% rebate
  - Tier 2: 50,000 to 74,999 units $\to$ 4.0% rebate
  - Tier 3: $\ge 75,000$ units $\to$ 7.5% rebate

On December 1, cumulative annual purchases stand at **74,200 units**.
The production manager plans to order an additional **600 units** for December production, bringing total annual volume to **74,800 units**.
The procurement analyst proposes purchasing **75,000 units** instead (storing the extra 200 units in inventory).

- **Deliverable:**
  1. Compute the gross spend, earned rebate, and net spend for purchasing 74,800 units.
  2. Compute the gross spend, earned rebate, and net spend for purchasing 75,000 units.
  3. Calculate the marginal net cash difference of purchasing the additional 200 units. What is the effective marginal cost per unit of those 200 units?
  4. State the working capital and inventory holding cost trade-offs associated with this purchasing decision.
- **Acceptance criteria:** Exact net cash flows calculated; marginal cost per unit explicitly identified as negative.
- **Estimated time:** 30 minutes.

### Exercise 3: Incremental vs retroactive rebate curve comparison

Using the brake caliper contract parameters from Exercise 2 ($P_{\text{base}} = \$150.00$; Tier 1: 0–49,999 @ 0%; Tier 2: 50,000–74,999 @ 4.0%; Tier 3: $\ge 75,000$ @ 7.5%):
- **Deliverable:**
  1. Formulate the mathematical equation for total net spend as a function of volume $Q$ under an **incremental (marginal) rebate structure** across all three tiers.
  2. Compute the net spend and effective average unit price at $Q = 75,000$ units under both the retroactive and incremental structures.
  3. Calculate the dollar difference in supplier revenue between the two contract structures at $Q = 75,000$ units.
- **Acceptance criteria:** Clear piecewise mathematical equations; correct arithmetic comparison.
- **Estimated time:** 25 minutes.

### Exercise 4: Disruption simulation, Time-to-Survive, and Revenue-at-Risk

A medical diagnostic company produces a blood analyzer workstation selling for **$12,500.00 USD** retail.
Annual production is **5,000 units/year** over a 50-week operating year (**100 units/week**).
The instrument requires an optical sensor module procured from a Tier-1 supplier in Singapore.
The optical sensor depends on a specialized coated lens produced exclusively by a Tier-2 optics facility in Sendai, Japan.
The Sendai facility suffers a fire resulting in a Time-to-Recover ($\text{TTR}$) of **22.0 weeks**.
Inventory audits reveal:
- OEM finished goods and WIP buffer: 4.0 weeks
- Tier-1 Singapore assembly and transit buffer: 6.0 weeks
- Tier-2 Sendai finished lens buffer: 2.0 weeks

- **Deliverable:**
  1. Calculate the system-level Time-to-Survive ($\text{TTS}$).
  2. Calculate the disruption downtime window at OEM final assembly.
  3. Calculate the total units of blood analyzers lost.
  4. Calculate the total enterprise Revenue-at-Risk ($\text{RaR}$).
- **Acceptance criteria:** Correct summation of buffer stocks; accurate multiplication of lost run rate and retail price.
- **Estimated time:** 25 minutes.

### Exercise 5: Statistical verification sampling with Wilson score confidence intervals

A procurement organization implements an LLM-based contract extraction pipeline to process 1,200 legacy supplier agreements.
To certify extraction accuracy for "Audit Rights and ESG Compliance Clauses", the analytics team draws a random sample of **120 contracts**.
Human legal auditors review the sample and determine that the LLM correctly extracted the audit parameters in **106 contracts** and made errors in **14 contracts**.

- **Deliverable:**
  1. Calculate the sample accuracy proportion $\hat{p}$.
  2. Compute the 95% Wilson score confidence interval $[p_{\text{lower}}, p_{\text{upper}}]$ for the extraction pipeline.
  3. If enterprise governance policy mandates that automated database loading is permitted only if the lower bound of the 95% confidence interval is $\ge 85.0\%$, state whether this pipeline qualifies for automated production or requires mandatory human-in-the-loop review.
- **Acceptance criteria:** Correct application of the Wilson score formula without Gaussian approximation; clear pass/fail recommendation.
- **Estimated time:** 30 minutes.

### Exercise 6: Critique an unstructured contract data extraction architecture

An enterprise IT group presents a proposal to automate contract risk analytics.
Their architecture consists of ingesting supplier PDF contracts, passing entire 80-page agreements into a commercial LLM prompt, asking the LLM to summarize "all key risks, discounts, and penalties", and writing the generated free-text summary directly into an executive dashboard.

- **Deliverable:**
  Write a formal technical critique (400–600 words) identifying at least four critical flaws in this architecture. Address:
  1. Lack of structured schema validation.
  2. Omission of clause-level provenance and auditability.
  3. Absence of document stitching across amendments and side letters.
  4. Vulnerability to hallucinated commercial terms and lack of statistical verification.
- **Acceptance criteria:** Professional engineering critique addressing risk, governance, and architectural remediation.
- **Estimated time:** 35 minutes.

### Exercise 7: Design an end-to-end UFLPA / CSDDD chain-of-custody data pipeline

A solar module manufacturer sources aluminum frames, glass panels, and photovoltaic cells.
To clear US customs under UFLPA and satisfy EU CSDDD due diligence audits:
- **Deliverable:**
  Design a comprehensive data pipeline architecture.
  1. Define the relational or graph schema linking purchase orders, manufacturing lot numbers, shipping containers, and customs declarations from Tier-4 quartzite mines to finished solar panels.
  2. Specify the exact evidentiary documents required at each transformation tier.
  3. Define an automated screening rule that flags shipments lacking verified non-prohibited origin documentation before export containers are loaded onto ocean vessels.
- **Acceptance criteria:** Clear schema design, complete document inventory across all tiers, and concrete operational control rules.
- **Estimated time:** 45 minutes.

### Exercise 8: Automated rebate leakage detection and GL reconciliation system

An enterprise with $2.5 billion in annual spend suspects substantial contract rebate leakage across its top 50 suppliers.
- **Deliverable:**
  Design the technical and operational specification for an automated Rebate Leakage Detection Engine:
  1. Specify the four data sources (ERP POs, GRNs, AP Invoices, Extracted Contracts) and their required join keys.
  2. Write the algorithmic logic (pseudocode or relational algebra) to calculate earned rebates, compare them to GL accounts receivable / credit notes, and flag variances exceeding $10,000.
  3. Define the governance workflow and audit recovery letter protocol for claiming uncollected rebates.
- **Acceptance criteria:** End-to-end algorithmic specification with robust exception-handling logic.
- **Estimated time:** 40 minutes.

## 13. Self-check answers

### Answer 1: Multi-tier graph formulation and common-mode identification

1. **Graph Formulation:**
   - **Node set $V$:**
     - Facilities / Entities:
       - OEM Assembly Plant (Eindhoven) [Tier 0]
       - Supplier Alpha (Dublin) [Tier 1]
       - Supplier Beta (Munich) [Tier 1]
       - Fabricator Gamma (Austin) [Tier 2]
       - Fabricator Delta (Grenoble) [Tier 2]
       - Substrate Corp (Kyoto) [Tier 3]
     - Parts:
       - Telematics Unit TU-10, Telematics Unit TU-20 [Finished Goods]
       - Microcontroller Assembly (Alpha), Communication Module (Beta) [Sub-Assemblies]
       - Raw Microcontroller (Gamma), Transceiver (Delta) [Components]
       - Ceramic Substrate (Substrate Corp) [Raw Material / Sub-Tier Component]
   - **Edge set $E$:**
     - Path 1 (`TU-10`): Substrate Corp (Kyoto) $\to$ Fabricator Gamma (Austin) $\to$ Supplier Alpha (Dublin) $\to$ OEM (Eindhoven) $\to$ `TU-10`
     - Path 2 (`TU-20`): Substrate Corp (Kyoto) $\to$ Fabricator Delta (Grenoble) $\to$ Supplier Beta (Munich) $\to$ OEM (Eindhoven) $\to$ `TU-20`
2. **Common-Mode Failure Point:**
   Substrate Corp in Kyoto, Japan. Although the OEM multi-sources finished telematics units across two distinct Tier-1 suppliers (Alpha in Ireland and Beta in Germany), both supply chains converge completely at the Tier-3 ceramic substrate manufacturer in Kyoto.
3. **Downtime Calculation:**
   - Disruption duration at Substrate Corp: $\text{TTR} = 8.0 \text{ weeks}$.
   - For `TU-10`: $\text{TTS} = 5.0 \text{ weeks}$.
     $$\Delta t_{\text{disrupt}}(\text{TU-10}) = \max(0, 8.0 - 5.0) = 3.0 \text{ weeks}$$
   - For `TU-20`: $\text{TTS} = 3.0 \text{ weeks}$.
     $$\Delta t_{\text{disrupt}}(\text{TU-20}) = \max(0, 8.0 - 3.0) = 5.0 \text{ weeks}$$
   `TU-10` assembly halts for 3.0 weeks; `TU-20` assembly halts for 5.0 weeks.

### Answer 2: Quantitative rebate cliff and negative marginal cost calculation

1. **Option A: Purchase 74,800 units:**
   - Falls into Tier 2 (50,000 to 74,999 units) $\to$ 4.0% retroactive rebate.
   - Gross spend: $74,800 \times \$150.00 = \$11,220,000.00 \text{ USD}$.
   - Earned rebate: $\$11,220,000.00 \times 0.040 = \$448,800.00 \text{ USD}$.
   - Net spend: $\$11,220,000.00 - \$448,800.00 = \$10,771,200.00 \text{ USD}$.
   - Effective unit cost: $\$10,771,200.00 / 74,800 = \$144.00 \text{ USD/unit}$.
2. **Option B: Purchase 75,000 units:**
   - Crosses into Tier 3 ($\ge 75,000$ units) $\to$ 7.5% retroactive rebate.
   - Gross spend: $75,000 \times \$150.00 = \$11,250,000.00 \text{ USD}$.
   - Earned rebate: $\$11,250,000.00 \times 0.075 = \$843,750.00 \text{ USD}$.
   - Net spend: $\$11,250,000.00 - \$843,750.00 = \$10,406,250.00 \text{ USD}$.
   - Effective unit cost: $\$10,406,250.00 / 75,000 = \$138.75 \text{ USD/unit}$.
3. **Marginal Net Cash Difference:**
   $$\Delta \text{Net Cash} = \$10,406,250.00 - \$10,771,200.00 = -\$364,950.00 \text{ USD}$$
   By purchasing **200 more units**, the company’s total cash outflow decreases by **$364,950.00 USD**.
   - Marginal cost per unit of the additional 200 units:
     $$\text{MC} = \frac{-\$364,950.00}{200} = -\$1,824.75 \text{ USD/unit}$$
   The supplier effectively pays the buyer $1,824.75 for each of those 200 units.
4. **Trade-offs:**
   Purchasing the 200 units is a clear commercial imperative. The immediate cash savings ($364,950.00) vastly exceed the annual carrying cost of 200 brake calipers (at 20% annual holding cost, 200 units $\times \$138.75 \times 0.20 = \$5,550.00$/year). The extra 200 units can be consumed in the first week of the subsequent operating year.

### Answer 3: Incremental vs retroactive rebate curve comparison

1. **Incremental Net Spend Equation:**
   Let $P = 150.00$.
   - For $0 \le Q < 50,000$:
     $$C_{\text{inc}}(Q) = 150.00 Q$$
   - For $50,000 \le Q < 75,000$:
     $$C_{\text{inc}}(Q) = (150.00 \times 50,000) + 150.00(1 - 0.040)(Q - 50,000) = 7,500,000 + 144.00(Q - 50,000)$$
   - For $Q \ge 75,000$:
     $$\begin{aligned}
     C_{\text{inc}}(Q) &= 7,500,000 + 144.00(75,000 - 50,000) + 150.00(1 - 0.075)(Q - 75,000) \\
     &= 7,500,000 + 3,600,000 + 138.75(Q - 75,000) = 11,100,000 + 138.75(Q - 75,000)
     \end{aligned}$$
2. **Net Spend at $Q = 75,000$ Units:**
   - Under incremental structure:
     $$C_{\text{inc}}(75,000) = \$11,100,000.00 \text{ USD}$$
     $$\text{Average Price}_{\text{inc}} = \frac{\$11,100,000.00}{75,000} = \$148.00 \text{ USD/unit}$$
   - Under retroactive structure (from Exercise 2):
     $$C_{\text{retro}}(75,000) = \$10,406,250.00 \text{ USD}$$
     $$\text{Average Price}_{\text{retro}} = \$138.75 \text{ USD/unit}$$
3. **Difference in Supplier Revenue:**
   $$\Delta \text{Revenue} = C_{\text{inc}}(75,000) - C_{\text{retro}}(75,000) = \$11,100,000.00 - \$10,406,250.00 = \$693,750.00 \text{ USD}$$
   The supplier earns $693,750.00 more under the incremental structure because the 7.5% discount applies only to units above 75,000 rather than retroactively from dollar one.

### Answer 4: Disruption simulation, Time-to-Survive, and Revenue-at-Risk

1. **System-Level Time-to-Survive ($\text{TTS}$):**
   Sum inventory buffers across the pipeline downstream from the Sendai facility:
   $$\text{TTS} = 4.0 \text{ (OEM)} + 6.0 \text{ (Tier 1 Singapore)} + 2.0 \text{ (Tier 2 Sendai)} = 12.0 \text{ weeks}$$
2. **Disruption Downtime Window:**
   $$\Delta t_{\text{disrupt}} = \max(0, \text{TTR} - \text{TTS}) = 22.0 - 12.0 = 10.0 \text{ weeks}$$
3. **Blood Analyzer Units Lost:**
   Weekly production rate = $5,000 / 50 = 100 \text{ units/week}$.
   $$\text{Units Lost} = 10.0 \text{ weeks} \times 100 \text{ units/week} = 1,000 \text{ units}$$
4. **Total Enterprise Revenue-at-Risk:**
   $$\text{RaR} = 1,000 \text{ units} \times \$12,500.00 \text{ USD/unit} = \$12,500,000.00 \text{ USD}$$

### Answer 5: Statistical verification sampling with Wilson score confidence intervals

1. **Sample Proportion:**
   $$\hat{p} = \frac{106}{120} \approx 0.88333 \quad (88.33\%)$$
2. **95% Wilson Score Confidence Interval:**
   With $n = 120$, $k = 106$, and $z = 1.95996$:
   $$z^2 = 3.84144$$
   $$\text{Denominator} = 1 + \frac{3.84144}{120} = 1 + 0.032012 = 1.032012$$
   $$\text{Center} = \frac{0.88333 + \frac{3.84144}{240}}{1.032012} = \frac{0.88333 + 0.016006}{1.032012} = \frac{0.899339}{1.032012} \approx 0.87144$$
   $$\text{Variance term} = \frac{0.88333(1 - 0.88333)}{120} + \frac{3.84144}{4(14,400)} = \frac{0.103055}{120} + \frac{3.84144}{57,600} = 0.0008588 + 0.0000667 = 0.0009255$$
   $$\sqrt{\text{Variance term}} = \sqrt{0.0009255} \approx 0.030422$$
   $$\text{Spread} = \frac{1.95996 \times 0.030422}{1.032012} = \frac{0.059626}{1.032012} \approx 0.05778$$
   $$\text{CI}_{0.95} = [0.87144 - 0.05778, 0.87144 + 0.05778] = [0.81366, 0.92922] \implies [81.4\%, 92.9\%]$$
3. **Governance Decision:**
   The lower bound of the 95% confidence interval is **81.4%**, which is **below the mandated 85.0% threshold**.
   Therefore, this extraction pipeline **fails certification** for automated database loading.
   Mandatory human-in-the-loop review must be maintained for all audit and ESG clauses until prompt engineering, fine-tuning, or schema pre-filtering raises the lower bound above 85.0%.

### Rubrics for Exercises 6–8

#### Exercise 6 Rubric (Architecture Critique)
- **Score 4 (Excellent):** Identifies all four flaws clearly; explains why unconstrained LLM summaries produce non-reproducible outputs; articulates how missing provenance creates legal evidentiary failure; details risks of unstitched amendments overriding root MSAs; specifies a target state with JSON schemas, chunk-level OCR, and Wilson verification sampling.
- **Score 3 (Adequate):** Identifies 3–4 flaws with sound reasoning; explains provenance and hallucinations; recommendations are somewhat generic.
- **Score 2 (Needs improvement):** Focuses solely on LLM inaccuracies without addressing document layout, amendment stitching, or provenance.
- **Score 1 (Inadequate):** Recommends minor prompt tweaks without structural architectural critique.

#### Exercise 7 Rubric (UFLPA / CSDDD Data Pipeline Design)
- **Score 4 (Excellent):** Provides a rigorous data schema linking part lots, facilities, shipping containers, and customs entries; lists complete documentation requirements across all 4 tiers (mine extraction logs, smelter certificates, bills of lading, worker payrolls); includes an automated pre-shipment gate blocking container loading upon missing provenance.
- **Score 3 (Adequate):** Good data schema and document list; minor gaps in sub-tier validation rules or customs manifest integration.
- **Score 2 (Needs improvement):** Relies heavily on Tier-1 supplier self-declarations; lacks granular batch-level traceability.
- **Score 1 (Inadequate):** Superficial design lacking data schemas or regulatory operational grounding.

#### Exercise 8 Rubric (Rebate Leakage & GL Reconciliation)
- **Score 4 (Excellent):** Accurately defines join keys across PO, GRN, AP Invoice, and Contract models; writes clear algorithmic reconciliation pseudocode handling tiered boundaries, receipt date alignment, and credit notes; outlines a defensible commercial recovery protocol.
- **Score 3 (Adequate):** Clear reconciliation logic and data sources; minor ambiguities in handling disputed credit notes or calendar cutoff dates.
- **Score 2 (Needs improvement):** Incomplete join logic; ignores credit note netting or GL accrual timing.
- **Score 1 (Inadequate):** Fails to connect transactional ERP data to contractual rebate mechanics.

## 14. Interview questions

| # | Question | Strong answer contains | Weak answer reveals |
|---|---|---|---|
| 1 | How do you detect hidden supplier concentration when Tier-1 vendors claim they are fully diversified? | Constructs a heterogeneous graph linking parts to physical manufacturing sites ($V_{\text{facility}}$); executes a multi-level BOM explosion; computes graph intersection across upstream paths to find shared sub-tier foundries, smelters, or geographic clusters. | Assumes supplier diversity at the legal entity level guarantees resilience; relies on Tier-1 executive surveys without sub-tier BOM tracing. |
| 2 | What is the difference between Time-to-Survive (TTS) and Time-to-Recover (TTR), and how are they used in sourcing decisions? | Defines TTR as facility recovery or requalification duration; defines TTS as pipeline inventory survival across intermediate nodes; shows that disruption occurs when $TTR > TTS$; uses Revenue-at-Risk to justify dual-sourcing premiums or buffer stock sizing. | Confuses supplier lead time with TTR; assumes finished goods inventory is the only buffer; treats disruption risk as a qualitative high/medium/low score. |
| 3 | Why can a retroactive volume rebate create a negative marginal cost, and what operational behavior does it incentivize? | Proves mathematically that crossing a retroactive threshold applies the discount to all prior units; shows that the net spend delta is negative near the boundary; explains how buyers engage in end-of-year forward-buying or channel stuffing to hit thresholds. | Confuses retroactive rebates with marginal bracket discounts; claims that buying more units always increases total spend. |
| 4 | How do you validate contract data extracted by an LLM before committing terms to an enterprise spend database? | Rejects overall percentage accuracy; mandates stratified random sampling ($n \ge 100$) per clause type; computes Wilson score confidence intervals; enforces immutable provenance linking extracted fields to document, page, paragraph, and text hash. | Claims LLMs are 95% accurate and trustworthy; assumes manual review of a handful of random contracts certifies an entire enterprise database. |
| 5 | What evidence is required to overcome a US Customs detention under the Uyghur Forced Labor Prevention Act (UFLPA)? | Explains the statutory rebuttable presumption; details the complete batch-level chain of custody from raw material extraction (quartz/cotton/aluminum) through processing, ingots, components, and import bills, backed by payroll, production, and transit logs. | Suggests submitting a signed supplier certificate of origin or Tier-1 ISO certification; assumes foreign government declarations satisfy US CBP. |
| 6 | How do you calculate contractual take-or-pay shortfall penalties, and how does force majeure alter the liability? | Formulates penalty as shortfall units times liquidated damages or supplier's unamortized fixed cost ($P_{\text{base}} - c_{\text{avoidable}}$); notes that standard force majeure clauses exclude subcontractor failures unless specifically negotiated. | Believes take-or-pay requires paying full purchase price for unmade goods; assumes any external disruption automatically excuses performance. |
| 7 | What is commercial contract leakage, and how do you build an analytics workflow to recover it? | Identifies discrepancies between contractual terms and AP disbursements: uncollected retroactive rebates, unauthorized index price increases, and missed cash discounts; executes a four-way join across PO, GRN, invoice, and contract. | Views leakage purely as maverick off-catalog spend; relies on suppliers to self-report and credit earned volume rebates. |
| 8 | Under the EU CSDDD and German LkSG, what are an enterprise's legal obligations regarding Tier-2 and Tier-3 suppliers? | Differentiates regular annual risk management at Tier 1 from event-driven obligations at Tier N; explains that "substantiated knowledge" of sub-tier violations legally triggers mandatory risk analyses, corrective action plans, and potential civil liability. | Believes due diligence laws apply strictly to direct Tier-1 counterparties; claims that non-European suppliers are exempt from European value-chain regulations. |

## 15. Key terms

- [Audit rights clause](#57-regulatory-compliance-due-diligence-and-esg-traceability) — contractual right permitting buyer inspection of supplier books, facilities, and sub-tier records.
- [Bill of Materials (BOM)](#51-multi-tier-network-ontology-and-graph-modeling) — hierarchical engineering product tree defining component consumption per finished good.
- [BOM explosion](#52-multi-level-bom-traversal-and-propagation) — recursive traversal algorithm calculating gross multi-tier component requirements from end demand.
- [Clause-level provenance](#54-contract-data-extraction-architecture) — cryptographic and positional metadata linking extracted data to document, page, and paragraph.
- [Common-mode failure](#53-discovering-hidden-concentration-and-sub-tier-single-points-of-failure) — concurrent failure of multiple suppliers caused by a shared sub-tier dependency.
- [Contract leakage](#56-commercial-commitments-rebates-and-financial-mechanics) — unearned or uncollected financial value arising from non-enforcement of contractual terms.
- [Cumulative lead time](#52-multi-level-bom-traversal-and-propagation) — longest duration along the BOM critical path required to procure materials and manufacture an item.
- [CSDDD](#8-data-standards-and-regulatory-context) — EU Corporate Sustainability Due Diligence Directive establishing mandatory supply chain duty of care.
- [Flow-down clause](#57-regulatory-compliance-due-diligence-and-esg-traceability) — contract provision obligating a direct supplier to bind its subcontractors to identical standards.
- [Force majeure carveout](#6-worked-example) — specific contract exception defining which events excuse non-performance and excluding subcontractor failure.
- [GS1 EPCIS](#8-data-standards-and-regulatory-context) — open standard for recording and sharing physical supply chain visibility events across enterprises.
- [Incremental rebate](#56-commercial-commitments-rebates-and-financial-mechanics) — tiered rebate structure applying discounted rates only to marginal units within each bracket.
- [LkSG](#8-data-standards-and-regulatory-context) — German Supply Chain Due Diligence Act holding enterprises liable for human rights across supply networks.
- [Multi-tier network](#51-multi-tier-network-ontology-and-graph-modeling) — heterogeneous graph modeling entities, manufacturing sites, and parts across multiple supply echelons.
- [Product Carbon Footprint (PCF)](#8-data-standards-and-regulatory-context) — verified cradle-to-gate greenhouse gas emissions associated with a specific product unit under ISO 14067.
- [Rebate cliff](#56-commercial-commitments-rebates-and-financial-mechanics) — sharp financial discontinuity in retroactive rebates where small volume deltas cause large net spend shifts.
- [Retroactive rebate](#56-commercial-commitments-rebates-and-financial-mechanics) — volume rebate applying discount percentage across all purchased units once a threshold is reached.
- [Revenue-at-Risk (RaR)](#53-discovering-hidden-concentration-and-sub-tier-single-points-of-failure) — gross commercial revenue lost during an unbuffered supply disruption window.
- [Scope 3 Category 1](#8-data-standards-and-regulatory-context) — upstream greenhouse gas emissions embodied within purchased goods and services.
- [Take-or-pay](#56-commercial-commitments-rebates-and-financial-mechanics) — contractual commitment obligating a buyer to purchase minimum volume or pay liquidated shortfall damages.
- [Time-to-Recover (TTR)](#53-discovering-hidden-concentration-and-sub-tier-single-points-of-failure) — duration required for a disrupted supplier or facility to restore full operational output.
- [Time-to-Survive (TTS)](#53-discovering-hidden-concentration-and-sub-tier-single-points-of-failure) — duration an enterprise can sustain finished assembly operations following an upstream facility stoppage.
- [UFLPA](#8-data-standards-and-regulatory-context) — US trade statute barring imports containing materials from Xinjiang under a rebuttable presumption.
- [Wilson score interval](#55-verification-sampling-confidence-intervals-and-governance) — binomial confidence interval formulation robust for small sample sizes and extreme proportions.

## 16. Sources

1. European Parliament and Council, [Directive (EU) 2024/1760 (Corporate Sustainability Due Diligence Directive)](https://eur-lex.europa.eu/eli/dir/2024/1760/oj), Official Journal of the European Union, 2024; amended by Omnibus I, March 2026.
2. Federal Ministry of Labour and Social Affairs (Germany), [Act on Corporate Due Diligence Obligations in Supply Chains (Lieferkettensorgfaltspflichtengesetz - LkSG)](https://www.bmas.de/EN/Services/Laws/act-on-corporate-due-diligence-in-supply-chains.html), Bundesgesetzblatt, 2021/2023.
3. United States Congress, [Uyghur Forced Labor Prevention Act (Public Law 117-78; 19 U.S.C. § 1307)](https://www.congress.gov/bill/117th-congress/house-bill/1155), US Government Publishing Office, 2021.
4. Simchi-Levi, D., Schmidt, W., and Wei, Y. (2014). “From Superstorms to Factory Fires: Managing Silicon Valley's Supply Chain Risk.” *Harvard Business Review*, 92(1-2), 96–101. Foundation of Time-to-Survive and Time-to-Recover methodologies.
5. Basole, R. C., Bellamy, M. A., and Park, H. (2017). “Visualization of Global Supply Network Risks: A Graph-Theoretic Approach.” *Decision Support Systems*, 104, 110–121. [DOI: 10.1016/j.dss.2017.10.007](https://doi.org/10.1016/j.dss.2017.10.007).
6. Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Cambridge, MA. Mathematical foundations of non-linear pricing, two-part tariffs, and vertical contracts.
7. World Resources Institute and WBCSD, [GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard](https://ghgprotocol.org/standards/scope-3-standard), 2011; and PACT, [Pathfinder Framework for Product Carbon Footprints](https://carbon-transparency.com), Version 2.0, 2023.
8. GS1, [EPCIS Standard (Electronic Product Code Information Services) Version 2.0](https://www.gs1.org/standards/epcis), GS1 AISBL, 2022.
9. Wilson, E. B. (1927). “Probable Inference, the Law of Large Numbers, and Run of Mince.” *Journal of the American Statistical Association*, 22(158), 209–212. [DOI: 10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953). Formulation of the Wilson score confidence interval.

## 17. Links

- [Module 14 topic index](_index.md).
- [Module 14 survey](../../../course/14_procurement_supplier_risk_spend.md).
- [Source-to-pay](01_source-to-pay.md), sibling topic.
- [Spend cube construction](02_spend-cube.md), sibling topic.
- [Classification taxonomies](03_classification-taxonomies.md), sibling topic.
- [Kraljic segmentation](04_kraljic-segmentation.md), sibling topic.
- [Should-cost and total cost of ownership](05_should-cost-tco.md), sibling topic.
- [Sourcing optimization](06_sourcing-optimization.md), sibling topic.
- [Auctions and negotiations](07_auctions-negotiations.md), sibling topic.
- [Supplier scorecards](08_supplier-scorecards.md), sibling topic.
- [Supplier risk: financial and operational](09_supplier-risk.md), sibling topic.
- [Production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
