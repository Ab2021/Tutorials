# Module 42 — Treasury, Credit, and Financial Supply Chain Modelling

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Finance and Supply Chain tracks
> **Prerequisites:** Modules 04, 14, 25, 36
> **Tools:** Python, cash-flow models, credit scoring, FX and NPV libraries
> **Industry question:** "The supply chain is profitable on paper — why is there no cash?"

---

## 1. Why industry cares

Supply chains consume cash before they produce it. A company buys materials, pays suppliers, holds inventory, sells on credit, and collects later. The gap between cash out and cash in is the **cash conversion cycle**, and it is often larger than the operating margin.

Treasury and credit are therefore not back-office functions. They determine:

- Whether a supplier can be paid early enough to keep shipping.
- Whether a customer is a credit risk or a growth opportunity.
- Whether a trade finance instrument is cheaper than the working capital it releases.
- Whether a currency or interest-rate move wipes out the margin.

The analytics problem is **quantifying cash, credit, and financial risk across the supply chain**, and choosing instruments that release cash without transferring unacceptable risk.

## 2. Learning outcomes

By the end of this module, you can:

- Compute and decompose the cash conversion cycle.
- Build a rolling cash-flow forecast for a supply chain.
- Model working capital release from payment-term and inventory changes.
- Estimate counterparty credit risk with PD, LGD, and expected loss.
- Evaluate trade finance instruments: letters of credit, factoring, reverse factoring, dynamic discounting, and inventory finance.
- Compare the cost of finance with the cost of supply disruption.
- Measure and hedge FX and interest-rate exposure.
- Apply NPV and WACC to supply chain investments.
- Design credit limits, insurance, and monitoring.
- Recognise KYC, AML, sanctions, and fraud risks in supply chain finance.

## 3. Mental model

```text
   CASH OUT ──> MATERIALS ──> INVENTORY ──> SALES ──> RECEIVABLES ──> CASH IN
      │             │             │            │            │
   payment       DIO           DIO          DSO          collection
   terms         days          days         days         and credit
      │
   DPO: days payable outstanding

   CCC = DIO + DSO - DPO

   RISK LAYER: supplier credit, customer credit, FX, interest rate, commodity
   INSTRUMENTS: LC, guarantee, factoring, reverse factoring, insurance, hedge
```

The physical chain and the financial chain run in opposite directions. Every day of inventory or receivable is a day of funding.

### 3.1 The three financial flows

| Flow | Question | Owner |
|---|---|---|
| Operating cash | When does cash move? | Treasury and FP&A |
| Credit | Will the counterparty pay? | Credit and procurement |
| Market risk | Will FX or rates move? | Treasury |

## 4. Core concepts

### 4.1 Cash conversion cycle

\[
\text{CCC} = \text{DIO} + \text{DSO} - \text{DPO}
\]

where DIO is days inventory outstanding, DSO is days sales outstanding, and DPO is days payables outstanding.

A CCC of 75 days means the company funds 75 days of operations. At €1 million of daily cost of goods sold, that is €75 million of working capital.

### 4.2 Cash-flow forecasting

A supply chain cash forecast is built from:

- Purchase orders and payment terms.
- Supplier lead times and invoice timing.
- Inventory build and depletion.
- Sales forecast and collection profile.
- Payroll, tax, and capital expenditure.

The forecast should be **driver-based**, not a flat extrapolation. A 13-week rolling forecast is the standard liquidity tool; a 12-month forecast supports planning.

### 4.3 Credit risk

Expected loss on a counterparty exposure:

\[
\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}
\]

- **PD:** probability of default.
- **LGD:** loss given default, after recovery.
- **EAD:** exposure at default.

For 200 suppliers with €50 million of exposure, a 2% PD and 45% LGD:

\[
\text{EL} = 0.02 \times 0.45 \times 50{,}000{,}000 = €450{,}000
\]

Credit risk models use financial ratios, payment behaviour, trade references, and external ratings. A small supplier with a concentrated customer base can be riskier than its balance sheet suggests.

### 4.4 Trade finance instruments

| Instrument | What it does | Cost | Risk |
|---|---|---|---|
| Letter of credit | Bank guarantees payment | Bank fee | Documentation and compliance |
| Bank guarantee | Guarantees performance or payment | Fee | Counter-guarantee |
| Factoring | Sells receivables | Discount | Customer credit and recourse |
| Reverse factoring | Buyer's bank pays supplier early | Discount | Buyer bears credit risk |
| Dynamic discounting | Buyer pays early for a discount | Buyer cash | Opportunity cost |
| Purchase-order finance | Funds production against a PO | Interest and fee | Performance risk |
| Inventory finance | Funds stock | Interest | Collateral and control |
| Credit insurance | Covers receivable default | Premium | Exclusions and limits |

### 4.5 Reverse factoring economics

A supplier invoices €1,000,000 with 90-day terms. The buyer's bank offers early payment at day 60 at a 6% annual discount. The supplier's own borrowing cost is 10%.

Bank discount for 30 days:

\[
1{,}000{,}000 \times 0.06 \times 30/365 = €4{,}932
\]

Supplier saving versus borrowing for 30 days at 10%:

\[
1{,}000{,}000 \times 0.10 \times 30/365 = €8{,}219
\]

The supplier is better off by about €3,287, and the buyer may extend its own payment terms because the bank now carries the supplier's credit risk. The programme works when the buyer's credit is stronger than the supplier's.

### 4.6 FX and interest-rate exposure

FX exposure arises when costs and revenues are in different currencies. Hedging instruments:

- **Forward contract:** locks a rate.
- **Option:** protects against adverse moves while retaining upside.
- **Natural hedge:** match currency costs and revenues.
- **Netting and pooling:** offset exposures across entities.

The analytics question is **net exposure**, not gross transaction volume. A company with dollar revenue and dollar costs has little net exposure even if both are large.

### 4.7 Cost of capital and NPV

\[
\text{NPV} = \sum_{t=0}^{T} \frac{CF_t}{(1+r)^t}
\]

where r is the weighted average cost of capital adjusted for project risk. A working capital release is a one-time cash inflow; an inventory reduction also reduces holding cost. Both must be counted, and the release must not be double-counted as recurring savings.

### 4.8 Credit policy and limits

A credit policy defines:

- Who gets credit and for how long.
- Limits by customer, country, and sector.
- Security: guarantees, letters of credit, insurance.
- Escalation and exception approval.
- Monitoring and early warning.

The trade-off is between sales growth and bad debt. A credit score alone is not a policy; the policy needs a limit, a term, and a collection process.

### 4.9 Supply chain finance and the supplier's perspective

Buyer-led supply chain finance can be a lifeline for small suppliers, or a hidden cost. The supplier may:

- Receive early payment at a discount.
- Have no visibility of the programme's true cost.
- Become dependent on one buyer's bank.
- Lose bargaining power on price.

The analytics should report the **all-in cost to the supplier**, not only the buyer's saving.

### 4.10 Worked example — cash conversion cycle

A distributor has DIO of 60 days, DSO of 45 days, and DPO of 30 days.

\[
\text{CCC} = 60 + 45 - 30 = 75 \text{ days}
\]

Annual cost of goods sold is €365 million, so daily COGS is €1 million. Working capital is approximately €75 million.

If DSO falls by 10 days through better collections, €10 million is released. If DIO falls by 10 days, another €10 million is released. If DPO is extended by 15 days, €15 million is released — but the supplier bears the cost, and the extension may not be sustainable.

### 4.11 Worked example — letter of credit versus open account

A buyer sources $2,000,000 of goods from a supplier in a jurisdiction with weak contract enforcement. Payment terms are 90 days. The supplier's own borrowing cost is 9%, because its credit is weak. The buyer's bank offers a letter of credit at 0.6% of value, and will discount the accepted draft at 4.5% for the 90-day period.

Under open account, the supplier funds 90 days itself:

\[
2{,}000{,}000 \times 0.09 \times 90/365 = $44{,}384
\]

Under the letter of credit, the buyer pays a fee:

\[
2{,}000{,}000 \times 0.006 = $12{,}000
\]

and the supplier discounts at the bank's rate:

\[
2{,}000{,}000 \times 0.045 \times 90/365 = $22{,}192
\]

Total financing and fee cost is $34,192, about $10,200 less than open account. The letter of credit also converts the supplier's payment risk into the issuing bank's risk, which may be the more important benefit when enforcement is uncertain.

The comparison is not automatic. Documentation compliance, amendment fees, and the operational burden of presentation can outweigh the arithmetic, particularly for small or frequent shipments. The correct analysis lists the fees, the discount, and the risk transfer separately, then compares them against the alternative the supplier would actually use.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Invoices and payments | ERP / AP / AR | DIO, DSO, DPO |
| Bank statements | Treasury / bank | Cash position |
| Supplier financials | Credit bureau / filings | PD and limits |
| Customer financials | Credit bureau / filings | Credit risk |
| FX rates | Market feed | Exposure and valuation |
| Interest rates | Market feed | Discounting |
| Trade finance | Bank / TMS | Instrument cost |
| Contracts | CLM | Payment terms |

### 5.2 Standards and regulation

- **IFRS 9** and **IFRS 7** for financial instruments and credit loss.
- **Basel III** for bank capital and liquidity (indirect effect on pricing).
- **UCP 600** for letters of credit.
- **URDG 758** for demand guarantees.
- **KYC/AML** rules and **sanctions** screening.
- **Transfer pricing** rules for intercompany terms.
- **SOX** and internal control requirements where applicable.

### 5.3 Workflow

1. Forecast cash from operating drivers.
2. Measure CCC and working capital.
3. Assess counterparty credit.
4. Choose finance instruments.
5. Hedge market risk.
6. Monitor limits and exposures.
7. Report and reconcile.

## 6. Industry case studies

### 6.1 Reverse factoring programmes

**Context:** Large buyers offered early payment to suppliers through banks.
**Intervention:** Supply chain finance platforms and buyer-backed programmes.
**Evidence:** Academic and regulatory studies (grade B/C); accounting treatment has been debated.
**Outcome:** Suppliers received earlier cash; buyers extended terms; some programmes were reclassified as debt-like.
**What transferred:** the benefit depends on whose credit is used and how the programme is disclosed.
**What did not:** specific costs and adoption rates.

### 6.2 Retail supplier payment terms

**Context:** Retailers extended payment terms to release working capital.
**Intervention:** Terms changes and supplier finance.
**Evidence:** Company disclosures and press reporting (grade C).
**Outcome:** Retailers improved cash, but small suppliers bore the cost and some failed.
**What transferred:** working capital release is a transfer, not a free gain.
**What did not:** specific term lengths.

### 6.3 FX exposure in sourcing

**Context:** A company sourced in dollars and sold in euros.
**Intervention:** Forward hedging and natural hedging.
**Evidence:** Treasury case reporting (grade C).
**Outcome:** Hedging reduced volatility but did not eliminate competitive exposure.
**What transferred:** hedge the net exposure, and distinguish accounting hedge from economic hedge.
**What did not:** specific rates.

### 6.4 A composite failure — the supplier that looked healthy

**Context:** A supplier showed strong revenue growth and was given a large credit limit.
**Intervention:** No cash-flow or concentration analysis.
**Evidence:** Composite teaching case.
**Outcome:** The supplier's cash was consumed by a single large customer, and it failed.
**What transferred:** credit risk is a cash-flow and concentration question, not a revenue question.
**What did not:** the specific loss.

## 7. Labs

### 7.1 Lab 42.1 — Cash conversion cycle

Compute CCC for three companies and decompose the drivers. Estimate the cash released by a five-day improvement in each component.

**Deliverable:** CCC analysis and release estimate.

### 7.2 Lab 42.2 — Thirteen-week cash forecast

Build a driver-based 13-week cash forecast from orders, payment terms, and collections. Stress it with a demand drop.

**Deliverable:** Forecast model and liquidity analysis.

### 7.3 Lab 42.3 — Credit scoring

Build a supplier credit score from financial ratios and payment behaviour. Set limits and test the expected loss.

**Deliverable:** Scorecard and limit policy.

### 7.4 Lab 42.4 — Trade finance comparison

Compare a letter of credit, reverse factoring, and dynamic discounting for a €5 million annual spend.

**Deliverable:** Cost comparison and recommendation.

### 7.5 Lab 42.5 — FX hedge

Given a 12-month USD payable schedule, compare no hedge, forward, and option. Quantify the residual risk.

**Deliverable:** Hedge analysis and policy memo.

## 8. Assignment / project

Design the financial supply chain for a company or programme.

**Required components:**

1. Cash conversion cycle and working capital.
2. Cash-flow forecast.
3. Counterparty credit assessment.
4. Trade finance instrument choice.
5. FX and interest-rate exposure.
6. NPV and cost of capital.
7. Credit policy and monitoring.
8. Compliance and fraud controls.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Cash cycle | 20% | Computes CCC | Decomposes drivers | Release and sustainability |
| Forecast | 20% | Extrapolates | Driver-based | Stress and scenario |
| Credit | 20% | Uses rating | PD/LGD/EL | Concentration and early warning |
| Instruments | 15% | Lists options | Cost comparison | All-in cost and risk transfer |
| Market risk | 15% | Names FX | Exposure and hedge | Net exposure and residual risk |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Accounting:** supply chain finance may be debt-like; disclosure matters.
- **AML/KYC:** counterparties and beneficial owners must be screened.
- **Sanctions:** payments and trade can be restricted.
- **Transfer pricing:** intercompany terms must be arm's length.
- **Fair dealing:** extending terms to weak suppliers can transfer distress.
- **Fraud:** invoice fraud, double financing, and fake receivables are real risks.
- **Data protection:** financial and personal data must be protected.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| CCC ignored | Cash crunch | Growth consumes cash | Insolvency | CCC management |
| Revenue-based credit | Bad debt | Cash flow not assessed | Loss | Cash-flow scoring |
| Hidden SCF debt | Audit query | Accounting treatment | Restatement | Disclosure |
| Supplier distress | Late deliveries | Buyer extends terms | Supply failure | Supplier monitoring |
| FX mismatch | Margin volatility | Unhedged exposure | Loss | Net exposure hedge |
| Instrument cost ignored | Expensive cash | Fee and discount | Margin erosion | All-in cost |
| Double financing | Bank alert | Same invoice financed twice | Fraud loss | Registry and controls |
| Sanctions breach | Screening gap | Party missed | Penalty | Screening system |

## 12. Interview and discussion questions

1. What is the cash conversion cycle? *(Hint: DIO + DSO − DPO.)*
2. Why can a profitable company run out of cash? *(Hint: working capital and timing.)*
3. How do you forecast cash for a supply chain? *(Hint: driver-based, orders and terms.)*
4. What is expected loss? *(Hint: PD × LGD × EAD.)*
5. How do you assess a small supplier's credit? *(Hint: cash flow, concentration, payment behaviour.)*
6. What is reverse factoring? *(Hint: buyer's bank pays supplier early.)*
7. Who bears the cost of extended payment terms? *(Hint: often the supplier.)*
8. How do you hedge FX exposure? *(Hint: net exposure, forwards, options.)*
9. When is a letter of credit worth the fee? *(Hint: cross-border, weak counterparty.)*
10. How do you avoid double financing? *(Hint: registry and invoice control.)*
11. What is the all-in cost of supply chain finance? *(Hint: discount, fees, dependency.)*
12. How does working capital release affect NPV? *(Hint: one-time cash inflow plus holding cost.)*

## 13. Key terms

| Term | Definition |
|---|---|
| CCC | Cash conversion cycle |
| DIO | Days inventory outstanding |
| DSO | Days sales outstanding |
| DPO | Days payables outstanding |
| Working capital | Current assets minus current liabilities |
| PD | Probability of default |
| LGD | Loss given default |
| EAD | Exposure at default |
| Letter of credit | Bank payment guarantee |
| Reverse factoring | Buyer-backed early supplier payment |
| Dynamic discounting | Early payment for a discount |
| Factoring | Sale of receivables |
| FX exposure | Risk from currency moves |
| Forward | Contract to exchange currency at a future rate |
| WACC | Weighted average cost of capital |
| KYC | Know your customer |
| AML | Anti-money-laundering |

## 14. Sources and further reading

### Standards and regulation

- IFRS 9, Financial Instruments. https://www.ifrs.org/
- Basel Committee, Basel III framework. https://www.bis.org/
- ICC, UCP 600 and URDG 758. https://iccwbo.org/
- FATF, AML and KYC guidance. https://www.fatf-gafi.org/
- OECD, transfer pricing guidelines. https://www.oecd.org/

### Reports and official sources

- Bank for International Settlements, trade finance data. https://www.bis.org/
- World Bank, trade finance gap reports. https://www.worldbank.org/
- national central bank financial stability reports.

### Peer-reviewed and professional

- Brealey, R., Myers, S., Allen, F., *Principles of Corporate Finance*.
- Altman, E. I., credit scoring and distress prediction research.
- Klapper, L., trade finance and supply chain finance research.
- Hofmann, E., Belin, O., *Supply Chain Finance Solutions*.

### Industry

- Association of Financial Professionals, treasury guidance. https://www.afponline.org/
- ICC, trade finance surveys. https://iccwbo.org/
- rating agency methodology reports.

> **Next:** [Module 43](43_security_engineering_supply_chain.md) addresses the adversarial layer: how the chain is attacked and how it is defended.
