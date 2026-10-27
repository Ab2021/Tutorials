# Access Pathways, Licensing, and Personal Data

> Module 02 — The Supply Chain Data Landscape
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01, 02.02, 02.03, 02.04, 02.05, 02.06

---

## 1. Why this topic exists

A dataset can be accurate, complete, well keyed, and still unusable.

Three situations recur in supply chain analytics, and none of them is a data-quality problem:

1. **The panel file that cannot leave the building.** A category team buys a syndicated retail
   panel. An analyst joins it to internal shipment data, builds a share-shift model, and emails
   the scored output to a customer. The output contains panel-derived market shares at
   account level. The subscription permits internal business use and limited excerpting, not
   redistribution of a derived product. The analysis was good and the delivery breached the
   agreement.
2. **The telematics file reused for a new question.** A fleet's GPS unit data was collected to
   route vehicles and to defend insurance claims. Two years later a data scientist uses the same
   table to score driver braking behaviour for a performance dashboard. The data is identical.
   The purpose is not, and the original basis may not cover it.
3. **The 3PL that offers to share "its" data.** A logistics provider proposes to give a
   manufacturer delivery-performance data about the manufacturer's own orders. The provider
   processed that data on the manufacturer's behalf. A processor does not acquire the right to
   license the data by holding it, and the manufacturer may already be the controller of the
   same records.

Each case has the same shape. **Access is a chain of permissions, not a copy of a file.** The
permission has four dimensions — who may use the data, for what purpose, in which territory,
and for how long — and the weakest link in the chain governs the whole use.

Most analytics curricula treat access as a procurement step: find the source, request the
extract, receive the CSV. That model works for a public dataset and fails for almost everything
else in a commercial supply chain. The failure is expensive in three ways. It wastes the
analysis. It creates legal exposure that the analyst usually does not own but often causes. And
it destroys the commercial relationship that made the data available, which is the only asset
that cannot be rebuilt from a backup.

This topic follows the transport topic (02.06) because transport determines what arrives and
when, and precedes data quality (02.08) because quality is only meaningful inside a permitted
use. A clean dataset used outside its licence is not high quality; it is a liability with good
column names.

The topic is written for analysts, not lawyers. It teaches you to recognise the questions, to
read the clauses that answer them, and to know when the answer must come from counsel. It does
not teach you to give the answer yourself.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** any supply chain data source into one of the four access pathways — direct
   system access, partner exchange, commercial purchase, public or open — and **state** its
   cost, latency, and dominant legal obligation.
2. **Explain** why a licence is part of the data's definition, and **identify** the four clauses
   that most often constrain analytics: redistribution, reverse engineering, combination, and
   model training.
3. **Distinguish** personal data from business contact data using the identifiability test, and
   **classify** driver, consignee, and customer records accordingly.
4. **Assign** controller and processor roles in a described logistics arrangement, and **state**
   what each role permits an analytics team to do.
5. **Select** a cross-border transfer mechanism for a described data flow — adequacy, standard
   contractual clauses, or a derogation — and **explain** why a cloud region is a compliance
   decision.
6. **Apply** the purpose-limitation test to a proposed secondary use, and **distinguish** a
   compatible further use from one that requires a new basis.
7. **Trace** a proposed use case through a six-gate review and **produce** the
   alternatives that would make a blocked use permissible.
8. **Read** a data-sharing agreement for the five operational clauses that determine what an
   analytics team can build and what it must delete.

Each outcome maps to an exercise in section 12 and a self-check in section 13.

## 3. The core mental model

Hold one picture: four doors into one room, and the room has a ceiling that is lower than any
single door.

```text
                    FOUR DOORS, ONE ROOM, ONE CEILING

  +--------------+   +--------------+   +--------------+   +--------------+
  |   DIRECT     |   |   PARTNER    |   |  COMMERCIAL  |   |   PUBLIC /   |
  |   SYSTEM     |   |   EXCHANGE   |   |   PURCHASE   |   |     OPEN     |
  |   ACCESS     |   |  (EDI/API)   |   |  (panel,     |   |  (statistics,|
  | ERP/WMS/TMS  |   |  portal      |   |   market data)|  |   registers) |
  +------+-------+   +------+-------+   +------+-------+   +------+-------+
         |                  |                  |                  |
         v                  v                  v                  v
  +-----------------------------------------------------------------------+
  |                      THE PERMISSION ENVELOPE                          |
  |                                                                       |
  |   lawful basis  x  contract or licence  x  transfer mechanism  x      |
  |   retention limit  x  security obligation                             |
  |                                                                       |
  |   Every use must fit inside ALL FIVE walls at once.                   |
  +-----------------------------------------------------------------------+
         |                  |                  |                  |
         v                  v                  v                  v
   internal use      shared analysis     derived insight     redistribution
   (lowest risk)     (contract-bound)    (licence-bound)     (highest risk)
```

The picture has five properties worth memorising.

**Property 1 — A door is a permission, not a pipe.** Direct access is not "we can read the
table". It is "named users may read specified fields for specified purposes, under access
control, for as long as the role lasts". Partner exchange is not "the retailer sends us POS"; it
is an extract under a trading-partner agreement that limits what we may do with it. The
mechanism and the permission are separate objects, and engineers see only the first.

**Property 2 — The licence travels with the row.** When you join two tables, you join two
permission envelopes. The result is governed by the intersection, not the union. A join between
an unrestricted internal table and a restricted panel table produces a restricted table. Most
analysts intuitively apply the union — "it is mostly my own data" — and that intuition is
wrong.

**Property 3 — The most restrictive clause wins, and it is usually not the one you read.**
Redistribution limits are visible and discussed. Combination limits, model-training limits, and
derived-output limits are buried. A licence that permits internal use but prohibits "use to
develop or train any machine-learning model" constrains the analytics team far more than the
redistribution clause constrains the sales team.

**Property 4 — Deletion is distributed, and therefore usually incomplete.** The rows sit in a
warehouse, a feature store, a notebook cache, a backup tape, a vendor's sandbox, and a model's
training set. Deleting the source table satisfies a naive reading of "delete on termination" and
none of the substance. Section 5 treats deletion as a lifecycle problem, not an event.

**Property 5 — The same record has different legal status in different hands.** A consignee's
name and address is personal data when the retailer holds it, personal data when the carrier
holds it as a processor, and possibly anonymous when aggregated to a postcode-level
delivery-success rate. The record did not change; the holder, purpose, and transformation did.

### 3.1 The four pathways compared

| Pathway | Typical examples | Money cost | Latency to first usable extract | Dominant legal obligation | Who usually owns the relationship |
|---|---|---|---|---|---|
| Direct system access | ERP, WMS, TMS, OMS, POS, telematics platform | Integration effort and licence seats; marginal cost near zero | Minutes to days | Employment and access-control policy; internal data governance; purpose recorded in the privacy notice | Internal IT and data owner |
| Partner exchange | EDI 850/856/810, supplier portal, carrier API, retailer vendor portal | Mapping and integration; often charged per document or per connection | Hours to days, set by the partner's batch cycle | Trading-partner agreement; data-processing agreement where personal data is involved; partner's own restrictions | Commercial or procurement, with IT delivery |
| Commercial purchase | Syndicated retail panel, market and trade data, credit and risk data, paid weather or geospatial feeds | Subscription, often annual and usage-tiered | Days to weeks; publication lag is part of the product | Licence terms: scope of use, redistribution, combination, model training, audit | Procurement, with legal review |
| Public and open | National statistics, customs and trade data, open geospatial, company registers, open transport feeds | Near zero licence cost; substantial analyst time | Weeks to quarters; publication lag is structural | Open licence: attribution, share-alike, and sometimes non-commercial restriction | Nobody, which is the problem |

Read the last column. The pathway determines who you negotiate with, and the negotiation is
usually harder than the integration. A team that plans only for the technical work will be
surprised by a three-month legal review on a dataset it could already query in a sandbox.

### 3.2 What each pathway is good for

| Decision | Best-fit pathway | Why | What it cannot answer |
|---|---|---|---|
| Replenishment for your own DC | Direct | You own the inventory and the orders | Competitor or category context |
| Retailer compliance and chargebacks | Partner exchange | The retailer's rules are in the documents | Whether the rule is commercially sensible |
| Category share and distribution | Commercial purchase | Only a panel measures the total market consistently | Anything below the panel's reporting grain |
| Macro demand context | Public and open | Cheap, stable, defensible | Supplier-level or store-level behaviour |
| Driver safety and route performance | Direct telematics | Only you hold the vehicle data | Whether the driver was at fault |
| Supplier financial distress | Commercial purchase | Credit data is a product | Whether the supplier will prioritise you |

The practical rule: **buy the market view, exchange the partner view, own the operational view,
and use open data for context.** A use case that needs two of these at once is a use case that
needs two permission envelopes, and therefore two reviews.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Personal data | Information relating to an identified or identifiable natural person | Treated as a synonym for "sensitive" | Legal category |
| Data subject | The natural person the data relates to | Confused with the customer account | Person identifier |
| Controller | The party that determines the purposes and means of processing | Assumed to be whoever holds the data | Role, not system |
| Processor | A party that processes personal data on the controller's behalf | Treated as a co-owner of the data | Role, not system |
| Joint controllers | Two or more parties that jointly determine purposes and means | Confused with controller–processor | Role, with an arrangement under Art. 26 |
| Sub-processor | A processor engaged by another processor | Omitted from the contract chain | Role, one level deeper |
| Data-processing agreement (DPA) | The contract required when a processor handles personal data | Treated as a formality | Contract, Art. 28 |
| Lawful basis | The legal ground that makes processing permissible | Assumed to be consent in every case | One of six bases under Art. 6(1) |
| Consent | Freely given, specific, informed, unambiguous indication of wishes | Used where refusal is not genuinely possible | Flag plus record of evidence |
| Contract necessity | Processing needed to perform a contract with the data subject | Stretched to cover marketing and analytics | Basis code |
| Legitimate interest | A real and present interest, balanced against the data subject's rights | Treated as a default permission | Documented three-part test |
| Balancing test | The third limb of the legitimate-interest assessment | Skipped, or performed once and never reviewed | Documented assessment |
| Purpose limitation | Data may be used only for the purpose it was collected for, or a compatible one | Read as "any internal purpose" | Purpose register entry |
| Compatible further use | A new purpose that passes the Art. 6(4) assessment | Assumed to cover any "analytics" | Assessment record |
| Data minimisation | Only the data adequate, relevant, and necessary for the purpose | Read as "collect everything, delete later" | Field-level inventory |
| Storage limitation | Data kept no longer than necessary | Confused with backup retention | Retention schedule |
| Retention schedule | The defined period and deletion trigger for each data class | Absent, or set to "forever" by default | Days or months per class |
| Special category data | Data revealing health, biometrics, ethnicity, and similar | Missed in free-text and image data | Art. 9 flag |
| Pseudonymisation | Replacing identifiers so data cannot be attributed without extra information | Confused with anonymisation | Technique, reversible |
| Anonymisation | Irreversibly removing identifiability | Asserted without a re-identification test | Technique, irreversible |
| Re-identification | Recovering identity from supposedly anonymous data | Ignored at small cell sizes | Risk estimate, k or similar |
| k-anonymity | Every record is indistinguishable from at least k−1 others on quasi-identifiers | Reported without stating the quasi-identifiers | Integer k |
| Adequacy decision | A Commission finding that a country provides adequate protection | Confused with data localisation | Country list |
| Standard contractual clauses (SCCs) | Pre-approved contract modules for transfers | Signed without a transfer impact assessment | Decision (EU) 2021/914 |
| Transfer impact assessment (TIA) | Documented assessment of destination-country law and safeguards | Treated as a one-off at signing | Document per transfer |
| Data localisation | A legal requirement that data stay in a territory | Confused with adequacy | Statute-specific |
| Data Privacy Framework (DPF) | The EU–US adequacy arrangement for certified US organisations | Assumed to cover any US recipient | Certification list |
| Syndicated data | A panel product sold to multiple subscribers on standard terms | Treated as a public dataset once purchased | Subscription |
| Panel | A sample of stores, households, or consumers projected to a population | Treated as a census | Sample design |
| Redistribution | Supplying data or derived output to a third party | Assumed to be allowed internally | Licence clause |
| Reverse engineering | Reconstructing the underlying data or methods from an output | Rarely recognised by analysts | Licence clause |
| Derivative work | An output substantially derived from licensed data | Assumed to belong to the analyst | Licence clause |
| Model-training restriction | A clause limiting use of data to train models | Missed in the AI addendum | Licence clause |
| Data clean room | An environment where outputs are released but row-level data is not | Assumed to make any use lawful | Technical control plus contract |
| Open licence | A licence permitting reuse with limited conditions | Assumed to mean "no conditions" | Licence identifier |
| ODbL | Open Database License: attribution plus share-alike for databases | Applied to outputs that are not databases | Licence identifier |
| CC BY | Creative Commons attribution licence | Confused with CC BY-SA | Licence identifier |
| Share-alike | A condition that derivatives carry the same licence | Ignored when combining sources | Licence clause |
| Data-sharing agreement (DSA) | A contract governing exchange between organisations | Used loosely for any NDA | Contract |
| Audit right | A contractual right to inspect the other party's compliance | Assumed to be theoretical | Contract clause |
| Trade secret | Commercially valuable confidential information | Asserted to block any sharing | Legal category |
| Connected product | A product that obtains, generates, or collects data about its use | Confused with any device with a sensor | Data Act definition |
| Data holder | The party entitled to make connected-product data available | Confused with the data subject | Data Act role |
| Data recipient | A third party receiving data at the user's request | Confused with the controller | Data Act role |
| Exceptional need | The B2G condition for compelled data sharing | Read as any public-interest request | Data Act Art. 15 |
| Records of processing (ROPA) | The inventory of processing activities | Maintained by legal, unused by analytics | Art. 30 record |
| Data protection impact assessment (DPIA) | A structured risk assessment for high-risk processing | Performed after launch | Art. 35 document |
| Right to erasure | The data subject's right to have data deleted | Confused with the contractual deletion clause | Request workflow |

**Controller and processor.** The controller decides *why* and *how* the data is processed. The
processor acts on the controller's documented instructions. In a logistics arrangement the
retailer is normally the controller of the order and consignee data, and the carrier or 3PL is
normally the processor. The distinction is not about who stores the data or who can query it.
It is about who determines the purpose. A processor that starts using the data for its own
product development has, in effect, become a controller for that use — without a basis, and
usually without telling anyone.

**Pseudonymisation and anonymisation.** Pseudonymised data is still personal data. Replacing a
name with a customer ID while keeping the mapping table changes the risk, not the legal status.
Anonymisation means the data can no longer be attributed to a person by anyone, using any means
reasonably likely to be used. The second is a much higher bar, and it is an outcome to be
tested, not a label to be applied.

**Purpose limitation and compatible further use.** Purpose limitation does not forbid every
secondary use. It requires that a new purpose be compatible with the original one, judged on
the link between the purposes, the context of collection, the nature of the data, the
consequences for data subjects, and the safeguards applied. Research and statistical purposes
receive a presumption of compatibility in EU guidance, subject to safeguards — but the
presumption is not unlimited, and it does not convert a commercial product into research.

**Licence and copyright.** A data licence is a contract, and it can prohibit things copyright
never would. Copyright protects the expression of a database; a contract can restrict the use of
facts extracted from it. That is why a panel agreement can forbid using the data to train a
model even though the model is a new work. The contract, not copyright, is the operative
instrument for most commercial data.

**Trade secret and personal data are different regimes.** A trade secret can be lost by
independent derivation; personal data protection cannot be waived by the holder. A dataset can
be both, and then the stricter constraint on each dimension governs.

**Business contact data is a question, not a category.** A generic `info@company.com` is
generally treated as the legal person's contact detail and outside the GDPR;
`name.surname@company.com` relates to an employee and is generally personal data. The test is
identifiability of a natural person, not the word "business". Supplier portal users, carrier
account managers, and buyer contacts are therefore usually personal data even when the
surrounding record is a company.

## 5. The mechanics, step by step

The sequence below is the order in which the questions must be answered. The order matters
because a later answer can invalidate an earlier one, and because the cheapest point to stop a
bad use case is gate one.

### Phase A — Classify what you have

**Step 1. Identify the data subjects, if any.**

- Question: does any field relate to an identified or identifiable natural person?
- Consignee name and address: yes. Driver identifier: yes. Vehicle registration: yes, where it
  can be linked to a driver. Store-level sales: usually no. Category-level panel share: no.
- Branch: **does the dataset contain free text, images, or location traces?** These frequently
  contain personal data that the schema does not label. A delivery note field can contain a
  phone number. A dashcam clip contains faces and plates.
- Output: a field-level personal-data inventory, not a table-level label.

**Step 2. Classify sensitivity and commercial confidentiality separately.**

- Personal-data sensitivity: ordinary, special category, or criminal-offence data.
- Commercial sensitivity: public, internal, confidential, restricted, or trade secret.
- The two axes are independent. A supplier price list is not personal data and is highly
  commercially sensitive. A driver's home address is personal data and not commercially
  sensitive. Access control must handle both.
- Branch: **does the data reveal something about a person that they would not expect the holder
  to know?** Route traces reveal home location, religious observance through visit patterns, and
  health through hospital deliveries. Expectation is a factor in the balancing test.

**Step 3. Establish the pathway and the counterparty.**

- Direct: the data owner is an internal role.
- Partner: the counterparty is a trading partner, and the governing document is a trading-partner
  agreement plus, where personal data is involved, a DPA.
- Commercial: the counterparty is a vendor, and the governing document is a licence.
- Open: the counterparty is a licensor in name only, and the governing document is the licence
  text.
- Branch: **is the data about your own operations, or about someone else's?** Data about your own
  operations that you generated is usually the simplest to use. Data about your operations that
  a partner generated may still be yours to direct, but the contract decides.

### Phase B — Establish the basis

**Step 4. Determine the role.**

- Are you the controller, a processor, or a joint controller for this processing?
- A retailer directing a carrier is a controller. The carrier is a processor. A carrier that also
  uses the data to benchmark its own network for its own purposes is a controller for that use.
- Branch: **does the contract say "processor" while the practice says "controller"?** Practice
  follows the facts, not the label. A vendor that decides which customers to profile, or that
  reuses data for its own products, is determining purposes.
- Consequence for analytics: a processor's analytics team can build what the controller
  instructs, on the controller's data, for the controller's purposes. It cannot build a
  cross-client benchmark and sell it, because that is a new purpose determined by the processor.

**Step 5. Identify the lawful basis for each purpose.**

- Contract necessity covers what is genuinely needed to deliver: the consignee address for the
  delivery, the driver identity for the shift.
- Legitimate interest covers some operational purposes: fraud prevention, network security,
  limited service-quality analysis. It requires a documented three-part test — a real and
  present interest, necessity, and a balancing assessment — and it is neither a last resort nor
  a default.
- Consent is rarely workable for employees, because the employment relationship makes refusal
  costly and consent therefore not freely given. It is sometimes workable for consumers, and
  almost never workable for a dataset you did not collect.
- Branch: **was the basis recorded at collection, or reconstructed afterwards?** A basis invented
  after the fact is a red flag in an audit and a weak position in a complaint.

**Step 6. Test the proposed purpose against the original one.**

- If the purpose is unchanged, proceed.
- If it is new, run the compatibility assessment: link between purposes, context of collection,
  nature of data, consequences, safeguards.
- Branch: **is the new purpose research or statistics?** EU guidance presumes compatibility for
  statistical and scientific research subject to safeguards. Marketing, commercial benchmarking,
  and product development are not research, however analytically sophisticated they are.
- Output: a written purpose statement, one sentence, that a reviewer could disagree with.

### Phase C — Contract

**Step 7. Read the five clauses that constrain analytics.**

1. **Scope of use.** "Internal business purposes" is narrow. It excludes sharing with customers,
   publication, and often use by affiliates.
2. **Redistribution and derived output.** Does the licence cover outputs, or only inputs? Many
   licences restrict "any product derived from" the data.
3. **Combination.** Does it permit joining with other sources? Some restrict combination with
   competitor data or with data from specified sources.
4. **Reverse engineering and benchmarking.** Does it prohibit reconstructing the underlying
   panel, or publishing comparisons that reveal another party's data?
5. **Model training and AI.** Newer terms restrict generative AI use to internal summarisation
   and prohibit training models on the data.
- Branch: **is there an AI or data-science addendum?** It is usually where the training
  restriction lives.

**Step 8. Establish the operational clauses.**

- Audit right: can the licensor inspect your systems, and with how much notice?
- Sub-processor and vendor approval: may the data sit in a cloud service the licensor has not
  approved?
- Security schedule, termination and deletion, and change of control.
- Branch: **does the contract require deletion of derived data or model artefacts?** This is the
  clause most often ignored and most often impossible to satisfy retrospectively.

**Step 9. Confirm the transfer mechanism.**

- Where will the data be stored and processed, and where will it be accessed from?
- If outside the origin jurisdiction, is there an adequacy decision, SCCs, or a derogation?
- Branch: **is the cloud region the same as the legal transfer?** No. A region determines where
  the data sits; the transfer mechanism determines whether the transfer is lawful. Support
  access from another country is itself a transfer in many interpretations.

### Phase D — Use, retain, and exit

**Step 10. Minimise before modelling.**

- Drop fields not needed for the purpose. Aggregate where row level is unnecessary. Truncate
  identifiers. Replace direct identifiers with pseudonyms and keep the mapping separately.
- Record what was dropped and why. The record is the defence.

**Step 11. Set retention and deletion at the start.**

- Define a period per data class, with a trigger: contract end, purpose completion, or a fixed
  period after the last event.
- Identify every copy: warehouse, feature store, notebook, sandbox, vendor environment, backup,
  model training set, exported report.
- Branch: **can the derived artefacts be deleted?** A trained model may encode the data in a way
  that cannot be surgically removed. If deletion is required, the model may need retraining
  without the data, or the licence must permit retention of the model.

**Step 12. Prepare for audit and exit.**

- Keep the basis record, the compatibility assessment, the data inventory, the retention log, and
  the deletion evidence.
- On exit, produce a deletion certificate that names the systems covered and the exceptions
  claimed, and get the exceptions approved rather than assumed.

## 6. Worked example

Calder Retail Group operates 640 grocery stores and delivers 2.4 million online orders a year.
It has three years of telematics from 340 delivery vehicles, supplied by a fleet-management
vendor, and a delivery record for every order containing the consignee name, address, delivery
window, and outcome.

The proposed use case, called "PromiseBench", is technically feasible and commercially agreed:

- Build a model that predicts delivery-window reliability by postcode district, vehicle, and
  time of day.
- Sell a monthly benchmark report to 40 consumer goods suppliers, at €12,000 per supplier per
  year, for a contract value of €480,000.
- Include, for each supplier, a comparison of its own delivery performance against the category
  average, at postcode-district level.

The supplier contracts are signed. The data engineering is scoped. Then legal review blocks it.
The walk-through below shows why, and what would make a version of it permissible.

### 6.1 Gate 1 — Is the data personal?

| Field | Personal data? | Reasoning |
|---|---|---|
| Consignee name | Yes | Identifies a natural person |
| Delivery address | Yes, in most cases | Identifies a household; a flat number plus name is direct identification |
| Postcode district | No on its own | Too coarse to identify; becomes a quasi-identifier in combination |
| Vehicle registration | Yes where linked to a driver | Identifies the driver through the fleet roster |
| Driver identifier | Yes | Direct identifier of an employee |
| Telematics speed and braking events | Yes | Relates to an identified driver |
| Delivery outcome code | No on its own | Becomes personal in combination with the above |

Result: the dataset is personal data, and the supplier-facing output may or may not be,
depending on its grain. The gate is passed only with the personal data acknowledged.

### 6.2 Gate 2 — What is the role?

Calder is the controller of the order and consignee data. The fleet vendor processes telematics
on Calder's behalf under a DPA and is a processor. Some of the telematics is also used by the
vendor for its own fleet-benchmarking product, which makes the vendor a controller for that
separate use — a fact that matters because it means the vendor cannot simply hand the data back
in a form that Calder may repurpose.

Result: Calder may direct the use of the telematics for its own purposes, but only within the
basis on which the data was collected.

### 6.3 Gate 3 — What was the original purpose?

The telematics was collected for route optimisation, insurance claim defence, and driver safety
management. The privacy notice said so. The works council agreement covering vehicle tracking
covers safety and routing, not commercial benchmarking.

The proposed purpose — commercial benchmarking sold to suppliers — is not route optimisation,
insurance defence, or safety management. It is a new purpose, and it is commercial.

Result: a compatibility assessment is required, and it is unlikely to succeed on the current
facts. The link between the purposes is weak, the context of collection was employment, the
consequences for drivers include performance comparison they did not expect, and the safeguards
proposed do not change the commercial nature of the use.

**Note:** this is the point at which the use case is usually described as "blocked by GDPR". It
is more accurate to say it is blocked by a mismatch between the original purpose and the new
one, which is a design problem with design solutions, not a prohibition.

### 6.4 Gate 4 — What does the licence permit?

Calder also buys a syndicated retail panel, used in the same benchmark. Three clauses matter:

- The panel licence permits internal business purposes and limited excerpting in reports about
  Calder's own products, with attribution.
- It prohibits resale of the data or of derivative works, and prohibits disclosure that is
  competitive with the panel provider.
- It prohibits use of the data to train machine-learning models.

The PromiseBench output is a derivative work sold to third parties. That is resale of a
derivative work on the face of the clause. The model would also be trained on panel data.

Result: two independent licence breaches, either of which is sufficient to stop the use case.

### 6.5 Gate 5 — Is the output personal data?

At postcode-district level, with a supplier's own volumes, the cells are small. Take a supplier
with 4,000 deliveries a year across 120 postcode districts. At district-week grain, the average
cell size is:

```text
4,000 deliveries / 52 weeks / 120 districts = 0.64 deliveries per district-week
```

Assuming deliveries are approximately Poisson-distributed across cells with mean 0.64, the
fraction of records that sit alone in their cell is `e^-0.64 = 0.527`, or about 53%. More than
half of that supplier's delivery records would be singling-out candidates at the proposed grain.

Result: the output is not reliably anonymous at the proposed grain. It would need coarser
geography, a minimum cell size, or suppression.

### 6.6 Gate 6 — Transfer and retention

The benchmark would be hosted in a cloud region in the United States, with support access from
India. Consignee data would be included in the training set, and supplier-facing outputs would
be generated there.

- The US region requires a transfer mechanism. If the cloud provider is DPF-certified, that
  covers the provider; it does not cover Calder's own support staff access from India, which
  needs its own mechanism.
- Retention was unspecified. The draft said "retain for the life of the product", which is not a
  retention period.

Result: two further gaps, both fixable, neither fixed.

### 6.7 The alternatives that would make it permissible

| Alternative | What changes | Cost and time | Residual risk | Does it preserve the commercial case? |
|---|---|---|---|---|
| Aggregate to region and quarter, minimum cell size 50 | Output no longer personal data; panel clause still binds | Low; 2–3 weeks of engineering | Panel resale still prohibited | Partly: less granular, lower price |
| Use only Calder's own data, no panel | Removes the licence breach | Low; 2 weeks | Purpose limitation on telematics remains | Partly: no market context |
| Renegotiate the panel licence for derived-output use | Removes the resale and training restrictions | High; 3–6 months, fee increase likely | Provider may refuse | Yes, if granted |
| Obtain a new basis for the telematics use | New notice, works council agreement, possibly consent | High; 4–6 months | Consent in employment is fragile | Yes, for the telematics component |
| Build the model on synthetic telematics, calibrate on aggregates | Removes personal data from training | Medium; €150k–€250k | Synthetic fidelity may be insufficient | Partly |
| Offer the benchmark as a service operated inside a clean room | Row-level data never leaves; only outputs released | Medium–high; €120k setup plus €60k a year | Clean room does not cure a missing basis | Yes, for suppliers who accept output-only access |
| Drop the supplier-facing product; use the model internally | Internal use fits the original purpose more comfortably | Low | Internal use still needs a basis and retention | No: removes the revenue |

The realistic path is a combination: aggregate output at a defensible grain, drop the panel from
the supplier-facing product, and keep the panel for internal category analysis under its
existing terms. That version is worth less than €480,000 and is buildable. The version that was
signed is worth €480,000 and is not.

### 6.8 What the analyst should have asked at the start

1. Which output fields could identify a person, and at what grain does that stop being true?
2. Is the panel data inside the model, inside the output, or both? The licence treats those
   differently.
3. Does the works council agreement cover commercial benchmarking, and who is the controller of
   each input?
4. What is the deletion plan for the model and the feature store, not just the source table?

None of these is a modelling question. All of them determine whether the model may exist.

## 7. Code, query, or calculation

This section is a decision table rather than code. Read it top to bottom and stop at the first
row that applies.

| # | Condition | Permitted action | Required record | Escalate to counsel |
|---:|---|---|---|---|
| 1 | Data is not personal, not confidential, and under an open licence with attribution only | Use, publish, and redistribute with attribution | Licence identifier and attribution text | No |
| 2 | Data is not personal but is commercially confidential under contract | Use within the contract's scope only | Clause reference for scope and redistribution | If the intended use is not clearly inside scope |
| 3 | Data is personal, you are the controller, and the purpose is unchanged | Process on the recorded basis, minimised | Basis code, ROPA entry, retention entry | No |
| 4 | Data is personal, you are the controller, and the purpose is new | Run the compatibility assessment first | Written purpose statement and assessment | Yes, before building |
| 5 | Data is personal and you are a processor | Act only on documented instructions | DPA reference and instruction record | Yes, if you want to use it for your own purpose |
| 6 | Data is personal and you are a joint controller | Agree the arrangement and the allocation of duties | Art. 26 arrangement | Yes |
| 7 | Data leaves the origin jurisdiction | Confirm adequacy, SCCs, or a derogation before transfer | Transfer record and TIA where SCCs apply | Yes, for any new destination |
| 8 | Output could identify a person at the planned grain | Coarsen, suppress small cells, or treat as personal data | k or minimum cell size, and the test used | Yes, if claiming anonymisation |
| 9 | Licence contains a model-training or derived-output restriction | Do not train or ship until the restriction is cleared | Clause reference and written clearance | Yes |
| 10 | None of the above is clear | Stop | Open question log | Yes |

Two rows carry most of the practical weight. Row 4 is the analyst's most common trap: a new
purpose that feels like the same work. Row 8 is the most common technical error: an aggregation
that is described as anonymous without a test.

## 8. Data, standards, and regulatory context

> **Note:** this section is educational. It describes regulatory structure as reported by the
> sources in section 16, as of October 2026. It is not legal advice, it does not create a
> lawyer–client relationship, and it should not be relied on for a decision about a specific
> processing activity. Regulatory positions on several points below are contested or in flux.
> Obtain qualified advice for any live question.

### 8.1 The GDPR principles that bind analytics

| Principle | Article | What it constrains in a supply chain dataset | Grade |
|---|---|---|---|
| Lawfulness, fairness, transparency | Art. 5(1)(a) | You need a basis, and the notice must describe the actual use | B |
| Purpose limitation | Art. 5(1)(b) | Reuse beyond the collected purpose requires compatibility | B |
| Data minimisation | Art. 5(1)(c) | Field-level necessity, not table-level availability | B |
| Accuracy | Art. 5(1)(d) | Wrong consignee or driver records must be correctable | B |
| Storage limitation | Art. 5(1)(e) | Retention periods per class, with deletion | B |
| Integrity and confidentiality | Art. 5(1)(f) | Access control and security proportionate to risk | B |
| Accountability | Art. 5(2) | You must be able to demonstrate the above | B |

The six lawful bases under Art. 6(1) are consent, contract, legal obligation, vital interests,
public task, and legitimate interests. For logistics analytics, contract and legitimate interest
do most of the work, and consent does less than practitioners expect. The Commission's guidance
illustrates contract necessity with a sales contract covering name, delivery address, and card
number — exactly the consignee data in a delivery record.

Legitimate interest is the most over-used basis. The EDPB's Guidelines 1/2024 set out three
cumulative conditions: a legitimate interest that is lawful, clearly articulated, real and
present; necessity, assessed together with data minimisation; and a balancing test covering
reasonable expectations and mitigating measures. The basis is neither a last resort nor a
default, and the assessment must be documented.

### 8.2 Purpose limitation and secondary use

| Question | Source of the test | Practical meaning | Grade |
|---|---|---|---|
| Is the new purpose compatible with the original? | Art. 6(4): link between purposes, context, nature, consequences, safeguards | A written assessment, not an assertion | B |
| Is it research or statistics? | Art. 5(1)(b) with Art. 89(1) safeguards | Presumption of compatibility in EU guidance, subject to safeguards | B |
| Does the presumption cover commercial benchmarking? | Commission and CNIL guidance, EDPS opinion | Generally no; research is not a synonym for analysis | B for the guidance; the application is contestable |
| Is the data subject likely to expect this use? | Recital 50 and balancing case law | Expectation is a factor, not a veto | B |

The Commission's guidance states that for statistics or scientific research the compatibility
test is not necessary, because those purposes are presumed compatible subject to Art. 89(1)
safeguards. The CNIL's 2024 guidance makes the same point while noting that a valid basis and
information duties still apply; the EDPS has cautioned that the presumption is not a blanket
authorisation. Treat it as a route that requires documentation, not as an exemption.

### 8.3 Controller, processor, and the chain

| Instrument | What it establishes | Consequence for an analytics team | Grade |
|---|---|---|---|
| Art. 28 GDPR | Processor acts on documented instructions | A processor's analytics may not serve the processor's own purposes | B |
| Art. 26 GDPR | Joint controllers agree their arrangement | Duties must be allocated in writing, and data subjects can exercise rights against either | B |
| EDPB Opinion 22/2024 | Controller must know the processor and sub-processor chain; verification scales with risk | You need a current list of every party that touches the data | B |
| Art. 30 GDPR | Records of processing | The analytics use must appear in the record, not only the operational use | B |
| Art. 35 GDPR | DPIA for high-risk processing | Large-scale location or performance monitoring is a candidate | B |

EDPB Opinion 22/2024, adopted in October 2024, is the most useful recent statement on the chain.
It holds that controllers must have the identity and processing description of all processors
and relevant sub-processors readily available, that the duty to verify "sufficient guarantees"
applies at every risk level with the depth scaling to risk, and that controllers remain
responsible for transfer mapping and transfer impact assessments even when a processor does the
exporting. For an analytics team, the practical consequence is that "the vendor handles
compliance" is not a defensible position.

### 8.4 Cross-border transfers

| Mechanism | When it applies | What it requires | Grade |
|---|---|---|---|
| Adequacy decision | Recipient is in a country or territory covered by a Commission decision | Confirm the decision covers the recipient and the data type | B |
| Standard contractual clauses | Most other transfers | Decision (EU) 2021/914 module; documented transfer impact assessment under Clause 14 | B |
| Derogation | Occasional, non-repetitive transfers, or explicit consent | Narrow; not suitable for a recurring analytics pipeline | B |
| EU–US Data Privacy Framework | US recipients certified under the DPF | Verify current certification on the official list | B |

The adequacy list as of late 2025 and early 2026 includes Andorra, Argentina, Brazil, Canada for
commercial organisations, the Faroe Islands, Guernsey, Israel, the Isle of Man, Japan, Jersey,
New Zealand, the Republic of Korea, Switzerland, the United Kingdom, the United States for
commercial organisations certified under the DPF, and Uruguay, plus the European Patent
Organisation. The UK decisions were renewed in December 2025, and the Korea arrangement became
mutual in September 2025.

The DPF survived its first judicial test. In *Latombe v Commission* (Case T-553/23), the General
Court dismissed the action against the DPF adequacy decision on 3 September 2025. An appeal was
lodged on 31 October 2025 as Case C-703/25 P and was pending at the time of writing. The
practical reading is that the DPF is in force and usable, with a live appeal that could change
that. Any transfer design that depends on it should record the dependency.

SCCs are not a signature. Clause 14 requires the parties to assess the destination country's
laws and practices, the specific circumstances of the transfer, and any supplementary measures,
and to document that assessment. The Commission's Q&A confirms the requirement and points to
the EDPB's Recommendations 01/2020 for the method. A signed SCC schedule with an empty Annex II
is a common audit finding.

**A cloud region is not a transfer mechanism.** Choosing Frankfurt removes the transfer for
storage; it does not remove the need for one if support engineers elsewhere can access the data.
Region is a risk-reduction decision, the mechanism is a legality decision, and both are needed.

### 8.5 The EU Data Act

Regulation (EU) 2023/2854 applies from 12 September 2025. Its connected-product provisions are
the part that matters for equipment telemetry.

| Provision | What it requires | Bearing on supply chain analytics | Grade |
|---|---|---|---|
| Art. 3 — transparency | Inform users what data a connected product generates, whether it is continuous, and how third parties can access it | Telematics and machine-data contracts must describe the data, not just the service | B |
| Art. 3 — access by design | Products placed on the market after 12 September 2026 must make data directly accessible in a structured, machine-readable format | New equipment should arrive with an export path | B |
| Art. 4 — user access | Data holders must give users the data, free of charge, at the same quality, continuously where technically feasible | Buyers of equipment gain a stronger claim to its operating data | B |
| Art. 5 — third-party sharing | Users may direct the data holder to share with a third party | A fleet operator can require the OEM to share with its analytics provider | B |
| Art. 6 — recipient obligations | Purpose limitation, no use to develop competing products, no onward sharing to gatekeepers | A telemetry aggregator's permitted uses are narrowed | B |
| Art. 9 — compensation | B2B data holders may claim reasonable compensation in some cases | "Free" access is free to the user, not necessarily to a commercial recipient | B |
| Chapter V — B2G | Public bodies may request data for exceptional need; trade-secret safeguards in Art. 19 | Relevant to regulators and to emergency planning, less to routine analytics | B |

Scope is limited to raw and pre-processed data, including necessary metadata. Inferred or derived
data is outside the access right — the commercially important boundary, because the OEM must
share the sensor readings, not the model built from them.

The B2G provisions are in flux. A February 2026 Council document on the Digital Omnibus
indicates that the non-emergency route in Art. 15(1)(b) would be deleted, focusing Chapter V on
public emergencies while Art. 19 trade-secret handling remains. Check the consolidated text.

### 8.6 The EU AI Act's data-governance obligations

Regulation (EU) 2024/1689, Article 10, applies to high-risk AI systems that use model training.
Its requirements are unusually specific about data provenance, and they overlap directly with
purpose limitation.

| Art. 10 requirement | What it asks for | Why an analytics team should care now | Grade |
|---|---|---|---|
| Design choices | Documented assumptions about what the data represents | Forces an explicit statement of what a feature means | B |
| Collection processes and original purpose | Origin of the data and, for personal data, the purpose of collection | Makes purpose limitation a documented design input | B |
| Preparation operations | Annotation, labelling, cleaning, updating, enrichment, aggregation | The preparation pipeline becomes auditable | B |
| Availability, quantity, suitability | An assessment of whether the dataset can support the intended use | Blocks "we have a lot of rows, therefore it is fine" | B |
| Bias examination and mitigation | Examination of biases affecting health, safety, fundamental rights, or prohibited discrimination, including feedback loops | Relevant to routing, credit, and service allocation models | B |
| Data gaps | Identification of gaps and how they are addressed | Makes missing populations visible | B |
| Representativeness and context | Relevant, sufficiently representative, complete for the purpose, and reflecting the intended setting | A model trained on one region may not transfer | B |

The timing is frequently misreported. The Act became generally applicable on 2 August 2026, but
the AI Omnibus, in force 27 July 2026, deferred the high-risk obligations: Annex III systems now
apply from 2 December 2027 and Annex I products from 2 August 2028. A forecasting or routing
system is not automatically high-risk; classification depends on the use. Article 10's
provenance requirements remain a good template for documenting data use even where the Act does
not yet bite.

### 8.7 California

| Instrument | Requirement | Bearing on supply chain data | Grade |
|---|---|---|---|
| CCPA as amended by CPRA | Notice, purpose limitation, consumer rights, service-provider contracts | Applies to California residents' data, including employees in some respects | B |
| CCPA regulations, risk assessments (Art. 10) | Assessment for selling or sharing, sensitive PI, and certain automated decision-making | A retail analytics programme may trigger this | B |
| CCPA regulations, automated decision-making (Art. 11) | Pre-use notice, access, opt-out, and appeal for qualifying decisions | Compliance date 1 January 2027 | B |
| CCPA regulations, cybersecurity audits (Art. 9) | Independent audit above thresholds, with board certification | Staggered from 2028 by revenue band | B |
| Service-provider contract terms | Restrictions on retaining, using, or disclosing personal information | Determines whether a vendor relationship is a "sale" | B |

The CPPA Board adopted the final package on 24 July 2025; it was filed on 22 September 2025 and
took effect on 1 January 2026. Risk-assessment obligations began the same day, with the first
attestation due 1 April 2028. ADMT compliance is due 1 January 2027, and audit certifications
run from 1 April 2028 for the largest revenue band to 1 April 2030 for the smallest.

### 8.8 Open licences

| Licence | Commercial use | Attribution | Share-alike | Grade |
|---|---|---|---|---|
| CC0 / public domain dedication | Yes | Not required | No | B |
| CC BY 4.0 | Yes | Required, with a licence link and an indication of changes | No | B |
| CC BY-SA 4.0 | Yes | Required | Yes, on adaptations | B |
| ODC-BY | Yes | Required | No | B |
| ODbL 1.0 | Yes | Required | Yes, on derivative databases; produced works are not required to be ODbL | B |
| Non-commercial variants | No | Required | Varies | B |

The ODbL distinction between a derivative database and a produced work is the one that trips up
analytics teams. A chart built from an ODbL database is a produced work and need not carry the
ODbL; a new database built by joining and restructuring it is a derivative database and must.
Combining an ODbL source with a proprietary source can therefore create an obligation the
proprietary licence forbids — a conflict with no clean resolution except separation.

## 9. Industry practice and case evidence

### 9.1 What syndicated licences actually say

Two current licence texts illustrate the pattern. Both are contracts, so the grade is B for the
text as published and C for any interpretation of how a clause applies to a specific use.

Circana's European master terms describe a perpetual, non-exclusive, non-transferable,
non-sublicensable licence for internal business purposes. Limited excerpts are permitted in the
client's own reports and analyses, disclosed to third parties only as reasonably necessary to
market the client's products, and not in a quantity or quality that would be separately
marketable by Circana. The terms prohibit public dissemination without prior written consent,
resale of the data or derivative works, use in a manner competitive with Circana, reverse
engineering of tools, and removal of proprietary notices. Disclosure to a retailer contributing
to the dataset is permitted with a written confidentiality agreement, and only for that
retailer's account-level data.

NielsenIQ's general licence terms for syndicated services, effective 15 January 2026, grant a
non-exclusive, non-transferable, revocable licence without sublicensing, for use solely within
the client's own business organisation. Limited excerpts may go to advertising agencies,
retailers, brokers, distributors, and wholesalers to support marketing the client's products,
provided they do not have independent commercial value. Other third-party disclosure is limited
to consultants who have signed NielsenIQ's third-party access agreement. The terms restrict
generative AI use to internal summarisation, querying, and translation; prohibit training
machine-learning or AI models on the data; require secure enterprise environments; and require a
data-sharing agreement or data-warehouse exhibit for third-party data sharing.

Three lessons follow, and they generalise beyond these two vendors:

1. **"Internal business purposes" excludes the customer-facing deliverable** in most analytics
   products. A benchmark sold to suppliers is not internal, however much of the analysis is yours.
2. **The AI clause is now a first-class restriction.** It can prohibit the model even where the
   data use would otherwise be permitted.
3. **Derived output is usually addressed, and restrictively.** The clause prevents the licensee
   from recreating the product, which is exactly what a good benchmark does.

### 9.2 Enforcement in delivery and telematics

| Case | Authority and date | Facts | Outcome | Grade |
|---|---|---|---|---|
| Autotrasporti Cuccu Riccardo S.r.l. | Garante (Italy), decision 16 January 2025 | GPS and telematics on about 50 lorries, continuous tracking including rest breaks, 180-day retention, incomplete and contradictory notice, failure to answer an information request | €50,000 fine; notice rewritten; retention and notice practices ordered to change | B |
| Pioneer Hi-Bred Italia Sementi | Garante (Italy), decision 18 December 2025 | Telematics devices scoring driving behaviour on company cars including private trips, 13-month retention, no works agreement, no processing agreements with group companies | €120,000 fine; deletion of private-trip data ordered | B |
| Foodinho / Glovo | Garante (Italy), November 2024 | Rider geolocation shared with third parties without knowledge; facial recognition | €5 million fine | B |

The pattern across the three is consistent: continuous location tracking, retention longer than
the purpose requires, notices that do not describe the processing, and processor chains never
papered. None of these companies was fined for building a model; they were fined for the
conditions under which the data was held.

The Cuccu decision is the most instructive for a data team because retention is the headline:
180 days on continuous GPS was excessive. Any feature store that keeps raw telematics
indefinitely inherits that finding.

### 9.3 Contractual data-sharing in retail supply chains

New Zealand's Grocery Supply Code 2025, determined by the Commerce Commission on 16 October 2025
with commencement on 1 May 2026, contains a confidentiality provision (clause 26) that bars a
retailer from using supplier confidential information — product development, proposed
promotions, pricing — for any purpose other than the one for which it was disclosed. It limits
internal access to staff who need it, requires the retailer to establish and monitor compliance
systems, and restricts demands for commercially sensitive information during price negotiations.

This is a regulatory floor for one jurisdiction, not a global standard. Its significance is
structural: it makes the retailer a steward of supplier data, not merely a recipient. An
analytics team building a supplier-facing product from supplier-submitted data is inside the
scope of that kind of clause, and the compliance evidence is a system, not a policy.

Outside regulated grocery codes, the equivalent obligations live in the supplier agreement. A
typical clause defines confidential information broadly, permits use only for the purpose of the
agreement, requires confidentiality flow-down to subcontractors, and carves out aggregated or
derived data — often without defining what "aggregated" means. That undefined carve-out is where
most disputes would arise.

### 9.4 Data clean rooms and the limits of technical controls

Data clean rooms are marketed as a solution to the sharing problem: contributors put data in,
outputs come out, row-level data does not move. The IAB's April 2025 guidance on clean rooms
under US state privacy law makes the necessary correction. A clean room processes personal
information; hashing and salting are pseudonymisation, not anonymisation. The data contributors,
not the clean-room provider, are typically the business or controller. Disclosing personal
information to a clean room can be a "sale" absent a valid service-provider relationship, and
record-level outputs such as audience building are typically treated as a sale where aggregate
reporting is not.

The practical conclusion is that a clean room changes the architecture and the contract, not
the lawful basis. It is a control, not a permission. IAB Europe's late-2025 blueprint makes the
same point: clean rooms do not inherently guarantee GDPR compliance.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating the licence as a download | Nobody can name the clause that permits the use | Procurement forwards a PDF, no one reads the scope | Redistribution breach discovered after delivery | Clause register per source, reviewed before build |
| Joining restricted data without changing handling | Two teams disagree about whether the output is restricted | Union intuition instead of intersection | Restricted data propagates into dashboards | Tag rows with a permission class at ingest |
| Purpose creep in a feature store | A new model reads a table collected for another purpose | Convenience of existing data | Secondary use without a basis | Purpose tag per table; access review on new consumers |
| "Anonymous" aggregation with small cells | Output contains rows with one delivery | No minimum cell size | Re-identification risk, and a false claim | Minimum cell size and a documented test |
| Processor building its own product | Vendor offers a cross-client benchmark | Processor determines a new purpose | Vendor is a controller without a basis; client exposed | Contractual prohibition plus audit right |
| Consent used for employee data | Works council not consulted; consent form in the onboarding pack | Imbalance makes consent invalid | Basis fails; monitoring is unlawful | Works agreement and legitimate-interest assessment |
| Retention set to "life of the product" | No deletion job exists | Storage limitation ignored | Regulator finding, and growing breach surface | Retention schedule per class with automated deletion |
| SCC signed with empty annexes | Annex II lists no measures | TIA never performed | Transfer is undocumented and indefensible | TIA per transfer, reviewed annually |
| Cloud region mistaken for a transfer mechanism | "Data stays in Frankfurt" is the whole answer | Support access crosses borders | Unlawful transfer through the back door | Map access paths, not only storage |
| Clean room assumed to cure the basis | "It is in a clean room" ends the discussion | Technical control confused with legal permission | Processing without a basis | Basis review before architecture |
| Open licence share-alike ignored | ODbL source joined into a proprietary product | Copyleft obligation attaches to the derivative database | Product must be relicensed or rebuilt | Licence compatibility check at design |
| Model trained on licensed data | Training set assembled from every available table | AI clause unnoticed | Model may have to be retrained or withdrawn | Data provenance record per training run |

Two of these deserve a note.

**Purpose creep in a feature store** most often originates with the analyst rather than the
lawyer. The table exists, the join is easy, and the original notice said "service improvement";
the new use is "commercial benchmarking". Nobody decided to break the rule; the path of least
resistance did. The mitigation is technical — purpose tags and access review — because a policy
alone will not survive a deadline.

**Model trained on licensed data** is the newest and least understood. A model is not a copy of
its training data, which is why teams assume the licence does not reach it. But the licence is a
contract and can prohibit the act of training regardless of what the model contains. Where the
clause exists, the model is in scope; where it does not, treat the position as an open question.

## 11. Assumptions and limits

> **Note:** this topic is educational and is not legal advice. It describes structures and
> sources, not conclusions for your situation. Several points below are genuinely contested:
> the scope of "compatible" further processing, the boundary between pseudonymisation and
> anonymisation at small cell sizes, whether a model trained on licensed data is a derivative
> work, and the treatment of B2B contact data. Do not treat any statement here as a settled
> legal conclusion. Obtain qualified advice before acting.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The dataset's personal-data status can be assessed from its fields | Classification drives every later obligation | Free text, images, and location traces hide personal data | Field-level review, including unstructured fields |
| The contract describes the actual processing | The licence is the operative permission | Practice diverges from the papered role | Compare the DPA to the system diagram and the vendor's product |
| The original purpose was recorded at collection | Purpose limitation depends on a baseline | Notice was generic, or the basis was reconstructed | Read the notice version in force at collection |
| Aggregation at the chosen grain removes identifiability | Outputs are shared on that basis | Small cells, rare attributes, or auxiliary data enable linkage | k-anonymity or comparable test, with the quasi-identifiers stated |
| Adequacy status is stable over the project | Transfer design assumes continuity | Court challenges, political change, decision suspension | Check the current decision and any pending litigation |
| The vendor's certification covers the transfer | DPF reliance is common | The entity, the data type, or the onward transfer is out of scope | Verify the certification entry and the recipient entity |
| Retention periods are technically enforceable | A schedule without deletion is a statement of intent | Backups, caches, and vendor copies | Trace a deletion request end to end |
| Deletion removes the data's influence | Erasure requests assume it does | Models, embeddings, and aggregates retain it | Test whether retraining is required and cost it |
| The licence's AI clause is the only AI constraint | Teams read one clause | Separate addenda, vendor terms, and open-licence conditions | Full contract review, not a summary |
| Open data is unrestricted | It is free to obtain | Attribution, share-alike, non-commercial terms | Read the licence identifier, not the portal's description |
| The counterparty has authority to grant the licence | You rely on the grant | The counterparty is a processor, or lacks rights | Confirm the chain of title to the data |

Four limits of this topic are worth stating plainly.

**The regulatory position changes faster than any curriculum.** The Data Act's B2G provisions,
the AI Act's timeline, the DPF litigation, and the CCPA regulations were all moving during 2025
and 2026. Every date in section 8 should be checked against the primary source before it is used
in a decision.

**Jurisdiction is not optional.** This topic leans on EU and California instruments because they
are the most developed and the most documented. A supply chain that operates in Brazil, India,
Japan, or the Gulf faces different rules, and some of them are stricter. The mental model
transfers; the specifics do not.

**Contracts are private and vary.** No summary of a licence substitutes for the licence. Two
vendors in the same market can have materially different terms, and the same vendor can have
different terms across regions and product lines.

**This is not a substitute for counsel.** The topic teaches you to ask the right question and to
recognise when an answer is needed. It does not qualify you to give the answer, and an analyst
who treats it as a qualification is more dangerous than one who does not.

## 12. Exercises

### Exercise 1 — Classify four sources (recall and application, 40 minutes)

Take four sources you have used or can describe: an internal warehouse table, a retailer's EDI
feed, a purchased market panel, and a national statistics dataset.

**Deliverable:** a one-page table with pathway, money cost, latency, dominant obligation, and the
clause or rule most likely to constrain an analytics use.
**Acceptance criteria:** all four pathways represented; each obligation names an instrument
(contract clause, article, or licence identifier); latency stated in a unit.

### Exercise 2 — Cost the pathways (calculation, 35 minutes)

A team needs category share data for three years. Option A is a panel subscription at €180,000
per year with a four-week publication lag. Option B is a partner exchange with 40 retailers at
€15,000 integration each plus €2,000 per month per connection, with a one-day lag but no market
total. Option C is open data plus 200 analyst-hours per year at €95 per hour.

**Deliverable:** three-year total cost of ownership for each option, cost per usable month, and a
recommendation with the decision it supports.
**Acceptance criteria:** all costs included; the lag converted into a usable-unit penalty where
relevant; the recommendation names the decision and states what the cheapest option cannot
answer.

### Exercise 3 — Read a licence (critique, 50 minutes)

Obtain any publicly available data licence or terms of use — an open licence, a vendor's online
terms, or a dataset's access conditions.

**Deliverable:** a clause map covering scope of use, redistribution, combination, reverse
engineering, model training, audit, and deletion; plus three analytics uses that are permitted
and three that are not.
**Acceptance criteria:** each finding cites a clause or section; at least one permitted use is
non-obvious; at least one prohibited use would be technically easy and commercially tempting.

### Exercise 4 — Controller or processor (application, 40 minutes)

A retailer contracts a 3PL to store and deliver online orders. The 3PL also aggregates its
clients' delivery data to publish a quarterly "last-mile performance index".

**Deliverable:** a role map for each processing activity, with the basis for each classification
and the contractual change that would make the index lawful or unlawful.
**Acceptance criteria:** the map separates the delivery processing from the index; identifies the
moment the 3PL determines a purpose; states what the retailer's DPA should say; does not assert a
legal conclusion without naming the reasoning.

### Exercise 5 — Purpose-limitation assessment (design, 55 minutes)

A grocery chain collected loyalty-card data for points and personalised offers. It now wants to
use the data to forecast store-level demand and to sell aggregated demand signals to suppliers.

**Deliverable:** a compatibility assessment for each of the two new purposes using the five
Art. 6(4) factors, with a recommendation and the safeguards that would change the answer.
**Acceptance criteria:** both purposes assessed separately; each factor addressed with a
specific fact; the supplier-facing purpose is not treated as research; the safeguards are
concrete.

### Exercise 6 — Transfer design (design, 45 minutes)

A European manufacturer's analytics platform runs in a US cloud region. Support engineers in
India and the Philippines have production access. The vendor is DPF-certified.

**Deliverable:** a transfer map with every access path, the mechanism for each, and the
documentation required.
**Acceptance criteria:** storage and access separated; at least three distinct transfer paths
identified; the DPF's scope stated correctly; the TIA requirement identified where SCCs apply.

### Exercise 7 — Data Act impact (application, 45 minutes)

A fleet buys 200 connected trucks from an OEM. The OEM's telematics platform is the only way to
get sensor data, and the OEM sells a benchmarking product built on it.

**Deliverable:** a memorandum on what the buyer can require under the Data Act, what it cannot,
and what the contract should be amended to say.
**Acceptance criteria:** distinguishes raw and pre-processed data from derived data; identifies
the relevant articles; notes the access-by-design date; states at least one point that is
uncertain or in flux.

### Exercise 8 — Deletion plan (design, 50 minutes)

A contract terminates and requires deletion of all data and derived works within 30 days.

**Deliverable:** a deletion plan naming every system and artefact class, the action for each, the
exceptions claimed, and the evidence produced.
**Acceptance criteria:** at least six artefact classes, including backups, feature store, model
artefacts, and vendor environments; each exception justified; the certificate format specified;
one class identified where deletion is technically infeasible without retraining.

### Exercise 9 — Redesign the blocked use case (design, 60 minutes)

Return to the PromiseBench case in section 6. Legal has blocked the supplier-facing benchmark.

**Deliverable:** two redesign options with cost, time, residual risk, and the commercial value
retained; plus a recommendation.
**Acceptance criteria:** each option names the specific gate it fixes; no option relies on
"anonymising" without a test; the commercial value is quantified against the €480,000 baseline;
the recommendation states what would have to be true for the full version to proceed.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
Option A (panel):
  subscription      180,000 x 3            = 540,000
  integration       assume 25,000 one-off  =  25,000
  total                                     = 565,000
  usable months     36, minus 4-week lag    = 35.1 effective months
  cost per month    565,000 / 35.1          =  16,097

Option B (partner exchange, 40 retailers):
  integration       15,000 x 40            = 600,000
  connection fees   2,000 x 40 x 36        = 2,880,000
  total                                     = 3,480,000
  cost per month    3,480,000 / 36          =  96,667

Option C (open data):
  analyst time      200 x 95 x 3           = 57,000
  total                                     = 57,000
  cost per month    57,000 / 36             =   1,583
```

Option A is the only one that produces a market total, and it is roughly ten times the cost of
Option C per month. Option B is the most expensive by a wide margin and still does not produce a
market total, because 40 retailers are not the market. A common wrong answer compares the
subscription line only for Option A against the full cost of Option B and concludes that the
panel is cheap; the correct comparison includes integration and the lag penalty, and asks what
each option can answer.

If the panel is used only internally, its per-month cost is the right comparison. If the output
is sold, the licence restriction changes the calculation entirely, and the correct conclusion
may be that no option is available at the desired grain.

**Exercise 5 model answer outline.**

For demand forecasting, the compatibility case is strong. The link between personalised offers and
demand forecasting is the same commercial relationship, the data is the same transaction history,
the consequences for customers are modest, and pseudonymisation plus access control are
proportionate safeguards. A reasonable conclusion is that the purpose is compatible, with
documentation.

For selling aggregated demand signals to suppliers, the case is weak. The link is commercial
rather than operational, the context is a loyalty relationship the customer did not associate
with supplier benchmarking, the consequences include indirect commercial effects the customer
would not expect, and aggregation at a coarse grain helps but does not by itself address the
relational shift. The safeguard that would change the answer is a separate basis with clear
notice at collection — for example, a notice that explicitly describes supplier-facing
aggregated insights — rather than a retrospective justification.

A submission that treats both purposes as "analytics" and gives one answer should be marked
down. The exercise is testing whether the student separates the purposes before assessing them.

**Exercise 6 model answer.**

Transfer paths: (1) storage in the US region — DPF if the provider entity is certified, otherwise
SCCs Module Two; (2) support access from India — SCCs Module Two with India as the destination,
plus a TIA; (3) support access from the Philippines — SCCs Module Two with a separate TIA,
because the assessment is destination-specific; (4) any onward transfer by the vendor to a
sub-processor — covered only if the sub-processor is listed and the same mechanism extends to it.
The DPF certification covers the certified entity for the specified data types; it does not
cover the customer's own staff access from a third country. Four paths, at least three
mechanisms, and one common failure: a TIA written once for "the vendor" and applied to every
destination.

**Exercise 7 model answer.**

The buyer can require transparency about what data the truck generates (Art. 3), access to the
raw and pre-processed data free of charge and at the same quality (Art. 4), and can direct the
OEM to share that data with a third party such as an analytics provider (Art. 5). The recipient
is bound by purpose limitation and a prohibition on using the data to develop competing products
(Art. 6). The buyer cannot require the OEM's derived benchmarking product, because inferred and
derived data fall outside the access right. Access by design applies to products placed on the
market after 12 September 2026, so an existing fleet relies on the access right rather than the
design obligation. The uncertain point is compensation: Art. 9 permits reasonable compensation
in B2B cases, and the boundary between free user access and compensated commercial sharing is
not settled in practice. A strong answer says so rather than picking a number.

**Exercise 8 rubric.**

A strong plan lists: source warehouse tables; the feature store; notebook and sandbox copies;
exported reports and extracts; the vendor's environment; backup generations; model artefacts
including weights, embeddings, and evaluation sets; and any downstream dashboard extracts. For
each it states the action, the owner, the date, and the evidence. It claims an exception only
where the contract permits one, and it identifies the model artefact as the hard case: deleting
the training data does not remove its influence, so either the model is retrained without it or
the contract must permit retention. A weak plan lists "the database" and "the backups" and
asserts completion.

**Exercise 9 rubric.**

A strong answer fixes specific gates. Coarsening the output to region and quarter with a minimum
cell size of 50 addresses gate 5 and part of gate 3, but not gate 4, because the panel licence
still prohibits resale of a derivative work. Removing the panel addresses gate 4 but weakens the
product. Renegotiating the panel licence addresses gate 4 directly and may take six months.
A new basis for telematics addresses gate 3 and requires a works agreement. A clean room
addresses the architecture and the supplier access model but not the basis. The strongest answer
combines two or three, quantifies the retained value, and states the condition for the full
version: a written licence variation covering derived outputs and model training, plus a works
agreement that names commercial benchmarking.

**A note on judging legal reasoning.** Do not grade for the conclusion. Grade for whether the
student names the right instrument, applies the right test, separates settled from contested
points, and states what would change the answer. Better reasoning beats the expected answer.

## 14. Interview questions

**1. "What are the ways a company can get supply chain data, and how do they differ?"**

A strong answer gives the four pathways and compares them on cost, latency, and obligation, not
just on source type. It notes that the pathway determines who you negotiate with and that the
legal work often exceeds the integration work. A weak answer lists "internal systems and external
data" and stops.

**2. "Why does the licence matter if the data is already in our warehouse?"**

A strong answer explains that the licence is a contract that travels with the data and restricts
use regardless of physical possession. It gives a concrete example: internal-use-only terms, or a
model-training prohibition. A weak answer says "legal handles that", which is the answer of
someone who has not shipped a data product.

**3. "Is a delivery address personal data?"**

A strong answer says yes where it identifies a household or a person, notes that contract
necessity usually covers the delivery itself, and distinguishes the delivery purpose from
analytics reuse. It mentions that postcode alone is usually not personal data but becomes a
quasi-identifier in combination. A weak answer says "it is business data" or "it is anonymised
because we removed the name", which confuses a field with a dataset.

**4. "A 3PL offers you a benchmark built from its clients' data. What do you ask?"**

A strong answer asks who the controller is for each input, whether the 3PL had a basis to
repurpose data it processed for others, whether the clients consented or contracted for that
use, and whether the aggregation is genuinely anonymous. It notes that the 3PL may have become a
controller without a basis. A weak answer asks only about price and coverage.

**5. "How would you decide whether a new use of existing data is allowed?"**

A strong answer runs purpose limitation first, then basis, then licence, then output grain, then
transfer. It states that a new commercial purpose is not automatically compatible and that
research and statistics are a special case with safeguards. A weak answer says "if it is
internal, it is fine".

**6. "What is the difference between a controller and a processor, and why does it matter to an
analyst?"**

A strong answer defines both by who determines purpose, gives a retailer–carrier example, and
states the consequence: a processor's analytics must serve the controller's purposes, so a
cross-client product is out of scope. A weak answer describes data storage or system ownership.

**7. "Our data sits in a Frankfurt region. Are we compliant on transfers?"**

A strong answer says region is not a mechanism, maps support and access paths, and identifies
the mechanism for each destination. It mentions SCCs and the TIA requirement, and DPF scope
limits. A weak answer says yes because the data is in the EU.

**8. "What does the EU Data Act change for equipment telemetry?"**

A strong answer describes user access, third-party sharing at the user's direction, recipient
obligations, the exclusion of derived data, and the access-by-design date. It notes that the
buyer's right is to raw and pre-processed data, not to the OEM's model. A weak answer says it
"forces data sharing" without scope.

**9. "How would you anonymise a delivery dataset for a supplier benchmark?"**

A strong answer states the quasi-identifiers, chooses a grain and a minimum cell size, tests the
result, and acknowledges that anonymisation is a property of the dataset plus its context, not a
label. It mentions that removing names is not enough. A weak answer says "hash the customer ID".

**10. "What clauses do you look for in a data-sharing agreement?"**

A strong answer names scope of use, redistribution and derived output, combination, model
training, audit, sub-processor approval, deletion, and change of control. It explains what each
does to a project. A weak answer names confidentiality and term.

**11. "When would you escalate to legal, and what would you bring them?"**

A strong answer describes a threshold: new purpose, new jurisdiction, new output audience, or an
ambiguous licence clause. It brings the purpose statement, the data map, the clause references,
and the proposed safeguards. A weak answer says "whenever there is personal data", which
escalates everything and therefore nothing.

**12. "Can we train a model on purchased panel data?"**

A strong answer says it depends on the licence, that many current terms prohibit it explicitly,
and that the absence of a clause is not permission. It notes that a model is not a copy of the
data but the contract can still reach the act of training. A weak answer says yes because the
model is a new work, which is a copyright argument applied to a contract question.

## 15. Key terms

- **Access pathway** — direct, partner, commercial, or open; determines cost, latency, and
  obligation.
- **Permission envelope** — the five constraints that must all hold: basis, contract, transfer,
  retention, security.
- **Personal data** — information relating to an identified or identifiable natural person.
- **Controller** — the party that determines the purposes and means of processing.
- **Processor** — a party processing on the controller's behalf and instructions.
- **Joint controllers** — parties that jointly determine purposes and means.
- **DPA** — data-processing agreement; the Art. 28 contract governing a processor.
- **Lawful basis** — the Art. 6 ground that permits processing.
- **Legitimate interest** — a basis requiring a documented three-part test.
- **Purpose limitation** — use only for the collected or a compatible purpose.
- **Compatible further use** — a new purpose passing the Art. 6(4) assessment.
- **Data minimisation** — adequate, relevant, and necessary data only.
- **Storage limitation** — retention no longer than necessary.
- **Retention schedule** — defined period and deletion trigger per data class.
- **Pseudonymisation** — reversible de-identification; still personal data.
- **Anonymisation** — irreversible removal of identifiability; requires a test.
- **k-anonymity** — every record indistinguishable from k−1 others on quasi-identifiers.
- **Adequacy decision** — a Commission finding of adequate protection in a country.
- **SCCs** — standard contractual clauses, Decision (EU) 2021/914.
- **TIA** — transfer impact assessment, required under SCC Clause 14.
- **DPF** — EU–US Data Privacy Framework for certified US organisations.
- **Syndicated data** — panel product sold to multiple subscribers on standard terms.
- **Redistribution** — supplying data or derived output to a third party.
- **Derivative work** — an output substantially derived from licensed data.
- **Model-training restriction** — a licence clause limiting use to train models.
- **Data clean room** — output-only environment; a control, not a permission.
- **Open licence** — reuse permitted with limited conditions such as attribution.
- **ODbL** — attribution plus share-alike for derivative databases.
- **Connected product** — a product that generates data about its use, under the Data Act.
- **Exceptional need** — the B2G condition for compelled sharing under the Data Act.

## 16. Sources

**Peer-reviewed and academic**

- De Noyette, Stähler, and Margoni (2025). "Data Secrets: The Data Act's New Trade Secrets
  Framework." *IIC — International Review of Intellectual Property and Competition Law*.
  DOI: 10.1007/s40319-025-01601-9. Grade A. Analysis of the Data Act's trade-secret regime and
  the ex-ante role of competent authorities.
- Mylly, U.-M. (2024). "Trade Secrets and the Data Act." *IIC — International Review of
  Intellectual Property and Competition Law*. DOI: 10.1007/s40319-024-01432-0. Grade A.
  Frames the Data Act as a compulsory-licensing scheme and examines over-claiming of trade
  secrets.
- European Data Protection Supervisor (2020). *A Preliminary Opinion on data protection and
  scientific research*. Grade A for the analysis; note that it is an opinion, not a binding
  instrument.

**Regulatory and official**

- Regulation (EU) 2016/679 (General Data Protection Regulation), Arts. 5, 6, 26, 28, 30, 35, 44–49.
  Grade B.
- European Data Protection Board (2024). *Guidelines 1/2024 on processing of personal data based
  on Article 6(1)(f) GDPR*. Adopted 8 October 2024. Grade B.
- European Data Protection Board (2024). *Opinion 22/2024 on certain obligations following from
  the reliance on processor(s) and sub-processor(s)*. Adopted 7 October 2024. Grade B.
- European Commission. "Can we use data for another purpose?" Guidance on Art. 6(4) and
  compatible further processing. Grade B.
- European Commission. "Legal grounds for processing data" and "Principles of the GDPR."
  Grade B.
- CNIL (2024). Guidance on ensuring lawfulness in the re-use of data, including AI training
  datasets. Grade B.
- Commission Implementing Decision (EU) 2021/914 on standard contractual clauses, Module Two, and
  the Commission's Q&A on the SCCs. Grade B.
- Regulation (EU) 2023/2854 (Data Act), Arts. 3–9, 14–22. Grade B.
- Regulation (EU) 2024/1689 (AI Act), Art. 10. Grade B.
- European Commission, "AI Omnibus enters into force," 27 July 2026; Council of the EU, AI Act
  timeline. Grade B.
- European Commission, "Adequacy decisions" page, including the December 2025 UK renewal and the
  September 2025 Korea arrangement. Grade B.
- General Court, Case T-553/23, *Latombe v Commission*, judgment of 3 September 2025,
  ECLI:EU:T:2025:831; appeal C-703/25 P lodged 31 October 2025. Grade B.
- California Privacy Protection Agency, CCPA Updates regulations page; final package adopted
  24 July 2025, filed 22 September 2025, effective 1 January 2026. Grade B.
- Garante per la protezione dei dati personali (Italy), decision of 16 January 2025 concerning
  Autotrasporti Cuccu Riccardo S.r.l. Grade B.
- Garante per la protezione dei dati personali (Italy), decision of 18 December 2025 concerning
  Pioneer Hi-Bred Italia Sementi. Grade B.
- New Zealand Commerce Commission, *Grocery Supply Code 2025*, determined 16 October 2025,
  clause 26. Grade B.
- Open Data Commons, ODbL 1.0 text; Creative Commons, CC BY 4.0 and CC BY-SA 4.0 deeds.
  Grade B.

**Company and contract sources**

- Circana, Master Terms and Conditions (Europe). Grade B for the text; C for any interpretation
  of scope.
- NielsenIQ, General License Terms for Syndicated Strategic Analytics & Insights Services,
  effective 15 January 2026. Grade B for the text; C for interpretation.
- NielsenIQ, "NIQ Launches Global Data Clean Room on Snowflake," 2 October 2025. Grade C.

**Industry and analyst**

- IAB (2025). *Data Clean Rooms: A U.S. State Privacy Law Perspective*, April 2025. Grade C.
- IAB Europe (2025). *Blueprint for Data Collaboration in Commerce Media*. Grade C.
- Open Data Institute, *Reuser's Guide to Open Data Licensing*. Grade C.
- Various law-firm advisories on the Data Act and connected products, 2025. Grade C; used for
  orientation only.

**Press**

- Reuters, "EU court backs latest data transfer deal agreed by US and EU," 3 September 2025.
  Grade E; corroborates the judgment reported at Grade B.
- Reuters, "Italy watchdog fines Foodinho 5 million euros for rider data breaches,"
  22 November 2024. Grade E; corroborates the Garante decision.

**Note on citations.** Regulatory dates and scopes are reported as the primary sources state
them and are not independently verified. Several were in motion during 2025 and 2026. Verify
against the authority before relying on any. Contract terms are summarised, not quoted in full.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_data-movement.md`](06_data-movement.md)
- Next topic: [`08_data-quality-dimensions.md`](08_data-quality-dimensions.md)
- Related: [`10_source-selection-trust.md`](10_source-selection-trust.md)
- Foundation: [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
