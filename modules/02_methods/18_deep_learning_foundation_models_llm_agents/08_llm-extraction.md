# LLMs for Extraction and Supplier Communication

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic spreadsheet arithmetic; concepts are introduced here  
> Version 2026.5

## 1. Why this topic exists

A buyer receives a supplier email that says a shipment will be late.
A planner needs the affected purchase order, item, quantity, promised date,
and the date proposed in the message before deciding whether to expedite.
A model can propose those fields, but a plausible answer is not evidence that
it read the right attachment or interpreted a conditional promise correctly.
The business decision remains with an authorized person.

Contract teams face a related problem at a different timescale.
A clause about notice or liability may be scattered across a signed agreement,
an amendment, and an incorporated schedule.
A wrong extraction can cause an avoidable missed deadline or a false claim
that a supplier accepted a commitment.
A clause locator is not a legal opinion and the latest file is not necessarily
the controlling agreement.

A large language model (LLM) generates text from patterns learned during
training and conditioned on supplied instructions and context.
It is not a database of verified facts and it may omit, invent, or misstate
material even when its response is fluent.
The appropriate product is an evidence-linked candidate record with an
abstention path, followed by risk-based verification.

This unit connects document handling, field definitions, human review,
quality measurement, vendor controls, and economics.
It does not assume Module 06: labels, sampling, precision, recall, and
uncertainty are defined before use.
The illustrative organization and numbers below are fictional.
They are not a claim about a real vendor's performance.

## 2. Learning outcomes

After studying this topic, you can:

1. Separate source evidence, a model proposal, a reviewer decision, and an authorized operational action.
   Exercise 1 tests those boundaries.
2. Design a contract and supplier-email extraction schema with evidence and abstention.
   Exercise 2 asks for a field specification.
3. Resolve dates, quantities, document versions, and conditional language without inventing missing facts.
   Exercise 3 tests a difficult email.
4. Compute precision, recall, review burden, and an annual business case.
   Exercises 4 and 5 require the arithmetic and its interpretation.
5. Design a representative held-out evaluation and a release decision.
   Exercise 6 checks sampling and subgroup risk.
6. Specify secure processing, human verification, and incident controls.
   Exercises 7 and 8 require an operational design.
7. Assess a vendor's data, security, and change-management claims against evidence.
   Exercise 8 uses an acceptance rubric.

A valid outcome may be to keep the manual process.
A pilot decision is not authorization to send messages or alter purchase orders.

## 3. The core mental model

```text
Inbound email or executed document
          |
          v
Identity + version + access checks -----> quarantine / ask for missing source
          |
          v
Text and layout extraction + immutable source locator
          |
          v
LLM proposes typed fields + exact supporting passages + abstentions
          |
          v
Deterministic schema, citation and business-rule checks
          |
          v
Human sees source in context and accepts / corrects / escalates
          |
          v
Authorized workflow action, separate approval and audit event
          |
          v
Outcome, sampled audits, vendor/change monitoring and rollback
```

The model's proposed value is a hypothesis about the document.
The source is evidence; the reviewed record is a human decision; a purchase
order change or outbound email is a separate controlled action.
A readable citation is necessary but not sufficient: the passage might be
negated, superseded, conditional, attributed to the wrong party, or irrelevant.

A forecast is an estimate of a future outcome, not a supplier commitment.
A plan is the buyer's intended action; a target is a desired result.
A commitment is a promise authorized by the appropriate party and governed
by the actual agreement.
An email that says “we expect” is not automatically a commitment, and an
extracted delivery date must not silently become one.

The record must preserve what was known at the time of review.
If the supplier later corrects a date, create a new version rather than
rewriting the historical decision.
The reviewer needs the full surrounding context, not just a highlighted phrase.
For high-impact ambiguity, escalation to procurement or legal is a valid result.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| LLM | Generative model that predicts text conditioned on context | Treating fluency as factual accuracy |
| Extraction | Mapping source content to a defined field | Asking for interpretation without specifying the field |
| Source document | Specific email, attachment, or agreement version | Assuming a file name proves authority |
| Field | Named item with type, unit, and semantics | Mixing requested and committed dates |
| Evidence span | Exact source passage and stable location | Treating any relevant-looking passage as proof |
| Citation | Reference linking an output to source and location | Accepting a reference without checking context |
| Abstention | Explicit no-value or escalation when evidence is inadequate | Converting missing evidence to zero |
| Null | Machine representation of an absent value | Conflating not stated, illegible, and not applicable |
| Conditional | Valid only if a stated condition holds | Recording as an unconditional promise |
| Amendment | Instrument that may change an agreement | Assuming upload time establishes precedence |
| Ground truth | Adjudicated reference labels for an evaluation | Treating one reviewer's unchecked opinion as certainty |
| Precision | Correct proposed positive fields divided by all proposed positive fields | Ignoring omissions |
| Recall | Correct proposed positive fields divided by all truly present fields | Ignoring unsupported proposals |
| Field exact match | Share of field instances whose normalized values match the adjudicated value | Treating matching format as evidence correctness |
| Sampling frame | Defined population from which evaluation items are selected | Testing only convenient, clean examples |
| Holdout | Labeled examples kept out of prompt and threshold selection | Calling tuned examples an independent test |
| Drift | Change in input, process, or performance over time | Assuming model output changed for only one reason |
| Prompt injection | Untrusted source text attempts to redirect system behavior | Treating document instructions as user authorization |
| OCR | Optical character recognition converts image pixels to text | Assuming OCR preserves layout and all characters |
| PII | Personally identifiable information, such as a named contact | Sending all personal information when only a role is needed |
| API | Application programming interface for system-to-system requests | Assuming provider defaults meet procurement policy |
| SLA | Service-level agreement specifying contractual service expectations | Equating uptime with extraction accuracy |
| FTE | Full-time equivalent, a staffing capacity unit | Claiming cash savings whenever minutes are saved |
| RACI | Responsible, accountable, consulted, informed ownership matrix | Assigning accountability to the model |

## 5. The mechanics, step by step

### 5.1 Choose a decision and a bounded task

1. Identify the decision owner and the operational event.
   A buyer may triage a delayed order; counsel may locate renewal terms.
   The output is a candidate record for an existing controlled workflow.
2. Enumerate actions explicitly excluded from the pilot.
   Do not let the model change master data, accept contract terms, or send mail.
3. Write the cost of each relevant error in operational terms.
   A missed notice date and an incorrect contact name have different impacts.
4. Map the existing manual workflow, its volume, elapsed time, and rework.
   Measure a baseline before claiming time or cost savings.
5. Select a narrow population with an accountable owner.
   Separate incoming emails from legal agreements because their authority,
   formats, review times, and tolerance for error differ.
6. Define an exit condition before deployment.
   If a high-consequence field cannot be supported and verified, stop or escalate.

### 5.2 Establish document identity and authority

1. Assign a unique source identifier to each message and attachment.
   Link thread, sender, recipient, received timestamp, and attachment versions.
2. Authenticate through the organization's established mail and identity controls.
   A familiar display name alone is not evidence of supplier identity.
3. Store a content hash to detect changes, and retain the access-controlled original.
   A hash identifies bytes; it does not prove a document was signed or is authoritative.
4. For an agreement, collect the signed agreement, amendments, and schedules.
   Record effective dates, execution status, parties, and document relationships.
5. Ask counsel or the contract owner to resolve conflicts and precedence.
   Do not infer legal hierarchy simply from the newest modification timestamp.
6. For email, distinguish quoted thread history from the new message.
   Record who said what, when, and whether the statement is a proposal or confirmation.
7. Route missing attachments, unreadable scans, and identity mismatches to a queue.
   Do not silently substitute a previous attachment or a similarly named contract.

### 5.3 Design the extraction contract

1. Define one business meaning per field.
   For example, `proposed_arrival_date` is not `current_committed_arrival_date`.
2. Specify a type, required unit, allowed states, and normalization rules.
   Dates should carry time zone or “not stated” where cutoffs matter.
3. Include raw text, normalized value, source identifier, locator, and status.
   Preserve original units alongside any conversion.
4. Represent `not_stated`, `ambiguous`, `illegible`, and `not_applicable` distinctly.
   A null value with a reason is better than an invented value.
5. Capture qualifiers: conditionality, party, scope, effective period, and exceptions.
   These qualifiers may determine whether the field can be used at all.
6. Separate model confidence, if supplied, from a calibrated probability.
   Self-reported confidence is not a substitute for a measured error rate.
7. Obtain examples and counterexamples from buyers, counsel, and data owners.
   Version the schema and seek approval before measuring a model against it.

### 5.4 Prepare content without erasing meaning

1. Classify the source under the organization's information policy.
   Include personal, commercial, export-controlled, and privileged material where relevant.
2. Apply approved data minimization before external transmission.
   Preserve enough context to interpret the field and its qualifiers.
3. Extract text with a documented OCR and layout pipeline when needed.
   Record page, table cell, section, paragraph, or message offset as appropriate.
4. Preserve tables, footnotes, strikeouts, handwritten amendments, and page order.
   OCR that merges two table columns can reverse the commercial meaning.
5. Check coverage and readability before calling the model.
   If required pages are missing or low quality, send to manual review.
6. Split large documents at meaningful boundaries and retain source coordinates.
   Overlap alone does not guarantee that cross-document exceptions are found.
7. Avoid presenting retrieved or document text as a trusted instruction.
   The source is data, even if it contains instructions addressed to an assistant.

### 5.5 Request structured proposals

1. Provide the field specification, permitted values, and abstention instructions.
   Require exact supporting text with each non-null proposed value.
2. Delimit source content and tell the model not to obey instructions inside it.
   This reduces but does not eliminate prompt-injection risk.
3. Use a typed structured response where supported, with a schema version.
   Parse failure must be visible and routed, not silently repaired into a fact.
4. Ask for explicit qualifiers rather than a confident summary.
   A condition such as “subject to carrier confirmation” belongs in the record.
5. Do not ask for a hidden chain of thought or treat model reasoning as evidence.
   The source passage and reviewer decision are the auditable artifacts.
6. Capture model identifier, provider version if available, prompt version,
   request timestamp, preprocessing version, and approved configuration.
7. Treat unavailable citations, unsupported fields, and response truncation as failures.
   Do not fill gaps from remembered knowledge or a previous customer's record.

### 5.6 Validate before review

1. Parse the response against a strict schema and reject unknown fields.
   Type validation proves format, not truth.
2. Check that cited text exists in the exact source version at the given locator.
   Reject altered quotations and references to a different attachment.
3. Check deterministic constraints such as date syntax and permitted currency codes.
   Keep the raw value when normalization cannot be justified.
4. Check whether source identity and relevant business keys match the target order.
   A valid date attached to the wrong order remains a critical error.
5. Flag contradictions across the current message, quoted history, and attachments.
   A rule should flag conflict rather than select a legally controlling clause.
6. Route unsupported, ambiguous, critical, or out-of-policy records to a person.
   Failing closed means no downstream change is made while the record is unresolved.

### 5.7 Verify with a human who can actually decide

1. Present the proposed value and the exact source in its surrounding context.
   Show the full document when nearby text or another amendment may change meaning.
2. Blind or delay the model suggestion in a sampled review to test anchoring.
   Reviewers otherwise may accept a plausible proposal without independently reading.
3. Require an explicit decision for each critical field: accept, correct, or escalate.
   Capture reviewer identity, time, reason, and source location.
4. Separate extraction review from authorization of a change or outbound message.
   The person checking a date may not have authority to accept a revised commitment.
5. Define a second-review rule for high-impact or contested items.
   Counsel decides legal interpretation; operations decides order actions within policy.
6. Monitor reviewer capacity, backlog, corrections, and disagreement by segment.
   A rushed or unavailable reviewer is not an effective control.
7. Use periodic independent audit of accepted values, not only corrections.
   The errors reviewers miss are absent from ordinary correction logs.

### 5.8 Evaluate before release

1. Define the deployment population, business period, and sampling frame.
   Include supplier, language, format, source type, and difficulty strata.
2. Select and label examples independently of prompt construction.
   Label presence, exact value, source evidence, conditionality, and actionability.
3. Resolve labeling disagreements with a documented adjudication process.
   Preserve uncertainty when even the owner cannot establish a unique value.
4. Split by supplier, contract family, and time where applicable.
   Near-duplicate threads or amendments across train and test create leakage.
5. Use development data for prompt changes and reserve a final held-out test.
   Reusing the test for repeated tuning turns it into development data.
6. Score field-level precision and recall, citation support, abstention, and
   end-to-end errors after human review, including critical subgroup slices.
7. Compare the human-plus-model workflow to the current manual workflow.
   Measure total handling time, queue delay, cost, and operational quality.
8. Report denominators and uncertainty, particularly for rare critical errors.
   Zero observed errors in a small sample does not prove zero risk.
9. Predefine acceptance thresholds and business-owner approval.
   A failing critical field blocks release even if average accuracy looks good.

### 5.9 Operate, monitor, and recover

1. Start with a shadow mode that cannot trigger business actions.
   Compare proposed and independently reviewed records before a limited pilot.
2. Restrict access, record versions, and separate development and production data.
   Protect source text, prompts, outputs, and logs under the same classification.
3. Sample completed work continuously, including apparently successful cases.
   Track performance by source, supplier, language, field, and severity.
4. Alert on missing citations, unsupported values, latency, backlog, and cost.
   Define an owner and a response window for each alert.
5. Treat a provider model update, OCR change, or new document family as a change.
   Reevaluate against relevant held-out and adversarial examples before promotion.
6. Maintain a kill switch and a manual processing path with tested access.
   If evidence quality degrades, stop publishing suggestions and drain the queue safely.
7. Investigate incidents, preserve evidence, notify affected owners, and correct records.
   Meet applicable contractual and legal notification obligations through counsel.
8. Periodically review retention, vendor access, and deletion evidence.
   A deleted source should not remain indefinitely in unmanaged logs or exports.

### 5.10 Field-by-field specification cards

Use these cards as examples, not a universal legal schema.
A source quotation must accompany every populated candidate field.
A reviewer should see all qualifying text and relevant cross-references.

#### Field 1: Purchase order identifier

- **Meaning:** Text identifier exactly as issued by buyer.
- **Positive evidence:** The current message or attachment explicitly associates it with the supplier response.
- **Common trap:** A quoted historical order or an unrelated subject-line number.
- **Control:** Check format and match against an authorized order; never infer from sender alone.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 2: Supplier identity

- **Meaning:** Verified supplier master identifier plus stated legal name.
- **Positive evidence:** Authenticated communication linked to the supplier master and actual contracting party.
- **Common trap:** Display name matches but domain or signatory is inconsistent.
- **Control:** Escalate mismatch through the approved supplier verification channel.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 3: Item identifier

- **Meaning:** Buyer item identifier with supplier part number retained separately.
- **Positive evidence:** Line-level correspondence between the buyer item and the supplier part.
- **Common trap:** One part number is shared across several order lines.
- **Control:** Ask the buyer to resolve mapping using the source and item master.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 4: Affected quantity

- **Meaning:** Decimal amount with explicit unit and order line.
- **Positive evidence:** Supplier states how many units are delayed for a named line.
- **Common trap:** Total order quantity is mistaken for the delayed quantity.
- **Control:** Preserve original unit; apply only approved conversion factors.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 5: Current committed date

- **Meaning:** Previously authorized delivery date and any relevant time zone.
- **Positive evidence:** Order record or binding confirmation supports the date.
- **Common trap:** An old thread message contains an unapproved proposed date.
- **Control:** Show the prior approved order record and its effective timestamp.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 6: Proposed delivery date

- **Meaning:** Date the supplier proposes, with named destination and qualifier.
- **Positive evidence:** Current message explicitly states a date for the affected line.
- **Common trap:** The sender gives an estimate or a date conditional on carrier confirmation.
- **Control:** Record the condition and request a human decision before changing the order.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 7: Ship date

- **Meaning:** Date goods are expected or committed to leave the origin.
- **Positive evidence:** Message explicitly identifies dispatch or pickup date.
- **Common trap:** A delivery date is relabeled as a shipping date.
- **Control:** Maintain separate fields and record timezone if relevant.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 8: Delivery destination

- **Meaning:** Named destination or identifier in the source.
- **Positive evidence:** The text links the date and quantity to that destination.
- **Common trap:** A corporate address is confused with the receiving location.
- **Control:** Match against the authorized order and flag conflicts.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 9: Notice period

- **Meaning:** Number and unit of days or other period before an event.
- **Positive evidence:** Controlling clause specifies period, event, and applicable party.
- **Common trap:** Calendar days are silently converted to business days.
- **Control:** Preserve wording and ask counsel if counting rules are unclear.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 10: Renewal mechanism

- **Meaning:** Whether renewal is automatic, optional, or unsupported.
- **Positive evidence:** Clause states the mechanism and necessary conditions.
- **Common trap:** An amendment changes the base clause.
- **Control:** Show all potentially relevant versions and escalate precedence.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 11: Termination right

- **Meaning:** Party, trigger, procedure, and notice associated with a right.
- **Positive evidence:** Executed terms identify the party and circumstances.
- **Common trap:** A heading is mistaken for an unconditional right.
- **Control:** Extract the exception and cross-reference; obtain legal review.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 12: Liability cap

- **Meaning:** Amount or formula, currency, scope, and exceptions.
- **Positive evidence:** Relevant executed terms and amendments identify the cap.
- **Common trap:** A carve-out is omitted or a cap is applied to every claim.
- **Control:** Link each supporting and qualifying clause; legal review required.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 13: Governing law

- **Meaning:** Jurisdiction named in the governing-law provision.
- **Positive evidence:** Executed agreement identifies the relevant jurisdiction.
- **Common trap:** Forum selection is confused with governing law.
- **Control:** Keep dispute forum and governing law as separate fields.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 14: Effective date

- **Meaning:** Date the identified instrument becomes effective.
- **Positive evidence:** Executed instrument explicitly states its effective date.
- **Common trap:** Signature date, upload date, and effective date are treated as identical.
- **Control:** Record both instrument and date; escalate ambiguity.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 15: Contracting parties

- **Meaning:** Legal entities and roles as named in the agreement.
- **Positive evidence:** Executed agreement names the entities and any assignment.
- **Common trap:** A brand name is mistaken for a legal entity.
- **Control:** Preserve exact names and validate against approved master data.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 16: Amendment scope

- **Meaning:** Specific provisions and agreement version modified.
- **Positive evidence:** A signed amendment identifies the provisions it changes.
- **Common trap:** A later document is assumed to replace the entire agreement.
- **Control:** Record clause references and send unresolved precedence to counsel.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 17: Contact and notification channel

- **Meaning:** Named role, address, and required delivery method.
- **Positive evidence:** Relevant clause specifies where and how a notice must be delivered.
- **Common trap:** An email signature is treated as the contract notice address.
- **Control:** Keep operational contacts separate from formal notice requirements.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

#### Field 18: Response deadline

- **Meaning:** Date or duration stated for a specific request.
- **Positive evidence:** Current message identifies the event and deadline.
- **Common trap:** A sender time zone or starting event is missing.
- **Control:** Preserve raw wording and route unclear calculations for review.
- **Record:** source identifier, locator, raw quote, normalized value, and status.
- **Decision:** accept, correct, or escalate; do not silently complete the field.

### 5.11 Operating controls and ownership

The following example responsibility assignment is a starting point.
An organization must map the roles to real people and local policy.
The buyer owns the operational decision, while security and legal advise within
their authority; no role is delegated to the model.

| Activity | Responsible | Accountable | Consulted | Evidence |
|---|---|---|---|---|
| Define email field semantics | Procurement analyst | Procurement process owner | Buyers and data steward | Approved schema version |
| Resolve contract meaning | Contract reviewer | Legal owner | Procurement | Adjudicated source and rationale |
| Configure model endpoint | Platform engineer | Technology owner | Security and privacy | Configuration and change record |
| Review candidate fields | Trained reviewer | Operations lead | Buyer | Decision, source and timestamp |
| Accept order change | Authorized buyer | Procurement owner | Planner | Approved order transaction |
| Audit accepted results | Independent quality reviewer | Risk owner | Operations and legal | Sample design and findings |
| Pause or restart workflow | On-call operations lead | Service owner | Security and business owner | Incident and approval record |
| Approve vendor processing | Vendor manager | Data controller or owner | Legal, privacy, security | Contract and risk acceptance |

An access matrix should distinguish reading source documents, reviewing
suggestions, correcting records, approving order changes, and administering
model configuration.
Use least privilege and periodically recertify access.
A reviewer who can read a contract is not necessarily authorized to disclose
its pricing in a supplier email.
A service account should have only the permissions needed for its specific step.

Create a queue for unreadable pages and unresolved field conflicts.
Give each queue item an owner, a creation time, a due time, a severity, and a
clear escalation path.
Monitor oldest-item age as well as average time: an average can conceal a
critical notice approaching its deadline.
A queue deadline is an internal target, not a contractually binding notice date.
If the queue cannot meet its service target, increase capacity or revert to the
existing manual process rather than auto-approving candidates.

### 5.12 Decision gates and measurement cadence

Before pilot approval, document the allowed source population and excluded
high-impact actions.
Approve security controls and provider terms before using real documents.
Agree on minimum quality by critical field, maximum unsupported-value rate,
review time, backlog, and the owner of each decision.
Numerical thresholds are business-specific and should be fixed before seeing
the final held-out test results.
Do not select a threshold after the fact solely to make a result pass.

During a pilot, compare a representative sample of assisted work with an
independent manual assessment.
Measure both the error rate in model candidates and the residual error rate
after review; these are different quantities.
Include rejected records and abstentions in workload and coverage reporting.
An abstention can be correct, but a system that abstains on every document
has no extraction coverage and may not produce a useful business benefit.

At each periodic review, compare current volumes, supplier mix, language,
source formats, latency, costs, and audit findings with the approved baseline.
Investigate changes rather than assuming that a single metric explains them.
For a provider or prompt change, retain the previous approved configuration
and evaluate the whole pipeline including OCR and validation.
If a critical metric breaches its gate, halt the affected route, preserve
records, assess downstream impact, and return to the manual process.
The service owner authorizes restart after evidence of correction and review.


## 6. Worked example

### 6.1 The fictional business situation

Northbank Components buys packaging film from fictional supplier Larch Films.
It receives 12,000 supplier emails per year and reviews each manually today.
The proposed pilot handles email triage, not autonomous order changes.
Northbank's buyer remains responsible for accepting a new delivery date.
A separate contract workstream will not inherit the email pilot's quality claim.

The purchase order is `PO-4817`, line 20, for 1,200 rolls.
The order's previously approved delivery commitment is 18 March 2026.
The destination is the buyer's Leeds plant.
On 12 March 2026, an authenticated supplier contact writes:

> For PO-4817 line 20, 300 rolls are delayed.
> We expect them to reach Leeds on 24 March, subject to carrier confirmation.
> The remaining 900 rolls are still scheduled for 18 March.
> Please let us know whether split delivery is acceptable.

The attachment contains an earlier draft schedule dated 10 March 2026.
Its proposed 22 March date is not evidence that the current email promised 22 March.
A full reviewer must inspect the thread and attachment because there may be
other relevant information, but the model may not invent a binding revision.

### 6.2 Candidate record and decision

| Field | Candidate | Evidence and review |
|---|---|---|
| Order | PO-4817 | First sentence; match to authorized order |
| Line | 20 | First sentence; match to order line |
| Affected quantity | 300 rolls | First sentence; not the 1,200-roll total |
| Proposed arrival | 24 March 2026 | Second sentence; conditional estimate |
| Unaffected quantity | 900 rolls | Third sentence; still scheduled, not a new promise |
| Split delivery | Requested | Fourth sentence; buyer approval not yet given |
| Carrier confirmation | Not provided | Condition remains open |
| Revised commitment | Not established | A proposal is not an accepted change |

Arithmetic check: 300 rolls + 900 rolls = 1,200 rolls.
The affected portion is 300 / 1,200 = 25% of the ordered rolls.
The proposed late portion is six calendar days after 18 March.
That six-day difference is not necessarily the contractual delay period;
contract wording, time zones, and acceptance may change the legal result.
The buyer can investigate alternatives and request carrier confirmation.
The buyer cannot treat this extraction alone as an accepted order amendment.

### 6.3 Evaluation sample

Suppose an independently adjudicated sample contains 200 field instances.
Of these, 160 instances have a supported value and 40 are genuinely absent.
The system proposes values for 150 instances.
Of the proposed values, 135 are correct and supported, while 15 are wrong
or unsupported; 25 supported instances were not correctly extracted.
Here a wrong proposal for a present field is counted both as an incorrect
proposal and as a missed correct field, which makes the accounting explicit.

Precision = 135 / 150 = 90.0%.
Recall = 135 / 160 = 84.375%.
These are illustrative aggregate field metrics, not guarantees for each field.
If only two of the 200 instances concern a liability cap, the aggregate cannot
establish acceptable performance for that critical field.
A citation match alone also does not establish that the cited language supports
the extracted interpretation.

### 6.4 Annual business case

Baseline manual handling is 8 minutes per email at a loaded labor rate of
USD 36 per hour.
12,000 emails × 8 / 60 hours × USD 36/hour = USD 57,600 per year.
The assisted process requires 4.5 reviewer minutes per email on average.
That time includes ordinary corrections but excludes separately priced audit.
12,000 × 4.5 / 60 × USD 36/hour = USD 32,400 per year.
Annual ordinary review capacity released is USD 25,200, or 700 hours.

The model and document-processing service costs USD 0.45 per email:
12,000 × USD 0.45 = USD 5,400 per year.
Independent quality audit is 100 hours per year at USD 36/hour = USD 3,600.
Integration support and monitoring cost USD 6,000 per year.
Annual operating cost is USD 32,400 + USD 5,400 + USD 3,600 + USD 6,000
= USD 47,400; annual net operating benefit is USD 10,200.
One-time implementation is USD 18,000, so first-year net benefit is
USD 10,200 − USD 18,000 = negative USD 7,800.
Simple payback is USD 18,000 / USD 10,200 = 1.76 years, assuming benefits
and costs stay constant and ignoring financing and discounting.

These are capacity values, not automatically cashable savings.
Realized cash benefit requires an achievable staffing or spend decision.
Audit hours and platform fees must not already be counted inside the baseline.
The case excludes quantified impacts on missed delays, expedited freight,
security incidents, and legal errors because no evidence is supplied.
A pilot should measure these outcomes rather than claim them in advance.

### 6.5 Sensitivity and decision

If reviewer time is 6 minutes instead of 4.5 minutes per email, review costs
12,000 × 6 / 60 × USD 36 = USD 43,200 per year.
Adding USD 5,400, USD 3,600, and USD 6,000 gives USD 58,200 per year.
The assisted process then costs USD 600 per year more than the baseline,
before the USD 18,000 implementation expense.
The business case is therefore sensitive to actual review time.

A reasonable decision is a restricted pilot only if the owner agrees in advance
to accuracy, security, backlog, and review-time gates.
If the pilot misses any critical gate, retain manual processing and investigate.
The net benefit calculation does not authorize autonomous supplier communication.

## 7. Code, query, or calculation

This standard-library Python example checks a model-like proposal against a
small source and computes field metrics and the annual business case.
It does not call an LLM, authenticate an email, read attachments, establish
contract authority, or approve a purchase order.
The candidate is deliberately supplied as test data so the example runs offline.
A matching quote is only a location check, not proof of semantic support.
Production systems need stable document identifiers and page or character
coordinates as well as access controls and the full review workflow.

Save as `extraction_demo.py` and run `python extraction_demo.py`.
All dates and values are fictional.

```python
from decimal import Decimal
import json

SOURCE_ID = "email-2026-03-12-4817"
SOURCE = (
    "For PO-4817 line 20, 300 rolls are delayed. "
    "We expect them to reach Leeds on 24 March, "
    "subject to carrier confirmation. "
    "The remaining 900 rolls are still scheduled for 18 March. "
    "Please let us know whether split delivery is acceptable."
)

# This is an illustrative model proposal, not a model call.
proposal = {
    "order_id": {
        "value": "PO-4817",
        "source_id": SOURCE_ID,
        "quote": "For PO-4817 line 20, 300 rolls are delayed.",
        "status": "proposed",
    },
    "delayed_quantity": {
        "value": {"amount": 300, "unit": "rolls"},
        "source_id": SOURCE_ID,
        "quote": "For PO-4817 line 20, 300 rolls are delayed.",
        "status": "proposed",
    },
    "proposed_arrival": {
        "value": "2026-03-24",
        "source_id": SOURCE_ID,
        "quote": (
            "We expect them to reach Leeds on 24 March, "
            "subject to carrier confirmation."
        ),
        "status": "conditional",
    },
    "revised_commitment": {
        "value": None,
        "source_id": SOURCE_ID,
        "quote": None,
        "status": "not_established",
    },
}


def check_locations(record, source_id, source):
    """Flag absent or nonmatching quotations, not semantic correctness."""
    problems = []
    for name, item in record.items():
        if item["source_id"] != source_id:
            problems.append(f"{name}: wrong source")
        if item["value"] is not None:
            quote = item["quote"]
            if not quote or quote not in source:
                problems.append(f"{name}: missing quotation")
    return problems


def ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


correct = 135
proposed = 150
present = 160
precision = ratio(correct, proposed)
recall = ratio(correct, present)

volume = Decimal("12000")
rate = Decimal("36")
baseline_minutes = Decimal("8")
review_minutes = Decimal("4.5")
baseline = volume * baseline_minutes / 60 * rate
review = volume * review_minutes / 60 * rate
service = volume * Decimal("0.45")
audit = Decimal("100") * rate
support = Decimal("6000")
implementation = Decimal("18000")
operating_benefit = baseline - review - service - audit - support
first_year = operating_benefit - implementation
payback = implementation / operating_benefit if operating_benefit > 0 else None

print("Location problems:", json.dumps(check_locations(proposal, SOURCE_ID, SOURCE)))
print(f"Precision: {precision:.2%}; recall: {recall:.3%}")
print(f"Baseline: USD {baseline:.2f}/year")
print(f"Assisted operating cost: USD {review + service + audit + support:.2f}/year")
print(f"Annual operating benefit: USD {operating_benefit:.2f}/year")
print(f"First-year net benefit: USD {first_year:.2f}")
print(f"Simple payback: {payback:.2f} years")
print("Human approval required; no order change has been made.")
```

Expected output:

```text
Location problems: []
Precision: 90.00%; recall: 84.375%
Baseline: USD 57600.00/year
Assisted operating cost: USD 47400.00/year
Annual operating benefit: USD 10200.00/year
First-year net benefit: USD -7800.00
Simple payback: 1.76 years
Human approval required; no order change has been made.
```

The empty problem list does not mean the proposal is approved.
For example, a real quotation may support the opposite of the model's value.
A unit test for substring presence cannot decide that question.
The input year in the example comes from the surrounding scenario, not the
quoted date alone; production normalization needs explicit provenance for it.

## 8. Data, standards, and regulatory context

### 8.1 Information categories and rights

Commercial agreements may contain prices, rebates, capacity, security terms,
trade secrets, named contacts, and privileged communications.
Supplier messages may contain personal data and information subject to
confidentiality commitments.
Classify each dataset before transmission and confirm that contracts and
applicable law permit the proposed processing and subprocessors.
A supplier relationship does not automatically permit use of its documents
for provider model training or unrelated product improvement.

Data minimization means transmitting only what is necessary for the defined
purpose, while retaining enough context for a reliable interpretation.
Masking a name may protect identity but can also remove the party needed to
interpret a clause; test both protection and task validity.
Access policies must apply to originals, derived text, prompts, outputs,
indexes, audit logs, backups, and exported review files.
Use approved retention and deletion schedules, with legal-hold exceptions.

### 8.2 Privacy and law

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679,
sets obligations for processing personal data where it applies.
In particular, Article 5 lists principles including purpose limitation,
data minimization, accuracy, storage limitation, and security.
Article 28 addresses processor arrangements; international transfers have
additional requirements under Chapter V.
The applicability, legal basis, roles, and transfer mechanism depend on the
facts and require qualified privacy advice.
Not every supplier document is personal data, and encryption alone does not
resolve the legal basis or transfer question.

The EU AI Act, Regulation (EU) 2024/1689, has phased application and its
obligations depend on the system, role, purpose, and applicable dates.
Document extraction is not automatically a high-risk use merely because it
uses an LLM; assess the actual use and current implementation guidance.
This educational discussion is not a legal classification or advice.
Contract notice requirements arise from the governing agreement and applicable
law; a model-generated summary does not change them.

### 8.3 Standards and conventions

ISO/IEC 27001:2022 provides requirements for an information security
management system; certification scope and audit evidence matter.
ISO/IEC 42001:2023 specifies requirements for an AI management system.
Neither certification proves that a particular extraction is correct or that
all supplier documents are covered by the certified scope.
NIST's AI Risk Management Framework (AI RMF 1.0) and its Generative AI Profile
are voluntary risk-management guidance, not universal legal requirements.
They support governance, measurement, monitoring, and response design.

Electronic data interchange (EDI) uses structured business messages, whereas
email extraction interprets unstructured content.
Where an existing validated structured order response is available, compare it
as a baseline before adding generative extraction.
An EDI message is still subject to identity, business-rule, and authorization
checks; structured format does not itself prove a supplier promise is valid.
Use ISO 8601 date representations and ISO 4217 currency codes where relevant,
while retaining original wording and resolving ambiguous local dates explicitly.
The standard representation does not decide whether a date is a commitment.

### 8.4 Provider and vendor due diligence

Ask the provider to identify data locations, subprocessors, retention periods,
training and product-improvement use, and deletion mechanics for each service.
Obtain the actual contract, data-processing terms, and configuration evidence;
a sales slide is not a binding control.
Distinguish account defaults from settings that the customer has enabled.
Verify encryption, key management, access restrictions, incident notification,
auditing, independent assurance, and business-continuity commitments.

Determine whether provider staff can access content for abuse monitoring,
support, or troubleshooting, and how exceptions are authorized and logged.
Ask how backups, caches, embeddings, and operational logs are treated.
Confirm how model updates are announced, whether a version can be pinned,
and what happens when a pinned version is retired.
Agree how a breach, subcontractor change, or loss of service is escalated.
Compare exit and portability rights with the ability to delete derived data.
Record unresolved questions as vendor risks with owners and due dates.

## 9. Industry practice and case evidence

### 9.1 What public evidence can and cannot show

The National Institute of Standards and Technology (NIST) publishes a
voluntary AI risk-management framework and a generative-AI profile.
These publications recommend organizational governance and measurement
practices; they do not report accuracy for this fictional supplier workflow.
Their relevance here is the lifecycle approach: govern, map, measure, manage.

The UK Information Commissioner's Office (ICO) has published guidance on AI
and data protection, including accuracy and governance considerations.
That guidance is a regulatory explanation, not proof that any particular
model or procurement workflow complies with the law.
A local privacy team must decide applicability and appropriate controls.

The OWASP Foundation documents prompt injection as an LLM application risk.
A supplier message or attachment may contain instructions intended to alter
the model's behavior, so untrusted text must not become a source of authority.
A prompt alone is not an isolation boundary; tool permissions and downstream
approval enforce the boundary even if the model follows hostile content.

### 9.2 Evidence hierarchy for a procurement decision

A peer-reviewed task study may show that an approach works on a specific corpus.
A provider's published benchmark may describe a different language, document
quality, label definition, and error distribution from the buyer's population.
A customer story can reveal an implementation pattern, but its stated savings
may not include audit, staff redeployment, integration, or selection effects.
An internal controlled pilot can estimate local workflow impact if it has a
representative sample, comparison group, documented costs, and held-out test.

For each supplier claim ask for numerator, denominator, dates, benchmark corpus,
scoring method, model version, and whether humans corrected the result.
Ask whether the reported unit is a document, a field, a page, or an action.
If accuracy is measured only on completed fields, omitted critical fields
may be invisible.
If labor time excludes human verification or rework, it is not end-to-end time.
No named company's financial savings are assumed in the worked example.

### 9.3 Adoption pattern

[INFERENCE] A low-risk entry point is a read-only triage queue with citations,
followed by a controlled pilot and sampled audits.
The inference follows from the error and authority boundaries described above;
it is not a claim that a named company used this exact workflow.
Organizations with low volume or difficult contract interpretation may prefer
a search tool and manual review over a hosted model.
For supplier communications, identity verification and authorization often
matter as much as extraction quality.

A draft reply can be useful if it is labeled as a draft and reviewed against
the full thread.
It should not promise acceptance, disclose confidential terms, or send itself.
When a response requires negotiation, the buyer owns the position and wording.
Measure whether drafting reduces total work rather than merely moving editing
from one screen to another.

## 10. Failure modes and red flags

The table lists a warning, the causal mechanism, the business consequence,
and a control; none of the controls eliminates the need for ownership.

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|

| Fabricated value | Quote missing or irrelevant | Model fills an unstated field | False operational fact | Require source and human verification |

| Wrong purchase order | Order ID absent or mismatched | Thread contains multiple orders | Action against another line | Cross-check authenticated source and order master |

| Condition dropped | Qualifier absent from candidate | Model compresses conditional language | Estimate treated as promise | Capture condition and block automatic changes |

| Superseded clause | Old clause cited alone | Amendment omitted from context | Incorrect legal interpretation | Collect versions and obtain counsel adjudication |

| Wrong party | Party name differs across documents | Affiliate and parent conflated | Rights attributed incorrectly | Validate legal entity and role |

| OCR digit error | Low-resolution page or inconsistent totals | Image conversion swaps characters | Wrong amount or date | Show image and require critical-field verification |

| Layout loss | Table row values appear under wrong header | OCR flattens columns | Quantity assigned to another SKU | Retain cell coordinates and compare original |

| Quote laundering | Verbatim quote exists but does not entail value | Citation validation checks presence only | False confidence in unsupported claim | Human checks meaning and qualifiers |

| Prompt injection | Document asks model to ignore rules | Untrusted text treated as instruction | Disclosure or invalid output | Isolate data, restrict tools, review output |

| Email impersonation | Sender domain or payment details change | Display name trusted as identity | Fraud or unauthorized change | Authenticate through independent approved channel |

| Reviewer anchoring | Very low correction rate despite audit misses | Reviewers accept fluent output | Errors pass the human control | Blind sample and monitor independent audits |

| Queue overload | Review backlog and aging increase | Capacity below arrival rate | Missed response or notice deadline | Staff queue and fail over to manual workflow |

| Distribution shift | New language or template appears | Training and test no longer represent use | Subgroup quality declines | Segment monitoring and requalification |

| Vendor model change | Version or behavior changes unexpectedly | Provider updates model behind endpoint | Previously approved quality degrades | Version control, regression evaluation, rollback |

| Data leakage | Sensitive content appears in logs or exports | Broad logging and retention | Confidentiality or privacy incident | Minimize, encrypt, restrict access, delete |

| Overstated savings | Review or integration hours omitted | Business case counts gross minutes only | Negative return despite approval | Measure end-to-end time and recurring costs |

| Silent parse repair | Malformed response accepted after guessing | Parser replaces missing fields | Unsupported value enters record | Reject malformed output and route to review |

| Quoted history confusion | Date comes from old message in thread | Old text presented without speaker chronology | Outdated schedule used | Identify sender and timestamp for each quote |

Treat security incidents, incorrect commitments, and missed formal notice dates
as potentially severe even if they are infrequent.
A release gate should specify who can stop the workflow and how quickly.
An accepted extraction that later proves wrong requires correction of all
consuming records, not merely an updated model prompt.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|

| Source is complete | Context determines meaning | Missing schedule or amendment | Check attachment inventory and ask owner |

| Source is authentic | Sender authority affects reliance | Spoofed email or unexecuted draft | Verify identity and execution status |

| Field definition is stable | Labels and scoring need a common target | Policy changes the meaning of commitment | Version schema and re-label relevant cases |

| Adjudicators are competent | Reference labels require domain knowledge | Reviewers disagree on legal scope | Use counsel and record unresolved ambiguity |

| Test resembles deployment | Local performance must transfer to live traffic | New supplier, language, or channel | Monitor strata and run targeted evaluation |

| Reviewers have time | Human verification is an active control | Backlog or incentive to accept quickly | Measure elapsed review and audit accepted records |

| Provider settings stay approved | Confidential processing depends on configuration | Default retention or subprocessors change | Recheck contract and deployed settings |

| Cost rate is realistic | Annual benefit depends on labor economics | Saved time cannot be redeployed | Measure capacity and separately state cash benefit |

| Historical order is reliable | Comparison needs a committed baseline | Master record is stale or wrong | Reconcile against approved order history |

| Date normalization is justified | Machine dates need year and locale | Ambiguous 03/04 or missing time zone | Retain raw text and abstain when unresolved |

| Audit sample can detect issues | Monitoring must cover important cases | Rare critical errors absent from sample | Use targeted sampling and report uncertainty |

| Model version is known | A test applies to a specific system | Opaque provider update | Request change notice and reevaluate |

These assumptions are not promises by a vendor.
If a critical assumption fails, the owner must either change the workflow,
obtain more evidence, or retain manual handling.
An observed low error rate does not eliminate legal interpretation risk.

## 12. Exercises

Each exercise is independent unless it explicitly refers to the worked case.
State units, denominators, ownership, and assumptions in your deliverable.

### Exercise 1 — Separate the decision layers

Time: 12 minutes.
A supplier writes: “We expect 200 cartons on Friday if customs clears.”
A model returns “200 cartons committed for Friday” and a buyer accepts it.
The order system then automatically updates its delivery commitment.
Deliverable: a four-row source/proposal/review/action analysis.
Acceptance: identify the condition, the unsupported change in status, the
reviewer's error, and the unauthorized automation risk.

### Exercise 2 — Design a field

Time: 18 minutes.
Specify a contract notice-period field for an agreement with amendments.
Deliverable: a schema card with type, unit, triggering event, source,
qualifiers, allowed abstentions, and human owner.
Acceptance: distinguish calendar from business days and recognize that an
amendment may modify the base agreement.

### Exercise 3 — Resolve a conflicting thread

Time: 15 minutes.
An older message proposes 6 May; the current email proposes 8 May if a
carrier confirms; the order currently records 5 May.
Deliverable: a three-row table of current commitment, current proposal, and
unresolved condition, plus the next authorized step.
Acceptance: retain all three dates and do not silently accept 8 May.

### Exercise 4 — Compute extraction metrics

Time: 15 minutes.
There are 250 adjudicated field instances; 200 have supported values.
The system proposes 180 values, 153 correct and supported.
Deliverable: precision and recall with formulas and denominators.
Acceptance: calculate 85.0% precision and 76.5% recall, and state one limitation.

### Exercise 5 — Recalculate the business case

Time: 18 minutes.
Use the worked case but assume 6 minutes of assisted review per email.
Keep every other annual cost and implementation expense unchanged.
Deliverable: assisted annual cost, annual net benefit, first-year result,
and a decision on whether a positive simple payback exists.
Acceptance: include audit and support costs and distinguish annual and one-time amounts.

### Exercise 6 — Evaluate a critical-field launch

Time: 25 minutes.
A vendor reports 98% field accuracy on 100 clean English contracts it selected.
Two instances concern liability caps; no amendments were included.
Deliverable: a test plan with sampling strata, holdout rules, labeling,
critical-field gates, and uncertainty reporting.
Acceptance: reject the aggregate claim as sufficient launch evidence and
explain how amendment coverage and adjudication enter the plan.

### Exercise 7 — Design incident response

Time: 20 minutes.
A provider changes its model and a review audit finds unsupported delivery dates.
Deliverable: a sequence of containment, assessment, correction, communication,
requalification, and restart decisions.
Acceptance: include a named accountable role, a stop mechanism, and a manual path.

### Exercise 8 — Assess a provider and workflow

Time: 25 minutes.
A provider says it is secure, retains prompts for an unspecified period, and
may use subprocessors without explaining where data is stored.
Deliverable: a due-diligence checklist and a risk-based pilot recommendation.
Acceptance: require contractual evidence, data-use terms, retention and deletion,
access controls, model changes, incident response, and an exit route.

## 13. Self-check answers

### Answer 1

The source states an estimate of 200 cartons on Friday, conditional on customs.
The proposal converts an estimate into a commitment and drops the condition.
The reviewer should correct or escalate instead of accepting it.
An order update requires a separately authorized workflow and must not happen
merely because a candidate field was accepted.

### Answer 2

A strong card stores the stated period, unit, triggering event, party, clause,
source version, exact quotation, and relevant amendment references.
It distinguishes calendar days from business days where the contract does.
Absent, ambiguous, and illegible are different outcomes.
The contract owner or counsel resolves legal interpretation and precedence.
An answer that stores only `30` without a unit and trigger is insufficient.

### Answer 3

The existing recorded commitment is 5 May, subject to verifying the order.
The earlier proposal was 6 May; it is historical, not the current proposal.
The current proposal is 8 May, conditional on carrier confirmation.
The buyer should request confirmation and decide through the approved change
process; the model must not mark 8 May as the new commitment by itself.

### Answer 4

Precision is 153 / 180 = 0.85 = 85.0%.
Recall is 153 / 200 = 0.765 = 76.5%.
Precision uses proposed values as its denominator; recall uses supported values.
Neither average reveals which specific critical fields failed or whether
citations and conditions were interpreted correctly.

### Answer 5

Review labor is 12,000 × 6 / 60 × USD 36 = USD 43,200 per year.
Service, audit, and support are USD 5,400, USD 3,600, and USD 6,000 per year.
The assisted operating cost is USD 58,200 per year.
Compared with USD 57,600 baseline cost, annual net benefit is negative USD 600.
After USD 18,000 implementation, the first-year result is negative USD 18,600.
There is no positive simple payback under a constant negative annual benefit.

### Answer 6 rubric

A strong design samples the intended deployment population by supplier,
language, format, contract family, amendments, and critical field.
It reserves a time- or family-separated holdout and documents label adjudication.
It measures supported values, omissions, citation quality, and review outcomes.
It reports denominators and uncertainty for liability caps and other rare cases.
It has preapproved critical gates; two cap cases cannot establish robust quality.
An aggregate accuracy result alone is not an adequate safety decision.

### Answer 7 rubric

A strong response gives operations authority to stop publication or changes,
retains the original sources and event logs, and routes work to manual review.
It identifies affected records and their downstream actions, corrects them,
and notifies security, business owners, and relevant parties under policy.
It isolates the provider version change, evaluates a candidate recovery,
and requires documented owner approval before controlled restart.

### Answer 8 rubric

A strong answer requests actual contract and data-processing terms, not just
marketing claims, and identifies permitted uses, locations, and subprocessors.
It checks encryption, access, incident notification, audit evidence, deletion,
backup treatment, versioning, continuity, and exit.
The unresolved retention and subprocessor questions prevent exposure of
confidential production data until an authorized owner accepts or resolves risk.
A limited test with synthetic data may proceed only under local policy.

## 14. Interview questions

### 1. Why is extraction not the same as authorization?

A strong answer separates a proposed field, evidence review, and an action
approved by a person with the appropriate business authority.
A weak answer assumes that valid JSON can update a purchase order.

### 2. What makes a citation useful?

A strong answer names the exact source version, locator, surrounding context,
and a check that the quoted passage actually supports the claim.
A weak answer counts any link as proof of truth.

### 3. How would you handle a conflicting contract amendment?

A strong answer collects executed instruments, preserves document identity,
flags the conflict, and escalates precedence to the contract owner or counsel.
A weak answer chooses the latest upload without checking authority.

### 4. How do precision and recall differ?

A strong answer gives both denominators and explains that unsupported values
and missing values can create distinct business harm.
A weak answer reports a single average without a population or denominator.

### 5. What is a representative evaluation?

A strong answer samples the target population, stratifies risk, separates
supplier families and time, adjudicates labels, and protects a holdout.
A weak answer uses only the vendor's clean demonstration documents.

### 6. Why can human review still fail?

A strong answer discusses anchoring, workload, missing context, expertise,
independent audits, and the reviewer's authority to escalate.
A weak answer treats the presence of an accept button as a complete control.

### 7. How should supplier email instructions be treated?

A strong answer regards them as untrusted content, verifies identity, and
limits tools and actions regardless of model instruction-following.
A weak answer lets a message authorize data disclosure or a payment change.

### 8. How do you evaluate a vendor's data practices?

A strong answer requests contractual terms and operational evidence for use,
retention, subprocessors, access, deletion, incidents, versioning, and exit.
A weak answer relies solely on a security badge or salesperson's assurance.

### 9. What is the business case denominator?

A strong answer states volume, time per item, loaded labor rate, all recurring
costs, one-time investment, realized cash assumptions, and risk exclusions.
A weak answer multiplies an unmeasured time saving by a salary and calls it cash.

### 10. What triggers a rollback?

A strong answer defines preapproved thresholds, an owner, a stop switch,
manual fallback, affected-record investigation, and restart criteria.
A weak answer waits indefinitely for a model vendor to fix the issue.

## 15. Key terms

The section references point to the primary explanation in this unit.

- [Abstention](#4-definitions-and-vocabulary): explicit refusal to populate an unsupported field.
- [Adjudication](#58-evaluate-before-release): resolution of reference-label disagreement.
- [Amendment](#52-establish-document-identity-and-authority): instrument potentially modifying an agreement.
- [Authority](#3-the-core-mental-model): permission to make a business or legal decision.
- [Baseline](#64-annual-business-case): current process used for comparison.
- [Business case](#64-annual-business-case): costs and benefits over a specified period.
- [Citation](#56-validate-before-review): link to a source passage and location.
- [Commitment](#3-the-core-mental-model): authorized promise, distinct from an estimate.
- [Conditionality](#62-candidate-record-and-decision): circumstances that constrain a statement.
- [Data minimization](#81-information-categories-and-rights): limiting processing to what is necessary.
- [Document identity](#52-establish-document-identity-and-authority): specific source and version.
- [Evidence span](#4-definitions-and-vocabulary): exact relevant source passage.
- [Extraction](#1-why-this-topic-exists): conversion of source content to candidate fields.
- [Field](#53-design-the-extraction-contract): defined data item with a type and meaning.
- [Ground truth](#58-evaluate-before-release): adjudicated evaluation reference.
- [Holdout](#58-evaluate-before-release): data protected from development and tuning.
- [Human verification](#57-verify-with-a-human-who-can-actually-decide): review of evidence and candidate value.
- [LLM](#1-why-this-topic-exists): model generating text conditioned on supplied context.
- [OCR](#54-prepare-content-without-erasing-meaning): optical character recognition for image text.
- [Precision](#63-evaluation-sample): correct proposals divided by all proposals.
- [Prompt injection](#54-prepare-content-without-erasing-meaning): source content attempting to redirect the system.
- [Recall](#63-evaluation-sample): correct proposals divided by supported field instances.
- [Reviewer anchoring](#10-failure-modes-and-red-flags): uncritical reliance on the model's suggestion.
- [Sampling frame](#58-evaluate-before-release): population eligible for evaluation selection.
- [Schema](#53-design-the-extraction-contract): typed structure and interpretation rules.
- [Source version](#52-establish-document-identity-and-authority): exact document revision used as evidence.
- [Supplier authentication](#52-establish-document-identity-and-authority): verification of communication identity.
- [Vendor risk](#84-provider-and-vendor-due-diligence): exposure from a provider's controls and terms.

## 16. Sources

These sources support the general standards and risks; they do not validate
the fictional case's performance or economics.
Confirm current versions and applicability with qualified owners.

1. NIST, [AI Risk Management Framework (AI RMF 1.0), AI 100-1](https://doi.org/10.6028/NIST.AI.100-1), 2023.
   Voluntary framework for governing, mapping, measuring, and managing AI risk.
2. NIST, [Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile, AI 600-1](https://doi.org/10.6028/NIST.AI.600-1), 2024.
   Generative-AI-specific risk and suggested actions.
3. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), 2016.
   Primary text for privacy obligations and data-subject rights.
4. European Union, [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), 2024.
   Primary text for the AI Act and its phased application.
5. UK Information Commissioner's Office, [Guidance on AI and data protection](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/).
   Regulatory guidance; check its jurisdiction and current revision.
6. OWASP Foundation, [OWASP Top 10 for Large Language Model Applications](https://genai.owasp.org/).
   Practitioner resource on risks including prompt injection.
7. ISO, [ISO/IEC 27001 information security management](https://www.iso.org/standard/27001).
   Standard overview; certification scope must be examined separately.
8. ISO, [ISO/IEC 42001 AI management systems](https://www.iso.org/standard/81230.html).
   Management-system standard overview; not an extraction accuracy guarantee.
9. ISO, [ISO 8601 date and time format](https://www.iso.org/iso-8601-date-and-time-format.html).
   Date/time representation, not a rule for interpreting contractual intent.
10. ISO, [ISO 4217 currency codes](https://www.iso.org/iso-4217-currency-codes.html).
    Codes for currencies, not a source of exchange rates or contract terms.

## 17. Links

- [Module 18 topic index](_index.md).
- [Module 18 survey](../../../course/18_deep_learning_foundation_models_llm_agents.md).
- [Neural network fundamentals](01_neural-network-fundamentals.md).
- [Evaluating zero-shot and fine-tuned models](06_zero-shot-evaluation.md).
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
  Use the lab context to consider downstream decision and implementation controls.
