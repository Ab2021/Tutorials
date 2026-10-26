# Regulation and Policy

> Module 22 — Sources and Further Reading
> Topic 03 of 09
> Estimated study time: 3 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

Regulatory sources carry more weight per sentence than any other source in this module. A methods paper that is wrong produces a bad model. A regulation that is misread produces a product that cannot be sold, a submission that is rejected, or a compliance programme aimed at a requirement that no longer exists. The cost of a regulatory citation error is measured in months and in legal fees, and it is almost always avoidable with a status check that takes minutes.

The errors are predictable:

- A team cites a draft guidance as if it were final, because the PDF was saved during the comment period and the filename did not change.
- A team cites a guidance that has been superseded, because the agency kept the old URL alive and the search result ranks it first.
- A team applies a European Union requirement to a United States launch, or applies a federal requirement without checking state law.
- A team reads the effective date as the publication date and plans a launch that is not yet lawful, or delays one that is.
- A team treats a harmonised standard as binding when the legal instrument only creates a presumption of conformity.
- A team cites a regulation's article without noticing that an amending regulation changed the article's text.
- A team quotes a date from a secondary summary that was accurate when written and is now wrong because the timeline was amended.

The last error is the most interesting, because it does not involve carelessness. The secondary summary may have been correct at the time, from a reputable source, and still be wrong at the review date. Regulatory timelines are amended, transition periods are extended, and implementing acts change the practical date. A source that was right in 2025 can be wrong in 2026 without anyone being at fault.

The people blocked by this are specific. The regulatory lead cannot sign a strategy without a defensible reading. The product manager cannot sequence a launch. The quality lead cannot map a process to the current requirement. The investor cannot assess timeline risk. The clinical lead cannot tell whether a claim is permitted in labelling. A shared method for reading and maintaining regulatory sources gives each of them the same answer.

This topic covers the United States and the European Union as the two systems most represented in the course, with international standards as the connective tissue. It is educational and not legal advice. The method is portable to other jurisdictions: identify the instrument, establish its status, establish its dates, establish its scope, and record the version.

A note on why regulation is treated as a source problem rather than a legal specialty. Most of the errors that reach a submission are not subtle legal questions. They are ordinary source errors: a stale date, a superseded guidance, a missing article number, or a jurisdiction that was assumed rather than checked. The legal specialty begins where the source work ends.

The topics that follow apply the same discipline to methods, clinical background, datasets, and industry evidence. The regulatory tier differs in consequence, not in kind. The same five tests apply, and the same habit of dating a claim is what makes it auditable.

> **Note:** A regulatory claim is only as good as its citation. "FDA requires" is not a citation. "21 CFR 820.10, as amended effective 2 February 2026" is. The difference is not pedantry; it is the difference between a claim that can be checked and one that cannot.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** binding instruments from guidance, standards, FAQs, and commentary, and state the legal status of each for a named jurisdiction. Assessed by Exercise 1 and self-check item 1.
2. **Establish** the status of an instrument at a review date: draft, final, in force, in transition, superseded, or withdrawn. Assessed by Exercise 2 and section 7.
3. **Read** an effective date, an application date, and a transition provision correctly, and compute the practical deadline for a product. Assessed by Exercise 3 and the worked example.
4. **Trace** a requirement through the hierarchy from statute to guidance to standard, and identify which layer creates the obligation. Assessed by Exercise 4.
5. **Detect** a stale regulatory citation in a document and replace it with the current instrument and version. Assessed by Exercise 5.
6. **Apply** a jurisdiction check to a claim that was written for one market and is being reused in another. Assessed by Exercise 6.
7. **Build** a regulatory watch list with owners, cadences, and event triggers. Assessed by Exercise 7.
8. **Write** a regulatory sentence that names the instrument, the provision, the version, and the date, and that a reviewer can verify. Assessed by Exercise 8.

The outcomes deliberately stop short of legal advice. A learner who can do all eight can prepare a regulatory question for counsel, which is the skill the role requires.

Each outcome is observable in a document. "Understands the AI Act" is not an outcome; "writes a dated citation with the article and the amending instrument" is.

The exercises are cumulative in the same way as the outcomes. Exercise 2 produces the status table that Exercise 3 uses for the date calculation, and Exercise 5 audits a real document with the vocabulary from Exercise 1.

## 3. The core mental model

A regulatory statement has four coordinates: the instrument, the version, the jurisdiction, and the date. A sentence that omits any of them is ambiguous, and ambiguity in this domain is expensive.

```text
                    QUESTION
        "May we market X for use Y in market Z?"
                              |
                              v
                    INSTRUMENT LAYER
        statute -> regulation -> guidance -> standard
                              |
                              v
                    STATUS CHECK
        draft | final | in force | transition | superseded | withdrawn
                              |
                              v
                    DATE CHECK
        publication | entry into force | application | transition end
                              |
                              v
                    SCOPE CHECK
        jurisdiction | product type | classification | intended use
                              |
                              v
                    OBLIGATION CHECK
        binding requirement | presumption of conformity | recommendation
                              |
                              v
                    CITATION
        instrument, provision, version, date, jurisdiction
```

Read the diagram downwards. Each layer can invalidate the one below it. A final guidance that interprets a regulation which has since been amended is not current, even though the guidance itself has not been withdrawn. A standard that is harmonised for one directive does not create a presumption of conformity under a different regulation.

**The binding hierarchy is not a quality hierarchy.** A guidance document can be clearer and more detailed than the regulation, and it is still not the obligation. When the two appear to conflict, the regulation governs, and the guidance describes how the agency reads it. This matters because guidance is often written in mandatory language ("must", "should") that reads like law.

**The date that matters is the application date.** Publication, entry into force, and application can be three different dates, sometimes years apart. A regulation can be in force and not yet applicable. A transition provision can allow an older requirement to be met for a defined period. A launch plan built on the publication date is a plan built on the wrong date.

**Jurisdiction is not a formality.** The same product can be a medical device in one market and not in another, can be class II in one and class IIa in another, and can require a notified body in one and a self-declaration in another. A claim written for one market cannot be reused without a scope check.

**Status decays silently.** Agencies withdraw, revise, and replace documents without changing the URL. A saved PDF carries no status information. The status has to be checked at the issuer, at the review date, every time the claim is reused.

| Layer | Example (US) | Example (EU) | Creates an obligation |
|---|---|---|---|
| Statute | Federal Food, Drug, and Cosmetic Act | Treaty and regulations adopted under it | Yes |
| Regulation | 21 CFR Part 820 | MDR 2017/745 | Yes |
| Guidance | FDA guidance document | MDCG guidance | Generally no |
| Standard | Recognised consensus standard | Harmonised standard | Only through the instrument that cites it |
| Commentary | Law firm note, newsletter | Trade association note | No |

The four coordinates are easy to remember as a sentence template: "Under [instrument] [version], in [jurisdiction], as of [date], the requirement is [provision]." If a draft cannot be fitted into the template without inventing a coordinate, the draft is not yet a regulatory statement.

A second template is useful for the obligation: "[Instrument] requires [action] for [product class] from [date]." This one forces the product class into the open, which is where most scope errors hide.

The two templates are not a writing style. They are a test of whether the claim has been thought through far enough to be checked.

## 4. Definitions and vocabulary

### 4.1 Instrument types

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Statute | A law enacted by a legislature. | Used for any requirement. | Public law number |
| Regulation | A binding legal instrument adopted under delegated authority. | Confused with guidance. | Regulation number |
| Directive | An EU instrument that member states implement into national law. | Assumed to apply directly. | Directive number |
| Guidance | An agency's interpretation or recommendation, generally non-binding. | Cited as law. | Document number and date |
| Draft guidance | A proposed position open for comment. | Cited as the agency's position. | Draft status |
| Final guidance | A position issued after comment. | Assumed binding. | Final status and date |
| FAQ | An informal answer to a common question. | Treated as a formal position. | Web page |
| Standard | A technical specification adopted by a standards body. | Assumed legally binding everywhere. | Standard number and edition |
| Harmonised standard | An EU standard cited in the Official Journal, giving presumption of conformity. | Treated as the only way to comply. | Citation in the OJ |
| Recognised consensus standard | A standard the FDA has recognised for a declared use. | Treated as mandatory. | FDA recognition listing |
| MDCG guidance | Guidance endorsed by the Medical Device Coordination Group. | Treated as Commission regulation. | MDCG number and year |
| IMDRF document | International guidance from regulators' forum. | Treated as binding in any jurisdiction. | Document number |
| Sunset clause | A provision that expires on a stated date. | Assumed not to exist. | Date |
| Grace period | An informal term for a transition; not a legal term of art. | Used as if it were defined. | Avoid in citations |
| Derogation | An exception granted under defined conditions. | Confused with a transition. | Article reference |
| Implementing decision | A decision applying an act to a specific case. | Confused with general guidance. | Decision number |

### 4.2 Status and dates

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Publication date | The date the instrument is published in the official journal or register. | Confused with the effective date. | Date |
| Entry into force | The date the instrument becomes legally valid. | Confused with the date it applies. | Date |
| Application date | The date from which the obligations apply. | Omitted from planning. | Date |
| Effective date | The date an obligation becomes enforceable; usage varies by jurisdiction. | Assumed to equal publication. | Date |
| Transition period | A window during which an older requirement may still be met. | Ignored, producing a false compliance claim. | Date range |
| Sunset | A date on which a provision or instrument expires. | Assumed not to exist. | Date |
| Superseded | Replaced by a later instrument or version. | Cited because the URL still resolves. | Status flag |
| Withdrawn | Removed by the issuer before replacement. | Assumed to reflect current policy. | Status flag |
| Consolidated version | A version that incorporates amendments for reading convenience. | Treated as the official legal text. | Version identifier |
| Corrigendum | A correction to a published instrument. | Ignored. | Notice |
| Implementing act | A later instrument that supplies detail or dates. | Missed when reading the parent regulation. | Act number |
| Delegated act | A later instrument that supplements the parent regulation. | Confused with implementing acts. | Act number |

### 4.3 Jurisdiction and scope

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Jurisdiction | The legal territory whose rules apply. | Assumed to be the location of the developer. | Country or region |
| Placing on the market | Making a device available for the first time in a jurisdiction. | Confused with distribution or use. | Regulatory event |
| Intended use | The manufacturer's stated purpose, which determines classification. | Written after the regulatory strategy. | Statement |
| Indication for use | The clinical condition and population the device is intended for. | Treated as marketing copy. | Statement |
| Classification | The risk category that determines the conformity route. | Assumed to be the same across markets. | Class I, IIa, IIb, III; I, II, III |
| Conformity assessment | The process of demonstrating compliance. | Confused with the technical file. | Route |
| Notified body | An EU organisation that assesses conformity for higher-risk devices. | Assumed to be optional. | NANDO number |
| Predicate | A legally marketed device used as a comparison in a 510(k). | Confused with clinical evidence. | Device identifier |
| Substantial equivalence | The 510(k) standard of comparison to a predicate. | Treated as proof of safety and effectiveness. | Regulatory finding |
| De Novo | A classification route for a novel low-to-moderate risk device. | Confused with 510(k). | Grant number |
| Premarket approval | The route for high-risk devices with clinical evidence. | Used generically for any approval. | PMA number |
| Product code | The FDA classification code for a device type. | Confused with the device's own identifier. | Three-letter code |
| 510(k) summary | The public summary of a clearance. | Treated as the full submission. | Document |
| Special 510(k) | A streamlined route for certain modifications. | Assumed to cover any change. | Submission type |

### 4.4 Evidence and policy terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Clinical evaluation | The EU process of assessing clinical evidence for a device. | Confused with a clinical trial. | Report |
| Performance evaluation | The IVD equivalent of clinical evaluation. | Applied to medical devices. | Report |
| Real-world evidence | Evidence from routine care data. | Treated as a study design. | Evidence type |
| Predetermined change control plan | A pre-authorised description of anticipated model changes. | Assumed to permit any change. | PCCP |
| Good machine learning practice | Joint regulatory principles for ML device development. | Treated as a standard. | Principles |
| Post-market surveillance | Systematic monitoring after placing on the market. | Confused with complaint handling. | Process |
| Vigilance | Reporting of serious incidents to the authority. | Confused with surveillance. | Report |
| Recall | Removal or correction of a device in the field. | Confused with a safety notice. | Recall class |

## 5. The mechanics, step by step

The procedure has 22 steps in five phases. It is longer than the source-evaluation procedure in topic 01 because a regulatory statement has four coordinates and each one needs its own check. The output is a citation that names the instrument, the provision, the version, the jurisdiction, and the date, plus a note on what would change the answer.

A regulatory question is usually phrased as "are we allowed to", and the honest answer is often "it depends on the classification, the intended use, and the date". The procedure is designed to make those dependencies explicit rather than to produce a false yes.

| Phase | Question answered | Typical owner | Artefact |
|---|---|---|---|
| A Frame | What decision are we actually making? | Product or regulatory lead | Decision statement |
| B Identify | Which instrument governs, and is it current? | Regulatory lead | Instrument and status |
| C Dates | When does it apply, and what is the deadline? | Regulatory lead | Date table |
| D Scope | Does it apply to us, and how binding is it? | Regulatory and legal | Scope memo |
| E Record | What do we cite, and when do we check again? | Regulatory lead | Ledger entry |
| F Cross-check | What would a reviewer challenge? | Second reviewer | Review note |

### Phase A — Frame the question

**Step 1. Write the question as a decision with an actor, a product, and a market.**
Input: the product, the intended use, the markets, and the decision deadline.
Output: one sentence naming the actor, the product and intended use, the market, and the action being considered.
Rationale: "is this compliant" is not answerable. "May we place this device on the EU market for this intended use in Q3 2027 under the transition provisions" is answerable.
Branch: if the question spans markets, split it by jurisdiction before proceeding.
Edge case: a product can be a device in one market and a consumer product in another, which changes every subsequent step.

**Step 2. Identify the product's regulatory category in each market.**
Input: the intended use statement, the technology, and the market.
Output: device or not, the risk class, and the conformity route.
Rationale: category determines which instrument applies, and the same product can be classified differently across markets.
Branch: if the product is software, check the jurisdiction's software-specific guidance before assuming it is a device.
Edge case: a product with multiple functions can have different classifications for each function.

**Step 3. Identify the decision's evidence needs.**
Input: the question and the category.
Output: the evidence the regulator will expect, such as analytical validation, clinical evaluation, or performance evaluation.
Rationale: the evidence question determines which guidance and standards matter, and it prevents a regulatory search from becoming a general literature search.
Branch: if the evidence need is clinical, the relevant guidance is the clinical evaluation guidance, not the software guidance.
Edge case: a change to an already-marketed device may need a new submission or may fit within a predetermined change control plan.

**Step 4. Set the review date and the horizon.**
Input: the project calendar.
Output: an ISO review date and the date by which the answer must still hold.
Rationale: regulatory answers have a shelf life measured in months, and a launch that is lawful today may not be lawful at the planned date.
Branch: if the horizon crosses a known transition deadline, plan for both regimes.
Edge case: an implementing act can change the practical date after the plan is set.

### Phase B — Identify the instrument and its status

**Step 5. Find the instrument at the issuer, not in a summary.**
Input: the question and the jurisdiction.
Output: the statute, regulation, or guidance that governs, with its official identifier.
Rationale: summaries are useful for orientation and unreliable for citation. The issuer's register is the only place where status is authoritative.
Branch: if the only available source is a summary, use it to find the instrument and cite the instrument.
Edge case: some requirements live in national implementing measures that are not on the EU-level page.

**Step 6. Determine the instrument's status.**
Input: the official identifier and the issuer's register.
Output: one of the states in the table below.
Rationale: status determines whether the document can be cited as current policy.
Branch: a document can be final and not yet applicable, which is a distinct state from draft.
Edge case: an agency can withdraw a guidance without a replacement, leaving a gap that is itself informative.

| Status | Meaning | Citation action |
|---|---|---|
| Draft | Proposed, open for comment | Cite as a proposal, never as a requirement |
| Final, not yet applicable | Adopted but the obligations start later | Cite with the application date |
| In force | Obligations apply | Cite with the version and date |
| In force, transition open | Older route still available for a period | Cite both regimes and the deadline |
| Superseded | Replaced by a later version | Replace the citation |
| Withdrawn | Removed without replacement | Do not cite as policy; note the gap |
| Consolidated | Reading version incorporating amendments | Cite the amendments separately |

**Step 7. Check for amendments and consolidated versions.**
Input: the instrument's amendment history.
Output: a list of amendments that affect the provision relied on, with dates.
Rationale: a regulation can be amended without a new number, and the consolidated version is a reading aid rather than the legal text.
Branch: if an amendment changed the provision, cite the amending instrument as well as the consolidated version.
Edge case: a corrigendum can change a date or a threshold without an amendment.

**Step 8. Check the guidance attached to the instrument.**
Input: the agency's guidance list for the topic.
Output: the current guidance documents, their dates, and their status.
Rationale: guidance explains how the agency reads the regulation, and it is often revised more often than the regulation.
Branch: if the guidance predates a major amendment, treat it as unreliable for the amended provision.
Edge case: multiple guidance documents can cover the same topic for different product types.

**Step 9. Check the standards that the instrument cites.**
Input: the harmonised or recognised standard list.
Output: the standard number, edition, and the instrument that cites it.
Rationale: a standard creates a presumption of conformity only when it is cited by the instrument and listed in the official publication.
Branch: if the standard is not cited, it is a voluntary reference, not a compliance route.
Edge case: a standard can be withdrawn from the citation list, which removes the presumption even though the standard still exists.

### Phase C — Dates and transition

**Step 10. Separate publication, entry into force, and application.**
Input: the instrument's preamble and the official journal notice.
Output: three dates, each labelled.
Rationale: planning against the wrong date is the most common scheduling error in regulatory work.
Branch: if only one date is stated, check the instrument's articles for the application provision.
Edge case: different articles can have different application dates within one instrument.

**Step 11. Identify the transition provisions.**
Input: the instrument's transitional articles and any amending regulation.
Output: the conditions, the deadline, and the products eligible.
Rationale: a transition is not a grace period for new products; it usually applies to products already placed on the market or to certificates issued before a date.
Branch: if the product is new, the transition may not apply at all.
Edge case: transition deadlines differ by risk class, which is common in the EU device regulations.

**Step 12. Compute the practical deadline.**
Input: the transition conditions and the project schedule.
Output: a date by which the new route must be available, with a buffer.
Rationale: the legal deadline is not the operational deadline; notified body capacity and submission queues consume months.
Branch: if the buffer cannot be met, the product plan needs to change, not the citation.
Edge case: a deadline can be extended by a later amendment, which is exactly what happened with the EU device regulations.

| Date type | What it governs | Typical planning error |
|---|---|---|
| Publication | When the text becomes public | Treated as the compliance date |
| Entry into force | When the instrument becomes legally valid | Treated as the application date |
| Application | When the obligations begin | Missed when articles differ |
| Transition end | When the older route closes | Assumed to apply to new products |
| Certificate expiry | When a notified body certificate lapses | Confused with the transition deadline |
| Implementing act date | When the practical detail applies | Not monitored after the parent regulation |

**Step 13. Check for a later instrument that changes the date.**
Input: the issuer's news and the official journal.
Output: any amendment, delegated act, or implementing act that changes the timeline.
Rationale: dates are the most volatile part of a regulatory source, and secondary summaries lag them.
Branch: if a change is proposed but not adopted, record it as a risk rather than a fact.
Edge case: a political agreement is not an adopted instrument.

### Phase D — Scope, bindingness, and jurisdiction

**Step 14. Confirm the jurisdiction and any sub-national requirements.**
Input: the market and the product.
Output: the applicable jurisdictions and any additional requirements.
Rationale: federal systems can have state or provincial requirements that add to the national ones.
Branch: if the product is marketed in several countries, build a matrix rather than a single answer.
Edge case: the location of the manufacturer does not determine the jurisdiction; the market does.

**Step 15. Confirm the scope: product, population, and intended use.**
Input: the instrument's scope article and the product's intended use.
Output: a statement of whether the product falls inside the scope, with the reasoning.
Rationale: scope is where most "the regulation says" claims fail, because the product is outside the scope or the intended use is different.
Branch: if the product is at the boundary, look for the agency's boundary guidance.
Edge case: accessories and software can have their own scope rules.

**Step 16. Determine whether the provision is binding or interpretive.**
Input: the instrument type and the specific provision.
Output: binding requirement, presumption of conformity, recommendation, or commentary.
Rationale: the obligation layer determines what a failure means and what a citation can claim.
Branch: a guidance can describe a requirement without creating one; cite the requirement and the guidance separately.
Edge case: a guidance can be made binding through a contract, a payer policy, or an accreditation standard, even when the law does not bind it.

| Instrument | Binding on its own | Creates presumption of conformity | Typical use in a citation |
|---|---|---|---|
| Statute or regulation | Yes | Not applicable | The obligation |
| Guidance | No | No | The agency's interpretation |
| Harmonised standard | No | Yes, when cited in the OJ | The technical route |
| Recognised consensus standard | No | Yes, for the declared use | The technical route in the US |
| MDCG guidance | No | No | Interpretation in the EU |
| IMDRF document | No | No | International context |

**Step 17. Check enforcement and guidance practice.**
Input: warning letters, inspection findings, withdrawal notices, and published decisions.
Output: a statement of how the authority has applied the provision in practice.
Rationale: enforcement practice is evidence of interpretation and can differ from the text's plain reading.
Branch: if enforcement is inconsistent, say so rather than presenting one reading as settled.
Edge case: enforcement priorities change with leadership and resources.

**Step 18. Check the international layer where relevant.**
Input: the IMDRF documents and the jurisdiction's participation.
Output: a note on whether international guidance is adopted, referenced, or ignored.
Rationale: IMDRF documents can be influential without being binding, and the same document can have different status in different markets.
Branch: if a jurisdiction has not adopted the document, it is context rather than requirement.
Edge case: a standard can be adopted internationally and cited in one jurisdiction only.

### Phase E — Record and monitor

**Step 19. Write the citation in five parts.**
Input: the instrument, provision, version, jurisdiction, and date.
Output: a citation such as "Regulation (EU) 2017/745, Article 120, as amended by Regulation (EU) 2023/607, transition to 31 December 2027".
Rationale: the five parts make the claim checkable and expose missing coordinates.
Branch: if a part is unknown, write "not established" rather than omitting it.
Edge case: a consolidated version should be cited alongside the amending instrument, not instead of it.

**Step 20. State the residual uncertainty.**
Input: the checks above.
Output: a sentence naming what is not settled, such as a pending implementing act or an unclear classification.
Rationale: regulatory answers are rarely binary, and a stated uncertainty is more useful than a confident error.
Branch: if the uncertainty is material to the decision, escalate to counsel or to the authority.
Edge case: an informal answer from an authority is useful context and is not a binding position.

**Step 21. Record the version and the access date.**
Input: the sources consulted.
Output: a ledger entry with the URL, the document number, the version, and the access date.
Rationale: the record is what makes the next review cheap and the current answer auditable.
Branch: if the source is a PDF, record the document number and date from the cover page.
Edge case: a page can change without a new document number, which is why the access date matters.

**Step 22. Set the monitoring trigger.**
Input: the instrument class and the project horizon.
Output: a review interval and a list of events that force an earlier check.
Rationale: regulatory status changes on a legislative calendar, not on a fixed schedule.
Branch: the triggers should include a new draft, a new amendment, a transition deadline, and a change of government.
Edge case: an election can change the implementation of an adopted instrument.

The procedure produces a citation that a lawyer can check and a decision that a product team can act on. It does not produce legal advice, and it should not be used to resolve a genuinely ambiguous question without qualified review.

### Phase F — Cross-checks before the answer leaves the team

**Step 23. Check the classification against a second source.**
Input: the classification decision and the jurisdiction's classification rules.
Output: agreement or a documented difference, with the reasoning.
Rationale: classification drives every downstream requirement, and a classification error invalidates the rest of the analysis.
Branch: if two reasonable readings produce different classes, document both and take the more conservative route for planning.
Edge case: a classification can change when the intended use changes, even if the technology does not.

**Step 24. Check the intended use statement against the evidence and the label.**
Input: the intended use, the indications, the labelling, and the clinical evidence.
Output: a statement of consistency.
Rationale: regulators assess the device against its stated intended use, and an inconsistent label creates a compliance gap that no amount of evidence fixes.
Branch: if the label claims more than the evidence supports, narrow the label or collect evidence.
Edge case: a software function described in marketing but not in the intended use can still be assessed as part of the device.

**Step 25. Check the evidence guidance version.**
Input: the guidance that governs clinical or analytical evidence.
Output: the current version and its date.
Rationale: evidence expectations change more often than the statute, and a study designed to an old guidance can fail a current review.
Branch: if the guidance changed after the study was designed, assess the gap before submission.
Edge case: guidance can be issued as a draft and applied informally before finalisation.

**Step 26. Check whether a pilot, sandbox, or early-access programme applies.**
Input: the jurisdiction's current programmes.
Output: whether the product is eligible and what the programme permits.
Rationale: programmes can create a lawful route that is not obvious from the regulation alone.
Branch: if the programme is time-limited, record the end date.
Edge case: participation can create disclosure obligations that affect marketing.

**Step 27. Check the authority's FAQ and speeches for a contradictory position.**
Input: the agency's FAQ page and recent public statements.
Output: a note on any position that differs from the guidance.
Rationale: informal positions often precede formal guidance and can signal a change in interpretation.
Branch: if the FAQ contradicts the guidance, cite the guidance and note the difference.
Edge case: an FAQ can be removed without notice, so archive it with a date.

**Step 28. Check a second jurisdiction if the product is global.**
Input: the markets in scope.
Output: a matrix of requirements and dates by market.
Rationale: a single-market answer becomes a portfolio risk when the product is launched elsewhere.
Branch: if one market is stricter, use it as the design baseline where feasible.
Edge case: mutual recognition agreements can reduce duplication without eliminating differences.

### A branch worked in miniature: the word "must"

A guidance document states that a manufacturer "must" validate the model on an independent dataset. The team reads this as a legal requirement and plans a validation study. Two branches follow.

If the guidance interprets a regulation that contains an equivalent requirement, the obligation exists and the guidance explains how to meet it. The citation should name the regulation and use the guidance as the interpretation. The validation study is required.

If the guidance is describing a recommended practice with no equivalent obligation, the "must" is the agency's expectation rather than a legal requirement. The team may still choose to validate, because the agency will look for it during review, and the citation should say that the guidance recommends it rather than that the law requires it.

The distinction matters because it changes what happens if the team does not comply. A legal requirement creates enforcement exposure; a recommendation creates review risk. Both are real, and they are not the same, and a document that calls both "required" prevents the reader from making the distinction.

### Common citation defects

- The instrument number is missing and only the popular name is given.
- The article or section is missing, so the claim cannot be located in the text.
- The version is missing, so an amended provision is cited as if unchanged.
- The jurisdiction is implied rather than stated.
- The date is the publication date rather than the application date.
- A guidance is cited where a regulation is meant, or the reverse.
- A transition provision is omitted, making a lawful route look non-compliant.

Each defect is visible to a reviewer in under a minute, and each is fixed by adding a coordinate rather than by rewriting the argument.

### What this procedure does not decide

- Whether a specific product is a device. That is a classification judgement with legal consequences.
- Whether a clinical claim is permitted in labelling. That depends on the authorised indications.
- Whether a contract or payer policy adds requirements beyond the regulation.
- Whether an authority will accept a particular evidence package.

The procedure establishes what the sources say and when they say it. The judgement about the product belongs to the regulatory professional, and the legal conclusion belongs to counsel. Separating the two is what allows the source work to be reviewed on its own terms.

A useful discipline is to write the source note before writing the recommendation. If the note cannot be written, the recommendation is premature. If the note can be written but the recommendation does not follow from it, the gap is a judgement that should be labelled as one.

| Signal | Why it needs escalation | To whom |
|---|---|---|
| Classification is borderline | Classification determines the entire route | Regulatory lead and counsel |
| Two instruments appear to conflict | Priority rules may depend on the specific provisions | Counsel |
| A transition provision may not apply | Eligibility is fact-specific | Regulatory lead |
| An authority has issued an informal contrary position | Enforcement risk is unclear | Regulatory lead |
| The claim will appear in labelling or promotion | Regulatory and advertising rules interact | Regulatory and legal |

Two further checks apply to any answer that will be reused. Archive the issuer page with a date, because a register page can change without a new document number. And record the person who verified it, because regulatory status is a judgement made by someone accountable, not a property of a URL.

A regulatory record that names the instrument, the provision, the version, the jurisdiction, the date, the access date, and the reviewer is auditable a year later. One that names a title and a year is not.

The record should also say what was not checked. A note such as "national implementing measures not reviewed" is more useful than silence, because it tells the next reviewer where to start and prevents a false sense of completeness.

## 6. Worked example

A regulatory lead is preparing a board paper on the European Union launch of an AI-enabled diagnostic. The sentence in the draft reads: "High-risk obligations under the EU Artificial Intelligence Act apply to our device from August 2027." The board is being asked to approve a budget that assumes twelve months of preparation before that date.

### Framing the claim

The sentence contains four coordinates, and only one of them is explicit:

1. The instrument: the Artificial Intelligence Act, Regulation (EU) 2024/1689.
2. The obligation: high-risk obligations for a product covered by Annex I, which includes medical devices.
3. The jurisdiction: the European Union.
4. The date: August 2027.

The claim type is regulatory, so the minimum source is the legal text or the Commission's own statement of the timeline. A law firm note, however reputable, is a secondary source for a date.

The claim also carries an unstated assumption: that the product is high-risk under the Act. That depends on whether it is a safety component of a device, or is itself a device, and whether it requires third-party conformity assessment. The date question is only relevant if the classification assumption holds.

| Classification question | What determines the answer | Where to check |
|---|---|---|
| Is it a device? | Intended use and the definition in MDR or IVDR | The regulation and the Commission's device guidance |
| Is it a safety component? | Whether it performs a safety function or is essential to the device's function | The device's technical documentation |
| Does it require third-party assessment? | The device class and the conformity route | The classification rules |
| Is it high-risk under the AI Act? | Device status plus third-party assessment | Article 6 and Annex I |
| Are any obligations excluded? | Article 5(5) in-house exemption, if applicable | The relevant article |

### Candidate sources

| # | Candidate | Type | Date | Role |
|---|---|---|---|---|
| C1 | Law firm client alert on the AI Act and devices | Secondary commentary | 2025 | The likely origin of the 2027 date |
| C2 | European Commission AI Act policy page | Official overview | Current | Orientation, not a timetable |
| C3 | Commission AI Act Service Desk implementation timeline | Official timeline | Current | The date source |
| C4 | MDCG 2025-6 FAQ on MDR/IVDR and the AI Act | Official guidance | June 2025 | Interplay guidance, predates the amendment |
| C5 | Regulation (EU) 2024/1689, Article 113 | Legal text | 2024 | The original application provision |
| C6 | National competent authority page | National implementation | Current | Enforcement context |

### Scoring the candidates

**C1, the law firm alert.**
- Authority: moderate. The firm is competent and independent of the product, but the document is commentary.
- Currency: likely stale. It predates the amendment that changed the timeline.
- Applicability: general, not product-specific.
- Accessibility: public.
- Provenance: the reasoning is visible, which makes the error traceable.
- Verdict: background only. It explains why the 2027 date circulated and cannot establish the current date.

**C2, the Commission policy page.**
- Authority: high, as the issuer's overview.
- Currency: current, but the page is an overview and may not carry every date.
- Applicability: general.
- Accessibility: public.
- Provenance: official.
- Verdict: orientation only; it points to the timeline and the legal text.

**C3, the Service Desk timeline.**
- Authority: high. It is the Commission's own implementation resource.
- Currency: current, and it states that it reflects the amendments introduced by the Digital Omnibus on AI.
- Applicability: covers Annex I products, which includes medical devices.
- Accessibility: public.
- Provenance: official, with the amendment noted.
- Verdict: primary support for the date, with the caveat that it is a Commission service page rather than the legal text.

**C4, MDCG 2025-6.**
- Authority: high for the interplay between the regulations.
- Currency: predates the amendment, so its dates are unreliable.
- Applicability: directly relevant to device-AI interplay.
- Accessibility: public PDF.
- Provenance: jointly endorsed by the MDCG and the AI Board.
- Verdict: primary support for the interplay analysis, not for the date.

**C5, Article 113.**
- Authority: highest, as the legal text.
- Currency: the article as originally adopted stated 2 August 2027 for Annex I; the amendment changes the operative date.
- Applicability: general.
- Accessibility: public through EUR-Lex.
- Provenance: official.
- Verdict: cite only with the amending instrument, because the article alone is now misleading.

**C6, the national authority page.**
- Authority: high for national enforcement.
- Currency: variable; national pages lag EU-level changes.
- Applicability: one member state.
- Accessibility: public.
- Provenance: official.
- Verdict: context, not the date source.

### The date arithmetic

The difference between the two candidate dates is straightforward:

- Original Annex I date in Article 113: 2 August 2027.
- Date stated in the Commission timeline after the amendment: 2 August 2028.
- Difference: 365 days.

A one-year difference is material for a board decision. It changes the budget year, the hiring plan, and the sequencing with notified body engagement. It is also the kind of difference that a secondary source can carry for months without anyone noticing, because the original date was correct when it was written.

### Verdict

The sentence is not supportable as written at the review date of 26 October 2026. The Commission's implementation timeline, which states that it reflects the Digital Omnibus amendments, lists 2 August 2028 for high-risk AI embedded in products covered by Annex I. The 2027 date is the original Article 113 date, not the current operative date.

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| High-risk obligations apply from August 2027 | Reject | "The original Article 113 date was 2 August 2027; the Commission timeline now lists 2 August 2028." |
| The product is high-risk | Unresolved | "If the product is a safety component requiring third-party assessment, high-risk obligations apply." |
| Device requirements also apply | Pass | "MDR obligations apply independently of the AI Act." |
| A budget decision is needed now | Out of scope | A management judgement, not a regulatory fact |

The replacement sentence is longer and, for a board paper, more useful. It separates the regulatory fact from the classification assumption and from the business decision.

### Two diagnostics that would change the verdict

**Diagnostic 1: the amending instrument is annulled or not in force.** If the amendment that moved the date is invalidated, or if its entry into force is delayed, the original Article 113 date of 2 August 2027 would revive. The check is the status of the amending instrument in the Official Journal, not the Commission's summary page. This diagnostic matters because a timeline page can be updated before the underlying instrument is settled.

**Diagnostic 2: a sector-specific implementing act sets an earlier date for this device type.** The AI Act's application dates are general. If an implementing act, or a sectoral rule for a specific device category, sets an earlier or additional obligation, the practical date for this product changes. The check is the implementing acts adopted under the Act, which are easy to miss because they are published separately from the regulation.

A third diagnostic, which would change the recommendation rather than the verdict, is a notified body capacity constraint that makes the nominal date unachievable. It does not change the law and it changes the plan.

### What the team should record

The ledger entry names the instrument, the article, the amending instrument, the Commission timeline, and the access date. It records that the classification assumption is unresolved and that the date depends on an amendment whose status should be rechecked before the board paper is finalised.

The review trigger is any new implementing act, any change to the Commission timeline, and the next European elections, because implementation priorities can change with a new Commission.

The case illustrates the general rule for this topic: a regulatory date is a claim about a moving object, and the best source is the issuer's current statement plus the legal instrument that supports it. A single secondary source, however reputable, is not enough for a date that drives a budget.

### Why the 2027 date persisted

The 2027 date was correct when the Act was adopted, and it appeared in hundreds of summaries, slide decks, and compliance checklists. Each of those documents was accurate at the time of writing, and none of them was updated when the timeline changed.

The persistence mechanism is worth naming, because it is the same in every organisation. A summary is written once, circulated widely, and then treated as a stable fact. The primary source changes, and the summary becomes a fossil that is quoted by people who have no reason to doubt it.

The only defence is to cite the primary source and to date the citation. A sentence that says "as of October 2026, the Commission timeline lists 2 August 2028" is self-dating, and a reader can see when it needs rechecking. A sentence that says "applies from August 2027" carries no such signal.

### What would not change the verdict

- A law firm note repeating the 2027 date. Repetition is not independent evidence.
- A conference presentation from 2025. It reflects the timetable at the time.
- A vendor's compliance roadmap. It may be based on the same stale summary.
- An unofficial countdown or calendar tool. It is a derivative, not a source.

### Citation forms compared

| Form | What it implies | Risk |
|---|---|---|
| "EU AI Act, August 2027" | The original date is current | Wrong if the amendment is in force |
| "EU AI Act, August 2028" | The current date, without support | Unverifiable without the timeline |
| "Commission timeline, accessed 26 October 2026" | A dated official statement | Needs the amending instrument for legal certainty |
| "Article 113 as amended, 2 August 2028" | The legal position | Requires reading the amendment |

The last form is the strongest and the hardest to produce. The third is acceptable for a board paper, provided the access date is stated and the classification assumption is flagged.

For a board paper, the practical recommendation is the third form plus a footnote naming Article 113 and the amending instrument. That gives the reader a dated source and a path to the legal text without requiring the board to read the regulation.

### What the board actually needs

- The current operative date, with its source and access date.
- The assumption that makes the date relevant, stated explicitly.
- The cost of the date moving by twelve months.
- The trigger that would change the answer.

A paper that provides those four items is more useful than one that provides a single confident date, because it lets the board see the sensitivity of the decision to a fact that is outside the company's control.

## 7. Code, query, or calculation

The script below turns the status check into a repeatable calculation. Each instrument carries its issue date, its application date, and optional transition and supersession fields. The classifier returns a status and an action at the review date.

```python
from collections import Counter
from dataclasses import dataclass
from datetime import date

REVIEW = date(2026, 10, 26)


@dataclass
class Instrument:
    name: str
    issued: date
    applies_from: date
    status: str = "final"          # draft | final | withdrawn
    superseded_by: str = ""
    transition_until: date | None = None


def classify(i, review=REVIEW):
    if i.status == "withdrawn":
        return "withdrawn", "do not cite as policy"
    if i.status == "draft":
        return "draft", "cite as proposal only"
    if i.superseded_by:
        return "superseded", f"replace with {i.superseded_by}"
    if review < i.applies_from:
        return "not yet applicable", f"applies in {(i.applies_from - review).days} days"
    if i.transition_until and review < i.transition_until:
        return "in force, transition open", f"{(i.transition_until - review).days} days remaining"
    return "in force", "cite with version and date"


INSTRUMENTS = [
    Instrument("EU MDR 2017/745", date(2017, 5, 5), date(2021, 5, 26),
               transition_until=date(2028, 12, 31)),
    Instrument("EU IVDR 2017/746", date(2017, 5, 5), date(2022, 5, 26),
               transition_until=date(2029, 5, 26)),
    Instrument("EU AI Act 2024/1689, Annex I high-risk", date(2024, 7, 12),
               date(2028, 8, 2)),
    Instrument("EU AI Act 2024/1689, prohibitions", date(2024, 7, 12),
               date(2025, 2, 2)),
    Instrument("FDA QMSR (21 CFR 820)", date(2024, 2, 2), date(2026, 2, 2)),
    Instrument("FDA PCCP final guidance", date(2024, 12, 3), date(2024, 12, 3)),
    Instrument("MDCG 2025-6", date(2025, 6, 19), date(2025, 6, 19)),
    Instrument("FDA 2023 AI/ML draft guidance", date(2023, 4, 3),
               date(2023, 4, 3), status="draft"),
    Instrument("Hypothetical 2019 device guidance", date(2019, 1, 1),
               date(2019, 7, 1), superseded_by="2025 revision"),
]

rows = [(i.name,) + classify(i) for i in INSTRUMENTS]
print(f"review date {REVIEW.isoformat()}   instruments {len(rows)}")
print("-" * 96)
print(f"{'instrument':44s} {'status':26s} action")
print("-" * 96)
for name, status, action in rows:
    print(f"{name[:44]:44s} {status:26s} {action}")
print("-" * 96)
for status, n in Counter(r[1] for r in rows).most_common():
    print(f"{n:2d}  {status}")
```

Expected output:

```text
review date 2026-10-26   instruments 9
------------------------------------------------------------------------------------------------
instrument                                   status                     action
------------------------------------------------------------------------------------------------
EU MDR 2017/745                              in force, transition open  797 days remaining
EU IVDR 2017/746                             in force, transition open  943 days remaining
EU AI Act 2024/1689, Annex I high-risk       not yet applicable         applies in 646 days
EU AI Act 2024/1689, prohibitions            in force                   cite with version and date
FDA QMSR (21 CFR 820)                        in force                   cite with version and date
FDA PCCP final guidance                      in force                   cite with version and date
MDCG 2025-6                                  in force                   cite with version and date
FDA 2023 AI/ML draft guidance                draft                      cite as proposal only
Hypothetical 2019 device guidance            superseded                 replace with 2025 revision
------------------------------------------------------------------------------------------------
 4  in force
 2  in force, transition open
 1  not yet applicable
 1  draft
 1  superseded
```

Three results are worth reading carefully. The MDR and IVDR are in force and still in transition, which is why a product can be lawfully on the market under an older route while the new obligations already apply to new products. The AI Act Annex I entry is not yet applicable at the review date, although other parts of the same regulation are in force, which is the distinction between an instrument and its articles. And the draft guidance is classified as a proposal even though it has a 2023 date, because status is not the same as age.

To adapt the script, add each instrument your programme depends on, including the amending instruments, and update the review date at each cycle. The classifier does not read the instrument; it encodes a decision that a person must still verify at the issuer's register.

A useful extension is to add a `last_verified` field and a `source_url` field, then print the age of each verification. A register that has not been touched in a year is a register that is describing the past.

The script also makes a subtle point visible. Two instruments with the same issue date can have different statuses because one has a transition and the other does not. The status is a property of the instrument and the date, not of the instrument alone.

## 8. Data, standards, and regulatory context

### 8.1 The United States

Device regulation in the United States rests on the Federal Food, Drug, and Cosmetic Act and the regulations at Title 21 of the Code of Federal Regulations. The FDA issues guidance explaining how it interprets the statute and the regulations. Guidance is generally non-binding, and it is the document most often cited as if it were law.

| Element | Instrument | Status | What it constrains |
|---|---|---|---|
| Statute | Federal Food, Drug, and Cosmetic Act | Binding | The legal basis for device regulation |
| Quality regulation | 21 CFR Part 820, QMSR from 2 February 2026 | Binding | Quality management for device manufacturers |
| Clearance route | 510(k) | Binding procedure | Marketing with a predicate comparison |
| Novel route | De Novo | Binding procedure | Classification of novel lower-risk devices |
| Approval route | Premarket approval | Binding procedure | High-risk devices with clinical evidence |
| Guidance | FDA guidance documents | Non-binding | The agency's interpretation |
| Standards | FDA standards recognition | Non-binding on its own | A declared route to conformity |
| Change control | Predetermined change control plan | Optional, authorised | Pre-approved model updates |

The Quality Management System Regulation became effective on 2 February 2026 and incorporates ISO 13485:2016 by reference, with supplementary provisions retained in Part 820. A document written before that date that describes the US quality requirement in the older Quality System Regulation terms is out of date, even if it was correct when written.

### 8.2 The European Union

The EU system layers a regulation, implementing acts, harmonised standards, and guidance. The Medical Device Regulation and the In Vitro Diagnostic Regulation are directly applicable, which means member states do not transpose them, though they do adopt national implementing measures for parts of the system.

| Element | Instrument | Status | What it constrains |
|---|---|---|---|
| Medical devices | Regulation (EU) 2017/745 | Binding | Placing devices on the EU market |
| In vitro diagnostics | Regulation (EU) 2017/746 | Binding | Placing IVDs on the EU market |
| Artificial intelligence | Regulation (EU) 2024/1689 | Binding | AI systems, including device AI, by application date |
| Guidance | MDCG documents | Non-binding | Interpretation of the regulations |
| Standards | Harmonised standards cited in the Official Journal | Presumption of conformity | Technical compliance route |
| Notified bodies | Designated organisations | Binding role | Conformity assessment for higher-risk devices |
| Database | Eudamed | Binding registration | Device and certificate information |

Transition provisions matter more in the EU than in the US because the regulations replaced earlier directives and the transition has been extended more than once. The deadlines are staggered by risk class, so a single "MDR deadline" is always a simplification. The script in section 7 uses the latest deadline for each regulation and labels it as a transition window rather than a single date.

The Artificial Intelligence Act adds a second layer. A device can be compliant with the MDR and still be subject to AI Act obligations, and the AI Act applies on its own timetable. The MDCG and the AI Board issued joint guidance in June 2025 on the interplay, which is the right starting point for the relationship between the two regimes.

### 8.3 International standards

| Standard | Subject | Typical regulatory role |
|---|---|---|
| ISO 13485:2016 | Quality management systems for medical devices | Incorporated by reference in the US QMSR; harmonised in the EU |
| ISO 14971:2019 | Application of risk management to medical devices | The reference risk-management standard |
| IEC 62304 | Medical device software life-cycle processes | Software development and maintenance |
| IEC 62366-1 | Usability engineering | Formative and summative usability work |
| IEC 82304-1 | Health software products | Software that is not itself a device |
| ISO/IEC 27001 | Information security management | Security management, often by contract |

A standard is a technical document, and its legal effect comes from the instrument that cites it. ISO 14971 is not law; a regulation that requires risk management makes it relevant, and a harmonised-standard citation gives it a presumption of conformity. The distinction is the same one that applies to guidance, and it is the most common source of overclaiming in compliance documents.

### 8.4 What regulatory sources do not tell you

- Whether your specific product is inside the scope.
- Whether your evidence is sufficient for a particular reviewer.
- Whether a payer will cover the product.
- Whether a hospital will adopt it.
- Whether the clinical claim is true.

Those are separate questions with separate sources. A regulatory clearance is a permission to market, not a finding of clinical benefit, and a compliance programme that treats it as the latter will overstate the product in every downstream conversation.

### 8.5 The device-AI stack

A single AI-enabled device can be subject to four layers of requirements at once. Knowing which layer a sentence refers to prevents the most common category error in regulatory writing.

| Layer | Question it answers | Typical source | Binding |
|---|---|---|---|
| Device law | Is it a device, and what route applies? | MDR, IVDR, FD&C Act, 21 CFR | Yes |
| AI law | Is the AI system high-risk, and what obligations follow? | AI Act, implementing acts | Yes, on its own timetable |
| Standards | How is compliance demonstrated technically? | ISO 13485, ISO 14971, IEC 62304 | Through the instrument |
| Guidance | How does the agency interpret the requirement? | FDA guidance, MDCG documents | No |

A sentence such as "the AI Act requires a quality management system" conflates the layers. The AI Act requires a quality management system for high-risk systems, and a device manufacturer already has one under the MDR and ISO 13485. The practical question is whether the existing system covers the AI-specific elements, not whether a second system is needed.

### 8.6 US marketing pathways compared

| Pathway | Typical use | Evidence | Change control |
|---|---|---|---|
| 510(k) | Moderate-risk device with a predicate | Substantial equivalence, often bench and limited clinical | New submission for many changes; PCCP possible |
| De Novo | Novel device without a predicate, lower risk | Risk-benefit evidence and special controls | New submission or supplement |
| Premarket approval | High-risk device | Clinical evidence of safety and effectiveness | Supplements for changes |
| Exempt | Low-risk class I devices | None | Quality system still applies |

The pathway determines the evidence burden and the change-control strategy, and it is the first thing to establish before writing a regulatory sentence. A claim that is true for a 510(k) device may be false for a PMA device, and the reverse.

### 8.7 How to read a regulation article

- Read the article with its definitions, which are usually at the start of the instrument.
- Check whether the article has been amended or replaced.
- Check the application provision, which may set a different date for the article.
- Check the transitional articles, which may preserve the old rule for some products.
- Check the implementing acts, which may supply the detail the article omits.
- Check the annexes, which often contain the classification rules or the technical requirements.
- Record the article number and the version, not only the regulation number.

The last point is the one that most often goes wrong. "MDR Article 120" is a precise citation only if the reader knows which version of Article 120 is in force, and that requires the amendment history.

### 8.8 What changes most often

| Element | Change frequency | Why it changes | What to monitor |
|---|---|---|---|
| Statute | Low | Legislative amendment | Official journal |
| Regulation text | Low to moderate | Amending regulations | Consolidated versions |
| Application dates | Moderate | Amending and implementing acts | Commission timelines |
| Guidance | High | New drafts and revisions | Agency guidance database |
| Standards list | Moderate | New editions and citations | Official journal and FDA list |
| Enforcement practice | High | Priorities, leadership, resources | Warning letters and decisions |
| Fees and timelines | High | Agency capacity and policy | Agency performance pages |

The practical implication is that monitoring effort should follow change frequency. A team that rechecks the statute annually and never checks guidance has inverted the effort.

## 9. Industry practice and case evidence

The cases below are documented public facts, each with a date. The inferences are labelled.

**Quality Management System Regulation (documented).** The FDA's QMSR became effective on 2 February 2026, two years after the final rule, and incorporates ISO 13485:2016 by reference. The agency also changed its inspection programme and stopped using the older inspection technique on the same date. The practical lesson is that a regulatory change can alter both the requirement and the way it is inspected, so a document that updates only the requirement is incomplete.

**Predetermined change control plans (documented).** The FDA issued final guidance on PCCPs for AI-enabled device software functions on 3 December 2024. The guidance describes three components: a description of modifications, a modification protocol, and an impact assessment. The lesson is that a regulatory pathway can be optional and still change the design of a quality system, because the ability to update a model without a new submission depends on planning the update process in advance.

**MDCG 2025-6 (documented).** The Medical Device Coordination Group and the AI Board jointly endorsed a FAQ on the interplay between the MDR, the IVDR, and the AI Act in June 2025. The document clarifies, among other things, that device AI is high-risk under the AI Act only when it is a safety component or is itself a device and requires third-party conformity assessment, and that the AI Act does not change the device risk classification. The lesson is that two regimes can apply to one product without merging into one process.

**The AI Act timeline (documented, and amended).** The Act entered into force on 1 August 2024. Secondary summaries written in 2025 commonly state that high-risk obligations for Annex I products, including medical devices, apply from 2 August 2027. The Commission's AI Act Service Desk, which reflects the Digital Omnibus amendments, lists 2 August 2028 for high-risk AI embedded in Annex I products. The lesson is the one demonstrated in section 6: a date can be correct at writing and wrong at reading.

**The FDA device list (documented).** The FDA states that it has authorised over 1,600 AI-enabled medical devices as of September 2026, and it notes that the list is not comprehensive because devices are identified through AI-related terminology in summaries and classifications. The lesson is that even an official count carries a scope caveat, and a citation that drops the caveat overstates the precision.

### Documented facts and inferences

| Statement | Status |
|---|---|
| QMSR effective 2 February 2026 | Documented in the Federal Register and on the FDA site |
| PCCP final guidance issued 3 December 2024 | Documented in the Federal Register notice |
| MDCG 2025-6 endorsed June 2025 | Documented on the Commission's MDCG page |
| Annex I high-risk date now 2 August 2028 | Documented on the Commission's AI Act Service Desk timeline, accessed 26 October 2026 |
| Over 1,600 AI devices authorised as of September 2026 | Documented on the FDA page, with the agency's own caveat |
| Most compliance documents are not updated when a date changes | Inference from practice, not measured |
| Teams over-rely on law firm summaries for dates | Inference from the pattern of stale citations |

The two inferences are the reason this topic exists. They cannot be proved from public documents, and they are consistent with the documented cases and with the experience of anyone who has reviewed a compliance deck.

**The MDR transition extension (documented).** The EU extended the transition periods for the Medical Device Regulation through Regulation (EU) 2023/607, with deadlines staggered by risk class and conditions attached, including the requirement that a manufacturer apply to a notified body before a specified date. The case is the clearest example of a regulatory date that changed after many organisations had planned against the original. It also shows why a transition is not a simple extension: eligibility depends on actions taken before the deadline, so a team that waits loses the benefit.

**Good machine learning practice (documented).** The FDA, Health Canada, and the UK Medicines and Healthcare products Regulatory Agency published joint guiding principles for good machine learning practice in 2021. The document is principles, not regulation, and it has been influential in how regulators and manufacturers discuss model development. The case shows that a non-binding international document can shape expectations without creating an obligation, which is precisely the distinction that a citation must preserve.

### What the cases have in common

- Each involves a document whose status changed after it was widely circulated.
- Each produced a period during which reputable secondary sources were wrong.
- Each rewarded organisations that cited the primary source and dated the citation.
- Each punished organisations that had a single undated statement in a template.

The pattern is stable enough to plan for. A regulatory watch list with event triggers is the operational response, and it is the subject of Exercise 7.

### A short history of one sentence

A sentence that appeared in many 2024 compliance decks read: "High-risk AI obligations apply from 2 August 2027." The sentence was accurate when written. It became inaccurate when the timeline was amended, and the documents that contained it were not updated.

The sentence could have been written as: "As of [date], the Commission lists 2 August 2028 for Annex I high-risk products; the original Article 113 date was 2 August 2027." That version survives an amendment, because it states its own date and distinguishes the original from the current position.

The difference between the two sentences is not effort. It is the habit of dating a regulatory claim, which is the single most useful habit in this topic.

### What a good regulatory summary looks like

| Element | Weak version | Strong version |
|---|---|---|
| Instrument | "The AI Act" | "Regulation (EU) 2024/1689, Article 113, as amended" |
| Date | "August 2027" | "As of 26 October 2026, 2 August 2028" |
| Scope | "medical AI" | "Annex I products requiring third-party assessment" |
| Status | "in force" | "adopted; Annex I provisions not yet applicable" |
| Source | "law firm note" | "Commission timeline and EUR-Lex, access date recorded" |
| Uncertainty | none | "classification unresolved; implementing acts pending" |

### How to date a regulatory claim

- Write the statement with an explicit "as of" date.
- Name the source that supports the date.
- Distinguish the original position from the current one when they differ.
- Record the access date, not only the publication date.
- Note the next event that could change the answer.

A dated claim is not weaker than an undated one. It is more useful, because the reader knows exactly what is being asserted and when it was true.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Draft cited as final | No status line and no document number | Comment-period PDF saved and reused | Strategy built on a proposal | Check the guidance database |
| Superseded guidance | URL resolves, no revision date | Old page kept alive | Compliance plan misses a change | Check the current list |
| Wrong date type | Publication date used as compliance date | Three dates conflated | Launch before the obligation begins | Build a date table |
| Transition ignored | "The deadline is" with no class detail | Staggered deadlines simplified | Product planned against the wrong date | Check the transitional articles |
| Guidance treated as law | "FDA requires" with a guidance citation | Bindingness not checked | Over-compliance or misplaced confidence | Cite the regulation |
| Standard treated as mandatory | Standard cited without the instrument | Presumption confused with obligation | Unnecessary work or false assurance | Check the citation in the OJ or FDA list |
| Jurisdiction assumed | "Europe" or "the US" as the scope | Sub-national and national measures missed | Launch blocked in one market | Build a jurisdiction matrix |
| Amendment missed | Article quoted from memory | Consolidated text not checked | Citation is textually wrong | Check EUR-Lex or eCFR |
| Secondary summary cited for a date | Law firm note as the only source | Summaries lag amendments | Date is stale | Cite the issuer |
| Informal position treated as policy | FAQ or speech cited as requirement | Formal status not distinguished | Enforcement risk misjudged | Cite the guidance and note the FAQ |
| Classification assumed | Intended use written after the strategy | Scope not checked | Entire route is wrong | Check classification first |
| Transition assumed to apply | New product planned under an old route | Eligibility conditions not read | Product cannot be placed | Check eligibility |
| National measures missed | EU-level analysis only | Implementation is partly national | Local requirements unmet | Check the member state |
| Enforcement ignored | Text read without practice | Interpretation differs in practice | Review surprises | Read warning letters and decisions |
| Version not recorded | Citation has no date | Access date omitted | Cannot audit later | Record version and date |
| Two regimes conflated | AI Act and MDR requirements merged | Layer confusion | Duplicate or missing work | Map the device-AI stack |
| Guidance date precedes amendment | Guidance older than the amending act | Interpretation may be stale | Review surprise | Compare dates and check for updates |
| Standard edition not named | "ISO 14971" without a year | Edition changed | Technical file cites the wrong edition | Cite edition and amendment |
| Assumption hidden | Classification taken as given | Scope never tested | Whole analysis is conditional on an unstated premise | State the classification assumption |

Three red flags deserve separate mention. A regulatory deck with no dates on its citations is a deck that cannot be checked. A sentence that says "FDA/EU require" without an instrument is a sentence that has merged two jurisdictions. And a roadmap with a single regulatory milestone for a product sold in several markets is a roadmap that has not been built from a jurisdiction matrix.

### A ten-minute audit of a regulatory deck

- Does every citation name an instrument, not just an agency?
- Does every citation carry a date or a version?
- Is any guidance cited as a requirement?
- Is any transition period mentioned without its conditions?
- Does any sentence merge two jurisdictions?
- Is the classification assumption stated anywhere?
- Is the application date distinguished from the publication date?
- Is there a named owner for keeping the deck current?
- Is the source recorded in a way that a reviewer can check in five minutes?
- Is there a stated trigger for the next review?

A deck that fails three or more of these checks should not go to a regulator, a board, or an investor without rework. The checks take ten minutes and catch the failures that are most expensive to correct later.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The issuer's register is current | Status checks depend on it | The register lags a change | Cross-check the official journal |
| The instrument applies to the product | Scope reasoning needs a category | Classification is borderline | Check the classification rules |
| The transition conditions are met | Deadline planning assumes eligibility | Product placed after the cut-off | Read the transitional article |
| Guidance reflects the current regulation | Interpretation assumes alignment | Guidance predates an amendment | Compare dates |
| The national measures are known | EU compliance is partly national | Member state adds requirements | Check the competent authority |
| Enforcement matches the text | Risk assessment assumes it | Authority applies a different reading | Read enforcement actions |
| The summary was accurate when written | Historical citations assume good faith | Summary was wrong from the start | Check the primary source |
| The classification will not change | Planning assumes stability | Intended use changes | Re-run classification at each change |
| A date will not be amended again | Roadmaps assume stability | A new amendment or extension | Monitor the official journal |
| Legal review is available when needed | Escalation assumes capacity | No counsel engaged | Identify counsel before the question is urgent |
| The guidance's interpretation matches enforcement | Risk assessment assumes alignment | Enforcement priorities differ | Read enforcement actions |
| The register has an owner | Maintenance assumes accountability | No named owner | Assign an owner and a review date |
| The team will act on a dated claim | Dating assumes it changes behaviour | Undated templates persist | Add a dating rule to the template |
| The market scope is stable | Planning assumes it | New market added | Re-run the matrix at each market entry |

The largest limit is that this topic is educational. It describes how to read the sources, not how to reach a legal conclusion. The second is jurisdictional coverage: the method is portable, and the specific instruments in this file are US and EU only.

A third limit is temporal. Every date in this file was checked on 26 October 2026 and can change. The method is designed to survive that; the dates are not.

A fourth limit is scope of expertise. A regulatory lead can establish what the sources say and can draft a position, and only counsel can give a legal opinion on a contested interpretation. The procedure is designed to make the hand-off to counsel efficient, not to replace it.

A fifth limit is organisational. The method assumes that someone owns the regulatory source register and has time to maintain it. In organisations without that role, the register decays and the citations revert to inherited text.

## 12. Exercises

**Exercise 1 — Classify the instrument (recall, 20 minutes).**
Take five regulatory documents you have used and label each as statute, regulation, guidance, standard, or commentary, with its binding status.
Deliverable: a five-row table with document, type, issuer, and binding status.
Acceptance criteria: at least one document is reclassified from "requirement" to "guidance"; each row names the issuer; the status uses the vocabulary in section 5.
Time: 20 minutes.

**Exercise 2 — Status check (application, 30 minutes).**
For three instruments in your domain, determine their status at 26 October 2026 using the issuer's register.
Deliverable: a three-row status table with document number, status, date, and the URL consulted.
Acceptance criteria: every status is confirmed at the issuer; at least one instrument is found to be in transition or superseded; access dates are recorded.
Time: 30 minutes.

**Exercise 3 — Date table (calculation, 30 minutes).**
Build a date table for one product and one market: publication, entry into force, application, transition end, and the operational deadline with a buffer.
Deliverable: a date table with five rows and a one-paragraph explanation of the buffer.
Acceptance criteria: the dates are distinct and sourced; the buffer is justified by a stated assumption about review or notified body capacity.
Time: 30 minutes.

**Exercise 4 — Trace the hierarchy (application, 35 minutes).**
Take a requirement from a guidance document and trace it to the regulation that creates the obligation, or show that no obligation exists.
Deliverable: a three-layer trace with quotations and article numbers.
Acceptance criteria: the trace reaches a binding instrument or documents the absence of one; the guidance is not cited as the obligation; the article number is correct.
Time: 35 minutes.

**Exercise 5 — Find a stale citation (critique, 30 minutes).**
Audit a document for regulatory citations that are superseded, undated, or missing a provision.
Deliverable: a marked-up list of at least five findings with replacements.
Acceptance criteria: every finding names the defect and the fix; at least one finding concerns a date; the replacements are verified at the issuer.
Time: 30 minutes.

**Exercise 6 — Jurisdiction matrix (design, 40 minutes).**
Build a matrix for one product across two markets, comparing classification, route, evidence expectations, and key dates.
Deliverable: a two-column matrix with sources and dates.
Acceptance criteria: both markets are sourced at the regulator; differences are explained, not just listed; the matrix identifies the stricter market.
Time: 40 minutes.

**Exercise 7 — Watch list (design, 30 minutes).**
Build a regulatory watch list with instrument, owner, cadence, and event triggers.
Deliverable: a table of at least ten instruments with review intervals and triggers.
Acceptance criteria: intervals differ by instrument type; every instrument has an event trigger; the list fits a realistic monthly time budget.
Time: 30 minutes.

**Exercise 8 — Write the sentence (application, 25 minutes).**
Rewrite three regulatory sentences so that each names the instrument, provision, version, jurisdiction, and date.
Deliverable: three rewritten sentences with a verification note for each.
Acceptance criteria: all five coordinates are present or explicitly marked unknown; each sentence can be checked in under five minutes; no sentence says "FDA requires" without an instrument.
Time: 25 minutes.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Classify | Binding status correct for each document | Calling guidance "law" |
| 2 Status | Status confirmed at the issuer | Relying on a search snippet |
| 3 Date table | Five distinct dates | Publication used as application |
| 4 Hierarchy | Trace reaches a binding instrument | Stopping at the guidance |
| 5 Stale citation | Five findings with replacements | Only broken links reported |
| 6 Matrix | Both markets sourced | One market described twice |
| 7 Watch list | Triggers and cadences | Annual review for everything |
| 8 Sentence | Five coordinates present | "FDA requires" |

Each exercise should be done on a document the learner actually owns, because the point is to find the defects that are already in circulation rather than to invent examples.

A learner who completes Exercises 2 and 5 will usually find at least one stale citation in their own materials. That finding is the most useful outcome of the topic.

Exercise 4 is the one that changes how people read guidance. Once a learner has traced a guidance sentence to a regulation article, the difference between the two becomes concrete rather than theoretical.

## 13. Self-check answers

**1. What is the difference between a regulation and a guidance document?**
A regulation is legally binding and creates obligations. Guidance describes how an agency interprets or applies the regulation and is generally non-binding, although it can be persuasive and can be referenced in enforcement. A sentence that needs to establish an obligation must cite the regulation.

**2. Why are publication, entry into force, and application different dates?**
Publication makes the text public, entry into force makes it legally valid, and application sets when the obligations begin. An instrument can be in force for months or years before it applies, which is why a launch plan built on the publication date is wrong.

**3. What does a transition period permit?**
It permits an older compliance route to be used for defined products or certificates until a deadline. It usually does not permit a new product to be placed on the market under the old route, and the eligibility conditions are specific.

**4. Why is a harmonised standard not law?**
It is a technical standard that creates a presumption of conformity when cited in the Official Journal under the relevant instrument. The obligation comes from the regulation; the standard provides a route to demonstrating compliance.

**5. What does a guidance document's use of "must" establish?**
It establishes what the agency expects, not necessarily a legal obligation. The obligation must be traced to the statute or regulation. If the guidance interprets an existing requirement, the requirement is the source of the obligation.

**6. Why can a regulatory date in a reputable summary be wrong?**
Because summaries lag amendments. A date that was correct when written can be changed by an amending instrument or an implementing act, and the summary is rarely updated. The issuer's current statement is the only reliable source.

**7. What is the first check when a product is sold in several markets?**
Whether the product is a device, and in what class, in each market. Classification determines the route, the evidence, and the timeline, and it can differ across jurisdictions.

**8. Why does a regulatory clearance not establish clinical benefit?**
Because the clearance establishes that a submission met the standard for a marketing pathway for a stated intended use. It does not establish comparative effectiveness, and it does not replace clinical evidence or a coverage decision.

**Rubric for Exercises 6 and 7.** A strong jurisdiction matrix sources both columns at the regulator, explains the differences, and identifies the stricter market. A weak matrix lists requirements without dates or sources. A strong watch list has different cadences by instrument type and a trigger for each entry. A weak list reviews everything annually.

**9. Why is a transition period not a grace period for a new product?**
Because the transitional provisions usually apply to products already placed on the market or to certificates issued before a date, and they often require an application to a notified body before a cut-off. A new product generally must meet the new requirements from the application date.

**10. What is the practical effect of a standard being harmonised?**
It gives a presumption of conformity with the requirements it covers when it is cited in the Official Journal under the relevant instrument. A manufacturer can use another route and must then demonstrate conformity by other means.

**11. Why should a citation name the article and not only the regulation?**
Because a regulation contains many provisions, and a sentence that cites the whole instrument cannot be checked. The article number, with the version, is the smallest unit a reviewer can verify.

**12. What does an agency FAQ establish?**
It establishes the agency's informal answer at a point in time. It is useful evidence of interpretation and is not a binding position. If it contradicts the guidance, the guidance governs and the difference is worth noting.

**13. Why is a jurisdiction matrix more useful than a single-market answer?**
Because classification, evidence expectations, and dates differ across markets, and a product launched in several markets must satisfy all of them. The matrix makes the differences visible and identifies the stricter market as the design baseline.

**14. What is the difference between an implementing act and a delegated act?**
Both are adopted under the parent regulation and both can carry important detail, but they follow different procedures and have different scrutiny by the legislature. For the purpose of a citation, the practical point is that both must be checked, because either can change the operative detail.

**15. Why can a guidance document be more detailed than the regulation and still be less authoritative?**
Because detail is not authority. The regulation creates the obligation, and the guidance explains how the agency reads it. A guidance can be more specific and still be non-binding, and a citation that confuses specificity with authority will overstate the requirement.

**16. What should a regulatory source register contain?**
Instrument, provision, version, jurisdiction, status, application date, owner, access date, review date, and the trigger that forces an earlier check. Without the owner and the trigger, the register decays into a list.

**17. Why is "we checked it last month" not a status?**
Because the check may have been against a secondary source, and the instrument may have changed since. A status is a date, a source, and a reviewer, not a claim about recency.

## 14. Interview questions

**1. How do you tell whether a guidance is current?**
Strong answer: checks the agency's guidance database, the document number, the issue date, and any revision, and checks whether an amendment has changed the underlying regulation. Weak answer: checks the PDF date or the URL.

**2. What is the difference between a regulation and a directive in the EU?**
Strong answer: a regulation is directly applicable; a directive must be transposed into national law, so the national measure matters. Weak answer: treats them as synonyms.

**3. How do you plan against a transition deadline?**
Strong answer: confirms eligibility, computes the operational deadline with a buffer for review capacity, and checks for amendments that could extend or shorten it. Weak answer: uses the legal deadline as the plan date.

**4. A vendor says its product is "FDA cleared". What do you check?**
Strong answer: the clearance in the agency database, the product code, the intended use, the predicate, and the date. Weak answer: accepts the claim.

**5. How do you handle a requirement that appears in guidance but not in the regulation?**
Strong answer: treats it as an expectation, cites the guidance as guidance, and assesses review risk rather than legal exposure. Weak answer: either ignores it or calls it law.

**6. What would make you escalate a regulatory question?**
Strong answer: borderline classification, conflicting instruments, uncertain transition eligibility, or a claim that will appear in labelling. Weak answer: escalates everything or nothing.

**7. How do you keep a regulatory summary current?**
Strong answer: dates every statement, cites the instrument, and sets triggers for amendments, implementing acts, and transition deadlines. Weak answer: reviews it annually.

**8. What is the risk of citing a law firm note?**
Strong answer: it is secondary, may lag amendments, and may reflect a particular client context; it is useful for orientation and not for the operative date. Weak answer: says it is unreliable in general.

**9. How does the AI Act interact with the MDR?**
Strong answer: they are separate regimes with separate obligations and timetables; device AI can be high-risk under the AI Act when it is a safety component or a device requiring third-party assessment; the MDR classification is unchanged. Weak answer: assumes the AI Act replaces the MDR.

**10. What does "state of the art" mean in a regulatory context?**
Strong answer: the accepted practice at a point in time, evidenced by standards, guidance, and clinical practice, not merely the newest technology. Weak answer: treats it as marketing language.

**11. How do you handle two markets with different classification for the same product?**
Strong answer: builds a matrix, designs to the stricter route where feasible, and documents the divergence and its rationale. Weak answer: picks one market's classification and applies it everywhere.

**12. What is the risk of a single undated statement in a template?**
Strong answer: it propagates to every document that reuses the template and becomes wrong without anyone noticing; dating the statement contains the damage. Weak answer: says templates are administrative.

**13. How would you verify an AI Act date?**
Strong answer: checks the Commission's implementation timeline for the current position and the regulation and amending instruments for the legal basis, and records the access date. Weak answer: cites a law firm summary.

**14. What makes a regulatory watch list useful rather than decorative?**
Strong answer: each entry has an owner, a cadence, and an event trigger, and the list is reviewed on a schedule that the team can sustain. Weak answer: a long list of links with no owner.

**15. How do you handle a regulation that has been amended but not consolidated?**
Strong answer: reads the amending instrument alongside the original, cites both, and does not rely on an unofficial consolidation. Weak answer: quotes a consolidated PDF without checking its date.

**16. What is the first question to ask about a regulatory date in a deck?**
Strong answer: what event the date marks, and whether the deck means publication, application, or the end of a transition. Weak answer: whether the date is recent.

**17. How would you brief an executive in three sentences?**
Strong answer: current position with date and source, the assumption that makes it relevant, and what would change it. Weak answer: a summary of the regulation.

## 15. Key terms

- **Statute** — a law enacted by a legislature.
- **Regulation** — a binding instrument adopted under delegated authority.
- **Directive** — an EU instrument transposed into national law.
- **Guidance** — an agency's non-binding interpretation.
- **Draft guidance** — a proposal open for comment.
- **Harmonised standard** — a standard cited in the Official Journal giving presumption of conformity.
- **Recognised consensus standard** — a standard recognised by the FDA for a declared use.
- **MDCG** — the Medical Device Coordination Group.
- **IMDRF** — the International Medical Device Regulators Forum.
- **Entry into force** — the date an instrument becomes legally valid.
- **Application date** — the date obligations begin.
- **Transition period** — a window in which an older route remains available.
- **Sunset** — a date on which a provision expires.
- **Consolidated version** — a reading text incorporating amendments.
- **Corrigendum** — a correction to a published instrument.
- **Implementing act** — a later instrument supplying detail.
- **Delegated act** — a later instrument supplementing the parent regulation.
- **Notified body** — an organisation assessing conformity in the EU.
- **Predicate** — a legally marketed device used for comparison in a 510(k).
- **Substantial equivalence** — the 510(k) comparison standard.
- **De Novo** — a route for novel lower-risk devices.
- **Premarket approval** — the route for high-risk devices.
- **Intended use** — the manufacturer's stated purpose.
- **Classification** — the risk category determining the route.
- **Conformity assessment** — the process of demonstrating compliance.
- **PCCP** — predetermined change control plan.
- **Post-market surveillance** — systematic monitoring after marketing.
- **Vigilance** — reporting serious incidents.
- **Recall** — removal or correction of a device in the field.
- **State of the art** — accepted practice at a point in time.
- **Conformity route** — the assessment path determined by classification.
- **Technical documentation** — the evidence file supporting conformity.
- **Vigilance report** — the notification of a serious incident.
- **Field safety notice** — a communication about a corrective action.
- **Eudamed** — the EU device database.
- **Predicate comparison** — the 510(k) analysis against a marketed device.
- **Special controls** — device-specific requirements for a De Novo classification.
- **Transitional article** — the article that sets the transition rules.
- **Certification** — a notified body's confirmation of conformity.
- **Declaration of conformity** — the manufacturer's formal statement of compliance.
- **Authorised representative** — the EU entity acting for a non-EU manufacturer.
- **Importer** — the EU entity placing a third-country device on the market.
- **Serious incident** — an event that must be reported under vigilance rules.
- **Corrective action** — action to address a nonconformity or risk.
- **Field safety corrective action** — the EU term for a corrective action in the field.
- **Post-market clinical follow-up** — planned clinical evidence collection after marketing.

## 16. Sources

All URLs were checked on 26 October 2026. Regulatory status should be rechecked before reliance.

**Regulation (EU) 2017/745 (Medical Device Regulation).** https://eur-lex.europa.eu/eli/reg/2017/745/oj
Good for: the binding EU device requirements, classification rules, and transitional provisions. Not for: practical interpretation, which is in MDCG guidance. Use the consolidated version with the amending regulations.

**Regulation (EU) 2017/746 (In Vitro Diagnostic Regulation).** https://eur-lex.europa.eu/eli/reg/2017/746/oj
Good for: IVD-specific requirements and staggered transition deadlines. Not for: device requirements, which are in the MDR.

**Regulation (EU) 2024/1689 (Artificial Intelligence Act).** https://eur-lex.europa.eu/eli/reg/2024/1689/oj
Good for: the AI Act's obligations, Annex I scope, and Article 113 application dates. Not for: device-specific obligations, which remain in the MDR and IVDR. Check the amending instruments for changed dates.

**European Commission, AI Act Service Desk timeline.** https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
Good for: the current implementation dates, reflecting the Digital Omnibus amendments, including 2 August 2028 for Annex I high-risk products. Not for: the legal text; cite the regulation for the obligation.

**MDCG 2025-6, FAQ on the interplay between the MDR, IVDR, and the AI Act.** https://health.ec.europa.eu/system/files/2025-06/mdcg_2025-6_en.pdf
Good for: how the device and AI regimes apply together, including the high-risk condition and the treatment of in-house devices. Not for: legal advice on a specific product.

**FDA, Quality Management System Regulation (QMSR).** https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
Good for: the effective date of 2 February 2026 and the incorporation of ISO 13485:2016. Not for: the full regulatory text, which is at 21 CFR Part 820.

**Federal Register, Medical Devices; Quality System Regulation Amendments.** https://www.federalregister.gov/documents/2024/02/02/2024-01709/medical-devices-quality-system-regulation-amendments
Good for: the final rule, its preamble, and the basis for the effective date. Not for: current guidance on inspections, which is separate.

**FDA, Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions.** https://www.fda.gov/media/185050/download
Good for: the three PCCP components and the submission types. Not for: a requirement to use a PCCP, which remains optional.

**FDA, Artificial Intelligence-Enabled Medical Devices.** https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
Good for: the agency's statement that it has authorised over 1,600 AI-enabled devices as of September 2026, and the device list. Not for: a comprehensive count; the agency notes the list is not exhaustive.

**FDA, Search for Guidance Documents.** https://www.fda.gov/regulatory-information/search-fda-guidance-documents
Good for: confirming draft versus final status and issue dates. Not for: legal effect, which depends on the statute and regulation.

**International Medical Device Regulators Forum.** https://www.imdrf.org/
Good for: international consensus documents on device regulation. Not for: binding requirements in any jurisdiction.

**International Organization for Standardization, ISO 14971:2019.** https://www.iso.org/standard/72704.html
Good for: the risk-management standard referenced by device regulation. Not for: free access; the standard is paywalled.

**International Electrotechnical Commission, IEC 62304.** https://www.iec.ch/
Good for: the software life-cycle standard cited in device submissions. Not for: free access or guidance on how to interpret it for AI.

**U.S. Food and Drug Administration, Good Machine Learning Practice for Medical Device Development: Guiding Principles.** https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
Good for: joint US, Canadian, and UK principles on ML device development. Not for: a binding requirement or a checklist.

**European Commission, Medical Device Coordination Group guidance list.** https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en
Good for: the current MDCG documents and their endorsement dates. Not for: the legal status of each document, which must be read from the document itself.

**U.S. Food and Drug Administration, Artificial Intelligence-Enabled Medical Devices list.** https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
Good for: the device-level list with decision dates and product codes. Not for: a comprehensive census; the agency notes the list is not exhaustive.

**Medicines and Healthcare products Regulatory Agency.** https://www.gov.uk/government/organisations/medicines-and-healthcare-products-agency
Good for: the UK regime, which diverges from the EU after Brexit. Not for: EU compliance, which requires the EU instruments.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_supplied-clinical-library.md`](02_supplied-clinical-library.md)
- Next topic: [`04_methods.md`](04_methods.md)
- Keeping current: [`09_keeping-current.md`](09_keeping-current.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 14, device regulation: [`../14_devices_regulation/`](../14_devices_regulation/)
- Module 16, privacy and security: [`../16_privacy_security/`](../16_privacy_security/)
- Module 13, biopharma regulation: [`../13_biopharma/`](../13_biopharma/)
- Module 18, quality systems: [`../18_quality/`](../18_quality/)
- Module 09, clinical NLP: [`../09_clinical_nlp/`](../09_clinical_nlp/)

The next topic moves to methods sources, where the currency problem is different: the papers are stable, and the implementations and reporting standards are what change.

A reader who wants the practical version of this topic should keep two artefacts: a dated citation template and a watch list with owners. Everything else in the topic supports those two.

| Jurisdiction | Status source | What it shows | What it does not show |
|---|---|---|---|
| United States | Federal Register and eCFR | Final rules, effective dates, current text | Local enforcement practice |
| United States | FDA guidance database | Draft and final guidance, issue dates | Whether the guidance is binding |
| European Union | EUR-Lex | Regulation text, amendments, consolidated versions | National implementing measures |
| European Union | MDCG guidance list | Endorsed guidance and dates | Legal status of each document |
| European Union | Official Journal | Harmonised standard citations | Whether a standard fits your product |
| International | IMDRF document library | Consensus documents | Adoption in any jurisdiction |
