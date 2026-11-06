# Case Selection Bias: From Visible Stories to Defensible Decisions

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Topics 01–03; basic percentages and sampling  
> Version 2026.5

## 1. Why this topic exists

A supply-chain leader sees a dozen published success stories about automated replenishment.
The stories make a procurement decision appear straightforward.
The leader cannot see the customers that did not agree to publicity, stopped the project, or never bought the product.
The visible cases are a selected set, not a census of attempts.

A case library is an instrument for learning, not an automatic estimate of a market-wide success rate.
Its composition depends on who runs a project, who measures it, who permits publication, what a search engine indexes, and what language the reviewer understands.
Selection can occur at each stage, before anyone reads the final story.
A trustworthy calculation on the stories found can still answer the wrong population question.

The decision matters because a percentage copied into an investment model may drive a capital request, a supplier contract, or a service-level commitment.
A case can be useful as an example of a mechanism without estimating its average effect.
A carefully sourced account of one site's results is not evidence that every eligible site obtained those results.
The relevant question is not whether the story is interesting, but which decision it can support.

This lesson covers survivorship, organization size, vendor marketing, recency, and language.
These mechanisms overlap: a large, English-speaking customer with a successful recent rollout may be unusually easy to find for several reasons at once.
A control for one mechanism does not necessarily control the others.
Even a diverse-looking list may omit failures systematically.

**Evidence label:** External sources below document general methods or the particular public materials described.
**Claim label:** A company or vendor assertion belongs to its author until corroborated.
**Inference label:** The explanation that a particular missing case was suppressed requires evidence, not suspicion alone.
**Synthetic label:** All numerical cases in Sections 3, 5–7, and 12–13 are invented for teaching and are not market estimates.

You will define an eligible population, construct a search and screening ledger, inspect missingness, calculate sensitivity bounds, and choose a proportionate decision gate.
Selection bias concerns how cases enter the analysis; evidence grading concerns what a source can substantiate once a case is found.
A peer-reviewed case can still be selected, and a vendor case may be a useful lead without proving average impact.

## 2. Learning outcomes

After this lesson you can:

1. **Draw** the stages from an eligible project to a published case and identify where selection can occur.
2. **Distinguish** visibility, eligibility, observation, outcome measurement, and causal attribution.
3. **Calculate** a selected-case success fraction, population fraction, and missing-outcome bounds.
4. **Explain** why weighting cannot repair an unobserved stratum or unknown within-stratum selection.
5. **Design** a search protocol covering vendor, size, time, language, and discontinued-project sources.
6. **Audit** duplicates, publication incentives, denominators, and missing outcomes in a case register.
7. **Choose** a bounded business action and an evidence-gathering gate instead of extrapolating a headline.
8. **Write** a decision memo that separates evidence, attributed claim, inference, and synthetic assumption.

Exercises 1–2 test the selection model and denominator.
Exercises 3–4 test computation and weighting.
Exercises 5–6 test search design and source critique.
Exercises 7–8 test decision gates and communication.
Section 13 provides numerical answers or explicit design rubrics.

## 3. The core mental model

```text
Decision and target population
        |
        v
Eligible attempts, including stopped projects
        |
        +---- implementation survival and outcome measurement
        v
Organization agrees to disclose
        |
        +---- vendor/customer publication choices and incentives
        v
Document exists in a language and place
        |
        +---- search, date, access, and translation choices
        v
Retrieved record --> deduplication --> screened case --> extracted outcome
        |                                                  |
        +-------------- selection ledger ------------------+
                               |
                               v
                qualified inference and decision gate
```

A project can disappear at more than one arrow.
A failed deployment may leave a procurement record but no published result.
A successful deployment may exist only in a local-language trade publication.
A retrieved announcement may have no later outcome.
Each kind of absence needs a different response.

Let the target population contain eligible attempts indexed by `i`.
Let `S_i = 1` mean that an attempt enters the reviewed library and `Y_i = 1` mean that it meets a prespecified outcome definition.
The visible fraction is `P(Y=1 | S=1)`; the population fraction is `P(Y=1)`.
Equality is not guaranteed merely because all selected cases were counted correctly.
If entry depends on outcome, size, language, or an unobserved combination, the visible and target distributions can differ.

For an intervention effect, even an unbiased estimate of a success fraction is insufficient.
A causal effect additionally requires a credible comparison with what would have happened without the intervention.
Selecting cases and attributing causes are distinct problems.
A failure to control either one may overstate the transferable benefit.

Consider a synthetic population with 40 eligible projects, 15 of which meet a common success rule.
Fourteen projects appear in the library, and 11 of those meet the rule.
The selected fraction is 11/14, or 78.6%; the population fraction is 15/40, or 37.5%.
The difference of 41.1 percentage points is selection distortion in this constructed example, not a causal effect.
In a real investigation the 26 unseen outcomes would generally not be known.

### Three populations to keep apart

- **Target:** all attempts relevant to the decision, defined before seeing their outcomes.
- **Sampling frame:** the operational or documentary list from which a team could actually sample.
- **Observed:** the subset of cases found, eligible, and sufficiently measured for the analysis.

The target may include subsidiaries whose projects are confidential.
The frame may exclude them because procurement data are held separately.
The observed set may be smaller still because outcome reports are unavailable.
A frame-completeness assessment is necessary before treating the frame as a population census.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Target population | Units to which the decision question refers | All documents returned by a search | Project, site, or organization |
| Frame | Enumerated units available for sampling | Assuming an incomplete list is the target | Unique unit IDs |
| Unit of analysis | Entity counted once in an estimate | Counting three articles as three projects | Project, site, or firm |
| Eligibility | Rule determining whether a unit belongs | Changing the rule after seeing outcomes | Yes, no, unclear |
| Selection | Process determining observation in the analysis | Equating selection with fraud | `S=1` or `S=0` |
| Survivorship | Overrepresentation of units remaining visible or operating | Assuming all missing projects failed | Status and date |
| Size bias | Unequal visibility or selection by scale | Assuming large-company evidence is invalid | Employees, sites, or revenue |
| Vendor marketing bias | Selection and presentation shaped by promotional incentives | Treating a vendor document as independent confirmation | Sponsor and provenance |
| Recency bias | Unequal attention by publication or project date | Assuming old evidence is always obsolete | Event and publication dates |
| Language bias | Unequal retrieval or interpretation by language | Equating language with geography | Source language, search language |
| Outcome | Measured result under a defined rule | Equating implementation with business benefit | Metric, baseline, period |
| Success fraction | Share meeting the predefined threshold | Calling it a causal effect | Numerator/denominator, percent |
| Missing outcome | Eligible unit with outcome unavailable | Coding missing as failure or success | Unknown, with reason |
| Nonresponse | A sampled unit does not provide usable data | Assuming its outcome resembles respondents | Count and response rate |
| Positivity | Every target stratum has a positive chance of observation | Applying a weight to a stratum with no cases | Inclusion probability > 0 |
| Inclusion probability | Chance a unit enters a probability sample | Deriving a probability from a convenience count | Probability between 0 and 1 |
| Weight | Contribution assigned to a sampled unit | Treating weights as proof of representativeness | Usually inverse probability |
| Sensitivity bound | Range of values under explicit missing-data assumptions | Calling a broad range a confidence interval | Percentage points |
| Comparator | What outcome is compared against | Treating before/after as a counterfactual | Same metric and period |
| Dependence | Sources trace to the same underlying claim | Counting press rewrites as independent proof | Provenance graph |
| Publication date | Date a source was made available | Using it as the intervention date | ISO date where known |
| Outcome date | Period during which results were measured | Treating a recent article as a recent result | Start and end date |
| Search recall | Fraction of eligible records retrieved | Claiming recall without a reference set | Retrieved eligible / known eligible |
| Precision | Fraction of retrieved records eligible | Confusing a clean search with complete coverage | Eligible / retrieved |
| Denominator | Count of eligible units underlying a fraction | Using selected stories as the target population | Units and eligibility rule |
| Censoring | End of observation before an outcome is fully seen | Coding an unfinished project as a failure | Observation cut-off date |
| Transfer | Applicability to a new decision setting | Equating similarity of brand with similarity of mechanism | Context and decision |

## 5. The mechanics, step by step

The sequence below is designed for a decision-oriented case review.
Each step produces an artifact that a second reviewer can inspect.
The team may iterate, but records changes rather than silently changing the protocol.
### 1. Define the business decision

**Action:** Name the action and deadline before searching.

**Illustration:** A replenishment director must choose whether to fund a 12-week controlled pilot.

**Output:** A decision statement naming owner, alternatives, downside, and date.

**Branch:** If the outcome would not change the decision, narrow the review or stop; do not manufacture a research exercise.
### 2. Define the estimand

**Action:** State the population, outcome, time window, and unit in one sentence.

**Illustration:** Fraction of eligible distribution-center projects meeting a prespecified service and cost rule within 12 months.

**Output:** A target question distinct from an estimate of causal effect.

**Branch:** If the question is impact rather than frequency, specify the untreated comparison and assignment strategy separately.
### 3. Specify eligible attempts

**Action:** Include planned, completed, abandoned, and ongoing attempts when the decision concerns attempts.

**Illustration:** Eligible projects started during 2022–2024 in owned or contracted centers.

**Output:** A rule with start-event, geography, industry, and technology boundaries.

**Branch:** If only completed deployments are relevant, say so and do not generalize to all attempts.
### 4. Select a counting unit

**Action:** Choose one project or site and define multi-site rollouts before extracting results.

**Illustration:** One independently approved project; a phase extension is not automatically a new project.

**Output:** A unique unit key and parent-child mapping.

**Branch:** If sites are units, avoid treating highly correlated sites in one rollout as independent companies.
### 5. Enumerate potential frames

**Action:** List procurement records, project registers, professional literature, filings, and vendor announcements.

**Illustration:** A project register for internal work plus published documents for external cases.

**Output:** A frame inventory including ownership, coverage dates, and known exclusions.

**Branch:** If the target cannot be enumerated, treat public cases as a convenience series rather than a census.
### 6. Map selection mechanisms

**Action:** Draw the path from start to measured result, consent, publication, retrieval, and screening.

**Illustration:** Failed attempts may stop before a press release; local-language reports may not appear in English queries.

**Output:** A stage-by-stage mechanism register with possible direction and uncertainty.

**Branch:** If evidence about a mechanism is absent, describe it as a plausible risk, not an observed fact.
### 7. Freeze search terms

**Action:** Record terms for the decision, synonyms, alternatives, and failure or cancellation.

**Illustration:** Search replenishment implementation, abandonment, warehouse automation, and local equivalents.

**Output:** A dated query log with platforms, exact strings, dates, and filters.

**Branch:** If new terms are discovered, rerun consistently and preserve both protocol versions.
### 8. Diversify sources

**Action:** Search independently of the vendor site using filings, scholarly sources, trade press, and project records.

**Illustration:** Check company and regulator records in addition to promotional material.

**Output:** A source list with sponsor and document type.

**Branch:** If a source repeats the same announcement, retain its provenance but do not count it as another result.
### 9. Address organization size

**Action:** Define scale using a variable relevant to deployment rather than a brand impression.

**Illustration:** Use sites, employees, annual volume, or revenue, with the chosen measure recorded.

**Output:** A size distribution for the frame and observed records where available.

**Branch:** If target size shares are unknown, report observed composition and avoid unsupported population weights.
### 10. Address language

**Action:** Choose languages using target geography and available expertise, not simply reviewer fluency.

**Illustration:** Search relevant local-language synonyms and use a bilingual reviewer for outcome extraction.

**Output:** Queries, languages searched, translation method, and languages not covered.

**Branch:** If translation is unavailable, mark the restriction and its consequences rather than pretending coverage is global.
### 11. Address time

**Action:** Separate project start, intervention, outcome window, publication, and retrieval dates.

**Illustration:** A 2025 article may summarize an outcome measured during 2019.

**Output:** A date field for each relevant event and a prespecified review window.

**Branch:** If older cases use a materially different operating regime, analyze separately rather than deleting automatically.
### 12. Register candidates

**Action:** Save every retrieved record, including a persistent identifier or archived reference when permitted.

**Illustration:** Store title, source, author, publication date, link, language, and retrieval date.

**Output:** A reproducible candidate ledger, not only a list of favorable case studies.

**Branch:** If a link is unavailable, keep the retrieval trail and mark the content unverified.
### 13. Deduplicate records

**Action:** Match company, site, project, intervention, and time before estimating counts.

**Illustration:** One supplier release and two trade articles describe the same customer project.

**Output:** A mapping from source records to one case ID, with supporting and contradicting sources linked.

**Branch:** If identity is uncertain, mark a possible duplicate and perform sensitivity analysis.
### 14. Screen consistently

**Action:** Apply the same inclusion criteria to favorable, unfavorable, and unknown outcomes.

**Illustration:** Two reviewers independently screen a sample, resolve disagreement, and record reasons.

**Output:** A flow of retrieved, duplicate, excluded, eligible, and outcome-observed records.

**Branch:** If eligibility is unclear, seek clarification or retain an unclear category; do not silently exclude.
### 15. Extract outcome and denominator

**Action:** Record original metric, baseline, comparator, population, duration, and missingness reason.

**Illustration:** A 12% reduction in inventory is recorded with currency, period, and service consequences.

**Output:** A structured claim record linked to its exact source.

**Branch:** If the denominator or measurement definition is absent, preserve the claim without calculating a population rate.
### 16. Separate evidence from advocacy

**Action:** Identify who measured, who paid, who approved release, and whether raw results are available.

**Illustration:** A vendor-authored case is attributed as a vendor claim, even when the customer is named.

**Output:** A provenance and conflict-of-interest field for each substantive claim.

**Branch:** If independent confirmation is unavailable, do not upgrade promotional claims by repetition.
### 17. Quantify coverage

**Action:** Compare observed counts with a defensible frame by relevant strata and outcome availability.

**Illustration:** Four of 20 eligible small projects and 10 of 20 large projects have outcomes.

**Output:** Coverage ratios with clear denominators; missing or unknown strata are flagged.

**Branch:** If frame totals are uncertain, show the range or avoid numerical coverage claims.
### 18. Analyze missingness

**Action:** Tabulate unknown outcomes and plausible reasons for non-observation without inventing their outcomes.

**Illustration:** A discontinued project has no outcome report; its result is unknown.

**Output:** Known-outcome fraction and worst-case bounds on the full frame where the frame is complete.

**Branch:** If missingness could depend on success after conditioning, simple complete-case estimates remain biased.
### 19. Consider weighting carefully

**Action:** Use known positive sampling probabilities and an appropriate sampling design where available.

**Illustration:** Randomly sampled small and large projects with known unequal inclusion probabilities.

**Output:** An estimator and assumptions, with uncertainty reflecting the sample design.

**Branch:** If selection probabilities are unknown or a stratum has zero observed cases, weighting alone cannot identify the target rate.
### 20. Run sensitivity analyses

**Action:** Change one explicit assumption at a time and show how the recommendation changes.

**Illustration:** Compare a complete-case rate, worst-case bounds, and a prespecified nonresponse scenario.

**Output:** A scenario table with values, assumptions, and decision consequences.

**Branch:** If the decision flips across plausible scenarios, invest in information or reduce exposure.
### 21. Evaluate transfer

**Action:** Compare baseline, operating constraints, intervention, measurement, and implementation capability with the target setting.

**Illustration:** A multinational with dedicated integration staff may not be comparable to a regional distributor.

**Output:** A transfer table separating matched, mismatched, and unknown features.

**Branch:** If a critical dependency is missing, treat the case as a hypothesis rather than a rollout blueprint.
### 22. Make a decision gate

**Action:** Choose among defer, controlled pilot, buy more information, or scale with conditions.

**Illustration:** Authorize a capped pilot only if baseline, counterfactual, and stop rules are agreed.

**Output:** A signed decision memo with owner, limits, review date, and evidence required to advance.

**Branch:** If the downside exceeds the permitted exposure, defer or reduce scope even when stories are favorable.
### 23. Maintain the library

**Action:** Revisit open cases and record later outcomes, corrections, and changed operating conditions.

**Illustration:** A pilot announcement receives a follow-up check after its planned measurement period.

**Output:** A dated revision history retaining the old claim and the corrected status.

**Branch:** If a source disappears or changes, record what can and cannot be verified; do not erase the audit trail.
### Operating the selection ledger

Use one project row and a separate source table.
A source may support multiple projects; a project may have multiple sources.
This separation prevents a popular case from receiving extra weight because it was copied repeatedly.
Store the original quotation or location along with your interpretation, subject to access and copyright rules.

| Field | Example value | Review question |
|---|---|---|
| Case ID | `P-017` | Is this a unique project under the chosen unit rule? |
| Frame origin | Procurement register | What projects does this register exclude? |
| Inclusion status | Eligible | Was the same rule used for failed projects? |
| Stage | Discontinued | Was it ever implemented or merely proposed? |
| Company scale | 4 distribution centers | Does scale use the same definition across cases? |
| Publication language | Spanish | Was the search conducted in that language? |
| Event period | 2023-01 through 2023-12 | Is this an outcome period or a publication date? |
| Result | Unknown | Is the result unavailable, unmeasured, or not yet mature? |
| Source sponsor | Supplier | Who selected and approved the story? |
| Decision use | Hypothesis only | What additional information permits a higher-stakes use? |

The ledger is an audit device, not a magic statistical correction.
Unknown values remain unknown until evidence changes them.
A spreadsheet column labeled “representative” does not establish representativeness.

### Business decisions and the standard of evidence

A case may justify a conversation with an operations manager.
It may justify a bounded pilot if the exposure is small and measurement is credible.
It does not automatically justify a full-network commitment.
Increase the evidence standard as the financial or service downside increases.
A decision can remain rational under uncertainty if the uncertainty is explicit and the downside is contained.

### Review checklist for a live case library

Use these checks to turn the procedure into reviewable controls.
The answer may be unknown; assign an owner rather than inventing a value.

### Audit check 1: Attempt identification

**Question:** Does the list include projects that obtained approval but never launched?

**Control:** Reconcile approval dates with deployment records and keep a not-started status.

### Audit check 2: Start definition

**Question:** Is a signed contract being mistaken for a project start?

**Control:** Set a start event such as first implementation work and apply it consistently.

### Audit check 3: Cancelled procurement

**Question:** Does the target include evaluations canceled before purchase?

**Control:** Use a separate pre-purchase population if the decision concerns technology selection.

### Audit check 4: Multiple sites

**Question:** Can a rollout cover multiple sites without creating multiple independent projects?

**Control:** Retain parent project and site IDs and name the analysis unit.

### Audit check 5: Project phases

**Question:** Is a second phase a separate attempt or part of one rollout?

**Control:** Specify the boundary before outcomes are extracted.

### Audit check 6: Corporate changes

**Question:** Could mergers hide a project under an old organization name?

**Control:** Maintain an alias table with effective dates.

### Audit check 7: Frame access

**Question:** Can the research team see all relevant business units?

**Control:** List inaccessible units and avoid asserting complete coverage.

### Audit check 8: Missing vendor

**Question:** Does the project register include implementations by other suppliers?

**Control:** Reconcile supplier fields rather than using one vendor’s customer list as the market.

### Audit check 9: Shadow pilots

**Question:** Could a site run a small experiment outside central procurement?

**Control:** Ask local operators and label any remaining coverage uncertainty.

### Audit check 10: Negative results

**Question:** Are stopped or underperforming projects searchable?

**Control:** Use terms for abandonment, cancellation, delay, and incident as well as success.

### Audit check 11: Mixed outcomes

**Question:** Could a project improve inventory while harming service?

**Control:** Record each outcome and use the prespecified joint rule.

### Audit check 12: Success threshold

**Question:** Was the threshold chosen before reviewing cases?

**Control:** Freeze the criterion and show results under a labeled alternative threshold.

### Audit check 13: Outcome timing

**Question:** Are projects given equal follow-up?

**Control:** Track start and observation end separately and identify immature outcomes.

### Audit check 14: Calendar shocks

**Question:** Did demand disruptions overlap with measurement?

**Control:** Record the shock and consider a comparable control period or site.

### Audit check 15: Publication lag

**Question:** Can the latest projects have results even if they are not yet published?

**Control:** Distinguish unreported from not yet mature and set a follow-up date.

### Audit check 16: Old material

**Question:** Does an old document describe a still-relevant operating mechanism?

**Control:** Assess context and technology rather than excluding solely by publication age.

### Audit check 17: Retrieval date

**Question:** Can a reviewer reconstruct which version was accessed?

**Control:** Record the date and stable source location, preserving a permitted copy.

### Audit check 18: Date ambiguity

**Question:** Is the date an announcement, implementation, measurement, or revision date?

**Control:** Create separate fields and use unknown when the original does not say.

### Audit check 19: Search engine filter

**Question:** Could personalization or ranking hide relevant documents?

**Control:** Record platform and filters and use multiple discovery routes.

### Audit check 20: Source access

**Question:** Are paywalled or archived records systematically absent?

**Control:** Document access restrictions and use approved alternative access where possible.

### Audit check 21: Search synonyms

**Question:** Do project teams use different names for the same process?

**Control:** Include local and operational synonyms in the versioned query log.

### Audit check 22: Exact phrases

**Question:** Can a quoted phrase make recall too narrow?

**Control:** Run a broader query in addition to exact phrases and retain both counts.

### Audit check 23: Relevant geography

**Question:** Are languages selected from the target footprint?

**Control:** Map major operating regions to languages and record uncovered regions.

### Audit check 24: Language identification

**Question:** Is a source classified by country rather than actual language?

**Control:** Record source language independently from company headquarters.

### Audit check 25: Machine translation

**Question:** Can negation or technical units be mistranslated?

**Control:** Ask a qualified reviewer to check the outcome passage against the original.

### Audit check 26: Translation access

**Question:** Does the translation method expose confidential information?

**Control:** Use approved tools and redact or restrict material as required.

### Audit check 27: Sponsor identity

**Question:** Who created, paid for, or approved the source?

**Control:** Record author, sponsor, customer role, and release permissions.

### Audit check 28: Reference selection

**Question:** Were references selected from a complete customer list?

**Control:** Request the selection procedure and treat handpicked references as leads.

### Audit check 29: Independent sources

**Question:** Do articles trace back to the same announcement?

**Control:** Map each record to the original claim before counting corroboration.

### Audit check 30: Named customer

**Question:** Does a named customer independently verify the metric?

**Control:** Seek a separate statement or measurement without assuming naming proves impact.

### Audit check 31: Missing baseline

**Question:** Is the percentage change reported without starting value?

**Control:** Record the result as an unverified claim and request the baseline.

### Audit check 32: Metric denominator

**Question:** Does a rate use orders, units, sites, or customers?

**Control:** Capture numerator, denominator, unit, and measurement window.

### Audit check 33: Gross versus net

**Question:** Does reported benefit exclude integration, labor, or subscription costs?

**Control:** Reconstruct the net business case with named cost categories.

### Audit check 34: Currency conversion

**Question:** Were monetary results compared across currencies and years?

**Control:** Retain original currency and date and document any conversion.

### Audit check 35: Service tradeoff

**Question:** Did savings occur alongside degraded customer service?

**Control:** Include an explicit service guardrail in the success rule.

### Audit check 36: Measurement owner

**Question:** Who produced the reported result and could they review the method?

**Control:** Record the measurement owner and whether independent validation occurred.

### Audit check 37: Missing reason

**Question:** Is an empty result field explained?

**Control:** Code unavailable, not measured, censored, confidential, or unclear separately.

### Audit check 38: Nonresponse follow-up

**Question:** Do selected organizations decline a request for evidence?

**Control:** Log contacts and response status without inferring outcomes from silence.

### Audit check 39: Selection probability

**Question:** Is the chance of inclusion actually known?

**Control:** Use weights only when their design or model and assumptions can be justified.

### Audit check 40: Empty group

**Question:** Is any relevant population group completely unobserved?

**Control:** Do not calculate its observed rate; obtain data or retain uncertainty.

### Audit check 41: Group overlap

**Question:** Can one project be both small and local-language?

**Control:** Use nonoverlapping strata for standardization or model intersections explicitly.

### Audit check 42: Unknown group

**Question:** Can the size or language of an attempt be unknown?

**Control:** Retain an unknown category and test sensitivity to its allocation.

### Audit check 43: Frame shares

**Question:** Are population group proportions measured or guessed?

**Control:** State the source and date of the shares; use scenarios if unknown.

### Audit check 44: Within-group selection

**Question:** Are observed successes preferentially publicized within each size group?

**Control:** Compare observed and missing characteristics and avoid claiming stratification resolves unmeasured selection.

### Audit check 45: Case count

**Question:** Are multiple articles being counted as multiple projects?

**Control:** Link documents to a unique case and maintain a separate evidence table.

### Audit check 46: Case independence

**Question:** Are several sites managed by the same team and exposed to the same intervention?

**Control:** Report clustering and avoid treating the sites as independent firms.

### Audit check 47: Screening disagreement

**Question:** Do reviewers apply eligibility rules differently?

**Control:** Review a shared sample, reconcile differences, and record rule clarifications.

### Audit check 48: Post hoc exclusion

**Question:** Was a failure excluded after the result was known?

**Control:** Preserve the original decision and document any protocol deviation.

### Audit check 49: Causal alternative

**Question:** Did a simultaneous policy or staffing change affect the metric?

**Control:** Record competing explanations and design a credible comparison.

### Audit check 50: Before-and-after

**Question:** Does the case compare only two time points?

**Control:** Consider seasonality, trend, and concurrent changes before attributing an effect.

### Audit check 51: Economic threshold

**Question:** Is the success cutoff linked to the decision economics?

**Control:** Define cost and benefit categories before interpreting the case fraction.

### Audit check 52: Downside exposure

**Question:** Could a failed rollout damage availability or create safety issues?

**Control:** Set a decision gate proportional to the potential downside.

### Audit check 53: Pilot assignment

**Question:** Will pilot sites be chosen because they are unusually capable?

**Control:** Document selection and compare their capabilities with scale-up sites.

### Audit check 54: Stop rule

**Question:** Can the pilot continue indefinitely without a clear decision?

**Control:** Set an owner, budget limit, service guardrail, and review date.

### Audit check 55: Outcome maturity

**Question:** Can the pilot meet its stated measurement horizon before approval?

**Control:** Plan a follow-up period and do not classify an incomplete outcome as success.

### Audit check 56: Forecast discipline

**Question:** Is an optimistic scenario presented as a promise?

**Control:** Label forecast, plan, target, and authorized commitment separately.

### Audit check 57: Governance owner

**Question:** Can someone approve release of confidential project data?

**Control:** Obtain the required owner and legal or privacy review before disclosure.

### Audit check 58: Source revision

**Question:** Did a source change after it was recorded?

**Control:** Date the version and preserve a change note without deleting prior context.

### Audit check 59: Stale conclusions

**Question:** Has the operating model changed since a case was reviewed?

**Control:** Schedule a refresh and mark changed assumptions in the decision record.

### Audit check 60: Communication

**Question:** Does an executive summary show only the favorable rate?

**Control:** Include denominator, selection limitation, bounds, and next evidence gate together.

## 6. Worked example

### Decision and study boundary

A synthetic food distributor is considering automated replenishment at its regional distribution centers.
The operations director asks whether the company should scale immediately or run a controlled pilot.
The relevant population is 40 comparable projects started over a specified three-year period.
For this teaching exercise, a complete retrospective register and a complete outcome audit are assumed to exist.
This assumption permits a comparison with the truth; a real public search normally has no such complete audit.

The unit is one approved project, not one source document or one warehouse site.
A project is successful if, within 12 months of its start, it achieves both a predefined service threshold and positive measured net savings.
The rule is held fixed for all four groups.
The observed result is an outcome classification, not an estimated causal treatment effect.

To make the arithmetic visible, groups are mutually exclusive and each contains 10 projects.
The group names represent dominant visibility barriers in this invented register, not an assertion that these features never overlap in reality.
Actual projects may simultaneously be small, discontinued, and reported in a local language.

| Synthetic group | Eligible projects | True successes | Selected cases | Selected successes | Selected failures |
|---|---:|---:|---:|---:|---:|
| Large and public | 10 | 9 | 9 | 8 | 1 |
| Small operators | 10 | 3 | 3 | 2 | 1 |
| Local-language reports | 10 | 2 | 2 | 1 | 1 |
| Discontinued or not publicized | 10 | 1 | 0 | 0 | 0 |
| **Total** | **40** | **15** | **14** | **11** | **3** |

The observed 14 cases are fully measured under the teaching rule.
All 26 unselected cases have unknown outcomes from the perspective of a reviewer using only the selected-case library.
The full-audit values in the true-success column are revealed solely to show what can go wrong.
Do not use that column to pretend a real reviewer knows the missing outcomes.

### Step A: Calculate the visible fraction

The selected numerator is 8 + 2 + 1 + 0 = 11 successful projects.
The selected denominator is 9 + 3 + 2 + 0 = 14 projects.
The selected fraction is 11/14 = 0.7857, or **78.6% of observed projects**.
That is an accurate description of this selected set.
It does not establish a 78.6% chance of success for a new project.

### Step B: Reveal the synthetic population fraction

The complete audit finds 9 + 3 + 2 + 1 = 15 successful projects.
The target denominator is 40 eligible projects.
The population fraction is 15/40 = 0.375, or **37.5% of eligible projects**.
The selected fraction overstates this constructed population fraction by 78.6% − 37.5% = **41.1 percentage points**, after rounding.
This difference is not a treatment effect and should not be interpreted as savings.

### Step C: Inspect coverage

Coverage of the large group is 9/10 = 90%.
Coverage of the small group is 3/10 = 30%.
Coverage of the local-language group is 2/10 = 20%.
Coverage of the discontinued group is 0/10 = 0%.
Overall coverage is 14/40 = 35%.
Coverage is uneven, but group differences alone do not tell us whether selection within each group depends on outcomes.

### Step D: Calculate assumption-free outcome bounds

Hide the true-success column and suppose only the frame size and observed outcomes are known.
If none of the 26 unobserved projects succeeded, the population fraction is 11/40 = **27.5%**.
If all 26 succeeded, it is (11 + 26)/40 = **92.5%**.
The interval [27.5%, 92.5%] is a logical worst-case range under a complete 40-project frame and the binary success definition.
It is not a 95% confidence interval and uses no random-sampling guarantee.
If the frame itself omits projects, even this range does not cover the broader target without further assumptions.

### Step E: Inspect a tempting stratified estimate

Among observed cases, the success fractions are 8/9, 2/3, and 1/2 for the first three groups.
The fourth group has no observed outcome, so its within-group fraction is undefined, not zero.
If, solely as a sensitivity scenario, the first three observed fractions represented their full groups, their contribution would be:

`0.25 × (8/9 + 2/3 + 1/2) = 0.5139`, or **51.4 percentage points**.
The missing fourth group's contribution can range from 0 to 25 percentage points under this scenario.
The resulting scenario range is **51.4%–76.4%**.
It excludes the synthetic truth of 37.5% because the observed outcomes are selectively favorable within the observed groups too.
The narrower range is therefore assumption-dependent and can be misleading.
Equal group shares do not make the observed cases representative within each group.

### Step F: Link the finding to an investment decision

Suppose the director's planning worksheet treats a successful rollout as USD 420,000 net annual benefit and an unsuccessful rollout as a USD 180,000 net annual loss.
These are synthetic net outcomes for one comparable project, not a vendor forecast or contractual commitment.
Using the selected fraction produces `11/14 × 420,000 − 3/14 × 180,000 = USD 291,429` expected net value per project-year, rounded to the nearest dollar.
Using the known synthetic population fraction produces `15/40 × 420,000 − 25/40 × 180,000 = USD 45,000` per project-year.
The numerical gap is USD 246,429 per project-year in this stylized model.
The arithmetic assumes the two outcome values fully capture consequences and that the chosen success fraction is relevant to the new project.
It does not account for financing, implementation timing, or uncertainty in either payoff.

If only the logical bounds are known, the same payoff model produces a range from `0.275 × 420,000 − 0.725 × 180,000 = −USD 15,000` to `0.925 × 420,000 − 0.075 × 180,000 = USD 375,000` per project-year.
That range crosses zero and cannot justify the same claim as the selected-case estimate.
It is conditional on the complete frame and the payoff assumptions, not a probability distribution.

### Step G: Choose the gate

A defensible recommendation is a bounded pilot rather than immediate network-wide deployment.
The pilot plan specifies a comparison design, a common outcome definition, total cost, service guardrails, owner, and a stop date.
The forecast is the explicit scenario range; the plan is to run the pilot.
The target might be positive net value without a service decline; it is not a commitment to achieve it.
A commitment is limited to the authorized pilot budget and data collection, subject to approval.
A pilot result still requires causal and transfer assessment before scaling.

## 7. Code, query, or calculation

The following standard-library Python 3 script reproduces the synthetic example.
It needs no files, network access, or third-party packages.
The `truth_successes` fields are deliberately revealed teaching data; real case searches often lack them.
Selected outcomes are all known, and groups are assumed mutually exclusive with a complete 40-project frame.
The script computes a scenario range only after assuming representativeness inside the three observed groups.
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Group:
    name: str
    eligible: int
    truth_successes: int  # hidden from a real selected-case review
    selected: int
    selected_successes: int

groups = [
    Group("large", 10, 9, 9, 8),
    Group("small", 10, 3, 3, 2),
    Group("local_language", 10, 2, 2, 1),
    Group("discontinued", 10, 1, 0, 0),
]

population = sum(g.eligible for g in groups)
true_successes = sum(g.truth_successes for g in groups)
selected = sum(g.selected for g in groups)
observed_successes = sum(g.selected_successes for g in groups)
unobserved = population - selected

visible_rate = observed_successes / selected
true_rate = true_successes / population
lower = observed_successes / population
upper = (observed_successes + unobserved) / population

# Conditional scenario: observed within-group rates represent the
# whole observed group; unobserved groups can be anywhere from 0 to 1.
scenario_lower = sum(
    (g.eligible / population) * (g.selected_successes / g.selected)
    for g in groups if g.selected
)
missing_group_weight = sum(
    g.eligible / population for g in groups if not g.selected
)
scenario_upper = scenario_lower + missing_group_weight

def net_value(success_rate):
    return success_rate * 420_000 - (1 - success_rate) * 180_000

print(f"Selected: {observed_successes}/{selected} = {visible_rate:.1%}")
print(f"Synthetic truth: {true_successes}/{population} = {true_rate:.1%}")
print(f"Difference: {(visible_rate - true_rate) * 100:.1f} percentage points")
print(f"Unobserved: {unobserved}/{population}")
print(f"Logical bounds: {lower:.1%} to {upper:.1%}")
print(f"Conditional scenario: {scenario_lower:.1%} to {scenario_upper:.1%}")
print(f"Selected-rate value: USD {net_value(visible_rate):,.0f}/project-year")
print(f"Truth-rate value: USD {net_value(true_rate):,.0f}/project-year")
print(f"Lower-bound value: USD {net_value(lower):,.0f}/project-year")
print(f"Upper-bound value: USD {net_value(upper):,.0f}/project-year")
```

Expected output:

```text
Selected: 11/14 = 78.6%
Synthetic truth: 15/40 = 37.5%
Difference: 41.1 percentage points
Unobserved: 26/40
Logical bounds: 27.5% to 92.5%
Conditional scenario: 51.4% to 76.4%
Selected-rate value: USD 291,429/project-year
Truth-rate value: USD 45,000/project-year
Lower-bound value: USD -15,000/project-year
Upper-bound value: USD 375,000/project-year
```

The program computes descriptive arithmetic, not statistical inference.
It does not estimate unknown inclusion probabilities, infer causality, or validate the assumed payoff values.
If selected outcomes are missing as well, include those missing units in the unknown-outcome count rather than in the observed denominator.
If the frame is not complete, label the result as applying to the frame, not the wider market.
If no case has an observed outcome, avoid dividing by zero and report that no selected-case fraction is available.

## 8. Data, standards, and regulatory context

### Data sources and operational constraints

An internal project portfolio may offer a better starting frame than a public case search.
It can still miss informal experiments, supplier-led pilots, and projects coded under another budget.
Procurement records can capture purchases without showing whether deployment started.
Incident records may capture failures without describing routine successes.
A complete project denominator requires reconciliation across systems, owners, dates, and definitions.

External sources include company disclosures, regulator databases, research publications, trade reports, and supplier case pages.
They serve different purposes and need different interpretations.
A securities filing may be authoritative for a disclosed figure but may not report every project attempt.
A vendor page may document a real customer relationship while choosing which outcomes to publicize.
A press article may be independent reporting or a paraphrase of a vendor press release; inspect its provenance.

### Reporting guidance, not a law

The PRISMA 2020 statement provides a reporting framework for systematic reviews, including search and selection reporting.
It is a reporting guideline, not a guarantee that a review is unbiased and not a legal requirement for every business case library.
Its flow diagram motivates recording identified, screened, excluded, and included records.
A business team may adapt those ideas to project cases while being explicit that its review is not necessarily a systematic review.
See the primary statement in Section 16.

The Cochrane Handbook discusses bias arising from selection of reported results and missing evidence in evidence syntheses.
Its healthcare examples are not direct empirical estimates of bias in supply-chain vendor marketing.
The transferable lesson is methodological: absence and reporting processes matter when aggregating available evidence.
The team must still justify how the mechanism applies in its own domain.

### Data protection, contracts, and confidentiality

Project outcomes can reveal commercially sensitive prices, employee information, customer behavior, or contractual disputes.
Use authorized access, data minimization, role-based access, and approved retention periods.
Whether privacy law, disclosure obligations, or confidentiality terms apply depends on jurisdiction, data, and contracts.
This lesson does not provide legal advice or claim that a specific reporting rule mandates publication of failed projects.
Consult legal and privacy owners before collecting or disclosing sensitive evidence.

A translation provider or external reviewer may receive confidential documents only under the organization's approved controls.
Do not put sensitive source text into an unapproved translation service.
Record whether the translation was machine-generated, human-reviewed, or unavailable.
Preserve the original wording and a link to the source where permitted.

### Versioning and reproducibility

Record search date, search platform, query, filters, languages, and reviewer.
Indexing changes over time, so a repeat search need not return identical results.
Keep the original retrieval log and distinguish a changed search result from a changed historical outcome.
If a case study is edited, preserve the revision date and the exact version used where rights permit.
Record uncertainty when an original source is no longer accessible.

## 9. Industry practice and case evidence

### Published forecasting benchmark: M5

The M5 forecasting competition uses retail sales data made available through its public competition materials.
The competition is a useful example of a bounded benchmark with a defined dataset and evaluation task.
It is not a random sample of all retailers, all replenishment decisions, or all commercial deployments.
The M5 competition site and published competition materials are linked in Section 16.

**Documented context:** The competition page describes its dataset and forecasting task.
**Inference:** A method that performs well on this benchmark may be worth testing in another retail environment.
**Limit:** Benchmark rank alone does not establish realized financial returns in a different operating system.
**Selection question:** Which demand patterns, geographies, firms, time periods, and operational outcomes are absent from that dataset?
A successful leaderboard entry is also not evidence about failed implementations that were never submitted.

### Public reporting versus vendor-selected stories

The U.S. Securities and Exchange Commission's EDGAR system provides access to public-company filings.
A filing can supply a dated, attributable disclosure, but its scope depends on the registrant and filing requirements.
It should not be treated as a complete register of every technology project or every failed pilot.

**Documented fact:** EDGAR provides a searchable database of company filings.
**Claim:** A company statement in a filing remains a statement by that company; inspect the relevant wording and context.
**Inference:** Searching filings as well as promotional sites can reveal different information channels.
**Limit:** Differences in coverage do not, by themselves, establish that any particular vendor concealed an outcome.

A vendor customer-story page is useful for identifying named deployments and proposed operating mechanisms.
The source's commercial purpose makes case-selection and outcome-presentation questions particularly important.
Seek independent operational data where available, but do not infer that a case is false merely because the vendor published it.
Similarly, a company quote repeated in multiple articles remains one underlying claim unless additional independent evidence is found.

### The nonpublication problem

A case announcement may document intent rather than an implemented outcome.
A later case study may report an interim metric rather than long-run net value.
A completed project without a published case may have failed, succeeded privately, or produced an ambiguous result.
The absence of a public report is not itself evidence of failure.
The selection mechanism remains a hypothesis until observed or substantiated.

### Practical source comparison

When a supplier cites a customer, ask for the population of eligible customers, whether all were invited to participate, and how the featured case was chosen.
Request a reference from a customer with a comparable footprint and one with a difficult deployment, subject to consent.
An independent random sample from a complete customer frame would be stronger for prevalence than handpicked references, if access and consent permit it.
Reference calls alone do not substitute for audited operational records or a credible counterfactual.

When a public case is older, inspect whether data definitions, service levels, capacity, and labor constraints match today's setting.
When a case is recent, check whether enough time has elapsed to observe a stable outcome.
When a report is in another language, a competent translation can reveal a relevant case but cannot correct outcome selection by itself.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Successful-survivor list | All cases are award winners | Failures less visible | Inflated success fraction | Enumerate starts and discontinued projects |
| Large-firm dominance | Only household names appear | Large firms have publicity resources | Poor small-firm transfer | Stratify by relevant scale |
| Promotional denominator | Vendor reports three successes but not customer count | Featured stories selected | Rate cannot be computed | Request full eligible customer frame |
| Recency substitution | Article date used as rollout date | Publication and event dates differ | Wrong operating context | Capture both dates |
| Language restriction | Search only in English for global target | Retrieval excludes local sources | Regional undercoverage | Search target languages and document gaps |
| Duplicate counting | Identical quote across articles | One press release propagated | False corroboration | Map articles to underlying claim |
| Missing as failure | Blank outcome recoded zero | Unknown confused with negative | Downward distortion | Preserve unknown and bound results |
| Missing as success | Completion announcement marked successful | Implementation mistaken for benefit | Upward distortion | Define success and measure outcome |
| Post hoc eligibility | Unfavorable projects labeled irrelevant | Outcome changes inclusion rule | Biased sample | Freeze criteria and record deviations |
| Empty stratum weighting | No small or stopped cases found | Positivity failure | Unidentified population fraction | Obtain observations or retain bounds |
| Convenience weights | Weight set from observed counts | Selection probabilities unknown | False precision | Use defensible frame and design |
| Coverage overclaim | Search results described as all projects | Frame not validated | Unsupported denominator | Audit frame and report exclusions |
| Unmatured outcomes | Recent pilot lacks 12-month data | Follow-up not complete | Wrong success classification | Record censoring and follow up |
| Baseline drift | Benefit cited during demand collapse | Other factors change outcomes | False causal attribution | Use credible comparison and timing |
| Translation drift | Automatic translation changes metric | Meaning lost across languages | Misclassification | Bilingual verification and original text |
| Site multiplication | Ten sites counted as ten independent companies | Shared implementation and sponsor | Overstated independent evidence | Define project unit and cluster |
| Bad date filter | Search omits historical failures | Date choice reflects story visibility | Recency distortion | Prespecify event window |
| Selective outcome | Inventory reduction shown without service | Favorable metric chosen | Hidden business cost | Joint service and cost rule |
| Confidentiality overreach | Private project shared externally | Access exceeds permission | Legal or commercial harm | Approvals and minimization |
| Unbounded pilot | Pilot has no budget or stop criteria | Stories substitute for governance | Escalating exposure | Preapprove cap and decision gate |

A red flag indicates a question to investigate, not a verdict about a particular organization.
The direction of bias can be uncertain or even reversed.
For example, an incident database can overrepresent failures while a marketing page can overrepresent successes.
Combining both without knowing coverage does not automatically cancel the two distortions.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Frame covers target | Denominator refers to the decision population | Off-register or confidential projects | Reconcile multiple lists and report gaps |
| Unit is consistent | Each project has one contribution | One project counted by site and company | Deduplicate with parent-child IDs |
| Eligibility fixed | Avoid outcome-driven inclusion | Rules change after results are seen | Timestamp protocol and deviations |
| Outcome is comparable | Counts refer to the same definition | Different baselines or follow-up periods | Standardize rule and preserve originals |
| Outcome observed correctly | Numerator reflects actual status | Unverified marketing or translation | Trace and independently check claims |
| Observed projects represent strata | Needed for simple stratum extrapolation | Success influences disclosure within stratum | Compare observed and missing characteristics; sensitivity analysis |
| Positive inclusion chance | Needed for probability weighting | Some target units cannot be observed | Seek another frame or bound the result |
| Inclusion probabilities known | Needed for inverse-probability design weights | Convenience sample with unknown selection | Avoid claiming design-unbiased estimation |
| Group shares known | Needed for population standardization | External frame has no size distribution | Report scenarios, not a single weighted rate |
| Outcomes mature | A 12-month success rule is measurable | Project has only three months of follow-up | Track censoring and return later |
| Payoffs transfer | Investment calculation applies locally | Costs or service requirements differ | Local cost model and scenario ranges |
| Comparison identifies effect | Needed for causal claim | Concurrent changes or selection into treatment | Use design review and credible counterfactual |
| Translation preserves meaning | Non-English evidence is classified correctly | Technical terms or negations mistranslated | Bilingual review of relevant passages |
| Search is reproducible | Another reviewer can assess retrieval | Missing queries, dates, or inaccessible sources | Preserve a dated search log |

Observed differences across groups are not proof of discrimination, concealment, or a particular publication mechanism.
Inclusion probabilities rarely become known merely because one has a list of published cases.
Statistical uncertainty and selection uncertainty are different: a narrow interval for the selected set can coexist with large bias relative to the target.
Sensitivity bounds do not eliminate bias; they expose the consequences of missing information under stated assumptions.

## 12. Exercises

Use the synthetic data in Section 6 unless the exercise says otherwise.
State units, show calculations, and identify whether each conclusion is descriptive, causal, or hypothetical.
### Exercise 1. Draw the selection path

**Type:** Recall.

**Prompt:** Draw a six-stage flow from eligible attempt to extracted case, and mark at least three points where visibility can change.

**Deliverable:** A diagram and a short explanation of one missing-case mechanism.

**Acceptance criteria:** Include survival, publication, retrieval, and screening; avoid assuming every missing project failed.

**Time:** 12 minutes.
### Exercise 2. Identify the denominator

**Type:** Application.

**Prompt:** A supplier has 20 customer stories, 16 positive, from an undisclosed number of customers.

**Deliverable:** A two-sentence statement of what can and cannot be estimated.

**Acceptance criteria:** Compute 16/20 for stories only and explain why the customer-wide rate is unknown.

**Time:** 8 minutes.
### Exercise 3. Calculate selection distortion

**Type:** Calculation.

**Prompt:** Use the 40-project synthetic table to calculate observed rate, true rate, and difference.

**Deliverable:** Arithmetic with percentages and percentage-point difference.

**Acceptance criteria:** Use 11/14 and 15/40; do not label the difference a causal effect.

**Time:** 10 minutes.
### Exercise 4. Bound unknown outcomes

**Type:** Calculation.

**Prompt:** Assume only 14 selected outcomes and the complete 40-project frame are known.

**Deliverable:** Lower and upper logical bounds with a one-sentence interpretation.

**Acceptance criteria:** Assign all 26 unknowns to failure, then success; distinguish bounds from a confidence interval.

**Time:** 10 minutes.
### Exercise 5. Diagnose a weighted estimate

**Type:** Critique.

**Prompt:** A reviewer substitutes zero for the empty discontinued group and weights the four observed rates equally.

**Deliverable:** A critique and a statement of what additional information is needed.

**Acceptance criteria:** Identify undefined 0/0, positivity failure, and within-group selection; do not claim weighting fixes both.

**Time:** 12 minutes.
### Exercise 6. Design a search

**Type:** Design.

**Prompt:** A regional distributor wants evidence from large and small firms across three languages over five years.

**Deliverable:** A one-page protocol including terms, sources, eligibility, dates, translation, and exclusions.

**Acceptance criteria:** Include both favorable and unfavorable cases, sponsor disclosure, deduplication, and a documented search log.

**Time:** 25 minutes.
### Exercise 7. Choose a business gate

**Type:** Decision.

**Prompt:** The case-library success bounds span 27.5%–92.5%; full rollout would be costly.

**Deliverable:** A recommendation with owner, budget cap, outcome definition, stop rule, and evidence required for scale.

**Acceptance criteria:** Separate forecast, plan, target, and commitment; explain why observed stories alone do not authorize scale.

**Time:** 20 minutes.
### Exercise 8. Write an evidence memo

**Type:** Communication.

**Prompt:** A vendor claims a 20% inventory reduction in a named customer story; two articles repeat the release.

**Deliverable:** A four-row claim register and short decision memo.

**Acceptance criteria:** Label vendor claim, independent evidence if any, inference, and missing information; count the case once.

**Time:** 20 minutes.
## 13. Self-check answers

### Answer 1: Selection path

A valid sequence is eligible attempt → started project → outcome measured → organization consents → report published → report retrieved → case screened → outcome extracted.
Projects may disappear before outcome measurement because they stop early.
Reports may not be published because disclosure is confidential, regardless of success.
A published report may be missed because it uses another language or search term.
A good diagram keeps “unobserved” separate from “failed.”

### Answer 2: Denominator

The visible positive fraction is 16/20 = **80% of selected stories**.
The customer-wide fraction is unknown because the number of eligible customers and the selection process are unknown.
A story is not necessarily a distinct customer or a complete project outcome.
The fraction is descriptive of the reported collection, not a success probability for a new customer.

### Answer 3: Distortion

The observed fraction is 11/14 = **78.6%**.
The synthetic population fraction is 15/40 = **37.5%**.
The difference is approximately **41.1 percentage points**.
The comparison is possible only because the teaching example reveals the full synthetic register.
It measures a descriptive selection gap, not the causal effect of replenishment technology.

### Answer 4: Bounds

The 26 unknown outcomes yield a minimum of 11 successful projects and a maximum of 37.
The logical bounds are 11/40 = **27.5%** and 37/40 = **92.5%**.
They assume a complete frame, correct observed classifications, and a binary success rule.
They are not a confidence interval or a statement that all values within them are equally likely.

### Answer 5: Weighting

The discontinued group's observed rate is undefined because no outcome is observed.
Setting it to zero silently assigns all its projects as failures.
Even assigning an interval of zero to one leaves the additional untested assumption that the three observed groups are internally representative.
Useful next steps include finding discontinued projects, obtaining outcomes under appropriate permission, and reporting logical bounds.
A documented probability sample with positive inclusion probabilities can support design-based weighting, but a convenient public-story list does not supply those probabilities.

### Answer 6: Search rubric

A strong protocol states the target decision, eligible unit, project window, exact queries, platforms, and all three languages.
It includes sources beyond vendor pages, synonyms for cancellation and failure, deduplication rules, and a procedure for bilingual review.
It records exclusions, unknowns, search dates, sponsor relationships, and inaccessible sources.
It distinguishes a publication date from the measured outcome period.
A weak protocol searches only prominent English success stories and assumes the returned list is complete.

### Answer 7: Gate rubric

A strong recommendation caps pilot expenditure, designates an accountable owner, and sets a measurable outcome and stop date.
It explains how the comparison will be constructed and protects service levels.
It describes 27.5%–92.5% as frame-conditional logical bounds, not a probability forecast.
It states a plan to pilot, a target to be tested, and a commitment limited to authorized resources.
A weak recommendation multiplies the selected success fraction by a vendor saving and commits the entire network without further evidence.

### Answer 8: Memo rubric

The 20% figure is an attributed vendor claim until its measurement and provenance are checked.
Two press rewrites of the same release are two records but not necessarily two independent confirmations.
A valid inference is that the case merits further investigation, not that the vendor caused the reduction.
Missing information includes baseline, comparator, time, scope, costs, service effects, and how the customer was selected.
The memo should preserve the original claim and set a decision gate based on evidence actually available.

## 14. Interview questions
### Question 1: What is case selection bias?

**Strong answer:** Explain the difference between a target population and the observed cases, and give an outcome-dependent visibility mechanism.

**Weak answer:** Treats an accurate calculation on visible cases as automatically population-representative.
### Question 2: How does survivorship affect project evidence?

**Strong answer:** Includes stopped projects in the target where appropriate and distinguishes missing from failed.

**Weak answer:** Assumes all unpublished projects failed or only completed projects matter.
### Question 3: Why might size matter?

**Strong answer:** Connects publicity, resources, data maturity, and implementation capability to both observation and transfer.

**Weak answer:** Calls a large-company case false solely because the company is large.
### Question 4: What is different about vendor marketing?

**Strong answer:** Identifies commercial incentives, selection, claim provenance, and the need for independent corroboration.

**Weak answer:** Rejects every vendor claim automatically or treats every named customer story as proof.
### Question 5: How do you examine recency?

**Strong answer:** Separates start, intervention, outcome, publication, and retrieval dates and checks the current operating regime.

**Weak answer:** Uses the article date as the implementation date.
### Question 6: How do you address language bias?

**Strong answer:** Chooses languages from target coverage, logs queries, and arranges qualified translation while reporting remaining gaps.

**Weak answer:** Assumes an English search represents all countries.
### Question 7: Can weighting fix selection?

**Strong answer:** Requires known or defensible selection probabilities, overlap, and within-stratum assumptions; recognizes empty strata.

**Weak answer:** Assigns arbitrary weights to a convenience sample and calls the result unbiased.
### Question 8: What do worst-case bounds mean?

**Strong answer:** Computes extremes over unknown outcomes and states frame and measurement assumptions.

**Weak answer:** Calls a logical interval a 95% confidence interval.
### Question 9: How does selection differ from causation?

**Strong answer:** Separates who is observed from what would have happened without the intervention.

**Weak answer:** Says representative before/after outcomes prove causal benefit.
### Question 10: When would you approve a pilot?

**Strong answer:** Matches evidence standard to exposure and specifies cap, comparator, guardrails, owner, and decision date.

**Weak answer:** Treats the number of case studies as a substitute for a measured decision gate.
## 15. Key terms

Each term points to the section where its operational use is explained.

- [Target population](#3-the-core-mental-model) — units to which the decision question refers.
- [Sampling frame](#3-the-core-mental-model) — list from which cases could be sampled.
- [Observed set](#3-the-core-mental-model) — cases actually included and measured.
- [Unit of analysis](#4-definitions-and-vocabulary) — entity counted in a result.
- [Eligibility](#5-the-mechanics-step-by-step) — rule for including an attempt.
- [Selection](#3-the-core-mental-model) — process governing entry to the observed set.
- [Survivorship](#10-failure-modes-and-red-flags) — visibility advantage of units that remain.
- [Size bias](#5-the-mechanics-step-by-step) — uneven observation or transfer by scale.
- [Vendor marketing](#9-industry-practice-and-case-evidence) — commercially motivated source channel.
- [Recency bias](#5-the-mechanics-step-by-step) — distorted attention to recent material.
- [Language bias](#8-data-standards-and-regulatory-context) — uneven coverage across languages.
- [Outcome](#4-definitions-and-vocabulary) — result under a specified measurement rule.
- [Success fraction](#6-worked-example) — share meeting a defined success criterion.
- [Missing outcome](#11-assumptions-and-limits) — eligible unit without a usable result.
- [Coverage](#6-worked-example) — observed count divided by a specified frame count.
- [Positivity](#11-assumptions-and-limits) — nonzero selection opportunity in target strata.
- [Inclusion probability](#5-the-mechanics-step-by-step) — probability a unit is sampled.
- [Weight](#5-the-mechanics-step-by-step) — assigned contribution to an estimator.
- [Sensitivity bound](#6-worked-example) — range under stated missingness assumptions.
- [Comparator](#3-the-core-mental-model) — basis for evaluating an outcome or effect.
- [Dependence](#10-failure-modes-and-red-flags) — sources sharing an underlying claim.
- [Censoring](#11-assumptions-and-limits) — outcome not yet fully observable at cutoff.
- [Transfer](#9-industry-practice-and-case-evidence) — relevance to a new setting.
- [Decision gate](#6-worked-example) — conditions for advancing or stopping investment.
- [Provenance](#5-the-mechanics-step-by-step) — origin and chain of a claim.
- [Claim](#9-industry-practice-and-case-evidence) — assertion attributed to its source.
- [Inference](#9-industry-practice-and-case-evidence) — interpretation beyond direct observation.
- [Synthetic case](#6-worked-example) — constructed example, not a real-world outcome.

## 16. Sources

These sources are suitable starting points for checking the methods and public examples.
A source is cited for the specific role described, not as proof of an unrelated commercial impact.

1. Page, M. J., et al. “The PRISMA 2020 statement: an updated guideline for reporting systematic reviews.” *BMJ* 372 (2021): n71. [https://www.bmj.com/content/372/bmj.n71](https://www.bmj.com/content/372/bmj.n71). Reporting guidance and selection-flow transparency; not a business-law requirement.
2. Higgins, J. P. T., et al., editors. *Cochrane Handbook for Systematic Reviews of Interventions*, Chapter 13, “Assessing risk of bias due to missing results in a synthesis.” [https://training.cochrane.org/handbook/current/chapter-13](https://training.cochrane.org/handbook/current/chapter-13). Methodological discussion of missing results, not a supply-chain prevalence estimate.
3. World Bank. *Living Standards Measurement Study: Sampling*. [https://www.worldbank.org/en/programs/lsms](https://www.worldbank.org/en/programs/lsms). General source for survey-design context; weights require an appropriate sampling design.
4. Kaggle. “M5 Forecasting — Accuracy.” [https://www.kaggle.com/competitions/m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy). Public description of the benchmark task and dataset; benchmark performance is not commercial return.
5. U.S. Securities and Exchange Commission. “Search Filings.” [https://www.sec.gov/search-filings](https://www.sec.gov/search-filings). Access route to company filings; not a register of all private projects.
6. National Academies of Sciences, Engineering, and Medicine. *Nonresponse in Social Science Surveys: A Research Agenda* (2013). [https://nap.nationalacademies.org/catalog/18293/nonresponse-in-social-science-surveys-a-research-agenda](https://nap.nationalacademies.org/catalog/18293/nonresponse-in-social-science-surveys-a-research-agenda). General nonresponse concepts; application to case libraries requires judgment.

The synthetic figures and investment values in this lesson are instructional assumptions, not sourced estimates.
Access dates and current legal obligations should be checked for a real review.

## 17. Links

- [Module 28 topic index](_index.md)
- [Case anatomy](01_case-anatomy.md)
- [Evidence grades A–E](02_evidence-grades.md)
- [The transfer test](03_transfer-test.md)
- [Module 28 survey](../../../course/28_domain_case_library.md)
- [Simulation, optimization, and production labs survey](../../../course/30_simulation_optimization_production_labs.md)
