# Industry and Market

> Module 22 — Sources and Further Reading
> Topic 07 of 09
> Estimated study time: 2.5 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

Market numbers enter biomedical data work through decks, business cases, and diligence memos. They are used to size an opportunity, justify a hire, set a price, or decide whether a product category is worth entering. They are also the least verifiable numbers in the module, because the underlying data are often proprietary, the method is often undisclosed, and the definition of the market is often chosen to produce the desired answer.

The failures are predictable:

- A total addressable market is computed top-down from a national spend figure, with no check that the product could ever capture a fraction of it.
- A vendor white paper surveys the vendor's own customers and reports that most of them plan to buy.
- A press release announces a partnership as adoption, and the partnership appears in a market map as a customer.
- A growth rate is quoted from a report that used a different market definition from the one in the deck.
- A funding total is compared across years that used different inclusion criteria for deals.
- A percentage is reported without a denominator, so a doubling from 2 to 4 becomes "100 percent growth".

The people affected are the investor who cannot underwrite a claim, the product manager whose roadmap is justified by a number nobody can check, the analyst who is asked to "find a source" for a figure that was invented in a meeting, and the founder whose pitch is dismantled by a diligence team that knows the category.

This topic teaches how to read and evaluate market and industry sources: what the number means, how it was constructed, what the denominator is, and what the source's interest is. It does not teach finance. It teaches the source work that prevents a market claim from becoming an organisational fact.

The distinction matters because market numbers are used to make commitments. A hiring plan, a fundraise, and a product investment all rest on market claims, and the claims are often less examined than the model that will be built to serve them.

A reader who masters this topic will ask three questions about every market number: what is included, what is the denominator, and who paid. Those questions do not require finance expertise, and they expose most of the weak claims in a deck.

The questions also work in reverse. When writing a market sentence, answer them in the sentence, and the sentence becomes defensible without becoming longer than a paragraph.

> **Note:** A market size is a model, not a measurement. Even the best market reports are estimates built from assumptions, and the assumptions are the part that matters. A number without its method is a number without a meaning.

## 2. Learning outcomes

By the end of this topic you can:

1. **Decompose** a market claim into definition, population, spend per unit, and method. Assessed by Exercise 1 and self-check item 1.
2. **Distinguish** a filing, a government statistic, an analyst estimate, a vendor survey, and a press release, and state what each can support. Assessed by Exercise 2.
3. **Recompute** growth rates and shares from reported endpoints and detect arithmetic that does not reconcile. Assessed by Exercise 3 and section 7.
4. **Identify** the denominator and the inclusion criteria behind a market or funding number. Assessed by Exercise 4.
5. **Detect** the common market-report failures: TAM inflation, survey bias, category definition drift, and period mismatch. Assessed by Exercise 5.
6. **Assess** a source's commercial interest and its effect on the claim. Assessed by Exercise 6.
7. **Write** a market sentence that names the definition, the year, the method, and the uncertainty. Assessed by Exercise 7.
8. **Decide** when a market question needs primary research rather than a secondary source. Assessed by Exercise 8.

The outcomes are ordered from decomposition to decision. A learner who can decompose a claim but cannot decide whether to trust it has the vocabulary without the judgement.

Each outcome is observable in a document, and the exercises are designed to produce those documents rather than to test recall.

## 3. The core mental model

A market claim is a product of four factors: a definition, a population, a spend per unit, and a method. Change any one and the number changes, often by an order of magnitude.

```text
                    MARKET CLAIM
        "$X billion market growing at Y%"
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
    DEFINITION            POPULATION            SPEND PER UNIT
   what is in the        who is counted,       price, utilisation,
   category, what is     geography, year       reimbursement
   excluded
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                          METHOD
        top-down | bottom-up | survey | filing | analyst model
                              |
                              v
                    SOURCE AND INTEREST
        who produced it, who paid, what they sell
                              |
                              v
                    CLAIM WITH CONDITIONS
        definition | year | method | range | uncertainty
```

Read the diagram downwards. The definition decides which companies and products are counted. The population decides the denominator. The spend per unit decides the multiplier. The method decides how the three are combined. The source's interest decides how much confidence the number deserves.

**Top-down and bottom-up answers differ by design.** A top-down estimate starts from a national spend figure and takes a share. A bottom-up estimate starts from units, prices, and adoption. The first is fast and usually larger; the second is slower and usually more defensible.

**The denominator is the argument.** "The market is $10 billion" means nothing without knowing whether it counts software, services, devices, and drugs, and whether it is US or global, and whether it is 2024 revenue or 2030 projection.

**Market reports are products.** Analysts sell reports, vendors sell products, trade associations advocate, and consultancies sell projects. The interest does not make the number wrong; it makes the number a claim to be checked.

**Funding totals are especially fragile.** Deal counts, dollar totals, and median sizes move for reasons that include the inclusion criteria, the stage definitions, and the treatment of undisclosed deals. A year-on-year comparison is only valid if the method is stable.

| Source type | What it measures | Typical bias | Best use |
|---|---|---|---|
| SEC filing | Reported revenue for a defined entity | Timing and segment definition | Company size and growth |
| Government statistic | Population, employment, spending | Coverage and lag | Denominators and baselines |
| Analyst report | Modelled market estimate | Definition and client base | Orientation and ranges |
| Trade association | Industry aggregate | Membership and advocacy | Sector context |
| Vendor survey | Customer intentions | Selection of respondents | Hypothesis generation |
| Press release | An announcement | Framing and omission | Event evidence only |

The table is ordered by verifiability, not by usefulness. A government statistic is highly verifiable and may not answer the question; an analyst report is less verifiable and may be the only source for the category. The right choice depends on the claim, and the wrong choice is to treat all six as equivalent.

A useful discipline is to name the source type in the sentence. "The report estimates" and "the filing reports" are different claims, and a reader who sees the difference can calibrate.

| Source type | Sentence pattern | What the reader assumes |
|---|---|---|
| Filing | "The company reported" | Audited, defined entity |
| Government | "The agency reports" | Published method, defined population |
| Analyst | "The report estimates" | Model, category definition |
| Vendor | "The vendor states" | Commercial interest |
| Press release | "The company announced" | Event, not outcome |

The pattern is a small discipline that changes how a number is read. A reader who sees "the report estimates" knows to ask for the method; a reader who sees "the company reported" knows the number is audited within its accounting rules.

## 4. Definitions and vocabulary

### 4.1 Market sizing terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Total addressable market | Revenue if every possible customer bought. | Used as a forecast. | Currency per year |
| Serviceable addressable market | The segment a product can serve with its model. | Assumed equal to TAM. | Currency per year |
| Serviceable obtainable market | The share realistically winnable. | Omitted from the deck. | Currency per year |
| Top-down estimate | A share applied to an aggregate. | Presented as bottom-up. | Method |
| Bottom-up estimate | Units times price times adoption. | Assumed more precise than it is | Method |
| Penetration | The proportion of the population using a product. | Confused with market share. | Proportion |
| Average selling price | Revenue per unit. | Confused with list price. | Currency per unit |
| Utilisation | The frequency of use. | Assumed constant across settings. | Count per period |
| Reimbursement | Payment for a service or product. | Confused with price. | Currency per unit |

### 4.2 Growth and comparison terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Compound annual growth rate | The constant annual rate between two values. | Applied to a non-compounding series. | Proportion per year |
| Year-on-year growth | Change versus the same period a year earlier. | Compared across different period lengths. | Proportion |
| Sequential growth | Change versus the previous period. | Annualised incorrectly. | Proportion |
| Nominal versus real | Whether inflation is removed. | Mixed in one comparison. | Currency basis |
| Base effect | A small prior period inflating growth. | Ignored in a growth claim. | Context |
| Pro forma | Adjusted for events such as acquisitions. | Treated as reported. | Accounting basis |
| Constant currency | Adjusted for exchange-rate changes. | Confused with nominal. | Currency basis |

### 4.3 Company and funding terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| 10-K | Annual report filed with the SEC. | Treated as audited in every detail. | Filing |
| 10-Q | Quarterly report. | Compared without seasonal adjustment. | Filing |
| 8-K | Current report of a material event. | Treated as a strategy statement. | Filing |
| Revenue | Recognised sales under accounting rules. | Confused with bookings or billings. | Currency |
| Bookings | Contracted future revenue. | Reported as revenue. | Currency |
| Burn rate | Net cash use per period. | Confused with spend. | Currency per month |
| Runway | Months of cash at the current burn. | Computed on an optimistic burn. | Months |
| Venture round | Equity financing at a stage. | Counted as revenue. | Financing event |
| Mega-deal | A round above a threshold, often $100 million. | Compared across reports with different thresholds. | Currency |
| Median deal size | The midpoint of deal values. | Replaced by the mean. | Currency |

### 4.4 Health-system and payment terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Payer | The entity paying for care. | Treated as the patient. | Entity |
| Provider | The entity delivering care. | Confused with the clinician. | Entity |
| Capitation | Fixed payment per patient per period. | Confused with fee-for-service. | Currency per member per month |
| Fee-for-service | Payment per service. | Assumed to reward volume only. | Payment model |
| Bundled payment | A single payment for an episode. | Confused with capitation. | Payment model |
| Prior authorisation | Payer approval before a service. | Assumed to be clinical review. | Process |
| Coverage with evidence development | Payment tied to evidence collection. | Treated as unrestricted coverage. | Policy |
| Formulary | The list of covered drugs. | Confused with clinical guidelines. | List |

### 4.5 Evidence-quality terms for market sources

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Method disclosure | Whether the report explains how numbers were derived. | Assumed because the report is long. | Judgement |
| Sample frame | The population from which survey respondents were drawn. | Omitted from vendor surveys. | Description |
| Response rate | Respondents as a proportion of those invited. | Ignored in interpretation. | Proportion |
| Self-selection | Participation driven by interest. | Treated as random. | Bias type |
| Client base | The customers an analyst serves. | Assumed to be neutral. | Conflict |
| Advocacy position | A trade body's policy goal. | Read as neutral analysis. | Conflict |
| Forward-looking statement | A projection, often with legal safe harbour. | Read as a forecast. | Claim type |
| Panel | A defined set of reporting units followed over time. | Treated as a census | Sample design |
| Census | Complete enumeration of a population. | Assumed in panel data | Coverage |
| Restatement | A revision of previously reported figures. | Ignored in trend analysis | Accounting event |
| Segment change | A redefinition of reported segments. | Compared with the old segments | Disclosure change

## 5. The mechanics, step by step

The procedure has 22 steps in five phases. It evaluates a market claim the way a diligence team would: decompose it, find the method, check the arithmetic, assess the interest, and rewrite the sentence with its conditions.

The procedure does not produce a better market estimate. It produces a defensible statement about what the available sources support, which is usually a range with a stated method rather than a single number.

### Phase A — Decompose the claim

**Step 1. Write the market sentence with its four factors.**
Input: the sentence in the deck or memo.
Output: the definition, the population, the spend per unit, and the method, each stated explicitly or marked unknown.
Rationale: a market claim without its factors cannot be checked, and the factors are usually recoverable from the source even when the deck omits them.
Branch: if the sentence gives only a number, write "method unknown" and proceed to the source.
Edge case: a sentence can contain two claims, such as a size and a growth rate, each with its own method.
Check: are the four factors present.
Record: the decomposition.

**Step 2. Identify the category definition.**
Input: the sentence and the source's methodology.
Output: what is included and excluded, and the boundary rules.
Rationale: most market disagreements are definitional, not numerical.
Branch: if the source does not define the category, the number is not comparable with any other.
Edge case: a category can include services, software, and devices in one figure.
Check: is the definition stated.
Record: the definition.

**Step 3. Identify the population and geography.**
Input: the source's scope section.
Output: the countries, care settings, and customer types included.
Rationale: a global figure applied to a US opportunity overstates the market by the ratio of the two.
Branch: if the geography is unclear, treat the number as unusable for a local decision.
Edge case: a report can cover a region but exclude a major market.
Check: is the geography stated.
Record: the scope.

**Step 4. Identify the year and the basis.**
Input: the source's date and the figures.
Output: whether the number is historical revenue, current run rate, or a projection, and whether it is nominal or real.
Rationale: a 2030 projection and a 2024 revenue are different objects, and mixing them is common.
Branch: if the basis is unclear, assume the most favourable interpretation is being used and discount it.
Edge case: a report can mix historical and projected years in one chart.
Check: is the year and basis stated.
Record: the basis.

### Phase B — Find the method

**Step 5. Locate the methodology section.**
Input: the report or the source page.
Output: a description of how the number was derived, or a finding that no method is disclosed.
Rationale: the method determines whether the number is a measurement, a survey, or a model.
Branch: if no method is disclosed, the source is marketing rather than analysis.
Edge case: a short methodology note can be buried in an appendix.
Check: is there a method.
Record: the method summary.

**Step 6. Classify the method.**
Input: the methodology.
Output: top-down, bottom-up, survey, filing-based, or analyst model.
Rationale: the method determines the likely direction of error and the appropriate use.
Branch: a top-down estimate is useful for orientation and weak for a product decision.
Edge case: a report can combine methods, which should be described.
Check: is the method named.
Record: the classification.

**Step 7. Find the denominator.**
Input: the method and the figures.
Output: the base population, the number of units, or the number of respondents.
Rationale: a percentage without a denominator is uninterpretable, and the denominator is often the assumption that drives the result.
Branch: if the denominator is not stated, derive it from the other figures and check whether it is plausible.
Edge case: a survey can report percentages without the number of respondents.
Check: is the denominator stated.
Record: the denominator.

**Step 8. Identify the key assumptions.**
Input: the method and any model description.
Output: a list of assumptions with their values and sensitivity.
Rationale: market models are assumption-driven, and the assumptions are often more important than the data.
Branch: if the model is not disclosed, test the implied assumption from the output.
Edge case: a single assumption, such as penetration, can drive most of the variance.
Check: are the assumptions visible.
Record: the assumptions.

### Phase C — Check the arithmetic

**Step 9. Recompute the growth rate from the endpoints.**
Input: the start and end values and the period.
Output: the implied compound annual growth rate.
Rationale: a stated growth rate that does not reconcile with the endpoints is a source error or a different period.
Branch: if the implied rate differs from the stated rate, use the implied rate and note the discrepancy.
Edge case: a simple growth rate and a compound rate differ over long periods.
Check: does the arithmetic reconcile.
Record: the implied rate.

**Step 10. Recompute the shares.**
Input: the components and the total.
Output: the sum of components and any residual.
Rationale: components that do not sum to the total indicate a definitional difference or an omission.
Branch: if the residual is large, the total includes a category not shown.
Edge case: rounding can produce small residuals.
Check: do the components sum.
Record: the residual.

**Step 11. Check the period alignment.**
Input: the periods being compared.
Output: whether the comparison uses equal periods and the same inclusion rules.
Rationale: a half-year compared with a full year, or a period with different deal criteria, produces a false trend.
Branch: if the periods differ, annualise or compare like with like.
Edge case: a report can change its inclusion criteria between editions.
Check: are the periods aligned.
Record: the alignment.

**Step 12. Check the units and currency.**
Input: the figures.
Output: the currency, the unit (millions or billions), and whether it is nominal or real.
Rationale: unit and currency errors produce order-of-magnitude mistakes that survive into decks.
Branch: if the currency is unspecified, check the source's home market.
Edge case: a global figure can be in USD while the underlying data are local.
Check: are the units explicit.
Record: the units.

### Phase D — Assess the source and interest

**Step 13. Identify the publisher and its business model.**
Input: the report cover and the publisher's site.
Output: who produced it, who paid, and what they sell.
Rationale: the business model shapes the definition and the framing.
Branch: if the publisher sells products in the category, treat the number as promotional.
Edge case: an analyst firm can be paid by the companies it evaluates.
Check: is the interest identified.
Record: the interest.

**Step 14. Check the sample and the response rate for surveys.**
Input: the survey methodology.
Output: the sample frame, the number of respondents, and the response rate.
Rationale: a survey of the vendor's customers is not a survey of the market.
Branch: if the sample frame is the customer list, the result describes customers.
Edge case: a large sample from a biased frame is still biased.
Check: is the sample frame stated.
Record: the sample.

**Step 15. Check the publication date and the data date.**
Input: the report date and the underlying data period.
Output: both dates.
Rationale: a report published recently can use data that are a year old.
Branch: if the data date is not stated, treat the number as at least a year stale.
Edge case: a report can be updated without changing its data.
Check: are both dates stated.
Record: the dates.

**Step 16. Check the claims against a filing or government statistic.**
Input: the company or market in question.
Output: a comparison with an audited or official figure, where one exists.
Rationale: filings and government statistics are the closest thing to ground truth in this domain.
Branch: if no official figure exists, the market claim remains an estimate.
Edge case: a private company has no filing, so its revenue is an estimate.
Check: is there an official comparator.
Record: the comparison.

### Phase E — Rewrite and record

**Step 17. Write the claim with its conditions.**
Input: the decomposition and checks.
Output: a sentence with the definition, the geography, the year, the method, and a range.
Rationale: the conditions are what make the claim checkable and prevent it from being reused out of context.
Branch: if the range is wide, say so rather than choosing a midpoint.
Edge case: a range from two methods is more informative than one number.
Check: does the sentence include the conditions.
Record: the sentence.

**Step 18. State the uncertainty and the direction of likely bias.**
Input: the source's interest and method.
Output: a sentence naming the likely direction and magnitude of the bias.
Rationale: a vendor estimate is likely optimistic, and the reader should know.
Branch: if the direction is unknown, say so rather than implying neutrality.
Edge case: a conservative estimate can also be biased, for example to support a pricing argument.
Check: is the direction stated.
Record: the uncertainty.

**Step 19. Record the source and the access date.**
Input: the report or page.
Output: a citation with the publisher, title, date, and URL.
Rationale: the source must be checkable, and market pages change.
Branch: if the report is paywalled, record the public summary and the fact that the full method was not seen.
Edge case: a report can be withdrawn or revised.
Check: is the citation complete.
Record: the citation.

**Step 20. Record what the source does not support.**
Input: the decomposition and checks.
Output: a list of claims the source cannot support.
Rationale: the most useful part of the record is the boundary of the claim.
Branch: if the source supports a category estimate and not a company opportunity, say so.
Edge case: a global estimate can support a category narrative and not a pricing decision.
Check: is the boundary stated.
Record: the limits.

**Step 21. Set the review trigger.**
Input: the source's update cycle.
Output: a review date and event triggers.
Rationale: market numbers are revised, and a number used in a board deck should be rechecked before reuse.
Branch: the triggers should include a new edition, a revised figure, and a major transaction in the category.
Edge case: an annual report can be revised mid-year.
Check: is the trigger recorded.
Record: the trigger.

**Step 22. Decide whether primary research is needed.**
Input: the decision and the available sources.
Output: a recommendation to use the source, commission research, or change the decision.
Rationale: some questions cannot be answered from secondary sources, and the honest answer is to collect data.
Branch: if the decision depends on a number that no source can support, the project needs primary research or a different decision.
Edge case: a small survey of target buyers can be more useful than a large market report.
Check: is the recommendation stated.
Record: the recommendation.

The procedure produces a market sentence that a diligence team can check and a record of what the source does not support. It does not produce a market forecast, and it should not be presented as one.

### Market claim red flags

| Red flag | What it suggests | What to ask |
|---|---|---|
| A round number with no source | Estimate presented as fact | Where did the number come from |
| A growth rate with no base year | Base effect or cherry-picked period | What are the endpoints |
| A category name that is new | Definition created for the report | What is included |
| A vendor as the only source | Promotional estimate | Who else estimates this |
| A percentage with no denominator | Incomplete reporting | Percentage of what |
| A projection beyond five years | Compounding assumptions dominate | What is the implied rate |
| A global figure for a local plan | Geography mismatch | What is the local number |
| A survey of customers | Selection bias | Who was surveyed |

Three questions expose most of these: what is included, what is the denominator, and who paid for the analysis.

### How to test an assumption

| Assumption | Test | Data source |
|---|---|---|
| Penetration rate | Compare with analogous products | Adoption studies, filings |
| Price per unit | Compare with list and contract prices | Procurement data, filings |
| Eligible population | Compare with a government statistic | Census, payer data |
| Growth rate | Compare with adjacent categories | Multiple reports |
| Retention | Compare with published churn | Filings, surveys |

The test does not have to be precise. A factor-of-two check on the most important assumption is usually enough to tell whether the claim is plausible.

A market model is only as good as its weakest assumption, and the weakest assumption is usually the one that is not stated.

### What a market source cannot support

- A company-level revenue forecast from a category estimate.
- A pricing decision from a market size.
- A product decision from a funding total.
- A customer count from a market value.
- A growth forecast from one year of data.

Each of these is a different question, and the market report is designed to answer the category question, not the company one.

The distinction matters most in diligence. A category estimate can support a narrative about why the market matters, and it cannot support a revenue projection for a specific product. The projection needs a bottom-up model with named customers, prices, and adoption.

| Question | Source that can support it | Source that cannot |
|---|---|---|
| Is the category growing | Analyst report, filing trend | Press release |
| What will this product sell | Bottom-up model, pilots | Category estimate |
| What price will a hospital pay | Procurement data, interviews | List price |
| How many eligible sites exist | Government or association data | Analyst category |
| What share can we win | Sales pipeline, comparables | TAM |

The table is also a defence against a common request. When someone asks for "a source for the revenue projection", the honest answer is that no secondary source can supply it, and the conversation can move to the model that can.

### Questions to ask about a survey

| Question | Why it matters | Red flag |
|---|---|---|
| Who was invited | The sample frame determines the population | The vendor's customer list |
| How many responded | The response rate bounds the inference | "Hundreds of executives" with no denominator |
| How were they selected | Selection determines bias | Self-selected webinar attendees |
| What was asked | Wording shapes answers | Leading questions about future purchases |
| Who paid | Interest shapes framing | The vendor whose product is surveyed |
| What is the margin of error | Precision of the estimate | Precision claimed without a sample size |

A survey of 200 customers of a product can tell you about those customers. It cannot tell you about the market, and the report should say so.

### Reading a funding series

- Check the definition: which companies, which stages, which geographies.
- Check the inclusion rule: are undisclosed deals counted.
- Check the period: half-year, full-year, or year-to-date.
- Check the mix: dollars, deal count, median size, and concentration.
- Check the revision: reports can restate prior periods.
- Check the interest: who publishes the series and who buys it.

A funding total without the deal count and the concentration is a number without a market structure.

Two series can disagree and both be right, because they define the category differently. The disagreement is information, and the correct response is to report both rather than to choose the more convenient one.

- A dollar total without a deal count hides concentration.
- A deal count without a dollar total hides size.
- A median without a distribution hides the tail.
- A growth rate without a base hides the effect.
- A comparison without a period hides the seasonality.
- A market map without a source hides the selection.

## 6. Worked example

A strategy memo opens with: "Digital health funding is collapsing; the sector raised far less in 2026 than in previous years." The sentence is used to argue for a conservative hiring plan.

### Framing the claim

The sentence has three parts:

1. **A directional claim.** Funding is falling.
2. **A comparative claim.** 2026 is lower than previous years.
3. **A magnitude claim.** The fall is large enough to justify a specific decision.

The first two are empirical and checkable. The third is an inference from the data to a management decision, and it depends on which funding measure is used.

| Sub-claim | Type | What could support it |
|---|---|---|
| Funding fell in 2026 | Empirical | A consistent series with a stated method |
| The fall is large | Empirical | A comparison with a defined baseline |
| The fall justifies a hiring plan | Inference | A link between funding and the company's market |

### Candidate sources

| # | Candidate | Type | Role |
|---|---|---|---|
| C1 | Rock Health H1 2026 funding report | Industry report | The primary sector series |
| C2 | Rock Health Q3 2026 update | Industry report | Year-to-date context |
| C3 | A news article summarising the report | Secondary | The likely origin of "collapsing" |
| C4 | A vendor blog predicting a downturn | Commentary | The framing of the claim |
| C5 | A company filing from a digital health peer | Filing | A ground-truth comparator for one company |

### Scoring the candidates

**C1, the H1 report.**
- Authority: high for its own series, with a stated methodology.
- Currency: current, published in July 2026.
- Applicability: US digital health startups, which is the memo's sector.
- Accessibility: public.
- Provenance: methodology and definition are described.
- Verdict: primary support for the dollar and deal counts.

**C2, the Q3 update.**
- Authority: high, same series.
- Currency: current, October 2026.
- Applicability: same sector.
- Accessibility: public.
- Provenance: same methodology.
- Verdict: primary support for the year-to-date picture.

**C3, the news article.**
- Authority: moderate, secondary.
- Currency: current.
- Applicability: depends on the article's scope.
- Accessibility: public.
- Provenance: usually cites the report.
- Verdict: background; the report governs.

**C4, the vendor blog.**
- Authority: low, with a commercial interest in a narrative.
- Currency: unknown.
- Applicability: general.
- Accessibility: public.
- Provenance: no method.
- Verdict: background only.

**C5, the peer filing.**
- Authority: high for that company.
- Currency: depends on the filing date.
- Applicability: one company, not the sector.
- Accessibility: public.
- Provenance: audited.
- Verdict: comparator, not sector evidence.

### The arithmetic

The H1 2026 report gives $7.4 billion across 244 deals, against $6.4 billion across 245 deals in H1 2025.

- Year-on-year dollar change: (7.4 − 6.4) / 6.4 = 15.6 percent.
- Deal count change: (244 − 245) / 245 = −0.4 percent.
- Quarterly sum: 4.2 + 3.2 = 7.4, which reconciles with the total.
- Mega-deal capital: 45 percent of 7.4 = $3.33 billion across 20 deals.

The Q3 2026 update adds $3.7 billion across 114 deals, for a year-to-date total of $11.1 billion across 358 deals.

The arithmetic says the dollar total rose, the deal count was flat, and the capital was concentrated. That is not a collapse; it is a market where more money went into fewer, larger deals.

### Verdict

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| Funding is collapsing | Reject | "Dollar funding rose 15.6 percent year on year in H1 2026." |
| 2026 is lower than previous years | Reject for H1 dollars | "H1 2026 dollars exceeded H1 2025; the deal count was flat." |
| The sector is under stress | Conditional | "Capital concentrated in 20 mega-deals, which took 45 percent of H1 dollars." |
| The fall justifies a hiring plan | Unsupported | Requires a company-specific link, not a sector total |

The honest sentence is more interesting than the original. It says that capital is available and that it is concentrating, which is a different strategic signal from a collapse. A company outside the mega-deal tier may indeed face a harder market, and that is a claim about its segment, not about the sector.

### Two diagnostics that would change the verdict

**Diagnostic 1: full-year data show a decline.** If the Q4 data bring the full-year total below 2025, the directional claim becomes supportable for the annual period even though the half-year comparison went the other way. The check is the year-end report, not the mid-year one.

**Diagnostic 2: the relevant segment is early-stage deals.** If the company is raising a seed or Series A round, the relevant series is the count and size of early-stage deals, not the sector total. A flat total with concentration in mega-deals can coexist with a much weaker early-stage market, and the memo's claim would be supportable for that segment.

A third diagnostic, weaker, is a methodology change in the report's inclusion criteria. If the series changed what counts as a deal, the year-on-year comparison is not like for like, and the verdict would become "not comparable" rather than true or false.

### What the team should record

The ledger records the report, the edition, the definition, the figures, and the arithmetic. It also records what the source does not support: a company-level revenue forecast, a hiring decision, and a claim about the early-stage segment.

The memo's sentence is rewritten to the concentration finding, with the H1 comparison and the year-to-date figure. The hiring decision is then argued from the company's pipeline rather than from a sector total, which is the only evidence that can support it.

### What the numbers do and do not show

| Quantity | What it shows | What it does not show |
|---|---|---|
| H1 dollar total | Capital deployed in the period | Company-level demand |
| Deal count | Breadth of financing | Size of the average deal |
| Median deal size | The typical round | The distribution's tail |
| Mega-deal share | Concentration | The health of early-stage companies |
| Year-to-date total | Progress against the prior year | The full-year outcome |
| Average deal size | Capital per deal | The typical company |
| Dry powder | Committed but undeployed capital | Deployment timing |

The table is a reminder that a funding series is a description of financing activity, not a measure of the sector's clinical or commercial success. A sector can attract capital and still fail to produce durable companies.

A reader who wants to make a strategic claim from these numbers needs a second source, such as hiring data, procurement data, or company filings, and the claim should name both.

### What each stage shows

| Stage | What the data show | What they do not show |
|---|---|---|
| Seed | New company formation | Product viability |
| Early stage | First institutional validation | Commercial traction |
| Late stage | Scaling of proven models | Profitability |
| Mega-deals | Concentration of capital | Sector-wide health |
| Exits | Realised outcomes | Future formation |

A collapse in seed funding is a different signal from a collapse in late-stage funding, and a total that mixes them can hide both.

### Diligence questions for a funding claim

- What is the source series, and how does it define the category.
- Is the comparison like for like in period and inclusion.
- Are dollars, deals, and concentration all reported.
- Is the stage mix stable across the comparison.
- Is the prior period restated.
- Does the claim concern the sector or a segment.

A claim that survives these questions is a claim a diligence team can use. A claim that does not is a narrative.

- A narrative can be useful for orientation and cannot be used for a forecast.
- A segment claim needs segment data, not a sector total.
- A company claim needs company data, not a category estimate.

The hierarchy is simple: sector data for context, segment data for strategy, company data for a plan. Most bad decks use sector data for all three.

The hierarchy also explains why a market report can be simultaneously accurate and unhelpful. It is accurate about the sector, and the decision is about a company or a segment, and the report was never designed for that question.

| Level | Data that fits | Decision it supports |
|---|---|---|
| Sector | Analyst series, government statistics | Whether to enter a category |
| Segment | Segment reports, utilisation data | Where to compete |
| Company | Filings, pipeline, pilots | What to build and price |
| Customer | Interviews, procurement data | Whether a buyer will pay |

## 7. Code, query, or calculation

The script recomputes the quantities that market reports assert. It checks that quarters sum to the total, that a stated growth rate matches the endpoints, that a mega-deal share matches its dollar figure, and that a claimed CAGR is consistent with the start and end values.

```python
"""Market arithmetic checks."""

TOTAL_H1_2026 = 7.4
TOTAL_H1_2025 = 6.4
Q1_2026 = 4.2
Q2_2026 = 3.2
MEGA_SHARE = 0.45


def yoy(current, prior):
    return current / prior - 1


def cagr(start, end, years):
    return (end / start) ** (1 / years) - 1


def check(name, computed, stated, tol=0.005):
    ok = abs(computed - stated) <= tol
    return name, computed, stated, "ok" if ok else "MISMATCH"


rows = []
rows.append(check("H1 total vs quarters", Q1_2026 + Q2_2026, TOTAL_H1_2026))
rows.append(check("H1 year-on-year growth", yoy(TOTAL_H1_2026, TOTAL_H1_2025), 0.156, 0.002))
rows.append(check("mega-deal capital ($B)", TOTAL_H1_2026 * MEGA_SHARE, 3.33, 0.01))
rows.append(check("implied CAGR 50->200 in 5y", cagr(50, 200, 5), 0.3195, 0.001))

print(f"{'check':34s} {'computed':>10s} {'stated':>10s}  result")
print("-" * 70)
for name, computed, stated, result in rows:
    print(f"{name:34s} {computed:10.4f} {stated:10.4f}  {result}")
print("-" * 70)

start, end, years, claimed = 50, 200, 5, 0.25
implied = cagr(start, end, years)
print(f"claimed CAGR {claimed:.1%}; implied by {start}->{end} over {years}y: {implied:.1%}")
print("reconciles:", abs(implied - claimed) < 0.01)

components = {"mental health": 1.9, "weight management": 1.4, "cardiovascular": 0.9}
print("component sum:", round(sum(components.values()), 2), "of", TOTAL_H1_2026)
print("residual:", round(TOTAL_H1_2026 - sum(components.values()), 2))
```

Expected output:

```text
check                                computed     stated  result
----------------------------------------------------------------------
H1 total vs quarters                   7.4000     7.4000  ok
H1 year-on-year growth                 0.1562     0.1560  ok
mega-deal capital ($B)                 3.3300     3.3300  ok
implied CAGR 50->200 in 5y             0.3195     0.3195  ok
----------------------------------------------------------------------
claimed CAGR 25.0%; implied by 50->200 over 5y: 32.0%
reconciles: False
component sum: 4.2 of 7.4
residual: 3.2
```

Four readings matter. The first four checks pass, which means the report's internal arithmetic is consistent; the claim about collapse is not an arithmetic error. The fifth line shows a 25 percent CAGR claim that implies 32 percent growth from the stated endpoints, which is a different claim. The last two lines show that the components account for 4.2 of 7.4, so the chart is a partial view and the residual is the largest single category.

The script does not check whether the definitions are comparable across years. That is a judgement, and it is the check that matters most for a year-on-year claim.

To adapt the script, replace the constants with the figures from your source and add a check for every quantity the deck asserts. The value is not the arithmetic; it is the habit of asking whether the numbers reconcile before they are repeated.

## 8. Data, standards, and regulatory context

Market sources are governed by disclosure law, accounting standards, statistical standards, and advertising rules. The constraints differ by source type, and knowing which applies tells you how much weight a number can bear.

### 8.1 Disclosure and accounting

| Source | Rule | What it guarantees | What it does not |
|---|---|---|---|
| 10-K and 10-Q | SEC disclosure requirements | Specified financial statements, audited annually | That the strategy will work |
| 8-K | Material event disclosure | Timely notice of specified events | Completeness of context |
| GAAP financials | US accounting standards | Consistent recognition and measurement | Economic value |
| Non-GAAP measures | Company-defined adjustments | Reconciliation to GAAP is required | Comparability across companies |
| Proxy statement | SEC disclosure | Executive pay and governance | Operating performance |

A revenue figure in a filing is the most reliable company number available, and it is still shaped by recognition rules, segment definitions, and one-off items. A non-GAAP figure is a company's own adjustment, and the reconciliation is the part to read.

### 8.2 Government statistics

| Source | What it measures | Strength | Limit |
|---|---|---|---|
| Bureau of Labor Statistics | Employment and wages | Large samples, published method | Occupation definitions |
| CMS data | Medicare spending and utilisation | Claims-based, large | Medicare population only |
| CDC and NCHS | Health statistics | Population-based | Lag and survey design |
| Census | Population and business counts | Complete enumeration for some series | Definitional changes |
| AHRQ HCUP | Hospital utilisation | Multi-state discharge data | Purchase and training required |

A government statistic is not a market estimate. It measures a population or an activity, and turning it into a market requires assumptions that the statistic does not supply.

### 8.3 Advertising and claims

| Rule or guidance | Issuer | What it constrains |
|---|---|---|
| Endorsement guides | Federal Trade Commission | Testimonials and influencer claims |
| Health claims guidance | FTC and FDA | Substantiation of health benefit claims |
| Promotion rules for devices | FDA | Claims consistent with cleared intended use |
| Open Payments | CMS | Reporting of industry payments to clinicians |

A vendor testimonial in a market report is an advertisement as well as a data point, and the substantiation rules apply to the claim it makes.

### 8.4 What the standards do not establish

- Whether a market definition is the right one for a decision.
- Whether a survey sample represents the market.
- Whether a projection is achievable.
- Whether a company will grow.

The standards make numbers auditable and comparable within their rules. They do not make a market model true.

### 8.5 Where to find a company number

| Question | Filing section | What to read |
|---|---|---|
| Total revenue | Income statement | Revenue recognition note |
| Segment revenue | Segment note | Definition of segments |
| Growth | Comparative periods | Organic versus acquired |
| Recurring revenue | MD&A and metrics | Definition of recurring |
| Customer count | MD&A | Definition of a customer |
| Backlog | MD&A | Contract terms and cancellation |
| Risk factors | Item 1A | What the company says could go wrong |

The risk factors section is the most under-read part of a filing. It is written by lawyers to disclose uncertainty, and it often names the exact dependency that a market narrative ignores.

### 8.6 Common non-GAAP adjustments

| Adjustment | What it removes | Why it matters |
|---|---|---|
| Stock-based compensation | A real economic cost | Can make margins look better |
| Restructuring | One-off costs | Can recur every year |
| Acquisition costs | Transaction expenses | Can be part of strategy |
| Amortisation | Acquired intangible costs | Can be a real cost of the model |
| Litigation | Legal costs | Can be recurring |

A non-GAAP figure is not deceptive by definition, and it is not comparable across companies without reading the reconciliation. The reconciliation is usually a table below the income statement.

## 9. Industry practice and case evidence

**Digital health funding, H1 2026 (documented).** Rock Health reported $7.4 billion across 244 deals in the first half of 2026, against $6.4 billion across 245 deals in the first half of 2025. The report gives $4.2 billion for Q1 and $3.2 billion for Q2, a median deal size of $14 million, and 20 mega-deals of $100 million or more that accounted for 45 percent of the capital. The figures are the basis of the worked example.

**Digital health funding, Q3 2026 (documented).** Rock Health's Q3 update reported $3.7 billion across 114 deals, bringing the year-to-date total to $11.1 billion across 358 deals. The update is the source for the year-to-date context, and it illustrates why a half-year comparison and a year-to-date comparison can point in different directions.

**Occupational data (documented).** The Bureau of Labor Statistics reports a median annual wage of $120,230 for data scientists in May 2025, employment of 275,600 in 2025, and projected growth of 35 percent over 2025 to 2035. The figures are a government statistic with a published method, and they describe an occupation rather than a market for a product.

**AI device authorisations (documented).** The FDA states that it has authorised over 1,600 AI-enabled medical devices as of September 2026. The agency also notes that its list is not comprehensive, because devices are identified through AI-related terminology in summaries and classifications. The caveat is part of the source and should travel with the number.

**Hospital finance (documented, with a caution).** Kaufman Hall publishes a monthly National Hospital Flash Report on hospital operating margins and volumes. The reports are widely cited in industry commentary, and the underlying data are a self-selected panel of hospitals rather than a census. A margin figure from the report is a panel estimate, not a national accounting identity.

**Market research (documented, with a caution).** KLAS Research publishes vendor performance and adoption research based on interviews with healthcare organisations. The reports are useful for provider sentiment and are not a measure of clinical effectiveness.

### Documented facts and inferences

| Statement | Status |
|---|---|
| H1 2026 digital health funding was $7.4B across 244 deals | Documented by Rock Health |
| Q3 2026 added $3.7B, for $11.1B year to date | Documented by Rock Health |
| Data scientist median wage was $120,230 in May 2025 | Documented by BLS |
| FDA has authorised over 1,600 AI devices as of September 2026 | Documented by FDA, with the agency's caveat |
| Hospital margin reports are panel-based | Documented in the report methodology |
| Most decks quote the largest available number | Inference from practice |

### What practitioners do (inference)

- They use a market report for orientation and a filing for a company number.
- They keep the report edition and the definition with the figure.
- They show a range rather than a point estimate.
- They separate the sector, segment, and company claims.
- They check a number before it enters a board deck.

The practices are simple and they are not universal. The gap between the two is where most market claims fail in diligence.

### Two more sources worth knowing

**Hospital utilisation data (documented).** HCUP provides discharge-level data across participating states, with access through purchase or training. It is one of the few sources that supports bottom-up utilisation estimates, and its access cost is a barrier that many analyses ignore until late.

**Industry payments (documented).** CMS Open Payments publishes payments from manufacturers to clinicians and teaching hospitals. The data are useful for conflict-of-interest analysis and for understanding adoption influence, and they are not a measure of clinical benefit.

### What a good market memo contains

| Section | Content | Source type |
|---|---|---|
| Definition | What is in and out of the category | Analyst or own definition |
| Size | A range with the method | Two methods if possible |
| Growth | Endpoints and implied rate | Same series |
| Structure | Segments and concentration | Analyst and filings |
| Demand | Buyer behaviour and budget | Primary interviews |
| Competition | Named companies and shares | Filings and reports |
| Risks | What would make the number wrong | Analysis |

A memo with these sections can be challenged constructively. A memo with a number and a citation cannot.

The difference is not length. A two-page memo with the seven sections is more useful than a twenty-page deck with a number, because the reader can see which parts are evidence and which are judgement.

A useful test is whether the memo would survive a reviewer who disagreed with the conclusion. If the reviewer can attack the definition, the range, and the assumption in order, the memo is well built.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| TAM inflation | A very large number with no SAM | Top-down share with no evidence | Unrealistic plan | Compute SAM and SOM |
| Vendor survey bias | Respondents are customers | Self-selection | Optimistic demand | Check the sample frame |
| Category drift | Category name changed between years | Definition changed | False growth | Compare definitions |
| Period mismatch | Half-year compared with full year | Inconsistent periods | False trend | Align periods |
| Denominators missing | A percentage with no base | Incomplete reporting | Uninterpretable claim | Ask "of what" |
| Round number with no source | A suspiciously clean figure | Estimate presented as fact | Unverifiable claim | Trace to a method |
| Press release as evidence | Announcement cited as adoption | Event confused with outcome | Overstated traction | Find the contract |
| Filing segment mismatch | Company revenue used for a category | Entity and category conflated | Wrong comparison | Match the segment |
| Currency or unit error | Billions read as millions | Unit not stated | Order-of-magnitude error | State units |
| Base effect | High growth from a tiny base | Small prior period | Misleading trend | Show absolute values |
| Projection treated as forecast | A 2030 number in a 2026 plan | Projection confused with prediction | Plan built on a model | Label the projection |
| Interest undisclosed | No funding or client disclosure | Commercial interest | Biased framing | Identify the publisher |
| Single-source market | One report for every number | No triangulation | Fragile claim | Use two methods |
| Growth from acquisitions | Inorganic revenue growth | M&A counted as organic | Overstated momentum | Check organic growth |
| Survey without response rate | "Hundreds of leaders" | Unknown representativeness | Unsupported inference | Require the denominator |
| Category created for the report | A new acronym | Definition serves the narrative | Incomparable with other years | Use your own definition |
| Market map selection | Only friendly vendors shown | Curation by the publisher | False competitive picture | Check the inclusion criteria |
| Share without a market | "We have 10 percent" | Numerator from pipeline, denominator from a report | Inconsistent share | Recompute on one base |
| TAM used for pricing | Price justified by market size | Different questions conflated | Overpricing | Use willingness-to-pay evidence |
| Definition reused across products | A category for a different product | Convenience | Wrong market | Define per product |
| Growth from price increases | Volume flat, revenue up | Price effect | Misleading demand signal | Separate price and volume |
| Geographic extrapolation | US rate applied globally | Different systems | Wrong size | Localise the estimate |
| Channel confusion | Payer spend used as provider revenue | Different entities | Double counting | Map the money flow |

### A market claim audit in ten checks

- Is the definition stated.
- Is the geography stated.
- Is the year and basis stated.
- Is the method described.
- Is the denominator present.
- Do the numbers reconcile.
- Is the source's interest disclosed.
- Is there a second source.
- Is the claim sector, segment, or company.
- Does the claim match the decision.

A claim that fails three or more checks should not appear in a board deck without a stated uncertainty.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The definition matches the decision | Comparability depends on it | Category defined for the report | Compare with your own category |
| The geography matches the plan | Local decisions need local data | Global figure | Find a local estimate |
| The period matches the comparison | Trends depend on it | Different period lengths | Align periods |
| The method is disclosed | The number can be assessed | No methodology | Trace the number |
| The sample is representative | Survey inference depends on it | Customer list | Check the frame |
| The source is independent | Bias assessment needs it | Vendor or advocate | Check the publisher |
| The arithmetic reconciles | Errors are detectable | Components do not sum | Recompute |
| The projection is labelled | A plan needs a forecast | Projection presented as fact | Check the label |
| The segment is stable | Growth is comparable | Definition changed | Compare editions |
| A second source exists | Triangulation improves confidence | Single report | Search for another |
| The category is stable across editions | Growth is comparable | Definition changed | Compare editions |
| The report's data date is recent | The number is current | Data are a year old | Check the data date |
| The market map is complete | Competition is understood | Curation bias | Check the inclusion rule |
| The buyer budget exists | Demand is real | Intentions without budget | Interview buyers |
| The channel is understood | Revenue flow is correct | Payer and provider conflated | Map the money flow |
| The price is achievable | Revenue estimates are realistic | List price used | Check contract prices |
| The adoption curve is plausible | Growth is achievable | Analogies misapplied | Compare similar products |
| The regulatory path is clear | Timing is realistic | Clearance assumed | Check the pathway |

Two limits are structural. First, a market estimate is a model, and no amount of source checking turns it into a measurement. Second, the best sources describe the past; a decision about the future requires a judgement that the source cannot supply.

A third limit is access. Many market reports are paywalled, and the public summary may omit the method. When that happens, the honest record says that the method was not seen.

## 12. Exercises

**Exercise 1 — Decompose a claim (recall, 20 minutes).**
Take three market sentences from a deck and identify the definition, population, spend per unit, and method.
Deliverable: a three-row decomposition table.
Acceptance criteria: each factor is stated or marked unknown; at least one claim is shown to be unscoreable as written.
Time: 20 minutes.

**Exercise 2 — Source classification (application, 25 minutes).**
Classify five sources as filing, government statistic, analyst report, vendor survey, or press release, and state what each can support.
Deliverable: a five-row table with source type and permitted use.
Acceptance criteria: each classification is justified; at least one source is reclassified from evidence to context.
Time: 25 minutes.

**Exercise 3 — Arithmetic check (calculation, 30 minutes).**
Recompute the growth rate, shares, and sums for a market claim and identify any discrepancy.
Deliverable: a calculation sheet with the recomputed values and the discrepancies.
Acceptance criteria: the arithmetic is reproducible; discrepancies are explained; the corrected number is stated.
Time: 30 minutes.

**Exercise 4 — Denominator hunt (application, 25 minutes).**
Find the denominator behind three percentages in a report and state what it counts.
Deliverable: a three-row table with percentage, denominator, and source.
Acceptance criteria: each denominator is sourced; at least one percentage is shown to be ambiguous.
Time: 25 minutes.

**Exercise 5 — Failure-mode audit (critique, 35 minutes).**
Audit a market deck for the failures in section 10 and rank them by consequence.
Deliverable: a marked-up list of at least six findings with fixes.
Acceptance criteria: each finding names the failure and the fix; the ranking is justified; at least one finding concerns the definition.
Time: 35 minutes.

**Exercise 6 — Interest assessment (critique, 20 minutes).**
Identify the publisher, funder, and business model for three market sources and state the likely direction of bias.
Deliverable: a three-row table with interest and direction.
Acceptance criteria: each source's interest is documented; the direction is stated as a hypothesis; at least one source is assessed as neutral.
Time: 20 minutes.

**Exercise 7 — Rewrite the sentence (application, 25 minutes).**
Rewrite three market sentences with definition, geography, year, method, and a range.
Deliverable: three rewritten sentences with a source note.
Acceptance criteria: all five conditions are present; the range is justified; no sentence implies more precision than the source supports.
Time: 25 minutes.

**Exercise 8 — Primary research decision (design, 30 minutes).**
For a market question in your project, decide whether secondary sources suffice or primary research is needed.
Deliverable: a one-page memo with the question, the sources reviewed, the gap, and the recommendation.
Acceptance criteria: the memo names the decision; it states why the sources are insufficient; it proposes a feasible research design or a changed decision.
Time: 30 minutes.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Decompose | Four factors identified | Number only |
| 2 Classify | Permitted use stated | Source listed |
| 3 Arithmetic | Reconciles or explains | Repeats the number |
| 4 Denominator | Source for each base | Percentage accepted |
| 5 Audit | Six specific findings | General scepticism |
| 6 Interest | Direction stated | "Biased" with no reason |
| 7 Rewrite | Five conditions | Same sentence, new citation |
| 8 Research | Decision and gap | "More research needed" |

Two exercises produce artefacts that can be reused: the audit in Exercise 5 and the memo in Exercise 8. The others build the judgement that makes the artefacts reliable.

A learner who completes Exercise 3 will usually find at least one number in their own deck that does not reconcile. That finding is the most useful outcome of the topic.

## 13. Self-check answers

**1. What are the four factors of a market claim?**
Definition, population, spend per unit, and method. A change in any factor changes the number, and a claim without the factors cannot be checked.

**2. Why is a top-down estimate usually larger than a bottom-up estimate?**
Because it starts from an aggregate that includes customers and spending the product cannot serve, and applies a share that is often optimistic. A bottom-up estimate starts from units and prices and is constrained by what can actually be sold.

**3. What does a CAGR of 20 percent mean over five years?**
That the value grows by a factor of 1.2 to the fifth power, or about 2.49, over the period. It is a constant-rate summary and does not describe the path, which may include declines and recoveries.

**4. Why is a percentage without a denominator uninterpretable?**
Because the same percentage can describe a change from 2 to 4 or from 2 million to 4 million. The denominator determines the magnitude and the relevance.

**5. What is the difference between a filing and an analyst report?**
A filing is a required disclosure with audited financials and defined segments; an analyst report is a commercial publication with a category definition and a client base. The filing is stronger for company numbers and weaker for category context.

**6. Why can two funding series disagree and both be right?**
Because they define the category, the stages, the geography, and the inclusion criteria differently. The disagreement is a definitional difference, and the reader should report both rather than choose.

**7. What is a base effect?**
A high growth rate produced by a small prior period. It is common in emerging categories and should be shown with absolute values.

**8. When is primary research the right answer?**
When the decision depends on a number that no secondary source can support, such as the price a specific customer segment will pay. A small structured interview study is often more useful than a large market report.

**Rubric for Exercises 5 and 8.** A strong audit names specific failures with fixes and ranks them by consequence. A weak audit lists generic concerns. A strong research memo states the decision, the gap, and a feasible design. A weak memo calls for more research without a design.

**9. Why is a market report's category definition a source property rather than a detail?**
Because it determines which companies, products, and spending are counted. Two reports with different definitions produce different numbers, and a comparison across them is a comparison of definitions rather than of markets.

**10. What does it mean when a funding total rises while the deal count falls?**
It means capital is concentrating in larger rounds. The market may be healthy for companies that can raise large rounds and difficult for those that cannot, and the total alone hides the difference.

**11. Why is a company filing a better source for company revenue than an analyst report?**
Because the filing is a required disclosure with audited financials and defined segments, while the analyst report is a model. The filing is not perfect, and it is the stronger source for the specific number.

**12. When is a top-down estimate acceptable?**
When it is used for orientation, its assumptions are stated, and it is checked against a bottom-up estimate. It is not acceptable as the sole basis for a revenue plan.

**13. What is the difference between a projection and a forecast?**
A projection is a conditional statement about what would happen under stated assumptions; a forecast is a prediction of what will happen. Market reports often present projections and readers treat them as forecasts.

**14. Why is a segment definition more useful than a category definition for a product decision?**
Because the segment describes the customers a product can actually reach, while the category describes everyone the analyst chose to include. The product decision depends on the reachable segment.

**15. What does a market map hide?**
It hides the inclusion criteria and the selection of vendors. A map that shows ten companies and omits twenty is a curated view, and the curation is the argument.

**16. How do you check a growth claim against a filing?**
Find the company's reported revenue for the same period, compute the growth, and compare. If the company grows faster than the category, it is gaining share; if it grows slower, it is losing share, and either is a finding.

**17. What is the role of a range in a market claim?**
It communicates the uncertainty created by method and definition differences. A range from two methods is more informative than a point estimate from one.

## 14. Interview questions

**1. How do you evaluate a market size claim?**
Strong answer: decomposes definition, population, spend per unit, and method, checks the arithmetic, and asks who paid for the analysis. Weak answer: asks whether the source is reputable.

**2. What is the difference between TAM, SAM, and SOM?**
Strong answer: TAM is the total if every possible customer bought, SAM is the segment the product can serve, and SOM is the share realistically winnable. Weak answer: uses them interchangeably.

**3. How would you check a vendor's survey?**
Strong answer: sample frame, response rate, question wording, and funding; a customer survey describes customers. Weak answer: checks the sample size only.

**4. Why might a funding total be misleading?**
Strong answer: concentration in mega-deals, stage mix, inclusion criteria, and period effects can make a rising total consistent with a weak market for most companies. Weak answer: says funding is volatile.

**5. How do you compare a company's growth with a market's growth?**
Strong answer: matches the segment, the period, and the accounting basis, and separates organic from acquisitive growth. Weak answer: compares the headline numbers.

**6. When would you trust a government statistic over a market report?**
Strong answer: for population, employment, and spending baselines, because the method is published and the sample is defined. Weak answer: always, or never.

**7. How do you present a market number with appropriate uncertainty?**
Strong answer: a range with the method and the source, and a statement of what the number does not support. Weak answer: a point estimate with a citation.

**8. What does a market report's methodology section tell you?**
Strong answer: the definition, the data sources, the model, and the assumptions; its absence is the finding. Weak answer: says it is too long to read.

**9. How do you handle a claim that two credible sources dispute?**
Strong answer: compares definitions and periods, reports both, and explains the difference. Weak answer: picks one.

**10. What would make you commission primary research?**
Strong answer: a decision-critical number with no adequate secondary source, a specific customer segment, and a feasible design. Weak answer: whenever the report is paywalled.

**11. How do you separate a company's growth from its market's growth?**
Strong answer: compares the company's organic growth with the segment's growth, adjusts for acquisitions, and checks whether share is actually moving. Weak answer: compares both headline rates.

**12. What does a concentration metric add to a funding total?**
Strong answer: it shows whether the capital is broad or concentrated, which changes the interpretation for companies outside the top deals. Weak answer: treats the total as a measure of sector health.

**13. Why should a market claim name its source type?**
Strong answer: the reader calibrates differently for a filing, a government statistic, and a vendor estimate, and the source type is part of the claim's strength. Weak answer: cites a number without a source type.

**14. How would you value a market opportunity for a new product?**
Strong answer: bottom-up from eligible customers, expected price, adoption, and retention, with a range and a sensitivity on the weakest assumption. Weak answer: takes a share of a published TAM.

**15. What is the most common error in market decks?**
Strong answer: using a sector total to support a company-level plan, without a segment or bottom-up model. Weak answer: says the numbers are too optimistic.

**16. How would you check whether a market is growing or consolidating?**
Strong answer: compares dollar totals, deal counts, median size, and concentration, and checks whether growth is organic or acquisitive. Weak answer: looks at the total only.

**17. What does it mean if the top ten companies hold most of the market?**
Strong answer: the market is concentrated, so a new entrant faces a different sales problem from a fragmented market, and the TAM is not evenly accessible. Weak answer: treats the total as the opportunity.

**18. Why should the data date be reported with a market number?**
Because a recent publication can use data that are a year or more old, and the reader needs to know the period the number describes.

**19. What is the first thing you check about a survey-based claim?**
The sample frame and the response rate, because they determine who the result describes.

**20. How do you present a market number to an executive?**
Strong answer: one sentence with the definition, the source type, the year, and a range, plus what the number does not support. Weak answer: a chart with a large number.

## 15. Key terms

- **TAM** — total addressable market.
- **SAM** — serviceable addressable market.
- **SOM** — serviceable obtainable market.
- **Top-down estimate** — a share applied to an aggregate.
- **Bottom-up estimate** — units times price times adoption.
- **CAGR** — compound annual growth rate.
- **Year-on-year growth** — change versus the same period a year earlier.
- **Base effect** — growth inflated by a small prior period.
- **Nominal** — not adjusted for inflation.
- **Pro forma** — adjusted for specified events.
- **10-K** — annual SEC filing.
- **10-Q** — quarterly SEC filing.
- **8-K** — current report of a material event.
- **GAAP** — US accounting standards.
- **Non-GAAP** — company-adjusted financial measures.
- **Bookings** — contracted future revenue.
- **Burn rate** — net cash use per period.
- **Runway** — months of cash remaining.
- **Mega-deal** — a financing round above a threshold.
- **Median deal size** — the midpoint of deal values.
- **Penetration** — the proportion of a population using a product.
- **Average selling price** — revenue per unit.
- **Utilisation** — frequency of use.
- **Capitation** — fixed payment per patient per period.
- **Fee-for-service** — payment per service.
- **Bundled payment** — one payment for an episode.
- **Formulary** — the list of covered drugs.
- **Sample frame** — the population from which respondents are drawn.
- **Self-selection** — participation driven by interest.
- **Forward-looking statement** — a projection with legal safe harbour.
- **Panel data** — repeated observations from a defined set of units.
- **Triangulation** — checking a number against independent sources.
- **Segment** — a defined part of a market or company.
- **Category** — the analyst's grouping of products.
- **Addressable** — reachable by the product and the sales model.
- **Obtainable** — realistically winnable in a period.
- **Price per unit** — the revenue per sale.
- **Willingness to pay** — the maximum a buyer will pay.
- **Budget owner** — the person who controls the spend.
- **Procurement cycle** — the time from interest to contract.
- **Sales pipeline** — qualified opportunities with values.
- **Conversion rate** — the proportion of opportunities that close.
- **Churn** — the rate at which customers leave.
- **Net revenue retention** — revenue from existing customers over time.
- **Gross margin** — revenue minus the cost of delivery.
- **Contribution margin** — revenue minus variable costs.
- **Customer acquisition cost** — the cost of winning a customer.
- **Lifetime value** — the expected revenue from a customer.
- **Addressable spend** — the money available in a segment.
- **Budget cycle** — the annual planning cycle that governs purchases.

## 16. Sources

All URLs were checked on 26 October 2026. Figures are those published for the period named.

**Rock Health.** "H1 2026 funding and market overview: Durable roots, shifting routes." July 2026. https://rockhealth.com/insights/h1-2026-funding-and-market-overview-durable-roots-shifting-routes/
Good for: US digital health funding totals, deal counts, median size, and mega-deal concentration. Not for: company-level revenue or clinical outcomes.

**Rock Health.** "Q3 2026 market update." October 2026. https://rockhealth.com/insights/q3-2026-market-update-crystallizing-fifteen-years-of-digital-health/
Good for: year-to-date totals and quarterly context. Not for: a full-year forecast.

**U.S. Bureau of Labor Statistics.** *Occupational Outlook Handbook: Data Scientists.* https://www.bls.gov/ooh/math/data-scientists.htm
Good for: median wage, employment, and projected growth with a published method. Not for: a market for a specific product.

**U.S. Food and Drug Administration.** *Artificial Intelligence-Enabled Medical Devices.* https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
Good for: the agency's statement that it has authorised over 1,600 AI-enabled devices as of September 2026, with its caveat. Not for: a comprehensive census.

**U.S. Securities and Exchange Commission.** EDGAR full-text search. https://www.sec.gov/edgar/search/
Good for: filings, audited financials, and segment data. Not for: private companies.

**Kaufman Hall.** *National Hospital Flash Report.* https://www.kaufmanhall.com/
Good for: hospital margin and volume trends from a panel. Not for: a national census or a causal claim.

**KLAS Research.** Healthcare IT research. https://klasresearch.com/
Good for: provider sentiment and vendor performance research. Not for: clinical effectiveness.

**Centers for Medicare and Medicaid Services.** Open Payments. https://openpaymentsdata.cms.gov/
Good for: industry payments to clinicians. Not for: clinical evidence.

**Federal Trade Commission.** Endorsement guides. https://www.ftc.gov/business-guidance/resources/ftcs-endorsement-guides
Good for: the rules on testimonials and disclosures. Not for: the truth of the underlying claim.

**Centers for Medicare and Medicaid Services.** Coverage with Evidence Development. https://www.cms.gov/medicare/coverage/coverage-with-evidence-development
Good for: how payment can be tied to evidence collection. Not for: a market size.

**Agency for Healthcare Research and Quality.** Healthcare Cost and Utilization Project. https://hcup-us.ahrq.gov/
Good for: hospital utilisation data. Not for: free public access; purchase or training is required.

**National Center for Health Statistics.** Health, United States and related reports. https://www.cdc.gov/nchs/
Good for: population health statistics and denominators. Not for: a market forecast.

**Kaufman Hall.** Hospital finance data and methodology. https://www.kaufmanhall.com/
Good for: understanding the panel and the reported metrics. Not for: a national census.

**KLAS Research.** Methodology and research areas. https://klasresearch.com/
Good for: understanding how provider sentiment data are collected. Not for: clinical evidence.

**Rock Health.** Methodology and funding data definitions. https://rockhealth.com/
Good for: the definition of the funding series and its inclusion criteria. Not for: company-level financials.

**U.S. Securities and Exchange Commission.** How to read a 10-K. https://www.sec.gov/
Good for: the structure of filings and the sections that carry financial data. Not for: interpretation of a specific company's prospects.

**Federal Reserve Economic Data (FRED).** Economic time series. https://fred.stlouisfed.org/
Good for: macroeconomic series and deflators for nominal-to-real comparisons. Not for: sector-specific market sizes.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_datasets-and-benchmarks.md`](06_datasets-and-benchmarks.md)
- Next topic: [`08_communities-and-continuing-education.md`](08_communities-and-continuing-education.md)
- Regulation and policy: [`03_regulation-and-policy.md`](03_regulation-and-policy.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 17, product economics: [`../17_product_economics/`](../17_product_economics/)
- Module 20, career: [`../20_career/`](../20_career/)
- Module 19, capstone: [`../19_capstone/`](../19_capstone/)
- Module 21, datasets: [`../21_datasets/`](../21_datasets/)
- Module 18, quality: [`../18_quality/`](../18_quality/)
- Module 22, regulation: [`03_regulation-and-policy.md`](03_regulation-and-policy.md)
- Survey industry sources: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md) section 7

The next topic moves from market evidence to professional communities, where the source problem is the difference between a conference abstract and a peer-reviewed paper.
| Method | Starting point | Strength | Weakness |
|---|---|---|---|
| Top-down | National spend or category total | Fast, uses existing data | Assumes a share without evidence |
| Bottom-up | Units, prices, adoption | Transparent assumptions | Requires data that are often unavailable |
| Survey | Buyer intentions | Captures demand signals | Sample and self-selection bias |
| Filing-based | Reported revenue | Audited, comparable | Only covers public companies |
| Analyst model | Combined sources | Integrates evidence | Method often undisclosed |
| Comparable transactions | Deal values | Grounded in prices paid | Few comparables, strategic premia |

A diligence team will usually ask for two methods and compare. If the top-down and bottom-up estimates differ by a factor of three, the assumptions differ and the range is the honest answer.

### A branch worked in miniature: TAM versus SAM

A deck claims a $40 billion total addressable market for an AI triage product. The number comes from a report on global spending on clinical decision support.

The product is sold to US hospitals with an electronic health record that supports the integration, for emergency departments, at a price per bed. The serviceable addressable market is the number of eligible beds times the price, which is a much smaller number and is computable from public data.

The branch is not a debate about whether the $40 billion is correct. It is a question about which number the decision needs. For an investor, the TAM describes the ceiling; for a pricing decision, the SAM describes the opportunity; for a sales plan, the SOM describes the target.

A deck that shows only the TAM is not lying. It is answering a different question from the one the audience is asking, and the correction is to add the other two numbers rather than to dispute the first.

### Filing versus report

| Attribute | SEC filing | Analyst report |
|---|---|---|
| Legal status | Required disclosure | Commercial publication |
| Audit | Audited financials | Not audited |
| Definition | The company's segments | The analyst's category |
| Availability | Public | Often paywalled |
| Bias | Management framing | Client and publisher interests |
| Use | Company size and growth | Category orientation |

A filing and a report answer different questions. A filing tells you what a company reported; a report tells you what an analyst thinks the category is worth. Neither tells you what your product can sell.
