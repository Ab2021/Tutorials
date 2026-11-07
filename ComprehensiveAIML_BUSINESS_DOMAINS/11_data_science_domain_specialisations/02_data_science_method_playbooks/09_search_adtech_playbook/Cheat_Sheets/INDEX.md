# Search and AdTech Playbook — Cheat Sheets

## 1. One-paragraph overview

This collection supports engineers and analysts deciding how to retrieve and rank search candidates and reason about advertising eligibility. Readers should know basic logarithms, sets, and sorting. Readiness means reproducing a score and tracing a failure without confusing retrieval with ranking or auction clearing. The collection is incomplete; the only authored sheet below is a draft, not a reviewed publication.

## 2. Topic ladder

| Level | File / topic | Testable outcome | Prerequisite | Status |
|---|---|---|---|---|
| Foundations | Search fundamentals — planned | Explain postings and query eligibility | Sets and sorting | Planned |
| Core | [02 BM25 and classical retrieval](02_BM25_and_Classical_Retrieval.md) | Compute BM25 and diagnose retrieval failures | Basic logarithms, sets, sorting | Draft; not independently reviewed |
| Advanced | Search debugging flow — planned | Isolate retrieval from ranking failures | Retrieval fundamentals | Planned |
| Advanced | Ads auction guide — planned | Distinguish eligibility and clearing | Auction fundamentals | Planned |
| Advanced | Relevance judgment rubric — planned | Define and audit relevance labels | Search evaluation | Planned |
| Final | Common Issues — planned | Diagnose at least 25 distinct issues | Relevant completed sheets | Not authored |

## 3. Quick topic lookup

| Question | Location | Warning |
|---|---|---|
| How are TF-IDF and BM25 defined? | [Sheet 02, Section 4](02_BM25_and_Classical_Retrieval.md#4-core-formulas-with-full-derivation) | Definitions and implementations vary |
| How are postings and scores calculated? | [Sheet 02, Section 5](02_BM25_and_Classical_Retrieval.md#5-worked-numerical-example) | Example is synthetic, not measured |
| Why is a document missing? | [Sheet 02, Section 8](02_BM25_and_Classical_Retrieval.md#8-failure-modes--edge-cases) | Token mismatch, filtering, and freshness can look alike |
| How do candidate recall and latency trade off? | [Sheet 02, Section 9](02_BM25_and_Classical_Retrieval.md#9-complexity--cost-analysis) | Bounds are not latency benchmarks |

## 4. Reading paths — three options

- Beginner: review the prerequisites below, then read Sections 1–6 of sheet 02; the dedicated foundations sheet is planned.
- Practitioner: read sheet 02 in order, then use its failures and reference card; debugging and judgment sheets are planned extensions.
- Advanced: read all of sheet 02, reproduce its example, and design a controlled comparison; auction and hybrid retrieval handoffs remain planned.

## 5. Prerequisite skills

Know set union/intersection, logarithms, ratios, sorting, and basic precision/recall. The [Stanford IR book](https://nlp.stanford.edu/IR-book/) provides a public introduction to indexes and evaluation. Familiarity with field analyzers, source-to-index updates, and access-control predicates supports the production sections.

## 6. What this collection does NOT cover

The current draft does not implement learning to rank, neural retrieval, auction clearing, bidding, or a complete relevance-labeling program. These are planned handoffs, not available completed references. It does not assert any actual company's results or measured production benchmark.

## 7. Master symbol glossary

| Symbol | Meaning and units | Type/domain | First appearance |
|---|---|---|---|
| N | Documents in scoring population, count | Positive integer | [Sheet 02 §3](02_BM25_and_Classical_Retrieval.md#3-foundational-definitions) |
| df(t) | Documents containing term t, count | Integer 0 through N | [Sheet 02 §3](02_BM25_and_Classical_Retrieval.md#3-foundational-definitions) |
| f(t,d) | Analyzed term occurrences in field d, count | Nonnegative integer | [Sheet 02 §3](02_BM25_and_Classical_Retrieval.md#3-foundational-definitions) |
| L, A | Document field length and average length, tokens | Nonnegative scalar; A positive in BM25 | [Sheet 02 §3](02_BM25_and_Classical_Retrieval.md#3-foundational-definitions) |
| k1, b | BM25 saturation and length parameters, dimensionless | k1 positive; b in [0,1] | [Sheet 02 §4](02_BM25_and_Classical_Retrieval.md#4-core-formulas-with-full-derivation) |
| K | Top-result count in complexity discussion; also length-adjusted BM25 factor in Section 4 | Context-dependent scalar | [Sheet 02 §4](02_BM25_and_Classical_Retrieval.md#4-core-formulas-with-full-derivation) |

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Avoid condition |
|---|---|---|---|
| [Sheet 02 card](02_BM25_and_Classical_Retrieval.md#12-quick-reference-card) | Sum positive-IDF weighted saturated term responses | Consistent analysis and valid corpus statistics | Access controls cannot be enforced |

## 9. Common Issues locator

The final Common Issues artifact is planned, with zero completed issues and no stable issue anchors. Sheet 02 Section 8 contains ten hypothetical diagnostic scenarios; these are not a substitute for the required collection artifact.

## 10. Empirical sources, versions and collection maintenance

Sheet 02 cites Lucene 9.12.1 for its documented BM25 IDF and defaults, Robertson and Zaragoza (2009) for model-family background, Unicode UAX #15 for normalization, and official scikit-learn and Elasticsearch documentation for implementation terminology. Rolling documentation must be checked against the deployed version. Synthetic ranking and capacity examples are illustrative, not measured. Sheet 02 is a draft and has not been independently reviewed. Revisit it when analyzers, corpus statistics, scoring implementation, or security policies change. No benchmark has been measured.

## 11. Collection acceptance checklist

- [x] Current draft identifies scope, sources, and evidence limitations.
- [x] Current draft includes 12 numbered sections, worked arithmetic, failure diagnostics, interview answers, and final card.
- [x] Sheet 02 has at least 1,000 substantive nonblank, nonheading lines.
- [ ] All planned sheets exist and have been independently reviewed.
- [ ] Final Common Issues artifact contains at least 25 complete distinct issues.
- [ ] All collection links, primary-source claims, formulas, and implementation versions are independently reviewed.
- [ ] Collection is ready for publication.
