---
collection: search_adtech_playbook
file_id: '02'
topic: BM25 and classical retrieval
level: 2
status: draft
version: '1.0'
created: '2026-11-07'
reviewed: null
review_trigger: Analyzer, scoring implementation, corpus, or access-policy change
notation_style: plain-text
prerequisites: Basic logarithms, sets, probability, and sorting
pattern_tags: [retrieval, inverted-index, relevance]
source_status: OPEN
owner_role: Search relevance engineering
source_index: INDEX.md
---
# BM25 and Classical Retrieval

## 1. File Header

**Purpose:** Compute and diagnose a lexical candidate generator before downstream ranking or auction decisions.
**Audience:** Search engineers, data scientists, relevance analysts, and interview candidates.
**Prerequisites:** Basic logarithms, sets, probability, sorting, and the meaning of precision and recall.
**Key outcome:** Given an analyzer, a corpus snapshot, and an explicit scoring variant, construct postings, calculate scores, and explain the resulting order.
**Not covered:** Neural embedding training, full learning-to-rank implementation, and auction clearing mechanics.
**Evidence:** DERIVED identifies conclusions from stated assumptions; ILLUSTRATIVE examples and proposed systems are fictional and not measured.
**Version / reviewed:** 1.0 / not independently reviewed.
**Retrieval cue:** Start at Section 8 for a relevance regression and Section 5 for a hand calculation.
**Provenance:** `directory.txt` lists `Search_AdTech_Cheat_Sheets/01_Search_Fundamentals_IR.md` and `02_BM25_TF_IDF_Classical_Retrieval.md`; only their titles are available, not their contents.
The catalog's `Search_AdTech/01_Ecommerce_Search_Ranking_LTR_Neural_Retrieval.md` and `02_Real_Time_Bidding_Ad_Auction_System.md` titles motivate the commerce and ads boundaries; they are not evidence about a real system.
A retrieval score is a ranking feature, not a probability of relevance or a bid price.
The teaching examples use natural logarithms and fully specified synthetic corpus statistics.
Actual product outcomes, latency, and economics are NOT MEASURED.
Takeaway: Choose and verify the candidate-generation contract before interpreting its scores.

## 2. Concept Map / Mental Model

```text
Documents -> index analyzer -> term dictionary -> postings + field statistics
Query -> query analyzer -> terms -> Boolean/filter eligibility -> score postings
                                                    |                  |
                                             authorization       top-K candidates
                                                                       |
                                                    ranker / auction eligibility
                                                                       |
                                                 judgments + operational feedback
```
The arrows describe an illustrative architecture rather than a measured deployment.
The index is built from a particular corpus snapshot and analyzer version.
An index-time analyzer turns field text into tokens and usually records term positions when phrase queries are supported.
A query-time analyzer transforms the request into compatible tokens and may apply a different, explicitly tested search analyzer.
The vocabulary tells the engine which terms exist.
A postings list associates a term with documents containing it and may store frequencies and positions.
An eligibility gate restricts candidates by access policy, availability, locale, or other hard conditions.
The scorer orders eligible lexical matches using a named, versioned formula.
A top-K cutoff bounds what later ranking stages can see.
A ranker cannot rescue a relevant item absent from its candidate set without another retrieval path.
In an advertising application, matching is only one gate; campaign eligibility, budgets, pacing, policy, and auction logic require separate authoritative rules.
Search relevance and advertising revenue are distinct objectives, and score units do not imply a valid exchange rate between them.
A feedback loop should distinguish poor candidate recall from poor ordering within a good candidate set.

### Illustrative index lifecycle

Start from a source record with a stable entity identifier and monotonically ordered source version when the source supports one.
Choose whether the indexed unit is a product, a seller offer, or a product variant before computing document frequencies.
Define ownership of fields such as title, stock status, and access policy so independent feeds cannot silently overwrite each other.
Capture a consistent source snapshot or a documented checkpoint from which changes can be replayed.
Normalize each field with its assigned analyzer and preserve the analyzer version alongside the index generation.
Generate term-document occurrences and aggregate their frequencies for each indexed field.
Store positions only where phrase or proximity semantics justify their storage and decoding cost.
Build the term dictionary and sorted postings with a deterministic document-ID ordering.
Record field-length and corpus statistics using the exact population semantics of the selected engine.
Publish the new generation only after mandatory integrity, eligibility, and compatibility checks pass.
Route a controlled amount of shadow traffic to the generation without exposing its results to users.
Compare candidate differences for identical inputs and investigate differences before changing the active alias.
Keep a known-good generation available for rollback subject to deletion and retention rules.
Do not roll back to a generation that would restore access to records that must remain deleted.
For a source update, reject or reconcile stale versions rather than blindly overwriting a newer record.
When source ordering is not guaranteed, document the conflict-resolution rule and its limitations.
For a deletion, propagate a versioned tombstone to every serving copy and invalidate dependent caches.
Track event ingestion, index acknowledgment, and search visibility as distinct milestones.
Only advance a recovery checkpoint when the corresponding durability semantics are satisfied.
Replay must be idempotent or deduplicated by a stable event identity and source version.
Compaction and merging can reclaim obsolete data but do not replace an access-control check.
An analyzer change generally requires reprocessing existing text if the stored terms must change.
A query-only synonym change can be deployed independently only when the index and query representations remain compatible.
A schema change that alters the retrieval unit requires new relevance judgments or careful mapping to the new unit.
Statistics may shift as data and segments change even when the scoring configuration stays fixed.
Keep source offsets, index generation, schema version, and analyzer version in diagnostic traces without logging sensitive raw queries by default.
The preceding sequence is a proposed design, not a claim about a particular vendor's transaction semantics.

### Illustrative query lifecycle

Authenticate the caller and derive authorization predicates from trusted server-side identity.
Validate request size and complexity before allowing expensive expansions or large result windows.
Resolve locale, field selection, and the active index generation for the request.
Apply the configured query analyzer and inspect whether it emitted any terms.
Represent multi-token synonyms with the positions or graph structure needed by the chosen query semantics.
Separate mandatory constraints from score-only boosts in the query plan.
Apply access, availability, and policy constraints before returning results, snippets, or aggregations.
Look up term dictionaries and access the corresponding postings for each query clause.
Choose conjunction, disjunction, phrase, or minimum-should-match behavior explicitly.
Traverse candidate postings and calculate scores only within the selected execution semantics.
Use stable tie-breaking so identical inputs and an unchanged snapshot yield a reproducible order.
Apply the top-K cutoff at the documented stage, because early truncation may hide candidates from later rankers.
Pass candidates and their versioned features to a downstream ranker only if it is part of the specified system.
For advertising, apply campaign-specific eligibility and auction rules independently of organic lexical scores.
Return only authorized fields and avoid exposing raw scoring internals if they leak protected data.
Collect privacy-preserving counters for analyzed-empty queries, eligible candidates, latency, and result quality.
Sample explanations under an access-controlled diagnostic policy rather than storing unrestricted query text indefinitely.
When a shard fails, follow the explicit partial-result policy; do not silently treat partial retrieval as complete.
When an authorization dependency fails, use the organization's documented fail-closed policy.
The complete query path, including network and queueing, determines user-visible latency.

Takeaway: Separate tokenization, eligibility, scoring, and downstream ordering when explaining any failure.

## 3. Foundational Definitions

A **document** is the indexed retrieval unit, such as one product or one product variant; the choice changes counts and duplicates.
A **field** is a named text region such as title or description whose analyzer, length, and scoring may differ.
A **term** is a token emitted by the analyzer, not necessarily a whitespace-delimited word.
A **posting** records a document identifier for a term; frequency and position information are optional according to index configuration.
A **dictionary** maps a term to metadata or a pointer into its postings; it is not the original document store.
A **document frequency**, df(t), counts documents containing term t at least once, not the number of occurrences.
A **term frequency**, f(t,d), counts occurrences of t in document d's scored field after the stated analysis convention.
N is the number of indexed documents in the scoring population, and it must use the same population convention as df.
Length L(d) is the number of counted field tokens in document d under the selected scoring implementation.
Average length A is the sum of field lengths divided by the number of fields included by the selected implementation.
An absent field, an empty field, and a field with tokens removed by analysis may be counted differently by an engine.
A vocabulary term's **collection frequency** counts all occurrences across the collection and is not df.
A **query term** is an analyzed query token; repeated tokens may be deduplicated or weighted depending on the query builder.
A **candidate** is an eligible document presented to the next stage; its presence does not establish relevance.
A **posting position** is a token offset used to test order and proximity, not a byte offset in the original text.
A **precision at K** is the fraction of the first K results judged relevant under a specified binary judgment protocol.
A **recall at K** is the fraction of all relevant documents in the evaluated universe appearing in the first K results.
A **judgment pool** is the set of documents assessed by raters; missing judgments are not automatically reliable negatives.
An **analyzer** is a versioned sequence of character processing, tokenization, and token filters.
A **stopword** is a token removed by a configured filter, not a universally useless word.
A **stem** is a conflated token form, not necessarily a valid dictionary word or the same as a lemma.
A **synonym graph** can represent alternate token sequences; flattening it incorrectly can alter phrase semantics.
A **segment** is a component of an index whose lifecycle and statistics depend on the search engine.
A **tombstone** marks a deleted or superseded record until storage cleanup; it is not itself proof of immediate invisibility.
A **snapshot** identifies a reproducible corpus, configuration, and index generation for a comparison.
A **tie-breaker** supplies a deterministic order for equal scores, typically by a stable identifier.
A **calibrated probability** has an empirical probability interpretation; a BM25 score does not acquire one merely by being positive.
A **shard** is an index partition; statistics may be local or aggregated depending on the product and query mode.
A **tenant** is an isolation boundary; scoring statistics and query caches can still be shared unless explicitly designed otherwise.

### Analysis contract and multilingual retrieval

The following examples are ILLUSTRATIVE tokens, not claims about a particular analyzer's output.
The original source string and the emitted token stream are different artifacts with different security and relevance implications.
Character normalization happens before tokenization in many analyzers, but its actual order is part of the implementation contract.
Unicode permits visually similar text to have different underlying sequences, so inspect code points when exact matches fail.
Canonical normalization and compatibility normalization do not make identical promises; select one after assessing which distinctions are meaningful.
A compatibility fold may make some visually related characters match while erasing distinctions needed in a product identifier.
Unicode normalization forms and their definitions are specified in Unicode Standard Annex #15 [S6].
Case folding is language-sensitive in important cases, so test the deployment's specific implementation instead of assuming English lowercase rules.
Case-sensitive identifiers can require a separate exact field even when titles use case-insensitive analysis.
Accents can distinguish words in one setting while being optional in another; decide whether folding serves the query population.
Retain an exact field when accent folding would collapse legally or commercially distinct names.
An accent-insensitive field can supplement, rather than replace, an exact field when the desired behavior is explicitly tested.
Character substitutions may change offset mappings, so verify highlighting against the original text.
An analyzer that discards punctuation can join, split, or erase important catalog identifiers.
The product strings `AB-12` and `AB12` are not automatically interchangeable unless the business defines them that way.
An identifier-specific normalizer can preserve a canonical exact value alongside analyzed natural-language fields.
The sequence `C++` illustrates why a punctuation-stripping rule must be tested against programming and product vocabulary.
A slash in a size or model code may carry information that a general word tokenizer discards.
Hyphen handling affects both recall and phrase behavior because splitting introduces additional token positions.
Apostrophe handling matters for contractions, names, possessives, and languages whose spelling conventions differ.
Tokenization of scripts without whitespace word boundaries requires an algorithm suited to those scripts and the retrieval task.
Whitespace splitting alone is not a language-independent word segmentation strategy.
Character n-grams can supply partial matches in some languages but increase postings and may introduce accidental overlaps.
An n-gram configuration needs minimum and maximum lengths, boundary rules, and a clear strategy for short queries.
Short n-grams often have high document frequency and can broaden candidates substantially.
An n-gram match is evidence of shared characters, not proof that two words have the same meaning.
For a constructed trigram example, `shoes` yields `sho`, `hoe`, and `oes` without boundary padding.
The same example under a bigram analyzer yields `sh`, `ho`, `oe`, and `es`, a different index.
Character n-grams require a consistent definition of character, code point, or grapheme unit to avoid corrupting multicode-point text.
Unicode grapheme boundaries are defined separately from byte offsets and require their own versioned processing rules.
Language identification may be uncertain for a short query or a mixed-language product title.
A mistaken language label can route a query to a stemmer that conflates unrelated words.
Using a language-neutral exact field as an additional path can preserve important identifiers when language detection is uncertain.
Evaluate any fallback against real query distributions; a broad fallback can also increase irrelevant candidates.
Stemming maps surface forms into shared terms using a configured rule set.
Lemmatization may use linguistic analysis and can be more context-dependent than simple stemming.
Neither stemming nor lemmatization guarantees that all semantically related expressions share a token.
Aggressive stemming can cause false positives when distinct product names collapse to the same stem.
Insufficient stemming can cause false negatives when common inflected forms remain distinct.
An inflection regression set should include both intended conflations and pairs that must stay separate.
Synonyms can be expanded at indexing time, at query time, or through a separate query rewrite.
Index-time expansion changes stored terms, frequencies, and potentially field length depending on how the engine handles overlaps.
Query-time expansion can increase the number of clauses and the cost of traversing their postings.
Changing an index-time synonym rule may require reindexing historical documents to obtain consistent behavior.
Changing a query-time synonym rule can alter results immediately but may introduce expansion cycles or explosive clause counts.
Treat synonym rules as versioned search configuration with an owner, review process, and rollback path.
Directional synonyms encode an intended rewrite rather than an equivalence relation.
For example, mapping a broad category to a narrower brand is not generally safe in both directions.
Domain synonyms should be scoped by locale or product context when the same string has several senses.
A medical abbreviation and a consumer-product abbreviation can share letters while requiring different interpretations.
Multiword expansions require position-aware handling if exact phrases and proximity queries matter.
In a token graph, alternate paths can represent a multiword expression without pretending that every alternative has the same number of positions.
Testing only the list of emitted strings can miss position and offset errors that affect phrase matching.
An acronym may need both an exact identifier field and a natural-language expansion.
Synonym evaluation should include negative examples where the proposed replacement must not be applied.
Stopword lists should be versioned by language, field, and intended query behavior.
Removing `not` from a safety-related phrase can reverse an important distinction.
Removing an article from a named work can change its identity even if the article is common.
An all-stopword query needs an explicit fallback rather than an accidental match-all request.
If stopwords are removed from an indexed phrase, token position increments determine whether the remaining terms appear adjacent.
Position gaps preserve information about removed tokens in engines that support and configure them that way.
The same surface text can index differently when a stopword list or token filter version changes.
A phrase query's slop or proximity parameter must be documented as engine-specific behavior rather than universal English distance.
Field-specific analyzers let a title use linguistic processing while a SKU uses exact normalization.
Combining these fields requires an explicit scoring or candidate-union rule rather than assuming their scores are directly comparable.
An index-time analyzer can favor recall while a stricter search analyzer preserves query intent, but compatibility must be demonstrated.
Asymmetric analysis is a deliberate design choice, not a license to omit analyzer regression tests.
Maintain golden examples for accented words, inflections, punctuation, scripts, mixed languages, and identifier strings.
Each golden example should record input, analyzer version, emitted terms, positions, and expected candidate behavior.
When text contains personally identifying information, use synthetic regression strings or carefully governed test data.
A tokenizer upgrade needs a compatibility assessment across historical indexes and current query workers.
The observed zero-result rate should be broken down by locale and query class before selecting a multilingual fix.
Relevance labels should include native-language or domain-qualified review when literal token overlap is an unreliable proxy.
Do not infer a language's search quality from English-only examples.
No multilingual quality gain, latency effect, or coverage rate is measured in this sheet.

### Candidate, score, and evidence boundaries

Exact lexical matching is a relationship between analyzed representations, not necessarily between the original strings.
An exact term match can still be irrelevant when a word has several meanings.
A document with no exact match can still be relevant through a synonym, translation, or implicit user intent.
The candidate set is defined before downstream ranking, so its recall bounds the ranker's achievable recall on that same universe.
This bound does not imply that increasing candidate count always improves final ranked quality.
A fixed top-K candidate budget can cause one retrieval route to crowd out another unless merge policy is designed explicitly.
The same document can enter through multiple routes and should be deduplicated using a stable entity key when appropriate.
Deduplication by product family may be wrong if the user needs distinct available variants.
Deduplication by offer may be wrong if repeated sellers overwhelm a product-focused result page.
The chosen entity grain determines what counts as a duplicate and what qualifies as a relevant result.
Score explanation is useful only when it identifies the exact field, term, statistics, parameters, and generation used.
A diagnostic score cannot establish that the user saw the result or that it was relevant.
A high score under one query is not necessarily comparable with a high score under another query.
Different terms have different IDF values, and different queries can contain different numbers of scoring clauses.
Scores can also change with corpus updates, analyzer changes, boosts, shard statistics, and field-length conventions.
Thresholding raw scores across queries therefore requires separate validation and monitoring.
The index may share statistics across tenants even when authorization filters prevent cross-tenant documents from being returned.
Shared statistics can affect rank and may require a threat assessment if score explanations or timing expose sensitive changes.
Eligibility filters must be checked independently from score computations.
Result counts, facets, autocomplete, snippets, and spelling suggestions all have their own exposure surfaces.
The absence of a document in the first page does not prove that it was absent from the candidate set.
The presence of a candidate does not prove that the downstream ranker received it if intermediate truncation occurred.
Trace identifiers can join stages without requiring the full query or document content in every log.
Judgment labels need a clearly stated query intent and document entity; relevance is not a property of a document in isolation.
A label can become stale when product availability, content, location, or user intent changes.
Clicks are observations of both relevance and exposure, not unbiased judgments by default.
Purchases additionally depend on price, inventory, delivery terms, and many factors outside lexical matching.
An index-level recall measure should not be advertised as an end-to-end purchase lift.
Document-level permission checks and business eligibility checks can change independently and require separate diagnostics.
For an ads workflow, an eligible creative is not necessarily a valid winner; auction mechanics remain outside this sheet.
Takeaway: Define the retrieval unit, field, and population before calculating frequency or length.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

**Validity:** Use the following explicitly defined teaching TF-IDF variant with a nonempty corpus and positive df for each scored term.
This is a convention, not a universal definition of TF-IDF.
For one term, a raw count f represents within-document repetition.
Weight a count by the logarithm of inverse document frequency to reduce the contribution of widely occurring terms.
**FORMULA T1:** weight(t,d) = f(t,d) * ln(N / df(t)).
**WHERE:** N and df are document counts; f is a nonnegative token count; ln is the natural logarithm.
**CONSTRAINT:** 1 <= df <= N and N > 0; a term absent from the dictionary has no postings to score.
**NUMERICAL EXAMPLE:** N=4, df=2, f=2 gives 2*ln(2) = approximately 1.386294.
**INTERPRETATION:** Under this particular formula, two occurrences of a term found in half the documents contribute 1.386294 score units.
A term found in every document receives zero weight in T1; this is not true for all smoothed conventions.
The logarithm makes multiplying the rarity ratio by a fixed factor add a fixed amount of IDF rather than multiplying it.
A query weight and document-vector normalization are additional choices, not implicit components of T1.
The scikit-learn documentation specifies its own smoothed IDF and normalization conventions; do not substitute T1 while claiming implementation equivalence [S2].

### 4.2 General Case

**Validity:** The following single-field, distinct-query-term BM25 teaching variant assumes N>0, A>0, k1>0, 0<=b<=1, nonnegative integer f, and positive L for documents with a match.
The positive IDF below matches the formula documented for Lucene 9.12.1 BM25Similarity; score scaling and encoded field lengths can differ in an actual implementation [S1].
The chosen expression has the conventional (k1+1) numerator, which is a positive query-independent scaling for fixed k1 and is not asserted to reproduce Lucene's exact score magnitude.
Derivation sketch: start with a probabilistic relevance weighting motivation, approximate missing relevance statistics by corpus rarity, and use a saturating term-frequency response adjusted by relative document length [S3].
The resulting formula is an engineered ranking model, not an exact posterior relevance probability.
**FORMULA B1:** idf+(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).
**FORMULA B2:** c(t,d) = f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*L(d)/A)).
**FORMULA B3:** score(q,d) = sum over distinct analyzed query terms t of idf+(t)*c(t,d).
**WHERE:** k1 controls frequency saturation, b controls length sensitivity, and all scores are dimensionless ranking units.
**CONSTRAINT:** The scorer sees a consistent corpus snapshot; query-term repetition is deliberately ignored here.
**NUMERICAL EXAMPLE:** N=4, df=2 yields idf+=ln(2)=0.693147; at f=1 and L=A, c=1 for any positive k1.
**INTERPRETATION:** A single occurrence in an average-length field contributes approximately 0.693147.
Let K(d)=k1*(1-b+b*L(d)/A), so the term response is (k1+1)*f/(f+K).
At fixed K>0, the derivative with respect to a continuous relaxation of f is (k1+1)*K/(f+K)^2 > 0.
The second derivative is negative, so the incremental gain from additional occurrences decreases.
At f=0 the term contributes zero, and the algebraic continuous limit as f grows is k1+1.
At b=0, K=k1 regardless of length; length normalization is switched off.
At b=1, K=k1*L/A, so normalization follows relative field length fully within this formula.
For equal f>0 and b>0, a longer field receives a smaller contribution, holding all other quantities fixed.
If A=0, the expression is undefined and must not be silently evaluated.
If scores are produced from multiple fields, field-specific populations and lengths must be described instead of treating all tokens as one field by assumption.

### 4.3 Key Special Cases / Variants

| Variant | Change | Use | Avoid / caveat |
|---|---|---|---|
| Raw-count TF-IDF T1 | Linear count times unsmoothed log rarity | Transparent baseline | Repetition can dominate; ubiquitous term scores zero |
| Vector TF-IDF | Query and document weights with optional L2 normalization | Compare directions in weighted term space | Cannot assume every implementation uses T1 or the same tokenizer |
| BM25 B1–B3 | Saturating count and length normalization | Lexical candidate ranking | Not semantic matching or calibrated relevance |
| Signed RSJ-style IDF | ln((N-df+0.5)/(df+0.5)) when relevance feedback is absent | Reproduce a specified signed formulation | Common terms can contribute negatively |
| Positive Lucene IDF | ln(1+(N-df+0.5)/(df+0.5)) | Match documented IDF convention | Formula alone does not reproduce engine scores |
| Fielded BM25F | Combine field evidence using a specified fielded model | Fields differ in utility and length | A simple sum of field scores is not automatically BM25F |

The signed IDF becomes negative when df>N/2, zero at df=N/2, and positive below that threshold.
A negative term contribution can make a document matching a common query term score below another document; candidate eligibility and treatment of unmatched documents must be specified before drawing conclusions.
Clipping negative values to zero, replacing them with a small constant, and using positive IDF are distinct scoring policies.
None is a neutral numerical implementation detail because each may change ranking.
Original Robertson–Zaragoza treatment describes probabilistic relevance framework and extensions; pin the exact formula and parameters when reproducing a paper [S3].
Lucene 9.12.1 documents defaults k1=1.2 and b=0.75, overlap-token handling, and its IDF calculation [S1].
Defaults and internal details are version-sensitive; consult the deployed version rather than assuming this historical version.

### Probabilistic motivation and model limitations

The probabilistic relevance framework ranks documents using evidence associated with relevance under a specified model [S3].
A relevance variable in that framework refers to a document-query relationship, not an inherent permanent document property.
Odds are a probability divided by its complement, provided the probability lies strictly between zero and one.
Taking a logarithm transforms products of positive odds factors into sums.
That algebra explains why a model based on independent evidence factors can produce additive term contributions.
The conditional independence assumptions in a simple probabilistic model are approximations, not facts about natural language.
Terms such as `running` and `shoes` are correlated in a commerce corpus.
Synonym expansion can create additional dependencies between terms generated from the same original query.
The relevance probability of one term cannot generally be inferred from document frequency alone.
Document frequency is available without individual relevance judgments and can serve as a pragmatic surrogate for rarity.
The smoothing constants in the signed expression avoid zero numerator and denominator in finite populations.
Those constants do not guarantee that estimated term effects are correct for a new query distribution.
The positive-IDF expression in B1 changes the signed expression's behavior for common terms.
At df=N/2, the signed weight is zero, while the positive weight is ln(2).
A scoring variant that removes common terms may lose meaningful matches for short queries.
A scoring variant that retains common terms can incur additional postings traversal.
The right decision depends on both judged relevance and the actual execution plan.
The term-frequency response in B2 is bounded but does not represent a literal count of independent relevance events.
Length normalization compensates for some relationship between document size and observed term frequency.
It does not fully distinguish verbose relevant documents from verbose irrelevant documents.
The parameter b expresses the strength of the selected correction, not a measured causal effect of length.
The parameter k1 changes the shape of frequency saturation and must be considered with the declared score scaling.
At fixed relative length and positive k1, the maximum response approaches k1+1 as frequency grows.
For an average-length field, the one-occurrence response remains one under this scaling.
Multiplying every term contribution by the same positive factor preserves ordering within a fixed query.
That property does not imply that separately scaled fields or heterogeneous query clauses preserve ordering.
In particular, a changed BM25 scale can change a combined score when non-BM25 features retain their old scale.
An implementation's raw score need not equal the textbook-scale score to yield the same single-field ranking.
To reproduce absolute scores, inspect boosts, implementation scaling, encoded norms, and the exact rewritten query.
To reproduce rankings, also specify tie-breaking and the eligible candidate universe.
Neither agreement of one score nor agreement of one ranking proves equivalence for all queries.
Use multiple synthetic boundary cases and held-out real judgments to distinguish implementation mismatch from expected variation.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

For a universal term with df=N, positive IDF remains greater than zero: ln(1+0.5/(N+0.5)).
For the same term the signed RSJ-style IDF is negative, so the two conventions are not interchangeable.
For a term absent from the snapshot, df=0 is mathematically admissible in B1 but there are no postings and thus no matching contribution.
A newly added document can change df, N, and A, potentially changing scores of documents whose text did not change.
A rare identifier can receive high lexical weight even when it occurs in an irrelevant context.
BM25's saturation limits repetition but does not guarantee immunity to keyword stuffing.
If a rare term is a typo, a high IDF may amplify an accidental exact match instead of recovering the intended term.
An all-stopword query can produce zero tokens; the fallback must be a separately specified product decision.
An eligibility filter is a hard constraint, while a score boost is not an access-control mechanism.
Takeaway: The exact variant, candidate universe, and boundary conditions determine whether the math applies.

## 5. Worked Numerical Example

**ILLUSTRATIVE, not measured:** Four fictional product documents contain only the terms shown; no stopwords, synonyms, field boosts, or duplicate query terms are used.
The query is `red shoe`, analyzed into two distinct lowercase terms.
The indexed lengths include every displayed token, including `boot` and `blue`.
Document A: `red red shoe`, L=3, f(red)=2, f(shoe)=1.
Document B: `red shoe shoe shoe`, L=4, f(red)=1, f(shoe)=3.
Document C: `blue shoe`, L=2, f(red)=0, f(shoe)=1.
Document D: `red boot boot`, L=3, f(red)=1, f(shoe)=0.
The average length A=(3+4+2+3)/4=3 tokens.
N=4 documents, df(red)=3, and df(shoe)=3.
Each query term has positive IDF ln(1+1.5/3.5)=ln(10/7) approximately 0.356675.
Choose k1=1.2 and b=0.75 solely for this example.
The length multipliers 1-b+b*L/A are 1, 1.25, 0.75, and 1 for A, B, C, and D respectively.
Thus K values are 1.2, 1.5, 0.9, and 1.2 respectively.
A red response is 2*2.2/(2+1.2)=1.375.
A shoe response is 1*2.2/(1+1.2)=1.
A score is 2.375*ln(10/7), approximately 0.847103.
B red response is 2.2/(1+1.5)=0.88.
B shoe response is 3*2.2/(3+1.5)=1.466667.
B score is 2.346667*ln(10/7), approximately 0.836997.
C shoe response is 2.2/(1+0.9)=1.157895.
C score is 1.157895*ln(10/7), approximately 0.412992.
D red response is 2.2/(1+1.2)=1.
D score is ln(10/7), approximately 0.356675.
The ranking for a disjunctive query over all four eligible documents is A > B > C > D.
A conjunctive query requiring both terms would instead admit only A and B.
The common positive IDF lets us independently check A versus B using 2.375 > 2.346667 before multiplying.
The score units are dimensionless and have no interpretation as a purchase probability.
If one mistakenly uses raw frequency without length normalization, B's total count of four exceeds A's three; that is a different scorer, not an arithmetic shortcut for B1–B3.
With b=0, B would outrank A because it has the same red count advantage in the other direction and an extra shoe repetition; changing b changes the model.
An actual engine can quantize field lengths or use a different score scale, so compare order and exact implementation-specific explanations, not this number blindly.
The corpus is too small to infer real relevance, click effects, or statistical reliability.
At production scale, postings decoding, filtering, concurrency, and index freshness must be measured independently.

### Contrast 1: Raw TF-IDF versus the BM25 ordering

The following calculation uses the same four synthetic documents and the same disjunctive query as the main example.
Use T1 with unnormalized raw document counts and a unit query weight for each distinct query term.
Both query terms have df=3 and N=4, so T1 assigns each an IDF of ln(4/3), approximately 0.287682.
Document A contains three query-term occurrences, so its T1 score is 3*ln(4/3), approximately 0.863046.
Document B contains four query-term occurrences, so its T1 score is 4*ln(4/3), approximately 1.150728.
Document C contains one query-term occurrence, so its T1 score is ln(4/3), approximately 0.287682.
Document D also contains one query-term occurrence and ties C under this particular T1 definition.
The raw TF-IDF ranking is B > A > C = D before applying a deterministic tie-breaker.
The earlier BM25 ranking was A > B > C > D for k1=1.2 and b=0.75.
The difference arises from both frequency saturation and length-dependent normalization.
That observation alone does not isolate which of those two mechanisms is responsible.
At b=0 with the same k1, A's combined response is 1.375+1=2.375.
At b=0, B's combined response is 1+1.571429=2.571429.
Thus B ranks above A without length normalization even though frequency still saturates.
The length factor reverses the A/B order for the selected b in this synthetic example.
Neither ordering is objectively superior without query-specific relevance judgments.
Applying cosine normalization to T1 would define another model and would require the complete vocabulary vector.
Using only query-term dimensions in that norm is a different convention from normalizing over the entire document vocabulary.
An unmatched term such as `boot` can affect a full-document cosine norm even though it contributes nothing to the dot-product numerator.
Comparing reported scores across these definitions without naming the query and document norms is invalid.

### Contrast 2: Length normalization sensitivity

Hold A=3 tokens, k1=1.2, and a single term's IDF fixed while changing b, length, and frequency.
These rows report the B2 frequency response before multiplication by IDF.
For b=0 and f=1, lengths 1.5, 3, and 6 all have response 1.
For b=0 and f=2, the same three lengths all have response 1.375.
For b=0.75, length 1.5 and f=1 give response approximately 1.257143.
For b=0.75, length 3 and f=1 give response exactly 1.
For b=0.75, length 6 and f=1 give response approximately 0.709677.
For b=0.75, length 1.5 and f=2 give response exactly 1.6.
For b=0.75, length 3 and f=2 give response exactly 1.375.
For b=0.75, length 6 and f=2 give response approximately 1.073171.
For b=1, length 1.5 and f=1 give response exactly 1.375.
For b=1, length 3 and f=1 give response exactly 1.
For b=1, length 6 and f=1 give response approximately 0.647059.
For b=1, length 1.5 and f=2 give response approximately 1.692308.
For b=1, length 3 and f=2 give response exactly 1.375.
For b=1, length 6 and f=2 give response exactly 1.
The length 1.5 and f=2 row is an algebraic sensitivity calculation, not a physically possible integer-token field; the other 1.5-length row is also not a literal token count.
The same relationship applies to actual integer lengths with the same L/A ratio.
At L=A, the normalization multiplier is one for every valid b.
At fixed f and L>A, increasing b decreases the response.
At fixed f and L<A, increasing b increases the response.
Length normalization is not simply a penalty applied to every document.
If all indexed documents have the same field length, changing b leaves their B2 responses unchanged.
Changing b cannot correct missing terms caused by a mismatched analyzer.
When L is correlated with product category, a global b can affect categories unevenly.
Stratify relevance evaluation by field length and category before adopting a different value.

### Contrast 3: Positive versus signed rarity weights

Consider an independent synthetic corpus of N=10 documents.
For a term appearing in df=2 documents, positive IDF is ln(1+8.5/2.5), approximately 1.481605.
For the same term, signed RSJ-style IDF is ln(8.5/2.5), approximately 1.223775.
For a term appearing in df=8 documents, positive IDF is ln(1+2.5/8.5), approximately 0.257829.
For the same common term, signed IDF is ln(2.5/8.5), approximately -1.223775.
Suppose both terms occur once in an average-length field with k1=1.2.
Their B2 responses are each one, so positive-IDF scoring gives approximately 1.481605+0.257829=1.739434.
Signed-IDF scoring gives approximately 1.223775-1.223775=0.
These two numbers belong to distinct model conventions; neither should be substituted for the other in a reproduction.
Suppose another candidate matches only the rare term with the same length and frequency.
Its positive-IDF score is approximately 1.481605, below the candidate matching both terms.
Its signed-IDF score is approximately 1.223775, above the candidate matching both terms.
Under a conjunction requiring both terms, the rare-only candidate is not eligible, regardless of its hypothetical score.
The eligibility rule therefore matters as much as the signed score when comparing the final result list.
If negative IDF is clipped to zero, the two candidates tie in this deliberately simple example.
The tie must still be resolved using a documented stable rule.
If the corpus is tenant-filtered after calculating global IDF, the df used here need not equal the tenant-visible df.
Recompute statistics on the declared population, not on an imagined population inferred from visible hits.

### Contrast 4: Query repetition and field weights

Suppose an analyzed query emits `red`, `red`, and `shoe`.
B3 deliberately sums distinct terms and therefore scores this query exactly like `red shoe`.
An alternative linear query-count policy could multiply the red contribution by two.
That alternative would change relative ordering for documents whose red and shoe contributions differ.
For A in the main example, the unweighted red and shoe responses are 1.375 and 1.
The distinct-term response sum is 2.375 before multiplication by their shared IDF.
Under the stated linear query-count alternative, A's response sum would be 2*1.375+1=3.75.
For B, the distinct-term response sum is approximately 0.88+1.466667=2.346667.
Under that same alternative, B's response sum becomes 2*0.88+1.466667=3.226667.
The linear query-count alternative is illustrative and is not asserted to match a particular engine's query builder.
Repeated query terms can instead be deduplicated, boosted, saturated, or transformed by another documented rule.
In a two-field example, define score=2*title_score+description_score as an explicit weighted sum.
If document X has title score 1 and description score 0.2, X receives 2.2.
If document Y has title score 0.5 and description score 1.3, Y receives 2.3.
Y wins under that rule even though X has the stronger title match.
Changing the title weight to 3 produces X=3.2 and Y=2.8, reversing their order.
This weighted sum is a constructed multi-field scoring policy, not a claim that it is BM25F.
Field scores can have different statistical distributions, and a larger weight does not by itself make them calibrated.
A field-boost decision should be evaluated across queries and field-length slices, not inferred from these two documents.

### Contrast 5: TF-IDF vectors and cosine normalization

Use an independent two-dimensional teaching space whose coordinates already contain the chosen term weights.
Let the query vector be q=(1,0), candidate X be (3,1), and candidate Y be (2,1).
The dot product with X is 3 and the dot product with Y is 2.
The L2 norm of q is 1, the norm of X is sqrt(10), and the norm of Y is sqrt(5).
The cosine score of X is 3/sqrt(10), approximately 0.948683.
The cosine score of Y is 2/sqrt(5), approximately 0.894427.
Both dot product and cosine rank X first in this example, but their score magnitudes differ.
Now consider Z=(10,10), whose dot product with q is 10.
Its cosine score is 10/sqrt(200), approximately 0.707107.
Dot product ranks Z above X, while cosine ranks X above Z.
The second coordinate raises Z's norm without contributing to this particular query's dot product.
Cosine similarity is undefined for a zero vector unless an implementation supplies an explicit convention.
The assumed vector coordinates are illustrative; they are not the T1 weights of the four-document example.
The example isolates vector normalization and does not establish that cosine is more relevant than BM25.
The scikit-learn transformer can apply L2 normalization, but its IDF smoothing and tokenization must also be specified [S2].
For nonzero nonnegative vectors, cosine lies between zero and one; signed feature vectors require a different range.
Cosine scores are still not calibrated relevance probabilities.

### Contrast 6: Corpus changes and score stability

Take a term that occurs once in a single document and set its B2 response to one.
In a synthetic N=100 corpus with df=1, B1 gives approximately 4.209655.
For N=100 and df=99, B1 instead gives approximately 0.014963.
The contrast shows that a term's contribution depends on its population frequency, not only its count in the document.
Now add a new matching document while holding the original document's text and length fixed.
Both N and df change, so recompute IDF rather than assuming the old weight remains valid.
If average field length also changes, the original document's B2 response can change as well.
An unchanged item's score can therefore drift without any change to that item's content.
Such score drift is not automatically a defect if the intended statistics have changed.
Unexpected rank changes should still be investigated against generation, shard, and statistics provenance.
A global score threshold chosen on a previous corpus snapshot can lose its original operating characteristics.
Revalidate any threshold after a population or analyzer change, using judgments from the intended deployment.
These are algebraic consequences of the declared model, not measured observations from a production index.

Takeaway: The average-length assumption, document eligibility, and scoring variant are essential to reproducing the order.

## 6. Visual / Geometric Intuition

```text
Term response for k1=1.2 and L=A (schematic; not to scale)
response
  2.2 |                         ........ asymptote
      |                   .....
  1.0 |        * f=1
      |     ...
  0.0 |*_______________________________ frequency f
       0       1                         ->
```
The curve starts at zero, rises with frequency, and approaches 2.2 without reaching it at finite f.
At f=1, the response equals one when the field length equals the average.
Longer documents shift the curve downward for the same f when b>0.
The axes are dimensionless response and token count; the drawing is illustrative rather than measured.

```text
red postings:   A(2), B(1), D(1)
shoe postings:  A(1), B(3), C(1)
AND intersection: A, B
OR union:          A, B, C, D
```
Parentheses show term frequencies, not ranking scores.
Positions would be needed to distinguish `red shoe` from `shoe red` as a phrase.
Takeaway: Postings determine which documents can be evaluated; saturation determines how repeated matches affect their scores.

## 7. Comparison / Selection Table

| Choice | Preconditions | Quality and cost | Appropriate when | Avoid when |
|---|---|---|---|---|
| Boolean AND | Compatible terms and postings | Strict matching; can lose recall | All terms are mandatory identifiers | Optional or paraphrased terms are common |
| Boolean OR | Candidate cap and safe filters | Broad recall; scoring work can grow | Partial matches are useful | A common term overwhelms a strict deadline without controls |
| TF-IDF T1 | Known statistics and token counts | Transparent linear response | A simple explainable baseline is needed | Frequency stuffing or length variation dominates |
| BM25 B1–B3 | Valid field statistics and tuned policy | Saturating lexical score; no semantic guarantee | Exact terms and interpretable baseline matter | Users rely on synonyms not covered by analysis |
| Phrase query | Positions stored and analyzer compatibility | Enforces positional constraint at added index/query cost | Exact sequence is essential | Word order varies legitimately |
| Hybrid candidates | Lexical and semantic routes plus merge contract | Can complement lexical vocabulary mismatch; actual gain unmeasured | Labels show recoverable lexical misses | Access filtering or evaluation is inconsistent between routes |

First disqualify any path that cannot enforce authorization and mandatory eligibility.
Next select strict matching when missing any term makes the result invalid.
Otherwise compare BM25 with a simple Boolean or TF-IDF baseline on held-out judgments and latency.
Use a separate semantic candidate route only when its incremental recall justifies operational complexity under measured constraints.
If relevance and latency conflict, preserve hard security and deadline constraints first, then test quality under the remaining budget.
Use matched queries, corpus snapshot, judgment protocol, software version, hardware, concurrency, and uncertainty intervals for comparisons.
No comparative quality, latency, or cost result is claimed here; the benchmark remains NOT MEASURED.

### Commerce selection scenarios

The following scenarios are illustrative decision exercises, not descriptions of real deployments.
For a query containing an exact SKU, preserve a normalized exact-identifier route before applying linguistic stemming.
A SKU route can fail if visually similar characters are normalized together when they denote distinct products.
An identifier lookup can bypass full-text ranking but still must enforce availability and authorization.
If multiple sellers offer the same SKU, the product-versus-offer entity decision determines how many results should appear.
For a broad category query, a disjunctive lexical route may provide useful candidates without forcing every modifier to match.
If a modifier represents a mandatory safety requirement, treat it as an explicit filter rather than merely a weak score contribution.
For a brand query, protect exact brand matches while testing whether broad synonym expansion introduces competing brands.
For a misspelled query, an exact lexical baseline provides a useful measurement of the gap before introducing correction.
A spelling correction must preserve the original interpretation as an option when the rewrite is uncertain.
For a product with a long description and short title, field-specific scoring avoids assuming the same length distribution for both.
A title boost should be tuned against judgments, because some products have uninformative or template-generated titles.
For a new catalog item with little description, missing text can limit lexical recall independently of BM25 parameters.
Inventory or geographic filters can remove an otherwise relevant product, which is an eligibility issue rather than a scorer defect.
If a seller duplicates a listing, deduplication policy can alter result diversity even when each score is calculated correctly.
A query such as `red shoes` may express a required color, a preference, or a category phrase depending on the product context.
The system should not infer this business semantics solely from a mathematical IDF value.
If exact phrase order matters for a branded name, use a position-aware phrase condition or an exact field.
If word order varies naturally, a strict phrase condition can suppress relevant results.
When returning variant-level results, labels must specify whether the desired color and size apply to the actual variant.
A product-level match is not necessarily fulfillable in the user's selected size.
When eligibility changes frequently, stale inventory can dominate perceived quality even if text scoring is accurate.
When seller incentives reward keyword repetition, audit term distributions and misleading descriptions.
Lexical transparency helps investigators explain a match but does not make the underlying seller text trustworthy.
The correct tradeoff between variety, price, fulfillment, and relevance belongs to the product objective and evaluation design.

### Advertising retrieval boundaries

An advertisement candidate can be indexed by keyword, creative text, product feed, or another specified targeting representation.
The exact retrieval representation and its normalization rules are configuration choices, not properties of advertising in general.
An indexed match should be treated as a potential candidate, not an entitlement to display.
The serving system must separately apply campaign status, targeting, budget, pacing, policy, and legal constraints where applicable.
An eligible candidate may still lose an auction or be suppressed by placement rules.
A paid result's lexical score is not itself a bid, clearing price, click probability, or expected revenue.
If lexical and auction features are combined, their units and combination rule must be explicit and validated.
Campaign targeting terms can include exact or broad matching modes whose semantics depend on the advertising platform.
Do not assume that a search engine's Boolean query is identical to a particular platform's keyword match type.
Negative keyword rules are separate from negative IDF; the former exclude or constrain matches, while the latter is a scoring convention.
Budget exhaustion is a campaign-state event and cannot be repaired by changing BM25 k1.
Delayed budget updates can produce serving errors even when candidate retrieval and lexical scores are correct.
For a diagnostic, compare matching candidates before campaign eligibility with the candidates after each eligibility gate.
Audit whether advertiser and user information is exposed through shared query logs or diagnostic APIs.
A retrieval system must avoid leaking one advertiser's campaign details through another advertiser's reporting interface.
Use synthetic campaign examples when developing cross-tenant security tests.
Offline ad relevance labels should be separated from auction outcomes to avoid conflating text match with bid strength.
Online ad-click outcomes depend on exposure, placement, creative quality, and user behavior as well as retrieval.
Changing candidate generation can alter auction competition, so an online experiment may have marketplace interference.
The appropriate interference model depends on the marketplace and is not derivable from this lexical example.
An ad candidate cutoff can affect both auction participation and computational cost.
Measure candidate coverage by advertiser segment and query class before changing that cutoff.
Do not claim that an increase in ad candidates increases revenue without a controlled, policy-compliant evaluation.
An ad may be textually relevant but unsuitable for display because a separate safety or eligibility rule fails.
Hard policy constraints take precedence over a relevance score or a bid.
The design and fairness of auction clearing are explicitly outside this sheet's scope.

### Deployment and measurement decision record

Write down the specific decision that the retrieval change is intended to improve.
Name the baseline analyzer, scorer, parameters, index generation, and candidate budget.
Record the proposed configuration with a distinct version and an owner who can approve rollback.
Identify the corpus snapshot and source offsets used for both sides of an offline comparison.
Use the same eligibility and authorization policies when comparing lexical quality.
Identify any unavoidable differences in policy state and report their expected effect on interpretation.
Define the query sample and its weighting before looking at the outcome.
Record the label source, grading rubric, missing-label convention, and adjudication method.
Specify primary retrieval metrics and slice-level guardrails before tuning.
Choose an exact baseline or a documented stronger reference for assessing approximate retrieval.
Measure index construction, storage, query work, and update impact as distinct costs.
Record the engine build, runtime settings, hardware, shard layout, and replica count.
Specify offered load, concurrency, warmup, cache state, and the duration of each latency measurement.
Use a workload representative of the deployment rather than a single hand-picked query.
Include high-frequency terms, long queries, identifiers, multilingual text, and restrictive filters in the workload.
Include update and merge activity if they will coexist with serving traffic.
Measure the distribution of end-to-end latency, not just mean service time.
Record timeouts, rejected requests, and partial results as outcomes rather than silently dropping them.
If using a percentile, state its population and observation window.
Do not add a shard P99 to a network P99 and call the sum an end-to-end P99.
Measure actual end-to-end requests or use an appropriately justified distributional model.
Report resource utilization and cost alongside quality so a quality improvement is not mistaken for a free change.
Currency estimates require a price date, billing basis, included resources, and excluded operational costs.
No currency estimate is supplied here because the environment and prices are not specified.
Check whether changes in candidate count affect downstream ranker capacity.
Check whether new analyzer resources need distribution to every query and indexing node.
Verify that older indexed text and newer query analyzers remain compatible throughout a rolling deployment.
Prepare a rollback plan that preserves deletion and authorization requirements.
Deploy with a bounded exposure and prespecified stop conditions where the serving platform permits it.
Compare diagnostic signals against the same traffic population and time period when possible.
Avoid declaring success solely because a single synthetic query has improved.
Record negative results and uncertainty as well as improvements.
When an experiment is underpowered, do not convert a nonsignificant result into proof of equivalence.
When quality improves but a security constraint fails, reject the deployment until the constraint is met.
When latency exceeds a hard deadline, choose a safer configuration or reduce workload only with a documented quality impact.
When label coverage is weak for a locale, collect qualified judgments before extrapolating from other languages.
When the corpus distribution changes, schedule a re-evaluation rather than assuming the old optimum remains valid.
Keep a reproducible record linking the release to source versions, analyzer resources, scorer parameters, and benchmark artifacts.
Document which claims are derived, sourced, measured, or still open.
The decision record is an illustrative operational practice, not evidence of a completed production rollout.

Takeaway: Select a constrained retrieval policy against labels and operational budgets rather than treating one scorer as universally best.

## 8. Failure Modes & Edge Cases

All cases below are HYPOTHETICAL diagnostic exercises, not reported incidents.

**F-01 — Analyzer mismatch.** A known relevant item disappears after a deployment.
Competing explanations include stale indexing, an eligibility filter, and inconsistent query/index tokenization.
Inspect emitted tokens for the exact document field and query with their respective version identifiers.
If the query token is absent from the field's postings but the unfiltered source contains the intended text, analyzer mismatch is supported.
A matching token does not eliminate filtering or stale visibility as separate causes.
Restore a compatible analyzer or rebuild the index under a tested version; verify with a fixed token-and-hit regression set.

**F-02 — Incorrect population statistics.** The same term changes rank unexpectedly after a shard move.
Compare N, df, and average field length used by each shard against the intended global or local population.
If statistics differ while query, field values, and parameters are unchanged, a local-statistics effect is plausible.
A matched-statistics replay is needed before attributing the whole regression to that effect.
Use a documented global-statistics mode where supported or evaluate the tradeoff of the deployed local mode.
Monitor ranking and score-explanation drift by shard and index generation.

**F-03 — Long-field suppression.** A relevant long description ranks below a weaker short description.
Inspect field length, average length, term frequencies, and per-term response with and without normalization.
A lower contribution at equal frequency and higher length is expected under B2, not necessarily an implementation bug.
Compare field-specific analysis and a tuned b on held-out judgments rather than removing normalization universally.
Verify by document-length buckets and guard against stuffing after any change.

**F-04 — Missing fresh catalog item.** A newly published SKU cannot be found by its identifier.
Compare source commit time, change-event offset, indexing acknowledgment, refresh visibility, and delete/update ordering.
If the source event is present but visibility has not advanced, the delay is in the indexing or refresh path.
If the event is missing from the stream, reindexing the query service alone does not repair the source feed.
Restore a verified checkpoint or replay idempotently, then confirm visible generation and result eligibility.
Track source-to-search lag separately from request latency.

**F-05 — Unauthorized hit.** A restricted record appears in another tenant's results or facets.
Treat this as a security incident; contain access before investigating relevance quality.
Check whether the authorization predicate was applied before result exposure, aggregation, snippets, and caching.
A post-ranking UI filter does not protect snippets, counts, timing, or an alternate endpoint.
Fix the trusted server-side authorization path, invalidate unsafe caches, audit affected access, and verify negative tests across tenants.
If authorization data is unavailable, fail closed according to the product's explicit security policy.

**F-06 — Query unexpectedly returns nothing.** Analysis may remove all tokens or a strict conjunction may overconstrain the query.
Inspect analyzed terms and compare eligible counts for each individual posting and their intersection.
If no terms remain, do not interpret an empty result as proof that the corpus lacks relevant documents.
Apply a documented safe fallback, such as a user prompt or curated browse experience, rather than silently dropping access constraints.
Add empty-analysis and punctuation-only cases to analyzer regression tests.

**F-07 — Latency spike on common terms.** A broad query may decode many postings, but queueing, disk misses, and network skew are alternative causes.
Compare postings visited, cache misses, per-shard service time, queue time, and fan-out for matched traffic slices.
High visited-postings counts with stable queue time support query-work pressure; high queue time without increased work suggests capacity contention.
Use safe pruning, bounded candidate windows, or resource controls only after checking their effect on recall and exactness.
Replay under realistic concurrency and examine end-to-end latency percentiles rather than adding component percentiles.

**F-08 — Unexpected negative scores.** A signed IDF variant may penalize very common terms.
Recompute df/N and inspect the exact scoring source and any IDF floor.
If df>N/2 and signed RSJ is in use, the negative IDF is expected.
Do not clip scores without defining the changed ranking semantics and evaluating them.
Record the deployed formula and make query-operator behavior explicit when unmatched documents compete.

**F-09 — Clicks improve while usefulness falls.** Position and exposure bias can distort click-based labels.
Inspect judged relevance and outcomes by rank, query type, and cohort alongside online outcomes.
Randomized or otherwise defensibly controlled exposure is needed for causal comparison; raw click counts alone are insufficient.
Keep business and user-harm guardrails separate from click-through rate.
A result can attract a click through an attractive snippet while failing the underlying task.

**F-10 — Deleted data remains accessible.** A deletion may reach the source but not all serving replicas, caches, or snapshots.
Trace the deletion version through the change stream, each index generation, cache invalidation, and authorized read path.
Distinguish logical search invisibility from later physical removal in segments, replicas, backups, and logs.
Contain exposure and follow the applicable retention and deletion policy before declaring resolution.
Verify both ordinary search and alternate retrieval paths with a non-sensitive synthetic deletion probe.
A planned collection Common Issues artifact does not yet supply stable issue IDs.

### Authorization failure analysis and containment

The following cases are hypothetical and use a proposed search architecture rather than documented incidents.
An access-control list, or ACL, associates an entity with principals or groups permitted to access it.
The request's identity must come from an authenticated context, not from an arbitrary client-provided tenant string.
Resolve group membership and tenant membership according to the service's policy authority.
Build the search predicate from trusted policy data and bind it to the authenticated request.
Apply the predicate to hits, total counts, facets, snippets, highlights, autocomplete, and other exposed derivations.
A policy applied only after snippets are generated can leave sensitive fragments in logs or intermediate caches.
An aggregation over unauthorized records can reveal their existence even if individual records are hidden.
A count difference before and after a filter can be sensitive in a small population.
Do not assume that hiding a field from the response prevents all inference from ranking or counts.
Whether timing is an exploitable side channel depends on the threat model and requires separate assessment.
An administrator's diagnostic endpoint needs its own authorization boundary and audit trail.
Do not expose raw document identifiers or term explanations merely because the UI hides document content.
Cache keys for result sets must include all authorization-relevant dimensions or use a provably safe equivalent.
A cached result generated for one tenant must not be served to another tenant with a different policy context.
Group membership changes can invalidate old cache entries even when the query text is unchanged.
A query cache shared across principals needs a documented separation between reusable unfiltered structures and sensitive results.
An index-level filter is not a substitute for verifying that the correct policy version reached each replica.
When policy updates lag, define a maximum permissible staleness according to the actual security requirement.
If a revocation must take effect immediately, eventual propagation alone may not satisfy that requirement.
A separate authoritative deny check at exposure time can be considered, subject to its own availability and latency contract.
The fail-closed behavior of that check must be specified for timeouts and partial outages.
Filtering after taking a top-K list may return fewer than K authorized hits even when more exist.
Increasing the retrieval window can reduce that shortfall but cannot repair a missing or incorrect authorization predicate.
Pushing the authorization predicate into candidate generation can avoid wasting ranker capacity on disallowed documents.
Whether a specific engine physically evaluates a filter before scoring is an implementation detail to verify separately.
The security property is that unauthorized information cannot reach the caller through any supported output path.
An attacker-controlled query may contain broad wildcards, repeated clauses, or expansions designed to consume resources.
Apply query complexity budgets and rate limits appropriate to the service's authenticated and anonymous traffic.
An attacker-controlled document can repeat terms to manipulate lexical score even with saturating response.
Use source trust, ingestion validation, and relevance review rather than assuming BM25 removes this risk.
Index poisoning can affect both direct matches and shared corpus statistics.
Separating trust domains or constraining untrusted content can reduce some effects but requires evaluation of coverage and cost.
Personal data in queries can appear in access logs, traces, relevance examples, and error reports.
Minimize retention and access to these records according to the governing policy and applicable law.
Prefer aggregated metrics or synthetic probes when raw query content is unnecessary.
Encryption at rest and in transit does not replace authorization at query time.
Key rotation, backup access, and incident access need separate operational ownership.
Access tests should cover a permitted principal, a denied principal, and a recently revoked principal.
Test an unauthorized document that would otherwise be the highest-scoring result.
Test a no-hit query where an unauthorized record contains the only matching term.
Test whether facets or counts change when that unauthorized record exists.
Test autocomplete and suggested queries under the same tenant boundaries as full search.
Test caches after changing group membership without changing the query text.
Test partial failures of the policy service, index replicas, and cache invalidation path.
Record the policy version, index generation, and cache state needed to reproduce a failure without storing secret content.
When a leak is suspected, disable or isolate the affected route before optimizing recall.
Preserve relevant audit evidence under the incident process without widening access to the leaked material.
Evaluate whether source documents, derived snippets, snapshots, logs, and backups require separate remediation.
A successful synthetic regression test verifies the tested path, not the absence of every possible side channel.
Independent security review is appropriate when policy complexity or data sensitivity exceeds the team's threat-model expertise.

### Freshness, updates, and recovery

Consider a fictional catalog update that changes a product from available to unavailable.
The authoritative source may update before the search index refreshes.
During that gap, an old indexed availability flag can allow a stale result unless a stronger gate checks current availability.
The acceptable freshness bound depends on the harm of showing an unavailable item and on fulfillment behavior.
Measure source-event time to search-visible time separately from request-response latency.
Clock skew can make cross-system timestamp differences misleading unless clocks and timestamp semantics are controlled.
Track source offsets or version numbers as well as wall-clock timestamps when possible.
A queue backlog can increase freshness lag without increasing the latency of already indexed queries.
A service can meet its search latency objective while serving an unacceptably stale corpus.
A duplicate event can be harmless only if processing is idempotent under the actual update semantics.
An older event arriving after a newer event can regress the indexed record if version checks are absent.
A tombstone must be ordered relative to updates so that a late update does not resurrect a deleted record.
Replay requires retaining enough source history or a current snapshot to reconstruct the intended state.
A checkpoint should identify both the data position and the index generation to which it applies.
Rebuilding from an inconsistent snapshot and then applying an overlapping change stream requires deduplication or version checks.
The exact snapshot-and-stream handshake depends on the source and must not be invented from generic diagrams.
During a blue-green rebuild, compare document counts only as a coarse check, not as proof of content equivalence.
Two indexes can have the same document count while containing different identifiers or field values.
Compare sampled identifiers, analyzer outputs, deletion state, and query-level outcomes across generations.
A rolling alias switch can create mixed-generation observations if different requests resolve at different times.
Include the resolved generation in traces to make those differences diagnosable.
Rollback is unsafe if it reintroduces records deleted for privacy or security reasons.
Apply current tombstones and policy updates to any rollback target before exposure.
Restore procedures must also cover index settings, analyzer resources, synonym files, and access-policy metadata.
A backup that lacks its schema or analyzer version may restore bytes without restoring the intended behavior.
The recovery point objective concerns tolerated data loss, whereas recovery time objective concerns restoration duration.
Neither objective is established merely by creating periodic backups.
Exercise recovery with representative data volume and verify the resulting query behavior.
An active index can be healthy while a passive replica is stale or missing a recent deletion.
Check every serving replica or use a routing guarantee that excludes unhealthy replicas.
Freshness dashboards should distinguish normal refresh delay from event-ingestion failure.
For a stuck document, trace source version, stream partition, processing result, index acknowledgment, and visibility.
Escalate to the source owner when the source event never existed; search-side tuning cannot create missing authoritative data.

### Diagnostic experiments that distinguish competing causes

For a missing item, first query a known stable identifier with the same principal and eligibility context as the user.
If the identifier is absent, inspect source version and index visibility before tuning relevance.
If the identifier is present but the natural-language query misses, inspect analyzer output and query operators.
If the candidate is retrieved but ranks poorly, compare per-field contributions and downstream ranker features.
If the item is highly ranked for an administrator but absent for a user, compare policy and locale predicates.
For an unexpected match, inspect the exact token path, synonym expansion, and field that admitted the document.
If only the expanded query matches, the synonym rule is a more direct suspect than corpus freshness.
If only a long description matches, inspect field boosts and whether the title should carry more evidence.
For score drift, hold query text, principal, analyzer version, and document ID fixed across index generations.
Then compare N, df, average field length, field norms, boosts, and shard-local statistics.
For high latency, compare queue time against actual execution time before changing the scorer.
If queue time rises while per-request work remains stable, capacity or admission control may be the dominant issue.
If postings visited rise on the same query class, inspect new query expansion, corpus growth, and term frequency skew.
If only one shard is slow, inspect that shard's size, cache state, storage, and routing distribution.
If all shards are slow, investigate shared dependencies, traffic, or a common query-plan change.
If latency improves when caches are warm, measure cold-start behavior separately rather than reporting only the warm result.
For a quality regression, compare candidate recall before comparing final ranking metrics.
If recall is unchanged but final order deteriorates, inspect downstream ranking and feature freshness.
If recall declines, examine analyzer, eligibility, candidate cutoff, and index visibility before tuning the reranker.
For suspicious click gains, inspect result exposure and position distributions in addition to click counts.
If the treated group receives different query types, randomization or stratification needs investigation.
For a synonym change, compare both intended matches and precision regressions on ambiguous terms.
For a length-normalization change, compare judged outcomes by length bucket and field type.
For an index migration, compare exact eligible sets before attributing differences to scoring.
For a policy migration, compare denied-case behavior even if aggregate relevance appears unchanged.
Document a negative test result as evidence against a specific hypothesis, not as proof that every other subsystem is healthy.

Takeaway: A useful incident diagnosis distinguishes competing mechanisms and verifies a bounded correction.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let T be the total number of analyzed token occurrences, V the distinct term count, and P the total number of distinct term-document postings.
Let m be the number of distinct analyzed query terms, p_i the length of posting list i, and K the requested output count.
A simple in-memory inverted-index build can consume O(T) expected hash operations before any optional sorting, assuming expected constant-time hashing.
Sorting postings or dictionary keys adds implementation-dependent sorting work; external indexing also incurs disk I/O and merges.
Stored term-document postings require O(P) entries, with additional space for dictionary, frequencies, positions, field lengths, and metadata.
Storing positions can require O(T) position entries before compression; repeated tokens matter even when P is small.
A straightforward OR traversal can visit up to sum(p_i) postings, excluding filter, decoding, and heap costs.
If each matched document receives a heap update, an illustrative top-K implementation adds up to O(U log K) work for U distinct matched documents.
A Boolean AND can intersect sorted lists in O(sum(p_i)) worst-case pointer advances; skip structures may help on specific distributions without changing that generic bound.
A hash-based accumulator may require O(U) auxiliary score storage; alternative document-at-a-time designs have different working sets.
Compression saves space and I/O at the cost of decoding work; whether it improves latency is workload dependent.
Update cost depends on segment creation, refresh, merge, replication, and deletion policy, not just changed token count.
Recovery may require replaying source events or rebuilding T tokens; the choice depends on checkpoints and retained source history.
Distributed search adds fan-out, shard-level coordination, network transfer, and straggler sensitivity.
The number of returned hits is not a proxy for the number of postings examined.

### 9b. Back-of-Envelope Sizing

**ILLUSTRATIVE, NOT MEASURED:** Suppose 1,000,000 documents each contain 100 indexed token occurrences on average.
The total is 100,000,000 token occurrences, before excluding repeated terms within a document.
Assume, only for payload budgeting, that one position requires an average of two bytes after a hypothetical encoding.
Position payload alone is 100,000,000 * 2 = 200,000,000 bytes, approximately 190.73 MiB.
This calculation does not imply that a particular codec achieves two bytes per position.
Three complete replicas of that position payload would require 600,000,000 bytes, approximately 572.20 MiB.
The calculation excludes term dictionaries, document IDs, term frequencies, norms, source storage, metadata, checksums, merges, and filesystem overhead.
If actual position encoding averages four rather than two bytes, the position component doubles; measure on a representative index.
The environment, hardware, engine version, concurrency, and monetary cost are unspecified; no throughput or price is inferred.
A meaningful production experiment measures index size and merge amplification alongside P50/P95/P99 request latency at a stated offered load.
Percentiles describe distributions and are not additive across components.

### Query evaluation and exactness

A posting list is normally ordered by document ID, not by BM25 score.
The term dictionary lookup locates each query term's list before documents are accumulated or traversed.
An accumulator-based OR implementation can add each term's contribution into a per-document map.
After all relevant postings are processed, it can select the highest-scoring eligible documents.
This simple algorithm is exact for the declared score if no contribution or eligible document is omitted.
Its memory can grow with the number of unique matched documents.
A document-at-a-time traversal instead aligns postings cursors by document ID and computes a candidate's total contribution together.
Its practical memory requirements depend on the implementation and the way top-K candidates are maintained.
The two strategies can return the same exact ranking while having different cache and memory behavior.
An AND query admits only IDs present in every required list.
An OR query admits IDs present in at least one optional list, subject to minimum-match and eligibility rules.
A query with both required and optional clauses needs an explicit rule for whether optional clauses affect eligibility or only score.
The shortest-list-first heuristic can reduce intermediate intersections, but actual cost depends on correlation and skip behavior.
The shortest list does not necessarily identify the most relevant document first.
For sorted postings A=[1,4,9] and B=[2,4,8,9], an exact intersection emits 4 and 9.
Start at 1 versus 2 and advance A because 1 cannot appear later in B.
Compare 4 versus 2 and advance B because 2 cannot appear later in A.
Compare 4 versus 4, emit 4, and advance both cursors.
Compare 9 versus 8 and advance B; then compare 9 versus 9 and emit 9.
At least one cursor reaches its end, so the traversal terminates.
For the same inputs, the union is [1,2,4,8,9], with equal IDs emitted once.
The intersection size is two and the union size is five.
These synthetic lists illustrate correctness and do not measure CPU time.
A posting skip pointer can jump over IDs known to be below the current target.
The jump must not pass a possible match if exact conjunction is required.
Skip structure density trades index space against traversal work.
The optimal skip configuration depends on list lengths, compression blocks, query distribution, and hardware.
An engine may decode postings in blocks, so advancing one logical posting need not equal one physical memory access.
Compressed docID gaps can reduce storage because sorted identifiers are represented as differences.
The first ID and each later positive gap reconstruct the original strictly increasing document IDs.
For IDs [4,9,15], the initial ID and gaps are [4,5,6].
Adding gaps cumulatively recovers 4, 9, and 15.
Duplicate docIDs for a term require aggregation or a different declared postings representation.
Frequency values can be stored alongside each docID or derived from positions, depending on the design.
Position gaps can also be compressed, but phrase verification still needs the correct relative positions.
For a document with `red` at positions [1,5] and `shoe` at [2,7], the exact adjacent phrase occurs at positions 1 and 2.
The occurrences at positions 5 and 7 are separated by one intervening position and do not constitute the same exact adjacency.
If tokenization removes words or emits synonyms, position increments must be interpreted under that analyzer's contract.
The original character offsets used for highlighting are distinct from posting positions used for phrase matching.
An index without positions cannot generally distinguish an adjacent phrase from an arbitrary co-occurrence using postings alone.
One alternative is a slower verification against stored text, but that requires compatible analysis and has a different cost.
Phrase clauses can be combined with term clauses as hard constraints or as boosts.
Treating a phrase as a boost preserves nonphrase candidates; treating it as a required clause can reduce recall.
A query that requires every analyzed word can be brittle when users include optional adjectives.
A minimum-match rule relaxes that constraint but needs a specific threshold and behavior for short queries.
For four optional terms and a minimum of three, a document matching exactly two is ineligible.
For a two-term query, the same absolute threshold of three would make every document ineligible.
Percentage-based rules also need rounding semantics for short queries.
An exact top-K algorithm may discard a candidate only when a valid upper bound proves it cannot enter the result set.
The upper bound must cover every scoring contribution that the candidate could still receive.
Negative contributions, boosts, and custom scoring clauses can invalidate a bound derived for a simpler nonnegative model.
Ties require an upper-bound comparison that preserves the documented stable tie-breaker.
A heuristic cutoff that skips possible winners is approximate and should be evaluated for its recall or ranking loss.
Limiting the number of returned hits alone does not ensure that fewer postings were visited.
Early termination can be exact under special sort and scoring conditions, but those conditions must be established for the deployed engine.
An approximate method should report its quality relative to an exact or stronger reference on representative queries.
Measure traversal work per query class rather than assuming equal work for every query of the same length.
Rare terms usually have shorter lists than common terms, but the actual dictionary statistics determine each list's length.
An adversarial common-term query can approach a full-index traversal even with an inverted index.
Caching a hot term's postings can reduce I/O while leaving scoring and filter work.
Caching final hits can avoid more computation but requires authorization- and freshness-safe keys.
Memory pressure can evict useful posting blocks and make latency distributions sensitive to concurrent queries.
Measure cold-cache and steady-state behavior under the intended update load.
Index merging competes for CPU and I/O with foreground queries and can alter tail latency.
Separating query and indexing resources can reduce contention, but replication and consistency add their own costs.
Top-K heap operations have different cost when K is one, ten, or thousands.
Deep pagination can require retaining or transferring far more candidates than a first-page request.
Cursor-based pagination needs a stable sort key and snapshot semantics to avoid duplicate or missing results during updates.
Returning a total hit count may require additional work even when the caller only displays the first page.
An approximate count must be labeled as such and must still respect access restrictions.
Distributed top-K search typically combines per-shard candidates at a coordinator.
If shard-local statistics differ from global statistics, scores may not be directly comparable without the selected engine's correction strategy.
A per-shard candidate cutoff can omit a document needed for the global top K if the local scoring and merge contracts do not support it.
Network transfer grows with the number and size of returned candidates, fields, explanations, and facets.
Fetching large stored fields can dominate after scoring even when candidate selection is fast.
Separate query-phase and fetch-phase metrics to identify which component needs optimization.
Compare resource cost per successful authorized query, not only the average cost per incoming request.

### Evaluation protocol and quality measurement

Define the retrieval task before selecting a metric: product discovery, exact SKU lookup, and ad eligibility have different error costs.
Define the query population, time window, locale, user context, and eligible document universe.
Keep a test set separate from the examples used to tune analyzers, boosts, and BM25 parameters.
Sample both frequent and rare queries if the deployment serves a long-tailed query distribution.
Report a traffic-weighted aggregate and important query slices rather than relying on one average.
An aggregate can improve while a protected locale or high-stakes identifier slice deteriorates.
Create a relevance rubric that defines the difference between exact, acceptable, and irrelevant matches.
If labels are graded, document how each grade maps to the selected gain function.
Ask assessors to judge query-document pairs without displaying the candidate's current rank when feasible.
Blinding rank helps reduce anchoring but does not eliminate all assessor bias.
Use multiple assessors and an adjudication policy for ambiguous or high-impact examples.
Record disagreements rather than silently converting disputed judgments into certainty.
An incomplete judgment pool can make an unjudged relevant result look irrelevant under some evaluation conventions.
Report how unjudged documents are treated and how pools were built.
Pooling results from multiple candidate systems can reduce, but not eliminate, blind spots.
Evaluate candidate recall at the actual downstream budget K, not only at a much larger offline cutoff.
If the total relevant universe is not known, label the recall estimate as pool-relative.
Precision at K needs a clear denominator when fewer than K eligible documents exist.
One common definition still divides by K; another product-specific measure may divide by returned results.
Those conventions answer different questions and should not share an unlabeled metric name.
For a binary illustration, suppose the first five results have relevance labels [1,0,1,1,0].
Under the fixed-K definition, precision at five is 3/5=0.6.
If the evaluation universe has six relevant documents, recall at five is 3/6=0.5.
The two values differ because their denominators represent different populations.
If the universe has only four known relevant documents in an incomplete pool, 3/4=0.75 is pool-relative recall, not established true recall.
For ranked utility, a gain-and-discount metric can credit higher ranks more than lower ranks.
Its exact gain function, discount, cutoff, and zero-ideal handling must be specified before numbers are compared.
Do not compare two reported NDCG values unless their label mapping and evaluation protocols are compatible.
Candidate recall can be high while first-page precision is poor because the ranker orders the candidates badly.
First-page precision can look good while long-tail candidate recall is poor if easy queries dominate.
Measure zero-result rate, but distinguish no eligible documents from analysis failures or overly strict matching.
Measure reformulations and abandonment cautiously because users can reformulate for reasons unrelated to relevance.
Measure clicks by position because highly visible results receive more exposure.
Click-through rate uses clicks divided by a stated exposure denominator, not simply the number of users in a cohort.
Clicks do not necessarily imply satisfaction, purchase, or a successful task.
Purchases and revenue are affected by price, stock, delivery, and other changes that may confound search comparisons.
Avoid treating a revenue increase as a relevance improvement without a causal design and guardrails.
Randomized online experiments can support causal comparison when assignment, exposure, and interference assumptions are reasonable.
Define the randomization unit, treatment, eligibility population, primary outcome, and duration before examining outcomes.
If the same user appears in both variants because assignment is unstable, the comparison can be contaminated.
If sellers alter behavior in response to treatment, interference may complicate the interpretation.
Check sample ratio and exposure logs before interpreting an experiment's estimated effect.
Monitor harm and operational guardrails separately from the primary outcome.
Use confidence intervals or another prespecified uncertainty procedure appropriate to the sampling and assignment design.
Repeated queries from the same user are not necessarily independent observations.
The unit used in uncertainty estimation should reflect the actual randomization and dependence structure.
Multiple tuning experiments can select a seemingly successful variant by chance.
Maintain a held-out confirmation set or a prespecified confirmation experiment for the final choice.
Offline judgments can diagnose mechanisms but cannot alone establish causal online impact.
Online engagement can detect behavioral changes but does not by itself explain whether retrieval or ranking caused them.
Log candidate-stage and ranking-stage identifiers so controlled analyses can distinguish those mechanisms.
Protect user privacy when linking sessions, outcomes, and treatment assignment.
Retain only the data needed for the specified evaluation and governance requirements.
Publish the corpus snapshot, query sample, scoring configuration, and uncertainty alongside any internal comparison.
This sheet provides a protocol only; no relevance, conversion, or latency improvement has been measured.


### Index statistics and distributed scoring diagnostics

The following discussion describes possible implementations; consult the deployed engine for its actual behavior.
A corpus population may count every indexed document, only documents with a field, or another implementation-defined population.
Field-specific statistics can differ when some documents omit that field entirely.
An empty field is not necessarily treated like an absent field for norms or field counts.
Multi-valued fields may be analyzed as one logical sequence or as separate values with configurable position gaps.
Those gaps can affect phrase matching across values even when the token strings are identical.
If one physical product becomes several variant documents, document frequency changes even if source text is unchanged.
Duplicating a term across many variants increases its document frequency more than repeating it within one document.
Duplicating an entire catalog can therefore change rarity weights and the relative importance of terms.
Indexing separate seller offers can overweight products with many sellers in candidate sets.
Grouping by product after retrieving offers can reduce duplicates but may reduce the number of unique products returned.
An upstream grouping decision changes both the evaluation unit and the computational workload.
Distributed shards can contain different proportions of product categories or languages.
If scoring uses local df, a term concentrated on one shard can receive a different weight than on another.
If global statistics are requested, an extra coordination phase may be required depending on the engine.
The latency cost and consistency semantics of that phase are implementation- and workload-dependent.
Global statistics can improve consistency of weights without guaranteeing improved judged relevance.
For a reproducible scoring investigation, capture the exact query mode and shard allocation.
Also capture whether deleted documents remain included in any engine statistic before segment cleanup.
Do not assume logical deletion immediately changes every low-level statistic in every engine.
Segment merges can change the physical layout and statistics available to an implementation.
An unchanged query may show score differences after a merge even when the visible document set is stable.
Before treating that difference as a defect, reproduce it against the engine's documented statistical semantics.
If exact score matching is required for a migration, include analyzer versions, norms encoding, and floating-point precision.
Many search engines quantize or encode norms, so a mathematically exact token count may not reproduce their scoring value.
Term overlap treatment matters when a synonym filter emits several alternatives at the same position.
Counting overlaps can increase recorded length while discounting overlaps can leave it unchanged.
The selected overlap policy affects normalization and can alter relative scores.
Repeated terms in a query may become separate clauses or a single weighted clause depending on the query parser.
Capture the rewritten query tree rather than inferring query semantics solely from the user-entered string.
Field boosts can be applied before or after other score transformations depending on the query implementation.
The order of non-linear transformations can change rank even if the same nominal weights are used.
If score explanation does not include a feature or boost, inspect the downstream stage rather than assuming it was absent.
Scores from independent shards or retrieval routes should not be merged blindly when their scales have different meanings.
Rank-based fusion is one possible alternative, but its choice and weights require separate evaluation.
Calibration of scores requires labels and a target definition, and can drift with corpus or query mix.
Adding a constant to all candidates preserves a ranking only if the same constant is actually applied to every candidate.
Multiplying all scores by the same positive constant also preserves their order.
Multiplying by a negative constant reverses order and is not a harmless scale change.
Applying different constants by field, document, or shard can change order.
An explanation can support debugging without establishing that the score is causally responsible for an observed business outcome.

### Capacity planning for a lexical service

Begin a capacity estimate with the expected corpus size and the distribution of analyzed field lengths.
Estimate distinct term-document postings separately from total token occurrences.
Repeated words increase token occurrences without necessarily increasing distinct postings.
Estimate position storage only for fields whose query features require positions.
Estimate term dictionary size using vocabulary growth from languages, identifiers, and n-gram policies.
Estimate stored field payload separately from inverted-index payload.
Budget replication as an explicit multiplier on replicated components rather than assuming every byte is replicated identically.
Budget transient build and merge space in addition to the final steady-state index size.
Budget retained snapshots and backups independently of live-serving replicas.
Consider how deletion retention and compaction delay affect peak disk consumption.
Estimate query load as a distribution across time, not only a daily average.
Include bursts caused by traffic spikes, cache expiration, scheduled crawlers, and client retries.
Record the fraction of exact identifier, common-term, phrase, filtered, and expanded queries.
Estimate postings traversal from the actual query mix rather than average term frequency alone.
A small number of very common terms can dominate work for a subset of requests.
Measure fan-out and coordinator cost as shard count changes.
More shards can increase parallelism while also increasing network and coordination overhead.
Fewer shards can reduce fan-out while making individual shards larger and slower to rebuild.
Replica count can improve read capacity but also increases update traffic and storage.
The best shard and replica count depends on workload, reliability requirements, and topology.
Test an index rebuild while representative query traffic is active if they share resources.
Measure how merge activity changes CPU, I/O, disk headroom, and query tail latency.
Measure the effect of analyzer expansion on both vocabulary size and postings length.
Measure the effect of field boosts on ranking quality; boosts themselves may have negligible or substantial cost depending on implementation.
Measure the cost of complex authorization predicates at the expected number of principals and groups.
High-cardinality ACLs can increase index size or filtering work depending on representation.
Cache effectiveness depends on whether users share queries and authorization contexts.
Privacy requirements may disallow broad sharing of otherwise reusable query results.
Use controlled overload tests to identify the point where queue time and rejection rate rise.
Admission control can protect the service from overload but may reject expensive legitimate queries.
Define a user-visible error or degraded response when a deadline is exceeded.
A partial result must disclose its incompleteness where the product contract requires it.
Do not bypass security or policy filters to meet a latency target.
An emergency fallback should have its own quality, access-control, and observability tests.
Track freshness lag during overload because query success can hide an accumulating indexing backlog.
Track time to recover from that backlog after traffic returns to normal.
If arrival rate remains above sustainable indexing capacity, added buffering merely delays the eventual failure.
Capacity margins are planning assumptions until validated under representative load.
Separate infrastructure spend from engineering labor and from downstream business outcomes.
Compare cost per successful authorized query with quality and freshness held to acceptable levels.
For ad retrieval, account for candidate generation cost without assuming every candidate produces a payable event.
Review capacity assumptions after meaningful changes in traffic mix, corpus size, or analyzer expansion.
This is a measurement plan; no deployment size, cost, or performance outcome is claimed.

### Additional held-out evaluation design

Split queries by time when a future deployment must generalize to new catalog and query patterns.
Prevent the same near-duplicate query from leaking into both tuning and holdout sets when that would inflate generalization estimates.
Retain a separate exact-identifier slice because generic natural-language queries can obscure its failure rate.
Retain a mixed-script slice to detect tokenization failures that occur only in multilingual titles.
Retain a sparse-document slice for new items with short or missing descriptions.
Retain a long-document slice to detect unintended length-normalization effects.
Retain a rare-query slice even if its traffic-weighted contribution is small.
Retain an ambiguous-query slice where synonyms or spelling corrections can produce false positives.
Retain an out-of-stock slice to check whether eligibility changes the set of relevant fulfillable items.
Retain permission-denied cases to verify that quality optimization never expands the authorized universe.
Measure each slice with an uncertainty interval and report its sample size.
A wide interval is a reason to collect more evidence, not a reason to declare success.
Use paired query-level comparisons when both systems are evaluated on the same queries.
Pairing controls some query-to-query variability but does not repair biased relevance labels.
Report the distribution of per-query changes, not only a positive mean.
Inspect large regressions individually to find specific failure mechanisms.
Keep a fixed diagnostic set for regression detection and refresh a separate sample to represent current traffic.
An old fixed set supports comparability but may miss emerging products, languages, and query patterns.
Correct for sampling weights when constructing a traffic-level estimate from deliberately oversampled rare slices.
State whether the estimate targets requests, users, sessions, products, or another population.
Changing the unit of analysis can change both the effect estimate and its uncertainty.
For online deployment, monitor the fraction of assigned traffic that actually receives the intended index generation.
An assignment label without confirmed exposure can obscure an implementation or routing failure.
Retain an explicit no-change baseline for distinguishing expected seasonal movement from release effects.
If a rollback happens during an experiment, record the affected exposure intervals rather than treating the cohort as uniform.
Translate offline gains into an online hypothesis, not a guaranteed outcome.
Review the experiment against safety, privacy, access-control, and platform policies before serving treatment traffic.
The hypothetical protocol supplies no measured effect and no claim of universal statistical sufficiency.

Takeaway: Complexity guides hypotheses, while actual latency and storage require workload-specific measurement.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Practical implication | Does not establish | Source |
|---|---|---|---|---|
| Monotone log | ln(x) increases for x>0 | In T1, rarer terms receive greater IDF for fixed N | Relevance is proportional to IDF | Elementary logarithm identity; derivation in Section 4 |
| Saturation | For K>0, B2 increases with f and approaches k1+1 | Extra repeats have diminishing marginal contribution | Stuffing cannot influence ranking | DERIVED in Section 4 |
| Sorted intersection | Advancing the smaller current docID cannot skip a common earlier ID | Merge intersection is correct for sorted lists | Same bound with arbitrary unsorted postings | Stanford IR book [S4] |
| Lucene IDF | The documented expression is B1 in version 9.12.1 | Use it when reconciling that documented component | Complete numerical equivalence of B3 with Lucene | Lucene API [S1] |
| Ranking model | BM25 is a probabilistic-relevance-framework family with practical approximations | Explain design motivation and variant choices | BM25 score is a calibrated probability | Robertson and Zaragoza [S3] |

For sorted intersection, compare the current smallest IDs from both lists.
If the first ID is smaller, it cannot appear later in the second list because that list is sorted and already points to a larger ID.
Advance only the smaller pointer; on equality emit the ID and advance both pointers.
Every step advances a pointer, so finite lists terminate after at most the sum of their lengths in pointer advances.
This argument applies to exact intersection and does not itself justify approximate pruning.

**Primary and official sources**
[S1] Apache Lucene 9.12.1, `BM25Similarity` API: https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html — IDF, parameters, and overlap handling; implementation-specific, not a benchmark.
[S2] scikit-learn, `TfidfTransformer` API: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html — documented smoothed IDF and optional vector normalization; stable documentation is rolling and must be pinned to the deployed version.
[S3] Stephen Robertson and Hugo Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009: https://doi.org/10.1561/1500000019 — primary model derivation and extensions; model family, not proof of a particular deployment.
[S4] Manning, Raghavan, and Schütze, *Introduction to Information Retrieval*, 2008, inverted index processing: https://nlp.stanford.edu/IR-book/html/htmledition/processing-boolean-queries-1.html — textbook algorithm and sorted-postings intersection.
[S5] Elasticsearch, analysis concepts: https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis.html — official analyzer terminology; rolling product documentation, verify the deployed version.
[S6] Unicode Consortium, *Unicode Standard Annex #15: Unicode Normalization Forms*: https://www.unicode.org/reports/tr15/ — normative definitions of normalization forms; use the revision corresponding to the deployed Unicode version.
These sources support the specified methods, not the fictional example's outcomes or any latency claim.
Takeaway: Mathematical properties are conditional on the defined variant and do not certify production relevance.

## 11. Interview / Exam Questions

Scoring for each question: 0 incorrect; 1 partial; 2 correct with assumptions; 3 correct with a defensible tradeoff.
The questions and numbers below are illustrative, not attributed to an employer.

### Level 1 — Definition / Recall

**Q1 — What distinguishes term frequency from document frequency?**
Answer: f(t,d) counts occurrences in a particular field of a document; df(t) counts documents with at least one occurrence.
For `red red shoe`, red has f=2 in that document, but contributes only one to df(red).
Trap: Using the total number of red occurrences as df makes the IDF population inconsistent.
Follow-up: A field-specific index may exclude documents without that field from its statistics.
Source: Section 3.

**Q2 — Why does an inverted index help lexical search?**
Answer: It maps each analyzed term to documents containing it, avoiding a full document-text scan for a selective term.
A frequent term can still have a list almost as large as the corpus.
Trap: An inverted index does not guarantee sublinear cost for every query.
Follow-up: Positions are needed for exact phrase verification.
Source: Sections 2 and 9.

**Q3 — Is a BM25 score a probability?**
Answer: No; it is a dimensionless relative ranking score produced by a chosen variant and corpus statistics.
A score of 2 does not mean a 200% chance of relevance.
A separately fitted and validated calibration model may estimate a probability for a defined target population.
Trap: Positive IDF does not establish probabilistic calibration.
Source: Section 4 and [S3].

### Level 2 — Derivation / Apply

**Q4 — Calculate positive IDF for N=4 and df=2.**
Answer: B1 gives ln(1+(4-2+0.5)/(2+0.5))=ln(2), approximately 0.693147.
The signed variant happens to give zero for these values, demonstrating that the convention matters.
Follow-up: For df=4, positive IDF remains above zero while signed IDF is negative.
Trap: Switching variants after computing the count changes the answer.
Source: Section 4.

**Q5 — Why do frequency gains saturate?**
Answer: For K>0, B2's derivative is (k1+1)*K/(f+K)^2, positive and decreasing as f rises.
Thus each additional occurrence has less marginal effect, although the total response still rises.
At L=A and f=1, the response is exactly one.
Trap: Saturation does not make repeated text irrelevant.
Source: Section 4.

**Q6 — Trace the conjunction of A,B,D and A,B,C.**
Answer: Compare A with A and emit A, then B with B and emit B; C is smaller than D and advances to exhaustion.
The intersection is A,B, and both lists must be sorted under the same document-ID order.
A hash-set intersection is also defensible when its memory and construction costs fit the workload.
Trap: A union would incorrectly include C and D.
Source: Sections 6 and 10.

**Q7 — Which document wins the worked example if both query terms are required?**
Answer: A wins over B, approximately 0.847103 versus 0.836997, under B1–B3 and the stated length convention.
C and D are ineligible under conjunction even though each has a positive disjunctive score.
A comparison before multiplication confirms 2.375 > 2.346667.
Trap: Raw total term counts give a different result and are not BM25.
Source: Section 5.

### Level 3 — Judgement / Design / Debug

**Q8 — A query returns zero results after an analyzer upgrade. What do you check first?**
Answer: Compare query and indexed tokens and analyzer versions, then inspect per-term postings and mandatory filters.
Also check index generation and refresh lag before concluding that tokenization is the root cause.
If tokens match and postings exist, investigate eligibility and intersection instead.
Trap: Rebuilding blindly can conceal the actual issue and cause unnecessary disruption.
Source: F-01 and F-06.

**Q9 — A rare term dominates an irrelevant result. Should IDF be removed?**
Answer: Inspect tokenization, identifier semantics, fields, judged queries, and the exact term contribution before changing IDF.
A field-specific policy or query understanding may be more targeted; validate it on held-out relevance judgments.
Removing rarity weighting globally may damage queries where identifiers are highly discriminative.
Trap: One bad example does not establish a corpus-wide optimum.
Source: Sections 4 and 7.

**Q10 — Candidate recall rises but clicks do not. Is the retrieval change a failure?**
Answer: Not necessarily; the ranker may fail to promote the new candidates or the judgments may not match the online population.
Measure candidate recall, final ranking quality, position effects, and user outcomes separately.
A controlled online experiment with guardrails is required before claiming a causal business effect.
Trap: Raw clicks are affected by exposure and position.
Source: Sections 2 and 8.

**Q11 — A strict latency deadline conflicts with a broad OR query. What do you change?**
Answer: Profile postings, queues, caches, and shard fan-out before choosing a mitigation.
Possible controls include bounded candidate windows or an exact pruning algorithm with valid score upper bounds.
Measure recall and tail latency under representative concurrency and preserve hard authorization filters.
Trap: Limiting results after scoring does not necessarily reduce postings traversal.
Source: Section 9.

**Q12 — How can index deletion be verified?**
Answer: Trace the deletion version from source to each serving replica and cache, then query a synthetic probe through all supported access paths.
Record whether the guarantee is logical invisibility or physical erasure, and check backup and retention policies separately.
If a replica has not consumed the deletion, containment takes precedence over a relevance fix.
Trap: A source database deletion alone does not prove search replicas or caches have removed access.
Source: F-10.

**Q13 — How would you compare a lexical retriever with a hybrid one?**
Answer: Hold the corpus snapshot, eligible universe, candidate budget, query set, judgments, and downstream ranker fixed where possible.
Measure incremental candidate recall and final utility with uncertainty, latency distributions, and cost at realistic load.
A hybrid route is justified only when the measured improvement is worth its complexity under the stated constraints.
Trap: Comparing different authorization filters or candidate counts confounds the result.
Source: Section 7.
Takeaway: A strong answer specifies the scoring contract, checks assumptions, and separates diagnosis from intervention.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Versioned lexical scoring and candidate retrieval for search and ads eligibility |
| Core formula | B3: sum idf+(t) * f*(k1+1)/(f+k1*(1-b+b*L/A)); B1 defines idf+, with distinct query terms |
| Key assumption | Consistent nonempty corpus and positive average field length; compatible analysis and enforced eligibility |
| Complexity | Simple OR visits up to sum(p_i) postings; top-K heap work up to O(U log K), with U distinct candidates |
| Use when | Interpretable exact-term candidates are useful and relevance and latency are evaluated on the target workload |
| Avoid when | Authorization cannot be enforced or required semantic matches have no compatible lexical route |
| Go deeper | [Collection index](INDEX.md); fielded ranking and auction-clearing handoffs are planned, not yet authored |
