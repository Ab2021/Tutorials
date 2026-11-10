---
title: "Phrase and positional retrieval"
topic: "exact phrase and proximity matching over positional inverted indexes"
collection: "Part 03 — Recommendation, Search, Ads and Ranking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Phrase and positional retrieval

All corpus statistics below are constructed teaching values, marked `ILLUSTRATIVE`; every formula is
derived from stated assumptions, and every arithmetic result is recomputed by hand in the text. No
search engine, vendor, or organisation's measured latency, index size, or relevance number is claimed
here. Where a real implementation detail matters (token ordinals, stopword gaps, slop semantics) the
statement is a property of the algorithm, not a benchmark.

## 1. File Header

**Purpose.** Decide, for one concrete search workload, whether a phrase or proximity constraint should be
enforced by positional postings, by a precomputed n-gram index, or by a suffix structure — and predict
the latency, memory and exactness that the decision buys before the query is written.

**Audience.** Search relevance engineers, ranking-infrastructure owners, and interview candidates who
must justify exactness and cost under a fixed latency budget rather than quote a library default.

**Prerequisites, each assumed and not re-taught here.** Inverted-index basics (term dictionary, posting
list, document frequency); the difference between analysis and matching; BM25 term saturation;
asymptotic notation for list intersection; and the habit of attaching units to every number.

**Outcome after working through the sheet.** You can (a) state whether a candidate filter is necessary
or sufficient for a phrase match, (b) size position storage for a corpus from token counts and
compression assumptions, (c) choose an anchor term and predict probe counts, (d) explain why a
doc-level filter that returns 5,600 documents can still yield 1,850 true matches, and (e) reject a
slop change that silently converts an exact constraint into a bag-of-words constraint.

**Exclusions.** This sheet does not cover learning-to-rank feature weights, neural dense retrieval, or
query relaxation policy. It does not claim any engine's published throughput. It does not replace the
operational authority of whoever owns the cluster: a number here sizes a plan, it never authorises a
deployment.

**Notation convention, used consistently below.** Every symbol is introduced as meaning / units /
valid range. Position means the integer ordinal of a token inside one field of one document, counted
from 0 after analysis. A posting is `(docID, positions[])`. Lists are sorted ascending by docID and,
inside a document, by position. Arithmetic is decimal, not binary, unless bytes are explicitly
converted with 1024.

**Evidence boundary.** Statistics such as `df` and phrase frequency are constructed to be internally
consistent; they are not crawled from a real corpus. Arithmetic is verified twice by independent
routes. A reader who needs a production number must measure `df` for the phrase itself, because the
sheet shows that independence models under-predict phrase frequency by four to five orders of
magnitude.

**How to use the sheet.** Read sections 2 and 3 for vocabulary, section 4 for the cost and exactness
derivations, section 5 for one fully worked sizing exercise, section 7 as a selection table at design
review, section 8 as a debugging checklist, and section 12 as the card to keep beside a terminal.

## 2. Concept Map / Mental Model

The mental model that prevents most phrase-retrieval mistakes is a two-stage funnel whose second stage
is mandatory, not an optimisation. Stage one intersects document identifiers and is *necessary*.
Stage two checks token adjacency inside each surviving document and is *sufficient*. Every failure
mode in section 8 is either stage one being skipped (you get a bag-of-words result and call it a
phrase), or stage two being fed inconsistent positions (you get a silent, unfalsifiable mismatch).

```text
query text
   |
   v
[analysis]  tokenise -> normalise -> optional stopword removal (position gaps preserved)
   |
   v
[term lookup]  term -> posting list (docID, positions[])     <- stage-one data
   |
   v
[stage one]  intersect docIDs, rarest term as anchor          <- necessary condition
   |
   v
[stage two]  for each candidate doc: test pos(b)=pos(a)+1 ... <- sufficient condition
   |
   v
[scoring]    treat matched phrase as a pseudo-term with its own tf and df
   |
   v
[ranking]    merge with other clauses under a top-k protocol (WAND / block-max)
```

Three tensions govern every design decision on this topic, and they cannot all be optimised at once.

- **Exactness versus memory.** Adjacency is only checkable if positions are stored. Storing positions
  multiplies index size by roughly the ratio of stored positions to stored postings entries; for a
  corpus with 120 tokens per document the multiplier is large because every token produces a position.
- **Latency versus anchor choice.** Intersection cost is governed by the *rarest* term only while the
  rarest term also has few positions per document. A term that is rare in document frequency but
  repeated within documents inverts the usual cost model.
- **Precision versus recall at query-construction time.** A slop value above zero trades an ordered
  adjacency constraint for a proximity constraint. That trade is a product decision about what the
  user meant, not a tuning knob, and it must be documented as such.

A second useful mental picture is that positions are *relative information*. Nothing in the matcher
needs the absolute ordinal 87; it needs the difference between two ordinals. This is why delta
encoding dominates position compression, and why a document's position list length can be derived from
its token count rather than stored separately.

The final mental note: phrase matching is decoupled from phrase scoring. Matching answers "does this
document contain the string". Scoring answers "how much does the match matter". Conflating them
produces the classic bug where a relevance tweak (changing `df` to the phrase's own document
frequency) is blamed for a recall regression (which is caused by stopword gaps being compacted).

## 3. Foundational Definitions

**Token** — the atomic unit emitted by the analyzer for one field of one document; unit: dimensionless
string; range: any non-empty normalised substring. Tokens are what get ordinals; characters do not.

**Term** — the dictionary key a token maps to after normalisation, stemming and synonym expansion; a
term may be produced by several distinct tokens, which is exactly why exact-string intuition fails on
stemmed indexes.

**Position (token ordinal)** — the integer index of a token within one field of one document, counted
from 0 after analysis; unit: token slots; range: 0 to the field's token count minus one, subject to the
storage integer width. Positions are per-field, so a single document can carry several coordinate
systems.

**Position increment gap** — the difference between consecutive stored positions of one term inside one
document; unit: token slots; range: at least 1 for distinct tokens, and greater than 1 whenever an
intervening token was removed or was a different term. A gap above 1 is information, not waste.

**Posting list** — the sorted sequence of `(docID, positions[])` pairs for one term; unit: entries;
range: empty to the corpus document count. Sorting is the invariant that makes every intersection
algorithm in section 7 correct.

**Document frequency `df`** — the count of documents whose posting list contains a given term; unit:
documents; range: 0 to `N`. `df` is a document-level statistic and says nothing about how often the
term repeats inside a document.

**Phrase** — an ordered sequence of `k` terms that must occupy `k` consecutive positions in the same
field of the same document. The definition is positional, not string-based: two texts that look
identical after Unicode normalisation can match or fail depending on analyzer output.

**Span / proximity query** — a constraint that the terms occur within a window of `w` positions,
either ordered or unordered; unit: token slots; range: `w >= k`. Slop is the usual query-language
surface for the same concept, with engine-specific accounting for reordering cost.

**Anchor term** — the term whose posting list drives iteration during intersection, normally the one
with the fewest entries; unit: a term; chosen at query time from dictionary statistics. The anchor
determines the number of outer iterations, hence most of the cost.

**Candidate document** — a document that survives stage one (contains every query term) but has not yet
been positionally verified; unit: documents. The ratio of candidates to true matches is the false
candidate rate, and it is a direct multiplier on stage-two work.

**Pseudo-term** — a synthetic dictionary entry standing for a whole phrase, with its own `tf` (phrase
occurrences in the document) and its own `df` (documents containing the phrase). Needed because BM25's
saturation behaves differently for a phrase than for its rarest constituent.

**Slop** — the maximum permitted total displacement of query terms from their target positions when
reordering and intervening tokens are allowed; unit: position moves; range: 0 upward. At 0 the
constraint is exact ordered adjacency, and every increment weakens it monotonically.

**Doc-at-a-time evaluation** — scoring one document completely before moving to the next, which is the
only order compatible with positional verification; contrast with term-at-a-time accumulation, which
is faster for pure disjunctions and structurally unable to check adjacency.

**Skip pointer** — a stored shortcut inside a posting list that jumps ahead by a fixed or estimated
number of entries; unit: entry offset; used to convert linear scans into galloping searches. Skips
cost space proportional to list length divided by skip interval.

**Block-max metadata** — the maximum impact score within a block of postings, used by top-k protocols
to prove that a block cannot enter the result set and therefore need not be decoded; unit: score;
valid only while the scoring function's contributions are monotone within the block.

**Snapshot** — an immutable view of the index at one generation, over which all lists of one query are
read; unit: a version identifier. Mixing generations inside a query produces phantom matches, because
a document's positions may have been rewritten between reads.

**Analyzer version** — the identifier of the exact tokenisation, normalisation, stemming and stopword
configuration used to build positions; unit: an opaque tag stored with the segment. Two analyzers
produce incomparable coordinate systems even when they agree on most tokens.

**Position width overflow** — the condition where a field's token count exceeds the integer width used
for positions, so later tokens wrap or clamp; unit: tokens; range: engine-defined, historically 65,535
for a 16-bit unsigned field. Overflow removes the tail of a document from phrase reach silently.

**Exactness (of a matcher)** — the property that the returned set equals the set defined by the formal
constraint, independent of index layout, sharding, or list order; unit: boolean. Exactness is provable
by construction, not by sampling; sampling only estimates recall loss.

## 4. Core Formulas with Full Derivation

Every block below follows one anatomy: FORMULA, WHERE (symbols, units, ranges), CONSTRAINT (when the
identity is valid), NUMERICAL EXAMPLE (arithmetic shown), INTERPRETATION (what a practitioner does with
it). Section 4.4 states where each identity breaks, because a formula applied outside its constraint is
the most common source of wrong capacity plans.

### 4.1 Base case

The base case is the two-term ordered phrase. Its correctness condition is the definition of adjacency,
and its cost condition is the size of the anchor list plus the number of positional probes.

```text
FORMULA [P-01] adjacency test for phrase (t1, t2)
WHERE: P1, P2 are ascending position arrays of t1, t2 inside one document [token ordinals];
       a match is a pair (x, y) with y = x + 1; count of matches m2 = |{x in P1 : x+1 in P2}|.
CONSTRAINT: P1 and P2 must come from the same analyzer version, the same field, and the same
       snapshot; positions must be absolute ordinals, not per-segment renumberings.
NUMERICAL EXAMPLE: P1 = [4, 19, 51]; P2 = [5, 20, 60]. Check x=4 -> 5 present, hit; x=19 -> 20
       present, hit; x=51 -> 52 absent, miss. m2 = 2, and 2 of 3 anchors matched.
INTERPRETATION: m2 is the phrase's term frequency inside the document, and it is the number BM25
       saturates when the phrase is scored as a pseudo-term.
```

The cost of that test is not the number of pairs; it is the number of probes. For each anchor position
the matcher must locate `x+1` in `P2`. Because `P2` is sorted, the lookup is a bounded search.

```text
FORMULA [P-02] probe count for galloping search of one element in a sorted list
WHERE: n = length of the searched list [entries]; j = index of the located element, 0-based.
       Cost = 2*ceil(log2(j+1)) + 1 probes for exponential advance, then a binary narrowing of the
       last interval.
CONSTRAINT: applies when j is known only by searching; if the caller already knows the neighbourhood,
       a single comparison suffices and the bound overstates cost.
NUMERICAL EXAMPLE: n = 1,200,000, target near index 900,000. log2(900,001) = 19.78, ceil = 20,
       probes = 2*20 + 1 = 41 worst case, against 900,000 for a linear scan.
INTERPRETATION: the logarithmic bound is what makes anchor-driven iteration viable; it also means that
       halving list length saves only one probe, so shortening lists is a weaker lever than choosing a
       better anchor.
```

The two formulas compose into the base-case cost model. Iterations are governed by the anchor's
`positions` count, probes per iteration by the logarithm of the second list, and verification by the
surviving candidates.

```text
FORMULA [P-03] base-case stage-one plus stage-two work
WHERE: C1 = |P_anchor| * (k-1) * q, with q = probes per list [probes]; C2 = D_cand * p_bar, with
       p_bar = mean positional probes per candidate [probes]; D_cand = candidate documents [docs].
CONSTRAINT: valid for disjoint-term phrases where the anchor is the rarest term by position count and
       all lists are read from one snapshot; degrades when the anchor is repeated inside documents.
NUMERICAL EXAMPLE: anchor positions 94,800, k = 3, q = 4 -> C1 = 94,800 * 2 * 4 = 758,400 probes.
       Candidates 5,600, p_bar = 1.2 -> C2 = 6,720 probes. Stage one is 99.1% of probe volume.
INTERPRETATION: optimisation effort belongs in stage one for long lists and in stage two only when the
       candidate set is huge relative to the anchor, which happens with weak anchors.
```

### 4.2 General case

For `k` terms the adjacency constraint becomes: there exists a position `p` such that term `i` occupies
`p + i` for every `i`. The general test is a chain of `k-1` relative lookups per anchor position, but
the order of lookups matters for cost, not for correctness.

```text
FORMULA [P-04] k-term phrase match set
WHERE: for anchor a with positions A, matched count m_k = |{p in A : for all i != a, p + off_i in P_i}|,
       off_i = i - a is the relative offset of term i from the anchor [token slots].
CONSTRAINT: offsets must be computed from the query, not from the document; a term that appears both
       before and after the anchor is covered by exactly one offset.
NUMERICAL EXAMPLE: query (data, quality, pipeline), anchor pipeline at index 2. offsets: data -> -2,
       quality -> -1. An anchor position p = 37 requires 35 in P_data and 36 in P_quality.
INTERPRETATION: the anchor need not be the first term; correctness is offset-based and independent of
       which term drives iteration.
```

The candidate set after stage one is bounded by the smallest document frequency, and the number of
documents containing all `k` terms is what stage two must scan. Independence gives an estimate that is
almost always wrong in the pessimistic direction for topical phrases.

```text
FORMULA [P-05] doc-level candidate estimate under independence
WHERE: D_cand_hat = N * product_i (df_i / N) = product_i df_i / N^(k-1); N = corpus documents [docs];
       df_i = document frequency of term i [docs].
CONSTRAINT: requires term occurrences to be independent across documents, which fails whenever the
       phrase names a topic, a product, or a boilerplate sentence.
NUMERICAL EXAMPLE: N = 10,000,000; df = 1,200,000, 300,000, 90,000. D_cand_hat = 1.2e6 * 3.0e5 *
       9.0e4 / 1.0e14 = 3.24e16 / 1.0e14 = 324 documents. Constructed observed value: 5,600.
INTERPRETATION: the estimate under-predicts by 17.3x here; using it to size a scan budget under-
       provisions stage two by the same factor.
```

### 4.3 Variants

Variant one is proximity instead of adjacency. The constraint becomes a window, and cost grows with the
number of position pairs rather than the number of anchor positions, because a wide window permits many
pairings per anchor.

```text
FORMULA [P-06] span test cost for unordered proximity
WHERE: W = window width [token slots]; with per-term position counts r_i inside a document, the number
       of candidate pairs for two terms is at most r_1 * r_2; a sorted merge makes it
       O(r_1 + r_2) comparisons by advancing the lagging pointer.
CONSTRAINT: the merge bound holds because both arrays are sorted; the quadratic count is an upper bound
       on pairs, not on work, and using it to estimate latency overstates cost by the same factor.
NUMERICAL EXAMPLE: r_1 = 3, r_2 = 4 -> pair upper bound 12, merge comparisons 7, and with W = 8 only
       pairs whose gap is at most 8 count, here 3.
INTERPRETATION: implement proximity by a merge over sorted arrays and bound the work by lengths, never
       by the product of lengths.
```

Variant two is a repeated term inside the phrase, which breaks the disjoint-list assumption because both
occurrences read the same position array.

```text
FORMULA [P-07] repeated-term constraint
WHERE: for phrase (t, t) the requirement is p and p+1 both in P_t; a match exists iff two consecutive
       integers appear in one array, so the count is |{p in P_t : p+1 in P_t}|.
CONSTRAINT: the two occurrences must be distinct array entries; a naive set-membership test over the
       same array returns true for every p, producing phantom matches on a single occurrence.
NUMERICAL EXAMPLE: P_t = [7, 8, 8, 21] after duplicate collapse -> matches at p=7 only (7,8); the
       repeated 8 cannot pair with itself. Result 1 match.
INTERPRETATION: duplicate positions arise from synonym expansion at the same offset; deduplicate per
       (term, position) before any repeated-term test.
```

Variant three is multi-token expansion at one position, usually produced by synonym or decompounding
rules. Several tokens share one ordinal, so a phrase can succeed through different expansion paths and
the match count can exceed the occurrence count.

```text
FORMULA [P-08] same-position expansion multiplicity
WHERE: if term t expands to e tokens at each occurrence, the position array for expansion token j is
       P_j; a phrase containing an expansion slot matches if any j satisfies the offset test, so the
       effective number of paths is e per slot and e^k per phrase of k expansion slots.
CONSTRAINT: paths must be enumerated, not assumed disjoint; two expansions may share a token and the
       same position, double-counting matches unless deduplicated by resulting offset.
NUMERICAL EXAMPLE: e = 3 expansions, k = 2 expansion slots -> 9 paths; if 3 paths collapse to the same
       (document, offset) the deduplicated count is 6, not 9.
INTERPRETATION: deduplicate on (docID, offset) after expansion, or phrase term frequency inflates and
       BM25 over-weights the phrase.
```

Variant four is a precomputed n-gram index, which removes position storage for the n-grams themselves
but pays in dictionary size. The trade is exact and computable.

```text
FORMULA [P-09] n-gram dictionary growth
WHERE: V_n = number of distinct n-grams actually occurring, bounded above by min(V^n, total windows)
       with V = vocabulary size [terms]; storage = V_n * (id bytes + posting overhead).
CONSTRAINT: the upper bound V^n is reachable only for small V and short documents; real corpora sit far
       below it because windows repeat, so always measure V_n before committing.
NUMERICAL EXAMPLE: V = 200,000 and n = 2 -> naive bound 4.0e10, unaffordable; a corpus with 1.2e9
       windows and heavy repetition might realise V_2 = 3.1e7 distinct bigrams, which is storable.
INTERPRETATION: bigram dictionaries are sized by observed distinct windows, never by V^n; the naive
       bound is a warning, not an estimate.
```

### 4.4 Limits and counterexamples

The independence model in [P-05] fails precisely where phrase queries are most useful. A phrase such as
"data quality pipeline" names a topic, so documents that contain one term are far more likely to
contain the others than chance allows. The counterexample is quantitative: the model predicts 324
candidates and the constructed corpus yields 5,600, a lift of 17.3. Any capacity plan built on the
model would under-allocate by that factor and would also mis-order terms by "expected selectivity".

The positional-independence model fails harder. Treating tokens as independent draws with token shares
`s_i` gives an expected phrase count per document of `(L-k+1) * product s_i`, which for token shares
0.0012, 0.000275 and 0.000079 and `L = 120` yields 0.0308 occurrences across a ten-million-document
corpus, against a constructed observed document frequency of 1,850. The failure factor is about 60,000.
The lesson is not that the model is imprecise; it is that the model is unusable for phrase frequency,
and phrase `df` must be measured by the matcher itself.

The galloping bound in [P-02] also has a floor. For lists shorter than roughly 30 entries the
exponential advance costs more than a linear scan from the current cursor, because the first probes are
pure overhead when the target is near. Engines therefore switch strategies by list length; a single
strategy is wrong at one end of the distribution.

The Elias-Fano space identity quoted in section 10 assumes `U/n >= 1`, that is, at least one unused slot
per stored element. For a term that occupies a large fraction of a short field, positions are dense and
the low-bits part dominates, so the bound degrades to about `n * log2(U)` bits, which is worse than
delta plus variable-byte for small `n`.

**Validity condition for the whole of section 4.** Every formula above presumes that all position arrays
for one query come from one snapshot, one field, and one analyzer version, and that positions are
absolute ordinals rather than per-document renumberings. When any of those three presumptions is false,
the formulas still compute a number, and the number no longer describes the query that ran.

## 5. Worked Numerical Example

The exercise is a product-catalogue search where users type `"wireless charging case"` and expect the
three words adjacent. All corpus figures are `ILLUSTRATIVE`; the arithmetic is the deliverable.

**Fixed inputs.**

- `N` = 10,000,000 documents in the catalogue.
- mean field length `L_bar` = 120 tokens; total tokens `T` = `N * L_bar` = 10,000,000 * 120 = 1.2e9.
- mean occurrences per (term, document) pair = 1.2, so distinct postings entries ≈ `1.2e9 / 1.2` = 1.0e9.
- docID delta encoded at 1.5 bytes average; position delta encoded at 1.0 byte average uncompressed
  widths are 4 bytes each, used only as the pessimistic comparison.
- document frequencies: `wireless` 1,200,000; `charging` 380,000; `case` 2,900,000.
- per-document mean occurrence counts: 1.4, 1.1, 1.6 respectively.

**Step 1 — total stored positions versus stored postings.**

```text
positions_raw = T * 4 bytes = 1.2e9 * 4 = 4.8e9 bytes = 4.47 GiB
postings_raw  = entries * 4 bytes = 1.0e9 * 4 = 4.0e9 bytes = 3.72 GiB
ratio_raw     = 4.8e9 / 4.0e9 = 1.20
```

Re-checked independently: per document there are 120 position values and 100 posting entries, so the
ratio is 120/100 = 1.20 with equal widths. Consistent.

**Step 2 — the same ratio after delta encoding.**

```text
positions_cmp = T * 1.0 byte  = 1.2e9 bytes = 1.12 GiB
postings_cmp  = entries * 1.5 = 1.5e9 bytes = 1.40 GiB
total_cmp     = 2.7e9 bytes   = 2.51 GiB
overhead_vs_doc_only = 2.7 / 1.5 = 1.80x
```

Two different answers, 1.20 and 1.80, describe the same design. The first compares like with like and
understates the real cost, because docID deltas are larger than 1 byte and position deltas are not
uniformly 1 byte. The second is the honest planning number: **storing positions costs about 1.8x the
doc-only index**, or roughly 1.12 GiB of additional disk for this corpus. The gap between 1.20 and
1.80 is entirely a units artefact of assuming equal widths.

**Step 3 — anchor selection changes total work, not total bytes.** Position counts decide the anchor:

```text
pos_total(wireless) = 1,200,000 * 1.4 = 1,680,000
pos_total(charging) =   380,000 * 1.1 =   418,000
pos_total(case)     = 2,900,000 * 1.6 = 4,640,000
sum of lists                          = 6,738,000 position entries
```

`charging` is the anchor by both measures, so anchor choice agrees here. It would not agree if
`charging` had 6.0 occurrences per document: then 380,000 * 6.0 = 2,280,000, ahead of `wireless`
(1,680,000), and the "rarest `df` wins" rule would pick the worse anchor.

**Step 4 — two cost models for the same intersection.**

```text
random-access model [P-02/P-03]:
  q(wireless) = 2*ceil(log2(1,680,001)) + 1 = 2*21 + 1 = 43 probes
  q(case)     = 2*ceil(log2(4,640,001)) + 1 = 2*23 + 1 = 47 probes
  C1 = 418,000 * (43 + 47) = 37,620,000 probes
cursor-merge model (cursors advance, never re-seek):
  C1' = 1,680,000 + 418,000 + 4,640,000 = 6,738,000 visited entries
ratio = 37,620,000 / 6,738,000 = 5.58
```

Both are legitimate. The random-access figure applies when each candidate document is tested
independently, for example when a filter supplies document IDs out of order; the merge figure applies
when the matcher walks all lists forward in lockstep. At 100 million probe operations per second on one
core the two give 376 ms and 67 ms, which is the difference between breaching and meeting a 200 ms
budget. Nothing about the query changed, only the execution order.

**Step 5 — stage two on the survivors.** Constructed candidate count after doc intersection is 5,600,
and the constructed count of documents containing the true phrase is 1,850.

```text
anchor probes   = 5,600 candidate docs * 1.1 anchor positions = 6,160
lookups         = 6,160 * 2 remaining terms = 12,320
false_cand_rate = 1 - 1,850/5,600 = 1 - 0.33036 = 0.66964
filter precision = 1,850/5,600 = 0.33036
total_probes    = 6,738,000 + 12,320 = 6,750,320
stage_two_share = 12,320/6,750,320 = 0.182%
```

Stage two is 0.18% of probe volume but 100% of the exactness guarantee. That asymmetry is the reason to
never "skip stage two for speed": removing it saves almost nothing and returns unusable results.

**Step 6 — scoring the phrase as a pseudo-term.** With `N` = 1.0e7, `df` = 1,850, BM25 `k1` = 1.2,
`b` = 0.75, `dl` = `L_bar` = 120:

```text
idf_phrase = ln(1 + (N - df + 0.5)/(df + 0.5))
           = ln(1 + (10,000,000 - 1,850 + 0.5)/1,850.5)
           = ln(1 + 5,402.86) = ln(5,403.86) = 8.5949
tf_component(tf=1) = 1*(1.2+1)/(1 + 1.2*(1-0.75+0.75*120/120)) = 2.2/(1+1.2) = 1.0000
score_1 = 8.5949 * 1.0 = 8.5949
tf_component(tf=2) = 2*2.2/(2+1.2) = 4.4/3.2 = 1.3750
score_2 = 8.5949 * 1.3750 = 11.818
increment from second occurrence = 11.818/8.5949 - 1 = 37.5%
```

Now score the rarest constituent alone, `charging` with `df` = 380,000:

```text
idf_term = ln(1 + (10,000,000 - 380,000 + 0.5)/380,000.5)
         = ln(1 + 25.3158) = ln(26.3158) = 3.2702
ratio_phrase_to_term = 8.5949/3.2702 = 2.628
```

A document that satisfies the phrase outranks one that merely mentions `charging` by a factor of 2.63 on
the idf component alone. That is the intended effect of an exact constraint, and it is also why an
accidental slop relaxation degrades rankings abruptly rather than gradually.

**Step 7 — sensitivity, three independent knobs.**

```text
(a) mean length 40 instead of 120: T' = 4.0e8; positions_cmp = 0.373 GiB;
    entries' = 4.0e8/1.2 = 3.33e8; postings_cmp = 0.466 GiB; overhead = 0.839/0.466 = 1.80x (unchanged)
(b) 1.6 occurrences per pair instead of 1.2: entries'' = 1.2e9/1.6 = 7.5e8;
    postings_cmp = 1.048 GiB; overhead = 2.166/1.048 = 2.067x (worse, fewer postings per position)
(c) position delta 1.4 bytes instead of 1.0: positions_cmp = 1.565 GiB; overhead = 2.962/1.397 = 2.12x
```

The ratio is invariant to document length (a) and moves with the occurrence-density assumption (b) and
with compression quality (c). So the honest way to quote the cost is "positions cost 1.8x to 2.1x the
doc-only index for this workload", with the range tied to measured occurrence density, not to `L_bar`.

**Independent re-check of steps 1–2 by a second route.** Express everything per document: 120 positions
at 1 byte plus 100 postings at 1.5 bytes = 120 + 150 = 270 bytes per document, of which 120 bytes are
positions; dropping positions therefore reduces the index from 270 to 150 bytes, a factor of
270/150 = 1.80x. Per document the same figures appear from corpus totals: 1.2e9 positions / 10,000,000
documents = 120 per document, 1.0e9 entries / 10,000,000 = 100 per document. Consistent. Note that the
positions-versus-postings ratio is 120/150 = 0.80, that is positions add 80% on top of the doc-only
index; quoting "1.8x" and quoting "80% overhead" describe the same design and a reviewer must state
which basis is meant.

## 6. Visual / Geometric Intuition

**Position arrays as intervals on a number line.** Picture one document's field as a ruler marked 0 to
119. Each term's postings are pins on that ruler. A phrase is a rigid template of pins: for
`(wireless, charging, case)` the template is `[p, p+1, p+2]`. Matching means sliding the template along
the ruler and checking that every slot is occupied by the right term's pin.

```text
positions of "wireless" : 5, 41, 88
positions of "charging" : 6, 42, 60, 89
positions of "case"     : 7, 43, 61, 90

template slide at p=5 : needs 5,6,7   -> occupied (wireless5, charging6, case7)      MATCH
template slide at p=41: needs 41,42,43-> occupied (wireless41, charging42, case43)    MATCH
template slide at p=88: needs 88,89,90-> occupied (wireless88, charging89, case90)    MATCH
charging at 60, case at 61: no wireless at 59 or 61 -> no match
```

Three matches in one document, so the pseudo-term `tf` is 3. Reading the template as rigid is what
distinguishes a phrase query from a three-term AND, which would have matched this document once and
counted it as one undifferentiated hit.

**Why ordered adjacency is cheaper than it looks.** The template never has to slide to every offset; it
only slides to offsets where the anchor has a pin. That is the geometric content of anchor selection:
the number of slides equals the number of anchor pins, not the document length.

```text
slots 0                                                        119
      |----------------------------------------------------------|
      ^ anchor pins at 6, 42, 60, 89  -> 4 slides, not 120
```

**Wide windows turn a template into a rubber band.** With `W` = 8, the template for two terms permits
any pair whose distance is at most 8, so the number of admissible alignments grows approximately
linearly in `W` early on and quadratically once `W` exceeds the typical inter-term distance, because each
anchor pin gains multiple partners.

```text
W = 1 : pin pairs (x, x+1)                  -> 1 partner per pin
W = 4 : pin pairs with gap in 1..4          -> up to 4 partners per pin
W = 40: neighbouring occurrences interleave -> partners per pin approach the local density
```

A useful field test: if doubling `W` more than doubles your top-10 churn on a fixed query set, `W` is
past the point where the constraint still means "phrase".

**Skip pointers as a coarse second ruler.** A posting list of 4,640,000 entries cannot be walked one
entry at a time when the caller jumps across documents. Skips add marks every 128 or 256 entries so the
matcher can hop, then scan the residual.

```text
list  : [d1 ... d128][d129 ... d256][d257 ... d384] ...
skips :      ^skip to 129   ^skip to 257
seek target d300: hop to 257, scan 43 entries instead of 299
```

The residual scan is why skip intervals are never made huge: halving the interval doubles skip storage
and halves the residual, a trade that has an optimum near the point where skip metadata costs as much as
the residual scan saves.

**The stopword gap picture.** When `the` is removed from "the wireless charging case", the analyzer
emits `wireless@1, charging@2, case@3` and leaves position 0 unused. Visually the ruler has a hole.

```text
text      :  the  wireless  charging  case
position  :   0      1         2       3
stored    :   -    wireless  charging  case      <- hole at 0 preserves offset arithmetic
compacted : wireless charging case               <- offsets shift, "the X" now matches "X"
```

Holes are cheap: at most one unused slot per removed token. Compaction is the tempting optimisation that
breaks the semantics, because after compaction the query `"wireless charging"` matches a document that
says "wireless the charging".

**Position width as a vertical cliff.** Ordinals are stored in a fixed width. A field with 70,000 tokens
overflows a 16-bit unsigned field at 65,536. The ruler simply stops existing past the cliff, and no
error surfaces at query time.

```text
ordinal  0 ................................ 65,535 | 65,536 ... 69,999
stored   [============ representable ============] [ wrapped / clamped ]
query    "phrase at token 66,000" -> no match, no warning, no log line
```

The visual is worth remembering because the failure is silently one-sided: recall drops only for very
long documents, which are exactly the reference pages and legal filings where phrase search matters
most.

**Two coordinate systems in one document.** Each field has its own ruler starting at 0. A query that
spans fields cannot be evaluated by subtracting ordinals, because `title:5` and `body:5` are different
locations.

```text
title : |0 ... 12|
body  : |0 ............................................. 3,400|
cross-field query "wireless charging" spanning title->body: undefined offset, needs field-scoped run
```

This is why phrase queries are field-scoped by default and why "phrase across title and body" is a
product decision with an explicit implementation (concatenate with a sentinel), not a natural extension
of the offset arithmetic.

## 7. Comparison / Selection Table

Choose one mechanism per phrase clause; mixing mechanisms within one clause creates ambiguity about
which constraint actually ran.

| Mechanism | Exactness for phrases | Extra space | Query-time cost | Build cost | Choose when |
|---|---|---|---|---|---|
| Doc-level AND only | none; false positives at every adjacency | 0 | lowest | lowest | never for a phrase; only for a relaxed AND the user accepts |
| Positional postings, adjacency | exact | ~1.8x doc-only index (`ILLUSTRATIVE`, §5) | merge over lists; anchor-driven | positions written once per token | default for exact phrase search |
| Positional postings, window `W` | exact for the window, weaker semantics | same as adjacency | grows with partners per pin | same | when the query language exposes slop deliberately |
| Bigram index | exact for 2-token phrases only | dictionary grows to observed distinct bigrams | one lookup, no verification | heavy: enumerate all windows | short, high-QPS phrase set with a fixed schema |
| Trigram index | exact for 3-token phrases only | larger dictionary than bigrams | one lookup | heaviest | fixed 3-token phrase set, e.g. codes or names |
| Suffix array / FM-index | exact for arbitrary substrings | `O(T log T)` bits for the array, or compressed | `O(|pattern|)` search plus locate cost | full rebuild or incremental merge | long-pattern search, bioinformatics-style text |
| Stored field + post-filter | exact but unbounded | full text stored | proportional to candidates times text length | none | low-QPS admin/debug path only |

Reading the table as a decision procedure:

1. **Is the constraint truly a phrase?** If the product owner will accept any document containing all
   terms, do not build positional infrastructure; use AND and rename the feature so the UI does not lie.
2. **Is the phrase set fixed and short?** A bigram or trigram index beats positional matching on query
   latency but loses the ability to answer arbitrary phrases and inflates the dictionary. Only take it
   when the phrase inventory is enumerable.
3. **Is the pattern longer than a few tokens?** Positional posting lookup stays `O(sum of list lengths)`
   regardless of `k`, while a suffix structure answers long patterns in time proportional to the pattern.
   That crossover is the one case where suffix structures win on pure query cost.
4. **Is the corpus write-heavy?** Positions are written with the document, so positional indexing adds
   encode work at ingest but nothing at query time. Suffix structures are notoriously hostile to
   incremental updates, which usually disqualifies them for catalogue search.
5. **Is QPS high and the phrase vocabulary narrow?** Precomputed n-grams amortise verification cost
   across queries. Otherwise the positional path is the only general answer.

Two selection errors appear repeatedly in design reviews, and both are decidable from the table.

- **Choosing a bigram index to "avoid storing positions".** The dictionary then carries `V_2` entries
  with postings, and for a corpus with 1.2e9 windows the observed distinct bigram count dominates
  storage. The positional index stores 1 position per token; the bigram index stores roughly 1 posting
  per window, which is the same order, plus a much larger dictionary. The supposed saving usually
  inverts.
- **Choosing suffix arrays for "exactness".** Exactness is available from positional postings too; the
  suffix array buys long-pattern asymptotics, not correctness. Paying rebuild cost for a property you
  already have is the most expensive mistake in this table.

A third, subtler selection criterion is the **query mix**. If 90% of traffic is single-term and 2% is
phrase, the positional index's space cost must be justified by 2% of traffic, or the phrase feature must
be scoped to a subset of fields. Scoping positions to `title` and `brand` only, and leaving `body`
positionless, is a legitimate design that cuts the space multiplier to the fraction of tokens in those
fields.

```text
ILLUSTRATIVE scoping arithmetic:
  total tokens 1.2e9; title+brand share 8% -> positional tokens 9.6e7
  positions_cmp = 9.6e7 bytes = 0.089 GiB
  doc-only index 1.40 GiB -> overhead = 1.489/1.40 = 1.064x, i.e. 6.4% instead of 80%
  cost: phrase queries on body fields must return "unsupported" rather than silently degrade
```

That last line is the whole discipline of this sheet in one sentence: an unsupported answer is a correct
answer, and a silently degraded one is not.

## 8. Failure Modes & Edge Cases

Each entry gives the observable symptom, the competing causes that produce that same symptom, the check
that discriminates between them, and the correction. The discipline matters because every one of these
symptoms has at least two plausible causes, and fixing the wrong one is how indexes get rebuilt for no
reason.

### 8.1 Phrase query returns documents where the words are far apart

**Symptom.** A user reports that searching `"wireless charging case"` returns a page where the three
words appear in three different paragraphs, or in a different order.

**Competing causes.**
(a) The clause is being executed as a doc-level AND because the query builder emitted `should` clauses
instead of a phrase clause; (b) positions were never indexed for that field, so the engine degraded to
AND silently; (c) a slop value above zero was inherited from a template or a default analyzer config;
(d) the field was indexed with a shingle filter that rewrote the query into an OR of shingles.

**Discriminating checks.** Ask the engine to explain the clause and look for a positional operator: if
the explanation shows a doc-level intersection with no position test, cause (a) or (b) applies. Then
inspect the field mapping for `positions: false`; that confirms (b). If positions are on and the clause
is positional, dump the parsed query: a nonzero slop confirms (c) and an OR of n-gram clauses confirms
(d). The three checks are ordered by cost — explanation first, mapping second, parsed query third.

**Correction.** Emit an explicit phrase clause, enable positions on exactly the fields where phrase
semantics are promised, and remove inherited defaults by making slop an explicit required parameter of
the query builder rather than an optional one with a fallback.

**Prevention.** A regression test that runs a fixed phrase whose terms all exist in a document but are
not adjacent, asserting zero hits. That single test catches (a), (b) and (d) because all three make the
false document match.

### 8.2 Phrase query returns nothing although the phrase is visibly present

**Symptom.** A document containing the exact string is not returned; manual inspection of the page text
confirms the phrase.

**Competing causes.**
(a) Stopword removal compacted positions and thereby shifted offsets, so a query that includes a
stopword cannot align; (b) stemming mapped one token to a different term than the query's, so one slot of
the template is never occupied; (c) the phrase spans a field boundary (title to body) and ordinals are
per-field; (d) the document sits in a segment built with a different analyzer version, so its coordinate
system is incomparable; (e) position width overflow truncated the field beyond 65,535 tokens.

**Discriminating checks.** Re-run the query with each slot replaced by a single term, then with
progressively shorter sub-phrases: the longest failing sub-phrase localises the offending slot. If the
offending slot is a stopword, (a) applies; if it is an inflected word whose base form matches, (b)
applies. If all sub-phrases succeed individually inside one field but the full phrase fails across
fields, (c) applies. Compare the segment's analyzer tag against the query analyzer tag for (d). If the
document's token count exceeds the stored width, (e) applies; that check needs the field length, which is
usually available in the doc-values or stored field.

**Correction.** Preserve position increments for removed tokens instead of compacting; align the query
analyzer with the index analyzer; scope cross-field phrases to an explicit multi-field concatenation with
a sentinel; rebuild segments that carry a stale analyzer tag; widen position width or split oversized
fields at ingest.

**Prevention.** Store the analyzer tag as a queryable metadata field and assert equality at query time,
failing loudly rather than returning an empty result that looks like "no match".

### 8.3 Phrase scoring is wildly out of proportion to its selectivity

**Symptom.** A rare phrase dominates the ranking, or conversely a highly selective phrase contributes
almost nothing, and relevance looks inverted relative to intuition.

**Competing causes.**
(a) The phrase was scored by summing its constituent terms' scores, so a common term's idf dominated;
(b) the phrase was scored as a pseudo-term but its `df` was taken from the rarest constituent instead of
the phrase; (c) `tf` for the pseudo-term counts overlapping occurrences that share tokens, so it
double-counts.

**Discriminating checks.** Compute the phrase's true `df` by running the phrase query with scoring
disabled and counting results; compare with the `df` the scorer used. A large gap means (b). Then
examine a document with `k`-overlapping occurrences: if `tf` exceeds the number of non-overlapping
matches, (c) applies. If the explanation shows separate per-term contributions with no phrase node, (a)
applies.

**Correction.** Score the phrase as one pseudo-term with measured `df`; define `tf` as the number of
non-overlapping match start positions; keep the constituent terms as separate optional clauses only if
the product intends partial credit.

**Prevention.** Log the phrase `df` and `tf` used at scoring time for a sample of queries; a divergence
from a nightly recomputation of the same counts is a leading indicator of all three causes.

### 8.4 Results are non-deterministic for the same phrase query

**Symptom.** The same query returns different result sets, or different orderings, across identical
repeated calls with no index writes between them.

**Competing causes.**
(a) Posting lists were read from different segments because a merge or refresh landed mid-query, so
documents were seen twice or missed; (b) shards were queried without a consistent snapshot and one shard
had a different refresh generation; (c) the top-k protocol terminated early with ties broken by
non-deterministic internal order; (d) an `ids` or `terms` filter was resolved from a cache with a
different snapshot than the posting lists.

**Discriminating checks.** Repeat the query with a fixed searcher or reader pinned: if results become
stable, (a) is confirmed. Compare per-shard doc counts for the phrase: an asymmetry confirms (b).
Re-run with a much larger `k`; if the instability disappears, (c) applies because ties were no longer
truncated. Check whether the filter resolution path has its own cache generation for (d).

**Correction.** Pin one searcher per query; require all shards to answer against a named snapshot;
break score ties with a stable secondary key such as docID; resolve filters against the same reader as
the posting lists.

**Prevention.** Make the searcher generation part of the request context and log it with the response;
then non-determinism is attributable instead of mysterious.

### 8.5 Stage two dominates latency on a small candidate set

**Symptom.** A query that matches few documents is slower than one matching many, and the tail latency
is concentrated on a narrow class of queries.

**Competing causes.**
(a) The anchor has enormous within-document repetition, so anchor position lists are long even though
`df` is small; (b) the candidate set is small but each candidate is a very long document, so per-candidate
merge cost is proportional to length; (c) the phrase contains a term expanded to many synonyms, so each
slot has many position arrays and paths multiply; (d) the matcher falls back to fetching stored text to
verify, which is I/O-bound rather than CPU-bound.

**Discriminating checks.** Log anchor position count alongside `df`: a small `df` with a large position
count confirms (a). Correlate stage-two time with document length for the candidates; a positive
correlation confirms (b). Count expansion paths per slot for (c). If stage two time correlates with disk
read counters rather than CPU, (d) applies.

**Correction.** Choose the anchor by position count, not `df`; add a length-aware work budget and reject
candidates above it rather than verifying them; deduplicate expansion paths before matching; never
verify against stored text when positions are available.

**Prevention.** Track stage-one and stage-two time as separate histograms so a shift between them is
visible before users report it.

### 8.6 Edge case: a one-token phrase

A one-token phrase is a term query and must not be routed through the phrase path. The template has a
single slot, every occurrence matches, and the match count degenerates to the term's `tf`. The edge case
matters because query parsers that quote single words are common, and the phrase code path may assume
`k >= 2` when computing offsets, indexing `off[1]` out of bounds or silently returning nothing. The
correction is an explicit guard that collapses `k = 1` to a term clause, with a test asserting identical
result sets for `word` and `"word"`.

### 8.7 Edge case: phrase whose window exceeds the document length

For a proximity query with `W` larger than the field's token count, every pair of occurrences satisfies
the window, so the constraint is vacuous and the query is a doc-level AND with extra cost. A matcher
should detect `W >= L_max` for the field and either reject the query as unbounded or execute it as AND
without positional verification. Not detecting it produces the worst of both worlds: full phrase cost
with AND semantics.

### 8.8 Edge case: overlapping matches inside one document

The phrase `"a a a"` in the text `a a a a a` has three non-overlapping matches or four overlapping ones,
depending on counting rule. The counting rule changes `tf`, which changes the score, which changes the
ranking — so the rule is a specification, not an implementation detail. State it, test it, and use the
same rule in the offline evaluator as in the serving path, or offline relevance estimates will not
transfer.

## 9. Complexity & Cost Analysis

Complexity here has three independent axes, and a single big-O hides the one that binds. Report all
three: list length (document count), position density (positions per posting), and `k` (terms per
phrase).

**Stage-one intersection.** With `L_i` the length of list `i` in entries, the classic merge is
`O(sum_i L_i)` time and `O(k)` cursor space. A leapfrog or galloping join is
`O(L_anchor * (k-1) * log(L_max))` in the worst case but frequently better in practice because it never
touches entries between matches. Which is faster depends entirely on how sparse the intersection is: for
a dense intersection the merge wins; for a sparse one the galloping join wins. The crossover is
approximately where the number of matched documents exceeds `L_max / log2(L_max)`.

```text
ILLUSTRATIVE crossover arithmetic:
  L_max = 4,640,000 -> log2 = 22.146
  crossover matches = 4,640,000 / 22.146 = 209,522 documents
  below that count, galloping is cheaper; above it, merge is cheaper
```

**Stage-two verification.** `O(D_cand * r_bar * k)` where `r_bar` is mean anchor positions per candidate,
because each anchor position costs up to `k-1` bounded lookups. The constant is tiny, which is why stage
two rarely dominates — except in the pathological case 8.5(a), where `r_bar` is large because one term
repeats heavily inside documents.

**Space.** Positions cost `T * w_p` where `w_p` is the mean encoded position width in bits; posting
entries cost `E * w_d` with `E` the number of distinct (term, document) pairs. The ratio is
`(T * w_p)/(E * w_d) = (T/E) * (w_p/w_d)`, and since `T/E` is the mean occurrences per (term, document)
pair — a corpus property, typically 1.1 to 1.6 — the space multiplier is stable across corpus sizes for
a fixed tokenisation. That stability is why the 1.8x figure in section 5 generalises better than any
single byte count.

**Building.** Writing positions is linear in tokens: one encode operation per token, plus delta
computation. The expensive part of the build is not positions but merging segments, because positions
must be rewritten on merge rather than copied when docIDs are renumbered. A merge policy that rewrites
positions for every merge multiplies ingest I/O by the merge fan-out; that is the real cost driver, and
it is usually invisible in capacity plans that count only final index size.

**Caching.** Positional postings are poor cache citizens compared with doc-level postings: for the same
`df`, the bytes touched per query are larger by the position multiplier. On a node with a fixed page
cache, the practical effect is that a working set which fit as doc-only postings may not fit with
positions, converting cache hits into disk reads. The countermeasure is scoping (section 7) rather than
more RAM.

**Tail latency.** Stage two is per-candidate and bounded, so it contributes to the mean but rarely to the
tail. The tail comes from stage one on the longest lists and from the top-k protocol re-scanning blocks
that cannot be pruned. Concretely, if the block-max metadata says a block's maximum score cannot beat
the current threshold, the block is skipped; if the scoring function is not monotone within the block —
as happens when a phrase pseudo-term's `tf` is computed lazily — pruning is unsound and the protocol must
disable it, which is precisely when the tail appears.

```text
ILLUSTRATIVE pruning benefit:
  blocks scanned without pruning   = 4,640,000 / 128 = 36,250 blocks
  blocks pruned at threshold k=10  = 71% -> 10,512 scanned
  decode work ratio = 10,512 / 36,250 = 0.29, i.e. 2.9x saving on the pruned clause
```

**Cost per query, expressed so it can be checked.** Let `P` be probes per second achievable on one core
for the merge (constructed value 100 million/s for a tight merge loop) and `B` the same for stage-two
bounded searches (constructed value 20 million/s, since each lookup has branch-heavy control flow):

```text
t_query = L_sum / P + (D_cand * r_bar * (k-1)) / B
ILLUSTRATIVE: L_sum = 6,738,000; D_cand = 5,600; r_bar = 1.1; k = 3
  t_query = 6,738,000/1e8 + (5,600*1.1*2)/2e7
          = 0.06738 s + 12,320/2e7
          = 0.06738 + 0.000616 = 0.06800 s = 68.0 ms
```

Stage two is 0.9% of that total, consistent with the 0.18% probe share in section 5 once the different
throughputs are applied. Any plan that quotes only one of the two terms will mispredict by up to a third
when the anchor's density changes.

## 10. Key Theorems / Results Reference

Each result below is stated with the condition under which it holds. None of them is a benchmark; they are
properties of the data structures that let you reason before measuring.

**R1 — Adjacency is a necessary-and-sufficient condition for a two-term ordered phrase, given
positions.** Formally, `doc matches (t1,t2)` iff `exists x in P1(t1): x+1 in P2(t2)` with both arrays
from the same field and analyzer. The "given positions" clause carries the entire weight: without
positions the condition is undecidable at the doc level, not merely approximate. This is the theorem that
makes the two-stage funnel sound.

**R2 — The candidate set from doc-level intersection is a superset of the phrase match set.** By
definition every phrase occurrence occupies one position per term, so any document containing the phrase
contains every term. Superset-hood is what licenses the funnel; its strictness (the false-candidate rate)
is a corpus property, measured in section 5 at 0.66964.

**R3 — Sorted-array merge intersection runs in `O(sum_i L_i)` comparisons, with `O(k)` auxiliary
space.** Standard multiway merge; the invariant is that each cursor only advances. Correctness requires
each list sorted ascending by docID, which is why engines forbid unsorted postings even when a
compressed representation would permit them.

**R4 — Galloping search costs `O(log j)` comparisons to locate the `j`-th entry.** Exponential advance
doubles the step until overshoot, then binary-searches the last interval; the constant is approximately 2
per doubling, giving the `2*ceil(log2(j+1))+1` form used in `[P-02]`. Valid for `j >= 1`; at `j = 0` the
first comparison already succeeds and the formula overstates cost by three probes.

**R5 — Elias-Fano encodes a monotone sequence of `n` values in `[0,U)` in at most
`n*(2 + ceil(log2(U/n)))` bits.** The `n` term accounts for the unary-encoded high parts and the
remaining term for the low bits. Valid when `U >= n`; as noted in 4.4, dense sequences break the
assumption and the bound stops being useful.

**R6 — Delta encoding is optimal-in-expectation for sorted positions under a geometric gap model.**
With independent gap probability `p` of continuing, the gap distribution has mean `1/p` and entropy
`H(p)/p` bits, which is the variable-byte code's expected length floor. Valid for positions generated by
independent-token models; real text has bursty repetition, so measured sizes sit above the floor.

**R7 — BM25's effect on a pseudo-term is monotone in `df` and saturating in `tf`.** The idf factor
`ln(1 + (N-df+0.5)/(df+0.5))` decreases as `df` grows, and the tf factor
`tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl))` is concave. Consequence: a phrase pseudo-term's score cannot be
recovered from its constituents' scores, because `df_phrase` is not a function of the constituent `df`s
(the independence model fails by four to five orders of magnitude, §4.4).

**R8 — Top-k pruning is sound only if the scoring contribution of a block is bounded by its stored
maximum.** If a phrase's `tf` is computed during evaluation, its contribution to a block's maximum is not
known when the metadata is written, so pruning must be disabled for that clause or the metadata must be
computed with phrase scoring in mind. This is the theorem behind the "why is the tail slow after adding
phrase features" question.

**R9 — Position storage is invariant to corpus size for fixed tokenisation.** From section 9,
`space_multiplier = (T/E)*(w_p/w_d)`, with `T/E` the mean occurrences per (term, document) pair. Doubling
`N` while holding the token distribution fixed leaves both factors unchanged, so the multiplier is a
property of language and analyzer, not of scale.

**R10 — Stopword gaps preserve offset arithmetic at a bounded cost.** If each removed token leaves its
slot unused, the number of unused slots per document is at most the number of removed tokens, so the
space overhead is bounded by the stopword rate times `T`. The alternative (compaction) preserves space
and destroys phrase semantics; the theorem is that you cannot have both without storing offsets
separately.

**R11 — A phrase query is field-scoped because field ordinals are independent coordinate systems.**
There is no well-defined offset between `title:5` and `body:5`, so cross-field adjacency requires an
explicit concatenation with a sentinel token, which converts the problem into a single-field problem at
the cost of storing a second representation.

**R12 — Exactness is decidable, not estimable.** Whether the matcher returns the formally defined set is
a property provable from the implementation (positions available, same snapshot, same analyzer). Sampling
recall against a labelled set measures product quality, not exactness; confusing the two leads teams to
"verify" exactness with an A/B test.

## 11. Interview / Exam Questions

Each question has its own answer; the mix is deliberate — recall, calculation, diagnosis and judgment.

**Q1 (recall).** What makes a document a phrase match, stated as a condition on positions?

Answer: there must exist a position `p` such that term `i` of the phrase occupies `p + i` for every `i`,
with all positions drawn from the same field, same document, same snapshot and same analyzer version. The
condition is existential, so a document with several occurrences is one match at the doc level and `m_k`
matches at the occurrence level. Any statement of phrase matching that omits the field or analyzer clause
is incomplete, because both define the coordinate system in which adjacency is checked.

**Q2 (calculation).** A term has `df = 420,000` in a 9,000,000-document corpus and 2.4 occurrences per
document. How many position entries does it contribute, and how does that change anchor choice?

Answer: 420,000 * 2.4 = 1,008,000 position entries. Anchor choice must use the position count, so a
competitor with `df = 300,000` and 1.1 occurrences contributes 330,000 entries and is the better anchor
despite being "less rare" only if the other list is over three times bigger — here 1,008,000 vs 330,000,
so the smaller-`df` term also wins on positions. The interesting case is the inverse: `df = 250,000` with
5.0 occurrences gives 1,250,000 entries, which loses to the larger-`df`, lower-density term.

**Q3 (diagnosis).** Phrase queries on a new collection return nothing; the terms exist in the documents.
What do you check first, and why?

Answer: check whether the field stores positions at all, because that is the cheapest explanation and it
is silent — an engine with `positions: false` on that field cannot answer the query and many
implementations degrade to AND without an error. Only after confirming positions exist do you compare
analyzer tags between index and query, since a stemmer change also produces empty results. The order
matters because enabling positions requires a reindex, while an analyzer mismatch requires the same
reindex but also a mapping change; diagnosing them in the wrong order can cause the wrong rebuild.

**Q4 (calculation).** Two candidate anchors: `df` 800,000 with 1.0 occurrence, and `df` 200,000 with 6.5
occurrences. Which drives fewer probes, by what factor?

Answer: 800,000 * 1.0 = 800,000 entries versus 200,000 * 6.5 = 1,300,000 entries. The high-`df` term is
the better anchor, by a factor of 1,300,000/800,000 = 1.625. This is the counterexample that breaks the
"rarest term is the anchor" heuristic, and it appears whenever a term is a common phrase fragment with
internal repetition, such as a brand name inside a product title.

**Q5 (judgment).** A product manager asks to raise slop from 0 to 5 so that "more results appear". What
is the correct response?

Answer: slop is a semantic change, not a recall knob: at 5 the query no longer means "these words
adjacent" but "these words within five moves", and the result set may include documents whose text reads
nothing like the query. The correct response is to ask which user intent is being served and, if the
intent is "words near each other", expose the window explicitly and measure precision on a labelled set;
if the intent is "more results", add a separate relaxed clause that is visibly labelled in ranking
rather than silently widening the phrase. Raising slop without changing the label misleads both users
and the offline evaluator.

**Q6 (diagnosis).** A phrase clause is 0.2% of probe volume but 40% of the p99 latency on one shard.
What is the likely mechanism?

Answer: pruning is being disabled or defeated for that clause, so the top-k protocol scans blocks it
should skip. The most common cause is that the phrase's contribution is computed during evaluation, so
block-max metadata written at index time cannot bound it; the second cause is a threshold that stays at
zero because the first ten results all score zero, which forces full scans. Check by comparing blocks
scanned against blocks decoded on that clause, and by confirming whether the metadata includes phrase
scores.

**Q7 (calculation).** A field averages 150 tokens and holds 12,000,000 documents. Positions encode at
1.15 bytes; postings at 1.45 bytes with 1.3 occurrences per (term, document) pair. Compute total bytes
and the positions share.

Answer: tokens `T` = 12,000,000 * 150 = 1.8e9; positions = 1.8e9 * 1.15 = 2.07e9 bytes = 1.93 GiB.
Entries `E` = 1.8e9 / 1.3 = 1.3846e9; postings = 1.3846e9 * 1.45 = 2.008e9 bytes = 1.87 GiB. Total =
4.078e9 bytes = 3.80 GiB, of which positions are 2.07/4.078 = 50.8%. Cross-check: per document 150*1.15 =
172.5 position bytes and 115.4*1.45 = 167.3 posting bytes, total 339.8 bytes; times 12,000,000 =
4.078e9 bytes. Consistent.

**Q8 (recall).** Why can a bigram index not answer an arbitrary phrase query?

Answer: a bigram index stores adjacency for exactly two tokens, so a three-token phrase can only be
answered by intersecting two overlapping bigram postings, which requires a positional or offset alignment
to enforce the overlap. Without that alignment the intersection degenerates to "both bigrams occur
somewhere", which is strictly weaker than the phrase. So a bigram index answers two-token phrases exactly
and longer phrases only with additional alignment machinery — which is most of what a positional index
would have cost anyway.

**Q9 (diagnosis).** Relevance looks correct but the same query returns different top-10 across repeated
calls within a second. Where do you look?

Answer: at snapshot consistency across the readers used by one request. If posting lists come from one
searcher and the filter's ID set from a cache populated by another, documents appear and disappear
between calls. The second candidate is tie-breaking: identical scores ordered by an unstable internal
order change when segment layout changes. Distinguish them by pinning a searcher — if results stabilise,
consistency was the cause; if they stabilise only when `k` is raised, ties were being truncated.

**Q10 (calculation).** A phrase has `df = 2,400` in a 6,000,000-document corpus. Compute BM25 idf and
compare with the rarest constituent at `df = 260,000`. What does the ratio mean operationally?

Answer: `idf_phrase = ln(1 + (6,000,000 - 2,400 + 0.5)/2,400.5) = ln(1 + 2,498.4797) = ln(2,499.4797) =
7.8238`. `idf_term = ln(1 + (6,000,000 - 260,000 + 0.5)/260,000.5) = ln(1 + 22.0769) = ln(23.0769) =
3.1388`. Ratio = 7.8238/3.1388 = 2.493. Operationally, matching the phrase roughly multiplies the clause's
weight by 2.49 relative to mentioning the rarest word, so a phrase clause can outrank a strong
single-term match even when its own `tf` is 1; that is intended for navigational queries and surprising
for broad ones.

**Q11 (judgment).** Positions on `body` cost 80% extra index size for a feature used by 2% of queries.
What are the defensible options?

Answer: three options, each with a stated cost. (1) Keep positions everywhere and accept the 80% for
correctness on all fields. (2) Scope positions to the fields where phrase intent is real — title, brand,
category — which cuts the overhead to the token share of those fields, at the price of an explicit
"unsupported on body" answer. (3) Serve body phrases from a separate, smaller index built over a
subset of documents, which keeps the main index lean but adds a second system to operate. Choosing
option 2 without the explicit refusal is the indefensible variant, because it silently returns weaker
results.

**Q12 (diagnosis).** Stage two verifies 5,000 candidates and finds 40 matches. Is that a bug?

Answer: not necessarily; it is a measured false-candidate rate of 1 - 40/5,000 = 0.992. Whether that is a
bug depends on the query: for a phrase naming a common topic, high false-candidate rates are expected
because term co-occurrence is far above independence. A bug is indicated only if the rate exceeds what
the corpus's own co-occurrence statistics predict by a large factor, or if the 40 matches themselves are
wrong — which is a positional-correctness question, not a selectivity question. The two must be measured
separately.

**Q13 (calculation).** Skip pointers every 128 entries add 4 bytes per skip. For a list of 4,640,000
entries, compute skip overhead and the average residual scan.

Answer: skips = 4,640,000 / 128 = 36,250; overhead = 36,250 * 4 = 145,000 bytes = 0.138 MiB, against a
list whose postings alone occupy 4,640,000 * 1.45 = 6.728e6 bytes = 6.42 MiB, so 2.2% overhead. Average
residual scan is about half the interval, 64 entries, versus 4,640,000 for a linear walk. Changing the
interval to 64 doubles skips to 72,500 and halves the residual to 32; the overhead rises to 4.4% for a
residual saving of 32 entries per seek, which is only worth it when seeks per query exceed roughly
`4,640,000` divided by the entries saved — i.e. rarely.

**Q14 (recall).** What is the difference between "the phrase appears" and "the phrase's tokens appear in
the required offsets"?

Answer: they differ exactly in the treatment of intervening tokens and shared positions. "Appears"
connotes a substring of the analysed token stream, which is what the offset condition formalises; but an
analyser that emits several tokens at one offset (synonyms, decompounding) makes the offset condition true
for several distinct token sequences, so the formal condition is broader than the naive substring
reading. Knowing this prevents two errors: rejecting a correct match because the surface string differs,
and accepting a match assembled from unrelated expansions.

**Q15 (judgment).** You must choose between a positional index and a suffix array for a 500-million-token
legal corpus with frequent long quoted passages. What decides it?

Answer: the update regime and the pattern length distribution. Suffix arrays answer long patterns in time
proportional to pattern length, which fits quoted-passage search, but they are hostile to incremental
updates and the corpus grows daily. Positional postings answer any pattern in time proportional to the
sum of list lengths, which for a long quoted passage is dominated by the rarest term and is perfectly
acceptable, and they update per document. So the positional index wins unless patterns routinely exceed
tens of tokens and rebuilds are affordable — the decision is driven by operations, not by the
asymptotics alone.

## 12. Quick Reference Card

| Item | Value / rule | Why it matters |
|---|---|---|
| Phrase definition | `exists p: term_i at p + i`, same field, same analyzer, same snapshot | any clause missing makes the test meaningless |
| Two-stage funnel | doc intersection is necessary; positional check is sufficient | skipping stage two returns bag-of-words results |
| Anchor choice | fewest **position entries** (`df * occurrences`), not fewest `df` | repetition inside documents inverts the usual rule |
| Merge vs gallop | merge is `O(sum L_i)`; gallop is `O(L_anchor * k * log L_max)` | sparse intersections favour gallop, dense favour merge |
| Positions cost | ~1.8x to 2.1x the doc-only index (`ILLUSTRATIVE`) | `(T/E)*(w_p/w_d)`; invariant to corpus size |
| Stopword handling | leave gaps; never compact | compaction makes `"a b"` match `"a the b"` |
| Phrase scoring | pseudo-term with measured phrase `df` and non-overlapping `tf` | constituent-sum scoring mis-weights by ~2.5x |
| Slop | 0 = exact; any positive value is a semantic change requiring a label | otherwise users and evaluators read the same query differently |
| Position width | overflow past the field's integer width truncates silently | long documents lose tail phrase recall with no error |
| Failure to watch | stage-two share rising while candidate count falls | indicates anchor density, not query selectivity |

The card is deliberately short: ten rows that decide most design arguments. Everything else in the sheet
exists to justify these ten, and any of them can be re-derived from the formulas in section 4 without
consulting an engine's documentation.
