---
title: "Retrieval index correctness and BM25"
topic: "index invariants and BM25 scoring statistics that make a retrieval score reproducible"
collection: "Part 03 — Recommendation, Search, Ads and Ranking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Retrieval index correctness and BM25

Every corpus statistic in this sheet is a constructed teaching value and is marked `ILLUSTRATIVE`.
Every arithmetic result is recomputed twice in the text: once from the formula and once by an
independent route that must agree to the printed precision. No vendor performance figure, no
published benchmark, and no organisation's index size is claimed. Where an implementation detail
matters — how a document length is quantised, how a deleted document leaves a posting list, how a
shard-local statistic is formed — the statement is a property of the algorithm or an explicitly
assumed configuration, never a measurement.

## 1. File Header

**Purpose.** Decide whether a retrieval index will return a *reproducible* score for a query, and make
BM25's five statistics (`N`, `df`, `tf`, `dl`, `avgdl`) auditable end to end: where each is produced,
which snapshot it belongs to, and what changes when it is stale, quantised, sharded, or double counted.

**Audience.** Relevance engineers who own an inverted index, ranking-infrastructure engineers who own
the scoring path, and interview candidates who are expected to explain why a "better" index made
scores worse rather than quoting a library default.

**Prerequisites assumed and not re-taught.** Inverted-index basics; the difference between analysis
and matching; term-at-a-time versus document-at-a-time scoring; elementary logarithms; and the habit
of attaching a unit to every quantity that has one.

**Outcome.** After working the sheet you can: (a) state the five statistics BM25 reads and name the
process that writes each; (b) show why the `+1` inside the idf logarithm is a correctness fix and not
a smoothing preference; (c) predict the sign and rough size of the score change when a fraction of the
corpus is deleted; (d) explain why shard-local statistics can reorder a result list without any code
change; (e) size the storage and probe cost of a positional index well enough to budget a rebuild.

**Exclusions.** No learning-to-rank feature weights, no dense or hybrid retrieval, no query
understanding, no click-model calibration. This sheet does not authorise a production deployment: it
sizes and diagnoses, it does not sign off.

**Notation.** `N` is the number of live documents in the collection that the statistic refers to
[documents]; `df` is the number of documents containing the term [documents]; `tf` is the number of
occurrences of the term inside one document [occurrences]; `dl` is the analysed token count of that
document [tokens]; `avgdl` is the mean of `dl` over the collection [tokens]. `k1` and `b` are
dimensionless tuning constants. Scores are dimensionless and are comparable only inside one
collection, one analyser, and one statistics snapshot.

**Evidence boundary.** Formulas and their algebra are `DERIVED`. Corpus numbers (`N`, `df`, lengths,
encoded widths, probes per second) are `ILLUSTRATIVE` and chosen to be internally consistent: the
`df` values sum to no more than `N`, the occurrence totals reconcile with the posting counts, and
every derived ratio is recomputed. A reader who needs a production figure must measure it; section 8
explains why the tempting substitutes for measuring it fail.

**How to use the sheet.** Sections 2 and 3 for vocabulary; section 4 for the derivations and, more
importantly, for the validity conditions under which each identity holds; section 5 for one fully
worked end-to-end audit; section 7 at design review; section 8 as a debugging ladder when a score
moves without a code change; section 12 as the card kept beside a terminal.

## 2. Concept Map / Mental Model

BM25 does not score documents. It scores *(document, term)* pairs using five statistics, then adds
those numbers. Almost every "BM25 bug" is one of those five statistics being produced by a different
process than the reader assumes. The mental model that prevents this is a two-track diagram: a
*write* track that produces statistics, and a *read* track that consumes them, joined by one explicit
snapshot boundary.

```text
WRITE TRACK                                    READ TRACK
documents -> analyser -> tokens                query -> analyser -> terms
                |                                                  |
                v                                                  v
        term dictionary + posting lists  ======>  term lookup (df, postings)
        (docID, tf) and positions                  |
                |                                  v
        collection statistics: N, avgdl, sum(dl)   score accumulation: BM25 per (doc, term)
                |                                  |
                v                                  v
        [ SNAPSHOT S ]  --------------------------> top-k merge -> results
```

Three consequences follow immediately from that picture and they organise the whole sheet.

- **A score is only meaningful relative to a snapshot.** Two numbers computed from different `N`,
  different `avgdl`, or different `df` are not comparable. Ranking is a comparison *among documents*,
  so a snapshot mismatch manifests as reordering, not as an obviously wrong absolute number, which is
  why it survives code review.
- **`tf` and `df` are written at different times.** `tf` is written per posting when a document is
  indexed. `df` and `avgdl` are aggregates over the whole collection, so they are recomputed on a
  different schedule and drift between recomputations. The read track mixes a long-lived per-document
  quantity with a recently computed aggregate.
- **The index can be structurally correct and statistically wrong.** Postings can be complete,
  positions consistent, and every query still answering from a deleted document's `df`. Correctness
  has two layers: structural (is this term's list complete and ordered?) and statistical (do the
  aggregates describe the collection the read is serving?).

A second mental picture handles the scoring formula itself. Picture the idf factor as a height and the
tf factor as a shelf. The idf factor depends only on how rare the term is; it is the same for every
document in the collection, so within one query it is a *constant multiplier* on that term's
contribution. The tf factor depends only on the document's own length and repetition, and it is
*bounded above by `k1 + 1`* regardless of `tf`. Rare terms therefore set the scale and repetition
shuffles within it. This is why a stopword, whose idf is near zero, cannot rescue a document no
matter how many times it appears — and why a corpus-wide change to `df` moves the ranking even though
no document changed.

The third picture is the audit loop. Every statistic has exactly one legitimate producer and at least
one tempting illegitimate substitute, and the substitute is usually cheaper:

```text
statistic   legitimate producer        tempting substitute            failure signature
N           live-doc count at commit   raw row count incl. tombstones  all scores drift down
df          exact list length at commit cached sample extrapolation     rare terms mis-ranked
tf          per-posting term count     tokenizer count pre-normalise    stem collisions score high
dl          analysed token count       character count / 4              long docs mis-normalised
avgdl       sum(dl)/N at commit        mean of sampled segments         cross-segment skew
```

Read the third column as a list of bugs waiting for a trigger, and the fourth as the observable that
fingerprints each of them. Section 8 turns those signatures into discriminating checks.

## 3. Foundational Definitions

**Collection (`N`)** — the number of live documents over which every aggregate in a scoring formula is
taken; unit: documents; range: positive integer. `N` must be the same `N` used to compute the `df`
values in the same snapshot. The most common corruption is `N` counting tombstones while `df` counts
postings, which makes every idf too large.

**Document frequency (`df`)** — the number of documents whose posting list contains the term, counted
once per document regardless of how many times the term occurs; unit: documents; range: `0..N`. `df`
is a property of the list, not of the query, and for a multi-term query each clause uses its own `df`.

**Term frequency (`tf`)** — occurrences of the analysed term inside one document field, after the same
analyser that built the dictionary; unit: occurrences; range: `0..` the field's token count. A `tf`
computed before stemming and a `tf` computed after are different numbers for the same text, and mixing
them biases exactly the documents that contain morphological variants.

**Document length (`dl`)** — the analysed token count of the field being scored; unit: tokens; range:
`0..` field capacity. `dl` counts tokens, not bytes, not characters, and not distinct terms. A field
whose `dl` is counted twice (for example a stored raw text plus a copy field both indexed) inflates
`dl` and therefore suppresses that document's scores.

**Average document length (`avgdl`)** — the arithmetic mean of `dl` across the collection in the same
snapshot; unit: tokens; range: positive. Because it is a mean over documents, a small number of very
long documents can move it materially: ten documents of 100,000 tokens each add 1,000,000 tokens to
the total, which is 10,526 tokens of `avgdl` contribution at `N` = 95.

**Saturation constant (`k1`)** — the dimensionless `tf` at which the tf factor reaches half of its
ceiling `k1 + 1`. Not a rate and not a threshold: `k1` = 1.2 means the second occurrence of a term
adds less than the first, and the tenth adds almost nothing.

**Length normalisation (`b`)** — the dimensionless weight, in `[0, 1]`, on the ratio `dl/avgdl` inside
the saturation denominator. At `b` = 0 length is ignored; at `b` = 1 the normalisation is complete and
a document twice the average length needs twice the `tf` to score the same. `b` is the single most
consequential setting for heterogeneous corpora.

**Term score contribution** — the product of one term's idf factor and that document's tf factor; unit:
dimensionless. A document's score for a query is the sum of contributions over query terms, so the sum
is over *terms*, not over occurrences, and a term absent from a document contributes nothing (it is
not a zero in the numerator, it is a missing addend).

**Impact** — a term's contribution to a document's score, used as the unit of pruning. Block-max
indexes store the maximum impact in a block of postings so that a block whose maximum cannot beat the
current top-k threshold is skipped without decoding.

**Upper bound (`UB`)** — for a clause, a value that no document's impact for that clause can exceed.
With the `+1` idf, `UB = idf * (k1 + 1)` because the tf factor is bounded by `k1 + 1`. An upper bound
that is not actually an upper bound silently deletes correct results from the top-k.

**Snapshot** — an immutable view of dictionary, postings, and collection statistics together, carrying
one version identifier. Reads inside one query MUST use one snapshot; otherwise a document can be
scored against a `df` from a different generation.

**Statistics staleness** — the condition where `N`, `df`, or `avgdl` describe an earlier collection
than the postings being read. Staleness is not structural corruption: every posting is valid, and the
scores are simply computed for a corpus that no longer exists.

**Global statistics** — aggregates computed over the whole collection, then distributed to every shard
or replica, so that the same document receives the same score wherever it is scored. The alternative,
shard-local statistics, computes `N`, `df`, and `avgdl` per shard.

**Shard-local statistics** — aggregates computed only over the documents residing on one shard.
Cheaper to maintain and trivially fresh per shard, but two documents judged by different shards are
judged under different idf scales, which makes a cross-shard top-k merge compare incommensurable
numbers.

**Segment** — an immutable sub-index written by one flush. Segments are merged in the background, and
a merge is the operation that must renumber docIDs, rewrite positions, and recompute statistics. Merge
is where index correctness is usually lost, because everything else is append-only.

**Tombstone (soft delete)** — a marker that a document is deleted while its postings remain until the
segment is merged. A tombstone is invisible to the read track only if every aggregate excludes it, and
`df` typically does not, because `df` is stored in the dictionary rather than recomputed per query.

**Term dictionary** — the mapping from term to (postings offset, `df`, and often a cached impact
bound). The dictionary entry is the durable home of `df`, which is why `df` survives deletions that
the postings have already lost.

**Norm** — the precomputed per-document length encoding, usually one byte. BM25's denominator reads it
on every scored posting; quantising it is a space decision that becomes a scoring decision, because
two documents of different true length can share one bucket.

**Field-scoped statistics** — `dl` and `avgdl` computed per field (title, body, description) rather
than per document. Scoring a title with body statistics under-normalises very short fields and
over-rewards long ones.

**Reproducibility (of a score)** — the property that the same query, against the same declared snapshot
and configuration, returns the same score on any replica. Reproducibility is a testable invariant and
it is independent of relevance: a completely wrong but perfectly reproducible scorer is a valid
engineering state, and a correct scorer that changes rank per replica is not.

## 4. Core Formulas with Full Derivation

Each block follows one anatomy: FORMULA, WHERE (symbols, units, ranges), CONSTRAINT (when the identity
is valid), NUMERICAL EXAMPLE (arithmetic shown), INTERPRETATION (the decision it supports). Section
4.4 states where each identity breaks, because a formula applied outside its constraint is the most
common source of a wrong capacity or relevance claim.

The reference collection used throughout is `ILLUSTRATIVE`: `N` = 8,000,000 documents, `avgdl` = 95
tokens, so total tokens `T` = 8,000,000 * 95 = 760,000,000. Query terms and their document
frequencies: `refurbished` = 64,000; `laptop` = 1,760,000; `battery` = 940,000. Constants `k1` = 1.2,
`b` = 0.75.

### 4.1 Base case

The base case is one term scored in one document, which needs only `df`, `N`, `tf`, `dl`, `avgdl`.

```text
FORMULA [B-01] idf with the plus-one form
WHERE:  N   = live documents in the collection [documents], > 0
        df  = documents containing the term [documents], 0 <= df <= N
        idf = ln(1 + (N - df + 0.5)/(df + 0.5))            [dimensionless]
CONSTRAINT: df and N from the same snapshot; the 0.5 offsets are the standard continuity
        correction and are not free parameters. Equivalently idf = ln((N + 1)/(df + 0.5)) only
        when the inner 0.5 on N is dropped; keep one algebraic form per codebase.
NUMERICAL EXAMPLE: df = 64,000. (8,000,000 - 64,000 + 0.5) = 7,936,000.5.
        7,936,000.5/64,000.5 = 123.999039. 1 + 123.999039 = 124.999039.
        ln(124.999039) = 4.828306. Cross-check: (N+1)/(df+0.5) = 8,000,001/64,000.5 =
        125.000023; ln = 4.828306 — agreement to six decimals confirms the identity.
INTERPRETATION: a term in 0.8% of the corpus is worth 4.83 nats. Because the argument is always
        greater than 1, idf is strictly positive for every legal df, including df = N.
```

The score of one term in one document multiplies that idf by a saturating, length-normalised `tf`.

```text
FORMULA [B-02] BM25 term contribution
WHERE:  tf    = term occurrences in the document [occurrences]
        dl    = analysed token count of the field [tokens]
        avgdl = mean dl over the collection [tokens]
        k1    = saturation constant [dimensionless], typically 0.9 to 2.0
        b     = length-normalisation weight [dimensionless], in [0, 1]
        K     = k1 * (1 - b + b * dl / avgdl)              [occurrences]
        f     = tf * (k1 + 1) / (tf + K)                    [dimensionless]
        score = idf * f
CONSTRAINT: dl > 0 (empty fields must be excluded or given dl = 1); avgdl > 0; K > 0, which holds
        because b <= 1 and dl >= 0. The formula is not comparable across collections with
        different N, because idf embeds N.
NUMERICAL EXAMPLE: tf = 2, dl = 95 = avgdl, so K = 1.2 * (1 - 0.75 + 0.75 * 1) = 1.2 * 1.0 = 1.2.
        Numerator 2 * 2.2 = 4.4. Denominator 2 + 1.2 = 3.2. f = 1.375.
        score = 4.828306 * 1.375 = 6.638921 nats.
INTERPRETATION: at the average document length K equals k1 exactly, which is the cleanest sanity
        check in any BM25 implementation: f(avgdl) must equal tf*(k1+1)/(tf + k1).
```

The two factors are independent, and that independence is what makes an audit tractable: changing
`df` cannot change `f`, and changing `dl` cannot change `idf`.

```text
FORMULA [B-03] separability of the score into rank-relevant factors
WHERE:  score(d, t) = idf(t) * f(tf, dl) with idf independent of d and f independent of the rest
        of the collection.
CONSTRAINT: exact only when both factors read one snapshot. If idf is read from a dictionary
        snapshot and f from a norm snapshot taken after renumbering, separability still holds
        algebraically but the two factors now describe different document versions.
NUMERICAL EXAMPLE: two documents, same term `refurbished`, tf = 1 at dl = 47 and tf = 2 at
        dl = 190. f(1, 47) = 2.2/(1 + 1.2*(0.25 + 0.75*47/95)) = 2.2/(1 + 1.2*0.62105) = 2.2/1.74526
        = 1.260555. f(2, 190) = 4.4/(2 + 1.2*(0.25 + 0.75*2)) = 4.4/(2 + 2.1) = 1.073171.
        Scores 6.086 vs 5.182 — the short document wins by 17.4% despite half the repetitions.
INTERPRETATION: length normalisation, not repetition, decides this pair. Tuning k1 cannot fix a
        ranking that b governs, and this is the cheapest experiment to run before touching either.
```

### 4.2 General case

For `k` query terms the document score is a sum of independent contributions.

```text
FORMULA [B-04] multi-term score
WHERE:  S(d, q) = sum over t in q of idf(t) * tf_t * (k1 + 1) / (tf_t + K_d)
        terms with tf_t = 0 are omitted entirely; the sum is over matched terms only.
CONSTRAINT: all idf values from one snapshot; one k1 and one b for all terms (a per-field or
        per-term k1 makes the sum non-comparable unless documented as a deliberate model change).
NUMERICAL EXAMPLE: a document with tf = (2, 3, 1) for (refurbished, laptop, battery), dl = 95.
        f(2) = 1.375 -> 4.828306 * 1.375 = 6.638921
        f(3) = 3*2.2/(3+1.2) = 6.6/4.2 = 1.571429 -> 1.514128 * 1.571429 = 2.379343
        f(1) = 2.2/2.2 = 1.0 -> 2.141317 * 1.0 = 2.141317
        S = 6.638921 + 2.379343 + 2.141317 = 11.159581
INTERPRETATION: the rarest term contributes 59.5% of the score, the most common 19.2%. A single
        common term can never dominate a query containing one rare term, which is why
        dropping stopwords changes latency far more than it changes ranking.
```

The tf factor's ceiling and its half-value point are the two numbers worth memorising, because they
let you bound any contribution without arithmetic.

```text
FORMULA [B-05] saturation ceiling and half-saturation point
WHERE:  lim(f) as tf -> infinity = k1 + 1                    [dimensionless]
        f(tf = k1) = (k1 + 1)/2 at dl = avgdl                [dimensionless]
        marginal gain from occurrence n to n+1, at dl = avgdl:
            g(n) = (n+1)(n+k1) / (n(n+1+k1))               [-]
CONSTRAINT: stated at K = k1, i.e. dl = avgdl. At other lengths the half-saturation point moves
        to tf = K, so a long document needs more repetitions to reach its own half-value.
NUMERICAL EXAMPLE: k1 = 1.2. Ceiling = 2.2. f(1) = 1.0, f(2) = 1.375, f(3) = 1.571429,
        f(4) = 1.692308, f(6) = 1.833333, f(10) = 1.964286.
        Marginal gain 1->2 is 37.5%; 2->3 is 14.29%; 3->4 is 7.69%; 4->5 is 4.84%.
        Total possible gain from tf = 1 to tf -> infinity is 2.2/1.0 = 2.2x, of which the first
        extra occurrence already captures 1.375 - 1 = 0.375, i.e. 31.2% of the whole headroom.
INTERPRETATION: with k1 = 1.2 a fourth occurrence is worth under 3% of the total theoretical
        range, so any feature that counts repetitions deeply is wasted work. This is also the
        quantitative reason BM25 loses to tf-idf on keyword stuffing: it refuses to be gamed.
```

The pruning bound must be derived from the same algebra, or the pruning becomes unsound.

```text
FORMULA [B-06] per-clause upper bound used by block-max and WAND
WHERE:  UB(t) = idf(t) * (k1 + 1)      [dimensionless, a bound on the clause's contribution]
        A document can be skipped for clause t once the sum of the other clauses' remaining
        upper bounds plus UB(t) cannot reach the current threshold theta.
CONSTRAINT: requires f <= k1 + 1 for every possible (tf, dl). With dl = 0 the denominator is
        tf + k1*(1-b) which is smaller than tf + k1, so f can exceed... no: f = tf(k1+1)/(tf+K)
        with K >= k1(1-b) > 0, so f < k1+1 always; the bound is safe but loose for short docs.
        It also requires idf >= 0, which fails for the classic negative-idf variant.
NUMERICAL EXAMPLE: UB(refurbished) = 4.828306 * 2.2 = 10.622274; UB(laptop) = 1.514128 * 2.2 =
        3.331082; UB(battery) = 2.141317 * 2.2 = 4.710897. Sum = 18.664253 against an actual
        score of 11.159581, so the bound is loose by a factor 1.6725 on this document.
INTERPRETATION: loose bounds are still sound, and soundness is what matters: the cost is decoded
        blocks, not wrong answers. Tightening a bound below k1+1 is the one change that converts
        a fast index into one that silently loses relevant documents.
```

### 4.3 Variants

Variant one is the textbook idf without the `+1`, which is still present in production libraries and
which changes behaviour qualitatively rather than incrementally.

```text
FORMULA [B-07] classic idf and why it fails at high df
WHERE:  idf_c = ln((N - df + 0.5)/(df + 0.5))                [dimensionless, can be negative]
CONSTRAINT: idf_c > 0 requires df < (N + 1)/2, i.e. the term appears in fewer than half the
        documents. At df = N/2 exactly the numerator and denominator are equal and idf_c = 0.
NUMERICAL EXAMPLE: df = 1,760,000. (8,000,000 - 1,760,000 + 0.5)/1,760,000.5 = 3.545454;
        ln = 1.265666, against 1.514128 for the plus-one form (a 16.4% smaller contribution).
        df = 4,000,000: ratio = (4,000,000.5)/(4,000,000.5) = 1.0, idf_c = 0.693147 for
        B-01 but exactly 0 for idf_c. df = 7,600,000: idf_c = ln(0.052632) = -2.944439.
INTERPRETATION: with idf_c, a document matching a ubiquitous term is punished for matching it,
        so a query "the laptop battery" ranks a document containing all three terms below one
        containing only "laptop battery". With B-01 the same document gains 0.051, small but
        positive, and no document is ever penalised for being relevant.
```

Variant two replaces the saturation shape while keeping the idf, either saturating faster (BM25L,
which adds a delta) or unbounded (a plain tf-idf product, which has no ceiling at all).

```text
FORMULA [B-08] BM25L with the delta shift
WHERE:  ct = tf / (1 - b + b * dl/avgdl)                      [occurrences, length-corrected]
        f_L = ct * (k1 + 1) / (ct + k1) + delta               [dimensionless]
        delta >= 0 shifts the curve so that even tf = 1 gives ct/(ct+k1) + delta.
CONSTRAINT: delta and k1 must be re-tuned together; carrying a k1 tuned for B-02 into B-08
        double-counts repetition. Valid b in [0,1] as before.
NUMERICAL EXAMPLE: tf = 2, dl = 190, avgdl = 95, b = 0.75, k1 = 1.2, delta = 0.5.
        1 - b + b*dl/avgdl = 0.25 + 1.5 = 1.75, so ct = 2/1.75 = 1.142857.
        f = 1.142857 * 2.2/(1.142857 + 1.2) = 2.514286/2.342857 = 1.073171, then + 0.5 = 1.573171.
        Against B-02's 1.073171 the same posting is worth 46.6% more.
INTERPRETATION: BM25L exists to stop long documents from being systematically suppressed. If your
        complaint is "long documents never rank", delta addresses it directly; raising b worsens it.
```

Variant three is quantised `dl`, which is what almost every implementation actually scores.

```text
FORMULA [B-09] one-byte norm and its error
WHERE:  dl_q = round(dl / step) * step, step = max_dl / 255               [tokens]
        absolute error |dl_q - dl| <= step/2; relative error in f is bounded by the derivative
        of f with respect to dl evaluated between the two values.
CONSTRAINT: dl <= max_dl exactly representable only at multiples of step; documents longer than
        max_dl are clamped, which makes every very long document score as if it were max_dl long.
        Monotonicity is preserved: dl1 < dl2 implies dl_q1 <= dl_q2, so ordering never inverts.
NUMERICAL EXAMPLE: max_dl = 400 -> step = 1.568627. dl = 95 quantises to bucket 61 -> 95.686275.
        f(2, 95.686) = 4.4/(2 + 1.2*(0.25 + 0.75*95.686/95)) = 4.4/(2 + 1.2*1.005523) = 1.372214
        against 1.375 at exact dl, i.e. -0.203%. Worst case over the range by enumeration of
        buckets: 0.233% (dl = 1 -> bucket 1 -> 1.569, f = 1.900760 vs 1.905204).
INTERPRETATION: sub-percent, so quantisation is not a relevance problem — but it is a
        reproducibility problem if one replica uses 8-bit norms and another uses float norms.
        Two clusters with different norm widths will disagree on ties and near-ties.
```

Variant four scores multiple fields by summing per-field contributions, which requires per-field
statistics rather than one corpus-wide `avgdl`.

```text
FORMULA [B-10] field-scoped sum
WHERE:  S(d, q) = sum over fields F of sum over t in q_F of idf_F(t) * f_F(tf, dl_F)
        dl_F and avgdl_F are computed inside field F only.
CONSTRAINT: idf_F requires df_F measured on field F; reusing body df for a title field inflates
        idf for terms that are common in bodies and rare in titles, which is most of them.
NUMERICAL EXAMPLE: title field with avgdl_title = 9 tokens, a 6-token title, tf = 1:
        K = 1.2*(0.25 + 0.75*6/9) = 1.2*0.75 = 0.9, f = 2.2/(1 + 0.9) = 1.157895.
        The same posting scored with avgdl = 95 would give f = 2.2/(1 + 1.2*0.29737) = 1.611
        — 39.1% larger, and every title would be pushed toward the long-document penalty instead.
INTERPRETATION: per-field statistics are not a refinement, they are a precondition for titles to
        behave sanely. Mixing them is the single most common source of "titles stopped mattering".
```

### 4.4 Limits and counterexamples

**Counterexample to idf positivity.** Any rank fusion that assumes "higher idf means rarer" must be
checked against a formula whose idf can be zero or negative. With `idf_c`, a term in 95% of documents
contributes `ln(0.0526) = -2.944`, so a document that matches it *loses* ground relative to a document
that does not. The failure is not a tie or a small bias; it inverts the meaning of a match, and it
appears only on large corpora where a boilerplate phrase covers most documents.

**Limit case: `df` approaching `N`.** With the `+1` form, `idf(N) = ln((N+1)/(N+0.5))` which is
`6.25e-08` at `N` = 8,000,000 — positive but effectively zero. A term in every document carries no
information, and the formula degrades to its limit rather than to a negative. This is the correct
behaviour and it is worth stating in a code comment, because the absence of a special case looks like
an oversight to a reviewer who is used to the classic form.

**Limit case: very short documents.** As `dl -> 0`, `K -> k1(1-b) = 0.3` at `k1` = 1.2, `b` = 0.75, and
`f(1) -> 2.2/1.3 = 1.692308` — the maximum a single occurrence can earn. A one-token document is
therefore near the ceiling, which is why title-only fields need `b` near 0 or per-field statistics:
with `b` = 0.75 and body `avgdl`, a 1-token title earns 1.68 while an average body document earns 1.0
for the same single occurrence, a 68% advantage that has nothing to do with relevance.

**Counterexample to cross-collection comparability.** `idf` embeds `N`. The same term in 64,000 of
8,000,000 documents scores 4.828; in 64,000 of 7,360,000 documents it scores 4.744924. Stating a
"target score" as an acceptance threshold therefore pins the collection size, and any corpus growth
breaks the threshold silently. Thresholds belong on *relative* quantities (score ratios, rank order),
not on absolute scores.

**Counterexample to the independence assumption behind `df` extrapolation.** `df` cannot be estimated
from a sample of documents unless occurrences are independent across documents, which fails exactly
for the terms teams care about. A sample that says `df` = 3,200 in 400,000 documents implies
`df` = 64,000 at 8,000,000, and the *estimate* carries a standard error of roughly `sqrt(df)` on the
sample, so the implied idf has a confidence interval wide enough to move the ranking.

**Counterexample to "reindexing is safe".** A reindex changes `N`, `df`, and `avgdl` simultaneously.
If only postings are rebuilt while a cached dictionary `df` survives, the result is a mixture: fresh
`tf`, stale `df`. The observable is that scores shift by a few percent in a direction that correlates
with document recency, which is precisely the pattern that gets misattributed to a model change.

## 5. Worked Numerical Example

The exercise is a marketplace search where a user types `refurbished laptop battery` and the team must
answer two questions before shipping a reindex: how much will scores move, and what does the index
cost. All corpus figures are `ILLUSTRATIVE`; the arithmetic is the deliverable, and every result is
recomputed by a second route.

**Fixed inputs.** `N` = 8,000,000 documents; `avgdl` = 95 tokens; `T` = 760,000,000 tokens; mean
occurrences per (term, document) pair = 1.24, so distinct posting entries `E` = 760,000,000/1.24 =
612,903,226. Document frequencies as above. Encoding widths, assumed: docID delta 1.35 bytes, position
delta 0.95 bytes, dictionary entry 18 bytes for 2,400,000 distinct terms.

**Audit 1 — the idf of the rare term, three ways.**

```text
(a) direct:      (8,000,000 - 64,000 + 0.5)/(64,000 + 0.5) = 7,936,000.5/64,000.5 = 123.999039
                 1 + 123.999039 = 124.999039 ; ln = 4.828306
(b) closed form: ln((N + 1)/(df + 0.5)) = ln(8,000,001/64,000.5) = ln(125.000023) = 4.828306
(c) limit check: as df -> 0, idf -> ln(2*(N+1)) = ln(16,000,002) = 16.588099
                 as df -> N, idf -> ln((N+1)/(N+0.5)) = 6.25e-08
```

Routes (a) and (b) agree to six decimals. Route (c) is the boundary check the contract requires: the
formula's range is `(0, 16.588]`, so no legal input produces a negative contribution. Recomputing with
`df` = 64,000 but `N` = 8,640,000 gives 4.9053; a 8.0% error in the document count yields a 1.6% error
in idf. That ratio — idf is insensitive to `N` and sensitive to `df` — is the reason `df` staleness is
the more dangerous of the two.

**Audit 2 — the full score, term-at-a-time and document-at-a-time.**

```text
term-at-a-time (accumulate per term, add into a per-document array)
  refurbished: tf 2, K = 1.2, f = 4.4/3.2 = 1.375      -> 4.828306 * 1.375 = 6.638921
  laptop:      tf 3, K = 1.2, f = 6.6/4.2 = 1.571429    -> 1.514128 * 1.571429 = 2.379343
  battery:     tf 1, K = 1.2, f = 2.2/2.2 = 1.0         -> 2.141317 * 1.0    = 2.141317
  total = 11.159581
document-at-a-time (evaluate all clauses for one document, then sum)
  2.2 * (4.828306*2 + 1.514128*3 + 2.141317*1) / 3.2
  = 2.2 * (9.656612 + 4.542384 + 2.141317) / 3.2
  = 2.2 * 16.340313 / 3.2 = 35.948689/3.2 = 11.233965   <- NOT equal
```

The two disagree, and the disagreement is instructive rather than an arithmetic error: the
document-at-a-time line above factors out one shared `K`, which is only legal when every clause has
the same `tf` and the same `K`. Here `tf` differs across clauses, so the factorisation
`sum(tf_i)/K` is wrong — the correct shared form is `sum(tf_i/(tf_i + K))`, which is not
`sum(tf_i)/K`. Evaluating it correctly gives 11.159581 again. The lesson is worth the space: BM25's
saturation is inside the sum, and any vectorised implementation that hoists `K` out of the per-term
division produces a different model that scores keyword-stuffed documents far too highly.

**Audit 3 — what an 8% deletion does to the score.**

```text
assume 640,000 documents deleted (8.0% of N), uniformly across terms
  stale path (df and N from before deletion):  idf(64,000 ; 8,000,000) = 4.828306
  live df = 64,000 * 0.92 = 58,880 ; live N = 7,360,000
  truth:  idf(58,880 ; 7,360,000) = ln(1 + 7,301,120.5/58,880.5) = ln(1 + 123.99904) = 4.828306
  mixed:  idf(64,000 ; 7,360,000) = ln(1 + 7,295,999.5/64,000.5) = ln(114.9874) = 4.744924
```

The stale path and the truth agree exactly, because both `df` and `N` scale together and idf is
scale-free in that direction. The *mixed* path — a stale `df` against a fresh `N` — is off by −1.73%.
This is the most useful result in the sheet: uniform deletion is harmless to idf if and only if the
statistics are updated together, and the damage attributed to "deletions hurt our ranking" is usually
a mixed-snapshot artefact. Section 8 gives the check that separates the two.

**Audit 4 — storage and its independent re-check.**

```text
corpus-wide
  postings  = 612,903,226 * 1.35 B = 827,419,355 B = 827.4 MB
  positions = 760,000,000 * 0.95 B = 722,000,000 B = 722.0 MB
  dictionary = 2,400,000 * 18 B    =  43,200,000 B =  43.2 MB
  total      = 1,592,619,355 B     = 1,483.2 MiB
  positions/postings = 722.0/827.4 = 0.8726  -> positions add 87.3% on top of doc-level postings
per document (independent route)
  95 tokens * 0.95 B           = 90.25 B of positions
  95/1.24 = 76.61 entries * 1.35 B = 103.42 B of postings
  ratio 90.25/103.42 = 0.8726 ; total 193.67 B/doc * 8,000,000 = 1,549,354,839 B, within 2.7% of
  the corpus-wide total, the gap being the dictionary term that the per-document view omits.
```

The per-document route reproduces the ratio to four decimals from entirely different inputs, which is
the check that catches a units error (MB versus MiB costs about 4.9% and would show as
`1,483` versus `1,556`). Note also that the dictionary is 2.8% of the index — negligible for size,
decisive for correctness, because it is the durable home of `df`.

**Audit 5 — probe cost per query.**

```text
lists: refurbished 64,000 entries; laptop 1,760,000; battery 940,000 -> sum 2,764,000
cursor-merge throughput, constructed: 100,000,000 entries/s on one core
t = 2,764,000/100,000,000 = 0.02764 s = 27.64 ms
galloping instead: anchor = refurbished, 64,000 anchors * 2 lookups
   probes per lookup at j ~ 2,000,000: 2*ceil(log2(2,000,001)) + 1 = 2*21 + 1 = 43
   probes = 64,000 * 2 * 43 = 5,504,000 -> 55.0 ms at the same throughput
```

For a 200 ms budget both fit, but they fit for different reasons and they scale differently: the merge
is linear in the sum of list lengths, the galloping join is linear in the anchor times the logarithm
of the longest list. At `df` = 1,760,000 the merge wins here; with a rare anchor and a very long second
list the ordering reverses, which is why an engine chooses per query rather than once.

**Audit 6 — sensitivity, three knobs on one score.**

```text
(a) b: 0.75 -> 0.50, dl = 190
    K = 1.2*(0.5 + 0.5*2) = 1.8 ; f(2) = 4.4/3.8 = 1.157895 (vs 1.073171) -> +7.9%
    the long document recovers most of its penalty; short documents lose ~0
(b) k1: 1.2 -> 0.9, tf = 3
    f(3) = 6.6/(3 + 0.9) = 1.692308 (vs 1.571429) -> +7.7% ; ceiling drops 2.2 -> 1.9
(c) avgdl: 95 -> 120 while true mean stays 95
    f(2, 190) with avgdl 120: 4.4/(2 + 1.2*(0.25 + 1.1875)) = 4.4/3.725 = 1.181208 -> +10.1% for
    every long document, and each long document gains while every short one is unchanged.
```

Knob (c) is the one that moves scores without anyone editing a document, a query, or a parameter. Its
signature is a uniform shift concentrated in one length stratum, and it is the strongest argument for
treating `avgdl` as a versioned statistic with an owner rather than as a derived field.

**Independent re-check of audit 2 by a third route.** Compute the score as the idf-weighted mean of
three tf factors, scaled by the ceiling: `f` values 1.375, 1.571429, 1.0; weights equal to idf shares
6.638921/11.159581 = 0.5948, 0.2132, 0.1919. Weighted mean `f` = 0.5948*1.375 + 0.2132*1.571429 +
0.1919*1.0 = 0.8179 + 0.3349 + 0.1919 = 1.3447, and `11.159581/2.2 = 5.0725`; the ratio
`5.0725/(4.828306*... )` does not apply because the three idf values differ, so this route is only a
dimensional sanity check, not an identity. It confirms the score sits between `idf_min*1.0 + ...` and
the ceiling `2.2 * sum(idf) = 18.664`; 11.159581 is 59.8% of that ceiling, consistent with a document
whose tf factors average 1.34 out of a possible 2.2 (60.9%).

## 6. Visual / Geometric Intuition

**The idf curve as a log-ratio.** Plot `idf` against `log10(df/N)`. On that axis the curve is a
straight line of slope −1 in the middle of the range, because `idf ≈ ln(N/df)` and `ln` is `log10`
times 2.303. The two ends bend: the `+1` pushes the right end asymptotically to zero instead of
diving to `-inf`, and the `+0.5` flattens the left end at `ln(2(N+1))`. Reading the curve this way
gives an instant answer to "how much is this term worth": at `df/N` = 0.008 the value is 4.83; at 0.01
it is 4.60; at 0.1 it is 2.19; at 0.5 it is 0.68. The whole useful dynamic range of idf is compressed
into the decade from 0.1% to 10% of the corpus.

**The saturation curve as a hyperbola.** `f(tf) = tf(k1+1)/(tf+k1)` at `dl = avgdl` is a rectangular
hyperbola with asymptote `k1+1 = 2.2` and value 1.0 at `tf = 1`. Its slope at the origin is
`(k1+1)/k1 = 1.833`, which is the largest marginal value any occurrence can have. Geometrically, the
first occurrence earns 54.5% of the ceiling, the second brings the total to 62.5%, the third to 71.4%,
and the tenth to 89.3%. Any model that multiplies `tf` linearly is the straight line through the
origin tangent to nothing — it crosses the hyperbola near `tf = 0` and diverges above it, which is
exactly the keyword-stuffing exploit.

**Length normalisation as a lever arm.** Fix `tf` and vary `dl` from 1 to 1,023 tokens with
`avgdl = 95`. At `dl` = 1 the factor is 1.905; at 95 it is 1.375; at 190 it is 1.073; at 380 it is
0.746; at 1,023 it is 0.367. The ratio between the shortest and longest document is 5.19x for the same
`tf` — the lever is stronger than saturation over the same range, which is why `b` matters more than
`k1` on heterogeneous corpora and why `b` = 0.75 plus body `avgdl` silently demotes titles.

**Scores live on a per-query scale, not a global one.** Picture each query as its own ruler. For
`refurbished laptop battery` the maximum attainable score is `2.2 * (4.828 + 1.514 + 2.141) = 18.664`,
and 11.16 is 60% of it. For a query of three stopwords the maximum is `2.2 * (0.05 + 0.06 + 0.07) =
0.396`, and the same 11.16 is impossible. Comparing a score across queries is like comparing a
percentage of different wholes; comparing ranks is valid, comparing raw scores is not. This is why
"score > 8 is a good match" is always a bug report waiting to happen.

**The snapshot boundary as a plane.** Draw the write track below a horizontal line and the read track
above it. Every statistic crosses the line exactly once, at commit. A query that reads `df` from the
dictionary but `avgdl` from a live counter has two crossings at two times, and every correctness
argument that assumes one plane now fails. The mental image is useful in review: point at each
statistic and ask which crossing carried it.

```text
idf
16.6 |*                                    (df -> 0)
     | *
12.0 |  *
     |    *
 8.0 |       *
     |          *
 4.83|              *   <- refurbished, df/N = 0.008
     |                  *
 2.19|                       *   <- df/N = 0.1
     |                            *
 0.69|                                  *   <- df/N = 0.5
 0.0 |                                        *  (df -> N, never negative)
     +------------------------------------------------
       0.001 0.01  0.1   0.3  0.5  0.9  1.0   df/N
```

The plot is drawn on a log x-axis; the visible curvature at both ends is the `+0.5` and the `+1`. A
team that plots its own `df` distribution on this axis can see immediately how much of its vocabulary
sits in the flat right-hand region where idf carries no signal — usually the majority of tokens, which
is the quantitative case for stopword removal being a latency measure, not a relevance measure.

## 7. Comparison / Selection Table

**Index statistics strategy.** Choose one row before the first document is indexed; migrating later
requires a full rebuild.

| Strategy | What is stored | Freshness | Cross-shard comparability | Choose when |
|---|---|---|---|---|
| Global, recomputed on commit | exact `N`, `df`, `avgdl` per term | lag = commit interval | exact by construction | top-k merges across shards, SLA on reproducibility |
| Global, cached with TTL | same, refreshed on a timer | lag = TTL | exact within one TTL | read-heavy, small write rate, tolerance stated in minutes |
| Shard-local | per-shard `N`, `df`, `avgdl` | instant per shard | broken unless corrected | single-shard deployments, or scoring only inside a shard |
| Sampled | estimated from a document sample | cheap | biased per shard | never for a scoring path; acceptable for capacity planning |

**BM25 variant selection.**

| Variant | idf | tf shape | Failure it fixes | Cost it adds |
|---|---|---|---|---|
| Classic (no `+1`) | `ln((N-df+.5)/(df+.5))` | saturating | nothing | negative idf on boilerplate |
| BM25 with `+1` | `ln(1 + ...)` | saturating | negative contributions | none |
| BM25+ | `+1` | saturating, lower asymptote lifted | long documents under-scoring at low tf | one extra constant to tune |
| BM25L | `+1` | shifted by delta | systematic long-document penalty | re-tune `k1` and delta together |
| tf-idf | any | unbounded | nothing | keyword stuffing, no ceiling (`UB` = infinity) |

**Document length representation.**

| Encoding | Bytes/doc | Worst relative error in `f` | Notes |
|---|---|---|---|
| float32 exact | 4 | 0 | reproducible across replicas, 4x space |
| uint8 over `max_dl` 400 | 1 | 0.233% | clamping above 400 tokens loses all length signal there |
| uint8 over `max_dl` 1023 | 1 | 0.412% | wider range, coarser near the average |
| uint16 tokens | 2 | 0 | no clamping, still exact; the cheap correct choice |
| char count / 4 | 0 (derived) | up to ~20% on tables and code | a tokenisation assumption, not a length |

**Deletion handling.**

| Approach | `df` accuracy | Read-path cost | Risk |
|---|---|---|---|
| Filter tombstones at read | exact after filtering | one lookup per candidate | every consumer must filter; one miss leaks |
| Decrement `df` on delete | exact at delete time | none | needs a durable per-term counter or it drifts |
| Recompute `df` on merge | exact at merge | none | stale between merges; lag is the merge interval |
| Ignore deletions | stale by the deletion rate | none | scores drift up for rare terms; §8.1 |

The selection rule that survives review: pick the row whose *lag* you can state in the same sentence as
your freshness SLA. "Global statistics with a 60-second TTL" is a decision; "we compute BM25" is not.

## 8. Failure Modes & Edge Cases

Each entry below gives the symptom as a user or operator observes it, at least two competing causes,
the check that separates them, and the correction. The competing-cause discipline matters because the
first plausible explanation is wrong often enough to be a bias.

### 8.1 Rare-term documents climb in rank after a bulk delete

**Symptom.** After archiving 8% of the catalogue, scores for rare terms rise a few percent and the top
ten reorder, with no relevance change to any surviving document.

**Competing causes.** (a) `df` was not decremented while `N` was, so idf reads a smaller corpus with an
unchanged numerator; (b) `df` and `N` both stale, which is harmless (section 5, audit 3); (c) a
changed `avgdl` because deleted documents were shorter than average, which raises `dl/avgdl` and
*reduces* scores for survivors.

**Discriminating check.** Query the dictionary for ten terms and compare `df` against a count taken
directly from the postings after tombstone filtering. If the counted value is 92% of the dictionary
value, cause (a). Then recompute `avgdl` from live documents; if it moved, cause (c) is co-present and
its sign is opposite, which also explains a muted net effect.

**Correction.** Recompute `df`, `N`, and `avgdl` in one commit so the snapshot is internally
consistent; make the statistics commit the same transaction as the tombstone write.

**Prevention and verification.** Emit a nightly invariant metric `sum over terms of df` versus the
recount on a sample of 1,000 terms; a sustained ratio above 1.005 blocks the release.

### 8.2 The same document ranks differently on two replicas

**Symptom.** A/B users see different orderings for an identical query, and replaying the query twice
on the same replica returns the same order, so it looks like assignment noise.

**Competing causes.** (a) shard-local statistics, so the merge compares idf from different `N`; (b)
different norm widths or analyser versions between replica generations; (c) non-deterministic tie
breaking in the top-k heap under equal scores.

**Discriminating check.** Score one document directly on each shard with a debug endpoint and compare
`idf` and `f` terms separately. If the idf values differ by the shard size ratio, cause (a): with
`df` = 20,000 on a 2,000,000-document shard, `idf` = 4.6051 against 4.8283 globally, and shards with
`df` = 12,000 give 5.1160, so a factor of 1.11 separates the extremes. If idf agrees and `f` differs,
cause (b). If both agree, cause (c).

**Correction.** Distribute global statistics to all shards, keyed by a statistics version, and include
that version in the query plan; make the heap comparator total by breaking ties on docID.

**Prevention and verification.** A canary query set with pinned expected scores, run against every
replica on deploy; any deviation above 0 is a page, because exact reproducibility is the invariant.

### 8.3 Scores drop a few percent after a merge or reindex

**Symptom.** A routine background merge completes; nothing was edited; every score for long documents
moves down by roughly 4–10%.

**Competing causes.** (a) `avgdl` was recomputed and the previous value was wrong, usually because it
was a mean of segment means rather than a document-weighted mean; (b) `dl` is now counted on a
different analyser (a new stemmer merging two tokens into one); (c) norms were requantised with a
smaller `max_dl`, so very long documents now clamp.

**Discriminating check.** Recompute `avgdl` two ways from segment metadata: unweighted mean of segment
averages, and `sum(segment_tokens)/sum(segment_docs)`. In a constructed four-segment case with sizes
3M/2M/2M/1M and means 88/95/101/112, the unweighted mean is 99.0 and the weighted mean 96.0 — a 3.1%
difference that alone produces a ~1% score shift for average-length documents and more for long ones.
Clamping is caught by querying the ten longest documents and checking whether their `dl` reads as
exactly `max_dl`.

**Correction.** Compute `avgdl` from summed tokens over summed documents; store `dl` and the analyser
version together; choose `max_dl` at the 99.9th percentile of measured length rather than at a round
number.

**Prevention and verification.** Persist `N`, `sum(dl)`, and the analyser hash in the segment footer and
assert `sum(dl)/N == avgdl` to within `1e-9` on load.

### 8.4 A document matching all three query terms ranks below one matching two

**Symptom.** The result looks like the scorer ignored a term, and the offending documents usually
contain a very common phrase.

**Competing causes.** (a) classic idf with `df > N/2`, contributing negative values; (b) `tf` clamping
at a 16-bit field, which is not zero and cannot explain a *negative*; (c) the third term's clause is
pruned by an unsound upper bound that assumed a non-negative contribution.

**Discriminating check.** Print per-clause contributions for both documents. A negative number
identifies (a) immediately: with `df` = 7,600,000 the classic contribution is `-2.944` at `tf` = 1 and
`-4.627` at `tf` = 3, so repetition makes it worse. If all three contributions are positive and the
losing document is missing a clause entirely, cause (c): check whether the clause's `UB` was computed
as `idf * (k1+1)` with a negative `idf`, which makes `UB` negative and the clause skippable.

**Correction.** Use the `+1` idf form; alternatively clamp the classic form at zero, but note that
clamping changes the model (a term in most documents becomes neutral rather than informative) and must
be documented as such.

**Prevention and verification.** Assert `idf >= 0` for every dictionary term at commit time; a
negative value fails the build rather than waiting for a query.

### 8.5 The top-k is fast and occasionally wrong on long queries

**Symptom.** Recall against a labelled set drops as the query grows past about six terms, and the loss
is concentrated in documents that match every term.

**Competing causes.** (a) block-max metadata computed from per-term maximum impacts while the score
sums several terms, so a block's stored maximum is the max of parts rather than the max of the sum; (b)
`k1` changed without recomputing the stored bounds; (c) the bound was tightened to `idf * f(tf_max)`
where `tf_max` is the observed maximum at build time rather than a proven ceiling.

**Discriminating check.** Disable pruning and re-run the failing queries. If recall returns to the
labelled value, the bound is unsound; then recompute the bound as `idf * (k1 + 1)` and confirm it
dominates every observed per-document contribution across the block. Cause (b) shows as a bound that
is still "correct-looking" but smaller than the recomputed one by the ratio of the k1 values' ceilings.

**Correction.** Store the bound as `idf * (k1 + 1)` with the `k1` value recorded beside it, and
invalidate bounds when `k1` or `b` changes; never derive a bound from observed data.

**Prevention and verification.** A nightly exact-versus-pruned comparison on 1,000 sampled queries; any
set difference is a release blocker.

### 8.6 A term's `df` is twice what the postings contain

**Symptom.** idf for a specific term is lower than a manual count implies, and the discrepancy appears
only for terms in recently replicated or re-ingested content.

**Competing causes.** (a) two segments hold the same documents because a re-ingest appended rather than
replaced; (b) `df` summed across replicas instead of taken from a canonical shard; (c) `df` includes
tombstoned documents that the postings filter out at read time.

**Discriminating check.** Count distinct docIDs, then count posting entries. Equal distinct counts with
a larger dictionary `df` points to (c) or (b); a distinct count equal to the dictionary `df` points to
(a), and the duplicate docIDs will be visible in a raw dump. Doubling `df` from 64,000 to 128,000 moves
idf from 4.828 to 4.135, a 14.4% loss, which is large enough to reorder a tight result set.

**Correction.** Make ingest idempotent on a content key; take `df` from one canonical statistics owner
per term; exclude tombstones before counting.

**Prevention and verification.** Assert `df <= N` per term and `sum(df over query terms) <= sum(N, ...)`
sanity bounds on every statistics publish; alert on any term whose `df` changes by more than 20% in one
commit without a corresponding ingest volume.

### 8.7 Edge case: an empty field scores as if it were average

**Symptom.** Documents with no body text rank normally for body terms they cannot possibly contain.

**Competing causes.** (a) `tf = 0` clauses are being added as zero instead of omitted, which is
harmless, so the real cause must be elsewhere; (b) `dl` stored as 0 but read as a default, so `K` is
computed from `avgdl` instead of from 0; (c) a copy-field or catch-all field indexing the title into the
body.

**Discriminating check.** For the failing document, read the norm back. `dl = 0` with any non-zero
contribution proves (b); a non-zero body `dl` on a document with no body proves (c). Note that `dl = 0`
with `b < 1` gives `K = k1(1-b) = 0.3`, so a spurious `tf = 1` would score 1.692 — the largest
single-occurrence value available — which is why the default must never be a positive length.

**Correction.** Store `dl = 0` for empty fields, skip the clause, and compute `avgdl` over documents
that have the field.

**Prevention and verification.** A schema assertion that every document has at least one analysed token
in its primary field, enforced at ingest.

### 8.8 Edge case: the same text indexed twice doubles `tf` and `dl`

**Symptom.** One product variant consistently outranks its twins, and its `dl` is about double what its
text implies.

**Competing causes.** (a) two analysers on one field (for example a stemmed and an unstemmed
sub-field) both contributing to the same `tf`; (b) the document genuinely repeats the description,
which is a data problem rather than an index problem; (c) `dl` counted on a concatenation of title and
body while the query scores the body field only.

**Discriminating check.** Compare `tf` from the index with `tf` from a re-tokenisation of the stored
source. Equal means the source repeats; double means the field is indexed twice in the same coordinate
system. Then check `dl` against a fresh tokenisation of the body alone — a body `dl` equal to the
title+body count identifies (c).

**Correction.** Score per field with per-field statistics; if a field must be indexed twice for
matching reasons, exclude the duplicate from the scoring field's statistics.

**Prevention and verification.** On a sample of 100 documents daily, re-tokenise and assert
`|dl_index - dl_source| <= 0` for the scoring field.

## 9. Complexity & Cost Analysis

Complexity has four independent axes here, and quoting one hides the one that binds: list lengths
(`df`), the number of query terms (`k`), the per-posting work inside the accumulator, and the merge
interval of the write track. Report all four or the capacity plan will be wrong in a specific and
predictable direction.

**Scoring time.** Document-at-a-time scoring over `k` clauses with a cursor-merge is `O(sum_i df_i)`
cursor advances plus `O(D_cand * k)` score accumulations, where `D_cand` is the number of candidate
documents after the first clause. With `k` = 3 and the section 5 lists, `sum df` = 2,764,000 and the
constructed per-core throughput of 100,000,000 entries/s gives 27.64 ms. The accumulator work is
`D_cand * 3` operations; at `D_cand` = 500,000 that is 1.5 million multiplies, roughly 1.5% of the
cursor work, which is why scoring arithmetic is rarely the bottleneck and decoding is.

**Decoding cost.** Compressed postings decode at a rate bounded by the codec, not by the CPU's multiply
throughput. At a constructed 250 MB/s per core, the 827.4 MB of doc-level postings for a full scan take
3.3 s; the 722.0 MB of positions take 0.80 s at a faster position codec. This is the real argument for
block-max pruning: it is not that comparisons are expensive, it is that decoding is, and a pruned block
is never decoded.

**Pruning benefit.** With blocks of 128 postings, the `laptop` list has 1,760,000/128 = 13,750 blocks.
Pruning at a threshold set by the current tenth-best score typically skips 60–75% of blocks on a
multi-term query; at 70% the decode work for that clause falls to `0.30 * 827.4 * (1.76/2.764)` = 158
MB from 527 MB, a 3.3x saving on the clause. The saving is bounded by the bound's tightness — 18.664
versus 11.160 in section 4.2 is a 1.67x slack, and slack directly reduces the skip rate.

**Space.** `space = E*w_d + T*w_p + V*w_v` for posting entries, positions, and vocabulary. The ratio
positions:postings is `(T/E)*(w_p/w_d)` = `1.24 * 0.704` = 0.873, a corpus property that is invariant
to corpus size for a fixed tokenisation and almost invariant to document length. Doubling `N` doubles
both terms and leaves the ratio unchanged, so "positions cost 87% on top" generalises where "1.5 GB"
does not.

**Write-side cost.** Posting and position writes are linear in tokens: 760,000,000 tokens, one encode
per token. The dominant cost is merging, because a merge rewrites positions and renumbers docIDs rather
than copying bytes. A merge policy with fan-out `f` rewrites each byte `log_f(segments)` times, so a
fan-out of 10 over 1,000 segments rewrites each byte 3 times; halving the fan-out to 5 adds one more
rewrite for a 20% I/O increase. That is the number that belongs in a rebuild budget, and it is absent
from every plan that counts only final index size.

**Statistics cost.** Recomputing `df` exactly requires a pass over the affected terms' lists: for the
three query terms that is 2,764,000 entries, or 27 ms of decode. Recomputing it for the whole
vocabulary requires a full pass over `E` = 612.9 million entries, roughly 2.5 s of decode per pass at
250 MB/s — cheap enough to do on every significant commit, which removes the entire class of staleness
failures in section 8.1. The reason teams do not do it is a write-path architecture that cannot afford
a stop-the-world scan, and the fix is incremental `df` maintenance rather than sampling.

**Caching.** Doc-level postings cache well; positional postings do not, because the bytes touched per
query are larger by the position multiplier for the same `df`. On a node with a fixed page cache, an
index that fits without positions may not fit with them, converting cache hits into disk reads. The
countermeasure is field scoping and pruning, not more RAM: 87% more bytes cannot be bought back by a
cache policy that is already full.

**Reproducibility cost.** Distributing one global statistics snapshot to every shard adds a
synchronisation point per commit and a version check per query. The measured price is one extra read of
a small object (kilobytes) per query plan and a bounded staleness equal to the publish interval. The
alternative — shard-local statistics — removes that cost and pays with the 1.11x idf spread measured in
section 8.2, which is a ranking defect that no amount of hardware fixes.

## 10. Key Theorems / Results Reference

Each result is stated with the condition under which it holds, because a formula quoted without its
condition is the origin of most of the failures in section 8.

**R1 — With the `+1` idf, the idf of every legal term is strictly positive.** `idf = ln(1 + r)` with
`r = (N-df+0.5)/(df+0.5) >= 0` for `0 <= df <= N`, so `idf > 0` and `idf -> 0` as `df -> N`. The
maximum is `ln(2(N+1))` at `df = 0`. Consequence: no legal match can reduce a document's score, which
is what makes score-based pruning with `UB = idf*(k1+1)` sound.

**R2 — The classic idf changes sign at `df = (N+1)/2`.** Solving `(N-df+0.5)/(df+0.5) = 1` gives
`df = N/2`. Beyond that point the contribution is negative and repetition (`tf`) amplifies the
negative. This is a *model* discontinuity, not a numerical edge case, and it is why the `+1` variant
became the default everywhere.

**R3 — The BM25 tf factor is bounded by `k1+1` and increasing in `tf`, with `f(tf=0) = 0`.** The
derivative is `(k1+1)*K/(tf+K)^2 > 0` for `K > 0`, and the limit is `k1+1`. Monotonicity plus a
ceiling is exactly the pair of properties a block-max index needs; an unbounded tf factor has no finite
upper bound and therefore cannot be pruned safely.

**R4 — Half-saturation occurs at `tf = K`, i.e. `tf = k1` when `dl = avgdl`.** Setting
`tf/(tf+K) = 1/2` yields `tf = K`. This gives the operational meaning of `k1`: it is the repetition
count at which the term has earned half of its available headroom, in an average-length document. A
document twice the average length needs `tf = k1*(1 + b)` to reach the same fraction.

**R5 — At `dl = avgdl` the tf factor reduces to `tf(k1+1)/(tf+k1)`, independent of `b`.** Because the
denominator becomes `k1*(1-b+b) = k1`. This is the invariant to assert in a unit test: it catches
sign errors in the `1-b+b*dl/avgdl` expression that are otherwise invisible on short test corpora.

**R6 — The score is not invariant to corpus growth, but rank order is invariant to uniform rescaling.**
`idf` embeds `N`, so absolute scores change when the collection grows. If every `df` scales with `N`,
every idf is unchanged (section 5, audit 3) and ranks are stable. If `df` scales with `N` but the
statistics are updated in different commits, ranks are not stable — the mixed-snapshot case, which is a
snapshot defect rather than a property of the formula.

**R7 — Uniform length quantisation preserves score order.** If `q(dl)` is monotone non-decreasing then
`dl1 < dl2` implies `f(dl1) >= f(dl2)`, so quantisation can create ties but cannot invert a pair that
differed only by length. This is why one-byte norms are safe for ranking and unsafe only for exact
reproduction across heterogeneous replicas.

**R8 — `avgdl` computed as a mean of segment means is biased toward small segments.** The unweighted
mean of segment averages weighs a 1,000-document segment as heavily as a 3,000,000-document one. The
correct estimator is `sum(tokens)/sum(documents)`; in the constructed four-segment case the two differ
by 3.1% (99.0 versus 96.0), which propagates into every score.

**R9 — Per-field `avgdl` is required for per-field scoring to be unbiased.** With a 9-token average
title, a 6-token title at `tf` = 1 earns `f` = 1.158 under title statistics and 1.611 under body
statistics — a 39% inflation for every short field. Field statistics are a precondition, not a tuning
choice.

**R10 — `df` is the durable, `tf` is the ephemeral quantity.** `df` lives in the dictionary and survives
document deletion until recomputation; `tf` is rewritten whenever a posting is written. Any staleness
bug therefore has a characteristic direction (rare terms over-scored after deletion) that distinguishes
it from fresh-`tf` effects.

**R11 — A top-k bound is sound only if it dominates the true per-clause maximum.** For BM25 with the
`+1` idf, `idf*(k1+1)` dominates because `f < k1+1`; deriving a bound from observed maxima at build
time does not dominate, because a later document can exceed it. Soundness is provable; tightness is
measurable; the two must not be conflated when a change is proposed to improve latency.

**R12 — Reproducibility requires one statistics snapshot per query, and is testable independently of
relevance.** Two replicas with identical postings, analyser, `k1`, `b`, and statistics version must
return identical scores. This is a strong, cheap invariant and it is the correct first test in any
index incident, because it partitions the space: if replay differs, the problem is configuration or
snapshot; if replay agrees, the problem is the statistics themselves.

## 11. Interview / Exam Questions

**Q1 (recall).** Name the five statistics BM25 reads, and for each, name the process that writes it.

Answer: `N` is written by the commit that publishes collection statistics from the live document count;
`df` is written by the dictionary when a posting list is created or extended, and must be decremented on
delete; `tf` is written per posting at index time; `dl` is written into the norm when the document is
indexed and rewritten on merge; `avgdl` is written by the same statistics commit as `N`, from the sum
of `dl`. The point of the question is that two of the five are aggregates over the whole collection and
two are per-document, so they have different owners and different staleness budgets. A candidate who
lists the five without the ownership distinction has memorised the formula but not the failure mode.

**Q2 (calculation).** A term appears in 12,000 of 2,000,000 documents. Compute idf both ways and say
which changes a ranking decision.

Answer: plus-one form: `(2,000,000 - 12,000 + 0.5)/12,000.5 = 165.66`; `1 + 165.66 = 166.66`;
`ln = 5.116`. Classic: `ln(165.66) = 5.110`. The difference is 0.1%, irrelevant. The decision-relevant
difference appears at high `df`, not low: at `df` = 1,760,000 of 8,000,000 the two are 1.514 and 1.266,
a 16.4% gap, and at `df` = 7,600,000 the classic value is `-2.944`, which reverses the sign of a match.
So the answer to "which changes a ranking decision" is: the idf variant matters only in the upper tail
of `df`, which is exactly where boilerplate lives.

**Q3 (diagnosis).** A query for a product name returns the product in position 4 on shard A and
position 1 on shard B. What do you check, in order, and what number tells you it is the cause?

Answer: first compare the idf terms for the query's rarest term on both shards. If shard A computes
`df` = 20,000 over `N` = 2,000,000 and shard B computes `df` = 12,000 over 2,000,000, the idfs are 4.605
and 5.116, an 11.1% spread, and everything else can be identical and the merge will still reorder. If
idf agrees, compare `f`, which isolates norm or analyser differences; if both agree, compare the tie
break in the heap. The order matters because fixing the statistics is a configuration change while
fixing the heap is a code change, and investigating them in the wrong order wastes an on-call rotation.

**Q4 (calculation).** With `k1` = 1.2 and `b` = 0.75, a document of 190 tokens contains a term twice.
What fraction of the term's ceiling does it earn, and how does that compare with a 47-token document
containing it once?

Answer: `K = 1.2*(0.25 + 0.75*2) = 2.1`, so `f = 4.4/(2 + 2.1) = 1.073`, i.e. 48.8% of the 2.2 ceiling.
The 47-token document: `K = 1.2*(0.25 + 0.75*0.4947) = 0.745`, `f = 2.2/(1 + 0.745) = 1.261`, 57.3% of
the ceiling with a single occurrence. So the short document earns more of the ceiling from one
occurrence than the long one does from two, and the ratio of scores is 1.261/1.073 = 1.175. The
follow-on conclusion is that on this corpus `b` dominates `k1` for ranking, so an experiment that
changes `k1` and reports "no effect" has tested the wrong knob.

**Q5 (judgment).** You are asked to accept "score above 8 means the result is good" as a regression
test. What is your response?

Answer: reject it, because a score is only meaningful inside one query, one collection, and one
statistics snapshot. For a three-term query on 8,000,000 documents the ceiling is
`2.2 * sum(idf)`, which for rare terms is 18.7 but for common ones can be under 1.0; a threshold of 8
is unreachable for the second query and trivially met for the first. Replace it with rank-based
assertions (document X in top 3 for query Y) and exact-score assertions on a pinned snapshot with a
frozen analyser, `k1`, `b`, and statistics version. The pinned test catches implementation drift; the
rank test catches relevance drift; neither can be replaced by the other.

**Q6 (diagnosis).** Recall drops 6% after a merge, and the index size also dropped 2%. Give two
mechanisms consistent with both facts and the check that separates them.

Answer: mechanism one is norm requantisation: a smaller `max_dl` shrinks norms (2% of total size is
plausible for a norm column) and clamps long documents, which lowers their scores and removes them
from top-k. Mechanism two is an analyser change that folds tokens, which shrinks both positions and the
dictionary and changes `dl`. Separate them by reading back a known long document's `dl`: a clamped value
equal to `max_dl` on many documents proves requantisation; a changed `dl` on documents whose text did
not change proves the analyser. Both are consistent with size shrink, which is why size is not
diagnostic on its own.

**Q7 (calculation).** How many probe entries does a two-term query cost if you score by galloping from
the rarer term, given `df` = 64,000 and 1,760,000, at the far end of the long list?

Answer: each anchor position costs a galloping search in the long list. At `j` near 2,000,000 the
bound `2*ceil(log2(j+1)) + 1` is `2*21 + 1 = 43` probes. With 64,000 anchors and two lookups each, that
is `64,000 * 2 * 43 = 5,504,000` probes, versus 2,764,000 cursor advances for a full merge. So here the
merge is cheaper by 2.0x, and galloping wins only when the anchor is far rarer relative to the longest
list — the crossover is where `df_anchor * log2(df_max)` falls below `sum(df)`. Candidates who quote
galloping as universally better than merging are quoting a textbook case, not a cost model.

**Q8 (diagnosis).** idf for a term is 14.2 on a 9,000,000-document corpus. Is anything wrong?

Answer: 14.2 is possible but suspicious. `idf = ln(1 + (N-df+0.5)/(df+0.5))`; solving for `df` with
`N` = 9,000,000 gives `df` near 0.5, i.e. a term in one or zero documents. That is legal for a
hash-like token, an ID, or a rare misspelling. The check is `df` from the dictionary against a direct
count: if `df` = 1 and the term appears in two documents, either the second document was indexed after
the statistics snapshot or the term is analysed differently in the two documents. If `df` reads 0, the
term has a dictionary entry with no postings, which is a merge artefact.

**Q9 (calculation).** Positions add what fraction to index size here, and what does that fraction
depend on?

Answer: with `T` = 760,000,000 positions at 0.95 bytes and `E` = 612,903,226 entries at 1.35 bytes,
positions are 722.0 MB against 827.4 MB of postings, i.e. +87.3%. The ratio is `(T/E)*(w_p/w_d)` =
`1.24 * 0.704`, so it depends on occurrences per posting pair (a corpus property), on the two encoding
widths (a codec choice), and not on document length or corpus size. That invariance is why "+87%" is a
transferable number while "722 MB" is not; the same corpus at double size would need 1,444 MB of
positions and still +87%.

**Q10 (judgment).** Your team wants to halve the index by dropping positions. What breaks, and what
would you measure first?

Answer: phrase and proximity matching break, because adjacency is undecidable without ordinals — not
approximate, undecidable, and most engines degrade to a conjunction silently rather than erroring.
Before deciding, measure the share of queries carrying a quoted phrase or a proximity operator, and the
share of those that are satisfied by adjacency in the top 10. If phrases are 4% of traffic and 70% of
those return an adjacent match, dropping positions removes a visible capability for a 47% size saving
(the 87.3% overhead halves the total only when the base is counted correctly: 1,483 MiB -> 761 MiB).
Also measure the rebuild time saved, which usually dominates the disk saving in real terms.

**Q11 (diagnosis).** Scores are exactly reproducible, ranking looks reasonable, but a labelled
evaluation says recall is 30 points below the previous index. Where do you look?

Answer: at the candidate set, not the scorer. Exact reproducibility plus plausible ranking is the
signature of a *filter* change: an analyser, a stopword list, a synonym map, or a clause added to the
query. The check is per-query: take ten labelled queries, log the number of candidates before scoring,
and compare with the previous index. If candidate counts fell, the scorer is exonerated and the
analyser or filter owns the regression. This is worth asking because teams routinely re-tune `k1` and
`b` to fix a recall collapse that has nothing to do with scoring.

**Q12 (calculation).** A reindex doubles `N` while every `df` doubles too. What happens to scores, and
what happens if only `N` doubles?

Answer: if both double, `(N-df)` terms grow proportionally and `idf` is unchanged for every term, so
scores are identical and the reindex is invisible. If only `N` doubles, every `df` is now a smaller
fraction of the collection, so every idf rises: the rare term goes from 4.828 to 4.905 (a 1.6% rise at
`df` = 64,000 and `N` = 8,640,000) and common terms rise more in relative terms because idf is
sub-linear. The observable is uniform score inflation with unchanged ordering among documents — a
snapshot artefact, not a relevance change, and the correct response is to republish `df`, not to
retune.

**Q13 (recall).** Why is `b` in `[0,1]` and what does `b = 1` mean operationally?

Answer: `b` weights the deviation of `dl/avgdl` from 1 inside `K = k1(1 - b + b*dl/avgdl)`. At `b` = 0,
`K = k1` and length is ignored entirely; at `b` = 1, `K = k1*dl/avgdl`, so a document twice the average
length needs twice the `tf` for the same factor. The operational statement of `b` = 1 is that
repetition and length are perfectly traded: a long document is not more likely to match merely by being
long. Values above 1 are meaningless because `K` could reach 0 or negative for short documents, which
would make the denominator non-monotone and break the pruning bound.

**Q14 (diagnosis).** Per-clause contribution for the top document is 4.83 + 0 + 0, and the second
document is 2.1 + 2.1 + 1.9. Which ranks first, and what does that reveal about the query?

Answer: the first ranks first at 4.83 versus 6.1 for the second — no, 2.1 + 2.1 + 1.9 = 6.1, so the
second document ranks first, and the reveal is that the first document matched only the rarest term
while the second matched three moderately rare ones. That is the correct behaviour of a sum of positive
contributions, and it is what a team usually wants to change — but the change belongs in the query
(require two of three terms) or in a rank-fusion weight, not in BM25. If the team expected otherwise,
they were implicitly assuming a "must match the rarest term" filter that the scorer never promised.

**Q15 (judgment).** You must choose between exact global statistics with a 60-second publish lag and
shard-local statistics with zero lag. Which do you ship, and what do you write down?

Answer: ship global with the lag, because the lag produces a bounded, monotone score shift that is
identical on every replica, while shard-local statistics produce an unbounded ranking defect whose
magnitude depends on the shard-size ratio (1.11x idf spread in the constructed case). Write down three
numbers: the publish lag in seconds, the maximum expected idf error during the lag (a few percent at an
8% churn rate), and the canary query set with pinned expected scores. The decision is defensible
because reproducibility is an invariant you can test and staleness is a quantity you can bound;
shard-local scoring inverts that.

## 12. Quick Reference Card

| Item | Value / rule | Why it matters |
|---|---|---|
| idf (use this) | `ln(1 + (N-df+0.5)/(df+0.5))`, always > 0 | classic form goes negative at `df > N/2` |
| idf range | `(0, ln(2(N+1))]`; 16.588 at `N` = 8M, `df` = 0 | a target absolute score pins `N` |
| tf factor | `tf(k1+1)/(tf+K)`, `K = k1(1-b+b*dl/avgdl)` | ceiling `k1+1` makes pruning sound |
| half saturation | `tf = K`, i.e. `tf = k1` at `dl = avgdl` | the operational meaning of `k1` |
| sanity test | `dl = avgdl` must give `f = tf(k1+1)/(tf+k1)` | catches `1-b+b*dl/avgdl` sign errors |
| upper bound | `UB = idf*(k1+1)` | never derive a bound from observed maxima |
| positions cost | `(T/E)*(w_p/w_d)`; 0.873 here | invariant to corpus size, not to codec |
| staleness | publish `N`, `df`, `avgdl` in one commit | mixed snapshots bias rare terms |
| reproducibility test | replay on every replica, compare exact scores | partitions config bugs from model bugs |
| length normalisation | `b` beats `k1` on heterogeneous corpora | 5.19x span at `b` = 0.75, `dl` 1..1023 |
