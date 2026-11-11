## 8. Failure Modes & Edge Cases

### 8.1 How to use this section

Each failure below is written so that a reader with production access can move from a symptom to a test that discriminates between competing causes. The discipline matters because the obvious explanation is usually wrong: a drop in click-through is attributed to the ranker when candidate recall fell, and a spike in empty slates is attributed to a catalog problem when a filter was applied after truncation.

The structure of each entry: **symptom** (what the alert says), **competing causes** (at least two, because a single-cause failure needs no diagnosis), **discriminating check** (a query or comparison that separates them and whose outcome is observable today), **correction** (what to change), **prevention** (what makes the class of failure impossible). Failures that share a symptom are grouped so the checks can be read against each other.

### F-01 — Revoked rights reach a display

**Symptom:** a policy audit finds a title rendered after its rights window closed. No error was logged and latency was normal.
**Competing causes:** (a) the cached slate was reused across requests without re-evaluating eligibility; (b) the index refresh ran but the served replica was the pre-refresh one; (c) the revocation was ingested with a future effective timestamp and the request time comparison used processing time instead of effect time.
**Discriminating check:** for the specific request, print the policy version and effective timestamp observed by the final check, and compare with the revocation record's effective timestamp. If the final check observed the post-revocation version, cause is not (a). If the replica identity in the response header is stale, cause is (b).
**Correction:** move the authoritative eligibility check to the last responsible moment before render, and make the check return a versioned decision that is logged with the response.
**Prevention:** an invariant test in staging that revokes a title mid-request and asserts the item is absent from the rendered slate, run on every deployment.

### F-02 — Filtered ANN underfills while eligible items exist

**Symptom:** slates with fewer items than slots, concentrated in narrow territories; recall unchanged on the global cohort.
**Competing causes:** (a) depth too small for the local density; (b) the filter is applied after truncation instead of during traversal; (c) the eligible set genuinely is small for that segment.
**Discriminating check:** compare |C(q)| against the maximum eligible relevant count over the same segment. If a filter-aware exact oracle on the same query returns 20 items regularly, cause (c) is excluded and the defect is in depth or in filter placement. Then check whether increasing depth from 20 to 400 changes the delivered count: if it does, cause is (a) or (b); if the delivered count stays at zero across all depths, the filter predicate is wrong.
**Correction:** provision depth from [D1] using the segment's measured density, and push the predicate into traversal so rejection does not consume the depth budget.
**Prevention:** a per-segment underfill dashboard with the eligible-set size as the companion series, so a small catalog and a broken filter never look alike.

### F-03 — Offline recall spikes after a label change

**Symptom:** recall@20 rises 12 points overnight with no deployment.
**Competing causes:** (a) the label horizon was shortened, shrinking the denominator; (b) the label rule's threshold changed (for example a minimum watch time); (c) a logging change increased exposure capture, adding labelled items that were previously invisible; (d) a genuine retrieval improvement shipped silently in a config change.
**Discriminating check:** recompute recall on the frozen cohort with the old and new label definitions. If the number moves with the definition, the metric moved, not the system. Then compare the denominator count: a rise in recall at constant numerator means the denominator fell, which points at (a) or (b), never at (d).
**Correction:** version the label rule alongside the model, and store the rule identifier in the evaluation output.
**Prevention:** make the label rule a first-class artefact with a version that appears in every metric's metadata; refuse to compare metrics whose label versions differ.

### F-04 — New viewer receives an empty slate

**Symptom:** first-session viewers get zero recommendations on the primary surface.
**Competing causes:** (a) the personalisation source requires an interaction history and returns nothing, and there is no fallback; (b) the fallback exists but its own eligibility filter removed everything because it keyed on a viewer attribute that is null for new accounts; (c) the request is malformed at the edge and never reaches retrieval.
**Discriminating check:** count requests by stage reached and by source-attempted outcome code. A request that reached the popularity source and got zero items after filtering points at (b); a request with no source attempts points at (c); a request with attempts and an explicit "insufficient history" code points at (a).
**Correction:** implement a tiered fallback: contextual, then locally eligible popularity, then editorial, each with its own eligibility filter that does not depend on the missing attribute.
**Prevention:** a synthetic new-account fixture in staging that must produce a full slate; alert on the share of requests whose slate is shorter than the slot count, split by account age.

### F-05 — New items never receive exposure

**Symptom:** items ingested in the last 14 days account for under 1% of impressions despite being 4% of the catalog.
**Competing causes:** (a) no retrieval source can propose an item without interactions, so content similarity is missing or unused; (b) exploration quota is set to zero; (c) the item is eligible but its embedding is a zero vector because the feature pipeline had not populated it at ingest; (d) the eligibility predicate rejects it for a legitimate reason such as an unapproved asset.
**Discriminating check:** for a sample of unexposed new items, compute their presence in each source's output when queried with a synthetic probe vector. Present in content source but never in slates points at (b). Absent from every source with a zero-norm embedding points at (c). Absent from the candidate pool entirely with a policy rejection code points at (d).
**Correction:** add a content-based source with its own quota, backfill embeddings before the item becomes eligible, and instrument a per-item "proposed count" counter.
**Prevention:** make "eligible but never proposed" a tracked state with an alert, because it is invisible in every outcome metric.

### F-06 — Approximate index degrades after a parameter change

**Symptom:** recall@20 drops 3 points two days after an index rebuild. No model changed.
**Competing causes:** (a) the rebuild used different graph parameters, changing fidelity; (b) the rebuild used a different embedding version because the artifact pointer moved; (c) the rebuild dropped vectors for a shard that failed and served from a smaller index.
**Discriminating check:** compute [A1] fidelity of the new index against a float32 exact oracle on a frozen query set. Fidelity down with identical vectors points at (a). Fidelity normal but score distribution shifted points at (b). Vector count in the index versus the catalog's eligible count points at (c).
**Correction:** pin the embedding version and index parameters in the build manifest; fail the build if the vector count deviates from expectation beyond a tolerance.
**Prevention:** store a fidelity report as a build artefact and gate promotion on it, so an index cannot reach production unmeasured.

### F-07 — Recall is high and outcomes are flat

**Symptom:** candidate recall improves from 0.38 to 0.48, and the online primary metric does not move.
**Competing causes:** (a) the ranker had already been saturating on the relevant items that were present, so extra candidates are redundant; (b) the added candidates are relevant under the offline label but not under the online objective (a label/objective mismatch); (c) the added candidates arrive late in the slate and are never examined; (d) the traffic split was not honoured, so the comparison is between cohorts that differ in ways unrelated to the treatment.
**Discriminating check:** compute ranker retention — the share of relevant candidates that reach the slate — before and after. If retention was already near 1.0, cause is (a). Then check the exposure-weighted position distribution of newly added relevant items; if they land below the last examined slot, cause is (c). Sample-inspect arm assignment against the configured split to exclude (d).
**Correction:** if (a) holds, the constraint is the ranker or the slate, so improving retrieval further has no value; re-target the work at ranking or at slot count.
**Prevention:** declare the binding constraint before starting work: candidate recall below roughly 0.6 with high ranker retention means retrieval is binding; the reverse means ranking is.

### F-08 — Two services disagree about eligibility

**Symptom:** an item appears in retrieval output and is rejected at display, at a rate of 9% of candidate rows.
**Competing causes:** (a) the two services read different policy sources, one of which lags; (b) they evaluate the predicate at different timestamps within the same request; (c) one service applies a default-allow on a policy lookup failure while the other applies default-deny.
**Discriminating check:** join both services' decision records for the same request and diff the policy version, the evaluated timestamp, and the failure code. A version mismatch points at (a); identical versions with timestamps a second apart points at (b); a failure code on one side with an allow decision points at (c).
**Correction:** make one service own the decision and have the other consume its signed result within a bounded age; treat any lookup failure as deny at both.
**Prevention:** a contract test that asserts both services return identical decisions for a corpus of constructed (item, viewer, timestamp) triples, run on every deploy of either.

### F-09 — A minority cohort sees poor coverage

**Symptom:** one language cohort has 30% fewer distinct items per week than the platform average.
**Competing causes:** (a) the cohort's eligible catalog is genuinely smaller; (b) the embedding model was trained on the majority cohort's behaviour and places the cohort's items far from its users' vectors; (c) a fallback source has no items in that cohort's language and dominates the slate.
**Discriminating check:** compare eligible catalog size per cohort first. If comparable, compute coverage — distinct items exposed over distinct eligible items — for each source separately, which isolates (c) from (b).
**Correction:** for (b), evaluate a per-cohort content source, and report recall per cohort rather than pooled so the deficit is visible.
**Prevention:** make cohort-stratified recall a standard output of the evaluation job, with a minimum sample per cohort, so pooled averages cannot hide a cohort at zero.

### F-10 — Duplicate items crowd the candidates

**Symptom:** the ranker sees 20 candidates but only 13 distinct product concepts; the slate looks repetitive.
**Competing causes:** (a) the same item ID appears from several sources and dedup keys on the row, not the ID; (b) variants (colour, size, edition) are distinct IDs but one concept, and dedup has no concept key; (c) the sources themselves contain duplicates because the upstream feed did.
**Discriminating check:** count rows, distinct item IDs, and distinct concept keys for the candidate set. rows > IDs points at (a); IDs > concepts points at (b); duplicates present within a single source's output points at (c).
**Correction:** dedup by ID first, then apply a concept-level cap so one concept cannot occupy more than a declared number of candidate slots.
**Prevention:** a candidate-set integrity assertion in the request path's test harness that fails when rows ≠ distinct IDs.

### F-11 — Latency regresses during index refresh

**Symptom:** p99 latency doubles for about 10 minutes every rebuild, then recovers.
**Competing causes:** (a) the rebuild competes for the same CPU or memory bandwidth as serving; (b) serving swaps to a cold index whose pages are not resident; (c) the swap causes a burst of cache misses in a downstream eligibility cache; (d) the new index is physically larger and traversal cost rose permanently, with the "recovery" being traffic variation rather than a fix.
**Discriminating check:** compare p99 in the window after swap with p99 an hour later at equal traffic. No recovery means (d). Rising resident-set size after swap with page-fault counters elevated means (b). Rebuild CPU pinned to the serving hosts means (a).
**Correction:** rebuild off-cluster, warm the new index with a replay of recent queries before the swap, and keep the old index resident until the new one is warm.
**Prevention:** measure p99 during a synthetic swap in staging, and make the swap a distinct deployable step with its own acceptance criteria.

### F-12 — A popularity fallback becomes unsafe

**Symptom:** the fallback source, meant for cold start, serves 40% of traffic after a primary outage and starts surfacing policy-sensitive items.
**Competing causes:** (a) the fallback was built before a policy was introduced and never re-filtered; (b) the fallback's precomputed list was snapshotted at build time and is stale; (c) the fallback bypasses the final eligibility check because it is considered trusted.
**Discriminating check:** run the fallback's output through the same eligibility oracle used for the primary. Non-empty rejections point at (a) or (b); distinguishing them requires comparing the list's build timestamp with the policy's effective timestamp. A fallback whose output never reaches the final check is visible in the trace as a missing policy call.
**Correction:** route fallback output through the identical filter chain, and rebuild the list from current state rather than embedding eligibility decisions in it.
**Prevention:** treat every source as untrusted for eligibility; the final check is the only place that decides.

### F-13 — Recall calculation divides by zero for a cohort

**Symptom:** a cohort reports recall of 0.00 or 1.00 with perfect confidence; the pipeline looks either broken or perfect.
**Competing causes:** (a) the cohort's labels are immature so R is empty; (b) the cohort's label rule never fires because its surface does not emit the outcome event; (c) the cohort is genuinely outside the rights envelope so R ∩ E is empty.
**Discriminating check:** report the count of requests with |R ∩ E| = 0 and the reason. If the count is high and the horizon is recent, cause is (a). If the outcome event count is zero for the surface, cause is (b).
**Correction:** exclude undefined requests from the ratio, report the exclusion count and reason distribution, and never substitute a numeric placeholder.
**Prevention:** make the metric function return a discriminated union — `defined(value, n)` or `undefined(reason)` — so a silent zero is impossible in the type system.

### F-14 — Item update never reaches the index

**Symptom:** an edited title or price is correct on the detail page and wrong in recommendations for days.
**Competing causes:** (a) the index is rebuilt on a schedule and the update fell between rebuilds; (b) the update event is consumed by the feature store but not by the index pipeline; (c) the index stores a copy of mutable fields and serves them as truth.
**Discriminating check:** compare the update event's consumption offsets across the two pipelines. A gap in the index pipeline's offset points at (b). An index that stores mutable fields is visible in its schema, which excludes (c) as a separate cause only if the schema is immutable.
**Correction:** index only identifiers and embeddings; hydrate mutable attributes at request time from the authoritative store.
**Prevention:** design the index as a pure retrieval structure with immutable keys, and enforce it with a schema check in the build.

### F-15 — Position confounds a source comparison

**Symptom:** source X appears better than source Y in a slate where X is placed first.
**Competing causes:** (a) genuine quality difference; (b) position effect, since the first slot is examined far more; (c) allocation policy that places the higher-scoring source first by construction; (d) different candidate budgets per source so X simply had more chances.
**Discriminating check:** compare per-source outcomes at matched slots. If the advantage disappears at slot 3 and above, cause is (b) or (c). If X also had a larger candidate count, cause is (d).
**Correction:** randomise slot assignment for the comparison, or correct by measured examination propensities, and equalise the candidate budget per source.
**Prevention:** log rendered position and arm for every exposure; without position in the log, no such correction is possible after the fact.

### F-16 — Vector rebuild changes rankings unexpectedly

**Symptom:** after an embedding retrain with unchanged evaluation metrics, the top-10 slate overlap with the previous model is 40%.
**Competing causes:** (a) the new model is genuinely different and the offline metric is insensitive; (b) vector normalisation changed so cosine and inner product no longer agree; (c) the item encoder and the query encoder were updated in different deployments, producing mismatched spaces.
**Discriminating check:** compute [A1] fidelity between the new index and its own exact oracle. High fidelity with low cross-version overlap isolates the change to the embedding, not the index. Then compare a cosine-ranked list with an inner-product-ranked list on the same vectors: divergence points at (b). Check encoder version metadata on both sides for (c).
**Correction:** deploy encoder pairs atomically with a shared version identifier and refuse to serve mismatched versions.
**Prevention:** evaluate not only recall but also slate stability — the overlap of top-k between versions on a frozen query set — so a large behavioural change is visible before rollout.

### F-17 — Exposure logging is incomplete

**Symptom:** a surface shows 20 items but the exposure table records 14 on average.
**Competing causes:** (a) client-side logging fires only for items that enter the viewport, while the server renders all 20; (b) the logger samples at a rate that was not recorded; (c) the client drops events on backgrounding; (d) the server renders a fallback the logger does not know about.
**Discriminating check:** join server render decisions with client exposure events by request ID and count the deficit by slot index. A deficit concentrated in low slots points at (a). A uniform 30% deficit points at (b). A deficit correlated with session-end points at (c).
**Correction:** log the server render decision as the authoritative exposure record and use client events only for viewability.
**Prevention:** an exposure reconciliation job whose output is a required field in every evaluation report.

### F-18 — Long-tail items disappear from candidates

**Symptom:** the bottom 80% of the eligible catalog by historical engagement accounts for 4% of exposures.
**Competing causes:** (a) popularity and co-visitation sources structurally cannot propose tail items; (b) the ANN graph is disconnected for tail items because they had too few neighbours during construction; (c) the ranker demotes them because their features are sparse; (d) exploration is disabled.
**Discriminating check:** measure presence of tail items in the raw source output before ranking. Present in candidates but absent from slates points at (c). Absent from candidates while present in the exact oracle points at (b). Absent from every source's output points at (a).
**Correction:** add an explicit tail source with a quota, ensure the index build includes items ingested after the last full rebuild, and reserve exploration slots.
**Prevention:** track coverage of the eligible catalog as a first-class metric, since no recall metric computed against observed labels will reveal items that never had a chance to be observed.

### F-19 — A deleted user's data persists in a derived store

**Symptom:** a deletion request is honoured in the primary store and the user still receives personalised recommendations.
**Competing causes:** (a) the feature store's deletion is asynchronous and lagging; (b) the interaction matrix used by co-visitation is rebuilt nightly and contains the user's rows until the next build; (c) the recommendations are non-personalised but appear personalised because they are popular.
**Discriminating check:** trace the deletion identifier through each derived store by lookup. Presence in the nightly matrix points at (b). Presence in the feature store with a lagging deletion timestamp points at (a). Absence everywhere plus identical output for a control account points at (c).
**Correction:** make deletion a pipeline with per-store verification and a bound on propagation time, not a single primary-store operation.
**Prevention:** maintain a derived-store inventory with an owner per store, and a deletion rehearsal that runs on a schedule and reports per-store latency.

### F-20 — Candidate budget grows without a decision

**Symptom:** the candidate budget silently rises from 200 to 1,400 over a year through successive small increases, and p99 latency rises with it.
**Competing causes:** (a) each change was locally justified by a recall measurement at constant k; (b) the ranker's cost per candidate fell, making more candidates cheap; (c) k was raised for a different surface and inherited by all surfaces.
**Discriminating check:** plot k over time against the deploy log. Attribute each step to a change record. If steps coincide with ranker efficiency work, cause is (b) and the increase may be justified; if steps coincide with retrieval changes, cause is (a).
**Correction:** declare k as a policy constant per surface with a written justification, and require a latency measurement with every change.
**Prevention:** put k, depth, and the latency budget in one versioned configuration reviewed together, so a change to one cannot land without the others in view.

### F-21 — An apparently negative item was never displayed

**Symptom:** a model learns to suppress a class of items, and inspection shows those items have low label rates.
**Competing causes:** (a) genuine low preference; (b) the class was never exposed above slot 12 where examination is near zero, so its label rate reflects position, not preference; (c) the class is legitimately constrained to a surface whose outcome event is under-logged.
**Discriminating check:** compute the label rate conditioned on slot for the affected class. If the rate is comparable to other classes at the same slot, cause is (b) rather than (a).
**Correction:** train with exposure and position as features, or restrict training to exposures in examined slots, and never treat an unexposed item as a negative.
**Prevention:** include "was exposed in an examined slot" as a filter in the label pipeline, and report label-rate-by-slot as a standard diagnostic.

### F-22 — Underfill is masked by a fallback

**Symptom:** every request returns 20 items and no alert fires, yet the personalisation appears weak.
**Competing causes:** (a) the fallback is filling most slots because the primary retrieval is under-delivering; (b) the fallback is genuinely appropriate for the cohort mix; (c) both are true for different cohorts.
**Discriminating check:** report candidate-set composition by originating source per request. A fallback share above roughly half means the primary path is not delivering, which is cause (a) whether or not the resulting slate looks acceptable.
**Correction:** alert on primary-path contribution, not on slate fill; a filled slate is not evidence that retrieval worked.
**Prevention:** make "slots filled by source" a required breakdown in the on-call dashboard, so a healthy-looking fill rate cannot hide an outage.

### 8.2 Measurement catalog

Each row is a proposed measurement. None of the thresholds are universal; they are starting points to be calibrated against a measured baseline. Rates require nonzero denominators and a declared window.

| Metric | Numerator | Denominator | What it detects | Benign explanation |
|---|---|---|---|---|
| Eligible catalog size | distinct IDs passing all predicates | request | segment starvation and policy over-reach | a genuinely narrow catalog in a small territory |
| Candidate recall@k | relevant eligible IDs in C_k | relevant eligible IDs | retrieval ceiling | an immature label horizon inflating the denominator |
| Pooled recall | hits summed over requests | relevant pairs summed | inventory-level recovery | a few heavy-interest requests dominating |
| Macro recall | per-request recall summed | requests with non-empty R | typical-request quality | tiny denominators making each request near 0 or 1 |
| Underfill rate | requests with fewer than k candidates | requests | depth provisioning and filter placement | catalog smaller than k |
| Primary-source share | candidates from the primary path | all candidates | hidden outages masked by fallback | a cohort that legitimately has no history |
| Rights rejection rate | candidates rejected for rights | candidates checked | territory misconfiguration | expected geography |
| Availability rejection rate | candidates rejected for stock | candidates checked | inventory lag versus index lag | genuine stock volatility |
| Authorization error rate | checks returning an error | checks attempted | fail-closed behaviour under dependency failure | a known dependency blip with no exposure |
| ANN fidelity@k | exact top-k IDs recovered | k, or exact list length | index configuration drift | tie-heavy score distributions |
| Depth-to-keep ratio | requested depth | delivered candidates | depth mis-provisioning | a dense cohort where depth is generous |
| Slate stability | top-k overlap with previous version | k | behaviour change from a model swap | an intended exploration increase |
| Exposure concentration | Gini over per-item exposure | — | feedback-loop narrowing | a head-heavy catalog with genuine demand |
| Coverage of eligible catalog | distinct items exposed | distinct eligible items | tail starvation | short measurement window |
| Label rate by slot | labelled outcomes at slot s | exposures at slot s | position confounding | a genuinely better item at slot 1 |
| Exposure reconciliation | client exposure events | server render decisions | logging loss | viewport-based logging by design |
| Primary-path latency p99 | request duration at p99 | — | depth or index regression | a rebuild window |
| Fallback activation rate | requests served with any fallback | requests | dependency health | a new-cohort influx |

### 8.3 Point-in-time and serving acceptance traces

These are constructed test cases for a deployment gate, not records of executed tests. Each names the behaviour that must hold and the wrong behaviour the trace is designed to catch. A trace that cannot fail is not a test.

| Trace | Setup | Required behaviour | Wrong behaviour it catches |
|---|---|---|---|
| Rights revoked mid-request | revoke between retrieval and render | item omitted; decision logged with the new policy version | cached allow replayed |
| Territory change between sessions | viewer attribute changes at t | next request uses new territory; no stale slate | territory cached at profile level |
| Entitlement downgrade | subscription ends before request | premium items excluded in the same request | slate cached across tier change |
| Policy service timeout | inject a controlled timeout | deny; no unchecked item rendered | timeout treated as allow |
| Inventory reaches zero | stock hits zero after retrieval | item omitted or shown as unavailable per policy | stale positive inventory served |
| Eligibility predicate empty | all items fail by construction | honest empty or editorial response | ineligible substitute injected |
| Global ANN underfill | depth 20 with density 0.02 | underfill reported, fallback invoked | silent 3-item candidate set passed to ranker |
| Segment density halved | territory density 0.01 | depth recomputed; underfill rate stays bounded | fixed depth retained, underfill doubles |
| Tie-heavy score boundary | 40 items share the k-th score | deterministic tie rule reproduced exactly | order-dependent output across replicas |
| Embedding version skew | query and item encoder versions differ | request refused; version mismatch logged | silent quality loss |
| Duplicate across sources | same ID from three sources | one candidate row | three rows consuming budget |
| Concept cap | six variants of one product | at most the declared per-concept cap | one concept filling the slate |
| Deleted account | deletion propagated | no personalised candidates | derived store still serving |
| Index swap | new index promoted | p99 within budget; fidelity report present | cold index served unmeasured |
| Exposure logging loss | client drops 30% of events | reconciliation flags the deficit | analysis run on partial exposures |
| Label horizon not matured | request inside the horizon | excluded from recall with a reason code | request counted as zero recall |

**Takeaway.** Diagnosis is a comparison, not an inspection: every entry above ends in a check that produces a number separating at least two hypotheses. A failure mode without such a check is a suspicion, and a runbook built from suspicions sends responders to the wrong service.
