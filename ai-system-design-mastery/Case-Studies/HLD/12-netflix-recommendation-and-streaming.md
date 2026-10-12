# Netflix — Streaming, Per-Title Encoding and Recommendation

> **Source note.** Netflix is named explicitly in the roadmap video (T1, phase 4: *"ML practice design and Netflix system plus fraud detection system"*). The transcripts name it without designing it, so the design below is **supplementary** — built from standard industry practice and published Netflix engineering material, not quoted from the transcripts. Numbers are labelled *stated assumption* (chosen for the exercise) or *derived* (arithmetic from the assumptions). Nothing here is presented as measured telemetry.

> **TL;DR.** Netflix is two systems wearing one name: a **content delivery** system whose whole job is moving pre-encoded bytes to a device as cheaply and reliably as possible, and a **recommendation** system whose whole job is deciding which of ~10,000 titles a person should see right now. Unlike YouTube, the catalog is **finite and licensed**, so Netflix encodes each title once, optimally, per-title and per-device — instead of transcoding an open-ended stream of user uploads. The distinctive moves are **Open Connect** (Netflix's own CDN appliances, placed *inside* ISPs, so most traffic never crosses the public internet), **per-title encoding** (a complexity-aware bitrate ladder computed per title rather than one fixed ladder for all), and an **offline/online recommendation split** (heavy batch precompute + a light online re-rank under a latency budget).

> **60-second interview answer.** "Netflix differs from YouTube in one structural way that drives everything else: the catalog is finite and licensed, so I can afford per-title optimization. I'd estimate the write side — encoding — at roughly 10,000 titles × ~20 renditions, which is a batch pipeline, not a stream. The read side is the interesting one: ~250M subscribers, peak evening concurrency in the tens of millions, each stream at 5–15 Mbps, so aggregate egress is on the order of tens of terabits per second. No origin can serve that, so I push bytes into Open Connect appliances inside ISP networks and let steering send each client to the nearest one. On top of that, recommendations run as an offline batch that precomputes candidate sets, plus an online re-rank that has a ~100 ms budget. The trade-off I'd name is that per-title encoding costs compute and adds pipeline latency but cuts bitrate at equal quality — which is a direct egress-cost saving at Netflix's scale."

---

## 1. Definition & mental model

**Definition.** Netflix is a subscription video streaming service engineered around three distinct subsystems:

| Subsystem | Job | Optimized for |
|---|---|---|
| **Encoding pipeline** | Turn one mezzanine source file into ~20–30 device-specific renditions | Cost per bit at equal quality; offline, so latency-tolerant |
| **Content delivery (Open Connect)** | Move the right rendition to the right device, cheaply | Egress cost, start-up latency, rebuffering |
| **Recommendation & personalisation** | Choose what to show, in what order, with which artwork | Engagement (hours watched), under a hard latency budget |

**Mental model — the library, not the newsstand.** YouTube is a newsstand: anyone can drop off a magazine, and the shop must process whatever arrives. Netflix is a library with a fixed, curated collection — so it can *catalogue each book perfectly once* rather than accept whatever shows up. That single difference explains per-title encoding, the finite encoding batch, and the absence of a viral-upload thundering-herd problem.

**What it is *not*.** It is not a transcoding-on-demand service (renditions exist before you press play); it is not a single monolithic recommendation model (it is a pipeline of ranking stages); and it is not, at the delivery layer, a traditional CDN customer — Netflix built its own CDN precisely because it is large enough to be cheaper than buying one.

---

## 2. Why it exists — the problem it solves

Three physical constraints force the architecture:

1. **Distance costs latency and money.** Bits crossing the public internet traverse many ASes, each of which charges. Serving 15% of peak downstream internet traffic from a handful of origins would be economically impossible and physically congested.
2. **Devices are heterogeneous and networks are unstable.** A phone on 3G, a 4K TV on fibre, and a laptop on hotel Wi-Fi need different bitrates. One rendition cannot serve all three without either wasting bandwidth or rebuffering.
3. **The catalog is large enough to need a ranking system but small enough that browsing it is futile.** ~10,000 titles is far too many to browse and far too few to need a web-scale crawl. Personalisation is the only workable interface.

Without the CDN, start-up latency and egress cost destroy the product. Without ABR, mobile users rebuffer. Without recommendations, the catalog becomes unusable and engagement collapses — which for a subscription business is the only metric that matters.

---

## 3. Internal mechanics

### 3.1 The whole system in one diagram

```
 ┌───────────────────────── ORIGIN / CONTROL PLANE ─────────────────────────┐
 │  Content ingest → Mezzanine store (S3-like)                              │
 │        ↓                                                                 │
 │  ENCODING PIPELINE (batch, per title)                                    │
 │   shot/complexity analysis → per-title bitrate ladder → chunked encode   │
 │   → per-title, per-device renditions + manifests (DASH/HLS)              │
 │        ↓                                                                 │
 │  Packaging + DRM (Widevine/PlayReady/FairPlay)                           │
 │        ↓                                                                 │
 │  Origin  ────────────────────────────────────────────────┐               │
 └──────────────────────────────────────────────────────────┼───────────────┘
                                                            │  fill (off-peak,
                                                            │  predictive)
                    ┌───────────────────────────────────────▼─────────────┐
                    │        OPEN CONNECT  (Netflix CDN appliances)       │
                    │   placed INSIDE ISP networks / IXPs                 │
                    │   • stores the popular ~90% of catalog              │
                    │   • fill from origin, not on client miss            │
                    └───────────────────────────────────────┬─────────────┘
                                                            │
   Client (TV / phone / browser)                            │
     │ 1. login, browse  ──────────►  API / Playback service  │
     │ 2. get manifest + steering URL ◄──────────────────────┘
     │ 3. ABR streaming: pick rendition per segment from buffer health
     └────────────────────────────────────────────────────────────────────

 ┌───────────────── RECOMMENDATION / PERSONALISATION ────────────────────────┐
 │  OFFLINE (batch, hours):                                                   │
 │   viewing history + catalogue embeddings → candidate generation           │
 │   → precomputed per-member ranked rows (a "page" of rows)                  │
 │  ONLINE (per request, ~100 ms budget):                                     │
 │   re-rank rows with fresh context (device, time, session)                  │
 │   + artwork selection (bandit) + row ordering                              │
 │  A/B TESTING PLATFORM: every change is an experiment                       │
 └───────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Open Connect — the defining move

Netflix places **appliances it owns inside ISP data centres and internet exchange points**. Two consequences follow:

- **Traffic is short-haul.** The client is usually a few network hops from the bytes, so start-up latency is low and ISP transit costs fall — which is why ISPs accept the appliance: it *reduces* their cost.
- **Fill is predictive, not reactive.** Because the catalog is finite and viewing is strongly predictable (a new season of a hit show is going to be watched), the appliance is filled **ahead of demand**, off-peak. A client request is a *cache hit by design*, not a hopeful lookup.

This inverts normal CDN economics. A conventional CDN's value is "we have many points of presence"; Open Connect's value is "the content is already inside your ISP before you asked".

### 3.3 Per-title encoding — the second defining move

A fixed bitrate ladder (e.g. 235/375/560/750/1050/1750/2350/3000/4300/5800 kbps for all titles) is wasteful: an animated film with flat colour needs far fewer bits than a grainy action sequence with fast motion to reach the same perceived quality.

The pipeline therefore, **per title**:

1. Analyses complexity — shot boundaries, spatial detail (texture), temporal detail (motion).
2. Searches for the **convex hull** of rate–quality points: for each target quality, the cheapest encoding that achieves it.
3. Emits a **title-specific ladder** — possibly fewer rungs, possibly different bitrates.
4. Encodes in **chunks** (a few seconds each), so chunks can be encoded in parallel and re-encoded independently if the ladder changes.

The payoff is a direct cash saving: at equal perceived quality, lower average bitrate means lower egress and lower CDN storage. At Netflix's scale this is one of the largest line-item savings available.

### 3.4 Playback and ABR

- The client receives a **manifest** listing available renditions and segment URLs.
- **ABR (adaptive bitrate)** picks the next segment's rendition from measured throughput and **buffer occupancy** — the buffer is the real signal, because it absorbs short-term variance.
- Start-up is a separate optimisation from steady-state: pick a conservative first segment to start fast, then ramp. Starting too high causes an immediate rebuffer, which users rate as worse than starting soft.
- **DRM** is enforced per device family (Widevine, PlayReady, FairPlay), so the device's capability set partly determines the manifest.

### 3.5 The recommendation pipeline

Netflix's recommender is best understood as **a sequence of stages, not one model**:

| Stage | Runs | Job |
|---|---|---|
| **Candidate generation** | Offline batch (hours) | Reduce ~10,000 titles to a few hundred plausible ones per member |
| **Ranking** | Offline batch | Score candidates; build a *page* of rows, pre-ranked per member |
| **Re-ranking / row ordering** | Online (~100 ms) | Apply fresh context: device, time of day, session behaviour, "continue watching" |
| **Artwork personalisation** | Online | Choose *which image* represents a title — often a bigger engagement lever than ordering |

Two design points worth stating in an interview:

- **The "two-row problem".** Ranking items *within* a row is not the same objective as ordering the rows themselves, and a model that is excellent at the first can be poor at the second. They are separate models.
- **Artwork is a bandit problem, not a ranking problem.** Which thumbnail wins is unknown until shown, so it is exploration/exploitation — a contextual bandit — not supervised ranking.
- **Everything is an A/B test.** The A/B platform is a first-class subsystem: assignment, exposure logging, metric computation, and sequential-testing guardrails so that peeking at results does not inflate false positives.

---

## 4. Variants & taxonomy

| Variant / choice | What it means | When it applies |
|---|---|---|
| **Own CDN (Open Connect)** | Netflix-operated appliances in ISPs | Only economical above a very large traffic floor; below that, buy a CDN |
| **Commercial CDN** | Third-party PoPs | Small services, or overflow/burst capacity |
| **Per-title encoding** | Ladder derived per title | Finite, high-value catalog. Not worth it for long-tail UGC |
| **Fixed ladder** | One ladder for all | UGC platforms; cheap, simple, predictable |
| **Chunked encoding** | Encode in parallel chunks | Needed for encode throughput; costs slight quality at boundaries |
| **Offline-first recommendation** | Precompute rows in batch | Stable catalogs; latency-tolerant |
| **Online-only recommendation** | Score at request time | Fast-changing inventory (e.g. ads); expensive per request |
| **Hybrid** | Offline candidates + online re-rank | What Netflix does; the standard answer |
| **Downloads** | Pre-positioned encrypted files on device | Mobile/offline; must handle licence expiry and storage limits |

---

## 5. Pros / Cons

### Open Connect (own CDN)

| Pros | Cons |
|---|---|
| Lowest marginal egress cost at scale | Enormous capital and operational cost |
| Short network path → low start-up latency | Requires ISP relationships and physical presence |
| Reduces ISP transit cost → ISPs cooperate | Slow to reach new regions; poor fit for small traffic |
| Predictable performance, insulated from public-internet congestion | Filling the wrong content wastes appliance storage |

### Per-title encoding

| Pros | Cons |
|---|---|
| Lower bitrate at equal perceived quality → direct cost saving | Much higher encode compute and pipeline complexity |
| Better quality on hard titles (grain, motion) | Longer lead time from ingest to availability |
| Fewer rungs can mean simpler client logic | Requires robust complexity analysis; errors are visible as quality regressions |

### Offline + online recommendation split

| Pros | Cons |
|---|---|
| Online path stays inside a tight latency budget | Precomputed rows are stale until the next batch |
| Batch can use expensive models over full history | Cold-start members have no history to batch over |
| Cheap to serve | Two systems to keep consistent; a batch failure degrades everyone at once |

---

## 6. Exceptions, edge cases & failure modes

| # | Failure | Mechanism | Mitigation |
|---|---|---|---|
| 6.1 | **New-season demand spike** | A hit release drives a step change in demand for one title across many ISPs | Predictive pre-positioning *before* release; the catalog is known, so this is forecastable, not reactive |
| 6.2 | **Appliance fill misses** | A title predicted unpopular becomes popular; the local appliance lacks it | Fall back to a peer appliance/origin; treat as a cache miss with a degraded start-up, not a failure |
| 6.3 | **ISP peering dispute** | An ISP de-peers or a region's appliance is unreachable | Multi-homed steering; reroute to a neighbouring appliance at higher latency |
| 6.4 | **ABR oscillation** | Throughput estimate is noisy; client flips between renditions, quality visibly pulses | Buffer-based ABR with hysteresis; damping so a single slow segment does not trigger a downshift |
| 6.5 | **Start-up vs rebuffer trade-off** | Starting aggressively raises start-up quality but causes an early rebuffer | Weight rebuffering far more heavily than start-up bitrate in the objective |
| 6.6 | **Cold-start member** | No history → batch has nothing to rank | Popularity + content-similarity priors; rapid online learning from the first session; onboarding "pick 3 shows" |
| 6.7 | **Feedback loop / popularity bias** | Recommender shows what it already ranks highly → that content accrues more data → it ranks higher still | Explicit exploration (bandits); holdout traffic without personalisation to measure true uplift |
| 6.8 | **Artwork feedback loop** | One image wins, so it is shown more, so it keeps winning | Forced exploration budget; periodic re-exploration of losing arms |
| 6.9 | **A/B test peeking** | Checking results continuously inflates false-positive rate | Sequential testing / always-valid inference; fixed-horizon tests for final calls |
| 6.10 | **DRM/licence edge cases** | Device unsupported, licence expired, region restriction mid-stream | Capability negotiation before manifest; clear error surfaces; download licence TTL |
| 6.11 | **Encode pipeline backlog** | A large ingest batch saturates encoders; titles are unavailable | Priority lanes by expected demand; the finite catalog makes the backlog bounded and schedulable |
| 6.12 | **Chunk boundary artefacts** | Independent chunk encoding creates visible seams at boundaries | Overlap/lookahead windows; closed-GOP alignment at chunk edges |
| 6.13 | **Metric gaming** | Optimising clicks or autoplay-starts can raise the metric without raising satisfaction | Optimise long-horizon engagement and retention, not click-through |
| 6.14 | **Regional catalogue divergence** | Licensing differs by country → different candidate sets per region | Region is an input to candidate generation, not a post-filter |

---

## 7. When to use / When NOT to use

**Take the Netflix approach (own CDN, per-title, offline recommendations) when:**

- Traffic is large enough that CDN egress is a top-three cost line.
- The catalog is finite, licensed, and stable enough to pre-process.
- Viewing is predictable enough to pre-position content.
- Perceived quality at lower bitrate is worth real money to you.

**Do NOT take it when:**

- You are a UGC platform — the long tail does not justify per-title encoding, and uploads are unpredictable, so YouTube-style fixed ladders and transcode-on-upload fit better.
- Your traffic is small — building a CDN below a very high floor is strictly more expensive than buying one.
- Your inventory changes per request (ads, live events) — offline precompute is the wrong shape; online ranking dominates.
- Your latency budget is sub-50 ms end-to-end — a multi-stage recommendation pipeline will not fit.

**The one-line decision rule:** *finite catalog + very large egress + engagement-driven product ⇒ Netflix architecture. Open catalog + unpredictable supply ⇒ YouTube architecture.*

---

## 8. Numbers, defaults & tuning

> Assumptions are marked **[A]** (stated for the exercise); everything else is **[D]** — derived arithmetic. Treat these as defensible in an interview, not as cited facts.

| Quantity | Value | Basis |
|---|---|---|
| Subscribers | ~250M paid | [A] order of magnitude |
| Catalog size | ~10,000 titles | [A] |
| Renditions per title | ~20–30 (ladder × device families) | [A] |
| Encoding jobs for full catalogue | 10,000 × 25 ≈ **250,000 renditions** | [D] |
| Peak concurrent streams | tens of millions | [A] |
| Average stream bitrate | ~5 Mbps (mixed SD→4K) | [A] |
| Peak aggregate egress | 20M streams × 5 Mbps = **100 Tbps** | [D] |
| Share of peak downstream internet | ~15% in some regions at peak | [A] published-order figure |
| Segment length | 2–6 s | [A] typical |
| Client buffer target | 30–60 s of video | [A] typical |
| Recommendation online budget | ~100 ms | [A] |
| Batch recommendation cadence | hours | [A] |
| Per-title saving vs fixed ladder | ~20–30% bitrate at equal quality | [A] published-order figure — the whole economic case rests on this |
| Storage for full catalogue | 10,000 titles × ~2 h × ~5 Mbps avg across renditions ≈ **tens of PB** | [D] |

**The interview-relevant arithmetic.** The number that matters is peak aggregate egress. Compute it as `concurrent streams × average bitrate`. At 20M × 5 Mbps that is 100 Tbps — and that single number is the entire justification for Open Connect. If you can produce that figure and then say "no origin fleet serves 100 Tbps economically, so bytes must be pre-positioned inside ISPs", you have justified the architecture numerically, which is the point of the exercise.

---

## 9. Interview Q&A

**Q1. Why does Netflix encode per title instead of using one ladder for everything?**
Because a fixed ladder is a compromise that over-serves easy content and under-serves hard content. An animated film with flat colour reaches perceptual transparency at a much lower bitrate than a grainy, fast-motion action sequence. Per-title encoding analyses spatial and temporal complexity, searches the rate–quality convex hull, and emits a ladder specific to that title. The payoff is roughly 20–30% lower bitrate at equal perceived quality, which at 100 Tbps of peak egress is an enormous cost saving. The cost is significantly more encode compute and a longer ingest-to-available lead time.

**Q2. Why does Netflix run its own CDN?**
Because above a certain traffic floor it is cheaper than buying one, and it buys performance as well as cost. Owning the appliances lets Netflix place them inside ISP networks, so most traffic never crosses the public internet — lower start-up latency, less exposure to public-internet congestion, and lower ISP transit cost, which is why ISPs accept the appliances. It also makes fill *predictive*: the catalog is finite and demand is forecastable, so content is pushed before it is requested. Below that floor the capital and operational cost dominate and a commercial CDN wins.

**Q3. How is this different from designing YouTube?**
The catalog. YouTube accepts arbitrary user uploads, so it must transcode whatever arrives — an open-ended, unpredictable write stream, and it cannot afford per-title optimality for the long tail. Netflix has a finite, licensed catalog, so it encodes each title once, optimally, and pre-positions it. YouTube's hard problems are the upload pipeline, the transcode queue, and viral traffic spikes on arbitrary content; Netflix's are per-title quality optimisation and predictive fill. Both end up using a CDN, but Netflix built its own. The single sentence: *open catalogue ⇒ YouTube; finite catalogue ⇒ Netflix.*

**Q4. Walk me through what happens when I press play.**
Client asks the playback service for a manifest, sending device capability and DRM information. The service returns a manifest listing available renditions plus a steering decision pointing at a specific Open Connect appliance near the client. The client fetches the first few segments — chosen conservatively so playback starts fast — then ABR ramps up as it measures throughput and builds buffer. Steady state is buffer-driven: the client picks the rendition that keeps the buffer healthy, not the one the instantaneous throughput could theoretically sustain.

**Q5. How would you size the encoding pipeline?**
It is a bounded batch, which is the nice part. 10,000 titles × ~25 renditions is ~250,000 rendition jobs, plus re-encodes when the ladder changes or a codec improves. That is a throughput problem — provision enough encoder capacity to finish the catalogue (and any new title within its release window) in the available off-peak time — rather than a concurrency problem. Priority lanes by expected demand matter, because a title that will be watched on release day cannot sit behind a decade-old back-catalogue item.

**Q6. How do you handle a new season release?**
Because the catalog is known in advance, this is a forecasting problem rather than a reactive-scaling problem. The title is encoded and pre-positioned into Open Connect appliances ahead of release, concentrated where demand is predicted. The failure mode to name is mis-forecasting: if a title is predicted unpopular and becomes popular, the local appliance lacks it and those clients fall back to a peer or origin at higher latency — a degraded start-up, not an outage. Mitigation is a fallback path that is always available, plus conservative over-fill for release titles.

**Q7. Why split recommendation into offline and online stages?**
Because the two jobs have incompatible cost profiles. Candidate generation over full viewing history with an expensive model is affordable in a batch job that runs for hours, and it must be, because it needs the whole corpus. But the response to a page load must fit in roughly 100 ms. So you precompute candidates and a pre-ranked page offline, then re-rank online using only cheap, fresh signals — device, time of day, session context, continue-watching. The trade-off is staleness: until the next batch, the precomputed rows reflect old data, which is why the online re-rank exists.

**Q8. What is the "two-row problem"?**
Ranking items within a row and ranking the rows themselves are different objectives, and a model good at one can be poor at the other. Within a row you are comparing items that are broadly similar — which of these comedies should be second? Across rows you are deciding what kind of thing the member should be offered at all — comedy versus thriller versus continue-watching. They need separate models and separate evaluation, and conflating them is a classic design error.

**Q9. Why is artwork a bandit problem rather than a ranking problem?**
Because you do not know which image will win until you show it. There is no labelled dataset of "best artwork"; you learn by exposing members to options. That is exploration versus exploitation under uncertainty, so it is solved with a contextual bandit — showing the currently-best arm most of the time while reserving traffic to keep learning. In practice artwork choice often moves engagement more than ordering does, which makes it worth the machinery.

**Q10. How do you avoid a feedback loop where recommendations entrench themselves?**
Three levers. First, an explicit exploration budget so a fraction of traffic sees non-top-ranked content and you keep collecting unbiased signal. Second, a holdout cohort that receives non-personalised recommendations, so you can measure true uplift rather than the recommender's own opinion of itself. Third, evaluate on long-horizon metrics — retention and hours watched over weeks — rather than click-through, because short-horizon metrics reward the loop.

**Q11. How do you evaluate a recommendation change?**
Almost always by A/B test, with sequential-testing statistics so that continuously monitoring the result does not inflate the false-positive rate. Metrics are tiered: a primary long-horizon metric (retention or hours), guardrail metrics that must not regress (start-up time, playback failures, latency), and diagnostic metrics for explaining movement. The two common failures are peeking at a fixed-horizon test and shipping on a short-horizon metric that later harms retention.

**Q12. Where does caching fit in this design?**
Three different caches with three different roles. Open Connect is the dominant one — a content cache with predictive fill, sized to hold the popular majority of the catalog. Inside the control plane there is ordinary metadata caching for the API tier (catalog metadata, manifests, artwork). And on the client there is the playback buffer, which is a cache under a latency constraint rather than a throughput one and is what ABR actually steers on. Conflating these three is a common answer error.

**Q13. What breaks first under a sudden traffic surge?**
The steering and playback control plane, not the data plane. Bytes are pre-positioned and an appliance serving more clients is exactly what it is built for; but every client starts by calling the playback service for a manifest and a steering decision, so a synchronised surge (a live event, a major release) hammers a comparatively small control-plane tier. Mitigation is caching manifests and steering decisions at the edge, plus admission control, rather than trying to scale the control plane instantly.

**Q14. ⚡ Pushback: "Why not just use a commercial CDN and avoid all that complexity?"**
Below a certain traffic floor, that is exactly right and I would do it. The question is whether you are above the floor. The relevant test is whether CDN egress is one of your largest cost lines and whether your traffic is predictable enough to pre-position. Netflix is on the far side of both, and has the volume to amortise building and operating its own fleet — plus, owning it removes the margin a CDN would charge and gives control over caching policy that a vendor will not surrender. If I were designing for a smaller service, I would say "use a commercial CDN, and revisit owning only if egress becomes a top-three cost."

**Q15. ⚡ Pushback: "Per-title encoding sounds like premature optimisation. Just use a good fixed ladder."**
For a small or UGC catalog I would agree — the machinery is not worth it. The reason it is not premature at Netflix's scale is that the saving is proportional to *egress*, so a 20–30% bitrate reduction applies to tens of terabits per second, forever, and is one of the largest available cost levers. The test is whether the saving × your traffic exceeds the cost of the encode pipeline and its complexity. At small scale it does not; at Netflix scale nothing else comes close.

**Q16. ⚡ Pushback: "Precomputing recommendations in a batch job means everyone sees stale content. Isn't that a bug?"**
It is a deliberate trade, not a bug. The batch computes the expensive part — candidates and coarse ranking over full history — where staleness of a few hours is acceptable because taste does not change hourly. The online stage then applies everything that *does* change quickly: the current session, the device, the time of day, and continue-watching state. So the member-visible output is fresh even though the heavy computation is not. What I would actually call a bug is a stale batch: if the offline job fails, everyone gets old rows simultaneously, which is why the batch needs its own SLO and a fallback to popularity-based rows.

---

## 10. Common mistakes candidates make

1. **Designing Netflix as YouTube.** Treating the catalog as open and user-generated, then proposing transcode-on-upload and a viral-spike pipeline. The finite licensed catalog is the premise of the whole design.
2. **Forgetting the encoding pipeline entirely.** Jumping straight to CDN and recommendations, when the encode stage is where the cost/quality trade is actually made.
3. **Saying "use a CDN" without doing the arithmetic.** The interesting answer is *why Netflix built its own* — which requires producing the peak-egress number first.
4. **Proposing a single recommendation model.** Missing the candidate-generation / ranking / re-ranking / artwork stages, and missing that rows and items are separate problems.
5. **Ignoring A/B testing as a subsystem.** At Netflix, the experimentation platform is a first-class component, not an afterthought.
6. **No cold-start story.** New members have no history, so the batch has nothing to rank. Name popularity priors and onboarding explicitly.
7. **Treating the CDN as a cache only.** Open Connect's defining property is *predictive fill*, not reactive caching. That distinction is the senior signal.
8. **Missing DRM and device capability.** Manifest generation depends on device family and licence state; ignoring it makes the playback path incomplete.
9. **Optimising the wrong metric.** Click-through or autoplay-starts are gameable; retention and hours are not.
10. **No feedback-loop awareness.** Recommenders entrench themselves without exploration, holdouts, and long-horizon metrics.

---

## 11. Cheat-sheet summary

| | |
|---|---|
| **Premise** | Finite, licensed catalog (~10k titles) — *not* UGC |
| **Three subsystems** | Encoding pipeline · Open Connect delivery · Recommendation |
| **Defining move 1** | **Open Connect** — own CDN appliances inside ISPs; predictive fill, not reactive caching |
| **Defining move 2** | **Per-title encoding** — complexity analysis → rate–quality convex hull → title-specific ladder; ~20–30% bitrate saving |
| **Defining move 3** | **Offline + online recommendation** — batch candidates/rank (hours) + online re-rank (~100 ms) |
| **The justifying number** | `concurrent streams × bitrate` = 20M × 5 Mbps ≈ **100 Tbps peak egress** |
| **Encode sizing** | 10k titles × ~25 renditions ≈ **250k jobs** — a bounded batch, not a stream |
| **ABR signal** | Buffer occupancy, not instantaneous throughput; start conservatively, ramp up |
| **Rows vs items** | Two different ranking problems — the "two-row problem" |
| **Artwork** | A contextual bandit, not a ranker; often the bigger engagement lever |
| **vs YouTube** | Open catalogue ⇒ YouTube (transcode-on-upload, fixed ladder) · Finite catalogue ⇒ Netflix (per-title, pre-positioned) |
| **Failure modes to name** | Fill miss · ABR oscillation · cold start · feedback loop · A/B peeking · control-plane surge |
| **The close** | "If the catalogue became open and user-generated, I'd drop per-title encoding and move to a YouTube-style transcode-on-upload pipeline." |

---

## 12. Cross-links

- [YouTube — upload and recommendation](05-youtube-upload-and-recommendation.md) — the *other* video design; read the pair together, they differ on exactly one premise
- [CDN deep dive](../../HLD/02-networking/02-cdn-deep-dive.md) — push vs pull, origin shield, and why Open Connect inverts normal CDN economics
- [Blob & object storage](../../HLD/04-data/04-blob-and-object-storage.md) — mezzanine storage, tiering, lifecycle policy
- [Caching strategies and layers](../../HLD/03-scaling/02-caching-strategies-and-layers.md) — the three distinct caches in this design
- [Vector databases](../../HLD/04-data/03-vector-databases.md) — content embeddings for candidate generation
- [Fraud detection system](13-fraud-detection-system.md) — the other ML system named alongside Netflix in the source roadmap
- [Capacity estimation playbook](../../HLD/01-foundations/05-capacity-estimation-playbook.md) — the arithmetic used in §8
- [Production RAG system](../../HLD/05-ai-systems/01-production-rag-system.md) — recommendation as a retrieval-and-rank pipeline
